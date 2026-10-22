// Device code of src/tensorfold/families/qwen4_exp/cuda/gdn.cu (lines 1-213, comments and ATen includes dropped), checked by zig/tests/cuda/copies.py.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace tf_fn_gdn {

constexpr int DK = 128, DV = 128, TAPS = 4;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float warp_sum(float x) {
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float softplusf_(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

__device__ __forceinline__ void update(float (&s)[4][4], const float (&kk)[4], const float* vrow, int warp,
                                       float g, float beta) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[j][i] = s[j][i] * g;
            kv = kv + s[j][i] * kk[i];
        }
        kv = warp_sum(kv);
        const float delta = (vrow[warp * 4 + j] - kv) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
    }
}

template <int NK, int NV, bool AHEAD>
__global__ void __launch_bounds__(1024) chain_kernel(
        const __nv_bfloat16* __restrict__ P, const __nv_bfloat16* __restrict__ cs,
        const __nv_bfloat16* __restrict__ cw, const float* __restrict__ state_in,
        const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        const __nv_bfloat16* __restrict__ norm_w, float eps, int rows,
        __nv_bfloat16* __restrict__ out, float* __restrict__ xs, float* __restrict__ state_out,
        float* __restrict__ k_save, __nv_bfloat16* __restrict__ v_save, float* __restrict__ g_save,
        float* __restrict__ b_save) {
    constexpr int C = 2 * NK * DK + NV * DV;          // conv channels: q | k | v
    constexpr int PW = C + NV * DV + 2 * NV;          // projection row: qkv | z | b | a
    const int hv = blockIdx.x, hk = hv / (NV / NK);
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float qs[DK], ks[DK], vs[DV], ys[DV];
    __shared__ float gates[2], rinv;
    int c = -1;
    if (t < DK) c = hk * DK + t;
    else if (t < 2 * DK) c = NK * DK + hk * DK + (t - DK);
    else if (t < 2 * DK + DV) c = 2 * NK * DK + hv * DV + (t - 2 * DK);
    float s[4][4];
    const size_t sbase = (size_t)hv * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    float w[TAPS] = {}, win[TAPS - 1] = {};
    __nv_bfloat16 xin = {}, zin = {}, bin = {}, ain = {};
    const __nv_bfloat16 *pz = P + C + hv * DV + (t < DV ? t : 0), *pb = P + C + NV * DV + hv, *pa = pb + NV;
    if (AHEAD && rows > 0) {
        if (c >= 0) {
#pragma unroll
            for (int tap = 0; tap < TAPS; ++tap) w[tap] = __bfloat162float(cw[c * TAPS + tap]);
#pragma unroll
            for (int tap = 0; tap < TAPS - 1; ++tap) win[tap] = __bfloat162float(cs[tap * C + c]);
            xin = P[c];
        }
        if (t < DV) zin = pz[0];
        if (warp == 2 && lane == 0) { bin = pb[0]; ain = pa[0]; }
    }
    for (int r = 0; r < rows; ++r) {
        const __nv_bfloat16 xr = xin, zr = zin, br = bin, ar = ain;
        if (AHEAD && r + 1 < rows) {
            const size_t next = (size_t)(r + 1) * PW;
            if (c >= 0) xin = P[next + c];
            if (t < DV) zin = pz[next];
            if (warp == 2 && lane == 0) { bin = pb[next]; ain = pa[next]; }
        }
        if (c >= 0) {
            float acc = 0.0f;
            if constexpr (AHEAD) {
                const float xn = __bfloat162float(xr);
#pragma unroll
                for (int tap = 0; tap < TAPS - 1; ++tap) acc = acc + w[tap] * win[tap];
                acc = acc + w[TAPS - 1] * xn;
                win[0] = win[1]; win[1] = win[2]; win[2] = xn;
            } else {
#pragma unroll
                for (int tap = 0; tap < TAPS; ++tap) {
                    const int at = r + tap;
                    const float x = at < TAPS - 1 ? __bfloat162float(cs[at * C + c])
                                                  : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * PW + c]);
                    acc = acc + __bfloat162float(cw[c * TAPS + tap]) * x;
                }
            }
            const float act = bf(acc / (1.0f + expf(-acc)));
            if (t < DK) qs[t] = act;
            else if (t < 2 * DK) ks[t - DK] = act;
            else vs[t - 2 * DK] = act;
        }
        __syncthreads();
        if (warp < 2) {
            float* x = warp == 0 ? qs : ks;
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { v4[i] = x[lane * 4 + i]; ss = ss + v4[i] * v4[i]; }
            ss = warp_sum(ss);
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            if (warp == 0) inv = inv * (1.0f / sqrtf((float)DK));
            __syncwarp();
#pragma unroll
            for (int i = 0; i < 4; ++i) x[lane * 4 + i] = v4[i] * inv;
        } else if (warp == 2 && lane == 0) {
            const float b = __bfloat162float(AHEAD ? br : pb[(size_t)r * PW]);
            const float a = __bfloat162float(AHEAD ? ar : pa[(size_t)r * PW]);
            gates[0] = expf(-expf(a_log[hv]) * softplusf_(a + dt_bias[hv]));
            gates[1] = bf(sigmoidf_(b));
        }
        __syncthreads();
        const float g = gates[0], beta = gates[1];
        float kk[4], qq[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { kk[i] = ks[lane * 4 + i]; qq[i] = qs[lane * 4 + i]; }
        update(s, kk, vs, warp, g, beta);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
            o = warp_sum(o);
            if (lane == 0) ys[warp * 4 + j] = bf(o);
        }
        if (k_save != nullptr) {
            if (t < DK && hv % (NV / NK) == 0) k_save[((size_t)r * NK + hk) * DK + t] = ks[t];
            if (t < DV) v_save[((size_t)r * NV + hv) * DV + t] = __float2bfloat16_rn(vs[t]);
            if (t == 0) { g_save[r * NV + hv] = g; b_save[r * NV + hv] = beta; }
        }
        __syncthreads();
        if (warp == 0) {
            float ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { const float y = ys[lane * 4 + i]; ss = ss + y * y; }
            ss = warp_sum(ss);
            if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
        }
        __syncthreads();
        if (t < DV) {
            const float yn = bf(bf(ys[t] * rinv) * __bfloat162float(norm_w[t]));
            const float z = __bfloat162float(AHEAD ? zr : pz[(size_t)r * PW]);
            const float o = bf(yn * sigmoidf_(z));
            out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(o);
            const float gs = warp_sum(o);
            if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + warp] = gs;
        }
        __syncthreads();
    }
    if (state_out != nullptr) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
    }
}

template <int NK, int NV>
__global__ void __launch_bounds__(1024) replay_kernel(
        const float* __restrict__ state_in, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save,
        const float* __restrict__ b_save, int rows, float* __restrict__ state_out) {
    const int hv = blockIdx.x, hk = hv / (NV / NK);
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float vs[DV];
    float s[4][4];
    const size_t sbase = (size_t)hv * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    for (int r = 0; r < rows; ++r) {
        if (t < DV) vs[t] = __bfloat162float(v_save[((size_t)r * NV + hv) * DV + t]);
        __syncthreads();
        float kk[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) kk[i] = k_save[((size_t)r * NK + hk) * DK + lane * 4 + i];
        update(s, kk, vs, warp, g_save[r * NV + hv], b_save[r * NV + hv]);
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
}

} // namespace tf_fn_gdn

// The instantiations gdn_chain_cuda and gdn_replay_cuda launch: one GPU (16, 48) and a TP=2 rank (8, 24).
#define TF_CHAIN(NK, NV, AHEAD) template __global__ void tf_fn_gdn::chain_kernel<NK, NV, AHEAD>( \
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, const float*, \
    const __nv_bfloat16*, float, int, __nv_bfloat16*, float*, float*, float*, __nv_bfloat16*, float*, float*);
TF_CHAIN(16, 48, true)
TF_CHAIN(16, 48, false)
TF_CHAIN(8, 24, true)
TF_CHAIN(8, 24, false)
#define TF_REPLAY(NK, NV) template __global__ void tf_fn_gdn::replay_kernel<NK, NV>(const float*, const float*, \
    const __nv_bfloat16*, const float*, const float*, int, float*);
TF_REPLAY(16, 48)
TF_REPLAY(8, 24)
