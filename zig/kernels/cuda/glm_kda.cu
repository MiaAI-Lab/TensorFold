// GLM-5.3-Flash KDA (Kimi delta attention) decode and prefix replay.
//
// Device code of families/glm5_next/cuda/kda.cu (TensorFold v0.6.0 plus the recipe's patches;
// Ash Hart and the TensorFold contributors). Arithmetic, including the shared ``update`` and the
// wide path's prep / step<4,16> / out split, is that file's, in the same order. The torch extension
// glue (kda.cpp, kda_chain_*_cuda) is not in this image: zig/src/families/glm_cuda/cuda_kda.zig
// launches these symbols with that wrapper's grids. Built -O3 --fmad=false, as tensorfold_glm_kda_v2.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace tf_glm_kda {


constexpr int DK = 128, DV = 128, TAPS = 4;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float warp_sum(float x) {
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

// One delta-rule step on this thread's 4 x 4 block of the state: per-channel decay, read (k), correct toward v.
__device__ __forceinline__ void update(float (&s)[4][4], const float (&kk)[4], const float (&gg)[4],
                                       const float* vrow, int warp, float beta) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float kv = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            s[j][i] = s[j][i] * gg[i];
            kv = kv + s[j][i] * kk[i];
        }
        kv = warp_sum(kv);
        const float delta = (vrow[warp * 4 + j] - kv) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = s[j][i] + kk[i] * delta;
    }
}

__global__ void __launch_bounds__(1024) chain_kernel(
        int H, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off,
        const __nv_bfloat16* __restrict__ A, int a_stride, const __nv_bfloat16* __restrict__ G, int g_stride,
        const __nv_bfloat16* __restrict__ cs, const __nv_bfloat16* __restrict__ cw,
        const float* __restrict__ state_in, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        const __nv_bfloat16* __restrict__ norm_w, float eps, float lower, int rows,
        __nv_bfloat16* __restrict__ out, float* __restrict__ state_out,
        float* __restrict__ k_save, __nv_bfloat16* __restrict__ v_save, float* __restrict__ g_save,
        float* __restrict__ b_save) {
    const int C = 3 * H * DK;                         // conv channels: q | k | v
    const int h = blockIdx.x;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float qs[DK], ks[DK], vs[DV], ys[DV], gs[DK];
    __shared__ float beta_s, rinv;
    int c = -1;
    if (t < 3 * DK) c = (t / DK) * H * DK + h * DK + (t % DK);
    float s[4][4];
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    const float decay_rate = expf(a_log[h]);
    for (int r = 0; r < rows; ++r) {
        if (c >= 0) {
            float acc = 0.0f;
#pragma unroll
            for (int tap = 0; tap < TAPS; ++tap) {
                const int at = r + tap;
                const float x = at < TAPS - 1 ? __bfloat162float(cs[(size_t)at * C + c])
                                              : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * p_stride + c]);
                acc = acc + __bfloat162float(cw[(size_t)c * TAPS + tap]) * x;
            }
            const float act = bf(acc / (1.0f + expf(-acc)));
            if (t < DK) qs[t] = act;
            else if (t < 2 * DK) ks[t - DK] = act;
            else vs[t - 2 * DK] = act;
        } else if (t >= 512 && t < 512 + DK) {
            const int i = t - 512;
            const float a = __bfloat162float(A[(size_t)r * a_stride + h * DK + i]) + dt_bias[h * DK + i];
            gs[i] = expf(lower * sigmoidf_(decay_rate * a));
        } else if (t == 1023) {
            beta_s = bf(sigmoidf_(__bfloat162float(P[(size_t)r * p_stride + b_off + h])));
        }
        __syncthreads();
        if (warp < 2) {
            float* x = warp == 0 ? qs : ks;
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { v4[i] = x[lane * 4 + i]; ss = ss + v4[i] * v4[i]; }
            ss = warp_sum(ss);
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            __syncwarp();
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float y = v4[i] * inv;
                if (warp == 0) y = y * (1.0f / sqrtf((float)DK));
                x[lane * 4 + i] = y;
            }
        }
        __syncthreads();
        const float beta = beta_s;
        float kk[4], qq[4], gg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { kk[i] = ks[lane * 4 + i]; qq[i] = qs[lane * 4 + i]; gg[i] = gs[lane * 4 + i]; }
        update(s, kk, gg, vs, warp, beta);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
            o = warp_sum(o);
            if (lane == 0) ys[warp * 4 + j] = bf(o);
        }
        if (k_save != nullptr) {
            const size_t base = ((size_t)r * H + h) * DK;
            if (t < DK) { k_save[base + t] = ks[t]; g_save[base + t] = gs[t]; }
            if (t >= DK && t < DK + DV) v_save[base + t - DK] = __float2bfloat16_rn(vs[t - DK]);
            if (t == 0) b_save[r * H + h] = beta;
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
            const float yn = ys[t] * rinv;
            const float yw = __bfloat162float(norm_w[t]) * yn;
            const float gate = __bfloat162float(G[(size_t)r * g_stride + h * DV + t]);
            out[(size_t)r * H * DV + h * DV + t] = __float2bfloat16_rn(yw * sigmoidf_(gate));
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

__global__ void __launch_bounds__(1024) replay_kernel(
        int H, const float* __restrict__ state_in, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save,
        const float* __restrict__ b_save, int rows, float* __restrict__ state_out) {
    const int h = blockIdx.x;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float vs[DV];
    float s[4][4];
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    for (int r = 0; r < rows; ++r) {
        const size_t base = ((size_t)r * H + h) * DK;
        if (t < DV) vs[t] = __bfloat162float(v_save[base + t]);
        __syncthreads();
        float kk[4], gg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { kk[i] = k_save[base + lane * 4 + i]; gg[i] = g_save[base + lane * 4 + i]; }
        update(s, kk, gg, vs, warp, b_save[r * H + h]);
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
}

// All layers at once: block (layer, head); per-layer strides of the state buffers and saved rows.
__global__ void __launch_bounds__(1024) replay_layers_kernel(
        int H, const float* __restrict__ state_in, size_t state_stride, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save, const float* __restrict__ b_save,
        size_t kv_stride, size_t b_stride, int rows, float* __restrict__ state_out) {
    const int layer = blockIdx.x / H, h = blockIdx.x % H;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float vs[DV];
    float s[4][4];
    const float* sin = state_in + layer * state_stride;
    float* sout = state_out + layer * state_stride;
    const float* ks = k_save + layer * kv_stride;
    const __nv_bfloat16* vsv = v_save + layer * kv_stride;
    const float* gsv = g_save + layer * kv_stride;
    const float* bsv = b_save + layer * b_stride;
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = sin[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    for (int r = 0; r < rows; ++r) {
        const size_t base = ((size_t)r * H + h) * DK;
        if (t < DV) vs[t] = __bfloat162float(vsv[base + t]);
        __syncthreads();
        float kk[4], gg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { kk[i] = ks[base + lane * 4 + i]; gg[i] = gsv[base + lane * 4 + i]; }
        update(s, kk, gg, vs, warp, bsv[r * H + h]);
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) sout[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
}

// Long windows: state-independent work for all rows (prep_kernel, out_kernel), the delta rule alone row by row (step_kernel); chain_kernel's bits.

// Block (row, head), 512 threads: conv + SiLU of the 3 x 128 q|k|v channels, the decay, beta, then the q/k norms.
__global__ void __launch_bounds__(512) prep_kernel(
        int H, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off,
        const __nv_bfloat16* __restrict__ A, int a_stride, const __nv_bfloat16* __restrict__ cs,
        const __nv_bfloat16* __restrict__ cw, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        float lower, float* __restrict__ q_out, float* __restrict__ k_save, __nv_bfloat16* __restrict__ v_save,
        float* __restrict__ g_save, float* __restrict__ b_save) {
    const int C = 3 * H * DK;
    const int r = blockIdx.x, h = blockIdx.y;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float qs[DK], ks[DK];
    const size_t base = ((size_t)r * H + h) * DK;
    if (t < 3 * DK) {
        const int c = (t / DK) * H * DK + h * DK + (t % DK);
        float acc = 0.0f;
#pragma unroll
        for (int tap = 0; tap < TAPS; ++tap) {
            const int at = r + tap;
            const float x = at < TAPS - 1 ? __bfloat162float(cs[(size_t)at * C + c])
                                          : __bfloat162float(P[(size_t)(at - (TAPS - 1)) * p_stride + c]);
            acc = acc + __bfloat162float(cw[(size_t)c * TAPS + tap]) * x;
        }
        const float act = bf(acc / (1.0f + expf(-acc)));
        if (t < DK) qs[t] = act;
        else if (t < 2 * DK) ks[t - DK] = act;
        else v_save[base + t - 2 * DK] = __float2bfloat16_rn(act);
    } else {
        const int i = t - 3 * DK;
        const float decay_rate = expf(a_log[h]);
        const float a = __bfloat162float(A[(size_t)r * a_stride + h * DK + i]) + dt_bias[h * DK + i];
        g_save[base + i] = expf(lower * sigmoidf_(decay_rate * a));
        if (i == 0) b_save[r * H + h] = bf(sigmoidf_(__bfloat162float(P[(size_t)r * p_stride + b_off + h])));
    }
    __syncthreads();
    if (warp < 2) {
        const float* x = warp == 0 ? qs : ks;
        float v4[4], ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { v4[i] = x[lane * 4 + i]; ss = ss + v4[i] * v4[i]; }
        ss = warp_sum(ss);
        float inv = 1.0f / sqrtf(ss + 1e-6f);
        float* dst = warp == 0 ? q_out : k_save;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float y = v4[i] * inv;
            if (warp == 0) y = y * (1.0f / sqrtf((float)DK));
            dst[base + lane * 4 + i] = y;
        }
    }
}

// Block (head, part): each warp owns 4 value rows; TR rows of saved k, q, decay, v and beta are staged in shared memory at a time.
template <int WARPS, int TR>
__global__ void __launch_bounds__(WARPS * 32) step_kernel(
        int H, const float* __restrict__ state_in, const float* __restrict__ q_in, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save, const float* __restrict__ b_save,
        int rows, __nv_bfloat16* __restrict__ y_out, float* __restrict__ state_out) {
    constexpr int NT = WARPS * 32, VR = WARPS * 4;
    const int h = blockIdx.x, v0 = blockIdx.y * VR;
    const int lw = threadIdx.x >> 5, lane = threadIdx.x & 31, warp = blockIdx.y * WARPS + lw;
    __shared__ __align__(16) float kt[TR][DK];
    __shared__ __align__(16) float qt[TR][DK];
    __shared__ __align__(16) float gt[TR][DK];
    __shared__ __align__(16) float vt[TR][VR];
    __shared__ float bt[TR];
    float s[4][4];
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    for (int r0 = 0; r0 < rows; r0 += TR) {
        const int n = rows - r0 < TR ? rows - r0 : TR;
        __syncthreads();
        for (int x = threadIdx.x; x < n * DK; x += NT) {
            const int rr = x / DK, c = x % DK;
            const size_t base = ((size_t)(r0 + rr) * H + h) * DK + c;
            kt[rr][c] = k_save[base];
            qt[rr][c] = q_in[base];
            gt[rr][c] = g_save[base];
        }
        for (int x = threadIdx.x; x < n * VR; x += NT) {
            const int rr = x / VR, c = x % VR;
            vt[rr][c] = __bfloat162float(v_save[((size_t)(r0 + rr) * H + h) * DV + v0 + c]);
        }
        if (threadIdx.x < n) bt[threadIdx.x] = b_save[(r0 + threadIdx.x) * H + h];
        __syncthreads();
        for (int rr = 0; rr < n; ++rr) {
            const float4 k4 = reinterpret_cast<const float4*>(kt[rr])[lane];      // one 16-byte load a lane:
            const float4 q4 = reinterpret_cast<const float4*>(qt[rr])[lane];      // no bank conflicts
            const float4 g4 = reinterpret_cast<const float4*>(gt[rr])[lane];
            const float4 v4 = reinterpret_cast<const float4*>(vt[rr])[lw];
            float kk[4] = {k4.x, k4.y, k4.z, k4.w}, qq[4] = {q4.x, q4.y, q4.z, q4.w};
            float gg[4] = {g4.x, g4.y, g4.z, g4.w}, vv[4] = {v4.x, v4.y, v4.z, v4.w};
            update(s, kk, gg, vv, 0, bt[rr]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float o = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
                o = warp_sum(o);
                if (lane == 0) y_out[((size_t)(r0 + rr) * H + h) * DV + warp * 4 + j] = __float2bfloat16_rn(o);
            }
        }
    }
    if (state_out != nullptr) {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
    }
}

// Block (row, head), 128 threads: the gated RMSNorm of the read-out.
__global__ void __launch_bounds__(128) out_kernel(
        int H, const __nv_bfloat16* __restrict__ y_in, const __nv_bfloat16* __restrict__ G, int g_stride,
        const __nv_bfloat16* __restrict__ norm_w, float eps, __nv_bfloat16* __restrict__ out) {
    const int r = blockIdx.x, h = blockIdx.y;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float rinv;
    const size_t base = ((size_t)r * H + h) * DV;
    if (warp == 0) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float y = __bfloat162float(y_in[base + lane * 4 + i]); ss = ss + y * y; }
        ss = warp_sum(ss);
        if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
    }
    __syncthreads();
    const float yn = __bfloat162float(y_in[base + t]) * rinv;
    const float yw = __bfloat162float(norm_w[t]) * yn;
    const float gate = __bfloat162float(G[(size_t)r * g_stride + h * DV + t]);
    out[(size_t)r * H * DV + h * DV + t] = __float2bfloat16_rn(yw * sigmoidf_(gate));
}

// Several streams' windows in one launch. A segment table (int32 [nseg, SEG_COLS]: first row, rows, state slot,
// parity, conv slot, keep) cuts the window's rows into streams; each segment reads its conv taps from its own
// conv slot and runs the recurrence from rec[slot, parity] into rec[slot, 1 - parity]. Per row the arithmetic is
// the solo kernels' line for line (same ``update``), so a segment's bits equal a solo run of its rows. A segment
// of 0 rows is skipped: a table sized for the most streams serves fewer (grids stay fixed for CUDA graphs).
constexpr int SEG_COLS = 6, SEG_ROW0 = 0, SEG_ROWS = 1, SEG_SLOT = 2, SEG_PARITY = 3, SEG_CONV = 4, SEG_KEEP = 5;

// prep_kernel for block (row, head) of the whole window; the row's conv taps come from its segment.
__global__ void __launch_bounds__(512) prep_seg_kernel(
        int H, const int* __restrict__ seg, int nseg, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off,
        const __nv_bfloat16* __restrict__ A, int a_stride, const __nv_bfloat16* __restrict__ conv,
        size_t conv_slot_stride, const __nv_bfloat16* __restrict__ cw, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, float lower, float* __restrict__ q_out, float* __restrict__ k_save,
        __nv_bfloat16* __restrict__ v_save, float* __restrict__ g_save, float* __restrict__ b_save) {
    const int C = 3 * H * DK;
    const int r = blockIdx.x, h = blockIdx.y;
    int row0 = -1, cslot = 0;
    for (int i = 0; i < nseg; ++i) {
        const int* e = seg + i * SEG_COLS;
        if (r >= e[SEG_ROW0] && r < e[SEG_ROW0] + e[SEG_ROWS]) { row0 = e[SEG_ROW0]; cslot = e[SEG_CONV]; }
    }
    if (row0 < 0) return;                             // a row no segment covers
    const int rr = r - row0;                          // the row within its stream's window
    const __nv_bfloat16* cs = conv + (size_t)cslot * conv_slot_stride;
    const __nv_bfloat16* Ps = P + (size_t)row0 * p_stride;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float qs[DK], ks[DK];
    const size_t base = ((size_t)r * H + h) * DK;
    if (t < 3 * DK) {
        const int c = (t / DK) * H * DK + h * DK + (t % DK);
        float acc = 0.0f;
#pragma unroll
        for (int tap = 0; tap < TAPS; ++tap) {
            const int at = rr + tap;
            const float x = at < TAPS - 1 ? __bfloat162float(cs[(size_t)at * C + c])
                                          : __bfloat162float(Ps[(size_t)(at - (TAPS - 1)) * p_stride + c]);
            acc = acc + __bfloat162float(cw[(size_t)c * TAPS + tap]) * x;
        }
        const float act = bf(acc / (1.0f + expf(-acc)));
        if (t < DK) qs[t] = act;
        else if (t < 2 * DK) ks[t - DK] = act;
        else v_save[base + t - 2 * DK] = __float2bfloat16_rn(act);
    } else {
        const int i = t - 3 * DK;
        const float decay_rate = expf(a_log[h]);
        const float a = __bfloat162float(A[(size_t)r * a_stride + h * DK + i]) + dt_bias[h * DK + i];
        g_save[base + i] = expf(lower * sigmoidf_(decay_rate * a));
        if (i == 0) b_save[r * H + h] = bf(sigmoidf_(__bfloat162float(P[(size_t)r * p_stride + b_off + h])));
    }
    __syncthreads();
    if (warp < 2) {
        const float* x = warp == 0 ? qs : ks;
        float v4[4], ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { v4[i] = x[lane * 4 + i]; ss = ss + v4[i] * v4[i]; }
        ss = warp_sum(ss);
        float inv = 1.0f / sqrtf(ss + 1e-6f);
        float* dst = warp == 0 ? q_out : k_save;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float y = v4[i] * inv;
            if (warp == 0) y = y * (1.0f / sqrtf((float)DK));
            dst[base + lane * 4 + i] = y;
        }
    }
}

// step_kernel for block (segment x head, part): the segment's rows from rec[slot, parity] into rec[slot, 1 - parity].
template <int WARPS, int TR>
__global__ void __launch_bounds__(WARPS * 32) step_seg_kernel(
        int H, const int* __restrict__ seg, float* __restrict__ rec, size_t slot_stride, size_t parity_stride,
        const float* __restrict__ q_in, const float* __restrict__ k_save, const __nv_bfloat16* __restrict__ v_save,
        const float* __restrict__ g_save, const float* __restrict__ b_save, __nv_bfloat16* __restrict__ y_out) {
    constexpr int NT = WARPS * 32, VR = WARPS * 4;
    const int* e = seg + (blockIdx.x / H) * SEG_COLS;
    const int row0 = e[SEG_ROW0], rows = e[SEG_ROWS], slot = e[SEG_SLOT], parity = e[SEG_PARITY];
    if (rows <= 0) return;
    const float* state_in = rec + (size_t)slot * slot_stride + (size_t)parity * parity_stride;
    float* state_out = rec + (size_t)slot * slot_stride + (size_t)(1 - parity) * parity_stride;
    const int h = blockIdx.x % H, v0 = blockIdx.y * VR;
    const int lw = threadIdx.x >> 5, lane = threadIdx.x & 31, warp = blockIdx.y * WARPS + lw;
    __shared__ __align__(16) float kt[TR][DK];
    __shared__ __align__(16) float qt[TR][DK];
    __shared__ __align__(16) float gt[TR][DK];
    __shared__ __align__(16) float vt[TR][VR];
    __shared__ float bt[TR];
    float s[4][4];
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = state_in[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    for (int r0 = 0; r0 < rows; r0 += TR) {
        const int n = rows - r0 < TR ? rows - r0 : TR;
        __syncthreads();
        for (int x = threadIdx.x; x < n * DK; x += NT) {
            const int rr = x / DK, c = x % DK;
            const size_t base = ((size_t)(row0 + r0 + rr) * H + h) * DK + c;
            kt[rr][c] = k_save[base];
            qt[rr][c] = q_in[base];
            gt[rr][c] = g_save[base];
        }
        for (int x = threadIdx.x; x < n * VR; x += NT) {
            const int rr = x / VR, c = x % VR;
            vt[rr][c] = __bfloat162float(v_save[((size_t)(row0 + r0 + rr) * H + h) * DV + v0 + c]);
        }
        if (threadIdx.x < n) bt[threadIdx.x] = b_save[(row0 + r0 + threadIdx.x) * H + h];
        __syncthreads();
        for (int rr = 0; rr < n; ++rr) {
            const float4 k4 = reinterpret_cast<const float4*>(kt[rr])[lane];
            const float4 q4 = reinterpret_cast<const float4*>(qt[rr])[lane];
            const float4 g4 = reinterpret_cast<const float4*>(gt[rr])[lane];
            const float4 v4 = reinterpret_cast<const float4*>(vt[rr])[lw];
            float kk[4] = {k4.x, k4.y, k4.z, k4.w}, qq[4] = {q4.x, q4.y, q4.z, q4.w};
            float gg[4] = {g4.x, g4.y, g4.z, g4.w}, vv[4] = {v4.x, v4.y, v4.z, v4.w};
            update(s, kk, gg, vv, 0, bt[rr]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float o = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) o = o + s[j][i] * qq[i];
                o = warp_sum(o);
                if (lane == 0)
                    y_out[((size_t)(row0 + r0 + rr) * H + h) * DV + warp * 4 + j] = __float2bfloat16_rn(o);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) state_out[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
}

// replay_layers_kernel for block (segment, layer, head): the segment's first ``keep`` rows from rec[slot, parity]
// into rec[slot, 1 - parity]; keep == rows (the chain already left that state) or rows == 0 skips the segment.
__global__ void __launch_bounds__(1024) replay_layers_seg_kernel(
        int H, int layers, const int* __restrict__ seg, float* __restrict__ rec, size_t slot_stride,
        size_t parity_stride, size_t layer_stride, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save, const float* __restrict__ b_save,
        size_t kv_stride, size_t b_stride) {
    const int* e = seg + (blockIdx.x / (layers * H)) * SEG_COLS;
    const int row0 = e[SEG_ROW0], rows = e[SEG_ROWS], slot = e[SEG_SLOT], parity = e[SEG_PARITY];
    const int keep = e[SEG_KEEP];
    if (rows <= 0 || keep < 0 || keep >= rows) return;
    const int layer = (blockIdx.x / H) % layers, h = blockIdx.x % H;
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float vs[DV];
    float s[4][4];
    const float* sin = rec + (size_t)slot * slot_stride + (size_t)parity * parity_stride + layer * layer_stride;
    float* sout = rec + (size_t)slot * slot_stride + (size_t)(1 - parity) * parity_stride + layer * layer_stride;
    const float* ks = k_save + layer * kv_stride + (size_t)row0 * H * DK;
    const __nv_bfloat16* vsv = v_save + layer * kv_stride + (size_t)row0 * H * DV;
    const float* gsv = g_save + layer * kv_stride + (size_t)row0 * H * DK;
    const float* bsv = b_save + layer * b_stride + (size_t)row0 * H;
    const size_t sbase = (size_t)h * DV * DK;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) s[j][i] = sin[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i];
    for (int r = 0; r < keep; ++r) {
        const size_t base = ((size_t)r * H + h) * DK;
        if (t < DV) vs[t] = __bfloat162float(vsv[base + t]);
        __syncthreads();
        float kk[4], gg[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) { kk[i] = ks[base + lane * 4 + i]; gg[i] = gsv[base + lane * 4 + i]; }
        update(s, kk, gg, vs, warp, bsv[r * H + h]);
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int i = 0; i < 4; ++i) sout[sbase + (size_t)(warp * 4 + j) * DK + lane * 4 + i] = s[j][i];
}


}  // namespace tf_glm_kda

// The instantiations kda_chain_wide_cuda and kda_chain_wide_segments_cuda launch.
template __global__ void tf_glm_kda::step_kernel<4, 16>(
    int, const float*, const float*, const float*, const __nv_bfloat16*, const float*, const float*,
    int, __nv_bfloat16*, float*);
template __global__ void tf_glm_kda::step_seg_kernel<4, 16>(
    int, const int*, float*, size_t, size_t, const float*, const float*, const __nv_bfloat16*,
    const float*, const float*, __nv_bfloat16*);
