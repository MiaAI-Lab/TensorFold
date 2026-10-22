// GLM-5.3-Flash KDA prompt chunks in chunked (WY) form.
//
// Device code of families/glm5_next/cuda/kda_chunk.cu (TensorFold v0.6.0 plus the recipe's patches;
// Ash Hart and the TensorFold contributors). The sub-chunk arithmetic is that file's, in the same
// order (explicit __fmaf_rn, CUDA expf, the same div_rn). The torch glue (kda_chunk.cpp,
// kda_chunk_cuda) is not in this image: zig/src/families/glm_cuda/cuda_kda.zig launches
// chunk_kernel with that wrapper's grid (H, 256) and dynamic shared memory sizeof(Smem).
// Built -O3 --fmad=false, as tensorfold_glm_kda_chunk. Not the serial chain's bits
// (kda_chunked.py): a row depends on its 32-row sub-chunk and the state entering it.
// KDA_PROF is left off; the engine does not ship the clock counters.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tf_glm_kda_chunk {


constexpr int DK = 128, DV = 128, TAPS = 4;
constexpr int BT = 32;                  // rows of a sub-chunk
constexpr int HALO = TAPS - 1;          // conv rows before a sub-chunk
constexpr int NT = 256, NW = NT / 32;   // threads, warps
constexpr int VPT = 4;                  // value rows of the state a thread holds
constexpr int KP = NT * VPT / DV;       // threads sharing those value rows (8), each 16 keys
constexpr int NQ = DK / 4 / KP;         // key quads a thread holds (4)
constexpr int OWN = BT / 2;             // rows a thread finishes (16, one parity)
constexpr int LS = BT + 4;              // row stride of L
constexpr int PCH = 3 * DK;             // projection channels of a head (q | k | v)
constexpr int PF_P = ((BT + HALO) * PCH / 8 + NT - 1) / NT;   // 16-byte loads a thread prefetches (7)
constexpr int PF_A = BT * DK / 8 / NT;                         // (2)
static_assert(KP == 8 && NQ == 4 && PF_A * NT * 8 == BT * DK, "layout");

struct Smem {
    float q[BT][DK];          // q -> q E(s) -> V'
    float k[BT][DK];          // k -> k E(s) -> (q E) S0^T
    float g[BT][DK];          // decays -> Y
    union {                   // staged projection rows (conv input) -> k E(s ->), L, T, M
        __nv_bfloat16 raw_p[BT + HALO][PCH];
        struct {
            float ke[BT][DK];         // k E(s ->)
            float L[BT][LS];          // L[s][t]
            float tt[BT][BT];         // T transposed: tt[t][s] = T[s][t]
            float mt[BT][BT];         // M transposed: mt[t][s] = M[s][t]
        } x;
    } u;
    __nv_bfloat16 v[BT][DV];
    __nv_bfloat16 raw_a[BT][DK];      // staged decay rows
    __nv_bfloat16 cw[PCH][TAPS];      // this head's conv weights
    float raw_b[BT];
    float beta[BT];
    float egl[DK];            // E(31)
};

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ldb(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float fma_(float a, float b, float c) { return __fmaf_rn(a, b, c); }
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

// a / b with IEEE rounding, branch-free: the fast path of CUDA's division (a refined reciprocal, then one
// correction), which is the correctly rounded quotient whenever ``div_ok`` holds (checked on 742 M operand pairs
// against a / b); callers take a / b for the rare operands outside (a huge denominator, a tiny numerator).
__device__ __forceinline__ float div_rn(float a, float b) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(b));
    const float e = __fmaf_rn(-b, r, 1.0f);
    r = __fmaf_rn(e, r, r);
    const float q = a * r;
    const float rem = __fmaf_rn(-b, q, a);
    return __fmaf_rn(rem, r, q);
}

__device__ __forceinline__ bool div_ok(float a, float b) {
    return (a == 0.0f || fabsf(a) > 1e-36f) && b >= 1.0f && b < 8e37f;
}

__device__ __forceinline__ float warp_sum(float x) {
    for (int o = 16; o; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

// 8 bf16 from p: one 16-byte load when aligned, else 8 loads.
__device__ __forceinline__ uint4 ld8(const __nv_bfloat16* p, bool vec) {
    if (vec) return *reinterpret_cast<const uint4*>(p);
    const unsigned short* s = reinterpret_cast<const unsigned short*>(p);
    uint4 r;
    r.x = s[0] | ((unsigned)s[1] << 16); r.y = s[2] | ((unsigned)s[3] << 16);
    r.z = s[4] | ((unsigned)s[5] << 16); r.w = s[6] | ((unsigned)s[7] << 16);
    return r;
}

// One level of a transposing shuffle tree: this lane keeps the half of a[0..N) its ``hi`` bit selects, adds the
// partner's (lane ^ mask) copy of that half, and sends the other half.
template <int N>
__device__ __forceinline__ void tree_level(float (&a)[N], int mask, bool hi) {
#pragma unroll
    for (int j = 0; j < N / 2; ++j) {
        const float send = hi ? a[j] : a[j + N / 2];
        const float keep = hi ? a[j + N / 2] : a[j];
        a[j] = keep + __shfl_xor_sync(0xffffffffu, send, mask);
    }
}

// Sums of a[0..8) over the lanes xor 4, 2, 1: lane bits (2, 1, 0) = i gets the sum of a[i].
__device__ __forceinline__ float tree8(float (&a)[8], int lane) {
    tree_level<8>(a, 4, lane & 4);
    float b[4] = {a[0], a[1], a[2], a[3]};
    tree_level<4>(b, 2, lane & 2);
    float c[2] = {b[0], b[1]};
    tree_level<2>(c, 1, lane & 1);
    return c[0];
}

// Sums of a[0..16) over the lanes xor m0 > m1 > m2 > m3: the lane whose bits (m0, m1, m2, m3) spell i gets a[i]'s.
__device__ __forceinline__ float tree16(float (&a)[16], int lane, int m0, int m1, int m2, int m3) {
    tree_level<16>(a, m0, lane & m0);
    float b[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) b[j] = a[j];
    tree_level<8>(b, m1, lane & m1);
    float c[4] = {b[0], b[1], b[2], b[3]};
    tree_level<4>(c, m2, lane & m2);
    float d[2] = {c[0], c[1]};
    tree_level<2>(d, m3, lane & m3);
    return d[0];
}

#ifdef KDA_PROF
__device__ unsigned long long g_prof[16];
#define MARK(i) \
    do { if (tid == 0 && h == 0) { const long long t1 = clock64(); g_prof[i] += t1 - t0_; t0_ = t1; } } while (0)
#else
#define MARK(i) do {} while (0)
#endif

__device__ __forceinline__ void prefetch_l2(const void* p) {
    asm volatile("prefetch.global.L2 [%0];" :: "l"(p));
}

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(d), "l"(src));
}

// Warm L2 with the rows of the sub-chunk whose row 0 is prompt-chunk row r0 (projection rows r0 - 3 .. r0 + 31 of
// this head's q | k | v channels, its decay rows and beta logits; not when ``next`` is false) and the output gate
// rows from gate_r0.
__device__ __forceinline__ void warm(int tid, int h, int H, int r0, bool next, int rows, int gate_r0,
                                     const __nv_bfloat16* __restrict__ P, int p_stride, int b_off,
                                     const __nv_bfloat16* __restrict__ A, int a_stride,
                                     const __nv_bfloat16* __restrict__ G, int g_stride) {
    constexpr int LP = (BT + HALO) * 3 * 2, LA = BT * 2, LB = BT, LG = BT * 2;   // 128-byte lines
    for (int x = next ? tid : tid + LP + LA + LB; x < LP + LA + LB + LG; x += NT) {
        if (x < LP) {
            const int row = x / 6, part = (x % 6) / 2, half = x % 2, r = r0 - HALO + row;
            if (r >= 0 && r < rows) prefetch_l2(P + (size_t)r * p_stride + part * H * DK + h * DK + 64 * half);
        } else if (x < LP + LA) {
            const int y = x - LP, r = r0 + y / 2;
            if (r >= 0 && r < rows) prefetch_l2(A + (size_t)r * a_stride + h * DK + 64 * (y % 2));
        } else if (x < LP + LA + LB) {
            const int r = r0 + (x - LP - LA);
            if (r >= 0 && r < rows) prefetch_l2(P + (size_t)r * p_stride + b_off + h);
        } else {
            const int y = x - LP - LA - LB, r = gate_r0 + y / 2;
            if (r >= 0 && r < rows) prefetch_l2(G + (size_t)r * g_stride + h * DV + 64 * (y % 2));
        }
    }
}

// Stage the rows of the sub-chunk whose row 0 is prompt-chunk row r0 into shared memory: projection rows r0 - 3 ..
// r0 + 31 (the conv state before row 0, zero rows outside), decay rows and the beta logits. Global to shared
// directly (cp.async) when the rows are 16-byte aligned; the caller waits (cp.async.wait_all) and syncs.
__device__ __forceinline__ void stage(Smem& sm, int tid, int h, int H, int r0, int rows,
                                      const __nv_bfloat16* __restrict__ P, int p_stride, int b_off,
                                      const __nv_bfloat16* __restrict__ A, int a_stride,
                                      const __nv_bfloat16* __restrict__ cs, bool vec_p, bool vec_a) {
    const int C = 3 * H * DK;
    for (int item = tid; item < (BT + HALO) * PCH / 8; item += NT) {
        const int row = item / (PCH / 8), q8 = item % (PCH / 8);
        const int ch = (q8 / (DK / 8)) * H * DK + h * DK + 8 * (q8 % (DK / 8));
        const int r = r0 - HALO + row;
        uint4* dst = reinterpret_cast<uint4*>(&sm.u.raw_p[row][0]) + q8;
        const __nv_bfloat16* src = nullptr;
        if (r >= 0 && r < rows) src = P + (size_t)r * p_stride + ch;
        else if (r < 0 && r >= -HALO) src = cs + (size_t)(r + HALO) * C + ch;
        if (src == nullptr) *dst = make_uint4(0, 0, 0, 0);
        else if (vec_p) cp16(dst, src);
        else *dst = ld8(src, false);
    }
    for (int item = tid; item < BT * DK / 8; item += NT) {
        const int row = item / (DK / 8), c8 = item % (DK / 8);
        const int r = r0 + row;
        uint4* dst = reinterpret_cast<uint4*>(&sm.raw_a[row][0]) + c8;
        const __nv_bfloat16* src = A + (size_t)r * a_stride + h * DK + 8 * c8;
        if (r < 0 || r >= rows) *dst = make_uint4(0, 0, 0, 0);
        else if (vec_a) cp16(dst, src);
        else *dst = ld8(src, false);
    }
    if (tid < BT) {
        const int r = r0 + tid;
        sm.raw_b[tid] = (r >= 0 && r < rows) ? ldb(P + (size_t)r * p_stride + b_off + h) : 0.0f;
    }
}

__global__ void __launch_bounds__(NT, 1) chunk_kernel(
        int H, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off,
        const __nv_bfloat16* __restrict__ A, int a_stride, const __nv_bfloat16* __restrict__ G, int g_stride,
        const __nv_bfloat16* __restrict__ cs, const __nv_bfloat16* __restrict__ cw,
        const float* __restrict__ state_in, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
        const __nv_bfloat16* __restrict__ norm_w, float eps, float lower, int rows, int off,
        __nv_bfloat16* __restrict__ out, float* __restrict__ state_out) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
    const int h = blockIdx.x, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int vg = tid / KP, kp = tid % KP;
    const int par = kp < 4 ? 1 : 0;                   // parity of the rows this thread finishes
    const int ov = VPT * vg + (kp & 3);               // their value column
    const size_t sbase = (size_t)h * DV * DK;
    const bool vec_p = (p_stride % 8 == 0) && ((uintptr_t)P % 16 == 0) && ((uintptr_t)cs % 16 == 0);
    const bool vec_a = (a_stride % 8 == 0) && ((uintptr_t)A % 16 == 0);

    float s[VPT][NQ][4];
#pragma unroll
    for (int vv = 0; vv < VPT; ++vv)
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
            const float4 x =
                reinterpret_cast<const float4*>(state_in + sbase + (size_t)(VPT * vg + vv) * DK)[kp + KP * i];
            s[vv][i][0] = x.x; s[vv][i][1] = x.y; s[vv][i][2] = x.z; s[vv][i][3] = x.w;
        }
    const float rate = expf(a_log[h]);
    const float dtb = dt_bias[h * DK + (tid & (DK - 1))];     // the decay channel this thread computes (tid % 128)
    float nw[4];                                      // the norm weights of value columns 4 lane .. 4 lane + 3
#pragma unroll
    for (int x = 0; x < 4; ++x) nw[x] = ldb(norm_w + 4 * lane + x);
    const bool vec_g = (g_stride % 4 == 0) && ((uintptr_t)G % 8 == 0);
    const bool vec_o = ((H * DV) % 4 == 0) && ((uintptr_t)out % 8 == 0);
    const int nsub = (rows + off + BT - 1) / BT;
    for (int x = tid; x < PCH * TAPS; x += NT) {
        const int ch = x / TAPS;
        sm.cw[ch][x % TAPS] = cw[(size_t)((ch / DK) * H * DK + h * DK + ch % DK) * TAPS + x % TAPS];
    }
#ifdef KDA_PROF
    long long t0_ = clock64();
#endif

    for (int c = 0; c < nsub; ++c) {
        const int r0 = c * BT - off;                  // the prompt chunk's row at the sub-chunk's row 0

        // 0. stage this sub-chunk's rows (warmed in L2 during the previous one), warm the next one's and this
        // one's output gates
        stage(sm, tid, h, H, r0, rows, P, p_stride, b_off, A, a_stride, cs, vec_p, vec_a);
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads(); MARK(0);
        warm(tid, h, H, r0 + BT, c + 1 < nsub, rows, r0, P, p_stride, b_off, A, a_stride, G, g_stride);

        // 1. conv + SiLU (kda.cu's arithmetic) in 4-row blocks: warp w takes rows 4 w .. 4 w + 3 of q, then of k,
        // then of v, lane l the channels 4 l .. 4 l + 3 (kda.cu's norm lanes), and the L2 norms of its q and k rows
        // (kda.cu's sums); then the decays and beta.
#pragma unroll 1
        for (int part = 0; part < 3; ++part) {
            const int rb = 4 * warp, ch = part * DK + 4 * lane;
            float wt[4][TAPS];
#pragma unroll
            for (int x = 0; x < 4; ++x)
#pragma unroll
                for (int tap = 0; tap < TAPS; ++tap) wt[x][tap] = __bfloat162float(sm.cw[ch + x][tap]);
            float xs[4 + HALO][4];
#pragma unroll
            for (int j = 0; j < 4 + HALO; ++j) {
                const uint2 raw = *reinterpret_cast<const uint2*>(&sm.u.raw_p[rb + j][ch]);
                const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
                const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
                xs[j][0] = __low2float(lo); xs[j][1] = __high2float(lo);
                xs[j][2] = __low2float(hi); xs[j][3] = __high2float(hi);
            }
            float acc[4][4], act[4][4];
            bool slow = false;
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    acc[i][x] = 0.0f;
#pragma unroll
                    for (int tap = 0; tap < TAPS; ++tap) acc[i][x] = acc[i][x] + wt[x][tap] * xs[i + tap][x];
                    const float den = 1.0f + expf(-acc[i][x]);
                    act[i][x] = div_rn(acc[i][x], den);
                    slow |= !div_ok(acc[i][x], den);
                }
            if (slow) {                                       // kda.cu's IEEE division for the rare operands
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int x = 0; x < 4; ++x) act[i][x] = acc[i][x] / (1.0f + expf(-acc[i][x]));
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = r0 + rb + i;
#pragma unroll
                for (int x = 0; x < 4; ++x) act[i][x] = (r >= 0 && r < rows) ? bf(act[i][x]) : 0.0f;
            }
            if (part < 2) {                                   // warp-uniform
                float ss[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    ss[i] = 0.0f;
#pragma unroll
                    for (int x = 0; x < 4; ++x) ss[i] = ss[i] + act[i][x] * act[i][x];
                }
#pragma unroll
                for (int o = 16; o; o >>= 1)                  // warp_sum's tree, the rows interleaved
#pragma unroll
                    for (int i = 0; i < 4; ++i) ss[i] += __shfl_xor_sync(0xffffffffu, ss[i], o);
                float (*dst)[DK] = part ? sm.k : sm.q;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float inv = 1.0f / sqrtf(ss[i] + 1e-6f);
                    float y[4];
#pragma unroll
                    for (int x = 0; x < 4; ++x) {
                        y[x] = act[i][x] * inv;
                        if (part == 0) y[x] = y[x] * (1.0f / sqrtf((float)DK));
                    }
                    *reinterpret_cast<float4*>(&dst[rb + i][4 * lane]) = make_float4(y[0], y[1], y[2], y[3]);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    __nv_bfloat162 lo = __floats2bfloat162_rn(act[i][0], act[i][1]);
                    __nv_bfloat162 hi = __floats2bfloat162_rn(act[i][2], act[i][3]);
                    uint2 o;
                    o.x = *reinterpret_cast<unsigned*>(&lo); o.y = *reinterpret_cast<unsigned*>(&hi);
                    *reinterpret_cast<uint2*>(&sm.v[rb + i][4 * lane]) = o;
                }
            }
        }
        {
            const int cc = tid & (DK - 1);
#pragma unroll
            for (int j = tid >> 7; j < BT; j += NT / DK) {
                const int r = r0 + j;
                float gv = 1.0f;
                if (r >= 0 && r < rows) {
                    const float a = __bfloat162float(sm.raw_a[j][cc]) + dtb;
                    const float den = 1.0f + expf(-(rate * a));
                    const float sg = div_ok(1.0f, den) ? div_rn(1.0f, den) : 1.0f / den;   // kda.cu's sigmoidf_
                    gv = expf(lower * sg);
                }
                sm.g[j][cc] = gv;
            }
            if (tid < BT) {
                const int r = r0 + tid;
                sm.beta[tid] = (r >= 0 && r < rows) ? bf(sigmoidf_(sm.raw_b[tid])) : 0.0f;
            }
        }
        __syncthreads(); MARK(1);

        // 3. L and M. Group pr = tid / 16 takes column t = pr (rows s > pr), then t = 31 - pr (31 entries in all,
        // every group alike), in batches of 8; lane cg holds the key quads cg and cg + 16, k_t E(t -> s) multiplied
        // row after row; a batch's 16 sums reduce over the 16 lanes by a transposing tree.
        {
            const int pr = tid >> 4, cg = tid & 15;
            const int n1 = BT - 1 - pr;               // entries of column pr
            float kt[8], e[8];
            auto load_t = [&](int tt) {
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const float4 x = reinterpret_cast<const float4*>(sm.k[tt])[cg + 16 * u];
                    kt[4 * u] = x.x; kt[4 * u + 1] = x.y; kt[4 * u + 2] = x.z; kt[4 * u + 3] = x.w;
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) e[i] = kt[i];     // k_t E(t -> s), multiplied row after row
            };
            {   // the diagonal of M for both columns
                float dg[2];
#pragma unroll
                for (int hh = 0; hh < 2; ++hh) {
                    const int tt = hh ? BT - 1 - pr : pr;
                    load_t(tt);
                    float a = 0.0f;
#pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        const float4 x = reinterpret_cast<const float4*>(sm.q[tt])[cg + 16 * u];
                        a = fma_(x.x, kt[4 * u], a); a = fma_(x.y, kt[4 * u + 1], a);
                        a = fma_(x.z, kt[4 * u + 2], a); a = fma_(x.w, kt[4 * u + 3], a);
                    }
                    dg[hh] = a;
                }
#pragma unroll
                for (int o = 8; o; o >>= 1) {
                    dg[0] += __shfl_xor_sync(0xffffffffu, dg[0], o);
                    dg[1] += __shfl_xor_sync(0xffffffffu, dg[1], o);
                }
                if (cg == 0) {
                    sm.u.x.mt[pr][pr] = dg[0];
                    sm.u.x.mt[BT - 1 - pr][BT - 1 - pr] = dg[1];
                }
            }
            load_t(pr);
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                float val[16];
#pragma unroll
                for (int ii = 0; ii < 8; ++ii) {
                    const int i = 8 * b + ii;
                    if (i == n1 && n1 < BT - 1) load_t(BT - 1 - pr);
                    const int s2 = i < n1 ? pr + 1 + i : BT - pr + (i - n1);
                    float ak = 0.0f, aq = 0.0f;
                    if (i < BT - 1) {
#pragma unroll
                        for (int u = 0; u < 2; ++u) {
                            const float4 gx = reinterpret_cast<const float4*>(sm.g[s2])[cg + 16 * u];
                            const float4 kx = reinterpret_cast<const float4*>(sm.k[s2])[cg + 16 * u];
                            const float4 qx = reinterpret_cast<const float4*>(sm.q[s2])[cg + 16 * u];
                            const float gg[4] = {gx.x, gx.y, gx.z, gx.w}, kk[4] = {kx.x, kx.y, kx.z, kx.w};
                            const float qq[4] = {qx.x, qx.y, qx.z, qx.w};
#pragma unroll
                            for (int x = 0; x < 4; ++x) {
                                e[4 * u + x] = e[4 * u + x] * gg[x];
                                ak = fma_(kk[x], e[4 * u + x], ak);
                                aq = fma_(qq[x], e[4 * u + x], aq);
                            }
                        }
                    }
                    val[2 * ii] = ak;
                    val[2 * ii + 1] = aq;
                }
                const float sum = tree16(val, lane, 8, 4, 2, 1);        // entry cg / 2 of the batch, L (even) or M
                const int i = 8 * b + (cg >> 1);
                if (i < BT - 1) {
                    const int tt = i < n1 ? pr : BT - 1 - pr;
                    const int s2 = i < n1 ? pr + 1 + i : BT - pr + (i - n1);
                    if (cg & 1) {
                        sm.u.x.mt[tt][s2] = sum;
                        sm.u.x.mt[s2][tt] = 0.0f;                        // M[tt][s2], above the diagonal
                    } else {
                        sm.u.x.L[s2][tt] = sm.beta[s2] * sum;
                    }
                }
            }
        }
        __syncthreads(); MARK(3);

        // 4. T = (I + L)^-1 by warp 0: lane l inverts column l % 16 of diagonal block l / 16 by forward
        // substitution, then block (1, 0) = -T11 (L10 T00). Warps 1-7 per key channel: k E(s ->), then k E(s) in
        // place, and E(31) (threads 32-159); q E(s) in place (threads 160-255, then 160-191 for the last 32 channels)
        if (warp == 0) {
            const int blk = lane >> 4, j = lane & 15, base = 16 * blk;
            float col[16];
#pragma unroll
            for (int s2 = 0; s2 < 16; ++s2) {
                float acc = 0.0f;
#pragma unroll
                for (int m = 0; m < s2; ++m) acc = fma_(sm.u.x.L[base + s2][base + m], col[m], acc);
                col[s2] = s2 == j ? 1.0f : (s2 > j ? -acc : 0.0f);
            }
#pragma unroll
            for (int s2 = 0; s2 < 16; ++s2) sm.u.x.tt[base + j][base + s2] = col[s2];
            if (blk == 1) {                                    // block (0, 1), above the diagonal: zeros
#pragma unroll
                for (int s2 = 0; s2 < 16; ++s2) sm.u.x.tt[16 + j][s2] = 0.0f;
            }
            __syncwarp();
            if (blk == 0) {
                float xr[16];                                  // (L10 T00)[:, j]
#pragma unroll
                for (int s2 = 0; s2 < 16; ++s2) {
                    float acc = 0.0f;
#pragma unroll
                    for (int m = 0; m < 16; ++m) acc = fma_(sm.u.x.L[16 + s2][m], col[m], acc);
                    xr[s2] = acc;
                }
#pragma unroll
                for (int s2 = 0; s2 < 16; ++s2) {
                    float acc = 0.0f;
#pragma unroll
                    for (int m = 0; m <= s2; ++m)              // T11[s2][m]
                        acc = fma_(sm.u.x.tt[16 + m][16 + s2], xr[m], acc);
                    sm.u.x.tt[j][16 + s2] = -acc;
                }
            }
        } else {
            const int t2 = tid - 32;
            if (t2 < DK) {
                const int cc = t2;
                float rr = 1.0f;
#pragma unroll 8
                for (int j = BT - 1; j >= 0; --j) { sm.u.x.ke[j][cc] = sm.k[j][cc] * rr; rr = rr * sm.g[j][cc]; }
                float pp = 1.0f;
#pragma unroll 8
                for (int j = 0; j < BT; ++j) { pp = pp * sm.g[j][cc]; sm.k[j][cc] = sm.k[j][cc] * pp; }
                sm.egl[cc] = pp;
            } else {
                for (int cc = t2 - DK; cc < DK; cc += NT - 32 - DK) {
                    float pp = 1.0f;
#pragma unroll 8
                    for (int j = 0; j < BT; ++j) { pp = pp * sm.g[j][cc]; sm.q[j][cc] = sm.q[j][cc] * pp; }
                }
            }
        }
        __syncthreads(); MARK(4);

        // 5. Every thread: its sums of (k E) S0^T and (q E) S0^T; Y = beta (v - (k E) S0^T) into g, the (q E) S0^T
        // of the rows it finishes kept in o1.
        float o1[OWN];
        {
            float4 ck[NQ], cq[NQ], nk[NQ], nq[NQ];
            auto load_row = [&](float4 (&kx)[NQ], float4 (&qx)[NQ], int r) {
#pragma unroll
                for (int i = 0; i < NQ; ++i) {
                    kx[i] = reinterpret_cast<const float4*>(sm.k[r])[kp + KP * i];
                    qx[i] = reinterpret_cast<const float4*>(sm.q[r])[kp + KP * i];
                }
            };
            // row r's 8 sums (k sums at ok, q sums at 4 - ok), reduced: entry kp, value column 4 vg + kp % 4
            auto row_sums = [&](const float4 (&kx)[NQ], const float4 (&qx)[NQ], int ok) {
                float a[8];
#pragma unroll
                for (int x = 0; x < 8; ++x) a[x] = 0.0f;
                const int oq = 4 - ok;
#pragma unroll
                for (int i = 0; i < NQ; ++i)
#pragma unroll
                    for (int vv = 0; vv < VPT; ++vv) {
                        a[ok + vv] = fma_(kx[i].x, s[vv][i][0], a[ok + vv]);
                        a[ok + vv] = fma_(kx[i].y, s[vv][i][1], a[ok + vv]);
                        a[ok + vv] = fma_(kx[i].z, s[vv][i][2], a[ok + vv]);
                        a[ok + vv] = fma_(kx[i].w, s[vv][i][3], a[ok + vv]);
                        a[oq + vv] = fma_(qx[i].x, s[vv][i][0], a[oq + vv]);
                        a[oq + vv] = fma_(qx[i].y, s[vv][i][1], a[oq + vv]);
                        a[oq + vv] = fma_(qx[i].z, s[vv][i][2], a[oq + vv]);
                        a[oq + vv] = fma_(qx[i].w, s[vv][i][3], a[oq + vv]);
                    }
                return tree8(a, lane);
            };
            load_row(ck, cq, 0);
#pragma unroll 1
            for (int rg = 0; rg < BT / 8; ++rg) {             // 8 rows: pairs 2 rp (even: k sums first), 2 rp + 1
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int r = 8 * rg + 2 * u;
                    load_row(nk, nq, r + 1);
                    const float se = row_sums(ck, cq, 0);
                    load_row(ck, cq, r + 2 < BT ? r + 2 : r);        // (the last pair reloads a row: no predicate)
                    const float so = row_sums(nk, nq, 4);
                    // this thread keeps the q sum of its parity's row and writes Y of the other
                    const float qs = par ? so : se, ks = par ? se : so;
                    const int ry = par ? r : r + 1;
                    sm.g[ry][ov] = (__bfloat162float(sm.v[ry][ov]) - ks) * sm.beta[ry];
                    o1[OWN - 4 + u] = qs;
                }
                if (rg + 1 < BT / 8) {                        // after the loop o1[j] is row 2 j + par's
#pragma unroll
                    for (int j = 0; j + 4 < OWN; ++j) o1[j] = o1[j + 4];
                }
            }
        }
        __syncthreads(); MARK(5);

        // 6. load this sub-chunk's output gates (warm in L2); (q E) S0^T into k (free now); V' = T Y into q. Tiles:
        // warp w the rows 4 w' .. 4 w' + 3 with w' = w (w < 4) or 11 - w (the two warps of a scheduler, w and w + 4,
        // then run 36 steps of t together), lane l the value columns 4 l .. 4 l + 3 (T is zero above its diagonal:
        // t runs to the tile's last row)
        const int rq = 4 * (warp < 4 ? warp : 11 - warp), vq = 4 * lane;
        float gate[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = r0 + rq + i;
            const bool ok = r >= 0 && r < rows;
            const __nv_bfloat16* gp = G + (size_t)(ok ? r : 0) * g_stride + h * DV + vq;
            if (vec_g) {
                const uint2 x = ok ? *reinterpret_cast<const uint2*>(gp) : make_uint2(0, 0);
                const __nv_bfloat162 g0 = *reinterpret_cast<const __nv_bfloat162*>(&x.x);
                const __nv_bfloat162 g1 = *reinterpret_cast<const __nv_bfloat162*>(&x.y);
                gate[i][0] = __low2float(g0); gate[i][1] = __high2float(g0);
                gate[i][2] = __low2float(g1); gate[i][3] = __high2float(g1);
            } else {
#pragma unroll
                for (int x = 0; x < 4; ++x) gate[i][x] = ok ? ldb(gp + x) : 0.0f;
            }
        }
#pragma unroll
        for (int j = 0; j < OWN; ++j) sm.k[2 * j + par][ov] = o1[j];
        {
            float acc[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int x = 0; x < 4; ++x) acc[i][x] = 0.0f;
#pragma unroll 4
            for (int t = 0; t < rq + 4; ++t) {
                const float4 tq = *reinterpret_cast<const float4*>(&sm.u.x.tt[t][rq]);
                const float4 y = *reinterpret_cast<const float4*>(&sm.g[t][vq]);
                const float tv[4] = {tq.x, tq.y, tq.z, tq.w}, yv[4] = {y.x, y.y, y.z, y.w};
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int x = 0; x < 4; ++x) acc[i][x] = fma_(tv[i], yv[x], acc[i][x]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
                *reinterpret_cast<float4*>(&sm.q[rq + i][vq]) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        }
        __syncthreads(); MARK(6);

        // 7. read-outs O = (q E) S0^T + M V' of the same tile, bf16, the gated RMSNorm (kda.cu's formula; a warp
        // holds its rows' 128 values); the state update
        {
            float o[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 x = *reinterpret_cast<const float4*>(&sm.k[rq + i][vq]);
                o[i][0] = x.x; o[i][1] = x.y; o[i][2] = x.z; o[i][3] = x.w;
            }
#pragma unroll 4
            for (int t = 0; t < rq + 4; ++t) {
                const float4 mq = *reinterpret_cast<const float4*>(&sm.u.x.mt[t][rq]);
                const float4 vn = *reinterpret_cast<const float4*>(&sm.q[t][vq]);
                const float mv[4] = {mq.x, mq.y, mq.z, mq.w}, vv4[4] = {vn.x, vn.y, vn.z, vn.w};
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int x = 0; x < 4; ++x) o[i][x] = fma_(mv[i], vv4[x], o[i][x]);
            }
            float ss[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                ss[i] = 0.0f;
#pragma unroll
                for (int x = 0; x < 4; ++x) { o[i][x] = bf(o[i][x]); ss[i] = ss[i] + o[i][x] * o[i][x]; }
            }
#pragma unroll
            for (int m = 16; m; m >>= 1)
#pragma unroll
                for (int i = 0; i < 4; ++i) ss[i] += __shfl_xor_sync(0xffffffffu, ss[i], m);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = r0 + rq + i;
                if (r < 0 || r >= rows) continue;
                const float rinv = 1.0f / sqrtf(ss[i] / (float)DV + eps);
                __nv_bfloat16 ob[4];
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    const float yn = o[i][x] * rinv;
                    const float yw = nw[x] * yn;
                    const float den = 1.0f + expf(-gate[i][x]);
                    const float sg = div_ok(1.0f, den) ? div_rn(1.0f, den) : 1.0f / den;   // kda.cu's sigmoidf_
                    ob[x] = __float2bfloat16_rn(yw * sg);
                }
                __nv_bfloat16* op = out + (size_t)r * H * DV + h * DV + vq;
                if (vec_o) *reinterpret_cast<uint2*>(op) = *reinterpret_cast<const uint2*>(ob);
                else {
#pragma unroll
                    for (int x = 0; x < 4; ++x) op[x] = ob[x];
                }
            }
        }
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
            const float4 ge = reinterpret_cast<const float4*>(sm.egl)[kp + KP * i];
#pragma unroll
            for (int vv = 0; vv < VPT; ++vv) {
                s[vv][i][0] = s[vv][i][0] * ge.x; s[vv][i][1] = s[vv][i][1] * ge.y;
                s[vv][i][2] = s[vv][i][2] * ge.z; s[vv][i][3] = s[vv][i][3] * ge.w;
            }
        }
#pragma unroll 4
        for (int t = 0; t < BT; ++t) {
            const float4 vn = *reinterpret_cast<const float4*>(&sm.q[t][VPT * vg]);
            const float vs[4] = {vn.x, vn.y, vn.z, vn.w};
#pragma unroll
            for (int i = 0; i < NQ; ++i) {
                const float4 kx = reinterpret_cast<const float4*>(sm.u.x.ke[t])[kp + KP * i];
#pragma unroll
                for (int vv = 0; vv < VPT; ++vv) {
                    s[vv][i][0] = fma_(vs[vv], kx.x, s[vv][i][0]); s[vv][i][1] = fma_(vs[vv], kx.y, s[vv][i][1]);
                    s[vv][i][2] = fma_(vs[vv], kx.z, s[vv][i][2]); s[vv][i][3] = fma_(vs[vv], kx.w, s[vv][i][3]);
                }
            }
        }
        __syncthreads(); MARK(7);
    }
#pragma unroll
    for (int vv = 0; vv < VPT; ++vv)
#pragma unroll
        for (int i = 0; i < NQ; ++i)
            reinterpret_cast<float4*>(state_out + sbase + (size_t)(VPT * vg + vv) * DK)[kp + KP * i] =
                make_float4(s[vv][i][0], s[vv][i][1], s[vv][i][2], s[vv][i][3]);
}


}  // namespace tf_glm_kda_chunk

// kda_chunk_cuda launches chunk_kernel with this many dynamic shared bytes (sizeof(Smem)).
// zig/src/families/glm_cuda/cuda_kda.zig `chunk_smem` must stay equal to it.
static_assert(sizeof(tf_glm_kda_chunk::Smem) == 98560, "chunk_smem");
