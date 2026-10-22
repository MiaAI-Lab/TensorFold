// GLM-5.3-Flash vision tower, the pieces around the matrix products: patches, norms, rotary, SwiGLU, residuals, the
// merger's LayerNorm + GELU. Plain CUDA ABI like glm_hc.cu.
//
// Written from the Python reference (tensorfold/vision/glm.py GlmVisionTower) and from torch 2.13's CUDA semantics:
// every kernel here reproduces the bits of the torch op chain it replaces (fp32 intermediates, each torch op rounding
// where torch rounds, ATen's reduction orders). Nothing here is fused across a rounding torch performs.
//
// Compiled with nvcc defaults (contraction on, like torch's own kernels); where torch performs two ops in two
// kernels, the intrinsics __fmul_rn/__fadd_rn keep them apart.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

extern "C" {

__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 tobf(float v) { return __float2bfloat16_rn(v); }

// ---------------------------------------------------------------------------------------------------------------
// patches: uint8 frames -> the processors' patch matrix. frames [T, 3, H, W] (T even; frame_stride 0 for a picture,
// whose two frames are the same), pair0 = first pair of the pass, rows = pairs * gh * gw, 1176 values per row in
// (channel, time, y, x) order, rows in merge-block order. (x - mean * 255) / (std * 255), IEEE division.
// out_f32 may be null (the bf16 matrix is the conv's input: .to(bfloat16) of the fp32 value).
__global__ void vis_patches(const uint8_t* __restrict__ frames, long long frame_stride, int pair0, int pairs, int gh,
                            int gw, int H, int W, float m0, float m1, float m2, float s0, float s1, float s2,
                            __nv_bfloat16* __restrict__ out_bf, float* __restrict__ out_f32) {
    const long long total = (long long)pairs * gh * gw * 1176;
    const int hb = gh / 2, wb = gw / 2;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        const long long r = i / 1176;
        const int c = (int)(i - r * 1176);
        const int mw = (int)(r & 1), mh = (int)((r >> 1) & 1);
        long long blk = r >> 2;
        const int b = (int)(blk % wb);
        blk /= wb;
        const int a = (int)(blk % hb);
        const int p = (int)(blk / hb);
        const int px = c % 14, py = (c / 14) % 14, t = (c / 196) & 1, ch = c / 392;
        const int y = (2 * a + mh) * 14 + py, x = (2 * b + mw) * 14 + px;
        const uint8_t* f = frames + (long long)(2 * (pair0 + p) + t) * frame_stride;
        const float v = (float)f[((long long)ch * H + y) * W + x];
        const float mean = ch == 0 ? m0 : (ch == 1 ? m1 : m2);
        const float stdv = ch == 0 ? s0 : (ch == 1 ? s1 : s2);
        const float o = __fdiv_rn(__fsub_rn(v, mean), stdv);
        if (out_f32) out_f32[i] = o;
        if (out_bf) out_bf[i] = tobf(o);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The 2-D rotary tables of a gh x gw grid (_rope): cos/sin [S, 64] fp32, S = gh * gw, tokens in merge-block order.
// inv_freq = 1 / (10000 ** (arange(0, 32, 2) / 32)) as torch computes it on the device (powf, reciprocal, * 1.0),
// angle = float(position) * inv_freq, cosf / sinf.
__global__ void vis_rope_table(float* __restrict__ cs, float* __restrict__ sn, int gh, int gw, float theta) {
    const int S = gh * gw;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < S * 64; i += gridDim.x * blockDim.x) {
        const int s = i >> 6, d = i & 63, j = d & 31;
        const int blk = s >> 2, mh = (s >> 1) & 1, mw = s & 1;
        const int b = blk % (gw / 2), a = blk / (gw / 2);
        const int pos = j < 16 ? 2 * a + mh : 2 * b + mw;
        const int q = j & 15;
        const float e = __fdiv_rn((float)(2 * q), 32.0f);
        const float inv = __fmul_rn(__fdiv_rn(1.0f, powf(theta, e)), 1.0f);
        const float ang = __fmul_rn((float)pos, inv);
        cs[i] = cosf(ang);
        sn[i] = sinf(ang);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// _rms on [rows, 1024] bf16: w * (xf * rsqrt(xf.pow(2).mean(-1) + eps)).to(bf16), as torch computes it. The mean is
// ATen's reduce_kernel (vectorized input, 4 accumulators): a block of `bw` lanes per row (32 when there are 16 rows
// or more -- ATen's block shape follows the row count), lane l adds float4 vectors l, l + bw, ... into four
// accumulators, folds them ((a0 + a1) + a2) + a3, and the lanes fold with offsets bw/2 .. 1 (shared memory above 32).
// One block (bw, rows_per_block) handles rows_per_block rows.
__global__ void vis_rms1024(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                            __nv_bfloat16* __restrict__ out, int rows, float eps) {
    extern __shared__ float sh[];
    const int bw = blockDim.x;
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    const int l = threadIdx.x;
    const __nv_bfloat16* xr = x + (long long)row * 1024;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int v = l; v < 256; v += bw) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float f = bf(xr[4 * v + i]);
            acc[i] = __fadd_rn(acc[i], __fmul_rn(f, f));
        }
    }
    float s = __fadd_rn(__fadd_rn(__fadd_rn(acc[0], acc[1]), acc[2]), acc[3]);
    float* mine = sh + threadIdx.y * bw;
    if (bw > 32) {
        mine[l] = s;
        for (int off = bw / 2; off >= 32; off >>= 1) {
            __syncthreads();
            if (l < off) {
                s = __fadd_rn(s, mine[l + off]);
                mine[l] = s;
            }
        }
    }
    __syncthreads();
    // warp 0 of the row holds the partial sums in lanes 0..31 (l < 32); the others idle
    float tot = s;
    if (l < 32) {
        for (int off = 16; off > 0; off >>= 1) tot = __fadd_rn(tot, __shfl_down_sync(0xffffffffu, tot, off));
        if (l == 0) mine[0] = tot;
    }
    __syncthreads();
    const float mean = __fmul_rn(mine[0], 0.0009765625f);
    const float r = rsqrtf(__fadd_rn(mean, eps));
    for (int d = l; d < 1024; d += bw) {
        const float y = bf(tobf(__fmul_rn(bf(xr[d]), r)));
        out[(long long)row * 1024 + d] = tobf(__fmul_rn(bf(w[d]), y));
    }
}

// q/k norm + rotary: qkv [rows, 3072] bf16 (q at 0, k at 1024, v at 2048; head h at 64 h), one warp per (row, head, q or k).
// _rms over the 64 head values (ATen: lane l holds elements l and l + 32, added, then the shuffle fold 16 .. 1), then
// _rotate: (y * cos + rotate_half(y) * sin) with fp32 products and sum, rounded to bf16 once. out_q/out_k [rows, 1024].
__global__ void vis_qk_norm_rope(const __nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ qw,
                                 const __nv_bfloat16* __restrict__ kw, const float* __restrict__ cs,
                                 const float* __restrict__ sn, __nv_bfloat16* __restrict__ out_q,
                                 __nv_bfloat16* __restrict__ out_k, int rows, int S, float eps) {
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int l = threadIdx.x & 31;
    if (wid >= rows * 32) return;
    const int row = wid >> 5, hk = wid & 31, h = hk & 15, isk = hk >> 4;
    const __nv_bfloat16* src = qkv + (long long)row * 3072 + isk * 1024 + h * 64;
    const __nv_bfloat16* w = isk ? kw : qw;
    const float x0 = bf(src[l]), x1 = bf(src[l + 32]);
    float s = __fadd_rn(__fmul_rn(x0, x0), __fmul_rn(x1, x1));
    for (int off = 16; off > 0; off >>= 1) s = __fadd_rn(s, __shfl_down_sync(0xffffffffu, s, off));
    s = __shfl_sync(0xffffffffu, s, 0);
    const float mean = __fmul_rn(s, 0.015625f);
    const float r = rsqrtf(__fadd_rn(mean, eps));
    const float y0 = bf(tobf(__fmul_rn(bf(w[l]), bf(tobf(__fmul_rn(x0, r))))));
    const float y1 = bf(tobf(__fmul_rn(bf(w[l + 32]), bf(tobf(__fmul_rn(x1, r))))));
    const int p = row % S;
    const float* c = cs + (long long)p * 64;
    const float* n = sn + (long long)p * 64;
    // d = l: rotate_half gives -y[l + 32]; d = l + 32: y[l]
    const float o0 = __fadd_rn(__fmul_rn(y0, c[l]), __fmul_rn(-y1, n[l]));
    const float o1 = __fadd_rn(__fmul_rn(y1, c[l + 32]), __fmul_rn(y0, n[l + 32]));
    __nv_bfloat16* dst = (isk ? out_k : out_q) + (long long)row * 1024 + h * 64;
    dst[l] = tobf(o0);
    dst[l + 32] = tobf(o1);
}

// ---------------------------------------------------------------------------------------------------------------
// x + y in bf16 (torch: float add, one rounding)
__global__ void vis_add(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
                        __nv_bfloat16* __restrict__ out, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        out[i] = tobf(__fadd_rn(bf(a[i]), bf(b[i])));
}

// F.silu(gate.clamp(max=L)) * up.clamp(-L, L) on bf16: silu in fp32 rounded to bf16, then the product rounded again.
// `gate` and `up` are [rows, n] bf16 (row strides gs, us); out [rows, n].
__global__ void vis_swiglu(const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
                           __nv_bfloat16* __restrict__ out, long long n, float limit) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        float g = bf(gate[i]);
        float u = bf(up[i]);
        if (g > limit) g = limit;
        if (u > limit) u = limit;
        if (u < -limit) u = -limit;
        const float s = bf(tobf(g / (1.0f + expf(-g))));
        out[i] = tobf(__fmul_rn(s, u));
    }
}


// ---------------------------------------------------------------------------------------------------------------
// The merger's LayerNorm + GELU on [rows, 4096] bf16: torch's vectorized_layer_norm_kernel (4 bf16 per vector,
// Welford per thread over vectors thrx, thrx + numx, ..., the warp folded with shfl_down offsets 16 .. 1 through
// cuWelfordCombine, the BY warps of the block folded by halving through shared memory, variance = sigma2 / N,
// rstd = rsqrt(variance + eps), y = gamma * (rstd * (x - mean)) + beta), rounded to bf16; then F.gelu (exact erf) on
// that bf16 value. Launch: grid rows, block (32, BY). `ln_out` may be null; `out` receives the GELU output.
struct Wel { float mean, s2, cnt; };
__device__ __forceinline__ Wel wel_add(float v, Wel a, int variant) {
    const float delta = v - a.mean;
    const float nc = a.cnt + 1.f;
    const float nm = (variant & 4) ? a.mean + delta / nc : a.mean + delta * (1.f / nc);
    return {nm, a.s2 + delta * (v - nm), nc};
}
__device__ __forceinline__ Wel wel_comb(Wel a, Wel b, int variant) {
    const float delta = b.mean - a.mean;
    const float cnt = a.cnt + b.cnt;
    Wel r;
    if (cnt > 0.f) {
        const float coef = 1.f / cnt;
        const float nA = (variant & 8) ? a.cnt / cnt : a.cnt * coef;
        const float nB = (variant & 8) ? b.cnt / cnt : b.cnt * coef;
        r.mean = nA * a.mean + nB * b.mean;
        r.s2 = a.s2 + b.s2 + delta * delta * a.cnt * nB;
    } else {
        r.mean = 0.f;
        r.s2 = 0.f;
    }
    r.cnt = cnt;
    return r;
}
__global__ void vis_layernorm_gelu(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ gamma,
                                   const __nv_bfloat16* __restrict__ beta, __nv_bfloat16* __restrict__ ln_out,
                                   __nv_bfloat16* __restrict__ out, float eps, int variant,
                                   const unsigned short* __restrict__ gelu_lut) {
    constexpr int N = 4096, VEC = 4, NV = N / VEC;   // ATen: 128 threads, vectors of 4 (found against native_layer_norm's own mean)
    extern __shared__ float buf[];
    const int row = blockIdx.x;
    const __nv_bfloat16* xr = x + (long long)row * N;
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    Wel wd = {0.f, 0.f, 0.f};
    for (int i = thrx; i < NV; i += numx) {
        for (int ii = 0; ii < VEC; ++ii) wd = wel_add(bf(xr[i * VEC + ii]), wd, variant);
    }
    for (int off = 16; off > 0; off >>= 1) {
        Wel b = {__shfl_down_sync(0xffffffffu, wd.mean, off), __shfl_down_sync(0xffffffffu, wd.s2, off),
                 __shfl_down_sync(0xffffffffu, wd.cnt, off)};
        wd = wel_comb(wd, b, variant);
    }
    float* msbuf = buf;
    float* cbuf = buf + blockDim.y;
    __shared__ float stat[2];
    if (blockDim.y > 1) {
        for (int off = blockDim.y / 2; off > 0; off >>= 1) {
            if (threadIdx.x == 0 && threadIdx.y >= off && threadIdx.y < 2 * off) {
                const int w = threadIdx.y - off;
                msbuf[2 * w] = wd.mean;
                msbuf[2 * w + 1] = wd.s2;
                cbuf[w] = wd.cnt;
            }
            __syncthreads();
            if (threadIdx.x == 0 && threadIdx.y < off) {
                Wel b = {msbuf[2 * threadIdx.y], msbuf[2 * threadIdx.y + 1], cbuf[threadIdx.y]};
                wd = wel_comb(wd, b, variant);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            stat[0] = wd.mean;
            stat[1] = wd.s2 / N;
        }
        __syncthreads();
    } else {
        const float m = __shfl_sync(0xffffffffu, wd.mean, 0);
        const float v = __shfl_sync(0xffffffffu, wd.s2, 0) / N;
        if (threadIdx.x == 0) {
            stat[0] = m;
            stat[1] = v;
        }
        __syncthreads();
    }
    const float mean = stat[0];
    const float rstd = (variant & 2) ? 1.0f / sqrtf(stat[1] + eps) : rsqrtf(stat[1] + eps);
    for (int i = thrx; i < NV; i += numx) {
        for (int ii = 0; ii < VEC; ++ii) {
            const int c = i * VEC + ii;
            const float y = (variant & 1) ? __fadd_rn(__fmul_rn(bf(gamma[c]), __fmul_rn(rstd, __fsub_rn(bf(xr[c]), mean))), bf(beta[c])) : bf(gamma[c]) * (rstd * (bf(xr[c]) - mean)) + bf(beta[c]);
            const __nv_bfloat16 yb = tobf(y);
            if (ln_out) ln_out[(long long)row * N + c] = yb;
            const float xv = bf(yb);
            if (gelu_lut) out[(long long)row * N + c] = __ushort_as_bfloat16(gelu_lut[__bfloat16_as_ushort(yb)]);
            else out[(long long)row * N + c] = tobf(xv * 0.5f * (1.f + erff(xv * 0.7071067811865476f)));
        }
    }
}


// ---------------------------------------------------------------------------------------------------------------
// A product whose accumulation order is the tensor-core order of a cuDNN implicit GEMM: out[m][n] = sum over k of
// A[m][amap[k]] * W[n][wmap[k]] accumulated in fp32 by a chain of mma.m16n8k16 (bf16) over k = 0, 16, 32, ... (a map
// entry < 0 is a zero). Results are rounded to bf16 once; with `bias` the bf16 sum is rounded again after adding the
// bias in fp32 (torch adds a convolution's bias in a separate bf16 add). Block 128 threads = 2 x 2 warps, tile 64 x 64.
__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(128) vis_gemm_chain(const __nv_bfloat16* __restrict__ A, long long lda,
                                                      const __nv_bfloat16* __restrict__ W, long long ldw,
                                                      const int* __restrict__ amap, const int* __restrict__ wmap,
                                                      int kpad, const __nv_bfloat16* __restrict__ bias,
                                                      __nv_bfloat16* __restrict__ out, int M, int N, int fresh) {
    __shared__ __align__(16) __nv_bfloat16 sa[64][64 + 8];
    __shared__ __align__(16) __nv_bfloat16 sw[64][64 + 8];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp & 1, wn = warp >> 1;
    const int g = lane >> 2, t4 = lane & 3;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    float acc[2][4][4], mst[2][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = mst[i][j][e] = 0.f;
    const __nv_bfloat16 zero = __float2bfloat16_rn(0.f);
    for (int kc = 0; kc < kpad; kc += 64) {
        __syncthreads();
        for (int e = tid; e < 64 * 64; e += 128) {
            const int r = e >> 6, c = e & 63;
            const int k = kc + c;
            __nv_bfloat16 va = zero, vw = zero;
            if (k < kpad) {
                const int ia = amap ? amap[k] : k, iw = wmap ? wmap[k] : k;
                if (ia >= 0 && m0 + r < M) va = A[(long long)(m0 + r) * lda + ia];
                if (iw >= 0 && n0 + r < N) vw = W[(long long)(n0 + r) * ldw + iw];
            }
            sa[r][c] = va;
            sw[r][c] = vw;
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            if (kc + ks * 16 >= kpad) break;
            uint32_t af[2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r0 = wm * 32 + i * 16 + g;
                af[i][0] = *reinterpret_cast<const uint32_t*>(&sa[r0][ks * 16 + 2 * t4]);
                af[i][1] = *reinterpret_cast<const uint32_t*>(&sa[r0 + 8][ks * 16 + 2 * t4]);
                af[i][2] = *reinterpret_cast<const uint32_t*>(&sa[r0][ks * 16 + 8 + 2 * t4]);
                af[i][3] = *reinterpret_cast<const uint32_t*>(&sa[r0 + 8][ks * 16 + 8 + 2 * t4]);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int nr = wn * 32 + j * 8 + g;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sw[nr][ks * 16 + 2 * t4]);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sw[nr][ks * 16 + 8 + 2 * t4]);
#pragma unroll
                for (int i = 0; i < 2; ++i) mma16816(acc[i][j], af[i], b0, b1);
            }
            // `fresh` k elements per partial sum: the partial is added to the running sum in fp32 and restarted
            if (fresh > 0 && (((kc + ks * 16 + 16) % fresh) == 0)) {
#pragma unroll
                for (int i = 0; i < 2; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j)
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            mst[i][j][e] = __fadd_rn(mst[i][j][e], acc[i][j][e]);
                            acc[i][j][e] = 0.f;
                        }
            }
        }
    }
    if (fresh > 0) {
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][j][e] = __fadd_rn(mst[i][j][e], acc[i][j][e]);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int m = m0 + wm * 32 + i * 16 + g + (e >> 1) * 8;
                const int n = n0 + wn * 32 + j * 8 + 2 * t4 + (e & 1);
                if (m < M && n < N) {
                    float v = bf(tobf(acc[i][j][e]));
                    if (bias) v = __fadd_rn(v, bf(bias[n]));
                    out[(long long)m * N + n] = tobf(v);
                }
            }
}


// ---------------------------------------------------------------------------------------------------------------
// Attention for one image (or frame pair) at a time, with the arithmetic of PyTorch's flash-attention-2 forward
// kernel for head_dim 64, bf16, 128 x 128 tiles (pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64, 128, 128,
// 4 warps>>), which is what F.scaled_dot_product_attention runs for these shapes:
//   * S = Q K^T by a chain of 4 mma.m16n8k16 (head dim 0..63 in steps of 16) into a zeroed fp32 accumulator;
//   * key blocks of 128 visited from the last (the only one with masked columns) to the first; columns >= S are -inf;
//   * online softmax in base 2: row max over the thread's columns then across the quad of lanes (xor 2, xor 1),
//     p = ex2(fma(s, scale*log2e, -max*scale*log2e)), the row sum a per-thread running sum in column order (the quad
//     is folded once at the end), rescale of the sum and of O by ex2((max_prev - max) * scale*log2e) between blocks;
//   * P rounded to bf16 and multiplied into V by a chain of mma.m16n8k16 over the block's keys in order;
//   * O scaled by 1 / sum (rcp.approx when `fast_rcp`, else IEEE) and rounded to bf16.
// Rows of a tile are independent, so a CTA here is 64 rows (4 warps x 16) of one (group, head). q/k/o [rows, 1024]
// (head h at 64 h), v [rows, vstride] (pointer at head 0 of row 0).
enum { V_RCP = 1, V_EXP_NOFMA = 2, V_EXP2F = 4, V_QK_REV = 8, V_PV_REV = 16, V_SUM_NOFMA = 32 };
__device__ __forceinline__ float ex2f(float x) {
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
    __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&p);
}

__global__ void __launch_bounds__(128) vis_flash(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                                                const __nv_bfloat16* __restrict__ v, long long vstride,
                                                __nv_bfloat16* __restrict__ o, int S, float scale_log2, int variant) {
    __shared__ __align__(16) __nv_bfloat16 ks[128][64 + 8];
    __shared__ __align__(16) __nv_bfloat16 vt[64][128 + 8];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
    const int h = blockIdx.y;
    const long long base = (long long)blockIdx.z * S;
    const int r0 = blockIdx.x * 64 + warp * 16;
    uint32_t qa[4][4];
#pragma unroll
    for (int ks_ = 0; ks_ < 4; ++ks_) {
        const int ra = r0 + g, rb = r0 + g + 8;
        const uint32_t* pa = reinterpret_cast<const uint32_t*>(q + (base + ra) * 1024 + h * 64 + ks_ * 16);
        const uint32_t* pb = reinterpret_cast<const uint32_t*>(q + (base + rb) * 1024 + h * 64 + ks_ * 16);
        qa[ks_][0] = ra < S ? pa[t4] : 0u;
        qa[ks_][1] = rb < S ? pb[t4] : 0u;
        qa[ks_][2] = ra < S ? pa[4 + t4] : 0u;
        qa[ks_][3] = rb < S ? pb[4 + t4] : 0u;
    }
    float acc_o[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc_o[j][e] = 0.f;
    float row_max[2] = {-INFINITY, -INFINITY}, row_sum[2] = {0.f, 0.f};
    const int nblk = (S + 127) / 128;
    for (int kb = nblk - 1; kb >= 0; --kb) {
        __syncthreads();
        {   // key `tid` of the block: K row to ks, V row transposed to vt (zero past S)
            const int key = kb * 128 + tid;
            uint4 kv[8], vv[8];
            if (key < S) {
                const uint4* kp = reinterpret_cast<const uint4*>(k + (base + key) * 1024 + h * 64);
                const uint4* vp = reinterpret_cast<const uint4*>(v + (base + key) * vstride + h * 64);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    kv[i] = kp[i];
                    vv[i] = vp[i];
                }
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) kv[i] = vv[i] = make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) *reinterpret_cast<uint4*>(&ks[tid][i * 8]) = kv[i];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&vv[i]);
#pragma unroll
                for (int j = 0; j < 8; ++j) vt[i * 8 + j][tid] = e[j];
            }
        }
        __syncthreads();
        float s[2][32];
        {
            float acc_s[16][4];
#pragma unroll
            for (int j = 0; j < 16; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc_s[j][e] = 0.f;
#pragma unroll
            for (int kk_ = 0; kk_ < 4; ++kk_) {
                const int ks_ = (variant & V_QK_REV) ? 3 - kk_ : kk_;
#pragma unroll
                for (int j = 0; j < 16; ++j) {
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&ks[8 * j + g][ks_ * 16 + 2 * t4]);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&ks[8 * j + g][ks_ * 16 + 8 + 2 * t4]);
                    uint32_t qr[4] = {qa[0][0], qa[0][1], qa[0][2], qa[0][3]};
                    switch (ks_) {
                        case 1: qr[0] = qa[1][0]; qr[1] = qa[1][1]; qr[2] = qa[1][2]; qr[3] = qa[1][3]; break;
                        case 2: qr[0] = qa[2][0]; qr[1] = qa[2][1]; qr[2] = qa[2][2]; qr[3] = qa[2][3]; break;
                        case 3: qr[0] = qa[3][0]; qr[1] = qa[3][1]; qr[2] = qa[3][2]; qr[3] = qa[3][3]; break;
                        default: break;
                    }
                    mma16816(acc_s[j], qr, b0, b1);
                }
            }
#pragma unroll
            for (int j = 0; j < 16; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int col = kb * 128 + 8 * j + 2 * t4 + e;
                    s[0][2 * j + e] = col < S ? acc_s[j][e] : -INFINITY;
                    s[1][2 * j + e] = col < S ? acc_s[j][2 + e] : -INFINITY;
                }
        }
        const bool first = kb == nblk - 1;
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            float m = first ? s[mi][0] : row_max[mi];
#pragma unroll
            for (int ni = first ? 1 : 0; ni < 32; ++ni) m = fmaxf(m, s[mi][ni]);
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
            float rescale = 1.f;
            if (!first) {
                const float scale = ex2f(__fmul_rn(__fsub_rn(row_max[mi], m), scale_log2));
                rescale = scale;
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    acc_o[j][2 * mi] = __fmul_rn(acc_o[j][2 * mi], scale);
                    acc_o[j][2 * mi + 1] = __fmul_rn(acc_o[j][2 * mi + 1], scale);
                }
            }
            row_max[mi] = m;
            const float ms = m == -INFINITY ? 0.f : __fmul_rn(m, scale_log2);
#pragma unroll
            for (int ni = 0; ni < 32; ++ni) {
                const float arg = (variant & V_EXP_NOFMA) ? __fsub_rn(__fmul_rn(s[mi][ni], scale_log2), ms) : __fmaf_rn(s[mi][ni], scale_log2, -ms);
                s[mi][ni] = (variant & V_EXP2F) ? exp2f(arg) : ex2f(arg);
            }
            // The sum is rescaled and extended in the same statement stream in flash-attention's source
            // (row_sum *= scale; ... row_sum + p0 + p1 ...), so the compiler fuses the rescale with the first add:
            // fma(row_sum, scale, p0) (V_SUM_NOFMA keeps them apart).
            float sum;
            if (first) sum = s[mi][0];
            else sum = (variant & V_SUM_NOFMA) ? __fadd_rn(__fmul_rn(row_sum[mi], rescale), s[mi][0]) : __fmaf_rn(row_sum[mi], rescale, s[mi][0]);
#pragma unroll
            for (int ni = 1; ni < 32; ++ni) sum = __fadd_rn(sum, s[mi][ni]);
            row_sum[mi] = sum;
        }
#pragma unroll
        for (int kk0 = 0; kk0 < 8; ++kk0) {
            const int kk = (variant & V_PV_REV) ? 7 - kk0 : kk0;
            uint32_t pa[4];
            pa[0] = pack2(s[0][4 * kk], s[0][4 * kk + 1]);
            pa[1] = pack2(s[1][4 * kk], s[1][4 * kk + 1]);
            pa[2] = pack2(s[0][4 * kk + 2], s[0][4 * kk + 3]);
            pa[3] = pack2(s[1][4 * kk + 2], s[1][4 * kk + 3]);
#pragma unroll
            for (int nj = 0; nj < 8; ++nj) {
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&vt[8 * nj + g][kk * 16 + 2 * t4]);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&vt[8 * nj + g][kk * 16 + 8 + 2 * t4]);
                mma16816(acc_o[nj], pa, b0, b1);
            }
        }
    }
#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        float sum = row_sum[mi];
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 2));
        sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, 1));
        float inv;
        if (sum == 0.f || sum != sum) inv = 1.f;
        else if (variant & V_RCP) asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inv) : "f"(sum));
        else inv = __fdiv_rn(1.f, sum);
        const int r = r0 + g + 8 * mi;
        if (r < S) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                __nv_bfloat162 pr = __floats2bfloat162_rn(__fmul_rn(acc_o[j][2 * mi], inv), __fmul_rn(acc_o[j][2 * mi + 1], inv));
                *reinterpret_cast<__nv_bfloat162*>(o + (base + r) * 1024 + h * 64 + 8 * j + 2 * t4) = pr;
            }
        }
    }
}

// A plain attention for the same shapes without the flash kernel's order: fp32 scores, an fp32 online softmax and an
// fp32-accumulated P V (P kept in fp32); one warp per (query row, head). Not the bit-reproducing kernel, a fallback.
__global__ void vis_attn_plain(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                               const __nv_bfloat16* __restrict__ v, long long vstride, __nv_bfloat16* __restrict__ o,
                               int S, int rows) {
    const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (wid >= rows * 16) return;
    const int row = wid >> 4, h = wid & 15;
    const long long base = (long long)(row / S) * S;
    const float q0 = bf(q[(long long)row * 1024 + h * 64 + lane]), q1 = bf(q[(long long)row * 1024 + h * 64 + lane + 32]);
    float mx = -INFINITY, sum = 0.f, a0 = 0.f, a1 = 0.f;
    for (int j = 0; j < S; ++j) {
        const __nv_bfloat16* kr = k + (base + j) * 1024 + h * 64;
        float d = q0 * bf(kr[lane]) + q1 * bf(kr[lane + 32]);
        for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
        d *= 0.125f;
        const float nm = fmaxf(mx, d);
        const float sc = expf(mx - nm), p = expf(d - nm);
        const __nv_bfloat16* vr = v + (base + j) * vstride + h * 64;
        sum = sum * sc + p;
        a0 = a0 * sc + p * bf(vr[lane]);
        a1 = a1 * sc + p * bf(vr[lane + 32]);
        mx = nm;
    }
    o[(long long)row * 1024 + h * 64 + lane] = tobf(a0 / sum);
    o[(long long)row * 1024 + h * 64 + lane + 32] = tobf(a1 / sum);
}


// F.gelu (exact, erf) on bf16, torch's formula: x * 0.5 * (1 + erf(x * sqrt(1/2))) in fp32, one rounding.
__global__ void vis_gelu(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, long long n,
                         const unsigned short* __restrict__ gelu_lut) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const float xv = bf(x[i]);
        if (gelu_lut) { out[i] = __ushort_as_bfloat16(gelu_lut[__bfloat16_as_ushort(x[i])]); continue; }
        out[i] = tobf(xv * 0.5f * (1.f + erff(xv * 0.7071067811865476f)));
    }
}


// Candidates for torch's bf16 GELU / SiLU, evaluated at every bf16 input (bit pattern i): mode 0 gelu in fp32, 1 gelu with
// the erf argument formed in double and erf in double (the constant M_SQRT1_2 is a double), 2 gelu fully in double,
// 3 silu in fp32, 4 silu in double. out[i] is the bf16 result.
__global__ void vis_act_candidates(__nv_bfloat16* __restrict__ out, int mode) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 65536) return;
    const unsigned short u = (unsigned short)i;
    const float x = __uint_as_float((unsigned)u << 16);
    float r;
    switch (mode) {
        case 0: r = x * 0.5f * (1.f + erff(x * 0.7071067811865476f)); break;
        case 1: r = (float)((double)x * 0.5 * (1.0 + erf((double)x * 0.70710678118654757))); break;
        case 2: r = (float)((double)x * 0.5f * (1.0 + erf((double)(x * 0.7071067811865476f)))); break;
        case 3: r = x / (1.f + expf(-x)); break;
        case 4: r = (float)((double)x / (1.0 + exp(-(double)x))); break;
        case 5: r = 0.5f * x * erfcf(-x * 0.7071067811865476f); break;
        case 6: r = x * normcdff(x); break;
        case 7: { const float h = 0.5f * x; r = fmaf(h, erff(x * 0.7071067811865476f), h); break; }
        case 8: r = __fmul_rn(__fmul_rn(x, 0.5f), __fadd_rn(1.f, erff(__fmul_rn(x, 0.7071067811865476f)))); break;
        case 9: r = (float)((double)(x * 0.5f) * (1.0 + (double)erff(x * 0.7071067811865476f))); break;
        default: r = 0.f; break;
    }
    out[i] = tobf(r);
}


// out[m][n] = sum_k A[m][amap[k]] * W[n][wmap[k]] accumulated one product at a time in fp32 in the order of the maps
// (acc = fma(a, w, acc); a bf16 product is exact in fp32, so this is RN(acc + a w)): the order cuDNN's patch-embedding
// kernel adds in. Results rounded to bf16; with `bias` a second rounding after the bf16 bias add. 64 x 64 tile, 256
// threads, 4 x 4 outputs each; K staged through shared memory in chunks of 24 map entries.
__global__ void __launch_bounds__(256) vis_gemm_seq(const __nv_bfloat16* __restrict__ A, long long lda,
                                                    const __nv_bfloat16* __restrict__ W, long long ldw,
                                                    const int* __restrict__ amap, const int* __restrict__ wmap, int klen,
                                                    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
                                                    int M, int N) {
    constexpr int KC = 24;
    __shared__ float sa[KC][64 + 1];
    __shared__ float sw[KC][64 + 1];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
    for (int kc = 0; kc < klen; kc += KC) {
        __syncthreads();
        for (int e = tid; e < KC * 64; e += 256) {
            const int kk = e / 64, r = e % 64;
            const int k = kc + kk;
            float va = 0.f, vw = 0.f;
            if (k < klen) {
                const int ia = amap ? amap[k] : k, iw = wmap ? wmap[k] : k;
                if (ia >= 0 && m0 + r < M) va = bf(A[(long long)(m0 + r) * lda + ia]);
                if (iw >= 0 && n0 + r < N) vw = bf(W[(long long)(n0 + r) * ldw + iw]);
            }
            sa[kk][r] = va;
            sw[kk][r] = vw;
        }
        __syncthreads();
        const int kn = min(KC, klen - kc);
        for (int kk = 0; kk < kn; ++kk) {
            float a[4], w[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = sa[kk][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; ++j) w[j] = sw[kk][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = __fmaf_rn(a[i], w[j], acc[i][j]);
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int m = m0 + ty + 16 * i, n = n0 + tx + 16 * j;
            if (m < M && n < N) {
                float v = bf(tobf(acc[i][j]));
                if (bias) v = __fadd_rn(v, bf(bias[n]));
                out[(long long)m * N + n] = tobf(v);
            }
        }
}


// y[r][c] = bf16(float(y[r][c]) + float(bias[c])) in place: torch adds a convolution's bias in a separate bf16 add.
__global__ void vis_bias_add(__nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ bias, int cols, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        y[i] = tobf(__fadd_rn(bf(y[i]), bf(bias[i % cols])));
}

// The downsample weight [4096 out][1024 c][2][2] (checkpoint layout) to channels-last [4096][2 kh][2 kw][1024 c], the
// layout PyTorch hands cuDNN when the input is channels-last.
__global__ void vis_down_weight_nhwc(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < 4096LL * 4096; i += (long long)gridDim.x * blockDim.x) {
        const int c = (int)(i & 1023), s = (int)((i >> 10) & 3), o = (int)(i >> 12);   // dst index = o*4096 + s*1024 + c, s = kh*2 + kw
        dst[i] = src[(long long)o * 4096 + c * 4 + s];
    }
}

}  // extern "C"
