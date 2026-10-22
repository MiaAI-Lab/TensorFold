// Flash Next (qwen4_exp) torch ops on the CUDA engine's hot path, with torch's rounding points kept one for one.
// TensorFold 0.6.5 sources: nvfp4_moe.MoE4.shared_act, decode.Engine.sample_draft and forward.candidates; the fills
// replace Tensor.fill_ on 64-bit device scalars (gdn_io's conv pointer). Compile with -O3 --fmad=false --ftz=false
// (torch_ops/README.md); tools/zig/check_flashnext_ops.py compares raw bytes with torch.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "launch_shape.h"

// MoE4.shared_act, then `buf.act[:rows, top_k] = ...`: g [rows, 2 ni] bf16 (gate columns, then up) ->
// bf16(f32(bf16(gate / (1 + exp(-gate)))) * up), row r at out + r * out_stride. torch runs neg (exact), exp, add,
// div, the bf16 round trip and mul as separate fp32 kernels; each is one IEEE operation here.
extern "C" __global__ void tf_fn_shared_swiglu_kernel(const __nv_bfloat16* g, __nv_bfloat16* out, uint64_t rows,
                                                      uint64_t ni, uint64_t out_stride) {
    const uint64_t count = rows * ni;
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) {
        const uint64_t r = i / ni;
        const uint64_t c = i - r * ni;
        const float gate = __bfloat162float(g[r * 2 * ni + c]);
        const float up = __bfloat162float(g[r * 2 * ni + ni + c]);
        const float e = expf(-gate);
        const float act = __fdiv_rn(gate, __fadd_rn(e, 1.0f));
        const float rounded = __bfloat162float(__float2bfloat16_rn(act));
        out[r * out_stride + c] = __float2bfloat16_rn(__fmul_rn(rounded, up));
    }
}

// Tensor.fill_ of `count` 64-bit words.
extern "C" __global__ void tf_fn_fill_u64_kernel(uint64_t* out, uint64_t value, uint64_t count) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += uint64_t(gridDim.x) * blockDim.x) out[i] = value;
}

// Engine.sample_draft, greedy: torch.cat([top, lse, col.float()]) of row = logits[:1].float(), where top, col =
// row.max(dim=-1) (the first maximum: `col` from the argmax of the same bf16 row, whose fp32 widening is exact).
extern "C" __global__ void tf_fn_draft_pick_kernel(const __nv_bfloat16* logits, const int32_t* col, const float* lse,
                                                   float* out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int32_t c = *col;
    out[0] = __bfloat162float(logits[c]);
    out[1] = *lse;
    out[2] = __ll2float_rn((long long)c);
}

// Engine.sample_draft, sampled: torch.cat([vals, lse, idx.float()], dim=1) of one row's top-k (k values, int64
// columns) and its fp32 log-sum-exp -> out [2k + 1].
extern "C" __global__ void tf_fn_draft_pack_kernel(const float* vals, const int64_t* idx, const float* lse, float* out,
                                                   uint64_t k) {
    for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i <= k;
         i += uint64_t(gridDim.x) * blockDim.x) {
        if (i == k) {
            out[k] = *lse;
            continue;
        }
        out[i] = vals[i];
        out[k + 1 + i] = __ll2float_rn((long long)idx[i]);
    }
}

// forward.candidates (two ranks): row r of c [rows, 2 cand + 1] = [vals | ids as int32 bits | lse], ids =
// (id_map[idx] if id_map else idx + offset).to(int32) (the low word, as torch narrows int64).
extern "C" __global__ void tf_fn_candidates_kernel(const float* vals, const int64_t* idx, const int64_t* id_map,
                                                   int64_t offset, const float* lse, float* out, uint64_t rows,
                                                   uint64_t cand) {
    const uint64_t r = blockIdx.x;
    if (r >= rows) return;
    float* row = out + r * (2 * cand + 1);
    for (uint64_t j = threadIdx.x; j <= cand; j += blockDim.x) {
        if (j == cand) {
            row[2 * cand] = lse[r];
            continue;
        }
        const int64_t i = idx[r * cand + j];
        const int64_t id = id_map ? id_map[i] : i + offset;
        row[j] = vals[r * cand + j];
        row[cand + j] = __int_as_float((int32_t)(uint32_t)(uint64_t)id);
    }
}

// sampling.nucleus_rows' fixed-point mass (top_k off): scaled = f64(f32(logit)) / t (torch: logits.float().double() /
// max(temperature, 1e-6)), mass = int64(floor(exp(scaled - top[r]) * 2^40)) (MASS), and each row's sum (an exact
// int64 sum in any order). torch runs the division, the subtraction, exp, the multiply and floor as separate fp64
// kernels: each is one IEEE operation here (exp is CUDA's fp64 exp, which torch's elementwise exp calls).
extern "C" __global__ void tf_fn_nucleus_mass_kernel(const __nv_bfloat16* logits, uint64_t ld, uint64_t cols, double t,
                                                     const double* top, int64_t* mass, unsigned long long* sums) {
    const uint64_t r = blockIdx.y;
    const uint64_t c = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= cols) return;
    const double scaled = __ddiv_rn((double)__bfloat162float(logits[r * ld + c]), t);
    const double e = exp(__dsub_rn(scaled, top[r]));
    const int64_t m = (int64_t)floor(__dmul_rn(e, 1099511627776.0));
    mass[r * cols + c] = m;
    atomicAdd(&sums[r], (unsigned long long)m);
}

// nvfp4.matmul with one K slice (SK 1: the shared expert's gate/up, K 2560) and bf16-pattern tables (PACKED 0), the
// same bits as Triton's `_fp4mm` there: for each 16-input block b in order, p = HMMA.16816.F32.BF16(x rows, the
// block's 16 x 8 weights, C = 0) and acc = fma(p, S[b, n], acc) (the cubins' HMMA with RZ, then FFMA), the bf16 store
// rounding to nearest. `_fp4mm` runs a 64-column tile a program walking all of K, so 10 programs cover N 640 and the
// chain waits on each block's loads; here a warp owns 16 rows x 8 columns, a CTA 16 x 32, and the blocks stream
// through a STAGES-deep cp.async ring. X [M, K] bf16 rows x_stride apart; W the 64-column tiles of 16 x 64 blocks
// (tile t at t * (K / 64) * 64 * 64 elements, block b at b * 16 * 64, row k, column n % 64); S [K / 16, N] fp32.
#define TF_FP4S_STAGES 16
__device__ __forceinline__ void tf_cp16(void* smem, const void* g, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    const int n = ok ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(g), "r"(n) : "memory");
}
extern "C" __global__ void __launch_bounds__(128) tf_fn_fp4_serial_kernel(
        const __nv_bfloat16* __restrict__ X, const uint16_t* __restrict__ W, const float* __restrict__ S, void* OUT,
        uint32_t M, uint32_t N, uint32_t K, uint32_t x_stride, uint32_t f32_out) {
    __shared__ __align__(16) uint16_t xs[TF_FP4S_STAGES][16][16];
    __shared__ __align__(16) uint16_t ws[TF_FP4S_STAGES][16][32];
    __shared__ __align__(16) float ss[TF_FP4S_STAGES][32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const uint32_t m0 = blockIdx.x * 16, n0 = blockIdx.y * 32;
    const uint32_t blocks = K / 16;
    const uint16_t* tile = W + (size_t)(n0 / 64) * ((K / 64) * 64 * 64) + (n0 % 64);
    // one stage: x 16 rows x 32 bytes (threads 0-31), w 16 rows x 64 bytes (32-95), s 128 bytes (96-103)
    auto load = [&](uint32_t b, int st) {
        if (tid < 32) {
            const uint32_t r = tid >> 1, h = tid & 1;
            const bool ok = m0 + r < M;
            const __nv_bfloat16* src = X + (size_t)(ok ? m0 + r : 0) * x_stride + b * 16 + h * 8;
            tf_cp16(&xs[st][r][h * 8], src, ok);
        } else if (tid < 96) {
            const uint32_t i = tid - 32, r = i >> 2, q = i & 3;
            tf_cp16(&ws[st][r][q * 8], tile + (size_t)b * 16 * 64 + r * 64 + q * 8, true);
        } else if (tid < 104) {
            const uint32_t q = tid - 96;
            tf_cp16(&ss[st][q * 4], S + (size_t)b * N + n0 + q * 4, true);
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    };
    for (int st = 0; st < TF_FP4S_STAGES - 1; ++st) {
        if ((uint32_t)st < blocks) load(st, st);
        else asm volatile("cp.async.commit_group;\n" ::: "memory");
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const int c = warp * 8 + g;  // this lane's B column within the CTA's 32
    for (uint32_t b = 0; b < blocks; ++b) {
        const int st = b % TF_FP4S_STAGES;
        asm volatile("cp.async.wait_group %0;\n" ::"n"(TF_FP4S_STAGES - 2) : "memory");
        __syncthreads();
        // the next block into the stage the previous step finished with
        const uint32_t nb = b + TF_FP4S_STAGES - 1;
        if (nb < blocks) load(nb, nb % TF_FP4S_STAGES);
        else asm volatile("cp.async.commit_group;\n" ::: "memory");
        // two bf16 a register, the lower k in the low half (no type punning of the uint16_t stage)
        const auto pair = [&](int r, int k) { return (uint32_t)xs[st][r][k] | ((uint32_t)xs[st][r][k + 1] << 16); };
        const uint32_t a0 = pair(g, 2 * t), a1 = pair(g + 8, 2 * t), a2 = pair(g, 2 * t + 8), a3 = pair(g + 8, 2 * t + 8);
        const uint32_t b0 = (uint32_t)ws[st][2 * t][c] | ((uint32_t)ws[st][2 * t + 1][c] << 16);
        const uint32_t b1 = (uint32_t)ws[st][2 * t + 8][c] | ((uint32_t)ws[st][2 * t + 9][c] << 16);
        float p0, p1, p2, p3;
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                     "{%8, %9}, {%10, %10, %10, %10};\n"
                     : "=f"(p0), "=f"(p1), "=f"(p2), "=f"(p3)
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(0.0f));
        const float s0 = ss[st][warp * 8 + 2 * t], s1 = ss[st][warp * 8 + 2 * t + 1];
        acc[0] = __fmaf_rn(p0, s0, acc[0]);
        acc[1] = __fmaf_rn(p1, s1, acc[1]);
        acc[2] = __fmaf_rn(p2, s0, acc[2]);
        acc[3] = __fmaf_rn(p3, s1, acc[3]);
    }
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    const uint32_t col = n0 + warp * 8 + 2 * t;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint32_t row = m0 + g + 8 * h;
        if (row >= M) continue;
        if (f32_out) {
            float* o = static_cast<float*>(OUT) + (size_t)row * N + col;
            o[0] = acc[2 * h];
            o[1] = acc[2 * h + 1];
        } else {
            __nv_bfloat16* o = static_cast<__nv_bfloat16*>(OUT) + (size_t)row * N + col;
            o[0] = __float2bfloat16_rn(acc[2 * h]);
            o[1] = __float2bfloat16_rn(acc[2 * h + 1]);
        }
    }
}

// C launchers for the development parity checks (tools/zig/check_flashnext_ops.py); Zig loads the fatbin.
extern "C" cudaError_t tf_fn_shared_swiglu(const void* g, void* out, uint64_t rows, uint64_t ni, uint64_t out_stride,
                                           cudaStream_t stream) {
    const uint64_t count = rows * ni;
    if (count == 0) return cudaSuccess;
    if (!g || !out || out_stride < ni) return cudaErrorInvalidValue;
    tf_fn_shared_swiglu_kernel<<<tf_launch_blocks(count, 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(g), static_cast<__nv_bfloat16*>(out), rows, ni, out_stride);
    return cudaGetLastError();
}

extern "C" cudaError_t tf_fn_fill_u64(void* out, uint64_t value, uint64_t count, cudaStream_t stream) {
    if (count == 0) return cudaSuccess;
    if (!out) return cudaErrorInvalidValue;
    tf_fn_fill_u64_kernel<<<tf_launch_blocks(count, 256), 256, 0, stream>>>(static_cast<uint64_t*>(out), value, count);
    return cudaGetLastError();
}

extern "C" cudaError_t tf_fn_draft_pick(const void* logits, const void* col, const void* lse, void* out,
                                        cudaStream_t stream) {
    if (!logits || !col || !lse || !out) return cudaErrorInvalidValue;
    tf_fn_draft_pick_kernel<<<1, 32, 0, stream>>>(static_cast<const __nv_bfloat16*>(logits),
        static_cast<const int32_t*>(col), static_cast<const float*>(lse), static_cast<float*>(out));
    return cudaGetLastError();
}

extern "C" cudaError_t tf_fn_draft_pack(const void* vals, const void* idx, const void* lse, void* out, uint64_t k,
                                        cudaStream_t stream) {
    if (!vals || !idx || !lse || !out) return cudaErrorInvalidValue;
    tf_fn_draft_pack_kernel<<<tf_launch_blocks(k + 1, 256), 256, 0, stream>>>(static_cast<const float*>(vals),
        static_cast<const int64_t*>(idx), static_cast<const float*>(lse), static_cast<float*>(out), k);
    return cudaGetLastError();
}

extern "C" cudaError_t tf_fn_candidates(const void* vals, const void* idx, const void* id_map, int64_t offset,
                                        const void* lse, void* out, uint64_t rows, uint64_t cand,
                                        cudaStream_t stream) {
    if (rows == 0) return cudaSuccess;
    if (!vals || !idx || !lse || !out || rows > 65535) return cudaErrorInvalidValue;
    tf_fn_candidates_kernel<<<uint32_t(rows), 64, 0, stream>>>(static_cast<const float*>(vals),
        static_cast<const int64_t*>(idx), static_cast<const int64_t*>(id_map), offset, static_cast<const float*>(lse),
        static_cast<float*>(out), rows, cand);
    return cudaGetLastError();
}

extern "C" cudaError_t tf_fn_nucleus_mass(const void* logits, uint64_t ld, uint64_t rows, uint64_t cols, double t,
                                          const void* top, void* mass, void* sums, cudaStream_t stream) {
    if (rows == 0 || cols == 0) return cudaSuccess;
    if (!logits || !top || !mass || !sums || rows > 65535) return cudaErrorInvalidValue;
    cudaError_t e = cudaMemsetAsync(sums, 0, rows * sizeof(unsigned long long), stream);
    if (e != cudaSuccess) return e;
    tf_fn_nucleus_mass_kernel<<<dim3(uint32_t((cols + 255) / 256), uint32_t(rows)), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits), ld, cols, t, static_cast<const double*>(top),
        static_cast<int64_t*>(mass), static_cast<unsigned long long*>(sums));
    return cudaGetLastError();
}
