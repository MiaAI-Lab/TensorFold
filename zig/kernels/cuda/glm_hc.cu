// GLM-5.3-Flash's hyper-connection mixing dots for decode rows (hc_pre's first half) with more blocks than
// glue._hc_partial, reproducing its bits: that Triton kernel (4 warps, a [32, 128] bf16 tile in the blocked layout
// sizePerThread [1, 8], threadsPerWarp [2, 16]) sums, for each of 8 steps of 128 inputs, a row's 8 products per
// thread in input order (products of bf16 values are exact in fp32, so fused or not), then across the 16 lanes of
// the row with butterflies 8, 4, 2, 1, and adds the steps to a zero accumulator in order; the sum of squares keeps
// one input per thread (4 warps x 32 lanes), adds the 8 steps' squares to zero in order, then butterflies 16 .. 1
// within each warp and (w0 + w2) + (w1 + w3) across the warps. PART [rows, 16 blocks, 32]: the 24 dots, then the sum.

#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda_bf16.h>

// Python source: glm5_next/cuda/hc.cu hc_partial_kernel.
// Ash Hart and TensorFold contributors; plain CUDA ABI follows jschmied PR #443.
extern "C" {

constexpr int WIDE = 16384, NB = 16, KB = WIDE / NB, SUB = 128, STEPS = KB / SUB, ROWS = 24;

__device__ __forceinline__ void unpack8(const uint4 v, float (&f)[8]) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i] = __uint_as_float(w[i] << 16);
        f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}

// Block (K block, part, row group): warps 0..7 the dots of mixing rows 8 part .. 8 part + 7 (their weights read once
// for up to RPB rows), warp 8 of part 0 the squares.
constexpr int RPB = 4;

__global__ void __launch_bounds__(288) hc_partial_kernel(const __nv_bfloat16* __restrict__ X,
                                                        const __nv_bfloat16* __restrict__ FN, float* __restrict__ part,
                                                        int rows) {
    const int b = blockIdx.x, p = blockIdx.y, r0 = blockIdx.z * RPB;
    const int r1 = min(r0 + RPB, rows);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (warp < 8) {
        const int m = p * 8 + warp;
        const int h = lane >> 4, l = lane & 15;
        const __nv_bfloat16* wr = FN + (size_t)m * WIDE + b * KB;
        float wf[4][8];
#pragma unroll
        for (int i = 0; i < 4; ++i)                    // steps 2 i + h, inputs 8 l .. 8 l + 7 of the step
            unpack8(__ldg(reinterpret_cast<const uint4*>(wr + (2 * i + h) * SUB + 8 * l)), wf[i]);
        for (int r = r0; r < r1; ++r) {
            const __nv_bfloat16* xr = X + (size_t)r * WIDE + b * KB;
            uint4 xv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) xv[i] = __ldg(reinterpret_cast<const uint4*>(xr + (2 * i + h) * SUB + 8 * l));
            float s[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float xf[8];
                unpack8(xv[i], xf);
                float a = __fmul_rn(wf[i][0], xf[0]);
#pragma unroll
                for (int j = 1; j < 8; ++j) a = __fmaf_rn(wf[i][j], xf[j], a);
#pragma unroll
                for (int o = 8; o; o >>= 1) a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, o));
                s[i] = a;
            }
            float acc = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc = __fadd_rn(acc, s[i]);                                      // step 2 i (half 0)
                acc = __fadd_rn(acc, __shfl_sync(0xffffffffu, s[i], 16));       // step 2 i + 1 (half 1)
            }
            if (lane == 0) part[((size_t)r * NB + b) * 32 + m] = acc;
        }
    } else if (p == 0) {
        for (int r = r0; r < r1; ++r) {
            const __nv_bfloat16* xr = X + (size_t)r * WIDE + b * KB;
            float w[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {              // the Triton kernel's warp j: input 32 j + lane of each step
                float ss = 0.f;
#pragma unroll
                for (int t = 0; t < STEPS; ++t) {
                    const float x = __bfloat162float(xr[t * SUB + 32 * j + lane]);
                    ss = __fmaf_rn(x, x, ss);
                }
#pragma unroll
                for (int o = 16; o; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
                w[j] = ss;
            }
            if (lane == 0) part[((size_t)r * NB + b) * 32 + ROWS] = __fadd_rn(__fadd_rn(w[0], w[2]), __fadd_rn(w[1], w[3]));
        }
    }
}

}  // extern "C"

