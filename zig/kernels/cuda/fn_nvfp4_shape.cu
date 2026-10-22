// Flash Next's routed NVFP4 experts in other launch shapes (decode plan D3, work/research/R1-decode.md 3.3): the same
// per-unit arithmetic as fn_nvfp4_experts.cu (TensorFold's cuda/nvfp4/experts.cu by Ash Hart, copied SASS-equal by
// zig/tests/cuda/copies.py), with WARPS warps a block and D k-groups of loads in flight a warp instead of 4 and 2.
//
// A unit (one plan item's rows x 32 output columns) is one warp's work in both kernels, with no shared memory and no
// cross-warp exchange: the unit loop below is the original's, and it runs the original's load_stage, compute_stage
// (the k-groups in increasing order, each mma and fmaf as there), the fp32 scale and the epilogue. WARPS moves only
// which block a unit runs in; D moves only how far ahead load_stage issues, never the order of compute_stage calls.
// So every output element keeps its bits (checked byte for byte by zig/tests/cuda/flashnext/experts_shape_test.cu
// against the original on every row count and epilogue, and by the engine's gates).

#include "fn_nvfp4_experts.cu"

namespace tf_fn_nvfp4_shape {

using namespace tf_fn_nvfp4_experts;

// k_loop with D stages in flight: stage d % D holds k-group d until compute_stage consumes it, then takes d + D.
template <int M, int D>
__device__ __forceinline__ void k_loop_d(float (&acc)[M][1][NTW][4], const uint4* blk, int KG, int lane, int t,
                                         const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1) {
  Stage<M> st[D];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < KG) load_stage<M>(st[d], blk, d, lane, t, x0, x1, v0, v1);
  for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int g = g0 + d;
      if (g < KG) {
        compute_stage<M>(acc, st[d], v1);
        if (g + D < KG) load_stage<M>(st[d], blk, g + D, lane, t, x0, x1, v0, v1);
      }
    }
  }
}

// nvfp4_expert_kernel with the k_loop above: every other line is the original's.
template <int M, int EPI, int WARPS, int D>
__global__ void __launch_bounds__(WARPS * 32)
    expert_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                  const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                  const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                  float limit, int skip) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / NB, cb = unit - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK4);
    for (int r0 = 0; r0 < cnt; r0 += 16) {
      const bool v0 = r0 + gq < cnt, v1 = r0 + gq + 8 < cnt;
      const int pr0 = v0 ? __ldg(members + first + r0 + gq) : 0;
      const int pr1 = v1 ? __ldg(members + first + r0 + gq + 8) : 0;
      const int x0r = slots ? pr0 / slots : pr0, x1r = slots ? pr1 / slots : pr1;
      const __nv_bfloat16* x0 = X + (size_t)x0r * x_stride;
      const __nv_bfloat16* x1 = X + (size_t)x1r * x_stride;
      float acc[M][1][NTW][4];
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int j = 0; j < NTW; ++j) acc[m][0][j][0] = acc[m][0][j][1] = acc[m][0][j][2] = acc[m][0][j][3] = 0.f;
      k_loop_d<M, D>(acc, blk, KG, lane, t, x0, x1, v0, v1);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float g = __ldg(scale + e * M + m);
#pragma unroll
        for (int j = 0; j < NTW; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[m][0][j][q] *= g;
      }
      epilogue<EPI, M, 1>(acc, 0, out, N, cb * COLS + 2 * t, pr0, pr1, v0, v1, limit);
    }
  }
}

// Fewer n8 tiles a warp: a unit's 32 columns split into 4 / NT parts, each a warp's (units x 4 / NT warps). Every
// output element is still one n8 tile's mma over the same fragments, scaled and summed over the k-groups in the same
// order with the same fmaf, and the epilogue's per-element code is the original's: only which warp computes a tile
// changes. compute_stage and epilogue restricted to tiles [j0, j0 + NT):
template <int M, int NT>
__device__ __forceinline__ void compute_stage_nt(float (&acc)[M][1][NTW][4], const Stage<M>& st, bool hi, int j0) {
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t a0 = st.xa[h].x, a2 = st.xa[h].y, a1 = st.xb[h].x, a3 = st.xb[h].y;
#pragma unroll
      for (int jj = 0; jj < NT; ++jj) {
        const int j = j0 + jj;
        const uint32_t word = comp(st.w[m], j);
        float p[4] = {0.f, 0.f, 0.f, 0.f};
        mma(p, a0, a1, a2, a3, fp4pair(word, 8 * h), fp4pair(word, 8 * h + 4));
        const uint32_t sw = comp(st.s[m], 2 * h + (j >> 1));
        const int sh = (j & 1) * 16;
        const float s0 = e4m3f((sw >> sh) & 0xFFu), s1 = e4m3f((sw >> (sh + 8)) & 0xFFu);
        float(&a)[4] = acc[m][0][jj];
        a[0] = fmaf(p[0], s0, a[0]);
        a[1] = fmaf(p[1], s1, a[1]);
        if (hi) {
          a[2] = fmaf(p[2], s0, a[2]);
          a[3] = fmaf(p[3], s1, a[3]);
        }
      }
    }
}

template <int M, int D, int NT>
__device__ __forceinline__ void k_loop_nt(float (&acc)[M][1][NTW][4], const uint4* blk, int KG, int lane, int t,
                                          const __nv_bfloat16* x0, const __nv_bfloat16* x1, bool v0, bool v1, int j0) {
  Stage<M> st[D];
#pragma unroll
  for (int d = 0; d < D; ++d)
    if (d < KG) load_stage<M>(st[d], blk, d, lane, t, x0, x1, v0, v1);
  for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const int g = g0 + d;
      if (g < KG) {
        compute_stage_nt<M, NT>(acc, st[d], v1, j0);
        if (g + D < KG) load_stage<M>(st[d], blk, g + D, lane, t, x0, x1, v0, v1);
      }
    }
  }
}

// experts.cuh's epilogue for tiles [j0, j0 + NT) (acc slot jj holds tile j0 + jj), each element's code unchanged.
template <int EPI, int M, int NT>
__device__ __forceinline__ void epilogue_nt(const float (&acc)[M][1][NTW][4], void* out, int N, int col0, int pr0,
                                            int pr1, bool v0, bool v1, float limit, int j0) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!(h ? v1 : v0)) continue;
    const size_t row = (size_t)(h ? pr1 : pr0) * N;
#pragma unroll
    for (int jj = 0; jj < NT; ++jj) {
      const int col = col0 + 8 * (j0 + jj);
      const float a0 = acc[0][0][jj][2 * h], a1 = acc[0][0][jj][2 * h + 1];
      if constexpr (EPI == 0) {
        *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + row + col) = make_float2(a0, a1);
      } else {
        float o0, o1;
        if constexpr (EPI == 2) {
          o0 = swiglu(a0, acc[M - 1][0][jj][2 * h], limit);
          o1 = swiglu(a1, acc[M - 1][0][jj][2 * h + 1], limit);
        } else {
          static_assert(EPI == 3, "epilogues 0, 2, 3");
          o0 = a0;
          o1 = a1;
        }
        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + row + col) =
            __floats2bfloat162_rn(o0, o1);
      }
    }
  }
}

template <int M, int EPI, int WARPS, int D, int NT>
__global__ void __launch_bounds__(WARPS * 32)
    expert_nt_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                     const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                     const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                     float limit, int skip) {
  constexpr int PARTS = NTW / NT;
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int units = __ldg(counts) * NB * PARTS;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int whole = unit / PARTS, part = unit - whole * PARTS, j0 = part * NT;
    const int it = whole / NB, cb = whole - it * NB;
    const int e = __ldg(items + 3 * it), first = __ldg(items + 3 * it + 1), cnt = __ldg(items + 3 * it + 2);
    if (e == skip) continue;
    const uint4* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (M * BLOCK4);
    for (int r0 = 0; r0 < cnt; r0 += 16) {
      const bool v0 = r0 + gq < cnt, v1 = r0 + gq + 8 < cnt;
      const int pr0 = v0 ? __ldg(members + first + r0 + gq) : 0;
      const int pr1 = v1 ? __ldg(members + first + r0 + gq + 8) : 0;
      const int x0r = slots ? pr0 / slots : pr0, x1r = slots ? pr1 / slots : pr1;
      const __nv_bfloat16* x0 = X + (size_t)x0r * x_stride;
      const __nv_bfloat16* x1 = X + (size_t)x1r * x_stride;
      float acc[M][1][NTW][4];
#pragma unroll
      for (int m = 0; m < M; ++m)
#pragma unroll
        for (int j = 0; j < NTW; ++j) acc[m][0][j][0] = acc[m][0][j][1] = acc[m][0][j][2] = acc[m][0][j][3] = 0.f;
      k_loop_nt<M, D, NT>(acc, blk, KG, lane, t, x0, x1, v0, v1, j0);
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const float g = __ldg(scale + e * M + m);
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[m][0][j][q] *= g;
      }
      epilogue_nt<EPI, M, NT>(acc, out, N, cb * COLS + 2 * t, pr0, pr1, v0, v1, limit, j0);
    }
  }
}
}  // namespace tf_fn_nvfp4_shape

// The shape cuda_kernels.zig launches for small gate/up calls (at most 230 units: one row at TP=1, one or two at TP=2):
// one warp a block, two stages, one n8 tile a warp (zig/tests/cuda/flashnext/experts_shape_test.cu: TP=2 one row
// 80.8 -> 66.3 us, two rows 100.8 -> 93.4; TP=1 one row 103.1 -> 92.0). The test instantiates every other shape itself.
template __global__ void tf_fn_nvfp4_shape::expert_nt_kernel<2, 2, 1, 2, 1>(const __nv_bfloat16*, int, int,
    const uint4*, const float*, int, int, const int*, const int*, const int*, void*, int, float, int);
