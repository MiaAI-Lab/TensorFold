// GPTQ int4 (symmetric: zero point 8, fp16 scale per group of GS inputs and output column) W4A16 matmuls for the
// INT4-AutoRound Flash Next checkpoint: the routed experts (grouped by the experts plan, gate/up with the SwiGLU
// epilogue, down) and the lm_head (dense rows). Ours (TensorFold Zig engine, no Python counterpart).
//
// Weights are exact in bf16 MMAs: a code q (0..15) becomes the bf16 pair (128 + q) - 136 = q - 8. Per group of GS
// inputs the products of one output column run in one fp32 MMA chain (k32 blocks in order, each block's two k16 steps
// in order, the k permutation below fixed), then acc = fmaf(group sum, scale, acc) in group order. A row's result
// therefore depends on its own inputs and the weights only: never on the other rows of the call, the row count, the
// items the plan cuts, the column tiles a warp takes (NT) or the grid.
//
// Packed layout (cuda_int4.zig packs it from the checkpoint's qweight [K/8, N] int32): words uint32
// [N/8][K/GS][32 lanes][GS/32]: lane l of an n8 tile holds column 8 tile + l/4 and, for each k32 block, the inputs
// 8 (l%4) .. 8 (l%4) + 7 of the block (GPTQ's word for those 8 inputs) with its nibbles reordered so that nibble i
// (i < 4) is input 2i and nibble i + 4 input 2i + 1: the pair at shift 4p is inputs (2p, 2p + 1). The k16 step s of
// the block takes pairs 2s (as b0) and 2s + 1 (as b1), against the lane's activations at the same inputs (a uint4 of
// 8 bf16 per row: a0/a1 = inputs 4s, 4s + 1, a2/a3 = 4s + 2, 4s + 3). Scales fp16 [N/8][K/GS][8].

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace tf_fn_int4 {

constexpr uint32_t LOW = 0x000F000Fu;
constexpr uint32_t K128 = 0x43004300u;   // the bf16 pair (128, 128)
constexpr uint32_t K136 = 0x43084308u;   // (136, 136): 128 + the zero point 8

__device__ __forceinline__ uint32_t pair(uint32_t w, int p) {
  uint32_t v = ((w >> (4 * p)) & LOW) | K128;
  const uint32_t k = K136;
  __nv_bfloat162 r = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&v), *reinterpret_cast<const __nv_bfloat162*>(&k));
  return *reinterpret_cast<uint32_t*>(&r);
}

__device__ __forceinline__ void mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ld_nc(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p));
  return r;
}

__device__ __forceinline__ uint2 ld_nc2(const uint2* p) {
  uint2 r;
  asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];\n" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}

__device__ __forceinline__ uint32_t comp(const uint4& v, int c) {
  return c == 0 ? v.x : c == 1 ? v.y : c == 2 ? v.z : v.w;
}

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// experts.cuh's SwiGLU (the NVFP4 experts' epilogue 2): gate and up rounded to bf16, silu rounded, times up.
__device__ __forceinline__ float swiglu(float g, float u, float limit) {
  float gv = bf(g), uv = bf(u);
  if (limit > 0.f) {
    gv = fminf(gv, limit);
    uv = fminf(fmaxf(uv, -limit), limit);
  }
  return bf(gv / (1.f + expf(-gv))) * uv;
}

// A lane's words of one n8 tile and group: GS/32 uint32 (uint4 for 128, uint2 for 64).
template <int GS>
struct Words {
  uint32_t v[GS / 32];
};

template <int GS>
__device__ __forceinline__ Words<GS> load_words(const uint32_t* p) {
  Words<GS> w;
  if constexpr (GS == 128) {
    const uint4 u = ld_nc(reinterpret_cast<const uint4*>(p));
    w.v[0] = u.x;
    w.v[1] = u.y;
    w.v[2] = u.z;
    w.v[3] = u.w;
  } else {
    static_assert(GS == 64, "groups of 64 or 128 inputs");
    const uint2 u = ld_nc2(reinterpret_cast<const uint2*>(p));
    w.v[0] = u.x;
    w.v[1] = u.y;
  }
  return w;
}

// One stage: a group's weights and scales of every (matrix, n8 tile) and the activations of rows gq and gq + 8 of
// each of the MT row tiles (16 rows each) a pass takes.
template <int GS, int NT, int MATS, int MT>
struct Stage {
  Words<GS> w[MATS][NT];
  uint32_t s[MATS][NT];               // half2: the scales of the lane's two columns
  uint4 xa[MT][GS / 32], xb[MT][GS / 32];  // rows gq, gq + 8: inputs 8 t .. 8 t + 7 of each k32 block
};

template <int GS, int NT, int MATS, int MT>
__device__ __forceinline__ void load_stage(Stage<GS, NT, MATS, MT>& st, const uint32_t* const (&w)[MATS],
                                           const __half* const (&s)[MATS], int KG, int g, int lane, int t,
                                           const __nv_bfloat16* const (&x0)[MT], const __nv_bfloat16* const (&x1)[MT],
                                           const bool (&v0)[MT], const bool (&v1)[MT]) {
#pragma unroll
  for (int m = 0; m < MATS; ++m)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const size_t tg = (size_t)j * KG + g;   // tile j (relative) and group g
      st.w[m][j] = load_words<GS>(w[m] + (tg * 32 + lane) * (GS / 32));
      st.s[m][j] = __ldg(reinterpret_cast<const unsigned int*>(s[m] + tg * 8 + 2 * t));
    }
  const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
  for (int r = 0; r < MT; ++r)
#pragma unroll
    for (int b = 0; b < GS / 32; ++b) {
      const int k0 = g * GS + b * 32 + 8 * t;
      st.xa[r][b] = v0[r] ? __ldg(reinterpret_cast<const uint4*>(x0[r] + k0)) : zero;
      st.xb[r][b] = v1[r] ? __ldg(reinterpret_cast<const uint4*>(x1[r] + k0)) : zero;
    }
}

template <int GS, int NT, int MATS, int MT>
__device__ __forceinline__ void compute_stage(float (&acc)[MT][MATS][NT][4], const Stage<GS, NT, MATS, MT>& st,
                                              const bool (&hi)[MT]) {
#pragma unroll
  for (int m = 0; m < MATS; ++m)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const __half2 h = *reinterpret_cast<const __half2*>(&st.s[m][j]);
      const float s0 = __low2float(h), s1 = __high2float(h);
#pragma unroll
      for (int r = 0; r < MT; ++r) {
        float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int b = 0; b < GS / 32; ++b) {
          const uint32_t word = st.w[m][j].v[b];
#pragma unroll
          for (int s = 0; s < 2; ++s)
            mma(d, comp(st.xa[r][b], 2 * s), comp(st.xb[r][b], 2 * s), comp(st.xa[r][b], 2 * s + 1),
                comp(st.xb[r][b], 2 * s + 1), pair(word, 2 * s), pair(word, 2 * s + 1));
        }
        float(&a)[4] = acc[r][m][j];
        a[0] = fmaf(d[0], s0, a[0]);
        a[1] = fmaf(d[1], s1, a[1]);
        if (hi[r]) {
          a[2] = fmaf(d[2], s0, a[2]);
          a[3] = fmaf(d[3], s1, a[3]);
        }
      }
    }
}

// EPI 2: SwiGLU of matrices 0 (gate) and 1 (up), bf16; 0: fp32; 3: bf16.
template <int NT, int MATS, int EPI>
__device__ __forceinline__ void store(const float (&acc)[MATS][NT][4], void* out, int out_stride, int col0, int pr0,
                                      int pr1, bool v0, bool v1, float limit) {
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    if (!(h ? v1 : v0)) continue;
    const size_t row = (size_t)(h ? pr1 : pr0) * out_stride;
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const int col = col0 + 8 * j;
      const float a0 = acc[0][j][2 * h], a1 = acc[0][j][2 * h + 1];
      if constexpr (EPI == 0) {
        *reinterpret_cast<float2*>(reinterpret_cast<float*>(out) + row + col) = make_float2(a0, a1);
      } else {
        float o0 = a0, o1 = a1;
        if constexpr (EPI == 2) {
          o0 = swiglu(a0, acc[MATS - 1][j][2 * h], limit);
          o1 = swiglu(a1, acc[MATS - 1][j][2 * h + 1], limit);
        }
        *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out) + row + col) =
            __floats2bfloat162_rn(o0, o1);
      }
    }
  }
}

// A warp a unit: (item, NT n8 tiles of every matrix). Items (expert, first member, count) from the experts plan, or,
// with `items` null, the dense rows [16 i, 16 i + 16) of `rows` (the head: one matrix, members the rows themselves).
// `slots`: a member p reads input row p / slots (gate/up from token rows), 0: row p (down from pair rows). A pass
// takes MT tiles of 16 rows: the weights a group are loaded once for them (prompt items of up to 64 pairs).
template <int GS, int NT, int MT, int MATS, int EPI, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    int4_kernel(const __nv_bfloat16* __restrict__ X, int x_stride, int slots, const uint32_t* __restrict__ W,
                const __half* __restrict__ S, int K, int N, const int* __restrict__ items,
                const int* __restrict__ counts, const int* __restrict__ members, int rows, void* __restrict__ out,
                int out_stride, float limit, int skip) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, t = lane & 3;
  const int KG = K / GS, NB = N / (8 * NT);
  const int n_items = items ? __ldg(counts) : (rows + 15) / 16;
  const int units = n_items * NB;
  const size_t mat_words = (size_t)N * K / 8, mat_scales = (size_t)N * KG;
  for (int unit = blockIdx.x * WARPS + (threadIdx.x >> 5); unit < units; unit += gridDim.x * WARPS) {
    const int it = unit / NB, cb = unit - it * NB;
    int e = 0, first = 16 * it, cnt = min(16, rows - 16 * it);
    if (items) {
      e = __ldg(items + 3 * it);
      first = __ldg(items + 3 * it + 1);
      cnt = __ldg(items + 3 * it + 2);
      if (e == skip) continue;
    }
    const uint32_t* w[MATS];
    const __half* s[MATS];
#pragma unroll
    for (int m = 0; m < MATS; ++m) {
      const size_t mi = (size_t)e * MATS + m;
      w[m] = W + mi * mat_words + (size_t)cb * NT * KG * 32 * (GS / 32);
      s[m] = S + mi * mat_scales + (size_t)cb * NT * KG * 8;
    }
    for (int r0 = 0; r0 < cnt; r0 += 16 * MT) {
      bool v0[MT], v1[MT];
      int pr0[MT], pr1[MT];
      const __nv_bfloat16* x0[MT];
      const __nv_bfloat16* x1[MT];
#pragma unroll
      for (int r = 0; r < MT; ++r) {
        const int a = r0 + 16 * r + gq;
        v0[r] = a < cnt;
        v1[r] = a + 8 < cnt;
        if (items) {
          pr0[r] = v0[r] ? __ldg(members + first + a) : 0;
          pr1[r] = v1[r] ? __ldg(members + first + a + 8) : 0;
        } else {
          pr0[r] = v0[r] ? first + a : 0;
          pr1[r] = v1[r] ? first + a + 8 : 0;
        }
        const int x0r = slots ? pr0[r] / slots : pr0[r], x1r = slots ? pr1[r] / slots : pr1[r];
        x0[r] = X + (size_t)x0r * x_stride;
        x1[r] = X + (size_t)x1r * x_stride;
      }
      float acc[MT][MATS][NT][4];
#pragma unroll
      for (int r = 0; r < MT; ++r)
#pragma unroll
        for (int m = 0; m < MATS; ++m)
#pragma unroll
          for (int j = 0; j < NT; ++j) acc[r][m][j][0] = acc[r][m][j][1] = acc[r][m][j][2] = acc[r][m][j][3] = 0.f;
      constexpr int D = 2;   // stages in flight
      Stage<GS, NT, MATS, MT> st[D];
#pragma unroll
      for (int d = 0; d < D; ++d)
        if (d < KG) load_stage<GS, NT, MATS, MT>(st[d], w, s, KG, d, lane, t, x0, x1, v0, v1);
      for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
          const int g = g0 + d;
          if (g < KG) {
            compute_stage<GS, NT, MATS, MT>(acc, st[d], v1);
            if (g + D < KG) load_stage<GS, NT, MATS, MT>(st[d], w, s, KG, g + D, lane, t, x0, x1, v0, v1);
          }
        }
      }
#pragma unroll
      for (int r = 0; r < MT; ++r)
        store<NT, MATS, EPI>(acc[r], out, out_stride, cb * NT * 8 + 2 * t, pr0[r], pr1[r], v0[r], v1[r], limit);
    }
  }
}

}  // namespace tf_fn_int4

// gate/up SwiGLU (2 matrices, groups of 128), down fp32 / bf16 (groups of 128 at one GPU, 64 at a TP=2 rank: its
// half of the 640 inputs cuts a group of 128), the head (bf16); 1, 2 or 4 n8 tiles a warp, 1 or 2 row tiles a pass
// (the same bits).
#define TF_INT4(GS, NT, MT, MATS, EPI) template __global__ void tf_fn_int4::int4_kernel<GS, NT, MT, MATS, EPI, 4>( \
    const __nv_bfloat16*, int, int, const uint32_t*, const __half*, int, int, const int*, const int*, const int*, int, \
    void*, int, float, int);
#define TF_INT4_NT(GS, MATS, EPI) TF_INT4(GS, 1, 1, MATS, EPI) TF_INT4(GS, 2, 1, MATS, EPI) TF_INT4(GS, 4, 1, MATS, EPI) \
    TF_INT4(GS, 1, 2, MATS, EPI) TF_INT4(GS, 2, 2, MATS, EPI) TF_INT4(GS, 4, 2, MATS, EPI)
TF_INT4_NT(128, 2, 2)
TF_INT4_NT(128, 1, 0)
TF_INT4_NT(128, 1, 3)
TF_INT4_NT(64, 1, 0)
TF_INT4_NT(64, 1, 3)
