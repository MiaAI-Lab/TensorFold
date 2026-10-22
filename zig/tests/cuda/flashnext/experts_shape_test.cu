// Decode plan D3 (work/research/R1-decode.md 3.3): the routed NVFP4 experts' launch shapes (zig/kernels/cuda/
// fn_nvfp4_shape.cu) against the engine's kernel (fn_nvfp4_experts.cu, <M, EPI, 4 warps, 2 stages>) on one GPU, no
// checkpoint: random NVFP4 blocks for 512 experts at Flash Next's TP=1 and TP=2 shapes, random routings (10 of 512 a
// row plus the shared slot the kernels skip, the plan's items of at most 16 pairs, and a skewed routing where most rows
// share a few experts), every row count the decode windows and shared rounds use. For each shape it checks that the
// output is byte-equal to the original kernel's (gate/up SwiGLU bf16, down fp32 and bf16) and times it: GPU time a call
// over many calls with fresh routings (so the experts' weights come from DRAM, as in a forward), launch grid as
// cuda_kernels.zig sizes it (ceil(max units / warps) capped at the resident blocks).
//
// Build (PyTorch image): nvcc -O3 -std=c++20 -arch=sm_121 -I zig/kernels/cuda -o experts_shape_test
//   zig/tests/cuda/flashnext/experts_shape_test.cu      Run: ./experts_shape_test [iters=200]
// Lines: PASS / FAIL / RESULT; exit 1 on any byte difference.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "fn_nvfp4_shape.cu"

#define CK(x)                                                                                 \
  do {                                                                                        \
    cudaError_t e_ = (x);                                                                     \
    if (e_ != cudaSuccess) {                                                                  \
      std::fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(2);                                                                           \
    }                                                                                         \
  } while (0)

using Kern = void (*)(const __nv_bfloat16*, int, int, const uint4*, const float*, int, int, const int*, const int*,
                      const int*, void*, int, float, int);

constexpr int E = 512, TOPK = 10, SLOTS = TOPK + 1, TILE = 16, BLOCK4 = 36;

struct Shape {
  const char* name;
  int warps, stages;
  Kern k[3];  // epilogue 2 (M 2), 0, 3 (M 1)
  int parts = 1;  // warps a unit's 32 columns take (expert_nt_kernel: 4 / NT)
};

template <int W, int D>
Shape shape(const char* name) {
  return {name, W, D,
          {tf_fn_nvfp4_shape::expert_kernel<2, 2, W, D>, tf_fn_nvfp4_shape::expert_kernel<1, 0, W, D>,
           tf_fn_nvfp4_shape::expert_kernel<1, 3, W, D>}};
}

template <int W, int D, int NT>
Shape nt_shape(const char* name) {
  Shape s = {name, W, D,
             {tf_fn_nvfp4_shape::expert_nt_kernel<2, 2, W, D, NT>, tf_fn_nvfp4_shape::expert_nt_kernel<1, 0, W, D, NT>,
              tf_fn_nvfp4_shape::expert_nt_kernel<1, 3, W, D, NT>}};
  s.parts = 4 / NT;
  return s;
}

struct Plan {
  std::vector<int> members, items, counts;
};

// cuda/experts.cu's plan in effect: pairs grouped by expert (ascending), each group cut into items of TILE pairs.
static Plan make_plan(const std::vector<int>& picks, int rows) {
  std::vector<std::vector<int>> by(E + 1);
  for (int p = 0; p < rows * SLOTS; ++p) by[picks[p]].push_back(p);
  Plan pl;
  for (int e = 0; e <= E; ++e)
    for (size_t i = 0; i < by[e].size(); i += TILE) {
      const int first = (int)pl.members.size();
      const int cnt = (int)std::min<size_t>(TILE, by[e].size() - i);
      for (int j = 0; j < cnt; ++j) pl.members.push_back(by[e][i + j]);
      pl.items.insert(pl.items.end(), {e, first, cnt});
    }
  pl.counts = {(int)(pl.items.size() / 3)};
  return pl;
}

static std::vector<int> route(std::mt19937_64& rng, int rows, bool skewed) {
  std::vector<int> picks(rows * SLOTS);
  for (int r = 0; r < rows; ++r) {
    std::vector<int> chosen;
    while ((int)chosen.size() < TOPK) {
      int e = (int)(rng() % E);
      if (skewed && (rng() % 4) != 0) e = (int)(rng() % 12);  // most picks among 12 hot experts
      if (std::find(chosen.begin(), chosen.end(), e) == chosen.end()) chosen.push_back(e);
    }
    for (int k = 0; k < TOPK; ++k) picks[r * SLOTS + k] = chosen[k];
    picks[r * SLOTS + TOPK] = E;  // the shared expert's slot (skip)
  }
  return picks;
}

static size_t max_items(size_t pairs) { return std::min<size_t>(pairs, E + 1) + pairs / TILE; }

struct TestGeo {
  const char* name;
  int dims, width;  // hidden 2560; expert width 640 (TP=1) or 320 (a TP=2 rank)
};

struct Dev {
  uint4 *up, *down;
  float *up_s, *down_s;
  __nv_bfloat16 *x, *act;
  void *out_ref, *out;
  int *members, *items, *counts;
};

static void fill_blocks(std::mt19937_64& rng, std::vector<uint4>& h) {
  // each block of BLOCK4 uint4: 32 code words (any nibbles), then 4 of e4m3 scales kept finite and moderate
  for (size_t i = 0; i < h.size(); ++i) {
    uint32_t* w = reinterpret_cast<uint32_t*>(&h[i]);
    const bool scales = (i % BLOCK4) >= 32;
    for (int c = 0; c < 4; ++c) {
      uint32_t v = (uint32_t)rng();
      if (scales) {
        uint32_t s = 0;
        for (int b = 0; b < 4; ++b) s |= (0x20u + ((v >> (8 * b)) & 0x1Fu)) << (8 * b);
        v = s;
      }
      w[c] = v;
    }
  }
}

static void fill_bf16(std::mt19937_64& rng, std::vector<__nv_bfloat16>& h, float scale) {
  std::normal_distribution<float> nd(0.f, scale);
  for (auto& v : h) v = __float2bfloat16(nd(rng));
}

static int occupancy(Kern k, int threads) {
  int n = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, k, threads, 0));
  return std::max(1, n);
}

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
  int sms = 0;
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  std::printf("INFO %s sm_%d%d, %d SMs, L2 %d MiB\n", prop.name, prop.major, prop.minor, sms, prop.l2CacheSize >> 20);

  const Shape orig = {"w4d2-orig", 4, 2,
                      {tf_fn_nvfp4_experts::nvfp4_expert_kernel<2, 2, 4>,
                       tf_fn_nvfp4_experts::nvfp4_expert_kernel<1, 0, 4>,
                       tf_fn_nvfp4_experts::nvfp4_expert_kernel<1, 3, 4>}};
  const std::vector<Shape> shapes = {
      shape<4, 2>("w4d2"), shape<4, 3>("w4d3"), shape<4, 4>("w4d4"), shape<4, 6>("w4d6"),
      shape<2, 2>("w2d2"), shape<2, 3>("w2d3"), shape<2, 4>("w2d4"), shape<2, 6>("w2d6"),
      shape<1, 2>("w1d2"), shape<1, 3>("w1d3"), shape<1, 4>("w1d4"), shape<1, 6>("w1d6"),
      nt_shape<1, 2, 2>("n2w1d2"), nt_shape<2, 2, 2>("n2w2d2"), nt_shape<4, 2, 2>("n2w4d2"), nt_shape<2, 4, 2>("n2w2d4"),
      nt_shape<1, 2, 1>("n1w1d2"), nt_shape<2, 2, 1>("n1w2d2"), nt_shape<4, 2, 1>("n1w4d2"), nt_shape<2, 4, 1>("n1w2d4"),
  };
  const TestGeo geos[] = {{"tp2", 2560, 320}, {"tp1", 2560, 640}};
  const int row_counts[] = {1, 2, 3, 4, 5, 7, 8, 12, 16, 24, 32, 64};
  const int n_plans = 64;
  std::mt19937_64 rng(0x5eedD3);
  int failures = 0;

  for (const TestGeo& g : geos) {
    const int D = g.dims, NI = g.width;
    const size_t up_n = (size_t)E * (NI / 32) * (D / 32) * 2 * BLOCK4;    // M 2: gate and up
    const size_t down_n = (size_t)E * (D / 32) * (NI / 32) * 1 * BLOCK4;  // M 1
    const int max_rows = 64;
    std::vector<uint4> h_up(up_n), h_down(down_n);
    fill_blocks(rng, h_up);
    fill_blocks(rng, h_down);
    std::vector<float> h_us(E * 2), h_ds(E);
    for (auto& v : h_us) v = 0.5f + (float)(rng() % 1000) / 1000.f;
    for (auto& v : h_ds) v = 0.5f + (float)(rng() % 1000) / 1000.f;
    std::vector<__nv_bfloat16> h_x((size_t)max_rows * D), h_act((size_t)max_rows * SLOTS * NI);
    fill_bf16(rng, h_x, 1.0f);
    fill_bf16(rng, h_act, 0.5f);
    Dev d{};
    CK(cudaMalloc(&d.up, up_n * 16));
    CK(cudaMalloc(&d.down, down_n * 16));
    CK(cudaMalloc(&d.up_s, h_us.size() * 4));
    CK(cudaMalloc(&d.down_s, h_ds.size() * 4));
    CK(cudaMalloc(&d.x, h_x.size() * 2));
    CK(cudaMalloc(&d.act, h_act.size() * 2));
    const size_t out_bytes = (size_t)max_rows * SLOTS * std::max(D, NI) * 4;
    CK(cudaMalloc(&d.out_ref, out_bytes));
    CK(cudaMalloc(&d.out, out_bytes));
    const size_t plan_ints = (size_t)max_rows * SLOTS * 4 + 4096;
    CK(cudaMalloc(&d.members, plan_ints * 4 * n_plans));
    CK(cudaMalloc(&d.items, plan_ints * 4 * n_plans));
    CK(cudaMalloc(&d.counts, 4 * n_plans));
    CK(cudaMemcpy(d.up, h_up.data(), up_n * 16, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.down, h_down.data(), down_n * 16, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.up_s, h_us.data(), h_us.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.down_s, h_ds.data(), h_ds.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.x, h_x.data(), h_x.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d.act, h_act.data(), h_act.size() * 2, cudaMemcpyHostToDevice));
    std::printf("INFO %s: dims %d width %d, gate/up %.1f MB, down %.1f MB (512 experts, one layer)\n", g.name, D, NI,
                up_n * 16 / 1e6, down_n * 16 / 1e6);

    for (const int R : row_counts) {
      for (int skewed = 0; skewed < (R >= 8 ? 2 : 1); ++skewed) {
        // n_plans routings of R rows each, uploaded side by side
        for (int p = 0; p < n_plans; ++p) {
          const Plan pl = make_plan(route(rng, R, skewed), R);
          CK(cudaMemcpy(d.members + p * plan_ints, pl.members.data(), pl.members.size() * 4, cudaMemcpyHostToDevice));
          CK(cudaMemcpy(d.items + p * plan_ints, pl.items.data(), pl.items.size() * 4, cudaMemcpyHostToDevice));
          CK(cudaMemcpy(d.counts + p, pl.counts.data(), 4, cudaMemcpyHostToDevice));
        }
        const size_t items_max = max_items((size_t)R * SLOTS);
        for (int which = 0; which < 3; ++which) {
          const bool gu = which == 0;
          const int NB = gu ? NI / 32 : D / 32, KG = gu ? D / 32 : NI / 32, N = gu ? NI : D;
          const uint4* Wt = gu ? d.up : d.down;
          const float* sc = gu ? d.up_s : d.down_s;
          const __nv_bfloat16* X = gu ? d.x : d.act;
          const int xs = gu ? D : NI, slots = gu ? SLOTS : 0;
          const size_t units = items_max * NB;
          const size_t bytes_out = (size_t)R * SLOTS * N * (which == 1 ? 4 : 2);
          float limit = 0.f;  // SwiGLU's clamp: Flash Next's experts have none; the exactness pass also runs 2.5
          auto launch = [&](const Shape& s, int p, void* out) {
            const Kern k = s.k[which];
            const size_t cap = (size_t)occupancy(k, 32 * s.warps) * sms;
            const int grid = (int)std::min<size_t>((units * s.parts + s.warps - 1) / s.warps, cap);
            k<<<grid, 32 * s.warps>>>(X, xs, slots, Wt, sc, KG, NB, d.items + p * plan_ints, d.counts + p,
                                      d.members + p * plan_ints, out, N, limit, E);
          };
          // exactness: every plan, the original then the shape, byte compare
          std::vector<unsigned char> a(bytes_out), b(bytes_out);
          for (const Shape& s : shapes) {
            size_t diff = 0;
            for (int p = 0; p < 2 * n_plans; ++p) {
              limit = p < n_plans ? 0.f : 2.5f;
              CK(cudaMemset(d.out_ref, 0xA5, bytes_out));
              CK(cudaMemset(d.out, 0xA5, bytes_out));
              launch(orig, p % n_plans, d.out_ref);
              launch(s, p % n_plans, d.out);
              CK(cudaDeviceSynchronize());
              CK(cudaMemcpy(a.data(), d.out_ref, bytes_out, cudaMemcpyDeviceToHost));
              CK(cudaMemcpy(b.data(), d.out, bytes_out, cudaMemcpyDeviceToHost));
              for (size_t i = 0; i < bytes_out; ++i) diff += a[i] != b[i];
            }
            limit = 0.f;
            if (diff) {
              std::printf("FAIL %s R=%d%s epi%d %s: %zu bytes differ from the original\n", g.name, R,
                          skewed ? " skewed" : "", which == 0 ? 2 : which == 1 ? 0 : 3, s.name, diff);
              ++failures;
            }
          }
          // timing: the original and each shape, iters calls over the plans, interleaved twice
          cudaEvent_t e0, e1;
          CK(cudaEventCreate(&e0));
          CK(cudaEventCreate(&e1));
          auto time_us = [&](const Shape& s) {
            for (int i = 0; i < 10; ++i) launch(s, i % n_plans, d.out);
            CK(cudaEventRecord(e0));
            for (int i = 0; i < iters; ++i) launch(s, i % n_plans, d.out);
            CK(cudaEventRecord(e1));
            CK(cudaEventSynchronize(e1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            return 1000.0 * ms / iters;
          };
          double t_orig = 0;
          std::vector<double> t(shapes.size(), 0.0);
          for (int rep = 0; rep < 2; ++rep) {
            t_orig += time_us(orig) / 2;
            for (size_t i = 0; i < shapes.size(); ++i) t[i] += time_us(shapes[i]) / 2;
          }
          size_t best = 0;
          for (size_t i = 1; i < shapes.size(); ++i)
            if (t[i] < t[best]) best = i;
          std::printf("RESULT %s R=%-2d%s %-7s orig %7.1f us |", g.name, R, skewed ? "s" : " ",
                      which == 0 ? "gateup" : which == 1 ? "down32" : "down16", t_orig);
          for (size_t i = 0; i < shapes.size(); ++i) std::printf(" %s %.1f", shapes[i].name, t[i]);
          std::printf(" | best %s %.2fx\n", shapes[best].name, t_orig / t[best]);
          CK(cudaEventDestroy(e0));
          CK(cudaEventDestroy(e1));
        }
      }
    }
    for (void* p : {(void*)d.up, (void*)d.down, (void*)d.up_s, (void*)d.down_s, (void*)d.x, (void*)d.act,
                    d.out_ref, d.out, (void*)d.members, (void*)d.items, (void*)d.counts})
      CK(cudaFree(p));
  }
  if (failures) {
    std::printf("FAIL %d shape/case combinations differ\n", failures);
    return 1;
  }
  std::printf("PASS every shape byte-equal to the original kernel on every case\n");
  return 0;
}
