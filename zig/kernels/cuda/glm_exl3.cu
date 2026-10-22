// Port of TensorFold families/glm5_next/cuda/exl3.cu; authored by Ash Hart and TensorFold contributors.
// Zig CUDA launch pattern authored by jschmied (TensorFold PR #443).
// GLM-5.3-Flash's EXL3 routed experts on CUDA: the grouped trellis GEMV and the Hadamard rotations around it.
//
// Format (see exl3.py, after ExLlamaV3, MIT, Copyright (c) 2025 Turboderp): a 16x16 tile is 32 little-endian
// 32-bit words; lane L of a warp decodes the tile's values 8L..8L+7 from words L-1 and L, and those eight values
// are exactly the B fragments of two mma.m16n8k16 (columns 0-7 and 8-15 of the tile), so a tile goes from memory
// to the tensor cores without a shuffle or a layout change.
//
// Every output depends only on its own row: the rows of a window share an mma tile but mma keeps rows
// independent, the K range of every warp and split is fixed by the shape, and warps and splits are summed in a
// fixed order (shared memory, then the epilogue kernels), never with atomics.

#include <cstdint>
#include <cuda_runtime.h>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace tf_glm_exl3 {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

// Two values of the "mcg" codebook from two 16-bit states, as a half2 (first state in .x).
__device__ __forceinline__ uint32_t mcg2(uint32_t s0, uint32_t s1) {
    uint32_t x0 = s0 * 0xCBAC1FEDu;
    uint32_t x1 = s1 * 0xCBAC1FEDu;
    x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    uint32_t lo = __byte_perm(x0, x1, 0x5410);
    uint32_t hi = __byte_perm(x0, x1, 0x7632);
    half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
    return *reinterpret_cast<uint32_t*>(&r);
}

// This lane's eight values of a 4-bit tile (word = tile[lane]) as the B fragments of its two n8 halves.
__device__ __forceinline__ void decode_tile(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    uint32_t s = __funnelshift_r(w, p, 20);
    b0[0] = mcg2((s >> 8) & 0xffffu, (s >> 4) & 0xffffu);
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2((w >> 12) & 0xffffu, (w >> 8) & 0xffffu);
    b1[1] = mcg2((w >> 4) & 0xffffu, w & 0xffffu);
}

// decode_tile's values with fewer integer instructions (the prompt kernels): the byte-aligned 16-bit states taken
// with one PRMT each and w >> 4 shared by two states. The same states, so the same bits.
__device__ __forceinline__ void decode_tile_p(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    const uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    const uint32_t s = __funnelshift_r(w, p, 20);
    const uint32_t t = w >> 4;
    b0[0] = mcg2(__byte_perm(s, 0u, 0x4421), (s >> 4) & 0xffffu);          // (s >> 8) & 0xffff, (s >> 4) & 0xffff
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2(__byte_perm(t, 0u, 0x4421), __byte_perm(w, 0u, 0x4421));   // (w >> 12) & 0xffff, (w >> 8) & 0xffff
    b1[1] = mcg2(__byte_perm(t, 0u, 0x4410), w & 0xffffu);                  // (w >> 4) & 0xffff, w & 0xffff
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t load_pair(const half* x, bool ok) {
    return ok ? *reinterpret_cast<const uint32_t*>(x) : 0u;
}

// 16 bytes of trellis words, read once a window (no L1 allocation); asm volatile keeps it where it stands relative
// to the (volatile) mma, so the ring depth is what the code says (patches/0580 of jayleaton/glm53-tensorfold-spark).
__device__ __forceinline__ uint4 ldg_nc_v4(const uint32_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// Grid (item, n block, mat * SK + split) -> Z[mat][split][pair][n]; warps sum fixed k ranges, added in warp order.
// SEQ (prompt chunks): grid (.., .., mat); one program runs the SK splits in turn and writes Z[mat][0][pair][n] =
// ((0 + split 0) + split 1) + ..., the epilogue's own sum, so an SK = 1 epilogue gives the same bits without the
// SK fp32 partials' round trip through memory.
template <int NT, int W, bool NFIRST, bool PF, bool SEQ>
__device__ __forceinline__ void grouped_body(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int E) {
    // NFIRST: the n blocks of an item are adjacent in launch order, so they read its rows while L2 holds them
    const int item = NFIRST ? blockIdx.y : blockIdx.x;
    const int nblock = NFIRST ? blockIdx.x : blockIdx.y;
    if (item >= counts[0]) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (e >= E) return;                                  // the shared expert (id E) is not EXL3
    const int mat = SEQ ? blockIdx.z : blockIdx.z / SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = mat ? T1 : T0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;

    const int per_split = KT / SK, per_warp = per_split / W;
    const int nt0 = nblock * NT;
    constexpr int OUT = 16 * NT * 16 / (W * 32);        // the sums a thread writes (SEQ keeps them in registers)
    float tot[OUT];
    __shared__ float red[W][16][NT * 16];
    for (int si = 0; si < (SEQ ? SK : 1); ++si) {
    const int split = SEQ ? si : blockIdx.z % SK;
    const int kt0 = split * per_split + warp * per_warp;
    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    if (PF) {
        // the next k step's words and A fragments load while this one decodes and multiplies (the same mma order)
        uint32_t words[NT], a[4];
#pragma unroll
        for (int i = 0; i < NT; ++i) words[i] = __ldg(tile + i * 32);
        {
            const int k = kt0 * 16;
            a[0] = load_pair(x0 + k, r0 >= 0); a[1] = load_pair(x1 + k, r1 >= 0);
            a[2] = load_pair(x0 + k + 8, r0 >= 0); a[3] = load_pair(x1 + k + 8, r1 >= 0);
        }
        for (int kt = kt0; kt < kt0 + per_warp; ++kt) {
            uint32_t nw[NT], na[4];
            const bool more = kt + 1 < kt0 + per_warp;
            if (more) {
                const uint32_t* next = tile + (size_t)NTILES * 32;
#pragma unroll
                for (int i = 0; i < NT; ++i) nw[i] = __ldg(next + i * 32);
                const int k = (kt + 1) * 16;
                na[0] = load_pair(x0 + k, r0 >= 0); na[1] = load_pair(x1 + k, r1 >= 0);
                na[2] = load_pair(x0 + k + 8, r0 >= 0); na[3] = load_pair(x1 + k + 8, r1 >= 0);
            }
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                uint32_t b0[2], b1[2];
                decode_tile(words[i], lane, b0, b1);
                mma16816(acc[i][0], a, b0);
                mma16816(acc[i][1], a, b1);
            }
            if (more) {
#pragma unroll
                for (int i = 0; i < NT; ++i) words[i] = nw[i];
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = na[c];
            }
            tile += (size_t)NTILES * 32;
        }
    } else {
        for (int kt = kt0; kt < kt0 + per_warp; ++kt) {
            uint32_t words[NT];
#pragma unroll
            for (int i = 0; i < NT; ++i) words[i] = __ldg(tile + i * 32);
            const int k = kt * 16;
            uint32_t a[4] = {load_pair(x0 + k, r0 >= 0), load_pair(x1 + k, r1 >= 0),
                             load_pair(x0 + k + 8, r0 >= 0), load_pair(x1 + k + 8, r1 >= 0)};
#pragma unroll
            for (int i = 0; i < NT; ++i) {
                uint32_t b0[2], b1[2];
                decode_tile(words[i], lane, b0, b1);
                mma16816(acc[i][0], a, b0);
                mma16816(acc[i][1], a, b1);
            }
            tile += (size_t)NTILES * 32;
        }
    }

    // warps' partial sums through shared memory, added in warp order
    if (SEQ && si > 0) __syncthreads();                // the previous split's sums are read
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < OUT; ++j) {
        const int idx = threadIdx.x + j * W * 32;
        const int row = idx / (NT * 16), col = idx % (NT * 16);
        const int r = rows_sh[row];
        float s = red[0][row][col];
#pragma unroll
        for (int w = 1; w < W; ++w) s += red[w][row][col];
        if (SEQ) {
            tot[j] = si == 0 ? 0.f + s : tot[j] + s;
        } else if (r >= 0) {
            Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
    }
    }  // splits
    if (SEQ) {
#pragma unroll
        for (int j = 0; j < OUT; ++j) {
            const int idx = threadIdx.x + j * W * 32;
            const int row = idx / (NT * 16), col = idx % (NT * 16);
            const int r = rows_sh[row];
            if (r >= 0) Z[((size_t)mat * P + r) * N + nt0 * 16 + col] = tot[j];
        }
    }
}

// Rows >= 64 leave the decode kernel (exl3_mm.py: R < PROMPT_ROWS). Python then launches grouped_kernel with the
// n blocks of an item adjacent (NFIRST), the next k step prefetched (PF) and, for gate/up, the K splits summed in
// one program (SEQ) so the epilogue reads SK = 1. Same chains and the same sum order as the expanded grouped kernel.
extern "C" __global__ void __launch_bounds__(4 * 32) glm_exl3_grouped_seq(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int E) {
    grouped_body<8, 4, true, true, true>(X0, X1, T0, T1, items, counts, members, Z, K, N, P, SK, E);
}

extern "C" __global__ void __launch_bounds__(4 * 32) glm_exl3_grouped_nf(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int E) {
    grouped_body<8, 4, true, true, false>(X0, X1, T0, T1, items, counts, members, Z, K, N, P, SK, E);
}

// Fast Walsh-Hadamard transform of 128 values held 4 per lane (lane L: values 4L..4L+3), natural order, fixed
// butterfly order: strides 1, 2 in registers, 4..64 across lanes.
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

// Program (member row, 128-block of K, matrix): Xh[mat][row][block] = fp16((x[row] * suh[mat][e]) @ H) for the
// gate and up projections of every routed slot (row, slot) with slot < slots - 1 (the last slot is the shared
// expert). One warp per program. ROWS (prompt chunks of a layer whose experts all share one suh for gate and up):
// program (row, block) writes Xh[row][block] from suh0 = that vector, the same arithmetic as every pair's input of
// that row, so each pair's fp16 input has the same bits, computed once a row instead of 2 x top_k times.
template <bool ROWS>
__global__ void rot_in_kernel(const __nv_bfloat16* __restrict__ x, int x_stride, const int* __restrict__ pick,
                              const half* __restrict__ suh0, const half* __restrict__ suh1, half* __restrict__ out0,
                              half* __restrict__ out1, int K, int slots) {
    const int p = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    int row, e;
    if (ROWS) {
        row = p;
        e = 0;
    } else {
        row = p / slots;
        const int slot = p % slots;
        if (slot == slots - 1) return;
        e = pick[row * slots + slot];
    }
    const int lane = threadIdx.x;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const __nv_bfloat16* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// Program (member row, 128-block of the rank's intermediate width): gate and up outputs summed over the splits in
// order, rotated, scaled by svh; GLM's limited SwiGLU with the grouped kernels' bf16 roundings; then the down
// projection's input rotation: Xd[row][block] = fp16((act * suh_d[e]) @ H).
__global__ void gateup_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                       const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                       const half* __restrict__ suh_d, half* __restrict__ xd, int P, int N, int SK,
                                       int slots, float limit) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int row = p / slots, slot = p % slots;
    if (slot == slots - 1) return;
    const int lane = threadIdx.x;
    const int e = pick[row * slots + slot];
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += Z[((size_t)(0 * SK + s) * P + p) * N + n + j];
            su += Z[((size_t)(1 * SK + s) * P + p) * N + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
        float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit), limit);
        float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}

// Program (member row, 128-block of the model width): the down projection's output summed over the splits in
// order, rotated and scaled by svh: Y[row][slot][:] (fp32, this rank's share of the expert's output).
__global__ void down_epilogue_kernel(const float* __restrict__ Z, const int* __restrict__ pick,
                                     const half* __restrict__ svh_d, float* __restrict__ y, int P, int D, int SK,
                                     int slots) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int row = p / slots, slot = p % slots;
    if (slot == slots - 1) return;
    const int lane = threadIdx.x;
    const int e = pick[row * slots + slot];
    const int n = blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < SK; ++k) s += Z[((size_t)k * P + p) * D + n + j];
        v[j] = s;
    }
    fwht128(v, lane);
    float* o = y + (size_t)p * D + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
}

// ---- decode windows: grouped_kernel's arithmetic with its epilogues fused -------------------------------------
// Every mma chain, warp sum and split sum is grouped_kernel's (non-SEQ, register prefetch of the next k step): a
// warp runs k tiles kt0 .. kt0 + STEPS - 1 of its split in order from zero, the warps are added in warp order, the
// splits in split order from 0. What changes:
// FUSE 1 (the down projection, SK = 1): the block's 128 columns are one Hadamard block, so it applies
//   down_epilogue_kernel's arithmetic to its own sums and writes Y, no Z round trip, no epilogue launch;
// FUSE 2 (gate and up): each block still writes its Z partial; the last of the 2 * SK blocks of an (item, n block)
//   to finish (an atomic count, reset for the next launch) runs gateup_epilogue_kernel's arithmetic on the item's
//   rows and 128 columns, reading the partials in split order: the same sums whichever block finishes last.
// XROW: X holds one row a token (rot_rows, a layer whose experts share one suh), pair p reads row p / slots.
// LD (TF_GLM_EXL3_LOADS; data movement only, every chain and sum as above, so the same bits):
//   0: 32-bit __ldg of word `lane` of each tile, one k step ahead, after the count -> item -> members round trips;
//   1: adapted from jayleaton/glm53-tensorfold-spark patches/0580 (exl3_ld.cu ld_kernel, LD_NC; Apache-2.0,
//      Copyright 2026 Jay Leaton). Changes: in our dec_kernel (its items / counts plan, fused epilogues and XROW
//      inputs kept), PD fixed at compile time, no probes / PDL. The count and the item are read together and the
//      first PD k steps' trellis words are issued before the member rows reach shared memory; the words come as
//      ld.global.nc.L1::no_allocate.v4 (lane l: bytes 16 l .. of each 512 B of a step, NT / 4 loads a lane), PD
//      steps in flight, through a 1 KB staging area in the warp's own slice of `red` (two __syncwarp a step), read
//      back as word `lane` of each tile.
template <int NT, int W, int STEPS, int FUSE, bool XROW, int LD = 0, int PD = 1, int SWZ = 0>
__global__ void __launch_bounds__(W * 32) dec_kernel(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int E, int slots,
    const half* __restrict__ sv0, const half* __restrict__ sv1, const half* __restrict__ su2, void* __restrict__ out,
    float limit, int* __restrict__ done) {
    static_assert(FUSE == 0 || NT * 16 == 128, "a fused epilogue needs one 128-column Hadamard block a program");
    static_assert(LD == 0 || (NT % 4 == 0 && PD >= 1 && STEPS % PD == 0), "LD 1: NT a multiple of 4, PD | STEPS");
    int item = blockIdx.x, nblock = blockIdx.y, zz = blockIdx.z;
    if constexpr (SWZ != 0) {                            // launch order only: the blocks of one k slice run together
        const unsigned lid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        nblock = (int)(lid % gridDim.y);
        const unsigned r = lid / gridDim.y;
        if (SWZ == 1) { item = (int)(r % gridDim.x); zz = (int)(r / gridDim.x); }
        else { zz = (int)(r % gridDim.z); item = (int)(r / gridDim.z); }
    }
    int e, first, cnt;
    if constexpr (LD == 0) {
        if (item >= counts[0]) return;
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    } else {                                             // one round trip: items holds a row for every block
        const int total = counts[0];
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
        if (item >= total) return;
    }
    if (e >= E) return;                                  // the shared expert (id E) is not EXL3
    const int mat = zz / SK, split = zz % SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = mat ? T1 : T0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    __shared__ __align__(16) float red[W][16][NT * 16];
    constexpr int NV = NT / 4;                           // LD 1: 16-byte loads a lane a k step
    uint4 rv[LD == 1 ? PD : 1][LD == 1 ? NV : 1];
    const uint32_t* tbase = T + (((size_t)e * KT + split * (KT / SK) + warp * STEPS) * NTILES + nblock * NT) * 32;
    const size_t kstep = (size_t)NTILES * 32;            // words from one k tile to the next
    if constexpr (LD == 1) {
#pragma unroll
        for (int d = 0; d < PD; ++d)
#pragma unroll
            for (int v = 0; v < NV; ++v) rv[d][v] = ldg_nc_v4(tbase + d * kstep + 128 * v + 4 * lane);
    }
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const int xr0 = r0 < 0 ? 0 : (XROW ? r0 / slots : r0), xr1 = r1 < 0 ? 0 : (XROW ? r1 / slots : r1);
    const half* x0 = X + (size_t)xr0 * K + 2 * t;
    const half* x1 = X + (size_t)xr1 * K + 2 * t;
    const int nt0 = nblock * NT;
    const int kt0 = split * (KT / SK) + warp * STEPS;
    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    if constexpr (LD == 1) {
        uint32_t ra[PD][4];
#pragma unroll
        for (int d = 0; d < PD; ++d) {
            const int k = (kt0 + d) * 16;
            ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
            ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
        }
        uint32_t* stage = reinterpret_cast<uint32_t*>(&red[warp][0][0]);
        for (int kb = 0; kb < STEPS; kb += PD) {
#pragma unroll
            for (int d = 0; d < PD; ++d) {
                const int s = kb + d;
                const bool more = s + PD < STEPS;
                uint32_t a[4], words[NT];
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = ra[d][c];
                __syncwarp();                            // every lane has read the previous step back
#pragma unroll
                for (int v = 0; v < NV; ++v) *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];
                if (more) {                              // the slot's registers are free again
#pragma unroll
                    for (int v = 0; v < NV; ++v)
                        rv[d][v] = ldg_nc_v4(tbase + (size_t)(s + PD) * kstep + 128 * v + 4 * lane);
                    const int k = (kt0 + s + PD) * 16;
                    ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
                    ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
                }
                __syncwarp();
#pragma unroll
                for (int i = 0; i < NT; ++i) words[i] = stage[i * 32 + lane];
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile(words[i], lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
        __syncwarp();                                    // the staging area becomes red[warp] below
    } else {
    // grouped_kernel's PF loop as it is (a runtime loop: unrolling it measured slower)
    uint32_t words[NT], a[4];
#pragma unroll
    for (int i = 0; i < NT; ++i) words[i] = __ldg(tile + i * 32);
    {
        const int k = kt0 * 16;
        a[0] = load_pair(x0 + k, r0 >= 0); a[1] = load_pair(x1 + k, r1 >= 0);
        a[2] = load_pair(x0 + k + 8, r0 >= 0); a[3] = load_pair(x1 + k + 8, r1 >= 0);
    }
    for (int kt = kt0; kt < kt0 + STEPS; ++kt) {
        uint32_t nw[NT], na[4];
        const bool more = kt + 1 < kt0 + STEPS;
        if (more) {
            const uint32_t* next = tile + (size_t)NTILES * 32;
#pragma unroll
            for (int i = 0; i < NT; ++i) nw[i] = __ldg(next + i * 32);
            const int k = (kt + 1) * 16;
            na[0] = load_pair(x0 + k, r0 >= 0); na[1] = load_pair(x1 + k, r1 >= 0);
            na[2] = load_pair(x0 + k + 8, r0 >= 0); na[3] = load_pair(x1 + k + 8, r1 >= 0);
        }
#pragma unroll
        for (int i = 0; i < NT; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile(words[i], lane, b0, b1);
            mma16816(acc[i][0], a, b0);
            mma16816(acc[i][1], a, b1);
        }
        if (more) {
#pragma unroll
            for (int i = 0; i < NT; ++i) words[i] = nw[i];
#pragma unroll
            for (int c = 0; c < 4; ++c) a[c] = na[c];
        }
        tile += (size_t)NTILES * 32;
    }
    }

#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();

    if constexpr (FUSE == 1) {
        // down_epilogue_kernel on this block's sums: v = 0 + (the warps' sums in warp order), rotated, times svh
        const half* svh_d = sv0;
        float* y = reinterpret_cast<float*>(out);
        for (int i = warp; i < cnt; i += W) {
            const int p = rows_sh[i];
            const int n = nt0 * 16 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = red[0][i][4 * lane + j];
#pragma unroll
                for (int w = 1; w < W; ++w) s += red[w][i][4 * lane + j];
                float z = 0.f;
                z += s;
                v[j] = z;
            }
            fwht128(v, lane);
            float* o = y + (size_t)p * N + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * N + n + j]);
        }
        return;
    } else {
        constexpr int OUT = 16 * NT * 16 / (W * 32);
#pragma unroll
        for (int j = 0; j < OUT; ++j) {
            const int idx = threadIdx.x + j * W * 32;
            const int row = idx / (NT * 16), col = idx % (NT * 16);
            const int r = rows_sh[row];
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < W; ++w) s += red[w][row][col];
            if (r >= 0) Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
        if constexpr (FUSE == 2) {
            __shared__ int last;
            __threadfence();                               // this block's partials before its count
            __syncthreads();
            if (threadIdx.x == 0) {
                int* c = done + (size_t)item * gridDim.y + nblock;
                last = atomicAdd(c, 1) == 2 * SK - 1;
                if (last) *c = 0;                          // the next launch starts from zero
            }
            __syncthreads();
            if (!last) return;
            __threadfence();                               // every block's partials are visible from here
            const half* svh_g = sv0;
            const half* svh_u = sv1;
            const half* suh_d = su2;
            half* xd = reinterpret_cast<half*>(out);
            for (int i = warp; i < cnt; i += W) {
                const int p = rows_sh[i];
                const int n = nt0 * 16 + 4 * lane;
                float gv[4], uv[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float sg = 0.f, su = 0.f;
                    for (int s = 0; s < SK; ++s) {
                        sg += __ldcg(Z + ((size_t)(0 * SK + s) * P + p) * N + n + j);
                        su += __ldcg(Z + ((size_t)(1 * SK + s) * P + p) * N + n + j);
                    }
                    gv[j] = sg;
                    uv[j] = su;
                }
                fwht128(gv, lane);
                fwht128(uv, lane);
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
                    float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])),
                                           -limit), limit);
                    float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                    v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
                }
                fwht128(v, lane);
                half* o = xd + (size_t)p * N + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
            }
        }
    }
}

// ---- prompt chunks: Y^T = W^T X^T --------------------------------------------------------------------------------
// A decoded trellis tile is, reordered, an m16n8k16 A fragment of W^T (lane (g, t): a0 = b0[0], a1 = b1[0],
// a2 = b0[1], a3 = b1[1]), so each warp decodes its own tiles straight into registers and every decoded fragment
// feeds the whole pass's member rows (up to 64 tokens, the prefill plan's item) as B fragments from shared memory.
// Each output is one fp32 chain of mma over ascending k tiles (no split K, no partials in memory), so a row's bits
// depend on its own values only (any chunking gives the same prompt state), though they are not the decode
// kernel's (which sums split and warp partials). The epilogues are gateup/down_epilogue_kernel's formulas.
constexpr int PBK = 32;              // k values a shared-memory stage (2 k tiles)
constexpr int PKT = PBK / 16;
constexpr int PSTAGES = 3;
constexpr int PXS = PBK + 8;         // stage row stride in halves (+16 B: ldmatrix rows on distinct banks)

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool ok) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

// A block covers 16 m tiles (gate/up: 8 of each matrix = one 128-column Hadamard block; down: 256 columns) for a
// pass of PM members with NW warps: each warp holds MT = 16 / NW m tiles for all PM members (PM / 8 n blocks), so a
// decoded tile feeds PM / 8 mma. PM 64 / 8 warps (two blocks an SM) or PM 128 / 16 warps (one): the same per-element
// chains, so the same bits. Every stage brings the members' next PBK k values and the 16 tiles' trellis words into
// shared memory (cp.async, PSTAGES deep). SHX (gate/up of a layer whose experts share one suh for both): gate and up
// read the same member rows (X0 = X1 = the per-row Xh, the members' rows from xrows), so a stage holds them once.
constexpr int PWORDS = 16 * PKT * 32;          // a stage's trellis words (16 m tiles x PKT k tiles x 32)

template <int MATS, int PM, int NW, bool SHX = false>
struct PromptCfg {
    static constexpr int MT = 16 / NW, NB = PM / 8, T = NW * 32;
    static constexpr int XM = SHX ? 1 : MATS;               // member-row copies a stage holds
    static constexpr size_t STAGE_HALVES = (size_t)XM * PM * PXS + 2 * PWORDS;
};

// Which (matrix, m tile) warp-tile q (0..15) of block cb is.
template <int MATS>
__device__ __forceinline__ void prompt_tile(int q, int cb, int& mat, int& nt) {
    mat = MATS == 2 ? q >> 3 : 0;
    nt = MATS == 2 ? cb * 8 + (q & 7) : cb * 16 + q;
}

template <int MATS, int PM, int NW, bool SHX = false>
__device__ __forceinline__ void prompt_mainloop(const half* __restrict__ X0, const half* __restrict__ X1,
                                                const uint32_t* __restrict__ T0, const uint32_t* __restrict__ T1,
                                                const int* rows_sh, int cnt, int e, int K, int NTILES, int cb,
                                                half* stage, float (&acc)[16 / NW][PM / 8][4]) {
    using C = PromptCfg<MATS, PM, NW, SHX>;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int mat, nt_unused;
    prompt_tile<MATS>(warp * C::MT, cb, mat, nt_unused);
    const int KT = K >> 4, chunks = K / PBK;
    const int nblocks = (cnt + 15) >> 4;                                            // 16-member blocks in use

#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int j = 0; j < C::NB; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = 0.f;

    auto load = [&](int chunk) {
        half* dst = stage + (size_t)(chunk % PSTAGES) * C::STAGE_HALVES;
        const int k0 = chunk * PBK;
        for (int idx = threadIdx.x; idx < C::XM * PM * (PBK / 8); idx += C::T) {
            const int m = idx / (PM * (PBK / 8)), rem = idx % (PM * (PBK / 8)), i = rem / (PBK / 8),
                      j = rem % (PBK / 8);
            const int r = rows_sh[i];
            const half* src = (m ? X1 : X0) + (size_t)(r < 0 ? 0 : r) * K + k0 + j * 8;
            cp_async16(dst + ((size_t)m * PM + i) * PXS + j * 8, src, r >= 0);
        }
        uint32_t* wdst = reinterpret_cast<uint32_t*>(dst + (size_t)C::XM * PM * PXS);
        for (int idx = threadIdx.x; idx < 16 * PKT * 8; idx += C::T) {       // 16 B pieces of 128-B tile words
            const int q = idx / (PKT * 8), kk = (idx / 8) % PKT, part = idx % 8;
            int wm, nt;
            prompt_tile<MATS>(q, cb, wm, nt);
            const uint32_t* src = (wm ? T1 : T0) + (((size_t)e * KT + chunk * PKT + kk) * NTILES + nt) * 32 + part * 4;
            cp_async16(wdst + ((size_t)q * PKT + kk) * 32 + part * 4, src, true);
        }
    };
#pragma unroll
    for (int c = 0; c < PSTAGES - 1; ++c) {
        if (c < chunks) load(c);
        cp_async_commit();
    }
    for (int c = 0; c < chunks; ++c) {
        cp_async_wait<PSTAGES - 2>();
        __syncthreads();
        if (c + PSTAGES - 1 < chunks) load(c + PSTAGES - 1);
        cp_async_commit();
        const half* base = stage + (size_t)(c % PSTAGES) * C::STAGE_HALVES;
        const half* xs = base + (size_t)(SHX ? 0 : mat) * PM * PXS;
        const uint32_t* ws = reinterpret_cast<const uint32_t*>(base + (size_t)C::XM * PM * PXS);
#pragma unroll
        for (int kk = 0; kk < PKT; ++kk) {
            uint32_t a[C::MT][4];
#pragma unroll
            for (int i = 0; i < C::MT; ++i) {
                uint32_t b0[2], b1[2];
                decode_tile_p(ws[((size_t)(warp * C::MT + i) * PKT + kk) * 32 + lane], lane, b0, b1);
                a[i][0] = b0[0]; a[i][1] = b1[0]; a[i][2] = b0[1]; a[i][3] = b1[1];
            }
#pragma unroll
            for (int jb = 0; jb < PM / 16; ++jb) {
                if (jb < nblocks) {
                    uint32_t r[4];
                    const int row = jb * 16 + (lane & 7) + ((lane >> 4) & 1) * 8;
                    ldsm_x4(r, xs + (size_t)row * PXS + kk * 16 + ((lane >> 3) & 1) * 8);
                    const uint32_t bl[2] = {r[0], r[1]}, bh[2] = {r[2], r[3]};
                    const bool upper = jb * 16 + 8 < cnt;         // members 8..15 of the block (else zero rows)
#pragma unroll
                    for (int i = 0; i < C::MT; ++i) {
                        mma16816(acc[i][2 * jb], a[i], bl);
                        if (upper) mma16816(acc[i][2 * jb + 1], a[i], bh);
                    }
                }
            }
        }
    }
    cp_async_wait<0>();
    __syncthreads();                                   // the stages become the epilogue's rows
}

// acc (this warp's m tiles, members PM / 2 * hf .. + PM / 2 - 1) -> E[member - PM / 2 * hf][col] fp32, row stride
// ES; col0: the first m tile's first column within the block's E columns.
template <int MT, int PM>
__device__ __forceinline__ void prompt_spill(float* E, int ES, int col0, const float (&acc)[MT][PM / 8][4], int hf) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int q = 0; q < PM / 16; ++q) {
            const int nb = hf * (PM / 16) + q;
            const int tok = q * 8 + 2 * t, col = col0 + i * 16 + g;
            E[(size_t)tok * ES + col] = acc[i][nb][0];
            E[(size_t)(tok + 1) * ES + col] = acc[i][nb][1];
            E[(size_t)tok * ES + col + 8] = acc[i][nb][2];
            E[(size_t)(tok + 1) * ES + col + 8] = acc[i][nb][3];
        }
}

constexpr int GU_ES = 128 + 4, DN_ES = 256 + 4;
template <int PM, int XM = 2>
constexpr size_t prompt_smem() {
    // max of: gate/up stages (XM copies of the member rows), down stages, gate/up epilogue half (2 x PM/2 x GU_ES),
    // down half (PM/2 x DN_ES)
    const size_t gu = (size_t)PSTAGES * (XM * PM * PXS + 2 * PWORDS) * 2;
    const size_t ep = (size_t)2 * (PM / 2) * GU_ES * 4;
    const size_t dn = (size_t)(PM / 2) * DN_ES * 4;
    return gu > ep ? (gu > dn ? gu : dn) : (ep > dn ? ep : dn);
}

// The launch order of a prompt plan's items (ORDER): the items of experts that take one item (their weights come
// from DRAM, few mma) and of experts that take several (their passes run together and share the weights through L2:
// mostly mma) merged evenly, each list in plan order, so DRAM and the tensor cores stay busy together instead of in
// turns. Only which block runs which item changes: every output keeps its bits. The shared expert's items (id E, the
// plan's last) are left out: order[slot] = item for slot < the routed items, -1 after. One block.
__global__ void __launch_bounds__(1024) prompt_order_kernel(const int* __restrict__ items,
                                                            const int* __restrict__ counts, int* __restrict__ order,
                                                            int cap, int E) {
    __shared__ int wsum[32];
    __shared__ int routed;
    const int all = min(counts[0], cap);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) routed = 0;
    __syncthreads();
    for (int i = tid; i < all; i += 1024)               // items are in expert order: the routed ones come first
        if (items[3 * i] < E && (i + 1 == all || items[3 * (i + 1)] >= E)) routed = i + 1;
    __syncthreads();
    const int n = routed;
    const int per = (n + 1023) / 1024;
    const int lo = min(n, tid * per), hi = min(n, lo + per);
    auto heavy = [&](int i) {
        const int e = items[3 * i];
        return (i > 0 && items[3 * (i - 1)] == e) || (i + 1 < n && items[3 * (i + 1)] == e);
    };
    int h = 0;
    for (int i = lo; i < hi; ++i) h += heavy(i);
    int v = h;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += u;
    }
    if (lane == 31) wsum[warp] = v;
    __syncthreads();
    if (warp == 0) {
        int w = wsum[lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += u;
        }
        wsum[lane] = w;
    }
    __syncthreads();
    const int nh = wsum[31], nl = n - nh;
    int rh = (warp ? wsum[warp - 1] : 0) + v - h;       // heavy items before lo
    for (int i = lo; i < hi; ++i) {
        if (heavy(i)) {                                // heavy rank r sits at (r + 1/2) / nh; lights before it: strictly
            const int r = rh++;
            const long long X = (2LL * r + 1) * nl;
            const int cl = X > nh ? (int)min((long long)nl, (X - nh + 2LL * nh - 1) / (2LL * nh)) : 0;
            order[r + cl] = i;
        } else {                                       // light rank l at (l + 1/2) / nl; heavies at or before it
            const int l = i - rh;
            const long long Y = (2LL * l + 1) * nh;
            const int ch = Y >= nl ? (int)min((long long)nh, (Y - nl) / (2LL * nl) + 1) : 0;
            order[l + ch] = i;
        }
    }
    for (int i = n + tid; i < all; i += 1024) order[i] = -1;
}

// SHX: Xg = Xu = Xh [rows, K], one row a token (pair p reads row p / slots); otherwise Xg, Xu [pairs, K].
template <int PM, int NW, bool SHX>
__global__ void __launch_bounds__(NW * 32, 16 / NW) prompt_gateup_kernel(
    const half* __restrict__ Xg, const half* __restrict__ Xu, const uint32_t* __restrict__ Tg,
    const uint32_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ svh_u,
    const half* __restrict__ suh_d, half* __restrict__ xd, int K, int N, int E, float limit, int slots,
    const int* __restrict__ order) {
    constexpr int MT = 16 / NW, PH = PM / 2;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __shared__ int rows_sh[PM];
    __shared__ int xrows_sh[SHX ? PM : 1];              // SHX: the members' token rows
    const int cb = blockIdx.x;
    if ((int)blockIdx.y >= counts[0]) return;
    const int item = order ? order[blockIdx.y] : blockIdx.y;
    if (item < 0) return;                                // ORDER leaves the shared expert's items out
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (e >= E) return;                                  // the shared expert
    if (threadIdx.x < PM) {
        const int p = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
        rows_sh[threadIdx.x] = p;
        if (SHX) xrows_sh[threadIdx.x] = p < 0 ? -1 : p / slots;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[MT][PM / 8][4];
    prompt_mainloop<2, PM, NW, SHX>(Xg, Xu, Tg, Tu, SHX ? xrows_sh : rows_sh, cnt, e, K, N >> 4, cb,
                                    reinterpret_cast<half*>(smem_raw), acc);
    float* Eg = reinterpret_cast<float*>(smem_raw);
    int mat, nt;
    prompt_tile<2>(warp * MT, cb, mat, nt);
    const int col0 = (nt - cb * 8) * 16;
    const int n = cb * 128 + 4 * lane;
    for (int hf = 0; hf < 2 && hf * PH < cnt; ++hf) {
    if (hf) __syncthreads();                             // the first half's rows are read
    prompt_spill<MT, PM>(Eg + (size_t)mat * PH * GU_ES, GU_ES, col0, acc, hf);
    __syncthreads();
    for (int i = warp; i < PH && hf * PH + i < cnt; i += NW) {
        const int p = rows_sh[hf * PH + i];
        float gv[4], uv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            gv[j] = Eg[(size_t)i * GU_ES + 4 * lane + j];
            uv[j] = Eg[(size_t)(PH + i) * GU_ES + 4 * lane + j];
        }
        fwht128(gv, lane);
        fwht128(uv, lane);
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
            float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit),
                             limit);
            float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
            v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
        }
        fwht128(v, lane);
        const half2 o01 = __halves2half2(__float2half_rn(v[0] * HAD_SCALE), __float2half_rn(v[1] * HAD_SCALE));
        const half2 o23 = __halves2half2(__float2half_rn(v[2] * HAD_SCALE), __float2half_rn(v[3] * HAD_SCALE));
        uint2 packed;
        packed.x = *reinterpret_cast<const uint32_t*>(&o01);
        packed.y = *reinterpret_cast<const uint32_t*>(&o23);
        *reinterpret_cast<uint2*>(xd + (size_t)p * N + n) = packed;
    }
    }  // halves
}

template <int PM, int NW>
__global__ void __launch_bounds__(NW * 32, 16 / NW) prompt_down_kernel(
    const half* __restrict__ Xd, const uint32_t* __restrict__ Td, const int* __restrict__ items,
    const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d,
    float* __restrict__ y, int K, int D, int E, const int* __restrict__ order) {
    constexpr int MT = 16 / NW, PH = PM / 2;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __shared__ int rows_sh[PM];
    const int cb = blockIdx.x;
    if ((int)blockIdx.y >= counts[0]) return;
    const int item = order ? order[blockIdx.y] : blockIdx.y;
    if (item < 0) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (e >= E) return;
    if (threadIdx.x < PM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[MT][PM / 8][4];
    prompt_mainloop<1, PM, NW>(Xd, Xd, Td, Td, rows_sh, cnt, e, K, D >> 4, cb, reinterpret_cast<half*>(smem_raw),
                               acc);
    float* Ed = reinterpret_cast<float*>(smem_raw);
    for (int hf = 0; hf < 2 && hf * PH < cnt; ++hf) {
    if (hf) __syncthreads();
    prompt_spill<MT, PM>(Ed, DN_ES, warp * MT * 16, acc, hf);
    __syncthreads();
    for (int i = warp; i < PH && hf * PH + i < cnt; i += NW) {
        const int p = rows_sh[hf * PH + i];
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
            const int n = cb * 256 + blk * 128 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = Ed[(size_t)i * DN_ES + blk * 128 + 4 * lane + j];
            fwht128(v, lane);
            const uint2 sv = *reinterpret_cast<const uint2*>(svh_d + (size_t)e * D + n);
            const half2 s01 = *reinterpret_cast<const half2*>(&sv.x), s23 = *reinterpret_cast<const half2*>(&sv.y);
            *reinterpret_cast<float4*>(y + (size_t)p * D + n) =
                make_float4(v[0] * HAD_SCALE * __low2float(s01), v[1] * HAD_SCALE * __high2float(s01),
                            v[2] * HAD_SCALE * __low2float(s23), v[3] * HAD_SCALE * __high2float(s23));
        }
    }
    }  // halves
}


extern "C" __global__ void glm_exl3_rot_0(const __nv_bfloat16* __restrict__ x, int x_stride, const int* __restrict__ pick,
                              const half* __restrict__ suh0, const half* __restrict__ suh1, half* __restrict__ out0,
                              half* __restrict__ out1, int K, int slots) {
    const int p = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    int row, e;
    if (0) {
        row = p;
        e = 0;
    } else {
        row = p / slots;
        const int slot = p % slots;
        if (slot == slots - 1) return;
        e = pick[row * slots + slot];
    }
    const int lane = threadIdx.x;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const __nv_bfloat16* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}
extern "C" __global__ void glm_exl3_rot_1(const __nv_bfloat16* __restrict__ x, int x_stride, const int* __restrict__ pick,
                              const half* __restrict__ suh0, const half* __restrict__ suh1, half* __restrict__ out0,
                              half* __restrict__ out1, int K, int slots) {
    const int p = blockIdx.x, blk = blockIdx.y, mat = blockIdx.z;
    int row, e;
    if (1) {
        row = p;
        e = 0;
    } else {
        row = p / slots;
        const int slot = p % slots;
        if (slot == slots - 1) return;
        e = pick[row * slots + slot];
    }
    const int lane = threadIdx.x;
    const half* suh = (mat ? suh1 : suh0) + (size_t)e * K + blk * 128 + 4 * lane;
    const __nv_bfloat16* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __bfloat162float(xr[j]) * __half2float(suh[j]);
    fwht128(v, lane);
    half* o = (mat ? out1 : out0) + (size_t)p * K + blk * 128 + 4 * lane;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}
extern "C" __global__ void glm_exl3_gateup(const float* __restrict__ Z, const int* __restrict__ pick,
                                       const half* __restrict__ svh_g, const half* __restrict__ svh_u,
                                       const half* __restrict__ suh_d, half* __restrict__ xd, int P, int N, int SK,
                                       int slots, float limit) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int row = p / slots, slot = p % slots;
    if (slot == slots - 1) return;
    const int lane = threadIdx.x;
    const int e = pick[row * slots + slot];
    const int n = blk * 128 + 4 * lane;
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float sg = 0.f, su = 0.f;
        for (int s = 0; s < SK; ++s) {
            sg += Z[((size_t)(0 * SK + s) * P + p) * N + n + j];
            su += Z[((size_t)(1 * SK + s) * P + p) * N + n + j];
        }
        gv[j] = sg;
        uv[j] = su;
    }
    fwht128(gv, lane);
    fwht128(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
        float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit), limit);
        float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
        v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
    }
    fwht128(v, lane);
    half* o = xd + (size_t)p * N + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
}
extern "C" __global__ void glm_exl3_down(const float* __restrict__ Z, const int* __restrict__ pick,
                                     const half* __restrict__ svh_d, float* __restrict__ y, int P, int D, int SK,
                                     int slots) {
    const int p = blockIdx.x, blk = blockIdx.y;
    const int row = p / slots, slot = p % slots;
    if (slot == slots - 1) return;
    const int lane = threadIdx.x;
    const int e = pick[row * slots + slot];
    const int n = blk * 128 + 4 * lane;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        for (int k = 0; k < SK; ++k) s += Z[((size_t)k * P + p) * D + n + j];
        v[j] = s;
    }
    fwht128(v, lane);
    float* o = y + (size_t)p * D + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * D + n + j]);
}
extern "C" __global__ void  __launch_bounds__(4 * 32) glm_exl3_grouped(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int E) {
    // false: the n blocks of an item are adjacent in launch order, so they read its rows while L2 holds them
    const int item = false ? blockIdx.y : blockIdx.x;
    const int nblock = false ? blockIdx.x : blockIdx.y;
    if (item >= counts[0]) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (e >= E) return;                                  // the shared expert (id E) is not EXL3
    const int mat = false ? blockIdx.z : blockIdx.z / SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = mat ? T1 : T0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const half* x0 = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
    const half* x1 = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;

    const int per_split = KT / SK, per_warp = per_split / 4;
    const int nt0 = nblock * 8;
    constexpr int OUT = 16 * 8 * 16 / (4 * 32);        // the sums a thread writes (false keeps them in registers)
    float tot[OUT];
    __shared__ float red[4][16][8 * 16];
    for (int si = 0; si < (false ? SK : 1); ++si) {
    const int split = false ? si : blockIdx.z % SK;
    const int kt0 = split * per_split + warp * per_warp;
    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;

    float acc[8][2][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    if (false) {
        // the next k step's words and A fragments load while this one decodes and multiplies (the same mma order)
        uint32_t words[8], a[4];
#pragma unroll
        for (int i = 0; i < 8; ++i) words[i] = __ldg(tile + i * 32);
        {
            const int k = kt0 * 16;
            a[0] = load_pair(x0 + k, r0 >= 0); a[1] = load_pair(x1 + k, r1 >= 0);
            a[2] = load_pair(x0 + k + 8, r0 >= 0); a[3] = load_pair(x1 + k + 8, r1 >= 0);
        }
        for (int kt = kt0; kt < kt0 + per_warp; ++kt) {
            uint32_t nw[8], na[4];
            const bool more = kt + 1 < kt0 + per_warp;
            if (more) {
                const uint32_t* next = tile + (size_t)NTILES * 32;
#pragma unroll
                for (int i = 0; i < 8; ++i) nw[i] = __ldg(next + i * 32);
                const int k = (kt + 1) * 16;
                na[0] = load_pair(x0 + k, r0 >= 0); na[1] = load_pair(x1 + k, r1 >= 0);
                na[2] = load_pair(x0 + k + 8, r0 >= 0); na[3] = load_pair(x1 + k + 8, r1 >= 0);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint32_t b0[2], b1[2];
                decode_tile(words[i], lane, b0, b1);
                mma16816(acc[i][0], a, b0);
                mma16816(acc[i][1], a, b1);
            }
            if (more) {
#pragma unroll
                for (int i = 0; i < 8; ++i) words[i] = nw[i];
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = na[c];
            }
            tile += (size_t)NTILES * 32;
        }
    } else {
        for (int kt = kt0; kt < kt0 + per_warp; ++kt) {
            uint32_t words[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) words[i] = __ldg(tile + i * 32);
            const int k = kt * 16;
            uint32_t a[4] = {load_pair(x0 + k, r0 >= 0), load_pair(x1 + k, r1 >= 0),
                             load_pair(x0 + k + 8, r0 >= 0), load_pair(x1 + k + 8, r1 >= 0)};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                uint32_t b0[2], b1[2];
                decode_tile(words[i], lane, b0, b1);
                mma16816(acc[i][0], a, b0);
                mma16816(acc[i][1], a, b1);
            }
            tile += (size_t)NTILES * 32;
        }
    }

    // warps' partial sums through shared memory, added in warp order
    if (false && si > 0) __syncthreads();                // the previous split's sums are read
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < OUT; ++j) {
        const int idx = threadIdx.x + j * 4 * 32;
        const int row = idx / (8 * 16), col = idx % (8 * 16);
        const int r = rows_sh[row];
        float s = red[0][row][col];
#pragma unroll
        for (int w = 1; w < 4; ++w) s += red[w][row][col];
        if (false) {
            tot[j] = si == 0 ? 0.f + s : tot[j] + s;
        } else if (r >= 0) {
            Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
    }
    }  // splits
    if (false) {
#pragma unroll
        for (int j = 0; j < OUT; ++j) {
            const int idx = threadIdx.x + j * 4 * 32;
            const int row = idx / (8 * 16), col = idx % (8 * 16);
            const int r = rows_sh[row];
            if (r >= 0) Z[((size_t)mat * P + r) * N + nt0 * 16 + col] = tot[j];
        }
    }
}
extern "C" __global__ void  __launch_bounds__(4 * 32) glm_exl3_dec_8_0_0(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int E, int slots,
    const half* __restrict__ sv0, const half* __restrict__ sv1, const half* __restrict__ su2, void* __restrict__ out,
    float limit, int* __restrict__ done) {
    static_assert(0 == 0 || 8 * 16 == 128, "a fused epilogue needs one 128-column Hadamard block a program");
    static_assert(1 == 0 || (8 % 4 == 0 && 1 >= 1 && 8 % 1 == 0), "1 1: 8 a multiple of 4, 1 | 8");
    int item = blockIdx.x, nblock = blockIdx.y, zz = blockIdx.z;
    if constexpr (0 != 0) {                            // launch order only: the blocks of one k slice run together
        const unsigned lid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        nblock = (int)(lid % gridDim.y);
        const unsigned r = lid / gridDim.y;
        if (0 == 1) { item = (int)(r % gridDim.x); zz = (int)(r / gridDim.x); }
        else { zz = (int)(r % gridDim.z); item = (int)(r / gridDim.z); }
    }
    int e, first, cnt;
    if constexpr (1 == 0) {
        if (item >= counts[0]) return;
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    } else {                                             // one round trip: items holds a row for every block
        const int total = counts[0];
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
        if (item >= total) return;
    }
    if (e >= E) return;                                  // the shared expert (id E) is not EXL3
    const int mat = zz / SK, split = zz % SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = mat ? T1 : T0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    __shared__ __align__(16) float red[4][16][8 * 16];
    constexpr int NV = 8 / 4;                           // 1 1: 16-byte loads a lane a k step
    uint4 rv[1 == 1 ? 1 : 1][1 == 1 ? NV : 1];
    const uint32_t* tbase = T + (((size_t)e * KT + split * (KT / SK) + warp * 8) * NTILES + nblock * 8) * 32;
    const size_t kstep = (size_t)NTILES * 32;            // words from one k tile to the next
    if constexpr (1 == 1) {
#pragma unroll
        for (int d = 0; d < 1; ++d)
#pragma unroll
            for (int v = 0; v < NV; ++v) rv[d][v] = ldg_nc_v4(tbase + d * kstep + 128 * v + 4 * lane);
    }
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const int xr0 = r0 < 0 ? 0 : (0 ? r0 / slots : r0), xr1 = r1 < 0 ? 0 : (0 ? r1 / slots : r1);
    const half* x0 = X + (size_t)xr0 * K + 2 * t;
    const half* x1 = X + (size_t)xr1 * K + 2 * t;
    const int nt0 = nblock * 8;
    const int kt0 = split * (KT / SK) + warp * 8;
    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;

    float acc[8][2][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    if constexpr (1 == 1) {
        uint32_t ra[1][4];
#pragma unroll
        for (int d = 0; d < 1; ++d) {
            const int k = (kt0 + d) * 16;
            ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
            ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
        }
        uint32_t* stage = reinterpret_cast<uint32_t*>(&red[warp][0][0]);
        for (int kb = 0; kb < 8; kb += 1) {
#pragma unroll
            for (int d = 0; d < 1; ++d) {
                const int s = kb + d;
                const bool more = s + 1 < 8;
                uint32_t a[4], words[8];
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = ra[d][c];
                __syncwarp();                            // every lane has read the previous step back
#pragma unroll
                for (int v = 0; v < NV; ++v) *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];
                if (more) {                              // the slot's registers are free again
#pragma unroll
                    for (int v = 0; v < NV; ++v)
                        rv[d][v] = ldg_nc_v4(tbase + (size_t)(s + 1) * kstep + 128 * v + 4 * lane);
                    const int k = (kt0 + s + 1) * 16;
                    ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
                    ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
                }
                __syncwarp();
#pragma unroll
                for (int i = 0; i < 8; ++i) words[i] = stage[i * 32 + lane];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile(words[i], lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
        __syncwarp();                                    // the staging area becomes red[warp] below
    } else {
    // grouped_kernel's PF loop as it is (a runtime loop: unrolling it measured slower)
    uint32_t words[8], a[4];
#pragma unroll
    for (int i = 0; i < 8; ++i) words[i] = __ldg(tile + i * 32);
    {
        const int k = kt0 * 16;
        a[0] = load_pair(x0 + k, r0 >= 0); a[1] = load_pair(x1 + k, r1 >= 0);
        a[2] = load_pair(x0 + k + 8, r0 >= 0); a[3] = load_pair(x1 + k + 8, r1 >= 0);
    }
    for (int kt = kt0; kt < kt0 + 8; ++kt) {
        uint32_t nw[8], na[4];
        const bool more = kt + 1 < kt0 + 8;
        if (more) {
            const uint32_t* next = tile + (size_t)NTILES * 32;
#pragma unroll
            for (int i = 0; i < 8; ++i) nw[i] = __ldg(next + i * 32);
            const int k = (kt + 1) * 16;
            na[0] = load_pair(x0 + k, r0 >= 0); na[1] = load_pair(x1 + k, r1 >= 0);
            na[2] = load_pair(x0 + k + 8, r0 >= 0); na[3] = load_pair(x1 + k + 8, r1 >= 0);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile(words[i], lane, b0, b1);
            mma16816(acc[i][0], a, b0);
            mma16816(acc[i][1], a, b1);
        }
        if (more) {
#pragma unroll
            for (int i = 0; i < 8; ++i) words[i] = nw[i];
#pragma unroll
            for (int c = 0; c < 4; ++c) a[c] = na[c];
        }
        tile += (size_t)NTILES * 32;
    }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();

    if constexpr (0 == 1) {
        // down_epilogue_kernel on this block's sums: v = 0 + (the warps' sums in warp order), rotated, times svh
        const half* svh_d = sv0;
        float* y = reinterpret_cast<float*>(out);
        for (int i = warp; i < cnt; i += 4) {
            const int p = rows_sh[i];
            const int n = nt0 * 16 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = red[0][i][4 * lane + j];
#pragma unroll
                for (int w = 1; w < 4; ++w) s += red[w][i][4 * lane + j];
                float z = 0.f;
                z += s;
                v[j] = z;
            }
            fwht128(v, lane);
            float* o = y + (size_t)p * N + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * N + n + j]);
        }
        return;
    } else {
        constexpr int OUT = 16 * 8 * 16 / (4 * 32);
#pragma unroll
        for (int j = 0; j < OUT; ++j) {
            const int idx = threadIdx.x + j * 4 * 32;
            const int row = idx / (8 * 16), col = idx % (8 * 16);
            const int r = rows_sh[row];
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < 4; ++w) s += red[w][row][col];
            if (r >= 0) Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
        if constexpr (0 == 2) {
            __shared__ int last;
            __threadfence();                               // this block's partials before its count
            __syncthreads();
            if (threadIdx.x == 0) {
                int* c = done + (size_t)item * gridDim.y + nblock;
                last = atomicAdd(c, 1) == 2 * SK - 1;
                if (last) *c = 0;                          // the next launch starts from zero
            }
            __syncthreads();
            if (!last) return;
            __threadfence();                               // every block's partials are visible from here
            const half* svh_g = sv0;
            const half* svh_u = sv1;
            const half* suh_d = su2;
            half* xd = reinterpret_cast<half*>(out);
            for (int i = warp; i < cnt; i += 4) {
                const int p = rows_sh[i];
                const int n = nt0 * 16 + 4 * lane;
                float gv[4], uv[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float sg = 0.f, su = 0.f;
                    for (int s = 0; s < SK; ++s) {
                        sg += __ldcg(Z + ((size_t)(0 * SK + s) * P + p) * N + n + j);
                        su += __ldcg(Z + ((size_t)(1 * SK + s) * P + p) * N + n + j);
                    }
                    gv[j] = sg;
                    uv[j] = su;
                }
                fwht128(gv, lane);
                fwht128(uv, lane);
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
                    float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])),
                                           -limit), limit);
                    float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                    v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
                }
                fwht128(v, lane);
                half* o = xd + (size_t)p * N + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
            }
        }
    }
}
extern "C" __global__ void  __launch_bounds__(4 * 32) glm_exl3_dec_8_1_0(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int E, int slots,
    const half* __restrict__ sv0, const half* __restrict__ sv1, const half* __restrict__ su2, void* __restrict__ out,
    float limit, int* __restrict__ done) {
    static_assert(1 == 0 || 8 * 16 == 128, "a fused epilogue needs one 128-column Hadamard block a program");
    static_assert(1 == 0 || (8 % 4 == 0 && 1 >= 1 && 8 % 1 == 0), "1 1: 8 a multiple of 4, 1 | 8");
    int item = blockIdx.x, nblock = blockIdx.y, zz = blockIdx.z;
    if constexpr (0 != 0) {                            // launch order only: the blocks of one k slice run together
        const unsigned lid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        nblock = (int)(lid % gridDim.y);
        const unsigned r = lid / gridDim.y;
        if (0 == 1) { item = (int)(r % gridDim.x); zz = (int)(r / gridDim.x); }
        else { zz = (int)(r % gridDim.z); item = (int)(r / gridDim.z); }
    }
    int e, first, cnt;
    if constexpr (1 == 0) {
        if (item >= counts[0]) return;
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    } else {                                             // one round trip: items holds a row for every block
        const int total = counts[0];
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
        if (item >= total) return;
    }
    if (e >= E) return;                                  // the shared expert (id E) is not EXL3
    const int mat = zz / SK, split = zz % SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = mat ? T1 : T0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    __shared__ __align__(16) float red[4][16][8 * 16];
    constexpr int NV = 8 / 4;                           // 1 1: 16-byte loads a lane a k step
    uint4 rv[1 == 1 ? 1 : 1][1 == 1 ? NV : 1];
    const uint32_t* tbase = T + (((size_t)e * KT + split * (KT / SK) + warp * 8) * NTILES + nblock * 8) * 32;
    const size_t kstep = (size_t)NTILES * 32;            // words from one k tile to the next
    if constexpr (1 == 1) {
#pragma unroll
        for (int d = 0; d < 1; ++d)
#pragma unroll
            for (int v = 0; v < NV; ++v) rv[d][v] = ldg_nc_v4(tbase + d * kstep + 128 * v + 4 * lane);
    }
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const int xr0 = r0 < 0 ? 0 : (0 ? r0 / slots : r0), xr1 = r1 < 0 ? 0 : (0 ? r1 / slots : r1);
    const half* x0 = X + (size_t)xr0 * K + 2 * t;
    const half* x1 = X + (size_t)xr1 * K + 2 * t;
    const int nt0 = nblock * 8;
    const int kt0 = split * (KT / SK) + warp * 8;
    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;

    float acc[8][2][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    if constexpr (1 == 1) {
        uint32_t ra[1][4];
#pragma unroll
        for (int d = 0; d < 1; ++d) {
            const int k = (kt0 + d) * 16;
            ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
            ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
        }
        uint32_t* stage = reinterpret_cast<uint32_t*>(&red[warp][0][0]);
        for (int kb = 0; kb < 8; kb += 1) {
#pragma unroll
            for (int d = 0; d < 1; ++d) {
                const int s = kb + d;
                const bool more = s + 1 < 8;
                uint32_t a[4], words[8];
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = ra[d][c];
                __syncwarp();                            // every lane has read the previous step back
#pragma unroll
                for (int v = 0; v < NV; ++v) *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];
                if (more) {                              // the slot's registers are free again
#pragma unroll
                    for (int v = 0; v < NV; ++v)
                        rv[d][v] = ldg_nc_v4(tbase + (size_t)(s + 1) * kstep + 128 * v + 4 * lane);
                    const int k = (kt0 + s + 1) * 16;
                    ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
                    ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
                }
                __syncwarp();
#pragma unroll
                for (int i = 0; i < 8; ++i) words[i] = stage[i * 32 + lane];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile(words[i], lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
        __syncwarp();                                    // the staging area becomes red[warp] below
    } else {
    // grouped_kernel's PF loop as it is (a runtime loop: unrolling it measured slower)
    uint32_t words[8], a[4];
#pragma unroll
    for (int i = 0; i < 8; ++i) words[i] = __ldg(tile + i * 32);
    {
        const int k = kt0 * 16;
        a[0] = load_pair(x0 + k, r0 >= 0); a[1] = load_pair(x1 + k, r1 >= 0);
        a[2] = load_pair(x0 + k + 8, r0 >= 0); a[3] = load_pair(x1 + k + 8, r1 >= 0);
    }
    for (int kt = kt0; kt < kt0 + 8; ++kt) {
        uint32_t nw[8], na[4];
        const bool more = kt + 1 < kt0 + 8;
        if (more) {
            const uint32_t* next = tile + (size_t)NTILES * 32;
#pragma unroll
            for (int i = 0; i < 8; ++i) nw[i] = __ldg(next + i * 32);
            const int k = (kt + 1) * 16;
            na[0] = load_pair(x0 + k, r0 >= 0); na[1] = load_pair(x1 + k, r1 >= 0);
            na[2] = load_pair(x0 + k + 8, r0 >= 0); na[3] = load_pair(x1 + k + 8, r1 >= 0);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile(words[i], lane, b0, b1);
            mma16816(acc[i][0], a, b0);
            mma16816(acc[i][1], a, b1);
        }
        if (more) {
#pragma unroll
            for (int i = 0; i < 8; ++i) words[i] = nw[i];
#pragma unroll
            for (int c = 0; c < 4; ++c) a[c] = na[c];
        }
        tile += (size_t)NTILES * 32;
    }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();

    if constexpr (1 == 1) {
        // down_epilogue_kernel on this block's sums: v = 0 + (the warps' sums in warp order), rotated, times svh
        const half* svh_d = sv0;
        float* y = reinterpret_cast<float*>(out);
        for (int i = warp; i < cnt; i += 4) {
            const int p = rows_sh[i];
            const int n = nt0 * 16 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = red[0][i][4 * lane + j];
#pragma unroll
                for (int w = 1; w < 4; ++w) s += red[w][i][4 * lane + j];
                float z = 0.f;
                z += s;
                v[j] = z;
            }
            fwht128(v, lane);
            float* o = y + (size_t)p * N + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * N + n + j]);
        }
        return;
    } else {
        constexpr int OUT = 16 * 8 * 16 / (4 * 32);
#pragma unroll
        for (int j = 0; j < OUT; ++j) {
            const int idx = threadIdx.x + j * 4 * 32;
            const int row = idx / (8 * 16), col = idx % (8 * 16);
            const int r = rows_sh[row];
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < 4; ++w) s += red[w][row][col];
            if (r >= 0) Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
        if constexpr (1 == 2) {
            __shared__ int last;
            __threadfence();                               // this block's partials before its count
            __syncthreads();
            if (threadIdx.x == 0) {
                int* c = done + (size_t)item * gridDim.y + nblock;
                last = atomicAdd(c, 1) == 2 * SK - 1;
                if (last) *c = 0;                          // the next launch starts from zero
            }
            __syncthreads();
            if (!last) return;
            __threadfence();                               // every block's partials are visible from here
            const half* svh_g = sv0;
            const half* svh_u = sv1;
            const half* suh_d = su2;
            half* xd = reinterpret_cast<half*>(out);
            for (int i = warp; i < cnt; i += 4) {
                const int p = rows_sh[i];
                const int n = nt0 * 16 + 4 * lane;
                float gv[4], uv[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float sg = 0.f, su = 0.f;
                    for (int s = 0; s < SK; ++s) {
                        sg += __ldcg(Z + ((size_t)(0 * SK + s) * P + p) * N + n + j);
                        su += __ldcg(Z + ((size_t)(1 * SK + s) * P + p) * N + n + j);
                    }
                    gv[j] = sg;
                    uv[j] = su;
                }
                fwht128(gv, lane);
                fwht128(uv, lane);
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
                    float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])),
                                           -limit), limit);
                    float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                    v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
                }
                fwht128(v, lane);
                half* o = xd + (size_t)p * N + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
            }
        }
    }
}
extern "C" __global__ void  __launch_bounds__(4 * 32) glm_exl3_dec_10_0_0(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int E, int slots,
    const half* __restrict__ sv0, const half* __restrict__ sv1, const half* __restrict__ su2, void* __restrict__ out,
    float limit, int* __restrict__ done) {
    static_assert(0 == 0 || 8 * 16 == 128, "a fused epilogue needs one 128-column Hadamard block a program");
    static_assert(1 == 0 || (8 % 4 == 0 && 1 >= 1 && 10 % 1 == 0), "1 1: 8 a multiple of 4, 1 | 10");
    int item = blockIdx.x, nblock = blockIdx.y, zz = blockIdx.z;
    if constexpr (0 != 0) {                            // launch order only: the blocks of one k slice run together
        const unsigned lid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        nblock = (int)(lid % gridDim.y);
        const unsigned r = lid / gridDim.y;
        if (0 == 1) { item = (int)(r % gridDim.x); zz = (int)(r / gridDim.x); }
        else { zz = (int)(r % gridDim.z); item = (int)(r / gridDim.z); }
    }
    int e, first, cnt;
    if constexpr (1 == 0) {
        if (item >= counts[0]) return;
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    } else {                                             // one round trip: items holds a row for every block
        const int total = counts[0];
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
        if (item >= total) return;
    }
    if (e >= E) return;                                  // the shared expert (id E) is not EXL3
    const int mat = zz / SK, split = zz % SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = mat ? T1 : T0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    __shared__ __align__(16) float red[4][16][8 * 16];
    constexpr int NV = 8 / 4;                           // 1 1: 16-byte loads a lane a k step
    uint4 rv[1 == 1 ? 1 : 1][1 == 1 ? NV : 1];
    const uint32_t* tbase = T + (((size_t)e * KT + split * (KT / SK) + warp * 10) * NTILES + nblock * 8) * 32;
    const size_t kstep = (size_t)NTILES * 32;            // words from one k tile to the next
    if constexpr (1 == 1) {
#pragma unroll
        for (int d = 0; d < 1; ++d)
#pragma unroll
            for (int v = 0; v < NV; ++v) rv[d][v] = ldg_nc_v4(tbase + d * kstep + 128 * v + 4 * lane);
    }
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const int xr0 = r0 < 0 ? 0 : (0 ? r0 / slots : r0), xr1 = r1 < 0 ? 0 : (0 ? r1 / slots : r1);
    const half* x0 = X + (size_t)xr0 * K + 2 * t;
    const half* x1 = X + (size_t)xr1 * K + 2 * t;
    const int nt0 = nblock * 8;
    const int kt0 = split * (KT / SK) + warp * 10;
    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;

    float acc[8][2][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    if constexpr (1 == 1) {
        uint32_t ra[1][4];
#pragma unroll
        for (int d = 0; d < 1; ++d) {
            const int k = (kt0 + d) * 16;
            ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
            ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
        }
        uint32_t* stage = reinterpret_cast<uint32_t*>(&red[warp][0][0]);
        for (int kb = 0; kb < 10; kb += 1) {
#pragma unroll
            for (int d = 0; d < 1; ++d) {
                const int s = kb + d;
                const bool more = s + 1 < 10;
                uint32_t a[4], words[8];
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = ra[d][c];
                __syncwarp();                            // every lane has read the previous step back
#pragma unroll
                for (int v = 0; v < NV; ++v) *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];
                if (more) {                              // the slot's registers are free again
#pragma unroll
                    for (int v = 0; v < NV; ++v)
                        rv[d][v] = ldg_nc_v4(tbase + (size_t)(s + 1) * kstep + 128 * v + 4 * lane);
                    const int k = (kt0 + s + 1) * 16;
                    ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
                    ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
                }
                __syncwarp();
#pragma unroll
                for (int i = 0; i < 8; ++i) words[i] = stage[i * 32 + lane];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile(words[i], lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
        __syncwarp();                                    // the staging area becomes red[warp] below
    } else {
    // grouped_kernel's PF loop as it is (a runtime loop: unrolling it measured slower)
    uint32_t words[8], a[4];
#pragma unroll
    for (int i = 0; i < 8; ++i) words[i] = __ldg(tile + i * 32);
    {
        const int k = kt0 * 16;
        a[0] = load_pair(x0 + k, r0 >= 0); a[1] = load_pair(x1 + k, r1 >= 0);
        a[2] = load_pair(x0 + k + 8, r0 >= 0); a[3] = load_pair(x1 + k + 8, r1 >= 0);
    }
    for (int kt = kt0; kt < kt0 + 10; ++kt) {
        uint32_t nw[8], na[4];
        const bool more = kt + 1 < kt0 + 10;
        if (more) {
            const uint32_t* next = tile + (size_t)NTILES * 32;
#pragma unroll
            for (int i = 0; i < 8; ++i) nw[i] = __ldg(next + i * 32);
            const int k = (kt + 1) * 16;
            na[0] = load_pair(x0 + k, r0 >= 0); na[1] = load_pair(x1 + k, r1 >= 0);
            na[2] = load_pair(x0 + k + 8, r0 >= 0); na[3] = load_pair(x1 + k + 8, r1 >= 0);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile(words[i], lane, b0, b1);
            mma16816(acc[i][0], a, b0);
            mma16816(acc[i][1], a, b1);
        }
        if (more) {
#pragma unroll
            for (int i = 0; i < 8; ++i) words[i] = nw[i];
#pragma unroll
            for (int c = 0; c < 4; ++c) a[c] = na[c];
        }
        tile += (size_t)NTILES * 32;
    }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();

    if constexpr (0 == 1) {
        // down_epilogue_kernel on this block's sums: v = 0 + (the warps' sums in warp order), rotated, times svh
        const half* svh_d = sv0;
        float* y = reinterpret_cast<float*>(out);
        for (int i = warp; i < cnt; i += 4) {
            const int p = rows_sh[i];
            const int n = nt0 * 16 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = red[0][i][4 * lane + j];
#pragma unroll
                for (int w = 1; w < 4; ++w) s += red[w][i][4 * lane + j];
                float z = 0.f;
                z += s;
                v[j] = z;
            }
            fwht128(v, lane);
            float* o = y + (size_t)p * N + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * N + n + j]);
        }
        return;
    } else {
        constexpr int OUT = 16 * 8 * 16 / (4 * 32);
#pragma unroll
        for (int j = 0; j < OUT; ++j) {
            const int idx = threadIdx.x + j * 4 * 32;
            const int row = idx / (8 * 16), col = idx % (8 * 16);
            const int r = rows_sh[row];
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < 4; ++w) s += red[w][row][col];
            if (r >= 0) Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
        if constexpr (0 == 2) {
            __shared__ int last;
            __threadfence();                               // this block's partials before its count
            __syncthreads();
            if (threadIdx.x == 0) {
                int* c = done + (size_t)item * gridDim.y + nblock;
                last = atomicAdd(c, 1) == 2 * SK - 1;
                if (last) *c = 0;                          // the next launch starts from zero
            }
            __syncthreads();
            if (!last) return;
            __threadfence();                               // every block's partials are visible from here
            const half* svh_g = sv0;
            const half* svh_u = sv1;
            const half* suh_d = su2;
            half* xd = reinterpret_cast<half*>(out);
            for (int i = warp; i < cnt; i += 4) {
                const int p = rows_sh[i];
                const int n = nt0 * 16 + 4 * lane;
                float gv[4], uv[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float sg = 0.f, su = 0.f;
                    for (int s = 0; s < SK; ++s) {
                        sg += __ldcg(Z + ((size_t)(0 * SK + s) * P + p) * N + n + j);
                        su += __ldcg(Z + ((size_t)(1 * SK + s) * P + p) * N + n + j);
                    }
                    gv[j] = sg;
                    uv[j] = su;
                }
                fwht128(gv, lane);
                fwht128(uv, lane);
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
                    float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])),
                                           -limit), limit);
                    float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                    v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
                }
                fwht128(v, lane);
                half* o = xd + (size_t)p * N + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
            }
        }
    }
}
extern "C" __global__ void  __launch_bounds__(4 * 32) glm_exl3_dec_10_1_0(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int E, int slots,
    const half* __restrict__ sv0, const half* __restrict__ sv1, const half* __restrict__ su2, void* __restrict__ out,
    float limit, int* __restrict__ done) {
    static_assert(1 == 0 || 8 * 16 == 128, "a fused epilogue needs one 128-column Hadamard block a program");
    static_assert(1 == 0 || (8 % 4 == 0 && 1 >= 1 && 10 % 1 == 0), "1 1: 8 a multiple of 4, 1 | 10");
    int item = blockIdx.x, nblock = blockIdx.y, zz = blockIdx.z;
    if constexpr (0 != 0) {                            // launch order only: the blocks of one k slice run together
        const unsigned lid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        nblock = (int)(lid % gridDim.y);
        const unsigned r = lid / gridDim.y;
        if (0 == 1) { item = (int)(r % gridDim.x); zz = (int)(r / gridDim.x); }
        else { zz = (int)(r % gridDim.z); item = (int)(r / gridDim.z); }
    }
    int e, first, cnt;
    if constexpr (1 == 0) {
        if (item >= counts[0]) return;
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    } else {                                             // one round trip: items holds a row for every block
        const int total = counts[0];
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
        if (item >= total) return;
    }
    if (e >= E) return;                                  // the shared expert (id E) is not EXL3
    const int mat = zz / SK, split = zz % SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = mat ? T1 : T0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    __shared__ __align__(16) float red[4][16][8 * 16];
    constexpr int NV = 8 / 4;                           // 1 1: 16-byte loads a lane a k step
    uint4 rv[1 == 1 ? 1 : 1][1 == 1 ? NV : 1];
    const uint32_t* tbase = T + (((size_t)e * KT + split * (KT / SK) + warp * 10) * NTILES + nblock * 8) * 32;
    const size_t kstep = (size_t)NTILES * 32;            // words from one k tile to the next
    if constexpr (1 == 1) {
#pragma unroll
        for (int d = 0; d < 1; ++d)
#pragma unroll
            for (int v = 0; v < NV; ++v) rv[d][v] = ldg_nc_v4(tbase + d * kstep + 128 * v + 4 * lane);
    }
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const int xr0 = r0 < 0 ? 0 : (0 ? r0 / slots : r0), xr1 = r1 < 0 ? 0 : (0 ? r1 / slots : r1);
    const half* x0 = X + (size_t)xr0 * K + 2 * t;
    const half* x1 = X + (size_t)xr1 * K + 2 * t;
    const int nt0 = nblock * 8;
    const int kt0 = split * (KT / SK) + warp * 10;
    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;

    float acc[8][2][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    if constexpr (1 == 1) {
        uint32_t ra[1][4];
#pragma unroll
        for (int d = 0; d < 1; ++d) {
            const int k = (kt0 + d) * 16;
            ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
            ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
        }
        uint32_t* stage = reinterpret_cast<uint32_t*>(&red[warp][0][0]);
        for (int kb = 0; kb < 10; kb += 1) {
#pragma unroll
            for (int d = 0; d < 1; ++d) {
                const int s = kb + d;
                const bool more = s + 1 < 10;
                uint32_t a[4], words[8];
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = ra[d][c];
                __syncwarp();                            // every lane has read the previous step back
#pragma unroll
                for (int v = 0; v < NV; ++v) *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];
                if (more) {                              // the slot's registers are free again
#pragma unroll
                    for (int v = 0; v < NV; ++v)
                        rv[d][v] = ldg_nc_v4(tbase + (size_t)(s + 1) * kstep + 128 * v + 4 * lane);
                    const int k = (kt0 + s + 1) * 16;
                    ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
                    ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
                }
                __syncwarp();
#pragma unroll
                for (int i = 0; i < 8; ++i) words[i] = stage[i * 32 + lane];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile(words[i], lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
        __syncwarp();                                    // the staging area becomes red[warp] below
    } else {
    // grouped_kernel's PF loop as it is (a runtime loop: unrolling it measured slower)
    uint32_t words[8], a[4];
#pragma unroll
    for (int i = 0; i < 8; ++i) words[i] = __ldg(tile + i * 32);
    {
        const int k = kt0 * 16;
        a[0] = load_pair(x0 + k, r0 >= 0); a[1] = load_pair(x1 + k, r1 >= 0);
        a[2] = load_pair(x0 + k + 8, r0 >= 0); a[3] = load_pair(x1 + k + 8, r1 >= 0);
    }
    for (int kt = kt0; kt < kt0 + 10; ++kt) {
        uint32_t nw[8], na[4];
        const bool more = kt + 1 < kt0 + 10;
        if (more) {
            const uint32_t* next = tile + (size_t)NTILES * 32;
#pragma unroll
            for (int i = 0; i < 8; ++i) nw[i] = __ldg(next + i * 32);
            const int k = (kt + 1) * 16;
            na[0] = load_pair(x0 + k, r0 >= 0); na[1] = load_pair(x1 + k, r1 >= 0);
            na[2] = load_pair(x0 + k + 8, r0 >= 0); na[3] = load_pair(x1 + k + 8, r1 >= 0);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile(words[i], lane, b0, b1);
            mma16816(acc[i][0], a, b0);
            mma16816(acc[i][1], a, b1);
        }
        if (more) {
#pragma unroll
            for (int i = 0; i < 8; ++i) words[i] = nw[i];
#pragma unroll
            for (int c = 0; c < 4; ++c) a[c] = na[c];
        }
        tile += (size_t)NTILES * 32;
    }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();

    if constexpr (1 == 1) {
        // down_epilogue_kernel on this block's sums: v = 0 + (the warps' sums in warp order), rotated, times svh
        const half* svh_d = sv0;
        float* y = reinterpret_cast<float*>(out);
        for (int i = warp; i < cnt; i += 4) {
            const int p = rows_sh[i];
            const int n = nt0 * 16 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = red[0][i][4 * lane + j];
#pragma unroll
                for (int w = 1; w < 4; ++w) s += red[w][i][4 * lane + j];
                float z = 0.f;
                z += s;
                v[j] = z;
            }
            fwht128(v, lane);
            float* o = y + (size_t)p * N + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * N + n + j]);
        }
        return;
    } else {
        constexpr int OUT = 16 * 8 * 16 / (4 * 32);
#pragma unroll
        for (int j = 0; j < OUT; ++j) {
            const int idx = threadIdx.x + j * 4 * 32;
            const int row = idx / (8 * 16), col = idx % (8 * 16);
            const int r = rows_sh[row];
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < 4; ++w) s += red[w][row][col];
            if (r >= 0) Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
        if constexpr (1 == 2) {
            __shared__ int last;
            __threadfence();                               // this block's partials before its count
            __syncthreads();
            if (threadIdx.x == 0) {
                int* c = done + (size_t)item * gridDim.y + nblock;
                last = atomicAdd(c, 1) == 2 * SK - 1;
                if (last) *c = 0;                          // the next launch starts from zero
            }
            __syncthreads();
            if (!last) return;
            __threadfence();                               // every block's partials are visible from here
            const half* svh_g = sv0;
            const half* svh_u = sv1;
            const half* suh_d = su2;
            half* xd = reinterpret_cast<half*>(out);
            for (int i = warp; i < cnt; i += 4) {
                const int p = rows_sh[i];
                const int n = nt0 * 16 + 4 * lane;
                float gv[4], uv[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float sg = 0.f, su = 0.f;
                    for (int s = 0; s < SK; ++s) {
                        sg += __ldcg(Z + ((size_t)(0 * SK + s) * P + p) * N + n + j);
                        su += __ldcg(Z + ((size_t)(1 * SK + s) * P + p) * N + n + j);
                    }
                    gv[j] = sg;
                    uv[j] = su;
                }
                fwht128(gv, lane);
                fwht128(uv, lane);
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
                    float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])),
                                           -limit), limit);
                    float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                    v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
                }
                fwht128(v, lane);
                half* o = xd + (size_t)p * N + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
            }
        }
    }
}
extern "C" __global__ void  __launch_bounds__(4 * 32) glm_exl3_dec_12_0_0(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int E, int slots,
    const half* __restrict__ sv0, const half* __restrict__ sv1, const half* __restrict__ su2, void* __restrict__ out,
    float limit, int* __restrict__ done) {
    static_assert(0 == 0 || 8 * 16 == 128, "a fused epilogue needs one 128-column Hadamard block a program");
    static_assert(1 == 0 || (8 % 4 == 0 && 1 >= 1 && 12 % 1 == 0), "1 1: 8 a multiple of 4, 1 | 12");
    int item = blockIdx.x, nblock = blockIdx.y, zz = blockIdx.z;
    if constexpr (0 != 0) {                            // launch order only: the blocks of one k slice run together
        const unsigned lid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        nblock = (int)(lid % gridDim.y);
        const unsigned r = lid / gridDim.y;
        if (0 == 1) { item = (int)(r % gridDim.x); zz = (int)(r / gridDim.x); }
        else { zz = (int)(r % gridDim.z); item = (int)(r / gridDim.z); }
    }
    int e, first, cnt;
    if constexpr (1 == 0) {
        if (item >= counts[0]) return;
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    } else {                                             // one round trip: items holds a row for every block
        const int total = counts[0];
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
        if (item >= total) return;
    }
    if (e >= E) return;                                  // the shared expert (id E) is not EXL3
    const int mat = zz / SK, split = zz % SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = mat ? T1 : T0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    __shared__ __align__(16) float red[4][16][8 * 16];
    constexpr int NV = 8 / 4;                           // 1 1: 16-byte loads a lane a k step
    uint4 rv[1 == 1 ? 1 : 1][1 == 1 ? NV : 1];
    const uint32_t* tbase = T + (((size_t)e * KT + split * (KT / SK) + warp * 12) * NTILES + nblock * 8) * 32;
    const size_t kstep = (size_t)NTILES * 32;            // words from one k tile to the next
    if constexpr (1 == 1) {
#pragma unroll
        for (int d = 0; d < 1; ++d)
#pragma unroll
            for (int v = 0; v < NV; ++v) rv[d][v] = ldg_nc_v4(tbase + d * kstep + 128 * v + 4 * lane);
    }
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const int xr0 = r0 < 0 ? 0 : (0 ? r0 / slots : r0), xr1 = r1 < 0 ? 0 : (0 ? r1 / slots : r1);
    const half* x0 = X + (size_t)xr0 * K + 2 * t;
    const half* x1 = X + (size_t)xr1 * K + 2 * t;
    const int nt0 = nblock * 8;
    const int kt0 = split * (KT / SK) + warp * 12;
    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;

    float acc[8][2][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    if constexpr (1 == 1) {
        uint32_t ra[1][4];
#pragma unroll
        for (int d = 0; d < 1; ++d) {
            const int k = (kt0 + d) * 16;
            ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
            ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
        }
        uint32_t* stage = reinterpret_cast<uint32_t*>(&red[warp][0][0]);
        for (int kb = 0; kb < 12; kb += 1) {
#pragma unroll
            for (int d = 0; d < 1; ++d) {
                const int s = kb + d;
                const bool more = s + 1 < 12;
                uint32_t a[4], words[8];
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = ra[d][c];
                __syncwarp();                            // every lane has read the previous step back
#pragma unroll
                for (int v = 0; v < NV; ++v) *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];
                if (more) {                              // the slot's registers are free again
#pragma unroll
                    for (int v = 0; v < NV; ++v)
                        rv[d][v] = ldg_nc_v4(tbase + (size_t)(s + 1) * kstep + 128 * v + 4 * lane);
                    const int k = (kt0 + s + 1) * 16;
                    ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
                    ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
                }
                __syncwarp();
#pragma unroll
                for (int i = 0; i < 8; ++i) words[i] = stage[i * 32 + lane];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile(words[i], lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
        __syncwarp();                                    // the staging area becomes red[warp] below
    } else {
    // grouped_kernel's PF loop as it is (a runtime loop: unrolling it measured slower)
    uint32_t words[8], a[4];
#pragma unroll
    for (int i = 0; i < 8; ++i) words[i] = __ldg(tile + i * 32);
    {
        const int k = kt0 * 16;
        a[0] = load_pair(x0 + k, r0 >= 0); a[1] = load_pair(x1 + k, r1 >= 0);
        a[2] = load_pair(x0 + k + 8, r0 >= 0); a[3] = load_pair(x1 + k + 8, r1 >= 0);
    }
    for (int kt = kt0; kt < kt0 + 12; ++kt) {
        uint32_t nw[8], na[4];
        const bool more = kt + 1 < kt0 + 12;
        if (more) {
            const uint32_t* next = tile + (size_t)NTILES * 32;
#pragma unroll
            for (int i = 0; i < 8; ++i) nw[i] = __ldg(next + i * 32);
            const int k = (kt + 1) * 16;
            na[0] = load_pair(x0 + k, r0 >= 0); na[1] = load_pair(x1 + k, r1 >= 0);
            na[2] = load_pair(x0 + k + 8, r0 >= 0); na[3] = load_pair(x1 + k + 8, r1 >= 0);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile(words[i], lane, b0, b1);
            mma16816(acc[i][0], a, b0);
            mma16816(acc[i][1], a, b1);
        }
        if (more) {
#pragma unroll
            for (int i = 0; i < 8; ++i) words[i] = nw[i];
#pragma unroll
            for (int c = 0; c < 4; ++c) a[c] = na[c];
        }
        tile += (size_t)NTILES * 32;
    }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();

    if constexpr (0 == 1) {
        // down_epilogue_kernel on this block's sums: v = 0 + (the warps' sums in warp order), rotated, times svh
        const half* svh_d = sv0;
        float* y = reinterpret_cast<float*>(out);
        for (int i = warp; i < cnt; i += 4) {
            const int p = rows_sh[i];
            const int n = nt0 * 16 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = red[0][i][4 * lane + j];
#pragma unroll
                for (int w = 1; w < 4; ++w) s += red[w][i][4 * lane + j];
                float z = 0.f;
                z += s;
                v[j] = z;
            }
            fwht128(v, lane);
            float* o = y + (size_t)p * N + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * N + n + j]);
        }
        return;
    } else {
        constexpr int OUT = 16 * 8 * 16 / (4 * 32);
#pragma unroll
        for (int j = 0; j < OUT; ++j) {
            const int idx = threadIdx.x + j * 4 * 32;
            const int row = idx / (8 * 16), col = idx % (8 * 16);
            const int r = rows_sh[row];
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < 4; ++w) s += red[w][row][col];
            if (r >= 0) Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
        if constexpr (0 == 2) {
            __shared__ int last;
            __threadfence();                               // this block's partials before its count
            __syncthreads();
            if (threadIdx.x == 0) {
                int* c = done + (size_t)item * gridDim.y + nblock;
                last = atomicAdd(c, 1) == 2 * SK - 1;
                if (last) *c = 0;                          // the next launch starts from zero
            }
            __syncthreads();
            if (!last) return;
            __threadfence();                               // every block's partials are visible from here
            const half* svh_g = sv0;
            const half* svh_u = sv1;
            const half* suh_d = su2;
            half* xd = reinterpret_cast<half*>(out);
            for (int i = warp; i < cnt; i += 4) {
                const int p = rows_sh[i];
                const int n = nt0 * 16 + 4 * lane;
                float gv[4], uv[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float sg = 0.f, su = 0.f;
                    for (int s = 0; s < SK; ++s) {
                        sg += __ldcg(Z + ((size_t)(0 * SK + s) * P + p) * N + n + j);
                        su += __ldcg(Z + ((size_t)(1 * SK + s) * P + p) * N + n + j);
                    }
                    gv[j] = sg;
                    uv[j] = su;
                }
                fwht128(gv, lane);
                fwht128(uv, lane);
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
                    float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])),
                                           -limit), limit);
                    float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                    v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
                }
                fwht128(v, lane);
                half* o = xd + (size_t)p * N + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
            }
        }
    }
}
extern "C" __global__ void  __launch_bounds__(4 * 32) glm_exl3_dec_12_1_0(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int E, int slots,
    const half* __restrict__ sv0, const half* __restrict__ sv1, const half* __restrict__ su2, void* __restrict__ out,
    float limit, int* __restrict__ done) {
    static_assert(1 == 0 || 8 * 16 == 128, "a fused epilogue needs one 128-column Hadamard block a program");
    static_assert(1 == 0 || (8 % 4 == 0 && 1 >= 1 && 12 % 1 == 0), "1 1: 8 a multiple of 4, 1 | 12");
    int item = blockIdx.x, nblock = blockIdx.y, zz = blockIdx.z;
    if constexpr (0 != 0) {                            // launch order only: the blocks of one k slice run together
        const unsigned lid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        nblock = (int)(lid % gridDim.y);
        const unsigned r = lid / gridDim.y;
        if (0 == 1) { item = (int)(r % gridDim.x); zz = (int)(r / gridDim.x); }
        else { zz = (int)(r % gridDim.z); item = (int)(r / gridDim.z); }
    }
    int e, first, cnt;
    if constexpr (1 == 0) {
        if (item >= counts[0]) return;
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    } else {                                             // one round trip: items holds a row for every block
        const int total = counts[0];
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
        if (item >= total) return;
    }
    if (e >= E) return;                                  // the shared expert (id E) is not EXL3
    const int mat = zz / SK, split = zz % SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = mat ? T1 : T0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    __shared__ __align__(16) float red[4][16][8 * 16];
    constexpr int NV = 8 / 4;                           // 1 1: 16-byte loads a lane a k step
    uint4 rv[1 == 1 ? 1 : 1][1 == 1 ? NV : 1];
    const uint32_t* tbase = T + (((size_t)e * KT + split * (KT / SK) + warp * 12) * NTILES + nblock * 8) * 32;
    const size_t kstep = (size_t)NTILES * 32;            // words from one k tile to the next
    if constexpr (1 == 1) {
#pragma unroll
        for (int d = 0; d < 1; ++d)
#pragma unroll
            for (int v = 0; v < NV; ++v) rv[d][v] = ldg_nc_v4(tbase + d * kstep + 128 * v + 4 * lane);
    }
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const int xr0 = r0 < 0 ? 0 : (0 ? r0 / slots : r0), xr1 = r1 < 0 ? 0 : (0 ? r1 / slots : r1);
    const half* x0 = X + (size_t)xr0 * K + 2 * t;
    const half* x1 = X + (size_t)xr1 * K + 2 * t;
    const int nt0 = nblock * 8;
    const int kt0 = split * (KT / SK) + warp * 12;
    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;

    float acc[8][2][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    if constexpr (1 == 1) {
        uint32_t ra[1][4];
#pragma unroll
        for (int d = 0; d < 1; ++d) {
            const int k = (kt0 + d) * 16;
            ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
            ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
        }
        uint32_t* stage = reinterpret_cast<uint32_t*>(&red[warp][0][0]);
        for (int kb = 0; kb < 12; kb += 1) {
#pragma unroll
            for (int d = 0; d < 1; ++d) {
                const int s = kb + d;
                const bool more = s + 1 < 12;
                uint32_t a[4], words[8];
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = ra[d][c];
                __syncwarp();                            // every lane has read the previous step back
#pragma unroll
                for (int v = 0; v < NV; ++v) *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];
                if (more) {                              // the slot's registers are free again
#pragma unroll
                    for (int v = 0; v < NV; ++v)
                        rv[d][v] = ldg_nc_v4(tbase + (size_t)(s + 1) * kstep + 128 * v + 4 * lane);
                    const int k = (kt0 + s + 1) * 16;
                    ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
                    ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
                }
                __syncwarp();
#pragma unroll
                for (int i = 0; i < 8; ++i) words[i] = stage[i * 32 + lane];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile(words[i], lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
        __syncwarp();                                    // the staging area becomes red[warp] below
    } else {
    // grouped_kernel's PF loop as it is (a runtime loop: unrolling it measured slower)
    uint32_t words[8], a[4];
#pragma unroll
    for (int i = 0; i < 8; ++i) words[i] = __ldg(tile + i * 32);
    {
        const int k = kt0 * 16;
        a[0] = load_pair(x0 + k, r0 >= 0); a[1] = load_pair(x1 + k, r1 >= 0);
        a[2] = load_pair(x0 + k + 8, r0 >= 0); a[3] = load_pair(x1 + k + 8, r1 >= 0);
    }
    for (int kt = kt0; kt < kt0 + 12; ++kt) {
        uint32_t nw[8], na[4];
        const bool more = kt + 1 < kt0 + 12;
        if (more) {
            const uint32_t* next = tile + (size_t)NTILES * 32;
#pragma unroll
            for (int i = 0; i < 8; ++i) nw[i] = __ldg(next + i * 32);
            const int k = (kt + 1) * 16;
            na[0] = load_pair(x0 + k, r0 >= 0); na[1] = load_pair(x1 + k, r1 >= 0);
            na[2] = load_pair(x0 + k + 8, r0 >= 0); na[3] = load_pair(x1 + k + 8, r1 >= 0);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile(words[i], lane, b0, b1);
            mma16816(acc[i][0], a, b0);
            mma16816(acc[i][1], a, b1);
        }
        if (more) {
#pragma unroll
            for (int i = 0; i < 8; ++i) words[i] = nw[i];
#pragma unroll
            for (int c = 0; c < 4; ++c) a[c] = na[c];
        }
        tile += (size_t)NTILES * 32;
    }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();

    if constexpr (1 == 1) {
        // down_epilogue_kernel on this block's sums: v = 0 + (the warps' sums in warp order), rotated, times svh
        const half* svh_d = sv0;
        float* y = reinterpret_cast<float*>(out);
        for (int i = warp; i < cnt; i += 4) {
            const int p = rows_sh[i];
            const int n = nt0 * 16 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = red[0][i][4 * lane + j];
#pragma unroll
                for (int w = 1; w < 4; ++w) s += red[w][i][4 * lane + j];
                float z = 0.f;
                z += s;
                v[j] = z;
            }
            fwht128(v, lane);
            float* o = y + (size_t)p * N + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * N + n + j]);
        }
        return;
    } else {
        constexpr int OUT = 16 * 8 * 16 / (4 * 32);
#pragma unroll
        for (int j = 0; j < OUT; ++j) {
            const int idx = threadIdx.x + j * 4 * 32;
            const int row = idx / (8 * 16), col = idx % (8 * 16);
            const int r = rows_sh[row];
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < 4; ++w) s += red[w][row][col];
            if (r >= 0) Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
        if constexpr (1 == 2) {
            __shared__ int last;
            __threadfence();                               // this block's partials before its count
            __syncthreads();
            if (threadIdx.x == 0) {
                int* c = done + (size_t)item * gridDim.y + nblock;
                last = atomicAdd(c, 1) == 2 * SK - 1;
                if (last) *c = 0;                          // the next launch starts from zero
            }
            __syncthreads();
            if (!last) return;
            __threadfence();                               // every block's partials are visible from here
            const half* svh_g = sv0;
            const half* svh_u = sv1;
            const half* suh_d = su2;
            half* xd = reinterpret_cast<half*>(out);
            for (int i = warp; i < cnt; i += 4) {
                const int p = rows_sh[i];
                const int n = nt0 * 16 + 4 * lane;
                float gv[4], uv[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float sg = 0.f, su = 0.f;
                    for (int s = 0; s < SK; ++s) {
                        sg += __ldcg(Z + ((size_t)(0 * SK + s) * P + p) * N + n + j);
                        su += __ldcg(Z + ((size_t)(1 * SK + s) * P + p) * N + n + j);
                    }
                    gv[j] = sg;
                    uv[j] = su;
                }
                fwht128(gv, lane);
                fwht128(uv, lane);
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
                    float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])),
                                           -limit), limit);
                    float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                    v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
                }
                fwht128(v, lane);
                half* o = xd + (size_t)p * N + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
            }
        }
    }
}
extern "C" __global__ void  __launch_bounds__(4 * 32) glm_exl3_dec_16_0_0(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int E, int slots,
    const half* __restrict__ sv0, const half* __restrict__ sv1, const half* __restrict__ su2, void* __restrict__ out,
    float limit, int* __restrict__ done) {
    static_assert(0 == 0 || 8 * 16 == 128, "a fused epilogue needs one 128-column Hadamard block a program");
    static_assert(1 == 0 || (8 % 4 == 0 && 1 >= 1 && 16 % 1 == 0), "1 1: 8 a multiple of 4, 1 | 16");
    int item = blockIdx.x, nblock = blockIdx.y, zz = blockIdx.z;
    if constexpr (0 != 0) {                            // launch order only: the blocks of one k slice run together
        const unsigned lid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        nblock = (int)(lid % gridDim.y);
        const unsigned r = lid / gridDim.y;
        if (0 == 1) { item = (int)(r % gridDim.x); zz = (int)(r / gridDim.x); }
        else { zz = (int)(r % gridDim.z); item = (int)(r / gridDim.z); }
    }
    int e, first, cnt;
    if constexpr (1 == 0) {
        if (item >= counts[0]) return;
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    } else {                                             // one round trip: items holds a row for every block
        const int total = counts[0];
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
        if (item >= total) return;
    }
    if (e >= E) return;                                  // the shared expert (id E) is not EXL3
    const int mat = zz / SK, split = zz % SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = mat ? T1 : T0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    __shared__ __align__(16) float red[4][16][8 * 16];
    constexpr int NV = 8 / 4;                           // 1 1: 16-byte loads a lane a k step
    uint4 rv[1 == 1 ? 1 : 1][1 == 1 ? NV : 1];
    const uint32_t* tbase = T + (((size_t)e * KT + split * (KT / SK) + warp * 16) * NTILES + nblock * 8) * 32;
    const size_t kstep = (size_t)NTILES * 32;            // words from one k tile to the next
    if constexpr (1 == 1) {
#pragma unroll
        for (int d = 0; d < 1; ++d)
#pragma unroll
            for (int v = 0; v < NV; ++v) rv[d][v] = ldg_nc_v4(tbase + d * kstep + 128 * v + 4 * lane);
    }
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const int xr0 = r0 < 0 ? 0 : (0 ? r0 / slots : r0), xr1 = r1 < 0 ? 0 : (0 ? r1 / slots : r1);
    const half* x0 = X + (size_t)xr0 * K + 2 * t;
    const half* x1 = X + (size_t)xr1 * K + 2 * t;
    const int nt0 = nblock * 8;
    const int kt0 = split * (KT / SK) + warp * 16;
    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;

    float acc[8][2][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    if constexpr (1 == 1) {
        uint32_t ra[1][4];
#pragma unroll
        for (int d = 0; d < 1; ++d) {
            const int k = (kt0 + d) * 16;
            ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
            ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
        }
        uint32_t* stage = reinterpret_cast<uint32_t*>(&red[warp][0][0]);
        for (int kb = 0; kb < 16; kb += 1) {
#pragma unroll
            for (int d = 0; d < 1; ++d) {
                const int s = kb + d;
                const bool more = s + 1 < 16;
                uint32_t a[4], words[8];
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = ra[d][c];
                __syncwarp();                            // every lane has read the previous step back
#pragma unroll
                for (int v = 0; v < NV; ++v) *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];
                if (more) {                              // the slot's registers are free again
#pragma unroll
                    for (int v = 0; v < NV; ++v)
                        rv[d][v] = ldg_nc_v4(tbase + (size_t)(s + 1) * kstep + 128 * v + 4 * lane);
                    const int k = (kt0 + s + 1) * 16;
                    ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
                    ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
                }
                __syncwarp();
#pragma unroll
                for (int i = 0; i < 8; ++i) words[i] = stage[i * 32 + lane];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile(words[i], lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
        __syncwarp();                                    // the staging area becomes red[warp] below
    } else {
    // grouped_kernel's PF loop as it is (a runtime loop: unrolling it measured slower)
    uint32_t words[8], a[4];
#pragma unroll
    for (int i = 0; i < 8; ++i) words[i] = __ldg(tile + i * 32);
    {
        const int k = kt0 * 16;
        a[0] = load_pair(x0 + k, r0 >= 0); a[1] = load_pair(x1 + k, r1 >= 0);
        a[2] = load_pair(x0 + k + 8, r0 >= 0); a[3] = load_pair(x1 + k + 8, r1 >= 0);
    }
    for (int kt = kt0; kt < kt0 + 16; ++kt) {
        uint32_t nw[8], na[4];
        const bool more = kt + 1 < kt0 + 16;
        if (more) {
            const uint32_t* next = tile + (size_t)NTILES * 32;
#pragma unroll
            for (int i = 0; i < 8; ++i) nw[i] = __ldg(next + i * 32);
            const int k = (kt + 1) * 16;
            na[0] = load_pair(x0 + k, r0 >= 0); na[1] = load_pair(x1 + k, r1 >= 0);
            na[2] = load_pair(x0 + k + 8, r0 >= 0); na[3] = load_pair(x1 + k + 8, r1 >= 0);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile(words[i], lane, b0, b1);
            mma16816(acc[i][0], a, b0);
            mma16816(acc[i][1], a, b1);
        }
        if (more) {
#pragma unroll
            for (int i = 0; i < 8; ++i) words[i] = nw[i];
#pragma unroll
            for (int c = 0; c < 4; ++c) a[c] = na[c];
        }
        tile += (size_t)NTILES * 32;
    }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();

    if constexpr (0 == 1) {
        // down_epilogue_kernel on this block's sums: v = 0 + (the warps' sums in warp order), rotated, times svh
        const half* svh_d = sv0;
        float* y = reinterpret_cast<float*>(out);
        for (int i = warp; i < cnt; i += 4) {
            const int p = rows_sh[i];
            const int n = nt0 * 16 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = red[0][i][4 * lane + j];
#pragma unroll
                for (int w = 1; w < 4; ++w) s += red[w][i][4 * lane + j];
                float z = 0.f;
                z += s;
                v[j] = z;
            }
            fwht128(v, lane);
            float* o = y + (size_t)p * N + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * N + n + j]);
        }
        return;
    } else {
        constexpr int OUT = 16 * 8 * 16 / (4 * 32);
#pragma unroll
        for (int j = 0; j < OUT; ++j) {
            const int idx = threadIdx.x + j * 4 * 32;
            const int row = idx / (8 * 16), col = idx % (8 * 16);
            const int r = rows_sh[row];
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < 4; ++w) s += red[w][row][col];
            if (r >= 0) Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
        if constexpr (0 == 2) {
            __shared__ int last;
            __threadfence();                               // this block's partials before its count
            __syncthreads();
            if (threadIdx.x == 0) {
                int* c = done + (size_t)item * gridDim.y + nblock;
                last = atomicAdd(c, 1) == 2 * SK - 1;
                if (last) *c = 0;                          // the next launch starts from zero
            }
            __syncthreads();
            if (!last) return;
            __threadfence();                               // every block's partials are visible from here
            const half* svh_g = sv0;
            const half* svh_u = sv1;
            const half* suh_d = su2;
            half* xd = reinterpret_cast<half*>(out);
            for (int i = warp; i < cnt; i += 4) {
                const int p = rows_sh[i];
                const int n = nt0 * 16 + 4 * lane;
                float gv[4], uv[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float sg = 0.f, su = 0.f;
                    for (int s = 0; s < SK; ++s) {
                        sg += __ldcg(Z + ((size_t)(0 * SK + s) * P + p) * N + n + j);
                        su += __ldcg(Z + ((size_t)(1 * SK + s) * P + p) * N + n + j);
                    }
                    gv[j] = sg;
                    uv[j] = su;
                }
                fwht128(gv, lane);
                fwht128(uv, lane);
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
                    float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])),
                                           -limit), limit);
                    float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                    v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
                }
                fwht128(v, lane);
                half* o = xd + (size_t)p * N + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
            }
        }
    }
}
extern "C" __global__ void  __launch_bounds__(4 * 32) glm_exl3_dec_16_1_0(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int E, int slots,
    const half* __restrict__ sv0, const half* __restrict__ sv1, const half* __restrict__ su2, void* __restrict__ out,
    float limit, int* __restrict__ done) {
    static_assert(1 == 0 || 8 * 16 == 128, "a fused epilogue needs one 128-column Hadamard block a program");
    static_assert(1 == 0 || (8 % 4 == 0 && 1 >= 1 && 16 % 1 == 0), "1 1: 8 a multiple of 4, 1 | 16");
    int item = blockIdx.x, nblock = blockIdx.y, zz = blockIdx.z;
    if constexpr (0 != 0) {                            // launch order only: the blocks of one k slice run together
        const unsigned lid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        nblock = (int)(lid % gridDim.y);
        const unsigned r = lid / gridDim.y;
        if (0 == 1) { item = (int)(r % gridDim.x); zz = (int)(r / gridDim.x); }
        else { zz = (int)(r % gridDim.z); item = (int)(r / gridDim.z); }
    }
    int e, first, cnt;
    if constexpr (1 == 0) {
        if (item >= counts[0]) return;
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    } else {                                             // one round trip: items holds a row for every block
        const int total = counts[0];
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
        if (item >= total) return;
    }
    if (e >= E) return;                                  // the shared expert (id E) is not EXL3
    const int mat = zz / SK, split = zz % SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = mat ? T1 : T0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    __shared__ __align__(16) float red[4][16][8 * 16];
    constexpr int NV = 8 / 4;                           // 1 1: 16-byte loads a lane a k step
    uint4 rv[1 == 1 ? 1 : 1][1 == 1 ? NV : 1];
    const uint32_t* tbase = T + (((size_t)e * KT + split * (KT / SK) + warp * 16) * NTILES + nblock * 8) * 32;
    const size_t kstep = (size_t)NTILES * 32;            // words from one k tile to the next
    if constexpr (1 == 1) {
#pragma unroll
        for (int d = 0; d < 1; ++d)
#pragma unroll
            for (int v = 0; v < NV; ++v) rv[d][v] = ldg_nc_v4(tbase + d * kstep + 128 * v + 4 * lane);
    }
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const int xr0 = r0 < 0 ? 0 : (0 ? r0 / slots : r0), xr1 = r1 < 0 ? 0 : (0 ? r1 / slots : r1);
    const half* x0 = X + (size_t)xr0 * K + 2 * t;
    const half* x1 = X + (size_t)xr1 * K + 2 * t;
    const int nt0 = nblock * 8;
    const int kt0 = split * (KT / SK) + warp * 16;
    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;

    float acc[8][2][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    if constexpr (1 == 1) {
        uint32_t ra[1][4];
#pragma unroll
        for (int d = 0; d < 1; ++d) {
            const int k = (kt0 + d) * 16;
            ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
            ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
        }
        uint32_t* stage = reinterpret_cast<uint32_t*>(&red[warp][0][0]);
        for (int kb = 0; kb < 16; kb += 1) {
#pragma unroll
            for (int d = 0; d < 1; ++d) {
                const int s = kb + d;
                const bool more = s + 1 < 16;
                uint32_t a[4], words[8];
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = ra[d][c];
                __syncwarp();                            // every lane has read the previous step back
#pragma unroll
                for (int v = 0; v < NV; ++v) *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];
                if (more) {                              // the slot's registers are free again
#pragma unroll
                    for (int v = 0; v < NV; ++v)
                        rv[d][v] = ldg_nc_v4(tbase + (size_t)(s + 1) * kstep + 128 * v + 4 * lane);
                    const int k = (kt0 + s + 1) * 16;
                    ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
                    ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
                }
                __syncwarp();
#pragma unroll
                for (int i = 0; i < 8; ++i) words[i] = stage[i * 32 + lane];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile(words[i], lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
        __syncwarp();                                    // the staging area becomes red[warp] below
    } else {
    // grouped_kernel's PF loop as it is (a runtime loop: unrolling it measured slower)
    uint32_t words[8], a[4];
#pragma unroll
    for (int i = 0; i < 8; ++i) words[i] = __ldg(tile + i * 32);
    {
        const int k = kt0 * 16;
        a[0] = load_pair(x0 + k, r0 >= 0); a[1] = load_pair(x1 + k, r1 >= 0);
        a[2] = load_pair(x0 + k + 8, r0 >= 0); a[3] = load_pair(x1 + k + 8, r1 >= 0);
    }
    for (int kt = kt0; kt < kt0 + 16; ++kt) {
        uint32_t nw[8], na[4];
        const bool more = kt + 1 < kt0 + 16;
        if (more) {
            const uint32_t* next = tile + (size_t)NTILES * 32;
#pragma unroll
            for (int i = 0; i < 8; ++i) nw[i] = __ldg(next + i * 32);
            const int k = (kt + 1) * 16;
            na[0] = load_pair(x0 + k, r0 >= 0); na[1] = load_pair(x1 + k, r1 >= 0);
            na[2] = load_pair(x0 + k + 8, r0 >= 0); na[3] = load_pair(x1 + k + 8, r1 >= 0);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile(words[i], lane, b0, b1);
            mma16816(acc[i][0], a, b0);
            mma16816(acc[i][1], a, b1);
        }
        if (more) {
#pragma unroll
            for (int i = 0; i < 8; ++i) words[i] = nw[i];
#pragma unroll
            for (int c = 0; c < 4; ++c) a[c] = na[c];
        }
        tile += (size_t)NTILES * 32;
    }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();

    if constexpr (1 == 1) {
        // down_epilogue_kernel on this block's sums: v = 0 + (the warps' sums in warp order), rotated, times svh
        const half* svh_d = sv0;
        float* y = reinterpret_cast<float*>(out);
        for (int i = warp; i < cnt; i += 4) {
            const int p = rows_sh[i];
            const int n = nt0 * 16 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = red[0][i][4 * lane + j];
#pragma unroll
                for (int w = 1; w < 4; ++w) s += red[w][i][4 * lane + j];
                float z = 0.f;
                z += s;
                v[j] = z;
            }
            fwht128(v, lane);
            float* o = y + (size_t)p * N + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * N + n + j]);
        }
        return;
    } else {
        constexpr int OUT = 16 * 8 * 16 / (4 * 32);
#pragma unroll
        for (int j = 0; j < OUT; ++j) {
            const int idx = threadIdx.x + j * 4 * 32;
            const int row = idx / (8 * 16), col = idx % (8 * 16);
            const int r = rows_sh[row];
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < 4; ++w) s += red[w][row][col];
            if (r >= 0) Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
        if constexpr (1 == 2) {
            __shared__ int last;
            __threadfence();                               // this block's partials before its count
            __syncthreads();
            if (threadIdx.x == 0) {
                int* c = done + (size_t)item * gridDim.y + nblock;
                last = atomicAdd(c, 1) == 2 * SK - 1;
                if (last) *c = 0;                          // the next launch starts from zero
            }
            __syncthreads();
            if (!last) return;
            __threadfence();                               // every block's partials are visible from here
            const half* svh_g = sv0;
            const half* svh_u = sv1;
            const half* suh_d = su2;
            half* xd = reinterpret_cast<half*>(out);
            for (int i = warp; i < cnt; i += 4) {
                const int p = rows_sh[i];
                const int n = nt0 * 16 + 4 * lane;
                float gv[4], uv[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float sg = 0.f, su = 0.f;
                    for (int s = 0; s < SK; ++s) {
                        sg += __ldcg(Z + ((size_t)(0 * SK + s) * P + p) * N + n + j);
                        su += __ldcg(Z + ((size_t)(1 * SK + s) * P + p) * N + n + j);
                    }
                    gv[j] = sg;
                    uv[j] = su;
                }
                fwht128(gv, lane);
                fwht128(uv, lane);
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
                    float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])),
                                           -limit), limit);
                    float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                    v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
                }
                fwht128(v, lane);
                half* o = xd + (size_t)p * N + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
            }
        }
    }
}
extern "C" __global__ void  __launch_bounds__(4 * 32) glm_exl3_dec_16_0_1(
    const half* __restrict__ X0, const half* __restrict__ X1, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int E, int slots,
    const half* __restrict__ sv0, const half* __restrict__ sv1, const half* __restrict__ su2, void* __restrict__ out,
    float limit, int* __restrict__ done) {
    static_assert(0 == 0 || 8 * 16 == 128, "a fused epilogue needs one 128-column Hadamard block a program");
    static_assert(1 == 0 || (8 % 4 == 0 && 1 >= 1 && 16 % 1 == 0), "1 1: 8 a multiple of 4, 1 | 16");
    int item = blockIdx.x, nblock = blockIdx.y, zz = blockIdx.z;
    if constexpr (0 != 0) {                            // launch order only: the blocks of one k slice run together
        const unsigned lid = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
        nblock = (int)(lid % gridDim.y);
        const unsigned r = lid / gridDim.y;
        if (0 == 1) { item = (int)(r % gridDim.x); zz = (int)(r / gridDim.x); }
        else { zz = (int)(r % gridDim.z); item = (int)(r / gridDim.z); }
    }
    int e, first, cnt;
    if constexpr (1 == 0) {
        if (item >= counts[0]) return;
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    } else {                                             // one round trip: items holds a row for every block
        const int total = counts[0];
        e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
        if (item >= total) return;
    }
    if (e >= E) return;                                  // the shared expert (id E) is not EXL3
    const int mat = zz / SK, split = zz % SK;
    const half* X = mat ? X1 : X0;
    const uint32_t* T = mat ? T1 : T0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    __shared__ __align__(16) float red[4][16][8 * 16];
    constexpr int NV = 8 / 4;                           // 1 1: 16-byte loads a lane a k step
    uint4 rv[1 == 1 ? 1 : 1][1 == 1 ? NV : 1];
    const uint32_t* tbase = T + (((size_t)e * KT + split * (KT / SK) + warp * 16) * NTILES + nblock * 8) * 32;
    const size_t kstep = (size_t)NTILES * 32;            // words from one k tile to the next
    if constexpr (1 == 1) {
#pragma unroll
        for (int d = 0; d < 1; ++d)
#pragma unroll
            for (int v = 0; v < NV; ++v) rv[d][v] = ldg_nc_v4(tbase + d * kstep + 128 * v + 4 * lane);
    }
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const int xr0 = r0 < 0 ? 0 : (1 ? r0 / slots : r0), xr1 = r1 < 0 ? 0 : (1 ? r1 / slots : r1);
    const half* x0 = X + (size_t)xr0 * K + 2 * t;
    const half* x1 = X + (size_t)xr1 * K + 2 * t;
    const int nt0 = nblock * 8;
    const int kt0 = split * (KT / SK) + warp * 16;
    const uint32_t* tile = T + (((size_t)e * KT + kt0) * NTILES + nt0) * 32 + lane;

    float acc[8][2][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    if constexpr (1 == 1) {
        uint32_t ra[1][4];
#pragma unroll
        for (int d = 0; d < 1; ++d) {
            const int k = (kt0 + d) * 16;
            ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
            ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
        }
        uint32_t* stage = reinterpret_cast<uint32_t*>(&red[warp][0][0]);
        for (int kb = 0; kb < 16; kb += 1) {
#pragma unroll
            for (int d = 0; d < 1; ++d) {
                const int s = kb + d;
                const bool more = s + 1 < 16;
                uint32_t a[4], words[8];
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = ra[d][c];
                __syncwarp();                            // every lane has read the previous step back
#pragma unroll
                for (int v = 0; v < NV; ++v) *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];
                if (more) {                              // the slot's registers are free again
#pragma unroll
                    for (int v = 0; v < NV; ++v)
                        rv[d][v] = ldg_nc_v4(tbase + (size_t)(s + 1) * kstep + 128 * v + 4 * lane);
                    const int k = (kt0 + s + 1) * 16;
                    ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
                    ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
                }
                __syncwarp();
#pragma unroll
                for (int i = 0; i < 8; ++i) words[i] = stage[i * 32 + lane];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile(words[i], lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
        __syncwarp();                                    // the staging area becomes red[warp] below
    } else {
    // grouped_kernel's PF loop as it is (a runtime loop: unrolling it measured slower)
    uint32_t words[8], a[4];
#pragma unroll
    for (int i = 0; i < 8; ++i) words[i] = __ldg(tile + i * 32);
    {
        const int k = kt0 * 16;
        a[0] = load_pair(x0 + k, r0 >= 0); a[1] = load_pair(x1 + k, r1 >= 0);
        a[2] = load_pair(x0 + k + 8, r0 >= 0); a[3] = load_pair(x1 + k + 8, r1 >= 0);
    }
    for (int kt = kt0; kt < kt0 + 16; ++kt) {
        uint32_t nw[8], na[4];
        const bool more = kt + 1 < kt0 + 16;
        if (more) {
            const uint32_t* next = tile + (size_t)NTILES * 32;
#pragma unroll
            for (int i = 0; i < 8; ++i) nw[i] = __ldg(next + i * 32);
            const int k = (kt + 1) * 16;
            na[0] = load_pair(x0 + k, r0 >= 0); na[1] = load_pair(x1 + k, r1 >= 0);
            na[2] = load_pair(x0 + k + 8, r0 >= 0); na[3] = load_pair(x1 + k + 8, r1 >= 0);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile(words[i], lane, b0, b1);
            mma16816(acc[i][0], a, b0);
            mma16816(acc[i][1], a, b1);
        }
        if (more) {
#pragma unroll
            for (int i = 0; i < 8; ++i) words[i] = nw[i];
#pragma unroll
            for (int c = 0; c < 4; ++c) a[c] = na[c];
        }
        tile += (size_t)NTILES * 32;
    }
    }

#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();

    if constexpr (0 == 1) {
        // down_epilogue_kernel on this block's sums: v = 0 + (the warps' sums in warp order), rotated, times svh
        const half* svh_d = sv0;
        float* y = reinterpret_cast<float*>(out);
        for (int i = warp; i < cnt; i += 4) {
            const int p = rows_sh[i];
            const int n = nt0 * 16 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = red[0][i][4 * lane + j];
#pragma unroll
                for (int w = 1; w < 4; ++w) s += red[w][i][4 * lane + j];
                float z = 0.f;
                z += s;
                v[j] = z;
            }
            fwht128(v, lane);
            float* o = y + (size_t)p * N + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * N + n + j]);
        }
        return;
    } else {
        constexpr int OUT = 16 * 8 * 16 / (4 * 32);
#pragma unroll
        for (int j = 0; j < OUT; ++j) {
            const int idx = threadIdx.x + j * 4 * 32;
            const int row = idx / (8 * 16), col = idx % (8 * 16);
            const int r = rows_sh[row];
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < 4; ++w) s += red[w][row][col];
            if (r >= 0) Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
        if constexpr (0 == 2) {
            __shared__ int last;
            __threadfence();                               // this block's partials before its count
            __syncthreads();
            if (threadIdx.x == 0) {
                int* c = done + (size_t)item * gridDim.y + nblock;
                last = atomicAdd(c, 1) == 2 * SK - 1;
                if (last) *c = 0;                          // the next launch starts from zero
            }
            __syncthreads();
            if (!last) return;
            __threadfence();                               // every block's partials are visible from here
            const half* svh_g = sv0;
            const half* svh_u = sv1;
            const half* suh_d = su2;
            half* xd = reinterpret_cast<half*>(out);
            for (int i = warp; i < cnt; i += 4) {
                const int p = rows_sh[i];
                const int n = nt0 * 16 + 4 * lane;
                float gv[4], uv[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float sg = 0.f, su = 0.f;
                    for (int s = 0; s < SK; ++s) {
                        sg += __ldcg(Z + ((size_t)(0 * SK + s) * P + p) * N + n + j);
                        su += __ldcg(Z + ((size_t)(1 * SK + s) * P + p) * N + n + j);
                    }
                    gv[j] = sg;
                    uv[j] = su;
                }
                fwht128(gv, lane);
                fwht128(uv, lane);
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
                    float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])),
                                           -limit), limit);
                    float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                    v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
                }
                fwht128(v, lane);
                half* o = xd + (size_t)p * N + n;
#pragma unroll
                for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
            }
        }
    }
}
extern "C" __global__ void  __launch_bounds__(1024) glm_exl3_order(const int* __restrict__ items,
                                                            const int* __restrict__ counts, int* __restrict__ order,
                                                            int cap, int E) {
    __shared__ int wsum[32];
    __shared__ int routed;
    const int all = min(counts[0], cap);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) routed = 0;
    __syncthreads();
    for (int i = tid; i < all; i += 1024)               // items are in expert order: the routed ones come first
        if (items[3 * i] < E && (i + 1 == all || items[3 * (i + 1)] >= E)) routed = i + 1;
    __syncthreads();
    const int n = routed;
    const int per = (n + 1023) / 1024;
    const int lo = min(n, tid * per), hi = min(n, lo + per);
    auto heavy = [&](int i) {
        const int e = items[3 * i];
        return (i > 0 && items[3 * (i - 1)] == e) || (i + 1 < n && items[3 * (i + 1)] == e);
    };
    int h = 0;
    for (int i = lo; i < hi; ++i) h += heavy(i);
    int v = h;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += u;
    }
    if (lane == 31) wsum[warp] = v;
    __syncthreads();
    if (warp == 0) {
        int w = wsum[lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += u;
        }
        wsum[lane] = w;
    }
    __syncthreads();
    const int nh = wsum[31], nl = n - nh;
    int rh = (warp ? wsum[warp - 1] : 0) + v - h;       // heavy items before lo
    for (int i = lo; i < hi; ++i) {
        if (heavy(i)) {                                // heavy rank r sits at (r + 1/2) / nh; lights before it: strictly
            const int r = rh++;
            const long long X = (2LL * r + 1) * nl;
            const int cl = X > nh ? (int)min((long long)nl, (X - nh + 2LL * nh - 1) / (2LL * nh)) : 0;
            order[r + cl] = i;
        } else {                                       // light rank l at (l + 1/2) / nl; heavies at or before it
            const int l = i - rh;
            const long long Y = (2LL * l + 1) * nh;
            const int ch = Y >= nl ? (int)min((long long)nh, (Y - nl) / (2LL * nl) + 1) : 0;
            order[l + ch] = i;
        }
    }
    for (int i = n + tid; i < all; i += 1024) order[i] = -1;
}
extern "C" __global__ void  __launch_bounds__(8 * 32, 16 / 8) glm_exl3_prompt_gu_0(
    const half* __restrict__ Xg, const half* __restrict__ Xu, const uint32_t* __restrict__ Tg,
    const uint32_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ svh_u,
    const half* __restrict__ suh_d, half* __restrict__ xd, int K, int N, int E, float limit, int slots,
    const int* __restrict__ order) {
    constexpr int MT = 16 / 8, PH = 64 / 2;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __shared__ int rows_sh[64];
    __shared__ int xrows_sh[0 ? 64 : 1];              // 0: the members' token rows
    const int cb = blockIdx.x;
    if ((int)blockIdx.y >= counts[0]) return;
    const int item = order ? order[blockIdx.y] : blockIdx.y;
    if (item < 0) return;                                // ORDER leaves the shared expert's items out
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (e >= E) return;                                  // the shared expert
    if (threadIdx.x < 64) {
        const int p = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
        rows_sh[threadIdx.x] = p;
        if (0) xrows_sh[threadIdx.x] = p < 0 ? -1 : p / slots;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[MT][64 / 8][4];
    prompt_mainloop<2, 64, 8, 0>(Xg, Xu, Tg, Tu, 0 ? xrows_sh : rows_sh, cnt, e, K, N >> 4, cb,
                                    reinterpret_cast<half*>(smem_raw), acc);
    float* Eg = reinterpret_cast<float*>(smem_raw);
    int mat, nt;
    prompt_tile<2>(warp * MT, cb, mat, nt);
    const int col0 = (nt - cb * 8) * 16;
    const int n = cb * 128 + 4 * lane;
    for (int hf = 0; hf < 2 && hf * PH < cnt; ++hf) {
    if (hf) __syncthreads();                             // the first half's rows are read
    prompt_spill<MT, 64>(Eg + (size_t)mat * PH * GU_ES, GU_ES, col0, acc, hf);
    __syncthreads();
    for (int i = warp; i < PH && hf * PH + i < cnt; i += 8) {
        const int p = rows_sh[hf * PH + i];
        float gv[4], uv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            gv[j] = Eg[(size_t)i * GU_ES + 4 * lane + j];
            uv[j] = Eg[(size_t)(PH + i) * GU_ES + 4 * lane + j];
        }
        fwht128(gv, lane);
        fwht128(uv, lane);
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
            float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit),
                             limit);
            float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
            v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
        }
        fwht128(v, lane);
        const half2 o01 = __halves2half2(__float2half_rn(v[0] * HAD_SCALE), __float2half_rn(v[1] * HAD_SCALE));
        const half2 o23 = __halves2half2(__float2half_rn(v[2] * HAD_SCALE), __float2half_rn(v[3] * HAD_SCALE));
        uint2 packed;
        packed.x = *reinterpret_cast<const uint32_t*>(&o01);
        packed.y = *reinterpret_cast<const uint32_t*>(&o23);
        *reinterpret_cast<uint2*>(xd + (size_t)p * N + n) = packed;
    }
    }  // halves
}
extern "C" __global__ void  __launch_bounds__(8 * 32, 16 / 8) glm_exl3_prompt_gu_1(
    const half* __restrict__ Xg, const half* __restrict__ Xu, const uint32_t* __restrict__ Tg,
    const uint32_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ svh_u,
    const half* __restrict__ suh_d, half* __restrict__ xd, int K, int N, int E, float limit, int slots,
    const int* __restrict__ order) {
    constexpr int MT = 16 / 8, PH = 64 / 2;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __shared__ int rows_sh[64];
    __shared__ int xrows_sh[1 ? 64 : 1];              // 1: the members' token rows
    const int cb = blockIdx.x;
    if ((int)blockIdx.y >= counts[0]) return;
    const int item = order ? order[blockIdx.y] : blockIdx.y;
    if (item < 0) return;                                // ORDER leaves the shared expert's items out
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (e >= E) return;                                  // the shared expert
    if (threadIdx.x < 64) {
        const int p = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
        rows_sh[threadIdx.x] = p;
        if (1) xrows_sh[threadIdx.x] = p < 0 ? -1 : p / slots;
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[MT][64 / 8][4];
    prompt_mainloop<2, 64, 8, 1>(Xg, Xu, Tg, Tu, 1 ? xrows_sh : rows_sh, cnt, e, K, N >> 4, cb,
                                    reinterpret_cast<half*>(smem_raw), acc);
    float* Eg = reinterpret_cast<float*>(smem_raw);
    int mat, nt;
    prompt_tile<2>(warp * MT, cb, mat, nt);
    const int col0 = (nt - cb * 8) * 16;
    const int n = cb * 128 + 4 * lane;
    for (int hf = 0; hf < 2 && hf * PH < cnt; ++hf) {
    if (hf) __syncthreads();                             // the first half's rows are read
    prompt_spill<MT, 64>(Eg + (size_t)mat * PH * GU_ES, GU_ES, col0, acc, hf);
    __syncthreads();
    for (int i = warp; i < PH && hf * PH + i < cnt; i += 8) {
        const int p = rows_sh[hf * PH + i];
        float gv[4], uv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            gv[j] = Eg[(size_t)i * GU_ES + 4 * lane + j];
            uv[j] = Eg[(size_t)(PH + i) * GU_ES + 4 * lane + j];
        }
        fwht128(gv, lane);
        fwht128(uv, lane);
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j])), limit);
            float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j])), -limit),
                             limit);
            float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
            v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
        }
        fwht128(v, lane);
        const half2 o01 = __halves2half2(__float2half_rn(v[0] * HAD_SCALE), __float2half_rn(v[1] * HAD_SCALE));
        const half2 o23 = __halves2half2(__float2half_rn(v[2] * HAD_SCALE), __float2half_rn(v[3] * HAD_SCALE));
        uint2 packed;
        packed.x = *reinterpret_cast<const uint32_t*>(&o01);
        packed.y = *reinterpret_cast<const uint32_t*>(&o23);
        *reinterpret_cast<uint2*>(xd + (size_t)p * N + n) = packed;
    }
    }  // halves
}
extern "C" __global__ void  __launch_bounds__(8 * 32, 16 / 8) glm_exl3_prompt_down(
    const half* __restrict__ Xd, const uint32_t* __restrict__ Td, const int* __restrict__ items,
    const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d,
    float* __restrict__ y, int K, int D, int E, const int* __restrict__ order) {
    constexpr int MT = 16 / 8, PH = 64 / 2;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __shared__ int rows_sh[64];
    const int cb = blockIdx.x;
    if ((int)blockIdx.y >= counts[0]) return;
    const int item = order ? order[blockIdx.y] : blockIdx.y;
    if (item < 0) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (e >= E) return;
    if (threadIdx.x < 64) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    float acc[MT][64 / 8][4];
    prompt_mainloop<1, 64, 8>(Xd, Xd, Td, Td, rows_sh, cnt, e, K, D >> 4, cb, reinterpret_cast<half*>(smem_raw),
                               acc);
    float* Ed = reinterpret_cast<float*>(smem_raw);
    for (int hf = 0; hf < 2 && hf * PH < cnt; ++hf) {
    if (hf) __syncthreads();
    prompt_spill<MT, 64>(Ed, DN_ES, warp * MT * 16, acc, hf);
    __syncthreads();
    for (int i = warp; i < PH && hf * PH + i < cnt; i += 8) {
        const int p = rows_sh[hf * PH + i];
#pragma unroll
        for (int blk = 0; blk < 2; ++blk) {
            const int n = cb * 256 + blk * 128 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) v[j] = Ed[(size_t)i * DN_ES + blk * 128 + 4 * lane + j];
            fwht128(v, lane);
            const uint2 sv = *reinterpret_cast<const uint2*>(svh_d + (size_t)e * D + n);
            const half2 s01 = *reinterpret_cast<const half2*>(&sv.x), s23 = *reinterpret_cast<const half2*>(&sv.y);
            *reinterpret_cast<float4*>(y + (size_t)p * D + n) =
                make_float4(v[0] * HAD_SCALE * __low2float(s01), v[1] * HAD_SCALE * __high2float(s01),
                            v[2] * HAD_SCALE * __low2float(s23), v[3] * HAD_SCALE * __high2float(s23));
        }
    }
    }  // halves
}

} // namespace
