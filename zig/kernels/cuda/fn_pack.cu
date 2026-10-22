// Flash Next's n-gram table on the GPU (ours, flashnext/cuda_weights.zig NgramTable.gpu): a rank holds the rows of
// its n-gram heads [head0, head0 + heads) as stored (FP8 e4m3 codes, or bf16), global rows [base, base + count), and
// gathers a window's rows into bf16 bits [tokens][heads * width], the values host_table.FP8Table.gather gives
// (code -> lut[code] = bf16_rne(e4m3 x weight_scale), NaN codes 0x7FC0). Pure data movement: no arithmetic, so the
// bits are the host gather's. An id outside the rank's rows writes 0x7FC0 (the host gather refuses it).
#include <cstdint>

namespace {

__device__ __forceinline__ bool split(long long i, int tokens, int heads, int width, long long* t, int* h, int* c) {
    const long long total = (long long)tokens * heads * width;
    if (i >= total) return false;
    *c = (int)(i % width);
    const long long th = i / width;
    *h = (int)(th % heads);
    *t = th / heads;
    return true;
}

}  // namespace

// ids: int64 [tokens][ids_stride] global row ids (every head's; this rank reads heads head0 .. head0 + heads - 1)
extern "C" __global__ void fn_ngram_gather(const uint8_t* __restrict__ rows, const uint16_t* __restrict__ lut,
                                           const long long* __restrict__ ids, int ids_stride, int head0, int heads,
                                           long long base, long long count, int width, int tokens,
                                           uint16_t* __restrict__ out) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long t;
    int h, c;
    if (!split(i, tokens, heads, width, &t, &h, &c)) return;
    const long long id = ids[t * ids_stride + head0 + h] - base;
    out[i] = (id >= 0 && id < count) ? lut[rows[id * width + c]] : (uint16_t)0x7FC0;
}

// the same over a bf16 table (rows as stored)
extern "C" __global__ void fn_ngram_gather_bf16(const uint16_t* __restrict__ rows, const long long* __restrict__ ids,
                                                int ids_stride, int head0, int heads, long long base, long long count,
                                                int width, int tokens, uint16_t* __restrict__ out) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long t;
    int h, c;
    if (!split(i, tokens, heads, width, &t, &h, &c)) return;
    const long long id = ids[t * ids_stride + head0 + h] - base;
    out[i] = (id >= 0 && id < count) ? rows[id * width + c] : (uint16_t)0x7FC0;
}
