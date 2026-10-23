// GLM-5.3-Flash TF_GLM_KV_SHARD (zig/src/families/glm_cuda/cuda_shard.zig, cuda_shard_decode.zig,
// cuda_shard_prompt.zig): port of wojo's GLM 0118 (PR #152, glm5_next/cuda/shard.py) to the Zig engine. The byte
// movers owner-computed decode needs around its two passes of the captured H=32 attention kernels: the passes' token
// counts (a row this rank does not own reads no key) and the select that keeps this rank's heads of every row (its own
// pass's for the rows it owns, the owner's pass with its queries for the others); and a prompt chunk's history fetch:
// the union of the positions its tokens name, marked, counted, compacted and renumbered. Nothing here computes on the
// data: every byte lands unchanged and every id is a count.
// Built like glm_ops (torch_ops flags); raw device pointers, the caller's stream. Written on a host without nvcc:
// tools/glm_shard_emulate runs the kernels on the CPU against a plain reference and checks the device code's syntax
// with clang's CUDA front end; the GPU checks are docs/glm-port/gpu-validation.md's.
#include <cuda_runtime.h>
#include <stdint.h>

// shard.attention's masked counts (ce = counts * own): out[r] = counts[r] for a row this rank owns (own[r] != 0), 0 for
// one it does not (its stream's latent rows are the other rank's; with the glue's sparse_eff 1 the pass reads none).
// int32 [rows]. Grid (1), 64 threads (a decode window's rows; more loop).
extern "C" __global__ void __launch_bounds__(64) glm_shard_counts(const int* counts, const int* own, int* out, int rows) {
    for (int r = threadIdx.x; r < rows; r += blockDim.x) out[r] = own[r] != 0 ? counts[r] : 0;
}

// shard.attention's torch.where(own, mine, peer): row r of `out` (this rank's heads, its own pass's output) takes row
// r of `recv` (the owner's pass with this rank's queries) where own[r] == 0; the rows it owns are left as they are.
// Rows of `words` 16-byte words, both 16-byte aligned. Grid (rows), 256 threads.
extern "C" __global__ void __launch_bounds__(256) glm_shard_select(const uint4* recv, uint4* out, const int* own,
                                                                   long long words) {
    const long long r = blockIdx.x;
    if (own[r] != 0) return;
    const long long at = r * words;
    for (long long w = threadIdx.x; w < words; w += blockDim.x) out[at + w] = recv[at + w];
}

// ---- a prompt chunk's history fetch (shard.py history / fetch, forward.py _sharded_attend; cuda_shard.zig's
// markHistory, prefixSum and renumber and cuda_shard_prompt.zig's compactRef are the host models) ----
// The owner of a stream's latent rows sends the other rank the history rows a chunk's tokens name, once each and in
// position order; the other rank attends over that compact copy with the tokens renumbered to its rows, so every row
// reads the same bytes in the same order. Positions are marked, the marks scanned in place (cum[t]: the marked
// positions up to t), the owner's rows compacted by cum and the other rank's tokens renumbered by it.

// mark[t] = 1 for every token t below `limit` of tokens [n] (int32, -1 none: a chunk's rows of 2,051 picks), the marks
// zeroed first (cuMemsetD32). Grid (ceil(n / 256)), 256 threads.
extern "C" __global__ void __launch_bounds__(256) glm_shard_mark(const int* tokens, long long n, int* mark, int limit) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const int t = tokens[i];
    if (t >= 0 && t < limit) mark[t] = 1;
}

// The marks' inclusive prefix sums in place (a[n]), in three launches over GLM_SHARD_SCAN_SEGS segments of
// ceil(n / segs) entries with no barrier inside one (DeepSeek's dsv41_shard.cu union scan): sums[s] = segment s's
// total; one thread turns them into exclusive offsets; each segment rescans itself from its offset, the last writing
// the total to sums[segs] (the count the host reads). sums: int32 [segs + 1]. Grids (segs / 256) x 256 threads,
// (1) x 1 thread, (segs / 256) x 256 threads.
#define GLM_SHARD_SCAN_SEGS 1024
extern "C" __global__ void __launch_bounds__(256) glm_shard_sums(const int* a, long long n, int* sums) {
    const long long s = (long long)blockIdx.x * 256 + threadIdx.x;
    if (s >= GLM_SHARD_SCAN_SEGS) return;
    const long long per = (n + GLM_SHARD_SCAN_SEGS - 1) / GLM_SHARD_SCAN_SEGS;
    const long long lo = s * per;
    const long long hi = lo + per < n ? lo + per : n;
    int t = 0;
    for (long long i = lo; i < hi; i++) t += a[i];
    sums[s] = t;
}

extern "C" __global__ void glm_shard_offsets(int* sums) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int run = 0;
    for (int s = 0; s < GLM_SHARD_SCAN_SEGS; s++) {
        const int v = sums[s];
        sums[s] = run;
        run += v;
    }
}

extern "C" __global__ void __launch_bounds__(256) glm_shard_scan(int* a, long long n, int* sums) {
    const long long s = (long long)blockIdx.x * 256 + threadIdx.x;
    if (s >= GLM_SHARD_SCAN_SEGS) return;
    const long long per = (n + GLM_SHARD_SCAN_SEGS - 1) / GLM_SHARD_SCAN_SEGS;
    const long long lo = s * per;
    const long long hi = lo + per < n ? lo + per : n;
    int run = sums[s];
    for (long long i = lo; i < hi; i++) {
        run += a[i];
        a[i] = run;
    }
    if (s == GLM_SHARD_SCAN_SEGS - 1) sums[GLM_SHARD_SCAN_SEGS] = run;
}

// The owner's copy of the union: latent row t of the stream's view (`words` 16-byte words a row: 33, an FP8 row's 528
// bytes) into row cum[t] - 1 of `dst` for each marked t below `limit` (where cum steps); cum null: every row t to row
// t (a chunk with rows under the dense limit fetches its whole history at its positions). A kernel, not a copy: the
// plane may lie in the display span, whose two registrations a memcpy or a transport may not cross. Grid
// (ceil(limit / 16)), 256 threads: 16 rows a block, a row's 16 lanes moving its words.
extern "C" __global__ void __launch_bounds__(256) glm_shard_compact(const uint4* src, const int* cum, long long limit,
                                                                    long long words, uint4* dst) {
    const long long t = (long long)blockIdx.x * 16 + threadIdx.x / 16;
    const int lane = threadIdx.x % 16;
    if (t >= limit) return;
    long long d = t;
    if (cum != nullptr) {
        const int prev = t > 0 ? cum[t - 1] : 0;
        if (cum[t] == prev) return;
        d = prev;
    }
    const uint4* s = src + t * words;
    uint4* o = dst + d * words;
    for (long long w = lane; w < words; w += 16) o[w] = s[w];
}

// The other rank's tokens [n] renumbered in place to rows of its compact copy: a history token t (below `limit`) to
// cum[t] - 1, its place among the fetched rows; a chunk row's own (t at or past `limit`) to the fetched count
// cum[limit - 1] plus t - limit, after them; -1 kept. Grid (ceil(n / 256)), 256 threads.
extern "C" __global__ void __launch_bounds__(256) glm_shard_renumber(int* tokens, long long n, const int* cum, int limit) {
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const int t = tokens[i];
    if (t < 0) return;
    const int fetched = limit > 0 ? cum[limit - 1] : 0;
    tokens[i] = t < limit ? cum[t] - 1 : fetched + (t - limit);
}
