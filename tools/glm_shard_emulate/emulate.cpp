// TF_GLM_KV_SHARD kernels (glm_shard.cu: owner decode's masked counts and select; a prompt chunk's history fetch:
// mark, scan, compact, renumber) run on the host, every block and thread in turn, against a plain reference
// (cuda_shard_decode.zig's countsRef / selectRef, cuda_shard.zig's markHistory / prefixSum / renumber and
// cuda_shard_prompt.zig's compactRef are the same rules).
// No nvcc or GPU: clang++ -std=c++20 -I tools/glm_shard_emulate tools/glm_shard_emulate/emulate.cpp -o <out> && <out>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include "../../zig/kernels/cuda/glm_shard.cu"

// Every thread of every block in turn, blocks and threads in a shuffled order (the kernels never wait on another
// thread, and no two threads of a launch write one place: any order must give the same result).
static std::mt19937_64 order(0x118);
template <class K> static void run(unsigned gx, unsigned bx, K&& k) {
  gridDim = {gx, 1, 1};
  blockDim = {bx, 1, 1};
  std::vector<unsigned> bs(gx), ts(bx);
  for (unsigned i = 0; i < gx; i++) bs[i] = i;
  for (unsigned i = 0; i < bx; i++) ts[i] = i;
  std::shuffle(bs.begin(), bs.end(), order);
  for (unsigned b : bs) {
    std::shuffle(ts.begin(), ts.end(), order);
    for (unsigned t : ts) {
      blockIdx = {b, 0, 0};
      threadIdx = {t, 0, 0};
      k();
    }
  }
}

// -- a prompt chunk's history fetch --------------------------------------------------------------------------------

static const int W = 2051;  // a row's tokens
static const long long ROW16 = 33;  // an FP8 latent row's 528 bytes in 16-byte words

static void mark(const std::vector<int>& tok, long long n, std::vector<int>& m, int limit) {
  std::fill(m.begin(), m.begin() + limit, 0);  // the caller's cuMemsetD32
  run((unsigned)((n + 255) / 256), 256, [&] { glm_shard_mark(tok.data(), n, m.data(), limit); });
}
static int scan(std::vector<int>& m, long long n, std::vector<int>& sums) {
  run(GLM_SHARD_SCAN_SEGS / 256, 256, [&] { glm_shard_sums(m.data(), n, sums.data()); });
  run(1, 1, [&] { glm_shard_offsets(sums.data()); });
  run(GLM_SHARD_SCAN_SEGS / 256, 256, [&] { glm_shard_scan(m.data(), n, sums.data()); });
  return sums[GLM_SHARD_SCAN_SEGS];
}

// A chunk's tokens as the selection leaves them: every row `count` picks among positions at or before its own (history
// and the chunk's earlier rows), ascending, -1 past them; a row now and then with none.
static std::vector<int> tokens(std::mt19937_64& g, int rows, int pos, int most) {
  std::vector<int> t((size_t)rows * W, -1);
  for (int r = 0; r < rows; r++) {
    const int q = pos + r;
    const int count = g() % 9 == 0 ? 0 : 1 + (int)(g() % (unsigned)most);
    std::vector<int> pick;
    for (int i = 0; i < count; i++) pick.push_back(g() % 5 == 0 && q > pos ? pos + (int)(g() % (unsigned)(q - pos + 1)) : (int)(g() % (unsigned)(q + 1)));
    std::sort(pick.begin(), pick.end());
    for (int i = 0; i < count; i++) t[(size_t)r * W + i] = pick[i];
  }
  return t;
}

static int fetch(std::mt19937_64& g) {
  int fails = 0;
  // mark: every token below the limit, nothing else (-1, the chunk's own rows past it); any n
  for (int trial = 0; trial < 60; trial++) {
    const int rows = 1 + g() % 12, limit = (int)(g() % 6000);
    std::vector<int> tok = tokens(g, rows, limit, 40);
    const long long n = trial % 5 == 0 ? (long long)(g() % (rows * W)) : (long long)rows * W;
    std::vector<int> m(limit + 1, 7), want(limit + 1, 7);
    std::fill(want.begin(), want.begin() + limit, 0);
    for (long long i = 0; i < n; i++)
      if (tok[i] >= 0 && tok[i] < limit) want[tok[i]] = 1;
    mark(tok, n, m, limit);
    if (m != want) { fails++; printf("mark trial %d differs\n", trial); }
  }
  // scan: inclusive prefix sums in place, the total after the segments' sums; empty segments, n past segs * 256
  for (long long n : {0LL, 1LL, 7LL, 255LL, 1023LL, 1024LL, 1025LL, 5000LL, 262147LL, 1000003LL}) {
    std::vector<int> a(n + 1, 9), want(n + 1, 9);
    for (long long i = 0; i < n; i++) a[i] = g() % 3 == 0 ? 1 : 0;
    int run_ = 0;
    for (long long i = 0; i < n; i++) want[i] = run_ += a[i];
    std::vector<int> sums(GLM_SHARD_SCAN_SEGS + 1, -5);
    const int total = scan(a, n, sums);
    if (a != want || total != run_) { fails++; printf("scan of %lld differs (total %d, want %d)\n", n, total, run_); }
  }
  // compact: marked rows in position order (cum), or every row at its position (null); nothing past them written
  for (int trial = 0; trial < 60; trial++) {
    const long long limit = trial == 0 ? 0 : 1 + g() % 3000;
    std::vector<uint4> plane((limit + 1) * ROW16), dst((limit + 2) * ROW16), want;
    for (auto& w : plane) w = {(unsigned)g(), (unsigned)g(), (unsigned)g(), (unsigned)g()};
    for (auto& w : dst) w = {7, 7, 7, 7};
    want = dst;
    const bool whole = trial % 4 == 0;
    std::vector<int> cum(limit + 1);
    long long k = 0;
    for (long long t = 0; t < limit; t++) {
      const bool on = whole || g() % 3 == 0;
      if (on) memcpy(&want[(whole ? t : k) * ROW16], &plane[t * ROW16], ROW16 * 16);
      k += on;
      cum[t] = (int)k;
    }
    run((unsigned)((limit + 15) / 16), 256, [&] { glm_shard_compact(plane.data(), whole ? nullptr : cum.data(), limit, ROW16, dst.data()); });
    if (memcmp(dst.data(), want.data(), dst.size() * 16)) { fails++; printf("compact trial %d differs\n", trial); }
  }
  // renumber: a history token to its place among the marked, a chunk row's own after them, -1 kept, in place
  for (int trial = 0; trial < 60; trial++) {
    const int rows = 1 + g() % 12, limit = trial == 0 ? 0 : (int)(g() % 6000);
    std::vector<int> tok = tokens(g, rows, limit, 40), want = tok;
    const long long n = (long long)rows * W;
    std::vector<int> m(limit + 1), sums(GLM_SHARD_SCAN_SEGS + 1);
    mark(tok, n, m, limit);
    const int fetched = scan(m, limit, sums);
    for (auto& t : want)
      if (t >= 0) t = t < limit ? m[t] - 1 : fetched + (t - limit);
    run((unsigned)((n + 255) / 256), 256, [&] { glm_shard_renumber(tok.data(), n, m.data(), limit); });
    if (tok != want) { fails++; printf("renumber trial %d differs\n", trial); }
  }
  // the fetch as the two ranks compose it: the owner marks, scans and compacts its rows; the other rank marks, scans,
  // renumbers its tokens and writes the chunk's own rows after the fetched; every token then names, in the copy, the
  // bytes the owner's plane holds at the original position (whole-history chunks: at the same positions)
  for (int trial = 0; trial < 40; trial++) {
    const int rows = 1 + g() % 16;
    const bool whole = trial % 5 == 0;
    const int limit = whole ? (int)(g() % 2051) : 2051 + (int)(g() % 6000);
    std::vector<int> tok = tokens(g, rows, limit, whole ? 64 : 300), orig = tok;
    const long long n = (long long)rows * W;
    std::vector<uint4> plane(((size_t)limit + rows) * ROW16);  // the owner's view: history, then the chunk's own rows
    for (auto& w : plane) w = {(unsigned)g(), (unsigned)g(), (unsigned)g(), (unsigned)g()};
    std::vector<int> m[2], sums[2];
    std::vector<uint4> copy[2];
    int count[2];
    for (int k = 0; k < 2; k++) {  // k 0: the owner, 1: the other rank (the same tokens: the selection is replicated)
      m[k].assign(limit + 1, 0);
      sums[k].assign(GLM_SHARD_SCAN_SEGS + 1, 0);
      copy[k].assign(((size_t)limit + rows + 1) * ROW16, uint4{1, 2, 3, 4});
      if (whole) {
        count[k] = limit;
      } else {
        mark(tok, n, m[k], limit);
        count[k] = scan(m[k], limit, sums[k]);
      }
    }
    run((unsigned)((limit + 15) / 16), 256, [&] { glm_shard_compact(plane.data(), whole ? nullptr : m[0].data(), limit, ROW16, copy[0].data()); });
    memcpy(copy[1].data(), copy[0].data(), (size_t)count[0] * ROW16 * 16);  // the send: count rows, the receive the same
    memcpy(&copy[1][(size_t)count[1] * ROW16], &plane[(size_t)limit * ROW16], (size_t)rows * ROW16 * 16);  // own rows
    if (!whole) run((unsigned)((n + 255) / 256), 256, [&] { glm_shard_renumber(tok.data(), n, m[1].data(), limit); });
    bool ok = count[0] == count[1];
    for (long long i = 0; ok && i < n; i++) {
      if (orig[i] < 0) { ok = tok[i] == -1; continue; }
      ok = tok[i] >= 0 && tok[i] < count[1] + rows && !memcmp(&copy[1][(size_t)tok[i] * ROW16], &plane[(size_t)orig[i] * ROW16], ROW16 * 16);
      if (ok && i > 0 && orig[i - 1] >= 0 && i % W != 0) ok = (orig[i - 1] < orig[i]) == (tok[i - 1] < tok[i]);  // order kept
    }
    if (!ok) { fails++; printf("fetch trial %d: a token reads other bytes in the copy\n", trial); }
  }
  return fails;
}

int main() {
  std::mt19937_64 g(118);
  int fails = 0;
  // the masked counts: a row it owns keeps its count, one it does not reads none; any block size covers every row
  for (int trial = 0; trial < 300; trial++) {
    const int rows = 1 + g() % 64;
    std::vector<int> counts(rows), own(rows), out(rows + 4, 77), want(rows + 4, 77);
    for (int r = 0; r < rows; r++) {
      counts[r] = (int)(g() % 513);
      own[r] = g() % 3 == 0 ? 0 : (g() % 4 == 0 ? (int)(g() % 5) + 1 : 1);  // any nonzero is owned
      want[r] = own[r] != 0 ? counts[r] : 0;
    }
    const unsigned threads = trial % 3 == 0 ? 64 : 1 + g() % 80;
    run(1, threads, [&] { glm_shard_counts(counts.data(), own.data(), out.data(), rows); });
    if (out != want) { fails++; printf("counts trial %d differs (%d rows, %u threads)\n", trial, rows, threads); }
  }
  // the select: rows it does not own take the received row, its own rows and everything past `rows` stay untouched
  for (int trial = 0; trial < 200; trial++) {
    const int rows = 1 + g() % 64;
    const long long words = trial % 4 == 0 ? 2048 : 1 + g() % 600;  // 2048: a row of 32 heads x 512 bf16
    std::vector<uint4> recv(rows * words), out((rows + 1) * words), want;
    for (auto& w : recv) w = {(unsigned)g(), (unsigned)g(), (unsigned)g(), (unsigned)g()};
    for (auto& w : out) w = {(unsigned)g(), (unsigned)g(), (unsigned)g(), (unsigned)g()};
    std::vector<int> own(rows);
    for (auto& o : own) o = g() % 2;
    if (trial % 7 == 0) for (auto& o : own) o = 1;  // every row its own: nothing moves
    if (trial % 11 == 0) for (auto& o : own) o = 0;  // none: every row the owner's
    want = out;
    for (int r = 0; r < rows; r++)
      if (own[r] == 0) memcpy(&want[r * words], &recv[r * words], words * sizeof(uint4));
    const unsigned threads = trial % 2 ? 256 : 1 + g() % 300;
    run(rows, threads, [&] { glm_shard_select(recv.data(), out.data(), own.data(), words); });
    if (memcmp(out.data(), want.data(), out.size() * sizeof(uint4))) { fails++; printf("select trial %d differs\n", trial); }
  }
  // the two together on two ranks, as owner decode composes them: rank k's result row r is its own pass's when it owns
  // the row, else the owner's pass with rank k's queries (here: a row's bytes tagged by the rank that computed them)
  for (int trial = 0; trial < 100; trial++) {
    const int rows = 1 + g() % 64;
    const long long words = 1 + g() % 64;
    std::vector<int> orank(rows);
    for (auto& o : orank) o = g() % 2;
    std::vector<uint4> pass[2][2];  // pass[computed on][heads of]
    for (int c = 0; c < 2; c++)
      for (int h = 0; h < 2; h++) {
        pass[c][h].resize(rows * words);
        for (long long i = 0; i < rows * words; i++) pass[c][h][i] = {(unsigned)c, (unsigned)h, (unsigned)(i / words), 0};
      }
    for (int k = 0; k < 2; k++) {
      std::vector<int> own(rows), counts(rows, 5), ce(rows);
      for (int r = 0; r < rows; r++) own[r] = orank[r] == k;
      run(1, 64, [&] { glm_shard_counts(counts.data(), own.data(), ce.data(), rows); });
      auto out = pass[k][k];        // its own pass: its heads, right only for its rows
      auto recv = pass[1 - k][k];   // the peer's pass with its queries, sent back: right only for the peer's rows
      run(rows, 256, [&] { glm_shard_select(recv.data(), out.data(), own.data(), words); });
      for (int r = 0; r < rows; r++) {
        const uint4 w = out[r * words];
        if (w.x != (unsigned)orank[r] || w.y != (unsigned)k || w.z != (unsigned)r || ce[r] != (orank[r] == k ? 5 : 0)) {
          fails++;
          printf("compose trial %d rank %d row %d: computed on %u for heads of %u\n", trial, k, r, w.x, w.y);
          break;
        }
      }
    }
  }
  fails += fetch(g);
  printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
  return fails != 0;
}
