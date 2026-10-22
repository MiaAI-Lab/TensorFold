//! The n-gram embedding's row ids on the host (Python qwen4_exp/cuda/ngram.py): each head's table size is a prime past
//! the vocabulary base, the multipliers come from splitmix64, and a row id is numpy's signed int64 product-xor of the
//! last n tokens (EOS resets the n-grams), floor-mod the head's size, plus its offset. Same values as numpy.
const std = @import("std");

const golden: u64 = 0x9E3779B97F4A7C15;
const prime_step: u64 = 10007;

pub const max_n = 4;
pub const max_heads = 32;

fn splitmix64(v: u64) u64 {
    var x = v +% golden;
    x = (x ^ (x >> 30)) *% 0xBF58476D1CE4E5B9;
    x = (x ^ (x >> 27)) *% 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

fn isPrime(v: u64) bool {
    if (v < 2) return false;
    if (v % 2 == 0) return v == 2;
    var d: u64 = 3;
    while (d * d <= v) : (d += 2) {
        if (v % d == 0) return false;
    }
    return true;
}

/// The `count`-th prime after `start` (Python `_nth_prime_after`).
fn nthPrimeAfter(start: u64, count: u64) u64 {
    var p = start;
    for (0..count) |_| {
        p += 1;
        while (!isPrime(p)) p += 1;
    }
    return p;
}

pub const Options = struct {
    vocab: u64,
    ngram_size: u32,
    heads_per_ngram: u32,
    vocab_base: u64,
    divisor: u64,
    seed: u64,
    eos: i64,
    embed_dim: u32,
    ple_index: u32 = 0,
};

pub const NGram = struct {
    n: u32,
    per_ngram: u32,
    heads: u32,
    eos: i64,
    sizes: [max_heads]i64 = @splat(0),
    offsets: [max_heads]i64 = @splat(0),
    multipliers: [max_n]i64 = @splat(0),
    /// table rows: the sizes' total rounded up to the divisor
    rows: u64,
    /// values a head's row holds
    dims: u32,

    pub fn init(o: Options) !NGram {
        if (o.ngram_size < 2 or o.ngram_size > max_n) return error.UnsupportedNgramSize;
        const heads = (o.ngram_size - 1) * o.heads_per_ngram;
        if (heads == 0 or heads > max_heads or o.embed_dim % heads != 0) return error.UnsupportedNgramHeads;
        var g: NGram = .{ .n = o.ngram_size, .per_ngram = o.heads_per_ngram, .heads = heads, .eos = o.eos, .rows = 0, .dims = o.embed_dim / heads };
        var total: u64 = 0;
        for (0..heads) |h| {
            const size = nthPrimeAfter(o.vocab_base - 1, @as(u64, o.ple_index) * heads + h + 1);
            g.sizes[h] = @intCast(size);
            g.offsets[h] = @intCast(total);
            total += size;
        }
        g.rows = (total + o.divisor - 1) / o.divisor * o.divisor;
        // Python: half = max(1, ((2**63 - 1) // vocab) // 2); 2 * (splitmix64(base + golden * (i + 1)) % half) + 1
        const half: u64 = @max(1, (std.math.maxInt(i64) / @max(o.vocab, 1)) / 2);
        const base = o.seed +% prime_step *% o.ple_index;
        for (0..o.ngram_size) |i| g.multipliers[i] = @intCast(2 * (splitmix64(base +% golden *% (i + 1)) % half) + 1);
        return g;
    }

    /// The tokens a later call keeps as its history: the last n - 1 of `history` ++ `tokens`.
    pub fn context(g: NGram) u32 {
        return g.n - 1;
    }

    /// The checkpoint's shipped constants must equal the derived ones (Python `check`).
    pub fn check(g: NGram, multipliers: []const i64, offsets: []const i64, sizes: []const i64) !void {
        if (!std.mem.eql(i64, multipliers, g.multipliers[0..g.n])) return error.NgramMultipliersDiffer;
        if (!std.mem.eql(i64, offsets, g.offsets[0..g.heads])) return error.NgramOffsetsDiffer;
        if (!std.mem.eql(i64, sizes, g.sizes[0..g.heads])) return error.NgramSizesDiffer;
    }

    /// Row ids for `tokens` after `history` (n - 1 earlier tokens, EOS-filled at a sequence's start) into
    /// `out[tokens.len * heads]`, row-major [token][head] (Python `ids`).
    pub fn ids(g: NGram, history: []const i64, tokens: []const i64, out: []i64) !void {
        if (history.len != g.n - 1) return error.NgramHistoryLength;
        if (out.len != tokens.len * g.heads) return error.NgramOutLength;
        const width = history.len + tokens.len;
        const at = struct {
            fn f(h: []const i64, t: []const i64, p: usize) i64 {
                return if (p < h.len) h[p] else t[p - h.len];
            }
        }.f;
        // the latest EOS strictly before each position: a running maximum, as numpy's maximum.accumulate
        var last_eos: i64 = -1;
        var p: usize = 0;
        while (p < width) : (p += 1) {
            const before = last_eos;
            if (at(history, tokens, p) == g.eos) last_eos = @intCast(p);
            if (p < history.len) continue;
            const in_segment = @as(i64, @intCast(p)) - (before + 1);
            var shifted: [max_n]i64 = undefined;
            for (0..g.n) |shift| {
                const source = @as(i64, @intCast(p)) - @as(i64, @intCast(shift));
                shifted[shift] = if (in_segment >= @as(i64, @intCast(shift)) and source >= 0) at(history, tokens, @intCast(source)) else g.eos;
            }
            const row = out[(p - history.len) * g.heads ..][0..g.heads];
            var k: usize = 0;
            for (2..g.n + 1) |ngram| {
                var mixed: i64 = shifted[0] *% g.multipliers[0];
                for (1..ngram) |q| mixed ^= shifted[q] *% g.multipliers[q];
                const first = (ngram - 2) * g.per_ngram;
                for (0..g.per_ngram) |j| {
                    row[k] = @mod(mixed, g.sizes[first + j]) + g.offsets[first + j];
                    k += 1;
                }
            }
        }
    }

    /// The history after keeping `kept` of `tokens` (Python commit: concat(history, tokens[:keep])[-(n-1):]).
    pub fn advance(g: NGram, history: []i64, tokens: []const i64) void {
        const c = g.n - 1;
        if (tokens.len >= c) {
            @memcpy(history, tokens[tokens.len - c ..]);
            return;
        }
        std.mem.copyForwards(i64, history[0 .. c - tokens.len], history[tokens.len..c]);
        @memcpy(history[c - tokens.len ..], tokens);
    }

    pub fn initialHistory(g: NGram, history: []i64) void {
        @memset(history[0 .. g.n - 1], g.eos);
    }
};

// Values from Python's NGram with the nvidia/Qwen3.8-Flash-Next-NVFP4 config (vocab 248320, n 3, 8 heads per n-gram,
// base 20,000,000, divisor 128, seed 1234, eos 248044, 2560 dims, PLE index 0).
const test_options: Options = .{ .vocab = 248320, .ngram_size = 3, .heads_per_ngram = 8, .vocab_base = 20000000, .divisor = 128, .seed = 1234, .eos = 248044, .embed_dim = 2560 };

test "derived constants equal Python's" {
    const g = try NGram.init(test_options);
    try std.testing.expectEqualSlices(i64, &.{ 20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171 }, g.sizes[0..16]);
    try std.testing.expectEqualSlices(i64, &.{ 0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275 }, g.offsets[0..16]);
    try std.testing.expectEqualSlices(i64, &.{ 23703573157769, 20109073645365, 8052911324071 }, g.multipliers[0..3]);
    try std.testing.expectEqual(@as(u64, 320001536), g.rows);
    try std.testing.expectEqual(@as(u32, 160), g.dims);
}

test "row ids equal Python's, EOS resets included" {
    const g = try NGram.init(test_options);
    var h: [2]i64 = undefined;
    g.initialHistory(&h);
    const toks = [_]i64{ 9707, 11, 1879, 248044, 314, 2001, 248319, 0, 248044, 248044, 77 };
    var out: [toks.len * 16]i64 = undefined;
    try g.ids(&h, &toks, &out);
    const want = [_][16]i64{
        .{ 16410909, 39682429, 55103279, 60931720, 87006904, 116506179, 131512017, 152932897, 169641436, 182022480, 209277433, 237891023, 256841529, 277007269, 290665954, 300984276 },
        .{ 18158303, 36390029, 45652312, 70783524, 98191154, 114024957, 127804916, 159566246, 175132467, 192583600, 217919476, 237199054, 259336807, 261799367, 289359313, 307699687 },
        .{ 6380558, 26411572, 56460672, 78566983, 94693008, 106742196, 124822692, 148942556, 164226950, 190352573, 210933682, 238908951, 242182004, 265475238, 299910982, 312121804 },
        .{ 18827606, 27700912, 56521518, 65780303, 81132822, 100518937, 120474864, 142052741, 163793416, 184135950, 219854666, 227756662, 248975183, 271245822, 285569892, 318943450 },
        .{ 8260854, 33106041, 59264976, 60072089, 98935463, 112687005, 134061582, 143955837, 178749106, 199051338, 215607266, 231149282, 254184348, 278446456, 293473221, 310651623 },
        .{ 5856173, 25242257, 54975642, 76647512, 92408179, 104337021, 122238348, 146121840, 160422013, 182017932, 205651970, 226550590, 251029236, 275858420, 297546152, 306568227 },
        .{ 14050029, 21157844, 59096366, 65120935, 83416240, 110450110, 131877770, 148751508, 163250688, 186851054, 206373379, 225160375, 248013854, 271920027, 286789403, 302890280 },
        .{ 3745007, 25177969, 59639583, 68080282, 84924354, 118004746, 128374230, 156931756, 176658937, 191889199, 214212374, 225019246, 242093785, 260069218, 286261388, 318722933 },
        .{ 16421076, 34964127, 58645314, 64737717, 85974391, 107327379, 120238726, 145766809, 167206240, 197096504, 210176586, 239333033, 244402808, 270436996, 292662780, 305397995 },
        .{ 9663979, 26558231, 56120240, 74755659, 80459717, 109265651, 132697467, 151022725, 170054832, 192967038, 200687722, 225763581, 259275737, 272983544, 297596484, 300986548 },
        .{ 16611476, 29512305, 49704971, 66165983, 89881645, 105251896, 139055756, 152191225, 178497739, 192880870, 205859405, 228292887, 258357841, 269648670, 287018169, 315908561 },
    };
    for (want, 0..) |row, i| try std.testing.expectEqualSlices(i64, &row, out[i * 16 ..][0..16]);
    // a later call from a kept history gives what the long call gave for the same token
    var h2 = [_]i64{ 123, 456 };
    var one: [16]i64 = undefined;
    try g.ids(&h2, &.{789}, &one);
    try std.testing.expectEqualSlices(i64, &.{ 2491316, 28761835, 51917996, 68359989, 88189145, 101470006, 131395511, 157970613, 166816437, 197523925, 206707499, 235951520, 251357128, 266767674, 285239060, 317603747 }, &one);
}

test "advance keeps the last n - 1 tokens" {
    const g = try NGram.init(test_options);
    var h = [_]i64{ 1, 2 };
    g.advance(&h, &.{3});
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, &h);
    g.advance(&h, &.{ 4, 5, 6 });
    try std.testing.expectEqualSlices(i64, &.{ 5, 6 }, &h);
    g.advance(&h, &.{});
    try std.testing.expectEqualSlices(i64, &.{ 5, 6 }, &h);
}
