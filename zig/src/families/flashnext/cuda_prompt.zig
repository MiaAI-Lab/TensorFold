//! The prompt path's matmuls without split-K partials (work/research/R2-prefill.md W-A): `_b16mm_ks` and
//! `_fp4mm_ks` (src/tensorfold/families/qwen4_exp/cuda/prompt_mm.py, in the kernel set through
//! tools/zig/flashnext_prompt_spec.py) run all K slices of a tile in one program and add the slice totals in slice
//! order, the fp32 adds `_reduce` makes, so they give `_b16mm` + `_reduce`'s (and `_fp4mm` + `_reduce`'s) bits with
//! no [SK, M, N] fp32 buffer written and read back. Decode windows (<= 128 rows) keep the split kernels.
//! `TF_FLASHNEXT_PROMPT_MM=0` keeps Python's two launches for prompts too (the A/B and the fallback when a kernel
//! set has no `_ks` entries). `check` replays both on random, cancelling and edge-case inputs and compares bytes.
const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;
const tri = @import("cuda_triton.zig");

pub const env = "TF_FLASHNEXT_PROMPT_MM";

const bf16 = "*bf16";
const f32p = "*fp32";

/// Rows from which the in-program slices run: the wrappers' 128-row bucket (bf16.matmul's bm = 128 if m > 128).
pub const min_rows = 129;
/// `_b16mm_ks`'s row tiles a band (tools/zig/flashnext_prompt_spec.py GROUP).
const group = 8;

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

fn int(name: []const u8, v: usize) aot.Arg {
    return aot.int(name, @intCast(v));
}

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

fn cb(name: []const u8, v: bool) aot.Const {
    return aot.ci(name, @intFromBool(v));
}

fn run(t: tri.Tri, name: []const u8, grid: [3]usize, args: []const aot.Arg, consts: []const aot.Const) !void {
    const g: [3]u32 = .{ @intCast(grid[0]), @intCast(grid[1]), @intCast(grid[2]) };
    try t.set.?.run(t.s, name, g, args, consts);
}

/// Whether the kernel set holds the in-program kernels (an older set: Python's split path everywhere).
pub fn available(set: *const aot.Set) bool {
    return set.smallestConst("_b16mm_ks", "SK", 0) != null and set.smallestConst("_fp4mm_ks", "SK", 0) != null;
}

/// Whether a bf16 matmul of `m` rows takes `_b16mm_ks`.
pub fn b16Takes(on: bool, m: usize, n: usize, k: usize) bool {
    return on and m >= min_rows and k % 64 == 0 and tri.b16SplitK(n, k) > 1;
}

/// Whether an NVFP4-table matmul of `m` rows takes `_fp4mm_ks`.
pub fn fp4Takes(on: bool, m: usize, n: usize, k: usize) bool {
    return on and m >= min_rows and n % 64 == 0 and tri.fp4SplitK(n, k) > 1;
}

/// bf16.matmul's bits for m >= min_rows and split K: x [M, K] (rows `x_stride` apart) @ w [N, K].T -> out [M, N].
pub fn b16(t: tri.Tri, x: u64, x_stride: usize, w: u64, out: u64, fp32: bool, m: usize, n: usize, k: usize) !void {
    const sk = tri.b16SplitK(n, k);
    if (m < min_rows or sk < 2 or k % 64 != 0) return error.NotAPromptSplit;
    try run(t, "_b16mm_ks", .{ cdiv(m, 128) * cdiv(n, 64), 1, 1 }, &.{ aot.ptr("X", bf16, x), aot.ptr("W", bf16, w), aot.ptr("OUT", if (fp32) f32p else bf16, out), int("M", m), int("x_stride", x_stride) }, &.{ ci("N", n), ci("K", k), ci("SK", sk), ci("BM", 128), ci("BLOCK_N", 64), ci("BK", 64), cb("F32", fp32), ci("GROUP", group) });
}

/// nvfp4.matmul's bits for m >= min_rows and split K (the shared expert's tables).
pub fn fp4(t: tri.Tri, x: u64, x_stride: usize, fp: tri.Fp4, out: u64, fp32: bool, m: usize, n: usize, k: usize) !void {
    const sk = tri.fp4SplitK(n, k);
    if (m < min_rows or sk < 2 or n % 64 != 0) return error.NotAPromptSplit;
    const gpi = gpiFor((k / 16) / sk, 1); // nvfp4.CONFIG[128]: one block a step
    try run(t, "_fp4mm_ks", .{ cdiv(m, 128), n / 64, 1 }, &.{ aot.ptr("X", bf16, x), aot.ptr("W", if (fp.codes) "*u8" else "*u16", fp.weight), aot.ptr("S", if (fp.codes) "*u8" else f32p, fp.scale), aot.ptr("S2", f32p, fp.scale2), aot.ptr("OUT", if (fp32) f32p else bf16, out), int("M", m), int("x_stride", x_stride) }, &.{ ci("N", n), ci("K", k), ci("SK", sk), ci("BM", 128), ci("SBN", 64), ci("BLOCK_N", 64), ci("GPI", gpi), cb("F32", fp32), cb("PACKED", fp.codes) });
}

fn gpiFor(per: usize, want: usize) usize {
    for ([_]usize{ want, 8, 4, 2, 1 }) |g| if (g <= want and per % g == 0) return g;
    return 1;
}

/// `_b16mm` alone into the partials (the profile's split of a matmul from its `_reduce`; Tri.b16mm's launch).
pub fn b16Slices(t: tri.Tri, x: u64, x_stride: usize, w: u64, part: u64, fp32: bool, m: usize, n: usize, k: usize) !void {
    const sk = tri.b16SplitK(n, k);
    const bm: usize = if (m > 128) 128 else 16;
    try run(t, "_b16mm", .{ cdiv(m, bm), cdiv(n, 64), sk }, &.{ aot.ptr("X", bf16, x), aot.ptr("W", bf16, w), aot.ptr("OUT", if (fp32) f32p else bf16, part), aot.ptr("PART", f32p, part), int("M", m), int("x_stride", x_stride) }, &.{ ci("N", n), ci("K", k), ci("SK", sk), ci("BM", bm), ci("BLOCK_N", 64), ci("BK", 64), cb("F32", fp32) });
}

/// `_reduce` of `sk` slices (Tri.reduce).
pub fn reduce(t: tri.Tri, part: u64, out: u64, fp32: bool, total: usize, sk: usize) !void {
    try run(t, "_reduce", .{ cdiv(total, 1024), 1, 1 }, &.{ aot.ptr("PART", f32p, part), aot.ptr("OUT", if (fp32) f32p else bf16, out), int("total", total) }, &.{ ci("SK", sk), ci("BLOCK", 1024), cb("F32", fp32) });
}

// -- the check: both paths on the same inputs, bytes compared --------------------------------------------------

const Case = struct { n: usize, k: usize, fp32: bool, fp4: bool, tp: u32 };

/// Every prompt matmul with split K at TP=1 and TP=2 (kernels.json's 128-row `_b16mm` / `_fp4mm` entries with SK > 1).
pub const cases = [_]Case{
    .{ .n = 324, .k = 10240, .fp32 = true, .fp4 = false, .tp = 3 }, // HC down + inject
    .{ .n = 320, .k = 10240, .fp32 = true, .fp4 = false, .tp = 3 }, // mixer down
    .{ .n = 2560, .k = 2560, .fp32 = false, .fp4 = false, .tp = 3 }, // PLE key/value, MTP fc_e / fc_h
    .{ .n = 2560, .k = 6144, .fp32 = false, .fp4 = false, .tp = 1 }, // TP=1 out_proj / o_proj
    .{ .n = 2560, .k = 3072, .fp32 = true, .fp4 = false, .tp = 2 }, // TP=2 out_proj / o_proj partials
    .{ .n = 8240, .k = 2560, .fp32 = false, .fp4 = false, .tp = 2 }, // TP=2 GDN in_proj
    .{ .n = 7296, .k = 2560, .fp32 = false, .fp4 = false, .tp = 2 }, // TP=2 attention proj
    .{ .n = 2560, .k = 640, .fp32 = false, .fp4 = true, .tp = 1 }, // TP=1 shared down
    .{ .n = 2560, .k = 640, .fp32 = true, .fp4 = true, .tp = 1 },
    .{ .n = 2560, .k = 320, .fp32 = false, .fp4 = true, .tp = 2 }, // TP=2 shared down
    .{ .n = 2560, .k = 320, .fp32 = true, .fp4 = true, .tp = 2 },
};

const Fill = enum { normal, wide, cancel, edge };

fn bf16Bits(x: f32) u16 {
    const b: u32 = @bitCast(x);
    if (std.math.isNan(x)) return 0x7FC0;
    const r = b + 0x7FFF + ((b >> 16) & 1);
    return @intCast(r >> 16);
}

/// A bf16 value for `fill` (`edge`: zeros of both signs, subnormals, the largest finite values, a few inf).
fn sample(r: std.Random, fill: Fill, i: usize) u16 {
    return switch (fill) {
        .normal => bf16Bits(r.floatNorm(f32)),
        .wide => blk: {
            // sign, exponent across 2^-30 .. 2^30, random mantissa: sums that round at every step
            const e: i32 = r.intRangeAtMost(i32, -30, 30);
            const m: f32 = 1.0 + r.float(f32);
            const v = std.math.ldexp(m, e);
            break :blk bf16Bits(if (r.boolean()) -v else v);
        },
        .cancel => blk: {
            // pairs +a, -a (with a few bits off) so products cancel and the low bits decide
            const a = std.math.ldexp(1.0 + r.float(f32), r.intRangeAtMost(i32, -4, 8));
            const off: f32 = if (r.uintLessThan(u32, 8) == 0) std.math.ldexp(@as(f32, 1.0), r.intRangeAtMost(i32, -12, -2)) else 0;
            break :blk bf16Bits(if (i % 2 == 0) a else -a + off);
        },
        .edge => switch (r.uintLessThan(u32, 16)) {
            0 => 0x0000,
            1 => 0x8000,
            2 => @intCast(r.intRangeAtMost(u16, 1, 0x7F)), // subnormals
            3 => @as(u16, @intCast(r.intRangeAtMost(u16, 1, 0x7F))) | 0x8000,
            4 => 0x7F7F, // the largest finite
            5 => 0xFF7F,
            else => bf16Bits(r.floatNorm(f32) * 1e-3),
        },
    };
}

fn fillBuf(gpa: std.mem.Allocator, buf: cuda.DeviceBuffer, count: usize, r: std.Random, fill: Fill) !void {
    const h = try gpa.alloc(u16, count);
    defer gpa.free(h);
    for (h, 0..) |*v, i| v.* = sample(r, fill, i);
    try buf.upload(0, std.mem.sliceAsBytes(h));
}

/// Both paths of every case at TP `tp` (1, 2 or 3 for both) over `rows`, each input kind; true when every output
/// byte is equal. Prints one line a case and row count.
pub fn check(gpa: std.mem.Allocator, d: *const cuda.Driver, t: tri.Tri, tp: u32, rows: []const usize) !bool {
    var prng = std.Random.DefaultPrng.init(0x5eed_9f1);
    const r = prng.random();
    var all = true;
    var max_rows: usize = 0;
    for (rows) |m| max_rows = @max(max_rows, m);
    for (cases) |c| {
        if (c.tp & tp == 0) continue;
        const stride_extra: usize = 64;
        const xs_stride = c.k + stride_extra;
        const sk = if (c.fp4) tri.fp4SplitK(c.n, c.k) else tri.b16SplitK(c.n, c.k);
        var x = try cuda.DeviceBuffer.alloc(d, max_rows * xs_stride * 2);
        defer x.free();
        var w = try cuda.DeviceBuffer.alloc(d, c.n * c.k * 2);
        defer w.free();
        const es: usize = if (c.fp32) 4 else 2;
        var o1 = try cuda.DeviceBuffer.alloc(d, max_rows * c.n * es);
        defer o1.free();
        var o2 = try cuda.DeviceBuffer.alloc(d, max_rows * c.n * es);
        defer o2.free();
        var part = try cuda.DeviceBuffer.alloc(d, sk * max_rows * c.n * 4);
        defer part.free();
        var sc = try cuda.DeviceBuffer.alloc(d, (c.k / 16) * c.n * 4 + 256);
        defer sc.free();
        var s2 = try cuda.DeviceBuffer.alloc(d, c.n * 4 + 256);
        defer s2.free();
        try s2.fill32(@bitCast(@as(f32, 1.0)), t.s.handle);
        const h1 = try gpa.alloc(u8, max_rows * c.n * es);
        defer gpa.free(h1);
        const h2 = try gpa.alloc(u8, max_rows * c.n * es);
        defer gpa.free(h2);
        for ([_]Fill{ .normal, .wide, .cancel, .edge }) |fill| {
            try fillBuf(gpa, x, max_rows * xs_stride, r, fill);
            try fillBuf(gpa, w, c.n * c.k, r, if (fill == .edge) .normal else fill);
            if (c.fp4) {
                // the shared expert's scales are 1.0; other values check the scale step too
                const hs = try gpa.alloc(f32, (c.k / 16) * c.n);
                defer gpa.free(hs);
                for (hs) |*v| v.* = if (fill == .normal) 1.0 else std.math.ldexp(1.0 + r.float(f32), r.intRangeAtMost(i32, -3, 3));
                try sc.upload(0, std.mem.sliceAsBytes(hs));
            }
            for (rows) |m| {
                const strides = [_]usize{ c.k, xs_stride };
                for (strides) |stride| {
                    if (m * stride > max_rows * xs_stride) continue;
                    try o1.fill8(0xA5, t.s.handle);
                    try o2.fill8(0x5A, t.s.handle);
                    if (c.fp4) {
                        const fp: tri.Fp4 = .{ .weight = w.ptr, .scale = sc.ptr, .scale2 = s2.ptr };
                        try t.fp4mm(x.ptr, stride, fp, o1.ptr, c.fp32, part.ptr, m, c.n, c.k);
                        try fp4(t, x.ptr, stride, fp, o2.ptr, c.fp32, m, c.n, c.k);
                    } else {
                        try t.b16mm(x.ptr, stride, w.ptr, o1.ptr, c.fp32, part.ptr, m, c.n, c.k);
                        try b16(t, x.ptr, stride, w.ptr, o2.ptr, c.fp32, m, c.n, c.k);
                    }
                    try t.s.synchronize();
                    const bytes = m * c.n * es;
                    try o1.download(0, h1[0..bytes]);
                    try o2.download(0, h2[0..bytes]);
                    var differ: usize = 0;
                    var first: ?usize = null;
                    var i: usize = 0;
                    while (i < bytes) : (i += es) {
                        if (!std.mem.eql(u8, h1[i .. i + es], h2[i .. i + es])) {
                            differ += 1;
                            if (first == null) first = i / es;
                        }
                    }
                    const ok = differ == 0;
                    if (!ok) all = false;
                    std.debug.print("{s} {s} N {d} K {d} SK {d} {s} M {d} stride {d} {s}: {s}", .{ if (ok) "EQUAL" else "DIFFER", if (c.fp4) "_fp4mm_ks" else "_b16mm_ks", c.n, c.k, sk, if (c.fp32) "fp32" else "bf16", m, stride, @tagName(fill), if (ok) "" else "" });
                    if (first) |f0| std.debug.print("{d} of {d} values differ, first at row {d} col {d}", .{ differ, bytes / es, f0 / c.n, f0 % c.n });
                    std.debug.print("\n", .{});
                }
            }
        }
    }
    return all;
}

/// Time `reps` launches of each path for one case (ms a call), the speed side of the check.
pub fn bench(gpa: std.mem.Allocator, d: *const cuda.Driver, t: tri.Tri, c: Case, m: usize, reps: usize) !void {
    _ = gpa;
    const sk = if (c.fp4) tri.fp4SplitK(c.n, c.k) else tri.b16SplitK(c.n, c.k);
    var x = try cuda.DeviceBuffer.alloc(d, m * c.k * 2);
    defer x.free();
    try x.fill8(0x3c, t.s.handle);
    var w = try cuda.DeviceBuffer.alloc(d, c.n * c.k * 2);
    defer w.free();
    try w.fill8(0x3b, t.s.handle);
    var o = try cuda.DeviceBuffer.alloc(d, m * c.n * 4);
    defer o.free();
    var part = try cuda.DeviceBuffer.alloc(d, sk * m * c.n * 4);
    defer part.free();
    var sc = try cuda.DeviceBuffer.alloc(d, (c.k / 16) * c.n * 4 + 256);
    defer sc.free();
    try sc.fill32(@bitCast(@as(f32, 1.0)), t.s.handle);
    const fp: tri.Fp4 = .{ .weight = w.ptr, .scale = sc.ptr, .scale2 = sc.ptr };
    var e0 = try cuda.Event.init(d, true);
    defer e0.deinit();
    var e1 = try cuda.Event.init(d, true);
    defer e1.deinit();
    var e2 = try cuda.Event.init(d, true);
    defer e2.deinit();
    for (0..2) |_| {
        try e0.record(t.s);
        for (0..reps) |_| if (c.fp4) try t.fp4mm(x.ptr, c.k, fp, o.ptr, c.fp32, part.ptr, m, c.n, c.k) else try t.b16mm(x.ptr, c.k, w.ptr, o.ptr, c.fp32, part.ptr, m, c.n, c.k);
        try e1.record(t.s);
        for (0..reps) |_| if (c.fp4) try fp4(t, x.ptr, c.k, fp, o.ptr, c.fp32, m, c.n, c.k) else try b16(t, x.ptr, c.k, w.ptr, o.ptr, c.fp32, m, c.n, c.k);
        try e2.record(t.s);
        try e2.synchronize();
    }
    const a = try cuda.Event.elapsedMs(e0, e1) / @as(f32, @floatFromInt(reps));
    const b = try cuda.Event.elapsedMs(e1, e2) / @as(f32, @floatFromInt(reps));
    const flop = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(c.n)) * @as(f64, @floatFromInt(c.k));
    std.debug.print("bench {s} N {d} K {d} SK {d} M {d}: split + _reduce {d:.3} ms ({d:.1} TFLOPS), in-program {d:.3} ms ({d:.1} TFLOPS), {d:.2}x\n", .{ if (c.fp4) "fp4" else "b16", c.n, c.k, sk, m, a, flop / a / 1e9, b, flop / b / 1e9, a / b });
}

// -- the indexer's scoring and selection of a prompt block, by rows ---------------------------------------------

/// attention.qsa_rows' `_scores` alone (Tri.qsaRows' first launch): rows [0, rows) of the block whose first row's
/// position is at `pos0`, scores into `sc.scores`.
pub fn qsaScores(t: tri.Tri, iq: u64, pooled: u64, pos0: u64, sc: tri.AttnScratch, rows: usize, g: tri.AttnGeometry, context: ?usize, d: tri.Dims) !void {
    const blocks = g.blocks(context);
    try run(t, "_scores", .{ rows, cdiv(blocks, 64), 1 }, &.{ aot.ptr("IQ", bf16, iq), aot.ptr("POOLED", bf16, pooled), aot.ptr("POS0", "*i32", pos0), aot.ptr("SC", f32p, sc.scores), int("NB", g.nb) }, &.{ ci("HI", d.index_heads), ci("DI", d.index_dim), ci("RATIO", g.ratio), ci("TOP", g.budget / g.ratio), ci("BB", 64) });
}

/// Tri.qsaRows' second launch: `_select` (or `_select_tiles` past the registers) of the scored rows.
pub fn qsaSelect(t: tri.Tri, pos0: u64, sc: tri.AttnScratch, rows: usize, g: tri.AttnGeometry, context: ?usize) !void {
    const top = g.budget / g.ratio;
    const blocks = g.blocks(context);
    const lists = [_]aot.Arg{ aot.ptr("SC", f32p, sc.scores), aot.ptr("POS0", "*i32", pos0), aot.ptr("IDS", "*i32", sc.ids), aot.ptr("NKR", "*i32", sc.nk), aot.ptr("SPR", "*i32", sc.sparse), int("NB", g.nb) };
    var width: usize = 1;
    while (width < blocks) width *= 2;
    if (width <= tri.AttnGeometry.select_regs) {
        try run(t, "_select", .{ rows, 1, 1 }, &lists, &.{ ci("RATIO", g.ratio), ci("TOP", top), ci("IDW", g.idw), ci("BLOCK", width) });
    } else {
        try run(t, "_select_tiles", .{ rows, 1, 1 }, &lists, &.{ ci("RATIO", g.ratio), ci("TOP", top), ci("IDW", g.idw), ci("TB", if (rows >= 64) 4096 else 8192) });
    }
}

/// The scratch of rows [r, ..) of a block: scores, key lists, lengths and flags at their rows.
pub fn scratchAt(sc: tri.AttnScratch, g: tri.AttnGeometry, r: usize) tri.AttnScratch {
    var s = sc;
    s.scores = sc.scores + r * g.nb * 4;
    s.ids = sc.ids + r * g.idw * 4;
    s.nk = sc.nk + r * 4;
    s.sparse = sc.sparse + r * 4;
    return s;
}

/// `_scores_rows`: `_scores` for `rt` rows a program (each block tile's pooled keys loaded once for them).
pub fn qsaScoresRows(t: tri.Tri, iq: u64, pooled: u64, pos0: u64, sc: tri.AttnScratch, rows: usize, g: tri.AttnGeometry, context: ?usize, d: tri.Dims, rt: usize) !void {
    const blocks = g.blocks(context);
    try run(t, "_scores_rows", .{ cdiv(rows, rt), cdiv(blocks, 64), 1 }, &.{ aot.ptr("IQ", bf16, iq), aot.ptr("POOLED", bf16, pooled), aot.ptr("POS0", "*i32", pos0), aot.ptr("SC", f32p, sc.scores), int("NB", g.nb), int("ROWS", rows) }, &.{ ci("HI", d.index_heads), ci("DI", d.index_dim), ci("RATIO", g.ratio), ci("TOP", g.budget / g.ratio), ci("BB", 64), ci("RT", rt) });
}

/// `_select_tiles` for any row length (Python's own choice past `_select`'s registers; the same lists).
pub fn qsaSelectTiles(t: tri.Tri, pos0: u64, sc: tri.AttnScratch, rows: usize, g: tri.AttnGeometry) !void {
    const lists = [_]aot.Arg{ aot.ptr("SC", f32p, sc.scores), aot.ptr("POS0", "*i32", pos0), aot.ptr("IDS", "*i32", sc.ids), aot.ptr("NKR", "*i32", sc.nk), aot.ptr("SPR", "*i32", sc.sparse), int("NB", g.nb) };
    try run(t, "_select_tiles", .{ rows, 1, 1 }, &lists, &.{ ci("RATIO", g.ratio), ci("TOP", g.budget / g.ratio), ci("IDW", g.idw), ci("TB", if (rows >= 64) 4096 else 8192) });
}

/// `qsa-check`: `_scores_rows` against `_scores` (score bytes) and `_select_tiles` against `_select` (key lists,
/// lengths, flags) on random and tie-heavy indexer keys, rows past several positions. True when all equal.
pub fn qsaCheck(gpa: std.mem.Allocator, d: *const cuda.Driver, t: tri.Tri) !bool {
    const capacity: usize = 262144 + 8;
    const g = tri.AttnGeometry.init(capacity, 2048, 4);
    const dims: tri.Dims = .{};
    const DI = dims.index_dim;
    const HI = dims.index_heads;
    const rows: usize = 256;
    var prng = std.Random.DefaultPrng.init(0x95a_0001);
    const r = prng.random();
    const hp = try gpa.alloc(u16, g.nb * DI);
    defer gpa.free(hp);
    const hq = try gpa.alloc(u16, rows * HI * DI);
    defer gpa.free(hq);
    var pooled = try cuda.DeviceBuffer.alloc(d, hp.len * 2);
    defer pooled.free();
    var iq = try cuda.DeviceBuffer.alloc(d, hq.len * 2);
    defer iq.free();
    var pos = try cuda.DeviceBuffer.alloc(d, 256);
    defer pos.free();
    var bufs: [2][4]cuda.DeviceBuffer = undefined;
    const sz = [4]usize{ rows * g.nb * 4, rows * g.idw * 4, rows * 4, rows * 4 };
    for (&bufs) |*set| for (set, sz) |*b, n| {
        b.* = try cuda.DeviceBuffer.alloc(d, n + 256);
    };
    defer for (&bufs) |*set| for (set) |*b| b.free();
    const host1 = try gpa.alloc(u8, sz[0]);
    defer gpa.free(host1);
    const host2 = try gpa.alloc(u8, sz[0]);
    defer gpa.free(host2);
    var all = true;
    // 16 B-aligned start positions: dense rows, register select, near and past the tiled switch (131,072 keys)
    const positions = [_]usize{ 1000, 2040, 20000, 70000, 129000, 131000, 200000, 261000 };
    for ([_]bool{ false, true }) |ties| {
        for (hp, 0..) |*v, i| v.* = if (ties) (if (i % 7 == 0) 0x3f80 else 0) else bf16Bits(r.floatNorm(f32));
        for (hq) |*v| v.* = bf16Bits(r.floatNorm(f32));
        try pooled.upload(0, std.mem.sliceAsBytes(hp));
        try iq.upload(0, std.mem.sliceAsBytes(hq));
        for (positions) |p0| {
            const n = if (p0 + rows > capacity) capacity - p0 else rows;
            try pos.fill32(@intCast(p0), t.s.handle);
            const ends = p0 + n;
            var scs: [2]tri.AttnScratch = undefined;
            for (&scs, bufs) |*sc, set| {
                for (set) |b| try b.fill8(0xA7, t.s.handle);
                sc.* = .{ .po = 0, .pm = 0, .pl = 0, .ids = set[1].ptr, .nk = set[2].ptr, .sparse = set[3].ptr, .scores = set[0].ptr };
            }
            try qsaScores(t, iq.ptr, pooled.ptr, pos.ptr, scs[0], n, g, ends, dims);
            try qsaSelect(t, pos.ptr, scs[0], n, g, ends);
            for ([_]usize{ 8, 16 }) |rt| {
                try bufs[1][0].fill8(0xA7, t.s.handle);
                try qsaScoresRows(t, iq.ptr, pooled.ptr, pos.ptr, scs[1], n, g, ends, dims, rt);
                try t.s.synchronize();
                try bufs[0][0].download(0, host1[0 .. n * g.nb * 4]);
                try bufs[1][0].download(0, host2[0 .. n * g.nb * 4]);
                const ok = std.mem.eql(u8, host1[0 .. n * g.nb * 4], host2[0 .. n * g.nb * 4]);
                if (!ok) all = false;
                std.debug.print("{s} _scores_rows RT {d} position {d} rows {d}{s}: score bytes\n", .{ if (ok) "EQUAL" else "DIFFER", rt, p0, n, if (ties) " tie-heavy" else "" });
            }
            // the lists from the same scores by the tiled select
            try qsaSelectTiles(t, pos.ptr, scs[1], n, g);
            try t.s.synchronize();
            var ok = true;
            for (1..4) |k| {
                const bytes = if (k == 1) n * g.idw * 4 else n * 4;
                try bufs[0][k].download(0, host1[0..bytes]);
                try bufs[1][k].download(0, host2[0..bytes]);
                ok = ok and std.mem.eql(u8, host1[0..bytes], host2[0..bytes]);
            }
            if (!ok) all = false;
            std.debug.print("{s} _select_tiles = _select position {d} rows {d}{s}: key lists, lengths, flags\n", .{ if (ok) "EQUAL" else "DIFFER", p0, n, if (ties) " tie-heavy" else "" });
        }
    }
    return all;
}
