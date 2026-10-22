//! `int4-check` (no checkpoint): fn_int4.cu on random GPTQ int4 weights at the INT4-AutoRound shapes, against a
//! host reference read straight from the GPTQ bytes (not from the packed layout):
//!  - the kernel's bits: every output against the documented order emulated on the host (each group's products
//!    summed exactly in fp64, rounded to fp32, then acc = fmaf(sum, scale, acc) in group order): the MMA chain's own
//!    rounding is the only difference allowed (reported in fp32 ulps), and the error against the full fp64 dot
//!    product relative to the sum of |products| is bounded;
//!  - row invariance: each row of an m-row call is byte-equal to the 1-row call of that row, at every m;
//!  - the n8 tiles a warp takes (1, 2, 4) never change a byte;
//!  - the routed experts on the plan (gate/up SwiGLU, down fp32 and bf16, groups of 128 and of 64) byte-equal to the
//!    dense kernel on each pair's expert, the skipped (shared) slot untouched;
//!  - timings: the head GEMV, decode experts, a prompt chunk's experts.
const std = @import("std");
const cuda = @import("cuda");
const kern = @import("cuda_kernels.zig");
const int4 = @import("cuda_int4.zig");

const Allocator = std.mem.Allocator;

fn bf16ToF32(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

fn f32ToBf16(f: f32) u16 {
    const u: u32 = @bitCast(f);
    if (std.math.isNan(f)) return 0x7FC0;
    const r = u + 0x7FFF + ((u >> 16) & 1);
    return @intCast(r >> 16);
}

/// Random GPTQ bytes for [n, k]: qweight int32 [k/8, n], scales fp16 [k/128, n] in [2^-9, 2^-6), qzeros all 7.
const Gen = struct {
    qweight: []u8,
    scales: []u8,
    n: usize,
    k: usize,

    fn init(gpa: Allocator, r: std.Random, n: usize, k: usize) !Gen {
        const q = try gpa.alloc(u8, k / 8 * n * 4);
        r.bytes(q);
        const s = try gpa.alloc(u8, k / 128 * n * 2);
        for (0..s.len / 2) |i| {
            const v: f16 = @floatCast(std.math.ldexp(1.0 + r.float(f32), -9 + @as(i32, r.intRangeLessThan(i32, 0, 3))));
            std.mem.writeInt(u16, s[2 * i ..][0..2], @bitCast(v), .little);
        }
        return .{ .qweight = q, .scales = s, .n = n, .k = k };
    }

    fn deinit(g: *Gen, gpa: Allocator) void {
        gpa.free(g.qweight);
        gpa.free(g.scales);
    }

    fn src(g: Gen) int4.Source {
        return .{ .qweight = g.qweight, .scales = g.scales, .n_full = g.n, .k_full = g.k };
    }
};

/// The documented order on the host: column n of `src` (inputs [k0, k0 + k), groups of `gs`) against a bf16 row.
fn emulate(src: int4.Source, n: usize, k0: usize, k: usize, gs: usize, x: []const u16) struct { y: f32, exact: f64, mag: f64 } {
    var acc: f32 = 0;
    var exact: f64 = 0;
    var mag: f64 = 0;
    var g: usize = 0;
    while (g < k) : (g += gs) {
        var p: f64 = 0;
        for (g..g + gs) |kk| {
            const q: f64 = @floatFromInt(@as(i32, int4.code(src, n, k0 + kk)) - 8);
            p += @as(f64, bf16ToF32(x[kk])) * q;
        }
        const s = int4.scale(src, n, k0 + g);
        acc = @mulAdd(f32, @floatCast(p), s, acc);
        exact += p * s;
        for (g..g + gs) |kk| mag += @abs(@as(f64, bf16ToF32(x[kk])) * int4.weight(src, n, k0 + kk));
    }
    return .{ .y = acc, .exact = exact, .mag = mag };
}

fn ulps(a: f32, b: f32) u32 {
    const ia: i64 = @as(i32, @bitCast(a));
    const ib: i64 = @as(i32, @bitCast(b));
    const fa = if (ia < 0) -(ia & 0x7FFFFFFF) else ia;
    const fb = if (ib < 0) -(ib & 0x7FFFFFFF) else ib;
    return @intCast(@min(@abs(fa - fb), std.math.maxInt(u32)));
}

const Dev = struct {
    d: *const cuda.Driver,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    gpa: Allocator,

    fn put(dv: *Dev, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(dv.d, bytes);
        try dv.bufs.append(dv.gpa, b);
        return b.ptr;
    }

    fn zeros(dv: *Dev, n: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(dv.d, @max(n, 16));
        try b.fill8(0, null);
        try dv.bufs.append(dv.gpa, b);
        return b.ptr;
    }

    fn get(dv: *Dev, ptr: u64, out: []u8) !void {
        const b: cuda.DeviceBuffer = .{ .d = dv.d, .ptr = ptr, .len = out.len };
        try b.download(0, out);
    }

    fn deinit(dv: *Dev) void {
        for (dv.bufs.items) |*b| b.free();
        dv.bufs.deinit(dv.gpa);
    }
};

/// Packs `g`'s slice to the device: (words, scales) pointers.
fn packDev(gpa: Allocator, dv: *Dev, g: Gen, sl: int4.Slice) ![2]u64 {
    const w = try gpa.alloc(u32, int4.wordsOf(sl.n, sl.k));
    defer gpa.free(w);
    try int4.packWords(g.src(), sl, w);
    const s = try gpa.alloc(u16, int4.scalesOf(sl.n, sl.k, sl.gs));
    defer gpa.free(s);
    try int4.packScales(g.src(), sl, s);
    return .{ try dv.put(std.mem.sliceAsBytes(w)), try dv.put(std.mem.sliceAsBytes(s)) };
}

fn randRows(gpa: Allocator, r: std.Random, rows: usize, k: usize) ![]u16 {
    const x = try gpa.alloc(u16, rows * k);
    for (x) |*v| v.* = f32ToBf16(r.floatNorm(f32) * 0.5);
    return x;
}

pub const Report = struct { ok: bool = true, max_ulps: u32 = 0, max_rel: f64 = 0 };

/// The dense kernel against the host order, row invariance and NT invariance: one [n, k] matrix (inputs from k0).
fn checkDense(gpa: Allocator, k: *const int4.Kernels, s: cuda.Stream, dv: *Dev, r: std.Random, n: usize, k_full: usize, k0: usize, kk: usize, gs: usize, rows_list: []const usize, rep: *Report) !void {
    var g = try Gen.init(gpa, r, n, k_full);
    defer g.deinit(gpa);
    const sl: int4.Slice = .{ .n0 = 0, .n = n, .k0 = k0, .k = kk, .gs = gs };
    const p = try packDev(gpa, dv, g, sl);
    const m: int4.Mat = .{ .w = p[0], .s = p[1], .n = @intCast(n), .k = @intCast(kk), .gs = @intCast(gs) };
    var most: usize = 0;
    for (rows_list) |x| most = @max(most, x);
    const x = try randRows(gpa, r, most, kk);
    defer gpa.free(x);
    const xd = try dv.put(std.mem.sliceAsBytes(x));
    const out = try dv.zeros(most * n * 4);
    const host = try gpa.alloc(f32, most * n);
    defer gpa.free(host);
    const one = try gpa.alloc(f32, n);
    defer gpa.free(one);
    // the 1-row results of every row (the reference for row invariance)
    const solo = try gpa.alloc(f32, most * n);
    defer gpa.free(solo);
    for (0..most) |row| {
        try int4.dense(k, s, xd + row * kk * 2, kk, m, out, true, 1);
        try s.synchronize();
        try dv.get(out, std.mem.sliceAsBytes(solo[row * n ..][0..n]));
    }
    // against the host order (a sample of columns at the widest shapes)
    const step: usize = if (n > 8192) 97 else 1;
    var bad: usize = 0;
    for (0..@min(most, 4)) |row| {
        var col: usize = 0;
        while (col < n) : (col += step) {
            const e = emulate(g.src(), col, k0, kk, gs, x[row * kk ..][0..kk]);
            const got = solo[row * n + col];
            rep.max_ulps = @max(rep.max_ulps, ulps(got, e.y));
            const rel = @abs(@as(f64, got) - e.exact) / @max(e.mag, 1e-30);
            rep.max_rel = @max(rep.max_rel, rel);
            if (rel > 1e-4) bad += 1;
        }
    }
    if (bad > 0) {
        std.debug.print("  dense n {d} k {d} gs {d}: {d} outputs past 1e-4 of the fp64 sum\n", .{ n, kk, gs, bad });
        rep.ok = false;
    }
    for (rows_list) |rows| for ([_]usize{ 1, 2, 4 }) |nt| {
        if (n % (8 * nt) != 0) continue;
        try int4.denseNt(k, s, xd, kk, m, out, true, rows, nt);
        try s.synchronize();
        try dv.get(out, std.mem.sliceAsBytes(host[0 .. rows * n]));
        if (!std.mem.eql(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(host[0 .. rows * n])), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(solo[0 .. rows * n])))) {
            var diff: usize = 0;
            for (host[0 .. rows * n], solo[0 .. rows * n]) |a, b| diff += @intFromBool(@as(u32, @bitCast(a)) != @as(u32, @bitCast(b)));
            std.debug.print("  dense n {d} k {d} gs {d}: {d} rows, NT {d}: {d} outputs differ from the 1-row calls\n", .{ n, kk, gs, rows, nt, diff });
            rep.ok = false;
        }
    };
    // bf16 out = the fp32 out rounded
    const rows = @min(most, 17);
    try int4.dense(k, s, xd, kk, m, out, false, rows);
    try s.synchronize();
    const hb = try gpa.alloc(u16, rows * n);
    defer gpa.free(hb);
    try dv.get(out, std.mem.sliceAsBytes(hb));
    for (hb, solo[0 .. rows * n]) |a, b| if (a != f32ToBf16(b)) {
        rep.ok = false;
        std.debug.print("  dense bf16 out is not the fp32 out rounded\n", .{});
        break;
    };
    std.debug.print("  dense n {d} k {d} (from {d}) gs {d}: rows {any} NT 1/2/4 byte-equal to 1-row calls; vs host order max {d} ulps, max |err|/sum|x w| {e:.2}\n", .{ n, kk, k0, gs, rows_list, rep.max_ulps, rep.max_rel });
}

/// The group order, enforced: weights and inputs whose every group sum is exact in fp32 (one nonzero code a column
/// and group, inputs that are small powers of two), so the MMA chain's own rounding cannot hide anything, and scales
/// that make the groups cancel (2^25 s, -2^25 s, s, ...): the kernel must equal acc = fmaf(sum_g, scale_g, acc) in
/// group order bit for bit (any other order or a sum of products first gives other bits).
fn checkOrder(gpa: Allocator, k: *const int4.Kernels, s: cuda.Stream, dv: *Dev, r: std.Random, gs: usize, rep: *Report) !void {
    const n: usize = 64;
    const kk: usize = 1024;
    const kg = kk / 128;
    const qw = try gpa.alloc(u8, kk / 8 * n * 4);
    defer gpa.free(qw);
    const sc = try gpa.alloc(u8, kg * n * 2);
    defer gpa.free(sc);
    // codes 8 (weight 0) everywhere, then one code a column and group of 128
    for (0..kk / 8 * n) |i| std.mem.writeInt(u32, qw[4 * i ..][0..4], 0x88888888, .little);
    const x = try gpa.alloc(u16, kk);
    defer gpa.free(x);
    for (x, 0..) |*v, i| v.* = f32ToBf16(std.math.ldexp(@as(f32, if (i % 3 == 0) -1.0 else 1.0), @as(i32, @intCast(i % 7)) - 3));
    for (0..n) |col| {
        for (0..kg) |g| {
            const kin = g * 128 + r.uintLessThan(usize, 128);
            const code: u32 = if (r.boolean()) 15 else 1; // weights 7 or -7
            const word_i = (kin / 8) * n + col;
            var w = std.mem.readInt(u32, qw[4 * word_i ..][0..4], .little);
            const sh: u5 = @intCast(4 * (kin % 8));
            w = (w & ~(@as(u32, 0xF) << sh)) | (code << sh);
            std.mem.writeInt(u32, qw[4 * word_i ..][0..4], w, .little);
            // cancelling magnitudes: groups 0 and 1 huge and opposite, the rest small
            const e: i32 = switch (g) {
                0, 1 => 12,
                else => -12 + @as(i32, @intCast(g)),
            };
            const v: f16 = @floatCast(std.math.ldexp(@as(f32, 1.0) + @as(f32, @floatFromInt(col % 7)) / 8.0, e));
            std.mem.writeInt(u16, sc[2 * (g * n + col) ..][0..2], @bitCast(v), .little);
        }
    }
    // make group 1's term the negative of group 0's in every column (same |x q|, opposite sign)
    for (0..n) |col| {
        var k0: usize = 0;
        var k1: usize = 0;
        for (0..128) |j| {
            if (int4.code(.{ .qweight = qw, .scales = sc, .n_full = n, .k_full = kk }, col, j) != 8) k0 = j;
            if (int4.code(.{ .qweight = qw, .scales = sc, .n_full = n, .k_full = kk }, col, 128 + j) != 8) k1 = 128 + j;
        }
        x[k1] = f32ToBf16(-bf16ToF32(x[k0]));
        const c0 = int4.code(.{ .qweight = qw, .scales = sc, .n_full = n, .k_full = kk }, col, k0);
        const word_i = (k1 / 8) * n + col;
        var w = std.mem.readInt(u32, qw[4 * word_i ..][0..4], .little);
        const sh: u5 = @intCast(4 * (k1 % 8));
        w = (w & ~(@as(u32, 0xF) << sh)) | (@as(u32, c0) << sh);
        std.mem.writeInt(u32, qw[4 * word_i ..][0..4], w, .little);
        std.mem.writeInt(u16, sc[2 * (128 / 128 * n + col) ..][0..2], std.mem.readInt(u16, sc[2 * col ..][0..2], .little), .little);
    }
    const src: int4.Source = .{ .qweight = qw, .scales = sc, .n_full = n, .k_full = kk };
    const g: Gen = .{ .qweight = qw, .scales = sc, .n = n, .k = kk };
    const p = try packDev(gpa, dv, g, .{ .n0 = 0, .n = n, .k0 = 0, .k = kk, .gs = gs });
    const m: int4.Mat = .{ .w = p[0], .s = p[1], .n = @intCast(n), .k = @intCast(kk), .gs = @intCast(gs) };
    const xd = try dv.put(std.mem.sliceAsBytes(x));
    const out = try dv.zeros(n * 4);
    try int4.dense(k, s, xd, kk, m, out, true, 1);
    try s.synchronize();
    var got: [64]f32 = undefined;
    try dv.get(out, std.mem.sliceAsBytes(&got));
    var bad: usize = 0;
    var nonzero: usize = 0;
    for (0..n) |col| {
        const e = emulate(src, col, 0, kk, gs, x);
        nonzero += @intFromBool(e.y != 0);
        if (@as(u32, @bitCast(got[col])) != @as(u32, @bitCast(e.y))) bad += 1;
    }
    if (bad > 0) rep.ok = false;
    std.debug.print("  group order (gs {d}, cancelling groups, exact group sums): {d} of {d} columns bit-equal to fmaf in group order ({d} nonzero){s}\n", .{ gs, n - bad, n, nonzero, if (bad > 0) " FAIL" else "" });
}

/// Routed experts on the plan against the dense kernel on each pair's expert.
fn checkExperts(gpa: Allocator, k: *const int4.Kernels, ops: kern.Ops, dv: *Dev, r: std.Random, E: usize, ni: usize, full_ni: usize, lo: usize, D: usize, top: usize, R: usize, rep: *Report) !void {
    const s = ops.s;
    const slots = top + 1;
    const gs_down: usize = if (lo % 128 != 0 or ni % 128 != 0) 64 else 128;
    // per expert: gate [ni rows of full_ni], up, down [D, full_ni] with input columns [lo, lo + ni)
    const up_words = 2 * int4.wordsOf(ni, D);
    const up_sc = 2 * int4.scalesOf(ni, D, 128);
    const dn_words = int4.wordsOf(D, ni);
    const dn_sc = int4.scalesOf(D, ni, gs_down);
    const uw = try gpa.alloc(u32, E * up_words);
    defer gpa.free(uw);
    const us = try gpa.alloc(u16, E * up_sc);
    defer gpa.free(us);
    const dw = try gpa.alloc(u32, E * dn_words);
    defer gpa.free(dw);
    const ds = try gpa.alloc(u16, E * dn_sc);
    defer gpa.free(ds);
    for (0..E) |e| {
        for (0..2) |mi| {
            var g = try Gen.init(gpa, r, full_ni, D);
            defer g.deinit(gpa);
            const sl: int4.Slice = .{ .n0 = lo, .n = ni, .k0 = 0, .k = D, .gs = 128 };
            try int4.packWords(g.src(), sl, uw[(e * 2 + mi) * int4.wordsOf(ni, D) ..][0..int4.wordsOf(ni, D)]);
            try int4.packScales(g.src(), sl, us[(e * 2 + mi) * int4.scalesOf(ni, D, 128) ..][0..int4.scalesOf(ni, D, 128)]);
        }
        var g = try Gen.init(gpa, r, D, full_ni);
        defer g.deinit(gpa);
        const sl: int4.Slice = .{ .n0 = 0, .n = D, .k0 = lo, .k = ni, .gs = gs_down };
        try int4.packWords(g.src(), sl, dw[e * dn_words ..][0..dn_words]);
        try int4.packScales(g.src(), sl, ds[e * dn_sc ..][0..dn_sc]);
    }
    const ex: int4.Experts = .{ .up = try dv.put(std.mem.sliceAsBytes(uw)), .up_s = try dv.put(std.mem.sliceAsBytes(us)), .down = try dv.put(std.mem.sliceAsBytes(dw)), .down_s = try dv.put(std.mem.sliceAsBytes(ds)), .count = @intCast(E), .width = @intCast(ni), .dims = @intCast(D), .gs_down = @intCast(gs_down) };
    // picks: each row's top distinct experts, then the shared slot (expert E, skipped)
    const picks = try gpa.alloc(i32, R * slots);
    defer gpa.free(picks);
    for (0..R) |row| {
        for (0..top) |j| {
            while (true) {
                const e: i32 = @intCast(if (r.uintLessThan(u32, 4) == 0) r.uintLessThan(u32, 2) else r.uintLessThan(u32, @intCast(E)));
                const dup = for (picks[row * slots .. row * slots + j]) |q| {
                    if (q == e) break true;
                } else false;
                if (!dup) {
                    picks[row * slots + j] = e;
                    break;
                }
            }
        }
        picks[row * slots + top] = @intCast(E);
    }
    const pairs = R * slots;
    const pk = try dv.put(std.mem.sliceAsBytes(picks));
    const pl: kern.Plan = .{ .members = try dv.zeros(pairs * 4), .items = try dv.zeros(3 * 4 * (pairs + E + 1)), .counts = try dv.zeros(64), .rank = try dv.zeros(pairs * 4), .hist = try dv.zeros(4 * (E + 1) * ((pairs + 1023) / 1024 + 1)) };
    try ops.plan(pk, pairs, E + 1, kern.plan_tile, pl);
    const x = try randRows(gpa, r, R, D);
    defer gpa.free(x);
    const xd = try dv.put(std.mem.sliceAsBytes(x));
    const act = try dv.zeros(pairs * ni * 2);
    const y = try dv.zeros(pairs * D * 4);
    const yb = try dv.zeros(pairs * D * 2);
    // a sentinel in the shared slot's rows: the kernels never write them
    const sentinel = try gpa.alloc(u8, pairs * D * 4);
    defer gpa.free(sentinel);
    @memset(sentinel, 0x5A);
    try (cuda.DeviceBuffer{ .d = dv.d, .ptr = y, .len = pairs * D * 4 }).upload(0, sentinel);
    try int4.gateUp(k, s, xd, D, ex, pl, slots, E + 1, act, R, @intCast(E), kern.plan_tile);
    try int4.down(k, s, act, ni, ex, pl, slots, E + 1, y, true, R, @intCast(E), kern.plan_tile);
    try int4.down(k, s, act, ni, ex, pl, slots, E + 1, yb, false, R, @intCast(E), kern.plan_tile);
    try s.synchronize();
    const ha = try gpa.alloc(u16, pairs * ni);
    defer gpa.free(ha);
    try dv.get(act, std.mem.sliceAsBytes(ha));
    const hy = try gpa.alloc(f32, pairs * D);
    defer gpa.free(hy);
    try dv.get(y, std.mem.sliceAsBytes(hy));
    const hyb = try gpa.alloc(u16, pairs * D);
    defer gpa.free(hyb);
    try dv.get(yb, std.mem.sliceAsBytes(hyb));
    // NT invariance of the plan kernels (forced 1, 2, 4)
    const act2 = try dv.zeros(pairs * ni * 2);
    const y2 = try dv.zeros(pairs * D * 4);
    const items = kern.maxItems(pairs, E + 1, kern.plan_tile);
    const ha2 = try gpa.alloc(u16, pairs * ni);
    defer gpa.free(ha2);
    const hy2 = try gpa.alloc(f32, pairs * D);
    defer gpa.free(hy2);
    const dk: usize = if (gs_down == 128) 1 else 3;
    // the prompt plan (items of 64 pairs, two row tiles a pass) gives the same bytes
    const pl64: kern.Plan = .{ .members = try dv.zeros(pairs * 4), .items = try dv.zeros(3 * 4 * (pairs + E + 1)), .counts = try dv.zeros(64), .rank = try dv.zeros(pairs * 4), .hist = try dv.zeros(4 * (E + 1) * ((pairs + 1023) / 1024 + 1)) };
    try ops.plan(pk, pairs, E + 1, int4.prompt_tile, pl64);
    {
        const act64 = try dv.zeros(pairs * ni * 2);
        const y64 = try dv.zeros(pairs * D * 4);
        try int4.gateUp(k, s, xd, D, ex, pl64, slots, E + 1, act64, R, @intCast(E), int4.prompt_tile);
        try int4.down(k, s, act, ni, ex, pl64, slots, E + 1, y64, true, R, @intCast(E), int4.prompt_tile);
        try s.synchronize();
        const h64 = try gpa.alloc(u16, pairs * ni);
        defer gpa.free(h64);
        try dv.get(act64, std.mem.sliceAsBytes(h64));
        const y64h = try gpa.alloc(f32, pairs * D);
        defer gpa.free(y64h);
        try dv.get(y64, std.mem.sliceAsBytes(y64h));
        for (0..pairs) |p| if (@as(usize, @intCast(picks[p])) < E and (!std.mem.eql(u16, ha[p * ni ..][0..ni], h64[p * ni ..][0..ni]) or !std.mem.eql(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(hy[p * D ..][0..D])), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(y64h[p * D ..][0..D]))))) {
            std.debug.print("  experts tile 64 / two row tiles: pair {d} differs from tile 16\n", .{p});
            rep.ok = false;
            break;
        };
    }
    for ([_]usize{ 1, 2, 4 }) |nt| {
        if (ni % (8 * nt) == 0) {
            try int4.expertsNt(k, s, 0, xd, D, slots, ex.up, ex.up_s, D, ni, pl, items, act2, @intCast(E), nt, 1);
            try s.synchronize();
            try dv.get(act2, std.mem.sliceAsBytes(ha2));
            for (0..pairs) |p| if (@as(usize, @intCast(picks[p])) < E and !std.mem.eql(u16, ha[p * ni ..][0..ni], ha2[p * ni ..][0..ni])) {
                std.debug.print("  experts gate/up NT {d}: pair {d} differs\n", .{ nt, p });
                rep.ok = false;
                break;
            };
        }
        try int4.expertsNt(k, s, dk, act, ni, 0, ex.down, ex.down_s, ni, D, pl, items, y2, @intCast(E), nt, 1);
        try s.synchronize();
        try dv.get(y2, std.mem.sliceAsBytes(hy2));
        for (0..pairs) |p| if (@as(usize, @intCast(picks[p])) < E and !std.mem.eql(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(hy[p * D ..][0..D])), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(hy2[p * D ..][0..D])))) {
            std.debug.print("  experts down NT {d}: pair {d} differs\n", .{ nt, p });
            rep.ok = false;
            break;
        };
    }
    // each pair against the dense kernel on its expert's matrices
    const one = try dv.zeros(D * 4);
    const host_one = try gpa.alloc(f32, @max(D, ni));
    defer gpa.free(host_one);
    const host_up = try gpa.alloc(f32, ni);
    defer gpa.free(host_up);
    var swiglu_ulp: usize = 0;
    var checked: usize = 0;
    for (0..pairs) |p| {
        const e: usize = @intCast(picks[p]);
        const row = p / slots;
        if (e == E) {
            const raw = std.mem.sliceAsBytes(hy[p * D ..][0..D]);
            if (!std.mem.allEqual(u8, raw, 0x5A)) {
                std.debug.print("  experts: the skipped slot of row {d} was written\n", .{row});
                rep.ok = false;
            }
            continue;
        }
        if (checked >= 64 and p % 7 != 0) continue;
        checked += 1;
        const gm: int4.Mat = .{ .w = ex.up + (e * 2) * int4.wordsOf(ni, D) * 4, .s = ex.up_s + (e * 2) * int4.scalesOf(ni, D, 128) * 2, .n = @intCast(ni), .k = @intCast(D) };
        try int4.dense(k, s, xd + row * D * 2, D, gm, one, true, 1);
        try s.synchronize();
        try dv.get(one, std.mem.sliceAsBytes(host_one[0..ni]));
        var um = gm;
        um.w += int4.wordsOf(ni, D) * 4;
        um.s += int4.scalesOf(ni, D, 128) * 2;
        try int4.dense(k, s, xd + row * D * 2, D, um, one, true, 1);
        try s.synchronize();
        try dv.get(one, std.mem.sliceAsBytes(host_up));
        for (0..ni) |c| {
            const gv = bf16ToF32(f32ToBf16(host_one[c]));
            const uv = bf16ToF32(f32ToBf16(host_up[c]));
            const want = f32ToBf16(bf16ToF32(f32ToBf16(gv / (1.0 + @exp(-gv)))) * uv);
            const got = ha[p * ni + c];
            if (got != want) {
                const d = @abs(@as(i32, got) - @as(i32, want));
                if (d > 1) {
                    std.debug.print("  experts gate/up pair {d} col {d}: {x} vs host SwiGLU {x}\n", .{ p, c, got, want });
                    rep.ok = false;
                    break;
                }
                swiglu_ulp += 1;
            }
        }
        const dm: int4.Mat = .{ .w = ex.down + e * dn_words * 4, .s = ex.down_s + e * dn_sc * 2, .n = @intCast(D), .k = @intCast(ni), .gs = @intCast(gs_down) };
        try int4.dense(k, s, act + p * ni * 2, ni, dm, one, true, 1);
        try s.synchronize();
        try dv.get(one, std.mem.sliceAsBytes(host_one[0..D]));
        if (!std.mem.eql(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(host_one[0..D])), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(hy[p * D ..][0..D])))) {
            std.debug.print("  experts down pair {d}: differs from the dense kernel on expert {d}\n", .{ p, e });
            rep.ok = false;
        }
        for (0..D) |c| if (hyb[p * D + c] != f32ToBf16(hy[p * D + c])) {
            std.debug.print("  experts down bf16 pair {d}: not the fp32 rounded\n", .{p});
            rep.ok = false;
            break;
        };
    }
    std.debug.print("  experts E {d} width {d} (of {d}, from {d}) D {d} top {d} rows {d}: plan == dense on {d} pairs; NT 1/2/4 and 64-pair items equal; skipped slot untouched; SwiGLU vs host: {d} outputs 1 bf16 ulp apart (host expf)\n", .{ E, ni, full_ni, lo, D, top, R, checked, swiglu_ulp });
}

fn timeIt(s: cuda.Stream, d: *const cuda.Driver, reps: usize, ctx: anytype, comptime f: fn (@TypeOf(ctx)) anyerror!void) !f64 {
    var a = try cuda.Event.init(d, true);
    defer a.deinit();
    var b = try cuda.Event.init(d, true);
    defer b.deinit();
    try f(ctx);
    try s.synchronize();
    try a.record(s);
    for (0..reps) |_| try f(ctx);
    try b.record(s);
    try b.synchronize();
    return @as(f64, try cuda.Event.elapsedMs(a, b)) * 1000.0 / @as(f64, @floatFromInt(reps));
}

/// Speeds: the head at 1 / 8 rows, the decode experts (one row: top-5 of 512) and a 4096-row prompt chunk's.
pub fn bench(gpa: Allocator, k: *const int4.Kernels, ops: kern.Ops, d: *const cuda.Driver, world: usize) !void {
    var dv: Dev = .{ .d = d, .gpa = gpa };
    defer dv.deinit();
    var prng = std.Random.DefaultPrng.init(99);
    const r = prng.random();
    const s = ops.s;
    const V: usize = 248320 / world;
    const D: usize = 2560;
    {
        // the head: random words (the bits do not matter for speed)
        const w = try dv.zeros(V * D / 2);
        const sc = try dv.zeros(V * D / 128 * 2);
        const m: int4.Mat = .{ .w = w, .s = sc, .n = @intCast(V), .k = D };
        const x = try dv.zeros(16 * D * 2);
        const out = try dv.zeros(16 * V * 2);
        for ([_]usize{ 1, 8, 16 }) |rows| {
            const C = struct { k: *const int4.Kernels, s: cuda.Stream, x: u64, m: int4.Mat, out: u64, rows: usize };
            const us = try timeIt(s, d, 50, C{ .k = k, .s = s, .x = x, .m = m, .out = out, .rows = rows }, struct {
                fn f(c: C) anyerror!void {
                    try int4.dense(c.k, c.s, c.x, D, c.m, c.out, false, c.rows);
                }
            }.f);
            const bytes: f64 = @floatFromInt(V * D / 2 + V * D / 128 * 2);
            std.debug.print("  head [{d}, {d}] int4, {d} rows: {d:.1} us ({d:.0} GB/s)\n", .{ V, D, rows, us, bytes / us / 1e3 });
        }
    }
    {
        const E: usize = 512;
        const ni: usize = 640 / world;
        const gs_down: usize = if (ni % 128 != 0) 64 else 128;
        const ex: int4.Experts = .{ .up = try dv.zeros(E * 2 * ni * D / 2), .up_s = try dv.zeros(E * 2 * ni * (D / 128) * 2), .down = try dv.zeros(E * D * ni / 2), .down_s = try dv.zeros(E * D * (ni / gs_down) * 2), .count = E, .width = @intCast(ni), .dims = D, .gs_down = @intCast(gs_down) };
        const top: usize = 5;
        const slots = top + 1;
        for ([_]usize{ 1, 4, 8, 32, 4096 }) |R| {
            const picks = try gpa.alloc(i32, R * slots);
            defer gpa.free(picks);
            for (0..R) |row| {
                for (0..top) |j| picks[row * slots + j] = @intCast((r.uintLessThan(u32, @intCast(E / top))) * top + j);
                picks[row * slots + top] = @intCast(E);
            }
            const pairs = R * slots;
            const pk = try dv.put(std.mem.sliceAsBytes(picks));
            const pl: kern.Plan = .{ .members = try dv.zeros(pairs * 4), .items = try dv.zeros(3 * 4 * (pairs + E + 1)), .counts = try dv.zeros(64), .rank = try dv.zeros(pairs * 4), .hist = try dv.zeros(4 * (E + 1) * ((pairs + 1023) / 1024 + 1)) };
            try ops.plan(pk, pairs, E + 1, int4.tileFor(pairs), pl);
            const x = try dv.zeros(R * D * 2);
            const act = try dv.zeros(pairs * ni * 2);
            const y = try dv.zeros(pairs * D * 2);
            const C = struct { k: *const int4.Kernels, s: cuda.Stream, x: u64, ex: int4.Experts, pl: kern.Plan, slots: usize, E: usize, act: u64, y: u64, R: usize, part: u8 };
            var t: [2]f64 = undefined;
            for (0..2) |part| {
                t[part] = try timeIt(s, d, if (R > 64) 5 else 50, C{ .k = k, .s = s, .x = x, .ex = ex, .pl = pl, .slots = slots, .E = E, .act = act, .y = y, .R = R, .part = @intCast(part) }, struct {
                    fn f(c: C) anyerror!void {
                        const tile = int4.tileFor(c.R * c.slots);
                        if (c.part == 0) try int4.gateUp(c.k, c.s, c.x, D, c.ex, c.pl, c.slots, c.E + 1, c.act, c.R, @intCast(c.E), tile) else try int4.down(c.k, c.s, c.act, c.ex.width, c.ex, c.pl, c.slots, c.E + 1, c.y, false, c.R, @intCast(c.E), tile);
                    }
                }.f);
            }
            const used: f64 = @floatFromInt(@min(E, R * top));
            const bytes = used * @as(f64, @floatFromInt(3 * ni * D / 2));
            std.debug.print("  experts width {d}, {d} rows (top {d}): gate/up {d:.1} us, down {d:.1} us ({d:.0} GB/s of distinct expert weights)\n", .{ ni, R, top, t[0], t[1], bytes / (t[0] + t[1]) / 1e3 });
        }
    }
}

/// Every check; true when all pass.
pub fn check(gpa: Allocator, ctx: *const cuda.Context, ops: kern.Ops) !bool {
    var k = try int4.Kernels.load(ctx);
    defer k.deinit();
    var dv: Dev = .{ .d = ctx.d, .gpa = gpa };
    defer dv.deinit();
    var prng = std.Random.DefaultPrng.init(1234);
    const r = prng.random();
    var rep: Report = .{};
    const rows = [_]usize{ 1, 2, 3, 7, 8, 15, 16, 17, 31, 33, 64, 100 };
    // the head's shape (a slice of the vocabulary), gate/up, down at one GPU and a TP=2 rank's down (groups of 64)
    try checkDense(gpa, &k, ops.s, &dv, r, 4096, 2560, 0, 2560, 128, &rows, &rep);
    try checkDense(gpa, &k, ops.s, &dv, r, 640, 2560, 0, 2560, 128, &rows, &rep);
    try checkDense(gpa, &k, ops.s, &dv, r, 2560, 640, 0, 640, 128, &rows, &rep);
    try checkDense(gpa, &k, ops.s, &dv, r, 2560, 640, 320, 320, 64, &rows, &rep);
    try checkDense(gpa, &k, ops.s, &dv, r, 2560, 640, 0, 320, 64, &rows, &rep);
    try checkOrder(gpa, &k, ops.s, &dv, r, 128, &rep);
    try checkOrder(gpa, &k, ops.s, &dv, r, 64, &rep);
    try checkExperts(gpa, &k, ops, &dv, r, 24, 640, 640, 0, 2560, 5, 37, &rep);
    try checkExperts(gpa, &k, ops, &dv, r, 24, 320, 640, 320, 2560, 5, 37, &rep);
    try checkExperts(gpa, &k, ops, &dv, r, 24, 320, 640, 0, 2560, 5, 3, &rep);
    try checkExperts(gpa, &k, ops, &dv, r, 8, 640, 640, 0, 2560, 5, 130, &rep);
    try bench(gpa, &k, ops, ctx.d, 1);
    try bench(gpa, &k, ops, ctx.d, 2);
    return rep.ok;
}
