//! TensorFold glm5_next/cuda/exl3_mm.py routed and cuda/experts.py route, authored by Ash Hart and contributors.
//! Reuses TensorFold's experts image and jschmied's Zig CUDA launch interface (PR #443).
const std = @import("std");
const cuda = @import("cuda");
const mm = @import("cuda_matmuls.zig");
const Entries = mm.Entries;
pub const Plan = struct { members: u64, items: u64, counts: u64, rank: u64, hist: u64, tile: usize = 16, experts: usize };
pub const Experts = struct { gt: u64, ut: u64, dt: u64, suh_g: u64, suh_u: u64, svh_g: u64, svh_u: u64, suh_d: u64, svh_d: u64, shared_suh: u64 = 0, dims: usize = 4096, width: usize, count: usize };
/// Capacity is the allocated rows, not the active rows. Z: 2*4*capacity*slots*max(D,NI) floats.
pub const Scratch = struct { xg: u64, xu: u64, xd: u64, z: u64, done: u64, order: u64 = 0, capacity: usize, slots: usize = 9 };
fn ci(v: usize) c_int {
    return @intCast(v);
}
fn cfg(x: usize, y: usize, z: usize, threads: usize, shared: u32) cuda.launch.Config {
    return .{ .grid = .{ .x = @intCast(x), .y = @intCast(y), .z = @intCast(z) }, .block = .{ .x = @intCast(threads) }, .shared = shared };
}
pub fn maxItems(pairs: usize, experts: usize, tile: usize) usize {
    return @min(pairs, experts) + pairs / tile;
}
pub fn plan(e: *const Entries, stream: cuda.Stream, picks: u64, pairs: usize, p: Plan) !void {
    if (pairs == 0 or p.experts == 0 or p.experts > 1024 or (p.tile != 16 and p.tile != 64)) return error.Invalid;
    var a: cuda.Args = .{};
    a.add(picks);
    a.add(ci(pairs));
    a.add(ci(p.experts));
    if (pairs <= 1024) {
        a.add(ci(p.tile));
        for ([_]u64{ p.members, p.items, p.counts }) |v| a.add(v);
        return e.launch(3, "_ZN10tf_experts11plan_kernelEPKiiiiPiS2_S2_", cfg(1, 1, 1, 1024, 0), stream, &a);
    }
    if (p.rank == 0 or p.hist == 0) return error.Invalid;
    const blocks = (pairs + 1023) / 1024;
    a.add(p.rank);
    a.add(p.hist);
    try e.launch(3, "_ZN10tf_experts9plan_rankEPKiiiPiS2_", cfg(blocks, 1, 1, 1024, 0), stream, &a);
    var b: cuda.Args = .{};
    for ([_]usize{ blocks, p.experts, p.tile }) |v| b.add(ci(v));
    for ([_]u64{ p.hist, p.items, p.counts }) |v| b.add(v);
    try e.launch(3, "_ZN10tf_experts12plan_offsetsEiiiPiS0_S0_", cfg(1, 1, 1, 1024, 0), stream, &b);
    var c: cuda.Args = .{};
    c.add(picks);
    c.add(ci(pairs));
    c.add(ci(p.experts));
    for ([_]u64{ p.rank, p.hist, p.members }) |v| c.add(v);
    try e.launch(3, "_ZN10tf_experts12plan_scatterEPKiiiS1_S1_Pi", cfg((pairs + 255) / 256, 1, 1, 256, 0), stream, &c);
}
/// Default recipe: nc loads one step ahead, fused down, 64-pair prompt passes, optional prompt item order.
pub fn routed(e: *const Entries, stream: cuda.Stream, x: u64, stride: usize, picks: u64, p: Plan, ex: Experts, s: Scratch, y: u64, rows: usize, limit: f32, force_grouped: bool) !void {
    if (rows == 0 or rows > s.capacity or ex.dims == 0 or ex.width == 0 or ex.dims % 256 != 0 or ex.width % 128 != 0 or (p.tile != 16 and p.tile != 64) or s.slots == 0) return error.Invalid;
    const pairs = s.capacity * s.slots;
    const items = maxItems(rows * s.slots, p.experts, p.tile);
    const dec = ex.dims == 4096 and (ex.width == 1024 or ex.width == 768 or ex.width == 640 or ex.width == 512) and rows < 64 and !force_grouped;
    const shared = ex.shared_suh != 0 and (p.tile == 64 or dec);
    const xu = if (shared) s.xg else s.xu;
    if (shared) try e.glm_exl3_rot_1(cfg(rows, ex.dims / 128, 1, 32, 0), stream, x, ci(stride), 0, ex.shared_suh, ex.shared_suh, s.xg, s.xg, ci(ex.dims), 1) else try e.glm_exl3_rot_0(cfg(rows * s.slots, ex.dims / 128, 2, 32, 0), stream, x, ci(stride), picks, ex.suh_g, ex.suh_u, s.xg, s.xu, ci(ex.dims), ci(s.slots));
    if (p.tile == 64) {
        if (s.order != 0) try e.glm_exl3_order(cfg(1, 1, 1, 1024, 0), stream, p.items, p.counts, s.order, ci(items), ci(ex.count));
        const gc = cfg(ex.width / 128, items, 1, 256, mm.promptShared(shared));
        if (shared) try e.glm_exl3_prompt_gu_1(gc, stream, s.xg, xu, ex.gt, ex.ut, p.items, p.counts, p.members, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, ci(ex.dims), ci(ex.width), ci(ex.count), limit, ci(s.slots), s.order) else try e.glm_exl3_prompt_gu_0(gc, stream, s.xg, xu, ex.gt, ex.ut, p.items, p.counts, p.members, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, ci(ex.dims), ci(ex.width), ci(ex.count), limit, ci(s.slots), s.order);
        try e.glm_exl3_prompt_down(cfg(ex.dims / 256, items, 1, 256, mm.promptShared(false)), stream, s.xd, ex.dt, p.items, p.counts, p.members, ex.svh_d, y, ci(ex.width), ci(ex.dims), ci(ex.count), s.order);
        return;
    }
    // exl3_mm.py: R >= 64 leaves dec and runs grouped_kernel with n blocks of an item adjacent. Gate/up also sums
    // its K splits in the program (SEQ) and the epilogue reads that one partial. Same bits as the split epilogue.
    const wide = !dec and rows >= 64;
    const logging = try loggingOn(stream);
    const t0: ?cuda.Event = if (logging) try mark(stream) else null;
    const gc = if (wide) cfg(ex.width / 128, items, 2, 128, 0) else cfg(items, ex.width / 128, 8, 128, 0);
    if (dec) {
        if (shared) try e.glm_exl3_dec_16_0_1(gc, stream, s.xg, xu, ex.gt, ex.ut, p.items, p.counts, p.members, s.z, ci(ex.dims), ci(ex.width), ci(pairs), 4, ci(ex.count), ci(s.slots), ex.svh_g, ex.svh_u, ex.suh_d, s.xd, limit, s.done) else try e.glm_exl3_dec_16_0_0(gc, stream, s.xg, xu, ex.gt, ex.ut, p.items, p.counts, p.members, s.z, ci(ex.dims), ci(ex.width), ci(pairs), 4, ci(ex.count), ci(s.slots), ex.svh_g, ex.svh_u, ex.suh_d, s.xd, limit, s.done);
    } else if (wide) {
        try e.glm_exl3_grouped_seq(gc, stream, s.xg, xu, ex.gt, ex.ut, p.items, p.counts, p.members, s.z, ci(ex.dims), ci(ex.width), ci(pairs), 4, ci(ex.count));
    } else try e.glm_exl3_grouped(gc, stream, s.xg, xu, ex.gt, ex.ut, p.items, p.counts, p.members, s.z, ci(ex.dims), ci(ex.width), ci(pairs), 4, ci(ex.count));
    const gsk: c_int = if (wide) 1 else 4;
    try e.glm_exl3_gateup(cfg(rows * s.slots, ex.width / 128, 1, 32, 0), stream, s.z, picks, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, ci(pairs), ci(ex.width), gsk, ci(s.slots), limit);
    const t1: ?cuda.Event = if (logging) try mark(stream) else null;
    const dc = if (wide) cfg(ex.dims / 128, items, 1, 128, 0) else cfg(items, ex.dims / 128, 1, 128, 0);
    if (dec) {
        var args: cuda.Args = .{};
        for ([_]u64{ s.xd, s.xd, ex.dt, ex.dt, p.items, p.counts, p.members, s.z }) |v| args.add(v);
        for ([_]usize{ ex.width, ex.dims, pairs, 1, ex.count, s.slots }) |v| args.add(ci(v));
        for ([_]u64{ ex.svh_d, ex.svh_d, ex.svh_d, y }) |v| args.add(v);
        args.add(@as(f32, 0));
        args.add(s.done);
        var buf: [48]u8 = undefined;
        const symbol = try printZ(&buf, "glm_exl3_dec_{d}_1_0", .{ex.width / 64});
        try e.launch(2, symbol, dc, stream, &args);
    } else if (wide) {
        try e.glm_exl3_grouped_nf(dc, stream, s.xd, s.xd, ex.dt, ex.dt, p.items, p.counts, p.members, s.z, ci(ex.width), ci(ex.dims), ci(pairs), 1, ci(ex.count));
        try e.glm_exl3_down(cfg(rows * s.slots, ex.dims / 128, 1, 32, 0), stream, s.z, picks, ex.svh_d, y, ci(pairs), ci(ex.dims), 1, ci(s.slots));
    } else {
        try e.glm_exl3_grouped(dc, stream, s.xd, s.xd, ex.dt, ex.dt, p.items, p.counts, p.members, s.z, ci(ex.width), ci(ex.dims), ci(pairs), 1, ci(ex.count));
        try e.glm_exl3_down(cfg(rows * s.slots, ex.dims / 128, 1, 32, 0), stream, s.z, picks, ex.svh_d, y, ci(pairs), ci(ex.dims), 1, ci(s.slots));
    }
    const t2: ?cuda.Event = if (logging) try mark(stream) else null;
    try noteItems(stream, p.counts, rows, t0, t1, t2);
}
test "plan bounds and prompt shared memory match reference" {
    try std.testing.expectEqual(@as(usize, 9), maxItems(9, 289, 16));
    try std.testing.expectEqual(@as(usize, 289 + 1152), maxItems(18432, 289, 16));
    try std.testing.expectEqual(@as(u32, 43008), mm.promptShared(false));
    try std.testing.expectEqual(@as(u32, 33792), mm.promptShared(true));
}

fn printZ(buf: []u8, comptime format: []const u8, args: anytype) ![:0]u8 {
    return std.fmt.bufPrintSentinel(buf, format, args, 0);
}

/// TF_GLM_EXL3_LOG=1 and TF_GLM_EAGER=1: expert-tile count and the two expert launches' GPU time. Graphs skip it.
const Acc = struct { rows: usize = 0, layers: u32 = 0, items: u64 = 0, gu: f32 = 0, dn: f32 = 0 };
var acc: Acc = .{};

fn loggingOn(stream: cuda.Stream) !bool {
    if (std.c.getenv("TF_GLM_EXL3_LOG") == null) return false;
    return try cuda.graph.captureStatus(stream) == .none;
}

fn mark(stream: cuda.Stream) !cuda.Event {
    const ev = try cuda.Event.init(stream.d, true);
    try ev.record(stream);
    return ev;
}

fn take(ev: ?cuda.Event) void {
    var e = ev orelse return;
    e.deinit();
}

fn noteItems(stream: cuda.Stream, counts: u64, rows: usize, t0: ?cuda.Event, t1: ?cuda.Event, t2: ?cuda.Event) !void {
    defer {
        take(t0);
        take(t1);
        take(t2);
    }
    if (t0 == null) return;
    try stream.synchronize();
    var raw: [4]u8 = undefined;
    try stream.d.check(stream.d.api.cuMemcpyDtoH_v2(&raw, counts, 4), "exl3 counts");
    const n = std.mem.readInt(i32, &raw, .little);
    acc.rows = rows;
    acc.layers += 1;
    if (n > 0) acc.items += @intCast(n);
    if (t0 != null and t1 != null) acc.gu += try t0.?.elapsedMs(t1.?);
    if (t1 != null and t2 != null) acc.dn += try t1.?.elapsedMs(t2.?);
}

pub fn logFlush() void {
    if (acc.layers == 0) return;
    const layers: f32 = @floatFromInt(acc.layers);
    const items: f32 = @floatFromInt(acc.items);
    // 8-row TP=2 decode launches glm_exl3_dec_16_0_1 as (maxItems, width/128, 8) and glm_exl3_dec_16_1_0 as
    // (maxItems, dims/128, 1). maxItems(72, 289, 16) is 76; blocks past the tile count return before the trellis load.
    std.log.info("exl3 rows={d} moe-layers={d} expert-tiles={d} ({d:.1}/layer) gate+up {d:.2} ms down {d:.2} ms", .{ acc.rows, acc.layers, acc.items, items / layers, acc.gu, acc.dn });
    acc = .{};
}
