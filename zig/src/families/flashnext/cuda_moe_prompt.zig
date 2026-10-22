//! The routed NVFP4 experts of a prompt chunk (work/research/R2-prefill.md W-B; zig/kernels/cuda/fn_experts_prompt.cu):
//! a warp applies each loaded and decoded weight block to T row tiles of 16 pairs instead of one, on a plan whose items
//! hold `tile` pairs (32 by default) instead of 16. The bits are nvfp4_expert_kernel's (the same per-pair math in the
//! same order; `check` compares bytes for several tiles and T). Measured on GB10 (experts-check, 320-wide TP=2 rank):
//! gate/up 1.11x at 4096 rows but 0.87x at 2048, down 1.11-1.18x from 2048 rows, so gate/up takes it from 4096 rows
//! (`gu_rows`), down from 2048 (`down_rows`); decode windows and short chunks keep Python's kernel.
//! `TF_FLASHNEXT_PROMPT_EXPERTS=0` keeps Python's kernel for every chunk.
const std = @import("std");
const cuda = @import("cuda");
const kern = @import("cuda_kernels.zig");

pub const env = "TF_FLASHNEXT_PROMPT_EXPERTS";

/// Pairs a plan item holds by default (any size works: a warp takes 16 * T of them a pass).
pub const default_tile: usize = 32;
/// Rows from which a chunk takes it (fewer rows: few pairs an expert, Python's 16-pair items lose nothing).
pub const min_rows: usize = 129;

const threads: usize = 128;

pub const Prompt4 = struct {
    mod: cuda.Module,
    gu: [3]cuda.Function, // row tiles a warp T = 2, 3, 4
    f32: [2]cuda.Function, // T = 2, 4
    b16: [2]cuda.Function,
    resident: [3]usize,
    /// the plan's item size and the row tiles a warp (TF_FLASHNEXT_PROMPT_EXPERTS_CFG=tile,Tgu,Tdown; same bits)
    tile: usize = default_tile,
    t_gu: usize = 2,
    t_down: usize = 2,
    /// rows from which gate/up and down take this kernel (below: Python's on 16-pair items, faster there;
    /// TF_FLASHNEXT_PROMPT_EXPERTS_ROWS=gu,down)
    gu_rows: usize = 4096,
    down_rows: usize = 2048,

    pub fn init(ctx: *const cuda.Context) !Prompt4 {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var p: Prompt4 = .{ .mod = undefined, .gu = undefined, .f32 = undefined, .b16 = undefined, .resident = undefined };
        p.mod = try cuda.Module.load(ctx.d, cuda.kernels.fn_experts_prompt);
        errdefer p.mod.unload();
        p.gu = .{ try p.mod.function("fn_prompt4_gu_t2"), try p.mod.function("fn_prompt4_gu_t3"), try p.mod.function("fn_prompt4_gu_t4") };
        p.f32 = .{ try p.mod.function("fn_prompt4_f32_t2"), try p.mod.function("fn_prompt4_f32_t4") };
        p.b16 = .{ try p.mod.function("fn_prompt4_b16_t2"), try p.mod.function("fn_prompt4_b16_t4") };
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        for (p.gu, &p.resident) |f, *n| n.* = @as(usize, @max(1, try f.occupancy(threads, 0))) * sms;
        if (std.c.getenv("TF_FLASHNEXT_PROMPT_EXPERTS_CFG")) |v| {
            var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
            if (it.next()) |x| p.tile = std.fmt.parseInt(usize, x, 10) catch default_tile;
            if (it.next()) |x| p.t_gu = std.fmt.parseInt(usize, x, 10) catch 2;
            if (it.next()) |x| p.t_down = std.fmt.parseInt(usize, x, 10) catch 2;
        }
        if (std.c.getenv("TF_FLASHNEXT_PROMPT_EXPERTS_ROWS")) |v| {
            var it = std.mem.tokenizeScalar(u8, std.mem.span(v), ',');
            if (it.next()) |x| p.gu_rows = std.fmt.parseInt(usize, x, 10) catch 4096;
            if (it.next()) |x| p.down_rows = std.fmt.parseInt(usize, x, 10) catch 2048;
        }
        if (p.tile < 16 or p.tile > 1024 or p.tile % 16 != 0) return error.BadPromptTile;
        return p;
    }

    pub fn deinit(p: *Prompt4) void {
        p.mod.unload();
    }

    /// experts.route with items of `tile` pairs (the members are the same as with 16: pairs by expert, in order).
    pub fn plan(q: *const Prompt4, o: kern.Ops, picks: u64, pairs: usize, experts: usize, p: kern.Plan) !void {
        return planTile(o, picks, pairs, experts, q.tile, p);
    }


    fn run(p: *const Prompt4, s: cuda.Stream, f: cuda.Function, x: u64, x_stride: usize, slots: usize, w: u64, scale: u64, kg: usize, nb: usize, pl: kern.Plan, out: u64, n: usize, limit: f32, skip: c_int, max_items: usize) !void {
        const units = max_items * nb;
        const grid = @min((units + 3) / 4, p.resident[0]);
        if (grid < 1) return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(int(x_stride));
        a.add(int(slots));
        a.add(w);
        a.add(scale);
        a.add(int(kg));
        a.add(int(nb));
        for ([_]u64{ pl.items, pl.counts, pl.members, out }) |v| a.add(v);
        a.add(int(n));
        a.add(limit);
        a.add(skip);
        try go(s, f, .{ grid, 1, 1 }, threads, &a);
    }

    /// nvfp4/experts.gate_up on this plan: x [R, D] -> out [R * slots, NI] bf16 SwiGLU of each routed pair.
    pub fn gateUp(p: *const Prompt4, s: cuda.Stream, x: u64, x_stride: usize, ex: kern.Experts4, pl: kern.Plan, plan_slots: usize, plan_experts: usize, out: u64, rows: usize, skip: c_int) !void {
        const f = switch (p.t_gu) {
            2 => p.gu[0],
            3 => p.gu[1],
            4 => p.gu[2],
            else => return error.BadRowTiles,
        };
        try p.run(s, f, x, x_stride, plan_slots, ex.up, ex.up_scale, ex.dims / 32, ex.width / 32, pl, out, ex.width, ex.limit, skip, kern.maxItems(rows * plan_slots, plan_experts, p.tile));
    }

    /// nvfp4/experts.down on this plan: act [R * slots, NI] -> out [R * slots, D], fp32 (`f32_out`) or bf16.
    pub fn down(p: *const Prompt4, s: cuda.Stream, act: u64, act_stride: usize, ex: kern.Experts4, pl: kern.Plan, plan_slots: usize, plan_experts: usize, out: u64, f32_out: bool, rows: usize, skip: c_int) !void {
        const fs = if (f32_out) p.f32 else p.b16;
        const f = switch (p.t_down) {
            2 => fs[0],
            4 => fs[1],
            else => return error.BadRowTiles,
        };
        try p.run(s, f, act, act_stride, 0, ex.down, ex.down_scale, ex.width / 32, ex.dims / 32, pl, out, ex.dims, 0, skip, kern.maxItems(rows * plan_slots, plan_experts, p.tile));
    }
};

/// experts.route with items of `tile` pairs (the members do not depend on the tile).
pub fn planTile(o: kern.Ops, picks: u64, pairs: usize, experts: usize, tile: usize, p: kern.Plan) !void {
    if (true) {
        if (experts > kern.plan_max_experts) return error.Invalid;
        const k = o.k;
        if (pairs <= kern.plan_small) {
            var a: cuda.Args = .{};
            a.add(picks);
            for ([_]usize{ pairs, experts, tile }) |v| a.add(int(v));
            for ([_]u64{ p.members, p.items, p.counts }) |v| a.add(v);
            return go(o.s, k.plan_small, .{ 1, 1, 1 }, 1024, &a);
        }
        const nblk = (pairs + 1023) / 1024;
        var a: cuda.Args = .{};
        a.add(picks);
        a.add(int(pairs));
        a.add(int(experts));
        a.add(p.rank);
        a.add(p.hist);
        try go(o.s, k.plan_rank, .{ nblk, 1, 1 }, 1024, &a);
        var b: cuda.Args = .{};
        for ([_]usize{ nblk, experts, tile }) |v| b.add(int(v));
        for ([_]u64{ p.hist, p.items, p.counts }) |v| b.add(v);
        try go(o.s, k.plan_offsets, .{ 1, 1, 1 }, 1024, &b);
        var c: cuda.Args = .{};
        c.add(picks);
        c.add(int(pairs));
        c.add(int(experts));
        for ([_]u64{ p.rank, p.hist, p.members }) |v| c.add(v);
        try go(o.s, k.plan_scatter, .{ (pairs + 255) / 256, 1, 1 }, 256, &c);
    }
}

fn int(x: usize) c_int {
    return @intCast(x);
}

fn go(s: cuda.Stream, f: cuda.Function, grid: [3]usize, block: usize, args: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = @intCast(block) } }, s, args);
}

// -- the check: Python's kernel on 16-pair items against this one on 128-pair items, bytes compared --------------

fn bf16Bits(x: f32) u16 {
    const b: u32 = @bitCast(x);
    const r = b + 0x7FFF + ((b >> 16) & 1);
    return @intCast(r >> 16);
}

fn upload(d: *const cuda.Driver, bytes: []const u8) !cuda.DeviceBuffer {
    var b = try cuda.DeviceBuffer.alloc(d, @max(bytes.len, 256));
    errdefer b.free();
    try b.upload(0, bytes);
    return b;
}

/// Random packed experts (code words any bits, e4m3 scale bytes finite), skewed routing (a few experts take
/// hundreds of pairs), every chunk size in `rows`, width `ni` (320 at TP=2, 640 at TP=1): gate/up and both down
/// faces of both kernels compared byte for byte, then timed. True when all equal.
pub fn check(gpa: std.mem.Allocator, ctx: *const cuda.Context, ops: kern.Ops, p: *const Prompt4, ni: usize, rows: []const usize, skew: bool) !bool {
    const d = ctx.d;
    const E: usize = 512;
    const D: usize = 2560;
    const top: usize = 10;
    const slots = top + 1;
    var prng = std.Random.DefaultPrng.init(0xe4e4_0001 + ni);
    const r = prng.random();
    // blocks: words then scale bytes, a (32 columns, 32 inputs) block of 144 int32
    const blk_words = 144;
    const gu_n = E * (ni / 32) * (D / 32) * 2 * blk_words;
    const dn_n = E * (D / 32) * (ni / 32) * blk_words;
    const host_w = try gpa.alloc(u32, @max(gu_n, dn_n));
    defer gpa.free(host_w);
    const fillBlocks = struct {
        fn f(rr: std.Random, w: []u32) void {
            for (w, 0..) |*v, i| {
                if (i % blk_words < 128) {
                    v.* = rr.int(u32);
                } else {
                    var x: u32 = 0;
                    for (0..4) |q| x |= (@as(u32, rr.intRangeAtMost(u8, 0x18, 0x5F)) | (if (rr.uintLessThan(u8, 8) == 0) @as(u32, 0x80) else 0)) << @intCast(8 * q);
                    v.* = x;
                }
            }
        }
    }.f;
    fillBlocks(r, host_w[0..gu_n]);
    var w_gu = try upload(d, std.mem.sliceAsBytes(host_w[0..gu_n]));
    defer w_gu.free();
    fillBlocks(r, host_w[0..dn_n]);
    var w_dn = try upload(d, std.mem.sliceAsBytes(host_w[0..dn_n]));
    defer w_dn.free();
    var sc: [2 * 512]f32 = undefined;
    for (&sc) |*v| v.* = std.math.ldexp(1.0 + r.float(f32), r.intRangeAtMost(i32, -10, -4));
    var s_gu = try upload(d, std.mem.sliceAsBytes(&sc));
    defer s_gu.free();
    var s_dn = try upload(d, std.mem.sliceAsBytes(sc[0..E]));
    defer s_dn.free();
    const ex: kern.Experts4 = .{ .up = w_gu.ptr, .down = w_dn.ptr, .up_scale = s_gu.ptr, .down_scale = s_dn.ptr, .width = ni, .dims = D };
    var max_r: usize = 0;
    for (rows) |m| max_r = @max(max_r, m);
    const pairs_max = max_r * slots;
    // x rows, picks, plan scratch, outputs
    const hx = try gpa.alloc(u16, max_r * D);
    defer gpa.free(hx);
    for (hx) |*v| v.* = bf16Bits(r.floatNorm(f32));
    var x = try upload(d, std.mem.sliceAsBytes(hx));
    defer x.free();
    const hact = try gpa.alloc(u16, pairs_max * ni);
    defer gpa.free(hact);
    for (hact) |*v| v.* = bf16Bits(r.floatNorm(f32) * 0.5);
    var act_in = try upload(d, std.mem.sliceAsBytes(hact));
    defer act_in.free();
    var picks = try cuda.DeviceBuffer.alloc(d, pairs_max * 4);
    defer picks.free();
    const nblk = (pairs_max + 1023) / 1024;
    const max_items = kern.maxItems(pairs_max, E + 1, kern.plan_tile);
    var members = try cuda.DeviceBuffer.alloc(d, pairs_max * 4);
    defer members.free();
    var items = try cuda.DeviceBuffer.alloc(d, max_items * 12);
    defer items.free();
    var counts = try cuda.DeviceBuffer.alloc(d, 256);
    defer counts.free();
    var rank = try cuda.DeviceBuffer.alloc(d, pairs_max * 4);
    defer rank.free();
    var hist = try cuda.DeviceBuffer.alloc(d, nblk * (E + 1) * 4 + 256);
    defer hist.free();
    const pl: kern.Plan = .{ .members = members.ptr, .items = items.ptr, .counts = counts.ptr, .rank = rank.ptr, .hist = hist.ptr };
    var bufs: [4]cuda.DeviceBuffer = undefined;
    const sizes = [4]usize{ pairs_max * ni * 2, pairs_max * ni * 2, pairs_max * D * 4, pairs_max * D * 4 };
    for (&bufs, sizes) |*b, n| b.* = try cuda.DeviceBuffer.alloc(d, n);
    defer for (&bufs) |*b| b.free();
    const h1 = try gpa.alloc(u8, pairs_max * D * 4);
    defer gpa.free(h1);
    const h2 = try gpa.alloc(u8, pairs_max * D * 4);
    defer gpa.free(h2);
    const hp = try gpa.alloc(i32, pairs_max);
    defer gpa.free(hp);
    const skip: c_int = @intCast(E);
    var all = true;
    for (rows) |R| {
        // routing: ten distinct experts a row, a quarter of the picks from 8 hot experts (hundreds of pairs each)
        for (0..R) |row| {
            var k: usize = 0;
            while (k < top) {
                // hot experts (hundreds of pairs) when `skew`, else a mild preference for 32 warm ones
                const hot = if (skew) r.uintLessThan(u32, 4) == 0 else r.uintLessThan(u32, 10) == 0;
                const e: i32 = if (hot) @intCast(r.uintLessThan(u32, if (skew) 8 else 32)) else @intCast(r.uintLessThan(u32, E));
                const dup = for (hp[row * slots .. row * slots + k]) |q| {
                    if (q == e) break true;
                } else false;
                if (dup) continue;
                hp[row * slots + k] = e;
                k += 1;
            }
            hp[row * slots + top] = @intCast(E);
        }
        try picks.upload(0, std.mem.sliceAsBytes(hp[0 .. R * slots]));
        const pairs = R * slots;
        const faces = [_]struct { name: []const u8, f32_out: bool }{ .{ .name = "gate/up", .f32_out = false }, .{ .name = "down fp32", .f32_out = true }, .{ .name = "down bf16", .f32_out = false } };
        for (faces, 0..) |face, fi| {
            const n = if (fi == 0) ni else D;
            const es: usize = if (face.f32_out) 4 else 2;
            const bytes = pairs * n * es;
            const outs = [2]u64{ bufs[if (fi == 0) 0 else 2].ptr, bufs[if (fi == 0) 1 else 3].ptr };
            for (outs, 0..) |o, i| {
                const b: cuda.DeviceBuffer = .{ .d = d, .ptr = o, .len = bytes };
                try b.fill8(if (i == 0) 0x11 else 0x11, ops.s.handle);
            }
            var ms: [2]f32 = .{ 0, 0 };
            for (0..2) |which| {
                var e0 = try cuda.Event.init(d, true);
                defer e0.deinit();
                var e1 = try cuda.Event.init(d, true);
                defer e1.deinit();
                const reps: usize = 3;
                for (0..reps + 1) |rep| {
                    if (rep == 1) try e0.record(ops.s);
                    if (which == 0) {
                        try ops.plan(picks.ptr, pairs, E + 1, kern.plan_tile, pl);
                        if (fi == 0) try ops.nvfp4GateUp(x.ptr, D, ex, pl, slots, E + 1, outs[0], R, skip) else try ops.nvfp4Down(act_in.ptr, ni, ex, pl, slots, E + 1, outs[0], face.f32_out, R, skip);
                    } else {
                        try p.plan(ops, picks.ptr, pairs, E + 1, pl);
                        if (fi == 0) try p.gateUp(ops.s, x.ptr, D, ex, pl, slots, E + 1, outs[1], R, skip) else try p.down(ops.s, act_in.ptr, ni, ex, pl, slots, E + 1, outs[1], face.f32_out, R, skip);
                    }
                }
                try e1.record(ops.s);
                try e1.synchronize();
                ms[which] = try cuda.Event.elapsedMs(e0, e1) / @as(f32, @floatFromInt(reps));
            }
            const b1: cuda.DeviceBuffer = .{ .d = d, .ptr = outs[0], .len = bytes };
            const b2: cuda.DeviceBuffer = .{ .d = d, .ptr = outs[1], .len = bytes };
            try b1.download(0, h1[0..bytes]);
            try b2.download(0, h2[0..bytes]);
            var differ: usize = 0;
            var i: usize = 0;
            while (i < bytes) : (i += es) differ += @intFromBool(!std.mem.eql(u8, h1[i .. i + es], h2[i .. i + es]));
            if (differ != 0) all = false;
            std.debug.print("{s} experts {s} NI {d} rows {d}{s} ({d} pairs): {d} of {d} values differ; 16-pair items {d:.3} ms, {d}-pair items (T {d}/{d}) {d:.3} ms ({d:.2}x)\n", .{ if (differ == 0) "EQUAL" else "DIFFER", face.name, ni, R, if (skew) " skewed" else "", pairs, differ, bytes / es, ms[0], p.tile, p.t_gu, p.t_down, ms[1], ms[0] / @max(ms[1], 1e-6) });
        }
    }
    return all;
}
