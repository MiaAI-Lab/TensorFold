//! glm5_next/cuda/{multi_tune,dflash2,dflash2_multi}.py torch operations.
//! Shared copies/conversions/radix selection use Flash Next's qualified operators.
const std = @import("std");
const cuda = @import("cuda");
pub const shared = @import("flashnext").torch_ops;
/// Blocks a greedy row is scanned by (GREEDY_SPLIT in glm_ops.cu); `part` holds greedy_split * 2 words a row.
pub const greedy_split = 16;
pub const Ops = struct {
    module: cuda.Module,
    base: shared.Ops,
    sort: cuda.Function,
    pack: cuda.Function,
    index_copy: cuda.Function,
    rotary: cuda.Function,
    ring: cuda.Function,
    rank_add: cuda.Function,
    int_add: cuda.Function,
    scaled_max: cuda.Function,
    mass_pack: cuda.Function,
    greedy_part: cuda.Function,
    greedy_pick: cuda.Function,
    l2pf: cuda.Function,
    media_rows: cuda.Function,
    pub fn load(ctx: *const cuda.Context) !Ops {
        var base = try shared.Ops.load(ctx);
        errdefer base.deinit();
        var m = try cuda.Module.load(ctx.d, cuda.kernels.glm_ops);
        errdefer m.unload();
        return .{ .module = m, .base = base, .sort = try m.function("glm_sort"), .pack = try m.function("glm_pack"), .index_copy = try m.function("glm_index_copy"), .rotary = try m.function("glm_rotary"), .ring = try m.function("glm_ring"), .rank_add = try m.function("glm_rank_add"), .int_add = try m.function("glm_int_add"), .scaled_max = try m.function("glm_scaled_max"), .mass_pack = try m.function("glm_mass_pack"), .greedy_part = try m.function("glm_greedy_part"), .greedy_pick = try m.function("glm_greedy_pick"), .l2pf = try m.function("glm_l2pf"), .media_rows = try m.function("glm_media_rows") };
    }
    pub fn deinit(o: *Ops) void {
        o.module.unload();
        o.base.deinit();
    }
    pub fn on(o: *const Ops, s: cuda.Stream) Torch {
        return .{ .ops = o, .s = s, .base = o.base.on(s) };
    }
};
pub const Torch = struct {
    ops: *const Ops,
    s: cuda.Stream,
    base: shared.Torch,
    fn go(t: Torch, f: cuda.Function, rows: usize, args: *cuda.Args) !void {
        if (rows == 0) return;
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, t.s, args);
    }
    /// Selection membership follows the existing radix compaction. Sorted equal-value
    /// permutations are not yet qualified against ATen (host draws lexsort by id).
    pub fn topkSorted(t: Torch, input: u64, columns: usize, rows: usize, k: usize, vals: u64, ids: u64, scratch: u64) !void {
        if (k == 0 or k > columns or k > 16384 or rows > 65535) return error.TopkShape;
        try t.base.topk(input, columns, rows, k, vals, ids, scratch);
        var a: cuda.Args = .{};
        a.add(vals);
        a.add(ids);
        a.add(@as(u32, @intCast(k)));
        try t.go(t.ops.sort, rows, &a);
    }
    /// Greedy top-1 of `rows` rows of bf16 logits (`columns` wide, contiguous) packed [value | int32 id] per row into `out`
    /// (2 words a row): what to_f32 + topkSorted(k=1) + pack write. Two launches: 16 blocks a row leave their best in `part`
    /// (greedy_split * 2 words a row), then one warp a row picks among them.
    pub fn greedyPick(t: Torch, logits: u64, part: u64, out: u64, rows: usize, columns: usize, offset: i64) !void {
        if (columns == 0 or columns > std.math.maxInt(u32) or rows > 65535) return error.TopkShape;
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(part);
        a.add(@as(u64, columns));
        try cuda.launch.launch(t.ops.greedy_part, .{ .grid = .{ .x = greedy_split, .y = @intCast(rows) }, .block = .{ .x = 256 } }, t.s, &a);
        var b: cuda.Args = .{};
        b.add(logits);
        b.add(part);
        b.add(out);
        b.add(@as(u64, columns));
        b.add(offset);
        try cuda.launch.launch(t.ops.greedy_pick, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, t.s, &b);
    }
    pub fn pack(t: Torch, vals: u64, ids: u64, out: u64, rows: usize, k: usize, offset: i64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ vals, ids, out }) |x| a.add(x);
        a.add(@as(u32, @intCast(k)));
        a.add(offset);
        try t.go(t.ops.pack, rows, &a);
    }
    pub fn indexCopy(t: Torch, src: u64, dst: u64, ids: u64, rows: usize, dest_rows: usize, inner_bytes: usize, outer: usize, invalid: u64) !void {
        if (rows == 0 or outer == 0 or inner_bytes == 0) return;
        var a: cuda.Args = .{};
        for ([_]u64{ src, dst, ids, rows, dest_rows, inner_bytes, outer, invalid }) |x| a.add(x);
        try cuda.launch.launch(t.ops.index_copy, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(@min(outer, 65535)) }, .block = .{ .x = 256 } }, t.s, &a);
    }
    pub fn rotary(t: Torch, pos: u64, inv: u64, cos: u64, sin: u64, rows: usize, half: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ pos, inv, cos, sin, rows, half }) |x| a.add(x);
        if (rows * half != 0) try t.go(t.ops.rotary, blocks(rows * half), &a);
    }
    pub fn ring(t: Torch, pos: u64, slots: u64, out: u64, rows: usize, cap: i64) !void {
        if (cap <= 0) return error.BadRing;
        var a: cuda.Args = .{};
        for ([_]u64{ pos, slots, out, rows }) |x| a.add(x);
        a.add(cap);
        if (rows != 0) try t.go(t.ops.ring, blocks(rows), &a);
    }
    pub fn intAdd(t: Torch, src: u64, dst: u64, count: usize, value: i64, bytes: usize) !void {
        if (bytes != 4 and bytes != 8) return error.BadIntegerType;
        var a: cuda.Args = .{};
        for ([_]u64{ src, dst, count }) |x| a.add(x);
        a.add(value);
        a.add(@as(u32, @intCast(bytes)));
        if (count != 0) try t.go(t.ops.int_add, blocks(count), &a);
    }
    pub fn rankAdd(t: Torch, ranks: u64, out: u64, count: usize, world: usize) !void {
        if (world < 1 or world > 4) return error.UnsupportedTp;
        var a: cuda.Args = .{};
        for ([_]u64{ ranks, out, count }) |x| a.add(x);
        a.add(@as(u32, @intCast(world)));
        if (count != 0) try t.go(t.ops.rank_add, blocks(count), &a);
    }
    pub fn scaledMax(t: Torch, logits: u64, out: u64, rows: usize, columns: usize, temp: f64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ logits, out, columns }) |x| a.add(x);
        a.add(temp);
        try t.go(t.ops.scaled_max, rows, &a);
    }
    /// Feature rows `rows[i]` (absolute prompt rows, u32 on the device) over the chunk starting at `pos`: x is the
    /// chunk's [rows, streams * width] bf16 hidden state, `feats` bf16 [count, width].
    pub fn mediaRows(t: Torch, rows: u64, feats: u64, x: u64, count: usize, pos: usize, width: usize, streams: usize) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        for ([_]u64{ rows, feats, x }) |v| a.add(v);
        a.add(@as(u32, @intCast(pos)));
        a.add(@as(u32, @intCast(width)));
        a.add(@as(u32, @intCast(streams)));
        try t.go(t.ops.media_rows, count, &a);
    }
    pub fn massPack(t: Torch, vals: u64, ids: u64, mass: u64, sums: u64, out: u64, rows: usize, columns: usize, selected: usize, count: usize, offset: i64, temp: f64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ vals, ids, mass, sums, out, columns }) |x| a.add(x);
        a.add(@as(u32, @intCast(selected)));
        a.add(@as(u32, @intCast(count)));
        a.add(offset);
        a.add(temp);
        try t.go(t.ops.mass_pack, rows, &a);
    }
};
fn blocks(n: usize) usize {
    return @min((n + 255) / 256, 65535);
}
test "analyze GLM launch ABI without a device" {
    _ = Ops.load;
    _ = Torch.topkSorted;
    _ = Torch.pack;
    _ = Torch.greedyPick;
    _ = Torch.indexCopy;
    _ = Torch.rotary;
    _ = Torch.ring;
    _ = Torch.rankAdd;
    _ = Torch.intAdd;
    _ = Torch.scaledMax;
    _ = Torch.massPack;
    _ = Torch.mediaRows;
}
