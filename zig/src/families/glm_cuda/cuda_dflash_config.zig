//! TensorFold glm5_next/cuda/dflash2.py:Drafter.__init__ and tp.py:draft_parts.
//! Ash Hart and TensorFold contributors. IncoAI DFlash2 checkpoint: CC BY-NC-ND.
const std = @import("std");
const layouts = @import("cuda_layouts.zig");
pub const Config = struct {
    mask: u32 = 154856,
    eps: f32 = 1e-5,
    theta: f32 = 10000,
    pub const hidden = 4096;
    pub const block = 8;
    pub const layers = 5;
    pub const ring = 2176;
    pub const taps = [_]usize{ 5, 14, 24, 33, 42 };
    /// Validate the served checkpoint geometry rather than silently accepting incompatible tensors.
    pub fn parse(gpa: std.mem.Allocator, json: []const u8) !Config {
        const p = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
        defer p.deinit();
        const root = p.value;
        const dc = try field(root, "dflash_config");
        for ([_]struct { []const u8, i64 }{ .{ "hidden_size", 4096 }, .{ "num_hidden_layers", 5 }, .{ "num_attention_heads", 32 }, .{ "num_key_value_heads", 8 }, .{ "head_dim", 128 }, .{ "intermediate_size", 12288 }, .{ "sliding_window", 2048 }, .{ "vocab_size", 154880 } }) |pair| if (try integer(root, pair[0]) != pair[1]) return error.UnsupportedDraftConfig;
        for ([_]struct { []const u8, i64 }{ .{ "block_size", 8 }, .{ "conv_group_size", 16 }, .{ "conv_kernel_size", 2 }, .{ "selector_rank", 256 }, .{ "selector_top_k", 16 } }) |pair| if (try integer(dc, pair[0]) != pair[1]) return error.UnsupportedDraftConfig;
        const causal = try field(root, "is_causal");
        if (causal != .bool or causal.bool) return error.UnsupportedDraftConfig;
        const tap = try field(dc, "target_layer_ids");
        if (tap != .array or tap.array.items.len != 5) return error.UnsupportedDraftConfig;
        for (tap.array.items, taps) |x, want| if (x != .integer or x.integer != want) return error.UnsupportedDraftConfig;
        const mask = try integer(dc, "mask_token_id");
        const eps = try number(root, "rms_norm_eps");
        const theta = try number(try field(root, "rope_parameters"), "rope_theta");
        if (mask < 0 or mask >= 154880 or eps <= 0 or theta <= 0 or !std.math.isFinite(eps) or !std.math.isFinite(theta)) return error.UnsupportedDraftConfig;
        return .{ .mask = @intCast(mask), .eps = eps, .theta = theta };
    }
    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }
};
fn field(v: std.json.Value, name: []const u8) !std.json.Value {
    if (v != .object) return error.BadDraftConfig;
    return v.object.get(name) orelse error.BadDraftConfig;
}
fn integer(v: std.json.Value, name: []const u8) !i64 {
    const x = try field(v, name);
    if (x != .integer) return error.BadDraftConfig;
    return x.integer;
}
fn number(v: std.json.Value, name: []const u8) !f32 {
    const x = try field(v, name);
    return switch (x) {
        .float => @floatCast(x.float),
        .integer => @floatFromInt(x.integer),
        else => error.BadDraftConfig,
    };
}
pub const Rank = struct {
    world: usize,
    rank: usize,
    heads: usize,
    kv: usize,
    inter: usize,
    vocab: usize,
    vocab_start: usize,
    pub fn init(world: usize, rank: usize) !Rank {
        if (world < 1 or world > 4 or rank >= world) return error.UnsupportedTp;
        var qs: [4]u32 = undefined;
        var kvs: [4]u32 = undefined;
        var ms: [4]u32 = undefined;
        var vs: [4]u32 = undefined;
        try layouts.draftParts(32, 8, 12288, @intCast(world), &qs, &kvs, &ms);
        _ = try layouts.splitSizes(154880, @intCast(world), 64, &vs);
        const v = layouts.shareOf(vs[0..world], @intCast(rank));
        return .{ .world = world, .rank = rank, .heads = qs[rank], .kv = kvs[rank], .inter = ms[rank], .vocab = v.stop - v.start, .vocab_start = v.start };
    }
};
test "KV groups stay intact over unequal ranks" {
    const r = try Rank.init(3, 2);
    try std.testing.expectEqual(@as(usize, 8), r.heads);
    try std.testing.expectEqual(@as(usize, 2), r.kv);
    try std.testing.expectEqual(@as(usize, 4096), r.inter);
    try std.testing.expectEqual(@as(usize, 51584), r.vocab);
}
