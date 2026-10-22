//! GLM-5.3-Flash's tensor-parallel layout and EXL3 split geometry on the host, byte for byte as the Python engine
//! decides them:
//!   parts / split_sizes   cuda/shares.py parts, families/glm5_next/cuda/tp.py split_sizes
//!   Layout                tp.Layout: heads, KDA heads, dense/moe/shared intermediates, vocabulary
//!   rule / familyOf       split.py rule (ROW/COL/REP and EXL3 trellis companions) and tp.Layout.family
//!   cut                   Layout.cut: a rank's [start, stop) of an axis, scaled to packed words / tiles
//!   draft_parts           tp.draft_parts: DFlash2 KV heads in whole groups, MLP in units of 128
//!   wordsShape            exl3_mm.words: a 4-bit trellis int16 [..., 64] as int32 [..., 32], the same bytes
//!
//! Patch 0066's split: whole units (a head, a 128-column Hadamard block, 64 vocabulary rows, a DFlash2 KV group),
//! as even as they go, the remainder to the lowest ranks; two ranks of an even count get exact halves.
//!
//! Python source (TensorFold, https://github.com/ashhart/TensorFold): shares.py, glm5_next/cuda/tp.py, split.py,
//! exl3_mm.words, dflash2.Drafter's rank slices.
const std = @import("std");
const cfgs = @import("cuda_config.zig");

pub const Family = enum { heads, lin, dense, moe, shared, vocab };

pub const units = struct {
    pub const heads: u32 = 1;
    pub const lin: u32 = 1;
    pub const dense: u32 = 128;
    pub const moe: u32 = 128;
    pub const shared: u32 = 128;
    pub const vocab: u32 = 64;
};

pub fn unitOf(f: Family) u32 {
    return switch (f) {
        .heads => units.heads,
        .lin => units.lin,
        .dense => units.dense,
        .moe => units.moe,
        .shared => units.shared,
        .vocab => units.vocab,
    };
}

pub const Kind = enum {
    row,
    col,
    dim1,
    rep,
    drop,

    pub fn axis(self: Kind) ?u8 {
        return switch (self) {
            .row => 0,
            .col, .dim1 => 1,
            .rep, .drop => null,
        };
    }

    pub fn text(self: Kind) []const u8 {
        return switch (self) {
            .row => "row",
            .col => "col",
            .dim1 => "dim1",
            .rep => "rep",
            .drop => "drop",
        };
    }
};

/// cuda/shares.py parts: each rank's share of `total` (a multiple of `unit`) in rank order.
pub fn partsInto(total: u32, world: u32, unit: u32, out: []u32) ![]u32 {
    if (world < 1 or unit < 1 or world > out.len) return error.BadSplit;
    if (total % unit != 0) return error.BadSplit;
    const n_units = total / unit;
    if (n_units < world) return error.BadSplit;
    const base = n_units / world;
    const extra = n_units % world;
    for (0..world) |r| out[r] = (base + @intFromBool(r < extra)) * unit;
    return out[0..world];
}

/// tp.split_sizes: halves at two ranks whenever the total is even, else whole units as even as they go.
pub fn splitSizes(total: u32, world: u32, unit: u32, out: []u32) ![]u32 {
    if (world == 2 and total % 2 == 0) {
        if (out.len < 2) return error.BadSplit;
        out[0] = total / 2;
        out[1] = total / 2;
        return out[0..2];
    }
    return partsInto(total, world, unit, out);
}

pub fn shareOf(sizes: []const u32, rank: u32) struct { start: u32, stop: u32 } {
    var start: u32 = 0;
    for (sizes[0..rank]) |s| start += s;
    return .{ .start = start, .stop = start + sizes[rank] };
}

/// Rank `rank` of `world`: the totals of every family and this rank's [start, stop) of each.
pub const Layout = struct {
    rank: u32,
    world: u32,
    heads_total: u32,
    lin_total: u32,
    dense_total: u32,
    moe_total: u32,
    shared_total: u32,
    vocab_total: u32,
    kinds: []const cfgs.LayerKind,

    pub fn fromConfig(cfg: *const cfgs.Config, rank: u32, world: u32) !Layout {
        if (world < 1 or rank >= world) return error.BadWorld;
        const lay = Layout{
            .rank = rank,
            .world = world,
            .heads_total = cfg.heads,
            .lin_total = cfg.lin_heads,
            .dense_total = cfg.dense_width,
            .moe_total = cfg.moe_width,
            .shared_total = cfg.shared_width,
            .vocab_total = cfg.vocab,
            .kinds = cfg.kinds[0..cfg.layers],
        };
        var buf: [8]u32 = undefined;
        inline for ([_]Family{ .heads, .lin, .dense, .moe, .shared, .vocab }) |f| {
            _ = try lay.sizes(f, &buf);
        }
        return lay;
    }

    pub fn total(self: Layout, f: Family) u32 {
        return switch (f) {
            .heads => self.heads_total,
            .lin => self.lin_total,
            .dense => self.dense_total,
            .moe => self.moe_total,
            .shared => self.shared_total,
            .vocab => self.vocab_total,
        };
    }

    pub fn sizes(self: Layout, f: Family, out: []u32) ![]u32 {
        return splitSizes(self.total(f), self.world, unitOf(f), out);
    }

    pub fn size(self: Layout, f: Family) !u32 {
        var buf: [8]u32 = undefined;
        const s = try self.sizes(f, &buf);
        return s[self.rank];
    }

    pub fn start(self: Layout, f: Family) !u32 {
        var buf: [8]u32 = undefined;
        const s = try self.sizes(f, &buf);
        return shareOf(s, self.rank).start;
    }

    /// This rank's [start, stop) of an axis of `length` elements holding `family`'s total.
    pub fn cut(self: Layout, f: Family, length: u32) !struct { a: u32, b: u32 } {
        const tot = self.total(f);
        const a = try self.start(f);
        const b = a + try self.size(f);
        if ((@as(u64, a) * length) % tot != 0 or (@as(u64, b) * length) % tot != 0) return error.BadSplit;
        return .{ .a = @intCast(@as(u64, a) * length / tot), .b = @intCast(@as(u64, b) * length / tot) };
    }

    pub fn familyOf(self: Layout, name: []const u8) ?Family {
        if (indexOf(name, ".self_attn.o_proj.")) |_| {
            const layer = layerIndex(name);
            if (layer) |i| {
                if (i < self.kinds.len and self.kinds[i] == .kda) return .lin;
            }
            return .heads;
        }
        if (containsRe(name, ".mlp.experts.", ". ")) return .moe; // experts.<digits>.
        if (indexOf(name, ".mlp.shared_experts.") != null) return .shared;
        if (indexOf(name, ".mlp.gate_proj.") != null or indexOf(name, ".mlp.up_proj.") != null or indexOf(name, ".mlp.down_proj.") != null) return .dense;
        if (indexOf(name, ".self_attn.q_b_proj.") != null or indexOf(name, ".self_attn.kv_b_proj.") != null) return .heads;
        if (kdaLinName(name)) return .lin;
        if (std.mem.startsWith(u8, name, "lm_head.")) return .vocab;
        return null;
    }
};

fn containsRe(name: []const u8, prefix: []const u8, _: []const u8) bool {
    const i = indexOf(name, prefix) orelse return false;
    const rest = name[i + prefix.len ..];
    if (rest.len == 0 or rest[0] < '0' or rest[0] > '9') return false;
    return true;
}

fn kdaLinName(name: []const u8) bool {
    const keys = [_][]const u8{
        ".self_attn.q_proj.", ".self_attn.k_proj.", ".self_attn.v_proj.",
        ".self_attn.q_conv1d.", ".self_attn.k_conv1d.", ".self_attn.v_conv1d.",
        ".self_attn.f_b_proj.", ".self_attn.g_b_proj.", ".self_attn.b_proj.",
    };
    for (keys) |k| if (indexOf(name, k) != null) return true;
    if (std.mem.endsWith(u8, name, ".self_attn.A_log") or std.mem.endsWith(u8, name, ".self_attn.dt_bias")) return true;
    return false;
}

pub fn layerIndex(name: []const u8) ?u32 {
    const key = "layers.";
    const i = indexOf(name, key) orelse return null;
    const rest = name[i + key.len ..];
    var n: u32 = 0;
    var saw = false;
    for (rest) |ch| {
        if (ch < '0' or ch > '9') break;
        saw = true;
        n = n * 10 + (ch - '0');
    }
    return if (saw) n else null;
}

fn indexOf(hay: []const u8, needle: []const u8) ?usize {
    return std.mem.indexOf(u8, hay, needle);
}

/// split.py rule: EXL3 expert companions first, then ROW / COL / REP, vision dropped.
pub fn rule(name: []const u8) !Kind {
    if (std.mem.startsWith(u8, name, "model.visual.")) return .drop;
    if (exl3Kind(name)) |k| return k;
    const row = rowName(name);
    const col = colName(name);
    const rep = repName(name);
    const n: u32 = @intFromBool(row) + @intFromBool(col) + @intFromBool(rep);
    if (n != 1) return error.AmbiguousSplit;
    if (row) return .row;
    if (col) return .col;
    return .rep;
}

fn exl3Kind(name: []const u8) ?Kind {
    // .mlp.experts.<id>.(gate|up|down)_proj.(trellis|suh|svh|mcg)
    const p = indexOf(name, ".mlp.experts.") orelse return null;
    const rest = name[p + ".mlp.experts.".len ..];
    var i: usize = 0;
    while (i < rest.len and rest[i] >= '0' and rest[i] <= '9') i += 1;
    if (i == 0 or i >= rest.len or rest[i] != '.') return null;
    const tail = rest[i + 1 ..];
    const proj: enum { gate, up, down } = if (std.mem.startsWith(u8, tail, "gate_proj."))
        .gate
    else if (std.mem.startsWith(u8, tail, "up_proj."))
        .up
    else if (std.mem.startsWith(u8, tail, "down_proj."))
        .down
    else
        return null;
    const part = if (std.mem.startsWith(u8, tail, "gate_proj."))
        tail["gate_proj.".len..]
    else if (std.mem.startsWith(u8, tail, "up_proj."))
        tail["up_proj.".len..]
    else
        tail["down_proj.".len..];
    const which: enum { trellis, suh, svh, mcg } = if (std.mem.eql(u8, part, "trellis"))
        .trellis
    else if (std.mem.eql(u8, part, "suh"))
        .suh
    else if (std.mem.eql(u8, part, "svh"))
        .svh
    else if (std.mem.eql(u8, part, "mcg"))
        .mcg
    else
        return null;
    if (which == .mcg) return .rep;
    const key_proj: enum { gate, down } = if (proj == .up) .gate else if (proj == .gate) .gate else .down;
    return switch (key_proj) {
        .gate => switch (which) {
            .trellis => .dim1,
            .suh => .rep,
            .svh => .row,
            .mcg => .rep,
        },
        .down => switch (which) {
            .trellis => .row,
            .suh => .row,
            .svh => .rep,
            .mcg => .rep,
        },
    };
}

fn rowName(name: []const u8) bool {
    if (indexOf(name, ".mlp.experts.") != null and (indexOf(name, ".gate_proj.") != null or indexOf(name, ".up_proj.") != null)) return true;
    if (indexOf(name, ".mlp.shared_experts.gate_proj.") != null or indexOf(name, ".mlp.shared_experts.up_proj.") != null) return true;
    if (indexOf(name, ".mlp.gate_proj.") != null or indexOf(name, ".mlp.up_proj.") != null) return true;
    if (indexOf(name, ".self_attn.q_proj.") != null or indexOf(name, ".self_attn.k_proj.") != null or indexOf(name, ".self_attn.v_proj.") != null) return true;
    if (indexOf(name, ".self_attn.q_conv1d.") != null or indexOf(name, ".self_attn.k_conv1d.") != null or indexOf(name, ".self_attn.v_conv1d.") != null) return true;
    if (indexOf(name, ".self_attn.f_b_proj.") != null or indexOf(name, ".self_attn.g_b_proj.") != null or indexOf(name, ".self_attn.b_proj.") != null) return true;
    if (std.mem.endsWith(u8, name, ".self_attn.A_log") or std.mem.endsWith(u8, name, ".self_attn.dt_bias")) return true;
    if (indexOf(name, ".self_attn.q_b_proj.") != null or indexOf(name, ".self_attn.kv_b_proj.") != null) return true;
    return false;
}

fn colName(name: []const u8) bool {
    if (indexOf(name, ".mlp.experts.") != null and indexOf(name, ".down_proj.") != null) return true;
    if (indexOf(name, ".mlp.shared_experts.down_proj.") != null) return true;
    if (indexOf(name, ".mlp.down_proj.") != null) return true;
    if (indexOf(name, ".self_attn.o_proj.") != null) return true;
    return false;
}

fn repName(name: []const u8) bool {
    if (std.mem.startsWith(u8, name, "lm_head.")) return true;
    if (indexOf(name, "embed_tokens.") != null) return true;
    if (std.mem.eql(u8, name, "model.language_model.norm.weight")) return true;
    if (indexOf(name, "_layernorm.weight") != null) return true;
    if (indexOf(name, ".hc_attn_fn") != null or indexOf(name, ".hc_attn_base") != null or indexOf(name, ".hc_attn_scale") != null) return true;
    if (indexOf(name, ".hc_ffn_fn") != null or indexOf(name, ".hc_ffn_base") != null or indexOf(name, ".hc_ffn_scale") != null) return true;
    if (indexOf(name, ".mlp.gate.weight") != null or indexOf(name, ".mlp.gate.e_score_correction_bias") != null) return true;
    if (indexOf(name, ".self_attn.indexer.") != null) return true;
    if (indexOf(name, ".self_attn.q_a_proj.") != null or indexOf(name, ".self_attn.kv_a_proj_with_mqa.") != null) return true;
    if (indexOf(name, ".self_attn.f_a_proj.") != null or indexOf(name, ".self_attn.g_a_proj.") != null) return true;
    if (indexOf(name, ".self_attn.o_norm.weight") != null) return true;
    if (indexOf(name, ".eh_proj.") != null) return true;
    if (indexOf(name, ".enorm.weight") != null or indexOf(name, ".hnorm.weight") != null) return true;
    if (indexOf(name, ".shared_head.norm.weight") != null) return true;
    if (indexOf(name, ".input_layernorm.weight") != null or indexOf(name, ".post_attention_layernorm.weight") != null) return true;
    return false;
}

/// DFlash2's split: KV heads in whole groups (each with heads/kv_heads query heads), the MLP in units of 128.
pub fn draftParts(heads: u32, kv_heads: u32, inter: u32, world: u32, q_out: []u32, kv_out: []u32, mlp_out: []u32) !void {
    if (kv_heads == 0 or heads % kv_heads != 0) return error.BadSplit;
    const group = heads / kv_heads;
    const kv = try splitSizes(kv_heads, world, 1, kv_out);
    const q = q_out[0..kv.len];
    for (kv, q) |k, *o| o.* = k * group;
    const mlp_unit: u32 = if (inter % 128 == 0) 128 else 1;
    _ = try splitSizes(inter, world, mlp_unit, mlp_out);
}

pub const hadamard = 128;
pub const tile = 16;
pub const exl3_bits: u32 = 4;

/// A 4-bit trellis int16 [..., 64] as the kernels read it: int32 [..., 32], the same bytes (exl3_mm.words).
pub fn trellisDeviceLast(stored_last: usize, dtype_size: usize) ?struct { last: usize, item: usize } {
    if (stored_last == 16 * exl3_bits and dtype_size == 2) return .{ .last = 8 * exl3_bits, .item = 4 };
    return null;
}

pub const align_bytes: u32 = 256;

pub fn alignUp(n: u64) u64 {
    return (n + (align_bytes - 1)) & ~@as(u64, align_bytes - 1);
}

test "parts: 64 heads over 3 ranks and 2048 columns in units of 128" {
    var buf: [8]u32 = undefined;
    const h = try partsInto(64, 3, 1, &buf);
    try std.testing.expectEqualSlices(u32, &.{ 22, 21, 21 }, h);
    const m = try partsInto(2048, 3, 128, &buf);
    try std.testing.expectEqualSlices(u32, &.{ 768, 640, 640 }, m);
    const v = try partsInto(154880, 3, 64, &buf);
    try std.testing.expectEqualSlices(u32, &.{ 51648, 51648, 51584 }, v);
}

test "split_sizes: two even ranks are halves, world 1 is the total" {
    var buf: [8]u32 = undefined;
    try std.testing.expectEqualSlices(u32, &.{ 1024, 1024 }, try splitSizes(2048, 2, 128, &buf));
    try std.testing.expectEqualSlices(u32, &.{2048}, try splitSizes(2048, 1, 128, &buf));
    try std.testing.expectEqualSlices(u32, &.{ 512, 512, 512, 512 }, try splitSizes(2048, 4, 128, &buf));
}

test "draft_parts: 32 / 8 over 3 ranks" {
    var q: [4]u32 = undefined;
    var kv: [4]u32 = undefined;
    var mlp: [4]u32 = undefined;
    try draftParts(32, 8, 12288, 3, &q, &kv, &mlp);
    try std.testing.expectEqualSlices(u32, &.{ 12, 12, 8 }, q[0..3]);
    try std.testing.expectEqualSlices(u32, &.{ 3, 3, 2 }, kv[0..3]);
    try std.testing.expectEqualSlices(u32, &.{ 4096, 4096, 4096 }, mlp[0..3]);
}

test "split.rule covers EXL3 companions, ROW/COL/REP and vision" {
    const L = "model.language_model.";
    try std.testing.expectEqual(Kind.dim1, try rule(L ++ "layers.3.mlp.experts.0.gate_proj.trellis"));
    try std.testing.expectEqual(Kind.rep, try rule(L ++ "layers.3.mlp.experts.0.gate_proj.suh"));
    try std.testing.expectEqual(Kind.row, try rule(L ++ "layers.3.mlp.experts.0.gate_proj.svh"));
    try std.testing.expectEqual(Kind.rep, try rule(L ++ "layers.3.mlp.experts.0.gate_proj.mcg"));
    try std.testing.expectEqual(Kind.dim1, try rule(L ++ "layers.3.mlp.experts.7.up_proj.trellis"));
    try std.testing.expectEqual(Kind.row, try rule(L ++ "layers.3.mlp.experts.0.down_proj.trellis"));
    try std.testing.expectEqual(Kind.row, try rule(L ++ "layers.3.mlp.experts.0.down_proj.suh"));
    try std.testing.expectEqual(Kind.rep, try rule(L ++ "layers.3.mlp.experts.0.down_proj.svh"));
    try std.testing.expectEqual(Kind.row, try rule(L ++ "layers.0.self_attn.q_proj.weight"));
    try std.testing.expectEqual(Kind.col, try rule(L ++ "layers.0.self_attn.o_proj.weight"));
    try std.testing.expectEqual(Kind.row, try rule(L ++ "layers.0.self_attn.A_log"));
    try std.testing.expectEqual(Kind.rep, try rule("lm_head.weight"));
    try std.testing.expectEqual(Kind.rep, try rule(L ++ "embed_tokens.weight"));
    try std.testing.expectEqual(Kind.rep, try rule(L ++ "norm.weight"));
    try std.testing.expectEqual(Kind.drop, try rule("model.visual.blocks.0.attn.qkv.weight"));
    try std.testing.expectError(error.AmbiguousSplit, rule(L ++ "layers.3.mystery.weight"));
}

test "Layout.familyOf: o_proj follows the layer kind, MTP is heads" {
    const kinds = [_]cfgs.LayerKind{ .kda, .kda, .kda, .dsa };
    const lay = Layout{
        .rank = 0,
        .world = 2,
        .heads_total = 64,
        .lin_total = 64,
        .dense_total = 12288,
        .moe_total = 2048,
        .shared_total = 2048,
        .vocab_total = 154880,
        .kinds = &kinds,
    };
    try std.testing.expectEqual(Family.lin, lay.familyOf("model.language_model.layers.0.self_attn.o_proj.weight").?);
    try std.testing.expectEqual(Family.heads, lay.familyOf("model.language_model.layers.3.self_attn.o_proj.weight").?);
    try std.testing.expectEqual(Family.heads, lay.familyOf("model.language_model.layers.45.self_attn.o_proj.weight").?);
    try std.testing.expectEqual(Family.moe, lay.familyOf("model.language_model.layers.3.mlp.experts.12.gate_proj.trellis").?);
    try std.testing.expectEqual(Family.shared, lay.familyOf("model.language_model.layers.3.mlp.shared_experts.down_proj.weight").?);
    try std.testing.expectEqual(Family.dense, lay.familyOf("model.language_model.layers.0.mlp.gate_proj.weight").?);
    try std.testing.expectEqual(Family.vocab, lay.familyOf("lm_head.weight").?);
}

test "Layout.cut scales a trellis's 128 tile columns for 2048 expert columns over 3 ranks" {
    const kinds = [_]cfgs.LayerKind{.dsa};
    const lay0 = Layout{
        .rank = 0,
        .world = 3,
        .heads_total = 64,
        .lin_total = 64,
        .dense_total = 12288,
        .moe_total = 2048,
        .shared_total = 2048,
        .vocab_total = 154880,
        .kinds = &kinds,
    };
    const c0 = try lay0.cut(.moe, 128);
    try std.testing.expectEqual(@as(u32, 0), c0.a);
    try std.testing.expectEqual(@as(u32, 48), c0.b);
    const lay1 = Layout{ .rank = 1, .world = 3, .heads_total = 64, .lin_total = 64, .dense_total = 12288, .moe_total = 2048, .shared_total = 2048, .vocab_total = 154880, .kinds = &kinds };
    const c1 = try lay1.cut(.moe, 128);
    try std.testing.expectEqual(@as(u32, 48), c1.a);
    try std.testing.expectEqual(@as(u32, 88), c1.b);
}
