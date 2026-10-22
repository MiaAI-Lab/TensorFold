//! GLM-5.3-Flash geometry: glm5_next/cuda/tp.py and cuda/shares.py; SPEC §1/§4.
//! TensorFold by Ash Hart and contributors; CUDA family registry by jschmied (PR #443).
const std = @import("std");
pub const Share = struct { start: usize, size: usize };
pub fn share(total: usize, unit: usize, world: usize, rank: usize) !Share {
    if (world < 1 or world > 4) return error.UnsupportedTp;
    if (rank >= world) return error.BadRank;
    if (unit == 0 or total % unit != 0 or total / unit < world) return error.BadSplit;
    const n = total / unit;
    return .{ .start = (rank * (n / world) + @min(rank, n % world)) * unit, .size = (n / world + @intFromBool(rank < n % world)) * unit };
}
/// Small fixed-model interface independent of the weights package's Config.
pub const Geometry = struct {
    world: usize = 1,
    rank: usize = 0,
    heads: usize = 64,
    lin_heads: usize = 64,
    dense_width: usize = 12288,
    moe_width: usize = 2048,
    shared_width: usize = 2048,
    vocab: usize = 154880,
    vocab_start: usize = 0,
    pub const hidden = 4096;
    pub const streams = 4;
    pub const layers = 45;
    pub const kda_layers = 34;
    pub const dsa_layers = 11;
    pub const head_dim = 256;
    pub const lin_dim = 128;
    pub const q_lora = 1536;
    pub const kv_lora = 512;
    pub const index_heads = 32; // replicated at every TP size
    pub const index_dim = 128;
    pub const experts = 288;
    pub const top_k = 8;
    pub const slots = 9;
    pub const eps: f32 = 1e-5;
    pub const latent_row = 528;
    pub const pool_row = 144;
    pub const pool = 4;
    pub const tokens = 2051;
    pub const extent = 2048;
    pub fn init(world: usize, rank: usize) !Geometry {
        const h = try share(64, 1, world, rank);
        const v = try share(154880, 64, world, rank);
        return .{ .world = world, .rank = rank, .heads = h.size, .lin_heads = h.size, .dense_width = (try share(12288, 128, world, rank)).size, .moe_width = (try share(2048, 128, world, rank)).size, .shared_width = (try share(2048, 128, world, rank)).size, .vocab = v.size, .vocab_start = v.start };
    }
    pub fn wide(_: Geometry) usize {
        return streams * hidden;
    }
    pub fn convDim(g: Geometry) usize {
        return 3 * g.lin_heads * lin_dim;
    }
    pub fn projWidth(g: Geometry) usize {
        return g.convDim() + 2 * lin_dim + g.lin_heads;
    }
    pub fn attnWidth(g: Geometry) usize {
        return g.heads * head_dim;
    }
    pub fn isDsa(layer: usize) bool {
        return layer % 4 == 3;
    }
    pub fn layerIndex(layer: usize) usize {
        return if (isDsa(layer)) layer / 4 else layer - layer / 4;
    }
    pub fn indexRing(capacity: usize, rows: usize) usize {
        return @min(capacity, std.mem.alignForward(usize, rows + 3, 64));
    }
};
test "SPEC per-rank geometry TP 1 through 4 conserves every split" {
    const heads = [_][4]usize{ .{ 64, 0, 0, 0 }, .{ 32, 32, 0, 0 }, .{ 22, 21, 21, 0 }, .{ 16, 16, 16, 16 } };
    const moe = [_][4]usize{ .{ 2048, 0, 0, 0 }, .{ 1024, 1024, 0, 0 }, .{ 768, 640, 640, 0 }, .{ 512, 512, 512, 512 } };
    const vocab = [_][4]usize{ .{ 154880, 0, 0, 0 }, .{ 77440, 77440, 0, 0 }, .{ 51648, 51648, 51584, 0 }, .{ 38720, 38720, 38720, 38720 } };
    for (1..5) |w| {
        var end: usize = 0;
        var dense: usize = 0;
        for (0..w) |r| {
            const g = try Geometry.init(w, r);
            try std.testing.expectEqual(heads[w - 1][r], g.heads);
            try std.testing.expectEqual(moe[w - 1][r], g.moe_width);
            try std.testing.expectEqual(vocab[w - 1][r], g.vocab);
            try std.testing.expectEqual(end, g.vocab_start);
            try std.testing.expectEqual(g.moe_width, g.shared_width);
            end += g.vocab;
            dense += g.dense_width;
        }
        try std.testing.expectEqual(@as(usize, 154880), end);
        try std.testing.expectEqual(@as(usize, 12288), dense);
    }
    const g = try Geometry.init(2, 0);
    try std.testing.expectEqual(@as(usize, 12576), g.projWidth());
    try std.testing.expectEqual(@as(usize, 12288), g.convDim());
    try std.testing.expectEqual(@as(usize, 64), Geometry.indexRing(2048, 32));
    try std.testing.expectError(error.UnsupportedTp, Geometry.init(5, 0));
    try std.testing.expectError(error.BadRank, Geometry.init(2, 2));
}
