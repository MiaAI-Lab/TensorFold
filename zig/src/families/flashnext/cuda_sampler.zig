//! Flash Next's host draws (Python qwen4_exp/cuda/decode.py): the keyed rule of core/lanes/sampling.zig over the
//! candidates the GPU hands back, with the draft head's temperature-1 probability for the chain's stop rule.
//! One rank: each row's top_k + MARGIN values (any order), their columns (mapped to token ids for the draft head)
//! and the row's log-sum-exp. Two ranks: every rank's top CAND values, global ids and log-sum-exp, gathered
//! [world][rows][2 * CAND + 1] f32 words (`candidates` in forward.py), drawn the same on both ranks.
const std = @import("std");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;
const S = lanes.sampling;

extern "c" fn exp(x: f64) f64;
extern "c" fn log(x: f64) f64;

/// Candidates beyond top_k read back, so values tied at the cut resolve by id on the host.
pub const margin = 8;
/// Two ranks: candidates a rank gathers per row (top_k 20 plus the margin, rounded up).
pub const cand = 32;

pub const Draw = struct { token: u64, prob: f64 = 0.0 };

/// Whether the gathered candidates cover a request's sampler (Python `_gathered_fits`).
pub fn gatheredFits(s: ?S.Sampling) bool {
    const smp = s orelse return true;
    return smp.temperature <= 0 or (smp.top_k != 0 and smp.top_k + margin <= cand);
}

/// The lowest-id maximum of `values` (numpy lexsort((ids, -values))[0]: value descending, then id ascending).
fn firstBest(values: []const f32, ids: []const u64) usize {
    var best: usize = 0;
    for (values, ids, 0..) |v, id, i| {
        if (i == 0) continue;
        if (v > values[best] or (v == values[best] and id < ids[best])) best = i;
    }
    return best;
}

/// exp(value of `token` among the candidates - total), 0 when the token is not among them (Python `probs`).
fn probOf(values: []const f32, ids: []const u64, token: u64, total: f64) f64 {
    for (values, ids) |v, id| {
        if (id == token) return exp(@as(f64, v) - total);
    }
    return 0.0;
}

/// One row of one rank's candidates: `values` f32 (top_k + margin, any order) with token `ids`, at `position`.
/// Greedy (null or temperature <= 0) takes the lowest-id maximum; `lse` (the row's f32 log-sum-exp) gives the
/// temperature-1 probability of the drawn token (Python `sample_draft` and `sample_mapped`).
pub fn drawRow(gpa: Allocator, values: []const f32, ids: []const u64, position: u64, s: ?S.Sampling, lse: ?f32) !Draw {
    const token: u64 = if (s == null or s.?.temperature <= 0)
        ids[firstBest(values, ids)]
    else blk: {
        const wide = try gpa.alloc(f64, values.len);
        defer gpa.free(wide);
        for (wide, values) |*w, v| w.* = v;
        break :blk try S.choose(gpa, wide, ids, position, s.?);
    };
    return .{ .token = token, .prob = if (lse) |l| probOf(values, ids, token, @as(f64, l)) else 0.0 };
}

/// Every rank's gathered candidates `got` [world][rows][2 * cand + 1] (values, ids as int32 bits, log-sum-exp) ->
/// each row's draw at `positions[r]`, the same on every rank (Python `choose_gathered`; `with_prob` adds each
/// token's temperature-1 probability from the ranks' log-sum-exps).
pub fn chooseGathered(gpa: Allocator, got: []const f32, world: usize, rows: usize, positions: []const u64, s: ?S.Sampling, with_prob: bool, out: []Draw) !void {
    const width = 2 * cand + 1;
    if (got.len < world * rows * width or positions.len != rows or out.len != rows) return error.GatheredShape;
    const n = world * cand;
    const values = try gpa.alloc(f32, n);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, n);
    defer gpa.free(ids);
    for (0..rows) |r| {
        // the ranks' candidates side by side, rank 0 first (np.concatenate(axis=1))
        var top = -std.math.inf(f64);
        for (0..world) |k| {
            const row = got[(k * rows + r) * width ..][0..width];
            for (0..cand) |j| {
                values[k * cand + j] = row[j];
                const bits: u32 = @bitCast(row[cand + j]);
                ids[k * cand + j] = @intCast(@as(i64, @as(i32, @bitCast(bits))));
            }
            top = @max(top, @as(f64, row[2 * cand]));
        }
        const draw = try drawRow(gpa, values, ids, positions[r], s, null);
        out[r] = draw;
        if (with_prob) {
            // total = top + log(sum_k exp(lse_k - top)), the ranks added in order (numpy sum over axis 0)
            var sum: f64 = 0.0;
            for (0..world) |k| sum += exp(@as(f64, got[(k * rows + r) * width + 2 * cand]) - top);
            out[r].prob = probOf(values, ids, draw.token, top + log(sum));
        }
    }
}

test "gathered draws and probabilities equal Python's choose_gathered" {
    const gpa = std.testing.allocator;
    const text = @embedFile("fixtures_cuda_gathered.json");
    const Fixture = struct {
        world: usize,
        R: usize,
        cand: usize,
        vals: []const u32,
        ids: []const i64,
        lse: []const u32,
        positions: []const u64,
        seed: u64,
        temperature: f64,
        top_k: u32,
        top_p: f64,
        sampled: []const u64,
        greedy: []const u64,
        p_sampled: []const u64,
        p_greedy: []const u64,
    };
    const parsed = try std.json.parseFromSlice(Fixture, gpa, text, .{});
    defer parsed.deinit();
    const f = parsed.value;
    try std.testing.expectEqual(@as(usize, cand), f.cand);
    // pack the fixture as forward.py's `candidates` writes it
    const width = 2 * cand + 1;
    const got = try gpa.alloc(f32, f.world * f.R * width);
    defer gpa.free(got);
    for (0..f.world) |k| for (0..f.R) |r| {
        const row = got[(k * f.R + r) * width ..][0..width];
        for (0..cand) |j| {
            row[j] = @bitCast(f.vals[(k * f.R + r) * cand + j]);
            row[cand + j] = @bitCast(@as(u32, @bitCast(@as(i32, @intCast(f.ids[(k * f.R + r) * cand + j])))));
        }
        row[2 * cand] = @bitCast(f.lse[k * f.R + r]);
    };
    var out: [3]Draw = undefined;
    const s: S.Sampling = .{ .seed = f.seed, .temperature = f.temperature, .top_k = f.top_k, .top_p = f.top_p };
    try chooseGathered(gpa, got, f.world, f.R, f.positions, s, true, out[0..f.R]);
    for (out[0..f.R], f.sampled, f.p_sampled) |d, t, p| {
        try std.testing.expectEqual(t, d.token);
        try std.testing.expectEqual(p, @as(u64, @bitCast(d.prob)));
    }
    try chooseGathered(gpa, got, f.world, f.R, f.positions, null, true, out[0..f.R]);
    for (out[0..f.R], f.greedy, f.p_greedy) |d, t, p| {
        try std.testing.expectEqual(t, d.token);
        try std.testing.expectEqual(p, @as(u64, @bitCast(d.prob)));
    }
}

test "the gathered candidates cover top_k 20 and greedy, not top_k 30" {
    try std.testing.expect(gatheredFits(null));
    try std.testing.expect(gatheredFits(.{ .seed = 1 }));
    try std.testing.expect(!gatheredFits(.{ .seed = 1, .top_k = 30 }));
    try std.testing.expect(!gatheredFits(.{ .seed = 1, .top_k = 0 }));
    try std.testing.expect(gatheredFits(.{ .seed = 1, .temperature = 0 }));
}
