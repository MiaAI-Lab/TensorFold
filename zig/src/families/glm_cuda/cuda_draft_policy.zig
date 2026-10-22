//! TensorFold glm5_next/cuda/dflash2.py: merge_candidates, Drafter.chain, best_depth.
//! Ash Hart and TensorFold contributors. Proposals only; keyed target sampling stays in P7.
const std = @import("std");
const sampling = @import("lanes").sampling;
extern "c" fn exp(f64) f64;
extern "c" fn log(f64) f64;
pub const Sampling = sampling.Sampling;
pub const Candidate = struct { id: u32, value: f64 };
fn before(_: void, a: Candidate, b: Candidate) bool {
    return a.value > b.value or (a.value == b.value and a.id < b.id);
}
/// Packed [world, rows, 2*k] fp32: values followed by bitcast int32 global IDs.
pub fn merge(pack_buf: []const f32, world: usize, rows: usize, k: usize, vocab: usize, out: []Candidate) !void {
    if (world == 0 or world > 4 or k == 0 or k > 16 or pack_buf.len != world * rows * 2 * k or out.len != rows * k) return error.BadCandidates;
    var buf: [64]Candidate = undefined;
    for (0..rows) |r| {
        for (0..world) |rank| for (0..k) |j| {
            const base = (rank * rows + r) * 2 * k;
            const id: u32 = @bitCast(pack_buf[base + k + j]);
            if (id >= vocab or !std.math.isFinite(pack_buf[base + j])) return error.BadCandidates;
            buf[rank * k + j] = .{ .id = id, .value = pack_buf[base + j] };
        };
        std.mem.sort(Candidate, buf[0 .. world * k], {}, before);
        @memcpy(out[r * k ..][0..k], buf[0..k]);
    }
}
pub const Selector = struct {
    pred: []const f32,
    succ: []const f32,
    rank: usize = 256,
    pub fn validate(s: Selector, vocab: usize) !void {
        if (s.rank == 0 or s.pred.len != vocab * s.rank or s.succ.len != s.pred.len) return error.BadSelector;
    }
};
pub const Policy = struct {
    most: usize = 7,
    threshold: f64 = 0.3,
    beta: f64 = 2,
    /// Supported served policy, as parsed by multi_tune.py: fnc7:0.3.
    pub fn parse(code: []const u8) !Policy {
        if (!std.mem.startsWith(u8, code, "fnc")) return error.BadDraftPolicy;
        const colon = std.mem.indexOfScalar(u8, code, ':') orelse return error.BadDraftPolicy;
        const n = std.fmt.parseInt(usize, code[3..colon], 10) catch return error.BadDraftPolicy;
        const p = std.fmt.parseFloat(f64, code[colon + 1 ..]) catch return error.BadDraftPolicy;
        if (n < 1 or n > 7 or !std.math.isFinite(p) or p < 0 or p > 1) return error.BadDraftPolicy;
        return .{ .most = n, .threshold = p };
    }
};
pub fn bestDepth(q: []const f64, round_ms: []const f64) !usize {
    if (round_ms.len < q.len + 1) return error.BadDraftCosts;
    var alive: f64 = 1;
    var expect: f64 = 1;
    var best: f64 = -1;
    var depth: usize = 0;
    for (0..q.len + 1) |k| {
        if (k > 0) {
            alive *= q[k - 1];
            expect += alive;
        }
        if (!std.math.isFinite(round_ms[k]) or round_ms[k] <= 0) return error.BadDraftCosts;
        const rate = expect / round_ms[k];
        if (rate > best) {
            best = rate;
            depth = k;
        }
    }
    return depth;
}
/// Candidate rows and selector projection are private copies of one stream's block results.
pub fn chain(candidates: []const Candidate, projection: []const f32, selector: Selector, anchor: u32, first: usize, s: ?Sampling, policy: Policy, out: []u32, costs: ?[]const f64) !usize {
    const k: usize = 16;
    if (policy.most < 1 or policy.most > 7 or !std.math.isFinite(policy.beta) or policy.beta <= 0 or !std.math.isFinite(policy.threshold) or policy.threshold < 0 or policy.threshold > 1) return error.BadDraftPolicy;
    if (candidates.len % k != 0 or projection.len != candidates.len / k * selector.rank or selector.rank == 0 or selector.pred.len % selector.rank != 0) return error.BadCandidates;
    const vocab = selector.pred.len / selector.rank;
    try selector.validate(vocab);
    if (anchor >= vocab) return error.BadCandidates;
    const depth = @min(@min(out.len, policy.most), candidates.len / k);
    var prev = anchor;
    var product: f64 = 1;
    var conf: [7]f64 = undefined;
    var count: usize = 0;
    const sampled = s != null and s.?.temperature > 0;
    const temp: f64 = if (sampled) s.?.temperature else 1;
    for (0..depth) |d| {
        var z: [16]f64 = undefined;
        var max_pick: f64 = -std.math.inf(f64);
        var max_z: f64 = -std.math.inf(f64);
        var pick: usize = 0;
        for (candidates[d * k ..][0..k], 0..) |c, j| {
            if (c.id >= vocab or !std.math.isFinite(c.value)) return error.BadCandidates;
            var edge: f64 = 0;
            for (0..selector.rank) |i| edge += @as(f64, selector.succ[@as(usize, c.id) * selector.rank + i]) * (@as(f64, selector.pred[@as(usize, prev) * selector.rank + i]) * projection[d * selector.rank + i]);
            const score = (c.value + 0.6 * edge) / temp;
            const noise: f64 = if (sampled) -log(-log(sampling.uniform(s.?.seed, first + d, c.id))) else 0;
            const value = score + 0.7 * noise;
            if (value > max_pick) {
                max_pick = value;
                pick = j;
            }
            z[j] = (score + noise) / policy.beta;
            max_z = @max(max_z, z[j]);
        }
        var sum: f64 = 0;
        for (&z) |*v| {
            v.* = exp(v.* - max_z);
            sum += v.*;
        }
        conf[d] = z[pick] / sum;
        product *= conf[d];
        if (costs == null and d > 0 and product < policy.threshold) break;
        prev = candidates[d * k + pick].id;
        out[count] = prev;
        count += 1;
    }
    return if (costs) |ms| try bestDepth(conf[0..count], ms) else count;
}
test "merge rank ordered values and bitcast IDs, reject corrupt IDs" {
    var out: [2]Candidate = undefined;
    const g = [_]f32{ 5, 4, @bitCast(@as(u32, 9)), @bitCast(@as(u32, 2)), 5, 3, @bitCast(@as(u32, 1)), @bitCast(@as(u32, 3)) };
    try merge(&g, 2, 1, 2, 10, &out);
    try std.testing.expectEqual(@as(u32, 1), out[0].id);
    try std.testing.expectEqual(@as(u32, 9), out[1].id);
    try std.testing.expectError(error.BadCandidates, merge(&g, 2, 1, 2, 5, &out));
}
test "fnc always keeps first, cumulative noisy confidence stops later drafts" {
    const p = try Policy.parse("fnc7:0.3");
    var cs: [112]Candidate = undefined;
    for (&cs, 0..) |*c, i| c.* = .{ .id = @intCast(i % 16), .value = 0 };
    const code: [16]f32 = @splat(0);
    const proj: [7]f32 = @splat(0);
    var out: [7]u32 = undefined;
    try std.testing.expectEqual(@as(usize, 1), try chain(&cs, &proj, .{ .pred = &code, .succ = &code, .rank = 1 }, 0, 22, null, p, &out, null));
    try std.testing.expectEqual(@as(usize, 7), try chain(&cs, &proj, .{ .pred = &code, .succ = &code, .rank = 1 }, 0, 22, null, .{ .threshold = 0 }, &out, null));
    try std.testing.expectEqual(@as(usize, 1), try bestDepth(&.{ 0.9, 0.1 }, &.{ 1, 1.1, 2 }));
}
