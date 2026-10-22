//! Port of glm5_next/cuda/multi_tune.py:144 sample_packed and engine/exact_sampling.py.
//! Ash Hart and TensorFold contributors. Seed/position/id draws reuse core/lanes/sampling.
const std = @import("std");
const cuda = @import("cuda");
const S = @import("lanes").sampling;
const nu = @import("cuda_nucleus.zig");
const ops = @import("cuda_torch_ops.zig");
const Arena = @import("cuda_state.zig").Arena;
pub const Comm = @import("cuda_comm.zig").Comm;
pub const Part = struct { probs: ?[]f64 = null, logits: u64, positions: []const u64, sampling: ?S.Sampling, out: []u32 };
pub fn isGreedy(p: Part) bool {
    return p.sampling == null or p.sampling.?.temperature <= 0;
}
/// samplePacked's fast path: glm_greedy_pick writes one [value | id] a row, so only a greedy part planned with one
/// candidate may take it. A greedy draft that wants its confidence (probs) is planned with up to 28 (plan), which
/// only the top-k branch writes; sent to glm_greedy_pick, its ids would be read from stale memory.
pub fn usesGreedyPick(p: Part, q: Plan, legacy_greedy: bool) bool {
    return isGreedy(p) and !legacy_greedy and q.count == 1;
}
/// The order glm_greedy_pick (and the radix top-k's selection_key) ranks floats by: any NaN above everything, then the
/// IEEE total order of the bits (+0 above -0, -inf lowest).
pub fn greedyKey(v: f32) u32 {
    const bits: u32 = @bitCast(v);
    if (bits & 0x7fffffff > 0x7f800000) return std.math.maxInt(u32);
    return if (bits & 0x80000000 != 0) ~bits else bits | 0x80000000;
}
/// Host model of the greedy top-1: the column of the largest key, the lowest column among equal keys.
pub fn greedyColumn(vals: []const f32) usize {
    var best: usize = 0;
    for (vals, 0..) |v, i| if (greedyKey(v) > greedyKey(vals[best])) {
        best = i;
    };
    return best;
}
pub const Plan = struct { rows: usize, count: usize, offset: usize, nucleus: bool };
pub fn plan(parts: []const Part, columns: usize, out: []Plan) !usize {
    if (columns == 0 or parts.len != out.len) return error.BadSamplingShape;
    var at: usize = 0;
    for (parts, out) |p, *q| {
        if (p.positions.len == 0 or p.out.len != p.positions.len or (p.probs != null and p.probs.?.len != p.out.len)) return error.BadSamplingShape;
        const greedy = isGreedy(p);
        const nucleus = !greedy and p.sampling.?.top_k == 0;
        const n = if (nucleus) 1 else if (greedy) (if (p.probs != null) @min(columns, 28) else @as(usize, 1)) else @min(columns, @as(usize, p.sampling.?.top_k) + 8);
        q.* = .{ .rows = p.positions.len, .count = n, .offset = at, .nucleus = nucleus };
        at += p.positions.len * 2 * n;
    }
    return at;
}
/// First packed gather [world, words], part's rows [values | int32 global ids as f32 bits].
pub fn choosePacked(gpa: std.mem.Allocator, got: []const f32, world: usize, words: usize, q: Plan, positions: []const u64, s: ?S.Sampling, out: []u32) !void {
    if (world < 1 or got.len != world * words or q.nucleus or positions.len != q.rows or out.len != q.rows or q.offset + q.rows * 2 * q.count > words) return error.BadPackedSampling;
    const vals = try gpa.alloc(f64, world * q.count);
    defer gpa.free(vals);
    const ids = try gpa.alloc(u64, vals.len);
    defer gpa.free(ids);
    for (0..q.rows) |r| {
        for (0..world) |rank| {
            const row = got[rank * words + q.offset + r * 2 * q.count ..][0 .. 2 * q.count];
            for (0..q.count) |j| {
                vals[rank * q.count + j] = row[j];
                const id: i32 = @bitCast(row[q.count + j]);
                if (id < 0) return error.BadTokenId;
                ids[rank * q.count + j] = @intCast(id);
            }
        }
        if (s != null and s.?.temperature > 0) {
            out[r] = @intCast(try S.choose(gpa, vals, ids, positions[r], s.?));
        } else {
            var best: usize = 0;
            for (vals, ids, 0..) |v, id, j| if (v > vals[best] or (v == vals[best] and id < ids[best])) {
                best = j;
            };
            out[r] = @intCast(ids[best]);
        }
    }
}
const Work = struct {
    arena: Arena,
    rows: usize = 0,
    f32: u64,
    vals: u64,
    ids: u64,
    scratch: u64,
    mass: u64,
    sums: u64,
    top: u64,
    fn init(d: *const cuda.Driver, rows: usize, columns: usize) !Work {
        const sizes = [_]usize{ rows * columns * 4, rows * columns * 4, rows * columns * 8, ops.shared.topkScratchBytes(rows, columns), rows * columns * 8, rows * 8, rows * 8 };
        var a = try Arena.init(d, &sizes);
        errdefer a.deinit();
        const converted = a.take(sizes[0]);
        const vals = a.take(sizes[1]);
        const ids = a.take(sizes[2]);
        const scratch = a.take(sizes[3]);
        const mass = a.take(sizes[4]);
        const sums = a.take(sizes[5]);
        const top = a.take(sizes[6]);
        return .{ .arena = a, .rows = rows, .f32 = converted, .vals = vals, .ids = ids, .scratch = scratch, .mass = mass, .sums = sums, .top = top };
    }
};
/// One collective's persistent buffers (send and gathered, device; gathered, pinned host). They only ever grow, so a
/// round allocates, zeroes and frees nothing. The gathered result is a slice of the pinned buffer, valid until the
/// same Exchange gathers again.
const Exchange = struct {
    send: ?cuda.DeviceBuffer = null,
    recv: ?cuda.DeviceBuffer = null,
    host: ?cuda.HostBuffer = null,
    world: usize = 1,
    fn deinit(x: *Exchange) void {
        if (x.send) |*b| b.free();
        if (x.recv) |*b| b.free();
        if (x.host) |*b| b.free();
        x.* = .{};
    }
    /// Room for `bytes` sent per rank (a multiple of 4), `world` ranks gathered.
    fn ensure(x: *Exchange, d: *const cuda.Driver, bytes: usize, world: usize) !void {
        if (x.send != null and x.send.?.len >= bytes and x.world == world) return;
        x.deinit();
        const n = @max(bytes, 4096);
        x.send = try cuda.DeviceBuffer.alloc(d, n);
        errdefer x.deinit();
        x.recv = try cuda.DeviceBuffer.alloc(d, n * world);
        x.host = try cuda.HostBuffer.alloc(d, n * world);
        x.world = world;
    }
};
pub const Sampler = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    th: ops.Torch,
    comm: ?*Comm = null,
    columns: usize,
    vocab_offset: i64,
    union_cover: bool = false,
    /// TF_GLM_GREEDY_CHECK=1: every greedy pick is recomputed through to_f32 + radix top-k + sort + pack and compared bit for bit.
    check_greedy: bool = false,
    /// TF_GLM_GREEDY_TOPK=1: greedy through to_f32 + radix top-k + sort + pack, as before glm_greedy_pick (A/B timing).
    legacy_greedy: bool = false,
    /// first / nucleus second / nucleus retry collectives, and the per-part vocabulary-wide workspaces (grow-only).
    xs: [3]Exchange = @splat(.{}),
    works: [8]?Work = @splat(null),
    /// greedy_pick's per-block partial bests (64 rows).
    part: ?cuda.DeviceBuffer = null,
    pub fn deinit(s: *Sampler) void {
        for (&s.xs) |*x| x.deinit();
        for (&s.works) |*w| if (w.*) |*x| x.arena.deinit();
        s.works = @splat(null);
        if (s.part) |*b| b.free();
        s.part = null;
    }
    fn partial(s: *Sampler) !u64 {
        if (s.part == null) s.part = try cuda.DeviceBuffer.alloc(s.d, 64 * ops.greedy_split * 8);
        return s.part.?.ptr;
    }
    fn world(s: *const Sampler) usize {
        return if (s.comm) |cm| cm.world else 1;
    }
    /// `bytes` of x.send (already filled on the stream) gathered from every rank: pinned host words [world * bytes / 4].
    fn gather(s: *Sampler, x: *Exchange, bytes: usize) ![]f32 {
        const send = x.send.?;
        const recv = x.recv.?;
        if (s.comm) |cm| try cm.allGather(send.ptr, recv.ptr, bytes / 4, .f32, s.th.s.handle) else try s.th.base.copy(recv.ptr, send.ptr, bytes);
        const n = bytes * s.world();
        try recv.downloadAsync(0, x.host.?.bytes[0..n], s.th.s.handle);
        try s.th.s.synchronize();
        return x.host.?.slice(f32)[0 .. n / 4];
    }
    fn checkGreedy(s: *Sampler, i: usize, p: Part, q: Plan, dst: u64) !void {
        if (q.rows > 64) return;
        const w = try s.work(i, q.rows);
        var old = try cuda.DeviceBuffer.alloc(s.d, q.rows * 8);
        defer old.free();
        try s.th.base.toF32(p.logits, w.f32, q.rows * s.columns);
        try s.th.topkSorted(w.f32, s.columns, q.rows, 1, w.vals, w.ids, w.scratch);
        try s.th.pack(w.vals, w.ids, old.ptr, q.rows, 1, s.vocab_offset);
        try s.th.s.synchronize();
        var a: [64 * 2]u32 = undefined;
        var b: [64 * 2]u32 = undefined;
        const n = q.rows * 2;
        try old.download(0, std.mem.sliceAsBytes(a[0..n]));
        try (cuda.DeviceBuffer{ .d = s.d, .ptr = dst, .len = n * 4 }).download(0, std.mem.sliceAsBytes(b[0..n]));
        if (!std.mem.eql(u32, a[0..n], b[0..n])) {
            std.log.err("greedy pick differs from top-k: {any} vs {any}", .{ a[0..n], b[0..n] });
            return error.GreedyPickMismatch;
        }
    }
    /// The part's workspace, with room for `rows` rows (kept between calls).
    fn work(s: *Sampler, i: usize, rows: usize) !*Work {
        if (s.works[i]) |*w| {
            if (w.rows >= rows) return w;
            w.arena.deinit();
            s.works[i] = null;
        }
        s.works[i] = try Work.init(s.d, rows, s.columns);
        return &s.works[i].?;
    }
    /// All top-k and maxima in one collective, then every nucleus part's candidates
    /// in one second collective. An uncovered part retries WIDER (opt-in) / whole shards.
    pub fn samplePacked(s: *Sampler, parts: []const Part) !void {
        if (parts.len == 0) return;
        if (parts.len > s.works.len) return error.BadSamplingShape;
        var plans_buf: [8]Plan = undefined;
        const plans = plans_buf[0..parts.len];
        const words = try plan(parts, s.columns, plans);
        try s.xs[0].ensure(s.d, words * 4, s.world());
        const first = s.xs[0].send.?;
        var nucleus_rows: usize = 0;
        for (parts, plans, 0..) |p, q, i| {
            const dst = first.ptr + q.offset * 4;
            if (q.nucleus) {
                try s.th.scaledMax(p.logits, dst, q.rows, s.columns, @max(p.sampling.?.temperature, 1e-6));
                nucleus_rows += q.rows;
            } else if (usesGreedyPick(p, q, s.legacy_greedy)) {
                try s.th.greedyPick(p.logits, try s.partial(), dst, q.rows, s.columns, s.vocab_offset);
                if (s.check_greedy) try s.checkGreedy(i, p, q, dst);
            } else {
                const w = try s.work(i, q.rows);
                try s.th.base.toF32(p.logits, w.f32, q.rows * s.columns);
                try s.th.topkSorted(w.f32, s.columns, q.rows, q.count, w.vals, w.ids, w.scratch);
                try s.th.pack(w.vals, w.ids, dst, q.rows, q.count, s.vocab_offset);
            }
        }
        const got = try s.gather(&s.xs[0], words * 4);
        const count = nu.nucleus;
        const second_bytes = @max(1, nucleus_rows * (3 * count + 2) * 8);
        var row0: usize = 0;
        for (parts, plans, 0..) |p, q, i| {
            if (!q.nucleus) {
                try choosePacked(s.gpa, got, s.world(), words, q, p.positions, p.sampling, p.out);
                if (p.probs) |probs| for (probs, 0..) |*prob, r| {
                    const vals = try s.gpa.alloc(f64, s.world() * q.count);
                    defer s.gpa.free(vals);
                    const ids = try s.gpa.alloc(u64, vals.len);
                    defer s.gpa.free(ids);
                    for (0..s.world()) |rank| for (0..q.count) |j| {
                        const at = rank * words + q.offset + r * 2 * q.count;
                        vals[rank * q.count + j] = got[at + j];
                        ids[rank * q.count + j] = @as(u32, @bitCast(got[at + q.count + j]));
                    };
                    prob.* = try probability(s.gpa, vals, ids, p.out[r], p.sampling);
                };
                continue;
            }
            if (s.xs[1].send == null or s.xs[1].send.?.len < second_bytes or s.xs[1].world != s.world()) try s.xs[1].ensure(s.d, second_bytes, s.world());
            const w = try s.work(i, q.rows);
            const tops = try s.gpa.alloc(f64, q.rows);
            defer s.gpa.free(tops);
            for (tops, 0..) |*top, r| {
                top.* = -std.math.inf(f64);
                for (0..s.world()) |rank| {
                    const at = rank * words + q.offset + 2 * r;
                    const bits = @as(u64, @as(u32, @bitCast(got[at]))) | (@as(u64, @as(u32, @bitCast(got[at + 1]))) << 32);
                    top.* = @max(top.*, @as(f64, @bitCast(bits)));
                }
            }
            try s.th.s.synchronize();
            try (cuda.DeviceBuffer{ .d = s.d, .ptr = w.top, .len = q.rows * 8 }).upload(0, std.mem.sliceAsBytes(tops));
            const temp = @max(p.sampling.?.temperature, 1e-6);
            try s.th.base.nucleusMass(p.logits, s.columns, q.rows, s.columns, temp, w.top, w.mass, w.sums);
            try s.th.base.toF32(p.logits, w.f32, q.rows * s.columns);
            try s.packNucleus(w.*, q.rows, count, s.xs[1].send.?.ptr + row0 * (3 * count + 2) * 8, temp);
            row0 += q.rows;
        }
        if (nucleus_rows == 0) return;
        const got2 = try s.gather(&s.xs[1], second_bytes);
        row0 = 0;
        for (parts, plans, 0..) |p, q, i| {
            if (!q.nucleus) continue;
            var ok = try s.drawNucleus(got2, nucleus_rows, row0, q.rows, count, p);
            const widest = widestShard(got2, s.world(), nucleus_rows, row0, q.rows, count);
            const w = s.works[i].?;
            if (!ok and s.union_cover and widest > 16384) ok = try s.retry(w, q.rows, 16384, p);
            if (!ok) ok = try s.retry(w, q.rows, widest, p);
            if (!ok) return error.NucleusUncovered;
            row0 += q.rows;
        }
    }
    fn packNucleus(s: *Sampler, w: Work, rows: usize, count: usize, dst: u64, temp: f64) !void {
        const selected = @min(count, s.columns);
        // Full-shard retry needs no device sort: _draw lexsorts on the host.
        if (selected <= 16384) try s.th.topkSorted(w.f32, s.columns, rows, selected, w.vals, w.ids, w.scratch) else try s.th.base.topk(w.f32, s.columns, rows, selected, w.vals, w.ids, w.scratch);
        try s.th.massPack(w.vals, w.ids, w.mass, w.sums, dst, rows, s.columns, selected, count, s.vocab_offset, temp);
    }
    fn retry(s: *Sampler, w: Work, rows: usize, count: usize, p: Part) !bool {
        const bytes = rows * (3 * count + 2) * 8;
        try s.xs[2].ensure(s.d, bytes, s.world());
        try s.packNucleus(w, rows, count, s.xs[2].send.?.ptr, @max(p.sampling.?.temperature, 1e-6));
        const got = try s.gather(&s.xs[2], bytes);
        return s.drawNucleus(got, rows, 0, rows, count, p);
    }
    fn drawNucleus(s: *const Sampler, got: []const f32, all_rows: usize, row0: usize, rows: usize, count: usize, p: Part) !bool {
        const n = s.world() * rows * count;
        const vals = try s.gpa.alloc(f64, n);
        defer s.gpa.free(vals);
        const ids = try s.gpa.alloc(i64, n);
        defer s.gpa.free(ids);
        const mass = try s.gpa.alloc(i64, n);
        defer s.gpa.free(mass);
        const sums = try s.gpa.alloc(i64, s.world() * rows);
        defer s.gpa.free(sums);
        const widths = try s.gpa.alloc(i64, sums.len);
        defer s.gpa.free(widths);
        const width = 3 * count + 2;
        for (0..s.world()) |rank| for (0..rows) |r| {
            const start = (rank * all_rows + row0 + r) * width * 2;
            for (0..width) |j| {
                const bits = @as(u64, @as(u32, @bitCast(got[start + 2 * j]))) | (@as(u64, @as(u32, @bitCast(got[start + 2 * j + 1]))) << 32);
                const at = (rank * rows + r) * count;
                if (j < count) vals[at + j] = @bitCast(bits) else if (j < 2 * count) ids[at + j - count] = @bitCast(bits) else if (j < 3 * count) mass[at + j - 2 * count] = @bitCast(bits) else if (j == 3 * count) sums[rank * rows + r] = @bitCast(bits) else widths[rank * rows + r] = @bitCast(bits);
            }
        };
        const out = try s.gpa.alloc(nu.Drawn, rows);
        defer s.gpa.free(out);
        const ok = try nu.drawSharesWithUnion(s.gpa, .{ .world = s.world(), .rows = rows, .count = count, .vals = vals, .ids = ids, .mass = mass, .sums = sums, .widths = widths }, p.positions, p.sampling.?, out, s.union_cover);
        if (ok) for (out, p.out, 0..) |x, *token, r| {
            token.* = @intCast(x.token);
            if (p.probs) |probs| probs[r] = x.share;
        };
        return ok;
    }
};
test "packed greedy merge ties by global id across ranks" {
    const id = struct {
        fn bits(x: u32) f32 {
            return @bitCast(x);
        }
    }.bits;
    var out: [2]u32 = undefined;
    try choosePacked(std.testing.allocator, &.{ 5, id(8), 7, id(9), 5, id(2), 6, id(1) }, 2, 4, .{ .rows = 2, .count = 1, .offset = 0, .nucleus = false }, &.{ 10, 11 }, null, &out);
    try std.testing.expectEqualSlices(u32, &.{ 2, 9 }, &out);
}
test "analyze packed device pipeline without running GPU" {
    _ = Sampler.samplePacked;
}

fn widestShard(got: []const f32, world: usize, all_rows: usize, row0: usize, rows: usize, count: usize) usize {
    var widest: usize = 0;
    for (0..world) |rank| for (0..rows) |r| {
        const at = ((rank * all_rows + row0 + r) * (3 * count + 2) + 3 * count + 1) * 2;
        const bits = @as(u64, @as(u32, @bitCast(got[at]))) | (@as(u64, @as(u32, @bitCast(got[at + 1]))) << 32);
        widest = @max(widest, @as(usize, @intCast(bits)));
    };
    return widest;
}
test "packed sampled draws and uniform bits match Python reference CPU fixtures" {
    const Fixture = struct { world: usize, rows: usize, count: usize, vals: []u32, ids: []u32, positions: []u64, seed: u64, temperature: f64, top_k: u32, top_p: f64, min_p: f64, sampled: []u32, uniform: []u64 };
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice([]Fixture, gpa, @embedFile("fixtures_cuda_sampling.json"), .{});
    defer parsed.deinit();
    for (parsed.value) |f| {
        const words = f.rows * 2 * f.count;
        const got = try gpa.alloc(f32, f.world * words);
        defer gpa.free(got);
        for (0..f.world) |rank| for (0..f.rows) |r| for (0..f.count) |j| {
            const src = (rank * f.rows + r) * f.count + j;
            const dst = rank * words + r * 2 * f.count + j;
            got[dst] = @bitCast(f.vals[src]);
            got[dst + f.count] = @bitCast(f.ids[src]);
        };
        const out = try gpa.alloc(u32, f.rows);
        defer gpa.free(out);
        try choosePacked(gpa, got, f.world, words, .{ .rows = f.rows, .count = f.count, .offset = 0, .nucleus = false }, f.positions, .{ .seed = f.seed, .temperature = f.temperature, .top_k = f.top_k, .top_p = f.top_p, .min_p = f.min_p }, out);
        try std.testing.expectEqualSlices(u32, f.sampled, out);
        for (0..f.world) |rank| for (0..f.count) |j| {
            const u = S.uniform(f.seed, f.positions[0], f.ids[rank * f.rows * f.count + j]);
            try std.testing.expectEqual(f.uniform[rank * f.count + j], @as(u64, @bitCast(u)));
        };
    }
}
test "more parts than persistent workspaces is refused before any device call" {
    var s: Sampler = undefined;
    s.works = @splat(null);
    var out: [1]u32 = undefined;
    const part: Part = .{ .logits = 0, .positions = &.{1}, .sampling = null, .out = &out };
    const nine: [9]Part = @splat(part);
    try std.testing.expectError(error.BadSamplingShape, s.samplePacked(&nine));
}
test "plan sizes the first collective: one value and one id a greedy row" {
    var out: [2]Plan = undefined;
    var a: [2]u32 = undefined;
    var b: [3]u32 = undefined;
    const parts = [_]Part{ .{ .logits = 0, .positions = &.{ 1, 2 }, .sampling = null, .out = &a }, .{ .logits = 0, .positions = &.{ 1, 2, 3 }, .sampling = null, .out = &b } };
    try std.testing.expectEqual(@as(usize, 10), try plan(&parts, 100, &out));
    try std.testing.expectEqual(@as(usize, 4), out[1].offset);
}
test "greedy order: NaN first, +0 over -0, lowest column among ties" {
    const nan: f32 = @bitCast(@as(u32, 0x7fc00001));
    const neg_nan: f32 = @bitCast(@as(u32, 0xffc00000));
    const inf = std.math.inf(f32);
    try std.testing.expectEqual(@as(usize, 2), greedyColumn(&.{ 1, inf, nan, nan, 5 }));
    try std.testing.expectEqual(@as(usize, 1), greedyColumn(&.{ -inf, neg_nan, nan }));
    try std.testing.expectEqual(@as(usize, 1), greedyColumn(&.{ -0.0, 0.0, 0.0 }));
    try std.testing.expectEqual(@as(usize, 0), greedyColumn(&.{ -0.0, -0.0, -1.0 }));
    try std.testing.expectEqual(@as(usize, 0), greedyColumn(&.{ -inf, -inf, -inf }));
    try std.testing.expectEqual(@as(usize, 3), greedyColumn(&.{ -3, -2, -2, -1.5, -1.5 }));
    try std.testing.expectEqual(@as(usize, 2), greedyColumn(&.{ 1e-40, 0, 2e-40, 1e-40 }));
    // monotone over all finite bf16 values, as the radix keys are
    var prev: u32 = 0;
    var bits: u32 = 0;
    var first = true;
    // negative values from -max up to -0, then +0 up to +max
    var order: [2 * 0x7f80]u32 = undefined;
    var n: usize = 0;
    bits = 0xff7f;
    while (bits >= 0x8000) : (bits -= 1) {
        order[n] = bits << 16;
        n += 1;
    }
    bits = 0;
    while (bits < 0x7f80) : (bits += 1) {
        order[n] = bits << 16;
        n += 1;
    }
    for (order[0..n]) |o| {
        const k = greedyKey(@bitCast(o));
        if (!first) try std.testing.expect(k > prev);
        prev = k;
        first = false;
    }
}

/// decode.py:_probability: lexsort, float64 softmax, top-p renormalization; min_p is deliberately absent.
extern "c" fn exp(f64) f64;
pub fn probability(gpa: std.mem.Allocator, values: []const f64, ids: []const u64, chosen: u32, sampling: ?S.Sampling) !f64 {
    const C = struct {
        value: f64,
        id: u64,
        fn less(_: void, a: @This(), b: @This()) bool {
            return a.value > b.value or (a.value == b.value and a.id < b.id);
        }
    };
    const order = try gpa.alloc(C, values.len);
    defer gpa.free(order);
    for (order, values, ids) |*o, v, id| o.* = .{ .value = v, .id = id };
    std.mem.sort(C, order, {}, C.less);
    const k = if (sampling != null and sampling.?.top_k != 0) @min(order.len, sampling.?.top_k) else order.len;
    const temp = if (sampling != null and sampling.?.temperature > 0) sampling.?.temperature else 1;
    const p = try gpa.alloc(f64, k);
    defer gpa.free(p);
    for (p, order[0..k]) |*v, o| v.* = exp(o.value / temp - order[0].value / temp);
    const sum = S.pairwiseSum(p);
    for (p) |*v| v.* /= sum;
    var keep = k;
    var renormalize = false;
    if (sampling) |s| if (s.top_p > 0 and s.top_p < 1) {
        renormalize = true;
        var cum: f64 = 0;
        keep = 0;
        for (p) |v| {
            cum += v;
            keep += 1;
            if (cum >= s.top_p) break;
        }
    };
    const mass = if (renormalize) S.pairwiseSum(p[0..keep]) else 1;
    for (order[0..keep], p[0..keep]) |o, v| if (o.id == chosen) return v / mass;
    return 0;
}
test "MTP confidence normalizes greedy competitors and sampled top-p; ignores min-p like Python" {
    const a = std.testing.allocator;
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), try probability(a, &.{ 0, 0 }, &.{ 2, 1 }, 1, null), 1e-15);
    try std.testing.expectEqual(@as(f64, 1), try probability(a, &.{ 0, 0 }, &.{ 2, 1 }, 1, .{ .seed = 0, .top_k = 2, .top_p = 0.4, .min_p = 1 }));
    try std.testing.expectEqual(@as(f64, 0), try probability(a, &.{ 0, 0 }, &.{ 2, 1 }, 2, .{ .seed = 0, .top_k = 2, .top_p = 0.4 }));
}

test "MTP confidence matches 36 small NumPy fixtures from Python decode._probability" {
    const F = struct { values: []f64, ids: []u64, chosen: u32, temperature: f64, top_p: f64, top_k: u32, probability: f64 };
    const parsed = try std.json.parseFromSlice([]F, std.testing.allocator, @embedFile("fixtures_mtp_probability.json"), .{});
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 36), parsed.value.len);
    for (parsed.value) |f| {
        const p = try probability(std.testing.allocator, f.values, f.ids, f.chosen, .{ .seed = 0, .temperature = f.temperature, .top_p = f.top_p, .top_k = f.top_k });
        try std.testing.expectApproxEqAbs(f.probability, p, 1e-15);
    }
}
test "a greedy draft that wants its confidence is planned 28 wide and stays off the one-candidate kernel" {
    var probs: [2]f64 = undefined;
    var o: [2]u32 = undefined;
    const pos = [_]u64{ 1, 2 };
    const zero: S.Sampling = .{ .seed = 1, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 };
    const hot_k: S.Sampling = .{ .seed = 1, .temperature = 0.7, .top_k = 20, .top_p = 1, .min_p = 0 };
    const hot_p: S.Sampling = .{ .seed = 1, .temperature = 0.7, .top_k = 0, .top_p = 0.9, .min_p = 0 };
    const Case = struct { part: Part, columns: usize, legacy: bool, count: usize, fast: bool };
    const cases = [_]Case{
        // MTP / DFlash2 drafts with confidence: as wide as the plan, never the one-candidate kernel
        .{ .part = .{ .probs = &probs, .logits = 0, .positions = &pos, .sampling = null, .out = &o }, .columns = 100, .legacy = false, .count = 28, .fast = false },
        .{ .part = .{ .probs = &probs, .logits = 0, .positions = &pos, .sampling = zero, .out = &o }, .columns = 100, .legacy = false, .count = 28, .fast = false },
        // a shard narrower than 28 columns: every column, still the top-k branch
        .{ .part = .{ .probs = &probs, .logits = 0, .positions = &pos, .sampling = null, .out = &o }, .columns = 10, .legacy = false, .count = 10, .fast = false },
        // one column: one candidate, which the fast kernel writes exactly
        .{ .part = .{ .probs = &probs, .logits = 0, .positions = &pos, .sampling = null, .out = &o }, .columns = 1, .legacy = false, .count = 1, .fast = true },
        // verification and plain greedy: one candidate, the fast kernel unless TF_GLM_GREEDY_TOPK
        .{ .part = .{ .logits = 0, .positions = &pos, .sampling = null, .out = &o }, .columns = 100, .legacy = false, .count = 1, .fast = true },
        .{ .part = .{ .logits = 0, .positions = &pos, .sampling = zero, .out = &o }, .columns = 100, .legacy = false, .count = 1, .fast = true },
        .{ .part = .{ .logits = 0, .positions = &pos, .sampling = null, .out = &o }, .columns = 100, .legacy = true, .count = 1, .fast = false },
        // sampled parts never take it
        .{ .part = .{ .logits = 0, .positions = &pos, .sampling = hot_k, .out = &o }, .columns = 100, .legacy = false, .count = 28, .fast = false },
        .{ .part = .{ .logits = 0, .positions = &pos, .sampling = hot_p, .out = &o }, .columns = 100, .legacy = false, .count = 1, .fast = false },
    };
    for (cases) |c| {
        var q: [1]Plan = undefined;
        _ = try plan(&.{c.part}, c.columns, &q);
        try std.testing.expectEqual(c.count, q[0].count);
        try std.testing.expectEqual(c.fast, usesGreedyPick(c.part, q[0], c.legacy));
    }
}
test "a 28-wide packed greedy row: the global top-1 across ranks and rows, and its confidence over the competitors" {
    const gpa = std.testing.allocator;
    const id = struct {
        fn bits(x: u32) f32 {
            return @bitCast(x);
        }
    }.bits;
    const count = 28;
    const rows = 2;
    const world = 2;
    const offset = 6; // a part after another part's words
    const words = offset + rows * 2 * count + 4; // and one more part after it
    const got = try gpa.alloc(f32, world * words);
    defer gpa.free(got);
    @memset(got, std.math.nan(f32)); // unrelated storage: never read
    for (0..world) |rank| for (0..rows) |r| {
        const row = got[rank * words + offset + r * 2 * count ..][0 .. 2 * count];
        for (0..count) |j| {
            row[j] = -100;
            row[count + j] = id(@intCast(1000 + rank * 100 + r * 30 + j));
        }
    };
    // row 0: both ranks hold 9.0; the lower global id (rank 1's 50) wins
    got[0 * words + offset + 3] = 9;
    got[0 * words + offset + count + 3] = id(100);
    got[1 * words + offset + 5] = 9;
    got[1 * words + offset + count + 5] = id(50);
    // row 1: rank 0 holds 7.0 (id 300), rank 1 the larger 8.0 (id 200) at its last column
    got[0 * words + offset + 2 * count + 0] = 7;
    got[0 * words + offset + 3 * count + 0] = id(300);
    got[1 * words + offset + 2 * count + count - 1] = 8;
    got[1 * words + offset + 3 * count + count - 1] = id(200);
    var out: [rows]u32 = undefined;
    try choosePacked(gpa, got, world, words, .{ .rows = rows, .count = count, .offset = offset, .nucleus = false }, &.{ 10, 11 }, null, &out);
    try std.testing.expectEqualSlices(u32, &.{ 50, 200 }, &out);
    // the confidence a greedy draft gets from those competitors (decode.py:_probability): one value 5, 27 zeros
    var vals: [count]f64 = @splat(0);
    var ids: [count]u64 = undefined;
    for (&ids, 0..) |*x, k| x.* = 10 + k;
    vals[0] = 5;
    const p = try probability(gpa, &vals, &ids, 10, null);
    try std.testing.expectApproxEqAbs(1.0 / (1.0 + 27.0 * @exp(-5.0)), p, 1e-12);
}
