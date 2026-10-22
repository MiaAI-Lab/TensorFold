//! Port of tensorfold/cuda/sampling.py _draw, _union_covers.
//! Flash Next's top_k-off draws (Python tensorfold/cuda/sampling.nucleus_rows): each token's share of the row's mass
//! in fixed point (floor(exp(f64(logit) / t - top) * 2^40), int64, on the GPU with CUDA's fp64 exp as torch's
//! elementwise kernels run it: cuda_torch_ops.nucleusMass), every rank's top candidates (value, global id, mass) with
//! its shard's mass sum and width, and the keyed draw over the top_p nucleus then min_p on the host (`_draw`): the
//! nucleus cut by exact integer mass sums, the same on every rank. A rank whose nucleus runs past its candidates
//! sends its whole shard instead. tools/zig/check_flashnext_nucleus.py checks the mass kernel against torch and
//! records fixtures of Python's own _shares / _draw, which the test here replays.
const std = @import("std");
const lanes = @import("lanes");

const Allocator = std.mem.Allocator;
const S = lanes.sampling;

extern "c" fn log(x: f64) f64;

/// sampling.MASS and NUCLEUS
pub const mass_scale: f64 = 1099511627776.0;
pub const nucleus: usize = 1024;

/// `_shares`' gathered arrays, [world][rows][count] row-major (and [world][rows] for sums and widths).
pub const Shares = struct {
    world: usize,
    rows: usize,
    count: usize,
    vals: []const f64,
    ids: []const i64,
    mass: []const i64,
    sums: []const i64,
    widths: []const i64,
};

pub const Drawn = struct { token: u64, share: f64 };

/// math.ceil(Fraction(top_p) * total), exact: top_p is m * 2^e.
fn need(top_p: f64, total: i128) i128 {
    const b: u64 = @bitCast(top_p);
    const raw_exp: i64 = @intCast((b >> 52) & 0x7ff);
    var m: u128 = b & ((@as(u64, 1) << 52) - 1);
    var e: i64 = undefined;
    if (raw_exp == 0) {
        e = -1074;
    } else {
        m |= @as(u128, 1) << 52;
        e = raw_exp - 1075;
    }
    const num: u128 = m * @as(u128, @intCast(total));
    if (e >= 0) return @intCast(num << @intCast(e));
    const sh: u7 = @intCast(@min(-e, 127));
    const one: u128 = @as(u128, 1) << sh;
    return @intCast((num + one - 1) >> sh);
}

const Entry = struct { v: f64, id: i64, m: i64 };

/// np.lexsort((ids, -vals)): value descending, then id ascending.
fn before(_: void, a: Entry, b: Entry) bool {
    if (a.v != b.v) return a.v > b.v;
    return a.id < b.id;
}

/// sampling._draw: each row's (token, share of the mass), or false when some rank's nucleus runs past its
/// candidates (the caller then sends whole shards).
pub fn drawShares(gpa: Allocator, sh: Shares, positions: []const u64, s: S.Sampling, out: []Drawn) !bool {
    return drawSharesWithUnion(gpa, sh, positions, s, out, false);
}

pub fn drawSharesWithUnion(gpa: Allocator, sh: Shares, positions: []const u64, s: S.Sampling, out: []Drawn, union_cover: bool) !bool {
    const cut = 0.0 < s.top_p and s.top_p < 1.0;
    const min_log = s.minLog();
    const list = try gpa.alloc(Entry, sh.world * sh.count);
    defer gpa.free(list);
    for (0..sh.rows) |r| {
        var total: i128 = 0;
        for (0..sh.world) |k| total += sh.sums[k * sh.rows + r];
        const want: i128 = if (cut) need(s.top_p, total) else 0;
        var top = -std.math.inf(f64);
        for (0..sh.world) |k| for (sh.vals[(k * sh.rows + r) * sh.count ..][0..sh.count]) |v| {
            top = @max(top, v);
        };
        const floor = top + min_log;
        if (cut and union_cover) {
            var n: usize = 0;
            var boundary = -std.math.inf(f64);
            for (0..sh.world) |k| {
                const at = (k * sh.rows + r) * sh.count;
                var real: usize = 0;
                var last = std.math.inf(f64);
                for (0..sh.count) |j| if (sh.ids[at + j] >= 0) {
                    list[n] = .{ .v = sh.vals[at + j], .id = sh.ids[at + j], .m = sh.mass[at + j] };
                    n += 1;
                    real += 1;
                    last = @min(last, sh.vals[at + j]);
                };
                if (real != sh.widths[k * sh.rows + r]) boundary = @max(boundary, last);
            }
            std.mem.sort(Entry, list[0..n], {}, before);
            var acc: i128 = 0;
            var covered = boundary == -std.math.inf(f64);
            for (list[0..n]) |x| {
                acc += x.m;
                if (acc >= want) {
                    covered = x.v > boundary;
                    break;
                }
            }
            if (!covered) return false;
        }
        // Per-shard coverage in the default path; opt-in union tested above.
        for (0..if (cut and union_cover) 0 else sh.world) |k| {
            const at = (k * sh.rows + r) * sh.count;
            var n: usize = 0;
            for (0..sh.count) |j| if (sh.ids[at + j] >= 0) {
                list[n] = .{ .v = sh.vals[at + j], .id = sh.ids[at + j], .m = sh.mass[at + j] };
                n += 1;
            };
            if (@as(i64, @intCast(n)) == sh.widths[k * sh.rows + r]) continue; // its whole shard
            std.mem.sort(Entry, list[0..n], {}, before);
            var covered = false;
            if (cut) {
                var acc: i128 = 0;
                for (list[0..n]) |x| {
                    acc += x.m;
                    if (acc >= want) {
                        covered = x.v > list[n - 1].v;
                        break;
                    }
                }
            } else if (s.min_p > 0.0) {
                for (list[0..n]) |x| if (x.v < floor) {
                    covered = true;
                    break;
                };
            }
            if (!covered) return false;
        }
        var n: usize = 0;
        for (0..sh.world) |k| {
            const at = (k * sh.rows + r) * sh.count;
            for (0..sh.count) |j| if (sh.ids[at + j] >= 0) {
                list[n] = .{ .v = sh.vals[at + j], .id = sh.ids[at + j], .m = sh.mass[at + j] };
                n += 1;
            };
        }
        std.mem.sort(Entry, list[0..n], {}, before);
        var keep = n;
        if (cut) {
            var acc: i128 = 0;
            var below: usize = 0;
            for (list[0..n]) |x| {
                acc += x.m;
                if (acc < want) below += 1;
            }
            keep = below + 1;
        }
        if (s.min_p > 0.0) {
            var above: usize = 0;
            for (list[0..n]) |x| above += @intFromBool(x.v >= floor);
            keep = @min(keep, above);
        }
        keep = @min(keep, n);
        if (keep == 0) return error.EmptyNucleus;
        var best: usize = 0;
        var best_score: f64 = undefined;
        for (list[0..keep], 0..) |x, j| {
            const u = S.uniform(s.seed, positions[r], @bitCast(x.id));
            const score = x.v - log(-log(u));
            if (j == 0 or score > best_score) {
                best = j;
                best_score = score;
            }
        }
        out[r] = .{ .token = @intCast(list[best].id), .share = @as(f64, @floatFromInt(list[best].m)) / @as(f64, @floatFromInt(total)) };
    }
    return true;
}

test "need is math.ceil(Fraction(top_p) * total)" {
    try std.testing.expectEqual(@as(i128, 95), need(0.95, 100)); // 0.95 is just under 95/100: ceil gives 95
    try std.testing.expectEqual(@as(i128, 1), need(0.5, 1));
    try std.testing.expectEqual(@as(i128, 0), need(0.5, 0));
    try std.testing.expectEqual(@as(i128, 549755813888), need(0.5, 1099511627776));
}

fn flatten(gpa: Allocator, comptime T: type, v: std.json.Value, out: *std.ArrayList(T)) !void {
    switch (v) {
        .array => |a| for (a.items) |x| try flatten(gpa, T, x, out),
        .integer => |i| try out.append(gpa, if (T == f64) @bitCast(@as(u64, @bitCast(i))) else @intCast(i)),
        // f64 values travel as their u64 bits; ids, masses, sums and widths as signed integers
        .number_string => |s| try out.append(gpa, if (T == f64) @bitCast(try std.fmt.parseInt(u64, s, 10)) else try std.fmt.parseInt(i64, s, 10)),
        else => return error.BadFixture,
    }
}

const Owned = struct {
    vals: std.ArrayList(f64) = .empty,
    ids: std.ArrayList(i64) = .empty,
    mass: std.ArrayList(i64) = .empty,
    sums: std.ArrayList(i64) = .empty,
    widths: std.ArrayList(i64) = .empty,

    fn deinit(o: *Owned, gpa: Allocator) void {
        o.vals.deinit(gpa);
        o.ids.deinit(gpa);
        o.mass.deinit(gpa);
        o.sums.deinit(gpa);
        o.widths.deinit(gpa);
    }

    fn shares(o: *Owned, gpa: Allocator, v: std.json.ObjectMap, world: usize, rows: usize) !Shares {
        try flatten(gpa, f64, v.get("vals").?, &o.vals);
        try flatten(gpa, i64, v.get("ids").?, &o.ids);
        try flatten(gpa, i64, v.get("mass").?, &o.mass);
        try flatten(gpa, i64, v.get("sums").?, &o.sums);
        try flatten(gpa, i64, v.get("widths").?, &o.widths);
        return .{ .world = world, .rows = rows, .count = try std.fmt.parseInt(usize, v.get("count").?.number_string, 10), .vals = o.vals.items, .ids = o.ids.items, .mass = o.mass.items, .sums = o.sums.items, .widths = o.widths.items };
    }
};

test "the draws equal Python's _draw on its own candidates, one rank and two, whole-shard fallbacks included" {
    const gpa = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, @embedFile("fixtures_cuda_nucleus.json"), .{ .parse_numbers = false });
    defer parsed.deinit();
    for (parsed.value.object.get("cases").?.array.items, 0..) |cv, ci| {
        const c = cv.object;
        const world: usize = @intCast(try std.fmt.parseInt(i64, c.get("world").?.number_string, 10));
        const rows: usize = @intCast(try std.fmt.parseInt(i64, c.get("rows").?.number_string, 10));
        var pos: [8]u64 = undefined;
        for (c.get("positions").?.array.items, 0..) |p, i| pos[i] = try std.fmt.parseInt(u64, p.number_string, 10);
        const so = c.get("sampling").?.object;
        const s: S.Sampling = .{ .seed = try std.fmt.parseInt(u64, so.get("seed").?.number_string, 10), .temperature = try std.fmt.parseFloat(f64, so.get("temperature").?.number_string), .top_k = 0, .top_p = try std.fmt.parseFloat(f64, so.get("top_p").?.number_string), .min_p = try std.fmt.parseFloat(f64, so.get("min_p").?.number_string) };
        var first: Owned = .{};
        defer first.deinit(gpa);
        var out: [8]Drawn = undefined;
        var ok = try drawShares(gpa, try first.shares(gpa, c.get("first").?.object, world, rows), pos[0..rows], s, out[0..rows]);
        var fb: Owned = .{};
        defer fb.deinit(gpa);
        if (c.get("fallback").? != .null) {
            errdefer std.debug.print("case {d}: the first candidates should not cover\n", .{ci});
            try std.testing.expect(!ok);
            ok = try drawShares(gpa, try fb.shares(gpa, c.get("fallback").?.object, world, rows), pos[0..rows], s, out[0..rows]);
        }
        errdefer std.debug.print("case {d}\n", .{ci});
        try std.testing.expect(ok);
        for (c.get("drawn").?.array.items, out[0..rows]) |want, got| {
            try std.testing.expectEqual(try std.fmt.parseInt(u64, want.array.items[0].number_string, 10), got.token);
            try std.testing.expectEqual(try std.fmt.parseInt(u64, want.array.items[1].number_string, 10), @as(u64, @bitCast(got.share)));
        }
    }
}

test "union covers a nucleus that no individual shard covers, strict tie boundary requires fallback" {
    const gpa = std.testing.allocator;
    var drawn: [1]Drawn = undefined;
    var sh: Shares = .{ .world = 2, .rows = 1, .count = 2, .vals = &.{ 3, 1, 2, 0 }, .ids = &.{ 0, 1, 2, 3 }, .mass = &.{ 40, 10, 35, 5 }, .sums = &.{ 55, 45 }, .widths = &.{ 3, 3 } };
    const sampling: S.Sampling = .{ .seed = 1, .top_k = 0, .top_p = 0.7 };
    try std.testing.expect(!try drawShares(gpa, sh, &.{7}, sampling, &drawn));
    try std.testing.expect(try drawSharesWithUnion(gpa, sh, &.{7}, sampling, &drawn, true));
    sh.vals = &.{ 3, 2, 2, 0 };
    try std.testing.expect(!try drawSharesWithUnion(gpa, sh, &.{7}, sampling, &drawn, true));
}
