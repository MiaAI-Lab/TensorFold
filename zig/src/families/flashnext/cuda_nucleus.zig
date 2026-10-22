//! Flash Next's top_k-off draws (Python tensorfold/cuda/sampling.nucleus_rows): each token's share of the row's mass
//! in fixed point (floor(exp(f64(logit) / t - top) * 2^40), int64, on the GPU with CUDA's fp64 exp as torch's
//! elementwise kernels run it: cuda_torch_ops.nucleusMass), every rank's top candidates (value, global id, mass) with
//! its shard's mass sum and width, and the keyed draw over the top_p nucleus then min_p on the host (`_draw`): the
//! nucleus cut by exact integer mass sums, the same on every rank. A rank whose nucleus runs past its candidates
//! sends its whole shard instead. tools/zig/check_flashnext_nucleus.py checks the mass kernel against torch and
//! records fixtures of Python's own _shares / _draw, which the test here replays.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const fwd = @import("cuda_forward.zig");

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
        // a rank's share must end inside its candidates
        for (0..sh.world) |k| {
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

/// Device scratch for one row: its masses, top and sum.
pub const Scratch = struct { mass: u64, top: u64, sum: u64 };

pub fn scratchBytes(columns: usize) [3]usize {
    return .{ columns * 8, 8, 8 };
}

fn bf16(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

/// nucleus_rows over `rows` rows of this rank's bf16 logits (`columns` wide at `logits`, ids = offset + column, or
/// `id_map[column]` for the draft head), row r keyed at positions[r]; tokens into `out`, their shares into
/// `shares` (the draw's temperature-1 probability for the draft chain). Every rank calls it alike.
pub fn sampleRows(f: *fwd.Forward, sc: Scratch, logits: u64, rows: usize, columns: usize, id_map: ?[]const u32, offset: i64, positions: []const u64, s: S.Sampling, out: []u32, shares: ?[]f64) !void {
    const gpa = f.gpa;
    const t = @max(s.temperature, 1e-6);
    const world: usize = if (f.comm) |cm| cm.world else 1;
    const row = try gpa.alloc(u16, columns);
    defer gpa.free(row);
    const scaled = try gpa.alloc(f64, columns);
    defer gpa.free(scaled);
    const mass = try gpa.alloc(i64, columns);
    defer gpa.free(mass);
    const order = try gpa.alloc(u32, columns);
    defer gpa.free(order);
    for (0..rows) |r| {
        try f.s.synchronize();
        const lg: cuda.DeviceBuffer = .{ .d = f.d, .ptr = logits + r * columns * 2, .len = columns * 2 };
        try lg.download(0, std.mem.sliceAsBytes(row));
        var local = -std.math.inf(f64);
        for (row, scaled) |x, *y| {
            y.* = @as(f64, bf16(x)) / t;
            local = @max(local, y.*);
        }
        // every rank's maximum (the global top) through the gather
        const top = try maxOverRanks(f, local);
        const topb: cuda.DeviceBuffer = .{ .d = f.d, .ptr = sc.top, .len = 8 };
        try topb.upload(0, std.mem.asBytes(&top));
        try f.th.nucleusMass(logits + r * columns * 2, columns, 1, columns, t, sc.top, sc.mass, sc.sum);
        try f.s.synchronize();
        const mb: cuda.DeviceBuffer = .{ .d = f.d, .ptr = sc.mass, .len = columns * 8 };
        try mb.download(0, std.mem.sliceAsBytes(mass));
        var sum: i64 = undefined;
        const sb: cuda.DeviceBuffer = .{ .d = f.d, .ptr = sc.sum, .len = 8 };
        try sb.download(0, std.mem.asBytes(&sum));
        // this rank's candidates by value (any order among equal values: _draw's rules do not depend on it)
        for (order, 0..) |*o, i| o.* = @intCast(i);
        const Ctx = struct {
            v: []const f64,
            fn lt(c: @This(), a: u32, b: u32) bool {
                return c.v[a] > c.v[b] or (c.v[a] == c.v[b] and a < b);
            }
        };
        std.mem.sort(u32, order, Ctx{ .v = scaled }, Ctx.lt);
        var count: usize = @min(nucleus, columns);
        var drawn: [1]Drawn = undefined;
        while (true) {
            var got = try exchange(f, gpa, world, count, order, scaled, mass, sum, columns, id_map, offset);
            defer got.deinit(gpa);
            const ok = try drawShares(gpa, got.shares, positions[r .. r + 1], s, &drawn);
            if (ok) break;
            if (count >= got.most) return error.NucleusUncovered;
            count = got.most; // whole shards
        }
        out[r] = @intCast(drawn[0].token);
        if (shares) |p| p[r] = drawn[0].share;
    }
}

fn maxOverRanks(f: *fwd.Forward, local: f64) !f64 {
    const cm = f.comm orelse return local;
    const words = try gatherWords(f, &.{@bitCast(local)}, cm.world);
    defer f.gpa.free(words);
    var top = -std.math.inf(f64);
    for (words) |w| top = @max(top, @as(f64, @bitCast(w)));
    return top;
}

/// Every rank's u64 words (the same count on each), rank 0's first, through the communicator.
fn gatherWords(f: *fwd.Forward, mine: []const u64, world: usize) ![]u64 {
    const cm = f.comm.?;
    var send = try cuda.DeviceBuffer.alloc(f.d, mine.len * 8);
    defer send.free();
    var recv = try cuda.DeviceBuffer.alloc(f.d, world * mine.len * 8);
    defer recv.free();
    try send.upload(0, std.mem.sliceAsBytes(mine));
    try cm.allGather(send.ptr, recv.ptr, mine.len, .u64, f.s.handle);
    try f.s.synchronize();
    const out = try f.gpa.alloc(u64, world * mine.len);
    errdefer f.gpa.free(out);
    try recv.download(0, std.mem.sliceAsBytes(out));
    return out;
}

const Gathered = struct {
    shares: Shares,
    most: usize,
    vals: []f64,
    ids: []i64,
    mass: []i64,
    sums: []i64,
    widths: []i64,

    fn deinit(g: *Gathered, gpa: Allocator) void {
        gpa.free(g.vals);
        gpa.free(g.ids);
        gpa.free(g.mass);
        gpa.free(g.sums);
        gpa.free(g.widths);
    }
};

/// _shares: this rank's top `count` (value, global id, mass), padded with (-inf, -1, 0), its mass sum and width,
/// every rank's gathered (one rank: its own).
fn exchange(f: *fwd.Forward, gpa: Allocator, world: usize, count: usize, order: []const u32, scaled: []const f64, mass: []const i64, sum: i64, columns: usize, id_map: ?[]const u32, offset: i64) !Gathered {
    const width = 3 * count + 2;
    const mine = try gpa.alloc(u64, width);
    defer gpa.free(mine);
    for (0..count) |j| {
        if (j < columns) {
            const c = order[j];
            mine[j] = @bitCast(scaled[c]);
            mine[count + j] = @bitCast(if (id_map) |m| @as(i64, m[c]) else @as(i64, c) + offset);
            mine[2 * count + j] = @bitCast(mass[c]);
        } else {
            mine[j] = @bitCast(-std.math.inf(f64));
            mine[count + j] = @bitCast(@as(i64, -1));
            mine[2 * count + j] = 0;
        }
    }
    mine[3 * count] = @bitCast(sum);
    mine[3 * count + 1] = columns;
    const words = if (world > 1) try gatherWords(f, mine, world) else try gpa.dupe(u64, mine);
    defer gpa.free(words);
    // [world][1][count] arrays from the gathered words (u64 -> f64 / i64)
    const vals = try gpa.alloc(f64, world * count);
    errdefer gpa.free(vals);
    const ids = try gpa.alloc(i64, world * count);
    errdefer gpa.free(ids);
    const ms = try gpa.alloc(i64, world * count);
    errdefer gpa.free(ms);
    const sums = try gpa.alloc(i64, world);
    errdefer gpa.free(sums);
    const widths = try gpa.alloc(i64, world);
    var most: usize = 0;
    for (0..world) |k| {
        const w = words[k * width ..][0..width];
        for (0..count) |j| {
            vals[k * count + j] = @bitCast(w[j]);
            ids[k * count + j] = @bitCast(w[count + j]);
            ms[k * count + j] = @bitCast(w[2 * count + j]);
        }
        sums[k] = @bitCast(w[3 * count]);
        widths[k] = @bitCast(w[3 * count + 1]);
        most = @max(most, @as(usize, @intCast(widths[k])));
    }
    return .{ .shares = .{ .world = world, .rows = 1, .count = count, .vals = vals, .ids = ids, .mass = ms, .sums = sums, .widths = widths }, .most = most, .vals = vals, .ids = ids, .mass = ms, .sums = sums, .widths = widths };
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
