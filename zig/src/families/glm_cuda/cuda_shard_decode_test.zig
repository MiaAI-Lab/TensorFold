//! cuda_shard_decode.zig's host tests (no GPU): the sharded passes' Triton launches against the captured variants at
//! every window size, the class check, the exchange plans of two ranks, and a two-rank host model of owner decode
//! against each rank's unsharded attention: bit for bit, captured and uncaptured exchanges, the side stream on and
//! off, no race between the streams, no read or write off a buffer. The model also kills every cuda_shard_decode
//! Mutant. The Triton kernels are modelled (`exec`), not emulated: a row's output is a function of its queries and of
//! exactly the key rows the real kernel reads, so a wrong pointer, offset, block, count or order changes the bits.
const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;
const dec = @import("cuda_shard_decode.zig");
const shard = @import("cuda_shard.zig");
const seg = @import("cuda_segments.zig");
const mla = @import("cuda_mla.zig");
const fwd = @import("cuda_forward.zig");
const tri_mod = @import("cuda_triton.zig");
const Tri = tri_mod.Tri;
const Recorder = tri_mod.Recorder;
const G = @import("cuda_geometry.zig").Geometry;

const gpa = std.testing.allocator;

// -- the fake device: run's device calls, logged with the recorders' counts at the time ---------------------------------

const Op = union(enum) {
    counts: struct { in: u64, own: u64, out: u64, rows: usize },
    fork,
    exchange: dec.Plan,
    join,
    select: struct { recv: u64, out: u64, own: u64, rows: usize },
};
const Entry = struct { op: Op, main: usize, side: usize };

const Log = struct {
    main: *Recorder,
    side: *Recorder,
    rank: usize = 0,
    fixed: bool = true,
    roce: bool = true,
    lists: shard.Lists = .{},
    n: usize = 0,
    at: [16]Entry = undefined,

    fn add(l: *Log, op: Op) !void {
        if (l.n == l.at.len) return error.TooManyOps;
        l.at[l.n] = .{ .op = op, .main = l.main.n, .side = l.side.n };
        l.n += 1;
    }
    pub fn counts(l: *Log, _: cuda.Stream, in: u64, own: u64, out: u64, rows: usize) !void {
        try l.add(.{ .counts = .{ .in = in, .own = own, .out = out, .rows = rows } });
    }
    pub fn select(l: *Log, _: cuda.Stream, recv: u64, out: u64, own: u64, rows: usize) !void {
        try l.add(.{ .select = .{ .recv = recv, .out = out, .own = own, .rows = rows } });
    }
    pub fn fork(l: *Log, _: Tri) !Tri {
        try l.add(.fork);
        return .{ .rec = l.side };
    }
    pub fn join(l: *Log, _: Tri) !void {
        try l.add(.join);
    }
    pub fn exchange(l: *Log, x: shard.Exchange, send: u64, recv: u64, rows: usize, _: cuda.Stream) !void {
        try l.add(.{ .exchange = dec.plan(x, send, recv, rows, 1 - l.rank, l.fixed, l.roce, &l.lists) });
    }
};

fn arg(s: *const Recorder.Saved, name: []const u8) aot.Arg {
    for (s.args[0..s.nargs]) |a| if (std.mem.eql(u8, a.name, name)) return a;
    std.debug.panic("{s} has no argument {s}", .{ s.name, name });
}
fn ptrArg(s: *const Recorder.Saved, name: []const u8) u64 {
    return arg(s, name).value.ptr.addr;
}
fn constArg(s: *const Recorder.Saved, name: []const u8) i64 {
    for (s.consts[0..s.nconsts]) |c| if (std.mem.eql(u8, c.name, name)) return c.int.?;
    std.debug.panic("{s} has no constexpr {s}", .{ s.name, name });
}

/// The first captured variant a recorded launch would run (aot.matches, the engine's own choice).
fn variant(ks: []const aot.KernelJson, s: *const Recorder.Saved) ?usize {
    for (ks, 0..) |k, i| {
        if (std.mem.eql(u8, k.@"fn", s.name) and aot.matches(k, s.args[0..s.nargs], s.consts[0..s.nconsts])) return i;
    }
    return null;
}

/// Pointers an arena carves for `sizes` from `base` (state.Arena.take: 256-byte steps).
fn carve(base: u64, sizes: []const usize, out: []u64) void {
    var at = base;
    for (sizes, out) |n, *p| {
        p.* = at;
        at += std.mem.alignForward(usize, n, 256);
    }
}

fn hbOf(rows: usize) usize {
    return if (rows >= 8) 32 else 16;
}

test "owner decode's launches find the variant of their unsharded launch at every window size" {
    const parsed = try std.json.parseFromSlice(aot.SetJson, gpa, @embedFile("fixtures_aot_variants.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const ks = parsed.value.kernels;
    // the engine's window (cuda_engine: 64 rows, 8 segments) on its device tables, Scratch and the exchange buffers as
    // their allocations carve them
    const sh = try shard.Shard.init(1, 2, 8 * 2048, shard.trash);
    var tables = try seg.SegRows.initShard(gpa, 64, 8, 2112, sh);
    defer tables.deinit();
    tables.dev = .{ .d = undefined, .ptr = 0x7f00_0000_0000, .len = tables.host.len * 4, .borrowed = true };
    const sizes = fwd.Scratch.layout(try G.init(2, 1), 64, 1 << 20);
    var s: [45]u64 = undefined;
    carve(0x7e00_0000_0000, &sizes, &s);
    var b: [7]u64 = undefined;
    carve(0x7d00_0000_0000, &dec.layout(64), &b);
    const p: dec.Ptrs = .{ .qall = b[0], .olp = b[1], .orecv = b[2], .ce = b[3], .po = b[4], .pm = b[5], .pl = b[6] };
    try std.testing.expectEqual(@as(?[]const u8, null), try dec.offClass(&tables, p));
    const lc: u64 = 0x7c00_0000_0000;
    const pos = try tables.ptr(0);
    const rec = try gpa.create(Recorder);
    defer gpa.destroy(rec);
    const side = try gpa.create(Recorder);
    defer gpa.destroy(side);
    const plain = try gpa.create(Recorder);
    defer gpa.destroy(plain);
    for (1..65) |r| {
        for ([_]bool{ true, false }) |overlap| {
            rec.* = .{};
            side.* = .{};
            plain.* = .{};
            const t: Tri = .{ .rec = rec };
            try dec.latWrite(t, &tables, r, s[15], lc);
            var log: Log = .{ .main = rec, .side = side, .rank = 1 };
            try dec.run(&log, t, .{ .rows = r, .rank = 1, .hb = hbOf(r), .qa = s[20], .lc = lc, .pos = pos, .lbase = try dec.plane(&tables, .lbase), .sparse = try dec.plane(&tables, .sparse_eff), .own = try dec.plane(&tables, .own), .tokens = s[27], .counts = s[28], .out = s[21], .po = s[23], .pm = s[24], .pl = s[25], .overlap = overlap }, p);
            // cuda_forward.dsaMtp's unsharded launches of the same window
            const u: Tri = .{ .rec = plain };
            const hb = hbOf(r);
            const ri: i32 = @intCast(r);
            try u.seg_lat_write(.{ r, 1, 1 }, s[15], 512, lc, lc, pos, try tables.ptr(1), .{ .LW = 512, .RS = 528, .FP8 = 1 });
            try u.seg_chunks(.{ r, (32 + hb - 1) / hb, 5 }, s[20], lc, lc, pos, try tables.ptr(1), try tables.ptr(3), s[27], s[28], s[23], s[24], s[25], ri, .{ .W = 2051, .H = 32, .LW = 512, .CH = 512, .SCALE = 0.0625, .HBT = @intCast(hb), .KTT = 32, .RS = 528, .FP8 = 1 });
            try u.merge(.{ r, 32, 1 }, s[23], s[24], s[25], s[21], s[28], ri, .{ .H = 32, .LW = 512, .NCH = 5, .SPARSE = 0 });
            try std.testing.expectEqual(@as(usize, if (overlap) 3 else 5), rec.n);
            try std.testing.expectEqual(@as(usize, if (overlap) 2 else 0), side.n);
            for ([_]*const Recorder{ rec, side }) |which| for (which.launches[0..which.n]) |*l| {
                const twin = for (plain.launches[0..plain.n]) |*x| {
                    if (std.mem.eql(u8, x.name, l.name)) break x;
                } else unreachable;
                const want = variant(ks, twin) orelse return error.UnshardedHasNoVariant;
                const got = variant(ks, l) orelse {
                    std.debug.print("{s} at {d} rows: no captured variant\n", .{ l.name, r });
                    return error.MissingTritonVariant;
                };
                try std.testing.expectEqual(want, got);
                try std.testing.expectEqual(twin.grid, l.grid);
            };
        }
    }
}

test "a decode table or buffer off the kernels' 16-byte class is named (Decode.init refuses it before any capture)" {
    const sh = try shard.Shard.init(0, 2, 8 * 2048, shard.trash);
    const p: dec.Ptrs = .{ .qall = 0x1000, .olp = 0x2000, .orecv = 0x3000, .ce = 0x4000, .po = 0x5000, .pm = 0x6000, .pl = 0x7000 };
    // the engine's window: every glue plane on a 16-byte boundary
    var ok = try seg.SegRows.initShard(gpa, 64, 8, 2112, sh);
    defer ok.deinit();
    ok.dev = .{ .d = undefined, .ptr = 0x10000, .len = ok.host.len * 4, .borrowed = true };
    try std.testing.expectEqual(@as(?[]const u8, null), try dec.offClass(&ok, p));
    // six rows: lbase 8 bytes off the class the captured launches have (5 * 6 + 4 * 4 words before it)
    var odd = try seg.SegRows.initShard(gpa, 6, 4, 64, sh);
    defer odd.deinit();
    odd.dev = .{ .d = undefined, .ptr = 0x10000, .len = odd.host.len * 4, .borrowed = true };
    try std.testing.expectEqualStrings("lbase", (try dec.offClass(&odd, p)).?);
    var q = p;
    q.ce += 4;
    try std.testing.expectEqualStrings("ce", (try dec.offClass(&ok, q)).?);
    // an unsharded table has no glue planes
    var plain = try seg.SegRows.init(gpa, 64, 8, 2112);
    defer plain.deinit();
    plain.dev = .{ .d = undefined, .ptr = 0x10000, .len = plain.host.len * 4, .borrowed = true };
    try std.testing.expectError(error.NoGluePlanes, dec.plane(&plain, .lbase));
}

test "exchanges: a fixed block a rank when captured, RoCE up to its slot, else the crossing rows, paired across ranks" {
    var ls: shard.Lists = .{};
    // captured: every row of the window, the transport by the block's size (16 rows fill the default 512 KiB slot)
    try std.testing.expectEqual(@as(usize, 512 << 10), 16 * dec.row_bytes);
    const g = dec.plan(.q, 0x1000, 0x100000, 16, 1, true, true, &ls);
    try std.testing.expectEqual(dec.Way.gather, g.way);
    try std.testing.expectEqual(@as(usize, 16 * 32 * 512), g.count);
    try std.testing.expectEqual(@as(u64, 0x100000), g.recv);
    const w = dec.plan(.out, 0x1000, 0x100000, 17, 0, true, false, &ls);
    try std.testing.expectEqual(dec.Way.swap, w.way);
    try std.testing.expectEqual(@as(u64, 0x100000), w.recv); // the peer's block of rank 0 is the first
    try std.testing.expectEqual(@as(u64, 0x100000 + 17 * dec.row_bytes), dec.plan(.out, 0x1000, 0x100000, 17, 1, true, false, &ls).recv);
    // uncaptured: the crossing rows, as runs of whole streams; rank r's sends meet rank 1 - r's receives
    var prng = std.Random.DefaultPrng.init(118);
    const rnd = prng.random();
    for (0..500) |_| {
        const rows = rnd.intRangeAtMost(usize, 1, 64);
        var orank: [64]i32 = undefined;
        var r: usize = 0;
        var streams: usize = 0;
        while (r < rows) : (streams += 1) {
            const n = @min(rows - r, rnd.intRangeAtMost(usize, 1, 16));
            const o: i32 = rnd.intRangeAtMost(i32, 0, 1);
            @memset(orank[r..][0..n], o);
            r += n;
        }
        var lists: [2]shard.Lists = undefined;
        for (0..2) |k| lists[k] = try shard.lists(orank[0..rows], k);
        for ([_]shard.Exchange{ .q, .out }) |x| {
            var sent: usize = 0;
            for (0..2) |k| {
                const mine = dec.plan(x, 0x1000 + k * 0x1000000, 0x8000000 + k * 0x1000000, rows, 1 - k, false, rnd.boolean(), &lists[k]);
                const theirs = dec.plan(x, 0x1000 + (1 - k) * 0x1000000, 0x8000000 + (1 - k) * 0x1000000, rows, k, false, rnd.boolean(), &lists[1 - k]);
                try std.testing.expectEqual(dec.Way.rows, mine.way);
                try std.testing.expect(mine.out.n <= streams);
                try std.testing.expectEqual(mine.out.n, theirs.in.n);
                for (mine.out.slice(), theirs.in.slice()) |o, i| {
                    try std.testing.expectEqual(o[1], i[1]);
                    // a run leaves from its rows of `send` and lands at the same rows of the peer's block of `recv`
                    try std.testing.expectEqual(o[0] - (0x1000 + k * 0x1000000), i[0] - dec.block(0x8000000 + (1 - k) * 0x1000000, k, rows));
                    sent += o[1];
                }
                // what a rank sends: the queries of the rows the peer owns, the peer's heads of its own
                for (mine.out.slice()) |o| {
                    const first = (o[0] - (0x1000 + k * 0x1000000)) / dec.row_bytes;
                    for (first..first + o[1] / dec.row_bytes) |row| try std.testing.expectEqual(x == .out, orank[row] == @as(i32, @intCast(k)));
                }
            }
            // every row crosses once an exchange, one way or the other
            try std.testing.expectEqual(rows * dec.row_bytes, sent);
        }
    }
}

// -- the two-rank model ---------------------------------------------------------------------------------------------

/// The model's memory: named host buffers, a pointer valid only inside one of them.
const Mem = struct {
    names: [64][]const u8 = undefined,
    spans: [64][]u8 = undefined,
    n: usize = 0,
    fn add(m: *Mem, name: []const u8, b: []u8) u64 {
        m.names[m.n] = name;
        m.spans[m.n] = b;
        m.n += 1;
        return @intFromPtr(b.ptr);
    }
    fn of(m: *const Mem, addr: u64) ?usize {
        for (m.spans[0..m.n], 0..) |s, i| if (addr >= @intFromPtr(s.ptr) and addr < @intFromPtr(s.ptr) + s.len) return i;
        return null;
    }
    fn at(m: *const Mem, addr: u64, len: usize) ![]u8 {
        const i = m.of(addr) orelse return error.OffBuffer;
        const s = m.spans[i];
        const off = addr - @intFromPtr(s.ptr);
        if (off + len > s.len) return error.OffBuffer;
        return s[off..][0..len];
    }
    fn word(m: *const Mem, addr: u64) !i32 {
        return std.mem.readInt(i32, (try m.at(addr, 4))[0..4], .little);
    }
};

fn toBf16(x: f32) u16 {
    if (std.math.isNan(x)) return 0x7fc0;
    const b: u32 = @bitCast(x);
    return @truncate((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}
fn fromBf16(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// The modelled Triton kernels: `_seg_lat_write` (a row's latent quantized into LC at BASE + POS), `_seg_chunks` with
/// `_merge` (a row's output from its queries and the key rows it reads: BASE + its tokens' positions when SPR, else
/// BASE + 0 .. POS; NaN with none), each address checked against the buffers.
fn exec(m: *const Mem, s: *const Recorder.Saved) !void {
    const rows: usize = s.grid[0];
    if (std.mem.eql(u8, s.name, "_seg_lat_write")) {
        const stride: usize = @intCast(arg(s, "lat_stride").value.i32);
        for (0..rows) |r| {
            const row: usize = @intCast(try m.word(ptrArg(s, "BASE") + 4 * r) + try m.word(ptrArg(s, "POS") + 4 * r));
            var x: [512]f32 = undefined;
            const lat = try m.at(ptrArg(s, "LAT") + 2 * r * stride, 1024);
            for (&x, 0..) |*v, j| v.* = fromBf16(std.mem.readInt(u16, lat[2 * j ..][0..2], .little));
            mla.quantizeRow(&x, try m.at(ptrArg(s, "LC") + row * 528, 528));
        }
    } else if (std.mem.eql(u8, s.name, "_seg_chunks")) {
        const w: usize = @intCast(constArg(s, "W"));
        for (0..rows) |r| {
            const spr = try m.word(ptrArg(s, "SPR") + 4 * r);
            const base: usize = @intCast(try m.word(ptrArg(s, "BASE") + 4 * r));
            const n: usize = @intCast(if (spr != 0) try m.word(ptrArg(s, "CNT") + 4 * r) else try m.word(ptrArg(s, "POS") + 4 * r) + 1);
            var acc: [512]f32 = @splat(0);
            for (0..n) |i| {
                const tok: usize = @intCast(if (spr != 0) try m.word(ptrArg(s, "TOK") + 4 * (r * w + i)) else @as(i32, @intCast(i)));
                var k: [512]f32 = undefined;
                mla.dequantRow(try m.at(ptrArg(s, "LC") + (base + tok) * 528, 528), &k);
                for (&acc, k) |*a, v| a.* += v;
            }
            for (0..32) |h| {
                const q = try m.at(ptrArg(s, "QA") + 2 * (r * 32 + h) * 512, 1024);
                const o = try m.at(ptrArg(s, "PO") + 4 * (r * 32 + h) * 512, 2048);
                for (0..512) |j| {
                    const v: f32 = if (n == 0) std.math.nan(f32) else fromBf16(std.mem.readInt(u16, q[2 * j ..][0..2], .little)) * (1 + acc[j]);
                    std.mem.writeInt(u32, o[4 * j ..][0..4], @bitCast(v), .little);
                }
            }
            _ = try m.at(ptrArg(s, "PM") + 4 * r * 32, 4 * 32);
            _ = try m.at(ptrArg(s, "PL") + 4 * r * 32, 4 * 32);
        }
    } else if (std.mem.eql(u8, s.name, "_merge")) {
        for (0..rows) |r| {
            const from = try m.at(ptrArg(s, "PO") + 4 * r * 32 * 512, 4 * 32 * 512);
            const to = try m.at(ptrArg(s, "OUT") + 2 * r * 32 * 512, 2 * 32 * 512);
            for (0..32 * 512) |j| std.mem.writeInt(u16, to[2 * j ..][0..2], toBf16(@bitCast(std.mem.readInt(u32, from[4 * j ..][0..4], .little))), .little);
        }
    } else return error.UnmodelledKernel;
}

/// The buffers a launch or a device call reads and writes (by Mem index), for the race check.
const Access = struct {
    reads: std.StaticBitSet(64) = .empty,
    writes: std.StaticBitSet(64) = .empty,
    fn add(a: *Access, m: *const Mem, addr: u64, write: bool) void {
        const i = m.of(addr) orelse return;
        if (write) a.writes.set(i) else a.reads.set(i);
    }
    fn launch(a: *Access, m: *const Mem, s: *const Recorder.Saved) void {
        const lat = std.mem.eql(u8, s.name, "_seg_lat_write");
        for (s.args[0..s.nargs]) |x| if (x.value == .ptr) {
            const out = for ([_][]const u8{ "PO", "PM", "PL", "OUT" }) |o| {
                if (std.mem.eql(u8, o, x.name)) break true;
            } else lat and std.mem.eql(u8, x.name, "LC");
            a.add(m, x.value.ptr.addr, out);
        };
    }
    fn op(a: *Access, m: *const Mem, o: *const Op) void {
        switch (o.*) {
            .counts => |c| {
                a.add(m, c.in, false);
                a.add(m, c.own, false);
                a.add(m, c.out, true);
            },
            .exchange => |p| {
                a.add(m, p.send, false);
                a.add(m, p.recv, true);
            },
            .select => |c| {
                a.add(m, c.recv, false);
                a.add(m, c.own, false);
                a.add(m, c.out, true);
            },
            .fork, .join => {},
        }
    }
};

/// Between the fork and the join, nothing the side stream writes is touched on the main stream and nothing the main
/// stream writes is read on the side.
fn races(m: *const Mem, l: *const Log) !void {
    var f: ?usize = null;
    var j: ?usize = null;
    for (l.at[0..l.n], 0..) |e, i| switch (e.op) {
        .fork => f = i,
        .join => j = i,
        else => {},
    };
    const fi = f orelse return;
    const ji = j orelse return error.NeverJoined;
    var side: Access = .{};
    var main: Access = .{};
    for (l.side.launches[l.at[fi].side..l.at[ji].side]) |*s| side.launch(m, s);
    for (l.main.launches[l.at[fi].main..l.at[ji].main]) |*s| main.launch(m, s);
    for (l.at[fi + 1 .. ji]) |*e| main.op(m, &e.op);
    var touched = main.reads;
    touched.setUnion(main.writes);
    if (side.writes.intersectWith(touched).count() != 0 or main.writes.intersectWith(side.reads).count() != 0) return error.StreamsRace;
}

/// One rank's progress through its log: the launches before each device call, then the call.
const Cursor = struct {
    l: *const Log,
    i: usize = 0,
    mi: usize = 0,
    si: usize = 0,
    fn launches(c: *Cursor, m: *const Mem, main: usize, side: usize) !void {
        while (c.mi < main) : (c.mi += 1) try exec(m, &c.l.main.launches[c.mi]);
        while (c.si < side) : (c.si += 1) try exec(m, &c.l.side.launches[c.si]);
    }
    /// Up to the next exchange (returned) or the end (null).
    fn next(c: *Cursor, m: *const Mem) !?*const dec.Plan {
        while (c.i < c.l.n) {
            const e = &c.l.at[c.i];
            try c.launches(m, e.main, e.side);
            c.i += 1;
            switch (e.op) {
                .exchange => |*p| return p,
                .counts => |x| dec.countsRef(ints(try m.at(x.in, 4 * x.rows)), ints(try m.at(x.own, 4 * x.rows)), ints(try m.at(x.out, 4 * x.rows))),
                .select => |x| dec.selectRef(try m.at(x.recv, x.rows * dec.row_bytes), try m.at(x.out, x.rows * dec.row_bytes), ints(try m.at(x.own, 4 * x.rows)), dec.row_bytes),
                .fork, .join => {},
            }
        }
        try c.launches(m, c.l.main.n, c.l.side.n);
        return null;
    }
};

fn ints(b: []u8) []i32 {
    return @alignCast(std.mem.bytesAsSlice(i32, b));
}

/// Both ranks' halves of one exchange, as the transport moves them (the peer's bytes from before the exchange).
fn swap(m: *const Mem, p: [2]*const dec.Plan) !void {
    if (p[0].way != p[1].way or p[0].count != p[1].count) return error.ExchangeMismatch;
    const n = p[0].count * 2;
    for (0..2) |k| {
        const mine = p[k];
        const theirs = p[1 - k];
        switch (mine.way) {
            .gather => {
                @memcpy(try m.at(mine.recv + (1 - k) * n, n), try m.at(theirs.send, n));
                @memcpy(try m.at(mine.recv + k * n, n), try m.at(mine.send, n));
            },
            .swap => @memcpy(try m.at(mine.recv, n), try m.at(theirs.send, n)),
            .rows => {
                if (mine.in.n != theirs.out.n) return error.ExchangeMismatch;
                for (mine.in.slice(), theirs.out.slice()) |i, o| {
                    if (i[1] != o[1]) return error.ExchangeMismatch;
                    @memcpy(try m.at(i[0], i[1]), try m.at(o[0], o[1]));
                }
            },
        }
    }
}

/// Both ranks' logs in step: each up to its next exchange, the exchange, on to the end.
fn lockstep(m: *const Mem, logs: [2]*const Log) !void {
    var c: [2]Cursor = .{ .{ .l = logs[0] }, .{ .l = logs[1] } };
    while (true) {
        const a = try c[0].next(m);
        const b = try c[1].next(m);
        if (a == null and b == null) return;
        if (a == null or b == null) return error.ExchangeMismatch;
        try swap(m, .{ a.?, b.? });
    }
}

/// One rank's buffers.
const Rank = struct {
    tables: seg.SegRows,
    full: []u8, // the unsharded latent plane (every pool row)
    local: []u8, // the sharded one: its part, then the trash rows
    bufs: [16][]u8 = undefined,
    nb: usize = 0,
    fn buf(r: *Rank, m: *Mem, name: []const u8, len: usize) !u64 {
        const b = try gpa.alignedAlloc(u8, .fromByteUnits(256), len);
        @memset(b, 0);
        r.bufs[r.nb] = b;
        r.nb += 1;
        return m.add(name, b);
    }
    fn deinit(r: *Rank) void {
        for (r.bufs[0..r.nb]) |b| gpa.free(@as([]align(256) u8, @alignCast(b)));
        gpa.free(r.full);
        gpa.free(r.local);
        r.tables.deinit();
    }
};

const pool_rows = 8 * 2048;
const sparse_from = 32;

/// One random window on both ranks: unsharded attention per rank, then owner decode; true when each rank's heads of
/// every row and its latent rows are the unsharded ones', with no race and nothing off a buffer.
fn trial(rnd: std.Random, fixed: bool, overlap: bool) !bool {
    var m: Mem = .{};
    // streams in distinct extents, either part; rows dense (positions under `sparse_from`) or sparse
    const nseg = rnd.intRangeAtMost(usize, 1, 4);
    var slots: [8]usize = .{ 0, 1, 2, 3, 4, 5, 6, 7 };
    rnd.shuffle(usize, &slots);
    var segs: [4]seg.Segment = undefined;
    var rows: usize = 0;
    for (segs[0..nseg], slots[0..nseg]) |*s, slot| {
        const n = rnd.intRangeAtMost(usize, 1, 4);
        const pos = if (rnd.boolean()) rnd.intRangeAtMost(usize, 0, sparse_from + 2) else rnd.intRangeAtMost(usize, sparse_from + 8, 1900);
        s.* = .{ .base = slot * 2048, .pos = pos, .rows = n, .ring_base = slot * 64 };
        rows += n;
    }
    var rk: [2]Rank = undefined;
    var made: usize = 0;
    defer for (rk[0..made]) |*r| r.deinit();
    for (0..2) |k| {
        const sh = try shard.Shard.init(k, 2, pool_rows, shard.trash);
        rk[k] = .{ .tables = try seg.SegRows.initShard(gpa, 64, 8, 64, sh), .full = try gpa.alloc(u8, pool_rows * 528), .local = try gpa.alloc(u8, (sh.part + sh.trash) * 528) };
        made += 1;
        _ = try rk[k].tables.fill(segs[0..nseg], sparse_from);
        rk[k].tables.dev = .{ .d = undefined, .ptr = m.add("tables", std.mem.sliceAsBytes(rk[k].tables.host)), .len = rk[k].tables.host.len * 4, .borrowed = true };
        @memset(rk[k].full, 0);
        _ = m.add("full", rk[k].full);
        _ = m.add("local", rk[k].local);
    }
    // the history: latent rows before each stream's position, alike on both ranks (and on the owner's part)
    var x: [512]f32 = undefined;
    for (segs[0..nseg]) |s| for (0..s.pos) |t| {
        for (&x) |*v| v.* = mla.bf16(rnd.float(f32) * 4 - 2);
        mla.quantizeRow(&x, rk[0].full[(s.base + t) * 528 ..][0..528]);
    };
    @memcpy(rk[1].full, rk[0].full);
    for (0..2) |k| {
        const part = pool_rows / 2;
        @memcpy(rk[k].local[0 .. part * 528], rk[k].full[k * part * 528 ..][0 .. part * 528]);
        rnd.bytes(rk[k].local[part * 528 ..]); // the trash rows: anything
    }
    // the window's inputs: the latent rows and the selection replicated, the queries each rank's own heads
    const lat_len = rows * 512 * 2;
    var lat_bytes: [16 * 1024]u8 = undefined;
    for (0..rows * 512) |j| std.mem.writeInt(u16, lat_bytes[2 * j ..][0..2], toBf16(rnd.float(f32) * 4 - 2), .little);
    var tokens: [16 * 2051]i32 = @splat(0);
    var counts: [64]i32 = @splat(0);
    {
        var r: usize = 0;
        for (segs[0..nseg]) |s| for (0..s.rows) |i| {
            const q = s.pos + i;
            if (q >= sparse_from) {
                counts[r] = rnd.intRangeAtMost(i32, 1, 12);
                for (0..@intCast(counts[r])) |j| tokens[r * 2051 + j] = rnd.intRangeLessThan(i32, 0, @intCast(q));
            } else {
                counts[r] = rnd.intRangeAtMost(i32, 0, 4); // read by nothing
                for (0..8) |j| tokens[r * 2051 + j] = rnd.intRangeLessThan(i32, 0, 2000);
            }
            r += 1;
        };
    }
    const row = dec.row_bytes;
    var ptr: [2]struct { lat: u64, qa: u64, ol: u64, ref: u64, po: u64, pm: u64, pl: u64, tok: u64, cnt: u64, p: dec.Ptrs } = undefined;
    for (0..2) |k| {
        const r = &rk[k];
        ptr[k].lat = try r.buf(&m, "lat", lat_len);
        @memcpy(try m.at(ptr[k].lat, lat_len), lat_bytes[0..lat_len]);
        ptr[k].qa = try r.buf(&m, "qa", rows * row);
        const qa = try m.at(ptr[k].qa, rows * row);
        for (0..rows * 32 * 512) |j| std.mem.writeInt(u16, qa[2 * j ..][0..2], toBf16(rnd.float(f32) * 2 - 1), .little);
        ptr[k].ol = try r.buf(&m, "ol", rows * row);
        ptr[k].ref = try r.buf(&m, "ref", rows * row);
        ptr[k].po = try r.buf(&m, "po", 5 * rows * 32 * 512 * 4);
        ptr[k].pm = try r.buf(&m, "pm", 5 * rows * 32 * 4);
        ptr[k].pl = try r.buf(&m, "pl", 5 * rows * 32 * 4);
        ptr[k].tok = try r.buf(&m, "tokens", rows * 2051 * 4);
        @memcpy(try m.at(ptr[k].tok, rows * 2051 * 4), std.mem.sliceAsBytes(tokens[0 .. rows * 2051]));
        ptr[k].cnt = try r.buf(&m, "counts", 64 * 4);
        @memcpy(try m.at(ptr[k].cnt, 64 * 4), std.mem.sliceAsBytes(&counts));
        const sz = dec.layout(rows);
        ptr[k].p = .{ .qall = try r.buf(&m, "qall", sz[0]), .olp = try r.buf(&m, "olp", sz[1]), .orecv = try r.buf(&m, "orecv", sz[2]), .ce = try r.buf(&m, "ce", sz[3]), .po = try r.buf(&m, "po2", sz[4]), .pm = try r.buf(&m, "pm2", sz[5]), .pl = try r.buf(&m, "pl2", sz[6]) };
    }
    const recs = try gpa.alloc(Recorder, 6);
    defer gpa.free(recs);
    for (recs) |*r| r.* = .{};
    const hb = hbOf(rows);
    const ri: i32 = @intCast(rows);
    // each rank unsharded (cuda_forward.dsaMtp): its heads over the whole plane
    for (0..2) |k| {
        const t: Tri = .{ .rec = &recs[4 + k] };
        const tb = &rk[k].tables;
        const lc: u64 = @intFromPtr(rk[k].full.ptr);
        try t.seg_lat_write(.{ rows, 1, 1 }, ptr[k].lat, 512, lc, lc, try tb.ptr(0), try tb.ptr(1), .{ .LW = 512, .RS = 528, .FP8 = 1 });
        try t.seg_chunks(.{ rows, (32 + hb - 1) / hb, 5 }, ptr[k].qa, lc, lc, try tb.ptr(0), try tb.ptr(1), try tb.ptr(3), ptr[k].tok, ptr[k].cnt, ptr[k].po, ptr[k].pm, ptr[k].pl, ri, .{ .W = 2051, .H = 32, .LW = 512, .CH = 512, .SCALE = 0.0625, .HBT = @intCast(hb), .KTT = 32, .RS = 528, .FP8 = 1 });
        try t.merge(.{ rows, 32, 1 }, ptr[k].po, ptr[k].pm, ptr[k].pl, ptr[k].ref, ptr[k].cnt, ri, .{ .H = 32, .LW = 512, .NCH = 5, .SPARSE = 0 });
        for (recs[4 + k].launches[0..recs[4 + k].n]) |*s| try exec(&m, s);
    }
    // each rank sharded: the latent write through the glue, then owner decode
    var logs: [2]Log = undefined;
    const roce = rnd.boolean(); // the ranks' RoCE choice is the size's, alike on both
    for (0..2) |k| {
        const tb = &rk[k].tables;
        logs[k] = .{ .main = &recs[2 * k], .side = &recs[2 * k + 1], .rank = k, .fixed = fixed, .roce = roce, .lists = try shard.lists(tb.plane(9 + @as(usize, @backingInt(shard.Glue.orank)))[0..rows], k) };
        const t: Tri = .{ .rec = &recs[2 * k] };
        const lc: u64 = @intFromPtr(rk[k].local.ptr);
        try dec.latWrite(t, tb, rows, ptr[k].lat, lc);
        try dec.run(&logs[k], t, .{ .rows = rows, .rank = k, .hb = hb, .qa = ptr[k].qa, .lc = lc, .pos = try tb.ptr(0), .lbase = try dec.plane(tb, .lbase), .sparse = try dec.plane(tb, .sparse_eff), .own = try dec.plane(tb, .own), .tokens = ptr[k].tok, .counts = ptr[k].cnt, .out = ptr[k].ol, .po = ptr[k].po, .pm = ptr[k].pm, .pl = ptr[k].pl, .overlap = overlap }, ptr[k].p);
    }
    for (&logs) |*l| races(&m, l) catch return false;
    lockstep(&m, .{ &logs[0], &logs[1] }) catch |e| switch (e) {
        error.OffBuffer, error.ExchangeMismatch => return false,
        else => return e,
    };
    // each rank's heads of every row, bit for bit; the latent rows where the glue puts them
    for (0..2) |k| {
        if (!std.mem.eql(u8, try m.at(ptr[k].ol, rows * row), try m.at(ptr[k].ref, rows * row))) return false;
        const sh = rk[k].tables.shard.?;
        var r: usize = 0;
        for (segs[0..nseg]) |s| for (0..s.rows) |i| {
            const want = rk[k].full[(s.base + s.pos + i) * 528 ..][0..528];
            const at = if (sh.owns(s.base)) sh.local(s.base) + s.pos + i else sh.trashRow(r);
            if (!std.mem.eql(u8, rk[k].local[at * 528 ..][0..528], want)) return false;
            r += 1;
        };
    }
    return true;
}

test "two ranks' owner decode gives each rank its unsharded heads, captured or not, side stream on or off" {
    var prng = std.Random.DefaultPrng.init(1180);
    const rnd = prng.random();
    for (0..24) |i| {
        if (!try trial(rnd, i % 2 == 0, i % 4 < 2)) {
            std.debug.print("owner decode trial {d} differs from the unsharded attention\n", .{i});
            return error.TestUnexpectedResult;
        }
    }
}

test "every owner decode mutant is caught by the two-rank model" {
    defer dec.mutant = .none;
    var prng = std.Random.DefaultPrng.init(1181);
    const rnd = prng.random();
    for (std.enums.values(dec.Mutant)) |mu| {
        if (mu == .none) continue;
        dec.mutant = mu;
        var caught = false;
        for (0..12) |i| {
            // the select-before-join race needs the side stream; the others show in either form
            if (!try trial(rnd, i % 2 == 0, mu == .select_before_join or i % 4 < 2)) {
                caught = true;
                break;
            }
        }
        if (!caught) {
            std.debug.print("owner decode mutant {t} survived\n", .{mu});
            return error.MutantSurvived;
        }
    }
}
