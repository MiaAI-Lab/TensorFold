//! cuda_shard_prompt.zig's host tests (no GPU): the sharded prompt chunk's Triton launches against the captured
//! variants, the copy's sizing, a rank's side, and a two-rank host model of the history fetch against each rank's
//! unsharded attention, bit for bit: chunks with rows under the dense limit (the whole history at its positions) and
//! past it (the union of the positions the tokens name), the union in row blocks when it outgrows the copy, transfers
//! in pieces, held against a split's side stream or not, either rank the owner. The model also kills every
//! cuda_shard_prompt Mutant. The attention launches are cuda_prefill's own (ShardAttn on a recording engine); the
//! Triton kernels are modelled (`exec`): a row's output is a function of its queries and of exactly the latent rows
//! the real kernel reads, in the order it reads them, so a wrong pointer, offset, block, count, token or order changes
//! the bits.
const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;
const sp = @import("cuda_shard_prompt.zig");
const shard = @import("cuda_shard.zig");
const state = @import("cuda_state.zig");
const pre = @import("cuda_prefill.zig");
const grid = @import("cuda_prefill_grid.zig");
const tri_mod = @import("cuda_triton.zig");
const Tri = tri_mod.Tri;
const Recorder = tri_mod.Recorder;
const G = @import("cuda_geometry.zig").Geometry;

const gpa = std.testing.allocator;
const W = G.tokens;
const RS = G.latent_row;
/// A rank's heads at TP=2, and a row's absorbed queries or attention output (bf16 [32][512]).
const H = 32;
const qrow = H * 512 * 2;

fn arg(s: *const Recorder.Saved, name: []const u8) aot.Arg {
    for (s.args[0..s.nargs]) |a| if (std.mem.eql(u8, a.name, name)) return a;
    std.debug.panic("{s} has no argument {s}", .{ s.name, name });
}
fn ptrArg(s: *const Recorder.Saved, name: []const u8) u64 {
    return arg(s, name).value.ptr.addr;
}

/// The captured kernel set the engine replays (zig/tests/cuda/glm_p1/aot/aot.json, read from the repository root: the
/// test step's working directory). fixtures_aot_variants.json, which the decode tests use, holds too few of the prompt
/// kernels' variants.
const Captured = struct {
    text: []u8,
    parsed: std.json.Parsed(aot.SetJson),
    fn load() !Captured {
        const text = std.Io.Dir.cwd().readFileAlloc(std.testing.io, "zig/tests/cuda/glm_p1/aot/aot.json", gpa, .limited(1 << 24)) catch |e| switch (e) {
            error.FileNotFound => return error.SkipZigTest,
            else => return e,
        };
        errdefer gpa.free(text);
        return .{ .text = text, .parsed = try std.json.parseFromSlice(aot.SetJson, gpa, text, .{ .ignore_unknown_fields = true }) };
    }
    fn deinit(c: *Captured) void {
        c.parsed.deinit();
        gpa.free(c.text);
    }
};

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

/// The prompt scratch's pointers the attention launches take, and POS.
const Bufs = struct { lat: u64, qa: u64, ol: u64, po: u64, pm: u64, pl: u64, dummy: u64, tokens: u64, counts: u64, origins: u64, pos: u64 };

/// A prompt engine as far as the attention launches read it: `g`, `tri` (recording), `pos_dev` and the scratch
/// pointers they take (the rest undefined).
fn engine(rec: *Recorder, rank: usize, at: Bufs) !pre.Engine {
    var e: pre.Engine = undefined;
    e.g = try G.init(2, rank);
    e.tri = .{ .rec = rec };
    e.pos_dev = at.pos;
    e.lane = null;
    e.s.lat = at.lat;
    e.s.qa = at.qa;
    e.s.ol = at.ol;
    e.s.po = at.po;
    e.s.pm = at.pm;
    e.s.pl = at.pl;
    e.s.dummy = at.dummy;
    e.s.tokens = at.tokens;
    e.s.counts = at.counts;
    e.s.origins = at.origins;
    e.s.nch = 64;
    e.s.part_rows = grid.att_rows;
    return e;
}

test "the prompt fetch's launches find the variant of their unsharded launch at every chunk size" {
    var set = try Captured.load();
    defer set.deinit();
    const ks = set.parsed.value.kernels;
    // the prompt scratch's buffers and the fetch's as arenas carve them (every buffer 256-aligned), the stream's latent
    // rows at an extent of its part, POS at a slot's 32-byte stride
    var b: [11]u64 = undefined;
    carve(0x7e00_0000_0000, &.{ 2048 * 1024, 2048 * qrow, 2048 * qrow, 512 * H * 512 * 4, 4096, 4096, 16, 2048 * W * 4, 2048 * 4, 40 * 16, 32 }, &b);
    var f: [4]u64 = undefined;
    carve(0x7d00_0000_0000, &sp.layout(1 << 20, 2048), &f);
    const p: sp.Ptrs = .{ .copy = f[0], .mark = f[1], .sums = f[2], .zero = f[3] };
    const lc: u64 = 0x7c00_0000_0000 + 3 * 2048 * RS;
    const rec = try gpa.create(Recorder);
    defer gpa.destroy(rec);
    const plain = try gpa.create(Recorder);
    defer gpa.destroy(plain);
    var prng = std.Random.DefaultPrng.init(1182);
    const rnd = prng.random();
    for ([_]usize{ 1, 3, 63, 64, 65, 150, 512, 513, 2048 }) |rows| for ([_]usize{ 0, 1000, 2000, 2051, 9000 }) |pos| {
        rec.* = .{};
        plain.* = .{};
        const sparse_rows = pos + rows - 1 >= grid.sparse_from;
        const at: Bufs = .{ .lat = b[0], .qa = b[1], .ol = b[2], .po = b[3], .pm = b[4], .pl = b[5], .dummy = b[6], .tokens = b[7], .counts = b[8], .origins = b[9], .pos = b[10] };
        const e = try engine(rec, 1, at);
        const u = try engine(plain, 1, at);
        // cuda_prefill.dsaLayer's unsharded launches: the latent write, the dense rows, the sparse rows
        try u.latWrite(lc, u.pos_dev, rows);
        try u.denseRows(lc, rows, pos, sparse_rows);
        if (sparse_rows) try u.onepass(lc, 0, rows);
        // the receiving rank's: its own rows into the copy (at their positions, or after a union's n rows from the zero
        // word), the dense rows over the copy, the sparse rows over it whole or in 64-row blocks
        const a: pre.ShardAttn = .{ .e = &e, .rows = rows, .pos = pos, .sparse_rows = sparse_rows };
        if (pos < grid.sparse_from) {
            try a.latWrite(p.copy, null);
            try a.dense(p.copy);
            if (sparse_rows) try a.sparse(p.copy, 0, rows);
        } else {
            try a.latWrite(p.copy + rnd.intRangeAtMost(usize, 0, sp.historyRows(1 << 20)) * RS, p.zero);
            var lo: usize = 0;
            while (lo < rows) : (lo += sp.block_rows * rnd.intRangeAtMost(usize, 1, 4)) try a.sparse(p.copy, lo, @min(rows, lo + sp.block_rows * 4));
        }
        // a chunk with dense rows: the unsharded launches with LC moved, one for one (the same grids in the same order);
        // past the dense limit: the latent write and each block's onepass against the unsharded ones
        const whole = pos < grid.sparse_from;
        if (whole) try std.testing.expectEqual(plain.n, rec.n);
        for (rec.launches[0..rec.n], 0..) |*l, i| {
            const twin = if (whole) &plain.launches[i] else for (plain.launches[0..plain.n]) |*x| {
                if (std.mem.eql(u8, x.name, l.name)) break x;
            } else return error.NoUnshardedTwin;
            try std.testing.expectEqualStrings(twin.name, l.name);
            if (whole) try std.testing.expectEqual(twin.grid, l.grid);
            const want = variant(ks, twin) orelse return error.UnshardedHasNoVariant;
            const got = variant(ks, l) orelse {
                std.debug.print("{s} at {d} rows from {d}: no captured variant\n", .{ l.name, rows, pos });
                return error.MissingTritonVariant;
            };
            try std.testing.expectEqual(want, got);
        }
    };
}

test "the copy holds the whole history up to 256K tokens, half past it, and every 64-row block's positions" {
    try std.testing.expectEqual(@as(usize, 131072), sp.historyRows(131072));
    try std.testing.expectEqual(@as(usize, 262144), sp.historyRows(262144));
    try std.testing.expectEqual(@as(usize, 262144), sp.historyRows(300000));
    try std.testing.expectEqual(@as(usize, 524288), sp.historyRows(1 << 20));
    for ([_]usize{ 4096, 131072, 262144, 262145, 600000, 1 << 20, 1 << 21 }) |c| {
        // a 64-row block names at most 64 x 2,051 positions, and a history this short is held whole
        try std.testing.expect(sp.historyRows(c) >= @min(c, sp.block_rows * W));
        try std.testing.expect(sp.historyRows(c) >= @min(c, grid.sparse_from));
    }
    // ~269 MiB at 1M and 2,048-row chunks (the copy 265, the marks 4), ~68 MiB at 128K
    const at_1m = sp.bytes(1 << 20, 2048);
    try std.testing.expect(at_1m > 265 << 20 and at_1m < 275 << 20);
    try std.testing.expectEqual(@as(usize, (524288 + 2048) * 528), sp.layout(1 << 20, 2048)[0]);
    try std.testing.expectEqual(@as(usize, 4 << 20), sp.layout(1 << 20, 2048)[1]);
    try std.testing.expect(sp.bytes(131072, 2048) < 68 << 20);
}

test "a rank's side of a sharded chunk: the owner's view sends, a view with no latent rows receives, a borrowing prompt runs unsharded on both ranks" {
    const sh = try shard.Shard.init(0, 2, 8 * 2048, shard.trash);
    var c: state.Caches = .{ .arena = undefined, .rows = 8 * 2048, .streams = 2, .ring = 2112, .latent = @splat(0x100000), .pooled = @splat(0x200000), .keys = @splat(0x300000), .gates = @splat(0x400000), .shard = sh };
    try std.testing.expectEqual(sp.Role.send, (try sp.roleOf(try c.view(2048, 2048, 0))).?);
    try std.testing.expectEqual(sp.Role.recv, (try sp.roleOf(try c.view(4 * 2048 + 2048, 2048, 1))).?);
    // a twin: no role on either rank (rank 0 here holds rank 1's extent's twin; rank 1 holds the extent itself)
    var c1 = c;
    c1.shard = try shard.Shard.init(1, 2, 8 * 2048, shard.trash);
    const lent = [2]state.CacheView{ try c.viewBorrowed(4 * 2048, 2048, 1, 2048), try c1.viewBorrowed(4 * 2048, 2048, 1, 2048) };
    for (lent) |v| try std.testing.expectEqual(@as(?sp.Role, null), try sp.roleOf(v));
    // another rank's rows here without the mark: no view this engine makes
    var odd = lent[0];
    odd.twin = false;
    try std.testing.expectError(error.KvShardTwinPrompt, sp.roleOf(odd));
    // mutant: a borrowing prompt's chunk fetches (the owner sends what the other rank holds), and the twin's side fails
    sp.mutant = .twin_fetches;
    defer sp.mutant = .none;
    try std.testing.expectEqual(sp.Role.send, (try sp.roleOf(lent[1])).?);
    try std.testing.expectError(error.KvShardTwinPrompt, sp.roleOf(lent[0]));
    sp.mutant = .none;
    c.shard = null;
    try std.testing.expectEqual(sp.Role.send, (try sp.roleOf(try c.view(4 * 2048, 2048, 0))).?); // (never asked unsharded)
}

test "compactRef gathers the marked rows in position order (cuda_shard.gatherRows over historyRows), or every row" {
    var prng = std.Random.DefaultPrng.init(1183);
    const rnd = prng.random();
    for (0..40) |_| {
        const limit = rnd.intRangeAtMost(usize, 0, 300);
        const plane = try gpa.alloc(u8, (limit + 1) * RS);
        defer gpa.free(plane);
        rnd.bytes(plane);
        var marks: [300]i32 = undefined;
        for (marks[0..limit]) |*x| x.* = @intFromBool(rnd.uintLessThan(u8, 3) == 0);
        var cum: [300]i32 = undefined;
        const n = shard.prefixSum(marks[0..limit], cum[0..limit]);
        var hist: [300]u32 = undefined;
        const at = shard.historyRows(marks[0..limit], &hist);
        const want = try gpa.alloc(u8, (limit + 1) * RS);
        defer gpa.free(want);
        const got = try gpa.alloc(u8, (limit + 1) * RS);
        defer gpa.free(got);
        @memset(want, 0xaa);
        @memset(got, 0xaa);
        shard.gatherRows(plane, RS, at, want);
        sp.compactRef(plane, cum[0..limit], limit, got);
        try std.testing.expectEqual(at.len, n);
        try std.testing.expectEqualSlices(u8, want, got);
        sp.compactRef(plane, null, limit, got);
        try std.testing.expectEqualSlices(u8, plane[0 .. limit * RS], got[0 .. limit * RS]);
    }
}

// -- the two-rank model ---------------------------------------------------------------------------------------------

/// The model's memory: named host buffers, a pointer valid only inside one of them.
const Mem = struct {
    names: [96][]const u8 = undefined,
    spans: [96][]u8 = undefined,
    n: usize = 0,
    fn add(m: *Mem, name: []const u8, b: []u8) u64 {
        m.names[m.n] = name;
        m.spans[m.n] = b;
        m.n += 1;
        return @intFromPtr(b.ptr);
    }
    fn at(m: *const Mem, addr: u64, len: usize) ![]u8 {
        for (m.spans[0..m.n]) |s| if (addr >= @intFromPtr(s.ptr) and addr < @intFromPtr(s.ptr) + s.len) {
            const off = addr - @intFromPtr(s.ptr);
            if (off + len > s.len) return error.OffBuffer;
            return s[off..][0..len];
        };
        return error.OffBuffer;
    }
    fn word(m: *const Mem, addr: u64) !i32 {
        return std.mem.readInt(i32, (try m.at(addr, 4))[0..4], .little);
    }
};

fn ints(b: []u8) []i32 {
    return @alignCast(std.mem.bytesAsSlice(i32, b));
}

/// A latent row into a running hash: what the kernels read, in the order they read it.
fn keyHash(m: *const Mem, lc: u64, t: usize, h: *u64) !void {
    const b = try m.at(lc + t * RS, RS);
    for (0..RS / 8) |i| h.* = (h.* ^ std.mem.readInt(u64, b[i * 8 ..][0..8], .little)) *% 0x100000001b3 +% 0x9e3779b97f4a7c15;
}
/// `_lat_write`'s quantization as a model: a chunk row's 1,024 bf16 bytes into a 528-byte latent row, a function of all
/// of them (mla.quantizeRow, the real rule, searches 127 codes an element: far too slow for the model's thousands).
fn packRow(lat: []const u8, out: []u8) void {
    var h: u64 = 0x6a09e667f3bcc908;
    for (0..512) |j| {
        out[j] = lat[2 * j] ^ lat[2 * j + 1];
        h = (h ^ lat[2 * j] ^ (@as(u64, lat[2 * j + 1]) << 8)) *% 0x100000001b3;
    }
    for (0..16) |i| out[512 + i] = @truncate(h >> @intCast((i % 8) * 8));
}

/// A row's output element from its query element and the hash of the keys it read.
fn mix(q: u16, h: u64, j: usize) u16 {
    return q ^ @as(u16, @truncate(h >> @intCast((j % 4) * 16)));
}

/// The modelled Triton kernels: `_lat_write` (a chunk row's latent packed into LC at POS + r), `_dense_chunks` with
/// `_merge` (row r of the block reads LC rows 0 .. POS + r), `_sparse_onepass` (row r reads the CNT rows its tokens
/// name, in order; rows with CNT 0 left alone), each address checked against the buffers.
fn exec(m: *const Mem, s: *const Recorder.Saved) !void {
    const n: usize = s.grid[0];
    if (std.mem.eql(u8, s.name, "_lat_write")) {
        const stride: usize = @intCast(arg(s, "lat_stride").value.i32);
        const at: usize = @intCast(try m.word(ptrArg(s, "POS")));
        for (0..n) |r| packRow(try m.at(ptrArg(s, "LAT") + 2 * r * stride, 1024), try m.at(ptrArg(s, "LC") + (at + r) * RS, RS));
    } else if (std.mem.eql(u8, s.name, "_dense_chunks")) {
        const at: usize = @intCast(try m.word(ptrArg(s, "POS")));
        for (0..n) |r| {
            var h: u64 = 0xcbf29ce484222325;
            for (0..at + r + 1) |t| try keyHash(m, ptrArg(s, "LC"), t, &h);
            const q = try m.at(ptrArg(s, "QA") + r * qrow, qrow);
            const po = try m.at(ptrArg(s, "PO") + r * H * 512 * 4, H * 512 * 4);
            for (0..H * 512) |j| std.mem.writeInt(u32, po[4 * j ..][0..4], mix(std.mem.readInt(u16, q[2 * j ..][0..2], .little), h, j), .little);
        }
    } else if (std.mem.eql(u8, s.name, "_merge")) {
        for (0..n) |r| {
            const po = try m.at(ptrArg(s, "PO") + r * H * 512 * 4, H * 512 * 4);
            const out = try m.at(ptrArg(s, "OUT") + r * qrow, qrow);
            for (0..H * 512) |j| std.mem.writeInt(u16, out[2 * j ..][0..2], @truncate(std.mem.readInt(u32, po[4 * j ..][0..4], .little)), .little);
        }
    } else if (std.mem.eql(u8, s.name, "_sparse_onepass")) {
        for (0..n) |r| {
            const cnt = try m.word(ptrArg(s, "CNT") + 4 * r);
            if (cnt == 0) continue;
            var h: u64 = 0x84222325cbf29ce4;
            for (0..@intCast(cnt)) |i| {
                const t = try m.word(ptrArg(s, "TOK") + 4 * (r * W + i));
                if (t < 0) return error.BadToken;
                try keyHash(m, ptrArg(s, "LC"), @intCast(t), &h);
            }
            const q = try m.at(ptrArg(s, "QA") + r * qrow, qrow);
            const out = try m.at(ptrArg(s, "OUT") + r * qrow, qrow);
            for (0..H * 512) |j| std.mem.writeInt(u16, out[2 * j ..][0..2], mix(std.mem.readInt(u16, q[2 * j ..][0..2], .little), h, j), .little);
        }
    } else return error.UnmodelledKernel;
}

/// The wire between the two ranks: what the sender's transfers carried, in order.
const Wire = struct {
    msgs: std.ArrayList([]u8) = .empty,
    next: usize = 0,
    fn deinit(w: *Wire) void {
        for (w.msgs.items) |b| gpa.free(b);
        w.msgs.deinit(gpa);
    }
};

/// cuda_shard_prompt.attend's device in the model: glm_shard.cu by its host rules, the transfers through the wire (each
/// inside hold / release when the chunk has a side stream), and cuda_prefill's attention launches recorded and run at
/// once, each on a captured variant.
const Dev = struct {
    m: *Mem,
    a: pre.ShardAttn,
    rec: *Recorder,
    ks: []const aot.KernelJson,
    wire: *Wire,
    lane: bool,
    held: bool = false,

    fn flush(d: *Dev) !void {
        for (d.rec.launches[0..d.rec.n]) |*l| {
            if (variant(d.ks, l) == null) return error.MissingTritonVariant;
            try exec(d.m, l);
        }
        d.rec.n = 0;
    }
    pub fn mark(d: *Dev, tok: u64, n: usize, mk: u64, limit: usize) !void {
        shard.markHistory(ints(try d.m.at(tok, 4 * n)), limit, ints(try d.m.at(mk, 4 * limit)));
    }
    pub fn scan(d: *Dev, mk: u64, n: usize, sums: u64) !void {
        const a = ints(try d.m.at(mk, 4 * n));
        var run: i32 = 0;
        for (a) |*x| {
            run += x.*;
            x.* = run;
        }
        ints(try d.m.at(sums, 4 * (sp.scan_segs + 1)))[sp.scan_segs] = run;
    }
    pub fn count(d: *Dev, sums: u64) !usize {
        return @intCast(try d.m.word(sums + 4 * sp.scan_segs));
    }
    pub fn compact(d: *Dev, src: u64, cum: u64, limit: usize, dst: u64) !void {
        const c: ?[]const i32 = if (cum == 0) null else ints(try d.m.at(cum, 4 * limit));
        const rows: usize = if (c) |x| (if (limit == 0) 0 else @intCast(x[limit - 1])) else limit;
        sp.compactRef(try d.m.at(src, limit * RS), c, limit, try d.m.at(dst, rows * RS));
    }
    pub fn renumber(d: *Dev, tok: u64, n: usize, cum: u64, limit: usize) !void {
        const t = ints(try d.m.at(tok, 4 * n));
        const was = try gpa.dupe(i32, t);
        defer gpa.free(was);
        shard.renumber(was, limit, ints(try d.m.at(cum, 4 * limit)), t);
    }
    pub fn hold(d: *Dev) !void {
        if (d.held) return error.HeldTwice;
        d.held = true;
    }
    pub fn release(d: *Dev) !void {
        if (!d.held) return error.NotHeld;
        d.held = false;
    }
    pub fn send(d: *Dev, at: u64, n: usize) !void {
        if (d.lane and !d.held) return error.Unheld;
        try d.wire.msgs.append(gpa, try gpa.dupe(u8, try d.m.at(at, n)));
    }
    pub fn recv(d: *Dev, at: u64, n: usize) !void {
        if (d.lane and !d.held) return error.Unheld;
        if (d.wire.next >= d.wire.msgs.items.len) return error.ExchangeMismatch;
        const b = d.wire.msgs.items[d.wire.next];
        d.wire.next += 1;
        if (b.len != n) return error.ExchangeMismatch;
        @memcpy(try d.m.at(at, n), b);
    }
    pub fn latWrite(d: *Dev, att: u64, pos: ?u64) !void {
        try d.a.latWrite(att, pos);
        try d.flush();
    }
    pub fn dense(d: *Dev, att: u64) !void {
        try d.a.dense(att);
        try d.flush();
    }
    pub fn sparse(d: *Dev, att: u64, lo: usize, hi: usize) !void {
        try d.a.sparse(att, lo, hi);
        try d.flush();
    }
};

/// One rank's buffers.
const Rank = struct {
    bufs: [24][]u8 = undefined,
    nb: usize = 0,
    fn buf(r: *Rank, m: *Mem, name: []const u8, len: usize) !u64 {
        const b = try gpa.alignedAlloc(u8, .fromByteUnits(256), @max(len, 16));
        @memset(b, 0);
        r.bufs[r.nb] = b;
        r.nb += 1;
        return m.add(name, b);
    }
    fn deinit(r: *Rank) void {
        for (r.bufs[0..r.nb]) |b| gpa.free(@as([]align(256) u8, @alignCast(b)));
    }
};

/// A chunk on both ranks: `rows` rows from `pos`, rows past the dense limit naming up to `most` tokens each, a copy of
/// `hist` history rows, transfers of `piece` rows, the stream's rows on rank `owner`, a split side stream (`lane`).
const Case = struct { rows: usize, pos: usize, most: usize = 24, hist: usize = 1 << 20, piece: usize = 97, owner: usize = 0, lane: bool = false };

/// The case on both ranks, unsharded then sharded; each rank's counters when its heads of every row equal its unsharded
/// attention's bit for bit (null: they differ, or a transfer, a read or a launch went wrong).
fn trial(rnd: std.Random, ks: []const aot.KernelJson, c: Case) !?[2]sp.Stats {
    var m: Mem = .{};
    var rk: [2]Rank = .{ .{}, .{} };
    defer for (&rk) |*r| r.deinit();
    const rows = c.rows;
    const end = c.pos + rows;
    // the history, alike on both ranks (the owner's plane holds it, the unsharded engine's planes hold it everywhere)
    const hist = try gpa.alloc(u8, c.pos * RS);
    defer gpa.free(hist);
    rnd.bytes(hist);
    // the chunk's latents (replicated), its tokens and counts (the selection replicated): a row past the dense limit
    // names up to `most` positions at or before its own, ascending, -1 after them; a row under it leaves garbage that
    // nothing reads (`_tokens` writes pools' tokens with count 0)
    const lat = try gpa.alloc(u8, rows * 1024);
    defer gpa.free(lat);
    rnd.bytes(lat);
    const toks = try gpa.alloc(i32, rows * W);
    defer gpa.free(toks);
    const counts = try gpa.alloc(i32, @max(rows, 4));
    defer gpa.free(counts);
    @memset(toks, -1);
    @memset(counts, 0);
    for (0..rows) |r| {
        const q = c.pos + r;
        const t = toks[r * W ..][0..W];
        if (q < grid.sparse_from) {
            for (t[0..64]) |*x| x.* = rnd.intRangeLessThan(i32, 0, 8192);
            continue;
        }
        const n = if (rnd.uintLessThan(u8, 12) == 0) 0 else rnd.intRangeAtMost(usize, 1, c.most);
        for (t[0..n]) |*x| x.* = @intCast(if (rnd.uintLessThan(u8, 5) == 0) rnd.intRangeAtMost(usize, c.pos, q) else rnd.intRangeAtMost(usize, 0, q));
        std.mem.sort(i32, t[0..n], {}, std.sort.asc(i32));
        counts[r] = @intCast(n);
    }
    const recs = try gpa.alloc(Recorder, 2);
    defer gpa.free(recs);
    var engines: [2]pre.Engine = undefined;
    var full: [2]u64 = undefined;
    var ref: [2]u64 = undefined;
    var local: [2]u64 = undefined;
    var ptrs: [2]sp.Ptrs = undefined;
    const context = end + 1;
    for (0..2) |k| {
        const r = &rk[k];
        recs[k] = .{};
        const tok = try r.buf(&m, "tokens", rows * W * 4);
        @memcpy(ints(try m.at(tok, rows * W * 4)), toks);
        const cnt = try r.buf(&m, "counts", @max(rows, 4) * 4);
        @memcpy(ints(try m.at(cnt, @max(rows, 4) * 4)), counts);
        const qa = try r.buf(&m, "qa", rows * qrow);
        rnd.bytes(try m.at(qa, rows * qrow)); // each rank's own heads
        const origins = try r.buf(&m, "origins", 40 * 16);
        for (0..grid.chunksFor(c.pos, rows)) |i| ints(try m.at(origins + i * 16, 4))[0] = @intCast(c.pos + i * grid.att_rows);
        const pos_dev = try r.buf(&m, "pos", 32);
        ints(try m.at(pos_dev, 4))[0] = @intCast(c.pos);
        const lat_dev = try r.buf(&m, "lat", rows * 1024);
        @memcpy(try m.at(lat_dev, rows * 1024), lat);
        engines[k] = try engine(&recs[k], k, .{
            .lat = lat_dev,
            .qa = qa,
            .ol = try r.buf(&m, "ol", rows * qrow),
            .po = try r.buf(&m, "po", @as(usize, @min(rows, grid.att_rows)) * H * 512 * 4),
            .pm = try r.buf(&m, "pm", 64),
            .pl = try r.buf(&m, "pl", 64),
            .dummy = try r.buf(&m, "dummy", 16),
            .tokens = tok,
            .counts = cnt,
            .origins = origins,
            .pos = pos_dev,
        });
        full[k] = try r.buf(&m, "full", end * RS);
        @memcpy((try m.at(full[k], end * RS))[0 .. c.pos * RS], hist);
        ref[k] = try r.buf(&m, "ref", rows * qrow);
        local[k] = if (k == c.owner) try r.buf(&m, "local", end * RS) else 0;
        if (local[k] != 0) @memcpy((try m.at(local[k], end * RS))[0 .. c.pos * RS], hist);
        const sizes = sp.layout(context, rows);
        ptrs[k] = .{ .copy = try r.buf(&m, "copy", (@min(c.hist, sp.historyRows(context)) + rows) * RS), .mark = try r.buf(&m, "mark", sizes[1]), .sums = try r.buf(&m, "sums", sizes[2]), .zero = try r.buf(&m, "zero", sizes[3]) };
    }
    const sparse_rows = end - 1 >= grid.sparse_from;
    // each rank unsharded (cuda_prefill.dsaLayer): the chunk's latents at their positions, its heads over the plane (a
    // row past the dense limit with no token is left alone, so both runs start from the same bytes)
    for (0..2) |k| {
        const e = &engines[k];
        var d: Dev = .{ .m = &m, .a = .{ .e = e, .rows = rows, .pos = c.pos, .sparse_rows = sparse_rows }, .rec = &recs[k], .ks = ks, .wire = undefined, .lane = false };
        @memset(try m.at(e.s.ol, rows * qrow), 0x5a);
        try d.latWrite(full[k], null);
        try d.dense(full[k]);
        if (sparse_rows) try d.sparse(full[k], 0, rows);
        @memcpy(try m.at(ref[k], rows * qrow), try m.at(e.s.ol, rows * qrow));
        @memset(try m.at(e.s.ol, rows * qrow), 0x5a);
    }
    // each rank sharded, the owner first (its sends go out before the other rank's receives take them)
    var wire: Wire = .{};
    defer wire.deinit();
    var stats: [2]sp.Stats = .{ .{}, .{} };
    const lim: sp.Limits = .{ .hist = @min(c.hist, sp.historyRows(context)), .chunk = rows, .piece = c.piece };
    for ([_]usize{ c.owner, 1 - c.owner }) |k| {
        const e = &engines[k];
        const send = k == c.owner;
        var d: Dev = .{ .m = &m, .a = .{ .e = e, .rows = rows, .pos = c.pos, .sparse_rows = sparse_rows }, .rec = &recs[k], .ks = ks, .wire = &wire, .lane = c.lane };
        if (send) try d.latWrite(local[k], null); // dsaLayer: the owner's latents into its own rows first
        sp.attend(&d, ptrs[k], lim, .{ .role = if (send) .send else .recv, .rows = rows, .pos = c.pos, .sparse = sparse_rows, .tokens = e.s.tokens, .lc = local[k] }, &stats[k]) catch |err| switch (err) {
            error.OffBuffer, error.ExchangeMismatch, error.Unheld, error.MissingTritonVariant, error.BadToken, error.KvShardFetchRows => return null,
            else => return err,
        };
        if (d.held) return null;
    }
    if (wire.next != wire.msgs.items.len) return null;
    for (0..2) |k| if (!std.mem.eql(u8, try m.at(engines[k].s.ol, rows * qrow), try m.at(ref[k], rows * qrow))) return null;
    return stats;
}

test "two ranks' sharded prompt chunks give each rank its unsharded heads: whole history, union, row blocks, pieces" {
    var set = try Captured.load();
    defer set.deinit();
    const ks = set.parsed.value.kernels;
    var prng = std.Random.DefaultPrng.init(1184);
    const rnd = prng.random();
    const cases = [_]Case{
        .{ .rows = 64, .pos = 0 }, // a first chunk: no history, the copy holds the chunk's own rows
        .{ .rows = 96, .pos = 1000, .piece = 50, .lane = true, .owner = 1 }, // the whole history in 20 pieces
        .{ .rows = 160, .pos = 2000, .most = 12 }, // 51 dense rows, then sparse ones over the whole history
        .{ .rows = 64, .pos = 5000, .most = 40, .piece = 113, .owner = 1 }, // the union, one block
        .{ .rows = 1, .pos = 2051 }, // the first position past the dense limit
        .{ .rows = 192, .pos = 7000, .most = 7, .hist = 450, .lane = true }, // the union in row blocks
        .{ .rows = 150, .pos = 6000, .most = 12, .hist = 500, .owner = 1, .piece = 31 }, // blocks of a ragged chunk
        .{ .rows = 600, .pos = 100, .piece = 4096 }, // 600 dense rows: two dense blocks of the copy
    };
    for (cases, 0..) |c, i| {
        const st = (try trial(rnd, ks, c)) orelse {
            std.debug.print("prompt fetch case {d} differs from the unsharded attention\n", .{i});
            return error.TestUnexpectedResult;
        };
        // both ranks moved the same rows and cut alike; the dense cases need no count
        try std.testing.expectEqual(st[0].rows, st[1].rows);
        try std.testing.expectEqual(st[0].blocks, st[1].blocks);
        try std.testing.expectEqual(st[0].syncs, st[1].syncs);
        if (c.pos < grid.sparse_from) {
            try std.testing.expectEqual(@as(u64, c.pos), st[0].rows);
            try std.testing.expectEqual(@as(u64, 0), st[0].syncs);
        }
        try std.testing.expectEqual(@as(u64, @intFromBool(c.hist < 1000)), st[0].blocked);
        if (c.hist < 1000) try std.testing.expect(st[0].blocks >= 2 and st[0].syncs > st[0].blocks);
        try std.testing.expectEqual(@as(u64, @intFromBool(st[0].rows > 0)), st[0].fetches);
    }
    // random chunks either side of the dense limit, the copy bounded or not
    for (0..12) |i| {
        const pos = if (i % 3 == 0) rnd.intRangeAtMost(usize, 0, 2050) else rnd.intRangeAtMost(usize, 2051, 6000);
        const rows = rnd.intRangeAtMost(usize, 1, 140);
        const c: Case = .{ .rows = rows, .pos = pos, .most = 6, .hist = if (i % 2 == 0) 450 else 1 << 20, .piece = rnd.intRangeAtMost(usize, 1, 300), .owner = i % 2, .lane = i % 4 < 2 };
        if (pos < grid.sparse_from and c.hist < pos) continue; // (a whole history never exceeds a real copy)
        if ((try trial(rnd, ks, c)) == null) {
            std.debug.print("random prompt fetch {d} ({d} rows from {d}) differs from the unsharded attention\n", .{ i, rows, pos });
            return error.TestUnexpectedResult;
        }
    }
}

test "every prompt fetch mutant is caught by the two-rank model" {
    defer sp.mutant = .none;
    var set = try Captured.load();
    defer set.deinit();
    const ks = set.parsed.value.kernels;
    const cases = [_]Case{
        .{ .rows = 64, .pos = 5000, .most = 40, .lane = true },
        .{ .rows = 96, .pos = 1000, .piece = 50, .lane = true, .owner = 1 },
        .{ .rows = 192, .pos = 7000, .most = 7, .hist = 450, .lane = true },
        .{ .rows = 150, .pos = 6000, .most = 12, .hist = 500, .owner = 1 },
        .{ .rows = 160, .pos = 2000, .most = 12, .owner = 1, .lane = true },
    };
    for (std.enums.values(sp.Mutant)) |mu| {
        if (mu == .none or mu == .twin_fetches) continue; // (a chunk's role: the roles test kills it)
        var caught = false;
        for (cases, 0..) |c, i| {
            // the same chunk passes unmutated and fails mutated
            var prng = std.Random.DefaultPrng.init(1185 + i);
            sp.mutant = .none;
            if ((try trial(prng.random(), ks, c)) == null) return error.TestUnexpectedResult;
            prng = std.Random.DefaultPrng.init(1185 + i);
            sp.mutant = mu;
            if ((try trial(prng.random(), ks, c)) == null) {
                caught = true;
                break;
            }
        }
        if (!caught) {
            std.debug.print("prompt fetch mutant {t} survived\n", .{mu});
            return error.MutantSurvived;
        }
    }
}
