//! TF_GLM_KV_SHARD's owner-computed decode (patch 0118, glm5_next/cuda/shard.py `attention`, two ranks): a decode
//! window's latent attention when each stream's latent rows live on one rank only (cuda_shard.zig). Every rank's
//! absorbed queries [R, 32, 512] reach the other rank; each row's owner attends all 64 heads over its own rows as two
//! passes of the captured H=32 `_seg_chunks` + `_merge` (its own queries, then the peer's), reading the window's glue
//! (`lbase`: the row's stream in this rank's latent planes; `sparse_eff` 1 and its count 0 for a row it does not own,
//! which reads no key); the peer's heads go back to it and every rank keeps its own heads of each row (glm_shard.cu's
//! select). The same kernels over the same keys in the same order, so every bit is the unsharded engine's; no new
//! Triton variant either: every pointer the passes take in place of an unsharded one is in its class (`offClass`).
//! The exchanges: in a captured graph a fixed block of every row a rank (sizes fixed per R: the one-shot RoCE
//! all-gather when a block fits its slot, else an NCCL send and receive), outside a capture (graph warm-ups,
//! TF_GLM_EAGER) the crossing rows alone (cuda_shard.lists, one NCCL group). The own-head pass runs on a side stream
//! beside the queries' exchange and the peer-head pass. Pooled index keys stay replicated: the selection is unchanged.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const shard = @import("cuda_shard.zig");
const seg = @import("cuda_segments.zig");
const state = @import("cuda_state.zig");
const side_mod = @import("cuda_side.zig");
const Tri = @import("cuda_triton.zig").Tri;
const Comm = @import("cuda_comm.zig").Comm;
const G = @import("cuda_geometry.zig").Geometry;

/// A rank's heads (the captured kernels' H at TP=2) and a latent row's width.
pub const heads = 32;
const lw = G.kv_lora;
/// One window row's absorbed queries, or its attention output, of one rank's heads: bf16 [32][512].
pub const row_bytes = heads * lw * 2;
/// The decode passes' chunks (cuda_forward's NCH of a segment window).
const nch = 5;

comptime {
    // the peer's block of a gathered buffer starts at rows * row_bytes: 16-byte aligned at every window size
    std.debug.assert(row_bytes % 16 == 0);
}

// -- buffers ----------------------------------------------------------------------------------------------------------

/// The exchange buffers of a window of `rows` rows, allocated before any graph captures (sizes fixed by the window):
/// both ranks' queries and returned outputs as an all-gather lays them out, [2][R][32][512] bf16 (an uncaptured round's
/// crossing rows land at the same places), the peer-head pass's outputs, the masked counts, and the own-head pass's
/// split partials (it runs beside the peer-head pass, which keeps cuda_forward.Scratch's).
pub const Ptrs = struct { qall: u64, olp: u64, orecv: u64, ce: u64, po: u64, pm: u64, pl: u64 };

pub fn layout(rows: usize) [7]usize {
    const ah = rows * heads * lw;
    return .{ 2 * ah * 2, ah * 2, 2 * ah * 2, rows * 4, nch * ah * 4, nch * rows * heads * 4, nch * rows * heads * 4 };
}

/// Bytes `Buffers` takes (cuda_engine's sizing counts them on the sharded side).
pub fn bytes(rows: usize) usize {
    return state.aligned(&layout(rows));
}

/// The peer's (or any rank's) block of a gathered buffer for a window of `rows` rows.
pub fn block(base: u64, rank: usize, rows: usize) u64 {
    return base + rank * rows * row_bytes;
}

pub const Buffers = struct {
    arena: state.Arena,
    p: Ptrs,
    rows: usize,
    pub fn init(d: *const cuda.Driver, rows: usize) !Buffers {
        const sizes = layout(rows);
        var a = try state.Arena.init(d, &sizes);
        errdefer a.deinit();
        var at: [7]u64 = undefined;
        for (&at, sizes) |*p, n| p.* = a.take(n);
        return .{ .arena = a, .p = .{ .qall = at[0], .olp = at[1], .orecv = at[2], .ce = at[3], .po = at[4], .pm = at[5], .pl = at[6] }, .rows = rows };
    }
    pub fn deinit(b: *Buffers) void {
        b.arena.deinit();
    }
};

// -- glm_shard.cu ------------------------------------------------------------------------------------------------------

pub const Kernels = struct {
    module: cuda.Module,
    counts_fn: cuda.Function,
    select_fn: cuda.Function,
    /// glm_shard.cu's image (cuda.kernels.glm_shard; empty in a host-only build).
    pub fn load(d: *const cuda.Driver, image: []const u8) !Kernels {
        if (image.len == 0) return error.NoKvShardKernels;
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        return .{ .module = m, .counts_fn = try m.function("glm_shard_counts"), .select_fn = try m.function("glm_shard_select") };
    }
    pub fn unload(k: *Kernels) void {
        k.module.unload();
    }
    /// out[r] = counts[r] where own[r] != 0, else 0 (int32 [rows]; `countsRef`).
    pub fn counts(k: *const Kernels, s: cuda.Stream, in: u64, own: u64, out: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ in, own, out }) |x| a.add(x);
        a.add(@as(c_int, @intCast(rows)));
        try cuda.launch.launch(k.counts_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 64 } }, s, &a);
    }
    /// Rows of `out` whose own[r] is 0 take `recv`'s, rows of `row` bytes (`selectRef`).
    pub fn select(k: *const Kernels, s: cuda.Stream, recv: u64, out: u64, own: u64, rows: usize, row: usize) !void {
        if ((recv | out | row) % 16 != 0) return error.BadSelect;
        var a: cuda.Args = .{};
        for ([_]u64{ recv, out, own }) |x| a.add(x);
        a.add(@as(i64, @intCast(row / 16)));
        try cuda.launch.launch(k.select_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }
};

/// glm_shard.cu's two kernels as host code: the model test's device (tools/glm_shard_emulate holds the kernels to the
/// same rules).
pub fn countsRef(in: []const i32, own: []const i32, out: []i32) void {
    for (in, own, out) |c, o, *x| x.* = if (o != 0) c else 0;
}
pub fn selectRef(recv: []const u8, out: []u8, own: []const i32, row: usize) void {
    for (own, 0..) |o, r| if (o == 0) @memcpy(out[r * row ..][0..row], recv[r * row ..][0..row]);
}

// -- the exchanges ----------------------------------------------------------------------------------------------------

/// How an exchange moves: captured, a fixed block of every row a rank, by the RoCE all-gather when the block fits its
/// slot (`gather`; both ranks decide by the size alone) or else an NCCL send and receive (`swap`, no local copy);
/// uncaptured, the crossing rows alone (`rows`).
pub const Way = enum { gather, swap, rows };

/// Runs of whole window rows, (address, bytes) each, in window order.
pub const Runs = struct {
    n: usize = 0,
    at: [shard.max_rows][2]u64 = undefined,
    pub fn slice(r: *const Runs) []const [2]u64 {
        return r.at[0..r.n];
    }
};

/// A list's rows (ascending, as cuda_shard.lists makes them) as runs of consecutive rows from `base`: a stream's rows
/// are consecutive in the window and share their owner, so there are at most as many runs as segments.
pub fn runs(l: *const shard.List, base: u64) Runs {
    var out: Runs = .{};
    for (l.slice()) |r| {
        const at = base + @as(u64, r) * row_bytes;
        if (out.n > 0 and out.at[out.n - 1][0] + out.at[out.n - 1][1] == at) {
            out.at[out.n - 1][1] += row_bytes;
            continue;
        }
        out.at[out.n] = .{ at, row_bytes };
        out.n += 1;
    }
    return out;
}

/// One exchange as this rank makes it. `send`: the window's rows of this rank ([R][32][512]); `recv`: the gathered
/// buffer ([2][R][32][512]), whose peer block it fills. gather: `count` elements a rank from `send` into `recv`; swap:
/// `count` from `send` to the peer while its block arrives at `recv`'s peer block; rows: `out` to the peer, `in` from
/// it (rank r's `out` is rank 1 - r's `in`, in order and size: cuda_shard.lists).
pub const Plan = struct { way: Way, count: usize, send: u64, recv: u64, out: Runs = .{}, in: Runs = .{} };

pub fn plan(x: shard.Exchange, send: u64, recv: u64, rows: usize, peer: usize, fixed: bool, roce: bool, ls: *const shard.Lists) Plan {
    const count = rows * heads * lw;
    if (fixed) return if (roce) .{ .way = .gather, .count = count, .send = send, .recv = recv } else .{ .way = .swap, .count = count, .send = send, .recv = block(recv, peer, rows) };
    const at = block(recv, peer, rows);
    return .{ .way = .rows, .count = count, .send = send, .recv = at, .out = runs(ls.of(x, true), send), .in = runs(ls.of(x, false), at) };
}

// -- one DSA layer's sharded attention --------------------------------------------------------------------------------

/// A window's attention pointers: the unsharded call's (cuda_forward.dsaMtp's), with the glue planes for BASE (`lbase`)
/// and SPR (`sparse`), the glue's `own`, and `out` (Scratch.ol) where this rank's heads of every row end.
pub const Attn = struct {
    rows: usize,
    rank: usize,
    hb: usize,
    qa: u64,
    lc: u64,
    pos: u64,
    lbase: u64,
    sparse: u64,
    own: u64,
    tokens: u64,
    counts: u64,
    out: u64,
    po: u64,
    pm: u64,
    pl: u64,
    overlap: bool = true,
};

const Pass = struct { q: u64, out: u64, po: u64, pm: u64, pl: u64 };

/// One H=32 pass over the rows this rank owns: dsaMtp's `_seg_chunks` and `_merge`, the same constants, BASE the
/// glue's lbase, SPR its sparse_eff, CNT the masked counts.
fn pass(t: Tri, a: Attn, ce: u64, p: Pass) !void {
    const r = a.rows;
    try t.seg_chunks(.{ r, (heads + a.hb - 1) / a.hb, nch }, p.q, a.lc, a.lc, a.pos, a.lbase, a.sparse, a.tokens, ce, p.po, p.pm, p.pl, @intCast(r), .{ .W = G.tokens, .H = heads, .LW = lw, .CH = 512, .SCALE = 0.0625, .HBT = @intCast(a.hb), .KTT = 32, .RS = G.latent_row, .FP8 = 1 });
    try t.merge(.{ r, heads, 1 }, p.po, p.pm, p.pl, p.out, ce, @intCast(r), .{ .H = heads, .LW = lw, .NCH = nch, .SPARSE = 0 });
}

/// shard.attention for two ranks: the counts masked (main stream); the own-head pass on the side stream (`fork`) with
/// its own partials while the main stream exchanges the queries, runs the peer-head pass and sends its outputs back;
/// the side joined, the rows this rank does not own take what came back. `dev` is the device's side of it: counts,
/// select (glm_shard.cu), exchange (cuda_comm), fork and join (a side stream); Device on the GPU, the tests' log.
pub fn run(dev: anytype, t: Tri, a: Attn, p: Ptrs) !void {
    const r = a.rows;
    if (r == 0 or r > shard.max_rows or a.rank > 1 or a.hb == 0) return error.BadRows;
    const peer = 1 - a.rank;
    const theirs = block(p.qall, if (mut(.peer_block_mine)) a.rank else peer, r);
    const back = block(p.orecv, if (mut(.back_mine)) a.rank else peer, r);
    const ce = if (mut(.counts_unmasked)) a.counts else p.ce;
    const own: Pass = .{ .q = a.qa, .out = a.out, .po = p.po, .pm = p.pm, .pl = p.pl };
    const other: Pass = .{ .q = theirs, .out = p.olp, .po = a.po, .pm = a.pm, .pl = a.pl };
    try dev.counts(t.s, a.counts, a.own, p.ce, r);
    if (a.overlap) {
        try pass(try dev.fork(t), a, ce, own);
        try dev.exchange(.q, a.qa, p.qall, r, t.s);
        if (mut(.out_before_peer)) try dev.exchange(.out, p.olp, p.orecv, r, t.s);
        try pass(t, a, ce, other);
        if (!mut(.out_before_peer)) try dev.exchange(.out, p.olp, p.orecv, r, t.s);
        if (mut(.select_before_join)) {
            try dev.select(t.s, back, a.out, a.own, r);
            return dev.join(t);
        }
        try dev.join(t);
    } else {
        try dev.exchange(.q, a.qa, p.qall, r, t.s);
        try pass(t, a, ce, own);
        try pass(t, a, ce, other);
        try dev.exchange(.out, p.olp, p.orecv, r, t.s);
    }
    try dev.select(t.s, back, a.out, a.own, r);
}

/// A sharded window's latent write (dsaMtp's `_seg_lat_write`): a row this rank owns at its stream's local row and
/// position, one it does not at its trash row (the glue's lbase + wpos, Python's `_Lat(rows.wpos, rows.lbase)`).
pub fn latWrite(t: Tri, tables: *const seg.SegRows, rows: usize, lat: u64, lc: u64) !void {
    try t.seg_lat_write(.{ rows, 1, 1 }, lat, lw, lc, lc, try plane(tables, .wpos), try plane(tables, .lbase), .{ .LW = lw, .RS = G.latent_row, .FP8 = 1 });
}

/// Glue plane `g`'s device address.
pub fn plane(tables: *const seg.SegRows, g: shard.Glue) !u64 {
    if (tables.shard == null) return error.NoGluePlanes;
    return tables.ptr(9 + @as(usize, @backingInt(g)));
}

/// The captured Triton kernels are specialized on each pointer's 16-byte divisibility, and every decode launch's are
/// 16-divisible (fixtures_aot_variants.json): a glue plane or buffer the sharded launches take in place of an
/// unsharded pointer off that class would find no variant (MissingTritonVariant) at its first round. The first such
/// plane or buffer by name (null: none), checked before any graph captures. R is the window's, so its class is the
/// unsharded launch's.
pub fn offClass(tables: *const seg.SegRows, p: Ptrs) !?[]const u8 {
    for ([_]shard.Glue{ .lbase, .sparse_eff, .wpos }) |g| if (try plane(tables, g) % 16 != 0) return @tagName(g);
    const named = [_]struct { []const u8, u64 }{ .{ "qall", p.qall }, .{ "olp", p.olp }, .{ "orecv", p.orecv }, .{ "ce", p.ce }, .{ "po", p.po }, .{ "pm", p.pm }, .{ "pl", p.pl } };
    for (named) |n| if (n[1] % 16 != 0) return n[0];
    return null;
}

// -- the engine's side --------------------------------------------------------------------------------------------------

/// Owner decode's counters (/health's kv_shard `owner_rounds`, `owner_bytes`): served rounds, and the bytes this rank's
/// exchanges sent in them.
pub const Stats = struct { rounds: u64 = 0, bytes: u64 = 0 };

/// What one served round of `rows` rows sends from a rank: a captured graph's two exchanges move every row's block each
/// DSA layer (`gather` or `swap`); outside one the crossing rows alone, each row once a layer (the queries of the rows
/// the peer owns, the peer's heads of the rows this rank owns).
pub fn roundBytes(rows: usize, captured: bool) usize {
    return G.dsa_layers * rows * row_bytes * @as(usize, if (captured) 2 else 1);
}

/// One rank's owner decode: its view of the parts, the kernels, buffers and side stream (all made before any graph
/// captures), and the round's state `round` sets.
pub const Decode = struct {
    sh: shard.Shard,
    comm: *const Comm,
    k: Kernels,
    b: Buffers,
    side: side_mod.Side,
    /// The own-head pass on the side stream (false: one stream, as Python; the same bits either way).
    overlap: bool = true,
    /// This round's exchanges: captured (a fixed block per R) or not (the crossing rows of `lists`).
    fixed: bool = true,
    lists: shard.Lists = .{},
    stats: Stats = .{},

    pub fn init(d: *const cuda.Driver, comm: *const Comm, sh: shard.Shard, rows: usize, tables: *const seg.SegRows) !Decode {
        if (sh.world != 2 or comm.world != 2 or comm.rank != sh.rank) return error.KvShardNeedsTwoRanks;
        if (rows == 0 or rows > shard.max_rows or rows > sh.trash or tables.rows != rows or tables.shard == null) return error.BadWindow;
        var b = try Buffers.init(d, rows);
        errdefer b.deinit();
        if (try offClass(tables, b.p)) |what| {
            std.log.err("TF_GLM_KV_SHARD: the decode's {s} ({d} rows, {d} segments) is not 16-byte aligned: the captured kernels have no variant for it", .{ what, tables.rows, tables.max_segs });
            return error.KvShardVariantClass;
        }
        var k = try Kernels.load(d, cuda.kernels.glm_shard);
        errdefer k.unload();
        return .{ .sh = sh, .comm = comm, .k = k, .b = b, .side = try side_mod.Side.init(d) };
    }

    pub fn deinit(dec: *Decode) void {
        dec.side.deinit();
        dec.k.unload();
        dec.b.deinit();
    }

    /// A window's start (Forward.compute, after its tables were staged): whether `s` is being captured (both ranks
    /// capture the same windows, so they agree), and the window's crossing rows from the host table's owners.
    pub fn round(dec: *Decode, tables: *const seg.SegRows, s: cuda.Stream) !void {
        dec.fixed = switch (try cuda.graph.captureStatus(s)) {
            .none => false,
            .active => true,
            else => return error.CaptureInvalidated,
        };
        dec.lists = try shard.lists(tables.plane(9 + @as(usize, @backingInt(shard.Glue.orank)))[0..tables.used], dec.sh.rank);
    }

    /// A served round of `rows` rows (cuda_verify's forward: its captured graph, or uncaptured), counted.
    pub fn tally(dec: *Decode, rows: usize, captured: bool) void {
        dec.stats.rounds += 1;
        dec.stats.bytes += roundBytes(rows, captured);
    }

    /// One DSA layer's attention of the window (`run`), this rank's heads of every row into `a.out`.
    pub fn attend(dec: *const Decode, t: Tri, a: Attn) !void {
        if (a.rows > dec.b.rows or a.rank != dec.sh.rank) return error.BadRows;
        var b = a;
        b.overlap = dec.overlap;
        var dev: Device = .{ .dec = dec };
        try run(&dev, t, b, dec.b.p);
    }
};

/// `run`'s device on the GPU.
const Device = struct {
    dec: *const Decode,
    fn counts(dv: *Device, s: cuda.Stream, in: u64, own: u64, out: u64, rows: usize) !void {
        try dv.dec.k.counts(s, in, own, out, rows);
    }
    fn select(dv: *Device, s: cuda.Stream, recv: u64, out: u64, own: u64, rows: usize) !void {
        try dv.dec.k.select(s, recv, out, own, rows, row_bytes);
    }
    fn fork(dv: *Device, t: Tri) !Tri {
        const pair = &dv.dec.side.pair;
        try pair.fork(t.s);
        var u = t;
        u.s = pair.stream;
        return u;
    }
    fn join(dv: *Device, t: Tri) !void {
        try dv.dec.side.pair.join(t.s);
    }
    fn exchange(dv: *Device, x: shard.Exchange, send: u64, recv: u64, rows: usize, s: cuda.Stream) !void {
        const dec = dv.dec;
        const peer = dec.sh.peer();
        const p = plan(x, send, recv, rows, peer, dec.fixed, dec.comm.roceFits(rows * row_bytes), &dec.lists);
        switch (p.way) {
            .gather => try dec.comm.allGather(p.send, p.recv, p.count, .bf16, s.handle),
            .swap => try dec.comm.exchange(@intCast(peer), p.send, p.recv, p.count, .bf16, s.handle),
            .rows => try dec.comm.exchangeBytes(@intCast(peer), p.out.slice(), p.in.slice(), s.handle),
        }
    }
};

// -- mutants: each rule above turned the wrong way, in test builds only (cuda_shard_decode_test.zig kills them) --------

pub const Mutant = enum {
    none,
    peer_block_mine, // the peer-head pass reads this rank's own queries as the peer's
    back_mine, // the select takes this rank's own block of the returned outputs
    counts_unmasked, // a row it does not own reads its tokens at the trash rows
    out_before_peer, // the outputs go back before the peer-head pass wrote them
    select_before_join, // the select runs before the own-head pass joined
};
pub var mutant: Mutant = .none;
inline fn mut(m: Mutant) bool {
    return builtin.is_test and mutant == m;
}
