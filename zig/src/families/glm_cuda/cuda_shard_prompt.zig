//! TF_GLM_KV_SHARD's prompt path (patch 0118, glm5_next/cuda/forward.py `_sharded_attend` / `_shard_sparse`, shard.py
//! `history` / `fetch`, two ranks): a prompt chunk's DSA attention when its stream's latent rows live on one rank only
//! (cuda_shard.zig). The selection runs replicated (the pooled index keys are on both ranks). The owner then sends the
//! other rank the history rows the chunk's tokens name, once each and in position order (marked, scanned, compacted:
//! glm_shard.cu), a bounded piece at a time; the other rank writes the chunk's own latents after them and attends over
//! that compact copy with its tokens renumbered in place, which reads the same bytes in the same order: the same bits.
//! A chunk with rows under the dense limit fetches its whole history at its positions instead (those rows read every
//! earlier position). The attention launches are the unsharded chunk's (`_lat_write`, `_dense_chunks` + `_merge`,
//! `_sparse_onepass`) with LC moved to the copy: no new Triton variant. The copy is made once and bounded
//! (`historyRows`): a chunk whose union outgrows it goes in row blocks, each with its own union (counted). Prompt
//! chunks are never captured, so a union's count is read on the host: one stream synchronize a layer (Python's
//! `int(cum[-1])` / `nonzero` did the same).
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const shard = @import("cuda_shard.zig");
const state = @import("cuda_state.zig");
const grid = @import("cuda_prefill_grid.zig");
const Comm = @import("cuda_comm.zig").Comm;
const G = @import("cuda_geometry.zig").Geometry;

/// A latent row (FP8: 512 values and their scale) and a chunk row's tokens.
const row = G.latent_row;
const width = G.tokens;

/// glm_shard.cu's scan segments (GLM_SHARD_SCAN_SEGS): `sums` holds this many int32 and the total after them.
pub const scan_segs = 1024;
/// A context this short holds its whole history in the copy (no chunk ever blocks); a longer one half of it.
pub const whole_rows: usize = 262144;
/// The row blocks a chunk is cut into when its union outgrows the copy are whole tiles of this many rows: their tokens,
/// counts and queries stay 16-byte aligned (`_sparse_onepass`'s one captured variant), and the positions 64 rows name
/// (64 x 2,051) fit every copy, so the halving always ends.
pub const block_rows: usize = 64;

comptime {
    std.debug.assert(block_rows * width <= whole_rows);
    std.debug.assert(row % 16 == 0 and (block_rows * width * 4) % 16 == 0);
}

/// History rows the copy holds before a chunk's own, at a context of `context` tokens: all of it up to `whole_rows`,
/// else half (Python measured ~43% of a 1M history in one 2,048-row chunk's union, ~230 MB a layer).
pub fn historyRows(context: usize) usize {
    return if (context <= whole_rows) context else @max(whole_rows, context / 2);
}

// -- buffers ----------------------------------------------------------------------------------------------------------

/// The fetch's buffers, made once at start (the sharded pool's sizing counts them): the copy (`historyRows` rows, then a
/// chunk's own), the marks (one a position of the context, scanned in place into cum), the scan's sums and total, and a
/// zero word (the POS of the chunk's own rows in the copy, which start after the fetched rows).
pub const Ptrs = struct { copy: u64, mark: u64, sums: u64, zero: u64 };

pub fn layout(context: usize, chunk: usize) [4]usize {
    return .{ (historyRows(context) + chunk) * row, context * 4, (scan_segs + 1) * 4, 16 };
}

/// Bytes `Buffers` takes (~269 MiB at 1M tokens and 2,048-row chunks, ~68 MiB at 128K).
pub fn bytes(context: usize, chunk: usize) usize {
    return state.aligned(&layout(context, chunk));
}

pub const Buffers = struct {
    arena: state.Arena,
    p: Ptrs,
    pub fn init(d: *const cuda.Driver, context: usize, chunk: usize) !Buffers {
        const sizes = layout(context, chunk);
        var a = try state.Arena.init(d, &sizes); // zeroed: the zero word stays 0
        errdefer a.deinit();
        var at: [4]u64 = undefined;
        for (&at, sizes) |*p, n| p.* = a.take(n);
        return .{ .arena = a, .p = .{ .copy = at[0], .mark = at[1], .sums = at[2], .zero = at[3] } };
    }
    pub fn deinit(b: *Buffers) void {
        b.arena.deinit();
    }
};

// -- glm_shard.cu --------------------------------------------------------------------------------------------------------

pub const Kernels = struct {
    module: cuda.Module,
    mark_fn: cuda.Function,
    sums_fn: cuda.Function,
    offsets_fn: cuda.Function,
    scan_fn: cuda.Function,
    compact_fn: cuda.Function,
    renumber_fn: cuda.Function,
    /// glm_shard.cu's image (cuda.kernels.glm_shard; empty in a host-only build).
    pub fn load(d: *const cuda.Driver, image: []const u8) !Kernels {
        if (image.len == 0) return error.NoKvShardKernels;
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        return .{
            .module = m,
            .mark_fn = try m.function("glm_shard_mark"),
            .sums_fn = try m.function("glm_shard_sums"),
            .offsets_fn = try m.function("glm_shard_offsets"),
            .scan_fn = try m.function("glm_shard_scan"),
            .compact_fn = try m.function("glm_shard_compact"),
            .renumber_fn = try m.function("glm_shard_renumber"),
        };
    }
    pub fn unload(k: *Kernels) void {
        k.module.unload();
    }
    /// The marks [limit] zeroed, then a one at every token below `limit` of tokens [n] (cuda_shard.markHistory).
    pub fn mark(k: *const Kernels, s: cuda.Stream, tokens: u64, n: usize, m: u64, limit: usize) !void {
        if (limit > 0) try s.d.check(s.d.api.cuMemsetD32Async(m, 0, limit, s.handle), "cuMemsetD32Async");
        if (n == 0 or limit == 0) return;
        var a: cuda.Args = .{};
        a.add(tokens);
        a.add(@as(i64, @intCast(n)));
        a.add(m);
        a.add(try int(limit));
        try cuda.launch.launch(k.mark_fn, .{ .grid = .{ .x = try blocks(n, 256) }, .block = .{ .x = 256 } }, s, &a);
    }
    /// m [n] scanned in place (cuda_shard.prefixSum), its total to sums[scan_segs].
    pub fn scan(k: *const Kernels, s: cuda.Stream, m: u64, n: usize, sums: u64) !void {
        var a: cuda.Args = .{};
        a.add(m);
        a.add(@as(i64, @intCast(n)));
        a.add(sums);
        try cuda.launch.launch(k.sums_fn, .{ .grid = .{ .x = scan_segs / 256 }, .block = .{ .x = 256 } }, s, &a);
        var o: cuda.Args = .{};
        o.add(sums);
        try cuda.launch.launch(k.offsets_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } }, s, &o);
        try cuda.launch.launch(k.scan_fn, .{ .grid = .{ .x = scan_segs / 256 }, .block = .{ .x = 256 } }, s, &a);
    }
    /// Latent row t of `src` to row cum[t] - 1 of `dst` where cum steps, for t below `limit` (cum 0: every row t to row
    /// t; `compactRef`).
    pub fn compact(k: *const Kernels, s: cuda.Stream, src: u64, cum: u64, limit: usize, dst: u64) !void {
        if (limit == 0) return;
        if ((src | dst) % 16 != 0) return error.BadCompact;
        var a: cuda.Args = .{};
        for ([_]u64{ src, cum }) |x| a.add(x);
        a.add(@as(i64, @intCast(limit)));
        a.add(@as(i64, row / 16));
        a.add(dst);
        try cuda.launch.launch(k.compact_fn, .{ .grid = .{ .x = try blocks(limit, 16) }, .block = .{ .x = 256 } }, s, &a);
    }
    /// tokens [n] renumbered in place to rows of the copy (cuda_shard.renumber).
    pub fn renumber(k: *const Kernels, s: cuda.Stream, tokens: u64, n: usize, cum: u64, limit: usize) !void {
        if (n == 0) return;
        var a: cuda.Args = .{};
        a.add(tokens);
        a.add(@as(i64, @intCast(n)));
        a.add(cum);
        a.add(try int(limit));
        try cuda.launch.launch(k.renumber_fn, .{ .grid = .{ .x = try blocks(n, 256) }, .block = .{ .x = 256 } }, s, &a);
    }
};

fn int(n: usize) !c_int {
    if (n > std.math.maxInt(c_int)) return error.BadRows;
    return @intCast(n);
}
fn blocks(n: usize, per: usize) !u32 {
    const b = (n + per - 1) / per;
    if (b > std.math.maxInt(i32)) return error.BadRows;
    return @intCast(b);
}

/// glm_shard_compact as host code (tools/glm_shard_emulate holds the kernel to it): row t of `plane` to row cum[t] - 1
/// of `out` where cum steps, for t below `limit`; cum null: every row t to row t.
pub fn compactRef(plane: []const u8, cum: ?[]const i32, limit: usize, out: []u8) void {
    for (0..limit) |t| {
        var d = t;
        if (cum) |c| {
            const prev: i32 = if (t > 0) c[t - 1] else 0;
            if (c[t] == prev) continue;
            d = @intCast(prev);
        }
        @memcpy(out[d * row ..][0..row], plane[t * row ..][0..row]);
    }
}

// -- one chunk's DSA layer --------------------------------------------------------------------------------------------

/// This rank's side of a sharded chunk: the stream's owner sends its rows, the other rank receives them and attends over
/// the copy.
pub const Role = enum { send, recv };

/// A rank's side of a chunk on cache view `v` (cuda_state.CacheView): the owner's view sends, a view with no latent
/// rows here receives; null on both ranks while the prompt borrows a twin (TF_GLM_KV_SHARD_TWINS: the view marked
/// `twin`, the owner's own rows and the other rank's twin), which runs the chunk unsharded: each rank writes and
/// attends its own copy, nothing is sent. Another rank's rows here without the mark are no view this engine makes.
pub fn roleOf(v: state.CacheView) !?Role {
    if (v.twin and !mut(.twin_fetches)) return null;
    if (v.owner) return .send;
    if (!v.hasLatent()) return .recv;
    return error.KvShardTwinPrompt;
}

/// The fetch's counters, this rank's (rank 0's go to kv_parts' prompt_*: it takes part in every fetch).
pub const Stats = struct {
    fetches: u64 = 0, // chunk layers that moved history rows (sent or received)
    rows: u64 = 0, // latent rows moved
    bytes: u64 = 0,
    syncs: u64 = 0, // host reads of a union's count
    blocked: u64 = 0, // chunk layers whose union outgrew the copy and went in row blocks
    blocks: u64 = 0, // the row blocks those ran
};

/// The copy's bounds: `hist` history rows before a chunk's own (`historyRows`), `chunk` rows a chunk at most, and the
/// rows a transfer carries (`piece`: cuda_shard.piece, 64 MiB; both ranks cut alike, so each send meets its receive).
pub const Limits = struct { hist: usize, chunk: usize, piece: usize = shard.piece(row) };

/// A chunk's DSA layer after its selection: `rows` rows from position `pos`; `sparse` when rows past the dense limit
/// attend to their tokens ([rows][2,051] int32 at `tokens`, -1 none; renumbered in place on the receiving rank); `lc`
/// the sending rank's view of the stream's latent rows, the chunk's own already written.
pub const Chunk = struct { role: Role, rows: usize, pos: usize, sparse: bool, tokens: u64, lc: u64 = 0 };

/// forward.py `_sharded_attend` for two ranks. `dev` is the device's side: glm_shard.cu (mark, scan, count, compact,
/// renumber), the transfers (send, recv, inside hold / release), and the chunk's attention launches (latWrite, dense,
/// sparse); the GPU's in cuda_prefill.zig, the tests' model in cuda_shard_prompt_test.zig.
/// A chunk with rows under the dense limit fetches the history [0, pos) whole at its positions, and each rank attends as
/// unsharded over its view or the copy. Otherwise both ranks mark the positions below `pos` the tokens name and read
/// their count; the sender compacts and sends those rows; the receiver writes the chunk's own rows after them, renumbers
/// its tokens and attends over the copy; the sender attends over its own view after its sends. A union past
/// `lim.hist` rows is cut in halves (whole `block_rows` tiles) until each fits; both ranks read the same counts (the
/// tokens are replicated), so they cut alike.
pub fn attend(dev: anytype, p: Ptrs, lim: Limits, c: Chunk, st: *Stats) !void {
    if (c.rows == 0 or c.rows > lim.chunk or lim.piece == 0) return error.BadRows;
    const send = c.role == .send;
    if (send and c.lc == 0) return error.NoLatentRows;
    // what the attention reads: the sender its own view, the receiver the copy
    const att = if (send and !mut(.owner_reads_copy)) c.lc else p.copy;
    var moved: usize = 0;
    defer st.fetches += @intFromBool(moved > 0);
    if (c.pos < grid.sparse_from) {
        // a row under the dense limit reads every earlier position: the history whole, at its positions
        if (c.pos > lim.hist) return error.KvShardFetchRows;
        const whole = send and c.pos > 0;
        if (whole and !mut(.send_before_compact)) try dev.compact(c.lc, 0, c.pos, p.copy);
        moved += try move(dev, send, p.copy, c.pos, lim.piece, st);
        if (whole and mut(.send_before_compact)) try dev.compact(c.lc, 0, c.pos, p.copy);
        if (!send) try dev.latWrite(p.copy, null);
        try dev.dense(att);
        if (c.sparse) try dev.sparse(att, 0, c.rows);
        return;
    }
    if (!c.sparse) return error.MissingIndex; // past the dense limit every row attends to its tokens
    var todo: [16][2]usize = undefined;
    todo[0] = .{ 0, c.rows };
    var left: usize = 1;
    var cut = false;
    while (left > 0) {
        left -= 1;
        const lo = todo[left][0];
        const hi = todo[left][1];
        const tok = c.tokens + lo * width * 4;
        const n_tok = (hi - lo) * width;
        if (mut(.block_marks_chunk)) try dev.mark(c.tokens, c.rows * width, p.mark, c.pos) else try dev.mark(tok, n_tok, p.mark, c.pos);
        try dev.scan(p.mark, c.pos, p.sums);
        const n = try dev.count(p.sums);
        st.syncs += 1;
        if (n > lim.hist) {
            // a union the copy cannot hold: the block in two (a 64-row tile names at most 64 x 2,051 positions)
            if (hi - lo <= block_rows or left + 2 > todo.len) return error.KvShardFetchRows;
            const half = if (mut(.cut_unaligned)) (hi - lo) / 2 else std.mem.alignForward(usize, (hi - lo) / 2, block_rows);
            todo[left] = .{ lo + half, hi };
            todo[left + 1] = .{ lo, lo + half };
            left += 2;
            cut = true;
            continue;
        }
        if (cut) st.blocks += 1;
        if (send and !mut(.send_before_compact)) try dev.compact(c.lc, p.mark, c.pos, p.copy);
        moved += try move(dev, send, p.copy, n, lim.piece, st);
        if (send) {
            if (mut(.send_before_compact)) try dev.compact(c.lc, p.mark, c.pos, p.copy);
            continue;
        }
        // the chunk's own rows after the fetched ones (rows of a block name positions up to its last row's)
        try dev.latWrite(if (mut(.own_rows_first)) p.copy else p.copy + n * row, p.zero);
        if (!mut(.skip_renumber)) try dev.renumber(tok, n_tok, p.mark, c.pos);
        try dev.sparse(p.copy, lo, hi);
    }
    if (cut) st.blocked += 1;
    // the owner attends after its sends: the other rank starts on its copy at once
    if (send) try dev.sparse(att, 0, c.rows);
}

/// `n` rows of the copy from the sender to the receiver, `piece` rows a transfer, held against the split's side stream
/// (shard.py `_sharded_attend`: the fetch waits for that stream and it for the fetch, so two collectives of one
/// communicator never run at once). The rows moved.
fn move(dev: anytype, send: bool, at: u64, n: usize, piece: usize, st: *Stats) !usize {
    if (n == 0) return 0;
    if (!mut(.no_hold)) try dev.hold();
    var i: usize = 0;
    while (i < n) : (i += piece) {
        const k = @min(piece, n - i);
        if (send) try dev.send(at + i * row, k * row) else try dev.recv(at + i * row, k * row);
    }
    if (!mut(.no_hold)) try dev.release();
    st.rows += n;
    st.bytes += n * row;
    return n;
}

// -- the engine's side --------------------------------------------------------------------------------------------------

/// One rank's prompt fetch: its view of the parts, the kernels and buffers (made at start), and the counters. The prompt
/// engine's `shard` points here (cuda_prefill.Engine); its device is cuda_prefill's ShardDev.
pub const Fetch = struct {
    sh: shard.Shard,
    comm: *const Comm,
    k: Kernels,
    b: Buffers,
    lim: Limits,
    stats: Stats = .{},

    pub fn init(d: *const cuda.Driver, comm: *const Comm, sh: shard.Shard, context: usize, chunk: usize) !Fetch {
        if (sh.world != 2 or comm.world != 2 or comm.rank != sh.rank) return error.KvShardNeedsTwoRanks;
        if (context == 0 or chunk == 0) return error.BadRows;
        var b = try Buffers.init(d, context, chunk);
        errdefer b.deinit();
        var k = try Kernels.load(d, cuda.kernels.glm_shard);
        errdefer k.unload();
        return .{ .sh = sh, .comm = comm, .k = k, .b = b, .lim = .{ .hist = historyRows(context), .chunk = chunk } };
    }

    pub fn deinit(f: *Fetch) void {
        f.k.unload();
        f.b.deinit();
    }
};

// -- mutants: each rule above turned the wrong way, in test builds only (cuda_shard_prompt_test.zig kills them) --------

pub const Mutant = enum {
    none,
    skip_renumber, // the receiver attends with the tokens' history positions
    own_rows_first, // the chunk's own rows written over the copy's first rows
    send_before_compact, // the sender sends its copy before compacting the rows into it
    owner_reads_copy, // the sender attends over its copy, not its view
    no_hold, // a transfer not held against the split's side stream
    cut_unaligned, // a row block off the 64-row tiles (off `_sparse_onepass`'s variant)
    block_marks_chunk, // every row block marks the whole chunk's tokens
    twin_fetches, // a borrowing prompt's chunk fetches as if it had no twin (its owner then sends rows the other has)
};
pub var mutant: Mutant = .none;
inline fn mut(m: Mutant) bool {
    return builtin.is_test and mutant == m;
}
