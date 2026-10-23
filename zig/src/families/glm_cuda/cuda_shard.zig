//! TF_GLM_KV_SHARD (patch 0118, glm5_next/cuda/shard.py): the latent cache stored once over the two ranks, host side.
//! The pool is cut in two parts, one a rank (cuda_pool.Pool.initParts): an extent lives in one part, and that part's
//! rank, its owner, holds its rows of the 11 DSA layers' latent planes; a rank's planes are a part long plus `trash`
//! rows, where a decode row it does not own writes. The pooled index keys (36 B a token) stay on both ranks over the
//! whole pool, so the token selection runs replicated; KDA states, conv windows and index rings are a slot's, not the
//! pool's. Decode: each row's owner attends all 64 heads over its rows (`glue`, `lists`). A prompt chunk: the owner
//! sends the history rows the chunk selects and the other rank attends over a compact copy, its token ids renumbered
//! (`markHistory` .. `renumber`). Same keys in the same order, so the same bits as unsharded.
//! Arithmetic only: sizes, addresses, tables and host models of the device steps (no CUDA).
const std = @import("std");
const builtin = @import("builtin");
const G = @import("cuda_geometry.zig").Geometry;
const Pool = @import("cuda_pool.zig").Pool;

/// Latent rows past a part: one a decode window's row (Engine.window), where a row this rank does not own writes.
pub const trash: usize = 64;
/// A part smaller than this (or than a quarter of the context) is refused: the server would serve a fraction of it.
pub const min_part: usize = 131072;
/// The owner's staging: history rows travel in pieces of this many MiB.
pub const send_mib: usize = 64;
/// A latent row (FP8), and a token's share of a pooled-key row, a DSA layer.
pub const latent_row = G.latent_row;
pub const pooled_bytes = G.pool_row / G.pool;

// -- switches -------------------------------------------------------------------------------------------------------

/// Whether this build carries sharding's device paths: owner decode (cuda_shard_decode.zig), the prompt's history
/// fetch (cuda_shard_prompt.zig), prompt borrowing and kept states' carries (cuda_shard_keep.zig). Without them
/// TF_GLM_KV_SHARD=1 is refused at start, after the ranks agree on it: nothing half-works.
pub const gpu_paths = true;

/// TF_GLM_KV_SHARD ("1": on) and its knobs, as every rank reads them (cuda_native agrees them over the settings
/// frame): TF_GLM_KV_SHARD_CARRY, a kept state's latent rows carried to the other part when its own has no room for the
/// resume, which would otherwise recompute (on unless "0"; cuda_shard_keep.zig); TF_GLM_KV_SHARD_TWINS, a filling prompt
/// borrowing free rows of the other part (on unless "0"; prompts only, no decode twins); TF_GLM_KV_SHARD_CHECK, decode rounds between the ranks' layout digests
/// (DS TF_DS_KV_SHARD_CHECK: `check_rounds` unless set; a digest also rides each placement; 0: none, each placement
/// still checked). TF_GLM_MULTI_LONE ("1") is read only to be refused. Off: the knobs are not read, and nothing
/// changes. TF_GLM_SPILL_V2 is not read here: spill v2's admission refuses it with sharding, and only with its tier on
/// (--spill-gib and --snapshot-dir), where it would store a kept state that does not name its part.
pub const Switches = struct {
    on: bool = false,
    carry: bool = true,
    twins: bool = true,
    check: u32 = check_rounds,
    lone: bool = false,

    pub fn parse(on: ?[]const u8, carry: ?[]const u8, twins: ?[]const u8, check: ?[]const u8, lone: ?[]const u8) !Switches {
        if (!try flag(on, false, error.BadKvShard)) return .{};
        return .{
            .on = true,
            .carry = try flag(carry, true, error.BadKvShardCarry),
            .twins = try flag(twins, true, error.BadKvShardTwins),
            .check = try count(check, check_rounds, error.BadKvShardCheck),
            .lone = std.mem.eql(u8, std.mem.trim(u8, lone orelse "", " \t\r\n"), "1"),
        };
    }

    /// Whether decode round `round` (from 1) ends with a layout digest: every `check` rounds (0: never).
    pub fn digestAt(s: Switches, round: u64) bool {
        return s.on and s.check != 0 and round != 0 and round % s.check == 0;
    }

    /// This process's environment.
    pub fn read() !Switches {
        return parse(env("TF_GLM_KV_SHARD"), env("TF_GLM_KV_SHARD_CARRY"), env("TF_GLM_KV_SHARD_TWINS"), env("TF_GLM_KV_SHARD_CHECK"), env("TF_GLM_MULTI_LONE"));
    }

    /// The settings text's words (cuda_native.settings, hashed and compared over the link): none when off, so the text
    /// stays as it was; a rank that cannot read its switches sends `unreadable`, so the other rank refuses with it.
    pub fn text(s: Switches, buf: *[80]u8) []const u8 {
        if (!s.on) return "";
        return std.fmt.bufPrint(buf, " kv_shard=1 carry={d} twins={d} check={d} lone={d}", .{ @intFromBool(s.carry), @intFromBool(s.twins), s.check, @intFromBool(s.lone) }) catch unreachable;
    }
};

/// TF_GLM_KV_SHARD_CHECK's default: decode rounds between layout digests (DS's 1024).
pub const check_rounds: u32 = 1024;

/// The settings words of a rank whose TF_GLM_KV_SHARD* it cannot read (DS's sentinel): unlike any rank's real ones.
pub const unreadable = " kv_shard=unreadable";

/// "0" or "1" (blank or unset: `default`), else `bad`.
fn flag(raw: ?[]const u8, default: bool, bad: anyerror) !bool {
    const t = std.mem.trim(u8, raw orelse "", " \t\r\n");
    if (t.len == 0) return default;
    if (std.mem.eql(u8, t, "1")) return true;
    if (std.mem.eql(u8, t, "0")) return false;
    return bad;
}

/// Decimal digits only, a u32 (blank or unset: `default`), else `bad`.
fn count(raw: ?[]const u8, default: u32, bad: anyerror) !u32 {
    const t = std.mem.trim(u8, raw orelse "", " \t\r\n");
    if (t.len == 0) return default;
    for (t) |c| if (!std.ascii.isDigit(c)) return bad;
    return std.fmt.parseInt(u32, t, 10) catch bad;
}

fn env(name: [:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

/// The settings sharding cannot serve, refused at start (Python engine.py's ValueError), alike on every rank (each
/// input is in the agreed settings text): two ranks only (a part a rank), more than one stream (a stream lives in one
/// part), no MTP head (its caches are its own, not the pool's), no TF_GLM_MULTI_LONE.
pub fn refuse(s: Switches, world: u32, parallel: u32, mtp: bool) !void {
    if (!s.on) return;
    if (world != 2) return error.KvShardNeedsTwoRanks;
    if (parallel < 2) return error.KvShardNeedsParallel;
    if (mtp) return error.KvShardWithMtp;
    if (s.lone) return error.KvShardWithLone;
}

/// glm_shard.cu's image in a build that embeds kernels (`images`; a host-only build embeds none and launches nothing).
/// It is optional, so a prebuilt fatbin set from before it still builds a server that runs unsharded; TF_GLM_KV_SHARD=1
/// is then refused at start (one binary: every rank alike), not at the first load of its kernels.
pub fn kernelsIn(images: bool, glm_shard: []const u8) !void {
    if (images and glm_shard.len == 0) return error.NoKvShardKernels;
}

/// The start-up's words for a refusal of TF_GLM_KV_SHARD (null: not one of its errors).
pub fn why(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.BadKvShard => "TF_GLM_KV_SHARD: 0 (off, the default) or 1",
        error.BadKvShardCarry => "TF_GLM_KV_SHARD_CARRY: 1 (the default) or 0",
        error.BadKvShardTwins => "TF_GLM_KV_SHARD_TWINS: 1 (the default) or 0",
        error.BadKvShardCheck => "TF_GLM_KV_SHARD_CHECK: decode rounds between the ranks' layout digests (1024, the default; 0: none)",
        error.KvShardNeedsTwoRanks => "TF_GLM_KV_SHARD=1 needs two ranks (--tp 2)",
        error.KvShardNeedsParallel => "TF_GLM_KV_SHARD=1 needs --parallel above 1",
        error.KvShardWithMtp => "TF_GLM_KV_SHARD=1 does not run with the MTP drafter (DRAFTER=mtp)",
        error.KvShardWithLone => "TF_GLM_KV_SHARD=1 does not run with TF_GLM_MULTI_LONE=1",
        error.KvShardNotBuilt => "TF_GLM_KV_SHARD=1: KV sharding's GPU paths are not in this build yet",
        error.NoKvShardKernels => "TF_GLM_KV_SHARD=1: this build has no glm_shard kernels (its fatbins came without glm_shard.fatbin)",
        else => null,
    };
}

// -- sizes ----------------------------------------------------------------------------------------------------------

/// The sharded pool's rows in the memory `rows` unsharded rows' DSA planes take: each token's latent once over `world`
/// ranks, its pooled keys on every rank, `trash_rows` latent rows a rank; whole parts of extents (0: the memory does
/// not hold the trash rows).
pub fn budget(rows: usize, world: usize, trash_rows: usize) usize {
    const unit = world * G.extent;
    const mem = rows * G.dsa_layers * (latent_row + pooled_bytes);
    const t = if (mut(.budget_no_trash)) 0 else trash_rows * G.dsa_layers * latent_row;
    if (mem <= t) return 0;
    const n = (mem - t) * world / (G.dsa_layers * (latent_row + pooled_bytes * world));
    return if (mut(.budget_unaligned)) n else n / unit * unit;
}

/// The start-up's sharded pool: `rows`, its `part`s, and the longest request a part holds (`context`: the one asked
/// for, or less).
pub const Plan = struct { rows: usize, part: usize, context: usize };

/// The sharded pool in the memory of `unsharded` rows (`budget`, never more). A stream lives in one part, so a part
/// short of the requested `context` plus a decode window (`window` rows) shortens what a request may use; a part
/// below `min_part` or a quarter of the context is refused.
pub fn plan(unsharded: usize, context: usize, window: usize, world: usize, trash_rows: usize) !Plan {
    return fit(budget(unsharded, world, trash_rows), context, window, world);
}

/// `plan` of a sharded pool of `rows` (whole parts): the longest request a part holds, or PartTooSmall.
pub fn fit(rows: usize, context: usize, window: usize, world: usize) !Plan {
    const part = rows / world;
    const fits = @min(context, part -| window);
    if (fits >= context or mut(.plan_no_shrink)) return .{ .rows = rows, .part = part, .context = context };
    if (fits < 1 or (part < @max(min_part, context / 4) and !mut(.plan_no_floor))) return error.PartTooSmall;
    return .{ .rows = rows, .part = part, .context = fits };
}

/// (pool rows, display rows) every rank uses: the smallest of each. A part's owner is base / part, so every rank
/// must cut the same parts, and each sized its pool from its own memory.
pub fn agree(proposals: []const [2]u64) [2]u64 {
    var a: [2]u64 = if (mut(.agree_max)) .{ 0, 0 } else .{ std.math.maxInt(u64), std.math.maxInt(u64) };
    for (proposals) |p| {
        for (0..2) |i| a[i] = if (mut(.agree_max)) @max(a[i], p[i]) else @min(a[i], p[i]);
    }
    return a;
}

/// Whether every rank's word is the same (the sharded pool each rank sized from the agreed unsharded one).
pub fn same(words: []const u64) bool {
    for (words) |w| if (w != words[0]) return false;
    return true;
}

/// The display reservation's rows at a sharded token's cost, never past the pool: the sharded `rows` less those the
/// memory of the unsharded pool without the reservation (`unsharded - display`) holds.
pub fn displayRows(rows: usize, unsharded: usize, display: usize, world: usize, trash_rows: usize) usize {
    return rows -| budget(unsharded -| display, world, trash_rows);
}

/// Bytes a carry of `n` tokens sends from the source's owner to the destination's: the latent rows of every DSA layer.
pub fn carryBytes(n: usize) usize {
    return n * latent_row * G.dsa_layers;
}

// -- addresses ------------------------------------------------------------------------------------------------------

/// This rank's part of copying an extent's latent rows: within a part its owner copies (`local`); across parts the
/// source's owner sends and the destination's receives, in the same order (`send`, `recv`). Latent rows of this rank's
/// planes. The pooled keys are copied on every rank (`pooledRows`).
pub const Move = union(enum) {
    none,
    local: struct { from: usize, to: usize, rows: usize },
    send: struct { from: usize, rows: usize, peer: usize },
    recv: struct { to: usize, rows: usize, peer: usize },
};

/// Pooled-key rows of tokens [from, from + n) copied to [to, to + n) on every rank: a last row part full goes whole.
pub const Span = struct { from: usize, to: usize, rows: usize };
pub fn pooledRows(from: usize, to: usize, n: usize) Span {
    return .{ .from = from / G.pool, .to = to / G.pool, .rows = (n + G.pool - 1) / G.pool };
}

/// One rank's view of the parts: every rank holds the same numbers but its own `rank`.
pub const Shard = struct {
    rank: usize,
    world: usize,
    pool: usize, // rows of every part
    part: usize, // rows a part
    trash: usize, // latent rows past a part

    /// TP2 only; a pool of whole extents a part.
    pub fn init(rank: usize, world: usize, pool: usize, trash_rows: usize) !Shard {
        if (world != 2) return error.KvShardNeedsTwoRanks;
        if (rank >= world) return error.BadRank;
        if (pool == 0 or pool % (world * G.extent) != 0) return error.PoolNotWholeParts;
        if (trash_rows == 0) return error.BadTrash;
        return .{ .rank = rank, .world = world, .pool = pool, .part = pool / world, .trash = trash_rows };
    }
    /// The rank holding the latent rows of pool row `row`.
    pub fn owner(s: Shard, row: usize) usize {
        return row / s.part;
    }
    pub fn owns(s: Shard, row: usize) bool {
        return s.owner(row) == s.rank;
    }
    /// The peer (TP2).
    pub fn peer(s: Shard) usize {
        return 1 - s.rank;
    }
    /// Pool row `row` of this rank's part as a row of its latent planes.
    pub fn local(s: Shard, row: usize) usize {
        std.debug.assert(s.owns(row));
        return if (mut(.local_global)) row else row - s.rank * s.part;
    }
    /// Rows of a rank's latent planes: its part's, then the trash rows.
    pub fn latentRows(s: Shard) usize {
        return s.part + s.trash;
    }
    /// The trash row a decode window's row `i` writes to on the rank that does not own it.
    pub fn trashRow(s: Shard, i: usize) usize {
        std.debug.assert(i < s.trash);
        return s.part + i;
    }
    /// The latent row an extent at pool row `base` starts at on this rank (null: the other rank holds it).
    pub fn latentBase(s: Shard, base: usize) ?usize {
        return if (s.owns(base)) s.local(base) else null;
    }
    /// Whether [base, base + n) lies in one part.
    pub fn within(s: Shard, base: usize, n: usize) bool {
        if (base >= s.pool or n > s.pool - base) return false;
        return n == 0 or s.owner(base) == s.owner(base + n - 1);
    }
    /// Tokens [from, from + n) of an extent to [to, to + n) of its part (pool.py Arena.copy): an extent never changes
    /// part this way.
    pub fn copy(s: Shard, from: usize, to: usize, n: usize) !Move {
        if (!s.within(from, n) or !s.within(to, n)) return error.CrossesPart;
        if (n == 0) return .none;
        if (s.owner(from) != s.owner(to)) return if (mut(.copy_across)) .none else error.OtherPart;
        if (!s.owns(from)) return .none;
        return .{ .local = .{ .from = s.local(from), .to = s.local(to), .rows = n } };
    }
    /// `copy` into another part (Arena.carry): the source's owner sends, the destination's receives; the source stays.
    pub fn carry(s: Shard, from: usize, to: usize, n: usize) !Move {
        if (!s.within(from, n) or !s.within(to, n)) return error.CrossesPart;
        if (n == 0) return .none;
        const so = s.owner(from);
        const do = s.owner(to);
        if (so == do) return s.copy(from, to, n);
        const rows = if (mut(.carry_rows_minus_1)) n - 1 else n;
        if (so == s.rank) return .{ .send = .{ .from = s.local(from), .rows = rows, .peer = do } };
        if (do == s.rank) return .{ .recv = .{ .to = s.local(to), .rows = rows, .peer = so } };
        return .none;
    }
};

// -- a decode window's glue ---------------------------------------------------------------------------------------

/// TF_GLM_KV_SHARD's row planes of a decode window's segment table (segments.py), after its five row and four
/// segment planes: a row's stream in this rank's latent planes or the trash rows (`lbase`), whether this rank owns it
/// (`own`), whether it reads no key here (`sparse_eff`: a row it does not own), where it writes (`wpos`: its position,
/// or its index in the window, its trash row), and its owner (`orank`).
pub const Glue = enum(u8) { lbase, own, sparse_eff, wpos, orank };

/// Where glue plane `g` starts in a table of `rows` rows and `segs` segments (int32 words).
pub fn glueOffset(rows: usize, segs: usize, g: Glue) usize {
    return 5 * rows + 4 * segs + @backingInt(g) * rows;
}

/// Rank `s.rank`'s glue of a window's first `used` rows from its segment table's row planes (SegRows planes 0, 1 and
/// 3: a row's position, its extent's base, whether it attends sparsely); the rest zero.
pub fn glue(s: Shard, used: usize, pos: []const i32, base: []const i32, sparse: []const i32, out: *const [5][]i32) !void {
    if (used > s.trash and !mut(.glue_no_trash_check)) return error.WindowPastTrash;
    if (pos.len < used or base.len < used or sparse.len < used) return error.BadRows;
    for (out) |o| {
        if (o.len < used) return error.BadRows;
        @memset(o, 0);
    }
    for (0..used) |r| {
        if (base[r] < 0 or @as(usize, @intCast(base[r])) >= s.pool) return error.BadRows;
        const b: usize = @intCast(base[r]);
        const own = s.owns(b);
        out[@backingInt(Glue.lbase)][r] = @intCast(if (own) s.local(b) else s.part);
        out[@backingInt(Glue.own)][r] = @intFromBool(own);
        out[@backingInt(Glue.sparse_eff)][r] = if (own or mut(.glue_reads_unowned)) sparse[r] else 1;
        out[@backingInt(Glue.wpos)][r] = if (own or mut(.glue_trash_pos)) pos[r] else @intCast(r);
        out[@backingInt(Glue.orank)][r] = @intCast(s.owner(b));
    }
}

// -- a prompt chunk's history fetch (shard.history, shard.fetch): the host model of the device steps -------------------

/// Marks every history position (below `host_pos`) the chunk's tokens name ([rows][G.tokens] int32, -1 none):
/// `marks` holds `host_pos` of them, 1 a position fetched.
pub fn markHistory(tokens: []const i32, host_pos: usize, marks: []i32) void {
    @memset(marks[0..host_pos], 0);
    for (tokens) |t| {
        if (t >= 0 and @as(usize, @intCast(t)) < host_pos) marks[@intCast(t)] = 1;
        if (t < 0 and host_pos > 0 and mut(.mark_padding)) marks[0] = 1;
    }
}

/// Their prefix sums (`cum[i]`: marked positions up to i); the rows fetched are cum[host_pos - 1].
pub fn prefixSum(marks: []const i32, cum: []i32) usize {
    var n: i32 = 0;
    for (marks, cum[0..marks.len]) |m, *c| {
        n += m;
        c.* = n;
    }
    return @intCast(n);
}

/// The owner's: the marked positions, ascending (what it sends).
pub fn historyRows(marks: []const i32, out: []u32) []u32 {
    var n: usize = 0;
    for (marks, 0..) |m, i| {
        if (m == 0) continue;
        out[n] = @intCast(i);
        n += 1;
    }
    return out[0..n];
}

/// The other rank's: the tokens as rows of the compact copy, a history position its place among the fetched rows, a
/// chunk row's own (at or past `host_pos`) after them, -1 kept: the ids a sort and a search would give.
pub fn renumber(tokens: []const i32, host_pos: usize, cum: []const i32, out: []i32) void {
    const fetched: i32 = if (host_pos == 0 or mut(.own_rows_first)) 0 else cum[host_pos - 1];
    const late: i32 = @intFromBool(mut(.renumber_off_by_one));
    for (tokens, out[0..tokens.len]) |t, *o| {
        if (t < 0) {
            o.* = -1;
        } else if (@as(usize, @intCast(t)) < host_pos) {
            o.* = cum[@intCast(t)] - 1 + late;
        } else {
            o.* = fetched + t - @as(i32, @intCast(host_pos));
        }
    }
}

/// The owner's send buffer: its latent rows `hist` of a stream (`plane`, rows of `row` bytes) in order.
pub fn gatherRows(plane: []const u8, row: usize, hist: []const u32, out: []u8) void {
    for (hist, 0..) |h, i| @memcpy(out[i * row ..][0..row], plane[@as(usize, h) * row ..][0..row]);
}

/// Rows a send carries (both ranks cut alike, so each send meets its receive).
pub fn piece(row: usize) usize {
    return @max(1, (send_mib << 20) / row);
}

// -- owner decode's crossing rows ---------------------------------------------------------------------------------

/// Rows a decode window holds at most (Engine.window).
pub const max_rows = 64;

/// Rows of a window one exchange moves on one rank, in order.
pub const List = struct {
    n: usize = 0,
    rows: [max_rows]u8 = @splat(0),
    pub fn slice(l: *const List) []const u8 {
        return l.rows[0..l.n];
    }
    fn push(l: *List, r: usize) void {
        l.rows[l.n] = @intCast(r);
        l.n += 1;
    }
    pub fn eql(a: List, b: List) bool {
        return std.mem.eql(u8, a.slice(), b.slice());
    }
};

/// Owner decode's two exchanges a DSA layer: the absorbed queries of the rows the peer owns go to it (`q`), and the
/// heads it computed for this rank's queries come back (`out`).
pub const Exchange = enum(u1) { q, out };
pub const Lists = struct {
    send: [2]List = @splat(.{}),
    recv: [2]List = @splat(.{}),
    pub fn of(ls: *const Lists, x: Exchange, sending: bool) *const List {
        return if (sending) &ls.send[@backingInt(x)] else &ls.recv[@backingInt(x)];
    }
};

/// Rank `rank`'s lists of a window whose rows' owners are `orank` (the glue's plane): it sends its queries of the
/// peer's rows and receives the peer's of its own; it sends the peer's heads of its own rows and receives its heads of
/// the peer's. Rank r's send list is rank 1 - r's receive list.
pub fn lists(orank: []const i32, rank: usize) !Lists {
    if (orank.len > max_rows or rank > 1) return error.BadRows;
    var mine: List = .{};
    var theirs: List = .{};
    for (orank, 0..) |o, i| {
        if (o < 0 or o > 1) return error.BadRows;
        if (o == rank) mine.push(i) else theirs.push(i);
    }
    var ls: Lists = .{};
    const q = @backingInt(Exchange.q);
    const o = @backingInt(Exchange.out);
    if (mut(.lists_swapped)) {
        ls.send[q] = mine;
        ls.recv[q] = theirs;
    } else {
        ls.send[q] = theirs;
        ls.recv[q] = mine;
    }
    ls.send[o] = mine;
    ls.recv[o] = theirs;
    return ls;
}

/// A sharded DSA layer's transfers, in the order both ranks issue them: a decode window's queries then outputs, a
/// prompt chunk's history rows.
pub const Step = enum(u8) { q, out, fetch };
pub const Kind = enum(u8) { decode, prompt };
pub fn steps(k: Kind) []const Step {
    return switch (k) {
        .decode => &.{ .q, .out },
        .prompt => &.{.fetch},
    };
}

/// The check of a layer's transfers as they are issued: `begin` at its first, `take` before each, `done` at its end.
/// Any other order is ShardOrder, before a launch, so neither rank waits on a transfer the other never makes. Not
/// called by the device paths: each issues a layer's transfers in one function from replicated inputs
/// (cuda_shard_decode.run, cuda_shard_prompt.attend), whose two-rank models pair every send with its receive.
pub const Cursor = struct {
    s: []const Step = &.{},
    i: usize = 0,
    pub fn begin(c: *Cursor, k: Kind) !void {
        if (!c.done()) return error.ShardOrder;
        c.* = .{ .s = steps(k) };
    }
    pub fn take(c: *Cursor, s: Step) !void {
        if (mut(.cursor_any_order)) {
            c.i += @intFromBool(c.i < c.s.len);
            return;
        }
        if (c.i >= c.s.len or c.s[c.i] != s) return error.ShardOrder;
        c.i += 1;
    }
    pub fn done(c: *const Cursor) bool {
        return c.i == c.s.len;
    }
};

// -- the placement check (rank 1 replays rank 0's calls, and an owner derives from the base) ------------------------

/// Where rank 0 placed a stream's extent, as cuda_lanes' `.place` note carries it to rank 1 with the calls: its base,
/// rows and part, the slot it serves (`owner`), and with TF_GLM_KV_SHARD_CHECK the digest of rank 0's whole pool
/// after it (0: none). Only what the same calls decide alike on both ranks: an extent's id is each pool's own count,
/// which a call one rank made and undid (a lane rank 1 opened before the ranks refused the prefill) leaves apart.
/// Rank 1 placed the same calls itself; a placement of its own anywhere else would be another rank's rows read as
/// this one's, silently, so a mismatch ends the pair (`verify`).
pub const Placement = struct {
    base: u64,
    size: u64,
    part: u32,
    owner: u64,
    digest: u64 = 0,

    pub const bytes = 36;
    /// `owner` of an extent no stream writes in.
    pub const no_owner = std.math.maxInt(u64);

    /// Extent `eid` of `pool` (its digest too when `check`).
    pub fn of(pool: *const Pool, eid: u64, check: bool) !Placement {
        const x = try pool.extent(eid);
        return .{ .base = x.base, .size = x.size, .part = @intCast(pool.partOf(x.base)), .owner = x.owner orelse no_owner, .digest = if (check) pool.digest() else 0 };
    }

    pub fn encode(p: Placement, out: *[bytes]u8) void {
        std.mem.writeInt(u64, out[0..8], p.base, .little);
        std.mem.writeInt(u64, out[8..16], p.size, .little);
        std.mem.writeInt(u32, out[16..20], p.part, .little);
        std.mem.writeInt(u64, out[20..28], p.owner, .little);
        std.mem.writeInt(u64, out[28..36], p.digest, .little);
    }

    pub fn decode(b: []const u8) !Placement {
        if (b.len != bytes) return error.BadPlacement;
        return .{ .base = std.mem.readInt(u64, b[0..8], .little), .size = std.mem.readInt(u64, b[8..16], .little), .part = std.mem.readInt(u32, b[16..20], .little), .owner = std.mem.readInt(u64, b[20..28], .little), .digest = std.mem.readInt(u64, b[28..36], .little) };
    }

    /// Rank 1's side: its own pool holds an extent at the same base, of the same rows, in the same part, for the same
    /// slot, and (a digest sent) its whole layout hashes alike; else KvShardPlacement.
    pub fn verify(p: Placement, pool: *const Pool) !void {
        const x = pool.extentAt(p.base) orelse return error.KvShardPlacement;
        if (x.size != p.size or pool.partOf(x.base) != p.part or (x.owner orelse no_owner) != p.owner) return error.KvShardPlacement;
        if (p.digest != 0) try checkLayout(pool, p.digest);
    }
};

/// Rank 1's side of a layout digest rank 0 sent (with a placement, or every TF_GLM_KV_SHARD_CHECK decode rounds):
/// its own pool's alike, else KvShardPlacement.
pub fn checkLayout(pool: *const Pool, digest: u64) !void {
    if (pool.digest() != digest and !mut(.digest_ignored)) return error.KvShardPlacement;
}

// -- mutants: each rule above turned the wrong way, in test builds only (GLM tools/kv_shard_mutants.sh) -----------------

pub const Mutant = enum {
    none,
    budget_no_trash, // the trash rows' memory not counted
    budget_unaligned, // a pool that is not whole parts of extents
    plan_no_floor, // a tiny part served
    plan_no_shrink, // a part short of the context serves it anyway
    agree_max, // the ranks take the largest pool
    local_global, // an owned row read at its pool row
    glue_trash_pos, // a row it does not own written at its position, not its trash row
    glue_reads_unowned, // a row it does not own reads keys here
    glue_no_trash_check, // a window past the trash rows
    mark_padding, // a -1 token fetches position 0
    renumber_off_by_one, // a history position named one row late
    own_rows_first, // the chunk's own rows named before the fetched ones
    copy_across, // a copy across parts does nothing instead of failing
    carry_rows_minus_1, // a carry sends a row short
    lists_swapped, // the queries' lists turned round
    cursor_any_order, // a transfer out of order taken
    digest_ignored, // a placement check that does not compare the pools' layouts
};
pub var mutant: Mutant = .none;
inline fn mut(m: Mutant) bool {
    return builtin.is_test and mutant == m;
}
