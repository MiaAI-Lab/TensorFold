//! Spill v2's GLM tier on the host, for its tests (cuda_spill_tier_test.zig, and native/cuda.zig's through the prompt
//! cache): an engine whose kept states are real GLM kept-state buffers (cuda_snapshot.sections at TP=2) in host memory,
//! their bytes a function of their tokens, so a pass resumed from a restored state draws what a fresh pass draws
//! exactly when every byte of the state came back (`first`); and a pair of ranks in one thread, each with its lane
//! backend, its own store under TF_SPILL_TEST_DIR and the tier, rank 1 following rank 0's notes and asks. Sharded
//! (`Pair.initWith(.., .{ .sharded = true })`, TF_GLM_KV_SHARD): each rank's layout owner-sharded and its own, its
//! sequences extents of a two-part pool (cuda_pool) placed as the engine places them, a kept state holding its latent
//! rows on its home part's owner only, a resume placed in its home part, moved there, carried, or from 0
//! (cuda_shard_keep).
//! Test code only: glm_cuda exports it in test builds alone (cuda.zig's `spill_tier_rig`).
const std = @import("std");
const core = @import("core");
const lanes = @import("lanes");
const lane_backend = @import("cuda_lanes.zig");
const st = @import("cuda_spill_state.zig");
const snapshot = @import("cuda_snapshot.zig");
const state = @import("cuda_state.zig");
const tier = @import("cuda_spill_tier.zig");
const wire = @import("cuda_spill_wire.zig");
const shard_keep = @import("cuda_shard_keep.zig");
const Pool = @import("cuda_pool.zig").Pool;

const spill = core.spill;
const format = spill.format;
const G = state.Geometry;
const testing = std.testing;

extern "c" fn memcmp(a: [*]const u8, b: [*]const u8, n: usize) c_int;

/// An engine whose sequence is its tokens and whose kept state is a GLM kept-state buffer filled from them (`fill`).
/// Its device is host memory: copies are synchronous, an event stays pending `delay` polls after its record. With a
/// `pool` (TF_GLM_KV_SHARD's two parts) a sequence holds an extent and a kept state its home (cuda_engine's rules).
pub const HostEngine = struct {
    pub const snap_holds_rows = true;
    pub const Seq = struct { tokens: std.ArrayList(u32) = .empty, eid: ?u64 = null };
    pub const Buf = struct { ptr: u64, bytes: []u8 };
    pub const Snap = struct { id: u64 = 0, pos: usize = 0, mtp_len: usize = 0, buf: ?Buf = null, sec: snapshot.Sections = undefined, refs: std.atomic.Value(u32) = .init(1), home: ?u8 = null };
    pub const PrefillOptions = struct { keep_at: ?usize = null, resume_from: ?*const Snap = null, live_from: ?usize = null, mask: []const u32 = &.{}, media: ?*const lanes.Media = null };
    pub const Prefilled = struct { first: u32, kept: ?*Snap = null };
    pub const Window = struct { seq: *Seq, tokens: []const u32 };
    pub const DraftReq = struct { seq: *Seq };

    gpa: std.mem.Allocator,
    layout: st.Layout,
    bound: ?*Seq = null,
    next_id: u64 = 0,
    /// kept-state buffers alive
    snaps: usize = 0,
    /// passes that resumed a kept state, and those whose state was a fresh pass's byte for byte
    resumed: usize = 0,
    exact: usize = 0,
    /// each event's polls left before it completes (0: complete), and what a record leaves
    events: []u8 = &.{},
    delay: u8 = 0,
    waits: usize = 0,
    /// tierDest refuses (a rank at its keep floor)
    refuse_dest: bool = false,
    scratch: []u8 = &.{},
    /// TF_GLM_KV_SHARD's parts (null: off), TF_GLM_KV_SHARD_CARRY, and the resumes placeResume carried or sent to 0
    pool: ?Pool = null,
    carry: bool = true,
    carries: usize = 0,
    refused_small: usize = 0,

    pub fn init(gpa: std.mem.Allocator, layout: st.Layout) HostEngine {
        return .{ .gpa = gpa, .layout = layout };
    }

    pub fn deinit(e: *HostEngine) void {
        e.gpa.free(e.events);
        e.gpa.free(e.scratch);
        if (e.pool) |*p| p.deinit();
    }

    pub fn first(tokens: []const u32) u32 {
        return @truncate(std.hash.Wyhash.hash(3, std.mem.sliceAsBytes(tokens)));
    }

    fn mix(h: u64, t: u32) u64 {
        var x = (h ^ t) *% 0x9e3779b97f4a7c15;
        x ^= x >> 29;
        return x *% 0xbf58476d1ce4e5b9;
    }

    fn row(dst: []u8, v: u64) void {
        @memset(dst, @truncate(v >> 11));
        std.mem.writeInt(u64, dst[0..8], v, .little);
        std.mem.writeInt(u64, dst[dst.len - 8 ..][0..8], ~v, .little);
    }

    /// The state ranges a kept state holds (every layer's latent rows when it holds them, its complete pools, each
    /// section): `f` on each.
    fn ranges(l: *const st.Layout, sec: snapshot.Sections, ctx: anytype, comptime f: fn (@TypeOf(ctx), usize, usize) void) void {
        for (0..G.dsa_layers) |i| {
            if (sec.rows) f(ctx, sec.latent + i * sec.latentBytes(), sec.at * G.latent_row);
            f(ctx, sec.pooled + i * sec.pooledBytes(), sec.at / 4 * G.pool_row);
        }
        const at = [_]usize{ sec.rings, sec.rec, sec.conv, sec.draft };
        for (l.specs(), l.order[0..l.n]) |s, o| f(ctx, at[@backingInt(o)], s.len);
    }

    /// The state a fresh pass of `tokens` keeps at their end: each latent row from the tokens through it (when the
    /// buffer holds them), each complete pool from its four, the sections from all of them; pooled row at/4 and the
    /// padding stale bytes (0xee).
    pub fn fill(l: *const st.Layout, out: []u8, sec: snapshot.Sections, tokens: []const u32) void {
        @memset(out[0..sec.total], 0xee);
        var h: u64 = 0x51ed;
        for (tokens[0..sec.at], 0..) |t, r| {
            h = mix(h, t);
            for (0..G.dsa_layers) |i| {
                if (sec.rows) row(out[sec.latent + i * sec.latentBytes() + r * G.latent_row ..][0..G.latent_row], h +% i);
                if (r % 4 == 3) row(out[sec.pooled + i * sec.pooledBytes() + r / 4 * G.pool_row ..][0..G.pool_row], h ^ (@as(u64, i) << 40));
            }
        }
        const at = [_]usize{ sec.rings, sec.rec, sec.conv, sec.draft };
        for (l.specs(), l.order[0..l.n]) |s, o| row(out[at[@backingInt(o)]..][0..s.len], h ^ @backingInt(o));
    }

    /// Whether `a` and `b` hold the same state (the ranges a kept state holds; pooled row at/4 and padding aside).
    pub fn same(l: *const st.Layout, sec: snapshot.Sections, a: []const u8, b: []const u8) bool {
        const C = struct {
            a: []const u8,
            b: []const u8,
            ok: bool = true,
            fn f(c: *@This(), off: usize, len: usize) void {
                if (len > 0 and memcmp(c.a[off..].ptr, c.b[off..].ptr, len) != 0) c.ok = false;
            }
        };
        var c: C = .{ .a = a, .b = b };
        ranges(l, sec, &c, C.f);
        return c.ok;
    }

    pub fn bytesOf(snap: *const Snap) []u8 {
        return snap.buf.?.bytes;
    }

    /// A kept-state buffer of `at` tokens kept in part `home` (null: unsharded): its latent rows only where this rank
    /// owns that part (cuda_engine takeSnap / tierDest).
    fn newSnap(e: *HostEngine, at: usize, home: ?u8) !*Snap {
        const sec = e.layout.sectionsOf(at, try e.layout.owns(home));
        const bytes = try e.gpa.alloc(u8, sec.total);
        errdefer e.gpa.free(bytes);
        const s = try e.gpa.create(Snap);
        e.next_id += 1;
        s.* = .{ .id = e.next_id, .pos = at, .buf = .{ .ptr = @intFromPtr(bytes.ptr), .bytes = bytes }, .sec = sec, .home = home };
        e.snaps += 1;
        return s;
    }

    /// The part sequence `s`'s extent lies in (null: unsharded).
    pub fn partOf(e: *HostEngine, s: *const Seq) ?u8 {
        const p = if (e.pool) |*q| q else return null;
        return @intCast(p.partOf((p.extent(s.eid.?) catch unreachable).base));
    }

    /// Whether `snap` holds the state a fresh pass of `tokens` keeps.
    pub fn exactFor(e: *HostEngine, snap: *const Snap, tokens: []const u32) bool {
        if (e.scratch.len < snap.sec.total) {
            e.gpa.free(e.scratch);
            e.scratch = e.gpa.alloc(u8, snap.sec.total) catch return false;
        }
        fill(&e.layout, e.scratch, snap.sec, tokens);
        return same(&e.layout, snap.sec, e.scratch, bytesOf(snap));
    }

    pub fn seqBytes(_: *const HostEngine, rows: usize) usize {
        return rows * 10;
    }
    pub fn snapBytes(e: *const HostEngine, at: usize) usize {
        return e.layout.sections(at).total;
    }
    pub fn newSeq(e: *HostEngine, rows: usize) !*Seq {
        const s = try e.gpa.create(Seq);
        errdefer e.gpa.destroy(s);
        s.* = .{};
        if (e.pool) |*p| {
            s.eid = (try p.alloc(rows)) orelse return error.NoPoolExtent;
            (try p.get(s.eid.?)).owner = s.eid.?; // (a live stream: the placement evens the parts' streams out)
        }
        return s;
    }
    pub fn freeSeq(e: *HostEngine, s: *Seq) void {
        if (e.pool) |*p| p.remove(s.eid.?) catch unreachable;
        s.tokens.deinit(e.gpa);
        e.gpa.destroy(s);
    }
    /// cuda_engine placeResume: the fresh extent stays in the state's home part, moves there, keeps its part for a
    /// carry, or the prompt runs from 0.
    pub fn placeResume(e: *HostEngine, s: *Seq, snap: *const Snap) !bool {
        const p = if (e.pool) |*q| q else return true;
        switch (try shard_keep.resumeIn(p, s.eid.?, snap.home orelse return error.KvShardNoHome, snap.pos, e.carry)) {
            .here => return true,
            .move => |base| {
                _ = try p.move(s.eid.?, base, null, true);
                return true;
            },
            .carry => {
                e.carries += 1;
                return true;
            },
            .recompute => |why| {
                e.refused_small += @intFromBool(why == .small);
                return false;
            },
        }
    }
    pub fn bind(e: *HostEngine, s: *Seq) void {
        e.bound = s;
    }
    pub fn copyPrefix(_: *HostEngine, _: *Seq, _: *Seq, _: usize, _: usize) !void {
        return error.NotUsed;
    }
    pub fn commitWindow(_: *HostEngine, _: usize, _: usize) !void {
        return error.NotUsed;
    }
    pub fn current(_: *const HostEngine) void {}

    /// A resumed pass draws what a fresh one draws only from a state byte-identical to a fresh pass's.
    pub fn prefillWith(e: *HostEngine, prompt: []const u32, _: ?lanes.Sampling, o: PrefillOptions) !Prefilled {
        const q = e.bound.?;
        var ok = true;
        if (o.live_from) |p| {
            if (o.resume_from != null or q.tokens.items.len != p) return error.ResumeMustExtend;
        } else if (o.resume_from) |snap| {
            if (snap.pos >= prompt.len) return error.ResumeOtherPrompt;
            e.resumed += 1;
            ok = e.exactFor(snap, prompt[0..snap.pos]);
            if (ok) e.exact += 1;
            q.tokens.clearRetainingCapacity();
            try q.tokens.appendSlice(e.gpa, prompt[0..snap.pos]);
        } else q.tokens.clearRetainingCapacity();
        try q.tokens.appendSlice(e.gpa, prompt[q.tokens.items.len..]);
        var kept: ?*Snap = null;
        if (o.keep_at) |k| {
            kept = try e.newSnap(k, e.partOf(q));
            fill(&e.layout, bytesOf(kept.?), kept.?.sec, prompt[0..k]);
        }
        return .{ .first = if (ok) first(q.tokens.items) else first(q.tokens.items) ^ 1, .kept = kept };
    }

    pub fn freeSnapshot(e: *HostEngine, snap: *Snap) void {
        if (snap.refs.fetchSub(1, .acq_rel) != 1) return;
        e.gpa.free(snap.buf.?.bytes);
        e.gpa.destroy(snap);
        e.snaps -= 1;
    }

    // -- the tier's device (cuda_spill_tier) --------------------------------------------------------------------

    pub fn tierKept(_: *const HostEngine, snap: *const Snap) st.Kept {
        return .{ .buf = snap.buf.?.ptr, .sec = snap.sec, .home = snap.home };
    }
    fn d2h(_: *anyopaque, dst: []u8, src: u64) anyerror!void {
        const s: [*]const u8 = @ptrFromInt(src);
        @memcpy(dst, s[0..dst.len]);
    }
    fn h2d(_: *anyopaque, dst: u64, src: []const u8) anyerror!void {
        const d: [*]u8 = @ptrFromInt(dst);
        @memcpy(d[0..src.len], src);
    }
    fn fillBytes(_: *anyopaque, dst: u64, byte: u8, n: usize) anyerror!void {
        const d: [*]u8 = @ptrFromInt(dst);
        @memset(d[0..n], byte);
    }
    pub fn tierCopier(e: *HostEngine) st.Copier {
        return .{ .ptr = e, .d2h = d2h, .h2d = h2d, .fill = fillBytes };
    }
    pub fn tierEvents(e: *HostEngine, n: usize) !void {
        if (e.events.len >= n) return;
        const ev = try e.gpa.alloc(u8, n);
        @memset(ev, 0);
        e.gpa.free(e.events);
        e.events = ev;
    }
    pub fn tierRecord(e: *HostEngine, ev: usize) void {
        e.events[ev] = e.delay;
    }
    pub fn tierDone(e: *HostEngine, ev: usize) bool {
        if (e.events[ev] == 0) return true;
        e.events[ev] -= 1;
        return false;
    }
    pub fn tierWait(e: *HostEngine, ev: usize) void {
        if (e.events[ev] > 0) e.waits += 1;
        e.events[ev] = 0;
    }
    pub fn tierDest(e: *HostEngine, at: usize, home: ?u8) !*Snap {
        if (e.refuse_dest) return error.BelowMemoryReserve;
        const s = try e.newSnap(at, home);
        @memset(bytesOf(s), 0x33);
        return s;
    }
    /// cuda_engine tierResumable: a restore only where its resume could be placed or carried.
    pub fn tierResumable(e: *const HostEngine, at: usize, home: u8) !bool {
        const p = if (e.pool) |*q| q else return true;
        return shard_keep.restorable(p, home, at, e.carry);
    }
};

pub const Lanes = lane_backend.Lanes(HostEngine);
pub const Leader = tier.Leader(HostEngine);
pub const Follower = tier.Follower(HostEngine);

/// Rank 0's notes and asks in order (an ask behind `ask_mark`), for rank 1 to apply.
pub const Sink = struct {
    gpa: std.mem.Allocator,
    notes: std.ArrayList([]u8) = .empty,
    flushes: usize = 0,
    const ask_mark = 0xff;

    fn send(ptr: *anyopaque, bytes: []const u8) void {
        const s: *Sink = @ptrCast(@alignCast(ptr));
        s.notes.append(s.gpa, s.gpa.dupe(u8, bytes) catch return) catch {};
    }
    fn ask(ptr: *anyopaque, bytes: []const u8) anyerror!void {
        const s: *Sink = @ptrCast(@alignCast(ptr));
        const b = try s.gpa.alloc(u8, 1 + bytes.len);
        errdefer s.gpa.free(b);
        b[0] = ask_mark;
        @memcpy(b[1..], bytes);
        try s.notes.append(s.gpa, b);
    }
    fn flush(ptr: *anyopaque) void {
        const s: *Sink = @ptrCast(@alignCast(ptr));
        s.flushes += 1;
    }
    pub fn notesOf(s: *Sink) lane_backend.Notes {
        return .{ .ptr = s, .send = send, .ask = ask, .flush = flush };
    }
    /// Every note and ask so far applied on rank 1, in order (an ask's gather inside it).
    pub fn take(s: *Sink, r1: *Lanes) !void {
        while (s.notes.items.len > 0) {
            const n = s.notes.orderedRemove(0);
            defer s.gpa.free(n);
            if (n.len > 0 and n[0] == ask_mark) try r1.applyAsk(n[1..]) else try r1.applyNote(n);
        }
    }
    pub fn deinit(s: *Sink) void {
        for (s.notes.items) |n| s.gpa.free(n);
        s.notes.deinit(s.gpa);
    }
};

/// Two ranks' gather in one thread: rank 0's hands what waits in the sink to rank 1 first, whose gather (inside its
/// ask's handler) finds rank 0's words there.
pub const Gather = struct {
    sink: *Sink,
    r1: *Lanes,
    mine0: [wire.stats_words]u64 = undefined,
    mine1: [wire.stats_words]u64 = undefined,
    n: usize = 0,
    fatals: usize = 0,
    asks: usize = 0, // rank 0's gathers (its asks)

    pub fn gate(g: *Gather, rank: u32) lane_backend.Gate {
        return .{ .ptr = g, .agree = agree, .fatal = fatal, .gather = if (rank == 0) gather0 else gather1 };
    }
    fn of(p: *anyopaque) *Gather {
        return @ptrCast(@alignCast(p));
    }
    fn agree(_: *anyopaque, ok: bool) anyerror!bool {
        return ok;
    }
    fn fatal(p: *anyopaque, _: []const u8, _: anyerror) void {
        of(p).fatals += 1;
    }
    fn gather0(p: *anyopaque, mine: []const u64, out: []u64) anyerror!usize {
        const g = of(p);
        g.asks += 1;
        g.n = mine.len;
        @memcpy(g.mine0[0..mine.len], mine);
        try g.sink.take(g.r1);
        return g.both(out);
    }
    fn gather1(p: *anyopaque, mine: []const u64, out: []u64) anyerror!usize {
        const g = of(p);
        if (mine.len != g.n) return error.TestWidthsDiffer;
        @memcpy(g.mine1[0..mine.len], mine);
        return g.both(out);
    }
    fn both(g: *Gather, out: []u64) usize {
        @memcpy(out[0..g.n], g.mine0[0..g.n]);
        @memcpy(out[g.n..][0..g.n], g.mine1[0..g.n]);
        return 2;
    }
};

/// The layout the tests use: TP=2's, a 64-row index ring, no drafter (the sections still ~74 MiB: the recurrences).
pub fn testLayout() !st.Layout {
    return st.Layout.init(try G.init(2, 0), 64, null);
}

/// Rank `rank`'s owner-sharded layout of the same shape (TF_GLM_KV_SHARD).
pub fn shardedLayout(rank: usize) !st.Layout {
    return (try st.Layout.init(try G.init(2, rank), 64, null)).sharded();
}

/// Rows of the sharded rig's pool: two parts of 8,192 (two 2,200-token streams' extents each).
pub const pool_rows = 8 * 2048;

/// Two ranks with the tier on each (`lead`, `fol`), their stores in fresh folders.
pub const Pair = struct {
    gpa: std.mem.Allocator,
    /// rank 0's layout (the replicated tests' on both ranks); `layouts` each rank's own
    layout: st.Layout,
    layouts: [2]st.Layout,
    e: [2]HostEngine,
    r: [2]Lanes,
    sink: Sink,
    pg: Gather,
    settings: tier.Settings,
    paths: [2][256]u8 = undefined,
    roots: [2][:0]const u8 = undefined,
    staging: [2][]align(spill.slots.alignment) u8 = undefined,
    compat: format.Digest,
    lead: ?*Leader = null,
    fol: ?*Follower = null,
    hooks: lane_backend.CacheHooks = undefined,

    /// Settings for the tests: a small store over the scratch folders, a staging of `write` block slots, one read-ahead
    /// slot, one small state and one restore (and `restores` of each when asked).
    pub fn settingsFor(restores: u32) tier.Settings {
        return .{ .dir = "", .gib = 4, .min_free_gib = 0, .min_tokens = 64, .stage_mib = 26, .read_blocks = 2, .smalls = 1, .restores = restores, .readers = 2, .direct = false, .hold_bytes = 1 << 30 };
    }

    pub const Opts = struct {
        /// TF_GLM_KV_SHARD: owner-sharded layouts (each rank's own), a two-part pool on each engine
        sharded: bool = false,
    };

    pub fn init(p: *Pair, gpa: std.mem.Allocator, tag: []const u8, s: tier.Settings) !void {
        return p.initWith(gpa, tag, s, .{});
    }

    pub fn initWith(p: *Pair, gpa: std.mem.Allocator, tag: []const u8, s: tier.Settings, o: Opts) !void {
        const layout = try testLayout();
        const ls: [2]st.Layout = if (o.sharded) .{ try shardedLayout(0), try shardedLayout(1) } else .{ layout, layout };
        p.* = .{ .gpa = gpa, .layout = ls[0], .layouts = ls, .e = .{ HostEngine.init(gpa, ls[0]), HostEngine.init(gpa, ls[1]) }, .r = undefined, .sink = .{ .gpa = gpa }, .pg = undefined, .settings = s, .compat = ls[0].compat(ls[0].digest()) };
        if (o.sharded) for (&p.e) |*e| {
            e.pool = try Pool.initParts(gpa, pool_rows, 2);
        };
        p.r = .{ Lanes.init(gpa, &p.e[0], true, 1 << 20, 8), Lanes.init(gpa, &p.e[1], true, 1 << 20, 8) };
        p.r[1].follower = true;
        p.r[0].notes = p.sink.notesOf();
        p.pg = .{ .sink = &p.sink, .r1 = &p.r[1] };
        p.r[0].gate = p.pg.gate(0);
        p.r[1].gate = p.pg.gate(1);
        p.hooks = p.r[0].cacheHooks(1 << 40, .{}, 0, 5);
        const plan = try tier.plan(s, &layout);
        for (0..2) |i| {
            var tb: [64]u8 = undefined;
            p.roots[i] = try spill.fsio.testRoot(&p.paths[i], try std.fmt.bufPrint(&tb, "glm-tier-{s}-r{d}", .{ tag, i }));
            _ = std.c.close(try spill.fsio.freshDir(gpa, p.roots[i]));
            p.staging[i] = try gpa.alignedAlloc(u8, .fromByteUnits(spill.slots.alignment), plan.bytes());
        }
        try p.open();
    }

    /// Each rank's store opened (recovering what its folder holds), the tier on both.
    pub fn open(p: *Pair) !void {
        var sides: [2]*Leader.S = undefined;
        for (0..2) |i| {
            var s = p.settings;
            s.dir = p.roots[i];
            sides[i] = try Leader.S.open(p.gpa, testing.io, &p.e[i], p.layouts[i], s, @intCast(i), p.compat, 1 << 20, p.staging[i], .{});
        }
        p.lead = try Leader.create(p.gpa, sides[0], &p.r[0], p.settings, @splat(7));
        p.fol = try Follower.create(p.gpa, sides[1], &p.r[1]);
        p.r[1].spill_side = p.fol.?.spillSide();
    }

    /// The tiers closed (a process's end: nothing flushed).
    pub fn close(p: *Pair) void {
        p.r[1].spill_side = null;
        if (p.lead) |l| l.destroy();
        if (p.fol) |f| f.destroy();
        p.lead = null;
        p.fol = null;
        for (p.sink.notes.items) |n| p.gpa.free(n);
        p.sink.notes.clearRetainingCapacity();
    }

    /// A restart: every kept state gone from memory on both ranks (each released to its tier first, as the store
    /// does), the tiers closed and opened again over the same folders.
    pub fn restart(p: *Pair) !void {
        try p.dropAll();
        p.close();
        try p.open();
    }

    pub fn deinit(p: *Pair) void {
        p.close();
        p.r[0].deinit();
        p.r[1].deinit();
        p.sink.deinit();
        for (0..2) |i| {
            spill.fsio.dropDir(p.gpa, p.roots[i]);
            p.gpa.free(p.staging[i]);
            p.e[i].deinit();
        }
    }

    pub fn lead0(p: *Pair) *Leader {
        return p.lead.?;
    }

    /// A pass of `prompt` on both ranks keeping its state at `mark` (the store's mark hook on rank 0 adopts it and
    /// writes it at keep: Tier.keep); rank 1 hears the pass's notes. Rank 0's kept state and its point's id.
    pub fn keepPass(p: *Pair, prompt: []const u32, mark: u32, group: u64) !struct { saved: *anyopaque, point: ?u64 } {
        var w: Writer = .{ .p = p, .group = group };
        _ = try p.passBoth(prompt, &.{mark}, .{ .ptr = &w, .at = Writer.at }, null);
        return .{ .saved = w.saved orelse return error.TestNothingKept, .point = w.point };
    }

    /// A pass of `prompt` on both ranks cut at `marks`, rank 0's mark hook `hook` (the prompt cache's), resumed from
    /// rank 0's `from` (and rank 1's copy under its key) when given; rank 1 hears the pass's notes after. Rank 0's first
    /// token.
    pub fn passBoth(p: *Pair, prompt: []const u32, marks: []const u32, hook: ?lanes.stream.MarkHook, from: ?*anyopaque) !u32 {
        const at: u32 = if (from) |f| @as(*Lanes.Kept, @ptrCast(@alignCast(f))).at else 0;
        const from1: ?*anyopaque = if (from) |f| p.r[1].savedFor(@as(*Lanes.Kept, @ptrCast(@alignCast(f))).key) else null;
        var s0 = try lanes.Stream.init(p.gpa, .{ .id = "k", .prompt = prompt, .max_new = 4, .reuse = .{ .saved = from, .at = at, .marks = marks, .hook = hook } });
        defer s0.deinit(p.gpa);
        var s1 = try lanes.Stream.init(p.gpa, .{ .id = "k", .prompt = prompt, .max_new = 4, .reuse = .{ .saved = from1, .at = at, .marks = marks } });
        defer s1.deinit(p.gpa);
        try Lanes.prefillFn(&p.r[0], &s0);
        try Lanes.prefillFn(&p.r[1], &s1);
        const first = try firstOf(&p.r[0], &s0);
        if (try firstOf(&p.r[1], &s1) != first) return error.TestRanksDrewApart;
        Lanes.releaseFn(&p.r[0], &s0);
        Lanes.releaseFn(&p.r[1], &s1);
        try p.sink.take(&p.r[1]);
        return first;
    }

    const Writer = struct {
        p: *Pair,
        group: u64,
        saved: ?*anyopaque = null,
        point: ?u64 = null,
        fn at(ptr: *anyopaque, s: *lanes.Stream, mark: u32) void {
            const w: *Writer = @ptrCast(@alignCast(ptr));
            const saved = w.p.hooks.save(w.p.hooks.ptr, null, mark) catch return;
            w.saved = saved;
            w.point = w.p.lead0().keep(.{ .saved = saved, .key = s.prompt()[0..mark], .at = mark, .group = w.group });
        }
    };

    /// Sharded: a live extent of `rows` in part `part` of both ranks' pools (another stream's), so placements go
    /// elsewhere while it stays. Its id on each rank.
    pub fn occupy(p: *Pair, part: usize, rows: usize) ![2]u64 {
        var ids: [2]u64 = undefined;
        for (&p.e, &ids) |*e, *id| {
            const pool = &e.pool.?;
            id.* = try pool.add((try pool.placeIn(rows, null, part)) orelse return error.TestNoRoom, rows);
            (try pool.get(id.*)).owner = id.*;
        }
        return ids;
    }

    pub fn vacate(p: *Pair, ids: [2]u64) !void {
        for (&p.e, ids) |*e, id| try e.pool.?.remove(id);
    }

    /// Rank `rank`'s kept state of rank 0's `saved` (rank 0's own, or rank 1's copy under its key).
    pub fn snapOn(p: *Pair, rank: usize, saved: *anyopaque) !*HostEngine.Snap {
        const k0: *Lanes.Kept = @ptrCast(@alignCast(saved));
        if (rank == 0) return k0.snap;
        const k1: *Lanes.Kept = @ptrCast(@alignCast(p.r[1].savedFor(k0.key) orelse return error.TestNotOnRank1));
        return k1.snap;
    }

    /// Pumps both ranks (rank 1 by rank 0's pump notes) until neither has a write to stage or publish.
    pub fn settle(p: *Pair) !void {
        var i: usize = 0;
        while (i < 10_000) : (i += 1) {
            p.lead0().pump(5 * std.time.ns_per_ms);
            try p.sink.take(&p.r[1]);
            const a = p.lead0().side.store.counts();
            const b = p.fol.?.side.store.counts();
            if (a.pending_jobs == 0 and b.pending_jobs == 0 and !p.lead0().side.busy() and !p.fol.?.side.busy()) return;
            std.Io.sleep(testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.TestNotSettled;
    }

    /// Rank 0's kept state `saved` leaves memory as the store frees it: the tier's release, then the family's drop
    /// (a `.drop` note, rank 1's release and free).
    pub fn evict(p: *Pair, saved: *anyopaque) !void {
        p.lead0().release(saved);
        p.hooks.drop(p.hooks.ptr, saved);
        try p.sink.take(&p.r[1]);
    }

    pub fn dropAll(p: *Pair) !void {
        while (p.r[0].kept.items.len > 0) try p.evict(p.r[0].kept.items[0]);
    }

    /// A restore of the longest point `prompt` extends, polled to its end; rank 1 hears every note on the way.
    pub fn restore(p: *Pair, prompt: []const u32) !tier.Poll {
        const l = p.lead0();
        const pt = l.lookup(prompt, 0, @intCast(prompt.len - 1), &.{}, 0) orelse return error.TestNoPoint;
        const job = try l.begin(pt.id, pt.at, pt.shared);
        try p.sink.take(&p.r[1]);
        var i: usize = 0;
        while (i < 10_000) : (i += 1) {
            const r = l.poll(job, 5 * std.time.ns_per_ms);
            try p.sink.take(&p.r[1]);
            if (r != .pending) return r;
            std.Io.sleep(testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.TestRestoreHung;
    }

    /// A pass of `prompt` resumed from rank 0's `saved` on rank 0 and from rank 1's own copy under the same key: both
    /// first tokens.
    pub fn resumeBoth(p: *Pair, prompt: []const u32, saved: *anyopaque, at: u32) ![2]u32 {
        const k0: *Lanes.Kept = @ptrCast(@alignCast(saved));
        var s0 = try lanes.Stream.init(p.gpa, .{ .id = "r", .prompt = prompt, .max_new = 4, .reuse = .{ .saved = saved, .at = at } });
        defer s0.deinit(p.gpa);
        var s1 = try lanes.Stream.init(p.gpa, .{ .id = "r", .prompt = prompt, .max_new = 4, .reuse = .{ .saved = p.r[1].savedFor(k0.key), .at = at } });
        defer s1.deinit(p.gpa);
        try Lanes.prefillFn(&p.r[0], &s0);
        try Lanes.prefillFn(&p.r[1], &s1);
        defer Lanes.releaseFn(&p.r[0], &s0);
        defer Lanes.releaseFn(&p.r[1], &s1);
        return .{ try firstOf(&p.r[0], &s0), try firstOf(&p.r[1], &s1) };
    }

    fn firstOf(r: *Lanes, s: *lanes.Stream) !u32 {
        return Lanes.readFn(r, try Lanes.firstFn(r, s, s.prompt_len));
    }

    /// Rank `rank`'s store folder: `<root>/glm-spill-<16 hex>/rank<N>/<sub>` (gpa-owned).
    pub fn storeDir(p: *Pair, rank: usize, sub: []const u8) ![]u8 {
        var root = try std.Io.Dir.cwd().openDir(testing.io, p.roots[rank], .{ .iterate = true });
        defer root.close(testing.io);
        var it = root.iterate();
        while (try it.next(testing.io)) |ns| {
            if (ns.kind != .directory or !std.mem.startsWith(u8, ns.name, tier.prefix)) continue;
            return std.fmt.allocPrint(p.gpa, "{s}/{s}/rank{d}/{s}", .{ p.roots[rank], ns.name, rank, sub });
        }
        return error.TestNoStore;
    }

    pub const Damage = enum { flip, truncate, remove };

    /// Damage to every block file of rank `rank`'s store.
    pub fn damage(p: *Pair, rank: usize, how: Damage) !void {
        const path = try p.storeDir(rank, "blocks");
        defer p.gpa.free(path);
        var d = try std.Io.Dir.cwd().openDir(testing.io, path, .{ .iterate = true });
        defer d.close(testing.io);
        var it = d.iterate();
        var names: std.ArrayList([]u8) = .empty;
        defer {
            for (names.items) |n| p.gpa.free(n);
            names.deinit(p.gpa);
        }
        while (try it.next(testing.io)) |b| if (b.kind == .file) try names.append(p.gpa, try p.gpa.dupe(u8, b.name));
        if (names.items.len == 0) return error.TestNoBlocks;
        for (names.items) |n| switch (how) {
            .remove => try d.deleteFile(testing.io, n),
            .flip, .truncate => {
                var f = try d.openFile(testing.io, n, .{ .mode = .read_write });
                defer f.close(testing.io);
                if (how == .truncate) {
                    try f.setLength(testing.io, (try f.length(testing.io)) / 2);
                } else {
                    var one: [1]u8 = undefined;
                    _ = try f.readPositionalAll(testing.io, &one, 4099);
                    one[0] ^= 0xff;
                    try f.writePositionalAll(testing.io, &one, 4099);
                }
            },
        };
    }

    /// Every point manifest of rank `rank`'s store removed (a store that lost them: recovery drops the rest).
    pub fn dropManifests(p: *Pair, rank: usize) !void {
        const path = try p.storeDir(rank, "points");
        defer p.gpa.free(path);
        var d = try std.Io.Dir.cwd().openDir(testing.io, path, .{ .iterate = true });
        defer d.close(testing.io);
        var it = d.iterate();
        var names: std.ArrayList([]u8) = .empty;
        defer {
            for (names.items) |n| p.gpa.free(n);
            names.deinit(p.gpa);
        }
        while (try it.next(testing.io)) |b| if (std.mem.endsWith(u8, b.name, ".man")) try names.append(p.gpa, try p.gpa.dupe(u8, b.name));
        if (names.items.len == 0) return error.TestNoPoints;
        for (names.items) |n| try d.deleteFile(testing.io, n);
    }
};

/// `buf` filled with ids from `seed`.
pub fn idsOf(buf: []u32, seed: u32) []u32 {
    var prng = std.Random.DefaultPrng.init(seed);
    for (buf) |*t| t.* = prng.random().intRangeLessThan(u32, 0, 154880);
    return buf;
}
