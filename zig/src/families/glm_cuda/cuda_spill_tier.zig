//! Spill v2's GLM tier: core/prompt_cache's `Tier` over every rank's core/spill block store, with step A's packing
//! (cuda_spill_state) and step B's rank protocol (cuda_spill_wire over Flash Next's mirror). Port of wojo's GLM 0110
//! (PR #124; BadAd84's block tier: write at keep through a pinned staging ring pumped 4 blocks between rounds, 16
//! read-ahead blocks, HAS / LOADPART / LOADED, room by conversation) in the shape of the DeepSeek port's
//! dsv41/spill_tier.zig (Side, Leader, Follower).
//!
//! Every rank keeps its own store (its own disk, its own copy of each kept state). Rank 0 runs the `Leader`: the prompt
//! cache's tier and every decision (what is written, what is restored, whether a restore commits), each one a mirror
//! note or ask in its call order. Rank 1 runs a `Follower` (cuda_lanes.SpillSide) from the same ordered stream:
//!   - write at keep: rank 0's store keeps the state inside its pass (Tier.keep): its sections and tail are copied off
//!     the kept buffer at once, its whole blocks staged between rounds (`pump`, 4 blocks at most); a `keep` note makes
//!     rank 1 write its own copy once its pass adopted the same state, and `pump` notes stage rank 1's blocks as long;
//!   - restore: `has` (every rank holds the point, or no rank restores it), a `begin` note (each rank reserves its
//!     destination under rank 0's key and starts its reads), `progress` notes as rank 0's copies go (rank 1 copies
//!     as far), then once rank 0's own copies are installed `validate` (every rank's copies passed): commit on every
//!     rank, or abort, and every rank forgets the point when any rank's copy was damaged;
//!   - a kept state about to be freed while its write still reads it is held (cuda_spill_state.Guard), or its rest is
//!     staged at once past the guard's limit.
//! Nothing here waits on a disk while the engine runs: the stores' threads do the file I/O, the engine thread only
//! enqueues copies on its stream and polls their events. The engine `E` (cuda_engine.Engine; a host-memory one in the
//! tests) gives: `tierKept(snap)` (the buffer and its sections), `tierCopier()`, `tierEvents(n)`, `tierRecord(ev)`,
//! `tierDone(ev)`, `tierWait(ev)`, `tierDest(at)` (a new kept-state buffer, under the keep floor), `freeSnapshot`,
//! `current`.
const std = @import("std");
const builtin = @import("builtin");
const core = @import("core");
const lane_backend = @import("cuda_lanes.zig");
const st = @import("cuda_spill_state.zig");
const wire = @import("cuda_spill_wire.zig");

const spill = core.spill;
const format = spill.format;
const Allocator = std.mem.Allocator;
const Digest = format.Digest;

/// The store's namespace prefix for GLM (GLM v1, DeepSeek's store and the Python forks never prune it).
pub const prefix = "glm-spill-";
/// Blocks one pump stages at most (GLM 0110's PUMP_BLOCKS).
pub const pump_blocks = 4;
/// A following rank's copies for one progress note (its reads already in slots: no disk wait).
const progress_ns: u64 = 2 * std.time.ns_per_ms;
/// How long a following rank's validate may take to finish its reads and copies: then an answer (`timeout`), never
/// silence, since rank 0 waits in the gather.
const follower_copy_ns: u64 = 30 * std.time.ns_per_s;
/// Rank 0 has the other ranks stage their own writes at most this often (a link frame each while idle).
const pump_every_ns: u64 = 10 * std.time.ns_per_ms;
/// Every rank's counters at most this often (a gather each), and only after rank 0 sent the others something.
const stats_every_ns: u64 = std.time.ns_per_s;

/// The tier's settings on a rank (the server's spill flags and TF_GLM_SPILL_*, cli.spill). Off unless `dir` and
/// `gib` > 0.
pub const Settings = struct {
    dir: []const u8,
    gib: f64,
    min_free_gib: f64 = 50,
    min_tokens: u32 = 8192,
    /// TF_GLM_SPILL_QUOTA: one conversation's share of the cap before its own old turns and forks go first.
    quota: f64 = 0,
    /// TF_GLM_SPILL_TURNS: the newest states kept on a conversation's path (0: every turn until room is needed).
    turns: u32 = 4,
    /// TF_GLM_SPILL_READ_BLOCKS: read-ahead slots (pinned) every restore's blocks are read into.
    read_blocks: u32 = 16,
    /// TF_GLM_SPILL_STAGE_MIB: the pinned ring writes are staged through (whole blocks of it; a tail holds one from
    /// its keep until it is written).
    stage_mib: u32 = 256,
    /// Queued writes' small states (each a whole small state's slot, ~103 MiB with DFlash2), and restores at once.
    smalls: u32 = 4,
    restores: u32 = 2,
    readers: u32 = 4,
    direct: bool = true,
    /// The release guard's limit: released kept states held while their writes still read them (cuda_spill_state.Guard).
    hold_bytes: u64 = 4 << 30,

    pub fn enabled(s: Settings) bool {
        return s.dir.len > 0 and s.gib > 0;
    }
    pub fn capBytes(s: Settings) u64 {
        return @intFromFloat(s.gib * (1 << 30));
    }
    pub fn floorBytes(s: Settings) u64 {
        return @intFromFloat(@max(s.min_free_gib, 0) * (1 << 30));
    }
};

/// The pinned staging of a layout: `stage_mib` of write slots (two at least), `read_blocks` read slots, then the small
/// states' slots. error.StagingTooSmall for a ring under two blocks or a count of 0.
pub fn plan(s: Settings, l: *const st.Layout) !spill.slots.Plan {
    try check(s);
    const bs = std.mem.alignForward(usize, l.block_bytes, spill.slots.alignment);
    const ss = std.mem.alignForward(usize, l.small_bytes, spill.slots.alignment);
    const write = (@as(usize, s.stage_mib) << 20) / bs;
    return .{ .write = write, .read = s.read_blocks, .small = s.smalls, .restore = s.restores, .block_slot = bs, .small_slot = ss };
}

/// The staging settings refused before anything loads: a write ring under two blocks, no read-ahead slot, no small
/// state or restore slot (plan's refusals, independent of the ring and the drafter).
pub fn check(s: Settings) !void {
    const block = std.mem.alignForward(usize, st.Layout.spanBytes(st.block_tokens), spill.slots.alignment);
    if ((@as(usize, s.stage_mib) << 20) / block < 2 or s.read_blocks == 0 or s.smalls == 0 or s.restores == 0) return error.StagingTooSmall;
}

/// Events a rank needs: one a staging slot (stable for the store's life) and one for its own waits.
pub fn eventCount(p: spill.slots.Plan) usize {
    return p.write + p.read + p.small + p.restore + 1;
}

/// The id the prompt cache knows a point by: its key's first 8 bytes (the key is a SHA-256).
pub fn pointId(key: Digest) u64 {
    return std.mem.readInt(u64, key[0..8], .little);
}

/// The longest of a request's chunk starts (in any order) under `at`; 0: none.
fn startBelow(starts: []const u32, at: u32) u32 {
    var best: u32 = 0;
    for (starts) |s| if (s < at and s > best) {
        best = s;
    };
    return best;
}

/// One step of a kept state's lineage, as the prompt cache names it (prompt_cache.Fork, field for field: TierOf).
pub const Fork = struct { name: u64, at: u32 };

/// A durable point found for a prompt (prompt_cache.Point without `bytes`: a restored copy takes the family's).
pub const Point = struct { id: u64, at: u32, tokens: u32, group: u64 = 0, shared: bool = false };

/// A restore job after a poll (prompt_cache.Poll).
pub const Poll = union(enum) {
    pending: struct { done: u64 = 0, total: u64 = 0 },
    ready: struct { saved: *anyopaque, lineage: []const Fork = &.{} },
    failed: anyerror,
};

/// One rank's counters, field for field the prompt cache's TierCounts (TierOf checks both ways), as the `stats` ask
/// carries them (wire.words). Gauges as there.
pub const Counts = struct {
    points: u64 = 0,
    blocks: u64 = 0,
    stored_bytes: u64 = 0,
    cap_bytes: u64 = 0,
    disk_free_bytes: u64 = 0,
    staging_bytes: u64 = 0,
    staging_peak: u64 = 0,
    writes_pending: u64 = 0,
    writes_queued: u64 = 0,
    writes_committed: u64 = 0,
    writes_cancelled: u64 = 0,
    writes_cancelled_guard: u64 = 0,
    writes_cancelled_staging: u64 = 0,
    writes_cancelled_shutdown: u64 = 0,
    writes_failed: u64 = 0,
    forced_staging: u64 = 0,
    no_space: u64 = 0,
    dedup_blocks: u64 = 0,
    dedup_bytes: u64 = 0,
    written_bytes: u64 = 0,
    presence_refused: u64 = 0,
    compatible_misses: u64 = 0,
    crc_errors: u64 = 0,
    size_errors: u64 = 0,
    reads: u64 = 0,
    read_bytes: u64 = 0,
    read_failed: u64 = 0,
    evicted_turns: u64 = 0,
    evicted_forks: u64 = 0,
    evicted_conversations: u64 = 0,
    evicted_shared: u64 = 0,
    quota_trims: u64 = 0,
    recovered_points: u64 = 0,
    dropped_temporaries: u64 = 0,
};

fn monotonicNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// A failed restore's reason as an error the prompt cache logs.
fn failureError(why: wire.Failure) anyerror {
    return switch (why) {
        .presence => error.SpillNotOnEveryRank,
        .reserve => error.SpillReserve,
        .read => error.SpillRead,
        .crc => error.SpillDamaged,
        .truncated => error.SpillTruncated,
        .missing => error.SpillMissing,
        .cancelled => error.Cancelled,
        .timeout => error.SpillTimeout,
        .unpack => error.SpillUnpack,
        .shutdown => error.SpillShutdown,
        .unknown => error.SpillUnknownRestore,
        else => error.SpillAborted,
    };
}

fn readWhy(e: ?spill.store.ReadError) wire.Failure {
    return switch (e orelse return .read) {
        .missing => .missing,
        .damaged, .small => .crc,
        .io => .read,
        .cancelled => .cancelled,
    };
}

/// The kept state type of engine `E` (prefillWith's `kept`).
fn SnapOf(comptime E: type) type {
    const P = @typeInfo(@typeInfo(@TypeOf(E.prefillWith)).@"fn".return_type.?).error_union.payload;
    return @typeInfo(@typeInfo(@FieldType(P, "kept")).optional.child).pointer.child;
}

/// When this rank runs as root (the recipe's containers), its folders in the spill directory are handed to the
/// directory's owner, as the Python tier gave its files (0088): the files inside stay root's (0600), but folders of
/// the owner's let them list and remove everything without sudo. Returns whether they were handed over.
pub fn giveFolders(s: *const spill.Store) bool {
    if (std.c.geteuid() != 0) return false;
    const o = ownerOf(s.root_fd) orelse return false;
    if (o.uid == 0) return false;
    var all = true;
    for ([_]std.c.fd_t{ s.top_fd, s.rank_fd, s.blocks_fd, s.points_fd }) |fd| all = all and std.c.fchown(fd, o.uid, o.gid) == 0;
    return all;
}

fn ownerOf(fd: std.c.fd_t) ?struct { uid: std.c.uid_t, gid: std.c.gid_t } {
    if (builtin.os.tag == .linux) {
        var sx: std.os.linux.Statx = undefined;
        if (std.c.statx(fd, "", 0x1000, .{ .UID = true, .GID = true }, &sx) != 0) return null; // AT_EMPTY_PATH
        return .{ .uid = sx.uid, .gid = sx.gid };
    }
    var s: std.c.Stat = undefined;
    if (std.c.fstat(fd, &s) != 0) return null;
    return .{ .uid = s.uid, .gid = s.gid };
}

// -- one rank's side: its store, staging, events, writes and restores -------------------------------------------------

pub fn Side(comptime E: type) type {
    return struct {
        const Self = @This();
        pub const Snap = SnapOf(E);
        const Guard = st.Guard(Snap);

        /// A kept state's write in flight here, by its Snap id: `snap` while its rows may still be read (null once its
        /// release let it go: nothing of it is staged again).
        const Write = struct { job: spill.JobId, key: Digest, snap: ?*Snap };
        const Copy = struct {
            ev: u32,
            what: union(enum) {
                stage: struct { t: spill.Ticket, snap: u64 },
                capture: struct { job: spill.JobId, snap: u64 },
                read: struct { rx: u64, slot: spill.Slot },
                small: struct { rx: u64 },
            },

            fn source(cp: Copy) ?u64 {
                return switch (cp.what) {
                    .stage => |x| x.snap,
                    .capture => |x| x.snap,
                    else => null,
                };
            }
            fn restore(cp: Copy) ?u64 {
                return switch (cp.what) {
                    .read => |x| x.rx,
                    .small => |x| x.rx,
                    else => null,
                };
            }
        };

        /// A restore on this rank (`id`: its transaction): its reads, its destination (a new kept state) and copies.
        pub const Rx = struct {
            id: u64,
            key: Digest,
            at: u32,
            blocks: u32,
            rid: ?spill.ReadId = null,
            /// this rank's beginRead was refused for capacity (a restore ending here frees its slot): asked again
            retry: bool = false,
            dest: ?*Snap = null,
            un: ?st.Unpacker = null,
            copied: u32 = 0, // blocks whose copy is enqueued (in order)
            inflight: u32 = 0, // of those, copies whose event has not completed
            small: enum { none, pending, done } = .none,
            why: wire.Failure = .none,

            pub fn fail(rx: *Rx, why: wire.Failure) void {
                if (rx.why == .none) rx.why = why;
            }
        };

        pub const KeepArgs = struct {
            keyed: []const u32, // [0, at)
            key: Digest, // format.pointKey(keyed)
            snap: *Snap,
            lineage: format.Lineage,
            origin: ?format.Origin = null,
            shared: bool = false,
            group: u64 = 0,
        };
        pub const Kept = enum { queued, stored, refused };

        gpa: Allocator,
        io: std.Io,
        e: *E,
        store: *spill.Store,
        layout: st.Layout,
        plan: spill.slots.Plan,
        regions: [4][]align(spill.slots.alignment) u8 = undefined,
        lists: [4][1]spill.slots.Buffer = undefined,
        copies: std.ArrayList(Copy) = .empty,
        writes: std.AutoHashMapUnmanaged(u64, Write) = .empty,
        rxs: std.AutoHashMapUnmanaged(u64, *Rx) = .empty,
        reqs: []spill.StageReq = &.{},
        oks: []bool = &.{},
        guard: Guard,
        cached: Counts = .{},
        cancelled_shutdown: u64 = 0,

        /// Opens this rank's store under `s.dir` (namespace `prefix` + `compat`) over `staging` (pinned, at least
        /// plan's bytes, 4 KiB aligned; the caller's), its events made by the engine (eventCount). `layout`: the
        /// engine's (Engine.spillLayout); `max_tokens`: the longest point (the context).
        pub fn open(gpa: Allocator, io: std.Io, e: *E, layout: st.Layout, s: Settings, rank: u32, compat: Digest, max_tokens: u32, staging: []align(spill.slots.alignment) u8, hooks: spill.store.Hooks) !*Self {
            const p = try plan(s, &layout);
            if (staging.len < p.bytes()) return error.StagingTooSmall;
            try e.tierEvents(eventCount(p));
            const self = try gpa.create(Self);
            errdefer gpa.destroy(self);
            self.* = .{ .gpa = gpa, .io = io, .e = e, .store = undefined, .layout = layout, .plan = p, .guard = .{ .limit = s.hold_bytes } };
            var off: usize = 0;
            for ([_]usize{ p.write * p.block_slot, p.read * p.block_slot, p.small * p.small_slot, p.restore * p.small_slot }, 0..) |n, i| {
                self.regions[i] = @alignCast(staging[off..][0..n]);
                self.lists[i] = .{self.regions[i]};
                off += n;
            }
            self.reqs = try gpa.alloc(spill.StageReq, @max(p.write, pump_blocks));
            errdefer gpa.free(self.reqs);
            self.oks = try gpa.alloc(bool, self.reqs.len);
            errdefer gpa.free(self.oks);
            const dir = try gpa.dupeSentinel(u8, s.dir, 0);
            defer gpa.free(dir);
            self.store = try spill.Store.open(gpa, io, .{
                .root = dir,
                .prefix = prefix,
                .rank = rank,
                .compat = compat,
                .limits = layout.limits(max_tokens),
                .cap_bytes = s.capBytes(),
                .floor_bytes = s.floorBytes(),
                .quota = s.quota,
                .turns = s.turns,
                .min_tokens = s.min_tokens,
                .staging = .{ .write = &self.lists[0], .read = &self.lists[1], .small = &self.lists[2], .restore = &self.lists[3] },
                .readers = s.readers,
                .max_reads = @intCast(p.restore),
                .direct = s.direct,
                .hooks = hooks,
            });
            self.refresh();
            return self;
        }

        /// Every copy still on the device waited for, every restore ended, every held state let go, the store closed
        /// (unpublished points stay unpublished).
        pub fn close(self: *Self) void {
            self.waitAll();
            while (self.rxs.count() > 0) {
                var it = self.rxs.valueIterator();
                self.rxEnd(it.next().?.*, .cancel, false);
            }
            self.guard.deinit(self.gpa, self, dropFn);
            self.store.close();
            self.copies.deinit(self.gpa);
            self.writes.deinit(self.gpa);
            self.rxs.deinit(self.gpa);
            self.gpa.free(self.reqs);
            self.gpa.free(self.oks);
            self.gpa.destroy(self);
        }

        fn packer(self: *Self) st.Packer {
            return .{ .layout = &self.layout, .c = self.e.tierCopier() };
        }

        fn slotEvent(self: *const Self, slot: spill.Slot) u32 {
            const p = self.plan;
            const base: usize = switch (slot.pool) {
                .write => 0,
                .read => p.write,
                .small => p.write + p.read,
                .restore => p.write + p.read + p.small,
            };
            return @intCast(base + slot.index);
        }

        /// The event of the slot a store-given slice starts in (Queued.small, SmallView.bytes).
        fn addrEvent(self: *const Self, pool: spill.slots.Pool, bytes: []const u8) u32 {
            const region = self.regions[@backingInt(pool)];
            const per = if (pool == .write or pool == .read) self.plan.block_slot else self.plan.small_slot;
            const i = (@intFromPtr(bytes.ptr) - @intFromPtr(region.ptr)) / per;
            return self.slotEvent(.{ .pool = pool, .index = @intCast(i) });
        }

        /// The event for this rank's own waits (a release's staging).
        fn spareEvent(self: *const Self) u32 {
            return @intCast(eventCount(self.plan) - 1);
        }

        // -- writes ---------------------------------------------------------------------------------------------

        /// The write of a kept state: its sections and its tail copied off its buffer now, its whole blocks staged
        /// later by `service`. Never waits on the disk.
        pub fn keepPoint(self: *Self, k: KeepArgs) Kept {
            const kept = self.e.tierKept(k.snap);
            const at = kept.sec.at;
            if (k.keyed.len != at) return .refused;
            self.layout.check(kept.sec) catch return .refused;
            if (self.writes.contains(k.snap.id)) return .stored;
            self.writes.ensureUnusedCapacity(self.gpa, 1) catch return .refused;
            self.copies.ensureUnusedCapacity(self.gpa, 1) catch return .refused;
            const r = self.store.keep(.{ .keyed = k.keyed, .lineage = k.lineage, .origin = k.origin, .shared = k.shared, .group = k.group, .sections = self.layout.specs(), .tail_bytes = self.layout.tailBytes(at), .source = st.sourceOf(k.snap.id) });
            const q = switch (r) {
                .queued => |q| q,
                .stored => return .stored,
                else => return .refused,
            };
            const ev = self.addrEvent(.small, q.small);
            const pk = self.packer();
            const ok = blk: {
                pk.packSmall(kept, q.small, q.sections) catch break :blk false;
                if (q.tail) |tail| {
                    const sp = self.layout.span(at, (self.layout.blocks(at) catch unreachable) - 1);
                    pk.packBlock(kept, sp.first, sp.tokens, tail) catch break :blk false;
                }
                break :blk true;
            };
            self.e.tierRecord(ev);
            if (!ok) {
                self.e.tierWait(ev); // (a copy that failed partway may still have read the buffer)
                self.store.captured(q.job, false);
                return .refused;
            }
            self.copies.appendAssumeCapacity(.{ .ev = ev, .what = .{ .capture = .{ .job = q.job, .snap = k.snap.id } } });
            self.writes.putAssumeCapacity(k.snap.id, .{ .job = q.job, .key = k.key, .snap = k.snap });
            return .queued;
        }

        /// One staged block's copy (its event the slot's), or the store told it could not be made.
        fn packReq(self: *Self, req: spill.StageReq) void {
            const w = self.writes.get(req.source.id) orelse return self.store.staged(req.ticket, false);
            const snap = w.snap orelse return self.store.staged(req.ticket, false);
            self.copies.ensureUnusedCapacity(self.gpa, 1) catch return self.store.staged(req.ticket, false);
            const ev = self.slotEvent(req.ticket.slot);
            const ok = if (self.packer().packBlock(self.e.tierKept(snap), req.first, req.tokens, req.dst)) true else |_| false;
            self.e.tierRecord(ev);
            if (!ok) {
                self.e.tierWait(ev);
                return self.store.staged(req.ticket, false);
            }
            self.copies.appendAssumeCapacity(.{ .ev = ev, .what = .{ .stage = .{ .t = req.ticket, .snap = snap.id } } });
        }

        /// Kept state `snap` leaves this rank's cache (Tier.release on rank 0, the `.drop` note on the others), before
        /// its free: while its write still reads it the guard holds it (one reference more) or, past its limit, stages
        /// the rest now or cancels the write; either way nothing reads it once the family frees it.
        pub fn release(self: *Self, snap: *Snap) void {
            const w = self.writes.getPtr(snap.id) orelse return;
            if (w.snap == null) return;
            const src = st.sourceOf(snap.id);
            switch (self.guard.release(self.gpa, snap, self.store.pins(src, 0) or self.hasCopies(snap.id))) {
                .free => w.snap = null,
                .held => {},
                .stage => {
                    const r = self.packer().stageRest(self.store, src, self.e.tierKept(snap), self.reqs, self.oks);
                    const ev = self.spareEvent();
                    self.e.tierRecord(ev);
                    self.e.tierWait(ev); // (after every copy of it on the stream: its capture and staged blocks too)
                    for (self.reqs[0..r.staged], self.oks[0..r.staged]) |q, ok| self.store.staged(q.ticket, ok);
                    self.waitCopies(snap.id);
                    w.snap = null;
                },
            }
        }

        /// The point `key`'s queued write here is cancelled, never staged, and nothing reads its state once this
        /// returns (a copy in flight is waited for): Tier.forget's contract.
        pub fn cancelWrite(self: *Self, key: Digest) void {
            var it = self.writes.iterator();
            while (it.next()) |kv| {
                if (!std.mem.eql(u8, &kv.value_ptr.key, &key)) continue;
                _ = self.store.cancel(kv.value_ptr.job);
                self.waitCopies(kv.key_ptr.*);
                kv.value_ptr.snap = null;
            }
        }

        /// Whether a write of point `key` is still queued here (published soon: a `has` answers `incomplete`).
        pub fn writing(self: *const Self, key: Digest) bool {
            var it = self.writes.valueIterator();
            while (it.next()) |w| if (std.mem.eql(u8, &w.key, &key)) return true;
            return false;
        }

        fn hasCopies(self: *const Self, snap: u64) bool {
            for (self.copies.items) |cp| if (cp.source() == snap) return true;
            return false;
        }

        /// Every copy reading kept state `snap` waited for and handed on.
        fn waitCopies(self: *Self, snap: u64) void {
            var i: usize = 0;
            while (i < self.copies.items.len) {
                const cp = self.copies.items[i];
                if (cp.source() != snap) {
                    i += 1;
                    continue;
                }
                self.e.tierWait(cp.ev);
                _ = self.copies.swapRemove(i);
                self.done(cp);
            }
        }

        fn waitAll(self: *Self) void {
            while (self.copies.pop()) |cp| {
                self.e.tierWait(cp.ev);
                self.done(cp);
            }
        }

        /// A copy's event completed: its result to the store.
        fn done(self: *Self, cp: Copy) void {
            switch (cp.what) {
                .stage => |x| self.store.staged(x.t, true),
                .capture => |x| self.store.captured(x.job, true),
                .read => |x| {
                    self.store.readRelease(x.slot);
                    if (self.rxs.get(x.rx)) |rx| rx.inflight -= 1;
                },
                .small => |x| if (self.rxs.get(x.rx)) |rx| {
                    rx.small = .done;
                },
            }
        }

        /// Every copy whose event completed, handed on (never waits); how many.
        pub fn retire(self: *Self) usize {
            var n: usize = 0;
            var i: usize = 0;
            while (i < self.copies.items.len) {
                const cp = self.copies.items[i];
                if (self.e.tierDone(cp.ev)) {
                    _ = self.copies.swapRemove(i);
                    self.done(cp);
                    n += 1;
                } else i += 1;
            }
            return n;
        }

        /// Between rounds, between a fill's slices and at idle: completed copies handed on, at most `max_blocks` more
        /// blocks staged within `budget_ns`, finished writes collected, held states nothing reads let go. Bounded;
        /// never waits. The copies retired and blocks staged (0: nothing moved).
        pub fn service(self: *Self, budget_ns: u64, max_blocks: usize) usize {
            const deadline = monotonicNs() +| budget_ns;
            var moved = self.retire();
            var left = max_blocks;
            while (left > 0 and monotonicNs() < deadline) {
                const n = self.store.stage(self.reqs[0..@min(self.reqs.len, left)]);
                if (n == 0) break;
                for (self.reqs[0..n]) |req| self.packReq(req);
                left -= n;
                moved += n;
            }
            self.collect();
            moved += self.guard.settle(self, idleFn, dropFn);
            self.refresh();
            return moved;
        }

        /// The store's completions: finished writes leave `writes`.
        fn collect(self: *Self) void {
            var out: [16]spill.Completion = undefined;
            while (true) {
                const n = self.store.poll(&out);
                for (out[0..n]) |c| {
                    const job: spill.JobId = switch (c) {
                        inline else => |x| x.job,
                    };
                    var it = self.writes.iterator();
                    while (it.next()) |kv| if (kv.value_ptr.job == job) {
                        _ = self.writes.remove(kv.key_ptr.*);
                        break;
                    };
                }
                if (n < out.len) break;
            }
        }

        fn idleFn(ctx: *anyopaque, snap: *Snap) bool {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return !self.store.pins(st.sourceOf(snap.id), 0) and !self.hasCopies(snap.id);
        }

        fn dropFn(ctx: *anyopaque, snap: *Snap) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            if (self.writes.getPtr(snap.id)) |w| w.snap = null;
            self.e.freeSnapshot(snap);
        }

        /// Copies on the device, writes waiting for the engine, or held states to let go.
        pub fn busy(self: *Self) bool {
            return self.copies.items.len > 0 or self.guard.held.items.len > 0 or self.store.pendingStage() > 0;
        }

        /// A clean stop: every queued write staged and drained by `deadline_ns` (monotonic); the points left
        /// unpublished (never published to make the count).
        pub fn flush(self: *Self, deadline_ns: u64) u64 {
            while (monotonicNs() < deadline_ns and (self.store.pendingStage() > 0 or self.copies.items.len > 0)) {
                // nothing retired or staged (the device busy, or every write slot still held by the writer): sleep
                if (self.service(deadline_ns -| monotonicNs(), std.math.maxInt(usize)) == 0) std.Io.sleep(self.io, .fromMilliseconds(1), .awake) catch {};
            }
            self.waitAll();
            self.collect();
            const d = self.store.drain(deadline_ns -| monotonicNs());
            self.cancelled_shutdown += d.unpublished;
            self.collect();
            _ = self.guard.settle(self, idleFn, dropFn);
            self.refresh();
            return d.unpublished;
        }

        // -- restores -------------------------------------------------------------------------------------------

        pub fn newRx(self: *Self, id: u64, key: Digest, at: u32) !*Rx {
            const blocks = self.layout.blocks(at) catch return error.BadPoint;
            try self.rxs.ensureUnusedCapacity(self.gpa, 1);
            const rx = try self.gpa.create(Rx);
            rx.* = .{ .id = id, .key = key, .at = at, .blocks = @intCast(blocks) };
            self.rxs.putAssumeCapacity(id, rx);
            return rx;
        }

        /// This rank's reads of `rx`'s point (a refusal for capacity is asked again at the next step: `retry`).
        pub fn rxRead(self: *Self, rx: *Rx) void {
            if (rx.rid != null or rx.why != .none) return;
            rx.rid = self.store.beginRead(rx.key) catch |e| {
                switch (e) {
                    error.NoSlot, error.TooMany => rx.retry = true,
                    error.Missing => rx.fail(.missing),
                    else => rx.fail(.read),
                }
                return;
            };
            rx.retry = false;
        }

        /// `rx`'s destination: a new kept state of the point's size (the engine's, under its keep floor).
        pub fn rxDest(self: *Self, rx: *Rx) void {
            if (rx.why != .none or rx.dest != null) return;
            const snap = self.e.tierDest(rx.at) catch return rx.fail(.reserve);
            rx.dest = snap;
            rx.un = st.Unpacker.init(&self.layout, self.e.tierCopier(), self.e.tierKept(snap)) catch return rx.fail(.reserve);
        }

        /// Ready blocks before `upto` (null: all) copied into the destination, within `budget_ns`; never waits.
        pub fn rxStep(self: *Self, rx: *Rx, upto: ?u32, budget_ns: u64) void {
            if (rx.retry) self.rxRead(rx);
            const rid = rx.rid orelse return;
            if (rx.un == null) return;
            const stop = @min(upto orelse rx.blocks, rx.blocks);
            const deadline = monotonicNs() +| budget_ns;
            while (rx.why == .none and rx.copied < stop) {
                var one: [1]spill.ReadyBlock = undefined;
                const got = self.store.readReady(rid, stop, &one);
                if (got.n == 0) {
                    if (got.status == .failed) rx.fail(readWhy(self.store.readFailure(rid).err));
                    return;
                }
                const b = one[0];
                self.copies.ensureUnusedCapacity(self.gpa, 1) catch {
                    self.store.readRelease(b.slot);
                    return rx.fail(.unpack);
                };
                const ev = self.slotEvent(b.slot);
                const ok = if (rx.un.?.block(b.first, b.tokens, b.data)) true else |_| false;
                self.e.tierRecord(ev);
                if (!ok) {
                    self.e.tierWait(ev);
                    self.store.readRelease(b.slot);
                    return rx.fail(.unpack);
                }
                self.copies.appendAssumeCapacity(.{ .ev = ev, .what = .{ .read = .{ .rx = rx.id, .slot = b.slot } } });
                rx.inflight += 1;
                rx.copied = b.index + 1;
                if (monotonicNs() >= deadline) return;
            }
        }

        /// With every block copied: the sections, once read and checked, into the destination, and its pooled rows at/4
        /// zeroed (Unpacker.finish).
        pub fn rxSmall(self: *Self, rx: *Rx) void {
            if (rx.why != .none or rx.small != .none or rx.copied < rx.blocks) return;
            const rid = rx.rid orelse return;
            const view = self.store.readSmallView(rid) orelse {
                const f = self.store.readFailure(rid);
                if (f.err != null) rx.fail(readWhy(f.err));
                return;
            };
            self.copies.ensureUnusedCapacity(self.gpa, 1) catch return rx.fail(.unpack);
            const ev = self.addrEvent(.restore, view.bytes);
            const un = &rx.un.?;
            const ok = blk: {
                un.smallState(view.bytes, view.sections) catch break :blk false;
                un.finish() catch break :blk false;
                break :blk true;
            };
            self.e.tierRecord(ev);
            if (!ok) {
                self.e.tierWait(ev);
                return rx.fail(.unpack);
            }
            self.copies.appendAssumeCapacity(.{ .ev = ev, .what = .{ .small = .{ .rx = rx.id } } });
            rx.small = .pending;
        }

        /// Every block and the sections installed (their events completed).
        pub fn rxReady(rx: *const Rx) bool {
            return rx.why == .none and rx.copied == rx.blocks and rx.inflight == 0 and rx.small == .done;
        }

        fn rxCopies(self: *const Self, rx: *const Rx) bool {
            for (self.copies.items) |cp| if (cp.restore() == rx.id) return true;
            return false;
        }

        /// Every copy of `rx` on the device waited for and handed on.
        fn rxQuiesce(self: *Self, rx: *const Rx) void {
            var i: usize = 0;
            while (i < self.copies.items.len) {
                const cp = self.copies.items[i];
                if (cp.restore() != rx.id) {
                    i += 1;
                    continue;
                }
                self.e.tierWait(cp.ev);
                _ = self.copies.swapRemove(i);
                self.done(cp);
            }
        }

        /// Every copy of `rx` completed by `deadline_ns`, polling its events (never an unbounded device wait).
        fn rxSettle(self: *Self, rx: *const Rx, deadline_ns: u64) bool {
            while (true) {
                _ = self.retire();
                if (!self.rxCopies(rx)) return true;
                if (monotonicNs() >= deadline_ns) return false;
                std.Io.sleep(self.io, .fromMicroseconds(200), .awake) catch {};
            }
        }

        fn recycle(ctx: ?*anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx.?));
            _ = self.retire();
        }

        /// Every block and the sections copied and installed by `deadline_ns`, with bounded waits that recycle this
        /// rank's read slots meanwhile (PR 124's follower hang); a deadline passed is `timeout`. `rx.why` tells.
        pub fn rxFinish(self: *Self, rx: *Rx, deadline_ns: u64) void {
            while (rx.why == .none and (rx.rid == null or rx.copied < rx.blocks)) {
                _ = self.retire();
                self.rxStep(rx, rx.blocks, 5 * std.time.ns_per_ms);
                if (rx.why != .none or (rx.rid != null and rx.copied >= rx.blocks)) break;
                const now = monotonicNs();
                if (now >= deadline_ns) {
                    rx.fail(.timeout);
                    break;
                }
                // (no read yet: its slot comes back when another restore's copies end)
                if (rx.rid) |rid| {
                    _ = self.store.readAwait(rid, rx.copied, @min(deadline_ns - now, 50 * std.time.ns_per_ms), .{ .ctx = self, .recycle = recycle });
                } else std.Io.sleep(self.io, .fromMilliseconds(1), .awake) catch {};
            }
            while (rx.why == .none and rx.small == .none) {
                _ = self.retire();
                self.rxSmall(rx);
                if (rx.why != .none or rx.small != .none) break;
                if (monotonicNs() >= deadline_ns) {
                    rx.fail(.timeout);
                    break;
                }
                std.Io.sleep(self.io, .fromMicroseconds(200), .awake) catch {};
            }
            if (rx.why == .none and !self.rxSettle(rx, deadline_ns)) rx.fail(.timeout);
        }

        /// A restore ends without a commit: its copies waited for, the point forgotten here when `forget` (with this
        /// rank's own damaged and missing block lists), its reads ended, its destination freed.
        pub fn rxEnd(self: *Self, rx: *Rx, outcome: spill.Store.Outcome, forget: bool) void {
            self.rxQuiesce(rx);
            if (rx.rid) |rid| {
                if (forget) {
                    const f = self.store.readFailure(rid);
                    self.store.forget(rx.key, f.bad, f.lost);
                }
                _ = self.store.endRead(rid, outcome) catch {};
            } else if (forget) self.store.forget(rx.key, &.{}, &.{});
            if (rx.dest) |d| self.e.freeSnapshot(d);
            self.rxDrop(rx);
        }

        /// A restore's reads end as committed: its lineage (gpa-owned). The destination stays `rx`'s (rxTake).
        pub fn rxCommit(self: *Self, rx: *Rx) !spill.ReadDone {
            const rid = rx.rid orelse return error.Incomplete;
            return (try self.store.endRead(rid, .commit)) orelse error.Incomplete;
        }

        /// A committed restore's destination, now the caller's; `rx` is dropped.
        pub fn rxTake(self: *Self, rx: *Rx) *Snap {
            const d = rx.dest.?;
            rx.dest = null;
            self.rxDrop(rx);
            return d;
        }

        pub fn rxDrop(self: *Self, rx: *Rx) void {
            _ = self.rxs.remove(rx.id);
            self.gpa.destroy(rx);
        }

        // -- counters -------------------------------------------------------------------------------------------

        /// The cached counters brought up to date (by service and flush: the prompt cache's counts never lock the store).
        pub fn refresh(self: *Self) void {
            const c = self.store.counts();
            self.cached = .{
                .points = c.points,
                .blocks = c.blocks,
                .stored_bytes = c.stored_bytes,
                .cap_bytes = self.store.cfg.cap_bytes,
                .disk_free_bytes = c.disk_free_bytes,
                .staging_bytes = c.staging_bytes,
                .staging_peak = c.staging_peak_bytes,
                .writes_pending = c.pending_jobs,
                .writes_queued = c.queued,
                .writes_committed = c.committed,
                .writes_cancelled = c.cancelled,
                .writes_cancelled_guard = c.cancelled_guard + c.cancelled_source,
                .writes_cancelled_staging = c.no_staging + c.backlog_refused,
                .writes_cancelled_shutdown = self.cancelled_shutdown,
                .writes_failed = c.write_failed,
                .forced_staging = c.guard_staged,
                .no_space = c.no_space,
                .dedup_blocks = c.dedup_blocks,
                .dedup_bytes = c.dedup_bytes,
                .written_bytes = c.written_bytes + c.written_small_bytes,
                .crc_errors = c.crc_errors,
                .size_errors = c.size_errors + c.missing_errors,
                .reads = c.loaded,
                .read_bytes = c.loaded_bytes,
                .read_failed = c.load_failed,
                .evicted_turns = c.pruned_turns + c.trimmed_turns,
                .evicted_forks = c.pruned_forks,
                .evicted_conversations = c.pruned_conversations,
                .evicted_shared = c.pruned_shared,
                .quota_trims = c.quota_trims,
                .recovered_points = c.recovered_points,
                .dropped_temporaries = c.temps_removed + c.recovered_dropped,
            };
        }
    };
}

// -- rank 0 -------------------------------------------------------------------------------------------------------------

pub fn Leader(comptime E: type) type {
    return struct {
        const Self = @This();
        pub const S = Side(E);
        pub const L = lane_backend.Lanes(E);

        /// The prompt cache's Tier over this Leader (`pc`: the engine API's prompt_cache; TierOf).
        pub fn Tier(comptime pc: type) type {
            return TierOf(Self, pc);
        }

        const Job = struct { rx: *S.Rx, dest: *L.Kept, point: u64, shared: bool, sent: u32 = 0 };
        const PointRef = struct { key: Digest, at: u32 };

        /// A kept state the prompt cache writes (prompt_cache.Write): `saved` is the lane backend's Kept.
        pub const Write = struct {
            saved: *anyopaque,
            key: []const u32,
            at: u32,
            group: u64 = 0,
            shared: bool = false,
            parent: ?u64 = null,
            fork: u32 = 0,
            lineage: []const Fork = &.{},
        };

        gpa: Allocator,
        side: *S,
        lanes: *L,
        min_tokens: u32,
        max_restores: u32,
        jobs: std.AutoHashMapUnmanaged(u64, *Job) = .empty,
        next_txn: u64 = 1,
        points: std.AutoHashMapUnmanaged(u64, PointRef) = .empty,
        fwd: std.AutoHashMapUnmanaged(u64, [16]u8) = .empty, // lineage names: the cache's u64s to the store's
        rev: std.AutoHashMapUnmanaged([16]u8, u64) = .empty,
        next_name: u64 = 0xC000_0000_0000_0000,
        boot: [16]u8,
        lin: std.ArrayList(Fork) = .empty, // a ready restore's lineage (valid during its poll)
        peers: [wire.max_world - 1]Counts = @splat(.{}), // the other ranks' counters (their last stats answers)
        peers_n: usize = 0,
        stats_at: u64 = 0,
        /// rank 0 sent the other ranks something since its last stats ask, so their counters may have moved (true at
        /// open: their stores recovered their points)
        stale: bool = true,
        /// the other ranks may still stage writes rank 0 noted (cleared once their counters show none pending)
        peer_writes: bool = false,
        pump_at: u64 = 0,
        presence_refused: u64 = 0,
        misses: u64 = 0,

        /// `side` is moved in (closed by destroy). `boot` salts this process's lineage names.
        pub fn create(gpa: Allocator, side: *S, lanes: *L, s: Settings, boot: [16]u8) !*Self {
            const self = try gpa.create(Self);
            self.* = .{ .gpa = gpa, .side = side, .lanes = lanes, .min_tokens = s.min_tokens, .max_restores = @intCast(@max(1, @min(s.restores, side.plan.restore))), .boot = boot };
            return self;
        }

        /// Restores still in flight ended (no notes: the pair is stopping), the side closed.
        pub fn destroy(self: *Self) void {
            var it = self.jobs.valueIterator();
            while (it.next()) |j| {
                self.side.rxEnd(j.*.rx, .cancel, false);
                self.lanes.unreserve(j.*.dest);
                self.gpa.destroy(j.*);
            }
            self.jobs.deinit(self.gpa);
            self.side.close();
            self.points.deinit(self.gpa);
            self.fwd.deinit(self.gpa);
            self.rev.deinit(self.gpa);
            self.lin.deinit(self.gpa);
            self.gpa.destroy(self);
        }

        /// The other ranks (Flash Next's mirror) hear this tier's notes and answer its asks.
        fn paired(self: *const Self) bool {
            return self.lanes.notes != null;
        }

        /// A note to every other rank (cuda_lanes.spillNote); their counters are stale from here.
        fn note(self: *Self, m: wire.Msg, eager: bool) !void {
            self.stale = true;
            return self.lanes.spillNote(m, eager);
        }

        /// An ask every rank answers (has, validate: cuda_lanes.spillAsk); their counters are stale from here.
        fn ask(self: *Self, m: wire.Msg, mine: []const u64, out: []u64) !usize {
            self.stale = true;
            return self.lanes.spillAsk(m, mine, out);
        }

        // -- lineage names --------------------------------------------------------------------------------------

        fn name16(self: *const Self, n: u64) [16]u8 {
            if (self.fwd.get(n)) |x| return x;
            var h = std.crypto.hash.sha2.Sha256.init(.{});
            h.update("tf-glm-lineage");
            h.update(&self.boot);
            h.update(std.mem.asBytes(&n));
            return h.finalResult()[0..16].*;
        }

        fn nameOf(self: *Self, n: [16]u8) !u64 {
            if (self.rev.get(n)) |x| return x;
            try self.fwd.ensureUnusedCapacity(self.gpa, 1);
            try self.rev.ensureUnusedCapacity(self.gpa, 1);
            const id = self.next_name;
            self.next_name += 1;
            self.fwd.putAssumeCapacity(id, n);
            self.rev.putAssumeCapacity(n, id);
            return id;
        }

        /// The store's lineage of a cache lineage (its forks in `out`); none: a root of its own.
        fn toLineage(self: *Self, forks: []const Fork, out: *std.ArrayList(format.Fork)) !format.Lineage {
            if (forks.len == 0) {
                self.next_name += 1;
                return .{ .root = self.name16(self.next_name) };
            }
            for (forks[1..]) |f| try out.append(self.gpa, .{ .name = self.name16(f.name), .at = f.at });
            return .{ .root = self.name16(forks[0].name), .forks = out.items };
        }

        fn fromLineage(self: *Self, l: format.Lineage) ![]const Fork {
            self.lin.clearRetainingCapacity();
            try self.lin.append(self.gpa, .{ .name = try self.nameOf(l.root), .at = 0 });
            for (l.forks) |f| try self.lin.append(self.gpa, .{ .name = try self.nameOf(f.name), .at = f.at });
            return self.lin.items;
        }

        // -- the prompt cache's tier ----------------------------------------------------------------------------

        /// The longest point `key` extends past `beyond`, at `max_at` at most, on the grid (and on one of `starts` when
        /// any): rank 0's own index, no disk I/O. A longer point off the starts is passed over for the longest start
        /// under it, and the index asked again there.
        pub fn lookup(self: *Self, key: []const u32, beyond: u32, max_at: u32, starts: []const u32, lookahead: u32) ?Point {
            var top = max_at;
            const found = while (top > beyond) {
                const h = self.side.store.lookup(.{ .keyed = key, .longer_than = beyond, .max_at = top, .grid = st.grid }) orelse break null;
                if (starts.len == 0 or std.mem.indexOfScalar(u32, starts, h.at) != null) break h;
                top = startBelow(starts, h.at);
            } else null;
            const hit = found orelse {
                self.misses += 1;
                return null;
            };
            const id = pointId(hit.key);
            self.points.put(self.gpa, id, .{ .key = hit.key, .at = hit.at }) catch return null;
            return .{ .id = id, .at = hit.at, .tokens = hit.at + lookahead, .group = hit.group, .shared = hit.shared };
        }

        /// The write of a state the prompt cache kept (inside its pass): this rank's, and a `keep` note for every
        /// other rank's own copy (a dedup too, so a rank whose store lost the point writes it again). Its point's id,
        /// or null when nothing was written.
        pub fn keep(self: *Self, w: Write) ?u64 {
            self.side.e.current();
            const at = w.at;
            if (at == 0 or at % st.grid != 0 or at < self.min_tokens or w.key.len < at) return null;
            const kp: *L.Kept = @ptrCast(@alignCast(w.saved));
            if (kp.at != at) return null;
            var forks: std.ArrayList(format.Fork) = .empty;
            defer forks.deinit(self.gpa);
            const lin = self.toLineage(w.lineage, &forks) catch return null;
            if (lin.forks.len > wire.max_forks) return null; // (a keep note carries no more: no rank holds a point another lacks)
            const origin: ?format.Origin = if (w.parent) |p| (if (self.points.get(p)) |r| .{ .key = r.key, .at = w.fork } else null) else null;
            const keyed = w.key[0..at];
            const key = format.pointKey(keyed, @intCast(st.block_tokens), &.{});
            // (room first: once the write is queued nothing may fail, or the cache, holding no point, never releases
            // the state the write reads)
            self.points.ensureUnusedCapacity(self.gpa, 1) catch return null;
            if (self.side.keepPoint(.{ .keyed = keyed, .key = key, .snap = kp.snap, .lineage = lin, .origin = origin, .shared = w.shared, .group = w.group }) == .refused) return null;
            const id = pointId(key);
            self.points.putAssumeCapacity(id, .{ .key = key, .at = at });
            if (self.paired()) {
                self.note(.{ .keep = .{ .kept = kp.key, .at = at, .point = key, .group = w.group, .shared = w.shared, .lineage = lin, .origin = origin } }, false) catch {};
                self.peer_writes = true;
            }
            return id;
        }

        /// The point's memory twin `saved` is about to be freed (the store's free, before the family's drop).
        pub fn release(self: *Self, saved: *anyopaque) void {
            self.side.e.current();
            const kp: *L.Kept = @ptrCast(@alignCast(saved));
            self.side.release(kp.snap);
        }

        pub fn share(self: *Self, id: u64) void {
            const r = self.points.get(id) orelse return;
            self.side.store.markShared(r.key);
            self.note(.{ .share = r.key }, false) catch {};
        }

        pub fn touch(self: *Self, id: u64) void {
            const r = self.points.get(id) orelse return;
            self.side.store.touch(r.key);
            self.note(.{ .touch = r.key }, false) catch {};
        }

        /// The point is unusable: dropped here and on every rank, a queued write of it cancelled (never staged).
        pub fn forget(self: *Self, id: u64) void {
            self.side.e.current();
            const r = self.points.fetchRemove(id) orelse return;
            self.side.cancelWrite(r.value.key);
            self.side.store.forget(r.value.key, &.{}, &.{});
            self.note(.{ .forget = r.value.key }, false) catch {};
        }

        /// begin's capacity refusals, asked before the prompt cache makes room: the restores in flight at their cap,
        /// or this rank's store out of reads or restore slots.
        pub fn full(self: *Self) bool {
            return self.jobs.count() >= self.max_restores or self.side.store.readsFull();
        }

        /// begin's refusals of point `id` itself that no room cures, from this rank's own records with nothing read,
        /// sent or reserved: UnknownPoint and BadPoint (not a point lookup gave), SpillMissing (gone from this rank's
        /// store: forgotten here). The prompt cache asks it before it makes room for the restore
        /// (prompt_cache.Tier.VTable.check), so such a refusal evicts no kept state; begin asks it again.
        pub fn check(self: *Self, id: u64, at: u32) !void {
            _ = try self.checked(id, at);
        }

        fn checked(self: *Self, id: u64, at: u32) !PointRef {
            const side = self.side;
            const ref = self.points.get(id) orelse return error.UnknownPoint;
            if (at != ref.at) return error.BadPoint;
            if (!side.store.presence(ref.key)) {
                _ = self.points.remove(id);
                return error.SpillMissing;
            }
            return ref;
        }

        /// A restore of point `id` (found by lookup): `check`'s refusals first, then this rank's reads (TooMany,
        /// NoSlot: refused for capacity, nothing sent), then `has` on every rank (a rank that lacks the point or holds
        /// it damaged makes every rank forget it), then the destination, and a `begin` note that has every other rank
        /// reserve its own under the same key and start its reads. Its transaction.
        pub fn begin(self: *Self, id: u64, at: u32, shared: bool) !u64 {
            self.side.e.current();
            const side = self.side;
            if (self.jobs.count() >= self.max_restores) return error.TooManyRestores;
            const ref = try self.checked(id, at);
            try self.jobs.ensureUnusedCapacity(self.gpa, 1);
            const txn = self.next_txn;
            const rx = try side.newRx(txn, ref.key, at);
            rx.rid = side.store.beginRead(ref.key) catch |e| {
                side.rxDrop(rx);
                return e;
            };
            const mine = [1]u64{@backingInt(wire.Absence.present)};
            var all: [wire.max_world]u64 = undefined;
            const ranks = self.ask(.{ .has = .{ .point = ref.key, .at = at, .blocks = rx.blocks } }, &mine, &all) catch |e| {
                side.rxEnd(rx, .cancel, false);
                return e;
            };
            const v = wire.presence(all[0..ranks]);
            if (!v.ok) {
                self.presence_refused += 1;
                side.rxEnd(rx, .cancel, v.forget);
                if (v.forget) _ = self.points.remove(id);
                return error.SpillNotOnEveryRank;
            }
            side.rxDest(rx);
            if (rx.why != .none) {
                side.rxEnd(rx, .cancel, false);
                return error.SpillReserve;
            }
            const dest = self.lanes.reserveKept(null) catch |e| {
                side.rxEnd(rx, .cancel, false);
                return e;
            };
            const job = self.gpa.create(Job) catch |e| {
                self.lanes.unreserve(dest);
                side.rxEnd(rx, .cancel, false);
                return e;
            };
            job.* = .{ .rx = rx, .dest = dest, .point = id, .shared = shared };
            self.note(.{ .begin = .{ .txn = txn, .kept = dest.key, .point = ref.key, .at = at, .blocks = rx.blocks } }, true) catch |e| {
                self.gpa.destroy(job);
                self.lanes.unreserve(dest);
                side.rxEnd(rx, .cancel, false);
                return e;
            };
            self.next_txn += 1;
            self.jobs.putAssumeCapacity(txn, job);
            return txn;
        }

        /// A restore's next copies within `budget_ns`, a `progress` note when they moved (the other ranks copy as far
        /// now: they have no idle hook), and once this rank's copies are installed its read committed and `validate`
        /// asked: ready on every rank, or failed everywhere.
        pub fn poll(self: *Self, txn: u64, budget_ns: u64) Poll {
            self.side.e.current();
            const side = self.side;
            const job = self.jobs.get(txn) orelse return .{ .failed = error.UnknownJob };
            const rx = job.rx;
            _ = side.retire();
            side.rxStep(rx, null, budget_ns);
            side.rxSmall(rx);
            if (rx.why != .none) return self.fail(txn, job, rx.why, .abort);
            if (rx.copied > job.sent) {
                self.note(.{ .progress = .{ .txn = txn, .upto = rx.copied } }, true) catch return self.fail(txn, job, .read, .abort);
                job.sent = rx.copied;
            }
            if (!S.rxReady(rx)) return .{ .pending = .{ .done = rx.copied, .total = rx.blocks } };
            // this rank's read ends first: a failure there is no damage (every rank aborts, none forgets)
            var done_ = side.rxCommit(rx) catch return self.fail(txn, job, .read, .abort);
            defer done_.lineage.deinit(self.gpa);
            _ = self.jobs.remove(txn);
            const at = rx.at;
            const snap = side.rxTake(rx);
            const dest = job.dest;
            const point = job.point;
            const mine = [1]u64{@backingInt(wire.Failure.none)};
            var all: [wire.max_world]u64 = undefined;
            const v: wire.Verdict = blk: {
                const ranks = self.ask(.{ .validate = .{ .txn = txn, .shared = job.shared } }, &mine, &all) catch break :blk .{ .ok = false, .code = @backingInt(wire.Failure.read) };
                break :blk wire.validation(all[0..ranks]);
            };
            self.gpa.destroy(job);
            if (!v.ok) {
                if (v.forget) {
                    side.store.forget(done_.key, &.{}, &.{});
                    _ = self.points.remove(point);
                }
                side.e.freeSnapshot(snap);
                self.lanes.unreserve(dest);
                return .{ .failed = failureError(@fromBackingInt(v.code)) };
            }
            self.lanes.adoptReserved(dest, snap, at, &.{}) catch |e| {
                side.e.freeSnapshot(snap);
                self.lanes.forgetReserved(dest); // (every other rank committed its copy under this key)
                return .{ .failed = e };
            };
            const lin = self.fromLineage(done_.lineage) catch &.{};
            return .{ .ready = .{ .saved = dest, .lineage = lin } };
        }

        /// A job ends without a commit: here (its copies waited for, its destination freed) and on every other rank
        /// (a `cancel` note, with `forget` when the point is damaged).
        fn fail(self: *Self, txn: u64, job: *Job, why: wire.Failure, outcome: spill.Store.Outcome) Poll {
            _ = self.jobs.remove(txn);
            const gone = wire.damage(why);
            self.note(.{ .cancel = .{ .txn = txn, .why = why, .forget = gone } }, true) catch {};
            self.side.rxEnd(job.rx, outcome, gone);
            self.lanes.unreserve(job.dest);
            if (gone) _ = self.points.remove(job.point);
            self.gpa.destroy(job);
            return .{ .failed = failureError(why) };
        }

        pub fn cancel(self: *Self, txn: u64) void {
            self.side.e.current();
            const job = self.jobs.get(txn) orelse return;
            _ = self.fail(txn, job, .cancelled, .cancel);
        }

        /// Between rounds and at idle: this rank's writes staged (4 blocks at most) and, at most every pump_every_ns
        /// while anything is in flight, a `pump` note so every other rank stages its own as long (with the next round's
        /// notes while streams decode, at once when none does); every rank's counters at most once a second while they
        /// may have moved.
        pub fn pump(self: *Self, budget_ns: u64) void {
            self.side.e.current();
            const moved = self.side.service(budget_ns, pump_blocks);
            if (!self.paired()) return;
            const now = monotonicNs();
            if ((moved > 0 or self.side.busy() or self.peer_writes) and now -| self.pump_at >= pump_every_ns) {
                self.pump_at = now;
                self.note(.{ .pump = budget_ns }, self.lanes.lanes.count() == 0) catch {};
            }
            self.askStats(now);
        }

        /// Every rank's counters (a gather), at most once a second and only while they may have moved (`stale`, or
        /// writes pending there): an idle pair asks nothing, since the gather syncs both ranks' engine streams and wakes
        /// rank 1. The other ranks' pending writes clear `peer_writes`.
        fn askStats(self: *Self, now: u64) void {
            if (!self.stale and !self.peer_writes) return;
            if (self.stats_at != 0 and now -| self.stats_at < stats_every_ns) return;
            self.stats_at = now;
            var mine: [wire.stats_words]u64 = undefined;
            wire.words(Counts, &self.side.cached, &mine);
            var all: [wire.max_world * wire.stats_words]u64 = undefined;
            const ranks = self.lanes.spillAsk(.stats, &mine, &all) catch return;
            self.stale = false;
            var pending = false;
            for (1..ranks) |r| {
                self.peers[r - 1] = wire.fromWords(Counts, all[r * wire.stats_words ..][0..wire.stats_words]);
                pending = pending or self.peers[r - 1].writes_pending > 0;
            }
            self.peers_n = ranks - 1;
            if (!pending) self.peer_writes = false;
        }

        /// Writes still to stage here or on another rank, copies on the device, held states to let go.
        pub fn busy(self: *Self) bool {
            return self.side.busy() or self.peer_writes;
        }

        /// A clean stop (engine thread): restores ended everywhere, a `flush` note (every rank drains its own writes
        /// meanwhile), this rank's drained within `budget_ns`, then a stats ask, which every other rank answers only
        /// after its own drain. The points left unpublished here.
        pub fn flush(self: *Self, budget_ns: u64) u64 {
            self.side.e.current();
            const deadline = monotonicNs() +| budget_ns;
            while (self.jobs.count() > 0) {
                var it = self.jobs.iterator();
                const kv = it.next().?;
                _ = self.fail(kv.key_ptr.*, kv.value_ptr.*, .shutdown, .cancel);
            }
            self.note(.{ .flush = budget_ns }, true) catch {};
            const left = self.side.flush(deadline);
            if (self.paired()) {
                self.stats_at = 0;
                self.askStats(monotonicNs());
            }
            return left;
        }

        /// Each rank's counters into `out` (this rank first, the others' as their last stats answers carried them).
        pub fn counts(self: *const Self, out: []Counts) usize {
            if (out.len == 0) return 0;
            var c = self.side.cached;
            c.presence_refused = self.presence_refused;
            c.compatible_misses = self.misses;
            out[0] = c;
            const n = @min(self.peers_n, out.len - 1);
            @memcpy(out[1..][0..n], self.peers[0..n]);
            return 1 + n;
        }
    };
}

// -- the other ranks ------------------------------------------------------------------------------------------------------

/// A following rank's tier (cuda_lanes.SpillSide): rank 0's notes and asks in its call order. A note handler never
/// waits on a disk beyond its bounds and records a restore's failure for `validate` instead of failing; an answer never
/// fails (every rank must reach the gather).
pub fn Follower(comptime E: type) type {
    return struct {
        const Self = @This();
        pub const S = Side(E);
        pub const L = lane_backend.Lanes(E);
        const Fx = struct { rx: *S.Rx, dest: *L.Kept, allowed: u32 = 0 };

        gpa: Allocator,
        side: *S,
        lanes: *L,
        restores: std.AutoHashMapUnmanaged(u64, Fx) = .empty, // by transaction
        skipped: u64 = 0, // keeps this rank wrote no copy of (its ids or state not rank 0's, or its store refused)

        /// `side` is moved in (closed by destroy).
        pub fn create(gpa: Allocator, side: *S, lanes: *L) !*Self {
            const self = try gpa.create(Self);
            self.* = .{ .gpa = gpa, .side = side, .lanes = lanes };
            return self;
        }

        pub fn destroy(self: *Self) void {
            var it = self.restores.valueIterator();
            while (it.next()) |fx| {
                self.side.rxEnd(fx.rx, .cancel, false);
                self.lanes.unreserve(fx.dest);
            }
            self.restores.deinit(self.gpa);
            self.side.close();
            self.gpa.destroy(self);
        }

        pub fn spillSide(self: *Self) lane_backend.SpillSide {
            return .{ .ptr = self, .note = noteFn, .answer = answerFn, .settle = settleFn, .release = releaseFn };
        }

        fn of(ptr: *anyopaque) *Self {
            return @ptrCast(@alignCast(ptr));
        }

        fn noteFn(ptr: *anyopaque, m: *const wire.Msg, kept: ?*anyopaque) anyerror!void {
            const self = of(ptr);
            const side = self.side;
            _ = side.retire();
            switch (m.*) {
                .keep => |k| {
                    const kp: *L.Kept = @ptrCast(@alignCast(kept orelse return error.KeepWithoutState));
                    // this rank's own ids must name rank 0's point, or a restore could never agree on it
                    if (kp.at != k.at or kp.tokens.len < k.at or !std.mem.eql(u8, &format.pointKey(kp.tokens[0..k.at], @intCast(st.block_tokens), &.{}), &k.point)) {
                        self.skipped += 1;
                        return;
                    }
                    if (side.keepPoint(.{ .keyed = kp.tokens[0..k.at], .key = k.point, .snap = kp.snap, .lineage = k.lineage, .origin = k.origin, .shared = k.shared, .group = k.group }) == .refused) self.skipped += 1;
                },
                .begin => |b| try self.begin(b),
                .progress => |p| {
                    const fx = self.restores.getPtr(p.txn) orelse return error.UnknownRestore;
                    fx.allowed = @max(fx.allowed, p.upto);
                    side.rxStep(fx.rx, fx.allowed, progress_ns);
                    if (fx.allowed >= fx.rx.blocks) side.rxSmall(fx.rx);
                },
                .cancel => |c| {
                    const kv = self.restores.fetchRemove(c.txn) orelse return;
                    side.rxEnd(kv.value.rx, if (c.why == .cancelled or c.why == .shutdown) .cancel else .abort, c.forget);
                    self.lanes.unreserve(kv.value.dest);
                },
                .touch => |p| side.store.touch(p),
                .share => |p| side.store.markShared(p),
                .forget => |p| {
                    side.cancelWrite(p);
                    side.store.forget(p, &.{}, &.{});
                },
                .pump => |ns| _ = side.service(ns, pump_blocks),
                .flush => |ns| _ = side.flush(monotonicNs() +| ns),
                .has, .validate, .stats => return error.AskAsNote,
            }
        }

        /// Rank 0's restore here: the destination's wrapper under rank 0's key, this rank's reads, its destination.
        fn begin(self: *Self, b: wire.Begin) !void {
            const side = self.side;
            try self.restores.ensureUnusedCapacity(self.gpa, 1);
            const dest = try self.lanes.reserveKept(b.kept);
            errdefer self.lanes.unreserve(dest);
            const rx = try side.newRx(b.txn, b.point, b.at);
            if (rx.blocks != b.blocks) rx.fail(.unpack);
            side.rxRead(rx);
            side.rxDest(rx);
            self.restores.putAssumeCapacity(b.txn, .{ .rx = rx, .dest = dest });
        }

        fn answerFn(ptr: *anyopaque, m: *const wire.Msg, out: []u64) void {
            const self = of(ptr);
            const side = self.side;
            _ = side.retire();
            switch (m.*) {
                .has => |h| out[0] = @backingInt(self.absence(h.point)),
                .validate => |v| out[0] = @backingInt(self.validate(v.txn)),
                .stats => {
                    side.refresh();
                    wire.words(Counts, &side.cached, out);
                },
                else => @memset(out, 0),
            }
        }

        /// This rank's store compat is the pair's (the agreement at start): presence is the check. A point whose write
        /// this rank still has queued is `incomplete` (published here soon; rank 0 keeps it).
        fn absence(self: *Self, point: Digest) wire.Absence {
            if (self.side.store.presence(point)) return .present;
            if (self.side.writing(point)) return .incomplete;
            return .missing;
        }

        /// Copy the rest (bounded waits that recycle slots), install the sections, report.
        fn validate(self: *Self, txn: u64) wire.Failure {
            const fx = self.restores.getPtr(txn) orelse return .unknown;
            fx.allowed = fx.rx.blocks;
            self.side.rxFinish(fx.rx, monotonicNs() +| follower_copy_ns);
            return fx.rx.why;
        }

        fn settleFn(ptr: *anyopaque, m: *const wire.Msg, all: []const u64) anyerror!void {
            const self = of(ptr);
            const side = self.side;
            switch (m.*) {
                .has => |h| {
                    const v = wire.presence(all);
                    if (!v.ok and v.forget) {
                        side.cancelWrite(h.point);
                        side.store.forget(h.point, &.{}, &.{});
                    }
                },
                .validate => |v| {
                    const verdict = wire.validation(all);
                    const kv = self.restores.fetchRemove(v.txn) orelse return;
                    const rx = kv.value.rx;
                    const dest = kv.value.dest;
                    if (!verdict.ok) {
                        side.rxEnd(rx, .abort, verdict.forget);
                        self.lanes.unreserve(dest);
                        return;
                    }
                    var done_ = side.rxCommit(rx) catch |e| {
                        side.rxEnd(rx, .abort, false);
                        self.lanes.unreserve(dest);
                        return e;
                    };
                    done_.lineage.deinit(self.gpa);
                    const at = rx.at;
                    const snap = side.rxTake(rx);
                    self.lanes.adoptReserved(dest, snap, at, &.{}) catch |e| {
                        side.e.freeSnapshot(snap);
                        self.lanes.unreserve(dest);
                        return e;
                    };
                },
                .stats => {},
                else => return error.NoteAsAsk,
            }
        }

        fn releaseFn(ptr: *anyopaque, kept: *anyopaque) void {
            const self = of(ptr);
            const kp: *L.Kept = @ptrCast(@alignCast(kept));
            self.side.release(kp.snap);
        }
    };
}

// -- core/prompt_cache's Tier over a Leader --------------------------------------------------------------------------

/// The prompt cache's Tier over rank 0's `Lead` (a Leader): `pc` is the engine API's prompt_cache, which native/cuda.zig
/// passes (this module cannot import the engine API). Fork and TierCounts are the prompt cache's field for field.
pub fn TierOf(comptime Lead: type, comptime pc: type) type {
    comptime {
        if (@sizeOf(pc.Fork) != @sizeOf(Fork) or @alignOf(pc.Fork) != @alignOf(Fork) or @offsetOf(pc.Fork, "name") != @offsetOf(Fork, "name") or @offsetOf(pc.Fork, "at") != @offsetOf(Fork, "at"))
            @compileError("prompt_cache.Fork and the tier's Fork differ");
        for (@typeInfo(pc.TierCounts).@"struct".field_names) |n| if (!@hasField(Counts, n)) @compileError("the tier's Counts lack prompt_cache.TierCounts." ++ n);
        for (@typeInfo(Counts).@"struct".field_names) |n| if (!@hasField(pc.TierCounts, n)) @compileError("prompt_cache.TierCounts lacks the tier's " ++ n);
    }
    return struct {
        pub const vtable: pc.Tier.VTable = .{
            .lookup = lookupFn,
            .begin = beginFn,
            .full = fullFn,
            .check = checkFn,
            .poll = pollFn,
            .cancel = cancelFn,
            .keep = keepFn,
            .release = releaseFn,
            .share = shareFn,
            .touch = touchFn,
            .forget = forgetFn,
            .pump = pumpFn,
            .busy = busyFn,
            .flush = flushFn,
            .counts = countsFn,
        };

        pub fn tier(l: *Lead) pc.Tier {
            return .{ .ptr = l, .vtable = &vtable };
        }

        fn of(ptr: *anyopaque) *Lead {
            return @ptrCast(@alignCast(ptr));
        }

        fn lookupFn(ptr: *anyopaque, want: *const pc.Want) ?pc.Point {
            const p = of(ptr).lookup(want.key, want.beyond, want.max_at, want.starts, want.lookahead) orelse return null;
            return .{ .id = p.id, .at = p.at, .tokens = p.tokens, .group = p.group, .shared = p.shared };
        }
        fn beginFn(ptr: *anyopaque, point: *const pc.Point) anyerror!u64 {
            return of(ptr).begin(point.id, point.at, point.shared);
        }
        fn fullFn(ptr: *anyopaque) bool {
            return of(ptr).full();
        }
        fn checkFn(ptr: *anyopaque, point: *const pc.Point) anyerror!void {
            return of(ptr).check(point.id, point.at);
        }
        fn pollFn(ptr: *anyopaque, job: u64, budget_ns: u64) pc.Poll {
            return switch (of(ptr).poll(job, budget_ns)) {
                .pending => |x| .{ .pending = .{ .done = x.done, .total = x.total } },
                .ready => |x| .{ .ready = .{ .saved = x.saved, .lineage = @ptrCast(x.lineage) } },
                .failed => |e| .{ .failed = e },
            };
        }
        fn cancelFn(ptr: *anyopaque, job: u64) void {
            of(ptr).cancel(job);
        }
        fn keepFn(ptr: *anyopaque, w: *const pc.Write) ?u64 {
            return of(ptr).keep(.{ .saved = w.saved, .key = w.key, .at = w.at, .group = w.group, .shared = w.shared, .parent = w.parent, .fork = w.fork, .lineage = @ptrCast(w.lineage) });
        }
        fn releaseFn(ptr: *anyopaque, _: u64, saved: pc.Saved) void {
            of(ptr).release(saved);
        }
        fn shareFn(ptr: *anyopaque, point: u64) void {
            of(ptr).share(point);
        }
        fn touchFn(ptr: *anyopaque, point: u64) void {
            of(ptr).touch(point);
        }
        fn forgetFn(ptr: *anyopaque, point: u64) void {
            of(ptr).forget(point);
        }
        fn pumpFn(ptr: *anyopaque, budget_ns: u64) void {
            of(ptr).pump(budget_ns);
        }
        fn busyFn(ptr: *anyopaque) bool {
            return of(ptr).busy();
        }
        fn flushFn(ptr: *anyopaque, budget_ns: u64) u64 {
            return of(ptr).flush(budget_ns);
        }
        fn countsFn(ptr: *anyopaque, out: []pc.TierCounts) usize {
            var mine: [wire.max_world]Counts = undefined;
            const n = of(ptr).counts(mine[0..@min(out.len, mine.len)]);
            for (mine[0..n], out[0..n]) |c, *o| {
                o.* = .{};
                inline for (@typeInfo(Counts).@"struct".field_names) |f| @field(o, f) = @field(c, f);
            }
            return n;
        }
    };
}

test {
    _ = @import("cuda_spill_tier_test.zig");
}
