//! Port of glm5_next/cuda/multi.py lane operations using Flash Next cuda_lanes.zig contracts.
//! Ash Hart and TensorFold contributors; jschmied CUDA serving path.
//! Generic over P6 Engine: newSeq/freeSeq/bind/prefillWith, verifyMany/sampleWindow/commitWindow,
//! draftMany, prompt snapshots and optional chunked fills. Costs are measured by tune(), not Metal estimates.
const std = @import("std");
const lanes = @import("lanes");
const decode = @import("flashnext").decode;
const shard = @import("cuda_shard.zig");

const be = lanes.backend;

/// A verify window's rows at most (the pending token and 15 drafts; cuda_state.max_rows).
pub const max_rows = 64;
/// Total rows of a captured GLM verify, shared by every stream.
pub const max_verify_rows = 64;
pub const max_drafts = max_rows - 1;
/// The MTP head's chain stops before a later draft under this temperature-1 probability (Python CONFIDENCE).
pub const default_confidence = 0.30;

/// Initial scheduling prior inherited from Flash Next; observations update it.
/// A GLM cost/acceptance calibration still requires a live P6 engine.
pub const draft_prior = [_]f64{ 0.82, 0.76, 0.72, 0.68, 0.65, 0.62, 0.6, 0.58, 0.56, 0.54, 0.52, 0.5, 0.5, 0.5, 0.5 };

/// First tokens a handle names (the prompt's draw is on the host once prefill returns).
const ring = 1024;

/// Two ranks' prepare / agree / commit (work/research/R3): each call's fallible local steps (admission, sequences,
/// commits of the last window, checks) run first, then `agree` hands this rank's outcome to one small gather that
/// precedes the call's first collective; when any rank failed, every rank returns an error before a collective.
/// An engine error after the agreement may leave the ranks apart: `fatal` ends this one (NCCL abort, exit), so
/// neither waits in a collective for the other. One rank has no gate.
pub const Gate = @import("flashnext").lane_backend.Gate;
pub const Notes = @import("flashnext").lane_backend.Notes;

/// `place`: TF_GLM_KV_SHARD's record of where rank 0 placed a stream's extent (cuda_shard.Placement), checked on rank 1;
/// `layout`: TF_GLM_KV_SHARD_CHECK's digest of rank 0's whole pool every that many decode rounds, compared on rank 1.
pub const Note = enum(u8) { pass = 1, drop = 2, spill = 3, load = 4, place = 5, layout = 6, _ };

/// The prompt cache's family calls (core/prompt_cache.zig Snapshots.VTable, same signatures) and its budget:
/// native/cuda.zig makes the Store from them (this module cannot import the engine API's types).
/// The kept-state rules and policy the family hands the store (core/prompt_cache.zig Rules and Policy, as plain fields).
pub const CacheRules = struct {
    grid: u32 = 64,
    min_gap: u32 = 256,
    min_prompt: u32 = 64,
    entries: u32 = 0,
    bytes_cap: u64 = 0,
    per_chat: u32 = 0,
    shared_least: u32 = 0,
};

/// The spill tier's family calls (core/prompt_cache.zig Snapshots.VTable evict / stored / load, same signatures).
pub const SpillHooks = struct {
    evict: *const fn (ptr: *anyopaque, saved: *anyopaque, tokens: []const u32, at: u32) void,
    stored: *const fn (ptr: *anyopaque, prompt: []const u32, longer_than: u32, at: *u32, key: *u64, tokens: *[]const u32) bool,
    load: *const fn (ptr: *anyopaque, key: u64, at: u32) anyerror!*anyopaque,
    highwater: f64 = 0,
    flush_s: f64 = 60,
    stats: *const fn (ptr: *anyopaque, out: *[8]u64) void,
};

pub const CacheHooks = struct {
    spill: ?SpillHooks = null,
    rules: CacheRules = .{},
    ptr: *anyopaque,
    bytes: *const fn (ptr: *anyopaque, at: u32) u64,
    save: *const fn (ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque,
    restore: *const fn (ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void,
    drop: *const fn (ptr: *anyopaque, saved: *anyopaque) void,
    budget: u64,
};

pub fn Lanes(comptime E: type) type {
    // The engine can draw among a grammar's allowed tokens (cuda_engine's sampleWindowMasked and a masked first draw).
    const masked = @hasDecl(E, "sampleWindowMasked");
    return struct {
        const Self = @This();
        /// the engine's kept states hold their rows themselves (no keeper sequence): GLM's snapshot buffer
        const holds_rows = @hasDecl(E, "snap_holds_rows") and E.snap_holds_rows;
        /// the engine's sequence handle, as newSeq returns it
        const SeqPtr = @typeInfo(@typeInfo(@TypeOf(E.newSeq)).@"fn".return_type.?).error_union.payload;

        /// prefillWith's result and its kept state (cuda_engine's Prefilled / Snapshot)
        const Prefilled = @typeInfo(@typeInfo(@TypeOf(E.prefillWith)).@"fn".return_type.?).error_union.payload;
        const SnapPtr = @typeInfo(@FieldType(Prefilled, "kept")).optional.child;
        /// the engine's prompt fill (fillBegin / fillStep / fillFree), when it has one
        const FillPtr = if (@hasDecl(E, "fillBegin")) @typeInfo(@typeInfo(@TypeOf(E.fillBegin)).@"fn".return_type.?).error_union.payload else *anyopaque;

        /// A kept prompt state (the prompt cache's saved state): the engine's snapshot at `at` and a keeper
        /// sequence holding the prompt's first `at` cache rows, so any later stream resumes it (copyPrefix, then
        /// prefillWith from the snapshot). `key` names it to the other rank: rank 0's address of it.
        pub const Kept = struct {
            snap: SnapPtr,
            /// null when the snapshot holds the rows itself (E.snap_holds_rows: GLM's device buffer)
            keeper: ?SeqPtr,
            at: u32,
            bytes: u64,
            key: u64 = 0,
            /// the prompt's first `at` tokens (the spill tier writes them with the state)
            tokens: []u32 = &.{},
        };

        const Tentative = struct { at: u32, kept: *Kept, adopted: bool = false };

        const Lane = struct {
            seq: SeqPtr,
            /// the last verify's rows of this stream, not yet committed (the round loop keeps only on drops, and
            /// a shared round drafts before it keeps)
            pending_rows: ?usize = null,
            /// this stream's window in the round `round` (the engine's sampleWindow / commitWindow index)
            index: usize = 0,
            round: u64 = 0,
            held: [max_drafts]u32 = undefined,
            held_n: usize = 0,
            /// the caches of this stream's whole window, reserved at admission (the guaranteed class)
            reserved: usize = 0,
            /// the prompt filling between rounds (sliced fills), null once it is in
            fill: ?Filling = null,
            /// the first token's handle of a burst's prompt (firstFn gives it once; else the last one taken)
            first: ?u64 = null,
        };

        /// A prompt filling in slices: the engine's fill of the segment in progress (up to the next mark, or to
        /// the end), the state the segment resumed from, and where the pass stands.
        const Filling = struct {
            fl: FillPtr,
            cur: ?SnapPtr = null,
            owned: bool = false,
            at: usize = 0,
            final: bool = false,
        };

        gpa: std.mem.Allocator,
        e: *E,
        drafting: bool,
        context: usize,
        /// a window's rows at most: the engine's depth + 1 (its buffers' rows and the kernel set's windows)
        window: usize,
        /// streams a shared forward holds (--parallel; the engine's Options.streams)
        streams: usize = 1,
        confidence: f64 = default_confidence,
        /// a running product (confidence < 0) applies while at most this many streams are drafting; above it the
        /// per-draft rule at `wide_confidence` (cuda_engine's served hybrid; maxInt: the product always)
        product_streams: usize = std.math.maxInt(usize),
        wide_confidence: f64 = 0.5,
        gate: ?Gate = null,
        /// admission (R3 c): a stream is admitted only when its whole window's caches fit the engine's sequence
        /// budget beside every admitted stream's, so no stream fails to grow mid-reply; `overcommit`
        /// (TF_FLASHNEXT_OVERCOMMIT=1) admits on what is free now instead
        overcommit: bool = false,
        reserved: usize = 0,
        /// kept prompt states: every live one (the store's), the last pass's not yet adopted ones, their bytes
        kept: std.ArrayList(*Kept) = .empty,
        tentative: std.ArrayList(Tentative) = .empty,
        kept_bytes: u64 = 0,
        /// rank 1: rank 0 asked for a state to be written that this rank has not adopted yet (its `.pass` note comes
        /// after the early write's): written when the pass's note arrives
        spill_wait: std.ArrayList(u64) = .empty,
        /// rank 1 at two ranks: the last pass's kept states wait for rank 0's note of which the store took
        follower: bool = false,
        /// rank 0 at two ranks: where the store's decisions go to rank 1 (cuda_mirror's Leader.note)
        notes: ?Notes = null,
        /// prompts fill in slices between rounds (prefill_begin / prefill_step), `fill_layers` layers a step
        sliced: bool = false,
        fill_layers: usize = 8,
        lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, *Lane) = .empty,
        /// the last verify's lanes whose windows wait for their commits: committed before the next forward
        opened: std.ArrayList(*Lane) = .empty,
        round: u64 = 0,
        drawn: [ring]u32 = undefined,
        next: u64 = 0,
        costs: [max_rows]lanes.config.Cost = undefined,
        cost_count: usize = 0,
        shared: [shared_points]lanes.config.Cost = undefined,
        shared_count: usize = 0,
        mtp_ms: f64 = 0,

        pub fn init(gpa: std.mem.Allocator, e: *E, drafting: bool, context: usize, window: usize) Self {
            return .{ .gpa = gpa, .e = e, .drafting = drafting, .context = context, .window = @max(1, @min(window, max_rows)) };
        }

        pub fn deinit(self: *Self) void {
            var it = self.lanes.valueIterator();
            while (it.next()) |l| {
                self.stopFill(l.*);
                self.e.freeSeq(l.*.seq);
                self.gpa.destroy(l.*);
            }
            self.lanes.deinit(self.gpa);
            self.opened.deinit(self.gpa);
            self.dropTentative();
            self.spill_wait.deinit(self.gpa);
            self.tentative.deinit(self.gpa);
            for (self.kept.items) |k| self.freeKept(k);
            self.kept.deinit(self.gpa);
        }

        pub fn backend(self: *Self) be.Backend {
            if (@hasDecl(E, "fillBegin") and self.sliced) return .{ .ptr = self, .vtable = &.{
                .prefill = prefillFn,
                .first = firstFn,
                .queue = queueFn,
                .read = readFn,
                .verify = verifyFn,
                .keep = keepFn,
                .draft = draftFn,
                .tree = heldFn,
                .first_masked = if (masked) firstMaskedFn else null,
                .held = if (masked) heldTokensFn else null,
                .release = releaseFn,
                .prefill_begin = prefillBeginFn,
                .fill_step = prefillStepFn,
            } };
            return .{ .ptr = self, .vtable = &.{
                .prefill = prefillFn,
                .first = firstFn,
                .queue = queueFn,
                .read = readFn,
                .verify = verifyFn,
                .keep = keepFn,
                .draft = draftFn,
                .tree = heldFn,
                .first_masked = if (masked) firstMaskedFn else null,
                .held = if (masked) heldTokensFn else null,
                .release = releaseFn,
                .prefill_many = if (@hasDecl(E, "prefillMany")) prefillManyFn else null,
            } };
        }

        // -- prompts filling between rounds (the lanes core's prefill_begin / prefill_step) --------------------

        /// The prefill's admission and agreements, its resume made, the first segment's fill started.
        fn prefillBeginFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
            const self = of(ptr);
            const e = self.e;
            const l = self.preparePrefill(s) catch |err| {
                try self.agree(err);
                unreachable;
            };
            self.agree(null) catch |err| {
                self.dropLane(s, l);
                return err;
            };
            self.placed(l);
            const requested = s.reuse.at > 0;
            var from: ?*Kept = if (requested) self.keptOf(s.reuse.saved) else null;
            if (requested) {
                if (!(try self.agreeOn(from != null))) from = null;
            }
            var f: Filling = .{ .fl = undefined };
            if (from) |k| {
                if (k.keeper) |kp| e.copyPrefix(l.seq, kp, k.snap.pos, k.snap.mtp_len) catch |err| {
                    self.dropLane(s, l);
                    return self.lost("a prompt's resume", err);
                };
                f.cur = k.snap;
                f.at = k.snap.pos;
            }
            s.cached = @intCast(f.at);
            s.reuse_failed = requested and from == null;
            self.dropTentative();
            self.startSegment(s, l, &f) catch |err| {
                self.dropLane(s, l);
                return self.lost("a prompt's fill", err);
            };
            l.fill = f;
        }

        /// The next segment's fill: up to the next mark (its state kept there), else to the prompt's end.
        fn startSegment(self: *Self, s: *lanes.Stream, l: *Lane, f: *Filling) !void {
            const ids = s.prompt();
            const next_mark: ?u32 = for (s.reuse.marks) |m| {
                if (m > f.at and m < ids.len) break m;
            } else null;
            if (next_mark) |m| {
                f.fl = try self.e.fillBegin(l.seq, ids[0..m], s.sampling, .{ .keep_at = m, .resume_from = f.cur });
                f.final = false;
            } else {
                f.fl = try self.e.fillBegin(l.seq, ids, s.sampling, .{ .resume_from = f.cur });
                f.final = true;
            }
        }

        /// `fill_layers` more layers of every filling prompt; a segment that ends keeps its mark's state and
        /// starts the next; a prompt that is in draws its first token (`first` returns it).
        fn prefillStepFn(ptr: *anyopaque, streams: []const *lanes.Stream, states: []be.FillState) anyerror!void {
            const self = of(ptr);
            try self.agree(null); // nothing to prepare: the ranks only meet before the slices' collectives
            for (streams, states) |s, *st| {
                const l = self.lanes.get(s) orelse return error.UnknownStream;
                const f = &(l.fill orelse return error.NotFilling);
                const done = self.e.fillStep(f.fl, self.fill_layers) catch |err| return self.lost("a prompt's fill", err);
                if (!done) continue;
                if (!f.final) {
                    const m: u32 = @intCast(f.fl.prompt.len);
                    const snap = f.fl.kept orelse return self.lost("a prompt's fill", error.NoStateKept);
                    f.fl.kept = null; // the pass's now
                    self.e.fillFree(f.fl);
                    if (f.owned) self.e.freeSnapshot(f.cur.?);
                    f.cur = snap;
                    f.owned = true;
                    if (self.keep(l, snap, m, s.prompt())) |k| {
                        f.owned = false;
                        self.tentative.append(self.gpa, .{ .at = m, .kept = k }) catch |err| return self.lost("a prompt's fill", err);
                    } else |err| std.log.warn("prompt cache: no state kept at {d} tokens ({s})", .{ m, @errorName(err) });
                    if (s.reuse.hook) |h| h.at(h.ptr, s, m);
                    f.at = m;
                    self.startSegment(s, l, f) catch |err| return self.lost("a prompt's fill", err);
                    continue;
                }
                _ = self.take(f.fl.first);
                self.e.fillFree(f.fl);
                if (f.owned) self.e.freeSnapshot(f.cur.?);
                l.fill = null;
                self.endPass();
                st.* = .done;
            }
        }

        fn stopFill(self: *Self, l: *Lane) void {
            if (comptime !@hasDecl(E, "fillBegin")) return; // an engine without fills never starts one
            const f = l.fill orelse return;
            self.e.fillFree(f.fl);
            if (f.owned) self.e.freeSnapshot(f.cur.?);
            l.fill = null;
        }

        /// The facts the round loop reads at setup: windows up to `window` rows (16 at most), up to `streams`
        /// streams' windows in one forward (each row's bits as alone), this GPU's costs.
        pub fn facts(self: *const Self) lanes.Model {
            const many = self.streams > 1;
            return .{
                .exact_width = if (self.drafting) @intCast(self.window) else 1,
                .gpu_tokens = false,
                .mtp = self.drafting,
                .speculate = self.drafting,
                .speculate_early = false,
                .draft_prior = &draft_prior,
                .drafts = @intCast(self.window - 1),
                .window_costs = self.costs[0..self.cost_count],
                .mtp_step_ms = self.mtp_ms,
                .streams_exact = true,
                .hidden_rows = many,
                .batch_rows = @intCast(@min(max_verify_rows, self.streams * self.window)),
                .max_streams = @intCast(self.streams),
                .shared_costs = self.shared[0..self.shared_count],
                .draft_streams = many,
                // the running-product rule (confidence < 0) stops each chain itself: the round loop asks the most
                .draft_stops = self.drafting and self.confidence < 0,
                .draft_stops_most = @intCast(@min(self.product_streams, std.math.maxInt(u32))),
                // each sequence keeps its prompt's last streams until its first drafts: joiners draft together
                .join_batch = self.drafting,
            };
        }

        /// The lane, its sequence bound and its stream's sampling set.
        fn lane(self: *Self, s: *const lanes.Stream) !*Lane {
            const l = self.lanes.get(s) orelse return error.UnknownStream;
            self.e.bind(l.seq);
            if (@hasDecl(E, "setSampling")) self.e.setSampling(s.sampling);
            return l;
        }

        /// Every window of the last verify still uncommitted, committed whole (the round kept all of its rows,
        /// or keep would have cut it): before the next forward overwrites the round's rows.
        fn settle(self: *Self) !void {
            defer self.opened.clearRetainingCapacity();
            for (self.opened.items) |o| {
                const rows = o.pending_rows orelse continue;
                o.pending_rows = null;
                try self.e.commitWindow(o.index, rows);
            }
        }

        /// Every rank's prepare done: return this rank's error, or error.OtherRankFailed, before any collective.
        fn agree(self: *Self, failed: ?anyerror) !void {
            if (self.gate) |g| {
                const all = g.agree(g.ptr, failed == null) catch |e| {
                    g.fatal(g.ptr, "the ranks' agreement", e);
                    return e;
                };
                if (!all) return failed orelse error.OtherRankFailed;
            }
            if (failed) |e| return e;
        }

        /// An engine call failed after the agreement: with two ranks this one leaves (they may be apart now).
        fn lost(self: *Self, what: []const u8, err: anyerror) anyerror {
            if (self.gate) |g| g.fatal(g.ptr, what, err);
            return err;
        }

        fn take(self: *Self, token: u32) u64 {
            const h = self.next;
            self.drawn[h % ring] = token;
            self.next += 1;
            return h;
        }

        fn value(self: *const Self, feed: be.Feed) u32 {
            return switch (feed) {
                .handle => |h| self.drawn[h % ring],
                .value => |v| v,
            };
        }

        fn of(ptr: *anyopaque) *Self {
            return @ptrCast(@alignCast(ptr));
        }

        // -- the vtable -----------------------------------------------------------------------------------------

        /// A new sequence for the stream, then its prompt (chunks and the head's absorb inside the engine) and the
        /// first draw at the prompt's length, keyed with the stream's sampling.
        fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
            const self = of(ptr);
            const e = self.e;
            const l = self.preparePrefill(s) catch |err| {
                try self.agree(err);
                unreachable;
            };
            self.agree(null) catch |err| {
                self.dropLane(s, l);
                return err;
            };
            self.placed(l);
            // a resume both ranks can make (each holds the state), else both prefill from the start
            const requested = s.reuse.at > 0;
            var from: ?*Kept = if (requested) self.keptOf(s.reuse.saved) else null;
            if (requested) {
                if (!(try self.agreeOn(from != null))) from = null;
            }
            e.bind(l.seq);
            const first = self.pass(s, l, from) catch |err| {
                self.dropLane(s, l);
                return self.lost("a prompt's prefill", err);
            };
            _ = self.take(first);
        }

        /// A burst of prompts admitted together (no resumes, no marks): their admissions and sequences made first
        /// (a refusal anywhere refuses the burst, nothing run: error.BurstRefused, the host prefills them singly), then
        /// the prompts that fit one pass together share it (Engine.prefillMany), the rest prefill alone. Each stream's
        /// first token is kept for its own `first`.
        fn prefillManyFn(ptr: *anyopaque, ss: []const *lanes.Stream) anyerror!void {
            const self = of(ptr);
            if (comptime !@hasDecl(E, "prefillMany")) return error.BurstRefused;
            const e = self.e;
            if (ss.len < 2 or ss.len > max_rows) return error.BurstRefused;
            // a prompt with pictures or clips prefills alone (its rows come by broadcast inside its own prefill)
            for (ss) |s| if (s.reuse.at > 0 or s.reuse.marks.len > 0 or s.media != null) return error.BurstRefused;
            var ls: [max_rows]*Lane = undefined;
            var made: usize = 0;
            var failure: ?anyerror = null;
            for (ss) |s| {
                ls[made] = self.preparePrefill(s) catch |err| {
                    failure = err;
                    break;
                };
                made += 1;
            }
            self.agree(failure) catch {
                for (ss[0..made], ls[0..made]) |s, l| self.dropLane(s, l);
                return error.BurstRefused;
            };
            for (ls[0..made]) |l| self.placed(l);
            for (ss) |s| {
                s.cached = 0;
                s.reuse_failed = false;
            }
            self.dropTentative();
            var i: usize = 0;
            while (i < ss.len) {
                var j = i;
                var rows: usize = 0;
                while (j < ss.len and rows + ss[j].prompt().len <= e.prefill_rows) : (j += 1) rows += ss[j].prompt().len;
                if (j - i >= 2) {
                    var items: [max_rows]E.ManyPrompt = undefined;
                    for (items[0 .. j - i], ss[i..j], ls[i..j]) |*it, s, l| it.* = .{ .seq = l.seq, .prompt = s.prompt(), .sampling = s.sampling };
                    if (e.manyFit(items[0 .. j - i])) {
                        var firsts: [max_rows]u32 = undefined;
                        e.prefillMany(items[0 .. j - i], &firsts) catch |err| return self.burstLost(ss, ls[0..made], err);
                        for (ls[i..j], firsts[0 .. j - i]) |l, f| l.first = self.take(f);
                        i = j;
                        continue;
                    }
                }
                e.bind(ls[i].seq);
                const r = e.prefillWith(ss[i].prompt(), ss[i].sampling, .{}) catch |err| return self.burstLost(ss, ls[0..made], err);
                ls[i].first = self.take(r.first);
                i += 1;
            }
            self.endPass();
        }

        fn burstLost(self: *Self, ss: []const *lanes.Stream, ls: []*Lane, err: anyerror) anyerror {
            for (ss[0..ls.len], ls) |s, l| self.dropLane(s, l);
            return self.lost("a burst's prefill", err);
        }

        /// The prompt pass: from a kept state (its rows copied in) or from 0, cut at each mark, where the state
        /// is kept (a tentative Kept the prompt cache adopts through the stream's hook), then to the end.
        fn pass(self: *Self, s: *lanes.Stream, l: *Lane, from: ?*Kept) !u32 {
            const e = self.e;
            const ids = s.prompt();
            var cur: ?SnapPtr = null; // the state the next segment resumes
            var owned = false; // `cur` is this pass's own (no Kept holds it)
            defer if (owned) e.freeSnapshot(cur.?);
            var live: ?usize = null; // a refused save left the sequence here with live rows: the next segment goes on from them
            var at: usize = 0;
            if (from) |k| {
                if (k.keeper) |kp| try e.copyPrefix(l.seq, kp, k.snap.pos, k.snap.mtp_len);
                cur = k.snap;
                at = k.snap.pos;
            }
            s.cached = @intCast(at);
            s.reuse_failed = s.reuse.at > 0 and from == null;
            self.dropTentative();
            for (s.reuse.marks) |m| {
                if (m <= at or m >= ids.len) continue;
                // a keep at m must leave every rank's keep floor (the memory reserve) alone: the ranks
                // agree on that before the pass keeps, so a rank that cannot leaves all without it (a state one rank
                // holds alone is no use: a resume needs it on all), instead of an engine error after the agreement
                const room = if (comptime @hasDecl(E, "keepFits")) e.keepFits(self.stateBytes(m)) else true;
                if (!(try self.agreeOn(room))) {
                    std.log.warn("prompt cache: no state kept at {d} tokens (a rank is at its keep floor)", .{m});
                    continue;
                }
                const r = try e.prefillWith(ids[0..m], s.sampling, .{ .keep_at = m, .resume_from = if (live == null) cur else null, .live_from = live });
                // the save's outcome, agreed too: a rank the engine refused after all holds no state, so none keeps it,
                // and every rank goes on the same way, from the rows it just ran (no restore, no rows again)
                if (!(try self.agreeOn(r.kept != null))) {
                    std.log.warn("prompt cache: no state kept at {d} tokens (a save was refused)", .{m});
                    if (r.kept) |k| e.freeSnapshot(k);
                    if (owned) e.freeSnapshot(cur.?);
                    cur = null;
                    owned = false;
                    live = m;
                    at = m;
                    continue;
                }
                if (owned) e.freeSnapshot(cur.?);
                cur = r.kept.?;
                owned = true;
                live = null;
                if (self.keep(l, cur.?, m, ids)) |k| {
                    owned = false; // the Kept holds it now
                    try self.tentative.append(self.gpa, .{ .at = m, .kept = k });
                } else |err| std.log.warn("prompt cache: no state kept at {d} tokens ({s})", .{ m, @errorName(err) });
                if (s.reuse.hook) |h| h.at(h.ptr, s, m);
                at = m;
            }
            // structured output: the first token is drawn among the grammar's allowed tokens (the bits of rank 0's
            // grammar when a Leader carries them to rank 1, else the stream's own)
            var bits: ?[]u32 = null;
            defer if (bits) |m| self.gpa.free(m);
            const mask: []const u32 = if (comptime masked) try self.firstMask(s, &bits) else &.{};
            const r = try e.prefillWith(ids, s.sampling, .{ .resume_from = if (live == null) cur else null, .live_from = live, .mask = mask, .media = s.media });
            self.endPass();
            return r.first;
        }

        /// The bits of the stream's first token: the driver's (`first_mask`, set for rank 1 and by a Leader), else its grammar's.
        fn firstMask(self: *Self, s: *lanes.Stream, owned: *?[]u32) ![]const u32 {
            if (s.first_mask.len > 0) return s.first_mask;
            const g = s.grammar orelse return &.{};
            const bits = (try g.next(self.gpa)) orelse return &.{};
            owned.* = bits;
            return bits;
        }

        /// The first token again among the allowed ones: the prompt's draw was masked already (same bits), so it is the one.
        fn firstMaskedFn(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
            _ = mask;
            return firstFn(ptr, s, position);
        }

        /// The stream's held drafts as host tokens in window order (the grammar walks them); rows past the head's chain
        /// repeat its last draft, as `prepareVerify` fills them.
        fn heldTokensFn(ptr: *anyopaque, s: *lanes.Stream, out: []u32) anyerror!void {
            const self = of(ptr);
            const l = self.lanes.get(s) orelse return error.UnknownStream;
            for (out, 0..) |*d, j| d.* = if (j < l.held_n) l.held[j] else if (l.held_n > 0) l.held[l.held_n - 1] else s.pending orelse 0;
        }

        /// A tentative Kept of the pass at `at`: a keeper sequence takes the prompt's first `at` rows.
        fn keep(self: *Self, l: *Lane, snap: SnapPtr, at: u32, ids: []const u32) !*Kept {
            const e = self.e;
            var keeper: ?SeqPtr = null;
            if (comptime !holds_rows) {
                const kp = try e.newSeq(at + 1);
                errdefer e.freeSeq(kp);
                try e.copyPrefix(kp, l.seq, at, snap.mtp_len);
                // the keeper now holds `at` rows: say so, or a later copyPrefix from it refuses (PrefixPastSource)
                if (@hasField(E, "f")) {
                    try e.f.setPos(&kp.st, at);
                    try e.f.setMtpLen(&kp.st, snap.mtp_len);
                }
                keeper = kp;
            }
            errdefer if (keeper) |kp| e.freeSeq(kp);
            const k = try self.gpa.create(Kept);
            errdefer self.gpa.destroy(k);
            const tokens = try self.gpa.dupe(u32, ids[0..at]);
            k.* = .{ .snap = snap, .keeper = keeper, .at = at, .bytes = self.stateBytes(at), .tokens = tokens };
            k.key = @intFromPtr(k);
            self.kept_bytes += k.bytes;
            return k;
        }

        fn freeKept(self: *Self, k: *Kept) void {
            self.kept_bytes -= k.bytes;
            if (k.keeper) |kp| self.e.freeSeq(kp);
            self.e.freeSnapshot(k.snap);
            self.gpa.free(k.tokens);
            self.gpa.destroy(k);
        }

        /// The pass ended: rank 0 (or one rank) frees what the store did not adopt and tells rank 1 which it did;
        /// rank 1 keeps them all until that note.
        fn endPass(self: *Self) void {
            if (self.follower) return;
            if (self.notes) |n| if (self.tentative.items.len > 0) {
                var buf: [1 + 4 + 64 * 12]u8 = undefined;
                var len: usize = 5;
                buf[0] = @backingInt(Note.pass);
                var count: u32 = 0;
                for (self.tentative.items) |t| if (t.adopted and len + 12 <= buf.len) {
                    std.mem.writeInt(u32, buf[len..][0..4], t.at, .little);
                    std.mem.writeInt(u64, buf[len + 4 ..][0..8], t.kept.key, .little);
                    len += 12;
                    count += 1;
                };
                std.mem.writeInt(u32, buf[1..5], count, .little);
                n.send(n.ptr, buf[0..len]);
            };
            self.dropTentative();
        }

        /// The last pass's states the store did not adopt.
        fn dropTentative(self: *Self) void {
            for (self.tentative.items) |t| if (!t.adopted) self.freeKept(t.kept);
            self.tentative.clearRetainingCapacity();
        }

        fn keptOf(self: *Self, saved: ?*anyopaque) ?*Kept {
            const p = saved orelse return null;
            for (self.kept.items) |k| if (@as(*anyopaque, @ptrCast(k)) == p) return k;
            return null;
        }

        /// TF_GLM_KV_SHARD, rank 0: where the lane's new extent lies, to rank 1 with the next calls (`.place`). Rank 1
        /// placed it from the same calls itself; both ranks hold it once they agreed, until the stream's release.
        fn placed(self: *Self, l: *const Lane) void {
            if (comptime !@hasDecl(E, "placement")) return;
            const n = self.notes orelse return;
            const p = self.e.placement(l.seq) orelse return;
            var buf: [1 + shard.Placement.bytes]u8 = undefined;
            buf[0] = @backingInt(Note.place);
            p.encode(buf[1..]);
            n.send(n.ptr, &buf);
        }

        /// TF_GLM_KV_SHARD_CHECK, rank 0, at a decode round's end: every that many rounds its pool's layout digest to
        /// rank 1 (`.layout`), which compares its own after the same round. A drift between placements (a release, a
        /// twin) is caught within those rounds, not only at the next placement.
        fn layoutNote(self: *Self) void {
            if (comptime !@hasDecl(E, "roundDigest")) return;
            const n = self.notes orelse return;
            const d = self.e.roundDigest(self.round) orelse return;
            var buf: [9]u8 = undefined;
            buf[0] = @backingInt(Note.layout);
            std.mem.writeInt(u64, buf[1..9], d, .little);
            n.send(n.ptr, &buf);
        }

        /// Both ranks' answer to a yes/no both must share (one rank: its own).
        fn agreeOn(self: *Self, ok: bool) !bool {
            const g = self.gate orelse return ok;
            return g.agree(g.ptr, ok) catch |e| {
                g.fatal(g.ptr, "the ranks' agreement", e);
                return e;
            };
        }

        // -- the prompt cache's family (core/prompt_cache.zig Snapshots; native/cuda.zig makes the Store) ------

        pub fn cacheHooks(self: *Self, cache_budget: u64, rules: CacheRules, highwater: f64, flush_s: f64) CacheHooks {
            const spills = @hasDecl(E, "spillOut") and self.e.spillOn();
            return .{ .spill = if (spills) .{ .evict = evictFn, .stored = storedFn, .load = loadFn, .highwater = highwater, .flush_s = flush_s, .stats = spillStatsFn } else null, .rules = rules, .ptr = self, .bytes = bytesFn, .save = saveFn, .restore = restoreFn, .drop = dropFn, .budget = cache_budget };
        }

        fn bytesFn(ptr: *anyopaque, at: u32) u64 {
            return of(ptr).stateBytes(at);
        }

        /// What a kept state of `at` tokens holds: its own device buffer, or a keeper sequence's caches.
        fn stateBytes(self: *const Self, at: usize) u64 {
            return if (comptime holds_rows) self.e.snapBytes(at) else self.footprint(at);
        }

        /// The store keeps the pass's state at `at`: the tentative Kept made there.
        fn saveFn(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
            _ = owner;
            const self = of(ptr);
            for (self.tentative.items) |*t| if (t.at == at and !t.adopted) {
                t.adopted = true;
                try self.kept.append(self.gpa, t.kept);
                return t.kept;
            };
            return error.NoStateKept;
        }

        /// The backend restores inside its pass (LaneHost's lookup, not begin).
        fn restoreFn(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
            return error.RestoredInThePass;
        }

        // -- the spill tier (E.spillOut / spillBest / spillIn; core/prompt_cache.zig evict / stored / load) -------

        /// The store lets a state go to disk (rank 0 decided): this rank writes it, and rank 1 is told to write its copy.
        fn evictFn(ptr: *anyopaque, saved: *anyopaque, tokens: []const u32, at: u32) void {
            const self = of(ptr);
            if (comptime !@hasDecl(E, "spillOut")) return;
            self.e.current();
            const k = self.keptOf(saved) orelse return;
            self.e.spillOut(k.snap, tokens, at);
            if (self.notes) |n| {
                var buf: [9]u8 = undefined;
                buf[0] = @backingInt(Note.spill);
                std.mem.writeInt(u64, buf[1..9], k.key, .little);
                n.send(n.ptr, &buf);
            }
        }

        fn storedFn(ptr: *anyopaque, prompt: []const u32, longer_than: u32, at: *u32, key: *u64, tokens: *[]const u32) bool {
            const self = of(ptr);
            if (comptime !@hasDecl(E, "spillOut")) return false;
            return self.e.spillBest(prompt, longer_than, at, key, tokens);
        }

        /// A stored state read into a kept one (rank 0); rank 1 is told to read its file into the same.
        fn loadFn(ptr: *anyopaque, tier_key: u64, at: u32) anyerror!*anyopaque {
            const self = of(ptr);
            if (comptime !@hasDecl(E, "spillOut")) return error.NoSpillTier;
            self.e.current();
            const k = try self.adoptStored(tier_key, at);
            if (self.notes) |n| {
                var buf: [21]u8 = undefined;
                buf[0] = @backingInt(Note.load);
                std.mem.writeInt(u64, buf[1..9], tier_key, .little);
                std.mem.writeInt(u64, buf[9..17], k.key, .little);
                std.mem.writeInt(u32, buf[17..21], at, .little);
                n.send(n.ptr, &buf);
            }
            return k;
        }

        fn spillStatsFn(ptr: *anyopaque, out: *[8]u64) void {
            if (comptime !@hasDecl(E, "spillOut")) return;
            of(ptr).e.spillStats(out);
        }

        fn adoptStored(self: *Self, tier_key: u64, at: u32) !*Kept {
            const got = try self.e.spillIn(tier_key, at);
            errdefer self.e.freeSnapshot(got.snap);
            const k = try self.gpa.create(Kept);
            errdefer self.gpa.destroy(k);
            errdefer self.gpa.free(got.tokens);
            k.* = .{ .snap = got.snap, .keeper = null, .at = at, .bytes = self.stateBytes(at), .tokens = got.tokens };
            k.key = @intFromPtr(k);
            try self.kept.append(self.gpa, k);
            self.kept_bytes += k.bytes;
            return k;
        }

        fn dropFn(ptr: *anyopaque, saved: *anyopaque) void {
            const self = of(ptr);
            if (comptime @hasDecl(E, "current")) self.e.current();
            const k = self.keptOf(saved) orelse return;
            if (self.notes) |n| {
                var buf: [9]u8 = undefined;
                buf[0] = @backingInt(Note.drop);
                std.mem.writeInt(u64, buf[1..9], k.key, .little);
                n.send(n.ptr, &buf);
            }
            self.forgetKept(k);
        }

        fn forgetKept(self: *Self, k: *Kept) void {
            if (std.mem.indexOfScalar(*Kept, self.kept.items, k)) |i| _ = self.kept.swapRemove(i);
            // the store may drop a state the pass kept a moment ago (a cap): the pass's list must not name it any more
            var i: usize = 0;
            while (i < self.tentative.items.len) {
                if (self.tentative.items[i].kept == k) _ = self.tentative.swapRemove(i) else i += 1;
            }
            self.freeKept(k);
        }

        // -- rank 1: rank 0's prompt cache decisions ----------------------------------------------------------

        /// A note of rank 0's store (cuda_mirror carries it): the last pass's adopted states, or a dropped one.
        pub fn applyNote(self: *Self, bytes: []const u8) !void {
            if (bytes.len == 0) return error.BadNote;
            switch (@as(Note, @fromBackingInt(bytes[0]))) {
                .pass => {
                    if (bytes.len < 5) return error.BadNote;
                    const n = std.mem.readInt(u32, bytes[1..5], .little);
                    if (bytes.len != 5 + 12 * @as(usize, n)) return error.BadNote;
                    for (0..n) |i| {
                        const at = std.mem.readInt(u32, bytes[5 + 12 * i ..][0..4], .little);
                        const key = std.mem.readInt(u64, bytes[9 + 12 * i ..][0..8], .little);
                        for (self.tentative.items) |*t| if (t.at == at and !t.adopted) {
                            t.adopted = true;
                            t.kept.key = key;
                            try self.kept.append(self.gpa, t.kept);
                            if (comptime @hasDecl(E, "spillOut")) if (std.mem.indexOfScalar(u64, self.spill_wait.items, key)) |w| {
                                _ = self.spill_wait.swapRemove(w);
                                self.e.spillOut(t.kept.snap, t.kept.tokens, t.kept.at);
                            };
                            break;
                        };
                    }
                    self.dropTentative();
                },
                .drop => {
                    if (bytes.len != 9) return error.BadNote;
                    const key = std.mem.readInt(u64, bytes[1..9], .little);
                    for (self.kept.items) |k| if (k.key == key) return self.forgetKept(k);
                },
                .spill => {
                    if (bytes.len != 9) return error.BadNote;
                    if (comptime !@hasDecl(E, "spillOut")) return;
                    const key = std.mem.readInt(u64, bytes[1..9], .little);
                    for (self.kept.items) |k| if (k.key == key) return self.e.spillOut(k.snap, k.tokens, k.at);
                    // not adopted yet: the pass that kept it ends with a `.pass` note after this one
                    if (self.spill_wait.items.len >= 64) _ = self.spill_wait.orderedRemove(0);
                    try self.spill_wait.append(self.gpa, key);
                },
                .place => {
                    if (comptime !@hasDecl(E, "checkPlacement")) return error.BadNote;
                    try self.e.checkPlacement(try shard.Placement.decode(bytes[1..]));
                },
                .layout => {
                    if (comptime !@hasDecl(E, "checkDigest")) return error.BadNote;
                    if (bytes.len != 9) return error.BadNote;
                    try self.e.checkDigest(std.mem.readInt(u64, bytes[1..9], .little));
                },
                .load => {
                    if (bytes.len != 21) return error.BadNote;
                    if (comptime !@hasDecl(E, "spillOut")) return;
                    const tier_key = std.mem.readInt(u64, bytes[1..9], .little);
                    const key = std.mem.readInt(u64, bytes[9..17], .little);
                    const at = std.mem.readInt(u32, bytes[17..21], .little);
                    // a rank that cannot read its copy holds no such state: the resume then runs from 0 on both ranks
                    const k = self.adoptStored(tier_key, at) catch |err| return std.log.warn("spill: rank 1 could not read {x} ({s})", .{ tier_key, @errorName(err) });
                    k.key = key;
                },
                _ => return error.BadNote,
            }
        }

        /// Rank 1's own state for rank 0's key (null: it holds none; the resume then runs from 0 on both).
        pub fn savedFor(self: *Self, key: u64) ?*anyopaque {
            for (self.kept.items) |k| if (k.key == key) return k;
            return null;
        }

        /// The prefill's local steps: admission, the last round's commits, the stream's lane and a new sequence.
        fn preparePrefill(self: *Self, s: *lanes.Stream) !*Lane {
            const e = self.e;
            const ids = s.prompt();
            const limit = ids.len + s.max_new + self.window;
            if (ids.len == 0 or ids.len + s.max_new > self.context) return error.PromptTooLong;
            const need = self.footprint(limit);
            const before = if (self.lanes.get(s)) |old| old.reserved else 0;
            if (!self.overcommit and self.reserved - before + need + self.kept_bytes > self.budget()) return error.NoRoom;
            try self.settle();
            // a prompt with pictures or clips: its checks and device buffers now, before the ranks agree
            if (comptime @hasDecl(E, "reserveMedia")) if (s.media) |m| try e.reserveMedia(m, ids.len);
            errdefer if (comptime @hasDecl(E, "detachMedia")) e.detachMedia();
            const l = if (self.lanes.get(s)) |old| blk: {
                e.freeSeq(old.seq);
                break :blk old;
            } else blk: {
                const l = try self.gpa.create(Lane);
                errdefer self.gpa.destroy(l);
                try self.lanes.put(self.gpa, s, l);
                break :blk l;
            };
            self.reserved -= before;
            l.* = .{ .seq = e.newSeq(limit) catch |err| {
                _ = self.lanes.remove(s);
                self.gpa.destroy(l);
                return err;
            } };
            l.reserved = need;
            self.reserved += need;
            return l;
        }

        /// A stream's caches at its whole window (the engine's seqBytes; 0 for an engine that does not say).
        fn footprint(self: *const Self, rows: usize) usize {
            return if (@hasDecl(E, "seqBytes")) self.e.seqBytes(rows) else 0;
        }

        /// The bytes the sequences' caches may take on this rank (the engine's budget; unbounded without one).
        pub fn budget(self: *const Self) usize {
            return if (@hasField(E, "budget")) self.e.budget.limit else std.math.maxInt(usize);
        }

        fn dropLane(self: *Self, s: *const lanes.Stream, l: *Lane) void {
            _ = self.lanes.remove(s);
            self.stopFill(l);
            self.reserved -= l.reserved;
            self.forget(l);
            if (comptime @hasDecl(E, "detachMedia")) self.e.detachMedia();
            self.e.freeSeq(l.seq);
            self.gpa.destroy(l);
        }

        fn forget(self: *Self, l: *Lane) void {
            for (self.opened.items, 0..) |o, i| if (o == l) {
                _ = self.opened.swapRemove(i);
                return;
            };
        }

        fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
            const self = of(ptr);
            if (position != s.prompt_len) return error.PositionMismatch;
            if (self.lanes.get(s)) |l| if (l.first) |h| {
                l.first = null;
                return h;
            };
            if (self.next == 0) return error.NoFirstToken;
            return self.next - 1;
        }

        fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
            _ = .{ ptr, s, feed, position };
            return error.NotPipelined;
        }

        fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
            const self = of(ptr);
            if (handle >= self.next or self.next - handle > ring) return error.NoSuchToken;
            return self.drawn[handle % ring];
        }

        /// Every stream's window in one forward: its pending token, then its held drafts or the host's, each row
        /// drawn at its position with its stream's sampling. Held drafts past the head's chain (the prompt's first
        /// chain stopped early on confidence) repeat its last draft: a row is kept only when it equals the
        /// target's draw, so they cost rows, not bits.
        fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
            const self = of(ptr);
            var ids: [max_streams][max_rows]u32 = undefined;
            var wins: [max_streams]E.Window = undefined;
            var ls: [max_streams]*Lane = undefined;
            self.prepareVerify(windows, &ids, &wins, &ls) catch |err| {
                try self.agree(err);
                unreachable;
            };
            try self.agree(null);
            const e = self.e;
            const n = windows.len;
            e.verifyMany(wins[0..n]) catch |err| return self.lost("a round's forward", err);
            self.round += 1;
            if (comptime @hasDecl(E, "sampleWindows")) {
                // every window's draws in one sampler call: one gather and one readback for the round
                var samplings: [max_streams]?lanes.Sampling = undefined;
                var outs: [max_streams][]u32 = undefined;
                var masks: [max_streams][]const u32 = undefined;
                for (windows, out, 0..) |w, o, i| {
                    samplings[i] = w.stream.sampling;
                    outs[i] = o.sampled[0..w.rows()];
                    masks[i] = if (comptime masked) w.masks else &.{};
                }
                e.sampleWindows(samplings[0..n], outs[0..n], masks[0..n]) catch |err| return self.lost("the round's draws", err);
            }
            for (windows, out, ls[0..n], 0..) |w, o, l, i| {
                const rows = w.rows();
                if (comptime !@hasDecl(E, "sampleWindows")) (if (comptime masked) e.sampleWindowMasked(i, w.stream.sampling, o.sampled[0..rows], w.masks) else e.sampleWindow(i, w.stream.sampling, o.sampled[0..rows])) catch |err| return self.lost("a window's draws", err);
                @memcpy(o.drafts[0 .. rows - 1], ids[i][1..rows]);
                l.held_n = 0;
                l.pending_rows = rows;
                l.index = i;
                l.round = self.round;
                self.opened.append(self.gpa, l) catch |err| return self.lost("a window's bookkeeping", err);
            }
            self.layoutNote();
        }

        /// The windows' checks and rows, the last round committed.
        fn prepareVerify(self: *Self, windows: []const be.Window, ids: *[max_streams][max_rows]u32, wins: *[max_streams]E.Window, ls: *[max_streams]*Lane) !void {
            if (windows.len == 0 or windows.len > self.streams) return error.TooManyStreams;
            try self.opened.ensureTotalCapacity(self.gpa, windows.len);
            try self.settle();
            for (windows, 0..) |w, i| {
                if (w.parents != null) return error.TreesNotBuilt;
                const rows = w.rows();
                if (rows > self.window) return error.WindowTooWide;
                const l = try self.lane(w.stream);
                for (windows[0..i]) |v| if (v.stream == w.stream) return error.StreamTwiceInRound;
                for (w.positions, 0..) |p, r| if (p != self.e.pos() + 1 + r) return error.PositionMismatch;
                ids[i][0] = w.pending;
                for (ids[i][1..][0..w.held], 0..) |*d, j| d.* = if (j < l.held_n) l.held[j] else if (l.held_n > 0) l.held[l.held_n - 1] else w.pending;
                @memcpy(ids[i][1 + w.held ..][0..w.tokens.len], w.tokens);
                wins[i] = .{ .seq = l.seq, .tokens = ids[i][0..rows] };
                ls[i] = l;
            }
        }

        /// Each window keeps its path's rows (a window its draft already committed is left as it is).
        fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
            const self = of(ptr);
            if (windows.len != paths.len) return error.PathsWindowsMismatch;
            // every window checked before any is touched: a refused batch leaves them all pending
            var ls: [max_streams]*Lane = undefined;
            var idx: [max_streams]usize = undefined;
            var keeps: [max_streams]usize = undefined;
            var n: usize = 0;
            for (windows, paths) |w, path| {
                if (path.len == 0) return error.EmptyPath;
                for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
                const l = self.lanes.get(w.stream) orelse return error.UnknownStream;
                const rows = l.pending_rows orelse continue;
                if (path.len > rows) return error.PathTooLong;
                if (n == max_streams) return error.TooManyStreams;
                for (ls[0..n]) |seen| if (seen == l) return error.StreamTwiceInRound;
                ls[n] = l;
                idx[n] = l.index;
                keeps[n] = path.len;
                n += 1;
            }
            for (ls[0..n]) |l| l.pending_rows = null;
            // a commit runs no collective, but a failed one leaves this rank's caches apart from the other's
            try self.commitAll(idx[0..n], keeps[0..n]);
        }

        /// The windows' commits: at once when the engine can (one commit table, one replay, one conv shift, one tap
        /// pass for the round), else one by one.
        fn commitAll(self: *Self, idx: []const usize, keeps: []const usize) !void {
            if (idx.len == 0) return;
            if (comptime @hasDecl(E, "commitWindows")) {
                self.e.commitWindows(idx, keeps) catch |err| return self.lost("the windows' commit", err);
            } else for (idx, keeps) |i, k| self.e.commitWindow(i, k) catch |err| return self.lost("a window's commit", err);
        }

        /// Every request's head absorbs its kept rows with the token after each (after a prompt: its last row and
        /// first token), then chains up to `depth` drafts, all streams in shared head steps (the engine's
        /// draftMany: cuda_decode.draft's rule a stream); `tree` hands the round loop each one's count.
        fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
            const self = of(ptr);
            self.prepareDraft(requests) catch |err| {
                try self.agree(err);
                unreachable;
            };
            try self.agree(null);
            const e = self.e;
            var reqs: [max_streams]E.DraftReq = undefined;
            var firsts: [max_streams][1]u32 = undefined;
            var ls: [max_streams]*Lane = undefined;
            const conf = self.confidenceNow();
            // a shared round drafts before it keeps: the kept rows are these requests', commit them first (at once)
            var idx: [max_streams]usize = undefined;
            var keeps: [max_streams]usize = undefined;
            var n: usize = 0;
            for (requests) |r| {
                const l = self.lanes.get(r.stream).?;
                if (r.rows) |rows| if (l.pending_rows) |_| {
                    l.pending_rows = null;
                    idx[n] = l.index;
                    keeps[n] = rows.len;
                    n += 1;
                };
            }
            try self.commitAll(idx[0..n], keeps[0..n]);
            for (requests, 0..) |r, i| {
                const l = self.lanes.get(r.stream).?;
                ls[i] = l;
                l.held_n = 0;
                const depth = @min(r.depth, self.window - 1);
                if (r.rows == null) firsts[i] = .{self.value(r.first.?)};
                reqs[i] = .{
                    .seq = l.seq,
                    .source = if (r.rows != null) .{ .window = l.index } else .prefill_last,
                    .next = if (r.rows != null) r.follow else &firsts[i],
                    .count = depth,
                    .sampling = r.stream.sampling,
                    .confidence = conf,
                    .out = &l.held,
                };
            }
            e.draftMany(reqs[0..requests.len]) catch |err| return self.lost("the head's drafts", err);
            for (reqs[0..requests.len], ls[0..requests.len]) |q, l| l.held_n = q.drafted;
        }

        /// The stop rule for this step's drafts: the running product while few streams draft (the lanes not filling a
        /// prompt; the round loop's live count), the per-draft rule above (both ranks hold the same lanes).
        fn confidenceNow(self: *const Self) f64 {
            if (self.confidence >= 0) return self.confidence;
            var live: usize = 0;
            var it = self.lanes.valueIterator();
            while (it.next()) |l| live += @intFromBool(l.*.fill == null);
            return if (live <= self.product_streams) self.confidence else self.wide_confidence;
        }

        /// The drafts' checks: each request's window is the last round's, its draft lands after its kept rows.
        fn prepareDraft(self: *Self, requests: []const be.DraftRequest) !void {
            if (!self.drafting) return error.NoDraftHead;
            if (requests.len > self.streams) return error.TooManyStreams;
            for (requests) |r| {
                if (r.lanes != null or r.early) return error.TreesNotBuilt;
                const l = try self.lane(r.stream);
                var after = self.e.pos();
                if (r.rows) |rows| {
                    for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                    if (r.follow.len != rows.len or rows.len == 0) return error.FollowMismatch;
                    if (l.round != self.round) return error.NoWindowThisRound;
                    if (l.pending_rows) |pending| {
                        if (rows.len > pending) return error.PathTooLong;
                        after += rows.len;
                    }
                } else if (r.first == null) return error.NoFirstToken;
                if (r.position != after + 1) return error.PositionMismatch;
            }
        }

        /// The drafts held for the stream's next window: the chain's length (it may stop short of the depth asked).
        fn heldFn(ptr: *anyopaque, s: *lanes.Stream, gpa: std.mem.Allocator) anyerror!?lanes.stream.Held {
            _ = gpa;
            const self = of(ptr);
            const l = self.lanes.get(s) orelse return null;
            return .{ .count = @intCast(l.held_n) };
        }

        fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
            const self = of(ptr);
            const kv = self.lanes.fetchRemove(s) orelse return;
            if (kv.value.fill != null) {
                self.stopFill(kv.value); // a cancelled fill (the host discards it between steps, on both ranks)
                self.dropTentative();
            }
            self.reserved -= kv.value.reserved;
            self.forget(kv.value);
            self.e.freeSeq(kv.value.seq);
            self.gpa.destroy(kv.value);
        }

        // -- costs ----------------------------------------------------------------------------------------------

        /// What a verify window of 1..`window` rows and one head level cost on this engine, in ms (median of 5
        /// after a warm one), and with several streams a whole shared round of 2, 4, ... `streams` full windows,
        /// the batched head's drafts included (the lanes core's shared-costs table, to streams x window rows; C3):
        /// each timed on sequences of their own over
        /// `ids` (real text, at least 48 tokens). Every call is the same on both ranks, so a two-rank engine's
        /// collectives stay in step while each rank times itself.
        pub fn measure(self: *Self, io: std.Io, ids: []const u32) !void {
            if (!self.drafting) return;
            const reps = 5;
            const e = self.e;
            if (ids.len < 3 * max_rows) return error.CostTextTooShort;
            try self.settle();
            const top = self.window;
            // 19 prompt rows (the head absorbs 18): a multiple of 16 rows asks the kernel set for a variant
            // (Triton's M % 16 specialization) a capture may not hold
            const lead = max_rows + 3;
            const cont = ids[lead..];
            var c = try Costing(E).init(self.gpa, e, @min(self.context, 4096), 1, ids[0..lead]);
            defer c.deinit();
            for (0..32) |_| try c.round(1, cont, null, 0); // the GPU at its working clocks before any window is timed
            var verify_all: [max_rows + 1]f64 = @splat(0);
            const verify = verify_all[0 .. top + 1];
            var times: [reps + 1]f64 = undefined;
            for (1..top + 1) |w| {
                for (&times) |*x| try c.round(w, cont, .{ .io = io, .out = x }, 0);
                verify[w] = median(times[1..]);
            }
            var chain: [2]f64 = undefined;
            const deep = @max(2, @min(8, top - 1));
            for ([_]usize{ 1, deep }, &chain) |levels, *out| {
                for (&times) |*x| x.* = try c.chain(io, levels);
                out.* = median(times[1..]);
            }
            const k = @import("core").draft_depth.Costs.measured(verify, (chain[1] - chain[0]) / @as(f64, @floatFromInt(deep - 1)));
            self.cost_count = 0;
            for (1..k.rows + 1) |w| {
                self.costs[self.cost_count] = .{ .width = @intCast(w), .ms = k.verify[w] };
                self.cost_count += 1;
            }
            self.mtp_ms = k.level;
            // shared rounds: n streams' full windows in one forward
            self.shared_count = 0;
            var n: usize = 2;
            while (n <= self.streams and self.shared_count < shared_points) : (n = if (n * 2 > self.streams and n < self.streams) self.streams else n * 2) {
                var many = try Costing(E).init(self.gpa, e, @min(self.context, 2048), n, ids[0..lead]);
                defer many.deinit();
                try many.round(1, cont, null, 0);
                // a whole shared round: the forward, the draws and the batched head drafting every stream's next
                // window (C3: the allocator weighs rows against what the round costs, the head included)
                for (&times) |*x| try many.round(top, cont, .{ .io = io, .out = x }, top - 1);
                self.shared[self.shared_count] = .{ .width = @intCast(n * top), .ms = median(times[1..]) };
                self.shared_count += 1;
            }
        }
    };
}

/// Shared rounds' cost points at most (2, 4, ... 64 streams).
const shared_points = 8;
/// Streams one shared forward holds at most (the engine's round segments).
pub const max_streams = 8;

/// The cost measurement's sequences and rounds: `n` streams each prefilled with `lead` and its head in step.
fn Costing(comptime E: type) type {
    return struct {
        const Self = @This();
        const SeqPtr = @typeInfo(@typeInfo(@TypeOf(E.newSeq)).@"fn".return_type.?).error_union.payload;
        e: *E,
        seqs: [max_streams]SeqPtr = undefined,
        pending: [max_streams]u32 = undefined,
        n: usize = 0,

        fn init(gpa: std.mem.Allocator, e: *E, limit: usize, n: usize, lead: []const u32) !Self {
            _ = gpa;
            var c: Self = .{ .e = e };
            errdefer c.deinit();
            for (0..n) |i| {
                c.seqs[i] = try e.newSeq(limit);
                c.n += 1;
                e.bind(c.seqs[i]);
                if (@hasDecl(E, "setSampling")) e.setSampling(null);
                c.pending[i] = try e.prefill(lead, null);
            }
            var reqs: [max_streams]E.DraftReq = undefined;
            var firsts: [max_streams][1]u32 = undefined;
            var none: [1]u32 = undefined;
            for (0..n) |i| {
                firsts[i] = .{c.pending[i]};
                reqs[i] = .{ .seq = c.seqs[i], .source = .prefill_last, .next = &firsts[i], .count = 0, .out = &none };
            }
            try e.draftMany(reqs[0..n]);
            return c;
        }

        fn deinit(c: *Self) void {
            for (c.seqs[0..c.n]) |q| c.e.freeSeq(q);
            c.n = 0;
        }

        const Timing = struct { io: std.Io, out: *f64 };

        /// One round of every stream's `w`-row window (its pending token, then text), its first row kept and
        /// absorbed by the head, which then drafts `levels` (confidence 0: every level runs); `timing` times the
        /// forward and the draws, and the head when it drafts.
        fn round(c: *Self, w: usize, text: []const u32, timing: ?Timing, levels: usize) !void {
            const e = c.e;
            var ids: [max_streams][max_rows]u32 = undefined;
            var wins: [max_streams]E.Window = undefined;
            for (0..c.n) |i| {
                ids[i][0] = c.pending[i];
                @memcpy(ids[i][1..w], text[0 .. w - 1]);
                wins[i] = .{ .seq = c.seqs[i], .tokens = ids[i][0..w] };
            }
            var sampled: [max_streams][max_rows]u32 = undefined;
            const t0 = if (timing) |t| std.Io.Timestamp.now(t.io, .awake) else undefined;
            try e.verifyMany(wins[0..c.n]);
            for (0..c.n) |i| try e.sampleWindow(i, null, sampled[i][0..w]);
            if (timing != null and levels == 0) timing.?.out.* = msSince(timing.?.io, t0);
            var reqs: [max_streams]E.DraftReq = undefined;
            var drafts: [max_streams][max_drafts]u32 = undefined;
            for (0..c.n) |i| {
                try e.commitWindow(i, 1);
                c.pending[i] = sampled[i][0];
                reqs[i] = .{ .seq = c.seqs[i], .source = .{ .window = i }, .next = sampled[i][0..1], .count = levels, .confidence = 0, .out = &drafts[i] };
            }
            try e.draftMany(reqs[0..c.n]);
            if (timing != null and levels > 0) timing.?.out.* = msSince(timing.?.io, t0);
        }

        /// One stream's kept row, then its head chaining `levels` drafts (greedy, confidence 0), timed.
        fn chain(c: *Self, io: std.Io, levels: usize) !f64 {
            const e = c.e;
            var ids = [1]u32{c.pending[0]};
            var sampled: [1]u32 = undefined;
            try e.verifyMany(&.{.{ .seq = c.seqs[0], .tokens = &ids }});
            try e.sampleWindow(0, null, &sampled);
            try e.commitWindow(0, 1);
            c.pending[0] = sampled[0];
            var drafts: [max_drafts]u32 = undefined;
            var reqs = [1]E.DraftReq{.{ .seq = c.seqs[0], .source = .{ .window = 0 }, .next = &sampled, .count = levels, .confidence = 0, .out = &drafts }};
            const t0 = std.Io.Timestamp.now(io, .awake);
            try e.draftMany(&reqs);
            return msSince(io, t0);
        }
    };
}

fn msSince(io: std.Io, t0: std.Io.Timestamp) f64 {
    const d = t0.durationTo(std.Io.Timestamp.now(io, .awake));
    return @as(f64, @floatFromInt(d.nanoseconds)) / 1e6;
}

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return if (xs.len % 2 == 1) xs[xs.len / 2] else (xs[xs.len / 2 - 1] + xs[xs.len / 2]) / 2;
}

test "GLM lane backend interleaves eight streams and preserves serial keyed output" {
    const gpa = std.testing.allocator;
    const Toy = @import("flashnext").lane_backend.Toy;
    var t: Toy = .{ .gpa = gpa };
    var b = Lanes(Toy).init(gpa, &t, true, 4096, 8);
    defer b.deinit();
    b.streams = 8;
    for (b.costs[0..8], 1..) |*c, width| c.* = .{ .width = @intCast(width), .ms = 8 + 0.3 * @as(f64, @floatFromInt(width)) };
    b.cost_count = 8;
    b.mtp_ms = 0.4;
    var cfg = try lanes.Config.init(gpa, b.facts(), 8, 7);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var loop = lanes.Engine.init(gpa, &cfg, b.backend(), clock.clock());
    defer loop.deinit();
    var streams = [_]lanes.Stream{
        try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &.{ 1, 2, 3 }, .max_new = 40, .sampling = .{ .seed = 47, .temperature = 0.7 } }),
        try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &.{ 5, 6 }, .max_new = 40 }),
        try lanes.Stream.init(gpa, .{ .id = "c", .prompt = &.{ 7, 8, 9, 10 }, .max_new = 40 }),
        try lanes.Stream.init(gpa, .{ .id = "d", .prompt = &.{ 11, 12 }, .max_new = 40 }),
        try lanes.Stream.init(gpa, .{ .id = "e", .prompt = &.{ 13, 14, 15 }, .max_new = 40 }),
        try lanes.Stream.init(gpa, .{ .id = "f", .prompt = &.{ 16, 17 }, .max_new = 40 }),
        try lanes.Stream.init(gpa, .{ .id = "g", .prompt = &.{ 18, 19 }, .max_new = 40 }),
        try lanes.Stream.init(gpa, .{ .id = "h", .prompt = &.{ 20, 21, 22 }, .max_new = 40 }),
    };
    defer for (&streams) |*s| s.deinit(gpa);
    for (&streams) |*s| try loop.addStream(s);
    while (loop.activeCount() != 0) try loop.step();
    // the round's draws and commits went through the batched calls, once a round (the per-window ones are the fallback)
    try std.testing.expectEqual(t.rounds, t.batched_draws);
    try std.testing.expect(t.batched_commits >= 1 and t.batched_commits <= t.rounds);
    try std.testing.expectEqual(@as(usize, 0), t.seqs);
    for (&streams) |*s| {
        var serial: Toy = .{ .gpa = gpa };
        const seq = try serial.newSeq(4096);
        defer serial.freeSeq(seq);
        serial.bind(seq);
        const first = try serial.prefill(s.prompt(), s.sampling);
        var result = try decode.serial(Toy, &serial, gpa, first, .{ .count = 40 });
        defer result.deinit(gpa);
        try std.testing.expectEqualSlices(u32, result.tokens.items, s.emitted());
    }
}

// -- kept states that hold their rows (GLM): the prompt pass, the store's calls and rank 1's notes, on a mini engine ------

/// An engine whose sequence is its tokens and whose kept state copies them (GLM's snapshot buffer): the first draw is a
/// hash of every token, so a resumed pass equals a fresh one exactly when the state restored whole. Its spill tier is a
/// map in memory. With a `pool` (TF_GLM_KV_SHARD's two parts) each sequence holds an extent placed as the engine's.
const MiniEngine = struct {
    pub const snap_holds_rows = true;
    pub const Seq = struct { tokens: std.ArrayList(u32) = .empty, eid: ?u64 = null };
    pub const Snap = struct { id: u64, pos: usize, mtp_len: usize = 0, tokens: []u32 };
    pub const PrefillOptions = struct { keep_at: ?usize = null, resume_from: ?*const Snap = null, live_from: ?usize = null, mask: []const u32 = &.{}, media: ?*const lanes.Media = null };
    pub const Prefilled = struct { first: u32, kept: ?*Snap = null };
    pub const Window = struct { seq: *Seq, tokens: []const u32 };
    pub const DraftReq = struct { seq: *Seq };
    const Disk = struct { key: u64, at: u32, tokens: []u32 };

    gpa: std.mem.Allocator,
    bound: ?*Seq = null,
    snaps: usize = 0,
    seqs: usize = 0,
    resumed: usize = 0,
    next_id: u64 = 0,
    disk: std.ArrayList(Disk) = .empty,
    writes: usize = 0,
    spill_on: bool = false,
    keep_fits: bool = true, // what keepFits answers: this rank's room for a kept state
    refuse_keep: bool = false, // prefillWith keeps nothing (a save refused after the agreement)
    refuse_keep_at: ?usize = null, // prefillWith keeps nothing at this mark only
    trace: std.ArrayList([2]usize) = .empty, // every prefillWith's rows: [begin, end)
    pool: ?@import("cuda_pool.zig").Pool = null, // TF_GLM_KV_SHARD's parts (null: off)
    check: u32 = 0, // TF_GLM_KV_SHARD_CHECK

    fn hash(tokens: []const u32) u32 {
        return @truncate(std.hash.Wyhash.hash(3, std.mem.sliceAsBytes(tokens)));
    }
    pub fn seqBytes(_: *const MiniEngine, rows: usize) usize {
        return rows * 10;
    }
    pub fn snapBytes(_: *const MiniEngine, at: usize) usize {
        return 1000 + at * 4;
    }
    pub fn keepFits(e: *const MiniEngine, _: u64) bool {
        return e.keep_fits;
    }
    pub fn newSeq(e: *MiniEngine, rows: usize) !*Seq {
        const s = try e.gpa.create(Seq);
        s.* = .{};
        if (e.pool) |*p| {
            errdefer e.gpa.destroy(s);
            s.eid = (try p.alloc(rows)) orelse return error.NoPoolExtent;
            (try p.get(s.eid.?)).owner = e.seqs;
        }
        e.seqs += 1;
        return s;
    }
    pub fn freeSeq(e: *MiniEngine, s: *Seq) void {
        if (e.pool) |*p| p.remove(s.eid.?) catch unreachable;
        s.tokens.deinit(e.gpa);
        e.gpa.destroy(s);
        e.seqs -= 1;
    }
    pub fn bind(e: *MiniEngine, s: *Seq) void {
        e.bound = s;
    }
    pub fn copyPrefix(_: *MiniEngine, _: *Seq, _: *Seq, _: usize, _: usize) !void {
        return error.NotUsed;
    }
    pub fn freeSnapshot(e: *MiniEngine, s: *Snap) void {
        e.gpa.free(s.tokens);
        e.gpa.destroy(s);
        e.snaps -= 1;
    }
    fn snapOf(e: *MiniEngine, tokens: []const u32) !*Snap {
        const s = try e.gpa.create(Snap);
        e.next_id += 1;
        s.* = .{ .id = e.next_id, .pos = tokens.len, .tokens = try e.gpa.dupe(u32, tokens) };
        e.snaps += 1;
        return s;
    }
    pub fn prefillWith(e: *MiniEngine, prompt: []const u32, _: ?lanes.Sampling, o: PrefillOptions) !Prefilled {
        const q = e.bound.?;
        if (o.live_from) |p| {
            // the sequence's own rows: it must stand exactly there (as the engine checks st.pos)
            if (o.resume_from != null or q.tokens.items.len != p or !std.mem.eql(u32, q.tokens.items, prompt[0..p])) return error.ResumeMustExtend;
        } else if (o.resume_from) |snap| {
            if (snap.pos >= prompt.len or !std.mem.eql(u32, snap.tokens, prompt[0..snap.pos])) return error.ResumeOtherPrompt;
            q.tokens.clearRetainingCapacity();
            try q.tokens.appendSlice(e.gpa, snap.tokens);
            e.resumed += 1;
        } else q.tokens.clearRetainingCapacity(); // the engine's cold reset of a used sequence
        try e.trace.append(e.gpa, .{ q.tokens.items.len, prompt.len });
        try q.tokens.appendSlice(e.gpa, prompt[q.tokens.items.len..]);
        const kept: ?*Snap = if (o.keep_at) |k| (if (e.refuse_keep or e.refuse_keep_at == k) null else try e.snapOf(prompt[0..k])) else null;
        return .{ .first = hash(q.tokens.items), .kept = kept };
    }
    pub fn commitWindow(_: *MiniEngine, _: usize, _: usize) !void {
        return error.NotUsed;
    }
    pub fn placement(e: *MiniEngine, s: *const Seq) ?shard.Placement {
        const p = if (e.pool) |*q| q else return null;
        return shard.Placement.of(p, s.eid.?, e.check != 0) catch null;
    }
    pub fn checkPlacement(e: *MiniEngine, at: shard.Placement) !void {
        const p = if (e.pool) |*q| q else return error.KvShardPlacement;
        try at.verify(p);
    }
    pub fn roundDigest(e: *MiniEngine, round: u64) ?u64 {
        const p = if (e.pool) |*q| q else return null;
        return if ((shard.Switches{ .on = true, .check = e.check }).digestAt(round)) p.digest() else null;
    }
    pub fn checkDigest(e: *MiniEngine, digest: u64) !void {
        const p = if (e.pool) |*q| q else return error.KvShardPlacement;
        try shard.checkLayout(p, digest);
    }
    pub fn current(_: *const MiniEngine) void {}
    pub fn spillOn(e: *const MiniEngine) bool {
        return e.spill_on;
    }
    pub fn spillOut(e: *MiniEngine, snap: *Snap, tokens: []const u32, at: u32) void {
        if (snap.pos != at) return;
        e.disk.append(e.gpa, .{ .key = hash(tokens), .at = at, .tokens = e.gpa.dupe(u32, tokens) catch return }) catch return;
        e.writes += 1;
    }
    pub fn spillBest(e: *MiniEngine, prompt: []const u32, longer_than: u32, at: *u32, key: *u64, tokens: *[]const u32) bool {
        var found = false;
        for (e.disk.items) |d| {
            if (d.at <= longer_than or d.tokens.len > prompt.len or d.at >= prompt.len or !std.mem.eql(u32, prompt[0..d.tokens.len], d.tokens)) continue;
            if (found and d.at <= at.*) continue;
            found = true;
            at.*, key.*, tokens.* = .{ d.at, d.key, d.tokens };
        }
        return found;
    }
    pub const Stored = struct { snap: *Snap, tokens: []u32 };
    pub fn spillIn(e: *MiniEngine, key: u64, at: u32) !Stored {
        for (e.disk.items) |d| if (d.key == key and d.at == at) return .{ .snap = try e.snapOf(d.tokens), .tokens = try e.gpa.dupe(u32, d.tokens) };
        return error.NotStored;
    }
    pub fn spillStats(_: *MiniEngine, out: *[8]u64) void {
        out.* = @splat(0);
    }
    fn deinit(e: *MiniEngine) void {
        for (e.disk.items) |d| e.gpa.free(d.tokens);
        e.disk.deinit(e.gpa);
        e.trace.deinit(e.gpa);
        if (e.pool) |*p| p.deinit();
    }
};

/// The prompt cache's mark hook: the store keeps the state the pass stands at.
const Adopter = struct {
    hooks: CacheHooks,
    saved: [4]*anyopaque = undefined,
    n: usize = 0,
    /// the store drops the first state at the second mark, inside the pass (a cap), instead of writing it
    drop_first: bool = false,
    fn at(ptr: *anyopaque, s: *lanes.Stream, mark: u32) void {
        const a: *Adopter = @ptrCast(@alignCast(ptr));
        a.saved[a.n] = a.hooks.save(a.hooks.ptr, null, mark) catch return;
        a.n += 1;
        // the second state kept: the store writes the first early (the highwater), inside the pass, before the pass ends
        if (a.n == 2 and a.drop_first) a.hooks.drop(a.hooks.ptr, a.saved[0]) else if (a.n == 2) a.hooks.spill.?.evict(a.hooks.ptr, a.saved[0], s.prompt()[0..64], 64);
    }
};

const NoteSink = struct {
    gpa: std.mem.Allocator,
    notes: std.ArrayList([]u8) = .empty,
    fn send(ptr: *anyopaque, bytes: []const u8) void {
        const s: *NoteSink = @ptrCast(@alignCast(ptr));
        s.notes.append(s.gpa, s.gpa.dupe(u8, bytes) catch return) catch {};
    }
    fn take(s: *NoteSink, into: anytype) !void {
        for (s.notes.items) |n| {
            try into.applyNote(n);
            s.gpa.free(n);
        }
        s.notes.clearRetainingCapacity();
    }
    fn deinit(s: *NoteSink) void {
        for (s.notes.items) |n| s.gpa.free(n);
        s.notes.deinit(s.gpa);
    }
};

test "kept states that hold their rows: a pass keeps at its marks, a later pass resumes equal to a fresh one, rank 1 follows by notes, an evicted state goes to the spill tier and comes back" {
    const gpa = std.testing.allocator;
    const B = Lanes(MiniEngine);
    var e0: MiniEngine = .{ .gpa = gpa, .spill_on = true };
    defer e0.deinit();
    var e1: MiniEngine = .{ .gpa = gpa, .spill_on = true };
    defer e1.deinit();
    var r0 = B.init(gpa, &e0, true, 1 << 20, 8);
    var r1 = B.init(gpa, &e1, true, 1 << 20, 8);
    r1.follower = true;
    var sink: NoteSink = .{ .gpa = gpa };
    defer sink.deinit();
    r0.notes = .{ .ptr = &sink, .send = NoteSink.send };
    const hooks = r0.cacheHooks(1 << 30, .{}, 0.5, 5);
    try std.testing.expect(hooks.spill != null);

    var prompt: [150]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i * 7 % 101);
    // 1. a cold pass of 150 tokens with marks at 64 and 128: two tentative states on each rank
    var adopter: Adopter = .{ .hooks = hooks };
    var s1 = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &prompt, .max_new = 4, .reuse = .{ .marks = &.{ 64, 128 }, .hook = .{ .ptr = &adopter, .at = Adopter.at } } });
    defer s1.deinit(gpa);
    try B.prefillFn(&r0, &s1);
    try B.prefillFn(&r1, &s1); // rank 1 runs the same call
    try std.testing.expectEqual(@as(usize, 2), adopter.n);
    const fresh_first = try firstToken(B, &r0, &s1);
    try std.testing.expectEqual(MiniEngine.hash(&prompt), fresh_first);
    // 2. the store adopted both on rank 0 (and wrote the first early, in the pass); the pass's end tells rank 1 which, and
    // the early write waits there for the adoption
    const k64 = adopter.saved[0];
    const k128 = adopter.saved[1];
    try std.testing.expectEqual(@as(usize, 1), e0.writes);
    try sink.take(&r1);
    try std.testing.expectEqual(@as(usize, 1), e1.writes);
    try std.testing.expectEqual(@as(usize, 2), r0.kept.items.len);
    try std.testing.expectEqual(@as(usize, 2), r1.kept.items.len);
    try std.testing.expect(r1.savedFor(r0.keptOf(k128).?.key) != null);
    B.releaseFn(&r0, &s1);
    B.releaseFn(&r1, &s1);
    // 3. a longer prompt resumes at 128 on both ranks and draws what a fresh pass draws
    var longer: [180]u32 = undefined;
    @memcpy(longer[0..150], &prompt);
    for (longer[150..], 0..) |*t, i| t.* = @intCast(900 + i);
    const resumed0, const resumed1 = .{ e0.resumed, e1.resumed };
    var s2 = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &longer, .max_new = 4, .reuse = .{ .saved = k128, .at = 128 } });
    defer s2.deinit(gpa);
    try B.prefillFn(&r0, &s2);
    var s2b = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &longer, .max_new = 4, .reuse = .{ .saved = r1.savedFor(r0.keptOf(k128).?.key), .at = 128 } });
    defer s2b.deinit(gpa);
    try B.prefillFn(&r1, &s2b);
    try std.testing.expectEqual(resumed0 + 1, e0.resumed);
    try std.testing.expectEqual(resumed1 + 1, e1.resumed);
    try std.testing.expectEqual(@as(u32, 128), s2.cached);
    try std.testing.expectEqual(MiniEngine.hash(&longer), (try firstToken(B, &r0, &s2)));
    // 4. the 64-token state is dropped (it is on disk already on both ranks)
    hooks.drop(hooks.ptr, k64);
    try sink.take(&r1);
    try std.testing.expectEqual(@as(usize, 1), r0.kept.items.len);
    try std.testing.expectEqual(@as(usize, 1), r1.kept.items.len);
    // 5. a prompt that extends it finds it on disk, loads it, and rank 1 holds the same state under the same key
    var at: u32 = 0;
    var key: u64 = 0;
    var toks: []const u32 = &.{};
    try std.testing.expect(hooks.spill.?.stored(hooks.ptr, prompt[0..100], 0, &at, &key, &toks));
    try std.testing.expectEqual(@as(u32, 64), at);
    const loaded = try hooks.spill.?.load(hooks.ptr, key, at);
    try sink.take(&r1);
    try std.testing.expectEqual(@as(usize, 2), r0.kept.items.len);
    try std.testing.expect(r1.savedFor(r0.keptOf(loaded).?.key) != null);
    // 6. the state loaded resumes a pass
    var s3 = try lanes.Stream.init(gpa, .{ .id = "c", .prompt = prompt[0..100], .max_new = 4, .reuse = .{ .saved = loaded, .at = 64 } });
    defer s3.deinit(gpa);
    try B.prefillFn(&r0, &s3);
    try std.testing.expectEqual(MiniEngine.hash(prompt[0..100]), (try firstToken(B, &r0, &s3)));
    B.releaseFn(&r0, &s2);
    B.releaseFn(&r1, &s2b);
    B.releaseFn(&r0, &s3);
    // everything is freed with the backends (the testing allocator checks the buffers; sequences and snapshots by count)
    r0.deinit();
    r1.deinit();
    try std.testing.expectEqual(@as(usize, 0), e0.snaps);
    try std.testing.expectEqual(@as(usize, 0), e1.snaps);
    try std.testing.expectEqual(@as(usize, 0), e0.seqs);
    try std.testing.expectEqual(@as(usize, 0), e1.seqs);
}

test "TF_GLM_KV_SHARD: rank 0's placements reach rank 1 with the calls; one of its own elsewhere ends the pair, and with the check any other layout does" {
    const gpa = std.testing.allocator;
    const Pool = @import("cuda_pool.zig").Pool;
    const B = Lanes(MiniEngine);
    for ([_]bool{ false, true }) |check| {
        var e0: MiniEngine = .{ .gpa = gpa, .pool = try Pool.initParts(gpa, 8 * 2048, 2), .check = @intFromBool(check) };
        defer e0.deinit();
        var e1: MiniEngine = .{ .gpa = gpa, .pool = try Pool.initParts(gpa, 8 * 2048, 2), .check = @intFromBool(check) };
        defer e1.deinit();
        var r0 = B.init(gpa, &e0, true, 1 << 20, 8);
        defer r0.deinit();
        var r1 = B.init(gpa, &e1, true, 1 << 20, 8);
        defer r1.deinit();
        r1.follower = true;
        var sink: NoteSink = .{ .gpa = gpa };
        defer sink.deinit();
        r0.notes = .{ .ptr = &sink, .send = NoteSink.send };
        const prompt = testPrompt(7);
        // 1. the same placements on both ranks: one note a stream, each checked
        var s1 = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &prompt, .max_new = 4 });
        defer s1.deinit(gpa);
        try B.prefillFn(&r0, &s1);
        try B.prefillFn(&r1, &s1);
        try std.testing.expectEqual(@as(usize, 1), sink.notes.items.len);
        try std.testing.expectEqual(@as(u8, @backingInt(Note.place)), sink.notes.items[0][0]);
        const at = try shard.Placement.decode(sink.notes.items[0][1..]);
        try std.testing.expectEqual(@as(u64, 0), at.base);
        try std.testing.expectEqual(check, at.digest != 0);
        try sink.take(&r1);
        // 2. rank 1's layout drifts (an extent of its own moved), a placement still alike: only the check sees it
        const first = r1.lanes.get(&s1).?.seq.eid.?;
        _ = try e1.pool.?.move(first, 4096, null, false);
        var s2 = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &prompt, .max_new = 4 });
        defer s2.deinit(gpa);
        try B.prefillFn(&r0, &s2);
        try B.prefillFn(&r1, &s2);
        try std.testing.expectEqual(@as(u64, 8192), (try shard.Placement.decode(sink.notes.items[0][1..])).base); // the part with fewer streams
        if (check) try std.testing.expectError(error.KvShardPlacement, sink.take(&r1)) else try sink.take(&r1);
        for (sink.notes.items) |n| gpa.free(n);
        sink.notes.clearRetainingCapacity();
        // 3. a placement of rank 1's own elsewhere (an extent rank 0 never made) is refused, checked or not
        _ = try e1.pool.?.add(8192 + 4096, 2048);
        var s3 = try lanes.Stream.init(gpa, .{ .id = "c", .prompt = &prompt, .max_new = 4 });
        defer s3.deinit(gpa);
        try B.prefillFn(&r0, &s3);
        try B.prefillFn(&r1, &s3);
        try std.testing.expectError(error.KvShardPlacement, sink.take(&r1));
        for (sink.notes.items) |n| gpa.free(n);
        sink.notes.clearRetainingCapacity();
        // the extent rank 1 made itself goes before the lanes release theirs
        for (e1.pool.?.extents.items) |x| if (x.owner == null) {
            try e1.pool.?.remove(x.eid);
            break;
        };
    }
    // off (no pool): no note at all, as before
    var e0: MiniEngine = .{ .gpa = gpa };
    defer e0.deinit();
    var r0 = B.init(gpa, &e0, true, 1 << 20, 8);
    defer r0.deinit();
    var sink: NoteSink = .{ .gpa = gpa };
    defer sink.deinit();
    r0.notes = .{ .ptr = &sink, .send = NoteSink.send };
    const prompt = testPrompt(7);
    var s = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &prompt, .max_new = 4 });
    defer s.deinit(gpa);
    try B.prefillFn(&r0, &s);
    r0.round = shard.check_rounds;
    r0.layoutNote();
    try std.testing.expectEqual(@as(usize, 0), sink.notes.items.len);
}

test "TF_GLM_KV_SHARD: a prefill rank 0 refuses before its sequence leaves rank 1's extent ids one ahead, and the placements and digests still check" {
    const gpa = std.testing.allocator;
    const Pool = @import("cuda_pool.zig").Pool;
    const B = Lanes(MiniEngine);
    for ([_]u32{ 0, 1 }) |check| {
        var e0: MiniEngine = .{ .gpa = gpa, .pool = try Pool.initParts(gpa, 8 * 2048, 2), .check = check };
        defer e0.deinit();
        var e1: MiniEngine = .{ .gpa = gpa, .pool = try Pool.initParts(gpa, 8 * 2048, 2), .check = check };
        defer e1.deinit();
        // rank 0 refuses a 150-token prompt before it opens a sequence (a check of its own, as the engine's
        // NoVisionTower); rank 1 opens one, then drops it when the ranks agree on the refusal
        var r0 = B.init(gpa, &e0, true, 100, 8);
        defer r0.deinit();
        var r1 = B.init(gpa, &e1, true, 1 << 20, 8);
        defer r1.deinit();
        r1.follower = true;
        var sink: NoteSink = .{ .gpa = gpa };
        defer sink.deinit();
        r0.notes = .{ .ptr = &sink, .send = NoteSink.send };
        var gate: TestGate = .{ .other = &.{ true, false, true, true, true, true } };
        r0.gate = .{ .ptr = &gate, .agree = TestGate.agree, .fatal = TestGate.fatal };
        r1.gate = .{ .ptr = &gate, .agree = TestGate.agree, .fatal = TestGate.fatal };
        const prompt = testPrompt(7);
        var s1 = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &prompt, .max_new = 4 });
        defer s1.deinit(gpa);
        try std.testing.expectError(error.PromptTooLong, B.prefillFn(&r0, &s1));
        try std.testing.expectError(error.OtherRankFailed, B.prefillFn(&r1, &s1));
        try std.testing.expectEqual(@as(u64, 0), e0.pool.?.next_id);
        try std.testing.expectEqual(@as(u64, 1), e1.pool.?.next_id);
        // the next prompts: the same extents at the same bases on both ranks, under other ids; each note checks
        var s2 = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = prompt[0..50], .max_new = 4 });
        defer s2.deinit(gpa);
        var s3 = try lanes.Stream.init(gpa, .{ .id = "c", .prompt = prompt[0..50], .max_new = 4 });
        defer s3.deinit(gpa);
        for ([_]*lanes.Stream{ &s2, &s3 }) |s| {
            try B.prefillFn(&r0, s);
            try B.prefillFn(&r1, s);
            // rank 1 takes the note before its next call, as the mirror delivers it
            try std.testing.expectEqual(@as(usize, 1), sink.notes.items.len);
            try std.testing.expectEqual(check != 0, (try shard.Placement.decode(sink.notes.items[0][1..])).digest != 0);
            try sink.take(&r1);
        }
        // a decode round's digest of the same layout under other ids
        r0.round = 1;
        r0.layoutNote();
        try std.testing.expectEqual(@as(usize, @intFromBool(check != 0)), sink.notes.items.len);
        try sink.take(&r1);
    }
}

test "TF_GLM_KV_SHARD_CHECK: every that many decode rounds rank 0's layout digest reaches rank 1, which ends the pair on a drift; 0 sends none" {
    const gpa = std.testing.allocator;
    const Pool = @import("cuda_pool.zig").Pool;
    const B = Lanes(MiniEngine);
    for ([_]u32{ 0, 2 }) |check| {
        var e0: MiniEngine = .{ .gpa = gpa, .pool = try Pool.initParts(gpa, 8 * 2048, 2), .check = check };
        defer e0.deinit();
        var e1: MiniEngine = .{ .gpa = gpa, .pool = try Pool.initParts(gpa, 8 * 2048, 2), .check = check };
        defer e1.deinit();
        var r0 = B.init(gpa, &e0, true, 1 << 20, 8);
        defer r0.deinit();
        var r1 = B.init(gpa, &e1, true, 1 << 20, 8);
        defer r1.deinit();
        r1.follower = true;
        var sink: NoteSink = .{ .gpa = gpa };
        defer sink.deinit();
        r0.notes = .{ .ptr = &sink, .send = NoteSink.send };
        const prompt = testPrompt(7);
        var s = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &prompt, .max_new = 4 });
        defer s.deinit(gpa);
        try B.prefillFn(&r0, &s);
        try B.prefillFn(&r1, &s);
        try sink.take(&r1);
        // round 1: none; round 2: the digest, alike
        r0.round = 1;
        r0.layoutNote();
        try std.testing.expectEqual(@as(usize, 0), sink.notes.items.len);
        r0.round = 2;
        r0.layoutNote();
        try std.testing.expectEqual(@as(usize, @intFromBool(check != 0)), sink.notes.items.len);
        if (check != 0) try std.testing.expectEqual(@as(u8, @backingInt(Note.layout)), sink.notes.items[0][0]);
        try sink.take(&r1);
        // rank 1 drifts between placements (its stream's extent elsewhere): the next digest ends the pair
        _ = try e1.pool.?.move(r1.lanes.get(&s).?.seq.eid.?, 4096, null, false);
        r0.round = 4;
        r0.layoutNote();
        if (check != 0) try std.testing.expectError(error.KvShardPlacement, sink.take(&r1)) else try std.testing.expectEqual(@as(usize, 0), sink.notes.items.len);
        for (sink.notes.items) |n| gpa.free(n);
        sink.notes.clearRetainingCapacity();
        try std.testing.expectError(error.BadNote, r1.applyNote(&.{ @backingInt(Note.layout), 1, 2 }));
    }
}

fn firstToken(comptime B: type, r: *B, s: *lanes.Stream) !u32 {
    return B.readFn(r, try B.firstFn(r, s, s.prompt_len));
}

test "a state the store drops inside the pass that kept it is gone from the pass's list on both ranks" {
    const gpa = std.testing.allocator;
    const B = Lanes(MiniEngine);
    var e0: MiniEngine = .{ .gpa = gpa };
    defer e0.deinit();
    var e1: MiniEngine = .{ .gpa = gpa };
    defer e1.deinit();
    var r0 = B.init(gpa, &e0, true, 1 << 20, 8);
    var r1 = B.init(gpa, &e1, true, 1 << 20, 8);
    r1.follower = true;
    var sink: NoteSink = .{ .gpa = gpa };
    defer sink.deinit();
    r0.notes = .{ .ptr = &sink, .send = NoteSink.send };
    const hooks = r0.cacheHooks(1 << 30, .{}, 0, 5);
    var prompt: [150]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i * 7 % 101);
    var adopter: Adopter = .{ .hooks = hooks, .drop_first = true };
    var s1 = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &prompt, .max_new = 4, .reuse = .{ .marks = &.{ 64, 128 }, .hook = .{ .ptr = &adopter, .at = Adopter.at } } });
    defer s1.deinit(gpa);
    try B.prefillFn(&r0, &s1);
    try B.prefillFn(&r1, &s1);
    try sink.take(&r1); // the drop of a state rank 1 has not adopted, then the pass's list of one
    try std.testing.expectEqual(@as(usize, 1), r0.kept.items.len);
    try std.testing.expectEqual(@as(usize, 1), r1.kept.items.len);
    try std.testing.expectEqual(@as(usize, 1), e0.snaps);
    try std.testing.expectEqual(@as(usize, 1), e1.snaps);
    B.releaseFn(&r0, &s1);
    B.releaseFn(&r1, &s1);
    r0.deinit();
    r1.deinit();
    try std.testing.expectEqual(@as(usize, 0), e0.snaps);
    try std.testing.expectEqual(@as(usize, 0), e1.snaps);
}

/// A gate for tests: the ranks' all-gather, with the other rank's vote scripted per ask. A prefill asks once for the
/// prepare agreement (`agree`), then per mark it would keep at: once for the room (`agreeOn`), and once for the save's
/// outcome when the room was agreed. `other` holds the other rank's votes in that order, rank 0's asks first, then
/// rank 1's (the tests run the ranks one after the other).
const TestGate = struct {
    other: []const bool,
    asked: usize = 0,
    fn agree(ptr: *anyopaque, ok: bool) anyerror!bool {
        const g: *TestGate = @ptrCast(@alignCast(ptr));
        const vote = g.other[g.asked];
        g.asked += 1;
        return ok and vote;
    }
    fn fatal(_: *anyopaque, _: []const u8, _: anyerror) void {}
};

/// Two ranks on one gate: rank 1 follows rank 0's notes through `sink`.
const Pair = struct {
    e0: MiniEngine,
    e1: MiniEngine,
    r0: Lanes(MiniEngine),
    r1: Lanes(MiniEngine),
    sink: NoteSink,
    gate: TestGate,
    fn init(p: *Pair, gpa: std.mem.Allocator, other: []const bool) void {
        p.e0 = .{ .gpa = gpa };
        p.e1 = .{ .gpa = gpa };
        p.r0 = Lanes(MiniEngine).init(gpa, &p.e0, true, 1 << 20, 8);
        p.r1 = Lanes(MiniEngine).init(gpa, &p.e1, true, 1 << 20, 8);
        p.r1.follower = true;
        p.sink = .{ .gpa = gpa };
        p.r0.notes = .{ .ptr = &p.sink, .send = NoteSink.send };
        p.gate = .{ .other = other };
        p.r0.gate = .{ .ptr = &p.gate, .agree = TestGate.agree, .fatal = TestGate.fatal };
        p.r1.gate = .{ .ptr = &p.gate, .agree = TestGate.agree, .fatal = TestGate.fatal };
    }
    fn deinit(p: *Pair) void {
        p.r0.deinit();
        p.r1.deinit();
        p.sink.deinit();
        p.e0.deinit();
        p.e1.deinit();
    }
    /// Both ranks run the stream's pass; rank 1 takes the notes; both first tokens are the fresh prompt's.
    fn pass(p: *Pair, s: *lanes.Stream) !void {
        const B = Lanes(MiniEngine);
        try B.prefillFn(&p.r0, s);
        try B.prefillFn(&p.r1, s);
        try p.sink.take(&p.r1);
        try std.testing.expectEqual(MiniEngine.hash(s.prompt()), try firstToken(B, &p.r0, s));
        try std.testing.expectEqual(MiniEngine.hash(s.prompt()), try firstToken(B, &p.r1, s));
        try std.testing.expectEqualSlices([2]usize, p.e0.trace.items, p.e1.trace.items);
        B.releaseFn(&p.r0, s);
        B.releaseFn(&p.r1, s);
    }
};

fn testPrompt(seed: u32) [150]u32 {
    var prompt: [150]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast((i * seed) % 101);
    return prompt;
}

const whole_pass: []const [2]usize = &.{ .{ 0, 64 }, .{ 64, 128 }, .{ 128, 150 } }; // every row once

test "a keep that would cross a rank's keep floor is refused on every rank before the agreement; the pass goes on and keeps again when there is room" {
    const gpa = std.testing.allocator;
    var p: Pair = undefined;
    // rank 0 has no room at either mark: it votes no itself; rank 1 (room) hears that no from the all-gather
    p.init(gpa, &.{ true, true, true, true, false, false });
    defer p.deinit();
    p.e0.keep_fits = false;
    const hooks = p.r0.cacheHooks(1 << 30, .{}, 0, 5);
    const prompt = testPrompt(7);
    var adopter: Adopter = .{ .hooks = hooks, .drop_first = true }; // the store caps at one, as the existing tests
    // 1. marks at 64 and 128: every rank asks, no rank keeps, no rows run twice, nothing leaks
    var s1 = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &prompt, .max_new = 4, .reuse = .{ .marks = &.{ 64, 128 }, .hook = .{ .ptr = &adopter, .at = Adopter.at } } });
    defer s1.deinit(gpa);
    try p.pass(&s1);
    try std.testing.expectEqual(@as(usize, 6), p.gate.asked); // each rank: the prepare agreement and the two rooms
    try std.testing.expectEqual(@as(usize, 0), adopter.n);
    try std.testing.expectEqual(@as(usize, 0), p.r0.kept.items.len);
    try std.testing.expectEqual(@as(usize, 0), p.r1.kept.items.len);
    try std.testing.expectEqualSlices([2]usize, &.{.{ 0, 150 }}, p.e0.trace.items);
    try std.testing.expectEqual(@as(usize, 0), p.e0.snaps);
    try std.testing.expectEqual(@as(usize, 0), p.e1.snaps);
    // 2. with room on every rank, the next pass keeps at both marks (the store drops the first: one left) and rank 1 follows
    p.gate.other = &.{ true, true, true, true, true, true, true, true, true, true };
    p.gate.asked = 0;
    p.e0.keep_fits = true;
    p.e0.trace.clearRetainingCapacity();
    p.e1.trace.clearRetainingCapacity();
    var s2 = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &prompt, .max_new = 4, .reuse = .{ .marks = &.{ 64, 128 }, .hook = .{ .ptr = &adopter, .at = Adopter.at } } });
    defer s2.deinit(gpa);
    try p.pass(&s2);
    try std.testing.expectEqual(@as(usize, 10), p.gate.asked); // each rank: prepare, then room and outcome at two marks
    try std.testing.expectEqual(@as(usize, 2), adopter.n);
    try std.testing.expectEqual(@as(usize, 1), p.r0.kept.items.len);
    try std.testing.expectEqual(@as(usize, 1), p.r1.kept.items.len);
    try std.testing.expectEqualSlices([2]usize, whole_pass, p.e0.trace.items);
}

test "a save the engine refuses leaves the pass going on from the rows it ran: no state, no rows twice, the reply the fresh one; with an earlier state, that one stays" {
    const gpa = std.testing.allocator;
    const B = Lanes(MiniEngine);
    var e0: MiniEngine = .{ .gpa = gpa, .refuse_keep = true };
    defer e0.deinit();
    var r0 = B.init(gpa, &e0, true, 1 << 20, 8);
    defer r0.deinit();
    const hooks = r0.cacheHooks(1 << 30, .{}, 0, 5);
    const prompt = testPrompt(11);
    var adopter: Adopter = .{ .hooks = hooks, .drop_first = true };
    // 1. both saves refused from cold: the segments go on from the live rows
    var s1 = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &prompt, .max_new = 4, .reuse = .{ .marks = &.{ 64, 128 }, .hook = .{ .ptr = &adopter, .at = Adopter.at } } });
    defer s1.deinit(gpa);
    try B.prefillFn(&r0, &s1);
    try std.testing.expectEqual(@as(usize, 0), adopter.n);
    try std.testing.expectEqual(@as(usize, 0), r0.kept.items.len);
    try std.testing.expectEqualSlices([2]usize, whole_pass, e0.trace.items);
    try std.testing.expectEqual(MiniEngine.hash(&prompt), try firstToken(B, &r0, &s1));
    B.releaseFn(&r0, &s1);
    try std.testing.expectEqual(@as(usize, 0), e0.snaps);
    // 2. the 64 state kept, the save at 128 refused: the pass goes on live from 128, the 64 state stays kept
    e0.refuse_keep = false;
    e0.refuse_keep_at = 128;
    e0.trace.clearRetainingCapacity();
    var s2 = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &prompt, .max_new = 4, .reuse = .{ .marks = &.{ 64, 128 }, .hook = .{ .ptr = &adopter, .at = Adopter.at } } });
    defer s2.deinit(gpa);
    try B.prefillFn(&r0, &s2);
    try std.testing.expectEqual(@as(usize, 1), adopter.n);
    try std.testing.expectEqual(@as(usize, 1), r0.kept.items.len);
    try std.testing.expectEqualSlices([2]usize, whole_pass, e0.trace.items);
    try std.testing.expectEqual(MiniEngine.hash(&prompt), try firstToken(B, &r0, &s2));
    B.releaseFn(&r0, &s2);
    try std.testing.expectEqual(@as(usize, 1), e0.snaps); // the kept 64 state
    // 3. nothing refused: both marks keep again
    e0.refuse_keep_at = null;
    e0.trace.clearRetainingCapacity();
    var s3 = try lanes.Stream.init(gpa, .{ .id = "c", .prompt = &prompt, .max_new = 4, .reuse = .{ .marks = &.{ 64, 128 }, .hook = .{ .ptr = &adopter, .at = Adopter.at } } });
    defer s3.deinit(gpa);
    try B.prefillFn(&r0, &s3);
    try std.testing.expectEqual(@as(usize, 3), adopter.n);
    try std.testing.expectEqualSlices([2]usize, whole_pass, e0.trace.items);
    try std.testing.expectEqual(MiniEngine.hash(&prompt), try firstToken(B, &r0, &s3));
    B.releaseFn(&r0, &s3);
}

/// One rank's save refused after the room agreement: the other rank frees the state it took, both go on from the live
/// rows the same way, and both keep at the next mark.
fn lateRefusal(refuser: u8) !void {
    const gpa = std.testing.allocator;
    var p: Pair = undefined;
    // each rank: prepare, room at 64, the save at 64, room at 128, the save at 128; the refusing rank votes no itself
    p.init(gpa, if (refuser == 1) &.{ true, true, false, true, true, true, true, true, true, true } else &.{ true, true, true, true, true, true, true, false, true, true });
    defer p.deinit();
    if (refuser == 1) p.e1.refuse_keep_at = 64 else p.e0.refuse_keep_at = 64;
    const hooks = p.r0.cacheHooks(1 << 30, .{}, 0, 5);
    const prompt = testPrompt(13);
    var adopter: Adopter = .{ .hooks = hooks, .drop_first = true };
    var s1 = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &prompt, .max_new = 4, .reuse = .{ .marks = &.{ 64, 128 }, .hook = .{ .ptr = &adopter, .at = Adopter.at } } });
    defer s1.deinit(gpa);
    try p.pass(&s1);
    try std.testing.expectEqual(@as(usize, 10), p.gate.asked);
    try std.testing.expectEqual(@as(usize, 1), adopter.n); // the 128 state only
    try std.testing.expectEqual(@as(usize, 1), p.r0.kept.items.len);
    try std.testing.expectEqual(@as(usize, 1), p.r1.kept.items.len);
    try std.testing.expectEqual(@as(u32, 128), p.r0.kept.items[0].at);
    try std.testing.expectEqual(@as(u32, 128), p.r1.kept.items[0].at);
    try std.testing.expectEqualSlices([2]usize, whole_pass, p.e0.trace.items);
    try std.testing.expectEqual(@as(usize, 1), p.e0.snaps); // the state taken at 64 and not agreed is gone
    try std.testing.expectEqual(@as(usize, 1), p.e1.snaps);
}

test "a save refused on rank 1 only is kept by no rank; both go on from the live rows and keep at the next mark" {
    try lateRefusal(1);
}

test "a save refused on rank 0 only is kept by no rank; both go on from the live rows and keep at the next mark" {
    try lateRefusal(0);
}

test "a keep batch with a bad path commits nothing and leaves every window pending; the next good batch commits them" {
    const gpa = std.testing.allocator;
    const Toy = @import("flashnext").lane_backend.Toy;
    var t: Toy = .{ .gpa = gpa };
    var b = Lanes(Toy).init(gpa, &t, true, 4096, 8);
    defer b.deinit();
    b.streams = 2;
    const B = Lanes(Toy);
    const prompt = [_]u32{ 5, 6, 7, 8, 9, 10, 11, 12 };
    var s1 = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &prompt, .max_new = 8 });
    defer s1.deinit(gpa);
    var s2 = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &prompt, .max_new = 8 });
    defer s2.deinit(gpa);
    try B.prefillFn(&b, &s1);
    try B.prefillFn(&b, &s2);
    const t1 = try firstToken(B, &b, &s1);
    const t2 = try firstToken(B, &b, &s2);
    // a round of two windows: the pending token and two forced drafts each
    const d1 = [_]u32{ 1, 2 };
    const d2 = [_]u32{ 3, 4 };
    const pos = [_]u64{ prompt.len + 1, prompt.len + 2, prompt.len + 3 };
    const windows = [_]be.Window{
        .{ .stream = &s1, .pending = t1, .held = 0, .tokens = &d1, .parents = null, .positions = &pos },
        .{ .stream = &s2, .pending = t2, .held = 0, .tokens = &d2, .parents = null, .positions = &pos },
    };
    var out: [2]be.Verified = undefined;
    var sampled: [2][3]u32 = undefined;
    var drafts: [2][2]u32 = undefined;
    for (&out, 0..) |*o, i| o.* = .{ .sampled = &sampled[i], .drafts = &drafts[i] };
    try B.verifyFn(&b, &windows, &out);
    const commits = t.batched_commits;
    // the second path is empty: nothing commits, both windows stay pending
    const good = [_]u32{ 0, 1 };
    try std.testing.expectError(error.EmptyPath, B.keepFn(&b, &windows, &.{ &good, &.{} }));
    try std.testing.expectEqual(commits, t.batched_commits);
    try std.testing.expectEqual(@as(?usize, 3), b.lanes.get(&s1).?.pending_rows);
    try std.testing.expectEqual(@as(?usize, 3), b.lanes.get(&s2).?.pending_rows);
    // the good batch commits both at once
    const one = [_]u32{0};
    try B.keepFn(&b, &windows, &.{ &good, &one });
    try std.testing.expectEqual(commits + 1, t.batched_commits);
    try std.testing.expectEqual(@as(?usize, null), b.lanes.get(&s1).?.pending_rows);
    try std.testing.expectEqual(@as(?usize, null), b.lanes.get(&s2).?.pending_rows);
    B.releaseFn(&b, &s1);
    B.releaseFn(&b, &s2);
}
