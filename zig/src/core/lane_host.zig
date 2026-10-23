//! The lane core served to the HTTP threads: one thread owns ``core`` and steps rounds while any stream lives.
//!
//! With a durable tier on the prompt cache (spill v2: wojo's GLM 0110, the DeepSeek port's lane host on v1.0.7's
//! admission), a request whose prompt a longer kept state on disk would serve is parked before its prompt pass on a
//! restore job (shared by every request that wants the same point), or deferred briefly when the tier refuses the
//! restore for capacity, or for anything but the point itself while other restores are in flight (it waits for them
//! rather than prefilling beside them, and asks again when one ends). Parked requests hold no lane and never join a
//! burst; they count as waiting and wake ahead of the queue, where an ordinary lookup runs again. The tier is serviced
//! after each turn's admissions and prompt chunk (between rounds) and every 2 ms when idle, and a clean stop flushes it
//! on this thread. With no tier nothing is parked, serviced or published.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const status = @import("lane_host_status.zig");
const learn = @import("lane_host_learn.zig");
const Allocator = std.mem.Allocator;
const Id = api.Id;
const Request = api.Request;
const Sink = api.Sink;
const Event = api.Event;
const Engine = api.Engine;
const Info = api.Info;
const Memory = api.Memory;
const Reason = api.Reason;
const Stats = api.Stats;
const SubmitError = api.SubmitError;

pub const LaneHost = struct {
    gpa: Allocator,
    io: std.Io,
    core: *lanes.Engine,
    info_: Info,
    min_match: i64 = 4,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queued: std.ArrayList(*Job) = .empty,
    admitted: std.ArrayList(*Job) = .empty,
    filling: std.ArrayList(*Job) = .empty, // admitted jobs whose prompt is still going in, a chunk a round, oldest first
    cancels: std.ArrayList(Id) = .empty,
    closing: bool = false,
    thread: ?std.Thread = null,
    decoded: std.ArrayList(status.Mark) = .empty, // tokens a round landed, for the 2 s decode rate
    prefill_rate: f64 = 0,
    prefill_at: i96 = 0,
    live_tokens: std.ArrayList(u32) = .empty,
    /// A Metal engine's keepalive target, set by the family that owns the queue; null keeps the ticker off.
    keepalive_target: ?api.keepalive.Target = null,
    live_generated: u64 = 0,
    /// the prompt cache's numbers as of its last change (status.noteKept), for /health
    kept_view: ?pc.KeptStats = null,
    extra_status: ?struct { ctx: *anyopaque, read: *const fn (*anyopaque, *api.Status) void } = null,
    lone: ?api.Lone = null, // the backend's driver for a lone greedy stream; null: every stream in the lane core
    lone_job: ?*Job = null, // the job that driver holds now
    cache: ?*pc.Store = null, // kept prompt states (engine thread only); the backend restores and saves them
    decoded_rows: bool = false, // the backend prefills Request.decode_spans with decoded rows' arithmetic, cache or not
    memory: ?api.MemorySource = null, // the backend's memory counts; null: Engine.memory reports none
    explain: ?api.Explain = null, // the backend's words for a request it refuses; null: the error's name
    learner: ?api.Learner = null, // the family's Sliding Weights learner; null: learn requests are refused
    lessons: std.ArrayList(Lesson) = .empty, // learn requests waiting for an idle engine
    lesson_open: bool = false, // the learner holds a begun job (engine thread only)
    // -- parked requests (spill v2). The lists are the engine thread's; changes take the lock (status reads their lengths).
    ready: std.ArrayList(*Job) = .empty, // woken: admitted before the queue, in the order they woke
    loading: std.ArrayList(Load) = .empty, // each waits for a restore job (several may share one)
    deferred: std.ArrayList(Defer) = .empty, // restores the tier refused for capacity, asked again shortly
    turn: usize = 0, // which restore a service polls first (one budget, shared fairly)
    lone_commits: u32 = 0, // the lone driver's commits since its last pump
    passes_ns: u64 = 0, // engine-thread time nothing could be serviced: admissions' prompt passes, turns an entry was held
    held_from: ?i96 = null, // since this turn's admissions an admitted pass holds a looked-up entry (entryHeld)
    tier: TierPolicy = .{},
    host_counts: HostCounts = .{},
    /// What status reports under `cache` (published on the engine thread at each noteLive; null without a tier).
    tele: ?api.CacheStats = null,
    /// A backend's KV-ownership stats (its parts), read on the engine thread at each noteLive, published as `kv_tele`.
    kv_parts: ?lanes.kv_parts.Source = null,
    kv_tele: ?api.KvParts = null,

    const Lesson = struct { request: *const api.LearnRequest, sink: api.LearnSink };
    /// How often an idle host polls work in flight (restores, staged writes, deferred retries).
    const poll_ns: i64 = 2 * std.time.ns_per_ms;
    /// The lone driver pumps staged writes every this many commits.
    const lone_pump_commits: u32 = 16;

    /// A request parked on `restore` since `since` (`passes`: passes_ns then).
    const Load = struct { job: *Job, restore: *pc.Restore, since: i96, passes: u64 };
    /// A request waiting to ask again at `retry_at`; `until_end`: also as soon as no restore is left in flight.
    const Defer = struct { job: *Job, retry_at: i96, until_end: bool = false };

    /// How parked work is serviced (LaneHost.configure).
    pub const TierPolicy = struct {
        /// One elapsed-time budget per service for every restore together, then staged writes (TF_GLM_SPILL_LOAD_MS).
        service_ns: u64 = 40 * std.time.ns_per_ms,
        /// Staging's share of a service while streams decode (GLM pumped 4 blocks a round); idle, all that is left.
        pump_ns: u64 = 2 * std.time.ns_per_ms,
        /// How long a request waits parked on a restore before it is admitted without it (TF_SPILL_LOAD_WAIT_MS), not
        /// counting the engine thread's time in other requests' prompt passes (nothing is serviced then).
        load_wait_ns: i96 = 30 * std.time.ns_per_s,
        /// A clean shutdown's bound for draining writes (--spill-flush-s).
        flush_ns: u64 = 60 * std.time.ns_per_s,
        /// A restore the tier refused in a way waiting may change (Store.deferrable) is asked again when a restore in
        /// flight ends (and a capacity refusal every defer_retry_ns), for at most defer_ns (TF_SPILL_DEFER_MS; 0: never,
        /// admitted on memory at once), when it would save at least defer_min tokens over memory (0: the store's grid,
        /// or 1).
        defer_ns: i96 = 5 * std.time.ns_per_s,
        defer_retry_ns: i96 = 25 * std.time.ns_per_ms,
        defer_min: u32 = 0,
        /// A prompt pass that would compute at least hold_min tokens waits while other requests are parked on
        /// restores, at most defer_ns of the time they could be serviced (holdPass; 0: no pass is held): a whole pass
        /// services nothing, so every parked request, whose restore needs only a few services, would wait for all of
        /// it. No setting: 8192 tokens is about 4 s of a GLM prefill (g2's 2K tokens a second), and TF_SPILL_DEFER_MS=0
        /// turns the hold off with the deferral.
        hold_min: u32 = 8192,
    };

    pub const HostCounts = struct {
        restore_waits: u64 = 0, // requests parked on a restore
        restore_wait_timeouts: u64 = 0, // of them admitted without it at load_wait_ns
        restore_wait_ns: u64 = 0,
        restore_wait_max_ns: u64 = 0,
        restores_deferred: u64 = 0, // requests whose restore was refused (Store.deferrable) and asked again
        restores_deferred_expired: u64 = 0, // of them, admitted without it: at defer_ns, or refused with none in flight
        prefills_held: u64 = 0, // requests whose long pass waited for others' restores (holdPass)
        prefills_held_expired: u64 = 0, // of them, admitted at defer_ns while requests were still parked
        cached_tokens: u64 = 0, // prompt tokens passes resumed (memory or restored)
        computed_tokens: u64 = 0, // and computed
    };

    const Job = struct {
        host: *LaneHost,
        id: Id,
        request: *const Request,
        sink: Sink,
        stream: lanes.Stream = undefined,
        proposer: lanes.SuffixLookup = undefined,
        delivered: usize = 0,
        started: bool = false,
        prefill_sent: bool = false, // a lone driver's prefilled event went out
        fill_began: bool = false, // its first prompt chunk ran
        began: i96 = 0,
        prefilled: ?i96 = null,
        entry: ?*pc.Entry = null, // the kept state the backend restores, until its prompt pass reports
        marks: []const u32 = &.{}, // where the pass keeps states (gpa-owned)
        kept0: u64 = 0, // the store's kept count when the job looked it up
        join: pc.Join = .{}, // the conversation its kept states join (0111), decided at its lookup
        tried_disk: bool = false, // it waited for a restore once: a later lookup takes what memory has
        waited: bool = false, // it waited on restores (parked on its own, or deferred): its pass is never held (holdPass)
        deferred_since: ?i96 = null, // its restore was first refused (Store.deferrable) then (asked again until defer_ns)
        deferred_passes: u64 = 0, // passes_ns then
        held_since: ?i96 = null, // its long pass first waited for others' restores then (holdPass, until defer_ns)
        held_passes: u64 = 0, // passes_ns then
        parent: ?u64 = null, // the durable point its pass resumes from, and where (spill v2's lineage)
        fork: u32 = 0,
        lineage: []pc.Fork = &.{}, // the computation its pass's rows belong to (gpa-owned; Store.passLineage)

        /// The backend's prompt pass stands at a mark: the cache keeps the stream's state there.
        fn kept(ptr: *anyopaque, s: *lanes.Stream, at: u32) void {
            const job: *Job = @ptrCast(@alignCast(ptr));
            job.reported(); // before a keep can evict the entry the pass restored
            if (job.host.cache) |store| {
                _ = store.keepWith(job.request.prompt, at, s, job.request.chunks, job.host.spans(job.request), job.keepInfo());
                status.noteKept(job.host);
            }
        }

        /// What its keeps record: its conversation, the point its pass resumed and the pass's lineage.
        fn keepInfo(job: *const Job) pc.KeepInfo {
            return .{ .chat = job.join.chat, .parent = job.parent, .fork = job.fork, .lineage = job.lineage };
        }

        /// Report a restored prefix or its failed copy; an untouched prefix remains kept.
        fn reported(job: *Job) void {
            const e = job.entry orelse return;
            job.entry = null;
            const store = job.host.cache orelse return;
            if (!job.started) return;
            job.stream.reuse.saved = null; // the backend restores before its first chunk; a later keep may free it
            const ok = !job.stream.reuse_failed and job.stream.cached == e.at;
            if (!ok) { // it did not compute from that state's rows: a lineage of its own
                job.parent = null;
                job.fork = 0;
                if (job.lineage.len > 0) {
                    const fresh: []pc.Fork = store.passLineage(null) catch &.{};
                    job.host.gpa.free(job.lineage);
                    job.lineage = fresh;
                }
            }
            if (job.stream.reuse_failed) store.resumed(e, job.request.prompt, false) else if (ok) store.resumed(e, job.request.prompt, true);
        }
    };

    pub fn init(gpa: Allocator, io: std.Io, core: *lanes.Engine, info_: Info) LaneHost {
        var enforced = info_;
        enforced.loop_guard = true;
        return .{ .gpa = gpa, .io = io, .core = core, .info_ = enforced };
    }

    /// The lane core's word that a filling prompt is in (or its fill ended): emit `prefilled`, or end the job.
    fn filled(ptr: *anyopaque, s: *lanes.Stream, err: ?anyerror) void {
        const h: *LaneHost = @ptrCast(@alignCast(ptr));
        const job: *Job = @alignCast(@fieldParentPtr("stream", s));
        const e = err orelse return h.prefilled(job, job.began); // the run loop delivers its tokens
        if (e == error.Cancelled) {
            _ = h.cancel(job);
        } else _ = h.drop(job, @errorName(e));
    }

    /// Spill v2's service bounds (before start, after the cache and its tier are set: Store.setTier). Telemetry is
    /// published from here on when the cache has a tier.
    pub fn configure(h: *LaneHost, sp: api.Spill) void {
        // (a 0 ns budget never progresses a restore: 1 ms then)
        h.tier.service_ns = if (sp.load_ms > 0) @intFromFloat(@max(1, @min(sp.load_ms, std.math.maxInt(u32)) * std.time.ns_per_ms)) else std.time.ns_per_ms;
        h.tier.pump_ns = @min(h.tier.pump_ns, h.tier.service_ns);
        h.tier.load_wait_ns = @as(i96, @max(sp.load_wait_ms, 1)) * std.time.ns_per_ms;
        h.tier.flush_ns = @as(u64, sp.flush_ms) * std.time.ns_per_ms;
        h.tier.defer_ns = @as(i96, sp.defer_ms) * std.time.ns_per_ms;
        h.tele = if (h.parking()) .{} else null;
        if (h.tele != null) if (sp.highwater) |hw| info("--spill-highwater {d:.2} is the v1 tier's early-write mark: the v2 tier writes each state when it is kept and ignores it", .{hw});
    }

    pub fn start(h: *LaneHost) !void {
        h.core.fill_hook = .{ .ptr = h, .done = filled };
        h.thread = try std.Thread.spawn(.{ .stack_size = 16 << 20 }, run, .{h});
    }

    /// Stops admitting, cancels what is left and joins the engine thread.
    pub fn stop(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
        h.closing = true;
        h.wake.broadcast(h.io);
        h.mutex.unlock(h.io);
        if (h.thread) |t| t.join();
        h.thread = null;
        const closed: api.LearnEvent = .{ .done = .{ .message = "the engine closed" } };
        for (h.lessons.items) |l| l.sink.event(l.sink.ctx, &closed);
        h.lessons.deinit(h.gpa);
        h.queued.deinit(h.gpa);
        h.admitted.deinit(h.gpa);
        h.filling.deinit(h.gpa);
        h.cancels.deinit(h.gpa);
        h.decoded.deinit(h.gpa);
        h.live_tokens.deinit(h.gpa);
        h.ready.deinit(h.gpa);
        h.loading.deinit(h.gpa);
        h.deferred.deinit(h.gpa);
    }

    pub fn engine(h: *LaneHost) Engine {
        return .{ .ctx = h, .vtable = &.{ .info = infoFn, .submit = submitFn, .cancel = cancelFn, .status = status.read, .memory = memoryFn, .keepalive = keepaliveFn, .learn = learn.submit } };
    }

    /// The family's queue as a keepalive target, when it set one.
    fn keepaliveFn(ctx: *anyopaque) ?api.keepalive.Target {
        return self(ctx).keepalive_target;
    }

    fn self(ctx: *anyopaque) *LaneHost {
        return @ptrCast(@alignCast(ctx));
    }

    fn infoFn(ctx: *anyopaque) Info {
        return self(ctx).info_;
    }

    fn submitFn(ctx: *anyopaque, id: Id, request: *const Request, sink: Sink) SubmitError!void {
        const h = self(ctx);
        const job = h.gpa.create(Job) catch return error.Busy;
        job.* = .{ .host = h, .id = id, .request = request, .sink = sink };
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        if (h.closing) {
            h.gpa.destroy(job);
            return error.Closed;
        }
        // foreground before background, each in arrival order (the Python job queue's priority)
        var at = h.queued.items.len;
        if (!request.background) {
            while (at > 0 and h.queued.items[at - 1].request.background) at -= 1;
        }
        h.queued.insert(h.gpa, at, job) catch {
            h.gpa.destroy(job);
            return error.Busy;
        };
        h.wake.signal(h.io);
    }

    fn cancelFn(ctx: *anyopaque, id: Id) void {
        const h = self(ctx);
        h.mutex.lockUncancelable(h.io);
        defer h.mutex.unlock(h.io);
        h.cancels.append(h.gpa, id) catch {};
        h.wake.signal(h.io);
    }

    fn memoryFn(ctx: *anyopaque, reset_peak: bool) ?Memory {
        const source = self(ctx).memory orelse return null;
        return source.read(source.ctx, reset_peak);
    }

    fn emit(job: *Job, event: Event) void {
        job.sink.event(job.sink.ctx, job.id, &event);
    }

    /// A job's cancel hook for its prompt pass: its id is in `cancels`, or the host is closing (read under the lock).
    fn cancelled(ctx: *anyopaque) bool {
        const job: *Job = @ptrCast(@alignCast(ctx));
        job.host.lock();
        defer job.host.unlock();
        return job.host.closing or std.mem.indexOfScalar(Id, job.host.cancels.items, job.id) != null;
    }

    /// A background pass's yield hook: a foreground request waits or a stream decodes (read under the lock).
    fn yieldsTo(ctx: *anyopaque) bool {
        const job: *Job = @ptrCast(@alignCast(ctx));
        job.host.lock();
        defer job.host.unlock();
        return job.host.foregroundWaiting() or job.host.core.activeCount() > 0;
    }

    /// A foreground request is queued (they queue ahead of every background one); the caller holds the lock.
    fn foregroundWaiting(h: *const LaneHost) bool {
        return h.queued.items.len > 0 and !h.queued.items[0].request.background;
    }

    /// Requests parked or woken (the lock held).
    pub fn waitingCount(h: *const LaneHost) usize {
        return h.ready.items.len + h.loading.items.len + h.deferred.items.len;
    }

    /// Hands a stream the tokens its rounds committed since the last delivery (their logprob rows first).
    fn send(h: *LaneHost, job: *Job) void {
        const emitted = job.stream.emitted();
        if (emitted.len > job.delivered) {
            if (job.stream.logprobs != null) emit(job, .{ .logprobs = job.stream.rows.items[job.delivered..emitted.len] });
            emit(job, .{ .tokens = emitted[job.delivered..] });
            status.noteDecoded(h, emitted.len - job.delivered);
            job.delivered = emitted.len;
        }
    }

    /// Sends a stream's new tokens; true once it has finished (its job freed).
    fn deliver(h: *LaneHost, job: *Job) bool {
        h.send(job);
        if (!job.stream.finished) return false;
        const reason: Reason = switch (job.stream.reason) {
            .length => .length,
            .cancelled => .cancelled,
            .@"error" => .failed,
            else => .stop,
        };
        h.finish(job, reason, job.stream.problem);
        return true;
    }

    fn finish(h: *LaneHost, job: *Job, reason: Reason, message: []const u8) void {
        job.reported();
        h.gpa.free(job.marks);
        h.gpa.free(job.lineage);
        const s = &job.stream;
        const stats: Stats = if (job.started) .{ .rounds = s.rounds, .drafted = s.drafted, .accepted = s.accepted, .min_rows = s.min_rows, .loop_period = s.loop_period, .prefill_seconds = if (job.prefilled) |done| @as(f64, @floatFromInt(@as(i64, @intCast(@max(0, done - job.began))))) / 1e9 else null } else .{};
        emit(job, .{ .finished = .{ .reason = reason, .stats = stats, .message = message } });
        if (job.started) {
            if (s.grammar) |g| g.deinit(h.gpa);
            s.deinit(h.gpa);
            job.proposer.deinit();
        }
        h.gpa.destroy(job);
    }

    /// Cancels queued and admitted jobs named since the last round.
    fn takeCancels(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
        const ids = h.gpa.dupe(Id, h.cancels.items) catch &.{};
        h.cancels.clearRetainingCapacity();
        var dropped: std.ArrayList(*Job) = .empty;
        var unloaded: std.ArrayList(*pc.Restore) = .empty; // restores a cancelled job waited for
        for (ids) |id| {
            for (h.queued.items, 0..) |job, i| if (job.id == id) {
                dropped.append(h.gpa, h.queued.orderedRemove(i)) catch {};
                break;
            };
            for (h.ready.items, 0..) |job, i| if (job.id == id) {
                dropped.append(h.gpa, h.ready.orderedRemove(i)) catch {};
                break;
            };
            for (h.loading.items, 0..) |l, i| if (l.job.id == id) {
                _ = h.loading.orderedRemove(i);
                dropped.append(h.gpa, l.job) catch {};
                unloaded.append(h.gpa, l.restore) catch {};
                break;
            };
            for (h.deferred.items, 0..) |d, i| if (d.job.id == id) {
                dropped.append(h.gpa, h.deferred.orderedRemove(i).job) catch {};
                break;
            };
        }
        h.mutex.unlock(h.io);
        // a restore no request waits for any more stops (its reads, DMA and reservations go), exactly once
        for (unloaded.items, 0..) |r, k| {
            if (std.mem.indexOfScalar(*pc.Restore, unloaded.items[0..k], r) != null) continue;
            if (!h.wanted(r)) if (h.cache) |store| store.restoreCancel(r);
        }
        unloaded.deinit(h.gpa);
        for (dropped.items) |job| h.finish(job, .cancelled, "");
        dropped.deinit(h.gpa);
        for (ids) |id| {
            for (h.admitted.items, 0..) |job, i| if (job.id == id) {
                h.unfill(job);
                if (!job.stream.finished) h.core.discard(&job.stream);
                h.lock();
                _ = h.admitted.orderedRemove(i);
                h.unlock();
                h.finish(job, .cancelled, "");
                break;
            };
        }
        h.gpa.free(ids);
    }

    fn lock(h: *LaneHost) void {
        h.mutex.lockUncancelable(h.io);
    }

    fn unlock(h: *LaneHost) void {
        h.mutex.unlock(h.io);
    }

    const Prep = union(enum) { none, dropped, parked, job: *Job };

    /// Takes the next woken or queued request into a free lane and builds its stream, not yet prefilled: `none` when
    /// none waits or no lane is free, `dropped` when it ended here, `parked` when it waits for a restore (no lane).
    fn admitPrepare(h: *LaneHost) Prep {
        h.lock();
        if ((h.queued.items.len == 0 and h.ready.items.len == 0) or h.admitted.items.len >= h.info_.lanes) {
            h.unlock();
            return .none;
        }
        const from = if (h.ready.items.len > 0) &h.ready else &h.queued; // woken requests first
        if (h.info_.admit) |a| { // a shared cache pool: the next request waits for room (always there when idle)
            const r = from.items[0].request;
            if (h.core.activeCount() > 0 and !a.fits(a.ctx, @intCast(r.prompt.len), r.max_tokens)) {
                h.unlock();
                return .none;
            }
        }
        const job = from.orderedRemove(0);
        if (h.parking()) {
            h.unlock();
            if (h.park(job)) return .parked; // waits for a restore, holding no lane
            h.lock();
        }
        h.admitted.append(h.gpa, job) catch {
            h.unlock();
            h.finish(job, .failed, "out of memory");
            return .dropped;
        };
        h.unlock();
        const r = job.request;
        var reuse: lanes.stream.Reuse = .{};
        // the entry stays alive until the backend restores it: nothing keeps between here and this stream's own pass
        // a prompt with pictures or clips is neither resumed nor kept
        if (r.media == null) if (h.cache) |store| if (store.lookupRewind(h.gpa, r.prompt, r.history_len, r.rewind_len, r.shared_prefixes, r.chunks, h.spans(r))) |l| {
            job.entry = l.entry;
            job.kept0 = store.counts.kept;
            job.marks = l.marks;
            reuse = .{ .saved = if (l.entry) |e| e.saved else null, .at = if (l.entry) |e| e.at else 0, .marks = l.marks, .hook = .{ .ptr = job, .at = Job.kept } };
            job.join = store.join(r.prompt, l.entry, r.drafts);
            if (l.entry) |e| {
                job.parent = e.point;
                job.fork = e.at;
            }
            // its rows' computation: the resumed state's forked at its position (written or not), else its own
            job.lineage = store.passLineage(l.entry) catch &.{};
        } else |_| {};
        job.proposer = lanes.SuffixLookup.init(h.gpa, .{ .min_match = h.min_match }) catch return h.droppedJob(job, "the drafter could not start");
        job.stream = lanes.Stream.init(h.gpa, .{
            .id = "request",
            .prompt = r.prompt,
            .max_new = r.max_tokens,
            .eos = r.eos,
            .sampling = r.sampling,
            .drafts = r.drafts,
            .proposer = job.proposer.proposer(),
            .stop_check = if (r.stop) |s| .{ .ptr = s.ctx, .check = s.check } else null,
            .cancel_check = .{ .ptr = job, .check = cancelled },
            .yield_check = if (r.background and r.max_tokens == 0) .{ .ptr = job, .check = yieldsTo } else null,
            .think_budget = r.think_budget,
            .think_close = r.think_close,
            .think_end = if (r.think_end) |t| t else -1,
            .logprobs = r.logprobs,
            .call = if (r.call) |c| .{ .opener = c.opener, .think_open = if (c.think_open) |t| t else -1, .think_end = if (c.think_end) |t| t else -1, .blank = c.blank } else null,
            .loop_guard = r.loop_guard,
            .loop_rule = if (r.loop_recipe) .recipe else .mac,
            .chunks = r.chunks,
            .history_len = r.history_len,
            .shared_prefixes = r.shared_prefixes,
            .reuse = reuse,
            .grammar = if (r.structure) |st| h.constraint(st) catch |e| {
                job.proposer.deinit();
                return h.droppedJob(job, if (e == error.OutOfMemory) "out of memory" else "the reply's grammar could not start (xgrammar)");
            } else null,
            .decode_spans = h.spans(r),
            .decide = r.decide,
            .media = r.media,
        }) catch {
            job.proposer.deinit();
            return h.droppedJob(job, "out of memory");
        };
        job.started = true;
        job.began = std.Io.Clock.awake.now(h.io).toNanoseconds();
        return .{ .job = job };
    }

    fn droppedJob(h: *LaneHost, job: *Job, message: []const u8) Prep {
        _ = h.drop(job, message);
        return .dropped;
    }

    /// Prefills the next queued request into a free lane; false when none waits or no lane is free.
    fn admitOne(h: *LaneHost) bool {
        switch (h.admitPrepare()) {
            .none => return false,
            .dropped, .parked => return true,
            .job => |job| return h.startJob(job),
        }
    }

    /// Every request waiting for a lane, admitted now: prompts that arrive together (a burst, with no resume or kept
    /// state to take) prefill in one backend call, where the backend can (one pass over their rows). A request parked
    /// on a restore takes no lane and joins no burst. While requests are parked, an admission runs one whole pass: the
    /// rest wait for the run loop's service, so no restore waits behind a run of passes. A burst counts as passes too
    /// (a whole-pass backend may run its prompts one after another: Mia's GLM engine does, its `manyFit` never holds),
    /// so then it takes no second prompt.
    fn admitAll(h: *LaneHost) void {
        var burst: [16]*Job = undefined;
        var n: usize = 0;
        var ran = false; // a prompt pass ran whole on this thread (nothing is serviced during one)
        while (!(ran and h.loading.items.len > 0)) switch (h.admitPrepare()) {
            .none => break,
            .dropped, .parked => {},
            .job => |job| {
                const r = job.request;
                if (r.max_tokens == 0 or h.loneFits(job)) {
                    ran = ran or r.max_tokens > 0 or h.info_.warm_turns or h.wholePasses(); // (a lone run, a warm pass)
                    h.timed(startJob, job);
                } else if (n < burst.len and h.core.batches() and job.stream.reuse.at == 0 and
                    job.stream.reuse.marks.len == 0 and r.media == null and r.logprobs == null and r.structure == null)
                {
                    burst[n] = job;
                    n += 1;
                    ran = ran or h.wholePasses();
                } else {
                    ran = ran or h.wholePasses();
                    h.timed(startJob, job);
                }
            },
        };
        h.timed(startBurst, burst[0..n]);
    }

    /// The backend prefills a prompt in one call on this thread (GLM's addStream), not a chunk or a slice a turn.
    fn wholePasses(h: *const LaneHost) bool {
        return !h.core.fills() and !h.core.slices();
    }

    /// An admission's prompt pass (`pass`: startJob or startBurst). Under a tier its engine-thread time is counted at
    /// once (passes_ns: nothing is serviced during it), so a request parked or deferred later in the same admission
    /// leaves it out too.
    fn timed(h: *LaneHost, comptime pass: anytype, arg: anytype) void {
        const t0 = if (h.parking()) std.Io.Clock.awake.now(h.io).toNanoseconds() else null;
        _ = pass(h, arg);
        if (t0) |t| h.passes_ns += @intCast(@max(0, std.Io.Clock.awake.now(h.io).toNanoseconds() - t));
    }

    /// The jobs' prompts in one backend call, then each joined and delivered in admission order; a refused burst
    /// prefills one at a time.
    fn startBurst(h: *LaneHost, jobs: []*Job) void {
        if (jobs.len == 0) return;
        if (jobs.len == 1) {
            _ = h.startJob(jobs[0]);
            return;
        }
        var ss: [16]*lanes.Stream = undefined;
        const began = std.Io.Clock.awake.now(h.io).toNanoseconds();
        for (jobs, ss[0..jobs.len]) |job, *s| {
            job.began = began;
            s.* = &job.stream;
        }
        h.core.prefillBurst(ss[0..jobs.len]) catch |e| {
            for (jobs) |job| {
                if (e == error.BurstRefused) _ = h.startJob(job) else if (e == error.Cancelled) _ = h.cancel(job) else _ = h.drop(job, @errorName(e));
            }
            return;
        };
        for (jobs) |job| {
            h.core.addPrefilled(&job.stream) catch |e| {
                _ = switch (e) {
                    error.Cancelled => h.cancel(job),
                    else => h.drop(job, h.words(e)),
                };
                continue;
            };
            if (job.stream.reprefill) h.continueGate(job) catch |e| {
                _ = h.drop(job, h.words(e));
                continue;
            };
            h.prefilled(job, began);
            if (h.deliver(job)) h.remove(job);
        }
    }

    /// A call-gate fix is in the reply. Send it, then prefill the prompt plus that reply so the next token
    /// matches Python's `generate_gated` (a new prefill, not a decode of the replaced token).
    fn continueGate(h: *LaneHost, job: *Job) !void {
        var n: u8 = 0;
        while (job.stream.reprefill) {
            n += 1;
            if (n > 4) return error.GateRestart;
            h.send(job);
            job.stream.reuse = .{};
            job.stream.cached = 0;
            job.stream.reuse_failed = false;
            h.core.release(&job.stream);
            try job.stream.absorbIntoPrompt();
            job.delivered = 0;
            if (job.stream.max_new == 0) {
                job.stream.finished = true;
                job.stream.reason = .length;
                return;
            }
            try h.core.addStream(&job.stream);
        }
    }

    /// One prepared job: its prompt prefilled alone (or its fill started), delivered.
    fn startJob(h: *LaneHost, job: *Job) bool {
        const r = job.request;
        const began = job.began;
        if (r.max_tokens == 0 and h.info_.warm_turns) return h.warmPass(job, began);
        if (h.loneFits(job)) return h.runLone(job, began);
        if (h.core.fills()) {
            h.filling.append(h.gpa, job) catch return h.drop(job, "out of memory");
            return true;
        }
        // a backend that fills prompts between rounds starts this one and admits the next (prompts that arrive
        // together fill together); the fill hook reports it in
        if (h.core.slices()) {
            const filling = h.core.beginStream(&job.stream) catch |e| return switch (e) {
                error.Cancelled => h.cancel(job),
                error.LaneMemory => h.requeue(job),
                else => h.drop(job, h.words(e)),
            };
            if (filling) return true;
            h.prefilled(job, began);
            if (h.deliver(job)) h.remove(job);
            return true;
        }
        h.core.addStream(&job.stream) catch |e| return switch (e) {
            error.Cancelled => h.cancel(job),
            error.LaneMemory => h.requeue(job),
            else => h.drop(job, h.words(e)),
        };
        if (job.stream.reprefill) h.continueGate(job) catch |e| return h.drop(job, h.words(e));
        h.prefilled(job, began);
        if (h.deliver(job)) h.remove(job);
        return true;
    }

    /// The oldest filling job's next prompt chunk: the other streams' rounds run between chunks, not after the prompt.
    fn fillOne(h: *LaneHost) void {
        if (h.filling.items.len == 0) return;
        const job = h.filling.items[0];
        const first = !job.fill_began;
        job.fill_began = true;
        const done = h.core.fillStream(&job.stream, first) catch |e| {
            _ = switch (e) {
                error.Cancelled => h.cancel(job),
                error.LaneMemory => blk: {
                    h.unfill(job);
                    break :blk h.requeue(job);
                },
                else => h.drop(job, h.words(e)),
            };
            return;
        };
        if (!done) return;
        _ = h.filling.orderedRemove(0);
        if (job.stream.reprefill) h.continueGate(job) catch |e| {
            _ = h.drop(job, h.words(e));
            return;
        };
        h.prefilled(job, job.began);
        if (h.deliver(job)) h.remove(job);
    }

    fn unfill(h: *LaneHost, job: *Job) void {
        for (h.filling.items, 0..) |j, i| if (j == job) {
            _ = h.filling.orderedRemove(i);
            return;
        };
    }

    /// A reply's grammar from the server's compiler (Info.structures engines get requests that carry one).
    fn constraint(h: *LaneHost, st: api.Structure) !*lanes.grammar.Constraint {
        const c = st.compiler orelse return error.NoGrammarCompiler;
        return c.constraint(h.gpa, h.io, @fromBackingInt(@intCast(@backingInt(st.kind))), st.text, st.after);
    }

    fn prefilled(h: *LaneHost, job: *Job, began: i96) void {
        const done = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.lock();
        if (done > began) h.prefill_rate = @as(f64, @floatFromInt(job.request.prompt.len - job.stream.cached)) / (@as(f64, @floatFromInt(done - began)) / 1e9);
        h.prefill_at = done;
        job.prefilled = done;
        h.unlock();
        job.reported();
        h.host_counts.cached_tokens += job.stream.cached;
        h.host_counts.computed_tokens += job.request.prompt.len -| job.stream.cached;
        if (h.cache) |store| {
            store.continued(job.join, job.stream.cached, job.request.prompt.len);
            store.report(job.request.prompt.len, job.stream.cached, store.counts.kept - job.kept0);
        }
        status.noteKept(h);
        emit(job, .{ .prefilled = job.stream.cached });
    }

    /// A request's decoded prompt rows, for a backend that keeps their bits (`decoded_rows`); otherwise none.
    fn spans(h: *const LaneHost, r: *const Request) []const [2]u32 {
        return if (h.decoded_rows) r.decode_spans else &.{};
    }

    /// A prompt-only pass on a `warm_turns` engine: the backend prefills and the cache keeps where it ends.
    fn warmPass(h: *LaneHost, job: *Job, began: i96) bool {
        const be = h.core.backend;
        be.prefill(&job.stream) catch |e| { // a yield ends it like a cancel: what it prefilled is kept
            if (e != error.Cancelled and e != error.Yielded) return h.drop(job, h.words(e));
            be.release(&job.stream);
            return h.cancel(job);
        };
        h.prefilled(job, began);
        if (h.cache) |store| _ = store.keepWith(job.request.prompt, @intCast(job.request.prompt.len), &job.stream, job.request.chunks, h.spans(job.request), job.keepInfo());
        status.noteKept(h);
        be.release(&job.stream);
        h.remove(job);
        h.finish(job, .length, "");
        return true;
    }

    /// An idle backend driver takes a lone drafted request, sampled when supported; never one with logprobs.
    fn loneFits(h: *LaneHost, job: *Job) bool {
        const r = job.request;
        const lone = h.lone orelse return false;
        if ((r.sampling != null and !lone.sampled) or !r.drafts or r.think_budget > 0 or r.loop_guard or r.call != null or r.structure != null or r.logprobs != null or r.media != null) return false;
        h.lock();
        defer h.unlock();
        return h.admitted.items.len == 1 and !h.foregroundWaiting() and h.cancels.items.len == 0 and h.core.activeCount() == 0 and h.core.fillingCount() == 0 and h.waitingCount() == 0;
    }

    /// Send lone-driver tokens as they land; arrivals and cancellation return its stream to the lane core.
    fn runLone(h: *LaneHost, job: *Job, began: i96) bool {
        h.lone_job = job;
        job.delivered = 0;
        const lone = h.lone.?;
        const Hooks = struct {
            fn committed(ctx: *anyopaque) void {
                const host: *LaneHost = @ptrCast(@alignCast(ctx));
                const j = host.lone_job.?;
                if (!j.prefill_sent) {
                    j.prefill_sent = true;
                    host.prefilled(j, j.began);
                }
                host.send(j);
                status.noteLive(host);
                // writes enqueued at keep are staged during a long lone decode too (parked work makes it yield)
                host.lone_commits += 1;
                if (host.lone_commits >= lone_pump_commits and host.parking()) {
                    host.lone_commits = 0;
                    host.cache.?.pump(host.tier.pump_ns);
                }
            }
            fn yield(ctx: *anyopaque) bool {
                const host: *LaneHost = @ptrCast(@alignCast(ctx));
                host.lock();
                defer host.unlock();
                return host.foregroundWaiting() or host.cancels.items.len > 0 or host.closing or host.waitingCount() > 0;
            }
        };
        job.began = began;
        const paused = lone.run(lone.ctx, &job.stream, .{ .ctx = h, .committed = Hooks.committed, .yield = Hooks.yield });
        h.lone_job = null;
        const handed = paused catch |e| {
            if (!job.prefill_sent) emit(job, .{ .prefilled = 0 });
            h.remove(job);
            h.finish(job, if (e == error.Cancelled) .cancelled else .failed, if (e == error.Cancelled) "" else h.words(e));
            return true;
        };
        if (!job.prefill_sent) h.prefilled(job, began);
        if (handed) {
            h.core.adopt(&job.stream) catch |e| return h.drop(job, @errorName(e));
            h.send(job);
            return true;
        }
        if (h.deliver(job)) h.remove(job);
        return true;
    }

    pub fn words(h: *const LaneHost, e: anyerror) []const u8 {
        const x = h.explain orelse return @errorName(e);
        return x.text(x.ctx, e) orelse @errorName(e);
    }

    fn drop(h: *LaneHost, job: *Job, message: []const u8) bool {
        h.remove(job);
        if (job.started and !job.stream.finished) h.core.discard(&job.stream);
        h.finish(job, .failed, message);
        return true;
    }

    /// No memory for another lane yet: the job waits at the queue's head, and admission holds until a lane frees.
    fn requeue(h: *LaneHost, job: *Job) bool {
        job.stream.deinit(h.gpa);
        job.proposer.deinit();
        job.started = false;
        job.fill_began = false;
        job.entry = null; // looked up again at its next admission
        job.join = .{};
        job.parent = null;
        job.fork = 0;
        h.gpa.free(job.lineage);
        job.lineage = &.{};
        h.gpa.free(job.marks);
        job.marks = &.{};
        h.lock();
        defer h.unlock();
        for (h.admitted.items, 0..) |j, i| if (j == job) {
            _ = h.admitted.orderedRemove(i);
            break;
        };
        h.queued.insert(h.gpa, 0, job) catch {
            h.unlock();
            h.finish(job, .failed, "out of memory");
            h.lock();
            return true;
        };
        if (h.admitted.items.len > 0) h.info_.lanes = @intCast(h.admitted.items.len);
        return false;
    }

    /// A job cancelled in its prompt pass, its lane already released.
    fn cancel(h: *LaneHost, job: *Job) bool {
        h.remove(job);
        h.finish(job, .cancelled, "");
        return true;
    }

    fn remove(h: *LaneHost, job: *Job) void {
        h.unfill(job);
        h.lock();
        defer h.unlock();
        for (h.admitted.items, 0..) |j, i| if (j == job) {
            _ = h.admitted.orderedRemove(i);
            return;
        };
    }

    fn run(h: *LaneHost) void {
        while (true) {
            h.takeCancels();
            if (h.parking()) h.unhold();
            h.admitAll(); // (its prompt passes are timed: nothing is serviced meanwhile, so restore waits leave them out)
            if (h.parking() and h.entryHeld()) h.held_from = std.Io.Clock.awake.now(h.io).toNanoseconds(); // (no restore begins or progresses: waits leave it out too)
            h.fillOne();
            if (h.parking()) h.service(h.tier.service_ns); // restores, staged writes and deferred retries: between rounds and chunks, and at idle
            status.noteLive(h);
            h.lock();
            if (h.closing) {
                const left = h.queued.items.len + h.admitted.items.len + h.waitingCount();
                h.unlock();
                if (h.lesson_open) {
                    h.learner.?.abort(h.learner.?.ctx);
                    h.lesson_open = false;
                }
                if (left == 0) return h.shutdownTier();
                h.closeAll();
                continue;
            }
            if (h.core.activeCount() == 0 and h.core.fillingCount() == 0) {
                if (h.filling.items.len > 0) {
                    h.unlock();
                    continue;
                }
                if (learn.ready(h)) {
                    h.unlock();
                    learn.step(h);
                    continue;
                }
                if (h.cancels.items.len == 0 and ((h.queued.items.len == 0 and h.ready.items.len == 0) or h.admitted.items.len >= h.info_.lanes)) {
                    const ms: i64 = if (h.pending()) poll_ns / std.time.ns_per_ms else 100; // work in flight: poll it
                    h.wake.waitTimeout(h.io, &h.mutex, .{ .duration = .{ .raw = .fromMilliseconds(ms), .clock = .awake } }) catch {};
                }
                h.unlock();
                continue;
            }
            h.unlock();
            h.core.step() catch |e| {
                h.failAll(@errorName(e));
                continue;
            };
            var i: usize = 0;
            while (i < h.admitted.items.len) {
                const job = h.admitted.items[i];
                if (job.stream.reprefill) h.continueGate(job) catch |e| {
                    _ = h.drop(job, h.words(e));
                    continue;
                };
                if (h.deliver(job)) {
                    h.lock();
                    _ = h.admitted.orderedRemove(i);
                    h.unlock();
                } else i += 1;
            }
        }
    }

    /// A failed round ends every stream it held, with the backend's error.
    fn failAll(h: *LaneHost, message: []const u8) void {
        h.lock();
        const jobs = h.gpa.dupe(*Job, h.admitted.items) catch &.{};
        h.admitted.clearRetainingCapacity();
        h.filling.clearRetainingCapacity();
        h.unlock();
        for (jobs) |job| {
            if (!job.stream.finished) h.core.discard(&job.stream);
            h.finish(job, .failed, message);
        }
        h.gpa.free(jobs);
    }

    fn closeAll(h: *LaneHost) void {
        h.lock();
        for (h.queued.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        for (h.admitted.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        for (h.ready.items) |job| h.cancels.append(h.gpa, job.id) catch {};
        for (h.loading.items) |l| h.cancels.append(h.gpa, l.job.id) catch {};
        for (h.deferred.items) |d| h.cancels.append(h.gpa, d.job.id) catch {};
        h.unlock();
        h.takeCancels();
    }

    // -- parked work (spill v2) -----------------------------------------------------------------------------------

    /// Whether admission may park a request: the cache has a durable tier.
    fn parking(h: *const LaneHost) bool {
        const store = h.cache orelse return false;
        return store.tier != null;
    }

    /// Work in flight the host must poll (engine thread): requests parked or woken, restores, writes to stage.
    fn pending(h: *const LaneHost) bool {
        if (h.waitingCount() > 0) return true;
        const store = h.cache orelse return false;
        return store.busy();
    }

    /// A parked request still waits for restore `r`.
    fn wanted(h: *const LaneHost, r: *pc.Restore) bool {
        for (h.loading.items) |l| if (l.restore == r) return true;
        return false;
    }

    /// An admitted request whose pass has not restored its looked-up state yet (a fill that has not reached it): a
    /// restore's room, begun or adopted, may evict any kept state, so restores are neither begun nor polled until it
    /// has (park, service).
    fn entryHeld(h: *const LaneHost) bool {
        for (h.admitted.items) |job| if (job.entry != null) return true;
        return false;
    }

    /// The turns since the last admission in which a pass held a looked-up entry (held_from), counted as passes are:
    /// no restore could begin or progress.
    fn unhold(h: *LaneHost) void {
        const t = h.held_from orelse return;
        h.held_from = null;
        h.passes_ns += @intCast(@max(0, std.Io.Clock.awake.now(h.io).toNanoseconds() - t));
    }

    /// The time since `since` that parked work could be serviced: the wall time less the prompt passes admissions ran
    /// meanwhile (`passes`: passes_ns at `since`) and the turns a pass held a looked-up entry, when the engine thread
    /// serviced nothing.
    fn serviceable(h: *const LaneHost, now: i96, since: i96, passes: u64) i96 {
        const held: i96 = if (h.held_from) |t| @max(0, now - @max(t, since)) else 0;
        return now - since - @as(i96, h.passes_ns - passes) - held;
    }

    /// Before `job`'s pass (engine thread, no lock): parked on a restore of its own or deferred (parkOwn), else held
    /// while others are parked if its pass would be long (holdPass). False: admit it now.
    fn park(h: *LaneHost, job: *Job) bool {
        return h.parkOwn(job) or h.holdPass(job);
    }

    /// Park `job` on a restore of a longer durable point (once), or defer it when the tier refuses the restore in a way
    /// waiting may change (Store.deferrable), or while another admitted pass holds a looked-up entry the restore's room
    /// could evict. False: no restore of its own now.
    fn parkOwn(h: *LaneHost, job: *Job) bool {
        const store = h.cache orelse return false;
        const r = job.request;
        if (job.tried_disk or r.media != null) return false;
        const rows = h.spans(r);
        const probe = store.probe(r.prompt, r.chunks, rows);
        const p = probe.durable orelse return false;
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        if (h.entryHeld()) return h.holdOff(job, now);
        job.tried_disk = true;
        h.lock();
        const room = h.loading.ensureUnusedCapacity(h.gpa, 1);
        h.unlock();
        room catch return false;
        // (the best memory state stays while the budget holds it beside the restore: refused, the pass resumes it; a
        // full tier, or a point the tier's check refuses, is refused before any room is made)
        const begun = store.restoreBeginOver(p, r.prompt, .{ .have = probe.memory, .again = job.deferred_since != null });
        status.noteKept(h); // (its room may have evicted kept states, begun or not)
        if (begun) |rs| {
            h.lock();
            h.loading.appendAssumeCapacity(.{ .job = job, .restore = rs, .since = now, .passes = h.passes_ns });
            h.unlock();
            h.host_counts.restore_waits += 1;
            job.deferred_since = null;
            job.waited = true;
            return true;
        } else |err| {
            const left = store.find(r.prompt, r.chunks, rows); // (the room may have evicted probe.memory: never read it now)
            if (h.deferRestore(job, p, left, err, now)) return true;
            if (left) |e| {
                note("reading {d} kept tokens from disk did not start ({s}); resuming the {d} memory holds", .{ p.at, @errorName(err), e.at });
            } else note("reading {d} kept tokens from disk did not start ({s}); prefilling", .{ p.at, @errorName(err) });
        }
        return false;
    }

    /// `job` has no restore of its own to wait for while other requests are parked on theirs. A backend's whole prompt
    /// pass is synchronous on this thread (Mia's GLM engine: addStream or a burst, no hook between its chunks), so a
    /// restore is polled and installed only between passes; a long pass now would hold every parked request, whose
    /// restores need a few services, for all of it (g2: 18.8 s for a 38K-token prefill). When its pass would compute at
    /// least hold_min tokens past its best memory state, `job` waits on the deferred list (no lane, counted waiting)
    /// and asks again when a restore ends or every defer_retry_ns, at most defer_ns of the time parked work could be
    /// serviced (passes run meanwhile left out, as a deferral's); true then. False: admit it now; always for a job that
    /// already waited on restores (`waited`: its own installed, failed or timed out, or a deferral gave up), so no
    /// request waits twice.
    fn holdPass(h: *LaneHost, job: *Job) bool {
        const store = h.cache orelse return false;
        if (job.waited or h.loading.items.len == 0 or h.tier.hold_min == 0 or h.tier.defer_ns <= 0 or !h.wholePasses()) return false;
        const r = job.request;
        const have: u32 = if (r.media != null) 0 else if (store.find(r.prompt, r.chunks, h.spans(r))) |e| e.at else 0;
        if (r.prompt.len -| have < h.tier.hold_min) return false;
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        const first = job.held_since orelse now;
        const passes = if (job.held_since != null) job.held_passes else h.passes_ns;
        if (h.serviceable(now, first, passes) >= h.tier.defer_ns) {
            h.host_counts.prefills_held_expired += 1;
            note("a {d}-token prompt pass waited {d} ms for others' disk restores; it runs now", .{ r.prompt.len -| have, @divTrunc(now - first, std.time.ns_per_ms) });
            return false;
        }
        h.lock();
        defer h.unlock();
        h.deferred.append(h.gpa, .{ .job = job, .retry_at = now + h.tier.defer_retry_ns }) catch return false;
        if (job.held_since == null) {
            h.host_counts.prefills_held += 1;
            info("a {d}-token prompt pass waits for {d} parked disk restores ({d} ms at most)", .{ r.prompt.len -| have, h.loading.items.len, @divTrunc(h.tier.defer_ns, std.time.ns_per_ms) });
        }
        job.held_since = first;
        job.held_passes = passes;
        job.tried_disk = true; // (it asked the tier once: a retry looks up memory only)
        return true;
    }

    /// A restore of `p` the tier or the budget refused with `err`: when asking again may help (Store.deferrable:
    /// capacity, or anything but the point while others are in flight) and the point saves at least defer_min tokens
    /// over `have` (the best memory state), `job` waits on the deferred list (holding no lane) and asks again when a
    /// restore in flight ends, defer_ns at most from its first refusal; true then. A capacity refusal (the tier did
    /// nothing) is also asked again every defer_retry_ns; any other (the tier may have begun reads and asked every rank
    /// first, as GLM's reserves its destination last) only when a restore ends, none is left in flight, or at the bound.
    /// False: admit it now on what memory holds (a deferred request that gives up, at the bound or refused again with
    /// nothing left to wait for, is counted expired; it waited: holdPass never holds it after).
    fn deferRestore(h: *LaneHost, job: *Job, p: pc.Point, have: ?*pc.Entry, err: anyerror, now: i96) bool {
        const store = h.cache orelse return false;
        if (h.tier.defer_ns <= 0 or !store.deferrable(err)) {
            if (job.deferred_since != null) h.host_counts.restores_deferred_expired += 1;
            return false;
        }
        const at: u32 = if (have) |e| e.at else 0;
        const min = if (h.tier.defer_min > 0) h.tier.defer_min else @max(store.rules.grid, 1);
        if (p.at <= at or p.at - at < min) return false;
        const first = job.deferred_since orelse now;
        const passes = if (job.deferred_since != null) job.deferred_passes else h.passes_ns;
        if (h.serviceable(now, first, passes) >= h.tier.defer_ns) {
            h.host_counts.restores_deferred_expired += 1;
            note("reading {d} kept tokens from disk was refused for {d} ms ({s}); admitted on what memory holds", .{ p.at, @divTrunc(now - first, std.time.ns_per_ms), @errorName(err) });
            return false;
        }
        const capacity = err == error.NoRoom or std.mem.indexOfScalar(anyerror, &pc.capacity_errors, err) != null;
        const left = h.tier.defer_ns - h.serviceable(now, first, passes); // (the bound, if the restores outlast it)
        h.lock();
        defer h.unlock();
        const retry = if (capacity) h.tier.defer_retry_ns else @max(h.tier.defer_retry_ns, left);
        h.deferred.append(h.gpa, .{ .job = job, .retry_at = now + retry, .until_end = !capacity }) catch return false;
        if (job.deferred_since == null) h.host_counts.restores_deferred += 1;
        job.deferred_since = first;
        job.deferred_passes = passes;
        job.tried_disk = false; // (it asks for the restore again when it wakes)
        job.waited = true;
        return true;
    }

    /// `job` would park while an admitted pass holds a looked-up entry (entryHeld): it waits on the deferred list,
    /// holding no lane, and asks again at its next retry. No capacity refusal: nothing is counted and its deferral
    /// bound, if one runs, leaves the hold out (serviceable). False when it cannot wait: admitted now on memory.
    fn holdOff(h: *LaneHost, job: *Job, now: i96) bool {
        h.lock();
        defer h.unlock();
        h.deferred.append(h.gpa, .{ .job = job, .retry_at = now + h.tier.defer_retry_ns }) catch return false;
        return true;
    }

    /// Deferred restores whose retry time came, or every one when a restore in flight ended (`ended`: its read slot
    /// and charge are free now), or one waiting for the restores in flight when none is left (cancelled meanwhile), go
    /// to the ready list to ask again.
    fn retryDeferred(h: *LaneHost, now: i96, ended: bool) void {
        if (h.deferred.items.len == 0) return;
        const none = if (h.cache) |store| store.restoring.items.len == 0 else true;
        var moved = false;
        var i: usize = 0;
        while (i < h.deferred.items.len) {
            const d = h.deferred.items[i];
            if (!ended and d.retry_at > now and !(d.until_end and none)) {
                i += 1;
                continue;
            }
            h.lock();
            _ = h.deferred.orderedRemove(i);
            h.toReady(d.job); // (releases the lock)
            moved = true;
        }
        if (!moved) return;
        h.lock();
        h.wake.signal(h.io);
        h.unlock();
    }

    /// A woken job to the ready list (the lock held; released here). Out of memory fails it alone.
    fn toReady(h: *LaneHost, job: *Job) void {
        h.ready.append(h.gpa, job) catch {
            h.unlock();
            return h.finish(job, .failed, "out of memory");
        };
        h.unlock();
    }

    /// Parked work's service (engine thread: after admissions and a prompt chunk, between rounds, at idle): restores
    /// progress under one elapsed-time budget shared fairly (each gets an equal share of what is left, a different one
    /// first each time); writes are staged with what remains. Restores that end wake every request waiting for them;
    /// deferred restores whose retry came ask again.
    fn service(h: *LaneHost, budget_ns: u64) void {
        const store = h.cache orelse return;
        const t0 = std.Io.Clock.awake.now(h.io).toNanoseconds();
        h.expireLoads(t0);
        var ended = false;
        const n = store.restoring.items.len;
        if (n > 0 and !h.entryHeld()) { // at most 64 a service, a different first one each time: every restore is polled in turn
            var buf: [64]*pc.Restore = undefined;
            const m = @min(n, buf.len);
            h.turn = (h.turn + 1) % n;
            for (0..m) |k| buf[k] = store.restoring.items[(h.turn + k) % n];
            for (buf[0..m], 0..) |r, k| {
                const spent: u64 = @intCast(@max(0, std.Io.Clock.awake.now(h.io).toNanoseconds() - t0));
                if (spent >= budget_ns and k > 0) break;
                const share = (budget_ns -| spent) / (m - k);
                if (store.restorePoll(r, share) != .pending) {
                    h.restoreEnded(r);
                    ended = true;
                }
            }
        }
        if (ended) status.noteKept(h); // (adopted states, and what their room evicted)
        h.retryDeferred(std.Io.Clock.awake.now(h.io).toNanoseconds(), ended);
        const spent: u64 = @intCast(@max(0, std.Io.Clock.awake.now(h.io).toNanoseconds() - t0));
        const idle = h.filling.items.len == 0 and h.core.activeCount() == 0 and h.core.fillingCount() == 0; // nothing decodes or fills: staging takes what is left
        store.pump(if (idle) budget_ns -| spent else @min(budget_ns -| spent, h.tier.pump_ns));
    }

    /// Requests parked on a restore longer than load_wait_ns (prompt passes not counted) are admitted without it (they
    /// look up what memory holds); a restore no request waits for any more stops, once.
    fn expireLoads(h: *LaneHost, now: i96) void {
        const store = h.cache orelse return;
        var any = false;
        var i: usize = 0;
        while (i < h.loading.items.len) {
            const l = h.loading.items[i];
            if (h.serviceable(now, l.since, l.passes) < h.tier.load_wait_ns) {
                i += 1;
                continue;
            }
            any = true;
            h.lock();
            _ = h.loading.orderedRemove(i);
            const waited: u64 = @intCast(@max(0, now - l.since));
            h.host_counts.restore_wait_timeouts += 1;
            h.host_counts.restore_wait_ns += waited;
            h.host_counts.restore_wait_max_ns = @max(h.host_counts.restore_wait_max_ns, waited);
            h.toReady(l.job); // (releases the lock)
            if (!h.wanted(l.restore)) store.restoreCancel(l.restore);
        }
        if (!any) return;
        h.lock();
        h.wake.signal(h.io);
        h.unlock();
    }

    /// Restore `r` ended (adopted, dropped or failed; `r` is already freed, only its address is compared): its
    /// waiters wake in arrival order, to look up what memory holds now.
    fn restoreEnded(h: *LaneHost, r: *pc.Restore) void {
        const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
        while (true) {
            h.lock();
            const i = for (h.loading.items, 0..) |l, i| {
                if (l.restore == r) break i;
            } else {
                h.wake.signal(h.io);
                return h.unlock();
            };
            const l = h.loading.orderedRemove(i);
            const waited: u64 = @intCast(@max(0, now - l.since));
            h.host_counts.restore_wait_ns += waited;
            h.host_counts.restore_wait_max_ns = @max(h.host_counts.restore_wait_max_ns, waited);
            h.toReady(l.job);
        }
    }

    /// Clean shutdown on the engine thread, nothing left to serve: restores stopped and writes drained within
    /// flush_ns while the engine and the family's states are alive (the backend's close frees them after this thread).
    fn shutdownTier(h: *LaneHost) void {
        if (!h.parking()) return;
        _ = h.cache.?.tierFlush(h.tier.flush_ns);
        status.noteLive(h);
    }
};

/// A line on the engine's log; tests stay quiet (the build runner fails a test that writes to stderr).
fn note(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.warn("lane host: " ++ fmt, args);
}

/// An informational line, quiet in tests too.
fn info(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.info("lane host: " ++ fmt, args);
}

test {
    _ = @import("lane_host_test.zig");
}
