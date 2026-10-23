//! LaneHost with a durable tier (spill v2: wojo's GLM 0110's scheduler hooks, the DeepSeek port's host tests on
//! v1.0.7's admission): requests parked on restores, their service between rounds, between prompt chunks and at idle,
//! restores the tier refuses for capacity (deferred, asked again, bounded), cancellation and shutdown, through the fake
//! lane backend and an in-memory tier.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const lh = @import("lane_host.zig");
const ft = @import("prompt_cache_fake_tier.zig");
const LaneHost = lh.LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Reason = api.Reason;
const gpa = std.testing.allocator;
const io = std.testing.io;

pub const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    cached: ?u32 = null,
    done: ?Reason = null,

    pub fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .prefilled => |c| b.cached = c,
            .logprobs => {},
            .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
            .finished => |f| b.done = f.reason,
        }
    }
    pub fn sink(b: *Box) api.Sink {
        return .{ .ctx = b, .event = event };
    }
    pub fn wait(b: *Box) Reason {
        while (true) {
            b.mutex.lockUncancelable(io);
            const d = b.done;
            b.mutex.unlock(io);
            if (d) |r| return r;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
    }
    pub fn deinit(b: *Box) void {
        b.tokens.deinit(gpa);
    }
    pub fn finished(b: *Box) ?Reason {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return b.done;
    }
    pub fn count(b: *Box) usize {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return b.tokens.items.len;
    }
    /// The reply equals the fake target's over `prompt` (any slip in the restored history changes it).
    pub fn exact(b: *Box, prompt: []const u32) !void {
        var history: std.ArrayList(u32) = .empty;
        defer history.deinit(gpa);
        try history.appendSlice(gpa, prompt);
        for (b.tokens.items) |t| {
            try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
            try history.append(gpa, t);
        }
    }
};

/// The fake backend, a prompt cache over its states, an in-memory tier and the host, at stable addresses.
pub const Rig = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine,
    store: pc.Store,
    tier: ft.FakeTier,
    host: LaneHost,
    stopped: bool = false,

    fn bytes(_: *anyopaque, at: u32) u64 {
        return 64 + 4 * @as(u64, at);
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return f.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.BackendRestores;
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        f.drop(saved);
    }
    fn pack(_: *anyopaque, a: std.mem.Allocator, saved: pc.Saved) anyerror![]u8 {
        const kept: *std.ArrayList(u32) = @ptrCast(@alignCast(saved));
        return a.dupe(u8, std.mem.sliceAsBytes(kept.items));
    }
    fn unpack(_: *anyopaque, b: []const u8) anyerror!pc.Saved {
        const kept = try gpa.create(std.ArrayList(u32));
        kept.* = .empty;
        const n = b.len / 4;
        try kept.resize(gpa, n);
        @memcpy(std.mem.sliceAsBytes(kept.items), b[0 .. n * 4]);
        return kept;
    }

    pub fn create(sp: api.Spill) !*Rig {
        return createWith(sp, .{});
    }

    /// `stepped`: the backend fills a prompt a chunk a call (prefill_step), the host a chunk a turn (fillOne).
    pub fn createOn(sp: api.Spill, stepped: bool) !*Rig {
        return createWith(sp, .{ .stepped = stepped });
    }

    pub const Opts = struct {
        stepped: bool = false,
        burst: bool = false, // the backend prefills a burst's prompts one after another in one call (Fake.backendBurst)
        lanes: u32 = 2,
        budget: u64 = 1 << 20, // the store's bytes (Rig.bytes: 64 + 4 a token)
        hold_min: ?u32 = null, // the host's TierPolicy.hold_min (its default: no test prompt is long enough to be held)
    };

    pub fn createWith(sp: api.Spill, o: Opts) !*Rig {
        const r = try gpa.create(Rig);
        errdefer gpa.destroy(r);
        r.* = .{ .cfg = undefined, .target = .{ .gpa = gpa }, .core = undefined, .store = undefined, .tier = .{ .gpa = gpa, .codec = .{ .ctx = r, .pack = pack, .unpack = unpack } }, .host = undefined };
        r.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        const backend = if (o.stepped) r.target.stepped() else if (o.burst) r.target.backendBurst() else r.target.backend();
        r.core = lanes.Engine.init(gpa, &r.cfg, backend, r.clock.clock());
        r.store = pc.Store.init(gpa, .{ .ptr = &r.target, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } }, .{ .min_gap = 4, .min_prompt = 0 }, o.budget);
        r.store.setTier(r.tier.tier(), 0);
        r.host = LaneHost.init(gpa, io, &r.core, .{ .lanes = o.lanes });
        r.host.cache = &r.store;
        r.host.configure(sp);
        if (o.hold_min) |m| r.host.tier.hold_min = m;
        try r.host.start();
        return r;
    }

    pub fn stop(r: *Rig) void {
        if (r.stopped) return;
        r.stopped = true;
        r.host.stop();
    }

    pub fn destroy(r: *Rig) void {
        r.stop();
        r.store.deinit();
        r.tier.deinit();
        r.core.deinit();
        r.target.deinit();
        r.cfg.deinit(gpa);
        gpa.destroy(r);
    }

    /// A published point of `key[0..at]` whose state is those tokens (the fake backend's state is its history).
    pub fn point(r: *Rig, key: []const u32, at: u32) !u64 {
        return r.tier.put(key[0..at], at, std.mem.sliceAsBytes(key[0..at]), 0, 0, false);
    }

    pub fn stats(r: *Rig) api.CacheStats {
        var st: api.Status = .{};
        r.host.engine().status(&st, &.{});
        return st.cache.?;
    }

    /// Waits (5 s at most) until the published counters satisfy `ok`.
    pub fn until(r: *Rig, comptime ok: fn (api.CacheStats) bool) !void {
        for (0..5000) |_| {
            if (ok(r.stats())) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
};

pub const t1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
pub const t2 = t1 ++ [_]u32{ 7, 7, 5, 3, 8, 8 };

test "a turn whose state left memory waits parked for its restore, resumes it exactly, and writes its own keeps forked from it" {
    const r = try Rig.create(.{ .load_ms = 5, .flush_ms = 1000 });
    defer r.destroy();
    r.tier.polls = 3;
    const id = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try b.exact(&t2);
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.writes_enqueued == 1;
        }
    }.ok);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.restore_waits);
    try std.testing.expectEqual(@as(u64, 1), s.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 8), s.restored_tokens);
    try std.testing.expectEqual(@as(u64, 1), s.memory_hits);
    try std.testing.expectEqual(@as(u64, 8), s.cached_tokens);
    try std.testing.expectEqual(@as(u64, 8), s.computed_tokens);
    try std.testing.expectEqual(@as(u64, 0), s.parked);
    try std.testing.expect(s.spill != null);
    r.stop();
    try std.testing.expectEqual(@as(u64, 1), r.tier.flushed); // the clean shutdown drained the tier, on the engine thread
    try std.testing.expectEqual(@as(u64, 1), r.store.tier_stats.flushes);
    const kept = for (r.tier.points.items) |p| {
        if (p.at == 14) break p;
    } else return error.TestUnexpectedResult;
    try std.testing.expectEqual(id, kept.parent.?); // its lineage forks the restored point at the resume point
    try std.testing.expectEqual(@as(u32, 8), kept.fork);
    try std.testing.expect(kept.group != 0); // a conversation of its own (the restored state is no turn's own)
    try std.testing.expectEqual(@as(usize, 1), kept.lineage.len); // the point carried none: a lineage of its own
}

test "two requests on one point share its restore; it stops once, when the last of them is cancelled" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    const e = r.host.engine();
    try e.submit(1, &req, a.sink());
    try e.submit(2, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 2;
        }
    }.ok);
    var st: api.Status = .{};
    e.status(&st, &.{});
    try std.testing.expectEqual(@as(u32, 2), st.waiting);
    try std.testing.expectEqual(@as(u32, 0), st.running);
    try std.testing.expectEqual(@as(u64, 1), st.cache.?.restores_begun);
    try std.testing.expectEqual(@as(u64, 1), st.cache.?.restores_shared);
    try std.testing.expectEqual(@as(u64, 2), st.cache.?.parked);
    e.cancel(1);
    try std.testing.expectEqual(Reason.cancelled, a.wait());
    try std.testing.expectEqual(@as(u64, 1), r.stats().restores_active); // still wanted by the other
    e.cancel(2);
    try std.testing.expectEqual(Reason.cancelled, b.wait());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restores_cancelled == 1 and s.restores_active == 0 and s.parked == 0;
        }
    }.ok);
    r.stop();
    try std.testing.expectEqual(@as(u64, 1), r.tier.cancels);
    try std.testing.expectEqual(@as(u64, 0), r.tier.misuse);
    try std.testing.expectEqual(@as(usize, 0), r.tier.jobs.items.len);
}

test "a restore that fails on any rank prefills from the start, exactly" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.fail_next = error.CrcMismatch;
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 0), b.cached);
    try b.exact(&t2);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.restores_failed);
    try std.testing.expectEqual(@as(u64, 0), s.disk_hits_committed);
}

test "restores progress between another request's prompt chunks (a backend that fills a chunk a turn), then their request resumes" {
    const r = try Rig.createOn(.{ .load_ms = 5 }, true);
    defer r.destroy();
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    const e = r.host.engine();
    try e.submit(1, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 1;
        }
    }.ok);
    const Watch = struct {
        tier: *ft.FakeTier,
        seen: [6]u64 = @splat(0),
        fn call(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
            const w: *@This() = @ptrCast(@alignCast(ctx));
            if (s.prompt_len == 12 and chunk < w.seen.len) w.seen[chunk] = @atomicLoad(u64, &w.tier.polled, .monotonic); // (the engine thread's)
        }
    };
    var watch: Watch = .{ .tier = &r.tier };
    r.target.prefill_chunks = 6;
    r.target.prefill_hook_ctx = &watch;
    r.target.prefill_hook = Watch.call;
    const other = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };
    var o: Box = .{};
    defer o.deinit();
    const oreq: Request = .{ .prompt = &other, .max_tokens = 1 };
    try e.submit(2, &oreq, o.sink());
    try std.testing.expectEqual(Reason.length, o.wait());
    try std.testing.expect(watch.seen[5] > watch.seen[0]); // polled inside the other's pass, between its chunks
    r.tier.gate.store(true, .release);
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try b.exact(&t2);
}

test "shutdown with requests parked on a restore that never ends: they are cancelled, the restore stops once, the tier flushes" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(1, &req, a.sink());
    try r.host.engine().submit(2, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 2;
        }
    }.ok);
    r.stop();
    try std.testing.expectEqual(Reason.cancelled, a.wait());
    try std.testing.expectEqual(Reason.cancelled, b.wait());
    try std.testing.expectEqual(@as(u64, 1), r.tier.cancels);
    try std.testing.expectEqual(@as(u64, 0), r.tier.misuse);
    try std.testing.expectEqual(@as(u64, 1), r.tier.flushed);
}

test "without a tier nothing is parked, configured or not, and status carries no cache counters" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Rig.bytes, .save = Rig.save, .restore = Rig.restore, .drop = Rig.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var host = LaneHost.init(gpa, io, &core, .{ .lanes = 2 });
    host.cache = &store;
    host.configure(.{ .dir = "/x", .gib = 1, .highwater = 0.7 });
    try std.testing.expect(host.tele == null);
    try host.start();
    defer host.stop();
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 3, .history_len = 14 };
    try host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    var st: api.Status = .{};
    host.engine().status(&st, &.{});
    try std.testing.expect(st.cache == null);
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    try std.testing.expectEqual(@as(u64, 0), host.host_counts.restore_waits);
    try std.testing.expectEqual(@as(u64, 0), store.tier_stats.disk_lookups);
}

test "a request parked on a restore past TF_SPILL_LOAD_WAIT_MS is admitted without it, and the restore stops" {
    const r = try Rig.create(.{ .load_ms = 5, .load_wait_ms = 5 });
    defer r.destroy();
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 0), b.cached);
    try b.exact(&t2);
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_wait_timeouts == 1 and s.restores_cancelled == 1 and s.restores_active == 0 and s.restoring_bytes == 0;
        }
    }.ok);
    r.stop();
    try std.testing.expectEqual(@as(u64, 1), r.tier.cancels);
    try std.testing.expectEqual(@as(u64, 0), r.tier.misuse);
}

test "another request's prompt pass does not count against a restore's wait: nothing is serviced during it" {
    const r = try Rig.create(.{ .load_ms = 5, .load_wait_ms = 200 });
    defer r.destroy();
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    const e = r.host.engine();
    try e.submit(1, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 1;
        }
    }.ok);
    const Slow = struct { // the other's pass takes three times the wait bound (the engine thread is in it throughout)
        fn call(_: *anyopaque, s: *lanes.Stream, chunk: usize) void {
            if (s.prompt_len == 12 and chunk == 0) std.Io.sleep(io, .fromMilliseconds(600), .awake) catch {};
        }
    };
    r.target.prefill_hook_ctx = r;
    r.target.prefill_hook = Slow.call;
    const other = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };
    var o: Box = .{};
    defer o.deinit();
    const oreq: Request = .{ .prompt = &other, .max_tokens = 1 };
    try e.submit(2, &oreq, o.sink());
    try std.testing.expectEqual(Reason.length, o.wait());
    r.tier.gate.store(true, .release);
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached); // its restore, not a prefill from 0
    try b.exact(&t2);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 0), s.restore_wait_timeouts);
    try std.testing.expect(s.restore_wait_max_ns >= 600 * std.time.ns_per_ms); // (the wall-clock wait, as counted)
}

test "restores are serviced between decode rounds while another stream decodes" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    const e = r.host.engine();
    const along = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };
    var a: Box = .{};
    defer a.deinit();
    const areq: Request = .{ .prompt = &along, .max_tokens = 100000 };
    try e.submit(1, &areq, a.sink());
    while (a.count() == 0) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try e.submit(2, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 1;
        }
    }.ok);
    const p0 = @atomicLoad(u64, &r.tier.polled, .monotonic);
    const n0 = a.count();
    var grew = false;
    for (0..5000) |_| {
        if (a.count() > n0 + 32 and @atomicLoad(u64, &r.tier.polled, .monotonic) > p0 + 8) {
            grew = true;
            break;
        }
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expect(grew); // polled round after round while the core is never idle
    try std.testing.expect(a.finished() == null);
    r.tier.gate.store(true, .release);
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try b.exact(&t2);
    e.cancel(1);
    try std.testing.expectEqual(Reason.cancelled, a.wait());
}

/// A short turn of `x` kept at 4 tokens (80 bytes), through the host.
fn shortTurn(r: *Rig, x: []const u32) !void {
    var a: Box = .{};
    defer a.deinit();
    const short: Request = .{ .prompt = x[0..6], .max_tokens = 1, .history_len = 4 };
    try r.host.engine().submit(9, &short, a.sink());
    try std.testing.expectEqual(Reason.length, a.wait());
}

const xs = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26 };
const ys = [_]u32{ 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46 };

test "a request that would park while another's fill holds its looked-up state waits for that pass: the restore's room never evicts it" {
    // (a backend that fills a chunk a turn: an admitted fill holds its looked-up state until its pass restores it)
    const r = try Rig.createWith(.{ .load_ms = 5 }, .{ .stepped = true, .lanes = 3, .budget = 150 });
    defer r.destroy();
    const e = r.host.engine();
    try shortTurn(r, &xs);
    _ = try r.point(&ys, 8); // 96 bytes: with the kept state's 80, past the budget
    const Gate = struct {
        r: *Rig,
        entered: std.atomic.Value(bool) = .init(false),
        open: std.atomic.Value(bool) = .init(false),
        lost: std.atomic.Value(bool) = .init(false),
        fn call(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
            const g: *@This() = @ptrCast(@alignCast(ctx));
            if (s.prompt_len == 12) { // holds the engine thread while both requests queue: one admission takes them
                g.entered.store(true, .release);
                while (!g.open.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
                return;
            }
            if (s.reuse.at != 4) return; // xs's resumed fill, on the engine thread: is its state still kept?
            const saved = s.reuse.saved orelse return;
            for (g.r.store.entries.items) |en| if (en.saved == saved) return;
            g.lost.store(true, .release);
            s.reuse.saved = null; // (the fake would read the freed state)
        }
    };
    var g: Gate = .{ .r = r };
    r.target.prefill_hook_ctx = &g;
    r.target.prefill_hook = Gate.call;
    const blocker = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };
    var c: Box = .{};
    defer c.deinit();
    const creq: Request = .{ .prompt = &blocker, .max_tokens = 1 };
    try e.submit(1, &creq, c.sink());
    while (!g.entered.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    const areq: Request = .{ .prompt = &xs, .max_tokens = 2, .history_len = 14 };
    const breq: Request = .{ .prompt = &ys, .max_tokens = 2, .history_len = 14 };
    try e.submit(2, &areq, a.sink());
    try e.submit(3, &breq, b.sink());
    g.open.store(true, .release);
    try std.testing.expectEqual(Reason.length, c.wait());
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expect(!g.lost.load(.acquire));
    try std.testing.expectEqual(@as(?u32, 4), a.cached);
    try a.exact(&xs);
    try std.testing.expectEqual(@as(?u32, 8), b.cached); // its restore, begun once the other's pass had its state
    try b.exact(&ys);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 0), s.restores_deferred); // (no capacity refusal)
    try std.testing.expectEqual(@as(u64, 1), s.disk_hits_committed);
}

test "a parked request's wait leaves out the turns another's fill holds its looked-up state (no restore progresses then)" {
    const r = try Rig.createWith(.{ .load_ms = 5, .load_wait_ms = 150 }, .{ .stepped = true });
    defer r.destroy();
    const e = r.host.engine();
    try shortTurn(r, &xs);
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const breq: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try e.submit(1, &breq, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 1;
        }
    }.ok);
    const Slow = struct { // xs's resumed fill: ten chunks, 40 ms each, holding its state until the last
        fn call(_: *anyopaque, s: *lanes.Stream, _: usize) void {
            if (s.reuse.at == 4) std.Io.sleep(io, .fromMilliseconds(40), .awake) catch {};
        }
    };
    r.target.prefill_chunks = 10;
    r.target.prefill_hook_ctx = r;
    r.target.prefill_hook = Slow.call;
    var a: Box = .{};
    defer a.deinit();
    const areq: Request = .{ .prompt = &xs, .max_tokens = 2, .history_len = 14 };
    try e.submit(2, &areq, a.sink());
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(@as(?u32, 4), a.cached);
    r.tier.gate.store(true, .release);
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached); // its restore, not a prefill from 0
    try b.exact(&t2);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 0), s.restore_wait_timeouts);
    try std.testing.expect(s.restore_wait_max_ns >= 400 * std.time.ns_per_ms); // (the wall-clock wait, as counted)
}

test "a park whose restore's room evicts kept states shows at once in the kept view (/health), as in the cache's counters" {
    const r = try Rig.createWith(.{ .load_ms = 5 }, .{ .budget = 150 });
    defer r.destroy();
    try shortTurn(r, &t2);
    _ = try r.point(&t2, 8); // 96 bytes: the kept state's 80 cannot stay beside it
    r.tier.gate.store(false, .release);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    const e = r.host.engine();
    try e.submit(1, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 1 and s.entries == 0;
        }
    }.ok);
    var st: api.Status = .{};
    e.status(&st, &.{});
    try std.testing.expectEqual(@as(u32, 0), st.kept.?.prompts);
    try std.testing.expectEqual(@as(u64, 0), st.kept.?.bytes);
    r.tier.gate.store(true, .release);
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try b.exact(&t2);
}

test "configure publishes the cache's counters only over a tier; its bounds come from the settings" {
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var tier: ft.FakeTier = .{ .gpa = gpa, .codec = .{ .ctx = &target, .pack = Rig.pack, .unpack = Rig.unpack } };
    defer tier.deinit();
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Rig.bytes, .save = Rig.save, .restore = Rig.restore, .drop = Rig.drop } }, .{}, 1 << 20);
    defer store.deinit();
    var host = LaneHost.init(gpa, io, &core, .{ .lanes = 2 });
    defer host.stop(); // (never started: frees the lists)
    const sp: api.Spill = .{ .dir = "/x", .gib = 1, .min_tokens = 0, .load_ms = 0, .load_wait_ms = 250, .defer_ms = 0, .flush_ms = 1500 };
    host.configure(sp);
    try std.testing.expect(host.tele == null); // no prompt cache
    host.cache = &store;
    host.configure(sp);
    try std.testing.expect(host.tele == null); // a prompt cache with no tier
    store.setTier(tier.tier(), 7);
    host.configure(sp);
    var st: api.Status = .{};
    host.engine().status(&st, &.{});
    try std.testing.expect(st.cache != null);
    try std.testing.expectEqual(@as(u64, std.time.ns_per_ms), host.tier.service_ns); // 0 never progresses a restore: 1 ms
    try std.testing.expectEqual(@as(u64, std.time.ns_per_ms), host.tier.pump_ns);
    try std.testing.expectEqual(@as(i96, 250 * std.time.ns_per_ms), host.tier.load_wait_ns);
    try std.testing.expectEqual(@as(i96, 0), host.tier.defer_ns);
    try std.testing.expectEqual(@as(u64, 1500 * std.time.ns_per_ms), host.tier.flush_ns);
    try std.testing.expectEqual(@as(u32, 8192), host.tier.hold_min); // not --spill-min-tokens (whose 0 writes every state)
    host.configure(.{ .dir = "/x", .gib = 1, .load_ms = 2.5 }); // TF_GLM_SPILL_LOAD_MS=2.5 (Python's float)
    try std.testing.expectEqual(@as(u64, 2_500_000), host.tier.service_ns);
}

// -- restores refused for capacity ---------------------------------------------------------------------------------

test "a restore the tier refuses for capacity is asked again shortly, then resumes exactly" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    r.tier.refuse = 2; // two refusals (TooManyRestores: restores in flight), then the begin goes through
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(1, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try b.exact(&t2);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.restores_deferred); // once a request, however often it asked
    try std.testing.expectEqual(@as(u64, 0), s.restores_deferred_expired);
    try std.testing.expectEqual(@as(u64, 1), s.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 2), r.tier.refused);
    try std.testing.expectEqual(@as(u64, 3), r.tier.begins);
    try std.testing.expectEqual(@as(u64, 0), s.parked);
}

test "a deferral's bound leaves out a prompt pass run earlier in the same admission, before the deferred request asks again" {
    const r = try Rig.create(.{ .load_ms = 5, .defer_ms = 300 });
    defer r.destroy();
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    var w: Box = .{};
    defer w.deinit();
    const wreq: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    const e = r.host.engine();
    try e.submit(1, &wreq, w.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 1;
        }
    }.ok);
    r.tier.refuse = 1 << 20; // at capacity: the next restore is refused and deferred
    const other = [_]u32{ 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4 };
    _ = try r.point(&other, 8);
    var b: Box = .{};
    defer b.deinit();
    const breq: Request = .{ .prompt = &other, .max_tokens = 6, .history_len = 12 };
    try e.submit(2, &breq, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restores_deferred == 1;
        }
    }.ok);
    const Slow = struct { // the first request's resumed pass takes twice the deferral bound
        fn call(_: *anyopaque, s: *lanes.Stream, chunk: usize) void {
            if (s.prompt_len == 16 and chunk == 0) std.Io.sleep(io, .fromMilliseconds(600), .awake) catch {};
        }
    };
    r.target.prefill_hook_ctx = r;
    r.target.prefill_hook = Slow.call;
    // its restore ends: it wakes ahead of the deferred one (which asks again when a restore ends), in one admission
    r.tier.gate.store(true, .release);
    try std.testing.expectEqual(Reason.length, w.wait());
    try std.testing.expectEqual(@as(?u32, 8), w.cached);
    r.tier.refuse = 0; // room again
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached); // its restore, not a prefill from 0
    try b.exact(&other);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.restores_deferred);
    try std.testing.expectEqual(@as(u64, 0), s.restores_deferred_expired);
    try std.testing.expectEqual(@as(u64, 2), s.disk_hits_committed);
}

/// A memory state of t2 at 4 (a short turn's keep), then t2's durable point at 8.
fn shortThenPoint(r: *Rig) !void {
    var a: Box = .{};
    defer a.deinit();
    const short: Request = .{ .prompt = t2[0..6], .max_tokens = 1, .history_len = 4 };
    try r.host.engine().submit(1, &short, a.sink());
    try std.testing.expectEqual(Reason.length, a.wait());
    _ = try r.point(&t2, 8);
}

test "a full tier refuses before any room is made: the request asks again, the refusal counted once" {
    const r = try Rig.create(.{ .load_ms = 5 });
    defer r.destroy();
    try shortThenPoint(r);
    r.tier.full = 3; // the restores in flight at the tier's cap for three asks
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(2, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), b.cached);
    try b.exact(&t2);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.restores_deferred);
    try std.testing.expectEqual(@as(u64, 1), s.restores_refused); // once a request, not once an ask
    try std.testing.expectEqual(@as(u64, 1), r.tier.begins); // never asked to begin while full
    try std.testing.expectEqual(@as(u64, 0), s.evicted);
}

test "a restore still refused past its deferral bound resumes the shorter memory state, never a prefill from 0" {
    const r = try Rig.create(.{ .load_ms = 5, .defer_ms = 40 });
    defer r.destroy();
    try shortThenPoint(r);
    r.tier.refuse = 1 << 20; // at capacity throughout
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(2, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 4), b.cached); // the memory state, kept while the restore was refused
    try b.exact(&t2);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.restores_deferred);
    try std.testing.expectEqual(@as(u64, 1), s.restores_deferred_expired);
    try std.testing.expect(r.tier.refused >= 2); // asked again while deferred
    try std.testing.expectEqual(@as(u64, 0), s.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 0), s.parked);
}

test "with the deferral off (TF_SPILL_DEFER_MS=0) a refused restore resumes what memory holds at once" {
    const r = try Rig.create(.{ .load_ms = 5, .defer_ms = 0 });
    defer r.destroy();
    try shortThenPoint(r);
    r.tier.refuse = 1 << 20;
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(2, &req, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 4), b.cached);
    try b.exact(&t2);
    try std.testing.expectEqual(@as(u64, 0), r.stats().restores_deferred);
    try std.testing.expectEqual(@as(u64, 1), r.tier.refused);
}

test "a refusal that waiting cannot cure is not deferred; a deferred request cancelled leaves every list once" {
    const r = try Rig.create(.{ .load_ms = 5, .defer_ms = 10_000 });
    defer r.destroy();
    r.tier.refuse = 1;
    r.tier.refuse_err = error.PointMissing; // not a capacity refusal: admitted at once, prefilled
    _ = try r.point(&t2, 8);
    var a: Box = .{};
    defer a.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    const e = r.host.engine();
    try e.submit(1, &req, a.sink());
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(@as(?u32, 0), a.cached);
    try std.testing.expectEqual(@as(u64, 0), r.stats().restores_deferred);
    // at capacity for long: the request waits deferred (no lane, counted waiting), and its cancel ends it there
    r.tier.refuse = 1 << 20;
    r.tier.refuse_err = error.TooManyRestores;
    const other = [_]u32{ 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6 };
    _ = try r.point(&other, 8);
    var b: Box = .{};
    defer b.deinit();
    const req_b: Request = .{ .prompt = &other, .max_tokens = 6, .history_len = 14 };
    try e.submit(2, &req_b, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restores_deferred == 1 and s.parked == 1;
        }
    }.ok);
    var st: api.Status = .{};
    e.status(&st, &.{});
    try std.testing.expectEqual(@as(u32, 1), st.waiting);
    try std.testing.expectEqual(@as(u32, 0), st.running);
    e.cancel(2);
    try std.testing.expectEqual(Reason.cancelled, b.wait());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.parked == 0;
        }
    }.ok);
    try std.testing.expectEqual(@as(usize, 0), r.tier.jobs.items.len); // nothing was ever begun for it
}

test "shutdown with a deferred request cancels it once" {
    const r = try Rig.create(.{ .load_ms = 5, .defer_ms = 10_000 });
    defer r.destroy();
    r.tier.refuse = 1 << 20;
    _ = try r.point(&t2, 8);
    var b: Box = .{};
    defer b.deinit();
    const req: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try r.host.engine().submit(1, &req, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restores_deferred == 1;
        }
    }.ok);
    r.stop();
    try std.testing.expectEqual(@as(?Reason, Reason.cancelled), b.finished());
}

// -- two restores at once (spill_check's dual-restore) --------------------------------------------------------------

/// spill_check's dual-restore on the host: two requests whose states are only on the tier, taken by one admission while
/// a third stream decodes. The tier holds one restore at a time and refuses the other's begin with `busy` while it is
/// in flight. `stepped`: a fill a chunk a turn, and a third request admitted first in that admission resumes a memory
/// state, so it holds its looked-up entry while the two would park. Both must resume their own restores, neither
/// prefill from 0 (g2's FAIL: the second admission prefilled 38K tokens from 0 while the first's restore waited).
fn dualRestore(busy: anyerror, stepped: bool) !void {
    var a: Box = .{}; // (the boxes outlive the host: a failed check stops it first)
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    var c: Box = .{};
    defer c.deinit();
    var d: Box = .{};
    defer d.deinit();
    var x: Box = .{};
    defer x.deinit();
    const r = try Rig.createWith(.{ .load_ms = 5 }, .{ .stepped = stepped, .lanes = 5 });
    defer r.destroy();
    const e = r.host.engine();
    if (stepped) try shortTurn(r, &xs); // xs kept at 4: the third request resumes it
    r.tier.busy_err = busy;
    r.tier.polls = 3;
    const one = [_]u32{ 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66 };
    const two = [_]u32{ 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86 };
    _ = try r.point(&one, 8);
    _ = try r.point(&two, 8);
    const along = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };
    const dreq: Request = .{ .prompt = &along, .max_tokens = 100000 };
    try e.submit(1, &dreq, d.sink());
    while (d.count() == 0) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    const Gate = struct {
        entered: std.atomic.Value(bool) = .init(false),
        open: std.atomic.Value(bool) = .init(false),
        fn call(ctx: *anyopaque, s: *lanes.Stream, _: usize) void {
            const g: *@This() = @ptrCast(@alignCast(ctx));
            if (s.prompt_len != 13) return; // holds the engine thread while the others queue: one admission takes them
            g.entered.store(true, .release);
            while (!g.open.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
    };
    var g: Gate = .{};
    r.target.prefill_chunks = 3;
    r.target.prefill_hook_ctx = &g;
    r.target.prefill_hook = Gate.call;
    const blocker = [_]u32{ 9, 9, 9, 8, 8, 8, 7, 7, 7, 6, 6, 6, 5 };
    const creq: Request = .{ .prompt = &blocker, .max_tokens = 1 };
    try e.submit(2, &creq, c.sink());
    while (!g.entered.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    const xreq: Request = .{ .prompt = &xs, .max_tokens = 2, .history_len = 14 };
    if (stepped) try e.submit(3, &xreq, x.sink());
    const areq: Request = .{ .prompt = &one, .max_tokens = 4, .history_len = 14 };
    const breq: Request = .{ .prompt = &two, .max_tokens = 4, .history_len = 14 };
    try e.submit(4, &areq, a.sink());
    try e.submit(5, &breq, b.sink());
    g.open.store(true, .release);
    try std.testing.expectEqual(Reason.length, c.wait());
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), a.cached); // its own restore
    try a.exact(&one);
    try std.testing.expectEqual(@as(?u32, 8), b.cached); // its own restore too, not a prefill from 0
    try b.exact(&two);
    if (stepped) {
        try std.testing.expectEqual(Reason.length, x.wait());
        try std.testing.expectEqual(@as(?u32, 4), x.cached);
        try x.exact(&xs);
    }
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 2), s.restores_begun);
    try std.testing.expectEqual(@as(u64, 2), s.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 2), s.restore_waits);
    try std.testing.expectEqual(@as(u64, 1), s.restores_deferred); // the second asked again once the first ended
    try std.testing.expectEqual(@as(u64, 0), s.restores_deferred_expired);
    try std.testing.expect(r.tier.refused >= 1);
    try std.testing.expect(d.finished() == null); // the third stream decoded throughout
    e.cancel(1);
    try std.testing.expectEqual(Reason.cancelled, d.wait());
    try d.exact(&along);
}

test "two restores at once while a stream decodes: the tier refuses the second for capacity, which waits and resumes its own" {
    try dualRestore(error.TooManyRestores, false);
    try dualRestore(error.NoSlot, true);
}

test "two restores at once while a stream decodes: a second begin refused for something the first holds until it ends waits for it, never a prefill from 0" {
    try dualRestore(error.HeldByAnother, false); // (a refusal that is neither capacity nor the point's)
    try dualRestore(error.HeldByAnother, true);
}

test "a restore refused for its destination's memory (GLM's SpillReserve) waits for the one in flight; refused again with none left, it prefills at once: counted expired" {
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    const r = try Rig.create(.{ .load_ms = 5, .defer_ms = 10_000 });
    defer r.destroy();
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    const e = r.host.engine();
    const areq: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try e.submit(1, &areq, a.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 1;
        }
    }.ok);
    r.tier.refuse = 1 << 20; // every later begin refused for its destination (with the first in flight, and after)
    r.tier.refuse_err = error.SpillReserve;
    const other = [_]u32{ 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6 };
    _ = try r.point(&other, 8);
    const breq: Request = .{ .prompt = &other, .max_tokens = 6, .history_len = 14 };
    try e.submit(2, &breq, b.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restores_deferred == 1 and s.parked == 2;
        }
    }.ok);
    try std.testing.expect(b.finished() == null); // waiting for the first restore, not prefilling
    std.Io.sleep(io, .fromMilliseconds(100), .awake) catch {};
    try std.testing.expectEqual(@as(u64, 1), r.tier.refused); // not asked again while the first is in flight
    r.tier.gate.store(true, .release);
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(@as(?u32, 8), a.cached);
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 0), b.cached); // asked again with nothing in flight: no wait could help
    try b.exact(&other);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.restores_deferred);
    try std.testing.expectEqual(@as(u64, 1), s.restores_deferred_expired);
    try std.testing.expectEqual(@as(u64, 1), s.restores_begun);
    try std.testing.expectEqual(@as(u64, 0), s.parked);
    try std.testing.expectEqual(@as(u64, 2), r.tier.refused); // once more, when the first ended
}

test "a restore refused because its home part has no room (GLM's KvShardHomeFull) beside one in flight is admitted on memory at once, not deferred" {
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    const r = try Rig.create(.{ .load_ms = 5, .defer_ms = 10_000 });
    defer r.destroy();
    r.tier.gate.store(false, .release); // the first restore stays in flight until the end
    _ = try r.point(&t2, 8);
    const e = r.host.engine();
    const areq: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try e.submit(1, &areq, a.sink());
    try r.until(struct {
        fn ok(s: api.CacheStats) bool {
            return s.restore_waits == 1;
        }
    }.ok);
    // the second point is on disk but its resume could only start from 0 (its home part full, no carry): its own
    // placement, which no restore in flight holds, so its prompt prefills now rather than waiting for the first restore
    r.tier.refuse = 1 << 20;
    r.tier.refuse_err = error.KvShardHomeFull;
    const other = [_]u32{ 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6 };
    _ = try r.point(&other, 8);
    const breq: Request = .{ .prompt = &other, .max_tokens = 6, .history_len = 14 };
    try e.submit(2, &breq, b.sink());
    for (0..5000) |_| {
        if (b.finished() != null) break;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expectEqual(@as(?Reason, Reason.length), b.finished()); // with the first restore still in flight
    try std.testing.expectEqual(@as(?u32, 0), b.cached);
    try b.exact(&other);
    try std.testing.expect(a.finished() == null);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 0), s.restores_deferred);
    try std.testing.expectEqual(@as(u64, 0), s.restores_deferred_expired);
    try std.testing.expectEqual(@as(u64, 1), r.tier.refused); // asked once, never again
    r.tier.gate.store(true, .release);
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(@as(?u32, 8), a.cached);
    try a.exact(&t2);
    try std.testing.expectEqual(@as(u64, 1), r.stats().restores_begun);
}

// -- prompt passes while requests are parked (whole passes service nothing) -----------------------------------------

/// Records, at each watched pass's first chunk (on the engine thread), how many restores the store had adopted.
const Seen = struct {
    r: *Rig,
    gate: std.atomic.Value(bool) = .init(true), // the blocker's pass (13 tokens) waits while false
    entered: std.atomic.Value(bool) = .init(false),
    adopted: [32]u64 = @splat(std.math.maxInt(u64)), // by prompt length
    fn call(ctx: *anyopaque, s: *lanes.Stream, chunk: usize) void {
        const w: *@This() = @ptrCast(@alignCast(ctx));
        if (chunk != 0 or s.prompt_len >= w.adopted.len) return;
        if (s.prompt_len == 13) {
            w.entered.store(true, .release);
            while (!w.gate.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        @atomicStore(u64, &w.adopted[s.prompt_len], w.r.store.tier_stats.disk_hits_committed, .release);
    }
    fn at(w: *@This(), len: usize) u64 {
        return @atomicLoad(u64, &w.adopted[len], .acquire);
    }
};

const blocker13 = [_]u32{ 9, 9, 9, 8, 8, 8, 7, 7, 7, 6, 6, 6, 5 };
const cold12 = [_]u32{ 8, 6, 7, 5, 3, 0, 9, 2, 1, 4, 4, 4 };
const cold14 = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8, 2, 8, 4, 5, 9, 0 };

test "while a request is parked an admission runs one whole pass: its restore is installed before the next one" {
    var a: Box = .{};
    defer a.deinit();
    var c: Box = .{};
    defer c.deinit();
    var p: Box = .{};
    defer p.deinit();
    var q: Box = .{};
    defer q.deinit();
    const r = try Rig.create(.{ .load_ms = 5 }); // (hold_min's default: no pass is long enough to wait)
    defer r.destroy();
    _ = try r.point(&t2, 8);
    var w: Seen = .{ .r = r };
    w.gate.store(false, .release);
    r.target.prefill_hook_ctx = &w;
    r.target.prefill_hook = Seen.call;
    const e = r.host.engine();
    const creq: Request = .{ .prompt = &blocker13, .max_tokens = 1 };
    try e.submit(1, &creq, c.sink());
    while (!w.entered.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    // one admission takes them all once the blocker's pass ends: the parked one, then two cold passes
    const areq: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    const preq: Request = .{ .prompt = &cold12, .max_tokens = 2 };
    const qreq: Request = .{ .prompt = &cold14, .max_tokens = 2 };
    try e.submit(2, &areq, a.sink());
    try e.submit(3, &preq, p.sink());
    try e.submit(4, &qreq, q.sink());
    w.gate.store(true, .release);
    for ([_]*Box{ &c, &a, &p, &q }) |b| try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), a.cached);
    try a.exact(&t2);
    try p.exact(&cold12);
    try q.exact(&cold14);
    try std.testing.expectEqual(@as(u64, 1), w.at(12)); // its restore was installed before the next pass ran
    try std.testing.expectEqual(@as(u64, 1), w.at(14));
    try std.testing.expectEqual(@as(u64, 0), r.stats().prefills_held);
}

test "a long cold pass waits while a request is parked on its restore, a short one does not; the restore installs first" {
    var a: Box = .{};
    defer a.deinit();
    var p: Box = .{};
    defer p.deinit();
    var s: Box = .{};
    defer s.deinit();
    const r = try Rig.createWith(.{ .load_ms = 5 }, .{ .hold_min = 8 }); // (passes of 8 tokens and more wait)
    defer r.destroy();
    var w: Seen = .{ .r = r };
    r.target.prefill_hook_ctx = &w;
    r.target.prefill_hook = Seen.call;
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    const e = r.host.engine();
    const areq: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try e.submit(1, &areq, a.sink());
    try r.until(struct {
        fn ok(st: api.CacheStats) bool {
            return st.restore_waits == 1;
        }
    }.ok);
    const preq: Request = .{ .prompt = &cold12, .max_tokens = 2 };
    try e.submit(2, &preq, p.sink());
    try r.until(struct {
        fn ok(st: api.CacheStats) bool {
            return st.prefills_held == 1 and st.parked == 2;
        }
    }.ok);
    const short = [_]u32{ 6, 5, 4, 3, 2, 1, 0 }; // 7 tokens: runs at once
    const sreq: Request = .{ .prompt = &short, .max_tokens = 2 };
    try e.submit(3, &sreq, s.sink());
    try std.testing.expectEqual(Reason.length, s.wait());
    try s.exact(&short);
    std.Io.sleep(io, .fromMilliseconds(50), .awake) catch {};
    try std.testing.expect(p.finished() == null); // still waiting for the restore, holding no lane
    // counted waiting (read until seen: each 25 ms retry takes the held request off one list before the other)
    const waiting = for (0..5000) |_| {
        var st: api.Status = .{};
        e.status(&st, &.{});
        if (st.waiting == 2) break st.waiting;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    } else 0;
    try std.testing.expectEqual(@as(u32, 2), waiting);
    r.tier.gate.store(true, .release);
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(@as(?u32, 8), a.cached);
    try a.exact(&t2);
    try std.testing.expectEqual(Reason.length, p.wait());
    try std.testing.expectEqual(@as(?u32, 0), p.cached);
    try p.exact(&cold12);
    try std.testing.expectEqual(@as(u64, 1), w.at(12)); // its pass ran once the restore was installed
    const c = r.stats();
    try std.testing.expectEqual(@as(u64, 1), c.prefills_held);
    try std.testing.expectEqual(@as(u64, 0), c.prefills_held_expired);
    try std.testing.expectEqual(@as(u64, 0), c.parked);
}

test "a held pass runs after TF_SPILL_DEFER_MS while the restore it waited for still has not installed" {
    var a: Box = .{};
    defer a.deinit();
    var p: Box = .{};
    defer p.deinit();
    const r = try Rig.createWith(.{ .load_ms = 5, .defer_ms = 60 }, .{ .hold_min = 8 });
    defer r.destroy();
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    const e = r.host.engine();
    const areq: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try e.submit(1, &areq, a.sink());
    try r.until(struct {
        fn ok(st: api.CacheStats) bool {
            return st.restore_waits == 1;
        }
    }.ok);
    const preq: Request = .{ .prompt = &cold12, .max_tokens = 2 };
    try e.submit(2, &preq, p.sink());
    try std.testing.expectEqual(Reason.length, p.wait()); // the restore still pending
    try std.testing.expectEqual(@as(?u32, 0), p.cached);
    try p.exact(&cold12);
    try std.testing.expect(a.finished() == null);
    const c = r.stats();
    try std.testing.expectEqual(@as(u64, 1), c.prefills_held);
    try std.testing.expectEqual(@as(u64, 1), c.prefills_held_expired);
    try std.testing.expectEqual(@as(u64, 0), c.disk_hits_committed);
    r.tier.gate.store(true, .release);
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(@as(?u32, 8), a.cached);
    try a.exact(&t2);
}

test "while a request is parked a burst takes no second prompt (a whole-pass backend runs them one after another); with none parked it does" {
    var a: Box = .{};
    defer a.deinit();
    var c: Box = .{};
    defer c.deinit();
    var p: Box = .{};
    defer p.deinit();
    var q: Box = .{};
    defer q.deinit();
    const r = try Rig.createWith(.{ .load_ms = 5 }, .{ .burst = true }); // (no pass is long enough to wait)
    defer r.destroy();
    _ = try r.point(&t2, 8);
    var w: Seen = .{ .r = r };
    w.gate.store(false, .release);
    r.target.prefill_hook_ctx = &w;
    r.target.prefill_hook = Seen.call;
    const e = r.host.engine();
    const creq: Request = .{ .prompt = &blocker13, .max_tokens = 1 };
    try e.submit(1, &creq, c.sink());
    while (!w.entered.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    // one admission takes them once the blocker's pass ends: the parked one, then two cold prompts a burst would take
    const areq: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    const preq: Request = .{ .prompt = &cold12, .max_tokens = 2 };
    const qreq: Request = .{ .prompt = &cold14, .max_tokens = 2 };
    try e.submit(2, &areq, a.sink());
    try e.submit(3, &preq, p.sink());
    try e.submit(4, &qreq, q.sink());
    w.gate.store(true, .release);
    for ([_]*Box{ &c, &a, &p, &q }) |b| try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(?u32, 8), a.cached);
    try a.exact(&t2);
    try p.exact(&cold12);
    try q.exact(&cold14);
    try std.testing.expectEqual(@as(u64, 1), w.at(14)); // the restore was installed between the two cold passes
    try std.testing.expectEqual(@as(usize, 0), r.target.bursts);
    // nothing parked: two cold prompts in one admission still prefill in one backend call
    var p2: Box = .{};
    defer p2.deinit();
    var q2: Box = .{};
    defer q2.deinit();
    var c2: Box = .{};
    defer c2.deinit();
    w.entered.store(false, .release);
    w.gate.store(false, .release);
    try e.submit(5, &creq, c2.sink());
    while (!w.entered.load(.acquire)) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    const cold11 = [_]u32{ 1, 4, 1, 4, 2, 1, 3, 5, 6, 2, 3 };
    const cold15 = [_]u32{ 1, 7, 3, 2, 0, 5, 0, 8, 0, 7, 5, 6, 8, 8, 7 };
    const p2req: Request = .{ .prompt = &cold11, .max_tokens = 2 };
    const q2req: Request = .{ .prompt = &cold15, .max_tokens = 2 };
    try e.submit(6, &p2req, p2.sink());
    try e.submit(7, &q2req, q2.sink());
    w.gate.store(true, .release);
    for ([_]*Box{ &c2, &p2, &q2 }) |b| try std.testing.expectEqual(Reason.length, b.wait());
    try p2.exact(&cold11);
    try q2.exact(&cold15);
    try std.testing.expectEqual(@as(usize, 1), r.target.bursts);
}

test "a deferral that reaches its bound with the restore it waited for still in flight is not held again: its pass runs" {
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    const r = try Rig.createWith(.{ .load_ms = 5, .defer_ms = 100 }, .{ .hold_min = 8 });
    defer r.destroy();
    r.tier.gate.store(false, .release);
    _ = try r.point(&t2, 8);
    const other = [_]u32{ 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6 };
    _ = try r.point(&other, 8);
    const e = r.host.engine();
    const areq: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 14 };
    try e.submit(1, &areq, a.sink());
    try r.until(struct {
        fn ok(st: api.CacheStats) bool {
            return st.restore_waits == 1;
        }
    }.ok);
    r.tier.refuse = 1 << 20; // every later begin refused for its destination's memory (GLM's), a's restore still pending
    r.tier.refuse_err = error.SpillReserve;
    const breq: Request = .{ .prompt = &other, .max_tokens = 6, .history_len = 14 }; // 16 tokens: a pass long enough to hold
    try e.submit(2, &breq, b.sink());
    try std.testing.expectEqual(Reason.length, b.wait()); // a's restore still pending
    try std.testing.expectEqual(@as(?u32, 0), b.cached);
    try b.exact(&other);
    try std.testing.expect(a.finished() == null);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.restores_deferred);
    try std.testing.expectEqual(@as(u64, 1), s.restores_deferred_expired);
    try std.testing.expectEqual(@as(u64, 0), s.prefills_held); // it waited its bound once, not twice
    try std.testing.expectEqual(@as(u64, 0), s.prefills_held_expired);
    r.tier.gate.store(true, .release);
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(@as(?u32, 8), a.cached);
    try a.exact(&t2);
}

test "a request whose own restore installed resumes it at once, never held for another's restore still pending" {
    var a: Box = .{};
    defer a.deinit();
    var b: Box = .{};
    defer b.deinit();
    const r = try Rig.createWith(.{ .load_ms = 5, .defer_ms = 200 }, .{ .hold_min = 8 });
    defer r.destroy();
    const other = [_]u32{ 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6 };
    _ = try r.point(&other, 8);
    const long = [_]u32{ 4, 8, 15, 16, 23, 42, 4, 8, 15, 16, 23, 42, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 14, 23, 37, 60, 7, 1 };
    _ = try r.point(&long, 8); // its pass computes 20 tokens past the restored 8
    r.tier.polls = 1 << 20; // b's restore never installs
    const e = r.host.engine();
    const breq: Request = .{ .prompt = &other, .max_tokens = 6, .history_len = 14 };
    try e.submit(1, &breq, b.sink());
    try r.until(struct {
        fn ok(st: api.CacheStats) bool {
            return st.restore_waits == 1;
        }
    }.ok);
    r.tier.polls = 1; // a's installs at its first poll
    const areq: Request = .{ .prompt = &long, .max_tokens = 4, .history_len = 26 };
    try e.submit(2, &areq, a.sink());
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(@as(?u32, 8), a.cached);
    try a.exact(&long);
    try std.testing.expect(b.finished() == null);
    const s = r.stats();
    try std.testing.expectEqual(@as(u64, 1), s.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 0), s.prefills_held); // it waited for its own restore already
    try std.testing.expectEqual(@as(u64, 0), s.prefills_held_expired);
    e.cancel(1);
    try std.testing.expectEqual(Reason.cancelled, b.wait());
}
