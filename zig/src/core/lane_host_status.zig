//! A lane host's status counts: running and waiting requests, the 2 s decode and prompt rates, each stream's tokens,
//! and with a durable tier the prompt cache's counters.
const std = @import("std");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;

/// Tokens a round landed, and when.
pub const Mark = struct { at: i96, tokens: u64 };
const window_ns: i96 = 2 * std.time.ns_per_s;

/// Engine.status: the counts, read under the host's lock.
pub fn read(ctx: *anyopaque, out: *api.Status, stream_tokens: []u32) void {
    const h: *LaneHost = @ptrCast(@alignCast(ctx));
    h.mutex.lockUncancelable(h.io);
    defer h.mutex.unlock(h.io);
    const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
    var tokens: u64 = 0;
    for (h.decoded.items) |m| {
        if (m.at >= now - window_ns) tokens += m.tokens;
    }
    const n = @min(stream_tokens.len, h.live_tokens.items.len);
    @memcpy(stream_tokens[0..n], h.live_tokens.items[0..n]);
    out.* = .{
        .running = @intCast(h.admitted.items.len),
        .waiting = @intCast(h.queued.items.len + h.waitingCount()),
        .decode_tokens_per_second = @as(f64, @floatFromInt(tokens)) / 2.0,
        .prefill_tokens_per_second = if (now - h.prefill_at <= window_ns) h.prefill_rate else 0,
        .preemptions = 0,
        .streams = n,
        .generation_tokens = h.live_generated,
        .kept = h.kept_view,
        .cache = h.tele,
    };
    if (h.extra_status) |source| source.read(source.ctx, out);
}

/// The store's numbers for /health, copied under the lock (the store itself is the engine thread's alone).
pub fn noteKept(h: *LaneHost) void {
    const store = h.cache orelse return;
    const view = store.keptStats();
    h.mutex.lockUncancelable(h.io);
    defer h.mutex.unlock(h.io);
    h.kept_view = view;
}

/// `n` tokens delivered now; marks older than the window drop.
pub fn noteDecoded(h: *LaneHost, n: usize) void {
    const now = std.Io.Clock.awake.now(h.io).toNanoseconds();
    h.mutex.lockUncancelable(h.io);
    defer h.mutex.unlock(h.io);
    var keep: usize = 0;
    for (h.decoded.items) |m| {
        if (m.at < now - window_ns) continue;
        h.decoded.items[keep] = m;
        keep += 1;
    }
    h.decoded.shrinkRetainingCapacity(keep);
    h.decoded.append(h.gpa, .{ .at = now, .tokens = n }) catch {};
}

/// Each started stream's context and generated tokens, and with a tier the cache's counters (read here, on the engine
/// thread, before the lock), for the next status read.
pub fn noteLive(h: *LaneHost) void {
    const tele = if (h.tele != null) cacheStats(h) else null;
    h.mutex.lockUncancelable(h.io);
    defer h.mutex.unlock(h.io);
    h.live_tokens.clearRetainingCapacity();
    h.live_generated = 0;
    for (h.admitted.items) |job| if (job.started) {
        h.live_tokens.append(h.gpa, @intCast(job.stream.context.items.len)) catch {};
        h.live_generated += @intCast(job.stream.emitted().len);
    };
    if (tele) |t| h.tele = t;
}

/// The prompt cache's counters with its tier, the store's and the host's (engine thread).
fn cacheStats(h: *const LaneHost) api.CacheStats {
    var t: api.CacheStats = .{};
    if (h.cache) |store| {
        const c = store.counts;
        const s = store.tier_stats;
        t.memory_hits = c.hits;
        t.memory_misses = c.misses;
        t.kept = c.kept;
        t.evicted = c.evicted;
        t.refused = c.refused;
        t.failed = c.failed;
        t.entries = store.entries.items.len;
        t.held_bytes = store.held;
        t.disk_lookups = s.disk_lookups;
        t.disk_hits = s.disk_hits;
        t.disk_hits_committed = s.disk_hits_committed;
        t.restores_begun = s.restores_begun;
        t.restores_shared = s.restores_shared;
        t.restores_failed = s.restores_failed;
        t.restores_cancelled = s.restores_cancelled;
        t.restores_dropped = s.restores_dropped;
        t.restores_refused = s.restores_refused;
        t.restores_active = store.restoring.items.len;
        t.restoring_bytes = store.restoring_bytes;
        for (store.restoring.items) |r| {
            t.restore_blocks_done += r.done;
            t.restore_blocks_total += r.total;
        }
        t.restored_tokens = s.restored_tokens;
        t.writes_enqueued = s.writes_enqueued;
        t.shared_promoted = s.shared_promoted;
        t.flushes = s.flushes;
        t.flush_timeouts = s.flush_timeouts;
        t.flush_unpublished = s.flush_unpublished;
        t.chats_continued = c.chats_continued;
        t.chats_continued_prefill_tokens = c.chats_continued_prefill_tokens;
        t.spill = store.tierCounts();
    }
    const hc = h.host_counts;
    t.restore_waits = hc.restore_waits;
    t.restore_wait_timeouts = hc.restore_wait_timeouts;
    t.restore_wait_ns = hc.restore_wait_ns;
    t.restore_wait_max_ns = hc.restore_wait_max_ns;
    t.restores_deferred = hc.restores_deferred;
    t.restores_deferred_expired = hc.restores_deferred_expired;
    t.prefills_held = hc.prefills_held;
    t.prefills_held_expired = hc.prefills_held_expired;
    t.cached_tokens = hc.cached_tokens;
    t.computed_tokens = hc.computed_tokens;
    t.parked = h.loading.items.len + h.deferred.items.len;
    return t;
}
