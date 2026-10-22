//! Smooth streaming (TF_GLM_STREAM_SMOOTH=1; recipe patch 0061, Python `cuda/stream_pacing.py`): a streamed reply's text
//! goes out token by token at a steady pace from a short playout buffer, instead of a burst of drafted tokens every
//! round and a pause whenever the engine fills another request's prompt.
//!
//! The engine's rounds come in bursts: a round accepts ~3 drafted tokens, a round takes longer while other requests
//! decode, and while a new prompt fills the rounds wait for its layer slices. The request's tokens enter a queue as they
//! arrive and a writer thread sends them one event each: the next goes out `hold / queued` after the previous one (never
//! before it arrived), so the queue settles at about `hold` of tokens and a pause in the rounds shorter than that is
//! spread over the tokens already queued instead of showing as a freeze. No token is held longer than `longest_hold`
//! times `hold`.
//!
//! The text, its order and the reply's end are unchanged: only when events go out. Events that are not text (tool-call
//! deltas, keep-alives) keep their place in the queue and are not paced; the round that ends the reply flushes the queue
//! at once. The engine never waits: the request thread only queues.
const std = @import("std");
const json = @import("json");
const chat = @import("chat.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// The default playout buffer: about this much text is held, so shorter pauses never show.
pub const hold_default_ms: u32 = 400;
/// No token is held longer than 5/2 times the buffer.
const longest_num: i96 = 5;
const longest_den: i96 = 2;

pub const Item = struct { delta: Value, paced: bool, arrived: i96 };

/// The queue's pacing rule, apart from threads and clocks.
pub const Timing = struct {
    hold_ns: i96,
    last: ?i96 = null, // when the previous event went out
    paced: usize = 0, // paced items queued
    flushing: bool = false,

    /// When the front of the queue goes out (the queue is not empty).
    pub fn due(t: *const Timing, front: Item, now: i96) i96 {
        if (t.flushing or !front.paced or t.last == null) return now;
        const spread = @divTrunc(t.hold_ns, @as(i96, @intCast(@max(t.paced, 1))));
        const at = @max(front.arrived, t.last.? + spread);
        return @min(at, front.arrived + @divTrunc(t.hold_ns * longest_num, longest_den));
    }
};

pub const Playout = struct {
    gpa: Allocator,
    io: std.Io,
    out: chat.Sink,
    cancelled: ?Cancelled,
    timing: Timing,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    queue: std.ArrayList(Item) = .empty,
    /// The queued deltas' own copies: a delta's text may lie in a buffer the reply moves on (the decode buffer grows).
    copies: std.heap.ArenaAllocator = undefined,
    head: usize = 0,
    closed: bool = false,
    gone: bool = false,
    thread: ?std.Thread = null,

    pub const Cancelled = struct { ctx: *anyopaque, check: *const fn (ctx: *anyopaque) bool };

    pub fn start(p: *Playout, gpa: Allocator, io: std.Io, hold_ms: u32, out: chat.Sink, cancelled: ?Cancelled) !void {
        p.* = .{ .gpa = gpa, .io = io, .out = out, .cancelled = cancelled, .timing = .{ .hold_ns = @as(i96, hold_ms) * std.time.ns_per_ms }, .copies = .init(gpa) };
        p.thread = try std.Thread.spawn(.{}, run, .{p});
    }

    /// The sink the reply writes to: it queues.
    pub fn sink(p: *Playout) chat.Sink {
        return .{ .ctx = p, .call = put };
    }

    fn now(p: *const Playout) i96 {
        return std.Io.Clock.awake.now(p.io).toNanoseconds();
    }

    /// A text delta is paced; calls, keepalives and the like keep their place and go out as soon as they are first.
    fn pacedDelta(delta: Value) bool {
        return switch (delta) {
            .string => true,
            .object => delta.get("content") != null or delta.get("reasoning_content") != null,
            else => false,
        };
    }

    fn put(ctx: *anyopaque, delta: Value) error{Closed}!void {
        const p: *Playout = @ptrCast(@alignCast(ctx));
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        if (p.gone) return error.Closed;
        if (p.closed) return;
        const paced = pacedDelta(delta);
        const own = json.deepCopy(p.copies.allocator(), delta) catch return error.Closed;
        p.queue.append(p.gpa, .{ .delta = own, .paced = paced, .arrived = p.now() }) catch return error.Closed;
        p.timing.paced += @intFromBool(paced);
        p.cond.signal(p.io);
    }

    /// Everything queued goes out at once from now on (the reply's last round).
    pub fn flush(p: *Playout) void {
        p.mutex.lockUncancelable(p.io);
        defer p.mutex.unlock(p.io);
        p.timing.flushing = true;
        p.cond.signal(p.io);
    }

    /// Sends everything still queued at once and stops the writer (the reply's end follows).
    pub fn close(p: *Playout) void {
        p.mutex.lockUncancelable(p.io);
        p.timing.flushing = true;
        p.closed = true;
        p.cond.signal(p.io);
        p.mutex.unlock(p.io);
        p.join();
    }

    /// Drops what is queued and stops the writer.
    pub fn abort(p: *Playout) void {
        p.mutex.lockUncancelable(p.io);
        p.queue.clearRetainingCapacity();
        p.head = 0;
        p.timing.paced = 0;
        p.closed = true;
        p.cond.signal(p.io);
        p.mutex.unlock(p.io);
        p.join();
    }

    fn join(p: *Playout) void {
        if (p.thread) |t| t.join();
        p.thread = null;
        p.queue.deinit(p.gpa);
        p.queue = .empty;
        p.copies.deinit();
        p.copies = .init(p.gpa);
    }

    fn run(p: *Playout) void {
        while (true) {
            p.mutex.lockUncancelable(p.io);
            var item: Item = undefined;
            var held = false;
            while (true) {
                if (p.head >= p.queue.items.len) {
                    if (p.closed) {
                        p.mutex.unlock(p.io);
                        return;
                    }
                    p.cond.waitUncancelable(p.io, &p.mutex);
                    continue;
                }
                const t = p.now();
                const at = p.timing.due(p.queue.items[p.head], t);
                if (at <= t) break;
                // a new token shortens the wait; a flush ends it
                p.cond.waitTimeout(p.io, &p.mutex, .{ .duration = .{ .raw = .fromNanoseconds(@intCast(at - t)), .clock = .awake } }) catch {};
            }
            item = p.queue.items[p.head];
            p.head += 1;
            if (p.head == p.queue.items.len) {
                p.queue.clearRetainingCapacity();
                p.head = 0;
            }
            p.timing.paced -= @intFromBool(item.paced);
            held = item.paced and !p.timing.flushing and p.now() > item.arrived;
            p.mutex.unlock(p.io);
            var ok = true;
            if (held) if (p.cancelled) |c| if (c.check(c.ctx)) {
                ok = false;
            };
            if (ok) p.out.call(p.out.ctx, item.delta) catch {
                ok = false;
            };
            p.mutex.lockUncancelable(p.io);
            p.timing.last = p.now();
            if (!ok) {
                p.gone = true;
                p.queue.clearRetainingCapacity();
                p.head = 0;
                p.timing.paced = 0;
                p.mutex.unlock(p.io);
                return;
            }
            p.mutex.unlock(p.io);
        }
    }
};

fn textItem(arrived: i96) Item {
    return .{ .delta = .{ .string = "x" }, .paced = true, .arrived = arrived };
}

test "the next token goes out hold / queued after the previous one, never before it arrived" {
    var t: Timing = .{ .hold_ns = 400 * std.time.ns_per_ms, .paced = 4 };
    // nothing went out yet: the first goes at once
    try std.testing.expectEqual(@as(i96, 1000), t.due(textItem(0), 1000));
    t.last = 1000 * std.time.ns_per_ms;
    // 4 queued: 100 ms after the previous
    try std.testing.expectEqual(@as(i96, 1100 * std.time.ns_per_ms), t.due(textItem(500 * std.time.ns_per_ms), 1050 * std.time.ns_per_ms));
    // a token that arrived later waits for its own arrival
    try std.testing.expectEqual(@as(i96, 1300 * std.time.ns_per_ms), t.due(textItem(1300 * std.time.ns_per_ms), 1050 * std.time.ns_per_ms));
    // a long queue: the spread is short
    t.paced = 40;
    try std.testing.expectEqual(@as(i96, 1010 * std.time.ns_per_ms), t.due(textItem(900 * std.time.ns_per_ms), 1005 * std.time.ns_per_ms));
    // one queued: the whole hold
    t.paced = 1;
    t.last = 1000 * std.time.ns_per_ms;
    // but no token is held longer than 2.5 times the buffer after it arrived
    try std.testing.expectEqual(@as(i96, 1000 * std.time.ns_per_ms), t.due(textItem(0), 900 * std.time.ns_per_ms));
    // an event that is not text goes out at once, and a flush ends every wait
    const call: Item = .{ .delta = .null, .paced = false, .arrived = 0 };
    try std.testing.expectEqual(@as(i96, 5), t.due(call, 5));
    t.flushing = true;
    try std.testing.expectEqual(@as(i96, 5), t.due(textItem(0), 5));
}

const Collect = struct {
    io: std.Io,
    stamps: [32]i96 = undefined,
    order: [32]u8 = undefined,
    n: usize = 0,

    fn call(ctx: *anyopaque, delta: Value) error{Closed}!void {
        const c: *Collect = @ptrCast(@alignCast(ctx));
        if (c.n >= c.order.len) return error.Closed;
        c.stamps[c.n] = std.Io.Clock.awake.now(c.io).toNanoseconds();
        c.order[c.n] = delta.string[0];
        c.n += 1;
    }
};

test "queued tokens are spread over the buffer in order, and a close sends the rest at once" {
    var c: Collect = .{ .io = std.testing.io };
    var p: Playout = undefined;
    try p.start(std.testing.allocator, std.testing.io, 200, .{ .ctx = &c, .call = Collect.call }, null);
    const start = std.Io.Clock.awake.now(std.testing.io).toNanoseconds();
    const sink = p.sink();
    const letters = "abcde";
    for (0..5) |i| try sink.call(sink.ctx, .{ .string = letters[i .. i + 1] });
    // all five arrived together: the first goes at once, the next after hold / 4, then hold / 3 ... the last after the whole
    // buffer, but never past 2.5 times it
    std.Io.sleep(std.testing.io, .fromMilliseconds(800), .awake) catch {};
    p.close();
    try std.testing.expectEqual(@as(usize, 5), c.n);
    try std.testing.expectEqualStrings("abcde", c.order[0..5]);
    try std.testing.expect(c.stamps[0] - start < 30 * std.time.ns_per_ms);
    try std.testing.expect(c.stamps[1] - c.stamps[0] >= 40 * std.time.ns_per_ms); // spread, not a burst
    try std.testing.expect(c.stamps[4] - c.stamps[0] >= 300 * std.time.ns_per_ms);
    try std.testing.expect(c.stamps[4] - start < 600 * std.time.ns_per_ms); // within 2.5 times the buffer
}

test "a flush sends what is queued at once, and a client that left stops the writer" {
    var c: Collect = .{ .io = std.testing.io };
    var p: Playout = undefined;
    try p.start(std.testing.allocator, std.testing.io, 5000, .{ .ctx = &c, .call = Collect.call }, null);
    const sink = p.sink();
    for (0..4) |_| try sink.call(sink.ctx, .{ .string = "t" });
    p.flush();
    std.Io.sleep(std.testing.io, .fromMilliseconds(100), .awake) catch {};
    p.close();
    try std.testing.expectEqual(@as(usize, 4), c.n);
    // a sink that fails: the writer stops and the next put says the client is gone
    var full: Collect = .{ .io = std.testing.io, .n = 32 };
    var q: Playout = undefined;
    try q.start(std.testing.allocator, std.testing.io, 5, .{ .ctx = &full, .call = Collect.call }, null);
    const s2 = q.sink();
    try s2.call(s2.ctx, .{ .string = "z" });
    std.Io.sleep(std.testing.io, .fromMilliseconds(100), .awake) catch {};
    try std.testing.expectError(error.Closed, s2.call(s2.ctx, .{ .string = "z" }));
    q.close();
}
