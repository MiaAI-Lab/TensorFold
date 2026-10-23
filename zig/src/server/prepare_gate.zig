//! The memory gate before a prompt is rendered and tokenized (rank 0's host side). A conversation's text is rendered
//! and tokenized three times with the prompt cache on (prompt.prepare: the prompt, its history and the rewind probe,
//! each after GLM's history repairs), so a resumed conversation of a million tokens needs memory in proportion to its
//! text while it prepares, and on a unified-memory machine (a GB10, where the KV pool leaves a few GiB free) eight of
//! them at once can take MemAvailable to zero. Each prepare reserves `per_byte` bytes for every byte of its text and
//! goes when MemAvailable, less what the ones in flight reserved, stays at the floor or over with its own reservation.
//! Prepares that cannot go wait in arrival order: only the oldest waiter may go, so a large prepare is never starved by
//! a stream of small ones that would each fit beside the ones in flight (they queue behind it). A prepare that arrives
//! with none in flight and no earlier waiter always goes at once, as before the order (no prepare's end could free
//! memory for it). A waiter waits for memory at most `TENSORFOLD_PREPARE_WAIT_MS`; past it, it goes as soon as none is
//! in flight (it can never wait on itself, so the gate never deadlocks; 0 gives that rule at once). At most
//! `TENSORFOLD_PREPARE_MAX_WAITING` prepares wait; one more is refused (`error.Full`, which chat.prepare answers as a
//! capacity refusal: 429 with `Retry-After` under the GLM recipe, else 503). The gate only ever turns concurrent
//! prepares into queued ones; it changes no reply. Off with a floor of 0, and where MemAvailable cannot be read (no
//! /proc/meminfo).
const std = @import("std");
const json = @import("json");
const log = @import("log.zig");
const Value = json.Value;

/// The variables that set it (serve.zig reads them; capabilities lists them beside cli.env).
pub const env = [_][]const u8{ "TENSORFOLD_PREPARE_FLOOR_GIB", "TENSORFOLD_PREPARE_WAIT_MS", "TENSORFOLD_PREPARE_MAX_WAITING" };
/// TENSORFOLD_PREPARE_FLOOR_GIB unset: the MemAvailable a prepare leaves at least, in GiB.
pub const default_floor_gib: f64 = 3;
/// TENSORFOLD_PREPARE_WAIT_MS unset: how long a waiter (one queued behind prepares in flight) waits for memory before
/// it goes with none in flight. Long enough for what frees rank 0's memory beside the prepares (a restore's staging, a
/// finished request's arena), short enough that on a GB10 already under its floor a queued request waits seconds, not
/// for good. A prepare that finds the gate empty never waits.
pub const default_wait_ms: u64 = 5000;
/// TENSORFOLD_PREPARE_MAX_WAITING unset: the prepares that may wait at once (0: no bound). Every lane's request and
/// a queue of the same size several times over; past it a client is told to retry rather than held unseen.
pub const default_max_waiting: u32 = 64;
/// Bytes reserved a text byte, measured on a long GLM agent conversation (prepare_gate_test.zig): a render's or a
/// tokenization's transient scratch (~16-19 and ~12-13 bytes a text byte, one at a time) over what the request holds
/// meanwhile (its body and parsed messages, ~3; three renders' repaired and normalized messages, texts and id lists,
/// ~11-25 by the arena's steps): 30-46, with room to spare. DeepSeek's two renders take 24.
pub const default_per_byte: u64 = 48;
/// A prepare refused for the waiter bound: the request's capacity refusal (errors.Kind.capacity).
pub const full_message = "too many prompts are waiting for memory to be prepared; retry shortly";
/// How often the oldest waiter reads MemAvailable again (memory frees outside the gate too, so it polls).
const poll_ms = 10;

pub const Gate = struct {
    mutex: std.Io.Mutex = .init,
    floor: u64 = 0, // bytes; 0: off
    per_byte: u64 = default_per_byte,
    /// How long a waiter waits for memory before it goes with none in flight (ns; 0: at once).
    max_wait_ns: u64 = default_wait_ms * std.time.ns_per_ms,
    /// The waiters allowed at once (0: no bound).
    max_waiting: u32 = default_max_waiting,
    reserved: u64 = 0,
    inflight: u32 = 0,
    /// The waiters in arrival order (each a waiting prepare's own stack node), and how many there are.
    first: ?*Waiter = null,
    last: ?*Waiter = null,
    waiting: u32 = 0,
    /// Prepares that had to wait, the longest wait (ns), those that went with none in flight past the maximum wait,
    /// and those refused for the waiter bound: for the log and the tests.
    waited: u64 = 0,
    longest_ns: u64 = 0,
    overdue: u64 = 0,
    refused: u64 = 0,
    /// MemAvailable in bytes, null when it cannot be read (the gate lets everything through then).
    avail: *const fn (io: std.Io) ?u64 = memAvailable,

    pub const Ticket = struct { need: u64 = 0, held: bool = false };
    pub const Error = error{ Cancelled, Full };

    const Waiter = struct {
        bytes: u64,
        since: i96,
        prev: ?*Waiter = null,
        next: ?*Waiter = null,
    };

    /// A gate leaving `floor_gib` GiB (0: off), with the default wait and waiter bound.
    pub fn init(floor_gib: f64) Gate {
        return .{ .floor = if (floor_gib > 0) @intFromFloat(floor_gib * (1 << 30)) else 0 };
    }

    /// A gate with its three settings (TENSORFOLD_PREPARE_FLOOR_GIB, _WAIT_MS, _MAX_WAITING).
    pub fn configured(floor_gib: f64, wait_ms: u64, max_waiting: u32) Gate {
        var g = init(floor_gib);
        g.max_wait_ns = std.math.mul(u64, wait_ms, std.time.ns_per_ms) catch std.math.maxInt(u64);
        g.max_waiting = max_waiting;
        return g;
    }

    /// The reservation of a prepare of `text_bytes` bytes of text.
    pub fn need(g: *const Gate, text_bytes: usize) u64 {
        return std.math.mul(u64, text_bytes, g.per_byte) catch std.math.maxInt(u64);
    }

    /// Waits its turn (arrival order) until a prepare reserving `bytes` fits, or, past the maximum wait, until none is
    /// in flight; then holds its reservation until leave. error.Cancelled when `gone` says the client left while it
    /// waited (it leaves the queue, so the next waiter moves up); error.Full, at once, when the waiter bound is reached.
    pub fn enter(g: *Gate, io: std.Io, bytes: u64, gone: anytype) Error!Ticket {
        if (g.floor == 0) return .{};
        var me: Waiter = .{ .bytes = bytes, .since = std.Io.Clock.awake.now(io).toNanoseconds() };
        g.mutex.lockUncancelable(io);
        // uncontended (none in flight, no earlier waiter): it goes at once, as before the order, even under the floor:
        // no prepare's end can free memory for it, so waiting would only add latency
        if (g.first == null and (g.inflight == 0 or g.goes(io, &me) != .wait)) return g.take(io, &me, null);
        if (g.max_waiting > 0 and g.waiting >= g.max_waiting) {
            g.refused += 1;
            const n = g.waiting;
            g.mutex.unlock(io);
            log.line("prepare refused: {d} prepares already wait for memory (TENSORFOLD_PREPARE_MAX_WAITING={d}); the client is told to retry", .{ n, g.max_waiting });
            return error.Full;
        }
        g.push(&me);
        const ahead = g.waiting - 1;
        const queued = g.inflight;
        const reserved = g.reserved;
        g.mutex.unlock(io);
        log.line("prepare waits for memory: {d} MiB reserved for it, {d} prepares in flight hold {d} MiB, {d} wait before it, floor {d} MiB", .{ bytes >> 20, queued, reserved >> 20, ahead, g.floor >> 20 });
        while (true) {
            if (gone.check()) {
                g.mutex.lockUncancelable(io);
                g.unlink(&me);
                g.mutex.unlock(io);
                return error.Cancelled;
            }
            std.Io.sleep(io, .fromMilliseconds(poll_ms), .awake) catch {};
            g.mutex.lockUncancelable(io);
            if (g.first == &me) {
                const how = g.goes(io, &me);
                if (how != .wait) {
                    g.unlink(&me);
                    return g.take(io, &me, how);
                }
            }
            g.mutex.unlock(io);
        }
    }

    const Go = enum { wait, fits, overdue };

    /// (locked) Whether `w` may go now: .fits when its reservation leaves MemAvailable, less every reservation in
    /// flight, at the floor or over; .overdue when it does not but `w` has waited the maximum and none is in flight.
    fn goes(g: *Gate, io: std.Io, w: *const Waiter) Go {
        if (g.avail(io)) |avail| {
            if (avail >= g.reserved +| w.bytes +| g.floor) return .fits;
        } else return .fits;
        if (g.inflight == 0 and waitedNs(io, w) >= g.max_wait_ns) return .overdue;
        return .wait;
    }

    fn waitedNs(io: std.Io, w: *const Waiter) u64 {
        return @intCast(@max(0, std.Io.Clock.awake.now(io).toNanoseconds() - w.since));
    }

    /// (locked; unlocks) `w`'s reservation is held from here until leave; `how` it went after waiting (null: at once).
    fn take(g: *Gate, io: std.Io, w: *const Waiter, how: ?Go) Ticket {
        g.reserved +|= w.bytes;
        g.inflight += 1;
        var ns: u64 = 0;
        if (how) |h| {
            ns = waitedNs(io, w);
            g.waited += 1;
            g.longest_ns = @max(g.longest_ns, ns);
            if (h == .overdue) g.overdue += 1;
        }
        g.mutex.unlock(io);
        if (how == .overdue) log.line("prepare goes after {d} ms with none in flight (TENSORFOLD_PREPARE_WAIT_MS): its {d} MiB would take MemAvailable under the floor", .{ ns / std.time.ns_per_ms, w.bytes >> 20 });
        return .{ .need = w.bytes, .held = true };
    }

    /// (locked) `w` joins the queue's end.
    fn push(g: *Gate, w: *Waiter) void {
        w.prev = g.last;
        w.next = null;
        if (g.last) |l| l.next = w else g.first = w;
        g.last = w;
        g.waiting += 1;
    }

    /// (locked) `w` leaves the queue, wherever it is.
    fn unlink(g: *Gate, w: *Waiter) void {
        if (w.prev) |p| p.next = w.next else g.first = w.next;
        if (w.next) |n| n.prev = w.prev else g.last = w.prev;
        w.prev = null;
        w.next = null;
        g.waiting -= 1;
    }

    pub fn leave(g: *Gate, io: std.Io, t: Ticket) void {
        if (!t.held) return;
        g.mutex.lockUncancelable(io);
        defer g.mutex.unlock(io);
        g.reserved -|= t.need;
        g.inflight -|= 1;
    }

    /// The waiters now (for the tests).
    pub fn waiters(g: *Gate, io: std.Io) u32 {
        g.mutex.lockUncancelable(io);
        defer g.mutex.unlock(io);
        return g.waiting;
    }
};

/// TENSORFOLD_PREPARE_FLOOR_GIB: GiB, 0 turns the gate off; unset or blank: the default.
pub fn floorGib(raw: ?[]const u8) error{BadPrepareFloor}!f64 {
    const t = std.mem.trim(u8, raw orelse "", " \t\r\n");
    if (t.len == 0) return default_floor_gib;
    const v = std.fmt.parseFloat(f64, t) catch return error.BadPrepareFloor;
    if (!(v >= 0) or v > 1 << 20) return error.BadPrepareFloor;
    return v;
}

/// TENSORFOLD_PREPARE_WAIT_MS: milliseconds (0: a waiter goes as soon as none is in flight); unset or blank: the
/// default. At most a day.
pub fn waitMs(raw: ?[]const u8) error{BadPrepareWait}!u64 {
    const t = std.mem.trim(u8, raw orelse "", " \t\r\n");
    if (t.len == 0) return default_wait_ms;
    const v = std.fmt.parseInt(u64, t, 10) catch return error.BadPrepareWait;
    if (v > 24 * 3600 * 1000) return error.BadPrepareWait;
    return v;
}

/// TENSORFOLD_PREPARE_MAX_WAITING: a count (0: no bound); unset or blank: the default.
pub fn maxWaiting(raw: ?[]const u8) error{BadPrepareWaiting}!u32 {
    const t = std.mem.trim(u8, raw orelse "", " \t\r\n");
    if (t.len == 0) return default_max_waiting;
    return std.fmt.parseInt(u32, t, 10) catch error.BadPrepareWaiting;
}

/// The bytes of text a request renders: every string in `v` (message contents, tool specs, names), keys included.
pub fn textBytes(v: Value) usize {
    return switch (v) {
        .string => |s| s.len,
        .int => |s| s.len,
        .array => |items| blk: {
            var n: usize = 0;
            for (items) |x| n +|= textBytes(x);
            break :blk n;
        },
        .object => |o| blk: {
            var n: usize = 0;
            for (o.keys(), o.values()) |k, x| n +|= k.len +| textBytes(x);
            break :blk n;
        },
        else => 8,
    };
}

/// /proc/meminfo's MemAvailable in bytes (null where there is none, or it does not parse).
pub fn memAvailable(io: std.Io) ?u64 {
    var buf: [4096]u8 = undefined;
    const text = std.Io.Dir.cwd().readFile(io, "/proc/meminfo", &buf) catch return null;
    return parseAvailable(text);
}

pub fn parseAvailable(text: []const u8) ?u64 {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "MemAvailable:")) continue;
        var it = std.mem.tokenizeAny(u8, line["MemAvailable:".len..], " \t");
        const n = std.fmt.parseInt(u64, it.next() orelse return null, 10) catch return null;
        const unit = it.next() orelse return null;
        if (!std.mem.eql(u8, unit, "kB")) return null;
        return std.math.mul(u64, n, 1024) catch null;
    }
    return null;
}

const Never = struct {
    pub fn check(_: Never) bool {
        return false;
    }
};

test "MemAvailable from /proc/meminfo's text" {
    try std.testing.expectEqual(@as(?u64, 6 << 30), parseAvailable("MemTotal:       127535424 kB\nMemFree:         1048576 kB\nMemAvailable:    6291456 kB\n"));
    try std.testing.expectEqual(@as(?u64, null), parseAvailable("MemTotal: 1 kB\n"));
    try std.testing.expectEqual(@as(?u64, null), parseAvailable("MemAvailable: 12 MB\n"));
    try std.testing.expectEqual(@as(?u64, null), parseAvailable("MemAvailable: x kB\n"));
}

test "the floor's setting" {
    try std.testing.expectEqual(default_floor_gib, try floorGib(null));
    try std.testing.expectEqual(default_floor_gib, try floorGib(" "));
    try std.testing.expectEqual(@as(f64, 0), try floorGib("0"));
    try std.testing.expectEqual(@as(f64, 2.5), try floorGib("2.5"));
    try std.testing.expectError(error.BadPrepareFloor, floorGib("-1"));
    try std.testing.expectError(error.BadPrepareFloor, floorGib("nan"));
    try std.testing.expectError(error.BadPrepareFloor, floorGib("lots"));
    try std.testing.expectEqual(@as(u64, 0), Gate.init(0).floor);
    try std.testing.expectEqual(@as(u64, 3 << 30), Gate.init(3).floor);
    try std.testing.expectEqual(default_wait_ms, try waitMs(null));
    try std.testing.expectEqual(@as(u64, 0), try waitMs(" 0 "));
    try std.testing.expectEqual(@as(u64, 250), try waitMs("250"));
    try std.testing.expectError(error.BadPrepareWait, waitMs("-1"));
    try std.testing.expectError(error.BadPrepareWait, waitMs("1.5"));
    try std.testing.expectError(error.BadPrepareWait, waitMs("86400001"));
    try std.testing.expectEqual(default_max_waiting, try maxWaiting(""));
    try std.testing.expectEqual(@as(u32, 0), try maxWaiting("0"));
    try std.testing.expectEqual(@as(u32, 8), try maxWaiting("8"));
    try std.testing.expectError(error.BadPrepareWaiting, maxWaiting("-8"));
    try std.testing.expectError(error.BadPrepareWaiting, maxWaiting("many"));
    const c = Gate.configured(2, 250, 8);
    try std.testing.expectEqual(@as(u64, 2 << 30), c.floor);
    try std.testing.expectEqual(@as(u64, 250 * std.time.ns_per_ms), c.max_wait_ns);
    try std.testing.expectEqual(@as(u32, 8), c.max_waiting);
}

test "a request's text bytes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const parsed = try json.parse(a, "[{\"role\":\"user\",\"content\":\"hello\"},{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"hi there\"}]}]");
    try std.testing.expect(parsed == .ok);
    // keys and strings: role+user, content+hello; role+assistant, content, type+text, text+hi there
    try std.testing.expectEqual(@as(usize, 4 + 4 + 7 + 5 + 4 + 9 + 7 + 4 + 4 + 4 + 8), textBytes(parsed.ok));
}

var fake_avail: std.atomic.Value(u64) = .init(0);
fn fakeAvail(_: std.Io) ?u64 {
    return fake_avail.load(.acquire);
}
fn noAvail(_: std.Io) ?u64 {
    return null;
}

test "with a wait of 0 the first prepare always goes; a second waits while it would take MemAvailable under the floor" {
    const io = std.testing.io;
    var g = Gate.configured(3, 0, default_max_waiting);
    g.avail = fakeAvail;
    fake_avail.store(4 << 30, .release);
    // the first goes whatever MemAvailable says: none is in flight, and it need not wait for memory (wait 0)
    const a = try g.enter(io, 2 << 30, Never{});
    try std.testing.expect(a.held);
    // a second of 1 GiB fits only with 2 (reserved) + 1 + 3 (floor) = 6 GiB available
    const Waiter = struct {
        fn run(gate: *Gate, done: *std.atomic.Value(bool), t: *Gate.Ticket) void {
            t.* = gate.enter(std.testing.io, 1 << 30, Never{}) catch unreachable;
            done.store(true, .release);
        }
    };
    var done: std.atomic.Value(bool) = .init(false);
    var second: Gate.Ticket = .{};
    const th = try std.Thread.spawn(.{}, Waiter.run, .{ &g, &done, &second });
    std.Io.sleep(io, .fromMilliseconds(5 * poll_ms), .awake) catch {};
    try std.testing.expect(!done.load(.acquire));
    // the first leaves: the second goes (none in flight)
    g.leave(io, a);
    th.join();
    try std.testing.expect(done.load(.acquire) and second.held);
    try std.testing.expectEqual(@as(u64, 1), g.waited);
    try std.testing.expectEqual(@as(u64, 1 << 30), g.reserved);
    // with room, a third goes beside it at once
    fake_avail.store(16 << 30, .release);
    const third = try g.enter(io, 1 << 30, Never{});
    try std.testing.expectEqual(@as(u32, 2), g.inflight);
    g.leave(io, third);
    g.leave(io, second);
    try std.testing.expectEqual(@as(u64, 0), g.reserved);
    try std.testing.expectEqual(@as(u32, 0), g.inflight);
}

test "a waiting prepare whose client leaves is cancelled; off and unreadable gates never wait" {
    const io = std.testing.io;
    var g = Gate.configured(3, 0, default_max_waiting);
    g.avail = fakeAvail;
    fake_avail.store(1 << 30, .release);
    const a = try g.enter(io, 1 << 30, Never{});
    const Left = struct {
        pub fn check(_: @This()) bool {
            return true;
        }
    };
    try std.testing.expectError(error.Cancelled, g.enter(io, 1 << 30, Left{}));
    try std.testing.expectEqual(@as(u32, 1), g.inflight);
    try std.testing.expectEqual(@as(u32, 0), g.waiting);
    try std.testing.expect(g.first == null and g.last == null);
    g.leave(io, a);
    var off = Gate.init(0);
    off.avail = fakeAvail;
    const x = try off.enter(io, 1 << 40, Never{});
    const y = try off.enter(io, 1 << 40, Never{});
    try std.testing.expect(!x.held and !y.held);
    off.leave(io, x);
    var blind = Gate.init(3);
    blind.avail = noAvail;
    const p = try blind.enter(io, 1 << 40, Never{});
    const q = try blind.enter(io, 1 << 40, Never{});
    blind.leave(io, p);
    blind.leave(io, q);
    try std.testing.expectEqual(@as(u32, 0), blind.inflight);
}

/// A prepare that takes its place in a test's order when it goes, then leaves at once.
const Ordered = struct {
    fn run(gate: *Gate, bytes: u64, order: *std.atomic.Value(u32), at: *u32) void {
        const t = gate.enter(std.testing.io, bytes, Never{}) catch unreachable;
        at.* = order.fetchAdd(1, .acq_rel);
        gate.leave(std.testing.io, t);
    }
};

test "a large prepare behind a stream of small ones goes in its turn, before every one that came after it" {
    const io = std.testing.io;
    var g = Gate.init(3);
    g.avail = fakeAvail;
    // 3 GiB over the floor: a small one (1 GiB) fits beside another, the large one (2.5 GiB) fits only alone
    fake_avail.store(6 << 30, .release);
    const first = try g.enter(io, 1 << 30, Never{});
    var order: std.atomic.Value(u32) = .init(0);
    var large_at: u32 = 99;
    const large = try std.Thread.spawn(.{}, Ordered.run, .{ &g, 5 << 29, &order, &large_at });
    std.Io.sleep(io, .fromMilliseconds(5 * poll_ms), .awake) catch {};
    // small ones keep arriving while the first is in flight: each would fit beside it
    var small_at: [4]u32 = .{ 99, 99, 99, 99 };
    var small: [4]std.Thread = undefined;
    for (&small, &small_at) |*th, *at| {
        th.* = try std.Thread.spawn(.{}, Ordered.run, .{ &g, 1 << 30, &order, at });
        std.Io.sleep(io, .fromMilliseconds(5 * poll_ms), .awake) catch {};
    }
    g.leave(io, first);
    large.join();
    for (small) |th| th.join();
    try std.testing.expectEqual(@as(u32, 0), large_at);
    for (small_at) |at| try std.testing.expect(at > large_at and at < 5);
    try std.testing.expectEqual(@as(u32, 0), g.inflight);
    try std.testing.expectEqual(@as(u64, 0), g.reserved);
}

/// A client that leaves when the test says so.
const Flag = struct {
    left: *std.atomic.Value(bool),
    pub fn check(f: Flag) bool {
        return f.left.load(.acquire);
    }
};

/// A prepare on its own thread: its outcome, and its place in the order when it went (it leaves at once then).
const Prep = struct {
    bytes: u64,
    left: std.atomic.Value(bool) = .init(false),
    done: std.atomic.Value(bool) = .init(false),
    result: Gate.Error!void = {},
    at: u32 = 99,

    fn run(p: *Prep, gate: *Gate, order: *std.atomic.Value(u32)) void {
        defer p.done.store(true, .release);
        const t = gate.enter(std.testing.io, p.bytes, Flag{ .left = &p.left }) catch |e| {
            p.result = e;
            return;
        };
        p.at = order.fetchAdd(1, .acq_rel);
        gate.leave(std.testing.io, t);
    }
};

/// Polls until the gate has `n` waiters (at most 2 s).
fn untilWaiting(g: *Gate, n: u32) !void {
    for (0..200) |_| {
        if (g.waiters(std.testing.io) == n) return;
        std.Io.sleep(std.testing.io, .fromMilliseconds(poll_ms), .awake) catch {};
    }
    return error.TestUnexpectedResult;
}

test "a waiter queued behind a prepare in flight waits for memory at most TENSORFOLD_PREPARE_WAIT_MS, and then goes only with none in flight" {
    const io = std.testing.io;
    var g = Gate.configured(3, 100, default_max_waiting);
    g.avail = fakeAvail;
    var order: std.atomic.Value(u32) = .init(0);
    // past the maximum, a waiter still waits while the one in flight is there, and goes once it has left
    fake_avail.store(16 << 30, .release);
    const a = try g.enter(io, 1 << 20, Never{});
    fake_avail.store(1 << 30, .release); // under the floor whatever goes
    var b: Prep = .{ .bytes = 1 << 20 };
    const tb = try std.Thread.spawn(.{}, Prep.run, .{ &b, &g, &order });
    std.Io.sleep(io, .fromMilliseconds(300), .awake) catch {};
    try std.testing.expect(!b.done.load(.acquire));
    g.leave(io, a);
    tb.join();
    try b.result;
    try std.testing.expectEqual(@as(u64, 1), g.waited);
    try std.testing.expectEqual(@as(u64, 1), g.overdue);
    // the one in flight leaves at once: the waiter, memory still short, waits out the maximum (never for good)
    fake_avail.store(16 << 30, .release);
    const c = try g.enter(io, 1 << 20, Never{});
    fake_avail.store(1 << 30, .release);
    var d: Prep = .{ .bytes = 1 << 20 };
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    const td = try std.Thread.spawn(.{}, Prep.run, .{ &d, &g, &order });
    try untilWaiting(&g, 1);
    g.leave(io, c);
    td.join();
    const took: u64 = @intCast(std.Io.Clock.awake.now(io).toNanoseconds() - t0);
    try d.result;
    try std.testing.expect(took >= 100 * std.time.ns_per_ms and took < 5 * std.time.ns_per_s);
    try std.testing.expectEqual(@as(u64, 2), g.overdue);
    // uncontended again: at once, not counted as a wait
    const e = try g.enter(io, 1 << 20, Never{});
    try std.testing.expectEqual(@as(u64, 2), g.waited);
    g.leave(io, e);
    try std.testing.expectEqual(@as(u32, 0), g.inflight);
    try std.testing.expectEqual(@as(u64, 0), g.reserved);
}

test "past TENSORFOLD_PREPARE_MAX_WAITING waiters a prepare is refused at once, and a waiter that leaves makes room" {
    const io = std.testing.io;
    var g = Gate.configured(3, 60_000, 2);
    g.avail = fakeAvail;
    fake_avail.store(16 << 30, .release);
    const a = try g.enter(io, 1 << 30, Never{});
    fake_avail.store(1 << 30, .release); // nothing more fits
    var order: std.atomic.Value(u32) = .init(0);
    var b: Prep = .{ .bytes = 1 << 30 };
    var c: Prep = .{ .bytes = 1 << 30 };
    const tb = try std.Thread.spawn(.{}, Prep.run, .{ &b, &g, &order });
    try untilWaiting(&g, 1);
    const tc = try std.Thread.spawn(.{}, Prep.run, .{ &c, &g, &order });
    try untilWaiting(&g, 2);
    // a third waiter: refused, without waiting
    try std.testing.expectError(error.Full, g.enter(io, 1 << 20, Never{}));
    try std.testing.expectEqual(@as(u64, 1), g.refused);
    // b's client leaves: b is cancelled, and a new one may wait in its place
    b.left.store(true, .release);
    tb.join();
    try std.testing.expectError(error.Cancelled, b.result);
    try std.testing.expectEqual(@as(u32, 1), g.waiters(io));
    var d: Prep = .{ .bytes = 1 << 30 };
    const td = try std.Thread.spawn(.{}, Prep.run, .{ &d, &g, &order });
    try untilWaiting(&g, 2);
    // memory comes back and a leaves: c, then d, in arrival order
    fake_avail.store(16 << 30, .release);
    g.leave(io, a);
    tc.join();
    td.join();
    try c.result;
    try d.result;
    try std.testing.expect(c.at < d.at);
    try std.testing.expectEqual(@as(u64, 1), g.refused);
    try std.testing.expectEqual(@as(u32, 0), g.inflight);
    try std.testing.expectEqual(@as(u32, 0), g.waiting);
}

test "a waiter whose client leaves gives up its place: the one behind it moves up and goes" {
    const io = std.testing.io;
    var g = Gate.configured(3, 60_000, default_max_waiting);
    g.avail = fakeAvail;
    fake_avail.store(6 << 30, .release); // 3 GiB over the floor
    const a = try g.enter(io, 1 << 30, Never{});
    var order: std.atomic.Value(u32) = .init(0);
    // a large one (2.5 GiB) waits beside a; a small one (1 GiB) would fit beside a, but waits behind it
    var large: Prep = .{ .bytes = 5 << 29 };
    var small: Prep = .{ .bytes = 1 << 30 };
    const tl = try std.Thread.spawn(.{}, Prep.run, .{ &large, &g, &order });
    try untilWaiting(&g, 1);
    const ts = try std.Thread.spawn(.{}, Prep.run, .{ &small, &g, &order });
    try untilWaiting(&g, 2);
    std.Io.sleep(io, .fromMilliseconds(5 * poll_ms), .awake) catch {};
    try std.testing.expect(!small.done.load(.acquire));
    // the large one's client leaves: the small one goes beside a, which is still in flight
    large.left.store(true, .release);
    tl.join();
    ts.join();
    try std.testing.expectError(error.Cancelled, large.result);
    try small.result;
    try std.testing.expectEqual(@as(u32, 0), small.at);
    try std.testing.expectEqual(@as(u32, 1), g.inflight);
    g.leave(io, a);
    try std.testing.expectEqual(@as(u32, 0), g.inflight);
    try std.testing.expectEqual(@as(u64, 0), g.reserved);
    try std.testing.expect(g.first == null and g.last == null);
}

test "uncontended: with nothing in flight and no waiter a prepare goes at once, even under the floor, and nothing counts it as waiting" {
    const io = std.testing.io;
    var g = Gate.init(3); // the default wait (5000 ms) and waiter bound
    g.avail = fakeAvail;
    fake_avail.store(1 << 30, .release); // MemAvailable under the 3 GiB floor
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    const a = try g.enter(io, 1 << 30, Never{});
    const took: u64 = @intCast(std.Io.Clock.awake.now(io).toNanoseconds() - t0);
    try std.testing.expect(a.held);
    try std.testing.expect(took < 50 * std.time.ns_per_ms);
    // it never joined the queue: no wait counted, nothing overdue, so no "prepare waits" or "goes after" line
    try std.testing.expectEqual(@as(u64, 0), g.waited);
    try std.testing.expectEqual(@as(u64, 0), g.overdue);
    try std.testing.expectEqual(@as(u32, 0), g.waiting);
    try std.testing.expectEqual(@as(u32, 1), g.inflight);
    g.leave(io, a);
}
