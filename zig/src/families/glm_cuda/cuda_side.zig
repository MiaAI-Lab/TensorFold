//! TF_GLM_SIDE=1 (recipe patch 0105, glm5_next/cuda/forward.py moe_block): a decode window's MoE shared expert runs on a
//! second stream beside the router, the selection, the plan and the routed experts, and is joined before the combine
//! reads its output. The same kernels with the same arguments on disjoint buffers (sgu / sact / sums / sy and the
//! matmul's split-K scratch are touched by nothing between the fork and the join), so the bits are the single stream's.
//! Decode and verify windows only (Python: `not prefill`); the prompt chunk never has a side stream. The events are
//! recorded while the window is captured as a graph, so the fork and the join become graph edges (the second stream
//! joins the capture through the fork event and leaves it through the join event).
//! Off by default, as in Python; the recipe's scripts turn it on at TP=4 only (config.sh: TF_GLM_SIDE, "off at two and
//! three until measured"). Ash Hart and TensorFold contributors wrote the Python engine; this port is by Mia's AI Lab.
const std = @import("std");
const cuda = @import("cuda");

/// TF_GLM_SIDE as Python reads it: unset or empty is 0, only 0 and 1 are valid.
pub fn parse(value: ?[]const u8) !bool {
    const raw = value orelse return false;
    const v = std.mem.trim(u8, raw, " \t\r\n");
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadSideSetting;
}

pub fn fromEnv() !bool {
    const v = std.c.getenv("TF_GLM_SIDE") orelse return false;
    return parse(std.mem.span(v));
}

/// The second stream and the two events, generic over the stream and event types so a fake can log the order.
/// `E.record(stream)` and `S.wait(event)` are the real types' methods (cuda.Stream / cuda.Event).
pub fn Pair(comptime S: type, comptime E: type) type {
    return struct {
        stream: S,
        forked: E,
        joined: E,

        const Self = @This();

        /// Everything queued on `main` so far (the normed rows) is done before the side stream starts.
        pub fn fork(p: *const Self, main: S) !void {
            try p.forked.record(main);
            try p.stream.wait(p.forked);
        }
        /// `main` waits for everything queued on the side stream (the shared expert's output) before going on.
        pub fn join(p: *const Self, main: S) !void {
            try p.joined.record(p.stream);
            try main.wait(p.joined);
        }
    };
}

pub const Side = struct {
    pair: Pair(cuda.Stream, cuda.Event),

    pub fn init(d: *const cuda.Driver) !Side {
        var s = try cuda.Stream.init(d, true);
        errdefer s.deinit();
        var a = try cuda.Event.init(d, false);
        errdefer a.deinit();
        const b = try cuda.Event.init(d, false);
        return .{ .pair = .{ .stream = s, .forked = a, .joined = b } };
    }
    pub fn deinit(s: *Side) void {
        s.pair.joined.deinit();
        s.pair.forked.deinit();
        s.pair.stream.deinit();
    }
};

test "TF_GLM_SIDE is 0 or 1, unset is off" {
    try std.testing.expect(!try parse(null));
    try std.testing.expect(!try parse(""));
    try std.testing.expect(!try parse(" 0 "));
    try std.testing.expect(try parse("1"));
    try std.testing.expectError(error.BadSideSetting, parse("2"));
    try std.testing.expectError(error.BadSideSetting, parse("true"));
}

const Log = struct {
    items: [16][]const u8 = undefined,
    n: usize = 0,
    fn add(l: *Log, s: []const u8) void {
        l.items[l.n] = std.testing.allocator.dupe(u8, s) catch unreachable;
        l.n += 1;
    }
};
const FakeStream = struct {
    name: []const u8,
    log: *Log,
    fn wait(s: FakeStream, e: FakeEvent) !void {
        var b: [64]u8 = undefined;
        s.log.add(std.fmt.bufPrint(&b, "{s} waits {s}", .{ s.name, e.name }) catch unreachable);
    }
};
const FakeEvent = struct {
    name: []const u8,
    fn record(e: FakeEvent, s: FakeStream) !void {
        var b: [64]u8 = undefined;
        s.log.add(std.fmt.bufPrint(&b, "{s} records {s}", .{ s.name, e.name }) catch unreachable);
    }
};

test "fork then join order: the side stream starts after main's work and main resumes after the side's" {
    var log: Log = .{};
    defer for (log.items[0..log.n]) |s| std.testing.allocator.free(s);
    const main: FakeStream = .{ .name = "main", .log = &log };
    const p: Pair(FakeStream, FakeEvent) = .{ .stream = .{ .name = "side", .log = &log }, .forked = .{ .name = "fork" }, .joined = .{ .name = "join" } };
    try p.fork(main);
    log.add("(side kernels)");
    try p.join(main);
    const want = [_][]const u8{ "main records fork", "side waits fork", "(side kernels)", "side records join", "main waits join" };
    try std.testing.expectEqual(want.len, log.n);
    for (want, log.items[0..log.n]) |w, g| try std.testing.expectEqualStrings(w, g);
}
