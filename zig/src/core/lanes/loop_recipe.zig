//! The recipe's loop guard rule (patch 0091, TF_GLM_LOOP_GUARD; Python `engine/loop_guard.py`): a think block that
//! collapsed into repeating itself ends. Two collapses count:
//!
//! * an exact cycle of at most `max_period` tokens that held for `window` tokens in a row ("maybe maybe maybe ...");
//! * one token id taking at least half of the last `window` tokens, the cycle broken by other text ("the" in every
//!   other word of a degenerate block).
//!
//! The Python guard keeps counters as tokens arrive; this reads the same two facts off the committed tokens, which a
//! lane stream already holds (and the speculative cut reads through two slices), so it needs no state.
const std = @import("std");

pub const window: usize = 256; // committed think tokens a collapse has to fill
pub const max_period: usize = 16; // longest exact cycle (in tokens) that counts as a loop
const dominant: usize = window / 2; // tokens of one id in the window that count as collapsed

fn at(head: []const u32, tail: []const u32, i: usize) u32 {
    return if (i < head.len) head[i] else tail[i - head.len];
}

/// Whether the first `n` tokens of `head ++ tail` have collapsed, ending at their last: the cycle's period (1 to
/// `max_period`), or 0 for one dominant token; null when not.
pub fn collapsedAt(head: []const u32, tail: []const u32, n: usize) ?u32 {
    if (n < window) return null;
    for (1..max_period + 1) |p| {
        if (n < window + p) break;
        var held = true;
        for (n - window..n) |i| {
            if (at(head, tail, i) != at(head, tail, i - p)) {
                held = false;
                break;
            }
        }
        if (held) return @intCast(p);
    }
    const last = at(head, tail, n - 1);
    var count: usize = 0;
    for (n - window..n) |i| count += @intFromBool(at(head, tail, i) == last);
    return if (count >= dominant) 0 else null;
}

const fixture = @embedFile("fixtures_loop_guard.json");

test "the rule fires at the token the Python LoopGuard fires at" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), fixture, .{});
    for (root.object.get("cases").?.array.items) |case| {
        const items = case.object.get("tokens").?.array.items;
        const think_end: u32 = @intCast(case.object.get("think_end").?.integer);
        const tokens = try arena.allocator().alloc(u32, items.len);
        for (items, tokens) |v, *t| t.* = @intCast(v.integer);
        // the stream stops watching at the think block's close
        var fired: i64 = -1;
        for (0..tokens.len) |i| {
            if (tokens[i] == think_end) break;
            if (collapsedAt(tokens[0 .. i + 1], &.{}, i + 1) != null) {
                fired = @intCast(i);
                break;
            }
        }
        errdefer std.debug.print("case {s}\n", .{case.object.get("label").?.string});
        try std.testing.expectEqual(case.object.get("fires_at").?.integer, fired);
        // two slices read the same as one
        if (fired >= 0) {
            const cut: usize = @intCast(@divTrunc(fired, 2));
            const end: usize = @intCast(fired + 1);
            try std.testing.expect(collapsedAt(tokens[0..cut], tokens[cut..end], end) != null);
        }
    }
}
