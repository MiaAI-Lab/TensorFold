//! glm5_next/cuda/multi.py:_round and decode.py: greedy CLI rounds.
//! Ash Hart and TensorFold contributors; jschmied CUDA serving patterns.
const std = @import("std");
const rounds = @import("cuda_rounds.zig");

/// Draft statistics of the last run (rounds, proposed rows, accepted proposals); the CLI prints them.
pub var stats: struct { rounds: u64 = 0, drafted: u64 = 0, accepted: u64 = 0 } = .{};

/// The only emitted tokens are target draws; drafts are compared before their input rows commit.
pub fn run(comptime E: type, e: *E, gpa: std.mem.Allocator, seq: anytype, first: u32, max_tokens: usize, eos: []const u32, drafts: bool) ![]u32 {
    stats = .{};
    var out = try gpa.alloc(u32, max_tokens);
    errdefer gpa.free(out);
    if (max_tokens == 0) return out;
    out[0] = first;
    var n: usize = 1;
    var proposal: [15]u32 = undefined;
    var sampled: [16]u32 = undefined;
    var inputs: [16]u32 = undefined;
    var next: []const u32 = out[0..1];
    var source: E.DraftSource = .prefill_last;
    var round: rounds.Round(E) = .{ .engine = e };
    while (n < max_tokens and std.mem.indexOfScalar(u32, eos, out[n - 1]) == null) {
        var drafted: usize = 0;
        if (drafts) {
            var req = [_]E.DraftReq{.{ .seq = seq, .source = source, .next = next, .count = @min(15, max_tokens - n - 1), .out = &proposal }};
            try e.draftMany(&req);
            drafted = req[0].drafted;
            if (drafted > req[0].count or drafted > proposal.len) return error.BadDraftCount;
        }
        inputs[0] = out[n - 1];
        @memcpy(inputs[1..][0..drafted], proposal[0..drafted]);
        const rows = 1 + drafted;
        var keeps: [1]usize = undefined;
        try round.run(&.{.{ .state = &seq.st, .tokens = inputs[0..rows], .eos = eos }}, sampled[0..rows], &keeps);
        const keep = keeps[0];
        stats.rounds += 1;
        stats.drafted += drafted;
        stats.accepted += keep -| 1;
        for (sampled[0..keep]) |t| {
            out[n] = t;
            n += 1;
            if (std.mem.indexOfScalar(u32, eos, t) != null) break;
        }
        next = sampled[0..keep];
        source = .{ .window = 0 };
    }
    return gpa.realloc(out, n);
}

test "CLI rounds emit serial greedy draws despite wrong drafts, EOS and token limits" {
    const State = @import("cuda_state.zig").State;
    const Toy = struct {
        const Self = @This();
        const Seq = struct { st: State };
        pub const DraftSource = union(enum) { prefill_last, window: usize };
        pub const DraftReq = struct { seq: *Seq, source: DraftSource, next: []const u32, count: usize, out: []u32, drafted: usize = 0 };
        position: usize = 0,
        replayed: usize = 0,
        commits: usize = 0,
        pub fn draftMany(_: *Self, reqs: []DraftReq) !void {
            for (reqs) |*r| {
                r.drafted = @min(r.count, 3);
                // First proposal agrees, second intentionally diverges.
                for (r.out[0..r.drafted], 0..) |*t, i| t.* = if (i == 0) r.next[r.next.len - 1] + 1 else 999;
            }
        }
        pub fn forward(e: *Self, windows: []const rounds.Window, _: []const i32) !void {
            e.position = windows[0].state.pos;
        }
        pub fn sample(e: *Self, _: []const rounds.Window, out: []u32) !void {
            for (out, 0..) |*t, i| t.* = @intCast(e.position + i + 2);
        }
        pub fn stageCommit(_: *Self, _: []const i32) !void {}
        pub fn replay(e: *Self) !void {
            e.replayed += 1;
        }
        pub fn convShift(_: *Self) !void {}
        pub fn setPosition(_: *Self, _: *State, _: usize) !void {}
        pub fn commitDrafts(e: *Self, _: []const rounds.Window, _: []const usize) !void {
            e.commits += 1;
        }
    };
    for ([_]usize{ 0, 1, 2, 7, 20 }) |limit| {
        var a: Toy.Seq = .{ .st = .{ .slot = 0, .base = 0, .capacity = 2048, .fixed = undefined, .cache = undefined } };
        var b = a;
        var serial: Toy = .{};
        var drafted: Toy = .{};
        const x = try run(Toy, &serial, std.testing.allocator, &a, 1, limit, &.{8}, false);
        defer std.testing.allocator.free(x);
        const y = try run(Toy, &drafted, std.testing.allocator, &b, 1, limit, &.{8}, true);
        defer std.testing.allocator.free(y);
        try std.testing.expectEqualSlices(u32, x, y);
        try std.testing.expectEqual(a.st.pos, b.st.pos);
        if (limit == 20) try std.testing.expect(drafted.replayed > 0);
    }
}
