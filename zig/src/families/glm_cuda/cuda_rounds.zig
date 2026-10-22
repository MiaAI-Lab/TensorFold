//! glm5_next/cuda/{multi.py:_round,verify.py:BatchedVerify.commit}.
//! Ash Hart and TensorFold contributors. P6 supplies forward/replay launches;
//! _conv_shift_seg stays Triton and is replayed by P1, never reimplemented here.
const std = @import("std");
const State = @import("cuda_state.zig").State;
pub const Window = struct { state: *State, tokens: []const u32, eos: []const u32 = &.{} };
/// keep = 1 + matching draft prefix, stopping before feeding an EOS draw.
pub fn accept(tokens: []const u32, sampled: []const u32, eos: []const u32) !usize {
    if (tokens.len == 0 or sampled.len != tokens.len) return error.BadWindow;
    var keep: usize = 1;
    for (tokens[1..], sampled[0 .. sampled.len - 1]) |draft, drawn| {
        if (draft != drawn or std.mem.indexOfScalar(u32, eos, drawn) != null) break;
        keep += 1;
    }
    return keep;
}
/// Per-segment KDA table entries (verify.py:_tables): row0, rows, slot,
/// parity, conv slot, keep. All recurrent layers have the same parity.
pub fn table(windows: []const Window, keeps: ?[]const usize, out: []i32) !usize {
    if (windows.len == 0 or windows.len > 8 or out.len != windows.len * 6 or (keeps != null and keeps.?.len != windows.len)) return error.BadWindow;
    var row0: usize = 0;
    for (windows, 0..) |w, i| {
        if (w.tokens.len == 0 or w.state.pos > w.state.capacity or w.tokens.len > w.state.capacity - w.state.pos) return error.BadWindow;
        for (windows[0..i]) |prev| if (prev.state == w.state or prev.state.slot == w.state.slot) return error.DuplicateSlot;
        for (w.state.cur) |cur| if (cur != w.state.cur[0]) return error.MixedParity;
        const keep = if (keeps) |k| k[i] else w.tokens.len;
        if (keep < 1 or keep > w.tokens.len) return error.BadKeep;
        const fields = [_]usize{ row0, w.tokens.len, w.state.slot, w.state.cur[0], w.state.slot, keep };
        for (fields, 0..) |x, j| {
            if (x > std.math.maxInt(i32)) return error.BadWindow;
            out[6 * i + j] = @intCast(x);
        }
        row0 += w.tokens.len;
    }
    return row0;
}
/// Backend E: forward(windows, table), sample(windows, out), stageCommit(table),
/// replay(), convShift(), setPosition(state, pos), commitDrafts(windows, keeps).
/// Device failures propagate; callers must stop a failed multi-rank engine.
pub fn Round(comptime E: type) type {
    return struct {
        engine: *E,
        limit: usize = 64,
        active: bool = false,
        windows: [8]Window = undefined,
        count: usize = 0,
        rows: usize = 0,
        ktable: [48]i32 = undefined,
        pub fn verify(r: *@This(), windows: []const Window, sampled: []u32) !void {
            if (r.active) return error.UncommittedWindow;
            if (windows.len == 0 or windows.len > 8) return error.BadWindow;
            const rows = try table(windows, null, r.ktable[0 .. windows.len * 6]);
            if (sampled.len != rows or rows > r.limit) return error.BadWindow;
            try r.engine.forward(windows, r.ktable[0 .. windows.len * 6]);
            try r.engine.sample(windows, sampled);
            @memcpy(r.windows[0..windows.len], windows);
            r.count = windows.len;
            r.rows = rows;
            r.active = true;
        }
        pub fn commit(r: *@This(), keeps: []const usize) !void {
            if (!r.active or keeps.len != r.count) return error.NoVerifyWindow;
            const windows = r.windows[0..r.count];
            _ = try table(windows, keeps, r.ktable[0 .. r.count * 6]);
            try r.engine.stageCommit(r.ktable[0 .. r.count * 6]);
            var replay = false;
            for (windows, keeps) |w, k| replay = replay or k < w.tokens.len;
            if (replay) try r.engine.replay();
            try r.engine.convShift();
            for (windows, keeps) |w, k| try r.engine.setPosition(w.state, w.state.pos + k);
            for (windows, keeps) |w, k| {
                w.state.pos += k;
                w.state.cur = @splat(1 - w.state.cur[0]);
            }
            r.active = false;
            try r.engine.commitDrafts(windows, keeps);
        }
        pub fn run(r: *@This(), windows: []const Window, sampled: []u32, keeps: []usize) !void {
            if (windows.len != keeps.len) return error.BadWindow;
            try r.verify(windows, sampled);
            var at: usize = 0;
            for (windows, keeps) |w, *k| {
                k.* = try accept(w.tokens, sampled[at..][0..w.tokens.len], w.eos);
                at += w.tokens.len;
            }
            try r.commit(keeps);
        }
    };
}
/// cat(taps[0..4], dim=-1)[:keep] strides, source [5, total_rows, 4096] bf16.
pub fn packTaps(th: anytype, source: u64, total_rows: usize, row0: usize, keep: usize, dst: u64) !void {
    if (row0 > total_rows or keep > total_rows - row0) return error.BadWindow;
    for (0..5) |i| try th.slotCopy(source + (i * total_rows + row0) * 4096 * 2, 4096 * 2, dst + i * 4096 * 2, 5 * 4096 * 2, 4096 * 2, keep);
}
test "accept rejects mismatch and stops EOS before committing its input row" {
    try std.testing.expectEqual(@as(usize, 3), try accept(&.{ 1, 2, 3 }, &.{ 2, 3, 4 }, &.{}));
    try std.testing.expectEqual(@as(usize, 2), try accept(&.{ 1, 2, 3 }, &.{ 2, 9, 4 }, &.{}));
    try std.testing.expectEqual(@as(usize, 1), try accept(&.{ 1, 2, 3 }, &.{ 2, 3, 4 }, &.{2}));
}
test "round replays partial windows once, shifts conv, then publishes parity and position" {
    const Toy = struct {
        replayed: usize = 0,
        shifted: usize = 0,
        drafted: usize = 0,
        fn forward(_: *@This(), _: []const Window, _: []const i32) !void {}
        fn sample(_: *@This(), _: []const Window, out: []u32) !void {
            @memcpy(out, &[_]u32{ 2, 9, 8, 5 });
        }
        fn stageCommit(_: *@This(), t: []const i32) !void {
            try std.testing.expectEqualSlices(i32, &.{ 0, 3, 0, 0, 0, 2, 3, 1, 1, 1, 1, 1 }, t);
        }
        fn replay(e: *@This()) !void {
            e.replayed += 1;
        }
        fn convShift(e: *@This()) !void {
            e.shifted += 1;
        }
        fn setPosition(_: *@This(), _: *State, _: usize) !void {}
        fn commitDrafts(e: *@This(), _: []const Window, _: []const usize) !void {
            e.drafted += 1;
        }
    };
    var a: State = .{ .slot = 0, .base = 0, .capacity = 2048, .fixed = undefined, .cache = undefined };
    var b: State = .{ .slot = 1, .base = 2048, .capacity = 2048, .pos = 7, .cur = @splat(1), .fixed = undefined, .cache = undefined };
    var toy: Toy = .{};
    var r: Round(Toy) = .{ .engine = &toy };
    var out: [4]u32 = undefined;
    var keeps: [2]usize = undefined;
    try r.run(&.{ .{ .state = &a, .tokens = &.{ 1, 2, 3 } }, .{ .state = &b, .tokens = &.{4} } }, &out, &keeps);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, &keeps);
    try std.testing.expectEqual(@as(usize, 2), a.pos);
    try std.testing.expectEqual(@as(usize, 8), b.pos);
    try std.testing.expectEqual(@as(u1, 1), a.cur[0]);
    try std.testing.expectEqual(@as(u1, 0), b.cur[0]);
    try std.testing.expectEqual(@as(usize, 1), toy.replayed);
    try std.testing.expectEqual(@as(usize, 1), toy.shifted);
    try std.testing.expectError(error.NoVerifyWindow, r.commit(&keeps));
}

test "drafter tap concatenation copies five row-strided planes into accepted rows" {
    const Spy = struct {
        called: usize = 0,
        fn slotCopy(s: *@This(), src: u64, src_ld: usize, dst: u64, dst_ld: usize, row_bytes: usize, rows: usize) !void {
            try std.testing.expectEqual(@as(u64, 0x1000 + (s.called * 10 + 3) * 8192), src);
            try std.testing.expectEqual(@as(u64, 0x100000 + s.called * 8192), dst);
            try std.testing.expectEqual(@as(usize, 8192), src_ld);
            try std.testing.expectEqual(@as(usize, 5 * 8192), dst_ld);
            try std.testing.expectEqual(@as(usize, 8192), row_bytes);
            try std.testing.expectEqual(@as(usize, 2), rows);
            s.called += 1;
        }
    };
    var spy: Spy = .{};
    try packTaps(&spy, 0x1000, 10, 3, 2, 0x100000);
    try std.testing.expectEqual(@as(usize, 5), spy.called);
    try std.testing.expectError(error.BadWindow, packTaps(&spy, 0x1000, 10, 9, 2, 0x100000));
}
