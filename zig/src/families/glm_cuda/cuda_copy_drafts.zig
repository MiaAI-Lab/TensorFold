//! Port of TensorFold glm5_next/cuda/copy_drafts.py: CopyDrafts.
//! Ash Hart and TensorFold contributors; host index from core/copy_index.zig.
const std = @import("std");
const CopyIndex = @import("core").CopyIndex;
pub const Settings = struct {
    match: usize = 8,
    most: usize = 15,
    reply_match: usize = 16,
    pad: []const usize = &.{16},
    miss_most: usize = 0,
    pub fn validate(s: Settings) !void {
        if (s.match < 2 or s.match > 64 or s.most < 1 or s.most > 63 or (s.reply_match != 0 and (s.reply_match <= s.match or s.reply_match > 64)) or s.miss_most > s.most) return error.BadCopySettings;
        var prev: usize = 8;
        for (s.pad) |w| {
            if (w <= prev or w > 64) return error.BadCopySettings;
            prev = w;
        }
    }
};
pub const Copy = struct {
    index: CopyIndex,
    settings: Settings,
    prompt: usize,
    proposed: usize = 0,
    missed: bool = false,
    /// Context includes the pending token; prompt excludes that token.
    pub fn init(gpa: std.mem.Allocator, context: []const u32, settings: Settings) !Copy {
        try settings.validate();
        return .{ .index = try CopyIndex.init(gpa, context), .settings = settings, .prompt = context.len -| 1 };
    }
    pub fn deinit(c: *Copy) void {
        c.index.deinit();
    }
    /// Only serial target draws enter the index (including the new pending draw).
    pub fn extend(c: *Copy, tokens: []const u32) !void {
        if (tokens.len == 0) return;
        try c.index.extend(tokens);
        if (c.proposed > 0) {
            c.missed = tokens.len <= c.proposed;
            c.proposed = 0;
        }
    }
    pub fn propose(c: *Copy, out: []u32) []const u32 {
        c.proposed = 0;
        const ctx = c.index.ctx.items;
        const s = c.settings;
        const most = if (c.missed and s.miss_most > 0) @min(s.most, s.miss_most) else s.most;
        const k = @min(out.len, most);
        if (ctx.len <= s.match or k == 0) return &.{};
        const suffix = ctx.len - s.match;
        const hits = c.index.seen.get(ctx[ctx.len - 1]) orelse return &.{};
        var earliest: ?usize = null;
        var full: ?usize = null;
        for (hits.items) |last| {
            const end = @as(usize, last) + 1;
            if (end < s.match or end >= ctx.len) continue;
            const start = end - s.match;
            if (!std.mem.eql(u32, ctx[start..end], ctx[suffix..])) continue;
            if (s.reply_match > s.match and start >= c.prompt) {
                const extra = s.reply_match - s.match;
                if (start < extra or suffix < extra or !std.mem.eql(u32, ctx[start - extra .. start], ctx[suffix - extra .. suffix])) continue;
            }
            if (earliest == null) earliest = end;
            if (ctx.len - end >= k) full = end;
        }
        const at = full orelse earliest orelse return &.{};
        var n = @min(k, ctx.len - at);
        @memcpy(out[0..n], ctx[at..][0..n]);
        c.proposed = n;
        if (n + 1 > 8) for (s.pad) |w| {
            if (w < n + 1) continue;
            if (w > n + 1 and w - 1 <= out.len) {
                @memset(out[n .. w - 1], out[n - 1]);
                n = w - 1;
            }
            break;
        };
        return out[0..n];
    }
};
test "copy latest full match, miss cap and padding count only copied tokens" {
    var c = try Copy.init(std.testing.allocator, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5, 6, 7, 8 }, .{ .reply_match = 0, .miss_most = 2 });
    defer c.deinit();
    var out: [15]u32 = undefined;
    const got = c.propose(&out);
    try std.testing.expectEqual(@as(usize, 15), got.len);
    try std.testing.expectEqual(@as(u32, 9), got[0]);
    try c.extend(&.{99});
    try std.testing.expect(c.missed);
    try std.testing.expectEqual(@as(usize, 0), c.propose(&out).len);
}
test "reply matches require the longer suffix and empty extends do not settle" {
    var c = try Copy.init(std.testing.allocator, &.{ 9, 9 }, .{ .match = 2, .most = 3, .reply_match = 4, .pad = &.{} });
    defer c.deinit();
    try c.extend(&.{ 1, 2, 3, 4, 5, 8, 2, 3 });
    var out: [3]u32 = undefined;
    try std.testing.expectEqual(@as(usize, 0), c.propose(&out).len);
    c.settings.reply_match = 0;
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 8 }, c.propose(&out));
    try c.extend(&.{});
    try std.testing.expectEqual(@as(usize, 3), c.proposed);
}
