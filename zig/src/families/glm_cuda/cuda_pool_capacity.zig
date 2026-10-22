//! Ports glm5_next/cuda/engine.py pool_rows and tensorfold/cuda/capacity.py reserve policy.
//! TensorFold by Ash Hart and contributors; MiaAI-Lab recipe MEMORY_RESERVE_GIB / KV_POOL_GIB.
//! Uses live allocation room after Zig's fixed arenas and drafter are installed, rather than Python estimates.
const std = @import("std");
const cuda = @import("cuda");
const display = @import("cuda_display_kv.zig");
const pool = @import("cuda_pool.zig");
pub fn gibBytes(raw: ?[]const u8, default: f64) !usize {
    const g = if (raw) |r| std.fmt.parseFloat(f64, std.mem.trim(u8, r, " \t\r\n")) catch return error.InvalidPoolGiB else default;
    const bytes = g * (1 << 30);
    if (!std.math.isFinite(bytes) or bytes < 0 or bytes >= 0x1p64) return error.InvalidPoolGiB;
    return @intFromFloat(bytes);
}
pub fn reserve(raw: ?[]const u8, total: usize, unified: bool) !usize {
    const value = if (raw) |r| (if (std.mem.trim(u8, r, " \t\r\n").len == 0) null else raw) else null;
    const n = try gibBytes(value, @as(f64, @floatFromInt(@max(if (unified) @as(usize, 4 << 30) else 1 << 30, total / (if (unified) @as(usize, 10) else 32)))) / (1 << 30));
    if (n < (if (unified) @as(usize, 2 << 30) else 1 << 29) or n > total) return error.InvalidMemoryReserve;
    return n;
}
pub const Counts = struct { total: usize, available: usize };
pub fn meminfo(text: []const u8) !Counts {
    var total: ?usize = null;
    var available: ?usize = null;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var words = std.mem.tokenizeAny(u8, line, ": \t");
        const key = words.next() orelse continue;
        if (!std.mem.eql(u8, key, "MemTotal") and !std.mem.eql(u8, key, "MemAvailable")) continue;
        const kib = try std.fmt.parseInt(usize, words.next() orelse return error.BadMeminfo, 10);
        if (!std.mem.eql(u8, words.next() orelse return error.BadMeminfo, "kB") or words.next() != null) return error.BadMeminfo;
        const bytes = try std.math.mul(usize, kib, 1024);
        if (std.mem.eql(u8, key, "MemTotal")) {
            if (total != null) return error.BadMeminfo;
            total = bytes;
        } else {
            if (available != null) return error.BadMeminfo;
            available = bytes;
        }
    }
    const t = total orelse return error.BadMeminfo;
    const a = available orelse return error.BadMeminfo;
    if (t == 0 or a > t) return error.BadMeminfo;
    return .{ .total = t, .available = a };
}
pub fn room(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context) !usize {
    const unified = (try ctx.attribute(.integrated)) != 0;
    const counts: Counts = if (unified) blk: {
        var f = try std.Io.Dir.cwd().openFile(io, "/proc/meminfo", .{});
        defer f.close(io);
        var buf: [4096]u8 = undefined;
        var r = f.readerStreaming(io, &buf);
        const text = try r.interface.allocRemaining(gpa, .limited(1 << 20));
        defer gpa.free(text);
        break :blk try meminfo(text);
    } else blk: {
        const m = try ctx.memInfo();
        break :blk .{ .total = m.total, .available = m.free };
    };
    const res = try reserve(display.env("TENSORFOLD_MEMORY_RESERVE_GIB") orelse display.env("MEMORY_RESERVE_GIB"), counts.total, unified);
    if (counts.available < res) return error.PoolMemoryBudget;
    var left = counts.available - res;
    if (display.env("TENSORFOLD_CUDA_MEMORY_LIMIT_GB")) |limit| left = @min(left, (try gibBytes(limit, 0)) -| cuda.usage(false).device);
    return left;
}
/// Whole extents, with display credited above the ordinary-memory cap. No kept-state charge until caching exists.
pub fn rows(capacity: usize, spare: usize, display_bytes: usize, per_token: usize) !usize {
    if (per_token == 0) return error.InvalidTokenBytes;
    const extra = try std.math.add(usize, spare, display_bytes);
    const n = try std.math.add(usize, capacity, extra / per_token);
    const result = std.mem.alignBackward(usize, n, pool.alignment);
    if (result < pool.alignment) return error.BadPool;
    return result;
}
test "pool reserve, host counts, cap and display credit use integer extent arithmetic" {
    const g = 1 << 30;
    try std.testing.expectEqual(@as(usize, 145 * g / 10), try reserve("14.5", 120 * g, true));
    try std.testing.expectEqual(@as(usize, 12 * g), try reserve(null, 120 * g, true));
    try std.testing.expectError(error.InvalidMemoryReserve, reserve("1", 120 * g, true));
    for ([_][]const u8{ "nan", "inf", "-1" }) |s| try std.testing.expectError(error.InvalidPoolGiB, gibBytes(s, 3));
    try std.testing.expectEqual(Counts{ .total = 102400, .available = 51200 }, try meminfo("MemTotal: 100 kB\nMemAvailable: 50 kB\n"));
    try std.testing.expectError(error.BadMeminfo, meminfo("MemTotal: 100 kB\nMemAvailable: 101 kB\n"));
    try std.testing.expectEqual(@as(usize, 4096), try rows(4096, 0, 0, 6204));
    try std.testing.expectEqual(@as(usize, 6144), try rows(4096, 6204 * 2048, 0, 6204));
    try std.testing.expectEqual(@as(usize, 6144), try rows(4096, 0, 6204 * 2048, 6204));
}
