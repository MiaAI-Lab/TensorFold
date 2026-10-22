//! Test helper: fixtures live in zig/tests/vision/ and are read relative to the repository root (the test step
//! runs with that as its working directory).
const std = @import("std");

pub const dir = "zig/tests/vision/";

pub fn read(allocator: std.mem.Allocator, name: []const u8, max: usize) ![]u8 {
    const path = try std.fmt.allocPrint(allocator, "{s}{s}", .{ dir, name });
    defer allocator.free(path);
    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, allocator, .limited(max)) catch |e| switch (e) {
        error.FileNotFound => return error.SkipZigTest,
        else => return e,
    };
}

pub fn exists(name: []const u8) bool {
    var buf: [512]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}{s}", .{ dir, name }) catch return false;
    std.Io.Dir.cwd().access(std.testing.io, path, .{}) catch return false;
    return true;
}

pub fn hexDigest(bytes: []const u8) [64]u8 {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &h, .{});
    return std.fmt.bytesToHex(h, .lower);
}

/// The generator zig/tests/vision/gen/common.py `lcg_bytes` uses for synthetic inputs.
pub fn lcgBytes(out: []u8, seed: u64) void {
    var x = seed;
    for (out) |*o| {
        x = x *% 6364136223846793005 +% 1442695040888963407;
        o.* = @intCast(x >> 56);
    }
}
