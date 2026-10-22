//! GlmVisionLimits, ImageLimits, VideoLimits and the picture-cache knobs (glm.py, images.py, videos.py).
const std = @import("std");
const errors = @import("errors.zig");
const Diag = errors.Diag;

/// Environment lookup (the server passes its own; `libcGetenv` reads the process environment).
pub const Getenv = *const fn (name: []const u8) ?[]const u8;

pub fn libcGetenv(name: []const u8) ?[]const u8 {
    var buf: [128:0]u8 = undefined;
    if (name.len >= buf.len) return null;
    @memcpy(buf[0..name.len], name);
    buf[name.len] = 0;
    const p = std.c.getenv(&buf) orelse return null;
    return std.mem.span(p);
}

/// glm.py `_env_int`: empty or unset -> default; otherwise decimal digits within [low, high].
pub fn envInt(getenv: Getenv, name: []const u8, default: i64, low: i64, high: i64, diag: ?*Diag) errors.Error!i64 {
    const value = getenv(name) orelse return default;
    if (value.len == 0) return default;
    const ok = blk: {
        for (value) |c| if (c < '0' or c > '9') break :blk false;
        const v = std.fmt.parseInt(i64, value, 10) catch break :blk false;
        break :blk v >= low and v <= high;
    };
    if (!ok) {
        if (diag) |d| {
            var lo: [32]u8 = undefined;
            var hi: [32]u8 = undefined;
            var a: [1024]u8 = undefined;
            const repr = pyReprInto(&a, value);
            d.set("{s}: {s} to {s}, not {s}", .{ name, errors.commas(&lo, @intCast(low)), errors.commas(&hi, @intCast(high)), repr });
        }
        return error.Input;
    }
    return std.fmt.parseInt(i64, value, 10) catch unreachable;
}

fn pyReprInto(buf: []u8, s: []const u8) []const u8 {
    var fba = std.heap.FixedBufferAllocator.init(buf);
    return errors.pyRepr(fba.allocator(), s) catch s;
}

pub const GlmVisionLimits = struct {
    image_tokens: u32 = 2048, // a picture's tokens at most
    video_tokens: u32 = 16384, // a clip's tokens at most, over all its frame pairs
    video_frames: u32 = 128, // frames sampled from a clip at most
    fps: f64 = 2.0,
    min_tokens: u32 = 16,
    max_images: u32 = 50,
    max_videos: u32 = 4,
    request_image_tokens: u32 = 16384,
    request_video_tokens: u32 = 32768,

    pub fn fromEnv(getenv: Getenv, diag: ?*Diag) errors.Error!GlmVisionLimits {
        const d: GlmVisionLimits = .{};
        return .{
            .image_tokens = @intCast(try envInt(getenv, "TENSORFOLD_GLM_IMAGE_TOKENS", d.image_tokens, 16, 8000, diag)),
            .video_tokens = @intCast(try envInt(getenv, "TENSORFOLD_GLM_VIDEO_TOKENS", d.video_tokens, 64, 240000, diag)),
            .video_frames = @intCast(try envInt(getenv, "TENSORFOLD_GLM_VIDEO_FRAMES", d.video_frames, 2, 2048, diag)),
            .max_images = @intCast(try envInt(getenv, "TENSORFOLD_GLM_MAX_IMAGES", d.max_images, 1, 256, diag)),
            .max_videos = @intCast(try envInt(getenv, "TENSORFOLD_GLM_MAX_VIDEOS", d.max_videos, 1, 16, diag)),
            .request_image_tokens = @intCast(try envInt(getenv, "TENSORFOLD_GLM_REQUEST_IMAGE_TOKENS", d.request_image_tokens, 16, 262144, diag)),
            .request_video_tokens = @intCast(try envInt(getenv, "TENSORFOLD_GLM_REQUEST_VIDEO_TOKENS", d.request_video_tokens, 64, 1048576, diag)),
        };
    }

    /// Each picture's token cap in a request with `pictures` of them.
    pub fn pictureTokens(l: GlmVisionLimits, pictures: usize) u32 {
        const share: u32 = @intCast(@min(l.request_image_tokens / @max(1, pictures), std.math.maxInt(u32)));
        return @max(l.min_tokens, @min(l.image_tokens, share));
    }

    /// Each clip's token cap in a request with `clips` of them.
    pub fn clipTokens(l: GlmVisionLimits, clips: usize) u32 {
        const share: u32 = @intCast(@min(l.request_video_tokens / @max(1, clips), std.math.maxInt(u32)));
        return @max(64, @min(l.video_tokens, share));
    }
};

pub const PictureCacheConfig = struct {
    enabled: bool = true,
    mb: u32 = 384,
    canvases: u32 = 8,

    pub const uncapped: u64 = 8 * 1024 * 1024; // a decode with less room than this was squeezed; not remembered

    pub fn fromEnv(getenv: Getenv, diag: ?*Diag) errors.Error!PictureCacheConfig {
        var c: PictureCacheConfig = .{};
        if (getenv("TENSORFOLD_GLM_PICTURE_CACHE")) |raw| {
            const v = std.mem.trim(u8, raw, " \t\n\r\x0b\x0c");
            var lower: [16]u8 = undefined;
            const low = if (v.len <= lower.len) std.ascii.lowerString(&lower, v) else v;
            c.enabled = !(std.mem.eql(u8, low, "0") or std.mem.eql(u8, low, "off") or std.mem.eql(u8, low, "no") or std.mem.eql(u8, low, "false"));
        }
        c.mb = @intCast(try envInt(getenv, "TENSORFOLD_GLM_PICTURE_CACHE_MB", 384, 16, 8192, diag));
        c.canvases = @intCast(try envInt(getenv, "TENSORFOLD_GLM_PICTURE_CANVASES", 8, 0, 64, diag));
        return c;
    }
};

/// images.py ImageLimits.
pub const ImageLimits = struct {
    max_images: u32 = 4,
    max_encoded_bytes: u64 = 10 * 1024 * 1024,
    max_total_encoded_bytes: u64 = 20 * 1024 * 1024,
    max_dimension: u32 = 8192,
    max_pixels: u64 = 16 * 1024 * 1024,
    max_total_pixels: u64 = 32 * 1024 * 1024,
    timeout_seconds: f64 = 10.0,
    total_timeout_seconds: f64 = 30.0,
    max_redirects: u32 = 3,
    max_url_chars: u32 = 4096,

    pub const default: ImageLimits = .{};

    /// GlmVision.image_limits: the shared limits with this frontend's picture count; bytes and pixels scale with it.
    pub fn forGlm(glm: GlmVisionLimits) ImageLimits {
        const d: ImageLimits = .default;
        const n = glm.max_images;
        var out = d;
        out.max_images = n;
        if (n <= d.max_images) return out;
        out.max_total_encoded_bytes = 64 * 1024 * 1024;
        out.max_total_pixels = @max(d.max_total_pixels, @as(u64, n) * 8 * 1024 * 1024);
        out.total_timeout_seconds = @max(d.total_timeout_seconds, 2.0 * @as(f64, @floatFromInt(n)));
        return out;
    }
};

/// videos.py VideoLimits (the shared defaults; GLM's own caps are in GlmVisionLimits).
pub const VideoLimits = struct {
    max_videos: u32 = 2,
    max_encoded_bytes: u64 = 64 * 1024 * 1024,
    max_total_encoded_bytes: u64 = 96 * 1024 * 1024,
    max_dimension: u32 = 8192,
    max_seconds: f64 = 3600.0,
    timeout_seconds: f64 = 20.0,
    total_timeout_seconds: f64 = 120.0,
    max_redirects: u32 = 3,
    max_url_chars: u32 = 4096,

    pub const default: VideoLimits = .{};
};

var test_env: ?[]const []const u8 = null; // name, value pairs
fn testGetenv(name: []const u8) ?[]const u8 {
    const e = test_env orelse return null;
    var i: usize = 0;
    while (i + 1 < e.len) : (i += 2) if (std.mem.eql(u8, e[i], name)) return e[i + 1];
    return null;
}

test "defaults, token shares and env validation" {
    test_env = null;
    var diag: Diag = .{};
    const l = try GlmVisionLimits.fromEnv(testGetenv, &diag);
    try std.testing.expectEqual(@as(u32, 2048), l.image_tokens);
    try std.testing.expectEqual(@as(u32, 2048), l.pictureTokens(8)); // 16384 / 8
    try std.testing.expectEqual(@as(u32, 327), l.pictureTokens(50)); // 16384 // 50
    try std.testing.expectEqual(@as(u32, 16), l.pictureTokens(10000));
    try std.testing.expectEqual(@as(u32, 16384), l.clipTokens(2));
    try std.testing.expectEqual(@as(u32, 8192), l.clipTokens(4));
    const il = ImageLimits.forGlm(l);
    try std.testing.expectEqual(@as(u64, 64 * 1024 * 1024), il.max_total_encoded_bytes);
    try std.testing.expectEqual(@as(u64, 400 * 1024 * 1024), il.max_total_pixels);
    try std.testing.expectEqual(@as(f64, 100.0), il.total_timeout_seconds);
    const four = ImageLimits.forGlm(.{ .max_images = 3 });
    try std.testing.expectEqual(@as(u32, 3), four.max_images);
    try std.testing.expectEqual(@as(u64, 20 * 1024 * 1024), four.max_total_encoded_bytes);

    test_env = &.{ "TENSORFOLD_GLM_IMAGE_TOKENS", "4096", "TENSORFOLD_GLM_VIDEO_FRAMES", "" };
    const l2 = try GlmVisionLimits.fromEnv(testGetenv, &diag);
    try std.testing.expectEqual(@as(u32, 4096), l2.image_tokens);
    try std.testing.expectEqual(@as(u32, 128), l2.video_frames);

    test_env = &.{ "TENSORFOLD_GLM_MAX_IMAGES", "0" };
    try std.testing.expectError(error.Input, GlmVisionLimits.fromEnv(testGetenv, &diag));
    try std.testing.expectEqualStrings("TENSORFOLD_GLM_MAX_IMAGES: 1 to 256, not '0'", diag.message());
    test_env = &.{ "TENSORFOLD_GLM_REQUEST_VIDEO_TOKENS", "x1" };
    try std.testing.expectError(error.Input, GlmVisionLimits.fromEnv(testGetenv, &diag));
    try std.testing.expectEqualStrings("TENSORFOLD_GLM_REQUEST_VIDEO_TOKENS: 64 to 1,048,576, not 'x1'", diag.message());
    test_env = &.{ "TENSORFOLD_GLM_PICTURE_CACHE", " Off ", "TENSORFOLD_GLM_PICTURE_CACHE_MB", "16" };
    const pc = try PictureCacheConfig.fromEnv(testGetenv, &diag);
    try std.testing.expect(!pc.enabled);
    try std.testing.expectEqual(@as(u32, 16), pc.mb);
    test_env = null;
}
