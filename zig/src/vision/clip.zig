//! `decode_clip`: a clip's sampled frames, each fitted onto the clip's canvas as it is decoded.
const std = @import("std");
const av = @import("av.zig");
const errors = @import("errors.zig");
const geometry = @import("geometry.zig");
const items_mod = @import("items.zig");
const limits_mod = @import("limits.zig");
const resize = @import("resize.zig");
const Allocator = std.mem.Allocator;
const Diag = errors.Diag;

pub const Params = struct {
    limits: limits_mod.GlmVisionLimits, // with video_tokens already the clip's share
    max_dimension: u32 = 8192,
    max_seconds: f64 = 3600.0,
};

fn invalidMessage(diag: ?*Diag, code: c_int) errors.Error {
    return errors.fail(diag, "video bytes are invalid or unsupported ({s}); use MP4 or WebM", .{av.errorName(code)});
}

/// `deadline_ns`: monotonic (std.Io.Clock.awake) nanoseconds.
pub fn decodeClip(a: Allocator, io: std.Io, libs: *const av.Libs, data: []const u8, params: Params, deadline_ns: i96, diag: ?*Diag) (Allocator.Error || errors.Error)!items_mod.Clip {
    const L = params.limits;
    var c = av.Clip.open(a, libs, data) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.NoVideoStream => return errors.fail(diag, "the video has no video stream", .{}),
        error.Unsupported => return errors.fail(diag, "video bytes are invalid or unsupported (UnsupportedFFmpeg); use MP4 or WebM", .{}),
        error.Failed => return errors.fail(diag, "video bytes are invalid or unsupported (InvalidDataError); use MP4 or WebM", .{}),
    };
    defer c.close();
    const info = c.info;
    if (@min(info.width, info.height) == 0 or @max(info.width, info.height) > params.max_dimension)
        return errors.fail(diag, "video dimensions are missing or exceed the pixel limit", .{});
    // fps = float(average_rate or guessed_rate or 0)
    var fps: f64 = 0;
    if (info.avg_num != 0 and info.avg_den != 0) {
        fps = @as(f64, @floatFromInt(info.avg_num)) / @as(f64, @floatFromInt(info.avg_den));
    } else if (info.guessed_num != 0 and info.guessed_den != 0) {
        fps = @as(f64, @floatFromInt(info.guessed_num)) / @as(f64, @floatFromInt(info.guessed_den));
    }
    var total: i64 = info.nb_frames;
    if (total == 0 and fps > 0) { // WebM and some MKV files carry no frame count
        var seconds: f64 = 0.0;
        if (info.duration != av.nopts and info.duration != 0) {
            // float(stream.duration * stream.time_base): the exact rational, correctly rounded
            const num: i128 = @as(i128, info.duration) * info.time_base_num;
            seconds = @as(f64, @floatFromInt(num)) / @as(f64, @floatFromInt(info.time_base_den));
        } else if (info.container_duration != av.nopts and info.container_duration != 0) {
            seconds = @as(f64, @floatFromInt(info.container_duration)) / 1e6;
        }
        total = geometry.pyRound(seconds * fps);
    }
    if (!(fps > 0 and fps <= 1000) or total <= 0) return errors.fail(diag, "the video's frame count or rate is missing", .{});
    const total_f: f64 = @floatFromInt(total);
    if (total_f / fps > params.max_seconds) {
        const minutes = params.max_seconds / 60.0;
        return errors.fail(diag, "videos are limited to {d} minutes", .{geometry.pyRound(minutes)});
    }
    if (total > std.math.maxInt(u32)) return errors.fail(diag, "the video's frame count or rate is missing", .{});
    const picks = try geometry.sampleFrames(a, @intCast(total), fps, total_f / fps, L);
    defer a.free(picks);
    const canvas = try geometry.smartResize(@intCast(picks.len), info.height, info.width, L.video_tokens, L.min_tokens, diag);
    const content = geometry.contentSize(@intCast(picks.len), info.height, info.width, canvas, L.min_tokens);
    const plane = 3 * @as(usize, canvas.h) * canvas.w;

    // wanted frames -> their canvases, by index
    var wanted: std.AutoHashMapUnmanaged(u32, []u8) = .empty;
    defer {
        var it = wanted.valueIterator();
        while (it.next()) |v| a.free(v.*);
        wanted.deinit(a);
    }
    for (picks) |p| try wanted.put(a, p, &.{});
    const last_pick = picks[picks.len - 1];
    c.openDecoder() catch |e| switch (e) {
        error.Failed => return invalidMessage(diag, c.code),
        error.Unsupported => return errors.fail(diag, "video bytes are invalid or unsupported (UnsupportedFFmpeg); use MP4 or WebM", .{}),
    };
    var n: u32 = 0;
    var best: ?u32 = null; // max(got)
    while (true) {
        const more = c.advance() catch return invalidMessage(diag, c.code);
        if (!more) break;
        if (n > last_pick or std.Io.Clock.awake.now(io).toNanoseconds() >= deadline_ns) break;
        if (wanted.getPtr(n)) |slot| {
            if (slot.len == 0) {
                const frame = c.convert(a) catch |e| switch (e) {
                    error.OutOfMemory => return error.OutOfMemory,
                    error.Failed => return invalidMessage(diag, c.code),
                };
                defer a.free(frame.pixels);
                const buf = try a.alloc(u8, plane);
                errdefer a.free(buf);
                try resize.fitInto(a, frame.pixels, frame.h, frame.w, canvas, content, buf);
                slot.* = buf;
                if (best == null or n > best.?) best = n;
            }
        }
        n += 1;
    }
    if (std.Io.Clock.awake.now(io).toNanoseconds() >= deadline_ns) return errors.fail(diag, "video decoding exceeds the time limit", .{});
    const last = best orelse return errors.fail(diag, "the video has no decodable frames", .{});
    const frames = try a.alloc(u8, picks.len * plane);
    errdefer a.free(frames);
    for (picks, 0..) |p, i| {
        const src_idx: u32 = if (p <= last) p else last;
        const slot = wanted.get(src_idx).?;
        const pick_src = if (slot.len != 0) slot else wanted.get(last).?; // got.get(i, got[last])
        @memcpy(frames[i * plane ..][0..plane], pick_src);
    }
    const seconds = try a.alloc(f64, picks.len / 2);
    errdefer a.free(seconds);
    var k: usize = 0;
    while (k < picks.len) : (k += 2) seconds[k / 2] = @as(f64, @floatFromInt(picks[k])) / fps;
    return .{ .frames = frames, .t = @intCast(picks.len), .h = canvas.h, .w = canvas.w, .seconds = seconds };
}

const fixtures = @import("fixtures.zig");

fn readJson(a: Allocator, name: []const u8) !std.json.Parsed(std.json.Value) {
    const text = try fixtures.read(a, name, 1 << 24);
    defer a.free(text);
    return std.json.parseFromSlice(std.json.Value, a, text, .{ .allocate = .alloc_always });
}

fn loadOrSkip(a: Allocator) !av.Libs {
    return av.load(a, std.testing.io) catch |e| switch (e) {
        error.LibrariesMissing => return error.SkipZigTest,
        else => return e,
    };
}

test "FFmpeg frames are byte-identical to PyAV's (rgb24 per frame, stream properties)" {
    const a = std.testing.allocator;
    var libs = try loadOrSkip(a);
    defer libs.close();
    const parsed = try readJson(a, "vid/expected.json");
    defer parsed.deinit();
    var checked: usize = 0;
    for (parsed.value.array.items) |e| {
        const o = e.object;
        const name = o.get("file").?.string;
        var path_buf: [128]u8 = undefined;
        const data = try fixtures.read(a, try std.fmt.bufPrint(&path_buf, "vid/{s}", .{name}), 1 << 24);
        defer a.free(data);
        var c = try av.Clip.open(a, &libs, data);
        defer c.close();
        try std.testing.expectEqual(@as(u32, @intCast(o.get("width").?.integer)), c.info.width);
        try std.testing.expectEqual(@as(u32, @intCast(o.get("height").?.integer)), c.info.height);
        try std.testing.expectEqual(@as(i64, @intCast(o.get("frames").?.integer)), c.info.nb_frames);
        if (o.get("avg_rate").?.array.items.len == 2) {
            try std.testing.expectEqual(@as(i32, @intCast(o.get("avg_rate").?.array.items[0].integer)), c.info.avg_num);
        }
        try c.openDecoder();
        const hashes = o.get("frame_hashes").?.array.items;
        var n: usize = 0;
        while (try c.advance()) : (n += 1) {
            const rgb = try c.convert(a);
            defer a.free(rgb.pixels);
            try std.testing.expect(n < hashes.len);
            const hex = fixtures.hexDigest(rgb.pixels);
            if (!std.mem.eql(u8, hashes[n].string, &hex)) {
                std.debug.print("frame {d} of {s} differs\n", .{ n, name });
                return error.FrameMismatch;
            }
        }
        try std.testing.expectEqual(hashes.len, n);
        checked += 1;
    }
    try std.testing.expect(checked >= 15);
}

test "decode_clip: canvases, frame-pair times and errors as the recipe" {
    const a = std.testing.allocator;
    var libs = try loadOrSkip(a);
    defer libs.close();
    const parsed = try readJson(a, "vid/clips.json");
    defer parsed.deinit();
    for (parsed.value.array.items) |e| {
        const o = e.object;
        const name = o.get("file").?.string;
        var path_buf: [128]u8 = undefined;
        const data = try fixtures.read(a, try std.fmt.bufPrint(&path_buf, "vid/{s}", .{name}), 1 << 24);
        defer a.free(data);
        var lim: limits_mod.GlmVisionLimits = .{};
        lim.video_tokens = @intCast(o.get("video_tokens").?.integer);
        lim.video_frames = @intCast(o.get("video_frames").?.integer);
        var diag: Diag = .{};
        const res = decodeClip(a, std.testing.io, &libs, data, .{ .limits = lim }, std.math.maxInt(i96), &diag);
        if (o.get("error")) |want| {
            if (res) |ok| {
                var it: items_mod.Item = .{ .clip = ok };
                it.deinit(a);
                std.debug.print("{s}/{s}: decoded but the recipe refuses ({s})\n", .{ name, o.get("variant").?.string, want.string });
                return error.Mismatch;
            } else |_| try std.testing.expectEqualStrings(want.string, diag.message());
            continue;
        }
        var clip = res catch |err| {
            std.debug.print("{s}/{s}: refused ({s})\n", .{ name, o.get("variant").?.string, diag.message() });
            return err;
        };
        var item: items_mod.Item = .{ .clip = clip };
        defer item.deinit(a);
        const shape = o.get("shape").?.array.items;
        try std.testing.expectEqual(@as(i64, @intCast(clip.t)), shape[0].integer);
        try std.testing.expectEqual(@as(i64, @intCast(clip.h)), shape[2].integer);
        try std.testing.expectEqual(@as(i64, @intCast(clip.w)), shape[3].integer);
        const secs = o.get("seconds").?.array.items;
        try std.testing.expectEqual(secs.len, clip.seconds.len);
        for (secs, clip.seconds) |s, got| try std.testing.expectEqual(try std.fmt.parseFloat(f64, s.string), got);
        const hash = items_mod.clipContentHash(clip);
        if (!std.mem.eql(u8, o.get("hash").?.string, &hash)) {
            std.debug.print("{s}/{s}: pixels differ\n", .{ name, o.get("variant").?.string });
            return error.PixelMismatch;
        }
        clip = undefined;
    }
}
