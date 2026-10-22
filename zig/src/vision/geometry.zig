//! The processors' geometry (glm.py): smart_resize, content_size, sample_frames. Float math as CPython/numpy do it.
const std = @import("std");
const errors = @import("errors.zig");
const limits_mod = @import("limits.zig");
const Diag = errors.Diag;

pub const Size = struct { h: u32, w: u32 };

/// Python's round() on a float: half to even.
pub fn pyRound(x: f64) i64 {
    const f = @floor(x);
    const d = x - f;
    if (d < 0.5) return @intFromFloat(f);
    if (d > 0.5) return @intFromFloat(f + 1);
    const fi: i64 = @intFromFloat(f);
    return if (@mod(fi, 2) == 0) fi else fi + 1;
}

fn alignUp(v: u64, factor: u64) u64 {
    // math.ceil(v / factor) * factor, v and factor small ints
    return (v + factor - 1) / factor * factor;
}

fn ceilToU64(x: f64) u64 {
    return @intFromFloat(@ceil(x));
}

/// `smart_resize(frames, height, width, max_tokens, min_tokens=16, temporal=2, factor=28)`.
pub fn smartResize(frames: u32, height: u32, width: u32, max_tokens: u64, min_tokens: u64, diag: ?*Diag) errors.Error!Size {
    const temporal: u64 = 2;
    const factor: u64 = 28;
    const per_token = temporal * factor * factor;
    const low_pixels = min_tokens * per_token;
    const high_pixels = max_tokens * per_token;

    const aligned: u64 = @max(temporal, @as(u64, @intCast(pyRound(@as(f64, @floatFromInt(frames)) / @as(f64, @floatFromInt(temporal))))) * temporal);
    var h = alignUp(height, factor);
    var w = alignUp(width, factor);
    if (aligned * h * w < low_pixels) {
        const scale = @sqrt(@as(f64, @floatFromInt(low_pixels)) / @as(f64, @floatFromInt(@as(u64, frames) * height * width)));
        h = alignUp(@max(1, ceilToU64(@as(f64, @floatFromInt(height)) * scale)), factor);
        w = alignUp(@max(1, ceilToU64(@as(f64, @floatFromInt(width)) * scale)), factor);
    }
    if (aligned * h * w > high_pixels) {
        if (high_pixels < aligned * factor * factor) return errors.fail(diag, "the token cap is too small for one patch per frame pair", .{});
        var lo: u64 = 1;
        var hi: u64 = height;
        h = factor;
        w = factor;
        while (lo <= hi) {
            const mid = (lo + hi) / 2;
            const ch = alignUp(mid, factor);
            const cw = alignUp(@max(1, @as(u64, @intFromFloat(@floor(@as(f64, @floatFromInt(@as(u64, width) * mid)) / @as(f64, @floatFromInt(height)))))), factor);
            if (aligned * ch * cw <= high_pixels) {
                h = ch;
                w = cw;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
    }
    return .{ .h = @intCast(h), .w = @intCast(w) };
}

/// `content_size(frames, height, width, canvas, min_tokens=16)`: the resized picture inside the canvas.
pub fn contentSize(frames: u32, height: u32, width: u32, canvas: Size, min_tokens: u64) Size {
    const temporal: u64 = 2;
    const factor: u64 = 28;
    const th: f64 = @floatFromInt(canvas.h);
    const tw: f64 = @floatFromInt(canvas.w);
    const fh: f64 = @floatFromInt(height);
    const fw: f64 = @floatFromInt(width);
    var scale = @min(th / fh, tw / fw);
    if (@as(u64, frames) * height * width >= temporal * factor * factor * min_tokens) scale = @min(1.0, scale);
    const ch: u64 = @max(1, @min(@as(u64, canvas.h), @as(u64, @intFromFloat(@floor(fh * scale)))));
    const cw: u64 = @max(1, @min(@as(u64, canvas.w), @as(u64, @intFromFloat(@floor(fw * scale)))));
    return .{ .h = @intCast(ch), .w = @intCast(cw) };
}

/// numpy.linspace(start, stop, num, dtype=int).tolist() (endpoint, floored).
pub fn linspaceInt(allocator: std.mem.Allocator, start: i64, stop: i64, num: usize) ![]u32 {
    const out = try allocator.alloc(u32, num);
    errdefer allocator.free(out);
    if (num == 0) return out;
    const fstart: f64 = @floatFromInt(start);
    const fstop: f64 = @floatFromInt(stop);
    if (num == 1) {
        out[0] = @intCast(start);
        return out;
    }
    const div: f64 = @floatFromInt(num - 1);
    const delta = fstop - fstart;
    const step = delta / div;
    for (out, 0..) |*o, i| {
        const fi: f64 = @floatFromInt(i);
        var y = if (step == 0) (fi / div) * delta else fi * step;
        y += fstart;
        if (i == num - 1) y = fstop;
        o.* = @intCast(@as(i64, @intFromFloat(@floor(y))));
    }
    return out;
}

/// `sample_frames(total, fps, duration, limits)`: one frame each 1/fps seconds, at most video_frames, duplicates
/// dropped, an even count (the last frame repeated). Caller frees the result.
pub fn sampleFrames(allocator: std.mem.Allocator, total: u32, fps: f64, duration_in: f64, limits: limits_mod.GlmVisionLimits) ![]u32 {
    var duration = duration_in;
    if (duration == 0) duration = @floatFromInt(pyRound(@as(f64, @floatFromInt(@as(i64, total) - 1)) / fps) + 1);
    const max_seconds: f64 = @floatFromInt(@as(i64, @intFromFloat(duration)));
    const want_f = @as(i64, @intFromFloat(duration * limits.fps));
    const want: usize = @intCast(@min(want_f, @as(i64, limits.video_frames)));
    var picks: []u32 = undefined;
    if (total < want) {
        picks = try linspaceInt(allocator, 0, @as(i64, total) - 1, want);
    } else {
        var list: std.ArrayList(u32) = .empty;
        errdefer list.deinit(allocator);
        var second: f64 = 0.0;
        const step: f64 = 1.0 / limits.fps;
        var i: u32 = 0;
        while (i < total) : (i += 1) {
            if (@as(f64, @floatFromInt(i)) / fps >= second) {
                second += step;
                try list.append(allocator, i);
                if (second >= max_seconds) break;
            }
        }
        picks = try list.toOwnedSlice(allocator);
    }
    errdefer allocator.free(picks);
    if (picks.len < want) {
        const s: i64, const e: i64 = if (picks.len > 0) .{ picks[0], picks[picks.len - 1] } else .{ 0, @max(@as(i64, total) - 1, 0) };
        const n = try linspaceInt(allocator, s, e, want);
        allocator.free(picks);
        picks = n;
    } else if (picks.len > want) {
        const n = try linspaceInt(allocator, 0, @as(i64, total) - 1, want);
        allocator.free(picks);
        picks = n;
    }
    // unique, order kept; at least [0]; even count
    var uniq: std.ArrayList(u32) = .empty;
    errdefer uniq.deinit(allocator);
    var seen: std.AutoHashMapUnmanaged(u32, void) = .empty;
    defer seen.deinit(allocator);
    for (picks) |p| {
        const gop = try seen.getOrPut(allocator, p);
        if (!gop.found_existing) try uniq.append(allocator, p);
    }
    allocator.free(picks);
    if (uniq.items.len == 0) try uniq.append(allocator, 0);
    if (uniq.items.len % 2 == 1) try uniq.append(allocator, uniq.items[uniq.items.len - 1]);
    return uniq.toOwnedSlice(allocator);
}

/// A canvas's placeholder tokens: (H // 28) * (W // 28) a picture, and the same per frame pair.
pub fn rowsPerFrame(h: usize, w: usize) usize {
    return h / 28 * (w / 28);
}

const fixtures = @import("fixtures.zig");

test "smart_resize / content_size against the Python processors" {
    const a = std.testing.allocator;
    const text = try fixtures.read(a, "geometry_smart.txt", 1 << 22);
    defer a.free(text);
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    var n: usize = 0;
    while (lines.next()) |line| {
        var t = std.mem.tokenizeScalar(u8, line, ' ');
        _ = t.next(); // "smart"
        const frames = try std.fmt.parseInt(u32, t.next().?, 10);
        const h = try std.fmt.parseInt(u32, t.next().?, 10);
        const w = try std.fmt.parseInt(u32, t.next().?, 10);
        const mx = try std.fmt.parseInt(u64, t.next().?, 10);
        const mn = try std.fmt.parseInt(u64, t.next().?, 10);
        const ch = t.next().?;
        var diag: Diag = .{};
        if (std.mem.eql(u8, ch, "ERR")) {
            try std.testing.expectError(error.Input, smartResize(frames, h, w, mx, mn, &diag));
            try std.testing.expectEqualStrings(t.rest(), diag.message());
        } else {
            const cw = try std.fmt.parseInt(u32, t.next().?, 10);
            const sh = try std.fmt.parseInt(u32, t.next().?, 10);
            const sw = try std.fmt.parseInt(u32, t.next().?, 10);
            const got = try smartResize(frames, h, w, mx, mn, &diag);
            try std.testing.expectEqual(try std.fmt.parseInt(u32, ch, 10), got.h);
            try std.testing.expectEqual(cw, got.w);
            const c = contentSize(frames, h, w, got, mn);
            try std.testing.expectEqual(sh, c.h);
            try std.testing.expectEqual(sw, c.w);
        }
        n += 1;
    }
    try std.testing.expect(n >= 1500);
}

test "sample_frames against the Python processor" {
    const a = std.testing.allocator;
    const text = try fixtures.read(a, "geometry_sample.txt", 1 << 24);
    defer a.free(text);
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    var n: usize = 0;
    while (lines.next()) |line| {
        var t = std.mem.tokenizeScalar(u8, line, ' ');
        _ = t.next();
        const total = try std.fmt.parseInt(u32, t.next().?, 10);
        const fps = try std.fmt.parseFloat(f64, t.next().?);
        const dur = try std.fmt.parseFloat(f64, t.next().?);
        const vf = try std.fmt.parseInt(u32, t.next().?, 10);
        const count = try std.fmt.parseInt(usize, t.next().?, 10);
        const got = try sampleFrames(a, total, fps, dur, .{ .video_frames = vf });
        defer a.free(got);
        try std.testing.expectEqual(count, got.len);
        for (got) |g| try std.testing.expectEqual(try std.fmt.parseInt(u32, t.next().?, 10), g);
        n += 1;
    }
    try std.testing.expect(n >= 600);
}
