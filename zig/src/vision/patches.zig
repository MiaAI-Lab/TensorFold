//! Canvas -> patches: `GlmVisionTower.patches` on the CPU, the unit-test oracle for the GPU tower.
//!
//! Input: uint8 frames [T, 3, H, W] (T even, H and W multiples of 28), the canvas as `fit` leaves it.
//! Output: float32 [T/2 * (H/14) * (W/14), 1176], row-major, where
//!   x = (frames.float() - mean255) / std255     (IEEE fp32 subtract, then fp32 divide; mean255 = float32(MEAN) * 255
//!                                                and std255 = float32(STD) * 255, each product rounded to fp32)
//!   x.view(T/2, 2, C, H/14/2, 2, 14, W/14/2, 2, 14)
//!    .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)         # (pair, hblock, wblock, mh, mw, channel, t, y, x)
//!    .reshape(T/2 * (H/14) * (W/14), 3 * 2 * 14 * 14)
//! so rows are the 2x2 merge blocks' patches (pair-major, then merge block row/column, then the position inside
//! the block) and a row's 1176 values run (channel, time within the pair, y in the patch, x in the patch).
//! A single picture is encoded as a frame pair of the same frame twice (`frames.unsqueeze(0).expand(2, ...)`).
const std = @import("std");

pub const MEAN = [3]f32{ 0.48145466, 0.4578275, 0.40821073 }; // CLIP's, from processor_config.json
pub const STD = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };
pub const patch = 14;
pub const merge = 2;
pub const temporal = 2;
pub const row_len = 3 * temporal * patch * patch;

pub fn mean255() [3]f32 {
    var out: [3]f32 = undefined;
    for (MEAN, &out) |m, *o| o.* = m * @as(f32, 255.0);
    return out;
}

pub fn std255() [3]f32 {
    var out: [3]f32 = undefined;
    for (STD, &out) |m, *o| o.* = m * @as(f32, 255.0);
    return out;
}

pub fn rowCount(t: usize, h: usize, w: usize) usize {
    return t / 2 * (h / patch) * (w / patch);
}

/// Fill `out` (rowCount * 1176 floats) from `frames` ([t,3,h,w] uint8).
pub fn patches(out: []f32, frames: []const u8, t: usize, h: usize, w: usize) void {
    std.debug.assert(t % 2 == 0 and h % 28 == 0 and w % 28 == 0);
    std.debug.assert(frames.len == t * 3 * h * w and out.len == rowCount(t, h, w) * row_len);
    const m255 = mean255();
    const s255 = std255();
    const gh = h / patch / merge;
    const gw = w / patch / merge;
    var row: usize = 0;
    for (0..t / 2) |pair| {
        for (0..gh) |hb| for (0..gw) |wb| for (0..merge) |mh| for (0..merge) |mw| {
            var col: usize = 0;
            for (0..3) |c| for (0..temporal) |tt| for (0..patch) |py| {
                const y = (hb * merge + mh) * patch + py;
                const base = (((pair * 2 + tt) * 3 + c) * h + y) * w + (wb * merge + mw) * patch;
                const dst = out[row * row_len + col ..][0..patch];
                for (dst, frames[base..][0..patch]) |*d, v| d.* = (@as(f32, @floatFromInt(v)) - m255[c]) / s255[c];
                col += patch;
            };
            row += 1;
        };
    }
}

/// One fitted picture ([3, H, W] uint8) as the tower encodes it: the same frame twice (`unsqueeze(0).expand(2, ...)`).
pub fn patchesImage(out: []f32, canvas: []const u8, h: usize, w: usize) void {
    std.debug.assert(canvas.len == 3 * h * w and out.len == rowCount(2, h, w) * row_len);
    const m255 = mean255();
    const s255 = std255();
    const gh = h / patch / merge;
    const gw = w / patch / merge;
    var row: usize = 0;
    for (0..gh) |hb| for (0..gw) |wb| for (0..merge) |mh| for (0..merge) |mw| {
        var col: usize = 0;
        for (0..3) |c| for (0..temporal) |_| for (0..patch) |py| {
            const y = (hb * merge + mh) * patch + py;
            const base = (c * h + y) * w + (wb * merge + mw) * patch;
            const dst = out[row * row_len + col ..][0..patch];
            for (dst, canvas[base..][0..patch]) |*d, v| d.* = (@as(f32, @floatFromInt(v)) - m255[c]) / s255[c];
            col += patch;
        };
        row += 1;
    };
}

pub fn alloc(allocator: std.mem.Allocator, frames: []const u8, t: usize, h: usize, w: usize) ![]f32 {
    const out = try allocator.alloc(f32, rowCount(t, h, w) * row_len);
    patches(out, frames, t, h, w);
    return out;
}

const fixtures = @import("fixtures.zig");

test "patches match GlmVisionTower.patches bit for bit" {
    const a = std.testing.allocator;
    const text = try fixtures.read(a, "patches.txt", 1 << 16);
    defer a.free(text);
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    var n: usize = 0;
    while (lines.next()) |line| {
        var t = std.mem.tokenizeScalar(u8, line, ' ');
        _ = t.next();
        const seed = try std.fmt.parseInt(u64, t.next().?, 10);
        const T = try std.fmt.parseInt(usize, t.next().?, 10);
        const H = try std.fmt.parseInt(usize, t.next().?, 10);
        const W = try std.fmt.parseInt(usize, t.next().?, 10);
        const rows = try std.fmt.parseInt(usize, t.next().?, 10);
        const want = t.next().?;
        const frames = try a.alloc(u8, T * 3 * H * W);
        defer a.free(frames);
        fixtures.lcgBytes(frames, seed);
        const out = try alloc(a, frames, T, H, W);
        defer a.free(out);
        try std.testing.expectEqual(rows * row_len, out.len);
        const got = fixtures.hexDigest(std.mem.sliceAsBytes(out));
        try std.testing.expectEqualStrings(want, &got);
        n += 1;
    }
    try std.testing.expectEqual(@as(usize, 6), n);
}

test "a picture's patches equal those of the same frame twice" {
    const a = std.testing.allocator;
    const h = 56;
    const w = 84;
    var canvas: [3 * h * w]u8 = undefined;
    fixtures.lcgBytes(&canvas, 321);
    var doubled: [2 * 3 * h * w]u8 = undefined;
    @memcpy(doubled[0 .. 3 * h * w], &canvas);
    @memcpy(doubled[3 * h * w ..], &canvas);
    const want = try alloc(a, &doubled, 2, h, w);
    defer a.free(want);
    const got = try a.alloc(f32, want.len);
    defer a.free(got);
    patchesImage(got, &canvas, h, w);
    try std.testing.expectEqualSlices(f32, want, got);
}
