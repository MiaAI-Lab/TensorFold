//! `fit()`: torchvision.transforms.v2.functional.resize(uint8 [3,H,W] CPU tensor, content, BICUBIC, antialias=True),
//! then zero padding to the canvas -- bit-exact.
//!
//! What torch 2.13 runs for a channels-last uint8 frame (the layout the processor hands it) is the integer
//! separable path of `_upsample_bicubic2d_aa` (Pillow's ImagingResample scheme): the weights are computed in
//! float64 (`HelperInterpCubic::aa_filter`, a = -0.5, support = 2 * max(scale, 1)), normalised, and converted to
//! int16 with a per-dimension fixed-point precision (the largest `p` < 22 such that 0.5 + wmax * 2^(p+1) < 2^15);
//! the horizontal pass runs first and rounds/clamps to uint8 (`(sum + 2^(p-1)) >> p`), then the vertical pass.
//! A pass whose size does not change is skipped. This is arithmetic on integers, so the AVX2 (x86) and NEON
//! (aarch64) kernels and the scalar loop below agree bit for bit; the NEON header of torch 2.13 documents the
//! same operations. (A W == 1 output would take a different, float kernel when the input is NCHW-contiguous;
//! `fit` is only ever called with channels-last frames, so it never does.)
const std = @import("std");
const Allocator = std.mem.Allocator;
const geometry = @import("geometry.zig");

fn cubicAA(x_in: f64) f64 {
    const a: f64 = -0.5;
    const x = @abs(x_in);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

const Coeffs = struct {
    xmin: []i64,
    size: []i64,
    weights: []i16, // out * ksize
    ksize: usize,
    precision: u5,

    fn deinit(c: Coeffs, a: Allocator) void {
        a.free(c.xmin);
        a.free(c.size);
        a.free(c.weights);
    }
};

fn truncToI64(x: f64) i64 {
    return @intFromFloat(@trunc(x));
}

fn computeCoeffs(a: Allocator, in_size: usize, out_size: usize) Allocator.Error!Coeffs {
    const scale: f64 = @as(f64, @floatFromInt(in_size)) / @as(f64, @floatFromInt(out_size));
    const support: f64 = if (scale >= 1.0) 2.0 * scale else 2.0;
    const invscale: f64 = if (scale >= 1.0) 1.0 / scale else 1.0;
    const ksize: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 1;
    const xmin = try a.alloc(i64, out_size);
    errdefer a.free(xmin);
    const size = try a.alloc(i64, out_size);
    errdefer a.free(size);
    const fw = try a.alloc(f64, out_size * ksize);
    defer a.free(fw);
    @memset(fw, 0);
    var wt_max: f64 = 0.0;
    for (0..out_size) |i| {
        const center = scale * (@as(f64, @floatFromInt(i)) + 0.5);
        const lo = @max(truncToI64(center - support + 0.5), 0);
        var xs = @min(truncToI64(center + support + 0.5), @as(i64, @intCast(in_size))) - lo;
        xs = std.math.clamp(xs, 0, @as(i64, @intCast(ksize)));
        xmin[i] = lo;
        size[i] = xs;
        const row = fw[i * ksize ..][0..ksize];
        var total: f64 = 0.0;
        var j: usize = 0;
        while (j < xs) : (j += 1) {
            const w = cubicAA((@as(f64, @floatFromInt(@as(i64, @intCast(j)) + lo)) - center + 0.5) * invscale);
            row[j] = w;
            total += w;
        }
        if (total != 0.0) {
            j = 0;
            while (j < xs) : (j += 1) {
                row[j] /= total;
                wt_max = @max(wt_max, row[j]);
            }
        }
    }
    var precision: u5 = 0;
    while (precision < 22) : (precision += 1) {
        const next: i64 = truncToI64(0.5 + wt_max * @as(f64, @floatFromInt(@as(u64, 1) << (precision + 1))));
        if (next >= (1 << 15)) break;
    }
    const weights = try a.alloc(i16, out_size * ksize);
    errdefer a.free(weights);
    const mul: f64 = @floatFromInt(@as(u64, 1) << precision);
    for (fw, weights) |w, *o| {
        const v: i64 = if (w < 0) truncToI64(-0.5 + w * mul) else truncToI64(0.5 + w * mul);
        o.* = @intCast(v);
    }
    return .{ .xmin = xmin, .size = size, .weights = weights, .ksize = ksize, .precision = precision };
}

inline fn clamp8(v: i32) u8 {
    return @intCast(std.math.clamp(v, 0, 255));
}

/// Interleaved RGB8 [in_h, in_w, 3] -> [out_h, out_w, 3], torch's uint8 bicubic antialias resize.
pub fn resizeRgb(a: Allocator, src: []const u8, in_h: usize, in_w: usize, out_h: usize, out_w: usize) Allocator.Error![]u8 {
    std.debug.assert(src.len == in_h * in_w * 3);
    if (in_h == out_h and in_w == out_w) return a.dupe(u8, src);
    const need_h = in_w != out_w;
    const need_v = in_h != out_h;
    var mid: []u8 = undefined;
    var mid_owned = false;
    defer if (mid_owned) a.free(mid);
    if (need_h) {
        const c = try computeCoeffs(a, in_w, out_w);
        defer c.deinit(a);
        mid = try a.alloc(u8, in_h * out_w * 3);
        mid_owned = true;
        const init: i32 = @as(i32, 1) << (c.precision - 1);
        for (0..in_h) |y| {
            const line = src[y * in_w * 3 ..][0 .. in_w * 3];
            const dst = mid[y * out_w * 3 ..][0 .. out_w * 3];
            for (0..out_w) |x| {
                const k = c.weights[x * c.ksize ..][0..c.ksize];
                const base: usize = @intCast(c.xmin[x]);
                const n: usize = @intCast(c.size[x]);
                var s0: i32 = init;
                var s1: i32 = init;
                var s2: i32 = init;
                for (0..n) |i| {
                    const w: i32 = k[i];
                    const p = line[(base + i) * 3 ..][0..3];
                    s0 += w * p[0];
                    s1 += w * p[1];
                    s2 += w * p[2];
                }
                dst[x * 3] = clamp8(s0 >> c.precision);
                dst[x * 3 + 1] = clamp8(s1 >> c.precision);
                dst[x * 3 + 2] = clamp8(s2 >> c.precision);
            }
        }
    } else {
        mid = @constCast(src);
    }
    if (!need_v) {
        if (mid_owned) {
            mid_owned = false;
            return mid;
        }
        return a.dupe(u8, mid);
    }
    const c = try computeCoeffs(a, in_h, out_h);
    defer c.deinit(a);
    const out = try a.alloc(u8, out_h * out_w * 3);
    errdefer a.free(out);
    const init: i32 = @as(i32, 1) << (c.precision - 1);
    const row_len = out_w * 3;
    for (0..out_h) |y| {
        const k = c.weights[y * c.ksize ..][0..c.ksize];
        const base: usize = @intCast(c.xmin[y]);
        const n: usize = @intCast(c.size[y]);
        const dst = out[y * row_len ..][0..row_len];
        for (0..row_len) |j| {
            var s: i32 = init;
            for (0..n) |i| s += @as(i32, k[i]) * mid[(base + i) * row_len + j];
            dst[j] = clamp8(s >> c.precision);
        }
    }
    return out;
}

/// `fit(frame, canvas, content)`: the interleaved RGB8 `rgb` [h, w, 3] resized to `content` (when it differs),
/// laid out planar into `out` ([3, canvas.h, canvas.w] uint8), zero padded right and bottom.
pub fn fitInto(a: Allocator, rgb: []const u8, h: usize, w: usize, canvas: geometry.Size, content: geometry.Size, out: []u8) Allocator.Error!void {
    std.debug.assert(out.len == 3 * @as(usize, canvas.h) * canvas.w);
    @memset(out, 0);
    const resized = try resizeRgb(a, rgb, h, w, content.h, content.w);
    defer a.free(resized);
    const plane = @as(usize, canvas.h) * canvas.w;
    for (0..content.h) |y| for (0..content.w) |x| {
        const px = resized[(y * content.w + x) * 3 ..][0..3];
        const o = y * canvas.w + x;
        out[o] = px[0];
        out[plane + o] = px[1];
        out[2 * plane + o] = px[2];
    };
}

const fixtures = @import("fixtures.zig");

test "bicubic antialias resize matches torch 2.13 on aarch64 (uint8, channels-last)" {
    const a = std.testing.allocator;
    const text = try fixtures.read(a, "resize.txt", 1 << 20);
    defer a.free(text);
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    var n: usize = 0;
    var bad: usize = 0;
    while (lines.next()) |line| {
        var t = std.mem.tokenizeScalar(u8, line, ' ');
        const seed = try std.fmt.parseInt(u64, t.next().?, 10);
        const h = try std.fmt.parseInt(usize, t.next().?, 10);
        const w = try std.fmt.parseInt(usize, t.next().?, 10);
        const oh = try std.fmt.parseInt(usize, t.next().?, 10);
        const ow = try std.fmt.parseInt(usize, t.next().?, 10);
        const want = t.next().?;
        const src = try a.alloc(u8, h * w * 3);
        defer a.free(src);
        fixtures.lcgBytes(src, seed);
        const got = try resizeRgb(a, src, h, w, oh, ow);
        defer a.free(got);
        const hex = fixtures.hexDigest(got);
        if (!std.mem.eql(u8, want, &hex)) {
            std.debug.print("RESIZE MISMATCH {d}x{d} -> {d}x{d}\n", .{ w, h, ow, oh });
            bad += 1;
        }
        n += 1;
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
    try std.testing.expect(n >= 300);
}
