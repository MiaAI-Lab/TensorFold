//! Port of glm5_next/cuda/qmm.py quantize4_mse_raw, make_f8 and cuda/kernels/qmm.py pack.
//! Authored by Ash Hart and TensorFold contributors; numeric conversion helpers from TensorFold's Flash Next port.
const std = @import("std");
pub fn f32Bits(x: f32) u32 {
    return @bitCast(x);
}

pub fn bitsF32(b: u32) f32 {
    return @bitCast(b);
}

pub fn bf16ToF32(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// fp32 -> bf16, round to nearest even; NaN -> 0x7FC0 (c10::BFloat16's round_to_nearest_even).
pub fn bf16Rne(x: f32) u16 {
    const u = f32Bits(x);
    if (std.math.isNan(x)) return 0x7FC0;
    const r = u +% 0x7FFF +% ((u >> 16) & 1);
    return @intCast(r >> 16);
}

/// e4m3fn byte -> its exact fp32 value (no infinities; 0x7F and 0xFF are NaN).
pub fn e4m3ToF32(b: u8) f32 {
    const e: u32 = (b >> 3) & 0xF;
    const m: u32 = b & 0x7;
    const sign: u32 = @as(u32, b & 0x80) << 24;
    if (e == 15 and m == 7) return bitsF32(0x7FC00000 | sign);
    if (e == 0) {
        const v: f32 = @as(f32, @floatFromInt(m)) * (1.0 / 512.0); // m * 2**-9, exact
        return bitsF32(f32Bits(v) | sign);
    }
    return bitsF32(sign | ((e + 120) << 23) | (m << 20));
}

/// fp32 -> e4m3fn, round to nearest even, as c10's fp8e4m3fn_from_fp32_value (|x| >= 480 -> NaN 0x7F).
pub fn f32ToE4m3(x: f32) u8 {
    var bits = f32Bits(x);
    const sign = bits & 0x80000000;
    bits ^= sign;
    var out: u8 = undefined;
    if (bits >= (@as(u32, 1087) << 20)) {
        out = 0x7F;
    } else if (bits < (@as(u32, 121) << 23)) {
        const denorm: u32 = @as(u32, 141) << 23;
        const sum = bitsF32(bits) + bitsF32(denorm);
        out = @intCast(f32Bits(sum) - denorm);
    } else {
        const odd = (bits >> 20) & 1;
        bits = bits -% (@as(u32, 120) << 23) +% 0x7FFFF; // the exponent rebased by 7 - 127, then RNE
        bits +%= odd;
        out = @truncate(bits >> 20);
    }
    return out | @as(u8, @intCast(sign >> 24));
}

pub fn roundEven(x: f32) f32 {
    const a = @floor(x);
    const d = x - a;
    return if (d < 0.5 or (d == 0.5 and @mod(a, 2) == 0)) a else a + 1;
}
fn code(x: f32, s: f32, b: f32) u32 {
    return @intFromFloat(std.math.clamp(roundEven((x - b) / s), 0, 15));
}
/// Input BF16 [n,k]; raw MLX words [n,k/8], scales and biases [n,k/64].
/// MSE sum uses a fixed binary tree; CUDA ATen reduction parity must be verified before loader integration.
pub fn quantize4(rows: []const u16, n: usize, k: usize, words: []u32, scales: []u16, biases: []u16, mse: bool) void {
    @setFloatMode(.strict);
    std.debug.assert(k % 64 == 0 and rows.len == n * k and words.len == n * k / 8 and scales.len == n * k / 64 and biases.len == scales.len);
    const kg = k / 64;
    for (0..n * kg) |g| {
        const x = rows[g * 64 ..][0..64];
        var lo: f32 = std.math.inf(f32);
        var hi: f32 = -std.math.inf(f32);
        for (x) |v| {
            const w = bf16ToF32(v);
            lo = @min(lo, w);
            hi = @max(hi, w);
        }
        var best: f32 = std.math.inf(f32);
        var bs: u16 = 0;
        var bb: u16 = 0;
        for (0..if (mse) @as(usize, 16) else 1) |i| {
            const a: f32 = @floatCast(1.0 - 0.45 * @as(f64, @floatFromInt(i)) / 15.0);
            const l = lo * a;
            const h = hi * a;
            const sb = bf16Rne(@max((h - l) / 15.0, 1e-8));
            const bi = bf16Rne(l);
            const s = bf16ToF32(sb);
            const b = bf16ToF32(bi);
            var errs: [64]f32 = undefined;
            for (x, &errs) |v, *e| {
                const w = bf16ToF32(v);
                const q: f32 = @floatFromInt(code(w, s, b));
                const d = q * s + b - w;
                e.* = d * d;
            }
            var step: usize = 32;
            while (step > 0) : (step /= 2) for (0..step) |j| {
                errs[j] += errs[j + step];
            };
            if (i == 0 or errs[0] < best) {
                best = errs[0];
                bs = sb;
                bb = bi;
            }
        }
        scales[g] = bs;
        biases[g] = bb;
        for (0..8) |j| {
            var w: u32 = 0;
            for (0..8) |p| w |= code(bf16ToF32(x[j * 8 + p]), bf16ToF32(bs), bf16ToF32(bb)) << @as(u5, @intCast(p * 4));
            words[g * 8 + j] = w;
        }
    }
}
/// Tile permutation [npad/64, k/64, 8,32,2], and group-major scale/bias arrays, n padded to 128.
pub fn pack(words: []const u32, scales: []const u16, biases: []const u16, n: usize, k: usize, out: []u32, so: []u16, bo: []u16) void {
    const npad = (n + 127) / 128 * 128;
    const kg = k / 64;
    std.debug.assert(k % 64 == 0 and out.len == npad * k / 8 and so.len == npad * kg and bo.len == so.len);
    @memset(out, 0);
    @memset(so, 0);
    @memset(bo, 0);
    packRows(words, scales, biases, n, k, out, so, bo, 0, n);
}
/// pack for source rows r0..r1 only (the outputs of different rows are disjoint; the caller zeroed them).
pub fn packRows(words: []const u32, scales: []const u16, biases: []const u16, n: usize, k: usize, out: []u32, so: []u16, bo: []u16, r0: usize, r1: usize) void {
    const npad = (n + 127) / 128 * 128;
    const kg = k / 64;
    const offsets = [_]usize{ 0, 8, 16, 24, 1, 9, 17, 25 };
    for (r0..r1) |r| for (0..kg) |g| {
        so[g * npad + r] = scales[r * kg + g];
        bo[g * npad + r] = biases[r * kg + g];
        for (0..4) |c| for (0..2) |v| {
            var w: u32 = 0;
            for (offsets, 0..) |off, p| {
                const col = g * 64 + v * 32 + 2 * c + off;
                const q = (words[r * (k / 8) + col / 8] >> @as(u5, @intCast((col % 8) * 4))) & 15;
                w |= q << @as(u5, @intCast(p * 4));
            }
            out[((r / 64 * kg + g) * 8 + r % 64 / 8) * 64 + (r % 8 * 4 + c) * 2 + v] = w;
        };
    };
}
pub fn makeF8(rows: []const u16, codes: []u8, scales: []f32) void {
    @setFloatMode(.strict);
    std.debug.assert(rows.len % 128 == 0 and codes.len == rows.len and scales.len == rows.len / 128);
    for (scales, 0..) |*s, g| {
        var top: f32 = 0;
        for (rows[g * 128 ..][0..128]) |v| top = @max(top, @abs(bf16ToF32(v)));
        // ATen divides a tensor by a Python scalar as a multiply by the fp32 reciprocal.
        const recip: f32 = 1.0 / 448.0;
        s.* = @max(top * recip, 1e-30);
        for (0..128) |j| codes[g * 128 + j] = f32ToE4m3(std.math.clamp(bf16ToF32(rows[g * 128 + j]) / s.*, -448, 448));
    }
}
/// Checkpoint int16 trellis bits, [K/16,N/16,64], viewed little endian int32 without reordering.
pub fn trellisWords(src: []const u16, dst: []u32) void {
    std.debug.assert(src.len == dst.len * 2);
    for (dst, 0..) |*w, i| w.* = @as(u32, src[2 * i]) | (@as(u32, src[2 * i + 1]) << 16);
}
pub fn sharedSuh(suh: []const u16, up: []const u16, experts: usize, k: usize) bool {
    if (!std.mem.eql(u16, suh, up)) return false;
    std.debug.assert(suh.len == experts * k and experts > 0);
    for (1..experts) |e| if (!std.mem.eql(u16, suh[0..k], suh[e * k ..][0..k])) return false;
    return true;
}
test "MSE ties, padded pack, FP8 and trellis" {
    var x: [64]u16 = @splat(bf16Rne(1));
    var w: [8]u32 = undefined;
    var s: [1]u16 = undefined;
    var b: [1]u16 = undefined;
    quantize4(&x, 1, 64, &w, &s, &b, true);
    try std.testing.expectEqual(bf16Rne(1), b[0]);
    try std.testing.expectEqual(@as(u32, 0), w[0]);
    var out: [1024]u32 = undefined;
    var so: [128]u16 = undefined;
    var bo: [128]u16 = undefined;
    pack(&w, &s, &b, 1, 64, &out, &so, &bo);
    try std.testing.expectEqual(b[0], bo[0]);
    try std.testing.expectEqual(@as(u16, 0), bo[1]);
    var t: [1]u32 = undefined;
    trellisWords(&.{ 0x1234, 0xabcd }, &t);
    try std.testing.expectEqual(@as(u32, 0xabcd1234), t[0]);
    try std.testing.expectEqual(@as(f32, 2), roundEven(2.5));
    try std.testing.expectEqual(@as(f32, 4), roundEven(3.5));
    var fp: [128]u8 = undefined;
    var fs: [1]f32 = undefined;
    const fx: [128]u16 = @splat(bf16Rne(448));
    makeF8(&fx, &fp, &fs);
    try std.testing.expectEqual(@as(f32, 1), fs[0]);
    try std.testing.expectEqual(@as(u8, 0x7e), fp[0]);
}
test "nonzero g64 permutation across N padding and multiple K groups" {
    const n = 65;
    const k = 128;
    const npad = 128;
    var words: [n * k / 8]u32 = @splat(0);
    var sc: [n * k / 64]u16 = undefined;
    var bi: [n * k / 64]u16 = undefined;
    for (0..n) |r| for (0..k) |c| {
        const nib: u32 = @intCast((r * 3 + c / 32 + c * 7) % 16);
        words[r * k / 8 + c / 8] |= nib << @as(u5, @intCast(4 * (c % 8)));
    };
    for (&sc, &bi, 0..) |*s, *b, i| {
        s.* = @intCast(i + 1);
        b.* = @intCast(1000 + i);
    }
    var dst: [npad * k / 8]u32 = undefined;
    var so: [npad * k / 64]u16 = undefined;
    var bo: [npad * k / 64]u16 = undefined;
    pack(&words, &sc, &bi, n, k, &dst, &so, &bo);
    const offsets = [_]usize{ 0, 8, 16, 24, 1, 9, 17, 25 };
    for (dst, 0..) |w, idx| {
        const v = idx % 2;
        const lane = idx / 2 % 32;
        const j = idx / 64 % 8;
        const group = idx / 512 % 2;
        const block = idx / 1024;
        const r = block * 64 + j * 8 + lane / 4;
        for (offsets, 0..) |off, p| {
            const c = group * 64 + v * 32 + 2 * (lane % 4) + off;
            const expected: u32 = if (r < n) @intCast((r * 3 + c / 32 + c * 7) % 16) else 0;
            try std.testing.expectEqual(expected, (w >> @as(u5, @intCast(4 * p))) & 15);
        }
    }
    for (0..2) |g| for (0..npad) |r| {
        try std.testing.expectEqual(if (r < n) sc[r * 2 + g] else @as(u16, 0), so[g * npad + r]);
        try std.testing.expectEqual(if (r < n) bi[r * 2 + g] else @as(u16, 0), bo[g * npad + r]);
    };
}
test "shared gate and up suh compares bits including signed zero" {
    try std.testing.expect(sharedSuh(&.{ 0, 0x8000, 0, 0x8000 }, &.{ 0, 0x8000, 0, 0x8000 }, 2, 2));
    try std.testing.expect(!sharedSuh(&.{ 0, 0x8000, 0, 0x8000 }, &.{ 0, 0, 0, 0 }, 2, 2));
    try std.testing.expect(!sharedSuh(&.{ 0, 0, 0, 0x8000 }, &.{ 0, 0, 0, 0x8000 }, 2, 2));
}
test "MSE clipping improves an outlier group over the full range" {
    var x: [64]u16 = undefined;
    for (&x, 0..) |*v, i| v.* = bf16Rne(if (i == 63) @as(f32, 4) else @as(f32, @floatFromInt(i % 9)) * 0.0625 - 0.25);
    var w: [8]u32 = undefined;
    var s: [1]u16 = undefined;
    var b: [1]u16 = undefined;
    var mw: [8]u32 = undefined;
    var ms: [1]u16 = undefined;
    var mb: [1]u16 = undefined;
    quantize4(&x, 1, 64, &w, &s, &b, false);
    quantize4(&x, 1, 64, &mw, &ms, &mb, true);
    var err: f32 = 0;
    var merr: f32 = 0;
    for (x, 0..) |v, i| {
        const q: f32 = @floatFromInt((w[i / 8] >> @as(u5, @intCast(i % 8 * 4))) & 15);
        const mq: f32 = @floatFromInt((mw[i / 8] >> @as(u5, @intCast(i % 8 * 4))) & 15);
        const d = q * bf16ToF32(s[0]) + bf16ToF32(b[0]) - bf16ToF32(v);
        const md = mq * bf16ToF32(ms[0]) + bf16ToF32(mb[0]) - bf16ToF32(v);
        err += d * d;
        merr += md * md;
    }
    try std.testing.expect(merr <= err);
}
