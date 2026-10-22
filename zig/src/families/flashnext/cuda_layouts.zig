//! Flash Next's device layouts made on the host, byte for byte as the Python engine makes them on the GPU:
//!   packExpert   cuda/nvfp4/experts.py _pack: NVFP4 codes and e4m3 scales -> [N/32][K/32][144] int32 blocks
//!   tileBits     qwen4_exp/cuda/nvfp4.py _tile_bits: a bf16 grid -> [N/64][K/64][64][64] (the shared expert's tables)
//!   quantize4    qwen4_exp/cuda/bf16.py quantize4: bf16 rows -> MLX affine 4-bit in groups of 32 (the draft head)
//!   packQ4       cuda/kernels/qmm.py pack: MLX words -> the lane matmul's [N/64][K/32][8][32] fragments
//!   quantizeFp4  cuda/nvfp4/experts.py _quantize: bf16 experts -> NVFP4 by ModelOpt's recipe (the MTP experts)
//!   dequantFp8   weights.weight_bf16 with Fp8BlockLinear.column_scales: FP8_PB_WO rows -> bf16
//!   ngramLut     host_table.FP8Table: e4m3 code -> bf16_rne(e4m3 x scale), NaN codes 0x7FC0
//!   invFreq      weights.load: theta ** (-i / half) in fp64, stored fp32
//! The Python steps that run as CUDA torch ops keep CUDA torch's arithmetic: a tensor divided by a Python float is a
//! multiply by the fp32 reciprocal (ATen div_true_kernel_cuda's CPU-scalar path), a tensor by a tensor a true division,
//! fp32 -> bf16 and fp32 -> e4m3 round to nearest even, argmin keeps the first minimum, and nothing contracts to FMA.
//! Python source: Ash Hart (ashhart) for every function above; weight_bf16 by Jürgen Schmied; FP8Table by tournierjc
//! and ashhart (TensorFold). fixtures_cuda_weights.py records their outputs; the tests below compare.
const std = @import("std");

// ---- scalar formats ---------------------------------------------------------------------------------------------

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

// ---- routed experts: cuda/nvfp4/experts.py ----------------------------------------------------------------------

pub const cols_a_block = 32;
pub const words_a_block = 144;

/// A matrix of NVFP4 codes and e4m3 scales as rows of a larger stored array: n rows from `row0`, `k` inputs from input
/// `k0` (codes: byte k0 / 2 on, scales: byte k0 / 16 on; k0 a multiple of 16), each array's row pitch in bytes.
pub const Fp4Rows = struct {
    words: []const u8,
    word_pitch: usize,
    scales: []const u8,
    scale_pitch: usize,
    row0: usize = 0,
    k0: usize = 0,
};

/// One expert matrix [n, k] -> `out[cb][g][m_count * 144]`, its 144 words at `m * 144` of each (cb, g) cell
/// (Experts4.up holds gate at m 0 and up at m 1, down a single matrix).
///
/// `_pack` puts input 16h + 4t + q of a 32-input group at nibble slot 2h + q / 2 + 4 (q % 2) of word
/// (lane gq * 4 + t, tile j): with the codes' bytes b0, b1 (h 0) and b2, b3 (h 1), low nibble first, the word is their
/// four low nibbles in its low half and their four high nibbles in its high half.
pub fn packExpert(src: Fp4Rows, n: usize, k: usize, out: []u32, m: usize, m_count: usize) void {
    const nb = n / cols_a_block;
    const kg = k / 32;
    std.debug.assert(out.len == nb * kg * m_count * words_a_block and src.k0 % 16 == 0);
    for (0..nb) |cb| for (0..kg) |g| {
        const cell = out[((cb * kg + g) * m_count + m) * words_a_block ..][0..words_a_block];
        for (0..8) |gq| for (0..4) |j| {
            const r = src.row0 + cb * 32 + j * 8 + gq;
            const row = src.words[r * src.word_pitch + src.k0 / 2 + g * 16 ..][0..16];
            for (0..4) |t| {
                const b0: u32 = row[2 * t];
                const b1: u32 = row[2 * t + 1];
                const b2: u32 = row[8 + 2 * t];
                const b3: u32 = row[8 + 2 * t + 1];
                const lo = (b0 & 0xF) | (b1 & 0xF) << 4 | (b2 & 0xF) << 8 | (b3 & 0xF) << 12;
                const hi = (b0 >> 4) | (b1 >> 4) << 4 | (b2 >> 4) << 8 | (b3 >> 4) << 12;
                cell[(gq * 4 + t) * 4 + j] = lo | hi << 16;
            }
        };
        // scale bytes [t][h][j][c]: column cb*32 + j*8 + t*2 + c, 16-input block g*2 + h; 64 bytes as 16 words
        const bytes = std.mem.sliceAsBytes(cell[128..144]);
        for (0..4) |t| for (0..4) |j| for (0..2) |c| {
            const r = src.row0 + cb * 32 + j * 8 + t * 2 + c;
            const sc = src.scales[r * src.scale_pitch + src.k0 / 16 + g * 2 ..][0..2];
            bytes[t * 16 + j * 2 + c] = sc[0];
            bytes[t * 16 + 8 + j * 2 + c] = sc[1];
        };
    };
}

// ---- the shared expert's FP4 tables: qwen4_exp/cuda/nvfp4.py fp4_from_bf16 ---------------------------------------

/// A bf16 grid [n, k] (as rows of `pitch` values from column `k0`) -> `out[n/64][k/64][64 k][64 n]`.
pub fn tileBits(rows: []const u16, pitch: usize, k0: usize, n: usize, k: usize, out: []u16) void {
    std.debug.assert(n % 64 == 0 and k % 64 == 0 and out.len == n * k);
    const kt = k / 64;
    for (0..n / 64) |tn| for (0..kt) |tk| for (0..64) |kk| {
        const dst = out[((tn * kt + tk) * 64 + kk) * 64 ..][0..64];
        for (dst, 0..) |*d, nn| d.* = rows[(tn * 64 + nn) * pitch + k0 + tk * 64 + kk];
    };
}

// ---- MLX 4-bit: qwen4_exp/cuda/bf16.py quantize4 and cuda/kernels/qmm.py pack ------------------------------------

/// bf16 [n, k] (rows of `pitch` from `rows`) -> MLX words [n, k/8], bf16 scales and biases [n, k/32].
pub fn quantize4(rows: []const u16, n: usize, k: usize, words: []u32, scales: []u16, biases: []u16) void {
    const kg = k / 32;
    const inv15: f32 = @as(f32, 1.0) / @as(f32, 15.0); // `/ 15.0`: a multiply by the fp32 reciprocal on CUDA
    const floor: f32 = @floatCast(@as(f64, 1e-8));
    for (0..n) |r| for (0..kg) |g| {
        const v = rows[r * k + g * 32 ..][0..32];
        var mn: f32 = std.math.inf(f32);
        var mx: f32 = -std.math.inf(f32);
        for (v) |b| {
            const x = bf16ToF32(b);
            mn = @min(mn, x);
            mx = @max(mx, x);
        }
        const s = @max((mx - mn) * inv15, floor);
        var q: [32]u32 = undefined;
        for (v, &q) |b, *o| o.* = @intFromFloat(std.math.clamp(roundEven((bf16ToF32(b) - mn) / s), 0, 15));
        for (0..4) |w| {
            var word: u32 = 0;
            for (0..8) |p| word |= q[w * 8 + p] << @intCast(4 * p);
            words[r * (k / 8) + g * 4 + w] = word;
        }
        scales[r * kg + g] = bf16Rne(s);
        biases[r * kg + g] = bf16Rne(mn);
    };
}

/// torch.round: halves to even (@round takes them away from zero).
pub fn roundEven(x: f32) f32 {
    if (@abs(x - @trunc(x)) == 0.5) return 2 * @round(x / 2);
    return @round(x);
}

pub const q4_offsets = [8]usize{ 0, 8, 16, 24, 1, 9, 17, 25 };

/// MLX words [n, k/8] -> the lane matmul's int32 [npad/64][k/32][8][32] (npad = n rounded up to 128, zero rows).
pub fn packQ4(words: []const u32, n: usize, k: usize, out: []u32) void {
    const kg = k / 32;
    const npad = (n + 127) / 128 * 128;
    std.debug.assert(out.len == npad / 64 * kg * 8 * 32);
    for (0..npad / 64) |tt| for (0..kg) |g| for (0..8) |j| for (0..8) |r| for (0..4) |c| {
        const col = tt * 64 + j * 8 + r;
        var w: u32 = 0;
        if (col < n) for (q4_offsets, 0..) |off, p| {
            const i = g * 32 + 2 * c + off;
            const nib = (words[col * (k / 8) + i / 8] >> @intCast(4 * (i % 8))) & 0xF;
            w |= nib << @intCast(4 * p);
        };
        out[((tt * kg + g) * 8 + j) * 32 + r * 4 + c] = w;
    };
}

/// scales or biases [n, kg] -> [kg, npad] with zero columns past n.
pub fn majorQ4(src: []const u16, n: usize, kg: usize, out: []u16) void {
    const npad = (n + 127) / 128 * 128;
    std.debug.assert(out.len == kg * npad);
    @memset(out, 0);
    for (0..n) |r| for (0..kg) |g| {
        out[g * npad + r] = src[r * kg + g];
    };
}

// ---- NVFP4 by ModelOpt's recipe: cuda/nvfp4/experts.py _quantize ------------------------------------------------

const e2m1 = [8]f32{ 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 };

/// The per-expert scale g: amax(|x|) * fp32(1 / 2688), at least 1e-30.
pub fn fp4Global(rows: []const u16) f32 {
    var amax: f32 = 0;
    for (rows) |b| amax = @max(amax, @abs(bf16ToF32(b)));
    const inv: f32 = @as(f32, 1.0) / @as(f32, 6.0 * 448.0);
    return @max(amax * inv, @as(f32, @floatCast(@as(f64, 1e-30))));
}

/// One expert's bf16 [n, k] (rows of `pitch` from column `k0`) -> codes [n, k/2] (low nibble the even input) and e4m3
/// scales [n, k/16] with the expert's g (fp4Global over the same n x k values).
pub fn quantizeFp4(rows: []const u16, pitch: usize, k0: usize, n: usize, k: usize, g: f32, words: []u8, scales: []u8) void {
    const inv6: f32 = @as(f32, 1.0) / @as(f32, 6.0);
    const tiny: f32 = @floatCast(@as(f64, 1e-30));
    for (0..n) |r| for (0..k / 16) |b| {
        const v = rows[r * pitch + k0 + b * 16 ..][0..16];
        var amax: f32 = 0;
        for (v) |x| amax = @max(amax, @abs(bf16ToF32(x)));
        const s8 = f32ToE4m3(@min((amax * inv6) / g, 448.0));
        scales[r * (k / 16) + b] = s8;
        const step = e4m3ToF32(s8) * g;
        for (0..8) |p| {
            var pair: [2]u8 = undefined;
            for (0..2) |h| {
                const x = bf16ToF32(v[2 * p + h]);
                const y: f32 = if (step > 0) x / @max(step, tiny) else 0;
                const a = @abs(y);
                var best: usize = 0;
                var dist = @abs(a - e2m1[0]);
                for (1..8) |i| {
                    const d = @abs(a - e2m1[i]);
                    if (d < dist) {
                        dist = d;
                        best = i;
                    }
                }
                pair[h] = @intCast(best + @as(usize, if (y < 0) 8 else 0));
            }
            words[r * (k / 2) + b * 8 + p] = pair[0] | (pair[1] << 4);
        }
    };
}

// ---- FP8_PB_WO: weights.weight_bf16 ------------------------------------------------------------------------------

/// e4m3 codes [n, k] and bf16 weight_scale_inv [ceil(n/128), k/128] -> bf16 [n, k]: code * fp32(block scale).
pub fn dequantFp8(codes: []const u8, scale_inv: []const u16, n: usize, k: usize, out: []u16) void {
    const kb = k / 128;
    for (0..n) |r| for (0..k) |c| {
        const s = bf16ToF32(scale_inv[(r / 128) * kb + c / 128]);
        out[r * k + c] = bf16Rne(e4m3ToF32(codes[r * k + c]) * s);
    };
}

// ---- the FP8 n-gram table's LUT and inv_freq ---------------------------------------------------------------------

pub fn ngramLut(scale: f32) [256]u16 {
    var t: [256]u16 = undefined;
    for (&t, 0..) |*o, c| {
        o.* = if (c & 0x7F == 0x7F) 0x7FC0 else bf16Rne(e4m3ToF32(@intCast(c)) * scale);
    }
    return t;
}

/// theta ** (-i / half) for i in [0, half), fp64, rounded to fp32 (half = rotary_dim / 2).
pub fn invFreq(theta: f64, half: usize, out: []f32) void {
    for (out[0..half], 0..) |*o, i| {
        const e = -@as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(half));
        o.* = @floatCast(std.math.pow(f64, theta, e));
    }
}

// ---- tests against the Python engine's outputs (fixtures_cuda_weights.json) --------------------------------------

/// splitmix64 of seed * 2**32 + i (fixtures_cuda_weights.py `rnd`).
pub fn rnd(seed: u64, i: u64) u64 {
    var x = i +% (seed << 32) +% 0x9E3779B97F4A7C15;
    x = (x ^ (x >> 30)) *% 0xBF58476D1CE4E5B9;
    x = (x ^ (x >> 27)) *% 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

fn bf16Gen(seed: u64, i: u64) u16 {
    const r = rnd(seed, i);
    return @intCast(((r >> 63) << 15) | ((112 + (r >> 40) % 16) << 7) | ((r >> 8) & 0x7F));
}

fn u8Gen(seed: u64, i: u64) u8 {
    return @truncate(rnd(seed, i));
}

const T = std.testing;

fn fixture(a: std.mem.Allocator) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, a, @embedFile("fixtures_cuda_weights.json"), .{});
}

fn expectSha(want: std.json.Value, key: []const u8, bytes: []const u8) !void {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    const hex = std.fmt.bytesToHex(d, .lower);
    const w = want.object.get(key) orelse return error.FixtureMissing;
    if (!std.mem.eql(u8, w.string, &hex)) {
        std.debug.print("{s}: got {s}, Python {s}\n", .{ key, hex, w.string });
        return error.DigestDiffers;
    }
}

test "e4m3 and bf16 conversions" {
    for (0..256) |c| {
        const b: u8 = @intCast(c);
        const v = e4m3ToF32(b);
        if (c & 0x7F == 0x7F) {
            try T.expect(std.math.isNan(v));
            continue;
        }
        try T.expectEqual(b, f32ToE4m3(v)); // every value round-trips (0x80, -0, included)
    }
    try T.expectEqual(@as(u8, 0x7E), f32ToE4m3(448));
    try T.expectEqual(@as(u8, 0x7E), f32ToE4m3(463.9));
    try T.expectEqual(@as(u8, 0x7E), f32ToE4m3(464)); // halfway to the NaN code: to even, 448
    try T.expectEqual(@as(u8, 0x7F), f32ToE4m3(480)); // c10's first NaN (1087 << 20)
    try T.expectEqual(@as(u8, 0x00), f32ToE4m3(1.0 / 1024.0)); // half the smallest subnormal: to even (0)
    try T.expectEqual(@as(u8, 0x02), f32ToE4m3(3.0 / 1024.0)); // 1.5 subnormal steps: to even (2)
    try T.expectEqual(@as(u16, 0x3F80), bf16Rne(1.0));
    try T.expectEqual(@as(u16, 0x3F80), bf16Rne(bitsF32(0x3F808000))); // a tie: to even
    try T.expectEqual(@as(u16, 0x3F82), bf16Rne(bitsF32(0x3F818000)));
}

test "routed experts pack as nvx.make, one GPU and rank 1 of 2" {
    var p = try fixture(T.allocator);
    defer p.deinit();
    const a = T.allocator;
    const E = 3;
    const NI = 64;
    const D = 96;
    const Mat = struct { w: []u8, s: []u8, s2: [E]f32 };
    var mats: [3]Mat = undefined;
    for (&mats, [_]u64{ 1, 4, 7 }, [_][2]usize{ .{ NI, D }, .{ NI, D }, .{ D, NI } }) |*m, seed, nk| {
        m.w = try a.alloc(u8, E * nk[0] * nk[1] / 2);
        m.s = try a.alloc(u8, E * nk[0] * nk[1] / 16);
        for (m.w, 0..) |*x, i| x.* = u8Gen(seed, i);
        for (m.s, 0..) |*x, i| x.* = u8Gen(seed + 1, i);
        for (&m.s2, 0..) |*x, i| x.* = @as(f32, @floatFromInt(rnd(seed + 2, i) % 1000 + 1)) / 1024.0;
    }
    defer for (mats) |m| {
        a.free(m.w);
        a.free(m.s);
    };
    for ([_][]const u8{ "experts.w1", "experts.w2r1" }, [_]usize{ 1, 2 }) |tag, world| {
        const ni = NI / world;
        const lo = if (world == 2) ni else 0;
        const up = try a.alloc(u32, E * (ni / 32) * (D / 32) * 2 * 144);
        defer a.free(up);
        const down = try a.alloc(u32, E * (D / 32) * (ni / 32) * 144);
        defer a.free(down);
        for (0..E) |e| {
            for (0..2) |m| {
                const src: Fp4Rows = .{ .words = mats[m].w[e * NI * D / 2 ..][0 .. NI * D / 2], .word_pitch = D / 2, .scales = mats[m].s[e * NI * D / 16 ..][0 .. NI * D / 16], .scale_pitch = D / 16, .row0 = lo };
                packExpert(src, ni, D, up[e * (ni / 32) * (D / 32) * 2 * 144 ..][0 .. (ni / 32) * (D / 32) * 2 * 144], m, 2);
            }
            const src: Fp4Rows = .{ .words = mats[2].w[e * D * NI / 2 ..][0 .. D * NI / 2], .word_pitch = NI / 2, .scales = mats[2].s[e * D * NI / 16 ..][0 .. D * NI / 16], .scale_pitch = NI / 16, .k0 = lo };
            packExpert(src, D, ni, down[e * (D / 32) * (ni / 32) * 144 ..][0 .. (D / 32) * (ni / 32) * 144], 0, 1);
        }
        var us: [E * 2]f32 = undefined;
        for (0..E) |e| {
            us[2 * e] = mats[0].s2[e];
            us[2 * e + 1] = mats[1].s2[e];
        }
        const want = p.value.object.get(tag).?;
        try expectSha(want, "up", std.mem.sliceAsBytes(up));
        try expectSha(want, "down", std.mem.sliceAsBytes(down));
        try expectSha(want, "up_scale", std.mem.sliceAsBytes(&us));
        try expectSha(want, "down_scale", std.mem.sliceAsBytes(&mats[2].s2));
    }
}

test "the shared expert's identity-scaled FP4 tables, one GPU and rank 1 of 2" {
    var p = try fixture(T.allocator);
    defer p.deinit();
    const a = T.allocator;
    const NI = 128;
    const D = 128;
    var sg: [NI * D]u16 = undefined;
    var su: [NI * D]u16 = undefined;
    var sd: [D * NI]u16 = undefined;
    for (&sg, &su, &sd, 0..) |*g, *u, *d, i| {
        g.* = bf16Gen(10, i);
        u.* = bf16Gen(11, i);
        d.* = bf16Gen(12, i);
    }
    for ([_][]const u8{ "shared.w1", "shared.w2r1" }, [_]usize{ 1, 2 }) |tag, world| {
        const ni = NI / world;
        const lo = if (world == 2) ni else 0;
        const cat = try a.alloc(u16, 2 * ni * D);
        defer a.free(cat);
        @memcpy(cat[0 .. ni * D], sg[lo * D ..][0 .. ni * D]);
        @memcpy(cat[ni * D ..], su[lo * D ..][0 .. ni * D]);
        const gu = try a.alloc(u16, 2 * ni * D);
        defer a.free(gu);
        tileBits(cat, D, 0, 2 * ni, D, gu);
        const dn = try a.alloc(u16, D * ni);
        defer a.free(dn);
        tileBits(&sd, NI, lo, D, ni, dn);
        const ones = try a.alloc(f32, 2 * ni * D / 16 + D * ni / 16);
        defer a.free(ones);
        @memset(ones, 1.0);
        const want = p.value.object.get(tag).?;
        try expectSha(want, "gu.weight", std.mem.sliceAsBytes(gu));
        try expectSha(want, "down.weight", std.mem.sliceAsBytes(dn));
        try expectSha(want, "gu.scale", std.mem.sliceAsBytes(ones[0 .. 2 * ni * D / 16]));
        try expectSha(want, "gu.scale2", std.mem.sliceAsBytes(ones[0 .. 2 * ni]));
        try expectSha(want, "down.scale", std.mem.sliceAsBytes(ones[0 .. D * ni / 16]));
        try expectSha(want, "down.scale2", std.mem.sliceAsBytes(ones[0..D]));
    }
}

test "the draft head's quantize4 and fragment pack equal Python's on CUDA" {
    var p = try fixture(T.allocator);
    defer p.deinit();
    const N = 200;
    const K = 128;
    var w: [N * K]u16 = undefined;
    for (&w, 0..) |*x, i| x.* = bf16Gen(13, i);
    @memset(w[7 * K ..][0..K], w[7 * K]);
    for (0..K / 2) |i| w[8 * K + 2 * i] = w[8 * K];
    var words: [N * K / 8]u32 = undefined;
    var sc: [N * K / 32]u16 = undefined;
    var bi: [N * K / 32]u16 = undefined;
    quantize4(&w, N, K, &words, &sc, &bi);
    var packed_: [256 / 64 * (K / 32) * 8 * 32]u32 = undefined;
    packQ4(&words, N, K, &packed_);
    var ms: [(K / 32) * 256]u16 = undefined;
    var mb: [(K / 32) * 256]u16 = undefined;
    majorQ4(&sc, N, K / 32, &ms);
    majorQ4(&bi, N, K / 32, &mb);
    const want = p.value.object.get("quantize4").?;
    try expectSha(want, "weight", std.mem.sliceAsBytes(&packed_));
    try expectSha(want, "scales", std.mem.sliceAsBytes(&ms));
    try expectSha(want, "biases", std.mem.sliceAsBytes(&mb));
}

test "nvx.quantize (ModelOpt's recipe) equals Python's on CUDA" {
    var p = try fixture(T.allocator);
    defer p.deinit();
    const E = 3;
    const N = 64;
    const K = 64;
    var x: [E * N * K]u16 = undefined;
    for (&x, 0..) |*v, i| v.* = bf16Gen(14, i);
    @memset(x[2 * N * K ..], 0);
    @memset(x[N * K ..][0..16], 0);
    var words: [E * N * K / 2]u8 = undefined;
    var scales: [E * N * K / 16]u8 = undefined;
    var g: [E]f32 = undefined;
    for (0..E) |e| {
        const rows = x[e * N * K ..][0 .. N * K];
        g[e] = fp4Global(rows);
        quantizeFp4(rows, K, 0, N, K, g[e], words[e * N * K / 2 ..][0 .. N * K / 2], scales[e * N * K / 16 ..][0 .. N * K / 16]);
    }
    const want = p.value.object.get("nvx_quantize").?;
    try expectSha(want, "words", &words);
    try expectSha(want, "scales", &scales);
    try expectSha(want, "g", std.mem.sliceAsBytes(&g));
}

test "FP8_PB_WO dequantization, and the MTP experts' chain to packed NVFP4" {
    var p = try fixture(T.allocator);
    defer p.deinit();
    const a = T.allocator;
    const N = 256;
    const K = 256;
    const codes = try a.alloc(u8, N * K);
    defer a.free(codes);
    for (codes, 0..) |*c, i| {
        c.* = u8Gen(15, i);
        if (c.* & 0x7F == 0x7F) c.* -= 1;
    }
    var inv: [4]u16 = undefined;
    for (&inv, 0..) |*v, i| v.* = bf16Gen(16, i) & 0x7FFF;
    const deq = try a.alloc(u16, N * K);
    defer a.free(deq);
    dequantFp8(codes, &inv, N, K, deq);
    try expectSha(p.value.object.get("fp8_block").?, "bf16", std.mem.sliceAsBytes(deq));

    // both = stack([deq[:64], deq[64:128]]): gate and up are the same two experts, down their transposes
    const E = 2;
    const n = 64;
    const up = try a.alloc(u32, E * (n / 32) * (K / 32) * 2 * 144);
    defer a.free(up);
    const down = try a.alloc(u32, E * (K / 32) * (n / 32) * 144);
    defer a.free(down);
    var us: [E * 2]f32 = undefined;
    var ds: [E]f32 = undefined;
    const tr = try a.alloc(u16, K * n);
    defer a.free(tr);
    var w: [n * K / 2]u8 = undefined;
    var s: [n * K / 16]u8 = undefined;
    for (0..E) |e| {
        const rows = deq[e * n * K ..][0 .. n * K];
        const g = fp4Global(rows);
        quantizeFp4(rows, K, 0, n, K, g, &w, &s);
        for (0..2) |m| packExpert(.{ .words = &w, .word_pitch = K / 2, .scales = &s, .scale_pitch = K / 16 }, n, K, up[e * (n / 32) * (K / 32) * 2 * 144 ..][0 .. (n / 32) * (K / 32) * 2 * 144], m, 2);
        us[2 * e] = g;
        us[2 * e + 1] = g;
        for (0..K) |c| for (0..n) |r| {
            tr[c * n + r] = rows[r * K + c];
        };
        const gd = fp4Global(tr);
        var wd: [K * n / 2]u8 = undefined;
        var sd: [K * n / 16]u8 = undefined;
        quantizeFp4(tr, n, 0, K, n, gd, &wd, &sd);
        packExpert(.{ .words = &wd, .word_pitch = n / 2, .scales = &sd, .scale_pitch = n / 16 }, K, n, down[e * (K / 32) * (n / 32) * 144 ..][0 .. (K / 32) * (n / 32) * 144], 0, 1);
        ds[e] = gd;
    }
    const want = p.value.object.get("fp8_chain").?;
    try expectSha(want, "up", std.mem.sliceAsBytes(up));
    try expectSha(want, "down", std.mem.sliceAsBytes(down));
    try expectSha(want, "up_scale", std.mem.sliceAsBytes(&us));
    try expectSha(want, "down_scale", std.mem.sliceAsBytes(&ds));
}

test "the FP8 n-gram LUT and inv_freq equal Python's" {
    var p = try fixture(T.allocator);
    defer p.deinit();
    var it = p.value.object.get("lut").?.object.iterator();
    var n: usize = 0;
    while (it.next()) |kv| {
        const bits = try std.fmt.parseInt(u16, kv.key_ptr.*, 16);
        const lut = ngramLut(bf16ToF32(bits));
        const hex = std.fmt.bytesToHex(std.mem.toBytes(lut), .lower);
        try T.expectEqualStrings(kv.value_ptr.string, &hex);
        n += 1;
    }
    try T.expectEqual(@as(usize, 2), n); // the checkpoint's own scale (0x3951) among them
    var inv: [32]f32 = undefined;
    invFreq(1e7, 32, &inv);
    for (p.value.object.get("inv_freq").?.array.items, inv) |want, got| try T.expectEqual(@as(u32, @intCast(want.integer)), f32Bits(got));
}
