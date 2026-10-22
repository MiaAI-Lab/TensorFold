//! WebP lossy (VP8 key frame) decoder: bool decoder, header, intra prediction, token partitions, inverse DCT/WHT,
//! loop filters. Arithmetic follows RFC 6386 / libwebp 1.6.0 (which Pillow 12.3 bundles), integer for integer.
const std = @import("std");
const tables = @import("vp8_tables.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{Invalid} || Allocator.Error;

pub const Planes = struct {
    width: usize,
    height: usize,
    y: []u8,
    u: []u8,
    v: []u8,
    y_stride: usize,
    uv_stride: usize,

    pub fn deinit(p: *Planes, a: Allocator) void {
        a.free(p.y);
        a.free(p.u);
        a.free(p.v);
    }
};

// ---- boolean entropy decoder (libwebp's VP8BitReader, 56-bit loads) -------------------------------------------------
const BitReader = struct {
    buf: []const u8,
    pos: usize = 0,
    value: u64 = 0,
    range: u32 = 254, // range - 1
    bits: i32 = -8,
    eof: bool = false,

    fn init(buf: []const u8) BitReader {
        var r: BitReader = .{ .buf = buf };
        r.loadNewBytes();
        return r;
    }

    fn loadNewBytes(r: *BitReader) void {
        if (r.pos + 8 <= r.buf.len) {
            const in_bits = std.mem.readInt(u64, r.buf[r.pos..][0..8], .big);
            r.pos += 7;
            r.value = (in_bits >> 8) | (r.value << 56);
            r.bits += 56;
        } else if (r.pos < r.buf.len) {
            r.bits += 8;
            r.value = @as(u64, r.buf[r.pos]) | (r.value << 8);
            r.pos += 1;
        } else if (!r.eof) {
            r.value <<= 8;
            r.bits += 8;
            r.eof = true;
        } else {
            r.bits = 0;
        }
    }

    inline fn getBit(r: *BitReader, prob: u32) u32 {
        var range = r.range;
        if (r.bits < 0) r.loadNewBytes();
        const pos: u6 = @intCast(r.bits);
        const split = (range * prob) >> 8;
        const value: u32 = @truncate(r.value >> pos);
        var bit: u32 = undefined;
        if (value > split) {
            range -= split;
            r.value -= @as(u64, split + 1) << pos;
            bit = 1;
        } else {
            range = split + 1;
            bit = 0;
        }
        const shift: u5 = @intCast(7 ^ (31 - @clz(range)));
        range <<= shift;
        r.bits -= shift;
        r.range = range - 1;
        return bit;
    }

    fn getValue(r: *BitReader, nbits: u5) u32 {
        var v: u32 = 0;
        var n = nbits;
        while (n > 0) {
            n -= 1;
            v |= r.getBit(0x80) << n;
        }
        return v;
    }

    fn getSignedValue(r: *BitReader, nbits: u5) i32 {
        const v: i32 = @intCast(r.getValue(nbits));
        return if (r.getBit(0x80) != 0) -v else v;
    }
};

const zigzag = [16]u8{ 0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15 };
const bands = [17]u8{ 0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 0 };
const cat3 = [_]u8{ 173, 148, 140 };
const cat4 = [_]u8{ 176, 155, 140, 135 };
const cat5 = [_]u8{ 180, 157, 141, 134, 130 };
const cat6 = [_]u8{ 254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129 };
const cat_tabs = [4][]const u8{ &cat3, &cat4, &cat5, &cat6 };

const Segment = struct {
    use: bool = false,
    update_map: bool = false,
    absolute: bool = false,
    quantizer: [4]i32 = .{ 0, 0, 0, 0 },
    filter_strength: [4]i32 = .{ 0, 0, 0, 0 },
    probs: [3]u32 = .{ 255, 255, 255 },
};

const Quant = struct { y1: [2]i32, y2: [2]i32, uv: [2]i32 };

const FInfo = struct { limit: u8 = 0, ilevel: u8 = 0, inner: bool = false, hev: u8 = 0 };

const B_DC = 0;
const B_TM = 1;
const B_VE = 2;
const B_HE = 3;
const B_RD = 4;
const B_VR = 5;
const B_LD = 6;
const B_VL = 7;
const B_HD = 8;
const B_HU = 9;

const kYModesIntra4 = [18]i8{
    -B_DC, 1,
    -B_TM, 2,
    -B_VE, 3,
    4,     6,
    -B_HE, 5,
    -B_RD, -B_VR,
    -B_LD, 7,
    -B_VL, 8,
    -B_HD, -B_HU,
};

inline fn clip8(v: i32) u8 {
    return @intCast(std.math.clamp(v, 0, 255));
}

// 32-bit wrapping arithmetic throughout the transforms: corrupt coefficients overflow in libwebp's C `int` too,
// and must neither trap here nor depend on the build mode.
inline fn mul1(a: i32) i32 {
    return (((a *% 20091) >> 16) +% a);
}
inline fn mul2(a: i32) i32 {
    return (a *% 35468) >> 16;
}

/// Inverse 4x4 DCT added to the prediction in `dst` (stride `bps`).
fn transformOne(in: *const [16]i16, dst: []u8, bps: usize) void {
    var tmp: [16]i32 = undefined;
    for (0..4) |i| {
        const a: i32 = @as(i32, in[i]) + in[8 + i];
        const b: i32 = @as(i32, in[i]) - in[8 + i];
        const c = mul2(in[4 + i]) -% mul1(in[12 + i]);
        const d = mul1(in[4 + i]) +% mul2(in[12 + i]);
        tmp[4 * i + 0] = a +% d;
        tmp[4 * i + 1] = b +% c;
        tmp[4 * i + 2] = b -% c;
        tmp[4 * i + 3] = a -% d;
    }
    for (0..4) |i| {
        const dc = tmp[i] +% 4;
        const a = dc +% tmp[8 + i];
        const b = dc -% tmp[8 + i];
        const c = mul2(tmp[4 + i]) -% mul1(tmp[12 + i]);
        const d = mul1(tmp[4 + i]) +% mul2(tmp[12 + i]);
        const row = dst[i * bps ..][0..4];
        row[0] = clip8(@as(i32, row[0]) +% ((a +% d) >> 3));
        row[1] = clip8(@as(i32, row[1]) +% ((b +% c) >> 3));
        row[2] = clip8(@as(i32, row[2]) +% ((b -% c) >> 3));
        row[3] = clip8(@as(i32, row[3]) +% ((a -% d) >> 3));
    }
}

fn transformWHT(in: *const [16]i16, out: []i16) void {
    var tmp: [16]i32 = undefined;
    for (0..4) |i| {
        const a0: i32 = @as(i32, in[0 + i]) + in[12 + i];
        const a1: i32 = @as(i32, in[4 + i]) + in[8 + i];
        const a2: i32 = @as(i32, in[4 + i]) - in[8 + i];
        const a3: i32 = @as(i32, in[0 + i]) - in[12 + i];
        tmp[0 + i] = a0 + a1;
        tmp[8 + i] = a0 - a1;
        tmp[4 + i] = a3 + a2;
        tmp[12 + i] = a3 - a2;
    }
    for (0..4) |i| {
        const dc = tmp[0 + i * 4] + 3;
        const a0 = dc + tmp[3 + i * 4];
        const a1 = tmp[1 + i * 4] + tmp[2 + i * 4];
        const a2 = tmp[1 + i * 4] - tmp[2 + i * 4];
        const a3 = dc - tmp[3 + i * 4];
        out[i * 64 + 0] = @truncate((a0 + a1) >> 3);
        out[i * 64 + 16] = @truncate((a3 + a2) >> 3);
        out[i * 64 + 32] = @truncate((a0 - a1) >> 3);
        out[i * 64 + 48] = @truncate((a3 - a2) >> 3);
    }
}

const BPS = 32;

inline fn avg3(a: u8, b: u8, c: u8) u8 {
    return @intCast((@as(u32, a) + 2 * @as(u32, b) + c + 2) >> 2);
}
inline fn avg2(a: u8, b: u8) u8 {
    return @intCast((@as(u32, a) + b + 1) >> 1);
}

/// 4x4 intra prediction into `d` (stride BPS) from the border pixels around it.
fn predict4(mode: u8, d: []u8, o: usize) void {
    const top = o - BPS;
    switch (mode) {
        B_DC => {
            var dc: u32 = 4;
            for (0..4) |i| dc += @as(u32, d[top + i]) + d[o - 1 + i * BPS];
            const v: u8 = @intCast(dc >> 3);
            for (0..4) |y| @memset(d[o + y * BPS ..][0..4], v);
        },
        B_TM => {
            const tl: i32 = d[top - 1];
            for (0..4) |y| {
                const l: i32 = d[o - 1 + y * BPS];
                for (0..4) |x| d[o + y * BPS + x] = clip8(l + d[top + x] - tl);
            }
        },
        B_VE => {
            const vals = [4]u8{ avg3(d[top - 1], d[top], d[top + 1]), avg3(d[top], d[top + 1], d[top + 2]), avg3(d[top + 1], d[top + 2], d[top + 3]), avg3(d[top + 2], d[top + 3], d[top + 4]) };
            for (0..4) |y| @memcpy(d[o + y * BPS ..][0..4], &vals);
        },
        B_HE => {
            const A = d[top - 1];
            const B = d[o - 1];
            const C = d[o - 1 + BPS];
            const D = d[o - 1 + 2 * BPS];
            const E = d[o - 1 + 3 * BPS];
            @memset(d[o..][0..4], avg3(A, B, C));
            @memset(d[o + BPS ..][0..4], avg3(B, C, D));
            @memset(d[o + 2 * BPS ..][0..4], avg3(C, D, E));
            @memset(d[o + 3 * BPS ..][0..4], avg3(D, E, E));
        },
        B_RD => {
            const I = d[o - 1];
            const J = d[o - 1 + BPS];
            const K = d[o - 1 + 2 * BPS];
            const L = d[o - 1 + 3 * BPS];
            const X = d[top - 1];
            const A = d[top];
            const B = d[top + 1];
            const C = d[top + 2];
            const D = d[top + 3];
            put(d, o, 0, 3, avg3(J, K, L));
            const v1 = avg3(I, J, K);
            put(d, o, 1, 3, v1);
            put(d, o, 0, 2, v1);
            const v2 = avg3(X, I, J);
            put(d, o, 2, 3, v2);
            put(d, o, 1, 2, v2);
            put(d, o, 0, 1, v2);
            const v3 = avg3(A, X, I);
            put(d, o, 3, 3, v3);
            put(d, o, 2, 2, v3);
            put(d, o, 1, 1, v3);
            put(d, o, 0, 0, v3);
            const v4 = avg3(B, A, X);
            put(d, o, 3, 2, v4);
            put(d, o, 2, 1, v4);
            put(d, o, 1, 0, v4);
            const v5 = avg3(C, B, A);
            put(d, o, 3, 1, v5);
            put(d, o, 2, 0, v5);
            put(d, o, 3, 0, avg3(D, C, B));
        },
        B_VR => {
            const I = d[o - 1];
            const J = d[o - 1 + BPS];
            const K = d[o - 1 + 2 * BPS];
            const X = d[top - 1];
            const A = d[top];
            const B = d[top + 1];
            const C = d[top + 2];
            const D = d[top + 3];
            const a1 = avg2(X, A);
            put(d, o, 0, 0, a1);
            put(d, o, 1, 2, a1);
            const a2 = avg2(A, B);
            put(d, o, 1, 0, a2);
            put(d, o, 2, 2, a2);
            const a3 = avg2(B, C);
            put(d, o, 2, 0, a3);
            put(d, o, 3, 2, a3);
            put(d, o, 3, 0, avg2(C, D));
            put(d, o, 0, 3, avg3(K, J, I));
            put(d, o, 0, 2, avg3(J, I, X));
            const b1 = avg3(I, X, A);
            put(d, o, 0, 1, b1);
            put(d, o, 1, 3, b1);
            const b2 = avg3(X, A, B);
            put(d, o, 1, 1, b2);
            put(d, o, 2, 3, b2);
            const b3 = avg3(A, B, C);
            put(d, o, 2, 1, b3);
            put(d, o, 3, 3, b3);
            put(d, o, 3, 1, avg3(B, C, D));
        },
        B_LD => {
            const A = d[top];
            const B = d[top + 1];
            const C = d[top + 2];
            const D = d[top + 3];
            const E = d[top + 4];
            const F = d[top + 5];
            const G = d[top + 6];
            const H = d[top + 7];
            put(d, o, 0, 0, avg3(A, B, C));
            const c1 = avg3(B, C, D);
            put(d, o, 1, 0, c1);
            put(d, o, 0, 1, c1);
            const c2 = avg3(C, D, E);
            put(d, o, 2, 0, c2);
            put(d, o, 1, 1, c2);
            put(d, o, 0, 2, c2);
            const c3 = avg3(D, E, F);
            put(d, o, 3, 0, c3);
            put(d, o, 2, 1, c3);
            put(d, o, 1, 2, c3);
            put(d, o, 0, 3, c3);
            const c4 = avg3(E, F, G);
            put(d, o, 3, 1, c4);
            put(d, o, 2, 2, c4);
            put(d, o, 1, 3, c4);
            const c5 = avg3(F, G, H);
            put(d, o, 3, 2, c5);
            put(d, o, 2, 3, c5);
            put(d, o, 3, 3, avg3(G, H, H));
        },
        B_VL => {
            const A = d[top];
            const B = d[top + 1];
            const C = d[top + 2];
            const D = d[top + 3];
            const E = d[top + 4];
            const F = d[top + 5];
            const G = d[top + 6];
            const H = d[top + 7];
            put(d, o, 0, 0, avg2(A, B));
            const e1 = avg2(B, C);
            put(d, o, 1, 0, e1);
            put(d, o, 0, 2, e1);
            const e2 = avg2(C, D);
            put(d, o, 2, 0, e2);
            put(d, o, 1, 2, e2);
            const e3 = avg2(D, E);
            put(d, o, 3, 0, e3);
            put(d, o, 2, 2, e3);
            put(d, o, 0, 1, avg3(A, B, C));
            const f1 = avg3(B, C, D);
            put(d, o, 1, 1, f1);
            put(d, o, 0, 3, f1);
            const f2 = avg3(C, D, E);
            put(d, o, 2, 1, f2);
            put(d, o, 1, 3, f2);
            const f3 = avg3(D, E, F);
            put(d, o, 3, 1, f3);
            put(d, o, 2, 3, f3);
            put(d, o, 3, 2, avg3(E, F, G));
            put(d, o, 3, 3, avg3(F, G, H));
        },
        B_HD => {
            const I = d[o - 1];
            const J = d[o - 1 + BPS];
            const K = d[o - 1 + 2 * BPS];
            const L = d[o - 1 + 3 * BPS];
            const X = d[top - 1];
            const A = d[top];
            const B = d[top + 1];
            const C = d[top + 2];
            const g1 = avg2(I, X);
            put(d, o, 0, 0, g1);
            put(d, o, 2, 1, g1);
            const g2 = avg2(J, I);
            put(d, o, 0, 1, g2);
            put(d, o, 2, 2, g2);
            const g3 = avg2(K, J);
            put(d, o, 0, 2, g3);
            put(d, o, 2, 3, g3);
            put(d, o, 0, 3, avg2(L, K));
            put(d, o, 3, 0, avg3(A, B, C));
            put(d, o, 2, 0, avg3(X, A, B));
            const h1 = avg3(I, X, A);
            put(d, o, 1, 0, h1);
            put(d, o, 3, 1, h1);
            const h2 = avg3(J, I, X);
            put(d, o, 1, 1, h2);
            put(d, o, 3, 2, h2);
            const h3 = avg3(K, J, I);
            put(d, o, 1, 2, h3);
            put(d, o, 3, 3, h3);
            put(d, o, 1, 3, avg3(L, K, J));
        },
        B_HU => {
            const I = d[o - 1];
            const J = d[o - 1 + BPS];
            const K = d[o - 1 + 2 * BPS];
            const L = d[o - 1 + 3 * BPS];
            put(d, o, 0, 0, avg2(I, J));
            const hu1 = avg2(J, K);
            put(d, o, 2, 0, hu1);
            put(d, o, 0, 1, hu1);
            const hu2 = avg2(K, L);
            put(d, o, 2, 1, hu2);
            put(d, o, 0, 2, hu2);
            put(d, o, 1, 0, avg3(I, J, K));
            const j1 = avg3(J, K, L);
            put(d, o, 3, 0, j1);
            put(d, o, 1, 1, j1);
            const j2 = avg3(K, L, L);
            put(d, o, 3, 1, j2);
            put(d, o, 1, 2, j2);
            put(d, o, 3, 2, L);
            put(d, o, 2, 2, L);
            put(d, o, 0, 3, L);
            put(d, o, 1, 3, L);
            put(d, o, 2, 3, L);
            put(d, o, 3, 3, L);
        },
        else => unreachable,
    }
}

inline fn put(d: []u8, o: usize, x: usize, y: usize, v: u8) void {
    d[o + x + y * BPS] = v;
}

fn trueMotion(d: []u8, o: usize, size: usize) void {
    const top = o - BPS;
    const tl: i32 = d[top - 1];
    for (0..size) |y| {
        const l: i32 = d[o - 1 + y * BPS];
        for (0..size) |x| d[o + y * BPS + x] = clip8(l + d[top + x] - tl);
    }
}

/// Predict a 16x16 luma or 8x8 chroma block. `size` is 16 or 8.
fn predictBig(mode: u8, d: []u8, o: usize, size: usize, mb_x: usize, mb_y: usize) void {
    const top = o - BPS;
    switch (mode) {
        B_DC => {
            const shift: u5 = if (size == 16) 5 else 4;
            var dc: u32 = 0;
            const has_top = mb_y != 0;
            const has_left = mb_x != 0;
            if (!has_top and !has_left) {
                dc = 0x80;
            } else if (!has_top) {
                dc = @intCast(size / 2);
                for (0..size) |j| dc += d[o - 1 + j * BPS];
                dc >>= shift - 1;
            } else if (!has_left) {
                dc = @intCast(size / 2);
                for (0..size) |j| dc += d[top + j];
                dc >>= shift - 1;
            } else {
                dc = @intCast(size);
                for (0..size) |j| dc += @as(u32, d[top + j]) + d[o - 1 + j * BPS];
                dc >>= shift;
            }
            for (0..size) |y| @memset(d[o + y * BPS ..][0..size], @intCast(dc));
        },
        B_TM => trueMotion(d, o, size),
        B_VE => for (0..size) |y| @memcpy(d[o + y * BPS ..][0..size], d[top..][0..size]),
        B_HE => for (0..size) |y| @memset(d[o + y * BPS ..][0..size], d[o - 1 + y * BPS]),
        else => unreachable,
    }
}

// ---- loop filter -----------------------------------------------------------------------------------------------------
inline fn sclip1(v: i32) i32 {
    return std.math.clamp(v, -128, 127);
}
inline fn sclip2(v: i32) i32 {
    return std.math.clamp(v, -16, 15);
}

inline fn px(p: []u8, i: isize) i32 {
    return p[@intCast(i)];
}

fn doFilter2(p: []u8, base: usize, step: usize) void {
    const s: isize = @intCast(step);
    const b: isize = @intCast(base);
    const p1 = px(p, b - 2 * s);
    const p0 = px(p, b - s);
    const q0 = px(p, b);
    const q1 = px(p, b + s);
    const a = 3 * (q0 - p0) + sclip1(p1 - q1);
    const a1 = sclip2((a + 4) >> 3);
    const a2 = sclip2((a + 3) >> 3);
    p[base - step] = clip8(p0 + a2);
    p[base] = clip8(q0 - a1);
}

fn doFilter4(p: []u8, base: usize, step: usize) void {
    const s: isize = @intCast(step);
    const b: isize = @intCast(base);
    const p1 = px(p, b - 2 * s);
    const p0 = px(p, b - s);
    const q0 = px(p, b);
    const q1 = px(p, b + s);
    const a = 3 * (q0 - p0);
    const a1 = sclip2((a + 4) >> 3);
    const a2 = sclip2((a + 3) >> 3);
    const a3 = (a1 + 1) >> 1;
    p[base - 2 * step] = clip8(p1 + a3);
    p[base - step] = clip8(p0 + a2);
    p[base] = clip8(q0 - a1);
    p[base + step] = clip8(q1 - a3);
}

fn doFilter6(p: []u8, base: usize, step: usize) void {
    const s: isize = @intCast(step);
    const b: isize = @intCast(base);
    const p2 = px(p, b - 3 * s);
    const p1 = px(p, b - 2 * s);
    const p0 = px(p, b - s);
    const q0 = px(p, b);
    const q1 = px(p, b + s);
    const q2 = px(p, b + 2 * s);
    const a = sclip1(3 * (q0 - p0) + sclip1(p1 - q1));
    const a1 = (27 * a + 63) >> 7;
    const a2 = (18 * a + 63) >> 7;
    const a3 = (9 * a + 63) >> 7;
    p[base - 3 * step] = clip8(p2 + a3);
    p[base - 2 * step] = clip8(p1 + a2);
    p[base - step] = clip8(p0 + a1);
    p[base] = clip8(q0 - a1);
    p[base + step] = clip8(q1 - a2);
    p[base + 2 * step] = clip8(q2 - a3);
}

inline fn hev(p: []u8, base: usize, step: usize, thresh: i32) bool {
    const s: isize = @intCast(step);
    const b: isize = @intCast(base);
    const p1 = px(p, b - 2 * s);
    const p0 = px(p, b - s);
    const q0 = px(p, b);
    const q1 = px(p, b + s);
    return @abs(p1 - p0) > thresh or @abs(q1 - q0) > thresh;
}

inline fn needsFilter(p: []u8, base: usize, step: usize, t: i32) bool {
    const s: isize = @intCast(step);
    const b: isize = @intCast(base);
    const p1 = px(p, b - 2 * s);
    const p0 = px(p, b - s);
    const q0 = px(p, b);
    const q1 = px(p, b + s);
    return 4 * @as(i32, @intCast(@abs(p0 - q0))) + @as(i32, @intCast(@abs(p1 - q1))) <= t;
}

fn needsFilter2(p: []u8, base: usize, step: usize, t: i32, it: i32) bool {
    const s: isize = @intCast(step);
    const b: isize = @intCast(base);
    const p3 = px(p, b - 4 * s);
    const p2 = px(p, b - 3 * s);
    const p1 = px(p, b - 2 * s);
    const p0 = px(p, b - s);
    const q0 = px(p, b);
    const q1 = px(p, b + s);
    const q2 = px(p, b + 2 * s);
    const q3 = px(p, b + 3 * s);
    if (4 * @as(i32, @intCast(@abs(p0 - q0))) + @as(i32, @intCast(@abs(p1 - q1))) > t) return false;
    return @abs(p3 - p2) <= it and @abs(p2 - p1) <= it and @abs(p1 - p0) <= it and @abs(q3 - q2) <= it and @abs(q2 - q1) <= it and @abs(q1 - q0) <= it;
}

fn simpleFilter(p: []u8, base: usize, hstride: usize, vstride: usize, count: usize, thresh: i32) void {
    const thresh2 = 2 * thresh + 1;
    for (0..count) |i| {
        const at = base + i * vstride;
        if (needsFilter(p, at, hstride, thresh2)) doFilter2(p, at, hstride);
    }
}

fn filterLoop26(p: []u8, base: usize, hstride: usize, vstride: usize, count: usize, thresh: i32, ithresh: i32, hev_thresh: i32) void {
    const thresh2 = 2 * thresh + 1;
    for (0..count) |i| {
        const at = base + i * vstride;
        if (needsFilter2(p, at, hstride, thresh2, ithresh)) {
            if (hev(p, at, hstride, hev_thresh)) doFilter2(p, at, hstride) else doFilter6(p, at, hstride);
        }
    }
}

fn filterLoop24(p: []u8, base: usize, hstride: usize, vstride: usize, count: usize, thresh: i32, ithresh: i32, hev_thresh: i32) void {
    const thresh2 = 2 * thresh + 1;
    for (0..count) |i| {
        const at = base + i * vstride;
        if (needsFilter2(p, at, hstride, thresh2, ithresh)) {
            if (hev(p, at, hstride, hev_thresh)) doFilter2(p, at, hstride) else doFilter4(p, at, hstride);
        }
    }
}

// ---- the decoder -------------------------------------------------------------------------------------------------------
pub const FrameInfo = struct { width: usize, height: usize };

/// Dimensions from a key frame's header (the 10 bytes of the VP8 chunk), or null.
pub fn frameInfo(data: []const u8) ?FrameInfo {
    if (data.len < 10) return null;
    const bits = @as(u32, data[0]) | (@as(u32, data[1]) << 8) | (@as(u32, data[2]) << 16);
    if ((bits & 1) != 0) return null; // not a key frame
    if (data[3] != 0x9d or data[4] != 0x01 or data[5] != 0x2a) return null;
    const w = (@as(usize, data[6]) | (@as(usize, data[7]) << 8)) & 0x3fff;
    const h = (@as(usize, data[8]) | (@as(usize, data[9]) << 8)) & 0x3fff;
    if (w == 0 or h == 0) return null;
    return .{ .width = w, .height = h };
}

fn mbDefaultsLeft() void {}

pub fn decode(a: Allocator, data: []const u8) Error!Planes {
    if (data.len < 10) return error.Invalid;
    const bits = @as(u32, data[0]) | (@as(u32, data[1]) << 8) | (@as(u32, data[2]) << 16);
    const key_frame = (bits & 1) == 0;
    const profile = (bits >> 1) & 7;
    const show = (bits >> 4) & 1;
    const part0_len: usize = bits >> 5;
    if (!key_frame or profile > 3 or show == 0) return error.Invalid;
    const info = frameInfo(data) orelse return error.Invalid;
    const width = info.width;
    const height = info.height;
    var rest = data[10..];
    if (part0_len > rest.len) return error.Invalid;
    var br = BitReader.init(rest[0..part0_len]);
    rest = rest[part0_len..];

    _ = br.getBit(0x80); // colorspace
    _ = br.getBit(0x80); // clamp type
    // segment header
    var seg: Segment = .{};
    seg.use = br.getBit(0x80) != 0;
    if (seg.use) {
        seg.update_map = br.getBit(0x80) != 0;
        if (br.getBit(0x80) != 0) {
            seg.absolute = br.getBit(0x80) != 0;
            for (&seg.quantizer) |*q| q.* = if (br.getBit(0x80) != 0) br.getSignedValue(7) else 0;
            for (&seg.filter_strength) |*f| f.* = if (br.getBit(0x80) != 0) br.getSignedValue(6) else 0;
        }
        if (seg.update_map) for (&seg.probs) |*p| {
            p.* = if (br.getBit(0x80) != 0) br.getValue(8) else 255;
        };
    } else seg.update_map = false;
    // filter header
    const simple = br.getBit(0x80) != 0;
    const level: i32 = @intCast(br.getValue(6));
    const sharpness: i32 = @intCast(br.getValue(3));
    const use_lf_delta = br.getBit(0x80) != 0;
    var ref_lf_delta = [4]i32{ 0, 0, 0, 0 };
    var mode_lf_delta = [4]i32{ 0, 0, 0, 0 };
    if (use_lf_delta) {
        if (br.getBit(0x80) != 0) {
            for (&ref_lf_delta) |*d| if (br.getBit(0x80) != 0) {
                d.* = br.getSignedValue(6);
            };
            for (&mode_lf_delta) |*d| if (br.getBit(0x80) != 0) {
                d.* = br.getSignedValue(6);
            };
        }
    }
    const filter_type: u8 = if (level == 0) 0 else if (simple) 1 else 2;
    // partitions
    const num_parts: usize = @as(usize, 1) << @intCast(br.getValue(2));
    const last_part = num_parts - 1;
    if (rest.len < 3 * last_part) return error.Invalid;
    var part_start = rest[3 * last_part ..];
    var size_left = part_start.len;
    var parts: [8]BitReader = undefined;
    for (0..last_part) |p| {
        const sz = rest[3 * p ..][0..3];
        var psize: usize = @as(usize, sz[0]) | (@as(usize, sz[1]) << 8) | (@as(usize, sz[2]) << 16);
        if (psize > size_left) psize = size_left;
        parts[p] = BitReader.init(part_start[0..psize]);
        part_start = part_start[psize..];
        size_left -= psize;
    }
    parts[last_part] = BitReader.init(part_start);
    // quantizers
    const base_q0: i32 = @intCast(br.getValue(7));
    const dqy1_dc: i32 = if (br.getBit(0x80) != 0) br.getSignedValue(4) else 0;
    const dqy2_dc: i32 = if (br.getBit(0x80) != 0) br.getSignedValue(4) else 0;
    const dqy2_ac: i32 = if (br.getBit(0x80) != 0) br.getSignedValue(4) else 0;
    const dquv_dc: i32 = if (br.getBit(0x80) != 0) br.getSignedValue(4) else 0;
    const dquv_ac: i32 = if (br.getBit(0x80) != 0) br.getSignedValue(4) else 0;
    var dqm: [4]Quant = undefined;
    for (0..4) |i| {
        var q: i32 = undefined;
        if (seg.use) {
            q = seg.quantizer[i];
            if (!seg.absolute) q += base_q0;
        } else {
            if (i > 0) {
                dqm[i] = dqm[0];
                continue;
            }
            q = base_q0;
        }
        dqm[i].y1[0] = tables.dc_table[@intCast(std.math.clamp(q + dqy1_dc, 0, 127))];
        dqm[i].y1[1] = tables.ac_table[@intCast(std.math.clamp(q, 0, 127))];
        dqm[i].y2[0] = @as(i32, tables.dc_table[@intCast(std.math.clamp(q + dqy2_dc, 0, 127))]) * 2;
        dqm[i].y2[1] = (@as(i32, tables.ac_table[@intCast(std.math.clamp(q + dqy2_ac, 0, 127))]) * 101581) >> 16;
        if (dqm[i].y2[1] < 8) dqm[i].y2[1] = 8;
        dqm[i].uv[0] = tables.dc_table[@intCast(std.math.clamp(q + dquv_dc, 0, 117))];
        dqm[i].uv[1] = tables.ac_table[@intCast(std.math.clamp(q + dquv_ac, 0, 127))];
    }
    _ = br.getBit(0x80); // update_proba (ignored for key frames)
    // token probabilities
    var proba: [4][8][3][11]u8 = undefined;
    for (0..4) |t| for (0..8) |b| for (0..3) |c| for (0..11) |p| {
        const idx = ((t * 8 + b) * 3 + c) * 11 + p;
        proba[t][b][c][p] = if (br.getBit(tables.coeffs_update_proba[idx]) != 0) @intCast(br.getValue(8)) else tables.coeffs_proba0[idx];
    };
    const use_skip_proba = br.getBit(0x80) != 0;
    const skip_p: u32 = if (use_skip_proba) br.getValue(8) else 0;

    // filter strengths
    var fstrength: [4][2]FInfo = undefined;
    for (0..4) |s| {
        var base_level: i32 = undefined;
        if (seg.use) {
            base_level = seg.filter_strength[s];
            if (!seg.absolute) base_level += level;
        } else base_level = level;
        for (0..2) |is4| {
            var lv = base_level;
            if (use_lf_delta) {
                lv += ref_lf_delta[0];
                if (is4 == 1) lv += mode_lf_delta[0];
            }
            lv = std.math.clamp(lv, 0, 63);
            var fi: FInfo = .{};
            if (lv > 0) {
                var il = lv;
                if (sharpness > 0) {
                    il >>= if (sharpness > 4) 2 else 1;
                    if (il > 9 - sharpness) il = 9 - sharpness;
                }
                if (il < 1) il = 1;
                fi.ilevel = @intCast(il);
                fi.limit = @intCast(2 * lv + il);
                fi.hev = if (lv >= 40) 2 else if (lv >= 15) 1 else 0;
            } else fi.limit = 0;
            fi.inner = is4 == 1;
            fstrength[s][is4] = fi;
        }
    }

    const mb_w = (width + 15) >> 4;
    const mb_h = (height + 15) >> 4;
    const y_stride = mb_w * 16;
    const uv_stride = mb_w * 8;
    const yp = try a.alloc(u8, y_stride * mb_h * 16);
    errdefer a.free(yp);
    const up = try a.alloc(u8, uv_stride * mb_h * 8);
    errdefer a.free(up);
    const vp = try a.alloc(u8, uv_stride * mb_h * 8);
    errdefer a.free(vp);
    const finfo = try a.alloc(FInfo, mb_w * mb_h);
    defer a.free(finfo);

    // contexts
    const intra_top = try a.alloc(u8, mb_w * 4);
    defer a.free(intra_top);
    @memset(intra_top, B_DC);
    const top_y_nz = try a.alloc(u8, mb_w * 4);
    defer a.free(top_y_nz);
    @memset(top_y_nz, 0);
    const top_u_nz = try a.alloc(u8, mb_w * 2);
    defer a.free(top_u_nz);
    @memset(top_u_nz, 0);
    const top_v_nz = try a.alloc(u8, mb_w * 2);
    defer a.free(top_v_nz);
    @memset(top_v_nz, 0);
    const top_dc_nz = try a.alloc(u8, mb_w);
    defer a.free(top_dc_nz);
    @memset(top_dc_nz, 0);
    const top_y = try a.alloc(u8, mb_w * 16);
    defer a.free(top_y);
    const top_u = try a.alloc(u8, mb_w * 8);
    defer a.free(top_u);
    const top_v = try a.alloc(u8, mb_w * 8);
    defer a.free(top_v);

    var yb: [BPS * 17 + BPS]u8 = undefined; // rows -1..15 of the luma work block; row r at (r + 1) * BPS
    var ub: [BPS * 9 + BPS]u8 = undefined;
    var vb: [BPS * 9 + BPS]u8 = undefined;
    const Y0 = BPS + 8; // offset of pixel (0, 0) in yb
    const U0 = BPS + 8;

    var mb_y: usize = 0;
    while (mb_y < mb_h) : (mb_y += 1) {
        const tbr = &parts[mb_y & last_part];
        var intra_left = [4]u8{ B_DC, B_DC, B_DC, B_DC };
        var left_y_nz = [4]u8{ 0, 0, 0, 0 };
        var left_u_nz = [2]u8{ 0, 0 };
        var left_v_nz = [2]u8{ 0, 0 };
        var left_dc_nz: u8 = 0;
        // left border for x = 0
        for (0..16) |j| yb[Y0 + j * BPS - 1] = 129;
        for (0..8) |j| {
            ub[U0 + j * BPS - 1] = 129;
            vb[U0 + j * BPS - 1] = 129;
        }
        if (mb_y > 0) {
            yb[Y0 - BPS - 1] = 129;
            ub[U0 - BPS - 1] = 129;
            vb[U0 - BPS - 1] = 129;
        } else {
            @memset(yb[Y0 - BPS - 1 ..][0 .. 16 + 4 + 1], 127);
            @memset(ub[U0 - BPS - 1 ..][0 .. 8 + 1], 127);
            @memset(vb[U0 - BPS - 1 ..][0 .. 8 + 1], 127);
        }
        var mb_x: usize = 0;
        while (mb_x < mb_w) : (mb_x += 1) {
            // ---- intra modes (partition 0)
            var segment: usize = 0;
            if (seg.update_map) {
                segment = if (br.getBit(seg.probs[0]) == 0) br.getBit(seg.probs[1]) else br.getBit(seg.probs[2]) + 2;
            }
            var skip = false;
            if (use_skip_proba) skip = br.getBit(skip_p) != 0;
            const is_i4x4 = br.getBit(145) == 0;
            var imodes: [16]u8 = undefined;
            var ymode16: u8 = 0;
            const top = intra_top[mb_x * 4 ..][0..4];
            if (!is_i4x4) {
                ymode16 = if (br.getBit(156) != 0) (if (br.getBit(128) != 0) B_TM else B_HE) else (if (br.getBit(163) != 0) B_VE else B_DC);
                @memset(top, ymode16);
                @memset(&intra_left, ymode16);
            } else {
                for (0..4) |y| {
                    var ymode = intra_left[y];
                    for (0..4) |x| {
                        const prob = tables.bmodes_proba[(@as(usize, top[x]) * 10 + ymode) * 9 ..][0..9];
                        var i: i32 = kYModesIntra4[br.getBit(prob[0])];
                        while (i > 0) i = kYModesIntra4[@as(usize, @intCast(2 * i)) + br.getBit(prob[@intCast(i)])];
                        ymode = @intCast(-i);
                        top[x] = ymode;
                        imodes[y * 4 + x] = ymode;
                    }
                    intra_left[y] = top[3];
                }
            }
            const uvmode: u8 = if (br.getBit(142) == 0) B_DC else if (br.getBit(114) == 0) B_VE else if (br.getBit(183) != 0) B_TM else B_HE;

            // ---- residuals (token partition)
            var coeffs: [384]i16 = @splat(0);
            var non_zero_y: u32 = 0;
            var non_zero_uv: u32 = 0;
            var has_coeffs = false;
            const q = &dqm[segment];
            if (!skip) {
                var ac_type: usize = 3;
                var first: usize = 0;
                if (!is_i4x4) {
                    var dc: [16]i16 = @splat(0);
                    const ctx = top_dc_nz[mb_x] + left_dc_nz;
                    const nz = getCoeffs(tbr, &proba[1], ctx, q.y2, 0, &dc);
                    top_dc_nz[mb_x] = @intFromBool(nz > 0);
                    left_dc_nz = top_dc_nz[mb_x];
                    if (nz > 1) {
                        transformWHT(&dc, &coeffs);
                    } else {
                        const dc0: i16 = @truncate((@as(i32, dc[0]) + 3) >> 3);
                        var i: usize = 0;
                        while (i < 256) : (i += 16) coeffs[i] = dc0;
                    }
                    first = 1;
                    ac_type = 0;
                }
                for (0..4) |y| {
                    var l: u8 = left_y_nz[y];
                    for (0..4) |x| {
                        const ctx = l + top_y_nz[mb_x * 4 + x];
                        const blk: *[16]i16 = coeffs[(y * 4 + x) * 16 ..][0..16];
                        const nz = getCoeffs(tbr, &proba[ac_type], ctx, q.y1, first, blk);
                        l = @intFromBool(nz > first);
                        top_y_nz[mb_x * 4 + x] = l;
                        non_zero_y = (non_zero_y << 2) | nzCode(nz, blk[0] != 0);
                    }
                    left_y_nz[y] = l;
                }
                inline for (.{ 0, 1 }) |ch| {
                    const left_arr = if (ch == 0) &left_u_nz else &left_v_nz;
                    const top_arr = if (ch == 0) top_u_nz else top_v_nz;
                    for (0..2) |y| {
                        var l: u8 = left_arr[y];
                        for (0..2) |x| {
                            const ctx = l + top_arr[mb_x * 2 + x];
                            const blk: *[16]i16 = coeffs[256 + (ch * 4 + y * 2 + x) * 16 ..][0..16];
                            const nz = getCoeffs(tbr, &proba[2], ctx, q.uv, 0, blk);
                            l = @intFromBool(nz > 0);
                            top_arr[mb_x * 2 + x] = l;
                            non_zero_uv = (non_zero_uv << 2) | nzCode(nz, blk[0] != 0);
                        }
                        left_arr[y] = l;
                    }
                }
                has_coeffs = (non_zero_y | non_zero_uv) != 0;
                skip = !has_coeffs;
            } else {
                @memset(&left_y_nz, 0);
                @memset(top_y_nz[mb_x * 4 ..][0..4], 0);
                @memset(&left_u_nz, 0);
                @memset(top_u_nz[mb_x * 2 ..][0..2], 0);
                @memset(&left_v_nz, 0);
                @memset(top_v_nz[mb_x * 2 ..][0..2], 0);
                if (!is_i4x4) {
                    left_dc_nz = 0;
                    top_dc_nz[mb_x] = 0;
                }
            }
            if (tbr.eof) return error.Invalid;

            if (filter_type > 0) {
                var fi = fstrength[segment][@intFromBool(is_i4x4)];
                fi.inner = fi.inner or !skip;
                finfo[mb_y * mb_w + mb_x] = fi;
            }

            // ---- reconstruction
            // top borders
            if (mb_y > 0) {
                @memcpy(yb[Y0 - BPS ..][0..16], top_y[mb_x * 16 ..][0..16]);
                @memcpy(ub[U0 - BPS ..][0..8], top_u[mb_x * 8 ..][0..8]);
                @memcpy(vb[U0 - BPS ..][0..8], top_v[mb_x * 8 ..][0..8]);
                if (mb_x >= mb_w - 1) {
                    @memset(yb[Y0 - BPS + 16 ..][0..4], top_y[mb_x * 16 + 15]);
                } else {
                    @memcpy(yb[Y0 - BPS + 16 ..][0..4], top_y[(mb_x + 1) * 16 ..][0..4]);
                }
            } else {
                @memset(yb[Y0 - BPS ..][0..20], 127);
                @memset(ub[U0 - BPS ..][0..8], 127);
                @memset(vb[U0 - BPS ..][0..8], 127);
            }
            // top-right replicated for sub-block rows 1..3
            for ([_]usize{ 3, 7, 11 }) |r| @memcpy(yb[Y0 + r * BPS + 16 ..][0..4], yb[Y0 - BPS + 16 ..][0..4]);

            if (!is_i4x4) {
                predictBig(ymode16, &yb, Y0, 16, mb_x, mb_y);
                if (has_coeffs) addResiduals16(&coeffs, &yb, Y0, non_zero_y);
            } else {
                for (0..16) |n| {
                    const bx = n & 3;
                    const by = n >> 2;
                    const o = Y0 + by * 4 * BPS + bx * 4;
                    predict4(imodes[n], &yb, o);
                    const code = (non_zero_y >> @intCast(2 * (15 - n))) & 3;
                    if (code != 0) transformOne(coeffs[n * 16 ..][0..16], yb[o..], BPS);
                }
            }
            predictBig(uvmode, &ub, U0, 8, mb_x, mb_y);
            predictBig(uvmode, &vb, U0, 8, mb_x, mb_y);
            if (has_coeffs) {
                for (0..4) |n| {
                    const o = U0 + (n >> 1) * 4 * BPS + (n & 1) * 4;
                    if (((non_zero_uv >> @intCast(2 * (7 - n))) & 3) != 0) transformOne(coeffs[256 + n * 16 ..][0..16], ub[o..], BPS);
                    if (((non_zero_uv >> @intCast(2 * (3 - n))) & 3) != 0) transformOne(coeffs[256 + (4 + n) * 16 ..][0..16], vb[o..], BPS);
                }
            }
            // ---- store: planes, top lines, left columns for the next MB
            for (0..16) |j| @memcpy(yp[(mb_y * 16 + j) * y_stride + mb_x * 16 ..][0..16], yb[Y0 + j * BPS ..][0..16]);
            for (0..8) |j| {
                @memcpy(up[(mb_y * 8 + j) * uv_stride + mb_x * 8 ..][0..8], ub[U0 + j * BPS ..][0..8]);
                @memcpy(vp[(mb_y * 8 + j) * uv_stride + mb_x * 8 ..][0..8], vb[U0 + j * BPS ..][0..8]);
            }
            @memcpy(top_y[mb_x * 16 ..][0..16], yb[Y0 + 15 * BPS ..][0..16]);
            @memcpy(top_u[mb_x * 8 ..][0..8], ub[U0 + 7 * BPS ..][0..8]);
            @memcpy(top_v[mb_x * 8 ..][0..8], vb[U0 + 7 * BPS ..][0..8]);
            // rotate: column 15 (rows -1..15) becomes column -1
            for (0..17) |j| yb[Y0 + (j * BPS) - BPS - 1] = yb[Y0 + (j * BPS) - BPS + 15];
            for (0..9) |j| {
                ub[U0 + (j * BPS) - BPS - 1] = ub[U0 + (j * BPS) - BPS + 7];
                vb[U0 + (j * BPS) - BPS - 1] = vb[U0 + (j * BPS) - BPS + 7];
            }
        }
    }
    if (br.eof and false) return error.Invalid;

    // ---- loop filter over the whole frame, macroblocks in raster order
    if (filter_type > 0) {
        var my: usize = 0;
        while (my < mb_h) : (my += 1) {
            var mx: usize = 0;
            while (mx < mb_w) : (mx += 1) {
                const fi = finfo[my * mb_w + mx];
                const limit: i32 = fi.limit;
                if (limit == 0) continue;
                const yo = my * 16 * y_stride + mx * 16;
                const uo = my * 8 * uv_stride + mx * 8;
                if (filter_type == 1) {
                    if (mx > 0) simpleFilter(yp, yo, 1, y_stride, 16, limit + 4);
                    if (fi.inner) {
                        for (1..4) |k| simpleFilter(yp, yo + 4 * k, 1, y_stride, 16, limit);
                    }
                    if (my > 0) simpleFilter(yp, yo, y_stride, 1, 16, limit + 4);
                    if (fi.inner) {
                        for (1..4) |k| simpleFilter(yp, yo + 4 * k * y_stride, y_stride, 1, 16, limit);
                    }
                } else {
                    const il: i32 = fi.ilevel;
                    const hv: i32 = fi.hev;
                    if (mx > 0) {
                        filterLoop26(yp, yo, 1, y_stride, 16, limit + 4, il, hv);
                        filterLoop26(up, uo, 1, uv_stride, 8, limit + 4, il, hv);
                        filterLoop26(vp, uo, 1, uv_stride, 8, limit + 4, il, hv);
                    }
                    if (fi.inner) {
                        for (1..4) |k| filterLoop24(yp, yo + 4 * k, 1, y_stride, 16, limit, il, hv);
                        filterLoop24(up, uo + 4, 1, uv_stride, 8, limit, il, hv);
                        filterLoop24(vp, uo + 4, 1, uv_stride, 8, limit, il, hv);
                    }
                    if (my > 0) {
                        filterLoop26(yp, yo, y_stride, 1, 16, limit + 4, il, hv);
                        filterLoop26(up, uo, uv_stride, 1, 8, limit + 4, il, hv);
                        filterLoop26(vp, uo, uv_stride, 1, 8, limit + 4, il, hv);
                    }
                    if (fi.inner) {
                        for (1..4) |k| filterLoop24(yp, yo + 4 * k * y_stride, y_stride, 1, 16, limit, il, hv);
                        filterLoop24(up, uo + 4 * uv_stride, uv_stride, 1, 8, limit, il, hv);
                        filterLoop24(vp, uo + 4 * uv_stride, uv_stride, 1, 8, limit, il, hv);
                    }
                }
            }
        }
    }
    return .{ .width = width, .height = height, .y = yp, .u = up, .v = vp, .y_stride = y_stride, .uv_stride = uv_stride };
}

fn addResiduals16(coeffs: *const [384]i16, yb: *[BPS * 17 + BPS]u8, y0: usize, non_zero_y: u32) void {
    for (0..16) |n| {
        const code = (non_zero_y >> @intCast(2 * (15 - n))) & 3;
        if (code == 0) continue;
        const o = y0 + (n >> 2) * 4 * BPS + (n & 3) * 4;
        transformOne(coeffs[n * 16 ..][0..16], yb[o..], BPS);
    }
}

inline fn nzCode(nz: usize, dc_nz: bool) u32 {
    return if (nz > 3) 3 else if (nz > 1) 2 else @intFromBool(dc_nz);
}

fn getLargeValue(br: *BitReader, p: *const [11]u8) i32 {
    var v: i32 = undefined;
    if (br.getBit(p[3]) == 0) {
        if (br.getBit(p[4]) == 0) v = 2 else v = 3 + @as(i32, @intCast(br.getBit(p[5])));
    } else {
        if (br.getBit(p[6]) == 0) {
            if (br.getBit(p[7]) == 0) {
                v = 5 + @as(i32, @intCast(br.getBit(159)));
            } else {
                v = 7 + 2 * @as(i32, @intCast(br.getBit(165)));
                v += @intCast(br.getBit(145));
            }
        } else {
            const bit1 = br.getBit(p[8]);
            const bit0 = br.getBit(p[9 + bit1]);
            const cat = 2 * bit1 + bit0;
            v = 0;
            for (cat_tabs[cat]) |pr| v += v + @as(i32, @intCast(br.getBit(pr)));
            v += 3 + (@as(i32, 8) << @intCast(cat));
        }
    }
    return v;
}

/// Returns the position after the last non-zero coefficient (0 if none), writing dequantized values.
fn getCoeffs(br: *BitReader, prob: *const [8][3][11]u8, ctx: usize, dq: [2]i32, first: usize, out: *[16]i16) usize {
    var n = first;
    var p: *const [11]u8 = &prob[bands[n]][ctx];
    while (n < 16) : (n += 1) {
        if (br.getBit(p[0]) == 0) return n; // previous coeff was last non-zero coeff
        while (br.getBit(p[1]) == 0) { // sequence of zero coeffs
            n += 1;
            p = &prob[bands[n]][0];
            if (n == 16) return 16;
        }
        // non zero coeff
        const next_band = &prob[bands[n + 1]];
        var v: i32 = undefined;
        if (br.getBit(p[2]) == 0) {
            v = 1;
            p = &next_band[1];
        } else {
            v = getLargeValue(br, p);
            p = &next_band[2];
        }
        const sign = br.getBit(0x80);
        const sv = if (sign != 0) -v else v;
        out[zigzag[n]] = @truncate(sv * dq[@intFromBool(n > 0)]);
    }
    return 16;
}
