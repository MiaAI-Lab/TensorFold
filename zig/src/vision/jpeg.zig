//! JPEG decoder bit-exact with libjpeg-turbo's defaults as Pillow 12.3 uses them: ISLOW integer IDCT ("jidctint"),
//! fancy (triangle) chroma upsampling for h2v1 / h2v2 / h1v2 and replication otherwise, the table-driven YCC->RGB
//! conversion, YCCK/CMYK (Adobe inverted samples), baseline / extended sequential / progressive Huffman, restart
//! intervals. Not supported (refused): arithmetic coding, lossless, 12-bit, DNL.
const std = @import("std");
const raster = @import("raster.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{Invalid} || Allocator.Error;

const zigzag = [64]u8{
    0,  1,  8,  16, 9,  2,  3,  10,
    17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
};

const Component = struct {
    id: u8,
    h: u8,
    v: u8,
    tq: u8,
    // geometry (after SOF)
    blocks_w: usize = 0, // allocated blocks per row (padded to the MCU grid)
    blocks_h: usize = 0,
    width_in_blocks: usize = 0, // real blocks (for non-interleaved scans)
    height_in_blocks: usize = 0,
    ds_w: usize = 0, // downsampled width/height in samples
    ds_h: usize = 0,
    coefs: []i16 = &.{},
};

const HuffTable = struct {
    present: bool = false,
    nsym: usize = 0,
    // canonical decoding
    maxcode: [18]i32 = undefined,
    valptr: [17]i32 = undefined,
    mincode: [17]i32 = undefined,
    values: [256]u8 = undefined,
    lookup: [512]u16 = undefined, // 9-bit lookahead: (len << 8) | symbol, 0 = needs the long path

    fn build(t: *HuffTable, counts: *const [16]u8, vals: []const u8) bool {
        var total: usize = 0;
        for (counts) |c| total += c;
        if (total > 256 or vals.len < total) return false;
        @memcpy(t.values[0..total], vals[0..total]);
        t.nsym = total;
        var code: i32 = 0;
        var k: i32 = 0;
        @memset(&t.lookup, 0);
        for (1..17) |l| {
            const n: i32 = counts[l - 1];
            t.valptr[l] = k;
            t.mincode[l] = code;
            if (n != 0) {
                if (code + n > (@as(i32, 1) << @intCast(l))) return false;
                if (l <= 9) {
                    var i: i32 = 0;
                    while (i < n) : (i += 1) {
                        const c = code + i;
                        const base: usize = @intCast(c << @intCast(9 - l));
                        const cnt: usize = @as(usize, 1) << @intCast(9 - l);
                        for (0..cnt) |j| t.lookup[base + j] = @intCast((@as(u32, @intCast(l)) << 8) | t.values[@intCast(k + i)]);
                    }
                }
                k += n;
                code += n;
                t.maxcode[l] = code - 1;
            } else t.maxcode[l] = -1;
            code <<= 1;
        }
        t.maxcode[17] = 0x7fffffff;
        t.present = true;
        return true;
    }
};

const BitReader = struct {
    data: []const u8,
    pos: usize,
    bits: u32 = 0,
    count: u6 = 0, // bits in `bits` (real ones first, then `fake` zero padding)
    fake: u6 = 0, // zero bits supplied after a marker
    marker: ?u8 = null, // a marker found inside the entropy-coded data (its FF is at `pos`)
    eof: bool = false, // the file ended inside entropy-coded data (libjpeg would suspend: Pillow calls it truncated)
    insufficient: bool = false, // zero bits were consumed (libjpeg's insufficient_data): the rest of the segment is skipped

    fn fill(r: *BitReader) void {
        while (r.count <= 24) {
            var b: u32 = 0;
            var padded = false;
            if (r.marker == null and !r.eof) {
                if (r.pos >= r.data.len) {
                    r.eof = true;
                    padded = true;
                } else {
                    const c = r.data[r.pos];
                    if (c == 0xff) {
                        var p = r.pos + 1;
                        while (p < r.data.len and r.data[p] == 0xff) p += 1;
                        if (p >= r.data.len) {
                            r.eof = true;
                            r.pos = r.data.len;
                            padded = true;
                        } else if (r.data[p] == 0) {
                            b = 0xff;
                            r.pos = p + 1;
                        } else {
                            r.marker = r.data[p];
                            r.pos = p - 1; // at the FF of the marker
                            padded = true;
                        }
                    } else {
                        b = c;
                        r.pos += 1;
                    }
                }
            } else padded = true;
            r.bits |= b << @intCast(24 - @as(u32, r.count));
            r.count += 8;
            if (padded) r.fake += 8;
        }
    }

    inline fn peek(r: *BitReader, n: u5) u32 {
        if (r.count < n) r.fill();
        return r.bits >> @intCast(32 - @as(u32, n));
    }
    inline fn skip(r: *BitReader, n: u5) void {
        r.bits <<= @intCast(n);
        r.count -= @as(u6, n);
        if (r.count < r.fake) {
            r.insufficient = true;
            r.fake = r.count;
        }
    }
    inline fn get(r: *BitReader, n: u5) u32 {
        if (n == 0) return 0;
        const v = r.peek(n);
        r.skip(n);
        return v;
    }
    fn reset(r: *BitReader) void {
        r.bits = 0;
        r.count = 0;
        r.fake = 0;
    }

    /// jpeg_huff_decode: a code that matches nothing yields symbol 0 after 17 bits (libjpeg warns and goes on).
    fn decode(r: *BitReader, t: *const HuffTable) Error!u8 {
        if (r.count < 25) r.fill();
        const look = t.lookup[r.bits >> 23];
        if (look != 0) {
            r.skip(@intCast(look >> 8));
            return @intCast(look & 0xff);
        }
        var l: u5 = 10;
        while (l <= 16) : (l += 1) {
            const code: i32 = @intCast(r.bits >> @intCast(32 - @as(u32, l)));
            if (t.maxcode[l] >= 0 and code <= t.maxcode[l]) {
                r.skip(l);
                const idx = t.valptr[l] + code - t.mincode[l];
                return t.values[@as(usize, @intCast(idx)) & 0xff];
            }
        }
        r.skip(17);
        return 0;
    }
};

inline fn extend(v: u32, s: u5) i32 {
    if (s == 0) return 0;
    const vt: i32 = @as(i32, 1) << @intCast(s - 1);
    const vi: i32 = @intCast(v);
    return if (vi < vt) vi - (@as(i32, 1) << @intCast(s)) + 1 else vi;
}

const Frame = struct {
    w: usize,
    h: usize,
    progressive: bool,
    comps: [4]Component,
    n: usize,
    hmax: u8,
    vmax: u8,
    mcus_x: usize,
    mcus_y: usize,
};

const Info = struct {
    jfif: bool = false,
    adobe: bool = false,
    adobe_transform: u8 = 0,
    exif: ?[]const u8 = null,
    xmp: ?[]const u8 = null,
};

/// (width, height) from the first SOF; null if the markers up to it are malformed.
pub fn header(data: []const u8) ?struct { w: u32, h: u32 } {
    if (data.len < 4 or data[0] != 0xff or data[1] != 0xd8) return null;
    var p: usize = 2;
    while (p + 4 <= data.len) {
        if (data[p] != 0xff) {
            p += 1;
            continue;
        }
        const m = data[p + 1];
        if (m == 0xff) {
            p += 1;
            continue;
        }
        if (m == 0 or m == 0x01 or (m >= 0xd0 and m <= 0xd8)) {
            p += 2;
            continue;
        }
        if (m == 0xd9 or m == 0xda) return null;
        const len = std.mem.readInt(u16, data[p + 2 ..][0..2], .big);
        if (len < 2 or p + 2 + len > data.len) return null;
        if (m >= 0xc0 and m <= 0xcf and m != 0xc4 and m != 0xc8 and m != 0xcc) {
            if (len < 8) return null;
            const body = data[p + 4 ..];
            const h = std.mem.readInt(u16, body[1..3], .big);
            const w = std.mem.readInt(u16, body[3..5], .big);
            if (body[0] != 8 or w == 0 or h == 0) return null;
            if (body[5] != 1 and body[5] != 3 and body[5] != 4) return null;
            return .{ .w = w, .h = h };
        }
        p += 2 + len;
    }
    return null;
}

// ---- IDCT (jidctint.c, jpeg_idct_islow) ---------------------------------------------------------------------
const CONST_BITS = 13;
const PASS1_BITS = 2;
const F_0_298631336: i32 = 2446;
const F_0_390180644: i32 = 3196;
const F_0_541196100: i32 = 4433;
const F_0_765366865: i32 = 6270;
const F_0_899976223: i32 = 7373;
const F_1_175875602: i32 = 9633;
const F_1_501321110: i32 = 12299;
const F_1_847759065: i32 = 15137;
const F_1_961570560: i32 = 16069;
const F_2_053119869: i32 = 16819;
const F_2_562915447: i32 = 20995;
const F_3_072711026: i32 = 25172;

inline fn descale(x: i32, comptime n: u5) i32 {
    return (x +% (@as(i32, 1) << (n - 1))) >> n;
}

inline fn rangeLimit(x: i32) u8 {
    const t = x & 1023;
    const s = if (t < 512) t else t - 1024;
    return @intCast(std.math.clamp(s + 128, 0, 255));
}

fn idctIslow(coef: *const [64]i16, quant: *const [64]u16, out: []u8, stride: usize) void {
    var ws: [64]i32 = undefined;
    for (0..8) |c| {
        const q = quant;
        var all_zero = true;
        for (1..8) |r| if (coef[r * 8 + c] != 0) {
            all_zero = false;
            break;
        };
        if (all_zero) {
            const dc: i32 = (@as(i32, coef[c]) *% @as(i32, q[c])) << PASS1_BITS;
            for (0..8) |r| ws[r * 8 + c] = dc;
            continue;
        }
        var z2: i32 = @as(i32, coef[2 * 8 + c]) *% q[2 * 8 + c];
        var z3: i32 = @as(i32, coef[6 * 8 + c]) *% q[6 * 8 + c];
        var z1 = (z2 +% z3) *% F_0_541196100;
        var tmp2 = z1 +% z3 *% -F_1_847759065;
        var tmp3 = z1 +% z2 *% F_0_765366865;
        z2 = @as(i32, coef[c]) *% q[c];
        z3 = @as(i32, coef[4 * 8 + c]) *% q[4 * 8 + c];
        var tmp0 = (z2 +% z3) << CONST_BITS;
        var tmp1 = (z2 -% z3) << CONST_BITS;
        const tmp10 = tmp0 +% tmp3;
        const tmp13 = tmp0 -% tmp3;
        const tmp11 = tmp1 +% tmp2;
        const tmp12 = tmp1 -% tmp2;
        tmp0 = @as(i32, coef[7 * 8 + c]) *% q[7 * 8 + c];
        tmp1 = @as(i32, coef[5 * 8 + c]) *% q[5 * 8 + c];
        tmp2 = @as(i32, coef[3 * 8 + c]) *% q[3 * 8 + c];
        tmp3 = @as(i32, coef[1 * 8 + c]) *% q[1 * 8 + c];
        z1 = tmp0 +% tmp3;
        z2 = tmp1 +% tmp2;
        z3 = tmp0 +% tmp2;
        var z4 = tmp1 +% tmp3;
        const z5 = (z3 +% z4) *% F_1_175875602;
        tmp0 = tmp0 *% F_0_298631336;
        tmp1 = tmp1 *% F_2_053119869;
        tmp2 = tmp2 *% F_3_072711026;
        tmp3 = tmp3 *% F_1_501321110;
        z1 = z1 *% -F_0_899976223;
        z2 = z2 *% -F_2_562915447;
        z3 = z3 *% -F_1_961570560;
        z4 = z4 *% -F_0_390180644;
        z3 +%= z5;
        z4 +%= z5;
        tmp0 +%= z1 +% z3;
        tmp1 +%= z2 +% z4;
        tmp2 +%= z2 +% z3;
        tmp3 +%= z1 +% z4;
        ws[0 * 8 + c] = descale(tmp10 +% tmp3, CONST_BITS - PASS1_BITS);
        ws[7 * 8 + c] = descale(tmp10 -% tmp3, CONST_BITS - PASS1_BITS);
        ws[1 * 8 + c] = descale(tmp11 +% tmp2, CONST_BITS - PASS1_BITS);
        ws[6 * 8 + c] = descale(tmp11 -% tmp2, CONST_BITS - PASS1_BITS);
        ws[2 * 8 + c] = descale(tmp12 +% tmp1, CONST_BITS - PASS1_BITS);
        ws[5 * 8 + c] = descale(tmp12 -% tmp1, CONST_BITS - PASS1_BITS);
        ws[3 * 8 + c] = descale(tmp13 +% tmp0, CONST_BITS - PASS1_BITS);
        ws[4 * 8 + c] = descale(tmp13 -% tmp0, CONST_BITS - PASS1_BITS);
    }
    for (0..8) |r| {
        const w = ws[r * 8 ..][0..8];
        const o = out[r * stride ..][0..8];
        var z2: i32 = w[2];
        var z3: i32 = w[6];
        var z1 = (z2 +% z3) *% F_0_541196100;
        var tmp2 = z1 +% z3 *% -F_1_847759065;
        var tmp3 = z1 +% z2 *% F_0_765366865;
        var tmp0 = (@as(i32, w[0]) +% w[4]) << CONST_BITS;
        var tmp1 = (@as(i32, w[0]) -% w[4]) << CONST_BITS;
        const tmp10 = tmp0 +% tmp3;
        const tmp13 = tmp0 -% tmp3;
        const tmp11 = tmp1 +% tmp2;
        const tmp12 = tmp1 -% tmp2;
        tmp0 = w[7];
        tmp1 = w[5];
        tmp2 = w[3];
        tmp3 = w[1];
        z1 = tmp0 +% tmp3;
        z2 = tmp1 +% tmp2;
        z3 = tmp0 +% tmp2;
        var z4 = tmp1 +% tmp3;
        const z5 = (z3 +% z4) *% F_1_175875602;
        tmp0 = tmp0 *% F_0_298631336;
        tmp1 = tmp1 *% F_2_053119869;
        tmp2 = tmp2 *% F_3_072711026;
        tmp3 = tmp3 *% F_1_501321110;
        z1 = z1 *% -F_0_899976223;
        z2 = z2 *% -F_2_562915447;
        z3 = z3 *% -F_1_961570560;
        z4 = z4 *% -F_0_390180644;
        z3 +%= z5;
        z4 +%= z5;
        tmp0 +%= z1 +% z3;
        tmp1 +%= z2 +% z4;
        tmp2 +%= z2 +% z3;
        tmp3 +%= z1 +% z4;
        o[0] = rangeLimit(descale(tmp10 +% tmp3, CONST_BITS + PASS1_BITS + 3));
        o[7] = rangeLimit(descale(tmp10 -% tmp3, CONST_BITS + PASS1_BITS + 3));
        o[1] = rangeLimit(descale(tmp11 +% tmp2, CONST_BITS + PASS1_BITS + 3));
        o[6] = rangeLimit(descale(tmp11 -% tmp2, CONST_BITS + PASS1_BITS + 3));
        o[2] = rangeLimit(descale(tmp12 +% tmp1, CONST_BITS + PASS1_BITS + 3));
        o[5] = rangeLimit(descale(tmp12 -% tmp1, CONST_BITS + PASS1_BITS + 3));
        o[3] = rangeLimit(descale(tmp13 +% tmp0, CONST_BITS + PASS1_BITS + 3));
        o[4] = rangeLimit(descale(tmp13 -% tmp0, CONST_BITS + PASS1_BITS + 3));
    }
}

// ---- colour conversion tables (jdcolor.c) -----------------------------------------------------------------------
const ColorTables = struct {
    cr_r: [256]i32,
    cb_b: [256]i32,
    cr_g: [256]i32,
    cb_g: [256]i32,
};

fn fix(x: f64) i32 {
    return @intFromFloat(x * 65536.0 + 0.5);
}

fn colorTables() ColorTables {
    var t: ColorTables = undefined;
    for (0..256) |i| {
        const x: i32 = @as(i32, @intCast(i)) - 128;
        t.cr_r[i] = (fix(1.40200) * x + 32768) >> 16;
        t.cb_b[i] = (fix(1.77200) * x + 32768) >> 16;
        t.cr_g[i] = -fix(0.71414) * x;
        t.cb_g[i] = -fix(0.34414) * x + 32768;
    }
    return t;
}

inline fn clamp8(v: i32) u8 {
    return @intCast(std.math.clamp(v, 0, 255));
}

// ---- decoder ------------------------------------------------------------------------------------------------------
const Decoder = struct {
    allocator: Allocator,
    data: []const u8,
    quant: [4][64]u16 = undefined, // natural order
    quant_set: [4]bool = .{ false, false, false, false },
    dc: [4]HuffTable = .{ .{}, .{}, .{}, .{} },
    ac: [4]HuffTable = .{ .{}, .{}, .{}, .{} },
    frame: ?Frame = null,
    restart: usize = 0,
    info: Info = .{},
    scans: usize = 0,
    eob_run: u32 = 0,

    fn deinit(d: *Decoder) void {
        if (d.frame) |*f| for (f.comps[0..f.n]) |c| if (c.coefs.len != 0) d.allocator.free(c.coefs);
    }

    fn u16at(d: *Decoder, p: usize) Error!u16 {
        if (p + 2 > d.data.len) return error.Invalid;
        return std.mem.readInt(u16, d.data[p..][0..2], .big);
    }

    fn parseSof(d: *Decoder, seg: []const u8, progressive: bool) Error!void {
        if (d.frame != null) return error.Invalid;
        if (seg.len < 6) return error.Invalid;
        if (seg[0] != 8) return error.Invalid;
        const h = std.mem.readInt(u16, seg[1..3], .big);
        const w = std.mem.readInt(u16, seg[3..5], .big);
        const n = seg[5];
        if (w == 0 or h == 0 or (n != 1 and n != 3 and n != 4) or seg.len < 6 + 3 * @as(usize, n)) return error.Invalid;
        var f: Frame = undefined;
        f.w = w;
        f.h = h;
        f.progressive = progressive;
        f.n = n;
        f.hmax = 1;
        f.vmax = 1;
        for (0..n) |i| {
            const b = seg[6 + 3 * i ..][0..3];
            const hs = b[1] >> 4;
            const vs = b[1] & 15;
            if (hs == 0 or hs > 4 or vs == 0 or vs > 4 or b[2] > 3) return error.Invalid;
            f.comps[i] = .{ .id = b[0], .h = hs, .v = vs, .tq = b[2] };
            f.hmax = @max(f.hmax, hs);
            f.vmax = @max(f.vmax, vs);
        }
        f.mcus_x = (w + 8 * @as(usize, f.hmax) - 1) / (8 * @as(usize, f.hmax));
        f.mcus_y = (h + 8 * @as(usize, f.vmax) - 1) / (8 * @as(usize, f.vmax));
        for (f.comps[0..n]) |*c| {
            c.blocks_w = f.mcus_x * c.h;
            c.blocks_h = f.mcus_y * c.v;
            c.ds_w = (@as(usize, w) * c.h + f.hmax - 1) / f.hmax;
            c.ds_h = (@as(usize, h) * c.v + f.vmax - 1) / f.vmax;
            c.width_in_blocks = (c.ds_w + 7) / 8;
            c.height_in_blocks = (c.ds_h + 7) / 8;
            const total = c.blocks_w * c.blocks_h * 64;
            c.coefs = try d.allocator.alloc(i16, total);
            @memset(c.coefs, 0);
        }
        d.frame = f;
    }

    fn parseDht(d: *Decoder, seg: []const u8) Error!void {
        var p: usize = 0;
        while (p < seg.len) {
            if (p + 17 > seg.len) return error.Invalid;
            const tc = seg[p] >> 4;
            const th = seg[p] & 15;
            if (tc > 1 or th > 3) return error.Invalid;
            const counts: *const [16]u8 = seg[p + 1 ..][0..16];
            var total: usize = 0;
            for (counts) |c| total += c;
            if (p + 17 + total > seg.len) return error.Invalid;
            const t = if (tc == 0) &d.dc[th] else &d.ac[th];
            if (!t.build(counts, seg[p + 17 .. p + 17 + total])) return error.Invalid;
            p += 17 + total;
        }
    }

    fn parseDqt(d: *Decoder, seg: []const u8) Error!void {
        var p: usize = 0;
        while (p < seg.len) {
            const pq = seg[p] >> 4;
            const tq = seg[p] & 15;
            if (tq > 3 or pq > 1) return error.Invalid;
            p += 1;
            const need: usize = if (pq == 0) 64 else 128;
            if (p + need > seg.len) return error.Invalid;
            for (0..64) |i| {
                const v: u16 = if (pq == 0) seg[p + i] else std.mem.readInt(u16, seg[p + 2 * i ..][0..2], .big);
                d.quant[tq][zigzag[i]] = v;
            }
            d.quant_set[tq] = true;
            p += need;
        }
    }

    fn app(d: *Decoder, marker: u8, seg: []const u8) void {
        if (marker == 0xe0 and seg.len >= 5 and std.mem.eql(u8, seg[0..5], "JFIF\x00")) d.info.jfif = true;
        if (marker == 0xee and seg.len >= 12 and std.mem.eql(u8, seg[0..5], "Adobe")) {
            d.info.adobe = true;
            d.info.adobe_transform = seg[11];
        }
        if (marker == 0xe1) {
            if (seg.len >= 5 and std.mem.eql(u8, seg[0..5], "Exif\x00")) {
                if (d.info.exif == null) d.info.exif = seg;
            } else if (seg.len >= 29 and std.mem.eql(u8, seg[0..29], "http://ns.adobe.com/xap/1.0/\x00")) {
                if (d.info.xmp == null) d.info.xmp = seg[29..];
            }
        }
    }

    /// next_marker: scan forward for FF xx (xx not 0 or FF). Returns the position of the FF and xx.
    fn nextMarker(d: *Decoder, from: usize) ?struct { at: usize, m: u8 } {
        var p = from;
        while (p + 1 < d.data.len) {
            if (d.data[p] == 0xff) {
                var q = p + 1;
                while (q < d.data.len and d.data[q] == 0xff) q += 1;
                if (q >= d.data.len) return null;
                if (d.data[q] != 0) return .{ .at = q - 1, .m = d.data[q] };
                p = q + 1;
            } else p += 1;
        }
        return null;
    }

    /// process_restart + read_restart_marker + jpeg_resync_to_restart
    fn processRestart(d: *Decoder, r: *BitReader, next_rst: *u8, pred: *[4]i32) Error!void {
        r.reset();
        if (r.marker == null) {
            if (r.eof) return error.Invalid;
            const nm = d.nextMarker(r.pos) orelse return error.Invalid;
            r.marker = nm.m;
            r.pos = nm.at;
        }
        const desired: u8 = next_rst.*;
        while (true) {
            const m = r.marker.?;
            if (m == 0xd0 + desired) {
                r.pos += 2;
                r.marker = null;
                break;
            }
            var action: u8 = 1;
            if (m < 0xc0) {
                action = 2;
            } else if (m < 0xd0 or m > 0xd7) {
                action = 3;
            } else {
                const k: u8 = m - 0xd0;
                if (k == ((desired + 1) & 7) or k == ((desired + 2) & 7)) {
                    action = 3;
                } else if (k == ((desired -% 1) & 7) or k == ((desired -% 2) & 7)) {
                    action = 2;
                } else action = 1;
            }
            switch (action) {
                1 => {
                    r.pos += 2;
                    r.marker = null;
                },
                2 => {
                    const nm = d.nextMarker(r.pos + 2) orelse {
                        r.eof = true;
                        return error.Invalid;
                    };
                    r.marker = nm.m;
                    r.pos = nm.at;
                    continue;
                },
                else => {},
            }
            break;
        }
        next_rst.* = (desired + 1) & 7;
        pred.* = @splat(0);
        d.eob_run = 0;
        if (r.marker == null) r.insufficient = false;
    }

    fn decodeScan(d: *Decoder, hdr: []const u8, start: usize) Error!usize {
        const f = &(d.frame orelse return error.Invalid);
        if (hdr.len < 1) return error.Invalid;
        const ns = hdr[0];
        if (ns < 1 or ns > 4 or hdr.len < 1 + 2 * @as(usize, ns) + 3) return error.Invalid;
        var scan_comps: [4]usize = undefined;
        var dc_sel: [4]u8 = undefined;
        var ac_sel: [4]u8 = undefined;
        for (0..ns) |i| {
            const cid = hdr[1 + 2 * i];
            const sel = hdr[2 + 2 * i];
            var found: ?usize = null;
            for (f.comps[0..f.n], 0..) |c, k| if (c.id == cid) {
                found = k;
                break;
            };
            scan_comps[i] = found orelse return error.Invalid;
            dc_sel[i] = sel >> 4;
            ac_sel[i] = sel & 15;
            if (dc_sel[i] > 3 or ac_sel[i] > 3) return error.Invalid;
        }
        const ss = hdr[1 + 2 * @as(usize, ns)];
        const se = hdr[2 + 2 * @as(usize, ns)];
        const ahl = hdr[3 + 2 * @as(usize, ns)];
        const ah: u5 = @intCast(ahl >> 4);
        const al: u5 = @intCast(ahl & 15);
        const prog_dc = f.progressive and ss == 0;
        if (f.progressive) {
            const bad_band = if (prog_dc) se != 0 else (se < ss or se >= 64 or ns != 1);
            if (bad_band or (ah != 0 and al != ah -% 1) or al > 13) return error.Invalid;
        }
        // tables the scan needs must exist; DC tables only hold symbols 0..15
        for (0..ns) |i| {
            const need_dc = !f.progressive or (prog_dc and ah == 0);
            const need_ac = !f.progressive or !prog_dc;
            if (need_dc) {
                const t = &d.dc[dc_sel[i]];
                if (!t.present) return error.Invalid;
                for (t.values[0..t.nsym]) |v| if (v > 15) return error.Invalid;
            }
            if (need_ac and !d.ac[ac_sel[i]].present) return error.Invalid;
        }
        var r: BitReader = .{ .data = d.data, .pos = start };
        var pred: [4]i32 = @splat(0);
        d.eob_run = 0;
        var restarts_left: usize = d.restart;
        var next_rst: u8 = 0;

        const interleaved = ns > 1;
        if (interleaved) { // libjpeg: D_MAX_BLOCKS_IN_MCU
            var blocks: usize = 0;
            for (0..ns) |si| blocks += @as(usize, f.comps[scan_comps[si]].h) * f.comps[scan_comps[si]].v;
            if (blocks > 10) return error.Invalid;
        }
        var mcus_across: usize = undefined;
        var mcus_down: usize = undefined;
        if (interleaved) {
            mcus_across = f.mcus_x;
            mcus_down = f.mcus_y;
        } else {
            const c = f.comps[scan_comps[0]];
            mcus_across = c.width_in_blocks;
            mcus_down = c.height_in_blocks;
        }
        var my: usize = 0;
        while (my < mcus_down) : (my += 1) {
            var mx: usize = 0;
            while (mx < mcus_across) : (mx += 1) {
                if (d.restart != 0) {
                    if (restarts_left == 0) {
                        try d.processRestart(&r, &next_rst, &pred);
                        restarts_left = d.restart;
                    }
                    restarts_left -= 1;
                }
                if (r.eof) return error.Invalid;
                if (r.insufficient) continue; // out of data: this MCU stays as it is (zeros)
                for (0..ns) |si| {
                    const ci = scan_comps[si];
                    const c = &f.comps[ci];
                    const blocks_h: usize = if (interleaved) c.v else 1;
                    const blocks_w: usize = if (interleaved) c.h else 1;
                    for (0..blocks_h) |by| for (0..blocks_w) |bx| {
                        const row = if (interleaved) my * c.v + by else my;
                        const col = if (interleaved) mx * c.h + bx else mx;
                        const block = c.coefs[(row * c.blocks_w + col) * 64 ..][0..64];
                        if (!f.progressive) {
                            try decodeBlockSeq(&r, block, &d.dc[dc_sel[si]], &d.ac[ac_sel[si]], &pred[si]);
                        } else if (prog_dc) {
                            if (ah == 0) {
                                const s = try r.decode(&d.dc[dc_sel[si]]);
                                const diff = if (s != 0) extend(r.get(@intCast(s)), @intCast(s)) else 0;
                                pred[si] +%= diff;
                                block[0] = @truncate(pred[si] << al);
                            } else {
                                if (r.get(1) != 0) block[0] |= @as(i16, 1) << @intCast(al);
                            }
                        } else if (ah == 0) {
                            try decodeAcFirst(d, &r, block, &d.ac[ac_sel[si]], ss, se, al);
                        } else {
                            try decodeAcRefine(d, &r, block, &d.ac[ac_sel[si]], ss, se, al);
                        }
                    };
                }
            }
        }
        if (r.eof) return error.Invalid;
        // the scan ends at the next marker (a pending one, or the first after the consumed bytes)
        if (r.marker != null) return r.pos;
        const nm = d.nextMarker(r.pos) orelse return error.Invalid;
        return nm.at;
    }

    fn decodeBlockSeq(r: *BitReader, block: *[64]i16, dc: *const HuffTable, ac: *const HuffTable, pred: *i32) Error!void {
        if (!dc.present or !ac.present) return error.Invalid;
        const s = try r.decode(dc);
        if (s > 16) return error.Invalid;
        const diff = if (s != 0) extend(r.get(@intCast(s)), @intCast(s)) else 0;
        pred.* +%= diff;
        block[0] = @truncate(pred.*);
        var k: usize = 1;
        while (k < 64) : (k += 1) {
            const rs = try r.decode(ac);
            const rr = rs >> 4;
            const ss: u5 = @intCast(rs & 15);
            if (ss != 0) {
                k += rr;
                const v = extend(r.get(ss), ss);
                block[zigzag[@min(k, 63)]] = @truncate(v);
            } else {
                if (rr != 15) break;
                k += 15;
            }
        }
    }

    fn decodeAcFirst(d: *Decoder, r: *BitReader, block: *[64]i16, ac: *const HuffTable, ss: u8, se: u8, al: u5) Error!void {
        if (!ac.present) return error.Invalid;
        if (d.eob_run > 0) {
            d.eob_run -= 1;
            return;
        }
        var k: usize = ss;
        while (k <= se) : (k += 1) {
            const rs = try r.decode(ac);
            var rr: u32 = rs >> 4;
            const s: u5 = @intCast(rs & 15);
            if (s != 0) {
                k += rr;
                const v = extend(r.get(s), s);
                block[zigzag[@min(k, 63)]] = @truncate(v * (@as(i32, 1) << al));
            } else {
                if (rr == 15) {
                    k += 15;
                } else {
                    d.eob_run = @as(u32, 1) << @intCast(rr);
                    if (rr != 0) {
                        d.eob_run += r.get(@intCast(rr));
                        rr = 0;
                    }
                    d.eob_run -= 1;
                    break;
                }
            }
        }
    }

    fn decodeAcRefine(d: *Decoder, r: *BitReader, block: *[64]i16, ac: *const HuffTable, ss: u8, se: u8, al: u5) Error!void {
        if (!ac.present) return error.Invalid;
        const p1: i16 = @as(i16, 1) << @intCast(al);
        const m1: i16 = -p1;
        var k: usize = ss;
        if (d.eob_run == 0) {
            while (k <= se) : (k += 1) {
                const rs = try r.decode(ac);
                var rr: i32 = rs >> 4;
                const s0: u32 = rs & 15;
                var s: i16 = 0;
                if (s0 != 0) {
                    s = if (r.get(1) != 0) p1 else m1;
                } else if (rr != 15) {
                    d.eob_run = @as(u32, 1) << @intCast(rr);
                    if (rr != 0) d.eob_run += r.get(@intCast(rr));
                    break;
                }
                while (k <= se) {
                    const pos = zigzag[k];
                    if (block[pos] != 0) {
                        if (r.get(1) != 0) {
                            if ((block[pos] & p1) == 0) {
                                if (block[pos] >= 0) block[pos] +%= p1 else block[pos] +%= m1;
                            }
                        }
                    } else {
                        rr -= 1;
                        if (rr < 0) break;
                    }
                    k += 1;
                }
                if (s != 0) {
                    if (k > 63) return error.Invalid;
                    block[zigzag[k]] = s;
                }
            }
        }
        if (d.eob_run > 0) {
            while (k <= se) : (k += 1) {
                const pos = zigzag[k];
                if (block[pos] != 0) {
                    if (r.get(1) != 0) {
                        if ((block[pos] & p1) == 0) {
                            if (block[pos] >= 0) block[pos] +%= p1 else block[pos] +%= m1;
                        }
                    }
                }
            }
            d.eob_run -= 1;
        }
    }
};

/// Fancy upsampling (jdsample.c) of component `c`'s plane (samples, stride `stride`) to w_up x h_up.
inline fn put(o: []u8, i: usize, v: u32) void {
    if (i < o.len) o[i] = @intCast(v);
}

fn upsample(a: Allocator, f: *const Frame, c: *const Component, plane: []const u8, stride: usize, out_w: usize, out_h: usize) Allocator.Error![]u8 {
    const hexp: usize = f.hmax / c.h;
    const vexp: usize = f.vmax / c.v;
    const out = try a.alloc(u8, out_w * out_h);
    errdefer a.free(out);
    const dw = c.ds_w;
    const dh = c.ds_h;
    if (hexp == 1 and vexp == 1) {
        for (0..out_h) |y| @memcpy(out[y * out_w ..][0..out_w], plane[y * stride ..][0..out_w]);
        return out;
    }
    const fancy_w = dw > 2;
    if (hexp == 2 and vexp == 1 and fancy_w) {
        for (0..out_h) |y| {
            const in = plane[y * stride ..][0..dw];
            const o = out[y * out_w ..][0..out_w];
            // out_w may be smaller than 2*dw by one (odd width): only the first out_w samples are kept
            var tmp: [2]u8 = undefined;
            _ = &tmp;
            var x: usize = 0;
            put(o, 0, in[0]);
            put(o, 1, (@as(u32, in[0]) * 3 + in[1] + 2) >> 2);
            x = 1;
            while (x < dw - 1) : (x += 1) {
                const v: u32 = @as(u32, in[x]) * 3;
                put(o, 2 * x, (v + in[x - 1] + 1) >> 2);
                put(o, 2 * x + 1, (v + in[x + 1] + 2) >> 2);
            }
            put(o, 2 * x, (@as(u32, in[x]) * 3 + in[x - 1] + 1) >> 2);
            put(o, 2 * x + 1, in[x]);
        }
        return out;
    }
    if (hexp == 2 and vexp == 2 and fancy_w) {
        for (0..out_h) |y| {
            const r = y / 2;
            const v: usize = y % 2;
            const r0 = @min(r, dh - 1);
            const r1: usize = if (v == 0) (if (r0 == 0) 0 else r0 - 1) else @min(r0 + 1, dh - 1);
            const in0 = plane[r0 * stride ..][0..dw];
            const in1 = plane[r1 * stride ..][0..dw];
            const o = out[y * out_w ..][0..out_w];
            var this: u32 = @as(u32, in0[0]) * 3 + in1[0];
            var next: u32 = @as(u32, in0[1]) * 3 + in1[1];
            put(o, 0, (this * 4 + 8) >> 4);
            put(o, 1, (this * 3 + next + 7) >> 4);
            var last = this;
            this = next;
            var x: usize = 1;
            while (x < dw - 1) : (x += 1) {
                next = @as(u32, in0[x + 1]) * 3 + in1[x + 1];
                put(o, 2 * x, (this * 3 + last + 8) >> 4);
                put(o, 2 * x + 1, (this * 3 + next + 7) >> 4);
                last = this;
                this = next;
            }
            put(o, 2 * x, (this * 3 + last + 8) >> 4);
            put(o, 2 * x + 1, (this * 4 + 7) >> 4);
        }
        return out;
    }
    if (hexp == 1 and vexp == 2) {
        for (0..out_h) |y| {
            const r = y / 2;
            const v: usize = y % 2;
            const r0 = @min(r, dh - 1);
            const r1: usize = if (v == 0) (if (r0 == 0) 0 else r0 - 1) else @min(r0 + 1, dh - 1);
            const bias: u32 = if (v == 0) 1 else 2;
            for (0..out_w) |x| {
                const s: u32 = @as(u32, plane[r0 * stride + x]) * 3 + plane[r1 * stride + x];
                out[y * out_w + x] = @intCast((s + bias) >> 2);
            }
        }
        return out;
    }
    // replication (h2v1/h2v2 narrower than 3 samples, h4v1, ...)
    for (0..out_h) |y| {
        const r = @min(y / vexp, dh -| 1);
        for (0..out_w) |x| out[y * out_w + x] = plane[r * stride + @min(x / hexp, dw -| 1)];
    }
    return out;
}

pub fn decode(a: Allocator, data: []const u8) Error!raster.Raster {
    if (data.len < 4 or data[0] != 0xff or data[1] != 0xd8) return error.Invalid;
    var d: Decoder = .{ .allocator = a, .data = data };
    defer d.deinit();
    var p: usize = 2;
    var seen_scan = false;
    while (true) {
        // find the next marker
        while (p < data.len and data[p] != 0xff) p += 1;
        while (p < data.len and data[p] == 0xff) p += 1;
        if (p >= data.len) return error.Invalid; // no EOI: the file is truncated
        const m = data[p];
        p += 1;
        if (m == 0 or m == 0x01 or (m >= 0xd0 and m <= 0xd8)) continue;
        if (m == 0xd9) break;
        const len: usize = try d.u16at(p);
        if (len < 2 or p + len > data.len) return error.Invalid;
        const seg = data[p + 2 .. p + len];
        switch (m) {
            0xc0, 0xc1 => try d.parseSof(seg, false),
            0xc2 => try d.parseSof(seg, true),
            0xc4 => try d.parseDht(seg),
            0xdb => try d.parseDqt(seg),
            0xdd => {
                if (seg.len < 2) return error.Invalid;
                d.restart = std.mem.readInt(u16, seg[0..2], .big);
            },
            0xda => {
                if (d.scans >= 600) return error.Invalid; // a scan costs a pass over the image: bound hostile scan lists
                const np = try d.decodeScan(seg, p + len);
                seen_scan = true;
                d.scans += 1;
                p = np;
                continue;
            },
            0xe0...0xef => d.app(m, seg),
            0xc3, 0xc5...0xc7, 0xc9...0xcb, 0xcd...0xcf => return error.Invalid, // lossless / arithmetic / hierarchical
            else => {},
        }
        p += len;
    }
    if (!seen_scan) return error.Invalid;
    const f = d.frame orelse return error.Invalid;
    for (f.comps[0..f.n]) |c| if (!d.quant_set[c.tq]) return error.Invalid;

    // IDCT every component into a plane
    var planes: [4][]u8 = undefined;
    var strides: [4]usize = undefined;
    var have: usize = 0;
    defer for (planes[0..have]) |pl| a.free(pl);
    for (f.comps[0..f.n], 0..) |*c, ci| {
        const stride = c.blocks_w * 8;
        const plane = try a.alloc(u8, stride * c.blocks_h * 8);
        planes[ci] = plane;
        strides[ci] = stride;
        have += 1;
        const q = &d.quant[c.tq];
        for (0..c.blocks_h) |by| for (0..c.blocks_w) |bx| {
            const blk: *const [64]i16 = c.coefs[(by * c.blocks_w + bx) * 64 ..][0..64];
            idctIslow(blk, q, plane[(by * 8) * stride + bx * 8 ..], stride);
        };
    }
    const w = f.w;
    const h = f.h;
    // upsample to full resolution
    var full: [4][]u8 = undefined;
    var nfull: usize = 0;
    defer for (full[0..nfull]) |pl| a.free(pl);
    for (f.comps[0..f.n], 0..) |*c, ci| {
        if (f.hmax % c.h != 0 or f.vmax % c.v != 0) return error.Invalid; // fractional sampling ratio
        full[ci] = try upsample(a, &f, c, planes[ci], strides[ci], w, h);
        nfull += 1;
    }

    const mode: raster.Mode = switch (f.n) {
        1 => .l,
        3 => .rgb,
        else => .cmyk,
    };
    const out = try a.alloc(u8, w * h * mode.bytes());
    errdefer a.free(out);
    const tab = colorTables();
    switch (f.n) {
        1 => @memcpy(out, full[0][0 .. w * h]),
        3 => {
            const ycc = blk: {
                if (d.info.jfif) break :blk true;
                if (d.info.adobe) break :blk d.info.adobe_transform != 0;
                const ids = .{ f.comps[0].id, f.comps[1].id, f.comps[2].id };
                if (ids[0] == 1 and ids[1] == 2 and ids[2] == 3) break :blk true;
                if (ids[0] == 'R' and ids[1] == 'G' and ids[2] == 'B') break :blk false;
                break :blk true;
            };
            for (0..w * h) |i| {
                const y: i32 = full[0][i];
                const cb = full[1][i];
                const cr = full[2][i];
                if (ycc) {
                    out[i * 3] = clamp8(y + tab.cr_r[cr]);
                    out[i * 3 + 1] = clamp8(y + ((tab.cb_g[cb] + tab.cr_g[cr]) >> 16));
                    out[i * 3 + 2] = clamp8(y + tab.cb_b[cb]);
                } else {
                    out[i * 3] = full[0][i];
                    out[i * 3 + 1] = cb;
                    out[i * 3 + 2] = cr;
                }
            }
        },
        else => {
            const ycck = d.info.adobe and d.info.adobe_transform != 0;
            for (0..w * h) |i| {
                const k = full[3][i];
                var c4: [3]u8 = undefined;
                if (ycck) {
                    const y: i32 = full[0][i];
                    const cb = full[1][i];
                    const cr = full[2][i];
                    c4[0] = clamp8(255 - (y + tab.cr_r[cr]));
                    c4[1] = clamp8(255 - (y + ((tab.cb_g[cb] + tab.cr_g[cr]) >> 16)));
                    c4[2] = clamp8(255 - (y + tab.cb_b[cb]));
                } else {
                    c4 = .{ full[0][i], full[1][i], full[2][i] };
                }
                // Pillow reads these as "CMYK;I": every byte inverted
                out[i * 4] = 255 - c4[0];
                out[i * 4 + 1] = 255 - c4[1];
                out[i * 4 + 2] = 255 - c4[2];
                out[i * 4 + 3] = 255 - k;
            }
        },
    }
    var r: raster.Raster = .{ .w = @intCast(w), .h = @intCast(h), .mode = mode, .data = out, .allocator = a };
    errdefer r.deinit();
    if (d.info.exif) |e| r.exif = try a.dupe(u8, e);
    if (d.info.xmp) |x| r.xmp = try a.dupe(u8, x);
    return r;
}
