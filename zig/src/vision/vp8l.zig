//! WebP lossless (VP8L) decoder: prefix codes, LZ77 with the colour cache, the four inverse transforms. Output is
//! ARGB (0xAARRGGBB) per pixel, exactly what libwebp reconstructs.
const std = @import("std");
const tables = @import("vp8_tables.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{Invalid} || Allocator.Error;

const BitReader = struct {
    data: []const u8,
    pos: usize = 0, // next byte to load
    val: u64 = 0,
    nbits: u7 = 0,
    over: u32 = 0, // zero bytes supplied beyond the end of the data

    fn init(data: []const u8) BitReader {
        return .{ .data = data };
    }

    fn refill(r: *BitReader) void {
        while (r.nbits <= 56) {
            if (r.pos < r.data.len) {
                r.val |= @as(u64, r.data[r.pos]) << @intCast(r.nbits);
                r.pos += 1;
            } else r.over += 1;
            r.nbits += 8;
        }
    }

    inline fn read(r: *BitReader, n: u6) u32 {
        if (n == 0) return 0;
        if (r.nbits < n) r.refill();
        const v: u32 = @intCast(r.val & ((@as(u64, 1) << n) - 1));
        r.val >>= n;
        r.nbits -= n;
        return v;
    }

    /// whether more bits were consumed than the data holds
    fn overrun(r: *const BitReader) bool {
        return r.over * 8 > r.nbits;
    }
};

const max_code_len = 15;

/// One prefix code. `table` is indexed by the next `bits` bits (LSB first); entries are (len << 16) | symbol.
const Huffman = struct {
    table: []u32 = &.{},
    bits: u5 = 0,
    single: bool = false,
    symbol: u16 = 0,

    fn deinit(h: *Huffman, a: Allocator) void {
        if (h.table.len != 0) a.free(h.table);
        h.table = &.{};
    }

    /// Build from code lengths. A code with one symbol uses zero bits; otherwise the lengths must be complete.
    fn build(h: *Huffman, a: Allocator, lengths: []const u8) Error!void {
        var count: [(max_code_len + 1)]u32 = @splat(0);
        var used: usize = 0;
        var last: usize = 0;
        for (lengths, 0..) |l, s| {
            if (l > max_code_len) return error.Invalid;
            if (l != 0) {
                count[l] += 1;
                used += 1;
                last = s;
            }
        }
        if (used == 0) return error.Invalid;
        if (used == 1) {
            h.single = true;
            h.symbol = @intCast(last);
            h.bits = 0;
            return;
        }
        var left: i64 = 1;
        var maxlen: u5 = 0;
        for (1..max_code_len + 1) |l| {
            left = left * 2 - count[l];
            if (left < 0) return error.Invalid;
            if (count[l] != 0) maxlen = @intCast(l);
        }
        if (left != 0) return error.Invalid; // incomplete code
        var next: [(max_code_len + 2)]u32 = @splat(0);
        var code: u32 = 0;
        for (1..max_code_len + 1) |l| {
            code = (code + count[l - 1]) << 1;
            next[l] = code;
        }
        const size = @as(usize, 1) << maxlen;
        h.table = try a.alloc(u32, size);
        h.bits = maxlen;
        @memset(h.table, 0);
        for (lengths, 0..) |l, s| {
            if (l == 0) continue;
            const c = next[l];
            next[l] += 1;
            var rev: u32 = 0;
            var cc = c;
            for (0..l) |_| {
                rev = (rev << 1) | (cc & 1);
                cc >>= 1;
            }
            var i: usize = rev;
            const step = @as(usize, 1) << @intCast(l);
            while (i < size) : (i += step) h.table[i] = (@as(u32, l) << 16) | @as(u32, @intCast(s));
        }
    }

    inline fn decode(h: *const Huffman, r: *BitReader) u32 {
        if (h.single) return h.symbol;
        if (r.nbits < 15) r.refill();
        const e = h.table[@intCast(r.val & ((@as(u64, 1) << h.bits) - 1))];
        const l: u6 = @intCast(e >> 16);
        r.val >>= l;
        r.nbits -= l;
        return e & 0xffff;
    }
};

const code_length_order = [19]u8{ 17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

fn readCode(a: Allocator, r: *BitReader, alphabet: usize, h: *Huffman) Error!void {
    var lengths_buf: [256 + 24 + 2048]u8 = undefined;
    @memset(&lengths_buf, 0);
    const lengths = lengths_buf[0..alphabet];
    if (r.read(1) == 1) {
        // simple code of one or two symbols
        const num = r.read(1) + 1;
        const first_8 = r.read(1) == 1;
        const s0 = r.read(if (first_8) 8 else 1);
        lengths_buf[s0] = 1;
        if (num == 2) {
            const s1 = r.read(8);
            lengths_buf[s1] = 1;
        }
    } else {
        var cl: [19]u8 = @splat(0);
        const num_cl = r.read(4) + 4;
        if (num_cl > 19) return error.Invalid;
        for (0..num_cl) |i| cl[code_length_order[i]] = @intCast(r.read(3));
        var clh: Huffman = .{};
        defer clh.deinit(a);
        try clh.build(a, &cl);
        var max_symbol: usize = alphabet;
        if (r.read(1) == 1) {
            const length_nbits: u6 = @intCast(2 + 2 * r.read(3));
            max_symbol = 2 + r.read(length_nbits);
            if (max_symbol > alphabet) return error.Invalid;
        }
        var sym: usize = 0;
        var prev: u8 = 8;
        while (sym < alphabet) {
            if (max_symbol == 0) break;
            max_symbol -= 1;
            const code_len = clh.decode(r);
            if (r.overrun()) return error.Invalid;
            if (code_len < 16) {
                lengths[sym] = @intCast(code_len);
                sym += 1;
                if (code_len != 0) prev = @intCast(code_len);
            } else {
                const slot = code_len - 16;
                const extra_bits: u6 = switch (slot) {
                    0 => 2,
                    1 => 3,
                    else => 7,
                };
                const offset: usize = switch (slot) {
                    0 => 3,
                    1 => 3,
                    else => 11,
                };
                const repeat = offset + r.read(extra_bits);
                if (sym + repeat > alphabet) return error.Invalid;
                const v: u8 = if (slot == 0) prev else 0;
                @memset(lengths[sym .. sym + repeat], v);
                sym += repeat;
            }
        }
    }
    if (r.overrun()) return error.Invalid;
    try h.build(a, lengths);
}

const HTreeGroup = struct { codes: [5]Huffman = .{ .{}, .{}, .{}, .{}, .{} } };

fn deinitGroups(a: Allocator, groups: []HTreeGroup) void {
    for (groups) |*g| for (&g.codes) |*c| c.deinit(a);
    a.free(groups);
}

const Transform = struct {
    kind: u2, // 0 predictor, 1 cross colour, 2 subtract green, 3 colour indexing
    bits: u5 = 0,
    xsize: usize = 0,
    ysize: usize = 0,
    data: []u32 = &.{},
};

fn subSampleSize(size: usize, bits: u5) usize {
    return (size + (@as(usize, 1) << bits) - 1) >> bits;
}

const prefix_extra = struct {
    fn value(r: *BitReader, symbol: u32) u32 {
        if (symbol < 4) return symbol + 1;
        const extra_bits: u6 = @intCast((symbol - 2) >> 1);
        const offset = (2 + (symbol & 1)) << @intCast(extra_bits);
        return offset + r.read(extra_bits) + 1;
    }
};

fn hashCache(argb: u32, bits: u5) u32 {
    return (argb *% 0x1e35a7bd) >> @intCast(32 - @as(u6, bits));
}

/// Decode an entropy-coded image of xsize x ysize ARGB pixels (the main image when `main`, else a sub-image).
fn decodeImage(a: Allocator, r: *BitReader, xsize: usize, ysize: usize, main: bool) Error![]u32 {
    var color_cache_bits: u5 = 0;
    if (r.read(1) == 1) {
        const b = r.read(4);
        if (b < 1 or b > 11) return error.Invalid;
        color_cache_bits = @intCast(b);
    }
    var meta_bits: u5 = 0;
    var meta: []u32 = &.{};
    defer if (meta.len != 0) a.free(meta);
    var num_groups: usize = 1;
    if (main and r.read(1) == 1) {
        meta_bits = @intCast(r.read(3) + 2);
        const mw = subSampleSize(xsize, meta_bits);
        const mh = subSampleSize(ysize, meta_bits);
        meta = try decodeImage(a, r, mw, mh, false);
        for (meta) |*p| {
            p.* = (p.* >> 8) & 0xffff;
            num_groups = @max(num_groups, @as(usize, p.*) + 1);
        }
    }
    const cache_size: usize = if (color_cache_bits != 0) (@as(usize, 1) << color_cache_bits) else 0;
    const groups = try a.alloc(HTreeGroup, num_groups);
    for (groups) |*g| g.* = .{};
    defer deinitGroups(a, groups);
    for (groups) |*g| {
        for (0..5) |j| {
            const alphabet: usize = switch (j) {
                0 => 256 + 24 + cache_size,
                4 => 40,
                else => 256,
            };
            try readCode(a, r, alphabet, &g.codes[j]);
        }
    }
    const total = xsize * ysize;
    const out = try a.alloc(u32, total);
    errdefer a.free(out);
    var cache: []u32 = &.{};
    defer if (cache.len != 0) a.free(cache);
    if (cache_size != 0) {
        cache = try a.alloc(u32, cache_size);
        @memset(cache, 0);
    }
    var pos: usize = 0;
    var x: usize = 0;
    var y: usize = 0;
    var g: *const HTreeGroup = &groups[0];
    const mask: usize = if (meta_bits != 0) (@as(usize, 1) << meta_bits) - 1 else 0;
    const meta_w = if (meta_bits != 0) subSampleSize(xsize, meta_bits) else 0;
    if (meta_bits != 0) g = &groups[meta[0]];
    var last_cached: usize = 0;
    while (pos < total) {
        if (meta_bits != 0 and (x & mask) == 0) g = &groups[meta[(y >> meta_bits) * meta_w + (x >> meta_bits)]];
        const s = g.codes[0].decode(r);
        if (r.overrun()) return error.Invalid;
        if (s < 256) {
            const red = g.codes[1].decode(r);
            const blue = g.codes[2].decode(r);
            const alpha = g.codes[3].decode(r);
            if (r.overrun()) return error.Invalid;
            out[pos] = (alpha << 24) | (red << 16) | (s << 8) | blue;
            pos += 1;
            x += 1;
            if (x >= xsize) {
                x = 0;
                y += 1;
            }
        } else if (s < 256 + 24) {
            const length = prefix_extra.value(r, s - 256);
            const dsym = g.codes[4].decode(r);
            const dcode = prefix_extra.value(r, dsym);
            if (r.overrun()) return error.Invalid;
            var dist: usize = undefined;
            if (dcode > 120) {
                dist = dcode - 120;
            } else {
                const c = tables.code_to_plane[dcode - 1];
                const yo: i64 = c >> 4;
                const xo: i64 = 8 - @as(i64, c & 15);
                const dd = yo * @as(i64, @intCast(xsize)) + xo;
                dist = if (dd >= 1) @intCast(dd) else 1;
            }
            if (dist > pos or pos + length > total) return error.Invalid;
            var k: usize = 0;
            while (k < length) : (k += 1) out[pos + k] = out[pos + k - dist];
            pos += length;
            x += length;
            while (x >= xsize) {
                x -= xsize;
                y += 1;
            }
            // the group for the next pixel is looked up at the loop head only when x is on a block boundary;
            // a copy can end mid-block, so refresh it here
            if (meta_bits != 0 and pos < total and (x & mask) != 0) g = &groups[meta[(y >> meta_bits) * meta_w + (x >> meta_bits)]];
        } else {
            const idx = s - (256 + 24);
            if (cache_size == 0 or idx >= cache_size) return error.Invalid;
            // insert everything decoded so far, then fetch
            while (last_cached < pos) : (last_cached += 1) cache[hashCache(out[last_cached], color_cache_bits)] = out[last_cached];
            out[pos] = cache[idx];
            pos += 1;
            x += 1;
            if (x >= xsize) {
                x = 0;
                y += 1;
            }
        }
    }
    if (r.overrun()) return error.Invalid;
    return out;
}

inline fn avg2(a: u32, b: u32) u32 {
    return (((a ^ b) & 0xfefefefe) >> 1) + (a & b);
}

inline fn channel(v: u32, shift: u5) i32 {
    return @intCast((v >> shift) & 0xff);
}

inline fn clip255(v: i32) u32 {
    return @intCast(std.math.clamp(v, 0, 255));
}

fn selectPred(l: u32, t: u32, tl: u32) u32 {
    var pl: i32 = 0;
    var pt: i32 = 0;
    inline for (.{ 24, 16, 8, 0 }) |sh| {
        const s: u5 = sh;
        const p = channel(l, s) + channel(t, s) - channel(tl, s);
        pl += @intCast(@abs(p - channel(l, s)));
        pt += @intCast(@abs(p - channel(t, s)));
    }
    return if (pl < pt) l else t;
}

fn clampAddSubFull(a_: u32, b: u32, c: u32) u32 {
    var out: u32 = 0;
    inline for (.{ 24, 16, 8, 0 }) |sh| {
        const s: u5 = sh;
        out |= clip255(channel(a_, s) + channel(b, s) - channel(c, s)) << s;
    }
    return out;
}

fn clampAddSubHalf(a_: u32, b: u32) u32 {
    var out: u32 = 0;
    inline for (.{ 24, 16, 8, 0 }) |sh| {
        const s: u5 = sh;
        const av = channel(a_, s);
        out |= clip255(av + @divTrunc(av - channel(b, s), 2)) << s;
    }
    return out;
}

fn predict(mode: u32, l: u32, t: u32, tr: u32, tl: u32) u32 {
    return switch (mode) {
        0 => 0xff000000,
        1 => l,
        2 => t,
        3 => tr,
        4 => tl,
        5 => avg2(avg2(l, tr), t),
        6 => avg2(l, tl),
        7 => avg2(l, t),
        8 => avg2(tl, t),
        9 => avg2(t, tr),
        10 => avg2(avg2(l, tl), avg2(t, tr)),
        11 => selectPred(l, t, tl),
        12 => clampAddSubFull(l, t, tl),
        13 => clampAddSubHalf(avg2(l, t), tl),
        else => 0xff000000, // modes 14 and 15 behave as black in libwebp
    };
}

inline fn addPixels(a_: u32, b: u32) u32 {
    const alpha_green = (a_ & 0xff00ff00) +% (b & 0xff00ff00);
    const red_blue = (a_ & 0x00ff00ff) +% (b & 0x00ff00ff);
    return (alpha_green & 0xff00ff00) | (red_blue & 0x00ff00ff);
}

fn inversePredictor(t: *const Transform, px: []u32, width: usize, height: usize) void {
    const bw = subSampleSize(width, t.bits);
    // first pixel: black prediction
    px[0] = addPixels(px[0], 0xff000000);
    for (1..width) |x| px[x] = addPixels(px[x], px[x - 1]);
    for (1..height) |y| {
        const row = px[y * width ..][0..width];
        const up = px[(y - 1) * width ..][0..width];
        row[0] = addPixels(row[0], up[0]);
        var x: usize = 1;
        while (x < width) {
            const mode = (t.data[(y >> t.bits) * bw + (x >> t.bits)] >> 8) & 0xf;
            const end = @min(width, ((x >> t.bits) + 1) << t.bits);
            while (x < end) : (x += 1) {
                const tr: u32 = if (x + 1 < width) up[x + 1] else row[0]; // the rightmost pixel's TR is the row's first pixel
                row[x] = addPixels(row[x], predict(mode, row[x - 1], up[x], tr, up[x - 1]));
            }
        }
    }
}

inline fn colorDelta(t: u32, c: u32) i32 {
    const ts: i32 = @as(i8, @bitCast(@as(u8, @truncate(t))));
    const cs: i32 = @as(i8, @bitCast(@as(u8, @truncate(c))));
    return (ts * cs) >> 5;
}

fn inverseCrossColor(t: *const Transform, px: []u32, width: usize, height: usize) void {
    const bw = subSampleSize(width, t.bits);
    for (0..height) |y| {
        for (0..width) |x| {
            const m = t.data[(y >> t.bits) * bw + (x >> t.bits)];
            const green_to_red = m & 0xff;
            const green_to_blue = (m >> 8) & 0xff;
            const red_to_blue = (m >> 16) & 0xff;
            const argb = px[y * width + x];
            const green = (argb >> 8) & 0xff;
            var red: i32 = @intCast((argb >> 16) & 0xff);
            var blue: i32 = @intCast(argb & 0xff);
            red += colorDelta(green_to_red, green);
            red &= 0xff;
            blue += colorDelta(green_to_blue, green);
            blue += colorDelta(red_to_blue, @intCast(red));
            blue &= 0xff;
            px[y * width + x] = (argb & 0xff00ff00) | (@as(u32, @intCast(red)) << 16) | @as(u32, @intCast(blue));
        }
    }
}

fn inverseSubtractGreen(px: []u32) void {
    for (px) |*p| {
        const green = (p.* >> 8) & 0xff;
        const rb = ((p.* & 0x00ff00ff) +% ((green << 16) | green)) & 0x00ff00ff;
        p.* = (p.* & 0xff00ff00) | rb;
    }
}

/// Decode a VP8L bit stream. With `header` the 5-byte header is read first (a VP8L chunk); otherwise the
/// dimensions come from the caller (the alpha plane of a lossy file). Returns ARGB pixels.
pub fn decode(a: Allocator, data: []const u8, with_header: bool, width_in: usize, height_in: usize) Error!struct { argb: []u32, width: usize, height: usize } {
    var r = BitReader.init(data);
    var width = width_in;
    var height = height_in;
    if (with_header) {
        if (data.len < 5 or data[0] != 0x2f) return error.Invalid;
        _ = r.read(8);
        width = r.read(14) + 1;
        height = r.read(14) + 1;
        _ = r.read(1); // alpha_is_used
        if (r.read(3) != 0) return error.Invalid; // version
    }
    var transforms: [4]Transform = undefined;
    var ntrans: usize = 0;
    var seen: [4]bool = @splat(false);
    errdefer for (transforms[0..ntrans]) |t| if (t.data.len != 0) a.free(t.data);
    var xsize = width;
    while (r.read(1) == 1) {
        const kind: u2 = @intCast(r.read(2));
        if (seen[kind]) return error.Invalid;
        seen[kind] = true;
        var t: Transform = .{ .kind = kind, .xsize = xsize, .ysize = height };
        switch (kind) {
            0, 1 => {
                t.bits = @intCast(r.read(3) + 2);
                t.data = try decodeImage(a, &r, subSampleSize(xsize, t.bits), subSampleSize(height, t.bits), false);
            },
            2 => {},
            3 => {
                const num_colors = r.read(8) + 1;
                const bits: u5 = if (num_colors > 16) 0 else if (num_colors > 4) 1 else if (num_colors > 2) 2 else 3;
                const pal = try decodeImage(a, &r, num_colors, 1, false);
                // delta decode
                var i: usize = 1;
                while (i < num_colors) : (i += 1) pal[i] = addPixels(pal[i], pal[i - 1]);
                // expand to a 2^(8 - ...) sized table so out-of-range indexes read transparent black
                const full = try a.alloc(u32, @as(usize, 1) << @intCast(@as(u32, 8) >> bits));
                @memset(full, 0);
                @memcpy(full[0..num_colors], pal);
                a.free(pal);
                t.data = full;
                t.bits = bits;
                xsize = subSampleSize(xsize, bits);
            },
        }
        transforms[ntrans] = t;
        ntrans += 1;
    }
    const argb_small = try decodeImage(a, &r, xsize, height, true);
    errdefer a.free(argb_small);
    // invert the transforms, last read first
    var cur = argb_small;
    var cur_w = xsize;
    var ti = ntrans;
    while (ti > 0) {
        ti -= 1;
        const t = &transforms[ti];
        switch (t.kind) {
            0 => inversePredictor(t, cur, t.xsize, height),
            1 => inverseCrossColor(t, cur, t.xsize, height),
            2 => inverseSubtractGreen(cur),
            3 => {
                const full_w = t.xsize;
                const out = try a.alloc(u32, full_w * height);
                const bits = t.bits;
                if (bits == 0) {
                    for (cur, out) |p, *o| o.* = t.data[(p >> 8) & 0xff];
                } else {
                    const per: usize = @as(usize, 1) << bits;
                    const depth: u5 = @intCast(@as(u32, 8) >> bits);
                    const m: u32 = (@as(u32, 1) << depth) - 1;
                    for (0..height) |y| for (0..full_w) |x| {
                        const p = cur[y * cur_w + x / per];
                        const idx = ((p >> 8) >> @intCast((x % per) * depth)) & m;
                        out[y * full_w + x] = t.data[idx];
                    };
                }
                a.free(cur);
                cur = out;
                cur_w = full_w;
            },
        }
    }
    for (transforms[0..ntrans]) |t| if (t.data.len != 0) a.free(t.data);
    return .{ .argb = cur, .width = width, .height = height };
}
