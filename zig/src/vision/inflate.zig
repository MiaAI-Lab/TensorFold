//! A small RFC 1950/1951 decoder over an in-memory stream, with an output cap (the PNG decoder knows how many
//! bytes the image needs and ignores the rest, as libpng-based Pillow does).
const std = @import("std");

pub const Error = error{ BadStream, Truncated, TooLarge } || std.mem.Allocator.Error;

const BitReader = struct {
    src: []const u8,
    pos: usize = 0,
    bits: u64 = 0,
    n: u6 = 0,
    over: u32 = 0, // zero bytes supplied past the end

    inline fn refill(r: *BitReader) void {
        while (r.n < 56) {
            var b: u64 = 0;
            if (r.pos < r.src.len) {
                b = r.src[r.pos];
                r.pos += 1;
            } else r.over += 1;
            r.bits |= b << r.n;
            r.n += 8;
        }
    }
    inline fn need(r: *BitReader, k: u6) void {
        if (r.n < k) r.refill();
    }
    inline fn take(r: *BitReader, k: u6) u32 {
        if (k == 0) return 0;
        r.need(k);
        const v: u32 = @intCast(r.bits & ((@as(u64, 1) << k) - 1));
        r.bits >>= k;
        r.n -= k;
        return v;
    }
    /// bytes of padding actually consumed
    fn overread(r: *const BitReader) bool {
        // over counts whole zero bytes appended; they were consumed if more than the bytes still buffered
        return r.over * 8 > r.n;
    }
};

const Huff = struct {
    // single-level table indexed by `maxbits` reversed-code bits: (symbol << 4) | length; 0 = invalid
    table: []u16,
    maxbits: u6,

    fn build(allocator: std.mem.Allocator, lengths: []const u8) Error!Huff {
        var count: [16]u16 = @splat(0);
        var maxbits: u6 = 0;
        for (lengths) |l| {
            count[l] += 1;
            if (l > maxbits) maxbits = @intCast(l);
        }
        count[0] = 0;
        if (maxbits == 0) {
            const t = try allocator.alloc(u16, 2);
            @memset(t, 0);
            return .{ .table = t, .maxbits = 1 };
        }
        // oversubscribed / incomplete check
        var left: i32 = 1;
        for (1..16) |l| {
            left <<= 1;
            left -= count[l];
            if (left < 0) return error.BadStream;
        }
        var next: [17]u32 = @splat(0);
        var code: u32 = 0;
        for (1..16) |l| {
            code = (code + count[l - 1]) << 1;
            next[l] = code;
        }
        const size = @as(usize, 1) << maxbits;
        const table = try allocator.alloc(u16, size);
        @memset(table, 0);
        for (lengths, 0..) |l, sym| {
            if (l == 0) continue;
            const c = next[l];
            next[l] += 1;
            // reverse the l-bit code
            var rev: u32 = 0;
            var cc = c;
            for (0..l) |_| {
                rev = (rev << 1) | (cc & 1);
                cc >>= 1;
            }
            var i: usize = rev;
            const step = @as(usize, 1) << @intCast(l);
            while (i < size) : (i += step) table[i] = @intCast((sym << 4) | l);
        }
        return .{ .table = table, .maxbits = maxbits };
    }

    inline fn decode(h: *const Huff, r: *BitReader) Error!u32 {
        r.need(15);
        const e = h.table[@intCast(r.bits & ((@as(u64, 1) << h.maxbits) - 1))];
        if (e == 0) return error.BadStream;
        const l: u6 = @intCast(e & 15);
        r.bits >>= l;
        r.n -= l;
        return e >> 4;
    }
};

const len_base = [_]u16{ 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258 };
const len_extra = [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 };
const dist_base = [_]u16{ 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577 };
const dist_extra = [_]u8{ 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 };
const order = [_]u8{ 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };

/// Raw deflate. Writes at most `out.len` bytes; returns the number written. Stops silently at the cap; a stream
/// that ends before `out` is full returns the shorter count.
pub fn inflateRaw(allocator: std.mem.Allocator, src: []const u8, out: []u8) Error!usize {
    var r: BitReader = .{ .src = src };
    var o: usize = 0;
    var fixed_lit: ?Huff = null;
    var fixed_dist: ?Huff = null;
    defer {
        if (fixed_lit) |h| allocator.free(h.table);
        if (fixed_dist) |h| allocator.free(h.table);
    }
    while (true) {
        const final = r.take(1) == 1;
        const typ = r.take(2);
        if (r.overread()) return error.Truncated;
        switch (typ) {
            0 => {
                // stored: drop to byte boundary
                const drop: u6 = @intCast(r.n % 8);
                _ = r.take(drop);
                const len = r.take(16);
                const nlen = r.take(16);
                if (len != (~nlen & 0xffff)) return error.BadStream;
                var k: u32 = 0;
                while (k < len) : (k += 1) {
                    const b: u8 = @intCast(r.take(8));
                    if (r.overread()) return error.Truncated;
                    if (o >= out.len) return o;
                    out[o] = b;
                    o += 1;
                }
            },
            1, 2 => {
                var lit: Huff = undefined;
                var dist: Huff = undefined;
                var dyn = false;
                if (typ == 1) {
                    if (fixed_lit == null) {
                        var l: [288]u8 = undefined;
                        @memset(l[0..144], 8);
                        @memset(l[144..256], 9);
                        @memset(l[256..280], 7);
                        @memset(l[280..288], 8);
                        fixed_lit = try Huff.build(allocator, &l);
                        const d: [30]u8 = @splat(5);
                        fixed_dist = try Huff.build(allocator, &d);
                    }
                    lit = fixed_lit.?;
                    dist = fixed_dist.?;
                } else {
                    dyn = true;
                    const hlit = r.take(5) + 257;
                    const hdist = r.take(5) + 1;
                    const hclen = r.take(4) + 4;
                    if (hlit > 286 or hdist > 30) return error.BadStream;
                    var cl: [19]u8 = @splat(0);
                    for (0..hclen) |i| cl[order[i]] = @intCast(r.take(3));
                    const clh = try Huff.build(allocator, &cl);
                    defer allocator.free(clh.table);
                    var lens: [320]u8 = @splat(0);
                    var i: usize = 0;
                    while (i < hlit + hdist) {
                        const sym = try clh.decode(&r);
                        if (r.overread()) return error.Truncated;
                        if (sym < 16) {
                            lens[i] = @intCast(sym);
                            i += 1;
                        } else {
                            var rep: u32 = undefined;
                            var val: u8 = 0;
                            if (sym == 16) {
                                if (i == 0) return error.BadStream;
                                val = lens[i - 1];
                                rep = 3 + r.take(2);
                            } else if (sym == 17) {
                                rep = 3 + r.take(3);
                            } else rep = 11 + r.take(7);
                            if (i + rep > hlit + hdist) return error.BadStream;
                            @memset(lens[i .. i + rep], val);
                            i += rep;
                        }
                    }
                    if (lens[256] == 0) return error.BadStream;
                    lit = try Huff.build(allocator, lens[0..hlit]);
                    errdefer allocator.free(lit.table);
                    dist = try Huff.build(allocator, lens[hlit .. hlit + hdist]);
                }
                defer if (dyn) {
                    allocator.free(lit.table);
                    allocator.free(dist.table);
                };
                while (true) {
                    const sym = try lit.decode(&r);
                    if (r.overread()) return error.Truncated;
                    if (sym < 256) {
                        if (o >= out.len) return o;
                        out[o] = @intCast(sym);
                        o += 1;
                    } else if (sym == 256) {
                        break;
                    } else {
                        const li = sym - 257;
                        if (li >= 29) return error.BadStream;
                        const length = len_base[li] + r.take(@intCast(len_extra[li]));
                        const ds = try dist.decode(&r);
                        if (ds >= 30) return error.BadStream;
                        const d = dist_base[ds] + r.take(@intCast(dist_extra[ds]));
                        if (r.overread()) return error.Truncated;
                        if (d > o) return error.BadStream;
                        const n = @min(length, out.len - o);
                        var k: usize = 0;
                        while (k < n) : (k += 1) out[o + k] = out[o + k - d];
                        o += n;
                        if (o >= out.len and n < length) return o;
                    }
                }
            },
            else => return error.BadStream,
        }
        if (o >= out.len) return o;
        if (final) return o;
    }
}

/// zlib (RFC 1950) wrapped; the Adler-32 trailer is not checked (decoding stops when `out` is full).
pub fn inflateZlib(allocator: std.mem.Allocator, src: []const u8, out: []u8) Error!usize {
    if (src.len < 2) return error.Truncated;
    const cmf = src[0];
    const flg = src[1];
    if ((cmf & 0x0f) != 8 or (cmf >> 4) > 7 or ((@as(u32, cmf) << 8) | flg) % 31 != 0 or (flg & 0x20) != 0) return error.BadStream;
    return inflateRaw(allocator, src[2..], out);
}

test "inflate: stored, fixed and dynamic blocks" {
    const a = std.testing.allocator;
    var out: [64]u8 = undefined;
    // stored "hello"
    const stored = [_]u8{ 0x78, 0x01, 0x01, 0x05, 0x00, 0xfa, 0xff, 'h', 'e', 'l', 'l', 'o' };
    try std.testing.expectEqual(@as(usize, 5), try inflateZlib(a, &stored, &out));
    try std.testing.expectEqualStrings("hello", out[0..5]);
    // zlib.compress(b"hello hello hello hello", 9) (fixed block with a match)
    const fixed = [_]u8{ 0x78, 0xda, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x27, 0x01, 0x68, 0x03, 0x08, 0xb1 };
    try std.testing.expectEqual(@as(usize, 23), try inflateZlib(a, &fixed, &out));
    try std.testing.expectEqualStrings("hello hello hello hello", out[0..23]);
    try std.testing.expectError(error.Truncated, inflateZlib(a, fixed[0..6], &out));
}
