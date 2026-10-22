//! PNG decoder with Pillow 12's modes, sample unpacking and `info["transparency"]` (all colour types and bit depths,
//! Adam7, PLTE/tRNS, eXIf). Strict like `Image.open` + `verify()` + `load()`: every chunk's CRC is checked and the
//! file must reach IEND.
const std = @import("std");
const raster = @import("raster.zig");
const inflate = @import("inflate.zig");
const Allocator = std.mem.Allocator;
const Raster = raster.Raster;
const Crc32 = std.hash.crc.@"CRC-32/ISO-HDLC";

pub const Error = error{ Invalid, Animated } || Allocator.Error;

pub const signature = "\x89PNG\r\n\x1a\n";

pub fn matches(data: []const u8) bool {
    return data.len >= 8 and std.mem.eql(u8, data[0..8], signature);
}

/// Size from IHDR (as `Image.open` reads it: a valid first chunk), or null.
pub fn header(data: []const u8) ?struct { w: u32, h: u32 } {
    if (!matches(data) or data.len < 8 + 12 + 13) return null;
    if (std.mem.readInt(u32, data[8..12], .big) != 13 or !std.mem.eql(u8, data[12..16], "IHDR")) return null;
    var crc = Crc32.init();
    crc.update(data[12..29]);
    if (crc.final() != std.mem.readInt(u32, data[29..33], .big)) return null;
    const w = std.mem.readInt(u32, data[16..20], .big);
    const h = std.mem.readInt(u32, data[20..24], .big);
    if (w == 0 or h == 0 or w > 0x7fffffff or h > 0x7fffffff) return null;
    if (data[26] != 0 or data[27] != 0 or data[28] > 1 or modeFor(data[25], data[24]) == null) return null;
    return .{ .w = w, .h = h };
}

fn paeth(a: u8, b: u8, c: u8) u8 {
    const p: i32 = @as(i32, a) + b - c;
    const pa = @abs(p - a);
    const pb = @abs(p - b);
    const pc = @abs(p - c);
    if (pa <= pb and pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

fn unfilter(buf: []u8, rows: usize, row_bytes: usize, bpp: usize) Error!void {
    // buf holds rows * (1 + row_bytes) bytes: filter type, then the row
    var prev: ?[]u8 = null;
    for (0..rows) |y| {
        const line = buf[y * (1 + row_bytes) ..][0 .. 1 + row_bytes];
        const ft = line[0];
        const row = line[1..];
        switch (ft) {
            0 => {},
            1 => for (bpp..row_bytes) |i| {
                row[i] +%= row[i - bpp];
            },
            2 => if (prev) |p| for (0..row_bytes) |i| {
                row[i] +%= p[i];
            },
            3 => for (0..row_bytes) |i| {
                const a: u32 = if (i >= bpp) row[i - bpp] else 0;
                const b: u32 = if (prev) |p| p[i] else 0;
                row[i] +%= @intCast((a + b) >> 1);
            },
            4 => for (0..row_bytes) |i| {
                const a: u8 = if (i >= bpp) row[i - bpp] else 0;
                const b: u8 = if (prev) |p| p[i] else 0;
                const c: u8 = if (i >= bpp and prev != null) prev.?[i - bpp] else 0;
                row[i] +%= paeth(a, b, c);
            },
            else => return error.Invalid,
        }
        prev = row;
    }
}

const Pass = struct { x0: u32, y0: u32, dx: u32, dy: u32 };
const adam7 = [7]Pass{ .{ .x0 = 0, .y0 = 0, .dx = 8, .dy = 8 }, .{ .x0 = 4, .y0 = 0, .dx = 8, .dy = 8 }, .{ .x0 = 0, .y0 = 4, .dx = 4, .dy = 8 }, .{ .x0 = 2, .y0 = 0, .dx = 4, .dy = 4 }, .{ .x0 = 0, .y0 = 2, .dx = 2, .dy = 4 }, .{ .x0 = 1, .y0 = 0, .dx = 2, .dy = 2 }, .{ .x0 = 0, .y0 = 1, .dx = 1, .dy = 2 } };

fn passDim(total: u32, start: u32, step: u32) u32 {
    return if (total > start) (total - start + step - 1) / step else 0;
}

const Info = struct { w: u32, h: u32, depth: u8, ctype: u8, interlace: u8, mode: raster.Mode };

fn modeFor(ctype: u8, depth: u8) ?raster.Mode {
    return switch (ctype) {
        0 => switch (depth) {
            1 => .bit1,
            2, 4, 8 => .l,
            16 => .i16,
            else => null,
        },
        2 => if (depth == 8 or depth == 16) .rgb else null,
        3 => if (depth == 1 or depth == 2 or depth == 4 or depth == 8) .p else null,
        4 => if (depth == 8 or depth == 16) (if (depth == 8) raster.Mode.la else raster.Mode.rgba) else null,
        6 => if (depth == 8 or depth == 16) .rgba else null,
        else => null,
    };
}

fn channels(ctype: u8) usize {
    return switch (ctype) {
        0, 3 => 1,
        2 => 3,
        4 => 2,
        6 => 4,
        else => 0,
    };
}

/// Write the unfiltered scanline `row` (packed samples) into pixel row `dst` positions x0 + k*dx.
fn storeRow(r: *Raster, info: Info, row: []const u8, y: u32, x0: u32, dx: u32, count: u32) void {
    const nb = r.mode.bytes();
    const d = info.depth;
    var k: u32 = 0;
    while (k < count) : (k += 1) {
        const x = x0 + k * dx;
        const out = r.data[(@as(usize, y) * r.w + x) * nb ..][0..nb];
        switch (info.ctype) {
            0 => switch (d) {
                16 => {
                    out[0] = row[k * 2 + 1];
                    out[1] = row[k * 2];
                },
                8 => out[0] = row[k],
                else => {
                    const per: u32 = 8 / d;
                    const byte = row[k / per];
                    const shift: u3 = @intCast(8 - d - (k % per) * d);
                    const v = (byte >> shift) & ((@as(u8, 1) << @intCast(d)) - 1);
                    out[0] = switch (d) {
                        1 => if (v != 0) 255 else 0,
                        2 => v * 85,
                        4 => v * 17,
                        else => unreachable,
                    };
                },
            },
            3 => switch (d) {
                8 => out[0] = row[k],
                else => {
                    const per: u32 = 8 / d;
                    const byte = row[k / per];
                    const shift: u3 = @intCast(8 - d - (k % per) * d);
                    out[0] = (byte >> shift) & ((@as(u8, 1) << @intCast(d)) - 1);
                },
            },
            2 => if (d == 8) @memcpy(out, row[k * 3 ..][0..3]) else {
                out[0] = row[k * 6];
                out[1] = row[k * 6 + 2];
                out[2] = row[k * 6 + 4];
            },
            4 => if (d == 8) @memcpy(out, row[k * 2 ..][0..2]) else {
                // LA;16B -> RGBA: gray from the high byte, alpha from the high byte
                out[0] = row[k * 4];
                out[1] = row[k * 4];
                out[2] = row[k * 4];
                out[3] = row[k * 4 + 2];
            },
            6 => if (d == 8) @memcpy(out, row[k * 4 ..][0..4]) else {
                out[0] = row[k * 8];
                out[1] = row[k * 8 + 2];
                out[2] = row[k * 8 + 4];
                out[3] = row[k * 8 + 6];
            },
            else => unreachable,
        }
    }
}

pub fn decode(allocator: Allocator, data: []const u8, max_pixels: u64) Error!Raster {
    if (!matches(data)) return error.Invalid;
    var pos: usize = 8;
    var info: ?Info = null;
    var plte: ?[]const u8 = null;
    var trns: ?[]const u8 = null;
    var exif: ?[]const u8 = null;
    var xmp: ?[]const u8 = null;
    var idat: std.ArrayList(u8) = .empty;
    defer idat.deinit(allocator);
    var seen_idat = false;
    var seen_iend = false;
    var actl: ?u32 = null;
    var fctl_before_idat = false;
    while (pos + 12 <= data.len) {
        const len = std.mem.readInt(u32, data[pos..][0..4], .big);
        if (len > 0x7fffffff) return error.Invalid;
        const typ = data[pos + 4 ..][0..4];
        if (pos + 12 + len > data.len) return error.Invalid;
        const body = data[pos + 8 ..][0..len];
        const want_crc = std.mem.readInt(u32, data[pos + 8 + len ..][0..4], .big);
        var crc = Crc32.init();
        crc.update(typ);
        crc.update(body);
        if (crc.final() != want_crc) return error.Invalid;
        pos += 12 + len;
        if (info == null and !std.mem.eql(u8, typ, "IHDR")) return error.Invalid;
        if (std.mem.eql(u8, typ, "IHDR")) {
            if (info != null or len != 13) return error.Invalid;
            const w = std.mem.readInt(u32, body[0..4], .big);
            const h = std.mem.readInt(u32, body[4..8], .big);
            const depth = body[8];
            const ctype = body[9];
            if (body[10] != 0 or body[11] != 0 or body[12] > 1) return error.Invalid;
            if (w == 0 or h == 0 or w > 0x7fffffff or h > 0x7fffffff) return error.Invalid;
            if (@as(u64, w) * h > max_pixels) return error.Invalid;
            const mode = modeFor(ctype, depth) orelse return error.Invalid;
            info = .{ .w = w, .h = h, .depth = depth, .ctype = ctype, .interlace = body[12], .mode = mode };
        } else if (std.mem.eql(u8, typ, "PLTE")) {
            if (!seen_idat) plte = body;
        } else if (std.mem.eql(u8, typ, "tRNS")) {
            if (!seen_idat) trns = body;
        } else if (std.mem.eql(u8, typ, "IDAT")) {
            seen_idat = true;
            try idat.appendSlice(allocator, body);
        } else if (std.mem.eql(u8, typ, "eXIf")) {
            exif = body;
        } else if (std.mem.eql(u8, typ, "acTL")) {
            if (len != 8) return error.Invalid;
            actl = std.mem.readInt(u32, body[0..4], .big);
        } else if (std.mem.eql(u8, typ, "fcTL")) {
            if (!seen_idat) fctl_before_idat = true;
        } else if (std.mem.eql(u8, typ, "iTXt")) {
            // XMP: keyword "XML:com.adobe.xmp", flags, language, translated keyword, text
            const kw = "XML:com.adobe.xmp";
            if (body.len > kw.len + 5 and std.mem.startsWith(u8, body, kw) and body[kw.len] == 0 and body[kw.len + 1] == 0 and body[kw.len + 2] == 0) {
                if (std.mem.indexOfScalarPos(u8, body, kw.len + 3, 0)) |e1| {
                    if (std.mem.indexOfScalarPos(u8, body, e1 + 1, 0)) |e2| xmp = body[e2 + 1 ..];
                }
            }
        } else if (std.mem.eql(u8, typ, "IEND")) {
            seen_iend = true;
            break;
        }
    }
    if (!seen_iend) return error.Invalid;
    const im = info orelse return error.Invalid;
    if (actl) |n| {
        if (n != 0 and n <= 0x80000000) {
            const frames = n + @as(u32, if (fctl_before_idat) 0 else 1);
            if (frames != 1) return error.Animated;
            return error.Invalid; // a one-frame APNG with a frame control chunk: not supported here
        }
    }
    if (im.ctype == 3 and plte == null) {
        // Pillow keeps its default (grey ramp) palette; kept below
    }

    // raw bytes needed
    const bits_pp: usize = channels(im.ctype) * im.depth;
    var total_raw: usize = 0;
    if (im.interlace == 0) {
        const rb = (@as(usize, im.w) * bits_pp + 7) / 8;
        total_raw = (rb + 1) * im.h;
    } else for (adam7) |p| {
        const pw = passDim(im.w, p.x0, p.dx);
        const ph = passDim(im.h, p.y0, p.dy);
        if (pw == 0 or ph == 0) continue;
        total_raw += ((@as(usize, pw) * bits_pp + 7) / 8 + 1) * ph;
    }
    const raw = try allocator.alloc(u8, total_raw);
    defer allocator.free(raw);
    const got = inflate.inflateZlib(allocator, idat.items, raw) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.Invalid,
    };
    if (got != total_raw) return error.Invalid;

    const nb = im.mode.bytes();
    const pix = try allocator.alloc(u8, @as(usize, im.w) * im.h * nb);
    errdefer allocator.free(pix);
    var r: Raster = .{ .w = im.w, .h = im.h, .mode = im.mode, .data = pix, .allocator = allocator };
    errdefer r.deinit();
    const bpp = @max(1, bits_pp / 8);
    if (im.interlace == 0) {
        const rb = (@as(usize, im.w) * bits_pp + 7) / 8;
        try unfilter(raw, im.h, rb, bpp);
        for (0..im.h) |y| storeRow(&r, im, raw[y * (rb + 1) + 1 ..][0..rb], @intCast(y), 0, 1, im.w);
    } else {
        var off: usize = 0;
        for (adam7) |p| {
            const pw = passDim(im.w, p.x0, p.dx);
            const ph = passDim(im.h, p.y0, p.dy);
            if (pw == 0 or ph == 0) continue;
            const rb = (@as(usize, pw) * bits_pp + 7) / 8;
            try unfilter(raw[off..], ph, rb, bpp);
            for (0..ph) |k| storeRow(&r, im, raw[off + k * (rb + 1) + 1 ..][0..rb], p.y0 + @as(u32, @intCast(k)) * p.dy, p.x0, p.dx, pw);
            off += (rb + 1) * ph;
        }
    }

    if (im.mode == .p) {
        if (plte) |pl| r.setPalette(pl) else {
            var ramp: [768]u8 = undefined;
            for (0..256) |i| {
                ramp[i * 3] = @intCast(i);
                ramp[i * 3 + 1] = @intCast(i);
                ramp[i * 3 + 2] = @intCast(i);
            }
            r.palette = ramp;
        }
    }
    if (trns) |s| switch (im.mode) {
        .p => {
            // _simple_palette = ^\xff*\x00\xff*$ : one fully transparent entry, the rest opaque -> that index
            var zeros: usize = 0;
            var first_zero: usize = 0;
            var others_opaque = true;
            for (s, 0..) |b, i| {
                if (b == 0) {
                    if (zeros == 0) first_zero = i;
                    zeros += 1;
                } else if (b != 255) others_opaque = false;
            }
            if (zeros == 1 and others_opaque) {
                r.transparency = .{ .index = @intCast(first_zero) };
            } else {
                var t: [256]u8 = @splat(255);
                const n = @min(s.len, 256);
                @memcpy(t[0..n], s[0..n]);
                r.transparency = .{ .table = t };
            }
        },
        .bit1, .l, .i16 => if (s.len >= 2) {
            r.transparency = .{ .gray = std.mem.readInt(u16, s[0..2], .big) };
        },
        .rgb => if (s.len >= 6) {
            r.transparency = .{ .rgb = .{ std.mem.readInt(u16, s[0..2], .big), std.mem.readInt(u16, s[2..4], .big), std.mem.readInt(u16, s[4..6], .big) } };
        },
        else => {},
    };
    if (exif) |e| r.exif = try allocator.dupe(u8, e);
    if (xmp) |x| r.xmp = try allocator.dupe(u8, x);
    return r;
}
