//! Decoded pictures as Pillow holds them after `load()` (mode + pixels + `info["transparency"]`), and the
//! conversions images._decode applies: exif_transpose, convert("RGB"), or convert("RGBA") pasted on white.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Mode = enum {
    bit1, // "1": one byte per pixel, 0 or 255
    l, // "L"
    la, // "LA": 2 bytes
    p, // "P": palette index
    rgb, // 3 bytes
    rgba, // 4 bytes
    cmyk, // 4 bytes (JPEG CMYK/YCCK, inverted Adobe samples already undone as Pillow does)
    i16, // "I;16": 2 bytes little endian

    pub fn bytes(m: Mode) usize {
        return switch (m) {
            .bit1, .l, .p => 1,
            .la, .i16 => 2,
            .rgb => 3,
            .rgba, .cmyk => 4,
        };
    }
    pub fn hasAlphaBand(m: Mode) bool {
        return m == .la or m == .rgba;
    }
};

/// `image.info["transparency"]`
pub const Transparency = union(enum) {
    none,
    index: u8, // an int (P with one fully transparent entry; or an L/1 value)
    gray: u16, // L, 1 or I;16 (PNG tRNS sample)
    rgb: [3]u16, // RGB (PNG tRNS)
    table: [256]u8, // P: one alpha per palette entry (bytes), entries past the table's length opaque (255)
};

pub const Raster = struct {
    w: u32,
    h: u32,
    mode: Mode,
    data: []u8, // w * h * mode.bytes()
    palette: ?[768]u8 = null, // P only
    transparency: Transparency = .none,
    exif: ?[]u8 = null, // owned: the TIFF block (after any "Exif\0\0" header)
    xmp: ?[]u8 = null, // owned
    allocator: Allocator,

    pub fn deinit(r: *Raster) void {
        r.allocator.free(r.data);
        if (r.exif) |e| r.allocator.free(e);
        if (r.xmp) |x| r.allocator.free(x);
    }

    pub fn setPalette(r: *Raster, bytes: []const u8) void {
        var p: [768]u8 = @splat(0);
        @memcpy(p[0..@min(768, bytes.len)], bytes[0..@min(768, bytes.len)]);
        r.palette = p;
    }
};

/// Pillow's `DIV255(a)`: (tmp = a + 128, ((tmp >> 8) + tmp) >> 8)
inline fn div255(a: u32) u8 {
    const tmp = a + 128;
    return @intCast(((tmp >> 8) + tmp) >> 8);
}

/// What PIL.Image.Exif says about tag 0x0112 in IFD0.
pub const OrientationTag = union(enum) {
    absent,
    value: i64, // an integer (SHORT/LONG/SSHORT/SLONG)
    other, // present, but not an int (BYTE, ...): no transform
};

fn typeSize(typ: u16) usize {
    return switch (typ) {
        1, 2, 6, 7 => 1,
        3, 8 => 2,
        4, 9, 11 => 4,
        5, 10, 12 => 8,
        else => 0,
    };
}

/// The Orientation entry of a TIFF block (the EXIF payload after any `Exif\0\0` header), as Pillow reads it.
pub fn exifOrientationTag(tiff_in: []const u8) OrientationTag {
    const tiff = if (std.mem.startsWith(u8, tiff_in, "Exif\x00\x00")) tiff_in[6..] else tiff_in;
    if (tiff.len < 8) return .absent;
    const little = std.mem.eql(u8, tiff[0..4], "II*\x00");
    if (!little and !std.mem.eql(u8, tiff[0..4], "MM\x00*")) return .absent;
    const endian: std.builtin.Endian = if (little) .little else .big;
    const off = std.mem.readInt(u32, tiff[4..8], endian);
    if (off > tiff.len or tiff.len - off < 2) return .absent;
    const n = std.mem.readInt(u16, tiff[off..][0..2], endian);
    var found: OrientationTag = .absent;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const e = @as(usize, off) + 2 + i * 12;
        if (e + 12 > tiff.len) break;
        const tag = std.mem.readInt(u16, tiff[e..][0..2], endian);
        const typ = std.mem.readInt(u16, tiff[e + 2 ..][0..2], endian);
        const count = std.mem.readInt(u32, tiff[e + 4 ..][0..4], endian);
        const unit = typeSize(typ);
        if (unit == 0) continue;
        const size = @as(u64, count) * unit;
        var data: []const u8 = tiff[e + 8 ..][0..4];
        if (size > 4) {
            const o = std.mem.readInt(u32, tiff[e + 8 ..][0..4], endian);
            if (@as(u64, o) + size > tiff.len) continue; // Possibly corrupt EXIF data
            data = tiff[o..][0..@intCast(size)];
        } else data = data[0..@intCast(size)];
        if (tag != 0x0112) continue;
        if (count == 0) continue;
        found = switch (typ) {
            3 => .{ .value = std.mem.readInt(u16, data[0..2], endian) },
            4 => .{ .value = std.mem.readInt(u32, data[0..4], endian) },
            8 => .{ .value = std.mem.readInt(i16, data[0..2], endian) },
            9 => .{ .value = std.mem.readInt(i32, data[0..4], endian) },
            else => .other,
        };
    }
    return found;
}

/// `tiff:Orientation(="|>)([0-9])` in XMP text (first match).
pub fn xmpOrientation(xmp: []const u8) ?u8 {
    if (!std.unicode.utf8ValidateSlice(xmp)) return null;
    const key = "tiff:Orientation";
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, xmp, from, key)) |i| {
        var j = i + key.len;
        if (j < xmp.len and xmp[j] == '>') {
            j += 1;
        } else if (j + 1 < xmp.len and xmp[j] == '=' and xmp[j + 1] == '"') {
            j += 2;
        } else {
            from = i + 1;
            continue;
        }
        if (j < xmp.len and xmp[j] >= '0' and xmp[j] <= '9') return xmp[j] - '0';
        from = i + 1;
    }
    return null;
}

/// ImageOps.exif_transpose's orientation for a loaded picture: the EXIF Orientation, else XMP's tiff:Orientation;
/// 1 when there is none or it is not 2..8.
pub fn orientationOf(r: *const Raster) u8 {
    var o: i64 = 1;
    var have = false;
    if (r.exif) |e| switch (exifOrientationTag(e)) {
        .absent => {},
        .value => |v| {
            o = v;
            have = true;
        },
        .other => {
            have = true;
            o = 1;
        },
    };
    if (!have) if (r.xmp) |x| if (xmpOrientation(x)) |v| {
        o = v;
    };
    return if (o >= 2 and o <= 8) @intCast(o) else 1;
}

/// Transpose an RGB buffer by an EXIF orientation (ImageOps.exif_transpose). Returns the new dims.
pub fn orient(allocator: Allocator, rgb: []u8, w: u32, h: u32, o: u8) Allocator.Error!struct { data: []u8, w: u32, h: u32 } {
    if (o <= 1 or o > 8) return .{ .data = rgb, .w = w, .h = h };
    const swap = o >= 5;
    const nw = if (swap) h else w;
    const nh = if (swap) w else h;
    const out = try allocator.alloc(u8, rgb.len);
    errdefer allocator.free(out);
    for (0..nh) |y| for (0..nw) |x| {
        // source coordinates for output (x, y)
        const sx: usize, const sy: usize = switch (o) {
            2 => .{ w - 1 - x, y }, // FLIP_LEFT_RIGHT
            3 => .{ w - 1 - x, h - 1 - y }, // ROTATE_180
            4 => .{ x, h - 1 - y }, // FLIP_TOP_BOTTOM
            5 => .{ y, x }, // TRANSPOSE
            6 => .{ y, h - 1 - x }, // ROTATE_270
            7 => .{ w - 1 - y, h - 1 - x }, // TRANSVERSE
            8 => .{ w - 1 - y, x }, // ROTATE_90
            else => unreachable,
        };
        @memcpy(out[(y * nw + x) * 3 ..][0..3], rgb[(sy * w + sx) * 3 ..][0..3]);
    };
    allocator.free(rgb);
    return .{ .data = out, .w = nw, .h = nh };
}
