//! What images._decode does after loading: exif_transpose, then convert("RGB") -- or, when the picture has an
//! alpha band or a `transparency` entry, convert("RGBA") pasted on white with Pillow's integer blend.
const std = @import("std");
const raster = @import("raster.zig");
const Raster = raster.Raster;
const Allocator = std.mem.Allocator;

inline fn div255(a: u32) u8 {
    const tmp = a + 128;
    return @intCast(((tmp >> 8) + tmp) >> 8);
}

/// `BLEND(mask, 255, c)`: ImagingPaste's per-channel mix of white and the source by the mask.
inline fn onWhite(c: u8, a: u8) u8 {
    return div255(@as(u32, 255) * (255 - a) + @as(u32, c) * a);
}

inline fn clip16(lo: u8, hi: u8) u8 {
    return if (hi != 0) 255 else lo;
}

/// Pillow's CMYK -> RGB (Convert.c cmyk2rgb): nk = 255 - k; channel = nk - MULDIV255(c, nk).
inline fn cmykChannel(c: u8, k: u8) u8 {
    const nk: u32 = 255 - k;
    const t = @as(u32, c) * nk + 128;
    const m = ((t >> 8) + t) >> 8;
    return @intCast(@min(255, nk -| m));
}

/// RGB bytes (w*h*3) of the picture as `_decode` produces them, before orientation.
pub fn toRgb(allocator: Allocator, r: *const Raster) Allocator.Error![]u8 {
    const n = @as(usize, r.w) * r.h;
    const out = try allocator.alloc(u8, n * 3);
    errdefer allocator.free(out);
    const has_alpha = r.mode.hasAlphaBand() or r.transparency != .none;
    const d = r.data;
    if (!has_alpha) {
        switch (r.mode) {
            .bit1, .l => for (0..n) |i| {
                @memset(out[i * 3 ..][0..3], d[i]);
            },
            .p => {
                const pal = r.palette.?;
                for (0..n) |i| @memcpy(out[i * 3 ..][0..3], pal[@as(usize, d[i]) * 3 ..][0..3]);
            },
            .rgb => @memcpy(out, d),
            .i16 => for (0..n) |i| {
                @memset(out[i * 3 ..][0..3], clip16(d[i * 2], d[i * 2 + 1]));
            },
            .cmyk => for (0..n) |i| {
                const k = d[i * 4 + 3];
                out[i * 3] = cmykChannel(d[i * 4], k);
                out[i * 3 + 1] = cmykChannel(d[i * 4 + 1], k);
                out[i * 3 + 2] = cmykChannel(d[i * 4 + 2], k);
            },
            .la, .rgba => unreachable,
        }
        return out;
    }
    for (0..n) |i| {
        var px: [4]u8 = undefined;
        switch (r.mode) {
            .la => px = .{ d[i * 2], d[i * 2], d[i * 2], d[i * 2 + 1] },
            .rgba => px = .{ d[i * 4], d[i * 4 + 1], d[i * 4 + 2], d[i * 4 + 3] },
            .l => {
                const v = d[i];
                const a: u8 = switch (r.transparency) {
                    .gray => |t| if (v == t) 0 else 255,
                    .index => |t| if (v == t) 0 else 255,
                    else => 255,
                };
                px = .{ v, v, v, a };
            },
            .bit1 => {
                const a: u8 = switch (r.transparency) {
                    .gray => |t| if (d[i] == t) 0 else 255,
                    else => 255,
                };
                px = .{ d[i], d[i], d[i], a };
            },
            .i16 => {
                const v = clip16(d[i * 2], d[i * 2 + 1]);
                px = .{ v, v, v, 255 };
            },
            .rgb => {
                const a: u8 = switch (r.transparency) {
                    .rgb => |t| if (d[i * 3] == t[0] and d[i * 3 + 1] == t[1] and d[i * 3 + 2] == t[2]) 0 else 255,
                    else => 255,
                };
                px = .{ d[i * 3], d[i * 3 + 1], d[i * 3 + 2], a };
            },
            .p => {
                const pal = r.palette.?;
                const idx = d[i];
                const a: u8 = switch (r.transparency) {
                    .index => |t| if (idx == t) 0 else 255,
                    .table => |t| t[idx],
                    else => 255,
                };
                px = .{ pal[idx * 3], pal[idx * 3 + 1], pal[idx * 3 + 2], a };
            },
            .cmyk => {
                const k = d[i * 4 + 3];
                px = .{ cmykChannel(d[i * 4], k), cmykChannel(d[i * 4 + 1], k), cmykChannel(d[i * 4 + 2], k), 255 };
            },
        }
        out[i * 3] = onWhite(px[0], px[3]);
        out[i * 3 + 1] = onWhite(px[1], px[3]);
        out[i * 3 + 2] = onWhite(px[2], px[3]);
    }
    return out;
}
