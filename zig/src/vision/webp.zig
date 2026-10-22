//! WebP container, lossy (VP8 + ALPH) and lossless (VP8L) pictures as libwebp 1.6.0 decodes them for Pillow 12.3
//! (WebPAnimDecoder, RGBA output, fancy upsampling, no dithering). Still pictures only: more than one frame is
//! refused with the recipe's "animated" message.
const std = @import("std");
const raster = @import("raster.zig");
const vp8 = @import("vp8.zig");
const vp8l = @import("vp8l.zig");
const Allocator = std.mem.Allocator;

pub const Error = error{ Invalid, Animated } || Allocator.Error;

const Chunk = struct { id: [4]u8, payload: []const u8 };

fn le32(b: []const u8) u32 {
    return std.mem.readInt(u32, b[0..4], .little);
}

fn le24(b: []const u8) u32 {
    return @as(u32, b[0]) | (@as(u32, b[1]) << 8) | (@as(u32, b[2]) << 16);
}

/// Walk the RIFF container. Returns the chunk list (caller frees) within the declared RIFF size.
fn chunks(a: Allocator, data: []const u8) Error![]Chunk {
    if (data.len < 20 or !std.mem.eql(u8, data[0..4], "RIFF") or !std.mem.eql(u8, data[8..12], "WEBP")) return error.Invalid;
    const riff = le32(data[4..8]);
    if (riff < 12 or riff > 0xfffffff6) return error.Invalid;
    const total: u64 = @as(u64, riff) + 8;
    if (data.len < total) return error.Invalid; // truncated
    const end: usize = @intCast(total);
    var list: std.ArrayList(Chunk) = .empty;
    errdefer list.deinit(a);
    var p: usize = 12;
    while (p + 8 <= end) {
        const size = le32(data[p + 4 ..]);
        const body = p + 8;
        if (@as(u64, body) + size > end) return error.Invalid;
        try list.append(a, .{ .id = data[p..][0..4].*, .payload = data[body..][0..size] });
        p = body + size + (size & 1);
    }
    return list.toOwnedSlice(a);
}

pub fn header(data: []const u8) ?struct { w: u32, h: u32 } {
    if (data.len < 25 or !std.mem.eql(u8, data[0..4], "RIFF") or !std.mem.eql(u8, data[8..12], "WEBP")) return null;
    const id = data[12..16];
    const size = le32(data[16..20]);
    if (20 + @as(u64, size) > data.len) return null;
    const body = data[20..][0..size];
    if (std.mem.eql(u8, id, "VP8 ")) {
        const fi = vp8.frameInfo(body) orelse return null;
        return .{ .w = @intCast(fi.width), .h = @intCast(fi.height) };
    } else if (std.mem.eql(u8, id, "VP8L")) {
        if (body.len < 5 or body[0] != 0x2f) return null;
        const bits = std.mem.readInt(u32, body[1..5], .little);
        return .{ .w = (bits & 0x3fff) + 1, .h = ((bits >> 14) & 0x3fff) + 1 };
    } else if (std.mem.eql(u8, id, "VP8X")) {
        if (body.len < 10) return null;
        return .{ .w = le24(body[4..]) + 1, .h = le24(body[7..]) + 1 };
    }
    return null;
}

// ---- YUV -> RGB (libwebp yuv.h) and the fancy upsampler (upsampling.c) -----------------------------------------------
inline fn multHi(v: i32, coeff: i32) i32 {
    return (v * coeff) >> 8;
}

inline fn clip8(v: i32) u8 {
    return if ((v & ~@as(i32, (256 << 6) - 1)) == 0) @intCast(v >> 6) else if (v < 0) 0 else 255;
}

inline fn yuvToRgba(y: u8, u: u8, v: u8, out: []u8) void {
    const yy: i32 = y;
    out[0] = clip8(multHi(yy, 19077) + multHi(v, 26149) - 14234);
    out[1] = clip8(multHi(yy, 19077) - multHi(u, 6419) - multHi(v, 13320) + 8708);
    out[2] = clip8(multHi(yy, 19077) + multHi(u, 33050) - 17685);
    out[3] = 255;
}

/// One output pair of rows from two chroma rows (top_c above, cur_c below), as UpsampleRgbaLinePair.
fn upsampleLinePair(top_y: []const u8, bottom_y: ?[]const u8, top_u: []const u8, top_v: []const u8, cur_u: []const u8, cur_v: []const u8, top_dst: []u8, bottom_dst: ?[]u8, len: usize) void {
    const last_pixel_pair = (len - 1) >> 1;
    var tl_u: u32 = top_u[0];
    var tl_v: u32 = top_v[0];
    var l_u: u32 = cur_u[0];
    var l_v: u32 = cur_v[0];
    {
        const cu0: u8 = @intCast((3 * tl_u + l_u + 2) >> 2);
        const cv0: u8 = @intCast((3 * tl_v + l_v + 2) >> 2);
        yuvToRgba(top_y[0], cu0, cv0, top_dst[0..4]);
    }
    if (bottom_y) |by| {
        const cu0: u8 = @intCast((3 * l_u + tl_u + 2) >> 2);
        const cv0: u8 = @intCast((3 * l_v + tl_v + 2) >> 2);
        yuvToRgba(by[0], cu0, cv0, bottom_dst.?[0..4]);
    }
    var x: usize = 1;
    while (x <= last_pixel_pair) : (x += 1) {
        const t_u: u32 = top_u[x];
        const t_v: u32 = top_v[x];
        const c_u: u32 = cur_u[x];
        const c_v: u32 = cur_v[x];
        const avg_u = tl_u + t_u + l_u + c_u + 8;
        const avg_v = tl_v + t_v + l_v + c_v + 8;
        const diag12_u = (avg_u + 2 * (t_u + l_u)) >> 3;
        const diag12_v = (avg_v + 2 * (t_v + l_v)) >> 3;
        const diag03_u = (avg_u + 2 * (tl_u + c_u)) >> 3;
        const diag03_v = (avg_v + 2 * (tl_v + c_v)) >> 3;
        {
            const cu0: u8 = @intCast((diag12_u + tl_u) >> 1);
            const cv0: u8 = @intCast((diag12_v + tl_v) >> 1);
            const cu1: u8 = @intCast((diag03_u + t_u) >> 1);
            const cv1: u8 = @intCast((diag03_v + t_v) >> 1);
            yuvToRgba(top_y[2 * x - 1], cu0, cv0, top_dst[(2 * x - 1) * 4 ..][0..4]);
            yuvToRgba(top_y[2 * x], cu1, cv1, top_dst[(2 * x) * 4 ..][0..4]);
        }
        if (bottom_y) |by| {
            const cu0: u8 = @intCast((diag03_u + l_u) >> 1);
            const cv0: u8 = @intCast((diag03_v + l_v) >> 1);
            const cu1: u8 = @intCast((diag12_u + c_u) >> 1);
            const cv1: u8 = @intCast((diag12_v + c_v) >> 1);
            yuvToRgba(by[2 * x - 1], cu0, cv0, bottom_dst.?[(2 * x - 1) * 4 ..][0..4]);
            yuvToRgba(by[2 * x], cu1, cv1, bottom_dst.?[(2 * x) * 4 ..][0..4]);
        }
        tl_u = t_u;
        tl_v = t_v;
        l_u = c_u;
        l_v = c_v;
    }
    if ((len & 1) == 0) {
        {
            const cu0: u8 = @intCast((3 * tl_u + l_u + 2) >> 2);
            const cv0: u8 = @intCast((3 * tl_v + l_v + 2) >> 2);
            yuvToRgba(top_y[len - 1], cu0, cv0, top_dst[(len - 1) * 4 ..][0..4]);
        }
        if (bottom_y) |by| {
            const cu0: u8 = @intCast((3 * l_u + tl_u + 2) >> 2);
            const cv0: u8 = @intCast((3 * l_v + tl_v + 2) >> 2);
            yuvToRgba(by[len - 1], cu0, cv0, bottom_dst.?[(len - 1) * 4 ..][0..4]);
        }
    }
}

/// EmitFancyRGB over the whole picture.
fn toRgbaFancy(p: *const vp8.Planes, out: []u8) void {
    const w = p.width;
    const h = p.height;
    const row = w * 4;
    const ys = p.y_stride;
    const cs = p.uv_stride;
    // row 0 uses chroma row 0 for both neighbours
    upsampleLinePair(p.y[0..w], null, p.u[0..], p.v[0..], p.u[0..], p.v[0..], out[0..row], null, w);
    var y: usize = 1;
    while (y < h) : (y += 2) {
        // rows y (above) and y + 1 (below), chroma rows (y-1)/2 and (y+1)/2
        const k = (y + 1) / 2;
        const top_c = k - 1;
        const have_bottom = y + 1 < h;
        const cur_c = if (have_bottom) k else top_c; // the very last row of an even-sized picture repeats the chroma row
        const bottom_y: ?[]const u8 = if (have_bottom) p.y[(y + 1) * ys ..][0..w] else null;
        const bottom_dst: ?[]u8 = if (have_bottom) out[(y + 1) * row ..][0..row] else null;
        upsampleLinePair(p.y[y * ys ..][0..w], bottom_y, p.u[top_c * cs ..], p.v[top_c * cs ..], p.u[cur_c * cs ..], p.v[cur_c * cs ..], out[y * row ..][0..row], bottom_dst, w);
    }
}

// ---- alpha plane ------------------------------------------------------------------------------------------------------
fn decodeAlpha(a: Allocator, data: []const u8, w: usize, h: usize) Error![]u8 {
    if (data.len < 1) return error.Invalid;
    const method = data[0] & 3;
    const filter = (data[0] >> 2) & 3;
    const pre = (data[0] >> 4) & 3;
    if (method > 1 or pre > 1) return error.Invalid;
    const body = data[1..];
    const alpha = try a.alloc(u8, w * h);
    errdefer a.free(alpha);
    if (method == 0) {
        if (body.len < w * h) return error.Invalid;
        @memcpy(alpha, body[0 .. w * h]);
    } else {
        const dec = vp8l.decode(a, body, false, w, h) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Invalid => return error.Invalid,
        };
        defer a.free(dec.argb);
        for (dec.argb, alpha) |p, *o| o.* = @intCast((p >> 8) & 0xff);
    }
    if (filter != 0) {
        // unfilter in place, row by row (prev = the previous unfiltered row)
        for (0..h) |y| {
            const row = alpha[y * w ..][0..w];
            const prev: ?[]const u8 = if (y == 0) null else alpha[(y - 1) * w ..][0..w];
            switch (filter) {
                1 => { // horizontal
                    var pred: u8 = if (prev) |pv| pv[0] else 0;
                    for (row) |*v| {
                        v.* +%= pred;
                        pred = v.*;
                    }
                },
                2 => { // vertical
                    if (prev) |pv| {
                        for (row, pv) |*v, pp| v.* +%= pp;
                    } else {
                        var pred: u8 = 0;
                        for (row) |*v| {
                            v.* +%= pred;
                            pred = v.*;
                        }
                    }
                },
                else => { // gradient
                    if (prev) |pv| {
                        var top: i32 = pv[0];
                        var top_left: i32 = top;
                        var left: i32 = top;
                        for (row, 0..) |*v, i| {
                            top = pv[i];
                            const g = std.math.clamp(left + top - top_left, 0, 255);
                            left = @as(u8, @truncate(@as(u32, @intCast(@as(i32, v.*) + g))));
                            top_left = top;
                            v.* = @intCast(left);
                        }
                    } else {
                        var pred: u8 = 0;
                        for (row) |*v| {
                            v.* +%= pred;
                            pred = v.*;
                        }
                    }
                },
            }
        }
    }
    return alpha;
}

pub fn decode(a: Allocator, data: []const u8, max_pixels: u64) Error!raster.Raster {
    const list = try chunks(a, data);
    defer a.free(list);
    if (list.len == 0) return error.Invalid;
    var extended = false;
    var flags: u8 = 0;
    var canvas_w: u32 = 0;
    var canvas_h: u32 = 0;
    var image: ?Chunk = null;
    var alph: ?[]const u8 = null;
    var exif: ?[]const u8 = null;
    var xmp: ?[]const u8 = null;
    var frames: usize = 0;
    const first = list[0].id;
    if (std.mem.eql(u8, &first, "VP8X")) {
        extended = true;
        const px = list[0].payload;
        if (px.len < 10) return error.Invalid;
        flags = px[0];
        canvas_w = le24(px[4..]) + 1;
        canvas_h = le24(px[7..]) + 1;
    } else if (!std.mem.eql(u8, &first, "VP8 ") and !std.mem.eql(u8, &first, "VP8L")) return error.Invalid;
    for (list) |c| {
        if (std.mem.eql(u8, &c.id, "VP8 ") or std.mem.eql(u8, &c.id, "VP8L")) {
            if (image == null) image = c;
        } else if (std.mem.eql(u8, &c.id, "ALPH")) {
            if (extended and alph == null and image == null) alph = c.payload;
        } else if (std.mem.eql(u8, &c.id, "ANMF")) {
            frames += 1;
        } else if (std.mem.eql(u8, &c.id, "EXIF")) {
            if (exif == null) exif = c.payload;
        } else if (std.mem.eql(u8, &c.id, "XMP ")) {
            if (xmp == null) xmp = c.payload;
        }
    }
    if (extended and (flags & 2) != 0) {
        if (frames > 1) return error.Animated;
        return error.Invalid; // a one-frame animation: not supported here
    }
    if (frames != 0) return error.Invalid;
    const img = image orelse return error.Invalid;
    // the bitstream's own size, checked before anything is allocated for it
    {
        const dims: struct { w: u64, h: u64 } = if (std.mem.eql(u8, &img.id, "VP8 ")) blk: {
            const fi = vp8.frameInfo(img.payload) orelse return error.Invalid;
            break :blk .{ .w = fi.width, .h = fi.height };
        } else blk: {
            if (img.payload.len < 5 or img.payload[0] != 0x2f) return error.Invalid;
            const bits = std.mem.readInt(u32, img.payload[1..5], .little);
            break :blk .{ .w = (bits & 0x3fff) + 1, .h = ((bits >> 14) & 0x3fff) + 1 };
        };
        if (dims.w * dims.h > max_pixels) return error.Invalid;
        if (extended and (canvas_w != dims.w or canvas_h != dims.h)) return error.Invalid;
    }
    var rgba: []u8 = undefined;
    var w: usize = undefined;
    var h: usize = undefined;
    var has_alpha = extended;
    if (std.mem.eql(u8, &img.id, "VP8 ")) {
        var planes = vp8.decode(a, img.payload) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Invalid => return error.Invalid,
        };
        defer planes.deinit(a);
        w = planes.width;
        h = planes.height;
        if (extended and (canvas_w != w or canvas_h != h)) return error.Invalid;
        rgba = try a.alloc(u8, w * h * 4);
        errdefer a.free(rgba);
        toRgbaFancy(&planes, rgba);
        if (alph) |al| {
            const plane = try decodeAlpha(a, al, w, h);
            defer a.free(plane);
            for (0..w * h) |i| rgba[i * 4 + 3] = plane[i];
        }
    } else {
        const dec = vp8l.decode(a, img.payload, true, 0, 0) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Invalid => return error.Invalid,
        };
        defer a.free(dec.argb);
        w = dec.width;
        h = dec.height;
        if (extended and (canvas_w != w or canvas_h != h)) return error.Invalid;
        has_alpha = true;
        rgba = try a.alloc(u8, w * h * 4);
        errdefer a.free(rgba);
        for (dec.argb, 0..) |p, i| {
            rgba[i * 4] = @intCast((p >> 16) & 0xff);
            rgba[i * 4 + 1] = @intCast((p >> 8) & 0xff);
            rgba[i * 4 + 2] = @intCast(p & 0xff);
            rgba[i * 4 + 3] = @intCast(p >> 24);
        }
    }
    var r: raster.Raster = .{ .w = @intCast(w), .h = @intCast(h), .mode = .rgba, .data = rgba, .allocator = a };
    errdefer r.deinit();
    if (!has_alpha) {
        // simple lossy file: Pillow's mode is RGB
        const rgb = try a.alloc(u8, w * h * 3);
        for (0..w * h) |i| @memcpy(rgb[i * 3 ..][0..3], rgba[i * 4 ..][0..3]);
        a.free(rgba);
        r.data = rgb;
        r.mode = .rgb;
    }
    if (exif) |e| r.exif = try a.dupe(u8, e);
    if (xmp) |x| r.xmp = try a.dupe(u8, x);
    return r;
}
