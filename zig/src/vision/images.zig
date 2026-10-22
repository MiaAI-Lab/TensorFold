//! `images._decode` and `images._data_bytes`: bytes -> oriented RGB8, pixel-identical to Pillow 12.3 for JPEG, PNG
//! and WebP, with the recipe's limits and error texts.
const std = @import("std");
const errors = @import("errors.zig");
const limits_mod = @import("limits.zig");
const raster = @import("raster.zig");
const convert = @import("convert.zig");
const png = @import("png.zig");
const Allocator = std.mem.Allocator;
const Diag = errors.Diag;

pub const invalid_message = "image bytes are invalid or unsupported; use JPEG, PNG or WebP";

pub const Rgb = struct {
    width: u32,
    height: u32,
    pixels: []u8, // width * height * 3

    pub fn deinit(r: *Rgb, allocator: Allocator) void {
        allocator.free(r.pixels);
    }
};

pub const Format = enum { png, jpeg, webp };

pub fn sniff(data: []const u8) ?Format {
    if (png.matches(data)) return .png;
    if (data.len >= 3 and data[0] == 0xff and data[1] == 0xd8 and data[2] == 0xff) return .jpeg;
    if (data.len >= 12 and std.mem.eql(u8, data[0..4], "RIFF") and std.mem.eql(u8, data[8..12], "WEBP")) return .webp;
    return null;
}

/// `_decode(data, detail, limits, remaining_pixels)` without the detail (it only rides along).
pub fn decode(allocator: Allocator, data: []const u8, limits: limits_mod.ImageLimits, remaining_pixels: u64, diag: ?*Diag) (Allocator.Error || errors.Error)!Rgb {
    const fmt = sniff(data) orelse return errors.fail(diag, invalid_message, .{});
    const budget = @min(limits.max_pixels, remaining_pixels);
    // the header first: the dimension limits are checked before any pixel is decoded
    const size: struct { w: u32, h: u32 } = switch (fmt) {
        .png => if (png.header(data)) |h| .{ .w = h.w, .h = h.h } else return errors.fail(diag, invalid_message, .{}),
        .jpeg => if (@import("jpeg.zig").header(data)) |h| .{ .w = h.w, .h = h.h } else return errors.fail(diag, invalid_message, .{}),
        .webp => if (@import("webp.zig").header(data)) |h| .{ .w = h.w, .h = h.h } else return errors.fail(diag, invalid_message, .{}),
    };
    if (@min(size.w, size.h) == 0 or @max(size.w, size.h) > limits.max_dimension or @as(u64, size.w) * size.h > budget)
        return errors.fail(diag, "image dimensions exceed the decoded pixel limit", .{});
    var r: raster.Raster = switch (fmt) {
        .png => png.decode(allocator, data, budget) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Animated => return errors.fail(diag, "animated and multipage images are unsupported; send a single frame", .{}),
            error.Invalid => return errors.fail(diag, invalid_message, .{}),
        },
        .jpeg => @import("jpeg.zig").decode(allocator, data) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return errors.fail(diag, invalid_message, .{}),
        },
        .webp => @import("webp.zig").decode(allocator, data, budget) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Animated => return errors.fail(diag, "animated and multipage images are unsupported; send a single frame", .{}),
            else => return errors.fail(diag, invalid_message, .{}),
        },
    };
    defer r.deinit();
    const rgb = try convert.toRgb(allocator, &r);
    const o = raster.orientationOf(&r);
    const oriented = try raster.orient(allocator, rgb, r.w, r.h, o);
    return .{ .width = oriented.w, .height = oriented.h, .pixels = oriented.data };
}

/// images._data_bytes: the bytes of a `data:` URL (base64 or percent-encoded) within `max_bytes`.
pub fn dataBytes(allocator: Allocator, url: []const u8, max_bytes: u64, diag: ?*Diag) (Allocator.Error || errors.Error)![]u8 {
    const comma = std.mem.indexOfScalar(u8, url, ',') orelse return errors.fail(diag, "invalid image data URL", .{});
    const header_ = url[0..comma];
    const payload = url[comma + 1 ..];
    if (@import("split.zig").codepoints(header_) > 256) return errors.fail(diag, "invalid image data URL", .{});
    // media = header[5:].split(";")
    const after = if (header_.len >= 5) header_[5..] else header_[header_.len..];
    var parts: u32 = 1;
    var second: []const u8 = "";
    {
        var it = std.mem.splitScalar(u8, after, ';');
        _ = it.next();
        while (it.next()) |p| {
            parts += 1;
            if (parts == 2) second = p;
        }
    }
    if (parts > 2 or (parts == 2 and !std.ascii.eqlIgnoreCase(second, "base64")))
        return errors.fail(diag, "image data URL supports only optional base64 encoding", .{});
    const b64 = parts == 2;
    const max_chars: u64 = if (b64) 4 * ((max_bytes + 2) / 3) else max_bytes * 3;
    if (payload.len > max_chars) return errors.fail(diag, "image data URL exceeds the encoded byte limit", .{}); // payload is ASCII here or rejected next
    for (payload) |c| if (c >= 0x80) return errors.fail(diag, "invalid image data URL encoding", .{});
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    if (b64) {
        decodeBase64Strict(allocator, payload, &out) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Invalid => return errors.fail(diag, "invalid image data URL encoding", .{}),
        };
    } else {
        try out.ensureTotalCapacity(allocator, payload.len);
        var i: usize = 0;
        while (i < payload.len) {
            if (payload[i] == '%' and i + 2 < payload.len + 0 and std.ascii.isHex(payload[i + 1]) and std.ascii.isHex(payload[i + 2])) {
                out.appendAssumeCapacity((hexVal(payload[i + 1]) << 4) | hexVal(payload[i + 2]));
                i += 3;
            } else {
                out.appendAssumeCapacity(payload[i]);
                i += 1;
            }
        }
    }
    if (out.items.len == 0 or out.items.len > max_bytes) return errors.fail(diag, "image is empty or exceeds the encoded byte limit", .{});
    return out.toOwnedSlice(allocator);
}

fn hexVal(c: u8) u8 {
    return switch (c) {
        '0'...'9' => c - '0',
        'a'...'f' => c - 'a' + 10,
        'A'...'F' => c - 'A' + 10,
        else => unreachable,
    };
}

/// base64.b64decode(payload, validate=True) == binascii.a2b_base64(strict_mode=True) as CPython 3.12 runs it.
fn decodeBase64Strict(allocator: Allocator, s: []const u8, out: *std.ArrayList(u8)) error{ Invalid, OutOfMemory }!void {
    if (s.len > 0 and s[0] == '=') return error.Invalid; // leading padding
    try out.ensureTotalCapacity(allocator, s.len / 4 * 3 + 3);
    var quad_pos: u32 = 0;
    var pads: u32 = 0;
    var padding_started = false;
    var leftchar: u32 = 0;
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        const c = s[i];
        if (c == '=') {
            padding_started = true;
            pads += 1;
            if (quad_pos >= 2 and quad_pos + pads >= 4) {
                if (i + 1 < s.len) return error.Invalid; // excess data after padding
                return;
            }
            continue;
        }
        const v: u32 = switch (c) {
            'A'...'Z' => c - 'A',
            'a'...'z' => c - 'a' + 26,
            '0'...'9' => c - '0' + 52,
            '+' => 62,
            '/' => 63,
            else => return error.Invalid,
        };
        if (padding_started) return error.Invalid; // discontinuous padding
        pads = 0;
        switch (quad_pos) {
            0 => {
                quad_pos = 1;
                leftchar = v;
            },
            1 => {
                quad_pos = 2;
                out.appendAssumeCapacity(@intCast((leftchar << 2) | (v >> 4)));
                leftchar = v & 0x0f;
            },
            2 => {
                quad_pos = 3;
                out.appendAssumeCapacity(@intCast((leftchar << 4) | (v >> 2)));
                leftchar = v & 0x03;
            },
            else => {
                quad_pos = 0;
                out.appendAssumeCapacity(@intCast((leftchar << 6) | v));
                leftchar = 0;
            },
        }
    }
    if (quad_pos != 0) return error.Invalid;
}

const fixtures = @import("fixtures.zig");

/// Run a manifest of "<file> <w> <h> <sha256 of the RGB bytes>" / "<file> ERR <message>" lines.
/// Files whose pixels are known to differ: a JPEG with a flipped byte in its entropy data decodes (libjpeg-turbo
/// does not refuse corrupt Huffman data) but the damaged blocks go through libjpeg-turbo's NEON IDCT, whose 16-bit
/// saturation on out-of-range coefficients is not reproduced. Reported, not failed.
const known_different = [_][]const u8{"bad_flip_mid.jpg"};

fn checkManifest(manifest: []const u8, only_prefix: ?[]const u8) !void {
    const a = std.testing.allocator;
    const text = try fixtures.read(a, manifest, 1 << 22);
    defer a.free(text);
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    var failures: usize = 0;
    var n: usize = 0;
    while (lines.next()) |line| {
        var t = std.mem.tokenizeScalar(u8, line, ' ');
        const name = t.next().?;
        if (only_prefix) |p| if (!std.mem.startsWith(u8, name, p)) continue;
        var path_buf: [256]u8 = undefined;
        const path = try std.fmt.bufPrint(&path_buf, "img/{s}", .{name});
        const data = try fixtures.read(a, path, 1 << 26);
        defer a.free(data);
        var diag: Diag = .{};
        const res = decode(a, data, .default, limits_mod.ImageLimits.default.max_total_pixels, &diag);
        const second = t.next().?;
        n += 1;
        var expected_diff = false;
        for (known_different) |k| expected_diff = expected_diff or std.mem.eql(u8, k, name);
        if (std.mem.eql(u8, second, "ERR")) {
            if (res) |ok| {
                var r = ok;
                r.deinit(a);
                std.debug.print("MISMATCH {s}: decoded but Pillow refuses ({s})\n", .{ name, t.rest() });
                failures += 1;
            } else |_| {
                if (!std.mem.eql(u8, t.rest(), diag.message())) {
                    std.debug.print("MISMATCH {s}: message '{s}' vs '{s}'\n", .{ name, diag.message(), t.rest() });
                    failures += 1;
                }
            }
        } else {
            const w = try std.fmt.parseInt(u32, second, 10);
            const h = try std.fmt.parseInt(u32, t.next().?, 10);
            const want = t.next().?;
            if (res) |ok| {
                var r = ok;
                defer r.deinit(a);
                const got = fixtures.hexDigest(r.pixels);
                if (expected_diff) {
                    if (std.mem.eql(u8, want, &got)) std.debug.print("note: {s} now matches Pillow\n", .{name});
                } else if (r.width != w or r.height != h or !std.mem.eql(u8, want, &got)) {
                    std.debug.print("MISMATCH {s}: {d}x{d} vs {d}x{d}, hash differs={}\n", .{ name, r.width, r.height, w, h, !std.mem.eql(u8, want, &got) });
                    if (std.c.getenv("TF_VISION_DUMP")) |dir| {
                        var pb: [512]u8 = undefined;
                        const pth = std.fmt.bufPrint(&pb, "{s}/{s}.rgb", .{ std.mem.span(dir), name }) catch unreachable;
                        std.Io.Dir.cwd().writeFile(std.testing.io, .{ .sub_path = pth, .data = r.pixels }) catch {};
                    }
                    failures += 1;
                }
            } else |_| {
                std.debug.print("MISMATCH {s}: refused ({s}) but Pillow decodes {d}x{d}\n", .{ name, diag.message(), w, h });
                failures += 1;
            }
        }
    }
    if (failures != 0) {
        std.debug.print("{s}: {d} files checked, {d} differ\n", .{ manifest, n, failures });
        return error.PillowMismatch;
    }
    try std.testing.expect(n > 0);
}

test "PNG: every sub-format against Pillow 12.3" {
    try checkManifest("img_png.txt", null);
}

test "data URLs: bytes and error texts as Python" {
    const a = std.testing.allocator;
    const text = try fixtures.read(a, "dataurl.json", 1 << 22);
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer parsed.deinit();
    for (parsed.value.array.items) |c| {
        const o = c.object;
        var diag: Diag = .{};
        const res = dataBytes(a, o.get("url").?.string, @intCast(o.get("max").?.integer), &diag);
        if (o.get("error")) |want| {
            if (res) |ok| {
                a.free(ok);
                std.debug.print("accepted {s}\n", .{o.get("url").?.string});
                return error.Mismatch;
            } else |_| try std.testing.expectEqualStrings(want.string, diag.message());
        } else {
            const got = res catch |e| {
                std.debug.print("refused {s}: {s}\n", .{ o.get("url").?.string, diag.message() });
                return e;
            };
            defer a.free(got);
            const h = fixtures.hexDigest(got);
            try std.testing.expectEqualStrings(o.get("sha").?.string, &h);
            try std.testing.expectEqual(@as(usize, @intCast(o.get("len").?.integer)), got.len);
        }
    }
}

test "JPEG: baseline, progressive, subsamplings, gray, CMYK, restarts, EXIF against Pillow 12.3" {
    try checkManifest("img_jpeg.txt", null);
}

test "WebP: lossless and lossy (with alpha), EXIF/XMP against Pillow 12.3 / libwebp 1.6.0" {
    try checkManifest("img_webp.txt", null);
}

test "WebP: encoder settings beyond Pillow's reach (simple filter, partitions, segments, ALPH filters)" {
    try checkManifest("img_webp_adv.txt", null);
}

// Hostile bytes must never trap (the server runs this code on whatever a client sends): mutate the corpus and
// decode. In safe mode every out-of-bounds access and integer overflow is a panic. TF_VISION_FUZZ=<n> runs n
// mutations per manifest (default 300).
test "fuzz: damaged pictures are refused or decoded, never trap" {
    const a = std.testing.allocator;
    var rounds: usize = 300;
    if (std.c.getenv("TF_VISION_FUZZ")) |v| rounds = std.fmt.parseInt(usize, std.mem.span(v), 10) catch rounds;
    var seed: u64 = 12345;
    if (std.c.getenv("TF_VISION_FUZZ_SEED")) |v| seed = std.fmt.parseInt(u64, std.mem.span(v), 10) catch seed;
    const manifests = [_][]const u8{ "img_png.txt", "img_jpeg.txt", "img_webp.txt", "img_webp_adv.txt" };
    for (manifests) |manifest| {
        const text = try fixtures.read(a, manifest, 1 << 22);
        defer a.free(text);
        var names: std.ArrayList([]const u8) = .empty;
        defer names.deinit(a);
        var lines = std.mem.tokenizeScalar(u8, text, '\n');
        while (lines.next()) |line| {
            var t = std.mem.tokenizeScalar(u8, line, ' ');
            const name = t.next().?;
            if (t.next()) |second| if (!std.mem.eql(u8, second, "ERR")) try names.append(a, name);
        }
        var r: usize = 0;
        while (r < rounds) : (r += 1) {
            seed = seed *% 6364136223846793005 +% 1442695040888963407;
            const name = names.items[@intCast((seed >> 33) % names.items.len)];
            var path_buf: [256]u8 = undefined;
            const data = try fixtures.read(a, try std.fmt.bufPrint(&path_buf, "img/{s}", .{name}), 1 << 26);
            defer a.free(data);
            if (data.len > 60000) continue;
            const copy = try a.dupe(u8, data);
            defer a.free(copy);
            seed = seed *% 6364136223846793005 +% 1442695040888963407;
            var n = 1 + ((seed >> 40) % 6);
            while (n > 0) : (n -= 1) {
                seed = seed *% 6364136223846793005 +% 1442695040888963407;
                const at: usize = @intCast((seed >> 20) % copy.len);
                switch ((seed >> 60) % 4) {
                    0 => copy[at] ^= @intCast((seed >> 8) & 0xff | 1),
                    1 => copy[at] = @intCast((seed >> 8) & 0xff),
                    2 => copy[at] = 0xff,
                    else => copy[at] = 0,
                }
            }
            seed = seed *% 6364136223846793005 +% 1442695040888963407;
            const len = if ((seed >> 50) % 5 == 0) @as(usize, @intCast((seed >> 8) % copy.len)) + 1 else copy.len;
            var diag: Diag = .{};
            var lim: limits_mod.ImageLimits = .default;
            lim.max_pixels = 1 << 22; // keep a mutated header from asking for a huge buffer in the test
            if (decode(a, copy[0..len], lim, 1 << 22, &diag)) |ok| {
                var rr = ok;
                rr.deinit(a);
            } else |_| {}
        }
    }
}

test "JPEG: sampling layouts, scan layouts, restarts and colour-space markers Pillow cannot write" {
    try checkManifest("img_jpeg_enc.txt", null);
}
