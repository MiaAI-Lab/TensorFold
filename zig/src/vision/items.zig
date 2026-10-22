//! A request's prepared media: pictures (a remembered record or a fresh canvas) and clips, their feature-row
//! counts and the content hashes / 62-bit item keys of glm.py (`GlmPrepared.item_keys`, `content_hash`).
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;
const geometry = @import("geometry.zig");

pub const Detail = enum {
    auto,
    low,
    high,

    pub fn parse(s: []const u8) ?Detail {
        return std.meta.stringToEnum(Detail, s);
    }
};

/// A reference-counted byte buffer (a remembered picture's canvas, shared between the cache and its users).
pub const Shared = struct {
    refs: std.atomic.Value(u32),
    bytes: []u8,
    allocator: std.mem.Allocator,

    /// Takes ownership of `bytes` (allocated with `allocator`). One reference.
    pub fn create(allocator: std.mem.Allocator, bytes: []u8) !*Shared {
        const s = try allocator.create(Shared);
        s.* = .{ .refs = .init(1), .bytes = bytes, .allocator = allocator };
        return s;
    }

    pub fn retain(s: *Shared) *Shared {
        _ = s.refs.fetchAdd(1, .monotonic);
        return s;
    }

    pub fn release(s: *Shared) void {
        if (s.refs.fetchSub(1, .acq_rel) == 1) {
            const a = s.allocator;
            a.free(s.bytes);
            a.destroy(s);
        }
    }
};

/// glm.py FittedPicture: what reading a picture once cost, remembered by the source's bytes.
pub const Picture = struct {
    rows: u32, // placeholder tokens: (H // 28) * (W // 28)
    key: u64, // 62-bit key of kind, shape and pixels
    shape: [3]u32, // the fitted canvas: 3, H, W
    pixels: u64, // the decoded picture's width * height (the request's pixel budget counts it)
    digest: [32]u8, // sha256 of the canvas
    detail: Detail = .auto,
    tokens: u32 = 0,
    min_tokens: u32 = 16,
    canvas: ?*Shared = null, // the fitted canvas while it is still in hand (retained)
    data: ?*Shared = null, // the source's bytes, to fit it again if its canvas has been let go (retained)

    pub fn deinit(p: *Picture) void {
        if (p.canvas) |c| c.release();
        if (p.data) |d| d.release();
        p.canvas = null;
        p.data = null;
    }

    /// A second owner of the same record (canvas and bytes retained).
    pub fn retained(p: Picture) Picture {
        var out = p;
        if (p.canvas) |c| out.canvas = c.retain();
        if (p.data) |d| out.data = d.retain();
        return out;
    }
};

/// A fitted picture, uint8 [3, H, W] (right/bottom zero padded).
pub const Canvas = struct {
    data: []u8,
    h: u32,
    w: u32,
};

/// A clip's sampled frames at the tower's canvas: uint8 [T, 3, H, W], and each frame pair's first time.
pub const Clip = struct {
    frames: []u8,
    t: u32,
    h: u32,
    w: u32,
    seconds: []f64, // t / 2 entries
};

pub const Item = union(enum) {
    picture: Picture,
    canvas: Canvas,
    clip: Clip,

    pub fn kind(i: Item) []const u8 {
        return if (i == .clip) "video" else "image";
    }

    /// `GlmVision._rows`: feature rows (= placeholder tokens) of the item.
    pub fn rows(i: Item) u64 {
        return switch (i) {
            .picture => |p| p.rows,
            .canvas => |c| geometry.rowsPerFrame(c.h, c.w),
            .clip => |c| @as(u64, c.t / 2) * geometry.rowsPerFrame(c.h, c.w),
        };
    }

    pub fn deinit(i: *Item, allocator: std.mem.Allocator) void {
        switch (i.*) {
            .picture => |*p| p.deinit(),
            .canvas => |c| allocator.free(c.data),
            .clip => |c| {
                allocator.free(c.frames);
                allocator.free(c.seconds);
            },
        }
    }
};

/// Python `json.dumps(list(shape))`: "[3, 56, 56]".
pub fn shapeJson(buf: []u8, shape: []const u32) []const u8 {
    var w = std.Io.Writer.fixed(buf);
    w.writeByte('[') catch unreachable;
    for (shape, 0..) |s, i| {
        if (i != 0) w.writeAll(", ") catch unreachable;
        w.print("{d}", .{s}) catch unreachable;
    }
    w.writeByte(']') catch unreachable;
    return w.buffered();
}

/// `int.from_bytes(sha256(head + b"\0" + raw).digest()[:8], "big") >> 2`
pub fn itemKey(head: []const u8, raw: []const u8) u64 {
    var h = Sha256.init(.{});
    h.update(head);
    h.update("\x00");
    h.update(raw);
    var d: [32]u8 = undefined;
    h.final(&d);
    return std.mem.readInt(u64, d[0..8], .big) >> 2;
}

/// The record `_remember` builds for a freshly fitted canvas (`canvas` is retained, not copied).
pub fn rememberPicture(canvas: *Shared, h: u32, w: u32, pixels: u64, detail: Detail, tokens: u32, min_tokens: u32) Picture {
    var sb: [64]u8 = undefined;
    const shape = [3]u32{ 3, h, w };
    var head_buf: [80]u8 = undefined;
    const sj = shapeJson(&sb, &shape);
    const head = std.fmt.bufPrint(&head_buf, "image{s}", .{sj}) catch unreachable;
    var digest: [32]u8 = undefined;
    Sha256.hash(canvas.bytes, &digest, .{});
    return .{
        .rows = @intCast(geometry.rowsPerFrame(h, w)),
        .key = itemKey(head, canvas.bytes),
        .shape = shape,
        .pixels = pixels,
        .digest = digest,
        .detail = detail,
        .tokens = tokens,
        .min_tokens = min_tokens,
        .canvas = canvas.retain(),
    };
}

pub const ItemHashes = struct {
    content_hash: [64]u8, // hex sha256 over the items (the prompt's content hash)
    keys: []u64,
    rows: []u32,
};

/// glm.py `prepare`'s tail: content hash, each item's 62-bit key and row count. `items` are in prompt order.
pub fn hashItems(allocator: std.mem.Allocator, items: []const Item) !ItemHashes {
    var digest = Sha256.init(.{});
    digest.update("tensorfold-glm-vision-v1\x00");
    const keys = try allocator.alloc(u64, items.len);
    errdefer allocator.free(keys);
    const rows = try allocator.alloc(u32, items.len);
    errdefer allocator.free(rows);
    for (items, keys, rows) |it, *key, *row| {
        row.* = @intCast(it.rows());
        var sb: [64]u8 = undefined;
        var head_buf: [96]u8 = undefined;
        switch (it) {
            .picture => |p| {
                const head = std.fmt.bufPrint(&head_buf, "image{s}", .{shapeJson(&sb, &p.shape)}) catch unreachable;
                digest.update(head);
                digest.update(&p.digest);
                key.* = p.key;
            },
            .canvas => |c| {
                const shape = [3]u32{ 3, c.h, c.w };
                const head = std.fmt.bufPrint(&head_buf, "image{s}", .{shapeJson(&sb, &shape)}) catch unreachable;
                digest.update(head);
                digest.update(c.data);
                key.* = itemKey(head, c.data);
            },
            .clip => |c| {
                const shape = [4]u32{ c.t, 3, c.h, c.w };
                const head = std.fmt.bufPrint(&head_buf, "video{s}", .{shapeJson(&sb, &shape)}) catch unreachable;
                digest.update(head);
                digest.update(c.frames);
                key.* = itemKey(head, c.frames);
            },
        }
    }
    var d: [32]u8 = undefined;
    digest.final(&d);
    return .{ .content_hash = std.fmt.bytesToHex(d, .lower), .keys = keys, .rows = rows };
}

/// glm.py `GlmClip.content_hash`: sha256(b"tensorfold-glm-clip-v1\0" + dumps(shape) + frames), hex.
pub fn clipContentHash(c: Clip) [64]u8 {
    var h = Sha256.init(.{});
    h.update("tensorfold-glm-clip-v1\x00");
    var sb: [64]u8 = undefined;
    const shape = [4]u32{ c.t, 3, c.h, c.w };
    h.update(shapeJson(&sb, &shape));
    h.update(c.frames);
    var d: [32]u8 = undefined;
    h.final(&d);
    return std.fmt.bytesToHex(d, .lower);
}

test "shape json and hashing structure" {
    var b: [64]u8 = undefined;
    try std.testing.expectEqualStrings("[3, 56, 56]", shapeJson(&b, &.{ 3, 56, 56 }));
    try std.testing.expectEqualStrings("[2, 3, 28, 28]", shapeJson(&b, &.{ 2, 3, 28, 28 }));
}
