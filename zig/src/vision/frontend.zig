//! The vision frontend the server calls: load sources (data URLs, HTTPS through a Fetcher), decode, fit to the
//! tower's canvas, remember pictures by their source bytes (glm.py `_pictures` / `_canvases`), expand the prompt.
const std = @import("std");
const errors = @import("errors.zig");
const limits_mod = @import("limits.zig");
const geometry = @import("geometry.zig");
const items_mod = @import("items.zig");
const images = @import("images.zig");
const resize = @import("resize.zig");
const av = @import("av.zig");
const clip_mod = @import("clip.zig");
const split = @import("split.zig");
const expand_mod = @import("expand.zig");
const Allocator = std.mem.Allocator;
const Diag = errors.Diag;
const Item = items_mod.Item;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const FetchRequest = struct {
    url: []const u8,
    max_bytes: u64,
    deadline_ns: i96, // io monotonic clock (Clock.awake)
    max_redirects: u32,
    max_url_chars: u32,
    video: bool, // media types: video/* rather than image/*
};

/// Retrieves the bytes of an HTTPS media URL with the recipe's SSRF protections (see fetch.zig).
pub const Fetcher = struct {
    ctx: ?*anyopaque = null,
    fetch: *const fn (ctx: ?*anyopaque, allocator: Allocator, io: std.Io, req: FetchRequest, diag: ?*Diag) (Allocator.Error || errors.Error)![]u8,
};

pub const Options = struct {
    limits: limits_mod.GlmVisionLimits = .{},
    cache: limits_mod.PictureCacheConfig = .{},
    video: limits_mod.VideoLimits = .default,
    allow_urls: bool = false,
    fetcher: ?Fetcher = null,

    /// The TENSORFOLD_GLM_* knobs (GlmVisionLimits.from_env and the picture-cache settings), validated as the
    /// Python frontend does; a bad value is an `error.Input` whose message is the ValueError text.
    pub fn fromEnv(getenv: limits_mod.Getenv, diag: ?*Diag) errors.Error!Options {
        return .{ .limits = try limits_mod.GlmVisionLimits.fromEnv(getenv, diag), .cache = try limits_mod.PictureCacheConfig.fromEnv(getenv, diag) };
    }
};

const CacheKey = struct {
    digest: [32]u8,
    detail: items_mod.Detail,
    tokens: u32,
    min_tokens: u32,

    fn eql(a: CacheKey, b: CacheKey) bool {
        return a.detail == b.detail and a.tokens == b.tokens and a.min_tokens == b.min_tokens and std.mem.eql(u8, &a.digest, &b.digest);
    }
};

const Rec = struct {
    key: CacheKey,
    picture: items_mod.Picture, // the cache's own references
    in_pictures: bool = false,
    in_canvases: bool = false,
};

pub const Frontend = struct {
    allocator: Allocator,
    io: std.Io,
    opts: Options,
    image_limits: limits_mod.ImageLimits,
    mutex: std.Io.Mutex = .init,
    pictures: std.ArrayList(*Rec) = .empty, // LRU by source bytes, oldest first
    canvases: std.ArrayList(*Rec) = .empty, // the last few fitted canvases held, oldest first
    av_libs: ?av.Libs = null, // FFmpeg, loaded on the first clip
    av_failed: bool = false,

    pub fn init(allocator: Allocator, io: std.Io, opts: Options) Frontend {
        return .{ .allocator = allocator, .io = io, .opts = opts, .image_limits = limits_mod.ImageLimits.forGlm(opts.limits) };
    }

    pub fn deinit(f: *Frontend) void {
        while (f.pictures.pop()) |r| {
            r.in_pictures = false;
            f.maybeFree(r);
        }
        f.pictures.deinit(f.allocator);
        while (f.canvases.pop()) |r| {
            r.in_canvases = false;
            f.maybeFree(r);
        }
        f.canvases.deinit(f.allocator);
        if (f.av_libs) |*l| l.close();
    }

    fn maybeFree(f: *Frontend, r: *Rec) void {
        if (r.in_pictures or r.in_canvases) return;
        r.picture.deinit();
        f.allocator.destroy(r);
    }

    fn now(f: *Frontend) i96 {
        return std.Io.Clock.awake.now(f.io).toNanoseconds();
    }

    fn secondsToNs(s: f64) i96 {
        return @intFromFloat(s * 1e9);
    }

    // ---- the picture cache --------------------------------------------------------------------------------------
    fn lookup(f: *Frontend, key: CacheKey) ?items_mod.Picture {
        f.mutex.lockUncancelable(f.io);
        defer f.mutex.unlock(f.io);
        for (f.pictures.items, 0..) |r, i| {
            if (r.key.eql(key)) {
                _ = f.pictures.orderedRemove(i);
                f.pictures.appendAssumeCapacity(r); // move_to_end
                return r.picture.retained();
            }
        }
        return null;
    }

    fn totalBytes(f: *Frontend) u64 {
        var n: u64 = 0;
        for (f.pictures.items) |r| n += r.picture.data.?.bytes.len;
        return n;
    }

    /// `_remember`: store a freshly fitted picture; returns the request's own reference to the record.
    fn remember(f: *Frontend, key: CacheKey, picture: items_mod.Picture, budget: u64) Allocator.Error!items_mod.Picture {
        f.mutex.lockUncancelable(f.io);
        defer f.mutex.unlock(f.io);
        const rec = try f.allocator.create(Rec);
        errdefer f.allocator.destroy(rec);
        rec.* = .{ .key = key, .picture = picture };
        try f.pictures.ensureUnusedCapacity(f.allocator, 1);
        try f.canvases.ensureUnusedCapacity(f.allocator, 1);
        const mine = rec.picture.retained();
        // a decode squeezed by the pixel budget is not kept by its bytes
        if (budget >= limits_mod.PictureCacheConfig.uncapped) {
            var replaced = false;
            for (f.pictures.items, 0..) |r, i| if (r.key.eql(key)) {
                // dict assignment to an existing key keeps its place
                r.in_pictures = false;
                f.pictures.items[i] = rec;
                rec.in_pictures = true;
                f.maybeFree(r);
                replaced = true;
                break;
            };
            if (!replaced) {
                f.pictures.appendAssumeCapacity(rec);
                rec.in_pictures = true;
            }
            var held = f.totalBytes();
            const cap = @as(u64, f.opts.cache.mb) * 1024 * 1024;
            while (held > cap and f.pictures.items.len > 1) {
                const old = f.pictures.orderedRemove(0);
                held -= old.picture.data.?.bytes.len;
                old.in_pictures = false;
                f.maybeFree(old);
            }
        }
        // `_hold_canvas`: keep the last few canvases in hand
        if (f.opts.cache.canvases != 0) {
            f.canvases.appendAssumeCapacity(rec);
            rec.in_canvases = true;
            while (f.canvases.items.len > f.opts.cache.canvases) {
                const old = f.canvases.orderedRemove(0);
                old.in_canvases = false;
                if (old != rec) {
                    if (old.picture.canvas) |c| c.release();
                    old.picture.canvas = null;
                }
                f.maybeFree(old);
            }
        }
        f.maybeFree(rec); // neither list kept it
        return mine;
    }

    // ---- fitting ----------------------------------------------------------------------------------------------------
    /// `_picture(image, tokens)`: the RGB picture fitted to its canvas (uint8 [3, H, W]).
    fn fitPicture(f: *Frontend, a: Allocator, rgb: images.Rgb, tokens: u32, diag: ?*Diag) (Allocator.Error || errors.Error)!items_mod.Canvas {
        const min_tokens = f.opts.limits.min_tokens;
        const canvas = try geometry.smartResize(2, rgb.height, rgb.width, tokens, min_tokens, diag);
        const content = geometry.contentSize(2, rgb.height, rgb.width, canvas, min_tokens);
        const data = try a.alloc(u8, 3 * @as(usize, canvas.h) * canvas.w);
        errdefer a.free(data);
        try resize.fitInto(a, rgb.pixels, rgb.height, rgb.width, canvas, content, data);
        return .{ .data = data, .h = canvas.h, .w = canvas.w };
    }

    fn fetchBytes(f: *Frontend, a: Allocator, url: []const u8, max_bytes: u64, deadline_ns: i96, video: bool, diag: ?*Diag) (Allocator.Error || errors.Error)![]u8 {
        const fetcher = f.opts.fetcher orelse return errors.fail(diag, "{s} URLs are not available in this build", .{if (video) "video" else "image"});
        const limits = if (video) .{ f.opts.video.timeout_seconds, f.opts.video.max_redirects, f.opts.video.max_url_chars } else .{ f.image_limits.timeout_seconds, f.image_limits.max_redirects, f.image_limits.max_url_chars };
        return fetcher.fetch(fetcher.ctx, a, f.io, .{
            .url = url,
            .max_bytes = max_bytes,
            .deadline_ns = @min(deadline_ns, f.now() + secondsToNs(limits[0])),
            .max_redirects = limits[1],
            .max_url_chars = limits[2],
            .video = video,
        }, diag);
    }

    /// `load_images`: the request's pictures, each decoded and fitted in turn. Items are owned by the caller
    /// (`Item.deinit` each; a `.picture` holds references into the cache's buffers).
    pub fn loadImages(f: *Frontend, a: Allocator, sources: []const split.ImageSource, diag: ?*Diag) (Allocator.Error || errors.Error)![]Item {
        const L = f.image_limits;
        if (sources.len > L.max_images) return errors.fail(diag, "a request supports at most {d} images", .{L.max_images});
        const tokens = f.opts.limits.pictureTokens(sources.len);
        var total_bytes: u64 = 0;
        var total_pixels: u64 = 0;
        var out: std.ArrayList(Item) = .empty;
        errdefer {
            for (out.items) |*it| it.deinit(a);
            out.deinit(a);
        }
        const deadline = f.now() + secondsToNs(L.total_timeout_seconds);
        for (sources) |source| {
            try split.checkImageUrl(source.url, L, f.opts.allow_urls, diag);
            const room: i64 = @as(i64, @intCast(L.max_total_encoded_bytes)) - @as(i64, @intCast(total_bytes));
            const remaining: i64 = @min(@as(i64, @intCast(L.max_encoded_bytes)), room);
            if (remaining <= 0 or f.now() >= deadline) return errors.fail(diag, "image request exceeds the total byte or time limit", .{});
            const max_bytes: u64 = @intCast(remaining);
            const data = if (std.mem.startsWith(u8, source.url, "data:"))
                try images.dataBytes(f.allocator, source.url, max_bytes, diag)
            else
                try f.fetchBytes(f.allocator, source.url, max_bytes, deadline, false, diag);
            var data_owned = true;
            errdefer if (data_owned) f.allocator.free(data);
            total_bytes += data.len;
            const budget: u64 = if (L.max_total_pixels > total_pixels) L.max_total_pixels - total_pixels else 0;
            var key: CacheKey = .{ .digest = undefined, .detail = source.detail, .tokens = tokens, .min_tokens = f.opts.limits.min_tokens };
            Sha256.hash(data, &key.digest, .{});
            if (f.opts.cache.enabled) {
                if (f.lookup(key)) |seen| {
                    f.allocator.free(data);
                    data_owned = false;
                    total_pixels += seen.pixels;
                    try out.append(a, .{ .picture = seen });
                    if (f.now() >= deadline) return errors.fail(diag, "image request exceeds the total time limit", .{});
                    continue;
                }
            }
            var rgb = try images.decode(a, data, L, budget, diag);
            defer rgb.deinit(a);
            total_pixels += @as(u64, rgb.width) * rgb.height;
            var canvas = try f.fitPicture(a, rgb, tokens, diag);
            if (!f.opts.cache.enabled) {
                f.allocator.free(data);
                data_owned = false;
                try out.append(a, .{ .canvas = canvas });
            } else {
                // the cache owns a copy of the canvas bytes (its allocator), the request gets its own reference
                const owned = try f.allocator.dupe(u8, canvas.data);
                a.free(canvas.data);
                canvas.data = &.{};
                const shared_canvas = try items_mod.Shared.create(f.allocator, owned);
                const shared_data = try items_mod.Shared.create(f.allocator, data);
                data_owned = false;
                var rec = items_mod.rememberPicture(shared_canvas, canvas.h, canvas.w, @as(u64, rgb.width) * rgb.height, source.detail, tokens, f.opts.limits.min_tokens);
                shared_canvas.release(); // rememberPicture retained its own
                rec.data = shared_data;
                const mine = blk: {
                    errdefer rec.deinit();
                    break :blk try f.remember(key, rec, budget);
                };
                try out.append(a, .{ .picture = mine });
            }
            if (f.now() >= deadline) return errors.fail(diag, "image request exceeds the total time limit", .{});
        }
        return out.toOwnedSlice(a);
    }

    /// The options `split.splitImages` takes for this frontend (GLM: up to 50 pictures and 4 clips a request, clips
    /// and tool-result media allowed, the frontend's own limits).
    pub fn splitOptions(f: *const Frontend) split.Options {
        return .{
            .limits = f.image_limits,
            .video_limits = f.opts.video,
            .allow_urls = f.opts.allow_urls,
            .allow_videos = true,
            .max_videos = f.opts.limits.max_videos,
            .tool_media = true,
        };
    }

    pub const Media = struct {
        pictures: []Item,
        clips: []Item,

        pub fn deinit(m: *Media, a: Allocator) void {
            for (m.pictures) |*it| it.deinit(a);
            for (m.clips) |*it| it.deinit(a);
            a.free(m.pictures);
            a.free(m.clips);
        }
    };

    /// `prepare_images`' loading half: the sources in the order the template rendered them (`order` from
    /// `MediaMarks.read`, or null), pictures and clips loaded and fitted. Feed `pictures` and `clips` to `expand`.
    pub fn loadSources(f: *Frontend, a: Allocator, sources: []const split.Source, order: ?[]const usize, diag: ?*Diag) (Allocator.Error || errors.Error)!Media {
        const part = try split.partition(a, sources, order);
        defer {
            a.free(part.images);
            a.free(part.videos);
        }
        const pictures = try f.loadImages(a, part.images, diag);
        errdefer {
            for (pictures) |*it| it.deinit(a);
            a.free(pictures);
        }
        const clips = if (part.videos.len != 0) try f.loadVideos(a, part.videos, diag) else try a.alloc(Item, 0);
        return .{ .pictures = pictures, .clips = clips };
    }

    /// A picture's canvas bytes: the one in hand, or the source fitted again (`GlmVision._canvas`).
    pub fn canvasOf(f: *Frontend, a: Allocator, p: *items_mod.Picture, diag: ?*Diag) (Allocator.Error || errors.Error)![]const u8 {
        if (p.canvas) |c| return c.bytes;
        const data = p.data orelse return errors.fail(diag, "the picture's pixels are no longer available", .{});
        var rgb = try images.decode(a, data.bytes, f.image_limits, f.image_limits.max_total_pixels, diag);
        defer rgb.deinit(a);
        const canvas = try f.fitPicture(a, rgb, p.tokens, diag);
        const shared = try items_mod.Shared.create(f.allocator, try f.allocator.dupe(u8, canvas.data));
        a.free(canvas.data);
        p.canvas = shared;
        return shared.bytes;
    }

    /// The pixels of an image item, as uint8 [3, H, W].
    pub fn imagePixels(f: *Frontend, a: Allocator, it: *Item, diag: ?*Diag) (Allocator.Error || errors.Error)![]const u8 {
        return switch (it.*) {
            .canvas => |c| c.data,
            .picture => |*p| try f.canvasOf(a, p, diag),
            .clip => unreachable,
        };
    }

    fn libs(f: *Frontend, diag: ?*Diag) errors.Error!*const av.Libs {
        f.mutex.lockUncancelable(f.io);
        defer f.mutex.unlock(f.io);
        if (f.av_libs) |*l| return l;
        if (!f.av_failed) {
            if (av.load(f.allocator, f.io)) |l| {
                f.av_libs = l;
                return &f.av_libs.?;
            } else |_| f.av_failed = true;
        }
        return errors.fail(diag, "video inputs require FFmpeg's shared libraries (PyAV's av.libs, or a directory named by TENSORFOLD_AV_LIBDIR); none could be loaded", .{});
    }

    /// `load_videos`: the request's clips decoded at their canvases (items own their frames).
    pub fn loadVideos(f: *Frontend, a: Allocator, sources: []const split.VideoSource, diag: ?*Diag) (Allocator.Error || errors.Error)![]Item {
        const L = f.opts.video;
        if (sources.len > f.opts.limits.max_videos) return errors.fail(diag, "a request supports at most {d} videos", .{f.opts.limits.max_videos});
        var limits = f.opts.limits;
        limits.video_tokens = limits.clipTokens(sources.len);
        var total: u64 = 0;
        var out: std.ArrayList(Item) = .empty;
        errdefer {
            for (out.items) |*it| it.deinit(a);
            out.deinit(a);
        }
        const deadline = f.now() + secondsToNs(L.total_timeout_seconds);
        for (sources) |source| {
            const room: i64 = @as(i64, @intCast(L.max_total_encoded_bytes)) - @as(i64, @intCast(total));
            const remaining: i64 = @min(@as(i64, @intCast(L.max_encoded_bytes)), room);
            if (remaining <= 0 or f.now() >= deadline) return errors.fail(diag, "video request exceeds the total byte or time limit", .{});
            const max_bytes: u64 = @intCast(remaining);
            const data = if (std.mem.startsWith(u8, source.url, "data:"))
                try images.dataBytes(f.allocator, source.url, max_bytes, diag)
            else if (f.opts.allow_urls)
                try f.fetchBytes(f.allocator, source.url, max_bytes, deadline, true, diag)
            else
                return errors.fail(diag, "video URLs are off on this server", .{});
            defer f.allocator.free(data);
            total += data.len;
            const l = try f.libs(diag);
            const clip = try clip_mod.decodeClip(a, f.io, l, data, .{ .limits = limits, .max_dimension = L.max_dimension, .max_seconds = L.max_seconds }, deadline, diag);
            try out.append(a, .{ .clip = clip });
        }
        return out.toOwnedSlice(a);
    }

    // ---- prompt ---------------------------------------------------------------------------------------------------
    /// `GlmVision.escape_quoted` / `MediaMarks.read(..., escape)` helpers are in expand.zig and split.zig.
    pub fn expand(f: *Frontend, a: Allocator, text: []const u8, pictures: []const Item, clips: []const Item, diag: ?*Diag) (Allocator.Error || errors.Error)!expand_mod.Expanded {
        _ = f;
        return expand_mod.expand(a, text, pictures, clips, diag);
    }

    pub fn finish(f: *Frontend, a: Allocator, ids: []const u32, image_token: u32, ordered: []const Item, max_prompt_tokens: ?usize, diag: ?*Diag) (Allocator.Error || errors.Error)!expand_mod.Prepared {
        _ = f;
        return expand_mod.finish(a, ids, image_token, ordered, max_prompt_tokens, diag);
    }
};

fn sampleDataUrl(a: Allocator, w: usize, h: usize) ![]u8 {
    // a PNG of the LCG bytes, written with stored (uncompressed) deflate blocks
    const raw_len = (w * 3 + 1) * h;
    const raw = try a.alloc(u8, raw_len);
    defer a.free(raw);
    var seed_bytes = try a.alloc(u8, w * 3 * h);
    defer a.free(seed_bytes);
    @import("fixtures.zig").lcgBytes(seed_bytes, 77);
    for (0..h) |y| {
        raw[y * (w * 3 + 1)] = 0;
        @memcpy(raw[y * (w * 3 + 1) + 1 ..][0 .. w * 3], seed_bytes[y * w * 3 ..][0 .. w * 3]);
    }
    var z: std.ArrayList(u8) = .empty;
    defer z.deinit(a);
    try z.appendSlice(a, &.{ 0x78, 0x01 });
    var off: usize = 0;
    while (off < raw.len) {
        const n = @min(raw.len - off, 65535);
        const final: u8 = if (off + n == raw.len) 1 else 0;
        try z.append(a, final);
        try z.appendSlice(a, &std.mem.toBytes(@as(u16, @intCast(n))));
        try z.appendSlice(a, &std.mem.toBytes(~@as(u16, @intCast(n))));
        try z.appendSlice(a, raw[off .. off + n]);
        off += n;
    }
    try z.appendSlice(a, &.{ 0, 0, 0, 0 }); // adler (not checked)
    var png: std.ArrayList(u8) = .empty;
    defer png.deinit(a);
    try png.appendSlice(a, "\x89PNG\r\n\x1a\n");
    const Crc = std.hash.crc.@"CRC-32/ISO-HDLC";
    const chunk = struct {
        fn put(list: *std.ArrayList(u8), al: Allocator, typ: []const u8, body: []const u8) !void {
            var len: [4]u8 = undefined;
            std.mem.writeInt(u32, &len, @intCast(body.len), .big);
            try list.appendSlice(al, &len);
            try list.appendSlice(al, typ);
            try list.appendSlice(al, body);
            var c = Crc.init();
            c.update(typ);
            c.update(body);
            var crc: [4]u8 = undefined;
            std.mem.writeInt(u32, &crc, c.final(), .big);
            try list.appendSlice(al, &crc);
        }
    }.put;
    var ihdr: [13]u8 = undefined;
    std.mem.writeInt(u32, ihdr[0..4], @intCast(w), .big);
    std.mem.writeInt(u32, ihdr[4..8], @intCast(h), .big);
    ihdr[8] = 8;
    ihdr[9] = 2;
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;
    try chunk(&png, a, "IHDR", &ihdr);
    try chunk(&png, a, "IDAT", z.items);
    try chunk(&png, a, "IEND", "");
    const prefix = "data:image/png;base64,";
    const enc = std.base64.standard.Encoder;
    const out = try a.alloc(u8, prefix.len + enc.calcSize(png.items.len));
    @memcpy(out[0..prefix.len], prefix);
    _ = enc.encode(out[prefix.len..], png.items);
    return out;
}

test "frontend: load, fit, remember, hit, evict" {
    const a = std.testing.allocator;
    var f = Frontend.init(a, std.testing.io, .{ .cache = .{ .mb = 16, .canvases = 1 } });
    defer f.deinit();
    const url1 = try sampleDataUrl(a, 60, 40);
    defer a.free(url1);
    const url2 = try sampleDataUrl(a, 41, 83);
    defer a.free(url2);
    var diag: Diag = .{};
    const srcs = [_]split.ImageSource{ .{ .url = url1, .detail = .auto }, .{ .url = url2, .detail = .low } };
    const first = try f.loadImages(a, &srcs, &diag);
    defer {
        for (first) |*it| @constCast(it).deinit(a);
        a.free(first);
    }
    try std.testing.expectEqual(@as(usize, 2), first.len);
    try std.testing.expect(first[0] == .picture);
    // 60x40 at 2 pictures -> tokens 2048: canvas 84x56? (multiples of 28)
    try std.testing.expect(first[0].picture.shape[1] % 28 == 0 and first[0].picture.shape[2] % 28 == 0);
    const keys = .{ first[0].picture.key, first[1].picture.key };
    // again: both hit the cache, same keys and digests
    const second = try f.loadImages(a, &srcs, &diag);
    defer {
        for (second) |*it| @constCast(it).deinit(a);
        a.free(second);
    }
    try std.testing.expectEqual(keys[0], second[0].picture.key);
    try std.testing.expectEqual(keys[1], second[1].picture.key);
    try std.testing.expect(std.mem.eql(u8, &first[0].picture.digest, &second[0].picture.digest));
    // the canvas of the older picture was let go (canvases = 1) but is fitted again from its bytes
    var again = second[0].picture.retained();
    defer again.deinit();
    try std.testing.expect(again.canvas == null);
    const px = try f.canvasOf(a, &again, &diag);
    try std.testing.expectEqual(3 * @as(usize, first[0].picture.shape[1]) * first[0].picture.shape[2], px.len);
    // too many images
    var many: [51]split.ImageSource = undefined;
    for (&many) |*m| m.* = .{ .url = url1, .detail = .auto };
    try std.testing.expectError(error.Input, f.loadImages(a, &many, &diag));
    try std.testing.expectEqualStrings("a request supports at most 50 images", diag.message());
}

/// prompts.py `image_slot`: requests decoding pictures at once (host memory), waiting requests, and the wait.
/// A refusal is `error.Capacity` (HTTP 503) with the Python text in the diag.
pub const Slots = struct {
    sem: std.Io.Semaphore,
    waiters: std.atomic.Value(u32) = .init(0),
    max_waiters: u32,
    wait_ns: i96,

    pub const Error = error{Capacity};

    pub fn init(slots: usize, max_waiters: u32, wait_seconds: f64) Slots {
        return .{ .sem = .{ .permits = slots }, .max_waiters = max_waiters, .wait_ns = @intFromFloat(wait_seconds * 1e9) };
    }

    /// IMAGE_SLOTS = 16, IMAGE_WAITERS = 128, IMAGE_WAIT_S = 60.
    pub fn default() Slots {
        return init(16, 128, 60.0);
    }

    pub fn acquire(s: *Slots, io: std.Io, diag: ?*Diag) Error!void {
        if (s.waiters.fetchAdd(1, .acq_rel) >= s.max_waiters) {
            _ = s.waiters.fetchSub(1, .acq_rel);
            if (diag) |d| d.set("image request queue is full; retry shortly", .{});
            return error.Capacity;
        }
        defer _ = s.waiters.fetchSub(1, .acq_rel);
        s.sem.waitTimeout(io, .{ .duration = .{ .raw = .fromNanoseconds(s.wait_ns), .clock = .awake } }) catch {
            if (diag) |d| d.set("image processing capacity is busy; retry shortly", .{});
            return error.Capacity;
        };
    }

    pub fn release(s: *Slots, io: std.Io) void {
        s.sem.post(io);
    }
};

test "slots refuse past the waiting room" {
    var s = Slots.init(1, 1, 0.05);
    var diag: Diag = .{};
    try s.acquire(std.testing.io, &diag);
    // the only slot is taken: the next one waits, times out, and is told the capacity is busy
    try std.testing.expectError(error.Capacity, s.acquire(std.testing.io, &diag));
    try std.testing.expectEqualStrings("image processing capacity is busy; retry shortly", diag.message());
    s.release(std.testing.io);
    try s.acquire(std.testing.io, &diag);
    s.release(std.testing.io);
}

test "a request end to end: split, marks, load, expand, finish" {
    const a = std.testing.allocator;
    const json = @import("json");
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    const url1 = try sampleDataUrl(aa, 60, 40);
    const url2 = try sampleDataUrl(aa, 28, 56);
    const body = try std.fmt.allocPrint(aa,
        \\[{{"role":"user","content":[{{"type":"text","text":"look at these"}},{{"type":"image_url","image_url":{{"url":"{s}"}}}},{{"type":"text","text":"and"}},{{"type":"image_url","image_url":{{"url":"{s}","detail":"low"}}}}]}}]
    , .{ url1, url2 });
    const messages = switch (try json.parse(aa, body)) {
        .ok => |v| v,
        .err => return error.BadJson,
    };
    var f = Frontend.init(a, std.testing.io, .{});
    defer f.deinit();
    var diag: Diag = .{};
    const sp = try split.splitImages(aa, messages, f.splitOptions(), &diag);
    try std.testing.expectEqual(@as(usize, 2), sp.sources.len);
    const marks = try split.MediaMarks.init(aa, sp.messages, .{ 1, 2, 3, 4, 5, 6, 7, 8 });
    try std.testing.expectEqual(@as(usize, 2), marks.count);
    // what the chat template would render: each part's mark, then the span
    const rendered = try std.fmt.allocPrint(aa, "<user>look at these{s}{s}and{s}{s}", .{ sp.messages.array[0].get("content").?.array[1].strField("tf_mark").?, expand_mod.IMAGE_SPAN, sp.messages.array[0].get("content").?.array[3].strField("tf_mark").?, expand_mod.IMAGE_SPAN });
    var read = try marks.read(a, rendered, true, &diag);
    defer read.deinit(a);
    try std.testing.expectEqualStrings("<user>look at these" ++ expand_mod.IMAGE_SPAN ++ "and" ++ expand_mod.IMAGE_SPAN, read.text);
    var media = try f.loadSources(a, sp.sources, read.order, &diag);
    defer media.deinit(a);
    var ex = try f.expand(a, read.text, media.pictures, media.clips, &diag);
    defer ex.deinit(a);
    // the stub tokenizer: <|image|> -> 154854, bytes otherwise
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(a);
    var pos: usize = 0;
    while (pos < ex.text.len) {
        if (std.mem.startsWith(u8, ex.text[pos..], expand_mod.IMAGE)) {
            try ids.append(a, 154854);
            pos += expand_mod.IMAGE.len;
        } else {
            try ids.append(a, ex.text[pos]);
            pos += 1;
        }
    }
    var fin = try f.finish(a, ids.items, 154854, ex.items, null, &diag);
    defer fin.deinit(a);
    var rows: usize = 0;
    for (media.pictures) |p| rows += @intCast(p.rows());
    try std.testing.expectEqual(rows, fin.positions.len);
    try std.testing.expectEqual(@as(usize, 2), fin.item_keys.len);
}

test "concurrent requests share the cache" {
    const a = std.testing.allocator;
    var f = Frontend.init(a, std.testing.io, .{ .cache = .{ .mb = 16, .canvases = 2 } });
    defer f.deinit();
    var urls: [5][]u8 = undefined;
    for (&urls, 0..) |*u, i| u.* = try sampleDataUrl(a, 30 + i * 7, 40 + i * 3);
    defer for (urls) |u| a.free(u);
    const Worker = struct {
        fn run(fe: *Frontend, us: []const []u8, seed: usize, failed: *std.atomic.Value(bool)) void {
            const al = std.testing.allocator;
            for (0..40) |k| {
                const pick = (seed * 7 + k * 3) % us.len;
                const pick2 = (seed + k) % us.len;
                const srcs = [_]split.ImageSource{ .{ .url = us[pick], .detail = .auto }, .{ .url = us[pick2], .detail = .auto } };
                var diag: Diag = .{};
                const items = fe.loadImages(al, &srcs, &diag) catch {
                    failed.store(true, .release);
                    return;
                };
                for (items) |*it| {
                    _ = fe.imagePixels(al, @constCast(it), &diag) catch failed.store(true, .release);
                    @constCast(it).deinit(al);
                }
                al.free(items);
            }
        }
    };
    var failed = std.atomic.Value(bool).init(false);
    var threads: [4]std.Thread = undefined;
    for (&threads, 0..) |*t, i| t.* = try std.Thread.spawn(.{}, Worker.run, .{ &f, &urls, i, &failed });
    for (threads) |t| t.join();
    try std.testing.expect(!failed.load(.acquire));
}
