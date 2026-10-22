//! The server-side request splitting of tensorfold.vision.images.split_images and tensorfold.server.prompts
//! (MediaMarks, has_images, tool_media_parts), as pure functions on the server's `json.Value`.
//!
//! `splitImages` turns image_url / video_url content parts of user messages (and tool results when `tool_media`)
//! into the chat template's parts ({"type":"image","detail":..} / {"type":"video"}) and lists the sources in the
//! order met. `MediaMarks` writes a per-request nonce mark into each such part (`part["tf_mark"]`), reads the
//! rendered prompt back (marks removed, media order recovered) and lets the frontend escape quoted markers.
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const expand_mod = @import("expand.zig");
const items = @import("items.zig");
const limits_mod = @import("limits.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const Diag = errors.Diag;
const Detail = items.Detail;

const media_keys = [_][]const u8{ "image", "images", "image_url", "input_image", "audio", "input_audio", "video", "video_url" };

pub const ImageSource = struct { url: []const u8, detail: Detail };
pub const VideoSource = struct { url: []const u8 };
pub const Source = union(enum) {
    image: ImageSource,
    video: VideoSource,

    pub fn url(s: Source) []const u8 {
        return switch (s) {
            .image => |i| i.url,
            .video => |v| v.url,
        };
    }
};

pub const Options = struct {
    limits: limits_mod.ImageLimits = .default,
    video_limits: limits_mod.VideoLimits = .default,
    allow_urls: bool = false,
    allow_videos: bool = false,
    max_videos: ?u32 = null, // null: the shared video limits' (2)
    tool_media: bool = false,
};

pub const Split = struct {
    messages: Value, // the messages with media parts replaced by template parts (new objects)
    sources: []Source,
};

/// Python len(str): code points (WTF-8 continuation bytes are not counted).
pub fn codepoints(s: []const u8) usize {
    var n: usize = 0;
    for (s) |c| n += @intFromBool((c & 0xc0) != 0x80);
    return n;
}

fn anyMedia(obj: Value, except: ?[]const u8) bool {
    for (media_keys) |k| {
        if (except) |e| if (std.mem.eql(u8, k, e)) continue;
        if (obj.get(k)) |v| if (v.truthy()) return true;
    }
    return false;
}

fn isRole(role: []const u8) bool {
    inline for (.{ "system", "developer", "user", "assistant", "tool" }) |r| if (std.mem.eql(u8, role, r)) return true;
    return false;
}

/// `_check_source` (the URL rules): data URLs within the byte limit, else public HTTPS when URLs are on.
pub fn checkImageUrl(url: []const u8, limits: limits_mod.ImageLimits, allow_urls: bool, diag: ?*Diag) errors.Error!void {
    if (url.len == 0) return errors.fail(diag, "image source must contain a non-empty URL", .{});
    if (std.mem.startsWith(u8, url, "data:")) {
        if (codepoints(url) > limits.max_encoded_bytes * 3 + 256) return errors.fail(diag, "image data URL exceeds the encoded byte limit", .{});
    } else if (!std.mem.startsWith(u8, url, "https://")) {
        return errors.fail(diag, "images require data URLs or public HTTPS URLs", .{});
    } else if (!allow_urls) {
        return errors.fail(diag, "image URLs are off on this server; send the image as a data URL, or start the server with --vision-urls", .{});
    } else if (codepoints(url) > limits.max_url_chars) {
        return errors.fail(diag, "image URL is too long", .{});
    }
}

fn imageSource(value: ?Value, opts: Options, diag: ?*Diag) errors.Error!ImageSource {
    const v = value orelse .null;
    const u = if (v == .object) v.strField("url") else null;
    if (v != .object or u == null or u.?.len == 0) return errors.fail(diag, "image_url must contain a non-empty url string", .{});
    var detail: Detail = .auto;
    if (v.get("detail")) |d| {
        const ds = if (d == .string) Detail.parse(d.string) else null;
        if (ds == null) return errors.fail(diag, "image detail must be auto, low or high", .{});
        detail = ds.?;
    }
    try checkImageUrl(u.?, opts.limits, opts.allow_urls, diag);
    return .{ .url = u.?, .detail = detail };
}

fn videoSource(value: ?Value, opts: Options, diag: ?*Diag) errors.Error!VideoSource {
    const v = value orelse .null;
    const u = if (v == .object) v.strField("url") else null;
    if (v != .object or u == null or u.?.len == 0) return errors.fail(diag, "video_url must contain a non-empty url string", .{});
    const url = u.?;
    if (std.mem.startsWith(u8, url, "data:")) {
        if (codepoints(url) > opts.video_limits.max_encoded_bytes * 3 / 2 + 256) return errors.fail(diag, "video data URL exceeds the encoded byte limit", .{});
    } else if (!std.mem.startsWith(u8, url, "https://")) {
        return errors.fail(diag, "videos require data URLs or public HTTPS URLs", .{});
    } else if (!opts.allow_urls) {
        return errors.fail(diag, "video URLs are off on this server; send the video as a data URL, or start the server with --vision-urls", .{});
    } else if (codepoints(url) > opts.video_limits.max_url_chars) {
        return errors.fail(diag, "video URL is too long", .{});
    }
    return .{ .url = url };
}

fn strValue(a: Allocator, s: []const u8) Allocator.Error!Value {
    return .{ .string = try a.dupe(u8, s) };
}

/// `split_images`. All new objects are allocated in `a` (use an arena); sources borrow the input's strings.
pub fn splitImages(a: Allocator, messages: Value, opts: Options, diag: ?*Diag) (Allocator.Error || errors.Error)!Split {
    if (messages != .array or messages.array.len == 0) return errors.fail(diag, "messages must be a non-empty list", .{});
    const out = try a.alloc(Value, messages.array.len);
    var sources: std.ArrayList(Source) = .empty;
    var n_images: usize = 0;
    var n_videos: usize = 0;
    for (messages.array, out) |message, *slot| {
        if (message != .object) return errors.fail(diag, "each message must be an object", .{});
        const role_v = message.get("role");
        if (role_v == null or role_v.? != .string or !isRole(role_v.?.string)) return errors.fail(diag, "invalid message role", .{});
        const role = role_v.?.string;
        if (anyMedia(message, null)) return errors.fail(diag, "images must be image_url parts in user message content", .{});
        const content = message.get("content") orelse .null;
        if (content == .null or content == .string) {
            slot.* = .{ .object = try json.copyObject(a, message.object) };
            continue;
        }
        if (content != .array) return errors.fail(diag, "message content must be text or an array of content parts", .{});
        const parts = try a.alloc(Value, content.array.len);
        for (content.array, parts) |part, *pslot| {
            if (part != .object) return errors.fail(diag, "each content part must be an object", .{});
            const kind: []const u8 = if (part.get("type")) |t| (if (t == .string) t.string else "") else "";
            const kind_is_str = if (part.get("type")) |t| t == .string else false;
            const user_ok = std.mem.eql(u8, role, "user") or (opts.tool_media and std.mem.eql(u8, role, "tool"));
            if (kind_is_str and std.mem.eql(u8, kind, "text")) {
                const t = part.get("text");
                if (t == null or t.? != .string or anyMedia(part, null)) return errors.fail(diag, "text parts must contain a text string without media", .{});
                pslot.* = .{ .object = try json.copyObject(a, part.object) };
            } else if (kind_is_str and std.mem.eql(u8, kind, "video_url") and opts.allow_videos) {
                if (!user_ok) return errors.fail(diag, "video_url parts are supported only in user messages", .{});
                if (anyMedia(part, "video_url")) return errors.fail(diag, "video_url parts cannot contain other media", .{});
                const most: u32 = if (opts.max_videos) |m| (if (m != 0) m else opts.video_limits.max_videos) else opts.video_limits.max_videos;
                if (n_videos >= most) return errors.fail(diag, "a request supports at most {d} videos", .{most});
                const src = try videoSource(part.get("video_url"), opts, diag);
                try sources.append(a, .{ .video = src });
                n_videos += 1;
                const o = try json.newObject(a);
                try o.put(a, "type", try strValue(a, "video"));
                pslot.* = .{ .object = o };
            } else if (kind_is_str and std.mem.eql(u8, kind, "image_url")) {
                if (!user_ok) return errors.fail(diag, "image_url parts are supported only in user messages", .{});
                if (anyMedia(part, "image_url")) return errors.fail(diag, "image_url parts cannot contain other media", .{});
                if (n_images >= opts.limits.max_images) return errors.fail(diag, "a request supports at most {d} images", .{opts.limits.max_images});
                const src = try imageSource(part.get("image_url"), opts, diag);
                try sources.append(a, .{ .image = src });
                n_images += 1;
                const o = try json.newObject(a);
                try o.put(a, "type", try strValue(a, "image"));
                try o.put(a, "detail", try strValue(a, @tagName(src.detail)));
                pslot.* = .{ .object = o };
            } else {
                return if (opts.allow_videos)
                    errors.fail(diag, "content parts must be text, image_url or video_url; audio is unsupported", .{})
                else
                    errors.fail(diag, "content parts must be text or image_url; audio and video are unsupported", .{});
            }
        }
        const m = try json.copyObject(a, message.object);
        try m.put(a, "content", .{ .array = parts });
        slot.* = .{ .object = m };
    }
    return .{ .messages = .{ .array = out }, .sources = try sources.toOwnedSlice(a) };
}

pub const Partitioned = struct {
    images: []ImageSource,
    videos: []VideoSource,
};

/// `prepare_images`' `sources = [sources[k] for k in order]` and the split into pictures and clips (each list in the
/// order the template rendered them). `order` null: the sources as they came.
pub fn partition(a: Allocator, sources: []const Source, order: ?[]const usize) Allocator.Error!Partitioned {
    var images_: std.ArrayList(ImageSource) = .empty;
    errdefer images_.deinit(a);
    var videos_: std.ArrayList(VideoSource) = .empty;
    errdefer videos_.deinit(a);
    const n = if (order) |o| o.len else sources.len;
    for (0..n) |i| {
        const s = sources[if (order) |o| o[i] else i];
        switch (s) {
            .image => |im| try images_.append(a, im),
            .video => |v| try videos_.append(a, v),
        }
    }
    return .{ .images = try images_.toOwnedSlice(a), .videos = try videos_.toOwnedSlice(a) };
}

fn isMediaPart(p: Value) bool {
    if (p != .object) return false;
    return p.typeIs("image_url") or p.typeIs("video_url");
}

/// prompts.has_images: any message content part is an image_url or video_url.
pub fn hasImages(messages: Value) bool {
    if (messages != .array) return false;
    for (messages.array) |m| {
        if (m != .object) continue;
        const c = m.get("content") orelse continue;
        if (c != .array) continue;
        for (c.array) |p| if (isMediaPart(p)) return true;
    }
    return false;
}

/// prompts.tool_media_parts: for a server without a vision frontend: messages whose pictures and clips are all in
/// tool results, those parts as the template's bare media parts; null when there are none or a non-tool message has some.
pub fn toolMediaParts(a: Allocator, messages: Value) Allocator.Error!?Value {
    if (!hasImages(messages)) return null;
    const out = try a.alloc(Value, messages.array.len);
    for (messages.array, out) |m, *slot| {
        const content = if (m == .object) m.get("content") else null;
        var has = false;
        if (content) |c| if (c == .array) for (c.array) |p| {
            if (isMediaPart(p)) has = true;
        };
        if (!has) {
            slot.* = m;
            continue;
        }
        const role = m.strField("role");
        if (role == null or !std.mem.eql(u8, role.?, "tool")) return null;
        const parts = try a.alloc(Value, content.?.array.len);
        for (content.?.array, parts) |p, *ps| {
            if (isMediaPart(p)) {
                const t = p.strField("type").?;
                const o = try json.newObject(a);
                try o.put(a, "type", try strValue(a, t[0 .. t.len - "_url".len]));
                ps.* = .{ .object = o };
            } else ps.* = p;
        }
        const mo = try json.copyObject(a, m.object);
        try mo.put(a, "content", .{ .array = parts });
        slot.* = .{ .object = mo };
    }
    return .{ .array = out };
}

/// `MediaMarks`: each picture's and clip's place in the rendered prompt, matched by a per-request nonce mark.
pub const MediaMarks = struct {
    nonce: [16]u8, // 16 lower-case hex digits
    count: usize,

    /// Writes `tf_mark` into every image/video part of `template` (mutating it) in message order.
    pub fn init(a: Allocator, template: Value, nonce_bytes: [8]u8) Allocator.Error!MediaMarks {
        var m: MediaMarks = .{ .nonce = std.fmt.bytesToHex(nonce_bytes, .lower), .count = 0 };
        if (template != .array) return m;
        for (template.array) |message| {
            const c = message.get("content") orelse continue;
            if (c != .array) continue;
            for (c.array) |part| {
                if (part == .object and (part.typeIs("image") or part.typeIs("video"))) {
                    const mark = try std.fmt.allocPrint(a, "\x00tf-media-{s}:{d}\x00", .{ m.nonce, m.count });
                    try part.object.put(a, "tf_mark", .{ .string = mark });
                    m.count += 1;
                }
            }
        }
        return m;
    }

    /// Whether the request carries pictures or clips at all.
    pub fn wanted(template: Value) bool {
        if (template != .array) return false;
        for (template.array) |m| {
            const c = m.get("content") orelse continue;
            if (c != .array) continue;
            for (c.array) |p| if (p == .object and (p.typeIs("image") or p.typeIs("video"))) return true;
        }
        return false;
    }

    pub const Read = struct {
        text: []u8,
        order: []usize, // the media's indices in the order the template rendered them

        pub fn deinit(r: *Read, a: Allocator) void {
            a.free(r.text);
            a.free(r.order);
        }
    };

    /// `read(text, escape)`: the text without its marks and the media order. With `escape`, markers the conversation
    /// only quotes are neutralized first (the frontend's `escape_quoted`, given where this request's marks end).
    pub fn read(m: MediaMarks, a: Allocator, text: []const u8, escape: bool, diag: ?*Diag) (Allocator.Error || errors.Error)!Read {
        var prefix_buf: [40]u8 = undefined;
        const prefix = std.fmt.bufPrint(&prefix_buf, "\x00tf-media-{s}:", .{m.nonce}) catch unreachable;
        var order: std.ArrayList(usize) = .empty;
        errdefer order.deinit(a);
        var ends: std.ArrayList(usize) = .empty;
        defer ends.deinit(a);
        var starts: std.ArrayList(usize) = .empty;
        defer starts.deinit(a);
        var pos: usize = 0;
        while (std.mem.indexOfPos(u8, text, pos, prefix)) |at| {
            var j = at + prefix.len;
            const ds = j;
            while (j < text.len and text[j] >= '0' and text[j] <= '9') j += 1;
            if (j == ds or j >= text.len or text[j] != 0) {
                pos = at + 1;
                continue;
            }
            const k = std.fmt.parseInt(usize, text[ds..j], 10) catch std.math.maxInt(usize);
            try order.append(a, k);
            try starts.append(a, at);
            try ends.append(a, j + 1);
            pos = j + 1;
        }
        for (order.items) |k| if (k >= m.count) return errors.fail(diag, "the rendered prompt's media marks do not match the request's media", .{});
        const source: []u8 = if (escape) try expand_mod.escapeQuoted(a, text, ends.items) else try a.dupe(u8, text);
        defer a.free(source);
        // sub: drop the marks. After escaping, the marks still stand where they were, so find them again.
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(a);
        var p2: usize = 0;
        while (std.mem.indexOfPos(u8, source, p2, prefix)) |at| {
            var j = at + prefix.len;
            const ds = j;
            while (j < source.len and source[j] >= '0' and source[j] <= '9') j += 1;
            if (j == ds or j >= source.len or source[j] != 0) {
                try out.appendSlice(a, source[p2 .. at + 1]);
                p2 = at + 1;
                continue;
            }
            try out.appendSlice(a, source[p2..at]);
            p2 = j + 1;
        }
        try out.appendSlice(a, source[p2..]);
        return .{ .text = try out.toOwnedSlice(a), .order = try order.toOwnedSlice(a) };
    }
};

const fixtures = @import("fixtures.zig");

fn buildOptions(o: json.Object) Options {
    var opts: Options = .{};
    if (o.get("max_images")) |v| opts.limits.max_images = @intCast(v.int64().?);
    if (o.get("allow_urls")) |v| opts.allow_urls = v.bool;
    if (o.get("allow_videos")) |v| opts.allow_videos = v.bool;
    if (o.get("tool_media")) |v| opts.tool_media = v.bool;
    if (o.get("max_videos")) |v| opts.max_videos = if (v == .null) null else @intCast(v.int64().?);
    return opts;
}

test "split_images, the Python messages and results" {
    const a = std.testing.allocator;
    const text = try fixtures.read(a, "split.json", 1 << 22);
    defer a.free(text);
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    const parsed = switch (try json.parseText(aa, text)) {
        .ok => |v| v,
        .err => |e| {
            std.debug.print("fixture: {s}\n", .{e});
            return error.BadFixture;
        },
    };
    var n: usize = 0;
    for (parsed.array) |c| {
        const opts = buildOptions(c.get("opts").?.object.*);
        var diag: Diag = .{};
        const res = splitImages(aa, c.get("in").?, opts, &diag);
        if (c.get("error")) |want| {
            try std.testing.expectError(error.Input, res);
            try std.testing.expectEqualStrings(want.string, diag.message());
        } else {
            const got = try res;
            try std.testing.expect(json.equal(c.get("out").?, got.messages));
            const want_src = c.get("sources").?.array;
            try std.testing.expectEqual(want_src.len, got.sources.len);
            for (want_src, got.sources) |w, g| {
                try std.testing.expectEqualStrings(w.get("url").?.string, g.url());
                switch (g) {
                    .image => |im| try std.testing.expectEqualStrings(w.get("detail").?.string, @tagName(im.detail)),
                    .video => try std.testing.expect(w.get("detail") == null),
                }
            }
        }
        n += 1;
    }
    try std.testing.expect(n >= 40);
}

test "MediaMarks, the Python template and rendered text" {
    const a = std.testing.allocator;
    const text = try fixtures.read(a, "marks.json", 1 << 22);
    defer a.free(text);
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    const parsed = switch (try json.parseText(aa, text)) {
        .ok => |v| v,
        .err => return error.BadFixture,
    };
    for (parsed.array) |c| {
        var diag: Diag = .{};
        const template = try json.deepCopy(aa, c.get("template").?);
        try std.testing.expectEqual(c.get("wanted").?.bool, MediaMarks.wanted(template));
        var nonce: [8]u8 = undefined;
        _ = try std.fmt.hexToBytes(&nonce, c.get("nonce").?.string);
        const marks = try MediaMarks.init(aa, template, nonce);
        try std.testing.expect(json.equal(c.get("marked").?, template));
        try std.testing.expectEqual(@as(usize, @intCast(c.get("count").?.int64().?)), marks.count);
        for (c.get("reads").?.array) |r| {
            const escape = r.get("escape").?.bool;
            const res = marks.read(aa, r.get("text").?.string, escape, &diag);
            if (r.get("error")) |want| {
                try std.testing.expectError(error.Input, res);
                try std.testing.expectEqualStrings(want.string, diag.message());
            } else {
                const got = try res;
                try std.testing.expectEqualStrings(r.get("out").?.string, got.text);
                const want_order = r.get("order").?.array;
                try std.testing.expectEqual(want_order.len, got.order.len);
                for (want_order, got.order) |w, g| try std.testing.expectEqual(@as(usize, @intCast(w.int64().?)), g);
            }
        }
    }
}
