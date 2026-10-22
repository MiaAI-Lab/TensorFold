//! Prompt expansion exactly as `GlmVision.prepare` does it: the rendered text's picture and clip spans become the
//! checkpoint's placeholder runs; markers a conversation only mentions stay text (zero-width space inside).
//!
//! The server tokenizes the expanded text with add_special_tokens=false and hands the ids to `finish`.
const std = @import("std");
const errors = @import("errors.zig");
const items_mod = @import("items.zig");
const Diag = errors.Diag;
const Item = items_mod.Item;
const Allocator = std.mem.Allocator;

pub const IMAGE = "<|image|>";
pub const VIDEO = "<|video|>";
pub const IMAGE_SPAN = "<|begin_of_image|><|image|><|end_of_image|>"; // what the chat template writes for a picture
pub const VIDEO_SPAN = "<|begin_of_video|><|video|><|end_of_video|>"; // and for a clip
const ZWSP = "\xe2\x80\x8b"; // U+200B

const mentions = [_][]const u8{ "<|begin_of_image|>", "<|end_of_image|>", "<|begin_of_video|>", "<|end_of_video|>", IMAGE, VIDEO };

/// `_as_text`: a zero-width space after the `<|` of every marker mention. Appends to `out`.
pub fn appendAsText(allocator: Allocator, out: *std.ArrayList(u8), text: []const u8) Allocator.Error!void {
    if (std.mem.indexOf(u8, text, "<|") == null) return out.appendSlice(allocator, text);
    var cur: []u8 = try allocator.dupe(u8, text);
    defer allocator.free(cur);
    for (mentions) |mark| {
        if (std.mem.indexOf(u8, cur, mark) == null) continue;
        var rep_buf: [32]u8 = undefined;
        const rep = std.fmt.bufPrint(&rep_buf, "<|{s}{s}", .{ ZWSP, mark[2..] }) catch unreachable;
        const n = std.mem.replacementSize(u8, cur, mark, rep);
        const next = try allocator.alloc(u8, n);
        _ = std.mem.replace(u8, cur, mark, rep, next);
        allocator.free(cur);
        cur = next;
    }
    try out.appendSlice(allocator, cur);
}

pub fn asText(allocator: Allocator, text: []const u8) Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendAsText(allocator, &out, text);
    return out.toOwnedSlice(allocator);
}

pub const Span = struct { start: usize, end: usize, video: bool };

/// `_marker_spans`: every picture and clip span in `text`, left to right.
pub fn markerSpans(allocator: Allocator, text: []const u8) Allocator.Error![]Span {
    var out: std.ArrayList(Span) = .empty;
    errdefer out.deinit(allocator);
    var pos: usize = 0;
    while (true) {
        const i = std.mem.indexOfPos(u8, text, pos, IMAGE_SPAN);
        const v = std.mem.indexOfPos(u8, text, pos, VIDEO_SPAN);
        if (i == null and v == null) break;
        if (v == null or (i != null and i.? < v.?)) {
            try out.append(allocator, .{ .start = i.?, .end = i.? + IMAGE_SPAN.len, .video = false });
            pos = i.? + IMAGE_SPAN.len;
        } else {
            try out.append(allocator, .{ .start = v.?, .end = v.? + VIDEO_SPAN.len, .video = true });
            pos = v.? + VIDEO_SPAN.len;
        }
    }
    return out.toOwnedSlice(allocator);
}

/// `GlmVision.escape_quoted(text, mark_ends)`: markers the conversation only quotes become text. `mark_ends` are
/// byte offsets in `text` where this request's own marks end (a span starting there is the template's own).
pub fn escapeQuoted(allocator: Allocator, text: []const u8, mark_ends: []const usize) Allocator.Error![]u8 {
    const spans = try markerSpans(allocator, text);
    defer allocator.free(spans);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var pos: usize = 0;
    for (spans) |s| {
        try appendAsText(allocator, &out, text[pos..s.start]);
        const own = std.mem.indexOfScalar(usize, mark_ends, s.start) != null;
        if (own) try out.appendSlice(allocator, text[s.start..s.end]) else try appendAsText(allocator, &out, text[s.start..s.end]);
        pos = s.end;
    }
    try appendAsText(allocator, &out, text[pos..]);
    return out.toOwnedSlice(allocator);
}

fn count(text: []const u8, needle: []const u8) usize {
    var n: usize = 0;
    var pos: usize = 0;
    while (std.mem.indexOfPos(u8, text, pos, needle)) |i| {
        n += 1;
        pos = i + needle.len;
    }
    return n;
}

pub const Expanded = struct {
    text: []u8, // owned by the caller's allocator
    items: []Item, // the media in prompt order (copies of the given records; they share the data)

    pub fn deinit(e: *Expanded, allocator: Allocator) void {
        allocator.free(e.text);
        allocator.free(e.items);
    }
};

/// `prepare`'s walk: `text` is the rendered prompt; `images` and `videos` are consumed in the order their spans
/// appear. Errors carry the Python text.
pub fn expand(allocator: Allocator, text: []const u8, images: []const Item, videos: []const Item, diag: ?*Diag) (Allocator.Error || errors.Error)!Expanded {
    if (count(text, IMAGE_SPAN) != images.len or count(text, VIDEO_SPAN) != videos.len)
        return errors.fail(diag, "the prompt's image and video markers do not match its images and videos (is <|image|> or <|video|> in the text?)", .{});
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var order: std.ArrayList(Item) = .empty;
    errdefer order.deinit(allocator);
    var ii: usize = 0;
    var vi: usize = 0;
    var rest = text;
    while (true) {
        const i = std.mem.indexOf(u8, rest, IMAGE_SPAN);
        const v = std.mem.indexOf(u8, rest, VIDEO_SPAN);
        if (i == null and v == null) {
            try appendAsText(allocator, &out, rest);
            break;
        }
        if (v == null or (i != null and i.? < v.?)) {
            const at = i.?;
            const pic = images[ii];
            ii += 1;
            const n: usize = @intCast(pic.rows());
            try appendAsText(allocator, &out, rest[0..at]);
            try out.appendSlice(allocator, "<|begin_of_image|>");
            for (0..n) |_| try out.appendSlice(allocator, IMAGE);
            try out.appendSlice(allocator, "<|end_of_image|>");
            try order.append(allocator, pic);
            rest = rest[at + IMAGE_SPAN.len ..];
        } else {
            const at = v.?;
            const clip = videos[vi];
            vi += 1;
            const c = clip.clip;
            const n = @import("geometry.zig").rowsPerFrame(c.h, c.w);
            try appendAsText(allocator, &out, rest[0..at]);
            try out.appendSlice(allocator, "<|begin_of_video|>");
            for (c.seconds) |s| {
                try out.appendSlice(allocator, "<|begin_of_image|>");
                for (0..n) |_| try out.appendSlice(allocator, IMAGE);
                try out.appendSlice(allocator, "<|end_of_image|>");
                var nb: [400]u8 = undefined;
                try out.appendSlice(allocator, pyFixed1(&nb, s));
                try out.appendSlice(allocator, " seconds");
            }
            try out.appendSlice(allocator, "<|end_of_video|>");
            try order.append(allocator, clip);
            rest = rest[at + VIDEO_SPAN.len ..];
        }
    }
    return .{ .text = try out.toOwnedSlice(allocator), .items = try order.toOwnedSlice(allocator) };
}

/// Python's `f"{s:.1f}"`: the exact binary value rounded half to even (Zig's `{d:.1}` rounds ties away).
pub fn pyFixed1(buf: []u8, s: f64) []const u8 {
    if (!std.math.isFinite(s) or @abs(s) >= 1e15) return std.fmt.bufPrint(buf, "{d:.1}", .{s}) catch "0.0";
    const bits: u64 = @bitCast(s);
    const neg = (bits >> 63) != 0;
    const exp_field: i32 = @intCast((bits >> 52) & 0x7ff);
    var mant: u128 = bits & ((1 << 52) - 1);
    var e: i32 = undefined;
    if (exp_field == 0) {
        e = -1074;
    } else {
        mant |= 1 << 52;
        e = exp_field - 1075;
    }
    // d = round_half_even(mant * 10 * 2^e)
    var d: u128 = undefined;
    const m10 = mant * 10;
    if (e >= 0) {
        d = m10 << @intCast(e);
    } else if (-e > 120) {
        d = 0;
    } else {
        const sh: u7 = @intCast(-e);
        const q = m10 >> sh;
        const rem = m10 & ((@as(u128, 1) << sh) - 1);
        const half = @as(u128, 1) << (sh - 1);
        d = q;
        if (rem > half or (rem == half and (q & 1) == 1)) d += 1;
    }
    return std.fmt.bufPrint(buf, "{s}{d}.{d}", .{ if (neg) "-" else "", d / 10, d % 10 }) catch "0.0";
}

pub const Prepared = struct {
    positions: []u32, // prompt positions of the feature rows, in order
    content_hash: [64]u8,
    item_rows: []u32,
    item_keys: []u64,

    pub fn deinit(p: *Prepared, allocator: Allocator) void {
        allocator.free(p.positions);
        allocator.free(p.item_rows);
        allocator.free(p.item_keys);
    }
};

/// The rest of `prepare` once the server has tokenized the expanded text: the context check, the positions of the
/// picture token, the feature-row count check, the content hash and item keys.
pub fn finish(allocator: Allocator, ids: []const u32, image_token: u32, items: []const Item, max_prompt_tokens: ?usize, diag: ?*Diag) (Allocator.Error || errors.Error)!Prepared {
    if (max_prompt_tokens) |m| if (m != 0 and ids.len >= m) {
        var b: [32]u8 = undefined;
        return errors.fail(diag, "{s} {d} tokens: the prompt with its images is {s} tokens, which exceeds the context window; send fewer or smaller images, or a shorter video", .{ errors.context_limit, m, errors.commas(&b, ids.len) });
    };
    var positions: std.ArrayList(u32) = .empty;
    errdefer positions.deinit(allocator);
    for (ids, 0..) |t, p| if (t == image_token) try positions.append(allocator, @intCast(p));
    var rows: u64 = 0;
    for (items) |it| rows += it.rows();
    if (positions.items.len != rows)
        return errors.fail(diag, "the prompt has {d} image positions for {d} feature rows", .{ positions.items.len, rows });
    const h = try items_mod.hashItems(allocator, items);
    return .{ .positions = try positions.toOwnedSlice(allocator), .content_hash = h.content_hash, .item_rows = h.rows, .item_keys = h.keys };
}

/// `GlmPrepared.keyed_ids`: the prompt's ids with each item's feature rows replaced by -1 - key.
pub fn keyedIds(allocator: Allocator, ids: []const u32, p: Prepared) Allocator.Error![]i64 {
    const out = try allocator.alloc(i64, ids.len);
    for (ids, out) |t, *o| o.* = t;
    var at: usize = 0;
    for (p.item_rows, p.item_keys) |rows, key| {
        for (p.positions[at .. at + rows]) |pos| out[pos] = -1 - @as(i64, @intCast(key));
        at += rows;
    }
    return out;
}

const fixtures = @import("fixtures.zig");

fn jsonU32(v: std.json.Value) u32 {
    return @intCast(v.integer);
}

test "escape_quoted against the Python frontend" {
    const a = std.testing.allocator;
    const text = try fixtures.read(a, "escape.json", 1 << 20);
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer parsed.deinit();
    for (parsed.value.array.items) |c| {
        const o = c.object;
        var ends: [8]usize = undefined;
        const arr = o.get("ends").?.array.items;
        for (arr, 0..) |e, i| ends[i] = @intCast(e.integer);
        const got = try escapeQuoted(a, o.get("text").?.string, ends[0..arr.len]);
        defer a.free(got);
        try std.testing.expectEqualStrings(o.get("out").?.string, got);
    }
}

test "prepare against the Python frontend (expansion, errors, hashes, keys, rows)" {
    const a = std.testing.allocator;
    const text = try fixtures.read(a, "expand.json", 1 << 22);
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer parsed.deinit();
    var checked: usize = 0;
    for (parsed.value.array.items) |c| {
        const o = c.object;
        const cache = o.get("cache").?.bool;
        var arena: std.heap.ArenaAllocator = .init(a);
        defer arena.deinit();
        const aa = arena.allocator();
        var images: std.ArrayList(Item) = .empty;
        for (o.get("images").?.array.items) |im| {
            const io = im.object;
            const h = jsonU32(io.get("h").?);
            const w = jsonU32(io.get("w").?);
            const data = try aa.alloc(u8, 3 * h * w);
            fixtures.lcgBytes(data, @intCast(io.get("seed").?.integer));
            if (cache) {
                const sh = try items_mod.Shared.create(a, try a.dupe(u8, data));
                defer sh.release();
                try images.append(aa, .{ .picture = items_mod.rememberPicture(sh, h, w, @as(u64, jsonU32(io.get("pw").?)) * jsonU32(io.get("ph").?), .auto, 2048, 16) });
            } else try images.append(aa, .{ .canvas = .{ .data = data, .h = h, .w = w } });
        }
        defer if (cache) for (images.items) |*it| it.deinit(a);
        var videos: std.ArrayList(Item) = .empty;
        for (o.get("videos").?.array.items) |vd| {
            const vo = vd.object;
            const t = jsonU32(vo.get("t").?);
            const h = jsonU32(vo.get("h").?);
            const w = jsonU32(vo.get("w").?);
            const data = try aa.alloc(u8, t * 3 * h * w);
            fixtures.lcgBytes(data, @intCast(vo.get("seed").?.integer));
            const secs = vo.get("seconds").?.array.items;
            const seconds = try aa.alloc(f64, secs.len);
            for (secs, seconds) |s, *d| d.* = switch (s) {
                .float => |f| f,
                .integer => |i| @floatFromInt(i),
                else => unreachable,
            };
            try videos.append(aa, .{ .clip = .{ .frames = data, .t = t, .h = h, .w = w, .seconds = seconds } });
        }
        var diag: Diag = .{};
        const max_prompt: ?usize = if (o.get("max_prompt").? == .null) null else @intCast(o.get("max_prompt").?.integer);
        var failed: ?[]const u8 = null;
        var ex = expand(a, o.get("text").?.string, images.items, videos.items, &diag) catch |e| switch (e) {
            error.Input => blk: {
                failed = diag.message();
                break :blk null;
            },
            else => return e,
        };
        if (ex) |*e| {
            defer e.deinit(a);
            // the stub tokenizer of the generator: <|image|> -> 154854, any other character -> its UTF-8 bytes
            var ids: std.ArrayList(u32) = .empty;
            defer ids.deinit(a);
            var pos: usize = 0;
            while (pos < e.text.len) {
                if (std.mem.startsWith(u8, e.text[pos..], IMAGE)) {
                    try ids.append(a, 154854);
                    pos += IMAGE.len;
                } else {
                    try ids.append(a, e.text[pos]);
                    pos += 1;
                }
            }
            if (o.get("error") == null) try std.testing.expectEqualStrings(o.get("expanded").?.string, e.text);
            if (finish(a, ids.items, 154854, e.items, max_prompt, &diag)) |fin_const| {
                var fin = fin_const;
                defer fin.deinit(a);
                try std.testing.expect(o.get("error") == null);
                try std.testing.expectEqual(@as(usize, @intCast(o.get("ids_len").?.integer)), ids.items.len);
                try std.testing.expectEqualStrings(o.get("content_hash").?.string, &fin.content_hash);
                const pos_exp = o.get("positions").?.array.items;
                try std.testing.expectEqual(pos_exp.len, fin.positions.len);
                for (pos_exp, fin.positions) |x, y| try std.testing.expectEqual(@as(u32, @intCast(x.integer)), y);
                for (o.get("rows").?.array.items, fin.item_rows) |x, y| try std.testing.expectEqual(@as(u32, @intCast(x.integer)), y);
                for (o.get("keys").?.array.items, fin.item_keys) |x, y| try std.testing.expectEqual(@as(u64, @intCast(x.integer)), y);
            } else |err| {
                try std.testing.expect(err == error.Input);
                failed = diag.message();
            }
        }
        if (o.get("error")) |want| {
            try std.testing.expectEqualStrings(want.string, failed.?);
        } else try std.testing.expect(failed == null);
        checked += 1;
    }
    try std.testing.expectEqual(@as(usize, 24), checked);
}

test "seconds formatting is Python's" {
    var b: [64]u8 = undefined;
    const cases = [_]struct { f64, []const u8 }{ .{ 0.0, "0.0" }, .{ 0.05, "0.1" }, .{ 0.25, "0.2" }, .{ 0.35, "0.3" }, .{ 2.675, "2.7" }, .{ 7.949999999999999, "7.9" }, .{ 33.333333333333336, "33.3" }, .{ 100000.05, "100000.1" }, .{ 1e-9, "0.0" }, .{ 0.15, "0.1" }, .{ 0.45, "0.5" }, .{ 1.25, "1.2" }, .{ 3600.0, "3600.0" } };
    for (cases) |c| try std.testing.expectEqualStrings(c[1], pyFixed1(&b, c[0]));
}

test "seconds formatting over 3000 frame times" {
    const a = std.testing.allocator;
    const text = try fixtures.read(a, "format1.txt", 1 << 20);
    defer a.free(text);
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    var b: [400]u8 = undefined;
    var n: usize = 0;
    while (lines.next()) |line| {
        var t = std.mem.tokenizeScalar(u8, line, ' ');
        const v = try std.fmt.parseFloat(f64, t.next().?);
        try std.testing.expectEqualStrings(t.next().?, pyFixed1(&b, v));
        n += 1;
    }
    try std.testing.expect(n > 3000);
}
