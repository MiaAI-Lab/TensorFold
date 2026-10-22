//! Bounded HTTPS retrieval of picture and clip URLs (images_http.py): only public https hosts on port 443, every
//! redirect re-validated and its host's addresses checked, TLS verified for the host on the address that was
//! checked, a byte cap, a content-type check and a deadline.
//!
//! Differences from the Python module (documented, deliberate): host names must be ASCII (no IDNA/punycode
//! conversion: write the xn-- form), DNS lookups run on the calling thread (no resolver slot pool), and the read
//! deadline is enforced with socket receive timeouts rather than a watchdog thread.
const std = @import("std");
const errors = @import("errors.zig");
const frontend = @import("frontend.zig");
const Allocator = std.mem.Allocator;
const Diag = errors.Diag;
const net = std.Io.net;

pub const image_types = [_][]const u8{ "image/jpeg", "image/png", "image/webp" };
pub const video_types = [_][]const u8{ "video/mp4", "video/webm", "video/quicktime", "video/x-matroska" };
pub const user_agent = "TensorFold/zig-vision";

fn isControl(c: u8) bool {
    return c <= 32 or c == 127;
}

// ---- URL --------------------------------------------------------------------------------------------------------
pub const Parsed = struct {
    host: []u8, // ASCII, lower case, no brackets
    target: []u8, // request target: path (and ?query) percent-quoted as Python's quote()

    pub fn deinit(p: Parsed, a: Allocator) void {
        a.free(p.host);
        a.free(p.target);
    }
};

const unreserved = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-~";

fn quoteInto(a: Allocator, out: *std.ArrayList(u8), s: []const u8, safe: []const u8) Allocator.Error!void {
    for (s) |c| {
        if (std.mem.indexOfScalar(u8, unreserved, c) != null or std.mem.indexOfScalar(u8, safe, c) != null) {
            try out.append(a, c);
        } else {
            var b: [3]u8 = undefined;
            _ = std.fmt.bufPrint(&b, "%{X:0>2}", .{c}) catch unreachable;
            try out.appendSlice(a, &b);
        }
    }
}

/// images_http._url
pub fn parseUrl(a: Allocator, value: []const u8, max_url_chars: usize, diag: ?*Diag) (Allocator.Error || errors.Error)!Parsed {
    const split_mod = @import("split.zig");
    var bad_chars = false;
    for (value) |c| if (isControl(c)) {
        bad_chars = true;
        break;
    };
    if (split_mod.codepoints(value) > max_url_chars or bad_chars)
        return errors.fail(diag, "image URL is too long or contains whitespace/control characters", .{});
    const bad = "image URL must be HTTPS on port 443, without credentials or a fragment";
    if (std.mem.indexOfScalar(u8, value, '\\') != null) return errors.fail(diag, bad, .{});
    for (value) |c| if (c >= 0x80) return errors.fail(diag, bad, .{}); // IDNA host names are not converted
    // scheme
    const colon = std.mem.indexOfScalar(u8, value, ':') orelse return errors.fail(diag, bad, .{});
    for (value[0..colon]) |c| if (!(std.ascii.isAlphanumeric(c) or c == '+' or c == '-' or c == '.')) return errors.fail(diag, bad, .{});
    if (colon == 0 or !std.ascii.isAlphabetic(value[0]) or !std.ascii.eqlIgnoreCase(value[0..colon], "https")) return errors.fail(diag, bad, .{});
    var rest = value[colon + 1 ..];
    if (!std.mem.startsWith(u8, rest, "//")) return errors.fail(diag, bad, .{});
    rest = rest[2..];
    const netloc_end = std.mem.indexOfAny(u8, rest, "/?#") orelse rest.len;
    const netloc = rest[0..netloc_end];
    var tail = rest[netloc_end..];
    // fragment, then query
    var fragment: []const u8 = "";
    if (std.mem.indexOfScalar(u8, tail, '#')) |i| {
        fragment = tail[i + 1 ..];
        tail = tail[0..i];
    }
    var query: ?[]const u8 = null;
    if (std.mem.indexOfScalar(u8, tail, '?')) |i| {
        query = tail[i + 1 ..];
        tail = tail[0..i];
    }
    if (fragment.len != 0) return errors.fail(diag, bad, .{});
    if (std.mem.indexOfScalar(u8, netloc, '@') != null) return errors.fail(diag, bad, .{});
    // host and port
    var hostname: []const u8 = undefined;
    var port_text: ?[]const u8 = null;
    if (std.mem.indexOfScalar(u8, netloc, '[') != null or std.mem.indexOfScalar(u8, netloc, ']') != null) {
        const lb = std.mem.indexOfScalar(u8, netloc, '[') orelse return errors.fail(diag, bad, .{});
        const rb = std.mem.indexOfScalar(u8, netloc, ']') orelse return errors.fail(diag, bad, .{});
        if (lb != 0 or rb < lb) return errors.fail(diag, bad, .{});
        hostname = netloc[lb + 1 .. rb];
        const after = netloc[rb + 1 ..];
        if (after.len != 0) {
            if (after[0] != ':') return errors.fail(diag, bad, .{});
            port_text = after[1..];
        }
    } else if (std.mem.indexOfScalar(u8, netloc, ':')) |i| {
        hostname = netloc[0..i];
        port_text = netloc[i + 1 ..];
    } else hostname = netloc;
    if (hostname.len == 0) return errors.fail(diag, bad, .{});
    if (port_text) |pt| if (pt.len != 0) {
        for (pt) |c| if (c < '0' or c > '9') return errors.fail(diag, bad, .{});
        const p = std.fmt.parseInt(u32, pt, 10) catch return errors.fail(diag, bad, .{});
        if (p > 65535 or p != 443) return errors.fail(diag, bad, .{});
    };
    // idna.encode on an ASCII name: every label 1..63 characters (a trailing dot is the root)
    {
        var labels = std.mem.splitScalar(u8, hostname, '.');
        var idx: usize = 0;
        const total = std.mem.count(u8, hostname, ".") + 1;
        while (labels.next()) |l| : (idx += 1) {
            const last_root = idx == total - 1 and idx != 0 and l.len == 0;
            if (!last_root and (l.len == 0 or l.len > 63)) return errors.fail(diag, bad, .{});
        }
    }
    if (std.mem.indexOfScalar(u8, hostname, '%') != null) return errors.fail(diag, bad, .{});
    const host = try a.alloc(u8, hostname.len);
    errdefer a.free(host);
    _ = std.ascii.lowerString(host, hostname);
    {
        const t = std.mem.trimEnd(u8, host, ".");
        const blocked = [_][]const u8{ "localhost", "metadata.google.internal", "instance-data" };
        for (blocked) |b| if (std.mem.eql(u8, t, b)) return errors.fail(diag, "image URLs must use public internet hosts", .{});
    }
    var target: std.ArrayList(u8) = .empty;
    errdefer target.deinit(a);
    try quoteInto(a, &target, if (tail.len == 0) "/" else tail, "/%:@!$&'()*+,;=-._~");
    if (query) |q| if (q.len != 0) {
        try target.append(a, '?');
        try quoteInto(a, &target, q, "/%?:@!$&'()*+,;=-._~");
    };
    return .{ .host = host, .target = try target.toOwnedSlice(a) };
}

// ---- addresses -----------------------------------------------------------------------------------------------------
const Net4 = struct { base: u32, bits: u6 };

fn in4(addr: u32, base: u32, bits: u6) bool {
    const shift: u6 = 32 - bits;
    return if (bits == 0) true else (addr >> @intCast(shift)) == (base >> @intCast(shift));
}

fn ip4(a: u8, b: u8, c: u8, d: u8) u32 {
    return (@as(u32, a) << 24) | (@as(u32, b) << 16) | (@as(u32, c) << 8) | d;
}

/// images_http._public_ip for an IPv4 address (Python 3.12 `is_global`, not multicast, not reserved, and not the
/// two extra blocks).
pub fn public4(addr: u32) bool {
    const private = [_]Net4{
        .{ .base = ip4(0, 0, 0, 0), .bits = 8 },      .{ .base = ip4(10, 0, 0, 0), .bits = 8 },         .{ .base = ip4(127, 0, 0, 0), .bits = 8 },
        .{ .base = ip4(169, 254, 0, 0), .bits = 16 }, .{ .base = ip4(172, 16, 0, 0), .bits = 12 },      .{ .base = ip4(192, 0, 0, 0), .bits = 29 },
        .{ .base = ip4(192, 0, 0, 170), .bits = 31 }, .{ .base = ip4(192, 0, 2, 0), .bits = 24 },       .{ .base = ip4(192, 168, 0, 0), .bits = 16 },
        .{ .base = ip4(198, 18, 0, 0), .bits = 15 },  .{ .base = ip4(198, 51, 100, 0), .bits = 24 },    .{ .base = ip4(203, 0, 113, 0), .bits = 24 },
        .{ .base = ip4(240, 0, 0, 0), .bits = 4 },    .{ .base = ip4(255, 255, 255, 255), .bits = 32 },
    };
    var is_private = false;
    for (private) |n| if (in4(addr, n.base, n.bits)) {
        is_private = true;
        break;
    };
    if (in4(addr, ip4(192, 0, 0, 9), 32) or in4(addr, ip4(192, 0, 0, 10), 32)) is_private = false;
    const is_global = !in4(addr, ip4(100, 64, 0, 0), 10) and !is_private;
    if (!is_global) return false;
    if (in4(addr, ip4(224, 0, 0, 0), 4)) return false; // multicast
    if (in4(addr, ip4(240, 0, 0, 0), 4)) return false; // reserved
    if (in4(addr, ip4(192, 0, 0, 0), 24) or addr == ip4(168, 63, 129, 16)) return false;
    return true;
}

fn in6(addr: [16]u8, comptime prefix: []const u8, bits: u8) bool {
    var p: [16]u8 = @splat(0);
    @memcpy(p[0..prefix.len], prefix);
    const full = bits / 8;
    if (!std.mem.eql(u8, addr[0..full], p[0..full])) return false;
    const rem: u3 = @intCast(bits % 8);
    if (rem == 0) return true;
    const mask: u8 = @as(u8, 0xff) << @intCast(8 - @as(u4, rem));
    return (addr[full] & mask) == (p[full] & mask);
}

pub fn public6(addr: [16]u8) bool {
    // is_private: the IANA special-purpose blocks Python lists, except the globally reachable exceptions
    const private = comptime [_]struct { p: []const u8, bits: u8 }{
        .{ .p = &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, .bits = 128 }, // ::1
        .{ .p = &.{}, .bits = 128 }, // ::
        .{ .p = &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff }, .bits = 96 }, // ::ffff:0:0/96
        .{ .p = &.{ 0, 0x64, 0xff, 0x9b, 0, 1 }, .bits = 48 }, // 64:ff9b:1::/48
        .{ .p = &.{ 1, 0, 0, 0, 0, 0, 0, 0 }, .bits = 64 }, // 100::/64
        .{ .p = &.{ 0x20, 0x01 }, .bits = 23 }, // 2001::/23
        .{ .p = &.{ 0x20, 0x01, 0x0d, 0xb8 }, .bits = 32 }, // 2001:db8::/32
        .{ .p = &.{ 0x20, 0x02 }, .bits = 16 }, // 2002::/16
        .{ .p = &.{0xfc}, .bits = 7 }, // fc00::/7
        .{ .p = &.{ 0xfe, 0x80 }, .bits = 10 }, // fe80::/10
    };
    const exceptions = comptime [_]struct { p: []const u8, bits: u8 }{
        .{ .p = &.{ 0x20, 0x01, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 }, .bits = 128 }, // 2001:1::1
        .{ .p = &.{ 0x20, 0x01, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2 }, .bits = 128 }, // 2001:1::2
        .{ .p = &.{ 0x20, 0x01, 0, 3 }, .bits = 32 }, // 2001:3::/32
        .{ .p = &.{ 0x20, 0x01, 0, 4, 0x01, 0x12 }, .bits = 48 }, // 2001:4:112::/48
        .{ .p = &.{ 0x20, 0x01, 0, 0x20 }, .bits = 28 }, // 2001:20::/28
        .{ .p = &.{ 0x20, 0x01, 0, 0x30 }, .bits = 28 }, // 2001:30::/28
    };
    var is_private = false;
    inline for (private) |n| if (in6(addr, n.p, n.bits)) {
        is_private = true;
    };
    inline for (exceptions) |n| if (in6(addr, n.p, n.bits)) {
        is_private = false;
    };
    if (is_private) return false;
    if (addr[0] == 0xff) return false; // multicast ff00::/8
    // is_reserved
    const reserved = comptime [_]struct { p: []const u8, bits: u8 }{
        .{ .p = &.{0}, .bits = 8 },    .{ .p = &.{1}, .bits = 8 },    .{ .p = &.{2}, .bits = 7 },         .{ .p = &.{4}, .bits = 6 },
        .{ .p = &.{8}, .bits = 5 },    .{ .p = &.{0x10}, .bits = 4 }, .{ .p = &.{0x40}, .bits = 3 },      .{ .p = &.{0x60}, .bits = 3 },
        .{ .p = &.{0x80}, .bits = 3 }, .{ .p = &.{0xa0}, .bits = 3 }, .{ .p = &.{0xc0}, .bits = 3 },      .{ .p = &.{0xe0}, .bits = 4 },
        .{ .p = &.{0xf0}, .bits = 5 }, .{ .p = &.{0xf8}, .bits = 6 }, .{ .p = &.{ 0xfe, 0 }, .bits = 9 },
    };
    inline for (reserved) |n| if (in6(addr, n.p, n.bits)) return false;
    // ipv4_mapped (::ffff:0:0/96 is private above), sixtofour 2002::/16 (private above), teredo 2001::/32
    if (in6(addr, &.{ 0x20, 0x01, 0, 0 }, 32)) return false;
    return true;
}

pub fn publicAddress(addr: net.IpAddress) bool {
    return switch (addr) {
        .ip4 => |x| public4(std.mem.readInt(u32, &x.bytes, .big)),
        .ip6 => |x| public6(x.bytes),
    };
}

/// images_http._public_ip on text (for tests and literals).
pub fn publicText(text: []const u8) bool {
    const parsed = net.IpAddress.parse(text, 443) catch return false;
    return publicAddress(parsed);
}

// ---- urljoin -------------------------------------------------------------------------------------------------------
fn removeDotSegments(a: Allocator, path: []const u8) Allocator.Error![]u8 {
    var out: std.ArrayList([]const u8) = .empty;
    defer out.deinit(a);
    var it = std.mem.splitScalar(u8, path, '/');
    var trailing_slash = false;
    var first = true;
    while (it.next()) |seg| {
        defer first = false;
        if (std.mem.eql(u8, seg, "..")) {
            if (out.items.len > (if (path.len > 0 and path[0] == '/') @as(usize, 1) else 0)) _ = out.pop();
            trailing_slash = true;
        } else if (std.mem.eql(u8, seg, ".")) {
            trailing_slash = true;
        } else {
            try out.append(a, seg);
            trailing_slash = false;
        }
    }
    var res: std.ArrayList(u8) = .empty;
    errdefer res.deinit(a);
    for (out.items, 0..) |seg, i| {
        if (i != 0) try res.append(a, '/');
        try res.appendSlice(a, seg);
    }
    if (trailing_slash and (res.items.len == 0 or res.items[res.items.len - 1] != '/')) try res.append(a, '/');
    return res.toOwnedSlice(a);
}

/// urllib.parse.urljoin(base, ref) for the cases a redirect produces (RFC 3986 reference resolution).
pub fn urljoin(a: Allocator, base_in: []const u8, ref_in: []const u8) Allocator.Error![]u8 {
    // urlsplit drops leading control characters and spaces
    const base = std.mem.trimStart(u8, base_in, "\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f ");
    const ref = std.mem.trimStart(u8, ref_in, "\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f ");
    if (base.len == 0) return a.dupe(u8, ref);
    if (ref.len == 0) return a.dupe(u8, base);
    // scheme of ref
    var ref_scheme: ?[]const u8 = null;
    if (std.mem.indexOfScalar(u8, ref, ':')) |i| {
        var ok = i > 0 and std.ascii.isAlphabetic(ref[0]);
        for (ref[0..i]) |c| ok = ok and (std.ascii.isAlphanumeric(c) or c == '+' or c == '-' or c == '.');
        if (ok) ref_scheme = ref[0..i];
    }
    const bcolon = std.mem.indexOfScalar(u8, base, ':') orelse return a.dupe(u8, ref);
    const base_scheme = base[0..bcolon];
    if (ref_scheme) |rs| {
        if (!std.ascii.eqlIgnoreCase(rs, base_scheme)) return a.dupe(u8, ref);
    }
    var rref = ref;
    if (ref_scheme) |rs| rref = ref[rs.len + 1 ..];
    // base parts
    var brest = base[bcolon + 1 ..];
    var b_auth: ?[]const u8 = null;
    if (std.mem.startsWith(u8, brest, "//")) {
        brest = brest[2..];
        const e = std.mem.indexOfAny(u8, brest, "/?#") orelse brest.len;
        b_auth = brest[0..e];
        brest = brest[e..];
    }
    var b_frag_i = std.mem.indexOfScalar(u8, brest, '#') orelse brest.len;
    _ = &b_frag_i;
    brest = brest[0..b_frag_i];
    var b_query: ?[]const u8 = null;
    var b_path = brest;
    if (std.mem.indexOfScalar(u8, brest, '?')) |i| {
        b_query = brest[i + 1 ..];
        b_path = brest[0..i];
    }
    // ref parts
    var r_auth: ?[]const u8 = null;
    var rr = rref;
    if (std.mem.startsWith(u8, rr, "//")) {
        rr = rr[2..];
        const e = std.mem.indexOfAny(u8, rr, "/?#") orelse rr.len;
        r_auth = rr[0..e];
        rr = rr[e..];
    }
    var r_frag: ?[]const u8 = null;
    if (std.mem.indexOfScalar(u8, rr, '#')) |i| {
        r_frag = rr[i + 1 ..];
        rr = rr[0..i];
    }
    var r_query: ?[]const u8 = null;
    var r_path = rr;
    if (std.mem.indexOfScalar(u8, rr, '?')) |i| {
        r_query = rr[i + 1 ..];
        r_path = rr[0..i];
    }
    var path: []u8 = undefined;
    var auth = b_auth;
    var query = r_query;
    if (r_auth != null) {
        auth = r_auth;
        path = try removeDotSegments(a, r_path);
    } else if (r_path.len == 0) {
        path = try a.dupe(u8, b_path);
        if (query == null) query = b_query;
    } else if (r_path[0] == '/') {
        path = try removeDotSegments(a, r_path);
    } else {
        const cut = std.mem.lastIndexOfScalar(u8, b_path, '/');
        var joined: std.ArrayList(u8) = .empty;
        defer joined.deinit(a);
        if (cut) |c| try joined.appendSlice(a, b_path[0 .. c + 1]) else if (b_auth != null) try joined.append(a, '/');
        try joined.appendSlice(a, r_path);
        path = try removeDotSegments(a, joined.items);
    }
    defer a.free(path);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, base_scheme);
    try out.append(a, ':');
    if (auth) |au| {
        try out.appendSlice(a, "//");
        try out.appendSlice(a, au);
    }
    try out.appendSlice(a, path);
    if (query) |q| {
        try out.append(a, '?');
        try out.appendSlice(a, q);
    }
    if (r_frag) |fr| if (fr.len != 0) {
        try out.append(a, '#');
        try out.appendSlice(a, fr);
    };
    return out.toOwnedSlice(a);
}

// ---- HTTP/1.1 response head ------------------------------------------------------------------------------------------
pub const Head = struct {
    status: u16,
    content_encoding: ?[]const u8 = null,
    content_type: ?[]const u8 = null,
    content_length: ?[]const u8 = null,
    location: ?[]const u8 = null,
    chunked: bool = false,
};

/// Parse a response head (status line and headers, without the final blank line).
pub fn parseHead(text: []const u8) ?Head {
    var lines = std.mem.splitSequence(u8, text, "\r\n");
    const status_line = lines.next() orelse return null;
    if (!std.mem.startsWith(u8, status_line, "HTTP/1.")) return null;
    var parts = std.mem.tokenizeScalar(u8, status_line, ' ');
    _ = parts.next();
    const code = parts.next() orelse return null;
    if (code.len != 3) return null;
    var h: Head = .{ .status = std.fmt.parseInt(u16, code, 10) catch return null };
    var count: usize = 0;
    while (lines.next()) |line| {
        count += 1;
        if (count > 100) return null;
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const name = line[0..colon];
        const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
        if (std.ascii.eqlIgnoreCase(name, "content-encoding")) {
            h.content_encoding = value;
        } else if (std.ascii.eqlIgnoreCase(name, "content-type")) {
            h.content_type = value;
        } else if (std.ascii.eqlIgnoreCase(name, "content-length")) {
            h.content_length = value;
        } else if (std.ascii.eqlIgnoreCase(name, "location")) {
            h.location = value;
        } else if (std.ascii.eqlIgnoreCase(name, "transfer-encoding")) {
            if (std.ascii.findIgnoreCase(value, "chunked") != null) h.chunked = true;
        }
    }
    return h;
}

/// Python's `Content-Type` media type: before ';', stripped, lower-cased.
pub fn mediaType(buf: []u8, content_type: []const u8) []const u8 {
    const semi = std.mem.indexOfScalar(u8, content_type, ';') orelse content_type.len;
    const t = std.mem.trim(u8, content_type[0..semi], " \t\r\n\x0b\x0c");
    const n = @min(t.len, buf.len);
    return std.ascii.lowerString(buf[0..n], t[0..n]);
}

// ---- the transfer ----------------------------------------------------------------------------------------------------
fn nowNs(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn remainingNs(io: std.Io, deadline_ns: i96, diag: ?*Diag) errors.Error!i96 {
    const r = deadline_ns - nowNs(io);
    if (r <= 0) return errors.fail(diag, "image download timed out", .{});
    return r;
}

fn setTimeouts(handle: std.posix.fd_t, ns: i96) void {
    const clamped: i96 = @max(ns, 1_000_000);
    var tv: std.c.timeval = .{ .sec = @intCast(@divTrunc(clamped, 1_000_000_000)), .usec = @intCast(@divTrunc(@mod(clamped, 1_000_000_000), 1000)) };
    inline for (.{ std.c.SO.RCVTIMEO, std.c.SO.SNDTIMEO }) |opt| {
        _ = std.c.setsockopt(handle, std.c.SOL.SOCKET, opt, @ptrCast(&tv), @sizeOf(std.c.timeval));
    }
}

extern "c" fn socket(domain: c_int, kind: c_int, protocol: c_int) c_int;
extern "c" fn connect(fd: c_int, addr: *const anyopaque, len: u32) c_int;
extern "c" fn close(fd: c_int) c_int;

/// A blocking TCP connect bounded by `ns` (SO_SNDTIMEO applies to connect(2) on Linux).
fn connectBounded(address: net.IpAddress, ns: i96) ?net.Stream {
    const v6 = address == .ip6;
    const fd = socket(if (v6) 10 else 2, 1 | 0x80000, 0);
    if (fd < 0) return null;
    setTimeouts(fd, ns);
    var rc: c_int = -1;
    switch (address) {
        .ip4 => |x| {
            var sa: extern struct { family: u16, port: u16, addr: [4]u8, zero: [8]u8 } = .{ .family = 2, .port = std.mem.nativeToBig(u16, 443), .addr = x.bytes, .zero = @splat(0) };
            rc = connect(fd, &sa, @sizeOf(@TypeOf(sa)));
        },
        .ip6 => |x| {
            var sa: extern struct { family: u16, port: u16, flowinfo: u32, addr: [16]u8, scope: u32 } = .{ .family = 10, .port = std.mem.nativeToBig(u16, 443), .flowinfo = 0, .addr = x.bytes, .scope = x.interface.index };
            rc = connect(fd, &sa, @sizeOf(@TypeOf(sa)));
        },
    }
    if (rc != 0) {
        _ = close(fd);
        return null;
    }
    var a = address;
    switch (a) {
        .ip4 => |*x| x.port = 443,
        .ip6 => |*x| x.port = 443,
    }
    return .{ .socket = .{ .handle = fd, .address = a } };
}

const Reply = struct {
    body: ?[]u8 = null, // owned
    location: ?[]u8 = null, // owned
};

/// One GET on a checked address; returns the body, or the redirect target.
fn request(a: Allocator, io: std.Io, bundle: *std.crypto.Certificate.Bundle, bundle_lock: *std.Io.RwLock, host: []const u8, addr: net.IpAddress, target: []const u8, req: frontend.FetchRequest, diag: ?*Diag) (Allocator.Error || errors.Error)!Reply {
    const generic = "image download failed or timed out";
    _ = try remainingNs(io, req.deadline_ns, diag);
    const stream = connectBounded(addr, try remainingNs(io, req.deadline_ns, diag)) orelse return errors.fail(diag, generic, .{});
    defer stream.close(io);
    setTimeouts(stream.socket.handle, try remainingNs(io, req.deadline_ns, diag));

    const tls_buf = std.crypto.tls.Client.min_buffer_len;
    const bufs = try a.alloc(u8, tls_buf * 4);
    defer a.free(bufs);
    var sock_read = stream.reader(io, bufs[0..tls_buf]);
    var sock_write = stream.writer(io, bufs[tls_buf .. 2 * tls_buf]);
    var entropy: [std.crypto.tls.Client.Options.entropy_len]u8 = undefined;
    io.random(&entropy);
    var tls = std.crypto.tls.Client.init(&sock_read.interface, &sock_write.interface, .{
        .host = .{ .explicit = host },
        .ca = .{ .bundle = .{ .gpa = a, .io = io, .lock = bundle_lock, .bundle = bundle } },
        .read_buffer = bufs[2 * tls_buf .. 3 * tls_buf],
        .write_buffer = bufs[3 * tls_buf ..],
        .entropy = &entropy,
        .realtime_now = std.Io.Clock.real.now(io),
        .allow_truncation_attacks = true,
    }) catch return errors.fail(diag, generic, .{});

    var accept: std.ArrayList(u8) = .empty;
    defer accept.deinit(a);
    const types: []const []const u8 = if (req.video) &video_types else &image_types;
    for (types, 0..) |t, i| {
        if (i != 0) try accept.appendSlice(a, ", ");
        try accept.appendSlice(a, t);
    }
    const head = try std.fmt.allocPrint(a, "GET {s} HTTP/1.1\r\nHost: {s}\r\nAccept: {s}\r\nAccept-Encoding: identity\r\nUser-Agent: {s}\r\nConnection: close\r\n\r\n", .{ target, host, accept.items, user_agent });
    defer a.free(head);
    tls.writer.writeAll(head) catch return errors.fail(diag, generic, .{});
    tls.writer.flush() catch return errors.fail(diag, generic, .{});
    sock_write.interface.flush() catch return errors.fail(diag, generic, .{});

    // response head
    var head_buf: std.ArrayList(u8) = .empty;
    defer head_buf.deinit(a);
    var body_start: std.ArrayList(u8) = .empty;
    defer body_start.deinit(a);
    var tmp: [4096]u8 = undefined;
    while (true) {
        setTimeouts(stream.socket.handle, try remainingNs(io, req.deadline_ns, diag));
        const n = tls.reader.readSliceShort(&tmp) catch return errors.fail(diag, generic, .{});
        if (n == 0) return errors.fail(diag, generic, .{});
        try head_buf.appendSlice(a, tmp[0..n]);
        if (std.mem.indexOf(u8, head_buf.items, "\r\n\r\n")) |i| {
            try body_start.appendSlice(a, head_buf.items[i + 4 ..]);
            head_buf.shrinkRetainingCapacity(i);
            break;
        }
        if (head_buf.items.len > 65536) return errors.fail(diag, generic, .{});
    }
    const h = parseHead(head_buf.items) orelse return errors.fail(diag, generic, .{});
    if (h.status == 301 or h.status == 302 or h.status == 303 or h.status == 307 or h.status == 308) {
        const loc = h.location orelse return errors.fail(diag, "image redirect has no destination", .{});
        return .{ .location = try a.dupe(u8, loc) };
    }
    if (h.status != 200) return errors.fail(diag, "image download returned HTTP {d}", .{h.status});
    if (h.content_encoding) |ce| if (!std.ascii.eqlIgnoreCase(ce, "identity")) return errors.fail(diag, "compressed HTTP image responses are unsupported", .{});
    var mbuf: [128]u8 = undefined;
    const media = mediaType(&mbuf, h.content_type orelse "");
    var allowed = false;
    for (types) |t| allowed = allowed or std.mem.eql(u8, t, media);
    if (!allowed) {
        if (req.video) return errors.fail(diag, "media URL content type must be one of video/mp4, video/webm, video/quicktime, video/x-matroska", .{});
        return errors.fail(diag, "image URL content type must be JPEG, PNG or WebP", .{});
    }
    if (h.content_length) |cl| {
        var ok = cl.len != 0;
        for (cl) |c| ok = ok and c >= '0' and c <= '9';
        const too_big = if (ok) blk: {
            const v = std.fmt.parseInt(u64, cl, 10) catch break :blk true;
            break :blk v > req.max_bytes;
        } else true;
        if (too_big) return errors.fail(diag, "image response exceeds the encoded byte limit", .{});
    }
    // body
    var data: std.ArrayList(u8) = .empty;
    errdefer data.deinit(a);
    const too_large = "image response exceeds the encoded byte limit";
    if (h.chunked) {
        var raw: std.ArrayList(u8) = try body_start.clone(a);
        defer raw.deinit(a);
        var pos: usize = 0;
        while (true) {
            // chunk size line
            var line_end: ?usize = null;
            while (true) {
                if (std.mem.indexOfPos(u8, raw.items, pos, "\r\n")) |i| {
                    line_end = i;
                    break;
                }
                setTimeouts(stream.socket.handle, try remainingNs(io, req.deadline_ns, diag));
                const n = tls.reader.readSliceShort(&tmp) catch return errors.fail(diag, generic, .{});
                if (n == 0) return errors.fail(diag, generic, .{});
                try raw.appendSlice(a, tmp[0..n]);
                if (raw.items.len - pos > 65536 + req.max_bytes and data.items.len > req.max_bytes) return errors.fail(diag, too_large, .{});
            }
            const size_text = std.mem.trim(u8, raw.items[pos..line_end.?], " \t");
            const semi = std.mem.indexOfScalar(u8, size_text, ';') orelse size_text.len;
            const size = std.fmt.parseInt(u64, size_text[0..semi], 16) catch return errors.fail(diag, generic, .{});
            pos = line_end.? + 2;
            if (size == 0) break;
            if (data.items.len + size > req.max_bytes) return errors.fail(diag, too_large, .{});
            while (raw.items.len < pos + size + 2) {
                setTimeouts(stream.socket.handle, try remainingNs(io, req.deadline_ns, diag));
                const n = tls.reader.readSliceShort(&tmp) catch return errors.fail(diag, generic, .{});
                if (n == 0) return errors.fail(diag, generic, .{});
                try raw.appendSlice(a, tmp[0..n]);
            }
            try data.appendSlice(a, raw.items[pos .. pos + size]);
            pos += size + 2;
            // drop consumed bytes from time to time
            if (pos > (1 << 20)) {
                const rest = raw.items[pos..];
                std.mem.copyForwards(u8, raw.items[0..rest.len], rest);
                raw.shrinkRetainingCapacity(rest.len);
                pos = 0;
            }
        }
    } else {
        try data.appendSlice(a, body_start.items);
        var expected: ?u64 = null;
        if (h.content_length) |cl| expected = std.fmt.parseInt(u64, cl, 10) catch null;
        while (data.items.len <= req.max_bytes) {
            if (expected) |e| if (data.items.len >= e) break;
            setTimeouts(stream.socket.handle, try remainingNs(io, req.deadline_ns, diag));
            const n = tls.reader.readSliceShort(&tmp) catch break;
            if (n == 0) break;
            try data.appendSlice(a, tmp[0..n]);
        }
        if (expected) |e| if (data.items.len < e) return errors.fail(diag, generic, .{});
        if (expected) |e| if (data.items.len > e) data.shrinkRetainingCapacity(@intCast(e));
    }
    if (data.items.len > req.max_bytes) return errors.fail(diag, too_large, .{});
    _ = try remainingNs(io, req.deadline_ns, diag);
    return .{ .body = try data.toOwnedSlice(a) };
}

/// Resolve `host` and require every address to be public.
fn resolve(a: Allocator, io: std.Io, host: []const u8, deadline_ns: i96, diag: ?*Diag) (Allocator.Error || errors.Error)!net.IpAddress {
    _ = try remainingNs(io, deadline_ns, diag);
    // literal addresses need no lookup
    if (net.IpAddress.parse(host, 443)) |lit| {
        if (!publicAddress(lit)) return errors.fail(diag, "image URLs must resolve only to public internet addresses", .{});
        return lit;
    } else |_| {}
    _ = a;
    const name_max = net.HostName.max_len;
    const trimmed = std.mem.trimEnd(u8, host, ".");
    if (trimmed.len == 0 or trimmed.len > name_max) return errors.fail(diag, "image host could not be resolved", .{});
    const hn = net.HostName.init(trimmed) catch return errors.fail(diag, "image host could not be resolved", .{});
    var results: [32]net.HostName.LookupResult = undefined;
    var queue: std.Io.Queue(net.HostName.LookupResult) = .init(&results);
    var canonical: [net.HostName.max_len]u8 = undefined;
    hn.lookup(io, &queue, .{ .port = 443, .canonical_name_buffer = &canonical }) catch return errors.fail(diag, "image host could not be resolved", .{});
    var first: ?net.IpAddress = null;
    var all_public = true;
    var any = false;
    while (true) {
        const item = queue.getOne(io) catch break;
        switch (item) {
            .address => |ad| {
                any = true;
                if (!publicAddress(ad)) all_public = false;
                if (first == null) first = ad;
            },
            .canonical_name => {},
        }
    }
    if (!any) return errors.fail(diag, "image host could not be resolved", .{});
    if (!all_public) return errors.fail(diag, "image URLs must resolve only to public internet addresses", .{});
    return first.?;
}

/// Shared state of a fetcher: the CA bundle, loaded on first use.
pub const Fetch = struct {
    allocator: Allocator,
    bundle: std.crypto.Certificate.Bundle = .empty,
    lock: std.Io.RwLock = .init,
    loaded: bool = false,
    mutex: std.Io.Mutex = .init,

    pub fn init(a: Allocator) Fetch {
        return .{ .allocator = a };
    }

    pub fn deinit(f: *Fetch) void {
        f.bundle.deinit(f.allocator);
    }

    pub fn fetcher(f: *Fetch) frontend.Fetcher {
        return .{ .ctx = f, .fetch = fetchOp };
    }

    fn ensureBundle(f: *Fetch, io: std.Io) bool {
        f.mutex.lockUncancelable(io);
        defer f.mutex.unlock(io);
        if (f.loaded) return true;
        f.bundle.rescan(f.allocator, io, std.Io.Clock.real.now(io)) catch return false;
        f.loaded = true;
        return true;
    }
};

fn fetchOp(ctx: ?*anyopaque, a: Allocator, io: std.Io, req: frontend.FetchRequest, diag: ?*Diag) (Allocator.Error || errors.Error)![]u8 {
    const f: *Fetch = @ptrCast(@alignCast(ctx.?));
    if (!f.ensureBundle(io)) return errors.fail(diag, "image download failed or timed out", .{});
    var url: []u8 = try a.dupe(u8, req.url);
    defer a.free(url);
    var redirect: u32 = 0;
    while (redirect <= req.max_redirects) : (redirect += 1) {
        const p = try parseUrl(a, url, req.max_url_chars, diag);
        defer p.deinit(a);
        const addr = try resolve(a, io, p.host, req.deadline_ns, diag);
        const reply = try request(a, io, &f.bundle, &f.lock, p.host, addr, p.target, req, diag);
        if (reply.body) |b| return b;
        defer a.free(reply.location.?);
        if (redirect == req.max_redirects) return errors.fail(diag, "image download has too many redirects", .{});
        const next = try urljoin(a, url, reply.location.?);
        a.free(url);
        url = next;
    }
    return errors.fail(diag, "image download failed", .{});
}

const fixtures = @import("fixtures.zig");

test "URL validation against images_http._url" {
    const a = std.testing.allocator;
    const text = try fixtures.read(a, "fetch.json", 1 << 22);
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer parsed.deinit();
    for (parsed.value.object.get("url").?.array.items) |e| {
        const o = e.object;
        const url = o.get("url").?.string;
        var diag: Diag = .{};
        const res = parseUrl(a, url, 4096, &diag);
        if (o.get("error")) |want| {
            if (res) |p| {
                p.deinit(a);
                std.debug.print("accepted {s} (Python: {s})\n", .{ url, want.string });
                return error.Mismatch;
            } else |_| {
                if (!std.mem.eql(u8, want.string, diag.message())) {
                    std.debug.print("{s}: '{s}' vs '{s}'\n", .{ url, diag.message(), want.string });
                    return error.Mismatch;
                }
            }
        } else {
            const p = res catch {
                // Python accepts internationalized host names; this implementation does not (documented)
                for (url) |c| if (c >= 0x80) break else continue;
                if (std.unicode.utf8ValidateSlice(url) and blk: {
                    for (url) |c| if (c >= 0x80) break :blk true;
                    break :blk false;
                }) continue;
                std.debug.print("refused {s}: {s}\n", .{ url, diag.message() });
                return error.Mismatch;
            };
            defer p.deinit(a);
            try std.testing.expectEqualStrings(o.get("host").?.string, p.host);
            try std.testing.expectEqualStrings(o.get("target").?.string, p.target);
        }
    }
}

test "public address classification against images_http._public_ip" {
    const a = std.testing.allocator;
    const text = try fixtures.read(a, "fetch.json", 1 << 22);
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer parsed.deinit();
    var bad: usize = 0;
    for (parsed.value.object.get("ip").?.array.items) |e| {
        const ip = e.object.get("ip").?.string;
        const want = e.object.get("public").?.bool;
        if (publicText(ip) != want) {
            std.debug.print("{s}: public={} expected {}\n", .{ ip, publicText(ip), want });
            bad += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
}

test "redirect joining against urllib.parse.urljoin" {
    const a = std.testing.allocator;
    const text = try fixtures.read(a, "fetch.json", 1 << 22);
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer parsed.deinit();
    var bad: usize = 0;
    for (parsed.value.object.get("join").?.array.items) |e| {
        const o = e.object;
        const got = try urljoin(a, o.get("base").?.string, o.get("loc").?.string);
        defer a.free(got);
        if (!std.mem.eql(u8, got, o.get("out").?.string)) {
            std.debug.print("urljoin({s}, {s}) = {s}, Python {s}\n", .{ o.get("base").?.string, o.get("loc").?.string, got, o.get("out").?.string });
            bad += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), bad);
}

test "response head and media type" {
    const h = parseHead("HTTP/1.1 200 OK\r\nContent-Type: Image/PNG; charset=x\r\ncontent-length: 12\r\nTransfer-Encoding: chunked\r\n").?;
    try std.testing.expectEqual(@as(u16, 200), h.status);
    try std.testing.expect(h.chunked);
    var b: [64]u8 = undefined;
    try std.testing.expectEqualStrings("image/png", mediaType(&b, h.content_type.?));
    try std.testing.expect(parseHead("garbage") == null);
}

fn liveFetch(f: *Fetch, url: []const u8, video: bool, diag: *Diag) ![]u8 {
    const a = std.testing.allocator;
    const io = std.testing.io;
    return fetchOp(f, a, io, .{ .url = url, .max_bytes = 10 * 1024 * 1024, .deadline_ns = nowNs(io) + 20 * std.time.ns_per_s, .max_redirects = 3, .max_url_chars = 4096, .video = video }, diag);
}

test "refusals need no network: private hosts, ports, schemes" {
    const a = std.testing.allocator;
    var f = Fetch.init(a);
    defer f.deinit();
    var diag: Diag = .{};
    const cases = [_]struct { url: []const u8, msg: []const u8 }{
        .{ .url = "https://127.0.0.1/a.png", .msg = "image URLs must resolve only to public internet addresses" },
        .{ .url = "https://[::1]/a.png", .msg = "image URLs must resolve only to public internet addresses" },
        .{ .url = "https://10.1.2.3/a.png", .msg = "image URLs must resolve only to public internet addresses" },
        .{ .url = "https://169.254.169.254/latest/meta-data", .msg = "image URLs must resolve only to public internet addresses" },
        .{ .url = "https://localhost/a.png", .msg = "image URLs must use public internet hosts" },
        .{ .url = "https://example.com:8443/a.png", .msg = "image URL must be HTTPS on port 443, without credentials or a fragment" },
        .{ .url = "http://example.com/a.png", .msg = "image URL must be HTTPS on port 443, without credentials or a fragment" },
    };
    for (cases) |c| {
        try std.testing.expectError(error.Input, liveFetch(&f, c.url, false, &diag));
        try std.testing.expectEqualStrings(c.msg, diag.message());
    }
}

test "live HTTPS fetch (skipped offline): image bytes, content-type refusal, redirect" {
    const a = std.testing.allocator;
    var f = Fetch.init(a);
    defer f.deinit();
    var diag: Diag = .{};
    const body = liveFetch(&f, "https://www.gstatic.com/webp/gallery/1.webp", false, &diag) catch |e| {
        if (e == error.Input and (std.mem.indexOf(u8, diag.message(), "could not be resolved") != null or std.mem.indexOf(u8, diag.message(), "failed or timed out") != null)) return error.SkipZigTest;
        std.debug.print("live fetch: {s}\n", .{diag.message()});
        return e;
    };
    defer a.free(body);
    try std.testing.expect(body.len > 1000);
    const images = @import("images.zig");
    var rgb = try images.decode(a, body, .default, 1 << 26, &diag);
    defer rgb.deinit(a);
    try std.testing.expect(rgb.width > 100);
    // an HTML page is refused by its content type
    try std.testing.expectError(error.Input, liveFetch(&f, "https://example.com/", false, &diag));
    try std.testing.expectEqualStrings("image URL content type must be JPEG, PNG or WebP", diag.message());
    // video media types
    try std.testing.expectError(error.Input, liveFetch(&f, "https://www.gstatic.com/webp/gallery/1.webp", true, &diag));
    try std.testing.expectEqualStrings("media URL content type must be one of video/mp4, video/webm, video/quicktime, video/x-matroska", diag.message());
    // size cap
    var small = Fetch.init(a);
    defer small.deinit();
    small.bundle = f.bundle; // share the loaded roots for the call below
    small.loaded = true;
    const io = std.testing.io;
    const r = fetchOp(&small, a, io, .{ .url = "https://www.gstatic.com/webp/gallery/1.webp", .max_bytes = 1000, .deadline_ns = nowNs(io) + 20 * std.time.ns_per_s, .max_redirects = 3, .max_url_chars = 4096, .video = false }, &diag);
    small.bundle = .empty;
    try std.testing.expectError(error.Input, r);
    try std.testing.expectEqualStrings("image response exceeds the encoded byte limit", diag.message());
}
