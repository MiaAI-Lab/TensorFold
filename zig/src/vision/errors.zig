//! ImageInputError and ValueError as Zig sees them: one error code and the Python message in a Diag.
//!
//! The server maps every `error.Input` to a 400 whose text is `diag.message()`; a message that starts with
//! `context_limit` is served as context_length_exceeded (errors.CONTEXT_LIMIT in the Python server).
const std = @import("std");

pub const Error = error{Input};

/// OpenAI's wording, which clients match to compact (tensorfold.server.errors.CONTEXT_LIMIT).
pub const context_limit = "This server's maximum context length is";

pub const Diag = struct {
    buf: [1024]u8 = undefined,
    len: usize = 0,

    pub fn message(d: *const Diag) []const u8 {
        return d.buf[0..d.len];
    }

    pub fn isContextLimit(d: *const Diag) bool {
        return std.mem.startsWith(u8, d.message(), context_limit);
    }

    pub fn set(d: *Diag, comptime fmt: []const u8, args: anytype) void {
        const out = std.fmt.bufPrint(&d.buf, fmt, args) catch d.buf[0..d.buf.len];
        d.len = out.len;
    }
};

/// Record `fmt` in `diag` (when given) and return the input error.
pub fn fail(diag: ?*Diag, comptime fmt: []const u8, args: anytype) Error {
    if (diag) |d| d.set(fmt, args);
    return error.Input;
}

/// Python's `f"{n:,}"`.
pub fn commas(buf: []u8, n: u64) []const u8 {
    var tmp: [32]u8 = undefined;
    const digits = std.fmt.bufPrint(&tmp, "{d}", .{n}) catch unreachable;
    var out: usize = 0;
    for (digits, 0..) |c, i| {
        if (i != 0 and (digits.len - i) % 3 == 0) {
            buf[out] = ',';
            out += 1;
        }
        buf[out] = c;
        out += 1;
    }
    return buf[0..out];
}

/// Python's repr() of a str (ASCII/UTF-8 text without exotic code points): quotes chosen as Python does.
pub fn pyRepr(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    const has_single = std.mem.indexOfScalar(u8, s, '\'') != null;
    const has_double = std.mem.indexOfScalar(u8, s, '"') != null;
    const quote: u8 = if (has_single and !has_double) '"' else '\'';
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.append(allocator, quote);
    for (s) |c| {
        switch (c) {
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            else => if (c == quote) {
                try out.append(allocator, '\\');
                try out.append(allocator, c);
            } else if (c < 0x20 or c == 0x7f) {
                var hb: [4]u8 = undefined;
                try out.appendSlice(allocator, try std.fmt.bufPrint(&hb, "\\x{x:0>2}", .{c}));
            } else try out.append(allocator, c),
        }
    }
    try out.append(allocator, quote);
    return out.toOwnedSlice(allocator);
}

test "commas and repr" {
    var b: [32]u8 = undefined;
    try std.testing.expectEqualStrings("1,234,567", commas(&b, 1234567));
    try std.testing.expectEqualStrings("16", commas(&b, 16));
    try std.testing.expectEqualStrings("262,144", commas(&b, 262144));
    const r = try pyRepr(std.testing.allocator, "it's");
    defer std.testing.allocator.free(r);
    try std.testing.expectEqualStrings("\"it's\"", r);
}
