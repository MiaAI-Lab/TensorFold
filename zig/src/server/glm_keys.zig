//! A GLM call whose `<arg_key>` the model left out (recipe patches 0036 and 0053, issue #22): `repair_glm_keys` of
//! `cuda/reply_text.py`.
const std = @import("std");
const reply_text = @import("reply_text.zig");
const Allocator = std.mem.Allocator;

pub const key_open = "<arg_key>";
pub const key_close = "</arg_key>";
pub const value_open = "<arg_value>";
pub const value_close = "</arg_value>";

/// Whether the end parser repairs missing `<arg_key>` marks (the GLM recipe); set once at start.
pub var enabled = false;

const Tag = enum { key_open, key_close, value_open, value_close };

const Found = struct { tag: Tag, start: usize, end: usize };

/// `re.finditer(r"</?arg_(?:key|value)>")`'s next match from `from`.
fn nextTag(text: []const u8, from: usize) ?Found {
    var i = from;
    while (std.mem.indexOfScalarPos(u8, text, i, '<')) |at| {
        const rest = text[at..];
        const tags = [_]struct { []const u8, Tag }{ .{ key_open, .key_open }, .{ key_close, .key_close }, .{ value_open, .value_open }, .{ value_close, .value_close } };
        for (tags) |t| if (std.mem.startsWith(u8, rest, t[0])) return .{ .tag = t[1], .start = at, .end = at + t[0].len };
        i = at + 1;
    }
    return null;
}

fn count(text: []const u8, needle: []const u8) usize {
    return std.mem.count(u8, text, needle);
}

/// `repair_glm_keys(block)`: a GLM call whose `<arg_key>` the model left out, writing whitespace where that token
/// belongs (`read path</arg_key><arg_value>..`, or `</arg_value>\npath</arg_key>`), with it put back: the word before
/// an unopened `</arg_key>` is its key. A block where nothing is missing comes back as it is.
pub fn repair(a: Allocator, block: []const u8) Allocator.Error![]const u8 {
    if (count(block, key_close) <= count(block, key_open)) return block;
    var out: std.ArrayList(u8) = .empty;
    var at: usize = 0; // text copied
    var prev: ?Tag = null; // the last tag seen
    var prev_end: usize = 0;
    var from: usize = 0;
    while (nextTag(block, from)) |tag| {
        from = tag.end;
        if (tag.tag == .key_close and (prev == null or prev.? == .value_close)) {
            const seg_start: usize = if (prev == null) 0 else prev_end;
            // `(\S+)\s*$` searched in [seg_start, tag.start): the last run of non-space characters
            const trimmed = std.mem.trimEnd(u8, block[seg_start..tag.start], " \t\r\n\x0b\x0c");
            if (trimmed.len == 0) return block;
            var word_start = seg_start + trimmed.len;
            while (word_start > seg_start and !reply_text.isSpaceByte(block[word_start - 1])) word_start -= 1;
            // the name's segment keeps a name before the key (`read path`): no word, or no name, is not repaired
            if (prev == null and reply_text.pyStrip(block[seg_start..word_start]).len == 0) return block;
            try out.appendSlice(a, block[at..word_start]);
            try out.appendSlice(a, key_open);
            at = word_start;
        }
        prev = tag.tag;
        prev_end = tag.end;
    }
    try out.appendSlice(a, block[at..]);
    return out.items;
}

test "an unopened arg_key is put back before the word that precedes its close" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // the name, then a key whose opener is missing
    try std.testing.expectEqualStrings(
        "read <arg_key>path</arg_key><arg_value>/tmp/x</arg_value>",
        try repair(a, "read path</arg_key><arg_value>/tmp/x</arg_value>"),
    );
    // after a value, the next key's opener is missing
    try std.testing.expectEqualStrings(
        "write<arg_key>path</arg_key><arg_value>/tmp/x</arg_value>\n<arg_key>text</arg_key><arg_value>hi</arg_value>",
        try repair(a, "write<arg_key>path</arg_key><arg_value>/tmp/x</arg_value>\ntext</arg_key><arg_value>hi</arg_value>"),
    );
}

test "a block where nothing is missing, or nothing can be told, comes back as it is" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const whole = "read<arg_key>path</arg_key><arg_value>/tmp/x</arg_value>";
    try std.testing.expectEqualStrings(whole, try repair(a, whole));
    // no name before the first unopened key: not repaired
    const nameless = "path</arg_key><arg_value>/tmp/x</arg_value>";
    try std.testing.expectEqualStrings(nameless, try repair(a, nameless));
    // no word at all before the close
    const empty = "read </arg_key><arg_value>x</arg_value>";
    try std.testing.expectEqualStrings(empty, try repair(a, empty));
}
