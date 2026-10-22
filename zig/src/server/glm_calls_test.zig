//! The recipe's GLM tool-call text rules against its Python reference: `ThinkSplit`, `GlmCallStreamer`,
//! `close_glm_call`, `repair_glm_keys` and `parse_tool_calls` of cuda/reply_text.py, replayed on the fixtures
//! tools/zig/e2e/gen_glm_calls_fixture.py makes from work/eng-v111 (patches 0001-0121): every prefix of every text.
const std = @import("std");
const json = @import("json");
const glm_calls = @import("glm_calls.zig");
const glm_keys = @import("glm_keys.zig");
const reply_text = @import("reply_text.zig");
const tool_parse = @import("tool_parse.zig");
const Value = json.Value;

const fixture = @embedFile("fixtures_glm_calls.json");

/// The first `chars` characters of `text` (the fixture counts Python characters).
fn charPrefix(text: []const u8, chars: usize) []const u8 {
    var n: usize = 0;
    var i: usize = 0;
    while (i < text.len and n < chars) : (n += 1) {
        i += std.unicode.utf8ByteSequenceLength(text[i]) catch 1;
    }
    return text[0..@min(i, text.len)];
}

fn expectDeltas(a: std.mem.Allocator, want: Value, got: []const Value, label: []const u8) !void {
    errdefer std.debug.print("case {s}\n", .{label});
    try std.testing.expectEqual(want.array.len, got.len);
    for (want.array, got) |w, g| {
        const one = g.get("tool_calls").?.array[0];
        try std.testing.expectEqualStrings(w.array[1].string, one.get("function").?.strField("name").?);
        try std.testing.expectEqualStrings(w.array[2].string, one.get("function").?.strField("arguments").?);
        try std.testing.expectEqual(w.array[0].int64().?, one.get("index").?.int64().?);
        try std.testing.expect(std.mem.startsWith(u8, one.strField("id").?, "call_"));
        try std.testing.expectEqual(@as(usize, 29), one.strField("id").?.len);
    }
    _ = a;
}

fn expectSplit(want: Value, got: reply_text.Split, stream_from: ?usize, text: []const u8, label: []const u8) !void {
    errdefer std.debug.print("case {s}\n", .{label});
    try std.testing.expectEqualStrings(want.array[0].string, got.reasoning);
    try std.testing.expectEqualStrings(want.array[1].string, got.answer);
    switch (want.array[2]) {
        .null => try std.testing.expectEqual(@as(?usize, null), stream_from),
        else => try std.testing.expectEqual(@as(usize, @intCast(want.array[2].int64().?)), reply_text.charCount(text[0..stream_from.?])),
    }
}

test "ThinkSplit and the call streamer follow the Python reference on every prefix of every text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const root = (try json.parse(arena.allocator(), fixture)).ok;
    const tools = root.get("tools").?.array;
    glm_keys.enabled = true; // the recipe's end parser puts a missing <arg_key> back
    defer glm_keys.enabled = false;
    var checked: usize = 0;
    for (root.get("streamer").?.array) |case| {
        var local: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer local.deinit();
        const a = local.allocator();
        const label = try std.fmt.allocPrint(a, "{s}/step {d}", .{ case.strField("label").?, case.get("step").?.int64().? });
        const text = case.strField("text").?;
        const thinking = case.get("thinking").?.bool;
        var think: glm_calls.ThinkSplit = .{ .a = a, .hold = glm_calls.think_call_hold };
        var streamer = try glm_calls.CallStreamer.init(a, tools);
        for (case.get("steps").?.array) |step| {
            const prefix = charPrefix(text, @intCast(step.get("text_len").?.int64().?));
            var feed_text = prefix;
            if (thinking) {
                const got = try think.split(prefix, false);
                try expectSplit(step.get("split").?, got, think.stream_from, prefix, label);
                feed_text = if (think.stream_from) |from| prefix[from..] else "";
            }
            const got = try streamer.feed(feed_text);
            try expectDeltas(a, step.get("deltas").?, got, label);
            try std.testing.expectEqual(step.get("open").?.bool, streamer.open);
            try std.testing.expectEqual(@as(usize, @intCast(step.get("sent").?.int64().?)), streamer.sent);
        }
        var feed_text = text;
        if (thinking) {
            const got = try think.split(text, true);
            try expectSplit(case.get("final_split").?, got, think.stream_from, text, label);
            feed_text = if (think.stream_from) |from| text[from..] else "";
        }
        try expectDeltas(a, case.get("final_deltas").?, try streamer.feed(feed_text), label);
        try std.testing.expectEqual(case.get("open").?.bool, streamer.open);
        try std.testing.expectEqual(@as(usize, @intCast(case.get("sent").?.int64().?)), streamer.sent);
        const want_calls = case.get("calls").?.array;
        try std.testing.expectEqual(want_calls.len, streamer.calls.items.len);
        for (want_calls, streamer.calls.items) |w, c| {
            try std.testing.expectEqualStrings(w.array[0].string, c.name);
            try std.testing.expectEqualStrings(w.array[1].string, c.arguments.items);
        }
        const rest = try streamer.skippedCalls();
        const want_rest = case.get("rest").?.array;
        try std.testing.expectEqual(want_rest.len, rest.len);
        for (want_rest, rest) |w, c| {
            errdefer std.debug.print("case {s} rest\n", .{label});
            try std.testing.expectEqualStrings(w.array[0].string, c.get("function").?.strField("name").?);
            try std.testing.expectEqualStrings(w.array[1].string, c.get("function").?.strField("arguments").?);
        }
        checked += 1;
    }
    try std.testing.expect(checked > 200);
}

test "close_glm_call, repair_glm_keys and the end parser agree with the Python reference" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const root = (try json.parse(arena.allocator(), fixture)).ok;
    const tools = root.get("tools").?.array;
    glm_keys.enabled = true;
    defer glm_keys.enabled = false;
    for (root.get("close").?.array) |case| {
        const a = arena.allocator();
        errdefer std.debug.print("close case {s}\n", .{case.strField("label").?});
        try std.testing.expectEqualStrings(case.strField("closing").?, try glm_calls.closeCall(a, case.strField("text").?, tools));
    }
    for (root.get("repair").?.array) |case| {
        errdefer std.debug.print("repair case {s}\n", .{case.strField("text").?});
        try std.testing.expectEqualStrings(case.strField("repaired").?, try glm_keys.repair(arena.allocator(), case.strField("text").?));
    }
    for (root.get("parse").?.array) |case| {
        const a = arena.allocator();
        errdefer std.debug.print("parse case {s}\n", .{case.strField("label").?});
        for ([_]struct { ?usize, []const u8, []const u8 }{ .{ null, "content", "calls" }, .{ 1, "single_content", "single" } }) |mode| {
            const got = try tool_parse.parse(a, case.strField("text").?, tools, mode[0]);
            // the recipe's parser strips a reply that makes no call too (the Mac port returns it as it is)
            try std.testing.expectEqualStrings(case.strField(mode[1]).?, if (got.calls == null) reply_text.pyStrip(got.content) else got.content);
            const want = case.get(mode[2]).?.array;
            const have = got.calls orelse &[_]Value{};
            try std.testing.expectEqual(want.len, have.len);
            for (want, have) |w, c| {
                try std.testing.expectEqualStrings(w.array[0].string, c.get("function").?.strField("name").?);
                try std.testing.expectEqualStrings(w.array[1].string, c.get("function").?.strField("arguments").?);
            }
        }
    }
}
