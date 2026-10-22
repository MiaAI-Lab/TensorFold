//! The reasoning of GLM's tool-calling replies, put back into later requests whose client dropped it (recipe patch
//! 0036; Python `families/glm5_next/cuda/kept_reasoning.py`, whose ReasoningMemory idea is jayleaton/glm53-tensorfold-spark
//! patch 0620, Apache-2.0).
//!
//! GLM-5.3's template renders the reasoning of every assistant step after the last user message (interleaved thinking
//! inside one tool loop). A client that does not send a step's reasoning back gets `<think></think>` for it: a context
//! the model never wrote, and a prompt that stops matching the previous one at that step, so the prefix cache re-reads
//! the rest. With the reasoning kept here, each prompt of a chain continues the last one as it was generated. A step
//! that carries its own reasoning (`reasoning_content` or `reasoning`, or a `<think>` block in its content) is never
//! changed; an empty one is none.
//!
//! TF_GLM_KEEP_REASONING: replies kept (default 1024; 0 turns this off); TF_GLM_KEEP_REASONING_MB: their text at most
//! (default 32 MiB); the oldest go first.
const std = @import("std");
const json = @import("json");
const reply_text = @import("reply_text.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

const Entry = struct { text: []u8, keys: [][]u8 };

pub const KeptReasoning = struct {
    gpa: Allocator,
    io: std.Io,
    entries: u32,
    chars: usize,
    mutex: std.Io.Mutex = .init,
    items: std.AutoArrayHashMapUnmanaged(u64, Entry) = .empty, // insertion order, oldest first (moved on a hit)
    by_key: std.StringHashMapUnmanaged(u64) = .empty,
    size: usize = 0,
    next: u64 = 0,

    pub fn init(gpa: Allocator, io: std.Io, entries: u32, megabytes: u32) KeptReasoning {
        return .{ .gpa = gpa, .io = io, .entries = entries, .chars = @as(usize, megabytes) << 20 };
    }

    pub fn deinit(k: *KeptReasoning) void {
        for (k.items.values()) |e| k.free(e);
        k.items.deinit(k.gpa);
        k.by_key.deinit(k.gpa);
    }

    fn free(k: *KeptReasoning, e: Entry) void {
        for (e.keys) |key| k.gpa.free(key);
        k.gpa.free(e.keys);
        k.gpa.free(e.text);
    }

    /// Keep the `reasoning` of a reply that made `calls` after the request's `messages`.
    pub fn remember(k: *KeptReasoning, a: Allocator, messages: Value, calls: []const Value, reasoning: ?[]const u8) Allocator.Error!void {
        const text = reasoning orelse return;
        if (calls.len == 0 or reply_text.pyStrip(text).len == 0 or text.len > k.chars) return;
        const found = try keysOf(a, if (messages == .array) messages.array else &.{}, calls);
        var keys: std.ArrayList([]u8) = .empty;
        errdefer {
            for (keys.items) |key| k.gpa.free(key);
            keys.deinit(k.gpa);
        }
        for (found) |key| try keys.append(k.gpa, try k.gpa.dupe(u8, key));
        const copy = try k.gpa.dupe(u8, text);
        errdefer k.gpa.free(copy);
        const owned = try keys.toOwnedSlice(k.gpa);
        k.mutex.lockUncancelable(k.io);
        defer k.mutex.unlock(k.io);
        const n = k.next;
        k.next += 1;
        try k.items.put(k.gpa, n, .{ .text = copy, .keys = owned });
        k.size += copy.len;
        for (owned) |key| try k.by_key.put(k.gpa, key, n); // a key kept again points at the newer reply
        while (k.items.count() > k.entries or k.size > k.chars) {
            const old = k.items.values()[0];
            const old_n = k.items.keys()[0];
            k.items.orderedRemoveAt(0);
            k.size -= old.text.len;
            for (old.keys) |key| if (k.by_key.get(key)) |at| if (at == old_n) {
                _ = k.by_key.remove(key);
            };
            k.free(old);
        }
    }

    fn find(k: *KeptReasoning, a: Allocator, keys: []const []const u8) Allocator.Error!?[]const u8 {
        k.mutex.lockUncancelable(k.io);
        defer k.mutex.unlock(k.io);
        for (keys) |key| if (k.by_key.get(key)) |n| {
            const index = k.items.getIndex(n) orelse continue;
            const entry = k.items.values()[index];
            const text = try a.dupe(u8, entry.text);
            // a hit is the newest again
            k.items.orderedRemoveAt(index);
            try k.items.put(k.gpa, n, entry);
            return text;
        };
        return null;
    }

    /// `messages` with the kept reasoning of each assistant step that calls tools and carries none (a copy; the
    /// caller's list and messages are left alone).
    pub fn restore(k: *KeptReasoning, a: Allocator, messages: Value) Allocator.Error!Value {
        if (messages != .array or k.items.count() == 0) return messages;
        var out: ?[]Value = null;
        for (messages.array, 0..) |m, i| {
            if (m != .object or !std.mem.eql(u8, m.strField("role") orelse "", "assistant")) continue;
            const calls = m.get("tool_calls") orelse continue;
            if (!calls.truthy()) continue;
            if (reply_text.pyStrip(textOf(m.get("reasoning_content"))).len > 0 or reply_text.pyStrip(textOf(m.get("reasoning"))).len > 0) continue;
            if (std.mem.indexOf(u8, textOf(m.get("content")), "</think>") != null) continue;
            const found = try keysOf(a, messages.array[0..i], if (calls == .array) calls.array else &.{});
            const kept = try k.find(a, found) orelse continue;
            if (out == null) out = try a.dupe(Value, messages.array);
            const copy = try json.copyObject(a, m.object);
            try copy.put(a, "reasoning_content", .{ .string = kept });
            out.?[i] = .{ .object = copy };
        }
        return if (out) |o| .{ .array = o } else messages;
    }
};

/// A content as text: a string, or the text parts of a list joined.
fn textOf(content: ?Value) []const u8 {
    const c = content orelse return "";
    return switch (c) {
        .string => |s| s,
        else => "", // a list's parts are joined by `textJoined` where the signature needs them
    };
}

fn textJoined(a: Allocator, content: ?Value) Allocator.Error![]const u8 {
    const c = content orelse return "";
    switch (c) {
        .string => |s| return s,
        .array => |parts| {
            var out: std.ArrayList(u8) = .empty;
            for (parts) |p| if (p == .object) if (p.strField("text")) |t| try out.appendSlice(a, t);
            return out.items;
        },
        else => return "",
    }
}

fn isOurId(id: []const u8) bool {
    if (id.len != 29 or !std.mem.startsWith(u8, id, "call_")) return false;
    for (id[5..]) |c| if (!std.ascii.isDigit(c) and !(c >= 'a' and c <= 'f')) return false;
    return true;
}

/// The arguments as a JSON value: a string parsed (as given when it is not JSON), none as `{}`.
fn argumentsOf(a: Allocator, v: ?Value) Allocator.Error!Value {
    const x = v orelse return .{ .object = try json.newObject(a) };
    switch (x) {
        .string => |s| {
            if (reply_text.pyStrip(s).len == 0) return .{ .object = try json.newObject(a) };
            return switch (try json.parseText(a, s)) {
                .ok => |parsed| parsed,
                .err => x,
            };
        },
        .null => return .{ .object = try json.newObject(a) },
        else => return x,
    }
}

/// Calls as `[name, arguments]`: ids dropped (clients renumber them), arguments parsed (clients re-serialise).
fn callsOf(a: Allocator, calls: []const Value) Allocator.Error!Value {
    const out = try a.alloc(Value, calls.len);
    for (calls, out) |call, *slot| {
        const f = if (call == .object) call.get("function") else null;
        const function: Value = if (f != null and f.? == .object) f.? else .{ .object = try json.newObject(a) };
        const pair = try a.alloc(Value, 2);
        pair[0] = .{ .string = function.strField("name") orelse "" };
        pair[1] = try argumentsOf(a, function.get("arguments"));
        slot.* = .{ .array = pair };
    }
    return .{ .array = out };
}

/// `json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
fn canonical(a: Allocator, w: *std.ArrayList(u8), v: Value) Allocator.Error!void {
    switch (v) {
        .array => |items| {
            try w.append(a, '[');
            for (items, 0..) |item, i| {
                if (i > 0) try w.append(a, ',');
                try canonical(a, w, item);
            }
            try w.append(a, ']');
        },
        .object => |o| {
            const names = try a.dupe([]const u8, o.keys());
            std.mem.sort([]const u8, names, {}, struct {
                fn less(_: void, x: []const u8, y: []const u8) bool {
                    return std.mem.lessThan(u8, x, y);
                }
            }.less);
            try w.append(a, '{');
            for (names, 0..) |name, i| {
                if (i > 0) try w.append(a, ',');
                try w.appendSlice(a, try json.quote(a, name, .{ .ascii = false }));
                try w.append(a, ':');
                try canonical(a, w, o.get(name).?);
            }
            try w.append(a, '}');
        },
        else => try w.appendSlice(a, try json.stringify(a, v, .{ .ascii = false, .compact = true })),
    }
}

/// The key of `calls` written after `history`: each message's role, text (stripped, as the template renders it) and
/// calls, without ids or reasoning, then the calls.
pub fn signature(a: Allocator, history: []const Value, calls: []const Value) Allocator.Error![]const u8 {
    var shape: std.ArrayList(Value) = .empty;
    for (history) |m| {
        if (m != .object) continue;
        const entry: std.ArrayList(Value) = blk: {
            var e: std.ArrayList(Value) = .empty;
            try e.append(a, .{ .string = m.strField("role") orelse "" });
            try e.append(a, .{ .string = reply_text.pyStrip(try textJoined(a, m.get("content"))) });
            if (std.mem.eql(u8, m.strField("role") orelse "", "assistant")) if (m.get("tool_calls")) |tc| if (tc.truthy() and tc == .array)
                try e.append(a, try callsOf(a, tc.array));
            break :blk e;
        };
        try shape.append(a, .{ .array = entry.items });
    }
    const top = try a.alloc(Value, 2);
    top[0] = .{ .array = shape.items };
    top[1] = try callsOf(a, calls);
    var text: std.ArrayList(u8) = .empty;
    try canonical(a, &text, .{ .array = top });
    var digest: [32]u8 = undefined;
    Sha256.hash(text.items, &digest, .{});
    return std.fmt.allocPrint(a, "sig:{x}", .{&digest});
}

fn keysOf(a: Allocator, history: []const Value, calls: []const Value) Allocator.Error![]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    for (calls) |c| if (c == .object) if (c.strField("id")) |id| if (isOurId(id)) try out.append(a, id);
    try out.append(a, try signature(a, history, calls));
    return out.items;
}

test "a kept reply's reasoning returns to the step whose client dropped it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var k = KeptReasoning.init(std.testing.allocator, std.testing.io, 8, 1);
    defer k.deinit();
    const history = (try json.parse(a, "[{\"role\":\"user\",\"content\":\"go\"}]")).ok;
    const calls = (try json.parse(a, "[{\"id\":\"call_0123456789abcdef01234567\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"a\\\":1}\"}}]")).ok;
    try k.remember(a, history, calls.array, "I should call f.");
    // the next request: the client renumbered the id and dropped the reasoning; the signature finds it
    const next = (try json.parse(a,
        \\[{"role":"user","content":" go "},{"role":"assistant","content":"","tool_calls":[{"id":"x1","type":"function","function":{"name":"f","arguments":{"a":1}}}]},{"role":"tool","tool_call_id":"x1","content":"ok"}]
    )).ok;
    const restored = try k.restore(a, next);
    try std.testing.expectEqualStrings("I should call f.", restored.array[1].strField("reasoning_content").?);
    try std.testing.expect(!next.array[1].has("reasoning_content")); // the caller's messages are left alone
    // a step that carries its own reasoning, or a think block, is never changed
    const own = (try json.parse(a,
        \\[{"role":"user","content":"go"},{"role":"assistant","content":"","reasoning_content":"mine","tool_calls":[{"id":"call_0123456789abcdef01234567","type":"function","function":{"name":"f","arguments":"{}"}}]}]
    )).ok;
    try std.testing.expect((try k.restore(a, own)).array.ptr == own.array.ptr);
    // another conversation is not given it
    const other = (try json.parse(a,
        \\[{"role":"user","content":"different"},{"role":"assistant","content":"","tool_calls":[{"id":"y","type":"function","function":{"name":"f","arguments":{"a":1}}}]}]
    )).ok;
    try std.testing.expect((try k.restore(a, other)).array.ptr == other.array.ptr);
    // the id the server made finds it too
    const by_id = (try json.parse(a,
        \\[{"role":"user","content":"unrelated"},{"role":"assistant","content":"","tool_calls":[{"id":"call_0123456789abcdef01234567","type":"function","function":{"name":"g","arguments":{}}}]}]
    )).ok;
    try std.testing.expectEqualStrings("I should call f.", (try k.restore(a, by_id)).array[1].strField("reasoning_content").?);
}

test "the oldest replies go first, by count and by size" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var k = KeptReasoning.init(std.testing.allocator, std.testing.io, 2, 1);
    defer k.deinit();
    const calls = (try json.parse(a, "[{\"id\":\"call_0123456789abcdef01234567\",\"function\":{\"name\":\"f\",\"arguments\":{}}}]")).ok;
    for (0..4) |i| {
        const history = (try json.parse(a, try std.fmt.allocPrint(a, "[{{\"role\":\"user\",\"content\":\"q{d}\"}}]", .{i}))).ok;
        const c = (try json.parse(a, try std.fmt.allocPrint(a, "[{{\"id\":\"x\",\"function\":{{\"name\":\"f\",\"arguments\":{{\"n\":{d}}}}}}}]", .{i}))).ok;
        try k.remember(a, history, c.array, "reason");
    }
    try std.testing.expectEqual(@as(usize, 2), k.items.count());
    _ = calls;
    // a reply longer than the whole budget is not kept
    var tiny = KeptReasoning.init(std.testing.allocator, std.testing.io, 8, 0);
    defer tiny.deinit();
    try tiny.remember(a, .{ .array = &.{} }, (try json.parse(a, "[{\"function\":{\"name\":\"f\"}}]")).ok.array, "x");
    try std.testing.expectEqual(@as(usize, 0), tiny.items.count());
}
