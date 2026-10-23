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
    pub fn remember(k: *KeptReasoning, messages: Value, calls: []const Value, reasoning: ?[]const u8) Allocator.Error!void {
        const text = reasoning orelse return;
        if (calls.len == 0 or reply_text.pyStrip(text).len == 0 or text.len > k.chars) return;
        var signer: Signer = .init();
        defer signer.deinit();
        if (messages == .array) for (messages.array) |m| try signer.add(m);
        const found = try signer.keys(calls);
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
    /// caller's list and messages are left alone). One pass: each message is signed once, and `a` takes only what is
    /// put back.
    pub fn restore(k: *KeptReasoning, a: Allocator, messages: Value) Allocator.Error!Value {
        if (messages != .array or k.items.count() == 0) return messages;
        var last: ?usize = null;
        for (messages.array, 0..) |m, i| if (bare(m)) {
            last = i;
        };
        const end = (last orelse return messages) + 1;
        var signer: Signer = .init();
        defer signer.deinit();
        var out: ?[]Value = null;
        for (messages.array[0..end], 0..) |m, i| {
            if (bare(m)) if (try k.find(a, try signer.keys(callList(m.get("tool_calls").?)))) |kept| {
                if (out == null) out = try a.dupe(Value, messages.array);
                const copy = try json.copyObject(a, m.object);
                try copy.put(a, "reasoning_content", .{ .string = kept });
                out.?[i] = .{ .object = copy };
            };
            try signer.add(m);
        }
        return if (out) |o| .{ .array = o } else messages;
    }
};

/// An assistant step that calls tools and carries no reasoning of its own (restore's to fill).
fn bare(m: Value) bool {
    if (m != .object or !std.mem.eql(u8, m.strField("role") orelse "", "assistant")) return false;
    const calls = m.get("tool_calls") orelse return false;
    if (!calls.truthy()) return false;
    if (reply_text.pyStrip(textOf(m.get("reasoning_content"))).len > 0 or reply_text.pyStrip(textOf(m.get("reasoning"))).len > 0) return false;
    return std.mem.indexOf(u8, textOf(m.get("content")), "</think>") == null;
}

fn callList(calls: Value) []const Value {
    return if (calls == .array) calls.array else &.{};
}

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

/// A message's part of a signature: its role, its text (stripped, as the template renders it) and an assistant's calls,
/// without ids or reasoning.
fn entryOf(a: Allocator, m: Value) Allocator.Error!Value {
    var e: std.ArrayList(Value) = .empty;
    try e.append(a, .{ .string = m.strField("role") orelse "" });
    try e.append(a, .{ .string = reply_text.pyStrip(try textJoined(a, m.get("content"))) });
    if (std.mem.eql(u8, m.strField("role") orelse "", "assistant")) if (m.get("tool_calls")) |tc| if (tc.truthy() and tc == .array)
        try e.append(a, try callsOf(a, tc.array));
    return .{ .array = e.items };
}

/// The signatures after each message of a conversation, in one pass: `[shape, calls]`'s canonical text is hashed as
/// it goes, each message's entry once, and a step's key finishes a copy of the hash with its calls. Signing each
/// step's whole history again was quadratic in the steps, and kept every history's text in the request's arena; here
/// a message's text is built in scratch that the next one reuses.
const Signer = struct {
    hash: Sha256,
    entries: usize = 0,
    scratch: std.heap.ArenaAllocator,

    fn init() Signer {
        var hash = Sha256.init(.{});
        hash.update("[[");
        return .{ .hash = hash, .scratch = .init(std.heap.page_allocator) };
    }

    fn deinit(s: *Signer) void {
        s.scratch.deinit();
    }

    /// `m` joins the history (a message that is not an object has no entry).
    fn add(s: *Signer, m: Value) Allocator.Error!void {
        if (m != .object) return;
        _ = s.scratch.reset(.retain_capacity);
        const a = s.scratch.allocator();
        var text: std.ArrayList(u8) = .empty;
        if (s.entries > 0) try text.append(a, ',');
        try canonical(a, &text, try entryOf(a, m));
        s.hash.update(text.items);
        s.entries += 1;
    }

    /// The keys of `calls` written after the history so far: the ids the server made, then the signature (in the
    /// scratch, until the next add or keys).
    fn keys(s: *Signer, calls: []const Value) Allocator.Error![]const []const u8 {
        _ = s.scratch.reset(.retain_capacity);
        const a = s.scratch.allocator();
        var out: std.ArrayList([]const u8) = .empty;
        for (calls) |c| if (c == .object) if (c.strField("id")) |id| if (isOurId(id)) try out.append(a, id);
        var text: std.ArrayList(u8) = .empty;
        try text.appendSlice(a, "],");
        try canonical(a, &text, try callsOf(a, calls));
        try text.append(a, ']');
        var hash = s.hash;
        hash.update(text.items);
        var digest: [32]u8 = undefined;
        hash.final(&digest);
        try out.append(a, try std.fmt.allocPrint(a, "sig:{x}", .{&digest}));
        return out.items;
    }
};

/// The key of `calls` written after `history`: each message's role, text (stripped, as the template renders it) and
/// calls, without ids or reasoning, then the calls.
pub fn signature(a: Allocator, history: []const Value, calls: []const Value) Allocator.Error![]const u8 {
    var signer: Signer = .init();
    defer signer.deinit();
    for (history) |m| try signer.add(m);
    const found = try signer.keys(calls);
    return a.dupe(u8, found[found.len - 1]);
}

test "a kept reply's reasoning returns to the step whose client dropped it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var k = KeptReasoning.init(std.testing.allocator, std.testing.io, 8, 1);
    defer k.deinit();
    const history = (try json.parse(a, "[{\"role\":\"user\",\"content\":\"go\"}]")).ok;
    const calls = (try json.parse(a, "[{\"id\":\"call_0123456789abcdef01234567\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"a\\\":1}\"}}]")).ok;
    try k.remember(history, calls.array, "I should call f.");
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
        try k.remember(history, c.array, "reason");
    }
    try std.testing.expectEqual(@as(usize, 2), k.items.count());
    _ = calls;
    // a reply longer than the whole budget is not kept
    var tiny = KeptReasoning.init(std.testing.allocator, std.testing.io, 8, 0);
    defer tiny.deinit();
    try tiny.remember(.{ .array = &.{} }, (try json.parse(a, "[{\"function\":{\"name\":\"f\"}}]")).ok.array, "x");
    try std.testing.expectEqual(@as(usize, 0), tiny.items.count());
}

test "the signatures are kept_reasoning.py's" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // signature(messages[:i], messages[i]["tool_calls"]) from kept_reasoning.py for each step: text parts joined and
    // stripped, a message that is not an object skipped, arguments parsed (or kept as text), keys sorted, non-ASCII as is
    const messages = (try json.parse(a,
        \\[{"role":"system","content":"  You read files.  "},{"role":"user","content":[{"type":"text","text":"Read "},{"type":"image_url","image_url":{"url":"x"}},{"type":"text","text":"a.zig, ünïcode ✓"}]},{"role":"assistant","content":"","tool_calls":[{"id":"call_0123456789abcdef01234567","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.zig\",\"z\":[1,2],\"b\":{\"y\":true,\"x\":null}}"}}]},{"role":"tool","tool_call_id":"call_0123456789abcdef01234567","content":"const a = \"q\";\n\ttab"},7,{"role":"assistant","content":"Next.","reasoning_content":"kept apart","tool_calls":[{"id":"r1","type":"function","function":{"name":"grep","arguments":{"pattern":"fn ","n":3}}},{"id":"r2","type":"function","function":{"name":"ls","arguments":""}}]},{"role":"tool","tool_call_id":"r1","content":"ok"},{"role":"tool","tool_call_id":"r2","content":"not json {"},{"role":"assistant","content":"x","tool_calls":[{"id":"r3","type":"function","function":{"name":"write","arguments":"not json {"}}]}]
    )).ok.array;
    const python = [_]struct { usize, []const u8 }{
        .{ 2, "sig:675ffa5ec9bb9dbed6704ec676468ace78556cec250c524bd9ea082316a6444b" },
        .{ 5, "sig:41682446462895adf60e7fffee858764cf0d65502fc19a9f61d3ed0d1b3e48a0" },
        .{ 8, "sig:2071534fd7f899c59640044d014c7eddc473946aedba57784334be514283183e" },
    };
    for (python) |p| try std.testing.expectEqualStrings(p[1], try signature(a, messages[0..p[0]], messages[p[0]].get("tool_calls").?.array));
    const lone = (try json.parse(a, "[{\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"a\\\":1}\"}}]")).ok.array;
    try std.testing.expectEqualStrings("sig:043f90839f1551cd19205713623501d3da4dc9b47dfdc47e1c1cf7c8976d8075", try signature(a, &.{}, lone));
}

test "one pass puts back each step's own reasoning, and takes nothing from the caller's allocator when none is kept" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var k = KeptReasoning.init(std.testing.allocator, std.testing.io, 64, 1);
    defer k.deinit();
    // a tool loop of steps whose ids the client renumbered: only the signatures find them
    var list: std.ArrayList(Value) = .empty;
    try list.append(a, (try json.parse(a, "{\"role\":\"user\",\"content\":\"go\"}")).ok);
    for (0..6) |i| {
        const step = (try json.parse(a, try std.fmt.allocPrint(a, "{{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{{\"id\":\"c{d}\",\"type\":\"function\",\"function\":{{\"name\":\"f\",\"arguments\":{{\"n\":{d}}}}}}}]}}", .{ i, i }))).ok;
        if (i % 2 == 0) try k.remember(.{ .array = list.items }, step.get("tool_calls").?.array, try std.fmt.allocPrint(a, "thought {d}", .{i}));
        try list.append(a, step);
        try list.append(a, (try json.parse(a, try std.fmt.allocPrint(a, "{{\"role\":\"tool\",\"tool_call_id\":\"c{d}\",\"content\":\"out {d}\"}}", .{ i, i }))).ok);
    }
    const restored = try k.restore(a, .{ .array = list.items });
    for (0..6) |i| {
        const step = restored.array[1 + 2 * i];
        if (i % 2 == 0) {
            try std.testing.expectEqualStrings(try std.fmt.allocPrint(a, "thought {d}", .{i}), step.strField("reasoning_content").?);
        } else try std.testing.expect(!step.has("reasoning_content"));
    }
    // a conversation none of whose steps is kept: every step is looked up, the caller's allocator is never touched
    // (the signatures' text lives in the restore's own scratch)
    const other = try a.dupe(Value, list.items);
    other[0] = (try json.parse(a, "{\"role\":\"user\",\"content\":\"another\"}")).ok;
    const same = try k.restore(std.testing.failing_allocator, .{ .array = other });
    try std.testing.expect(same.array.ptr == other.ptr);
}
