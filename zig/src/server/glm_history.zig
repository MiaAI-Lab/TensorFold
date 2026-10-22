//! A chat history as GLM-5.3's template can render it (recipe patches 0051 and 0036; Python
//! `families/glm5_next/prompts.py:agent_history` and `kept_reasoning.py`), applied before the messages are normalized.
//!
//! * `agentHistory`: a call's arguments sent as "" or null are no arguments; a call whose arguments are not a JSON
//!   object is left out with its result (logged), because an agent replays its history every turn and a refusal would
//!   end the conversation; results whose ids match none of their calls' ids are rendered in the order sent (the
//!   template sorts results by id and silently drops one it cannot place).
//! * `promoteReasoning`: `reasoning` (vLLM's newer replies, OpenRouter) is the step's `reasoning_content`.
const std = @import("std");
const json = @import("json");
const log = @import("log.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// `message.get("tool_call_id") or message.get("id")` of a dict, else None.
fn callId(message: Value) ?Value {
    if (message != .object) return null;
    if (message.get("tool_call_id")) |v| if (v.truthy()) return v;
    if (message.get("id")) |v| if (v.truthy()) return v;
    return null;
}

fn sameId(x: ?Value, y: ?Value) bool {
    if (x == null or y == null) return x == null and y == null;
    return json.equal(x.?, y.?);
}

fn isTool(m: Value) bool {
    return m == .object and m.strField("role") != null and std.mem.eql(u8, m.strField("role").?, "tool");
}

/// Python's `str.strip()` of a string being empty.
fn blank(s: []const u8) bool {
    return std.mem.trim(u8, s, " \t\r\n\x0b\x0c").len == 0;
}

fn withoutKey(a: Allocator, o: *const json.Object, keys: []const []const u8) Allocator.Error!Value {
    const out = try json.newObject(a);
    for (o.keys(), o.values()) |k, v| {
        var skip = false;
        for (keys) |drop| {
            if (std.mem.eql(u8, k, drop)) skip = true;
        }
        if (!skip) try out.put(a, k, v);
    }
    return .{ .object = out };
}

/// `agent_history(messages)`: `messages` itself when nothing needs fixing.
pub fn agentHistory(a: Allocator, messages: Value) Allocator.Error!Value {
    if (messages != .array) return messages;
    var out: std.ArrayList(Value) = .empty;
    try out.appendSlice(a, messages.array);
    var changed = false;
    var i: usize = 0;
    while (i < out.items.len) : (i += 1) {
        const message = out.items[i];
        const role = message.strField("role") orelse "";
        const calls_v = if (message == .object and std.mem.eql(u8, role, "assistant")) message.get("tool_calls") else null;
        if (calls_v == null or calls_v.? != .array or calls_v.?.array.len == 0) continue;
        const calls = calls_v.?.array;
        var fixed: std.ArrayList(Value) = .empty;
        var bad: std.ArrayList(usize) = .empty;
        var mended = false;
        for (calls, 0..) |call, j| {
            const function: ?Value = if (call == .object) call.get("function") else null;
            const have_function = function != null and function.? == .object;
            var arguments: ?Value = .{ .object = try json.newObject(a) }; // `{}` when the function is no dict
            if (have_function) arguments = function.?.get("arguments") orelse .null;
            const a_val = arguments.?;
            if (a_val == .null or (a_val == .string and blank(a_val.string))) {
                const fn_copy = try json.copyObject(a, function.?.object);
                try fn_copy.put(a, "arguments", .{ .object = try json.newObject(a) });
                const call_copy = try json.copyObject(a, call.object);
                try call_copy.put(a, "function", .{ .object = fn_copy });
                try fixed.append(a, .{ .object = call_copy });
                mended = true;
                continue;
            } else if (a_val == .string) {
                const parsed: ?Value = switch (try json.parseText(a, a_val.string)) {
                    .ok => |v| v,
                    .err => null,
                };
                if (parsed == null or parsed.? != .object) {
                    const shown = a_val.string[0..@min(a_val.string.len, 80)];
                    const id = callId(call);
                    log.line("messages[{d}].tool_calls[{d}] ({s}): arguments are not a JSON object ('{s}'); the call and its result are left out of the prompt", .{ i, j, if (id) |x| (if (x == .string) x.string else "?") else "None", shown });
                    try bad.append(a, j);
                    continue;
                }
            }
            try fixed.append(a, call);
        }
        var end = i + 1;
        while (end < out.items.len and isTool(out.items[end])) end += 1;
        if (bad.items.len > 0) {
            changed = true;
            var ids: std.ArrayList(Value) = .empty;
            for (bad.items) |j| if (callId(calls[j])) |id| try ids.append(a, id);
            const by_place = ids.items.len == 0 and end - i - 1 == calls.len;
            var keep: std.ArrayList(Value) = .empty;
            for (out.items[i + 1 .. end], 0..) |r, k| {
                var drop = false;
                if (callId(r)) |rid| for (ids.items) |id| {
                    if (json.equal(rid, id)) drop = true;
                };
                if (by_place and std.mem.indexOfScalar(usize, bad.items, k) != null) drop = true;
                if (!drop) try keep.append(a, r);
            }
            try out.replaceRange(a, i + 1, end - i - 1, keep.items);
            end = i + 1;
            while (end < out.items.len and isTool(out.items[end])) end += 1;
        }
        if (fixed.items.len > 0) {
            if (bad.items.len > 0 or mended) {
                const copy = try json.copyObject(a, message.object);
                try copy.put(a, "tool_calls", .{ .array = fixed.items });
                out.items[i] = .{ .object = copy };
                changed = true;
            }
        } else {
            out.items[i] = try withoutKey(a, message.object, &.{"tool_calls"});
            changed = true;
            continue;
        }
        // results whose ids match none of the calls' ids: rendered in the order sent, their ids dropped
        var orphan = false;
        for (out.items[i + 1 .. end]) |r| {
            const rid = callId(r) orelse continue;
            var known = false;
            for (fixed.items) |c| if (sameId(callId(c), rid)) {
                known = true;
            };
            if (!known) orphan = true;
        }
        if (orphan) {
            for (out.items[i + 1 .. end]) |*r| r.* = try withoutKey(a, r.object, &.{ "tool_call_id", "id" });
            changed = true;
        }
    }
    return if (changed) .{ .array = try out.toOwnedSlice(a) } else messages;
}

/// `normalize_messages`' rule for GLM: an assistant message with `reasoning` (a string) and no `reasoning_content`
/// string gets it as `reasoning_content` (without it a tool step's reasoning is dropped from the resent history).
pub fn promoteReasoning(a: Allocator, messages: Value) Allocator.Error!Value {
    if (messages != .array) return messages;
    var out: ?[]Value = null;
    for (messages.array, 0..) |m, i| {
        if (m != .object or !std.mem.eql(u8, m.strField("role") orelse "", "assistant")) continue;
        if (m.strField("reasoning_content") != null) continue;
        const r = m.strField("reasoning") orelse continue;
        if (out == null) out = try a.dupe(Value, messages.array);
        const copy = try json.copyObject(a, m.object);
        try copy.put(a, "reasoning_content", .{ .string = r });
        out.?[i] = .{ .object = copy };
    }
    return if (out) |o| .{ .array = o } else messages;
}


fn mediaKind(part: Value) ?[]const u8 {
    if (part != .object) return null;
    const t = part.strField("type") orelse return null;
    if (std.mem.eql(u8, t, "image_url")) return "image";
    if (std.mem.eql(u8, t, "video_url")) return "video";
    return null;
}

fn hasMediaPart(m: Value) bool {
    const content = if (m == .object) m.get("content") else null;
    if (content == null or content.? != .array) return false;
    for (content.?.array) |p| if (mediaKind(p) != null) return true;
    return false;
}

/// `prompts.tool_media_parts` (patch 0056, issue #28) for a server without a vision frontend: a tool result's pictures
/// and clips render as the template's own reminder ("You are unable to process this image ...") instead of a 400, so an
/// agent session that keeps a screenshot in its history goes on. Messages whose pictures are all in tool results come
/// back with those parts as the reminder text (never fetched or decoded); a user's (or any other message's) pictures
/// leave the messages as they were, to be refused as before.
pub fn toolMedia(a: Allocator, messages: Value) Allocator.Error!Value {
    if (messages != .array) return messages;
    var any = false;
    for (messages.array) |m| if (hasMediaPart(m)) {
        if (!isTool(m)) return messages;
        any = true;
    };
    if (!any) return messages;
    const out = try a.alloc(Value, messages.array.len);
    for (messages.array, out) |m, *slot| {
        slot.* = m;
        if (!hasMediaPart(m)) continue;
        const parts = try a.alloc(Value, m.get("content").?.array.len);
        for (m.get("content").?.array, parts) |part, *p| {
            if (mediaKind(part)) |kind| {
                const text = try std.fmt.allocPrint(a, "<reminder>You are unable to process this {s} because you don't have multi-modal input ability. Try different methods.</reminder>", .{kind});
                const o = try json.newObject(a);
                try o.put(a, "type", .{ .string = "text" });
                try o.put(a, "text", .{ .string = text });
                p.* = .{ .object = o };
            } else p.* = part;
        }
        const copy = try json.copyObject(a, m.object);
        try copy.put(a, "content", .{ .array = parts });
        slot.* = .{ .object = copy };
    }
    return .{ .array = out };
}

test "a tool result's pictures become the template's reminder; a user's are left to be refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try parseArr(a,
        \\[{"role":"user","content":"x"},{"role":"tool","tool_call_id":"c","content":[{"type":"text","text":"shot: "},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}},{"type":"video_url","video_url":{"url":"http://x/y.mp4"}}]}]
    );
    const out = try toolMedia(a, m);
    const parts = out.array[1].get("content").?.array;
    try std.testing.expectEqual(@as(usize, 3), parts.len);
    try std.testing.expectEqualStrings("shot: ", parts[0].strField("text").?);
    try std.testing.expectEqualStrings("<reminder>You are unable to process this image because you don't have multi-modal input ability. Try different methods.</reminder>", parts[1].strField("text").?);
    try std.testing.expect(std.mem.indexOf(u8, parts[2].strField("text").?, "process this video") != null);
    const user = try parseArr(a,
        \\[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:x"}}]},{"role":"tool","content":[{"type":"image_url","image_url":{"url":"data:x"}}]}]
    );
    try std.testing.expect((try toolMedia(a, user)).array.ptr == user.array.ptr);
    const none = try parseArr(a, "[{\"role\":\"user\",\"content\":\"x\"}]");
    try std.testing.expect((try toolMedia(a, none)).array.ptr == none.array.ptr);
}

fn parseArr(a: Allocator, text: []const u8) !Value {
    return (try json.parse(a, text)).ok;
}

fn dump(a: Allocator, v: Value) ![]u8 {
    return json.stringify(a, v, .{ .ascii = false });
}

test "empty and null arguments are no arguments" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try parseArr(a,
        \\[{"role":"user","content":"x"},{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":""}},{"id":"c2","type":"function","function":{"name":"g","arguments":null}},{"id":"c3","type":"function","function":{"name":"h"}}]},{"role":"tool","tool_call_id":"c1","content":"1"},{"role":"tool","tool_call_id":"c2","content":"2"},{"role":"tool","tool_call_id":"c3","content":"3"}]
    );
    const out = try agentHistory(a, m);
    const calls = out.array[1].get("tool_calls").?.array;
    for (calls) |c| try std.testing.expectEqual(@as(usize, 0), c.get("function").?.get("arguments").?.object.count());
    try std.testing.expectEqual(@as(usize, 5), out.array.len);
}

test "a call with arguments that are not a JSON object leaves with its result, by id" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try parseArr(a,
        \\[{"role":"user","content":"x"},{"role":"assistant","content":"hmm","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"a\": "}},{"id":"c2","type":"function","function":{"name":"g","arguments":"{\"b\":1}"}},{"id":"c3","type":"function","function":{"name":"h","arguments":"[1,2]"}}]},{"role":"tool","tool_call_id":"c1","content":"r1"},{"role":"tool","tool_call_id":"c2","content":"r2"},{"role":"tool","tool_call_id":"c3","content":"r3"}]
    );
    const out = try agentHistory(a, m);
    try std.testing.expectEqual(@as(usize, 3), out.array.len); // c1's and c3's results left with their calls
    const calls = out.array[1].get("tool_calls").?.array;
    try std.testing.expectEqual(@as(usize, 1), calls.len);
    try std.testing.expectEqualStrings("c2", calls[0].strField("id").?);
    try std.testing.expectEqualStrings("hmm", out.array[1].strField("content").?);
    try std.testing.expectEqualStrings("r2", out.array[2].strField("content").?);
    try std.testing.expectEqualStrings("c2", out.array[2].strField("tool_call_id").?);
}

test "every call bad: the message loses its tool_calls and keeps its text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try parseArr(a,
        \\[{"role":"user","content":"x"},{"role":"assistant","content":"hmm","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{trunc"}}]},{"role":"tool","tool_call_id":"c1","content":"r1"},{"role":"user","content":"again"}]
    );
    const out = try agentHistory(a, m);
    try std.testing.expectEqual(@as(usize, 3), out.array.len);
    try std.testing.expect(!out.array[1].has("tool_calls"));
    try std.testing.expectEqualStrings("hmm", out.array[1].strField("content").?);
    try std.testing.expectEqualStrings("user", out.array[2].strField("role").?);
}

test "no ids: the bad call's result goes by its place" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try parseArr(a,
        \\[{"role":"user","content":"x"},{"role":"assistant","content":"","tool_calls":[{"type":"function","function":{"name":"f","arguments":"{\"a\":1}"}},{"type":"function","function":{"name":"g","arguments":"nope"}}]},{"role":"tool","content":"r1"},{"role":"tool","content":"r2"}]
    );
    const out = try agentHistory(a, m);
    try std.testing.expectEqual(@as(usize, 3), out.array.len);
    try std.testing.expectEqualStrings("r1", out.array[2].strField("content").?);
}

test "results whose ids match none of the calls lose their ids and keep the order sent" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try parseArr(a,
        \\[{"role":"user","content":"x"},{"role":"assistant","content":"","tool_calls":[{"id":"call_a","type":"function","function":{"name":"f","arguments":{}}},{"id":"call_b","type":"function","function":{"name":"g","arguments":{}}}]},{"role":"tool","tool_call_id":"zzz","content":"r1"},{"role":"tool","tool_call_id":"yyy","content":"r2"}]
    );
    const out = try agentHistory(a, m);
    try std.testing.expect(!out.array[2].has("tool_call_id"));
    try std.testing.expect(!out.array[3].has("tool_call_id"));
    // a history with matching ids is the very same value
    const ok = try parseArr(a,
        \\[{"role":"user","content":"x"},{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":{"a":1}}}]},{"role":"tool","tool_call_id":"c1","content":"r1"}]
    );
    const same = try agentHistory(a, ok);
    try std.testing.expect(same.array.ptr == ok.array.ptr);
}

test "reasoning becomes reasoning_content where the step has none" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try parseArr(a,
        \\[{"role":"assistant","content":"a","reasoning":"why"},{"role":"assistant","content":"b","reasoning":"no","reasoning_content":"kept"}]
    );
    const out = try promoteReasoning(a, m);
    try std.testing.expectEqualStrings("why", out.array[0].strField("reasoning_content").?);
    try std.testing.expectEqualStrings("kept", out.array[1].strField("reasoning_content").?);
    _ = try dump(a, out);
}
