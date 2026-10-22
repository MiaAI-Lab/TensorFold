//! The recipe's tool-call serving, end to end on CPU: a scripted engine plays rounds of GLM text, the server streams
//! or returns them (patches 0036, 0053, 0055, 0075): each call sent whole once written, calls in the think block,
//! a call the token limit cuts never sent, one the model's end token leaves open closed when it parses.
const std = @import("std");
const json = @import("json");
const rig = @import("glm_rig.zig");
const Value = json.Value;

const tools =
    \\"tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"},"mode":{"type":["string","null"]}}}}},{"type":"function","function":{"name":"ls","parameters":{"type":"object","properties":{}}}}]
;
const user = "\"messages\":[{\"role\":\"user\",\"content\":\"go\"}]," ++ tools;

/// The `data:` events of an SSE body, decoded; the caller's arena owns them.
fn events(a: std.mem.Allocator, body: []const u8) ![]Value {
    var out: std.ArrayList(Value) = .empty;
    var lines = std.mem.splitSequence(u8, body, "\n\n");
    while (lines.next()) |block| {
        if (!std.mem.startsWith(u8, block, "data: ")) continue;
        const payload = block["data: ".len..];
        if (std.mem.eql(u8, payload, "[DONE]")) continue;
        try out.append(a, (try json.parse(a, payload)).ok);
    }
    return out.items;
}

fn delta(event: Value) Value {
    return event.get("choices").?.array[0].get("delta") orelse .null;
}

fn finishOf(event: Value) ?[]const u8 {
    const f = event.get("choices").?.array[0].get("finish_reason") orelse return null;
    return if (f == .string) f.string else null;
}

/// Everything a stream says, folded: text, reasoning, calls (index -> name, arguments), the finish reason, empty deltas.
const Folded = struct {
    content: []const u8 = "",
    reasoning: []const u8 = "",
    names: [8][]const u8 = @splat(""),
    arguments: [8][]const u8 = @splat(""),
    call_deltas: usize = 0,
    count: usize = 0,
    empty: usize = 0,
    finish: []const u8 = "",
    role_first: bool = false,
};

fn fold(a: std.mem.Allocator, body: []const u8) !Folded {
    var f: Folded = .{};
    var content: std.ArrayList(u8) = .empty;
    var reasoning: std.ArrayList(u8) = .empty;
    for (try events(a, body), 0..) |e, n| {
        const choices = e.get("choices") orelse continue;
        if (choices.array.len == 0) continue;
        const d = delta(e);
        if (n == 0) f.role_first = d.strField("role") != null;
        if (d == .object and d.object.count() == 0) f.empty += 1;
        if (d.strField("content")) |c| try content.appendSlice(a, c);
        if (d.strField("reasoning_content")) |c| try reasoning.appendSlice(a, c);
        if (d.get("tool_calls")) |calls| for (calls.array) |c| {
            f.call_deltas += 1;
            const i: usize = @intCast(c.get("index").?.int64().?);
            f.count = @max(f.count, i + 1);
            if (c.get("function")) |fun| {
                if (fun.strField("name")) |name| f.names[i] = name;
                if (fun.strField("arguments")) |args| f.arguments[i] = try std.mem.concat(a, u8, &.{ f.arguments[i], args });
            }
        };
        if (finishOf(e)) |r| f.finish = r;
    }
    f.content = content.items;
    f.reasoning = reasoning.items;
    return f;
}

const call_a = "<tool_call>read_file<arg_key>path</arg_key><arg_value>/tmp/a</arg_value><arg_key>limit</arg_key><arg_value>20</arg_value></tool_call>";

fn serve(r: *rig.Rig, extra: []const u8, rounds: []const []const u8, reason: @import("engine_api").Reason) !struct { status: u16, body: []u8 } {
    r.backend.rounds = rounds;
    r.backend.reason = reason;
    const body = try std.mem.concat(std.testing.allocator, u8, &.{ "{", user, ",\"max_tokens\":4000", extra, "}" });
    defer std.testing.allocator.free(body);
    const reply = try r.chat(body);
    return .{ .status = reply.status, .body = reply.body };
}

test "a streamed call goes out whole once written, text first, then finish tool_calls" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var r: rig.Rig = .{};
    try r.start(.{}, rig.default_config);
    defer r.finish();
    // the call arrives over four rounds
    const reply = try serve(&r, ",\"stream\":true,\"reasoning_effort\":\"none\"", &.{ "Sure, ", "reading it.\n<tool_call>read_file<arg_key>path</arg_k", "ey><arg_value>/tmp/a</arg_value><arg_key>limit</arg_key><arg_val", "ue>20</arg_value></tool_call>\x03" }, .stop);
    defer std.testing.allocator.free(reply.body);
    try std.testing.expectEqual(@as(u16, 200), reply.status);
    const f = try fold(arena.allocator(), reply.body);
    try std.testing.expect(f.role_first);
    try std.testing.expectEqualStrings("Sure, reading it.", f.content);
    try std.testing.expectEqual(@as(usize, 1), f.count);
    try std.testing.expectEqual(@as(usize, 1), f.call_deltas); // header and arguments in one delta
    try std.testing.expectEqualStrings("read_file", f.names[0]);
    try std.testing.expectEqualStrings("{\"path\":\"/tmp/a\",\"limit\":20}", f.arguments[0]);
    try std.testing.expectEqualStrings("tool_calls", f.finish);
    // the markup never reaches the text
    try std.testing.expect(std.mem.indexOf(u8, reply.body, "<tool_call>") == null);
}

test "two calls, then a non-streamed reply returns the same calls beside the text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var r: rig.Rig = .{};
    try r.start(.{}, rig.default_config);
    defer r.finish();
    const rounds = [_][]const u8{ "Two reads.\n" ++ call_a, "\n<tool_call>ls</tool_call>\x03" };
    const streamed = try serve(&r, ",\"stream\":true,\"reasoning_effort\":\"none\"", &rounds, .stop);
    defer std.testing.allocator.free(streamed.body);
    const f = try fold(a, streamed.body);
    try std.testing.expectEqual(@as(usize, 2), f.count);
    try std.testing.expectEqualStrings("read_file", f.names[0]);
    try std.testing.expectEqualStrings("ls", f.names[1]);
    try std.testing.expectEqualStrings("{}", f.arguments[1]);
    try std.testing.expectEqualStrings("tool_calls", f.finish);

    const whole = try serve(&r, ",\"reasoning_effort\":\"none\"", &rounds, .stop);
    defer std.testing.allocator.free(whole.body);
    const doc = (try json.parse(a, whole.body)).ok;
    const choice = doc.get("choices").?.array[0];
    const message = choice.get("message").?;
    try std.testing.expectEqualStrings("tool_calls", choice.strField("finish_reason").?);
    try std.testing.expectEqualStrings("Two reads.", message.strField("content").?); // the text before the calls stays
    const calls = message.get("tool_calls").?.array;
    try std.testing.expectEqual(@as(usize, 2), calls.len);
    try std.testing.expectEqualStrings("{\"path\":\"/tmp/a\",\"limit\":20}", calls[0].get("function").?.strField("arguments").?);
    try std.testing.expect(std.mem.startsWith(u8, calls[0].strField("id").?, "call_"));
}

test "a call the token limit cuts is never sent, nor its markup as content (0053)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var r: rig.Rig = .{};
    try r.start(.{}, rig.default_config);
    defer r.finish();
    const rounds = [_][]const u8{ "Writing now.\n<tool_call>read_file<arg_key>path</arg_key><arg_value>/tmp/a</arg_value><arg_key>mode</arg_key><arg_value>some long te", "xt that never ends" };
    const reply = try serve(&r, ",\"stream\":true,\"reasoning_effort\":\"none\"", &rounds, .length);
    defer std.testing.allocator.free(reply.body);
    const f = try fold(arena.allocator(), reply.body);
    try std.testing.expectEqual(@as(usize, 0), f.count);
    try std.testing.expectEqualStrings("Writing now.", f.content);
    try std.testing.expectEqualStrings("length", f.finish);
    // and the same without streaming
    const whole = try serve(&r, ",\"reasoning_effort\":\"none\"", &rounds, .length);
    defer std.testing.allocator.free(whole.body);
    const doc = (try json.parse(arena.allocator(), whole.body)).ok;
    const choice = doc.get("choices").?.array[0];
    try std.testing.expectEqualStrings("length", choice.strField("finish_reason").?);
    try std.testing.expect(choice.get("message").?.get("tool_calls") == null);
    try std.testing.expectEqualStrings("Writing now.", choice.get("message").?.strField("content").?);
}

test "a call the model's end token leaves open is closed when it parses, else it is text (0036, issue #22)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var r: rig.Rig = .{};
    try r.start(.{}, rig.default_config);
    defer r.finish();
    // the end token comes before </tool_call>, a value still open: both closers are added
    const open = [_][]const u8{"<tool_call>read_file<arg_key>path</arg_key><arg_value>/tmp/a\x03"};
    const reply = try serve(&r, ",\"stream\":true,\"reasoning_effort\":\"none\"", &open, .stop);
    defer std.testing.allocator.free(reply.body);
    const f = try fold(arena.allocator(), reply.body);
    try std.testing.expectEqual(@as(usize, 1), f.count);
    try std.testing.expectEqualStrings("{\"path\":\"/tmp/a\"}", f.arguments[0]);
    try std.testing.expectEqualStrings("tool_calls", f.finish);
    // a call that would not parse whole stays the reply's text, finish stop
    const bad = [_][]const u8{"Hmm <tool_call>delete_all<arg_key>x</arg_key><arg_value>1\x03"};
    const reply2 = try serve(&r, ",\"reasoning_effort\":\"none\"", &bad, .stop);
    defer std.testing.allocator.free(reply2.body);
    const doc = (try json.parse(arena.allocator(), reply2.body)).ok;
    const choice = doc.get("choices").?.array[0];
    try std.testing.expectEqualStrings("stop", choice.strField("finish_reason").?);
    try std.testing.expect(choice.get("message").?.get("tool_calls") == null);
}

test "calls written inside the think block are the reply's calls when they end it (0055)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var r: rig.Rig = .{};
    try r.start(.{}, rig.default_config);
    defer r.finish();
    // thinking on: the reply ends inside the think block on a call
    const ends = [_][]const u8{ "I need the file.\n", call_a, "\n</think>\x03" };
    const reply = try serve(&r, ",\"stream\":true", &ends, .stop);
    defer std.testing.allocator.free(reply.body);
    const f = try fold(arena.allocator(), reply.body);
    try std.testing.expectEqual(@as(usize, 1), f.count);
    try std.testing.expectEqualStrings("tool_calls", f.finish);
    try std.testing.expectEqualStrings("I need the file.\n", f.reasoning);
    try std.testing.expect(std.mem.indexOf(u8, f.reasoning, "<tool_call>") == null);
    // text after it makes it reasoning after all
    const more = [_][]const u8{ "I need the file.\n", call_a, "\nOr maybe not.\n</think>\nAnswer.\x03" };
    const reply2 = try serve(&r, ",\"stream\":true", &more, .stop);
    defer std.testing.allocator.free(reply2.body);
    const f2 = try fold(arena.allocator(), reply2.body);
    try std.testing.expectEqual(@as(usize, 0), f2.count);
    try std.testing.expectEqualStrings("stop", f2.finish);
    try std.testing.expectEqualStrings("Answer.", f2.content);
    try std.testing.expect(std.mem.indexOf(u8, f2.reasoning, "Or maybe not.") != null);
    // a call after the think block is an ordinary call
    const after = [_][]const u8{ "Think.\n</think>\n", call_a, "\x03" };
    const reply3 = try serve(&r, ",\"stream\":true", &after, .stop);
    defer std.testing.allocator.free(reply3.body);
    const f3 = try fold(arena.allocator(), reply3.body);
    try std.testing.expectEqual(@as(usize, 1), f3.count);
    try std.testing.expectEqualStrings("Think.", std.mem.trim(u8, f3.reasoning, "\n"));
}

test "parallel_tool_calls false keeps one call and sends it from the end parser" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var r: rig.Rig = .{};
    try r.start(.{}, rig.default_config);
    defer r.finish();
    const rounds = [_][]const u8{ "Reads.\n" ++ call_a ++ "<tool_call>ls</tool_call>\x03" };
    const reply = try serve(&r, ",\"stream\":true,\"reasoning_effort\":\"none\",\"parallel_tool_calls\":false", &rounds, .stop);
    defer std.testing.allocator.free(reply.body);
    const f = try fold(arena.allocator(), reply.body);
    try std.testing.expectEqual(@as(usize, 1), f.count);
    try std.testing.expectEqualStrings("read_file", f.names[0]);
    try std.testing.expectEqualStrings("tool_calls", f.finish);
}

test "a missing arg_key is put back and typed values keep their types" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var r: rig.Rig = .{};
    try r.start(.{}, rig.default_config);
    defer r.finish();
    const rounds = [_][]const u8{"<tool_call>read_file path</arg_key><arg_value>/q</arg_value><arg_key>limit</arg_key><arg_value>7</arg_value><arg_key>mode</arg_key><arg_value>null</arg_value></tool_call>\x03"};
    const reply = try serve(&r, ",\"stream\":true,\"reasoning_effort\":\"none\"", &rounds, .stop);
    defer std.testing.allocator.free(reply.body);
    const f = try fold(arena.allocator(), reply.body);
    try std.testing.expectEqual(@as(usize, 1), f.count);
    try std.testing.expectEqualStrings("{\"path\":\"/q\",\"limit\":7,\"mode\":null}", f.arguments[0]);
}

var nap_ns: u64 = 0;

fn nap(n: usize) void {
    if (n == 1) std.Io.sleep(std.testing.io, .fromNanoseconds(@intCast(nap_ns)), .awake) catch {};
}

test "a held call sends an empty delta every two seconds so an idle client keeps waiting" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var r: rig.Rig = .{};
    try r.start(.{}, rig.default_config);
    defer r.finish();
    nap_ns = 2_200_000_000;
    r.backend.before_round = nap;
    // the call opens in round 0, nothing arrives for 2.2 s, then the rest
    const rounds = [_][]const u8{ "<tool_call>read_file<arg_key>path</arg_key><arg_value>/tmp/", "a</arg_value></tool_call>\x03", "" };
    const reply = try serve(&r, ",\"stream\":true,\"reasoning_effort\":\"none\"", &rounds, .stop);
    defer std.testing.allocator.free(reply.body);
    const f = try fold(arena.allocator(), reply.body);
    try std.testing.expectEqual(@as(usize, 1), f.count);
    // round 2 (after the nap) found the call whole, so it was a delta, not a keepalive; hold the call open instead
    try std.testing.expectEqualStrings("{\"path\":\"/tmp/a\"}", f.arguments[0]);
    r.backend.before_round = nap;
    const held = [_][]const u8{ "<tool_call>read_file<arg_key>path</arg_key><arg_value>/tmp/", "a", "b</arg_value></tool_call>\x03" };
    const reply2 = try serve(&r, ",\"stream\":true,\"reasoning_effort\":\"none\"", &held, .stop);
    defer std.testing.allocator.free(reply2.body);
    const f2 = try fold(arena.allocator(), reply2.body);
    try std.testing.expect(f2.empty >= 1); // the nap fell while the call was open
    try std.testing.expectEqualStrings("{\"path\":\"/tmp/ab\"}", f2.arguments[0]);
}

test "a tool-calling step's reasoning is put back into the next request when the client drops it (TF_GLM_KEEP_REASONING)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var r: rig.Rig = .{};
    try r.start(.{}, rig.default_config);
    defer r.finish();
    const rounds = [_][]const u8{ "Let me read the file first.\n</think>\n", call_a, "\x03" };
    const reply = try serve(&r, "", &rounds, .stop);
    defer std.testing.allocator.free(reply.body);
    const doc = (try json.parse(a, reply.body)).ok;
    const message = doc.get("choices").?.array[0].get("message").?;
    const id = message.get("tool_calls").?.array[0].strField("id").?;
    // the agent sends the step back without its reasoning
    const next = try std.fmt.allocPrint(a,
        \\{{"max_tokens":4,{s},"messages":[{{"role":"user","content":"go"}},{{"role":"assistant","content":"","tool_calls":[{{"id":"{s}","type":"function","function":{{"name":"read_file","arguments":"{{\"path\":\"/tmp/a\",\"limit\":20}}"}}}}]}},{{"role":"tool","tool_call_id":"{s}","content":"file text"}}]}}
    , .{ tools, id, id });
    r.backend.rounds = &.{"done"};
    const second = try r.chat(next);
    defer std.testing.allocator.free(second.body);
    const prompt = try r.promptText();
    defer std.testing.allocator.free(prompt);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "<|assistant|><think>Let me read the file first.\n</think>") != null);
}

fn contentEvents(a: std.mem.Allocator, body: []const u8) !usize {
    var n: usize = 0;
    for (try events(a, body)) |e| {
        const choices = e.get("choices") orelse continue;
        if (choices.array.len == 0) continue;
        if (delta(e).strField("content") != null) n += 1;
    }
    return n;
}

test "smooth streaming sends a round's tokens one event each, the text and its end unchanged (0061)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const rounds = [_][]const u8{ "hello wo", "rld\x03" };
    {
        var r: rig.Rig = .{};
        try r.start(.{}, rig.default_config);
        defer r.finish();
        const reply = try serve(&r, ",\"stream\":true,\"reasoning_effort\":\"none\"", &rounds, .stop);
        defer std.testing.allocator.free(reply.body);
        try std.testing.expectEqual(@as(usize, 2), try contentEvents(a, reply.body)); // a round, an event
    }
    {
        var r: rig.Rig = .{};
        try r.start(.{ .stream_smooth = true, .stream_smooth_ms = 40 }, rig.default_config);
        defer r.finish();
        const reply = try serve(&r, ",\"stream\":true,\"reasoning_effort\":\"none\"", &rounds, .stop);
        defer std.testing.allocator.free(reply.body);
        const f = try fold(a, reply.body);
        try std.testing.expectEqualStrings("hello world", f.content);
        // a token, an event (the space waits for the character after it)
        try std.testing.expect(try contentEvents(a, reply.body) >= 10);
        try std.testing.expectEqualStrings("stop", f.finish);
        try std.testing.expect(f.role_first);
        // a tool call keeps its place after the text before it
        const calls = [_][]const u8{ "Reading.\n", call_a, "\x03" };
        const with_call = try serve(&r, ",\"stream\":true,\"reasoning_effort\":\"none\"", &calls, .stop);
        defer std.testing.allocator.free(with_call.body);
        const g = try fold(a, with_call.body);
        try std.testing.expectEqualStrings("Reading.", g.content);
        try std.testing.expectEqual(@as(usize, 1), g.count);
        try std.testing.expectEqualStrings("tool_calls", g.finish);
    }
}
