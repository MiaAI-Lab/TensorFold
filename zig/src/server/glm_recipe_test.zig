//! CPU tests of the recipe's prompt rules on the shipped GLM-5.3-Flash template: reasoning effort and its aliases
//! (patches 0023, 0057), TF_GLM_EFFORT_TAIL (0096), TF_GLM_CLEAR_THINKING (0060) and the history repairs (0051).
const std = @import("std");
const rig = @import("glm_rig.zig");
const json = @import("json");

const head = "[gMASK]<sop>";

fn promptFor(r: *rig.Rig, body: []const u8) ![]u8 {
    const reply = try r.chat(body);
    defer std.testing.allocator.free(reply.body);
    try std.testing.expectEqual(@as(u16, 200), reply.status);
    return r.promptText();
}

fn expectPrompt(r: *rig.Rig, body: []const u8, expected: []const u8) !void {
    const got = try promptFor(r, body);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings(expected, got);
}

test "reasoning effort: GLM's template says Max by default, High and Low as asked, aliases mapped (0023, 0057)" {
    var r: rig.Rig = .{};
    try r.start(.{}, rig.default_config);
    defer r.finish();
    const user = "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4";
    // no effort: the template's own default, Max
    try expectPrompt(&r, "{" ++ user ++ "}", head ++ "<|system|>Reasoning Effort: Max<|user|>Hi<|assistant|><think>");
    const cases = [_]struct { []const u8, []const u8 }{
        .{ "low", "Low" },     .{ "high", "High" }, .{ "max", "Max" },
        .{ "xhigh", "Max" }, // the template renders Max for xhigh
        .{ "medium", "High" }, // GLM lists high and leaves medium unnamed
        .{ "minimal", "Low" },
    };
    for (cases) |c| {
        const body = try std.mem.concat(std.testing.allocator, u8, &.{ "{" ++ user ++ ",\"reasoning_effort\":\"", c[0], "\"}" });
        defer std.testing.allocator.free(body);
        const expected = try std.mem.concat(std.testing.allocator, u8, &.{ head ++ "<|system|>Reasoning Effort: ", c[1], "<|user|>Hi<|assistant|><think>" });
        defer std.testing.allocator.free(expected);
        try expectPrompt(&r, body, expected);
        // and inside chat_template_kwargs, where vLLM's clients put it
        const kw = try std.mem.concat(std.testing.allocator, u8, &.{ "{" ++ user ++ ",\"chat_template_kwargs\":{\"reasoning_effort\":\"", c[0], "\"}}" });
        defer std.testing.allocator.free(kw);
        try expectPrompt(&r, kw, expected);
    }
    // none: thinking off, no effort line, an empty think block
    const off = head ++ "<|user|>Hi<|assistant|><think></think>";
    try expectPrompt(&r, "{" ++ user ++ ",\"reasoning_effort\":\"none\"}", off);
    try expectPrompt(&r, "{" ++ user ++ ",\"chat_template_kwargs\":{\"enable_thinking\":false}}", off);
    // chat_template_kwargs.thinking, as DeepSeek-V4 clients send it
    try expectPrompt(&r, "{" ++ user ++ ",\"chat_template_kwargs\":{\"thinking\":{\"type\":\"disabled\"}}}", off);
    try expectPrompt(&r, "{" ++ user ++ ",\"chat_template_kwargs\":{\"thinking\":{\"type\":\"enabled\"}}}", head ++ "<|system|>Reasoning Effort: Max<|user|>Hi<|assistant|><think>");
    // any other value leaves the server's default (thinking on)
    try expectPrompt(&r, "{" ++ user ++ ",\"chat_template_kwargs\":{\"thinking\":{\"type\":\"adaptive\"}}}", head ++ "<|system|>Reasoning Effort: Max<|user|>Hi<|assistant|><think>");
    // an effort the server does not know is a 400 with the recipe's list
    const bad = try r.chat("{" ++ user ++ ",\"reasoning_effort\":\"ultra\"}");
    defer std.testing.allocator.free(bad.body);
    try std.testing.expectEqual(@as(u16, 400), bad.status);
    try std.testing.expect(std.mem.indexOf(u8, bad.body, "reasoning_effort must be none, minimal, low, medium, high, xhigh or max") != null);
}

test "THINKING=0 serves without a think block unless a request asks for reasoning" {
    var r: rig.Rig = .{};
    var config = rig.default_config;
    config.enable_thinking = false;
    try r.start(.{}, config);
    defer r.finish();
    const user = "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4";
    try expectPrompt(&r, "{" ++ user ++ "}", head ++ "<|user|>Hi<|assistant|><think></think>");
    try expectPrompt(&r, "{" ++ user ++ ",\"reasoning_effort\":\"low\"}", head ++ "<|system|>Reasoning Effort: Low<|user|>Hi<|assistant|><think>");
}

test "TF_GLM_EFFORT_TAIL=1: the effort line goes before the generation prompt, and a switch leaves the conversation alone" {
    var r: rig.Rig = .{};
    try r.start(.{ .effort_tail = true }, rig.default_config);
    defer r.finish();
    const user = "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4";
    try expectPrompt(&r, "{" ++ user ++ "}", head ++ "<|user|>Hi<|system|>Reasoning Effort: Max<|assistant|><think>");
    try expectPrompt(&r, "{" ++ user ++ ",\"reasoning_effort\":\"low\"}", head ++ "<|user|>Hi<|system|>Reasoning Effort: Low<|assistant|><think>");
    // thinking off has no line, as before
    try expectPrompt(&r, "{" ++ user ++ ",\"reasoning_effort\":\"none\"}", head ++ "<|user|>Hi<|assistant|><think></think>");
    // the conversation before the tail renders the same for every effort and with thinking on or off
    const a = try promptFor(&r, "{" ++ user ++ ",\"reasoning_effort\":\"high\"}");
    defer std.testing.allocator.free(a);
    const b = try promptFor(&r, "{" ++ user ++ ",\"chat_template_kwargs\":{\"enable_thinking\":false}}");
    defer std.testing.allocator.free(b);
    const shared = head ++ "<|user|>Hi";
    try std.testing.expect(std.mem.startsWith(u8, a, shared) and std.mem.startsWith(u8, b, shared));
}

const history =
    \\"messages":[{"role":"user","content":"q1"},{"role":"assistant","content":"a1","reasoning_content":"R1"},{"role":"user","content":"q2"}],"max_tokens":4
;

test "earlier turns keep their reasoning by default; TF_GLM_CLEAR_THINKING=1 or the request drops it (0060)" {
    {
        var r: rig.Rig = .{};
        try r.start(.{}, rig.default_config);
        defer r.finish();
        const kept = try promptFor(&r, "{" ++ history ++ "}");
        defer std.testing.allocator.free(kept);
        try std.testing.expect(std.mem.indexOf(u8, kept, "<|assistant|><think>R1</think>a1<|user|>q2") != null);
        const cleared = try promptFor(&r, "{" ++ history ++ ",\"chat_template_kwargs\":{\"clear_thinking\":true}}");
        defer std.testing.allocator.free(cleared);
        try std.testing.expect(std.mem.indexOf(u8, cleared, "<|assistant|><think></think>a1<|user|>q2") != null);
    }
    {
        var r: rig.Rig = .{};
        try r.start(.{ .clear_thinking = true }, rig.default_config);
        defer r.finish();
        const cleared = try promptFor(&r, "{" ++ history ++ "}");
        defer std.testing.allocator.free(cleared);
        try std.testing.expect(std.mem.indexOf(u8, cleared, "<|assistant|><think></think>a1<|user|>q2") != null);
        // a request can ask for the reasoning back
        const kept = try promptFor(&r, "{" ++ history ++ ",\"chat_template_kwargs\":{\"clear_thinking\":false}}");
        defer std.testing.allocator.free(kept);
        try std.testing.expect(std.mem.indexOf(u8, kept, "<|assistant|><think>R1</think>a1<|user|>q2") != null);
    }
}

test "an agent's history is repaired, not refused (0051)" {
    var r: rig.Rig = .{};
    try r.start(.{}, rig.default_config);
    defer r.finish();
    // a call with truncated arguments leaves with its result; the other call and the conversation are kept
    const body =
        \\{"max_tokens":4,"messages":[{"role":"user","content":"go"},{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{\"a\": "}},{"id":"c2","type":"function","function":{"name":"g","arguments":"{\"b\":1}"}}]},{"role":"tool","tool_call_id":"c1","content":"RESULT-ONE"},{"role":"tool","tool_call_id":"c2","content":"RESULT-TWO"},{"role":"user","content":"next"}]}
    ;
    const text = try promptFor(&r, body);
    defer std.testing.allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "RESULT-ONE") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "RESULT-TWO") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "<tool_call>g<arg_key>b</arg_key><arg_value>1</arg_value></tool_call>") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "<tool_call>f") == null);
    // empty arguments are no arguments
    const empty =
        \\{"max_tokens":4,"messages":[{"role":"user","content":"go"},{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":""}}]},{"role":"tool","tool_call_id":"c1","content":"ok"},{"role":"user","content":"next"}]}
    ;
    const text2 = try promptFor(&r, empty);
    defer std.testing.allocator.free(text2);
    try std.testing.expect(std.mem.indexOf(u8, text2, "<tool_call>f</tool_call>") != null);
}

test "MAX_TOKENS: a request without max_tokens gets the default cut to the window, one that asks too much is refused (0044)" {
    var r: rig.Rig = .{};
    r.backend.context_window = 400;
    try r.start(.{}, rig.default_config); // default_config: --max-tokens 32768, as MAX_TOKENS in scripts/config.sh
    defer r.finish();
    const user = "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]";
    const ok = try r.chat("{" ++ user ++ "}");
    defer std.testing.allocator.free(ok.body);
    try std.testing.expectEqual(@as(u16, 200), ok.status);
    const prompt: u32 = @intCast(r.backend.prompt.len);
    try std.testing.expectEqual(@as(u32, 400) - prompt, r.backend.max_tokens); // 32768 cut to what the window has left
    // max_completion_tokens counts as max_tokens, and what fits goes through as asked
    const fits = try r.chat("{" ++ user ++ ",\"max_completion_tokens\":100}");
    defer std.testing.allocator.free(fits.body);
    try std.testing.expectEqual(@as(u32, 100), r.backend.max_tokens);
    // one that asks too much is refused with OpenAI's code, in the recipe's words
    const big = try r.chat("{" ++ user ++ ",\"max_tokens\":5000}");
    defer std.testing.allocator.free(big.body);
    try std.testing.expectEqual(@as(u16, 400), big.status);
    const expect_detail = try std.fmt.allocPrint(std.testing.allocator, "this request needs a {d}-token context ({d} prompt tokens plus max_tokens 5000), which exceeds the context window this server was started for; shorten the prompt or reply", .{ prompt + 5000, prompt });
    defer std.testing.allocator.free(expect_detail);
    try std.testing.expect(std.mem.indexOf(u8, big.body, expect_detail) != null);
    try std.testing.expect(std.mem.indexOf(u8, big.body, "\"code\": \"context_length_exceeded\"") != null or std.mem.indexOf(u8, big.body, "\"code\":\"context_length_exceeded\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, big.body, "This server's maximum context length is 400 tokens") != null);
    // a streamed request is refused the same way, before a stream opens
    const streamed = try r.chat("{" ++ user ++ ",\"max_tokens\":5000,\"stream\":true}");
    defer std.testing.allocator.free(streamed.body);
    try std.testing.expectEqual(@as(u16, 400), streamed.status);
    // a prompt that fills the window leaves no room even for one token
    var long: std.ArrayList(u8) = .empty;
    defer long.deinit(std.testing.allocator);
    try long.appendSlice(std.testing.allocator, "{\"messages\":[{\"role\":\"user\",\"content\":\"");
    try long.appendNTimes(std.testing.allocator, 'x', 500);
    try long.appendSlice(std.testing.allocator, "\"}]}");
    const full = try r.chat(long.items);
    defer std.testing.allocator.free(full.body);
    try std.testing.expectEqual(@as(u16, 400), full.status);
    try std.testing.expect(std.mem.indexOf(u8, full.body, "-token prompt), which exceeds the context window") != null);
}

// -- admission (0081, 0082, 0095) -------------------------------------------------------------------------------

const helper = @import("late_system_test.zig");
var second: ?[]u8 = null; // the raw reply to the request that arrived while the lane was taken
var second_body: []const u8 = "";
var second_rig: ?*rig.Rig = null;
var second_path: []const u8 = "/v1/chat/completions";

var fired = false;

fn arriveWhileBusy(n: usize) void {
    if (n != 0 or fired) return;
    fired = true;
    const r = second_rig.?;
    second = helper.exchangeRaw(r.srv, "POST", second_path, second_body) catch null;
}

test "TF_GLM_MAX_QUEUED=0: a request that arrives while the lane is taken is refused with 429 and Retry-After (0082)" {
    var r: rig.Rig = .{};
    r.backend.lanes = 1;
    try r.start(.{ .max_queued = 0 }, rig.default_config);
    defer r.finish();
    second_rig = &r;
    fired = false;
    second_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4}";
    r.backend.before_round = arriveWhileBusy;
    const first = try r.chat(second_body);
    defer std.testing.allocator.free(first.body);
    try std.testing.expectEqual(@as(u16, 200), first.status);
    const raw = second.?;
    second = null;
    defer std.testing.allocator.free(raw);
    try std.testing.expect(std.mem.startsWith(u8, raw, "HTTP/1.1 429 "));
    try std.testing.expect(std.mem.indexOf(u8, raw, "Retry-After: 5\r\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, raw, "all serving lanes are busy; retry shortly") != null);
    try std.testing.expectEqual(@as(u32, 1), r.backend.submits); // the refused request never reached the engine
    // once the lane is free again the next request is served
    r.backend.before_round = null;
    const again = try r.chat(second_body);
    defer std.testing.allocator.free(again.body);
    try std.testing.expectEqual(@as(u16, 200), again.status);
    try std.testing.expectEqual(@as(u32, 0), r.srv.in_system.load(.acquire));
}

test "a streamed request to a full server is refused before its 200 opens (0095)" {
    var r: rig.Rig = .{};
    r.backend.lanes = 1;
    try r.start(.{ .max_queued = 0 }, rig.default_config);
    defer r.finish();
    second_rig = &r;
    fired = false;
    second_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4,\"stream\":true}";
    r.backend.before_round = arriveWhileBusy;
    const first = try r.chat("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4}");
    defer std.testing.allocator.free(first.body);
    const raw = second.?;
    second = null;
    defer std.testing.allocator.free(raw);
    try std.testing.expect(std.mem.startsWith(u8, raw, "HTTP/1.1 429 "));
    try std.testing.expect(std.mem.indexOf(u8, raw, "text/event-stream") == null);
    try std.testing.expect(std.mem.indexOf(u8, raw, "Retry-After: 5\r\n") != null);
}

test "with room in the queue, or no cap, or a background request, nothing is refused" {
    {
        var r: rig.Rig = .{};
        r.backend.lanes = 1;
        try r.start(.{ .max_queued = 1 }, rig.default_config); // one may wait past the lane
        defer r.finish();
        second_rig = &r;
        fired = false;
        second_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4}";
        r.backend.before_round = arriveWhileBusy;
        const first = try r.chat(second_body);
        defer std.testing.allocator.free(first.body);
        const raw = second.?;
        second = null;
        defer std.testing.allocator.free(raw);
        try std.testing.expect(std.mem.startsWith(u8, raw, "HTTP/1.1 200 "));
    }
    {
        var r: rig.Rig = .{};
        r.backend.lanes = 1;
        try r.start(.{}, rig.default_config); // MAX_QUEUED unset: queue as ever
        defer r.finish();
        second_rig = &r;
        fired = false;
        second_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4}";
        r.backend.before_round = arriveWhileBusy;
        const first = try r.chat(second_body);
        defer std.testing.allocator.free(first.body);
        const raw = second.?;
        second = null;
        defer std.testing.allocator.free(raw);
        try std.testing.expect(std.mem.startsWith(u8, raw, "HTTP/1.1 200 "));
    }
    {
        var r: rig.Rig = .{};
        r.backend.lanes = 1;
        try r.start(.{ .max_queued = 0 }, rig.default_config);
        defer r.finish();
        second_rig = &r;
        fired = false;
        second_body = "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4,\"priority\":\"background\"}";
        r.backend.before_round = arriveWhileBusy;
        const first = try r.chat("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4}");
        defer std.testing.allocator.free(first.body);
        const raw = second.?;
        second = null;
        defer std.testing.allocator.free(raw);
        try std.testing.expect(std.mem.startsWith(u8, raw, "HTTP/1.1 200 "));
    }
}

test "a tool result's picture renders as the template's reminder, a user's is refused (0056, issue #28)" {
    var r: rig.Rig = .{};
    try r.start(.{}, rig.default_config);
    defer r.finish();
    const body =
        \\{"max_tokens":4,"messages":[{"role":"user","content":"look"},{"role":"assistant","content":"","tool_calls":[{"id":"c1","type":"function","function":{"name":"screenshot","arguments":{}}}]},{"role":"tool","tool_call_id":"c1","content":[{"type":"text","text":"taken: "},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]},{"role":"user","content":"and?"}]}
    ;
    const text = try promptFor(&r, body);
    defer std.testing.allocator.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "<tool_response>taken: <reminder>You are unable to process this image because you don't have multi-modal input ability. Try different methods.</reminder></tool_response>") != null);
    // a user's picture is refused as before
    const refused = try r.chat("{\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AAAA\"}}]}]}");
    defer std.testing.allocator.free(refused.body);
    try std.testing.expectEqual(@as(u16, 400), refused.status);
}

test "TF_GLM_LOOP_GUARD=1 asks the engine for the recipe's rule while a think block is open, and counts a close (0091)" {
    const user = "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4";
    {
        var r: rig.Rig = .{};
        try r.start(.{ .loop_guard = true }, rig.default_config);
        defer r.finish();
        r.backend.loop_period = 3; // the engine closed a collapsed think block
        r.backend.rounds = &.{"ok"};
        const reply = try r.chat("{" ++ user ++ "}");
        defer std.testing.allocator.free(reply.body);
        try std.testing.expect(r.backend.loop_guard and r.backend.loop_recipe);
        try std.testing.expect(std.mem.indexOf(u8, reply.body, "\"loop_guard\": 1") != null);
        // thinking off has no think block to watch
        const off = try r.chat("{" ++ user ++ ",\"reasoning_effort\":\"none\"}");
        defer std.testing.allocator.free(off.body);
        try std.testing.expect(!r.backend.loop_guard);
    }
    {
        var r: rig.Rig = .{};
        try r.start(.{}, rig.default_config); // off by default: nothing changes
        defer r.finish();
        const reply = try r.chat("{" ++ user ++ "}");
        defer std.testing.allocator.free(reply.body);
        try std.testing.expect(!r.backend.loop_guard and !r.backend.loop_recipe);
        try std.testing.expect(std.mem.indexOf(u8, reply.body, "loop_guard") == null);
    }
}

test "/health answers the recipe's fields beside the native ones (cuda/health.py)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var r: rig.Rig = .{};
    r.backend.lanes = 4;
    r.backend.context_window = 5000;
    try r.start(.{}, rig.default_config);
    defer r.finish();
    const before = try helper.exchangeMethod(r.srv, "GET", "/health", "");
    defer std.testing.allocator.free(before.body);
    const h0 = (try json.parse(a, before.body)).ok;
    try std.testing.expect(h0.get("ok").?.bool);
    try std.testing.expectEqualStrings("tensorfold", h0.strField("backend").?);
    try std.testing.expect(!h0.get("busy").?.bool);
    try std.testing.expectEqual(@as(i64, 0), h0.get("requests_running").?.int64().?);
    try std.testing.expectEqual(@as(i64, 5000), h0.get("context_length").?.int64().?);
    const streams = h0.get("streams").?;
    try std.testing.expectEqual(@as(i64, 4), streams.get("max").?.int64().?);
    try std.testing.expect(streams.has("decoding") and streams.has("prefilling") and streams.has("filling") and streams.has("paused"));
    // the totals a bench reads before and after a run
    r.backend.rounds = &.{"hello"};
    const reply = try r.chat("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":8}");
    defer std.testing.allocator.free(reply.body);
    const after = try helper.exchangeMethod(r.srv, "GET", "/health", "");
    defer std.testing.allocator.free(after.body);
    const h1 = (try json.parse(a, after.body)).ok;
    for ([_][]const u8{ "requests_total", "prompt_tokens_total", "completion_tokens_total", "prefill_seconds_total", "decode_seconds_total", "cached_tokens_total", "rounds_total", "drafted_total", "accepted_total" }) |key|
        try std.testing.expect(h1.has(key));
    try std.testing.expectEqual(@as(i64, 1), h1.get("requests_total").?.int64().?);
    try std.testing.expectEqual(@as(i64, 5), h1.get("completion_tokens_total").?.int64().?);
    try std.testing.expectEqual(@as(i64, @intCast(r.backend.prompt.len)), h1.get("prompt_tokens_total").?.int64().?);
    // the native fields stay
    try std.testing.expectEqualStrings("ok", h1.strField("status").?);
}

test "stop sequences cut the reply at the token that completes one, and name it (cuda/server.py StopStrings)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var r: rig.Rig = .{};
    try r.start(.{}, rig.default_config);
    defer r.finish();
    r.backend.rounds = &.{ "abcE", "NDdef" };
    const reply = try r.chat("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":50,\"reasoning_effort\":\"none\",\"stop\":[\"END\",\"zzz\"]}");
    defer std.testing.allocator.free(reply.body);
    const doc = (try json.parse(a, reply.body)).ok;
    const choice = doc.get("choices").?.array[0];
    try std.testing.expectEqualStrings("abc", choice.get("message").?.strField("content").?);
    try std.testing.expectEqualStrings("stop", choice.strField("finish_reason").?);
    try std.testing.expectEqualStrings("END", doc.strField("stop_sequence").?);
    // the usage counts the tokens through the one that completed the string
    try std.testing.expectEqual(@as(i64, 6), doc.get("usage").?.get("completion_tokens").?.int64().?);
    // a stream never shows the start of a stop string it then cut
    const streamed = try r.chat("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":50,\"reasoning_effort\":\"none\",\"stream\":true,\"stop\":\"END\"}");
    defer std.testing.allocator.free(streamed.body);
    try std.testing.expect(std.mem.indexOf(u8, streamed.body, "\"content\": \"abc\"") != null or std.mem.indexOf(u8, streamed.body, "abc") != null);
    try std.testing.expect(std.mem.indexOf(u8, streamed.body, "E\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, streamed.body, "\"finish_reason\": \"stop\"") != null);
    // a malformed stop is a 400
    const bad = try r.chat("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"stop\":[1]}");
    defer std.testing.allocator.free(bad.body);
    try std.testing.expectEqual(@as(u16, 400), bad.status);
}

test "n and logprobs are refused, penalties and logit_bias are accepted and ignored (README API notes)" {
    var r: rig.Rig = .{};
    try r.start(.{}, rig.default_config);
    defer r.finish();
    const user = "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4";
    const cases = [_]struct { []const u8, u16, []const u8 }{
        .{ ",\"n\":2", 400, "n must be 1; multiple choices are not supported" },
        .{ ",\"n\":1", 200, "" },
        .{ ",\"logprobs\":true", 400, "logprobs are not supported by this model or backend" },
        .{ ",\"logprobs\":true,\"top_logprobs\":3", 400, "logprobs are not supported by this model or backend" },
        .{ ",\"top_logprobs\":3", 400, "top_logprobs requires logprobs: true" },
        .{ ",\"logprobs\":false", 200, "" },
        .{ ",\"logit_bias\":{\"5\":-100},\"presence_penalty\":1,\"frequency_penalty\":1,\"repetition_penalty\":1.2", 200, "" },
    };
    for (cases) |c| {
        const body = try std.mem.concat(std.testing.allocator, u8, &.{ "{" ++ user, c[0], "}" });
        defer std.testing.allocator.free(body);
        const reply = try r.chat(body);
        defer std.testing.allocator.free(reply.body);
        errdefer std.debug.print("case {s}: {s}\n", .{ c[0], reply.body });
        try std.testing.expectEqual(c[1], reply.status);
        if (c[2].len > 0) try std.testing.expect(std.mem.indexOf(u8, reply.body, c[2]) != null);
    }
}

test "return_token_ids puts the reply's ids in the tensorfold block (tools/end_of_turn.py)" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var r: rig.Rig = .{};
    try r.start(.{}, rig.default_config);
    defer r.finish();
    r.backend.rounds = &.{"ab\x03"};
    const reply = try r.chat("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":8,\"return_token_ids\":true,\"reasoning_effort\":\"none\"}");
    defer std.testing.allocator.free(reply.body);
    const doc = (try json.parse(a, reply.body)).ok;
    const ids = doc.get("tensorfold").?.get("token_ids").?.array;
    try std.testing.expectEqual(@as(usize, 3), ids.len); // the end token too, as `out` in the Python server
    try std.testing.expectEqual(@as(i64, 97), ids[0].int64().?);
    try std.testing.expectEqual(@as(i64, 3), ids[2].int64().?);
    const plain = try r.chat("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":8}");
    defer std.testing.allocator.free(plain.body);
    try std.testing.expect(std.mem.indexOf(u8, plain.body, "token_ids") == null);
}

test "/metrics carries /health's figures as tensorfold_health: families (0045)" {
    var r: rig.Rig = .{};
    r.backend.lanes = 2;
    r.backend.context_window = 3000;
    try r.start(.{}, rig.default_config);
    defer r.finish();
    const reply = try helper.exchangeMethod(r.srv, "GET", "/metrics", "");
    defer std.testing.allocator.free(reply.body);
    try std.testing.expectEqual(@as(u16, 200), reply.status);
    for ([_][]const u8{ "tensorfold_health:requests_total 0", "# TYPE tensorfold_health:completion_tokens_total counter", "tensorfold_health:streams{state=\"decoding\"} 0", "tensorfold_health:context_length 3000", "tensorfold_health:streams_max 2", "tensorfold:requests_running" }) |want|
        try std.testing.expect(std.mem.indexOf(u8, reply.body, want) != null);
}

test "Anthropic Messages answers a full server with 429 and an overloaded_error body (cuda/http.py -> responses._send)" {
    var r: rig.Rig = .{};
    r.backend.lanes = 1;
    try r.start(.{ .max_queued = 0 }, rig.default_config);
    defer r.finish();
    second_rig = &r;
    fired = false;
    second_path = "/v1/messages";
    defer second_path = "/v1/chat/completions";
    second_body = "{\"model\":\"m\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}";
    r.backend.before_round = arriveWhileBusy;
    const first = try r.chat("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":4}");
    defer std.testing.allocator.free(first.body);
    const raw = second.?;
    second = null;
    defer std.testing.allocator.free(raw);
    try std.testing.expect(std.mem.startsWith(u8, raw, "HTTP/1.1 429 "));
    try std.testing.expect(std.mem.indexOf(u8, raw, "\"overloaded_error\"") != null);
}

fn grammarLibrary() bool {
    var lib = std.DynLib.open("libtfgrammar.so") catch return false;
    lib.close();
    return true;
}

test "response_format reaches the engine as a grammar compiled over the checkpoint's tokenizer; thinking starts it after </think>" {
    // needs libtfgrammar (zig build grammar -Dxgrammar=DIR, on LD_LIBRARY_PATH) and a GLM checkpoint folder in TF_TEST_GLM_DIR
    const dir = std.c.getenv("TF_TEST_GLM_DIR") orelse return error.SkipZigTest;
    if (!grammarLibrary()) return error.SkipZigTest;
    var r: rig.Rig = .{};
    r.backend.structures = true;
    var config = rig.default_config;
    config.model_dir = std.mem.span(dir);
    try r.start(.{}, config);
    defer r.finish();
    const schema = "{\"type\":\"object\",\"required\":[\"name\"],\"properties\":{\"name\":{\"type\":\"string\"}}}";
    const user = "\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":8,\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"p\",\"schema\":" ++ schema ++ "}}";
    const off = try r.chat("{" ++ user ++ ",\"reasoning_effort\":\"none\"}");
    defer std.testing.allocator.free(off.body);
    try std.testing.expectEqual(@as(u16, 200), off.status);
    const seen = r.backend.structure.?;
    try std.testing.expect(seen.kind == .json_schema);
    // the schema as Python's json.dumps writes it (grammar.py _schema_text), which is what xgrammar compiles
    try std.testing.expectEqualStrings("{\"type\": \"object\", \"required\": [\"name\"], \"properties\": {\"name\": {\"type\": \"string\"}}}", seen.text);
    try std.testing.expectEqual(@as(?u32, null), seen.after);
    const on = try r.chat("{" ++ user ++ "}");
    defer std.testing.allocator.free(on.body);
    try std.testing.expectEqual(@as(?u32, 1003), r.backend.structure.?.after); // </think> in the rig's tokenizer
    const object = try r.chat("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":8,\"reasoning_effort\":\"none\",\"response_format\":{\"type\":\"json_object\"}}");
    defer std.testing.allocator.free(object.body);
    try std.testing.expect(r.backend.structure.?.kind == .json);
    // a schema xgrammar cannot enforce is the request's 400, in the Python server's words
    const bad = try r.chat("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"p\",\"schema\":{\"type\":\"object\",\"properties\":{\"a\":{\"$ref\":\"#/nope\"}}}}}}");
    defer std.testing.allocator.free(bad.body);
    try std.testing.expectEqual(@as(u16, 400), bad.status);
    try std.testing.expect(std.mem.indexOf(u8, bad.body, "the grammar cannot be enforced") != null);
}

test "an engine that cannot enforce grammars refuses response_format by name" {
    var r: rig.Rig = .{};
    try r.start(.{}, rig.default_config);
    defer r.finish();
    const reply = try r.chat("{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"response_format\":{\"type\":\"json_object\"}}");
    defer std.testing.allocator.free(reply.body);
    try std.testing.expectEqual(@as(u16, 400), reply.status);
    try std.testing.expect(std.mem.indexOf(u8, reply.body, "structured output") != null);
}
