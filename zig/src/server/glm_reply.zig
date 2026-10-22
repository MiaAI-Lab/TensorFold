//! The end of a GLM reply that offered tools, as the recipe's `App.run` settles it (patches 0036, 0053, 0055, 0075):
//! the reasoning and answer split (calls written inside the think block count when they end the reply), the calls the
//! streamer followed plus the ones the end parser reads, a call the token limit cut (never sent, nor its markup as
//! content), and the finish reason that follows.
const std = @import("std");
const json = @import("json");
const glm_calls = @import("glm_calls.zig");
const reply_text = @import("reply_text.zig");
const tool_parse = @import("tool_parse.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const Input = struct {
    /// The reply's text through its last token, stop strings applied, `closeCall`'s suffix appended.
    text: []const u8,
    tools: []const Value,
    thinking: bool,
    markers: reply_text.Markers,
    think: ?*glm_calls.ThinkSplit,
    streamer: ?*glm_calls.CallStreamer,
    /// One call at most (`parallel_tool_calls: false`).
    single: bool,
    /// The model's end token ended the reply (not a stop string, the token limit or `ignore_eos`).
    ended: bool,
};

pub const Result = struct {
    reasoning: []const u8,
    /// The answer with its call blocks hidden (what a stream shows).
    answer: []const u8,
    /// The reply's content: the end parser's residue, or the answer when a call was cut.
    content: []const u8,
    calls: ?[]Value,
    /// Streamed calls that became whole at the end, as deltas.
    call_deltas: []Value,
    /// Calls already sent as deltas (the first of `calls`).
    streamed: usize,
    cut: bool,
};

pub fn finalize(a: Allocator, in: Input) Allocator.Error!Result {
    const split: reply_text.Split = if (in.think) |t|
        try t.split(in.text, true)
    else if (in.thinking)
        try reply_text.splitThinking(a, in.text, true, in.markers)
    else
        .{ .reasoning = "", .answer = in.text };
    var hidden: std.ArrayList(u8) = .empty;
    const answer = try reply_text.hideInto(a, &hidden, split.answer, true);
    const parsed = try tool_parse.parse(a, split.answer, in.tools, if (in.single) 1 else null);
    // `parse_tool_calls` strips what is left of the reply, calls or none
    var content: []const u8 = reply_text.pyStrip(parsed.content);
    var calls = parsed.calls;
    var call_deltas: []Value = &.{};
    var streamed: usize = 0;
    if (in.streamer) |cs| if (in.think == null or in.think.?.stream_from != null) {
        // the calls streamed as sent, then any the streamer did not follow; calls held in the think block to the end
        // (never streamed) are the end parser's, sent whole
        const feed_text = if (in.think) |t| in.text[t.stream_from.?..] else in.text;
        call_deltas = try cs.feed(feed_text);
        streamed = cs.sent;
        const all = try std.mem.concat(a, Value, &.{ try cs.followed(streamed), try cs.skippedCalls() });
        calls = if (all.len > 0) all else null;
    };
    // a call the token limit cut (the reply ended inside it): never sent, nor its markup as content; one the model's end
    // token left open that would not parse, its markup as text (`closeCall` closed the others)
    var cut = if (in.streamer) |cs| cs.open and !in.ended else false;
    if (cut) {
        content = reply_text.pyStrip(answer);
    } else if (!in.ended and in.tools.len > 0) {
        if (std.mem.lastIndexOf(u8, content, glm_calls.call_open)) |at| {
            if (std.mem.indexOf(u8, content[at..], glm_calls.call_close) == null) {
                content = reply_text.pyRstrip(content[0..at]);
                cut = true;
            }
        }
    }
    return .{ .reasoning = split.reasoning, .answer = answer, .content = content, .calls = calls, .call_deltas = call_deltas, .streamed = streamed, .cut = cut };
}

/// `{"tool_calls": [{"index", "id", "type", "function": {"name", "arguments"}}]}` for a call the end parser read: one
/// delta with its header and its arguments.
pub fn wholeDelta(a: Allocator, index: usize, call: Value) Allocator.Error!Value {
    const function = try json.newObject(a);
    const f = call.get("function") orelse Value{ .object = try json.newObject(a) };
    try function.put(a, "name", .{ .string = f.strField("name") orelse "" });
    try function.put(a, "arguments", .{ .string = f.strField("arguments") orelse "" });
    const one = try json.newObject(a);
    try one.put(a, "index", try json.intValue(a, index));
    try one.put(a, "id", .{ .string = call.strField("id") orelse "" });
    try one.put(a, "type", .{ .string = "function" });
    try one.put(a, "function", .{ .object = function });
    const list = try a.alloc(Value, 1);
    list[0] = .{ .object = one };
    const d = try json.newObject(a);
    try d.put(a, "tool_calls", .{ .array = list });
    return .{ .object = d };
}
