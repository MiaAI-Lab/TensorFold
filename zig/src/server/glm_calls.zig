//! GLM's tool calls in a reply as the recipe serves them (patches 0036, 0053, 0055; Python `cuda/reply_text.py`):
//!
//! * `ThinkSplit`: calls the model wrote inside the think block are its calls when they end the reply (the rule of
//!   jayleaton/glm53-tensorfold-spark patch 0620, Apache-2.0, as the recipe adapted it); a held call stays out of the
//!   reasoning until the reply ends on it or text follows it.
//! * `CallStreamer`: each `<tool_call>NAME<arg_key>..</arg_key><arg_value>..</arg_value></tool_call>` is followed while
//!   it is written (its arguments' JSON built as the text arrives, a typed value when whole) and sent whole, header and
//!   arguments in one delta, once its `</tool_call>` arrives. A call the reply ends inside (its token limit) is never
//!   sent: a client could store or run its cut arguments.
//! * `closeCall`: what closes the call a reply ended inside on an end token (issue #22), when it then parses.
const std = @import("std");
const json = @import("json");
const ids = @import("ids.zig");
const keys = @import("glm_keys.zig");
const reply_text = @import("reply_text.zig");
const tool_params = @import("tool_params.zig");
const tool_parse = @import("tool_parse.zig");
const tool_specs = @import("tool_specs.zig");
const Value = json.Value;
const Allocator = std.mem.Allocator;

pub const call_open = "<tool_call>";
pub const call_close = "</tool_call>";
pub const think_close = "</think>";
/// Characters of a call written inside the think block held before it streams as a call.
pub const think_call_hold: usize = 1024;
/// An idle client (one that drops a reply sending nothing) gets an empty delta this often while a call is held.
pub const keepalive_s: f64 = 2.0;

const partialTag = reply_text.partialTag;
const strip = reply_text.pyStrip;
const lstrip = reply_text.pyLstrip;

fn findFrom(text: []const u8, needle: []const u8, from: usize) ?usize {
    if (from > text.len) return null;
    return std.mem.indexOfPos(u8, text, from, needle);
}

/// `close_glm_call(text, tools)`: what closes the GLM call a reply ended inside when it parses as a whole call to an
/// offered tool (the model wrote its end token before `</tool_call>`): `</tool_call>`, after `</arg_value>` when a
/// value is open; "" when the reply is not inside a call, or the closed call would not parse (its markup is then text).
pub fn closeCall(a: Allocator, text: []const u8, tools: []const Value) Allocator.Error![]const u8 {
    const at = std.mem.lastIndexOf(u8, text, call_open) orelse return "";
    const block = text[at + call_open.len ..];
    if (std.mem.indexOf(u8, block, call_close) != null or std.mem.indexOf(u8, block, think_close) != null) return "";
    const value_open_at = std.mem.lastIndexOf(u8, block, keys.value_open);
    const value_close_at = std.mem.lastIndexOf(u8, block, keys.value_close);
    const open_value = if (value_open_at) |o| (if (value_close_at) |c| o > c else true) else false;
    const suffix: []const u8 = if (open_value) keys.value_close ++ call_close else call_close;
    // the strict reading: one whole call, nothing but its markup, arguments a JSON object
    const closed = try std.mem.concat(a, u8, &.{ call_open, block, suffix });
    return if ((try tool_parse.parse(a, closed, tools, 1)).calls != null) suffix else "";
}

pub const RunKind = enum { calls, pending, reasoning };

/// `_call_run`: (where the run of complete `<tool_call>` blocks from `start` ends, what it is): `calls` when the reply
/// ended on it (nothing after it but whitespace and a closing `</think>`), `pending` while that may still come true,
/// `reasoning` once text follows it (more reasoning, or an answer after the think block).
fn callRun(text: []const u8, start: usize, finished: bool) struct { usize, RunKind } {
    var end = start;
    var pos = start;
    while (std.mem.startsWith(u8, text[pos..], call_open)) {
        const close_at = findFrom(text, call_close, pos + call_open.len) orelse break;
        end = close_at + call_close.len;
        pos = text.len - lstrip(text[end..]).len;
    }
    var after = lstrip(text[end..]);
    if (std.mem.startsWith(u8, after, think_close) and strip(after[think_close.len..]).len == 0) after = "";
    if (after.len == 0) return .{ end, if (finished and end > start) .calls else .pending };
    if (!finished and (std.mem.startsWith(u8, after, call_open) or std.mem.startsWith(u8, call_open, after) or std.mem.startsWith(u8, think_close, after)))
        return .{ end, .pending }; // a block, or the think block's close, being written
    return .{ end, .reasoning };
}

/// (reasoning, answer) of a thinking GLM reply with tools, streamed or whole: the think block ends at its `</think>`,
/// and calls the model wrote inside it are its calls only when they end the reply. With `hold`, a held call longer than
/// `hold` characters is taken as a call at once (`committed`) so a long file write streams while it is written; nothing
/// after it can make it reasoning again. `stream_from` is where the text a call streamer reads begins once known.
pub const ThinkSplit = struct {
    a: Allocator,
    hold: ?usize = null,
    floor: usize = 0, // calls before this are reasoning
    committed: ?usize = null,
    stream_from: ?usize = null,

    pub fn split(s: *ThinkSplit, text: []const u8, finished: bool) Allocator.Error!reply_text.Split {
        if (s.committed) |at| {
            const rest = text[at..];
            const answer = if (std.mem.indexOf(u8, rest, think_close)) |i|
                lstrip(try std.mem.concat(s.a, u8, &.{ rest[0..i], rest[i + think_close.len ..] }))
            else
                lstrip(rest);
            return .{ .reasoning = text[0..at], .answer = answer };
        }
        const close = std.mem.indexOf(u8, text, think_close);
        while (true) {
            const call = findFrom(text, call_open, s.floor);
            if (close) |c| if (call == null or c < call.?) {
                s.stream_from = c + think_close.len;
                return .{ .reasoning = text[0..c], .answer = std.mem.trimStart(u8, text[s.stream_from.?..], "\n") };
            };
            const at = call orelse {
                const held = if (finished) 0 else @max(partialTag(text, think_close), partialTag(text, call_open));
                return .{ .reasoning = text[0 .. text.len - held], .answer = "" };
            };
            const end, const kind = callRun(text, at, finished);
            switch (kind) {
                .calls => return .{ .reasoning = text[0..at], .answer = text[at..end] },
                .reasoning => {
                    s.floor = @max(end, at + call_open.len);
                    continue;
                },
                .pending => {
                    if (s.hold) |h| if (text.len - at > h and close == null) {
                        s.committed = at;
                        s.stream_from = at;
                        return s.split(text, finished);
                    };
                    return .{ .reasoning = text[0..at], .answer = "" };
                },
            }
        }
    }
};

/// One call the streamer follows: its header went out when its name was read.
pub const Followed = struct {
    id: []const u8,
    name: []const u8,
    arguments: std.ArrayList(u8) = .empty,
};

fn isWordByte(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or ch == '_' or ch >= 0x80;
}

/// `_GLM_NAME_RE.fullmatch(name)`: `[\w.:-]+`.
fn globName(name: []const u8) bool {
    if (name.len == 0) return false;
    for (name) |ch| if (!isWordByte(ch) and ch != '.' and ch != ':' and ch != '-') return false;
    return true;
}

pub const CallStreamer = struct {
    a: Allocator,
    tools: []const Value,
    known: std.StringHashMapUnmanaged([]const u8) = .empty, // lowercase name -> the offered spelling
    schemas: tool_params.Schemas,
    calls: std.ArrayList(Followed) = .empty,
    skipped: std.ArrayList([]const u8) = .empty, // whole blocks it did not follow, for the end parser
    pos: usize = 0,
    state: enum { outside, name, skip, arguments, key, value } = .outside,
    start: usize = 0,
    open: bool = false, // the last call's header went out, its </tool_call> has not
    schema: ?*const json.Object = null, // the current call's parameter schemas
    value_schema: ?Value = null, // the current typed value's schema, null for text
    maybe_null: bool = false, // a nullable text value that may still be `null`: held
    sent: usize = 0, // calls sent whole (the last one may still be open)

    pub fn init(a: Allocator, tools: []const Value) Allocator.Error!CallStreamer {
        var s: CallStreamer = .{ .a = a, .tools = tools, .schemas = try tool_params.schemas(a, tools) };
        for (tools) |tool| {
            const name = try tool_specs.toolName(a, tool);
            try s.known.put(a, try std.ascii.allocLowerString(a, name), name);
        }
        return s;
    }

    fn args(s: *CallStreamer, piece: []const u8) Allocator.Error!void {
        try s.calls.items[s.calls.items.len - 1].arguments.appendSlice(s.a, piece);
    }

    fn jsonText(s: *CallStreamer, v: Value) Allocator.Error![]const u8 {
        return json.stringify(s.a, v, .{ .ascii = false, .compact = true });
    }

    /// The reply's answer so far (text that only grows): the calls that became whole since the last feed, each as one
    /// delta carrying its header and its arguments.
    pub fn feed(s: *CallStreamer, text: []const u8) Allocator.Error![]Value {
        try s.follow(text);
        const whole = s.calls.items.len - @intFromBool(s.open);
        var out: std.ArrayList(Value) = .empty;
        var i = s.sent;
        while (i < whole) : (i += 1) try out.append(s.a, try delta(s.a, i, s.calls.items[i]));
        s.sent = whole;
        return out.items;
    }

    /// `{"tool_calls": [{"index", "id", "type", "function": {"name", "arguments"}}]}`.
    pub fn delta(a: Allocator, index: usize, c: Followed) Allocator.Error!Value {
        const function = try json.newObject(a);
        try function.put(a, "name", .{ .string = c.name });
        try function.put(a, "arguments", .{ .string = c.arguments.items });
        const one = try json.newObject(a);
        try one.put(a, "index", try json.intValue(a, index));
        try one.put(a, "id", .{ .string = c.id });
        try one.put(a, "type", .{ .string = "function" });
        try one.put(a, "function", .{ .object = function });
        const list = try a.alloc(Value, 1);
        list[0] = .{ .object = one };
        const d = try json.newObject(a);
        try d.put(a, "tool_calls", .{ .array = list });
        return .{ .object = d };
    }

    /// Whether the end parser reads a block beginning `name` as a GLM call to an offered tool.
    fn callable(s: *CallStreamer, name: []const u8) Allocator.Error!bool {
        if (!globName(name)) return false;
        if (s.known.get(try std.ascii.allocLowerString(s.a, name)) == null) return false;
        // a name that is JSON (`true`, `1`) is read as JSON
        return switch (try json.parseText(s.a, name)) {
            .ok => false,
            .err => true,
        };
    }

    fn follow(s: *CallStreamer, text: []const u8) Allocator.Error!void {
        while (true) {
            const rest = text[@min(s.pos, text.len)..];
            switch (s.state) {
                .outside => {
                    const at = std.mem.indexOf(u8, rest, call_open) orelse return;
                    s.start = s.pos + at;
                    s.pos = s.start + call_open.len;
                    s.state = .name;
                },
                .name => { // the end parser's name: the text before <arg_key> or the end
                    const k = std.mem.indexOf(u8, rest, keys.key_open);
                    const c = std.mem.indexOf(u8, rest, call_close);
                    const stop = if (k) |ki| (if (c) |ci| @min(ki, ci) else ki) else c orelse {
                        if (reply_text.charCount(rest) > 256) {
                            s.state = .skip;
                            continue;
                        }
                        return;
                    };
                    const name = strip(rest[0..stop]);
                    if (!try s.callable(name)) {
                        s.state = .skip;
                        continue;
                    }
                    const offered = s.known.get(try std.ascii.allocLowerString(s.a, name)).?;
                    try s.calls.append(s.a, .{ .id = try ids.make(s.a, "call_", 24), .name = offered });
                    s.open = true;
                    s.schema = s.schemas.get(try std.ascii.allocLowerString(s.a, offered));
                    s.pos += stop;
                    s.state = .arguments;
                },
                .skip => { // a block the end parser decides: to its end
                    const at = std.mem.indexOf(u8, rest, call_close) orelse return;
                    s.pos += at + call_close.len;
                    s.state = .outside;
                    try s.skipped.append(s.a, try s.a.dupe(u8, text[s.start..s.pos])); // the text lies in a buffer that moves on
                },
                .arguments => { // text between arguments is ignored, as the end parser does
                    const k = std.mem.indexOf(u8, rest, keys.key_open);
                    const c = std.mem.indexOf(u8, rest, call_close);
                    if (c != null and (k == null or c.? < k.?)) {
                        const last = &s.calls.items[s.calls.items.len - 1];
                        try s.args(if (last.arguments.items.len > 0) "}" else "{}");
                        s.pos += c.? + call_close.len;
                        s.state = .outside;
                        s.open = false;
                        try s.repaired(text[s.start..s.pos]);
                    } else if (k) |ki| {
                        s.pos += ki + keys.key_open.len;
                        s.state = .key;
                    } else return;
                },
                .key => {
                    const end = std.mem.indexOf(u8, rest, keys.key_close);
                    // `\s*<arg_value>` right after the key's closer
                    var opens_end: ?usize = null;
                    if (end) |e| {
                        var j = e + keys.key_close.len;
                        while (j < rest.len and reply_text.isSpaceByte(rest[j])) j += 1;
                        if (std.mem.startsWith(u8, rest[j..], keys.value_open)) opens_end = j + keys.value_open.len;
                    }
                    const opens = opens_end orelse {
                        const after: []const u8 = if (end) |e| rest[e + keys.key_close.len ..] else "";
                        const cut = std.mem.indexOf(u8, if (end == null) rest else after, call_close) != null;
                        if (!cut and (end == null or strip(after).len == 0 or std.mem.startsWith(u8, keys.value_open, lstrip(after)))) return; // still being written
                        s.state = .arguments; // not an argument the end parser reads the same way
                        continue;
                    };
                    const key = strip(rest[0..end.?]);
                    const schema: Value = if (s.schema) |m| (m.get(key) orelse .{ .object = try json.newObject(s.a) }) else .{ .object = try json.newObject(s.a) };
                    s.value_schema = if (tool_params.typedParameter(schema)) schema else null;
                    s.maybe_null = s.value_schema == null and tool_params.nullableText(schema);
                    const last = &s.calls.items[s.calls.items.len - 1];
                    const opened: []const u8 = if (last.arguments.items.len == 0) "{" else ",";
                    const quote: []const u8 = if (s.value_schema != null or s.maybe_null) "" else "\"";
                    try s.args(try std.mem.concat(s.a, u8, &.{ opened, try json.quote(s.a, key, .{ .ascii = false }), ":", quote }));
                    s.pos += opens;
                    s.state = .value;
                },
                .value => {
                    const end = std.mem.indexOf(u8, rest, keys.value_close);
                    const written = if (end) |e| rest[0..e] else rest[0 .. rest.len - partialTag(rest, keys.value_close)];
                    if (s.value_schema) |schema| { // typed: decoded whole, as the end parser does
                        const e = end orelse return;
                        try s.args(try s.jsonText(try tool_params.decode(s.a, rest[0..e], schema, false)));
                    } else if (s.maybe_null) { // text or null: held while it may still be `null`
                        if (end != null and std.mem.eql(u8, written, "null")) {
                            try s.args("null");
                        } else if (end == null and std.mem.startsWith(u8, "null", written)) {
                            return;
                        } else {
                            try s.args("\"");
                            s.maybe_null = false;
                            continue;
                        }
                    } else { // text: as it arrives, less a tail that may begin the closer
                        const quoted = try json.quote(s.a, written, .{ .ascii = false });
                        const piece = try std.mem.concat(s.a, u8, &.{ quoted[1 .. quoted.len - 1], if (end != null) "\"" else "" });
                        if (piece.len > 0) try s.args(piece);
                        if (end == null) {
                            s.pos += written.len;
                            return;
                        }
                    }
                    s.pos += end.? + keys.value_close.len;
                    s.state = .arguments;
                },
            }
        }
    }

    /// A whole call whose markup lost an `<arg_key>`: its arguments as the end parser reads it repaired (the streamer
    /// skipped the argument it could not see); it has not been sent yet.
    fn repaired(s: *CallStreamer, block: []const u8) Allocator.Error!void {
        if (std.mem.count(u8, block, keys.key_close) <= std.mem.count(u8, block, keys.key_open)) return;
        const parsed = (try tool_parse.parse(s.a, block, s.tools, null)).calls orelse return;
        if (parsed.len != 1) return;
        const function = parsed[0].get("function") orelse return;
        const last = &s.calls.items[s.calls.items.len - 1];
        if (!std.mem.eql(u8, function.strField("name") orelse "", last.name)) return;
        last.arguments.clearRetainingCapacity();
        try last.arguments.appendSlice(s.a, function.strField("arguments") orelse "");
    }

    /// The calls the end parser reads in the blocks this streamer skipped (JSON or Qwen markup): sent at the end.
    pub fn skippedCalls(s: *CallStreamer) Allocator.Error![]Value {
        var out: std.ArrayList(Value) = .empty;
        for (s.skipped.items) |block| {
            const parsed = (try tool_parse.parse(s.a, block, s.tools, null)).calls orelse continue;
            try out.appendSlice(s.a, parsed);
        }
        return out.items;
    }

    /// The calls followed, as OpenAI tool calls (the first `n`).
    pub fn followed(s: *CallStreamer, n: usize) Allocator.Error![]Value {
        const out = try s.a.alloc(Value, @min(n, s.calls.items.len));
        for (out, s.calls.items[0..out.len]) |*slot, c| {
            const function = try json.newObject(s.a);
            try function.put(s.a, "name", .{ .string = c.name });
            try function.put(s.a, "arguments", .{ .string = c.arguments.items });
            const call = try json.newObject(s.a);
            try call.put(s.a, "id", .{ .string = c.id });
            try call.put(s.a, "type", .{ .string = "function" });
            try call.put(s.a, "function", .{ .object = function });
            slot.* = .{ .object = call };
        }
        return out;
    }
};
