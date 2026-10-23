//! The prepare gate's reservation measured on GLM (prepare_gate.default_per_byte): a long agent conversation through
//! prompt.prepare with the prompt cache on, as rank 0 prepares it. GLM renders and tokenizes it three times there (the
//! prompt, its history and the rewind probe), each render after glm_history's repairs and the normalized copy; the
//! system prefix probe renders a short fourth. Counted: the request's body and parsed messages, what its arena grows
//! by in prepare, and the render's and the tokenizer's scratch at their largest (one of them runs at a time).
const std = @import("std");
const tokenizer = @import("tokenizer");
const template = @import("template");
const json = @import("json");
const server = @import("root.zig");
const prompt = @import("prompt.zig");
const chat = @import("chat.zig");
const errors = @import("errors.zig");
const messages_mod = @import("messages.zig");
const glm_history = @import("glm_history.zig");
const hf_text = @import("hf_text.zig");
const prepare_gate = @import("prepare_gate.zig");
const rig = @import("glm_rig.zig");
const Allocator = std.mem.Allocator;
const Value = json.Value;

const source = @embedFile("fixtures_glm_chat.jinja");

/// Counts what an allocator holds now and at most.
const Counting = struct {
    child: Allocator,
    live: usize = 0,
    peak: usize = 0,

    fn allocator(c: *Counting) Allocator {
        return .{ .ptr = c, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn note(c: *Counting, add: usize, sub: usize) void {
        c.live = c.live + add - sub;
        c.peak = @max(c.peak, c.live);
    }
    fn alloc(ctx: *anyopaque, len: usize, al: std.mem.Alignment, ra: usize) ?[*]u8 {
        const c: *Counting = @ptrCast(@alignCast(ctx));
        const p = c.child.rawAlloc(len, al, ra) orelse return null;
        c.note(len, 0);
        return p;
    }
    fn resize(ctx: *anyopaque, m: []u8, al: std.mem.Alignment, n: usize, ra: usize) bool {
        const c: *Counting = @ptrCast(@alignCast(ctx));
        if (!c.child.rawResize(m, al, n, ra)) return false;
        c.note(n, m.len);
        return true;
    }
    fn remap(ctx: *anyopaque, m: []u8, al: std.mem.Alignment, n: usize, ra: usize) ?[*]u8 {
        const c: *Counting = @ptrCast(@alignCast(ctx));
        const p = c.child.rawRemap(m, al, n, ra) orelse return null;
        c.note(n, m.len);
        return p;
    }
    fn free(ctx: *anyopaque, m: []u8, al: std.mem.Alignment, ra: usize) void {
        const c: *Counting = @ptrCast(@alignCast(ctx));
        c.child.rawFree(m, al, ra);
        c.note(0, m.len);
    }
};

const specials = [_][]const u8{ "[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "<tool_response>", "</tool_response>" };

/// GPT-2's byte-to-character alphabet (the tokenizer's steps.byte_chars), for the vocabulary's byte symbols.
const byte_chars: [256]u21 = blk: {
    var table: [256]u21 = undefined;
    var next: u21 = 256;
    for (0..256) |b| {
        const printable = (b >= '!' and b <= '~') or (b >= 0xA1 and b <= 0xAC) or b >= 0xAE;
        table[b] = if (printable) b else next;
        if (!printable) next += 1;
    }
    break :blk table;
};

const words = [_][]const u8{ "the", "file", "function", "returns", "error", "test", "build", "memory", "prompt", "server", "rank", "token", "cache", "read", "write", "line", "value", "const", "var", "try", "if", "else", "for", "while", "struct", "pub", "fn", "self", "and", "of", "to", "in", "a", "is", "that", "it", "with", "this", "we", "should", "check", "change", "call", "path", "list", "count", "bytes", "page", "arena", "allocator", "There", "Each", "When", "return" };

/// Punctuation runs, indents and suffixes the text below splits into (GLM's pre-tokenizer keeps them whole).
const pieces = [_][]const u8{ ".", ".\n\n", ",", " (", ");\n", " {\n", "}\n", " =", " >", " *", " !", ";\n", ":", "`)", " `", "   ", "       ", "           ", ".zig", "!void" };

/// A byte-level BPE tokenizer with GLM's pre-tokenizer (the word regex, then ByteLevel), GLM's chat tokens as added
/// specials, every byte in the vocabulary and merges that join each of `words` (alone and after a space, an underscore,
/// a dot, a parenthesis, a backquote or a slash), `pieces` and the numbers below 1000: about 0.37 ids a text byte on
/// conversation()'s text (GLM's own tokenizer gives ~0.3 on English and code, so the per-id parts are if anything
/// overstated here).
fn glmLike(a: Allocator) !tokenizer.Tokenizer {
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    const w = &out.writer;
    try w.writeAll("{\"added_tokens\":[");
    for (specials, 0..) |s, i| try w.print("{s}{{\"id\":{d},\"content\":{f},\"special\":true,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"single_word\":false}}", .{ if (i > 0) "," else "", i, std.json.fmt(s, .{}) });
    try w.writeAll(
        \\],"normalizer":null,
        \\ "pre_tokenizer":{"type":"Sequence","pretokenizers":[
        \\  {"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},
        \\  {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]},
        \\ "decoder":{"type":"ByteLevel"},
        \\ "model":{"type":"BPE","dropout":null,"unk_token":null,"fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{
    );
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const aa = arena.allocator();
    var vocab: std.StringArrayHashMapUnmanaged(void) = .empty;
    var merges: std.ArrayList([2][]const u8) = .empty;
    for (0..256) |b| {
        var buf: [4]u8 = undefined;
        const n = try std.unicode.utf8Encode(byte_chars[b], &buf);
        try vocab.put(aa, try aa.dupe(u8, buf[0..n]), {});
    }
    const Bpe = struct {
        a: Allocator,
        vocab: *std.StringArrayHashMapUnmanaged(void),
        merges: *std.ArrayList([2][]const u8),

        /// The merge of `x` and `y` (in that rank order), and its token.
        fn join(b: @This(), x: []const u8, y: []const u8) ![]const u8 {
            const next = try std.mem.concat(b.a, u8, &.{ x, y });
            if (!b.vocab.contains(next)) {
                try b.vocab.put(b.a, next, {});
                try b.merges.append(b.a, .{ x, y });
            }
            return next;
        }

        /// `raw`'s bytes joined left to right; its token.
        fn chain(b: @This(), raw: []const u8) ![]const u8 {
            var cur: []const u8 = "";
            for (raw) |byte| {
                var buf: [4]u8 = undefined;
                const sym = try b.a.dupe(u8, buf[0..try std.unicode.utf8Encode(byte_chars[byte], &buf)]);
                cur = if (cur.len == 0) sym else try b.join(cur, sym);
            }
            return cur;
        }
    };
    const bpe: Bpe = .{ .a = aa, .vocab = &vocab, .merges = &merges };
    var whole: [words.len][]const u8 = undefined;
    for (words, &whole) |word, *t| t.* = try bpe.chain(word);
    // a word after its lead joins last, as a trained BPE ranks the longer pieces after their parts
    for ([_][]const u8{ " ", "_", ".", "(", "`", "/" }) |lead| for (whole) |t| {
        _ = try bpe.join(try bpe.chain(lead), t);
    };
    for (pieces) |piece| _ = try bpe.chain(piece);
    for (0..1000) |n| _ = try bpe.chain(try std.fmt.allocPrint(aa, "{d}", .{n}));
    for (vocab.keys(), specials.len..) |k, id| try w.print("{s}{f}:{d}", .{ if (id > specials.len) "," else "", std.json.fmt(k, .{}), id });
    try w.writeAll("},\"merges\":[");
    for (merges.items, 0..) |m, i| try w.print("{s}[{f},{f}]", .{ if (i > 0) "," else "", std.json.fmt(m[0], .{}), std.json.fmt(m[1], .{}) });
    try w.writeAll("]}}");
    return tokenizer.parse(a, out.written());
}

/// About `bytes` of prose: sentences of `words`, a number or a path now and then, blank lines.
fn prose(a: Allocator, r: std.Random, bytes: usize) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    while (out.items.len < bytes) {
        const n = 5 + r.uintLessThan(usize, 12);
        for (0..n) |i| {
            if (i > 0) try out.append(a, ' ');
            try out.appendSlice(a, words[r.uintLessThan(usize, words.len)]);
        }
        switch (r.uintLessThan(u8, 8)) {
            0 => try out.print(a, " {d}", .{r.uintLessThan(u32, 100_000)}),
            1 => try out.appendSlice(a, " (see `src/server/prompt.zig:109`)"),
            else => {},
        }
        try out.appendSlice(a, if (r.uintLessThan(u8, 5) == 0) ".\n\n" else ". ");
    }
    return out.items;
}

/// About `bytes` of code, as a file read back by a tool: indented lines of names, calls, numbers and braces.
fn code(a: Allocator, r: std.Random, bytes: usize) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    while (out.items.len < bytes) {
        for (0..r.uintLessThan(usize, 4) * 4) |_| try out.append(a, ' ');
        const x = words[r.uintLessThan(usize, words.len)];
        const y = words[r.uintLessThan(usize, words.len)];
        switch (r.uintLessThan(u8, 4)) {
            0 => try out.print(a, "const {s}_{s} = try self.{s}({s}, {d});\n", .{ x, y, y, x, r.uintLessThan(u32, 4096) }),
            1 => try out.print(a, "if ({s}.len > {d}) return error.{s};\n", .{ x, r.uintLessThan(u32, 64), y }),
            2 => try out.print(a, "pub fn {s}({s}: *const {s}) !void {{\n", .{ x, y, x }),
            else => try out.appendSlice(a, "}\n"),
        }
    }
    return out.items;
}

/// A chat request's body: a system prompt, turns of a user, an assistant's reasoning, reply and file read, and the
/// tool's answer, until about `bytes`; a last user message; eight tools. Without `reasoning` the steps carry none (a
/// client that drops it), the rest unchanged (the same turns).
fn conversation(a: Allocator, bytes: usize, reasoning: bool) ![]const u8 {
    var prng = std.Random.DefaultPrng.init(23);
    const r = prng.random();
    var out: std.Io.Writer.Allocating = .init(a);
    const w = &out.writer;
    try w.print("{{\"model\":\"glm5_next\",\"messages\":[{{\"role\":\"system\",\"content\":{f}}}", .{std.json.fmt(try prose(a, r, 8 << 10), .{})});
    var turn: usize = 0;
    var dropped: usize = 0;
    while (out.written().len + dropped < bytes) : (turn += 1) {
        try w.print(",{{\"role\":\"user\",\"content\":{f}}}", .{std.json.fmt(try prose(a, r, 600), .{})});
        const args = try std.fmt.allocPrint(a, "{{\"path\":\"src/{s}/{s}.zig\",\"line\":{d}}}", .{ words[turn % words.len], words[(turn * 7) % words.len], turn });
        const reply = try prose(a, r, 500);
        const thought = try prose(a, r, 1500);
        try w.print(",{{\"role\":\"assistant\",\"content\":{f}", .{std.json.fmt(reply, .{})});
        const field = try std.fmt.allocPrint(a, ",\"reasoning_content\":{f}", .{std.json.fmt(thought, .{})});
        if (reasoning) try w.writeAll(field) else dropped += field.len;
        try w.print(",\"tool_calls\":[{{\"id\":\"call_{d}\",\"type\":\"function\",\"function\":{{\"name\":\"read_file\",\"arguments\":{f}}}}}]}}", .{ turn, std.json.fmt(args, .{}) });
        try w.print(",{{\"role\":\"tool\",\"tool_call_id\":\"call_{d}\",\"content\":{f}}}", .{ turn, std.json.fmt(try code(a, r, 4000), .{}) });
    }
    try w.print(",{{\"role\":\"user\",\"content\":{f}}}],\"tools\":[", .{std.json.fmt(try prose(a, r, 400), .{})});
    for (0..8) |i| try w.print("{s}{{\"type\":\"function\",\"function\":{{\"name\":\"{s}_{d}\",\"description\":{f},\"parameters\":{{\"type\":\"object\",\"properties\":{{\"path\":{{\"type\":\"string\",\"description\":\"the file\"}},\"line\":{{\"type\":\"integer\"}}}},\"required\":[\"path\"]}}}}}}", .{ if (i > 0) "," else "", words[i], i, std.json.fmt(try prose(a, r, 400), .{}) });
    try w.writeAll("]}");
    return out.written();
}

/// A GLM server over the fixture template and glmLike, the prompt cache on (prepare renders the rewind probe).
const Glm = struct {
    hf: hf_text.HfText = undefined,
    backend: rig.Script = .{},
    srv: *server.Server = undefined,

    fn start(g: *Glm, gpa: Allocator) !void {
        g.hf = .{ .gpa = gpa, .arena = .init(gpa), .tok = try glmLike(gpa), .glm = true, .source = source, .compiled = try template.compile(gpa, source, null) };
        var c = rig.default_config;
        c.recipe.glm = true;
        g.srv = try server.Server.init(gpa, std.testing.io, g.backend.engine(), g.hf.text(), c, null);
        g.srv.info.prompt_cache = true;
    }

    fn finish(g: *Glm) void {
        g.srv.deinit();
        g.hf.compiled.?.deinit();
        g.hf.tok.deinit();
        g.hf.arena.deinit();
    }
};

/// Bytes a prepare holds, by what holds them (prepare_gate.textBytes is the measure of the text).
const Measured = struct {
    text: usize,
    body: usize,
    parsed: usize,
    prepared: usize, // the request arena's growth in prompt.prepare and systemPrefixLen
    render: usize, // one render's scratch at its largest
    encode: usize, // one tokenization's scratch at its largest
    ids: usize,

    fn perByte(m: Measured, n: usize) f64 {
        return @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(m.text));
    }

    /// What a prepare holds at most a text byte: what the request holds, and the larger scratch on top of it.
    fn need(m: Measured) f64 {
        return m.perByte(m.body + m.parsed + m.prepared + @max(m.render, m.encode));
    }
};

fn measure(g: *Glm, gpa: Allocator, bytes: usize, thinking: bool, reasoning: bool) !Measured {
    var side = std.heap.ArenaAllocator.init(gpa);
    defer side.deinit();
    const sa = side.allocator();
    const body = try conversation(sa, bytes, reasoning);
    var held: Counting = .{ .child = std.heap.page_allocator };
    var request = std.heap.ArenaAllocator.init(held.allocator());
    defer request.deinit();
    const a = request.allocator();
    const parsed = switch (try json.parse(a, body)) {
        .ok => |v| v,
        .err => return error.BadBody,
    };
    const parsed_bytes = held.peak;
    const tools = parsed.get("tools").?.array;
    var cx: errors.Cx = .{ .a = a };
    // as chat.prepare: the kept reasoning back (when the client dropped it), the gate's measure, the renders
    const input = try prompt.withKept(g.srv, a, .{ .messages = parsed.get("messages").?, .tools = tools, .fields = parsed });
    const messages = input.messages;
    const text_bytes = chat.promptBytes(input);
    const rendered = try prompt.prepare(g.srv, &cx, input, thinking, null);
    try std.testing.expect(rendered.history_len > 0 and rendered.rewind_len > rendered.history_len / 2); // three renders
    _ = prompt.systemPrefixLen(g.srv, &cx, messages, tools, rendered.ids, thinking, null);
    // one render and one tokenization again, as renderIds makes them, their scratch counted
    var sx: errors.Cx = .{ .a = sa };
    const sent = try glm_history.agentHistory(sa, try glm_history.promoteReasoning(sa, messages));
    const normalized = try messages_mod.toolArguments(&sx, try messages_mod.normalize(&sx, sent, g.srv.late_system, g.srv.needs_user_after_tool));
    var context: std.json.ObjectMap = .empty;
    try context.put(sa, "clear_thinking", .{ .bool = false });
    try context.put(sa, "enable_thinking", .{ .bool = thinking });
    try context.put(sa, "thinking_mode", .{ .string = if (thinking) "thinking" else "chat" });
    var render_scratch: Counting = .{ .child = std.heap.page_allocator };
    const text = try template.renderWith(sa, render_scratch.allocator(), &g.hf.compiled.?, try hf_text.toStd(sa, normalized), try hf_text.toStd(sa, .{ .array = tools }), .{ .object = context }, true, null);
    var encode_scratch: Counting = .{ .child = std.heap.page_allocator };
    const ids = try g.hf.tok.encodeWith(sa, text, .{ .scratch = encode_scratch.allocator() });
    if (thinking) try std.testing.expectEqualSlices(u32, rendered.ids, ids); // the render prepare made (thinking off cuts it)
    // the tokenizer's depth-first walk gives the breadth-first walk's ids on GLM's pre-tokenizer
    try std.testing.expectEqualSlices(u32, try g.hf.tok.encodeWith(sa, text, .{ .depth_first = false }), ids);
    return .{ .text = text_bytes, .body = body.len, .parsed = parsed_bytes, .prepared = held.peak - parsed_bytes, .render = render_scratch.peak, .encode = encode_scratch.peak, .ids = rendered.ids.len };
}

test "the prepare gate's reservation covers a long GLM agent conversation's prepare (three renders)" {
    const gpa = std.testing.allocator;
    var g: Glm = .{};
    try g.start(gpa);
    defer g.finish();
    for ([_]bool{ true, false }) |thinking| {
        const m = try measure(&g, gpa, 384 << 10, thinking, true);
        std.debug.print("GLM prepare, thinking {}: {d} text bytes, {d} ids; a text byte: body {d:.1}, parsed {d:.1}, prepare's arena {d:.1}, render scratch {d:.1}, encode scratch {d:.1}; needs {d:.1} of {d}\n", .{ thinking, m.text, m.ids, m.perByte(m.body), m.perByte(m.parsed), m.perByte(m.prepared), m.perByte(m.render), m.perByte(m.encode), m.need(), prepare_gate.default_per_byte });
        try std.testing.expect(m.need() <= @as(f64, @floatFromInt(prepare_gate.default_per_byte)));
    }
}

/// The prompt prepare renders for `body` (thinking on), its kept reasoning put back as chat.prepare does.
fn promptOf(g: *Glm, a: Allocator, body: []const u8) ![]const u32 {
    const parsed = (try json.parse(a, body)).ok;
    var cx: errors.Cx = .{ .a = a };
    const input = try prompt.withKept(g.srv, a, .{ .messages = parsed.get("messages").?, .tools = parsed.get("tools").?.array, .fields = parsed });
    return (try prompt.prepare(g.srv, &cx, input, true, null)).ids;
}

test "the reservation covers a resume whose client drops the reasoning while replies are kept (TF_GLM_KEEP_REASONING)" {
    const gpa = std.testing.allocator;
    var g: Glm = .{};
    try g.start(gpa);
    defer g.finish();
    const k = g.srv.kept_reasoning.?;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    // another conversation's reply: every step of this one is looked up, none is found
    const other = (try json.parse(a, "[{\"role\":\"user\",\"content\":\"unrelated\"}]")).ok;
    const call = (try json.parse(a, "[{\"id\":\"x\",\"function\":{\"name\":\"read_file\",\"arguments\":{}}}]")).ok;
    try k.remember(other, call.array, "reason");
    const none = try measure(&g, gpa, 384 << 10, true, false);
    std.debug.print("GLM prepare, reasoning dropped, none kept for it: {d} text bytes; a text byte: prepare's arena {d:.1}; needs {d:.1} of {d}\n", .{ none.text, none.perByte(none.prepared), none.need(), prepare_gate.default_per_byte });
    try std.testing.expect(none.need() <= @as(f64, @floatFromInt(prepare_gate.default_per_byte)));
    // every step's reply kept, as the server keeps them (the ids are not the server's: the signatures find them)
    const with = try conversation(a, 384 << 10, true);
    const messages = (try json.parse(a, with)).ok.get("messages").?.array;
    var kept: usize = 0;
    for (messages, 0..) |m, i| if (m.get("tool_calls")) |calls| {
        try k.remember(.{ .array = messages[0..i] }, calls.array, m.strField("reasoning_content").?);
        kept += 1;
    };
    // the client drops it all: prepare renders the prompt the reasoning makes, and the gate counts the reasoning
    try std.testing.expectEqualSlices(u32, try promptOf(&g, a, with), try promptOf(&g, a, try conversation(a, 384 << 10, false)));
    for ([_]bool{ true, false }) |thinking| {
        const m = try measure(&g, gpa, 384 << 10, thinking, false);
        std.debug.print("GLM prepare, thinking {}, reasoning dropped, {d} steps' kept: {d} text bytes; a text byte: prepare's arena {d:.1}, render scratch {d:.1}, encode scratch {d:.1}; needs {d:.1} of {d}\n", .{ thinking, kept, m.text, m.perByte(m.prepared), m.perByte(m.render), m.perByte(m.encode), m.need(), prepare_gate.default_per_byte });
        try std.testing.expect(m.text >= none.text + 1500 * kept); // the restored reasoning is in the measure
        try std.testing.expect(m.need() <= @as(f64, @floatFromInt(prepare_gate.default_per_byte)));
    }
}

/// Counts the bytes allocated through it; another thread may read the count meanwhile.
const Tally = struct {
    child: Allocator,
    bytes: std.atomic.Value(usize) = .init(0),

    fn allocator(t: *Tally) Allocator {
        return .{ .ptr = t, .vtable = &.{ .alloc = alloc, .resize = Allocator.noResize, .remap = Allocator.noRemap, .free = Allocator.noFree } };
    }
    fn alloc(ctx: *anyopaque, len: usize, al: std.mem.Alignment, ra: usize) ?[*]u8 {
        const t: *Tally = @ptrCast(@alignCast(ctx));
        const p = t.child.rawAlloc(len, al, ra) orelse return null;
        _ = t.bytes.fetchAdd(len, .monotonic);
        return p;
    }
};

var fake_avail: std.atomic.Value(u64) = .init(0);
fn fakeAvail(_: std.Io) ?u64 {
    return fake_avail.load(.acquire);
}

const Never = struct {
    pub fn check(_: Never) bool {
        return false;
    }
};

test "a chat request waits at the gate before it renders anything, and gives its reservation back when prepared" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var g: Glm = .{};
    try g.start(gpa);
    defer g.finish();
    g.srv.gate = .configured(3, 0, prepare_gate.default_max_waiting);
    g.srv.gate.avail = fakeAvail;
    fake_avail.store(4 << 30, .release); // under the 2 GiB held, the request's own and the 3 GiB floor
    var request = std.heap.ArenaAllocator.init(gpa);
    defer request.deinit();
    const body = (try json.parse(request.allocator(), try conversation(request.allocator(), 64 << 10, true))).ok;
    var tally: Tally = .{ .child = request.allocator() };
    // a prepare in flight holds 2 GiB
    const first = try g.srv.gate.enter(io, 2 << 30, Never{});
    const Run = struct {
        fn run(srv: *server.Server, a: Allocator, parsed: json.Value, out: *?chat.Prepared, done: *std.atomic.Value(bool)) void {
            var cx: errors.Cx = .{ .a = a };
            out.* = chat.prepare(srv, &cx, .{ .messages = parsed.get("messages").?, .tools = parsed.get("tools").?.array, .fields = parsed }, Never{}) catch null;
            done.store(true, .release);
        }
    };
    var prepared: ?chat.Prepared = null;
    var done: std.atomic.Value(bool) = .init(false);
    const th = try std.Thread.spawn(.{}, Run.run, .{ g.srv, tally.allocator(), body, &prepared, &done });
    std.Io.sleep(io, .fromMilliseconds(100), .awake) catch {};
    const waits = !done.load(.acquire);
    const before = tally.bytes.load(.monotonic);
    g.srv.gate.leave(io, first);
    th.join();
    // it waited, and had rendered nothing (a render of its 64 KiB takes several times that)
    try std.testing.expect(waits);
    try std.testing.expect(before < 4 << 10);
    const p = prepared orelse return error.TestUnexpectedResult;
    defer chat.release(g.srv, p.preparing);
    try std.testing.expect(p.prompt_len > 0 and tally.bytes.load(.monotonic) > 64 << 10);
    try std.testing.expectEqual(@as(u64, 1), g.srv.gate.waited);
    try std.testing.expectEqual(@as(u32, 0), g.srv.gate.inflight);
    try std.testing.expectEqual(@as(u64, 0), g.srv.gate.reserved);
}

test "a chat request past the gate's waiter bound is refused as a capacity refusal (429 under the recipe), having rendered nothing" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var g: Glm = .{};
    try g.start(gpa);
    defer g.finish();
    g.srv.gate = .configured(3, 60_000, 1);
    g.srv.gate.avail = fakeAvail;
    fake_avail.store(16 << 30, .release);
    const first = try g.srv.gate.enter(io, 2 << 30, Never{});
    fake_avail.store(1 << 30, .release); // nothing more fits
    // one waits (the bound)
    const Waits = struct {
        fn run(gate: *prepare_gate.Gate, left: *std.atomic.Value(bool), out: *?anyerror) void {
            const F = struct {
                l: *std.atomic.Value(bool),
                pub fn check(f: @This()) bool {
                    return f.l.load(.acquire);
                }
            };
            const t = gate.enter(std.testing.io, 1 << 20, F{ .l = left }) catch |e| {
                out.* = e;
                return;
            };
            gate.leave(std.testing.io, t);
        }
    };
    var left: std.atomic.Value(bool) = .init(false);
    var waited: ?anyerror = null;
    const th = try std.Thread.spawn(.{}, Waits.run, .{ &g.srv.gate, &left, &waited });
    for (0..200) |_| {
        if (g.srv.gate.waiters(io) == 1) break;
        std.Io.sleep(io, .fromMilliseconds(10), .awake) catch {};
    }
    try std.testing.expectEqual(@as(u32, 1), g.srv.gate.waiters(io));
    var request = std.heap.ArenaAllocator.init(gpa);
    defer request.deinit();
    const body = (try json.parse(request.allocator(), try conversation(request.allocator(), 4 << 10, true))).ok;
    var tally: Tally = .{ .child = request.allocator() };
    var cx: errors.Cx = .{ .a = tally.allocator() };
    const before = tally.bytes.load(.monotonic);
    try std.testing.expectError(error.Refused, chat.prepare(g.srv, &cx, .{ .messages = body.get("messages").?, .tools = body.get("tools").?.array, .fields = body }, Never{}));
    try std.testing.expectEqual(errors.Kind.capacity, cx.kind);
    try std.testing.expectEqual(errors.capacity_status, cx.status());
    try std.testing.expectEqualStrings(prepare_gate.full_message, cx.message);
    // the refusal's message is all it allocated: no render ran
    try std.testing.expect(tally.bytes.load(.monotonic) - before < 4 << 10);
    try std.testing.expectEqual(@as(u64, 1), g.srv.gate.refused);
    try std.testing.expectEqual(@as(i64, 0), g.srv.preparing.load(.acquire));
    left.store(true, .release);
    th.join();
    try std.testing.expectEqual(@as(?anyerror, error.Cancelled), waited);
    g.srv.gate.leave(io, first);
    try std.testing.expectEqual(@as(u32, 0), g.srv.gate.inflight);
}
