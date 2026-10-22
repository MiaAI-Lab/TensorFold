//! One reply, as ``ChatApp.chat`` makes it: render, submit to the engine, stream text, reasoning and calls.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const fields_mod = @import("fields.zig");
const reply_text = @import("reply_text.zig");
const tool_stream = @import("tool_stream.zig");
const tool_parse = @import("tool_parse.zig");
const prompt_mod = @import("prompt.zig");
const log = @import("log.zig");
const ids = @import("ids.zig");
const clock = @import("clock.zig");
const Server = @import("server.zig").Server;
const chunk_plan = @import("chunk_plan.zig");
const admission = @import("admission.zig");
const glm_calls = @import("glm_calls.zig");
const glm_reply = @import("glm_reply.zig");
const playout_mod = @import("playout.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub const Prompt = union(enum) { text: []const u8, ids: []const u32 };

/// What a route hands the reply: Python's chat() arguments.
pub const Input = struct {
    messages: Value = .{ .array = &.{} },
    tools: []const Value = &.{},
    prompt: ?Prompt = null,
    max_tokens: ?i64 = null,
    temperature: f64 = 0,
    /// The sampling fields: the request's own (``k in body``), its thinking switches and ``tool_call_required``.
    fields: Value,
    /// The reply's id as its client gets it, so the server's lines for the request carry the same id.
    id: []const u8 = "",
    /// ``logprobs: true`` with this many ``top_logprobs``; null: not asked.
    logprobs: ?u8 = null,
    /// The request's own messages when they hold picture or clip parts (--vision): prompt.prepareMedia renders them.
    media: ?Value = null,
};

/// A streamed piece: content text (a string) or a delta object (reasoning or tool calls).
pub const Sink = struct {
    ctx: *anyopaque,
    call: *const fn (ctx: *anyopaque, delta: Value) error{Closed}!void,
};

pub const Reply = struct {
    content: []const u8,
    stop_sequence: ?[]const u8,
    reasoning: ?[]const u8,
    tool_calls_streamed: bool,
    finish_reason: []const u8,
    prompt_tokens: usize,
    cached_tokens: usize,
    completion_tokens: usize,
    reasoning_tokens: usize,
    runtime: Value,
    speculative: Value,
    /// The template switches the prompt was rendered with (the next turn's prefill renders with them too).
    thinking: bool = false,
    effort: ?[]const u8 = null,
    /// ``choices[0].logprobs`` (``{"content": [...]}``) for a request that asked; null otherwise.
    logprobs: ?Value = null,
    /// The GLM recipe (tool calls, patches 0036-0055): `glm` is on for the checkpoint; `glm_done` says `calls` and
    /// `content` are settled here (the streamer's calls plus the end parser's), so the route parses nothing more.
    glm: bool = false,
    glm_done: bool = false,
    calls: ?[]Value = null,
};

pub const Failure = error{ Refused, Cancelled, Failed, OutOfMemory };

/// Events the engine thread hands this request, read by the request's own thread.
const Mailbox = struct {
    io: std.Io,
    gpa: Allocator,
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    tokens: std.ArrayList(u32) = .empty,
    chunks: std.ArrayList(usize) = .empty, // where each round's tokens end
    rows: std.ArrayList(api.LogprobRow) = .empty, // one a token of `tokens`, for requests with logprobs
    cached: ?u32 = null,
    prefilled_ns: ?i96 = null,
    finished: bool = false,
    reason: api.Reason = .stop,
    stats: api.Stats = .{},
    widths: []u32 = &.{},
    raised: []bool = &.{},
    telemetry: []u8 = "",
    message: []u8 = "",

    fn onEvent(ctx: *anyopaque, _: api.Id, event: *const api.Event) void {
        const m: *Mailbox = @ptrCast(@alignCast(ctx));
        m.mutex.lockUncancelable(m.io);
        defer m.mutex.unlock(m.io);
        switch (event.*) {
            .prefilled => |cached| {
                m.cached = cached;
                m.prefilled_ns = std.Io.Clock.awake.now(m.io).toNanoseconds();
            },
            .logprobs => |r| m.rows.appendSlice(m.gpa, r) catch {},
            .tokens => |t| {
                m.tokens.appendSlice(m.gpa, t) catch {};
                m.chunks.append(m.gpa, m.tokens.items.len) catch {};
            },
            .finished => |f| {
                m.finished = true;
                m.reason = f.reason;
                m.stats = f.stats;
                m.widths = m.gpa.dupe(u32, f.stats.prefill_widths) catch &.{};
                m.raised = m.gpa.dupe(bool, f.stats.prefill_raised) catch &.{};
                m.telemetry = m.gpa.dupe(u8, f.stats.telemetry_json) catch "";
                m.message = m.gpa.dupe(u8, f.message) catch "";
            },
        }
        m.cond.signal(m.io);
    }

    fn deinit(m: *Mailbox) void {
        m.tokens.deinit(m.gpa);
        m.chunks.deinit(m.gpa);
        m.rows.deinit(m.gpa);
        m.gpa.free(m.widths);
        m.gpa.free(m.raised);
        m.gpa.free(m.telemetry);
        m.gpa.free(m.message);
    }
};

const nowNs = clock.nowNs;
const seconds = clock.seconds;

/// Python's ``bool(fields.get(key))`` on the sampling fields.
fn flag(f: Value, key: []const u8) ?bool {
    const v = f.get(key) orelse return null;
    if (v == .null) return null;
    return v.truthy();
}

/// A request rendered and checked but not submitted; it holds the preparing count until generate or release.
pub const Prepared = struct {
    input: Input,
    request: api.Request,
    prompt_len: usize,
    received: i96,
    thinking: bool,
    effort: ?[]const u8,
    sampled: bool, // a sampling config, which the reply labels "exact"; greedy otherwise
    drafts: bool,
    stops: fields_mod.Stops,
    preparing: bool,
};

/// Render ``input`` and run every check that can refuse it, before anything reaches the client or the engine.
pub fn prepare(srv: *Server, cx: *Cx, input: Input, gone: anytype) Failure!Prepared {
    const a = cx.a;
    const io = srv.io;
    const received = nowNs(io);
    const f = input.fields;
    const stops_opt = try fields_mod.stopOptions(cx, f);
    var limit: i64 = @max(1, if (input.max_tokens) |m| (if (m != 0) m else srv.config.default_max_tokens) else srv.config.default_max_tokens);
    const priority = f.get("priority");
    const background = prompt_mod.isTitle(input.messages, input.tools.len > 0) or
        (priority != null and priority.? == .string and std.mem.eql(u8, priority.?.string, "background"));
    const preparing = !background;
    if (preparing) _ = srv.preparing.fetchAdd(1, .acq_rel);
    errdefer release(srv, preparing);
    if (gone.check()) return error.Cancelled;
    var thinking = flag(f, "enable_thinking") orelse srv.config.enable_thinking;
    if (input.prompt != null) thinking = false;
    if (f.get("clear_thinking")) |v| cx.clear_thinking = v.truthy(); // chat_template_kwargs.clear_thinking, for every render
    const effort = srv.effortFor(if (f.get("reasoning_effort")) |e| (if (e == .string) e.string else null) else null);
    const rendered = try prompt_mod.prepare(srv, cx, input, thinking, effort);
    if (gone.check()) return error.Cancelled;
    if (rendered.ids.len == 0) return cx.refuse("rendered prompt is empty");
    const window: i64 = srv.info.context_window;
    const n: i64 = @intCast(rendered.ids.len);
    if (window > 0) {
        const room = window - n;
        if (srv.config.recipe.glm) {
            // the recipe's GlmApp.check (patch 0044): a request that sets max_tokens must fit whole, one that does not
            // gets the default budget cut to what the window has left, never refused
            const asked: i64 = if (input.max_tokens) |m| m else 0;
            const need = n + (if (asked != 0) asked else 1);
            if (need > window) return cx.fail(.context_length, "{s} {d} tokens: this request needs a {d}-token context ({s}), which exceeds the context window this server was started for; shorten the prompt or reply", .{ errors.context_limit, window, need, if (asked != 0) try std.fmt.allocPrint(a, "{d} prompt tokens plus max_tokens {d}", .{ n, asked }) else try std.fmt.allocPrint(a, "a {d}-token prompt", .{n}) });
        } else {
            if (room < 1) return cx.fail(.context_length, "{s} {d} tokens{s}, but the rendered prompt has {d} tokens and leaves no room for a reply, which exceeds the context window. Compact or shorten the conversation.", .{ errors.context_limit, window, if (srv.info.context_fitted) ", the most this server's memory budget fits" else "", n });
            if (input.max_tokens != null and limit > room) return cx.fail(.context_length, "{s} {d} tokens, but the rendered prompt has {d} tokens and requests {d} reply tokens, which exceeds the context window. Reduce the prompt to at most {d} prompt tokens or request at most {d} reply tokens, including chat template and thinking tokens.", .{ errors.context_limit, window, n, limit, @max(0, window - limit), room });
        }
        limit = @min(limit, room);
    }
    const system_len: usize = if (input.prompt != null or rendered.media != null) 0 else prompt_mod.systemPrefixLen(srv, cx, input.messages, input.tools, rendered.ids, thinking, effort);
    var shared: std.ArrayList(u32) = .empty;
    if (system_len > 0) for ([_]i64{ @as(i64, @intCast(system_len)) - 2048, @as(i64, @intCast(system_len)) - 512, @intCast(system_len) }) |cut| {
        if (cut >= 512) try shared.append(a, @intCast(cut));
    };
    const sampling = try srv.resolveSampling(cx, f, input.temperature, rendered.ids);
    const draft_field = f.get("draft");
    const drafts = srv.config.use_drafts and !(draft_field != null and draft_field.? == .bool and !draft_field.?.bool);
    var request: api.Request = .{
        .prompt = rendered.ids,
        .max_tokens = @intCast(@min(limit, std.math.maxInt(u32))),
        .sampling = sampling,
        .eos = if (stops_opt.ignore_eos) &.{} else srv.eos,
        .drafts = drafts,
        .background = background,
        .history_len = @intCast(rendered.history_len),
        .rewind_len = @intCast(rendered.rewind_len),
        .shared_prefixes = shared.items,
        .chunks = try chunk_plan.withCut(a, try srv.chunks.starts(a, rendered.ids), if (srv.chunks.step > 0) @intCast(@max(system_len, 1) - 1) else 0, rendered.ids.len, srv.chunks.min_chunk), // a cut before the conversation's own text: fresh sessions resume their whole harness
        .decode_spans = try prompt_mod.replySpans(srv, cx, rendered.ids), // from the tokens alone, for every request: cached states are keyed by tokens
        .media = rendered.media,
        .tools_json = if (input.tools.len > 0) try json.stringify(a, .{ .array = @constCast(input.tools) }, .{ .ascii = false }) else "",
        .decide = if (input.prompt == null and !background) prompt_mod.decideOf(srv, rendered.ids) else null,
    };
    try srv.checkFeatures(cx, f, input.tools.len > 0, thinking, rendered.ids, input.tools, &request);
    if (thinking) {
        const budget_field = f.get("thinking_budget");
        const budget: i64 = if (budget_field != null and budget_field.?.truthy()) budget_field.?.int64() orelse 0 else srv.config.thinking_budget;
        if (srv.think_close.len > 0) request.loop_guard = srv.config.loop_guard;
        request.loop_recipe = request.loop_guard and srv.config.recipe.glm;
        if ((budget > 0 or request.loop_guard) and srv.think_close.len > 0) {
            if (budget > 0) request.think_budget = @intCast(@min(budget, std.math.maxInt(u32)));
            request.think_close = srv.think_close;
            request.think_end = srv.think_close_end;
        }
    }
    if (input.logprobs) |count| request.logprobs = try @import("logprobs.zig").admit(srv, cx, thinking, request.think_budget, count);
    return .{ .input = input, .request = request, .prompt_len = rendered.ids.len, .received = received, .thinking = thinking, .effort = effort, .sampled = sampling != null, .drafts = drafts, .stops = stops_opt, .preparing = preparing };
}

/// Submit a prepared request and collect its reply; ``sink`` hears the stream (null: not streamed).
pub fn generate(srv: *Server, cx: *Cx, prepared: Prepared, sink: ?Sink, gone: anytype) Failure!Reply {
    const a = cx.a;
    const io = srv.io;
    var request = prepared.request;
    const input = prepared.input;
    const received = prepared.received;
    const thinking = prepared.thinking;
    const effort = prepared.effort;
    const stops_opt = prepared.stops;
    var preparing = prepared.preparing;
    defer release(srv, preparing);
    if (request.background) {
        while (true) {
            const now = nowNs(io);
            const waiting = now < received + 150 * std.time.ns_per_ms or (srv.preparing.load(.acquire) > 0 and now < received + 2 * std.time.ns_per_s);
            if (!waiting) break;
            if (gone.check()) return error.Cancelled;
            std.Io.sleep(io, .fromMilliseconds(5), .awake) catch {};
        }
    }
    if (gone.check()) return error.Cancelled;
    var stop_hook: StopHook = .{ .srv = srv, .stops = .{ .strings = stops_opt.strings } };
    if (stops_opt.strings.len > 0) request.stop = .{ .ctx = &stop_hook, .check = StopHook.check };
    var box: Mailbox = .{ .io = io, .gpa = srv.gpa };
    defer box.deinit();
    const id = srv.next_id.fetchAdd(1, .monotonic);
    var slot = try admission.enter(srv, cx, request.background); // TF_GLM_MAX_QUEUED: past the lanes plus the cap, 429
    defer slot.leave();
    // TF_GLM_STREAM_SMOOTH=1: a streamed reply's tokens go out one event each at a steady pace from a short playout buffer
    // (playout.zig), so drafted bursts and other requests' prompt fills do not show
    var playout: playout_mod.Playout = undefined;
    var smoothing = false;
    var left = Left(@TypeOf(gone)){ .gone = gone };
    var wrapped = sink;
    if (sink != null and srv.config.recipe.glm and srv.config.recipe.stream_smooth) {
        playout.start(srv.gpa, io, srv.config.recipe.stream_smooth_ms, sink.?, .{ .ctx = &left, .check = Left(@TypeOf(gone)).check }) catch return cx.fail(.server, "the stream's playout could not start", .{});
        smoothing = true;
        wrapped = playout.sink();
    }
    defer if (smoothing) playout.abort(); // a no-op once closed
    if (srv.keepalive) |k| k.begin(); // the GPU is busy: the idle ticker holds its commits
    defer if (srv.keepalive) |k| k.end();
    srv.engine.submit(id, &request, .{ .ctx = &box, .event = Mailbox.onEvent }) catch |e| return switch (e) {
        error.Busy => cx.fail(.capacity, "the engine is busy; retry shortly", .{}),
        error.Closed => cx.fail(.other, "the scheduler is closed", .{}),
        error.InvalidSpans => cx.fail(.request, "the prompt span layout is invalid", .{}),
    };
    release(srv, preparing); // a background request waits only while a foreground one prepares
    preparing = false;
    var gen: Generation = .{ .srv = srv, .a = a, .box = &box, .id = id, .reply_id = input.id, .sink = wrapped, .smooth = if (smoothing) &playout else null, .thinking = thinking or reply_text.isChannel(srv.markers), .stops = .{ .strings = stops_opt.strings }, .ignore_eos = stops_opt.ignore_eos, .max_tokens = request.max_tokens, .tools = input.tools, .logprobs = request.logprobs != null, .return_ids = if (input.fields.get("return_token_ids")) |v| v.truthy() else false };
    defer srv.noteRequest(prepared.prompt_len, box.cached orelse 0, gen.collected.items.len, box.stats.drafted, box.stats.accepted, box.stats.rounds, received, gen.first_ns, gen.last_ns, box.stats.prefill_seconds);
    errdefer if (!gen.engine_done) gen.cancel(); // the engine writes to the mailbox until it says finished
    const result: Failure!Reply = blk: {
        if (srv.glm_calls and input.tools.len > 0) {
            // GLM's calls go out whole while they are written (one call a delta), those in the think block when the
            // reply ends on them; one call at most under parallel_tool_calls: false, which only the end parser reads
            const single = if (input.fields.get("tool_call_single")) |x| x.truthy() else false;
            const streams = sink != null and !single;
            gen.glm = .{
                .think = if (gen.thinking) .{ .a = a, .hold = if (streams) glm_calls.think_call_hold else null } else null,
                .streamer = if (streams) glm_calls.CallStreamer.init(a, input.tools) catch |e| break :blk e else null,
                .single = single,
                .quiet_since = nowNs(io),
            };
        } else if (sink != null and input.tools.len > 0) gen.calls = tool_stream.Streamer.init(a, input.tools) catch |e| break :blk e;
        gen.loop(gone) catch |e| break :blk e;
        const finished = gen.finish(cx, prepared.prompt_len, received, thinking, effort, prepared.sampled, prepared.drafts);
        if (smoothing) { // what is still queued goes out now, before the reply's end
            if (finished) |_| playout.close() else |_| playout.abort();
            if (playout.gone) {
                if (finished) |_| break :blk error.Cancelled else |_| {}
            }
        }
        break :blk finished;
    };
    if (result) |_| {} else |e| logEnded(input.id, e, cx.message, prepared.prompt_len, gen.collected.items.len, seconds(nowNs(io) - received));
    return result;
}

/// The ``ended`` line of a submitted reply that ends without one: its client left, or it failed.
fn logEnded(id: []const u8, e: Failure, message: []const u8, prompt: usize, tokens: usize, after: f64) void {
    const why: ?[]const u8 = switch (e) {
        error.Cancelled => null,
        error.Refused => message, // once submitted, only the engine's own failure refuses
        else => @errorName(e),
    };
    var buf: [1024]u8 = undefined;
    log.line("{s}", .{log.ended(&buf, id, why, prompt, tokens, after)});
}

/// The reply to ``input``; ``sink`` hears the stream (null: not streamed). ``gone`` says the client left.
pub fn run(srv: *Server, cx: *Cx, input: Input, sink: ?Sink, gone: anytype) Failure!Reply {
    if (srv.config.compact_at == null) return generate(srv, cx, try prepare(srv, cx, input, gone), sink, gone);
    return @import("compact.zig").run(srv, cx, input, sink, gone);
}

/// Give back a foreground request's preparing count: at its submit, or when it is never generated.
pub fn release(srv: *Server, preparing: bool) void {
    if (preparing) _ = srv.preparing.fetchSub(1, .acq_rel);
}

/// The text a stream has sent: Python's ``streamed = visible``, grown by its delta while it only grows.
const Shown = struct {
    text: std.ArrayList(u8) = .empty,
    chars: usize = 0,
    at: ?[*]const u8 = null, // where the last shown text lay; the decode buffer only grows, so that prefix holds
    extends: bool = false,

    /// Python's ``now[len(shown):]``; ``stable``: ``now`` lies in the append-only decode buffer.
    fn after(s: *Shown, now: []const u8, stable: bool) []const u8 {
        const sent = s.text.items;
        s.extends = (stable and s.at == now.ptr and now.len >= sent.len) or std.mem.startsWith(u8, now, sent);
        if (!s.extends) return reply_text.afterChars(now, s.chars);
        if (stable) s.at = now.ptr; // the buffer moved but kept its bytes: the next check is free again
        return now[sent.len..];
    }

    fn set(s: *Shown, a: Allocator, now: []const u8, delta: []const u8) Allocator.Error!void {
        if (s.extends) {
            try s.text.appendSlice(a, delta);
            s.chars += reply_text.charCount(delta);
        } else {
            s.text = .empty;
            try s.text.appendSlice(a, now);
            s.chars = reply_text.charCount(now);
        }
        s.at = now.ptr;
    }
};

/// Whether a client who has left ends the request now: always, except under TF_GLM_PROMPT_STOP=0 while its prompt is still
/// filling (every prompt then fills to its end).
fn leaves(r: @import("recipe.zig").Recipe, box: *Mailbox) bool {
    if (!r.glm or r.prompt_stop) return true;
    box.mutex.lockUncancelable(box.io);
    defer box.mutex.unlock(box.io);
    return box.prefilled_ns != null or box.finished;
}

/// A client that left, asked from the playout's writer thread.
fn Left(comptime G: type) type {
    return struct {
        gone: G,

        fn check(ctx: *anyopaque) bool {
            const l: *@This() = @ptrCast(@alignCast(ctx));
            return l.gone.check();
        }
    };
}

/// What the recipe's GLM tool-call handling keeps while a reply streams.
const GlmState = struct {
    think: ?glm_calls.ThinkSplit,
    streamer: ?glm_calls.CallStreamer,
    single: bool,
    quiet_since: i96, // the last event sent while a call is held (an empty delta every `keepalive_s` meanwhile)
};

/// The token loop and the text it streams.
const Generation = struct {
    srv: *Server,
    a: Allocator,
    box: *Mailbox,
    id: api.Id,
    reply_id: []const u8, // the id the client gets, which the done line prints
    sink: ?Sink,
    thinking: bool,
    stops: reply_text.Stops,
    ignore_eos: bool,
    max_tokens: u32,
    tools: []const Value,
    calls: ?tool_stream.Streamer = null,
    glm: ?GlmState = null,
    smooth: ?*playout_mod.Playout = null,
    return_ids: bool = false, // return_token_ids: the reply's ids come back in the "tensorfold" block
    string_stop: bool = false, // a stop string, not the model's end token, ended the reply
    collected: std.ArrayList(u32) = .empty,
    visible: reply_text.Incremental = .{},
    hidden: std.ArrayList(u8) = .empty, // reused for the answer without its call blocks
    streamed: Shown = .{}, // what content streamed, kept apart from the decode buffer that grows under slices
    streamed_reasoning: Shown = .{},
    streaming_done: bool = false,
    first_ns: ?i96 = null,
    last_ns: ?i96 = null,
    reason: ?[]const u8 = null, // the server ended the reply (a stop string, the length) before the engine said so
    engine_done: bool = false,
    consumed: usize = 0,
    chunk_index: usize = 0,
    logprobs: bool = false, // the request asked for logprobs: the mailbox holds a row a token

    fn eos(g: *const Generation, t: u32) bool {
        return !g.ignore_eos and std.mem.indexOfScalar(u32, g.srv.eos, t) != null;
    }

    /// What closes a call the model's end token left open (``tool_parse.closeCall``); nothing when a stop string, the length or ``ignore_eos`` ended the reply instead.
    fn closeCall(g: *const Generation, text: []const u8) Allocator.Error![]const u8 {
        const t = g.collected.items;
        if (g.tools.len == 0 or t.len == 0 or !g.eos(t[t.len - 1])) return "";
        return tool_parse.closeCall(g.a, text, g.tools);
    }

    /// The engine's stop check, here: the newest tokens' text holds a stop string.
    fn stopHit(g: *Generation) Allocator.Error!bool {
        if (g.stops.strings.len == 0) return false;
        const t = g.collected.items;
        const tail = t[t.len -| g.stops.tail()..];
        const text = try g.srv.text.decode(g.a, tail);
        for (g.stops.strings) |s| if (std.mem.indexOf(u8, text, s) != null) return true;
        return false;
    }

    /// Commits a round's tokens as ``LaneStream.commit`` would, then streams what they add.
    fn commit(g: *Generation, chunk: []const u32) Failure!void {
        var landed: usize = 0;
        for (chunk) |t| {
            if (g.reason != null) break;
            try g.collected.append(g.a, t);
            landed += 1;
            if (g.eos(t)) {
                g.reason = "stop";
            } else if (try g.stopHit()) {
                g.reason = "stop";
                g.string_stop = true;
                g.srv.engine.cancel(g.id); // the engine ends EOS and length itself; a stop string is the server's
            } else if (g.collected.items.len >= g.max_tokens) g.reason = "length";
        }
        if (landed == 0) return;
        const arrived = nowNs(g.srv.io);
        if (g.first_ns == null) g.first_ns = arrived;
        g.last_ns = arrived;
        const sink = g.sink orelse return;
        if (g.streaming_done) return;
        var fresh: std.ArrayList(u32) = .empty;
        for (chunk[0..landed]) |t| {
            if (g.eos(t)) {
                g.streaming_done = true;
                break;
            }
            try fresh.append(g.a, t);
        }
        const all = try g.visible.extend(g.a, g.srv.text, decodeText, fresh.items);
        const text = g.stops.visible(all, true);
        if (g.visible.pending()) return; // a character still split across tokens
        var answer = text;
        var wrote = false;
        if (g.thinking) {
            const split = if (g.glm) |*gl| (if (gl.think) |*t| try t.split(text, false) else try reply_text.splitThinking(g.a, text, false, g.srv.markers)) else try reply_text.splitThinking(g.a, text, false, g.srv.markers);
            const piece = g.streamed_reasoning.after(split.reasoning, true);
            if (piece.len > 0) {
                try g.streamed_reasoning.set(g.a, split.reasoning, piece);
                try g.emit(sink, try deltaOf(g.a, "reasoning_content", piece));
                wrote = true;
            }
            answer = split.answer;
        }
        const hiding = g.calls != null or g.glm != null;
        const shown = if (hiding) try reply_text.hideInto(g.a, &g.hidden, answer, false) else answer;
        const vis = reply_text.heldBack(reply_text.streamingVisible(shown), hiding);
        const copied = @intFromPtr(vis.ptr) >= @intFromPtr(g.hidden.items.ptr) and @intFromPtr(vis.ptr) <= @intFromPtr(g.hidden.items.ptr) + g.hidden.items.len;
        const delta = g.streamed.after(vis, !copied);
        if (delta.len > 0) {
            try g.streamed.set(g.a, vis, delta);
            try g.emit(sink, .{ .string = delta });
            wrote = true;
        }
        if (g.glm) |*gl| if (gl.streamer) |*cs| try g.streamGlmCalls(sink, gl, cs, text, wrote);
        if (g.calls) |*c| {
            var out: std.ArrayList(Value) = .empty;
            try c.feed(answer, &out); // never the reasoning: a call it mentions is not made
            for (out.items) |d| try g.emit(sink, d);
        }
    }

    /// The calls that became whole in this round, each as one delta; while one is held, an empty delta every
    /// `keepalive_s` keeps an idle client waiting (past the idle cut of LiteLLM and Node's fetch).
    fn streamGlmCalls(g: *Generation, sink: Sink, gl: *GlmState, cs: *glm_calls.CallStreamer, text: []const u8, wrote: bool) Failure!void {
        const feed_text = if (gl.think) |t| (if (t.stream_from) |from| text[from..] else "") else text;
        const parts = try cs.feed(feed_text);
        const now = nowNs(g.srv.io);
        for (parts) |d| try g.emit(sink, d);
        if (parts.len > 0 or wrote) {
            gl.quiet_since = now;
        } else if (cs.open and seconds(now - gl.quiet_since) >= glm_calls.keepalive_s) {
            try g.emit(sink, .{ .object = try json.newObject(g.a) });
            gl.quiet_since = now;
        }
    }

    fn emit(g: *Generation, sink: Sink, delta: Value) Failure!void {
        sink.call(sink.ctx, delta) catch {
            g.cancel();
            return error.Cancelled;
        };
    }

    /// Ends the request and waits for the engine; one it ended unfinished counts as a disconnect, as the Mac scheduler counts it.
    fn cancel(g: *Generation) void {
        if (!g.engine_done) g.srv.engine.cancel(g.id);
        g.drain();
        if (g.box.reason == .cancelled and g.reason == null) g.srv.metrics.disconnected(g.srv.io);
    }

    /// Waits for the engine's own end, so the request it holds may be freed.
    fn drain(g: *Generation) void {
        const m = g.box;
        m.mutex.lockUncancelable(m.io);
        defer m.mutex.unlock(m.io);
        while (!m.finished) m.cond.waitUncancelable(m.io, &m.mutex);
        g.engine_done = true;
    }

    /// Takes rounds until the reply ends; a client that leaves cancels it.
    fn loop(g: *Generation, gone: anytype) Failure!void {
        const m = g.box;
        while (true) {
            m.mutex.lockUncancelable(m.io);
            var chunk: ?[]u32 = null;
            var ended = false;
            if (g.chunk_index < m.chunks.items.len) {
                const end = m.chunks.items[g.chunk_index];
                g.chunk_index += 1;
                chunk = g.a.dupe(u32, m.tokens.items[g.consumed..end]) catch null;
                g.consumed = end;
            } else if (m.finished) {
                ended = true;
            } else {
                m.cond.waitTimeout(m.io, &m.mutex, .{ .duration = .{ .raw = .fromMilliseconds(50), .clock = .awake } }) catch {};
            }
            m.mutex.unlock(m.io);
            if (ended) {
                g.engine_done = true;
                return;
            }
            if (gone.check() and leaves(g.srv.config.recipe, m)) {
                g.cancel();
                return error.Cancelled;
            }
            const tokens = chunk orelse continue;
            if (g.smooth) |p| {
                // token by token into the playout queue, unless the reply ends in this round (then the queue is
                // flushed: the end is never later)
                for (tokens) |t| {
                    if (g.reason != null) break;
                    try g.commit(&.{t});
                    if (g.reason != null) p.flush();
                }
            } else if (g.reason == null) try g.commit(tokens);
        }
    }

    fn finish(g: *Generation, cx: *Cx, prompt_len: usize, received: i96, thinking: bool, effort: ?[]const u8, exact: bool, drafts: bool) Failure!Reply {
        const a = g.a;
        const m = g.box;
        if (!g.engine_done) g.drain();
        const reason: []const u8 = g.reason orelse switch (m.reason) {
            .stop => "stop",
            .length => "length",
            .cancelled => "cancelled",
            .failed => return cx.other(if (m.message.len > 0) try a.dupe(u8, m.message) else "the reply failed"),
        };
        const finished_ns = nowNs(g.srv.io);
        const content_tokens = if (g.ignore_eos) g.collected.items else reply_text.stripTrailing(g.collected.items, g.srv.eos);
        const raw = try g.srv.text.decode(a, content_tokens);
        const visible = g.stops.visible(raw, false);
        // the model's end token ended the reply (not a stop string, the token limit or ignore_eos)
        const ended = g.collected.items.len > 0 and g.eos(g.collected.items[g.collected.items.len - 1]) and !g.string_stop;
        // closed before the think split, so a call ending an unclosed think block is the answer
        const close = if (g.glm != null) (if (ended) try glm_calls.closeCall(a, visible, g.tools) else "") else try g.closeCall(visible);
        const text = if (close.len > 0) try std.mem.concat(a, u8, &.{ visible, close }) else visible;
        const recipe_glm = g.srv.config.recipe.glm;
        var content: []const u8 = text;
        var reasoning: ?[]const u8 = null;
        var settled: ?glm_reply.Result = null;
        if (g.glm) |*gl| {
            const r = try glm_reply.finalize(a, .{
                .text = text,
                .tools = g.tools,
                .thinking = g.thinking,
                .markers = g.srv.markers,
                .think = if (gl.think) |*t| t else null,
                .streamer = if (gl.streamer) |*cs| cs else null,
                .single = gl.single,
                .ended = ended,
            });
            settled = r;
            content = r.content;
            reasoning = if (r.reasoning.len > 0) r.reasoning else null;
        } else if (g.thinking) {
            const split = try reply_text.splitThinking(a, text, true, g.srv.markers);
            // the recipe's server returns the reasoning as the model wrote it
            const r = if (recipe_glm) split.reasoning else reply_text.pyStrip(split.reasoning);
            reasoning = if (r.len > 0) r else null;
            content = split.answer;
        } else if (!recipe_glm) {
            const h = reply_text.parseHarmony(text);
            content = h.content;
            reasoning = h.reasoning;
        }
        if (g.sink) |sink| {
            const thought = g.streamed_reasoning.text.items;
            if (reasoning) |r| if (std.mem.startsWith(u8, r, thought) and r.len > thought.len)
                try g.emit(sink, try deltaOf(a, "reasoning_content", r[thought.len..]));
            if (settled) |r| {
                const sent = g.streamed.text.items;
                if (std.mem.startsWith(u8, r.content, sent) and r.content.len > sent.len) try g.emit(sink, .{ .string = r.content[sent.len..] });
                for (r.call_deltas) |d| try g.emit(sink, d); // the calls that became whole at the end
                // under parallel_tool_calls: false the route sends the one call, the stream filter drops it here
                if (r.calls) |all| if (!g.glm.?.single) for (all[r.streamed..], r.streamed..) |call, i| try g.emit(sink, try glm_reply.wholeDelta(a, i, call));
            } else {
                const shown = if (g.calls != null) try reply_text.finishedProse(a, content) else content;
                const sent = g.streamed.text.items;
                if (std.mem.startsWith(u8, shown, sent) and shown.len > sent.len) try g.emit(sink, .{ .string = shown[sent.len..] });
                if (g.calls) |*c| if (close.len > 0) {
                    // the streamer reads the closers as markup the model wrote, so the call ends with the same deltas
                    var out: std.ArrayList(Value) = .empty;
                    try c.feed(if (g.thinking) content else text, &out);
                    for (out.items) |d| try g.emit(sink, d);
                };
            }
        }
        const prefilled = m.prefilled_ns;
        const decode_s: f64 = if (prefilled) |p| @max(0, seconds(finished_ns - p)) else 0;
        const decode_tokens = g.collected.items.len -| 1;
        const runtime = try json.newObject(a);
        try runtime.put(a, "enable_thinking", .{ .bool = thinking });
        try runtime.put(a, "reasoning_effort", if (!thinking) .{ .string = "none" } else if (effort) |e| .{ .string = e } else .null);
        try runtime.put(a, "engine", .{ .string = g.srv.info.name });
        try runtime.put(a, "tokens_per_second", .{ .float = if (decode_s > 0) @as(f64, @floatFromInt(decode_tokens)) / decode_s else 0 });
        try runtime.put(a, "seconds", .{ .float = @max(0, decode_s) }); // the reply's decode seconds, as tokens_per_second divides them
        try runtime.put(a, "prefill_seconds", if (m.stats.prefill_seconds) |p| .{ .float = p } else .null);
        const widths = try a.alloc(Value, m.widths.len);
        for (m.widths, widths) |w, *slot| slot.* = try json.intValue(a, w);
        try runtime.put(a, "prefill_widths", .{ .array = widths });
        const raised = try a.alloc(Value, m.raised.len);
        for (m.raised, raised) |r, *slot| slot.* = .{ .bool = r };
        try runtime.put(a, "prefill_raised", .{ .array = raised });
        try runtime.put(a, "time_to_first_token", if (g.first_ns) |t| .{ .float = seconds(t - received) } else .null);
        try runtime.put(a, "sampling", .{ .string = if (exact) "exact" else "greedy" });
        try runtime.put(a, "drafts", .{ .bool = drafts });
        const sha = @import("tokens.zig").tokenSha(g.collected.items);
        try runtime.put(a, "token_sha", .{ .string = try a.dupe(u8, &sha) });
        try runtime.put(a, "min_rows", try json.intValue(a, m.stats.min_rows));
        if (g.return_ids) { // the reply's ids in the "tensorfold" block, for exactness checks (tools/end_of_turn.py)
            const list = try a.alloc(Value, g.collected.items.len);
            for (g.collected.items, list) |t, *slot| slot.* = try json.intValue(a, t);
            try runtime.put(a, "token_ids", .{ .array = list });
        }
        if (m.stats.loop_period) |period| {
            const loop_field = try json.newObject(a);
            try loop_field.put(a, "period", try json.intValue(a, period));
            try runtime.put(a, "loop", .{ .object = loop_field });
            // the think block collapsed and was closed (TF_GLM_LOOP_GUARD): `usage.tensorfold.loop_guard` counts it
            if (recipe_glm) try runtime.put(a, "loop_guard", .{ .int = "1" });
        }
        const s = m.stats;
        const spec = try json.newObject(a);
        try spec.put(a, "rounds", try json.intValue(a, s.rounds));
        try spec.put(a, "drafted", try json.intValue(a, s.drafted));
        try spec.put(a, "accepted", try json.intValue(a, s.accepted));
        try spec.put(a, "acceptance_rate", .{ .float = if (s.drafted > 0) @as(f64, @floatFromInt(s.accepted)) / @as(f64, @floatFromInt(s.drafted)) else 0 });
        try spec.put(a, "tokens_per_round", .{ .float = if (s.rounds > 0) @as(f64, @floatFromInt(g.collected.items.len)) / @as(f64, @floatFromInt(s.rounds)) else 0 });
        if (m.telemetry.len > 0) if ((try json.parse(a, m.telemetry)) == .ok) try spec.put(a, "proposer", (try json.parse(a, m.telemetry)).ok);
        const think_end: ?u32 = if (thinking) g.srv.text.tokenId(g.srv.markers.close) else null;
        const reply: Reply = .{
            .content = content,
            .stop_sequence = g.stops.matched(raw),
            .reasoning = reasoning,
            .tool_calls_streamed = if (settled) |r| (r.calls != null and !g.glm.?.single) else g.calls != null and g.calls.?.streamed,
            .glm = recipe_glm,
            .glm_done = settled != null,
            .calls = if (settled) |r| r.calls else null,
            .finish_reason = if (settled) |r| (if (r.calls != null and !r.cut) "tool_calls" else reason) else reason,
            .prompt_tokens = prompt_len,
            .cached_tokens = m.cached orelse 0,
            .completion_tokens = g.collected.items.len,
            .reasoning_tokens = reply_text.reasoningCount(g.collected.items, think_end),
            .runtime = .{ .object = runtime },
            .speculative = .{ .object = spec },
            .thinking = thinking,
            .effort = effort,
            .logprobs = if (g.logprobs) try @import("logprobs.zig").value(g.srv, cx, a, g.thinking, content_tokens, m.rows.items) else null,
        };
        if (std.mem.eql(u8, reason, "length") and thinking and reply_text.pyStrip(content).len == 0)
            log.line("warning: a reply reached max_tokens while still thinking, so its content is empty and its text is all in reasoning_content; raise max_tokens, or send chat_template_kwargs {{\"enable_thinking\": false}} (server: --no-thinking)", .{});
        var cycle_text: [32]u8 = undefined;
        const cycle = if (s.loop_period) |period| std.fmt.bufPrint(&cycle_text, " loop=period:{d}", .{period}) catch "" else "";
        log.line("done {s} prompt={d} cached={d} thinking={s} effort={s} tokens={d} sha={s} finish={s}{s} rounds={d} accepted={d}/{d}", .{ g.reply_id, prompt_len, reply.cached_tokens, if (thinking) "True" else "False", if (thinking) effort orelse "none" else "none", g.collected.items.len, sha, reason, cycle, s.rounds, s.accepted, s.drafted });
        return reply;
    }
};

/// The request's stop strings as the engine checks them after each token: the newest tokens' text holds one.
const StopHook = struct {
    srv: *Server,
    stops: reply_text.Stops,

    fn check(ctx: *anyopaque, emitted: []const u32) bool {
        const h: *StopHook = @ptrCast(@alignCast(ctx));
        var arena: std.heap.ArenaAllocator = .init(h.srv.gpa);
        defer arena.deinit();
        const tail = emitted[emitted.len -| h.stops.tail()..];
        const text = h.srv.text.decode(arena.allocator(), tail) catch return false;
        for (h.stops.strings) |stop| if (std.mem.indexOf(u8, text, stop) != null) return true;
        return false;
    }
};

fn decodeText(t: anytype, a: Allocator, tokens: []const u32) ![]u8 {
    return t.decode(a, tokens);
}

/// ``{key: text}``: a streamed delta (role, content or reasoning_content).
pub fn deltaOf(a: Allocator, key: []const u8, text: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, key, .{ .string = text });
    return .{ .object = o };
}

test "a client that left ends the request at once, or after its prompt under TF_GLM_PROMPT_STOP=0" {
    var box: Mailbox = .{ .io = std.testing.io, .gpa = std.testing.allocator };
    defer box.deinit();
    try std.testing.expect(leaves(.{}, &box)); // not the recipe's checkpoint
    try std.testing.expect(leaves(.{ .glm = true }, &box)); // the default: the prompt stops at its next chunk
    try std.testing.expect(!leaves(.{ .glm = true, .prompt_stop = false }, &box)); // filling: it fills to its end
    box.prefilled_ns = 1;
    try std.testing.expect(leaves(.{ .glm = true, .prompt_stop = false }, &box)); // decoding: the reply stops
}
