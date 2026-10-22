//! A stream's way into the rounds: its prompt pass whole or a chunk a round, then its first token and drafts.
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const be = @import("backend.zig");
const ev = @import("events.zig");
const win = @import("windows.zig");
const trail = @import("trail.zig");
const Stream = sm.Stream;
const Feed = be.Feed;
const LogRow = @import("logprob.zig").Row;
const f = ev.f;
const str = trail.str;
const int = trail.int;

/// Prefill a stream, draw its first token and ask for its first drafts; it takes part from the next round.
pub fn add(e: *Engine, s: *Stream) !void {
    _ = e.arena.reset(.retain_capacity);
    try trail.event(e, &.{ f("ev", str("add")), f("stream", str(s.id)) });
    e.backend.prefill(s) catch |err| {
        if (err == error.Cancelled) e.backend.release(s); // the host finishes a cancelled stream without the core
        return err;
    };
    try opened(e, s);
}

/// After the prompt pass: the first token drawn and committed, the first drafts asked; rounds from the next.
pub fn opened(e: *Engine, s: *Stream) !void {
    if (s.isCancelled()) { // cancelled in its last chunk: no first token
        e.backend.release(s);
        return error.Cancelled;
    }
    s.context.shrinkRetainingCapacity(s.prompt_len);
    s.rows.clearRetainingCapacity();
    s.pending = null;
    s.cache_len = s.prompt_len;
    const position: u64 = s.prompt_len;
    const drawn = try e.drawFirst(s, position);
    var feed: Feed = .{ .handle = drawn };
    if (try e.forcedNext(s)) |t| feed = .{ .value = t };
    if (s.gateWatching() and feed == .handle) { // the call gate reads the first draw: a reply that skips the opener opens it
        const drawn_token = try e.backend.read(drawn);
        if (s.gateCut(&.{drawn_token}) != null) {
            // The opener replaces the draw. Python's generate_gated then prefills the prompt plus that fix;
            // decoding the opener on this cache samples the next token from a different kernel.
            const value = s.gate.?.spec.opener;
            _ = try s.commit(e.gpa, &.{value}, &.{});
            s.pending = value;
            s.reprefill = !s.finished;
            try trail.resolve(e);
            if (s.finished) {
                try trail.finish(e, s);
                e.release(s);
            }
            return;
        }
        feed = .{ .value = drawn_token };
    }
    var asked: ?u32 = null;
    if (e.cfg.family_mtp and s.drafts and !e.cfg.join_batch) {
        // the head reads the prompt's last row and the first token, and drafts the one after it
        const d: u32 = @intCast(try e.rule.depth(win.who(s)));
        asked = d;
        try e.backend.draft(&.{.{ .stream = s, .follow = &.{}, .first = feed, .rows = null, .start = s.prompt_len, .position = position + 1, .depth = d }});
        s.dropHeld(e.gpa);
        s.next = .{ .count = d };
        if (e.backend.vtable.tree) |tree| if (try tree(e.backend.ptr, s, e.gpa)) |held| {
            s.next = held; // a tree head's first drafts as host tokens, as every later round's
        };
    } else if (!(e.cfg.family_mtp and s.drafts) and e.cfg.pipelined and s.logprobs == null and s.grammar == null and !s.gateWatching()) {
        try e.queueNext(s, feed);
    }
    const value = try e.readFeed(feed);
    if (asked) |d| try trail.event(e, &.{ f("ev", str("draft")), f("stream", str(s.id)), f("depth", int(d)), f("position", int(position + 1)), f("follow", .{ .u32s = &.{value} }), f("rows", .null) });
    if (e.cfg.family_mtp and s.drafts and e.cfg.join_batch) {
        // the first token goes out now; its first drafts come with the next round's other joiners (joinDrafts)
        s.dropHeld(e.gpa);
        s.join_draft = true;
    }
    if (e.log != null) {
        const first = if (feed == .handle) value else try e.backend.read(drawn);
        try trail.event(e, &.{ f("ev", str("first")), f("stream", str(s.id)), f("position", int(position)), f("drawn", int(first)), f("token", int(value)) });
    }
    var first_row: [1]LogRow = undefined;
    if (s.logprobs != null) first_row[0] = (try e.backend.firstRow(s)).forToken(value);
    _ = try s.commit(e.gpa, &.{value}, if (s.logprobs != null) &first_row else &.{});
    s.pending = value;
    try trail.resolve(e);
    if (s.finished) {
        try trail.finish(e, s);
        e.release(s);
        return;
    }
    try e.live.append(e.gpa, s);
}

/// Whether the backend can run a prompt pass a chunk at a time between rounds (`fillStream`).
pub fn fills(e: *const Engine) bool {
    return e.backend.vtable.prefill_step != null;
}

/// One chunk of a stream's prompt pass; true once the stream is opened and joins the next round.
pub fn fillStream(e: *Engine, s: *Stream, first: bool) !bool {
    _ = e.arena.reset(.retain_capacity);
    if (first) try trail.event(e, &.{ f("ev", str("add")), f("stream", str(s.id)) });
    const chunk = e.backend.vtable.prefill_step orelse return error.NoPrefillSteps;
    const done = chunk(e.backend.ptr, s) catch |err| {
        if (err == error.Cancelled) e.backend.release(s); // the host finishes a cancelled stream without the core
        return err;
    };
    if (!done) return false;
    try opened(e, s);
    return true;
}
