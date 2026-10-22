//! One stream (Python LaneStream) and the round loop's state for it (Python's dicts keyed by stream id).
const std = @import("std");
const grammar = @import("grammar.zig");
const shape = @import("shape.zig");
const plan_lanes = @import("plan_lanes.zig");
const Allocator = std.mem.Allocator;
const Proposer = @import("proposer.zig").Proposer;
const Sampling = @import("sampling.zig").Sampling;
const State = @import("depth.zig").State;
const Row = @import("logprob.zig").Row;

pub const Reason = enum {
    none,
    stop,
    length,
    @"error",
    cancelled,

    pub fn name(r: Reason) []const u8 {
        return if (r == .none) "" else @tagName(r);
    }
};

pub const Mode = enum { pipe, drain, verify, exit };

/// Which rule closes a collapsed think block: the Mac server's short exact cycle, or the recipe's (loop_recipe.zig).
pub const LoopRule = enum { mac, recipe };

pub const loop_max_period: usize = 8;
pub const loop_min_run: usize = 256;
pub const loop_warm_in: usize = 64;

/// Return the shortest exact cycle at the end of a combined token prefix.
fn cycleAt(head: []const u32, tail: []const u32, n: usize) ?u32 {
    if (n < loop_min_run + 1) return null;
    for (1..loop_max_period + 1) |period| {
        if (n < loop_min_run + period) break;
        if (n - loop_min_run - period < loop_warm_in) break;
        var matches = true;
        for (1..loop_min_run + 1) |k| {
            const right = n - k;
            const left = right - period;
            const right_token = if (right < head.len) head[right] else tail[right - head.len];
            const left_token = if (left < head.len) head[left] else tail[left - head.len];
            if (right_token != left_token) {
                matches = false;
                break;
            }
        }
        if (matches) return @intCast(period);
    }
    return null;
}

fn firedAt(rule: LoopRule, head: []const u32, tail: []const u32, n: usize) ?u32 {
    return switch (rule) {
        .mac => cycleAt(head, tail, n),
        .recipe => @import("loop_recipe.zig").collapsedAt(head, tail, n),
    };
}

/// Return the first token index in a candidate batch that would fire the guard.
pub fn loopCut(s: *const Stream, tokens: []const u32) ?usize {
    if (!s.loop_guard or !s.think_open or s.think_end < 0 or s.loop_period != null) return null;
    for (tokens, 0..) |token, i| {
        if (@as(i64, token) == s.think_end) return null;
        if (firedAt(s.loop_rule, s.emitted(), tokens, s.emitted().len + i + 1) != null) return i;
    }
    return null;
}

/// Return the detected period at the end of committed tokens, or null.
pub fn detectLoop(emitted: []const u32) ?u32 {
    return cycleAt(emitted, &.{}, emitted.len);
}

fn detectWith(s: *const Stream) ?u32 {
    return firedAt(s.loop_rule, s.emitted(), &.{}, s.emitted().len);
}

/// tool_choice "required" or a named function: outside a think block the reply opens a call. The first token that
/// is neither the opener, whitespace, nor a think-block mark becomes the opener (Python call_gate.CallGate, answer
/// phase; the call's name is the model's own).
pub const CallSpec = struct {
    opener: u32,
    think_open: i64 = -1,
    think_end: i64 = -1,
    blank: []const u32 = &.{},
};

pub const Gate = struct {
    spec: CallSpec,
    armed: bool,
    done: bool = false,

    /// Armed unless the prompt leaves a think block open (its last opener after its last end).
    pub fn init(spec: CallSpec, prompt: []const u32) Gate {
        var last_open: isize = -1;
        var last_end: isize = -1;
        for (prompt, 0..) |t, i| {
            if (@as(i64, t) == spec.think_open) last_open = @intCast(i);
            if (@as(i64, t) == spec.think_end) last_end = @intCast(i);
        }
        const held = spec.think_open >= 0 and spec.think_end >= 0 and last_open > last_end;
        return .{ .spec = spec, .armed = !held };
    }

    fn isBlank(g: Gate, t: u32) bool {
        return std.sort.binarySearch(u32, g.spec.blank, t, struct {
            fn order(key: u32, item: u32) std.math.Order {
                return std.math.order(key, item);
            }
        }.order) != null;
    }

    /// The state after `t`, and whether `t` breaks the call (the opener replaces it).
    fn step(g: Gate, t: u32) struct { g: Gate, breaks: bool } {
        var n = g;
        if (n.done) return .{ .g = n, .breaks = false };
        if (!n.armed) {
            n.armed = @as(i64, t) == n.spec.think_end;
        } else if (@as(i64, t) == n.spec.think_open) {
            n.armed = false;
        } else if (t == n.spec.opener) {
            n.done = true;
        } else if (!g.isBlank(t)) {
            n.done = true;
            return .{ .g = n, .breaks = true };
        }
        return .{ .g = n, .breaks = false };
    }

    /// The index in the next committed `tokens` that the opener replaces.
    pub fn cut(g: Gate, tokens: []const u32) ?usize {
        var state = g;
        for (tokens, 0..) |t, i| {
            const r = state.step(t);
            if (r.breaks) return i;
            state = r.g;
        }
        return null;
    }

    pub fn observe(g: *Gate, t: u32) void {
        g.* = g.step(t).g;
    }
};

/// A stop-string check over the emitted tokens (the server's StopPolicy decodes their tail).
pub const StopCheck = struct {
    ptr: *anyopaque,
    check: *const fn (ptr: *anyopaque, emitted: []const u32) bool,
};

/// The host's answer to whether the stream's request was cancelled, asked between prompt chunks.
pub const CancelCheck = struct {
    ptr: *anyopaque,
    check: *const fn (ptr: *anyopaque) bool,
};

/// A kept prompt state the backend restores before its prompt pass (core/prompt_cache.zig), and where it keeps new ones.
pub const Reuse = struct {
    saved: ?*anyopaque = null, // the backend's copy of the state after `at` prompt tokens (null: start at 0)
    at: u32 = 0,
    marks: []const u32 = &.{}, // ascending: the pass cuts a chunk at each and calls `hook` there
    hook: ?MarkHook = null,
};

/// Called between prompt chunks, the pass standing at `at` (the GPU done with the chunk that ends there).
pub const MarkHook = struct {
    ptr: *anyopaque,
    at: *const fn (ptr: *anyopaque, s: *Stream, at: u32) void,
};

/// Drafts the backend holds for the stream's next round; a tree keeps its tokens and parents on the host.
pub const Held = struct {
    count: u32,
    tokens: ?[]u32 = null,
    parents: ?[]i32 = null,
};

/// One picture or clip for the vision tower: a uint8 canvas [frames, 3, h, w] (CHW, frames 1 for a picture, which the
/// tower pairs with itself; an even count for a clip). h and w are multiples of 28.
pub const MediaItem = struct {
    video: bool = false,
    frames: u32 = 1,
    h: u32,
    w: u32,
    data: []const u8,

    /// The feature rows the tower gives it: one per 28x28 block of each frame pair.
    pub fn rows(it: MediaItem) usize {
        const pairs: usize = if (it.video) it.frames / 2 else 1;
        return pairs * (it.h / 28) * (it.w / 28);
    }
};

/// A prompt's pictures and clips (GLM vision): the prompt rows the tower's features replace (ascending), and either
/// the features already encoded (bf16 [rows.len, width]) or the canvases for the engine's own tower. A prompt with
/// media is neither kept nor resumed, and prefills alone.
pub const Media = struct {
    rows: []const u32,
    width: u32 = 4096,
    /// host bf16 features [rows.len, width] (an out-of-process tower); empty: encode `items` on the engine's tower
    features: []const u8 = &.{},
    items: []const MediaItem = &.{},
    /// a following rank's copy (two ranks): the feature rows' count only; rows and features arrive by rank 0's broadcast
    follower_rows: u32 = 0,

    pub fn rowCount(m: *const Media) usize {
        return if (m.follower_rows > 0) m.follower_rows else m.rows.len;
    }

    pub fn featureBytes(m: *const Media) usize {
        return m.rowCount() * @as(usize, m.width) * 2;
    }

    pub fn check(m: *const Media, prompt_len: usize) error{BadMedia}!void {
        if (m.follower_rows > 0) {
            if (m.rows.len != 0 or m.features.len != 0 or m.items.len != 0 or m.width == 0 or m.follower_rows > prompt_len) return error.BadMedia;
            return;
        }
        if (m.rows.len == 0 or m.width == 0) return error.BadMedia;
        if (m.features.len != 0 and m.features.len != m.featureBytes()) return error.BadMedia;
        if (m.features.len == 0) {
            var n: usize = 0;
            for (m.items) |it| {
                if (it.h == 0 or it.w == 0 or it.h % 28 != 0 or it.w % 28 != 0 or (it.video and (it.frames == 0 or it.frames % 2 != 0)) or (!it.video and it.frames != 1)) return error.BadMedia;
                if (it.data.len != @as(usize, it.frames) * 3 * it.h * it.w) return error.BadMedia;
                n += it.rows();
            }
            if (n != m.rows.len) return error.BadMedia;
        }
        var last: ?u32 = null;
        for (m.rows) |r| {
            if (r >= prompt_len or (last != null and r <= last.?)) return error.BadMedia;
            last = r;
        }
    }
};

pub const Spec = struct {
    id: []const u8,
    prompt: []const u32,
    max_new: u32,
    eos: []const u32 = &.{},
    sampling: ?Sampling = null,
    drafts: bool = true,
    proposer: ?Proposer = null,
    stop_check: ?StopCheck = null,
    cancel_check: ?CancelCheck = null,
    yield_check: ?CancelCheck = null, // a background prompt pass: real work waits, so it keeps what it has and stops
    think_budget: u32 = 0,
    think_close: []const u32 = &.{},
    think_end: i64 = -1,
    think_open: ?bool = null, // null: open when a budget is set (the server's rule)
    loop_guard: bool = false,
    loop_rule: LoopRule = .mac,
    chunks: []const u32 = &.{}, // where prefill chunks start after 0 (Python's PrefillPlan); empty: the backend's step
    history_len: u32 = 0, // prompt prefix lengths a backend may keep for a later turn: the rendered history,
    shared_prefixes: []const u32 = &.{}, // then shared system blocks
    reuse: Reuse = .{},
    decide: ?u32 = null, // the prompt row where living lessons decide whether they act on the reply
    logprobs: ?u8 = null, // the target's log probabilities for each committed token, with this many best tokens
    grammar: ?*grammar.Constraint = null, // structured output: the reply stays in it (the creator frees it)
    call: ?CallSpec = null, // tool_choice required / a named function
    decode_spans: []const [2]u32 = &.{}, // prompt rows an earlier reply decoded (Request.decode_spans), for backends that keep their bits
    media: ?*const Media = null,
};

pub const Stream = struct {
    id: []const u8,
    prompt_len: usize,
    max_new: u32,
    eos: []const u32,
    sampling: ?Sampling,
    drafts: bool,
    proposer: ?Proposer,
    stop_check: ?StopCheck,
    cancel_check: ?CancelCheck,
    yield_check: ?CancelCheck,
    think_budget: u32,
    think_close: []const u32,
    think_end: i64,
    think_open: bool,
    loop_guard: bool,
    loop_rule: LoopRule = .mac,
    loop_period: ?u32 = null,
    /// Structured output: the first token's allowed bits a driver hands the backend (rank 1 of a pair has no grammar of
    /// its own); empty: the backend reads the stream's grammar.
    first_mask: []const u32 = &.{},
    chunks: []const u32,
    history_len: u32,
    shared_prefixes: []const u32,
    logprobs: ?u8 = null,
    grammar: ?*grammar.Constraint = null, // follows each committed token; its masks keep the draws inside it
    gate: ?Gate = null,
    problem: []const u8 = "", // why a stream finished with `error` (static text)
    rows: std.ArrayList(Row) = .empty, // with `logprobs`: one a token of emitted(), in order
    reuse: Reuse = .{},
    decode_spans: []const [2]u32 = &.{},
    decide: ?u32 = null,
    /// the prompt's pictures and clips (null: text); valid until the stream is released
    media: ?*const Media = null,
    cached: u32 = 0, // prompt tokens the backend restored from `reuse` (its prompt pass started there)
    reuse_failed: bool = false, // the backend's restore of `reuse` failed: it prefilled from 0
    context: std.ArrayList(u32) = .empty,
    pending: ?u32 = null,
    force: std.ArrayList(u32) = .empty,
    cache_len: u64 = 0,
    finished: bool = false,
    reason: Reason = .none,
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    branch_rows: u64 = 0, // suffix-match branch rows verified (lanes/fill.zig)
    branch_accepted: u64 = 0, // of those, rows on the accepted path
    graft_hit: f64 = 1, // recent share of grafted rows kept; a read of the chain that grafted none counts as a miss
    odds: ?shape.Odds = null, // how often the target took the head's rank-r lane at each depth (head trees)
    lanes: ?shape.Shape = null, // the shape of the held tree drafts
    landed: [shape.max_depth][shape.ranks][2]u32 = @splat(@splat(.{ 0, 0 })), // tree lanes verified and taken
    picks: [plan_lanes.sizes.len + 1]u32 = @splat(0), // the planner's picks: each tree size, then chain + grafts
    graft_room: ?u32 = null, // grafted rows the planner left this round's window (null: no planner)
    held_levels: u32 = 0, // head levels drafted for the held window
    join_draft: bool = false, // its first token is out; its first drafts come with the next round's (one batch)
    reprefill: bool = false, // a call-gate fix: prefill the prompt plus the reply again (Python generate_gated)
    round_over: ?f64 = null, // the stream's round time beyond its window and head levels (ms, recent): chains
    tree_over: ?f64 = null, // and trees
    measured: [plan_lanes.sizes.len + 1]plan_lanes.Measured = @splat(.{}), // each pick's round times past its table price
    priced: ?struct { choice: usize, ms: f64 } = null, // the pick that shaped the held window, at its table price
    min_rows: u32 = 0,
    paused: bool = false,
    depth: ?State = null,
    mode: ?Mode = null,
    next: ?Held = null,
    inflight: ?u64 = null,
    served: ?u64 = null,
    granted: ?u32 = null,
    copy_width: ?u32 = null,

    pub fn init(gpa: Allocator, spec: Spec) !Stream {
        var s: Stream = .{
            .id = spec.id,
            .prompt_len = spec.prompt.len,
            .max_new = spec.max_new,
            .eos = spec.eos,
            .sampling = spec.sampling,
            .drafts = spec.drafts,
            .proposer = if (spec.drafts) spec.proposer else null,
            .stop_check = spec.stop_check,
            .cancel_check = spec.cancel_check,
            .yield_check = spec.yield_check,
            .think_budget = spec.think_budget,
            .think_close = spec.think_close,
            .think_end = spec.think_end,
            .think_open = spec.think_open orelse (spec.think_budget > 0 or (spec.loop_guard and spec.think_end >= 0)),
            .loop_guard = spec.loop_guard,
            .loop_rule = spec.loop_rule,
            .chunks = spec.chunks,
            .history_len = spec.history_len,
            .shared_prefixes = spec.shared_prefixes,
            .reuse = spec.reuse,
            .decide = spec.decide,
            .logprobs = spec.logprobs,
            .grammar = spec.grammar,
            .gate = if (spec.call) |c| Gate.init(c, spec.prompt) else null,
            .decode_spans = spec.decode_spans,
            .media = spec.media,
        };
        // under a grammar a forced close ends at the think end: the grammar takes the reply from the next token
        if (spec.grammar) |g| if (g.after) |end| if (std.mem.indexOfScalar(u32, s.think_close, end)) |at| {
            s.think_close = s.think_close[0 .. at + 1];
        };
        try s.context.appendSlice(gpa, spec.prompt);
        return s;
    }

    pub fn deinit(s: *Stream, gpa: Allocator) void {
        s.context.deinit(gpa);
        s.rows.deinit(gpa);
        s.force.deinit(gpa);
        s.dropRoundState(gpa);
    }

    /// Forget the round loop's state (Python `_release_stream_state`).
    pub fn dropRoundState(s: *Stream, gpa: Allocator) void {
        if (s.depth) |*d| d.deinit(gpa);
        s.dropHeld(gpa);
        s.depth = null;
        s.mode = null;
        s.inflight = null;
        s.served = null;
        s.granted = null;
        s.copy_width = null;
    }

    pub fn dropHeld(s: *Stream, gpa: Allocator) void {
        if (s.next) |h| {
            if (h.tokens) |t| gpa.free(t);
            if (h.parents) |p| gpa.free(p);
        }
        s.next = null;
        if (s.lanes) |l| l.deinit(gpa);
        s.lanes = null;
    }

    /// The committed reply becomes the prompt's tail. Python's `generate_gated` stops at a call-gate fix and
    /// prefills `prompt + reply` so the next token is a prefill sample, not a decode of the replaced token.
    pub fn absorbIntoPrompt(s: *Stream) !void {
        const extra = s.emitted().len;
        if (extra == 0 or extra > s.max_new) return error.GateRestart;
        s.prompt_len += extra;
        s.max_new -= @intCast(extra);
        s.pending = null;
        s.cache_len = 0;
        s.reprefill = false;
    }

    pub fn prompt(s: *const Stream) []const u32 {
        return s.context.items[0..s.prompt_len];
    }

    pub fn emitted(s: *const Stream) []const u32 {
        return s.context.items[s.prompt_len..];
    }

    pub fn budgetLeft(s: *const Stream) i64 {
        return @as(i64, s.max_new) - @as(i64, @intCast(s.emitted().len));
    }

    /// Whether the request was cancelled (false with no hook); a prompt pass asks before each chunk.
    pub fn isCancelled(s: *const Stream) bool {
        const c = s.cancel_check orelse return false;
        return c.check(c.ptr);
    }

    /// Whether a background prompt pass should step aside now (false with no hook); asked before each chunk.
    pub fn yields(s: *const Stream) bool {
        const c = s.yield_check orelse return false;
        return c.check(c.ptr);
    }

    fn budgetActive(s: *const Stream) bool {
        return s.think_open and s.think_budget > 0 and s.think_close.len > 0;
    }

    /// The index the thinking budget replaces with `think_close[0]`, or null if the block closed first.
    pub fn thinkCut(s: *const Stream, tokens: anytype) ?usize {
        if (!s.budgetActive()) return null;
        const done = s.emitted().len;
        for (tokens, 0..) |t, i| {
            if (done + i + 1 >= s.think_budget) return i;
            if (@as(i64, t) == s.think_end) return null;
        }
        return null;
    }

    /// Python `think_cut([-1]) == 0`: the next position is the budget's cut.
    pub fn cutsNext(s: *const Stream) bool {
        return s.budgetActive() and s.emitted().len + 1 >= s.think_budget;
    }

    /// Begin the thinking budget's close: its first token now, the rest forced after it.
    pub fn startClose(s: *Stream, gpa: Allocator) !u32 {
        s.think_open = false;
        s.force.clearRetainingCapacity();
        try s.force.appendSlice(gpa, s.think_close[1..]);
        return s.think_close[0];
    }

    /// Tokens a round may commit before the length limit or the thinking budget's cut.
    pub fn draftRoom(s: *const Stream) i64 {
        var room = s.budgetLeft();
        if (s.budgetActive()) room = @min(room, @as(i64, s.think_budget) - @as(i64, @intCast(s.emitted().len)));
        return room;
    }

    pub fn isEos(s: *const Stream, token: u32) bool {
        return std.mem.indexOfScalar(u32, s.eos, token) != null;
    }

    /// Each held tree lane whose parent row the round reached: the target took it or not (its depth and rank).
    pub fn observeLanes(s: *Stream, l: *const shape.Shape, path: []const u32) void {
        var on: [256]bool = @splat(false);
        for (path) |r| if (r < on.len) {
            on[r] = true;
        };
        for (l.parents, l.depths, l.ranks, 0..) |par, d, r, i| {
            const parent_row: usize = if (par < 0) 0 else 1 + @as(usize, @intCast(par));
            if (!on[parent_row]) continue;
            const took = on[1 + i];
            s.odds.?.observe(d, r, took);
            if (d < shape.max_depth and r < shape.ranks) {
                s.landed[d][r][0] += 1;
                s.landed[d][r][1] += @intFromBool(took);
            }
        }
    }

    /// Append tokens, and with `logprobs` their rows, until the stream finishes; how many landed.
    pub fn commit(s: *Stream, gpa: Allocator, tokens: []const u32, rows: []const Row) !usize {
        if (rows.len != if (s.logprobs != null) tokens.len else 0) return error.LogprobRowsMismatch;
        var landed: usize = 0;
        for (tokens, 0..) |t, i| {
            if (s.finished) break;
            try s.context.append(gpa, t);
            if (rows.len > 0) try s.rows.append(gpa, rows[i]);
            landed += 1;
            if (@as(i64, t) == s.think_end) s.think_open = false;
            if (s.gate) |*g| g.observe(t);
            if (s.grammar) |g| {
                if (!try g.follow(t)) {
                    s.finished = true;
                    s.reason = .@"error";
                    s.problem = "the reply's grammar rejected a chosen token";
                    break;
                }
                if (g.finished() and !s.isEos(t)) { // the grammar's stop token ends the reply (eos ignored or not)
                    s.finished = true;
                    s.reason = .stop;
                    break;
                }
            }
            const fire = if (s.loop_guard and s.think_open and s.think_end >= 0 and s.loop_period == null) detectWith(s) else null;
            if (s.isEos(t) or s.stopped()) {
                s.finished = true;
                s.reason = .stop;
            } else if (fire) |period| {
                s.loop_period = period;
                if (s.emitted().len >= s.max_new) {
                    s.finished = true;
                    s.reason = .length;
                } else if (s.think_close.len == 0) {
                    s.finished = true;
                    s.reason = .stop;
                } else {
                    s.think_open = false;
                    s.force.clearRetainingCapacity();
                    try s.force.appendSlice(gpa, s.think_close);
                }
                break;
            } else if (s.emitted().len >= s.max_new) {
                s.finished = true;
                s.reason = .length;
            }
        }
        return landed;
    }

    fn stopped(s: *const Stream) bool {
        const c = s.stop_check orelse return false;
        return c.check(c.ptr, s.emitted());
    }

    /// The call gate still has to see the reply's first non-blank token: rounds are verify windows until then.
    pub fn gateWatching(s: *const Stream) bool {
        const g = s.gate orelse return false;
        return !g.done;
    }

    /// The index in `tokens` (the next committed) the call gate replaces with the opener.
    pub fn gateCut(s: *const Stream, tokens: []const u32) ?usize {
        const g = s.gate orelse return null;
        return g.cut(tokens);
    }

    pub fn popForce(s: *Stream) ?u32 {
        if (s.force.items.len == 0) return null;
        return s.force.orderedRemove(0);
    }
};

test "commit stops at eos and length" {
    const gpa = std.testing.allocator;
    var s = try Stream.init(gpa, .{ .id = "a", .prompt = &.{ 1, 2 }, .max_new = 3, .eos = &.{9} });
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), try s.commit(gpa, &.{ 5, 9, 7 }, &.{}));
    try std.testing.expectEqualSlices(u32, &.{ 5, 9 }, s.emitted());
    try std.testing.expect(s.finished and s.reason == .stop);
}

test "a logprob stream lands a row with each token it lands, and refuses tokens without rows" {
    const gpa = std.testing.allocator;
    var s = try Stream.init(gpa, .{ .id = "a", .prompt = &.{ 1, 2 }, .max_new = 3, .eos = &.{9}, .logprobs = 0 });
    defer s.deinit(gpa);
    try std.testing.expectError(error.LogprobRowsMismatch, s.commit(gpa, &.{5}, &.{}));
    const rows = [_]Row{ .{ .token = 5, .logprob = -0.5 }, .{ .token = 9, .logprob = -1 }, .{ .token = 7, .logprob = -2 } };
    try std.testing.expectEqual(@as(usize, 2), try s.commit(gpa, &.{ 5, 9, 7 }, &rows));
    try std.testing.expectEqual(@as(usize, 2), s.rows.items.len);
    try std.testing.expectEqual(@as(u32, 9), s.rows.items[1].token);
}

test "loop guard stops at the cap even when no cycle fires" {
    const gpa = std.testing.allocator;
    var tokens: [20]u32 = undefined;
    for (&tokens, 0..) |*token, i| token.* = @intCast(100 + i);
    var guarded = try Stream.init(gpa, .{ .id = "guarded", .prompt = &.{1}, .max_new = 10, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true });
    defer guarded.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 10), try guarded.commit(gpa, &tokens, &.{}));
    try std.testing.expect(guarded.finished and guarded.reason == .length);
}

test "loop guard fires, closes thinking, and latches once" {
    const gpa = std.testing.allocator;
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(gpa);
    for (0..loop_warm_in) |i| try tokens.append(gpa, @intCast(1000 + i));
    for (0..257) |_| try tokens.append(gpa, 7);
    var s = try Stream.init(gpa, .{ .id = "loop", .prompt = &.{1}, .max_new = 400, .think_close = &.{ 90, 91, 92 }, .think_end = 91, .loop_guard = true });
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 321), try s.commit(gpa, tokens.items, &.{}));
    try std.testing.expectEqual(@as(u32, 1), s.loop_period.?);
    try std.testing.expect(!s.finished and !s.think_open and s.force.items.len == 3);
    while (s.popForce()) |token| _ = try s.commit(gpa, &.{token}, &.{});
    try std.testing.expect(!s.finished and s.force.items.len == 0);
    _ = try s.commit(gpa, &.{8}, &.{});
    try std.testing.expect(!s.finished and s.loop_period.? == 1);
}

test "call gate: a reply that skips the opener opens it; blanks and think blocks pass" {
    const spec: CallSpec = .{ .opener = 50, .think_open = 60, .think_end = 61, .blank = &.{ 10, 11 } };
    // prompt ends with an empty think block: armed from the first token
    var g = Gate.init(spec, &.{ 1, 60, 61 });
    try std.testing.expect(g.armed);
    try std.testing.expectEqual(@as(?usize, 1), g.cut(&.{ 10, 7, 8 }));
    try std.testing.expectEqual(@as(?usize, null), g.cut(&.{ 10, 11, 50, 7 }));
    g.observe(10);
    g.observe(50);
    try std.testing.expect(g.done);
    try std.testing.expectEqual(@as(?usize, null), g.cut(&.{ 7, 8 }));
    // prompt leaves a think block open: held until its end
    var h = Gate.init(spec, &.{ 1, 60 });
    try std.testing.expect(!h.armed);
    try std.testing.expectEqual(@as(?usize, 2), h.cut(&.{ 7, 61, 8 }));
    h.observe(7);
    h.observe(61);
    try std.testing.expect(h.armed and !h.done);
    // a think block opened in the reply holds the gate again
    try std.testing.expectEqual(@as(?usize, null), (Gate.init(spec, &.{1})).cut(&.{ 60, 7, 8 }));
}

test "a call-gate fix folds the committed opener into the prompt" {
    const gpa = std.testing.allocator;
    var s = try Stream.init(gpa, .{ .id = "g", .prompt = &.{ 1, 2 }, .max_new = 12, .call = .{ .opener = 50 } });
    defer s.deinit(gpa);
    _ = try s.commit(gpa, &.{50}, &.{});
    s.reprefill = true;
    try s.absorbIntoPrompt();
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 50 }, s.prompt());
    try std.testing.expectEqual(@as(usize, 0), s.emitted().len);
    try std.testing.expectEqual(@as(u32, 11), s.max_new);
    try std.testing.expect(!s.reprefill);
}

test "call gate in commit: observe follows committed tokens" {
    const gpa = std.testing.allocator;
    var s = try Stream.init(gpa, .{ .id = "g", .prompt = &.{ 1, 61 }, .max_new = 10, .call = .{ .opener = 50, .think_open = 60, .think_end = 61 } });
    defer s.deinit(gpa);
    try std.testing.expect(s.gateWatching());
    try std.testing.expectEqual(@as(?usize, 0), s.gateCut(&.{9}));
    _ = try s.commit(gpa, &.{50}, &.{});
    try std.testing.expect(!s.gateWatching());
}

test "media rows and canvases must cover the prompt" {
    const pix: [3 * 28 * 56]u8 = @splat(0);
    const item: MediaItem = .{ .h = 28, .w = 56, .data = &pix };
    try std.testing.expectEqual(@as(usize, 2), item.rows());
    var m: Media = .{ .rows = &.{ 1, 2 }, .items = &.{item} };
    try m.check(5);
    try std.testing.expectError(error.BadMedia, m.check(2)); // a row past the prompt
    m.rows = &.{ 2, 1 };
    try std.testing.expectError(error.BadMedia, m.check(5)); // rows out of order
    m.rows = &.{ 1, 2, 3 };
    try std.testing.expectError(error.BadMedia, m.check(5)); // rows the canvases do not give
    const feats: [16]u8 = @splat(0);
    const f: Media = .{ .rows = &.{ 1, 2 }, .width = 4, .features = &feats };
    try f.check(5);
    const follower: Media = .{ .rows = &.{}, .follower_rows = 2 };
    try follower.check(5);
}
