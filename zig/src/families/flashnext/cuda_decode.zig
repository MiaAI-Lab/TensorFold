//! One sequence's decode loops (Python qwen4_exp/cuda/decode.py serial_decode, mtp_decode, draft): verify the
//! pending token and the MTP head's drafts in one window, keep the rows up to the first draft the target did not
//! draw, chain the next drafts from the kept rows. Drafts only propose: every kept token is the target's own keyed
//! draw at its position, so drafted output equals serial output token for token.
//!
//! Generic over the engine `E`, which supplies (all on one sequence, Python Engine's methods):
//!   pos(e) u64                                     committed rows
//!   forward(e, tokens) !void                        a window's rows at pos .. pos + tokens.len - 1
//!   sample(e, rows, first_position, out) !void      the target's draws of the window's rows
//!   commit(e, rows, keep) !void                     keep the window's first `keep` rows
//!   absorb(e, source, next_tokens) !void            the MTP cache takes kept rows (it drops last chain's drafts)
//!   chain(e, token, from_row) !void                 one more MTP step on a draft, from the head's output row
//!   sampleDraft(e, position) !Draw                  the head's draw at `position` and its temperature-1 probability
//!   drawDraft(e, position) !u32                     the head's draw without a probability (confidence 0)
//!   isEos(e, token) bool
const std = @import("std");

pub const Draw = struct { token: u32, prob: f64 };

/// Where `absorb` reads the main model's streams: the prefill's last row, or the last window's kept rows.
pub const Source = enum { prefill_last, window };

pub const Options = struct {
    count: usize, // tokens to return, the pending one included
    depth: usize = 6, // MTP drafts a round at most (Python DEPTH)
    confidence: f64 = 0.70, // a chain stops before a later draft under this probability (CONFIDENCE)
    stop_eos: bool = false,
};

pub const Result = struct {
    tokens: std.ArrayList(u32) = .empty,
    rounds: usize = 0,
    drafted: usize = 0,
    accepted: usize = 0,

    pub fn deinit(r: *Result, gpa: std.mem.Allocator) void {
        r.tokens.deinit(gpa);
    }
};

/// One token a step through the same kernels and sampler; `pending` is the prompt's first draw.
pub fn serial(comptime E: type, e: *E, gpa: std.mem.Allocator, pending: u32, o: Options) !Result {
    var r: Result = .{};
    errdefer r.deinit(gpa);
    try r.tokens.append(gpa, pending);
    var one: [1]u32 = undefined;
    while (r.tokens.items.len < o.count and !(o.stop_eos and e.isEos(last(r)))) {
        try e.forward(&.{last(r)});
        try e.sample(1, e.pos() + 1, &one);
        try e.commit(1, 1);
        try r.tokens.append(gpa, one[0]);
        r.rounds += 1;
    }
    return r;
}

fn last(r: Result) u32 {
    return r.tokens.items[r.tokens.items.len - 1];
}

/// The chain's stop rule from a `confidence` value: c > 0, each later draft only at or above c (Python's); c < 0, a
/// later draft j only while p1 * ... * pj >= -c (the running product: decode D4 after X1's measurements,
/// research/X1-drafts.md 3.4); 0, every draft to the depth. The first draft is always proposed. Drafts only: the
/// kept tokens are the target's whatever the rule.
pub const Stop = struct {
    c: f64,
    product: f64 = 1.0,

    /// Whether draft j (0-based) with temperature-1 probability `p` is proposed, and whether the chain may go on.
    pub fn take(s: *Stop, j: usize, p: f64) struct { propose: bool, more: bool } {
        if (s.c > 0) {
            const low = p < s.c;
            if (low and j > 0) return .{ .propose = false, .more = false };
            return .{ .propose = true, .more = !low };
        }
        if (s.c < 0) {
            s.product *= p;
            // the first draft always goes; below the line it ends the chain (no later product can reach it)
            if (s.product < -s.c) return .{ .propose = j == 0, .more = false };
        }
        return .{ .propose = true, .more = true };
    }
};

/// Absorb kept rows, then chain up to `count` drafts under the stop rule (`Stop`; Python `draft` for c > 0). Writes
/// the drafts into `out`; returns how many.
pub fn draft(comptime E: type, e: *E, source: Source, next_tokens: []const u32, position: u64, count: usize, confidence: f64, out: []u32) !usize {
    try e.absorb(source, next_tokens);
    var n: usize = 0;
    var from_row: usize = next_tokens.len - 1; // the first chain step reads the absorb's last row, later ones row 0
    var stop: Stop = .{ .c = confidence };
    for (0..count) |j| {
        var d: u32 = undefined;
        var more = true;
        if (confidence != 0) {
            const got = try e.sampleDraft(position + j);
            d = got.token;
            const t = stop.take(j, got.prob);
            if (!t.propose) break;
            more = t.more;
        } else d = try e.drawDraft(position + j);
        out[n] = d;
        n += 1;
        if (!more) break;
        if (j + 1 < count) {
            try e.chain(d, from_row);
            from_row = 0;
        }
    }
    return n;
}

/// Verify the pending token and its drafts each round, commit the rows before the first mismatch, chain the next
/// drafts from them (Python `mtp_decode`). `max_rows` bounds a window (depth + 1).
pub fn drafted(comptime E: type, e: *E, gpa: std.mem.Allocator, pending: u32, o: Options) !Result {
    var r: Result = .{};
    errdefer r.deinit(gpa);
    try r.tokens.append(gpa, pending);
    var window: [16]u32 = undefined;
    var sampled: [16]u32 = undefined;
    const depth = @min(o.depth, window.len - 1);
    // a round of d drafts adds up to d + 1 tokens: at most `left - 1` drafts when `left` tokens remain (a reply's last
    // round verifies no row past its end)
    var drafts_n = try draft(E, e, .prefill_last, &.{pending}, e.pos() + 1, @min(depth, (o.count -| r.tokens.items.len) -| 1), o.confidence, window[1..]);
    var unabsorbed: ?usize = null; // the last round's kept rows, not yet in the MTP cache
    while (r.tokens.items.len < o.count and !(o.stop_eos and e.isEos(last(r)))) {
        window[0] = last(r);
        const rows = 1 + drafts_n;
        try e.forward(window[0..rows]);
        try e.sample(rows, e.pos() + 1, sampled[0..rows]);
        var keep: usize = 1;
        for (window[1..rows], 0..) |d, i| {
            if (sampled[i] != d or (o.stop_eos and e.isEos(sampled[i]))) break;
            keep += 1;
        }
        try e.commit(rows, keep);
        unabsorbed = keep;
        r.rounds += 1;
        r.drafted += drafts_n;
        r.accepted += keep - 1;
        try r.tokens.appendSlice(gpa, sampled[0..keep]);
        if (r.tokens.items.len >= o.count or (o.stop_eos and e.isEos(last(r)))) break;
        // the kept rows absorbed every round (n 0: the head takes them and drafts nothing), so the MTP cache never
        // skips a block before the last
        const n = @min(depth, (o.count - r.tokens.items.len) -| 1);
        drafts_n = try draft(E, e, .window, sampled[0..keep], e.pos() + 1, n, o.confidence, window[1..]);
        unabsorbed = null;
    }
    // the MTP cache takes the last kept rows: it then covers the whole sequence
    if (unabsorbed) |keep| try e.absorb(.window, sampled[0..keep]);
    r.tokens.shrinkRetainingCapacity(@min(r.tokens.items.len, o.count));
    return r;
}

/// A toy engine for the loops' tests: the "target" draws a token from a hash of the committed sequence and the row's
/// position (keyed by position, as the real sampler), the "head" guesses it right unless a hash says otherwise.
const Toy = struct {
    gpa: std.mem.Allocator,
    seq: std.ArrayList(u32) = .empty, // committed tokens (the cache)
    window: [16]u32 = undefined,
    rows: usize = 0,
    head_seq: std.ArrayList(u32) = .empty, // what the head has absorbed and chained
    head_drafted: usize = 0,
    head_err: u64,
    eos: u32 = 7,

    fn next(seq: []const u32, extra: []const u32) u32 {
        var h = std.hash.Wyhash.init(11);
        h.update(std.mem.sliceAsBytes(seq));
        h.update(std.mem.sliceAsBytes(extra));
        return @intCast(h.final() % 50);
    }
    pub fn pos(t: *Toy) u64 {
        return t.seq.items.len;
    }
    pub fn forward(t: *Toy, tokens: []const u32) !void {
        @memcpy(t.window[0..tokens.len], tokens);
        t.rows = tokens.len;
    }
    pub fn sample(t: *Toy, rows: usize, first: u64, out: []u32) !void {
        std.debug.assert(first == t.seq.items.len + 1);
        for (0..rows) |i| out[i] = next(t.seq.items, t.window[0 .. i + 1]);
    }
    pub fn commit(t: *Toy, rows: usize, keep: usize) !void {
        std.debug.assert(keep >= 1 and keep <= rows);
        try t.seq.appendSlice(t.gpa, t.window[0..keep]);
    }
    pub fn absorb(t: *Toy, _: Source, next_tokens: []const u32) !void {
        t.head_seq.shrinkRetainingCapacity(t.head_seq.items.len - t.head_drafted);
        t.head_drafted = 0;
        t.head_seq.clearRetainingCapacity();
        try t.head_seq.appendSlice(t.gpa, t.seq.items);
        _ = next_tokens;
    }
    pub fn chain(t: *Toy, token: u32, _: usize) !void {
        try t.head_seq.append(t.gpa, token);
        t.head_drafted += 1;
    }
    fn guess(t: *Toy, position: u64) Draw {
        // the head sees the committed sequence plus its chain; the target's next token after them, maybe wrong
        const base = t.head_seq.items;
        const pending = t.window[0..0];
        _ = pending;
        const want = next(base[0..@min(base.len, t.seq.items.len)], base[@min(base.len, t.seq.items.len)..]);
        var h = std.hash.Wyhash.init(t.head_err);
        h.update(std.mem.asBytes(&position));
        const r = h.final() % 10;
        return if (r < 7) .{ .token = want, .prob = 0.9 } else .{ .token = (want + 1) % 50, .prob = if (r == 9) 0.5 else 0.8 };
    }
    pub fn sampleDraft(t: *Toy, position: u64) !Draw {
        return t.guess(position);
    }
    pub fn drawDraft(t: *Toy, position: u64) !u32 {
        return t.guess(position).token;
    }
    pub fn isEos(t: *Toy, token: u32) bool {
        return token == t.eos;
    }
    fn deinit(t: *Toy) void {
        t.seq.deinit(t.gpa);
        t.head_seq.deinit(t.gpa);
    }
};

test "the stop rules: per draft, running product, none" {
    var a: Stop = .{ .c = 0.7 };
    try std.testing.expect(a.take(0, 0.5).propose and !a.take(0, 0.5).more);
    try std.testing.expect(!a.take(1, 0.5).propose);
    var low: Stop = .{ .c = -0.4 };
    const first = low.take(0, 0.3); // a weak first draft: proposed, and the chain ends there
    try std.testing.expect(first.propose and !first.more);
    var b: Stop = .{ .c = -0.4 };
    try std.testing.expect(b.take(0, 0.9).more); // 0.9
    try std.testing.expect(b.take(1, 0.8).propose); // 0.72
    try std.testing.expect(b.take(2, 0.6).propose); // 0.432
    try std.testing.expect(!b.take(3, 0.9).propose); // 0.389
    var c: Stop = .{ .c = 0 };
    try std.testing.expect(c.take(5, 0.01).propose);
}

test "drafted decoding equals serial decoding at every depth and confidence, with and without EOS stops" {
    const gpa = std.testing.allocator;
    for ([_]u64{ 1, 2, 3, 4, 5 }) |seed| for ([_]usize{ 1, 2, 4, 6, 15 }) |depth| for ([_]f64{ 0.0, 0.7, 0.85, -0.4, -0.1 }) |conf| for ([_]bool{ false, true }) |stop| {
        var a: Toy = .{ .gpa = gpa, .head_err = seed };
        defer a.deinit();
        try a.seq.appendSlice(gpa, &.{ 3, 1, 4, 1, 5 }); // the prompt
        const pending = Toy.next(a.seq.items, &.{});
        var b: Toy = .{ .gpa = gpa, .head_err = seed };
        defer b.deinit();
        try b.seq.appendSlice(gpa, a.seq.items);
        const o: Options = .{ .count = 40, .depth = depth, .confidence = conf, .stop_eos = stop };
        var s = try serial(Toy, &a, gpa, pending, o);
        defer s.deinit(gpa);
        var d = try drafted(Toy, &b, gpa, pending, o);
        defer d.deinit(gpa);
        try std.testing.expectEqualSlices(u32, s.tokens.items, d.tokens.items);
        try std.testing.expect(d.rounds <= s.rounds);
    };
}
