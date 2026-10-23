//! Exact prompt reuse: states are keyed by tokens and canonical decoded intervals through their position.
const std = @import("std");
const Allocator = std.mem.Allocator;
const imprint = @import("prompt_imprint.zig");
const modes = @import("cache_modes.zig");
const learned = @import("prompt_learn.zig");

/// A family's copy of one state.
pub const Saved = *anyopaque;

pub fn singleReclaim(_: *anyopaque, dir: [:0]const u8, key: u64) !u64 {
    var buf: [1100]u8 = undefined;
    return @import("lanes").learned_dirs.fileBytes(try std.fmt.bufPrintSentinel(&buf, "{s}/{x:0>16}.bin", .{ dir, key }, 0));
}
pub fn singleForgetChecked(_: *anyopaque, dir: [:0]const u8, key: u64) !void {
    var buf: [1100]u8 = undefined;
    try @import("lanes").learned_dirs.unlink(try std.fmt.bufPrintSentinel(&buf, "{s}/{x:0>16}.bin", .{ dir, key }, 0));
}
/// What a family gives the cache: copies of its state while the prompt pass stands at a chunk end.
pub const Snapshots = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// The most new storage `save` takes at `at` tokens (checked against the budget before any copy).
        bytes: *const fn (ptr: *anyopaque, at: u32) u64,
        /// Copy the live state after `at` prompt tokens into new storage; `owner` is the lane core's stream (null: one stream).
        save: *const fn (ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!Saved,
        /// Make the live state `saved`'s: the next prompt chunk starts at its position.
        restore: *const fn (ptr: *anyopaque, owner: ?*anyopaque, saved: Saved) anyerror!void,
        drop: *const fn (ptr: *anyopaque, saved: Saved) void,
        /// The storage a kept state holds: new storage up to `bytes`, or the spare storage it took (null: `bytes`).
        charged: ?*const fn (ptr: *anyopaque, saved: Saved) u64 = null,
        /// Free storage the family keeps for later saves, inside the budget (null: none).
        spare: ?*const fn (ptr: *anyopaque) u64 = null,
        /// Free spare storage until at most `room` bytes stay spare.
        trim: ?*const fn (ptr: *anyopaque, room: u64) void = null,
        /// Whether a save at `at` would take spare storage (no new storage).
        reuses: ?*const fn (ptr: *anyopaque, at: u32) bool = null,
        /// Write `saved` as learned state `key`, in files of the family's own under `dir` (null: nothing kept on disk).
        write: ?*const fn (ptr: *anyopaque, saved: Saved, dir: [:0]const u8, key: u64) anyerror!void = null,
        /// Learned state `key` of `at` tokens, read back from its files under `dir` into new storage.
        read: ?*const fn (ptr: *anyopaque, dir: [:0]const u8, key: u64, at: u32) anyerror!Saved = null,
        /// Peer disk demand and reservation lifetime; null functions mean a single local rank.
        reclaim: ?*const fn (*anyopaque, [:0]const u8, u64) anyerror!u64 = null,
        peer_other_used: ?*const fn (*anyopaque, u64) anyerror!u64 = null,
        peer_need: ?*const fn (ptr: *anyopaque, saved: Saved, extra: u64) anyerror!u64 = null,
        peer_reclaim: ?*const fn (ptr: *anyopaque, key: u64) anyerror!u64 = null,
        peer_reserve: ?*const fn (ptr: *anyopaque, saved: Saved, key: u64, extra: u64) anyerror!void = null,
        peer_finish: ?*const fn (ptr: *anyopaque, key: u64, success: bool) anyerror!void = null,
        peer_other_next: ?*const fn (*anyopaque, ?u64) anyerror!?u64 = null,
        peer_other_bytes: ?*const fn (*anyopaque, u64) anyerror!u64 = null,
        peer_other_remove: ?*const fn (*anyopaque, u64) anyerror!void = null,
        forget_checked: ?*const fn (*anyopaque, [:0]const u8, u64) anyerror!void = null,
        /// A state leaving memory for room or a cap (the policy lets it go to disk): the family may store it first.
        evict: ?*const fn (ptr: *anyopaque, saved: Saved, tokens: []const u32, at: u32) void = null,
        /// The longest stored state `prompt` extends that is longer than `longer_than` (null: none).
        stored: ?*const fn (ptr: *anyopaque, prompt: []const u32, longer_than: u32, at: *u32, key: *u64, tokens: *[]const u32) bool = null,
        /// Read stored state `key` of `at` tokens into new storage; the store keeps it like a saved one.
        load: ?*const fn (ptr: *anyopaque, key: u64, at: u32) anyerror!Saved = null,
        /// The disk tier's numbers: entries, bytes, writes, write bytes, reads, read bytes, skipped, failed.
        spill_stats: ?*const fn (ptr: *anyopaque, out: *[8]u64) void = null,
        /// Learned state `key`'s files under `dir` removed (the cap needs room, or they no longer read back).
        forget: ?*const fn (ptr: *anyopaque, dir: [:0]const u8, key: u64) void = null,
    };
};

/// A state a family holds on disk (the spill tier): its position and tokens, and the family's key for it.
pub const Stored = struct { at: u32, key: u64, tokens: []const u32 };

/// What a paged store asks of the backend's pages: who holds each, and how many are free.
pub const Pages = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    /// Bytes of one page, over every layer.
    bytes: u64,

    pub const VTable = struct {
        retain: *const fn (ptr: *anyopaque, id: u32) void,
        release: *const fn (ptr: *anyopaque, id: u32) void,
        /// How many hold the page: the store counts as one.
        holders: *const fn (ptr: *anyopaque, id: u32) u32,
        /// Pages free and not promised to a stream.
        available: *const fn (ptr: *anyopaque) usize,
    };
};

/// What a family's states depend on beyond their position.
pub const Rules = struct {
    /// Tokens past the position a state read (Flash Next's MTP head keys row at-1 with token at: 1).
    lookahead: u32 = 0,
    /// Prompt kernels that change arithmetic with a chunk's rows: resume and keep only at the request's chunk starts.
    planned: bool = false,
    /// A planned family's own zero-anchored chunk rows, its starts when a request names none (0: none).
    grid: u32 = 0,
    /// A mark other than the history's is kept only this far from every other mark and the resume point.
    min_gap: u32 = 256,
    /// Shorter prompts keep nothing: below it a mark's extra prompt call costs more than a later turn's reuse saves.
    min_prompt: u32 = 4096,
    /// Replies are prefilled in background passes kept at their end: a pass resumed near its history keeps no mark.
    warm: bool = false,
    /// A paged store (prompt_radix.zig): the tokens a page holds, where its marks and resumes sit (0: not paged).
    page: u32 = 0,
    /// A paged store also keeps the prompt's last whole page, for a later turn whose history the request did not name.
    tail: bool = false,
};

/// What a family's recipe adds to the store's own order (GLM's kept-state policy, multi.py): off by default. The caps
/// are Thomas Wade's (ThomasWadeZ, PR #65), the recency of shared cuts meleesciony's (issue #75), the share of the pool
/// jdecker76's (issue #84), the disk tier Robert Wojciechowski's (wojo, patch 0088): the recipe's authors.
pub const Policy = struct {
    /// Kept states at most (0: no limit): the oldest superseded one goes first (`capVictim`), then the least recently used.
    entries: u32 = 0,
    /// The bytes the kept states hold together (0: only the budget): the states go by the same order, the newest stays.
    bytes_cap: u64 = 0,
    /// One conversation's own states at most (0: no limit), the least recently used of its own first; shared cuts are not counted.
    per_chat: u32 = 0,
    /// The recipe's order and marks: no rewind state, a conversation's shorter states superseded first, a shared cut only by recency.
    recipe: bool = false,
    /// The fewest tokens a shared-prefix state adds past the resume point (0: off): a state is also kept where the prompt
    /// stops sharing a kept one (rounded down to the grid), as engine.shared_points.
    shared_least: u32 = 0,
    /// States leaving memory may go to the family's disk tier, and a lookup may read one back (the spill tier).
    spill: bool = false,
    /// Past this share of the budget (0: never) the states eviction would take next are stored early, one a keep.
    spill_highwater: f64 = 0,
};

/// /health's kept-state view: counts, bytes, and the mix by origin and conversation.
pub const KeptStats = struct {
    prompts: u32 = 0,
    bytes: u64 = 0,
    bytes_cap: u64 = 0,
    own: u32 = 0,
    shared: u32 = 0,
    chats: u32 = 0,
    rows: u64 = 0,
    entries_cap: u32 = 0,
    per_chat: u32 = 0,
    budget: u64 = 0,
    /// [chat, states, longest] of the first chats (by chat id)
    by_chat: [8][3]u64 = @splat(.{ 0, 0, 0 }),
    by_chat_n: u8 = 0,
    spill_on: bool = false,
    spill: [8]u64 = @splat(0),
    hits: u64 = 0,
    misses: u64 = 0,
    evicted: u64 = 0,
    restored: u64 = 0,
    /// Requests that continued a conversation without resuming its own turn state (0111), and what they prefilled.
    chats_continued: u64 = 0,
    chats_continued_prefill_tokens: u64 = 0,
};

pub const Entry = struct {
    tokens: []u32, // the state's tokens and `lookahead` more
    decode_spans: []modes.Span = &.{}, // canonical row arithmetic through at, excluding lookahead
    at: u32,
    saved: Saved,
    bytes: u64,
    born: u32, // the length of the prompt that kept it: a later turn's is longer
    used: u64, // the store's clock at its last keep or resume
    last: []u32, // the last prompt that kept or resumed it
    shared: bool = false, // kept at a shared system block's cut: other conversations resume it, so its own never supersedes it
    chat: u32 = 0, // the conversation (Policy): its request's (`Store.join`), else the state it extends (`chatOf`)
    own: bool = true, // a conversation's own state, not a shared cut
    spilled: bool = false, // the family stored it on disk already (an early write)
};

pub const Counts = struct { hits: u64 = 0, misses: u64 = 0, kept: u64 = 0, evicted: u64 = 0, refused: u64 = 0, failed: u64 = 0, restored: u64 = 0, chats_continued: u64 = 0, chats_continued_prefill_tokens: u64 = 0 };

/// A later prompt continues a conversation when it repeats that conversation's first prompt and this many tokens of the
/// reply to it (0111): ids no other conversation wrote.
pub const chat_match: u32 = 32;
/// A continued request that prefills this many tokens or more is logged.
pub const chat_log_tokens: u32 = 4096;

/// The conversation a request's kept states join (`Store.join`), and whether it continues one without resuming that
/// conversation's own turn state (0111's counted case).
pub const Join = struct {
    chat: u32 = 0, // 0: none named (a keep takes the conversation of the state it extends)
    continued: bool = false,
    /// The matched state's tokens, and where the prompt left them: `back < prior_len` is a rewind before that state.
    prior_len: u32 = 0,
    back: u32 = 0,
    /// What the lookup resumed from.
    source: Source = .nothing,

    pub const Source = enum { nothing, shared, other };
};

/// Where a prompt pass starts and where it stops to keep its state (marks sorted, in `a`).
pub const Plan = struct { from: u32 = 0, marks: []const u32 = &.{} };

/// A paged store's `keep`: whether the mark holds a state, and how many leading pages the stream swaps for the store's.
pub const Kept = struct { held: bool, shared: usize };

/// The entry a backend restores itself (null: the pass starts at 0) and the pass's marks.
pub const Lookup = struct { entry: ?*Entry = null, marks: []const u32 = &.{} };

pub const Store = struct {
    gpa: Allocator,
    family: Snapshots,
    rules: Rules,
    budget: u64,
    held: u64 = 0,
    clock: u64 = 0,
    entries: std.ArrayList(*Entry) = .empty,
    counts: Counts = .{},
    shared_keys: std.ArrayList(u64) = .empty, // the tokens of recently planned shared cuts, hashed (at most SHARED_KEYS)
    imprint: ?*imprint.Imprint = null, // --learn: shared cuts' states on disk, read back by later sessions and servers
    active_rewind_at: u32 = 0, // the last lookup's rewind state (0: none); hosts keep a pass's states before the next lookup
    policy: Policy = .{},
    next_chat: u32 = 1,
    roots: std.AutoHashMapUnmanaged(u32, u32) = .empty, // each conversation's first prompt length (continuedChat)

    const SHARED_KEYS = 64;

    pub fn init(gpa: Allocator, family: Snapshots, rules: Rules, budget: u64) Store {
        return .{ .gpa = gpa, .family = family, .rules = rules, .budget = budget };
    }

    pub fn deinit(s: *Store) void {
        for (s.entries.items) |e| s.free(e);
        s.entries.deinit(s.gpa);
        s.shared_keys.deinit(s.gpa);
        s.roots.deinit(s.gpa);
        s.held = 0;
    }

    fn free(s: *Store, e: *Entry) void {
        s.family.vtable.drop(s.family.ptr, e.saved);
        s.gpa.free(e.tokens);
        s.gpa.free(e.decode_spans);
        s.gpa.free(e.last);
        s.gpa.destroy(e);
    }

    /// A state leaving memory for room or a cap: stored first when the policy allows it and a longer state of its
    /// conversation does not hold it (multi.py `_superseded`).
    fn evict(s: *Store, i: usize, to_disk: bool) void {
        const e = s.entries.items[i];
        if (to_disk and s.policy.spill and !e.spilled) if (s.family.vtable.evict) |f| if (!s.superseded(e)) f(s.family.ptr, e.saved, e.tokens, e.at);
        s.remove(i);
    }

    fn superseded(s: *const Store, e: *const Entry) bool {
        for (s.entries.items) |o| {
            if (o != e and o.chat == e.chat and o.tokens.len > e.tokens.len and std.mem.eql(u32, o.tokens[0..e.tokens.len], e.tokens)) return true;
        }
        return false;
    }

    fn remove(s: *Store, i: usize) void {
        const e = s.entries.orderedRemove(i);
        s.held -= e.bytes;
        s.free(e);
    }

    /// The family's free storage kept for later saves.
    pub fn spare(s: *const Store) u64 {
        return if (s.family.vtable.spare) |f| f(s.family.ptr) else 0;
    }

    /// What the budget leaves past the kept states and the spare storage: a family readies storage only inside it.
    pub fn room(s: *const Store) u64 {
        return s.budget -| s.held -| s.spare();
    }

    /// Whether a state at `at` may start or end a prompt pass: anywhere, or a chunk start for planned families.
    fn usable(s: *const Store, at: u32, starts: []const u32) bool {
        return at > 0 and (!s.rules.planned or s.startAt(starts, at) == at);
    }

    /// The chunk start at or before `at`: the request's, else the family's grid (0 when none).
    fn startAt(s: *const Store, starts: []const u32, at: u32) u32 {
        if (starts.len == 0 and s.rules.grid > 0) return at - at % s.rules.grid;
        return floorStart(starts, at);
    }

    /// The latest kept state this prompt can resume at `at` or under min_gap before it (null: none).
    fn keptNear(s: *const Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span, at: u32) ?u32 {
        var best: ?u32 = null;
        for (s.entries.items) |e| {
            if (e.at > at or at - e.at >= s.rules.min_gap or e.tokens.len > prompt.len or !s.usable(e.at, starts)) continue;
            if (!std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens) or !modes.equal(e.decode_spans, spans, e.at)) continue;
            if (best == null or e.at > best.?) best = e.at;
        }
        return best;
    }

    /// Find a kept token prefix with matching canonical spans through its position, leaving at least one row.
    pub fn find(s: *Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span) ?*Entry {
        var best: ?*Entry = null;
        for (s.entries.items) |e| {
            if (e.tokens.len > prompt.len or e.at >= prompt.len or !s.usable(e.at, starts)) continue;
            if (best != null and e.at <= best.?.at) continue;
            const n = e.tokens.len;
            if (prompt[n - 1] != e.tokens[n - 1] or !std.mem.eql(u32, prompt[0..n], e.tokens)) continue;
            if (!modes.equal(e.decode_spans, spans, e.at)) continue;
            best = e;
        }
        return best;
    }

    /// Restore the longest prefix matching tokens and canonical spans, then plan prompt-cache marks.
    pub fn begin(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32, owner: ?*anyopaque, spans: []const modes.Span) !Plan {
        return s.beginRewind(a, prompt, history_len, 0, shared, starts, owner, spans);
    }

    /// `begin`, also keeping a state before the latest user message (`rewind_len`, 0: none) that its edits resume.
    pub fn beginRewind(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, rewind_len: u32, shared: []const u32, starts: []const u32, owner: ?*anyopaque, spans: []const modes.Span) !Plan {
        const l = try s.lookupRewind(a, prompt, history_len, rewind_len, shared, starts, spans);
        const e = l.entry orelse return .{ .marks = l.marks };
        const ok = if (s.family.vtable.restore(s.family.ptr, owner, e.saved)) |_| true else |err| blk: {
            note("restoring {d} tokens failed ({s}); prefilling from the start", .{ e.at, @errorName(err) });
            break :blk false;
        };
        const from = if (ok) e.at else 0;
        s.resumed(e, prompt, ok);
        s.reserve(prompt, null, l.marks, spans); // restored: the state it resumed may go too (a peer restores before it drops)
        return .{ .from = from, .marks = l.marks };
    }

    /// Look up canonical-span state and marks without restoring; the caller later reports resumed.
    pub fn lookup(s: *Store, a: Allocator, prompt: []const u32, history_len: u32, shared: []const u32, starts: []const u32, spans: []const modes.Span) !Lookup {
        return s.lookupRewind(a, prompt, history_len, 0, shared, starts, spans);
    }

    /// `lookup` plus a rewind state on the starts or grid; a state kept under min_gap before it serves instead.
    pub fn lookupRewind(s: *Store, a: Allocator, prompt: []const u32, history_in: u32, rewind_in: u32, shared_in: []const u32, starts: []const u32, spans: []const modes.Span) !Lookup {
        const rewind_len = if (s.policy.recipe) 0 else rewind_in;
        const history_len = s.recipeHistory(prompt.len, history_in);
        const endpoint = if (s.rules.planned) s.startAt(starts, history_len) else history_len;
        const rewind = if (s.rules.planned) s.startAt(starts, rewind_len -| s.rules.lookahead) else rewind_len -| s.rules.lookahead;
        const endpoint_bytes = if (endpoint > 0 and (endpoint < prompt.len or (s.rules.warm and endpoint == prompt.len)) and endpoint + s.rules.lookahead <= prompt.len) s.family.vtable.bytes(s.family.ptr, endpoint) else 0;
        const rewind_bytes = if (rewind > 0 and rewind < prompt.len and rewind + s.rules.lookahead <= prompt.len and s.usable(rewind, starts)) s.family.vtable.bytes(s.family.ptr, rewind) else 0;
        const priority_bytes = if (endpoint_bytes <= s.budget) endpoint_bytes else 0;
        s.active_rewind_at = if (prompt.len >= s.rules.min_prompt and rewind_bytes > 0 and rewind != endpoint and rewind_bytes <= s.budget - priority_bytes) rewind else 0;
        if (s.active_rewind_at > 0) if (s.keptNear(prompt, starts, spans, s.active_rewind_at)) |at| {
            s.active_rewind_at = at; // a state kept under min_gap before it is the rewind state: no new copy
        };
        const e = s.recall(prompt, starts, spans, s.find(prompt, starts, spans));
        if (e == null) s.counts.misses += 1;
        const from = if (e) |x| x.at else 0;
        var merged: ?[]u32 = null;
        defer if (merged) |m| a.free(m);
        var shared: []const u32 = if (s.policy.recipe and s.policy.shared_least == 0) &.{} else shared_in; // TF_GLM_SHARED_PREFIX=0: no shared points
        if (s.policy.recipe and s.policy.shared_least > 0) if (s.lcpPoint(prompt, from, endpoint, starts)) |p| {
            const m = try a.alloc(u32, shared_in.len + 1);
            @memcpy(m[0..shared_in.len], shared_in);
            m[shared_in.len] = p;
            merged = m;
            shared = m;
        };
        const marks_ = if (s.active_rewind_at == 0)
            try s.fitting(a, try s.marks(a, prompt, from, history_len, shared, starts, if (e) |x| x.last else &.{}))
        else
            try s.fittingRewind(a, try s.fitting(a, try s.marksRewind(a, prompt, from, history_len, s.active_rewind_at, shared, starts, if (e) |x| x.last else &.{})), endpoint, s.active_rewind_at, endpoint_bytes);
        for (shared) |w| { // the shared cuts this pass keeps: their states serve other conversations too
            const at = if (s.rules.planned) s.startAt(starts, w) else w;
            if (std.mem.indexOfScalar(u32, marks_, at) != null) s.noteShared(prompt[0 .. at + s.rules.lookahead]);
        }
        s.reserve(prompt, e, marks_, spans);
        return .{ .entry = e, .marks = marks_ };
    }

    /// Room for every state the pass keeps, made before it: a peer that mirrors the keeps hears of these evictions with the request.
    fn reserve(s: *Store, prompt: []const u32, from: ?*Entry, marks_: []const u32, spans: []const modes.Span) void {
        var need: u64 = 0;
        for (marks_) |at| {
            const n = @as(usize, at) + s.rules.lookahead;
            const exists = for (s.entries.items) |e| {
                if (e.at == at and e.tokens.len == n and n <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..n]) and modes.equal(e.decode_spans, spans, at)) break true;
            } else false;
            if (!exists) need += s.family.vtable.bytes(s.family.ptr, at);
        }
        while (s.held + need > s.budget) {
            const i = s.victimBut(from, prompt) orelse return;
            s.evict(i, true);
            s.counts.evicted += 1;
        }
    }

    /// Every kept state goes after a weight change, and learned states on disk stop being read: old arithmetic.
    pub fn clear(s: *Store) void {
        while (s.entries.items.len > 0) s.remove(s.entries.items.len - 1);
        s.shared_keys.clearRetainingCapacity();
        s.imprint = null;
        s.active_rewind_at = 0;
    }

    /// Forget a failed peer state matching the token prefix and canonical spans through at.
    pub fn forget(s: *Store, prompt: []const u32, at: u32, spans: []const modes.Span) void {
        for (s.entries.items, 0..) |e, i| if (e.at == at and e.tokens.len <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]) and modes.equal(e.decode_spans, spans, at)) {
            s.counts.failed += 1;
            return s.remove(i);
        };
    }

    fn sharedKey(tokens: []const u32) u64 {
        return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(tokens));
    }

    fn noteShared(s: *Store, tokens: []const u32) void {
        const k = sharedKey(tokens);
        if (std.mem.indexOfScalar(u64, s.shared_keys.items, k) != null) return;
        if (s.shared_keys.items.len == SHARED_KEYS) _ = s.shared_keys.orderedRemove(0);
        s.shared_keys.append(s.gpa, k) catch {};
    }

    /// The marks whose state the budget can hold (a new slice in `a`); the rest are refused now, so no pass, nor a peer, copies them.
    fn fitting(s: *Store, a: Allocator, marks_: []const u32) ![]const u32 {
        defer a.free(marks_);
        var out: std.ArrayList(u32) = .empty;
        for (marks_) |at| {
            const bytes = s.family.vtable.bytes(s.family.ptr, at);
            if (bytes <= s.budget) try out.append(a, at) else {
                s.counts.refused += 1;
                note("kept nothing at {d} tokens: {d} MiB passes the {d} MiB budget", .{ at, bytes >> 20, s.budget >> 20 });
            }
        }
        return out.toOwnedSlice(a);
    }

    /// Fitted marks that the budget holds together: the history's first, then the rewind, then the rest in order.
    fn fittingRewind(s: *Store, a: Allocator, marks_: []const u32, endpoint: u32, rewind: u32, endpoint_bytes: u64) ![]const u32 {
        defer a.free(marks_);
        var out: std.ArrayList(u32) = .empty;
        var used: u64 = 0;
        if (endpoint_bytes <= s.budget and endpoint_bytes > 0) {
            used = endpoint_bytes;
            if (std.mem.indexOfScalar(u32, marks_, endpoint) != null) try out.append(a, endpoint);
        }
        if (s.active_rewind_at == rewind and std.mem.indexOfScalar(u32, marks_, rewind) != null) try out.append(a, rewind);
        if (s.active_rewind_at == rewind) used += s.family.vtable.bytes(s.family.ptr, rewind);
        for (marks_) |at| {
            if (at == endpoint or at == rewind) continue;
            const bytes = s.family.vtable.bytes(s.family.ptr, at);
            if (bytes <= s.budget -| used) {
                try out.append(a, at);
                used += bytes;
            }
        }
        std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
        return out.toOwnedSlice(a);
    }

    /// A looked-up entry's restore went through (it is now the prompt's), or failed (dropped: the pass ran from 0).
    pub fn resumed(s: *Store, e: *Entry, prompt: []const u32, ok: bool) void {
        const i = std.mem.indexOfScalar(*Entry, s.entries.items, e) orelse return;
        if (!ok) {
            s.counts.failed += 1;
            return s.remove(i);
        }
        s.clock += 1;
        e.used = s.clock;
        s.counts.hits += 1;
        const last = s.gpa.dupe(u32, prompt) catch return;
        s.gpa.free(e.last);
        e.last = last;
    }

    /// Where a pass from `from` keeps states: the history, then min_gap apart the stable prefix and shared blocks.
    pub fn marks(s: *const Store, a: Allocator, prompt: []const u32, from: u32, history_len: u32, shared: []const u32, starts: []const u32, previous: []const u32) ![]const u32 {
        return s.marksRewind(a, prompt, from, history_len, 0, shared, starts, previous);
    }

    /// `marks` with the active rewind second: kept however near the other marks, never at or before the resume point.
    fn marksRewind(s: *const Store, a: Allocator, prompt: []const u32, from: u32, history_len: u32, rewind: u32, shared: []const u32, starts: []const u32, previous: []const u32) ![]const u32 {
        if (prompt.len < s.rules.min_prompt) return &.{};
        var out: std.ArrayList(u32) = .empty;
        var want: std.ArrayList(u32) = .empty;
        defer want.deinit(a);
        try want.append(a, history_len);
        if (rewind > 0 and s.active_rewind_at == rewind) try want.append(a, rewind);
        if (previous.len > 0) {
            const stable: u32 = @intCast(std.mem.indexOfDiff(u32, previous, prompt) orelse @min(previous.len, prompt.len));
            if (stable > 0 and stable < history_len and stable >= history_len / 2) try want.append(a, stable);
        }
        try want.appendSlice(a, shared);
        for (want.items, 0..) |w, k| {
            const at = if (s.rules.planned) s.startAt(starts, w) else w;
            if (at <= from or at + s.rules.lookahead > prompt.len or at >= prompt.len or !s.usable(at, starts)) continue;
            const is_rewind = rewind > 0 and k == 1 and at == rewind;
            if (!is_rewind and at - from < s.rules.min_gap and (k > 0 or (from > 0 and s.rules.warm))) continue; // near the resume point
            if (k > 0 and !is_rewind) { // the history's mark always (warm families: away from the resume point); the others away from it and each other
                const near = for (out.items) |o| {
                    if (@max(o, at) - @min(o, at) < s.rules.min_gap) break true;
                } else false;
                if (near) continue;
            }
            if (std.mem.indexOfScalar(u32, out.items, at) == null) try out.append(a, at);
        }
        std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
        return out.toOwnedSlice(a);
    }

    /// Keep the state at `at` with its span prefix, evicting within the budget; `starts` are a learned cut's chunks.
    pub fn keep(s: *Store, prompt: []const u32, at: u32, owner: ?*anyopaque, starts: []const u32, spans: []const modes.Span) bool {
        return s.keepIn(prompt, at, owner, starts, spans, 0);
    }

    /// `keep` in conversation `chat` (its request's, `join`; 0: the conversation of the state it extends, `chatOf`).
    pub fn keepIn(s: *Store, prompt: []const u32, at: u32, owner: ?*anyopaque, starts: []const u32, spans: []const modes.Span, chat_in: u32) bool {
        const n = @as(usize, at) + s.rules.lookahead;
        if (at == 0 or n > prompt.len) return false;
        s.clock += 1;
        const shared = std.mem.indexOfScalar(u64, s.shared_keys.items, sharedKey(prompt[0..n])) != null;
        for (s.entries.items) |e| if (e.at == at and std.mem.eql(u32, e.tokens, prompt[0..n]) and modes.equal(e.decode_spans, spans, at)) {
            e.used = s.clock; // the same state again: no copy
            e.shared = e.shared or shared;
            if (shared) learned.learn(s, e, starts);
            return true;
        };
        const decoded = modes.prefix(s.gpa, spans, at) catch |err| return s.fail(at, err);
        var owns_decoded = true;
        defer if (owns_decoded) s.gpa.free(decoded);
        const bytes = s.family.vtable.bytes(s.family.ptr, at);
        if (bytes > s.budget) {
            s.counts.refused += 1;
            note("kept nothing at {d} tokens: {d} MiB passes the {d} MiB budget", .{ at, bytes >> 20, s.budget >> 20 });
            return false;
        }
        while (s.entries.items.len > 0) { // room: spare storage this save can take, else new storage inside the budget
            if (s.family.vtable.reuses) |f| if (f(s.family.ptr, at)) break;
            const free_ = s.spare();
            if (s.held + free_ + bytes <= s.budget) break;
            if (free_ > 0) {
                if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| s.held -| bytes); // spare that cannot take this state goes before any state
                if (s.spare() < free_) continue;
            }
            s.evict(s.victimBut(null, prompt).?, true); // its storage may come back spare and take this state
            s.counts.evicted += 1;
        }
        if (s.entries.items.len == 0) if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| bytes);
        const chat = if (chat_in != 0) chat_in else s.chatOf(prompt, at);
        const e = s.gpa.create(Entry) catch return s.fail(at, error.OutOfMemory);
        const tokens = s.gpa.dupe(u32, prompt[0..n]) catch {
            s.gpa.destroy(e);
            return s.fail(at, error.OutOfMemory);
        };
        const last = s.gpa.dupe(u32, prompt) catch {
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.fail(at, error.OutOfMemory);
        };
        const saved = s.family.vtable.save(s.family.ptr, owner, at) catch |err| {
            s.gpa.free(last);
            s.gpa.free(tokens);
            s.gpa.destroy(e);
            return s.fail(at, err);
        };
        const charged = if (s.family.vtable.charged) |f| f(s.family.ptr, saved) else bytes;
        e.* = .{ .tokens = tokens, .decode_spans = decoded, .at = at, .saved = saved, .bytes = charged, .born = @intCast(prompt.len), .used = s.clock, .last = last, .shared = shared, .chat = chat, .own = !shared };
        owns_decoded = false;
        s.entries.append(s.gpa, e) catch {
            s.free(e);
            return s.fail(at, error.OutOfMemory);
        };
        s.held += charged;
        s.counts.kept += 1;
        if (shared) learned.learn(s, e, starts);
        s.enforce(e);
        if (s.family.vtable.trim) |f| f(s.family.ptr, s.budget -| s.held); // spare storage only inside what the budget leaves
        return true;
    }

    /// A learned state on disk longer than `have`, read back as a shared entry; else `have` (none, or reading failed).
    fn recall(s: *Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span, have_in: ?*Entry) ?*Entry {
        const have = if (spans.len == 0) s.spillRecall(prompt, have_in) else have_in;
        const im = s.imprint orelse return have;
        const read = s.family.vtable.read orelse return have;
        const m = im.best(prompt, starts, s.rules.planned, if (have) |x| x.at else 0) orelse return have;
        if (!modes.equal(&.{}, spans, m.at)) return have; // the request decodes rows the learned prompt pass prefilled
        const bytes = s.family.vtable.bytes(s.family.ptr, m.at);
        while (bytes <= s.budget and s.held + s.spare() + bytes > s.budget) {
            s.evict(s.victimBut(have, prompt) orelse return have, true);
            s.counts.evicted += 1;
        }
        if (bytes > s.budget) return have;
        const e = s.gpa.create(Entry) catch return have;
        e.* = .{ .tokens = &.{}, .at = m.at, .saved = undefined, .bytes = bytes, .born = @intCast(prompt.len), .used = s.clock, .last = &.{}, .shared = true };
        e.tokens = s.gpa.dupe(u32, m.tokens) catch return s.drop(e, have, false);
        e.last = s.gpa.dupe(u32, prompt) catch return s.drop(e, have, false);
        const key = m.key;
        e.saved = read(s.family.ptr, im.dir, key, m.at) catch |err| {
            note("reading the learned {d} tokens failed ({s}); forgotten, a later pass learns them again", .{ e.at, @errorName(err) });
            learned.unlearn(s, im, key);
            return s.drop(e, have, false);
        };
        im.touch(key);
        s.entries.append(s.gpa, e) catch return s.drop(e, have, true);
        s.held += bytes;
        if (!@import("builtin").is_test) std.log.info("prompt cache: read {d} learned tokens from disk", .{m.at});
        return e;
    }

    /// The spill tier's longest stored state this prompt extends, when it is longer than `have`: read into new storage as an
    /// entry no conversation owns. Else `have`.
    fn spillRecall(s: *Store, prompt: []const u32, have: ?*Entry) ?*Entry {
        if (!s.policy.spill) return have;
        const find_stored = s.family.vtable.stored orelse return have;
        const load = s.family.vtable.load orelse return have;
        var hit: Stored = .{ .at = 0, .key = 0, .tokens = &.{} };
        if (!find_stored(s.family.ptr, prompt, if (have) |x| x.at else 0, &hit.at, &hit.key, &hit.tokens)) return have;
        const bytes = s.family.vtable.bytes(s.family.ptr, hit.at);
        if (bytes > s.budget) return have;
        while (s.held + s.spare() + bytes > s.budget) {
            s.evict(s.victimBut(have, prompt) orelse return have, true);
            s.counts.evicted += 1;
        }
        const e = s.gpa.create(Entry) catch return have;
        e.* = .{ .tokens = &.{}, .at = hit.at, .saved = undefined, .bytes = bytes, .born = @intCast(prompt.len), .used = s.clock, .last = &.{}, .own = false, .spilled = true, .chat = s.newChat(prompt.len) };
        e.tokens = s.gpa.dupe(u32, hit.tokens) catch return s.drop(e, have, false);
        e.last = s.gpa.dupe(u32, prompt) catch return s.drop(e, have, false);
        e.saved = load(s.family.ptr, hit.key, hit.at) catch |err| {
            note("reading {d} stored tokens failed ({s}); prefilling them", .{ hit.at, @errorName(err) });
            return s.drop(e, have, false);
        };
        s.entries.append(s.gpa, e) catch return s.drop(e, have, true);
        s.held += bytes;
        s.counts.restored += 1;
        return e;
    }

    /// Past the highwater share of the budget, the next state eviction would take is stored now (one a call), so its
    /// eviction frees its memory at once.
    fn earlyWrite(s: *Store) void {
        const hw = s.policy.spill_highwater;
        if (!s.policy.spill or hw <= 0) return;
        if (@as(f64, @floatFromInt(s.held)) <= hw * @as(f64, @floatFromInt(s.budget))) return;
        const f = s.family.vtable.evict orelse return;
        const i = s.capVictim(null, true) orelse return;
        const e = s.entries.items[i];
        e.spilled = true;
        if (!s.superseded(e)) f(s.family.ptr, e.saved, e.tokens, e.at);
    }

    /// A clean stop: every state not stored yet is stored, newest first, until `max_ns` have passed.
    pub fn flush(s: *Store, max_ns: u64) u32 {
        if (!s.policy.spill) return 0;
        const f = s.family.vtable.evict orelse return 0;
        var t0: std.c.timespec = undefined;
        _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t0);
        var n: u32 = 0;
        while (true) {
            var best: ?*Entry = null;
            for (s.entries.items) |e| if (!e.spilled and !s.superseded(e) and (best == null or e.used > best.?.used)) {
                best = e;
            };
            const e = best orelse break;
            e.spilled = true;
            f(s.family.ptr, e.saved, e.tokens, e.at);
            n += 1;
            var t1: std.c.timespec = undefined;
            _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t1);
            const spent: u64 = @intCast((t1.sec - t0.sec) * 1_000_000_000 + (t1.nsec - t0.nsec));
            if (spent > max_ns) break;
        }
        return n;
    }

    /// A half-made recalled entry undone (`saved` once read); the lookup goes on with `have`.
    fn drop(s: *Store, e: *Entry, have: ?*Entry, saved: bool) ?*Entry {
        if (saved) s.family.vtable.drop(s.family.ptr, e.saved);
        s.gpa.free(e.tokens);
        s.gpa.free(e.last);
        s.gpa.destroy(e);
        return have;
    }

    /// One log line after a prompt pass: where it resumed, how many states it kept, and what the store holds.
    pub fn report(s: *const Store, prompt: usize, from: u32, kept: u64) void {
        if (@import("builtin").is_test) return;
        std.log.info("prompt cache: {d} tokens, resumed at {d}, kept {d}; {d} states, {d} MiB and {d} MiB spare of {d} MiB (hits {d}, misses {d}, evicted {d}, refused {d}, failed {d})", .{ prompt, from, kept, s.entries.items.len, s.held >> 20, s.spare() >> 20, s.budget >> 20, s.counts.hits, s.counts.misses, s.counts.evicted, s.counts.refused, s.counts.failed });
    }

    fn fail(s: *Store, at: u32, err: anyerror) bool {
        defer s.counts.failed += 1;
        note("keeping {d} tokens failed ({s}); a later turn prefills them", .{ at, @errorName(err) });
        return false;
    }

    /// The entry to free first, never `skip`: a superseded one, oldest first; else the oldest, the turn's rewind last.
    fn victimBut(s: *const Store, skip: ?*Entry, prompt: []const u32) ?usize {
        if (s.policy.recipe) return s.capVictim(skip, false);
        var best: ?usize = null;
        for (s.entries.items, 0..) |e, i| {
            if (e.shared or e == skip or s.isActiveRewind(e, prompt)) continue; // a shared cut is evicted only as the least recently used
            const now = prompt.len > e.born and prompt.len > e.tokens.len and std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens); // this prompt extends it
            const moved_on = now or for (s.entries.items) |o| {
                if (o != e and o.born > e.born and o.tokens.len > e.tokens.len and std.mem.eql(u32, o.tokens[0..e.tokens.len], e.tokens)) break true;
            } else false;
            if (moved_on and (best == null or e.used < s.entries.items[best.?].used)) best = i;
        }
        if (best) |i| return i;
        var oldest: ?usize = null;
        for (s.entries.items, 0..) |e, i| if (e != skip and (oldest == null or s.before(e, s.entries.items[oldest.?], prompt))) {
            oldest = i;
        };
        return oldest;
    }

    /// The recipe keeps a prompt's own end state on the grid whatever the request names (engine.grid_point): without a
    /// history length (a raw prompt) the end of the prompt; a prompt that ends on the grid keeps one grid step short of
    /// its end, so a repeat still runs its last rows (the recipe keeps the head's logits row to replay; here none).
    fn recipeHistory(s: *const Store, n: usize, history: u32) u32 {
        if (!s.policy.recipe) return history;
        const h: u32 = if (history == 0 or history > n) @intCast(n) else history;
        const g = s.rules.grid;
        return if (s.rules.planned and g > 0 and h == n and h % g == 0 and h > g) h - g else h;
    }

    /// The recipe's cap_victim (multi.py): the least recently used state shorter than a longer one of its conversation, a
    /// shared cut excepted; else the least recently used of all. Never `skip`.
    fn capVictim(s: *const Store, skip: ?*Entry, unspilled: bool) ?usize {
        var oldest: ?usize = null;
        var shorter: ?usize = null;
        for (s.entries.items, 0..) |e, i| {
            if (e == skip or (unspilled and e.spilled)) continue;
            if (oldest == null or e.used < s.entries.items[oldest.?].used) oldest = i;
            if (e.shared) continue;
            const longer = for (s.entries.items) |o| {
                if (o != e and o.chat == e.chat and o.at > e.at) break true;
            } else false;
            if (longer and (shorter == null or e.used < s.entries.items[shorter.?].used)) shorter = i;
        }
        return shorter orelse oldest;
    }

    /// A new conversation whose first prompt is `root` tokens long (`continuedChat` matches past it).
    fn newChat(s: *Store, root: usize) u32 {
        const chat = s.next_chat;
        s.next_chat += 1;
        s.roots.put(s.gpa, chat, @intCast(root)) catch {};
        if (s.roots.count() > 4 * @max(16, s.entries.items.len, s.policy.entries)) s.pruneRoots(chat); // conversations no kept state is in
        return chat;
    }

    /// The conversation a state kept for `prompt` at `at` belongs to when its request named none: the longest kept
    /// state this prompt extends (the one it resumed from, or an earlier mark of this pass) that is not a shared cut,
    /// which is every conversation's; else a new one.
    fn chatOf(s: *Store, prompt: []const u32, at: u32) u32 {
        var best: ?*Entry = null;
        for (s.entries.items) |e| {
            if (e.shared or e.at >= at or e.tokens.len > prompt.len) continue;
            if (best != null and e.at <= best.?.at) continue;
            if (std.mem.eql(u32, prompt[0..e.tokens.len], e.tokens)) best = e;
        }
        if (best) |b| return b.chat;
        return s.newChat(prompt.len);
    }

    /// The conversation a request's kept states join, decided at its lookup (`hit`: the state it resumes, null: none).
    /// A hit on a conversation's own turn state keeps that conversation; otherwise (a shared cut, a state read back
    /// from disk, nothing) the conversation the prompt continues (`continuedChat`; `matchable`: drafted, as the recipe
    /// requires), counted at its prefill (`continued`); else a new one. Deterministic in the states' use order.
    pub fn join(s: *Store, prompt: []const u32, hit: ?*const Entry, matchable: bool) Join {
        const source: Join.Source = if (hit) |e| (if (e.shared) .shared else .other) else .nothing;
        if (hit) |e| if (ownTurn(e)) return .{ .chat = e.chat, .source = .other };
        if (matchable) if (s.continuedChat(prompt)) |c| return .{
            .chat = c.chat,
            .continued = true,
            .prior_len = @intCast(c.tokens.len),
            .back = @intCast(commonPrefix(c.tokens, prompt)),
            .source = source,
        };
        return .{ .chat = s.newChat(prompt.len), .source = source };
    }

    /// One of a conversation's own turn states: not a shared cut, nor a state read back from disk (no conversation's).
    /// Its tokens are that conversation's, so a hit on it is the evidence.
    fn ownTurn(e: *const Entry) bool {
        return e.own and !e.shared and e.chat != 0;
    }

    /// The own kept state whose conversation `prompt` continues (continued_chat): its tokens repeat the prompt through
    /// its conversation's first prompt and chat_match tokens of the reply to it. The most recently used first, a later
    /// entry winning a tie; a conversation whose first prompt was forgotten is passed over.
    pub fn continuedChat(s: *const Store, prompt: []const u32) ?*Entry {
        var best: ?*Entry = null;
        for (s.entries.items) |c| {
            if (!ownTurn(c)) continue;
            const root = s.roots.get(c.chat) orelse continue;
            const n = @as(usize, root) + chat_match;
            if (c.tokens.len < n or prompt.len < n or c.tokens[n - 1] != prompt[n - 1] or !std.mem.eql(u32, c.tokens[0..n], prompt[0..n])) continue;
            if (best == null or c.used >= best.?.used) best = c;
        }
        return best;
    }

    /// The first prompt lengths of conversations no kept state is in, forgotten (never `keep_`'s).
    fn pruneRoots(s: *Store, keep_: u32) void {
        var gone: std.ArrayList(u32) = .empty;
        defer gone.deinit(s.gpa);
        var it = s.roots.keyIterator();
        while (it.next()) |k| {
            if (k.* == keep_) continue;
            const held = for (s.entries.items) |e| {
                if (e.chat == k.*) break true;
            } else false;
            if (!held) gone.append(s.gpa, k.*) catch return;
        }
        for (gone.items) |k| _ = s.roots.remove(k);
    }

    /// After a request's prompt pass resumed at `cut` of `len` tokens: one that continued its conversation is counted,
    /// and a large prefill logged with where the prompt left the kept state and what it resumed from.
    pub fn continued(s: *Store, j: Join, cut: u32, len: usize) void {
        if (!j.continued) return;
        s.counts.chats_continued += 1;
        s.counts.chats_continued_prefill_tokens += len -| cut;
        if (!shouldLog(j, cut, len) or @import("builtin").is_test) return;
        var buf: [512]u8 = undefined;
        std.log.info("{s}", .{continuedLine(&buf, j, cut, len)});
    }

    /// The caps after a state is kept (multi.py `_keep`): the conversation's own states, then the entry count, then the bytes.
    fn enforce(s: *Store, newest: *Entry) void {
        const p = s.policy;
        if (p.per_chat > 0 and newest.own) while (true) {
            var count: u32 = 0;
            var oldest: ?usize = null;
            for (s.entries.items, 0..) |e, i| if (e.own and e.chat == newest.chat) {
                count += 1;
                if (oldest == null or e.used < s.entries.items[oldest.?].used) oldest = i;
            };
            if (count <= p.per_chat) break;
            if (s.entries.items[oldest.?] == newest) break; // the state this call kept stays
            s.evict(oldest.?, false); // superseded turns are not worth the disk tier
            s.counts.evicted += 1;
        };
        if (p.entries > 0) while (s.entries.items.len > @max(1, p.entries)) {
            s.evict(s.capVictim(newest, false) orelse break, true);
            s.counts.evicted += 1;
        };
        if (p.bytes_cap > 0) while (s.entries.items.len > 1 and s.held > p.bytes_cap) {
            s.evict(s.capVictim(newest, false) orelse break, false); // dropped for its bytes: not written
            s.counts.evicted += 1;
        };
        s.earlyWrite();
    }

    /// Where `prompt` stops sharing a kept state, rounded down to the grid, when that adds at least `shared_least` tokens past
    /// the resume point and lies before the history's end (engine.shared_points; the system block's cut comes from the host).
    fn lcpPoint(s: *const Store, prompt: []const u32, from: u32, endpoint: u32, starts: []const u32) ?u32 {
        const least = s.policy.shared_least;
        var best: usize = 0;
        for (s.entries.items) |e| {
            if (e.tokens.len < from + least) continue;
            const n = @min(prompt.len, e.tokens.len);
            const lcp = std.mem.indexOfDiff(u32, prompt[0..n], e.tokens[0..n]) orelse n;
            best = @max(best, lcp);
        }
        const at: u32 = if (s.rules.planned) s.startAt(starts, @intCast(best)) else @intCast(best);
        if (at < from + least or at >= endpoint) return null;
        return at;
    }

    /// /health's kept-state numbers: the first eight conversations by their state count (the recipe's kept_mix.by_chat).
    pub fn keptStats(s: *const Store) KeptStats {
        var k: KeptStats = .{ .prompts = @intCast(s.entries.items.len), .bytes = s.held, .bytes_cap = s.policy.bytes_cap, .entries_cap = s.policy.entries, .per_chat = s.policy.per_chat, .budget = s.budget };
        var chats: [64][3]u64 = undefined; // [chat, states, longest]
        var n: usize = 0;
        for (s.entries.items) |e| {
            if (e.own) k.own += 1 else k.shared += 1;
            k.rows += e.at;
            const slot = for (chats[0..n]) |*c| {
                if (c[0] == e.chat) break c;
            } else if (n < chats.len) blk: {
                chats[n] = .{ e.chat, 0, 0 };
                n += 1;
                break :blk &chats[n - 1];
            } else continue;
            slot[1] += 1;
            slot[2] = @max(slot[2], e.at);
        }
        k.chats = @intCast(n);
        std.mem.sort([3]u64, chats[0..n], {}, struct {
            fn lt(_: void, x: [3]u64, y: [3]u64) bool {
                return if (x[1] != y[1]) x[1] > y[1] else x[0] < y[0];
            }
        }.lt);
        k.hits, k.misses, k.evicted, k.restored = .{ s.counts.hits, s.counts.misses, s.counts.evicted, s.counts.restored };
        k.chats_continued, k.chats_continued_prefill_tokens = .{ s.counts.chats_continued, s.counts.chats_continued_prefill_tokens };
        if (s.policy.spill) if (s.family.vtable.spill_stats) |f| {
            k.spill_on = true;
            f(s.family.ptr, &k.spill);
        };
        k.by_chat_n = @intCast(@min(n, k.by_chat.len));
        @memcpy(k.by_chat[0..k.by_chat_n], chats[0..k.by_chat_n]);
        return k;
    }

    /// Eviction order past the superseded states: least recently used first, the turn's rewind state last.
    fn before(s: *const Store, e: *const Entry, o: *const Entry, prompt: []const u32) bool {
        const e_rewind = s.isActiveRewind(e, prompt);
        const o_rewind = s.isActiveRewind(o, prompt);
        return if (e_rewind != o_rewind) o_rewind else e.used < o.used;
    }

    /// The last lookup's rewind state, while the prompt still extends it.
    fn isActiveRewind(s: *const Store, e: *const Entry, prompt: []const u32) bool {
        return e.at == s.active_rewind_at and e.tokens.len <= prompt.len and std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len]);
    }
};

/// A line on the engine's log; tests stay quiet (the build runner fails a test that writes to stderr).
pub fn note(comptime fmt: []const u8, args: anytype) void {
    if (@import("builtin").is_test) return;
    std.log.warn("prompt cache: " ++ fmt, args);
}

/// Whether `Store.continued` logs a continued request's pass from `cut` of `len` tokens: its prefill is large.
pub fn shouldLog(j: Join, cut: u32, len: usize) bool {
    return j.continued and len -| cut >= chat_log_tokens;
}

/// `Store.continued`'s log line, in `buf`.
pub fn continuedLine(buf: []u8, j: Join, cut: u32, len: usize) []const u8 {
    const source = switch (j.source) {
        .nothing => "nothing",
        .shared => "a shared prefix",
        .other => "a state that is not a turn's own (read back from disk)",
    };
    var why_buf: [256]u8 = undefined;
    const why = (if (j.back < j.prior_len)
        std.fmt.bufPrint(&why_buf, "it went back to token {d}, before that conversation's newest kept state ({d} tokens)", .{ j.back, j.prior_len })
    else
        std.fmt.bufPrint(&why_buf, "its kept state of {d} tokens could not serve it (off the request's chunks or decoded rows, or a longer state served)", .{j.prior_len})) catch "";
    return std.fmt.bufPrint(buf, "prompt cache: conversation {d}: resumed at {d} of {d} tokens from {s}, prefilling {d}; {s}", .{ j.chat, cut, len, source, len -| cut, why }) catch buf[0..0];
}

/// The tokens two id lists share from the start.
pub fn commonPrefix(a: []const u32, b: []const u32) usize {
    return std.mem.indexOfDiff(u32, a, b) orelse @min(a.len, b.len);
}

/// The chunk start at or before `at` (0 when none).
fn floorStart(starts: []const u32, at: u32) u32 {
    var best: u32 = 0;
    for (starts) |p| if (p <= at and p > best) {
        best = p;
    };
    return best;
}

test {
    _ = @import("prompt_cache_test.zig");
    _ = @import("prompt_cache_rewind_test.zig");
    _ = @import("prompt_cache_policy_test.zig");
    _ = @import("prompt_cache_chat_test.zig");
    _ = @import("learned_fault_test.zig");
    _ = modes;
}
