//! Conversations on the prompt cache (patch 0111): a request that continues its conversation without resuming that
//! conversation's own turn state keeps its id. The continuation cases translate tools/chat_continued_check.py at the GLM
//! fork's main 8641c90; the rest drive the store as the lane host does (`Fake.hosted`).
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Store = pc.Store;
const Entry = pc.Entry;
const Fake = @import("prompt_cache_fake.zig").Fake;
const fresh = @import("prompt_cache_test.zig").fresh;
const Allocator = std.mem.Allocator;
const gpa = std.testing.allocator;

fn seq(comptime n: usize, comptime first: u32) [n]u32 {
    var out: [n]u32 = undefined;
    for (&out, 0..) |*t, i| t.* = first + @as(u32, @intCast(i));
    return out;
}

const system = seq(900, 100000); // a 900-token system prompt every conversation shares
const opening = seq(100, 200000); // and the same 100-token opening message

/// A conversation's prompts: the first, then each turn appending a reply and a tool result of its own (50 tokens).
fn convo(a: Allocator, seed: u32, turns: usize) ![][]u32 {
    const out = try a.alloc([]u32, turns + 1);
    var ids: std.ArrayList(u32) = .empty;
    try ids.appendSlice(a, &system);
    try ids.appendSlice(a, &opening);
    out[0] = try a.dupe(u32, ids.items);
    for (0..turns) |t| {
        for (0..50) |k| try ids.append(a, seed * 1000 + @as(u32, @intCast(t)) * 50 + @as(u32, @intCast(k)));
        out[t + 1] = try a.dupe(u32, ids.items);
    }
    return out;
}

const Kind = struct { own: bool = true, shared: bool = false };

/// A kept state of `tokens` in conversation `chat`, the most recently used (as a keep records it).
fn state(s: *Store, f: *Fake, tokens: []const u32, chat: u32, kind: Kind) !*Entry {
    const at: u32 = @intCast(tokens.len);
    f.at = at;
    const e = try gpa.create(Entry);
    s.clock += 1;
    e.* = .{ .tokens = try gpa.dupe(u32, tokens), .at = at, .saved = try s.family.vtable.save(s.family.ptr, null, at), .bytes = 0, .born = at, .used = s.clock, .last = try gpa.dupe(u32, tokens), .chat = chat, .own = kind.own, .shared = kind.shared };
    try s.entries.append(gpa, e);
    return e;
}

fn cat(a: Allocator, x: []const u32, y: []const u32) ![]u32 {
    const out = try a.alloc(u32, x.len + y.len);
    @memcpy(out[0..x.len], x);
    @memcpy(out[x.len..], y);
    return out;
}

test "common prefix" {
    const big = try gpa.alloc(u32, 500000);
    defer gpa.free(big);
    for (big, 0..) |*t, i| t.* = @intCast(i);
    const other = try gpa.dupe(u32, big);
    defer gpa.free(other);
    other[123457] = std.math.maxInt(u32);
    try std.testing.expectEqual(@as(usize, 2), pc.commonPrefix(&.{ 1, 2, 3, 4 }, &.{ 1, 2, 9 }));
    try std.testing.expectEqual(@as(usize, 3), pc.commonPrefix(&.{ 1, 2, 3 }, &.{ 1, 2, 3, 4 }));
    try std.testing.expectEqual(@as(usize, 0), pc.commonPrefix(&.{}, &.{1}));
    try std.testing.expectEqual(@as(usize, 123457), pc.commonPrefix(big, other));
}

test "continued conversations: a rewind before the kept turns continues, another conversation or an edit starts its own" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const ar = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{}, 1 << 30);
    defer s.deinit();
    const a = try convo(ar, 7, 6);
    const b = try convo(ar, 8, 6);
    try s.roots.put(gpa, 1, @intCast(a[0].len));
    _ = try state(&s, &f, &system, 0, .{ .own = false, .shared = true });
    _ = try state(&s, &f, a[5], 1, .{});
    const newest = try state(&s, &f, a[6], 1, .{});
    // issue #104: a rewind before the kept turns continues its conversation
    try std.testing.expectEqual(newest, s.continuedChat(try cat(ar, a[3], &.{ 9, 9, 9 })).?);
    // another conversation behind the same system prompt and opening starts its own
    try std.testing.expect(s.continuedChat(b[2]) == null);
    // a prompt that has not repeated the first reply yet starts its own
    const fives: [10]u32 = @splat(5);
    try std.testing.expect(s.continuedChat(try cat(ar, a[0], &fives)) == null);
    // an edit of the first message starts a new conversation
    var edited: std.ArrayList(u32) = .empty;
    try edited.appendSlice(ar, &system);
    try edited.appendSlice(ar, opening[0..50]);
    try edited.appendNTimes(ar, 0, 50);
    try edited.appendSlice(ar, a[2][a[0].len..]);
    try std.testing.expect(s.continuedChat(edited.items) == null);
}

test "continued conversations pass over shared cuts, states read back from disk and forgotten conversations; the most recently used wins" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const ar = arena.allocator();
    const a = try convo(ar, 7, 6);
    const prompt = try cat(ar, a[2], &.{1});
    // (the source's picture states have no counterpart: the store neither keeps nor resumes a prompt with pictures)
    const kinds = [_]struct { chat: u32, kind: Kind }{
        .{ .chat = 1, .kind = .{ .own = false, .shared = true } }, // a shared cut
        .{ .chat = 1, .kind = .{ .own = false } }, // read back from disk
        .{ .chat = 1, .kind = .{ .shared = true } }, // an own state a later pass kept as a shared cut
        .{ .chat = 2, .kind = .{} }, // a conversation whose first prompt was forgotten
    };
    for (kinds) |k| {
        var f: Fake = .{ .gpa = gpa };
        var s = Store.init(gpa, f.snapshots(), .{}, 1 << 30);
        defer s.deinit();
        try s.roots.put(gpa, 1, @intCast(a[0].len));
        _ = try state(&s, &f, a[4], k.chat, k.kind);
        try std.testing.expect(s.continuedChat(prompt) == null);
    }
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{}, 1 << 30);
    defer s.deinit();
    try s.roots.put(gpa, 1, @intCast(a[0].len));
    try s.roots.put(gpa, 2, @intCast(a[0].len));
    _ = try state(&s, &f, a[3], 1, .{});
    _ = try state(&s, &f, a[4], 2, .{});
    try std.testing.expectEqual(@as(u32, 2), s.continuedChat(prompt).?.chat); // the most recently used match wins
    s.entries.items[0].used = s.entries.items[1].used; // a tie of use: the later entry
    try std.testing.expectEqual(@as(u32, 2), s.continuedChat(prompt).?.chat);
}

test "join: an own turn keeps its conversation, a continuation names it, anything else starts one" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const ar = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{}, 1 << 30);
    defer s.deinit();
    s.next_chat = 41;
    const a = try convo(ar, 7, 6);
    const shared = try state(&s, &f, &system, 0, .{ .own = false, .shared = true });
    const first = s.join(a[0], shared, true); // a new conversation behind a shared prefix
    try std.testing.expectEqual(@as(u32, 41), first.chat);
    try std.testing.expect(!first.continued);
    try std.testing.expectEqual(@as(u32, @intCast(a[0].len)), s.roots.get(41).?);
    const own = try state(&s, &f, a[3], 41, .{});
    const turn = s.join(a[4], own, true); // its own turn state: the conversation, not counted
    try std.testing.expectEqual(@as(u32, 41), turn.chat);
    try std.testing.expect(!turn.continued);
    const back = try cat(ar, a[2], &.{9});
    const rewound = s.join(back, shared, true); // back before its kept turn, from the shared prefix
    try std.testing.expectEqual(@as(u32, 41), rewound.chat);
    try std.testing.expect(rewound.continued and rewound.source == .shared);
    try std.testing.expectEqual(@as(u32, @intCast(a[3].len)), rewound.prior_len);
    try std.testing.expectEqual(@as(u32, @intCast(a[2].len)), rewound.back);
    const disk = try state(&s, &f, a[1], 60, .{ .own = false }); // read back from disk: no conversation's own state
    const restored = s.join(back, disk, true);
    try std.testing.expect(restored.continued and restored.chat == 41 and restored.source == .other);
    const nothing = s.join(back, null, true);
    try std.testing.expect(nothing.continued and nothing.chat == 41 and nothing.source == .nothing);
    const undrafted = s.join(back, shared, false); // draft:false: never matched
    try std.testing.expectEqual(@as(u32, 42), undrafted.chat);
    try std.testing.expect(!undrafted.continued);
    // conversations no kept state is in are forgotten once there are many
    for (0..80) |_| _ = s.join(&.{ 1, 2, 3 }, null, true);
    try std.testing.expect(s.roots.count() <= 4 * 16 + 1);
    try std.testing.expect(s.roots.contains(41)); // a kept state still names it
}

test "a conversation's first prompt is remembered while the kept states' cap could still hold it (0111: 4 x cache_entries)" {
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{}, 1 << 30);
    defer s.deinit();
    s.policy = .{ .recipe = true, .entries = 32 };
    const long = s.join(&seq(64, 300), null, true); // a long first prompt still in its pass: nothing kept for it yet
    for (0..80) |i| _ = s.join(&.{ 1, 2, @intCast(i) }, null, true); // short requests meanwhile
    try std.testing.expect(s.roots.contains(long.chat)); // 81 roots: under 4 x 32, nothing forgotten
    for (0..60) |i| _ = s.join(&.{ 1, 3, @intCast(i) }, null, true);
    try std.testing.expect(s.roots.count() <= 4 * 32 + 1); // past it, conversations no kept state is in go
    try std.testing.expect(!s.roots.contains(long.chat));
}

test "a continued request is counted with its prefill; a large one's line says where it resumed and why" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const ar = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{}, 1 << 30);
    defer s.deinit();
    const long = try convo(ar, 7, 200);
    try s.roots.put(gpa, 1, @intCast(long[0].len));
    _ = try state(&s, &f, long[200], 1, .{});
    const shared = try state(&s, &f, &system, 0, .{ .own = false, .shared = true });
    const prompt = try cat(ar, long[150], &.{9});
    const j = s.join(prompt, shared, true);
    try std.testing.expect(j.continued and j.chat == 1);
    s.continued(j, system.len, prompt.len);
    try std.testing.expectEqual(@as(u64, 1), s.counts.chats_continued);
    try std.testing.expectEqual(@as(u64, prompt.len - system.len), s.counts.chats_continued_prefill_tokens);
    try std.testing.expect(pc.shouldLog(j, system.len, prompt.len));
    var buf: [512]u8 = undefined;
    const line = pc.continuedLine(&buf, j, system.len, prompt.len);
    try std.testing.expect(std.mem.indexOf(u8, line, "conversation 1: resumed at 900 of") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "from a shared prefix") != null);
    const went = try std.fmt.allocPrint(ar, "went back to token {d}", .{long[150].len});
    try std.testing.expect(std.mem.indexOf(u8, line, went) != null);
    // a prompt that repeats the whole state, which its lookup did not resume: not a rewind
    const past = s.join(try cat(ar, long[200], &.{ 9, 9 }), shared, true);
    try std.testing.expect(past.continued and past.back >= past.prior_len);
    try std.testing.expect(std.mem.indexOf(u8, pc.continuedLine(&buf, past, system.len, 9000), "could not serve it") != null);
    // a small prefill: counted, not logged; a request that did not continue: not counted (the host runs on rank 0 only)
    s.continued(j, @intCast(prompt.len - 10), prompt.len);
    try std.testing.expect(!pc.shouldLog(j, @intCast(prompt.len - 10), prompt.len));
    s.continued(.{ .chat = 5 }, 0, prompt.len);
    try std.testing.expectEqual(@as(u64, 2), s.counts.chats_continued);
    try std.testing.expectEqual(@as(u64, prompt.len - system.len + 10), s.counts.chats_continued_prefill_tokens);
    const k = s.keptStats();
    try std.testing.expectEqual(@as(u64, 2), k.chats_continued);
    try std.testing.expectEqual(@as(u64, prompt.len - system.len + 10), k.chats_continued_prefill_tokens);
    const none = pc.continuedLine(&buf, .{ .chat = 3, .continued = true, .prior_len = 50, .back = 50 }, 0, 9000);
    try std.testing.expect(std.mem.indexOf(u8, none, "from nothing") != null and std.mem.indexOf(u8, none, "could not serve it") != null);
}

test "the log threshold: a continued request's prefill of chat_log_tokens or more" {
    const j: pc.Join = .{ .chat = 1, .continued = true };
    try std.testing.expect(!pc.shouldLog(j, 100, 100 + pc.chat_log_tokens - 1));
    try std.testing.expect(pc.shouldLog(j, 100, 100 + pc.chat_log_tokens));
    try std.testing.expect(!pc.shouldLog(.{ .chat = 1 }, 0, 1 << 20)); // not continued
    try std.testing.expect(!pc.shouldLog(j, 9000, 100)); // nothing prefilled
}

/// A turn behind a 16-token system prompt: `prompt` with the 2-token generation prompt at its end.
fn turnOf(a: Allocator, parts: []const []const u32) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    try out.appendSlice(a, &seq(16, 900));
    for (parts) |p| try out.appendSlice(a, p);
    try out.appendSlice(a, &.{ 7, 7 });
    return out.items;
}

test "a rewind before a conversation's kept turns keeps its id (#104): per_chat counts it once, another conversation is untouched" {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const ar = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 4, .min_prompt = 0, .min_gap = 4 }, 1 << 20);
    defer s.deinit();
    s.policy = .{ .recipe = true, .per_chat = 2, .shared_least = 1000 };
    const open_a = [_]u32{ 1, 31, 32, 33, 34, 35, 36, 37 };
    const reply2 = seq(40, 300); // turn 2's reply and tool result
    const reply3 = seq(20, 400);
    const a1 = try turnOf(ar, &.{&open_a}); // 26 tokens: the cut at 16 and A's own 24
    const ja, const sa = try f.hosted(&s, ar, a1, @intCast(a1.len - 2), &.{16});
    try std.testing.expectEqual(fresh(a1), sa);
    const b1 = try turnOf(ar, &.{&seq(8, 500)}); // B, behind the same system prompt
    const jb, _ = try f.hosted(&s, ar, b1, @intCast(b1.len - 2), &.{16});
    try std.testing.expect(jb.chat != ja.chat and !jb.continued);
    const a2 = try turnOf(ar, &.{ &open_a, &.{ 7, 7 }, &reply2 }); // 68: resumes 24, keeps 64
    const j2, _ = try f.hosted(&s, ar, a2, @intCast(a2.len - 2), &.{16});
    const a3 = try turnOf(ar, &.{ &open_a, &.{ 7, 7 }, &reply2, &.{ 7, 7 }, &reply3 }); // 90: keeps 88; A's 24 goes
    const j3, _ = try f.hosted(&s, ar, a3, @intCast(a3.len - 2), &.{16});
    try std.testing.expect(j2.chat == ja.chat and j3.chat == ja.chat and !j2.continued and !j3.continued);
    // the client goes back into turn 2's reply: only the cut serves it, and the conversation is A's
    const rewind = try cat(ar, a2[0..62], &.{ 99, 98, 97, 96, 95, 94, 93, 92, 7, 7 });
    const jr, const sr = try f.hosted(&s, ar, rewind, @intCast(rewind.len - 2), &.{16});
    try std.testing.expectEqual(fresh(rewind), sr);
    try std.testing.expect(jr.continued and jr.chat == ja.chat and jr.source == .shared);
    try std.testing.expectEqual(@as(u32, 62), jr.back);
    try std.testing.expectEqual(@as(u64, 1), s.counts.chats_continued);
    try std.testing.expectEqual(@as(u64, rewind.len - 16), s.counts.chats_continued_prefill_tokens);
    // A holds two own states (its 88 and the rewind's 68), B still its 24, and the cut
    var own_a: u32 = 0;
    for (s.entries.items) |e| if (e.own and !e.shared and e.chat == ja.chat) {
        own_a += 1;
    };
    try std.testing.expectEqual(@as(u32, 2), own_a);
    try std.testing.expectEqual(@as(u32, 24), s.find(try turnOf(ar, &.{ &seq(8, 500), &.{ 1, 2, 3, 4, 5, 6 } }), &.{}, &.{}).?.at);
    try std.testing.expectEqual(@as(usize, 4), s.entries.items.len);
    // another conversation behind the same system prompt and opening: its own, once the cut is all it shares
    const c1 = try turnOf(ar, &.{&open_a});
    const jc, _ = try f.hosted(&s, ar, c1, @intCast(c1.len - 2), &.{16});
    try std.testing.expect(jc.chat != ja.chat and !jc.continued);
}
