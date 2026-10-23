//! The recipe's kept-state policy on the prompt cache (Policy): entries, bytes and per-conversation caps, the order that
//! frees a conversation's superseded states first, shared cuts by recency only, shared points where prompts stop sharing.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const Store = pc.Store;
const Fake = @import("prompt_cache_fake.zig").Fake;
const fresh = @import("prompt_cache_test.zig").fresh;

/// A conversation's turn: `n` tokens of a line that starts `[chat, 1, 2, ...]`, history `h` of them.
fn line(buf: []u32, chat: u32, n: usize) []u32 {
    buf[0] = chat;
    for (1..n) |i| buf[i] = @intCast(i);
    return buf[0..n];
}

fn turn(f: *Fake, s: *Store, a: std.mem.Allocator, prompt: []const u32, history: u32) !u64 {
    return f.pass(s, prompt, try s.begin(a, prompt, history, &.{}, &.{}, null, &.{}));
}

fn recipe(entries: u32) pc.Policy {
    return .{ .recipe = true, .entries = entries };
}

test "past the entry cap a conversation's superseded turns go first, then the least recently used" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 4, .min_prompt = 0, .min_gap = 4 }, 1 << 20);
    defer s.deinit();
    s.policy = recipe(3);
    var b1: [64]u32 = undefined;
    var b2: [64]u32 = undefined;
    var b3: [64]u32 = undefined;
    const x1 = line(&b1, 1, 13); // chat 1 turn 1: kept at 12
    _ = try turn(&f, &s, a, x1, 12);
    const y1 = line(&b2, 2, 13); // chat 2: its only state
    _ = try turn(&f, &s, a, y1, 12);
    const x2 = line(&b3, 1, 25); // chat 1 turn 2 extends turn 1: kept at 24
    try std.testing.expectEqual(fresh(x2), try turn(&f, &s, a, x2, 24));
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    var b4: [64]u32 = undefined;
    const x3 = line(&b4, 1, 37); // turn 3: four states, the cap is 3: chat 1's turn 1 (superseded) goes, not chat 2's older one
    try std.testing.expectEqual(fresh(x3), try turn(&f, &s, a, x3, 36));
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    for (s.entries.items) |e| try std.testing.expect(!(e.chat == s.entries.items[0].chat and e.at == 12 and e.tokens[0] == 1));
    try std.testing.expectEqual(@as(u32, 36), s.find(line(&b4, 1, 40), &.{}, &.{}).?.at);
    try std.testing.expectEqual(@as(u32, 12), s.find(line(&b2, 2, 20), &.{}, &.{}).?.at);
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
    try std.testing.expectEqual(s.entries.items.len, f.live);
    // a third conversation: chat 1's turn 2 is superseded by its turn 3, so it goes, not the older chat 2
    var b5: [64]u32 = undefined;
    const z1 = line(&b5, 3, 13);
    _ = try turn(&f, &s, a, z1, 12);
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    try std.testing.expect(s.find(line(&b4, 1, 30), &.{}, &.{}) == null);
    try std.testing.expect(s.find(line(&b2, 2, 20), &.{}, &.{}) != null);
    // a fourth: nothing is superseded now, so the least recently used of all goes (chat 2's, unused since)
    var b6: [64]u32 = undefined;
    _ = try turn(&f, &s, a, line(&b6, 4, 13), 12);
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    try std.testing.expect(s.find(line(&b2, 2, 20), &.{}, &.{}) == null);
    try std.testing.expect(s.find(line(&b4, 1, 40), &.{}, &.{}) != null);
}

test "a conversation holds at most per_chat own states; a retry that branches earlier keeps its own" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 4, .min_prompt = 0, .min_gap = 4 }, 1 << 20);
    defer s.deinit();
    s.policy = .{ .recipe = true, .per_chat = 2 };
    var bufs: [5][64]u32 = undefined;
    _ = try turn(&f, &s, a, line(&bufs[0], 1, 13), 12);
    _ = try turn(&f, &s, a, line(&bufs[1], 1, 25), 24);
    _ = try turn(&f, &s, a, line(&bufs[2], 1, 37), 36);
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len); // 12 went
    try std.testing.expect(s.find(line(&bufs[3], 1, 20), &.{}, &.{}) == null);
    try std.testing.expectEqual(@as(u32, 24), s.find(line(&bufs[3], 1, 30), &.{}, &.{}).?.at);
    // another conversation is untouched by the first's quota
    _ = try turn(&f, &s, a, line(&bufs[4], 2, 13), 12);
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
}

/// A conversation behind a 16-token system prompt every conversation shares: `n` tokens, `chat` first after it.
fn behind(buf: []u32, chat: u32, n: usize) []u32 {
    for (0..16) |i| buf[i] = @intCast(900 + i);
    buf[16] = chat;
    for (17..n) |i| buf[i] = @intCast(i);
    return buf[0..n];
}

test "two conversations behind one kept system-prompt cut are two: per_chat lets neither evict the other's turns" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 4, .min_prompt = 0, .min_gap = 4 }, 1 << 20);
    defer s.deinit();
    s.policy = .{ .recipe = true, .per_chat = 2, .shared_least = 1000 };
    var bufs: [5][64]u32 = undefined;
    const a1 = behind(&bufs[0], 1, 25); // A's first turn keeps the system block's cut (16, shared) and its own 24
    const plan = try s.begin(a, a1, 24, &.{16}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 16, 24 }, plan.marks);
    _ = f.pass(&s, a1, plan);
    const b1 = behind(&bufs[1], 2, 25); // B resumes the shared cut and keeps its own 24
    const plan_b = try s.begin(a, b1, 24, &.{16}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 16), plan_b.from);
    try std.testing.expectEqual(fresh(b1), f.pass(&s, b1, plan_b));
    var chat_a: u32 = 0;
    var chat_b: u32 = 0;
    for (s.entries.items) |e| if (e.at == 24) {
        if (e.tokens[16] == 1) chat_a = e.chat else chat_b = e.chat;
    };
    try std.testing.expect(chat_a != 0 and chat_b != 0 and chat_a != chat_b);
    _ = try turn(&f, &s, a, behind(&bufs[2], 1, 37), 36); // A's turns 2 and 3: A holds two of its own, its 24 goes
    _ = try turn(&f, &s, a, behind(&bufs[3], 1, 49), 48);
    try std.testing.expectEqual(@as(u32, 24), s.find(behind(&bufs[4], 2, 30), &.{}, &.{}).?.at); // B's turn is still kept
    try std.testing.expectEqual(@as(u32, 16), s.find(behind(&bufs[4], 1, 30), &.{}, &.{}).?.at);
    try std.testing.expectEqual(@as(usize, 4), s.entries.items.len); // the cut, B's 24, A's 36 and 48
}

test "the bytes cap drops by the same order and keeps the newest" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 4, .min_prompt = 0, .min_gap = 4 }, 1 << 20);
    defer s.deinit();
    s.policy = .{ .recipe = true, .bytes_cap = 250 }; // fake bytes: 100 + at
    var bufs: [4][64]u32 = undefined;
    _ = try turn(&f, &s, a, line(&bufs[0], 1, 13), 12); // 112
    _ = try turn(&f, &s, a, line(&bufs[1], 2, 13), 12); // 224
    _ = try turn(&f, &s, a, line(&bufs[2], 3, 13), 12); // 336 > 250: the oldest goes
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    try std.testing.expect(s.held <= 250);
    try std.testing.expect(s.find(line(&bufs[3], 1, 20), &.{}, &.{}) == null);
    try std.testing.expectEqual(@as(u32, 12), s.find(line(&bufs[3], 3, 20), &.{}, &.{}).?.at);
}

test "a shared cut is never superseded, only aged out with the rest, and /health counts own and shared apart" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 4, .min_prompt = 0, .min_gap = 4 }, 1 << 20);
    defer s.deinit();
    s.policy = .{ .recipe = true, .entries = 8, .shared_least = 8 };
    var b1: [64]u32 = undefined;
    const p1 = line(&b1, 1, 37);
    const plan = try s.begin(a, p1, 36, &.{16}, &.{}, null, &.{}); // the system block ends at 16
    try std.testing.expectEqualSlices(u32, &.{ 16, 36 }, plan.marks);
    _ = f.pass(&s, p1, plan);
    var k = s.keptStats();
    try std.testing.expectEqual(@as(u32, 2), k.prompts);
    try std.testing.expectEqual(@as(u32, 1), k.own);
    try std.testing.expectEqual(@as(u32, 1), k.shared);
    try std.testing.expectEqual(@as(u32, 2), k.chats); // the turn kept after the cut does not take its id: chatOf passes over shared cuts
    try std.testing.expectEqual(@as(u64, 52), k.rows);
    try std.testing.expectEqual(@as(u8, 2), k.by_chat_n);
    try std.testing.expectEqual([3]u64{ 1, 1, 16 }, k.by_chat[0]);
    try std.testing.expectEqual([3]u64{ 2, 1, 36 }, k.by_chat[1]);
    // the shorter state is the shared one: with the cap at 1 the own, longer one is "superseded"? no: the shared one is
    // exempt, so the oldest of all (the shared cut, kept first) is what the cap takes
    s.policy.entries = 1;
    var b2: [64]u32 = undefined;
    _ = try turn(&f, &s, a, line(&b2, 9, 13), 12);
    k = s.keptStats();
    try std.testing.expectEqual(@as(u32, 1), k.prompts);
}

test "a prompt that stops sharing a kept one is also kept where it diverges, on the grid" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 4, .min_prompt = 0, .min_gap = 4 }, 1 << 20);
    defer s.deinit();
    s.policy = .{ .recipe = true, .shared_least = 8 };
    var b1: [64]u32 = undefined;
    _ = try turn(&f, &s, a, line(&b1, 1, 33), 32); // a conversation keeps 32
    var b2: [64]u32 = undefined;
    const p2 = line(&b2, 1, 45);
    @memset(p2[22..], 99); // shares 22 tokens with the kept 32, then its own
    const plan = try s.begin(a, p2, 44, &.{}, &.{}, null, &.{});
    // resumes nothing (the kept state at 32 is past the divergence), keeps at the grid below 22 and at its own end
    try std.testing.expectEqual(@as(u32, 0), plan.from);
    try std.testing.expectEqualSlices(u32, &.{ 20, 44 }, plan.marks);
    try std.testing.expectEqual(fresh(p2), f.pass(&s, p2, plan));
    // a third prompt that shares the first 20 tokens resumes at 20, a state that served no conversation of its own
    var b3: [64]u32 = undefined;
    const p3 = line(&b3, 1, 40);
    @memset(p3[21..], 77);
    const plan3 = try s.begin(a, p3, 39, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 20), plan3.from);
    try std.testing.expectEqual(fresh(p3), f.pass(&s, p3, plan3));
}

test "the recipe keeps no rewind state" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 4, .min_prompt = 0, .min_gap = 4 }, 1 << 20);
    defer s.deinit();
    var b1: [64]u32 = undefined;
    const p1 = line(&b1, 1, 41);
    const without = try s.beginRewind(a, p1, 40, 20, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 20, 40 }, without.marks);
    s.clear();
    s.policy = recipe(0);
    const with = try s.beginRewind(a, p1, 40, 20, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{40}, with.marks);
}

test "a raw prompt keeps its own end on the grid; a prompt that ends on the grid keeps a step short" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 4, .min_prompt = 0, .min_gap = 4 }, 1 << 20);
    defer s.deinit();
    s.policy = recipe(0);
    var b1: [64]u32 = undefined;
    const p1 = line(&b1, 1, 43); // no history: the end, floored to the grid
    const plan = try s.begin(a, p1, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{40}, plan.marks);
    try std.testing.expectEqual(fresh(p1), f.pass(&s, p1, plan));
    var b2: [64]u32 = undefined;
    const p2 = line(&b2, 1, 52); // extends it: resumes at 40, keeps 48
    const plan2 = try s.begin(a, p2, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 40), plan2.from);
    try std.testing.expectEqualSlices(u32, &.{48}, plan2.marks);
    try std.testing.expectEqual(fresh(p2), f.pass(&s, p2, plan2));
    var b3: [64]u32 = undefined;
    const p3 = line(&b3, 2, 48); // on the grid: keeps at 44
    const plan3 = try s.begin(a, p3, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{44}, plan3.marks);
}

/// The fake family with a disk tier: a state written out is its position, sum and tokens in a slot.
const Disky = struct {
    f: Fake,
    slots: [8]Slot = @splat(.{}),
    written: u32 = 0,
    read: u32 = 0,
    fail_read: bool = false,

    const Slot = struct { used: bool = false, at: u32 = 0, sum: u64 = 0, n: usize = 0, tokens: [96]u32 = undefined };
    const vt: pc.Snapshots.VTable = .{ .bytes = bytes, .save = save, .restore = restore, .drop = drop, .evict = evict, .stored = stored, .load = load };

    fn of(ptr: *anyopaque) *Disky {
        return @ptrCast(@alignCast(ptr));
    }
    fn snaps(d: *Disky) pc.Snapshots {
        return .{ .ptr = d, .vtable = &vt };
    }
    fn inner() pc.Snapshots.VTable {
        var dummy: Fake = .{ .gpa = std.testing.allocator };
        return dummy.snapshots().vtable.*;
    }
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 100 + at;
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        return inner().save(&of(ptr).f, owner, at);
    }
    fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: pc.Saved) anyerror!void {
        return inner().restore(&of(ptr).f, owner, saved);
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        inner().drop(&of(ptr).f, saved);
    }
    fn evict(ptr: *anyopaque, saved: pc.Saved, tokens: []const u32, at: u32) void {
        const d = of(ptr);
        const st: *Fake.State = @ptrCast(@alignCast(saved));
        for (&d.slots) |*s| if (!s.used) {
            s.* = .{ .used = true, .at = at, .sum = st.sum, .n = tokens.len };
            @memcpy(s.tokens[0..tokens.len], tokens);
            d.written += 1;
            return;
        };
    }
    fn stored(ptr: *anyopaque, prompt: []const u32, longer_than: u32, at: *u32, key: *u64, tokens: *[]const u32) bool {
        const d = of(ptr);
        var found = false;
        for (&d.slots, 0..) |*s, i| {
            if (!s.used or s.at <= longer_than or s.n > prompt.len or s.at >= prompt.len) continue;
            if (!std.mem.eql(u32, prompt[0..s.n], s.tokens[0..s.n])) continue;
            if (found and s.at <= at.*) continue;
            found = true;
            at.*, key.*, tokens.* = .{ s.at, i, s.tokens[0..s.n] };
        }
        return found;
    }
    fn load(ptr: *anyopaque, key: u64, _: u32) anyerror!pc.Saved {
        const d = of(ptr);
        if (d.fail_read) return error.ReadFailed;
        const s = d.slots[key];
        const st = try d.f.gpa.create(Fake.State);
        st.* = .{ .at = s.at, .sum = s.sum };
        d.f.live += 1;
        d.read += 1;
        return st;
    }
};

fn diskStore(d: *Disky, policy: pc.Policy) Store {
    var s = Store.init(std.testing.allocator, d.snaps(), .{ .planned = true, .grid = 4, .min_prompt = 0, .min_gap = 4 }, 1 << 20);
    s.policy = policy;
    return s;
}

test "an evicted state goes to disk and a prompt that extends it reads it back, equal to a fresh pass; a superseded one is not written" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var d: Disky = .{ .f = .{ .gpa = gpa } };
    var s = diskStore(&d, .{ .recipe = true, .entries = 2, .spill = true });
    defer s.deinit();
    var b1: [96]u32 = undefined;
    var b2: [96]u32 = undefined;
    var b3: [96]u32 = undefined;
    const c1 = line(&b1, 1, 41);
    _ = d.f.pass(&s, c1, try s.begin(a, c1, 0, &.{}, &.{}, null, &.{})); // chat 1 at 40
    const c2 = line(&b2, 2, 41);
    _ = d.f.pass(&s, c2, try s.begin(a, c2, 0, &.{}, &.{}, null, &.{})); // chat 2 at 40
    const c3 = line(&b3, 3, 41);
    _ = d.f.pass(&s, c3, try s.begin(a, c3, 0, &.{}, &.{}, null, &.{})); // chat 3: the cap is 2, chat 1's goes, to disk
    try std.testing.expectEqual(@as(u32, 1), d.written);
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    var b4: [96]u32 = undefined;
    const c1b = line(&b4, 1, 61); // chat 1 continues: nothing in memory, the disk has 40
    const plan = try s.begin(a, c1b, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 40), plan.from);
    try std.testing.expectEqual(@as(u32, 1), d.read);
    try std.testing.expectEqual(@as(u64, 1), s.counts.restored);
    try std.testing.expectEqual(fresh(c1b), d.f.pass(&s, c1b, plan));
    // the restored state is a kept one now; evicting what extends it writes only the longer one
    try std.testing.expect(s.entries.items.len <= 2);
    // a state a longer one of its chat extends is not written when it goes
    const before = d.written;
    var b5: [96]u32 = undefined;
    const c4 = line(&b5, 4, 41);
    _ = d.f.pass(&s, c4, try s.begin(a, c4, 0, &.{}, &.{}, null, &.{}));
    try std.testing.expect(d.written >= before); // chat 2's (oldest) went to disk
}

test "states dropped for a conversation's quota or the byte cap are not written; a failed read prefills from the start" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var d: Disky = .{ .f = .{ .gpa = gpa } };
    var s = diskStore(&d, .{ .recipe = true, .per_chat = 1, .spill = true });
    defer s.deinit();
    var b1: [96]u32 = undefined;
    var b2: [96]u32 = undefined;
    _ = d.f.pass(&s, line(&b1, 1, 41), try s.begin(a, line(&b1, 1, 41), 0, &.{}, &.{}, null, &.{}));
    _ = d.f.pass(&s, line(&b2, 1, 61), try s.begin(a, line(&b2, 1, 61), 0, &.{}, &.{}, null, &.{}));
    try std.testing.expectEqual(@as(u32, 0), d.written); // the older turn went by the quota, unwritten
    s.policy.per_chat = 0;
    s.policy.entries = 1;
    var b3: [96]u32 = undefined;
    _ = d.f.pass(&s, line(&b3, 5, 41), try s.begin(a, line(&b3, 5, 41), 0, &.{}, &.{}, null, &.{}));
    try std.testing.expectEqual(@as(u32, 1), d.written); // the entry cap writes
    d.fail_read = true;
    var b4: [96]u32 = undefined;
    const c = line(&b4, 1, 81);
    const plan = try s.begin(a, c, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 0), plan.from); // the read failed: the pass starts at 0
    try std.testing.expectEqual(fresh(c), d.f.pass(&s, c, plan));
}

test "past the highwater the next victim is written early, and flush writes the rest newest first" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var d: Disky = .{ .f = .{ .gpa = gpa } };
    var s = diskStore(&d, .{ .recipe = true, .spill = true, .spill_highwater = 0.0001 });
    defer s.deinit();
    s.budget = 2000;
    var b1: [96]u32 = undefined;
    var b2: [96]u32 = undefined;
    var b3: [96]u32 = undefined;
    _ = d.f.pass(&s, line(&b1, 1, 41), try s.begin(a, line(&b1, 1, 41), 0, &.{}, &.{}, null, &.{}));
    try std.testing.expectEqual(@as(u32, 1), d.written); // held 140 > 0.0001 of 2000: written early, kept in memory
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    _ = d.f.pass(&s, line(&b2, 2, 41), try s.begin(a, line(&b2, 2, 41), 0, &.{}, &.{}, null, &.{}));
    try std.testing.expectEqual(@as(u32, 2), d.written);
    s.policy.spill_highwater = 0;
    _ = d.f.pass(&s, line(&b3, 3, 41), try s.begin(a, line(&b3, 3, 41), 0, &.{}, &.{}, null, &.{}));
    try std.testing.expectEqual(@as(u32, 2), d.written);
    try std.testing.expectEqual(@as(u32, 1), s.flush(std.time.ns_per_s)); // only chat 3's was not stored
    try std.testing.expectEqual(@as(u32, 3), d.written);
}

test "without shared_least the host's system-block cut keeps no state (TF_GLM_SHARED_PREFIX=0)" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 4, .min_prompt = 0, .min_gap = 4 }, 1 << 20);
    defer s.deinit();
    s.policy = recipe(8);
    var b1: [64]u32 = undefined;
    const plan = try s.begin(a, line(&b1, 1, 37), 36, &.{16}, &.{}, null, &.{});
    try std.testing.expectEqualSlices(u32, &.{36}, plan.marks);
}

test "only a longer state of the same conversation supersedes one: another conversation's longer state with the same tokens does not" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var d: Disky = .{ .f = .{ .gpa = gpa } };
    var s = diskStore(&d, .{ .recipe = true, .entries = 3, .spill = true });
    defer s.deinit();
    var bufs: [7][96]u32 = undefined;
    const b60 = line(&bufs[0], 1, 61);
    _ = d.f.pass(&s, b60, try s.begin(a, b60, 0, &.{}, &.{}, null, &.{})); // chat B at 60
    const a40 = line(&bufs[1], 1, 41); // the same tokens as far as they go; B's 60 is past this prompt: chat A at 40
    _ = d.f.pass(&s, a40, try s.begin(a, a40, 0, &.{}, &.{}, null, &.{}));
    const a48 = line(&bufs[2], 1, 51); // resumes A's 40 and keeps 48
    const plan = try s.begin(a, a48, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 40), plan.from);
    _ = d.f.pass(&s, a48, plan);
    const b_again = line(&bufs[3], 1, 64); // B's 60 resumes: the most recently used now
    const plan_b = try s.begin(a, b_again, 0, &.{}, &.{}, null, &.{});
    try std.testing.expectEqual(@as(u32, 60), plan_b.from);
    _ = d.f.pass(&s, b_again, plan_b);
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    const dd = line(&bufs[4], 9, 41);
    _ = d.f.pass(&s, dd, try s.begin(a, dd, 0, &.{}, &.{}, null, &.{})); // A's 40 goes: superseded by A's 48, not written
    try std.testing.expectEqual(@as(u32, 0), d.written);
    const ee = line(&bufs[5], 8, 41);
    _ = d.f.pass(&s, ee, try s.begin(a, ee, 0, &.{}, &.{}, null, &.{})); // A's 48, the oldest, goes: B's 60 has its tokens but is another chat
    try std.testing.expectEqual(@as(u32, 1), d.written);
}

test "the state a pass just kept is never the one its own caps drop, and a state read back from disk starts a conversation of its own" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .planned = true, .grid = 4, .min_prompt = 0, .min_gap = 4 }, 1 << 20);
    defer s.deinit();
    s.policy = recipe(0);
    var bufs: [4][96]u32 = undefined;
    const p60 = line(&bufs[0], 1, 61); // a conversation: 60
    _ = f.pass(&s, p60, try s.begin(a, p60, 0, &.{}, &.{}, null, &.{}));
    s.policy.entries = 3;
    const p48 = line(&bufs[1], 1, 51); // shorter than the 60 and resuming nothing (the 60 is past it): its own chat
    const plan = try s.begin(a, p48, 0, &.{}, &.{}, null, &.{});
    _ = f.pass(&s, p48, plan);
    // a conversation whose newest state is shorter than an older one of its chat (a retry that branches earlier)
    var e60: ?*pc.Entry = null;
    var e48: ?*pc.Entry = null;
    for (s.entries.items) |e| {
        if (e.at == 60) e60 = e;
        if (e.at == 48) e48 = e;
    }
    e48.?.chat = e60.?.chat; // as a resume of the 60's line would have it
    s.policy.entries = 1;
    const p52 = line(&bufs[2], 2, 53); // a third chat; the cap of 1 drops the others, never the state just kept
    _ = f.pass(&s, p52, try s.begin(a, p52, 0, &.{}, &.{}, null, &.{}));
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    try std.testing.expectEqual(@as(u32, 52), s.entries.items[0].at);
}
