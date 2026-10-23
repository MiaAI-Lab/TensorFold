//! The prompt cache's durable tier (wojo's GLM 0110, as the DeepSeek port's core has it): a fake in-memory tier, and a
//! family whose state is a running hash of the prompt, so a restored state equals a fresh pass exactly when it should.
//! (0111's continuation cases are prompt_cache_chat_test.zig's.)
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
const FakeTier = @import("prompt_cache_fake_tier.zig").FakeTier;
const Allocator = std.mem.Allocator;
const Store = pc.Store;
const gpa = std.testing.allocator;

/// A family over host memory: its live state is a position and a running hash of the prompt's tokens.
const Sum = struct {
    at: u32 = 0,
    sum: u64 = 0,
    live: usize = 0,
    stale: ?pc.Saved = null, // `usable` refuses this state

    const State = struct { at: u32, sum: u64 };

    fn snapshots(f: *Sum) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytesFn, .save = saveFn, .restore = restoreFn, .drop = dropFn, .usable = usableFn } };
    }
    fn of(ptr: *anyopaque) *Sum {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytesFn(_: *anyopaque, at: u32) u64 {
        return 100 + @as(u64, at);
    }
    fn saveFn(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f = of(ptr);
        if (f.at != at) return error.NotAtMark;
        const st = try gpa.create(State);
        st.* = .{ .at = f.at, .sum = f.sum };
        f.live += 1;
        return st;
    }
    fn restoreFn(ptr: *anyopaque, _: ?*anyopaque, saved: pc.Saved) anyerror!void {
        const st: *State = @ptrCast(@alignCast(saved));
        of(ptr).at = st.at;
        of(ptr).sum = st.sum;
    }
    fn dropFn(ptr: *anyopaque, saved: pc.Saved) void {
        of(ptr).live -= 1;
        gpa.destroy(@as(*State, @ptrCast(@alignCast(saved))));
    }
    fn usableFn(ptr: *anyopaque, saved: pc.Saved, _: u32) bool {
        return of(ptr).stale != saved;
    }
    /// The live state run on to `to` tokens of `prompt` (a fresh pass's arithmetic).
    fn feed(f: *Sum, prompt: []const u32, to: u32) void {
        for (prompt[f.at..to]) |t| f.sum = f.sum *% 31 +% t;
        f.at = to;
    }
    fn reset(f: *Sum) void {
        f.at = 0;
        f.sum = 0;
    }
    fn fresh(prompt: []const u32, to: u32) u64 {
        var x: Sum = .{};
        x.feed(prompt, to);
        return x.sum;
    }

    fn codec(f: *Sum) @import("prompt_cache_fake_tier.zig").Codec {
        return .{ .ctx = f, .pack = pack, .unpack = unpack };
    }
    fn pack(_: *anyopaque, a: Allocator, saved: pc.Saved) anyerror![]u8 {
        return a.dupe(u8, std.mem.asBytes(@as(*State, @ptrCast(@alignCast(saved)))));
    }
    fn unpack(ctx: *anyopaque, bytes: []const u8) anyerror!pc.Saved {
        if (bytes.len != @sizeOf(State)) return error.SizeMismatch;
        const st = try gpa.create(State);
        @memcpy(std.mem.asBytes(st), bytes);
        of(ctx).live += 1;
        return st;
    }
};

fn seq(comptime n: usize, base: u32) [n]u32 {
    var out: [n]u32 = undefined;
    for (&out, 0..) |*t, i| t.* = base + @as(u32, @intCast(i));
    return out;
}

/// Keep `prompt`'s state at `at` as a pass would: the family stands there.
fn keepAt(s: *Store, f: *Sum, prompt: []const u32, at: u32, info: pc.KeepInfo) bool {
    return keepSpans(s, f, prompt, at, &.{}, info);
}

/// `keepAt` for a request that decoded the rows `spans` names.
fn keepSpans(s: *Store, f: *Sum, prompt: []const u32, at: u32, spans: []const modes.Span, info: pc.KeepInfo) bool {
    f.reset();
    f.feed(prompt, at);
    return s.keepWith(prompt, at, null, &.{}, spans, info);
}

/// Poll `r` until it is no longer pending.
fn settle(s: *Store, r: *pc.Restore) pc.Restored {
    while (true) {
        const st = s.restorePoll(r, 1000);
        if (st != .pending) return st;
    }
}

test "without a tier the store asks nothing and counts nothing new" {
    var f: Sum = .{};
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    const p = seq(40, 1);
    try std.testing.expect(keepAt(&s, &f, &p, 16, .{}));
    const pr = s.probe(&p, &.{}, &.{});
    try std.testing.expectEqual(@as(u32, 16), pr.memory.?.at);
    try std.testing.expect(pr.durable == null);
    try std.testing.expectEqual(pc.TierStats{}, s.tier_stats);
    try std.testing.expect(s.entries.items[0].point == null and s.entries.items[0].lineage.len == 0);
    try std.testing.expectEqual(@as(u64, 0), s.tierFlush(0));
    try std.testing.expect(s.tierCounts() == null and !s.busy());
    try std.testing.expectEqual(@as(usize, 0), (try s.passLineage(null)).len);
    try std.testing.expectEqual(@as(u64, 1 << 20) - 116, s.room());
}

test "a keep enqueues its write with conversation and lineage; a state that left memory restores, equal to a fresh pass" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec(), .polls = 3 };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 150); // room for one state
    defer s.deinit();
    s.setTier(tier.tier(), 7);
    const a = seq(40, 1);
    const chat = s.join(&a, null, true).chat;
    const g = s.groupOf(chat);
    try std.testing.expectEqual(@as(u64, 7 << 32) + 1, g); // the process's seed above the chat id
    try std.testing.expect(keepAt(&s, &f, &a, 16, .{ .chat = chat }));
    try std.testing.expectEqual(@as(usize, 1), tier.queue.items.len);
    try std.testing.expectEqual(g, tier.queue.items[0].group);
    try std.testing.expectEqual(@as(?u64, null), tier.queue.items[0].parent);
    s.pump(1000);
    try std.testing.expectEqual(@as(usize, 1), tier.points.items.len);
    const b = seq(40, 1000); // another conversation's state takes the only room
    try std.testing.expect(keepAt(&s, &f, &b, 16, .{}));
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    var a2: [48]u32 = undefined;
    @memcpy(a2[0..40], &a);
    for (a2[40..], 0..) |*t, i| t.* = 500 + @as(u32, @intCast(i));
    const pr = s.probe(&a2, &.{}, &.{});
    try std.testing.expect(pr.memory == null);
    const p = pr.durable.?;
    try std.testing.expectEqual(@as(u32, 16), p.at);
    try std.testing.expectEqual(g, p.group);
    const r = try s.restoreBegin(p, &a2);
    try std.testing.expectEqual(pc.Restored.pending, s.restorePoll(r, 1000));
    try std.testing.expectEqual(pc.Restored.adopted, settle(&s, r));
    try std.testing.expectEqual(@as(usize, 0), s.restoring.items.len);
    const e = s.find(&a2, &.{}, &.{}).?; // the restored state, `b`'s evicted for its room
    // no conversation's own state: a conversation of its own, as v1 reads one back
    try std.testing.expect(!e.own and e.point.? == p.id and !e.shared and e.chat != chat and e.chat != 0);
    try s.family.vtable.restore(s.family.ptr, null, e.saved);
    f.feed(&a2, 48);
    try std.testing.expectEqual(Sum.fresh(&a2, 48), f.sum);
    s.resumed(e, &a2, true);
    try std.testing.expectEqual(@as(u64, 1), tier.touched);
    // a later keep of this pass forks the restored point's lineage at the resume point
    try std.testing.expect(keepAt(&s, &f, &a2, 32, .{ .chat = chat, .parent = e.point, .fork = 16 }));
    try std.testing.expectEqual(p.id, tier.queue.items[0].parent.?);
    try std.testing.expectEqual(@as(u32, 16), tier.queue.items[0].fork);
    try std.testing.expectEqual(@as(u64, 1), s.tier_stats.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 16), s.tier_stats.restored_tokens);
    try std.testing.expectEqual(@as(u64, 1), s.tier_stats.restores_begun);
    try std.testing.expectEqual(@as(u64, 1), s.tier_stats.disk_hits);
    try std.testing.expectEqual(@as(u64, 3), s.tier_stats.writes_enqueued);
}

test "a stale long memory state never shadows a durable point; points off the grid, the starts or the rows left are not offered" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec() };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .planned = true, .grid = 8 }, 1 << 20);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(64, 1);
    try std.testing.expect(keepAt(&s, &f, &a, 16, .{}));
    try std.testing.expect(keepAt(&s, &f, &a, 32, .{}));
    s.pump(0);
    try std.testing.expect(keepAt(&s, &f, &a, 48, .{})); // memory only (never pumped)
    // the 48 state goes stale: the lookup takes 32 from memory, and the durable 48 is not on disk yet
    f.stale = s.entries.items[2].saved;
    var pr = s.probe(&a, &.{}, &.{});
    try std.testing.expectEqual(@as(u32, 32), pr.memory.?.at);
    try std.testing.expect(pr.durable == null);
    try std.testing.expectEqual(@as(u32, 8), tier.wanted.?.grid); // the family's grid: its request named no starts
    try std.testing.expectEqual(@as(u32, 56), tier.wanted.?.max_at);
    s.pump(0);
    pr = s.probe(&a, &.{}, &.{});
    try std.testing.expectEqual(@as(u32, 48), pr.durable.?.at); // not shadowed by its stale memory twin
    // a prompt that ends at 48 leaves no row past it: 32 from memory, no point
    pr = s.probe(a[0..48], &.{}, &.{});
    try std.testing.expectEqual(@as(u32, 32), pr.memory.?.at);
    try std.testing.expect(pr.durable == null);
    // a lax tier offering off-grid or ineligible points: the store checks them itself
    tier.lax = true;
    _ = try tier.put(a[0..20], 20, std.mem.asBytes(&Sum.State{ .at = 20, .sum = 0 }), 0, 0, false);
    try std.testing.expect(s.probe(a[0..48], &.{}, &.{}).durable == null); // 48 leaves no row of 48
    try std.testing.expect(s.probe(a[0..30], &.{}, &.{}).durable == null); // 20 is off the grid
    // a request's chunk starts: the want carries them, and a point must stand on one
    tier.lax = false;
    const starts = [_]u32{ 12, 20, 40 };
    pr = s.probe(a[0..30], &starts, &.{});
    try std.testing.expectEqualSlices(u32, &starts, tier.wanted.?.starts);
    try std.testing.expectEqual(@as(u32, 0), tier.wanted.?.grid);
    try std.testing.expectEqual(@as(u32, 20), pr.durable.?.at);
    // the rule itself
    try std.testing.expect(!s.eligible(47, 64, &.{}) and s.eligible(48, 52, &.{}) and !s.eligible(48, 48, &.{}));
    try std.testing.expect(s.eligible(20, 21, &starts) and !s.eligible(16, 21, &starts));
    try std.testing.expectEqual(@as(?u32, 48), s.maxAt(52, &.{}));
    try std.testing.expectEqual(@as(?u32, 8), s.maxAt(9, &.{}));
    try std.testing.expectEqual(@as(?u32, null), s.maxAt(8, &.{}));
    try std.testing.expectEqual(@as(?u32, 20), s.maxAt(21, &starts));
    try std.testing.expectEqual(@as(?u32, 12), s.maxAt(20, &starts));
    var flat = Store.init(gpa, f.snapshots(), .{ .lookahead = 1 }, 1 << 20);
    defer flat.deinit();
    try std.testing.expectEqual(@as(?u32, 9), flat.maxAt(10, &.{}));
    try std.testing.expectEqual(@as(?u32, null), flat.maxAt(1, &.{}));
    var gridless = Store.init(gpa, f.snapshots(), .{ .planned = true }, 1 << 20);
    defer gridless.deinit();
    try std.testing.expectEqual(@as(?u32, null), gridless.maxAt(100, &.{})); // planned, no starts, no grid: nowhere
}

test "one restore per point: a second request joins it, and it is cancelled once" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec(), .polls = 5 };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(40, 1);
    _ = try tier.put(a[0..16], 16, std.mem.asBytes(&Sum.State{ .at = 16, .sum = Sum.fresh(&a, 16) }), 0, 3, true);
    const p = s.probe(&a, &.{}, &.{}).durable.?;
    try std.testing.expect(p.shared and p.group == 3);
    const r1 = try s.restoreBegin(p, &a);
    const r2 = try s.restoreBegin(p, a[0..30]);
    try std.testing.expectEqual(r1, r2);
    try std.testing.expectEqual(@as(u64, 1), s.tier_stats.restores_shared);
    try std.testing.expectEqual(@as(usize, 1), tier.jobs.items.len);
    s.restoreCancel(r1);
    try std.testing.expectEqual(@as(usize, 0), tier.jobs.items.len);
    try std.testing.expectEqual(@as(u64, 1), tier.cancels);
    try std.testing.expectEqual(@as(u64, 0), tier.misuse);
    try std.testing.expectEqual(@as(u64, 1), s.tier_stats.restores_cancelled);
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
    // an adopted shared point is a shared entry: other conversations resume it, its own never supersedes it
    const r3 = try s.restoreBegin(p, &a);
    try std.testing.expectEqual(pc.Restored.adopted, settle(&s, r3));
    try std.testing.expect(s.entries.items[0].shared and !s.entries.items[0].own);
    try std.testing.expectEqual(@as(usize, 1), f.live);
}

test "a restore that fails anywhere leaves nothing; one memory already holds is dropped, its copy freed" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec() };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(40, 1);
    _ = try tier.put(a[0..16], 16, std.mem.asBytes(&Sum.State{ .at = 16, .sum = Sum.fresh(&a, 16) }), 0, 0, false);
    const p = s.probe(&a, &.{}, &.{}).durable.?;
    tier.fail_next = error.CrcMismatch;
    try std.testing.expectEqual(pc.Restored.failed, settle(&s, try s.restoreBegin(p, &a)));
    try std.testing.expectEqual(@as(u64, 1), s.tier_stats.restores_failed);
    try std.testing.expectEqual(@as(u64, 1), tier.c.crc_errors);
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
    const r = try s.restoreBegin(p, &a);
    try std.testing.expect(keepAt(&s, &f, &a, 16, .{})); // a pass kept the same state meanwhile
    try std.testing.expectEqual(pc.Restored.dropped, settle(&s, r));
    try std.testing.expectEqual(@as(u64, 1), s.tier_stats.restores_dropped);
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    try std.testing.expectEqual(@as(usize, 1), f.live);
    // too large for the whole budget: never offered
    s.budget = 100;
    try std.testing.expect(s.probe(&a, &.{}, &.{}).durable == null);
}

test "a stale memory twin of a durable point is replaced by its restore; an evicted state restores too" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec(), .polls = 2 };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(40, 1);
    try std.testing.expect(keepAt(&s, &f, &a, 16, .{}));
    s.pump(1000); // on disk too
    // the family stales the kept state (KV sharding gave its rows to a placement): the entry stays in memory
    f.stale = s.entries.items[0].saved;
    const pr = s.probe(&a, &.{}, &.{});
    try std.testing.expect(pr.memory == null); // never resumed
    const p = pr.durable.?;
    try std.testing.expectEqual(@as(u32, 16), p.at);
    // the restore of the same point replaces the stale twin instead of being dropped (a recompute)
    try std.testing.expectEqual(pc.Restored.adopted, settle(&s, try s.restoreBegin(p, &a)));
    try std.testing.expectEqual(@as(u64, 1), s.tier_stats.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 0), s.tier_stats.restores_dropped);
    try std.testing.expectEqual(@as(usize, 1), s.entries.items.len);
    try std.testing.expect(!s.entries.items[0].own and s.entries.items[0].saved != f.stale.?);
    try std.testing.expectEqual(@as(usize, 1), f.live); // the stale state was given back to the family
    const e = s.find(&a, &.{}, &.{}).?;
    try s.family.vtable.restore(s.family.ptr, null, e.saved);
    f.feed(&a, 40);
    try std.testing.expectEqual(Sum.fresh(&a, 40), f.sum); // the restored state is the prompt's
    f.stale = null;
    // a usable copy kept while the restore ran still wins: that restore is dropped, its copy freed
    const c = seq(40, 2000);
    try std.testing.expect(keepAt(&s, &f, &c, 16, .{}));
    s.pump(1000);
    f.stale = s.entries.items[s.entries.items.len - 1].saved;
    const pcp = s.probe(&c, &.{}, &.{}).durable.?;
    const r = try s.restoreBegin(pcp, &c);
    f.stale = null; // (the same state usable again: as if a pass kept it anew meanwhile)
    try std.testing.expectEqual(pc.Restored.dropped, settle(&s, r));
    try std.testing.expectEqual(@as(u64, 1), s.tier_stats.restores_dropped);
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    try std.testing.expectEqual(@as(usize, 2), f.live);
    // an evicted state (its entry gone from memory for another's room): its restore is adopted, nothing dropped
    var g: Sum = .{};
    var tier2: FakeTier = .{ .gpa = gpa, .codec = g.codec(), .polls = 2 };
    defer tier2.deinit();
    var s2 = Store.init(gpa, g.snapshots(), .{ .min_prompt = 0 }, 150); // room for one state
    defer s2.deinit();
    s2.setTier(tier2.tier(), 0);
    const b = seq(40, 1000);
    try std.testing.expect(keepAt(&s2, &g, &b, 16, .{}));
    s2.pump(1000);
    try std.testing.expect(keepAt(&s2, &g, &c, 16, .{})); // evicts b's
    try std.testing.expectEqual(@as(u64, 1), s2.counts.evicted);
    const pb = s2.probe(&b, &.{}, &.{}).durable.?;
    try std.testing.expectEqual(pc.Restored.adopted, settle(&s2, try s2.restoreBegin(pb, &b)));
    try std.testing.expectEqual(@as(u64, 1), s2.tier_stats.disk_hits_committed);
    try std.testing.expectEqual(@as(u64, 0), s2.tier_stats.restores_dropped);
}

test "a write still queued is staged before its memory state is dropped; a new shared prefix is marked on disk" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec() };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .min_gap = 4 }, 150);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(40, 1);
    const b = seq(40, 900);
    try std.testing.expect(keepAt(&s, &f, &a, 16, .{}));
    try std.testing.expect(keepAt(&s, &f, &b, 16, .{})); // evicts a's state before its write was staged
    try std.testing.expectEqual(@as(u64, 1), tier.c.forced_staging);
    try std.testing.expectEqual(@as(usize, 1), tier.points.items.len);
    const st: *const Sum.State = @ptrCast(@alignCast(tier.points.items[0].bytes.ptr));
    try std.testing.expectEqual(Sum.fresh(&a, 16), st.sum); // the bytes it wrote are a's, staged in time
    // b's state turns out to be a shared cut: kept again, it is promoted on disk (made stale first so the lookup plans it)
    f.stale = s.entries.items[0].saved;
    s.budget = 1 << 20; // (room for the lookup's marks: it evicts nothing)
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    _ = try s.lookup(arena.allocator(), &b, 30, &.{16}, &.{}, &.{});
    f.stale = null;
    try std.testing.expect(keepAt(&s, &f, &b, 16, .{}));
    try std.testing.expect(s.entries.items[0].shared);
    try std.testing.expectEqual(@as(u64, 1), tier.shared_marks);
    try std.testing.expectEqual(@as(u64, 1), s.tier_stats.shared_promoted);
    try std.testing.expect(tier.queue.items[0].shared);
}

test "a shared cut on a planned family's grid is kept shared, on disk too" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec() };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .min_gap = 4, .planned = true, .grid = 8 }, 1 << 20);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const p = seq(40, 1);
    const l = try s.lookup(arena.allocator(), &p, 30, &.{21}, &.{}, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 16, 24 }, l.marks); // both on the grid
    try std.testing.expectEqual(@as(usize, 1), s.shared_keys.items.len);
    try std.testing.expect(keepAt(&s, &f, &p, 16, .{}));
    try std.testing.expect(s.entries.items[0].shared and !s.entries.items[0].own);
    try std.testing.expect(tier.queue.items[0].shared);
}

test "flush stops restores, drains writes, and reports what its deadline left unpublished" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec(), .polls = 100 };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(40, 1);
    const b = seq(40, 300);
    try std.testing.expect(keepAt(&s, &f, &a, 16, .{}));
    try std.testing.expect(s.busy());
    try std.testing.expectEqual(@as(u64, 0), s.tierFlush(1000));
    try std.testing.expect(!s.busy());
    try std.testing.expectEqual(@as(usize, 1), tier.points.items.len);
    const c = seq(40, 600);
    _ = try tier.put(c[0..16], 16, std.mem.asBytes(&Sum.State{ .at = 16, .sum = Sum.fresh(&c, 16) }), 0, 0, false);
    _ = try s.restoreBegin(s.probe(&c, &.{}, &.{}).durable orelse return error.TestUnexpectedResult, &c);
    try std.testing.expect(keepAt(&s, &f, &b, 16, .{}));
    tier.stuck = true;
    try std.testing.expectEqual(@as(u64, 1), s.tierFlush(1000));
    try std.testing.expectEqual(@as(usize, 0), s.restoring.items.len);
    try std.testing.expectEqual(@as(u64, 1), tier.cancels);
    try std.testing.expectEqual(@as(u64, 1), s.tier_stats.flush_timeouts);
    try std.testing.expectEqual(@as(u64, 1), s.tier_stats.flush_unpublished);
    try std.testing.expectEqual(@as(u64, 2), s.tier_stats.flushes);
    try std.testing.expectEqual(@as(u32, 0), s.flush(1000)); // (v1's flush: off under a tier)
}

test "forgetting a point drops it on disk, or cancels its write, and unlinks its memory twin" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec() };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(40, 1);
    try std.testing.expect(keepAt(&s, &f, &a, 16, .{}));
    s.pump(0);
    const id = s.entries.items[0].point.?;
    s.forgetPoint(id);
    try std.testing.expect(s.entries.items[0].point == null);
    try std.testing.expect(tier.point(id) == null);
    try std.testing.expectEqual(@as(u64, 0), tier.released);
    try std.testing.expect(keepAt(&s, &f, &a, 32, .{})); // still queued
    s.forgetPoint(s.entries.items[1].point.?);
    try std.testing.expectEqual(@as(usize, 0), tier.queue.items.len);
    try std.testing.expectEqual(@as(u64, 1), tier.c.writes_cancelled);
}

/// Sum with spare storage: a dropped state's bytes stay spare, a save takes spare first, trim frees it.
const Pool = struct {
    sum: Sum = .{},
    spare_: u64 = 0,

    fn snapshots(p: *Pool) pc.Snapshots {
        return .{ .ptr = p, .vtable = &.{ .bytes = Sum.bytesFn, .save = save, .restore = restore, .drop = drop, .spare = spare, .trim = trim, .reuses = reuses } };
    }
    fn of(ptr: *anyopaque) *Pool {
        return @ptrCast(@alignCast(ptr));
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const p = of(ptr);
        if (p.spare_ >= Sum.bytesFn(ptr, at)) p.spare_ -= Sum.bytesFn(ptr, at);
        return Sum.saveFn(&p.sum, owner, at);
    }
    fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: pc.Saved) anyerror!void {
        return Sum.restoreFn(&of(ptr).sum, owner, saved);
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const st: *Sum.State = @ptrCast(@alignCast(saved));
        of(ptr).spare_ += Sum.bytesFn(ptr, st.at);
        Sum.dropFn(&of(ptr).sum, saved);
    }
    fn spare(ptr: *anyopaque) u64 {
        return of(ptr).spare_;
    }
    fn trim(ptr: *anyopaque, room: u64) void {
        of(ptr).spare_ = @min(of(ptr).spare_, room);
    }
    fn reuses(ptr: *anyopaque, at: u32) bool {
        return of(ptr).spare_ >= Sum.bytesFn(ptr, at);
    }
};

test "a restore's destination is charged to the budget while it runs; spare storage in its way is trimmed, not every state" {
    var pool: Pool = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = pool.sum.codec(), .polls = 2 };
    defer tier.deinit();
    var s = Store.init(gpa, pool.snapshots(), .{ .min_prompt = 0 }, 400); // states of 100 + at bytes
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(40, 1);
    const b = seq(40, 300);
    const c = seq(40, 600);
    pool.sum.reset();
    pool.sum.feed(&a, 10);
    try std.testing.expect(s.keepWith(&a, 10, null, &.{}, &.{}, .{}));
    pool.sum.reset();
    pool.sum.feed(&b, 10);
    try std.testing.expect(s.keepWith(&b, 10, null, &.{}, &.{}, .{}));
    pool.sum.reset();
    pool.sum.feed(&c, 10);
    try std.testing.expect(s.keepWith(&c, 10, null, &.{}, &.{}, .{}));
    try std.testing.expectEqual(@as(u64, 330), s.held);
    const d = seq(40, 900);
    _ = try tier.put(d[0..20], 20, std.mem.asBytes(&Sum.State{ .at = 20, .sum = Sum.fresh(&d, 20) }), 0, 0, false);
    const p = s.probe(&d, &.{}, &.{}).durable.?;
    const r = try s.restoreBegin(p, &d); // 120 bytes: one state goes, its spare storage is trimmed to fit
    try std.testing.expectEqual(@as(u64, 120), s.restoring_bytes);
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    try std.testing.expect(s.held + s.restoring_bytes + s.spare() <= s.budget);
    try std.testing.expectEqual(@as(u64, 400 - 220 - 120 - s.spare()), s.room());
    // a second restore the budget cannot hold beside the first and the states it must keep: refused, nothing begun
    const e2 = seq(400, 2000);
    _ = try tier.put(e2[0..300], 300, std.mem.asBytes(&Sum.State{ .at = 300, .sum = 0 }), 0, 0, false);
    const p2 = s.probe(&e2, &.{}, &.{}).durable.?;
    try std.testing.expectError(error.NoRoom, s.restoreBegin(p2, &e2));
    try std.testing.expectEqual(@as(u64, 1), s.tier_stats.restores_refused);
    try std.testing.expectEqual(@as(usize, 1), s.restoring.items.len);
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len); // refused before evicting anything
    // adopted beside both remaining states: no eviction past what the reservation already made
    try std.testing.expectEqual(pc.Restored.adopted, settle(&s, r));
    try std.testing.expectEqual(@as(u64, 0), s.restoring_bytes);
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    try std.testing.expect(s.held + s.spare() <= s.budget);
    // a cancelled restore gives its charge back (a's state was staged to disk when it was evicted)
    const r3 = try s.restoreBegin(s.probe(&a, &.{}, &.{}).durable.?, &a);
    try std.testing.expectEqual(@as(u64, 110), s.restoring_bytes);
    s.restoreCancel(r3);
    try std.testing.expectEqual(@as(u64, 0), s.restoring_bytes);
    try std.testing.expectEqual(s.entries.items.len, pool.sum.live);
}

test "a keep the budget cannot hold beside the restores in flight is refused, nothing copied; it fits once they end" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec(), .polls = 1 << 20 };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 400); // states of 100 + at bytes
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const d = seq(40, 900);
    _ = try tier.put(d[0..20], 20, std.mem.asBytes(&Sum.State{ .at = 20, .sum = Sum.fresh(&d, 20) }), 0, 0, false);
    const r = try s.restoreBegin(s.probe(&d, &.{}, &.{}).durable.?, &d); // 120 of 400 in flight
    try std.testing.expectEqual(@as(u64, 400 - 120), s.room());
    const big = seq(300, 5000);
    try std.testing.expect(!keepAt(&s, &f, &big, 200, .{})); // 300 bytes beside the 120
    try std.testing.expectEqual(@as(u64, 1), s.counts.refused);
    try std.testing.expectEqual(@as(usize, 0), f.live);
    try std.testing.expectEqual(@as(usize, 0), tier.queue.items.len);
    s.restoreCancel(r);
    try std.testing.expect(keepAt(&s, &f, &big, 200, .{}));
}

test "lineage: a fresh pass starts its own, a resumed one forks its state's at the resume point, kept states carry it to disk and back" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec(), .ranks = 2 };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 150); // room for one state
    defer s.deinit();
    s.setTier(tier.tier(), 3);
    const a = seq(48, 1);
    const fresh = try s.passLineage(null);
    defer gpa.free(fresh);
    try std.testing.expectEqual(@as(usize, 1), fresh.len);
    try std.testing.expectEqual(@as(u32, 0), fresh[0].at);
    try std.testing.expectEqual(@as(u64, 3) << 32, fresh[0].name >> 32 << 32);
    try std.testing.expect(keepAt(&s, &f, &a, 16, .{ .lineage = fresh })); // below any write minimum it is still carried
    const e = s.entries.items[0];
    try std.testing.expectEqualSlices(pc.Fork, fresh, e.lineage);
    try std.testing.expectEqualSlices(pc.Fork, fresh, tier.queue.items[0].lineage);
    const forked = try s.passLineage(e);
    defer gpa.free(forked);
    try std.testing.expectEqual(@as(usize, 2), forked.len);
    try std.testing.expectEqual(fresh[0], forked[0]);
    try std.testing.expectEqual(@as(u32, 16), forked[1].at);
    try std.testing.expect(forked[1].name != fresh[0].name);
    // its state leaves memory (staged to disk first) and comes back with its lineage
    const b = seq(40, 900);
    try std.testing.expect(keepAt(&s, &f, &b, 16, .{}));
    const r = try s.restoreBegin(s.probe(&a, &.{}, &.{}).durable.?, &a);
    try std.testing.expectEqual(pc.Restored.adopted, settle(&s, r));
    try std.testing.expectEqualSlices(pc.Fork, fresh, s.find(&a, &.{}, &.{}).?.lineage);
    // without a tier there is no lineage to carry
    var plain = Store.init(gpa, f.snapshots(), .{}, 1 << 20);
    defer plain.deinit();
    try std.testing.expectEqual(@as(usize, 0), (try plain.passLineage(null)).len);
    // every rank's counters
    const rc = s.tierCounts().?;
    try std.testing.expectEqual(@as(u32, 2), rc.n);
    try std.testing.expectEqual(@as(u64, 1), rc.slice()[1].crc_errors);
}

test "a restore refused for room or capacity: the shorter memory state stays when it can, a restore that can start wins" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec() };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20); // states of 100 + at bytes
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(48, 1);
    try std.testing.expect(keepAt(&s, &f, &a, 10, .{})); // memory: 110 bytes at 10
    _ = try tier.put(a[0..30], 30, std.mem.asBytes(&Sum.State{ .at = 30, .sum = Sum.fresh(&a, 30) }), 0, 0, false);
    const pr = s.probe(&a, &.{}, &.{});
    const hit = pr.memory.?;
    const p = pr.durable.?;
    // the tier full before any room is made: refused with nothing evicted and nothing asked of begin, counted once a
    // request however often it asks again
    tier.full = 2;
    try std.testing.expectError(error.TierFull, s.restoreBeginOver(p, &a, .{ .have = hit }));
    try std.testing.expectError(error.TierFull, s.restoreBeginOver(p, &a, .{ .have = hit, .again = true }));
    try std.testing.expect(s.deferrable(error.TierFull));
    try std.testing.expectEqual(@as(u64, 1), s.tier_stats.restores_refused);
    try std.testing.expectEqual(@as(u64, 0), tier.begins);
    try std.testing.expectEqual(hit, s.find(&a, &.{}, &.{}).?); // the pass resumes at 10, not 0
    // room for both: the tier's own capacity refusal at begin leaves the hit too
    tier.refuse = 1;
    try std.testing.expectError(error.TooManyRestores, s.restoreBeginOver(p, &a, .{ .have = hit }));
    try std.testing.expect(s.deferrable(error.TooManyRestores));
    for ([_]anyerror{ error.TooMany, error.NoSlot, error.Busy }) |e| try std.testing.expect(s.deferrable(e));
    try std.testing.expect(!s.deferrable(error.PointMissing));
    try std.testing.expectEqual(@as(usize, 0), s.restoring.items.len);
    try std.testing.expectEqual(@as(u64, 0), s.restoring_bytes);
    try std.testing.expectEqual(hit, s.find(&a, &.{}, &.{}).?);
    // a budget that holds the restore (130) but not both (240): the restore starts, evicting the shorter state as
    // restoreBegin always did (a short memory hit never costs a long restore)
    s.budget = 200;
    const r = try s.restoreBeginOver(p, &a, .{ .have = hit });
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
    try std.testing.expectEqual(pc.Restored.adopted, settle(&s, r));
    try std.testing.expectEqual(@as(u32, 30), s.find(&a, &.{}, &.{}).?.at);
}

test "a restore the tier refuses for its point evicts no kept state for it; one the tier accepts still makes its room" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec() };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 250); // states of 100 + at bytes
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const d = seq(48, 1);
    const b = seq(48, 300);
    try std.testing.expect(keepAt(&s, &f, &d, 10, .{})); // the prompt's own memory state: 110 bytes at 10
    try std.testing.expect(keepAt(&s, &f, &b, 10, .{})); // another conversation's: 220 of 250 held
    _ = try tier.put(d[0..30], 30, std.mem.asBytes(&Sum.State{ .at = 30, .sum = Sum.fresh(&d, 30) }), 0, 0, false);
    const pr = s.probe(&d, &.{}, &.{});
    const hit = pr.memory.?;
    const p = pr.durable.?; // 130 bytes: beside the hit, b's state must go to make its room
    // refused for the point itself (GLM's KvShardHomeFull: its home part full, no carry), first and asked again: no
    // kept state evicted, nothing begun or charged
    tier.unfit = 2;
    try std.testing.expectError(error.KvShardHomeFull, s.restoreBeginOver(p, &d, .{ .have = hit }));
    try std.testing.expectError(error.KvShardHomeFull, s.restoreBeginOver(p, &d, .{ .have = hit, .again = true }));
    try std.testing.expectEqual(@as(u64, 2), tier.refused);
    try std.testing.expectEqual(@as(u64, 0), s.counts.evicted);
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    try std.testing.expectEqual(@as(u64, 220), s.held);
    try std.testing.expect(s.find(&b, &.{}, &.{}) != null);
    try std.testing.expectEqual(hit, s.find(&d, &.{}, &.{}).?); // the pass resumes at 10
    try std.testing.expectEqual(@as(usize, 0), s.restoring.items.len);
    try std.testing.expectEqual(@as(u64, 0), s.restoring_bytes);
    try std.testing.expectEqual(@as(u64, 0), tier.begins); // refused before begin
    try std.testing.expectEqual(@as(u64, 0), s.tier_stats.restores_refused); // (a refusal of the point, as begin's)
    // accepted: its room is made as before (b's state evicted, the hit spared beside it), and it restores
    const r = try s.restoreBeginOver(p, &d, .{ .have = hit });
    try std.testing.expectEqual(@as(u64, 1), s.counts.evicted);
    try std.testing.expect(s.find(&b, &.{}, &.{}) == null);
    try std.testing.expectEqual(hit, s.find(&d, &.{}, &.{}).?);
    try std.testing.expectEqual(@as(u64, 130), s.restoring_bytes);
    try std.testing.expect(s.held + s.restoring_bytes <= s.budget);
    try std.testing.expectEqual(pc.Restored.adopted, settle(&s, r));
    try std.testing.expectEqual(@as(u32, 30), s.find(&d, &.{}, &.{}).?.at);
}

test "a NoRoom is worth waiting for only when the restores in flight hold the rest of the budget" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec() };
    defer tier.deinit();
    tier.polls = 1 << 20; // restores stay in flight
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 300);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(48, 1);
    const b = seq(48, 101);
    _ = try tier.put(a[0..30], 30, std.mem.asBytes(&Sum.State{ .at = 30, .sum = Sum.fresh(&a, 30) }), 0, 0, false);
    _ = try tier.put(b[0..40], 40, std.mem.asBytes(&Sum.State{ .at = 40, .sum = Sum.fresh(&b, 40) }), 0, 0, false);
    const ra = try s.restoreBeginOver(s.probe(&a, &.{}, &.{}).durable.?, &a, .{}); // 130 of 300 in flight
    // b's 140 beside a's 130 fit; with the budget at 250 they do not, until a's ends
    s.budget = 250;
    try std.testing.expectError(error.NoRoom, s.restoreBeginOver(s.probe(&b, &.{}, &.{}).durable.?, &b, .{}));
    try std.testing.expect(s.deferrable(error.NoRoom));
    try std.testing.expectEqual(@as(u64, 1), s.tier_stats.restores_refused);
    s.restoreCancel(ra);
    try std.testing.expect(!s.deferrable(error.NoRoom)); // (nothing in flight: waiting frees nothing)
    // a budget that can never hold it: never deferred (the probe already refuses such a point; asked directly here)
    const pb = s.probe(&b, &.{}, &.{}).durable.?;
    s.budget = 100;
    try std.testing.expectError(error.NoRoom, s.restoreBeginOver(pb, &b, .{}));
    try std.testing.expect(!s.deferrable(error.NoRoom));
}

test "a begin refused while restores are in flight is worth waiting for unless it names the point; with none in flight, only capacity is" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec() };
    defer tier.deinit();
    tier.polls = 1 << 20; // restores stay in flight
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(48, 1);
    const b = seq(48, 101);
    _ = try tier.put(a[0..30], 30, std.mem.asBytes(&Sum.State{ .at = 30, .sum = Sum.fresh(&a, 30) }), 0, 0, false);
    _ = try tier.put(b[0..40], 40, std.mem.asBytes(&Sum.State{ .at = 40, .sum = Sum.fresh(&b, 40) }), 0, 0, false);
    const others = [_]anyerror{ error.SpillReserve, error.BelowMemoryReserve, error.OutOfMemory };
    for (others) |e| try std.testing.expect(!s.deferrable(e)); // (nothing in flight: waiting frees nothing)
    const ra = try s.restoreBeginOver(s.probe(&a, &.{}, &.{}).durable.?, &a, .{});
    // b's begin refused for its destination's memory (GLM's SpillReserve) with a's restore in flight, nothing begun:
    // worth asking again when a ends (b waits for it rather than prefilling beside it), not once none is in flight
    tier.busy_err = error.SpillReserve;
    try std.testing.expectError(error.SpillReserve, s.restoreBeginOver(s.probe(&b, &.{}, &.{}).durable.?, &b, .{}));
    try std.testing.expectEqual(@as(usize, 1), s.restoring.items.len);
    for (others) |e| try std.testing.expect(s.deferrable(e));
    for (pc.point_errors) |e| try std.testing.expect(!s.deferrable(e)); // the point itself: no wait cures it
    for (pc.capacity_errors) |e| try std.testing.expect(s.deferrable(e));
    s.restoreCancel(ra);
    for (others) |e| try std.testing.expect(!s.deferrable(e));
    for (pc.capacity_errors) |e| try std.testing.expect(s.deferrable(e));
}

test "decoded rows: a state holding them is not written, one before them is; a request naming them never looks the tier up" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec() };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(48, 1);
    const spans = [_]modes.Span{.{ 20, 30 }};
    try std.testing.expect(keepSpans(&s, &f, &a, 16, &spans, .{})); // its rows end before the decoded ones
    try std.testing.expect(keepSpans(&s, &f, &a, 24, &spans, .{})); // it holds decoded rows 20..24
    try std.testing.expect(s.entries.items[0].point != null and s.entries.items[1].point == null);
    try std.testing.expectEqual(@as(usize, 1), tier.queue.items.len);
    try std.testing.expectEqual(@as(u64, 1), s.tier_stats.writes_enqueued);
    s.pump(0);
    // the point at 16 is longer than anything a span-less prompt finds in memory, yet a request with spans asks nothing
    const b = seq(40, 700);
    _ = try tier.put(b[0..32], 32, std.mem.asBytes(&Sum.State{ .at = 32, .sum = Sum.fresh(&b, 32) }), 0, 0, false);
    try std.testing.expect(s.probe(&b, &.{}, &.{.{ 35, 38 }}).durable == null);
    try std.testing.expectEqual(@as(u64, 0), s.tier_stats.disk_lookups);
    try std.testing.expectEqual(@as(u32, 32), s.probe(&b, &.{}, &.{}).durable.?.at);
    try std.testing.expectEqual(@as(u64, 1), s.tier_stats.disk_lookups);
}

/// Sum with the family's v1 disk tier (evict / stored / load / spill_stats), counting each call.
const V1 = struct {
    sum: Sum = .{},
    evicts: u32 = 0,
    asks: u32 = 0,
    stats: u32 = 0,

    fn snapshots(v: *V1) pc.Snapshots {
        return .{ .ptr = v, .vtable = &.{ .bytes = Sum.bytesFn, .save = save, .restore = restore, .drop = drop, .evict = evict, .stored = stored, .load = load, .spill_stats = spillStats } };
    }
    fn of(ptr: *anyopaque) *V1 {
        return @ptrCast(@alignCast(ptr));
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        return Sum.saveFn(&of(ptr).sum, owner, at);
    }
    fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: pc.Saved) anyerror!void {
        return Sum.restoreFn(&of(ptr).sum, owner, saved);
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        Sum.dropFn(&of(ptr).sum, saved);
    }
    fn evict(ptr: *anyopaque, _: pc.Saved, _: []const u32, _: u32) void {
        of(ptr).evicts += 1;
    }
    fn stored(ptr: *anyopaque, _: []const u32, _: u32, _: *u32, _: *u64, _: *[]const u32) bool {
        of(ptr).asks += 1;
        return false;
    }
    fn load(_: *anyopaque, _: u64, _: u32) anyerror!pc.Saved {
        return error.NotStored;
    }
    fn spillStats(ptr: *anyopaque, out: *[8]u64) void {
        of(ptr).stats += 1;
        out.* = @splat(1);
    }
};

/// A v1 store at the budget of one state: keeps a's then b's (evicting a's), looks a up. The family's call counts.
fn v1Run(v: *V1, s: *Store) !void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = seq(40, 1);
    const b = seq(40, 900);
    v.sum.reset();
    v.sum.feed(&a, 16);
    try std.testing.expect(s.keepWith(&a, 16, null, &.{}, &.{}, .{}));
    v.sum.reset();
    v.sum.feed(&b, 16);
    try std.testing.expect(s.keepWith(&b, 16, null, &.{}, &.{}, .{}));
    _ = try s.lookup(arena.allocator(), &a, 30, &.{}, &.{}, &.{});
}

test "a tier turns the family's v1 disk tier off: no early write, eviction write, read or flush; without one v1 runs as before" {
    var v: V1 = .{};
    var s = Store.init(gpa, v.snapshots(), .{ .min_prompt = 0 }, 150);
    defer s.deinit();
    s.policy = .{ .spill = true, .spill_highwater = 0.5 };
    try v1Run(&v, &s);
    try std.testing.expect(v.evicts > 0 and v.asks > 0);
    try std.testing.expect(s.keptStats().spill_on);
    var w: V1 = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = w.sum.codec() };
    defer tier.deinit();
    var t = Store.init(gpa, w.snapshots(), .{ .min_prompt = 0 }, 150);
    defer t.deinit();
    t.policy = .{ .spill = true, .spill_highwater = 0.5 };
    t.setTier(tier.tier(), 0);
    try std.testing.expect(!t.policy.spill);
    t.policy.spill = true; // (a host that sets the policy after the tier: still off)
    try v1Run(&w, &t);
    try std.testing.expectEqual(@as(u32, 0), w.evicts);
    try std.testing.expectEqual(@as(u32, 0), w.asks);
    try std.testing.expectEqual(@as(u32, 0), t.flush(1 << 30));
    try std.testing.expect(!t.keptStats().spill_on);
    try std.testing.expectEqual(@as(u32, 0), w.stats);
    try std.testing.expectEqual(@as(u64, 2), tier.c.writes_queued); // v2 wrote both instead (a's staged at its eviction)
}

test "clear under a tier: restores stop, kept states' points are forgotten and never staged; deinit cancels and stages" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec(), .polls = 100 };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    s.setTier(tier.tier(), 0);
    const a = seq(40, 1);
    const b = seq(40, 300);
    const c = seq(40, 600);
    try std.testing.expect(keepAt(&s, &f, &a, 16, .{}));
    s.pump(0); // a's on disk
    try std.testing.expect(keepAt(&s, &f, &b, 16, .{})); // b's still queued
    _ = try tier.put(c[0..16], 16, std.mem.asBytes(&Sum.State{ .at = 16, .sum = Sum.fresh(&c, 16) }), 0, 0, false);
    _ = try s.restoreBegin(s.probe(&c, &.{}, &.{}).durable.?, &c);
    s.clear(); // new weights: every copy of the old arithmetic goes
    try std.testing.expectEqual(@as(usize, 0), s.entries.items.len);
    try std.testing.expectEqual(@as(usize, 0), s.restoring.items.len);
    try std.testing.expectEqual(@as(u64, 0), s.restoring_bytes);
    try std.testing.expectEqual(@as(u64, 1), tier.cancels);
    try std.testing.expectEqual(@as(u64, 0), tier.released); // nothing staged from the old states
    try std.testing.expectEqual(@as(usize, 0), tier.queue.items.len);
    try std.testing.expectEqual(@as(usize, 1), tier.points.items.len); // c's, which no memory copy named
    try std.testing.expectEqual(@as(usize, 0), f.live);
    // deinit: a restore in flight is cancelled once, a queued write staged before its state goes
    var d = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    d.setTier(tier.tier(), 0);
    try std.testing.expect(keepAt(&d, &f, &a, 16, .{}));
    _ = try d.restoreBegin(d.probe(&c, &.{}, &.{}).durable.?, &c);
    d.deinit();
    try std.testing.expectEqual(@as(u64, 2), tier.cancels);
    try std.testing.expectEqual(@as(u64, 0), tier.misuse);
    try std.testing.expectEqual(@as(u64, 1), tier.c.forced_staging);
    try std.testing.expectEqual(@as(usize, 0), f.live);
}

test "a restored state is no conversation's own: per_chat counts only own turns, and its resume joins by 0111's rule" {
    var f: Sum = .{};
    var tier: FakeTier = .{ .gpa = gpa, .codec = f.codec() };
    defer tier.deinit();
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0 }, 1 << 20);
    defer s.deinit();
    s.policy = .{ .per_chat = 1 };
    s.setTier(tier.tier(), 0);
    const a = seq(48, 1);
    const b = seq(48, 900);
    _ = try tier.put(b[0..16], 16, std.mem.asBytes(&Sum.State{ .at = 16, .sum = Sum.fresh(&b, 16) }), 0, s.groupOf(1), false);
    const chat = s.join(&a, null, true).chat;
    try std.testing.expect(keepAt(&s, &f, &a, 16, .{ .chat = chat }));
    try std.testing.expectEqual(pc.Restored.adopted, settle(&s, try s.restoreBegin(s.probe(&b, &.{}, &.{}).durable.?, &b)));
    const restored = s.find(&b, &.{}, &.{}).?;
    try std.testing.expect(!restored.own and restored.chat != chat);
    try std.testing.expect(keepAt(&s, &f, &a, 32, .{ .chat = chat })); // per_chat 1: a's 16 goes, the restored one stays
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    try std.testing.expectEqual(restored, s.find(&b, &.{}, &.{}).?);
    const k = s.keptStats();
    try std.testing.expectEqual(@as(u32, 1), k.own);
    try std.testing.expectEqual(@as(u32, 1), k.shared);
    // a request resuming it is not that conversation's turn: a new conversation (nothing it continues)
    const j = s.join(&b, restored, true);
    try std.testing.expect(j.chat != restored.chat and j.chat != chat and !j.continued and j.source == .other);
}
