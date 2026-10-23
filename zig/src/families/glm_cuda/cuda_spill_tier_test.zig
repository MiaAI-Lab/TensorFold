//! Spill v2's GLM tier on two ranks in one thread (cuda_spill_tier_rig.zig): host-memory kept states of the real GLM
//! layout, each rank's own core/spill store under TF_SPILL_TEST_DIR, rank 1 following rank 0's notes and asks.
const std = @import("std");
const core = @import("core");
const lanes = @import("lanes");
const tier = @import("cuda_spill_tier.zig");
const rig = @import("cuda_spill_tier_rig.zig");
const st = @import("cuda_spill_state.zig");
const wire = @import("cuda_spill_wire.zig");
const lane_backend = @import("cuda_lanes.zig");
const shard_keep = @import("cuda_shard_keep.zig");

const spill = core.spill;
const format = spill.format;
const testing = std.testing;
const HostEngine = rig.HostEngine;

const at = 2112; // one whole block and a 64-token tail

fn pointKey(ids: []const u32) format.Digest {
    return format.pointKey(ids, @intCast(st.block_tokens), &.{});
}

test "a kept state written at keep on both ranks, evicted, restored byte for byte on both, resumes as a fresh pass" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    try p.init(gpa, "restore", rig.Pair.settingsFor(1));
    defer p.deinit();
    var buf: [2200]u32 = undefined;
    const ids = rig.idsOf(&buf, 1);
    // the pass keeps at `at`; rank 0 writes at keep, rank 1 once its own pass adopted the state (the `.pass` note)
    const k = try p.keepPass(ids, at, 1 << 32 | 1);
    try testing.expect(k.point != null);
    try testing.expectEqual(tier.pointId(pointKey(ids[0..at])), k.point.?);
    try p.settle();
    const key = pointKey(ids[0..at]);
    try testing.expect(p.lead0().side.store.presence(key));
    try testing.expect(p.fol.?.side.store.presence(key));
    try testing.expectEqual(@as(u64, 0), p.fol.?.skipped);
    // its memory copy goes on both ranks; only the disks hold it
    const original = try gpa.dupe(u8, HostEngine.bytesOf(@as(*rig.Lanes.Kept, @ptrCast(@alignCast(k.saved))).snap));
    defer gpa.free(original);
    try p.evict(k.saved);
    try testing.expectEqual(@as(usize, 0), p.r[0].kept.items.len);
    try testing.expectEqual(@as(usize, 0), p.r[1].kept.items.len);
    try testing.expectEqual(@as(usize, 0), p.e[0].snaps + p.e[1].snaps);
    // restored: has, begin, progress, validate; both ranks hold it under rank 0's key, byte for byte the kept state
    const r = try p.restore(ids);
    try testing.expect(r == .ready);
    const k0: *rig.Lanes.Kept = @ptrCast(@alignCast(r.ready.saved));
    const k1: *rig.Lanes.Kept = @ptrCast(@alignCast(p.r[1].savedFor(k0.key) orelse return error.TestNotOnRank1));
    try testing.expectEqual(@as(u32, at), k0.at);
    try testing.expectEqual(@as(u32, at), k1.at);
    try testing.expect(HostEngine.same(&p.layout, k0.snap.sec, original, HostEngine.bytesOf(k0.snap)));
    try testing.expect(HostEngine.same(&p.layout, k1.snap.sec, original, HostEngine.bytesOf(k1.snap)));
    // pooled row at/4 of every layer reads zeros (cuda_spill_state: never state)
    for (0..11) |i| {
        const sec = k0.snap.sec;
        const row = HostEngine.bytesOf(k0.snap)[sec.pooled + i * sec.pooledBytes() + at / 4 * 144 ..][0..144];
        try testing.expect(std.mem.allEqual(u8, row, 0));
    }
    // the restored lineage comes back for the store (its root at least)
    try testing.expect(r.ready.lineage.len >= 1);
    // a longer prompt resumes from it on both ranks and draws what a fresh pass draws
    const exact = [2]usize{ p.e[0].exact, p.e[1].exact };
    const firsts = try p.resumeBoth(ids, r.ready.saved, at);
    try testing.expectEqual(HostEngine.first(ids), firsts[0]);
    try testing.expectEqual(HostEngine.first(ids), firsts[1]);
    for (0..2) |i| {
        try testing.expectEqual(exact[i] + 1, p.e[i].exact);
        try testing.expectEqual(p.e[i].resumed, p.e[i].exact); // (every resume on either rank from an exact state)
    }
    // every rank's counters reach rank 0 (a stats ask), each its own store's
    p.lead0().stats_at = 0;
    p.lead0().pump(1);
    var counts: [4]tier.Counts = undefined;
    try testing.expectEqual(@as(usize, 2), p.lead0().counts(&counts));
    for (counts[0..2]) |c| {
        try testing.expectEqual(@as(u64, 1), c.points);
        try testing.expectEqual(@as(u64, 1), c.writes_committed);
        try testing.expectEqual(@as(u64, 1), c.reads);
        try testing.expectEqual(@as(u64, 2), c.blocks);
        try testing.expectEqual(@as(u64, 0), c.points_part0 + c.points_part1); // (a replicated store has no parts)
    }
    try testing.expectEqual(@as(usize, 0), p.pg.fatals);
}

test "a point is found again after a restart on both ranks; one a rank lost is restored on no rank, and both forget it" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    try p.init(gpa, "restart", rig.Pair.settingsFor(1));
    defer p.deinit();
    var b1: [2200]u32 = undefined;
    var b2: [2300]u32 = undefined;
    const a = rig.idsOf(&b1, 2);
    const b = rig.idsOf(&b2, 3);
    // (one small-state slot: each write publishes before the next keep)
    _ = try p.keepPass(a, at, 1);
    try p.settle();
    _ = try p.keepPass(b, at + 64, 2);
    try p.settle();
    try testing.expectEqual(@as(u64, 2), p.lead0().side.store.counts().points);
    try testing.expectEqual(@as(u64, 2), p.fol.?.side.store.counts().points);
    // a restart: memory empty, both stores recover the published points (manifest last)
    try p.restart();
    try testing.expectEqual(@as(u64, 2), p.lead0().side.store.counts().recovered_points);
    try testing.expectEqual(@as(u64, 2), p.fol.?.side.store.counts().recovered_points);
    const r = try p.restore(a);
    try testing.expect(r == .ready);
    const firsts = try p.resumeBoth(a, r.ready.saved, at);
    try testing.expectEqual(HostEngine.first(a), firsts[0]);
    try testing.expectEqual(HostEngine.first(a), firsts[1]);
    // rank 1 loses its manifests across the next restart: rank 0 still finds b, `has` refuses it, both forget it
    try p.restart();
    p.close();
    try p.dropManifests(1);
    try p.open();
    try testing.expectEqual(@as(u64, 0), p.fol.?.side.store.counts().points);
    const kb = pointKey(b[0 .. at + 64]);
    try testing.expect(p.lead0().side.store.presence(kb));
    try testing.expectError(error.SpillNotOnEveryRank, p.restore(b));
    try testing.expect(!p.lead0().side.store.presence(kb));
    try testing.expectEqual(@as(u64, 1), p.lead0().presence_refused);
    try testing.expect(p.lead0().lookup(b, 0, @intCast(b.len - 1), &.{}, 0) == null);
    // the request prefills: exact from 0
    try testing.expectEqual(@as(usize, 0), p.r[0].kept.items.len);
    try testing.expectEqual(@as(usize, 0), p.pg.fatals);
}

test "damage on either rank (a flipped byte, a truncated or a missing block) aborts the restore everywhere and both ranks forget the point" {
    const gpa = testing.allocator;
    const Case = struct { rank: usize, how: rig.Pair.Damage, why: anyerror };
    for ([_]Case{
        .{ .rank = 0, .how = .flip, .why = error.SpillDamaged },
        .{ .rank = 1, .how = .flip, .why = error.SpillDamaged },
        .{ .rank = 1, .how = .truncate, .why = error.SpillDamaged },
        .{ .rank = 0, .how = .remove, .why = error.SpillMissing },
        .{ .rank = 1, .how = .remove, .why = error.SpillMissing },
    }, 0..) |c, n| {
        var p: rig.Pair = undefined;
        var tb: [32]u8 = undefined;
        try p.init(gpa, try std.fmt.bufPrint(&tb, "damage{d}", .{n}), rig.Pair.settingsFor(1));
        defer p.deinit();
        var buf: [2200]u32 = undefined;
        const ids = rig.idsOf(&buf, @intCast(10 + n));
        const k = try p.keepPass(ids, at, 1);
        try p.settle();
        try p.evict(k.saved);
        try p.damage(c.rank, c.how);
        const r = try p.restore(ids);
        try testing.expect(r == .failed);
        try testing.expectEqual(c.why, r.failed);
        // every rank forgot it; no rank holds a restored state; nothing is left reading or reserved
        const key = pointKey(ids[0..at]);
        try testing.expect(!p.lead0().side.store.presence(key));
        try testing.expect(!p.fol.?.side.store.presence(key));
        try testing.expectEqual(@as(usize, 0), p.r[0].kept.items.len);
        try testing.expectEqual(@as(usize, 0), p.r[1].kept.items.len);
        try testing.expectEqual(@as(usize, 0), p.e[0].snaps + p.e[1].snaps);
        try testing.expectEqual(@as(usize, 0), p.lead0().side.rxs.count() + p.fol.?.side.rxs.count());
        try testing.expect(p.lead0().lookup(ids, 0, @intCast(ids.len - 1), &.{}, 0) == null);
        // the request recomputes: a pass from 0 draws what a fresh one does
        const again = try p.keepPass(ids, at, 1);
        _ = again;
        try p.settle();
        try testing.expect(p.lead0().side.store.presence(key) and p.fol.?.side.store.presence(key));
    }
}

test "restores refused for capacity (the cap of restores in flight, no read slot) are the deferrable ones; a cancel frees every rank's destination" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    try p.init(gpa, "defer", rig.Pair.settingsFor(1));
    defer p.deinit();
    var b1: [2200]u32 = undefined;
    var b2: [2300]u32 = undefined;
    const a = rig.idsOf(&b1, 20);
    const b = rig.idsOf(&b2, 21);
    const ka = try p.keepPass(a, at, 1);
    try p.settle();
    const kb = try p.keepPass(b, at + 64, 2);
    try p.settle();
    try p.evict(ka.saved);
    try p.evict(kb.saved);
    const l = p.lead0();
    try testing.expect(!l.full());
    const pa = l.lookup(a, 0, @intCast(a.len - 1), &.{}, 0).?;
    const pb = l.lookup(b, 0, @intCast(b.len - 1), &.{}, 0).?;
    // `check`, which the prompt cache asks before it makes room: a point no lookup gave, or at another position, is
    // refused with nothing read; a's passes
    try testing.expectError(error.UnknownPoint, l.check(std.math.maxInt(u64), pa.at));
    try testing.expectError(error.BadPoint, l.check(pa.id, pa.at + 64));
    try l.check(pa.id, pa.at);
    try testing.expectEqual(@as(usize, 0), l.side.rxs.count());
    const job = try l.begin(pa.id, pa.at, false);
    try p.sink.take(&p.r[1]);
    // one restore in flight is the cap: full before any room is made, and a begin refused for capacity
    try testing.expect(l.full());
    try testing.expectError(error.TooManyRestores, l.begin(pb.id, pb.at, false));
    try testing.expect(std.mem.indexOfScalar(anyerror, &capacity_errors, error.TooManyRestores) != null);
    // with a wider cap the store's one restore slot is the bound (NoSlot or TooMany: the store's capacity refusals)
    l.max_restores = 2;
    const second: anyerror = if (l.begin(pb.id, pb.at, false)) |_| error.TestBegun else |e| e;
    try testing.expect(second == error.TooMany or second == error.NoSlot);
    try testing.expect(std.mem.indexOfScalar(anyerror, &capacity_errors, second) != null);
    l.max_restores = 1;
    // a cancel: every rank's destination goes, the point stays everywhere, and the next begin goes through
    l.cancel(job);
    try p.sink.take(&p.r[1]);
    try testing.expectEqual(@as(usize, 0), p.e[0].snaps + p.e[1].snaps);
    try testing.expectEqual(@as(usize, 0), p.fol.?.restores.count());
    try testing.expect(l.side.store.presence(pointKey(a[0..at])) and p.fol.?.side.store.presence(pointKey(a[0..at])));
    // (the read slot comes back once the cancelled read's blocks finish)
    var r: tier.Poll = undefined;
    var i: usize = 0;
    while (i < 5000) : (i += 1) {
        r = p.restore(b) catch |e| switch (e) {
            error.NoSlot, error.TooMany => {
                std.Io.sleep(testing.io, .fromMilliseconds(1), .awake) catch {};
                continue;
            },
            else => return e,
        };
        break;
    }
    try testing.expect(r == .ready);
    // a rank 1 at its keep floor cannot reserve: validate fails there (reserve), every rank aborts, the point stays
    try p.evict(r.ready.saved);
    p.e[1].refuse_dest = true;
    const f = try p.restore(b);
    try testing.expect(f == .failed);
    try testing.expectEqual(error.SpillReserve, f.failed);
    try testing.expect(l.side.store.presence(pointKey(b[0 .. at + 64])));
    try testing.expectEqual(@as(usize, 0), p.e[0].snaps + p.e[1].snaps);
}

/// prompt_cache.capacity_errors (glm_cuda cannot import the engine API; native/cuda.zig's test checks the two agree).
const capacity_errors = [_]anyerror{ error.TierFull, error.TooManyRestores, error.TooMany, error.NoSlot, error.Busy };

test "the release guard holds a released state while its write still reads it; past its limit the rest is staged at the release" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    var s = rig.Pair.settingsFor(1);
    s.stage_mib = 52; // a tail's slot and two whole blocks at once (the rest staged at a release)
    try p.init(gpa, "guard", s);
    defer p.deinit();
    var b1: [4300]u32 = undefined;
    const a = rig.idsOf(&b1, 30);
    // copies take a few polls to complete on both ranks: the write is still reading the state when it is released
    p.e[0].delay = 3;
    p.e[1].delay = 3;
    const k = try p.keepPass(a, 4224, 1); // two whole blocks and a tail
    const snap0 = @as(*rig.Lanes.Kept, @ptrCast(@alignCast(k.saved))).snap;
    try testing.expect(p.lead0().side.store.pins(st.sourceOf(snap0.id), 0));
    try p.evict(k.saved);
    // held on rank 0 (its buffer alive past the cache's free), and on rank 1 at the `.drop` note
    try testing.expectEqual(@as(usize, 1), p.lead0().side.guard.held.items.len);
    try testing.expectEqual(@as(usize, 1), p.fol.?.side.guard.held.items.len);
    try testing.expectEqual(@as(usize, 2), p.e[0].snaps + p.e[1].snaps);
    try p.settle();
    // let go once the blocks are staged and copied; the point published on both
    try testing.expectEqual(@as(usize, 0), p.lead0().side.guard.held.items.len + p.fol.?.side.guard.held.items.len);
    try testing.expectEqual(@as(usize, 0), p.e[0].snaps + p.e[1].snaps);
    const key = pointKey(a[0..4224]);
    try testing.expect(p.lead0().side.store.presence(key) and p.fol.?.side.store.presence(key));
    // past the limit: the rest staged at the release (waited for there), the buffer freed by the cache's free
    p.lead0().side.guard.limit = 0;
    p.fol.?.side.guard.limit = 0;
    var b2: [4300]u32 = undefined;
    const b = rig.idsOf(&b2, 31);
    const kb = try p.keepPass(b, 4224, 2);
    try p.evict(kb.saved);
    try testing.expectEqual(@as(usize, 0), p.e[0].snaps + p.e[1].snaps);
    try p.settle();
    const key_b = pointKey(b[0..4224]);
    try testing.expect(p.lead0().side.store.presence(key_b) and p.fol.?.side.store.presence(key_b));
    try testing.expect(p.lead0().side.store.counts().guard_staged >= 1);
    const r = try p.restore(b);
    try testing.expect(r == .ready);
    const firsts = try p.resumeBoth(b, r.ready.saved, 4224);
    try testing.expectEqual(HostEngine.first(b), firsts[0]);
    try testing.expectEqual(HostEngine.first(b), firsts[1]);
}

test "a clean stop: rank 0's flush drains every rank's writes before it returns, and a restart finds them" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    try p.init(gpa, "flush", rig.Pair.settingsFor(1));
    defer p.deinit();
    var b1: [2200]u32 = undefined;
    const a = rig.idsOf(&b1, 40);
    _ = try p.keepPass(a, at, 1);
    // nothing pumped yet: the flush stages, writes and publishes on both ranks (rank 1's at the `flush` note, before
    // it answers the stats ask that ends rank 0's flush)
    try testing.expectEqual(@as(u64, 0), p.lead0().flush(10 * std.time.ns_per_s));
    try testing.expectEqual(@as(u64, 1), p.lead0().side.store.counts().points);
    try testing.expectEqual(@as(u64, 1), p.fol.?.side.store.counts().points);
    try testing.expectEqual(@as(u64, 1), p.lead0().peers[0].points);
    try p.restart();
    try testing.expect((try p.restore(a)) == .ready);
}

test "keeps and lookups under the minimum or off the grid write and find nothing; a lineage the keep note cannot carry is not written" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    var s = rig.Pair.settingsFor(1);
    s.min_tokens = 2048;
    try p.init(gpa, "rules", s);
    defer p.deinit();
    var b1: [2200]u32 = undefined;
    const a = rig.idsOf(&b1, 50);
    // under the minimum: written on no rank (no keep note either)
    const k1 = try p.keepPass(a, 1024, 1);
    try testing.expect(k1.point == null);
    try testing.expectEqual(@as(u64, 0), p.lead0().side.store.counts().queued + p.fol.?.side.store.counts().queued);
    try testing.expect(p.lead0().keep(.{ .saved = k1.saved, .key = a, .at = 2000 }) == null); // off the grid
    // a lineage longer than a keep note carries: not written (no rank holds a point another lacks)
    const k2 = try p.keepPass(a, 2048, 1);
    try testing.expect(k2.point != null);
    var forks: [wire.max_forks + 2]tier.Fork = undefined;
    for (&forks, 0..) |*f, i| f.* = .{ .name = i + 1, .at = @intCast(i) };
    try testing.expect(p.lead0().keep(.{ .saved = k2.saved, .key = a, .at = 2048, .lineage = &forks }) == null);
    try p.settle();
    // found on its grid only, under max_at, and on the request's chunk starts when it names any
    try testing.expect(p.lead0().lookup(a, 0, 2047, &.{}, 0) == null);
    try testing.expectEqual(@as(u32, 2048), p.lead0().lookup(a, 0, 2100, &.{}, 0).?.at);
    try testing.expect(p.lead0().lookup(a, 0, 2100, &.{ 64, 1024 }, 0) == null);
    try testing.expectEqual(@as(u32, 2048), p.lead0().lookup(a, 0, 2100, &.{ 64, 2048 }, 0).?.at);
}

test "a lookup with chunk starts finds the longest point on one of them, under a longer point off them" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    try p.init(gpa, "starts", rig.Pair.settingsFor(1));
    defer p.deinit();
    var b1: [4300]u32 = undefined;
    const a = rig.idsOf(&b1, 55);
    _ = try p.keepPass(a, 2048, 1);
    try p.settle();
    _ = try p.keepPass(a, 4160, 1); // (an earlier turn's end, off this request's chunk starts)
    try p.settle();
    const l = p.lead0();
    try testing.expectEqual(@as(?u32, 4160), if (l.lookup(a, 0, 4200, &.{}, 0)) |x| x.at else null);
    try testing.expectEqual(@as(?u32, 2048), if (l.lookup(a, 0, 4200, &.{ 2048, 4096 }, 0)) |x| x.at else null);
    try testing.expectEqual(@as(?u32, 2048), if (l.lookup(a, 0, 4200, &.{ 4096, 64, 2048 }, 0)) |x| x.at else null); // (in any order)
    try testing.expect(l.lookup(a, 0, 4200, &.{ 64, 4096 }, 0) == null);
    try testing.expect(l.lookup(a, 2048, 4200, &.{ 2048, 4096 }, 0) == null); // (only past `beyond`)
    // the point found restores on both ranks
    const pt = l.lookup(a, 0, 4200, &.{ 2048, 4096 }, 0) orelse return error.TestNoPoint;
    const job = try l.begin(pt.id, pt.at, pt.shared);
    try p.sink.take(&p.r[1]);
    var r: tier.Poll = .{ .pending = .{} };
    var i: usize = 0;
    while (r == .pending and i < 10_000) : (i += 1) {
        r = l.poll(job, 5 * std.time.ns_per_ms);
        try p.sink.take(&p.r[1]);
        if (r == .pending) std.Io.sleep(testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try testing.expect(r == .ready);
    try testing.expectEqual(@as(u32, 2048), @as(*rig.Lanes.Kept, @ptrCast(@alignCast(r.ready.saved))).at);
}

test "a keep that cannot record its point queues no write: nothing reads a state the prompt cache then frees without a release" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    try p.init(gpa, "keep-oom", rig.Pair.settingsFor(1));
    defer p.deinit();
    var b1: [2200]u32 = undefined;
    const a = rig.idsOf(&b1, 56);
    // rank 0's point table cannot grow: the keep is null on every rank (the cache records no point, so never releases)
    p.lead0().gpa = testing.failing_allocator;
    const got = p.keepPass(a, at, 1);
    p.lead0().gpa = gpa;
    const k = try got;
    try testing.expect(k.point == null);
    try testing.expectEqual(@as(usize, 0), p.lead0().side.writes.count());
    try testing.expectEqual(@as(u64, 0), p.lead0().side.store.counts().pending_jobs + p.fol.?.side.store.counts().pending_jobs);
    // freed by the family's drop alone on both ranks; nothing stages from it after
    p.hooks.drop(p.hooks.ptr, k.saved);
    try p.sink.take(&p.r[1]);
    try testing.expectEqual(@as(usize, 0), p.e[0].snaps + p.e[1].snaps);
    try p.settle();
    try testing.expectEqual(@as(u64, 0), p.lead0().side.store.counts().points + p.fol.?.side.store.counts().points);
    // the next keep of it writes on both
    try testing.expect((try p.keepPass(a, at, 1)).point != null);
    try p.settle();
    const key = pointKey(a[0..at]);
    try testing.expect(p.lead0().side.store.presence(key) and p.fol.?.side.store.presence(key));
}

test "every rank's counters are asked for only after rank 0 sent the others something: an idle pair asks nothing" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    try p.init(gpa, "stats", rig.Pair.settingsFor(1));
    defer p.deinit();
    const l = p.lead0();
    // the first pump asks (the stores recovered their points at open); then nothing moved
    l.pump(1);
    try testing.expectEqual(@as(usize, 1), p.pg.asks);
    for (0..5) |_| {
        l.stats_at = 0; // (as if a second passed)
        l.pump(1);
        try p.sink.take(&p.r[1]);
    }
    try testing.expectEqual(@as(usize, 1), p.pg.asks);
    // a keep: asked again until an answer shows rank 1's write done, then idle again
    var b1: [2200]u32 = undefined;
    const a = rig.idsOf(&b1, 57);
    _ = try p.keepPass(a, at, 1);
    try p.settle();
    var i: usize = 0;
    while (l.peer_writes and i < 1000) : (i += 1) {
        l.stats_at = 0;
        l.pump(1);
        try p.sink.take(&p.r[1]);
    }
    try testing.expect(!l.peer_writes);
    const asked = p.pg.asks;
    try testing.expect(asked > 1);
    var counts: [2]tier.Counts = undefined;
    try testing.expectEqual(@as(usize, 2), l.counts(&counts));
    try testing.expectEqual(@as(u64, 1), counts[1].writes_committed);
    for (0..5) |_| {
        l.stats_at = 0;
        l.pump(1);
        try p.sink.take(&p.r[1]);
    }
    try testing.expectEqual(asked, p.pg.asks);
    // a touch is sent: the next pump asks once
    l.touch(tier.pointId(pointKey(a[0..at])));
    l.stats_at = 0;
    l.pump(1);
    try p.sink.take(&p.r[1]);
    try testing.expectEqual(asked + 1, p.pg.asks);
}

test "the staging plan: the ring, read-ahead and small states as asked; under two write blocks refused" {
    const l = try rig.testLayout();
    const p = try tier.plan(.{ .dir = "/s", .gib = 1 }, &l);
    try testing.expectEqual(@as(usize, 21), p.write); // 256 MiB of 12,705,792-byte blocks (Python's ring)
    try testing.expectEqual(@as(usize, 16), p.read);
    try testing.expectEqual(@as(usize, 4), p.small);
    try testing.expectEqual(@as(usize, 2), p.restore);
    try testing.expectEqual(@as(usize, l.block_bytes), p.block_slot);
    try testing.expectEqual(std.mem.alignForward(usize, l.small_bytes, 4096), p.small_slot);
    try testing.expectEqual(@as(usize, 21 + 16 + 4 + 2 + 1), tier.eventCount(p));
    try testing.expectError(error.StagingTooSmall, tier.plan(.{ .dir = "/s", .gib = 1, .stage_mib = 20 }, &l));
    try testing.expectError(error.StagingTooSmall, tier.plan(.{ .dir = "/s", .gib = 1, .read_blocks = 0 }, &l));
    // with DFlash2's rings the small slot grows by the drafter's 22,282,240 bytes
    const d = try st.Layout.init(l.g, 64, .{ .kv = 4, .ring = 2176 });
    try testing.expectEqual(std.mem.alignForward(usize, l.small_bytes + 22_282_240, 4096), (try tier.plan(.{ .dir = "/s", .gib = 1 }, &d)).small_slot);
    // TF_GLM_KV_SHARD: the same slots on either rank (a slot holds the rank's largest block, its own part's; the other
    // part's blocks and tails are smaller), so the staging the engine takes before its pool does not change
    for (0..2) |r| {
        const sh = try rig.shardedLayout(r);
        try testing.expectEqual(p, try tier.plan(.{ .dir = "/s", .gib = 1 }, &sh));
    }
}

test "spill v2 off: a two-rank pass with no tier sends only the pass's note, writes nothing and allocates no staging" {
    const gpa = testing.allocator;
    const layout = try rig.testLayout();
    var e0 = HostEngine.init(gpa, layout);
    defer e0.deinit();
    var e1 = HostEngine.init(gpa, layout);
    defer e1.deinit();
    var r0 = rig.Lanes.init(gpa, &e0, true, 1 << 20, 8);
    defer r0.deinit();
    var r1 = rig.Lanes.init(gpa, &e1, true, 1 << 20, 8);
    defer r1.deinit();
    r1.follower = true;
    var sink: rig.Sink = .{ .gpa = gpa };
    defer sink.deinit();
    r0.notes = sink.notesOf();
    const hooks = r0.cacheHooks(1 << 40, .{}, 0, 5);
    var buf: [2200]u32 = undefined;
    const ids = rig.idsOf(&buf, 60);
    const A = struct {
        hooks: lane_backend.CacheHooks,
        fn at(ptr: *anyopaque, _: *lanes.Stream, mark: u32) void {
            const a: *@This() = @ptrCast(@alignCast(ptr));
            _ = a.hooks.save(a.hooks.ptr, null, mark) catch return;
        }
    };
    var a: A = .{ .hooks = hooks };
    var s0 = try lanes.Stream.init(gpa, .{ .id = "o", .prompt = ids, .max_new = 4, .reuse = .{ .marks = &.{at}, .hook = .{ .ptr = &a, .at = A.at } } });
    defer s0.deinit(gpa);
    var s1 = try lanes.Stream.init(gpa, .{ .id = "o", .prompt = ids, .max_new = 4, .reuse = .{ .marks = &.{at} } });
    defer s1.deinit(gpa);
    try rig.Lanes.prefillFn(&r0, &s0);
    try rig.Lanes.prefillFn(&r1, &s1);
    try testing.expectEqual(@as(usize, 1), sink.notes.items.len);
    try testing.expectEqual(@as(u8, @backingInt(lane_backend.Note.pass)), sink.notes.items[0][0]);
    try testing.expectEqual(@as(usize, 0), sink.flushes);
    try sink.take(&r1);
    try testing.expect(r1.savedFor(r0.kept.items[0].key) != null);
    try testing.expectEqual(@as(usize, 0), e0.events.len + e1.events.len);
    rig.Lanes.releaseFn(&r0, &s0);
    rig.Lanes.releaseFn(&r1, &s1);
}

// -- TF_GLM_KV_SHARD: owner-sharded points (each rank its own view, the home part recorded) ---------------------------

/// A token's and a whole block's bytes on the home part's owner and on the other rank (cuda_spill_state).
const own_block = 12_705_792;
const own_token = 6204;
const other_block = 811_008;
const other_token = 396;

/// A sharded pair holding `a` kept in part 0 (an empty pool places it there) and `b` kept in part 1 (part 0 taken by
/// another stream meanwhile), both written on both ranks.
fn keepBothParts(p: *rig.Pair, a: []const u32, b: []const u32) ![2]*anyopaque {
    const ka = try p.keepPass(a, at, 1);
    try p.settle();
    const other = try p.occupy(0, rig.pool_rows / 2);
    const kb = try p.keepPass(b, at, 2);
    try p.settle();
    try p.vacate(other);
    return .{ ka.saved, kb.saved };
}

test "TF_GLM_KV_SHARD: each rank writes its own view of a kept state under its home part, and a restore gives each rank that view back with its home; the resume lands in the home part" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    try p.initWith(gpa, "shard-restore", rig.Pair.settingsFor(1), .{ .sharded = true });
    defer p.deinit();
    var b1: [2200]u32 = undefined;
    var b2: [2200]u32 = undefined;
    const a = rig.idsOf(&b1, 70);
    const b = rig.idsOf(&b2, 71);
    const ka = try p.keepPass(a, at, 1);
    try p.settle();
    // a's home is part 0: rank 0 wrote its latent and pooled rows (a whole block and a 64-token tail), rank 1 the
    // pooled rows alone
    try testing.expectEqual(@as(u64, own_block + 64 * own_token), p.lead0().side.store.counts().written_bytes);
    try testing.expectEqual(@as(u64, other_block + 64 * other_token), p.fol.?.side.store.counts().written_bytes);
    try testing.expectEqual(@as(u64, 0), p.fol.?.skipped);
    for ([_]*rig.Leader.S{ p.lead0().side, p.fol.?.side }) |side| { // (rank 0's own part, rank 1's other part)
        try testing.expectEqual(@as(u64, 1), side.cached.points_part0);
        try testing.expectEqual(@as(u64, 0), side.cached.points_part1);
    }
    const other = try p.occupy(0, rig.pool_rows / 2);
    const kb = try p.keepPass(b, at, 2);
    try p.settle();
    try p.vacate(other);
    // b's home is part 1: the reverse
    for ([_]*rig.Leader.S{ p.lead0().side, p.fol.?.side }) |side| try testing.expectEqual(@as(u64, own_block + 64 * own_token + other_block + 64 * other_token), side.store.counts().written_bytes);
    const keys = [2]format.Digest{ pointKey(a[0..at]), pointKey(b[0..at]) };
    const saved = [2]*anyopaque{ ka.saved, kb.saved };
    var originals: [2][2][]u8 = undefined; // [point][rank]
    for (saved, 0..) |sv, i| for (0..2) |r| {
        const snap = try p.snapOn(r, sv);
        try testing.expectEqual(@as(?u8, @intCast(i)), snap.home);
        try testing.expectEqual(r == i, snap.sec.rows); // (the latent rows on the home part's owner only)
        originals[i][r] = try gpa.dupe(u8, HostEngine.bytesOf(snap));
    };
    defer for (originals) |o| for (o) |x| gpa.free(x);
    for ([_]*rig.Leader.S{ p.lead0().side, p.fol.?.side }) |side| {
        try testing.expectEqual(@as(?u32, 0), side.store.partOf(keys[0]).?.part);
        try testing.expectEqual(@as(?u32, 1), side.store.partOf(keys[1]).?.part);
    }
    // every rank's points by part (the stats ask)
    p.lead0().stats_at = 0;
    p.lead0().pump(1);
    var counts: [4]tier.Counts = undefined;
    try testing.expectEqual(@as(usize, 2), p.lead0().counts(&counts));
    for (counts[0..2]) |c| {
        try testing.expectEqual(@as(u64, 2), c.points);
        try testing.expectEqual(@as(u64, 1), c.points_part0);
        try testing.expectEqual(@as(u64, 1), c.points_part1);
    }
    try p.evict(ka.saved);
    try p.evict(kb.saved);
    try testing.expectEqual(@as(usize, 0), p.e[0].snaps + p.e[1].snaps);
    const exact = [2]usize{ p.e[0].exact, p.e[1].exact }; // (a pass resumes its own mark's state too)
    // each restored: every rank holds its own view of the point's home again, byte for byte the state it kept
    for ([_][]const u32{ a, b }, 0..) |ids, i| {
        const r = try p.restore(ids);
        try testing.expect(r == .ready);
        for (0..2) |rank| {
            const snap = try p.snapOn(rank, r.ready.saved);
            try testing.expectEqual(@as(?u8, @intCast(i)), snap.home);
            try testing.expectEqual(rank == i, snap.sec.rows);
            try testing.expect(HostEngine.same(&p.layouts[rank], snap.sec, originals[i][rank], HostEngine.bytesOf(snap)));
        }
        // the resume: placed in part 0 (an empty pool), where a's home is (here) and from where b moves to its own;
        // nothing carried, and both ranks draw a fresh pass's token from their own view
        const firsts = try p.resumeBoth(ids, r.ready.saved, at);
        try testing.expectEqual(HostEngine.first(ids), firsts[0]);
        try testing.expectEqual(HostEngine.first(ids), firsts[1]);
    }
    for (&p.e, exact) |*e, x| {
        try testing.expectEqual(x + 2, e.exact);
        try testing.expectEqual(e.resumed, e.exact);
        try testing.expectEqual(@as(usize, 0), e.carries);
    }
    try testing.expectEqual(@as(usize, 0), p.pg.fatals);
}

test "TF_GLM_KV_SHARD: a restored state whose home part has no room resumes in the other part (carried), from 0 with the carry off" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    try p.initWith(gpa, "shard-carry", rig.Pair.settingsFor(1), .{ .sharded = true });
    defer p.deinit();
    var b1: [2200]u32 = undefined;
    const a = rig.idsOf(&b1, 72);
    const k = try p.keepPass(a, at, 1);
    try p.settle();
    try p.evict(k.saved);
    // part 0 full: the restore still starts (a carry takes 2,112 tokens: past the minimum), and the resume's extent
    // stays in part 1 with the home owner's rows carried; both ranks resume exactly from their own views
    const full = try p.occupy(0, rig.pool_rows / 2);
    const r = try p.restore(a);
    try testing.expect(r == .ready);
    try testing.expectEqual(@as(?u8, 0), (try p.snapOn(0, r.ready.saved)).home);
    const exact = [2]usize{ p.e[0].exact, p.e[1].exact };
    const firsts = try p.resumeBoth(a, r.ready.saved, at);
    try testing.expectEqual(HostEngine.first(a), firsts[0]);
    try testing.expectEqual(HostEngine.first(a), firsts[1]);
    for (&p.e, exact) |*e, x| {
        try testing.expectEqual(@as(usize, 1), e.carries);
        try testing.expectEqual(x + 1, e.exact);
        try testing.expectEqual(e.resumed, e.exact);
    }
    // the carry off (TF_GLM_KV_SHARD_CARRY=0 on both ranks) with the state in memory: the prompt from 0, nothing resumed
    for (&p.e) |*e| e.carry = false;
    const resumed = [2]usize{ p.e[0].resumed, p.e[1].resumed };
    try testing.expectEqual(HostEngine.first(a), (try p.resumeBoth(a, r.ready.saved, at))[0]);
    for (&p.e, resumed) |*e, x| {
        try testing.expectEqual(@as(usize, 1), e.carries);
        try testing.expectEqual(x, e.resumed);
    }
    try p.vacate(full);
    try testing.expectEqual(p.e[0].pool.?.digest(), p.e[1].pool.?.digest());
}

test "TF_GLM_KV_SHARD: a point whose home part has no room is not read when its resume could only start from 0 (the carry off, or a state under the minimum), and restores once the part has room" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    try p.initWith(gpa, "shard-homefull", rig.Pair.settingsFor(1), .{ .sharded = true });
    defer p.deinit();
    var b1: [2200]u32 = undefined;
    var b2: [2200]u32 = undefined;
    const a = rig.idsOf(&b1, 73);
    const s = rig.idsOf(&b2, 74);
    const small = 1984; // (under cuda_shard_keep.carry_min)
    comptime std.debug.assert(small < shard_keep.carry_min);
    const keys = [2]format.Digest{ pointKey(a[0..at]), pointKey(s[0..small]) };
    const ka = try p.keepPass(a, at, 1);
    try p.settle(); // (one small slot a rank)
    const ks = try p.keepPass(s, small, 2);
    try p.settle();
    for ([_]*anyopaque{ ka.saved, ks.saved }) |sv| {
        try testing.expectEqual(@as(?u8, 0), (try p.snapOn(0, sv)).home); // (an empty pool places both in part 0)
        try p.evict(sv);
    }
    const full = try p.occupy(0, rig.pool_rows / 2);
    // the carry off: a's restore is refused before any read, `has` or reservation on either rank (its point stays on
    // both); the carry on: s's too, a carry never taking a state under the minimum
    for (&p.e) |*e| e.carry = false;
    // (`check`, which the prompt cache asks before it makes room for the restore, refuses it the same way)
    const pa = p.lead0().lookup(a, 0, @intCast(a.len - 1), &.{}, 0).?;
    try testing.expectError(error.KvShardHomeFull, p.lead0().check(pa.id, pa.at));
    try testing.expectError(error.KvShardHomeFull, p.restore(a));
    for (&p.e) |*e| e.carry = true;
    try testing.expectError(error.KvShardHomeFull, p.restore(s));
    for ([_]*rig.Leader.S{ p.lead0().side, p.fol.?.side }) |side| {
        try testing.expectEqual(@as(u64, 0), side.store.counts().loaded + side.store.counts().load_failed);
        try testing.expectEqual(@as(usize, 0), side.rxs.count());
        for (keys) |key| try testing.expect(side.store.presence(key));
    }
    try testing.expectEqual(@as(u64, 0), p.lead0().presence_refused);
    try testing.expectEqual(@as(usize, 0), p.lead0().jobs.count());
    try testing.expectEqual(@as(usize, 0), p.e[0].snaps + p.e[1].snaps);
    // part 0 has room again: both restore with the carry off, each resumed in its home part exactly, nothing carried
    try p.vacate(full);
    try p.lead0().check(pa.id, pa.at);
    for (&p.e) |*e| e.carry = false;
    const resumed = [2]usize{ p.e[0].resumed, p.e[1].resumed }; // (a pass resumes its own mark's state too)
    for ([_][]const u32{ a, s }, [_]u32{ at, small }) |ids, n| {
        const r = try p.restore(ids);
        try testing.expect(r == .ready);
        const firsts = try p.resumeBoth(ids, r.ready.saved, n);
        try testing.expectEqual(HostEngine.first(ids), firsts[0]);
        try testing.expectEqual(HostEngine.first(ids), firsts[1]);
    }
    for (&p.e, resumed) |*e, x| {
        try testing.expectEqual(x + 2, e.resumed);
        try testing.expectEqual(e.resumed, e.exact);
        try testing.expectEqual(@as(usize, 0), e.carries);
    }
    try testing.expectEqual(@as(usize, 0), p.pg.fatals);
}

test "TF_GLM_KV_SHARD: a damaged block on the rank that does not own the point's home makes both ranks forget it" {
    const gpa = testing.allocator;
    for ([_]rig.Pair.Damage{ .flip, .truncate }, 0..) |how, n| for (0..2) |home| {
        var p: rig.Pair = undefined;
        var tb: [32]u8 = undefined;
        try p.initWith(gpa, try std.fmt.bufPrint(&tb, "shard-damage{d}{d}", .{ n, home }), rig.Pair.settingsFor(1), .{ .sharded = true });
        defer p.deinit();
        var buf: [2200]u32 = undefined;
        const ids = rig.idsOf(&buf, @intCast(80 + 2 * n + home));
        const other = if (home == 1) try p.occupy(0, rig.pool_rows / 2) else null;
        const k = try p.keepPass(ids, at, 1);
        try p.settle();
        if (other) |o| try p.vacate(o);
        try testing.expectEqual(@as(?u8, @intCast(home)), (try p.snapOn(0, k.saved)).home);
        try p.evict(k.saved);
        try p.damage(1 - home, how); // (its blocks: the pooled rows alone)
        const r = try p.restore(ids);
        try testing.expect(r == .failed);
        try testing.expectEqual(error.SpillDamaged, r.failed);
        const key = pointKey(ids[0..at]);
        try testing.expect(!p.lead0().side.store.presence(key));
        try testing.expect(!p.fol.?.side.store.presence(key));
        try testing.expectEqual(@as(usize, 0), p.e[0].snaps + p.e[1].snaps);
        try testing.expectEqual(@as(usize, 0), p.lead0().side.rxs.count() + p.fol.?.side.rxs.count());
        try testing.expect(p.lead0().lookup(ids, 0, @intCast(ids.len - 1), &.{}, 0) == null);
    };
}

test "TF_GLM_KV_SHARD: after a restart both ranks recover their points of both parts, and each restores into its home" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    try p.initWith(gpa, "shard-restart", rig.Pair.settingsFor(1), .{ .sharded = true });
    defer p.deinit();
    var b1: [2200]u32 = undefined;
    var b2: [2200]u32 = undefined;
    const a = rig.idsOf(&b1, 90);
    const b = rig.idsOf(&b2, 91);
    _ = try keepBothParts(&p, a, b);
    try p.restart();
    for ([_]*rig.Leader.S{ p.lead0().side, p.fol.?.side }) |side| {
        const c = side.store.counts();
        try testing.expectEqual(@as(u64, 2), c.recovered_points);
        try testing.expectEqual(@as(u64, 1), side.cached.points_part0);
        try testing.expectEqual(@as(u64, 1), side.cached.points_part1);
    }
    for ([_][]const u32{ a, b }, 0..) |ids, i| {
        const r = try p.restore(ids);
        try testing.expect(r == .ready);
        for (0..2) |rank| {
            const snap = try p.snapOn(rank, r.ready.saved);
            try testing.expectEqual(@as(?u8, @intCast(i)), snap.home);
            try testing.expect(p.e[rank].exactFor(snap, ids[0..at]));
        }
        const firsts = try p.resumeBoth(ids, r.ready.saved, at);
        try testing.expectEqual(HostEngine.first(ids), firsts[0]);
        try testing.expectEqual(HostEngine.first(ids), firsts[1]);
    }
}

test "TF_GLM_KV_SHARD: the same ids kept again from the other part by a rank that lost its copy are refused at has, and every rank forgets them" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    try p.initWith(gpa, "shard-reparted", rig.Pair.settingsFor(1), .{ .sharded = true });
    defer p.deinit();
    var b1: [2200]u32 = undefined;
    const a = rig.idsOf(&b1, 92);
    const key = pointKey(a[0..at]);
    _ = try p.keepPass(a, at, 1); // (part 0)
    try p.settle();
    // rank 1 loses its copy; the conversation comes back with part 0 full and the carry off: rank 0 finds its point but
    // refuses the restore before `has` (the resume could only start from 0), so no rank forgets it; the turn prefills
    // in part 1 and is kept again there: rank 0 dedups it (its point keeps part 0), rank 1 writes it under part 1
    try p.dropAll();
    p.close();
    try p.dropManifests(1);
    try p.open();
    const full = try p.occupy(0, rig.pool_rows / 2);
    for (&p.e) |*e| e.carry = false;
    try testing.expectError(error.KvShardHomeFull, p.restore(a));
    try testing.expectEqual(@as(u64, 0), p.lead0().presence_refused);
    _ = try p.keepPass(a, at, 1);
    try p.settle();
    try p.vacate(full);
    for (&p.e) |*e| e.carry = true;
    try testing.expectEqual(@as(?u32, 0), p.lead0().side.store.partOf(key).?.part);
    try testing.expectEqual(@as(?u32, 1), p.fol.?.side.store.partOf(key).?.part);
    try p.dropAll();
    // `has` names part 0: rank 1's point of these ids is the other view (incompatible), so no rank restores it and
    // every rank forgets it
    try testing.expectError(error.SpillNotOnEveryRank, p.restore(a));
    try testing.expectEqual(@as(u64, 1), p.lead0().presence_refused);
    try testing.expect(!p.lead0().side.store.presence(key));
    try testing.expect(!p.fol.?.side.store.presence(key));
    try testing.expectEqual(@as(usize, 0), p.e[0].snaps + p.e[1].snaps);
    try testing.expectEqual(@as(usize, 0), p.pg.fatals);
}

test "TF_GLM_KV_SHARD: a keep note whose part is not the home of rank 1's own state writes nothing there" {
    const gpa = testing.allocator;
    var p: rig.Pair = undefined;
    try p.initWith(gpa, "shard-skip", rig.Pair.settingsFor(1), .{ .sharded = true });
    defer p.deinit();
    var b1: [2200]u32 = undefined;
    const a = rig.idsOf(&b1, 93);
    // rank 1's pool alone has part 0 taken: its pass keeps the state in part 1 while rank 0's keeps it in part 0
    const pool = &p.e[1].pool.?;
    const x = try pool.add(0, rig.pool_rows / 2);
    (try pool.get(x)).owner = x;
    _ = try p.keepPass(a, at, 1);
    try p.settle();
    try pool.remove(x);
    try testing.expectEqual(@as(u64, 1), p.fol.?.skipped);
    try testing.expect(p.lead0().side.store.presence(pointKey(a[0..at])));
    try testing.expect(!p.fol.?.side.store.presence(pointKey(a[0..at])));
    try testing.expectEqual(@as(u64, 0), p.fol.?.side.store.counts().keeps);
}
