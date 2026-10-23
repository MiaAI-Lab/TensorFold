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
