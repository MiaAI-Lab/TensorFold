//! GLM tools/kv_shard_check.py and kv_shard_mutants.sh (patch 0118) for the host side of TF_GLM_KV_SHARD: the
//! pool's parts and twins (cuda_pool.zig) and the shard's sizes, addresses, glue, history fetch and crossing rows
//! (cuda_shard.zig). One suite of named checks, run as it is (every check passes, no section raises) and once a
//! mutant of either file (each must fail the check written for it). Python's decoder checks (_room's evictions,
//! _place_hit's carries, _relocate, the spill names) have no host side here yet: kept states live outside the pool.
const std = @import("std");
const pool = @import("cuda_pool.zig");
const shard = @import("cuda_shard.zig");
const G = @import("cuda_geometry.zig").Geometry;
const Pool = pool.Pool;
const A = 4096;
const H = 32 * A; // a part

const T = struct {
    gpa: std.mem.Allocator,
    pass: usize = 0,
    crash: usize = 0,
    failed: std.ArrayList([]const u8) = .empty,
    verbose: bool = false,
    fn check(t: *T, name: []const u8, ok: bool) void {
        if (ok) {
            t.pass += 1;
            return;
        }
        t.failed.append(t.gpa, name) catch @panic("out of memory");
        if (t.verbose) std.debug.print("FAIL {s}\n", .{name});
    }
    fn section(t: *T, name: []const u8, f: *const fn (*T) anyerror!void) void {
        f(t) catch |err| {
            t.crash += 1;
            if (t.verbose) std.debug.print("{s}: raised {s}\n", .{ name, @errorName(err) });
        };
    }
    fn killedBy(t: *const T, prefix: []const u8) bool {
        for (t.failed.items) |f| {
            if (std.mem.startsWith(u8, f, prefix)) return true;
        }
        return false;
    }
};

fn live(p: *Pool, base: usize, n: usize, sid: u64) !u64 {
    const eid = try p.add(base, n);
    (try p.get(eid)).owner = sid;
    return eid;
}

fn is(o: ?usize, v: usize) bool {
    return if (o) |x| x == v else false;
}

fn same(a: ?usize, b: ?usize) bool {
    if (a == null or b == null) return a == null and b == null;
    return a.? == b.?;
}

/// The part a placement landed in (null: none).
fn partAt(p: Pool, o: ?usize) ?usize {
    return if (o) |b| p.partOf(b) else null;
}

fn baseOf(p: Pool, eid: u64) usize {
    return (p.extent(eid) catch return std.math.maxInt(usize)).base;
}

/// No two extents share a row and none crosses a part's end.
fn sound(p: Pool) bool {
    const xs = p.extents.items;
    for (xs, 0..) |x, i| {
        if (p.partOf(x.base) != p.partOf(x.end() - 1) or x.end() > p.rows) return false;
        for (xs[i + 1 ..]) |y| {
            if (x.base < y.end() and y.base < x.end()) return false;
        }
    }
    return true;
}

/// Compacts part `part` (with `grow`), each step's move as the caller makes it.
fn compact(p: *Pool, part: usize, grow: ?u64) !void {
    var steps: usize = 0;
    while (try p.nextSlide(part, grow)) |s| {
        _ = try p.move(s.eid, s.base, null, false);
        steps += 1;
        if (steps > 64) return error.Loop;
    }
}

// -- parts (check.py 4, 7) ------------------------------------------------------------------------------------------

fn secParts(t: *T) !void {
    {
        var p = try Pool.initParts(t.gpa, 2 * H, 2);
        defer p.deinit();
        _ = try live(&p, 0, 8 * A, 1);
        _ = try p.add(8 * A, 8 * A); // part 0: one live stream, 16A free
        _ = try live(&p, H, 4 * A, 2);
        _ = try live(&p, H + 4 * A, 4 * A, 3); // part 1: two, 24A free
        t.check("parts: fewer live streams first", is(partAt(p, try p.place(4 * A, null)), 0));
    }
    {
        var p = try Pool.initParts(t.gpa, 2 * H, 2);
        defer p.deinit();
        _ = try p.add(0, 16 * A);
        _ = try p.add(H, 8 * A);
        t.check("parts: on a tie, more free rows", is(partAt(p, try p.place(4 * A, null)), 1));
        t.check("parts: free rows and live streams a part", p.freeIn(0) == H - 16 * A and p.freeIn(1) == H - 8 * A and p.live(0) == 0);
    }
    for ([_]usize{ 3, 4 }) |n| {
        var p = try Pool.initParts(t.gpa, n * H, n);
        defer p.deinit();
        t.check("parts: N parts", p.parts() == n and p.part == H);
        t.check("parts: free rows cut at every part's end", is(try p.place(H, null), 0) and (try p.place(H + A, null)) == null);
        _ = try live(&p, 0, 8 * A, 1);
        _ = try live(&p, H, 8 * A, 2);
        t.check("parts: a new stream to a part with no live stream", is(partAt(p, try p.place(4 * A, null)), 2));
        const crossed = if (p.add(H - A, 2 * A)) |_| true else |e| e != error.CrossesPart;
        t.check("parts: no extent across a part's end", !crossed and sound(p));
        t.check("parts: one in a part", is(try p.placeIn(4 * A, null, 1), H + 8 * A));
    }
    {
        const one = try Pool.initParts(t.gpa, 4 * H, 1);
        t.check("parts: one part is the pool", one.part == 0 and one.parts() == 1 and one.partRange(0).hi == 4 * H);
        const bad = [_]bool{
            if (Pool.initParts(t.gpa, 3 * 2048, 2)) |_| false else |_| true,
            if (Pool.initParts(t.gpa, 9 * H, 9)) |_| false else |_| true,
            if (Pool.initParts(t.gpa, 4 * H, 0)) |_| false else |_| true,
        };
        t.check("parts: not whole extents a part, refused", bad[0] and bad[1] and bad[2]);
    }
}

/// The pool before parts (0db3e2f6): one pool's reference.
const Ref = struct {
    rows: usize,
    xs: std.ArrayList(pool.Extent) = .empty,
    next: u64 = 0,
    fn place(r: Ref, size: usize, ignore: ?u64) ?usize {
        var at: usize = 0;
        for (r.xs.items) |x| {
            if (ignore != null and x.eid == ignore.?) continue;
            if (x.base - at >= size) return at;
            at = x.end();
        }
        return if (r.rows - at >= size) at else null;
    }
    fn ok(r: Ref, base: usize, size: usize, ignore: ?u64) bool {
        if (base % 2048 != 0 or base > r.rows or size > r.rows - base) return false;
        for (r.xs.items) |x| {
            if (ignore != null and x.eid == ignore.?) continue;
            if (x.base < base + size and base < x.end()) return false;
        }
        return true;
    }
    fn sort(r: *Ref) void {
        std.mem.sort(pool.Extent, r.xs.items, {}, struct {
            fn less(_: void, a: pool.Extent, b: pool.Extent) bool {
                return a.base < b.base;
            }
        }.less);
    }
};

fn secOnePool(t: *T) !void {
    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();
    var bad: usize = 0;
    for (0..200) |trial| {
        const rows = rnd.intRangeAtMost(usize, 4, 64) * 2048;
        var p = if (trial % 2 == 0) try Pool.init(t.gpa, rows) else try Pool.initParts(t.gpa, rows, 1);
        defer p.deinit();
        var r: Ref = .{ .rows = rows };
        defer r.xs.deinit(t.gpa);
        for (0..40) |_| {
            const size = std.mem.alignForward(usize, rnd.intRangeAtMost(usize, 1, rows / 2), 2048);
            switch (rnd.intRangeLessThan(u8, 0, 4)) {
                0 => {
                    const want = r.place(size, null);
                    const got = try p.alloc(size);
                    if (want) |b| {
                        try r.xs.append(t.gpa, .{ .base = b, .size = size, .eid = r.next });
                        r.next += 1;
                        r.sort();
                    }
                    if ((want == null) != (got == null)) bad += 1;
                },
                1 => if (r.xs.items.len > 0) {
                    const i = rnd.intRangeLessThan(usize, 0, r.xs.items.len);
                    const eid = r.xs.items[i].eid;
                    _ = r.xs.orderedRemove(i);
                    p.remove(eid) catch {
                        bad += 1;
                    };
                },
                2 => if (r.xs.items.len > 0) {
                    const x = &r.xs.items[rnd.intRangeLessThan(usize, 0, r.xs.items.len)];
                    const want = r.ok(x.base, size, x.eid);
                    const got = if (p.resize(x.eid, size)) |_| true else |_| false;
                    if (want) x.size = size;
                    if (want != got) bad += 1;
                },
                else => if (r.xs.items.len > 0) {
                    const x = r.xs.items[rnd.intRangeLessThan(usize, 0, r.xs.items.len)];
                    const want = r.place(x.size, x.eid);
                    if (!same(want, try p.place(x.size, x.eid))) bad += 1;
                    if (want) |b| {
                        _ = try p.move(x.eid, b, null, false);
                        for (r.xs.items) |*y| {
                            if (y.eid == x.eid) y.base = b;
                        }
                        r.sort();
                    }
                },
            }
            if (p.extents.items.len != r.xs.items.len) {
                bad += 1;
                break;
            }
            for (p.extents.items, r.xs.items) |a, b| {
                if (a.base != b.base or a.size != b.size or a.eid != b.eid) bad += 1;
            }
        }
    }
    t.check("one pool: every placement, growth, move and removal as before parts", bad == 0);
}

// -- twins (check.py 9) ---------------------------------------------------------------------------------------------

fn secTwins(t: *T) !void {
    {
        var p = try Pool.initParts(t.gpa, 3 * H, 3);
        defer p.deinit();
        const x = try live(&p, 0, 8 * A, 1);
        const full = try live(&p, 2 * H, H, 2);
        const none = !(try p.borrow(x));
        var left: usize = 0;
        for (p.extents.items) |e| left += @intFromBool(e.twin != null);
        t.check("twins: all or none (a part without room takes none anywhere)", none and left == 0 and !p.hasTwins(x));
        try p.remove(full);
        const made = try p.borrow(x);
        const t1 = p.twinIn(x, 1);
        const t2 = p.twinIn(x, 2);
        t.check("twins: one in every other part, the extent's size", made and is(t1, H) and is(t2, 2 * H) and p.twinIn(x, 0) == null and p.freeIn(1) == H - 8 * A and p.freeIn(2) == H - 8 * A);
    }
    {
        var p = try Pool.initParts(t.gpa, 2 * H, 2);
        defer p.deinit();
        const x = try live(&p, 0, 8 * A, 1);
        _ = try p.add(H, H - 4 * A);
        t.check("twins: free rows only", !(try p.borrow(x)) and !p.hasTwins(x));
    }
    var p = try Pool.initParts(t.gpa, 2 * H, 2);
    defer p.deinit();
    const x = try live(&p, 0, 8 * A, 1);
    _ = try p.add(H + 8 * A, 4 * A);
    t.check("twins: made", try p.borrow(x));
    const tb = p.twinIn(x, 1) orelse return error.NoTwin;
    var tw: u64 = 0;
    for (p.extents.items) |e| {
        if (e.twin == x) tw = e.eid;
    }
    t.check("twins: borrowed rows are never room", tb == H and p.freeIn(1) == H - 12 * A and is(try p.placeIn(H - 12 * A, null, 1), H + 12 * A) and (try p.placeIn(H - 8 * A, null, 1)) == null);
    const moved = [_]bool{
        if (p.move(tw, H + 16 * A, null, false)) |_| true else |e| e != error.Pinned,
        if (p.move(tw, 8 * A, null, true)) |_| true else |e| e != error.Pinned,
        if (p.resize(tw, 4 * A)) |_| true else |e| e != error.Pinned,
    };
    t.check("twins: a twin never moves, across parts or not", !moved[0] and !moved[1] and !moved[2] and baseOf(p, tw) == tb);
    const crossed = if (p.move(x, H + 16 * A, null, true)) |_| true else |e| e != error.Twinned;
    t.check("twins: an extent with twins never changes part", !crossed and baseOf(p, x) == 0);
    const alone = if (p.remove(tw)) |_| true else |e| e != error.Twin;
    const first = if (p.remove(x)) |_| true else |e| e != error.Twinned;
    const back = if (p.giveBack(x)) |_| true else |_| false;
    const again = if (p.giveBack(x)) |_| true else |e| e != error.NoTwin;
    t.check("twins: given back all at once, before their extent", !alone and !first and back and !p.hasTwins(x) and !again and p.freeIn(1) == H - 4 * A);
    const home = if (p.addTwin(16 * A, x)) |_| true else |e| e != error.SamePart;
    t.check("twins: a twin lies in another part", !home);
    p.remove(x) catch {};
    var one = try Pool.init(t.gpa, 2 * H);
    defer one.deinit();
    const y = try one.add(0, 4 * A);
    const unsharded = if (one.borrow(y)) |_| true else |e| e != error.NoParts;
    t.check("twins: none in one pool", !unsharded);
}

fn secTwoRanks(t: *T) !void {
    var prng = std.Random.DefaultPrng.init(9);
    const rnd = prng.random();
    var bad: usize = 0;
    for (0..100) |_| {
        var ps: [2]Pool = .{ try Pool.initParts(t.gpa, 2 * H, 2), try Pool.initParts(t.gpa, 2 * H, 2) };
        defer for (&ps) |*q| q.deinit();
        var sid: u64 = 1;
        for (0..30) |_| {
            const size = rnd.intRangeAtMost(usize, 1, 12) * A;
            const op = rnd.intRangeLessThan(u8, 0, 4);
            const pick = if (ps[0].extents.items.len == 0) 0 else rnd.intRangeLessThan(usize, 0, ps[0].extents.items.len);
            for (&ps) |*q| {
                switch (op) {
                    0 => if (try q.place(size, null)) |b| {
                        _ = try live(q, b, size, sid);
                    },
                    1 => if (q.extents.items.len > 0) {
                        const e = q.extents.items[pick];
                        if (e.twin == null and !q.hasTwins(e.eid)) _ = try q.borrow(e.eid);
                    },
                    2 => if (q.extents.items.len > 0) {
                        const e = q.extents.items[pick];
                        if (e.twin == null and q.hasTwins(e.eid)) try q.giveBack(e.eid);
                    },
                    else => if (q.extents.items.len > 0) {
                        const e = q.extents.items[pick];
                        if (e.twin == null and !q.hasTwins(e.eid)) try q.remove(e.eid);
                    },
                }
            }
            sid += 1;
        }
        if (ps[0].extents.items.len != ps[1].extents.items.len or !sound(ps[0])) bad += 1;
        for (ps[0].extents.items, ps[1].extents.items) |a, b| {
            if (!std.meta.eql(a, b)) bad += 1;
        }
    }
    t.check("twins: placed alike on every rank (the same calls, the same extents)", bad == 0);
}

// -- segments, gathering and compaction around twins (check.py 9) ---------------------------------------------------

fn secSegments(t: *T) !void {
    var p = try Pool.initParts(t.gpa, 2 * H, 2);
    defer p.deinit();
    const y = try live(&p, H, 8 * A, 1);
    _ = try p.addTwin(8 * A, y);
    _ = try p.add(H + 8 * A, 8 * A);
    var it = p.segments(0, .twins);
    const s0 = it.next();
    const s1 = it.next();
    const s2 = it.next();
    t.check("segments: cut at a twin and at the part", s0 != null and s1 != null and s2 == null and s0.?.lo == 0 and s0.?.hi == 8 * A and s1.?.lo == 16 * A and s1.?.hi == H);
    t.check("segments: compactable is the largest segment, not the sum", p.compactable(0, .twins) == H - 16 * A and p.compactable(1, .twins) == H - 16 * A);
    const at16 = p.segmentAt(0, .twins, 16 * A);
    t.check("segments: a twin's row is in none", p.segmentAt(0, .twins, 8 * A) == null and at16 != null and at16.?.lo == 16 * A);
}

fn secGather(t: *T) !void {
    {
        var p = try Pool.initParts(t.gpa, 2 * H, 2);
        defer p.deinit();
        _ = try live(&p, 0, 20 * A, 1);
        _ = try live(&p, H, 20 * A, 2);
        t.check("gatherable: free rows spread over parts are no room for one prompt", @max(p.gatherable(0, .held), p.gatherable(1, .held)) < 16 * A);
    }
    var p = try Pool.initParts(t.gpa, 2 * H, 2);
    defer p.deinit();
    const x = try live(&p, 0, 30 * A, 1);
    _ = try p.borrow(x);
    t.check("gatherable: a twin between chunks is room (it can be given back)", p.gatherable(1, .held) == H and p.compactable(1, .twins) == 2 * A);
    p.hold(x, true);
    t.check("gatherable: a held twin is no room", p.gatherable(1, .held) == 2 * A);
    p.hold(x, false);
    t.check("gatherable: released, room again", p.gatherable(1, .held) == H);
}

fn secCompact(t: *T) !void {
    {
        // part 1: free 8A | twin 8A | free 8A | k 8A
        var p = try Pool.initParts(t.gpa, 2 * H, 2);
        defer p.deinit();
        const x = try live(&p, 0, 8 * A, 1);
        const tw = try p.addTwin(H + 8 * A, x);
        const k = try p.add(H + 24 * A, 8 * A);
        const ok = if (compact(&p, 1, null)) |_| true else |_| false;
        t.check("compact: a twin stays where it is", ok and baseOf(p, tw) == H + 8 * A and sound(p));
        t.check("compact: nothing slides past a twin", ok and baseOf(p, k) == H + 16 * A);
        t.check("compact: the segments' free rows gathered, each its own", p.compactable(1, .twins) == 8 * A and (try p.placeIn(10 * A, null, 1)) == null);
    }
    {
        // free 4A | g 4A | free 4A | e 2A | free 2A | twin 8A | free 8A
        var p = try Pool.initParts(t.gpa, 2 * H, 2);
        defer p.deinit();
        const x = try live(&p, 0, 8 * A, 1);
        const tw = try p.addTwin(H + 16 * A, x);
        const g = try live(&p, H + 4 * A, 4 * A, 7);
        const e = try live(&p, H + 12 * A, 2 * A, 8);
        const fits = (try p.segmentRoom(g)) + 4 * A >= 14 * A;
        const ok = if (compact(&p, 1, g)) |_| true else |_| false;
        const room = p.roomAfter(g) catch 0;
        const grown = ok and room >= 10 * A and if (p.resize(g, 14 * A)) |_| true else |_| false;
        t.check("compact: around a twin the segment's rows gather after the extent", fits and grown and baseOf(p, g) == H and (try p.extent(g)).size == 14 * A and baseOf(p, e) == H + 14 * A and baseOf(p, tw) == H + 16 * A and sound(p));
        t.check("compact: past its segment's rows a grow is refused", (try p.segmentRoom(g)) + (try p.extent(g)).size < 20 * A);
    }
    {
        var p = try Pool.initParts(t.gpa, 2 * H, 2);
        defer p.deinit();
        const x = try p.add(H - 8 * A, 4 * A);
        t.check("roomAfter: stops at its part's end", (try p.roomAfter(x)) == 4 * A);
        const other = if (p.nextSlide(1, x)) |_| true else |e| e != error.OtherPart;
        t.check("compact: a grow in its own part only", !other);
    }
}

fn secCompactRandom(t: *T) !void {
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    var bad: usize = 0;
    var runs: usize = 0;
    for (0..300) |_| {
        var p = try Pool.initParts(t.gpa, 2 * H, 2);
        defer p.deinit();
        const x = try live(&p, 0, 8 * A, 1);
        var at: usize = H;
        var tb: ?usize = null;
        var others: std.ArrayList(u64) = .empty;
        defer others.deinit(t.gpa);
        var sid: u64 = 100;
        while (true) {
            const n = rnd.intRangeAtMost(usize, 1, 6) * A;
            const kind = rnd.intRangeLessThan(u8, 0, 4);
            if (kind == 3 and tb == null and at + 8 * A <= 2 * H) {
                _ = try p.addTwin(at, x);
                tb = at;
                at += 8 * A;
            } else {
                if (at + n > 2 * H) break;
                if (kind == 1) try others.append(t.gpa, try live(&p, at, n, sid)) else if (kind == 2) try others.append(t.gpa, try p.add(at, n));
                sid += 1;
                at += n;
            }
            at += if (rnd.boolean()) 0 else A;
            if (at >= 2 * H) break;
        }
        const twin_at = tb orelse continue;
        runs += 1;
        p.hold(x, rnd.boolean());
        const grow: ?u64 = if (others.items.len > 0 and rnd.boolean()) others.items[rnd.intRangeLessThan(usize, 0, others.items.len)] else null;
        compact(&p, 1, grow) catch {
            bad += 1;
            continue;
        };
        if (p.twinIn(x, 1) != twin_at or !sound(p)) {
            bad += 1;
            continue;
        }
        const need = rnd.intRangeAtMost(usize, 1, 20) * A;
        if (try p.placeIn(need, null, 1)) |b| {
            if (p.add(b, need)) |_| {} else |_| bad += 1;
        }
    }
    t.check("compact: random layouts: a twin never moves, nothing overlaps, a placement lands on free rows", bad == 0 and runs > 100);
}

// -- moves across parts (check.py 10 pool) --------------------------------------------------------------------------

fn secCross(t: *T) !void {
    var p = try Pool.initParts(t.gpa, 2 * H, 2);
    defer p.deinit();
    const x = try p.add(0, 8 * A);
    const refused = if (p.move(x, H, null, false)) |_| false else |e| e == error.OtherPart;
    t.check("cross: a move to another part is refused without cross", refused and baseOf(p, x) == 0);
    t.check("cross: made with cross (the old base back)", (try p.move(x, H, 8 * A, true)) == 0 and baseOf(p, x) == H);
    const straddles = if (p.move(x, H - A, null, true)) |_| true else |e| e != error.CrossesPart;
    t.check("cross: even with cross an extent never straddles a part's end", !straddles and baseOf(p, x) == H);
    const tw = try p.addTwin(0, x);
    const twin_moved = if (p.move(tw, H + 16 * A, null, true)) |_| true else |_| false;
    t.check("cross: a twin never moves across parts", !twin_moved);
    _ = try p.add(H + 8 * A, 4 * A);
    const taken = if (p.move(x, H + 4 * A, 8 * A, true)) |_| true else |e| e != error.Overlap;
    t.check("cross: a move onto taken rows is refused", !taken and baseOf(p, x) == H);
}

// -- sizes (check.py 8, 9 floor, 9 agree) ---------------------------------------------------------------------------

fn secSizes(t: *T) !void {
    const lat = shard.latent_row;
    const pk = shard.pooled_bytes;
    const L = G.dsa_layers;
    for ([_]usize{ 2, 3, 4 }, [_]f64{ 1.88, 2.66, 3.36 }) |n, want| {
        for ([_]usize{ 2048, shard.trash }) |tr| {
            const rows = shard.budget(2_000_000, n, tr);
            const x = @as(f64, @floatFromInt(rows)) / 2_000_000.0;
            t.check("sizes: whole parts of extents, the memory of 2M unsharded tokens holds ~1.88x/2.66x/3.36x", rows % (n * 2048) == 0 and @abs(x - want) < 0.06);
        }
    }
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    var over: usize = 0;
    var loose: usize = 0;
    for (0..300) |_| {
        const unsh = rnd.intRangeAtMost(usize, 64, 2048) * 2048;
        const tr = if (rnd.boolean()) shard.trash else 2048;
        const rows = shard.budget(unsh, 2, tr);
        if (rows == 0) continue;
        const used = rows * L * pk + (rows / 2 + tr) * L * lat;
        const mem = unsh * L * (lat + pk);
        if (used > mem) over += 1;
        const more = rows + 4096;
        if (more * L * pk + (more / 2 + tr) * L * lat <= mem) loose += 1;
    }
    t.check("sizes: never past the memory, and not a part of extents short of it", over == 0 and loose == 0);
    t.check("sizes: no room for the trash rows, no pool", shard.budget(1, 2, 2048) == 0);
    const lim = 1_048_576;
    const shrunk = [_][2]usize{ .{ 2, 1_118_208 }, .{ 2, 1_060_864 }, .{ 3, 1_060_864 }, .{ 4, 1_060_864 } };
    for (shrunk) |c| {
        const have = shard.budget(c[1], c[0], 2048);
        const part = have / c[0];
        const pl = try shard.plan(c[1], lim, 16, c[0], 2048);
        t.check("plan: a part short of the context shrinks it (to a part less a decode window), never past the budget", pl.rows == have and pl.part == part and part - 16 < lim and pl.context == part - 16);
    }
    const big = try shard.plan(4_000_000, lim, 16, 2, 2048);
    t.check("plan: a budget that holds the context is untouched", big.rows == shard.budget(4_000_000, 2, 2048) and big.context == lim);
    for ([_][2]usize{ .{ 2, 100_000 }, .{ 2, 250_000 }, .{ 4, 150_000 } }) |c| {
        const refused = if (shard.plan(c[1], lim, 16, c[0], 2048)) |_| false else |e| e == error.PartTooSmall;
        t.check("plan: a part too small (under 131,072 or a quarter of the context) is refused", refused);
    }
    t.check("plan: a smaller context a part holds is untouched", (try shard.plan(1_060_864, 262_144, 16, 2, 2048)).context == 262_144);
    const empty = if (shard.plan(2048, 2048, 64, 2, 64)) |_| false else |e| e == error.PartTooSmall;
    t.check("plan: a part a window or less is refused", empty);
    const a: u64 = 2_000_000 / 2048 * 2048;
    const b: u64 = a - 3 * 2048;
    t.check("plan: ranks sizing alone can disagree on a part's rows", shard.budget(a, 2, 2048) / 2 != shard.budget(b, 2, 2048) / 2);
    const ag = shard.agree(&.{ .{ a, 1000 }, .{ b, 500 }, .{ a, 700 } });
    t.check("agree: the smallest pool and display rows on every rank", ag[0] == b and ag[1] == 500);
    const rows = shard.budget(a, 2, shard.trash);
    t.check("display: the reservation at a sharded token's cost, never past the pool", shard.displayRows(rows, a, 0, 2, shard.trash) == 0 and shard.displayRows(rows, a, 65536, 2, shard.trash) == rows - shard.budget(a - 65536, 2, shard.trash) and shard.displayRows(rows, a, a, 2, shard.trash) == rows);
    t.check("sizes: a carry sends every DSA layer's latent rows", shard.carryBytes(10) == 10 * 528 * 11 and shard.pooled_bytes == 36);
}

// -- addresses and carries (check.py 10 carry) ------------------------------------------------------------------------

fn secAddress(t: *T) !void {
    const P = 8 * 2048;
    const errs = [_]bool{
        if (shard.Shard.init(0, 3, 3 * P, 64)) |_| false else |e| e == error.KvShardNeedsTwoRanks,
        if (shard.Shard.init(2, 2, 2 * P, 64)) |_| false else |e| e == error.BadRank,
        if (shard.Shard.init(0, 2, 2 * P + 2048, 64)) |_| false else |e| e == error.PoolNotWholeParts,
        if (shard.Shard.init(0, 2, 2 * P, 0)) |_| false else |e| e == error.BadTrash,
    };
    t.check("address: TP2, whole parts of extents and trash rows, or refused", errs[0] and errs[1] and errs[2] and errs[3]);
    const s0 = try shard.Shard.init(0, 2, 2 * P, 64);
    const s1 = try shard.Shard.init(1, 2, 2 * P, 64);
    t.check("address: a part a rank", s0.owner(P - 1) == 0 and s0.owner(P) == 1 and s1.owns(P) and !s1.owns(0) and s0.peer() == 1);
    t.check("address: an owned row in this rank's plane", s1.local(P + 2048) == 2048 and s0.local(2048) == 2048 and is(s1.latentBase(P + 4096), 4096) and s0.latentBase(P + 4096) == null);
    t.check("address: a part's rows and its trash rows a rank", s0.latentRows() == P + 64 and s1.trashRow(5) == P + 5);
    t.check("carry: within a part its owner copies, the other rank nothing", std.meta.eql(try s0.copy(0, 4096, 3000), shard.Move{ .local = .{ .from = 0, .to = 4096, .rows = 3000 } }) and (try s1.copy(0, 4096, 3000)) == .none and std.meta.eql(try s1.copy(P, P + 4096, 3000), shard.Move{ .local = .{ .from = 0, .to = 4096, .rows = 3000 } }));
    const across = if (s0.copy(0, P, 10)) |_| false else |e| e == error.OtherPart;
    t.check("carry: a copy never crosses parts", across);
    t.check("carry: the source's owner sends its rows, the destination's receives them (either way)", std.meta.eql(try s0.carry(2048, P + 4096, 3001), shard.Move{ .send = .{ .from = 2048, .rows = 3001, .peer = 1 } }) and std.meta.eql(try s1.carry(2048, P + 4096, 3001), shard.Move{ .recv = .{ .to = 4096, .rows = 3001, .peer = 0 } }) and std.meta.eql(try s1.carry(P + 2048, 4096, 5000), shard.Move{ .send = .{ .from = 2048, .rows = 5000, .peer = 0 } }) and std.meta.eql(try s0.carry(P + 2048, 4096, 5000), shard.Move{ .recv = .{ .to = 4096, .rows = 5000, .peer = 1 } }));
    t.check("carry: within one part it is the local copy", std.meta.eql(try s0.carry(0, 4096, 3000), try s0.copy(0, 4096, 3000)) and (try s1.carry(0, 4096, 3000)) == .none);
    t.check("carry: zero tokens is nothing", (try s0.carry(0, P, 0)) == .none and (try s1.carry(0, P, 0)) == .none);
    const past = [_]bool{
        if (s0.carry(0, 2 * P - 1000, 3000)) |_| false else |e| e == error.CrossesPart,
        if (s0.carry(P - 1000, P, 3000)) |_| false else |e| e == error.CrossesPart,
        if (s0.copy(0, 2 * P, 1)) |_| false else |e| e == error.CrossesPart,
    };
    t.check("carry: rows past a part's end or the pool are refused", past[0] and past[1] and past[2]);
    const sp = shard.pooledRows(2048, P + 4096, 3001);
    t.check("carry: the pooled keys on every rank, a last row part full taken whole", sp.from == 512 and sp.to == P / 4 + 1024 and sp.rows == 751);
}

// -- a decode window's glue (segments.py) ---------------------------------------------------------------------------

fn secGlue(t: *T) !void {
    const P = 8 * 2048;
    const s0 = try shard.Shard.init(0, 2, 2 * P, 64);
    const s1 = try shard.Shard.init(1, 2, 2 * P, 64);
    const pos = [_]i32{ 2050, 2051, 2052, 7, 8, 0 };
    const base = [_]i32{ 0, 0, 0, P, P, 0 };
    const sparse = [_]i32{ 0, 1, 1, 0, 0, 0 };
    var planes: [2][5][6]i32 = undefined;
    for (0..2) |r| {
        const s = if (r == 0) s0 else s1;
        for (&planes[r]) |*pl| pl.* = @splat(9);
        const out: [5][]i32 = .{ &planes[r][0], &planes[r][1], &planes[r][2], &planes[r][3], &planes[r][4] };
        try shard.glue(s, 5, &pos, &base, &sparse, &out);
    }
    const lb = @backingInt(shard.Glue.lbase);
    const own = @backingInt(shard.Glue.own);
    const se = @backingInt(shard.Glue.sparse_eff);
    const wp = @backingInt(shard.Glue.wpos);
    const orank = @backingInt(shard.Glue.orank);
    const eq = std.mem.eql;
    t.check("glue: a row it owns at its stream's latent rows and position", eq(i32, &planes[0][lb], &.{ 0, 0, 0, P, P, 0 }) and eq(i32, &planes[1][lb], &.{ P, P, P, 0, 0, 0 }) and eq(i32, planes[0][wp][0..3], pos[0..3]) and eq(i32, planes[1][wp][3..5], pos[3..5]));
    t.check("glue: a row it does not own writes its own trash row", eq(i32, &planes[0][wp], &.{ 2050, 2051, 2052, 3, 4, 0 }) and eq(i32, &planes[1][wp], &.{ 0, 1, 2, 7, 8, 0 }));
    t.check("glue: a row it does not own reads no key", eq(i32, &planes[0][se], &.{ 0, 1, 1, 1, 1, 0 }) and eq(i32, &planes[1][se], &.{ 1, 1, 1, 0, 0, 0 }));
    t.check("glue: whether it owns each row, and every row's owner alike on both ranks", eq(i32, &planes[0][own], &.{ 1, 1, 1, 0, 0, 0 }) and eq(i32, &planes[1][own], &.{ 0, 0, 0, 1, 1, 0 }) and eq(i32, &planes[0][orank], &.{ 0, 0, 0, 1, 1, 0 }) and eq(i32, &planes[1][orank], &planes[0][orank]));
    var big: [5][65]i32 = undefined;
    const wide: [65]i32 = @splat(0);
    const out: [5][]i32 = .{ &big[0], &big[1], &big[2], &big[3], &big[4] };
    const refused = if (shard.glue(s0, 65, &wide, &wide, &wide, &out)) |_| false else |e| e == error.WindowPastTrash;
    t.check("glue: a window past the trash rows is refused", refused);
    t.check("glue: its planes after the table's row and segment planes", shard.glueOffset(64, 4, .lbase) == 5 * 64 + 16 and shard.glueOffset(64, 4, .orank) == 9 * 64 + 16);
}

// -- a prompt chunk's history fetch (shard.history, shard.fetch) ----------------------------------------------------

fn lowerBound(xs: []const u32, v: u32) usize {
    var lo: usize = 0;
    var hi = xs.len;
    while (lo < hi) {
        const m = (lo + hi) / 2;
        if (xs[m] < v) lo = m + 1 else hi = m;
    }
    return lo;
}

fn secFetch(t: *T) !void {
    var prng = std.Random.DefaultPrng.init(17);
    const rnd = prng.random();
    const row = shard.latent_row;
    var bad_hist: usize = 0;
    var bad_ids: usize = 0;
    var bad_rows: usize = 0;
    for (0..60) |trial| {
        const host_pos = rnd.intRangeAtMost(usize, 2, 3000);
        const R = rnd.intRangeAtMost(usize, 1, 8);
        const tokens = try t.gpa.alloc(i32, R * G.tokens);
        defer t.gpa.free(tokens);
        for (tokens, 0..) |*x, i| {
            const q = host_pos + i / G.tokens; // the row's position: it reads the history and its chunk's rows up to it
            x.* = switch (rnd.intRangeLessThan(u8, 0, 4)) {
                0 => -1,
                1 => @intCast(rnd.intRangeAtMost(usize, host_pos, q)),
                else => @intCast(rnd.intRangeLessThan(usize, if (trial % 3 == 0) 1 else 0, host_pos)),
            };
        }
        if (trial % 3 == 0) tokens[0] = -1; // (no token names position 0: a -1 must not fetch it)
        const marks = try t.gpa.alloc(i32, host_pos);
        defer t.gpa.free(marks);
        const cum = try t.gpa.alloc(i32, host_pos);
        defer t.gpa.free(cum);
        shard.markHistory(tokens, host_pos, marks);
        const n = shard.prefixSum(marks, cum);
        const hbuf = try t.gpa.alloc(u32, host_pos);
        defer t.gpa.free(hbuf);
        const hist = shard.historyRows(marks, hbuf);
        // the reference: a sort and a dedup
        var want: std.ArrayList(u32) = .empty;
        defer want.deinit(t.gpa);
        for (tokens) |x| {
            if (x >= 0 and @as(usize, @intCast(x)) < host_pos) try want.append(t.gpa, @intCast(x));
        }
        std.mem.sort(u32, want.items, {}, std.sort.asc(u32));
        var u: usize = 0;
        for (want.items, 0..) |v, i| {
            if (i == 0 or v != want.items[i - 1]) {
                want.items[u] = v;
                u += 1;
            }
        }
        if (n != u or !std.mem.eql(u32, hist, want.items[0..u])) bad_hist += 1;
        const ids = try t.gpa.alloc(i32, tokens.len);
        defer t.gpa.free(ids);
        shard.renumber(tokens, host_pos, cum, ids);
        for (tokens, ids) |x, got| {
            const exp: i32 = if (x < 0) -1 else if (@as(usize, @intCast(x)) < host_pos) @intCast(lowerBound(want.items[0..u], @intCast(x))) else @as(i32, @intCast(u)) + x - @as(i32, @intCast(host_pos));
            if (exp != got) bad_ids += 1;
        }
        // the arena's rows of the stream (history and the chunk's own), and the compact copy the other rank attends
        const arena = try t.gpa.alloc(u8, (host_pos + R) * row);
        defer t.gpa.free(arena);
        rnd.bytes(arena);
        const comp = try t.gpa.alloc(u8, (n + R) * row);
        defer t.gpa.free(comp);
        shard.gatherRows(arena, row, hist, comp);
        @memcpy(comp[n * row ..], arena[host_pos * row ..]);
        for (tokens, ids) |x, got| {
            if (x < 0) continue;
            if (got < 0 or @as(usize, @intCast(got)) >= n + R) {
                bad_rows += 1;
                continue;
            }
            const a = arena[@as(usize, @intCast(x)) * row ..][0..row];
            const c = comp[@as(usize, @intCast(got)) * row ..][0..row];
            if (!std.mem.eql(u8, a, c)) bad_rows += 1;
        }
    }
    t.check("fetch: the owner sends the marked positions, ascending, each once (no -1)", bad_hist == 0);
    t.check("fetch: renumbered ids are a sort and a search (the chunk's own rows after the fetched)", bad_ids == 0);
    t.check("fetch: the compact copy holds the rows the arena does, for every token", bad_rows == 0);
    // a dense chunk: every history row fetched, the copy is the arena
    const host_pos = 300;
    var tokens: [host_pos + 4]i32 = undefined;
    for (&tokens, 0..) |*x, i| x.* = @intCast(i);
    var marks: [host_pos]i32 = undefined;
    var cum: [host_pos]i32 = undefined;
    var ids: [host_pos + 4]i32 = undefined;
    shard.markHistory(&tokens, host_pos, &marks);
    const n = shard.prefixSum(&marks, &cum);
    shard.renumber(&tokens, host_pos, &cum, &ids);
    t.check("fetch: a dense chunk's copy is the history itself", n == host_pos and std.mem.eql(i32, &ids, &tokens));
    t.check("fetch: pieces of 64 MiB, both ranks alike", shard.piece(528) == 127100 and shard.piece(1 << 30) == 1);
}

// -- owner decode's crossing rows -----------------------------------------------------------------------------------

fn secLists(t: *T) !void {
    var prng = std.Random.DefaultPrng.init(23);
    const rnd = prng.random();
    var mirror: usize = 0;
    var which: usize = 0;
    var orank: [shard.max_rows]i32 = undefined;
    for (0..200) |_| {
        const n = rnd.intRangeAtMost(usize, 0, shard.max_rows);
        for (orank[0..n]) |*o| o.* = rnd.intRangeAtMost(i32, 0, 1);
        const ls = [2]shard.Lists{ try shard.lists(orank[0..n], 0), try shard.lists(orank[0..n], 1) };
        for (0..2) |r| {
            for ([_]shard.Exchange{ .q, .out }) |x| {
                if (!ls[r].of(x, true).eql(ls[1 - r].of(x, false).*)) mirror += 1;
            }
            for (ls[r].of(.q, true).slice()) |i| {
                if (@as(usize, @intCast(orank[i])) == r) which += 1;
            }
            for (ls[r].of(.out, true).slice()) |i| {
                if (@as(usize, @intCast(orank[i])) != r) which += 1;
            }
            if (ls[r].of(.q, true).n + ls[r].of(.q, false).n != n) which += 1;
        }
    }
    t.check("lists: rank r's sends are rank 1 - r's receives", mirror == 0);
    t.check("lists: a rank sends its queries of the rows the peer owns and the peer's heads of its own", which == 0);
    const long: [shard.max_rows + 1]i32 = @splat(0);
    const errs = [_]bool{
        if (shard.lists(&long, 0)) |_| false else |e| e == error.BadRows,
        if (shard.lists(&.{ 0, 2 }, 0)) |_| false else |e| e == error.BadRows,
    };
    t.check("lists: a window past its rows or a third rank is refused", errs[0] and errs[1]);
}

fn secCursor(t: *T) !void {
    var c: shard.Cursor = .{};
    try c.begin(.decode);
    try c.take(.q);
    const early = c.begin(.prompt);
    try c.take(.out);
    t.check("cursor: a layer's transfers in order", c.done() and if (early) |_| false else |e| e == error.ShardOrder);
    try c.begin(.decode);
    const wrong = if (c.take(.out)) |_| false else |e| e == error.ShardOrder;
    var d: shard.Cursor = .{};
    try d.begin(.prompt);
    const extra = blk: {
        d.take(.fetch) catch break :blk false;
        break :blk if (d.take(.fetch)) |_| false else |e| e == error.ShardOrder;
    };
    t.check("cursor: another order is refused", wrong and !c.done() and extra);
}

// -- step B: the switches, the sharded pool's agreement and the placement check (no Python counterpart) --------------

fn secSwitches(t: *T) !void {
    const off = try shard.Switches.parse(null, null, null, null, null);
    t.check("switches: unset is off", !off.on);
    const zero = try shard.Switches.parse("0", "bad", "bad", "bad", "1");
    t.check("switches: off reads no knob", std.meta.eql(zero, shard.Switches{}));
    const on = try shard.Switches.parse("1", null, "", null, null);
    t.check("switches: on carries and borrows, a layout digest every 1,024 decode rounds, by default", on.on and on.carry and on.twins and on.check == 1024 and !on.lone);
    const set = try shard.Switches.parse(" 1 ", "0", "0", "1", "1");
    t.check("switches: each knob read", set.on and !set.carry and !set.twins and set.check == 1 and set.lone);
    // TF_GLM_KV_SHARD_CHECK is a count of decode rounds (DS TF_DS_KV_SHARD_CHECK, the recipe's port.sh): 0 sends no digest
    const rounds = try shard.Switches.parse("1", null, null, " 4096 ", null);
    const none = try shard.Switches.parse("1", null, null, "0", null);
    const most = try shard.Switches.parse("1", null, null, "4294967295", null);
    t.check("switches: the check a count of decode rounds", rounds.check == 4096 and none.check == 0 and most.check == std.math.maxInt(u32));
    t.check("switches: a digest every that many rounds, none at 0 or off", !rounds.digestAt(0) and !rounds.digestAt(4095) and rounds.digestAt(4096) and rounds.digestAt(8192) and set.digestAt(1) and !none.digestAt(1024) and !off.digestAt(1024));
    const bad = [_]struct { [5]?[]const u8, anyerror }{
        .{ .{ "yes", null, null, null, null }, error.BadKvShard },
        .{ .{ "1", "2", null, null, null }, error.BadKvShardCarry },
        .{ .{ "1", null, "on", null, null }, error.BadKvShardTwins },
    };
    for (bad) |c| {
        const refused = if (shard.Switches.parse(c[0][0], c[0][1], c[0][2], c[0][3], c[0][4])) |_| false else |e| e == c[1];
        t.check("switches: a value other than 0 or 1 is this rank's error", refused);
    }
    for ([_][]const u8{ "true", "-1", "+5", "1_000", "0x10", "1e3", "4294967296" }) |v| {
        const refused = if (shard.Switches.parse("1", null, null, v, null)) |_| false else |e| e == error.BadKvShardCheck;
        t.check("switches: a check that is not a count of decimal digits is this rank's error", refused);
    }
    var buf: [80]u8 = undefined;
    t.check("switches: off adds nothing to the settings text", off.text(&buf).len == 0);
    t.check("switches: on adds every knob to it", std.mem.eql(u8, set.text(&buf), " kv_shard=1 carry=0 twins=0 check=1 lone=1") and std.mem.eql(u8, most.text(&buf), " kv_shard=1 carry=1 twins=1 check=4294967295 lone=0"));
    t.check("switches: the unreadable word is no rank's text", std.mem.indexOf(u8, shard.unreadable, "kv_shard=") != null and !std.mem.eql(u8, shard.unreadable, on.text(&buf)));
    const R = struct {
        fn refused(s: shard.Switches, world: u32, parallel: u32, mtp: bool, want: anyerror) bool {
            return if (shard.refuse(s, world, parallel, mtp)) |_| false else |e| e == want;
        }
    };
    t.check("refuse: off refuses nothing", if (shard.refuse(off, 1, 1, true)) |_| true else |_| false);
    t.check("refuse: two ranks only", R.refused(on, 1, 4, false, error.KvShardNeedsTwoRanks) and R.refused(on, 4, 4, false, error.KvShardNeedsTwoRanks));
    t.check("refuse: more than one stream", R.refused(on, 2, 1, false, error.KvShardNeedsParallel));
    t.check("refuse: no MTP drafter", R.refused(on, 2, 2, true, error.KvShardWithMtp));
    t.check("refuse: no TF_GLM_MULTI_LONE", R.refused(set, 2, 8, false, error.KvShardWithLone));
    t.check("refuse: two ranks, streams, DFlash2 or no drafts", if (shard.refuse(on, 2, 8, false)) |_| true else |_| false);
    var worded = true;
    for ([_]anyerror{ error.BadKvShard, error.BadKvShardCarry, error.BadKvShardTwins, error.BadKvShardCheck, error.KvShardNeedsTwoRanks, error.KvShardNeedsParallel, error.KvShardWithMtp, error.KvShardWithLone, error.KvShardNotBuilt, error.NoKvShardKernels }) |e| worded = worded and shard.why(e) != null;
    t.check("refuse: every refusal in words", worded and shard.why(error.OutOfMemory) == null and std.mem.indexOf(u8, shard.why(error.NoKvShardKernels).?, "glm_shard.fatbin") != null);
    t.check("refuse: this build carries the GPU paths", shard.gpu_paths);
    // glm_shard.cu's image is optional in the build: one that embeds kernels without it refuses sharding at start
    const no_image = if (shard.kernelsIn(true, &.{})) |_| false else |e| e == error.NoKvShardKernels;
    const fine = if (shard.kernelsIn(true, "fatbin")) |_| true else |_| false;
    const host = if (shard.kernelsIn(false, &.{})) |_| true else |_| false;
    t.check("refuse: kernels embedded without glm_shard's, but not a host-only build or one with them", no_image and fine and host);
}

fn secAgreePool(t: *T) !void {
    for ([_]usize{ 1 << 20, 3_000_000 / 2048 * 2048 }) |u| {
        const a = try shard.plan(u, 1 << 20, 64, 2, shard.trash);
        const b = try shard.fit(shard.budget(u, 2, shard.trash), 1 << 20, 64, 2);
        t.check("fit: plan is fit over budget", std.meta.eql(a, b));
    }
    t.check("same: every rank's sharded pool alike, or not", shard.same(&.{ 8192, 8192 }) and !shard.same(&.{ 8192, 4096 }) and shard.same(&.{4096}));
}

fn secPlacement(t: *T) !void {
    const gpa = t.gpa;
    // two ranks' pools made by the same calls
    var p0 = try Pool.initParts(gpa, 2 * H, 2);
    defer p0.deinit();
    var p1 = try Pool.initParts(gpa, 2 * H, 2);
    defer p1.deinit();
    // rank 1 opened a lane and dropped it before rank 0 opened any (the ranks refused that prefill): its ids run one
    // ahead from here on, its layout does not
    try p1.remove((try p1.alloc(A)).?);
    var eids: [3]u64 = undefined; // rank 0's
    var fids: [3]u64 = undefined; // rank 1's
    for (&eids, &fids, 0..) |*e, *f, i| {
        e.* = (try p0.alloc(A)).?;
        (try p0.get(e.*)).owner = i;
        f.* = (try p1.alloc(A)).?;
        (try p1.get(f.*)).owner = i;
    }
    t.check("digest: the same calls, the same layout", p0.digest() == p1.digest() and p0.digest() != 0);
    const theirs = p1.extentAt(H) orelse return error.NoExtent;
    t.check("placement: ids each pool's own (a lane one rank made and dropped)", p0.next_id + 1 == p1.next_id and theirs.eid == fids[1] and fids[1] != eids[1] and (try p0.extent(eids[1])).base == H);
    const at = try shard.Placement.of(&p0, eids[1], true);
    var buf: [shard.Placement.bytes]u8 = undefined;
    at.encode(&buf);
    const back = try shard.Placement.decode(&buf);
    t.check("placement: a record round trip", std.meta.eql(at, back) and at.base == H and at.part == 1 and at.size == A and at.owner == 1);
    t.check("placement: a short record is refused", if (shard.Placement.decode(buf[1..])) |_| false else |e| e == error.BadPlacement);
    t.check("placement: rank 1's own alike passes, under another id", if (back.verify(&p1)) |_| true else |_| false);
    var other = at;
    other.owner = 2;
    t.check("placement: another slot's extent there is refused", if (other.verify(&p1)) |_| false else |e| e == error.KvShardPlacement);
    // rank 1 drifted: an extent elsewhere (not the one placed)
    _ = try p1.move(fids[0], 2 * A, null, false);
    const plain = try shard.Placement.of(&p0, eids[1], false);
    t.check("placement: without the check only the placement is compared", plain.digest == 0 and if (plain.verify(&p1)) |_| true else |_| false);
    t.check("placement: the digest compares the whole layout", if (at.verify(&p1)) |_| false else |e| e == error.KvShardPlacement);
    _ = try p1.move(fids[0], 0, null, false);
    t.check("digest: the layout back, the digest back", p0.digest() == p1.digest());
    // the placed extent itself elsewhere, of other rows, or missing
    _ = try p1.move(fids[1], H + A, null, false);
    t.check("placement: another base is refused", if (plain.verify(&p1)) |_| false else |e| e == error.KvShardPlacement);
    _ = try p1.move(fids[1], H, null, false);
    try p1.resize(fids[1], 2 * A);
    t.check("placement: other rows are refused", if (plain.verify(&p1)) |_| false else |e| e == error.KvShardPlacement);
    try p1.remove(fids[1]);
    t.check("placement: an extent rank 1 does not hold is refused", if (plain.verify(&p1)) |_| false else |e| e == error.KvShardPlacement);
    // twins count in the layout
    var q = try Pool.initParts(gpa, 2 * H, 2);
    defer q.deinit();
    const x = try live(&q, 0, A, 1);
    const before = q.digest();
    _ = try q.borrow(x);
    t.check("digest: a twin changes the layout", q.digest() != before);
    // the same twins on a pool whose ids ran ahead: the twin named by its extent's base, so alike
    var r = try Pool.initParts(gpa, 2 * H, 2);
    defer r.deinit();
    try r.remove(try r.add(4 * A, A));
    const y = try live(&r, 0, A, 1);
    _ = try r.borrow(y);
    t.check("digest: twins alike under other ids", r.digest() == q.digest() and y != x);
    const alike = if (shard.checkLayout(&r, q.digest())) |_| true else |_| false;
    const drifted = if (shard.checkLayout(&r, before)) |_| false else |e| e == error.KvShardPlacement;
    t.check("layout: rank 1's alike passes, another refused", alike and drifted);
}

// -- the suite, as it is and mutated --------------------------------------------------------------------------------

const sections = .{
    .{ "parts", secParts },
    .{ "one pool", secOnePool },
    .{ "twins", secTwins },
    .{ "two ranks", secTwoRanks },
    .{ "segments", secSegments },
    .{ "gather", secGather },
    .{ "compact", secCompact },
    .{ "compact random", secCompactRandom },
    .{ "cross", secCross },
    .{ "sizes", secSizes },
    .{ "address", secAddress },
    .{ "glue", secGlue },
    .{ "fetch", secFetch },
    .{ "lists", secLists },
    .{ "cursor", secCursor },
    .{ "switches", secSwitches },
    .{ "agree pool", secAgreePool },
    .{ "placement", secPlacement },
};

fn suite(gpa: std.mem.Allocator, verbose: bool) T {
    var t: T = .{ .gpa = gpa, .verbose = verbose };
    inline for (sections) |s| t.section(s[0], s[1]);
    return t;
}

test "kv shard checks, host side (GLM tools/kv_shard_check.py)" {
    var t = suite(std.testing.allocator, true);
    defer t.failed.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), t.failed.items.len);
    try std.testing.expectEqual(@as(usize, 0), t.crash);
    try std.testing.expect(t.pass > 70);
}

/// The check written for each pool mutant (its name's start): the mutant must fail it, not merely some check.
fn poolKiller(m: pool.Mutant) []const u8 {
    return switch (m) {
        .none => unreachable,
        .cross_part_add => "parts: no extent across a part's end",
        .gaps_uncut => "parts: free rows cut at every part's end",
        .order_ignores_live => "parts: fewer live streams first",
        .order_ignores_free => "parts: on a tie, more free rows",
        .move_crosses => "cross: a move to another part is refused without cross",
        .twin_moves => "twins: a twin never moves",
        .twin_free => "twins: borrowed rows are never room",
        .borrow_partial => "twins: all or none",
        .remove_twin_alone => "twins: given back all at once",
        .compact_moves_twin => "compact: a twin stays where it is",
        .compact_crosses_twin => "compact: nothing slides past a twin",
        .compactable_sum => "segments: compactable is the largest segment",
        .gather_held => "gatherable: a held twin is no room",
        .room_after_uncut => "roomAfter: stops at its part's end",
    };
}

fn shardKiller(m: shard.Mutant) []const u8 {
    return switch (m) {
        .none => unreachable,
        .budget_no_trash => "sizes: never past the memory",
        .budget_unaligned => "sizes: whole parts of extents",
        .plan_no_floor => "plan: a part too small",
        .plan_no_shrink => "plan: a part short of the context shrinks it",
        .agree_max => "agree: the smallest",
        .local_global => "address: an owned row in this rank's plane",
        .glue_trash_pos => "glue: a row it does not own writes its own trash row",
        .glue_reads_unowned => "glue: a row it does not own reads no key",
        .glue_no_trash_check => "glue: a window past the trash rows is refused",
        .mark_padding => "fetch: the owner sends the marked positions",
        .renumber_off_by_one => "fetch: renumbered ids are a sort and a search",
        .own_rows_first => "fetch: renumbered ids are a sort and a search",
        .copy_across => "carry: a copy never crosses parts",
        .carry_rows_minus_1 => "carry: the source's owner sends its rows",
        .lists_swapped => "lists: a rank sends its queries of the rows the peer owns",
        .cursor_any_order => "cursor: another order is refused",
        .digest_ignored => "placement: the digest compares the whole layout",
    };
}

fn killed(prefix: []const u8, name: []const u8) !bool {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const t = suite(arena.allocator(), false);
    if (t.killedBy(prefix)) return true;
    std.debug.print("SURVIVED {s}: \"{s}\" passed ({d} checks failed, {d} sections raised)\n", .{ name, prefix, t.failed.items.len, t.crash });
    return false;
}

test "every mutant is killed by its own check (GLM tools/kv_shard_mutants.sh)" {
    defer pool.mutant = .none;
    defer shard.mutant = .none;
    var survived: usize = 0;
    for (std.enums.values(pool.Mutant)) |m| {
        if (m == .none) continue;
        pool.mutant = m;
        if (!try killed(poolKiller(m), @tagName(m))) survived += 1;
    }
    pool.mutant = .none;
    for (std.enums.values(shard.Mutant)) |m| {
        if (m == .none) continue;
        shard.mutant = m;
        if (!try killed(shardKiller(m), @tagName(m))) survived += 1;
    }
    try std.testing.expectEqual(@as(usize, 0), survived);
}
