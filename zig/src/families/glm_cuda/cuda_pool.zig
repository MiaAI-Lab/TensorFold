//! Host extent placement from glm5_next/cuda/pool.py: Pool (device copies remain the caller's job).
//! TF_GLM_KV_SHARD (patch 0118): parts, one a rank, that no extent crosses, and twins: free rows of the other parts a
//! filling prompt borrows (pinned: never moved, never room, given back all at once). One part is the pool as before.
const std = @import("std");
const builtin = @import("builtin");
pub const alignment = 2048;
/// The most parts a pool is cut in.
pub const max_parts = 8;
pub const Extent = struct {
    base: usize,
    size: usize,
    eid: u64,
    owner: ?u64 = null,
    kept: usize = 0,
    /// a twin: the extent whose filling prompt borrows these rows in another part
    twin: ?u64 = null,
    /// a twin of a fill part way through its layers (its later layers read these rows): not given back yet
    held: bool = false,
    pub fn end(x: Extent) usize {
        return x.base + x.size;
    }
    pub fn free(x: Extent) bool {
        return x.owner == null and x.kept == 0 and x.twin == null;
    }
};
/// Pool rows [lo, hi).
pub const Range = struct {
    lo: usize,
    hi: usize,
    pub fn len(r: Range) usize {
        return r.hi - r.lo;
    }
    pub fn has(r: Range, row: usize) bool {
        return r.lo <= row and row < r.hi;
    }
};
/// Which twins cut a part in segments: every twin (none moves), or the held ones only (the others can be given back).
pub const Pins = enum { twins, held };
/// A compaction step: extent `eid` to `base` (the caller copies its rows, then moves it).
pub const Slide = struct { eid: u64, base: usize };
fn pinned(x: Extent, pins: Pins) bool {
    return x.twin != null and (pins == .twins or (x.held and !mut(.gather_held)));
}
/// The ranges of a part between its pinned twins, by base: compaction gathers free rows within one, never past a twin.
pub const Segments = struct {
    items: []const Extent,
    at: usize,
    hi: usize,
    pins: Pins,
    i: usize = 0,
    pub fn next(s: *Segments) ?Range {
        while (s.i < s.items.len) {
            const x = s.items[s.i];
            s.i += 1;
            if (x.base >= s.hi) {
                s.i = s.items.len;
                break;
            }
            if (x.base < s.at or !pinned(x, s.pins)) continue;
            const r: Range = .{ .lo = s.at, .hi = x.base };
            s.at = x.end();
            if (r.hi > r.lo) return r;
        }
        if (s.at >= s.hi) return null;
        defer s.at = s.hi;
        return .{ .lo = s.at, .hi = s.hi };
    }
};
pub const Pool = struct {
    gpa: std.mem.Allocator,
    rows: usize,
    extents: std.ArrayList(Extent) = .empty,
    next_id: u64 = 0,
    /// TF_GLM_KV_SHARD: rows a part (0: one pool)
    part: usize = 0,
    pub fn init(gpa: std.mem.Allocator, rows: usize) !Pool {
        if (rows == 0 or rows % alignment != 0) return error.BadPool;
        return .{ .gpa = gpa, .rows = rows };
    }
    /// `n` equal parts of whole extents, one a rank (1: init's one pool).
    pub fn initParts(gpa: std.mem.Allocator, rows: usize, n: usize) !Pool {
        if (n == 0 or n > max_parts or rows % (n * alignment) != 0) return error.BadPool;
        var p = try init(gpa, rows);
        if (n > 1) p.part = rows / n;
        return p;
    }
    pub fn deinit(p: *Pool) void {
        p.extents.deinit(p.gpa);
    }
    fn sizeOf(size: usize) !usize {
        if (size == 0 or size > std.math.maxInt(usize) - (alignment - 1)) return error.BadExtent;
        return std.mem.alignForward(usize, size, alignment);
    }
    pub fn get(p: *Pool, eid: u64) !*Extent {
        for (p.extents.items) |*x| {
            if (x.eid == eid) return x;
        }
        return error.NoExtent;
    }
    /// A copy of extent `eid`.
    pub fn extent(p: Pool, eid: u64) !Extent {
        for (p.extents.items) |x| {
            if (x.eid == eid) return x;
        }
        return error.NoExtent;
    }
    pub fn freeRows(p: Pool) usize {
        var n = p.rows;
        for (p.extents.items) |x| n -= x.size;
        return n;
    }
    pub fn parts(p: Pool) usize {
        return if (p.part == 0) 1 else p.rows / p.part;
    }
    /// The part pool row `row` lies in.
    pub fn partOf(p: Pool, row: usize) usize {
        return if (p.part == 0) 0 else row / p.part;
    }
    pub fn partRange(p: Pool, i: usize) Range {
        return if (p.part == 0) .{ .lo = 0, .hi = p.rows } else .{ .lo = i * p.part, .hi = (i + 1) * p.part };
    }
    /// Extents a live stream writes in (an owner) in part `i`.
    pub fn live(p: Pool, i: usize) usize {
        var n: usize = 0;
        for (p.extents.items) |x| n += @intFromBool(x.owner != null and p.partOf(x.base) == i);
        return n;
    }
    /// Rows of part `i` live streams' extents (an owner) and twins hold.
    pub fn rowsIn(p: Pool, i: usize) struct { live: usize, twins: usize } {
        var live_rows: usize = 0;
        var twins: usize = 0;
        for (p.extents.items) |x| {
            if (p.partOf(x.base) != i) continue;
            if (x.twin != null) twins += x.size else if (x.owner != null) live_rows += x.size;
        }
        return .{ .live = live_rows, .twins = twins };
    }
    /// A copy of the extent (not a twin) based at pool row `base` (null: none): an extent as every rank names it.
    pub fn extentAt(p: Pool, base: usize) ?Extent {
        for (p.extents.items) |x| {
            if (x.base == base and x.twin == null) return x;
        }
        return null;
    }
    /// The layout every rank's own placements must reproduce (TF_GLM_KV_SHARD_CHECK): the pool's rows and each
    /// extent's base, rows, owner and (a twin) its extent's base, in base order. No ids: each pool counts its own, and
    /// a call one rank made and undid leaves the counts apart. Never 0.
    pub fn digest(p: Pool) u64 {
        const none = std.math.maxInt(u64);
        var h = std.hash.Wyhash.init(0);
        h.update(std.mem.asBytes(&@as(u64, p.rows)));
        h.update(std.mem.asBytes(&@as(u64, p.part)));
        for (p.extents.items) |x| {
            const of: u64 = if (x.twin) |t| (if (p.extent(t)) |y| y.base else |_| none - 1) else none;
            const w = [4]u64{ x.base, x.size, x.owner orelse none, of };
            h.update(std.mem.asBytes(&w));
        }
        return @max(1, h.final());
    }
    /// Free rows of part `i` (a twin's rows are taken).
    pub fn freeIn(p: Pool, i: usize) usize {
        var n = p.partRange(i).len();
        for (p.extents.items) |x| {
            if (p.partOf(x.base) == i and !(x.twin != null and mut(.twin_free))) n -= x.size;
        }
        return n;
    }
    /// Ignore the moving extent when searching for a relocation; lowest-base first fit.
    pub fn place(p: Pool, requested: usize, ignore: ?u64) !?usize {
        return p.placeIn(requested, ignore, null);
    }
    /// `place` in part `part`, or (null) in the part with fewer live streams first (each row's owner computes its
    /// decode attention, so the ranks' shares stay even), then the one with more free rows; a part's end cuts its gaps.
    pub fn placeIn(p: Pool, requested: usize, ignore: ?u64, part: ?usize) !?usize {
        const size = try sizeOf(requested);
        if (part) |i| {
            if (i >= p.parts()) return error.BadPart;
            return p.fit(size, ignore, p.partRange(i));
        }
        var order: [max_parts]usize = undefined;
        const n = p.parts();
        for (order[0..n], 0..) |*o, i| o.* = i;
        std.mem.sort(usize, order[0..n], &p, lighter);
        for (order[0..n]) |i| {
            if (p.fit(size, ignore, p.partRange(i))) |base| return base;
        }
        return null;
    }
    fn lighter(p: *const Pool, a: usize, b: usize) bool {
        const la = p.live(a);
        const lb = p.live(b);
        if (la != lb and !mut(.order_ignores_live)) return la < lb;
        const fa = p.freeIn(a);
        const fb = p.freeIn(b);
        if (fa != fb and !mut(.order_ignores_free)) return fa > fb;
        return a < b;
    }
    fn fit(p: Pool, size: usize, ignore: ?u64, r: Range) ?usize {
        const hi = if (mut(.gaps_uncut)) p.rows else r.hi;
        var at = r.lo;
        for (p.extents.items) |x| {
            if (ignore != null and x.eid == ignore.?) continue;
            if (x.end() <= at) continue;
            if (x.base >= hi) break;
            if (x.base > at and x.base - at >= size) return at;
            at = x.end();
        }
        return if (at <= hi and hi - at >= size) at else null;
    }
    fn validate(p: Pool, base: usize, size: usize, ignore: ?u64) !void {
        if (base % alignment != 0 or base > p.rows or size > p.rows - base) return error.BadExtent;
        if (p.part != 0 and p.partOf(base) != p.partOf(base + size - 1) and !mut(.cross_part_add)) return error.CrossesPart;
        for (p.extents.items) |x| {
            if (ignore != null and x.eid == ignore.?) continue;
            if (x.base < base + size and base < x.end()) return error.Overlap;
        }
    }
    fn sort(p: *Pool) void {
        std.mem.sort(Extent, p.extents.items, {}, struct {
            fn less(_: void, a: Extent, b: Extent) bool {
                return a.base < b.base;
            }
        }.less);
    }
    pub fn add(p: *Pool, base: usize, requested: usize) !u64 {
        return p.insert(base, requested, null);
    }
    /// A twin of extent `of` at `base` in another part, its size (`borrow` places them).
    pub fn addTwin(p: *Pool, base: usize, of: u64) !u64 {
        const x = try p.extent(of);
        if (x.twin != null) return error.Twin;
        if (p.part == 0 or p.partOf(base) == p.partOf(x.base)) return error.SamePart;
        return p.insert(base, x.size, of);
    }
    fn insert(p: *Pool, base: usize, requested: usize, twin: ?u64) !u64 {
        const size = try sizeOf(requested);
        try p.validate(base, size, null);
        const eid = p.next_id;
        try p.extents.append(p.gpa, .{ .base = base, .size = size, .eid = eid, .twin = twin });
        p.next_id += 1;
        p.sort();
        return eid;
    }
    pub fn alloc(p: *Pool, requested: usize) !?u64 {
        return try p.add(try p.place(requested, null) orelse return null, requested);
    }
    pub fn resize(p: *Pool, eid: u64, requested: usize) !void {
        const size = try sizeOf(requested);
        const x = try p.get(eid);
        if (x.twin != null) return error.Pinned;
        try p.validate(x.base, size, eid);
        x.size = size;
    }
    /// An extent stays in its part (its rank holds its latents) unless `cross`: the caller then carries its rows to the
    /// other part's rank. A twin never moves; an extent with twins never changes part.
    pub fn move(p: *Pool, eid: u64, base: usize, requested: ?usize, cross: bool) !usize {
        const x = try p.get(eid);
        if (x.twin != null and !mut(.twin_moves)) return error.Pinned;
        const size = if (requested) |n| try sizeOf(n) else x.size;
        try p.validate(base, size, eid);
        if (p.partOf(base) != p.partOf(x.base)) {
            if (!cross and !mut(.move_crosses)) return error.OtherPart;
            if (p.hasTwins(eid)) return error.Twinned;
        }
        const old = x.base;
        x.base = base;
        x.size = size;
        p.sort();
        return old;
    }
    /// Twins go back through giveBack, all at once, and an extent's twins before it.
    pub fn remove(p: *Pool, eid: u64) !void {
        for (p.extents.items, 0..) |x, i| {
            if (x.eid == eid) {
                if (x.twin != null and !mut(.remove_twin_alone)) return error.Twin;
                if (p.part != 0 and p.hasTwins(eid)) return error.Twinned;
                _ = p.extents.orderedRemove(i);
                return;
            }
        }
        return error.NoExtent;
    }

    // -- twins (TF_GLM_KV_SHARD): a filling prompt's rows in the other parts --------------------------------------------

    /// Twins of extent `eid` in every other part, all or none, in free rows only (placed alike on every rank): false
    /// when a part has no room.
    pub fn borrow(p: *Pool, eid: u64) !bool {
        if (p.part == 0) return error.NoParts;
        const x = try p.extent(eid);
        if (x.twin != null) return error.Twin;
        if (p.hasTwins(eid)) return error.Twinned;
        var made: [max_parts]u64 = undefined;
        var n: usize = 0;
        errdefer for (made[0..n]) |t| p.drop(t);
        for (0..p.parts()) |q| {
            if (q == p.partOf(x.base)) continue;
            const base = try p.placeIn(x.size, null, q) orelse {
                if (!mut(.borrow_partial)) for (made[0..n]) |t| p.drop(t);
                return false;
            };
            made[n] = try p.addTwin(base, eid);
            n += 1;
        }
        return true;
    }
    pub fn hasTwins(p: Pool, eid: u64) bool {
        for (p.extents.items) |x| {
            if (x.twin == eid) return true;
        }
        return false;
    }
    /// The base of extent `eid`'s twin in part `part` (null: none there).
    pub fn twinIn(p: Pool, eid: u64, part: usize) ?usize {
        for (p.extents.items) |x| {
            if (x.twin == eid and p.partOf(x.base) == part) return x.base;
        }
        return null;
    }
    /// Pin extent `eid`'s twins while its fill is part way through its layers, or (false) let them be given back.
    pub fn hold(p: *Pool, eid: u64, held: bool) void {
        for (p.extents.items) |*x| {
            if (x.twin == eid) x.held = held;
        }
    }
    /// Whether a twin of extent `eid` is pinned (`hold`): its fill is part way through its layers.
    pub fn holding(p: Pool, eid: u64) bool {
        for (p.extents.items) |x| {
            if (x.twin == eid and x.held) return true;
        }
        return false;
    }
    /// Every twin of extent `eid` back to the pool (its prompt fills on with the owner's rows alone).
    pub fn giveBack(p: *Pool, eid: u64) !void {
        var n: usize = 0;
        var i: usize = 0;
        while (i < p.extents.items.len) {
            if (p.extents.items[i].twin == eid) {
                _ = p.extents.orderedRemove(i);
                n += 1;
            } else i += 1;
        }
        if (n == 0) return error.NoTwin;
    }
    fn drop(p: *Pool, eid: u64) void {
        for (p.extents.items, 0..) |x, i| {
            if (x.eid == eid) {
                _ = p.extents.orderedRemove(i);
                return;
            }
        }
    }

    // -- segments and compaction around pinned twins -------------------------------------------------------------------

    /// Part `part`'s segments (null: the pool's).
    pub fn segments(p: *const Pool, part: ?usize, pins: Pins) Segments {
        const r = if (part) |i| p.partRange(i) else Range{ .lo = 0, .hi = p.rows };
        return .{ .items = p.extents.items, .at = r.lo, .hi = r.hi, .pins = pins };
    }
    /// Free rows of `r`: its rows less those of the extents based in it.
    pub fn segmentFree(p: Pool, r: Range) usize {
        var n = r.len();
        for (p.extents.items) |x| {
            if (r.has(x.base)) n -= x.size;
        }
        return n;
    }
    /// The segment holding pool row `row` (null: a pinned twin's row).
    pub fn segmentAt(p: *const Pool, part: ?usize, pins: Pins, row: usize) ?Range {
        var it = p.segments(part, pins);
        while (it.next()) |r| {
            if (r.has(row)) return r;
        }
        return null;
    }
    /// The longest range of free rows compacting part `part` can gather: the most of any one segment's.
    pub fn compactable(p: *const Pool, part: ?usize, pins: Pins) usize {
        var best: usize = 0;
        var it = p.segments(part, pins);
        while (it.next()) |r| {
            if (mut(.compactable_sum)) best += p.segmentFree(r) else best = @max(best, p.segmentFree(r));
        }
        return best;
    }
    /// What a request can get in part `part`: a segment's free rows and the twins in it, which can be given back.
    pub fn gatherable(p: *const Pool, part: ?usize, pins: Pins) usize {
        var best: usize = 0;
        var it = p.segments(part, pins);
        while (it.next()) |r| {
            var n = p.segmentFree(r);
            for (p.extents.items) |x| {
                if (x.twin != null and r.has(x.base)) n += x.size;
            }
            best = @max(best, n);
        }
        return best;
    }
    /// Free rows of the segment extent `eid` lies in: what it can grow by once compaction gathers them after it.
    pub fn segmentRoom(p: *const Pool, eid: u64) !usize {
        const x = try p.extent(eid);
        const r = p.segmentAt(if (p.part == 0) null else p.partOf(x.base), .twins, x.base) orelse return 0;
        return p.segmentFree(r);
    }
    /// Free rows right after extent `eid`, before the next extent or its part's end.
    pub fn roomAfter(p: Pool, eid: u64) !usize {
        const x = try p.extent(eid);
        var next = if (mut(.room_after_uncut)) p.rows else p.partRange(p.partOf(x.base)).hi;
        for (p.extents.items) |y| {
            if (y.eid != eid and y.base >= x.end()) next = @min(next, y.base);
        }
        return next - x.end();
    }
    /// The next step of sliding part `part`'s extents down to its lowest free rows, around its twins (they stay); with
    /// `grow`, those up to it slide down and the ones after it up to the part's end, so its free rows follow it. Each
    /// extent moves once at most; null once the part is compact.
    pub fn nextSlide(p: Pool, part: usize, grow: ?u64) !?Slide {
        if (part >= p.parts()) return error.BadPart;
        var r = p.partRange(part);
        var gb: usize = 0;
        if (grow) |g| {
            gb = (try p.extent(g)).base;
            if (p.partOf(gb) != part) return error.OtherPart;
        }
        var at = r.lo;
        for (p.extents.items) |x| {
            if (!r.has(x.base)) continue;
            if (grow != null and x.base > gb) break;
            if (x.twin != null and !mut(.compact_moves_twin)) {
                if (!mut(.compact_crosses_twin)) at = x.end();
                continue;
            }
            if (x.base != at) return .{ .eid = x.eid, .base = at };
            at += x.size;
        }
        const g = grow orelse return null;
        var i = p.extents.items.len;
        while (i > 0) {
            i -= 1;
            const x = p.extents.items[i];
            if (!r.has(x.base)) continue;
            if (x.eid == g) break;
            if (x.twin != null and !mut(.compact_moves_twin)) {
                if (!mut(.compact_crosses_twin)) r.hi = x.base;
                continue;
            }
            r.hi -= x.size;
            if (x.base != r.hi) return .{ .eid = x.eid, .base = r.hi };
        }
        return null;
    }
};

// -- mutants: each guard above switched off, in test builds only (GLM tools/kv_shard_mutants.sh) -----------------------

pub const Mutant = enum {
    none,
    cross_part_add, // an extent may cross a part's end
    gaps_uncut, // a part's free rows run on into the next part
    order_ignores_live, // a new stream goes to the part with more free rows, however many streams it holds
    order_ignores_free, // ...and on a tie to the lowest part, not the one with more free rows
    move_crosses, // a move changes part without `cross`
    twin_moves, // a twin moves
    twin_free, // a part's free rows count its twins'
    borrow_partial, // a borrow that fails keeps the twins it placed
    remove_twin_alone, // one twin goes back alone
    compact_moves_twin, // compaction slides twins
    compact_crosses_twin, // compaction slides extents past twins
    compactable_sum, // a part's free rows count as gathered however many twins cut them
    gather_held, // a held twin's rows count as room
    room_after_uncut, // the room after an extent runs past its part's end
};
pub var mutant: Mutant = .none;
inline fn mut(m: Mutant) bool {
    return builtin.is_test and mutant == m;
}

test "deterministic first fit, growth, relocation and removal" {
    var p = try Pool.init(std.testing.allocator, 6 * alignment);
    defer p.deinit();
    const a = (try p.alloc(1)).?;
    const b = (try p.alloc(2049)).?;
    try std.testing.expectError(error.Overlap, p.resize(a, 4096));
    try std.testing.expectEqual(@as(usize, 2048), try p.move(b, 8192, null, false));
    try p.resize(a, 4096);
    try std.testing.expectEqual(@as(?usize, 4096), try p.place(2048, null));
    try p.remove(a);
    try std.testing.expectEqual(@as(?usize, 0), try p.place(4096, null));
    try std.testing.expectEqual(@as(usize, 8192), p.freeRows());
    try std.testing.expectEqual(@as(?usize, null), try p.place(16384, null));
}
