//! Contiguous int32 SegRows tables from glm5_next/cuda/segments.py: SegRows.fill/set.
const std = @import("std");
const cuda = @import("cuda");
const Stager = @import("cuda_stage.zig").Stager;
const shard = @import("cuda_shard.zig");
pub const Segment = struct { base: usize, pos: usize, rows: usize, ring_base: ?usize = null };
pub fn segCount(streams: usize) !usize {
    if (streams < 1 or streams > 8) return error.BadSegments;
    return @max(4, streams);
}
pub const SegRows = struct {
    gpa: std.mem.Allocator,
    rows: usize,
    max_segs: usize,
    ring: ?usize,
    host: []i32,
    dev: ?cuda.DeviceBuffer = null,
    used: usize = 0,
    /// TF_GLM_KV_SHARD: five more row planes after the segment planes (cuda_shard.Glue), filled with the others
    shard: ?shard.Shard = null,
    pub fn init(gpa: std.mem.Allocator, rows: usize, max_segs: usize, ring: ?usize) !SegRows {
        if (rows == 0 or rows > std.math.maxInt(i32) or max_segs < 1 or max_segs > 8 or (ring != null and ring.? < 4)) return error.BadSegments;
        const host = try gpa.alloc(i32, 5 * rows + 4 * max_segs);
        @memset(host, 0);
        return .{ .gpa = gpa, .rows = rows, .max_segs = max_segs, .ring = ring, .host = host };
    }
    /// A table with TF_GLM_KV_SHARD's glue planes 9..13 (lbase, own, sparse_eff, wpos, orank: cuda_shard.glueOffset)
    /// of `s`'s rank; a window may hold no more rows than the trash rows.
    pub fn initShard(gpa: std.mem.Allocator, rows: usize, max_segs: usize, ring: ?usize, s: shard.Shard) !SegRows {
        if (rows > s.trash) return error.WindowPastTrash;
        if (rows == 0 or rows > std.math.maxInt(i32) or max_segs < 1 or max_segs > 8 or (ring != null and ring.? < 4)) return error.BadSegments;
        const host = try gpa.alloc(i32, 10 * rows + 4 * max_segs);
        @memset(host, 0);
        return .{ .gpa = gpa, .rows = rows, .max_segs = max_segs, .ring = ring, .host = host, .shard = s };
    }
    pub fn deinit(t: *SegRows) void {
        if (t.dev) |*d| d.free();
        t.gpa.free(t.host);
    }
    pub fn initDevice(t: *SegRows, d: *const cuda.Driver) !void {
        if (t.dev != null) return error.AlreadyAllocated;
        t.dev = try cuda.DeviceBuffer.alloc(d, t.host.len * 4);
        errdefer {
            t.dev.?.free();
            t.dev = null;
        }
        try t.dev.?.fill8(0, null);
    }
    /// Row planes 0..4, then segment planes 5..8, then (TF_GLM_KV_SHARD) glue row planes 9..13. Strides retain
    /// capacity, independent of used rows.
    pub fn plane(t: SegRows, i: usize) []i32 {
        if (i >= 9) return t.host[t.offset(i)..][0..t.rows];
        return if (i < 5) t.host[i * t.rows ..][0..t.rows] else t.host[5 * t.rows + (i - 5) * t.max_segs ..][0..t.max_segs];
    }
    pub fn ptr(t: SegRows, i: usize) !u64 {
        const d = t.dev orelse return error.NoDeviceTables;
        if (i >= 9) return d.ptr + 4 * t.offset(i);
        return d.ptr + 4 * (if (i < 5) i * t.rows else 5 * t.rows + (i - 5) * t.max_segs);
    }
    fn offset(t: SegRows, i: usize) usize {
        std.debug.assert(t.shard != null and i < 14);
        return shard.glueOffset(t.rows, t.max_segs, @fromBackingInt(@intCast(i - 9)));
    }
    /// Validate all inputs before changing a table used by a captured graph.
    pub fn fill(t: *SegRows, segments: []const Segment, sparse_from: ?usize) !usize {
        if (segments.len < 1 or segments.len > t.max_segs) return error.BadSegments;
        var total: usize = 0;
        for (segments) |s| {
            if (s.rows == 0 or s.base % 2048 != 0 or s.base > std.math.maxInt(i32) or s.pos > std.math.maxInt(i32) or s.rows > @as(usize, std.math.maxInt(i32)) - s.pos) return error.BadSegment;
            if (s.rows > t.rows - total) return error.TooManyRows;
            if (t.ring) |ring| {
                if (s.ring_base == null or s.ring_base.? > std.math.maxInt(i32) or s.rows + 3 > ring) return error.BadRing;
            }
            // TF_GLM_KV_SHARD: a stream's extent lies in one part of the pool (its glue's owner)
            if (t.shard) |sh| if (!sh.within(s.base, 1)) return error.BadSegment;
            total += s.rows;
        }
        @memset(t.host, 0);
        var r: usize = 0;
        for (segments, 0..) |s, si| {
            for (0..s.rows) |i| {
                const q = s.pos + i;
                t.plane(0)[r + i] = @intCast(q);
                t.plane(1)[r + i] = @intCast(s.base);
                t.plane(2)[r + i] = @intCast(si);
                t.plane(3)[r + i] = @intFromBool(sparse_from != null and q >= sparse_from.?);
                t.plane(4)[r + i] = @intCast(if (t.ring != null) s.ring_base.? else s.base);
            }
            const last = s.pos + s.rows - 1;
            t.plane(5)[si] = @intCast(r);
            t.plane(6)[si] = @intCast(s.rows);
            t.plane(7)[si] = @intCast(s.base / 4);
            t.plane(8)[si] = @intCast(if (sparse_from != null and last >= sparse_from.?) (last + 1) / 4 else 0);
            r += s.rows;
        }
        if (t.shard) |sh| {
            var glue: [5][]i32 = undefined;
            for (&glue, 9..) |*g, i| g.* = t.plane(i);
            // cannot fail: every base is in the pool and the rows are at most the trash rows (checked above, initShard)
            shard.glue(sh, total, t.plane(0), t.plane(1), t.plane(3), &glue) catch unreachable;
        }
        t.used = total;
        return total;
    }
    /// Synchronous upload owns no pinned staging lifetime; P6 may add the reference's event-guarded async path.
    pub fn stage(t: SegRows) !void {
        const d = t.dev orelse return error.NoDeviceTables;
        try d.upload(0, std.mem.sliceAsBytes(t.host));
    }
    /// stage() as a stream-ordered copy from pinned memory: nothing waits, nothing drains.
    pub fn stageAsync(t: SegRows, st: *Stager) !void {
        const d = t.dev orelse return error.NoDeviceTables;
        try st.put(d.ptr, std.mem.sliceAsBytes(t.host));
    }
    pub fn anySparse(t: SegRows) bool {
        for (t.plane(3)[0..t.used]) |x| {
            if (x != 0) return true;
        }
        return false;
    }
};
test "SegRows dense boundary, ring bases and unused segments match Python" {
    var t = try SegRows.init(std.testing.allocator, 8, 4, 64);
    defer t.deinit();
    try std.testing.expectEqual(@as(usize, 5), try t.fill(&.{ .{ .base = 0, .pos = 2050, .rows = 3, .ring_base = 0 }, .{ .base = 4096, .pos = 7, .rows = 2, .ring_base = 64 } }, 2051));
    try std.testing.expectEqualSlices(i32, &.{ 2050, 2051, 2052, 7, 8, 0, 0, 0 }, t.plane(0));
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 1, 0, 0, 0, 0, 0 }, t.plane(3));
    try std.testing.expectEqualSlices(i32, &.{ 513, 0, 0, 0 }, t.plane(8));
    try std.testing.expectEqualSlices(i32, &.{ 0, 1024, 0, 0 }, t.plane(7));
    try std.testing.expect(t.anySparse());
    _ = try t.fill(&.{.{ .base = 2048, .pos = 1, .rows = 1, .ring_base = 64 }}, null);
    try std.testing.expect(!t.anySparse());
    try std.testing.expectEqual(@as(i32, 0), t.plane(6)[1]);
    try std.testing.expectError(error.BadSegment, t.fill(&.{.{ .base = 1, .pos = 0, .rows = 1 }}, 2051));
}

test "TF_GLM_KV_SHARD tables: the glue planes after the segment planes, filled with the rows, refused before a change" {
    const gpa = std.testing.allocator;
    const sh = try shard.Shard.init(1, 2, 8 * 2048, shard.trash);
    try std.testing.expectError(error.WindowPastTrash, SegRows.initShard(gpa, 65, 4, 64, sh));
    var t = try SegRows.initShard(gpa, 8, 4, 64, sh);
    defer t.deinit();
    try std.testing.expectEqual(@as(usize, 10 * 8 + 4 * 4), t.host.len);
    const segs = [_]Segment{ .{ .base = 0, .pos = 2050, .rows = 3, .ring_base = 0 }, .{ .base = 8192 + 2048, .pos = 7, .rows = 2, .ring_base = 64 } };
    try std.testing.expectEqual(@as(usize, 5), try t.fill(&segs, 2051));
    // the planes 0..8 as an unsharded table's
    var plain = try SegRows.init(gpa, 8, 4, 64);
    defer plain.deinit();
    _ = try plain.fill(&segs, 2051);
    for (0..9) |i| try std.testing.expectEqualSlices(i32, plain.plane(i), t.plane(i));
    // rank 1 owns the second stream (part 1): its rows read their keys here, at part-local rows; the first stream's
    // rows write their trash rows and read nothing
    try std.testing.expectEqualSlices(i32, &.{ 8192, 8192, 8192, 2048, 2048, 0, 0, 0 }, t.plane(9 + @backingInt(shard.Glue.lbase)));
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 0, 1, 1, 0, 0, 0 }, t.plane(9 + @backingInt(shard.Glue.own)));
    try std.testing.expectEqualSlices(i32, &.{ 1, 1, 1, 0, 0, 0, 0, 0 }, t.plane(9 + @backingInt(shard.Glue.sparse_eff)));
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 7, 8, 0, 0, 0 }, t.plane(9 + @backingInt(shard.Glue.wpos)));
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 0, 1, 1, 0, 0, 0 }, t.plane(9 + @backingInt(shard.Glue.orank)));
    // the glue planes where cuda_shard.glueOffset puts them
    for (0..5) |g| try std.testing.expectEqual(@as(usize, shard.glueOffset(8, 4, @fromBackingInt(@intCast(g)))), (@intFromPtr(t.plane(9 + g).ptr) - @intFromPtr(t.host.ptr)) / 4);
    // a base outside the pool is refused before the table changes
    const before = try gpa.dupe(i32, t.host);
    defer gpa.free(before);
    try std.testing.expectError(error.BadSegment, t.fill(&.{.{ .base = 8 * 2048, .pos = 0, .rows = 1, .ring_base = 0 }}, 2051));
    try std.testing.expectEqualSlices(i32, before, t.host);
    try std.testing.expectEqual(@as(usize, 5), t.used);
}
