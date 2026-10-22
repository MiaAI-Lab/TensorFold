//! Host extent placement from glm5_next/cuda/pool.py: Pool (device copies remain the caller's job).
const std = @import("std");
pub const alignment = 2048;
pub const Extent = struct {
    base: usize,
    size: usize,
    eid: u64,
    owner: ?u64 = null,
    kept: usize = 0,
    pub fn end(x: Extent) usize {
        return x.base + x.size;
    }
    pub fn free(x: Extent) bool {
        return x.owner == null and x.kept == 0;
    }
};
pub const Pool = struct {
    gpa: std.mem.Allocator,
    rows: usize,
    extents: std.ArrayList(Extent) = .empty,
    next_id: u64 = 0,
    pub fn init(gpa: std.mem.Allocator, rows: usize) !Pool {
        if (rows == 0 or rows % alignment != 0) return error.BadPool;
        return .{ .gpa = gpa, .rows = rows };
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
    pub fn freeRows(p: Pool) usize {
        var n = p.rows;
        for (p.extents.items) |x| n -= x.size;
        return n;
    }
    /// Ignore the moving extent when searching for a relocation; lowest-base first fit.
    pub fn place(p: Pool, requested: usize, ignore: ?u64) !?usize {
        const size = try sizeOf(requested);
        var at: usize = 0;
        for (p.extents.items) |x| {
            if (ignore != null and x.eid == ignore.?) continue;
            if (x.base - at >= size) return at;
            at = x.end();
        }
        return if (p.rows - at >= size) at else null;
    }
    fn validate(p: Pool, base: usize, size: usize, ignore: ?u64) !void {
        if (base % alignment != 0 or base > p.rows or size > p.rows - base) return error.BadExtent;
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
        const size = try sizeOf(requested);
        try p.validate(base, size, null);
        const eid = p.next_id;
        try p.extents.append(p.gpa, .{ .base = base, .size = size, .eid = eid });
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
        try p.validate(x.base, size, eid);
        x.size = size;
    }
    pub fn move(p: *Pool, eid: u64, base: usize, requested: ?usize) !usize {
        const x = try p.get(eid);
        const size = if (requested) |n| try sizeOf(n) else x.size;
        try p.validate(base, size, eid);
        const old = x.base;
        x.base = base;
        x.size = size;
        p.sort();
        return old;
    }
    pub fn remove(p: *Pool, eid: u64) !void {
        for (p.extents.items, 0..) |x, i| {
            if (x.eid == eid) {
                _ = p.extents.orderedRemove(i);
                return;
            }
        }
        return error.NoExtent;
    }
};
test "deterministic first fit, growth, relocation and removal" {
    var p = try Pool.init(std.testing.allocator, 6 * alignment);
    defer p.deinit();
    const a = (try p.alloc(1)).?;
    const b = (try p.alloc(2049)).?;
    try std.testing.expectError(error.Overlap, p.resize(a, 4096));
    try std.testing.expectEqual(@as(usize, 2048), try p.move(b, 8192, null));
    try p.resize(a, 4096);
    try std.testing.expectEqual(@as(?usize, 4096), try p.place(2048, null));
    try p.remove(a);
    try std.testing.expectEqual(@as(?usize, 0), try p.place(4096, null));
    try std.testing.expectEqual(@as(usize, 8192), p.freeRows());
    try std.testing.expectEqual(@as(?usize, null), try p.place(16384, null));
}
