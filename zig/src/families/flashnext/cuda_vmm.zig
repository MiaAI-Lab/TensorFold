//! CUDA virtual memory for Flash Next's caches (work/research/R3-concurrency-kv.md's KV items): a sequence reserves
//! address space for its whole window once and maps physical memory into it as it grows, so a cache never moves (no
//! copy, no old-plus-new peak, pointers that captured graphs keep), and its indexer keys live in a ring: one
//! physical chunk mapped again and again over the window's rows, since only the rows from the last incomplete
//! block on are ever read (attention.py `_pool_block` reads rows [p0 - 3, p0 + R) of the keys attn_prep wrote).
//! The driver's VMM entry points are looked up from libcuda here (the shared driver binding does not carry them).
const std = @import("std");
const cuda = @import("cuda");

const Allocator = std.mem.Allocator;
const Ptr = u64;
const Result = c_int;

/// CUmemLocation, CUmemAllocationProp and CUmemAccessDesc (cuda.h).
const Location = extern struct { type: c_int = 1, id: c_int = 0 }; // CU_MEM_LOCATION_TYPE_DEVICE
const AllocFlags = extern struct { compression: u8 = 0, rdma: u8 = 0, usage: u16 = 0, reserved: [4]u8 = @splat(0) };
const Prop = extern struct {
    type: c_int = 1, // CU_MEM_ALLOCATION_TYPE_PINNED
    handle_types: c_int = 0,
    location: Location = .{},
    win32: ?*anyopaque = null,
    flags: AllocFlags = .{},
};
const Access = extern struct { location: Location = .{}, flags: c_int = 3 }; // CU_MEM_ACCESS_FLAGS_PROT_READWRITE

comptime {
    std.debug.assert(@sizeOf(Prop) == 32 and @sizeOf(Access) == 12);
}

pub const Vmm = struct {
    lib: std.DynLib,
    device: c_int,
    granularity: usize,
    granularityFn: *const fn (*usize, *const Prop, c_int) callconv(.c) Result,
    create: *const fn (*u64, usize, *const Prop, u64) callconv(.c) Result,
    release: *const fn (u64) callconv(.c) Result,
    reserveFn: *const fn (*Ptr, usize, usize, Ptr, u64) callconv(.c) Result,
    addressFree: *const fn (Ptr, usize) callconv(.c) Result,
    map: *const fn (Ptr, usize, usize, u64, u64) callconv(.c) Result,
    unmap: *const fn (Ptr, usize) callconv(.c) Result,
    setAccess: *const fn (Ptr, usize, *const Access, usize) callconv(.c) Result,

    /// The VMM entry points of the driver that `ctx` runs on, and its allocation granularity (minimum).
    pub fn init(ctx: *const cuda.Context) !Vmm {
        var lib = std.DynLib.open("libcuda.so.1") catch return error.DriverUnavailable;
        errdefer lib.close();
        var v: Vmm = undefined;
        v.lib = lib;
        v.device = @intCast(ctx.device);
        inline for (.{ .{ "granularityFn", "cuMemGetAllocationGranularity" }, .{ "create", "cuMemCreate" }, .{ "release", "cuMemRelease" }, .{ "reserveFn", "cuMemAddressReserve" }, .{ "addressFree", "cuMemAddressFree" }, .{ "map", "cuMemMap" }, .{ "unmap", "cuMemUnmap" }, .{ "setAccess", "cuMemSetAccess" } }) |e| {
            @field(v, e[0]) = v.lib.lookup(@TypeOf(@field(v, e[0])), e[1]) orelse return error.MissingSymbol;
        }
        const want: Prop = .{ .location = .{ .id = v.device } };
        try check(v.granularityFn(&v.granularity, &want, 0), "cuMemGetAllocationGranularity");
        if (v.granularity == 0) return error.NoGranularity;
        return v;
    }

    pub fn deinit(v: *Vmm) void {
        v.lib.close();
    }

    fn props(v: *const Vmm) Prop {
        return .{ .location = .{ .id = v.device } };
    }

    pub fn up(v: *const Vmm, bytes: usize) usize {
        return std.mem.alignForward(usize, bytes, v.granularity);
    }
};

fn check(rc: Result, what: []const u8) !void {
    if (rc != 0) {
        std.log.err("{s}: CUDA error {d}", .{ what, rc });
        return error.CudaVmm;
    }
}

/// Address space for a cache that grows to `reserved` bytes; physical chunks mapped at its end as it grows.
pub const Region = struct {
    v: *const Vmm,
    base: Ptr,
    reserved: usize,
    mapped: usize = 0,
    handles: std.ArrayList(u64) = .empty,
    /// every mapping (offset, size), unmapped one by one
    maps: std.ArrayList([2]usize) = .empty,
    /// a ring's one chunk mapped over every granule of the region
    ring: bool = false,

    pub fn reserve(v: *const Vmm, bytes: usize) !Region {
        const size = v.up(@max(bytes, 1));
        var base: Ptr = 0;
        try check(v.reserveFn(&base, size, v.granularity, 0, 0), "cuMemAddressReserve");
        return .{ .v = v, .base = base, .reserved = size };
    }

    /// Physical memory under the first `bytes` (rounded up to the granularity); what it added.
    pub fn growTo(r: *Region, gpa: Allocator, bytes: usize) !usize {
        std.debug.assert(!r.ring);
        const want = r.v.up(bytes);
        if (want <= r.mapped) return 0;
        if (want > r.reserved) return error.RegionFull;
        const size = want - r.mapped;
        var h: u64 = 0;
        const p = r.v.props();
        try check(r.v.create(&h, size, &p, 0), "cuMemCreate");
        errdefer _ = r.v.release(h);
        try check(r.v.map(r.base + r.mapped, size, 0, h, 0), "cuMemMap");
        errdefer _ = r.v.unmap(r.base + r.mapped, size);
        const a: Access = .{ .location = .{ .id = r.v.device } };
        try check(r.v.setAccess(r.base + r.mapped, size, &a, 1), "cuMemSetAccess");
        try r.maps.append(gpa, .{ r.mapped, size });
        try r.handles.append(gpa, h);
        r.mapped = want;
        return size;
    }

    /// The whole region as a ring over one chunk of `chunk` bytes (a multiple of the granularity): address
    /// base + x reaches physical byte x mod chunk. Returns the physical bytes.
    pub fn mapRing(r: *Region, gpa: Allocator, chunk: usize) !usize {
        const size = r.v.up(chunk);
        if (r.reserved % size != 0) return error.RingDoesNotTile;
        var h: u64 = 0;
        const p = r.v.props();
        try check(r.v.create(&h, size, &p, 0), "cuMemCreate");
        errdefer _ = r.v.release(h);
        try r.handles.append(gpa, h);
        var at: usize = 0;
        while (at < r.reserved) : (at += size) {
            try check(r.v.map(r.base + at, size, 0, h, 0), "cuMemMap");
            try r.maps.append(gpa, .{ at, size });
        }
        const a: Access = .{ .location = .{ .id = r.v.device } };
        try check(r.v.setAccess(r.base, r.reserved, &a, 1), "cuMemSetAccess");
        r.mapped = r.reserved;
        r.ring = true;
        return size;
    }

    pub fn deinit(r: *Region, gpa: Allocator) void {
        for (r.maps.items) |m| _ = r.v.unmap(r.base + m[0], m[1]);
        for (r.handles.items) |h| _ = r.v.release(h);
        r.maps.deinit(gpa);
        r.handles.deinit(gpa);
        _ = r.v.addressFree(r.base, r.reserved);
        r.* = undefined;
    }
};

test "the VMM structs match cuda.h's layout" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(Prop));
    try std.testing.expectEqual(@as(usize, 8), @offsetOf(Prop, "location"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(Prop, "win32"));
    try std.testing.expectEqual(@as(usize, 24), @offsetOf(Prop, "flags"));
    try std.testing.expectEqual(@as(usize, 12), @sizeOf(Access));
}
