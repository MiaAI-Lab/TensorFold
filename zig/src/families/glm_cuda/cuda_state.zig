//! Device layouts from glm5_next/cuda/forward.py: Slots, Caches, State; kda.py: KDAScratchSet.
//! FP8 cache rows follow glm5_next/cuda/kv8.py. No arithmetic or quantization is performed here.
const std = @import("std");
const cuda = @import("cuda");
pub const Geometry = @import("cuda_geometry.zig").Geometry;
const G = Geometry;
const display = @import("cuda_display_kv.zig");
pub fn aligned(sizes: []const usize) usize {
    var n: usize = 0;
    for (sizes) |size| n += std.mem.alignForward(usize, size, 256);
    return n;
}
pub const Arena = struct {
    buf: cuda.DeviceBuffer,
    offset: usize = 0,
    pub fn init(d: *const cuda.Driver, sizes: []const usize) !Arena {
        var a: Arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, aligned(sizes)) };
        errdefer a.buf.free();
        try a.buf.fill8(0, null);
        return a;
    }
    pub fn take(a: *Arena, n: usize) u64 {
        if (n == 0) return 0;
        const at = a.offset;
        a.offset += std.mem.alignForward(usize, n, 256);
        std.debug.assert(a.offset <= a.buf.len);
        return a.buf.ptr + at;
    }
    pub fn deinit(a: *Arena) void {
        a.buf.free();
    }
};
pub const Slot = struct { rec: u64, conv: u64, proj: u64, out: u64, k: u64, v: u64, g: u64, b: u64, pos_dev: u64, mtp_pos_dev: u64 };
pub const Slots = struct {
    arena: Arena,
    g: Geometry,
    count: usize,
    rows: usize,
    rec: u64,
    conv: u64,
    proj: u64,
    out: u64,
    k: u64,
    v: u64,
    gates: u64,
    b: u64,
    pos_dev: u64,
    pub fn layout(g: Geometry, count: usize, rows: usize) [9]usize {
        const h = g.lin_heads;
        const l = G.kda_layers;
        const r = count * l * rows;
        // pos_dev and mtp_pos_dev, one int32 each, on their own 16-byte stride: the captured Triton kernels are specialised on 16-byte-aligned pointers (mtp at +4 missed _dense_chunks).
        return .{ count * 2 * l * h * 128 * 128 * 4, count * l * 3 * g.convDim() * 2, r * g.projWidth() * 2, r * h * 128 * 2, r * h * 128 * 4, r * h * 128 * 2, r * h * 128 * 4, r * h * 4, count * 32 };
    }
    pub fn bytes(g: Geometry, count: usize, rows: usize) usize {
        return aligned(&layout(g, count, rows));
    }
    pub fn init(d: *const cuda.Driver, g: Geometry, count: usize, rows: usize) !Slots {
        if (count < 1 or count > 8 or rows < 1) return error.BadSlots;
        const sizes = layout(g, count, rows);
        var a = try Arena.init(d, &sizes);
        errdefer a.deinit();
        var slots: Slots = .{ .arena = undefined, .g = g, .count = count, .rows = rows, .rec = a.take(sizes[0]), .conv = a.take(sizes[1]), .proj = a.take(sizes[2]), .out = a.take(sizes[3]), .k = a.take(sizes[4]), .v = a.take(sizes[5]), .gates = a.take(sizes[6]), .b = a.take(sizes[7]), .pos_dev = a.take(sizes[8]) };
        slots.arena = a;
        return slots;
    }
    pub fn deinit(s: *Slots) void {
        s.arena.deinit();
    }
    pub fn view(s: Slots, slot: usize) !Slot {
        if (slot >= s.count) return error.BadSlot;
        const n = layout(s.g, 1, s.rows);
        const pos = s.pos_dev + slot * 32;
        return .{ .rec = s.rec + slot * n[0], .conv = s.conv + slot * n[1], .proj = s.proj + slot * n[2], .out = s.out + slot * n[3], .k = s.k + slot * n[4], .v = s.v + slot * n[5], .g = s.gates + slot * n[6], .b = s.b + slot * n[7], .pos_dev = pos, .mtp_pos_dev = pos + 16 };
    }
};
pub const CacheView = struct { latent: [11]u64, pooled: [11]u64, keys: [11]u64, gates: [11]u64, ring_base: usize, capacity: usize };
/// Shared token arena plus per-stream index rings. Cache bytes are replicated on every TP rank.
pub const Caches = struct {
    arena: Arena,
    span: ?display.Span = null,
    rows: usize,
    streams: usize,
    ring: usize,
    latent: [11]u64,
    pooled: [11]u64,
    keys: [11]u64,
    gates: [11]u64,
    pub fn layout(rows: usize, streams: usize, ring: usize) [44]usize {
        var sizes: [44]usize = undefined;
        for (0..11) |i| {
            sizes[4 * i ..][0..4].* = .{ rows * G.latent_row, (rows / 4 + 2) * G.pool_row, streams * ring * 128 * 2, streams * ring * 128 * 2 };
        }
        return sizes;
    }
    pub fn bytes(rows: usize, streams: usize, ring: usize) usize {
        return aligned(&layout(rows, streams, ring));
    }
    pub fn init(d: *const cuda.Driver, rows: usize, streams: usize, window: usize) !Caches {
        return initDisplay(std.heap.c_allocator, std.Io.Threaded.global_single_threaded.io(), d, 0, rows, streams, window, .{});
    }
    pub fn initDisplay(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, device: c_int, rows: usize, streams: usize, window: usize, cfg: display.Config) !Caches {
        if (rows == 0 or rows % G.extent != 0 or streams < 1 or streams > 8 or window == 0) return error.BadCaches;
        const ring = G.indexRing(rows, window);
        if (ring < window + 3) return error.BadRing;
        var sizes = layout(rows, streams, ring);
        const plane = rows * G.latent_row;
        const count = if (cfg.bytes == 0) 0 else try display.planes(11, plane, cfg.bytes);
        var span: ?display.Span = if (count == 0) null else try display.map(gpa, io, d, device, cfg, count * plane - cfg.bytes);
        errdefer if (span) |*v| v.deinit();
        for (0..count) |i| sizes[4 * i] = 0;
        var a = try Arena.init(d, &sizes);
        errdefer a.deinit();
        var c: Caches = .{ .arena = undefined, .rows = rows, .streams = streams, .ring = ring, .latent = undefined, .pooled = undefined, .keys = undefined, .gates = undefined };
        if (span) |v| for (try display.zeroRanges(v, count * plane)) |r| {
            const part: cuda.DeviceBuffer = .{ .d = d, .ptr = r.ptr, .len = r.len };
            try part.fill8(0, null);
        };
        for (0..11) |i| {
            c.latent[i] = if (i < count) span.?.ptr + i * plane else a.take(sizes[4 * i]);
            c.pooled[i] = a.take(sizes[4 * i + 1]);
            c.keys[i] = a.take(sizes[4 * i + 2]);
            c.gates[i] = a.take(sizes[4 * i + 3]);
        }
        c.arena = a;
        c.span = span;
        return c;
    }
    /// One independent DSA plane for the checkpoint MTP head; no target cache aliases.
    pub fn initMtp(d: *const cuda.Driver, rows: usize, window: usize) !Caches {
        const ring = G.indexRing(rows, window);
        const sizes = [_]usize{ rows * G.latent_row, (rows / 4 + 2) * G.pool_row, ring * 128 * 2, ring * 128 * 2 };
        var a = try Arena.init(d, &sizes);
        errdefer a.deinit();
        const latent = a.take(sizes[0]);
        const pooled = a.take(sizes[1]);
        const keys = a.take(sizes[2]);
        const gates = a.take(sizes[3]);
        return .{ .arena = a, .rows = rows, .streams = 1, .ring = ring, .latent = @splat(latent), .pooled = @splat(pooled), .keys = @splat(keys), .gates = @splat(gates) };
    }
    pub fn deinit(c: *Caches) void {
        c.arena.deinit();
        if (c.span) |*v| v.deinit();
    }
    pub fn view(c: Caches, base: usize, size: usize, slot: usize) !CacheView {
        if (base % G.extent != 0 or size == 0 or size % G.extent != 0 or base > c.rows or size > c.rows - base or slot >= c.streams) return error.BadExtent;
        var v: CacheView = .{ .latent = undefined, .pooled = undefined, .keys = undefined, .gates = undefined, .ring_base = slot * c.ring, .capacity = size };
        for (0..11) |i| {
            v.latent[i] = c.latent[i] + base * G.latent_row;
            v.pooled[i] = c.pooled[i] + base / 4 * G.pool_row;
            v.keys[i] = c.keys[i] + slot * c.ring * 128 * 2;
            v.gates[i] = c.gates[i] + slot * c.ring * 128 * 2;
        }
        return v;
    }
};
/// Host commit metadata plus views; ownership belongs to Slots/Caches. Parity flips only after accepted replay.
pub const State = struct {
    slot: usize,
    base: usize,
    capacity: usize,
    pos: usize = 0,
    cur: [34]u1 = @splat(0),
    fixed: Slot,
    cache: CacheView,
    pub fn bind(slots: Slots, caches: Caches, slot: usize, base: usize, size: usize) !State {
        return .{ .slot = slot, .base = base, .capacity = size, .fixed = try slots.view(slot), .cache = try caches.view(base, size, slot) };
    }
    pub fn recurrent(s: State, g: Geometry, layer: usize, parity: u1) u64 {
        std.debug.assert(layer < 34);
        return s.fixed.rec + (@as(usize, parity) * 34 + layer) * g.lin_heads * 128 * 128 * 4;
    }
};
/// P0's stable common forward scratch; matmul/selection packages add their own shape-dependent workspaces.
pub const Buffers = struct {
    arena: Arena,
    rows: usize,
    ids: u64,
    x: u64,
    normed: u64,
    hc_part: u64,
    xs: u64,
    post: u64,
    comb: u64,
    part: u64,
    gathered: u64,
    hidden: u64,
    taps: u64,
    logits: u64,
    q_tmp: u64,
    y_tmp: u64,
    pub fn layout(g: Geometry, rows: usize, prefill: bool) [14]usize {
        const r = rows;
        return .{ r * 4, r * g.wide() * 2, r * 4096 * 2, r * 16 * 32 * 4, r * 64 * 4, r * 4 * 4, r * 16 * 4, r * 4096 * 4, g.world * r * 4096 * 4, r * 4096 * 2, 5 * r * 4096 * 2, (if (prefill) @as(usize, 1) else r) * g.vocab * 2, r * g.lin_heads * 128 * 4, r * g.lin_heads * 128 * 2 };
    }
    pub fn bytes(g: Geometry, rows: usize, prefill: bool) usize {
        return aligned(&layout(g, rows, prefill));
    }
    pub fn init(d: *const cuda.Driver, g: Geometry, rows: usize, prefill: bool) !Buffers {
        if (rows == 0) return error.BadRows;
        const sizes = layout(g, rows, prefill);
        var a = try Arena.init(d, &sizes);
        errdefer a.deinit();
        var b: Buffers = undefined;
        b.rows = rows;
        const ptrs = [_]*u64{ &b.ids, &b.x, &b.normed, &b.hc_part, &b.xs, &b.post, &b.comb, &b.part, &b.gathered, &b.hidden, &b.taps, &b.logits, &b.q_tmp, &b.y_tmp };
        for (ptrs, sizes) |p, n| p.* = a.take(n);
        b.arena = a;
        return b;
    }
    pub fn deinit(b: *Buffers) void {
        b.arena.deinit();
    }
};
test "TP2 slots retain both recurrent parities and all replay inputs" {
    const g = try G.init(2, 0);
    const n = Slots.layout(g, 1, 32);
    try std.testing.expectEqual(@as(usize, 142606336), n[0]);
    try std.testing.expectEqual(@as(usize, 34 * 3 * 12288 * 2), n[1]);
    try std.testing.expectEqual(@as(usize, 34 * 32 * 12576 * 2), n[2]);
    try std.testing.expectEqual(@as(usize, 32), n[8]);
    const c = Caches.layout(4096, 2, 64);
    try std.testing.expectEqual(@as(usize, 4096 * 528), c[0]);
    try std.testing.expectEqual(@as(usize, 1026 * 144), c[1]);
    try std.testing.expectEqual(@as(usize, 2 * 64 * 128 * 2), c[2]);
    try std.testing.expect(Buffers.bytes(g, 2048, true) > Buffers.bytes(g, 32, false));
}

test "every slot's position pair is 16-byte aligned (captured Triton kernels are specialised on it)" {
    const g = try G.init(2, 0);
    const slots: Slots = .{ .arena = undefined, .g = g, .count = 8, .rows = 32, .rec = 0x1000, .conv = 0x2000, .proj = 0x3000, .out = 0x4000, .k = 0x5000, .v = 0x6000, .gates = 0x7000, .b = 0x8000, .pos_dev = 0x9000 };
    for (0..8) |i| {
        const v = try slots.view(i);
        try std.testing.expectEqual(@as(u64, 0), v.pos_dev % 16);
        try std.testing.expectEqual(@as(u64, 0), v.mtp_pos_dev % 16);
        try std.testing.expect(v.mtp_pos_dev == v.pos_dev + 16);
    }
}

test "slot, parity and extent views use reference strides without device allocation" {
    const g = try G.init(2, 0);
    const slots: Slots = .{ .arena = undefined, .g = g, .count = 2, .rows = 32, .rec = 0x1000, .conv = 0x2000, .proj = 0x3000, .out = 0x4000, .k = 0x5000, .v = 0x6000, .gates = 0x7000, .b = 0x8000, .pos_dev = 0x9000 };
    const caches: Caches = .{ .arena = undefined, .rows = 8192, .streams = 2, .ring = 64, .latent = @splat(0x100000), .pooled = @splat(0x200000), .keys = @splat(0x300000), .gates = @splat(0x400000) };
    const s = try State.bind(slots, caches, 1, 2048, 4096);
    try std.testing.expectEqual(@as(u64, 0x1000 + 142606336), s.fixed.rec);
    try std.testing.expectEqual(@as(u64, 0x9000 + 32), s.fixed.pos_dev);
    try std.testing.expectEqual(@as(u64, 0x9000 + 48), s.fixed.mtp_pos_dev);
    try std.testing.expectEqual(@as(u64, 0x100000 + 2048 * 528), s.cache.latent[0]);
    try std.testing.expectEqual(@as(u64, 0x200000 + 512 * 144), s.cache.pooled[0]);
    try std.testing.expectEqual(@as(u64, 0x300000 + 64 * 256), s.cache.keys[0]);
    try std.testing.expectEqual(s.fixed.rec + 34 * 32 * 128 * 128 * 4, s.recurrent(g, 0, 1));
    try std.testing.expectError(error.BadExtent, caches.view(1, 2048, 0));
    try std.testing.expectError(error.BadExtent, caches.view(6144, 4096, 0));
}
