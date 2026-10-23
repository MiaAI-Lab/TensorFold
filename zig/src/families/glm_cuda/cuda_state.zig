//! Device layouts from glm5_next/cuda/forward.py: Slots, Caches, State; kda.py: KDAScratchSet.
//! FP8 cache rows follow glm5_next/cuda/kv8.py. No arithmetic or quantization is performed here.
const std = @import("std");
const cuda = @import("cuda");
pub const Geometry = @import("cuda_geometry.zig").Geometry;
const G = Geometry;
const display = @import("cuda_display_kv.zig");
const shard = @import("cuda_shard.zig");
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
/// `owner`: this rank holds the extent's latent rows (TF_GLM_KV_SHARD: false where the other rank's part holds them;
/// `latent` then points at the extent's twin here, else it is `no_latent`, as Python's Plane.view gives None: nothing
/// on this rank may write or copy a latent row of it, and a consumer refuses such a view or branches on `owner`. The
/// rows of its decode windows reach this rank's trash rows through the window's glue, not a view.) `capacity` stays
/// the extent's: the pooled keys hold every row on every rank.
pub const CacheView = struct {
    latent: [11]u64,
    pooled: [11]u64,
    keys: [11]u64,
    gates: [11]u64,
    ring_base: usize,
    capacity: usize,
    owner: bool = true,
    /// Whether `latent` addresses the extent's rows on this rank (its own, or its twin).
    pub fn hasLatent(v: CacheView) bool {
        return v.latent[0] != no_latent;
    }
};
/// `latent` of a view with no latent row on this rank (TF_GLM_KV_SHARD: the other rank's extent, no twin here).
pub const no_latent: u64 = 0;
/// Shared token arena plus per-stream index rings. Cache bytes are replicated on every TP rank, except with
/// TF_GLM_KV_SHARD (`shard`) the latent planes: a rank's hold its part of the pool and the trash rows.
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
    shard: ?shard.Shard = null,
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
    /// `layout` with TF_GLM_KV_SHARD: each latent plane `s`'s part and trash rows; the pooled keys pool-wide.
    pub fn layoutShard(rows: usize, streams: usize, ring: usize, s: shard.Shard) [44]usize {
        var sizes = layout(rows, streams, ring);
        for (0..11) |i| sizes[4 * i] = s.latentRows() * G.latent_row;
        return sizes;
    }
    pub fn shardBytes(rows: usize, streams: usize, ring: usize, s: shard.Shard) usize {
        return aligned(&layoutShard(rows, streams, ring, s));
    }
    pub fn init(d: *const cuda.Driver, rows: usize, streams: usize, window: usize) !Caches {
        return initDisplay(std.heap.c_allocator, std.Io.Threaded.global_single_threaded.io(), d, 0, rows, streams, window, .{});
    }
    pub fn initDisplay(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, device: c_int, rows: usize, streams: usize, window: usize, cfg: display.Config) !Caches {
        return initWith(gpa, io, d, device, rows, streams, window, cfg, null);
    }
    /// TF_GLM_KV_SHARD's caches (layoutShard) over the pool `s` cuts, the display reservation on its latent planes.
    pub fn initShard(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, device: c_int, streams: usize, window: usize, cfg: display.Config, s: shard.Shard) !Caches {
        return initWith(gpa, io, d, device, s.pool, streams, window, cfg, s);
    }
    fn initWith(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, device: c_int, rows: usize, streams: usize, window: usize, cfg: display.Config, sh: ?shard.Shard) !Caches {
        if (rows == 0 or rows % G.extent != 0 or streams < 1 or streams > 8 or window == 0) return error.BadCaches;
        const ring = G.indexRing(rows, window);
        if (ring < window + 3) return error.BadRing;
        var sizes = if (sh) |s| layoutShard(rows, streams, ring, s) else layout(rows, streams, ring);
        const plane = (if (sh) |s| s.latentRows() else rows) * G.latent_row;
        const reserved: display.Reserve = if (cfg.bytes == 0) .{ .planes = 0, .ordinary = 0 } else try display.reserve(11, plane, cfg.bytes);
        const count = reserved.planes;
        var span: ?display.Span = if (count == 0) null else try display.map(gpa, io, d, device, cfg, reserved.ordinary);
        errdefer if (span) |*v| v.deinit();
        for (0..count) |i| sizes[4 * i] = 0;
        var a = try Arena.init(d, &sizes);
        errdefer a.deinit();
        var c: Caches = .{ .arena = undefined, .rows = rows, .streams = streams, .ring = ring, .latent = undefined, .pooled = undefined, .keys = undefined, .gates = undefined, .shard = sh };
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
        return c.viewAt(base, size, slot, null);
    }
    /// `view` of a filling prompt whose extent the other rank owns, its latent rows in its twin at pool row `twin`, in
    /// this rank's part (TF_GLM_KV_SHARD's prompt borrowing; Python Caches.bind's twin_base). A twin has its extent's
    /// size, so the rows must lie in this rank's part.
    pub fn viewTwin(c: Caches, base: usize, size: usize, slot: usize, twin: usize) !CacheView {
        const s = c.shard orelse return error.BadTwin;
        if (s.owns(base) or !s.owns(twin) or twin % G.extent != 0 or !s.within(twin, size)) return error.BadTwin;
        return c.viewAt(base, size, slot, twin);
    }
    fn viewAt(c: Caches, base: usize, size: usize, slot: usize, twin: ?usize) !CacheView {
        if (base % G.extent != 0 or size == 0 or size % G.extent != 0 or base > c.rows or size > c.rows - base or slot >= c.streams) return error.BadExtent;
        // TF_GLM_KV_SHARD: an extent in one part; this rank's latent row of it, else its twin's, else none here
        const s = c.shard;
        if (s != null and !s.?.within(base, size)) return error.BadExtent;
        const owner = if (s) |sh| sh.owns(base) else true;
        const lrow: ?usize = if (s) |sh| (if (owner) sh.local(base) else if (twin) |t| sh.local(t) else null) else base;
        var v: CacheView = .{ .latent = undefined, .pooled = undefined, .keys = undefined, .gates = undefined, .ring_base = slot * c.ring, .capacity = size, .owner = owner };
        for (0..11) |i| {
            v.latent[i] = if (lrow) |r| c.latent[i] + r * G.latent_row else no_latent;
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

test "TF_GLM_KV_SHARD caches: latent planes a part and its trash rows, pooled keys pool-wide, views by owner" {
    const rows = 8 * 2048;
    const s0 = try shard.Shard.init(0, 2, rows, shard.trash);
    const s1 = try shard.Shard.init(1, 2, rows, shard.trash);
    const plain = Caches.layout(rows, 2, 64);
    const sharded = Caches.layoutShard(rows, 2, 64, s0);
    for (0..11) |i| {
        try std.testing.expectEqual(@as(usize, (rows / 2 + 64) * 528), sharded[4 * i]);
        try std.testing.expectEqualSlices(usize, plain[4 * i + 1 ..][0..3], sharded[4 * i + 1 ..][0..3]);
    }
    try std.testing.expectEqual(Caches.shardBytes(rows, 2, 64, s0), Caches.shardBytes(rows, 2, 64, s1)); // alike on every rank
    try std.testing.expect(Caches.shardBytes(rows, 2, 64, s0) < Caches.bytes(rows, 2, 64));
    const at = struct {
        fn caches(s: ?shard.Shard) Caches {
            return .{ .arena = undefined, .rows = rows, .streams = 2, .ring = 64, .latent = @splat(0x100000), .pooled = @splat(0x200000), .keys = @splat(0x300000), .gates = @splat(0x400000), .shard = s };
        }
    };
    // off: the pool row; rank 0 owns part 0, rank 1 part 1, each at its part-local row
    const off = try at.caches(null).view(4096, 2048, 1);
    try std.testing.expectEqual(@as(u64, 0x100000 + 4096 * 528), off.latent[0]);
    try std.testing.expect(off.owner);
    const mine = try at.caches(s1).view(8192 + 2048, 4096, 1);
    try std.testing.expect(mine.owner);
    try std.testing.expectEqual(@as(u64, 0x100000 + 2048 * 528), mine.latent[10]);
    try std.testing.expectEqual(@as(u64, 0x200000 + (8192 + 2048) / 4 * 144), mine.pooled[10]); // pooled keys at the pool row
    try std.testing.expectEqual(@as(u64, 0x300000 + 64 * 256), mine.keys[10]);
    // the other rank's extent: no latent row here (Python's None: no write or copy at its positions can land past the
    // 64 trash rows), its pooled keys and capacity as the owner's; or its twin's rows here
    const theirs = try at.caches(s0).view(8192 + 2048, 4096, 1);
    try std.testing.expect(!theirs.owner and !theirs.hasLatent() and mine.hasLatent() and off.hasLatent());
    for (theirs.latent) |l| try std.testing.expectEqual(no_latent, l);
    try std.testing.expectEqual(mine.pooled[0], theirs.pooled[0]);
    try std.testing.expectEqual(mine.capacity, theirs.capacity);
    const twin = try at.caches(s0).viewTwin(8192 + 2048, 4096, 1, 2048);
    try std.testing.expect(!twin.owner and twin.hasLatent());
    try std.testing.expectEqual(@as(u64, 0x100000 + 2048 * 528), twin.latent[0]);
    // an extent across the parts, a twin in the other part or past this one, or one of an owned extent: refused
    try std.testing.expectError(error.BadExtent, at.caches(s0).view(6144, 4096, 0));
    try std.testing.expectError(error.BadTwin, at.caches(s0).viewTwin(8192, 2048, 0, 8192 + 4096));
    try std.testing.expectError(error.BadTwin, at.caches(s0).viewTwin(8192, 4096, 0, 6144));
    try std.testing.expectError(error.BadTwin, at.caches(s0).viewTwin(0, 2048, 0, 2048));
    try std.testing.expectError(error.BadTwin, at.caches(null).viewTwin(8192, 2048, 0, 0));
    // State.bind takes the sharded view
    const g = try G.init(2, 1);
    const slots: Slots = .{ .arena = undefined, .g = g, .count = 2, .rows = 32, .rec = 0x1000, .conv = 0x2000, .proj = 0x3000, .out = 0x4000, .k = 0x5000, .v = 0x6000, .gates = 0x7000, .b = 0x8000, .pos_dev = 0x9000 };
    const st = try State.bind(slots, at.caches(s1), 0, 0, 2048);
    try std.testing.expect(!st.cache.owner and !st.cache.hasLatent());
}
