//! The one-shot RoCE all-gather between 2 to 4 ranks (decode plan D1, work/research/R1-decode.md 3.1): b12x's
//! "RoCEnante" protocol (https://github.com/local-inference-lab/b12x, b12x/comm/roce, by the b12x contributors at
//! Local Inference Lab, Apache License 2.0) in a new implementation: the host proxy in roce_proxy.c (libibverbs opened
//! at run time), the GPU kernel in zig/kernels/cuda/fn_roce.cu. A gather of up to `slot_bytes` a rank costs one kernel
//! launch, with no host work on the receive side; the byte layout is ncclAllGather's (rank 0's shard first), so it is
//! a drop-in for small gathers and graph-capturable (the sequence lives in device memory).
//!
//! Setup: `open` (region, rails, a queue pair per rail and peer), every rank's `blob` goes to every rank (the caller's
//! control link), beyond two ranks `planRoutes` pairs the rails of every two ranks by subnet and `setRoutes` (and, on
//! a ring of four cabled to the neighbours only, `setRing`: the neighbours relay the stripes to the rank across)
//! tell the proxy, then `connect`, `start` (the proxy thread). All ranks must make the same gathers in the same
//! order, as for NCCL.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const memory = @import("memory.zig");
const module = @import("module.zig");
const launch = @import("launch.zig");
const Stream = @import("stream.zig").Stream;
const kernels = @import("kernels.zig");

const Proxy = opaque {};
extern fn tf_roce_abi() c_int;
extern fn tf_roce_blob_bytes() u64;
extern fn tf_roce_layout(world: c_int, slot_bytes: u64, out: *[5]u64) c_int;
extern fn tf_roce_ipv4(hca: [*:0]const u8, gid_index: *c_int, ip: *u32, prefix: *c_int) c_int;
extern fn tf_roce_open(world: c_int, rank: c_int, names: [*]const [*:0]const u8, n_hca: c_int, gid_index: [*]const c_int, traffic_class: c_int, region: *anyopaque, region_bytes: u64, slot_bytes: u64, err: [*]u8, err_len: u64) ?*Proxy;
extern fn tf_roce_set_routes(r: *Proxy, peer: c_int, n: c_int, local: [*]const c_int, remote: [*]const c_int, n_flag: c_int) c_int;
extern fn tf_roce_set_ring(r: *Proxy) c_int;
extern fn tf_roce_blob_of(r: *Proxy, out: *anyopaque) c_int;
extern fn tf_roce_connect(r: *Proxy, blobs: *const anyopaque, len: u64) c_int;
extern fn tf_roce_start(r: *Proxy) c_int;
extern fn tf_roce_failed(r: *Proxy) c_int;
extern fn tf_roce_error(r: *Proxy) [*:0]const u8;
extern fn tf_roce_ops(r: *Proxy) u64;
extern fn tf_roce_relayed(r: *Proxy) u64;
extern fn tf_roce_close(r: *Proxy) void;

/// Rails (RDMA devices) a rank uses, and ranks: the proxy's ROCE_HCAS_MAX and ROCE_WORLD_MAX.
pub const max_ports = 4;
pub const max_world = 4;


/// Read-only bytes to pull into the L2 while a gather waits (decode plan D2): the next kernels' weights.
pub const Prefetch = struct { ptr: abi.DevicePtr = 0, bytes: usize = 0 };

/// L2 prefetch blocks for `bytes`: four 128-byte lines a thread of 512, at most 64 blocks.
fn prefetchBlocks(bytes: usize) usize {
    if (bytes == 0) return 0;
    return std.math.clamp((bytes / 128 + 2047) / 2048, 1, 64);
}

pub const prefetch_symbol: [:0]const u8 = "_ZN10tf_fn_roce8prefetchEPKcy";

/// The prefetch alone, for gathers NCCL makes: one launch on the gather's stream just before it; the lines it asks
/// for keep arriving while NCCL waits on the network.
pub const L2 = struct {
    mod: module.Module,
    f: module.Function,
    d: *const Driver,

    pub fn init(d: *const Driver) !L2 {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try module.Module.load(d, kernels.fn_roce);
        errdefer mod.unload();
        return .{ .mod = mod, .f = try mod.function(prefetch_symbol), .d = d };
    }

    pub fn deinit(l: *L2) void {
        l.mod.unload();
    }

    pub fn run(l: *const L2, p: Prefetch, stream: abi.Stream) !void {
        const blocks = prefetchBlocks(p.bytes);
        if (blocks == 0 or p.ptr == 0) return;
        var a: launch.Args = .{};
        a.add(p.ptr);
        a.add(@as(u64, p.bytes));
        const s: Stream = .{ .d = l.d, .handle = stream };
        try launch.launch(l.f, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = 512 } }, s, &a);
    }
};

/// The kernel's two forms: 16-byte units (aligned pointers, a multiple of 16 bytes) and 4-byte units.
pub const symbols = [2][:0]const u8{
    "_ZN10tf_fn_roce6gatherI5uint4EEvPKT_PS2_ijS5_S4_PKjPjyPNS_5StateEiiiyiPKcy",
    "_ZN10tf_fn_roce6gatherIjEEvPKT_PS1_ijS4_S3_PKjPjyPNS_5StateEiiiyiPKcy",
};

pub const Settings = struct {
    /// RDMA device names, one a port (rail), e.g. rocep1s0f1 and roceP2p1s0f1; the peer lists its own in the same order
    ports: []const []const u8,
    /// the GID index of every rail (one entry each), else `gid_index` for all
    gids: []const c_int = &.{},
    gid_index: c_int = 3,
    traffic_class: c_int = 0,
    /// the largest shard a rank, a multiple of 4096 (larger gathers go elsewhere)
    slot_bytes: u64 = 512 << 10,
    /// how long a kernel waits for the peer before it stops the stream (a trap: the peer is gone or its proxy failed)
    timeout_s: u64 = 600,
};

pub const Roce = struct {
    d: *const Driver,
    proxy: *Proxy,
    region: memory.HostBuffer,
    region_dev: abi.DevicePtr,
    state: memory.DeviceBuffer,
    mod: module.Module,
    fns: [2]module.Function,
    layout: [5]u64,
    rank: u32,
    world: u32,
    /// rails (RDMA devices)
    ports: u32,
    /// stripes a peer: the rails at two ranks, the routes' common count beyond (setRoutes)
    n_flag: u32,
    slot_bytes: u64,
    timeout_ns: u64,

    pub fn blobBytes() usize {
        return @intCast(tf_roce_blob_bytes());
    }

    /// Two ranks (Flash Next's form): rail i to the peer's rail i, every rail. See `openRanks`.
    pub fn open(d: *const Driver, rank: u32, s: Settings) !Roce {
        return openRanks(d, rank, 2, s);
    }

    /// 2 to 4 ranks. `d` must outlive the value; the current CUDA context is the engine's.
    pub fn openRanks(d: *const Driver, rank: u32, world: u32, s: Settings) !Roce {
        if (world < 2 or world > max_world or rank >= world or s.ports.len < 1 or s.ports.len > max_ports) return error.Invalid;
        if (s.gids.len != 0 and s.gids.len != s.ports.len) return error.Invalid;
        if (!kernels.available) return error.BuiltWithoutKernels;
        if (tf_roce_abi() != 7) return error.Invalid;
        var lay: [5]u64 = undefined;
        if (tf_roce_layout(@intCast(world), s.slot_bytes, &lay) != 0) return error.Invalid;
        var region = try memory.HostBuffer.allocMapped(d, @intCast(lay[4]));
        errdefer region.free();
        @memset(region.bytes, 0);
        const dev = try region.device();
        var names: [max_ports][256:0]u8 = undefined;
        var ptrs: [max_ports][*:0]const u8 = undefined;
        for (s.ports, 0..) |p, i| {
            if (p.len >= 256) return error.Invalid;
            @memcpy(names[i][0..p.len], p);
            names[i][p.len] = 0;
            ptrs[i] = &names[i];
        }
        var err: [512]u8 = @splat(0);
        var gids: [max_ports]c_int = undefined;
        for (0..s.ports.len) |i| gids[i] = if (s.gids.len != 0) s.gids[i] else s.gid_index;
        const proxy = tf_roce_open(@intCast(world), @intCast(rank), &ptrs, @intCast(s.ports.len), &gids, s.traffic_class, region.bytes.ptr, lay[4], s.slot_bytes, &err, err.len) orelse {
            std.log.err("RoCE: {s}", .{std.mem.sliceTo(&err, 0)});
            return error.RoceUnavailable;
        };
        errdefer tf_roce_close(proxy);
        var state = try memory.DeviceBuffer.alloc(d, 16);
        errdefer state.free();
        try state.fill32(0, null);
        var mod = try module.Module.load(d, kernels.fn_roce);
        errdefer mod.unload();
        var fns: [2]module.Function = undefined;
        for (symbols, &fns) |name, *f| f.* = try mod.function(name);
        return .{ .d = d, .proxy = proxy, .region = region, .region_dev = dev, .state = state, .mod = mod, .fns = fns, .layout = lay, .rank = rank, .world = world, .ports = @intCast(s.ports.len), .n_flag = @intCast(s.ports.len), .slot_bytes = s.slot_bytes, .timeout_ns = s.timeout_s * std.time.ns_per_s };
    }

    pub fn deinit(r: *Roce) void {
        tf_roce_close(r.proxy);
        r.mod.unload();
        r.state.free();
        r.region.free();
    }

    /// What the peer needs to connect (blobBytes() bytes into `out`).
    pub fn blob(r: *Roce, out: []u8) !void {
        if (out.len < blobBytes()) return error.Invalid;
        if (tf_roce_blob_of(r.proxy, out.ptr) != 0) return error.Invalid;
    }

    /// Stripes of the shard to `peer` over (local[i], remote[i]) rails; every peer gets the same count (`local.len`).
    pub fn setRoutes(r: *Roce, peer: u32, local: []const c_int, remote: []const c_int) !void {
        if (peer >= r.world or peer == r.rank or local.len == 0 or local.len != remote.len or local.len > max_ports) return error.Invalid;
        if (tf_roce_set_routes(r.proxy, @intCast(peer), @intCast(local.len), local.ptr, remote.ptr, @intCast(local.len)) != 0) {
            std.log.err("RoCE routes: {s}", .{std.mem.span(tf_roce_error(r.proxy))});
            return error.RoceUnavailable;
        }
        r.n_flag = @intCast(local.len);
    }

    /// Four ranks cabled in rank order, each only to its neighbours (after setRoutes to both): the neighbours relay.
    pub fn setRing(r: *Roce) !void {
        if (r.world != 4 or tf_roce_set_ring(r.proxy) != 0) {
            std.log.err("RoCE ring: {s}", .{std.mem.span(tf_roce_error(r.proxy))});
            return error.RoceUnavailable;
        }
    }

    /// Two ranks (Flash Next's form): the peer's blob; this rank's own is made here.
    pub fn connect(r: *Roce, peer: []const u8) !void {
        if (r.world != 2 or peer.len != blobBytes()) return error.BadRoceBlob;
        var both: [2 * 512]u8 = undefined;
        const n = blobBytes();
        if (2 * n > both.len) return error.BadRoceBlob;
        const own = both[r.rank * n ..][0..n];
        const other = both[(1 - r.rank) * n ..][0..n];
        try r.blob(own);
        @memcpy(other, peer);
        try r.connectAll(both[0 .. 2 * n]);
    }

    /// Every rank's blob, in rank order (`world * blobBytes()` bytes).
    pub fn connectAll(r: *Roce, blobs: []const u8) !void {
        if (blobs.len != r.world * blobBytes()) return error.BadRoceBlob;
        if (tf_roce_connect(r.proxy, blobs.ptr, blobs.len) != 0) {
            std.log.err("RoCE connect: {s}", .{std.mem.span(tf_roce_error(r.proxy))});
            return error.RoceUnavailable;
        }
    }

    pub fn start(r: *Roce) !void {
        if (tf_roce_start(r.proxy) != 0) {
            std.log.err("RoCE start: {s}", .{std.mem.span(tf_roce_error(r.proxy))});
            return error.RoceUnavailable;
        }
    }

    /// The proxy's failure, if any (an RDMA write that did not complete): the kernels then time out and trap.
    pub fn failure(r: *const Roce) ?[]const u8 {
        if (tf_roce_failed(r.proxy) == 0) return null;
        return std.mem.span(tf_roce_error(r.proxy));
    }

    /// A kernel's wait that timed out (it trapped, a fatal error for the context): the sequence and the peer.
    pub fn timedOut(r: *const Roce) ?struct { seq: u32, peer: u32 } {
        const ctrl: *const [32]u32 = @ptrCast(@alignCast(r.region.bytes[@intCast(r.layout[3])..][0..128].ptr));
        const seq = @atomicLoad(u32, &ctrl[2], .acquire);
        if (seq == 0) return null;
        return .{ .seq = seq, .peer = @atomicLoad(u32, &ctrl[3], .acquire) };
    }

    pub fn ops(r: *const Roce) u64 {
        return tf_roce_ops(r.proxy);
    }

    /// Stripes this rank's proxy has relayed for the rank across the ring.
    pub fn relayed(r: *const Roce) u64 {
        return tf_roce_relayed(r.proxy);
    }

    /// Gathers this rank fits: up to slot_bytes a rank, whole 4-byte words.
    pub fn fits(r: *const Roce, bytes: usize) bool {
        return bytes > 0 and bytes <= r.slot_bytes and bytes % 4 == 0;
    }

    /// fits, with both pointers 4-byte aligned and the send buffer either apart from the receive one or exactly this
    /// rank's block of it (NCCL's in-place form: the kernel then rewrites each word of that block with itself, after
    /// staging it, and never reads the peers' blocks from `send`).
    pub fn takes(r: *const Roce, send: abi.DevicePtr, recv: abi.DevicePtr, bytes: usize) bool {
        if (!r.fits(bytes) or send % 4 != 0 or recv % 4 != 0) return false;
        if (send == recv + r.rank * bytes) return true;
        return send + bytes <= recv or recv + r.world * bytes <= send;
    }

    /// `bytes` from `send` on each rank into `recv` [world][bytes], rank 0's first, on `stream` (capturable); `pf`'s lines
    /// prefetched into the L2 by extra blocks of the same launch. Gathers must run one after another (one stream, or
    /// streams ordered by events): every launch shares the device sequence and counters.
    pub fn allGather(r: *const Roce, send: abi.DevicePtr, recv: abi.DevicePtr, bytes: usize, stream: abi.Stream, pf: Prefetch) !void {
        if (!r.takes(send, recv, bytes)) return error.Invalid;
        const wide = bytes % 16 == 0 and send % 16 == 0 and recv % 16 == 0;
        const unit: usize = if (wide) 16 else 4;
        const n = bytes / unit;
        const threads: usize = 512;
        // the grid stays resident (every block waits for the peer): at most 16 blocks, ~2 units a thread
        const blocks = std.math.clamp((n + threads * 2 - 1) / (threads * 2), 1, 16);
        const padded: u32 = @intCast((bytes + 15) / 16 * 16);
        var a: launch.Args = .{};
        a.add(send);
        a.add(recv);
        a.add(@as(c_int, @intCast(n)));
        a.add(padded);
        a.add(r.region_dev + r.layout[2]); // send slots
        a.add(r.region_dev + r.layout[0]); // receive slots
        a.add(r.region_dev + r.layout[1]); // flags
        a.add(r.region_dev + r.layout[3]); // ctrl
        a.add(@as(u64, r.slot_bytes));
        a.add(r.state.ptr);
        a.add(@as(c_int, @intCast(r.rank)));
        a.add(@as(c_int, @intCast(r.world)));
        a.add(@as(c_int, @intCast(r.n_flag)));
        a.add(@as(u64, r.timeout_ns));
        a.add(@as(c_int, @intCast(blocks)));
        const pf_blocks = if (pf.ptr != 0) prefetchBlocks(pf.bytes) else 0;
        a.add(pf.ptr);
        a.add(@as(u64, if (pf_blocks > 0) pf.bytes else 0));
        const s: Stream = .{ .d = r.d, .handle = stream };
        try launch.launch(r.fns[if (wide) 0 else 1], .{ .grid = .{ .x = @intCast(blocks + pf_blocks) }, .block = .{ .x = @intCast(threads) } }, s, &a);
    }
};

/// NCCL_IB_HCA's device names ("rocep1s0f1,roceP2p1s0f1", an optional leading = and :1 suffixes; the proxy uses port
/// 1, so another port is refused), at most `max_ports`; `out` holds slices of `text`.
pub fn parseHcas(text: []const u8, out: *[max_ports][]const u8) ![]const []const u8 {
    var t = text;
    if (t.len > 0 and t[0] == '^') return error.ExcludeListUnsupported;
    if (t.len > 0 and t[0] == '=') t = t[1..];
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, t, ',');
    while (it.next()) |item| {
        var name = item;
        if (std.mem.indexOfScalar(u8, item, ':')) |c| {
            if (!std.mem.eql(u8, item[c + 1 ..], "1")) return error.PortOtherThanOne;
            name = item[0..c];
        }
        if (name.len == 0) continue;
        if (n == max_ports) break;
        out[n] = name;
        n += 1;
    }
    if (n == 0) return error.NoHcas;
    return out[0..n];
}

/// A rail's IPv4 address and prefix length (its first RoCE v2 IPv4 GID), and that GID's index.
pub const Addr = struct { ip: u32, prefix: u8 };
pub const Probe = struct { gid: c_int, addr: Addr };

/// The RoCE v2 IPv4 GID of an RDMA device (sysfs; the Python engine's _ipv4_gid), or null.
pub fn probe(name: [:0]const u8) ?Probe {
    var gid: c_int = 0;
    var ip: u32 = 0;
    var prefix: c_int = 0;
    if (tf_roce_ipv4(name.ptr, &gid, &ip, &prefix) != 0) return null;
    return .{ .gid = gid, .addr = .{ .ip = ip, .prefix = @intCast(std.math.clamp(prefix, 0, 32)) } };
}

/// The rail pairs a rank uses to reach a peer: stripe s leaves on local rail `local[s]` for the peer's rail `remote[s]`.
pub const Route = struct {
    n: u8 = 0,
    local: [max_ports]u8 = @splat(0),
    remote: [max_ports]u8 = @splat(0),
};

pub const Plan = struct {
    /// routes[rank][peer]; n == 0: no direct path (the rank across a ring)
    routes: [max_world][max_world]Route = @splat(@splat(.{})),
    /// stripes a peer: the fewest any pair has, the same for every pair
    n_flag: u8 = 0,
};

fn subnet(a: ?Addr) ?struct { net: u32, prefix: u8 } {
    const x = a orelse return null;
    return .{ .net = if (x.prefix == 0) 0 else x.ip >> @intCast(32 - @as(u32, x.prefix)), .prefix = x.prefix };
}

/// Every rank's rails (null: unknown) -> routes (roce.py plan_routes). Rails pair when they share a subnet, each rail of
/// a peer once, in order on both sides (so the two ends of a pair list the same pairs mirrored); every pair of ranks
/// gets the fewest stripes any pair has, which must be at least one. `ring` (four ranks cabled in rank order): the
/// ranks across the ring (rank and rank + 2) need no path; every neighbour does.
pub fn planRoutes(addrs: []const [max_ports]?Addr, ring: bool) !Plan {
    const world = addrs.len;
    if (world < 2 or world > max_world) return error.Invalid;
    if (ring and world != 4) return error.RingNeedsFour;
    var plan: Plan = .{};
    var least: u8 = max_ports + 1;
    for (0..world) |r| for (0..world) |p| {
        if (p == r) continue;
        if (ring and (p + world - r) % world == 2) continue; // across the ring
        var used: [max_ports]bool = @splat(false);
        var got: Route = .{};
        for (addrs[r], 0..) |a, h| {
            const na = subnet(a) orelse continue;
            for (addrs[p], 0..) |b, rh| {
                const nb = subnet(b) orelse continue;
                if (used[rh] or na.net != nb.net or na.prefix != nb.prefix) continue;
                used[rh] = true;
                got.local[got.n] = @intCast(h);
                got.remote[got.n] = @intCast(rh);
                got.n += 1;
                break;
            }
        }
        plan.routes[r][p] = got;
        least = @min(least, got.n);
    };
    if (least == 0) return error.NoSharedSubnet;
    for (0..world) |r| for (0..world) |p| {
        if (plan.routes[r][p].n > least) plan.routes[r][p].n = least;
    };
    plan.n_flag = least;
    return plan;
}

test "routes of a triangle: each pair over its own cable's two rails" {
    const ip = struct {
        fn f(a: u8, b: u8, c: u8, d: u8) Addr {
            return .{ .ip = (@as(u32, a) << 24) | (@as(u32, b) << 16) | (@as(u32, c) << 8) | d, .prefix = 24 };
        }
    }.f;
    const tri = [_][max_ports]?Addr{
        .{ ip(10, 1, 0, 1), ip(10, 1, 0, 2), ip(10, 2, 0, 1), ip(10, 2, 0, 2) },
        .{ ip(10, 1, 0, 11), ip(10, 1, 0, 12), ip(10, 3, 0, 1), ip(10, 3, 0, 2) },
        .{ ip(10, 2, 0, 21), ip(10, 2, 0, 22), ip(10, 3, 0, 21), ip(10, 3, 0, 22) },
    };
    const plan = try planRoutes(&tri, false);
    try std.testing.expectEqual(@as(u8, 2), plan.n_flag);
    // from the Python engine's plan_routes on the same addresses
    const want = [_]struct { r: usize, p: usize, l: [2]u8, rem: [2]u8 }{
        .{ .r = 0, .p = 1, .l = .{ 0, 1 }, .rem = .{ 0, 1 } }, .{ .r = 0, .p = 2, .l = .{ 2, 3 }, .rem = .{ 0, 1 } },
        .{ .r = 1, .p = 0, .l = .{ 0, 1 }, .rem = .{ 0, 1 } }, .{ .r = 1, .p = 2, .l = .{ 2, 3 }, .rem = .{ 2, 3 } },
        .{ .r = 2, .p = 0, .l = .{ 0, 1 }, .rem = .{ 2, 3 } }, .{ .r = 2, .p = 1, .l = .{ 2, 3 }, .rem = .{ 2, 3 } },
    };
    for (want) |w| {
        const rt = plan.routes[w.r][w.p];
        try std.testing.expectEqual(@as(u8, 2), rt.n);
        try std.testing.expectEqualSlices(u8, &w.l, rt.local[0..2]);
        try std.testing.expectEqualSlices(u8, &w.rem, rt.remote[0..2]);
    }
    // the two ends of a pair list the same pairs mirrored
    for (0..3) |r| for (0..3) |p| {
        if (r == p) continue;
        const a = plan.routes[r][p];
        const b = plan.routes[p][r];
        for (0..a.n) |s| {
            try std.testing.expectEqual(a.local[s], b.remote[s]);
            try std.testing.expectEqual(a.remote[s], b.local[s]);
        }
    };
}

test "routes of a ring of four: neighbours only, the rank across gets none" {
    var ring: [4][max_ports]?Addr = undefined;
    for (0..4) |i| {
        const prev: u32 = @intCast((i + 3) % 4);
        const me: u32 = @intCast(i);
        const mk = struct {
            fn f(sub: u32, host: u32) Addr {
                return .{ .ip = (10 << 24) | (sub << 16) | host, .prefix = 24 };
            }
        }.f;
        ring[i] = .{ mk(prev, 2 + me), mk(prev, 20 + me), mk(me, 1 + me), mk(me, 10 + me) };
    }
    const plan = try planRoutes(&ring, true);
    try std.testing.expectEqual(@as(u8, 2), plan.n_flag);
    for (0..4) |r| for (0..4) |p| {
        if (p == r) continue;
        const across = (p + 4 - r) % 4 == 2;
        try std.testing.expectEqual(@as(u8, if (across) 0 else 2), plan.routes[r][p].n);
    };
    // rank 0 -> 1 leaves on its next-cable rails (2, 3) for rank 1's previous-cable rails (0, 1); 0 -> 3 the other way
    try std.testing.expectEqualSlices(u8, &.{ 2, 3 }, plan.routes[0][1].local[0..2]);
    try std.testing.expectEqualSlices(u8, &.{ 0, 1 }, plan.routes[0][1].remote[0..2]);
    try std.testing.expectEqualSlices(u8, &.{ 0, 1 }, plan.routes[0][3].local[0..2]);
    try std.testing.expectEqualSlices(u8, &.{ 2, 3 }, plan.routes[0][3].remote[0..2]);
    try std.testing.expectError(error.RingNeedsFour, planRoutes(ring[0..3], true));
}

test "routes: a pair with no shared subnet is refused, an unknown rail is skipped, stripes are the fewest of any pair" {
    const a = Addr{ .ip = 0x0a000001, .prefix = 24 };
    const b = Addr{ .ip = 0x0a000002, .prefix = 24 };
    const c = Addr{ .ip = 0x0a000003, .prefix = 24 };
    const two = [_][max_ports]?Addr{ .{ a, null, null, null }, .{ b, c, null, null } };
    const plan = try planRoutes(&two, false);
    try std.testing.expectEqual(@as(u8, 1), plan.n_flag);
    try std.testing.expectEqual(@as(u8, 1), plan.routes[1][0].n);
    const apart = [_][max_ports]?Addr{ .{ a, null, null, null }, .{ .{ .ip = 0x0a090002, .prefix = 24 }, null, null, null } };
    try std.testing.expectError(error.NoSharedSubnet, planRoutes(&apart, false));
}

test "the timeout reader compiles" {
    _ = &Roce.timedOut;
    _ = &Roce.takes;
}

test "prefetch blocks: four lines a thread, at most 64 blocks" {
    try std.testing.expectEqual(@as(usize, 0), prefetchBlocks(0));
    try std.testing.expectEqual(@as(usize, 1), prefetchBlocks(128));
    try std.testing.expectEqual(@as(usize, 16), prefetchBlocks(4 << 20));
    try std.testing.expectEqual(@as(usize, 64), prefetchBlocks(64 << 20));
}

test "NCCL_IB_HCA lists give the RoCE ports" {
    var buf: [max_ports][]const u8 = undefined;
    const two = try parseHcas("rocep1s0f1,roceP2p1s0f1", &buf);
    try std.testing.expectEqual(@as(usize, 2), two.len);
    try std.testing.expectEqualStrings("roceP2p1s0f1", two[1]);
    const one = try parseHcas("=mlx5_0:1", &buf);
    try std.testing.expectEqualStrings("mlx5_0", one[0]);
    try std.testing.expectError(error.PortOtherThanOne, parseHcas("mlx5_0:2", &buf));
    try std.testing.expectError(error.ExcludeListUnsupported, parseHcas("^mlx5_1", &buf));
    try std.testing.expectError(error.NoHcas, parseHcas("", &buf));
}

test "the region layout: receive slots, flags, send slots, control" {
    var lay: [5]u64 = undefined;
    try std.testing.expectEqual(@as(c_int, 0), tf_roce_layout(2, 512 << 10, &lay));
    try std.testing.expectEqual(@as(u64, 0), lay[0]);
    try std.testing.expectEqual(@as(u64, 4 * (512 << 10)), lay[1]);
    try std.testing.expectEqual(lay[1] + 2 * 2 * max_ports * 128, lay[2]);
    try std.testing.expectEqual(lay[2] + 2 * (512 << 10), lay[3]);
    try std.testing.expectEqual(lay[3] + 128, lay[4]);
    try std.testing.expect(tf_roce_layout(2, 1000, &lay) != 0);
    try std.testing.expect(tf_roce_layout(5, 4096, &lay) != 0);
    // the flag area holds max_ports stripes from each of the world's ranks, whatever the route
    try std.testing.expectEqual(@as(c_int, 0), tf_roce_layout(4, 4096, &lay));
    try std.testing.expectEqual(@as(u64, 4 * 2 * 4096), lay[1]);
    try std.testing.expectEqual(lay[1] + 4 * 2 * max_ports * 128, lay[2]);
}

test "the blob is what the stub and the proxy both say" {
    try std.testing.expectEqual(@as(usize, 176), Roce.blobBytes());
}
