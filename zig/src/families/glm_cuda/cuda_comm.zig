//! GLM glm5_next/cuda/comm.py; adapted from TensorFold Flash Next cuda_comm.zig.
//! The two ranks' NCCL communicator (Python tensorfold.cuda.comm's role): rank 0 makes the NCCL id and hands it to
//! rank 1 over the control link (cuda_link.zig), both join, and every exchange is an all-gather on the engine's
//! stream, so CUDA graphs can capture it. Partials are gathered in fp32 and added by the reading kernel in rank
//! order (rank 0, then rank 1) with one rounding, so both ranks hold the same bits (work/PLAN.md, TP=2 design).
//! One GPU (world 1) opens no communicator: the gathers are copies the caller skips.
//!
//! TP=3 and TP=4 (work/zig-notes/r107-tpn.md): the same all-gather over `world` ranks and the same rank-order sum
//! (glm_rank_add reads the gathered [world, ...] partials rank 0 first), the Python engine's arithmetic at any N.
//! Rank 0 hands the NCCL id to every follower over the fan-out link. The one-shot RoCE all-gather runs at any of
//! the three sizes: beyond two ranks each pair's rails are paired by subnet (cuda/roce.zig planRoutes, the Python
//! engine's plan_routes), and on a ring of four cabled to the neighbours only (TF_ROCE_RING=1) the neighbours' proxies
//! relay the stripes to the rank across (recipe patches 0113/0114). A ring of Sparks wants NCCL_ALGO=Ring and the
//! per-rank NCCL_GRAPH_FILE the launch script sets for NCCL's own gathers; nothing here depends on them.
//!
//! TF_GLM_ROCE (default on; 0 off) (decode plan D1) adds the one-shot RoCE all-gather (cuda/roce.zig, b12x's RoCEnante protocol)
//! for every all-gather of up to 512 KiB a rank, on the ports TF_ROCE_HCA / NCCL_IB_HCA name with NCCL_IB_GID_INDEX;
//! larger gathers and the broadcasts stay on NCCL. The ranks use it only when every one opened and connected it (they
//! say so over the link), a rank that cannot (no NCCL_IB_HCA, no RDMA device, no route) leaves all on NCCL. The choice
//! per gather is by its byte count alone, so every rank makes it alike (a buffer RoCE cannot take is an error, never
//! a silent switch to NCCL on one rank). TF_GLM_L2PF (default on; 0
//! off) (decode plan D2) prefetches the weights a gather's
//! caller names (`allGatherThen`) into the L2 while the gather waits: extra blocks of the RoCE launch, or one small
//! launch before an NCCL gather. Neither changes a byte of any result.
const std = @import("std");
const cuda = @import("cuda");
const link_mod = @import("cuda_link.zig");
const Link = link_mod.Link;
const nccl = cuda.nccl;
const roce = cuda.roce;
const Allocator = std.mem.Allocator;

pub const Prefetch = roce.Prefetch;

/// The most ranks: the Python engine's tensor-parallel sizes are 2, 3 and 4.
pub const max_world: u32 = 4;

/// The control-link frame of the RoCE setup: the rank's row (1 byte opened, its rail count, the slot size, each rail's
/// IPv4 and prefix, the connection blob), then the verdict of the connection. See `Comm.handshake`.
const row_head: usize = 8 + 8 * roce.max_ports;
const roce_tag: link_mod.Tag = @fromBackingInt(@intCast(0x524f4345));

/// The RoCE all-gather, the L2 prefetch and the driver they were opened with (heap: Comm is copied by value).
const Extra = struct { driver: cuda.Driver, r: ?roce.Roce = null, l2: ?roce.L2 = null, prefetch: bool = false };

fn envIs(name: [:0]const u8, value: []const u8) bool {
    return if (std.c.getenv(name)) |v| std.mem.eql(u8, std.mem.span(v), value) else false;
}

pub const Comm = struct {
    lib: nccl.Library,
    comm: nccl.Comm,
    rank: u32,
    world: u32,
    extra: ?*Extra = null,

    /// Join the communicator on the current CUDA context. Rank 0 sends the id first (to every follower), the others
    /// wait for it. `link` is rank 0's fan-out to ranks 1..world-1, or a follower's one link to rank 0.
    pub fn init(gpa: Allocator, link: Link, rank: u32, world: u32) !Comm {
        if (world < 2 or world > max_world or rank >= world) return error.UnsupportedWorld;
        var lib = nccl.Library.open() catch |e| {
            std.log.err("cannot open libnccl.so.2 ({s}): the image's NCCL is needed for two ranks", .{@errorName(e)});
            return e;
        };
        errdefer lib.close();
        var id: nccl.UniqueId = undefined;
        if (rank == 0) {
            try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
            try link.send(.nccl_id, &id.internal);
        } else {
            const bytes = try link.expect(gpa, .nccl_id);
            defer gpa.free(bytes);
            if (bytes.len != id.internal.len) return error.BadNcclId;
            @memcpy(&id.internal, bytes);
        }
        var comm: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        var c: Comm = .{ .lib = lib, .comm = comm, .rank = rank, .world = world };
        errdefer _ = lib.api.ncclCommDestroy(comm);
        errdefer if (c.extra) |x| freeExtra(x);
        // on by default (decode D1, D2); "0" turns either off. RoCE is the two-rank protocol only (see above).
        const want_roce = !envIs("TF_GLM_ROCE", "0");
        const want_pf = !envIs("TF_GLM_L2PF", "0");
        var driver: ?cuda.Driver = null;
        if (want_roce or want_pf) driver = cuda.Driver.open() catch |e| blk: {
            std.log.warn("RoCE / L2 prefetch: no CUDA driver handle ({t}); NCCL alone", .{e});
            break :blk null;
        };
        if (driver) |d| {
            const x = try std.heap.c_allocator.create(Extra);
            x.* = .{ .driver = d };
            c.extra = x;
        }
        // both ranks always say whether they opened RoCE (one rank's setting alone never leaves the other waiting);
        // beyond two ranks nobody does, so there is nothing to say
        const r = try handshake(gpa, link, rank, world, if (want_roce) (if (c.extra) |x| &x.driver else null) else null);
        if (c.extra) |x| {
            x.r = r;
            if (want_pf) {
                x.l2 = roce.L2.init(&x.driver) catch |e| blk: {
                    std.log.warn("L2 prefetch off: {t}", .{e});
                    break :blk null;
                };
                x.prefetch = x.l2 != null;
            }
        } else if (r) |rr| {
            var keep = rr;
            keep.deinit();
        }
        return c;
    }

    /// Opens the RoCE side (none without a driver), swaps every rank's description and connection blob over the control
    /// link (rank 0 gathers them and sends all to all), plans the routes (beyond two ranks: rails paired by subnet;
    /// TF_ROCE_RING=1: a ring of four cabled to the neighbours, who relay to the rank across), connects and starts the
    /// proxy, and agrees that every rank did. Null when any rank did not (every rank then keeps NCCL for every
    /// gather: the choice is the same on all of them). Every rank takes this path whether or not it opened RoCE.
    fn handshake(gpa: Allocator, link: Link, rank: u32, world: u32, d: ?*const cuda.Driver) !?roce.Roce {
        var found: [roce.max_ports]?roce.Addr = @splat(null);
        var r: ?roce.Roce = if (d) |dd| openRoce(dd, rank, world, &found) else null;
        var keep = false;
        defer if (!keep) if (r) |*x| x.deinit();
        const n = roce.Roce.blobBytes();
        const row_bytes = row_head + n;
        const mine = try gpa.alloc(u8, row_bytes);
        defer gpa.free(mine);
        @memset(mine, 0);
        if (r) |*x| {
            mine[0] = 1;
            mine[1] = @intCast(x.ports);
            std.mem.writeInt(u32, mine[4..8], @intCast(x.slot_bytes >> 12), .little);
            for (found, 0..) |f, i| {
                std.mem.writeInt(u32, mine[8 + 8 * i ..][0..4], if (f) |a| a.ip else 0, .little);
                std.mem.writeInt(i32, mine[12 + 8 * i ..][0..4], if (f) |a| @as(i32, a.prefix) else -1, .little);
            }
            x.blob(mine[row_head..]) catch {
                mine[0] = 0;
            };
        }
        const all = try gatherToAll(gpa, link, rank, world, mine);
        defer gpa.free(all);
        const rows = all[1..];
        const peers_ok = all[0] == 1;
        if (mine[0] != 1 or !peers_ok) {
            std.log.warn("RoCE all-gather off: this rank {s}; some rank could not open it; NCCL carries every gather", .{if (mine[0] == 1) "opened it" else if (d == null) "has it off (TF_GLM_ROCE=0)" else "could not open it"});
            return null;
        }
        const ring = envIs("TF_ROCE_RING", "1");
        // the same rows on every rank give the same answer below: nobody needs to tell the others
        var addrs: [roce.max_world][roce.max_ports]?roce.Addr = undefined;
        for (0..world) |p| {
            const row = rows[p * row_bytes ..][0..row_bytes];
            if (row[1] != rows[1] and world == 2) {
                std.log.warn("RoCE all-gather off: the ranks name different numbers of rails; NCCL carries every gather", .{});
                return null;
            }
            if (!std.mem.eql(u8, row[4..8], rows[4..8])) {
                std.log.warn("RoCE all-gather off: the ranks' slot sizes differ (TF_ROCE_MAX_KB); NCCL carries every gather", .{});
                return null;
            }
            for (0..roce.max_ports) |i| {
                const prefix = std.mem.readInt(i32, row[12 + 8 * i ..][0..4], .little);
                addrs[p][i] = if (prefix < 0 or i >= row[1]) null else .{ .ip = std.mem.readInt(u32, row[8 + 8 * i ..][0..4], .little), .prefix = @intCast(prefix) };
            }
        }
        const x = &r.?;
        if (world > 2) {
            const plan = roce.planRoutes(addrs[0..world], ring) catch |e| {
                std.log.warn("RoCE all-gather off: routes: {t} (every pair of ranks needs rails on one subnet, a ring of four TF_ROCE_RING=1 with its neighbours'); NCCL carries every gather", .{e});
                return null;
            };
            for (0..world) |p| {
                const rt = plan.routes[rank][p];
                if (p == rank or rt.n == 0) continue;
                var local: [roce.max_ports]c_int = undefined;
                var remote: [roce.max_ports]c_int = undefined;
                for (0..rt.n) |i| {
                    local[i] = rt.local[i];
                    remote[i] = rt.remote[i];
                }
                var ok = true;
                x.setRoutes(@intCast(p), local[0..rt.n], remote[0..rt.n]) catch {
                    ok = false;
                };
                if (!ok) return try verdict(gpa, link, rank, world, null, false);
            }
            if (ring) x.setRing() catch return try verdict(gpa, link, rank, world, null, false);
        } else if (ring) {
            std.log.warn("TF_ROCE_RING=1 needs four ranks, not {d}; ignored", .{world});
        }
        var blobs = try gpa.alloc(u8, world * n);
        defer gpa.free(blobs);
        for (0..world) |p| @memcpy(blobs[p * n ..][0..n], rows[p * row_bytes + row_head ..][0..n]);
        var ok = true;
        x.connectAll(blobs) catch {
            ok = false;
        };
        if (ok) x.start() catch {
            ok = false;
        };
        const res = try verdict(gpa, link, rank, world, x, ok);
        if (res != null) {
            keep = true;
            std.log.info("RoCE all-gather on {d} rail(s), {d} stripe(s) a peer{s} for gathers up to {d} KiB a rank", .{ x.ports, x.n_flag, if (world == 4 and ring) ", ring with relays" else "", x.slot_bytes >> 10 });
        }
        return res;
    }

    /// The last agreement: every rank says whether it connected and started; all go on only when all did.
    fn verdict(gpa: Allocator, link: Link, rank: u32, world: u32, r: ?*roce.Roce, ok: bool) !?roce.Roce {
        const all = try gatherToAll(gpa, link, rank, world, &.{@intFromBool(ok)});
        defer gpa.free(all);
        if (all[0] != 1 or !ok) {
            std.log.warn("RoCE all-gather off: some rank could not connect; NCCL carries every gather", .{});
            return null;
        }
        return r.?.*;
    }

    /// Every rank's equal-length `mine`, in rank order, to every rank: rank 0 collects the followers' frames and sends
    /// one frame back, [1 byte: every rank sent a first byte of 1] ++ the rows in rank order.
    fn gatherToAll(gpa: Allocator, link: Link, rank: u32, world: u32, mine: []const u8) ![]u8 {
        if (rank == 0) {
            const out = try gpa.alloc(u8, 1 + world * mine.len);
            errdefer gpa.free(out);
            @memcpy(out[1..][0..mine.len], mine);
            var ok = mine[0] == 1;
            for (0..world - 1) |i| {
                const row = try link.at(i).expect(gpa, roce_tag);
                defer gpa.free(row);
                if (row.len != mine.len) return error.BadRoceRow;
                @memcpy(out[1 + (i + 1) * mine.len ..][0..mine.len], row);
                if (row[0] != 1) ok = false;
            }
            out[0] = @intFromBool(ok);
            try link.send(roce_tag, out);
            return out;
        }
        try link.send(roce_tag, mine);
        const out = try link.expect(gpa, roce_tag);
        if (out.len != 1 + world * mine.len) {
            gpa.free(out);
            return error.BadRoceRow;
        }
        return out;
    }

    /// This rank's rails: TF_ROCE_HCA, else NCCL_IB_HCA (at most two rails at two ranks, four beyond); the GID of every
    /// rail is TF_ROCE_GID_INDEX / NCCL_IB_GID_INDEX (3) at two ranks or when set, else each rail's own RoCE v2 IPv4
    /// GID. Beyond two ranks every rail is probed; its address and prefix go into `found` for the routes unless a fixed
    /// index names another GID (railAddr).
    fn openRoce(d: *const cuda.Driver, rank: u32, world: u32, found: *[roce.max_ports]?roce.Addr) ?roce.Roce {
        const hca_env = std.c.getenv("TF_ROCE_HCA") orelse std.c.getenv("NCCL_IB_HCA") orelse {
            std.log.warn("RoCE: NCCL_IB_HCA is not set", .{});
            return null;
        };
        var names: [roce.max_ports][]const u8 = undefined;
        const listed = roce.parseHcas(std.mem.span(hca_env), &names) catch |e| {
            std.log.warn("RoCE: NCCL_IB_HCA: {t}", .{e});
            return null;
        };
        const ports = if (world == 2) listed[0..@min(listed.len, 2)] else listed;
        const fixed: ?c_int = blk: {
            const g = std.c.getenv("TF_ROCE_GID_INDEX") orelse std.c.getenv("NCCL_IB_GID_INDEX") orelse break :blk null;
            break :blk std.fmt.parseInt(c_int, std.mem.span(g), 10) catch 3;
        };
        var gids: [roce.max_ports]c_int = @splat(fixed orelse 3);
        // beyond two ranks the routes pair rails by subnet, so every rail's address is probed even when the GID index
        // is fixed (a fixed index alone left `found` empty and every route NoSharedSubnet); the fixed index still wins
        if (world > 2) for (ports, 0..) |p, i| {
            var z: [256:0]u8 = undefined;
            if (p.len >= z.len) continue;
            @memcpy(z[0..p.len], p);
            z[p.len] = 0;
            if (roce.probe(z[0..p.len :0])) |pr| {
                found[i] = railAddr(fixed, pr);
                if (fixed) |f| {
                    if (found[i] == null) std.log.warn("RoCE: {s}: fixed GID index {d}, but its RoCE v2 IPv4 GID is at {d}: rail left out of the routes", .{ p, f, pr.gid });
                } else gids[i] = pr.gid;
            }
        };
        const tc: c_int = if (std.c.getenv("NCCL_IB_TC")) |t| std.fmt.parseInt(c_int, std.mem.span(t), 0) catch 0 else 0;
        // TF_ROCE_MAX_KB: the largest shard a rank RoCE takes (default 512 KiB; a multiple of 4096)
        const max: u64 = if (std.c.getenv("TF_ROCE_MAX_KB")) |m| std.fmt.parseInt(u64, std.mem.span(m), 10) catch 512 else 512;
        return roce.Roce.openRanks(d, rank, world, .{ .ports = ports, .gids = gids[0..ports.len], .traffic_class = tc, .slot_bytes = max * 1024, .timeout_s = roceTimeoutS() }) catch |e| {
            std.log.warn("RoCE: {t}", .{e});
            return null;
        };
    }

    /// TF_ROCE_WAIT_S (1 to 3600 s, the Python engine's name): how long a GPU wait for a peer's shard may take.
    fn roceTimeoutS() u64 {
        const v = std.c.getenv("TF_ROCE_WAIT_S") orelse return 600;
        const n = std.fmt.parseInt(u64, std.mem.span(v), 10) catch return 600;
        return if (n >= 1 and n <= 3600) n else 600;
    }

    pub fn deinit(c: *Comm) void {
        _ = c.lib.api.ncclCommDestroy(c.comm);
        c.lib.close();
        if (c.extra) |x| freeExtra(x);
        c.extra = null;
    }

    fn freeExtra(x: *Extra) void {
        if (x.r) |*r| r.deinit();
        if (x.l2) |*l| l.deinit();
        x.driver.close();
        std.heap.c_allocator.destroy(x);
    }

    /// True when small gathers go over RoCE.
    pub fn roceOn(c: *const Comm) bool {
        return if (c.extra) |x| x.r != null else false;
    }

    /// The RoCE side's proxy failure or a kernel's timed-out wait, if any, as text in `buf` (the stream traps on its own).
    pub fn roceFailure(c: *const Comm, buf: []u8) ?[]const u8 {
        const x = c.extra orelse return null;
        const r = &(x.r orelse return null);
        if (r.failure()) |f| return f;
        if (r.timedOut()) |t| return std.fmt.bufPrint(buf, "RoCE gather {d} waited past its timeout for rank {d}", .{ t.seq, t.peer }) catch "RoCE gather timed out";
        return null;
    }

    /// The peer is gone: end the communicator's pending collectives so a waiting stream returns (any thread).
    pub fn abort(c: *const Comm) void {
        _ = c.lib.api.ncclCommAbort(c.comm);
    }

    /// `count` elements of `dtype` from every rank into `recv` [world * count], rank 0's first, on `stream`.
    pub fn allGather(c: *const Comm, send: cuda.abi.DevicePtr, recv: cuda.abi.DevicePtr, count: usize, dtype: nccl.DataType, stream: cuda.abi.Stream) !void {
        return c.allGatherThen(send, recv, count, dtype, stream, .{});
    }

    /// allGather, and with TF_GLM_L2PF=1 `next` (the next kernels' weights) prefetched into the L2 meanwhile.
    pub fn allGatherThen(c: *const Comm, send: cuda.abi.DevicePtr, recv: cuda.abi.DevicePtr, count: usize, dtype: nccl.DataType, stream: cuda.abi.Stream, next: Prefetch) !void {
        if (c.extra) |x| {
            const pf: Prefetch = if (x.prefetch) next else .{};
            const bytes = count * size(dtype);
            // RoCE takes every size it fits whose layout it can (aligned; out of place or NCCL's in-place form), else
            // NCCL. Both ranks run the same calls on buffers carved the same way (the same allocation sequence,
            // offsets and in-place choices), so the layout test answers alike on both: no rank picks the other path.
            if (x.r) |*r| if (r.fits(bytes)) {
                if (!r.takes(send, recv, bytes)) return error.RoceBuffers;
                return r.allGather(send, recv, bytes, stream, pf);
            };
            if (x.l2) |*l| if (pf.bytes > 0) try l.run(pf, stream);
        }
        try c.lib.check(c.lib.api.ncclAllGather(send, recv, count, dtype, c.comm, stream), "ncclAllGather");
    }

    /// NCCL's in-place all-gather (`send` = `recv` + rank * count elements): the prompt glue's row halves. Never RoCE,
    /// which takes no overlapping buffers; NCCL on both ranks whatever the size (the same choice on both).
    pub fn allGatherInPlace(c: *const Comm, send: cuda.abi.DevicePtr, recv: cuda.abi.DevicePtr, count: usize, dtype: nccl.DataType, stream: cuda.abi.Stream) !void {
        try c.lib.check(c.lib.api.ncclAllGather(send, recv, count, dtype, c.comm, stream), "ncclAllGather");
    }

    /// `bytes` bytes of `buf` from `root` to every rank, in place, on `stream` (the mirror's per-round records).
    pub fn broadcast(c: *const Comm, buf: cuda.abi.DevicePtr, bytes: usize, root: u32, stream: cuda.abi.Stream) !void {
        try c.lib.check(c.lib.api.ncclBroadcast(buf, buf, bytes, .u8, @intCast(root), c.comm, stream), "ncclBroadcast");
    }

    /// Two ranks swap `count` elements: `send` goes to `peer_rank` while its elements arrive in `recv` (one NCCL group:
    /// a send and a receive, the same order on both ranks). The buffers must not overlap.
    pub fn exchange(c: *const Comm, peer_rank: u32, send: cuda.abi.DevicePtr, recv: cuda.abi.DevicePtr, count: usize, dtype: nccl.DataType, stream: cuda.abi.Stream) !void {
        const peer: c_int = @intCast(try c.peerOf(peer_rank));
        try c.lib.check(c.lib.api.ncclGroupStart(), "ncclGroupStart");
        c.lib.check(c.lib.api.ncclSend(send, count, dtype, peer, c.comm, stream), "ncclSend") catch |e| {
            _ = c.lib.api.ncclGroupEnd();
            return e;
        };
        c.lib.check(c.lib.api.ncclRecv(recv, count, dtype, peer, c.comm, stream), "ncclRecv") catch |e| {
            _ = c.lib.api.ncclGroupEnd();
            return e;
        };
        try c.lib.check(c.lib.api.ncclGroupEnd(), "ncclGroupEnd");
    }

    /// One NCCL group of point-to-point pairs with the peer: each `sends[i]` (bytes) goes out, each `recvs[i]`
    /// (bytes) comes in, in this order on both ranks (the peer's sends match this rank's receives).
    pub fn exchangeBytes(c: *const Comm, peer_rank: u32, sends: []const [2]u64, recvs: []const [2]u64, stream: cuda.abi.Stream) !void {
        const peer: c_int = @intCast(try c.peerOf(peer_rank));
        try c.lib.check(c.lib.api.ncclGroupStart(), "ncclGroupStart");
        errdefer _ = c.lib.api.ncclGroupEnd();
        for (sends) |sd| if (sd[1] > 0) try c.lib.check(c.lib.api.ncclSend(sd[0], sd[1], .u8, peer, c.comm, stream), "ncclSend");
        for (recvs) |rv| if (rv[1] > 0) try c.lib.check(c.lib.api.ncclRecv(rv[0], rv[1], .u8, peer, c.comm, stream), "ncclRecv");
        try c.lib.check(c.lib.api.ncclGroupEnd(), "ncclGroupEnd");
    }

    /// `peer_rank` when it is another rank of this communicator.
    fn peerOf(c: *const Comm, peer_rank: u32) !u32 {
        if (peer_rank >= c.world or peer_rank == c.rank) return error.BadPeer;
        return peer_rank;
    }

    /// The fp32 partials of a [rows, width] block into [world, rows, width] (the token mixer's and the MoE's).
    pub fn gatherPartials(c: *const Comm, send: cuda.abi.DevicePtr, recv: cuda.abi.DevicePtr, rows: usize, width: usize, stream: cuda.abi.Stream) !void {
        try c.allGather(send, recv, rows * width, .f32, stream);
    }

    /// gatherPartials with `next` prefetched meanwhile (allGatherThen).
    pub fn gatherPartialsThen(c: *const Comm, send: cuda.abi.DevicePtr, recv: cuda.abi.DevicePtr, rows: usize, width: usize, stream: cuda.abi.Stream, next: Prefetch) !void {
        try c.allGatherThen(send, recv, rows * width, .f32, stream, next);
    }
};

fn size(t: nccl.DataType) usize {
    return switch (t) {
        .i8, .u8 => 1,
        .f16, .bf16 => 2,
        .i32, .u32, .f32 => 4,
        .i64, .u64, .f64 => 8,
    };
}

/// The address a rail gives the routes beyond two ranks (openRoce): its probed RoCE v2 IPv4 address, unless a fixed GID
/// index names another GID. Then it is null: the rail is left out rather than described by another GID's address, and
/// planRoutes skips it (if a required pair of ranks is left without a shared subnet, there is no complete route plan
/// and every rank keeps NCCL).
fn railAddr(fixed: ?c_int, pr: roce.Probe) ?roce.Addr {
    if (fixed) |f| if (f != pr.gid) return null;
    return pr.addr;
}

test "the communicator refuses worlds beyond four and ranks outside the world" {
    // no GPU or NCCL needed: the check runs before the library opens
    const l: Link = .{ .fd = -1 };
    try std.testing.expectError(error.UnsupportedWorld, Comm.init(std.testing.allocator, l, 0, 1));
    try std.testing.expectError(error.UnsupportedWorld, Comm.init(std.testing.allocator, l, 2, 2));
    try std.testing.expectError(error.UnsupportedWorld, Comm.init(std.testing.allocator, l, 4, 4));
    try std.testing.expectError(error.UnsupportedWorld, Comm.init(std.testing.allocator, l, 0, 5));
}

test "an exchange names another rank of the world" {
    var c: Comm = undefined;
    c.rank = 1;
    c.world = 3;
    try std.testing.expectEqual(@as(u32, 2), try c.peerOf(2));
    try std.testing.expectEqual(@as(u32, 0), try c.peerOf(0));
    try std.testing.expectError(error.BadPeer, c.peerOf(1));
    try std.testing.expectError(error.BadPeer, c.peerOf(3));
}

fn gatherBody(l: Link, gpa: Allocator, rank: u32, world: u32, bad_rank: u32, out: *?anyerror, seen: *[5 * 3]u8) void {
    var mine = [_]u8{ @intFromBool(rank != bad_rank), @intCast(rank), 7 };
    const all = Comm.gatherToAll(gpa, l, rank, world, &mine) catch |e| {
        out.* = e;
        return;
    };
    defer gpa.free(all);
    if (all.len != 1 + world * 3) {
        out.* = error.WrongLength;
        return;
    }
    seen[0] = all[0];
    @memcpy(seen[1 .. 1 + world * 3], all[1..]);
}

fn runGather(world: u32, bad_rank: u32, errs: []?anyerror, seen: [][5 * 3]u8) !void {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const server = try link_mod.Server.open(try std.Io.net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    const Peer = struct {
        fn run(io2: std.Io, p: u16, rank: u32, w: u32, bad: u32, out: *?anyerror, sn: *[5 * 3]u8) void {
            const l = Link.join(io2, std.Io.net.IpAddress.parse("127.0.0.1", p) catch return, 5000) catch |e| {
                out.* = e;
                return;
            };
            defer l.close();
            l.hello(rank) catch |e| {
                out.* = e;
                return;
            };
            gatherBody(l, std.testing.allocator, rank, w, bad, out, sn);
        }
    };
    var threads: [3]std.Thread = undefined;
    for (1..world) |r| threads[r - 1] = try std.Thread.spawn(.{}, Peer.run, .{ io, server.port(), @as(u32, @intCast(r)), world, bad_rank, &errs[r], &seen[r] });
    const l = try server.acceptRanks(gpa, world, 5000);
    defer l.close();
    gatherBody(l, gpa, 0, world, bad_rank, &errs[0], &seen[0]);
    for (threads[0 .. world - 1]) |t| t.join();
}

test "the RoCE setup's gather gives every rank every rank's row, in rank order, and the all-ok byte" {
    for ([_]u32{ 2, 3, 4 }) |world| {
        var errs: [4]?anyerror = @splat(null);
        var seen: [4][5 * 3]u8 = undefined;
        try runGather(world, 99, &errs, &seen);
        for (0..world) |r| {
            try std.testing.expect(errs[r] == null);
            try std.testing.expectEqual(@as(u8, 1), seen[r][0]);
            for (0..world) |p| try std.testing.expectEqualSlices(u8, &.{ 1, @intCast(p), 7 }, seen[r][1 + p * 3 ..][0..3]);
        }
    }
}

test "one rank that could not open RoCE turns it off for all" {
    var errs: [4]?anyerror = @splat(null);
    var seen: [4][5 * 3]u8 = undefined;
    try runGather(4, 2, &errs, &seen);
    for (0..4) |r| {
        try std.testing.expect(errs[r] == null);
        try std.testing.expectEqual(@as(u8, 0), seen[r][0]);
    }
}

test "a rail's route address: the probed one, unless a fixed GID index names another GID" {
    const pr: roce.Probe = .{ .gid = 3, .addr = .{ .ip = 0x0a010001, .prefix = 24 } };
    try std.testing.expectEqual(pr.addr, railAddr(null, pr).?);
    try std.testing.expectEqual(pr.addr, railAddr(3, pr).?);
    try std.testing.expect(railAddr(5, pr) == null);
    try std.testing.expect(railAddr(0, pr) == null);
    // without a fixed index the probed GID is the rail's own, whatever its index
    try std.testing.expectEqual(pr.addr, railAddr(null, .{ .gid = 5, .addr = pr.addr }).?);
}

test "a fixed GID index on a triangle: a mismatched rail is left out; the plan holds while every pair keeps a rail" {
    // rank 0 <-> 1 over one cable with both port halves addressed (two rails), rank 0 <-> 2 and 1 <-> 2 one rail each
    // (3/3/2 rails); every probe at GID 3 unless a case moves one, and TF_ROCE_GID_INDEX fixed at 3
    const mk = struct {
        fn f(net: u8, host: u8) roce.Addr {
            return .{ .ip = (@as(u32, 10) << 24) | (@as(u32, net) << 16) | host, .prefix = 24 };
        }
    }.f;
    const rails = [3][3]?roce.Addr{
        .{ mk(1, 1), mk(11, 1), mk(2, 1) },
        .{ mk(1, 2), mk(11, 2), mk(3, 1) },
        .{ mk(2, 2), mk(3, 2), null },
    };
    const Case = struct { rank: usize = 9, rail: usize = 9 }; // the probe found at GID 5 instead
    const rows = struct {
        fn f(r: *const [3][3]?roce.Addr, moved: Case) [3][roce.max_ports]?roce.Addr {
            var out: [3][roce.max_ports]?roce.Addr = @splat(@splat(null));
            for (0..3) |k| for (0..3) |h| {
                const a = r[k][h] orelse continue;
                out[k][h] = railAddr(3, .{ .gid = if (k == moved.rank and h == moved.rail) 5 else 3, .addr = a });
            };
            return out;
        }
    }.f;
    // every rail at the fixed index: every pair has a route, one stripe a peer (the fewest any pair has)
    const all = try roce.planRoutes(&rows(&rails, .{}), false);
    try std.testing.expectEqual(@as(u8, 1), all.n_flag);
    // rank 0's second rail toward rank 1 at another GID: left out (null, not another GID's address); 0 <-> 1 keeps its
    // first rail, and rank 0's rail toward rank 2 keeps its index (2), so nothing is renumbered
    const one = rows(&rails, .{ .rank = 0, .rail = 1 });
    try std.testing.expect(one[0][1] == null);
    const plan = try roce.planRoutes(&one, false);
    try std.testing.expectEqual(@as(u8, 1), plan.n_flag);
    try std.testing.expectEqual(@as(u8, 0), plan.routes[0][1].local[0]);
    try std.testing.expectEqual(@as(u8, 0), plan.routes[0][1].remote[0]);
    try std.testing.expectEqual(@as(u8, 2), plan.routes[0][2].local[0]);
    try std.testing.expectEqual(@as(u8, 0), plan.routes[0][2].remote[0]);
    // rank 2's only rail toward rank 0 at another GID: that pair has no shared subnet, so there is no complete plan
    // (every rank keeps NCCL)
    try std.testing.expectError(error.NoSharedSubnet, roce.planRoutes(&rows(&rails, .{ .rank = 2, .rail = 0 }), false));
}
