//! The two ranks' NCCL communicator (Python tensorfold.cuda.comm's role): rank 0 makes the NCCL id and hands it to
//! rank 1 over the control link (cuda_link.zig), both join, and every exchange is an all-gather on the engine's
//! stream, so CUDA graphs can capture it. Partials are gathered in fp32 and added by the reading kernel in rank
//! order (rank 0, then rank 1) with one rounding, so both ranks hold the same bits (work/PLAN.md, TP=2 design).
//! One GPU (world 1) opens no communicator: the gathers are copies the caller skips.
//!
//! TF_FLASHNEXT_ROCE (default on; 0 off) (decode plan D1) adds the one-shot RoCE all-gather (cuda/roce.zig, b12x's RoCEnante protocol)
//! for every all-gather of up to 512 KiB a rank, on the ports NCCL_IB_HCA names with NCCL_IB_GID_INDEX; larger
//! gathers and the broadcasts stay on NCCL. Both ranks use it only when both opened it (they say so over the link),
//! a rank that cannot open it (no NCCL_IB_HCA, no RDMA device) leaves both on NCCL. TF_FLASHNEXT_L2PF (default on; 0
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

/// The control-link frame of the RoCE setup: one byte (1: opened) then the rank's connection blob.
const roce_tag: link_mod.Tag = @enumFromInt(0x524f4345);

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

    /// Join the two-rank communicator on the current CUDA context. Rank 0 sends the id first, rank 1 waits for it.
    pub fn init(gpa: Allocator, link: Link, rank: u32, world: u32) !Comm {
        if (world != 2 or rank >= world) return error.UnsupportedWorld;
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
        // on by default (decode D1, D2); "0" turns either off
        const want_roce = !envIs("TF_FLASHNEXT_ROCE", "0");
        const want_pf = !envIs("TF_FLASHNEXT_L2PF", "0");
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
        // both ranks always say whether they opened RoCE (one rank's setting alone never leaves the other waiting)
        const r = try handshake(gpa, link, rank, if (want_roce) (if (c.extra) |x| &x.driver else null) else null);
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

    /// Opens the RoCE side (none without a driver), tells the peer whether it did, and connects when both did; null
    /// when either did not.
    fn handshake(gpa: Allocator, link: Link, rank: u32, d: ?*const cuda.Driver) !?roce.Roce {
        var r: ?roce.Roce = if (d) |dd| openRoce(dd, rank) else null;
        var keep = false;
        defer if (!keep) if (r) |*x| x.deinit();
        const n = roce.Roce.blobBytes();
        const mine = try gpa.alloc(u8, 1 + n);
        defer gpa.free(mine);
        @memset(mine, 0);
        if (r) |*x| {
            mine[0] = 1;
            x.blob(mine[1..]) catch {
                mine[0] = 0;
            };
        }
        var theirs: []u8 = undefined;
        if (rank == 0) {
            try link.send(roce_tag, mine);
            theirs = try link.expect(gpa, roce_tag);
        } else {
            theirs = try link.expect(gpa, roce_tag);
            try link.send(roce_tag, mine);
        }
        defer gpa.free(theirs);
        const peer_ok = theirs.len == 1 + n and theirs[0] == 1;
        if (mine[0] != 1 or !peer_ok) {
            std.log.warn("RoCE all-gather off: this rank {s}, the peer {s}; NCCL carries every gather", .{ if (mine[0] == 1) "opened it" else if (d == null) "has it off (TF_FLASHNEXT_ROCE=0)" else "could not open it", if (peer_ok) "opened it" else "did not" });
            return null;
        }
        // both opened: from here a failure is fatal on both (the peer then times out in its first gather)
        try r.?.connect(theirs[1..]);
        try r.?.start();
        std.log.info("RoCE all-gather on {d} port(s) for gathers up to {d} KiB a rank", .{ r.?.ports, r.?.slot_bytes >> 10 });
        keep = true;
        return r;
    }

    fn openRoce(d: *const cuda.Driver, rank: u32) ?roce.Roce {
        const hca_env = std.c.getenv("NCCL_IB_HCA") orelse {
            std.log.warn("RoCE: NCCL_IB_HCA is not set", .{});
            return null;
        };
        var names: [roce.max_ports][]const u8 = undefined;
        const ports = roce.parseHcas(std.mem.span(hca_env), &names) catch |e| {
            std.log.warn("RoCE: NCCL_IB_HCA: {t}", .{e});
            return null;
        };
        const gid: c_int = if (std.c.getenv("NCCL_IB_GID_INDEX")) |g| std.fmt.parseInt(c_int, std.mem.span(g), 10) catch 3 else 3;
        // TF_FLASHNEXT_ROCE_MAX: the largest shard a rank RoCE takes (default 512 KiB; a multiple of 4096)
        const max: u64 = if (std.c.getenv("TF_FLASHNEXT_ROCE_MAX")) |m| std.fmt.parseInt(u64, std.mem.span(m), 10) catch (512 << 10) else (512 << 10);
        return roce.Roce.open(d, rank, .{ .ports = ports, .gid_index = gid, .slot_bytes = max }) catch |e| {
            std.log.warn("RoCE: {t}", .{e});
            return null;
        };
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

    /// allGather, and with TF_FLASHNEXT_L2PF=1 `next` (the next kernels' weights) prefetched into the L2 meanwhile.
    pub fn allGatherThen(c: *const Comm, send: cuda.abi.DevicePtr, recv: cuda.abi.DevicePtr, count: usize, dtype: nccl.DataType, stream: cuda.abi.Stream, next: Prefetch) !void {
        if (c.extra) |x| {
            const pf: Prefetch = if (x.prefetch) next else .{};
            const bytes = count * size(dtype);
            // RoCE takes every size it fits whose layout it can (aligned; out of place or NCCL's in-place form), else
            // NCCL. Both ranks run the same calls on buffers carved the same way (the same allocation sequence,
            // offsets and in-place choices), so the layout test answers alike on both: no rank picks the other path.
            if (x.r) |*r| if (r.takes(send, recv, bytes)) return r.allGather(send, recv, bytes, stream, pf);
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

    /// Two ranks swap `count` elements: `send` goes to the peer while the peer's arrive in `recv` (one NCCL group:
    /// a send and a receive, the same order on both ranks). The buffers must not overlap.
    pub fn exchange(c: *const Comm, send: cuda.abi.DevicePtr, recv: cuda.abi.DevicePtr, count: usize, dtype: nccl.DataType, stream: cuda.abi.Stream) !void {
        const peer: c_int = @intCast(1 - c.rank);
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
    pub fn exchangeBytes(c: *const Comm, sends: []const [2]u64, recvs: []const [2]u64, stream: cuda.abi.Stream) !void {
        const peer: c_int = @intCast(1 - c.rank);
        try c.lib.check(c.lib.api.ncclGroupStart(), "ncclGroupStart");
        errdefer _ = c.lib.api.ncclGroupEnd();
        for (sends) |sd| if (sd[1] > 0) try c.lib.check(c.lib.api.ncclSend(sd[0], sd[1], .u8, peer, c.comm, stream), "ncclSend");
        for (recvs) |rv| if (rv[1] > 0) try c.lib.check(c.lib.api.ncclRecv(rv[0], rv[1], .u8, peer, c.comm, stream), "ncclRecv");
        try c.lib.check(c.lib.api.ncclGroupEnd(), "ncclGroupEnd");
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

test "the communicator refuses worlds other than two" {
    // no GPU or NCCL needed: the check runs before the library opens
    const l: Link = .{ .fd = -1 };
    try std.testing.expectError(error.UnsupportedWorld, Comm.init(std.testing.allocator, l, 0, 1));
    try std.testing.expectError(error.UnsupportedWorld, Comm.init(std.testing.allocator, l, 2, 2));
}
