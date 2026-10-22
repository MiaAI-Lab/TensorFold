//! Flash Next's two-rank transport on two GPUs, before the engine needs it (work/PLAN.md, TP=2 design):
//! `tf-flashnext-comm RANK MASTER PORT [ITERS] [WARMUP]`, one process per box, rank 0 listening on MASTER:PORT.
//!
//! It opens the CUDA context, the control link (cuda_link.zig) and the NCCL communicator over it (cuda_comm.zig), then
//! for every gather the engine makes per layer and per round — the fp32 partials of decode windows [1..16, 2560], a
//! prompt chunk's [2048, 2560], and the head's gathered candidates [16, 65] — checks that the all-gather is bit-exact
//! (each rank's words, NaN payloads, signed zeros and denormals included, land in rank order, and both ranks hold the
//! same bytes: their sha256 is compared over the link) and times it: per call on the host (enqueue to stream done)
//! and on the GPU (events around the call), p50/p90 over ITERS calls after WARMUP, and the pipelined rate (ITERS calls
//! back to back, one synchronize). busbw is the bytes each rank sends (and receives) per call over the time, which is
//! nccl-tests' bus bandwidth for an all-gather on two ranks. It also times one control-link frame round trip.
//! Lines: PASS / FAIL / RESULT / INFO; exit 1 on any failure.
const std = @import("std");
const cuda = @import("cuda");
const flashnext = @import("flashnext");
const link_mod = flashnext.link;
const Comm = flashnext.comm.Comm;
const net = std.Io.net;
const Allocator = std.mem.Allocator;

const hidden: usize = 2560;

const Case = struct { name: []const u8, rows: usize, width: usize };

const cases = [_]Case{
    .{ .name = "decode-1", .rows = 1, .width = hidden },
    .{ .name = "decode-2", .rows = 2, .width = hidden },
    .{ .name = "decode-4", .rows = 4, .width = hidden },
    .{ .name = "decode-8", .rows = 8, .width = hidden },
    .{ .name = "decode-16", .rows = 16, .width = hidden },
    .{ .name = "decode-28", .rows = 28, .width = hidden },
    .{ .name = "decode-48", .rows = 48, .width = hidden },
    .{ .name = "prompt-2048", .rows = 2048, .width = hidden },
    .{ .name = "candidates-16", .rows = 16, .width = 65 },
    // not a multiple of 16 bytes (the RoCE path's 4-byte form, TF_FLASHNEXT_ROCE=1), and the 1-int agreement
    .{ .name = "candidates-7", .rows = 7, .width = 65 },
    .{ .name = "agree-1", .rows = 1, .width = 1 },
};

/// Seeds of the exactness trials (each a fresh fill of both ranks' partials).
const seeds = [_]u64{ 0x5eed_0001, 0x5eed_0002, 0x5eed_0003 };

const usage = "usage: tf-flashnext-comm RANK MASTER PORT [ITERS=1000] [WARMUP=100]\n";

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const rank = std.fmt.parseInt(u32, args[1], 10) catch {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    const port = std.fmt.parseInt(u16, args[3], 10) catch {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    const iters = if (args.len > 4) try std.fmt.parseInt(usize, args[4], 10) else 1000;
    const warmup = if (args.len > 5) try std.fmt.parseInt(usize, args[5], 10) else 100;
    if (rank > 1 or iters == 0) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const master = net.IpAddress.parse(args[2], port) catch {
        std.debug.print("MASTER is an IPv4 or IPv6 address, not {s}\n", .{args[2]});
        return 2;
    };
    run(init.gpa, init.io, rank, master, iters, warmup) catch |e| {
        std.debug.print("FAIL rank {d}: {t}\n", .{ rank, e });
        return 1;
    };
    return 0;
}

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn run(gpa: Allocator, io: std.Io, rank: u32, master: net.IpAddress, iters: usize, warmup: usize) !void {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var name_buf: [256]u8 = undefined;
    std.debug.print("INFO rank {d}: {s} sm_{d}, driver {d}\n", .{ rank, try ctx.name(&name_buf), try ctx.capability(), try driver.version() });

    // the control link: rank 0 listens on MASTER:PORT, rank 1 joins (retrying while rank 0 starts)
    var server: ?link_mod.Server = null;
    defer if (server) |s| s.close();
    const t_link = now(io);
    const link = if (rank == 0) blk: {
        server = try link_mod.Server.open(master);
        break :blk try server.?.accept(180_000);
    } else try link_mod.Link.join(io, master, 180_000);
    defer link.close();
    std.debug.print("INFO rank {d}: control link up in {d:.1} ms\n", .{ rank, ms(now(io) - t_link) });
    try link.agree(gpa, rank, "tf-flashnext-comm v1");

    try linkRoundTrip(gpa, io, link, rank, "first", 16, iters, warmup);
    try linkRoundTrip(gpa, io, link, rank, "first", 4096, iters, warmup);

    const t_comm = now(io);
    var comm = try Comm.init(gpa, link, rank, 2);
    defer comm.deinit();
    std.debug.print("INFO rank {d}: NCCL {d}, communicator world 2 in {d:.1} ms, small gathers over {s}\n", .{ rank, try comm.lib.version(), ms(now(io) - t_comm), if (comm.roceOn()) "RoCE (one-shot)" else "NCCL" });
    if (std.c.getenv("TF_FLASHNEXT_ROCE")) |v| if (std.mem.eql(u8, std.mem.span(v), "1") and !comm.roceOn()) {
        std.debug.print("FAIL rank {d}: TF_FLASHNEXT_ROCE=1 but the RoCE all-gather did not open (a build without kernels?)\n", .{rank});
        return error.TestFailed;
    };

    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var failed: usize = 0;
    for (cases) |c| {
        gatherCase(gpa, io, &driver, &comm, link, stream, rank, c, iters, warmup) catch |e| switch (e) {
            error.TestFailed, error.RanksDisagree => failed += 1,
            else => return e,
        };
    }
    // back-to-back gathers whose payload and size change every call (a stale slot or sequence would show)
    sequenceCheck(gpa, &driver, &comm, stream, rank) catch |e| switch (e) {
        error.TestFailed => failed += 1,
        else => return e,
    };
    // the link again, after the gathers (the first trips run on idle boxes)
    try linkRoundTrip(gpa, io, link, rank, "again", 16, iters, warmup);
    try linkRoundTrip(gpa, io, link, rank, "again", 4096, iters, warmup);
    try spinRoundTrip(io, link, rank, 16, iters, warmup);
    // the end: rank 0 says stop, rank 1 acknowledges, so neither closes while the other still reads
    if (rank == 0) {
        try link.send(.stop, "");
        gpa.free(try link.expect(gpa, .ack));
    } else {
        gpa.free(try link.expect(gpa, .stop));
        try link.send(.ack, "");
    }
    if (failed > 0) {
        std.debug.print("FAIL rank {d}: {d} of {d} gathers not exact\n", .{ rank, failed, cases.len });
        return error.TestFailed;
    }
    std.debug.print("PASS rank {d}: every gather bit-exact and in rank order on both ranks ({d} sizes x {d} seeds, and after timing)\n", .{ rank, cases.len, seeds.len });
}

fn ms(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// One frame of `bytes` from rank 0 to rank 1 and back (tag round, then ack), timed on rank 0.
fn linkRoundTrip(gpa: Allocator, io: std.Io, link: link_mod.Link, rank: u32, when: []const u8, bytes: usize, iters: usize, warmup: usize) !void {
    const payload = try gpa.alloc(u8, bytes);
    defer gpa.free(payload);
    for (payload, 0..) |*b, i| b.* = @truncate(i *% 131 +% 7);
    const times = try gpa.alloc(f64, iters);
    defer gpa.free(times);
    for (0..warmup + iters) |i| {
        if (rank == 0) {
            std.mem.writeInt(u64, payload[0..8], i, .little);
            const t0 = now(io);
            try link.send(.round, payload);
            const back = try link.expect(gpa, .ack);
            const t1 = now(io);
            defer gpa.free(back);
            if (!std.mem.eql(u8, back, payload)) {
                std.debug.print("FAIL control link: frame {d} of {d} bytes came back changed\n", .{ i, bytes });
                return error.TestFailed;
            }
            if (i >= warmup) times[i - warmup] = @as(f64, @floatFromInt(t1 - t0)) / 1e3;
        } else {
            const got = try link.expect(gpa, .round);
            defer gpa.free(got);
            try link.send(.ack, got);
        }
    }
    if (rank == 0) {
        const s = stats(times);
        std.debug.print("RESULT control link frame {d} B round trip ({s}): p50 {d:.1} us p90 {d:.1} us min {d:.1} us ({d} trips)\n", .{ bytes, when, s.p50, s.p90, s.min, iters });
    }
}

/// Reads `buf` full with non-blocking reads in a loop: the reader never sleeps in the kernel, so the trip shows the
/// link without the wake-up (what a busy-polling rank 1 would see between rounds).
fn spinRead(fd: std.posix.socket_t, buf: []u8) !void {
    const posix = std.posix;
    var done: usize = 0;
    while (done < buf.len) {
        const rc = posix.system.recvfrom(fd, buf[done..].ptr, buf.len - done, posix.MSG.DONTWAIT, null, null);
        switch (posix.errno(rc)) {
            .SUCCESS => {
                if (rc == 0) return error.PeerClosed;
                done += @intCast(rc);
            },
            .AGAIN, .INTR => continue,
            else => return error.LinkReadFailed,
        }
    }
}

/// The same frame round trip with both ends spinning on the socket (spinRead) instead of blocking in read.
fn spinRoundTrip(io: std.Io, link: link_mod.Link, rank: u32, comptime bytes: usize, iters: usize, warmup: usize) !void {
    var frame: [12 + bytes]u8 = undefined;
    var times_buf: [100_000]f64 = undefined;
    const times = times_buf[0..@min(iters, times_buf.len)];
    for (0..warmup + times.len) |i| {
        if (rank == 0) {
            var payload: [bytes]u8 = @splat(0);
            std.mem.writeInt(u64, payload[0..8], i, .little);
            const t0 = now(io);
            try link.send(.round, &payload);
            try spinRead(link.fd, &frame);
            const t1 = now(io);
            if (!std.mem.eql(u8, frame[12..], &payload)) return error.TestFailed;
            if (i >= warmup) times[i - warmup] = @as(f64, @floatFromInt(t1 - t0)) / 1e3;
        } else {
            try spinRead(link.fd, &frame);
            try link.send(.ack, frame[12..]);
        }
    }
    if (rank == 0) {
        const s = stats(times);
        std.debug.print("RESULT control link frame {d} B round trip (spinning reads): p50 {d:.1} us p90 {d:.1} us min {d:.1} us ({d} trips)\n", .{ bytes, s.p50, s.p90, s.min, times.len });
    }
}

const Stats = struct { p50: f64, p90: f64, min: f64, mean: f64 };

fn stats(v: []f64) Stats {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    var sum: f64 = 0;
    for (v) |x| sum += x;
    const at = struct {
        fn q(s: []const f64, p: f64) f64 {
            const i: usize = @intFromFloat(@floor(p * @as(f64, @floatFromInt(s.len - 1)) + 0.5));
            return s[i];
        }
    }.q;
    return .{ .p50 = at(v, 0.5), .p90 = at(v, 0.9), .min = v[0], .mean = sum / @as(f64, @floatFromInt(v.len)) };
}

fn splitmix(x: u64) u64 {
    var z = x +% 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

/// Rank `r`'s partial for one seed: arbitrary 32-bit words, the first ones the fp32 values a careless copy would
/// change (quiet and signalling NaNs with payloads, both zeros, denormals, infinities, the largest finite).
fn fill(words: []u32, seed: u64, r: u32) void {
    const special = [_]u32{ 0x7fc0_0001, 0xffc1_2345, 0x7f80_0001, 0x8000_0000, 0x0000_0000, 0x0000_0001, 0x807f_ffff, 0x7f80_0000, 0xff80_0000, 0x7f7f_ffff };
    for (words, 0..) |*w, i| {
        w.* = if (i < special.len) special[i] else @truncate(splitmix(seed ^ (@as(u64, r) << 56) ^ @as(u64, i)));
    }
    // the two ranks' blocks differ everywhere, so a swap or a duplicated block shows: past the specials each rank
    // draws its own stream, and rank 1 holds the specials in reverse order
    if (r == 1) std.mem.reverse(u32, words[0..@min(special.len, words.len)]);
}

const Failed = error{TestFailed};

/// 300 gathers in a row (every third in NCCL's in-place form), sizes cycling 1 / 7 / 16 / 2 rows (and 3 x 65 words), each call's payload drawn from
/// its index: every output checked against both ranks' expected words. RoCE's two slots and sequence numbers are
/// exercised with fresh data each time; nothing waits between calls but the download that checks them.
fn sequenceCheck(gpa: Allocator, d: *const cuda.Driver, comm: *const Comm, stream: cuda.Stream, rank: u32) !void {
    const shapes = [_][2]usize{ .{ 1, hidden }, .{ 7, hidden }, .{ 16, hidden }, .{ 2, hidden }, .{ 3, 65 } };
    const most = 16 * hidden;
    var send = try cuda.DeviceBuffer.alloc(d, most * 4);
    defer send.free();
    var recv = try cuda.DeviceBuffer.alloc(d, 2 * most * 4);
    defer recv.free();
    const want = try gpa.alloc(u32, 2 * most);
    defer gpa.free(want);
    const got = try gpa.alloc(u32, 2 * most);
    defer gpa.free(got);
    var bad: usize = 0;
    for (0..300) |i| {
        const sh = shapes[i % shapes.len];
        const count = sh[0] * sh[1];
        fill(want[0..count], 0xc0ffee00 + i, 0);
        fill(want[count .. 2 * count], 0xc0ffee00 + i, 1);
        if (i % 3 == 2) {
            // NCCL's in-place form: this rank's block already in place inside the receive buffer
            try recv.upload(rank * count * 4, std.mem.sliceAsBytes(want[rank * count ..][0..count]));
            try comm.gatherPartials(recv.ptr + rank * count * 4, recv.ptr, sh[0], sh[1], stream.handle);
        } else {
            try send.upload(0, std.mem.sliceAsBytes(want[rank * count ..][0..count]));
            try comm.gatherPartials(send.ptr, recv.ptr, sh[0], sh[1], stream.handle);
        }
        try stream.synchronize();
        try recv.download(0, std.mem.sliceAsBytes(got[0 .. 2 * count]));
        if (!std.mem.eql(u32, got[0 .. 2 * count], want[0 .. 2 * count])) bad += 1;
    }
    if (bad > 0) {
        std.debug.print("FAIL rank {d} sequence: {d} of 300 changing gathers wrong\n", .{ rank, bad });
        return error.TestFailed;
    }
    std.debug.print("PASS rank {d} sequence: 300 gathers of changing payloads and sizes exact\n", .{rank});
}

fn sameWords(what: []const u8, c: Case, rank: u32, got: []const u32, want: []const u32) Failed!void {
    if (std.mem.eql(u32, got, want)) return;
    var bad: usize = 0;
    var first: usize = 0;
    for (got, want, 0..) |g, w, i| {
        if (g != w) {
            if (bad == 0) first = i;
            bad += 1;
        }
    }
    const width = c.width;
    const block = c.rows * width;
    std.debug.print("FAIL rank {d} {s} {s}: {d} of {d} words differ, first at word {d} (block {d} row {d} col {d}): got 0x{x:0>8} want 0x{x:0>8}\n", .{ rank, c.name, what, bad, got.len, first, first / block, (first % block) / width, first % width, got[first], want[first] });
    return error.TestFailed;
}

/// The received bytes' sha256 compared with the peer's over the link (Link.agree); a mismatch is reported on both.
fn sameOnBoth(gpa: Allocator, link: link_mod.Link, rank: u32, c: Case, bytes: []const u8) !void {
    link.agree(gpa, rank, bytes) catch |e| {
        if (e == error.RanksDisagree) std.debug.print("FAIL rank {d} {s}: the two ranks hold different bytes\n", .{ rank, c.name });
        return e;
    };
}

fn gatherCase(gpa: Allocator, io: std.Io, d: *const cuda.Driver, comm: *const Comm, link: link_mod.Link, stream: cuda.Stream, rank: u32, c: Case, iters: usize, warmup: usize) !void {
    const count = c.rows * c.width;
    const bytes = count * 4;
    var send = try cuda.DeviceBuffer.alloc(d, bytes);
    defer send.free();
    var recv = try cuda.DeviceBuffer.alloc(d, 2 * bytes);
    defer recv.free();
    const mine = try gpa.alloc(u32, count);
    defer gpa.free(mine);
    const want = try gpa.alloc(u32, 2 * count);
    defer gpa.free(want);
    const got = try gpa.alloc(u32, 2 * count);
    defer gpa.free(got);
    var bad = false;

    // exactness: every seed, a sentinel in the receive buffer first, so a word NCCL never wrote shows
    for (seeds) |seed| {
        fill(want[0..count], seed, 0);
        fill(want[count..], seed, 1);
        @memcpy(mine, want[rank * count ..][0..count]);
        try send.upload(0, std.mem.sliceAsBytes(mine));
        try recv.fill32(0xdead_beef, null);
        try comm.gatherPartials(send.ptr, recv.ptr, c.rows, c.width, stream.handle);
        try stream.synchronize();
        try recv.download(0, std.mem.sliceAsBytes(got));
        // a rank that sees wrong words still exchanges its digest, so the two stay in step on the link
        sameWords("gather", c, rank, got, want) catch {
            bad = true;
        };
        // both ranks hold the same bytes: each sends its digest, a mismatch fails both
        sameOnBoth(gpa, link, rank, c, std.mem.sliceAsBytes(got)) catch |e| if (e == error.RanksDisagree) {
            bad = true;
        } else return e;
    }

    for (0..warmup) |_| try comm.gatherPartials(send.ptr, recv.ptr, c.rows, c.width, stream.handle);
    try stream.synchronize();

    var start = try cuda.Event.init(d, true);
    defer start.deinit();
    var end = try cuda.Event.init(d, true);
    defer end.deinit();
    const host = try gpa.alloc(f64, iters);
    defer gpa.free(host);
    const dev = try gpa.alloc(f64, iters);
    defer gpa.free(dev);
    for (0..iters) |i| {
        const t0 = now(io);
        try start.record(stream);
        try comm.gatherPartials(send.ptr, recv.ptr, c.rows, c.width, stream.handle);
        try end.record(stream);
        try end.synchronize();
        const t1 = now(io);
        host[i] = @as(f64, @floatFromInt(t1 - t0)) / 1e3;
        dev[i] = @as(f64, try cuda.Event.elapsedMs(start, end)) * 1e3;
    }
    // pipelined: the calls back to back on the stream, as a graph replays them, one synchronize at the end
    const t0 = now(io);
    for (0..iters) |_| try comm.gatherPartials(send.ptr, recv.ptr, c.rows, c.width, stream.handle);
    try stream.synchronize();
    const piped = @as(f64, @floatFromInt(now(io) - t0)) / 1e3 / @as(f64, @floatFromInt(iters));

    // still exact after the timed calls (the last seed's partials)
    try recv.download(0, std.mem.sliceAsBytes(got));
    sameWords("gather after timing", c, rank, got, want) catch {
        bad = true;
    };
    sameOnBoth(gpa, link, rank, c, std.mem.sliceAsBytes(got)) catch |e| if (e == error.RanksDisagree) {
        bad = true;
    } else return e;

    const h = stats(host);
    const g = stats(dev);
    const gb = @as(f64, @floatFromInt(bytes)) / 1e3; // bytes / us / 1e3 = GB/s
    std.debug.print("RESULT rank {d} {s} [{d}x{d}] f32 {d} B/rank: host p50 {d:.1} us p90 {d:.1} us | gpu p50 {d:.1} us p90 {d:.1} us | pipelined {d:.1} us/call | busbw p50(gpu) {d:.2} GB/s pipelined {d:.2} GB/s\n", .{ rank, c.name, c.rows, c.width, bytes, h.p50, h.p90, g.p50, g.p90, piped, gb / g.p50, gb / piped });
    if (bad) return error.TestFailed;
    std.debug.print("PASS rank {d} {s}: bit-exact, rank order, same bytes on both ranks ({d} seeds and after {d} calls)\n", .{ rank, c.name, seeds.len, warmup + 2 * iters });
}
