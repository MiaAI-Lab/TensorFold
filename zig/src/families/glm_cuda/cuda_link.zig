//! GLM control link; reuses TensorFold Flash Next framing (jschmied PR #443).
//! The two ranks' control link: one TCP connection from rank 1 to rank 0's `--master` address and port. It carries what
//! the collectives do not: the NCCL id, an agreement on the settings both ranks load with, and rank 0's decisions
//! (each request, each round's plan, the stop). Frames are a u32 tag, a u64 length and the bytes, in little endian.
//! The activations never pass here; they go over NCCL (cuda_comm.zig).
const std = @import("std");
const posix = std.posix;
const net = std.Io.net;
const Threaded = std.Io.Threaded;
const Allocator = std.mem.Allocator;

pub const Tag = enum(u32) {
    nccl_id = 1,
    settings = 2,
    request = 3,
    round = 4,
    stop = 5,
    ack = 6,
    /// A follower's first frame at TP>2: its rank (u32), so rank 0 puts the sockets in rank order.
    hello = 7,
    /// Rank 0's answer to the settings agreement at TP>2: one byte, 1 when every rank's settings matched.
    verdict = 8,
    /// Spill v2's compatibility fingerprint (cuda_spill_identity.Exchange) once the settings agreed: rank 0's to every
    /// follower, then each follower's to rank 0.
    fields = 9,
    /// Rank 0's answer to the fingerprints: the agreement, or the first rank and field that differ.
    fields_verdict = 10,
    _,
};

pub const Frame = struct {
    tag: Tag,
    bytes: []u8,

    pub fn deinit(f: Frame, gpa: Allocator) void {
        gpa.free(f.bytes);
    }
};

/// The most a frame may carry (a request with a 1M-token prompt is 4 MiB of ids; this leaves room).
pub const max_frame: u64 = 64 << 20;

fn socket(family: u32) !posix.socket_t {
    const rc = posix.system.socket(family, posix.SOCK.STREAM | posix.SOCK.CLOEXEC, 0);
    if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
    return @intCast(rc);
}

fn noDelay(fd: posix.socket_t) void {
    const one: c_int = 1;
    posix.setsockopt(fd, posix.IPPROTO.TCP, posix.TCP.NODELAY, std.mem.asBytes(&one)) catch {};
}

/// Where rank 0 listens. At TP=2 the `--master` address itself. At TP=3/4 every follower is given the address of rank 0
/// on ITS OWN cable (scripts/nodes.sh: `--master ${LINK_HEAD_ADDR[i]:-$MASTER_ADDR}` per rank; on the switchless ring
/// the ranks are on different subnets, each rank 0 address a different interface), so rank 0 listens on the
/// wildcard of the same family and port: one listener reaches all of them. TF_MASTER_BIND=exact keeps the one address.
pub fn bindAddress(master: net.IpAddress, world: u32, exact: bool) net.IpAddress {
    if (world <= 2 or exact) return master;
    return switch (master) {
        .ip4 => |a| .{ .ip4 = .unspecified(a.port) },
        .ip6 => |a| .{ .ip6 = .unspecified(a.port) },
    };
}

/// Rank 0's listening socket on the master address (port 0: any free port, read back with `port`).
pub const Server = struct {
    fd: posix.socket_t,

    pub fn open(address: net.IpAddress) !Server {
        var storage: Threaded.PosixAddress = undefined;
        const len = Threaded.addressToPosix(&address, &storage);
        const fd = try socket(if (address == .ip4) posix.AF.INET else posix.AF.INET6);
        errdefer _ = posix.system.close(fd);
        const one: c_int = 1;
        try posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&one));
        switch (posix.errno(posix.system.bind(fd, &storage.any, len))) {
            .SUCCESS => {},
            .ADDRINUSE => return error.MasterPortInUse,
            else => return error.BindFailed,
        }
        if (posix.errno(posix.system.listen(fd, 4)) != .SUCCESS) return error.ListenFailed;
        return .{ .fd = fd };
    }

    pub fn port(s: Server) u16 {
        var storage: Threaded.PosixAddress = undefined;
        var len: posix.socklen_t = @sizeOf(Threaded.PosixAddress);
        if (posix.errno(posix.system.getsockname(s.fd, &storage.any, &len)) != .SUCCESS) return 0;
        return Threaded.addressFromPosix(&storage).getPort();
    }

    /// The peer's connection, or error.PeerTimeout after `timeout_ms`.
    pub fn accept(s: Server, timeout_ms: i32) !Link {
        var fds = [_]posix.pollfd{.{ .fd = s.fd, .events = posix.POLL.IN, .revents = 0 }};
        if (try posix.poll(&fds, timeout_ms) == 0) return error.PeerTimeout;
        const rc = posix.system.accept(s.fd, null, null);
        if (posix.errno(rc) != .SUCCESS) return error.AcceptFailed;
        const fd: posix.socket_t = @intCast(rc);
        noDelay(fd);
        return .{ .fd = fd };
    }

    pub fn close(s: Server) void {
        _ = posix.system.close(s.fd);
    }

    /// TP=N, rank 0: the other `world - 1` ranks' connections, each named by its `hello` frame, as one fan-out link
    /// (`send` reaches all in rank order; `Link.at(i)` is rank i + 1 alone). error.PeerTimeout when a rank is
    /// missing after `timeout_ms` (a total), error.BadHello for a rank out of range or one that came twice.
    pub fn acceptRanks(s: Server, gpa: Allocator, world: u32, timeout_ms: i32) !Link {
        if (world < 2 or world > 2 + Link.max_more) return error.UnsupportedWorld;
        var socks: [1 + Link.max_more]posix.socket_t = @splat(-1);
        errdefer for (socks) |fd| if (fd >= 0) {
            _ = posix.system.close(fd);
        };
        var got: u32 = 0;
        while (got < world - 1) : (got += 1) {
            const l = try s.accept(timeout_ms);
            errdefer l.close();
            const bytes = try l.expect(gpa, .hello);
            defer gpa.free(bytes);
            if (bytes.len != 4) return error.BadHello;
            const rank = std.mem.readInt(u32, bytes[0..4], .little);
            if (rank < 1 or rank >= world or socks[rank - 1] >= 0) return error.BadHello;
            socks[rank - 1] = l.fd;
        }
        var out: Link = .{ .fd = socks[0] };
        for (socks[1 .. world - 1], 0..) |fd, i| out.more[i] = fd;
        return out;
    }
};

pub const Link = struct {
    fd: posix.socket_t,
    /// Rank 0 at TP=3/4: the other followers' sockets (-1: none). `send` goes to every one, in rank order; `recv`
    /// reads `fd` alone (rank 0 never reads the fan-out: the followers answer where a protocol needs it, in `Hub`).
    more: [max_more]posix.socket_t = @splat(-1),

    pub const max_more = 2;

    /// Every socket of this link, `fd` first.
    pub fn sockets(l: *const Link, out: *[1 + max_more]posix.socket_t) []const posix.socket_t {
        out[0] = l.fd;
        var n: usize = 1;
        for (l.more) |m| if (m >= 0) {
            out[n] = m;
            n += 1;
        };
        return out[0..n];
    }

    /// Rank 1: connect to rank 0, retrying every 200 ms (rank 0 may still be loading) until `timeout_ms`.
    pub fn join(io: std.Io, address: net.IpAddress, timeout_ms: u32) !Link {
        var storage: Threaded.PosixAddress = undefined;
        const len = Threaded.addressToPosix(&address, &storage);
        var waited: u32 = 0;
        while (true) {
            const fd = try socket(if (address == .ip4) posix.AF.INET else posix.AF.INET6);
            if (posix.errno(posix.system.connect(fd, &storage.any, len)) == .SUCCESS) {
                noDelay(fd);
                return .{ .fd = fd };
            }
            _ = posix.system.close(fd);
            if (waited >= timeout_ms) return error.MasterUnreachable;
            std.Io.sleep(io, .fromMilliseconds(200), .awake) catch {};
            waited += 200;
        }
    }

    /// The link to follower `i` (rank i + 1) alone.
    pub fn at(l: Link, i: usize) Link {
        var all: [1 + max_more]posix.socket_t = undefined;
        return .{ .fd = l.sockets(&all)[i] };
    }

    /// How many followers this link reaches.
    pub fn followers(l: Link) usize {
        var all: [1 + max_more]posix.socket_t = undefined;
        return l.sockets(&all).len;
    }

    /// A follower at TP>2: say which rank this is (rank 0's `acceptRanks` reads it first).
    pub fn hello(l: Link, rank: u32) !void {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, rank, .little);
        try l.send(.hello, &b);
    }

    pub fn close(l: Link) void {
        var all: [1 + max_more]posix.socket_t = undefined;
        for (l.sockets(&all)) |fd| {
            _ = posix.system.shutdown(fd, posix.SHUT.RDWR);
            _ = posix.system.close(fd);
        }
    }

    fn writeAll(l: Link, bytes: []const u8) !void {
        return writeFd(l.fd, bytes);
    }

    fn writeFd(fd: posix.socket_t, bytes: []const u8) !void {
        var done: usize = 0;
        while (done < bytes.len) {
            const rc = posix.system.write(fd, bytes[done..].ptr, bytes.len - done);
            switch (posix.errno(rc)) {
                .SUCCESS => done += @intCast(rc),
                .INTR => continue,
                else => return error.LinkWriteFailed,
            }
        }
    }

    fn readAll(l: Link, bytes: []u8) !void {
        var done: usize = 0;
        while (done < bytes.len) {
            const rc = posix.system.read(l.fd, bytes[done..].ptr, bytes.len - done);
            switch (posix.errno(rc)) {
                .SUCCESS => {
                    if (rc == 0) return error.PeerClosed;
                    done += @intCast(rc);
                },
                .INTR => continue,
                else => return error.LinkReadFailed,
            }
        }
    }

    pub fn send(l: Link, tag: Tag, bytes: []const u8) !void {
        var head: [12]u8 = undefined;
        std.mem.writeInt(u32, head[0..4], @backingInt(tag), .little);
        std.mem.writeInt(u64, head[4..12], bytes.len, .little);
        var all: [1 + max_more]posix.socket_t = undefined;
        for (l.sockets(&all)) |fd| {
            try writeFd(fd, &head);
            try writeFd(fd, bytes);
        }
    }

    pub fn recv(l: Link, gpa: Allocator) !Frame {
        var head: [12]u8 = undefined;
        try l.readAll(&head);
        const len = std.mem.readInt(u64, head[4..12], .little);
        if (len > max_frame) return error.FrameTooLarge;
        const bytes = try gpa.alloc(u8, @intCast(len));
        errdefer gpa.free(bytes);
        try l.readAll(bytes);
        return .{ .tag = @fromBackingInt(@intCast(std.mem.readInt(u32, head[0..4], .little))), .bytes = bytes };
    }

    /// The next frame, which must carry `tag`.
    pub fn expect(l: Link, gpa: Allocator, tag: Tag) ![]u8 {
        const f = try l.recv(gpa);
        if (f.tag != tag) {
            f.deinit(gpa);
            return error.UnexpectedFrame;
        }
        return f.bytes;
    }

    /// Both ranks must load with the same settings (checkpoint, window, streams, drafts, kernels, rope): each sends
    /// the sha256 of its canonical settings text and compares it with the peer's. Rank 0 sends first; both ranks get
    /// error.RanksDisagree on a mismatch.
    pub fn agree(l: Link, gpa: Allocator, rank: u32, settings: []const u8) !void {
        var mine: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(settings, &mine, .{});
        if (rank == 0) try l.send(.settings, &mine);
        const theirs = try l.expect(gpa, .settings);
        defer gpa.free(theirs);
        if (rank != 0) try l.send(.settings, &mine);
        // the caller says which settings differ: it holds the text, this only compares digests
        if (!std.mem.eql(u8, &mine, theirs)) return error.RanksDisagree;
    }

    /// `agree` for any number of ranks (TP=2, 3, 4): every follower sends the sha256 of its canonical settings to rank
    /// 0 after reading rank 0's; rank 0 compares them all and answers with one `verdict` frame, so every rank gets
    /// error.RanksDisagree when any one differs. Rank 0's link is the fan-out of `acceptRanks`; a follower's its one.
    pub fn agreeRanks(l: Link, gpa: Allocator, rank: u32, settings: []const u8) !void {
        var mine: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(settings, &mine, .{});
        if (rank == 0) {
            try l.send(.settings, &mine);
            var same = true;
            for (0..l.followers()) |i| {
                const theirs = try l.at(i).expect(gpa, .settings);
                defer gpa.free(theirs);
                if (!std.mem.eql(u8, &mine, theirs)) same = false;
            }
            try l.send(.verdict, &.{@intFromBool(same)});
            if (!same) return error.RanksDisagree;
        } else {
            const first = try l.expect(gpa, .settings);
            defer gpa.free(first);
            try l.send(.settings, &mine);
            const v = try l.expect(gpa, .verdict);
            defer gpa.free(v);
            if (v.len != 1 or v[0] != 1 or !std.mem.eql(u8, &mine, first)) return error.RanksDisagree;
        }
    }
};

test "a frame and a settings agreement over a local link" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const server = try Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    const port = server.port();
    const Peer = struct {
        fn run(io2: std.Io, p: u16, ok: *bool) void {
            const l = Link.join(io2, net.IpAddress.parse("127.0.0.1", p) catch return, 5000) catch return;
            defer l.close();
            const got = l.expect(std.testing.allocator, .request) catch return;
            defer std.testing.allocator.free(got);
            l.agree(std.testing.allocator, 1, "model=a context=8") catch return;
            ok.* = std.mem.eql(u8, got, "hello rank 1");
        }
    };
    var ok = false;
    const t = try std.Thread.spawn(.{}, Peer.run, .{ io, port, &ok });
    const l = try server.accept(5000);
    defer l.close();
    try l.send(.request, "hello rank 1");
    try l.agree(gpa, 0, "model=a context=8");
    t.join();
    try std.testing.expect(ok);
}

test "different settings are refused on both ranks" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const server = try Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    const Peer = struct {
        fn run(io2: std.Io, p: u16, refused: *bool) void {
            const l = Link.join(io2, net.IpAddress.parse("127.0.0.1", p) catch return, 5000) catch return;
            defer l.close();
            l.agree(std.testing.allocator, 1, "context=16") catch |e| {
                refused.* = e == error.RanksDisagree;
            };
        }
    };
    var refused = false;
    const t = try std.Thread.spawn(.{}, Peer.run, .{ io, server.port(), &refused });
    const l = try server.accept(5000);
    defer l.close();
    try std.testing.expectError(error.RanksDisagree, l.agree(gpa, 0, "context=8"));
    t.join();
    try std.testing.expect(refused);
}

/// Rank 0 and `world - 1` follower threads over loopback: every follower joins, says hello (in a scrambled order),
/// then `body` runs on each side with that rank's link.
fn hubRun(comptime body: fn (l: Link, gpa: Allocator, rank: u32, mismatch: u32, out: *?anyerror) void, world: u32, mismatch: u32, errs: []?anyerror) !void {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const server = try Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    const Peer = struct {
        fn run(io2: std.Io, p: u16, rank: u32, mm: u32, out: *?anyerror) void {
            const l = Link.join(io2, net.IpAddress.parse("127.0.0.1", p) catch return, 5000) catch |e| {
                out.* = e;
                return;
            };
            defer l.close();
            l.hello(rank) catch |e| {
                out.* = e;
                return;
            };
            body(l, std.testing.allocator, rank, mm, out);
        }
    };
    var threads: [3]std.Thread = undefined;
    const order = [_]u32{ 3, 1, 2 };
    var n: usize = 0;
    for (order) |r| if (r < world) {
        threads[n] = try std.Thread.spawn(.{}, Peer.run, .{ io, server.port(), r, mismatch, &errs[r] });
        n += 1;
    };
    const l = try server.acceptRanks(gpa, world, 5000);
    defer l.close();
    body(l, gpa, 0, mismatch, &errs[0]);
    for (threads[0 .. world - 1]) |t| t.join();
}

fn agreeBody(l: Link, gpa: Allocator, rank: u32, mismatch: u32, out: *?anyerror) void {
    l.agreeRanks(gpa, rank, if (rank == mismatch) "context=16" else "context=8") catch |e| {
        out.* = e;
    };
}

test "hello puts the followers in rank order and the settings agree across 2, 3 and 4 ranks" {
    for ([_]u32{ 2, 3, 4 }) |world| {
        var errs: [4]?anyerror = @splat(null);
        try hubRun(agreeBody, world, 99, &errs);
        for (errs) |e| try std.testing.expect(e == null);
    }
}

test "one rank's different settings are refused on every rank" {
    var errs: [4]?anyerror = @splat(null);
    try hubRun(agreeBody, 4, 2, &errs);
    for (errs) |e| try std.testing.expectEqual(@as(?anyerror, error.RanksDisagree), e);
}

fn fanBody(l: Link, gpa: Allocator, rank: u32, _: u32, out: *?anyerror) void {
    if (rank == 0) {
        l.send(.request, "all") catch |e| {
            out.* = e;
            return;
        };
        // each follower answers alone with its rank, read in rank order
        for (0..l.followers()) |i| {
            const b = l.at(i).expect(gpa, .ack) catch |e| {
                out.* = e;
                return;
            };
            defer gpa.free(b);
            if (b.len != 4 or std.mem.readInt(u32, b[0..4], .little) != i + 1) out.* = error.WrongRank;
        }
    } else {
        const b = l.expect(gpa, .request) catch |e| {
            out.* = e;
            return;
        };
        defer gpa.free(b);
        var me: [4]u8 = undefined;
        std.mem.writeInt(u32, &me, rank, .little);
        l.send(.ack, &me) catch |e| {
            out.* = e;
        };
    }
}

test "rank 0's send reaches every follower and Link.at is rank i + 1 alone" {
    var errs: [4]?anyerror = @splat(null);
    try hubRun(fanBody, 4, 99, &errs);
    for (errs) |e| try std.testing.expect(e == null);
}

test "a rank that says hello twice or out of range is refused" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const server = try Server.open(try net.IpAddress.parse("127.0.0.1", 0));
    defer server.close();
    const Peer = struct {
        fn run(io2: std.Io, p: u16, rank: u32) void {
            const l = Link.join(io2, net.IpAddress.parse("127.0.0.1", p) catch return, 5000) catch return;
            l.hello(rank) catch {};
            std.Io.sleep(io2, .fromMilliseconds(300), .awake) catch {};
            l.close();
        }
    };
    const a = try std.Thread.spawn(.{}, Peer.run, .{ io, server.port(), @as(u32, 1) });
    const b = try std.Thread.spawn(.{}, Peer.run, .{ io, server.port(), @as(u32, 1) });
    try std.testing.expectError(error.BadHello, server.acceptRanks(gpa, 3, 5000));
    a.join();
    b.join();
}

test "TP=3/4 rank 0 listens on the wildcard so followers reach it by any of its addresses" {
    const m = try net.IpAddress.parse("10.0.22.1", 29551);
    try std.testing.expectEqual(m.getPort(), bindAddress(m, 2, false).getPort());
    try std.testing.expect(bindAddress(m, 2, false).ip4.bytes[0] == 10);
    for ([_]u32{ 3, 4 }) |w| {
        const b = bindAddress(m, w, false);
        try std.testing.expectEqual(@as(u16, 29551), b.getPort());
        try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 0 }, &b.ip4.bytes);
        try std.testing.expectEqualSlices(u8, &m.ip4.bytes, &bindAddress(m, w, true).ip4.bytes);
    }
    const m6 = try net.IpAddress.parse("fd00::1", 7);
    try std.testing.expect(bindAddress(m6, 4, false) == .ip6);
}

test "a wildcard listener takes followers that dial its loopback address" {
    const gpa = std.testing.allocator;
    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const server = try Server.open(bindAddress(try net.IpAddress.parse("127.0.0.1", 0), 3, false));
    defer server.close();
    const Peer = struct {
        fn run(io2: std.Io, p: u16, rank: u32) void {
            const l = Link.join(io2, net.IpAddress.parse("127.0.0.1", p) catch return, 5000) catch return;
            defer l.close();
            l.hello(rank) catch return;
            const b = l.expect(std.testing.allocator, .request) catch return;
            std.testing.allocator.free(b);
        }
    };
    const a = try std.Thread.spawn(.{}, Peer.run, .{ io, server.port(), @as(u32, 2) });
    const b = try std.Thread.spawn(.{}, Peer.run, .{ io, server.port(), @as(u32, 1) });
    const l = try server.acceptRanks(gpa, 3, 5000);
    try l.send(.request, "go");
    a.join();
    b.join();
    l.close();
}
