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
