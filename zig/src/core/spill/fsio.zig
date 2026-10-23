//! Port of wojo's GLM 0088/0110 spill tier (PR #78, PR #124): the store's file primitives. Durable publication
//! (unique temporary name, short writes and EINTR handled, fsync, pages dropped, rename, directory fsync), aligned
//! O_DIRECT writes with a buffered fallback, hard links, the store owner lock and free-disk readings. Reads go
//! through `core/direct_io.zig`. Everything takes directory descriptors (`*at` calls), never assembled paths.
//! Tests replace single syscalls through `Hooks` (short writes, EINTR, ENOSPC, crash points, page events).
const std = @import("std");
const builtin = @import("builtin");
const c = std.c;
const direct_io = @import("../direct_io.zig");
const mutants = @import("mutants.zig");

pub const alignment = direct_io.alignment;
pub const Buffer = direct_io.Buffer;
const linux = builtin.os.tag == .linux;
const has_direct = @hasField(c.O, "DIRECT");

pub const Error = error{ Interrupted, NoSpace, Unsupported, Missing, Exists, Busy, Io };

/// A step of the write path where a test may simulate a crash (the writer touches the disk no more after it). The
/// hook also gets the file's name: `.written` with "x.blk" is a block written to its temporary name, not renamed.
pub const Step = enum { written, synced, renamed, linked, dir_synced, published, rewritten };

pub const Event = enum { fsync, drop_pages, dir_fsync, rename, write };

/// Test seams; production leaves every field null.
pub const Hooks = struct {
    ctx: ?*anyopaque = null,
    /// Replaces one pwrite: a byte count (short is fine) or an error (Interrupted is retried).
    pwrite: ?*const fn (ctx: ?*anyopaque, fd: c.fd_t, buf: []const u8, off: u64) Error!usize = null,
    /// True: the process "dies" right after `step` on `name` (the writer stops touching the disk).
    crash: ?*const fn (ctx: ?*anyopaque, step: Step, name: []const u8) bool = null,
    /// Observes durability events in order (`name`: the file or directory).
    event: ?*const fn (ctx: ?*anyopaque, ev: Event, name: []const u8) void = null,
    /// Overrides the free bytes of the store's file system.
    free: ?*const fn (ctx: ?*anyopaque) u64 = null,
    /// O_DIRECT opens fail with EINVAL (a file system without it).
    no_direct: bool = false,
    /// Counts every name the open's scan reads (a caller on another thread watches a long scan: a follower's open
    /// answers rank 0 while it runs).
    progress: ?*std.atomic.Value(u64) = null,
    /// Called for every name the open's scan reads (tests: a slow disk).
    scanned: ?*const fn (ctx: ?*anyopaque, name: []const u8) void = null,
};

pub const Crashed = error.Crashed;

fn note(h: *const Hooks, ev: Event, name: []const u8) void {
    if (h.event) |f| f(h.ctx, ev, name);
}

pub fn crashAt(h: *const Hooks, step: Step, name: []const u8) bool {
    return if (h.crash) |f| f(h.ctx, step, name) else false;
}

fn mapErrno(e: c.E) Error {
    return switch (e) {
        .INTR => error.Interrupted,
        .NOSPC, .DQUOT => error.NoSpace,
        .INVAL, .OPNOTSUPP => error.Unsupported,
        .NOENT => error.Missing,
        .EXIST => error.Exists,
        .AGAIN, .BUSY => error.Busy,
        else => error.Io,
    };
}

fn pwrite1(h: *const Hooks, fd: c.fd_t, buf: []const u8, off: u64) Error!usize {
    if (h.pwrite) |f| return f(h.ctx, fd, buf, off);
    const n = c.pwrite(fd, buf.ptr, buf.len, @intCast(off));
    if (n < 0) return mapErrno(c.errno(n));
    return @intCast(n);
}

/// All of `buf` at `off`: short writes continue where they stopped, EINTR is retried, a zero write is an error.
pub fn pwriteAll(h: *const Hooks, fd: c.fd_t, buf: []const u8, off: u64) Error!void {
    var done: usize = 0;
    while (done < buf.len) {
        const n = pwrite1(h, fd, buf[done..], off + done) catch |e| switch (e) {
            error.Interrupted => continue,
            else => return e,
        };
        if (n == 0) return error.Io;
        done += n;
    }
}

extern "c" fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) c_int;

/// Clean pages of `fd` leave the page cache (on a unified-memory GPU the page cache is the GPU's memory).
pub fn dropPages(h: *const Hooks, fd: c.fd_t, name: []const u8) void {
    if (linux) _ = posix_fadvise(fd, 0, 0, 4); // POSIX_FADV_DONTNEED
    note(h, .drop_pages, name);
}

pub fn fsync(h: *const Hooks, fd: c.fd_t, name: []const u8) Error!void {
    while (true) {
        if (c.fsync(fd) == 0) break;
        const e = c.errno(@as(c_int, -1));
        if (e == .INTR) continue;
        return mapErrno(e);
    }
    note(h, .fsync, name);
}

pub fn syncDir(h: *const Hooks, dirfd: c.fd_t, name: []const u8) Error!void {
    while (c.fsync(dirfd) != 0) {
        const e = c.errno(@as(c_int, -1));
        if (e == .INTR) continue;
        if (e == .INVAL) break; // a file system that cannot sync directories (tmpfs on some kernels)
        return mapErrno(e);
    }
    note(h, .dir_fsync, name);
}

fn openAt(dirfd: c.fd_t, name: [:0]const u8, flags: c.O, mode: c.mode_t) Error!c.fd_t {
    while (true) {
        const fd = c.openat(dirfd, name, flags, mode);
        if (fd >= 0) return fd;
        const e = c.errno(fd);
        if (e == .INTR) continue;
        return mapErrno(e);
    }
}

pub fn renameAt(h: *const Hooks, dirfd: c.fd_t, from: [:0]const u8, to: [:0]const u8) Error!void {
    if (c.renameat(dirfd, from, dirfd, to) != 0) return mapErrno(c.errno(@as(c_int, -1)));
    note(h, .rename, to);
}

pub fn unlinkAt(dirfd: c.fd_t, name: [:0]const u8) void {
    _ = c.unlinkat(dirfd, name, 0);
}

/// `to` becomes a second name of `from`'s inode (both in `dirfd`).
pub fn linkAt(dirfd: c.fd_t, from: [:0]const u8, to: [:0]const u8) Error!void {
    if (c.linkat(dirfd, from, dirfd, to, 0) != 0) return mapErrno(c.errno(@as(c_int, -1)));
}

pub const Stat = struct { size: u64, ino: u64, nlink: u64, mtime_ns: i96 };

pub fn statFd(fd: c.fd_t) Error!Stat {
    if (linux) {
        const L = std.os.linux;
        var sx: L.Statx = undefined;
        const want: L.STATX = .{ .TYPE = true, .MODE = true, .NLINK = true, .INO = true, .SIZE = true, .MTIME = true };
        if (c.statx(fd, "", 0x1000, want, &sx) != 0) return mapErrno(c.errno(@as(c_int, -1))); // AT_EMPTY_PATH
        return .{ .size = sx.size, .ino = sx.ino, .nlink = sx.nlink, .mtime_ns = @as(i96, sx.mtime.sec) * std.time.ns_per_s + sx.mtime.nsec };
    }
    var st: c.Stat = undefined;
    if (c.fstat(fd, &st) != 0) return mapErrno(c.errno(@as(c_int, -1)));
    const t = st.mtime();
    return .{ .size = @intCast(st.size), .ino = @intCast(st.ino), .nlink = @intCast(st.nlink), .mtime_ns = @as(i96, t.sec) * std.time.ns_per_s + t.nsec };
}

pub fn statAt(dirfd: c.fd_t, name: [:0]const u8) Error!Stat {
    const fd = try openAt(dirfd, name, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
    defer _ = c.close(fd);
    return statFd(fd);
}

/// Sets `name`'s modification time (a point's last use, persisted without rewriting it).
pub fn touchAt(dirfd: c.fd_t, name: [:0]const u8, ns: i96) void {
    const fd = openAt(dirfd, name, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0) catch return;
    defer _ = c.close(fd);
    const ts: c.timespec = .{ .sec = @intCast(@divFloor(ns, std.time.ns_per_s)), .nsec = @intCast(@mod(ns, std.time.ns_per_s)) };
    const times = [2]c.timespec{ ts, ts };
    _ = c.futimens(fd, &times);
}

/// Where a durable file's bytes come from: pieces written in order (buffered), or one aligned buffer holding
/// `len` bytes and room up to the next 4 KiB (written whole with O_DIRECT where the file system takes it).
pub const Source = union(enum) {
    parts: []const []const u8,
    aligned: struct { buf: Buffer, len: usize },
};

/// Shared state of a store's writers: whether O_DIRECT still works here, a sequence for temporary names.
pub const Writer = struct {
    hooks: *const Hooks,
    direct: std.atomic.Value(bool),
    seq: std.atomic.Value(u64) = .init(0),
    pid: c.pid_t,
    /// Opens with O_DIRECT that the file system refused (the store then writes buffered).
    direct_refused: std.atomic.Value(u64) = .init(0),

    pub fn init(hooks: *const Hooks, want_direct: bool) Writer {
        return .{ .hooks = hooks, .direct = .init(want_direct and has_direct), .pid = c.getpid() };
    }

    /// A unique temporary name for `name` in `buf`: ".<name>.<pid>.<seq>.tmp".
    pub fn tmpName(w: *Writer, buf: []u8, name: []const u8) [:0]const u8 {
        const seq = w.seq.fetchAdd(1, .monotonic);
        return std.fmt.bufPrintSentinel(buf, ".{s}.{d}.{d}.tmp", .{ name, w.pid, seq }, 0) catch unreachable;
    }

    fn openWrite(w: *Writer, dirfd: c.fd_t, tmp: [:0]const u8, want_direct: bool) Error!struct { fd: c.fd_t, direct: bool } {
        const base: c.O = .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .CLOEXEC = true };
        if (want_direct and w.direct.load(.monotonic)) {
            if (has_direct) {
                var f = base;
                f.DIRECT = true;
                const got = if (w.hooks.no_direct) error.Unsupported else openAt(dirfd, tmp, f, 0o600);
                if (got) |fd| return .{ .fd = fd, .direct = true } else |e| switch (e) {
                    error.Unsupported => {
                        w.direct.store(false, .monotonic);
                        _ = w.direct_refused.fetchAdd(1, .monotonic);
                    },
                    else => return e,
                }
            }
        }
        return .{ .fd = try openAt(dirfd, tmp, base, 0o600), .direct = false };
    }

    /// A simulated crash: the descriptor goes (as the process would), the temporary file stays.
    fn die(_: *Writer, fd: *c.fd_t, ok: *bool) error{Crashed} {
        _ = c.close(fd.*);
        fd.* = -1;
        ok.* = true;
        return error.Crashed;
    }

    /// `src` durably as `name` in `dirfd`: a unique temporary name, every byte written (short writes and EINTR
    /// handled; a partial O_DIRECT write finishes buffered), cut to its length, fsynced, its pages dropped, renamed.
    /// The directory is not synced here (callers batch that). On any failure no temporary file is left behind.
    /// Returns the CRC-32 of the bytes (computed from the source as written).
    pub fn put(w: *Writer, dirfd: c.fd_t, name: [:0]const u8, src: Source) (Error || error{Crashed})!u32 {
        var tb: [160]u8 = undefined;
        const tmp = w.tmpName(&tb, name);
        var crc: u32 = 0;
        const total: usize = switch (src) {
            .parts => |ps| blk: {
                var n: usize = 0;
                for (ps) |p| n += p.len;
                break :blk n;
            },
            .aligned => |a| a.len,
        };
        var opened = try w.openWrite(dirfd, tmp, src == .aligned);
        var fd = opened.fd;
        var ok = false;
        defer if (!ok) {
            _ = c.close(fd);
            unlinkAt(dirfd, tmp);
        };
        switch (src) {
            .parts => |ps| {
                var off: u64 = 0;
                var h = std.hash.Crc32.init();
                for (ps) |p| {
                    try pwriteAll(w.hooks, fd, p, off);
                    h.update(p);
                    off += p.len;
                }
                crc = h.final();
            },
            .aligned => |a| {
                std.debug.assert(a.buf.len >= std.mem.alignForward(usize, a.len, alignment));
                crc = std.hash.Crc32.hash(a.buf[0..a.len]);
                if (opened.direct) {
                    const whole = std.mem.alignForward(usize, a.len, alignment);
                    var done: usize = 0;
                    while (done < whole) {
                        const n = pwrite1(w.hooks, fd, a.buf[done..whole], done) catch |e| switch (e) {
                            error.Interrupted => continue,
                            else => return e,
                        };
                        if (n == 0) return error.Io;
                        done += n;
                        if (done < whole and done % alignment != 0) {
                            // a partial O_DIRECT write off the 4 KiB grid: the rest through the page cache
                            _ = c.close(fd);
                            fd = openAt(dirfd, tmp, .{ .ACCMODE = .WRONLY, .CLOEXEC = true }, 0) catch |e| {
                                fd = -1;
                                return e;
                            };
                            opened.direct = false;
                            if (done < a.len) try pwriteAll(w.hooks, fd, a.buf[done..a.len], done);
                            break;
                        }
                    }
                    if (c.ftruncate(fd, @intCast(a.len)) != 0) return mapErrno(c.errno(@as(c_int, -1)));
                } else {
                    try pwriteAll(w.hooks, fd, a.buf[0..a.len], 0);
                }
            },
        }
        note(w.hooks, .write, name);
        _ = total;
        if (crashAt(w.hooks, .written, name)) return w.die(&fd, &ok);
        if (mutants.on(.pages_dropped_before_fsync)) dropPages(w.hooks, fd, name);
        try fsync(w.hooks, fd, name);
        if (!mutants.on(.small_files_keep_pages) and !mutants.on(.pages_dropped_before_fsync)) dropPages(w.hooks, fd, name); // clean now: nothing of it stays in memory
        if (crashAt(w.hooks, .synced, name)) return w.die(&fd, &ok);
        _ = c.close(fd);
        fd = -1;
        renameAt(w.hooks, dirfd, tmp, name) catch |e| {
            unlinkAt(dirfd, tmp);
            ok = true;
            return e;
        };
        ok = true;
        if (crashAt(w.hooks, .renamed, name)) return error.Crashed;
        return crc;
    }
};

/// A file to read: direct where the file system allows it. Missing: its name is gone.
pub fn openRead(dirfd: c.fd_t, name: [:0]const u8, want_direct: bool) Error!direct_io.File {
    if (has_direct and want_direct) {
        var f: c.O = .{ .ACCMODE = .RDONLY, .CLOEXEC = true };
        f.DIRECT = true;
        if (openAt(dirfd, name, f, 0)) |fd| return .{ .fd = fd, .direct = true } else |e| switch (e) {
            error.Unsupported => {},
            else => return e,
        }
    }
    return .{ .fd = try openAt(dirfd, name, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0), .direct = false };
}

/// A whole small file into a heap buffer (manifest, ids): its exact length must be at most `max`.
pub fn readSmall(gpa: std.mem.Allocator, dirfd: c.fd_t, name: [:0]const u8, max: u64) (Error || error{ TooLarge, OutOfMemory, Short })![]u8 {
    const fd = try openAt(dirfd, name, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
    defer _ = c.close(fd);
    const st = try statFd(fd);
    if (st.size > max) return error.TooLarge;
    const buf = try gpa.alloc(u8, @intCast(st.size));
    errdefer gpa.free(buf);
    var got: usize = 0;
    while (got < buf.len) {
        const n = c.pread(fd, buf.ptr + got, buf.len - got, @intCast(got));
        if (n < 0) {
            if (c.errno(n) == .INTR) continue;
            return mapErrno(c.errno(n));
        }
        if (n == 0) return error.Short;
        got += @intCast(n);
    }
    return buf;
}

pub fn openDir(path: [:0]const u8) Error!c.fd_t {
    return openAt(c.AT.FDCWD, path, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true }, 0);
}

/// A directory under the store's root, never through a symlink (a planted link must not lead a scan or a prune
/// elsewhere): a symlink fails here.
pub fn openDirAt(dirfd: c.fd_t, name: [:0]const u8) Error!c.fd_t {
    return openAt(dirfd, name, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .NOFOLLOW = true, .CLOEXEC = true }, 0);
}

/// Whether `name` in `dirfd` is a symbolic link (not followed).
pub fn isSymlinkAt(dirfd: c.fd_t, name: [:0]const u8) bool {
    var b: [1]u8 = undefined;
    return c.readlinkat(dirfd, name, &b, b.len) >= 0;
}

/// `name` in `dirfd` as a directory of mode 0700 (made if missing).
pub fn ensureDirAt(dirfd: c.fd_t, name: [:0]const u8) Error!void {
    if (c.mkdirat(dirfd, name, 0o700) != 0) {
        const e = c.errno(@as(c_int, -1));
        if (e != .EXIST) return mapErrno(e);
    }
    const fd = try openDirAt(dirfd, name);
    defer _ = c.close(fd);
    _ = c.fchmod(fd, 0o700);
}

/// Every name in directory `dirfd` (gpa-owned list of gpa-owned names), "." and ".." left out.
pub fn list(gpa: std.mem.Allocator, dirfd: c.fd_t) !std.ArrayList([:0]u8) {
    var out: std.ArrayList([:0]u8) = .empty;
    errdefer {
        for (out.items) |n| gpa.free(n);
        out.deinit(gpa);
    }
    const dup = c.dup(dirfd);
    if (dup < 0) return error.Io;
    const d = c.fdopendir(dup) orelse {
        _ = c.close(dup);
        return error.Io;
    };
    defer _ = c.closedir(d);
    c.rewinddir(d);
    while (c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&ent.name)), 0);
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
        try out.append(gpa, try gpa.dupeSentinel(u8, name, 0));
    }
    return out;
}

pub fn freeList(gpa: std.mem.Allocator, l: *std.ArrayList([:0]u8)) void {
    for (l.items) |n| gpa.free(n);
    l.deinit(gpa);
}

/// Removes `name` in `dirfd` and everything under it (a pruned other build's rank directory). A symlink anywhere in
/// it is unlinked, never followed.
pub fn removeTree(gpa: std.mem.Allocator, dirfd: c.fd_t, name: [:0]const u8) void {
    const sub = openDirAt(dirfd, name) catch {
        unlinkAt(dirfd, name);
        return;
    };
    var names = list(gpa, sub) catch {
        _ = c.close(sub);
        return;
    };
    for (names.items) |n| {
        if (c.unlinkat(sub, n, 0) != 0) removeTree(gpa, sub, n);
    }
    freeList(gpa, &names);
    _ = c.close(sub);
    _ = c.unlinkat(dirfd, name, c.AT.REMOVEDIR);
}

/// The exclusive owner lock of one rank's store: a second process (or a second open) fails with Busy before it
/// scans or deletes anything. Held until `release`.
pub const Lock = struct {
    fd: c.fd_t,

    pub fn take(dirfd: c.fd_t) Error!Lock {
        const fd = try openAt(dirfd, "owner.lock", .{ .ACCMODE = .RDWR, .CREAT = true, .NOFOLLOW = true, .CLOEXEC = true }, 0o600);
        if (c.flock(fd, c.LOCK.EX | c.LOCK.NB) != 0) {
            _ = c.close(fd);
            return error.Busy;
        }
        var buf: [32]u8 = undefined;
        const s = std.fmt.bufPrint(&buf, "{d}\n", .{c.getpid()}) catch unreachable;
        _ = c.ftruncate(fd, 0);
        _ = c.pwrite(fd, s.ptr, s.len, 0);
        return .{ .fd = fd };
    }

    pub fn release(l: *Lock) void {
        _ = c.flock(l.fd, c.LOCK.UN);
        _ = c.close(l.fd);
        l.fd = -1;
    }
};

const Statvfs = if (linux) extern struct {
    f_bsize: c_ulong,
    f_frsize: c_ulong,
    f_blocks: u64,
    f_bfree: u64,
    f_bavail: u64,
    f_files: u64,
    f_ffree: u64,
    f_favail: u64,
    f_fsid: c_ulong,
    f_flag: c_ulong,
    f_namemax: c_ulong,
    spare: [6]c_int,
    slack: [64]u8, // room for any libc's extra fields
} else extern struct {
    f_bsize: c_ulong,
    f_frsize: c_ulong,
    f_blocks: c_uint,
    f_bfree: c_uint,
    f_bavail: c_uint,
    f_files: c_uint,
    f_ffree: c_uint,
    f_favail: c_uint,
    f_fsid: c_ulong,
    f_flag: c_ulong,
    f_namemax: c_ulong,
    slack: [64]u8,
};

extern "c" fn fstatvfs(fd: c_int, buf: *Statvfs) c_int;

/// Bytes an unprivileged writer may still use on `dirfd`'s file system (0 when unknown: no write passes a floor).
pub fn diskFree(h: *const Hooks, dirfd: c.fd_t) u64 {
    if (h.free) |f| return f(h.ctx);
    var st: Statvfs = undefined;
    if (fstatvfs(dirfd, &st) != 0) return 0;
    return @as(u64, st.f_bavail) * @as(u64, st.f_frsize);
}

// -- tests ---------------------------------------------------------------------------------------------------------
const testing = std.testing;

pub fn testRoot(buf: []u8, tag: []const u8) ![:0]const u8 {
    const base = if (c.getenv("TF_SPILL_TEST_DIR")) |p| std.mem.span(p) else "/tmp";
    return std.fmt.bufPrintSentinel(buf, "{s}/tf-spill-{s}-{d}", .{ base, tag, c.getpid() }, 0);
}

pub fn freshDir(gpa: std.mem.Allocator, path: [:0]const u8) !c.fd_t {
    const parent = std.fs.path.dirname(path) orelse "/";
    var pb: [512]u8 = undefined;
    const pz = try std.fmt.bufPrintSentinel(&pb, "{s}", .{parent}, 0);
    const pfd = try openDir(pz);
    defer _ = c.close(pfd);
    var nb: [256]u8 = undefined;
    const nz = try std.fmt.bufPrintSentinel(&nb, "{s}", .{std.fs.path.basename(path)}, 0);
    removeTree(gpa, pfd, nz);
    try ensureDirAt(pfd, nz);
    return openDir(path);
}

pub fn dropDir(gpa: std.mem.Allocator, path: [:0]const u8) void {
    const parent = std.fs.path.dirname(path) orelse return;
    var pb: [512]u8 = undefined;
    const pz = std.fmt.bufPrintSentinel(&pb, "{s}", .{parent}, 0) catch return;
    const pfd = openDir(pz) catch return;
    defer _ = c.close(pfd);
    var nb: [256]u8 = undefined;
    const nz = std.fmt.bufPrintSentinel(&nb, "{s}", .{std.fs.path.basename(path)}, 0) catch return;
    removeTree(gpa, pfd, nz);
}

/// For the tests' scripted partial writes: `c.pwrite` of `want` bytes the way a short write on the 4 KiB grid's far
/// side would leave them. A real O_DIRECT descriptor refuses a write off the grid (EINVAL on ext4; tmpfs takes it), so
/// that one write goes through the page cache and the descriptor is put back.
pub fn pwritePartial(fd: c.fd_t, buf: []const u8, want: usize, off: u64) isize {
    if (!has_direct) return c.pwrite(fd, buf.ptr, want, @intCast(off));
    const bit: c_int = @bitCast(@as(u32, @bitCast(c.O{ .DIRECT = true })));
    const fl = c.fcntl(fd, c.F.GETFL);
    if (fl < 0 or fl & bit == 0 or (want % alignment == 0 and off % alignment == 0)) return c.pwrite(fd, buf.ptr, want, @intCast(off));
    _ = c.fcntl(fd, c.F.SETFL, fl & ~bit);
    defer _ = c.fcntl(fd, c.F.SETFL, fl);
    return c.pwrite(fd, buf.ptr, want, @intCast(off));
}

const Script = struct {
    steps: []const Step2,
    i: usize = 0,
    calls: usize = 0,
    events: std.ArrayList(Event) = .empty,
    const Step2 = union(enum) { short: usize, intr, nospace, pass };

    fn pwrite(ctx: ?*anyopaque, fd: c.fd_t, buf: []const u8, off: u64) Error!usize {
        const s: *Script = @ptrCast(@alignCast(ctx.?));
        s.calls += 1;
        const step: Step2 = if (s.i < s.steps.len) s.steps[s.i] else .pass;
        if (s.i < s.steps.len) s.i += 1;
        const want = switch (step) {
            .intr => return error.Interrupted,
            .nospace => return error.NoSpace,
            .short => |k| @min(k, buf.len),
            .pass => buf.len,
        };
        const n = pwritePartial(fd, buf, want, off);
        if (n < 0) return mapErrno(c.errno(n));
        return @intCast(n);
    }

    fn event(ctx: ?*anyopaque, ev: Event, _: []const u8) void {
        const s: *Script = @ptrCast(@alignCast(ctx.?));
        s.events.append(testing.allocator, ev) catch {};
    }
};

fn readBack(gpa: std.mem.Allocator, dirfd: c.fd_t, name: [:0]const u8) ![]u8 {
    return readSmall(gpa, dirfd, name, 1 << 30);
}

fn countTemps(gpa: std.mem.Allocator, dirfd: c.fd_t) !usize {
    var names = try list(gpa, dirfd);
    defer freeList(gpa, &names);
    var n: usize = 0;
    for (names.items) |x| {
        if (x[0] == '.') n += 1;
    }
    return n;
}

test "put: short writes and EINTR still write every byte; pages dropped after the fsync, then renamed" {
    const gpa = testing.allocator;
    var rb: [256]u8 = undefined;
    const root = try testRoot(&rb, "put");
    const dir = try freshDir(gpa, root);
    defer {
        _ = c.close(dir);
        dropDir(gpa, root);
    }
    var script: Script = .{ .steps = &.{ .{ .short = 3 }, .intr, .{ .short = 1 }, .intr, .pass } };
    defer script.events.deinit(gpa);
    const hooks: Hooks = .{ .ctx = &script, .pwrite = Script.pwrite, .event = Script.event };
    var w = Writer.init(&hooks, true);
    const data = "the quick brown fox jumps over the lazy dog";
    const crc = try w.put(dir, "a.bin", .{ .parts = &.{ data[0..10], data[10..] } });
    try testing.expectEqual(std.hash.Crc32.hash(data), crc);
    const back = try readBack(gpa, dir, "a.bin");
    defer gpa.free(back);
    try testing.expectEqualStrings(data, back);
    try testing.expectEqualSlices(Event, &.{ .write, .fsync, .drop_pages, .rename }, script.events.items);
    try testing.expectEqual(@as(usize, 0), try countTemps(gpa, dir));
}

test "put: a full disk leaves no temporary file and no target" {
    const gpa = testing.allocator;
    var rb: [256]u8 = undefined;
    const root = try testRoot(&rb, "nospace");
    const dir = try freshDir(gpa, root);
    defer {
        _ = c.close(dir);
        dropDir(gpa, root);
    }
    var script: Script = .{ .steps = &.{ .{ .short = 4 }, .nospace } };
    defer script.events.deinit(gpa);
    const hooks: Hooks = .{ .ctx = &script, .pwrite = Script.pwrite };
    var w = Writer.init(&hooks, false);
    try testing.expectError(error.NoSpace, w.put(dir, "b.bin", .{ .parts = &.{"0123456789"} }));
    try testing.expectEqual(@as(usize, 0), try countTemps(gpa, dir));
    try testing.expectError(error.Missing, statAt(dir, "b.bin"));
}

test "put: aligned sources, direct where it works, buffered when the file system refuses O_DIRECT" {
    const gpa = testing.allocator;
    var rb: [256]u8 = undefined;
    const root = try testRoot(&rb, "aligned");
    const dir = try freshDir(gpa, root);
    defer {
        _ = c.close(dir);
        dropDir(gpa, root);
    }
    const buf = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), 3 * alignment);
    defer gpa.free(buf);
    for (buf, 0..) |*b, i| b.* = @intCast(i % 251);
    inline for (.{ false, true }) |refuse| {
        const hooks: Hooks = .{ .no_direct = refuse };
        var w = Writer.init(&hooks, true);
        const name = if (refuse) "buffered.blk" else "direct.blk";
        _ = try w.put(dir, name, .{ .aligned = .{ .buf = buf, .len = 2 * alignment + 100 } });
        const back = try readBack(gpa, dir, name);
        defer gpa.free(back);
        try testing.expectEqualSlices(u8, buf[0 .. 2 * alignment + 100], back);
        if (refuse and has_direct) try testing.expect(!w.direct.load(.monotonic) and w.direct_refused.load(.monotonic) == 1);
    }
}

test "a partial direct write off the 4 KiB grid finishes buffered" {
    if (!has_direct) return error.SkipZigTest;
    const gpa = testing.allocator;
    var rb: [256]u8 = undefined;
    const root = try testRoot(&rb, "partial");
    const dir = try freshDir(gpa, root);
    defer {
        _ = c.close(dir);
        dropDir(gpa, root);
    }
    const buf = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), 2 * alignment);
    defer gpa.free(buf);
    for (buf, 0..) |*b, i| b.* = @intCast(i % 249);
    var script: Script = .{ .steps = &.{.{ .short = 100 }} };
    defer script.events.deinit(gpa);
    const hooks: Hooks = .{ .ctx = &script, .pwrite = Script.pwrite };
    var w = Writer.init(&hooks, true);
    // (on a file system without O_DIRECT the open falls back first; either way the bytes are whole)
    _ = try w.put(dir, "p.blk", .{ .aligned = .{ .buf = buf, .len = alignment + 7 } });
    const back = try readBack(gpa, dir, "p.blk");
    defer gpa.free(back);
    try testing.expectEqualSlices(u8, buf[0 .. alignment + 7], back);
}

test "owner lock: a second owner is refused while the first holds it" {
    const gpa = testing.allocator;
    var rb: [256]u8 = undefined;
    const root = try testRoot(&rb, "lock");
    const dir = try freshDir(gpa, root);
    defer {
        _ = c.close(dir);
        dropDir(gpa, root);
    }
    var a = try Lock.take(dir);
    try testing.expectError(error.Busy, Lock.take(dir));
    a.release();
    var b = try Lock.take(dir);
    b.release();
}

test "directories: a symlink is never followed, by an open or a tree removal" {
    const gpa = testing.allocator;
    var rb: [256]u8 = undefined;
    const root = try testRoot(&rb, "symlink");
    const dir = try freshDir(gpa, root);
    defer {
        _ = c.close(dir);
        dropDir(gpa, root);
    }
    try ensureDirAt(dir, "target");
    const t = try openDirAt(dir, "target");
    defer _ = c.close(t);
    const f = try openAt(t, "keep.me", .{ .ACCMODE = .WRONLY, .CREAT = true, .CLOEXEC = true }, 0o600);
    _ = c.close(f);
    try ensureDirAt(dir, "tree");
    const tr = try openDirAt(dir, "tree");
    defer _ = c.close(tr);
    try testing.expect(c.symlinkat("../target", tr, "link") == 0);
    try testing.expect(c.symlinkat("target", dir, "top") == 0);
    try testing.expect(isSymlinkAt(dir, "top") and !isSymlinkAt(dir, "target"));
    if (openDirAt(dir, "top")) |fd| {
        _ = c.close(fd);
        return error.TestUnexpectedResult;
    } else |_| {}
    removeTree(gpa, dir, "tree");
    try testing.expect(!isSymlinkAt(tr, "link"));
    _ = try statAt(t, "keep.me"); // the link's target and its contents are untouched
}
