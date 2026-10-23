//! Port of wojo's GLM 0110 `tools/spill_blocks_check.py` and `tools/spill_blocks_mutants.sh` (PR #124) as Zig tests,
//! plus PORT-SPILL-V2.md's stage-1 (format, lineage, dedup, references, eviction, quotas) and stage-5 (durability
//! fault campaign) cases for the native store. A real store over a scratch directory, a fake "device" of rows per
//! source the test engine packs blocks from, and the engine's side of every call (stage, staged, captured,
//! readReady, readRelease) played by the test. Each check returns error.CheckFailed when a property fails; the
//! mutant test switches each authored fix back off (`mutants.zig`) and requires some check to fail.
const std = @import("std");
const c = std.c;
const format = @import("format.zig");
const fsio = @import("fsio.zig");
const slots = @import("slots.zig");
const store = @import("store.zig");
const mutants = @import("mutants.zig");
const Allocator = std.mem.Allocator;
const Store = store.Store;
const Digest = format.Digest;
const Lineage = format.Lineage;
const testing = std.testing;
const io = std.testing.io;

const block_tokens = 16;
const bpt = 256; // bytes a token on this rank
const block_bytes = block_tokens * bpt;
const max_tokens = 1024;
const small_max = 8192;
const secs = [_]store.SectionSpec{ .{ .name = "ring", .len = 300 }, .{ .name = "raw", .len = 200 } };

var quiet = false; // a mutant run: failures are expected, not printed
var serial: u32 = 0;

const CheckFailed = error.CheckFailed;

fn ensure(ok: bool, src: std.builtin.SourceLocation) !void {
    if (ok) return;
    if (!quiet) std.debug.print("     check failed at {s}:{d}\n", .{ src.file, src.line });
    return CheckFailed;
}

const Opts = struct {
    tag: []const u8,
    write_slots: usize = 8,
    read_slots: usize = 8,
    small_slots: usize = 4,
    restore_slots: usize = 2,
    cap: u64 = 64 << 20,
    floor: u64 = 0,
    turns: u32 = 4,
    quota: f64 = 0,
    max_reads: u32 = 2,
    keep_builds: u32 = 1,
    backlog: u64 = 1 << 30,
    free: ?u64 = null,
    no_direct: bool = false,
    prefix: []const u8 = "native-spill-", // the lifted checks' namespace
    bpt: usize = bpt, // bytes a token (the block's bytes follow)
};

const Ev = struct { ev: fsio.Event, name: [96]u8, len: usize };

/// The test's side of the hooks: event log, an action at a step, a crash point, a pwrite script, a disk-free value.
const Hk = struct {
    mutex: std.Io.Mutex = .init,
    events: std.ArrayList(Ev) = .empty,
    gpa: Allocator,
    action_step: ?fsio.Step = null,
    action_suffix: []const u8 = "",
    action: ?*const fn (h: *H) void = null,
    crash_step: ?fsio.Step = null,
    crash_suffix: []const u8 = "",
    crash_skip: u32 = 0,
    nospace_writes: u32 = 0, // the next pwrites failing with ENOSPC
    short_intr: bool = false, // every pwrite short (at most 1000 bytes), and every other one EINTR first
    flip: bool = false,
    free: ?u64 = null,
    h: *H = undefined,

    fn crash(ctx: ?*anyopaque, step: fsio.Step, name: []const u8) bool {
        const k: *Hk = @ptrCast(@alignCast(ctx.?));
        if (std.c.getenv("SPILL_DEBUG") != null) std.debug.print("hook {t} {s}\n", .{ step, name });
        if (k.action) |f| if (k.action_step == step and std.mem.endsWith(u8, name, k.action_suffix)) {
            k.action = null;
            f(k.h);
        };
        if (k.crash_step) |cs| if (cs == step and std.mem.endsWith(u8, name, k.crash_suffix)) {
            if (k.crash_skip > 0) {
                k.crash_skip -= 1;
                return false;
            }
            k.crash_step = null;
            return true;
        };
        return false;
    }

    fn event(ctx: ?*anyopaque, ev: fsio.Event, name: []const u8) void {
        const k: *Hk = @ptrCast(@alignCast(ctx.?));
        var e: Ev = .{ .ev = ev, .name = undefined, .len = @min(name.len, 96) };
        @memcpy(e.name[0..e.len], name[0..e.len]);
        k.mutex.lockUncancelable(io);
        defer k.mutex.unlock(io);
        k.events.append(k.gpa, e) catch {};
    }

    fn pwrite(ctx: ?*anyopaque, fd: c.fd_t, buf: []const u8, off: u64) fsio.Error!usize {
        const k: *Hk = @ptrCast(@alignCast(ctx.?));
        if (k.nospace_writes > 0) {
            k.nospace_writes -= 1;
            return error.NoSpace;
        }
        var want = buf.len;
        if (k.short_intr) {
            k.flip = !k.flip;
            if (k.flip) return error.Interrupted;
            if (off % fsio.alignment == 0 and want > 1000) want = 1000;
        }
        const n = fsio.pwritePartial(fd, buf, want, off);
        if (n < 0) return error.Io;
        return @intCast(n);
    }

    fn freeBytes(ctx: ?*anyopaque) u64 {
        const k: *Hk = @ptrCast(@alignCast(ctx.?));
        return k.free orelse (1 << 40);
    }

    fn clear(k: *Hk) void {
        k.mutex.lockUncancelable(io);
        defer k.mutex.unlock(io);
        k.events.clearRetainingCapacity();
    }
};

/// A store over a fresh scratch directory and the test's fake device.
const H = struct {
    gpa: Allocator,
    path: [256]u8 = undefined,
    root: [:0]const u8 = undefined,
    bufs: [4][]align(slots.alignment) u8 = @splat(&.{}),
    lists: [4][1]slots.Buffer = undefined,
    cfg: store.Config = undefined,
    s: *Store = undefined,
    rows: std.AutoHashMapUnmanaged(u64, []u8) = .empty,
    hk: Hk,
    opened: bool = false,
    bpt: usize = bpt,

    fn init(gpa: Allocator, o: Opts) !*H {
        const h = try gpa.create(H);
        errdefer gpa.destroy(h);
        h.* = .{ .gpa = gpa, .hk = .{ .gpa = gpa, .free = o.free }, .bpt = o.bpt };
        h.hk.h = h;
        serial += 1;
        var tb: [64]u8 = undefined;
        const tag = try std.fmt.bufPrint(&tb, "{s}-{d}", .{ o.tag, serial });
        const r = try fsio.testRoot(&h.path, tag);
        h.root = r;
        const dfd = try fsio.freshDir(gpa, r);
        _ = c.close(dfd);
        const counts = [4]usize{ o.write_slots, o.read_slots, o.small_slots, o.restore_slots };
        const bb = block_tokens * o.bpt;
        const slot = std.mem.alignForward(usize, bb, slots.alignment);
        const per = [4]usize{ slot, slot, small_max, small_max };
        for (0..4) |i| {
            h.bufs[i] = try gpa.alignedAlloc(u8, .fromByteUnits(slots.alignment), @max(counts[i] * per[i], 0));
            h.lists[i] = .{h.bufs[i]};
        }
        h.cfg = .{
            .root = h.root, .prefix = o.prefix, .rank = 0, .compat = @splat(0x5a),
            .limits = .{ .max_tokens = max_tokens, .block_tokens = block_tokens, .block_bytes = bb, .max_small_bytes = small_max },
            .cap_bytes = o.cap, .floor_bytes = o.floor, .quota = o.quota, .turns = o.turns, .min_tokens = 16, .backlog_bytes = o.backlog,
            .staging = .{ .write = &h.lists[0], .read = &h.lists[1], .small = &h.lists[2], .restore = &h.lists[3] },
            .readers = 3, .max_reads = o.max_reads, .keep_builds = o.keep_builds,
            .hooks = .{ .ctx = &h.hk, .crash = Hk.crash, .event = Hk.event, .pwrite = Hk.pwrite, .free = Hk.freeBytes, .no_direct = o.no_direct },
        };
        try h.open();
        return h;
    }

    fn open(h: *H) !void {
        h.s = try Store.open(h.gpa, io, h.cfg);
        h.opened = true;
    }

    fn closeStore(h: *H) void {
        if (h.opened) h.s.close();
        h.opened = false;
    }

    fn reopen(h: *H) !void {
        h.closeStore();
        try h.open();
    }

    fn deinit(h: *H) void {
        h.closeStore();
        fsio.dropDir(h.gpa, h.root);
        for (h.bufs) |b| h.gpa.free(b);
        var it = h.rows.valueIterator();
        while (it.next()) |r| h.gpa.free(r.*);
        h.rows.deinit(h.gpa);
        h.hk.events.deinit(h.gpa);
        h.gpa.destroy(h);
    }

    /// Source `id`'s rows on the fake device (deterministic bytes, made on first use).
    fn src(h: *H, id: u64) []u8 {
        const gop = h.rows.getOrPut(h.gpa, id) catch @panic("oom");
        if (!gop.found_existing) {
            const r = h.gpa.alloc(u8, max_tokens * h.bpt) catch @panic("oom");
            var prng = std.Random.DefaultPrng.init(id *% 0x9e3779b97f4a7c15 +% 1);
            prng.random().bytes(r);
            gop.value_ptr.* = r;
        }
        return gop.value_ptr.*;
    }

    fn dir(h: *H, which: enum { blocks, points }) c.fd_t {
        return if (which == .blocks) h.s.blocks_fd else h.s.points_fd;
    }
};

fn smallByte(source: u64, at: usize, i: usize) u8 {
    return @truncate(i *% 131 +% source *% 7 +% at);
}

const KO = struct {
    lineage: Lineage,
    source: u64,
    gen: u64 = 1,
    shared: bool = false,
    group: u64 = 0,
    media: []const format.Media = &.{},
    raw: ?[]const u32 = null, // with media: the raw ids (the same length as the keyed ones)
    capture: bool = true,
};

/// The engine at a keep: `keep`, then the small state and tail captured from the device, reported.
fn keepIds(h: *H, ids: []const u32, o: KO) store.KeepResult {
    const at = ids.len;
    const tail_tokens = at % block_tokens;
    const r = h.s.keep(.{
        .keyed = ids, .raw = if (o.raw) |r| r[0..ids.len] else null, .media = o.media, .lineage = o.lineage, .shared = o.shared, .group = o.group, .sections = &secs,
        .tail_bytes = tail_tokens * h.bpt, .source = .{ .id = o.source, .generation = o.gen },
    });
    if (r == .queued) {
        const q = r.queued;
        for (q.sections) |sec| {
            for (0..@intCast(sec.len)) |i| q.small[@intCast(sec.offset + i)] = smallByte(o.source, at, i);
        }
        if (q.tail) |t| @memcpy(t, h.src(o.source)[(at - tail_tokens) * h.bpt .. at * h.bpt]);
        if (o.capture) h.s.captured(q.job, true);
    }
    return r;
}

/// The engine between rounds: every requested block packed from its source and reported.
fn pump(h: *H) void {
    var reqs: [16]store.StageReq = undefined;
    while (true) {
        const n = h.s.stage(&reqs);
        if (n == 0) return;
        for (reqs[0..n]) |r| {
            @memcpy(r.dst, h.src(r.source.id)[r.first * h.bpt ..][0..r.dst.len]);
            h.s.staged(r.ticket, true);
        }
    }
}

/// The engine until nothing waits on it: blocks staged as write slots come back, then a drain.
fn settle(h: *H) store.Store.Drain {
    var i: usize = 0;
    while (i < 5000) : (i += 1) {
        pump(h);
        if (h.s.pendingStage() == 0) break;
        std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    return h.s.drain(5 * std.time.ns_per_s);
}

/// Keep and write `ids` at once; the point's key once it is published.
fn write(h: *H, ids: []const u32, o: KO) !Digest {
    const r = keepIds(h, ids, o);
    try ensure(r == .queued, @src());
    const d = settle(h);
    try ensure(d.done, @src());
    const key = format.pointKey(ids, block_tokens, o.media);
    try ensure(h.s.presence(key), @src());
    return key;
}

/// A restore of `key` through the engine's side: rows (at * h.bpt bytes, gpa-owned) or null when it fails.
fn restore(h: *H, key: Digest) !?[]u8 {
    const id = h.s.beginRead(key) catch return null;
    const p = h.s.points.get(key) orelse return null;
    const at = p.at;
    const out = try h.gpa.alloc(u8, at * h.bpt);
    errdefer h.gpa.free(out);
    var i: u32 = 0;
    const total: u32 = @intCast(p.blocks.len);
    while (i < total) {
        if (!h.s.readAwait(id, i, 5 * std.time.ns_per_s, .{})) break;
        var rb: [4]store.ReadyBlock = undefined;
        const got = h.s.readReady(id, null, &rb);
        if (got.status == .failed and got.n == 0) break;
        for (rb[0..got.n]) |b| {
            @memcpy(out[b.first * h.bpt ..][0..b.data.len], b.data);
            h.s.readRelease(b.slot);
            i += 1;
        }
    }
    var small_ok = false;
    for (0..2500) |_| {
        if (h.s.readSmallView(id) != null) {
            small_ok = true;
            break;
        }
        if (h.s.readFailure(id).err != null) break;
        std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
    }
    if (i < total or !small_ok) {
        _ = h.s.endRead(id, .abort) catch {};
        h.gpa.free(out);
        return null;
    }
    var done = (h.s.endRead(id, .commit) catch {
        h.gpa.free(out);
        return null;
    }) orelse {
        h.gpa.free(out);
        return null;
    };
    done.lineage.deinit(h.gpa);
    return out;
}

/// A restore that fails, its failure forgotten the way every rank forgets it: true when it failed.
fn restoreFails(h: *H, key: Digest) !bool {
    const id = h.s.beginRead(key) catch return true;
    const total: u32 = @intCast((h.s.points.get(key) orelse return true).blocks.len);
    var failed = false;
    var i: u32 = 0;
    while (i < total) {
        _ = h.s.readAwait(id, i, 5 * std.time.ns_per_s, .{});
        var rb: [4]store.ReadyBlock = undefined;
        const got = h.s.readReady(id, null, &rb);
        for (rb[0..got.n]) |b| h.s.readRelease(b.slot);
        i += @intCast(got.n);
        if (got.status == .failed) {
            failed = true;
            break;
        }
        if (got.n == 0 and got.status == .waiting and h.s.readFailure(id).err != null) {
            failed = true;
            break;
        }
    }
    const f = h.s.readFailure(id);
    if (f.err != null) failed = true;
    if (failed) {
        const bad = try h.gpa.dupe(Digest, f.bad);
        defer h.gpa.free(bad);
        const lost = try h.gpa.dupe(Digest, f.lost);
        defer h.gpa.free(lost);
        h.s.forget(key, bad, lost);
    }
    _ = h.s.endRead(id, if (failed) .abort else .cancel) catch {};
    return failed;
}

fn idsOf(gpa: Allocator, n: usize, seed: u64) ![]u32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const ids = try gpa.alloc(u32, n);
    for (ids) |*x| x.* = prng.random().intRangeAtMost(u32, 1, 150000);
    return ids;
}

fn cat(gpa: Allocator, a: []const u32, b: []const u32) ![]u32 {
    const out = try gpa.alloc(u32, a.len + b.len);
    @memcpy(out[0..a.len], a);
    @memcpy(out[a.len..], b);
    return out;
}

fn lin(n: u8) Lineage {
    return .{ .root = @splat(n) };
}

fn manifestShared(h: *H, key: Digest) ?bool {
    var nb: [80]u8 = undefined;
    const h64 = format.hex(&key);
    const name = std.fmt.bufPrintSentinel(&nb, "{s}.man", .{h64}, 0) catch return null;
    const raw = fsio.readSmall(h.gpa, h.s.points_fd, name, 1 << 20) catch return null;
    defer h.gpa.free(raw);
    var m = format.Manifest.parse(h.gpa, raw, h.cfg.compat, h.cfg.limits) catch return null;
    defer m.deinit(h.gpa);
    return m.flags.shared;
}

fn fileExists(dirfd: c.fd_t, name: []const u8) bool {
    var nb: [128]u8 = undefined;
    const z = std.fmt.bufPrintSentinel(&nb, "{s}", .{name}, 0) catch return false;
    _ = fsio.statAt(dirfd, z) catch return false;
    return true;
}

fn blockFile(buf: []u8, name: Digest) []const u8 {
    const x = format.hex(&name);
    return std.fmt.bufPrint(buf, "{s}.blk", .{x}) catch unreachable;
}

fn pointFile(buf: []u8, key: Digest, suffix: []const u8) []const u8 {
    const x = format.hex(&key);
    return std.fmt.bufPrint(buf, "{s}{s}", .{ x, suffix }) catch unreachable;
}

fn countFiles(gpa: Allocator, dirfd: c.fd_t, temps_only: bool) usize {
    var names = fsio.list(gpa, dirfd) catch return 0;
    defer fsio.freeList(gpa, &names);
    var n: usize = 0;
    for (names.items) |x| {
        if (temps_only and x[0] != '.') continue;
        n += 1;
    }
    return n;
}

fn waitUntil(cond: anytype, ctx: anytype, ms: u32) bool {
    var i: u32 = 0;
    while (i < ms) : (i += 2) {
        if (cond(ctx)) return true;
        std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
    }
    return cond(ctx);
}

// -- the checks ------------------------------------------------------------------------------------------------------

/// GLM review 1: the block a kept state's end cuts holds the rows of the keep, not what the stream writes after
/// it; it is captured with the small state, never staged later from the source.
fn checkTail(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "tail" });
    defer h.deinit();
    const n = 2 * block_tokens + 6;
    const ids = try idsOf(gpa, n, 1);
    defer gpa.free(ids);
    const kept = try gpa.dupe(u8, h.src(1)[0 .. n * bpt]);
    defer gpa.free(kept);
    const r = keepIds(h, ids, .{ .lineage = lin(1), .source = 1 });
    try ensure(r == .queued, @src());
    const had_tail = r.queued.tail != null;
    // the stream goes on decoding in its extent past the keep: the tail block's rows change
    var prng = std.Random.DefaultPrng.init(9);
    prng.random().bytes(h.src(1)[2 * block_tokens * bpt .. (n + 8) * bpt]);
    try ensure(settle(h).done, @src());
    const got = (try restore(h, format.pointKey(ids, block_tokens, &.{}))) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, kept), @src());
    try ensure(had_tail and h.s.counts().staged_blocks == 2, @src());
}

/// The reader reads; the engine copies, only as far as it is told (LOADPART's boundary), the rest after.
fn checkUpto(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "upto" });
    defer h.deinit();
    const ids = try idsOf(gpa, 3 * block_tokens, 2);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(2), .source = 2 });
    const id = try h.s.beginRead(key);
    for (0..3) |i| try ensure(h.s.readAwait(id, @intCast(i), 5 * std.time.ns_per_s, .{}), @src()); // (readers finish in any order)
    var rb: [8]store.ReadyBlock = undefined;
    const a = h.s.readReady(id, 2, &rb);
    try ensure(a.n == 2 and a.status == .done, @src());
    for (rb[0..a.n]) |b| h.s.readRelease(b.slot);
    const b2 = h.s.readReady(id, null, &rb);
    try ensure(b2.n == 1 and b2.status == .done and rb[0].index == 2, @src());
    h.s.readRelease(rb[0].slot);
    try ensure(waitUntil(struct {
        fn f(x: struct { *H, store.ReadId }) bool {
            return x[0].s.readSmallView(x[1]) != null;
        }
    }.f, .{ h, id }, 5000), @src());
    const v = h.s.readSmallView(id).?;
    try ensure(v.sections.len == 2 and std.mem.eql(u8, v.sections[0].name.slice(), "ring"), @src());
    for (v.sections) |sec| for (0..@intCast(sec.len)) |i| {
        try ensure(v.bytes[@intCast(sec.offset + i)] == smallByte(2, 3 * block_tokens, i), @src());
    };
    var done = (try h.s.endRead(id, .commit)) orelse return CheckFailed;
    defer done.lineage.deinit(gpa);
    try ensure(done.at == 3 * block_tokens and done.lineage.eql(lin(2)), @src());
}

/// Last use survives a restart (the manifest's time): the least recently used conversation is still evicted first.
fn checkLastUse(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "lastuse" });
    defer h.deinit();
    const a = try idsOf(gpa, block_tokens, 2100);
    defer gpa.free(a);
    const b = try idsOf(gpa, block_tokens, 2101);
    defer gpa.free(b);
    const ka = try write(h, a, .{ .lineage = lin(21), .source = 21 });
    const kb = try write(h, b, .{ .lineage = lin(22), .source = 22 });
    std.Io.sleep(io, .fromMilliseconds(5), .awake) catch {};
    h.s.touch(ka);
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    try h.reopen();
    const order = h.s.evictionOrder() catch return CheckFailed;
    defer gpa.free(order);
    try ensure(order.len == 2 and std.mem.eql(u8, &order[0].key, &kb) and std.mem.eql(u8, &order[1].key, &ka), @src());
}

/// Keeps the store refuses say why and hold nothing: too short, invalid, out of staging, past the backlog.
fn checkRefusals(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "refuse", .small_slots = 1, .backlog = 3 * block_bytes + 2048 });
    defer h.deinit();
    const ids = try idsOf(gpa, 2 * block_tokens, 2200);
    defer gpa.free(ids);
    try ensure(keepIds(h, ids[0..8], .{ .lineage = lin(1), .source = 1 }) == .short, @src());
    const bad = h.s.keep(.{ .keyed = ids, .lineage = lin(1), .sections = &.{.{ .name = "a-section-name-longer-than-32-bytes", .len = 8 }}, .source = .{ .id = 1, .generation = 1 } });
    try ensure(bad == .invalid, @src());
    const tail = h.s.keep(.{ .keyed = ids[0 .. block_tokens + 3], .lineage = lin(1), .sections = &secs, .tail_bytes = 0, .source = .{ .id = 1, .generation = 1 } });
    try ensure(tail == .invalid, @src());
    const r1 = keepIds(h, ids, .{ .lineage = lin(1), .source = 1, .capture = false });
    try ensure(r1 == .queued, @src());
    const other = try idsOf(gpa, block_tokens, 2201);
    defer gpa.free(other);
    try ensure(keepIds(h, other, .{ .lineage = lin(2), .source = 2 }) == .no_staging, @src());
    h.s.captured(r1.queued.job, true);
    try ensure(settle(h).done, @src());
    try ensure(keepIds(h, other, .{ .lineage = lin(2), .source = 2 }) == .queued, @src());
    try ensure(settle(h).done, @src());
    const big = try idsOf(gpa, 4 * block_tokens, 2202);
    defer gpa.free(big);
    try ensure(keepIds(h, big, .{ .lineage = lin(3), .source = 3 }) == .backlog, @src());
    const ct = h.s.counts();
    try ensure(ct.short == 1 and ct.invalid == 2 and ct.no_staging == 1 and ct.backlog_refused == 1 and ct.staging_bytes == 0, @src());
    try ensure(ct.committed == 2 and ct.points == 2, @src());
}

/// Shared flag 3: the same ids kept again as a shared prefix make the stored point shared, its manifest rewritten
/// through a temporary name, fsynced, renamed, then the folder fsynced; a restart reads it shared.
fn checkSharedUpgrade(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "upgrade" });
    defer h.deinit();
    const ids = try idsOf(gpa, block_tokens + 5, 51);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(51), .source = 51 });
    h.hk.clear();
    try ensure(keepIds(h, ids, .{ .lineage = lin(51), .source = 51, .shared = true }) == .stored, @src());
    try ensure(h.s.points.get(key).?.shared, @src());
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    try ensure(manifestShared(h, key) == true, @src());
    // the rewrite's own events: written to a temporary name, fsynced, pages dropped, renamed, the folder fsynced
    h.hk.mutex.lockUncancelable(io);
    var saw: [8]fsio.Event = undefined;
    var k: usize = 0;
    for (h.hk.events.items) |e| if (std.mem.endsWith(u8, e.name[0..e.len], ".man") or std.mem.eql(u8, e.name[0..e.len], "points")) {
        if (k < saw.len) saw[k] = e.ev;
        k += 1;
    };
    h.hk.mutex.unlock(io);
    try ensure(k == 5 and std.mem.eql(fsio.Event, saw[0..5], &.{ .write, .fsync, .drop_pages, .rename, .dir_fsync }), @src());
    try h.reopen();
    const p = h.s.points.get(key) orelse return CheckFailed;
    try ensure(p.shared, @src());
}

/// ... also when they were still waiting to be written.
fn checkSharedWaiting(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "waiting" });
    defer h.deinit();
    const ids = try idsOf(gpa, block_tokens + 5, 52);
    defer gpa.free(ids);
    try ensure(keepIds(h, ids, .{ .lineage = lin(52), .source = 52 }) == .queued, @src());
    try ensure(keepIds(h, ids, .{ .lineage = lin(52), .source = 52, .shared = true }) == .stored, @src());
    try ensure(settle(h).done, @src());
    const key = format.pointKey(ids, block_tokens, &.{});
    try ensure((h.s.points.get(key) orelse return CheckFailed).shared and manifestShared(h, key) == true, @src());
}

fn markWhileWriting(h: *H) void {
    const ids = h.rows.get(0xdead).?; // the ids stashed as bytes
    const s: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, ids));
    _ = keepIds(h, s, .{ .lineage = lin(53), .source = 53, .shared = true });
}

/// ... and when kept again as shared while its own files were being written: its manifest is rewritten shared.
fn checkSharedDuringWrite(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "during" });
    defer h.deinit();
    const ids = try idsOf(gpa, block_tokens + 5, 53);
    defer gpa.free(ids);
    try h.rows.put(gpa, 0xdead, try gpa.dupe(u8, std.mem.sliceAsBytes(ids)));
    h.hk.action_step = .written;
    h.hk.action_suffix = ".ids";
    h.hk.action = markWhileWriting;
    const key = try write(h, ids, .{ .lineage = lin(53), .source = 53 });
    try ensure(h.hk.action == null, @src()); // (it ran)
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    try ensure((h.s.points.get(key) orelse return CheckFailed).shared and manifestShared(h, key) == true, @src());
}

var drop_key: Digest = undefined;
fn dropDuring(h: *H) void {
    h.s.forget(drop_key, &.{}, &.{});
}

/// A point dropped while its manifest is rewritten does not get the manifest back (no resurrection).
fn checkDropDuringRewrite(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "resurrect" });
    defer h.deinit();
    const ids = try idsOf(gpa, block_tokens + 5, 54);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(54), .source = 54 });
    drop_key = key;
    h.hk.action_step = .written;
    h.hk.action_suffix = ".man";
    h.hk.action = dropDuring;
    try ensure(keepIds(h, ids, .{ .lineage = lin(54), .source = 54, .shared = true }) == .stored, @src());
    _ = h.s.drain(5 * std.time.ns_per_s);
    try ensure(h.hk.action == null and !h.s.points.contains(key), @src());
    var nb: [96]u8 = undefined;
    try ensure(!fileExists(h.s.points_fd, pointFile(&nb, key, ".man")), @src());
}

fn slowRewrite(_: *H) void {
    std.Io.sleep(io, .fromMilliseconds(300), .awake) catch {};
}

/// A drain (a clean stop) waits for a queued metadata rewrite.
fn checkDrainWaitsRewrite(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "drainrw" });
    defer h.deinit();
    const ids = try idsOf(gpa, block_tokens + 5, 55);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(55), .source = 55 });
    h.hk.action_step = .written;
    h.hk.action_suffix = ".man";
    h.hk.action = slowRewrite;
    try ensure(keepIds(h, ids, .{ .lineage = lin(55), .source = 55, .shared = true }) == .stored, @src());
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    try ensure(manifestShared(h, key) == true, @src());
}

/// A (2 blocks), B (other ids, the same rows: its blocks links to A's) and C (A's ids and a block more, A's lineage:
/// it names A's blocks).
fn linked(h: *H) ![3]Digest {
    const gpa = h.gpa;
    const ia = try idsOf(gpa, 2 * block_tokens, 61);
    defer gpa.free(ia);
    const ib = try idsOf(gpa, 2 * block_tokens, 62);
    defer gpa.free(ib);
    const more = try idsOf(gpa, block_tokens, 63);
    defer gpa.free(more);
    const ic = try cat(gpa, ia, more);
    defer gpa.free(ic);
    const ka = try write(h, ia, .{ .lineage = lin(1), .source = 60 });
    const kb = try write(h, ib, .{ .lineage = lin(2), .source = 60 });
    const kc = try write(h, ic, .{ .lineage = lin(1), .source = 60 });
    return .{ ka, kb, kc };
}

/// GLM review 4: a read failing on a missing block file drops that name and the points naming it, not the points
/// whose blocks are links to the same bytes (they still read back).
fn checkForgetLost(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "lost" });
    defer h.deinit();
    const k = try linked(h);
    const pa = h.s.points.get(k[0]) orelse return CheckFailed;
    const pb = h.s.points.get(k[1]) orelse return CheckFailed;
    try ensure(h.s.counts().dedup_blocks == 2, @src());
    for (0..2) |i| try ensure(h.s.have.get(pa.blocks[i].name).?.ino == h.s.have.get(pb.blocks[i].name).?.ino, @src());
    const pb_names = [2]Digest{ pb.blocks[0].name, pb.blocks[1].name };
    var nb: [96]u8 = undefined;
    var zb: [96]u8 = undefined;
    _ = c.unlinkat(h.s.blocks_fd, try std.fmt.bufPrintSentinel(&zb, "{s}", .{blockFile(&nb, pa.blocks[0].name)}, 0), 0);
    try ensure(try restoreFails(h, k[0]), @src());
    try ensure(!h.s.points.contains(k[0]) and !h.s.points.contains(k[2]), @src());
    try ensure(h.s.points.contains(k[1]), @src());
    _ = h.s.drain(5 * std.time.ns_per_s);
    for (pb_names) |n| try ensure(fileExists(h.s.blocks_fd, blockFile(&nb, n)), @src());
    const got = (try restore(h, k[1])) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(60)[0 .. 2 * block_tokens * bpt]), @src());
}

/// ... one failing on damaged bytes drops every name of them (A, B and C).
fn checkForgetDamaged(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "damaged" });
    defer h.deinit();
    const k = try linked(h);
    const pa = h.s.points.get(k[0]) orelse return CheckFailed;
    const pb = h.s.points.get(k[1]) orelse return CheckFailed;
    const b0 = pb.blocks[0].name;
    // C names A's blocks and is being read when A's read fails: it goes too (unlisted, freed when its read ends)
    const rc = try h.s.beginRead(k[2]);
    var nb: [96]u8 = undefined;
    var zb: [96]u8 = undefined;
    const name = try std.fmt.bufPrintSentinel(&zb, "{s}", .{blockFile(&nb, pa.blocks[0].name)}, 0);
    const fd = c.openat(h.s.blocks_fd, name, .{ .ACCMODE = .WRONLY, .CLOEXEC = true }, @as(c.mode_t, 0));
    try ensure(fd >= 0, @src());
    const junk: [64]u8 = @splat(0xff);
    _ = c.pwrite(fd, &junk, junk.len, 0);
    _ = c.close(fd);
    try ensure(try restoreFails(h, k[0]), @src());
    const gone = !h.s.points.contains(k[0]) and !h.s.points.contains(k[1]) and !h.s.points.contains(k[2]);
    _ = h.s.endRead(rc, .cancel) catch {};
    try ensure(gone, @src());
    try ensure(h.s.counts().points == 0, @src());
    _ = h.s.drain(5 * std.time.ns_per_s);
    var nb2: [96]u8 = undefined;
    try ensure(!fileExists(h.s.blocks_fd, blockFile(&nb2, b0)), @src());
}

/// A point's small files (.ids, .st, .man) each have their pages dropped after their fsync.
fn checkPages(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "pages" });
    defer h.deinit();
    const ids = try idsOf(gpa, block_tokens + 7, 71);
    defer gpa.free(ids);
    h.hk.clear();
    const key = try write(h, ids, .{ .lineage = lin(71), .source = 71 });
    var nb: [96]u8 = undefined;
    for ([_][]const u8{ ".ids", ".st", ".man" }) |suffix| {
        const want = pointFile(&nb, key, suffix);
        var seq: [4]fsio.Event = undefined;
        var k: usize = 0;
        h.hk.mutex.lockUncancelable(io);
        for (h.hk.events.items) |e| if (std.mem.eql(u8, e.name[0..e.len], want) and (e.ev == .fsync or e.ev == .drop_pages)) {
            if (k < seq.len) seq[k] = e.ev;
            k += 1;
        };
        h.hk.mutex.unlock(io);
        try ensure(k == 2 and seq[0] == .fsync and seq[1] == .drop_pages, @src());
    }
}

/// Publication order: every block renamed and its folder fsynced before the point's files; the ids and small state
/// renamed and the points folder fsynced before the manifest, which is renamed last.
fn checkPublishOrder(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "order" });
    defer h.deinit();
    const ids = try idsOf(gpa, 3 * block_tokens, 72);
    defer gpa.free(ids);
    h.hk.clear();
    const key = try write(h, ids, .{ .lineage = lin(72), .source = 72 });
    var nb: [96]u8 = undefined;
    const man = pointFile(&nb, key, ".man");
    var last_blk: ?usize = null;
    var blocks_sync: ?usize = null;
    var ids_ren: ?usize = null;
    var st_ren: ?usize = null;
    var points_sync: ?usize = null;
    var man_ren: ?usize = null;
    h.hk.mutex.lockUncancelable(io);
    for (h.hk.events.items, 0..) |e, i| {
        const n = e.name[0..e.len];
        if (e.ev == .rename and std.mem.endsWith(u8, n, ".blk")) last_blk = i;
        if (e.ev == .dir_fsync and std.mem.eql(u8, n, "blocks") and blocks_sync == null) blocks_sync = i;
        if (e.ev == .rename and std.mem.endsWith(u8, n, ".ids")) ids_ren = i;
        if (e.ev == .rename and std.mem.endsWith(u8, n, ".st")) st_ren = i;
        if (e.ev == .dir_fsync and std.mem.eql(u8, n, "points") and points_sync == null and st_ren != null) points_sync = i;
        if (e.ev == .rename and std.mem.eql(u8, n, man)) man_ren = i;
    }
    h.hk.mutex.unlock(io);
    const all = last_blk != null and blocks_sync != null and ids_ren != null and st_ren != null and points_sync != null and man_ren != null;
    try ensure(all, @src());
    try ensure(last_blk.? < blocks_sync.? and blocks_sync.? < ids_ren.? and ids_ren.? < st_ren.? and st_ren.? < points_sync.? and points_sync.? < man_ren.?, @src());
}

const Pending = struct {
    h: *H,
    held: std.ArrayList(slots.Slot) = .empty,

    fn recycle(ctx: ?*anyopaque) void {
        const p: *Pending = @ptrCast(@alignCast(ctx.?));
        for (p.held.items) |s| p.h.s.readRelease(s);
        p.held.clearRetainingCapacity();
    }
};

/// PR 124's hang: a follower whose peer failed early (LOADED with nothing copied before) and more blocks left than
/// read slots. Its copies complete only when it recycles; its wait for a block gives those slots back each time.
fn checkHang(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "hang", .read_slots = 4 });
    defer h.deinit();
    const n = 12 * block_tokens;
    const ids = try idsOf(gpa, n, 501);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(50), .source = 500 });
    const id = try h.s.beginRead(key);
    var pend: Pending = .{ .h = h };
    defer pend.held.deinit(gpa);
    const out = try gpa.alloc(u8, n * bpt);
    defer gpa.free(out);
    var ok = true;
    for (0..12) |i| {
        if (!h.s.readAwait(id, @intCast(i), 2 * std.time.ns_per_s, .{ .ctx = &pend, .recycle = Pending.recycle })) {
            ok = false;
            break;
        }
        var rb: [1]store.ReadyBlock = undefined;
        const got = h.s.readReady(id, @intCast(i + 1), &rb);
        if (got.n != 1) {
            ok = false;
            break;
        }
        @memcpy(out[rb[0].first * bpt ..][0..rb[0].data.len], rb[0].data);
        try pend.held.append(gpa, rb[0].slot); // its copy's event not complete yet
    }
    Pending.recycle(&pend);
    if (!ok) _ = h.s.endRead(id, .abort) catch {};
    try ensure(ok, @src());
    _ = waitUntil(struct {
        fn f(x: struct { *H, store.ReadId }) bool {
            return x[0].s.readSmallView(x[1]) != null;
        }
    }.f, .{ h, id }, 5000);
    var done = (h.s.endRead(id, .commit) catch return CheckFailed) orelse return CheckFailed;
    done.lineage.deinit(gpa);
    try ensure(std.mem.eql(u8, out, h.src(500)[0 .. n * bpt]), @src());
}

/// Read-ahead accounting: preallocated buffers are kept and only the shortfall is charged.
fn checkCarve(gpa: Allocator) !void {
    const per = 3 * slots.alignment;
    const pre = try gpa.alignedAlloc(u8, .fromByteUnits(slots.alignment), 2 * per + per / 2);
    defer gpa.free(pre);
    try ensure(slots.extraBytes(&.{pre}, per, 7) == 5 * per, @src());
    try ensure(slots.extraBytes(&.{pre}, per, 2) == 0, @src());
}

const Slow = struct {
    h: *H,
    calls: std.ArrayList(store.ReadId) = .empty,

    fn copy(ctx: ?*anyopaque, id: store.ReadId, b: store.ReadyBlock) bool {
        const s: *Slow = @ptrCast(@alignCast(ctx.?));
        s.calls.append(s.h.gpa, id) catch {};
        std.Io.sleep(io, .fromMilliseconds(30), .awake) catch {};
        s.h.s.readRelease(b.slot);
        return true;
    }
};

/// GLM review (wojo#2): the restores in flight share one copy budget a pause, each pause starting with the next.
fn checkBudget(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "budget", .read_slots = 16, .restore_slots = 3, .max_reads = 3 });
    defer h.deinit();
    var ids: [3]store.ReadId = undefined;
    for (0..3) |i| {
        const x = try idsOf(gpa, 5 * block_tokens, 80 + i);
        defer gpa.free(x);
        const key = try write(h, x, .{ .lineage = lin(@intCast(80 + i)), .source = 80 + i });
        ids[i] = try h.s.beginRead(key);
    }
    for (ids) |id| for (0..5) |i| try ensure(h.s.readAwait(id, @intCast(i), 5 * std.time.ns_per_s, .{}), @src());
    var slow: Slow = .{ .h = h };
    defer slow.calls.deinit(gpa);
    var turn: usize = 0;
    _ = h.s.stepReads(&ids, &turn, 40 * std.time.ns_per_ms, &slow, Slow.copy);
    // copies counted, not wall time (a loaded machine stretches each 30 ms sleep, which let the every-read-its-own-
    // budget mutant fit under a time bound): one shared 40 ms budget takes 1 or 2 copies; a budget each read takes
    // at least one copy per read, 3
    const first_calls = slow.calls.items.len;
    var firsts: [3]store.ReadId = undefined;
    for (0..3) |k| {
        slow.calls.clearRetainingCapacity();
        _ = h.s.stepReads(&ids, &turn, 40 * std.time.ns_per_ms, &slow, Slow.copy);
        firsts[k] = if (slow.calls.items.len > 0) slow.calls.items[0] else 0;
    }
    for (ids) |id| _ = h.s.endRead(id, .cancel) catch {};
    try ensure(first_calls >= 1 and first_calls <= 2, @src()); // (every read the whole budget: 3 or more)
    std.mem.sort(store.ReadId, &firsts, {}, std.sort.asc(store.ReadId));
    var want = ids;
    std.mem.sort(store.ReadId, &want, {}, std.sort.asc(store.ReadId));
    try ensure(std.mem.eql(store.ReadId, &firsts, &want), @src());
}

const CrashMode = enum {
    publish, // a new point's publication
    link, // ... whose first blocks equal a stored point's bytes (written as hard links)
    rewrite, // a published point's manifest rewritten with the shared flag
};

const crash_points = [_]struct { step: fsio.Step, suffix: []const u8, skip: u32 = 0, mode: CrashMode = .publish }{
    .{ .step = .written, .suffix = ".blk" },
    .{ .step = .synced, .suffix = ".blk" },
    .{ .step = .renamed, .suffix = ".blk" },
    .{ .step = .renamed, .suffix = ".blk", .skip = 1 },
    .{ .step = .dir_synced, .suffix = "blocks" },
    .{ .step = .written, .suffix = ".ids" },
    .{ .step = .renamed, .suffix = ".ids" },
    .{ .step = .written, .suffix = ".st" },
    .{ .step = .renamed, .suffix = ".st" },
    .{ .step = .dir_synced, .suffix = "points" },
    .{ .step = .written, .suffix = ".man" },
    .{ .step = .synced, .suffix = ".man" },
    .{ .step = .renamed, .suffix = ".man" },
    .{ .step = .published, .suffix = "points" },
    // PORT-SPILL-V2 stage 5: while a block is linked, and while shared metadata is rewritten
    .{ .step = .linked, .suffix = ".blk", .mode = .link },
    .{ .step = .linked, .suffix = ".blk", .skip = 1, .mode = .link },
    .{ .step = .written, .suffix = ".man", .mode = .rewrite },
    .{ .step = .synced, .suffix = ".man", .mode = .rewrite },
    .{ .step = .renamed, .suffix = ".man", .mode = .rewrite },
    .{ .step = .rewritten, .suffix = ".man", .mode = .rewrite },
};

fn crashed(h: *H) bool {
    return h.s.isCrashed();
}

/// The durability fault campaign: a kill after every write, fsync and rename of a point's publication, after a
/// dedup link is made (not yet renamed), and at every step of a shared-flag manifest rewrite. On restart an earlier
/// point verifies and restores; the cut point is either whole (published, shared or not) or gone; no temporary file,
/// orphan block or slot is left.
fn checkCrashes(gpa: Allocator) !void {
    for (crash_points, 0..) |cp, ci| {
        const h = try H.init(gpa, .{ .tag = "crash" });
        defer h.deinit();
        const a = try idsOf(gpa, 2 * block_tokens, 900 + ci);
        defer gpa.free(a);
        const ka = try write(h, a, .{ .lineage = lin(90), .source = 90 });
        const b = try idsOf(gpa, 3 * block_tokens + 3, 950 + ci);
        defer gpa.free(b);
        const src_b: u64 = if (cp.mode == .link) 90 else 91; // link: B's rows are A's (other ids, other names)
        var kb = format.pointKey(b, block_tokens, &.{});
        if (cp.mode == .rewrite) kb = try write(h, b, .{ .lineage = lin(91), .source = src_b });
        h.hk.crash_step = cp.step;
        h.hk.crash_suffix = cp.suffix;
        h.hk.crash_skip = cp.skip;
        if (cp.mode == .rewrite) {
            h.s.markShared(kb);
        } else {
            try ensure(keepIds(h, b, .{ .lineage = lin(91), .source = src_b }) == .queued, @src());
            pump(h);
        }
        if (!waitUntil(crashed, h, 3000)) {
            if (!quiet) std.debug.print("     no crash at {t} {s} (+{d}, {t})\n", .{ cp.step, cp.suffix, cp.skip, cp.mode });
            return CheckFailed;
        }
        const temps_before = countFiles(gpa, h.s.blocks_fd, true) + countFiles(gpa, h.s.points_fd, true);
        h.hk.crash_step = null;
        try h.reopen();
        try ensure(countFiles(gpa, h.s.blocks_fd, true) == 0 and countFiles(gpa, h.s.points_fd, true) == 0, @src());
        if (temps_before > 0) try ensure(h.s.counts().temps_removed >= temps_before, @src());
        try ensure(countFiles(gpa, h.s.blocks_fd, false) == h.s.have.count(), @src());
        try ensure(countFiles(gpa, h.s.points_fd, false) == 3 * h.s.points.count(), @src());
        const got = (try restore(h, ka)) orelse return CheckFailed;
        defer gpa.free(got);
        try ensure(std.mem.eql(u8, got, h.src(90)[0 .. 2 * block_tokens * bpt]), @src());
        const published = cp.mode == .rewrite or (cp.step == .renamed and std.mem.eql(u8, cp.suffix, ".man")) or cp.step == .published;
        try ensure(h.s.presence(kb) == published, @src());
        if (published) {
            const gb = (try restore(h, kb)) orelse return CheckFailed;
            defer gpa.free(gb);
            try ensure(std.mem.eql(u8, gb, h.src(src_b)[0 .. b.len * bpt]), @src());
        }
        if (cp.mode == .rewrite) { // whole either way; durably shared once the rewrite's directory fsync ran
            const on_disk = manifestShared(h, kb) orelse return CheckFailed;
            try ensure((h.s.points.get(kb) orelse return CheckFailed).shared == on_disk, @src());
            if (cp.step == .rewritten) try ensure(on_disk, @src());
            if (cp.step == .written or cp.step == .synced) try ensure(!on_disk, @src());
        }
    }
}

/// Recovery: a truncated block, a missing small state and a damaged manifest each drop only their point; a stale
/// temporary file and unreferenced blocks are collected.
fn checkRecovery(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "recover" });
    defer h.deinit();
    var keys: [4]Digest = undefined;
    for (0..4) |i| {
        const x = try idsOf(gpa, 2 * block_tokens + 4, 300 + i);
        defer gpa.free(x);
        keys[i] = try write(h, x, .{ .lineage = lin(@intCast(30 + i)), .source = 300 + i });
    }
    const p0 = h.s.points.get(keys[0]) orelse return CheckFailed;
    var nb: [96]u8 = undefined;
    var zb: [96]u8 = undefined;
    const bname = try std.fmt.bufPrintSentinel(&zb, "{s}", .{blockFile(&nb, p0.blocks[0].name)}, 0);
    const fd = c.openat(h.s.blocks_fd, bname, .{ .ACCMODE = .WRONLY, .CLOEXEC = true }, @as(c.mode_t, 0));
    _ = c.ftruncate(fd, block_bytes - 100);
    _ = c.close(fd);
    var nb2: [96]u8 = undefined;
    var zb2: [96]u8 = undefined;
    _ = c.unlinkat(h.s.points_fd, try std.fmt.bufPrintSentinel(&zb2, "{s}", .{pointFile(&nb2, keys[1], ".st")}, 0), 0);
    const mname = try std.fmt.bufPrintSentinel(&zb2, "{s}", .{pointFile(&nb2, keys[2], ".man")}, 0);
    const mfd = c.openat(h.s.points_fd, mname, .{ .ACCMODE = .WRONLY, .CLOEXEC = true }, @as(c.mode_t, 0));
    const junk: [4]u8 = @splat(0x33);
    _ = c.pwrite(mfd, &junk, 4, 40);
    _ = c.close(mfd);
    const tfd = c.openat(h.s.blocks_fd, ".stale.blk.1.1.tmp", .{ .ACCMODE = .WRONLY, .CREAT = true, .CLOEXEC = true }, @as(c.mode_t, 0o600));
    _ = c.close(tfd);
    try h.reopen();
    try ensure(!h.s.points.contains(keys[0]) and !h.s.points.contains(keys[1]) and !h.s.points.contains(keys[2]), @src());
    try ensure(h.s.points.contains(keys[3]), @src());
    try ensure(h.s.counts().recovered_dropped == 3 and h.s.counts().temps_removed >= 1, @src());
    try ensure(countFiles(gpa, h.s.blocks_fd, false) == h.s.have.count() and h.s.have.count() == 3, @src());
    const got = (try restore(h, keys[3])) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(303)[0 .. (2 * block_tokens + 4) * bpt]), @src());
}

/// CRC-32 forging: the 4 bytes that make `data[0 .. len - 4] ++ them` hash to `target` (reflected, ISO-HDLC).
fn forgeCrc(data: []u8, target: u32) void {
    var table: [256]u32 = undefined;
    for (&table, 0..) |*t, i| {
        var x: u32 = @intCast(i);
        for (0..8) |_| x = if (x & 1 != 0) (x >> 1) ^ 0xEDB88320 else x >> 1;
        t.* = x;
    }
    const head = data[0 .. data.len - 4];
    const before = ~std.hash.Crc32.hash(head); // the register after the head
    var x: u32 = ~target; // the register wanted after the last 4 bytes
    for (0..4) |_| {
        const top: u8 = @truncate(x >> 24);
        var idx: u32 = 0;
        for (table, 0..) |t, i| if (@as(u8, @truncate(t >> 24)) == top) {
            idx = @intCast(i);
            break;
        };
        x = ((x ^ table[idx]) << 8) | idx;
    }
    std.mem.writeInt(u32, data[data.len - 4 ..][0..4], x ^ before, .little);
}

/// Dedup links only equal bytes: a block with the same CRC-32 and length but other bytes is written, not linked.
fn checkDedupSha(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "dedup" });
    defer h.deinit();
    const ia = try idsOf(gpa, block_tokens, 41);
    defer gpa.free(ia);
    const ib = try idsOf(gpa, block_tokens, 42);
    defer gpa.free(ib);
    const ka = try write(h, ia, .{ .lineage = lin(41), .source = 41 });
    const want = std.hash.Crc32.hash(h.src(41)[0..block_bytes]);
    const rows = h.src(42);
    @memcpy(rows[0..block_bytes], h.src(41)[0..block_bytes]);
    rows[17] ^= 0x5a;
    forgeCrc(rows[0..block_bytes], want);
    try ensure(std.hash.Crc32.hash(rows[0..block_bytes]) == want, @src());
    const kb = try write(h, ib, .{ .lineage = lin(42), .source = 42 });
    const pa = h.s.points.get(ka) orelse return CheckFailed;
    const pb = h.s.points.get(kb) orelse return CheckFailed;
    try ensure(h.s.have.get(pa.blocks[0].name).?.ino != h.s.have.get(pb.blocks[0].name).?.ino, @src());
    const got = (try restore(h, kb)) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, rows[0..block_bytes]), @src());
}

/// A guard before source rows change: the point's blocks still to stage are staged now (the engine queues those
/// copies before the change), never cancelled while slots allow.
fn checkGuard(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "guard" });
    defer h.deinit();
    const ids = try idsOf(gpa, 4 * block_tokens, 7);
    defer gpa.free(ids);
    const kept = try gpa.dupe(u8, h.src(5)[0 .. 4 * block_tokens * bpt]);
    defer gpa.free(kept);
    try ensure(keepIds(h, ids, .{ .lineage = lin(7), .source = 5 }) == .queued, @src());
    try ensure(h.s.pins(.{ .id = 5, .generation = 1 }, 0), @src());
    var any: [3]bool = undefined;
    h.s.pinsAny(&.{ 6, 5, 9 }, &any); // (pins for many sources, one lock)
    try ensure(!any[0] and any[1] and !any[2], @src());
    var out: [8]store.StageReq = undefined;
    const g = h.s.guard(.{ .id = 5, .generation = 1 }, 0, &out);
    try ensure(g.staged == 4 and g.cancelled == 0, @src());
    h.s.pinsAny(&.{ 6, 5, 9 }, &any); // (every block staged: nothing left to pin, as pins says)
    try ensure(!any[1] and any[1] == h.s.pins(.{ .id = 5, .generation = 1 }, 0), @src());
    for (out[0..g.staged]) |r| @memcpy(r.dst, h.src(5)[r.first * bpt ..][0..r.dst.len]); // queued before the change
    @memset(h.src(5)[0 .. 4 * block_tokens * bpt], 0xee); // the change
    for (out[0..g.staged]) |r| h.s.staged(r.ticket, true);
    try ensure(settle(h).done, @src());
    const got = (try restore(h, format.pointKey(ids, block_tokens, &.{}))) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, kept), @src());
}

/// ... past the room it has, or for an older generation of the source, the unpublished point is cancelled, its
/// staging back, a completion saying why.
fn checkGuardCancel(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "guardc" });
    defer h.deinit();
    const ids = try idsOf(gpa, 4 * block_tokens, 8);
    defer gpa.free(ids);
    try ensure(keepIds(h, ids, .{ .lineage = lin(8), .source = 6 }) == .queued, @src());
    var out: [2]store.StageReq = undefined;
    const g = h.s.guard(.{ .id = 6, .generation = 1 }, 0, &out);
    try ensure(g.staged == 0 and g.cancelled == 1, @src());
    const ids2 = try idsOf(gpa, 2 * block_tokens, 9);
    defer gpa.free(ids2);
    try ensure(keepIds(h, ids2, .{ .lineage = lin(9), .source = 7, .gen = 1 }) == .queued, @src());
    var o2: [8]store.StageReq = undefined;
    const g2 = h.s.guard(.{ .id = 7, .generation = 2 }, 0, &o2);
    try ensure(g2.cancelled == 1, @src());
    var comp: [4]store.Completion = undefined;
    const n = h.s.poll(&comp);
    try ensure(n == 2 and comp[0] == .cancelled and comp[0].cancelled.why == .guard and comp[1].cancelled.why == .source_gone, @src());
    _ = settle(h);
    const ct = h.s.counts();
    try ensure(ct.staging_bytes == 0 and ct.points == 0 and ct.pending_jobs == 0, @src());
}

/// The staging cap holds for a point larger than all of staging (the GPU's 250k-token point is ~220 MB against a
/// 128 MiB cap): its blocks are staged a few slots at a time as the writer frees them, its restore reads through a
/// few read slots, a guard that cannot stage what is left cancels instead of overshooting, and the peak counter is
/// what was held at once (at most the slots carved), never more.
fn checkStagingCap(gpa: Allocator) !void {
    const o: Opts = .{ .tag = "stagecap", .write_slots = 3, .read_slots = 2, .small_slots = 2, .restore_slots = 1 };
    const cap: u64 = (o.write_slots + o.read_slots) * block_bytes + (o.small_slots + o.restore_slots) * small_max;
    const h = try H.init(gpa, o);
    defer h.deinit();
    const ids = try idsOf(gpa, max_tokens, 3100); // 64 blocks: 256 KiB against 44 KiB of staging
    defer gpa.free(ids);
    try ensure(max_tokens * bpt > 4 * cap, @src());
    try ensure(keepIds(h, ids, .{ .lineage = lin(31), .source = 310 }) == .queued, @src());
    var reqs: [16]store.StageReq = undefined;
    var staged: usize = 0;
    var i: usize = 0;
    while (i < 20000 and h.s.pendingStage() > 0) : (i += 1) {
        const n = h.s.stage(&reqs);
        try ensure(n <= o.write_slots, @src());
        try ensure(h.s.counts().staging_bytes <= cap, @src());
        for (reqs[0..n]) |r| {
            @memcpy(r.dst, h.src(310)[r.first * bpt ..][0..r.dst.len]);
            h.s.staged(r.ticket, true);
        }
        staged += n;
        if (n == 0) std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
    }
    try ensure(staged == max_tokens / block_tokens, @src());
    try ensure(h.s.drain(5 * std.time.ns_per_s).done, @src());
    const key = format.pointKey(ids, block_tokens, &.{});
    try ensure(h.s.presence(key), @src());
    const got = (try restore(h, key)) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(310)[0 .. max_tokens * bpt]), @src());
    // a second large point guarded before any of it staged: more blocks left than free write slots, so it is cancelled
    const ids2 = try idsOf(gpa, max_tokens, 3101);
    defer gpa.free(ids2);
    try ensure(keepIds(h, ids2, .{ .lineage = lin(32), .source = 311 }) == .queued, @src());
    var out: [64]store.StageReq = undefined;
    const g = h.s.guard(.{ .id = 311, .generation = 1 }, 0, &out);
    try ensure(g.staged == 0 and g.cancelled == 1, @src());
    _ = settle(h);
    const ct = h.s.counts();
    try ensure(ct.staging_peak_bytes <= cap and ct.staging_peak_bytes >= block_bytes, @src());
    try ensure(ct.staging_bytes == 0 and ct.guard_staged == 0 and ct.cancelled_guard == 1, @src());
}

/// 0110's eviction order: old turns, abandoned forks, whole conversations least recently used first (longest state
/// first within one), shared prefixes last.
fn checkEvictionOrder(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "evict", .turns = 2 });
    defer h.deinit();
    const sys = try idsOf(gpa, 2 * block_tokens, 1000);
    defer gpa.free(sys);
    const r1 = try idsOf(gpa, block_tokens, 1001);
    defer gpa.free(r1);
    const r2 = try idsOf(gpa, block_tokens, 1002);
    defer gpa.free(r2);
    const r3 = try idsOf(gpa, block_tokens, 1003);
    defer gpa.free(r3);
    const r4 = try idsOf(gpa, block_tokens, 1004);
    defer gpa.free(r4);
    const b1 = try cat(gpa, sys, r1);
    defer gpa.free(b1);
    const a1 = try cat(gpa, sys, r2);
    defer gpa.free(a1);
    const a2f = try cat(gpa, a1, r3);
    defer gpa.free(a2f);
    const a2 = try cat(gpa, a1, r4);
    defer gpa.free(a2);
    const ks = try write(h, sys, .{ .lineage = lin(1), .source = 1, .shared = true });
    const kb1 = try write(h, b1, .{ .lineage = lin(2), .source = 2 });
    const ka1 = try write(h, a1, .{ .lineage = lin(3), .source = 3 });
    const ka2f = try write(h, a2f, .{ .lineage = lin(4), .source = 4 });
    const ka2 = try write(h, a2, .{ .lineage = lin(5), .source = 5 });
    const order = h.s.evictionOrder() catch return CheckFailed;
    defer gpa.free(order);
    const W = Store.Why;
    const want = [_]struct { k: Digest, w: W }{
        .{ .k = ka2f, .w = .pruned_forks },
        .{ .k = kb1, .w = .pruned_conversations },
        .{ .k = ka2, .w = .pruned_conversations },
        .{ .k = ka1, .w = .pruned_conversations },
        .{ .k = ks, .w = .pruned_shared },
    };
    try ensure(order.len == want.len, @src());
    for (order, want) |o, w| try ensure(std.mem.eql(u8, &o.key, &w.k) and o.why == w.w, @src());
    const sm = h.s.summary();
    try ensure(sm.conversations == 2 and sm.shared_states == 1 and sm.forks == 1 and sm.turns_kept_max == 2, @src());
}

/// Old turns past retention on the active path go as soon as a newer turn is written; their blocks stay (the
/// newer turn names them).
fn checkTurns(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "turns", .turns = 2 });
    defer h.deinit();
    const t3 = try idsOf(gpa, 3 * block_tokens, 1100);
    defer gpa.free(t3);
    const k1 = try write(h, t3[0..block_tokens], .{ .lineage = lin(1), .source = 11 });
    const k2 = try write(h, t3[0 .. 2 * block_tokens], .{ .lineage = lin(1), .source = 11 });
    const k3 = try write(h, t3, .{ .lineage = lin(1), .source = 11 });
    try ensure(!h.s.points.contains(k1) and h.s.points.contains(k2) and h.s.points.contains(k3), @src());
    const ct = h.s.counts();
    try ensure(ct.trimmed_turns == 1 and ct.blocks == 3 and ct.staged_blocks == 3, @src());
    const got = (try restore(h, k3)) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(11)[0 .. 3 * block_tokens * bpt]), @src());
}

const glm_bpt = 3 * 256; // like GLM's 6,204 B a token: a whole block fills whole pages, a grid tail does not

fn parentIs(h: *H, key: Digest, want: ?Digest) bool {
    const got = h.s.parent.get(key) orelse return false;
    if (got == null or want == null) return got == null and want == null;
    return std.mem.eql(u8, &got.?, &want.?);
}

/// GLM's shape: one conversation (one 0111 group) kept on a grid inside blocks (4 here, GLM's 64 of 2,048), with
/// bytes a token that do not divide a page. Turns inside one block extend each other, so `turns` trims them. A turn
/// past a block boundary does not extend the turn before it (`extends` keeps only the last block's ids): it starts a
/// new path and the turns before it stay past `turns` (docs/glm-port/spill-store.md, open issue 1; a fix there
/// changes this check). Every point, its tail off the page grid, restores exactly.
fn checkGridTurns(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "grid", .turns = 2, .bpt = glm_bpt });
    defer h.deinit();
    const ids = try idsOf(gpa, 3 * block_tokens, 1150);
    defer gpa.free(ids);
    const ats = [_]usize{ 20, 24, 28, 36, 40, 44 };
    var k: [ats.len]Digest = undefined;
    for (ats[0..3], 0..) |at, i| k[i] = try write(h, ids[0..at], .{ .lineage = lin(15), .source = 15, .group = 9 });
    try ensure(!h.s.points.contains(k[0]) and h.s.points.contains(k[1]) and h.s.points.contains(k[2]), @src());
    try ensure(parentIs(h, k[2], k[1]) and h.s.counts().trimmed_turns == 1, @src());
    k[3] = try write(h, ids[0..ats[3]], .{ .lineage = lin(15), .source = 15, .group = 9 });
    try ensure(parentIs(h, k[3], null) and h.s.counts().trimmed_turns == 1, @src()); // the boundary: nothing trimmed
    k[4] = try write(h, ids[0..ats[4]], .{ .lineage = lin(15), .source = 15, .group = 9 });
    try ensure(parentIs(h, k[4], k[3]) and h.s.counts().trimmed_turns == 1, @src());
    k[5] = try write(h, ids[0..ats[5]], .{ .lineage = lin(15), .source = 15, .group = 9 });
    try ensure(parentIs(h, k[5], k[4]) and h.s.counts().trimmed_turns == 2, @src());
    try ensure(!h.s.points.contains(k[3]) and h.s.points.contains(k[4]) and h.s.points.contains(k[5]), @src());
    try ensure(h.s.points.contains(k[1]) and h.s.points.contains(k[2]), @src()); // stranded behind the boundary
    try ensure(h.s.summary().conversations == 1, @src()); // the group still makes them one conversation
    for ([_]usize{ 1, 2, 4, 5 }) |i| {
        try ensure((ats[i] % block_tokens) * glm_bpt % fsio.alignment != 0, @src());
        const got = (try restore(h, k[i])) orelse return CheckFailed;
        defer gpa.free(got);
        try ensure(std.mem.eql(u8, got, h.src(15)[0 .. ats[i] * glm_bpt]), @src());
    }
}

/// A resume forks the lineage at its cut: the same names before it (nothing restaged), its own after.
fn checkFork(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "fork" });
    defer h.deinit();
    const p = try idsOf(gpa, 3 * block_tokens, 1200);
    defer gpa.free(p);
    const more = try idsOf(gpa, 2 * block_tokens, 1201);
    defer gpa.free(more);
    const q = try cat(gpa, p[0 .. 2 * block_tokens], more);
    defer gpa.free(q);
    _ = try write(h, p, .{ .lineage = lin(12), .source = 12 });
    const before = h.s.counts().staged_blocks;
    var forked = try lin(12).fork(gpa, @splat(77), 2 * block_tokens);
    defer forked.deinit(gpa);
    const kq = try write(h, q, .{ .lineage = forked, .source = 12 });
    try ensure(h.s.counts().staged_blocks - before == 2, @src());
    const got = (try restore(h, kq)) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(12)[0 .. 4 * block_tokens * bpt]), @src());
}

/// 0111 groups: branches of one conversation that share only a shared prefix count as one conversation.
fn checkGroups(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "groups" });
    defer h.deinit();
    const sys = try idsOf(gpa, block_tokens, 1300);
    defer gpa.free(sys);
    _ = try write(h, sys, .{ .lineage = lin(13), .source = 13, .shared = true });
    for (0..3) |i| {
        const x = try idsOf(gpa, block_tokens, 1301 + i);
        defer gpa.free(x);
        const y = try cat(gpa, sys, x);
        defer gpa.free(y);
        _ = try write(h, y, .{ .lineage = lin(@intCast(14 + i)), .source = 14 + i, .group = if (i < 2) 7 else 0 });
    }
    try ensure(h.s.summary().conversations == 2, @src());
}

/// A conversation over its quota trims its own old turns and forks first, not another conversation's states.
fn checkQuota(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "quota", .cap = 400 << 10, .quota = 0.1, .turns = 0 });
    defer h.deinit();
    const other = try idsOf(gpa, 2 * block_tokens, 1400);
    defer gpa.free(other);
    const ko = try write(h, other, .{ .lineage = lin(40), .source = 40 });
    const conv = try idsOf(gpa, 8 * block_tokens, 1401);
    defer gpa.free(conv);
    var keys: [8]Digest = undefined;
    for (1..9) |t| {
        const r = keepIds(h, conv[0 .. t * block_tokens], .{ .lineage = lin(41), .source = 41 });
        if (r != .queued) break;
        try ensure(settle(h).done, @src());
        keys[t - 1] = format.pointKey(conv[0 .. t * block_tokens], block_tokens, &.{});
    }
    const ct = h.s.counts();
    try ensure(ct.quota_trims > 0 and ct.pruned_turns > 0, @src());
    try ensure(h.s.points.contains(ko), @src());
    try ensure(h.s.points.contains(keys[7]), @src());
}

/// The free-disk floor: no write leaves less free; room is made in eviction order, else the write is refused.
fn checkFloor(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "floor", .floor = 1 << 20, .free = (1 << 20) + 200 * 1024 });
    defer h.deinit();
    const a = try idsOf(gpa, 4 * block_tokens, 1500);
    defer gpa.free(a);
    const ka = try write(h, a, .{ .lineage = lin(50), .source = 50 });
    // the disk fills with other files: the next write evicts A first
    h.hk.free = (1 << 20) + 2 * 1024;
    h.s.refreshDiskFree();
    const b = try idsOf(gpa, 2 * block_tokens, 1501);
    defer gpa.free(b);
    try ensure(keepIds(h, b, .{ .lineage = lin(51), .source = 51 }) == .queued, @src());
    try ensure(!h.s.points.contains(ka), @src());
    _ = settle(h);
    // full of other things: nothing could make room, the write is refused
    h.hk.free = 1 << 10;
    h.s.refreshDiskFree();
    const c2 = try idsOf(gpa, 2 * block_tokens, 1502);
    defer gpa.free(c2);
    try ensure(keepIds(h, c2, .{ .lineage = lin(52), .source = 52 }) == .no_space, @src());
}

/// The cap: a write that would pass it evicts the least recently used conversation, not a newer one.
fn checkCap(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "cap", .cap = 12 * 1024 });
    defer h.deinit();
    var keys: [3]Digest = undefined;
    for (0..3) |i| {
        const x = try idsOf(gpa, block_tokens, 1600 + i);
        defer gpa.free(x);
        keys[i] = try write(h, x, .{ .lineage = lin(@intCast(60 + i)), .source = 60 + i });
    }
    try ensure(!h.s.points.contains(keys[0]) and h.s.points.contains(keys[1]) and h.s.points.contains(keys[2]), @src());
    try ensure(h.s.counts().pruned_conversations >= 1 and h.s.counts().stored_bytes <= h.cfg.cap_bytes, @src());
}

/// A second owner of a live store is refused before it scans or deletes anything.
fn checkOwner(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "owner" });
    defer h.deinit();
    const tfd = c.openat(h.s.blocks_fd, ".pending.blk.9.9.tmp", .{ .ACCMODE = .WRONLY, .CREAT = true, .CLOEXEC = true }, @as(c.mode_t, 0o600));
    _ = c.close(tfd);
    if (Store.open(gpa, io, h.cfg)) |other| {
        other.close();
        try ensure(false, @src());
    } else |e| try ensure(e == error.StoreBusy, @src());
    try ensure(fileExists(h.s.blocks_fd, ".pending.blk.9.9.tmp"), @src());
}

/// Other builds' namespaces: pruned unless live (locked).
fn checkPrune(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "prune", .keep_builds = 0 });
    defer h.deinit();
    const rfd = try fsio.openDir(h.root);
    defer _ = c.close(rfd);
    try fsio.ensureDirAt(rfd, "native-spill-00000000deadbeef");
    const ofd = try fsio.openDirAt(rfd, "native-spill-00000000deadbeef");
    defer _ = c.close(ofd);
    try fsio.ensureDirAt(ofd, "rank0");
    const r0 = try fsio.openDirAt(ofd, "rank0");
    var live = try fsio.Lock.take(r0);
    try h.reopen();
    try ensure(fileExists(ofd, "rank0"), @src());
    live.release();
    _ = c.close(r0);
    try h.reopen();
    try ensure(!fileExists(rfd, "native-spill-00000000deadbeef"), @src());
}

/// Families sharing a root: open prunes only its own family's namespaces, by their exact shape. Another family's
/// (DeepSeek's `native-spill-`) and a name that only starts like ours stay, unlocked as they are.
fn checkPruneFamilies(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "families", .keep_builds = 0, .prefix = "glm-spill-" });
    defer h.deinit();
    const rfd = try fsio.openDir(h.root);
    defer _ = c.close(rfd);
    const names = [_][:0]const u8{ "native-spill-00000000cafef00d", "glm-spill-notabuild", "glm-spill-00000000deadbeef0", "glm-spill-00000000deadbeef" };
    for (names) |n| {
        try fsio.ensureDirAt(rfd, n);
        const d = try fsio.openDirAt(rfd, n);
        defer _ = c.close(d);
        try fsio.ensureDirAt(d, "rank0");
    }
    try h.reopen();
    for (names[0..3]) |n| try ensure(fileExists(rfd, n), @src());
    try ensure(!fileExists(rfd, names[3]) and h.s.counts().builds_pruned == 1, @src());
    var nb: [64]u8 = undefined;
    try ensure(fileExists(rfd, try std.fmt.bufPrint(&nb, "glm-spill-{s}", .{format.hex(&h.cfg.compat)[0..16]})), @src());
}

/// Lookups: the longest complete point the prompt extends within its bounds; media must match.
fn checkLookup(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "lookup", .turns = 0 });
    defer h.deinit();
    const ids = try idsOf(gpa, 5 * block_tokens, 1700);
    defer gpa.free(ids);
    const m = [_]format.Media{.{ .start = 3, .len = 4, .digest = @splat(1) }};
    const raw = try gpa.dupe(u32, ids);
    defer gpa.free(raw);
    for (3..7) |i| { // the picture's keyed ids made from its digest, its raw ids the placeholder
        ids[i] = format.pictureId(&m[0].digest, @intCast(i - 3));
        raw[i] = 99;
    }
    // a point whose ids do not agree with its media is refused at keep (and would be at recovery)
    try ensure(keepIds(h, ids[0 .. 2 * block_tokens], .{ .lineage = lin(70), .source = 70, .media = &m }) == .invalid, @src());
    var forged = m;
    forged[0].digest[20] ^= 1; // (the first 8 bytes seed the ids: the same ids, another picture past them)
    const kf = try write(h, ids[0 .. 2 * block_tokens], .{ .lineage = lin(70), .source = 70, .media = &forged, .raw = raw });
    var seeded = m;
    seeded[0].digest[0] ^= 1; // (a seed byte changed: its ids no longer derive from it)
    try ensure(keepIds(h, ids[0 .. 2 * block_tokens], .{ .lineage = lin(70), .source = 70, .media = &seeded, .raw = raw }) == .invalid, @src());
    var placeholder = try gpa.dupe(u32, raw);
    defer gpa.free(placeholder);
    placeholder[4] = ids[4]; // a raw id inside the picture that is a picture's keyed id, not a model id
    try ensure(keepIds(h, ids[0 .. 2 * block_tokens], .{ .lineage = lin(70), .source = 70, .media = &m, .raw = placeholder }) == .invalid, @src());
    const k2 = try write(h, ids[0 .. 2 * block_tokens], .{ .lineage = lin(70), .source = 70, .media = &m, .raw = raw });
    const k4 = try write(h, ids[0 .. 4 * block_tokens], .{ .lineage = lin(70), .source = 70, .media = &m, .raw = raw });
    const q = struct {
        fn at(hh: *H, x: store.Query) ?Digest {
            return if (hh.s.lookup(x)) |hit| hit.key else null;
        }
    };
    try ensure(std.meta.eql(q.at(h, .{ .keyed = ids, .media = &m, .max_at = 70 }), k4), @src());
    try ensure(std.meta.eql(q.at(h, .{ .keyed = ids, .media = &m, .max_at = 50 }), k2), @src());
    try ensure(q.at(h, .{ .keyed = ids, .media = &m, .max_at = 50, .longer_than = 40 }) == null, @src());
    try ensure(std.meta.eql(q.at(h, .{ .keyed = ids, .media = &m, .max_at = 80, .grid = 64 }), k4), @src());
    var other = m;
    other[0].digest[0] = 2;
    try ensure(q.at(h, .{ .keyed = ids, .media = &other, .max_at = 80 }) == null, @src());
    // the same shortened ids from another payload (only the full digest tells them apart): its own point, never m's
    try ensure(std.meta.eql(q.at(h, .{ .keyed = ids, .media = &forged, .max_at = 80 }), kf), @src());
    try ensure(!std.meta.eql(kf, k2), @src());
    try ensure(q.at(h, .{ .keyed = ids, .media = &.{}, .max_at = 80 }) == null, @src());
    var changed = try gpa.dupe(u32, ids);
    defer gpa.free(changed);
    changed[20] +%= 1;
    try ensure(std.meta.eql(q.at(h, .{ .keyed = changed, .media = &m, .max_at = 80 }), null), @src());
    // across a restart: the points with pictures recover (their ids checked against their media) and match as before
    try h.reopen();
    try ensure(std.meta.eql(q.at(h, .{ .keyed = ids, .media = &m, .max_at = 70 }), k4), @src());
    try ensure(std.meta.eql(q.at(h, .{ .keyed = ids, .media = &forged, .max_at = 80 }), kf), @src());
    try ensure(q.at(h, .{ .keyed = ids, .media = &other, .max_at = 80 }) == null, @src());
}

/// Replace a point file's bytes (same name, truncated first).
fn rewritePointFile(h: *H, key: Digest, suffix: []const u8, bytes: []const u8) !void {
    var nb: [96]u8 = undefined;
    var zb: [96]u8 = undefined;
    const name = try std.fmt.bufPrintSentinel(&zb, "{s}", .{pointFile(&nb, key, suffix)}, 0);
    const fd = c.openat(h.s.points_fd, name, .{ .ACCMODE = .WRONLY, .TRUNC = true, .CLOEXEC = true }, @as(c.mode_t, 0));
    if (fd < 0) return CheckFailed;
    defer _ = c.close(fd);
    if (c.pwrite(fd, bytes.ptr, bytes.len, 0) != @as(isize, @intCast(bytes.len))) return CheckFailed;
}

/// A picture point's raw ids altered on disk with every CRC made good (the .ids file's in the manifest, the
/// manifest's trailer): raw ids are not in the point key, so only recovery's `format.mediaIds` check catches it. A
/// raw id inside the picture made a picture id, and a raw id outside it changed, each drop their point; an untouched
/// picture point beside them recovers and still matches.
fn checkPictureRecovery(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "picrecover", .turns = 0 });
    defer h.deinit();
    const m = [_]format.Media{.{ .start = 3, .len = 4, .digest = @splat(7) }};
    const n = 2 * block_tokens;
    var keys: [3]Digest = undefined;
    var keyed: [3][]u32 = undefined;
    for (0..3) |i| {
        keyed[i] = try idsOf(gpa, n, 1900 + i);
        const raw = try gpa.dupe(u32, keyed[i]);
        defer gpa.free(raw);
        for (3..7) |j| {
            keyed[i][j] = format.pictureId(&m[0].digest, @intCast(j - 3));
            raw[j] = 99;
        }
        keys[i] = try write(h, keyed[i], .{ .lineage = lin(@intCast(80 + i)), .source = 80 + i, .media = &m, .raw = raw });
    }
    defer for (keyed) |k| gpa.free(k);
    for (0..2) |i| {
        var nb: [96]u8 = undefined;
        var zb: [96]u8 = undefined;
        const ids_raw = try fsio.readSmall(gpa, h.s.points_fd, try std.fmt.bufPrintSentinel(&zb, "{s}", .{pointFile(&nb, keys[i], ".ids")}, 0), 1 << 20);
        defer gpa.free(ids_raw);
        try ensure(ids_raw.len >= 2 * n * 4, @src());
        const raw_ids: []align(1) u32 = std.mem.bytesAsSlice(u32, ids_raw[n * 4 ..][0 .. n * 4]);
        if (i == 0) raw_ids[4] = keyed[i][4] else raw_ids[10] +%= 1; // a picture id inside, another model id outside
        try rewritePointFile(h, keys[i], ".ids", ids_raw);
        const man = try fsio.readSmall(gpa, h.s.points_fd, try std.fmt.bufPrintSentinel(&zb, "{s}", .{pointFile(&nb, keys[i], ".man")}, 0), 1 << 20);
        defer gpa.free(man);
        var mf = try format.Manifest.parse(gpa, man, h.cfg.compat, h.cfg.limits);
        defer mf.deinit(gpa);
        const same = try mf.encode(gpa, h.cfg.compat);
        defer gpa.free(same);
        try ensure(std.mem.eql(u8, same, man), @src()); // (the re-encoding is the manifest's own bytes)
        mf.ids_crc = format.crc32(ids_raw);
        const fixed = try mf.encode(gpa, h.cfg.compat);
        defer gpa.free(fixed);
        try rewritePointFile(h, keys[i], ".man", fixed);
    }
    try h.reopen();
    try ensure(h.s.counts().recovered_dropped == 2, @src());
    try ensure(!h.s.presence(keys[0]) and !h.s.presence(keys[1]) and h.s.presence(keys[2]), @src());
    for (0..2) |i| try ensure(h.s.lookup(.{ .keyed = keyed[i], .media = &m, .max_at = 80 }) == null, @src());
    const hit = h.s.lookup(.{ .keyed = keyed[2], .media = &m, .max_at = 80 }) orelse return CheckFailed;
    try ensure(std.mem.eql(u8, &hit.key, &keys[2]), @src());
}

/// Disk faults on the write path: ENOSPC fails the point and leaves no temporary file or staging behind; short
/// writes and EINTR still write every byte; a file system without O_DIRECT is written buffered.
fn checkWriteFaults(gpa: Allocator) !void {
    {
        const h = try H.init(gpa, .{ .tag = "enospc" });
        defer h.deinit();
        const ids = try idsOf(gpa, 3 * block_tokens, 1800);
        defer gpa.free(ids);
        h.hk.nospace_writes = 1;
        try ensure(keepIds(h, ids, .{ .lineage = lin(80), .source = 80 }) == .queued, @src());
        _ = settle(h);
        var comp: [4]store.Completion = undefined;
        const n = h.s.poll(&comp);
        try ensure(n == 1 and comp[0] == .failed and comp[0].failed.why == .no_space, @src());
        try ensure(!h.s.presence(format.pointKey(ids, block_tokens, &.{})), @src());
        try ensure(countFiles(gpa, h.s.blocks_fd, true) == 0 and countFiles(gpa, h.s.points_fd, false) == 0, @src());
        try ensure(h.s.counts().staging_bytes == 0, @src());
    }
    inline for (.{ false, true }) |nd| {
        const h = try H.init(gpa, .{ .tag = "short", .no_direct = nd });
        defer h.deinit();
        const ids = try idsOf(gpa, 3 * block_tokens + 9, 1801);
        defer gpa.free(ids);
        h.hk.short_intr = true;
        const key = try write(h, ids, .{ .lineage = lin(81), .source = 81 });
        h.hk.short_intr = false;
        try h.reopen();
        const got = (try restore(h, key)) orelse return CheckFailed;
        defer gpa.free(got);
        try ensure(std.mem.eql(u8, got, h.src(81)[0 .. ids.len * bpt]), @src());
    }
}

/// A restore cancelled while its reads are in flight gives every slot back, and the point back to eviction.
fn checkCancelRead(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "cancelread", .read_slots = 3 });
    defer h.deinit();
    const ids = try idsOf(gpa, 8 * block_tokens, 1900);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(90), .source = 90 });
    const id = try h.s.beginRead(key);
    var rb: [1]store.ReadyBlock = undefined;
    _ = h.s.readAwait(id, 0, 5 * std.time.ns_per_s, .{});
    const got = h.s.readReady(id, null, &rb);
    _ = try h.s.endRead(id, .cancel);
    for (rb[0..got.n]) |b| h.s.readRelease(b.slot);
    try ensure(waitUntil(struct {
        fn f(hh: *H) bool {
            const ct = hh.s.counts();
            return ct.staging_bytes == 0 and hh.s.reads.items.len == 0;
        }
    }.f, h, 3000), @src());
    try ensure(!h.s.reading.contains(key), @src());
}

/// Two restores share the read slots: each gets blocks in turn.
fn checkFairReads(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "fair", .read_slots = 2 });
    defer h.deinit();
    var ids: [2]store.ReadId = undefined;
    for (0..2) |i| {
        const x = try idsOf(gpa, 6 * block_tokens, 2000 + i);
        defer gpa.free(x);
        const key = try write(h, x, .{ .lineage = lin(@intCast(100 + i)), .source = 100 + i });
        ids[i] = try h.s.beginRead(key);
    }
    // the first restore took both slots (the second was not there yet); the slots it gives back go to the second
    try ensure(h.s.readAwait(ids[0], 0, 5 * std.time.ns_per_s, .{}) and h.s.readAwait(ids[0], 1, 5 * std.time.ns_per_s, .{}), @src());
    var rb: [2]store.ReadyBlock = undefined;
    const got = h.s.readReady(ids[0], null, &rb);
    try ensure(got.n == 2, @src());
    h.s.readRelease(rb[0].slot);
    try ensure(h.s.readAwait(ids[1], 0, 5 * std.time.ns_per_s, .{}), @src());
    h.s.readRelease(rb[1].slot);
    for (ids) |id| _ = try h.s.endRead(id, .cancel);
    try ensure(waitUntil(struct {
        fn f(hh: *H) bool {
            return hh.s.counts().staging_bytes == 0;
        }
    }.f, h, 3000), @src());
}

/// Replicated and owner-sharded points never read as each other: a store of another layout drops them as foreign.
fn checkLayout(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "layout" });
    defer h.deinit();
    const ids = try idsOf(gpa, 2 * block_tokens, 2300);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(23), .source = 23 });
    h.cfg.owner_layout = .owner_sharded;
    try h.reopen();
    try ensure(!h.s.presence(key) and h.s.counts().recovered_dropped == 1, @src());
}

/// 0118's owner-sharded store: a rank keeps a point of its own part with full blocks of every plane and a point of
/// another part with only the replicated planes (foreign_block_bytes); every point names its part (a part-less keep is
/// invalid), the sizes survive a restart, a lookup and partOf say the part, the same ids under two parts are two
/// block names (the salt), and a rank that owns another part reads the first rank's manifests as damaged.
fn checkOwnerParts(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "parts" });
    defer h.deinit();
    const foreign = block_bytes / 4;
    h.cfg.owner_layout = .owner_sharded;
    h.cfg.limits.foreign_block_bytes = foreign;
    h.cfg.limits.own_part = 0;
    try h.reopen();
    const ids = try idsOf(gpa, 2 * block_tokens, 2500);
    defer gpa.free(ids);
    try ensure(keepIds(h, ids, .{ .lineage = lin(25), .source = 25 }) == .invalid, @src()); // (no part)
    const own = try write(h, ids, .{ .lineage = lin(25).inPart(0), .source = 25 });
    const other_ids = try idsOf(gpa, 3 * block_tokens, 2501);
    defer gpa.free(other_ids);
    const other = try write(h, other_ids, .{ .lineage = lin(26).inPart(1), .source = 26 });
    for (h.s.points.get(own).?.blocks) |b| try ensure(b.bytes == block_bytes, @src());
    for (h.s.points.get(other).?.blocks) |b| try ensure(b.bytes == foreign, @src());
    // the same ids from another part: other names (never the first point's blocks by name)
    const pa = h.s.points.get(own).?;
    var spans: [4]format.Span = undefined;
    var ch: std.ArrayList(Digest) = .empty;
    defer ch.deinit(gpa);
    try format.chain(gpa, ids, block_tokens, &ch);
    format.blockNames(lin(25).inPart(1), ch.items, ids, block_tokens, spans[0..2]);
    for (pa.blocks, spans[0..2]) |b, sp| try ensure(!std.mem.eql(u8, &b.name, &sp.name), @src());
    format.blockNames(lin(25).inPart(1).inPart(0), ch.items, ids, block_tokens, spans[0..2]); // (re-salted: replaced)
    for (pa.blocks, spans[0..2]) |b, sp| try ensure(std.mem.eql(u8, &b.name, &sp.name), @src());
    // the restored rows are the stored sizes: a foreign block is its source's first bytes of the block's rows
    const got = (try restore(h, other)) orelse return CheckFailed;
    defer gpa.free(got);
    for (0..3) |k| try ensure(std.mem.eql(u8, got[k * block_tokens * bpt ..][0..foreign], h.src(26)[k * block_tokens * bpt ..][0..foreign]), @src());
    try h.reopen();
    try ensure(h.s.presence(own) and h.s.presence(other), @src());
    try ensure(h.s.partOf(own).?.part == 0 and h.s.partOf(other).?.part == 1 and h.s.partOf(@splat(9)) == null, @src());
    try ensure(h.s.counts().points == 2 and h.s.counts().own_part_points == 1, @src()); // (one of this rank's part)
    const hit = h.s.lookup(.{ .keyed = other_ids, .max_at = max_tokens }) orelse return CheckFailed;
    try ensure(hit.part == 1 and std.mem.eql(u8, &hit.key, &other), @src());
    // a rank owning part 1 with these files: the sizes are not its own, both points damaged
    h.cfg.limits.own_part = 1;
    try h.reopen();
    try ensure(!h.s.presence(own) and !h.s.presence(other) and h.s.counts().recovered_dropped == 2, @src());
    // a foreign size past the block's, or one on a replicated store, is no configuration
    h.closeStore();
    h.cfg.limits.foreign_block_bytes = block_bytes + 1;
    try ensure(if (Store.open(gpa, io, h.cfg)) |_| false else |e| e == error.BadConfig, @src());
    h.cfg.limits.foreign_block_bytes = foreign;
    h.cfg.owner_layout = .replicated;
    try ensure(if (Store.open(gpa, io, h.cfg)) |_| false else |e| e == error.BadConfig, @src());
}

/// GLM 0110's pin (review of the native port): a resumed conversation that is the least recently used one, kept
/// under cap pressure, makes room without evicting the blocks it reuses. Its old turn goes (room), its blocks stay
/// (the new turn names them), the new turn stages only its new block, publishes and restores.
fn checkKeepReuse(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "reuse", .cap = 24 * 1024 });
    defer h.deinit();
    const a = try idsOf(gpa, 3 * block_tokens, 2400);
    defer gpa.free(a);
    const ka1 = try write(h, a[0 .. 2 * block_tokens], .{ .lineage = lin(24), .source = 24 });
    for (0..2) |i| {
        const x = try idsOf(gpa, block_tokens, 2401 + i);
        defer gpa.free(x);
        _ = try write(h, x, .{ .lineage = lin(@intCast(25 + i)), .source = 25 + i });
    }
    const before = h.s.counts();
    const ka2 = try write(h, a, .{ .lineage = lin(24), .source = 24 });
    const after = h.s.counts();
    try ensure(!h.s.points.contains(ka1) and after.pruned_conversations > before.pruned_conversations, @src()); // (the pressure)
    try ensure(after.staged_blocks - before.staged_blocks == 1 and after.stored_bytes <= h.cfg.cap_bytes, @src());
    const got = (try restore(h, ka2)) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(24)[0 .. 3 * block_tokens * bpt]), @src());
}

/// A cancelled write's unstaged blocks go to another waiting write that names them, and so does their
/// reservation: the backlog, cap and floor keep counting those bytes until they are written.
fn checkHandOver(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "handover" });
    defer h.deinit();
    const ids = try idsOf(gpa, 3 * block_tokens, 2600);
    defer gpa.free(ids);
    const r1 = keepIds(h, ids[0 .. 2 * block_tokens], .{ .lineage = lin(28), .source = 28 });
    try ensure(r1 == .queued, @src());
    try ensure(keepIds(h, ids, .{ .lineage = lin(28), .source = 28 }) == .queued, @src());
    try ensure(h.s.cancel(r1.queued.job), @src());
    try ensure(h.s.counts().queued_bytes >= 3 * block_bytes, @src()); // the second write's 3 blocks, all reserved
    try ensure(settle(h).done, @src());
    const got = (try restore(h, format.pointKey(ids, block_tokens, &.{}))) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(28)[0 .. 3 * block_tokens * bpt]), @src());
    try ensure(h.s.counts().queued_bytes == 0, @src());
}

/// A quota that is not a fraction in [0, 1], or a namespace prefix that is empty, too long or a path, is refused at
/// open.
fn checkBadConfig(gpa: Allocator) !void {
    const h = try H.init(gpa, .{ .tag = "badcfg" });
    defer h.deinit();
    h.closeStore();
    for ([_]f64{ std.math.nan(f64), -0.1, 1.5, std.math.inf(f64) }) |q| {
        var cfg = h.cfg;
        cfg.quota = q;
        if (Store.open(gpa, io, cfg)) |s2| {
            s2.close();
            try ensure(false, @src());
        } else |e| try ensure(e == error.BadConfig, @src());
    }
    for ([_][]const u8{ "", "0123456789abcdef0123456789abcdef-", "../glm-spill-", "glm\x00spill-" }) |p| {
        var cfg = h.cfg;
        cfg.prefix = p;
        if (Store.open(gpa, io, cfg)) |s2| {
            s2.close();
            try ensure(false, @src());
        } else |e| try ensure(e == error.BadConfig, @src());
    }
}

fn chmodAt(dirfd: c.fd_t, name: []const u8, mode: c.mode_t) !void {
    var zb: [128]u8 = undefined;
    const z = try std.fmt.bufPrintSentinel(&zb, "{s}", .{name}, 0);
    if (c.fchmodat(dirfd, z, mode, 0) != 0) return error.Unexpected;
}

/// Recovery deletes only damage: a point it cannot read for another reason (here: its ids, or one of its blocks,
/// unreadable) keeps every file, and so does every block; the next open recovers it whole.
fn checkRecoverTransient(gpa: Allocator) !void {
    if (c.geteuid() == 0) return; // (permissions do not stop root)
    const h = try H.init(gpa, .{ .tag = "transient" });
    defer h.deinit();
    const ia = try idsOf(gpa, 2 * block_tokens + 3, 2500);
    defer gpa.free(ia);
    const ib = try idsOf(gpa, 2 * block_tokens, 2501);
    defer gpa.free(ib);
    const ic = try idsOf(gpa, block_tokens, 2502);
    defer gpa.free(ic);
    const ka = try write(h, ia, .{ .lineage = lin(25), .source = 250 });
    const kb = try write(h, ib, .{ .lineage = lin(26), .source = 251 });
    const kc = try write(h, ic, .{ .lineage = lin(27), .source = 252 });
    const files = countFiles(gpa, h.s.blocks_fd, false) + countFiles(gpa, h.s.points_fd, false);
    var nb: [96]u8 = undefined;
    const b0 = (h.s.points.get(kb) orelse return CheckFailed).blocks[0].name;
    try chmodAt(h.s.points_fd, pointFile(&nb, ka, ".ids"), 0);
    try chmodAt(h.s.blocks_fd, blockFile(&nb, b0), 0);
    try h.reopen();
    const ct = h.s.counts();
    const deferred = !h.s.points.contains(ka) and !h.s.points.contains(kb) and h.s.points.contains(kc);
    const kept = countFiles(gpa, h.s.blocks_fd, false) + countFiles(gpa, h.s.points_fd, false);
    chmodAt(h.s.points_fd, pointFile(&nb, ka, ".ids"), 0o600) catch {}; // (gone if recovery deleted them)
    chmodAt(h.s.blocks_fd, blockFile(&nb, b0), 0o600) catch {};
    try ensure(deferred and ct.recover_deferred == 2 and ct.recovered_dropped == 0, @src());
    try ensure(kept == files, @src());
    try h.reopen();
    try ensure(h.s.points.contains(ka) and h.s.points.contains(kb) and h.s.points.contains(kc), @src());
    const got = (try restore(h, kb)) orelse return CheckFailed;
    defer gpa.free(got);
    try ensure(std.mem.eql(u8, got, h.src(251)[0 .. 2 * block_tokens * bpt]), @src());
}

// -- the tests -------------------------------------------------------------------------------------------------------
const Check = struct { name: []const u8, f: *const fn (Allocator) anyerror!void };
const checks = [_]Check{
    .{ .name = "tail", .f = checkTail },
    .{ .name = "upto", .f = checkUpto },
    .{ .name = "shared upgrade", .f = checkSharedUpgrade },
    .{ .name = "shared waiting", .f = checkSharedWaiting },
    .{ .name = "shared during write", .f = checkSharedDuringWrite },
    .{ .name = "drop during rewrite", .f = checkDropDuringRewrite },
    .{ .name = "drain waits for rewrites", .f = checkDrainWaitsRewrite },
    .{ .name = "forget lost", .f = checkForgetLost },
    .{ .name = "forget damaged", .f = checkForgetDamaged },
    .{ .name = "pages", .f = checkPages },
    .{ .name = "publish order", .f = checkPublishOrder },
    .{ .name = "hang", .f = checkHang },
    .{ .name = "carve", .f = checkCarve },
    .{ .name = "budget", .f = checkBudget },
    .{ .name = "crashes", .f = checkCrashes },
    .{ .name = "recovery", .f = checkRecovery },
    .{ .name = "dedup sha", .f = checkDedupSha },
    .{ .name = "guard", .f = checkGuard },
    .{ .name = "guard cancel", .f = checkGuardCancel },
    .{ .name = "eviction order", .f = checkEvictionOrder },
    .{ .name = "turns", .f = checkTurns },
    .{ .name = "grid turns", .f = checkGridTurns },
    .{ .name = "fork", .f = checkFork },
    .{ .name = "groups", .f = checkGroups },
    .{ .name = "quota", .f = checkQuota },
    .{ .name = "floor", .f = checkFloor },
    .{ .name = "cap", .f = checkCap },
    .{ .name = "owner", .f = checkOwner },
    .{ .name = "prune", .f = checkPrune },
    .{ .name = "prune families", .f = checkPruneFamilies },
    .{ .name = "lookup", .f = checkLookup },
    .{ .name = "picture recovery", .f = checkPictureRecovery },
    .{ .name = "write faults", .f = checkWriteFaults },
    .{ .name = "cancel read", .f = checkCancelRead },
    .{ .name = "fair reads", .f = checkFairReads },
    .{ .name = "last use", .f = checkLastUse },
    .{ .name = "refusals", .f = checkRefusals },
    .{ .name = "layout", .f = checkLayout },
    .{ .name = "keep reuse", .f = checkKeepReuse },
    .{ .name = "recover transient", .f = checkRecoverTransient },
    .{ .name = "hand over", .f = checkHandOver },
    .{ .name = "bad config", .f = checkBadConfig },
    .{ .name = "owner parts", .f = checkOwnerParts },
    .{ .name = "staging cap", .f = checkStagingCap },
};

test "spill store: the GLM 0110 checks and the native port's own, all pass" {
    var failed: usize = 0;
    for (checks) |ch| {
        ch.f(testing.allocator) catch |e| {
            std.debug.print("FAIL {s}: {t}\n", .{ ch.name, e });
            failed += 1;
        };
    }
    try testing.expectEqual(@as(usize, 0), failed);
}

/// Which checks each mutant must fail (the GLM mutants that apply to a single-rank store, and the port's own).
const kills = [_]struct { m: mutants.Mutant, checks: []const []const u8 }{
    .{ .m = .tail_staged_later, .checks = &.{"tail"} },
    .{ .m = .loadpart_past_upto, .checks = &.{"upto"} },
    .{ .m = .no_upgrade, .checks = &.{ "shared upgrade", "shared waiting", "shared during write" } },
    .{ .m = .upgrade_not_durable, .checks = &.{"shared upgrade"} },
    .{ .m = .waiting_not_flagged, .checks = &.{"shared waiting"} },
    .{ .m = .written_no_rewrite, .checks = &.{"shared during write"} },
    .{ .m = .rewrite_not_put, .checks = &.{"shared upgrade"} },
    .{ .m = .rewrite_no_dir_fsync, .checks = &.{"shared upgrade"} },
    .{ .m = .drain_skips_rewrites, .checks = &.{"drain waits for rewrites"} },
    .{ .m = .lost_is_damage, .checks = &.{"forget lost"} },
    .{ .m = .lost_names_kept, .checks = &.{"forget lost"} },
    .{ .m = .damage_not_widened, .checks = &.{"forget damaged"} },
    .{ .m = .small_files_keep_pages, .checks = &.{"pages"} },
    .{ .m = .pages_dropped_before_fsync, .checks = &.{"pages"} },
    .{ .m = .hang_no_recycle_in_wait, .checks = &.{"hang"} },
    .{ .m = .budget_each_read, .checks = &.{"budget"} },
    .{ .m = .budget_same_first, .checks = &.{"budget"} },
    .{ .m = .read_ahead_charges_preallocate, .checks = &.{"carve"} },
    .{ .m = .manifest_not_last, .checks = &.{"publish order"} },
    .{ .m = .scan_keeps_temps, .checks = &.{ "crashes", "recovery" } },
    .{ .m = .recover_no_size_check, .checks = &.{"recovery"} },
    .{ .m = .dedup_without_sha, .checks = &.{"dedup sha"} },
    .{ .m = .guard_cancels_only, .checks = &.{"guard"} },
    .{ .m = .evict_shared_first, .checks = &.{"eviction order"} },
    .{ .m = .floor_ignored, .checks = &.{"floor"} },
    .{ .m = .owner_lock_ignored, .checks = &.{"owner"} },
    .{ .m = .keep_unpinned, .checks = &.{"keep reuse"} },
    .{ .m = .forget_skips_reading, .checks = &.{"forget damaged"} },
    .{ .m = .recover_drops_on_io, .checks = &.{"recover transient"} },
    .{ .m = .handover_unreserved, .checks = &.{"hand over"} },
    .{ .m = .prune_other_families, .checks = &.{"prune families"} },
};

fn checkNamed(name: []const u8) *const fn (Allocator) anyerror!void {
    for (checks) |ch| if (std.mem.eql(u8, ch.name, name)) return ch.f;
    @panic("no such check");
}

test "spill store mutants: every authored fix switched back off is caught" {
    // a mutant may leak what its broken path never frees: its runs use the page allocator
    const gpa = std.heap.page_allocator;
    quiet = std.c.getenv("SPILL_VERBOSE") == null;
    defer {
        quiet = false;
        mutants.current = .none;
    }
    var survived: usize = 0;
    var crashes: usize = 0;
    for (kills) |k| {
        mutants.current = k.m;
        if (!quiet) std.debug.print("mutant {t}\n", .{k.m});
        var killed = false;
        for (k.checks) |name| {
            // a kill is a failed assertion (CheckFailed) only; any other error is the harness or the check
            // breaking on its way (GLM's harness requires a FAIL line, not a crash)
            if (checkNamed(name)(gpa)) |_| {} else |e| if (e == CheckFailed) {
                killed = true;
            } else {
                std.debug.print("CRASHED {t} in {s}: {t}\n", .{ k.m, name, e });
                crashes += 1;
            }
        }
        mutants.current = .none;
        if (!killed) {
            std.debug.print("SURVIVED {t}\n", .{k.m});
            survived += 1;
        }
    }
    try testing.expectEqual(@as(usize, 0), survived);
    try testing.expectEqual(@as(usize, 0), crashes);
}
