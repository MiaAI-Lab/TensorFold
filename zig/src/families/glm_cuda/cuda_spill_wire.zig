//! Spill v2's messages between the ranks of a GLM pair: rank 0's decisions for every rank's disk tier, in the order its
//! calls make them. Port of wojo's GLM 0110 (PR #124: HAS, LOAD, LOADPART, LOADED, FLUSH) in the shape of the DeepSeek
//! port's rank_wire.zig, but carried by Flash Next's mirror (cuda_mirror.zig) instead of a wire of its own: rank 1
//! replays rank 0's calls from one ordered record stream, so a message needs no sequence number, epoch or reply frame.
//! Two kinds:
//!   - notes (one-way): the lane backend's `.tier` note (cuda_lanes.Note) in the mirror's next batch, or at once when
//!     rank 0 flushes them (a restore's progress, staged writes and the flush must not wait for the next collective:
//!     rank 1 has no idle hook, it waits in a receive);
//!   - asks: the mirror's `spill` call, sent at once; every rank then gives `width` words to one all-gather
//!     (cuda_lanes.Gate.gather) and acts on every rank's words in rank order, so all reach the same verdict
//!     (`presence`, `validation`). A rank whose own step failed answers with its failure, never skips the gather.
//! A message is op u8, then its fields in this order, little endian, no padding, nothing after it:
//!   keep      kept u64, at u32, point [32], group u64, shared u8, root [16], part opt u32, forks (u32 count, then
//!             name [16] and at u32 each), origin opt (key [32], at u32)
//!   begin     txn u64, kept u64, point [32], at u32, blocks u32
//!   progress  txn u64, upto u32
//!   cancel    txn u64, why u8 (Failure), forget u8
//!   touch, share, forget   point [32]
//!   pump, flush            budget_ns u64
//!   has       point [32], at u32, blocks u32           (answer: 1 word, an Absence)
//!   validate  txn u64, shared u8                       (answer: 1 word, a Failure; none is ok)
//!   stats     nothing                                  (answer: stats_words words, `words` of the tier's counters)
//! where `opt X` is a u8 (0 absent, 1 present) and then X.
const std = @import("std");
const core = @import("core");
const format = core.spill.format;

pub const Digest = [32]u8;

/// Bumped with any change to the messages or their meaning (the fingerprint's `wire` field carries it, so ranks built
/// with another protocol refuse to start together).
pub const version: u16 = 1;

pub const Op = enum(u8) {
    // notes
    keep = 1, // write this rank's copy of kept state `kept` (rank 0's key) as point `point`
    begin = 2, // restore `point` into a new state every rank names `kept`: reserve, then read
    progress = 3, // rank 0 copied blocks [0, upto) of restore `txn`: copy this rank's that far
    cancel = 4, // restore `txn` ends without a commit (cancelled, or failed on rank 0 before validate)
    touch = 5, // the point was resumed (its last use)
    share = 6, // the point became a shared prefix
    forget = 7, // drop the point, published or queued
    pump = 8, // rank 0 staged its queued writes for `budget_ns`: stage this rank's as long
    flush = 9, // a clean stop: stage and drain this rank's writes within `budget_ns`
    // asks
    has = 16, // does this rank hold every block of the point, published?
    validate = 17, // did this rank's copies of restore `txn` pass their checks?
    stats = 18, // this rank's counters
    _,
};

/// A rank's answer to `has` (DeepSeek rank_wire's codes): present, else why not.
pub const Absence = enum(u8) {
    present = 0,
    missing = 1, // not held (pruned by this rank's cap, floor or turns; a write it cancelled or never made)
    incompatible = 2, // held under another layout or compat
    incomplete = 3, // this rank's own write of it is still queued (it will be there shortly)
    damaged = 4, // its copy failed a check
    disabled = 5, // this rank runs without the tier
    _,
};

/// Why a restore failed on a rank (a `validate` answer, a `cancel`'s reason); none: it passed.
pub const Failure = enum(u8) {
    none = 0,
    presence = 1, // a rank lacked the point (has)
    reserve = 2, // the destination could not be made
    read = 3, // a disk read failed
    crc = 4, // a block or the small state failed its CRC or digest
    truncated = 5, // a file was shorter than its manifest
    missing = 6, // a file was gone
    cancelled = 7, // the request went away
    timeout = 8, // this rank's own copy bound passed (an answer, not silence)
    unpack = 9, // a block or the small state did not install
    superseded = 10, // a better source appeared
    shutdown = 11,
    unknown = 12, // this rank holds no restore by that txn
    _,
};

/// Whether a failure is damage to the stored point: every rank then forgets it (DeepSeek spill_tier `damage`).
pub fn damage(why: Failure) bool {
    return switch (why) {
        .crc, .truncated, .missing, .unpack => true,
        else => false,
    };
}

pub const Keep = struct {
    kept: u64,
    at: u32,
    /// format.pointKey of the state's ids (lineage-free): the follower writes its copy only when its own ids give it
    point: Digest,
    group: u64,
    shared: bool,
    /// the store's lineage of the write (its forks live in the decode arena)
    lineage: format.Lineage,
    origin: ?format.Origin = null,
};
pub const Begin = struct { txn: u64, kept: u64, point: Digest, at: u32, blocks: u32 };
pub const Progress = struct { txn: u64, upto: u32 };
pub const Cancel = struct { txn: u64, why: Failure, forget: bool };
pub const Has = struct { point: Digest, at: u32, blocks: u32 };
pub const Validate = struct { txn: u64, shared: bool };

pub const Msg = union(enum) {
    keep: Keep,
    begin: Begin,
    progress: Progress,
    cancel: Cancel,
    touch: Digest,
    share: Digest,
    forget: Digest,
    pump: u64,
    flush: u64,
    has: Has,
    validate: Validate,
    stats,

    pub fn op(m: Msg) Op {
        return switch (m) {
            inline else => |_, tag| @field(Op, @tagName(tag)),
        };
    }

    /// The words each rank answers an ask with (null: a note, no answer).
    pub fn width(m: Msg) ?usize {
        return switch (m) {
            .has, .validate => 1,
            .stats => stats_words,
            else => null,
        };
    }
};

/// The most words a rank answers with (`stats`): a count, then the counters.
pub const stats_words = 64;
/// The most ranks an ask's gather holds (cuda_comm.max_world).
pub const max_world = 4;

/// A forks count past this is refused (core/spill format.Limits' default).
pub const max_forks = 4096;

// -- encoding -----------------------------------------------------------------------------------------------------

const Out = struct {
    list: *std.ArrayList(u8),
    gpa: std.mem.Allocator,

    fn int(o: Out, comptime T: type, v: T) !void {
        var b: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &b, v, .little);
        try o.list.appendSlice(o.gpa, &b);
    }
    fn flag(o: Out, v: bool) !void {
        try o.int(u8, @intFromBool(v));
    }
    fn bytes(o: Out, b: []const u8) !void {
        try o.list.appendSlice(o.gpa, b);
    }
};

/// `m`'s bytes, appended to `list`.
pub fn encode(list: *std.ArrayList(u8), gpa: std.mem.Allocator, m: Msg) !void {
    const o: Out = .{ .list = list, .gpa = gpa };
    try o.int(u8, @backingInt(m.op()));
    switch (m) {
        .keep => |k| {
            if (k.lineage.forks.len > max_forks) return error.TooManyForks;
            try o.int(u64, k.kept);
            try o.int(u32, k.at);
            try o.bytes(&k.point);
            try o.int(u64, k.group);
            try o.flag(k.shared);
            try o.bytes(&k.lineage.root);
            try o.flag(k.lineage.part != null);
            if (k.lineage.part) |p| try o.int(u32, p);
            try o.int(u32, @intCast(k.lineage.forks.len));
            for (k.lineage.forks) |f| {
                try o.bytes(&f.name);
                try o.int(u32, f.at);
            }
            try o.flag(k.origin != null);
            if (k.origin) |g| {
                try o.bytes(&g.key);
                try o.int(u32, g.at);
            }
        },
        .begin => |b| {
            try o.int(u64, b.txn);
            try o.int(u64, b.kept);
            try o.bytes(&b.point);
            try o.int(u32, b.at);
            try o.int(u32, b.blocks);
        },
        .progress => |p| {
            try o.int(u64, p.txn);
            try o.int(u32, p.upto);
        },
        .cancel => |c| {
            try o.int(u64, c.txn);
            try o.int(u8, @backingInt(c.why));
            try o.flag(c.forget);
        },
        .touch, .share, .forget => |p| try o.bytes(&p),
        .pump, .flush => |ns| try o.int(u64, ns),
        .has => |h| {
            try o.bytes(&h.point);
            try o.int(u32, h.at);
            try o.int(u32, h.blocks);
        },
        .validate => |v| {
            try o.int(u64, v.txn);
            try o.flag(v.shared);
        },
        .stats => {},
    }
}

// -- decoding -----------------------------------------------------------------------------------------------------

const In = struct {
    bytes: []const u8,
    at: usize = 0,

    fn int(i: *In, comptime T: type) !T {
        const n = @sizeOf(T);
        if (i.bytes.len - i.at < n) return error.ShortMessage;
        const v = std.mem.readInt(T, i.bytes[i.at..][0..n], .little);
        i.at += n;
        return v;
    }
    fn flag(i: *In) !bool {
        return switch (try i.int(u8)) {
            0 => false,
            1 => true,
            else => error.BadMessage,
        };
    }
    fn fixed(i: *In, comptime n: usize) ![n]u8 {
        if (i.bytes.len - i.at < n) return error.ShortMessage;
        defer i.at += n;
        return i.bytes[i.at..][0..n].*;
    }
    /// A position a point ends at (never 0).
    fn end(i: *In) !u32 {
        const v = try i.int(u32);
        return if (v == 0) error.BadMessage else v;
    }
};

/// A message's fields; a keep's forks live in `arena`. Refused: an unknown op or value, a flag past 1, a zero position or
/// block count, a cut message, bytes after it.
pub fn decode(arena: std.mem.Allocator, bytes: []const u8) !Msg {
    var i: In = .{ .bytes = bytes };
    const op: Op = @fromBackingInt(try i.int(u8));
    const m: Msg = switch (op) {
        .keep => blk: {
            var k: Keep = .{ .kept = try i.int(u64), .at = try i.end(), .point = try i.fixed(32), .group = try i.int(u64), .shared = try i.flag(), .lineage = .{ .root = try i.fixed(16) } };
            if (try i.flag()) k.lineage.part = try i.int(u32);
            const n = try i.int(u32);
            // every fork takes 20 bytes: a corrupt count cannot ask for more than the message holds
            if (n > max_forks or n > (bytes.len - i.at) / 20) return error.ShortMessage;
            const forks = try arena.alloc(format.Fork, n);
            for (forks) |*f| f.* = .{ .name = try i.fixed(16), .at = try i.int(u32) };
            k.lineage.forks = forks;
            if (try i.flag()) k.origin = .{ .key = try i.fixed(32), .at = try i.int(u32) };
            break :blk .{ .keep = k };
        },
        .begin => .{ .begin = .{ .txn = try i.int(u64), .kept = try i.int(u64), .point = try i.fixed(32), .at = try i.end(), .blocks = try i.end() } },
        .progress => .{ .progress = .{ .txn = try i.int(u64), .upto = try i.int(u32) } },
        .cancel => .{ .cancel = .{ .txn = try i.int(u64), .why = try failure(try i.int(u8)), .forget = try i.flag() } },
        .touch => .{ .touch = try i.fixed(32) },
        .share => .{ .share = try i.fixed(32) },
        .forget => .{ .forget = try i.fixed(32) },
        .pump => .{ .pump = try i.int(u64) },
        .flush => .{ .flush = try i.int(u64) },
        .has => .{ .has = .{ .point = try i.fixed(32), .at = try i.end(), .blocks = try i.end() } },
        .validate => .{ .validate = .{ .txn = try i.int(u64), .shared = try i.flag() } },
        .stats => .stats,
        _ => return error.UnknownOp,
    };
    if (i.at != bytes.len) return error.TrailingBytes;
    return m;
}

fn failure(v: u8) !Failure {
    const f: Failure = @fromBackingInt(v);
    return if (std.enums.tagName(Failure, f) == null) error.BadMessage else f;
}

// -- verdicts -----------------------------------------------------------------------------------------------------

/// What every rank does after an ask, from every rank's answer (the same answers, so the same verdict on every rank).
pub const Verdict = struct {
    /// every rank answered yes: the restore goes on (has) or commits (validate)
    ok: bool,
    /// every rank drops the point (a rank lacks it or holds it damaged, or a read found it damaged)
    forget: bool = false,
    /// the first rank that said no, and what it said (an Absence or a Failure's value)
    rank: ?u16 = null,
    code: u8 = 0,
};

/// `has`, from every rank's Absence (one word each, rank order): every rank holds the point, or no rank restores it.
/// A rank that lacks it (`missing`), holds it under another layout (`incompatible`) or damaged makes every rank forget
/// it: rank 0's copy could never restore on the pair, and the next keep writes it again on every rank. One still
/// writing it (`incomplete`) or running without the tier keeps it. A code this build does not know counts as damaged.
pub fn presence(codes: []const u64) Verdict {
    var v: Verdict = .{ .ok = true };
    for (codes, 0..) |c, r| {
        const a: Absence = if (c > 255) .damaged else @fromBackingInt(@as(u8, @intCast(c)));
        if (a == .present) continue;
        if (v.ok) {
            v = .{ .ok = false, .rank = @intCast(r), .code = @intCast(@min(c, 255)) };
        }
        switch (a) {
            .incomplete, .disabled => {},
            else => v.forget = true,
        }
    }
    return v;
}

/// `validate`, from every rank's Failure (one word each, rank order): the restore commits only when every rank's
/// copies passed; else every rank aborts it, and every rank forgets the point when any rank's failure is damage.
pub fn validation(codes: []const u64) Verdict {
    var v: Verdict = .{ .ok = true };
    for (codes, 0..) |c, r| {
        if (c == 0) continue;
        const f: Failure = if (c > 255) .unpack else @fromBackingInt(@as(u8, @intCast(c)));
        if (v.ok) v = .{ .ok = false, .rank = @intCast(r), .code = @intCast(@min(c, 255)) };
        // a code this build does not know: treated as damage, so no rank serves the point again
        if (damage(f) or std.enums.tagName(Failure, f) == null) v.forget = true;
    }
    return v;
}

/// What a rank without the tier answers an ask with: `disabled` to has (the point is kept elsewhere), `unknown` to
/// validate, zero counters to stats.
pub fn disabled(m: Msg, out: []u64) void {
    @memset(out, 0);
    switch (m) {
        .has => out[0] = @backingInt(Absence.disabled),
        .validate => out[0] = @backingInt(Failure.unknown),
        else => {},
    }
}

/// A struct of u64 counters as an answer's words: its field count, then each field in declaration order.
pub fn words(comptime T: type, v: *const T, out: []u64) void {
    const names = @typeInfo(T).@"struct".field_names;
    comptime std.debug.assert(names.len + 1 <= stats_words);
    std.debug.assert(out.len >= names.len + 1);
    @memset(out, 0);
    out[0] = names.len;
    inline for (names, 1..) |name, i| out[i] = @field(v, name);
}

/// The counters `words` wrote (fields past the count it gives stay zero).
pub fn fromWords(comptime T: type, in: []const u64) T {
    var v: T = .{};
    if (in.len == 0) return v;
    const n: usize = @intCast(@min(in[0], in.len - 1));
    inline for (@typeInfo(T).@"struct".field_names, 1..) |name, i| {
        if (i <= n) @field(v, name) = in[i];
    }
    return v;
}

// -- tests --------------------------------------------------------------------------------------------------------

fn roundTrip(arena: std.mem.Allocator, m: Msg) !Msg {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(gpa);
    try encode(&list, gpa, m);
    // a message cut anywhere is refused, never read past its end; one byte more is refused too
    for (0..list.items.len) |cut| {
        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        if (decode(scratch.allocator(), list.items[0..cut])) |_| return error.TestCutMessageDecoded else |_| {}
    }
    try list.append(gpa, 0);
    try std.testing.expectError(error.TrailingBytes, decode(arena, list.items));
    return decode(arena, list.items[0 .. list.items.len - 1]);
}

test "every message survives its bytes, a cut or longer one is refused, and only asks have answers" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const forks = [_]format.Fork{ .{ .name = @splat(7), .at = 2048 }, .{ .name = @splat(9), .at = 4160 } };
    const keep: Keep = .{ .kept = 0xdead_beef_0000_1111, .at = 4288, .point = @splat(3), .group = 5 << 32 | 2, .shared = true, .lineage = .{ .root = @splat(1), .part = 1, .forks = &forks }, .origin = .{ .key = @splat(4), .at = 4096 } };
    const k = (try roundTrip(a, .{ .keep = keep })).keep;
    try std.testing.expectEqual(keep.kept, k.kept);
    try std.testing.expectEqual(keep.at, k.at);
    try std.testing.expectEqual(keep.group, k.group);
    try std.testing.expect(k.shared and k.lineage.eql(keep.lineage));
    try std.testing.expectEqual(keep.origin.?, k.origin.?);
    const plain = (try roundTrip(a, .{ .keep = .{ .kept = 1, .at = 64, .point = @splat(0), .group = 0, .shared = false, .lineage = .{ .root = @splat(2) } } })).keep;
    try std.testing.expect(plain.lineage.part == null and plain.lineage.forks.len == 0 and plain.origin == null);
    try std.testing.expectEqual(Begin{ .txn = 3, .kept = 4, .point = @splat(5), .at = 4096, .blocks = 2 }, (try roundTrip(a, .{ .begin = .{ .txn = 3, .kept = 4, .point = @splat(5), .at = 4096, .blocks = 2 } })).begin);
    try std.testing.expectEqual(Progress{ .txn = 3, .upto = 1 }, (try roundTrip(a, .{ .progress = .{ .txn = 3, .upto = 1 } })).progress);
    try std.testing.expectEqual(Cancel{ .txn = 3, .why = .crc, .forget = true }, (try roundTrip(a, .{ .cancel = .{ .txn = 3, .why = .crc, .forget = true } })).cancel);
    try std.testing.expectEqual(@as(Digest, @splat(6)), (try roundTrip(a, .{ .forget = @splat(6) })).forget);
    try std.testing.expectEqual(@as(Digest, @splat(6)), (try roundTrip(a, .{ .touch = @splat(6) })).touch);
    try std.testing.expectEqual(@as(Digest, @splat(6)), (try roundTrip(a, .{ .share = @splat(6) })).share);
    try std.testing.expectEqual(@as(u64, 40_000_000), (try roundTrip(a, .{ .pump = 40_000_000 })).pump);
    try std.testing.expectEqual(@as(u64, 60_000_000_000), (try roundTrip(a, .{ .flush = 60_000_000_000 })).flush);
    const has: Msg = .{ .has = .{ .point = @splat(8), .at = 4288, .blocks = 3 } };
    try std.testing.expectEqual(has.has, (try roundTrip(a, has)).has);
    try std.testing.expectEqual(Validate{ .txn = 9, .shared = true }, (try roundTrip(a, .{ .validate = .{ .txn = 9, .shared = true } })).validate);
    try std.testing.expect((try roundTrip(a, .stats)) == .stats);
    try std.testing.expectEqual(@as(?usize, 1), has.width());
    try std.testing.expectEqual(@as(?usize, 1), (Msg{ .validate = .{ .txn = 1, .shared = false } }).width());
    try std.testing.expectEqual(@as(?usize, stats_words), (Msg{ .stats = {} }).width());
    try std.testing.expectEqual(@as(?usize, null), (Msg{ .pump = 1 }).width());
    try std.testing.expectEqual(@as(?usize, null), (Msg{ .keep = keep }).width());
}

test "the message layout is the documented one, and bad values are refused" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(gpa);
    try encode(&list, gpa, .{ .progress = .{ .txn = 0x0102030405060708, .upto = 5 } });
    try std.testing.expectEqualSlices(u8, &.{ 3, 8, 7, 6, 5, 4, 3, 2, 1, 5, 0, 0, 0 }, list.items);
    list.clearRetainingCapacity();
    try encode(&list, gpa, .{ .cancel = .{ .txn = 1, .why = .missing, .forget = true } });
    try std.testing.expectEqualSlices(u8, &.{ 4, 1, 0, 0, 0, 0, 0, 0, 0, 6, 1 }, list.items);
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectError(error.UnknownOp, decode(a, &.{99}));
    try std.testing.expectError(error.BadMessage, decode(a, &.{ 4, 1, 0, 0, 0, 0, 0, 0, 0, 99, 1 })); // a Failure this build does not know
    try std.testing.expectError(error.BadMessage, decode(a, &.{ 4, 1, 0, 0, 0, 0, 0, 0, 0, 6, 2 })); // a flag of 2
    var has: [41]u8 = @splat(0);
    has[0] = @backingInt(Op.has);
    has[37] = 1; // at 0: no point ends at 0
    try std.testing.expectError(error.BadMessage, decode(a, &has));
    // a keep whose forks count is past what its bytes could hold
    list.clearRetainingCapacity();
    try encode(&list, gpa, .{ .keep = .{ .kept = 1, .at = 64, .point = @splat(0), .group = 0, .shared = false, .lineage = .{ .root = @splat(2) } } });
    std.mem.writeInt(u32, list.items[1 + 8 + 4 + 32 + 8 + 1 + 16 + 1 ..][0..4], 1_000_000, .little);
    try std.testing.expectError(error.ShortMessage, decode(a, list.items));
    var many: [max_forks + 1]format.Fork = undefined;
    for (&many) |*f| f.* = .{ .name = @splat(0), .at = 1 };
    try std.testing.expectError(error.TooManyForks, encode(&list, gpa, .{ .keep = .{ .kept = 1, .at = 64, .point = @splat(0), .group = 0, .shared = false, .lineage = .{ .root = @splat(2), .forks = &many } } }));
}

test "has: every rank holds the point or no rank restores it; a lacking or damaged rank makes every rank forget it" {
    const P = @backingInt(Absence.present);
    try std.testing.expectEqual(Verdict{ .ok = true }, presence(&.{ P, P }));
    try std.testing.expectEqual(Verdict{ .ok = true }, presence(&.{ P, P, P, P }));
    try std.testing.expectEqual(Verdict{ .ok = false, .forget = true, .rank = 1, .code = 1 }, presence(&.{ P, @backingInt(Absence.missing) }));
    try std.testing.expectEqual(Verdict{ .ok = false, .forget = true, .rank = 1, .code = 2 }, presence(&.{ P, @backingInt(Absence.incompatible) }));
    try std.testing.expectEqual(Verdict{ .ok = false, .forget = true, .rank = 0, .code = 4 }, presence(&.{ @backingInt(Absence.damaged), P }));
    // still being written there, or a rank without the tier: refused, kept
    try std.testing.expectEqual(Verdict{ .ok = false, .rank = 1, .code = 3 }, presence(&.{ P, @backingInt(Absence.incomplete) }));
    try std.testing.expectEqual(Verdict{ .ok = false, .rank = 1, .code = 5 }, presence(&.{ P, @backingInt(Absence.disabled) }));
    // the first rank that refused is named; any rank's damage still forgets
    try std.testing.expectEqual(Verdict{ .ok = false, .forget = true, .rank = 1, .code = 3 }, presence(&.{ P, @backingInt(Absence.incomplete), @backingInt(Absence.missing), P }));
    // a code this build does not know: damaged
    try std.testing.expectEqual(Verdict{ .ok = false, .forget = true, .rank = 1, .code = 77 }, presence(&.{ P, 77 }));
    try std.testing.expectEqual(Verdict{ .ok = false, .forget = true, .rank = 1, .code = 255 }, presence(&.{ P, 1 << 40 }));
}

test "validate: a restore commits only when every rank's copies passed; damage anywhere makes every rank forget it" {
    try std.testing.expectEqual(Verdict{ .ok = true }, validation(&.{ 0, 0 }));
    try std.testing.expectEqual(Verdict{ .ok = false, .forget = true, .rank = 1, .code = 4 }, validation(&.{ 0, @backingInt(Failure.crc) }));
    try std.testing.expectEqual(Verdict{ .ok = false, .forget = true, .rank = 0, .code = 5 }, validation(&.{ @backingInt(Failure.truncated), 0 }));
    try std.testing.expectEqual(Verdict{ .ok = false, .forget = true, .rank = 1, .code = 6 }, validation(&.{ 0, @backingInt(Failure.missing) }));
    try std.testing.expectEqual(Verdict{ .ok = false, .forget = true, .rank = 1, .code = 9 }, validation(&.{ 0, @backingInt(Failure.unpack) }));
    // not damage: aborted, the point kept
    for ([_]Failure{ .reserve, .read, .cancelled, .timeout, .superseded, .shutdown, .unknown, .presence }) |f| {
        try std.testing.expectEqual(Verdict{ .ok = false, .rank = 1, .code = @backingInt(f) }, validation(&.{ 0, @backingInt(f) }));
    }
    try std.testing.expectEqual(Verdict{ .ok = false, .forget = true, .rank = 0, .code = 3 }, validation(&.{ @backingInt(Failure.read), @backingInt(Failure.crc) }));
    try std.testing.expectEqual(Verdict{ .ok = false, .forget = true, .rank = 1, .code = 200 }, validation(&.{ 0, 200 }));
    var out: [1]u64 = undefined;
    disabled(.{ .has = .{ .point = @splat(0), .at = 64, .blocks = 1 } }, &out);
    try std.testing.expectEqual(@as(u64, @backingInt(Absence.disabled)), out[0]);
    disabled(.{ .validate = .{ .txn = 1, .shared = false } }, &out);
    try std.testing.expectEqual(@as(u64, @backingInt(Failure.unknown)), out[0]);
}

test "a rank's counters as an answer's words and back, an older count leaving the rest zero" {
    const C = core.spill.store.Counters;
    var c: C = .{};
    inline for (@typeInfo(C).@"struct".field_names, 0..) |name, i| @field(c, name) = 1000 + i;
    var w: [stats_words]u64 = undefined;
    words(C, &c, &w);
    try std.testing.expectEqual(@as(u64, @typeInfo(C).@"struct".field_names.len), w[0]);
    try std.testing.expectEqual(c, fromWords(C, &w));
    w[0] = 3; // a rank that knows three counters
    const old = fromWords(C, &w);
    try std.testing.expectEqual(@as(u64, 1000), old.keeps);
    try std.testing.expectEqual(@as(u64, 1002), old.stored_skips);
    try std.testing.expectEqual(@as(u64, 0), old.short);
    try std.testing.expectEqual(C{}, fromWords(C, &.{}));
}
