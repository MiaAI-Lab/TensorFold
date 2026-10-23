//! Port of wojo's GLM 0110 spill blocks (PR #124; BadAd84's pinned staging ring, f32226b's read-ahead correction):
//! the store's host staging slots, carved out of caller-owned aligned buffers (pinned host memory on CUDA, allocated
//! and budgeted by the engine before it sizes its pool). The store never allocates, pins or frees staging; it only
//! hands slots out and takes them back. One partition each for block writes, block reads, queued small states and
//! restores' small states, so a backlog of one never starves another.
const std = @import("std");
const fsio = @import("fsio.zig");
const mutants = @import("mutants.zig");
const Allocator = std.mem.Allocator;

pub const alignment = fsio.alignment;
pub const Buffer = fsio.Buffer;

pub const Pool = enum(u2) { write, read, small, restore };
pub const pools = [_]Pool{ .write, .read, .small, .restore };

/// A slot: its partition and index (stable for the store's life; a family keeps one CUDA event per slot).
pub const Slot = packed struct(u32) {
    pool: Pool,
    index: u30,

    pub fn eql(a: Slot, b: Slot) bool {
        return @as(u32, @bitCast(a)) == @as(u32, @bitCast(b));
    }
};

/// The caller's buffers, by partition. Each must start on a 4 KiB boundary; slots are carved whole out of each one.
pub const Staging = struct {
    write: []const Buffer = &.{},
    read: []const Buffer = &.{},
    small: []const Buffer = &.{},
    restore: []const Buffer = &.{},

    fn of(s: Staging, p: Pool) []const Buffer {
        return switch (p) {
            .write => s.write,
            .read => s.read,
            .small => s.small,
            .restore => s.restore,
        };
    }
};

/// Slots of `per` bytes (a multiple of 4 KiB) that fit whole in `buffers`.
pub fn carve(buffers: []const Buffer, per: usize) usize {
    std.debug.assert(per > 0 and per % alignment == 0);
    var n: usize = 0;
    for (buffers) |b| n += b.len / per;
    return n;
}

/// PR 124's accounting: the bytes still to allocate for `want` slots of `per` when `preallocated` buffers (already
/// charged, allocated before the pool was sized) hold what they hold. Preallocated storage is reused, never
/// charged again; only the shortfall is.
pub fn extraBytes(preallocated: []const Buffer, per: usize, want: usize) usize {
    const have = if (mutants.on(.read_ahead_charges_preallocate)) 0 else carve(preallocated, per);
    return if (have >= want) 0 else (want - have) * per;
}

/// A split of an aggregate staging cap (bytes) among the partitions: `restores` restore small slots and `smalls`
/// queued small states first (each holds a whole small state), the rest shared by block writes and reads.
pub const Plan = struct {
    write: usize,
    read: usize,
    small: usize,
    restore: usize,
    block_slot: usize,
    small_slot: usize,

    pub fn bytes(p: Plan) u64 {
        return @as(u64, p.write + p.read) * p.block_slot + @as(u64, p.small + p.restore) * p.small_slot;
    }

    pub fn of(p: Plan, pool: Pool) usize {
        return switch (pool) {
            .write => p.write,
            .read => p.read,
            .small => p.small,
            .restore => p.restore,
        };
    }
};

pub fn plan(cap: u64, block_bytes: u64, small_bytes: u64, smalls: usize, restores: usize) error{StagingTooSmall}!Plan {
    const bs = std.mem.alignForward(u64, @max(block_bytes, 1), alignment);
    const ss = std.mem.alignForward(u64, @max(small_bytes, 1), alignment);
    const fixed = ss * (smalls + restores);
    if (fixed + 4 * bs > cap) return error.StagingTooSmall;
    const blocks = (cap - fixed) / bs;
    const write = blocks / 2;
    return .{ .write = @intCast(write), .read = @intCast(blocks - write), .small = smalls, .restore = restores, .block_slot = @intCast(bs), .small_slot = @intCast(ss) };
}

/// The slots of every partition. Not thread-safe: the store holds its lock around every call.
pub const Slots = struct {
    mem: [4][]Buffer = @splat(&.{}),
    free: [4]std.ArrayList(u30) = @splat(.empty),
    held: [4]usize = @splat(0),
    peak: [4]usize = @splat(0), // each partition's own peak (slots)
    peak_bytes: u64 = 0, // the most every partition held at once (bytes)
    per: [4]usize = @splat(0),
    owner: [4][]u64 = @splat(&.{}), // what holds each slot (the store's job id; 0: free)

    pub fn init(gpa: Allocator, staging: Staging, block_slot: usize, small_slot: usize) !Slots {
        var s: Slots = .{};
        errdefer s.deinit(gpa);
        for (pools) |p| {
            const per = if (p == .write or p == .read) block_slot else small_slot;
            const i = @intFromEnum(p);
            s.per[i] = per;
            const bufs = staging.of(p);
            for (bufs) |b| if (@intFromPtr(b.ptr) % alignment != 0) return error.UnalignedStaging;
            const n = carve(bufs, per);
            if (n > std.math.maxInt(u30)) return error.TooManySlots;
            s.mem[i] = try gpa.alloc(Buffer, n);
            s.owner[i] = try gpa.alloc(u64, n);
            @memset(s.owner[i], 0);
            var k: usize = 0;
            for (bufs) |b| {
                var off: usize = 0;
                while (off + per <= b.len) : (off += per) {
                    s.mem[i][k] = @alignCast(b[off..][0..per]);
                    k += 1;
                }
            }
            try s.free[i].ensureTotalCapacity(gpa, n);
            var j: usize = n;
            while (j > 0) {
                j -= 1;
                s.free[i].appendAssumeCapacity(@intCast(j)); // popped lowest first
            }
        }
        return s;
    }

    pub fn deinit(s: *Slots, gpa: Allocator) void {
        for (pools) |p| {
            const i = @intFromEnum(p);
            gpa.free(s.mem[i]);
            gpa.free(s.owner[i]);
            s.free[i].deinit(gpa);
        }
        s.* = undefined;
    }

    pub fn count(s: *const Slots, p: Pool) usize {
        return s.mem[@intFromEnum(p)].len;
    }

    pub fn available(s: *const Slots, p: Pool) usize {
        return s.free[@intFromEnum(p)].items.len;
    }

    pub fn take(s: *Slots, p: Pool, owner: u64) ?Slot {
        const i = @intFromEnum(p);
        const k = s.free[i].pop() orelse return null;
        std.debug.assert(s.owner[i][k] == 0);
        s.owner[i][k] = owner;
        s.held[i] += 1;
        s.peak[i] = @max(s.peak[i], s.held[i]);
        s.peak_bytes = @max(s.peak_bytes, s.heldBytes());
        return .{ .pool = p, .index = k };
    }

    pub fn give(s: *Slots, slot: Slot) void {
        const i = @intFromEnum(slot.pool);
        std.debug.assert(s.owner[i][slot.index] != 0); // a slot returned twice
        s.owner[i][slot.index] = 0;
        s.held[i] -= 1;
        s.free[i].appendAssumeCapacity(slot.index);
    }

    pub fn holder(s: *const Slots, slot: Slot) u64 {
        return s.owner[@intFromEnum(slot.pool)][slot.index];
    }

    pub fn bytes(s: *const Slots, slot: Slot) Buffer {
        return s.mem[@intFromEnum(slot.pool)][slot.index];
    }

    /// Bytes held now, every partition.
    pub fn heldBytes(s: *const Slots) u64 {
        var n: u64 = 0;
        for (0..4) |i| n += @as(u64, s.held[i]) * s.per[i];
        return n;
    }

    /// The most bytes held at once over every partition (never more than the slots carved: the staging cap). Not
    /// the sum of the partitions' own peaks, which were not reached together.
    pub fn peakBytes(s: *const Slots) u64 {
        return s.peak_bytes;
    }

    /// A partition's own peak, in slots.
    pub fn peakOf(s: *const Slots, p: Pool) usize {
        return s.peak[@intFromEnum(p)];
    }
};

// -- tests ---------------------------------------------------------------------------------------------------------
const testing = std.testing;

test "read-ahead accounting: preallocated buffers are kept and only the shortfall is charged" {
    const gpa = testing.allocator;
    const per = 3 * alignment;
    // a preallocated buffer that holds two and a half slots: two carved, its half slot unused
    const pre = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), 2 * per + per / 2 + alignment / 2);
    defer gpa.free(pre);
    const bufs = [_]Buffer{pre};
    try testing.expectEqual(@as(usize, 2), carve(&bufs, per));
    try testing.expectEqual(@as(usize, 5 * per), extraBytes(&bufs, per, 7));
    try testing.expectEqual(@as(usize, 0), extraBytes(&bufs, per, 2));
    // the extra piece the caller allocates plus the preallocated one carve at least the slots asked
    const extra = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), extraBytes(&bufs, per, 7));
    defer gpa.free(extra);
    const both = [_]Buffer{ pre, extra };
    var s = try Slots.init(gpa, .{ .read = &both }, per, alignment);
    defer s.deinit(gpa);
    try testing.expectEqual(@as(usize, 7), s.count(.read));
    // every slot lies inside one of the caller's buffers, aligned, disjoint
    var seen: usize = 0;
    for (0..7) |k| {
        const b = s.bytes(.{ .pool = .read, .index = @intCast(k) });
        try testing.expect(@intFromPtr(b.ptr) % alignment == 0 and b.len == per);
        const in_pre = @intFromPtr(b.ptr) >= @intFromPtr(pre.ptr) and @intFromPtr(b.ptr) + per <= @intFromPtr(pre.ptr) + pre.len;
        const in_extra = @intFromPtr(b.ptr) >= @intFromPtr(extra.ptr) and @intFromPtr(b.ptr) + per <= @intFromPtr(extra.ptr) + extra.len;
        try testing.expect(in_pre or in_extra);
        if (in_pre) seen += 1;
    }
    try testing.expectEqual(@as(usize, 2), seen);
}

test "slots: taken and given back, peaks and holders tracked, partitions independent" {
    const gpa = testing.allocator;
    const a = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), 4 * alignment);
    defer gpa.free(a);
    const b = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), 2 * alignment);
    defer gpa.free(b);
    var s = try Slots.init(gpa, .{ .write = &.{a}, .small = &.{b} }, alignment, 2 * alignment);
    defer s.deinit(gpa);
    try testing.expectEqual(@as(usize, 4), s.count(.write));
    try testing.expectEqual(@as(usize, 1), s.count(.small));
    try testing.expectEqual(@as(usize, 0), s.count(.read));
    const x = s.take(.write, 7).?;
    const y = s.take(.write, 8).?;
    try testing.expectEqual(@as(u64, 7), s.holder(x));
    try testing.expect(s.take(.read, 1) == null);
    const z = s.take(.small, 9).?;
    try testing.expect(s.take(.small, 9) == null);
    try testing.expectEqual(@as(u64, 2 * alignment + 2 * alignment), s.heldBytes());
    s.give(x);
    s.give(y);
    s.give(z);
    try testing.expectEqual(@as(u64, 0), s.heldBytes());
    try testing.expectEqual(@as(u64, 4 * alignment), s.peakBytes());
}

test "slots: the peak is what was held at once, not the partitions' peaks summed" {
    const gpa = testing.allocator;
    const a = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), 4 * alignment);
    defer gpa.free(a);
    const b = try gpa.alignedAlloc(u8, .fromByteUnits(alignment), 4 * alignment);
    defer gpa.free(b);
    var s = try Slots.init(gpa, .{ .write = &.{a}, .read = &.{b} }, alignment, alignment);
    defer s.deinit(gpa);
    // every write slot, given back, then every read slot: never more than four slots at once
    var held: [4]Slot = undefined;
    for (&held, 0..) |*x, k| x.* = s.take(.write, k + 1).?;
    for (held) |x| s.give(x);
    for (&held, 0..) |*x, k| x.* = s.take(.read, k + 1).?;
    try testing.expectEqual(@as(u64, 4 * alignment), s.peakBytes());
    try testing.expectEqual(@as(usize, 4), s.peakOf(.write));
    try testing.expectEqual(@as(usize, 4), s.peakOf(.read));
    s.give(held[0]);
    _ = s.take(.write, 9).?; // (four again)
    try testing.expectEqual(@as(u64, 4 * alignment), s.peakBytes());
}

test "plan: small states first, the rest split between writes and reads" {
    const p = try plan(128 << 20, 1_822_720, 6_684_672, 4, 2);
    try testing.expectEqual(@as(usize, 1_822_720), p.block_slot); // (445 pages: already aligned)
    try testing.expectEqual(@as(usize, 6_684_672), p.small_slot);
    try testing.expect(p.bytes() <= 128 << 20);
    try testing.expect(p.write >= 2 and p.read >= 2 and p.small == 4 and p.restore == 2);
    try testing.expectError(error.StagingTooSmall, plan(8 << 20, 1_822_720, 6_684_672, 4, 2));
}
