//! cuda_shard_keep.zig's tests: prompt borrowing on two ranks' pools, a resume's decision (here, moved home, carried,
//! from 0), each rank's part of a resume's latent rows, and the carry's transfers on a two-rank host model (each rank's
//! planes in host memory, the transfers through a wire whose sends and receives must meet in order and size). Every
//! keep mutant is caught by the check written for it (`killers`).
const std = @import("std");
const shard = @import("cuda_shard.zig");
const keep = @import("cuda_shard_keep.zig");
const Pool = @import("cuda_pool.zig").Pool;
const G = @import("cuda_geometry.zig").Geometry;

const gpa = std.testing.allocator;
const E = G.extent;
const row = G.latent_row;
const planes = G.dsa_layers;

// -- borrowing -------------------------------------------------------------------------------------------------------

/// Two ranks' pools from the same calls: an extent of 2 x E rows in part 0, its prompt borrowing part 1's rows.
fn borrowing() anyerror!bool {
    var p: [2]Pool = .{ try Pool.initParts(gpa, 8 * E, 2), try Pool.initParts(gpa, 8 * E, 2) };
    defer for (&p) |*q| q.deinit();
    var twin: [2]?usize = undefined;
    for (&p, &twin) |*q, *t| {
        const a = (try q.alloc(2 * E)).?;
        (try q.get(a)).owner = 0;
        t.* = try keep.borrow(q, a);
        // held for the fill: no placement takes it back
        if (keep.revocable(q, a) or !q.holding(a)) return false;
        q.hold(a, false);
        if (!keep.revocable(q, a)) return false;
        try q.giveBack(a);
        if (q.hasTwins(a)) return false;
    }
    // placed alike on both ranks, in the other part's first free rows
    if (twin[0] == null or twin[0].? != 4 * E or twin[1].? != twin[0].?) return false;
    // no room in the other part: none, and nothing left behind
    var full = try Pool.initParts(gpa, 8 * E, 2);
    defer full.deinit();
    const a = (try full.alloc(2 * E)).?;
    _ = try full.add(4 * E, 3 * E);
    if ((try keep.borrow(&full, a)) != null or full.hasTwins(a)) return false;
    return true;
}

test "borrowing: a twin in the other part, placed alike on both ranks, held for its fill, none without room" {
    try std.testing.expect(try borrowing());
}

test "borrowing is the two parts' (a part a rank): a pool of three refuses it, with nothing borrowed or held" {
    var p = try Pool.initParts(gpa, 9 * E, 3);
    defer p.deinit();
    const a = (try p.alloc(E)).?;
    (try p.get(a)).owner = 0;
    const free = p.freeRows();
    try std.testing.expectError(error.KvShardNeedsTwoRanks, keep.borrow(&p, a));
    try std.testing.expect(!p.hasTwins(a) and !p.holding(a) and p.freeRows() == free);
}

// -- a resume's decision --------------------------------------------------------------------------------------------------

/// Every case of `resumeIn`: the extent already home; moved home (its new base, and the move across parts applies);
/// carried where home has no room; from 0 with the carry off or a state under the minimum; a twinned extent refused.
fn resuming() anyerror!bool {
    var p = try Pool.initParts(gpa, 8 * E, 2);
    defer p.deinit();
    const a = (try p.alloc(2 * E)).?; // part 0
    const b = (try p.alloc(2 * E)).?; // part 1 (fewer live: none yet counted, more free rows)
    (try p.get(a)).owner = 0;
    (try p.get(b)).owner = 1;
    if (p.partOf((try p.extent(b)).base) != 1) return error.TestUnexpectedResult;
    if ((try keep.resumeIn(&p, a, 0, 5000, true)) != .here) return false;
    if ((try keep.resumeIn(&p, b, 1, 100, false)) != .here) return false; // (at home: no rule about the carry or the size)
    // b's state lives in part 0, which has room: moved there, not carried, whatever the carry
    const moved = try keep.resumeIn(&p, b, 0, 5000, true);
    if (moved != .move or moved.move != 2 * E) return false;
    if ((try keep.resumeIn(&p, b, 0, 100, false)) != .move) return false;
    // part 0 full: carried from the minimum up, else from 0 (by why)
    _ = try p.add(2 * E, 2 * E);
    if ((try keep.resumeIn(&p, b, 0, keep.carry_min, true)) != .carry) return false;
    const off = try keep.resumeIn(&p, b, 0, 5000, false);
    const small = try keep.resumeIn(&p, b, 0, keep.carry_min - 64, true);
    if (off != .recompute or off.recompute != .off or small != .recompute or small.recompute != .small) return false;
    // a home that is no part, or an extent holding twins (a fill's), refused
    if (keep.resumeIn(&p, b, 2, 5000, true)) |_| return false else |_| {}
    var q = try Pool.initParts(gpa, 8 * E, 2);
    defer q.deinit();
    const c = (try q.alloc(E)).?;
    _ = try keep.borrow(&q, c);
    if (keep.resumeIn(&q, c, 1, 5000, true)) |_| return false else |e| if (e != error.Twinned) return false;
    return true;
}

test "a resume: in its state's part as it is, moved there when it has room, else carried, else from 0 (carry off, or under the minimum)" {
    try std.testing.expect(try resuming());
    // the move applies across the parts (the extent is fresh: nothing to copy) and the state's part then holds it
    var p = try Pool.initParts(gpa, 8 * E, 2);
    defer p.deinit();
    _ = (try p.alloc(E)).?;
    const b = (try p.alloc(E)).?;
    const r = try keep.resumeIn(&p, b, 0, 4096, true);
    _ = try p.move(b, r.move, null, true);
    try std.testing.expectEqual(@as(usize, 0), p.partOf((try p.extent(b)).base));
}

test "a restore from disk: started when its resume could be placed in the home part or carried, refused when it could only start from 0" {
    comptime std.debug.assert(keep.carry_min == E);
    var p = try Pool.initParts(gpa, 8 * E, 2);
    defer p.deinit();
    _ = try p.add(0, 3 * E); // part 0: one extent of E rows left
    // room for the state's rows and the next token's: whatever the carry
    try std.testing.expect(try keep.restorable(&p, 0, E - 64, false));
    // E tokens leave the next one no row there: only a carry takes it (from carry_min, E)
    try std.testing.expect(!try keep.restorable(&p, 0, E, false));
    try std.testing.expect(try keep.restorable(&p, 0, E, true));
    // part 0 full (part 1's room does not count): carried from the minimum up, else refused
    _ = try p.add(3 * E, E);
    try std.testing.expect(try keep.restorable(&p, 0, 5000, true));
    try std.testing.expect(!try keep.restorable(&p, 0, 5000, false));
    try std.testing.expect(!try keep.restorable(&p, 0, keep.carry_min - 64, true));
    try std.testing.expect(try keep.restorable(&p, 1, keep.carry_min - 64, false));
    try std.testing.expectError(error.BadPart, keep.restorable(&p, 2, 5000, true));
}

/// Each rank's part for every (home, part): one copier when they agree, else one sender (the home's owner) and one
/// receiver (the extent's), the rank between them none.
fn roles() anyerror!bool {
    for (0..2) |home| for (0..2) |part| {
        var copy: usize = 0;
        var send: usize = 0;
        var recv: usize = 0;
        for (0..2) |rank| {
            const sh = try shard.Shard.init(rank, 2, 8 * E, shard.trash);
            switch (keep.latentRole(sh, home, part)) {
                .copy => copy += 1 + @as(usize, @intFromBool(rank != home)) * 10,
                .send => send += 1 + @as(usize, @intFromBool(rank != home)) * 10,
                .recv => recv += 1 + @as(usize, @intFromBool(rank != part)) * 10,
                .none => {},
            }
        }
        if (home == part and !(copy == 1 and send == 0 and recv == 0)) return false;
        if (home != part and !(copy == 0 and send == 1 and recv == 1)) return false;
    };
    return true;
}

test "a resume's latent rows: copied back where the state and the extent are one rank's, else sent by the state's and received by the extent's" {
    try std.testing.expect(try roles());
}

// -- the carry on two ranks ----------------------------------------------------------------------------------------------

/// The two ranks' devices: a wire of sends the other rank's receives must meet in order and size, host memory for the
/// planes, and checks that the transport writes only the stage and that the transfers are held.
const Wire = struct {
    sends: std.ArrayList([]const u8) = .empty,
    taken: usize = 0,
    fn deinit(w: *Wire) void {
        w.sends.deinit(gpa);
    }
};

const Dev = struct {
    wire: *Wire,
    stage: u64,
    stage_bytes: usize,
    held: bool = false,
    holds: usize = 0,
    bad: ?[]const u8 = null,
    pub fn send(d: *Dev, at: u64, n: usize) !void {
        if (!d.held) d.bad = "a send not held";
        const p: [*]const u8 = @ptrFromInt(at);
        try d.wire.sends.append(gpa, p[0..n]);
    }
    pub fn recv(d: *Dev, at: u64, n: usize) !void {
        if (!d.held) d.bad = "a receive not held";
        if (at != d.stage or n > d.stage_bytes) d.bad = "the transport wrote past the stage";
        if (d.wire.taken >= d.wire.sends.items.len) return error.NoMatchingSend;
        const sent = d.wire.sends.items[d.wire.taken];
        d.wire.taken += 1;
        if (sent.len != n) return error.UnmatchedSizes;
        const p: [*]u8 = @ptrFromInt(at);
        @memcpy(p[0..n], sent);
    }
    pub fn copy(_: *Dev, dst: u64, src: u64, n: usize) !void {
        const p: [*]u8 = @ptrFromInt(dst);
        const q: [*]const u8 = @ptrFromInt(src);
        @memcpy(p[0..n], q[0..n]);
    }
    pub fn hold(d: *Dev) !void {
        d.held = true;
        d.holds += 1;
    }
    pub fn release(d: *Dev) !void {
        d.held = false;
    }
};

/// A carry of `n` rows in pieces of `piece`, the home owner's state to the extent's owner at local row `to`: true when
/// the receiver's planes hold the state's rows there and nothing else of theirs changed, both ranks' calls met, the
/// stage alone took the transport's bytes and every transfer was held.
fn carried(rnd: std.Random, n: usize, piece: usize, to: usize, part_rows: usize) !bool {
    // the home owner's kept state: 11 planes of n rows (its latent section)
    const state = try gpa.alloc(u8, planes * n * row);
    defer gpa.free(state);
    rnd.bytes(state);
    // the receiver's latent planes: a part and its trash rows each, its rows as they were
    const plane_bytes = (part_rows + shard.trash) * row;
    const mine = try gpa.alloc(u8, planes * plane_bytes);
    defer gpa.free(mine);
    rnd.bytes(mine);
    const before = try gpa.dupe(u8, mine);
    defer gpa.free(before);
    const stage = try gpa.alloc(u8, piece * row);
    defer gpa.free(stage);
    var src: [planes]u64 = undefined;
    var dst: [planes]u64 = undefined;
    for (0..planes) |i| {
        src[i] = @intFromPtr(state.ptr) + i * n * row;
        dst[i] = @intFromPtr(mine.ptr) + i * plane_bytes + to * row;
    }
    var wire: Wire = .{};
    defer wire.deinit();
    var d0: Dev = .{ .wire = &wire, .stage = 0, .stage_bytes = 0 };
    var d1: Dev = .{ .wire = &wire, .stage = @intFromPtr(stage.ptr), .stage_bytes = stage.len };
    // the sender's calls first (its sends queue), then the receiver's meet them
    const sent = try keep.carry(&d0, .send, src, @splat(0), n, 0, piece);
    const got = keep.carry(&d1, .recv, @splat(0), dst, n, @intFromPtr(stage.ptr), piece) catch return false;
    if (wire.taken != wire.sends.items.len or sent != got or sent != shard.carryBytes(n)) return false;
    if (d0.bad != null or d1.bad != null or d0.holds != 1 or d1.holds != 1 or d0.held or d1.held) return false;
    for (0..planes) |i| {
        const p = mine[i * plane_bytes ..][0..plane_bytes];
        const b = before[i * plane_bytes ..][0..plane_bytes];
        if (!std.mem.eql(u8, p[to * row ..][0 .. n * row], state[i * n * row ..][0 .. n * row])) return false;
        if (!std.mem.eql(u8, p[0 .. to * row], b[0 .. to * row])) return false;
        if (!std.mem.eql(u8, p[(to + n) * row ..], b[(to + n) * row ..])) return false;
    }
    return true;
}

const carry_cases = [_][4]usize{
    // n, piece, to, part rows
    .{ 2112, 2112, 2048, 4096 }, // one piece a plane
    .{ 2112, 500, 0, 4096 }, // ragged pieces
    .{ 2048, 1024, 2048, 4096 }, // whole pieces
    .{ 1, 7, 6143, 6144 }, // a row, the part's last
};

test "a carry: the home owner's state's latent rows land in the extent owner's planes, through its stage, in pieces both ranks cut alike, held" {
    var prng = std.Random.DefaultPrng.init(1301);
    for (carry_cases) |c| try std.testing.expect(try carried(prng.random(), c[0], c[1], c[2], c[3]));
    // not a carry's role, or no piece
    var wire: Wire = .{};
    defer wire.deinit();
    var d: Dev = .{ .wire = &wire, .stage = 0, .stage_bytes = 0 };
    try std.testing.expectError(error.NotACarry, keep.carry(&d, .copy, @splat(0), @splat(0), 64, 0, 64));
    try std.testing.expectError(error.BadRows, keep.carry(&d, .send, @splat(0), @splat(0), 64, 0, 0));
}

// -- mutants --------------------------------------------------------------------------------------------------------------

fn anyCarry() anyerror!bool {
    var prng = std.Random.DefaultPrng.init(1302);
    for (carry_cases) |c| if (!try carried(prng.random(), c[0], c[1], c[2], c[3])) return false;
    return true;
}

/// The check that must fail under each mutant.
fn killer(m: keep.Mutant) *const fn () anyerror!bool {
    return switch (m) {
        .none => unreachable,
        .borrow_unheld => borrowing,
        .never_here, .no_move, .carry_small => resuming,
        .copy_anywhere => roles,
        .carry_ten_planes, .send_whole, .recv_in_place => anyCarry,
    };
}

test "every keep mutant is caught by the check written for it, which passes unmutated" {
    defer keep.mutant = .none;
    for (std.enums.values(keep.Mutant)) |m| {
        if (m == .none) continue;
        const check = killer(m);
        keep.mutant = .none;
        try std.testing.expect(try check());
        keep.mutant = m;
        const passed = check() catch false;
        if (passed) {
            std.debug.print("keep mutant {t} survived\n", .{m});
            return error.MutantSurvived;
        }
    }
}
