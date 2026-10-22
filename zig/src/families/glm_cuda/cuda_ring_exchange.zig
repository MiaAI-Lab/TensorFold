//! The routing of the Python engine's ring exchange (tensorfold/cuda/comm.py `_exchange_ring` / `_ring_links`, recipe
//! patches 0098 renumbered 0112 in the v1.11 tree): an all-to-all of row blocks on a ring of ranks cabled to their two
//! neighbours only. A message from rank s to rank t goes the shorter way round, one hop a step; to the rank opposite
//! (N even) half of each tensor goes each way (the first `numel / 2` elements clockwise). floor(N / 2) steps: in step h
//! every rank runs one two-rank all-gather on each of its two links, carrying what leaves that way on its h-th hop.
//!
//! This file holds the schedule alone (which bytes cross which link in which step, in the one order both ends of a link
//! list them), as pure functions with a byte-level simulation test, so the NCCL half (neighbour communicators, graph
//! files, a stream a link, events between hops) can be written against a checked plan. The Zig GLM prefill does not
//! split rows yet (it gathers whole fp32 partials: HANDOFF-ZIG.md P8b), so nothing calls this on the GPU today.
const std = @import("std");

pub const max_ranks = 4;
pub const max_tensors = 4;
pub const max_msgs = max_ranks * max_tensors * 2;

/// One message (or one half of one): destination rank, tensor index, element range, and the hops it travels.
pub const Msg = struct { t: u8, i: u8, lo: usize, hi: usize, hops: u8 };

pub const Moving = struct {
    n: usize = 0,
    msgs: [max_msgs]Msg = undefined,
};

const Seg = struct { lo: usize, hi: usize, dir: i8, hops: u8 };

/// The segments of the message s -> t of tensor i with `numel` elements: (lo, hi, direction, hops).
fn segments(n_ranks: usize, s: usize, t: usize, numel: usize, out: *[2]Seg) usize {
    const d = (t + n_ranks - s) % n_ranks;
    if (2 * d < n_ranks) {
        out[0] = .{ .lo = 0, .hi = numel, .dir = 1, .hops = @intCast(d) };
        return 1;
    }
    if (2 * d > n_ranks) {
        out[0] = .{ .lo = 0, .hi = numel, .dir = -1, .hops = @intCast(n_ranks - d) };
        return 1;
    }
    out[0] = .{ .lo = 0, .hi = numel / 2, .dir = 1, .hops = @intCast(d) };
    out[1] = .{ .lo = numel / 2, .hi = numel, .dir = -1, .hops = @intCast(d) };
    return 2;
}

/// The messages from `src` that travel `direction` (+1: to the next rank) on their h-th hop, in the one order both ends
/// of a link list them: by destination, tensor, segment. `numel[i]` is tensor i's element count, the same for every
/// peer (row blocks of equal shares), so a relaying rank knows each message's size.
pub fn moving(n_ranks: usize, numel: []const usize, src: usize, direction: i8, h: u8) Moving {
    var m: Moving = .{};
    for (0..n_ranks) |t| {
        if (t == src) continue;
        for (numel, 0..) |ne, i| {
            var segs: [2]Seg = undefined;
            const k = segments(n_ranks, src, t, ne, &segs);
            for (segs[0..k]) |sg| {
                if (sg.dir == direction and sg.hops >= h and sg.hi > sg.lo) {
                    m.msgs[m.n] = .{ .t = @intCast(t), .i = @intCast(i), .lo = sg.lo, .hi = sg.hi, .hops = sg.hops };
                    m.n += 1;
                }
            }
        }
    }
    return m;
}

/// What rank `me` sends on its link toward `direction` in step h and what arrives there: the source of the bytes that
/// leave (which `held` entries or, in step 1, which `sends` it packs) and of those that arrive.
pub const Step = struct { src_out: usize, src_in: usize, leave: Moving, arrive: Moving };

pub fn step(n_ranks: usize, numel: []const usize, me: usize, direction: i8, h: u8) Step {
    const n: i64 = @intCast(n_ranks);
    const d: i64 = direction;
    const src_out: usize = @intCast(@mod(@as(i64, @intCast(me)) - d * (h - 1), n));
    const src_in: usize = @intCast(@mod(@as(i64, @intCast(me)) + d * h, n));
    return .{ .src_out = src_out, .src_in = src_in, .leave = moving(n_ranks, numel, src_out, direction, h), .arrive = moving(n_ranks, numel, src_in, -direction, h) };
}

/// The launch order of a rank's two links in a step (so link kernels that cannot run at once never wait on each other
/// round the ring): the link to the previous rank first, except on rank 0, whose link 0 (to the next rank) comes first.
pub fn linkOrder(n_ranks: usize, me: usize) [2]i8 {
    const to_prev = (me + n_ranks - 1) % n_ranks;
    // sort by the index of the link: link a joins rank a and rank a + 1; the link toward the next rank is `me`, toward the
    // previous one `me - 1`
    return if (to_prev < me) .{ -1, 1 } else .{ 1, -1 };
}

fn Sim(comptime elem: usize) type {
    return struct {
        const Self = @This();
        n: usize,
        numel: []const usize,
        // sends[rank][peer][tensor] -> bytes
        sends: [max_ranks][max_ranks][max_tensors][]u8 = undefined,
        recvs: [max_ranks][max_ranks][max_tensors][]u8 = undefined,

        fn chunk(self: *Self, s: usize, t: usize, i: usize, lo: usize, hi: usize) []u8 {
            return self.sends[s][t][i][lo * elem .. hi * elem];
        }
    };
}

test "the ring exchange delivers every message to its rank, for 2 to 4 ranks and odd sizes" {
    const gpa = std.testing.allocator;
    for ([_]usize{ 2, 3, 4 }) |n_ranks| {
        for ([_][]const usize{ &.{7}, &.{5, 12}, &.{1, 2, 3} }) |numel| {
            const elem = 4;
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            const a = arena.allocator();
            var sim: Sim(elem) = .{ .n = n_ranks, .numel = numel };
            for (0..n_ranks) |s| for (0..n_ranks) |t| for (numel, 0..) |ne, i| {
                sim.sends[s][t][i] = try a.alloc(u8, ne * elem);
                sim.recvs[s][t][i] = try a.alloc(u8, ne * elem);
                @memset(sim.recvs[s][t][i], 0);
                for (sim.sends[s][t][i], 0..) |*b, k| b.* = @intCast((s * 61 + t * 29 + i * 13 + k * 5 + 1) & 0xff);
            };
            // held[rank]: relayed bytes keyed by (src, t, i, lo)
            const Key = struct { s: usize, t: usize, i: usize, lo: usize };
            var held: [max_ranks]std.AutoHashMap(Key, []u8) = undefined;
            for (0..n_ranks) |r| held[r] = std.AutoHashMap(Key, []u8).init(a);
            var h: u8 = 1;
            while (h <= n_ranks / 2) : (h += 1) {
                // every rank's packed bytes this step, per link direction, then what its neighbours read
                var wire: [max_ranks][2][]u8 = undefined; // wire[rank][0: toward +1, 1: toward -1]
                for (0..n_ranks) |me| for ([_]i8{ 1, -1 }) |dir| {
                    const st = step(n_ranks, numel, me, dir, h);
                    var total: usize = 0;
                    for (st.leave.msgs[0..st.leave.n]) |m| total += (m.hi - m.lo) * elem;
                    const buf = try a.alloc(u8, total);
                    var at: usize = 0;
                    for (st.leave.msgs[0..st.leave.n]) |m| {
                        const bytes = if (h == 1) sim.chunk(me, m.t, m.i, m.lo, m.hi) else held[me].get(.{ .s = st.src_out, .t = m.t, .i = m.i, .lo = m.lo }).?;
                        if (h != 1) _ = held[me].remove(.{ .s = st.src_out, .t = m.t, .i = m.i, .lo = m.lo });
                        @memcpy(buf[at..][0..bytes.len], bytes);
                        at += bytes.len;
                    }
                    wire[me][if (dir == 1) 0 else 1] = buf;
                };
                for (0..n_ranks) |me| for ([_]i8{ 1, -1 }) |dir| {
                    const st = step(n_ranks, numel, me, dir, h);
                    const neighbour = st.src_in; // not always the neighbour: the bytes' source; the link's other end is me + dir
                    _ = neighbour;
                    const other = (me + n_ranks + @as(usize, if (dir == 1) 1 else n_ranks - 1)) % n_ranks;
                    // the neighbour sent on its link toward -dir
                    const theirs = wire[other][if (dir == 1) 1 else 0];
                    var at: usize = 0;
                    for (st.arrive.msgs[0..st.arrive.n]) |m| {
                        const bytes = theirs[at..][0 .. (m.hi - m.lo) * elem];
                        at += bytes.len;
                        if (m.hops == h) {
                            try std.testing.expectEqual(@as(usize, m.t), me);
                            @memcpy(sim.recvs[me][st.src_in][m.i][m.lo * elem .. m.hi * elem], bytes);
                        } else {
                            try held[me].put(.{ .s = st.src_in, .t = m.t, .i = m.i, .lo = m.lo }, bytes);
                        }
                    }
                    try std.testing.expectEqual(theirs.len, at);
                };
            }
            for (0..n_ranks) |r| try std.testing.expectEqual(@as(u32, 0), held[r].count());
            for (0..n_ranks) |t| for (0..n_ranks) |s| {
                if (s == t) continue;
                for (numel, 0..) |_, i| try std.testing.expectEqualSlices(u8, sim.sends[s][t][i], sim.recvs[t][s][i]);
            };
        }
    }
}

test "at four ranks the opposite rank's tensor splits at numel / 2 and takes two hops" {
    const m = moving(4, &.{7}, 0, 1, 1);
    // to rank 1 (1 hop clockwise) all 7; to rank 2 (opposite) the first 3; to rank 3 the other way, so not here
    try std.testing.expectEqual(@as(usize, 2), m.n);
    try std.testing.expectEqual(@as(u8, 1), m.msgs[0].t);
    try std.testing.expectEqual(@as(usize, 7), m.msgs[0].hi);
    try std.testing.expectEqual(@as(u8, 2), m.msgs[1].t);
    try std.testing.expectEqual(@as(usize, 3), m.msgs[1].hi);
    // in the second hop only the opposite rank's half is still on its way
    try std.testing.expectEqual(@as(usize, 1), moving(4, &.{7}, 0, 1, 2).n);
    const back = moving(4, &.{7}, 0, -1, 1);
    try std.testing.expectEqual(@as(usize, 2), back.n); // rank 3 (1 hop) and the opposite rank's second half
    try std.testing.expectEqual(@as(usize, 3), back.msgs[0].lo);
}

test "link launch order: the link to the previous rank first, rank 0 starts with link 0" {
    try std.testing.expectEqual([2]i8{ 1, -1 }, linkOrder(4, 0));
    for (1..4) |me| try std.testing.expectEqual([2]i8{ -1, 1 }, linkOrder(4, me));
}
