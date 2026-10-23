//! TF_GLM_KV_SHARD (patch 0118, two ranks) beside filling prompts and kept prompt states: prompt borrowing (multi.py
//! `_admitted`'s twins, `_untwin`, `_revoke_twins`) and a resume's latent rows (`_place_hit`, `_rescue`, `Arena.carry`,
//! shard.py `send_rows` / `recv_rows`). GLM Zig keeps a prompt state as a copy outside the pool (cuda_snapshot.zig), not
//! as an extent in it, so a state's latent rows are on one rank only: the owner of the part its stream lay in when it
//! was kept (its home); every other section (pooled keys, rings, recurrent and conv states, the drafter's rings) is on
//! both. A resume puts its new extent in the home part when that part has the room (nothing travels), else carries the
//! rows from the home owner's state to the new extent's owner (TF_GLM_KV_SHARD_CARRY, from `carry_min` tokens), else
//! starts from 0. A fresh prompt borrows free rows of the other part while it fills (TF_GLM_KV_SHARD_TWINS): both ranks
//! hold its latent rows and attend as unsharded, nothing is sent; the twin is held for the whole fill and given back
//! when the prompt is in, before its first decode round. Host side, and the carry's transfers generic over the device;
//! cuda_engine.zig applies them, on every rank alike (the same pool, the same calls).
const std = @import("std");
const builtin = @import("builtin");
const shard = @import("cuda_shard.zig");
const Pool = @import("cuda_pool.zig").Pool;
const G = @import("cuda_geometry.zig").Geometry;

/// A kept state under this many tokens is never carried (multi.py CARRY_MIN): recomputing it costs under a second of
/// prefill, which a carry (11 planes sent and received while no lane decodes) does not repay.
pub const carry_min: usize = 2048;

// -- prompt borrowing ---------------------------------------------------------------------------------------------------

/// Extent `eid`'s fill borrows free rows of the other part (Pool.borrow: placed alike on every rank, all or none), held
/// until the prompt is in -> the twin's base (null: that part has no room; the prompt then fills sharded). Two parts
/// only (a part a rank); on an error nothing stays borrowed or held.
pub fn borrow(pool: *Pool, eid: u64) !?usize {
    if (pool.parts() != 2) return error.KvShardNeedsTwoRanks;
    if (!try pool.borrow(eid)) return null;
    errdefer pool.giveBack(eid) catch unreachable;
    const x = try pool.extent(eid);
    const twin = pool.twinIn(eid, 1 - pool.partOf(x.base)) orelse return error.NoTwin;
    if (!mut(.borrow_unheld)) pool.hold(eid, true);
    return twin;
}

/// Whether a placement may take extent `eid`'s twins back now (multi.py `_revoke_twins`): it has some, and no fill
/// part way through its layers holds them.
pub fn revocable(pool: *const Pool, eid: u64) bool {
    return pool.hasTwins(eid) and !pool.holding(eid);
}

// -- a resume's latent rows -----------------------------------------------------------------------------------------------

/// Why a resume starts from 0 instead (Python's carry refusals that apply here): the carry is off, or the state is
/// under `carry_min`.
pub const Why = enum { off, small };

/// Where a resume's latent rows come from: the new extent already in the home part (`here`), moved there before any row
/// is written (`move`: its new base), the state's rows carried into the extent where it is (`carry`), or none: the
/// prompt from 0 (`recompute`).
pub const Resume = union(enum) { here, move: usize, carry, recompute: Why };

/// A prefill resuming a kept state of `at` tokens whose home is part `home` into fresh extent `eid` (newSeq's: no row
/// written yet). Never a carry when the home part has the room, whatever the other part holds (Python's rule: carry only
/// to save a recompute). Every rank answers alike: the same pool, the same state, the agreed switch.
pub fn resumeIn(pool: *const Pool, eid: u64, home: usize, at: usize, carry_on: bool) !Resume {
    if (home >= pool.parts()) return error.BadPart;
    const x = try pool.extent(eid);
    if (x.twin != null or pool.hasTwins(eid)) return error.Twinned;
    if (pool.partOf(x.base) == home and !mut(.never_here)) return .here;
    if (!mut(.no_move)) if (try pool.placeIn(x.size, eid, home)) |base| return .{ .move = base };
    if (!carry_on) return .{ .recompute = .off };
    if (at < carry_min and !mut(.carry_small)) return .{ .recompute = .small };
    return .carry;
}

/// This rank's part of a resume's latent rows: the home owner's state holds them, the new extent's owner needs them.
/// `copy`: both here (a restore puts them back); `send` / `recv`: a carry; `none`: no latent row here either way.
pub const Latent = enum { copy, none, send, recv };

pub fn latentRole(sh: shard.Shard, home: usize, part: usize) Latent {
    if (home == part) return if (sh.rank == home or mut(.copy_anywhere)) .copy else .none;
    if (sh.rank == home) return .send;
    if (sh.rank == part) return .recv;
    return .none;
}

/// A carry: the home owner's kept state's latent planes (`src[i]`, `n` rows each) to the new owner's (`dst[i]`), the
/// same pieces of at most `piece` rows on both ranks (each send meets its receive). The receiver lands each piece in
/// `stage` and copies it into place, as a restore copies (a plane may lie in the display span, which a transport must
/// not write across). The transfers held against the prompt's side stream (`hold` / `release`, as the history fetch:
/// two collectives of one communicator never run at once). `dev`: send, recv, copy, hold, release (the engine's GPU
/// side, the tests' wire). The bytes this rank sent or received.
pub fn carry(dev: anytype, role: Latent, src: [G.dsa_layers]u64, dst: [G.dsa_layers]u64, n: usize, stage: u64, piece: usize) !usize {
    if (role != .send and role != .recv) return error.NotACarry;
    if (piece == 0) return error.BadRows;
    const row = G.latent_row;
    const planes: usize = if (mut(.carry_ten_planes)) G.dsa_layers - 1 else G.dsa_layers;
    var moved: usize = 0;
    try dev.hold();
    for (0..planes) |i| {
        const cut = if (role == .send and mut(.send_whole)) n else piece;
        var at: usize = 0;
        while (at < n) : (at += cut) {
            const k = @min(cut, n - at);
            if (role == .send) {
                try dev.send(src[i] + at * row, k * row);
            } else if (mut(.recv_in_place)) {
                try dev.recv(dst[i] + at * row, k * row);
            } else {
                try dev.recv(stage, k * row);
                try dev.copy(dst[i] + at * row, stage, k * row);
            }
            moved += k * row;
        }
    }
    try dev.release();
    return moved;
}

// -- mutants: each rule above turned the wrong way, in test builds only (cuda_shard_keep_test.zig kills them) ----------

pub const Mutant = enum {
    none,
    borrow_unheld, // a fill's twin not held: a placement could take it back part way through the prompt
    never_here, // a resume in its home part treated as elsewhere (moved or carried for nothing)
    no_move, // a resume whose home part has the room carries instead of moving there
    carry_small, // a state under carry_min carried
    copy_anywhere, // the rank without the state's rows copies them back (garbage, or a NoLatentRows)
    carry_ten_planes, // a carry sends one plane short
    send_whole, // the sender sends each plane whole while the receiver cuts pieces
    recv_in_place, // the receiver's transport writes into the plane (the display span) instead of the stage
};
pub var mutant: Mutant = .none;
inline fn mut(m: Mutant) bool {
    return builtin.is_test and mutant == m;
}
