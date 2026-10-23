//! A backend's KV-ownership counters for /health and /metrics (family-neutral): a pool cut into parts, one a rank
//! (dsv41's TF_DS_KV_SHARD, wojo's GLM 0118 `kv_shard` stats), what the parts' planner decided and moved, the sharded
//! attention's transfers, and every part's rows by what holds them. A backend that shards hands the host a `Source`;
//! the host reads it on its engine thread (the backend's own thread) and publishes the copy for status readers.
const std = @import("std");

pub const max_parts = 8;
/// Ranks whose own counters a Stats carries at most.
pub const max_ranks = 8;

/// One part's rows now (gauges).
pub const Part = struct {
    free: u64 = 0, // rows nothing holds
    live_rows: u64 = 0, // live streams' reservations
    kept_rows: u64 = 0, // kept prompt states' rows outside live extents
    twin_rows: u64 = 0, // borrowed twins
    hold_rows: u64 = 0, // reserved destinations (a restore landing, a live move in flight)
    live: u64 = 0, // live streams placed in it

    pub const help = .{
        .free = "Rows of this part nothing holds.",
        .live_rows = "Rows of this part live streams reserve.",
        .kept_rows = "Rows of this part kept prompt states hold outside live extents.",
        .twin_rows = "Rows of this part borrowed twins hold.",
        .hold_rows = "Rows of this part reserved for a restore or a live move in flight.",
        .live = "Live streams whose extent lies in this part.",
    };
};

/// One rank's own counters (rank 0's, and each follower's as its last stats reply carried them): the parts' planners
/// are replicas (equal by digest), but every rank fetches, attends and sends its own rows.
pub const Rank = struct {
    prompt_fetches: u64 = 0,
    prompt_rows: u64 = 0,
    prompt_picks: u64 = 0,
    prompt_bytes: u64 = 0,
    prompt_syncs: u64 = 0,
    fetch_rounds: u64 = 0,
    owner_rounds: u64 = 0,
    copies: u64 = 0,
    copy_bytes: u64 = 0,
    plans: u64 = 0, // op lists this rank applied
    checks: u64 = 0, // planner digest checks this rank took part in
    transfers_failed: u64 = 0, // this rank's own sides of copies, carries or moves that failed

    pub const help = .{
        .prompt_fetches = "Prompt-chunk layers on this rank that fetched the owner's compressed rows.",
        .prompt_rows = "Compressed rows this rank's prompt fetches moved.",
        .prompt_picks = "Rows this rank's prompt fetches would have moved unfolded.",
        .prompt_bytes = "Bytes this rank's prompt fetches moved.",
        .prompt_syncs = "Host reads this rank's prompt fetches waited on.",
        .fetch_rounds = "Decode rounds on this rank that fetched the owners' picked rows.",
        .owner_rounds = "Decode rounds this rank attended as an owner.",
        .copies = "Cross-part extent transfers this rank took part in.",
        .copy_bytes = "Bytes of cross-part extent transfers this rank took part in.",
        .plans = "Ownership op lists this rank applied.",
        .checks = "Planner digest checks this rank took part in.",
        .transfers_failed = "This rank's own sides of copies, carries or moves that failed.",
    };
};

pub const Stats = struct {
    parts: u64 = 0, // gauge: the parts (ranks)
    part_rows: u64 = 0, // gauge: positions a part
    carries: u64 = 0, // admissions that carried a kept state into another part
    carry_rescued: u64 = 0,
    carry_pressure: u64 = 0,
    carry_others: u64 = 0,
    carry_paused: u64 = 0,
    moves: u64 = 0, // live streams moved to another part
    moved_tokens: u64 = 0, // tokens carried or moved across parts
    moved_bytes: u64 = 0, // ...the owner-only bytes that took, by the layout's bytes a token
    transfers_failed: u64 = 0,
    staged: u64 = 0,
    staled: u64 = 0,
    twins_made: u64 = 0,
    twins_revoked: u64 = 0,
    refused_pinned: u64 = 0,
    refused_twin: u64 = 0,
    refused_loading: u64 = 0,
    refused_partial: u64 = 0,
    refused_no_target: u64 = 0,
    refused_small: u64 = 0,
    no_part: u64 = 0, // admissions no part could make room for
    plans: u64 = 0, // op lists sent to the followers
    checks: u64 = 0, // planner digests the ranks compared
    prompt_fetches: u64 = 0,
    prompt_rows: u64 = 0,
    prompt_picks: u64 = 0,
    prompt_bytes: u64 = 0,
    prompt_syncs: u64 = 0,
    prompt_blocked: u64 = 0, // prompt-chunk layers whose fetch outgrew its buffer (GLM: in row blocks)
    fetch_rounds: u64 = 0,
    owner_rounds: u64 = 0,
    owner_bytes: u64 = 0, // owner decode's query and head exchanges (GLM)
    copies: u64 = 0, // cross-part extent transfers the attention side ran
    copy_bytes: u64 = 0,
    ranks: u64 = 0, // gauge: the ranks whose own counters `rank` holds (0: none reported)
    part: [max_parts]Part = @splat(.{}),
    rank: [max_ranks]Rank = @splat(.{}),

    pub const gauges = [_][]const u8{ "parts", "part_rows", "ranks" };

    /// Each counter's /metrics help (a missing one fails the build).
    pub const help = .{
        .parts = "KV-ownership parts (one a rank).",
        .part_rows = "Pool positions a part holds.",
        .carries = "Admissions that carried a kept prompt state into another part.",
        .carry_rescued = "Carries made because the state's own part would drop it.",
        .carry_pressure = "Carries made because the state's own part would evict kept states.",
        .carry_others = "Carries made because resuming in place would stale another conversation's state.",
        .carry_paused = "Carries made instead of waiting for the live stream holding the state.",
        .moves = "Live streams moved whole to another part to admit a request.",
        .moved_tokens = "Tokens carried or moved across parts.",
        .moved_bytes = "Owner-only bytes the carries and moves sent.",
        .transfers_failed = "Copies, carries or moves that finished failed (the prompt ran from 0, or the stream stayed).",
        .staged = "Kept states whose pending spill writes were staged before their rows changed.",
        .staled = "Kept states given up for a placement.",
        .twins_made = "Filling prompts given borrowed twins.",
        .twins_revoked = "Borrowed twins taken back for an admission or a restore.",
        .refused_pinned = "Moves refused: a copy or carry reads the stream's rows.",
        .refused_twin = "Carries or moves refused: the stream holds twins.",
        .refused_loading = "Carries or moves refused: a transfer of the stream is in flight.",
        .refused_partial = "Carries or moves refused: a prompt chunk is in flight.",
        .refused_no_target = "Carries or moves refused: no part had the room.",
        .refused_small = "Carries refused: the state is under the minimum carry.",
        .no_part = "Admissions no part could make room for (the request waited).",
        .plans = "Ownership op lists rank 0 sent the followers.",
        .checks = "Planner digests the ranks compared (equal, or the pair ended).",
        .prompt_fetches = "Prompt-chunk layers that fetched the owner's compressed rows.",
        .prompt_rows = "Compressed rows the prompt fetches sent.",
        .prompt_picks = "Rows the prompt fetches would have sent unfolded (every pick of every row).",
        .prompt_bytes = "Bytes the prompt fetches sent.",
        .prompt_syncs = "Host reads the prompt fetches waited on.",
        .prompt_blocked = "Prompt-chunk layers whose fetched rows outgrew the fetch buffer and went in row blocks.",
        .fetch_rounds = "Decode rounds that fetched the owners' picked rows.",
        .owner_rounds = "Decode rounds the owners attended.",
        .owner_bytes = "Bytes owner decode's query and head exchanges sent.",
        .copies = "Cross-part extent transfers sent.",
        .copy_bytes = "Bytes the cross-part extent transfers sent.",
        .ranks = "Ranks whose own KV-ownership counters are reported (kv_shard_rank_*).",
    };
};

/// How the host reads a backend's stats (on its engine thread, the backend's own).
pub const Source = struct {
    ctx: *anyopaque,
    read: *const fn (ctx: *anyopaque, out: *Stats) void,
};

test "every counter has its help" {
    inline for (@typeInfo(Stats).@"struct".field_names) |n| {
        if (comptime std.mem.eql(u8, n, "part") or std.mem.eql(u8, n, "rank")) continue;
        try std.testing.expect(@field(Stats.help, n).len > 0);
    }
    inline for (@typeInfo(Part).@"struct".field_names) |n| try std.testing.expect(@field(Part.help, n).len > 0);
    inline for (@typeInfo(Rank).@"struct".field_names) |n| try std.testing.expect(@field(Rank.help, n).len > 0);
}
