//! A prompt chunk's GPU time by part (work/research/R2-prefill.md W-0): timing events recorded on the engine's
//! stream between the parts of a forward, read back after the chunk. Off unless `TF_FLASHNEXT_PROFILE=1`; when on,
//! `Forward.prof` points here and each `mark` closes the interval since the previous mark under that part's name.
//! The MTP head's absorb is counted apart (`mtp`). The events add a few microseconds a mark and change no bits.
//! Decode rounds (Engine.generateManyWith, run eagerly while profiling) mark their host gaps too: the interval
//! before a phase's first launch is the GPU waiting on the host (`gap_*`), and the MTP head's time by draft level.
const std = @import("std");
const cuda = @import("cuda");

pub const env = "TF_FLASHNEXT_PROFILE";

pub const Part = enum(u8) {
    start,
    stage,
    embed,
    ple,
    hc_writeback,
    hc_normed,
    hc_down,
    hc_act,
    hc_up,
    hc_mix,
    gdn_proj,
    gdn_front,
    gdn_chain,
    gdn_back,
    gdn_shift,
    out_proj,
    gather,
    attn_proj,
    attn_prep,
    qsa_pool,
    qsa_rows,
    qsa_scores,
    qsa_select,
    qsa_exchange,
    attention,
    attn_gate,
    router,
    topk_plan,
    experts_gate_up,
    shared_gate_up,
    shared_swiglu,
    experts_down,
    shared_down,
    slot_copy,
    moe_partial,
    moe_gather,
    reduce,
    mtp_in,
    finish,
    head,
    cands,
    commit,
    other,
    // decode rounds: host gaps before each phase's first launch, and the draws
    gap_verify,
    gap_draft,
    sample,
    draft_sample,
};

const names = @typeInfo(Part).@"enum".field_names;
const max_levels = 17;
const n_parts = names.len;

pub const Prof = struct {
    d: *const cuda.Driver,
    events: []cuda.Event,
    labels: []Part,
    in_mtp: []bool,
    levels: []u8,
    n: usize = 0,
    /// marks from here on count under the MTP head
    mtp: bool = false,
    /// and under this draft level (0: the absorb)
    level: u8 = 0,
    /// MTP head ms by draft level, since the last report
    by_level: [max_levels]f64 = @splat(0),
    /// ms by part: [0] the main layers, [1] the MTP head; and over every chunk reported
    sums: [2][n_parts]f64 = @splat(@splat(0)),
    total: [2][n_parts]f64 = @splat(@splat(0)),
    rows_total: usize = 0,
    chunks: usize = 0,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, capacity: usize) !*Prof {
        const p = try gpa.create(Prof);
        errdefer gpa.destroy(p);
        p.* = .{ .d = d, .events = try gpa.alloc(cuda.Event, capacity), .labels = try gpa.alloc(Part, capacity), .in_mtp = try gpa.alloc(bool, capacity), .levels = try gpa.alloc(u8, capacity) };
        for (p.events) |*e| e.* = try cuda.Event.init(d, true);
        return p;
    }

    pub fn deinit(p: *Prof, gpa: std.mem.Allocator) void {
        for (p.events) |*e| e.deinit();
        gpa.free(p.events);
        gpa.free(p.labels);
        gpa.free(p.in_mtp);
        gpa.free(p.levels);
        gpa.destroy(p);
    }

    /// The interval since the previous mark is `part`'s (the first mark of a chunk opens it: `.start`).
    pub fn mark(p: *Prof, s: cuda.Stream, part: Part) !void {
        if (p.n == p.events.len) try p.drain();
        try p.events[p.n].record(s);
        p.labels[p.n] = part;
        p.in_mtp[p.n] = p.mtp;
        p.levels[p.n] = p.level;
        p.n += 1;
    }

    /// Reads the recorded intervals into `sums` (waits for the last event); the last mark opens the next run.
    pub fn drain(p: *Prof) !void {
        if (p.n == 0) return;
        try p.events[p.n - 1].synchronize();
        for (1..p.n) |i| {
            const ms = try cuda.Event.elapsedMs(p.events[i - 1], p.events[i]);
            p.sums[@intFromBool(p.in_mtp[i])][@intFromEnum(p.labels[i])] += ms;
            if (p.in_mtp[i]) p.by_level[@min(p.levels[i], max_levels - 1)] += ms;
        }
        // keep the last event as the next interval's start
        std.mem.swap(cuda.Event, &p.events[0], &p.events[p.n - 1]);
        p.labels[0] = p.labels[p.n - 1];
        p.in_mtp[0] = p.in_mtp[p.n - 1];
        p.levels[0] = p.levels[p.n - 1];
        p.n = 1;
    }

    /// Decode rounds since the last report: ms a round by part (main model, then the MTP head and its levels), the
    /// whole of it against the rounds' wall time; then cleared.
    pub fn rounds(p: *Prof, rank: u32, label: []const u8, n_rounds: usize, wall_ms: f64) !void {
        try p.drain();
        p.n = 0;
        const per = 1.0 / @as(f64, @floatFromInt(@max(1, n_rounds)));
        var sum: [2]f64 = .{ 0, 0 };
        for (p.sums, 0..) |row, m| for (row) |v| {
            sum[m] += v;
        };
        std.debug.print("rounds rank {d} {s}: {d} rounds, wall {d:.2} ms a round, GPU timeline {d:.2} ms a round (main {d:.2}, MTP head {d:.2})\n", .{ rank, label, n_rounds, wall_ms * per, (sum[0] + sum[1]) * per, sum[0] * per, sum[1] * per });
        inline for (0..2) |m| {
            inline for (names, 0..) |name, i| {
                const v = p.sums[m][i];
                if (v * per >= 0.005) std.debug.print("rounds rank {d} {s}   {s}{s:<16} {d:>8.3} ms a round  {d:>5.1}%\n", .{ rank, label, if (m == 1) "mtp." else "", name, v * per, 100.0 * v / @max(sum[0] + sum[1], 1e-9) });
            }
        }
        for (p.by_level, 0..) |v, l| if (v > 0) std.debug.print("rounds rank {d} {s}   mtp level {d}{s} {d:>8.3} ms a round\n", .{ rank, label, l, if (l == 0) " (absorb)" else "", v * per });
        p.sums = @splat(@splat(0));
        p.by_level = @splat(0);
    }

    /// One chunk of `rows` rows done: its parts printed (rank `rank`), the totals kept; the next chunk starts anew.
    pub fn chunk(p: *Prof, rank: u32, rows: usize, wall_ms: f64) !void {
        try p.drain();
        p.n = 0;
        var sum: f64 = 0;
        for (p.sums) |row| for (row) |v| {
            sum += v;
        };
        std.debug.print("profile rank {d} chunk {d}: {d} rows, GPU {d:.2} ms (wall {d:.2} ms)", .{ rank, p.chunks, rows, sum, wall_ms });
        inline for (0..2) |m| {
            inline for (names, 0..) |name, i| {
                const v = p.sums[m][i];
                if (v >= 0.005) std.debug.print(" {s}{s}={d:.2}", .{ if (m == 1) "mtp." else "", name, v });
                p.total[m][i] += v;
            }
        }
        std.debug.print("\n", .{});
        p.sums = @splat(@splat(0));
        p.rows_total += rows;
        p.chunks += 1;
    }

    /// Every chunk since the last summary: ms by part, also per 2048 rows (both heads), then cleared.
    pub fn summary(p: *Prof, rank: u32, label: []const u8) void {
        if (p.chunks == 0) return;
        var sum: f64 = 0;
        for (p.total) |row| for (row) |v| {
            sum += v;
        };
        const per = 2048.0 / @as(f64, @floatFromInt(@max(1, p.rows_total)));
        std.debug.print("profile rank {d} {s}: {d} chunks, {d} rows, GPU {d:.1} ms = {d:.1} ms a 2048 rows ({d:.0} rows/s)\n", .{ rank, label, p.chunks, p.rows_total, sum, sum * per, @as(f64, @floatFromInt(p.rows_total)) * 1000.0 / @max(sum, 1e-9) });
        inline for (0..2) |m| {
            inline for (names, 0..) |name, i| {
                const v = p.total[m][i];
                if (v >= 0.05) std.debug.print("profile rank {d} {s}   {s}{s:<16} {d:>9.2} ms  {d:>8.2} ms/2048  {d:>5.1}%\n", .{ rank, label, if (m == 1) "mtp." else "", name, v, v * per, 100.0 * v / @max(sum, 1e-9) });
            }
        }
        p.total = @splat(@splat(0));
        p.rows_total = 0;
        p.chunks = 0;
    }
};

/// Whether the environment asks for the profile.
pub fn wanted(environ: ?[]const u8) bool {
    const v = environ orelse return false;
    return v.len > 0 and !std.mem.eql(u8, v, "0");
}
