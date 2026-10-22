//! Host loop over a long prompt: 2048-row chunks (1024 while others decode), the 64-token grid, pool growth, one segment.
//! glm5_next/cuda/decode.py prefill / prefill_chunk. SPLIT exchanges and overlap are P8b. Snapshots use gridPoint; this loop runs to the end.
//! Ash Hart and TensorFold contributors; CUDA launch patterns by jschmied (PR #443).
const std = @import("std");
const cuda = @import("cuda");
const pool = @import("cuda_pool.zig");
const seg = @import("cuda_segments.zig");
const grid = @import("cuda_prefill_grid.zig");
const prefill = @import("cuda_prefill.zig");

pub const gridPoint = grid.gridPoint;

/// Grow an extent in place so it covers `tokens`. Moving a cache is a later package.
pub fn ensureExtent(p: *pool.Pool, eid: u64, tokens: usize) !usize {
    const need = grid.extentTokens(tokens);
    if (need > (try p.get(eid)).size) try p.resize(eid, need);
    return (try p.get(eid)).size;
}

/// One segment for this chunk. `fill` always; `stage` only when the table has device memory.
pub fn noteChunk(t: *seg.SegRows, base: usize, pos: usize, rows: usize, ring_base: ?usize) !bool {
    _ = try t.fill(&.{.{ .base = base, .pos = pos, .rows = rows, .ring_base = ring_base }}, grid.sparse_from);
    if (t.dev != null) try t.stage();
    return t.anySparse();
}

fn stageIds(e: *const prefill.Engine, ids: []const i32) !void {
    if (ids.len > e.b.rows) return error.BadRows;
    const view: cuda.DeviceBuffer = .{ .d = e.tri.s.d, .ptr = e.b.ids, .len = e.b.rows * 4 };
    try view.uploadAsync(0, std.mem.sliceAsBytes(ids), e.tri.s.handle);
}

pub const Job = struct {
    tokens: []const i32,
    tap_ctx: ?*anyopaque = null,
    on_taps: ?*const fn (*anyopaque, usize) anyerror!void = null,
    begin: usize = 0,
    busy: bool = false,
    limit: usize = grid.chunk_rows,
    /// null or 0 with chunked KDA selects the 64-token grid.
    prompt_grid: ?usize = null,
    /// Sample flag: the head runs on every chunk and overwrites the one-row logits buffer.
    head: bool = false,
    extents: ?*pool.Pool = null,
    extent: ?u64 = null,
    base: usize = 0,
    slot: usize = 0,
    ring: ?usize = null,
    segments: ?*seg.SegRows = null,
};

/// Validate the resume point and the chunk width. Returns the rows one chunk may take.
pub fn checkJob(len: usize, begin: usize, limit: usize, requested: ?usize, busy: bool) !usize {
    if (len == 0 or begin >= len) return error.BadChunk;
    const g = try grid.promptGrid(limit, requested, true);
    try grid.checkResume(begin, g);
    const step = grid.chunkLimit(busy, limit);
    if (g != 0 and step % g != 0) return error.BadGrid;
    return step;
}

/// Commit every chunk through `len`. Parity flips inside the chunk. The conv window is shifted there, not again here.
/// Returns the sequence position after the last row.
pub fn run(e: *prefill.Engine, parity: *[34]u1, job: Job) !usize {
    const step = try checkJob(job.tokens.len, job.begin, job.limit, job.prompt_grid, job.busy);
    var pos = job.begin;
    while (pos < job.tokens.len) {
        const rows = @min(step, job.tokens.len - pos);
        const end = pos + rows;
        if (job.extents) |p| {
            const eid = job.extent orelse return error.NoExtent;
            const size = try ensureExtent(p, eid, end);
            if (size / 4 > e.s.pools) return error.ScoresScratch;
            e.cache.capacity = size;
        } else if (end > e.cache.capacity) return error.BadExtent;
        if (job.segments) |t| {
            const ring_base: ?usize = if (t.ring != null) job.slot * (job.ring orelse return error.BadRing) else null;
            _ = try noteChunk(t, job.base, pos, rows, ring_base);
        }
        try stageIds(e, job.tokens[pos..end]);
        try e.chunk(rows, pos, parity, job.head);
        if (job.on_taps) |hook| try hook(job.tap_ctx orelse return error.MissingTapContext, rows);
        pos = end;
    }
    return pos;
}

test "pool growth stays in place and a chunk is one segment" {
    var p = try pool.Pool.init(std.testing.allocator, 8192);
    defer p.deinit();
    const a = (try p.alloc(2048)).?;
    try std.testing.expectEqual(@as(usize, 4096), try ensureExtent(&p, a, 3000));
    const b = (try p.alloc(2048)).?;
    try std.testing.expectError(error.Overlap, ensureExtent(&p, a, 5000));
    try std.testing.expectError(error.BadExtent, ensureExtent(&p, b, 8192));
    var dense = try seg.SegRows.init(std.testing.allocator, 2048, 1, 2112);
    defer dense.deinit();
    try std.testing.expect(!try noteChunk(&dense, 0, 0, 2048, 0));
    var sparse = try seg.SegRows.init(std.testing.allocator, 8, 1, 2112);
    defer sparse.deinit();
    try std.testing.expect(try noteChunk(&sparse, 0, 2048, 8, 0));
    try std.testing.expectEqual(@as(i32, 1), sparse.plane(3)[3]);
    try std.testing.expectError(error.BadChunk, checkJob(0, 0, 2048, null, false));
    try std.testing.expectError(error.BadChunk, checkJob(5000, 5000, 2048, null, false));
    try std.testing.expectError(error.OffGrid, checkJob(5000, 100, 2048, null, false));
    try std.testing.expectEqual(@as(usize, 1024), try checkJob(5000, 2048, 2048, null, true));
    try std.testing.expectEqual(@as(?usize, 4992), gridPoint(5000, 0, 64));
}

test "compile the prompt loop without a GPU" {
    std.testing.refAllDecls(@This());
    if (std.c.getenv("TF_GLM_P8_COMPILE_PROBE") != null) {
        var engine: prefill.Engine = undefined;
        var parity: [34]u1 = @splat(0);
        _ = try run(&engine, &parity, .{ .tokens = &.{ 1, 2, 3 } });
    }
}
