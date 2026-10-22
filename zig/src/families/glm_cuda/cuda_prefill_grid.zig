//! Prompt chunk planner from glm5_next/cuda/{engine.py prompt_grid, sparse.py pool_count, forward.py dense_rows}.
//! UNSPLIT scheduling only. SPLIT exchanges are P8b. No device arithmetic.
const std = @import("std");

pub const chunk_rows: usize = 2048;
pub const busy_rows: usize = 1024;
pub const state_grid: usize = 64;
pub const att_rows: usize = 512;
pub const select_rows: usize = 512;
pub const sparse_from: usize = 2051;
pub const score_loop: usize = 32;
pub const topk_pools: usize = 512;
pub const tokens: usize = 2051;
pub const pool: usize = 4;
pub const split_rows: usize = 64;
pub const split_from: usize = 8192;
pub const prompt_rows: usize = 64;
pub const row_score_g: usize = 128;
pub const dense_scale: f32 = @bitCast(@as(u32, 0x3d800000));
pub const score_scale: f32 = @bitCast(@as(u32, 0x3db504f3));
pub const wscale_h32: f32 = @bitCast(@as(u32, 0x3e3504f3));
pub const index_eps: f32 = @bitCast(@as(u32, 0x358637bd));

pub const Span = struct { start: usize, rows: usize };

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// TF_GLM_PROMPT_GRID. Chunked KDA with an unset grid is 64. 0 is off only when chunked is false.
pub fn promptGrid(prefill_rows: usize, requested: ?usize, chunked: bool) !usize {
    // An explicit 0 with chunked KDA is the same as unset: the 32-row sub-chunks need a 64-token grid.
    var grid = requested orelse if (chunked) state_grid else 0;
    if (grid == 0) {
        if (!chunked) return 0;
        grid = state_grid;
    }
    if (grid != 16 and grid != 32 and grid != 64 and grid != 128) return error.BadGrid;
    if (chunked and grid % 32 != 0) return error.BadGrid;
    if (prefill_rows % grid != 0) return error.BadGrid;
    return grid;
}

/// engine.py grid_point. null means the kept state has nowhere to sit past `cut`.
pub fn gridPoint(n: usize, cut: usize, grid: usize) ?usize {
    if (grid == 0 or n % grid == 0) return n;
    const point = n / grid * grid;
    return if (point > cut) point else null;
}

pub fn checkResume(begin: usize, grid: usize) !void {
    if (grid != 0 and begin % grid != 0) return error.OffGrid;
}

pub fn chunkLimit(busy: bool, configured: usize) usize {
    return if (busy) @min(configured, busy_rows) else configured;
}

pub fn chunks(len: usize, begin: usize, limit: usize, out: []Span) ![]Span {
    if (limit == 0 or begin > len) return error.BadChunk;
    var n: usize = 0;
    var at = begin;
    while (at < len) {
        if (n == out.len) return error.TooManyChunks;
        const rows = @min(limit, len - at);
        out[n] = .{ .start = at, .rows = rows };
        n += 1;
        at += rows;
    }
    return out[0..n];
}

pub fn extentTokens(n: usize) usize {
    return std.mem.alignForward(usize, @max(n, 1), 2048);
}

/// Pools a prompt chunk scores: visible pools, rounded up to a multiple of 1024, at most np_max.
pub fn poolCount(pos: usize, rows: usize, np_max: usize) usize {
    const visible = (pos + rows) / pool + 1;
    return @min(np_max, cdiv(visible, 1024) * 1024);
}

/// Power-of-two bucket used by a tail shorter than PROMPT_ROWS, and by the scores allocation.
pub fn poolBucket(pos: usize, rows: usize, np_max: usize) usize {
    const visible = (pos + rows) / pool + 1;
    const pow = std.math.ceilPowerOfTwo(usize, @max(visible, 1)) catch unreachable;
    return @min(np_max, @max(@as(usize, 1024), pow));
}

pub fn chunksFor(pos: usize, rows: usize) usize {
    return cdiv(pos + rows, att_rows);
}

pub fn denseRows(rows: usize, host_pos: usize) usize {
    if (host_pos >= sparse_from) return 0;
    return @min(rows, sparse_from - host_pos);
}

pub fn headBlock(rows: usize) usize {
    return if (rows >= prompt_rows) 32 else 16;
}

pub fn routerBm(m: usize) usize {
    return if (m <= 16) 16 else if (m <= 32) 32 else if (m <= 64) 64 else 128;
}

pub const Onepass = struct { hbt: usize, warps: usize, stages: usize, prefetch: usize, split: usize };

pub fn onepassLaunch(heads: usize) Onepass {
    const pow = std.math.ceilPowerOfTwo(usize, @max(heads, 1)) catch unreachable;
    const hbt = @max(@as(usize, 16), @min(@as(usize, 32), pow));
    const small = heads <= 16;
    return .{ .hbt = hbt, .warps = if (small) 4 else 8, .stages = 1, .prefetch = 1, .split = 1 };
}

pub const SelectKind = enum { rows, split };
pub const SelectLaunch = struct { kind: SelectKind, block: usize, warps: usize, vis: bool };

/// `prompt_block` is a `_prompt_blocks` tile (VIS off). A short tail passes false (VIS on).
pub fn selectLaunch(n: usize, np: usize, prompt_block: bool) !SelectLaunch {
    if (np < topk_pools) return error.UnsupportedPools;
    if (n < split_rows and np >= split_from) return .{ .kind = .split, .block = 0, .warps = 4, .vis = !prompt_block };
    const wide = n < split_rows and np > 1024;
    return .{ .kind = .rows, .block = if (wide) 4096 else 1024, .warps = if (wide) 8 else 4, .vis = !prompt_block };
}

/// prompt_arith_code defaults: fp32 partials, MSA off, EXL3 prompt, sparse onepass, latent MMA, HC MMA, KDA chunked.
pub fn arithCode() [7]i32 {
    return .{ 0, 0, 1, 1, 1, 1, 1 };
}

test "prompt chunks, the 64-token grid and pool columns match the reference" {
    var buf: [8]Span = undefined;
    const full = try chunks(5000, 0, chunk_rows, &buf);
    try std.testing.expectEqual(@as(usize, 3), full.len);
    try std.testing.expectEqual(Span{ .start = 0, .rows = 2048 }, full[0]);
    try std.testing.expectEqual(Span{ .start = 2048, .rows = 2048 }, full[1]);
    try std.testing.expectEqual(Span{ .start = 4096, .rows = 904 }, full[2]);
    const busy = try chunks(5000, 0, chunkLimit(true, chunk_rows), &buf);
    try std.testing.expectEqual(@as(usize, 5), busy.len);
    try std.testing.expectEqual(Span{ .start = 4096, .rows = 904 }, busy[4]);
    try checkResume(2048, state_grid);
    try checkResume(0, state_grid);
    try std.testing.expectError(error.OffGrid, checkResume(100, state_grid));
    try std.testing.expectEqual(@as(?usize, 4992), gridPoint(5000, 0, state_grid));
    try std.testing.expectEqual(@as(?usize, null), gridPoint(5000, 4992, state_grid));
    try std.testing.expectEqual(@as(?usize, 5000), gridPoint(5000, 0, 0));
    try std.testing.expectEqual(state_grid, try promptGrid(chunk_rows, null, true));
    try std.testing.expectEqual(state_grid, try promptGrid(chunk_rows, 0, true));
    try std.testing.expectEqual(@as(usize, 0), try promptGrid(chunk_rows, 0, false));
    try std.testing.expectError(error.BadGrid, promptGrid(chunk_rows, 16, true));
    try std.testing.expectEqual(@as(usize, 1024), poolCount(0, 2048, 2048));
    try std.testing.expectEqual(@as(usize, 2048), poolCount(2048, 2048, 2048));
    try std.testing.expectEqual(@as(usize, 1024), poolBucket(0, 64, 4096));
    try std.testing.expectEqual(@as(usize, 4), chunksFor(0, 2048));
    try std.testing.expectEqual(@as(usize, 9), chunksFor(2050, 2048));
    try std.testing.expectEqual(@as(usize, 2048), denseRows(2048, 0));
    try std.testing.expectEqual(@as(usize, 3), denseRows(2048, 2048));
    try std.testing.expectEqual(@as(usize, 0), denseRows(64, sparse_from));
    try std.testing.expectEqual(@as(usize, 32), headBlock(64));
    try std.testing.expectEqual(@as(usize, 16), headBlock(63));
    try std.testing.expectEqual(@as(usize, 128), routerBm(2048));
    try std.testing.expectEqual(@as(usize, 16), routerBm(1));
    try std.testing.expectEqual(@as(usize, 32), onepassLaunch(32).hbt);
    try std.testing.expectEqual(@as(usize, 8), onepassLaunch(32).warps);
    try std.testing.expectEqual(@as(usize, 16), onepassLaunch(16).hbt);
    try std.testing.expectEqual(@as(usize, 4), onepassLaunch(16).warps);
    const wide = try selectLaunch(512, 2048, true);
    try std.testing.expectEqual(SelectKind.rows, wide.kind);
    try std.testing.expectEqual(false, wide.vis);
    try std.testing.expectEqual(@as(usize, 1024), wide.block);
    try std.testing.expectEqual(@as(usize, 4), wide.warps);
    const rem = try selectLaunch(8, 2048, true);
    try std.testing.expectEqual(@as(usize, 4096), rem.block);
    try std.testing.expectEqual(@as(usize, 8), rem.warps);
    try std.testing.expectEqual(SelectKind.split, (try selectLaunch(8, 8192, true)).kind);
    try std.testing.expectEqual(false, (try selectLaunch(8, 8192, true)).vis);
    try std.testing.expectEqual(true, (try selectLaunch(8, 8192, false)).vis);
    try std.testing.expectError(error.UnsupportedPools, selectLaunch(8, 400, true));
    try std.testing.expectEqual(@as(usize, 4096), extentTokens(3000));
    try std.testing.expectEqual(@as(i32, 1), arithCode()[6]);
    try std.testing.expectEqual(@as(u32, 0x3d800000), @as(u32, @bitCast(dense_scale)));
    try std.testing.expectEqual(@as(u32, 0x3db504f3), @as(u32, @bitCast(score_scale)));
    try std.testing.expectEqual(@as(u32, 0x3e3504f3), @as(u32, @bitCast(wscale_h32)));
    try std.testing.expectEqual(@as(u32, 0x358637bd), @as(u32, @bitCast(index_eps)));
    try std.testing.expectEqual(@as(u32, 0x358637bd), @as(u32, @bitCast(@as(f32, 1e-6))));
}
