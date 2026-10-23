//! Unsplit prompt chunk: glm5_next/cuda/forward.py layer_forward / kda_rows / dsa_rows, one stream.
//! Prompt matmuls, KDA chunk, and the Triton prompt kernels. SPLIT exchanges are P8b.
//! Ash Hart and TensorFold contributors; CUDA launch patterns by jschmied (PR #443).
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const mm = @import("cuda_matmuls.zig");
const ex = @import("cuda_experts.zig");
const state = @import("cuda_state.zig");
const kda = @import("cuda_kda.zig");
const hc = @import("cuda_hc.zig");
const mla = @import("cuda_mla.zig");
const Tri = @import("cuda_triton.zig").Tri;
const grid = @import("cuda_prefill_grid.zig");
const split = @import("cuda_prefill_split.zig");
const micro = @import("cuda_prompt_micro.zig");
const shard_prompt = @import("cuda_shard_prompt.zig");

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}
fn ci(n: usize) !i32 {
    if (n > std.math.maxInt(i32)) return error.BadRows;
    return @intCast(n);
}
fn cz(n: usize) i64 {
    return @intCast(n);
}

pub const Scratch = struct {
    arena: state.Arena,
    kproj: u64,
    ka: u64,
    kg: u64,
    k_out: u64,
    sums: u64,
    gu: u64,
    act: u64,
    sy: u64,
    router_part: u64,
    router_out: u64,
    pick: u64,
    wts: u64,
    ey: u64,
    dp: u64,
    qr: u64,
    lat: u64,
    q: u64,
    ikr: u64,
    igr: u64,
    qi: u64,
    qa: u64,
    ol: u64,
    o: u64,
    po: u64,
    pm: u64,
    pl: u64,
    scores: u64,
    pools_i: u64,
    tokens: u64,
    counts: u64,
    split: u64,
    origins: u64,
    dummy: u64,
    plan: ex.Plan,
    experts: ex.Scratch,
    /// the contiguous copy of a misaligned f_a/g_a block (TP=3), rows * 128 bf16
    stage: u64,
    rows: usize,
    pools: usize,
    nch: usize,
    part_rows: usize,
    select: usize,
    split_bytes: usize,
    /// TF_GLM_PROMPT_OWN_FRONT: every row's selected pools (int64 [rows, 512]), this rank's rows written, the peer's received.
    own_pools: u64,
    pub fn init(d: *const cuda.Driver, g: state.Geometry, rows: usize, cache_rows: usize) !Scratch {
        if (rows == 0 or rows > 16384 or cache_rows < 2048 or cache_rows % 2048 != 0) return error.BadScratch;
        const experts_n = state.Geometry.experts + 1;
        const pairs = rows * state.Geometry.slots;
        const items = ex.maxItems(pairs, experts_n, 16);
        const part_rows = @min(rows, grid.att_rows);
        const nch = cdiv(@min(cache_rows, 2560) + rows, grid.att_rows);
        const pools = cache_rows / 4;
        const select = @min(rows, grid.select_rows);
        const width = g.projWidth();
        const kh = rows * g.lin_heads * 128;
        const ah = rows * g.heads * 512;
        const split_rows = @min(rows, grid.split_rows - 1);
        const split_bytes: usize = if (pools >= grid.split_from) mla.splitScratch(@max(split_rows, 1), pools) * 4 else 4;
        const sizes = [_]usize{
            rows * width * 2,                      kh * 2,
            kh * 2,                                kh * 2,
            rows * 256 * 4,                        rows * 2 * g.dense_width * 2,
            rows * g.dense_width * 2,              rows * 4096 * 4,
            8 * rows * state.Geometry.experts * 4, rows * state.Geometry.experts * 4,
            pairs * 4,                        pairs * 4,
            pairs * 4096 * 4,                 rows * 2048 * 2,
            rows * 1536 * 2,                  rows * 512 * 2,
            rows * g.heads * 256 * 2,         rows * 160 * 2,
            rows * 128 * 4,                   rows * state.Geometry.index_heads * state.Geometry.index_dim * 2,
            ah * 2,                           ah * 2,
            rows * g.heads * 256 * 2,         nch * part_rows * g.heads * 512 * 4,
            nch * part_rows * g.heads * 4,    nch * part_rows * g.heads * 4,
            select * pools * 4,               select * grid.topk_pools * 8,
            rows * grid.tokens * 4,           rows * 4,
            split_bytes,                      40 * 16,
            4,                                pairs * 4,
            items * 3 * 4,                    experts_n * 4,
            pairs * 4,                        cdiv(pairs, 1024) * experts_n * 4,
            pairs * 4096 * 2,                 pairs * 4096 * 2,
            pairs * g.moe_width * 2,          4,
            4,                                (pairs / 16 + 1024) * 4,
            rows * grid.topk_pools * 8,
            rows * 128 * 2,
        };
        var a = try state.Arena.init(d, &sizes);
        errdefer a.deinit();
        var s: Scratch = undefined;
        const ptrs = [_]*u64{
            &s.kproj, &s.ka,      &s.kg,    &s.k_out, &s.sums, &s.gu,  &s.act,    &s.sy,      &s.router_part, &s.router_out,
            &s.pick,  &s.wts,     &s.ey,    &s.dp,    &s.qr,   &s.lat, &s.q,      &s.ikr,     &s.igr,         &s.qi,
            &s.qa,    &s.ol,      &s.o,     &s.po,    &s.pm,   &s.pl,  &s.scores, &s.pools_i, &s.tokens,      &s.counts,
            &s.split, &s.origins, &s.dummy,
        };
        for (ptrs, sizes[0..ptrs.len]) |p, n| p.* = a.take(n);
        s.plan = .{ .members = a.take(sizes[33]), .items = a.take(sizes[34]), .counts = a.take(sizes[35]), .rank = a.take(sizes[36]), .hist = a.take(sizes[37]), .tile = 64, .experts = experts_n };
        s.experts = .{ .xg = a.take(sizes[38]), .xu = a.take(sizes[39]), .xd = a.take(sizes[40]), .z = a.take(sizes[41]), .done = a.take(sizes[42]), .order = a.take(sizes[43]), .capacity = rows, .slots = state.Geometry.slots };
        s.own_pools = a.take(sizes[44]);
        s.stage = a.take(sizes[45]);
        s.arena = a;
        s.rows = rows;
        s.pools = pools;
        s.nch = nch;
        s.part_rows = part_rows;
        s.select = select;
        s.split_bytes = split_bytes;
        return s;
    }
    pub fn deinit(s: *Scratch) void {
        s.arena.deinit();
    }
};

/// A block's last projection into b.part, as row ranges (the split computes the peer's rows first, piece by piece).
pub const Out = union(enum) {
    /// matmul of x [rows, stride] (bf16) by w into b.part (fp32)
    mat: struct { x: u64, stride: usize, w: mm.Q4 },
    /// the MoE's combine of the routed slots and the shared row into b.part
    combine,
};

/// A block's row-independent start (hcsplit.py TF_GLM_PREFILL_OVERLAP=2): the projections, norms and query expansion
/// of a DSA block, a KDA block's input projections, the dense MLP's gate/up, the MoE's shared expert.
pub const Front = union(enum) {
    none,
    kda: fw.KdaW,
    dsa: fw.DsaW,
    dense: fw.MlpW,
    shared: fw.MlpW,
};

fn ffnFront(l: fw.Layer) Front {
    return switch (l.mlp) {
        .dense => |w| .{ .dense = w },
        .moe => |m| .{ .shared = m.shared },
    };
}
fn mixerFront(l: fw.Layer) Front {
    return switch (l.mixer) {
        .kda => |w| .{ .kda = w },
        .dsa => |w| .{ .dsa = w },
    };
}

pub const Engine = struct {
    g: state.Geometry,
    tri: Tri,
    entries: *const mm.Entries,
    weights: *const fw.Weights,
    b: state.Buffers,
    s: Scratch,
    cache: state.CacheView,
    conv: u64,
    rec: u64,
    ring: usize,
    pos_dev: u64,
    kda: ?kda.Ops = null,
    comm: ?*const fw.Comm = null,
    clusters: bool = false,
    indexed: bool = true,
    observer: ?struct { ctx: *anyopaque, call: *const fn (*anyopaque, usize, u64, usize) anyerror!void } = null,
    last_hidden: u64 = 0,
    /// TF_GLM_HC_SPLIT (two ranks): the glue split by rows, see cuda_prefill_split.zig. Off by default.
    split: split.Settings = .{},
    /// TF_GLM_PREFILL_OVERLAP: the exchanges' second stream (made by the engine when the setting is on).
    lane: ?split.Lane = null,
    /// The chunk running split (set by `chunk`), else null: block outputs then go through `out`.
    run: ?*const split.Run = null,
    /// TF_GLM_PROMPT_MICROBATCH: the settings, and (in a half's view) that its block outputs wait for the half's chain.
    micro_s: micro.Settings = .{},
    micro: bool = false,
    /// A half's view: the chunk row its first row is (DFlash2 tap rows are slot-strided, not shifted with the pointers).
    row0: usize = 0,
    /// Called right after the chunk's embedding with the hidden state and the chunk's first row and row count: a
    /// prompt with pictures writes the tower's rows over the placeholder rows (cuda_engine's media).
    inject: ?struct { ctx: *anyopaque, call: *const fn (*anyopaque, u64, usize, usize) anyerror!void } = null,
    /// TF_GLM_KV_SHARD: the history fetch of a stream whose latent rows are on one rank (cuda_shard_prompt.zig); null:
    /// off, every launch as before.
    shard: ?*shard_prompt.Fetch = null,

    pub fn putI32(e: Engine, addr: u64, value: i32) !void {
        try e.tri.s.d.check(e.tri.s.d.api.cuMemsetD32Async(addr, @as(c_uint, @bitCast(value)), 1, e.tri.s.handle), "cuMemsetD32Async");
    }
    pub fn mat(e: Engine, x: u64, stride: usize, w: mm.Q4, out: u64, rows: usize, fp32_out: bool) !void {
        try mm.matmul(e.entries, e.tri.s, x, stride, 0, w, out, 0, rows, fp32_out, e.clusters, true, .{ .ptr = e.s.stage, .bytes = e.s.rows * 128 * 2 });
    }
    pub fn norm(e: Engine, x: u64, stride: usize, w: u64, out: u64, d: usize, rows: usize) !void {
        try e.tri.rmsnorm(.{ rows, 1, 1 }, x, try ci(stride), w, out, try ci(d), e.b.xs, 1e-5, .{ .D = cz(d), .BLOCK = cz(@import("cuda_triton.zig").pow2(d)), .SUMS = 1 });
    }
    fn router(e: Engine, w: u64, out: u64, ne: usize, rows: usize) !void {
        const bm = grid.routerBm(rows);
        try e.tri.router_part(.{ cdiv(rows, bm), cdiv(ne, 32), 8 }, e.b.normed, w, e.s.router_part, try ci(rows), 4096, .{ .D = 4096, .NE = cz(ne), .BM = cz(bm), .BLOCK_E = 32, .BK = 64, .KS = 8 });
        try e.tri.router_sum(.{ cdiv(rows * ne, 1024), 1, 1 }, e.s.router_part, out, try ci(rows * ne), .{ .KS = 8, .BLOCK = 1024 });
    }
    fn hcPre(e: Engine, w: hc.Weights, rows: usize) !void {
        try e.hcPreRows(w, 0, rows);
    }
    /// hc_pre of rows [lo, lo + n): every row-indexed buffer at its row offset (rows are independent).
    fn hcPreRows(e: Engine, w: hc.Weights, lo: usize, n: usize) !void {
        const x = e.b.x + lo * e.g.wide() * 2;
        const part = e.b.hc_part + lo * 16 * 32 * 4;
        try e.tri.hc_partial_mma(.{ cdiv(n, 64), 16, 1 }, x, w.fn_w, part, try ci(n), .{ .WIDE = 16384, .NB = 16, .BM = 64, .BK = 64 });
        try e.tri.hc_finish(.{ n, 1, 1 }, x, part, w.base, w.scale, w.norm, e.b.normed + lo * 4096 * 2, e.b.xs + lo * 64 * 4, e.b.post + lo * 4 * 4, e.b.comb + lo * 16 * 4, 1e-5, 1e-6, .{ .D = 4096, .S = 4, .NB = 16, .ITERS = 20, .BLOCK = 4096 });
    }
    fn frontRange(e: Engine, f: Front, lo: usize, n: usize) !void {
        if (n == 0) return;
        const nx = e.b.normed + lo * 4096 * 2;
        switch (f) {
            .none => {},
            .kda => |w| {
                const width = e.g.projWidth();
                const p = e.s.kproj + lo * width * 2;
                try e.mat(nx, 4096, w.proj, p, n, false);
                try e.mat(p + w.fa_off * 2, width, w.fb, e.s.ka + lo * w.fb.n * 2, n, false);
                try e.mat(p + w.ga_off * 2, width, w.gb, e.s.kg + lo * w.gb.n * 2, n, false);
            },
            .dsa => |w| {
                const dp = e.s.dp + lo * 2048 * 2;
                try e.mat(nx, 4096, w.proj, dp, n, false);
                try e.normXs(dp, 2048, w.q_norm, e.s.qr + lo * 1536 * 2, 1536, n, e.b.xs + lo * 64 * 4);
                try e.normXs(dp + 1536 * 2, 2048, w.kv_norm, e.s.lat + lo * 512 * 2, 512, n, e.b.xs + lo * 64 * 4);
                try e.mat(e.s.qr + lo * 1536 * 2, 1536, w.qb, e.s.q + lo * w.qb.n * 2, n, false);
            },
            .dense => |w| try e.mlpFront(w, lo, n),
            .shared => |w| {
                try e.mlpFront(w, lo, n);
                try e.mat(e.s.act + lo * (w.gu.n / 2) * 2, w.gu.n / 2, w.down, e.s.sy + lo * 4096 * 4, n, true);
            },
        }
    }
    fn mlpFront(e: Engine, w: fw.MlpW, lo: usize, n: usize) !void {
        if (w.gu.n > 2 * e.g.dense_width) return error.BadScratch; // a half's view shifts gu / act by the dense widths
        try e.mat(e.b.normed + lo * 4096 * 2, 4096, w.gu, e.s.gu + lo * w.gu.n * 2, n, false);
        const width = w.gu.n / 2;
        const block = std.math.gcd(@as(usize, 512), width);
        try e.tri.swiglu(.{ n, width / block, 1 }, e.s.gu + lo * w.gu.n * 2, e.s.act + lo * width * 2, e.s.sums + lo * 1024, 10, .{ .W = cz(width), .BLOCK = cz(block) });
    }
    /// The block's front on the rows outside `done` (rows the glue's fronts already ran on).
    fn runFront(e: Engine, f: Front, done: ?[2]usize, rows: usize) !void {
        const d = done orelse return e.frontRange(f, 0, rows);
        if (d[0] > 0) try e.frontRange(f, 0, @min(d[0], rows));
        if (rows > d[1]) try e.frontRange(f, d[1], rows - d[1]);
    }
    fn normXs(e: Engine, x: u64, stride: usize, w: u64, out: u64, d: usize, rows: usize, xs: u64) !void {
        try e.tri.rmsnorm(.{ rows, 1, 1 }, x, try ci(stride), w, out, try ci(d), xs, 1e-5, .{ .D = cz(d), .BLOCK = cz(@import("cuda_triton.zig").pow2(d)), .SUMS = 1 });
    }
    /// Whether this chunk's DSA blocks take the own-row front (TF_GLM_PROMPT_OWN_FRONT, hcsplit-style: needs the split, the
    /// index cache, and at least 64 own rows on every rank so the prompt selection runs its block path).
    fn ownDsa(e: Engine, rows: usize) bool {
        const r = e.run orelse return false;
        return e.split.own_front and e.indexed and rows - r.h >= grid.prompt_rows;
    }
    fn routerAt(e: Engine, w: u64, out: u64, ne: usize, lo: usize, n: usize) !void {
        const bm = grid.routerBm(n);
        try e.tri.router_part(.{ cdiv(n, bm), cdiv(ne, 32), 8 }, e.b.normed + lo * 4096 * 2, w, e.s.router_part, try ci(n), 4096, .{ .D = 4096, .NE = cz(ne), .BM = cz(bm), .BLOCK_E = 32, .BK = 64, .KS = 8 });
        try e.tri.router_sum(.{ cdiv(n * ne, 1024), 1, 1 }, e.s.router_part, out, try ci(n * ne), .{ .KS = 8, .BLOCK = 1024 });
    }
    /// This rank's whole share (h rows from `mine`) of each (buffer, bytes a row) to the peer and the peer's in: the
    /// split's `share` (hcsplit.py), pad rows too, on this stream.
    fn shareRows(e: Engine, bufs: []const [2]u64) !void {
        const r = e.run.?;
        const comm = e.comm orelse return error.MissingCommunicator;
        var sends: [4][2]u64 = undefined;
        var recvs: [4][2]u64 = undefined;
        for (bufs, 0..) |b, i| {
            sends[i] = .{ b[0] + r.mine * b[1], r.h * b[1] };
            recvs[i] = .{ b[0] + r.theirs * b[1], r.h * b[1] };
        }
        if (e.lane) |l| {
            // all of this chunk's collectives ride the side stream, in one order on every rank
            try l.share_in.record(e.tri.s);
            try l.stream.wait(l.share_in);
            try comm.exchangeBytes(@intCast(1 - e.g.rank), sends[0..bufs.len], recvs[0..bufs.len], l.stream.handle);
            try l.share_out.record(l.stream);
            try e.tri.s.wait(l.share_out);
        } else try comm.exchangeBytes(@intCast(1 - e.g.rank), sends[0..bufs.len], recvs[0..bufs.len], e.tri.s.handle);
    }
    /// The own-row DSA front (own_front.py steps 1 and 2): q_a / kv_a, their norms and q_b on this rank's rows, the
    /// indexer's key, gate and (sparse rows) query projections, the index keys and the norms' rows shared; then q_b on
    /// the other ranks' rows, which have now arrived.
    fn dsaOwnFront(e: Engine, w: fw.DsaW, rows: usize, pos: usize, done: ?[2]usize) !void {
        const r = e.run.?;
        const lo = r.mine;
        const n = @min(r.h, rows - lo);
        const s = &e.s;
        if (done == null or done.?[0] != lo or done.?[1] != lo + n) try e.frontRange(.{ .dsa = w }, lo, n);
        try e.mat(e.b.normed + lo * 4096 * 2, 4096, w.kw, s.ikr + lo * w.kw.n * 2, n, false);
        try e.routerAt(w.gate, s.igr + lo * state.Geometry.index_dim * 4, state.Geometry.index_dim, lo, n);
        if (e.indexed and pos + rows - 1 >= grid.sparse_from) try e.mat(s.qr + lo * 1536 * 2, 1536, w.iqb, s.qi + lo * w.iqb.n * 2, n, false);
        try e.shareRows(&.{ .{ s.ikr, w.kw.n * 2 }, .{ s.igr, state.Geometry.index_dim * 4 } });
        try e.shareRows(&.{ .{ s.qr, 1536 * 2 }, .{ s.lat, 512 * 2 } });
        if (lo > 0) try e.mat(s.qr, 1536, w.qb, s.q, lo, false);
        if (rows > lo + n) try e.mat(s.qr + (lo + n) * 1536 * 2, 1536, w.qb, s.q + (lo + n) * w.qb.n * 2, rows - lo - n, false);
    }
    /// choose() without the tokens: the block's pools into e.s.pools_i.
    fn choosePools(e: Engine, n: usize, np: usize, pos: u64) !void {
        const launch = try grid.selectLaunch(n, np, true);
        if (launch.kind == .split) {
            try e.splitCall(n, np, pos, e.s.pools_i, e.s.tokens, e.s.counts, true);
            return;
        }
        try e.tri.select_rows(.{ n, 1, 1 }, e.s.scores, e.s.pools_i, try ci(np), e.s.scores, "*fp32", .{ .K = grid.topk_pools, .BLOCK = cz(launch.block), .VIS = 0 });
    }
    /// The own-row selection (own_front.py step 3): this rank's rows' pools over the shared index keys, shared in turn,
    /// then every row's tokens and counts from the pools (`_tokens`), block by block as the unsplit selection does.
    fn selectOwn(e: Engine, rows: usize, pos: usize, np_max: usize, pk: u64) !void {
        const r = e.run.?;
        const lo = r.mine;
        const n = @min(r.h, rows - lo);
        const np = grid.poolCount(pos, rows, np_max);
        const pool_row = grid.topk_pools * 8;
        var a: usize = 0;
        var k: usize = 0;
        while (a < n) : (k += 1) {
            const m = @min(grid.select_rows, n - a);
            const at = e.s.origins + (16 + k) * 16;
            try e.putI32(at, try ci(pos + lo + a));
            try e.scoreTile(e.s.qi + (lo + a) * 4096 * 2, e.s.ikr + (lo + a) * 160 * 2 + 128 * 2, pk, m, np, at);
            try e.choosePools(m, np, at);
            try e.memcpyD2D(e.s.own_pools + (lo + a) * pool_row, e.s.pools_i, m * pool_row);
            a += m;
        }
        try e.shareRows(&.{.{ e.s.own_pools, pool_row }});
        a = 0;
        var block: usize = 0;
        while (a < rows) : (block += 1) {
            const m = @min(grid.select_rows, rows - a);
            try e.memcpyD2D(e.s.pools_i, e.s.own_pools + a * pool_row, m * pool_row);
            const at: u64 = if (a == 0) e.pos_dev else e.s.origins + block * 16;
            try e.poolTokens(m, at, e.s.tokens + a * grid.tokens * 4, e.s.counts + a * 4);
            a += m;
        }
    }
    /// The chunk's rows [lo, ...) as a chunk of their own (hcsplit/microbatch.py `view`): every row-indexed buffer of the
    /// prompt buffers and scratch shifted by lo rows; the scratch one call uses at a time stays shared.
    fn rowView(e: Engine, lo: usize) Engine {
        var v = e;
        const g = e.g;
        v.row0 = lo;
        v.b.ids += lo * 4;
        v.b.x += lo * g.wide() * 2;
        v.b.normed += lo * 4096 * 2;
        v.b.hc_part += lo * 16 * 32 * 4;
        v.b.xs += lo * 64 * 4;
        v.b.post += lo * 4 * 4;
        v.b.comb += lo * 16 * 4;
        v.b.part += lo * 4096 * 4;
        v.b.hidden += lo * 4096 * 2;
        const s = &v.s;
        s.kproj += lo * g.projWidth() * 2;
        s.ka += lo * g.lin_heads * 128 * 2;
        s.kg += lo * g.lin_heads * 128 * 2;
        s.k_out += lo * g.lin_heads * 128 * 2;
        s.sums += lo * 256 * 4;
        s.gu += lo * 2 * g.dense_width * 2;
        s.act += lo * g.dense_width * 2;
        s.sy += lo * 4096 * 4;
        s.router_out += lo * state.Geometry.experts * 4;
        s.pick += lo * state.Geometry.slots * 4;
        s.wts += lo * state.Geometry.slots * 4;
        s.ey += lo * state.Geometry.slots * 4096 * 4;
        s.dp += lo * 2048 * 2;
        s.qr += lo * 1536 * 2;
        s.lat += lo * 512 * 2;
        s.q += lo * g.heads * 256 * 2;
        s.ikr += lo * 160 * 2;
        s.igr += lo * 128 * 4;
        s.qi += lo * 32 * 128 * 2;
        s.qa += lo * g.heads * 512 * 2;
        s.ol += lo * g.heads * 512 * 2;
        s.o += lo * g.heads * 256 * 2;
        s.tokens += lo * grid.tokens * 4;
        s.counts += lo * 4;
        s.own_pools += lo * grid.topk_pools * 8;
        return v;
    }
    /// Half `half`'s chain on the side stream (microbatch.py glue_side): its partial's exchange, then the glue on its own
    /// rows and the glued rows' swap, after the main stream's work so far; `microJoin` before reading what it wrote.
    fn microChain(e: Engine, half: usize, next: ?hc.Weights, tap: ?usize, final: bool, share: bool) !void {
        const l = e.lane.?;
        const r = e.run.?;
        try l.hf[half].record(e.tri.s);
        try l.stream.wait(l.hf[half]);
        var sv = e;
        sv.tri.s = l.stream;
        sv.lane = null;
        sv.micro = false;
        try sv.exchangePartial(r, 0, l.stream);
        _ = try sv.splitGlue(r, next, tap, final, .none, share);
        try l.hd[half].record(l.stream);
    }
    /// The layers of a chunk in two halves (microbatch.py layer_steps).
    fn microLayers(e: *Engine, rows: usize, pos: usize, parity: *[34]u1) !void {
        const l = e.lane.?;
        const h = micro.cut(rows, e.g.world);
        const rb = rows - h;
        const nb = split.padded(rb, 2);
        var plain = e.split;
        plain.overlap = 0; // each half: one piece, its chain queued whole
        var runs = [2]split.Run{ split.Run.init(h, e.g.rank, plain), split.Run.init(rb, e.g.rank, plain) };
        var va = e.rowView(0);
        va.run = &runs[0];
        va.micro = true;
        var vb = e.rowView(h);
        vb.run = &runs[1];
        vb.micro = true;
        vb.b.gathered += h * 4096 * 4;
        vb.s.origins += 8 * 16;
        vb.pos_dev = e.s.origins + 32 * 16;
        try e.putI32(vb.pos_dev, try ci(pos + h));
        for (0..cdiv(rb, grid.att_rows)) |i| try e.putI32(vb.s.origins + i * 16, try ci(pos + h + i * grid.att_rows));
        const nl = e.weights.layers.len;
        // the pad rows past the chunk (B's, to whole shares): zero stream and partial, never read
        if (h + nb > rows) {
            try e.memzero(e.b.x + rows * e.g.wide() * 2, (h + nb - rows) * e.g.wide() * 2);
            try e.memzero(e.b.part + rows * 4096 * 4, (h + nb - rows) * 4096 * 4);
        }
        try e.hcPre(e.weights.layers[0].attn, h + nb);
        var views = [2]*Engine{ &va, &vb };
        const poss = [2]usize{ pos, pos + h };
        const ns = [2]usize{ h, rb };
        var pending = [2]bool{ false, false };
        var ki: usize = 0;
        var ai: usize = 0;
        for (e.weights.layers, 0..) |layer, index| {
            for (0..2) |k| {
                if (pending[k]) try e.tri.s.wait(l.hd[k]);
                pending[k] = false;
                switch (layer.mixer) {
                    .kda => |w| {
                        if (ki >= 34) return error.BadRows;
                        try views[k].kdaLayer(w, ki, ns[k], poss[k], parity, null);
                    },
                    .dsa => |w| {
                        if (ai >= 11) return error.BadRows;
                        try views[k].dsaLayer(w, ai, ns[k], poss[k], null);
                    },
                }
                try views[k].microChain(k, layer.ffn, null, false, true);
                pending[k] = true;
            }
            switch (layer.mixer) {
                .kda => ki += 1,
                .dsa => ai += 1,
            }
            var tap: ?usize = null;
            for ([_]usize{ 5, 14, 24, 33, 42 }, 0..) |t, slot| if (index == t) {
                tap = slot;
            };
            for (0..2) |k| {
                if (pending[k]) try e.tri.s.wait(l.hd[k]);
                pending[k] = false;
                try views[k].ffn(layer, ns[k], null);
                const has_next = index + 1 < nl;
                const share = !(has_next and views[k].ownDsa(ns[k]) and e.weights.layers[index + 1].mixer == .dsa);
                try views[k].microChain(k, if (has_next) e.weights.layers[index + 1].attn else null, tap, !has_next, share);
                pending[k] = true;
            }
        }
        for (0..2) |k| if (pending[k]) try e.tri.s.wait(l.hd[k]);
    }
    fn memcpyD2D(e: Engine, dst: u64, src: u64, bytes: usize) !void {
        try e.tri.s.d.check(e.tri.s.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, e.tri.s.handle), "cuMemcpyDtoDAsync");
    }
    fn memzero(e: Engine, dst: u64, bytes: usize) !void {
        try e.tri.s.d.check(e.tri.s.d.api.cuMemsetD8Async(dst, 0, bytes, e.tri.s.handle), "cuMemsetD8Async");
    }
    fn fillRows(e: Engine, o: Out, lo: usize, n: usize) !void {
        if (n == 0) return;
        switch (o) {
            .mat => |m| try e.mat(m.x + lo * m.stride * 2, m.stride, m.w, e.b.part + lo * 4096 * 4, n, true),
            .combine => try e.tri.combine_sy(.{ n, 4, 1 }, e.s.ey + lo * state.Geometry.slots * 4096 * 4, e.s.sy + lo * 4096 * 4, e.s.wts + lo * state.Geometry.slots * 4, e.b.part + lo * 4096 * 4, .{ .D = 4096, .SLOTS = state.Geometry.slots, .BLOCK = 1024 }),
        }
    }
    /// piece j of this rank's partial of the peer's rows out, the peer's partial of this rank's rows in (the staging
    /// block of the piece: [2, n, D] fp32 in b.gathered, rank order), on `stream`.
    fn exchangePartial(e: Engine, r: *const split.Run, j: usize, stream: cuda.Stream) !void {
        const comm = e.comm orelse return error.MissingCommunicator;
        const piece = r.pieces[j];
        const peer: usize = 1 - e.g.rank;
        const slot = piece.n * 4096 * 4;
        const base = e.b.gathered + 2 * piece.off * 4096 * 4;
        try comm.exchangeBytes(@intCast(peer), &.{.{ e.b.part + (r.theirs + piece.off) * 4096 * 4, slot }}, &.{.{ base + peer * slot, slot }}, stream.handle);
    }
    /// A block's output rows in b.part, and (split) the partials' exchange (hcsplit.py partial): without the second
    /// stream one fill and one exchange on this stream; with it the peer's rows piece by piece, each piece's exchange
    /// on the side stream while the next piece (then this rank's own rows) is computed.
    fn blockOut(e: Engine, o: Out, rows: usize) !void {
        const r = e.run orelse return e.fillRows(o, 0, rows);
        if (e.micro) return e.fillRows(o, 0, rows); // the half's chain exchanges it on the side stream
        if (e.lane) |l| {
            for (0..r.np) |j| {
                const lo = r.theirs + r.pieces[j].off;
                const hi = @min(lo + r.pieces[j].n, rows);
                if (hi > lo) try e.fillRows(o, lo, hi - lo);
                try l.fill[j].record(e.tri.s);
                try l.stream.wait(l.fill[j]);
                try e.exchangePartial(r, j, l.stream);
                try l.part[j].record(l.stream);
            }
            const hi = @min(r.mine + r.h, rows);
            if (hi > r.mine) try e.fillRows(o, r.mine, hi - r.mine);
            return;
        }
        try e.fillRows(o, 0, rows);
        try e.exchangePartial(r, 0, e.tri.s);
    }
    /// The glue of a split site (hcsplit.py glue): own rows' hc_post (the two partials summed rank 0 first), the
    /// layer's DFlash2 tap and the final stream mean, the next hc_pre (`next`), and the bf16 rows swapped with the
    /// peer's; piece by piece with the swaps on the side stream when there is one, else once on this stream.
    fn splitGlue(e: Engine, r: *const split.Run, next: ?hc.Weights, tap: ?usize, final: bool, front: Front, share: bool) !?[2]usize {
        const comm = e.comm orelse return error.MissingCommunicator;
        const rank: usize = e.g.rank;
        const peer: usize = 1 - rank;
        const main = e.tri.s;
        const fronts = e.split.overlap == 2 and e.lane != null and front != .none;
        for (0..r.np) |j| {
            const pc = r.pieces[j];
            const lo = r.mine + pc.off;
            const slot = pc.n * 4096 * 4;
            const base = e.b.gathered + 2 * pc.off * 4096 * 4;
            if (e.lane) |l| try main.wait(l.part[j]);
            try e.memcpyD2D(base + rank * slot, e.b.part + lo * 4096 * 4, slot);
            const x = e.b.x + lo * e.g.wide() * 2;
            try e.tri.hc_post(.{ pc.n, 4, 1 }, x, x, base, e.b.post + lo * 4 * 4, e.b.comb + lo * 16 * 4, try ci(pc.n * 4096), .{ .D = 4096, .S = 4, .WORLD = 2, .BLOCK = 1024, .ROUND = 0 });
            var sends: [3][2]u64 = undefined;
            var recvs: [3][2]u64 = undefined;
            var n: usize = 0;
            const bytes = pc.n * 4096 * 2;
            const th = r.theirs + pc.off;
            if (tap) |t| {
                const tb = e.b.taps + (t * e.b.rows + e.row0) * 4096 * 2;
                try e.tri.stream_mean(.{ pc.n, 4, 1 }, x, tb + lo * 4096 * 2, .{ .D = 4096, .S = 4, .BLOCK = 1024 });
                sends[n] = .{ tb + lo * 4096 * 2, bytes };
                recvs[n] = .{ tb + th * 4096 * 2, bytes };
                n += 1;
            }
            if (final) {
                try e.tri.stream_mean(.{ pc.n, 4, 1 }, x, e.b.hidden + lo * 4096 * 2, .{ .D = 4096, .S = 4, .BLOCK = 1024 });
                sends[n] = .{ e.b.hidden + lo * 4096 * 2, bytes };
                recvs[n] = .{ e.b.hidden + th * 4096 * 2, bytes };
                n += 1;
            }
            if (next) |w| {
                try e.hcPreRows(w, lo, pc.n);
                if (share) {
                    sends[n] = .{ e.b.normed + lo * 4096 * 2, bytes };
                    recvs[n] = .{ e.b.normed + th * 4096 * 2, bytes };
                    n += 1;
                }
            }
            if (n > 0) {
                if (e.lane) |l| {
                    try l.glue[j].record(main);
                    try l.stream.wait(l.glue[j]);
                    try comm.exchangeBytes(@intCast(peer), sends[0..n], recvs[0..n], l.stream.handle);
                } else try comm.exchangeBytes(@intCast(peer), sends[0..n], recvs[0..n], main.handle);
            }
            // TF_GLM_PREFILL_OVERLAP=2: the next block's front on this piece's rows while its swap is on the wire
            if (fronts and lo < r.rows) try e.frontRange(front, lo, @min(pc.n, r.rows - lo));
        }
        if (e.lane) |l| {
            try l.done.record(l.stream);
            try main.wait(l.done);
        }
        return if (fronts) .{ r.mine, @min(r.mine + r.h, r.rows) } else null;
    }
    fn post(e: Engine, rows: usize) !void {
        const gathered = if (e.g.world == 1) e.b.part else e.b.gathered;
        if (e.g.world > 1) {
            const comm = e.comm orelse return error.MissingCommunicator;
            try comm.allGather(e.b.part, gathered, rows * 4096, .f32, e.tri.s.handle);
        }
        try hc.HC.post(e.tri, e.b, gathered, rows, e.g.world);
    }
    fn kdaLayer(e: Engine, w: fw.KdaW, li: usize, rows: usize, pos: usize, parity: *[34]u1, done: ?[2]usize) !void {
        const ops = e.kda orelse return error.MissingKdaPackage;
        const heads = e.g.lin_heads;
        const width = e.g.projWidth();
        const p = e.s.kproj;
        try e.runFront(.{ .kda = w }, done, rows);
        const cur = parity[li];
        const layer_bytes = heads * 128 * 128 * 4;
        const state_in = e.rec + (@as(usize, cur) * 34 + li) * layer_bytes;
        const state_out = e.rec + (@as(usize, 1 - cur) * 34 + li) * layer_bytes;
        try ops.chunk(heads, p, width, w.b_off, e.s.ka, heads * 128, e.s.kg, heads * 128, e.conv + li * 3 * e.g.convDim() * 2, w.conv, state_in, w.a_log, w.dt_bias, w.norm, 1e-5, -5, rows, kda.chunkOff(pos), e.s.k_out, state_out);
        parity[li] = 1 - cur;
        const channels = e.g.convDim();
        try e.tri.conv_shift(.{ 1, cdiv(channels, 1024), 1 }, e.conv + li * 3 * channels * 2, p, try ci(rows), try ci(3 * channels), try ci(rows * width), try ci(width), .{ .C = cz(channels), .TAPS = 3, .BLOCK = 1024 });
        try e.blockOut(.{ .mat = .{ .x = e.s.k_out, .stride = heads * 128, .w = w.o } }, rows);
    }
    fn denseBlock(e: Engine, qa: u64, lc: u64, ol: u64, n: usize, pos: u64, nch: usize, hb: usize) !void {
        if (nch > e.s.nch or n > e.s.part_rows) return error.DenseScratch;
        const h = e.g.heads;
        try e.tri.dense_chunks(.{ n, cdiv(h, hb), nch }, qa, lc, lc, pos, e.s.po, e.s.pm, e.s.pl, try ci(n), .{ .H = cz(h), .LW = 512, .CH = 512, .SCALE = grid.dense_scale, .HBT = cz(hb), .KTT = 32, .RS = 528, .FP8 = 1 });
        try e.tri.merge(.{ n, h, 1 }, e.s.po, e.s.pm, e.s.pl, ol, e.s.dummy, try ci(n), .{ .H = cz(h), .LW = 512, .NCH = cz(nch), .SPARSE = 0 });
    }
    fn poolTokens(e: Engine, n: usize, pos: u64, tok: u64, cnt: u64) !void {
        try e.tri.tokens(.{ n, 1, 1 }, e.s.pools_i, pos, tok, cnt, .{ .W = grid.tokens, .K = grid.topk_pools, .PL = grid.pool, .BLOCK = 1024 });
    }
    fn splitCall(e: Engine, n: usize, np: usize, pos: u64, out: u64, tok: u64, cnt: u64, pools_out: bool) !void {
        const bytes = mla.splitScratch(n, np) * 4;
        if (bytes > e.s.split_bytes) return error.SplitScratch;
        const cp = mla.splitChunks(np);
        const chs = mla.splitChunk(np);
        try e.tri.s.d.check(e.tri.s.d.api.cuMemsetD8Async(e.s.split, 0, n * 1024 * 4, e.tri.s.handle), "select_split zero totals");
        const hist = e.s.split + n * 1024 * 4;
        const gt = e.s.split + n * (1024 + cp * 256) * 4;
        const vis: i64 = @intFromBool(!pools_out);
        for (0..5) |step| {
            if (pools_out) {
                try e.tri.select_split_pools(.{ n, cp, 1 }, e.s.scores, try ci(np), try ci(np), e.s.dummy, e.s.dummy, e.s.split, hist, gt, out, e.s.dummy, e.s.dummy, .{ .W = grid.tokens, .K = grid.topk_pools, .PL = grid.pool, .CHS = cz(chs), .CP = cz(cp), .STEP = cz(step), .SEG = 0, .VIS = vis, .TOKS = vis, .BLK = 16 });
            } else {
                try e.tri.select_split(.{ n, cp, 1 }, e.s.scores, try ci(np), try ci(np), pos, e.s.dummy, e.s.split, hist, gt, e.s.dummy, tok, cnt, .{ .W = grid.tokens, .K = grid.topk_pools, .PL = grid.pool, .CHS = cz(chs), .CP = cz(cp), .STEP = cz(step), .SEG = 0, .VIS = vis, .TOKS = vis, .BLK = 16 });
            }
        }
    }
    fn choose(e: Engine, n: usize, np: usize, pos: u64, prompt_block: bool, tok: u64, cnt: u64) !void {
        const launch = try grid.selectLaunch(n, np, prompt_block);
        if (launch.kind == .split) {
            if (prompt_block) {
                try e.splitCall(n, np, pos, e.s.pools_i, e.s.tokens, e.s.counts, true);
                try e.poolTokens(n, pos, tok, cnt);
            } else try e.splitCall(n, np, pos, e.s.dummy, tok, cnt, false);
            return;
        }
        const vis = launch.vis;
        try e.tri.select_rows(.{ n, 1, 1 }, e.s.scores, e.s.pools_i, try ci(np), if (vis) pos else e.s.scores, if (vis) "*i32" else "*fp32", .{ .K = grid.topk_pools, .BLOCK = cz(launch.block), .VIS = @intFromBool(vis) });
        try e.poolTokens(n, pos, tok, cnt);
    }
    fn scoreTile(e: Engine, qi: u64, wts: u64, pk: u64, n: usize, np: usize, pos: u64) !void {
        if (n > e.s.select or np > e.s.pools) return error.ScoresScratch;
        const gy = cdiv(cdiv(np, 64), grid.score_loop);
        try e.tri.scores(.{ n, gy, 1 }, qi, wts, 160, pk, pk, e.s.scores, pos, try ci(n), try ci(np), grid.score_scale, grid.wscale_h32, .{ .H = 32, .HP = 32, .D = 128, .BP = 64, .L = grid.score_loop, .RS = 144, .FP8 = 1 });
    }
    fn selectRows(e: Engine, rows: usize, pos: usize, np_max: usize, pk: u64) !void {
        if (rows >= grid.prompt_rows) {
            const np = grid.poolCount(pos, rows, np_max);
            var a: usize = 0;
            var block: usize = 0;
            while (a < rows) {
                const n = @min(grid.select_rows, rows - a);
                const at: u64 = if (a == 0) e.pos_dev else e.s.origins + block * 16;
                try e.scoreTile(e.s.qi + a * 4096 * 2, e.s.ikr + a * 160 * 2 + 128 * 2, pk, n, np, at);
                try e.choose(n, np, at, true, e.s.tokens + a * grid.tokens * 4, e.s.counts + a * 4);
                a += n;
                block += 1;
            }
            return;
        }
        const bucket = grid.poolBucket(pos, rows, np_max);
        if (rows > e.s.select or bucket > e.s.pools) return error.ScoresScratch;
        // G (the programs a row) is speed only: program g scores pool blocks g, g+G, ... and stores each block's scores
        // on its own, with no reduction across programs, and programs past the last block return (sparse.py _scores_rows).
        // The captured kernel set has `_scores_rows` at G=128 in every argument class but only some smaller G (no G=88,
        // which a 20,995-token prompt with a 200-token budget needs), so launch at G=128: the same scores, no
        // MissingTritonVariant. Python launches min(128, cdiv(np_max, 64)) programs; the results do not depend on it.
        const g = grid.row_score_g;
        try e.tri.scores_rows(.{ rows, g, 1 }, e.s.qi, e.s.ikr + 128 * 2, 160, pk, pk, e.s.scores, try ci(bucket), e.pos_dev, e.pos_dev, e.pos_dev, try ci(rows), try ci(bucket), grid.score_scale, grid.wscale_h32, .{ .H = 32, .HP = 32, .D = 128, .BP = 64, .G = cz(g), .SEG = 0, .RS = 144, .FP8 = 1, .K = grid.topk_pools });
        try e.choose(rows, bucket, e.pos_dev, false, e.s.tokens, e.s.counts);
    }
    pub fn dsaLayer(e: *Engine, w: fw.DsaW, ai: usize, rows: usize, pos: usize, done: ?[2]usize) !void {
        // TF_GLM_KV_SHARD: the stream's latent rows on one rank: its owner sends the history rows the chunk reads and the
        // other rank attends over a copy of them (cuda_shard_prompt.zig); a prompt borrowing a twin runs as unsharded on
        // both ranks (no role); without either, a view with no latent rows here (the other rank's extent) has none to write
        const role: ?shard_prompt.Role = if (e.shard != null) try shard_prompt.roleOf(e.cache) else null;
        if (role == null and !e.cache.hasLatent()) return error.NoLatentRows;
        const h = e.g.heads;
        const s = &e.s;
        const own = e.ownDsa(rows);
        if (own) try e.dsaOwnFront(w, rows, pos, done) else try e.runFront(.{ .dsa = w }, done, rows);
        const lc = e.cache.latent[ai];
        // (the receiving rank writes the chunk's latents into its copy of the history, after the fetch)
        if (role == null or role.? == .send) try e.latWrite(lc, e.pos_dev, rows);
        const sparse_rows = e.indexed and pos + rows - 1 >= grid.sparse_from;
        if (!e.indexed and pos + rows > grid.sparse_from) return error.MissingIndex;
        if (e.indexed) {
            if (e.ring < rows + 3) return error.BadRing;
            const pk = e.cache.pooled[ai];
            if (!own) {
                try e.mat(e.b.normed, 4096, w.kw, s.ikr, rows, false);
                try e.router(w.gate, s.igr, state.Geometry.index_dim, rows);
            }
            try e.tri.index_write(.{ rows, 1, 1 }, s.ikr, 160, s.igr, w.ln_w, w.ln_b, e.cache.keys[ai], e.cache.gates[ai], e.pos_dev, try ci(e.ring), grid.index_eps, .{ .D = 128 });
            try e.tri.pool_keys(.{ rows / 4 + 2, 1, 1 }, e.cache.keys[ai], e.cache.gates[ai], w.ape, pk, pk, e.pos_dev, try ci(rows), try ci(e.ring), .{ .D = 128, .RS = 144, .FP8 = 1 });
        }
        try e.tri.absorb_mma(.{ h, cdiv(rows, 64), 4 }, s.q, w.wk, s.qa, try ci(rows), .{ .H = cz(h), .D = 256, .LW = 512, .BM = 64, .BN = 128, .BK = 64 });
        if (role) |r| {
            // TF_GLM_KV_SHARD (forward.py _sharded_rows): the selection as unsharded (replicated), then the fetch and
            // the attention over this rank's view or its copy
            if (sparse_rows) try e.selectTokens(w, ai, rows, pos, own);
            const f = e.shard.?;
            var dev: ShardDev = .{ .f = f, .a = .{ .e = e, .rows = rows, .pos = pos, .sparse_rows = sparse_rows } };
            try shard_prompt.attend(&dev, f.b.p, f.lim, .{ .role = r, .rows = rows, .pos = pos, .sparse = sparse_rows, .tokens = s.tokens, .lc = if (r == .send) lc else 0 }, &f.stats);
        } else {
            try e.denseRows(lc, rows, pos, sparse_rows);
            if (sparse_rows) {
                try e.selectTokens(w, ai, rows, pos, own);
                try e.onepass(lc, 0, rows);
            }
        }
        try e.tri.expand_mma(.{ h, cdiv(rows, 64), 2 }, s.ol, w.wv, s.o, try ci(rows), .{ .H = cz(h), .DV = 256, .LW = 512, .BM = 64, .BN = 128, .BK = 64 });
        try e.blockOut(.{ .mat = .{ .x = s.o, .stride = h * 256, .w = w.o } }, rows);
    }
    /// The chunk's latents (s.lat, bf16) into latent rows `lc` from the position POS names (`_lat_write`).
    pub fn latWrite(e: Engine, lc: u64, pos: u64, rows: usize) !void {
        try e.tri.lat_write(.{ rows, 1, 1 }, e.s.lat, 512, lc, lc, pos, .{ .LW = 512, .RS = 528, .FP8 = 1 });
    }
    /// The rows under the dense limit over latent rows `lc` (none past it): whole 512-row blocks, each from its origin.
    pub fn denseRows(e: Engine, lc: u64, rows: usize, pos: usize, sparse_rows: bool) !void {
        if (pos >= grid.sparse_from) return;
        const h = e.g.heads;
        const s = &e.s;
        const nch = grid.chunksFor(pos, rows);
        const dense = if (sparse_rows) grid.denseRows(rows, pos) else rows;
        const hb = grid.headBlock(rows);
        if (dense <= grid.att_rows) {
            try e.denseBlock(s.qa, lc, s.ol, dense, e.pos_dev, nch, hb);
        } else {
            var r0: usize = 0;
            var block: usize = 0;
            while (r0 < dense) {
                const n = @min(grid.att_rows, dense - r0);
                const at: u64 = if (r0 == 0) e.pos_dev else s.origins + block * 16;
                try e.denseBlock(s.qa + r0 * h * 512 * 2, lc, s.ol + r0 * h * 512 * 2, n, at, nch, hb);
                r0 += n;
                block += 1;
            }
        }
    }
    /// The chunk's tokens and counts (rows past the dense limit): the own-row selection or the unsplit one.
    fn selectTokens(e: *Engine, w: fw.DsaW, ai: usize, rows: usize, pos: usize, own: bool) !void {
        if (own) return e.selectOwn(rows, pos, e.cache.capacity / 4, e.cache.pooled[ai]);
        try e.mat(e.s.qr, 1536, w.iqb, e.s.qi, rows, false);
        try e.selectRows(rows, pos, e.cache.capacity / 4, e.cache.pooled[ai]);
    }
    /// Rows [lo, hi) of the chunk over their tokens in latent rows `lc` (`_sparse_onepass`: a row's bits depend on its
    /// queries, its tokens and the rows they name only; rows with count 0 are left alone).
    pub fn onepass(e: Engine, lc: u64, lo: usize, hi: usize) !void {
        const h = e.g.heads;
        const s = &e.s;
        const op = grid.onepassLaunch(h);
        try e.tri.sparse_onepass(.{ hi - lo, cdiv(h, op.hbt), op.split }, s.qa + lo * h * 512 * 2, lc, lc, s.tokens + lo * grid.tokens * 4, s.counts + lo * 4, s.ol + lo * h * 512 * 2, .{ .W = grid.tokens, .H = cz(h), .LW = 512, .SCALE = grid.dense_scale, .HBT = cz(op.hbt), .KTT = 32, .SPLIT = cz(op.split), .STAGES = cz(op.stages), .PREFETCH = cz(op.prefetch), .RS = 528, .FP8 = 1 });
    }
    fn mlp(e: Engine, w: fw.MlpW, rows: usize, out: u64) !void {
        try e.mlpFront(w, 0, rows);
        const width = w.gu.n / 2;
        if (out == e.b.part) try e.blockOut(.{ .mat = .{ .x = e.s.act, .stride = width, .w = w.down } }, rows) else try e.mat(e.s.act, width, w.down, out, rows, true);
    }
    pub fn ffn(e: Engine, layer: fw.Layer, rows: usize, done: ?[2]usize) !void {
        switch (layer.mlp) {
            .dense => |w| {
                try e.runFront(.{ .dense = w }, done, rows);
                try e.blockOut(.{ .mat = .{ .x = e.s.act, .stride = w.gu.n / 2, .w = w.down } }, rows);
            },
            .moe => |w| {
                // Plan bins are routed experts plus one (289). The weight stack stays 288; the prompt kernels take that count.
                try e.router(w.router, e.s.router_out, state.Geometry.experts, rows);
                try e.tri.topk(.{ rows, 1, 1 }, e.s.router_out, w.bias, e.s.pick, e.s.wts, 2.5, .{ .NE = state.Geometry.experts, .TOPK = state.Geometry.top_k, .SLOTS = state.Geometry.slots, .BLOCK = 512, .SLOTP = 16, .NORM = 1 });
                var plan = e.s.plan;
                plan.tile = 64;
                try ex.plan(e.entries, e.tri.s, e.s.pick, rows * state.Geometry.slots, plan);
                try e.runFront(.{ .shared = w.shared }, done, rows);
                try ex.routed(e.entries, e.tri.s, e.b.normed, 4096, e.s.pick, plan, w.experts, e.s.experts, e.s.ey, rows, 10, false);
                try e.blockOut(.combine, rows);
            },
        }
    }
    /// One unsplit chunk. `pos` is the first row's position. Parity flips per KDA layer. `pos_dev` ends at pos+rows.
    pub fn chunk(e: *Engine, rows: usize, pos: usize, parity: *[34]u1, head: bool) !void {
        if (rows == 0 or rows > e.b.rows or rows > e.s.rows) return error.BadRows;
        if (e.g.world > 1 and (e.comm == null or e.comm.?.world != e.g.world or e.comm.?.rank != e.g.rank)) return error.MissingCommunicator;
        const blocks = cdiv(rows, grid.att_rows);
        if (blocks > 32) return error.BadRows;
        const split_on = split.applies(e.split, e.g.world, rows, e.b.rows);
        var run = split.Run.init(rows, e.g.rank, e.split);
        e.run = if (split_on) &run else null;
        defer e.run = null;
        if (split_on and e.split.overlap != 0 and e.lane == null) return error.MissingSideStream;
        if (split_on) if (e.lane) |l| {
            try l.own.record(e.tri.s);
            try l.stream.wait(l.own);
        };
        try e.putI32(e.pos_dev, try ci(pos));
        for (0..blocks) |i| try e.putI32(e.s.origins + i * 16, try ci(pos + i * grid.att_rows));
        try e.tri.embed(e.b.ids, e.weights.embed, e.b.x, rows, 4);
        if (e.inject) |i| try i.call(i.ctx, e.b.x, pos, rows);
        var ki: usize = 0;
        var ai: usize = 0;
        const nl = e.weights.layers.len;
        const rp = split.padded(rows, 2);
        const halves = split_on and e.lane != null and micro.applies(e.micro_s, e.g.world, rows, true) and micro.cut(rows, 2) + split.padded(rows - micro.cut(rows, 2), 2) <= e.b.rows;
        if (split_on and !halves) {
            // the pad row of an odd chunk: zero stream and partial, never read
            if (rp > rows) {
                try e.memzero(e.b.x + rows * e.g.wide() * 2, (rp - rows) * e.g.wide() * 2);
                try e.memzero(e.b.part + rows * 4096 * 4, (rp - rows) * 4096 * 4);
            }
            try e.hcPre(e.weights.layers[0].attn, rp);
        }
        var done: ?[2]usize = null; // rows whose block front the glue already ran (TF_GLM_PREFILL_OVERLAP=2)
        if (halves) {
            e.run = null;
            try e.microLayers(rows, pos, parity);
        }
        for (e.weights.layers, 0..) |layer, index| {
            if (halves) break;
            if (!split_on) try e.hcPre(layer.attn, rows);
            switch (layer.mixer) {
                .kda => |w| {
                    if (ki >= 34) return error.BadRows;
                    try e.kdaLayer(w, ki, rows, pos, parity, done);
                    ki += 1;
                },
                .dsa => |w| {
                    if (ai >= 11) return error.BadRows;
                    try e.dsaLayer(w, ai, rows, pos, done);
                    ai += 1;
                },
            }
            if (split_on) done = try e.splitGlue(&run, layer.ffn, null, false, ffnFront(layer), true) else {
                try e.post(rows);
                try e.hcPre(layer.ffn, rows);
            }
            try e.ffn(layer, rows, done);
            done = null;
            var tap: ?usize = null;
            for ([_]usize{ 5, 14, 24, 33, 42 }, 0..) |t, slot| if (index == t) {
                tap = slot;
            };
            if (split_on) {
                done = try e.splitGlue(&run, if (index + 1 < nl) e.weights.layers[index + 1].attn else null, tap, index + 1 == nl, if (index + 1 < nl) mixerFront(e.weights.layers[index + 1]) else .none, !(index + 1 < nl and e.ownDsa(rows) and e.weights.layers[index + 1].mixer == .dsa));
            } else {
                try e.post(rows);
                if (tap) |slot| try e.tri.stream_mean(.{ rows, 4, 1 }, e.b.x, e.b.taps + slot * e.b.rows * 4096 * 2, .{ .D = 4096, .S = 4, .BLOCK = 1024 });
            }
            if (e.observer) |o| try o.call(o.ctx, index, e.b.x, rows * 16384 * 2);
        }
        if (!split_on) try e.tri.stream_mean(.{ rows, 4, 1 }, e.b.x, e.b.hidden, .{ .D = 4096, .S = 4, .BLOCK = 1024 });
        try e.norm(e.b.hidden, 4096, e.weights.norm, e.b.normed, 4096, rows);
        e.last_hidden = e.b.normed + (rows - 1) * 4096 * 2;
        if (head) {
            const cfg = mm.headConfig(1);
            try e.tri.fmm(.{ cdiv(1, cfg.bm), cdiv(e.g.vocab, cfg.bn), 1 }, e.last_hidden, e.weights.head, e.weights.head_scale, e.b.logits, e.b.logits, 1, 4096, .{ .N = cz(e.g.vocab), .K = 4096, .SK = 1, .BM = cz(cfg.bm), .BLOCK_N = cz(cfg.bn), .BK = 64, .F32 = 0, .SEQ = 0 });
        }
        try e.putI32(e.pos_dev, try ci(pos + rows));
    }
};

/// A sharded chunk's attention launches (cuda_shard_prompt.attend's device calls them): the unsharded chunk's, over
/// `att` (this rank's view or its copy of the history).
pub const ShardAttn = struct {
    e: *const Engine,
    rows: usize,
    pos: usize,
    sparse_rows: bool,
    /// The chunk's latents into `att` at their positions (`pos` null: the chunk's POS), or from the row a word names.
    pub fn latWrite(a: ShardAttn, att: u64, pos: ?u64) !void {
        try a.e.latWrite(att, pos orelse a.e.pos_dev, a.rows);
    }
    pub fn dense(a: ShardAttn, att: u64) !void {
        try a.e.denseRows(att, a.rows, a.pos, a.sparse_rows);
    }
    pub fn sparse(a: ShardAttn, att: u64, lo: usize, hi: usize) !void {
        try a.e.onepass(att, lo, hi);
    }
};

/// cuda_shard_prompt.attend's device on the GPU: glm_shard.cu's fetch kernels and the NCCL transfers on the chunk's
/// stream, held against the split's side stream (with microbatching the other half's chain runs its collectives there,
/// and two collectives of one communicator must never run at once: forward.py waits for that stream and it for the
/// fetch); the attention launches through `a`.
const ShardDev = struct {
    f: *shard_prompt.Fetch,
    a: ShardAttn,
    fn stream(dv: *const ShardDev) cuda.Stream {
        return dv.a.e.tri.s;
    }
    pub fn mark(dv: *ShardDev, tokens: u64, n: usize, m: u64, limit: usize) !void {
        try dv.f.k.mark(dv.stream(), tokens, n, m, limit);
    }
    pub fn scan(dv: *ShardDev, m: u64, n: usize, sums: u64) !void {
        try dv.f.k.scan(dv.stream(), m, n, sums);
    }
    /// The union's count (the scan's total): the host waits for the stream (prompt chunks are never captured).
    pub fn count(dv: *ShardDev, sums: u64) !usize {
        const s = dv.stream();
        try s.synchronize();
        var raw: [4]u8 = undefined;
        try s.d.check(s.d.api.cuMemcpyDtoH_v2(&raw, sums + shard_prompt.scan_segs * 4, 4), "shard fetch count");
        const n = std.mem.readInt(i32, &raw, .little);
        if (n < 0) return error.BadFetchCount;
        return @intCast(n);
    }
    pub fn compact(dv: *ShardDev, src: u64, cum: u64, limit: usize, dst: u64) !void {
        try dv.f.k.compact(dv.stream(), src, cum, limit, dst);
    }
    pub fn renumber(dv: *ShardDev, tokens: u64, n: usize, cum: u64, limit: usize) !void {
        try dv.f.k.renumber(dv.stream(), tokens, n, cum, limit);
    }
    pub fn hold(dv: *ShardDev) !void {
        const l = dv.a.e.lane orelse return;
        try l.share_out.record(l.stream);
        try dv.stream().wait(l.share_out);
    }
    pub fn release(dv: *ShardDev) !void {
        const l = dv.a.e.lane orelse return;
        try l.share_in.record(dv.stream());
        try l.stream.wait(l.share_in);
    }
    pub fn send(dv: *ShardDev, at: u64, n: usize) !void {
        try dv.f.comm.exchangeBytes(@intCast(dv.f.sh.peer()), &.{.{ at, n }}, &.{}, dv.stream().handle);
    }
    pub fn recv(dv: *ShardDev, at: u64, n: usize) !void {
        try dv.f.comm.exchangeBytes(@intCast(dv.f.sh.peer()), &.{}, &.{.{ at, n }}, dv.stream().handle);
    }
    pub fn latWrite(dv: *ShardDev, att: u64, pos: ?u64) !void {
        try dv.a.latWrite(att, pos);
    }
    pub fn dense(dv: *ShardDev, att: u64) !void {
        try dv.a.dense(att);
    }
    pub fn sparse(dv: *ShardDev, att: u64, lo: usize, hi: usize) !void {
        try dv.a.sparse(att, lo, hi);
    }
};

fn hook(ctx: *fw.PromptCtx, rows: usize, pos: usize, parity: *[34]u1, head: bool) anyerror!void {
    const e: *Engine = @ptrCast(@alignCast(ctx));
    try e.chunk(rows, pos, parity, head);
}

pub fn bindForwardHook() void {
    fw.prompt_enter = hook;
}

pub fn chunkHook(e: *Engine, rows: usize, pos: usize, parity: *[34]u1, head: bool) !void {
    bindForwardHook();
    try fw.enterPrompt(@ptrCast(e), rows, pos, parity, head);
}

/// Prompt-only Triton kernels. P1 captures these signatures; this package does not invent their arithmetic.
pub const p1_kernels = [_][]const u8{
    "_hc_partial_mma", "_lat_write",  "_absorb_mma",  "_expand_mma",  "_dense_chunks", "_index_write",
    "_pool_keys",      "_scores",     "_scores_rows", "_select_rows", "_select_split", "_tokens",
    "_sparse_onepass", "_conv_shift",
};

test "compile the unsplit prompt launch chain without a GPU" {
    std.testing.refAllDecls(Scratch);
    std.testing.refAllDecls(Engine);
    try std.testing.expectEqual(@as(usize, 14), p1_kernels.len);
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 1, 1, 1, 1, 1 }, &grid.arithCode());
    if (std.c.getenv("TF_GLM_P8_COMPILE_PROBE") != null) {
        var engine: Engine = undefined;
        var parity: [34]u1 = @splat(0);
        try engine.chunk(1, 0, &parity, false);
        var scratch = try Scratch.init(undefined, try state.Geometry.init(2, 0), 2048, 8192);
        scratch.deinit();
    }
}
