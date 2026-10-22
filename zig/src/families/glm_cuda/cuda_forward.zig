//! Decode layer loop: glm5_next/cuda/forward.py kda_segments, dsa_segments, moe_block, _compute.
//! Ash Hart and TensorFold contributors; CUDA launch patterns by jschmied (PR #443).
const std = @import("std");
const cuda = @import("cuda");
const mm = @import("cuda_matmuls.zig");
const ex = @import("cuda_experts.zig");
const state = @import("cuda_state.zig");
const seg = @import("cuda_segments.zig");
const Tri = @import("cuda_triton.zig").Tri;
const HC = @import("cuda_hc.zig").HC;
const hc = @import("cuda_hc.zig");
const kd = @import("cuda_kda_interface.zig");
const side_mod = @import("cuda_side.zig");
const l2pf = @import("cuda_l2pf.zig");
pub const Comm = @import("cuda_comm.zig").Comm;
pub const KdaW = struct { proj: mm.Q4, fb: mm.Q4, gb: mm.Q4, o: mm.Q4, fa_off: usize, ga_off: usize, b_off: usize, conv: u64, a_log: u64, dt_bias: u64, norm: u64 };
pub const DsaW = struct { proj: mm.Q4, qb: mm.Q4, o: mm.Q4, kw: mm.Q4, iqb: mm.Q4, q_norm: u64, kv_norm: u64, wk: u64, wv: u64, gate: u64, ln_w: u64, ln_b: u64, ape: u64 };
pub const MlpW = struct { gu: mm.Q4, down: mm.Q4 };
pub const MoeW = struct { router: u64, bias: u64, shared: MlpW, experts: ex.Experts };
pub const Layer = struct { attn: hc.Weights, ffn: hc.Weights, mixer: union(enum) { kda: KdaW, dsa: DsaW }, mlp: union(enum) { dense: MlpW, moe: MoeW } };
pub const MtpW = struct { enorm: u64, hnorm: u64, eh: mm.Q4, norm: u64, layer: Layer, head: mm.Q4 };
pub const Weights = struct { mtp: ?MtpW = null, embed: u64, norm: u64, head: u64, head_scale: u64, layers: [45]Layer };
/// Window-owned projections/replay inputs, distinct from slot state. SK scratch is sized by weight shape.
pub const Scratch = struct {
    arena: state.Arena,
    sk: u64,
    proj: u64,
    ka: u64,
    kg: u64,
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
    tokens: u64,
    counts: u64,
    split: u64,
    k_out: u64,
    k: u64,
    v: u64,
    gates: u64,
    beta: u64,
    plan: ex.Plan,
    experts: ex.Scratch,
    qr_sums: u64,
    /// the contiguous copy of a misaligned f_a/g_a block (TP=3: stride 8726 / 8341), rows * 128 bf16
    stage: u64,
    rows: usize,
    pools: usize,
    cp: usize,
    chs: usize,
    pub fn layout(g: state.Geometry, rows: usize, cache_rows: usize) [45]usize {
        const pools = cache_rows / 4;
        const mla = @import("cuda_mla.zig");
        const cp = mla.splitChunks(pools);
        const kh = rows * g.lin_heads * 128;
        const ah = rows * g.heads * 512;
        const pairs = rows * 9;
        const items = ex.maxItems(pairs, 289, 16);
        return .{
            8 * rows * @max(g.projWidth(), 2 * g.dense_width) * 4, 34 * rows * g.projWidth() * 2,
            kh * 2,                                                kh * 2,
            rows * 256 * 4,                                        rows * 2 * g.dense_width * 2,
            rows * g.dense_width * 2,                              rows * 4096 * 4,
            8 * rows * 288 * 4,                                    rows * 288 * 4,
            pairs * 4,                                             pairs * 4,
            pairs * 4096 * 4,                                      rows * 2048 * 2,
            rows * 1536 * 2,                                       rows * 512 * 2,
            rows * g.heads * 256 * 2,                              rows * 160 * 2,
            rows * 128 * 4,                                        rows * 4096 * 2,
            ah * 2,                                                ah * 2,
            rows * g.heads * 256 * 2,                              5 * ah * 4,
            5 * rows * g.heads * 4,                                5 * rows * g.heads * 4,
            rows * pools * 4,                                      rows * 2051 * 4,
            rows * 4,                                              rows * (1024 + cp * 257) * 4,
            34 * kh * 2,                                           34 * kh * 4,
            34 * kh * 2,                                           34 * kh * 4,
            34 * rows * g.lin_heads * 4,                           pairs * 4,
            items * 3 * 4,                                         289 * 4,
            pairs * 4096 * 2,                                      pairs * 4096 * 2,
            pairs * g.moe_width * 2,                               8 * pairs * @max(4096, g.moe_width) * 4,
            pairs * 4,                                             rows * 24 * 4,
            rows * 128 * 2,
        };
    }
    pub fn bytes(g: state.Geometry, rows: usize, cache_rows: usize) usize {
        return state.aligned(&layout(g, rows, cache_rows));
    }
    pub fn init(d: *const cuda.Driver, g: state.Geometry, rows: usize, cache_rows: usize) !Scratch {
        if (rows == 0 or rows > 64 or cache_rows < 2048 or cache_rows % 2048 != 0) return error.BadScratch;
        const pools = cache_rows / 4;
        const mla = @import("cuda_mla.zig");
        const chs = mla.splitChunk(pools);
        const cp = mla.splitChunks(pools);
        const sizes = layout(g, rows, cache_rows);
        var a = try state.Arena.init(d, &sizes);
        errdefer a.deinit();
        var s: Scratch = undefined;
        const ptrs = [_]*u64{ &s.sk, &s.proj, &s.ka, &s.kg, &s.sums, &s.gu, &s.act, &s.sy, &s.router_part, &s.router_out, &s.pick, &s.wts, &s.ey, &s.dp, &s.qr, &s.lat, &s.q, &s.ikr, &s.igr, &s.qi, &s.qa, &s.ol, &s.o, &s.po, &s.pm, &s.pl, &s.scores, &s.tokens, &s.counts, &s.split, &s.k_out, &s.k, &s.v, &s.gates, &s.beta };
        for (ptrs, sizes[0..ptrs.len]) |p, n| p.* = a.take(n);
        s.plan = .{ .members = a.take(sizes[35]), .items = a.take(sizes[36]), .counts = a.take(sizes[37]), .rank = 0, .hist = 0, .experts = 289 };
        s.experts = .{ .xg = a.take(sizes[38]), .xu = a.take(sizes[39]), .xd = a.take(sizes[40]), .z = a.take(sizes[41]), .done = a.take(sizes[42]), .capacity = rows };
        s.qr_sums = a.take(rows * 24 * 4);
        s.stage = a.take(rows * 128 * 2);
        s.arena = a;
        s.rows = rows;
        s.pools = pools;
        s.cp = cp;
        s.chs = chs;
        return s;
    }
    pub fn deinit(s: *Scratch) void {
        s.arena.deinit();
    }
};
fn ci(n: usize) i32 {
    return @intCast(n);
}
fn c(n: usize) i64 {
    return @intCast(n);
}
pub const Forward = struct {
    g: state.Geometry,
    tri: Tri,
    hc: HC,
    entries: *const mm.Entries,
    weights: *const Weights,
    b: state.Buffers,
    s: Scratch,
    slots: *const state.Slots,
    caches: *const state.Caches,
    tables: *const seg.SegRows,
    kseg: u64,
    comm: ?*const Comm = null,
    kda: kd.Interface = .{},
    clusters: bool = false,
    /// TF_GLM_SIDE=1: the MoE's shared expert on this second stream (decode and verify windows; null is the single stream).
    side: ?*const side_mod.Side = null,
    /// TF_GLM_L2PF: the next kernels' weights prefetched on a side stream at three sites a layer (l2pf.py); null: off.
    pf: ?*l2pf.Prefetch = null,
    /// Optional diagnostic observer runs outside graph capture; it synchronizes/copies at the layer boundary.
    observer: ?struct { ctx: *anyopaque, call: *const fn (*anyopaque, usize, u64, usize) anyerror!void } = null,
    pub fn mat(f: *const Forward, x: u64, stride: usize, xs: u64, w: mm.Q4, out: u64, rows: usize, fp32: bool) !void {
        try mm.matmul(f.entries, f.tri.s, x, stride, xs, w, out, f.s.sk, rows, fp32, f.clusters, false, .{ .ptr = f.s.stage, .bytes = f.s.rows * 128 * 2 });
    }
    pub fn sums(f: *const Forward, x: u64, stride: usize, k: usize, rows: usize) !u64 {
        try f.tri.group_sums(.{ rows, (k / 64 + 15) / 16, 1 }, x, f.s.sums, ci(stride), .{ .K = c(k), .GB = 16 });
        return f.s.sums;
    }
    fn router(f: *const Forward, w: u64, out: u64, ne: usize, rows: usize) !void {
        const bm: usize = if (rows <= 16) 16 else if (rows <= 32) 32 else 64;
        try f.tri.router_part(.{ (rows + bm - 1) / bm, (ne + 31) / 32, 8 }, f.b.normed, w, f.s.router_part, ci(rows), 4096, .{ .D = 4096, .NE = c(ne), .BM = c(bm), .BLOCK_E = 32, .BK = 64, .KS = 8 });
        try f.tri.router_sum(.{ (rows * ne + 1023) / 1024, 1, 1 }, f.s.router_part, out, ci(rows * ne), .{ .KS = 8, .BLOCK = 1024 });
    }
    pub fn norm(f: *const Forward, x: u64, stride: usize, w: u64, out: u64, d: usize, xs: u64, rows: usize) !void {
        try f.tri.rmsnorm(.{ rows, 1, 1 }, x, ci(stride), w, out, ci(d), xs, 1e-5, .{ .D = c(d), .BLOCK = c(@import("cuda_triton.zig").pow2(d)), .SUMS = 1 });
    }
    fn kdaMix(f: *const Forward, w: KdaW, li: usize, index: usize, rows: usize) !void {
        const h = f.g.lin_heads;
        const n = f.s.rows;
        const width = f.g.projWidth();
        const p = f.s.proj + li * n * width * 2;
        try f.mat(f.b.normed, 4096, f.b.xs, w.proj, p, rows, false);
        const fa = p + w.fa_off * 2;
        const ga = p + w.ga_off * 2;
        try f.mat(fa, width, try f.sums(fa, width, 128, rows), w.fb, f.s.ka, rows, false);
        try f.mat(ga, width, try f.sums(ga, width, 128, rows), w.gb, f.s.kg, rows, false);
        if (f.pf) |pre| try pre.site(f.tri.s, index, .o, rows);
        const off = li * n * h * 128;
        const out = f.s.k_out + off * 2;
        try f.kda.run(f.tri.s, .{ .seg = f.kseg, .nseg = f.tables.max_segs, .p = p, .p_stride = width, .b_off = w.b_off, .a = f.s.ka, .a_stride = h * 128, .g = f.s.kg, .g_stride = h * 128, .conv = f.slots.conv + li * 3 * f.g.convDim() * 2, .conv_slot = 34 * 3 * f.g.convDim(), .conv_w = w.conv, .rec = f.slots.rec + li * h * 128 * 128 * 4, .rec_slot = 2 * 34 * h * 128 * 128, .rec_parity = 34 * h * 128 * 128, .a_log = w.a_log, .dt_bias = w.dt_bias, .norm_w = w.norm, .rows = rows, .heads = h, .scratch = .{ .out = out, .k = f.s.k + off * 4, .v = f.s.v + off * 2, .g = f.s.gates + off * 4, .b = f.s.beta + li * n * h * 4, .q_tmp = f.b.q_tmp, .y_tmp = f.b.y_tmp } });
        try f.mat(out, h * 128, try f.sums(out, h * 128, h * 128, rows), w.o, f.b.part, rows, true);
    }
    pub fn dsaMix(f: *const Forward, w: DsaW, li: usize, index: usize, rows: usize) !void {
        try f.dsaMtp(w, li, index, rows, null);
    }
    /// A scalar origin selects Python mtp_compute's dense DSA arithmetic. Prefetch uses `index` only on the main model.
    pub fn dsaMtp(f: *const Forward, w: DsaW, li: usize, index: usize, rows: usize, origin: ?u64) !void {
        const t = f.tri;
        const s = f.s;
        const h = f.g.heads;
        const pos = try f.tables.ptr(0);
        const base = try f.tables.ptr(1);
        const spr = try f.tables.ptr(3);
        const ibase = try f.tables.ptr(4);
        const dense = origin != null and !f.tables.anySparse();
        const lc = f.caches.latent[li];
        const pk = f.caches.pooled[li];
        try f.mat(f.b.normed, 4096, f.b.xs, w.proj, s.dp, rows, false);
        try f.norm(s.dp, 2048, w.q_norm, s.qr, 1536, s.qr_sums, rows);
        // q_b consumes qr sums before the shared sums scratch is reused for latents.
        try f.mat(s.qr, 1536, s.qr_sums, w.qb, s.q, rows, false);
        try f.norm(s.dp + 1536 * 2, 2048, w.kv_norm, s.lat, 512, s.sums, rows);
        try t.seg_lat_write(.{ rows, 1, 1 }, s.lat, 512, lc, lc, pos, base, .{ .LW = 512, .RS = 528, .FP8 = 1 });
        try f.mat(f.b.normed, 4096, f.b.xs, w.kw, s.ikr, rows, false);
        try f.router(w.gate, s.igr, 128, rows);
        try t.seg_index_write(.{ rows, 1, 1 }, s.ikr, 160, s.igr, w.ln_w, w.ln_b, f.caches.keys[li], f.caches.gates[li], pos, ibase, ci(f.caches.ring), 1e-6, .{ .D = 128 }); // sparse.py:612: the indexer LayerNorm eps is 1e-6 (not the model eps)
        try t.seg_pool_keys(.{ rows, 1, 1 }, f.caches.keys[li], f.caches.gates[li], w.ape, pk, pk, pos, base, ibase, ci(f.caches.ring), .{ .D = 128, .RS = 144, .FP8 = 1 });
        if (!dense) {
            try f.mat(s.qr, 1536, s.qr_sums, w.iqb, s.qi, rows, false);
            const grid = @min(256, (s.pools + 63) / 64);
            try t.seg_scores(.{ f.tables.max_segs, grid, 1 }, s.qi, s.ikr + 128 * 2, 160, pk, pk, s.scores, ci(s.pools), pos, spr, try f.tables.ptr(5), try f.tables.ptr(6), try f.tables.ptr(7), try f.tables.ptr(8), 0.08838834764831845, 0.17677669529663687, .{ .H = 32, .HP = 32, .D = 128, .BP = 64, .G = c(grid), .RS = 144, .FP8 = 1 });
            try f.tri.s.d.check(f.tri.s.d.api.cuMemsetD8Async(s.split, 0, rows * 1024 * 4, f.tri.s.handle), "select_split zero totals");
            for (0..5) |step| try t.select_split(.{ rows, s.cp, 1 }, s.scores, ci(s.pools), ci(s.pools), pos, spr, s.split, s.split + rows * 1024 * 4, s.split + rows * (1024 + s.cp * 256) * 4, s.split, s.tokens, s.counts, .{ .W = 2051, .K = 512, .PL = 4, .CHS = c(s.chs), .CP = c(s.cp), .STEP = c(step), .SEG = 1, .VIS = 0, .TOKS = 0, .BLK = 16 });
        }
        try t.absorb_q(.{ h, 16, 1 }, s.q, w.wk, s.qa, ci(rows), .{ .H = c(h), .D = 256, .LW = 512, .BN = 32 });
        if (f.pf) |pre| try pre.site(f.tri.s, index, .o, rows);
        const hb: usize = if (rows >= 8) 32 else 16;
        const nch: usize = if (dense) (@as(usize, @intCast(f.tables.plane(0)[rows - 1])) + 1 + 511) / 512 else 5;
        if (dense) {
            try t.dense_chunks(.{ rows, (h + hb - 1) / hb, nch }, s.qa, lc, lc, origin.?, s.po, s.pm, s.pl, ci(rows), .{ .H = c(h), .LW = 512, .CH = 512, .SCALE = 0.0625, .HBT = c(hb), .KTT = 32, .RS = 528, .FP8 = 1 });
        } else try t.seg_chunks(.{ rows, (h + hb - 1) / hb, 5 }, s.qa, lc, lc, pos, base, spr, s.tokens, s.counts, s.po, s.pm, s.pl, ci(rows), .{ .W = 2051, .H = c(h), .LW = 512, .CH = 512, .SCALE = 0.0625, .HBT = c(hb), .KTT = 32, .RS = 528, .FP8 = 1 });
        try t.merge(.{ rows, h, 1 }, s.po, s.pm, s.pl, s.ol, s.counts, ci(rows), .{ .H = c(h), .LW = 512, .NCH = c(nch), .SPARSE = 0 });
        try t.expand_v(.{ h, 16, 1 }, s.ol, w.wv, s.o, ci(rows), .{ .H = c(h), .DV = 256, .LW = 512, .BN = 16 });
        try f.mat(s.o, h * 256, try f.sums(s.o, h * 256, h * 256, rows), w.o, f.b.part, rows, true);
    }
    fn mlp(f: *const Forward, w: MlpW, rows: usize, out: u64) !void {
        try f.mat(f.b.normed, 4096, f.b.xs, w.gu, f.s.gu, rows, false);
        const width = w.gu.n / 2;
        const block = std.math.gcd(@as(usize, 512), width);
        try f.tri.swiglu(.{ rows, width / block, 1 }, f.s.gu, f.s.act, f.s.sums, 10, .{ .W = c(width), .BLOCK = c(block) });
        try f.mat(f.s.act, width, f.s.sums, w.down, out, rows, true);
    }
    pub fn ffn(f: *const Forward, layer: Layer, rows: usize) !void {
        switch (layer.mlp) {
            .dense => |w| try f.mlp(w, rows, f.b.part),
            .moe => |w| {
                if (f.side) |side| {
                    // The shared expert on the side stream after everything queued so far (the normed rows), beside the
                    // router, the selection and the routed experts; joined before the combine reads f.s.sy.
                    try side.pair.fork(f.tri.s);
                    var on_side = f.*;
                    on_side.tri.s = side.pair.stream;
                    try (&on_side).mlp(w.shared, rows, f.s.sy);
                }
                try f.router(w.router, f.s.router_out, 288, rows);
                try f.tri.topk(.{ rows, 1, 1 }, f.s.router_out, w.bias, f.s.pick, f.s.wts, 2.5, .{ .NE = 288, .TOPK = 8, .SLOTS = 9, .BLOCK = 512, .SLOTP = 16, .NORM = 1 });
                try ex.plan(f.entries, f.tri.s, f.s.pick, rows * 9, f.s.plan);
                if (f.side == null) try f.mlp(w.shared, rows, f.s.sy);
                try ex.routed(f.entries, f.tri.s, f.b.normed, 4096, f.s.pick, f.s.plan, w.experts, f.s.experts, f.s.ey, rows, 10, false);
                if (f.side) |side| try side.pair.join(f.tri.s);
                try f.tri.combine_sy(.{ rows, 4, 1 }, f.s.ey, f.s.sy, f.s.wts, f.b.part, .{ .D = 4096, .SLOTS = 9, .BLOCK = 1024 });
            },
        }
    }
    fn post(f: *const Forward, rows: usize) !void {
        const gathered = if (f.g.world == 1) f.b.part else f.b.gathered;
        if (f.g.world > 1) {
            const comm = f.comm orelse return error.MissingCommunicator;
            try comm.allGather(f.b.part, gathered, rows * 4096, .f32, f.tri.s.handle);
        }
        try HC.post(f.tri, f.b, gathered, rows, f.g.world);
    }
    pub fn compute(f: *const Forward, rows: usize) !void {
        if (rows == 0 or rows > f.b.rows or rows > f.s.rows) return error.BadRows;
        if (f.g.world > 1 and (f.comm == null or f.comm.?.world != f.g.world or f.comm.?.rank != f.g.rank)) return error.MissingCommunicator;
        try f.tri.embed(f.b.ids, f.weights.embed, f.b.x, rows, 4);
        var ki: usize = 0;
        var ai: usize = 0;
        for (f.weights.layers, 0..) |layer, index| {
            try f.hc.pre(f.tri, f.b, layer.attn, rows);
            switch (layer.mixer) {
                .kda => |w| {
                    try f.kdaMix(w, ki, index, rows);
                    ki += 1;
                },
                .dsa => |w| {
                    try f.dsaMix(w, ai, index, rows);
                    ai += 1;
                },
            }
            if (f.pf) |pre| try pre.site(f.tri.s, index, .a, rows);
            try f.post(rows);
            try f.hc.pre(f.tri, f.b, layer.ffn, rows);
            try f.ffn(layer, rows);
            if (f.pf) |pre| try pre.site(f.tri.s, index, .f, rows);
            try f.post(rows);
            for ([_]usize{ 5, 14, 24, 33, 42 }, 0..) |tap, slot| if (index == tap) {
                try f.tri.stream_mean(.{ rows, 4, 1 }, f.b.x, f.b.taps + slot * f.b.rows * 4096 * 2, .{ .D = 4096, .S = 4, .BLOCK = 1024 });
            };
            if (f.observer) |o| try o.call(o.ctx, index, f.b.x, rows * 16384 * 2);
        }
        try f.tri.stream_mean(.{ rows, 4, 1 }, f.b.x, f.b.hidden, .{ .D = 4096, .S = 4, .BLOCK = 1024 });
        try f.norm(f.b.hidden, 4096, f.weights.norm, f.b.normed, 4096, f.b.xs, rows);
        const cfg = mm.headConfig(rows);
        try f.tri.fmm(.{ (rows + cfg.bm - 1) / cfg.bm, (f.g.vocab + cfg.bn - 1) / cfg.bn, 1 }, f.b.normed, f.weights.head, f.weights.head_scale, f.b.logits, f.b.logits, ci(rows), 4096, .{ .N = c(f.g.vocab), .K = 4096, .SK = 1, .BM = c(cfg.bm), .BLOCK_N = c(cfg.bn), .BK = 64, .F32 = 0, .SEQ = 0 });
        if (f.pf) |pre| try pre.join(f.tri.s);
        ex.logFlush();
    }
};
/// P8 binds this. Decode `compute` does not call it, so the two packages do not import each other.
pub const PromptCtx = opaque {};
pub var prompt_enter: ?*const fn (*PromptCtx, usize, usize, *[34]u1, bool) anyerror!void = null;
pub fn enterPrompt(ctx: *PromptCtx, rows: usize, pos: usize, parity: *[34]u1, head: bool) !void {
    const f = prompt_enter orelse return error.PrefillNotInstalled;
    try f(ctx, rows, pos, parity, head);
}
test "compile the entire decode launch chain without a GPU" {
    std.testing.refAllDecls(Forward);
    if (std.c.getenv("TF_GLM_P6_COMPILE_PROBE") != null) {
        const f: Forward = undefined;
        try f.compute(1);
        var scratch = try Scratch.init(undefined, try state.Geometry.init(2, 0), 32, 4096);
        scratch.deinit();
    }
}
test "an unbound prompt hook reports that prefill is not installed" {
    var parity: [34]u1 = @splat(0);
    try std.testing.expectError(error.PrefillNotInstalled, enterPrompt(undefined, 1, 0, &parity, false));
}
