//! Device buffers of Flash Next on CUDA (Python qwen4_exp/cuda/state.py): a window's or prompt chunk's scratch
//! (`Buffers`, static so a step can be graph-captured) and one sequence's committed caches (`State`: the GDN conv and
//! recurrent state, the attention layers' KV, indexer keys and pooled blocks, the PLE tail, the MTP head's own
//! attention cache). Sizes follow Python's shapes field by field, per rank: `Geometry` holds the rank's share
//! (heads, value heads, expert width, head rows) as the weight loader's `(rank, world)` split gives it.
const std = @import("std");
const cuda = @import("cuda");

pub const max_rows = 16; // a verify window's rows (MAX_DEPTH 15 drafts + the pending token)
pub const prefill_rows = 2048; // rows of a prompt chunk (Python PREFILL_ROWS)
pub const att_rows = 256; // prompt attention runs in blocks of this many rows (ATT_ROWS)
pub const ends = 16; // prompts one prompt pass can end (ENDS): rows that get the head
pub const cand = 32; // TP: candidates a rank gathers per row (CAND)
pub const chunk_keys = 512; // keys an attention chunk holds, at absolute positions (attention.CHUNK)
pub const gdn_dk = 128;
pub const gdn_dv = 128;
pub const grow_step = 8192; // caches grow this many rows at a time (State.ensure)

/// One rank's shapes. Per-rank fields hold the rank's share at `world` 2 (Python's cfg replaced per rank).
pub const Geometry = struct {
    hidden: usize, // 2560
    streams: usize, // hyper-connection streams, 4
    low: usize, // hyper-connection low rank, 320
    heads: usize, // attention query heads of this rank (24 / world)
    kv_heads: usize, // (2 / world)
    head_dim: usize, // 256
    index_heads: usize, // indexer heads, replicated (4)
    index_dim: usize, // 128
    index_budget: usize, // 2048
    index_ratio: usize, // 4
    nk: usize, // GDN key heads of this rank (16 / world)
    nv: usize, // GDN value heads of this rank (48 / world)
    conv_kernel: usize, // 4
    experts: usize, // 512
    top_k: usize, // 10
    moe_width: usize, // routed and shared expert width of this rank (640 / world)
    ple_dim: usize, // 2560
    heads_per_ngram: usize, // 8
    ngram_size: usize, // 3
    ple_kernel: usize, // 4
    head_n: usize, // the lm_head rows this rank holds (vocab / world)
    linear_layers: usize, // 36
    attention_layers: usize, // 12
    mtp: bool,
    world: usize,
    /// the shared expert's width on this rank (0: moe_width; INT4-AutoRound's healed shared expert is 1280 wide)
    shared_width: usize = 0,
    /// INT4-AutoRound: block-FP8 mixers and shared expert, GPTQ int4 experts and head (cuda_forward's scratch for them)
    int4ar: bool = false,
    /// the MTP layer's routed experts a row (0: top_k); drafts only (TF_FLASHNEXT_MTP_TOPK: INT4-AutoRound's MTP layer
    /// was trained at top-10 while its main layers route 5)
    mtp_top_k: usize = 0,

    /// The routed experts a row of the main layers or of the MTP layer.
    pub fn topKFor(g: Geometry, mtp: bool) usize {
        return if (mtp and g.mtp_top_k != 0) g.mtp_top_k else g.top_k;
    }

    pub fn sharedWidth(g: Geometry) usize {
        return if (g.shared_width == 0) g.moe_width else g.shared_width;
    }

    pub fn wide(g: Geometry) usize {
        return g.streams * g.hidden;
    }
    pub fn slots(g: Geometry) usize {
        return @max(g.top_k, g.mtp_top_k) + 1; // slot top_k is the shared expert (buffers: the larger of the two)
    }
    /// GDN conv channels and projection row width (gdn.widths).
    pub fn gdnConv(g: Geometry) usize {
        return 2 * g.nk * gdn_dk + g.nv * gdn_dv;
    }
    pub fn gdnWidth(g: Geometry) usize {
        return g.gdnConv() + g.nv * gdn_dv + 2 * g.nv;
    }
    /// The attention projection's row width: q with its gate, k, v, and the indexer's q heads and key.
    pub fn attnWidth(g: Geometry) usize {
        return g.heads * 2 * g.head_dim + 2 * g.kv_heads * g.head_dim + (g.index_heads + 1) * g.index_dim;
    }
    /// Attention chunks a row reads: dense below the budget, its selected blocks and tail past it (AttnScratch.nch).
    pub fn nch(g: Geometry, capacity: usize) usize {
        return std.math.divCeil(usize, @min(capacity, g.index_budget + g.index_ratio - 1), chunk_keys) catch unreachable;
    }
    pub fn qsa(g: Geometry, capacity: usize) bool {
        return capacity > g.index_budget;
    }
    pub fn blocks(g: Geometry, capacity: usize) usize {
        return std.math.divCeil(usize, capacity, g.index_ratio) catch unreachable;
    }
    /// n-gram rows a window looks up: two n-gram lengths' heads (16), of which a rank gathers its half at TP=2.
    pub fn pleHeads(g: Geometry) usize {
        return (g.ngram_size - 1) * g.heads_per_ngram;
    }
    pub fn pleHeadDim(g: Geometry) usize {
        return g.ple_dim / g.pleHeads();
    }
    pub fn layers(g: Geometry) usize {
        return g.linear_layers + g.attention_layers;
    }
};

/// Sub-allocations of one device allocation, 256-byte aligned like the driver's own.
pub const Arena = struct {
    buf: cuda.DeviceBuffer,
    used: usize = 0,

    pub fn take(a: *Arena, bytes: usize) u64 {
        const at = std.mem.alignForward(usize, a.used, 256);
        a.used = at + bytes;
        std.debug.assert(a.used <= a.buf.len);
        return a.buf.ptr + at;
    }
};

fn aligned(sizes: []const usize) usize {
    var total: usize = 0;
    for (sizes) |n| total += std.mem.alignForward(usize, n, 256);
    return total;
}

/// The expert plan's scratch for `pairs` (row, slot) pairs (cuda/experts.py Plan).
fn planBytes(pairs: usize, experts: usize) [5]usize {
    const wide = pairs > 1024;
    const items = @min(pairs, experts) + pairs / 16;
    return .{ pairs * 4, items * 3 * 4, 2 * 4, (if (wide) pairs else 1) * 4, (if (wide) (pairs + 1023) / 1024 * experts else 1) * 4 };
}

/// A window's (or, with `prefill`, a prompt chunk's) scratch: every field Python's Buffers allocates, by name.
pub const Buffers = struct {
    arena: Arena,
    rows: usize,
    prefill: bool,
    // inputs and the residual streams
    ids: u64,
    last: u64,
    h: u64,
    pss: u64,
    normed: u64,
    xs_normed: u64,
    dn: u64,
    dn_mix: u64,
    act: u64,
    xs_act: u64,
    inj_a: u64,
    inj_m: u64,
    up: u64,
    mixed: u64,
    xs_mixed: u64,
    branch: u64,
    // attention and its sparse-attention (QSA) scratch
    pa: u64,
    q: u64,
    iq: u64,
    att_po: u64,
    att_pm: u64,
    att_pl: u64,
    att_out: u64,
    att_ids: u64,
    att_nk: u64,
    att_sparse: u64,
    att_scores: u64, // 0 when the window cannot exceed the indexer's budget
    gated: u64,
    xs_gated: u64,
    // routed experts (MoEBuffers + grouped.Plan)
    moe_logits: u64,
    moe_pick: u64,
    moe_wts: u64,
    plan_members: u64,
    plan_items: u64,
    plan_counts: u64,
    plan_rank: u64,
    plan_hist: u64,
    moe_act: u64,
    moe_y: u64, // bf16 with the experts' prompt arithmetic (decode windows use it too), fp32 otherwise
    // GDN
    proj: u64,
    windows: u64, // prompt chunks: each row's conv taps
    sid: u64,
    conv_ptr: u64,
    pos_blk: u64,
    gout: u64,
    gxs: u64,
    attn_o: u64,
    streams: u64,
    logits: u64,
    // two ranks: fp32 partials and their gathers, the head's candidates
    part_branch: u64,
    part_moe: u64,
    g_branch: u64,
    g_moe: u64,
    cand: u64,
    cand_all: u64,
    // the n-gram embedding
    ple_v: u64,
    ple_emb: u64,
    xs_ple: u64,
    ple_keys: u64,
    ple_vals: u64,
    ple_gated: u64,
    ple_pss: u64,
    ple_nrow: u64,
    part: u64, // split-K partials of a window's largest matmul
    // the MTP head
    mtp_e: u64,
    mtp_xe: u64,
    mtp_eo: u64,
    mtp_hn: u64,
    mtp_xh: u64,
    mtp_hs: u64,
    mtp_in: u64,

    pub const Options = struct { prefill: bool = false, moe_prefill: ?bool = null, capacity: usize };

    const Field = struct { name: []const u8, bytes: usize };

    /// Every pointer field's bytes, in declaration order (Python's dtypes: bf16 2, f32 4, i32 4).
    pub fn layout(g: Geometry, rows: usize, o: Options) [field_count]usize {
        const R = rows;
        const D = g.hidden;
        const W = g.wide();
        const head_rows = if (o.prefill) ends else rows;
        const att_r = if (o.prefill) @min(rows, att_rows) else rows;
        const nch = g.nch(o.capacity);
        const ns = g.slots();
        const pairs = R * ns;
        const plan = planBytes(pairs, g.experts + 1);
        const moe_prefill = o.moe_prefill orelse o.prefill;
        const lin: usize = if (o.prefill) 1 else g.linear_layers;
        const two = g.world > 1;
        const nrow = R * g.pleHeads();
        const idw = g.index_budget + g.index_ratio;
        return .{
            R * 4,                                       R * 4,                                          R * W * 2,                                                      R * (D / 256) * g.streams * 4,
            R * W * 2,                                   R * (W / 32) * 4,                               R * (g.low + g.streams) * 2,                                    R * g.low * 2,
            R * g.low * 2,                               R * (g.low / 32) * 4,                           R * g.streams * 2,                                              R * g.streams * 2,
            R * W * 2,                                   R * D * 2,                                      R * (D / 32) * 4,                                               R * D * 2,
            R * g.attnWidth() * 2,                       R * g.heads * g.head_dim * 2,                   R * g.index_heads * g.index_dim * 2,                            att_r * nch * g.heads * g.head_dim * 4,
            att_r * nch * g.heads * 4,                   att_r * nch * g.heads * 4,                      att_r * g.heads * g.head_dim * 2,                               att_r * idw * 4,
            att_r * 4,                                   att_r * 4,                                      if (g.qsa(o.capacity)) att_r * g.blocks(o.capacity) * 4 else 0, R * g.heads * g.head_dim * 2,
            R * (g.heads * g.head_dim / 32) * 4,         R * (g.experts + 1) * 4,                        R * ns * 4,                                                     R * ns * 4,
            plan[0],                                     plan[1],                                        plan[2],                                                        plan[3],
            plan[4],                                     R * ns * g.moe_width * 2,                       R * ns * D * @as(usize, if (moe_prefill) 2 else 4),             lin * R * g.gdnWidth() * 2,
            if (o.prefill) R * g.conv_kernel * 4 else 0, if (o.prefill) R * 4 else 0,                    if (o.prefill) 8 else 0,                                        if (o.prefill) 4 else 0,
            R * g.nv * gdn_dv * 2,                       R * (g.nv * gdn_dv / 32) * 4,                   R * g.heads * g.head_dim * 2,                                   R * W * 2,
            head_rows * g.head_n * 2,                    if (two) R * D * 4 else 0,                      if (two) R * D * 4 else 0,                                      if (two) g.world * R * D * 4 else 0,
            if (two) g.world * R * D * 4 else 0,         if (two) head_rows * (2 * cand + 1) * 4 else 0, if (two) g.world * head_rows * (2 * cand + 1) * 4 else 0,       nrow * g.pleHeadDim() * 2,
            R * g.ple_dim * 2,                           R * (g.ple_dim / 32) * 4,                       R * W * 2,                                                      R * D * 2,
            R * W * 2,                                   R * g.streams * 4,                              R * W * 2,                                                      if (o.prefill) 4 else 32 * @max(rows, 4) * 2560 * 4,
            R * D * 2,                                   R * (D / 32) * 4,                               R * D * 2,                                                      R * W * 2,
            R * (W / 32) * 4,                            R * g.streams * D * 2,                          R * W * 2,
        };
    }

    pub const field_count = blk: {
        var n = 0;
        for (@typeInfo(Buffers).@"struct".field_types) |T| {
            if (T == u64) n += 1;
        }
        break :blk n;
    };

    pub fn bytes(g: Geometry, rows: usize, o: Options) usize {
        return aligned(&layout(g, rows, o));
    }

    pub fn init(d: *const cuda.Driver, g: Geometry, rows: usize, o: Options) !Buffers {
        const sizes = layout(g, rows, o);
        var b: Buffers = undefined;
        b.arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, aligned(&sizes)) };
        errdefer b.arena.buf.free();
        try b.arena.buf.fill8(0, null); // Python zeroes ids, last, proj and the PLE rows; the rest is overwritten
        b.rows = rows;
        b.prefill = o.prefill;
        var i: usize = 0;
        const info = @typeInfo(Buffers).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            if (T != u64) continue;
            @field(b, name) = if (sizes[i] == 0) 0 else b.arena.take(sizes[i]);
            i += 1;
        }
        return b;
    }

    pub fn deinit(b: *Buffers) void {
        b.arena.buf.free();
        b.* = undefined;
    }
};

/// One sequence's committed caches, grown `grow_step` rows at a time up to `limit` (Python State).
pub const State = struct {
    d: *const cuda.Driver,
    g: Geometry,
    limit: usize,
    capacity: usize,
    version: u32 = 0, // counts reallocations: captured graphs refresh their pointers on a change
    pos: usize = 0,
    mtp_len: usize = 0,
    mtp_drafted: usize = 0,
    rope_delta: i32 = 0,
    // the fixed part: GDN conv and recurrent state (double-buffered by `cur`), replay scratch, PLE tail
    fixed: Arena,
    conv: u64, // [linear, conv_kernel - 1, conv dim] bf16
    rec: u64, // [2, linear, nv, dv, dk] f32
    cur: [64]u1 = @splat(0),
    gdn_k: u64, // [linear][max_rows, nk, dk] f32, the last window's replay inputs
    gdn_v: u64, // [linear][max_rows, nv, dv] bf16
    gdn_g: u64, // [linear][max_rows, nv] f32
    gdn_b: u64, // [linear][max_rows, nv] f32
    ple_tail: u64, // [(ple_kernel - 1) * ngram_size, wide] bf16
    pos_dev: u64, // i32 scalars the kernels read: position, rope delta, the MTP head's length
    rope_delta_dev: u64,
    mtp_pos: u64,
    // the context caches, reallocated by `ensure`: per attention layer (and the MTP head's), k, v, ikc, pooled
    ctx: ?Arena = null,
    kc: [16]u64 = @splat(0),
    vc: [16]u64 = @splat(0),
    ikc: [16]u64 = @splat(0),
    pooled: [16]u64 = @splat(0),
    history: [3]i64 = @splat(0), // the n-gram history (n - 1 tokens)

    fn cacheLayers(g: Geometry) usize {
        return g.attention_layers + @intFromBool(g.mtp);
    }

    /// Bytes one attention layer's caches take at `rows` rows (KV bf16, indexer keys, pooled blocks).
    pub fn layerBytes(g: Geometry, rows: usize) usize {
        return rows * (2 * g.kv_heads * g.head_dim * 2 + g.index_dim * 2) + g.blocks(rows) * g.index_dim * 2;
    }

    /// The context caches' bytes at `rows` rows, the MTP head's included (State.cache_bytes).
    pub fn cacheBytes(g: Geometry, rows: usize) usize {
        return cacheLayers(g) * layerBytes(g, rows);
    }

    fn fixedSizes(g: Geometry) [10]usize {
        const n = g.linear_layers;
        return .{
            // one DeltaNet state a layer (the engine folds kept rows in place: cuda_forward.zig Round); the
            // replay inputs live in the engine's round scratch, not per sequence
            n * (g.conv_kernel - 1) * g.gdnConv() * 2,        n * g.nv * gdn_dv * gdn_dk * 4,
            0,                                                0,
            0,                                                0,
            (g.ple_kernel - 1) * g.ngram_size * g.wide() * 2, 4,
            4,                                                4,
        };
    }

    pub fn init(d: *const cuda.Driver, g: Geometry, capacity: usize, limit: usize) !State {
        if (g.linear_layers > 64 or cacheLayers(g) > 16) return error.TooManyLayers;
        const sizes = fixedSizes(g);
        var a: Arena = .{ .buf = try cuda.DeviceBuffer.alloc(d, aligned(&sizes)) };
        errdefer a.buf.free();
        try a.buf.fill8(0, null);
        var s: State = .{
            .d = d,
            .g = g,
            .limit = limit,
            .capacity = 0,
            .fixed = a,
            .conv = 0,
            .rec = 0,
            .gdn_k = 0,
            .gdn_v = 0,
            .gdn_g = 0,
            .gdn_b = 0,
            .ple_tail = 0,
            .pos_dev = 0,
            .rope_delta_dev = 0,
            .mtp_pos = 0,
        };
        const ptrs = [_]*u64{ &s.conv, &s.rec, &s.gdn_k, &s.gdn_v, &s.gdn_g, &s.gdn_b, &s.ple_tail, &s.pos_dev, &s.rope_delta_dev, &s.mtp_pos };
        for (ptrs, sizes) |p, n| p.* = s.fixed.take(n);
        try s.resize(@min(capacity, limit));
        return s;
    }

    pub fn deinit(s: *State) void {
        if (s.ctx) |*c| c.buf.free();
        s.fixed.buf.free();
        s.* = undefined;
    }

    /// Grow every context cache to hold `rows` rows, a `grow_step` at a time, at most `limit`; the bytes it added.
    pub fn ensure(s: *State, rows: usize) !usize {
        if (rows <= s.capacity) return 0;
        if (rows > s.limit) return error.ContextPastWindow;
        const before = cacheBytes(s.g, s.capacity);
        try s.resize(@min(s.limit, std.mem.alignForward(usize, rows, grow_step)));
        return cacheBytes(s.g, s.capacity) - before;
    }

    /// Reallocate the context caches at `rows` rows, keeping every committed row (the layers' `pos`, the MTP head's
    /// `mtp_len`). Python reallocates a layer at a time; here one allocation holds all, so a resize briefly holds both.
    pub fn resize(s: *State, rows: usize) !void {
        const keep = @max(s.pos, s.mtp_len);
        if (rows < keep) return error.CacheTooSmall;
        const g = s.g;
        const n = cacheLayers(g);
        const kv_row = g.kv_heads * g.head_dim * 2;
        var a: Arena = .{ .buf = try cuda.DeviceBuffer.alloc(s.d, n * (std.mem.alignForward(usize, rows * kv_row, 256) * 2 + std.mem.alignForward(usize, rows * g.index_dim * 2, 256) + std.mem.alignForward(usize, g.blocks(rows) * g.index_dim * 2, 256))) };
        errdefer a.buf.free();
        try a.buf.fill8(0, null);
        var kc: [16]u64 = @splat(0);
        var vc: [16]u64 = @splat(0);
        var ikc: [16]u64 = @splat(0);
        var pooled: [16]u64 = @splat(0);
        for (0..n) |i| {
            kc[i] = a.take(rows * kv_row);
            vc[i] = a.take(rows * kv_row);
            ikc[i] = a.take(rows * g.index_dim * 2);
            pooled[i] = a.take(g.blocks(rows) * g.index_dim * 2);
        }
        if (s.ctx) |*old| {
            for (0..n) |i| {
                // the MTP head's cache is the last layer and keeps its own committed rows
                const kept = if (g.mtp and i == n - 1) s.mtp_len else s.pos;
                const blocks_kept = g.blocks(kept);
                try copy(s.d, kc[i], s.kc[i], kept * kv_row);
                try copy(s.d, vc[i], s.vc[i], kept * kv_row);
                try copy(s.d, ikc[i], s.ikc[i], kept * g.index_dim * 2);
                try copy(s.d, pooled[i], s.pooled[i], blocks_kept * g.index_dim * 2);
            }
            old.buf.free();
        }
        s.ctx = a;
        s.kc = kc;
        s.vc = vc;
        s.ikc = ikc;
        s.pooled = pooled;
        s.capacity = rows;
        s.version +%= 1;
    }

    fn copy(d: *const cuda.Driver, dst: u64, src: u64, n: usize) !void {
        if (n == 0) return;
        try d.check(d.api.cuMemcpyDtoD_v2(dst, src, n), "cuMemcpyDtoD");
    }
};

const test_geometry: Geometry = .{
    .hidden = 2560,
    .streams = 4,
    .low = 320,
    .heads = 24,
    .kv_heads = 2,
    .head_dim = 256,
    .index_heads = 4,
    .index_dim = 128,
    .index_budget = 2048,
    .index_ratio = 4,
    .nk = 16,
    .nv = 48,
    .conv_kernel = 4,
    .experts = 512,
    .top_k = 10,
    .moe_width = 640,
    .ple_dim = 2560,
    .heads_per_ngram = 8,
    .ngram_size = 3,
    .ple_kernel = 4,
    .head_n = 248320,
    .linear_layers = 36,
    .attention_layers = 12,
    .mtp = true,
    .world = 1,
};

test "widths equal Python's for one GPU and a TP=2 rank" {
    const g = test_geometry;
    try std.testing.expectEqual(@as(usize, 16480), g.gdnWidth()); // 10240 conv + 6144 z + 48 b + 48 a
    try std.testing.expectEqual(@as(usize, 10240), g.gdnConv());
    try std.testing.expectEqual(@as(usize, 13952), g.attnWidth()); // 24*2*256 + 2*2*256 + 5*128
    try std.testing.expectEqual(@as(usize, 5), g.nch(262144)); // 2051 keys in 512-key chunks
    try std.testing.expect(g.qsa(262144));
    var r = g;
    r.heads = 12;
    r.kv_heads = 1;
    r.nk = 8;
    r.nv = 24;
    r.world = 2;
    try std.testing.expectEqual(@as(usize, 8240), r.gdnWidth());
    try std.testing.expectEqual(@as(usize, 7296), r.attnWidth()); // 12*2*256 + 2*1*256 + 5*128
}

test "the per-token cache cost and a 1M window at TP=2" {
    var g = test_geometry;
    // bf16: 12 + 1 layers x (KV 2 x 2 x 256 x 2 + indexer key 256 + pooled 256 / 4)
    try std.testing.expectEqual(@as(usize, 13 * (2048 + 256) * 4096 + 13 * 1024 * 256), State.cacheBytes(g, 4096));
    g.kv_heads = 1;
    g.world = 2;
    const gib = @as(f64, @floatFromInt(State.cacheBytes(g, 1 << 20))) / (1 << 30);
    // 13 layers x (1M x (2 x 1 x 256 x 2 + 256) + 256k blocks x 256) = 17.06 GiB a rank for one 1M-token sequence
    try std.testing.expect(gib > 17.0 and gib < 17.1);
}

test "buffer layouts have one size per pointer field" {
    const sizes = Buffers.layout(test_geometry, max_rows, .{ .capacity = 262144 });
    try std.testing.expectEqual(Buffers.field_count, sizes.len);
    const chunk = Buffers.bytes(test_geometry, prefill_rows, .{ .prefill = true, .capacity = 262144 });
    try std.testing.expect(chunk > Buffers.bytes(test_geometry, max_rows, .{ .capacity = 262144 }));
}
