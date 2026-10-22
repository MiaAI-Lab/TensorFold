//! The Flash Next CUDA engine (Python qwen4_exp/cuda/decode.py Engine and engine.py's setup), the contract of
//! work/PLAN.md "Engine API contract": the config, the extension kernels (cuda_kernels.zig), the torch-op kernels
//! (cuda_torch_ops.zig), the captured Triton set (`kernels_dir`: aot.json + cubins), the weights (cuda_weights.zig),
//! the decode window's, the MTP head's and the prompt chunk's buffers, and sequences (cuda_state.State each). A
//! prompt prefills in 2048-row chunks with the MTP head absorbing each chunk's next tokens, and the bound sequence
//! offers cuda_decode.zig's `E` interface, so cuda_decode.serial and cuda_decode.drafted run on it unchanged.
//!
//! Every call that runs collectives (two ranks) must be made in the same order with the same arguments on both.
//!
//! Python source (TensorFold, https://github.com/ashhart/TensorFold, qwen4_exp/cuda/decode.py and engine.py,
//! authored by Ash Hart (ashhart) with commits by vcruz305 inside TensorFold, which the owner allows for this port).
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const kern = @import("cuda_kernels.zig");
const tri = @import("cuda_triton.zig");
const tops = @import("cuda_torch_ops.zig");
const weights = @import("cuda_weights.zig");
const state = @import("cuda_state.zig");
const cfgs = @import("cuda_config.zig");
const comms = @import("cuda_comm.zig");
const rope = @import("cuda_rope.zig");
const sampler = @import("cuda_sampler.zig");
const decode = @import("cuda_decode.zig");
const fwd = @import("cuda_forward.zig");
const fp8 = @import("cuda_fp8.zig");
const int4 = @import("cuda_int4.zig");
const mtp = @import("cuda_mtp.zig");
const vmm = @import("cuda_vmm.zig");
const nucleus = @import("cuda_nucleus.zig");
const profs = @import("cuda_prof.zig");
const prompt_mm = @import("cuda_prompt.zig");
const moep = @import("cuda_moe_prompt.zig");

const Allocator = std.mem.Allocator;

/// MTP drafts a round by default (Python DEPTH) and their chain's confidence (CONFIDENCE).
pub const default_depth = 6;
pub const default_confidence = 0.70;

/// The chains' stop rule as served (cuda_decode.Stop): the running product p1 * ... * pj >= 0.4 (encoded -0.4; decode
/// D4 after X1, research/X1-drafts.md 3.4: TP=2 code +13-19%, sampled hash map +7.5% over depth 6 / 0.70; a fixed 0.5
/// at depth 15 cost hash-map prose 10% served), or TF_FLASHNEXT_CONFIDENCE (-1..1: c > 0 Python's per-draft rule, c < 0
/// the running product at -c, 0 every draft). The oracle runs keep `default_confidence`. Drafts only: the kept tokens
/// are the target's either way.
pub const served_confidence = -0.4;

/// The served hybrid: the running product (served_confidence) while at most this many streams are live, Python's
/// per-draft rule at `wide_confidence` above that (TF_FLASHNEXT_PRODUCT_STREAMS; 2 by default). W6's served A/B: the
/// product with the longest asks wins at one or two streams (code +10%) and loses prose at four and eight, where
/// deeper chains cost rows (STATUS 2026-10-07).
pub const product_streams_default = 2;
pub const wide_confidence = 0.5;

pub fn productStreams() usize {
    const v = std.c.getenv("TF_FLASHNEXT_PRODUCT_STREAMS") orelse return product_streams_default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch product_streams_default;
}

pub fn confidenceSetting() f64 {
    const v = std.c.getenv("TF_FLASHNEXT_CONFIDENCE") orelse return served_confidence;
    const x = std.fmt.parseFloat(f64, std.mem.span(v)) catch return served_confidence;
    return if (x >= -1 and x <= 1) x else served_confidence;
}
/// The most candidates a main-head draw reads back (top_k + MARGIN); larger top_k values are refused.
pub const max_k = 256;

pub const Options = struct {
    context: usize = 262144,
    mtp: bool = true,
    /// CUDA graphs of the decode windows and MTP steps (Python graphs.py; eager gives the same bits)
    graphs: bool = true,
    rank: u32 = 0,
    world: u32 = 1,
    comm: ?*const comms.Comm = null,
    /// null below 262,144 (cuda_rope.zig, W9)
    yarn: ?rope.Yarn = null,
    /// MTP drafts a round at most: the decode buffers hold depth + 1 rows (at least 8, Python max(8, depth + 1))
    depth: usize = default_depth,
    /// streams a shared round may hold (verifyMany): the decode buffers hold streams x (depth + 1) rows
    streams: usize = 1,
    /// bytes the sequences' caches may take on this rank; null: what is free once loaded, less `reserve_gib`
    kv_budget: ?usize = null,
    /// memory kept free beside the caches (the owner's floor is 10 GiB MemAvailable); env
    /// TENSORFOLD_MEMORY_RESERVE_GIB overrides it in the CLI and the server
    reserve_gib: f64 = 12,
    prefill_rows: usize = state.prefill_rows,
    /// sequences' caches in reserved address space, grown in place, indexer keys in a ring (cuda_vmm.zig);
    /// false: State's allocations, grown by copying
    vmm: bool = true,
};

/// Rows the indexer-key ring holds at least: a prompt chunk's rows and the 3 before them that a pool reads.
pub const ring_min_rows = 4096;

pub const Seq = fwd.Seq;

/// A captured decode window (or MTP step): its sequence's buffers, the window's rows, the DeltaNet buffers' parity
/// and the attention context bucket (Python Graphs.main / Graphs.mtp keys).
const GraphKey = struct { seq: usize, version: u32, mtp: bool, rows: u32, parity: u32, ctx: usize };

/// A shared round's (or shared MTP step's) graph and the sequences whose buffers it bakes in.
const MultiGraph = struct { exec: cuda.graph.Exec, seqs: []usize };

/// Shared-round graphs kept at most (drafted rounds vary their windows' lengths; a composition is captured
/// the second time it is seen).
const max_multi_graphs = 192;
/// TF_FLASHNEXT_MULTI_GRAPHS: the most streams a shared step is captured for. Wider steps rarely repeat their
/// composition, are GPU-bound (the launches hide behind the kernels), and a capture costs a second enqueue plus the
/// instantiation: measured at TP=2, x8/x16/x32 run 2-27% faster eagerly (work/research/W6-rounds-profile.md).
const multi_graphs_default = 4;

/// Graphs._bucket: the attention bound a captured window replays at, a power of two from 8192 up to the capacity.
pub fn bucket(end: usize, capacity: usize) usize {
    const p = std.math.ceilPowerOfTwo(usize, @max(end, 1)) catch end;
    return @min(capacity, @max(8192, p));
}

fn seconds(io: std.Io, since: std.Io.Timestamp) f64 {
    const now = std.Io.Clock.awake.now(io);
    return @as(f64, @floatFromInt(now.toNanoseconds() - since.toNanoseconds())) / 1e9;
}

/// /proc/meminfo's MemAvailable in bytes, or null.
fn memAvailable(io: std.Io) ?usize {
    var buf: [4096]u8 = undefined;
    const text = std.Io.Dir.cwd().readFile(io, "/proc/meminfo", &buf) catch return null;
    var it = std.mem.tokenizeScalar(u8, text, '\n');
    while (it.next()) |line| if (std.mem.startsWith(u8, line, "MemAvailable:")) {
        var f = std.mem.tokenizeAny(u8, line["MemAvailable:".len..], " \t");
        const kb = std.fmt.parseInt(usize, f.next() orelse return null, 10) catch return null;
        return kb * 1024;
    };
    return null;
}

/// cfgs.Yarn (the config's form) from cuda_rope's.
fn yarnConfig(y: rope.Yarn) cfgs.Yarn {
    return .{ .factor = y.factor, .original_max_position_embeddings = y.original, .beta_fast = y.beta_fast, .beta_slow = y.beta_slow, .attention_factor = if (y.attention_factor == 0) null else y.attention_factor, .truncate = y.truncate };
}

/// TF_FLASHNEXT_MTP_TOPK: the MTP layer's routed experts a row (drafts only; 0 or unset: the config's top_k).
fn mtpTopK(top_k: u32) usize {
    const v = envGet("TF_FLASHNEXT_MTP_TOPK") orelse return 0;
    const n = std.fmt.parseInt(usize, v, 10) catch return 0;
    // the kernel set routes the checkpoints' own values (10, INT4-AutoRound's 5); 10 is the only other one accepted
    if (n != 10 or n == top_k) {
        if (n != top_k) std.log.warn("TF_FLASHNEXT_MTP_TOPK={d} ignored: the kernel set has top-5 and top-10 routing", .{n});
        return 0;
    }
    return n;
}

/// The shapes one rank computes with (cuda_state.Geometry and cuda_triton.Dims) from the config's and the split.
pub fn geometry(c: *const cfgs.Config, world: u32, has_mtp: bool) !state.Geometry {
    const sh = try c.shares(world);
    return .{
        .hidden = c.hidden,
        .streams = c.streams,
        .low = c.low,
        .heads = sh.heads,
        .kv_heads = sh.kv_heads,
        .head_dim = c.head_dim,
        .index_heads = c.index_heads,
        .index_dim = c.index_dim,
        .index_budget = c.index_budget,
        .index_ratio = c.index_ratio,
        .nk = sh.nk,
        .nv = sh.nv,
        .conv_kernel = c.conv_kernel,
        .experts = c.experts,
        .top_k = c.top_k,
        .moe_width = sh.moe_width,
        .ple_dim = c.ple_dim,
        .heads_per_ngram = c.heads_per_ngram,
        .ngram_size = c.ngram_size,
        .ple_kernel = c.ple_kernel,
        .head_n = c.vocab / world,
        .linear_layers = c.count(.linear),
        .attention_layers = c.count(.attention),
        .mtp = has_mtp,
        .world = world,
        .shared_width = sh.shared_width,
        .int4ar = c.int4ar(),
        .mtp_top_k = mtpTopK(c.top_k),
    };
}

pub fn dims(c: *const cfgs.Config, g: state.Geometry) tri.Dims {
    return .{
        .hidden = g.hidden,
        .streams = g.streams,
        .low = g.low,
        .heads = g.heads,
        .kv_heads = g.kv_heads,
        .head_dim = g.head_dim,
        .index_heads = g.index_heads,
        .index_dim = g.index_dim,
        .rotary_half = c.rope.rotary_dim / 2,
        .experts = g.experts,
        .top_k = g.top_k,
        .moe_width = g.moe_width,
        .vocab = g.head_n,
        .key_heads = g.nk,
        .value_heads = g.nv,
        .gdn_dim = c.dk,
        .ngram_heads = g.pleHeads(),
        .ngram_dim = g.pleHeadDim(),
        .ple_taps = g.ple_kernel,
        .ngram_size = g.ngram_size,
        .eps = @floatCast(c.eps),
    };
}

/// Field-wise maximum of sampling scratch sizes.
fn sampleSizes(f: *const tops.Functions, cases: []const [3]usize) ![7]usize {
    var most: [7]usize = @splat(0);
    for (cases) |c| {
        const s = try tops.Torch.Sample.bytes(f, c[0], c[1], c[2]);
        for (&most, s) |*m, v| m.* = @max(m.*, v);
    }
    return most;
}

fn takeSample(a: *state.Arena, sz: [7]usize) tops.Torch.Sample {
    return .{ .f32 = a.take(sz[0]), .vals = a.take(sz[1]), .idx = a.take(sz[2]), .topk = a.take(sz[3]), .lse = a.take(sz[4]), .lse_scratch = a.take(sz[5]), .col = a.take(sz[6]) };
}

fn aligned(sizes: []const usize) usize {
    var t: usize = 0;
    for (sizes) |n| t += std.mem.alignForward(usize, n, 256);
    return t;
}

/// A non-blocking stream at `priority` (lower is higher; the driver clamps to its range).
fn streamWithPriority(d: *const cuda.Driver, priority: c_int) !cuda.Stream {
    if (priority == 0) return cuda.Stream.init(d, true);
    var lib = std.DynLib.open("libcuda.so.1") catch return cuda.Stream.init(d, true);
    defer lib.close();
    const F = *const fn (*cuda.abi.Stream, c_uint, c_int) callconv(.c) cuda.abi.Result;
    const f = lib.lookup(F, "cuStreamCreateWithPriority") orelse return cuda.Stream.init(d, true);
    var s: cuda.abi.Stream = null;
    try d.check(f(&s, cuda.abi.stream_non_blocking, priority), "cuStreamCreateWithPriority");
    return .{ .d = d, .handle = s };
}

fn envGet(name: [:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

fn envOff(name: [:0]const u8) bool {
    const v = envGet(name) orelse return false;
    return std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off");
}

/// TF_FLASHNEXT_PREFILL_ROWS: the prompt chunk's rows (both ranks the same; any chunking gives the same bits), else
/// the caller's.
/// Two ranks' prompt chunk rows: with the in-program K slices and the prompt expert kernel 4096 rows a call are
/// faster than 2048 (TP=2 A/B, work/research/R2-measured.md); one GPU keeps Python's 2048.
pub const tp2_prefill_rows: usize = 4096;

fn promptRows(default: usize) !usize {
    const v = envGet("TF_FLASHNEXT_PREFILL_ROWS") orelse return default;
    const n = std.fmt.parseInt(usize, v, 10) catch return error.BadPrefillRows;
    if (n < 16 or n > 16384) return error.BadPrefillRows;
    return n;
}

pub const Engine = struct {
    gpa: Allocator,
    io: std.Io,
    ctx: *const cuda.Context,
    stream: cuda.Stream,
    k: kern.Kernels,
    torch: tops.Ops,
    set: cuda.aot.Set,
    c: cfgs.Config,
    w: weights.Weights,
    g: state.Geometry,
    dims: tri.Dims,
    buf: fwd.Bufs,
    mbuf: ?fwd.Bufs = null,
    pbuf: fwd.Bufs,
    f: fwd.Forward,
    draws: ?mtp.Draws = null,
    /// decode D5: the MTP head's matrices in 4-bit (TF_FLASHNEXT_MTP_Q4=1), drafts only
    mtpq: ?@import("cuda_mtp_q4.zig").Table = null,
    /// host copy of the draft ids (this rank's), column -> token
    draft_host: []u32 = &.{},
    /// sampling scratch, the prompt's last logits and gathered candidates, the draft read-back
    extra: state.Arena,
    main_sample: tops.Torch.Sample,
    tp: ?mtp.TpScratch = null,
    /// top_k-off draws' device scratch (one row of masses)
    nsc: nucleus.Scratch = undefined,
    last_logits: u64,
    last_cand: u64,
    /// two ranks: the last shared round's gathered candidates on the host, read once for every window's draws
    cand_host: std.ArrayList(f32) = .empty,
    cand_round: u64 = 0,
    cand_have: u64 = std.math.maxInt(u64),
    /// TF_FLASHNEXT_DRAW_ONCE=0: each window's and draft's draws read the candidates again (the old way)
    draw_once: bool = true,
    /// generateMany's served hybrid (gate-many --served): a running product applies while at most this many streams
    /// are live, `wide_confidence` above (maxInt: the request's rule always)
    product_streams: usize = std.math.maxInt(usize),
    /// shared steps of at most this many streams are captured (TF_FLASHNEXT_MULTI_GRAPHS)
    multi_graphs: usize = multi_graphs_default,
    /// the Triton variants each launch site picked (the forward's launches skip `find`'s scan)
    memo: tri.Memo = undefined,
    /// the rows the context caches hold (Python max_len: the window plus depth + 1 speculative rows)
    max_len: usize,
    depth: usize,
    rank: u32,
    world: u32,
    prefill_rows: usize,
    sampling: ?lanes.Sampling = null,
    /// prompt chunks absorb into the MTP head (Python prefill's `mtp`; the serial reference prefills without)
    absorbing: bool = true,
    own: *Seq = undefined,
    bound: *Seq = undefined,
    rows: usize = 0,
    load_seconds: f64 = 0,
    graphs_on: bool = true,
    /// the last shared round's streams and their rows (verifyMany)
    round_segs: []fwd.Seg = &.{},
    round_n: usize = 0,
    times: Times = .{},
    budget: fwd.Budget = .{ .limit = std.math.maxInt(usize) },
    vm: ?vmm.Vmm = null,
    /// the fill whose chunk holds the prompt buffers between slices (one at a time; prefill waits for it)
    pfill: ?*const Fill = null,
    graphs: std.AutoHashMapUnmanaged(GraphKey, cuda.graph.Exec) = .empty,
    captures: usize = 0,
    /// shared rounds' and shared MTP steps' graphs by composition (multiKey), and compositions seen once
    multi: std.AutoHashMapUnmanaged(u64, MultiGraph) = .empty,
    seen: std.AutoHashMapUnmanaged(u64, void) = .empty,
    /// TF_FLASHNEXT_PROFILE: each prompt chunk's GPU time by part (cuda_prof.zig)
    prof: ?*profs.Prof = null,
    /// TF_FLASHNEXT_PROMPT_EXPERTS (default on): prompt chunks' experts read weights once a call
    px: ?moep.Prompt4 = null,
    /// TF_FLASHNEXT_PREFILL_DIGEST: each prompt chunk's final streams (every row) and last logits hashed and printed
    digest: bool = false,
    /// INT4-AutoRound: the block-FP8 lane matmul (cuda_fp8.zig) and the GPTQ int4 kernels (cuda_int4.zig)
    k8: ?fp8.Kernels = null,
    k4: ?int4.Kernels = null,

    pub fn init(gpa: Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels_dir: []const u8, o: Options) !*Engine {
        const t0 = std.Io.Clock.awake.now(io);
        if (o.world < 1 or o.world > 2 or o.rank >= o.world) return error.BadRank;
        if (o.world > 1 and o.comm == null) return error.TwoRanksNeedComm;
        if (o.depth > state.max_rows - 1) return error.DepthPastWindow;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = .{ .gpa = gpa, .io = io, .ctx = ctx, .stream = undefined, .k = undefined, .torch = undefined, .set = undefined, .c = undefined, .w = undefined, .g = undefined, .dims = undefined, .buf = undefined, .pbuf = undefined, .f = undefined, .extra = undefined, .main_sample = undefined, .last_logits = 0, .last_cand = 0, .max_len = 0, .depth = if (o.mtp) o.depth else 0, .rank = o.rank, .world = o.world, .prefill_rows = try promptRows(if (o.world == 2 and o.prefill_rows == state.prefill_rows) tp2_prefill_rows else o.prefill_rows), .graphs_on = o.graphs };
        const d = ctx.d;
        e.c = try cfgs.Config.read(gpa, io, dir, .{ .yarn = if (o.yarn) |y| yarnConfig(y) else null });
        errdefer e.c.deinit();
        e.stream = try cuda.Stream.init(d, true);
        errdefer e.stream.deinit();
        e.k = try kern.Kernels.load(ctx);
        errdefer e.k.deinit();
        e.torch = try tops.Ops.load(ctx);
        errdefer e.torch.deinit();
        e.set = try cuda.aot.Set.load(gpa, io, d, ctx.device, kernels_dir);
        errdefer e.set.deinit();
        // TF_FLASHNEXT_INT4AR_FAST=1: INT4-AutoRound's fast-fp8/ variant over the checkpoint (block-FP8 hyper-connections
        // and MTP experts; lossy against the bf16 ones, opt-in)
        var overlay_buf: [1024]u8 = undefined;
        const overlay: ?[]const u8 = if (e.c.int4ar() and envGet("TF_FLASHNEXT_INT4AR_FAST") != null and !envOff("TF_FLASHNEXT_INT4AR_FAST")) try std.fmt.bufPrint(&overlay_buf, "{s}/fast-fp8", .{dir}) else null;
        e.w = try weights.load(gpa, io, dir, &e.c, .{ .rank = o.rank, .world = o.world, .mtp = o.mtp, .draft_head = o.mtp, .mode = .device, .driver = d, .overlay = overlay });
        errdefer e.w.deinit();
        if (o.mtp and e.w.mtp == null) return error.NoMtpHead;
        e.g = try geometry(&e.c, o.world, e.w.mtp != null);
        e.dims = dims(&e.c, e.g);
        // Python: cache_slots = the window + depth + 1 (capacity.admit), every buffer's attention geometry at it
        e.max_len = o.context + e.depth + 1;
        const rows = @max(8, @max(1, o.streams) * (e.depth + 1));
        e.round_segs = try gpa.alloc(fwd.Seg, @max(1, o.streams));
        errdefer gpa.free(e.round_segs);
        const ag = tri.AttnGeometry.init(e.max_len, e.g.index_budget, e.g.index_ratio);
        e.buf = .{ .b = try state.Buffers.init(d, e.g, rows, .{ .moe_prefill = true, .capacity = e.max_len }), .y_f32 = false, .attn = ag };
        errdefer e.buf.deinit();
        if (e.w.mtp != null) e.mbuf = .{ .b = try state.Buffers.init(d, e.g, rows, .{ .capacity = e.max_len }), .y_f32 = true, .attn = ag };
        errdefer if (e.mbuf) |*m| m.deinit();
        e.pbuf = .{ .b = try state.Buffers.init(d, e.g, e.prefill_rows, .{ .prefill = true, .capacity = e.max_len }), .y_f32 = false, .attn = tri.AttnGeometry.init(e.max_len, e.g.index_budget, e.g.index_ratio) };
        errdefer e.pbuf.deinit();
        try e.initWindows();
        const th = e.torch.on(e.stream);
        e.memo = .{ .gpa = gpa };
        errdefer e.memo.deinit();
        e.f = try fwd.Forward.init(gpa, d, e.stream, .{ .k = &e.k, .s = e.stream }, .{ .set = &e.set, .s = e.stream, .memo = &e.memo }, th, &e.w, &e.c, e.g, e.dims, e.prefill_rows);
        errdefer e.f.deinit();
        e.f.comm = o.comm;
        // decode rounds' DeltaNet scratch: every stream's window rows, two rounds' kept inputs
        try e.f.initRound(rows, @max(1, o.streams));
        if (e.c.int4ar()) {
            e.k8 = try fp8.Kernels.load(ctx);
            e.k4 = try int4.Kernels.load(ctx);
            e.f.k8 = &e.k8.?;
            e.f.k4 = &e.k4.?;
        }
        errdefer if (e.k8) |*q| q.deinit();
        errdefer if (e.k4) |*q| q.deinit();
        // prompt matmuls without split-K partials when the kernel set has them (TF_FLASHNEXT_PROMPT_MM=0: off)
        e.f.prompt_mm = !envOff(prompt_mm.env) and prompt_mm.available(&e.set);
        e.draw_once = !envOff("TF_FLASHNEXT_DRAW_ONCE");
        // the shared expert beside the routed experts on a second stream (TF_FLASHNEXT_SHARED_SIDE=0: in line)
        if (!envOff("TF_FLASHNEXT_SHARED_SIDE")) try e.f.initSide();
        // the shared expert's single-slice gate/up on fn_ops' K-serial kernel where it is faster (the same bits,
        // fp4-check; TF_FLASHNEXT_FP4_SERIAL=0 off)
        e.f.fp4_serial = !envOff("TF_FLASHNEXT_FP4_SERIAL");
        if (envGet("TF_FLASHNEXT_MULTI_GRAPHS")) |v| e.multi_graphs = std.fmt.parseInt(usize, v, 10) catch multi_graphs_default;
        // shared rounds past the captured widths run their attention in one launch a kernel (attn_multi.py's
        // kernels, aot/w6-ks1 and later; without them, a stream at a time): TF_FLASHNEXT_ATTN_MULTI=N from N
        // streams (never at a captured width), 0 off
        if (tri.Tri.multiAvailable(&e.set)) {
            const want: usize = if (envGet("TF_FLASHNEXT_ATTN_MULTI")) |v| (std.fmt.parseInt(usize, v, 10) catch 0) else e.multi_graphs + 1;
            if (want > 0) e.f.attn_multi = @max(want, e.multi_graphs + 1, 2);
        }
        e.f.count_experts = if (envGet("TF_FLASHNEXT_COUNT_EXPERTS")) |v| std.mem.eql(u8, v, "1") else false;
        if (!envOff(moep.env)) {
            e.px = try moep.Prompt4.init(ctx);
            e.f.px = &e.px.?;
        }
        errdefer if (e.px) |*q| q.deinit();
        // two ranks: prompt glue split by rows (TF_FLASHNEXT_GLUE_SPLIT=0: every rank runs every row)
        e.f.glue_split = o.world == 2 and !envOff("TF_FLASHNEXT_GLUE_SPLIT");
        e.digest = envGet("TF_FLASHNEXT_PREFILL_DIGEST") != null;
        // the split glue's exchanges on their own stream beside the other rows' work (TF_FLASHNEXT_OVERLAP=0: in line)
        if (envGet("TF_FLASHNEXT_QSA_RT")) |v| {
            e.f.qsa_rt = std.fmt.parseInt(usize, v, 10) catch 0;
        }
        if (envGet("TF_FLASHNEXT_QSA_TILES_FROM")) |v| {
            e.f.qsa_tiles_from = std.fmt.parseInt(usize, v, 10) catch std.math.maxInt(usize);
        }
        if (envGet("TF_FLASHNEXT_QSA_SPLIT_KEYS")) |v| {
            e.f.qsa_split_keys = std.fmt.parseInt(usize, v, 10) catch 0;
        }
        // measured ~1% (research/R2-measured.md): off unless asked
        if (e.f.glue_split and envGet("TF_FLASHNEXT_OVERLAP") != null and !envOff("TF_FLASHNEXT_OVERLAP")) {
            if (envGet("TF_FLASHNEXT_OVERLAP")) |v| e.f.overlap = std.fmt.parseInt(u8, v, 10) catch 1;
            const prio: c_int = if (envGet("TF_FLASHNEXT_COMM_PRIORITY")) |v| (std.fmt.parseInt(c_int, v, 10) catch 0) else 0;
            e.f.cs = try streamWithPriority(d, prio);
            for (&e.f.cs_ev) |*ev| ev.* = try cuda.Event.init(d, false);
            // CTAs the persistent expert kernels leave free for NCCL's (the grids loop over units: same bits)
            if (envGet("TF_FLASHNEXT_EXPERT_SMS_FREE")) |v| {
                const free = std.fmt.parseInt(usize, v, 10) catch 0;
                const sms: usize = @intCast(e.k.sms);
                for (&e.k.nvfp4_blocks) |*n| n.* = n.* / sms * (sms - @min(free, sms - 1));
                if (e.px) |*q| for (&q.resident) |*n| {
                    n.* = n.* / sms * (sms - @min(free, sms - 1));
                };
            }
        }
        errdefer if (e.f.cs) |*cs| {
            cs.deinit();
            for (&e.f.cs_ev) |*ev| ev.deinit();
        };
        if (profs.wanted(envGet(profs.env))) {
            e.prof = try profs.Prof.init(gpa, d, 16384);
        }
        errdefer if (e.prof) |p| p.deinit(gpa);
        if (o.rank == 0) std.debug.print("prompt path: {d}-row chunks, glue {s}{s}, matmul K slices {s}, experts {s}{s}\n", .{ e.prefill_rows, if (e.f.glue_split) "split by rows" else "on every row", if (e.f.cs != null) " (exchanges overlapped)" else "", if (e.f.prompt_mm) "in one program (_b16mm_ks/_fp4mm_ks)" else "split with _reduce", if (e.px != null) "on 128-pair items (fn_experts_prompt)" else "on 16-pair items", if (e.prof != null) ", profiled" else "" });
        if (o.rank == 0) std.debug.print("prompt path: {d}-row chunks, glue {s}, matmul K slices {s}, experts {s}{s}\n", .{ e.prefill_rows, if (e.f.glue_split) "split by rows" else "on every row", if (e.f.prompt_mm) "in one program (_b16mm_ks/_fp4mm_ks)" else "split with _reduce", if (e.px != null) "on 128-pair items (fn_experts_prompt)" else "on 16-pair items", if (e.prof != null) ", profiled" else "" });
        if (e.w.mtp != null) if (std.c.getenv("TF_FLASHNEXT_MTP_Q4")) |v| if (!std.mem.eql(u8, std.mem.span(v), "0")) {
            const mse = if (std.c.getenv("TF_FLASHNEXT_MTP_Q4_MSE")) |x| std.mem.eql(u8, std.mem.span(x), "1") else false;
            e.mtpq = try @import("cuda_mtp_q4.zig").Table.build(gpa, d, &e.w, std.mem.span(v), mse);
            e.f.mtp_q4 = &e.mtpq.?;
            std.log.info("MTP head in 4-bit for the drafts: {d} matrices, {d:.1} -> {d:.1} MB", .{ e.mtpq.?.n, @as(f64, @floatFromInt(e.mtpq.?.bytes_bf16)) / 1e6, @as(f64, @floatFromInt(e.mtpq.?.bytes_q4)) / 1e6 });
        };
        errdefer if (e.mtpq) |*t| t.deinit();
        // sampling scratch: the main head's rows and the draft head's one row
        const draft_n: usize = if (e.w.draft_head) |q| q.n else e.g.head_n;
        const main_sz = try sampleSizes(&e.torch.f, &.{ .{ state.max_rows, e.g.head_n, max_k }, .{ 1, e.g.head_n, max_k }, .{ rows, e.g.head_n, state.cand }, .{ 1, draft_n, state.cand }, .{ rows, draft_n, state.cand } });
        const draft_sz = try sampleSizes(&e.torch.f, &.{.{ 1, draft_n, draft_n }});
        const pack = state.max_rows * (2 * max_k + 1) * 4;
        const fixed = [_]usize{ e.g.head_n * 2, o.world * (2 * state.cand + 1) * 4, (2 * draft_n + 1) * 4, if (o.world > 1) pack else 0, if (o.world > 1) o.world * pack else 0, @max(e.g.head_n, draft_n) * 8, 8, 8 };
        e.extra = .{ .buf = try cuda.DeviceBuffer.alloc(d, aligned(&main_sz) + aligned(&draft_sz) + aligned(&fixed)) };
        errdefer e.extra.buf.free();
        try e.extra.buf.fill8(0, null);
        e.main_sample = takeSample(&e.extra, main_sz);
        e.f.sample = e.main_sample;
        const draft_s = takeSample(&e.extra, draft_sz);
        e.last_logits = e.extra.take(fixed[0]);
        e.last_cand = e.extra.take(fixed[1]);
        const draft_out = e.extra.take(fixed[2]);
        if (o.world > 1) e.tp = .{ .s = e.main_sample, .pack = e.extra.take(fixed[3]), .all = e.extra.take(fixed[4]), .max_rows = state.max_rows, .max_k = max_k };
        e.nsc = .{ .mass = e.extra.take(fixed[5]), .top = e.extra.take(fixed[6]), .sum = e.extra.take(fixed[7]) };
        if (e.w.mtp != null) {
            if (e.w.draft_head != null) e.draft_host = try weights.draftIds(gpa, weights.draft_vocab, e.c.vocab, o.rank, o.world);
            e.draws = .{ .gpa = gpa, .ids = if (e.w.draft_head != null) e.draft_host else null, .columns = draft_n, .world = o.world, .s = draft_s, .out = draft_out, .max_k = draft_n, .tp = e.tp, .nsc = e.nsc };
        }
        errdefer gpa.free(e.draft_host);
        if (o.vmm) {
            e.vm = vmm.Vmm.init(ctx) catch |err| blk: {
                std.log.warn("CUDA VMM unavailable ({s}): caches grow by copying", .{@errorName(err)});
                break :blk null;
            };
            if (e.vm) |v| std.log.info("caches in reserved address space ({d} KiB granularity), indexer keys in a ring", .{v.granularity >> 10});
        }
        // the sequences' memory: what is free now (MemAvailable: GB10's memory is unified), less the reserve
        try e.stream.synchronize();
        // the floor the server and the recipe set (TENSORFOLD_MEMORY_RESERVE_GIB) wins over the option
        var reserve_gib = o.reserve_gib;
        if (std.c.getenv("TENSORFOLD_MEMORY_RESERVE_GIB")) |v| reserve_gib = std.fmt.parseFloat(f64, std.mem.span(v)) catch return error.BadMemoryReserve;
        const reserve: usize = @intFromFloat(reserve_gib * (1 << 30));
        // MemAvailable counts the mapped n-gram table's reclaimable pages (they then page from disk, as Python's
        // admission accepts); the device's own count of free memory leaves them out
        const free = memAvailable(io) orelse (try ctx.memInfo()).free;
        e.budget = .{ .limit = o.kv_budget orelse (free -| reserve) };
        // two ranks decide growth together: both take the smaller budget
        if (o.comm) |cm| {
            const v: cuda.DeviceBuffer = .{ .d = d, .ptr = e.f.sc.vote, .len = 8 };
            try v.upload(0, std.mem.asBytes(&@as(u64, e.budget.limit)));
            try cm.allGather(e.f.sc.vote, e.f.sc.vote + 64, 1, .u64, e.stream.handle);
            try e.stream.synchronize();
            var both: [2]u64 = undefined;
            const r: cuda.DeviceBuffer = .{ .d = d, .ptr = e.f.sc.vote + 64, .len = 16 };
            try r.download(0, std.mem.sliceAsBytes(&both));
            e.budget.limit = @min(both[0], both[1]);
        }
        e.f.budget = &e.budget;
        e.budget.floor = @intFromFloat(@min(reserve_gib, 10) * (1 << 30));
        std.log.info("sequence memory: {d:.2} GiB ({d:.2} GiB free, {d:.1} GiB kept free)", .{ @as(f64, @floatFromInt(e.budget.limit)) / (1 << 30), @as(f64, @floatFromInt(free)) / (1 << 30), reserve_gib });
        e.own = try e.newSeq(e.max_len);
        e.bound = e.own;
        try e.stream.synchronize();
        e.load_seconds = seconds(io, t0);
        return e;
    }

    /// The prompt buffers' DeltaNet taps: row r reads [conv state (3) | rows] at r .. r + 3, stream 0 (Python
    /// Buffers.windows and sid).
    fn initWindows(e: *Engine) !void {
        const b = &e.pbuf.b;
        const taps = e.g.conv_kernel;
        const host = try e.gpa.alloc(i32, b.rows * taps);
        defer e.gpa.free(host);
        for (0..b.rows) |r| for (0..taps) |j| {
            host[r * taps + j] = @intCast(r + j);
        };
        const buf: cuda.DeviceBuffer = .{ .d = e.ctx.d, .ptr = b.windows, .len = host.len * 4 };
        try buf.upload(0, std.mem.sliceAsBytes(host));
        const sid: cuda.DeviceBuffer = .{ .d = e.ctx.d, .ptr = b.sid, .len = b.rows * 4 };
        try sid.fill32(0, null);
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        var it = e.graphs.valueIterator();
        while (it.next()) |g| g.deinit();
        e.graphs.deinit(e.gpa);
        e.graphs = .empty;
        e.dropMulti(null);
        e.multi.deinit(e.gpa);
        e.multi = .empty;
        e.seen.deinit(e.gpa);
        e.seen = .empty;
        e.freeSeq(e.own);
        e.gpa.free(e.draft_host);
        e.gpa.free(e.round_segs);
        e.extra.buf.free();
        if (e.prof) |p| p.deinit(e.gpa);
        e.cand_host.deinit(e.gpa);
        e.memo.deinit();
        if (e.px) |*q| q.deinit();
        if (e.k8) |*q| q.deinit();
        if (e.k4) |*q| q.deinit();
        if (e.mtpq) |*t| t.deinit();
        if (e.f.cs) |*cs| {
            cs.synchronize() catch {};
            cs.deinit();
            for (&e.f.cs_ev) |*ev| ev.deinit();
        }
        e.f.deinit();
        e.pbuf.deinit();
        if (e.mbuf) |*m| m.deinit();
        e.buf.deinit();
        e.w.deinit();
        e.set.deinit();
        e.torch.deinit();
        e.k.deinit();
        e.stream.deinit();
        if (e.vm) |*v| v.deinit();
        e.c.deinit();
        e.gpa.destroy(e);
    }

    // -- sequences --------------------------------------------------------------------------------------------

    /// A new empty sequence whose caches grow up to `limit_rows` (at most the engine's max_len).
    pub fn newSeq(e: *Engine, limit_rows: usize) !*Seq {
        const limit = @min(limit_rows, e.max_len);
        const s = try e.gpa.create(Seq);
        errdefer e.gpa.destroy(s);
        // caches start at one growth step and grow with the stream (State.ensure), up to `limit`
        const first = @min(limit, state.grow_step);
        if (!e.fits(first)) return error.NoRoom;
        s.* = .{ .st = try state.State.init(e.ctx.d, e.g, if (e.vm != null) 1 else first, limit) };
        errdefer s.deinit(e.gpa);
        s.charged = s.st.fixed.buf.len + (if (s.st.ctx) |c| c.buf.len else 0) + e.g.wide() * 2;
        e.budget.used += s.charged;
        errdefer e.budget.used -|= s.charged;
        if (e.vm) |*v| try e.mapSeq(s, v, first);
        try e.f.register(s);
        errdefer e.f.unregister(s);
        s.last_streams = try cuda.DeviceBuffer.alloc(e.ctx.d, e.g.wide() * 2);
        try e.f.reset(s);
        return s;
    }

    /// A sequence's caches in reserved address space: State's own (one row) given back, then per cache layer the
    /// keys', values' and pooled blocks' ranges for the whole window, mapped up to `first` rows, and the indexer
    /// keys' range over one ring chunk.
    fn mapSeq(e: *Engine, s: *Seq, v: *const vmm.Vmm, first: usize) !void {
        const g = e.g;
        const st = &s.st;
        if (st.ctx) |*c| {
            s.charged -= c.buf.len;
            e.budget.used -= c.buf.len;
            c.buf.free();
            st.ctx = null;
        }
        const kv_row = g.kv_heads * g.head_dim * 2;
        const ikc_row = g.index_dim * 2;
        const chunk = v.up(@max(ring_min_rows * ikc_row, v.granularity));
        const layers = g.attention_layers + @intFromBool(g.mtp);
        const limit = st.limit;
        for (0..layers) |i| {
            const sizes = [4]usize{ limit * kv_row, limit * kv_row, std.mem.alignForward(usize, limit * ikc_row, chunk), g.blocks(limit) * ikc_row };
            for (sizes) |b| {
                const r = try vmm.Region.reserve(v, b);
                try s.regions.append(e.gpa, r);
            }
            const r = s.regions.items[4 * i ..][0..4];
            const ring = try r[2].mapRing(e.gpa, chunk);
            try e.f.th.zero(r[2].base, ring);
            s.charged += ring;
            e.budget.used += ring;
            st.kc[i] = r[0].base;
            st.vc[i] = r[1].base;
            st.ikc[i] = r[2].base;
            st.pooled[i] = r[3].base;
        }
        s.ring_rows = chunk / ikc_row;
        st.capacity = 0;
        try e.f.grow(s, first);
    }

    pub fn freeSeq(e: *Engine, s: *Seq) void {
        if (e.bound == s and s != e.own) e.bound = e.own;
        e.budget.used -|= s.charged;
        e.f.unregister(s);
        e.dropGraphs(@intFromPtr(s), null);
        e.dropMulti(@intFromPtr(s));
        s.deinit(e.gpa);
        e.gpa.destroy(s);
    }

    /// Bytes a new sequence holding `rows` context rows would take (its fixed state and its caches).
    pub fn seqBytes(e: *const Engine, rows: usize) usize {
        const g = e.g;
        const n = g.linear_layers;
        const fixed = n * (g.conv_kernel - 1) * g.gdnConv() * 2 + 2 * n * g.nv * state.gdn_dv * state.gdn_dk * 4 + n * state.max_rows * (g.nk * state.gdn_dk * 4 + g.nv * state.gdn_dv * 2 + 2 * g.nv * 4) + (g.ple_kernel - 1) * g.ngram_size * g.wide() * 2;
        const cap = @min(e.max_len, std.mem.alignForward(usize, @max(rows, 1), state.grow_step));
        return fixed + state.State.cacheBytes(g, cap) + 96 * 256 + g.wide() * 2;
    }

    /// Whether a new sequence that will hold `rows` rows fits beside the live ones now (admission).
    pub fn fits(e: *const Engine, rows: usize) bool {
        return e.budget.used + e.seqBytes(rows) <= e.budget.limit;
    }

    pub fn bind(e: *Engine, s: *Seq) void {
        e.bound = s;
    }

    /// The sampling rule `sample`, `sampleDraft` and `drawDraft` apply (null: greedy).
    pub fn setSampling(e: *Engine, s: ?lanes.Sampling) void {
        e.sampling = if (s) |x| (if (x.temperature > 0) x else null) else null;
    }

    pub fn ops(e: *Engine) *fwd.Forward {
        return &e.f;
    }

    // -- the prompt -------------------------------------------------------------------------------------------

    /// decode.prefill (fresh, no kept states): the prompt's chunks committed in order, each one's next tokens
    /// absorbed by the MTP head (`absorbing`), then the first draw at the prompt's length.
    pub fn prefill(e: *Engine, prompt: []const u32, sampling: ?lanes.Sampling) !u32 {
        return (try e.prefillWith(prompt, sampling, .{})).first;
    }

    /// A kept prompt state (Python State.snapshot with its MTP tail): the DeltaNet states, conv windows, n-gram
    /// tail and history at `pos`, the MTP head's length one short of it and the streams of row pos - 1, so a prompt
    /// that extends tokens[0..pos] resumes there (its caches' first pos rows must still be in the sequence: the
    /// one it was kept from, or one copyPrefix filled). The rows past pos may have been overwritten since.
    pub const Snapshot = struct {
        pos: usize,
        mtp_len: usize,
        history: [3]i64,
        tokens: []u32,
        mem: cuda.DeviceBuffer,
        rec: u64,
        conv: u64,
        ple_tail: u64,
        tail: u64,
        /// each cache layer's indexer keys of its incomplete block (a ring may overwrite them before a resume)
        ikc: u64,
    };

    pub const PrefillOptions = struct {
        /// keep the state at this many tokens (0 < keep_at <= prompt.len; Python entry_end is len - 1)
        keep_at: ?usize = null,
        /// resume from a kept state of this sequence (the prompt extends its tokens)
        resume_from: ?*const Snapshot = null,
    };

    pub const Prefilled = struct { first: u32, kept: ?*Snapshot = null };

    fn snapshotBytes(e: *const Engine) [5]usize {
        const g = e.g;
        return .{ g.linear_layers * g.nv * state.gdn_dv * state.gdn_dk * 4, g.linear_layers * (g.conv_kernel - 1) * g.gdnConv() * 2, (g.ple_kernel - 1) * g.ngram_size * g.wide() * 2, g.wide() * 2, (g.attention_layers + 1) * g.index_ratio * g.index_dim * 2 };
    }

    /// Cache layer i's incomplete-block rows [first, end) at the sequence's committed rows (pos, or the MTP
    /// head's `mtp_len` for its layer).
    fn ikcTail(e: *const Engine, i: usize, pos_rows: usize, mtp_rows: usize) [2]usize {
        const end = if (e.g.mtp and i == e.g.attention_layers) mtp_rows else pos_rows;
        return .{ end / e.g.index_ratio * e.g.index_ratio, end };
    }

    /// The bound sequence's committed state as a Snapshot (State.snapshot); `mtp_len` and the tail row given.
    fn snapshot(e: *Engine, seq: *Seq, mtp_len: usize, tail: u64) !*Snapshot {
        const st = &seq.st;
        try e.f.flush(seq); // the kept rows folded in: the state is the committed one
        const sz = e.snapshotBytes();
        const total = aligned(&sz);
        if (e.budget.used + total > e.budget.limit) return error.NoRoom;
        const snap = try e.gpa.create(Snapshot);
        errdefer e.gpa.destroy(snap);
        var a: state.Arena = .{ .buf = try cuda.DeviceBuffer.alloc(e.ctx.d, total) };
        errdefer a.buf.free();
        snap.* = .{ .pos = st.pos, .mtp_len = mtp_len, .history = st.history, .tokens = &.{}, .mem = a.buf, .rec = a.take(sz[0]), .conv = a.take(sz[1]), .ple_tail = a.take(sz[2]), .tail = a.take(sz[3]), .ikc = a.take(sz[4]) };
        const row = e.g.index_dim * 2;
        for (0..e.g.attention_layers + @intFromBool(e.g.mtp)) |i| {
            const t = e.ikcTail(i, st.pos, mtp_len);
            try e.f.th.copy(snap.ikc + i * e.g.index_ratio * row, st.ikc[i] + t[0] * row, (t[1] - t[0]) * row);
        }
        try e.f.th.copy(snap.rec, st.rec, sz[0]);
        try e.f.th.copy(snap.conv, st.conv, sz[1]);
        try e.f.th.copy(snap.ple_tail, st.ple_tail, sz[2]);
        try e.f.th.copy(snap.tail, tail, sz[3]);
        e.budget.used += total;
        return snap;
    }

    pub fn freeSnapshot(e: *Engine, snap: *Snapshot) void {
        e.stream.synchronize() catch {};
        e.budget.used -|= snap.mem.len;
        snap.mem.free();
        e.gpa.free(snap.tokens);
        e.gpa.destroy(snap);
    }

    /// State.restore: the snapshot's state into `seq` (its caches' first snap.pos rows must hold the prompt's).
    fn restore(e: *Engine, seq: *Seq, snap: *const Snapshot) !void {
        const st = &seq.st;
        const sz = e.snapshotBytes();
        if (snap.pos > st.capacity) return error.CachesTooShort;
        try e.f.th.copy(st.rec, snap.rec, sz[0]);
        st.cur = @splat(0);
        seq.held = 0;
        try e.f.th.copy(st.conv, snap.conv, sz[1]);
        try e.f.th.copy(st.ple_tail, snap.ple_tail, sz[2]);
        const row = e.g.index_dim * 2;
        for (0..e.g.attention_layers + @intFromBool(e.g.mtp)) |i| {
            const t = e.ikcTail(i, snap.pos, snap.mtp_len);
            try e.f.th.copy(st.ikc[i] + t[0] * row, snap.ikc + i * e.g.index_ratio * row, (t[1] - t[0]) * row);
        }
        st.history = snap.history;
        seq.staged = false;
        try e.f.setPos(st, snap.pos);
        st.rope_delta = 0;
        try e.f.th.fill32(st.rope_delta_dev, 0, 1);
        st.mtp_drafted = 0;
        try e.f.setMtpLen(st, snap.mtp_len);
    }

    /// State.copy_prefix: `src`'s first `pos` cache rows (and complete indexer blocks) and its MTP head's first
    /// `mtp_len` into `dst` (grown to hold them), so `dst` can resume from a snapshot kept on `src`.
    pub fn copyPrefix(e: *Engine, dst: *Seq, src: *Seq, rows: usize, mtp_rows: usize) !void {
        if (dst == src) return error.SameSequence;
        if (rows > src.st.pos or mtp_rows > src.st.mtp_len) return error.PrefixPastSource;
        try e.f.grow(dst, @max(rows, mtp_rows));
        const g = e.g;
        const kv_row = g.kv_heads * g.head_dim * 2;
        for (0..g.attention_layers + @intFromBool(g.mtp)) |i| {
            const n = if (g.mtp and i == g.attention_layers) mtp_rows else rows;
            try e.f.th.copy(dst.st.kc[i], src.st.kc[i], n * kv_row);
            try e.f.th.copy(dst.st.vc[i], src.st.vc[i], n * kv_row);
            // the indexer keys a later pool reads: the incomplete block's rows (a ring holds no older ones)
            const first_ikc = if (dst.ring_rows > 0 or src.ring_rows > 0) n / g.index_ratio * g.index_ratio else 0;
            try e.f.th.copy(dst.st.ikc[i] + first_ikc * g.index_dim * 2, src.st.ikc[i] + first_ikc * g.index_dim * 2, (n - first_ikc) * g.index_dim * 2);
            try e.f.th.copy(dst.st.pooled[i], src.st.pooled[i], (n / g.index_ratio) * g.index_dim * 2);
        }
    }

    /// decode.prefill with kept states: the prompt's chunks (one ends at `keep_at`, whose state is kept; prompt
    /// chunks are chunk-invariant, so a chunk ending there leaves the bits of a cut inside one), or a resume from a
    /// kept state of this sequence (its MTP tail absorbed with the prompt's next token first). The first draw.
    pub fn prefillWith(e: *Engine, prompt: []const u32, sampling: ?lanes.Sampling, o: PrefillOptions) !Prefilled {
        if (prompt.len == 0) return error.EmptyPrompt;
        if (e.pfill != null) return error.PromptBuffersBusy; // a sliced fill holds the prompt buffers mid-chunk
        if (prompt.len + e.depth + 1 > e.bound.st.limit) return error.ContextPastWindow;
        if (o.keep_at) |k| if (k == 0 or k > prompt.len) return error.BadKeepPoint;
        e.setSampling(sampling);
        const seq = e.bound;
        const Wd = e.g.wide();
        const pb = &e.pbuf;
        const use_mtp = e.absorbing and e.w.mtp != null;
        var start: usize = 0;
        if (o.resume_from) |snap| {
            if (!(snap.pos > 0 and snap.pos < prompt.len)) return error.ResumeMustExtend;
            if (!std.mem.eql(u32, snap.tokens, prompt[0..snap.pos])) return error.ResumeOtherPrompt;
            try e.restore(seq, snap);
            if (use_mtp) {
                _ = try mtp.forward(&e.f, seq, pb, prompt[snap.pos .. snap.pos + 1], snap.tail);
                try e.f.setMtpLen(&seq.st, seq.st.mtp_len + 1);
            }
            start = snap.pos;
        } else try e.f.reset(seq);
        var kept: ?*Snapshot = null;
        errdefer if (kept) |k| e.freeSnapshot(k);
        var chunk: usize = 0;
        while (start < prompt.len) : (chunk += 1) {
            var end = @min(start + e.prefill_rows, prompt.len);
            if (o.keep_at) |k| if (start < k and k < end) {
                end = k;
            };
            const R = end - start;
            const final = end == prompt.len;
            if (e.f.dump) |dm| dm.setChunk(chunk);
            const t_chunk = std.Io.Clock.awake.now(e.io);
            if (e.prof) |p| {
                // prompt chunks only (decode windows may be graph captures)
                try e.stream.synchronize();
                e.f.prof = p;
                p.mtp = false;
                try p.mark(e.stream, .start);
            }
            defer e.f.prof = null;
            const lg = try e.f.forward(seq, pb, prompt[start..end], final);
            if (final) {
                try e.f.th.copy(e.last_logits, lg.?, e.g.head_n * 2);
                if (e.world > 1) try e.f.th.copy(e.last_cand, pb.b.cand_all, e.world * (2 * state.cand + 1) * 4);
            }
            try e.f.th.copy(seq.last_streams.?.ptr, pb.b.streams + (R - 1) * Wd * 2, Wd * 2);
            if (e.digest) try e.chunkDigest(chunk, R, if (final) lg.? else null);
            const nxt = prompt[start + 1 .. @min(end + 1, prompt.len)];
            if (use_mtp and nxt.len > 0) {
                if (e.prof) |p| {
                    try p.mark(e.stream, .other);
                    p.mtp = true;
                }
                _ = try mtp.forward(&e.f, seq, pb, nxt, pb.b.streams);
                try e.f.setMtpLen(&seq.st, seq.st.mtp_len + nxt.len);
                if (e.prof) |p| {
                    try p.mark(e.stream, .other);
                    p.mtp = false;
                }
            }
            try e.f.commit(seq, pb, R, R);
            if (o.keep_at) |k| if (end == k and kept == null) {
                // the MTP head's length one short: a resume absorbs the next prompt's own token at row k - 1
                const snap = try e.snapshot(seq, if (use_mtp and !final) seq.st.mtp_len - 1 else seq.st.mtp_len, seq.last_streams.?.ptr);
                kept = snap;
                snap.tokens = try e.gpa.dupe(u32, prompt[0..k]);
            };
            if (e.prof) |p| {
                try p.mark(e.stream, .commit);
                try e.stream.synchronize();
                try p.chunk(e.rank, R, seconds(e.io, t_chunk) * 1e3);
            }
            start = end;
        }
        if (e.prof) |p| p.summary(e.rank, "prefill");
        if (e.f.dump) |dm| dm.setChunk(null);
        var first: [1]u32 = undefined;
        try e.draw(e.last_logits, e.last_cand, 1, prompt.len, &first);
        return .{ .first = first[0], .kept = kept };
    }

    /// One prompt of a burst (prefillMany).
    pub const ManyPrompt = struct { seq: *Seq, prompt: []const u32, sampling: ?lanes.Sampling };

    /// Whether `prefillMany` can take these prompts as one pass: two to `state.max_rows` prompts whose rows fit the
    /// prompt buffers together (each a whole prompt, from reset).
    pub fn manyFit(e: *const Engine, prompts: []const ManyPrompt) bool {
        if (prompts.len < 2 or prompts.len > state.max_rows or e.pfill != null) return false;
        var rows: usize = 0;
        for (prompts) |p| {
            if (p.prompt.len == 0 or p.prompt.len + e.depth + 1 > p.seq.st.limit) return false;
            rows += p.prompt.len;
        }
        return rows <= e.prefill_rows;
    }

    /// Several prompts in ONE prompt pass (a burst): their rows stage one after another, every row-wise step runs once
    /// over all of them, each prompt's attention, DeltaNet chain, n-gram tail and MTP cache run over its own rows from
    /// its own state (the prompt chunks are chunk-invariant), so each prompt's states and its first token equal its
    /// own prefillWith's. `firsts[i]` is prompt i's draw at its length. No prompt-cache marks or resumes here.
    pub fn prefillMany(e: *Engine, prompts: []const ManyPrompt, firsts: []u32) !void {
        if (!e.manyFit(prompts)) return error.BadBurst;
        if (firsts.len < prompts.len) return error.BadBurst;
        const n = prompts.len;
        const pb = &e.pbuf;
        const Wd = e.g.wide();
        const use_mtp = e.absorbing and e.w.mtp != null;
        if (e.rank == 0 and envGet("TF_FLASHNEXT_BURST_LOG") != null) {
            var rows: usize = 0;
            for (prompts) |p| rows += p.prompt.len;
            std.debug.print("burst prefill: {d} prompts, {d} rows in one pass\n", .{ n, rows });
        }
        var windows: [state.max_rows]fwd.Window = undefined;
        var segs: [state.max_rows]fwd.Seg = undefined;
        for (prompts, 0..) |p, i| {
            try e.f.reset(p.seq);
            windows[i] = .{ .seq = p.seq, .tokens = p.prompt };
        }
        _ = try e.f.stageMany(windows[0..n], pb, &segs);
        const lg = (try e.f.computeSegs(segs[0..n], pb, true)).?;
        for (prompts, segs[0..n]) |p, sg| try e.f.th.copy(p.seq.last_streams.?.ptr, pb.b.streams + (sg.a1 - 1) * Wd * 2, Wd * 2);
        if (use_mtp) {
            // each prompt's MTP rows: its next tokens over the main model's streams of its rows (a prompt of one row has none)
            var inputs: [state.max_rows]mtp.Input = undefined;
            var m: usize = 0;
            for (prompts, segs[0..n]) |p, sg| {
                if (p.prompt.len < 2) continue;
                inputs[m] = .{ .seq = p.seq, .next = p.prompt[1..], .streams = pb.b.streams + sg.a0 * Wd * 2 };
                m += 1;
            }
            if (m > 0) {
                var msegs: [state.max_rows]fwd.Seg = undefined;
                _ = try mtp.stageMany(&e.f, inputs[0..m], pb, &msegs);
                _ = try mtp.computeSegs(&e.f, msegs[0..m], pb);
                for (inputs[0..m]) |in| try e.f.setMtpLen(&in.seq.st, in.seq.st.mtp_len + in.next.len);
            }
        }
        for (prompts, segs[0..n]) |p, sg| try e.f.commitAt(p.seq, pb, sg.a1 - sg.a0, sg.a1 - sg.a0, sg.a0);
        for (prompts, 0..) |p, i| {
            const s: ?lanes.Sampling = if (p.sampling) |x| (if (x.temperature > 0) x else null) else null;
            try e.drawRows(lg, pb.b.cand_all, n, i, 1, p.prompt.len, s, firsts[i .. i + 1]);
        }
    }

    /// A prompt filling in slices between decode rounds (fillBegin, fillStep): the chunk in progress, its next
    /// layer and the pending write-back it carries (all in the prompt buffers, which rounds never touch).
    pub const Fill = struct {
        seq: *Seq,
        prompt: []u32,
        sampling: ?lanes.Sampling,
        keep_at: ?usize,
        use_mtp: bool,
        start: usize,
        end: usize = 0,
        chunk: usize = 0,
        layer: usize = 0,
        open: bool = false,
        pending: ?fwd.Pending = null,
        kept: ?*Snapshot = null,
        done: bool = false,
        first: u32 = 0,
    };

    /// prefillWith's start (reset, or restore and the MTP tail absorbed) for a prompt that then fills in slices.
    pub fn fillBegin(e: *Engine, seq: *Seq, prompt: []const u32, sampling: ?lanes.Sampling, o: PrefillOptions) !*Fill {
        if (prompt.len == 0) return error.EmptyPrompt;
        if (prompt.len + e.depth + 1 > seq.st.limit) return error.ContextPastWindow;
        if (o.keep_at) |k| if (k == 0 or k > prompt.len) return error.BadKeepPoint;
        const fl = try e.gpa.create(Fill);
        errdefer e.gpa.destroy(fl);
        fl.* = .{ .seq = seq, .prompt = try e.gpa.dupe(u32, prompt), .sampling = if (sampling) |x| (if (x.temperature > 0) x else null) else null, .keep_at = o.keep_at, .use_mtp = e.absorbing and e.w.mtp != null, .start = 0 };
        errdefer e.gpa.free(fl.prompt);
        if (o.resume_from) |snap| {
            if (!(snap.pos > 0 and snap.pos < prompt.len)) return error.ResumeMustExtend;
            if (!std.mem.eql(u32, snap.tokens, prompt[0..snap.pos])) return error.ResumeOtherPrompt;
            try e.restore(seq, snap);
            if (fl.use_mtp) {
                _ = try mtp.forward(&e.f, seq, &e.pbuf, prompt[snap.pos .. snap.pos + 1], snap.tail);
                try e.f.setMtpLen(&seq.st, seq.st.mtp_len + 1);
            }
            fl.start = snap.pos;
        } else try e.f.reset(seq);
        return fl;
    }

    pub fn fillFree(e: *Engine, fl: *Fill) void {
        if (e.pfill == fl) e.pfill = null;
        if (fl.kept) |k| e.freeSnapshot(k);
        e.gpa.free(fl.prompt);
        e.gpa.destroy(fl);
    }

    /// Up to `layers` more layer passes of the filling prompt (a chunk's embedding and finish ride with its first
    /// and last layer); true once the prompt is committed and `first` holds its draw. The same calls in the same
    /// order as prefillWith, so the same bits, with decode rounds of other sequences run between the slices.
    pub fn fillStep(e: *Engine, fl: *Fill, layers: usize) !bool {
        if (fl.done) return true;
        const pb = &e.pbuf;
        const seq = fl.seq;
        const Wd = e.g.wide();
        const n_layers = e.w.layers.len;
        var budget = @max(layers, 1);
        while (budget > 0) {
            if (!fl.open) {
                if (e.pfill != null and e.pfill.? != fl) return error.PromptBuffersBusy;
                e.pfill = fl;
                var end = @min(fl.start + e.prefill_rows, fl.prompt.len);
                if (fl.keep_at) |k| if (fl.start < k and k < end) {
                    end = k;
                };
                fl.end = end;
                try e.f.stage(seq, pb, fl.prompt[fl.start..end]);
                try e.f.t.embed(pb.b.ids, e.w.embed, pb.b.h, end - fl.start, e.g.hidden, e.g.streams);
                fl.layer = 0;
                fl.pending = null;
                fl.open = true;
            }
            const R = fl.end - fl.start;
            const segs = [1]fwd.Seg{.{ .seq = seq, .a0 = 0, .a1 = R }};
            if (fl.layer < n_layers) {
                fl.pending = try e.f.layerForward(&e.w.layers[fl.layer], &segs, pb, R, fl.pending, false);
                fl.layer += 1;
                budget -= 1;
                continue;
            }
            // the chunk's finish, MTP absorb and commit (as prefillWith's loop body)
            const final = fl.end == fl.prompt.len;
            const lg = try e.f.finish(&e.w.mixer, pb, R, fl.pending.?, final);
            if (final) {
                try e.f.th.copy(e.last_logits, lg.?, e.g.head_n * 2);
                if (e.world > 1) try e.f.th.copy(e.last_cand, pb.b.cand_all, e.world * (2 * state.cand + 1) * 4);
            }
            try e.f.th.copy(seq.last_streams.?.ptr, pb.b.streams + (R - 1) * Wd * 2, Wd * 2);
            const nxt = fl.prompt[fl.start + 1 .. @min(fl.end + 1, fl.prompt.len)];
            if (fl.use_mtp and nxt.len > 0) {
                _ = try mtp.forward(&e.f, seq, pb, nxt, pb.b.streams);
                try e.f.setMtpLen(&seq.st, seq.st.mtp_len + nxt.len);
            }
            try e.f.commit(seq, pb, R, R);
            if (fl.keep_at) |k| if (fl.end == k and fl.kept == null) {
                const snap = try e.snapshot(seq, if (fl.use_mtp and !final) seq.st.mtp_len - 1 else seq.st.mtp_len, seq.last_streams.?.ptr);
                fl.kept = snap;
                snap.tokens = try e.gpa.dupe(u32, fl.prompt[0..k]);
            };
            fl.start = fl.end;
            fl.open = false;
            e.pfill = null;
            fl.chunk += 1;
            if (fl.start == fl.prompt.len) {
                var first: [1]u32 = undefined;
                try e.drawRows(e.last_logits, e.last_cand, 1, 0, 1, fl.prompt.len, fl.sampling, &first);
                fl.first = first[0];
                fl.done = true;
                return true;
            }
            budget -= 1;
        }
        return false;
    }

    /// TF_FLASHNEXT_PREFILL_DIGEST: sha256 of a chunk's final streams (all R rows) and of its last logits.
    fn chunkDigest(e: *Engine, chunk: usize, R: usize, logits: ?u64) !void {
        const Wd = e.g.wide();
        const n = R * Wd * 2;
        const host = try e.gpa.alloc(u8, @max(n, e.g.head_n * 2));
        defer e.gpa.free(host);
        try e.read(e.pbuf.b.streams, host[0..n]);
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(host[0..n], &d, .{});
        std.debug.print("digest rank {d} chunk {d}: {d} rows, streams {s}", .{ e.rank, chunk, R, std.fmt.bytesToHex(d, .lower)[0..16] });
        if (logits) |lg| {
            try e.read(lg, host[0 .. e.g.head_n * 2]);
            std.crypto.hash.sha2.Sha256.hash(host[0 .. e.g.head_n * 2], &d, .{});
            std.debug.print(" logits {s}", .{std.fmt.bytesToHex(d, .lower)[0..16]});
        }
        std.debug.print("\n", .{});
    }

    /// The prompt's last row's streams (device, [1, S*D] bf16), what the first MTP absorb reads.
    pub fn lastStreams(e: *Engine) u64 {
        return e.bound.last_streams.?.ptr;
    }

    // -- draws ------------------------------------------------------------------------------------------------

    fn read(e: *Engine, ptr: u64, host: []u8) !void {
        try e.stream.synchronize();
        const buf: cuda.DeviceBuffer = .{ .d = e.ctx.d, .ptr = ptr, .len = host.len };
        try buf.download(0, host);
    }

    /// Engine.sample over `rows` rows of the main head's `logits` (one rank) or gathered candidates `cand_all`
    /// (two ranks), keyed at first_position + r.
    fn draw(e: *Engine, logits: u64, cand_all: u64, rows: usize, first_position: u64, out: []u32) !void {
        return e.drawRows(logits, cand_all, rows, 0, rows, first_position, e.sampling, out);
    }

    /// The draws of rows [row0, row0 + rows) of a forward's `total` rows (one stream's window of a shared round)
    /// with that stream's sampling.
    /// choose_gathered over rows [row0, row0 + rows) of the host copy `all` [world][total][2 CAND + 1].
    fn chooseFrom(e: *Engine, all: []const f32, total: usize, row0: usize, rows: usize, first_position: u64, s: ?lanes.Sampling, out: []u32) !void {
        const width = 2 * state.cand + 1;
        // this window's rows of every rank's block
        const got = try e.gpa.alloc(f32, e.world * rows * width);
        defer e.gpa.free(got);
        for (0..e.world) |r| @memcpy(got[r * rows * width ..][0 .. rows * width], all[(r * total + row0) * width ..][0 .. rows * width]);
        var positions: [state.max_rows]u64 = undefined;
        var draws: [state.max_rows]sampler.Draw = undefined;
        for (0..rows) |r| positions[r] = first_position + r;
        try sampler.chooseGathered(e.gpa, got, e.world, rows, positions[0..rows], s, false, draws[0..rows]);
        for (out[0..rows], draws[0..rows]) |*o, dr| o.* = @intCast(dr.token);
    }

    fn drawRows(e: *Engine, logits_base: u64, cand_all: u64, total: usize, row0: usize, rows: usize, first_position: u64, s: ?lanes.Sampling, out: []u32) !void {
        const V = e.g.head_n;
        if (rows > state.max_rows) return error.WindowPastRows;
        const logits = logits_base + row0 * V * 2;
        if (e.world > 1) {
            if (!sampler.gatheredFits(s)) {
                if (s.?.top_k == 0) return e.nucleusRows(logits, rows, first_position, s.?, out);
                return mtp.tpSampleRows(&e.f, e.tp.?, logits, rows, V, null, first_position, s.?, out, null);
            }
            const all = try e.gpa.alloc(f32, e.world * total * (2 * state.cand + 1));
            defer e.gpa.free(all);
            try e.read(cand_all, std.mem.sliceAsBytes(all));
            return e.chooseFrom(all, total, row0, rows, first_position, s, out);
        }
        const ms = e.main_sample;
        if (s == null) {
            try e.f.th.argmax(logits, V, V, ms.col, rows);
            var ids: [state.max_rows]i32 = undefined;
            try e.read(ms.col, std.mem.sliceAsBytes(ids[0..rows]));
            for (out[0..rows], ids[0..rows]) |*o, x| o.* = @intCast(x);
            return;
        }
        if (s.?.top_k == 0) return e.nucleusRows(logits, rows, first_position, s.?, out);
        const k = mtp.topK(V, s.?);
        if (k > max_k) return error.TopKPastScratch;
        try e.f.th.sampleTopk(logits, rows, V, k, ms);
        const vals = try e.gpa.alloc(f32, rows * k);
        defer e.gpa.free(vals);
        const cols = try e.gpa.alloc(i64, rows * k);
        defer e.gpa.free(cols);
        try e.read(ms.vals, std.mem.sliceAsBytes(vals));
        try e.read(ms.idx, std.mem.sliceAsBytes(cols));
        const ids = try e.gpa.alloc(u64, k);
        defer e.gpa.free(ids);
        for (0..rows) |r| {
            for (ids, cols[r * k ..][0..k]) |*t, c| t.* = @intCast(c);
            const got = try sampler.drawRow(e.gpa, vals[r * k ..][0..k], ids, first_position + r, s, null);
            out[r] = @intCast(got.token);
        }
    }

    /// sampling.nucleus_rows (top_k off) over the main head's rows (this rank's vocabulary shard).
    fn nucleusRows(e: *Engine, logits: u64, rows: usize, first_position: u64, s: lanes.Sampling, out: []u32) !void {
        var positions: [state.max_rows]u64 = undefined;
        for (0..rows) |r| positions[r] = first_position + r;
        try nucleus.sampleRows(&e.f, e.nsc, logits, rows, e.g.head_n, null, e.f.vocab_offset, positions[0..rows], s, out[0..rows], null);
    }

    // -- shared rounds: several streams' windows in one forward (Python forward's segs; MultiDecoder.round) --

    pub const Window = fwd.Window;

    /// One forward over every stream's window, each at its own position on its own caches; the dense layers,
    /// hyper-connections and experts once over all rows (each row's bits as alone), the DeltaNet chain, attention
    /// and n-gram tail per stream. Then sampleWindow / commitWindow each window, and draftMany. One window runs
    /// as `forward` does (graphs).
    pub fn verifyMany(e: *Engine, windows: []const Window) !void {
        if (windows.len == 0) return error.EmptyRound;
        try e.f.mark(.gap_verify);
        e.cand_round +%= 1;
        if (windows.len > e.round_segs.len) return error.TooManyStreams;
        if (windows.len == 1) {
            const keep = e.bound;
            e.bound = windows[0].seq;
            defer e.bound = keep;
            try e.forward(windows[0].tokens);
            e.round_segs[0] = .{ .seq = windows[0].seq, .a0 = 0, .a1 = windows[0].tokens.len };
            e.round_n = 1;
            return;
        }
        const R = try e.f.stageMany(windows, &e.buf, e.round_segs);
        e.round_n = windows.len;
        try e.runMulti(false, e.round_segs[0..windows.len], &e.buf);
        e.rows = R;
    }

    /// The target's draws of window `i`'s rows, keyed at its seq.pos + 1 + r with sampling `s` (its stream's).
    pub fn sampleWindow(e: *Engine, i: usize, s: ?lanes.Sampling, out: []u32) !void {
        if (i >= e.round_n) return error.NoSuchWindow;
        const sg = e.round_segs[i];
        const smp: ?lanes.Sampling = if (s) |x| (if (x.temperature > 0) x else null) else null;
        const total = e.round_segs[e.round_n - 1].a1;
        if (e.world > 1 and e.draw_once and sampler.gatheredFits(smp)) {
            // the round's candidates read once (one wait), every window's draws from the host copy
            if (e.cand_have != e.cand_round) {
                try e.cand_host.resize(e.gpa, e.world * total * (2 * state.cand + 1));
                try e.read(e.buf.b.cand_all, std.mem.sliceAsBytes(e.cand_host.items));
                e.cand_have = e.cand_round;
            }
            return e.chooseFrom(e.cand_host.items, total, sg.a0, sg.a1 - sg.a0, sg.seq.st.pos + 1, smp, out);
        }
        try e.drawRows(e.buf.b.logits, e.buf.b.cand_all, total, sg.a0, sg.a1 - sg.a0, sg.seq.st.pos + 1, smp, out);
    }

    /// Keep window `i`'s first `keep` rows (its stream's commit, at its rows of the shared window).
    pub fn commitWindow(e: *Engine, i: usize, keep: usize) !void {
        if (i >= e.round_n) return error.NoSuchWindow;
        const sg = e.round_segs[i];
        try e.f.commitAt(sg.seq, &e.buf, sg.a1 - sg.a0, keep, sg.a0);
    }

    /// Where a stream's absorb reads the main model's final streams: its prompt's last row, or its window's
    /// first rows in the last shared round.
    pub const DraftSource = union(enum) { prefill_last, window: usize };

    /// One stream's MTP drafting: absorb `next` (the kept tokens) with their streams, then chain up to `count`
    /// drafts with `sampling` and `confidence` into `out`; `drafted` is set to how many (Python _draft_all).
    pub const DraftReq = struct {
        seq: *Seq,
        source: DraftSource,
        next: []const u32,
        count: usize,
        sampling: ?lanes.Sampling = null,
        confidence: f64 = default_confidence,
        out: []u32,
        drafted: usize = 0,
    };

    /// MultiDecoder._draft_all: every stream absorbs its kept rows in one MTP step, then a step a depth over the
    /// streams still drafting (the first draft always kept, a later one only at or above the confidence), each
    /// draw with its stream's rule at its own position. Each stream's drafts equal its solo chain's.
    pub fn draftMany(e: *Engine, reqs: []DraftReq) !void {
        const mb = &(e.mbuf orelse return error.NoMtpHead);
        const d = &(e.draws orelse return error.NoMtpHead);
        if (reqs.len == 0) return;
        // profiling: the host gap before the head, then its time under the MTP head by draft level
        try e.f.mark(.gap_draft);
        if (e.f.prof) |p| {
            p.mtp = true;
            p.level = 0;
        }
        defer if (e.f.prof) |p| {
            p.mtp = false;
            p.level = 0;
        };
        if (reqs.len > e.round_segs.len) return error.TooManyStreams;
        const Wd = e.g.wide();
        var inputs: [64]mtp.Input = undefined;
        var segs: [64]fwd.Seg = undefined;
        var active: [64]struct { req: usize, row: usize } = undefined;
        if (reqs.len > inputs.len) return error.TooManyStreams;
        for (reqs, 0..) |*r, i| {
            r.drafted = 0;
            const st = &r.seq.st;
            if (st.mtp_drafted != 0) {
                try e.f.setMtpLen(st, st.mtp_len - st.mtp_drafted);
                st.mtp_drafted = 0;
            }
            const streams = switch (r.source) {
                .prefill_last => r.seq.last_streams.?.ptr,
                .window => |w| blk: {
                    if (w >= e.round_n or e.round_segs[w].seq != r.seq) return error.NoSuchWindow;
                    break :blk e.buf.b.streams + e.round_segs[w].a0 * Wd * 2;
                },
            };
            inputs[i] = .{ .seq = r.seq, .next = r.next, .streams = streams };
        }
        _ = try mtp.stageMany(&e.f, inputs[0..reqs.len], mb, &segs);
        try e.runMulti(true, segs[0..reqs.len], mb);
        const logits = mb.b.logits;
        var n_active: usize = 0;
        for (reqs, 0..) |*r, i| {
            try e.f.setMtpLen(&r.seq.st, r.seq.st.mtp_len + r.next.len);
            const room = @min(r.count, r.out.len);
            if (room > 0) {
                active[n_active] = .{ .req = i, .row = segs[i].a1 - 1 };
                n_active += 1;
            }
        }
        // each stream's stop rule (cuda_decode.Stop: per draft, running product or none), as its solo chain's
        var stops: [64]decode.Stop = undefined;
        for (reqs, 0..) |r, i| stops[i] = .{ .c = r.confidence };
        var lg = logits;
        var rows = reqs.len; // the head's rows of the step just run, active[k] at row k
        var row_of: [64]usize = undefined; // active k's row in that step
        for (0..n_active) |k| row_of[k] = active[k].req;
        var j: usize = 0;
        while (n_active > 0) : (j += 1) {
            var next: [64]struct { req: usize, row: usize, tok: u32 } = undefined;
            var n_next: usize = 0;
            // two ranks: the step's gathered candidates read once (one wait) for every stream's draw
            var cands: ?[]f32 = null;
            defer if (cands) |c| e.gpa.free(c);
            if (e.world > 1 and e.draw_once) {
                const c = try e.gpa.alloc(f32, e.world * rows * (2 * state.cand + 1));
                cands = c;
                try e.read(mb.b.cand_all, std.mem.sliceAsBytes(c));
            }
            for (active[0..n_active], 0..) |a, k| {
                const r = &reqs[a.req];
                const smp: ?lanes.Sampling = if (r.sampling) |x| (if (x.temperature > 0) x else null) else null;
                const got = if (cands != null and sampler.gatheredFits(smp))
                    try d.gatheredFrom(cands.?, row_of[k], rows, r.seq.st.pos + 1 + j, smp)
                else
                    try d.sampleDraftRow(&e.f, mb, lg, row_of[k], rows, r.seq.st.pos + 1 + j, smp);
                const t = stops[a.req].take(j, got.prob);
                if (!t.propose) continue;
                r.out[r.drafted] = @intCast(got.token);
                r.drafted += 1;
                if (t.more and j + 1 < @min(r.count, r.out.len)) {
                    next[n_next] = .{ .req = a.req, .row = a.row, .tok = @intCast(got.token) };
                    n_next += 1;
                }
            }
            try e.f.mark(.draft_sample);
            if (n_next == 0) break;
            if (e.f.prof) |p| p.level = @intCast(@min(j + 1, 255));
            var toks: [64][1]u32 = undefined;
            for (next[0..n_next], 0..) |nx, k| {
                toks[k] = .{nx.tok};
                inputs[k] = .{ .seq = reqs[nx.req].seq, .next = &toks[k], .streams = mb.b.streams + nx.row * Wd * 2 };
            }
            _ = try mtp.stageMany(&e.f, inputs[0..n_next], mb, &segs);
            try e.runMulti(true, segs[0..n_next], mb);
            lg = mb.b.logits;
            rows = n_next;
            for (next[0..n_next], 0..) |nx, k| {
                const st = &reqs[nx.req].seq.st;
                try e.f.setMtpLen(st, st.mtp_len + 1);
                st.mtp_drafted += 1;
                active[k] = .{ .req = nx.req, .row = segs[k].a0 };
                row_of[k] = k;
            }
            n_active = n_next;
        }
    }

    /// One request of a reference multi-stream run (generateMany): its prompt, reply cap, rule and drafting, and
    /// the round it arrives at (0: before the first; later ones fill in slices between rounds).
    pub const Request = struct {
        prompt: []const u32,
        max_tokens: usize,
        sampling: ?lanes.Sampling = null,
        drafts: bool = true,
        /// the chains' confidence rule (Python's 0.70 for the oracle gates; the server's is confidenceSetting())
        confidence: f64 = default_confidence,
        stop_eos: bool = true,
        arrive: usize = 0,
        tokens: std.ArrayList(u32) = .empty,
        rounds: usize = 0,
        accepted: usize = 0,
    };

    /// Seconds generateMany spent in each part of its last run (the stream synchronized at each boundary).
    pub const Times = struct { prefill: f64 = 0, verify: f64 = 0, verify_host: f64 = 0, sample: f64 = 0, draft: f64 = 0, fill: f64 = 0, rounds: usize = 0 };

    pub const ManyOptions = struct {
        /// layers a filling prompt runs between two rounds (0: an arriving prompt prefills at once)
        fill_layers: usize = 0,
        /// the prompts present at the start prefill together, one pass over each group that fits (prefillMany)
        burst: bool = false,
    };

    /// A request's stop rule this round: a running product (c < 0) only while at most product_streams are live.
    fn hybrid(e: *const Engine, c: f64, live: []const bool) f64 {
        if (c >= 0) return c;
        var n: usize = 0;
        for (live) |l| n += @intFromBool(l);
        return if (n <= e.product_streams) c else wide_confidence;
    }

    pub fn generateMany(e: *Engine, gpa: Allocator, reqs: []Request) !void {
        return e.generateManyWith(gpa, reqs, .{});
    }

    /// Every request decoded together in shared rounds (Python MultiDecoder.round's order: verify every window,
    /// draw each stream's rows, keep each up to its first miss, every drafting stream's next drafts in shared MTP
    /// steps). Requests arriving later fill their prompts in layer slices between rounds while the others decode.
    /// Each request's tokens must equal its solo run's.
    pub fn generateManyWith(e: *Engine, gpa: Allocator, reqs: []Request, mo: ManyOptions) !void {
        e.times = .{};
        var t = std.Io.Clock.awake.now(e.io);
        const n = reqs.len;
        if (n > e.round_segs.len) return error.TooManyStreams;
        const seqs = try gpa.alloc(?*Seq, n);
        defer gpa.free(seqs);
        @memset(seqs, null);
        defer for (seqs) |sq| if (sq) |x| e.freeSeq(x);
        const fills = try gpa.alloc(?*Fill, n);
        defer gpa.free(fills);
        @memset(fills, null);
        defer for (fills) |f| if (f) |x| e.fillFree(x);
        const counts = try gpa.alloc(usize, n);
        defer gpa.free(counts);
        const live = try gpa.alloc(bool, n);
        defer gpa.free(live);
        @memset(live, false);
        const finished = try gpa.alloc(bool, n);
        defer gpa.free(finished);
        @memset(finished, false);
        const drafts = try gpa.alloc([state.max_rows]u32, n);
        defer gpa.free(drafts);
        const n_drafts = try gpa.alloc(usize, n);
        defer gpa.free(n_drafts);
        @memset(n_drafts, 0);
        const keep_bound = e.bound;
        defer e.bound = keep_bound;
        var reqs_d: [64]DraftReq = undefined;
        var map: [64]usize = undefined;
        for (reqs, 0..) |*r, i| {
            const room = @as(i64, @intCast(e.max_len)) - @as(i64, @intCast(r.prompt.len)) - @as(i64, @intCast(e.depth)) - 1;
            if (room < 1) return error.PromptPastWindow;
            counts[i] = @max(1, @min(r.max_tokens, @as(usize, @intCast(room))));
            r.tokens.clearRetainingCapacity();
        }
        // a prompt's first draw: the stream goes live, its first drafts from the prompt's last row follow
        const Start = struct {
            fn first(eng: *Engine, gp: Allocator, rq: *Request, tok: u32, cnt: usize, lv: *bool) !void {
                try rq.tokens.append(gp, tok);
                lv.* = !((rq.stop_eos and eng.isEos(tok)) or cnt <= 1);
            }
        };
        var nd: usize = 0;
        const pre = try gpa.alloc(?u32, n);
        defer gpa.free(pre);
        @memset(pre, null);
        if (mo.burst) {
            // greedy groups of consecutive prompts of one head mode that fit one pass; a lone one prefills as usual
            var i: usize = 0;
            while (i < n) {
                if (!(reqs[i].arrive == 0 or mo.fill_layers == 0)) {
                    i += 1;
                    continue;
                }
                const absorbing = reqs[i].drafts and e.depth > 0;
                var items: [state.max_rows]ManyPrompt = undefined;
                var at: [state.max_rows]usize = undefined;
                var m: usize = 0;
                var rows: usize = 0;
                var j = i;
                while (j < n and m < state.max_rows and (reqs[j].arrive == 0 or mo.fill_layers == 0) and (reqs[j].drafts and e.depth > 0) == absorbing and rows + reqs[j].prompt.len <= e.prefill_rows) : (j += 1) {
                    seqs[j] = try e.newSeq(e.max_len);
                    items[m] = .{ .seq = seqs[j].?, .prompt = reqs[j].prompt, .sampling = reqs[j].sampling };
                    at[m] = j;
                    rows += reqs[j].prompt.len;
                    m += 1;
                }
                if (m >= 2 and e.manyFit(items[0..m])) {
                    var firsts: [state.max_rows]u32 = undefined;
                    e.absorbing = absorbing;
                    try e.prefillMany(items[0..m], &firsts);
                    for (at[0..m], firsts[0..m]) |k, f| pre[k] = f;
                }
                i = @max(j, i + 1);
            }
        }
        for (reqs, 0..) |*r, i| if (r.arrive == 0 or mo.fill_layers == 0) {
            if (seqs[i] == null) seqs[i] = try e.newSeq(e.max_len);
            e.bound = seqs[i].?;
            e.absorbing = r.drafts and e.depth > 0;
            const first = if (pre[i]) |f| f else try e.prefill(r.prompt, r.sampling);
            try Start.first(e, gpa, r, first, counts[i], &live[i]);
            finished[i] = !live[i];
            if (live[i] and r.drafts and e.depth > 0) {
                reqs_d[nd] = .{ .seq = seqs[i].?, .source = .prefill_last, .next = r.tokens.items[0..1], .count = @min(e.depth, (counts[i] - 1) -| 1), .sampling = r.sampling, .confidence = e.hybrid(r.confidence, live), .out = &drafts[i] };
                map[nd] = i;
                nd += 1;
            }
        };
        try e.stream.synchronize();
        e.times.prefill += seconds(e.io, t);
        t = std.Io.Clock.awake.now(e.io);
        // TF_FLASHNEXT_PROFILE: the rounds' GPU time by part (eager: graphs off while profiling)
        const t_rounds = t;
        e.f.prof = e.prof;
        defer e.f.prof = null;
        if (e.prof) |p| p.n = 0;
        try e.draftMany(reqs_d[0..nd]);
        for (reqs_d[0..nd], map[0..nd]) |dr, i| n_drafts[i] = dr.drafted;
        var windows: [64]Window = undefined;
        var widx: [64]usize = undefined;
        var toks: [64][state.max_rows]u32 = undefined;
        var sampled: [state.max_rows]u32 = undefined;
        var kept: [64][state.max_rows]u32 = undefined;
        var keeps: [64]usize = undefined;
        var round: usize = 0;
        while (true) : (round += 1) {
            // arrivals: the oldest arrived prompt starts filling once no other fill holds the prompt buffers
            if (mo.fill_layers > 0) {
                var busy = false;
                for (fills) |f| busy = busy or f != null;
                if (!busy) for (reqs, 0..) |*r, i| if (r.arrive > 0 and r.arrive <= round and seqs[i] == null) {
                    seqs[i] = try e.newSeq(e.max_len);
                    e.absorbing = r.drafts and e.depth > 0;
                    fills[i] = try e.fillBegin(seqs[i].?, r.prompt, r.sampling, .{});
                    break;
                };
            }
            var nw: usize = 0;
            for (reqs, 0..) |r, i| if (live[i]) {
                toks[i][0] = r.tokens.items[r.tokens.items.len - 1];
                @memcpy(toks[i][1 .. 1 + n_drafts[i]], drafts[i][0..n_drafts[i]]);
                windows[nw] = .{ .seq = seqs[i].?, .tokens = toks[i][0 .. 1 + n_drafts[i]] };
                widx[nw] = i;
                nw += 1;
            };
            var waiting = false;
            for (reqs, 0..) |_, i| waiting = waiting or (!finished[i] and !live[i]);
            if (nw == 0 and !waiting) break;
            if (nw > 0) {
                try e.stream.synchronize();
                e.times.draft += seconds(e.io, t);
                t = std.Io.Clock.awake.now(e.io);
                try e.verifyMany(windows[0..nw]);
                e.times.verify_host += seconds(e.io, t);
                try e.stream.synchronize();
                e.times.verify += seconds(e.io, t);
                t = std.Io.Clock.awake.now(e.io);
                e.times.rounds += 1;
                for (0..nw) |w| {
                    const i = widx[w];
                    const r = &reqs[i];
                    const rows = windows[w].tokens.len;
                    try e.sampleWindow(w, r.sampling, sampled[0..rows]);
                    try e.f.mark(.sample);
                    var keep: usize = 1;
                    for (windows[w].tokens[1..rows], 0..) |dt, k| {
                        if (sampled[k] != dt or (r.stop_eos and e.isEos(sampled[k]))) break;
                        keep += 1;
                    }
                    try e.commitWindow(w, keep);
                    try e.f.mark(.commit);
                    @memcpy(kept[i][0..keep], sampled[0..keep]);
                    keeps[i] = keep;
                    r.rounds += 1;
                    r.accepted += keep - 1;
                    try r.tokens.appendSlice(gpa, sampled[0..keep]);
                    const last = r.tokens.items[r.tokens.items.len - 1];
                    if (r.tokens.items.len >= counts[i] or (r.stop_eos and e.isEos(last))) {
                        live[i] = false;
                        finished[i] = true;
                    }
                }
                try e.stream.synchronize();
                if (e.f.prof) |p| try p.drain();
                e.times.sample += seconds(e.io, t);
                t = std.Io.Clock.awake.now(e.io);
            }
            nd = 0;
            for (0..nw) |w| {
                const i = widx[w];
                n_drafts[i] = 0;
                const r = &reqs[i];
                if (!live[i] or !r.drafts or e.depth == 0) continue;
                // room 0 (one token left): the head still absorbs the kept rows, drafting nothing
                const room = @min(e.depth, (counts[i] - r.tokens.items.len) -| 1);
                reqs_d[nd] = .{ .seq = seqs[i].?, .source = .{ .window = w }, .next = kept[i][0..keeps[i]], .count = room, .sampling = r.sampling, .confidence = e.hybrid(r.confidence, live), .out = &drafts[i] };
                map[nd] = i;
                nd += 1;
            }
            // filling prompts: a slice each; one that ends drafts from its last row with the others
            const tf = std.Io.Clock.awake.now(e.io);
            for (reqs, 0..) |*r, i| if (fills[i]) |fl| {
                if (!try e.fillStep(fl, mo.fill_layers)) continue;
                try Start.first(e, gpa, r, fl.first, counts[i], &live[i]);
                finished[i] = !live[i];
                e.fillFree(fl);
                fills[i] = null;
                n_drafts[i] = 0;
                if (live[i] and r.drafts and e.depth > 0) {
                    reqs_d[nd] = .{ .seq = seqs[i].?, .source = .prefill_last, .next = r.tokens.items[0..1], .count = @min(e.depth, (counts[i] - 1) -| 1), .sampling = r.sampling, .confidence = e.hybrid(r.confidence, live), .out = &drafts[i] };
                    map[nd] = i;
                    nd += 1;
                }
            };
            try e.stream.synchronize();
            e.times.fill += seconds(e.io, tf);
            try e.draftMany(reqs_d[0..nd]);
            for (reqs_d[0..nd], map[0..nd]) |dr, i| n_drafts[i] = dr.drafted;
        }
        for (reqs, counts) |*r, c| r.tokens.shrinkRetainingCapacity(@min(r.tokens.items.len, c));
        if (e.prof) |p| {
            try e.stream.synchronize();
            try p.rounds(e.rank, "decode", e.times.rounds, seconds(e.io, t_rounds) * 1000.0);
            for (&e.f.experts_seen, [_][]const u8{ "main", "MTP head" }) |*c, what| {
                if (c.layers > 0) std.debug.print("rounds rank {d} decode   {s}: {d:.1} distinct routed experts a MoE layer over {d:.1} rows\n", .{ e.rank, what, @as(f64, @floatFromInt(c.distinct)) / @as(f64, @floatFromInt(c.layers)), @as(f64, @floatFromInt(c.rows)) / @as(f64, @floatFromInt(c.layers)) });
                c.* = .{};
            }
        }
    }

    // -- CUDA graphs ------------------------------------------------------------------------------------------

    /// Graphs of sequence `seq` (all, or those of other buffer versions than `keep`) released.
    fn dropGraphs(e: *Engine, seq: usize, keep: ?u32) void {
        while (true) {
            var it = e.graphs.iterator();
            const found = while (it.next()) |kv| {
                if (kv.key_ptr.seq == seq and (keep == null or kv.key_ptr.version != keep.?)) break kv.key_ptr.*;
            } else null;
            const k = found orelse return;
            var g = e.graphs.fetchRemove(k).?.value;
            g.deinit();
        }
    }

    fn graphable(e: *Engine, rows: usize) bool {
        return e.graphs_on and e.f.dump == null and e.f.prof == null and rows <= e.buf.b.rows;
    }

    /// The round scratch's parity the staged window writes (a graph bakes in both parities' buffers).
    fn parity(e: *Engine) ?u32 {
        return e.f.cur_par;
    }

    fn body(e: *Engine, comptime head: bool, rows: usize) !void {
        if (head) {
            _ = try mtp.compute(&e.f, e.bound, &e.mbuf.?, rows);
        } else _ = try e.f.compute(e.bound, &e.buf, rows, true);
    }

    /// Replays the staged window's graph, capturing it first: an eager run at the bucket (this call's result),
    /// then the capture of the same launches (Graphs.forward / mtp_forward).
    fn graphed(e: *Engine, comptime head: bool, rows: usize, ctx: usize, par: u32) !void {
        const seq = e.bound;
        const key: GraphKey = .{ .seq = @intFromPtr(seq), .version = seq.st.version, .mtp = head, .rows = @intCast(rows), .parity = par, .ctx = ctx };
        e.f.context = ctx;
        defer e.f.context = null;
        if (e.graphs.get(key)) |g| return g.launchOn(e.stream);
        e.dropGraphs(key.seq, key.version);
        try e.body(head, rows);
        try e.stream.synchronize();
        try cuda.graph.beginCapture(e.stream, .thread_local);
        e.body(head, rows) catch |err| {
            if (cuda.graph.endCapture(e.stream)) |g| {
                var x = g;
                x.deinit();
            } else |_| {}
            return err;
        };
        var g = try cuda.graph.endCapture(e.stream);
        defer g.deinit();
        var exec = try g.instantiate();
        errdefer exec.deinit();
        try exec.upload(e.stream);
        try e.graphs.put(e.gpa, key, exec);
        e.captures += 1;
    }

    /// Graphs.warm: every decode window (1 .. depth + 1 rows) at both DeltaNet parities and every MTP step size
    /// captured on the engine's own sequence before serving, so no request waits on a capture; the sequence is
    /// emptied after. Returns the graphs captured. (A sequence from newSeq captures its own on first use.)
    pub fn warmGraphs(e: *Engine) !usize {
        if (!e.graphs_on) return 0;
        const before = e.captures;
        const keep = e.bound;
        defer e.bound = keep;
        e.bound = e.own;
        const zeros: [state.max_rows]u32 = @splat(0);
        const rows = @min(e.depth + 1, e.buf.b.rows, state.max_rows);
        // each window twice: a forward writes the other parity each time
        try e.f.reset(e.own);
        for (1..rows + 1) |r| for (0..2) |_| try e.forward(zeros[0..r]);
        try e.f.reset(e.own);
        if (e.mbuf) |*mb| for (1..rows + 1) |n| try e.mtpForward(mb, zeros[0..n], e.buf.b.streams);
        try e.f.reset(e.own);
        try e.stream.synchronize();
        return e.captures - before;
    }

    /// Shared graphs released: those baking in sequence `seq`, or all (null).
    fn dropMulti(e: *Engine, seq: ?usize) void {
        while (true) {
            var it = e.multi.iterator();
            const found = while (it.next()) |kv| {
                if (seq == null or std.mem.indexOfScalar(usize, kv.value_ptr.seqs, seq.?) != null) break kv.key_ptr.*;
            } else null;
            const k = found orelse return;
            var g = e.multi.fetchRemove(k).?.value;
            g.exec.deinit();
            e.gpa.free(g.seqs);
        }
    }

    /// A shared step's composition: every segment's sequence, buffer version, rows, DeltaNet parity and context
    /// bucket (written to `ctxs`), and which buffers; null when a sequence's DeltaNet layers are out of step.
    fn multiKey(e: *Engine, comptime head: bool, segs: []const fwd.Seg, ctxs: []usize) ?u64 {
        var h = std.hash.Wyhash.init(if (head) 0x6d7470 else 0x6d61696e);
        for (segs, ctxs) |sg, *c| {
            const st = &sg.seq.st;
            var par: u32 = 0;
            if (!head and e.g.linear_layers > 0) par = e.f.cur_par;
            const n = sg.a1 - sg.a0;
            c.* = bucket((if (head) st.mtp_len else st.pos) + n, st.capacity);
            const words = [_]u64{ @intFromPtr(sg.seq), st.version, n, par, c.* };
            h.update(std.mem.asBytes(&words));
        }
        return h.final();
    }

    /// A shared step through its graph (captured on the composition's second sight: an eager run at the
    /// buckets, which is this call's result, then the capture), else eagerly. Staging stays outside.
    fn runMulti(e: *Engine, comptime head: bool, segs: []const fwd.Seg, x: *const fwd.Bufs) !void {
        var ctxs: [64]usize = undefined;
        if (segs.len > ctxs.len) return error.TooManyStreams;
        const key = if (e.graphs_on and e.f.dump == null and e.f.prof == null and segs.len <= e.multi_graphs) e.multiKey(head, segs, ctxs[0..segs.len]) else null;
        const k = key orelse {
            if (head) _ = try mtp.computeSegs(&e.f, segs, x) else _ = try e.f.computeSegs(segs, x, true);
            return;
        };
        e.f.contexts = ctxs[0..segs.len];
        defer e.f.contexts = null;
        if (e.multi.get(k)) |g| return g.exec.launchOn(e.stream);
        if (e.seen.count() > 16384) e.seen.clearRetainingCapacity();
        const again = (try e.seen.getOrPut(e.gpa, k)).found_existing;
        if (head) _ = try mtp.computeSegs(&e.f, segs, x) else _ = try e.f.computeSegs(segs, x, true);
        if (!again) return;
        if (e.multi.count() >= max_multi_graphs) {
            e.dropMulti(null);
            e.seen.clearRetainingCapacity();
        }
        try e.stream.synchronize();
        try cuda.graph.beginCapture(e.stream, .thread_local);
        const r = if (head) mtp.computeSegs(&e.f, segs, x) else e.f.computeSegs(segs, x, true);
        _ = r catch |err| {
            if (cuda.graph.endCapture(e.stream)) |g| {
                var y = g;
                y.deinit();
            } else |_| {}
            return err;
        };
        var g = try cuda.graph.endCapture(e.stream);
        defer g.deinit();
        var exec = try g.instantiate();
        errdefer exec.deinit();
        try exec.upload(e.stream);
        const seqs = try e.gpa.alloc(usize, segs.len);
        errdefer e.gpa.free(seqs);
        for (segs, seqs) |sg, *q| q.* = @intFromPtr(sg.seq);
        try e.multi.put(e.gpa, k, .{ .exec = exec, .seqs = seqs });
        e.captures += 1;
    }

    /// mtp_forward through a graph when it fits one (the stage stays outside).
    fn mtpForward(e: *Engine, mb: *const fwd.Bufs, next: []const u32, streams: u64) !void {
        try mtp.stage(&e.f, e.bound, mb, next, streams);
        const n = next.len;
        if (e.graphable(n) and mb == &e.mbuf.?) {
            try e.graphed(true, n, bucket(e.bound.st.mtp_len + n, e.bound.st.capacity), 0);
        } else _ = try mtp.compute(&e.f, e.bound, mb, n);
    }

    // -- cuda_decode's E interface on the bound sequence ------------------------------------------------------

    pub fn pos(e: *Engine) u64 {
        return e.bound.st.pos;
    }

    /// A window's rows at pos .. pos + tokens.len - 1 (main head logits of every row).
    pub fn forward(e: *Engine, tokens: []const u32) !void {
        const R = tokens.len;
        try e.f.stage(e.bound, &e.buf, tokens);
        const par = e.parity();
        if (e.graphable(R) and par != null) {
            try e.graphed(false, R, bucket(e.bound.st.pos + R, e.bound.st.capacity), par.?);
        } else _ = try e.f.compute(e.bound, &e.buf, R, true);
        e.rows = R;
    }

    /// The target's draws of the window's first `rows` rows, keyed at first_position + r.
    pub fn sample(e: *Engine, rows: usize, first_position: u64, out: []u32) !void {
        try e.draw(e.buf.b.logits, e.buf.b.cand_all, rows, first_position, out);
    }

    pub fn commit(e: *Engine, rows: usize, keep: usize) !void {
        try e.f.commit(e.bound, &e.buf, rows, keep);
    }

    /// decode.absorb: the MTP cache drops the last chain's drafts and takes the kept rows' streams with the next
    /// tokens (the prompt's last row, or the last window's first next_tokens.len rows).
    pub fn absorb(e: *Engine, source: decode.Source, next_tokens: []const u32) !void {
        const mb = &(e.mbuf orelse return error.NoMtpHead);
        const st = &e.bound.st;
        if (st.mtp_drafted != 0) {
            try e.f.setMtpLen(st, st.mtp_len - st.mtp_drafted);
            st.mtp_drafted = 0;
        }
        const streams = switch (source) {
            .prefill_last => e.lastStreams(),
            .window => e.buf.b.streams,
        };
        try e.mtpForward(mb, next_tokens, streams);
        try e.f.setMtpLen(st, st.mtp_len + next_tokens.len);
    }

    /// One more MTP step on draft `token`, from the head's output row `from_row` (decode.draft's chain).
    pub fn chain(e: *Engine, token: u32, from_row: usize) !void {
        const mb = &(e.mbuf orelse return error.NoMtpHead);
        const st = &e.bound.st;
        const prev = mb.b.streams + from_row * e.g.wide() * 2;
        try e.mtpForward(mb, &.{token}, prev);
        try e.f.setMtpLen(st, st.mtp_len + 1);
        st.mtp_drafted += 1;
    }

    pub fn sampleDraft(e: *Engine, position: u64) !decode.Draw {
        const d = &(e.draws orelse return error.NoMtpHead);
        const got = try d.sampleDraft(&e.f, &e.mbuf.?, e.mbuf.?.b.logits, position, e.sampling);
        return .{ .token = @intCast(got.token), .prob = got.prob };
    }

    pub fn drawDraft(e: *Engine, position: u64) !u32 {
        const d = &(e.draws orelse return error.NoMtpHead);
        return @intCast(try d.drawDraft(&e.f, &e.mbuf.?, e.mbuf.?.b.logits, position, e.sampling));
    }

    pub fn isEos(e: *Engine, token: u32) bool {
        return e.c.isEos(token);
    }

    // -- the oracle checks' entry points ----------------------------------------------------------------------

    /// The teacher's MTP step: mtp_forward([next], buf.streams[:1]) at mtp_len, then mtp_len + 1; the draft head's
    /// row (device).
    pub fn teacherMtp(e: *Engine, next: u32) !u64 {
        const mb = &(e.mbuf orelse return error.NoMtpHead);
        const st = &e.bound.st;
        try e.mtpForward(mb, &.{next}, e.buf.b.streams);
        try e.f.setMtpLen(st, st.mtp_len + 1);
        return mb.b.logits;
    }

    pub fn reset(e: *Engine) !void {
        try e.f.reset(e.bound);
    }
};

test "graph buckets follow Graphs._bucket" {
    try std.testing.expectEqual(@as(usize, 8192), bucket(1, 262151));
    try std.testing.expectEqual(@as(usize, 8192), bucket(8192, 262151));
    try std.testing.expectEqual(@as(usize, 16384), bucket(8193, 262151));
    try std.testing.expectEqual(@as(usize, 262144), bucket(200000, 262151));
    try std.testing.expectEqual(@as(usize, 262151), bucket(262150, 262151));
    try std.testing.expectEqual(@as(usize, 4096), bucket(10, 4096));
}

test "geometry and dims follow the config's split" {
    // the shapes the Triton wrappers' fixtures were recorded with
    const d = tri.Dims{};
    try std.testing.expectEqual(@as(usize, 13952), d.attnWidth());
    try std.testing.expectEqual(@as(usize, 16480), d.projWidth());
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Engine);
}
