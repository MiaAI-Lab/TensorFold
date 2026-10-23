//! GLM CUDA decode engine: P4 KDA inside P6 forward, P7 rounds, lanes and sampler.
//! Prompt prefill is package P8. DFlash2 drafts are package P9. Triton cubins are package P1.
//! A missing piece returns its error. This file does not substitute that math.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const aot = cuda.aot;
const cfgs = @import("cuda_config.zig");
const weights = @import("cuda_weights.zig");
const model_weights = @import("cuda_model_weights.zig");
const state = @import("cuda_state.zig");
const seg = @import("cuda_segments.zig");
const kda = @import("cuda_kda.zig");
const mm = @import("cuda_matmuls.zig");
const hc = @import("cuda_hc.zig");
const tri = @import("cuda_triton.zig");
const fwd = @import("cuda_forward.zig");
const verify = @import("cuda_verify.zig");
const side_mod = @import("cuda_side.zig");
const prompt_micro = @import("cuda_prompt_micro.zig");
const prefill_split = @import("cuda_prefill_split.zig");
const rounds = @import("cuda_rounds.zig");
const sampler = @import("cuda_sampler.zig");
const grammar = @import("cuda_grammar.zig");
const comm = @import("cuda_comm.zig");
const mirror = @import("cuda_mirror.zig");
const ops = @import("cuda_torch_ops.zig");
const prefill_mod = @import("cuda_prefill.zig");
const prefill_loop = @import("cuda_prefill_loop.zig");
const draft_pkg = @import("cuda_dflash_package.zig");
const dflash = @import("cuda_dflash.zig");
const dflash_config = @import("cuda_dflash_config.zig");
const snapshot = @import("cuda_snapshot.zig");
const spill_state = @import("cuda_spill_state.zig");
const spill_tier = @import("cuda_spill_tier.zig");
const cache_plan = @import("cuda_cache_plan.zig");
const spill_mod = @import("cuda_spill.zig");
const mtp_mod = @import("cuda_mtp.zig");
const choice_mod = @import("cuda_mtp_choice.zig");
const vision_tower = @import("cuda_vision.zig");
const G = state.Geometry;
const pool_mod = @import("cuda_pool.zig");
const capacity = @import("cuda_pool_capacity.zig");
const display = @import("cuda_display_kv.zig");
const shard_mod = @import("cuda_shard.zig");
const shard_dec = @import("cuda_shard_decode.zig");
const shard_prompt = @import("cuda_shard_prompt.zig");
const shard_keep = @import("cuda_shard_keep.zig");

pub const Options = struct {
    context: usize,
    drafts: bool,
    streams: usize = 1,
    rank: u32 = 0,
    world: u32 = 1,
    comm: ?*const comm.Comm = null,
    /// Skip graph capture. The first forward still runs the real kernels and fails closed on a missing cubin.
    eager: bool = false,
    draft_dir: ?[]const u8 = null,
    draft_choice: ?choice_mod.Choice = null,
    /// picture and clip input; rank 0 also loads the tower (the Zig backend; the Python backend brings features)
    vision: bool = false,
    tower: bool = false,
    /// TF_GLM_KV_SHARD's switches, read and agreed by cuda_native (null: off, every rank's caches the whole pool's)
    kv_shard: ?shard_mod.Switches = null,
    /// TF_GLM_SPILL_V2=1: spill v2's settings, whose pinned staging is taken before the pool is sized (null: off)
    spill_v2: ?spill_tier.Settings = null,
};

/// TF_GLM_KV_SHARD's counters (/health's kv_shard): layout digests sent or compared (TF_GLM_KV_SHARD_CHECK; a
/// mismatch ends the pair), carries (each a resume whose state's home part had no room: kv_parts' `carry_rescued`), the
/// tokens and owner-only bytes they moved, resumes that started from 0 rather than carry a state under the minimum,
/// prompts filled with borrowed twins and twins taken back.
pub const KvCounts = struct { checks: u64 = 0, carries: u64 = 0, moved_tokens: u64 = 0, moved_bytes: u64 = 0, refused_small: u64 = 0, twins_made: u64 = 0, twins_revoked: u64 = 0 };

/// `tag`: the id of the kept state this sequence's caches stand at (set when it is saved from or restored into the
/// sequence; the position tells whether anything was written since), so a resume of the same state skips the copy.
/// `eid`: the shared KV pool extent this sequence holds.
pub const Seq = struct { slot: usize, st: state.State, alive: bool = false, tag: u64 = 0, eid: ?u64 = null };
/// A kept prompt state (cuda_snapshot.zig): one device buffer holding the sequence's state after `pos` tokens.
/// `refs`: the owner's, and one for each background write that reads the buffer (the last release frees it).
/// `home`: TF_GLM_KV_SHARD, the part its stream lay in, whose owner alone holds its latent rows (null: unsharded).
pub const Snap = struct { id: u64 = 0, pos: usize = 0, mtp_len: usize = 0, buf: ?cuda.DeviceBuffer = null, sec: snapshot.Sections = undefined, refs: std.atomic.Value(u32) = .init(1), home: ?u8 = null };
const SpillJob = struct { snap: *Snap, tokens: []u32, at: u32, key: u64 };
/// `mask`: structured output's bits for the first token (one row, the grammar's words); empty: unconstrained.
/// `media`: picture and clip rows for this prompt.
/// `live_from`: the bound sequence stands at this position with live rows and no snapshot (a refused save, cuda_lanes
/// pass): the prefill goes on from there, restoring nothing. `resume_from` and `live_from` exclude each other.
pub const PrefillOptions = struct { keep_at: ?usize = null, resume_from: ?*const Snap = null, live_from: ?usize = null, mask: []const u32 = &.{}, media: ?*const lanes.Media = null };
pub const Prefilled = struct { first: u32, kept: ?*Snap = null };

/// A prompt's feature rows while it prefills: the prompt rows they replace (host copy, ascending) and the device
/// copies of both. Every rank holds them (rank 1 by broadcast).
const Attached = struct {
    host_rows: []u32,
    rows: cuda.DeviceBuffer,
    feats: cuda.DeviceBuffer,
};

pub const media_width: usize = 4096;

pub const Engine = struct {
    /// Lane code looks these up on the engine type (E.Window, E.DraftReq, E.ManyPrompt).
    pub const Window = struct { seq: *Seq, tokens: []const u32 };
    pub const DraftSource = union(enum) { prefill_last, window: usize };
    pub const DraftReq = struct {
        seq: *Seq,
        source: DraftSource,
        next: []const u32,
        count: usize,
        sampling: ?lanes.Sampling = null,
        confidence: f64 = 0,
        out: []u32,
        drafted: usize = 0,
    };
    /// Kept states are device buffers of their own (takeSnap / restoreSnap); the lanes keep no keeper sequence.
    pub const snap_holds_rows = true;
    pub const ManyPrompt = struct { seq: *Seq, prompt: []const u32, sampling: ?lanes.Sampling };

    gpa: std.mem.Allocator,
    g: G,
    context: usize,
    window: usize,
    drafts: bool,
    drafter: ?*draft_pkg.Package = null,
    mtp: ?mtp_mod.Head = null,
    draft_choice: choice_mod.Choice = .dflash2,
    prompt_tokens: []const u32 = &.{},
    prompt_at: usize = 0,
    lt: ?cuda.cublaslt.Library = null,
    prefill_rows: usize = 2048,
    cfg: cfgs.Config,
    cat: weights.Catalog,
    plan: weights.Plan,
    prepared: model_weights.Prepared,
    set: aot.Set,
    kda_k: kda.Kernels,
    entries: mm.Entries,
    torch: ops.Ops,
    stream: cuda.Stream,
    slots: state.Slots,
    caches: state.Caches,
    pool: pool_mod.Pool,
    pool_free: std.atomic.Value(usize) = .init(0),
    tables: seg.SegRows,
    memo: tri.Memo,
    fwd: fwd.Forward,
    side: ?side_mod.Side = null,
    verify: verify.Verify,
    pe: prefill_mod.Engine,
    sampler: sampler.Sampler,
    comm: ?*const comm.Comm = null,
    seqs: [8]Seq = undefined,
    used: [8]bool = @splat(false),
    bound: ?*Seq = null,
    sampling: ?lanes.Sampling = null,
    round_n: usize = 0,
    round_row0: [8]usize = @splat(0),
    round_len: [8]usize = @splat(0),
    committed: [8]bool = @splat(false),
    host_tokens: [8][64]i32 = undefined,
    round_st: [8]*state.State = undefined,
    commit_words: [48]i32 = @splat(0),
    commit_nseg: usize = 0,
    /// structured output's logits mask, loaded at the first constrained draw
    gmasks: [8]?grammar.Mask = @splat(null), // structured output's bits, one buffer a window (a round's windows are masked before one draw)
    /// the last kept state's id (Snap.id, Seq.tag)
    snap_next: u64 = 0,
    /// MEMORY_RESERVE_GIB in bytes: a kept state is refused when the machine would be left under it (0: no check)
    keep_reserve: u64 = 0,
    /// the spill tier (SPILL_GIB; null: off), opened by cuda_native and owned by the engine
    spill: ?*spill_mod.Tier = null,
    /// the CUDA context: the prompt cache's calls run on the lane host's thread, which may not have made it current yet
    ctx: ?*const cuda.Context = null,
    /// the spill tier's writer: jobs wait here for its thread, which copies a state off the device and into its file
    spill_jobs: std.ArrayList(SpillJob) = .empty,
    spill_mu: std.atomic.Mutex = .unlocked,
    spill_thread: ?std.Thread = null,
    spill_quit: std.atomic.Value(bool) = .init(false),
    /// jobs queued or running
    spill_busy: std.atomic.Value(u32) = .init(0),
    /// how long a clean stop waits for the writer (SPILL_FLUSH_S)
    spill_wait_ns: u64 = 60 * std.time.ns_per_s,
    /// the tokens of the stored state `spillBest` last found (the lane thread's)
    spill_hit: std.ArrayList(u32) = .empty,
    /// the prompt in prefill when it has pictures or clips: the tower's rows, on the device (every rank)
    media: ?Attached = null,
    vision: bool = false,
    /// rank 0 with --vision-backend zig: the vision tower (null: the server brings the features)
    tower: ?vision_tower.Tower = null,
    /// TF_GLM_KV_SHARD: this rank's view of the pool's two parts (null: off), the switches and the counters
    shard: ?shard_mod.Shard = null,
    kv_switches: shard_mod.Switches = .{},
    kv_counts: KvCounts = .{},
    /// spill v2: its pinned staging (Options.spill_v2, taken before the pool) and one event a staging slot
    tier_staging: ?cuda.HostBuffer = null,
    tier_events: []cuda.Event = &.{},
    /// TF_GLM_KV_SHARD: owner decode's kernels, exchange buffers and side stream (e.fwd.shard points here)
    kv_dec: ?shard_dec.Decode = null,
    /// TF_GLM_KV_SHARD: the prompt chunks' history fetch, its kernels, buffers and counters (e.pe.shard points here)
    kv_prompt: ?shard_prompt.Fetch = null,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !*Engine {
        if (o.world < 1 or o.world > 4) return error.UnsupportedTp;
        if (o.rank >= o.world) return error.BadRank;
        const display_cfg = try display.Config.read();
        const streams: usize = @max(1, @min(o.streams, 8));
        // The window is 64 rows even without drafts: the segment table's row planes are then 256-byte apart, as in the
        // captured Python launches (a 1-row window puts POS and BASE 4 bytes apart, a Triton specialization never captured).
        const window: usize = 64;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        const extent = std.mem.alignForward(usize, o.context + window, G.extent);
        e.gpa = gpa;
        e.g = try G.init(@as(usize, o.world), @as(usize, o.rank));
        e.context = o.context;
        e.window = window;
        e.drafts = o.drafts;
        e.drafter = null;
        e.mtp = null;
        e.draft_choice = o.draft_choice orelse try choice_mod.environment();
        e.prompt_tokens = &.{};
        e.prompt_at = 0;
        e.lt = null;
        const draft_dir = if (o.drafts and e.draft_choice == .dflash2) o.draft_dir orelse return error.MissingDraftDirectory else null;
        // TF_GLM_PREFILL_ROWS: rows of a prompt chunk (any chunking gives the same bits); both ranks alike. Default 2048.
        e.prefill_rows = try prefill_split.rowsFromEnv();
        const pm = try prompt_micro.fromEnv();
        const hc_split = try prefill_split.fromEnv();
        if (pm.microbatch and (o.world != 2 or hc_split.overlap == 0)) std.log.warn("TF_GLM_PROMPT_MICROBATCH=1 needs two ranks and TF_GLM_PREFILL_OVERLAP=1 or 2 here; chunks run whole (the same bits)", .{});
        if (pm.own_front and o.world != 2) std.log.warn("TF_GLM_PROMPT_OWN_FRONT=1: the row split is two ranks only so far; prompts run unsplit (the same bits)", .{});
        e.comm = o.comm;
        e.used = @splat(false);
        e.bound = null;
        e.sampling = null;
        e.round_n = 0;
        e.commit_nseg = 0;
        e.snap_next = 0;
        e.keep_reserve = 0;
        e.spill = null;
        e.ctx = ctx;
        e.spill_jobs = .empty;
        e.spill_mu = .unlocked;
        e.spill_thread = null;
        e.spill_quit = .init(false);
        e.spill_busy = .init(0);
        e.spill_wait_ns = 60 * std.time.ns_per_s;
        e.spill_hit = .empty;
        e.media = null;
        e.vision = o.vision;
        e.tower = null;
        e.committed = @splat(false);
        e.shard = null;
        e.kv_switches = o.kv_shard orelse .{};
        e.kv_counts = .{};
        e.kv_dec = null;
        errdefer if (e.kv_dec) |*k| k.deinit();
        e.kv_prompt = null;
        errdefer if (e.kv_prompt) |*k| k.deinit();
        if (o.kv_shard) |sw| try shard_mod.refuse(sw, o.world, @intCast(streams), false);
        var why: cfgs.Why = .{};
        e.cfg = try cfgs.Config.read(gpa, io, dir, .{}, &why);
        errdefer e.cfg.deinit();
        const mtp_on = try choice_mod.load(choice_mod.setting(), e.draft_choice, o.drafts, e.cfg.mtp_layers, streams);
        if (o.kv_shard != null and mtp_on) return error.KvShardWithMtp;
        e.cat = try weights.loadCatalog(gpa, io, dir, draft_dir);
        errdefer e.cat.deinit();
        e.plan = try weights.plan(gpa, &e.cfg, &e.cat, .{ .rank = o.rank, .world = @intCast(o.world), .mtp = mtp_on });
        errdefer e.plan.deinit();
        // P2/P3: candidateMse is the host quantizer. It is not qualified against ATen CUDA clipping.
        std.log.warn("GLM EXL3 upload uses candidateMse; clipping parity against ATen CUDA is unproven", .{});
        std.log.warn("lm_head uses quant.makeF8; block-FP8 load parity is the P2/P3 qualification limit", .{});
        const t_load = std.Io.Timestamp.now(io, .awake);
        e.prepared = try model_weights.Prepared.load(gpa, io, ctx.d, &e.plan, dir, model_weights.parallelMse);
        std.log.info("weights loaded in {d} ms (dense MSE quantize {d} ms, expert stacks {d} ms, dense reads {d} ms, pack {d} ms, head {d} ms)", .{ @divTrunc(t_load.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds, 1_000_000), e.prepared.ns_quant / 1_000_000, e.prepared.ns_put / 1_000_000, e.prepared.ns_qread / 1_000_000, e.prepared.ns_pack / 1_000_000, e.prepared.ns_head / 1_000_000 });
        errdefer e.prepared.deinit();
        e.stream = try cuda.Stream.init(ctx.d, true);
        errdefer e.stream.deinit();
        e.set = try aot.Set.load(gpa, io, ctx.d, ctx.device, kernels);
        errdefer e.set.deinit();
        e.kda_k = try kda.Kernels.load(ctx.d);
        errdefer e.kda_k.deinit();
        e.entries = try mm.Entries.load(ctx);
        errdefer e.entries.deinit();
        e.torch = try ops.Ops.load(ctx);
        errdefer e.torch.deinit();
        e.slots = try state.Slots.init(ctx.d, e.g, streams, window);
        errdefer e.slots.deinit();
        // the index ring holds the longest chunk or window: forward.index_ring(capacity, max(max_rows, prefill_rows))
        e.caches = try state.Caches.init(ctx.d, extent, streams, @max(window, e.prefill_rows));
        var caches_live = true;
        errdefer if (caches_live) e.caches.deinit();
        e.tables = try seg.SegRows.init(gpa, window, 8, e.caches.ring);
        errdefer e.tables.deinit();
        try e.tables.initDevice(ctx.d);
        var buffers = try state.Buffers.init(ctx.d, e.g, window, false);
        errdefer buffers.deinit();
        var scratch = try fwd.Scratch.init(ctx.d, e.g, window, decodeScratchRows(e.caches.rows));
        var scratch_live = true;
        errdefer if (scratch_live) scratch.deinit();
        var hc_mod = try hc.HC.init(ctx.d);
        errdefer hc_mod.deinit();
        e.memo = .{ .gpa = gpa };
        errdefer e.memo.deinit();
        e.fwd = .{
            .g = e.g,
            .tri = .{ .set = &e.set, .s = e.stream, .memo = &e.memo },
            .hc = hc_mod,
            .entries = &e.entries,
            .weights = &e.prepared.weights,
            .b = buffers,
            .s = scratch,
            .slots = &e.slots,
            .caches = &e.caches,
            .tables = &e.tables,
            .kseg = 0,
            .comm = o.comm,
            .kda = kda.install(&e.kda_k),
        };
        // TF_GLM_SIDE=1: the MoE's shared expert on a second stream in decode and verify windows (graphs included); the
        // same bits, so ranks need not agree. Off unless set, as in Python.
        e.side = null;
        if (try side_mod.fromEnv()) {
            e.side = try side_mod.Side.init(ctx.d);
            e.fwd.side = &e.side.?;
            std.log.info("TF_GLM_SIDE=1: the MoE shared expert runs on a second stream in decode windows", .{});
        }
        errdefer if (e.side) |*sd| sd.deinit();
        e.fwd.pf = try @import("cuda_l2pf_plan.zig").build(gpa, ctx.d, e.torch.l2pf, &e.prepared.weights, e.prepared.buffers.items);
        errdefer if (e.fwd.pf) |p| {
            p.deinit();
            gpa.destroy(p);
        };
        e.verify = try verify.Verify.init(&e.fwd, &e.tables);
        errdefer e.verify.deinit();
        // The prompt path: its own 2048-row buffers and scratch, the same weights, kernels and caches.
        var pbuf = try state.Buffers.init(ctx.d, e.g, e.prefill_rows, true);
        errdefer pbuf.deinit();
        var pscratch = try prefill_mod.Scratch.init(ctx.d, e.g, e.prefill_rows, extent);
        errdefer pscratch.deinit();
        e.pe = .{
            .g = e.g,
            .tri = e.fwd.tri,
            .entries = &e.entries,
            .weights = &e.prepared.weights,
            .b = pbuf,
            .s = pscratch,
            .cache = undefined,
            .conv = 0,
            .rec = 0,
            .ring = e.caches.ring,
            .pos_dev = 0,
            .kda = .{ .k = &e.kda_k, .s = e.stream },
            .comm = o.comm,
            .split = hc_split,
            .micro_s = pm,
        };
        e.pe.lane = null;
        if (hc_split.on and hc_split.overlap != 0 and e.g.world == 2) e.pe.lane = try prefill_split.Lane.init(ctx.d);
        errdefer if (e.pe.lane) |*l| l.deinit();
        e.sampler = .{
            .gpa = gpa,
            .d = ctx.d,
            .th = e.torch.on(e.stream),
            .comm = if (o.comm) |c| @constCast(c) else null,
            .columns = e.g.vocab,
            .vocab_offset = @intCast(e.g.vocab_start),
            .check_greedy = std.c.getenv("TF_GLM_GREEDY_CHECK") != null,
            .legacy_greedy = std.c.getenv("TF_GLM_GREEDY_TOPK") != null,
        };
        if (mtp_on) {
            e.mtp = try mtp_mod.Head.init(gpa, ctx.d, e.g, extent);
        }
        errdefer if (e.mtp) |*m| m.deinit();
        if (o.vision and o.tower and o.rank == 0) {
            e.tower = try vision_tower.Tower.init(gpa, io, ctx.d, e.stream, dir, .{});
            std.log.info("vision tower loaded on rank 0", .{});
        }
        errdefer if (e.tower) |*t| t.deinit();
        if (draft_dir) |dd| {
            e.lt = try cuda.cublaslt.Library.open();
            errdefer e.lt.?.close();
            e.drafter = try draft_pkg.Package.open(gpa, io, &e.plan, dir, dd, streams, o.context, .{
                .driver = ctx.d,
                .stream = e.stream,
                .tri = e.fwd.tri,
                .torch = e.torch.on(e.stream),
                .entries = &e.entries,
                .comm = if (o.comm) |c| mirror.Coll.of(comm.Comm, c) else null,
                .lt = &e.lt.?,
                .embed = e.prepared.weights.embed,
            });
            errdefer e.drafter.?.deinit();
        }
        // spill v2 (Options.spill_v2): its pinned staging (the write ring, the read-ahead slots, the small states') taken
        // before the pool is sized, so the pool and the kept-state budget are what it leaves and it never takes pool rows
        // mid-run. Sized for the largest index ring the caches can end with (the final ring is set below).
        // TF_GLM_KV_SHARD plans the same slots: a slot holds a rank's largest block, its own part's (the other part's
        // are smaller).
        e.tier_staging = null;
        e.tier_events = &.{};
        if (o.spill_v2) |sv| {
            const ring = @max(e.caches.ring, G.indexRing(std.math.maxInt(usize), 2048));
            const l = try spill_state.Layout.init(e.g, ring, if (e.drafter) |p| .{ .kv = p.forward.rank.kv, .ring = dflash_config.Config.ring } else null);
            const p = try spill_tier.plan(sv, &l);
            e.tier_staging = try cuda.HostBuffer.alloc(ctx.d, @intCast(p.bytes() + 4096));
            std.log.info("rank {d}: spill v2's pinned staging, {d} MiB (a write ring of {d} blocks, {d} read-ahead blocks, {d} small states and {d} restores' at {d} MiB each), taken before the KV pool", .{ o.rank, p.bytes() >> 20, p.write, p.read, p.small, p.restore, p.small_slot >> 20 });
        }
        errdefer if (e.tier_staging) |*b| b.free();
        // Size after the fixed Zig arenas and drafter: free memory at startup, less the recipe reserve.
        // Cache surplus has the same cap as Python. No fictitious kept-state charge: this engine has no prompt cache.
        var pool_rows = extent;
        const available = try capacity.room(gpa, io, ctx);
        const available_room = available.bytes;
        if (streams > 1) {
            if (display.env("TF_GLM_POOL_ROWS")) |t| {
                const n = std.fmt.parseInt(usize, t, 10) catch return error.BadPoolRows;
                if (n < extent) return error.BadPoolRows;
                pool_rows = std.mem.alignForward(usize, n, G.extent);
            } else {
            // logged where the room sizes the pool (a lone stream or TF_GLM_POOL_ROWS does not use it)
            if (available.parked >= 1 << 28) std.log.info("rank {d}: the KV pool's budget counts the {d:.2} GiB free on the per-CPU page lists beyond their resting level (neither MemFree nor MemAvailable does)", .{ o.rank, @as(f64, @floatFromInt(available.parked)) / (1 << 30) });
            const spare = @min(available_room, try capacity.gibBytes(display.env("TF_GLM_CACHE_GIB") orelse display.env("KV_POOL_GIB"), 3));
            const delta = 65536;
            const bytes0 = state.Caches.bytes(extent, streams, 2048) + fwd.Scratch.bytes(e.g, window, decodeScratchRows(extent));
            const bytes1 = state.Caches.bytes(extent + delta, streams, 2048) + fwd.Scratch.bytes(e.g, window, decodeScratchRows(extent + delta));
            const per_token = @max(1, (bytes1 - bytes0 + delta - 1) / delta);
            pool_rows = try capacity.rows(extent, spare, display_cfg.bytes, per_token);
            // Piecewise selection scratch grows at pool buckets: check the actual Zig layout, then shrink whole extents.
            while (state.Caches.bytes(pool_rows, streams, 2048) + fwd.Scratch.bytes(e.g, window, decodeScratchRows(pool_rows)) - bytes0 > spare + display_cfg.bytes) {
                if (pool_rows <= extent) return error.PoolMemoryBudget;
                pool_rows -= G.extent;
            }
            }
            if (o.comm) |c| {
                // one word out (this rank's proposal) and one in per rank: every rank takes the smallest
                const w: usize = c.world;
                const all = try gatherWord(ctx, c, e.stream, @intCast(pool_rows));
                pool_rows = @intCast(agreedPool(all[0..w]));
            }
            if (o.kv_shard != null) {
                // TF_GLM_KV_SHARD: the latent planes once over the two ranks, in the memory of the agreed pool; a part's
                // owner is base / part, so the ranks check they cut the same parts (each sized them itself)
                const c = o.comm orelse return error.KvShardNeedsTwoRanks;
                const unsharded = pool_rows;
                const p = try capacity.sharded(unsharded, e.context, window, ShardCost{ .g = e.g, .streams = streams, .window = window, .fetch = shard_prompt.bytes(e.context, e.prefill_rows) });
                const all = try gatherWord(ctx, c, e.stream, p.rows);
                if (!shard_mod.same(all[0..c.world])) return error.KvShardPoolMismatch;
                e.shard = try shard_mod.Shard.init(o.rank, o.world, p.rows, shard_mod.trash);
                pool_rows = p.rows;
                if (o.rank == 0) {
                    std.log.info("TF_GLM_KV_SHARD: the latent cache once over the {d} ranks: a pool of {d} tokens in {d} parts of {d} (unsharded {d}), the same memory", .{ o.world, p.rows, o.world, p.part, unsharded });
                    if (p.context < e.context) std.log.info("TF_GLM_KV_SHARD: a request holds at most {d} tokens (a part of {d} less a {d}-row window), not {d}", .{ p.context, p.part, window, e.context });
                }
                e.context = p.context;
            }
            // Free the provisional single-window planes before their final allocation; graphs have not captured yet.
            e.caches.deinit();
            caches_live = false;
            e.caches = if (e.shard) |sh| try state.Caches.initShard(gpa, io, ctx.d, ctx.device, streams, 2048, display_cfg, sh) else try state.Caches.initDisplay(gpa, io, ctx.d, ctx.device, pool_rows, streams, 2048, display_cfg);
            caches_live = true;
            if (e.shard) |sh| {
                // the decode tables with the glue planes (cuda_segments), in place before any graph captures them
                var t = try seg.SegRows.initShard(gpa, window, 8, e.caches.ring, sh);
                errdefer t.deinit();
                try t.initDevice(ctx.d);
                e.tables.deinit();
                e.tables = t;
            }
            scratch.deinit();
            scratch_live = false;
            scratch = try fwd.Scratch.init(ctx.d, e.g, window, decodeScratchRows(pool_rows));
            scratch_live = true;
            e.fwd.s = scratch;
            if (e.shard) |sh| {
                // owner decode's kernels, exchange buffers and side stream, before any graph captures them
                e.kv_dec = try shard_dec.Decode.init(ctx.d, o.comm orelse return error.KvShardNeedsTwoRanks, sh, window, &e.tables);
                e.fwd.shard = &e.kv_dec.?;
                // the prompt chunks' history fetch: the copy sized from the context it serves (at most the one counted)
                e.kv_prompt = try shard_prompt.Fetch.init(ctx.d, o.comm.?, sh, e.context, e.prefill_rows);
                e.pe.shard = &e.kv_prompt.?;
            }
        }
        e.pool = if (e.shard != null) try pool_mod.Pool.initParts(gpa, pool_rows, 2) else try pool_mod.Pool.init(gpa, pool_rows);
        errdefer e.pool.deinit();
        e.pool_free = .init(pool_rows);
        std.log.info("GLM KV pool: {d} tokens, display {d} MiB ({s})", .{ pool_rows, if (streams > 1) display_cfg.bytes >> 20 else 0, @tagName(display_cfg.kind) });
        if (!o.eager) {
            if (e.drafter) |p| try p.forward.capture();
            var home = try state.State.bind(e.slots, e.caches, 0, 0, G.extent);
            try e.verify.captureAll(&home);
        }
        return e;
    }

    pub fn deinit(e: *Engine) void {
        const gpa = e.gpa;
        e.spillStop();
        for (e.tier_events) |*ev| ev.deinit();
        gpa.free(e.tier_events);
        if (e.tier_staging) |*b| b.free();
        e.spill_hit.deinit(e.gpa);
        if (e.spill) |t| {
            t.deinit();
            gpa.destroy(t);
        }
        if (e.mtp) |*m| m.deinit();
        if (e.tower) |*t| t.deinit();
        if (e.drafter) |p| p.deinit();
        for (&e.gmasks) |*g| if (g.*) |*m| m.deinit();
        if (e.lt) |*lt| lt.close();
        e.sampler.deinit();
        e.verify.deinit();
        if (e.side) |*sd| sd.deinit();
        if (e.pe.lane) |*l| l.deinit();
        if (e.fwd.pf) |p| {
            p.deinit();
            gpa.destroy(p);
        }
        e.pe.s.deinit();
        e.pe.b.deinit();
        e.fwd.hc.deinit();
        e.memo.deinit();
        if (e.kv_dec) |*k| k.deinit();
        if (e.kv_prompt) |*k| k.deinit();
        e.fwd.s.deinit();
        e.fwd.b.deinit();
        e.tables.deinit();
        e.caches.deinit();
        e.pool.deinit();
        e.slots.deinit();
        e.torch.deinit();
        e.entries.deinit();
        e.kda_k.deinit();
        e.set.deinit();
        e.stream.deinit();
        e.prepared.deinit();
        e.plan.deinit();
        e.cat.deinit();
        e.cfg.deinit();
        gpa.destroy(e);
    }

    /// One more sequence's slot estimate (forward.slot_bytes) plus one stream's DSA cache share.
    /// The cache term is an admission estimate, not a measured allocator total. TF_GLM_KV_SHARD: its latent rows on
    /// one rank of the two, a rank's share half of them.
    pub fn seqBytes(e: *const Engine, rows: usize) usize {
        const extent = std.mem.alignForward(usize, @max(rows, 1), G.extent);
        const slot = kda.slotBytes(G.kda_layers, e.g.lin_heads, e.window, e.g.projWidth()).slot();
        const ring = G.indexRing(extent, @max(e.window, e.prefill_rows));
        const latent = 11 * extent * G.latent_row / (if (e.shard) |s| s.world else 1);
        const caches = latent + 11 * (extent / 4 + 2) * G.pool_row + 11 * 2 * ring * 128 * 2;
        const mtp_bytes = if (e.mtp) |m| m.cache.arena.buf.len + m.b.arena.buf.len + m.arena.buf.len else 0;
        return slot + caches + mtp_bytes;
    }

    pub fn newSeq(e: *Engine, limit_rows: usize) !*Seq {
        if (limit_rows == 0 or limit_rows > e.context + e.window) return error.ContextFull;
        const slot = for (e.used, 0..) |on, i| {
            if (!on and i < e.slots.count) break i;
        } else return error.NoFreeSlot;
        const extent = std.mem.alignForward(usize, limit_rows, G.extent);
        // TF_GLM_KV_SHARD: a stream lives in one part (its owner's latent rows), placed where fewer streams live
        if (e.shard) |sh| if (extent > sh.part) return error.ContextFull;
        const eid = (try e.pool.alloc(extent)) orelse blk: {
            // TF_GLM_KV_SHARD: rows lent to a prompt no fill holds now come back first (multi.py _room's _revoke_twins)
            if (e.revokeTwins() == 0) return error.NoPoolExtent;
            break :blk (try e.pool.alloc(extent)) orelse return error.NoPoolExtent;
        };
        errdefer e.pool.remove(eid) catch unreachable;
        if (e.shard != null) (try e.pool.get(eid)).owner = slot;
        const base = (try e.pool.get(eid)).base;
        e.seqs[slot] = .{ .slot = slot, .st = try state.State.bind(e.slots, e.caches, slot, base, extent), .alive = true, .eid = eid };
        try e.zeroState(&e.seqs[slot].st);
        e.used[slot] = true;
        e.pool_free.store(e.pool.freeRows(), .release);
        return &e.seqs[slot];
    }

    /// The recurrent and convolution state of a sequence at its start (zero), and MTP's cursor. The clears go on the
    /// engine's stream, after the rows a used sequence may still have in flight there and before the next prefill's.
    fn zeroState(e: *Engine, st: *state.State) !void {
        const n = state.Slots.layout(e.g, 1, e.slots.rows);
        const rec: cuda.DeviceBuffer = .{ .d = e.stream.d, .ptr = st.fixed.rec, .len = n[0] };
        const conv: cuda.DeviceBuffer = .{ .d = e.stream.d, .ptr = st.fixed.conv, .len = n[1] };
        try rec.fill8(0, e.stream.handle);
        try conv.fill8(0, e.stream.handle);
        if (e.mtp) |*m| m.cursor = .{};
    }

    pub fn freeSeq(e: *Engine, s: *Seq) void {
        if (e.bound == s) e.bound = null;
        if (s.slot < e.used.len) e.used[s.slot] = false;
        if (s.eid) |eid| {
            // TF_GLM_KV_SHARD: a filling prompt's borrowed twins go back first
            if (e.shard != null and e.pool.hasTwins(eid)) e.pool.giveBack(eid) catch unreachable;
            e.pool.remove(eid) catch unreachable;
            s.eid = null;
            e.pool_free.store(e.pool.freeRows(), .release);
        }
        s.alive = false;
    }

    // -- TF_GLM_KV_SHARD: borrowing, a resume's latent rows (cuda_shard_keep.zig); every rank makes the same calls ----

    /// TF_GLM_KV_SHARD_TWINS: fresh sequence `s`'s prompt fills with free rows of the other part borrowed, so both ranks
    /// hold its latent rows and attend as unsharded (nothing sent); held until `giveRows`. False: off, not fresh, or no
    /// room there (the prompt fills sharded, the history fetched).
    pub fn borrowRows(e: *Engine, s: *Seq) !bool {
        if (e.shard == null or !e.kv_switches.twins) return false;
        const eid = s.eid orelse return false;
        if (s.st.pos != 0) return false;
        const twin = (try shard_keep.borrow(&e.pool, eid)) orelse return false;
        errdefer e.pool.giveBack(eid) catch unreachable;
        s.st.cache = try e.caches.viewBorrowed(s.st.base, s.st.capacity, s.slot, twin);
        e.kv_counts.twins_made += 1;
        e.pool_free.store(e.pool.freeRows(), .release);
        return true;
    }

    /// The rows `borrowRows` lent, back to the pool (the prompt is in, or a placement needs them): `s` goes on with its
    /// owner's rows alone. Nothing when it holds none.
    pub fn giveRows(e: *Engine, s: *Seq) void {
        const eid = s.eid orelse return;
        if (e.shard == null or !e.pool.hasTwins(eid)) return;
        e.pool.giveBack(eid) catch unreachable;
        s.st.cache = e.caches.view(s.st.base, s.st.capacity, s.slot) catch unreachable; // (bound so before the borrow)
        e.pool_free.store(e.pool.freeRows(), .release);
    }

    /// Every twin no fill holds back to the pool, its prompt going on with the owner's rows alone: how many prompts
    /// gave theirs (a fill holds its twin from start to end, so a placement between fills finds none in practice).
    fn revokeTwins(e: *Engine) usize {
        if (e.shard == null) return 0;
        var n: usize = 0;
        for (e.used[0..e.slots.count], e.seqs[0..e.slots.count]) |on, *s| {
            if (!on) continue;
            const eid = s.eid orelse continue;
            if (!shard_keep.revocable(&e.pool, eid)) continue;
            e.giveRows(s);
            e.kv_counts.twins_revoked += 1;
            n += 1;
        }
        return n;
    }

    /// A prefill about to resume kept state `snap` into fresh sequence `s` (cuda_shard_keep.resumeIn): the extent stays,
    /// moves into the state's home part (rebound before any row is written), or keeps its part and the rows are carried
    /// at the restore; false: it must start from 0 (carry off, or the state under carry_min). Every rank alike, before
    /// rank 0 records the placement. Off: true.
    pub fn placeResume(e: *Engine, s: *Seq, snap: *const Snap) !bool {
        if (e.shard == null) return true;
        const eid = s.eid orelse return error.NoPoolExtent;
        const home = snap.home orelse return error.KvShardNoHome;
        switch (try shard_keep.resumeIn(&e.pool, eid, home, snap.pos, e.kv_switches.carry)) {
            .here, .carry => return true,
            .move => |base| {
                _ = try e.pool.move(eid, base, null, true);
                const x = try e.pool.extent(eid);
                const at = s.st.pos;
                const cur = s.st.cur;
                s.st = try state.State.bind(e.slots, e.caches, s.slot, x.base, x.size);
                s.st.pos = at;
                s.st.cur = cur;
                return true;
            },
            .recompute => |why| {
                if (why == .small) e.kv_counts.refused_small += 1;
                return false;
            },
        }
    }

    /// TF_GLM_KV_SHARD_CARRY: a resume's latent rows from the home owner's kept state (`send`) into the new extent's
    /// owner's planes (`recv`), through the prompt fetch's copy (idle between prompt chunks), on the engine's stream;
    /// counted on both ranks.
    fn carryIn(e: *Engine, role: shard_keep.Latent, s: *const Seq, snap: *const Snap, buf: cuda.DeviceBuffer) !void {
        const f = &(e.kv_prompt orelse return error.KvShardNeedsTwoRanks);
        if (role == .send and !snap.sec.rows) return error.NoLatentRows;
        if (role == .recv and !s.st.cache.hasLatent()) return error.NoLatentRows;
        var src: [G.dsa_layers]u64 = @splat(0);
        if (snap.sec.rows) for (&src, 0..) |*p, i| {
            p.* = buf.ptr + snap.sec.latent + i * snap.sec.latentBytes();
        };
        var dev: CarryDev = .{ .e = e, .f = f };
        _ = try shard_keep.carry(&dev, role, src, s.st.cache.latent, snap.pos, f.b.p.copy, @min(f.lim.piece, f.lim.hist + f.lim.chunk));
        e.kv_counts.carries += 1;
        e.kv_counts.moved_tokens += snap.pos;
        e.kv_counts.moved_bytes += shard_mod.carryBytes(snap.pos);
        if (e.g.rank == 0 and snap.pos >= 8192) std.log.info("kv shard carry: {d} tokens, part {d} -> {d}", .{ snap.pos, snap.home.?, e.pool.partOf(s.st.base) });
    }

    /// TF_GLM_KV_SHARD: where sequence `s`'s extent lies, as rank 0 records it for rank 1 (null: off, or none); with
    /// TF_GLM_KV_SHARD_CHECK the pool's layout digest too, counted in `checks`.
    pub fn placement(e: *Engine, s: *const Seq) ?shard_mod.Placement {
        if (e.shard == null) return null;
        const eid = s.eid orelse return null;
        const digest = e.kv_switches.check != 0;
        const p = shard_mod.Placement.of(&e.pool, eid, digest) catch return null;
        if (digest) e.kv_counts.checks += 1;
        return p;
    }

    /// Rank 1: rank 0's placement against this rank's own. An owner derives from the base, so a placement of its own
    /// elsewhere would read and write the other rank's rows as this one's: KvShardPlacement ends the pair instead.
    pub fn checkPlacement(e: *Engine, p: shard_mod.Placement) !void {
        if (e.shard == null) return error.KvShardPlacement;
        if (p.digest != 0) e.kv_counts.checks += 1;
        p.verify(&e.pool) catch |err| {
            std.log.err("TF_GLM_KV_SHARD: rank 0 placed slot {d}'s extent at {d} ({d} rows, part {d}{s}); this rank's pool differs, so the ranks would read each other's rows", .{ p.owner, p.base, p.size, p.part, if (p.digest != 0) ", with its layout's digest" else "" });
            return err;
        };
    }

    /// TF_GLM_KV_SHARD_CHECK, rank 0: the pool's layout digest at the end of decode round `round`, once every that
    /// many rounds (null: not this round, or off), for rank 1 to compare with its own (`checkDigest`).
    pub fn roundDigest(e: *Engine, round: u64) ?u64 {
        if (e.shard == null or !e.kv_switches.digestAt(round)) return null;
        e.kv_counts.checks += 1;
        return e.pool.digest();
    }

    /// Rank 1: rank 0's layout digest against this rank's own pool: a drift between placements (a release, a twin)
    /// ends the pair too.
    pub fn checkDigest(e: *Engine, digest: u64) !void {
        if (e.shard == null) return error.KvShardPlacement;
        e.kv_counts.checks += 1;
        shard_mod.checkLayout(&e.pool, digest) catch |err| {
            std.log.err("TF_GLM_KV_SHARD: rank 0's pool layout differs from this rank's after the same calls, so the ranks would read each other's rows", .{});
            return err;
        };
    }

    /// TF_GLM_KV_SHARD's stats for /health and /metrics (lanes.kv_parts.Source: read on the lane thread).
    pub fn readKvParts(ctx: *anyopaque, out: *lanes.kv_parts.Stats) void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        kvPartsOf(&e.pool, e.kv_counts, out);
        if (e.kv_prompt) |*f| promptOf(f.stats, out);
        if (e.kv_dec) |*d| {
            out.owner_rounds = d.stats.rounds;
            out.owner_bytes = d.stats.bytes;
        }
    }

    /// HTTP threads read the published count, never the lane thread's mutable extent list.
    pub fn poolStatus(e: *const Engine) struct { pool_tokens: usize, pool_free_tokens: usize } {
        return .{ .pool_tokens = e.pool.rows, .pool_free_tokens = e.pool_free.load(.acquire) };
    }

    pub fn bind(e: *Engine, s: *Seq) void {
        e.bound = s;
    }

    pub fn setSampling(e: *Engine, s: ?lanes.Sampling) void {
        e.sampling = if (s) |x| (if (x.temperature > 0) x else null) else null;
    }

    pub fn pos(e: *Engine) u64 {
        return if (e.bound) |s| @intCast(s.st.pos) else 0;
    }

    /// glm5_next/cuda/decode.py prefill: the prompt in 2048-row chunks through the prompt path (cuda_prefill), then the
    /// first token from the last row's logits. `resume_from`: a kept state (takeSnap) of a prefix of the prompt on the
    /// 64-token grid, put back first; the chunks then start there. `keep_at`: a grid point where the state is kept
    /// (the result's `kept`); at the prompt's end nothing is sampled (`first` is 0), the pass only fills the caches.
    pub fn prefillWith(e: *Engine, prompt: []const u32, sampling: ?lanes.Sampling, o: PrefillOptions) !Prefilled {
        const s = e.bound orelse return error.NoBoundSequence;
        if (prompt.len == 0 or prompt.len > e.context) return error.ContextFull;
        if (o.keep_at) |k| if (k == 0 or k > prompt.len or k % 64 != 0) return error.BadKeepPoint;
        if (o.media) |m| {
            try e.attachMedia(m, prompt.len);
        }
        defer e.detachMedia();
        const st = &s.st;
        const ids = try e.gpa.alloc(i32, prompt.len);
        defer e.gpa.free(ids);
        for (prompt, ids) |t, *d| {
            if (t >= 154880) return error.BadToken;
            d.* = @intCast(t);
        }
        var begin: usize = 0;
        if (o.live_from) |p| {
            // the sequence's own rows up to p (a refused save left it there): no restore
            if (o.resume_from != null or p == 0 or p >= prompt.len or p % 64 != 0 or st.pos != p) return error.ResumeMustExtend;
            begin = p;
        } else if (o.resume_from) |snap| {
            if (snap.pos == 0 or snap.pos >= prompt.len or snap.pos % 64 != 0) return error.ResumeMustExtend;
            if (!(s.tag == snap.id and st.pos == snap.pos)) try e.restoreSnap(s, snap);
            begin = snap.pos;
        } else if (st.pos != 0) {
            // from the start on a sequence that ran rows already: its recurrent state is the start's again (position
            // 0 does not clear it by itself: the KDA kernels read the state they are given)
            try e.zeroState(st);
            st.pos = 0;
            s.tag = 0;
        }
        e.prompt_tokens = prompt;
        e.prompt_at = 0;
        defer e.prompt_tokens = &.{};
        if (e.mtp) |*m| try m.reset(e.gpa, prompt);
        if (e.drafter) |p| {
            try p.multi.reset(s.slot, prompt, .{});
            p.multi.contexts[s.slot].end = begin; // the window of rows [0, begin) is the restored (or live) one
        }
        e.pe.cache = st.cache;
        e.pe.conv = st.fixed.conv;
        e.pe.rec = st.fixed.rec;
        e.pe.pos_dev = st.fixed.pos_dev;
        var kept: ?*Snap = null;
        errdefer if (kept) |k| e.freeSnapshot(k);
        if (o.keep_at) |k| if (k > begin) {
            _ = try prefill_loop.run(&e.pe, &st.cur, .{ .tokens = ids[0..k], .begin = begin, .limit = e.prefill_rows, .head = false, .tap_ctx = e, .on_taps = promptTaps });
            st.pos = k;
            // the save may be refused (the keep floor, or no device memory): never the prompt, which goes on from its
            // live rows (the lanes continue from them, `live_from`; the ranks agreed on this keep from their free
            // memory first, cuda_lanes pass, so this is the rare case of memory going in between)
            kept = e.takeSnap(s, k) catch |err| switch (err) {
                error.BelowMemoryReserve, error.OutOfDeviceMemory => blk: {
                    std.log.warn("prompt cache: no state kept at {d} tokens ({s})", .{ k, @errorName(err) });
                    break :blk null;
                },
                else => return err,
            };
            begin = k;
        };
        if (begin == prompt.len) return .{ .first = 0, .kept = kept };
        // The reference keeps the prompt's state on the 64-row grid (engine.grid_point) and resumes from it to the end, so
        // the last chunk is cut there: prompt rows .. B, then B .. n (a 5021-token prompt: 2048, 2048, 896, then 29).
        const point = if (prompt.len % 64 == 0) prompt.len else prompt.len / 64 * 64;
        if (point > begin and point < prompt.len) _ = try prefill_loop.run(&e.pe, &st.cur, .{ .tokens = ids[0..point], .begin = begin, .limit = e.prefill_rows, .head = false, .tap_ctx = e, .on_taps = promptTaps });
        _ = try prefill_loop.run(&e.pe, &st.cur, .{ .tokens = ids, .limit = e.prefill_rows, .begin = if (point < prompt.len and point > begin) point else begin, .head = true, .tap_ctx = e, .on_taps = promptTaps });
        st.pos = prompt.len;
        var out: [1]u32 = undefined;
        const rule: ?lanes.Sampling = if (sampling) |x| (if (x.temperature > 0) x else null) else null;
        if (o.mask.len > 0) try e.maskRows(0, e.pe.b.logits, 1, o.mask);
        try e.sampler.samplePacked(&.{.{ .logits = e.pe.b.logits, .positions = &.{@as(u64, prompt.len)}, .sampling = rule, .out = &out }});
        return .{ .first = out[0], .kept = kept };
    }

    // -- kept prompt states (cuda_snapshot.zig) -----------------------------------------------------------------------

    /// This engine's state buffer for `at` tokens: rows, rings, recurrent states and (with drafts) DFlash2's rings.
    fn sectionsFor(e: *const Engine, at: usize) snapshot.Sections {
        return e.sectionsHeld(at, true);
    }

    /// `sectionsFor`, the latent rows held or not (TF_GLM_KV_SHARD: only the stream's owner holds them).
    fn sectionsHeld(e: *const Engine, at: usize, rows: bool) snapshot.Sections {
        const ring = e.caches.ring;
        const draft_bytes: usize = if (e.drafter) |p| draftOf(p, 0).bytes() else 0;
        return snapshot.sectionsOf(e.g, at, ring, draft_bytes, rows);
    }

    fn draftOf(p: *const draft_pkg.Package, slot: usize) snapshot.Draft {
        var d: snapshot.Draft = .{ .kc = undefined, .vc = undefined, .kv = p.forward.rank.kv, .ring = dflash_config.Config.ring, .cap = p.forward.streams * dflash_config.Config.ring + 64, .slot = slot };
        d.kc = p.forward.kc;
        d.vc = p.forward.vc;
        return d;
    }

    /// What a kept state of `at` tokens holds on the device (TF_GLM_KV_SHARD: on its home part's owner; the budget
    /// counts that on every rank, the other rank's state being the latent rows smaller).
    pub fn snapBytes(e: *const Engine, at: usize) usize {
        return e.sectionsFor(at).total;
    }

    /// Spill v2's layout of this engine's kept states (cuda_spill_state): its geometry, index ring and drafter;
    /// owner_sharded under TF_GLM_KV_SHARD (a state's latent rows on its home part's owner only).
    pub fn spillLayout(e: *const Engine) !spill_state.Layout {
        const l = try spill_state.Layout.init(e.g, e.caches.ring, if (e.drafter) |p| .{ .kv = p.forward.rank.kv, .ring = dflash_config.Config.ring } else null);
        return if (e.shard != null) l.sharded() else l;
    }

    // -- spill v2's device side (cuda_spill_tier): copies on the engine's stream, one event a staging slot ----------

    /// The pinned staging taken before the pool (Options.spill_v2), 4 KiB aligned; empty without spill v2.
    pub fn tierStaging(e: *const Engine) []align(4096) u8 {
        const b = e.tier_staging orelse return &.{};
        const off = std.mem.alignForward(usize, @intFromPtr(b.bytes.ptr), 4096) - @intFromPtr(b.bytes.ptr);
        return @alignCast(b.bytes[off..]);
    }

    pub fn tierKept(_: *const Engine, snap: *const Snap) spill_state.Kept {
        return .{ .buf = snap.buf.?.ptr, .sec = snap.sec, .home = snap.home };
    }

    fn tierD2h(ptr: *anyopaque, dst: []u8, src: u64) anyerror!void {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        try e.stream.d.check(e.stream.d.api.cuMemcpyDtoHAsync_v2(dst.ptr, src, dst.len, e.stream.handle), "cuMemcpyDtoHAsync");
    }

    fn tierH2d(ptr: *anyopaque, dst: u64, src: []const u8) anyerror!void {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        try e.stream.d.check(e.stream.d.api.cuMemcpyHtoDAsync_v2(dst, src.ptr, src.len, e.stream.handle), "cuMemcpyHtoDAsync");
    }

    fn tierFill(ptr: *anyopaque, dst: u64, byte: u8, n: usize) anyerror!void {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        try e.stream.d.check(e.stream.d.api.cuMemsetD8Async(dst, byte, n, e.stream.handle), "cuMemsetD8Async");
    }

    /// The tier's copies: pinned staging to and from kept-state buffers, in the engine stream's order.
    pub fn tierCopier(e: *Engine) spill_state.Copier {
        return .{ .ptr = e, .d2h = tierD2h, .h2d = tierH2d, .fill = tierFill };
    }

    /// `n` events, one a staging slot and one for the tier's own waits (made once, when the tier opens).
    pub fn tierEvents(e: *Engine, n: usize) !void {
        if (e.tier_events.len >= n) return;
        const evs = try e.gpa.alloc(cuda.Event, n);
        var made: usize = 0;
        errdefer {
            for (evs[0..made]) |*ev| ev.deinit();
            e.gpa.free(evs);
        }
        while (made < n) : (made += 1) evs[made] = try cuda.Event.init(e.stream.d, false);
        for (e.tier_events) |*ev| ev.deinit();
        e.gpa.free(e.tier_events);
        e.tier_events = evs;
    }

    /// Event `ev` after what the tier just enqueued; a record that fails waits for the stream instead (the event then
    /// reads as complete).
    pub fn tierRecord(e: *Engine, ev: usize) void {
        e.tier_events[ev].record(e.stream) catch e.stream.synchronize() catch {};
    }

    pub fn tierDone(e: *Engine, ev: usize) bool {
        return e.tier_events[ev].done() catch true;
    }

    pub fn tierWait(e: *Engine, ev: usize) void {
        e.tier_events[ev].synchronize() catch e.stream.synchronize() catch {};
    }

    /// The smallest of every rank's `mine` (one all-gather; one rank: its own). Every rank calls it at the same point.
    pub fn smallestOverRanks(e: *Engine, mine: u64) !u64 {
        const c = e.comm orelse return mine;
        const all = try gatherWord(e.ctx.?, c, e.stream, mine);
        return std.mem.min(u64, all[0..c.world]);
    }

    /// A restore's destination: a new kept-state buffer for `at` tokens, refused under the keep floor (as a save is).
    /// TF_GLM_KV_SHARD: the state of part `home` as takeSnap leaves it on this rank (its latent rows on the part's owner
    /// only), its home recorded for the resume's placement (placeResume).
    pub fn tierDest(e: *Engine, at: usize, home: ?u8) !*Snap {
        const rows = if (e.shard) |sh| (home orelse return error.KvShardNoHome) == sh.rank else if (home != null) return error.KvShardOff else true;
        const sec = e.sectionsHeld(at, rows);
        if (e.keep_reserve > 0) if (cache_plan.memNow()) |m| if (!cache_plan.fits(m.available, e.keep_reserve, sec.total)) return error.BelowMemoryReserve;
        const snap = try e.gpa.create(Snap);
        errdefer e.gpa.destroy(snap);
        const buf = try cuda.DeviceBuffer.alloc(e.stream.d, sec.total);
        e.snap_next += 1;
        snap.* = .{ .id = e.snap_next, .pos = at, .buf = buf, .sec = sec, .home = home };
        return snap;
    }

    /// TF_GLM_KV_SHARD: whether a restore of a state of `at` tokens kept in part `home` is worth its reads now (its
    /// resume could be placed or carried: cuda_shard_keep.restorable); rank 0 asks before any read. Off: true.
    pub fn tierResumable(e: *const Engine, at: usize, home: u8) !bool {
        if (e.shard == null) return true;
        return shard_keep.restorable(&e.pool, home, at, e.kv_switches.carry);
    }

    fn copyBytes(ptr: *anyopaque, dst: u64, src: u64, n: usize) anyerror!void {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        // a DRM display span is two host registrations, and rows of a latent plane can lie on both sides of the cut
        const cut: u64 = if (e.caches.span) |v| (if (v.base != 0) v.ptr + v.ordinary else 0) else 0;
        if (cut == 0) return e.stream.d.check(e.stream.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, e.stream.handle), "cuMemcpyDtoDAsync");
        for (display.copyPieces(cut, dst, src, n)) |p| if (p.len > 0)
            try e.stream.d.check(e.stream.d.api.cuMemcpyDtoDAsync_v2(dst + p.off, src + p.off, p.len, e.stream.handle), "cuMemcpyDtoDAsync");
    }

    fn copier(e: *Engine) snapshot.Copier {
        return .{ .ptr = e, .copy = copyBytes };
    }

    /// Whether a kept state of `bytes` leaves the keep floor (the memory reserve) alone now: the
    /// ranks ask this before they agree to keep at a mark, so a save that would cross it is refused on every rank
    /// instead of failing one rank's prefill after the agreement.
    pub fn keepFits(e: *const Engine, bytes: u64) bool {
        if (e.keep_reserve == 0) return true;
        const m = cache_plan.memNow() orelse return true;
        return cache_plan.fits(m.available, e.keep_reserve, bytes);
    }

    /// The bound sequence's state after `at` tokens (its position), copied into a new buffer. TF_GLM_KV_SHARD: the
    /// latent rows only on the rank that owns the stream's extent (a borrowed twin's are the other rank's), its home.
    pub fn takeSnap(e: *Engine, s: *Seq, at: usize) !*Snap {
        if (at == 0 or at != s.st.pos or at % 64 != 0) return error.BadKeepPoint;
        const sec = e.sectionsHeld(at, if (e.shard) |sh| sh.owns(s.st.base) else true);
        if (e.keep_reserve > 0) if (cache_plan.memNow()) |m| if (!cache_plan.fits(m.available, e.keep_reserve, sec.total)) return error.BelowMemoryReserve;
        const snap = try e.gpa.create(Snap);
        errdefer e.gpa.destroy(snap);
        var buf = try cuda.DeviceBuffer.alloc(e.stream.d, sec.total);
        errdefer buf.free();
        const draft: ?snapshot.Draft = if (e.drafter) |p| draftOf(p, s.slot) else null;
        if (draft != null) if (e.drafter.?.multi.contexts[s.slot].end != at) return error.BadDraftCommit;
        try snapshot.save(e.copier(), e.g, sec, buf.ptr, &s.st, draft);
        try e.stream.synchronize();
        e.snap_next += 1;
        snap.* = .{ .id = e.snap_next, .pos = at, .buf = buf, .sec = sec, .home = if (e.shard) |sh| @intCast(sh.owner(s.st.base)) else null };
        s.tag = snap.id;
        return snap;
    }

    /// Put a kept state into the sequence: its rows, rings, recurrent states and the drafter's window; the position.
    /// TF_GLM_KV_SHARD: the latent rows back where the state and the extent are on one rank, else carried from the home
    /// owner to the extent's (placeResume chose); never into a borrowed twin (a resume does not borrow).
    pub fn restoreSnap(e: *Engine, s: *Seq, snap: *const Snap) !void {
        const buf = snap.buf orelse return error.NoStateKept;
        const draft: ?snapshot.Draft = if (e.drafter) |p| draftOf(p, s.slot) else null;
        if ((snap.sec.draft != snap.sec.total) != (draft != null)) return error.DrafterMismatch;
        var rows = true;
        if (e.shard) |sh| {
            if (s.st.cache.twin) return error.KvShardTwinRestore;
            const role = shard_keep.latentRole(sh, snap.home orelse return error.KvShardNoHome, sh.owner(s.st.base));
            rows = role == .copy;
            if (role == .send or role == .recv) try e.carryIn(role, s, snap, buf);
        }
        try snapshot.restore(e.copier(), e.g, snap.sec, buf.ptr, &s.st, draft, rows);
        try e.stream.synchronize();
        s.st.pos = snap.pos;
        s.tag = snap.id;
        if (e.drafter) |p| p.multi.contexts[s.slot].end = snap.pos;
    }

    /// Before any collective of a prefill (the lane layer's prepare step, then its agreement): checks the media and
    /// takes the device buffers for its rows and features, so a refusal or an out-of-memory on one rank stops both.
    pub fn reserveMedia(e: *Engine, m: *const lanes.Media, prompt_len: usize) !void {
        e.detachMedia();
        if (!e.vision) return error.NoVision;
        if (m.width != media_width) return error.BadMedia;
        try m.check(prompt_len);
        if (e.g.rank == 0 and m.features.len == 0 and e.tower == null) return error.NoVisionTower;
        const n = m.rowCount();
        const d = e.stream.d;
        const host = try e.gpa.alloc(u32, n);
        errdefer e.gpa.free(host);
        var rows = try cuda.DeviceBuffer.alloc(d, n * 4);
        errdefer rows.free();
        var feats = try cuda.DeviceBuffer.alloc(d, n * media_width * 2);
        errdefer feats.free();
        e.media = .{ .host_rows = host, .rows = rows, .feats = feats };
    }

    /// The prompt's picture rows on every rank: rank 0 has them (host features, or the tower's output) and
    /// broadcasts their prompt rows and values; rank 1 receives. Both ranks call it at the same point.
    fn attachMedia(e: *Engine, m: *const lanes.Media, prompt_len: usize) !void {
        if (e.media == null) try e.reserveMedia(m, prompt_len);
        const a = e.media.?;
        const n = m.rowCount();
        if (n != a.host_rows.len) return error.BadMedia;
        if (e.g.rank == 0) {
            @memcpy(a.host_rows, m.rows);
            try a.rows.uploadAsync(0, std.mem.sliceAsBytes(a.host_rows), e.stream.handle);
            if (m.features.len > 0) {
                try a.feats.uploadAsync(0, m.features, e.stream.handle);
            } else {
                const t = &(e.tower orelse return error.NoVisionTower);
                const items = try e.gpa.alloc(vision_tower.Item, m.items.len);
                defer e.gpa.free(items);
                for (m.items, items) |it, *slot| slot.* = .{ .frames = it.frames, .h = it.h, .w = it.w, .pixels = it.data };
                try t.encode(items, a.feats.ptr);
            }
        }
        if (e.g.world > 1) {
            const link = e.comm orelse return error.MissingCommunicator;
            try link.broadcast(a.rows.ptr, n * 4, 0, e.stream.handle);
            try link.broadcast(a.feats.ptr, n * media_width * 2, 0, e.stream.handle);
        }
        try e.stream.synchronize();
        if (e.g.rank != 0) try a.rows.download(0, std.mem.sliceAsBytes(a.host_rows));
        e.pe.inject = .{ .ctx = e, .call = injectMedia };
    }

    pub fn detachMedia(e: *Engine) void {
        const a = e.media orelse return;
        e.pe.inject = null;
        e.stream.synchronize() catch {};
        var rows = a.rows;
        var feats = a.feats;
        rows.free();
        feats.free();
        e.gpa.free(a.host_rows);
        e.media = null;
    }

    fn injectMedia(ctx: *anyopaque, x: u64, at: usize, rows: usize) anyerror!void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        const a = e.media orelse return;
        const lo = std.sort.lowerBound(u32, a.host_rows, @as(u32, @intCast(at)), orderU32);
        const hi = std.sort.lowerBound(u32, a.host_rows, @as(u32, @intCast(at + rows)), orderU32);
        if (hi <= lo) return;
        try e.torch.on(e.stream).mediaRows(a.rows.ptr + lo * 4, a.feats.ptr + lo * media_width * 2, x, hi - lo, at, media_width, 4);
    }

    fn orderU32(key: u32, item: u32) std.math.Order {
        return std.math.order(key, item);
    }

    pub fn prefill(e: *Engine, prompt: []const u32, sampling: ?lanes.Sampling) !u32 {
        return (try e.prefillWith(prompt, sampling, .{})).first;
    }

    pub fn manyFit(_: *const Engine, _: []const ManyPrompt) bool {
        return false;
    }

    pub fn prefillMany(e: *Engine, prompts: []const ManyPrompt, firsts: []u32) !void {
        _ = .{ e, prompts, firsts };
        return error.PrefillNotBuilt;
    }

    pub fn copyPrefix(e: *Engine, dst: *Seq, src: *Seq, rows: usize, mtp_rows: usize) !void {
        _ = .{ e, dst, src, rows, mtp_rows };
        return error.PromptCacheNotBuilt;
    }

    // -- the spill tier: a kept state to disk and back (cuda_spill.zig) ---------------------------------------------------

    const spill_piece: usize = 64 << 20;

    /// Make the context current on this thread (the store's calls come before the first backend call on it).
    pub fn current(e: *const Engine) void {
        if (e.ctx) |c| c.makeCurrent() catch {};
    }

    pub fn spillOn(e: *const Engine) bool {
        return e.spill != null;
    }

    /// Write a kept state to this rank's spill folder, in the background: the job waits for the writer thread, which copies
    /// the state off the device and into its file while the lane thread goes on. A refusal (short, stored, over the cap or
    /// the disk floor) or a failure leaves the state where it is; nothing here is fatal.
    pub fn spillOut(e: *Engine, snap: *Snap, tokens: []const u32, at: u32) void {
        const t = e.spill orelse return;
        if (snap.buf == null or snap.pos != at or tokens.len != at) return;
        if (!t.wants(tokens, snap.sec.total)) return;
        const key = spill_mod.keyOf(tokens);
        const copy = e.gpa.dupe(u32, tokens) catch return;
        e.spillLock();
        defer e.spillUnlock();
        for (e.spill_jobs.items) |j| if (j.key == key) return e.gpa.free(copy);
        _ = snap.refs.fetchAdd(1, .acq_rel);
        e.spill_jobs.append(e.gpa, .{ .snap = snap, .tokens = copy, .at = at, .key = key }) catch {
            _ = snap.refs.fetchSub(1, .acq_rel);
            return e.gpa.free(copy);
        };
        _ = e.spill_busy.fetchAdd(1, .acq_rel);
        if (e.spill_thread == null) e.spill_thread = std.Thread.spawn(.{}, spillLoop, .{e}) catch null;
    }

    fn spillLock(e: *Engine) void {
        while (!e.spill_mu.tryLock()) std.Thread.yield() catch {};
    }

    fn spillUnlock(e: *Engine) void {
        e.spill_mu.unlock();
    }

    fn pause() void {
        var ts: std.c.timespec = .{ .sec = 0, .nsec = 2 * std.time.ns_per_ms };
        _ = std.c.nanosleep(&ts, null);
    }

    fn spillLoop(e: *Engine) void {
        e.current();
        while (true) {
            e.spillLock();
            const job: ?SpillJob = if (e.spill_jobs.items.len > 0) e.spill_jobs.orderedRemove(0) else null;
            e.spillUnlock();
            if (job) |j| {
                e.spillWrite(j);
                e.snapRelease(j.snap);
                e.gpa.free(j.tokens);
                _ = e.spill_busy.fetchSub(1, .acq_rel);
            } else if (e.spill_quit.load(.acquire)) {
                return;
            } else pause();
        }
    }

    fn spillWrite(e: *Engine, j: SpillJob) void {
        const t = e.spill orelse return;
        const buf = j.snap.buf orelse return;
        var w = t.begin(j.tokens, j.at, j.snap.sec.total) catch |err| {
            if (err != error.SpillShort and err != error.SpillExists) std.log.info("spill: {d} tokens not written ({s})", .{ j.at, @errorName(err) });
            return;
        };
        const stage = e.gpa.alloc(u8, @min(spill_piece, j.snap.sec.total)) catch return w.abort();
        defer e.gpa.free(stage);
        var off: usize = 0;
        while (off < j.snap.sec.total) {
            const n = @min(stage.len, j.snap.sec.total - off);
            buf.download(off, stage[0..n]) catch |err| return abortWith(&w, "copying off the device", err);
            w.put(stage[0..n]) catch |err| return abortWith(&w, "writing", err);
            off += n;
        }
        w.finish() catch |err| return std.log.warn("spill: finishing {d} tokens failed ({s})", .{ j.at, @errorName(err) });
        if (!@import("builtin").is_test) std.log.info("spill: wrote {d} tokens ({d} MiB)", .{ j.at, j.snap.sec.total >> 20 });
    }

    /// Wait until the writer has nothing queued or running, at most `max_ns`; whether it is idle.
    pub fn spillDrain(e: *Engine, max_ns: u64) bool {
        var t0: std.c.timespec = undefined;
        _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t0);
        while (e.spill_busy.load(.acquire) > 0) {
            var t1: std.c.timespec = undefined;
            _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t1);
            const spent: u64 = @intCast((t1.sec - t0.sec) * std.time.ns_per_s + (t1.nsec - t0.nsec));
            if (spent > max_ns) return false;
            pause();
        }
        return true;
    }

    /// The writer ends: what it has queued is written first, within the clean-stop budget.
    fn spillStop(e: *Engine) void {
        if (e.spill_thread) |th| {
            if (!e.spillDrain(e.spill_wait_ns)) std.log.warn("spill: {d} writes not finished in time; their partial files go at the next start", .{e.spill_busy.load(.acquire)});
            e.spill_quit.store(true, .release);
            th.join();
            e.spill_thread = null;
        }
        e.spill_jobs.deinit(e.gpa);
    }

    fn abortWith(w: *spill_mod.Tier.Writer, what: []const u8, err: anyerror) void {
        std.log.warn("spill: {s} failed ({s})", .{ what, @errorName(err) });
        w.abort();
    }

    /// The longest stored state `prompt` extends, longer than `longer_than`, that this engine can read.
    pub fn spillBest(e: *Engine, prompt: []const u32, longer_than: u32, at: *u32, key: *u64, tokens: *[]const u32) bool {
        const t = e.spill orelse return false;
        const hit = t.bestInto(prompt, longer_than, 64, &e.spill_hit, readable, e) orelse return false;
        at.* = hit.at;
        key.* = hit.key;
        tokens.* = e.spill_hit.items;
        return true;
    }

    /// Whether a stored state of `at` tokens and `payload` bytes is this engine's layout (another build's is not readable).
    fn readable(ctx: *anyopaque, at: u32, payload: u64) bool {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        return payload == e.sectionsFor(at).total;
    }

    /// `tokens`: the state's, owned by the caller (`gpa.free`)
    pub const Stored = struct { snap: *Snap, tokens: []u32 };

    /// Read stored state `key` into a new device buffer.
    pub fn spillIn(e: *Engine, key: u64, at: u32) !Stored {
        const t = e.spill orelse return error.NoSpillTier;
        e.current();
        var r = try t.reader(key);
        defer r.close();
        const sec = e.sectionsFor(at);
        if (r.meta.at != at or r.meta.payload != sec.total) return error.SpillRead;
        if (e.keep_reserve > 0) if (cache_plan.memNow()) |m| if (!cache_plan.fits(m.available, e.keep_reserve, sec.total)) return error.BelowMemoryReserve;
        const snap = try e.gpa.create(Snap);
        errdefer e.gpa.destroy(snap);
        var buf = try cuda.DeviceBuffer.alloc(e.stream.d, sec.total);
        errdefer buf.free();
        const stage = try e.gpa.alloc(u8, @min(spill_piece, sec.total));
        defer e.gpa.free(stage);
        var off: usize = 0;
        while (off < sec.total) {
            const n = @min(stage.len, sec.total - off);
            try r.read(stage[0..n]);
            try buf.upload(off, stage[0..n]);
            off += n;
        }
        e.snap_next += 1;
        snap.* = .{ .id = e.snap_next, .pos = at, .buf = buf, .sec = sec };
        return .{ .snap = snap, .tokens = r.takeTokens() };
    }

    /// The spill tier's numbers for /health.
    pub fn spillStats(e: *Engine, out: *[8]u64) void {
        const t = e.spill orelse return;
        const s = t.snapshot();
        out.* = .{ s.entries, s.bytes, s.writes, s.write_bytes, s.reads, s.read_bytes, s.skipped_short + s.skipped_exists + s.skipped_floor + s.skipped_cap, s.failed + s.crc_failures };
    }

    pub fn freeSnapshot(e: *Engine, snap: *Snap) void {
        e.snapRelease(snap);
    }

    /// One reference to `snap` let go; the last frees the buffer (a background write may be the last).
    fn snapRelease(e: *Engine, snap: *Snap) void {
        if (snap.refs.fetchSub(1, .acq_rel) != 1) return;
        e.current();
        if (snap.buf) |*b| b.free();
        e.gpa.destroy(snap);
    }

    pub fn forward(e: *Engine, windows: []const rounds.Window, table: []const i32) !void {
        _ = table;
        try e.stageVerify(windows);
    }

    pub fn sample(e: *Engine, windows: []const rounds.Window, out: []u32) !void {
        var at: usize = 0;
        for (windows, 0..) |w, i| {
            try e.sampleWindow(i, e.sampling, out[at..][0..w.tokens.len]);
            at += w.tokens.len;
        }
    }

    pub fn stageCommit(e: *Engine, table: []const i32) !void {
        if (table.len == 0 or table.len % 6 != 0 or table.len > e.commit_words.len) return error.BadWindow;
        @memset(&e.commit_words, 0);
        @memcpy(e.commit_words[0..table.len], table);
        e.commit_nseg = table.len / 6;
        const bytes = std.mem.sliceAsBytes(e.commit_words[0 .. e.tables.max_segs * 6]);
        if (e.verify.stager.ready()) {
            // stream-ordered behind the verify graph and the sampling; no drain
            try e.verify.stager.put(e.verify.kseg.ptr, bytes);
        } else {
            try e.stream.synchronize();
            try e.verify.kseg.upload(0, bytes);
        }
    }

    pub fn replay(e: *Engine) !void {
        if (e.commit_nseg == 0) return error.NoCommitTable;
        const h = e.g.lin_heads;
        const layers = G.kda_layers;
        const kv = e.fwd.s.rows * h * 128;
        try (kda.Ops{ .k = &e.kda_k, .s = e.stream }).replayLayersSegments(e.commit_nseg, h, layers, e.verify.kseg.ptr, e.slots.rec, 2 * layers * h * 128 * 128, layers * h * 128 * 128, h * 128 * 128, e.fwd.s.k, e.fwd.s.v, e.fwd.s.gates, e.fwd.s.beta, kv, e.fwd.s.rows * h);
    }

    pub fn convShift(e: *Engine) !void {
        if (e.commit_nseg == 0) return error.NoCommitTable;
        const c = e.g.convDim();
        const width = e.g.projWidth();
        const rows = e.fwd.s.rows;
        try e.fwd.tri.conv_shift_seg(.{ G.kda_layers, (c + 1023) / 1024, e.commit_nseg }, e.slots.conv, e.fwd.s.proj, e.verify.kseg.ptr, @intCast(G.kda_layers * 3 * c), @intCast(3 * c), @intCast(rows * width), @intCast(width), .{ .C = @intCast(c), .TAPS = 3, .COLS = 6, .BLOCK = 1024 });
    }

    pub fn setPosition(e: *Engine, st: *state.State, next: usize) !void {
        if (next > std.math.maxInt(i32)) return error.BadPosition;
        var word: i32 = @intCast(next);
        if (e.verify.stager.ready()) return e.verify.stager.put(st.fixed.pos_dev, std.mem.asBytes(&word));
        const view: cuda.DeviceBuffer = .{ .d = e.stream.d, .ptr = st.fixed.pos_dev, .len = 4 };
        try view.upload(0, std.mem.asBytes(&word));
    }

    pub fn commitDrafts(e: *Engine, windows: []const rounds.Window, keeps: []const usize) !void {
        // Round(Engine).commit has published the accepted positions/parities. Publish the same hidden-row
        // availability as commitWindow so the CLI's next MTP round can absorb those normalized target rows.
        for (windows, 0..) |_, i| e.committed[i] = true;
        if (e.drafter) |p| {
            var total: usize = 0;
            for (windows) |w| total += w.tokens.len;
            try p.multi.commitDraftsStrided(windows, keeps, e.fwd.b.taps, total, e.fwd.b.rows);
        } else if (e.drafts and e.mtp == null) return error.DrafterNotLoaded;
    }

    pub fn verifyMany(e: *Engine, windows: []const Window) !void {
        if (windows.len == 0 or windows.len > 8) return error.BadWindow;
        var spans_src: [8]rounds.Window = undefined;
        for (windows, 0..) |w, i| spans_src[i] = .{ .state = &w.seq.st, .tokens = w.tokens };
        try e.stageVerify(spans_src[0..windows.len]);
    }

    /// `rows` rows of this rank's logits at `logits` with the tokens `mask` (rows x the grammar's words, over the whole
    /// vocabulary) does not allow set to -inf: Python's `constraint.mask(logits, window, vocab_offset)`.
    fn maskRows(e: *Engine, i: usize, logits: u64, rows: usize, mask: []const u32) !void {
        if (rows == 0 or mask.len % rows != 0) return error.InvalidGrammarMasks;
        if (i >= e.gmasks.len) return error.NoSuchWindow;
        // window i's own bit buffer: the bits are uploaded on the default stream while the engine's stream may still
        // run another window's mask kernel over that window's buffer
        if (e.gmasks[i] == null) e.gmasks[i] = try grammar.Mask.load(e.stream.d, e.g.vocab);
        try e.gmasks[i].?.apply(e.gpa, e.stream, logits, rows, e.g.vocab, @intCast(e.g.vocab_start), mask, mask.len / rows);
    }

    pub fn sampleWindow(e: *Engine, i: usize, s: ?lanes.Sampling, out: []u32) !void {
        return e.sampleWindowMasked(i, s, out, &.{});
    }

    /// `sampleWindow` with structured output's bits for the window's rows (empty: none).
    pub fn sampleWindowMasked(e: *Engine, i: usize, s: ?lanes.Sampling, out: []u32, mask: []const u32) !void {
        if (i >= e.round_n) return error.NoSuchWindow;
        const rows = e.round_len[i];
        if (out.len != rows) return error.BadWindow;
        const st = e.round_state(i);
        var positions: [64]u64 = undefined;
        for (0..rows) |r| positions[r] = @as(u64, @intCast(st.pos)) + 1 + @as(u64, r);
        const rule: ?lanes.Sampling = if (s) |x| (if (x.temperature > 0) x else null) else null;
        if (mask.len > 0) try e.maskRows(i, e.fwd.b.logits + e.round_row0[i] * e.g.vocab * 2, rows, mask);
        try e.sampler.samplePacked(&.{.{
            .logits = e.fwd.b.logits + e.round_row0[i] * e.g.vocab * 2,
            .positions = positions[0..rows],
            .sampling = rule,
            .out = out,
        }});
    }

    /// Every window's draws in one packed sampler call (one gather and one readback for the round, where
    /// sampleWindow does one per window): `samplings[i]`, `outs[i]` (as many as window i's rows) and `masks[i]`
    /// (structured output's bits for its rows; empty: none).
    pub fn sampleWindows(e: *Engine, samplings: []const ?lanes.Sampling, outs: []const []u32, masks: []const []const u32) !void {
        const n = e.round_n;
        if (n == 0 or samplings.len != n or outs.len != n or masks.len != n) return error.BadWindow;
        var parts: [8]sampler.Part = undefined;
        var positions: [8][64]u64 = undefined;
        for (0..n) |i| {
            const rows = e.round_len[i];
            if (outs[i].len != rows) return error.BadWindow;
            const st = e.round_state(i);
            for (0..rows) |r| positions[i][r] = @as(u64, @intCast(st.pos)) + 1 + @as(u64, r);
            const rule: ?lanes.Sampling = if (samplings[i]) |x| (if (x.temperature > 0) x else null) else null;
            const logits = e.fwd.b.logits + e.round_row0[i] * e.g.vocab * 2;
            if (masks[i].len > 0) try e.maskRows(i, logits, rows, masks[i]);
            parts[i] = .{ .logits = logits, .positions = positions[i][0..rows], .sampling = rule, .out = outs[i] };
        }
        try e.sampler.samplePacked(parts[0..n]);
    }

    /// Several of the round's windows committed at once (commitWindow for each, the GPU work shared): one commit table
    /// and one replay for the windows cut short of their rows, one table and one conv shift for all, one tap pass
    /// for the drafter (per window past 64 kept rows).
    pub fn commitWindows(e: *Engine, idx: []const usize, keeps: []const usize) !void {
        if (idx.len != keeps.len or idx.len == 0 or idx.len > 8) return error.BadWindow;
        var table: [8 * 6]i32 = undefined;
        var cut: [8 * 6]i32 = undefined;
        var n_cut: usize = 0;
        var kept: usize = 0;
        for (idx, keeps, 0..) |i, keep, k| {
            if (i >= e.round_n) return error.NoSuchWindow;
            if (e.committed[i]) return error.AlreadyCommitted;
            for (idx[0..k]) |j| if (j == i) return error.AlreadyCommitted;
            const rows = e.round_len[i];
            if (keep < 1 or keep > rows) return error.BadKeep;
            const st = e.round_state(i);
            if (e.drafter) |p| if (p.multi.contexts[st.slot].end != st.pos) return error.BadDraftCommit;
            const row = [_]i32{
                @intCast(e.round_row0[i]),
                @intCast(rows),
                @intCast(st.slot),
                @intCast(st.cur[0]),
                @intCast(st.slot),
                @intCast(keep),
            };
            @memcpy(table[k * 6 ..][0..6], &row);
            if (keep < rows) {
                @memcpy(cut[n_cut * 6 ..][0..6], &row);
                n_cut += 1;
            }
            kept += keep;
        }
        if (n_cut > 0) {
            try e.stageCommit(cut[0 .. n_cut * 6]);
            try e.replay();
        }
        try e.stageCommit(table[0 .. idx.len * 6]);
        try e.convShift();
        var items: [8]dflash.Multi.TapItem = undefined;
        for (idx, keeps, 0..) |i, keep, k| {
            const st = e.round_state(i);
            try e.setPosition(st, st.pos + keep);
            st.pos += keep;
            st.cur = @splat(1 - st.cur[0]);
            e.committed[i] = true;
            items[k] = .{ .slot = st.slot, .row0 = e.round_row0[i], .keep = keep };
        }
        if (e.drafter) |p| {
            if (kept <= 64) {
                try p.multi.commitTaps(items[0..idx.len], e.fwd.b.taps, e.fwd.b.rows);
            } else for (items[0..idx.len]) |it| try e.absorbTaps(p, it.slot, e.fwd.b.taps, e.fwd.b.rows, it.row0, it.keep);
        }
    }

    pub fn commitWindow(e: *Engine, i: usize, keep: usize) !void {
        if (i >= e.round_n) return error.NoSuchWindow;
        if (e.committed[i]) return error.AlreadyCommitted;
        const rows = e.round_len[i];
        if (keep < 1 or keep > rows) return error.BadKeep;
        const st = e.round_state(i);
        const row = [_]i32{
            @intCast(e.round_row0[i]),
            @intCast(rows),
            @intCast(st.slot),
            @intCast(st.cur[0]),
            @intCast(st.slot),
            @intCast(keep),
        };
        try e.stageCommit(&row);
        if (keep < rows) try e.replay();
        try e.convShift();
        try e.setPosition(st, st.pos + keep);
        st.pos += keep;
        st.cur = @splat(1 - st.cur[0]);
        e.committed[i] = true;
        if (e.drafter) |p| {
            if (p.multi.contexts[st.slot].end != st.pos - keep) return error.BadDraftCommit;
            try e.absorbTaps(p, st.slot, e.fwd.b.taps, e.fwd.b.rows, e.round_row0[i], keep);
        }
    }

    pub fn draftSlot(e: *Engine, seq: *Seq) !usize {
        if (!seq.alive or seq.slot >= e.slots.count or !e.used[seq.slot]) return error.BadDraftSlot;
        return seq.slot;
    }

    pub fn draftMany(e: *Engine, reqs: []DraftReq) !void {
        if (reqs.len == 0) return;
        if (e.mtp) |*m| if (e.draft_choice == .mtp) return mtp_mod.draftMany(m, e, reqs);
        const p = e.drafter orelse return error.DrafterNotLoaded;
        try dflash.draftMany(&p.multi, e, reqs);
    }

    fn absorbTaps(e: *Engine, p: *draft_pkg.Package, slot: usize, source: u64, stride: usize, row0: usize, rows: usize) !void {
        var at: usize = 0;
        while (at < rows) {
            const n = @min(rows - at, 64);
            try rounds.packTaps(e.torch.on(e.stream).base, source, stride, row0 + at, n, p.forward.tap_in);
            try p.multi.addTaps(slot, p.forward.tap_in, n);
            at += n;
        }
    }

    fn promptTaps(ptr: *anyopaque, rows: usize) !void {
        const e: *Engine = @ptrCast(@alignCast(ptr));
        if (e.drafter) |p| try e.absorbTaps(p, e.bound.?.slot, e.pe.b.taps, e.pe.b.rows, 0, rows);
        if (e.mtp) |*m| if (e.drafts and e.draft_choice == .mtp) {
            try e.torch.on(e.stream).base.copy(m.pending, e.pe.last_hidden, 8192);
            const start = e.prompt_at;
            const end = @min(start + rows + 1, e.prompt_tokens.len);
            try m.prompt(e, e.prompt_tokens[start + 1 .. end], e.pe.b.normed);
        };
        e.prompt_at += rows;
    }

    fn round_state(e: *Engine, i: usize) *state.State {
        return e.round_st[i];
    }

    fn stageVerify(e: *Engine, windows: []const rounds.Window) !void {
        if (windows.len == 0 or windows.len > 8) return error.BadWindow;
        var spans: [8]verify.Segment = undefined;
        var row0: usize = 0;
        for (windows, 0..) |w, i| {
            if (w.tokens.len == 0 or w.tokens.len > 64) return error.BadWindow;
            for (w.tokens, 0..) |t, r| {
                if (t >= 154880) return error.BadToken;
                e.host_tokens[i][r] = @intCast(t);
            }
            e.round_st[i] = w.state;
            e.round_row0[i] = row0;
            e.round_len[i] = w.tokens.len;
            e.committed[i] = false;
            spans[i] = .{ .st = w.state, .tokens = e.host_tokens[i][0..w.tokens.len] };
            row0 += w.tokens.len;
        }
        e.round_n = windows.len;
        try e.verify.forward(spans[0..windows.len]);
    }
};

test "round and lane contracts compile against the GLM engine" {
    _ = rounds.Round(Engine).commit;
    _ = @import("cuda_serving.zig").Serving(Engine).init;
    _ = Engine.verifyMany;
    _ = Engine.sampleWindows;
    _ = Engine.commitWindows;
    _ = Engine.draftMany;
    _ = Engine.seqBytes;
}

test "prefill many, drafts and an uncaptured conv shift fail closed" {
    try std.testing.expectError(error.PrefillNotBuilt, Engine.prefillMany(undefined, &.{}, &.{}));
    try std.testing.expectError(error.PromptCacheNotBuilt, Engine.copyPrefix(undefined, undefined, undefined, 1, 0));
    try Engine.draftMany(undefined, &.{});
    try std.testing.expect(!Engine.manyFit(undefined, &.{}));
}

/// Preserve the captured score-workspace stride and selection bucket for smaller caches.
/// _seg_scores reads only SEG_POOLS; _select_split SEG=1 reads only (POS+1)/4.
/// Thus unused score columns need storage, but no extra latent/pooled-key KV rows.
pub fn decodeScratchRows(cache_rows: usize) usize {
    return @max(cache_rows, 8 * 133120);
}

test "decode scratch keeps captured selection bucket for smaller caches" {
    try std.testing.expectEqual(@as(usize, 1064960), decodeScratchRows(2048));
    try std.testing.expectEqual(@as(usize, 1064960), decodeScratchRows(8 * 133120));
    try std.testing.expectEqual(@as(usize, 2113536), decodeScratchRows(8 * 264192));
}

test "draft hooks fail closed only when no package is installed, and validate sequence slots on CPU" {
    var e: Engine = undefined;
    e.drafter = null;
    e.mtp = null;
    e.draft_choice = .dflash2;
    e.drafts = true;
    var req = [_]Engine.DraftReq{.{ .seq = undefined, .source = .prefill_last, .next = &.{1}, .count = 1, .out = &.{} }};
    try std.testing.expectError(error.DrafterNotLoaded, e.draftMany(&req));
    try std.testing.expectError(error.DrafterNotLoaded, e.commitDrafts(&.{}, &.{}));
    e.drafts = false;
    try e.commitDrafts(&.{}, &.{});
    e.slots.count = 2;
    e.used = @splat(false);
    e.used[1] = true;
    var seq: Seq = .{ .slot = 1, .st = undefined, .alive = true };
    try std.testing.expectEqual(@as(usize, 1), try e.draftSlot(&seq));
    seq.alive = false;
    try std.testing.expectError(error.BadDraftSlot, e.draftSlot(&seq));
}

test "pool status publishes allocation and release without reading mutable extents" {
    var e: Engine = undefined;
    e.pool = try pool_mod.Pool.init(std.testing.allocator, 8192);
    defer e.pool.deinit();
    e.pool_free = .init(e.pool.rows);
    e.used = @splat(false);
    e.bound = null;
    const eid = (try e.pool.alloc(2049)).?;
    e.pool_free.store(e.pool.freeRows(), .release);
    var seq: Seq = .{ .slot = 0, .st = undefined, .alive = true, .eid = eid };
    try std.testing.expectEqual(@as(usize, 8192), e.poolStatus().pool_tokens);
    try std.testing.expectEqual(@as(usize, 4096), e.poolStatus().pool_free_tokens);
    e.freeSeq(&seq);
    try std.testing.expectEqual(@as(usize, 8192), e.poolStatus().pool_free_tokens);
    e.freeSeq(&seq);
    try std.testing.expectEqual(@as(usize, 8192), e.poolStatus().pool_free_tokens);
}

test "CLI commit publishes normalized target rows for the next MTP absorb" {
    var e: Engine = undefined;
    e.drafter = null;
    e.mtp = null;
    e.drafts = false;
    e.committed = @splat(false);
    var st: state.State = undefined;
    try e.commitDrafts(&.{.{ .state = &st, .tokens = &.{ 1, 2 } }}, &.{1});
    try std.testing.expect(e.committed[0]);
    try std.testing.expect(!e.committed[1]);
}

/// A host engine of TF_GLM_KV_SHARD's pool for rank `r` (no device: views are arithmetic over made-up planes).
fn hostShardEngine(e: *Engine, r: usize, rows: usize) !void {
    e.g = try G.init(2, r);
    e.shard = try shard_mod.Shard.init(r, 2, rows, shard_mod.trash);
    e.kv_switches = .{ .on = true };
    e.kv_counts = .{};
    e.pool = try pool_mod.Pool.initParts(std.testing.allocator, rows, 2);
    e.pool_free = .init(rows);
    e.caches = .{ .arena = undefined, .rows = rows, .streams = 4, .ring = 64, .latent = @splat(0x100000), .pooled = @splat(0x200000), .keys = @splat(0x300000), .gates = @splat(0x400000), .shard = e.shard };
    e.slots = .{ .arena = undefined, .g = e.g, .count = 4, .rows = 32, .rec = 0x1000, .conv = 0x2000, .proj = 0x3000, .out = 0x4000, .k = 0x5000, .v = 0x6000, .gates = 0x7000, .b = 0x8000, .pos_dev = 0x9000 };
    e.used = @splat(false);
}

/// newSeq's placement and binding without its device clears.
fn hostSeq(e: *Engine, slot: usize, rows: usize) !*Seq {
    const eid = (try e.pool.alloc(rows)) orelse return error.NoPoolExtent;
    (try e.pool.get(eid)).owner = slot;
    const x = try e.pool.extent(eid);
    e.seqs[slot] = .{ .slot = slot, .st = try state.State.bind(e.slots, e.caches, slot, x.base, x.size), .alive = true, .eid = eid };
    e.used[slot] = true;
    return &e.seqs[slot];
}

test "TF_GLM_KV_SHARD: a fresh prompt borrows the other part's rows on both ranks and gives them back; a placement takes back only twins no fill holds; a resume moves home, else keeps its part for a carry, else starts from 0" {
    const rows = 8 * G.extent;
    var es: [2]Engine = undefined;
    for (&es, 0..) |*e, r| try hostShardEngine(e, r, rows);
    defer for (&es) |*e| e.pool.deinit();
    // stream 0 in part 0 (rank 0's): its prompt borrows 2 extents of part 1 on both ranks, the views marked
    for (&es) |*e| {
        const s = try hostSeq(e, 0, 2 * G.extent);
        try std.testing.expect(try e.borrowRows(s));
        try std.testing.expect(s.st.cache.twin and s.st.cache.hasLatent());
        try std.testing.expectEqual(@as(u64, 1), e.kv_counts.twins_made);
        try std.testing.expectEqual(rows - 4 * G.extent, e.poolStatus().pool_free_tokens);
    }
    const s0 = &es[0].seqs[0];
    const t0 = &es[1].seqs[0];
    try std.testing.expect(s0.st.cache.owner and !t0.st.cache.owner);
    try std.testing.expectEqual(@as(u64, 0x100000), s0.st.cache.latent[0]); // rank 0: its own rows (local 0)
    try std.testing.expectEqual(@as(u64, 0x100000), t0.st.cache.latent[0]); // rank 1: the twin at its part's first row
    try std.testing.expectEqual(es[0].pool.digest(), es[1].pool.digest());
    // a placement takes back no twin a fill holds; one no fill holds goes back, the view as bound before the borrow
    for (&es) |*e| try std.testing.expectEqual(@as(usize, 0), e.revokeTwins());
    for (&es) |*e| {
        e.pool.hold(e.seqs[0].eid.?, false);
        try std.testing.expectEqual(@as(usize, 1), e.revokeTwins());
        try std.testing.expectEqual(@as(u64, 1), e.kv_counts.twins_revoked);
        try std.testing.expect(!e.seqs[0].st.cache.twin and !e.pool.hasTwins(e.seqs[0].eid.?));
    }
    try std.testing.expect(!t0.st.cache.hasLatent() and s0.st.cache.hasLatent());
    // borrowed again, then given back as a prompt's end does; twins off, or a sequence past its start, borrow nothing
    for (&es) |*e| {
        try std.testing.expect(try e.borrowRows(&e.seqs[0]));
        e.giveRows(&e.seqs[0]);
        e.giveRows(&e.seqs[0]); // (nothing to give twice)
        try std.testing.expectEqual(rows - 2 * G.extent, e.poolStatus().pool_free_tokens);
        e.kv_switches.twins = false;
        try std.testing.expect(!try e.borrowRows(&e.seqs[0]));
        e.kv_switches.twins = true;
        e.seqs[0].st.pos = 64;
        try std.testing.expect(!try e.borrowRows(&e.seqs[0]));
        e.seqs[0].st.pos = 0;
    }
    try std.testing.expect(!t0.st.cache.hasLatent() and !t0.st.cache.twin);
    // a resume of a state kept in part 0: stream 1 placed in part 1 (fewer streams), moved home where part 0 has room
    const home0: Snap = .{ .pos = 4096, .home = 0 };
    for (&es) |*e| {
        const s = try hostSeq(e, 1, 2 * G.extent);
        try std.testing.expectEqual(@as(usize, 4 * G.extent), s.st.base);
        try std.testing.expect(try e.placeResume(s, &home0));
        try std.testing.expectEqual(@as(usize, 2 * G.extent), s.st.base);
        try std.testing.expectEqual(e.g.rank == 0, s.st.cache.hasLatent()); // rebound: rank 0's rows now
    }
    try std.testing.expectEqual(es[0].pool.digest(), es[1].pool.digest());
    // part 0 full: stream 2 keeps part 1 and the restore will carry; the carry off, or a state under the minimum: from 0
    for (&es) |*e| {
        const s = try hostSeq(e, 2, 2 * G.extent);
        try std.testing.expect(try e.placeResume(s, &home0));
        try std.testing.expectEqual(@as(usize, 4 * G.extent), s.st.base);
        e.kv_switches.carry = false;
        try std.testing.expect(!try e.placeResume(s, &home0));
        // (so a restore of such a state from disk does not start; part 1's two free extents take 2,048 tokens and the
        // next one, not 4,096)
        try std.testing.expect(!try e.tierResumable(4096, 0));
        try std.testing.expect(!try e.tierResumable(4096, 1) and try e.tierResumable(G.extent, 1));
        e.kv_switches.carry = true;
        try std.testing.expect(try e.tierResumable(4096, 0) and !try e.tierResumable(shard_keep.carry_min - 64, 0));
        try std.testing.expect(!try e.placeResume(s, &.{ .pos = shard_keep.carry_min - 64, .home = 0 }));
        try std.testing.expectEqual(@as(u64, 1), e.kv_counts.refused_small);
        try std.testing.expectError(error.KvShardNoHome, e.placeResume(s, &.{ .pos = 4096 }));
        // the carry's counters reach /health as rescues
        e.kv_counts.carries = 1;
        var k: lanes.kv_parts.Stats = .{};
        kvPartsOf(&e.pool, e.kv_counts, &k);
        try std.testing.expect(k.carries == 1 and k.carry_rescued == 1 and k.refused_small == 1 and k.twins_made == 2 and k.twins_revoked == 1);
    }
    // unsharded: every resume as before, nothing borrowed
    var plain: Engine = undefined;
    plain.shard = null;
    var q: Seq = .{ .slot = 0, .st = undefined, .eid = 0 };
    try std.testing.expect(try plain.placeResume(&q, &home0));
    try std.testing.expect(try plain.tierResumable(4096, 0));
    try std.testing.expect(!try plain.borrowRows(&q));
    plain.giveRows(&q);
}

/// One u64 from every rank (`mine` this rank's), in rank order, over the pool agreement's all-gather.
fn gatherWord(ctx: *const cuda.Context, c: *const comm.Comm, stream: cuda.Stream, mine: u64) ![comm.max_world]u64 {
    const w: usize = c.world;
    var words = try cuda.DeviceBuffer.alloc(ctx.d, (1 + w) * 8);
    defer words.free();
    var word = mine;
    try words.upload(0, std.mem.asBytes(&word));
    try c.allGather(words.ptr, words.ptr + 8, 1, .u64, stream.handle);
    try stream.synchronize();
    var all: [comm.max_world]u64 = undefined;
    try words.download(8, std.mem.sliceAsBytes(all[0..w]));
    return all;
}

/// cuda_shard_keep.carry's device on the GPU: NCCL transfers and copies on the engine's stream (the prompt's), held
/// against the row split's side stream as the history fetch's are (cuda_prefill's ShardDev).
const CarryDev = struct {
    e: *Engine,
    f: *const shard_prompt.Fetch,
    pub fn send(dv: *CarryDev, at: u64, n: usize) !void {
        try dv.f.comm.exchangeBytes(@intCast(dv.f.sh.peer()), &.{.{ at, n }}, &.{}, dv.e.stream.handle);
    }
    pub fn recv(dv: *CarryDev, at: u64, n: usize) !void {
        try dv.f.comm.exchangeBytes(@intCast(dv.f.sh.peer()), &.{}, &.{.{ at, n }}, dv.e.stream.handle);
    }
    pub fn copy(dv: *CarryDev, dst: u64, src: u64, n: usize) !void {
        try Engine.copyBytes(dv.e, dst, src, n);
    }
    pub fn hold(dv: *CarryDev) !void {
        const l = dv.e.pe.lane orelse return;
        try l.share_out.record(l.stream);
        try dv.e.stream.wait(l.share_out);
    }
    pub fn release(dv: *CarryDev) !void {
        const l = dv.e.pe.lane orelse return;
        try l.share_in.record(dv.e.stream);
        try l.stream.wait(l.share_in);
    }
};

/// TF_GLM_KV_SHARD's pool sizing: the caches and the decode scratch at a pool's rows, unsharded and sharded
/// (cuda_pool_capacity.sharded); sharded adds owner decode's exchange buffers and the prompt fetch's (`fetch` bytes,
/// shard_prompt.bytes at the requested context: what the part's context allocates at most).
const ShardCost = struct {
    g: G,
    streams: usize,
    window: usize,
    fetch: usize = 0,
    pub fn plain(c: ShardCost, rows: usize) usize {
        return state.Caches.bytes(rows, c.streams, 2048) + fwd.Scratch.bytes(c.g, c.window, decodeScratchRows(rows));
    }
    pub fn sharded(c: ShardCost, rows: usize) usize {
        const s = shard_mod.Shard.init(0, 2, rows, shard_mod.trash) catch return std.math.maxInt(usize);
        return state.Caches.shardBytes(rows, c.streams, 2048, s) + fwd.Scratch.bytes(c.g, c.window, decodeScratchRows(rows)) + shard_dec.bytes(c.window) + c.fetch;
    }
};

/// A pool's parts as /health's kv_shard shows them: each part's free, live and borrowed rows and live streams, and
/// the counters. Kept prompt states live outside the pool here (cuda_snapshot), so no part holds kept rows; a carry
/// is made only where the state's own part had no room (every one `carry_rescued`).
pub fn kvPartsOf(p: *const pool_mod.Pool, counts: KvCounts, out: *lanes.kv_parts.Stats) void {
    const n = @min(p.parts(), out.part.len);
    out.parts = n;
    out.part_rows = if (p.part == 0) p.rows else p.part;
    out.checks = counts.checks;
    out.carries = counts.carries;
    out.carry_rescued = counts.carries;
    out.moved_tokens = counts.moved_tokens;
    out.moved_bytes = counts.moved_bytes;
    out.refused_small = counts.refused_small;
    out.twins_made = counts.twins_made;
    out.twins_revoked = counts.twins_revoked;
    for (out.part[0..n], 0..) |*q, i| {
        const held = p.rowsIn(i);
        q.* = .{ .free = p.freeIn(i), .live_rows = held.live, .twin_rows = held.twins, .live = p.live(i) };
    }
}

/// A rank's prompt fetch counters as /health's kv_shard shows them (rank 0's: it sends or receives every fetch).
pub fn promptOf(s: shard_prompt.Stats, out: *lanes.kv_parts.Stats) void {
    out.prompt_fetches = s.fetches;
    out.prompt_rows = s.rows;
    out.prompt_bytes = s.bytes;
    out.prompt_syncs = s.syncs;
    out.prompt_blocked = s.blocked;
}

/// The pool every rank can hold: the smallest of the ranks' proposals (one per rank, in rank order).
fn agreedPool(proposals: []const u64) u64 {
    var m: u64 = std.math.maxInt(u64);
    for (proposals) |p| m = @min(m, p);
    return m;
}

test "the agreed pool is the smallest proposal of every rank, whichever rank makes it (TP=2, 3, 4)" {
    try std.testing.expectEqual(@as(u64, 4096), agreedPool(&.{ 8192, 4096 }));
    try std.testing.expectEqual(@as(u64, 2048), agreedPool(&.{ 8192, 6144, 2048 })); // rank 2 smallest at TP=3
    try std.testing.expectEqual(@as(u64, 2048), agreedPool(&.{ 2048, 6144, 8192 }));
    try std.testing.expectEqual(@as(u64, 1024), agreedPool(&.{ 8192, 6144, 4096, 1024 })); // rank 3 at TP=4
}

test "TF_GLM_KV_SHARD: /health's parts from the pool, a stream's share of the latent rows, the sizing's two layouts" {
    var p = try pool_mod.Pool.initParts(std.testing.allocator, 8 * 2048, 2);
    defer p.deinit();
    const a = (try p.alloc(4096)).?;
    (try p.get(a)).owner = 0;
    const b = (try p.alloc(2048)).?;
    (try p.get(b)).owner = 1;
    try std.testing.expect(try p.borrow(a));
    var k: lanes.kv_parts.Stats = .{};
    kvPartsOf(&p, .{ .checks = 3, .twins_made = 1 }, &k);
    try std.testing.expectEqual(@as(u64, 2), k.parts);
    try std.testing.expectEqual(@as(u64, 8192), k.part_rows);
    try std.testing.expectEqual(@as(u64, 3), k.checks);
    try std.testing.expectEqual(@as(u64, 1), k.twins_made);
    // part 0: the 4096-row stream; part 1: the 2048-row stream and the first one's 4096-row twin
    try std.testing.expectEqual(lanes.kv_parts.Part{ .free = 4096, .live_rows = 4096, .live = 1 }, k.part[0]);
    try std.testing.expectEqual(lanes.kv_parts.Part{ .free = 2048, .live_rows = 2048, .twin_rows = 4096, .live = 1 }, k.part[1]);
    try std.testing.expectEqual(lanes.kv_parts.Part{}, k.part[2]);
    // one part (off): the pool as one, as the engine never reports it
    var one = try pool_mod.Pool.init(std.testing.allocator, 8192);
    defer one.deinit();
    var k1: lanes.kv_parts.Stats = .{};
    kvPartsOf(&one, .{}, &k1);
    try std.testing.expectEqual(@as(u64, 1), k1.parts);
    try std.testing.expectEqual(@as(u64, 8192), k1.part[0].free);
    // a stream's latent rows on one rank of two: a rank's share half of them, everything else alike
    var e: Engine = undefined;
    e.g = try G.init(2, 0);
    e.window = 64;
    e.prefill_rows = 2048;
    e.mtp = null;
    e.shard = null;
    const plain = e.seqBytes(131072);
    e.shard = try shard_mod.Shard.init(0, 2, 1 << 20, shard_mod.trash);
    try std.testing.expectEqual(plain - 11 * 131072 * 528 / 2, e.seqBytes(131072));
    // the sizing's layouts are cuda_state's, the scratch alike
    const c: ShardCost = .{ .g = e.g, .streams = 4, .window = 64, .fetch = shard_prompt.bytes(1 << 20, 2048) };
    const sh = try shard_mod.Shard.init(1, 2, 1 << 20, shard_mod.trash);
    // (and owner decode's exchange buffers and the prompt fetch's on the sharded side)
    try std.testing.expectEqual(state.Caches.bytes(1 << 20, 4, 2048) - state.Caches.shardBytes(1 << 20, 4, 2048, sh) - shard_dec.bytes(64) - shard_prompt.bytes(1 << 20, 2048), c.plain(1 << 20) - c.sharded(1 << 20));
    // the prompt fetch's counters on /health
    var k2: lanes.kv_parts.Stats = .{};
    promptOf(.{ .fetches = 11, .rows = 4000, .bytes = 4000 * 528, .syncs = 13, .blocked = 1, .blocks = 3 }, &k2);
    try std.testing.expectEqual(@as(u64, 11), k2.prompt_fetches);
    try std.testing.expectEqual(@as(u64, 4000 * 528), k2.prompt_bytes);
    try std.testing.expectEqual(@as(u64, 13), k2.prompt_syncs);
    try std.testing.expectEqual(@as(u64, 1), k2.prompt_blocked);
    try std.testing.expectEqual(@as(usize, std.math.maxInt(usize)), c.sharded(2048)); // not whole parts: never fits
}
