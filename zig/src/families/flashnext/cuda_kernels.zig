//! Flash Next's CUDA extension kernels: the fn_* fatbins (copies of the Python extensions' device code, SASS-equal,
//! zig/tests/cuda/copies.py) with each Python wrapper's launch: grid, block, dynamic shared memory, stream, arguments.
//!
//! Sources (all authored by ashhart in TensorFold, see each fatbin's copy header):
//!   fn_gdn           families/qwen4_exp/cuda/gdn.cu      gdn_chain_cuda, gdn_replay_cuda   (tensorfold_qwen4_exp_gdn)
//!   fn_gdn_io        families/qwen4_exp/cuda/gdn_io.cu   gdn_front_cuda, gdn_back_cuda     (tensorfold_qwen4_exp_gdn_io)
//!   fn_gdn_prefill   cuda/kernels/gdn_prefill.cu         gdn_prefill_cuda                  (tensorfold_gdn_v2)
//!   fn_gdn_tree      cuda/kernels/gdn.cu                 gdn_tree_cuda, gdn_replay_cuda (fp32 keys)
//!   fn_nvfp4_experts cuda/nvfp4/experts.cu               nvfp4_experts_cuda                (tensorfold_nvfp4_v3)
//!   fn_qmm           cuda/kernels/qmm.cu                 qmm_cuda, groups of 32            (tensorfold_qmm_v5)
//!   fn_qmm_prefill   cuda/kernels/qmm_prefill.cu         qmm_prefill_cuda, groups of 32, tile 0
//!   experts          cuda/experts.cu                     experts_plan_cuda (Nemotron's copy) (tensorfold_experts_v7)
//!   fn_nvfp4_shape   ours (decode D3): fn_nvfp4_experts' kernel in another launch shape, each unit's arithmetic the
//!                    original's (zig/tests/cuda/flashnext/experts_shape_test.cu checks the bytes)

const std = @import("std");
const cuda = @import("cuda");

/// Mangled names of every entry point resolved here (cuobjdump -symbols of each fatbin).
pub const sym = struct {
    pub const chain = [2][2][:0]const u8{
        .{ // 16 key heads, 48 value heads (one GPU): AHEAD false, true
            "_ZN9tf_fn_gdn12chain_kernelILi16ELi48ELb0EEEvPK13__nv_bfloat16S3_S3_PKfS5_S5_S3_fiPS1_PfS7_S7_S6_S7_S7_",
            "_ZN9tf_fn_gdn12chain_kernelILi16ELi48ELb1EEEvPK13__nv_bfloat16S3_S3_PKfS5_S5_S3_fiPS1_PfS7_S7_S6_S7_S7_",
        },
        .{ // 8 and 24 (a TP=2 rank)
            "_ZN9tf_fn_gdn12chain_kernelILi8ELi24ELb0EEEvPK13__nv_bfloat16S3_S3_PKfS5_S5_S3_fiPS1_PfS7_S7_S6_S7_S7_",
            "_ZN9tf_fn_gdn12chain_kernelILi8ELi24ELb1EEEvPK13__nv_bfloat16S3_S3_PKfS5_S5_S3_fiPS1_PfS7_S7_S6_S7_S7_",
        },
    };
    pub const replay = [2][:0]const u8{
        "_ZN9tf_fn_gdn13replay_kernelILi16ELi48EEEvPKfS2_PK13__nv_bfloat16S2_S2_iPf",
        "_ZN9tf_fn_gdn13replay_kernelILi8ELi24EEEvPKfS2_PK13__nv_bfloat16S2_S2_iPf",
    };
    pub const front = [2][:0]const u8{
        "_ZN12tf_fn_gdn_io12front_kernelILi16ELi48EEEvPK13__nv_bfloat16PKxPKiS7_S3_PKfS9_PfSA_PS1_SA_SA_",
        "_ZN12tf_fn_gdn_io12front_kernelILi8ELi24EEEvPK13__nv_bfloat16PKxPKiS7_S3_PKfS9_PfSA_PS1_SA_SA_",
    };
    pub const back = [2][:0]const u8{
        "_ZN12tf_fn_gdn_io11back_kernelILi16ELi48EEEvPK13__nv_bfloat16S3_S3_fPS1_Pf",
        "_ZN12tf_fn_gdn_io11back_kernelILi8ELi24EEEvPK13__nv_bfloat16S3_S3_fPS1_Pf",
    };
    /// [keys: fp32, bf16][value rows a block: 128, 64]
    pub const prefill = [2][2][:0]const u8{
        .{
            "_ZN17tf_fn_gdn_prefill12chain_kernelIfLi128ELi32EEEvPKT_S3_PK13__nv_bfloat16PKfS8_S8_PfPS4_iii",
            "_ZN17tf_fn_gdn_prefill12chain_kernelIfLi64ELi32EEEvPKT_S3_PK13__nv_bfloat16PKfS8_S8_PfPS4_iii",
        },
        .{
            "_ZN17tf_fn_gdn_prefill12chain_kernelI13__nv_bfloat16Li128ELi32EEEvPKT_S4_PKS1_PKfS8_S8_PfPS1_iii",
            "_ZN17tf_fn_gdn_prefill12chain_kernelI13__nv_bfloat16Li64ELi32EEEvPKT_S4_PKS1_PKfS8_S8_PfPS1_iii",
        },
    };
    pub const tree_replay: [:0]const u8 = "_ZN14tf_fn_gdn_tree13replay_kernelIfLi8ELi4EEEvPKxiPKiiS4_iPfiii";
    /// [epilogue 2 (gate/up SwiGLU, M 2), 0 (down, fp32), 3 (down, bf16)]
    pub const nvfp4 = [3][:0]const u8{
        "_ZN19tf_fn_nvfp4_experts19nvfp4_expert_kernelILi2ELi2ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi",
        "_ZN19tf_fn_nvfp4_experts19nvfp4_expert_kernelILi1ELi0ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi",
        "_ZN19tf_fn_nvfp4_experts19nvfp4_expert_kernelILi1ELi3ELi4EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi",
    };
    /// gate/up SwiGLU (M 2, epilogue 2) with one warp a block, two stages and one n8 tile a warp (fn_nvfp4_shape.cu)
    pub const nvfp4_nt: [:0]const u8 = "_ZN17tf_fn_nvfp4_shape16expert_nt_kernelILi2ELi2ELi1ELi2ELi1EEEvPK13__nv_bfloat16iiPK5uint4PKfiiPKiSA_SA_Pvifi";
    /// qmm_kernel's K-slice cluster forms (fn_qmm_cluster.cu, decode D5): [out: bf16, fp32][row tile: 16, 32, 64]
    pub const qmm_cluster = [2][3][:0]const u8{
        .{
            "_ZN9tf_fn_qmm10qmm_kernelILi32ELi16ELi64ELi1ELi4ELi4ELb0ELb1ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii",
            "_ZN9tf_fn_qmm10qmm_kernelILi32ELi32ELi64ELi1ELi4ELi4ELb0ELb1ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii",
            "_ZN9tf_fn_qmm10qmm_kernelILi32ELi64ELi64ELi1ELi4ELi4ELb0ELb1ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii",
        },
        .{
            "_ZN9tf_fn_qmm10qmm_kernelILi32ELi16ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii",
            "_ZN9tf_fn_qmm10qmm_kernelILi32ELi32ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii",
            "_ZN9tf_fn_qmm10qmm_kernelILi32ELi64ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii",
        },
    };
    /// [out: bf16, fp32][row tile: 16, 32, 64]
    pub const qmm = [2][3][:0]const u8{
        .{
            "_ZN9tf_fn_qmm10qmm_kernelILi32ELi16ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii",
            "_ZN9tf_fn_qmm10qmm_kernelILi32ELi32ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii",
            "_ZN9tf_fn_qmm10qmm_kernelILi32ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii",
        },
        .{
            "_ZN9tf_fn_qmm10qmm_kernelILi32ELi16ELi64ELi1ELi4ELi4ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii",
            "_ZN9tf_fn_qmm10qmm_kernelILi32ELi32ELi64ELi1ELi4ELi4ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii",
            "_ZN9tf_fn_qmm10qmm_kernelILi32ELi64ELi64ELi1ELi4ELi4ELb1ELb0ELb0EEEvPK13__nv_bfloat16PKfPKjS3_S3_PvPfiiiiiii",
        },
    };
    /// [out: bf16, fp32]
    pub const qmm_prefill = [2][:0]const u8{
        "_ZN17tf_fn_qmm_prefill14prefill_kernelILi32ELi128ELi128ELi2ELi4ELi3ELb0EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii",
        "_ZN17tf_fn_qmm_prefill14prefill_kernelILi32ELi128ELi128ELi2ELi4ELi3ELb1EEEvPK13__nv_bfloat16PKjS3_S3_Pviiiiii",
    };
    pub const plan = "_ZN10tf_experts11plan_kernelEPKiiiiPiS2_S2_";
    pub const plan_rank = "_ZN10tf_experts9plan_rankEPKiiiPiS2_";
    pub const plan_offsets = "_ZN10tf_experts12plan_offsetsEiiiPiS0_S0_";
    pub const plan_scatter = "_ZN10tf_experts12plan_scatterEPKiiiS1_S1_Pi";
};

/// gdn.cu's tree_kernel<float, SLOTS, R, WARPS, CHAIN> instantiations, in dispatch_tree's order.
pub const TreeVariant = struct { slots: u32, r: u32, warps: u32, chain: bool };

pub const tree_variants = [_]TreeVariant{
    .{ .slots = 0, .r = 8, .warps = 4, .chain = true },   .{ .slots = 1, .r = 8, .warps = 4, .chain = false },
    .{ .slots = 2, .r = 8, .warps = 2, .chain = false },  .{ .slots = 2, .r = 4, .warps = 4, .chain = false },
    .{ .slots = 4, .r = 2, .warps = 4, .chain = false },  .{ .slots = 8, .r = 2, .warps = 4, .chain = false },
    .{ .slots = 16, .r = 2, .warps = 2, .chain = false }, .{ .slots = 32, .r = 2, .warps = 1, .chain = false },
};

fn treeSymbol(comptime v: TreeVariant) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN14tf_fn_gdn_tree11tree_kernelIfLi{d}ELi{d}ELi{d}ELb{d}EEEvPKT_S3_PK13__nv_bfloat16" ++
        "PKfS8_S8_PKxPKiSC_iPS4_iiiNS_7PendingIS1_EEPfSA_b", .{ v.slots, v.r, v.warps, @intFromBool(v.chain) });
}

pub const tree_symbols = blk: {
    var out: [tree_variants.len][:0]const u8 = undefined;
    for (tree_variants, &out) |v, *s| s.* = treeSymbol(v);
    break :blk out;
};

/// dispatch_tree: the variant index for `slots` on this GPU ("wide": many-SM sm_120 with several streams).
pub fn treeIndex(slots: u32, streams: u32, major: c_int, minor: c_int, sms: c_int) usize {
    const wide = streams >= 2 and major == 12 and minor == 0 and sms >= 96;
    if (slots == 0) return 0;
    if (wide and slots <= 1) return 1;
    if (wide and slots <= 2) return 2;
    if (slots <= 2) return 3;
    if (slots <= 4) return 4;
    if (slots <= 8) return 5;
    if (slots <= 16) return 6;
    return 7;
}

/// launch_tree's dynamic shared memory: R value rows of every slot a warp, plus the plan rows of a tree window.
pub fn treeShared(v: TreeVariant, max_rows: usize) usize {
    return 16 * v.warps * v.slots * v.r * 32 + (if (v.chain) 0 else 4 * 3 * max_rows);
}

/// gdn.cu's Pending<float>, passed by value: the previous round's accepted rows (k 0: none).
pub const Pending = extern struct { k: u64 = 0, v: u64 = 0, g: u64 = 0, beta: u64 = 0, rows: u64 = 0, row_stride: c_int = 0, counts: u64 = 0, count_stride: c_int = 0 };

comptime {
    std.debug.assert(@sizeOf(Pending) == 64 and @offsetOf(Pending, "counts") == 48);
}

/// qmm_frag.cuh's LaneTile<32, BM, 64, 1, 4, 4>::SMEM: max(4 stages, the cluster partials) = 272 BM + 5120.
pub fn qmmShared(bm: usize) u32 {
    const mt = bm / 16;
    const stage = bm * 64 + 32 * 64 / 2 + 2 * 64 * 2 + bm * 4;
    const partials = mt * 2 * 4 * 128 * 4;
    return @intCast(@max(4 * stage, partials));
}

/// qmm_prefill.cu's Tile<32, 128, 128, 2, 4, 3>::SMEM.
pub const qmm_prefill_smem: u32 = 3 * (128 * 64 + 128 * 32 / 2 + 2 * 128 * 2);

/// qmm.bucket: the decode row tile (16, 32, else 64-row tiles side by side; never changes bits).
pub fn bucket(m: usize) usize {
    return if (m <= 16) 16 else if (m <= 32) 32 else 64;
}

fn bucketIndex(bm: usize) usize {
    return switch (bm) {
        16 => 0,
        32 => 1,
        else => 2,
    };
}

/// qmm.split_k for groups of 32: K slices fixed by the weight's shape, never by the row count.
pub fn splitK32(n: usize, k: usize) usize {
    const tiles = (n + 63) / 64;
    const groups = k / 32;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}

/// The L2 band qmm.cu and qmm_prefill.cu sweep column tiles in: inputs of `group` row tiles near 12 MB.
pub fn l2Group(rows_t: usize, bm: usize, k: usize) usize {
    return @max(1, @min(rows_t, (12 << 20) / (bm * k * 2)));
}

/// experts.max_items: an item per used expert plus one per `tile` pairs past its first.
pub fn maxItems(pairs: usize, experts: usize, tile: usize) usize {
    return @min(pairs, experts) + pairs / tile;
}

/// Plan tiles: Flash Next's NVFP4 routed experts take items of 16 pairs, decode and prompt alike
/// (cuda/experts.py TILE and cuda/nvfp4/experts.py PREFILL_TILE).
pub const plan_tile: usize = 16;
pub const plan_max_experts: usize = 1024; // experts.cu EMAX
pub const plan_small: usize = 1024; // experts.cu PSMALL: pairs the one-block plan takes

/// Scratch the expert plan writes: members, items (expert, first, count), counts, and the wide path's rank and hist.
pub const Plan = struct { members: u64, items: u64, counts: u64, rank: u64, hist: u64 };

/// One layer's NVFP4 routed experts (cuda/nvfp4/experts.py Experts4): blocks and fp32 (expert, matrix) scales.
pub const Experts4 = struct { up: u64, down: u64, up_scale: u64, down_scale: u64, width: usize, dims: usize, limit: f32 = 0 };

/// A packed 4-bit (groups of 32) weight (cuda/kernels/qmm.py Q4): words, (kg, npad) bf16 scales and biases.
pub const Q4 = struct { w: u64, s: u64, b: u64, n: usize, k: usize, npad: usize };

const n_mods = 10;

pub const Kernels = struct {
    d: *const cuda.Driver,
    mods: [n_mods]cuda.Module,
    chain: [2][2]cuda.Function, // [48, 24 value heads][AHEAD false, true]
    replay: [2]cuda.Function,
    front: [2]cuda.Function,
    back: [2]cuda.Function,
    prefill: [2][2]cuda.Function, // [fp32, bf16 keys][128, 64 rows]
    tree: [tree_variants.len]cuda.Function,
    tree_replay: cuda.Function,
    nvfp4: [3]cuda.Function, // epilogue 2, 0, 3
    nvfp4_blocks: [3]usize, // resident blocks the grid caps at: per SM (at least 1) times SMs
    /// decode D3: gate/up calls of at most `nt_units` units run nvfp4_nt (one warp a block, one n8 tile a warp: 4
    /// warps a unit), same bits; 0 (TF_FLASHNEXT_EXPERT_SHAPE=0) keeps the original launch everywhere
    nvfp4_nt: cuda.Function,
    nvfp4_nt_blocks: usize,
    nt_units: usize,
    qmm: [2][3]cuda.Function, // [bf16, fp32][16, 32, 64]
    qmm_cluster: [2][3]cuda.Function, // the same with K slices summed in a cluster (decode D5)
    qmm_prefill: [2]cuda.Function,
    plan_small: cuda.Function,
    plan_rank: cuda.Function,
    plan_offsets: cuda.Function,
    plan_scatter: cuda.Function,
    sms: c_int,
    major: c_int,
    minor: c_int,

    /// Loads every module and resolves every entry point; dynamic shared memory opted in as the wrappers do.
    pub fn load(ctx: *const cuda.Context) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const kk = cuda.kernels;
        var k: Kernels = undefined;
        k.d = ctx.d;
        const images = [n_mods][]const u8{ kk.fn_gdn, kk.fn_gdn_io, kk.fn_gdn_prefill, kk.fn_gdn_tree, kk.fn_nvfp4_experts, kk.fn_qmm, kk.fn_qmm_prefill, kk.experts, kk.fn_nvfp4_shape, kk.fn_qmm_cluster };
        var loaded: usize = 0;
        errdefer for (k.mods[0..loaded]) |*m| m.unload();
        for (images, 0..) |img, i| {
            k.mods[i] = try cuda.Module.load(ctx.d, img);
            loaded += 1;
        }
        const m = k.mods;
        for (0..2) |h| {
            for (0..2) |a| k.chain[h][a] = try m[0].function(sym.chain[h][a]);
            k.replay[h] = try m[0].function(sym.replay[h]);
            k.front[h] = try m[1].function(sym.front[h]);
            k.back[h] = try m[1].function(sym.back[h]);
            for (0..2) |r| k.prefill[h][r] = try m[2].function(sym.prefill[h][r]);
            k.qmm_prefill[h] = try m[6].function(sym.qmm_prefill[h]);
            for (0..3) |b| k.qmm[h][b] = try m[5].function(sym.qmm[h][b]);
            for (0..3) |b| k.qmm_cluster[h][b] = try m[9].function(sym.qmm_cluster[h][b]);
        }
        for (tree_symbols, &k.tree) |s, *f| f.* = try m[3].function(s);
        k.tree_replay = try m[3].function(sym.tree_replay);
        for (sym.nvfp4, &k.nvfp4) |s, *f| f.* = try m[4].function(s);
        k.plan_small = try m[7].function(sym.plan);
        k.plan_rank = try m[7].function(sym.plan_rank);
        k.plan_offsets = try m[7].function(sym.plan_offsets);
        k.plan_scatter = try m[7].function(sym.plan_scatter);
        k.nvfp4_nt = try m[8].function(sym.nvfp4_nt);
        // qmm.cu and qmm_prefill.cu's launch: cudaFuncSetAttribute(SMEM) once per kernel, whatever its size
        for (0..2) |h| {
            for (0..3) |b| try k.qmm[h][b].allowDynamicShared(qmmShared(@as(usize, 16) << @intCast(b)));
            for (0..3) |b| try k.qmm_cluster[h][b].allowDynamicShared(qmmShared(@as(usize, 16) << @intCast(b)));
            try k.qmm_prefill[h].allowDynamicShared(qmm_prefill_smem);
        }
        k.sms = try ctx.attribute(.multiprocessor_count);
        k.major = try ctx.attribute(.compute_capability_major);
        k.minor = try ctx.attribute(.compute_capability_minor);
        // nvfp4/experts.cu launch: cudaOccupancyMaxActiveBlocksPerMultiprocessor(128 threads, 0 bytes), at least 1
        for (k.nvfp4, &k.nvfp4_blocks) |f, *n| n.* = @as(usize, @max(1, try f.occupancy(128, 0))) * @as(usize, @intCast(k.sms));
        k.nvfp4_nt_blocks = @as(usize, @max(1, try k.nvfp4_nt.occupancy(32, 0))) * @as(usize, @intCast(k.sms));
        k.nt_units = ntUnits(if (std.c.getenv("TF_FLASHNEXT_EXPERT_SHAPE")) |v| std.mem.span(v) else null);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        for (&k.mods) |*m| m.unload();
    }
};

/// The largest gate/up launch (plan units) that takes the one-tile-a-warp shape: 230 = one row at TP=1 (20 column
/// blocks x 11 items), one or two rows at TP=2 (10 x 11, 10 x 23), where experts_shape_test.cu measured it faster
/// (TP=2 1.22x and 1.08x, TP=1 1.12x); at more units the original is as fast or faster. TF_FLASHNEXT_EXPERT_SHAPE=0
/// turns it off, =N sets the bound.
pub fn ntUnits(env: ?[]const u8) usize {
    const v = env orelse return 230;
    return std.fmt.parseInt(usize, v, 10) catch 230;
}

fn int(x: usize) c_int {
    return @intCast(x);
}

fn u(x: usize) u32 {
    return @intCast(x);
}

/// Value heads -> the instantiation index: 48 (16 key heads, one GPU) 0, 24 (8, a TP=2 rank) 1.
fn heads(nv: usize) !usize {
    return switch (nv) {
        48 => 0,
        24 => 1,
        else => error.UnsupportedHeads,
    };
}

/// Launches on one stream; each mirrors the C++ wrapper (and its Python caller) named in its comment.
pub const Ops = struct {
    k: *const Kernels,
    s: cuda.Stream,

    fn go(o: Ops, f: cuda.Function, grid: [3]usize, block: usize, shared: usize, args: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = u(grid[0]), .y = u(grid[1]), .z = u(grid[2]) }, .block = .{ .x = u(block) }, .shared = u(shared) }, o.s, args);
    }

    /// qwen4_exp gdn.chain (gdn_chain_cuda): `rows` rows of one stream through conv, norms, gates, the delta rule and
    /// the gated RMSNorm; `nv` value heads (48, or 24 a TP=2 rank). Zero addresses are the wrapper's empty tensors.
    pub fn gdnChain(o: Ops, nv: usize, p: u64, conv_state: u64, conv_w: u64, state_in: u64, a_log: u64, dt_bias: u64, norm_w: u64, eps: f32, rows: usize, out: u64, xs: u64, state_out: u64, k_save: u64, v_save: u64, g_save: u64, b_save: u64) !void {
        const h = try heads(nv);
        var a: cuda.Args = .{};
        for ([_]u64{ p, conv_state, conv_w, state_in, a_log, dt_bias, norm_w }) |v| a.add(v);
        a.add(eps);
        a.add(int(rows));
        for ([_]u64{ out, xs, state_out, k_save, v_save, g_save, b_save }) |v| a.add(v);
        try o.go(o.k.chain[h][@intFromBool(rows > 1)], .{ nv, 1, 1 }, 1024, 0, &a);
    }

    /// qwen4_exp gdn.replay (gdn_replay_cuda): the state after `rows` saved rows (k, v, g, beta of gdnChain).
    pub fn gdnReplay(o: Ops, nv: usize, state_in: u64, k_save: u64, v_save: u64, g_save: u64, b_save: u64, rows: usize, state_out: u64) !void {
        const h = try heads(nv);
        var a: cuda.Args = .{};
        for ([_]u64{ state_in, k_save, v_save, g_save, b_save }) |v| a.add(v);
        a.add(int(rows));
        a.add(state_out);
        try o.go(o.k.replay[h], .{ nv, 1, 1 }, 1024, 0, &a);
    }

    /// gdn_io.front (gdn_front_cuda): per (row, head) block, conv + SiLU and the q/k L2 norms (fp32 q, k), v (bf16),
    /// g and beta; `windows` (rows, 4) int32 taps, `sid` a stream id a row into `conv_ptrs` (int64 addresses).
    pub fn gdnFront(o: Ops, nv: usize, p: u64, conv_ptrs: u64, sid: u64, windows: u64, rows: usize, conv_w: u64, a_log: u64, dt_bias: u64, q: u64, k: u64, v: u64, g: u64, beta: u64) !void {
        const h = try heads(nv);
        if (rows == 0) return;
        var a: cuda.Args = .{};
        for ([_]u64{ p, conv_ptrs, sid, windows, conv_w, a_log, dt_bias, q, k, v, g, beta }) |x| a.add(x);
        try o.go(o.k.front[h], .{ rows, nv / 3 + nv, 1 }, 128, 0, &a);
    }

    /// gdn_io.back (gdn_back_cuda): y (rows, nv, 128) bf16 -> gated RMSNorm (z from p) into out and its 32-group sums.
    pub fn gdnBack(o: Ops, nv: usize, y: u64, p: u64, norm_w: u64, eps: f32, out: u64, xs: u64, rows: usize) !void {
        const h = try heads(nv);
        if (rows == 0) return;
        var a: cuda.Args = .{};
        for ([_]u64{ y, p, norm_w }) |x| a.add(x);
        a.add(eps);
        a.add(out);
        a.add(xs);
        try o.go(o.k.back[h], .{ rows, nv, 1 }, 128, 0, &a);
    }

    /// cuda/kernels/gdn.chain (prefill, gdn_prefill_cuda): `w` rows from `state` (read only) to `last`, y (w, hv, 128)
    /// bf16 (the caller's buffer; the wrapper allocates it). 128 value rows a block when hv >= SMs, else 64.
    pub fn gdnPrefill(o: Ops, fp32_keys: bool, q: u64, k: u64, v: u64, g: u64, beta: u64, state: u64, last: u64, y: u64, w: usize, hk: usize, hv: usize) !void {
        if (w == 0) return;
        const wide = hv >= @as(usize, @intCast(o.k.sms));
        const rows: usize = if (wide) 128 else 64;
        var a: cuda.Args = .{};
        for ([_]u64{ q, k, v, g, beta, state, last, y }) |x| a.add(x);
        for ([_]usize{ w, hk, hv }) |x| a.add(int(x));
        try o.go(o.k.prefill[@intFromBool(!fp32_keys)][@intFromBool(!wide)], .{ hv, 128 / rows, 1 }, 2 * rows, 0, &a);
    }

    /// cuda/kernels/gdn.tree with fp32 keys (gdn_tree_cuda -> dispatch_tree<float>): a window's chains or trees from
    /// one `state` (streams 1) or a `table` of every stream's states with `starts`; `tree_plan` (nodes, 3) int32.
    pub fn gdnTree(o: Ops, q: u64, k: u64, v: u64, g: u64, beta: u64, state: u64, table: u64, starts: u64, tree_plan: u64, nodes: usize, slots: u32, streams: usize, max_rows: usize, y: u64, hk: usize, hv: usize, dv: usize, pending: Pending, final_state: u64, final_table: u64) !void {
        if (slots > 32 or max_rows < 1 or (slots != 0 and max_rows > 1024)) return error.Invalid;
        const i = treeIndex(slots, u(streams), o.k.major, o.k.minor, o.k.sms);
        const t = tree_variants[i];
        const shared = treeShared(t, max_rows);
        if (shared > 48 * 1024) try o.k.tree[i].allowDynamicShared(u(shared));
        const step = t.r * t.warps;
        const vec = dv % t.r == 0 and v % (2 * @as(u64, @min(t.r, 8))) == 0;
        var a: cuda.Args = .{};
        for ([_]u64{ q, k, v, g, beta, state, table, starts, tree_plan }) |x| a.add(x);
        a.add(int(nodes));
        a.add(y);
        for ([_]usize{ hk, hv, dv }) |x| a.add(int(x));
        a.add(pending);
        a.add(final_state);
        a.add(final_table);
        a.add(vec);
        try o.go(o.k.tree[i], .{ (dv + step - 1) / step, hv, streams }, 32 * t.warps, shared, &a);
    }

    /// cuda/kernels/gdn.replay with fp32 keys (gdn_replay_cuda): each (stream, layer) replays its accepted rows; `out`
    /// 0 writes over the table's states (in place). Rows and counts strides in elements.
    pub fn gdnTreeReplay(o: Ops, table: u64, layers: usize, streams: usize, rows: u64, row_stride: usize, counts: u64, count_stride: usize, out: u64, hk: usize, hv: usize, dv: usize) !void {
        var a: cuda.Args = .{};
        a.add(table);
        a.add(int(layers));
        a.add(rows);
        a.add(int(row_stride));
        a.add(counts);
        a.add(int(count_stride));
        a.add(out);
        for ([_]usize{ hk, hv, dv }) |x| a.add(int(x));
        try o.go(o.k.tree_replay, .{ (dv + 31) / 32, hv, layers * streams }, 128, 0, &a);
    }

    /// experts.route (experts_plan_cuda): pairs grouped by expert into items of at most `tile` pairs.
    pub fn plan(o: Ops, picks: u64, pairs: usize, experts: usize, tile: usize, p: Plan) !void {
        if (experts > plan_max_experts or (tile != 16 and tile != 64)) return error.Invalid;
        if (pairs <= plan_small) {
            var a: cuda.Args = .{};
            a.add(picks);
            for ([_]usize{ pairs, experts, tile }) |v| a.add(int(v));
            for ([_]u64{ p.members, p.items, p.counts }) |v| a.add(v);
            return o.go(o.k.plan_small, .{ 1, 1, 1 }, 1024, 0, &a);
        }
        const nblk = (pairs + 1023) / 1024;
        var a: cuda.Args = .{};
        a.add(picks);
        a.add(int(pairs));
        a.add(int(experts));
        a.add(p.rank);
        a.add(p.hist);
        try o.go(o.k.plan_rank, .{ nblk, 1, 1 }, 1024, 0, &a);
        var b: cuda.Args = .{};
        for ([_]usize{ nblk, experts, tile }) |v| b.add(int(v));
        for ([_]u64{ p.hist, p.items, p.counts }) |v| b.add(v);
        try o.go(o.k.plan_offsets, .{ 1, 1, 1 }, 1024, 0, &b);
        var c: cuda.Args = .{};
        c.add(picks);
        c.add(int(pairs));
        c.add(int(experts));
        for ([_]u64{ p.rank, p.hist, p.members }) |v| c.add(v);
        try o.go(o.k.plan_scatter, .{ (pairs + 255) / 256, 1, 1 }, 256, 0, &c);
    }

    /// nvfp4 experts (nvfp4_experts_cuda): epilogue 2 (gate/up SwiGLU, bf16), 0 (down, fp32) or 3 (down, bf16);
    /// items of expert `skip` untouched; the grid is ceil(max_units / 4) capped at the resident blocks.
    pub fn nvfp4Experts(o: Ops, epi: u8, x: u64, x_stride: usize, slots: usize, w: u64, scale: u64, kg: usize, nb: usize, p: Plan, out: u64, n: usize, limit: f32, skip: c_int, max_units: usize) !void {
        const i: usize = switch (epi) {
            2 => 0,
            0 => 1,
            3 => 2,
            else => return error.Invalid,
        };
        const grid = @min((max_units + 3) / 4, o.k.nvfp4_blocks[i]);
        if (grid < 1) return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(int(x_stride));
        a.add(int(slots));
        a.add(w);
        a.add(scale);
        a.add(int(kg));
        a.add(int(nb));
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        a.add(int(n));
        a.add(limit);
        a.add(skip);
        try o.go(o.k.nvfp4[i], .{ grid, 1, 1 }, 128, 0, &a);
    }

    /// nvfp4/experts.gate_up: x [R, D] bf16 (token rows) -> out [R * slots, NI] bf16 SwiGLU of each routed pair.
    pub fn nvfp4GateUp(o: Ops, x: u64, x_stride: usize, ex: Experts4, p: Plan, plan_slots: usize, plan_experts: usize, out: u64, rows: usize, skip: c_int) !void {
        const nb = ex.width / 32;
        const units = maxItems(rows * plan_slots, plan_experts, plan_tile) * nb;
        if (units <= o.k.nt_units) return o.nvfp4GateUpNt(x, x_stride, ex, p, plan_slots, out, skip, units);
        try o.nvfp4Experts(2, x, x_stride, plan_slots, ex.up, ex.up_scale, ex.dims / 32, nb, p, out, ex.width, ex.limit, skip, units);
    }

    /// nvfp4GateUp in fn_nvfp4_shape's one-tile-a-warp shape: each unit's four n8 tiles in four one-warp blocks
    /// (grid: 4 units' worth capped at the resident blocks), every output element the original kernel's bits.
    fn nvfp4GateUpNt(o: Ops, x: u64, x_stride: usize, ex: Experts4, p: Plan, plan_slots: usize, out: u64, skip: c_int, units: usize) !void {
        const grid = @min(units * 4, o.k.nvfp4_nt_blocks);
        if (grid < 1) return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(int(x_stride));
        a.add(int(plan_slots));
        a.add(ex.up);
        a.add(ex.up_scale);
        a.add(int(ex.dims / 32));
        a.add(int(ex.width / 32));
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        a.add(int(ex.width));
        a.add(ex.limit);
        a.add(skip);
        try o.go(o.k.nvfp4_nt, .{ grid, 1, 1 }, 32, 0, &a);
    }

    /// nvfp4/experts.down: act [R * slots, NI] bf16 (pair rows) -> out [R * slots, D], fp32 (`f32`) or bf16.
    pub fn nvfp4Down(o: Ops, act: u64, act_stride: usize, ex: Experts4, p: Plan, plan_slots: usize, plan_experts: usize, out: u64, f32_out: bool, rows: usize, skip: c_int) !void {
        const nb = ex.dims / 32;
        const units = maxItems(rows * plan_slots, plan_experts, plan_tile) * nb;
        try o.nvfp4Experts(if (f32_out) 0 else 3, act, act_stride, 0, ex.down, ex.down_scale, ex.width / 32, nb, p, out, ex.dims, 0, skip, units);
    }

    /// qmm.matmul for groups of 32 (qmm_cuda, one K slice): x (m, k) bf16 rows `x_stride` apart with group sums xs
    /// (m, k/32) -> out (m, n) bf16 or fp32. The draft head's shapes take one slice; more are refused.
    pub fn qmm32(o: Ops, x: u64, x_stride: usize, xs: u64, q: Q4, out: u64, m: usize, f32_out: bool) !void {
        if (m < 1) return error.Invalid;
        if (splitK32(q.n, q.k) != 1) return error.SplitKUnsupported;
        const bm = bucket(m);
        const rows_t = (m + bm - 1) / bm;
        var a: cuda.Args = .{};
        for ([_]u64{ x, xs, q.w, q.s, q.b, out, 0 }) |v| a.add(v);
        for ([_]usize{ m, q.n, q.k, 1, q.npad, if (m == 1) q.k else x_stride, l2Group(rows_t, bm, q.k) }) |v| a.add(int(v));
        try o.go(o.k.qmm[@intFromBool(f32_out)][bucketIndex(bm)], .{ rows_t * ((q.n + 63) / 64), 1, 1 }, 128, qmmShared(bm), &a);
    }

    /// qmm.matmul for groups of 32 at any K-slice count qmm.split_k gives (decode D5, the MTP head's matrices):
    /// one slice as qmm32, else up to 8 slices summed in slice order inside a cluster (qmm_cuda on sm_90+).
    pub fn qmm32Split(o: Ops, x: u64, x_stride: usize, xs: u64, q: Q4, out: u64, m: usize, f32_out: bool) !void {
        if (m < 1) return error.Invalid;
        const sk = splitK32(q.n, q.k);
        if (sk == 1) return o.qmm32(x, x_stride, xs, q, out, m, f32_out);
        const bm = bucket(m);
        const rows_t = (m + bm - 1) / bm;
        var a: cuda.Args = .{};
        for ([_]u64{ x, xs, q.w, q.s, q.b, out, 0 }) |v| a.add(v);
        for ([_]usize{ m, q.n, q.k, sk, q.npad, if (m == 1) q.k else x_stride, l2Group(rows_t, bm, q.k) }) |v| a.add(int(v));
        try cuda.launch.launch(o.k.qmm_cluster[@intFromBool(f32_out)][bucketIndex(bm)], .{
            .grid = .{ .x = u(rows_t * ((q.n + 63) / 64)), .y = 1, .z = u(sk) },
            .block = .{ .x = 128 },
            .shared = qmmShared(bm),
            .cluster = .{ .x = 1, .y = 1, .z = u(sk) },
        }, o.s, &a);
    }

    /// qmm.prefill_matmul for groups of 32 (qmm_prefill_cuda, tile 0): weights rounded once to bf16, one fp32 chain
    /// over K; any chunking gives the same bits.
    pub fn qmmPrefill32(o: Ops, x: u64, x_stride: usize, q: Q4, out: u64, m: usize, f32_out: bool) !void {
        if (m < 1) return error.Invalid;
        const rows_t = (m + 127) / 128;
        var a: cuda.Args = .{};
        for ([_]u64{ x, q.w, q.s, q.b, out }) |v| a.add(v);
        for ([_]usize{ m, q.n, q.k, q.npad, if (m == 1) q.k else x_stride, l2Group(rows_t, 128, q.k) }) |v| a.add(int(v));
        try o.go(o.k.qmm_prefill[@intFromBool(f32_out)], .{ rows_t * ((q.n + 127) / 128), 1, 1 }, 256, qmm_prefill_smem, &a);
    }
};

test "shared memory, tiles and tree picks follow the C++ wrappers" {
    try std.testing.expectEqual(@as(u32, 9472), qmmShared(16));
    try std.testing.expectEqual(@as(u32, 13824), qmmShared(32));
    try std.testing.expectEqual(@as(u32, 22528), qmmShared(64));
    try std.testing.expectEqual(@as(u32, 32256), qmm_prefill_smem);
    try std.testing.expectEqual(@as(usize, 16), bucket(1));
    try std.testing.expectEqual(@as(usize, 32), bucket(17));
    try std.testing.expectEqual(@as(usize, 64), bucket(33));
    // the draft head (79,591 ids, half a rank at TP=2) over 2560 inputs: one K slice
    try std.testing.expectEqual(@as(usize, 1), splitK32(79591, 2560));
    try std.testing.expectEqual(@as(usize, 1), splitK32(39796, 2560));
    try std.testing.expectEqual(@as(usize, 1), l2Group(1, 16, 2560));
    try std.testing.expectEqual(@as(usize, 0), treeIndex(0, 4, 12, 1, 48));
    try std.testing.expectEqual(@as(usize, 3), treeIndex(1, 4, 12, 1, 48)); // GB10 is never "wide"
    try std.testing.expectEqual(@as(usize, 1), treeIndex(1, 4, 12, 0, 188));
    try std.testing.expectEqual(@as(usize, 7), treeIndex(17, 1, 12, 1, 48));
    try std.testing.expectEqual(@as(usize, 32768 + 12 * 16), treeShared(tree_variants[5], 16));
    try std.testing.expect(treeShared(tree_variants[7], 1024) <= 48 * 1024); // no variant opts in past 48 KiB
    try std.testing.expectEqual(@as(usize, 0), treeShared(tree_variants[0], 4096));
    try std.testing.expectEqual(@as(usize, 1154), maxItems(16384, 130, 16));
    try std.testing.expectEqualStrings("_ZN14tf_fn_gdn_tree11tree_kernelIfLi0ELi8ELi4ELb1EEEvPKT_S3_PK13__nv_bfloat16PKfS8_S8_PKxPKiSC_iPS4_iiiNS_7PendingIS1_EEPfSA_b", tree_symbols[0]);
    try std.testing.expectError(error.UnsupportedHeads, heads(32));
    // decode D3's bound: one TP=2 row (110 units) and two (230) take the one-tile shape, a TP=1 row too (220)
    try std.testing.expectEqual(@as(usize, 110), maxItems(11, 513, 16) * 10);
    try std.testing.expectEqual(@as(usize, 230), maxItems(22, 513, 16) * 10);
    try std.testing.expectEqual(@as(usize, 230), ntUnits(null));
    try std.testing.expectEqual(@as(usize, 0), ntUnits("0"));
}
