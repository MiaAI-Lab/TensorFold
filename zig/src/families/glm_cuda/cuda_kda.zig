//! GLM-5.3-Flash KDA launches, segment table, state-pool layout and conv-window shift.
//!
//! Device code: zig/kernels/cuda/glm_kda.cu (families/glm5_next/cuda/kda.cu) and glm_kda_chunk.cu
//! (kda_chunk.cu). Grids are the torch wrappers kda.cpp / kda_chunk.cpp (kda_chain_cuda,
//! kda_chain_wide_cuda, kda_chain_wide_segments_cuda, kda_replay_*_cuda, kda_chunk_cuda).
//! The segment table is kda.py `segment_table`. The conv shift is forward.py `_conv_shift` /
//! `_conv_shift_seg` (Triton): the host functions below move the same bf16 bits; the device
//! kernels are listed for the AOT-replay package (P1) and are not ported here.
//! State shapes: forward.py `Slots` / kda.py `KDAScratchSet`, SPEC section 4.

const std = @import("std");
const cuda = @import("cuda");

pub const dk: usize = 128;
pub const dv: usize = 128;
pub const taps: usize = 4;
/// Conv state rows kept between tokens (the 4-tap conv's history). Triton calls this TAPS.
pub const conv_rows: usize = 3;
pub const seg_cols: usize = 6;
pub const kda_layers: usize = 34;
pub const lin_heads: usize = 64;
/// TF_GLM_KDA_WIDE_ROWS default: every decode window takes prep / step<4,16> / out.
pub const wide_rows: usize = 1;
/// Recipe scripts/config.sh KDA_CHUNKED. Prompt chunks (a position is set) then take kda_chunk.cu.
pub const recipe_chunked: bool = true;
pub const gate_lower: f32 = -5.0;
pub const eps: f32 = 1e-5;
/// step_kernel / step_seg_kernel instantiation kda_chain_wide_*_cuda launches.
pub const step_warps: usize = 4;
pub const step_tr: usize = 16;
pub const step_block: usize = step_warps * 32;
/// chunk_kernel's block (kda_chunk.cu NT) and dynamic shared memory (sizeof(Smem), static_assert in the .cu).
pub const chunk_threads: usize = 256;
pub const chunk_smem: u32 = 98560;
pub const chunk_bt: usize = 32;

/// Triton kernels this package does not port. P1 captures them (SPEC section 7) and replays the cubin.
/// Grids are the Python launches at TP=2 (C = 3 * 32 * 128 = 12288, so cdiv(C, 1024) = 12).
pub const p1_triton = [_][]const u8{
    "_conv_shift families/glm5_next/cuda/forward.py:1121 grid (layers, cdiv(C,1024)) C=12288 TAPS=3 BLOCK=1024 warps 4",
    "_conv_shift_seg families/glm5_next/cuda/forward.py:1146 grid (34, 12, nseg) C=12288 TAPS=3 COLS=6 BLOCK=1024 warps 4",
};

pub const sym = struct {
    pub const chain: [:0]const u8 = "_ZN10tf_glm_kda12chain_kernelEiPK13__nv_bfloat16iiS2_iS2_iS2_S2_PKfS4_S4_S2_ffiPS0_PfS6_S5_S6_S6_";
    pub const replay: [:0]const u8 = "_ZN10tf_glm_kda13replay_kernelEiPKfS1_PK13__nv_bfloat16S1_S1_iPf";
    pub const replay_layers: [:0]const u8 = "_ZN10tf_glm_kda20replay_layers_kernelEiPKfmS1_PK13__nv_bfloat16S1_S1_mmiPf";
    pub const prep: [:0]const u8 = "_ZN10tf_glm_kda11prep_kernelEiPK13__nv_bfloat16iiS2_iS2_S2_PKfS4_fPfS5_PS0_S5_S5_";
    pub const step: [:0]const u8 = "_ZN10tf_glm_kda11step_kernelILi4ELi16EEEviPKfS2_S2_PK13__nv_bfloat16S2_S2_iPS3_Pf";
    pub const out: [:0]const u8 = "_ZN10tf_glm_kda10out_kernelEiPK13__nv_bfloat16S2_iS2_fPS0_";
    pub const prep_seg: [:0]const u8 = "_ZN10tf_glm_kda15prep_seg_kernelEiPKiiPK13__nv_bfloat16iiS4_iS4_mS4_PKfS6_fPfS7_PS2_S7_S7_";
    pub const step_seg: [:0]const u8 = "_ZN10tf_glm_kda15step_seg_kernelILi4ELi16EEEviPKiPfmmPKfS5_PK13__nv_bfloat16S5_S5_PS6_";
    pub const replay_seg: [:0]const u8 = "_ZN10tf_glm_kda24replay_layers_seg_kernelEiiPKiPfmmmPKfPK13__nv_bfloat16S4_S4_mm";
    pub const chunk: [:0]const u8 = "_ZN16tf_glm_kda_chunk12chunk_kernelEiPK13__nv_bfloat16iiS2_iS2_iS2_S2_PKfS4_S4_S2_ffiiPS0_Pf";
};

pub const Path = enum { serial, wide, chunk };

/// kda.py `chain`: a prompt position with chunking takes kda_chunk; else the wide path at `rows >= wide_rows`.
pub fn path(rows: usize, wide: ?bool, pos: ?usize, chunked: bool) Path {
    if (chunked and pos != null) return .chunk;
    if (wide orelse (rows >= wide_rows)) return .wide;
    return .serial;
}

/// kda_chunked.chain passes `pos % 32` as the sub-chunk offset.
pub fn chunkOff(pos: usize) usize {
    return pos % chunk_bt;
}

/// weights.py KDAW: proj row is [q | k | v | f_a | g_a | b], f_a and g_a 128 each (replicated), b one per head.
pub fn faOff(heads: usize) usize {
    return 3 * heads * dk;
}

pub fn gaOff(heads: usize) usize {
    return faOff(heads) + dk;
}

pub fn bOff(heads: usize) usize {
    return gaOff(heads) + dk;
}

pub fn projWidth(heads: usize) usize {
    return bOff(heads) + heads;
}

pub fn convChannels(heads: usize) usize {
    return 3 * heads * dk;
}

/// cuda/shares.py `share`: rank's [start, stop) of `total`, whole `unit`s, remainder to the lowest ranks.
pub fn share(total: usize, world: usize, rank: usize, unit: usize) !struct { start: usize, stop: usize } {
    if (world < 1 or unit < 1 or rank >= world or total % unit != 0) return error.Invalid;
    const units = total / unit;
    if (units < world) return error.Invalid;
    const base = units / world;
    const extra = units % world;
    var start: usize = 0;
    for (0..rank) |r| start += (base + @intFromBool(r < extra)) * unit;
    const n = (base + @intFromBool(rank < extra)) * unit;
    return .{ .start = start, .stop = start + n };
}

/// Element strides of rec [slots, 2, layers, heads, 128, 128] f32. A one-layer chain views
/// rec[:, :, layer] and passes slot/parity; replay_layers_segments passes all three.
pub const RecStride = struct { slot: usize, parity: usize, layer: usize };

pub fn recStride(layers: usize, heads: usize) RecStride {
    const layer = heads * dv * dk;
    const parity = layers * layer;
    return .{ .slot = 2 * parity, .parity = parity, .layer = layer };
}

/// conv [slots, layers, 3, C] bf16. A one-layer chain views conv[:, layer] (slot stride layers*3*C).
pub fn convSlotStride(layers: usize, heads: usize) usize {
    return layers * conv_rows * convChannels(heads);
}

/// KDAScratchSet element strides between layers: k/v/g are [layers, rows, heads, 128], b is [layers, rows, heads].
pub fn kvStride(rows: usize, heads: usize) usize {
    return rows * heads * dk;
}

pub fn betaStride(rows: usize, heads: usize) usize {
    return rows * heads;
}

/// Bytes of one stream slot's KDA tensors (forward.py Slots, one `count` entry). pos is two int32s
/// (pos_dev, mtp_pos_dev). Wide q/y scratch is process-global, not part of the slot.
pub const SlotBytes = struct {
    conv: usize,
    rec: usize,
    proj: usize,
    out: usize,
    k: usize,
    v: usize,
    g: usize,
    beta: usize,
    pos: usize,
    wide_q: usize,
    wide_y: usize,

    pub fn slot(self: SlotBytes) usize {
        return self.conv + self.rec + self.proj + self.out + self.k + self.v + self.g + self.beta + self.pos;
    }
};

pub fn slotBytes(layers: usize, heads: usize, rows: usize, width: usize) SlotBytes {
    const c = convChannels(heads);
    const state = heads * dv * dk;
    return .{
        .conv = layers * conv_rows * c * 2,
        .rec = 2 * layers * state * 4,
        .proj = layers * rows * width * 2,
        .out = layers * rows * heads * dv * 2,
        .k = layers * rows * heads * dk * 4,
        .v = layers * rows * heads * dv * 2,
        .g = layers * rows * heads * dk * 4,
        .beta = layers * rows * heads * 4,
        // forward.py slot_bytes adds a flat +16 for the int32 pair. The live tensors are two int32s (8 bytes);
        // the extra 8 bytes are the reference's accounting fudge, not a 16-byte stride.
        .pos = 16,
        .wide_q = rows * heads * dk * 4,
        .wide_y = rows * heads * dv * 2,
    };
}

/// One segment row, kda.py SEG_COLS: (first row, rows, state slot, parity, conv slot, keep).
pub const Seg = extern struct {
    row0: i32,
    rows: i32,
    slot: i32,
    parity: i32,
    conv: i32,
    keep: i32,
};

pub const SegIn = struct {
    rows: i32,
    slot: i32,
    parity: i32,
    conv: ?i32 = null,
    keep: ?i32 = null,
};

/// kda.py `segment_table`. Segments lay back to back from row 0. `out` past `in` becomes empty
/// (row0 = the window length, rows 0) so a CUDA graph's fixed table serves fewer streams.
pub fn writeSegments(in: []const SegIn, out: []Seg) error{ BadSegment, TooMany }!void {
    if (in.len > out.len) return error.TooMany;
    var row0: i32 = 0;
    for (in, 0..) |s, i| {
        const conv = s.conv orelse s.slot;
        const keep = s.keep orelse s.rows;
        if (s.rows < 0 or (s.parity != 0 and s.parity != 1) or keep < 0 or keep > s.rows) return error.BadSegment;
        out[i] = .{ .row0 = row0, .rows = s.rows, .slot = s.slot, .parity = s.parity, .conv = conv, .keep = keep };
        row0 += s.rows;
    }
    for (out[in.len..]) |*e| e.* = .{ .row0 = row0, .rows = 0, .slot = 0, .parity = 0, .conv = 0, .keep = 0 };
}

/// prep_seg_kernel's scan: the last segment that covers window row `r` (empty rows never cover). Uncovered = null.
pub fn segmentAt(table: []const Seg, r: i32) ?usize {
    var found: ?usize = null;
    for (table, 0..) |e, i| {
        if (r >= e.row0 and r < e.row0 + e.rows) found = i;
    }
    return found;
}

fn projAt(proj: []const u16, layer: usize, row: usize, c: usize, window: usize, row_stride: usize) error{Range}!u16 {
    if (c >= row_stride) return error.Range;
    const i = (layer * window + row) * row_stride + c;
    if (i >= proj.len) return error.Range;
    return proj[i];
}

fn convAt(conv: []const u16, layer: usize, row: usize, c: usize, C: usize) error{Range}!u16 {
    const i = (layer * conv_rows + row) * C + c;
    if (i >= conv.len) return error.Range;
    return conv[i];
}

/// forward.py `_shift_conv` / `_conv_shift`: conv [layers, 3, C] takes rows keep..keep+2 of [conv; proj].
/// proj is [layers, window, row_stride] with row_stride >= C. `row_base` is the segment's first window row
/// (0 for one stream): a proj source `src - 3` is window row `row_base + src - 3`, and `layer * window`
/// commutes with that add, matching `_conv_shift_seg`'s `PROJ + row0 * proj_row`. Loads all three sources
/// before storing, per channel.
pub fn convShift(conv: []u16, proj: []const u16, keep: i32, layers: usize, C: usize, window: usize, row_stride: usize, row_base: usize) error{Range}!void {
    if (keep < 0) return error.Range;
    var layer: usize = 0;
    while (layer < layers) : (layer += 1) {
        var c: usize = 0;
        while (c < C) : (c += 1) {
            var got: [3]u16 = undefined;
            var k: i32 = 0;
            while (k < 3) : (k += 1) {
                const src = keep + k;
                got[@intCast(k)] = if (src < conv_rows)
                    try convAt(conv, layer, @intCast(src), c, C)
                else
                    try projAt(proj, layer, row_base + @as(usize, @intCast(src)) - conv_rows, c, window, row_stride);
            }
            var t: usize = 0;
            while (t < 3) : (t += 1) {
                const i = (layer * conv_rows + t) * C + c;
                if (i >= conv.len) return error.Range;
                conv[i] = got[t];
            }
        }
    }
}

/// forward.py `conv_shift_segments`: conv [slots, layers, 3, C], proj [layers, window, row_stride].
/// Segment i shifts conv[conv slot] over its own rows; rows == 0 is skipped.
pub fn convShiftSegments(conv: []u16, proj: []const u16, table: []const Seg, layers: usize, C: usize, window: usize, row_stride: usize) error{Range}!void {
    const slot_elems = layers * conv_rows * C;
    for (table) |e| {
        if (e.rows <= 0) continue;
        if (e.conv < 0 or e.row0 < 0) return error.Range;
        const base: usize = @intCast(e.conv);
        const begin = base * slot_elems;
        const end = begin + slot_elems;
        if (end > conv.len) return error.Range;
        try convShift(conv[begin..end], proj, e.keep, layers, C, window, row_stride, @intCast(e.row0));
    }
}

/// kda.cu `warp_sum`: xor-shuffle butterfly, each step reading every lane's value from before that step.
pub fn warpSum(x: *[32]f32) void {
    var o: u32 = 16;
    while (o != 0) : (o >>= 1) {
        var next: [32]f32 = undefined;
        for (0..32) |lane| next[lane] = x[lane] + x[lane ^ o];
        x.* = next;
    }
}

/// kda.cu `update` over the whole 128 x 128 state. Warps do not interact. The same routine is the
/// chain's step and the replay, so a prefix of saved rows matches the serial state.
pub fn deltaStep(state: *[dv][dk]f32, k: *const [dk]f32, g: *const [dk]f32, v: *const [dv]f32, beta: f32) void {
    var warp: usize = 0;
    while (warp < 32) : (warp += 1) {
        var tile: [32][4][4]f32 = undefined;
        for (0..32) |lane| {
            for (0..4) |j| for (0..4) |i| {
                tile[lane][j][i] = state[warp * 4 + j][lane * 4 + i];
            };
        }
        for (0..4) |j| {
            var kv: [32]f32 = undefined;
            for (0..32) |lane| {
                var acc: f32 = 0;
                const kk = k[lane * 4 ..][0..4];
                const gg = g[lane * 4 ..][0..4];
                for (0..4) |i| {
                    tile[lane][j][i] = tile[lane][j][i] * gg[i];
                    acc = acc + tile[lane][j][i] * kk[i];
                }
                kv[lane] = acc;
            }
            warpSum(&kv);
            for (0..32) |lane| {
                const delta = (v[warp * 4 + j] - kv[lane]) * beta;
                const kk = k[lane * 4 ..][0..4];
                for (0..4) |i| tile[lane][j][i] = tile[lane][j][i] + kk[i] * delta;
            }
        }
        for (0..32) |lane| {
            for (0..4) |j| for (0..4) |i| {
                state[warp * 4 + j][lane * 4 + i] = tile[lane][j][i];
            };
        }
    }
}

pub const Geom = struct { grid: [3]u32, block: u32, shared: u32 = 0 };

fn grid3(x: usize, y: usize, z: usize) [3]u32 {
    return .{ @intCast(x), @intCast(y), @intCast(z) };
}

pub fn chainGeom(heads: usize) Geom {
    return .{ .grid = grid3(heads, 1, 1), .block = 1024 };
}

pub fn replayGeom(heads: usize) Geom {
    return .{ .grid = grid3(heads, 1, 1), .block = 1024 };
}

pub fn replayLayersGeom(layers: usize, heads: usize) Geom {
    return .{ .grid = grid3(layers * heads, 1, 1), .block = 1024 };
}

pub fn replaySegGeom(nseg: usize, layers: usize, heads: usize) Geom {
    return .{ .grid = grid3(nseg * layers * heads, 1, 1), .block = 1024 };
}

/// prep, step<4,16>, out. step's y grid is DV / 4 / WARPS = 8.
pub fn wideGeom(heads: usize, rows: usize) [3]Geom {
    const parts: usize = dv / 4 / step_warps;
    return .{
        .{ .grid = grid3(rows, heads, 1), .block = 512 },
        .{ .grid = grid3(heads, parts, 1), .block = step_block },
        .{ .grid = grid3(rows, heads, 1), .block = dv },
    };
}

pub fn wideSegGeom(heads: usize, rows: usize, nseg: usize) [3]Geom {
    const parts: usize = dv / 4 / step_warps;
    return .{
        .{ .grid = grid3(rows, heads, 1), .block = 512 },
        .{ .grid = grid3(nseg * heads, parts, 1), .block = step_block },
        .{ .grid = grid3(rows, heads, 1), .block = dv },
    };
}

pub fn chunkGeom(heads: usize) Geom {
    return .{ .grid = grid3(heads, 1, 1), .block = chunk_threads, .shared = chunk_smem };
}

fn int(x: usize) c_int {
    return @intCast(x);
}

pub const Kernels = struct {
    mods: [2]cuda.Module,
    chain: cuda.Function,
    replay: cuda.Function,
    replay_layers: cuda.Function,
    prep: cuda.Function,
    step: cuda.Function,
    out: cuda.Function,
    prep_seg: cuda.Function,
    step_seg: cuda.Function,
    replay_seg: cuda.Function,
    chunk: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.mods[0] = try cuda.Module.load(d, cuda.kernels.glm_kda);
        errdefer k.mods[0].unload();
        k.mods[1] = try cuda.Module.load(d, cuda.kernels.glm_kda_chunk);
        errdefer k.mods[1].unload();
        const a = k.mods[0];
        const b = k.mods[1];
        k.chain = try a.function(sym.chain);
        k.replay = try a.function(sym.replay);
        k.replay_layers = try a.function(sym.replay_layers);
        k.prep = try a.function(sym.prep);
        k.step = try a.function(sym.step);
        k.out = try a.function(sym.out);
        k.prep_seg = try a.function(sym.prep_seg);
        k.step_seg = try a.function(sym.step_seg);
        k.replay_seg = try a.function(sym.replay_seg);
        k.chunk = try b.function(sym.chunk);
        try k.chunk.allowDynamicShared(chunk_smem);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.mods[1].unload();
        k.mods[0].unload();
    }
};

pub const Ops = struct {
    k: *const Kernels,
    s: cuda.Stream,

    fn go(o: Ops, f: cuda.Function, g: Geom, args: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = g.grid[0], .y = g.grid[1], .z = g.grid[2] }, .block = .{ .x = g.block }, .shared = g.shared }, o.s, args);
    }

    /// kda_chain_cuda. Zero addresses are the wrapper's empty save / state_out tensors.
    pub fn chain(o: Ops, heads: usize, p: u64, p_stride: usize, b_off: usize, a: u64, a_stride: usize, g: u64, g_stride: usize, conv: u64, conv_w: u64, state_in: u64, a_log: u64, dt_bias: u64, norm_w: u64, eps_v: f32, lower: f32, rows: usize, out: u64, state_out: u64, k_save: u64, v_save: u64, g_save: u64, b_save: u64) !void {
        if (heads == 0 or rows == 0) return error.Invalid;
        var args: cuda.Args = .{};
        args.add(int(heads));
        args.add(p);
        args.add(int(p_stride));
        args.add(int(b_off));
        args.add(a);
        args.add(int(a_stride));
        args.add(g);
        args.add(int(g_stride));
        args.add(conv);
        args.add(conv_w);
        args.add(state_in);
        args.add(a_log);
        args.add(dt_bias);
        args.add(norm_w);
        args.add(eps_v);
        args.add(lower);
        args.add(int(rows));
        args.add(out);
        args.add(state_out);
        args.add(k_save);
        args.add(v_save);
        args.add(g_save);
        args.add(b_save);
        try o.go(o.k.chain, chainGeom(heads), &args);
    }

    /// kda_chain_wide_cuda: prep, step<4,16>, out. Same bits as `chain`.
    pub fn chainWide(o: Ops, heads: usize, p: u64, p_stride: usize, b_off: usize, a: u64, a_stride: usize, g: u64, g_stride: usize, conv: u64, conv_w: u64, state_in: u64, a_log: u64, dt_bias: u64, norm_w: u64, eps_v: f32, lower: f32, rows: usize, out: u64, state_out: u64, k_save: u64, v_save: u64, g_save: u64, b_save: u64, q_tmp: u64, y_tmp: u64) !void {
        if (heads == 0 or rows == 0) return error.Invalid;
        const geom = wideGeom(heads, rows);
        var prep_args: cuda.Args = .{};
        prep_args.add(int(heads));
        prep_args.add(p);
        prep_args.add(int(p_stride));
        prep_args.add(int(b_off));
        prep_args.add(a);
        prep_args.add(int(a_stride));
        prep_args.add(conv);
        prep_args.add(conv_w);
        prep_args.add(a_log);
        prep_args.add(dt_bias);
        prep_args.add(lower);
        prep_args.add(q_tmp);
        prep_args.add(k_save);
        prep_args.add(v_save);
        prep_args.add(g_save);
        prep_args.add(b_save);
        try o.go(o.k.prep, geom[0], &prep_args);
        var step_args: cuda.Args = .{};
        step_args.add(int(heads));
        step_args.add(state_in);
        step_args.add(q_tmp);
        step_args.add(k_save);
        step_args.add(v_save);
        step_args.add(g_save);
        step_args.add(b_save);
        step_args.add(int(rows));
        step_args.add(y_tmp);
        step_args.add(state_out);
        try o.go(o.k.step, geom[1], &step_args);
        var out_args: cuda.Args = .{};
        out_args.add(int(heads));
        out_args.add(y_tmp);
        out_args.add(g);
        out_args.add(int(g_stride));
        out_args.add(norm_w);
        out_args.add(eps_v);
        out_args.add(out);
        try o.go(o.k.out, geom[2], &out_args);
    }

    /// kda_replay_cuda. heads is b_save's head count (the wrapper reads b_save.size(1)).
    pub fn replay(o: Ops, heads: usize, state_in: u64, k_save: u64, v_save: u64, g_save: u64, b_save: u64, rows: usize, state_out: u64) !void {
        if (heads == 0) return error.Invalid;
        var args: cuda.Args = .{};
        args.add(int(heads));
        args.add(state_in);
        args.add(k_save);
        args.add(v_save);
        args.add(g_save);
        args.add(b_save);
        args.add(int(rows));
        args.add(state_out);
        try o.go(o.k.replay, replayGeom(heads), &args);
    }

    /// kda_replay_layers_cuda. Strides are elements, as the wrapper passes H*128*128 and rows*H*128.
    pub fn replayLayers(o: Ops, heads: usize, state_in: u64, state_stride: usize, k_save: u64, v_save: u64, g_save: u64, b_save: u64, kv_stride: usize, b_stride: usize, layers: usize, rows: usize, state_out: u64) !void {
        if (heads == 0 or layers == 0) return error.Invalid;
        var args: cuda.Args = .{};
        args.add(int(heads));
        args.add(state_in);
        args.add(state_stride);
        args.add(k_save);
        args.add(v_save);
        args.add(g_save);
        args.add(b_save);
        args.add(kv_stride);
        args.add(b_stride);
        args.add(int(rows));
        args.add(state_out);
        try o.go(o.k.replay_layers, replayLayersGeom(layers, heads), &args);
    }

    /// kda_chain_wide_segments_cuda. `conv_slot` / `slot` / `parity` are element strides of the views
    /// kda.cpp passes (conv.stride(0), rec.stride(0), rec.stride(1)).
    pub fn chainWideSegments(o: Ops, nseg: usize, heads: usize, seg: u64, p: u64, p_stride: usize, b_off: usize, a: u64, a_stride: usize, g: u64, g_stride: usize, conv: u64, conv_slot: usize, conv_w: u64, rec: u64, slot: usize, parity: usize, a_log: u64, dt_bias: u64, norm_w: u64, eps_v: f32, lower: f32, rows: usize, out: u64, k_save: u64, v_save: u64, g_save: u64, b_save: u64, q_tmp: u64, y_tmp: u64) !void {
        if (heads == 0 or rows == 0 or nseg == 0) return error.Invalid;
        const geom = wideSegGeom(heads, rows, nseg);
        var prep_args: cuda.Args = .{};
        prep_args.add(int(heads));
        prep_args.add(seg);
        prep_args.add(int(nseg));
        prep_args.add(p);
        prep_args.add(int(p_stride));
        prep_args.add(int(b_off));
        prep_args.add(a);
        prep_args.add(int(a_stride));
        prep_args.add(conv);
        prep_args.add(conv_slot);
        prep_args.add(conv_w);
        prep_args.add(a_log);
        prep_args.add(dt_bias);
        prep_args.add(lower);
        prep_args.add(q_tmp);
        prep_args.add(k_save);
        prep_args.add(v_save);
        prep_args.add(g_save);
        prep_args.add(b_save);
        try o.go(o.k.prep_seg, geom[0], &prep_args);
        var step_args: cuda.Args = .{};
        step_args.add(int(heads));
        step_args.add(seg);
        step_args.add(rec);
        step_args.add(slot);
        step_args.add(parity);
        step_args.add(q_tmp);
        step_args.add(k_save);
        step_args.add(v_save);
        step_args.add(g_save);
        step_args.add(b_save);
        step_args.add(y_tmp);
        try o.go(o.k.step_seg, geom[1], &step_args);
        var out_args: cuda.Args = .{};
        out_args.add(int(heads));
        out_args.add(y_tmp);
        out_args.add(g);
        out_args.add(int(g_stride));
        out_args.add(norm_w);
        out_args.add(eps_v);
        out_args.add(out);
        try o.go(o.k.out, geom[2], &out_args);
    }

    /// kda_replay_layers_segments_cuda. Segments with rows <= 0 or keep >= rows are skipped on device.
    pub fn replayLayersSegments(o: Ops, nseg: usize, heads: usize, layers: usize, seg: u64, rec: u64, slot: usize, parity: usize, layer: usize, k_save: u64, v_save: u64, g_save: u64, b_save: u64, kv_stride: usize, b_stride: usize) !void {
        if (heads == 0 or layers == 0 or nseg == 0) return error.Invalid;
        var args: cuda.Args = .{};
        args.add(int(heads));
        args.add(int(layers));
        args.add(seg);
        args.add(rec);
        args.add(slot);
        args.add(parity);
        args.add(layer);
        args.add(k_save);
        args.add(v_save);
        args.add(g_save);
        args.add(b_save);
        args.add(kv_stride);
        args.add(b_stride);
        try o.go(o.k.replay_seg, replaySegGeom(nseg, layers, heads), &args);
    }

    /// kda_chunk_cuda. `off` is the first row's place in its 32-row sub-chunk (0 .. 31).
    pub fn chunk(o: Ops, heads: usize, p: u64, p_stride: usize, b_off: usize, a: u64, a_stride: usize, g: u64, g_stride: usize, conv: u64, conv_w: u64, state_in: u64, a_log: u64, dt_bias: u64, norm_w: u64, eps_v: f32, lower: f32, rows: usize, off: usize, out: u64, state_out: u64) !void {
        if (heads == 0 or rows == 0 or off >= chunk_bt) return error.Invalid;
        var args: cuda.Args = .{};
        args.add(int(heads));
        args.add(p);
        args.add(int(p_stride));
        args.add(int(b_off));
        args.add(a);
        args.add(int(a_stride));
        args.add(g);
        args.add(int(g_stride));
        args.add(conv);
        args.add(conv_w);
        args.add(state_in);
        args.add(a_log);
        args.add(dt_bias);
        args.add(norm_w);
        args.add(eps_v);
        args.add(lower);
        args.add(int(rows));
        args.add(int(off));
        args.add(out);
        args.add(state_out);
        try o.go(o.k.chunk, chunkGeom(heads), &args);
    }
};

const kd = @import("cuda_kda_interface.zig");

/// P6 installs this on Forward before graph capture. The default Interface stays MissingKdaPackage.
pub fn install(k: *const Kernels) kd.Interface {
    return .{ .ctx = @ptrCast(@constCast(k)), .chain_segments = launchSegments };
}

fn launchSegments(ctx: ?*anyopaque, stream: cuda.Stream, r: kd.Request) !void {
    const k: *const Kernels = @ptrCast(@alignCast(ctx orelse return error.MissingKdaPackage));
    try (Ops{ .k = k, .s = stream }).chainWideSegments(r.nseg, r.heads, r.seg, r.p, r.p_stride, r.b_off, r.a, r.a_stride, r.g, r.g_stride, r.conv, r.conv_slot, r.conv_w, r.rec, r.rec_slot, r.rec_parity, r.a_log, r.dt_bias, r.norm_w, r.eps, r.lower, r.rows, r.scratch.out, r.scratch.k, r.scratch.v, r.scratch.g, r.scratch.b, r.scratch.q_tmp, r.scratch.y_tmp);
}

/// forward.py `slot_bytes`: one more stream slot, including the flat +16 on the position pair.
pub fn referenceSlotBytes(layers: usize, heads: usize, rows: usize, width: usize) usize {
    const scratch = rows * heads * dv * 2 + rows * heads * dk * 4 + rows * heads * dv * 2 + rows * heads * dk * 4 + rows * heads * 4;
    return layers * (taps - 1) * 3 * heads * dk * 2 + 2 * layers * heads * dk * dv * 4 + layers * rows * width * 2 + layers * scratch + 16;
}

test "segment table matches kda.py, including padding and last-match coverage" {
    var out: [4]Seg = undefined;
    try writeSegments(&.{
        .{ .rows = 3, .slot = 1, .parity = 0 },
        .{ .rows = 2, .slot = 4, .parity = 1, .conv = 7, .keep = 1 },
    }, &out);
    try std.testing.expectEqual(Seg{ .row0 = 0, .rows = 3, .slot = 1, .parity = 0, .conv = 1, .keep = 3 }, out[0]);
    try std.testing.expectEqual(Seg{ .row0 = 3, .rows = 2, .slot = 4, .parity = 1, .conv = 7, .keep = 1 }, out[1]);
    try std.testing.expectEqual(Seg{ .row0 = 5, .rows = 0, .slot = 0, .parity = 0, .conv = 0, .keep = 0 }, out[2]);
    try std.testing.expectEqual(Seg{ .row0 = 5, .rows = 0, .slot = 0, .parity = 0, .conv = 0, .keep = 0 }, out[3]);
    try std.testing.expectEqual(@as(?usize, 0), segmentAt(&out, 0));
    try std.testing.expectEqual(@as(?usize, 0), segmentAt(&out, 2));
    try std.testing.expectEqual(@as(?usize, 1), segmentAt(&out, 3));
    try std.testing.expectEqual(@as(?usize, null), segmentAt(&out, 5));
    try std.testing.expectError(error.BadSegment, writeSegments(&.{.{ .rows = 2, .slot = 0, .parity = 2 }}, &out));
    try std.testing.expectError(error.BadSegment, writeSegments(&.{.{ .rows = 2, .slot = 0, .parity = 0, .keep = 3 }}, &out));
    try std.testing.expectError(error.TooMany, writeSegments(&.{ .{ .rows = 1, .slot = 0, .parity = 0 }, .{ .rows = 1, .slot = 0, .parity = 0 } }, out[0..1]));
    // an overlapping later segment wins, which is what prep_seg_kernel's scan does
    const overlap: [2]Seg = .{
        .{ .row0 = 0, .rows = 4, .slot = 0, .parity = 0, .conv = 0, .keep = 4 },
        .{ .row0 = 2, .rows = 2, .slot = 1, .parity = 1, .conv = 1, .keep = 2 },
    };
    try std.testing.expectEqual(@as(?usize, 1), segmentAt(&overlap, 2));
    try std.testing.expectEqual(@as(?usize, 0), segmentAt(&overlap, 1));
}

test "conv shift moves the same bf16 bits as _conv_shift, reading before it writes" {
    const C: usize = 4;
    const W: usize = 4;
    const layers: usize = 2;
    const window: usize = 4;
    var conv: [2 * 3 * 4]u16 = undefined;
    var proj: [2 * 4 * 4]u16 = undefined;
    for (&conv, 0..) |*x, i| x.* = @intCast(100 + i);
    for (&proj, 0..) |*x, i| x.* = @intCast(200 + i);
    const before = conv;
    try convShift(&conv, &proj, 0, layers, C, window, W, 0);
    try std.testing.expectEqualSlices(u16, &before, &conv);
    try convShift(&conv, &proj, 1, layers, C, window, W, 0);
    // layer 0: row1, row2, proj row 0. layer 1 is the same relative to its own base.
    const l0 = conv[0 .. 3 * C];
    try std.testing.expectEqualSlices(u16, before[C .. 2 * C], l0[0..C]);
    try std.testing.expectEqualSlices(u16, before[2 * C .. 3 * C], l0[C .. 2 * C]);
    try std.testing.expectEqualSlices(u16, proj[0..C], l0[2 * C .. 3 * C]);
    const l1 = conv[3 * C ..];
    try std.testing.expectEqualSlices(u16, before[4 * C .. 5 * C], l1[0..C]);
    try std.testing.expectEqualSlices(u16, proj[window * W ..][0..C], l1[2 * C .. 3 * C]);
    // keep = 4 reads proj rows 1, 2, 3
    conv = before;
    try convShift(&conv, &proj, 4, layers, C, window, W, 0);
    try std.testing.expectEqualSlices(u16, proj[W .. 2 * W], conv[0..C]);
    try std.testing.expectEqualSlices(u16, proj[3 * W .. 4 * W], conv[2 * C .. 3 * C]);
}

test "segment conv shift skips empty segments and uses each conv slot's own rows" {
    const C: usize = 2;
    const layers: usize = 1;
    const window: usize = 4;
    var conv = [_]u16{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    // two slots. proj window rows: 20 21 | 22 23 | 24 25 | 26 27
    const proj = [_]u16{ 20, 21, 22, 23, 24, 25, 26, 27 };
    var table: [3]Seg = undefined;
    try writeSegments(&.{
        .{ .rows = 2, .slot = 0, .parity = 0, .keep = 1 },
        .{ .rows = 0, .slot = 1, .parity = 1 },
        .{ .rows = 2, .slot = 1, .parity = 1, .keep = 2 },
    }, &table);
    // writeSegments of 3 into 3: no padding. second is rows 0 at row0 2, third starts at row 2.
    try std.testing.expectEqual(@as(i32, 2), table[2].row0);
    const slot1 = conv[6..12].*;
    try convShiftSegments(&conv, &proj, &table, layers, C, window, C);
    // slot 0 keep 1: conv rows 1, 2 then proj row 0 (the segment starts at row 0)
    try std.testing.expectEqualSlices(u16, &.{ 3, 4, 5, 6, 20, 21 }, conv[0..6]);
    // empty segment does not touch slot 1; the third segment keep 2 starts at window row 2:
    // new = conv row 2, proj local 0 (= window row 2), proj local 1 (= window row 3)
    try std.testing.expectEqualSlices(u16, slot1[4..6], conv[6..8]); // overwritten? row0 of slot becomes old row2
    try std.testing.expectEqualSlices(u16, &.{ 11, 12, 24, 25, 26, 27 }, conv[6..12]);
}

test "segment conv shift uses row0 inside each layer's own window" {
    const C: usize = 2;
    const layers: usize = 2;
    const window: usize = 4;
    var conv: [2 * 3 * 2]u16 = undefined;
    for (&conv, 0..) |*x, i| x.* = @intCast(10 + i);
    var proj: [2 * 4 * 2]u16 = undefined;
    for (&proj, 0..) |*x, i| x.* = @intCast(100 + i);
    var table: [1]Seg = .{.{ .row0 = 2, .rows = 2, .slot = 0, .parity = 0, .conv = 0, .keep = 2 }};
    try convShiftSegments(&conv, &proj, &table, layers, C, window, C);
    // keep 2 at window row 2: conv row 2, then proj rows 2 and 3 of that layer.
    try std.testing.expectEqualSlices(u16, &.{ 14, 15, 104, 105, 106, 107 }, conv[0..6]);
    try std.testing.expectEqualSlices(u16, &.{ 20, 21, 112, 113, 114, 115 }, conv[6..12]);
}

test "an installed KDA callback still fails closed when the kernel table is absent" {
    const face: kd.Interface = .{ .ctx = null, .chain_segments = launchSegments };
    try std.testing.expectError(error.MissingKdaPackage, face.run(undefined, undefined));
}

test "pool shapes, TP shares and launch grids match the Python wrappers" {
    try std.testing.expectEqual(@as(usize, 12576), projWidth(32));
    try std.testing.expectEqual(@as(usize, 12288), convChannels(32));
    try std.testing.expectEqual(@as(usize, 12544), bOff(32));
    const tp2 = try share(lin_heads, 2, 1, 1);
    try std.testing.expectEqual(@as(usize, 32), tp2.start);
    try std.testing.expectEqual(@as(usize, 64), tp2.stop);
    const tp3 = try share(lin_heads, 3, 0, 1);
    try std.testing.expectEqual(@as(usize, 0), tp3.start);
    try std.testing.expectEqual(@as(usize, 22), tp3.stop);
    const tp3b = try share(lin_heads, 3, 2, 1);
    try std.testing.expectEqual(@as(usize, 43), tp3b.start);
    try std.testing.expectEqual(@as(usize, 64), tp3b.stop);
    const bytes = slotBytes(kda_layers, 32, 8, projWidth(32));
    try std.testing.expectEqual(@as(usize, 142_606_336), bytes.rec);
    try std.testing.expectEqual(@as(usize, 16), bytes.pos);
    try std.testing.expectEqual(referenceSlotBytes(kda_layers, 32, 8, projWidth(32)), bytes.slot());
    try std.testing.expectEqual(referenceSlotBytes(kda_layers, 22, 1, projWidth(22)), slotBytes(kda_layers, 22, 1, projWidth(22)).slot());
    const st = recStride(kda_layers, 32);
    try std.testing.expectEqual(@as(usize, 32 * dk * dv), st.layer);
    try std.testing.expectEqual(kda_layers * st.layer, st.parity);
    try std.testing.expectEqual(2 * st.parity, st.slot);
    try std.testing.expectEqual(@as(usize, kda_layers * conv_rows * 12288), convSlotStride(kda_layers, 32));
    // SPEC 2a: prep (R, 32) x 512, step (nseg*32, 8) x 128, out (R, 32) x 128; replay (nseg*34*32) x 1024
    const g = wideSegGeom(32, 8, 4);
    try std.testing.expectEqual([3]u32{ 8, 32, 1 }, g[0].grid);
    try std.testing.expectEqual(@as(u32, 512), g[0].block);
    try std.testing.expectEqual([3]u32{ 4 * 32, 8, 1 }, g[1].grid);
    try std.testing.expectEqual(@as(u32, 128), g[1].block);
    try std.testing.expectEqual([3]u32{ 8, 32, 1 }, g[2].grid);
    try std.testing.expectEqual(@as(u32, 128), g[2].block);
    const rg = replaySegGeom(4, kda_layers, 32);
    try std.testing.expectEqual([3]u32{ 4 * 34 * 32, 1, 1 }, rg.grid);
    try std.testing.expectEqual(@as(u32, 1024), rg.block);
    const cg = chunkGeom(32);
    try std.testing.expectEqual(@as(u32, 256), cg.block);
    try std.testing.expectEqual(chunk_smem, cg.shared);
    try std.testing.expect(cg.shared > 48 * 1024);
    try std.testing.expectEqual(Path.wide, path(1, null, null, false));
    try std.testing.expectEqual(Path.serial, path(3, false, null, false));
    try std.testing.expectEqual(Path.chunk, path(64, null, 40, true));
    try std.testing.expectEqual(@as(usize, 8), chunkOff(40));
    try std.testing.expectEqual(@as(usize, 2), p1_triton.len);
}

test "delta-rule prefix equals replay, and a segment's rows equal a solo run" {
    // 1e8 and -1e8 cancel in the butterfly before the 1 is added, so the sum is 1 rather than 0.
    var lanes: [32]f32 = @splat(0);
    lanes[0] = 1e8;
    lanes[1] = 1;
    lanes[16] = -1e8;
    warpSum(&lanes);
    for (lanes) |v| try std.testing.expectEqual(@as(f32, 1), v);

    var state = std.mem.zeroes([dv][dk]f32);
    var after = std.mem.zeroes([dv][dk]f32);
    for (0..dv) |r| for (0..dk) |c| {
        state[r][c] = @as(f32, @floatFromInt((r * 3 + c) % 5)) * 0.25;
        after[r][c] = state[r][c];
    };
    var ks: [4][dk]f32 = undefined;
    var gs: [4][dk]f32 = undefined;
    var vs: [4][dv]f32 = undefined;
    const beta = [_]f32{ 0.5, 1, 0.25, 0.75 };
    for (0..4) |row| {
        for (0..dk) |c| {
            ks[row][c] = @as(f32, @floatFromInt((c + row) % 7)) * 0.1 - 0.3;
            gs[row][c] = 0.5 + @as(f32, @floatFromInt(c % 3)) * 0.1;
        }
        for (0..dv) |c| vs[row][c] = @as(f32, @floatFromInt((c + row * 2) % 4)) * 0.2;
    }
    for (0..4) |row| deltaStep(&state, &ks[row], &gs[row], &vs[row], beta[row]);
    for (0..2) |row| deltaStep(&after, &ks[row], &gs[row], &vs[row], beta[row]);
    // a kept prefix is the state of the first rows alone, then the rest continues it
    var prefix = after;
    for (2..4) |row| deltaStep(&prefix, &ks[row], &gs[row], &vs[row], beta[row]);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&state), std.mem.asBytes(&prefix));
    // segment 1's two rows, started from the state the first segment left, match the solo suffix
    var solo = after;
    for (2..4) |row| deltaStep(&solo, &ks[row], &gs[row], &vs[row], beta[row]);
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&state), std.mem.asBytes(&solo));
    // the other segment's rows from the original state are a different state: rows do not leak across segments
    var other = std.mem.zeroes([dv][dk]f32);
    for (0..dv) |r| for (0..dk) |c| {
        other[r][c] = @as(f32, @floatFromInt((r * 3 + c) % 5)) * 0.25;
    };
    for (2..4) |row| deltaStep(&other, &ks[row], &gs[row], &vs[row], beta[row]);
    try std.testing.expect(!std.mem.eql(u8, std.mem.asBytes(&other), std.mem.asBytes(&after)));
}
