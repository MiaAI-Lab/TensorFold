//! GPTQ int4 weights of the INT4-AutoRound checkpoint (azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound): the
//! routed experts and the lm_head, symmetric 4-bit codes with an fp16 scale per 128 inputs and output column
//! (AutoGPTQ's v1 layout: qweight int32 [K/8, N], eight codes a word low nibble first; qzeros all 7, the stored zero
//! point minus one, so every zero point is 8; no g_idx). zig/kernels/cuda/fn_int4.cu runs them (ours, no Python
//! counterpart): exact weights in bf16 MMAs, one fp32 MMA chain a group, groups added with fmaf in order, so a row's
//! bits never depend on the other rows, the row count, the plan's items or the column tiles a warp takes.
//!
//! This file: the packed layouts (made on the host from the checkpoint's bytes, `packWords` / `packScales`), the
//! kernels' launches (`gateUp`, `down`, `dense`) and the host reference the checks use.
const std = @import("std");
const cuda = @import("cuda");
const kern = @import("cuda_kernels.zig");

/// One layer's routed experts: up = [E][gate, up] and down = [E] packed matrices with their scales.
pub const Experts = struct {
    up: u64 = 0,
    up_s: u64 = 0,
    down: u64 = 0,
    down_s: u64 = 0,
    count: u32 = 0,
    /// the rank's intermediate width (gate/up outputs, down inputs)
    width: u32 = 0,
    dims: u32 = 0,
    /// down's input group: 128, or 64 where a rank's half of the width cuts a group of 128 (TP=2: 320 of 640)
    gs_down: u32 = 128,
};

/// One packed matrix [n outputs, k inputs] (the head: a rank's vocabulary columns).
pub const Mat = struct { w: u64 = 0, s: u64 = 0, n: u32 = 0, k: u32 = 0, gs: u32 = 128 };

pub const checkpoint_group = 128;

/// Words a packed [n, k] matrix holds (8 codes a word).
pub fn wordsOf(n: usize, k: usize) usize {
    return n * k / 8;
}

/// fp16 scales a packed [n, k] matrix holds at groups of `gs`.
pub fn scalesOf(n: usize, k: usize, gs: usize) usize {
    return n * (k / gs);
}

/// A GPTQ word's eight codes (input j at nibble j) reordered for the kernel: nibble i = input 2i, nibble i + 4 =
/// input 2i + 1 (the pair at shift 4p is inputs 2p, 2p + 1).
pub fn shuffle(w: u32) u32 {
    var out: u32 = 0;
    inline for (0..4) |i| {
        out |= ((w >> (8 * i)) & 0xF) << (4 * i);
        out |= ((w >> (8 * i + 4)) & 0xF) << (4 * (i + 4));
    }
    return out;
}

/// The source of one checkpoint matrix: qweight int32 [k_full / 8, n_full] and scales fp16 [k_full / 128, n_full]
/// as little-endian bytes.
pub const Source = struct { qweight: []const u8, scales: []const u8, n_full: usize, k_full: usize };

pub const Slice = struct { n0: usize, n: usize, k0: usize, k: usize, gs: usize };

/// A slice the kernels take: groups of 64 or 128 inside the source's groups of 128, whole n8 tiles, in bounds, and
/// the source's byte lengths those of [k_full / 8, n_full] words and [k_full / 128, n_full] scales.
fn checkSlice(src: Source, sl: Slice) !void {
    if (sl.gs != 64 and sl.gs != 128) return error.BadSlice;
    if (sl.n == 0 or sl.k == 0 or sl.n % 8 != 0 or sl.k % sl.gs != 0 or sl.k0 % sl.gs != 0) return error.BadSlice;
    if (sl.n0 + sl.n > src.n_full or sl.k0 + sl.k > src.k_full or src.k_full % checkpoint_group != 0) return error.BadSlice;
    if (src.qweight.len != src.k_full / 8 * src.n_full * 4 or src.scales.len != src.k_full / checkpoint_group * src.n_full * 2) return error.BadLength;
}

/// Columns [n0, n0 + n) and inputs [k0, k0 + k) of `src` in the kernel's word order: [n/8][k/gs][32][gs/32].
pub fn packWords(src: Source, sl: Slice, out: []u32) !void {
    try checkSlice(src, sl);
    if (out.len != wordsOf(sl.n, sl.k)) return error.BadLength;
    const kg = sl.k / sl.gs;
    const per = sl.gs / 32;
    var at: usize = 0;
    for (0..sl.n / 8) |nt| for (0..kg) |g| for (0..32) |lane| {
        const col = sl.n0 + nt * 8 + lane / 4;
        const t = lane % 4;
        for (0..per) |b| {
            const kp = (sl.k0 + g * sl.gs + b * 32) / 8 + t;
            out[at] = shuffle(std.mem.readInt(u32, src.qweight[(kp * src.n_full + col) * 4 ..][0..4], .little));
            at += 1;
        }
    };
}

/// The scales of the same slice: fp16 bits [n/8][k/gs][8] (a group of 64 repeats its group of 128's scale).
pub fn packScales(src: Source, sl: Slice, out: []u16) !void {
    try checkSlice(src, sl);
    const kg = sl.k / sl.gs;
    if (out.len != scalesOf(sl.n, sl.k, sl.gs)) return error.BadLength;
    var at: usize = 0;
    for (0..sl.n / 8) |nt| for (0..kg) |g| for (0..8) |c| {
        const col = sl.n0 + nt * 8 + c;
        const row = (sl.k0 + g * sl.gs) / checkpoint_group;
        out[at] = std.mem.readInt(u16, src.scales[(row * src.n_full + col) * 2 ..][0..2], .little);
        at += 1;
    };
}

/// AutoGPTQ v1 symmetric zero points: every nibble of qzeros [k_full / 128, n_full / 8] is 7 (8 minus one).
pub fn zerosAreSymmetric(qzeros: []const u8) bool {
    var i: usize = 0;
    while (i + 8 <= qzeros.len) : (i += 8) if (std.mem.readInt(u64, qzeros[i..][0..8], .little) != 0x7777777777777777) return false;
    while (i < qzeros.len) : (i += 1) if (qzeros[i] != 0x77) return false;
    return true;
}

/// Code q of input k, column n of `src` (0..15; the weight is (q - 8) * scale).
pub fn code(src: Source, n: usize, k: usize) u4 {
    const w = std.mem.readInt(u32, src.qweight[((k / 8) * src.n_full + n) * 4 ..][0..4], .little);
    return @intCast((w >> @intCast(4 * (k % 8))) & 0xF);
}

pub fn scale(src: Source, n: usize, k: usize) f32 {
    const h: f16 = @bitCast(std.mem.readInt(u16, src.scales[((k / checkpoint_group) * src.n_full + n) * 2 ..][0..2], .little));
    return @floatCast(h);
}

/// The dequantized weight of input k, column n: (q - 8) * scale, exact in fp32.
pub fn weight(src: Source, n: usize, k: usize) f32 {
    return @as(f32, @floatFromInt(@as(i32, code(src, n, k)) - 8)) * scale(src, n, k);
}

// ---- the kernels -------------------------------------------------------------------------------------------------

const warps = 4;
/// [kind: gate/up SwiGLU (gs 128), down fp32 gs 128, down bf16 gs 128, fp32 gs 64, bf16 gs 64][NT: 1, 2, 4]
const n_kinds = 5;
const nts = [_]usize{ 1, 2, 4 };
/// row tiles of 16 a pass: 1 (decode), 2 (prompt items of up to 64 pairs: the weights read once for 32 rows)
const mts = [_]usize{ 1, 2 };

fn symbol(comptime gs: usize, comptime nt: usize, comptime mt: usize, comptime mats: usize, comptime epi: usize) [:0]const u8 {
    return std.fmt.comptimePrint("_ZN10tf_fn_int411int4_kernelILi{d}ELi{d}ELi{d}ELi{d}ELi{d}ELi4EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_iPvifi", .{ gs, nt, mt, mats, epi });
}

/// The plan tile of prompt calls (items of up to 64 pairs, two 16-row tiles a pass) and the pairs from which it is
/// taken; smaller calls take kern.plan_tile (16) and one tile a pass.
pub const prompt_tile: usize = 64;
pub const prompt_pairs: usize = 2048;

/// Measured on GB10 (int4-check, 4096 rows, top 5 of 512): 64-pair items with two row tiles a pass are slower
/// (gate/up 8.4 vs 6.0 ms, down 3.8 vs 3.2 ms: a third of the warps), so every call takes 16-pair items for now;
/// TF_FLASHNEXT_INT4_TILE=64 takes the prompt tile (the same bits either way, int4-check).
pub fn tileFor(pairs: usize) usize {
    const want = if (std.c.getenv("TF_FLASHNEXT_INT4_TILE")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch kern.plan_tile else kern.plan_tile;
    return if (want == prompt_tile and pairs >= prompt_pairs) prompt_tile else kern.plan_tile;
}

const kinds = [n_kinds]struct { gs: usize, mats: usize, epi: usize }{
    .{ .gs = 128, .mats = 2, .epi = 2 }, .{ .gs = 128, .mats = 1, .epi = 0 }, .{ .gs = 128, .mats = 1, .epi = 3 },
    .{ .gs = 64, .mats = 1, .epi = 0 },  .{ .gs = 64, .mats = 1, .epi = 3 },
};

pub const symbols = blk: {
    var out: [n_kinds][nts.len][mts.len][:0]const u8 = undefined;
    for (kinds, 0..) |k, i| for (nts, 0..) |nt, j| for (mts, 0..) |mt, q| {
        out[i][j][q] = symbol(k.gs, nt, mt, k.mats, k.epi);
    };
    break :blk out;
};

pub const Kernels = struct {
    module: cuda.Module,
    fns: [n_kinds][nts.len][mts.len]cuda.Function,
    /// resident blocks of each function on this GPU (occupancy x SMs)
    blocks: [n_kinds][nts.len][mts.len]usize,

    pub fn load(ctx: *const cuda.Context) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.module = try cuda.Module.load(ctx.d, cuda.kernels.fn_int4);
        errdefer k.module.unload();
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        for (symbols, 0..) |row, i| for (row, 0..) |cols, j| for (cols, 0..) |s, q| {
            k.fns[i][j][q] = try k.module.function(s);
            k.blocks[i][j][q] = @as(usize, @max(1, try k.fns[i][j][q].occupancy(warps * 32, 0))) * sms;
        };
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.module.unload();
    }
};

fn int(x: usize) c_int {
    return @intCast(x);
}

/// The widest n8-tile count whose units still fill the GPU twice over (the same bits at any NT).
fn pickNt(k: *const Kernels, kind: usize, max_items: usize, n: usize, q: usize) usize {
    var j: usize = nts.len;
    while (j > 1) {
        j -= 1;
        const units = max_items * (n / (8 * nts[j]));
        if (n % (8 * nts[j]) == 0 and units >= 2 * k.blocks[kind][j][q] * warps) return j;
    }
    return 0;
}

fn launch(k: *const Kernels, s: cuda.Stream, kind: usize, max_items: usize, x: u64, x_stride: usize, slots: usize, w: u64, sc: u64, kk: usize, n: usize, p: ?kern.Plan, rows: usize, out: u64, out_stride: usize, skip: c_int, force_nt: ?usize, mt: usize) !void {
    // uint4 loads of the input rows, uint4/uint2 weight words, half2 scales, float2/bf16x2 stores
    if (n % 8 != 0 or kk % kinds[kind].gs != 0 or x_stride % 8 != 0 or x % 16 != 0 or w % 16 != 0 or sc % 4 != 0 or out % 8 != 0) return error.Invalid;
    const q = std.mem.indexOfScalar(usize, &mts, mt) orelse return error.Invalid;
    const j = if (force_nt) |f| (std.mem.indexOfScalar(usize, &nts, f) orelse return error.Invalid) else pickNt(k, kind, max_items, n, q);
    if (n % (8 * nts[j]) != 0) return error.Invalid;
    const units = max_items * (n / (8 * nts[j]));
    const grid = @min((units + warps - 1) / warps, k.blocks[kind][j][q]);
    if (grid < 1) return;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(int(x_stride));
    a.add(int(slots));
    a.add(w);
    a.add(sc);
    a.add(int(kk));
    a.add(int(n));
    if (p) |pl| {
        a.add(pl.items);
        a.add(pl.counts);
        a.add(pl.members);
    } else {
        a.add(@as(u64, 0));
        a.add(@as(u64, 0));
        a.add(@as(u64, 0));
    }
    a.add(int(rows));
    a.add(out);
    a.add(int(out_stride));
    a.add(@as(f32, 0));
    a.add(skip);
    try cuda.launch.launch(k.fns[kind][j][q], .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = warps * 32 } }, s, &a);
}

/// The routed experts' gate/up with SwiGLU: x [R, D] token rows -> out [R * slots, width] bf16, a row per routed pair
/// of the plan (items of expert `skip` untouched). `tile`: the plan's (tileFor: 64 makes two row tiles a pass).
pub fn gateUp(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, ex: Experts, p: kern.Plan, plan_slots: usize, plan_experts: usize, out: u64, rows: usize, skip: c_int, tile: usize) !void {
    const items = kern.maxItems(rows * plan_slots, plan_experts, tile);
    try launch(k, s, 0, items, x, x_stride, plan_slots, ex.up, ex.up_s, ex.dims, ex.width, p, 0, out, ex.width, skip, null, if (tile > 16) 2 else 1);
}

/// The routed experts' down: act [R * slots, width] pair rows -> out [R * slots, D], fp32 (`f32_out`) or bf16.
pub fn down(k: *const Kernels, s: cuda.Stream, act: u64, act_stride: usize, ex: Experts, p: kern.Plan, plan_slots: usize, plan_experts: usize, out: u64, f32_out: bool, rows: usize, skip: c_int, tile: usize) !void {
    const items = kern.maxItems(rows * plan_slots, plan_experts, tile);
    const kind: usize = switch (ex.gs_down) {
        128 => if (f32_out) 1 else 2,
        64 => if (f32_out) 3 else 4,
        else => return error.Invalid,
    };
    try launch(k, s, kind, items, act, act_stride, 0, ex.down, ex.down_s, ex.width, ex.dims, p, 0, out, ex.dims, skip, null, if (tile > 16) 2 else 1);
}

/// A dense matmul (the head): x [rows, k] -> out [rows, n] bf16 (or fp32), rows `x_stride` apart.
pub fn dense(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, m: Mat, out: u64, f32_out: bool, rows: usize) !void {
    try denseNt(k, s, x, x_stride, m, out, f32_out, rows, null);
}

/// `dense` with the n8 tiles a warp forced (checks: the bits never depend on it).
pub fn denseNt(k: *const Kernels, s: cuda.Stream, x: u64, x_stride: usize, m: Mat, out: u64, f32_out: bool, rows: usize, nt: ?usize) !void {
    if (rows == 0) return;
    const kind: usize = switch (m.gs) {
        128 => if (f32_out) 1 else 2,
        64 => if (f32_out) 3 else 4,
        else => return error.Invalid,
    };
    try launch(k, s, kind, (rows + 15) / 16, x, if (rows == 1) m.k else x_stride, 0, m.w, m.s, m.k, m.n, null, rows, out, m.n, -1, nt, 1);
}

/// `gateUp` / `down` with the n8 tiles a warp forced (checks).
pub fn expertsNt(k: *const Kernels, s: cuda.Stream, kind: usize, x: u64, x_stride: usize, slots: usize, w: u64, sc: u64, kk: usize, n: usize, p: kern.Plan, max_items: usize, out: u64, skip: c_int, nt: usize, mt: usize) !void {
    try launch(k, s, kind, max_items, x, x_stride, slots, w, sc, kk, n, p, 0, out, n, skip, nt, mt);
}

// ---- host tests ----------------------------------------------------------------------------------------------------

test "shuffle puts input 2i at nibble i and 2i + 1 at nibble i + 4" {
    // inputs j = 0..7 hold the codes j + 1
    var w: u32 = 0;
    for (0..8) |j| w |= @as(u32, @intCast(j + 1)) << @intCast(4 * j);
    const s = shuffle(w);
    for (0..4) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(2 * i + 1)), (s >> @intCast(4 * i)) & 0xF);
        try std.testing.expectEqual(@as(u32, @intCast(2 * i + 2)), (s >> @intCast(4 * (i + 4))) & 0xF);
    }
}

test "packWords: a lane's word of a block holds its column's inputs 8 t .. 8 t + 7" {
    const gpa = std.testing.allocator;
    const n_full = 16;
    const k_full = 256;
    const qw = try gpa.alloc(u8, k_full / 8 * n_full * 4);
    defer gpa.free(qw);
    var prng = std.Random.DefaultPrng.init(7);
    prng.random().bytes(qw);
    const sc = try gpa.alloc(u8, k_full / 128 * n_full * 2);
    defer gpa.free(sc);
    for (0..sc.len / 2) |i| std.mem.writeInt(u16, sc[2 * i ..][0..2], @intCast(i), .little);
    const src: Source = .{ .qweight = qw, .scales = sc, .n_full = n_full, .k_full = k_full };
    for ([_]usize{ 128, 64 }) |gs| {
        const sl: Slice = .{ .n0 = 8, .n = 8, .k0 = 128, .k = 128, .gs = gs };
        const out = try gpa.alloc(u32, wordsOf(sl.n, sl.k));
        defer gpa.free(out);
        try packWords(src, sl, out);
        const kg = sl.k / gs;
        for (0..kg) |g| for (0..32) |lane| for (0..gs / 32) |b| {
            const word = out[(g * 32 + lane) * (gs / 32) + b];
            for (0..8) |j| {
                const nib: usize = if (j % 2 == 0) j / 2 else j / 2 + 4;
                const got: u4 = @intCast((word >> @intCast(4 * nib)) & 0xF);
                const kin = sl.k0 + g * gs + b * 32 + 8 * (lane % 4) + j;
                try std.testing.expectEqual(code(src, sl.n0 + lane / 4, kin), got);
            }
        };
        const scl = try gpa.alloc(u16, scalesOf(sl.n, sl.k, gs));
        defer gpa.free(scl);
        try packScales(src, sl, scl);
        for (0..kg) |g| for (0..8) |c| {
            try std.testing.expectEqual(@as(u16, @intCast(((sl.k0 + g * gs) / 128) * n_full + sl.n0 + c)), scl[g * 8 + c]);
        };
    }
}

test "symmetric zero points and the kernel symbols" {
    var z: [20]u8 = @splat(0x77);
    try std.testing.expect(zerosAreSymmetric(&z));
    z[19] = 0x87;
    try std.testing.expect(!zerosAreSymmetric(&z));
    z[19] = 0x77;
    z[3] = 0x78;
    try std.testing.expect(!zerosAreSymmetric(&z));
    try std.testing.expectEqualStrings("_ZN10tf_fn_int411int4_kernelILi128ELi4ELi1ELi2ELi2ELi4EEEvPK13__nv_bfloat16iiPKjPK6__halfiiPKiSA_SA_iPvifi", symbols[0][2][0]);
}
