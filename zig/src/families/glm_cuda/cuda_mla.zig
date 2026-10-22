//! GLM-5.3-Flash (family glm5_next) on the Zig CUDA runtime, package P5: the MLA attention with the DSA sparse indexer.
//!
//! Ports (Python reference: TensorFold v0.6.0 + the GLM-5.3 recipe's patches, families/glm5_next/cuda/):
//!   - msa.cu / msa.cpp / msa.py (TF_GLM_MSA=1, Mia's AI Lab, Apache-2.0): the one CUDA kernel of the attention path, as
//!     zig/kernels/cuda/glm_msa.cu; `Msa` below is msa.cpp's launch (`msa_prompt_cuda`) and msa.py's `supported`.
//!   - kv8.py: the FP8 row format (`scaleOf`, `quantizeRow`, `dequantRow`: kv8.scales / quantize_rows / dequantize),
//!     the 528-byte latent row and the 144-byte pooled-key row.
//!   - forward.py `index_ring`, segments.py `SegRows` (`SegTables`), sparse.py `split_chunk(s)` / `split_scratch`
//!     and the geometry constants: the host bookkeeping the Triton kernels' launches are made of.
//! Everything else in the path (latent write, indexer ring + pooled keys, scores, split selection, absorb, dense and
//! sparse attention, merge, expand) is Triton in the reference: not ported here, listed in `triton_kernels` for the
//! AOT-replay package (P1). `msaReference` is a plain fp32 host attention for checking the CUDA kernel; the kernel
//! itself is the original's device code, character for character.

const std = @import("std");
const cuda = @import("cuda");

// -- geometry (GLM-5.3-Flash, SPEC.md section 1/4) --------------------------------------------------------------------
pub const lw = 512; // latent values a token (kv_lora_rank)
pub const heads_max = 32; // query heads a rank at most (TP=1 would be 64; msa covers TP>=2)
pub const kv_pad = 16; // kv8.PAD: fp32 scale then zeros
pub const latent_row = lw + kv_pad; // 528 bytes: msa.cu's CROW8
pub const index_dim = 128; // indexer head dim, and a pooled key's width
pub const pooled_row = index_dim + kv_pad; // 144 bytes
pub const index_heads = 32;
pub const pool = 4; // sparse.POOL: tokens a pooled key
pub const topk_pools = 512; // sparse.TOPK_POOLS
pub const sparse_from = (topk_pools + 1) * pool - 1; // 2051: first position that attends sparsely
pub const tokens = topk_pools * pool + pool - 1; // 2051: a sparse row's attended tokens (sparse.TOKENS)
pub const extent = 2048; // segments.EXTENT: stream extents start at multiples of this many tokens
pub const max_segs = 8;
pub const segs_default = 4;
pub const scale: f32 = 0.0625; // 256^-0.5
pub const shift = 8; // kv8.SHIFT

// -- the FP8 row format (kv8.py) -------------------------------------------------------------------------------------

/// bf16 rounding (round to nearest even) of an f32, back as an f32.
pub fn bf16(x: f32) f32 {
    const b: u32 = @bitCast(x);
    if (std.math.isNan(x)) return x;
    const r = b +% 0x7fff +% ((b >> 16) & 1);
    return @bitCast(r & 0xffff0000);
}

/// kv8.scales: fp32 amax -> the fp32 power of two 2^(ceil(log2 amax) - 8), its biased exponent held in 1..254.
pub fn scaleOf(amax: f32) f32 {
    const bits: u32 = @bitCast(amax);
    const e: i32 = @as(i32, @intCast((bits >> 23) & 0xff)) + @intFromBool((bits & 0x7fffff) != 0);
    const s = std.math.clamp(e - shift, 1, 254);
    return @bitCast(@as(u32, @intCast(s)) << 23);
}

/// The reciprocal of `scaleOf`'s result (exact: a power of two): (254 - e) << 23.
pub fn invOf(s: f32) f32 {
    const e: i32 = @intCast(@as(u32, @bitCast(s)) >> 23);
    return @bitCast(@as(u32, @intCast(254 - e)) << 23);
}

/// An e4m3fn code as f32 (exact).
pub fn e4m3Decode(c: u8) f32 {
    const sign: f32 = if (c & 0x80 != 0) -1 else 1;
    const e: u32 = (c >> 3) & 0xf;
    const m: u32 = c & 7;
    if (e == 0xf and m == 7) return std.math.nan(f32);
    if (e == 0) return sign * @as(f32, @floatFromInt(m)) * 0x1p-9;
    return sign * (1 + @as(f32, @floatFromInt(m)) / 8) * std.math.pow(f32, 2, @as(f32, @floatFromInt(@as(i32, @intCast(e)) - 7)));
}

/// f32 -> e4m3fn code, round to nearest even, saturating at 448 (torch's `.to(float8_e4m3fn)` for finite values).
pub fn e4m3Encode(x: f32) u8 {
    const sign: u8 = if (std.math.signbit(x)) 0x80 else 0;
    const a = @abs(x);
    if (std.math.isNan(x)) return 0x7f;
    if (a >= 448) return sign | 0x7e;
    var best: u8 = 0;
    var best_d: f32 = std.math.inf(f32);
    // the code set is monotone in its low 7 bits: search the nearest, ties to the even mantissa
    var c: u8 = 0;
    while (c <= 0x7e) : (c += 1) {
        const d = @abs(e4m3Decode(c) - a);
        if (d < best_d or (d == best_d and c & 1 == 0 and best & 1 == 1)) {
            best = c;
            best_d = d;
        }
    }
    return sign | best;
}

/// kv8.quantize_rows for one row: `x` (bf16-valued f32, `lw` of them) -> 512 codes, the fp32 scale, 12 zero bytes.
pub fn quantizeRow(x: []const f32, out: []u8) void {
    std.debug.assert(out.len == x.len + kv_pad);
    var amax: f32 = 0;
    for (x) |v| amax = @max(amax, @abs(bf16(v)));
    const s = scaleOf(amax);
    const inv = invOf(s);
    @memset(out, 0);
    for (x, 0..) |v, i| out[i] = e4m3Encode(bf16(v) * inv);
    std.mem.writeInt(u32, out[x.len..][0..4], @bitCast(s), .little);
}

/// kv8.dequantize for one row (exact): code x scale.
pub fn dequantRow(row: []const u8, out: []f32) void {
    const n = out.len;
    const s: f32 = @bitCast(std.mem.readInt(u32, row[n..][0..4], .little));
    for (out, row[0..n]) |*o, c| o.* = e4m3Decode(c) * s;
}

// -- ring / segment bookkeeping --------------------------------------------------------------------------------------

/// forward.index_ring: rows of the indexer's key and gate ring for windows of up to `rows` rows (those and the 3 before
/// them, up to 64); the capacity when that is less (no row wraps).
pub fn indexRing(capacity: usize, rows: usize) usize {
    return @min(capacity, (rows + 3 + 63) / 64 * 64);
}

/// Where token `t` of the stream in slot `slot` sits in the index key and gate rings.
pub fn ringRow(slot: usize, ring: usize, t: usize) usize {
    return slot * ring + t % ring;
}

/// segments.seg_count: segments a batched window's tables hold for `streams` streams.
pub fn segCount(streams: usize) !usize {
    if (streams < 1 or streams > max_segs) return error.BadStreams;
    return @max(segs_default, streams);
}

pub const Segment = struct { base: i32, pos: i32, rows: i32, ring_base: i32 = -1 };

/// segments.SegRows.fill: the int32 table of a window (per row pos, base, seg, sparse, ibase; per segment start, rows,
/// pbase, pools), in the device layout. `t` holds 5 * rows + 4 * segs entries (zeroed here); `ring` null = no rings.
/// Returns the window's rows.
pub fn fillSegTables(t: []i32, rows: usize, segs: usize, segments: []const Segment, ring: ?usize, sparse_from_: ?i32) !usize {
    if (t.len != 5 * rows + 4 * segs) return error.BadTable;
    if (segments.len < 1 or segments.len > segs) return error.BadSegments;
    if (ring) |r| if (r < 4) return error.BadRing;
    @memset(t, 0);
    var r: usize = 0;
    for (segments, 0..) |sg, s| {
        if (sg.rows < 1 or sg.pos < 0 or sg.base < 0 or @mod(sg.base, extent) != 0) return error.BadSegment;
        var ibase = sg.base;
        if (ring) |rg| {
            if (sg.ring_base < 0) return error.BadSegment;
            ibase = sg.ring_base;
            if (sg.rows + 3 > rg) return error.BadSegment;
        }
        const n: usize = @intCast(sg.rows);
        if (r + n > rows) return error.BadSegments;
        for (0..n) |i| {
            const q = sg.pos + @as(i32, @intCast(i));
            t[r + i] = q;
            t[rows + r + i] = sg.base;
            t[2 * rows + r + i] = @intCast(s);
            t[3 * rows + r + i] = @intFromBool(sparse_from_ != null and q >= sparse_from_.?);
            t[4 * rows + r + i] = ibase;
        }
        const last = sg.pos + sg.rows - 1;
        const o = 5 * rows;
        t[o + s] = @intCast(r);
        t[o + segs + s] = sg.rows;
        t[o + 2 * segs + s] = @divTrunc(sg.base, pool);
        t[o + 3 * segs + s] = if (sparse_from_ != null and last >= sparse_from_.?) @divTrunc(last + 1, pool) else 0;
        r += n;
    }
    return r;
}

// -- the split selection's constexprs (sparse.py split_chunk / split_chunks / split_scratch) ---------------------------
pub const split_warps = 4;
pub const split_blk = 16;

/// Scores a chunk program takes: 2048 up to 16384 pooled keys a row, else 4096 (the kernel's CHS).
pub fn splitChunk(np: usize) usize {
    return if (np <= 16384) 2048 else 4096;
}

/// Chunk programs a row of `np` scores takes (the kernel's CP constexpr: a power of two).
pub fn splitChunks(np: usize) usize {
    const c = (np + splitChunk(np) - 1) / splitChunk(np);
    return std.math.ceilPowerOfTwo(usize, @max(c, 1)) catch unreachable;
}

/// int32 entries of select_split's scratch for `rows` rows of `np` scores.
pub fn splitScratch(rows: usize, np: usize) usize {
    return rows * (1024 + splitChunks(np) * 257);
}

pub const SplitVariant = struct { chs: u32, cp: u32 };

/// Every (CHS, CP) pair `_select_split` is launched with for 1 <= NP <= `max_pools` (the AOT package builds one cubin
/// per pair and STEP 0..4): CHS 2048 with CP 1, 2, 4, 8, then CHS 4096 with CP 8, 16, ... Returns the count written.
pub fn splitVariants(max_pools: usize, out: []SplitVariant) usize {
    var n: usize = 0;
    var np: usize = 1;
    while (np <= max_pools) {
        const v: SplitVariant = .{ .chs = @intCast(splitChunk(np)), .cp = @intCast(splitChunks(np)) };
        if (n == 0 or out[n - 1].chs != v.chs or out[n - 1].cp != v.cp) {
            out[n] = v;
            n += 1;
        }
        np = @as(usize, v.cp) * v.chs + 1; // the last pool count of this variant is CP * CHS
    }
    return n;
}

// -- msa.cu's launch (msa.cpp, msa.py) ---------------------------------------------------------------------------------

const keys_per_tile = 16; // msa.cu KT
const kv_row_halves = 128 + 8; // KROW
const spl = keys_per_tile / 8 * 4; // SPL

/// msa.cu `Shape<MT>`: MT m16 tiles of heads; warps, threads and dynamic shared bytes.
pub fn msaThreads(mt: usize) usize {
    return 4 * mt * 32;
}
pub fn msaShared(mt: usize) usize {
    const nw = 4 * mt;
    return 4 * mt * keys_per_tile * kv_row_halves * 2 + 2 * nw * 32 * spl * 4;
}

pub const msa_symbols = [2][:0]const u8{
    "_ZN10tf_glm_msa10msa_kernelILi1EEEvPK13__nv_bfloat16PKhPKiS7_PS1_iif",
    "_ZN10tf_glm_msa10msa_kernelILi2EEEvPK13__nv_bfloat16PKhPKiS7_PS1_iif",
};

/// msa.py `supported`: 1..32 heads of 512-wide absorbed queries over a 528-byte FP8 latent cache.
pub fn msaSupported(h: usize, q_width: usize, cache_row_bytes: usize) bool {
    return h >= 1 and h <= heads_max and q_width == lw and cache_row_bytes == latent_row;
}

pub const Msa = struct {
    mod: cuda.Module,
    f: [2]cuda.Function,

    pub fn load(d: *const cuda.Driver) !Msa {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var m = try cuda.Module.load(d, cuda.kernels.glm_msa);
        errdefer m.unload();
        var k: Msa = .{ .mod = m, .f = undefined };
        for (&k.f, msa_symbols, 1..) |*f, s, mt| {
            f.* = try m.function(s);
            // msa.cu launch<MT>: past 48 KiB of shared memory (two m tiles) the block must ask for it
            try f.allowDynamicShared(@intCast(msaShared(mt)));
        }
        return k;
    }

    pub fn deinit(k: *Msa) void {
        k.mod.unload();
    }

    /// msa_prompt_cuda: rows 0..`r` of absorbed queries `qa` (bf16 [r, h, 512]) over their selected tokens
    /// (`toks` int32 [>= r, w], ascending, the first `counts[row]` valid) of the FP8 latent `cache` (uint8 [rows, 528])
    /// into `out` (bf16 [r, h, 512]); rows with counts <= 0 are left alone. All arguments are device addresses.
    pub fn prompt(k: *const Msa, s: cuda.Stream, qa: u64, cache: u64, toks: u64, counts: u64, out: u64, r: usize, h: usize, w: usize, sc: f32) !void {
        if (h < 1 or h > heads_max) return error.UnsupportedHeads;
        if (r == 0) return;
        const mt: usize = if (h <= 16) 1 else 2;
        var a: cuda.Args = .{};
        for ([_]u64{ qa, cache, toks, counts, out }) |v| a.add(v);
        a.add(@as(c_int, @intCast(h)));
        a.add(@as(c_int, @intCast(w)));
        a.add(sc);
        try cuda.launch.launch(k.f[mt - 1], .{ .grid = .{ .x = @intCast(r), .y = 1, .z = 1 }, .block = .{ .x = @intCast(msaThreads(mt)), .y = 1, .z = 1 }, .shared = @intCast(msaShared(mt)) }, s, &a);
    }
};

/// Plain fp32 host attention of one row for checking msa: q [h][512] (bf16-valued), the selected tokens' dequantized
/// latent rows as both keys and values (MLA absorbs the value projection: the cache row is K and V), softmax scale
/// `sc`, out [h][512] (rounded to bf16). Not the kernel's bits (fp32 mma sums in another order, ex2.approx): the
/// GPU check compares within a tolerance.
pub fn msaReference(q: []const f32, h: usize, cache: []const u8, toks: []const i32, n: usize, sc: f32, out: []f32, alloc: std.mem.Allocator) !void {
    const keys = try alloc.alloc(f32, n * lw);
    defer alloc.free(keys);
    for (0..n) |i| dequantRow(cache[@as(usize, @intCast(toks[i])) * latent_row ..][0..latent_row], keys[i * lw ..][0..lw]);
    const p = try alloc.alloc(f32, n);
    defer alloc.free(p);
    for (0..h) |hh| {
        var mx: f32 = -std.math.inf(f32);
        for (0..n) |i| {
            var d: f32 = 0;
            for (0..lw) |j| d += q[hh * lw + j] * keys[i * lw + j];
            p[i] = d * sc;
            mx = @max(mx, p[i]);
        }
        var l: f32 = 0;
        for (p) |*v| {
            v.* = @exp(v.* - mx);
            l += v.*;
        }
        for (0..lw) |j| {
            var o: f32 = 0;
            for (0..n) |i| o += p[i] * keys[i * lw + j];
            out[hh * lw + j] = bf16(o / l);
        }
    }
}

// -- Triton kernels of the path, NOT ported: for the AOT-replay package (P1) ---------------------------------------------
pub const TritonKernel = struct { name: []const u8, at: []const u8, consts: []const u8 };

pub const triton_kernels = [_]TritonKernel{
    .{ .name = "_seg_lat_write", .at = "latent.py:763", .consts = "LW=512 RS=528 FP8=1 warps 4 (grid R); also _lat_write :392 (single stream)" },
    .{ .name = "_index_write", .at = "sparse.py:23", .consts = "D=128 warps 1 (grid R)" },
    .{ .name = "_seg_index_write", .at = "sparse.py:550", .consts = "D=128 warps 1 (grid R)" },
    .{ .name = "_pool_keys", .at = "sparse.py:40", .consts = "D=128 RS=144 FP8=1 warps 1 (grid R//4+2)" },
    .{ .name = "_seg_pool_keys", .at = "sparse.py:566", .consts = "D=128 RS=144 FP8=1 warps 1 (grid R)" },
    .{ .name = "_scores", .at = "sparse.py:95", .consts = "H=32 HP=32 D=128 BP=64 L=32 warps 4 (grid n, cdiv(cdiv(NP,64),32)) prompt blocks" },
    .{ .name = "_scores_rows", .at = "sparse.py:136", .consts = "H=32 HP=32 D=128 BP=64 (single-stream decode; unused at default)" },
    .{ .name = "_seg_scores", .at = "sparse.py:619", .consts = "H=32 HP=32 D=128 BP=64 G=min(256,cdiv(NP,64)) RS=144 FP8=1 warps 4 (grid 8,G)" },
    .{ .name = "_select_rows", .at = "sparse.py:206", .consts = "K=512 BLOCK=1024 (or 4096) warps 4/8 (prompt blocks)" },
    .{ .name = "_tokens", .at = "sparse.py:292", .consts = "W=2051 K=512 PL=4 BLOCK warps 4 (prompt blocks)" },
    .{ .name = "_seg_select", .at = "sparse.py:654", .consts = "W=2051 K=512 PL=4 SEG (non-split path; SPLIT=1 uses _select_split)" },
    .{ .name = "_select_split", .at = "sparse.py:775", .consts = "W=2051 K=512 PL=4 SEG=1 BLK=16 STEP=0..4 warps 4; CHS/CP per splitVariants() (CP = next_pow2(cdiv(NP,CHS)), NP = pool tokens / 4)" },
    .{ .name = "_absorb_q", .at = "latent.py:124", .consts = "H=32 D=256 LW=512 BN=32 warps 4 (grid 32,16)" },
    .{ .name = "_absorb_mma", .at = "latent.py:237", .consts = "H=32 D=256 LW=512 BM=64 BN=128 BK=64 warps 4 stages 3 (grid 32,cdiv(R,64),4) prompt" },
    .{ .name = "_dense_chunks", .at = "latent.py:458", .consts = "H=32 LW=512 CH=512 KTT=32 HBT=32|16 SCALE=0.0625 RS=528 FP8=1 warps 8 stages 3" },
    .{ .name = "_seg_chunks", .at = "latent.py:780", .consts = "W=2051 H=32 LW=512 CH=512 KTT=32 HB=32 (R>=8) | 16 SCALE=0.0625 RS=528 FP8=1 warps 8 stages 3 (grid R,cdiv(32,HB),5)" },
    .{ .name = "_sparse_onepass", .at = "latent.py:541", .consts = "W=2051 H=32 LW=512 HBT=32 KTT=32 SPLIT=1 STAGES=1 PREFETCH=1 warps 8 (prompt rows; replaced by glm_msa.cu when TF_GLM_MSA=1)" },
    .{ .name = "_sparse_chunks", .at = "latent.py:489", .consts = "single-stream decode only; unused at default" },
    .{ .name = "_merge", .at = "latent.py:605", .consts = "H=32 LW=512 NCH=5 (seg) | chunks_for(ctx) (dense) SPARSE=0|1 warps 4" },
    .{ .name = "_expand_v", .at = "latent.py:139", .consts = "H=32 DV=256 LW=512 BN=16 warps 4 (grid 32,16)" },
    .{ .name = "_expand_mma", .at = "latent.py:256", .consts = "H=32 DV=256 LW=512 BM=64 BN=128 BK=64 warps 4 stages 3 (grid 32,cdiv(R,64),2) prompt" },
};

// -- tests (host only) -------------------------------------------------------------------------------------------------

test "geometry matches SPEC" {
    try std.testing.expectEqual(@as(usize, 528), latent_row);
    try std.testing.expectEqual(@as(usize, 144), pooled_row);
    try std.testing.expectEqual(@as(usize, 2051), tokens);
    try std.testing.expectEqual(@as(i32, 2051), sparse_from);
    try std.testing.expectEqual(@as(usize, 32), heads_max);
}

test "msa shared memory sizes: 1 tile under 48 KiB, 2 tiles past it" {
    try std.testing.expectEqual(@as(usize, 4 * 16 * 136 * 2 + 2 * 4 * 32 * 8 * 4), msaShared(1));
    try std.testing.expectEqual(@as(usize, 4 * 2 * 16 * 136 * 2 + 2 * 8 * 32 * 8 * 4), msaShared(2));
    try std.testing.expect(msaShared(1) <= 48 * 1024);
    try std.testing.expect(msaShared(2) > 48 * 1024);
    try std.testing.expectEqual(@as(usize, 128), msaThreads(1));
    try std.testing.expectEqual(@as(usize, 256), msaThreads(2));
}

test "e4m3 round trip over every code" {
    var c: u16 = 0;
    while (c < 256) : (c += 1) {
        const code: u8 = @intCast(c);
        if (code & 0x7f == 0x7f) continue; // NaN
        const v = e4m3Decode(code);
        if (v == 0) continue; // -0 encodes as +0 sign-less in the search
        try std.testing.expectEqual(code, e4m3Encode(v));
    }
    try std.testing.expectEqual(@as(f32, 448), e4m3Decode(0x7e));
    try std.testing.expectEqual(@as(f32, 0x1p-9), e4m3Decode(1));
    try std.testing.expectEqual(@as(f32, 1), e4m3Decode(0x38));
    // ties go to the even mantissa: 1.0625 is between 1.0 (m=0) and 1.125 (m=1)
    try std.testing.expectEqual(@as(u8, 0x38), e4m3Encode(1.0625));
    try std.testing.expectEqual(@as(u8, 0x3a), e4m3Encode(1.1875)); // between 1.125 (m=1) and 1.25 (m=2)
}

test "kv8 scale: 2^(ceil(log2 amax) - 8), codes within 256" {
    try std.testing.expectEqual(@as(f32, 1), scaleOf(256)); // 2^8 / 2^8
    try std.testing.expectEqual(@as(f32, 0x1p-8), scaleOf(1)); // ceil(log2 1) = 0
    try std.testing.expectEqual(@as(f32, 2), scaleOf(300)); // ceil(log2 300) = 9
    try std.testing.expectEqual(@as(f32, 0x1p-126), scaleOf(0)); // biased exponent held at 1
}

test "kv8 row: the largest value lands in [128, 256], dequantization error is small" {
    var x: [lw]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    for (&x) |*v| v.* = bf16((rnd.float(f32) - 0.5) * 6);
    var row: [latent_row]u8 = undefined;
    quantizeRow(&x, &row);
    const zeros12: [12]u8 = @splat(0);
    try std.testing.expectEqualSlices(u8, &zeros12, row[lw + 4 ..]);
    const s: f32 = @bitCast(std.mem.readInt(u32, row[lw..][0..4], .little));
    var amax: f32 = 0;
    for (x) |v| amax = @max(amax, @abs(v));
    const top = amax / s;
    try std.testing.expect(top > 128 and top <= 256);
    var back: [lw]f32 = undefined;
    dequantRow(&row, &back);
    var worst: f32 = 0;
    for (x, back) |a, b| worst = @max(worst, @abs(a - b));
    try std.testing.expect(worst <= amax * 0.07); // e4m3: 3 mantissa bits, 6.25% relative at worst
}

test "a zero row quantizes to zero codes and the smallest scale" {
    const x = @as([lw]f32, @splat(0));
    var row: [latent_row]u8 = undefined;
    quantizeRow(&x, &row);
    for (row[0..lw]) |c| try std.testing.expectEqual(@as(u8, 0), c);
    const s: f32 = @bitCast(std.mem.readInt(u32, row[lw..][0..4], .little));
    try std.testing.expectEqual(@as(u32, 1 << 23), @as(u32, @bitCast(s)));
}

test "index ring and segment tables" {
    try std.testing.expectEqual(@as(usize, 64), indexRing(1 << 20, 8));
    try std.testing.expectEqual(@as(usize, 128), indexRing(1 << 20, 62));
    try std.testing.expectEqual(@as(usize, 100), indexRing(100, 500));
    try std.testing.expectEqual(@as(usize, 3 * 64 + 5), ringRow(3, 64, 64 * 9 + 5));
    try std.testing.expectEqual(@as(usize, 4), try segCount(1));
    try std.testing.expectEqual(@as(usize, 6), try segCount(6));
    try std.testing.expectError(error.BadStreams, segCount(9));

    const S = 4;
    const R = 6;
    var t: [5 * R + 4 * S]i32 = undefined;
    const segs = [_]Segment{ .{ .base = 0, .pos = 2049, .rows = 3, .ring_base = 0 }, .{ .base = 4096, .pos = 10, .rows = 2, .ring_base = 64 } };
    const n = try fillSegTables(&t, R, S, &segs, 64, sparse_from);
    try std.testing.expectEqual(@as(usize, 5), n);
    try std.testing.expectEqualSlices(i32, &.{ 2049, 2050, 2051, 10, 11, 0 }, t[0..R]); // pos
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 0, 4096, 4096, 0 }, t[R .. 2 * R]); // base
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 0, 1, 1 }, t[2 * R ..][0..5]); // seg
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 1, 0, 0 }, t[3 * R ..][0..5]); // sparse: pos >= 2051
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 0, 64, 64 }, t[4 * R ..][0..5]); // ibase
    const o = 5 * R;
    try std.testing.expectEqualSlices(i32, &.{ 0, 3, 0, 0 }, t[o .. o + S]); // seg_start
    try std.testing.expectEqualSlices(i32, &.{ 3, 2, 0, 0 }, t[o + S .. o + 2 * S]); // seg_rows
    try std.testing.expectEqualSlices(i32, &.{ 0, 1024, 0, 0 }, t[o + 2 * S .. o + 3 * S]); // seg_pbase
    try std.testing.expectEqualSlices(i32, &.{ 2052 / 4, 0, 0, 0 }, t[o + 3 * S .. o + 4 * S]); // seg_pools
    // a segment not a multiple of the extent, or past its ring, is refused
    const bad = [_]Segment{.{ .base = 100, .pos = 0, .rows = 1, .ring_base = 0 }};
    try std.testing.expectError(error.BadSegment, fillSegTables(&t, R, S, &bad, 64, sparse_from));
}

test "split selection constexprs" {
    try std.testing.expectEqual(@as(usize, 2048), splitChunk(100));
    try std.testing.expectEqual(@as(usize, 2048), splitChunk(16384));
    try std.testing.expectEqual(@as(usize, 4096), splitChunk(16385));
    try std.testing.expectEqual(@as(usize, 1), splitChunks(2048));
    try std.testing.expectEqual(@as(usize, 2), splitChunks(2049));
    try std.testing.expectEqual(@as(usize, 4), splitChunks(5000)); // cdiv = 3 -> 4
    try std.testing.expectEqual(@as(usize, 8), splitChunks(16384));
    try std.testing.expectEqual(@as(usize, 8), splitChunks(16385)); // cdiv(16385, 4096) = 5 -> 8
    try std.testing.expectEqual(@as(usize, 1024 + 8 * 257), splitScratch(1, 16384));
    // 262144 tokens of context = 65536 pools: CP 16
    try std.testing.expectEqual(@as(usize, 16), splitChunks(65536));
}

test "split variants are exactly the pairs a brute-force sweep sees" {
    const max = 1 << 18; // 1M tokens of context
    var got: [64]SplitVariant = undefined;
    const n = splitVariants(max, &got);
    var want: [64]SplitVariant = undefined;
    var m: usize = 0;
    var np: usize = 1;
    while (np <= max) : (np += 1) {
        const v: SplitVariant = .{ .chs = @intCast(splitChunk(np)), .cp = @intCast(splitChunks(np)) };
        var seen = false;
        for (want[0..m]) |w| seen = seen or (w.chs == v.chs and w.cp == v.cp);
        if (!seen) {
            want[m] = v;
            m += 1;
        }
    }
    try std.testing.expectEqual(m, n);
    for (got[0..n], want[0..m]) |a, b| {
        try std.testing.expectEqual(b.chs, a.chs);
        try std.testing.expectEqual(b.cp, a.cp);
    }
}

test "msa reference: one key returns its row" {
    const a = std.testing.allocator;
    var x: [lw]f32 = undefined;
    for (&x, 0..) |*v, i| v.* = bf16(@as(f32, @floatFromInt(i % 7)) - 3);
    var cache: [2 * latent_row]u8 = undefined;
    quantizeRow(&x, cache[latent_row..]);
    var q = @as([lw]f32, @splat(0.5));
    const toks = [_]i32{1};
    var out: [lw]f32 = undefined;
    try msaReference(&q, 1, &cache, &toks, 1, scale, &out, a);
    var back: [lw]f32 = undefined;
    dequantRow(cache[latent_row..], &back);
    for (out, back) |o, b| try std.testing.expectEqual(bf16(b), o);
}
