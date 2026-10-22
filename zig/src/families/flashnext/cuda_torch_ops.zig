//! Flash Next's torch ops on the CUDA hot path, replaced with the same bits: our fn_ops.cu and fn_logsumexp.cu
//! (zig/kernels/cuda/torch_ops) and the Nemotron operators that already cover an op (bf16 -> fp32, argmax, the
//! unsorted top-k, strided copies and row gathers), each launched with the geometry of the C launcher that
//! tools/zig/check_flashnext_ops.py qualifies against torch; plain copies, fills and zeroing are driver calls.
//!
//! What each Python op becomes (TensorFold 0.6.5 qwen4_exp/cuda):
//!   nvfp4_moe.MoE4.shared_act + `buf.act[:R, top_k] = ...`   sharedSwiglu (one kernel, torch's rounding points)
//!   `buf.y[:R, top_k] = shared down`                         slotCopy (movement.cu strided copy)
//!   `b.streams[:R].copy_(b.h[:R])`, clones, `pss[0].copy_`  copy (cuMemcpyDtoDAsync)
//!   `pos_blk.fill_`, `pos_dev.fill_` (int32)                 fill32 (cuMemsetD32Async); `conv_ptr.fill_` fill64
//!   State.reset's `zero_()`                                  zero (cuMemsetD8Async)
//!   `b.pss.index_select`, `b.streams.index_select` (ends)    gatherRows (movement.cu)
//!   `logits.float()`                                         toF32 (pointwise.cu)
//!   `logits.argmax(dim=-1)` / `row.max(dim=-1)`              argmax (argmax.cu, int32 ids, the first maximum)
//!   `torch.topk(x, k, sorted=False)`                          topk (topk.cu)
//!   `torch.logsumexp(row, dim=-1)`                            logsumexp (fn_logsumexp.cu)
//!   sample_draft's `torch.cat([...])`, candidates' packing    draftPick, draftPack, candidates (fn_ops.cu)

const std = @import("std");
const cuda = @import("cuda");
const kk = cuda.kernels;

/// topk.cu's column tile and threads; its workspace is four 8-bit digit passes, then ordered compaction.
const topk_tile = 4096;
const topk_threads = 256;

fn tiles(columns: usize) usize {
    return (columns + topk_tile - 1) / topk_tile;
}

/// Scratch bytes `topk` needs for `rows` rows of `columns` (tf_topk_f32_unsorted_scratch_bytes).
pub fn topkScratchBytes(rows: usize, columns: usize) usize {
    return rows * (tiles(columns) * 260 + 3) * 4;
}

/// launch_shape.h's tf_launch_blocks: one block a `threads` elements, at most 65535.
fn blocks(count: usize, threads: usize) u32 {
    return @intCast(@max(@min((count + threads - 1) / threads, 65535), 1));
}

/// The modules in this order: Nemotron's argmax, topk, pointwise and movement fatbins, then ours.
pub const Images = enum { argmax, topk, pointwise, movement, ops, lse };

pub fn images() [6][]const u8 {
    return .{ kk.torch_argmax, kk.torch_topk, kk.torch_pointwise, kk.torch_movement, kk.torch_fn_ops, kk.torch_fn_logsumexp };
}

pub const Functions = struct {
    argmax: cuda.Function,
    hist: cuda.Function,
    digit: cuda.Function,
    count: cuda.Function,
    prefix: cuda.Function,
    compact: cuda.Function,
    to_f32: cuda.Function,
    strided: cuda.Function,
    gather: cuda.Function,
    swiglu: cuda.Function,
    fill64: cuda.Function,
    draft_pick: cuda.Function,
    draft_pack: cuda.Function,
    candidates: cuda.Function,
    nucleus_mass: cuda.Function,
    fp4_serial: cuda.Function,
    lse: Lse,

    pub fn resolve(m: []const cuda.Module) !Functions {
        const i = struct {
            fn at(e: Images) usize {
                return @intFromEnum(e);
            }
        }.at;
        return .{
            .argmax = try m[i(.argmax)].function("tf_argmax_rows_i32_kernel"),
            .hist = try m[i(.topk)].function("tf_topk_f32_histogram_kernel"),
            .digit = try m[i(.topk)].function("tf_topk_f32_choose_digit_kernel"),
            .count = try m[i(.topk)].function("tf_topk_f32_count_kernel"),
            .prefix = try m[i(.topk)].function("tf_topk_f32_prefix_kernel"),
            .compact = try m[i(.topk)].function("tf_topk_f32_compact_kernel"),
            .to_f32 = try m[i(.pointwise)].function("tf_bf16_to_f32_kernel"),
            .strided = try m[i(.movement)].function("tf_strided_copy_kernel"),
            .gather = try m[i(.movement)].function("tf_gather_rows_kernel"),
            .swiglu = try m[i(.ops)].function("tf_fn_shared_swiglu_kernel"),
            .fill64 = try m[i(.ops)].function("tf_fn_fill_u64_kernel"),
            .draft_pick = try m[i(.ops)].function("tf_fn_draft_pick_kernel"),
            .draft_pack = try m[i(.ops)].function("tf_fn_draft_pack_kernel"),
            .candidates = try m[i(.ops)].function("tf_fn_candidates_kernel"),
            .nucleus_mass = try m[i(.ops)].function("tf_fn_nucleus_mass_kernel"),
            .fp4_serial = try m[i(.ops)].function("tf_fn_fp4_serial_kernel"),
            .lse = try Lse.resolve(m[i(.lse)]),
        };
    }
};

/// fn_logsumexp.cu: torch.logsumexp(x, dim=-1, keepdim=True) on contiguous fp32 rows in ATen's reduction order:
/// the row maxima (|max| == inf -> 0), the exp(x - max) sum by ATen's reduce config for (rows, n) on this device,
/// its per-CTA partials folded, then log(sum) + max. The config needs n >= 128 and rows <= 65535.
pub const Lse = struct {
    max: cuda.Function,
    sum: cuda.Function,
    finish: cuda.Function,
    /// the device's multiprocessors and threads a multiprocessor (ATen sizes its CTAs by them); Ops.load sets them
    mp: usize = 0,
    threads_mp: usize = 0,

    fn resolve(m: cuda.Module) !Lse {
        return .{ .max = try m.function("tf_fn_lse_max_kernel"), .sum = try m.function("tf_fn_lse_sum_kernel"), .finish = try m.function("tf_fn_lse_finish_kernel") };
    }

    pub const Config = struct { bw: usize, bh: usize, grid_x: usize, ctas: usize, split: bool };

    fn lastPow2(v: usize) usize {
        return if (v <= 1) 1 else std.math.floorPowerOfTwo(usize, v);
    }

    /// tf_fn_logsumexp_config: ATen's block (bw, bh), rows a block, and CTAs a row for a last-dimension reduction.
    pub fn config(rows: usize, n: usize, mp: usize, threads_mp: usize) !Config {
        if (rows < 1 or n < 128 or rows > 65535 or n > (1 << 31)) return error.LseShape;
        const max_threads: usize = 512;
        const d0 = if (n / 4 < max_threads) lastPow2(n / 4) else max_threads;
        const d1 = if (rows < max_threads) lastPow2(rows) else max_threads;
        var bw: usize = @min(d0, 32);
        const bh: usize = @min(d1, max_threads / bw);
        bw = @min(d0, max_threads / bh);
        var step = bw;
        var rows_per_block: usize = 1;
        const vpt = cdiv(n, step);
        const split = vpt >= @min(16 * bh, 256);
        if (split) step *= bh else rows_per_block = bh;
        const grid_x = cdiv(rows, rows_per_block);
        const target = mp * (threads_mp / (bw * bh));
        var ctas: usize = 1;
        const v = cdiv(n, step);
        if (split and v >= 256 and grid_x <= target) ctas = @max(@min(cdiv(target, grid_x), cdiv(v, 16)), cdiv(v, 256));
        return .{ .bw = bw, .bh = bh, .grid_x = grid_x, .ctas = ctas, .split = split };
    }

    /// Scratch bytes: rows maxima, then rows * ctas partials.
    pub fn scratchBytes(l: Lse, rows: usize, n: usize) !usize {
        const c = try config(rows, n, l.mp, l.threads_mp);
        return rows * (1 + c.ctas) * 4;
    }

    fn run(l: Lse, t: Torch, in: u64, rows: usize, n: usize, out: u64, scratch: u64) !void {
        const c = try config(rows, n, l.mp, l.threads_mp);
        const maxes = scratch;
        const partial = scratch + rows * 4;
        var a: cuda.Args = .{};
        a.add(in);
        a.add(maxes);
        a.add(@as(u32, @intCast(rows)));
        a.add(@as(u32, @intCast(n)));
        try t.go(l.max, .{ rows, 1 }, 256, &a);
        var b: cuda.Args = .{};
        for ([_]u64{ in, maxes, partial }) |v| b.add(v);
        for ([_]usize{ rows, n, c.ctas, @intFromBool(c.split) }) |v| b.add(@as(u32, @intCast(v)));
        try cuda.launch.launch(l.sum, .{ .grid = .{ .x = @intCast(c.grid_x), .y = @intCast(c.ctas) }, .block = .{ .x = @intCast(c.bw), .y = @intCast(c.bh) } }, t.s, &b);
        var f: cuda.Args = .{};
        for ([_]u64{ partial, maxes, out }) |v| f.add(v);
        f.add(@as(u32, @intCast(rows)));
        f.add(@as(u32, @intCast(c.ctas)));
        try cuda.launch.launch(l.finish, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = @intCast(c.bw), .y = @intCast(c.bh) } }, t.s, &f);
    }
};

/// Threads a multiprocessor by compute capability (the driver attribute this runtime does not list): 1536 on
/// sm_86/89 and sm_12x, 2048 on sm_80/90/100.
pub fn threadsPerMultiprocessor(cc: u32) usize {
    return switch (cc) {
        86, 87, 89, 120, 121 => 1536,
        else => 2048,
    };
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// The loaded modules and their entry points (one per process; any stream launches them).
pub const Ops = struct {
    mods: [6]cuda.Module,
    f: Functions,

    pub fn load(ctx: *const cuda.Context) !Ops {
        const d = ctx.d;
        var o: Ops = undefined;
        var n: usize = 0;
        errdefer for (o.mods[0..n]) |*m| m.unload();
        for (images(), 0..) |image, j| {
            o.mods[j] = try cuda.Module.load(d, image);
            n += 1;
        }
        o.f = try Functions.resolve(&o.mods);
        o.f.lse.mp = @intCast(try ctx.attribute(.multiprocessor_count));
        o.f.lse.threads_mp = threadsPerMultiprocessor(try ctx.capability());
        return o;
    }

    pub fn deinit(o: *Ops) void {
        for (&o.mods) |*m| m.unload();
        o.* = undefined;
    }

    pub fn on(o: *const Ops, s: cuda.Stream) Torch {
        return .{ .f = &o.f, .s = s };
    }
};

/// The shapes tf_fn_fp4_serial_kernel takes: 32-column CTA tiles inside the 64-column table tiles, 16-byte rows.
pub fn fp4SerialFits(x_stride: usize, n: usize, k: usize) bool {
    return n % 32 == 0 and k % 64 == 0 and x_stride % 8 == 0 and n / 32 <= 65535;
}

pub const Torch = struct {
    f: *const Functions,
    s: cuda.Stream,

    fn go(t: Torch, f: cuda.Function, grid: [2]usize, block: u32, args: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]) }, .block = .{ .x = block } }, t.s, args);
    }

    fn check(t: Torch, rc: anytype, what: []const u8) !void {
        try t.s.d.check(rc, what);
    }

    // -- copies, fills -------------------------------------------------------------------------------------

    /// Tensor.copy_ / clone of contiguous bytes (torch's same-dtype device copy is a memcpy).
    pub fn copy(t: Torch, dst: u64, src: u64, bytes: usize) !void {
        if (bytes == 0) return;
        try t.check(t.s.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, t.s.handle), "cuMemcpyDtoDAsync");
    }

    /// Tensor.fill_ of `count` 32-bit words (pos_dev, mtp_pos, rope_delta_dev, pos_blk).
    pub fn fill32(t: Torch, dst: u64, value: u32, count: usize) !void {
        if (count == 0) return;
        try t.check(t.s.d.api.cuMemsetD32Async(dst, value, count, t.s.handle), "cuMemsetD32Async");
    }

    /// Tensor.fill_ of `count` 64-bit words (prefill's conv-state pointer for gdn_io.front).
    pub fn fill64(t: Torch, dst: u64, value: u64, count: usize) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        a.add(dst);
        a.add(value);
        a.add(@as(u64, count));
        try t.go(t.f.fill64, .{ blocks(count, 256), 1 }, 256, &a);
    }

    /// Tensor.zero_ (State.reset: conv windows, recurrent states, the n-gram tail).
    pub fn zero(t: Torch, dst: u64, bytes: usize) !void {
        if (bytes == 0) return;
        try t.check(t.s.d.api.cuMemsetD8Async(dst, 0, bytes, t.s.handle), "cuMemsetD8Async");
    }

    /// A slot assignment `dst[:rows, slot] = src` (or any `.copy_` between row-strided views): `rows` rows of
    /// `row_bytes`, `src_ld` and `dst_ld` bytes apart, one block a row.
    pub fn slotCopy(t: Torch, src: u64, src_ld: usize, dst: u64, dst_ld: usize, row_bytes: usize, rows: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        for ([_]u64{ rows, 1, row_bytes, src_ld, row_bytes, dst_ld, row_bytes }) |v| a.add(v);
        try t.go(t.f.strided, .{ rows, 1 }, 256, &a);
    }

    /// Tensor.index_select(0, idx) of rows (int64 indices on the device); a bad index sets `invalid`.
    pub fn gatherRows(t: Torch, src: u64, dst: u64, indices: u64, rows: usize, source_rows: usize, row_bytes: usize, invalid: u64) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(indices);
        for ([_]u64{ rows, source_rows, row_bytes }) |v| a.add(v);
        a.add(invalid);
        try t.go(t.f.gather, .{ rows, 1 }, 256, &a);
    }

    // -- the shared expert ---------------------------------------------------------------------------------

    /// MoE4.shared_act into its slot: g [rows, 2 ni] bf16 (gate | up) -> out rows `out_stride` elements apart
    /// (buf.act[:, top_k]: (top_k + 1) * ni).
    pub fn sharedSwiglu(t: Torch, g: u64, out: u64, rows: usize, ni: usize, out_stride: usize) !void {
        if (rows * ni == 0) return;
        var a: cuda.Args = .{};
        a.add(g);
        a.add(out);
        for ([_]u64{ rows, ni, out_stride }) |v| a.add(v);
        try t.go(t.f.swiglu, .{ blocks(rows * ni, 256), 1 }, 256, &a);
    }

    // -- sampling ------------------------------------------------------------------------------------------

    /// `.float()` of `count` bf16 values (exact).
    pub fn toF32(t: Torch, in: u64, out: u64, count: usize) !void {
        if (count == 0) return;
        var a: cuda.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(u64, count));
        try t.go(t.f.to_f32, .{ blocks(count, 256), 1 }, 256, &a);
    }

    /// torch.argmax / max(dim=-1)'s index over rows of bf16 logits `ld` apart, as int32: the first maximum, a NaN
    /// first.
    pub fn argmax(t: Torch, logits: u64, vocab: usize, ld: usize, out: u64, rows: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(out);
        for ([_]usize{ rows, vocab, ld }) |v| a.add(@as(i64, @intCast(v)));
        a.add(@as(i32, 0)); // dtype 0: bf16
        try t.go(t.f.argmax, .{ rows, 1 }, 256, &a);
    }

    /// torch.topk(rows of fp32 `columns` wide, k, sorted=False): values [rows, k] fp32 and int64 columns, in
    /// torch's order; `scratch` is topkScratchBytes(rows, columns).
    pub fn topk(t: Torch, in: u64, columns: usize, rows: usize, k: usize, values: u64, indices: u64, scratch: u64) !void {
        const n = tiles(columns);
        const row_bytes: u64 = columns * 4;
        const hist = scratch;
        const threshold = hist + rows * n * 256 * 4;
        const remaining = threshold + rows * 4;
        const greater = remaining + rows * 4;
        const equal = greater + rows * n * 4;
        const greater_prefix = equal + rows * n * 4;
        const equal_prefix = greater_prefix + rows * n * 4;
        const greater_total = equal_prefix + rows * n * 4;
        var shift: u32 = 24;
        while (true) : (shift -= 8) {
            var h: cuda.Args = .{};
            for ([_]u64{ in, hist, threshold, columns, n, row_bytes, 4 }) |v| h.add(v);
            h.add(shift);
            try t.go(t.f.hist, .{ n, rows }, topk_threads, &h);
            var d: cuda.Args = .{};
            for ([_]u64{ hist, threshold, remaining, n }) |v| d.add(v);
            d.add(@as(u32, @intCast(k)));
            d.add(shift);
            try t.go(t.f.digit, .{ rows, 1 }, topk_threads, &d);
            if (shift == 0) break;
        }
        var c: cuda.Args = .{};
        for ([_]u64{ in, threshold, greater, equal, columns, n, row_bytes, 4 }) |v| c.add(v);
        try t.go(t.f.count, .{ n, rows }, topk_threads, &c);
        var p: cuda.Args = .{};
        for ([_]u64{ greater, equal, greater_prefix, equal_prefix, greater_total, n }) |v| p.add(v);
        try t.go(t.f.prefix, .{ rows, 1 }, 1, &p);
        var w: cuda.Args = .{};
        for ([_]u64{ in, values, indices, threshold, greater_prefix, equal_prefix, greater_total, columns, n, row_bytes, 4 }) |v| w.add(v);
        w.add(@as(u32, @intCast(k)));
        try t.go(t.f.compact, .{ n, rows }, topk_threads, &w);
    }

    /// torch.logsumexp(rows of fp32, dim=-1, keepdim=True) -> out [rows] fp32; `scratch` is Lse.scratchBytes.
    pub fn logsumexp(t: Torch, in: u64, rows: usize, columns: usize, out: u64, scratch: u64) !void {
        try t.f.lse.run(t, in, rows, columns, out, scratch);
    }

    /// sample_draft (greedy, one rank): out [3] = [the row's maximum, its log-sum-exp, the column as fp32].
    pub fn draftPick(t: Torch, logits: u64, col: u64, lse: u64, out: u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ logits, col, lse, out }) |v| a.add(v);
        try t.go(t.f.draft_pick, .{ 1, 1 }, 32, &a);
    }

    /// sample_draft (sampled, one rank): out [2k + 1] = [k values | log-sum-exp | k columns as fp32].
    pub fn draftPack(t: Torch, vals: u64, idx: u64, lse: u64, out: u64, k: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ vals, idx, lse, out, k }) |v| a.add(v);
        try t.go(t.f.draft_pack, .{ blocks(k + 1, 256), 1 }, 256, &a);
    }

    /// forward.candidates' row packing (two ranks): out [rows, 2 cand + 1] = [values | ids as int32 bits | lse],
    /// ids = id_map[idx] (the draft head; `id_map` int64) or idx + offset (the main head's rank shard).
    /// sampling.nucleus_rows' fixed-point mass of `rows` rows of bf16 logits (`ld` apart): mass [rows, cols] int64 =
    /// floor(exp(f64(logit) / t - top[r]) * 2^40) and each row's sum into `sums` (zeroed here first).
    pub fn nucleusMass(t: Torch, logits: u64, ld: usize, rows: usize, cols: usize, temp: f64, top: u64, mass: u64, sums: u64) !void {
        if (rows == 0 or cols == 0) return;
        try t.zero(sums, rows * 8);
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(u64, ld));
        a.add(@as(u64, cols));
        a.add(temp);
        a.add(top);
        a.add(mass);
        a.add(sums);
        try t.go(t.f.nucleus_mass, .{ (cols + 255) / 256, rows }, 256, &a);
    }

    /// nvfp4.matmul with one K slice and bf16-pattern tables (`_fp4mm` SK 1, PACKED 0) in the same bits: each
    /// 16-input block's HMMA (C 0) then fma by its scale, in block order (fn_ops.cu tf_fn_fp4_serial_kernel), a warp
    /// 16 rows x 8 columns, the blocks streamed through a cp.async ring. x [m, k] bf16 rows `x_stride` apart.
    pub fn fp4Serial(t: Torch, x: u64, x_stride: usize, w: u64, scale: u64, out: u64, fp32: bool, m: usize, n: usize, k: usize) !void {
        if (m == 0) return;
        if (!fp4SerialFits(x_stride, n, k)) return error.Invalid;
        var a: cuda.Args = .{};
        for ([_]u64{ x, w, scale, out }) |v| a.add(v);
        for ([_]usize{ m, n, k, x_stride, @intFromBool(fp32) }) |v| a.add(@as(u32, @intCast(v)));
        try t.go(t.f.fp4_serial, .{ (m + 15) / 16, n / 32 }, 128, &a);
    }

    pub fn candidates(t: Torch, vals: u64, idx: u64, id_map: ?u64, offset: i64, lse: u64, out: u64, rows: usize, cand: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(vals);
        a.add(idx);
        a.add(id_map orelse 0);
        a.add(offset);
        a.add(lse);
        a.add(out);
        a.add(@as(u64, rows));
        a.add(@as(u64, cand));
        try t.go(t.f.candidates, .{ rows, 1 }, 64, &a);
    }

    // -- the sampler's device work, as the Python functions run it ----------------------------------------

    /// Scratch of one row-sampling pass over `rows` rows of `columns` logits with `k` candidates.
    pub const Sample = struct {
        f32: u64, // rows * columns fp32 (`logits.float()`)
        vals: u64, // rows * k fp32
        idx: u64, // rows * k int64
        topk: u64, // topkScratchBytes(rows, columns)
        lse: u64, // rows fp32
        lse_scratch: u64, // Lse.scratchBytes(rows, columns)
        col: u64, // rows int32

        pub fn bytes(f: *const Functions, rows: usize, columns: usize, k: usize) ![7]usize {
            return .{ rows * columns * 4, rows * k * 4, rows * k * 8, topkScratchBytes(rows, columns), rows * 4, try f.lse.scratchBytes(rows, columns), rows * 4 };
        }
    };

    /// cuda/sampling.sample_rows with top_k > 0 (and decode.sample_mapped): `logits.float()` then the unsorted
    /// top_k + MARGIN; the host reads s.vals and s.idx (columns; the draft head's map to ids on the host).
    pub fn sampleTopk(t: Torch, logits: u64, rows: usize, columns: usize, k: usize, s: Sample) !void {
        try t.toF32(logits, s.f32, rows * columns);
        try t.topk(s.f32, columns, rows, k, s.vals, s.idx, s.topk);
    }

    /// Engine.sample_draft greedy: out [3] (draftPick) from the draft head's bf16 row.
    pub fn draftGreedy(t: Torch, logits: u64, columns: usize, s: Sample, out: u64) !void {
        try t.toF32(logits, s.f32, columns);
        try t.logsumexp(s.f32, 1, columns, s.lse, s.lse_scratch);
        try t.argmax(logits, columns, columns, s.col, 1);
        try t.draftPick(logits, s.col, s.lse, out);
    }

    /// Engine.sample_draft with a top_k: out [2k + 1] (draftPack) from the draft head's bf16 row.
    pub fn draftTopk(t: Torch, logits: u64, columns: usize, k: usize, s: Sample, out: u64) !void {
        try t.toF32(logits, s.f32, columns);
        try t.logsumexp(s.f32, 1, columns, s.lse, s.lse_scratch);
        try t.topk(s.f32, columns, 1, k, s.vals, s.idx, s.topk);
        try t.draftPack(s.vals, s.idx, s.lse, out, k);
    }

    /// forward.candidates (two ranks): `rows` rows of this rank's bf16 logits `columns` wide -> out [rows, 2 cand + 1]
    /// (before the all-gather into cand_all).
    pub fn rankCandidates(t: Torch, logits: u64, rows: usize, columns: usize, cand: usize, id_map: ?u64, offset: i64, s: Sample, out: u64) !void {
        try t.toF32(logits, s.f32, rows * columns);
        try t.topk(s.f32, columns, rows, cand, s.vals, s.idx, s.topk);
        try t.logsumexp(s.f32, rows, columns, s.lse, s.lse_scratch);
        try t.candidates(s.vals, s.idx, id_map, offset, s.lse, out, rows, cand);
    }
};

test "topk scratch matches topk.cu's workspace" {
    try std.testing.expectEqual(@as(usize, 16 * (61 * 260 + 3) * 4), topkScratchBytes(16, 248320));
    try std.testing.expectEqual(@as(usize, (20 * 260 + 3) * 4), topkScratchBytes(1, 79591));
}

test "launch blocks follow launch_shape.h" {
    try std.testing.expectEqual(@as(u32, 1), blocks(1, 256));
    try std.testing.expectEqual(@as(u32, 3), blocks(513, 256));
    try std.testing.expectEqual(@as(u32, 65535), blocks(1 << 40, 256));
    // the shared expert's SwiGLU at 16 rows of 640
    try std.testing.expectEqual(@as(u32, 40), blocks(16 * 640, 256));
}

test "logsumexp's reduce config follows ATen's for the sampler's rows (GB10: 48 multiprocessors, 1536 threads)" {
    // the draft head's row: one row split over CTAs; a rank's 16 candidate rows
    const one = try Lse.config(1, 79591, 48, 1536);
    try std.testing.expectEqual(@as(usize, 512), one.bw * one.bh);
    try std.testing.expect(one.split);
    const many = try Lse.config(16, 124160, 48, 1536);
    try std.testing.expectEqual(@as(usize, 16), many.grid_x);
    // the TP=1 head's width splits each row over CTAs (fn_logsumexp.cu's parity table on spark3)
    for ([_][2]usize{ .{ 1, 31 }, .{ 4, 31 }, .{ 5, 29 }, .{ 6, 24 }, .{ 7, 21 }, .{ 8, 18 }, .{ 12, 12 }, .{ 13, 12 }, .{ 16, 9 } }) |rc| {
        try std.testing.expectEqual(rc[1], (try Lse.config(rc[0], 248320, 48, 1536)).ctas);
    }
    try std.testing.expectEqual(@as(usize, 1), (try Lse.config(16, 124160, 48, 1536)).ctas);
    try std.testing.expect(!(try Lse.config(1, 4096, 48, 1536)).split);
    try std.testing.expectError(error.LseShape, Lse.config(1, 64, 48, 1536));
    try std.testing.expectEqual(@as(usize, 1536), threadsPerMultiprocessor(121));
}
