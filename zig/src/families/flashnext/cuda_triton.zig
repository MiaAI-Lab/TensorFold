//! Flash Next's Triton kernels from the captured cubins: one function a Python wrapper of the qwen4_exp CUDA engine
//! (TensorFold 0.6.5: qwen4_exp/cuda/{glue,bf16,nvfp4,attention,forward,exl3_mm}.py and cuda/moe.py) with that
//! wrapper's grid, runtime arguments (names, Triton pointer types, ints) and constexprs, so `aot.Set.find` picks the
//! variant Python launched. Shapes are the NVIDIA ModelOpt NVFP4 checkpoint's: at TP=1 Python's own, at TP=2 a rank's
//! (work/PLAN.md "TP=2 design": half the heads, half the expert width, fp32 partials gathered for `_hc_writeback`
//! mode 3). The tests replay fixtures_cuda_triton.json, which tools/zig/flashnext_triton_fixtures.py records by
//! calling the Python wrappers themselves with recorders in place of their kernels.
//!
//! Kernel names are Triton's function names as the pack writes them (`fn`). Three are shared by name with different
//! sources: `_reduce` (bf16.py and nvfp4.py, the same arguments and bits) and `_embed` (exl3_mm.py, the one the NVFP4
//! checkpoint's bf16 table takes; glue.py's 4-bit `_embed` has other argument names, so `find` keeps them apart).

const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;

const p = aot.ptr;

const bf16 = "*bf16";
const f32p = "*fp32";
const f16p = "*fp16";
const i32p = "*i32";

fn int(name: []const u8, v: usize) aot.Arg {
    return aot.int(name, @intCast(v));
}

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

fn cb(name: []const u8, v: bool) aot.Const {
    return aot.ci(name, @intFromBool(v));
}

fn u(x: usize) u32 {
    return @intCast(x);
}

fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// triton.next_power_of_2
pub fn pow2(n: usize) usize {
    return std.math.ceilPowerOfTwo(usize, @max(n, 1)) catch unreachable;
}

/// The checkpoint's dimensions a rank computes with (config.json's text_config; `rank(2)` halves what TP=2 splits).
pub const Dims = struct {
    hidden: usize = 2560,
    streams: usize = 4,
    low: usize = 320,
    heads: usize = 24,
    kv_heads: usize = 2,
    head_dim: usize = 256,
    index_heads: usize = 4,
    index_dim: usize = 128,
    rotary_half: usize = 32,
    experts: usize = 512,
    top_k: usize = 10,
    moe_width: usize = 640,
    vocab: usize = 248320,
    key_heads: usize = 16,
    value_heads: usize = 48,
    gdn_dim: usize = 128,
    ngram_heads: usize = 16,
    ngram_dim: usize = 160,
    ple_taps: usize = 4,
    ngram_size: usize = 3,
    eps: f32 = 1e-6,

    /// A rank's share at `world` ranks: heads, kv heads, DeltaNet heads, expert widths and the vocabulary split.
    pub fn rank(world: usize) Dims {
        const d: Dims = .{};
        return .{ .heads = d.heads / world, .kv_heads = d.kv_heads / world, .moe_width = d.moe_width / world, .vocab = d.vocab / world, .key_heads = d.key_heads / world, .value_heads = d.value_heads / world };
    }

    pub fn wide(d: Dims) usize {
        return d.streams * d.hidden;
    }
    /// Buffers.pa: [q | gate] per head, k, v, then the indexer's q heads and its k.
    pub fn attnWidth(d: Dims) usize {
        return d.heads * 2 * d.head_dim + 2 * d.kv_heads * d.head_dim + (d.index_heads + 1) * d.index_dim;
    }
    /// gdn.widths: the conv channels (q, k, v) and the projection row (+ z, b, a).
    pub fn convDim(d: Dims) usize {
        return 2 * d.key_heads * d.gdn_dim + d.value_heads * d.gdn_dim;
    }
    pub fn projWidth(d: Dims) usize {
        return d.convDim() + d.value_heads * d.gdn_dim + 2 * d.value_heads;
    }
    /// Rows of the n-gram conv tail: (ple_kernel - 1) * ngram_size.
    pub fn tailRows(d: Dims) usize {
        return (d.ple_taps - 1) * d.ngram_size;
    }
};

/// One launch as `aot.Set.run` takes it, copied (names and types are static strings) for the tests.
pub const Recorder = struct {
    launches: [64]Saved = undefined,
    n: usize = 0,

    pub const Saved = struct {
        name: []const u8,
        grid: [3]u32,
        args: [24]aot.Arg = undefined,
        nargs: usize,
        consts: [16]aot.Const = undefined,
        nconsts: usize,
    };

    fn add(r: *Recorder, name: []const u8, grid: [3]u32, args: []const aot.Arg, consts: []const aot.Const) !void {
        if (r.n == r.launches.len) return error.TooManyLaunches;
        const s = &r.launches[r.n];
        s.* = .{ .name = name, .grid = grid, .nargs = args.len, .nconsts = consts.len };
        @memcpy(s.args[0..args.len], args);
        @memcpy(s.consts[0..consts.len], consts);
        r.n += 1;
    }
};

/// What `_hc_writeback` adds to the streams before their read-out (glue.hc_writeback's `mode`).
pub const Branch = union(enum) {
    /// mode 0: no pending branch (the first layer); the streams' squared sums only
    none,
    /// mode 1: a block's bf16 output [R, D]
    bf16: u64,
    /// mode 2: the MoE's slots y [R, slots, D] (fp32 in the MTP head's buffers, bf16 in the main ones) and weights
    moe: struct { y: u64, y_f32: bool, wts: u64, slots: usize },
    /// mode 3: every rank's fp32 partials [world, R, D], summed rank 0 first (TP=2)
    ranks: struct { part: u64, world: usize },
    /// mode 4: a matmul's fp32 K slices [SK, R, D]
    slices: struct { part: u64, sk: usize },
};

/// attention.AttnScratch's geometry for a buffer's capacity: partial chunks, the sparse key lists, the block scores.
pub const AttnGeometry = struct {
    nch: usize,
    qsa: bool,
    idw: usize,
    nb: usize,
    budget: usize,
    ratio: usize,

    pub const chunk = 512;
    /// _select holds a row's block scores in registers up to this many
    pub const select_regs = 32768;

    pub fn init(capacity: usize, budget: usize, ratio: usize) AttnGeometry {
        return .{ .nch = cdiv(@min(capacity, budget + ratio - 1), chunk), .qsa = capacity > budget, .idw = budget + ratio, .nb = cdiv(capacity, ratio), .budget = budget, .ratio = ratio };
    }

    /// The chunks a launch covers for `context` keys (null: the whole scratch), as attention.attention bounds them.
    pub fn chunks(g: AttnGeometry, context: ?usize) usize {
        var keys = context orelse g.nch * chunk;
        if (g.qsa) keys = @min(keys, (g.budget / g.ratio + 1) * g.ratio - 1);
        return @min(g.nch, cdiv(keys, chunk));
    }

    /// qsa_rows' scored blocks for `context` keys.
    pub fn blocks(g: AttnGeometry, context: ?usize) usize {
        const c = context orelse return g.nb;
        return @min(g.nb, @max(1, cdiv(c, g.ratio)));
    }
};

/// AttnScratch's device buffers: chunk partials, each row's key list, its length and whether it is sparse, scores.
pub const AttnScratch = struct { po: u64, pm: u64, pl: u64, ids: u64, nk: u64, sparse: u64, scores: u64 };

/// The attention caches of one layer: keys and values (bf16, or int8 / int4 codes) and their fp16 scales (a bf16
/// cache keeps one-element scales, kvcache.KVCache, so the kernels take one argument list).
pub const Cache = struct {
    k: u64,
    v: u64,
    ks: u64,
    vs: u64,
    bits: usize = 0,

    fn codes(c: Cache) []const u8 {
        return switch (c.bits) {
            0 => bf16,
            8 => "*i8",
            else => "*u8",
        };
    }
};

/// An NVFP4 matmul's table (nvfp4.FP4): bf16 patterns and fp32 scales (the shared expert, `packed` false) or the
/// checkpoint's codes and e4m3 scales (`codes`, Python's `packed`).
pub const Fp4 = struct { weight: u64, scale: u64, scale2: u64, codes: bool = false };

/// bf16.split_k: K slices by the weight's shape alone, a power of two of whole 64-blocks.
pub fn b16SplitK(n: usize, k: usize) usize {
    const tiles = cdiv(n, 64);
    const blocks = k / 64;
    var sk: usize = 1;
    while (sk < 32 and tiles * sk < 160 and blocks % (sk * 2) == 0 and blocks / (sk * 2) >= 1) sk *= 2;
    return sk;
}

/// The fp32 scratch bf16.matmul's split K needs for `m` rows (0 without a split).
pub fn b16PartBytes(m: usize, n: usize, k: usize) usize {
    const sk = b16SplitK(n, k);
    return if (sk > 1) sk * m * n * 4 else 0;
}

/// nvfp4.split_for: the shapes tuned at up to 16 rows keep their K slices at every row count, else split_k.
pub fn fp4SplitK(n: usize, k: usize) usize {
    if (fp4Tuned(n, k)) |t| return t.sk;
    const tiles = cdiv(n, 64);
    const blocks = k / 16;
    var sk: usize = 1;
    while (sk < 32 and tiles * sk < 160 and blocks % (sk * 2) == 0 and blocks / (sk * 2) >= 4) sk *= 2;
    return sk;
}

pub fn fp4PartBytes(m: usize, n: usize, k: usize) usize {
    const sk = fp4SplitK(n, k);
    return if (sk > 1) sk * m * n * 4 else 0;
}

const Fp4Tuned = struct { sk: usize, gpi: usize };

/// nvfp4.SHAPES16 (the warps and stages it also sets are not constexprs).
fn fp4Tuned(n: usize, k: usize) ?Fp4Tuned {
    if (k == 2560 and (n == 640 or n == 1280)) return .{ .sk = 1, .gpi = 4 };
    if (n == 2560 and k == 640) return .{ .sk = 8, .gpi = 2 };
    return null;
}

/// nvfp4.bucket / moe._tile: 16 to 128 rows, then 128-row tiles.
fn bucket(m: usize) usize {
    for ([_]usize{ 16, 32, 64, 128 }) |b| if (m <= b) return b;
    return 128;
}

/// nvfp4.gpi_for
fn gpiFor(per: usize, want: usize) usize {
    for ([_]usize{ want, 8, 4, 2, 1 }) |g| if (g <= want and per % g == 0) return g;
    return 1;
}

/// The variant `aot.Set.find` picked for each (function, constexprs, arguments' specialization) seen, so a repeated
/// launch skips the scan over every captured variant: a decode round enqueues thousands of launches, and the scan
/// was most of their host time. The pick is the one `find` makes (the key holds everything `find` matches on).
pub const Memo = struct {
    gpa: std.mem.Allocator,
    map: std.AutoHashMapUnmanaged(u64, usize) = .empty,

    pub fn deinit(m: *Memo) void {
        m.map.deinit(m.gpa);
    }

    fn key(name: []const u8, args: []const aot.Arg, consts: []const aot.Const) u64 {
        var h = std.hash.Wyhash.init(0x7472693a);
        h.update(name);
        for (consts) |c| {
            h.update(&.{0xc0});
            h.update(c.name);
            if (c.int) |x| h.update(std.mem.asBytes(&x));
            if (c.f32) |x| h.update(std.mem.asBytes(&@as(u32, @bitCast(x))));
            h.update(&.{ @intFromBool(c.int != null), @intFromBool(c.f32 != null) });
        }
        for (args) |a| {
            h.update(&.{0xa0});
            h.update(a.name);
            switch (a.value) {
                .ptr => |x| {
                    h.update(x.ty);
                    h.update(&.{ 1, @intFromBool(x.addr % 16 == 0) });
                },
                .i32 => |x| h.update(&.{ 2, @intFromBool(x == 1), @intFromBool(@mod(x, 16) == 0) }),
                .f32 => h.update(&.{3}),
                .u64 => |x| h.update(&.{ 4, @intFromBool(x % 16 == 0) }),
            }
        }
        return h.final();
    }
};

pub const Tri = struct {
    set: ?*const aot.Set,
    s: cuda.Stream = undefined,
    /// tests: record each launch instead of running it
    rec: ?*Recorder = null,
    /// the variants already picked (null: `find` every launch)
    memo: ?*Memo = null,

    fn run(t: Tri, name: []const u8, grid: [3]usize, args: []const aot.Arg, consts: []const aot.Const) !void {
        const g: [3]u32 = .{ u(grid[0]), u(grid[1]), u(grid[2]) };
        if (t.rec) |r| return r.add(name, g, args, consts);
        const set = t.set.?;
        const m = t.memo orelse return set.run(t.s, name, g, args, consts);
        const k = Memo.key(name, args, consts);
        const at = m.map.get(k) orelse blk: {
            const v = try set.find(name, args, consts);
            const i = (@intFromPtr(v) - @intFromPtr(set.variants.ptr)) / @sizeOf(@TypeOf(set.variants[0]));
            try m.map.put(m.gpa, k, i);
            break :blk i;
        };
        // aot.Set.run's launch: runtime arguments in the variant's order
        const v = &set.variants[at];
        var packed_args: cuda.launch.Args = .{};
        for (v.spec.params) |prm| {
            const a = for (args) |x| {
                if (std.mem.eql(u8, x.name, prm.name)) break x;
            } else return error.MissingTritonArgument;
            switch (a.value) {
                .ptr => |x| packed_args.add(x.addr),
                .i32 => |x| packed_args.add(x),
                .f32 => |x| packed_args.add(x),
                .u64 => |x| packed_args.add(x),
            }
        }
        try v.kernel.launchOn(.{ .x = g[0], .y = g[1], .z = g[2] }, t.s, &packed_args, .{}, &.{});
    }

    // -- embedding, hyper-connections, norms ---------------------------------------------------------------

    /// exl3_mm.embed: ids [R] int32 -> out [R, copies * d] bf16, the unquantized table's row in each stream.
    pub fn embed(t: Tri, ids: u64, table: u64, out: u64, rows: usize, d: usize, copies: usize) !void {
        try t.run("_embed", .{ rows, d / 256, 1 }, &.{ p("IDS", i32p, ids), p("T", bf16, table), p("OUT", bf16, out) }, &.{ ci("D", d), ci("S", copies), ci("BLOCK", 256) });
    }

    /// glue.hc_writeback: the pending branch into the streams h [R, S*D] (in place when hout == h), each stream's
    /// squared sums per 256 dims into pss [R, D/256, S]; `inject` [R, S] bf16 gates (unused in mode 0).
    pub fn hcWriteback(t: Tri, h: u64, hout: u64, pss: u64, inject: ?u64, branch: Branch, rows: usize, d: usize, streams: usize) !void {
        const block = 256;
        var br = p("BR", bf16, h);
        var y = p("Y", bf16, h);
        var wts = p("WTS", bf16, h);
        var mode: usize = 0;
        var top: usize = 1;
        var slots: usize = 1;
        var world: usize = 1;
        switch (branch) {
            .none => {},
            .bf16 => |b| {
                mode = 1;
                br = p("BR", bf16, b);
            },
            .moe => |m| {
                mode = 2;
                y = p("Y", if (m.y_f32) f32p else bf16, m.y);
                wts = p("WTS", f32p, m.wts);
                top = m.slots - 1;
                slots = m.slots;
            },
            .ranks => |r| {
                mode = 3;
                br = p("BR", f32p, r.part);
                world = r.world;
            },
            .slices => |s| {
                mode = 4;
                br = p("BR", f32p, s.part);
                world = s.sk;
            },
        }
        if (mode != 0 and inject == null) return error.MissingInject;
        const inj = p("INJ", bf16, inject orelse h);
        try t.run("_hc_writeback", .{ rows, d / block, 1 }, &.{ p("H", bf16, h), p("HOUT", bf16, hout), p("PSS", f32p, pss), br, inj, y, wts, int("RS", rows * d) }, &.{ ci("D", d), ci("S", streams), ci("MODE", mode), ci("TOPK", top), ci("SLOTS", slots), ci("BLOCK", block), ci("WORLD", world) });
    }

    /// glue.hc_normed: normed [R, S*D] = bf16(h * rinv_s * scale) and its 32-group sums xs, rinv from pss.
    pub fn hcNormed(t: Tri, h: u64, pss: u64, scale: u64, normed: u64, xs: u64, rows: usize, d: usize, streams: usize, eps: f32) !void {
        const block = 512;
        try t.run("_hc_normed", .{ rows, streams * d / block, 1 }, &.{ p("H", bf16, h), p("PSS", f32p, pss), p("SCALE", f32p, scale), p("NORMED", bf16, normed), p("XS", f32p, xs), aot.float("eps", eps) }, &.{ ci("D", d), ci("S", streams), ci("NC", d / 256), ci("BLOCK", block) });
    }

    /// glue.hc_act: the down projection's fp32 rows dn [R, ndn] -> act [R, low] bf16 (SiLU), its group sums, and
    /// with `inject` (ndn = low + S) the inject gates [R, S].
    pub fn hcAct(t: Tri, dn: u64, act: u64, xs: u64, inject: ?u64, rows: usize, ndn: usize, streams: usize, low: usize) !void {
        try t.run("_hc_act", .{ rows, 1, 1 }, &.{ p("DN", f32p, dn), p("ACT", bf16, act), p("XS", f32p, xs), p("INJ", bf16, inject orelse act) }, &.{ ci("S", streams), ci("LOW", low), ci("LOWP", pow2(low)), cb("HAS_INJ", inject != null), ci("NDN", ndn) });
    }

    /// glue.hc_mix: mixed [R, D] = bf16(sum_s bf16(sigmoid(up_s) * normed_s) / S) and its group sums.
    pub fn hcMix(t: Tri, up: u64, normed: u64, mixed: u64, xs: u64, rows: usize, d: usize, streams: usize) !void {
        try t.run("_hc_mix", .{ rows, d / 256, 1 }, &.{ p("UP", bf16, up), p("NORMED", bf16, normed), p("MIXED", bf16, mixed), p("XS", f32p, xs) }, &.{ ci("D", d), ci("S", streams), ci("BLOCK", 256) });
    }

    /// glue.rmsnorm: each run of `group` features (null: the whole row) normalized, times w (fp32), bf16 out [R, d]
    /// and its 32-group sums; x rows `x_stride` elements apart.
    pub fn rmsnorm(t: Tri, x: u64, x_stride: usize, w: u64, out: u64, xs: u64, rows: usize, d: usize, group: ?usize, eps: f32) !void {
        const g = group orelse d;
        try t.run("_rmsnorm", .{ rows, d / g, 1 }, &.{ p("X", bf16, x), p("W", f32p, w), p("OUT", bf16, out), p("XS", f32p, xs), aot.float("eps", eps), int("x_stride", x_stride) }, &.{ ci("D", d), ci("G", g), ci("BLOCK", pow2(g)) });
    }

    /// mtp.mtp_compute's input: out [R, S*D] = bf16(e + hs) per stream.
    pub fn addStreams(t: Tri, e: u64, hs: u64, out: u64, rows: usize, d: usize, streams: usize) !void {
        try t.run("_add_streams", .{ rows, d / 256, 1 }, &.{ p("E", bf16, e), p("HS", bf16, hs), p("OUT", bf16, out) }, &.{ ci("D", d), ci("S", streams), ci("BLOCK", 256) });
    }

    // -- attention -----------------------------------------------------------------------------------------

    pub const Prep = struct { p: u64, pos0: u64, qw: u64, kw: u64, iw: u64, inv: u64, q: u64, cache: Cache, iq: u64, ikc: u64 };

    /// glue.attn_prep on text positions (MODE 0: ROPE and DELTA are POS0, as Python passes them without images):
    /// q, k, indexer heads normalized and rotated, keys and values stored at POS0 + r, raw indexer keys.
    pub fn attnPrep(t: Tri, a: Prep, rows: usize, d: Dims) !void {
        const pw = d.attnWidth();
        try t.run("_attn_prep", .{ rows, d.heads + d.kv_heads + d.index_heads + 1, 1 }, &.{ p("P", bf16, a.p), p("POS0", i32p, a.pos0), p("QW", f32p, a.qw), p("KW", f32p, a.kw), p("IW", f32p, a.iw), p("INV", f32p, a.inv), p("Q", bf16, a.q), p("KC", a.cache.codes(), a.cache.k), p("VC", a.cache.codes(), a.cache.v), p("KS", f16p, a.cache.ks), p("VS", f16p, a.cache.vs), p("IQ", bf16, a.iq), p("IKC", bf16, a.ikc), p("ROPE", i32p, a.pos0), p("DELTA", i32p, a.pos0), int("length", 0), aot.float("eps", d.eps) }, &.{ ci("PW", pw), ci("NQ", d.heads), ci("NKV", d.kv_heads), ci("HD", d.head_dim), ci("NI", d.index_heads), ci("IHD", d.index_dim), ci("HALF", d.rotary_half), ci("BITS", a.cache.bits), ci("MODE", 0), ci("S1", 11), ci("S2", 10) });
    }

    /// attention.attention: rows' queries [R, H, D] over their 512-key chunks (sparse rows through the scratch's
    /// key lists), merged in chunk order into out [R, H, D]; `context` bounds the chunks (null: the scratch's).
    pub fn attention(t: Tri, q: u64, cache: Cache, pos0: u64, sc: AttnScratch, out: u64, rows: usize, g: AttnGeometry, context: ?usize, d: Dims) !void {
        const h = d.heads;
        const hk = d.kv_heads;
        const grp = h / hk;
        if (grp > 16) return error.TooManyQueryHeads;
        const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(d.head_dim), -0.5));
        const ty = cache.codes();
        try t.run("_chunks", .{ rows, hk, g.chunks(context) }, &.{ p("Q", bf16, q), p("KC", ty, cache.k), p("VC", ty, cache.v), p("KSC", f16p, cache.ks), p("VSC", f16p, cache.vs), p("POS0", i32p, pos0), p("PO", f32p, sc.po), p("PM", f32p, sc.pm), p("PL", f32p, sc.pl), p("IDS", i32p, sc.ids), p("NKR", i32p, sc.nk), p("SPR", i32p, sc.sparse) }, &.{ ci("H", h), ci("HK", hk), ci("D", d.head_dim), ci("G", grp), ci("CH", AttnGeometry.chunk), ci("NCH", g.nch), aot.cf("SCALE", scale), ci("IDW", g.idw), cb("QSA", g.qsa), ci("BITS", cache.bits) });
        try t.run("_merge", .{ rows, hk, 1 }, &.{ p("PO", f32p, sc.po), p("PM", f32p, sc.pm), p("PL", f32p, sc.pl), p("POS0", i32p, pos0), p("OUT", bf16, out), p("NKR", i32p, sc.nk), p("SPR", i32p, sc.sparse) }, &.{ ci("H", h), ci("HK", hk), ci("D", d.head_dim), ci("G", grp), ci("CH", AttnGeometry.chunk), ci("NCH", g.nch), cb("QSA", g.qsa), ci("BITS", cache.bits) });
    }

    /// attention.qsa_pool on text positions: the RATIO-key blocks rows [P0, P0 + R) complete, pooled, normalized
    /// (w = the indexer's k_layernorm, fp32) and rotated (ROPE and DELTA None, as Python passes them).
    pub fn qsaPool(t: Tri, ikc: u64, pooled: u64, pos0: u64, w: u64, inv: u64, rows: usize, g: AttnGeometry, d: Dims) !void {
        try t.run("_pool", .{ rows / g.ratio + 2, 1, 1 }, &.{ p("IKC", bf16, ikc), p("POOLED", bf16, pooled), p("POS0", i32p, pos0), p("W", f32p, w), p("INV", f32p, inv), aot.float("eps", d.eps), int("R", rows), int("length", 0) }, &.{ ci("DI", d.index_dim), ci("HALF", d.rotary_half), ci("RATIO", g.ratio), ci("MODE", 0), ci("S1", 11), ci("S2", 10) });
    }

    /// attention.qsa_rows: each sparse row's block scores, then its TOP blocks and tail (_select while the blocks
    /// fit its registers, _select_tiles past them) into the scratch's key lists.
    pub fn qsaRows(t: Tri, iq: u64, pooled: u64, pos0: u64, sc: AttnScratch, rows: usize, g: AttnGeometry, context: ?usize, d: Dims) !void {
        const top = g.budget / g.ratio;
        const blocks = g.blocks(context);
        try t.run("_scores", .{ rows, cdiv(blocks, 64), 1 }, &.{ p("IQ", bf16, iq), p("POOLED", bf16, pooled), p("POS0", i32p, pos0), p("SC", f32p, sc.scores), int("NB", g.nb) }, &.{ ci("HI", d.index_heads), ci("DI", d.index_dim), ci("RATIO", g.ratio), ci("TOP", top), ci("BB", 64) });
        const lists = [_]aot.Arg{ p("SC", f32p, sc.scores), p("POS0", i32p, pos0), p("IDS", i32p, sc.ids), p("NKR", i32p, sc.nk), p("SPR", i32p, sc.sparse), int("NB", g.nb) };
        const width = pow2(blocks);
        if (width <= AttnGeometry.select_regs) {
            try t.run("_select", .{ rows, 1, 1 }, &lists, &.{ ci("RATIO", g.ratio), ci("TOP", top), ci("IDW", g.idw), ci("BLOCK", width) });
        } else {
            try t.run("_select_tiles", .{ rows, 1, 1 }, &lists, &.{ ci("RATIO", g.ratio), ci("TOP", top), ci("IDW", g.idw), ci("TB", if (rows >= 64) 4096 else 8192) });
        }
    }

    /// attention.qsa_select (a decode window): pool the blocks the window completes, then score and select.
    pub fn qsaSelect(t: Tri, iq: u64, ikc: u64, pooled: u64, pos0: u64, w: u64, inv: u64, sc: AttnScratch, rows: usize, g: AttnGeometry, context: ?usize, d: Dims) !void {
        try t.qsaPool(ikc, pooled, pos0, w, inv, rows, g, d);
        try t.qsaRows(iq, pooled, pos0, sc, rows, g, context, d);
    }

    // -- a shared round's attention in one launch a kernel (attn_multi.py) ------------------------------------

    /// attn_multi.Step's tables: each row's position and stream, each stream's first position and row count, and
    /// a layer's pointer table [6][N] int64 (keys, values, key scales, value scales, index keys, pooled).
    pub const Multi = struct { posr: u64, sid: u64, first: u64, counts: u64, n: usize };

    /// Whether the kernel set holds attn_multi's kernels (aot/w6-ks1 and later).
    pub fn multiAvailable(set: *const aot.Set) bool {
        return set.smallestConst("_chunks_multi", "NCH", 0) != null and set.smallestConst("_prep_multi", "PW", 0) != null and
            set.smallestConst("_merge_multi", "NCH", 0) != null and set.smallestConst("_pool_multi", "DI", 0) != null;
    }

    /// attn_multi._prep_multi: every row's attn_prep through its stream's caches (text positions, bf16 caches).
    pub fn prepMulti(t: Tri, m: Multi, cp: u64, pa: u64, qw: u64, kw: u64, iw: u64, inv: u64, q: u64, iq: u64, rows: usize, d: Dims) !void {
        try t.run("_prep_multi", .{ rows, d.heads + d.kv_heads + d.index_heads + 1, 1 }, &.{ p("P", bf16, pa), p("POSR", i32p, m.posr), p("SID", i32p, m.sid), p("CP", "*i64", cp), p("VP", "*i64", cp), p("QW", f32p, qw), p("KW", f32p, kw), p("IW", f32p, iw), p("INV", f32p, inv), p("Q", bf16, q), p("IQ", bf16, iq), aot.float("eps", d.eps), int("N", m.n) }, &.{ ci("PW", d.attnWidth()), ci("NQ", d.heads), ci("NKV", d.kv_heads), ci("HD", d.head_dim), ci("NI", d.index_heads), ci("IHD", d.index_dim), ci("HALF", d.rotary_half), ci("BITS", 0), cb("VISION", false), ci("S1", 11), ci("S2", 10) });
    }

    /// attn_multi._pool_multi: each stream's RATIO-key blocks its rows complete, pooled (`most` the most rows).
    pub fn poolMulti(t: Tri, m: Multi, cp: u64, w: u64, inv: u64, most: usize, g: AttnGeometry, d: Dims) !void {
        try t.run("_pool_multi", .{ m.n, most / g.ratio + 2, 1 }, &.{ p("CP", "*i64", cp), p("VP", "*i64", cp), p("P0", i32p, m.first), p("RS", i32p, m.counts), p("W", f32p, w), p("INV", f32p, inv), aot.float("eps", d.eps), int("N", m.n) }, &.{ ci("DI", d.index_dim), ci("HALF", d.rotary_half), ci("RATIO", g.ratio), cb("VISION", false), ci("S1", 11), ci("S2", 10) });
    }

    /// attn_multi._chunks_multi + _merge_multi: every row over its stream's caches (a row is sparse when its end
    /// passes TOP blocks: its key list from qsaRows), merged into out [R, H, D]; `keys` the most a row has.
    pub fn attentionMulti(t: Tri, m: Multi, cp: u64, q: u64, sc: AttnScratch, out: u64, rows: usize, g: AttnGeometry, keys: usize, d: Dims) !void {
        const h = d.heads;
        const hk = d.kv_heads;
        const grp = h / hk;
        if (grp > 16) return error.TooManyQueryHeads;
        const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(d.head_dim), -0.5));
        const top = g.budget / g.ratio;
        try t.run("_chunks_multi", .{ rows, hk, g.chunks(keys) }, &.{ p("Q", bf16, q), p("CP", "*i64", cp), p("POSR", i32p, m.posr), p("SID", i32p, m.sid), p("PO", f32p, sc.po), p("PM", f32p, sc.pm), p("PL", f32p, sc.pl), p("IDS", i32p, sc.ids), p("NKR", i32p, sc.nk), int("N", m.n) }, &.{ ci("H", h), ci("HK", hk), ci("D", d.head_dim), ci("G", grp), ci("CH", AttnGeometry.chunk), ci("NCH", g.nch), aot.cf("SCALE", scale), ci("IDW", g.idw), cb("QSA", g.qsa), ci("BITS", 0), ci("RATIO", g.ratio), ci("TOP", top) });
        try t.run("_merge_multi", .{ rows, hk, 1 }, &.{ p("PO", f32p, sc.po), p("PM", f32p, sc.pm), p("PL", f32p, sc.pl), p("POSR", i32p, m.posr), p("OUT", bf16, out), p("NKR", i32p, sc.nk) }, &.{ ci("H", h), ci("HK", hk), ci("D", d.head_dim), ci("G", grp), ci("CH", AttnGeometry.chunk), ci("NCH", g.nch), cb("QSA", g.qsa), ci("BITS", 0), ci("RATIO", g.ratio), ci("TOP", top) });
    }

    /// glue.attn_gate: out [R, H*D] = bf16(o * sigmoid(gate)), the gate from pa's [q | gate] pairs, and group sums.
    pub fn attnGate(t: Tri, o: u64, pa: u64, out: u64, xs: u64, rows: usize, d: Dims) !void {
        try t.run("_attn_gate", .{ rows, d.heads, 1 }, &.{ p("O", bf16, o), p("P", bf16, pa), p("OUT", bf16, out), p("XS", f32p, xs) }, &.{ ci("PW", d.attnWidth()), ci("NQ", d.heads), ci("HD", d.head_dim) });
    }

    // -- the n-gram embedding (PLE) ------------------------------------------------------------------------

    /// glue.ple_embed_bf16: the gathered rows (row r * heads + h, dh bf16 values) -> out [R, heads * dh] and group
    /// sums; `scale` the table's weight_scale (1.0 for the checkpoint's FP8 table: its LUT holds the scale).
    pub fn pleEmbedBf16(t: Tri, v: u64, out: u64, xs: u64, rows: usize, heads: usize, dh: usize, scale: f32) !void {
        try t.run("_ple_embed_bf16", .{ rows, heads, 1 }, &.{ p("V", bf16, v), p("OUT", bf16, out), p("XS", f32p, xs) }, &.{ ci("HEADS", heads), ci("DH", dh), aot.cf("SCALE", scale) });
    }

    /// glue.ple_gate: each stream's gate from its normalized key and query, gated values [R, S*D] and their
    /// squared sums [R, S].
    pub fn pleGate(t: Tri, keys: u64, vals: u64, h: u64, norm_key: u64, norm_query: u64, gated: u64, pss: u64, rows: usize, d: usize, streams: usize, eps: f32) !void {
        try t.run("_ple_gate", .{ rows, 1, 1 }, &.{ p("KEYS", bf16, keys), p("VALS", bf16, vals), p("H", bf16, h), p("NK", f32p, norm_key), p("NQ", f32p, norm_query), p("GATED", bf16, gated), p("PSS", f32p, pss), aot.float("eps", eps) }, &.{ ci("D", d), ci("S", streams), ci("BLOCK", 512) });
    }

    /// glue.ple_conv (one stream's rows): the dilated conv over [tail; rows], SiLU, both residual adds into hout,
    /// and the normed rows kept in nrow for the next tail; conv_w [S*D, taps] bf16.
    pub fn pleConv(t: Tri, gated: u64, pss: u64, norm_conv: u64, tail: u64, conv_w: u64, h: u64, hout: u64, nrow: u64, rows: usize, d: usize, streams: usize, taps: usize, dilation: usize, eps: f32) !void {
        try t.run("_ple_conv", .{ rows, streams * d / 512, 1 }, &.{ p("GATED", bf16, gated), p("PSS", f32p, pss), p("NC", f32p, norm_conv), p("TAIL", bf16, tail), p("CW", bf16, conv_w), p("H", bf16, h), p("HOUT", bf16, hout), p("NROW", bf16, nrow), aot.float("eps", eps), int("R", rows) }, &.{ ci("D", d), ci("S", streams), ci("TAPS", taps), ci("DIL", dilation), ci("BLOCK", 512) });
    }

    // -- matmuls -------------------------------------------------------------------------------------------

    /// bf16.matmul: x [M, K] bf16 (rows `x_stride` apart) @ w [N, K].T -> out [M, N] (bf16, or fp32 sums with
    /// `f32`); K slices (b16SplitK) into `part` (b16PartBytes) are summed in slice order by `_reduce`.
    pub fn b16mm(t: Tri, x: u64, x_stride: usize, w: u64, out: u64, fp32: bool, part: u64, m: usize, n: usize, k: usize) !void {
        if (k % 64 != 0) return error.KNotBlocked;
        const sk = b16SplitK(n, k);
        const bm: usize = if (m > 128) 128 else 16;
        const oty = if (fp32) f32p else bf16;
        const split = sk > 1;
        try t.run("_b16mm", .{ cdiv(m, bm), cdiv(n, 64), sk }, &.{ p("X", bf16, x), p("W", bf16, w), p("OUT", oty, out), p("PART", if (split) f32p else oty, if (split) part else out), int("M", m), int("x_stride", x_stride) }, &.{ ci("N", n), ci("K", k), ci("SK", sk), ci("BM", bm), ci("BLOCK_N", 64), ci("BK", 64), cb("F32", fp32) });
        if (split) try t.reduce(part, out, fp32, m * n, sk);
    }

    /// bf16._reduce and nvfp4._reduce: `total` fp32 sums of `sk` slices in order, one rounding for bf16 out.
    fn reduce(t: Tri, part: u64, out: u64, fp32: bool, total: usize, sk: usize) !void {
        try t.run("_reduce", .{ cdiv(total, 1024), 1, 1 }, &.{ p("PART", f32p, part), p("OUT", if (fp32) f32p else bf16, out), int("total", total) }, &.{ ci("SK", sk), ci("BLOCK", 1024), cb("F32", fp32) });
    }

    /// nvfp4.matmul at its defaults (the shared expert: gate|up [2NI, D], down [D, NI] from rows `x_stride` apart):
    /// x [M, K] bf16 @ fp.T -> out [M, N] bf16 (fp32 with `f32`), K slices (fp4SplitK) summed in order.
    pub fn fp4mm(t: Tri, x: u64, x_stride: usize, fp: Fp4, out: u64, fp32: bool, part: u64, m: usize, n: usize, k: usize) !void {
        if (n % 64 != 0) return error.NNotTiled;
        const bm = bucket(m);
        var want: usize = switch (bm) {
            16 => 4,
            32, 64 => 2,
            else => 1,
        };
        if (bm == 16) if (fp4Tuned(n, k)) |tuned| {
            want = tuned.gpi;
        };
        const sk = fp4SplitK(n, k);
        const gpi = gpiFor((k / 16) / sk, want);
        const oty = if (fp32) f32p else bf16;
        const split = sk > 1;
        try t.run("_fp4mm", .{ cdiv(m, bm), n / 64, sk }, &.{ p("X", bf16, x), p("W", if (fp.codes) "*u8" else "*u16", fp.weight), p("S", if (fp.codes) "*u8" else f32p, fp.scale), p("S2", f32p, fp.scale2), p("OUT", oty, out), p("PART", if (split) f32p else oty, if (split) part else out), int("M", m), int("x_stride", x_stride) }, &.{ ci("N", n), ci("K", k), ci("SK", sk), ci("BM", bm), ci("SBN", 64), ci("BLOCK_N", 64), ci("GPI", gpi), cb("F32", fp32), cb("PACKED", fp.codes) });
        if (split) try t.reduce(part, out, fp32, m * n, sk);
    }

    // -- MoE -----------------------------------------------------------------------------------------------

    /// moe.router: logits [M, E + 1] fp32 = x [M, D] . rows (the router's E rows, then the shared expert's gate).
    pub fn router(t: Tri, x: u64, x_stride: usize, rows_w: u64, out: u64, m: usize, d: usize, ne: usize) !void {
        const bm = bucket(m);
        const be: usize, const bk: usize = if (bm == 16) .{ 32, 256 } else .{ 64, 64 };
        try t.run("_router", .{ cdiv(m, bm), cdiv(ne, be), 1 }, &.{ p("X", bf16, x), p("W", bf16, rows_w), p("OUT", f32p, out), int("M", m), int("x_stride", x_stride) }, &.{ ci("D", d), ci("NE", ne), ci("BM", bm), ci("BLOCK_E", be), ci("BK", bk) });
    }

    /// moe.select_rows: each row's top_k experts and weights, then the shared expert as slot top_k (pick, wts
    /// [R, top_k + 1]); the grouping into a plan is the experts extension's.
    pub fn topkRows(t: Tri, logits: u64, pick: u64, wts: u64, rows: usize, experts: usize, top_k: usize) !void {
        try t.run("_topk_rows", .{ rows, 1, 1 }, &.{ p("L", f32p, logits), p("PICK", i32p, pick), p("WTS", f32p, wts) }, &.{ ci("NE", experts), ci("NL", experts + 1), ci("TOPK", top_k), ci("SLOTS", top_k + 1), ci("BLOCK", pow2(experts + 1)), ci("SLOTP", pow2(top_k + 1)) });
    }

    /// glue.moe_partial (TP=2): a rank's fp32 MoE share sum_k w_k y_k + w_s y_s [R, D], gathered for mode 3.
    pub fn moePartial(t: Tri, y: u64, y_f32: bool, wts: u64, out: u64, rows: usize, d: usize, slots: usize) !void {
        try t.run("_moe_partial", .{ rows, d / 256, 1 }, &.{ p("Y", if (y_f32) f32p else bf16, y), p("WTS", f32p, wts), p("OUT", f32p, out) }, &.{ ci("D", d), ci("TOPK", slots - 1), ci("SLOTS", slots), ci("BLOCK", 256) });
    }

    // -- commit --------------------------------------------------------------------------------------------

    /// forward.shift_windows: old [L, T, C] (in place, layers `old_l` elements apart) takes rows keep .. keep + T
    /// of [old; new rows], new rows `new_row` apart within a layer and `new_l` across layers (the first C columns).
    pub fn shiftWindows(t: Tri, old: u64, new: u64, keep: usize, old_l: usize, new_l: usize, new_row: usize, layers: usize, channels: usize, taps: usize) !void {
        try t.run("_shift_windows", .{ layers, cdiv(channels, 256), 1 }, &.{ p("OLD", bf16, old), p("NEW", bf16, new), int("keep", keep), int("OLD_L", old_l), int("NEW_L", new_l), int("NEW_ROW", new_row) }, &.{ ci("C", channels), ci("T", taps), ci("TP", pow2(taps)), ci("BLOCK", 256) });
    }
};

// -- tests: the Python wrappers' own launches (fixtures_cuda_triton.json) -------------------------------------

const testing = std.testing;
const fixture_text = @embedFile("fixtures_cuda_triton.json");

const Fixture = struct {
    parsed: std.json.Parsed(std.json.Value),

    fn load() !Fixture {
        return .{ .parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, fixture_text, .{}) };
    }
    fn deinit(f: *Fixture) void {
        f.parsed.deinit();
    }
    fn case(f: *const Fixture, id: []const u8) ![]const std.json.Value {
        const cases = f.parsed.value.object.get("cases").?.object;
        const c = cases.get(id) orelse {
            std.debug.print("no fixture case {s}\n", .{id});
            return error.MissingCase;
        };
        return c.array.items;
    }
};

fn jsonInt(v: std.json.Value) i64 {
    return v.integer;
}

/// The recorded launches equal the fixture's: kernel, grid, each runtime argument's type and value by name, and
/// every non-None constexpr (None ones are not runtime arguments and Zig passes nothing for them).
fn expectLaunches(f: *const Fixture, id: []const u8, rec: *const Recorder) !void {
    const want = try f.case(id);
    errdefer std.debug.print("case {s}\n", .{id});
    try testing.expectEqual(want.len, rec.n);
    for (want, rec.launches[0..rec.n]) |w, got| {
        const o = w.object;
        try testing.expectEqualStrings(o.get("fn").?.string, got.name);
        const grid = o.get("grid").?.array.items;
        for (grid, got.grid) |g, x| try testing.expectEqual(jsonInt(g), @as(i64, x));
        const args = o.get("args").?.array.items;
        try testing.expectEqual(args.len, got.nargs);
        for (args) |arg| {
            const name = arg.object.get("name").?.string;
            const ty = arg.object.get("type").?.string;
            const mine = for (got.args[0..got.nargs]) |x| {
                if (std.mem.eql(u8, x.name, name)) break x;
            } else {
                std.debug.print("{s}: no argument {s}\n", .{ got.name, name });
                return error.MissingArgument;
            };
            errdefer std.debug.print("{s}: argument {s}\n", .{ got.name, name });
            switch (mine.value) {
                .ptr => |x| try testing.expectEqualStrings(ty, x.ty),
                .i32 => |x| {
                    try testing.expectEqualStrings(ty, "i32");
                    try testing.expectEqual(jsonInt(arg.object.get("int").?), @as(i64, x));
                },
                .f32 => |x| {
                    try testing.expectEqualStrings(ty, "fp32");
                    try testing.expectEqual(jsonInt(arg.object.get("f32").?), @as(i64, @as(u32, @bitCast(x))));
                },
                .u64 => return error.UnexpectedWord,
            }
        }
        const consts = o.get("consts").?.object;
        var live: usize = 0;
        var it = consts.iterator();
        while (it.next()) |kv| {
            if (kv.value_ptr.* == .null) continue;
            live += 1;
            const name = kv.key_ptr.*;
            const mine = for (got.consts[0..got.nconsts]) |c| {
                if (std.mem.eql(u8, c.name, name)) break c;
            } else {
                std.debug.print("{s}: no constexpr {s}\n", .{ got.name, name });
                return error.MissingConst;
            };
            errdefer std.debug.print("{s}: constexpr {s}\n", .{ got.name, name });
            const v = kv.value_ptr.object;
            if (v.get("int")) |x| try testing.expectEqual(jsonInt(x), mine.int.?);
            if (v.get("f32")) |x| try testing.expectEqual(jsonInt(x), @as(i64, @as(u32, @bitCast(mine.f32.?))));
        }
        try testing.expectEqual(live, got.nconsts);
    }
}

/// Distinct 16-aligned stand-in addresses (types, not values, select a variant).
fn addr(i: u64) u64 {
    return 0x7f0000000000 + i * 0x100000;
}

fn check(f: *const Fixture, comptime fmt: []const u8, args: anytype, run: anytype) !void {
    var buf: [128]u8 = undefined;
    const id = try std.fmt.bufPrint(&buf, fmt, args);
    var rec: Recorder = .{};
    const t: Tri = .{ .set = null, .rec = &rec };
    try run.go(t);
    try expectLaunches(f, id, &rec);
}

/// tools/zig/flashnext_triton_fixtures.py ROWS: decode windows, the capture's prompt pieces and full chunks
const rows_cases = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 16, 18, 19, 29, 30, 116, 973, 974, 2048 };
const prompts = [_][2]usize{ .{ 0, 2048 }, .{ 4096, 19 }, .{ 139264, 2048 }, .{ 2048, 974 }, .{ 2048, 973 }, .{ 0, 30 }, .{ 0, 29 }, .{ 0, 18 }, .{ 2048, 206 }, .{ 2048, 205 } };

test "embedding, hyper-connection, norm and stream launches equal the Python wrappers'" {
    var f = try Fixture.load();
    defer f.deinit();
    const d: Dims = .{};
    const D = d.hidden;
    const S = d.streams;
    const W = d.wide();
    for (rows_cases) |r| {
        const R = struct {
            r: usize,
            d: Dims,
            copies: usize = 4,
            mode: usize = 0,
            y_f32: bool = false,
            ndn: usize = 0,
            group: usize = 0,
            fn go(c: @This(), t: Tri) !void {
                const dd = c.d;
                switch (c.mode) {
                    0 => try t.hcWriteback(addr(0), addr(0), addr(1), null, .none, c.r, dd.hidden, dd.streams),
                    1 => try t.hcWriteback(addr(0), addr(0), addr(1), addr(2), .{ .bf16 = addr(3) }, c.r, dd.hidden, dd.streams),
                    2 => try t.hcWriteback(addr(0), addr(0), addr(1), addr(2), .{ .moe = .{ .y = addr(3), .y_f32 = c.y_f32, .wts = addr(4), .slots = dd.top_k + 1 } }, c.r, dd.hidden, dd.streams),
                    3 => try t.hcWriteback(addr(0), addr(0), addr(1), addr(2), .{ .ranks = .{ .part = addr(3), .world = 2 } }, c.r, dd.hidden, dd.streams),
                    else => unreachable,
                }
            }
        };
        try check(&f, "hc_writeback/r{d}/m0", .{r}, R{ .r = r, .d = d });
        try check(&f, "hc_writeback/r{d}/m1", .{r}, R{ .r = r, .d = d, .mode = 1 });
        try check(&f, "hc_writeback/r{d}/m2/bf16", .{r}, R{ .r = r, .d = d, .mode = 2 });
        try check(&f, "hc_writeback/r{d}/m2/f32", .{r}, R{ .r = r, .d = d, .mode = 2, .y_f32 = true });
        try check(&f, "hc_writeback/r{d}/m3", .{r}, R{ .r = r, .d = d, .mode = 3 });
        inline for (.{ 4, 1 }) |copies| {
            try check(&f, "embed/r{d}/s{d}", .{ r, copies }, struct {
                r: usize,
                fn go(c: @This(), t: Tri) !void {
                    try t.embed(addr(0), addr(1), addr(2), c.r, 2560, copies);
                }
            }{ .r = r });
        }
        const Plain = struct {
            r: usize,
            d: Dims,
            which: enum { normed, act_inj, act, mix, norm_e, norm_h, add, ple_embed, ple_gate, ple_conv },
            fn go(c: @This(), t: Tri) !void {
                const dd = c.d;
                const wide = dd.wide();
                switch (c.which) {
                    .normed => try t.hcNormed(addr(0), addr(1), addr(2), addr(3), addr(4), c.r, dd.hidden, dd.streams, dd.eps),
                    .act_inj => try t.hcAct(addr(0), addr(1), addr(2), addr(3), c.r, dd.low + dd.streams, dd.streams, dd.low),
                    .act => try t.hcAct(addr(0), addr(1), addr(2), null, c.r, dd.low, dd.streams, dd.low),
                    .mix => try t.hcMix(addr(0), addr(1), addr(2), addr(3), c.r, dd.hidden, dd.streams),
                    .norm_e => try t.rmsnorm(addr(0), dd.hidden, addr(1), addr(2), addr(3), c.r, dd.hidden, null, dd.eps),
                    .norm_h => try t.rmsnorm(addr(0), wide, addr(1), addr(2), addr(3), c.r, wide, null, dd.eps),
                    .add => try t.addStreams(addr(0), addr(1), addr(2), c.r, dd.hidden, dd.streams),
                    .ple_embed => try t.pleEmbedBf16(addr(0), addr(1), addr(2), c.r, dd.ngram_heads, dd.ngram_dim, 1.0),
                    .ple_gate => try t.pleGate(addr(0), addr(1), addr(2), addr(3), addr(4), addr(5), addr(6), c.r, dd.hidden, dd.streams, dd.eps),
                    .ple_conv => try t.pleConv(addr(0), addr(1), addr(2), addr(3), addr(4), addr(5), addr(5), addr(6), c.r, dd.hidden, dd.streams, dd.ple_taps, dd.ngram_size, dd.eps),
                }
            }
        };
        try check(&f, "hc_normed/r{d}", .{r}, Plain{ .r = r, .d = d, .which = .normed });
        try check(&f, "hc_act/r{d}/n{d}", .{ r, d.low + S }, Plain{ .r = r, .d = d, .which = .act_inj });
        try check(&f, "hc_act/r{d}/n{d}", .{ r, d.low }, Plain{ .r = r, .d = d, .which = .act });
        try check(&f, "hc_mix/r{d}", .{r}, Plain{ .r = r, .d = d, .which = .mix });
        try check(&f, "rmsnorm/r{d}/d{d}", .{ r, D }, Plain{ .r = r, .d = d, .which = .norm_e });
        try check(&f, "rmsnorm/r{d}/d{d}", .{ r, W }, Plain{ .r = r, .d = d, .which = .norm_h });
        try check(&f, "add_streams/r{d}", .{r}, Plain{ .r = r, .d = d, .which = .add });
        try check(&f, "ple_embed_bf16/r{d}", .{r}, Plain{ .r = r, .d = d, .which = .ple_embed });
        try check(&f, "ple_embed_bf16/r{d}/tp2", .{r}, struct {
            r: usize,
            fn go(c: @This(), t: Tri) !void {
                try t.pleEmbedBf16(addr(0), addr(1), addr(2), c.r, 8, 160, 1.0);
            }
        }{ .r = r });
        try check(&f, "ple_gate/r{d}", .{r}, Plain{ .r = r, .d = d, .which = .ple_gate });
        try check(&f, "ple_conv/r{d}", .{r}, Plain{ .r = r, .d = d, .which = .ple_conv });
    }
}

test "MoE launches equal the Python wrappers': router, top-k, the shared expert's FP4 matmuls, TP=2 partials" {
    var f = try Fixture.load();
    defer f.deinit();
    for (rows_cases) |r| {
        try check(&f, "moe/r{d}", .{r}, struct {
            r: usize,
            fn go(c: @This(), t: Tri) !void {
                const d: Dims = .{};
                try t.router(addr(0), d.hidden, addr(1), addr(2), c.r, d.hidden, d.experts + 1);
                try t.topkRows(addr(2), addr(3), addr(4), c.r, d.experts, d.top_k);
            }
        }{ .r = r });
        inline for (.{ false, true }) |fp32| {
            try check(&f, "moe_partial/r{d}/{s}", .{ r, if (fp32) "f32" else "bf16" }, struct {
                r: usize,
                fn go(c: @This(), t: Tri) !void {
                    try t.moePartial(addr(0), fp32, addr(1), addr(2), c.r, 2560, 11);
                }
            }{ .r = r });
        }
        for ([_]usize{ 1, 2 }) |world| {
            const d = Dims.rank(world);
            const Gu = struct {
                r: usize,
                d: Dims,
                fn go(c: @This(), t: Tri) !void {
                    const n = 2 * c.d.moe_width;
                    try t.fp4mm(addr(0), c.d.hidden, .{ .weight = addr(1), .scale = addr(2), .scale2 = addr(3) }, addr(4), false, addr(5), c.r, n, c.d.hidden);
                }
            };
            try check(&f, "fp4/gu/r{d}/tp{d}", .{ r, world }, Gu{ .r = r, .d = d });
            inline for (.{ false, true }) |fp32| {
                try check(&f, "fp4/down/r{d}/tp{d}/{s}", .{ r, world, if (fp32) "f32" else "bf16" }, struct {
                    r: usize,
                    d: Dims,
                    fn go(c: @This(), t: Tri) !void {
                        const slot_stride = (c.d.top_k + 1) * c.d.moe_width;
                        try t.fp4mm(addr(0), slot_stride, .{ .weight = addr(1), .scale = addr(2), .scale2 = addr(3) }, addr(4), fp32, addr(5), c.r, c.d.hidden, c.d.moe_width);
                    }
                }{ .r = r, .d = d });
            }
        }
    }
}

test "bf16 matmul launches equal bf16.matmul's for every linear of the checkpoint, TP=1 and a TP=2 rank" {
    var f = try Fixture.load();
    defer f.deinit();
    const Mat = struct { name: []const u8, n: usize, k: usize, fp32: bool, replicated: bool };
    for ([_]usize{ 1, 2 }) |world| {
        const d = Dims.rank(world);
        const mats = [_]Mat{
            .{ .name = "hc_down", .n = d.low + d.streams, .k = d.wide(), .fp32 = true, .replicated = true },
            .{ .name = "mix_down", .n = d.low, .k = d.wide(), .fp32 = true, .replicated = true },
            .{ .name = "hc_up", .n = d.wide(), .k = d.low, .fp32 = false, .replicated = true },
            .{ .name = "gdn_proj", .n = d.projWidth(), .k = d.hidden, .fp32 = false, .replicated = false },
            .{ .name = "gdn_out", .n = d.hidden, .k = d.value_heads * d.gdn_dim, .fp32 = world > 1, .replicated = false },
            .{ .name = "attn_proj", .n = d.attnWidth(), .k = d.hidden, .fp32 = false, .replicated = false },
            .{ .name = "o_proj", .n = d.hidden, .k = d.heads * d.head_dim, .fp32 = world > 1, .replicated = false },
            .{ .name = "ple_key", .n = d.wide(), .k = d.hidden, .fp32 = false, .replicated = true },
            .{ .name = "ple_value", .n = d.hidden, .k = d.hidden, .fp32 = false, .replicated = true },
            .{ .name = "fc", .n = d.hidden, .k = d.hidden, .fp32 = false, .replicated = true },
            .{ .name = "head", .n = d.vocab, .k = d.hidden, .fp32 = false, .replicated = false },
        };
        for (rows_cases) |r| for (mats) |m| {
            if (world > 1 and m.replicated) continue;
            if (std.mem.eql(u8, m.name, "head") and r > 16) continue;
            try check(&f, "b16/{s}/r{d}/tp{d}", .{ m.name, r, world }, struct {
                r: usize,
                m: Mat,
                fn go(c: @This(), t: Tri) !void {
                    try t.b16mm(addr(0), c.m.k, addr(1), addr(2), c.m.fp32, addr(3), c.r, c.m.n, c.m.k);
                }
            }{ .r = r, .m = m });
        };
    }
    for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 18, 973, 2048 }) |n| {
        try check(&f, "b16/fc_h/r{d}", .{n * 4}, struct {
            r: usize,
            fn go(c: @This(), t: Tri) !void {
                try t.b16mm(addr(0), 2560, addr(1), addr(2), false, addr(3), c.r, 2560, 2560);
            }
        }{ .r = n * 4 });
    }
}

test "attention launches equal the Python wrappers': prep, gate, decode windows and prompt blocks" {
    var f = try Fixture.load();
    defer f.deinit();
    const sc: AttnScratch = .{ .po = addr(20), .pm = addr(21), .pl = addr(22), .ids = addr(23), .nk = addr(24), .sparse = addr(25), .scores = addr(26) };
    const cache: Cache = .{ .k = addr(10), .v = addr(11), .ks = addr(12), .vs = addr(12) };
    for ([_]usize{ 1, 2 }) |world| {
        const d = Dims.rank(world);
        for (rows_cases) |r| {
            try check(&f, "attn_prep/r{d}/tp{d}", .{ r, world }, struct {
                r: usize,
                d: Dims,
                fn go(c: @This(), t: Tri) !void {
                    try t.attnPrep(.{ .p = addr(0), .pos0 = addr(1), .qw = addr(2), .kw = addr(3), .iw = addr(4), .inv = addr(5), .q = addr(6), .cache = .{ .k = addr(10), .v = addr(11), .ks = addr(12), .vs = addr(13) }, .iq = addr(7), .ikc = addr(8) }, c.r, c.d);
                }
            }{ .r = r, .d = d });
            try check(&f, "attn_gate/r{d}/tp{d}", .{ r, world }, struct {
                r: usize,
                d: Dims,
                fn go(c: @This(), t: Tri) !void {
                    try t.attnGate(addr(0), addr(1), addr(2), addr(3), c.r, c.d);
                }
            }{ .r = r, .d = d });
        }
        const Decode = struct {
            rows: usize,
            pos: usize,
            ctx: ?usize,
            g: AttnGeometry,
            d: Dims,
            sc: AttnScratch,
            cache: Cache,
            fn go(c: @This(), t: Tri) !void {
                const keys = c.ctx orelse c.pos + c.rows;
                if (c.g.qsa) try t.qsaSelect(addr(30), addr(31), addr(32), addr(1), addr(33), addr(34), c.sc, c.rows, c.g, keys, c.d);
                try t.attention(addr(0), c.cache, addr(1), c.sc, addr(2), c.rows, c.g, keys, c.d);
            }
        };
        const Prefill = struct {
            start: usize,
            n: usize,
            g: AttnGeometry,
            d: Dims,
            sc: AttnScratch,
            cache: Cache,
            fn go(c: @This(), t: Tri) !void {
                if (c.g.qsa) try t.qsaPool(addr(31), addr(32), addr(1), addr(33), addr(34), c.n, c.g, c.d);
                var r0: usize = 0;
                while (r0 < c.n) : (r0 += 256) {
                    const m = @min(256, c.n - r0);
                    const ends = c.start + r0 + m;
                    if (c.g.qsa) try t.qsaRows(addr(30), addr(32), addr(1), c.sc, m, c.g, ends, c.d);
                    try t.attention(addr(0), c.cache, addr(1), c.sc, addr(2), m, c.g, ends, c.d);
                }
            }
        };
        for ([_]usize{ 1024, 262144, 262151, 1048576 }) |capacity| {
            const g = AttnGeometry.init(capacity, 2048, 4);
            for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 16 }) |rows| for ([_]usize{ 0, 100, 3000, 140000 }) |pos| for ([_]usize{ 0, 8192, 16384 }) |ctx| {
                if (pos + rows > capacity) continue;
                if (ctx != 0 and (ctx < pos + rows or ctx > capacity)) continue;
                try check(&f, "attention/tp{d}/c{d}/r{d}/p{d}/x{d}", .{ world, capacity, rows, pos, ctx }, Decode{ .rows = rows, .pos = pos, .ctx = if (ctx == 0) null else ctx, .g = g, .d = d, .sc = sc, .cache = cache });
            };
            for (prompts) |sn| {
                if (sn[0] + sn[1] > capacity) continue;
                try check(&f, "attention_prefill/tp{d}/c{d}/s{d}/n{d}", .{ world, capacity, sn[0], sn[1] }, Prefill{ .start = sn[0], .n = sn[1], .g = g, .d = d, .sc = sc, .cache = cache });
            }
        }
    }
}

test "commit launches equal forward.shift_windows': DeltaNet conv windows and the n-gram tail" {
    var f = try Fixture.load();
    defer f.deinit();
    const Shift = struct {
        keep: usize,
        old_l: usize,
        new_l: usize,
        new_row: usize,
        layers: usize,
        c: usize,
        taps: usize,
        fn go(s: @This(), t: Tri) !void {
            try t.shiftWindows(addr(0), addr(1), s.keep, s.old_l, s.new_l, s.new_row, s.layers, s.c, s.taps);
        }
    };
    for ([_]usize{ 1, 2 }) |world| {
        const d = Dims.rank(world);
        const conv = d.convDim();
        const proj = d.projWidth();
        for ([_][3]usize{ .{ 8, 1, 1 }, .{ 8, 7, 3 }, .{ 16, 16, 16 }, .{ 16, 5, 1 } }) |c| {
            try check(&f, "shift/conv/tp{d}/b{d}/r{d}/k{d}", .{ world, c[0], c[1], c[2] }, Shift{ .keep = c[2], .old_l = 3 * conv, .new_l = c[0] * proj, .new_row = proj, .layers = 36, .c = conv, .taps = 3 });
        }
        for ([_]usize{ 2048, 974, 30, 19, 18, 1 }) |n| {
            try check(&f, "shift/prefill/tp{d}/n{d}", .{ world, n }, Shift{ .keep = n, .old_l = 3 * conv, .new_l = 2048 * proj, .new_row = proj, .layers = 1, .c = conv, .taps = 3 });
        }
    }
    const d: Dims = .{};
    for ([_][3]usize{ .{ 8, 1, 1 }, .{ 8, 7, 4 }, .{ 2048, 2048, 2048 }, .{ 2048, 974, 974 }, .{ 2048, 30, 30 }, .{ 2048, 19, 19 }, .{ 2048, 18, 18 } }) |c| {
        try check(&f, "shift/ple/b{d}/r{d}/k{d}", .{ c[0], c[1], c[2] }, Shift{ .keep = c[2], .old_l = d.tailRows() * d.wide(), .new_l = c[0] * d.wide(), .new_row = d.wide(), .layers = 1, .c = d.wide(), .taps = d.tailRows() });
    }
}

test "split K and scratch sizes follow the Python shapes" {
    try testing.expectEqual(@as(usize, 32), b16SplitK(324, 10240));
    try testing.expectEqual(@as(usize, 1), b16SplitK(248320, 2560));
    try testing.expectEqual(@as(usize, 4), b16SplitK(2560, 6144));
    try testing.expectEqual(@as(usize, 2), b16SplitK(8240, 2560));
    try testing.expectEqual(@as(usize, 8), fp4SplitK(2560, 640));
    try testing.expectEqual(@as(usize, 4), fp4SplitK(2560, 320));
    try testing.expectEqual(@as(usize, 32 * 7 * 324 * 4), b16PartBytes(7, 324, 10240));
    const g = AttnGeometry.init(262144, 2048, 4);
    try testing.expectEqual(@as(usize, 5), g.nch);
    try testing.expectEqual(@as(usize, 65536), g.nb);
    try testing.expectEqual(@as(usize, 1), g.chunks(100));
    try testing.expectEqual(@as(usize, 5), g.chunks(null));
    try testing.expectEqual(@as(usize, 13952), (Dims{}).attnWidth());
    try testing.expectEqual(@as(usize, 7296), Dims.rank(2).attnWidth());
    try testing.expectEqual(@as(usize, 16480), (Dims{}).projWidth());
}
