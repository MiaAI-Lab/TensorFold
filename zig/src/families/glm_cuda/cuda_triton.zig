//! Triton replay for glm5_next/cuda/{glue,qmm,latent,sparse,forward,dflash2,dflash2_multi}.py.
//! Real capture signatures live in cuda_triton_generated.zig; launch methods preserve the Python ABI.
const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;
pub const Dims = @import("cuda_geometry.zig").Geometry;
fn u(n: usize) u32 {
    return @intCast(n);
}
pub fn pow2(n: usize) usize {
    return std.math.ceilPowerOfTwo(usize, @max(n, 1)) catch unreachable;
}
pub const Recorder = struct {
    launches: [64]Saved = undefined,
    n: usize = 0,
    pub const Saved = struct { name: []const u8, grid: [3]u32, args: [64]aot.Arg = undefined, nargs: usize, consts: [48]aot.Const = undefined, nconsts: usize };
    fn add(r: *Recorder, name: []const u8, grid: [3]u32, args: []const aot.Arg, consts: []const aot.Const) !void {
        if (r.n == r.launches.len or args.len > 64 or consts.len > 48) return error.TooManyLaunches;
        const s = &r.launches[r.n];
        s.* = .{ .name = name, .grid = grid, .nargs = args.len, .nconsts = consts.len };
        @memcpy(s.args[0..args.len], args);
        @memcpy(s.consts[0..consts.len], consts);
        r.n += 1;
    }
};
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

pub const captured_variants = @import("cuda_triton_generated.zig").variants;

pub const Tri = struct {
    /// Named typed wrappers, generated from the real Python launch signatures.
    pub const launch = @import("cuda_triton_generated.zig").Launches(Tri);
    set: ?*const aot.Set = null,
    s: cuda.Stream = undefined,
    rec: ?*Recorder = null,
    memo: ?*Memo = null,
    pub fn run(t: Tri, name: []const u8, grid: [3]usize, args: []const aot.Arg, consts: []const aot.Const) !void {
        const g: [3]u32 = .{ u(grid[0]), u(grid[1]), u(grid[2]) };
        if (t.rec) |r| return r.add(name, g, args, consts);
        const set = t.set orelse return error.MissingTritonSet;
        const m = t.memo orelse return set.run(t.s, name, g, args, consts);
        const k = Memo.key(name, args, consts);
        const at = m.map.get(k) orelse blk: {
            const v = set.find(name, args, consts) catch |e| {
                // TF_GLM_TRI_LENIENT=1 (bring-up): report each missing variant once and skip the launch, to list every
                // variant a run needs in one go. The results are garbage.
                if (e == error.MissingTritonVariant and std.c.getenv("TF_GLM_TRI_LENIENT") != null) {
                    try m.map.put(m.gpa, k, std.math.maxInt(usize));
                    return;
                }
                return e;
            };
            const i = (@intFromPtr(v) - @intFromPtr(set.variants.ptr)) / @sizeOf(@TypeOf(set.variants[0]));
            try m.map.put(m.gpa, k, i);
            break :blk i;
        };
        if (at == std.math.maxInt(usize)) return;
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

    /// glue.embed's BF16 branch, including the original pointer types and constexpr names.
    pub fn embed(t: Tri, ids: u64, table: u64, out: u64, rows: usize, copies: usize) !void {
        try t.run("_embed_b16", .{ rows, 64, 1 }, &.{ aot.ptr("IDS", "*i32", ids), aot.ptr("W", "*bf16", table), aot.ptr("OUT", "*bf16", out) }, &.{ aot.ci("D", 4096), aot.ci("COPIES", @intCast(copies)) });
    }
    /// glue.py:_hc_finish; argument names/types and constexprs are the reference ABI.
    pub fn hc_finish(t: Tri, grid: [3]usize, X: u64, PART: u64, BASE: u64, SCALE: u64, NW: u64, OUT: u64, XS: u64, POST: u64, COMB: u64, eps_norm: f32, hc_eps: f32, c: struct { D: i64, S: i64, NB: i64, ITERS: i64, BLOCK: i64 }) !void {
        try t.run("_hc_finish", grid, &.{ aot.ptr("X", "*bf16", X), aot.ptr("PART", "*fp32", PART), aot.ptr("BASE", "*fp32", BASE), aot.ptr("SCALE", "*fp32", SCALE), aot.ptr("NW", "*bf16", NW), aot.ptr("OUT", "*bf16", OUT), aot.ptr("XS", "*fp32", XS), aot.ptr("POST", "*fp32", POST), aot.ptr("COMB", "*fp32", COMB), aot.float("eps_norm", eps_norm), aot.float("hc_eps", hc_eps) }, &.{ aot.ci("D", c.D), aot.ci("S", c.S), aot.ci("NB", c.NB), aot.ci("ITERS", c.ITERS), aot.ci("BLOCK", c.BLOCK) });
    }
    /// glue.py:_hc_post; argument names/types and constexprs are the reference ABI.
    pub fn hc_post(t: Tri, grid: [3]usize, X: u64, XOUT: u64, G: u64, POST: u64, COMB: u64, RS: i32, c: struct { D: i64, S: i64, WORLD: i64, BLOCK: i64, ROUND: i64 }) !void {
        try t.run("_hc_post", grid, &.{ aot.ptr("X", "*bf16", X), aot.ptr("XOUT", "*bf16", XOUT), aot.ptr("G", "*fp32", G), aot.ptr("POST", "*fp32", POST), aot.ptr("COMB", "*fp32", COMB), aot.int("RS", RS) }, &.{ aot.ci("D", c.D), aot.ci("S", c.S), aot.ci("WORLD", c.WORLD), aot.ci("BLOCK", c.BLOCK), aot.ci("ROUND", c.ROUND) });
    }
    /// glue.py:_stream_mean; argument names/types and constexprs are the reference ABI.
    pub fn stream_mean(t: Tri, grid: [3]usize, X: u64, OUT: u64, c: struct { D: i64, S: i64, BLOCK: i64 }) !void {
        try t.run("_stream_mean", grid, &.{ aot.ptr("X", "*bf16", X), aot.ptr("OUT", "*bf16", OUT) }, &.{ aot.ci("D", c.D), aot.ci("S", c.S), aot.ci("BLOCK", c.BLOCK) });
    }
    /// glue.py:_swiglu; argument names/types and constexprs are the reference ABI.
    pub fn swiglu(t: Tri, grid: [3]usize, GU: u64, OUT: u64, XS: u64, LIMIT: f32, c: struct { W: i64, BLOCK: i64 }) !void {
        try t.run("_swiglu", grid, &.{ aot.ptr("GU", "*bf16", GU), aot.ptr("OUT", "*bf16", OUT), aot.ptr("XS", "*fp32", XS), aot.float("LIMIT", LIMIT) }, &.{ aot.ci("W", c.W), aot.ci("BLOCK", c.BLOCK) });
    }
    /// glue.py:_router_part; argument names/types and constexprs are the reference ABI.
    pub fn router_part(t: Tri, grid: [3]usize, X: u64, W: u64, PART: u64, M: i32, x_stride: i32, c: struct { D: i64, NE: i64, BM: i64, BLOCK_E: i64, BK: i64, KS: i64 }) !void {
        try t.run("_router_part", grid, &.{ aot.ptr("X", "*bf16", X), aot.ptr("W", "*bf16", W), aot.ptr("PART", "*fp32", PART), aot.int("M", M), aot.int("x_stride", x_stride) }, &.{ aot.ci("D", c.D), aot.ci("NE", c.NE), aot.ci("BM", c.BM), aot.ci("BLOCK_E", c.BLOCK_E), aot.ci("BK", c.BK), aot.ci("KS", c.KS) });
    }
    /// glue.py:_router_sum; argument names/types and constexprs are the reference ABI.
    pub fn router_sum(t: Tri, grid: [3]usize, PART: u64, OUT: u64, total: i32, c: struct { KS: i64, BLOCK: i64 }) !void {
        try t.run("_router_sum", grid, &.{ aot.ptr("PART", "*fp32", PART), aot.ptr("OUT", "*fp32", OUT), aot.int("total", total) }, &.{ aot.ci("KS", c.KS), aot.ci("BLOCK", c.BLOCK) });
    }
    /// glue.py:_topk; argument names/types and constexprs are the reference ABI.
    pub fn topk(t: Tri, grid: [3]usize, L: u64, BIAS: u64, PICK: u64, WTS: u64, scale: f32, c: struct { NE: i64, TOPK: i64, SLOTS: i64, BLOCK: i64, SLOTP: i64, NORM: i64 }) !void {
        try t.run("_topk", grid, &.{ aot.ptr("L", "*fp32", L), aot.ptr("BIAS", "*fp32", BIAS), aot.ptr("PICK", "*i32", PICK), aot.ptr("WTS", "*fp32", WTS), aot.float("scale", scale) }, &.{ aot.ci("NE", c.NE), aot.ci("TOPK", c.TOPK), aot.ci("SLOTS", c.SLOTS), aot.ci("BLOCK", c.BLOCK), aot.ci("SLOTP", c.SLOTP), aot.ci("NORM", c.NORM) });
    }
    /// glue.py:_combine_sy; argument names/types and constexprs are the reference ABI.
    pub fn combine_sy(t: Tri, grid: [3]usize, Y: u64, SY: u64, WTS: u64, OUT: u64, c: struct { D: i64, SLOTS: i64, BLOCK: i64 }) !void {
        try t.run("_combine_sy", grid, &.{ aot.ptr("Y", "*fp32", Y), aot.ptr("SY", "*fp32", SY), aot.ptr("WTS", "*fp32", WTS), aot.ptr("OUT", "*fp32", OUT) }, &.{ aot.ci("D", c.D), aot.ci("SLOTS", c.SLOTS), aot.ci("BLOCK", c.BLOCK) });
    }
    /// glue.py:_rmsnorm; argument names/types and constexprs are the reference ABI.
    pub fn rmsnorm(t: Tri, grid: [3]usize, X: u64, x_stride: i32, W: u64, OUT: u64, o_stride: i32, XS: u64, eps: f32, c: struct { D: i64, BLOCK: i64, SUMS: i64 }) !void {
        try t.run("_rmsnorm", grid, &.{ aot.ptr("X", "*bf16", X), aot.int("x_stride", x_stride), aot.ptr("W", "*bf16", W), aot.ptr("OUT", "*bf16", OUT), aot.int("o_stride", o_stride), aot.ptr("XS", "*fp32", XS), aot.float("eps", eps) }, &.{ aot.ci("D", c.D), aot.ci("BLOCK", c.BLOCK), aot.ci("SUMS", c.SUMS) });
    }
    /// qmm.py:_group_sums; argument names/types and constexprs are the reference ABI.
    pub fn group_sums(t: Tri, grid: [3]usize, X: u64, XS: u64, x_stride: i32, c: struct { K: i64, GB: i64 }) !void {
        try t.run("_group_sums", grid, &.{ aot.ptr("X", "*bf16", X), aot.ptr("XS", "*fp32", XS), aot.int("x_stride", x_stride) }, &.{ aot.ci("K", c.K), aot.ci("GB", c.GB) });
    }
    /// qmm.py:_fmm; argument names/types and constexprs are the reference ABI.
    pub fn fmm(t: Tri, grid: [3]usize, X: u64, W: u64, S: u64, OUT: u64, PART: u64, M: i32, x_stride: i32, c: struct { N: i64, K: i64, SK: i64, BM: i64, BLOCK_N: i64, BK: i64, F32: i64, SEQ: i64 }) !void {
        try t.run("_fmm", grid, &.{ aot.ptr("X", "*bf16", X), aot.ptr("W", "*fp8e4nv", W), aot.ptr("S", "*fp32", S), aot.ptr("OUT", "*bf16", OUT), aot.ptr("PART", "*bf16", PART), aot.int("x_stride", x_stride), aot.int("M", M) }, &.{ aot.ci("N", c.N), aot.ci("K", c.K), aot.ci("SK", c.SK), aot.ci("BM", c.BM), aot.ci("BLOCK_N", c.BLOCK_N), aot.ci("BK", c.BK), aot.ci("F32", c.F32), aot.ci("SEQ", c.SEQ) });
    }
    /// latent.py:_seg_lat_write; argument names/types and constexprs are the reference ABI.
    pub fn seg_lat_write(t: Tri, grid: [3]usize, LAT: u64, lat_stride: i32, LC: u64, LS: u64, POS: u64, BASE: u64, c: struct { LW: i64, RS: i64, FP8: i64 }) !void {
        try t.run("_seg_lat_write", grid, &.{ aot.ptr("LAT", "*bf16", LAT), aot.int("lat_stride", lat_stride), aot.ptr("LC", "*fp8e4nv", LC), aot.ptr("LS", "*fp32", LS), aot.ptr("POS", "*i32", POS), aot.ptr("BASE", "*i32", BASE) }, &.{ aot.ci("LW", c.LW), aot.ci("RS", c.RS), aot.ci("FP8", c.FP8) });
    }
    /// latent.py:_absorb_q; argument names/types and constexprs are the reference ABI.
    pub fn absorb_q(t: Tri, grid: [3]usize, Q: u64, WK: u64, QA: u64, R: i32, c: struct { H: i64, D: i64, LW: i64, BN: i64 }) !void {
        try t.run("_absorb_q", grid, &.{ aot.ptr("Q", "*bf16", Q), aot.ptr("WK", "*bf16", WK), aot.ptr("QA", "*bf16", QA), aot.int("R", R) }, &.{ aot.ci("H", c.H), aot.ci("D", c.D), aot.ci("LW", c.LW), aot.ci("BN", c.BN) });
    }
    /// latent.py:_expand_v; argument names/types and constexprs are the reference ABI.
    pub fn expand_v(t: Tri, grid: [3]usize, OL: u64, WV: u64, OUT: u64, R: i32, c: struct { H: i64, DV: i64, LW: i64, BN: i64 }) !void {
        try t.run("_expand_v", grid, &.{ aot.ptr("OL", "*bf16", OL), aot.ptr("WV", "*bf16", WV), aot.ptr("OUT", "*bf16", OUT), aot.int("R", R) }, &.{ aot.ci("H", c.H), aot.ci("DV", c.DV), aot.ci("LW", c.LW), aot.ci("BN", c.BN) });
    }
    /// latent.py:_seg_chunks; argument names/types and constexprs are the reference ABI.
    pub fn seg_chunks(t: Tri, grid: [3]usize, QA: u64, LC: u64, LS: u64, POS: u64, BASE: u64, SPR: u64, TOK: u64, CNT: u64, PO: u64, PM: u64, PL: u64, R: i32, c: struct { W: i64, H: i64, LW: i64, CH: i64, SCALE: f32, HBT: i64, KTT: i64, RS: i64, FP8: i64 }) !void {
        try t.run("_seg_chunks", grid, &.{ aot.ptr("QA", "*bf16", QA), aot.ptr("LC", "*fp8e4nv", LC), aot.ptr("LS", "*fp32", LS), aot.ptr("POS", "*i32", POS), aot.ptr("BASE", "*i32", BASE), aot.ptr("SPR", "*i32", SPR), aot.ptr("TOK", "*i32", TOK), aot.ptr("CNT", "*i32", CNT), aot.ptr("PO", "*fp32", PO), aot.ptr("PM", "*fp32", PM), aot.ptr("PL", "*fp32", PL), aot.int("R", R) }, &.{ aot.ci("W", c.W), aot.ci("H", c.H), aot.ci("LW", c.LW), aot.ci("CH", c.CH), aot.cf("SCALE", c.SCALE), aot.ci("HBT", c.HBT), aot.ci("KTT", c.KTT), aot.ci("RS", c.RS), aot.ci("FP8", c.FP8) });
    }
    /// latent.py:_merge; argument names/types and constexprs are the reference ABI.
    pub fn merge(t: Tri, grid: [3]usize, PO: u64, PM: u64, PL: u64, OUT: u64, CNT: u64, R: i32, c: struct { H: i64, LW: i64, NCH: i64, SPARSE: i64 }) !void {
        try t.run("_merge", grid, &.{ aot.ptr("PO", "*fp32", PO), aot.ptr("PM", "*fp32", PM), aot.ptr("PL", "*fp32", PL), aot.ptr("OUT", "*bf16", OUT), aot.ptr("CNT", "*i32", CNT), aot.int("R", R) }, &.{ aot.ci("H", c.H), aot.ci("LW", c.LW), aot.ci("NCH", c.NCH), aot.ci("SPARSE", c.SPARSE) });
    }
    /// sparse.py:_seg_index_write; argument names/types and constexprs are the reference ABI.
    pub fn seg_index_write(t: Tri, grid: [3]usize, KR: u64, k_stride: i32, GR: u64, LNW: u64, LNB: u64, IK: u64, IG: u64, POS: u64, IBASE: u64, RING: i32, eps: f32, c: struct { D: i64 }) !void {
        try t.run("_seg_index_write", grid, &.{ aot.ptr("KR", "*bf16", KR), aot.int("k_stride", k_stride), aot.ptr("GR", "*fp32", GR), aot.ptr("LNW", "*bf16", LNW), aot.ptr("LNB", "*bf16", LNB), aot.ptr("IK", "*bf16", IK), aot.ptr("IG", "*bf16", IG), aot.ptr("POS", "*i32", POS), aot.ptr("IBASE", "*i32", IBASE), aot.int("RING", RING), aot.float("eps", eps) }, &.{aot.ci("D", c.D)});
    }
    /// sparse.py:_seg_pool_keys; argument names/types and constexprs are the reference ABI.
    pub fn seg_pool_keys(t: Tri, grid: [3]usize, IK: u64, IG: u64, APE: u64, PK: u64, PKS: u64, POS: u64, BASE: u64, IBASE: u64, RING: i32, c: struct { D: i64, RS: i64, FP8: i64 }) !void {
        try t.run("_seg_pool_keys", grid, &.{ aot.ptr("IK", "*bf16", IK), aot.ptr("IG", "*bf16", IG), aot.ptr("APE", "*bf16", APE), aot.ptr("PK", "*fp8e4nv", PK), aot.ptr("PKS", "*fp32", PKS), aot.ptr("POS", "*i32", POS), aot.ptr("BASE", "*i32", BASE), aot.ptr("IBASE", "*i32", IBASE), aot.int("RING", RING) }, &.{ aot.ci("D", c.D), aot.ci("RS", c.RS), aot.ci("FP8", c.FP8) });
    }
    /// sparse.py:_seg_scores; argument names/types and constexprs are the reference ABI.
    pub fn seg_scores(t: Tri, grid: [3]usize, QI: u64, W: u64, w_stride: i32, PK: u64, PKS: u64, OUT: u64, o_stride: i32, POS: u64, SPR: u64, SEG_START: u64, SEG_ROWS: u64, SEG_PBASE: u64, SEG_POOLS: u64, scale: f32, wscale: f32, c: struct { H: i64, HP: i64, D: i64, BP: i64, G: i64, RS: i64, FP8: i64 }) !void {
        try t.run("_seg_scores", grid, &.{ aot.ptr("QI", "*bf16", QI), aot.ptr("W", "*bf16", W), aot.int("w_stride", w_stride), aot.ptr("PK", "*fp8e4nv", PK), aot.ptr("PKS", "*fp32", PKS), aot.ptr("OUT", "*fp32", OUT), aot.int("o_stride", o_stride), aot.ptr("POS", "*i32", POS), aot.ptr("SPR", "*i32", SPR), aot.ptr("SEG_START", "*i32", SEG_START), aot.ptr("SEG_ROWS", "*i32", SEG_ROWS), aot.ptr("SEG_PBASE", "*i32", SEG_PBASE), aot.ptr("SEG_POOLS", "*i32", SEG_POOLS), aot.float("scale", scale), aot.float("wscale", wscale) }, &.{ aot.ci("H", c.H), aot.ci("HP", c.HP), aot.ci("D", c.D), aot.ci("BP", c.BP), aot.ci("G", c.G), aot.ci("RS", c.RS), aot.ci("FP8", c.FP8) });
    }
    /// sparse.py:_select_split; argument names/types and constexprs are the reference ABI.
    /// forward.py:_conv_shift_seg. P1 supplies the cubin. No host arithmetic stands in for this launch.
    pub fn conv_shift_seg(t: Tri, grid: [3]usize, CONV: u64, PROJ: u64, SEG: u64, conv_slot: i32, conv_layer: i32, proj_layer: i32, proj_row: i32, c: struct { C: i64, TAPS: i64, COLS: i64, BLOCK: i64 }) !void {
        try t.run("_conv_shift_seg", grid, &.{ aot.ptr("CONV", "*bf16", CONV), aot.ptr("PROJ", "*bf16", PROJ), aot.ptr("SEG", "*i32", SEG), aot.int("conv_slot", conv_slot), aot.int("conv_layer", conv_layer), aot.int("proj_layer", proj_layer), aot.int("proj_row", proj_row) }, &.{ aot.ci("C", c.C), aot.ci("TAPS", c.TAPS), aot.ci("COLS", c.COLS), aot.ci("BLOCK", c.BLOCK) });
    }
    pub fn select_split(t: Tri, grid: [3]usize, S: u64, s_stride: i32, NPC: i32, POS: u64, SPR: u64, TOT: u64, HIST: u64, GT: u64, OUT: u64, TOK: u64, CNT: u64, c: struct { W: i64, K: i64, PL: i64, CHS: i64, CP: i64, STEP: i64, SEG: i64, VIS: i64, TOKS: i64, BLK: i64 }) !void {
        try t.run("_select_split", grid, &.{ aot.ptr("S", "*fp32", S), aot.int("s_stride", s_stride), aot.int("NPC", NPC), aot.ptr("POS", "*i32", POS), aot.ptr("SPR", "*i32", SPR), aot.ptr("TOT", "*i32", TOT), aot.ptr("HIST", "*i32", HIST), aot.ptr("GT", "*i32", GT), aot.ptr("OUT", "*i32", OUT), aot.ptr("TOK", "*i32", TOK), aot.ptr("CNT", "*i32", CNT) }, &.{ aot.ci("W", c.W), aot.ci("K", c.K), aot.ci("PL", c.PL), aot.ci("CHS", c.CHS), aot.ci("CP", c.CP), aot.ci("STEP", c.STEP), aot.ci("SEG", c.SEG), aot.ci("VIS", c.VIS), aot.ci("TOKS", c.TOKS), aot.ci("BLK", c.BLK) });
    }
    /// glm5_next/cuda/glue.py:_rmsnorm; BF16 rows, f32 group sums.
    pub fn draftRmsnorm(t: Tri, x: u64, stride: usize, weight: u64, out: u64, xs: u64, rows: usize, d: usize, eps: f32) !void {
        try t.run("_rmsnorm", .{ rows, 1, 1 }, &.{ aot.ptr("X", "*bf16", x), aot.int("x_stride", @intCast(stride)), aot.ptr("W", "*bf16", weight), aot.ptr("OUT", "*bf16", out), aot.int("o_stride", @intCast(d)), aot.ptr("XS", "*fp32", xs), aot.float("eps", eps) }, &.{ aot.ci("D", @intCast(d)), aot.ci("BLOCK", @intCast(pow2(d))), aot.ci("SUMS", 1) });
    }
    pub fn groupSums(t: Tri, x: u64, xs: u64, rows: usize, k: usize) !void {
        try t.run("_group_sums", .{ rows, (k / 64 + 15) / 16, 1 }, &.{ aot.ptr("X", "*bf16", x), aot.ptr("XS", "*fp32", xs), aot.int("x_stride", @intCast(k)) }, &.{ aot.ci("K", @intCast(k)), aot.ci("GB", 16) });
    }
    pub fn draftSwiglu(t: Tri, gu: u64, out: u64, xs: u64, rows: usize, width: usize) !void {
        const block = std.math.gcd(@as(usize, 512), width);
        if (block < 64) return error.BadDraftWidth;
        try t.run("_swiglu", .{ rows, width / block, 1 }, &.{ aot.ptr("GU", "*bf16", gu), aot.ptr("OUT", "*bf16", out), aot.ptr("XS", "*fp32", xs), aot.float("LIMIT", 1e30) }, &.{ aot.ci("W", @intCast(width)), aot.ci("BLOCK", @intCast(block)) });
    }
    /// dflash2.py:53; SEG prevents convolution across streams.
    pub fn draftConv(t: Tri, x: u64, dyn: u64, base: u64, residual: ?u64, out: u64, rows: usize, branch: usize, segment: usize) !void {
        if (branch > 1) return error.BadDraftBranch;
        try t.run("_dconv_kernel", .{ rows, 4, 1 }, &.{ aot.ptr("X", "*bf16", x), aot.ptr("DYN", "*bf16", dyn), aot.ptr("BASE", "*bf16", base), aot.ptr("RES", "*bf16", residual orelse x), aot.ptr("OUT", "*bf16", out) }, &.{ aot.ci("D", 4096), aot.ci("G", 256), aot.ci("GS", 16), aot.ci("BRANCH", @intCast(branch)), aot.ci("HAS_RES", @intFromBool(residual != null)), aot.ci("BLOCK", 1024), aot.ci("SEG", @intCast(segment)) });
    }
    /// dflash2.py:82. H=0 for committed target taps, local query heads for blocks.
    pub fn draftPrep(t: Tri, qkv: u64, qn: u64, kn: u64, cos: u64, sin: u64, qo: u64, ko: u64, vo: u64, rows: usize, heads: usize, kv: usize, eps: f32) !void {
        try t.run("_prep_kernel", .{ rows, heads + 2 * kv, 1 }, &.{ aot.ptr("QKV", "*bf16", qkv), aot.ptr("QN", "*bf16", qn), aot.ptr("KN", "*bf16", kn), aot.ptr("COS", "*fp32", cos), aot.ptr("SIN", "*fp32", sin), aot.ptr("QO", "*bf16", qo), aot.ptr("KO", "*bf16", ko), aot.ptr("VO", "*bf16", vo), aot.int("L", @intCast(rows)), aot.int("stride", @intCast((heads + 2 * kv) * 128)), aot.float("eps", eps) }, &.{ aot.ci("H", @intCast(heads)), aot.ci("HKV", @intCast(kv)), aot.ci("HALF", 64) });
    }
    /// dflash2_multi.py:40. POS/SLOT are int64 stream metadata, not target KV.
    pub fn draftAttention(t: Tri, q: u64, k: u64, v: u64, out: u64, pos: u64, slots: u64, streams: usize, heads: usize, kv: usize, slot_rows: usize) !void {
        if (kv == 0 or heads % kv != 0) return error.BadDraftHeads;
        try t.run("_dattn_seg_kernel", .{ kv, streams, 1 }, &.{ aot.ptr("Q", "*bf16", q), aot.ptr("K", "*bf16", k), aot.ptr("V", "*bf16", v), aot.ptr("OUT", "*bf16", out), aot.ptr("POS", "*i64", pos), aot.ptr("SLOT", "*i64", slots), aot.int("window", 2047), aot.float("scale", 0.08838834764831845), aot.int("L", @intCast(streams * 8)), aot.int("SLOT_ROWS", @intCast(slot_rows)) }, &.{ aot.ci("N", 8), aot.ci("G", @intCast(heads / kv)), aot.ci("NH", @intCast(heads)), aot.ci("HD", 128), aot.ci("CAP", 2176), aot.ci("BK", 64), aot.ci("CAUSAL", 0), aot.ci("RING", 1) });
    }
    /// glue.py:_hc_partial_mma. Prompt rows only; decode stays on hc.cu.
    pub fn hc_partial_mma(t: Tri, grid: [3]usize, X: u64, FN: u64, PART: u64, R: i32, c: struct { WIDE: i64, NB: i64, BM: i64, BK: i64 }) !void {
        try t.run("_hc_partial_mma", grid, &.{ aot.ptr("X", "*bf16", X), aot.ptr("FN", "*bf16", FN), aot.ptr("PART", "*fp32", PART), aot.int("R", R) }, &.{ aot.ci("WIDE", c.WIDE), aot.ci("NB", c.NB), aot.ci("BM", c.BM), aot.ci("BK", c.BK) });
    }
    /// latent.py:_lat_write. LC and LS are the two views of one FP8 row.
    pub fn lat_write(t: Tri, grid: [3]usize, LAT: u64, lat_stride: i32, LC: u64, LS: u64, POS: u64, c: struct { LW: i64, RS: i64, FP8: i64 }) !void {
        try t.run("_lat_write", grid, &.{ aot.ptr("LAT", "*bf16", LAT), aot.int("lat_stride", lat_stride), aot.ptr("LC", "*fp8e4nv", LC), aot.ptr("LS", "*fp32", LS), aot.ptr("POS", "*i32", POS) }, &.{ aot.ci("LW", c.LW), aot.ci("RS", c.RS), aot.ci("FP8", c.FP8) });
    }
    /// latent.py:_absorb_mma. R is runtime; the tile is constexpr.
    pub fn absorb_mma(t: Tri, grid: [3]usize, Q: u64, WK: u64, QA: u64, R: i32, c: struct { H: i64, D: i64, LW: i64, BM: i64, BN: i64, BK: i64 }) !void {
        try t.run("_absorb_mma", grid, &.{ aot.ptr("Q", "*bf16", Q), aot.ptr("WK", "*bf16", WK), aot.ptr("QA", "*bf16", QA), aot.int("R", R) }, &.{ aot.ci("H", c.H), aot.ci("D", c.D), aot.ci("LW", c.LW), aot.ci("BM", c.BM), aot.ci("BN", c.BN), aot.ci("BK", c.BK) });
    }
    /// latent.py:_expand_mma.
    pub fn expand_mma(t: Tri, grid: [3]usize, OL: u64, WV: u64, OUT: u64, R: i32, c: struct { H: i64, DV: i64, LW: i64, BM: i64, BN: i64, BK: i64 }) !void {
        try t.run("_expand_mma", grid, &.{ aot.ptr("OL", "*bf16", OL), aot.ptr("WV", "*bf16", WV), aot.ptr("OUT", "*bf16", OUT), aot.int("R", R) }, &.{ aot.ci("H", c.H), aot.ci("DV", c.DV), aot.ci("LW", c.LW), aot.ci("BM", c.BM), aot.ci("BN", c.BN), aot.ci("BK", c.BK) });
    }
    /// latent.py:_dense_chunks. SCALE is the f32 constexpr 0.0625.
    pub fn dense_chunks(t: Tri, grid: [3]usize, QA: u64, LC: u64, LS: u64, POS: u64, PO: u64, PM: u64, PL: u64, R: i32, c: struct { H: i64, LW: i64, CH: i64, SCALE: f32, HBT: i64, KTT: i64, RS: i64, FP8: i64 }) !void {
        try t.run("_dense_chunks", grid, &.{ aot.ptr("QA", "*bf16", QA), aot.ptr("LC", "*fp8e4nv", LC), aot.ptr("LS", "*fp32", LS), aot.ptr("POS", "*i32", POS), aot.ptr("PO", "*fp32", PO), aot.ptr("PM", "*fp32", PM), aot.ptr("PL", "*fp32", PL), aot.int("R", R) }, &.{ aot.ci("H", c.H), aot.ci("LW", c.LW), aot.ci("CH", c.CH), aot.cf("SCALE", c.SCALE), aot.ci("HBT", c.HBT), aot.ci("KTT", c.KTT), aot.ci("RS", c.RS), aot.ci("FP8", c.FP8) });
    }
    /// sparse.py:_index_write. eps is the prompt value 1e-6; the decode wrapper keeps its own call.
    pub fn index_write(t: Tri, grid: [3]usize, KR: u64, k_stride: i32, GR: u64, LNW: u64, LNB: u64, IK: u64, IG: u64, POS: u64, RING: i32, eps: f32, c: struct { D: i64 }) !void {
        try t.run("_index_write", grid, &.{ aot.ptr("KR", "*bf16", KR), aot.int("k_stride", k_stride), aot.ptr("GR", "*fp32", GR), aot.ptr("LNW", "*bf16", LNW), aot.ptr("LNB", "*bf16", LNB), aot.ptr("IK", "*bf16", IK), aot.ptr("IG", "*bf16", IG), aot.ptr("POS", "*i32", POS), aot.int("RING", RING), aot.float("eps", eps) }, &.{aot.ci("D", c.D)});
    }
    /// sparse.py:_pool_keys. R and RING are runtime.
    pub fn pool_keys(t: Tri, grid: [3]usize, IK: u64, IG: u64, APE: u64, PK: u64, PKS: u64, POS: u64, R: i32, RING: i32, c: struct { D: i64, RS: i64, FP8: i64 }) !void {
        try t.run("_pool_keys", grid, &.{ aot.ptr("IK", "*bf16", IK), aot.ptr("IG", "*bf16", IG), aot.ptr("APE", "*bf16", APE), aot.ptr("PK", "*fp8e4nv", PK), aot.ptr("PKS", "*fp32", PKS), aot.ptr("POS", "*i32", POS), aot.int("RING", RING), aot.int("R", R) }, &.{ aot.ci("D", c.D), aot.ci("RS", c.RS), aot.ci("FP8", c.FP8) });
    }
    /// sparse.py:_scores. OUT rows are packed NP columns apart, not the bucket width.
    pub fn scores(t: Tri, grid: [3]usize, QI: u64, W: u64, w_stride: i32, PK: u64, PKS: u64, OUT: u64, POS: u64, R: i32, NP: i32, scale: f32, wscale: f32, c: struct { H: i64, HP: i64, D: i64, BP: i64, L: i64, RS: i64, FP8: i64 }) !void {
        try t.run("_scores", grid, &.{ aot.ptr("QI", "*bf16", QI), aot.ptr("W", "*bf16", W), aot.int("w_stride", w_stride), aot.ptr("PK", "*fp8e4nv", PK), aot.ptr("PKS", "*fp32", PKS), aot.ptr("OUT", "*fp32", OUT), aot.ptr("POS", "*i32", POS), aot.int("R", R), aot.int("NP", NP), aot.float("scale", scale), aot.float("wscale", wscale) }, &.{ aot.ci("H", c.H), aot.ci("HP", c.HP), aot.ci("D", c.D), aot.ci("BP", c.BP), aot.ci("L", c.L), aot.ci("RS", c.RS), aot.ci("FP8", c.FP8) });
    }
    /// sparse.py:_scores_rows. SEG is false for a single-stream tail shorter than 64 rows.
    pub fn scores_rows(t: Tri, grid: [3]usize, QI: u64, W: u64, w_stride: i32, PK: u64, PKS: u64, OUT: u64, o_stride: i32, POS: u64, SPR: u64, BASE: u64, R: i32, NP: i32, scale: f32, wscale: f32, c: struct { H: i64, HP: i64, D: i64, BP: i64, G: i64, SEG: i64, RS: i64, FP8: i64, K: i64 }) !void {
        try t.run("_scores_rows", grid, &.{ aot.ptr("QI", "*bf16", QI), aot.ptr("W", "*bf16", W), aot.int("w_stride", w_stride), aot.ptr("PK", "*fp8e4nv", PK), aot.ptr("PKS", "*fp32", PKS), aot.ptr("OUT", "*fp32", OUT), aot.int("o_stride", o_stride), aot.ptr("POS", "*i32", POS), aot.ptr("SPR", "*i32", SPR), aot.ptr("BASE", "*i32", BASE), aot.int("R", R), aot.int("NP", NP), aot.float("scale", scale), aot.float("wscale", wscale) }, &.{ aot.ci("H", c.H), aot.ci("HP", c.HP), aot.ci("D", c.D), aot.ci("BP", c.BP), aot.ci("G", c.G), aot.ci("SEG", c.SEG), aot.ci("RS", c.RS), aot.ci("FP8", c.FP8), aot.ci("K", c.K) });
    }
    /// sparse.py:_select_rows. OUT is int64. VIS 0 passes the scores pointer, so POS is *fp32 in that cubin.
    pub fn select_rows(t: Tri, grid: [3]usize, S: u64, OUT: u64, NPS: i32, POS: u64, pos_ty: []const u8, c: struct { K: i64, BLOCK: i64, VIS: i64 }) !void {
        try t.run("_select_rows", grid, &.{ aot.ptr("S", "*fp32", S), aot.ptr("OUT", "*i64", OUT), aot.int("NPS", NPS), aot.ptr("POS", pos_ty, POS) }, &.{ aot.ci("K", c.K), aot.ci("BLOCK", c.BLOCK), aot.ci("VIS", c.VIS) });
    }
    /// sparse.py:_select_split when it returns pool ids. OUT is int64; the decode wrapper's OUT stays *i32.
    pub fn select_split_pools(t: Tri, grid: [3]usize, S: u64, s_stride: i32, NPC: i32, POS: u64, SPR: u64, TOT: u64, HIST: u64, GT: u64, OUT: u64, TOK: u64, CNT: u64, c: struct { W: i64, K: i64, PL: i64, CHS: i64, CP: i64, STEP: i64, SEG: i64, VIS: i64, TOKS: i64, BLK: i64 }) !void {
        try t.run("_select_split", grid, &.{ aot.ptr("S", "*fp32", S), aot.int("s_stride", s_stride), aot.int("NPC", NPC), aot.ptr("POS", "*i32", POS), aot.ptr("SPR", "*i32", SPR), aot.ptr("TOT", "*i32", TOT), aot.ptr("HIST", "*i32", HIST), aot.ptr("GT", "*i32", GT), aot.ptr("OUT", "*i64", OUT), aot.ptr("TOK", "*i32", TOK), aot.ptr("CNT", "*i32", CNT) }, &.{ aot.ci("W", c.W), aot.ci("K", c.K), aot.ci("PL", c.PL), aot.ci("CHS", c.CHS), aot.ci("CP", c.CP), aot.ci("STEP", c.STEP), aot.ci("SEG", c.SEG), aot.ci("VIS", c.VIS), aot.ci("TOKS", c.TOKS), aot.ci("BLK", c.BLK) });
    }
    /// sparse.py:_tokens. POOLS is the int64 [n, 512] tensor _select_rows wrote.
    pub fn tokens(t: Tri, grid: [3]usize, POOLS: u64, POS: u64, TOK: u64, CNT: u64, c: struct { W: i64, K: i64, PL: i64, BLOCK: i64 }) !void {
        try t.run("_tokens", grid, &.{ aot.ptr("POOLS", "*i64", POOLS), aot.ptr("POS", "*i32", POS), aot.ptr("TOK", "*i32", TOK), aot.ptr("CNT", "*i32", CNT) }, &.{ aot.ci("W", c.W), aot.ci("K", c.K), aot.ci("PL", c.PL), aot.ci("BLOCK", c.BLOCK) });
    }
    /// latent.py:_sparse_onepass. PREFETCH is the constexpr 0 or 1.
    pub fn sparse_onepass(t: Tri, grid: [3]usize, QA: u64, LC: u64, LS: u64, TOK: u64, CNT: u64, OUT: u64, c: struct { W: i64, H: i64, LW: i64, SCALE: f32, HBT: i64, KTT: i64, SPLIT: i64, STAGES: i64, PREFETCH: i64, RS: i64, FP8: i64 }) !void {
        try t.run("_sparse_onepass", grid, &.{ aot.ptr("QA", "*bf16", QA), aot.ptr("LC", "*fp8e4nv", LC), aot.ptr("LS", "*fp32", LS), aot.ptr("TOK", "*i32", TOK), aot.ptr("CNT", "*i32", CNT), aot.ptr("OUT", "*bf16", OUT) }, &.{ aot.ci("W", c.W), aot.ci("H", c.H), aot.ci("LW", c.LW), aot.cf("SCALE", c.SCALE), aot.ci("HBT", c.HBT), aot.ci("KTT", c.KTT), aot.ci("SPLIT", c.SPLIT), aot.ci("STAGES", c.STAGES), aot.ci("PREFETCH", c.PREFETCH), aot.ci("RS", c.RS), aot.ci("FP8", c.FP8) });
    }
    /// forward.py:_conv_shift. One KDA layer is a grid of (1, ceil(C/1024)).
    pub fn conv_shift(t: Tri, grid: [3]usize, CONV: u64, PROJ: u64, keep: i32, conv_layer: i32, proj_layer: i32, proj_row: i32, c: struct { C: i64, TAPS: i64, BLOCK: i64 }) !void {
        try t.run("_conv_shift", grid, &.{ aot.ptr("CONV", "*bf16", CONV), aot.ptr("PROJ", "*bf16", PROJ), aot.int("conv_layer", conv_layer), aot.int("proj_layer", proj_layer), aot.int("proj_row", proj_row), aot.int("keep", keep) }, &.{ aot.ci("C", c.C), aot.ci("TAPS", c.TAPS), aot.ci("BLOCK", c.BLOCK) });
    }
};
/// SPEC section 7: these stay Triton and belong to P1; none are approximated with CUDA arithmetic.
pub const pending_kernels = [_][]const u8{
    "_rmsnorm",        "_hc_partial",       "_hc_partial_mma", "_hc_finish",       "_hc_post",        "_stream_mean",   "_swiglu",
    "_router",         "_router_part",      "_router_sum",     "_topk",            "_combine",        "_combine_sy",    "_group_sums",
    "_reduce",         "_bmm",              "_fmm",            "_conv_shift",      "_conv_shift_seg", "_lat_write",     "_seg_lat_write",
    "_absorb_q",       "_expand_v",         "_absorb_mma",     "_expand_mma",      "_dense_chunks",   "_seg_chunks",    "_sparse_chunks",
    "_sparse_onepass", "_merge",            "_index_write",    "_seg_index_write", "_pool_keys",      "_seg_pool_keys", "_scores",
    "_scores_rows",    "_seg_scores",       "_select_rows",    "_select_split",    "_seg_select",     "_tokens",        "_dconv_kernel",
    "_prep_kernel",    "_dattn_seg_kernel",
};
test "Tri records GLM embedding signature without a CUDA driver" {
    var r: Recorder = .{};
    const t: Tri = .{ .rec = &r };
    try t.embed(0x1000, 0x2000, 0x3000, 7, 4);
    try std.testing.expectEqualStrings("_embed_b16", r.launches[0].name);
    try std.testing.expectEqual(@as([3]u32, .{ 7, 64, 1 }), r.launches[0].grid);
    try std.testing.expectEqual(@as(?i64, 4), r.launches[0].consts[1].int);
    try std.testing.expectError(error.MissingTritonSet, (Tri{}).embed(0, 0, 0, 1, 4));
}
test "memo keys track pointer alignment and scalar specialization" {
    const a = &.{ aot.ptr("X", "*bf16", 16), aot.int("R", 16) };
    const b = &.{ aot.ptr("X", "*bf16", 32), aot.int("R", 32) };
    try std.testing.expectEqual(Memo.key("k", a, &.{}), Memo.key("k", b, &.{}));
    try std.testing.expect(Memo.key("k", a, &.{}) != Memo.key("k", &.{ aot.ptr("X", "*bf16", 17), aot.int("R", 16) }, &.{}));
}

test "GLM glue and FP8 cache replay signatures preserve the reference views" {
    var rec: Recorder = .{};
    const tri: Tri = .{ .rec = &rec };
    try tri.hc_post(.{ 3, 4, 1 }, 16, 16, 32, 48, 64, 3 * 4096, .{ .D = 4096, .S = 4, .WORLD = 2, .BLOCK = 1024, .ROUND = 0 });
    try tri.seg_lat_write(.{ 3, 1, 1 }, 16, 512, 32, 32, 48, 64, .{ .LW = 512, .RS = 528, .FP8 = 1 });
    try tri.seg_chunks(.{ 3, 2, 5 }, 16, 32, 32, 48, 64, 80, 96, 112, 128, 144, 160, 3, .{ .W = 2051, .H = 32, .LW = 512, .CH = 512, .SCALE = 0.0625, .HBT = 16, .KTT = 32, .RS = 528, .FP8 = 1 });
    try std.testing.expectEqualStrings("_hc_post", rec.launches[0].name);
    try std.testing.expectEqual(@as(i32, 3 * 4096), rec.launches[0].args[5].value.i32);
    try std.testing.expectEqualStrings("*fp8e4nv", rec.launches[1].args[2].value.ptr.ty);
    try std.testing.expectEqual(rec.launches[1].args[2].value.ptr.addr, rec.launches[1].args[3].value.ptr.addr);
    try std.testing.expectEqual(@as(?f32, 0.0625), rec.launches[2].consts[4].f32);
}

test "conv shift segments keep the Triton ABI and fail closed without a set" {
    var rec: Recorder = .{};
    const tri: Tri = .{ .rec = &rec };
    try tri.conv_shift_seg(.{ 34, 12, 2 }, 16, 32, 48, 100, 200, 300, 400, .{ .C = 12288, .TAPS = 3, .COLS = 6, .BLOCK = 1024 });
    try std.testing.expectEqualStrings("_conv_shift_seg", rec.launches[0].name);
    try std.testing.expectEqual(@as([3]u32, .{ 34, 12, 2 }), rec.launches[0].grid);
    try std.testing.expectEqual(@as(?i64, 6), rec.launches[0].consts[2].int);
    try std.testing.expectError(error.MissingTritonSet, (Tri{}).conv_shift_seg(.{ 1, 1, 1 }, 0, 0, 0, 0, 0, 0, 0, .{ .C = 1, .TAPS = 3, .COLS = 6, .BLOCK = 1024 }));
    }
fn savedArg(s: Recorder.Saved, name: []const u8) aot.Arg {
    for (s.args[0..s.nargs]) |item| if (std.mem.eql(u8, item.name, name)) return item;
    unreachable;
}
fn savedConst(s: Recorder.Saved, name: []const u8) aot.Const {
    for (s.consts[0..s.nconsts]) |item| if (std.mem.eql(u8, item.name, name)) return item;
    unreachable;
}
test "prompt Triton wrappers keep the reference pointer types and constexprs" {
    var rec: Recorder = .{};
    const tri: Tri = .{ .rec = &rec };
    try tri.hc_partial_mma(.{ 32, 16, 1 }, 16, 32, 48, 2048, .{ .WIDE = 16384, .NB = 16, .BM = 64, .BK = 64 });
    try tri.lat_write(.{ 8, 1, 1 }, 16, 512, 32, 32, 48, .{ .LW = 512, .RS = 528, .FP8 = 1 });
    try tri.absorb_mma(.{ 32, 32, 4 }, 16, 32, 48, 2048, .{ .H = 32, .D = 256, .LW = 512, .BM = 64, .BN = 128, .BK = 64 });
    try tri.expand_mma(.{ 32, 32, 2 }, 16, 32, 48, 2048, .{ .H = 32, .DV = 256, .LW = 512, .BM = 64, .BN = 128, .BK = 64 });
    try tri.dense_chunks(.{ 512, 1, 4 }, 16, 32, 32, 48, 64, 80, 96, 512, .{ .H = 32, .LW = 512, .CH = 512, .SCALE = 0.0625, .HBT = 32, .KTT = 32, .RS = 528, .FP8 = 1 });
    try tri.index_write(.{ 8, 1, 1 }, 16, 160, 32, 48, 64, 80, 96, 112, 128, @as(f32, @bitCast(@as(u32, 0x358637bd))), .{ .D = 128 });
    try tri.pool_keys(.{ 4, 1, 1 }, 16, 32, 48, 64, 64, 80, 8, 128, .{ .D = 128, .RS = 144, .FP8 = 1 });
    try tri.scores(.{ 512, 1, 1 }, 16, 32, 160, 48, 48, 64, 80, 512, 1024, @as(f32, @bitCast(@as(u32, 0x3db504f3))), @as(f32, @bitCast(@as(u32, 0x3e3504f3))), .{ .H = 32, .HP = 32, .D = 128, .BP = 64, .L = 32, .RS = 144, .FP8 = 1 });
    try tri.scores_rows(.{ 8, 16, 1 }, 16, 32, 160, 48, 48, 64, 1024, 80, 80, 80, 8, 1024, 0, 0, .{ .H = 32, .HP = 32, .D = 128, .BP = 64, .G = 16, .SEG = 0, .RS = 144, .FP8 = 1, .K = 512 });
    try tri.select_rows(.{ 512, 1, 1 }, 16, 32, 1024, 48, "*fp32", .{ .K = 512, .BLOCK = 1024, .VIS = 0 });
    try tri.select_rows(.{ 8, 1, 1 }, 16, 32, 1024, 48, "*i32", .{ .K = 512, .BLOCK = 4096, .VIS = 1 });
    try tri.select_split_pools(.{ 8, 4, 1 }, 16, 8192, 8192, 32, 32, 48, 64, 80, 96, 112, 128, .{ .W = 2051, .K = 512, .PL = 4, .CHS = 2048, .CP = 4, .STEP = 4, .SEG = 0, .VIS = 0, .TOKS = 0, .BLK = 16 });
    try tri.tokens(.{ 8, 1, 1 }, 16, 32, 48, 64, .{ .W = 2051, .K = 512, .PL = 4, .BLOCK = 1024 });
    try tri.sparse_onepass(.{ 2048, 1, 1 }, 16, 32, 32, 48, 64, 80, .{ .W = 2051, .H = 32, .LW = 512, .SCALE = 0.0625, .HBT = 32, .KTT = 32, .SPLIT = 1, .STAGES = 1, .PREFETCH = 1, .RS = 528, .FP8 = 1 });
    try tri.conv_shift(.{ 1, 12, 1 }, 16, 32, 2048, 12288, 2048 * 12576, 12576, .{ .C = 12288, .TAPS = 3, .BLOCK = 1024 });
    try std.testing.expectEqual(@as(usize, 15), rec.n);
    try std.testing.expectEqualStrings("_hc_partial_mma", rec.launches[0].name);
    try std.testing.expectEqual(@as([3]u32, .{ 32, 16, 1 }), rec.launches[0].grid);
    try std.testing.expectEqualStrings("*fp8e4nv", savedArg(rec.launches[1], "LC").value.ptr.ty);
    try std.testing.expectEqual(savedArg(rec.launches[1], "LC").value.ptr.addr, savedArg(rec.launches[1], "LS").value.ptr.addr);
    try std.testing.expectEqual(@as(?f32, 0.0625), savedConst(rec.launches[4], "SCALE").f32);
    try std.testing.expectEqual(@as(?i64, 0), savedConst(rec.launches[8], "SEG").int);
    try std.testing.expectEqualStrings("*i64", savedArg(rec.launches[9], "OUT").value.ptr.ty);
    try std.testing.expectEqualStrings("*fp32", savedArg(rec.launches[9], "POS").value.ptr.ty);
    try std.testing.expectEqualStrings("*i32", savedArg(rec.launches[10], "POS").value.ptr.ty);
    try std.testing.expectEqualStrings("*i64", savedArg(rec.launches[11], "OUT").value.ptr.ty);
    try std.testing.expectEqualStrings("*i32", savedArg(rec.launches[11], "POS").value.ptr.ty);
    try std.testing.expectEqual(@as(?i64, 1), savedConst(rec.launches[13], "PREFETCH").int);
    try std.testing.expectEqual(@as(?f32, 0.0625), savedConst(rec.launches[13], "SCALE").f32);
    try std.testing.expectEqualStrings("_conv_shift", rec.launches[14].name);
}
test "named typed GLM wrappers match real Python launch arguments and grids" {
    try @import("cuda_triton_generated.zig").checkWrappers(Tri, Recorder);
}

/// Every hand-written wrapper must name a kernel, runtime arguments (names, types, order) and constexpr names that
/// one of the 176 variants captured from the real Python launches has (fixtures_aot_variants.json, from aot.json).
fn expectCapturedAbi(s: Recorder.Saved, specs: []const aot.KernelJson) !void {
    for (specs) |k| {
        if (!std.mem.eql(u8, k.@"fn", s.name)) continue;
        // runtime arguments the variant keeps, in the variant's order
        var at: usize = 0;
        var ok = true;
        for (s.args[0..s.nargs]) |a| {
            const idx = for (k.params, 0..) |p, i| {
                if (std.mem.eql(u8, p.name, a.name)) break i;
            } else null;
            if (idx) |i| {
                if (i < at) ok = false;
                at = i + 1;
                const want = k.params[i].type;
                switch (a.value) {
                    .ptr => |p| if (!std.mem.eql(u8, p.ty, want)) {
                        ok = false;
                    },
                    .i32 => if (!std.mem.eql(u8, want, "i32")) {
                        ok = false;
                    },
                    .f32 => if (!std.mem.eql(u8, want, "fp32")) {
                        ok = false;
                    },
                    .u64 => if (!std.mem.eql(u8, want, "u64")) {
                        ok = false;
                    },
                }
            } else if (k.consts.map.get(a.name) == null) ok = false; // an int Triton folds into a constexpr
        }
        for (k.params) |p| {
            const present = for (s.args[0..s.nargs]) |a| {
                if (std.mem.eql(u8, p.name, a.name)) break true;
            } else false;
            if (!present) ok = false;
        }
        for (s.consts[0..s.nconsts]) |c| if (k.consts.map.get(c.name) == null) {
            ok = false;
        };
        var it = k.consts.map.iterator();
        while (it.next()) |e| {
            const given = for (s.consts[0..s.nconsts]) |c| {
                if (std.mem.eql(u8, c.name, e.key_ptr.*)) break true;
            } else for (s.args[0..s.nargs]) |a| {
                if (std.mem.eql(u8, a.name, e.key_ptr.*)) break true;
            } else false;
            if (!given) ok = false;
        }
        if (ok) return;
    }
    std.debug.print("no captured variant has the ABI of {s}\n", .{s.name});
    return error.WrapperAbiNotCaptured;
}

test "every hand-written Tri wrapper has the argument and constexpr ABI of a captured Python launch" {
    var rec: Recorder = .{};
    const tri: Tri = .{ .rec = &rec };
    try tri.hc_finish(.{ 1, 1, 1 }, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 36864, 1.0, 1.0, .{ .D = 64, .S = 64, .NB = 64, .ITERS = 64, .BLOCK = 64 });
    try tri.hc_post(.{ 1, 1, 1 }, 4096, 8192, 12288, 16384, 20480, 4096, .{ .D = 64, .S = 64, .WORLD = 64, .BLOCK = 64, .ROUND = 64 });
    try tri.stream_mean(.{ 1, 1, 1 }, 4096, 8192, .{ .D = 64, .S = 64, .BLOCK = 64 });
    try tri.swiglu(.{ 1, 1, 1 }, 4096, 8192, 12288, 1.0, .{ .W = 64, .BLOCK = 64 });
    try tri.router_part(.{ 1, 1, 1 }, 4096, 8192, 12288, 4096, 4096, .{ .D = 64, .NE = 64, .BM = 64, .BLOCK_E = 64, .BK = 64, .KS = 64 });
    try tri.router_sum(.{ 1, 1, 1 }, 4096, 8192, 4096, .{ .KS = 64, .BLOCK = 64 });
    try tri.topk(.{ 1, 1, 1 }, 4096, 8192, 12288, 16384, 1.0, .{ .NE = 64, .TOPK = 64, .SLOTS = 64, .BLOCK = 64, .SLOTP = 64, .NORM = 64 });
    try tri.combine_sy(.{ 1, 1, 1 }, 4096, 8192, 12288, 16384, .{ .D = 64, .SLOTS = 64, .BLOCK = 64 });
    try tri.rmsnorm(.{ 1, 1, 1 }, 4096, 4096, 8192, 12288, 4096, 16384, 1.0, .{ .D = 64, .BLOCK = 64, .SUMS = 64 });
    try tri.group_sums(.{ 1, 1, 1 }, 4096, 8192, 4096, .{ .K = 64, .GB = 64 });
    try tri.fmm(.{ 1, 1, 1 }, 4096, 8192, 12288, 16384, 20480, 4096, 4096, .{ .N = 64, .K = 64, .SK = 64, .BM = 64, .BLOCK_N = 64, .BK = 64, .F32 = 64, .SEQ = 64 });
    try tri.seg_lat_write(.{ 1, 1, 1 }, 4096, 4096, 8192, 12288, 16384, 20480, .{ .LW = 64, .RS = 64, .FP8 = 64 });
    try tri.absorb_q(.{ 1, 1, 1 }, 4096, 8192, 12288, 4096, .{ .H = 64, .D = 64, .LW = 64, .BN = 64 });
    try tri.expand_v(.{ 1, 1, 1 }, 4096, 8192, 12288, 4096, .{ .H = 64, .DV = 64, .LW = 64, .BN = 64 });
    try tri.seg_chunks(.{ 1, 1, 1 }, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 36864, 40960, 45056, 4096, .{ .W = 64, .H = 64, .LW = 64, .CH = 64, .SCALE = 2.0, .HBT = 64, .KTT = 64, .RS = 64, .FP8 = 64 });
    try tri.merge(.{ 1, 1, 1 }, 4096, 8192, 12288, 16384, 20480, 4096, .{ .H = 64, .LW = 64, .NCH = 64, .SPARSE = 64 });
    try tri.seg_index_write(.{ 1, 1, 1 }, 4096, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 4096, 1.0, .{ .D = 64 });
    try tri.seg_pool_keys(.{ 1, 1, 1 }, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 4096, .{ .D = 64, .RS = 64, .FP8 = 64 });
    try tri.seg_scores(.{ 1, 1, 1 }, 4096, 8192, 4096, 12288, 16384, 20480, 4096, 24576, 28672, 32768, 36864, 40960, 45056, 1.0, 1.0, .{ .H = 64, .HP = 64, .D = 64, .BP = 64, .G = 64, .RS = 64, .FP8 = 64 });
    try tri.conv_shift_seg(.{ 1, 1, 1 }, 4096, 8192, 12288, 4096, 4096, 4096, 4096, .{ .C = 64, .TAPS = 64, .COLS = 64, .BLOCK = 64 });
    try tri.select_split(.{ 1, 1, 1 }, 4096, 4096, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 36864, .{ .W = 64, .K = 64, .PL = 64, .CHS = 64, .CP = 64, .STEP = 64, .SEG = 64, .VIS = 64, .TOKS = 64, .BLK = 64 });
    try tri.hc_partial_mma(.{ 1, 1, 1 }, 4096, 8192, 12288, 4096, .{ .WIDE = 64, .NB = 64, .BM = 64, .BK = 64 });
    try tri.lat_write(.{ 1, 1, 1 }, 4096, 4096, 8192, 12288, 16384, .{ .LW = 64, .RS = 64, .FP8 = 64 });
    try tri.absorb_mma(.{ 1, 1, 1 }, 4096, 8192, 12288, 4096, .{ .H = 64, .D = 64, .LW = 64, .BM = 64, .BN = 64, .BK = 64 });
    try tri.expand_mma(.{ 1, 1, 1 }, 4096, 8192, 12288, 4096, .{ .H = 64, .DV = 64, .LW = 64, .BM = 64, .BN = 64, .BK = 64 });
    try tri.dense_chunks(.{ 1, 1, 1 }, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 4096, .{ .H = 64, .LW = 64, .CH = 64, .SCALE = 2.0, .HBT = 64, .KTT = 64, .RS = 64, .FP8 = 64 });
    try tri.index_write(.{ 1, 1, 1 }, 4096, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 4096, 1.0, .{ .D = 64 });
    try tri.pool_keys(.{ 1, 1, 1 }, 4096, 8192, 12288, 16384, 20480, 24576, 4096, 4096, .{ .D = 64, .RS = 64, .FP8 = 64 });
    try tri.scores(.{ 1, 1, 1 }, 4096, 8192, 4096, 12288, 16384, 20480, 24576, 4096, 4096, 1.0, 1.0, .{ .H = 64, .HP = 64, .D = 64, .BP = 64, .L = 64, .RS = 64, .FP8 = 64 });
    try tri.scores_rows(.{ 1, 1, 1 }, 4096, 8192, 4096, 12288, 16384, 20480, 4096, 24576, 28672, 32768, 4096, 4096, 1.0, 1.0, .{ .H = 64, .HP = 64, .D = 64, .BP = 64, .G = 64, .SEG = 64, .RS = 64, .FP8 = 64, .K = 64 });
    try tri.select_split_pools(.{ 1, 1, 1 }, 4096, 4096, 4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 36864, .{ .W = 64, .K = 64, .PL = 64, .CHS = 64, .CP = 64, .STEP = 64, .SEG = 64, .VIS = 64, .TOKS = 64, .BLK = 64 });
    try tri.tokens(.{ 1, 1, 1 }, 4096, 8192, 12288, 16384, .{ .W = 64, .K = 64, .PL = 64, .BLOCK = 64 });
    try tri.sparse_onepass(.{ 1, 1, 1 }, 4096, 8192, 12288, 16384, 20480, 24576, .{ .W = 64, .H = 64, .LW = 64, .SCALE = 2.0, .HBT = 64, .KTT = 64, .SPLIT = 64, .STAGES = 64, .PREFETCH = 64, .RS = 64, .FP8 = 64 });
    try tri.conv_shift(.{ 1, 1, 1 }, 4096, 8192, 4096, 4096, 4096, 4096, .{ .C = 64, .TAPS = 64, .BLOCK = 64 });
    try tri.embed(4096, 8192, 12288, 4, 1);
    try tri.select_rows(.{ 1, 1, 1 }, 4096, 8192, 64, 12288, "*i32", .{ .K = 512, .BLOCK = 1024, .VIS = 1 });
    try tri.select_rows(.{ 1, 1, 1 }, 4096, 8192, 64, 12288, "*fp32", .{ .K = 512, .BLOCK = 1024, .VIS = 0 });
    try tri.draftRmsnorm(4096, 4096, 8192, 12288, 16384, 8, 4096, 1e-6);
    try tri.groupSums(4096, 8192, 8, 4096);
    try tri.draftSwiglu(4096, 8192, 12288, 8, 6144);
    try tri.draftConv(4096, 8192, 12288, null, 16384, 8, 0, 8);
    try tri.draftConv(4096, 8192, 12288, 20480, 16384, 8, 1, 8);
    try tri.draftPrep(4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 8, 16, 4, 1e-6);
    try tri.draftAttention(4096, 8192, 12288, 16384, 20480, 24576, 1, 16, 4, 2240);
    const parsed = try std.json.parseFromSlice(aot.SetJson, std.testing.allocator, @embedFile("fixtures_aot_variants.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try std.testing.expect(rec.n > 40);
    for (rec.launches[0..rec.n]) |s| {
        // Known gap: the prompt's pool _select_split (OUT *i64, only for 8192+ pools, ~32k tokens) was not in the
        // TP=2 capture; the Python cubin must be captured before such a prompt can run.
        if (std.mem.eql(u8, s.name, "_select_split") and std.mem.eql(u8, savedArg(s, "OUT").value.ptr.ty, "*i64")) continue;
        try expectCapturedAbi(s, parsed.value.kernels);
    }
}
