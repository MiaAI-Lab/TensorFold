//! GLM-5.3-Flash image and video input on CUDA, the vision tower: patches -> conv3d patch embed -> 24 blocks of
//! rotary attention + SwiGLU -> post norm -> 2x2 downsample -> merger -> bf16 [rows, 4096] on the device.
//!
//! The reference is the Python engine's GlmVisionTower (tensorfold/vision/glm.py, plain torch on bf16 weights). This
//! file reproduces its arithmetic: the linear layers go through cuBLASLt / cuBLAS with the calls PyTorch makes (so the
//! same kernels run), the convolutions are tensor-core GEMM chains in cuDNN's accumulation order, attention follows
//! flash-attention-2's forward kernel (what torch's SDPA runs), and the elementwise parts follow ATen's kernels. See
//! zig/kernels/cuda/glm_vision.cu and the report work/zig-notes/r107-vision-tower.md for what is bit-exact.
//!
//! Written from the Python reference and the CUDA/torch semantics, not from any other implementation.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const kernels = @import("cuda_kernels.zig");
const blas_mod = @import("cuda_vision_blas.zig");
const cudnn_mod = @import("cuda_vision_cudnn.zig");

const st = core.safetensors;

pub const Blas = blas_mod.Blas;

/// What the attention stage runs. `flash2` reproduces PyTorch's flash-attention-2 forward bits (see `flash_variant`);
/// `plain` is an fp32 online-softmax kernel (not bit-identical, slower), kept as a fallback and for measuring.
pub const Attention = enum { flash2, plain };

/// The K order cuDNN's implicit-GEMM patch embedding accumulates in (see `embedMaps`).
pub const EmbedOrder = enum { ctyx, tyxc, tyxc4, tyxc8, yxtc, yxtc8, yxct };
/// How the patch embedding accumulates: `seq` adds one product at a time in fp32 in the map order (cuDNN's order for
/// this convolution, found by probing), `mma` is a tensor-core chain in k16 groups.
pub const EmbedKind = enum { seq, mma };
/// The two convolutions: `cudnn` calls cuDNN's backend API the way PyTorch does (bit-identical to torch's convolutions;
/// needs libcudnn.so.9), `own` runs our tensor-core / sequential kernels (not bit-identical; no cuDNN needed).
pub const ConvImpl = enum { cudnn, own };

pub const Options = struct {
    /// Patches per pass (GlmVisionTower's PASS_PATCHES): a pass holds as many frame pairs of one grid as fit.
    pass_patches: usize = 8192,
    attention: Attention = .flash2,
    /// Arithmetic variants of the flash kernel (glm_vision.cu V_*): 1 rcp.approx, 2 unfused exp argument (PyTorch builds
    /// flash-attention with UNFUSE_FMA), 4 exp2f, 8 QK^T in reverse head-dim order, 16 PV in reverse key order.
    flash_variant: u32 = 2,
    /// LayerNorm arithmetic variants (glm_vision.cu): 1 unfused final, 2 1/sqrt, 4 Welford division, 8 combine division.
    ln_variant: u32 = 0,
    /// GELU through the table of torch's own results at every bf16 value (vision_gelu_bf16.bin): the erf formula in fp32
    /// is off by one bf16 step on 7 of the 65,536 inputs with some CUDA math libraries. false uses the formula.
    gelu_table: bool = true,
    conv: ConvImpl = .cudnn,
    embed_order: EmbedOrder = .tyxc,
    embed_kind: EmbedKind = .seq,
    /// partial-sum length of the downsample's tensor-core chain (0 = one running accumulator)
    downsample_fresh: u32 = 0,
    /// LayerNorm block rows (ATen's vectorized_layer_norm_kernel launches (32, BY) threads).
    ln_rows: u32 = 4,
    /// Called after each stage with the device pointer when set (checks, debugging); each call synchronizes the stream.
    stage_hook: ?StageHook = null,
    /// Time each kind of step with CUDA events (see `Tower.profile`); costs a few event records per layer.
    profile: bool = false,
};

pub const StageDType = enum { bf16, f32 };
pub const StageHook = struct {
    ctx: ?*anyopaque,
    call: *const fn (ctx: ?*anyopaque, name: []const u8, ptr: u64, count: usize, dtype: StageDType) anyerror!void,
};

/// One picture (`frames` = 1: a [3, H, W] canvas the tower expands to a frame pair) or one clip (`frames` even:
/// [frames, 3, H, W]); `pixels` is the uint8 data on the host, H and W multiples of 28.
pub const Item = struct {
    frames: usize,
    h: usize,
    w: usize,
    pixels: []const u8,

    pub fn pairs(i: Item) usize {
        return if (i.frames == 1) 1 else i.frames / 2;
    }
    pub fn rows(i: Item) usize {
        return i.pairs() * (i.h / 28) * (i.w / 28);
    }
    pub fn valid(i: Item) bool {
        return (i.frames == 1 or i.frames % 2 == 0) and i.h % 28 == 0 and i.w % 28 == 0 and i.h > 0 and i.w > 0 and i.pixels.len == i.frames * 3 * i.h * i.w;
    }
};

pub const Config = struct {
    hidden: usize = 1024,
    heads: usize = 16,
    depth: usize = 24,
    inter: usize = 4096,
    out: usize = 4096,
    proj_inter: usize = 10240,
    patch: usize = 14,
    temporal: usize = 2,
    merge: usize = 2,
    eps: f32 = 1e-5,
    limit: f32 = 10,
    theta: f32 = 10000,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, model_dir: []const u8) !Config {
        const path = try std.fs.path.join(gpa, &.{ model_dir, "config.json" });
        defer gpa.free(path);
        const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
        defer gpa.free(raw);
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, raw, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.BadConfig;
        const vc = parsed.value.object.get("vision_config") orelse return error.NoVisionTower;
        if (vc != .object) return error.BadConfig;
        var c: Config = .{};
        c.hidden = try int(vc.object, "hidden_size");
        c.heads = try int(vc.object, "num_heads");
        c.depth = try int(vc.object, "depth");
        c.inter = try int(vc.object, "intermediate_size");
        c.out = try int(vc.object, "out_hidden_size");
        c.proj_inter = try int(vc.object, "projection_intermediate_size");
        c.patch = try int(vc.object, "patch_size");
        c.temporal = try int(vc.object, "temporal_patch_size");
        c.merge = try int(vc.object, "spatial_merge_size");
        c.eps = @floatCast(try float(vc.object, "rms_norm_eps", 1e-5));
        c.limit = @floatCast(try float(vc.object, "swiglu_limit", 10));
        if (vc.object.get("rope_parameters")) |rp| if (rp == .object) {
            c.theta = @floatCast(try float(rp.object, "rope_theta", 10000));
        };
        if (c.hidden != 1024 or c.heads != 16 or c.patch != 14 or c.temporal != 2 or c.merge != 2 or c.out != 4096 or c.depth == 0) return error.UnsupportedVisionGeometry;
        return c;
    }
    fn int(o: std.json.ObjectMap, name: []const u8) !usize {
        const v = o.get(name) orelse return error.BadConfig;
        if (v != .integer or v.integer < 0) return error.BadConfig;
        return @intCast(v.integer);
    }
    fn float(o: std.json.ObjectMap, name: []const u8, default: f64) !f64 {
        const v = o.get(name) orelse return default;
        return switch (v) {
            .float => |f| f,
            .integer => |i| @floatFromInt(i),
            else => error.BadConfig,
        };
    }
};

pub const Block = struct {
    norm1: u64,
    qkv_w: u64,
    qkv_b: u64,
    q_norm: u64,
    k_norm: u64,
    proj_w: u64,
    proj_b: u64,
    norm2: u64,
    gate_w: u64,
    gate_b: u64,
    up_w: u64,
    up_b: u64,
    down_w: u64,
    down_b: u64,
};

pub const Weights = struct {
    buf: cuda.DeviceBuffer,
    embed_w: u64,
    embed_b: u64,
    blocks: []Block,
    post_norm: u64,
    down_w: u64,
    down_b: u64,
    merger_proj: u64,
    merger_norm_w: u64,
    merger_norm_b: u64,
    merger_gate: u64,
    merger_up: u64,
    merger_down: u64,
    bytes: usize,
};

const Fns = struct {
    patches: cuda.Function,
    rope: cuda.Function,
    rms: cuda.Function,
    qk: cuda.Function,
    add: cuda.Function,
    swiglu: cuda.Function,
    ln_gelu: cuda.Function,
    gelu: cuda.Function,
    chain: cuda.Function,
    seq: cuda.Function,
    bias_add: cuda.Function,
    down_perm: cuda.Function,
    flash: cuda.Function,
    plain: cuda.Function,
};

pub const mean_rgb = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
pub const std_rgb = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };

pub const Buffers = struct {
    cap: usize = 0, // patch rows the buffers hold
    x: cuda.DeviceBuffer = undefined, // residual stream [R, 1024]
    h: cuda.DeviceBuffer = undefined, // norm output [R, 1024]
    qkv: cuda.DeviceBuffer = undefined, // [R, 3072]
    q: cuda.DeviceBuffer = undefined, // [R, 1024]
    k: cuda.DeviceBuffer = undefined,
    o: cuda.DeviceBuffer = undefined,
    gate: cuda.DeviceBuffer = undefined, // [R, 4096]
    up: cuda.DeviceBuffer = undefined,
    rope_cos: cuda.DeviceBuffer = undefined, // [S, 64] fp32
    rope_sin: cuda.DeviceBuffer = undefined,
    patches: cuda.DeviceBuffer = undefined, // bf16 [R, 1176]
    patches_f32: cuda.DeviceBuffer = undefined, // only when a stage hook is set
    rope_cap: usize = 0,
    frames: cuda.DeviceBuffer = undefined, // uint8 pass input
    frames_cap: usize = 0,
};

/// torch's F.gelu of every bf16 value (index = the input's bit pattern), see tests/vision/tower/gen_luts.py
const gelu_table_bytes = @embedFile("vision_gelu_bf16.bin");

const Mark = struct { name: []const u8, ev: cuda.Event };

pub const Tower = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    cfg: Config,
    opts: Options,
    w: Weights,
    mod: cuda.Module,
    f: Fns,
    blas: Blas,
    b: Buffers = .{},
    maps: [2]cuda.DeviceBuffer, // embed A and W maps for opts.embed_order
    ds_map: ?cuda.DeviceBuffer = null, // downsample weight map
    cudnn: ?cudnn_mod.Cudnn = null,
    marks: std.ArrayList(Mark) = .empty,
    prof_ms: std.StringArrayHashMapUnmanaged(f64) = .empty,
    down_w: ?cuda.DeviceBuffer = null, // the downsample weight in channels-last order (for cuDNN)
    gelu_lut: ?cuda.DeviceBuffer = null,
    ds_map_ptr: u64 = 0,
    kpad: usize,
    peak_bytes: usize = 0,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, model_dir: []const u8, opts: Options) !Tower {
        var self: Tower = undefined;
        self.gpa = gpa;
        self.d = d;
        self.stream = stream;
        self.opts = opts;
        self.b = .{};
        self.peak_bytes = 0;
        self.ds_map = null;
        self.ds_map_ptr = 0;
        self.cudnn = null;
        self.down_w = null;
        self.gelu_lut = null;
        self.marks = .empty;
        self.prof_ms = .empty;
        self.cfg = try Config.load(gpa, io, model_dir);
        self.w = try loadWeights(gpa, io, d, model_dir, self.cfg);
        errdefer freeWeights(gpa, &self.w);
        self.mod = try kernels.load(d, .vision);
        errdefer self.mod.unload();
        self.f = .{
            .patches = try self.mod.function("vis_patches"),
            .rope = try self.mod.function("vis_rope_table"),
            .rms = try self.mod.function("vis_rms1024"),
            .qk = try self.mod.function("vis_qk_norm_rope"),
            .add = try self.mod.function("vis_add"),
            .swiglu = try self.mod.function("vis_swiglu"),
            .ln_gelu = try self.mod.function("vis_layernorm_gelu"),
            .gelu = try self.mod.function("vis_gelu"),
            .chain = try self.mod.function("vis_gemm_chain"),
            .seq = try self.mod.function("vis_gemm_seq"),
            .bias_add = try self.mod.function("vis_bias_add"),
            .down_perm = try self.mod.function("vis_down_weight_nhwc"),
            .flash = try self.mod.function("vis_flash"),
            .plain = try self.mod.function("vis_attn_plain"),
        };
        self.blas = try Blas.init(gpa, d, stream);
        errdefer self.blas.deinit();
        self.maps = .{ undefined, undefined };
        self.kpad = try self.uploadMaps(opts.embed_order);
        try self.ensureDownsampleMap();
        self.gelu_lut = try cuda.DeviceBuffer.fromHost(d, gelu_table_bytes);
        errdefer self.gelu_lut.?.free();
        if (opts.conv == .cudnn) {
            self.cudnn = try cudnn_mod.Cudnn.init(gpa, d, stream);
            errdefer self.cudnn.?.deinit();
            var dw = try cuda.DeviceBuffer.alloc(d, 4096 * 4096 * 2);
            errdefer dw.free();
            var a: cuda.Args = .{};
            a.add(self.w.down_w);
            a.add(dw.ptr);
            try cuda.launch.launch(self.f.down_perm, .{ .grid = .{ .x = 4096 }, .block = .{ .x = 256 } }, stream, &a);
            try stream.synchronize();
            self.down_w = dw;
        }
        return self;
    }

    pub fn deinit(self: *Tower) void {
        self.freeBuffers();
        for (&self.maps) |*m| m.free();
        if (self.ds_map) |*m| m.free();
        if (self.down_w) |*m| m.free();
        if (self.gelu_lut) |*m| m.free();
        if (self.cudnn) |*c| c.deinit();
        for (self.marks.items) |*m| m.ev.deinit();
        self.marks.deinit(self.gpa);
        self.prof_ms.deinit(self.gpa);
        self.blas.deinit();
        self.mod.unload();
        freeWeights(self.gpa, &self.w);
        self.* = undefined;
    }

    pub fn setEmbedOrder(self: *Tower, order: EmbedOrder) !void {
        for (&self.maps) |*m| m.free();
        self.opts.embed_order = order;
        self.kpad = try self.uploadMaps(order);
    }

    /// Feature rows an item produces.
    /// The cuDNN engine (global index) the heuristics pick for a convolution of `rows` rows, for diagnostics.
    pub fn convEngine(self: *Tower, downsample: bool, n_rows: usize) !i64 {
        const c = &(self.cudnn orelse return -1);
        const p = try c.plan(if (downsample) cudnn_mod.Spec.downsample(n_rows) else cudnn_mod.Spec.embed(n_rows));
        return p.engine;
    }

    pub fn rows(_: *const Tower, item: Item) usize {
        return item.rows();
    }

    /// Device bytes beyond the weights for a pass of `patches` patch rows (the Python tower reserves 0.75 GiB).
    pub fn workspaceBytes(patches: usize) usize {
        const R = patches;
        const bf = 2;
        return R * (1024 * bf * 4 + 3072 * bf + 4096 * bf * 2 + 1176 * bf) + (R / 1) * 64 * 4 * 2 + blas_mod.lt_workspace_bytes + blas_mod.gemm_workspace_bytes;
    }

    // ---- buffers ----
    fn freeBuffers(self: *Tower) void {
        if (self.b.cap > 0) {
            inline for (.{ "x", "h", "qkv", "q", "k", "o", "gate", "up", "patches" }) |n| @field(self.b, n).free();
            if (self.b.patches_f32.len > 0) self.b.patches_f32.free();
        }
        if (self.b.rope_cap > 0) {
            self.b.rope_cos.free();
            self.b.rope_sin.free();
        }
        if (self.b.frames_cap > 0) self.b.frames.free();
        self.b = .{};
    }

    pub fn ensure(self: *Tower, R: usize, S: usize, frame_bytes: usize) !void {
        if (R > self.b.cap) {
            const old = self.b;
            const keep_rope = old.rope_cap;
            _ = keep_rope;
            if (old.cap > 0) {
                inline for (.{ "x", "h", "qkv", "q", "k", "o", "gate", "up", "patches" }) |n| @field(self.b, n).free();
                if (old.patches_f32.len > 0) self.b.patches_f32.free();
            }
            const d = self.d;
            self.b.x = try cuda.DeviceBuffer.alloc(d, R * 1024 * 2);
            self.b.h = try cuda.DeviceBuffer.alloc(d, R * 1024 * 2);
            self.b.qkv = try cuda.DeviceBuffer.alloc(d, R * 3072 * 2);
            self.b.q = try cuda.DeviceBuffer.alloc(d, R * 1024 * 2);
            self.b.k = try cuda.DeviceBuffer.alloc(d, R * 1024 * 2);
            self.b.o = try cuda.DeviceBuffer.alloc(d, R * 1024 * 2);
            self.b.gate = try cuda.DeviceBuffer.alloc(d, @max(R * self.cfg.inter, R / 4 * self.cfg.proj_inter) * 2);
            self.b.up = try cuda.DeviceBuffer.alloc(d, @max(R * self.cfg.inter, R / 4 * self.cfg.proj_inter) * 2);
            self.b.patches = try cuda.DeviceBuffer.alloc(d, R * 1176 * 2);
            self.b.patches_f32 = if (self.opts.stage_hook != null) try cuda.DeviceBuffer.alloc(d, R * 1176 * 4) else .{ .d = d, .ptr = 0, .len = 0 };
            self.b.cap = R;
        } else if (self.opts.stage_hook != null and self.b.patches_f32.len < R * 1176 * 4) {
            if (self.b.patches_f32.len > 0) self.b.patches_f32.free();
            self.b.patches_f32 = try cuda.DeviceBuffer.alloc(self.d, R * 1176 * 4);
        }
        if (S > self.b.rope_cap) {
            if (self.b.rope_cap > 0) {
                self.b.rope_cos.free();
                self.b.rope_sin.free();
            }
            self.b.rope_cos = try cuda.DeviceBuffer.alloc(self.d, S * 64 * 4);
            self.b.rope_sin = try cuda.DeviceBuffer.alloc(self.d, S * 64 * 4);
            self.b.rope_cap = S;
        }
        if (frame_bytes > self.b.frames_cap) {
            if (self.b.frames_cap > 0) self.b.frames.free();
            self.b.frames = try cuda.DeviceBuffer.alloc(self.d, frame_bytes);
            self.b.frames_cap = frame_bytes;
        }
        const u = cuda.usage(false);
        self.peak_bytes = @max(self.peak_bytes, u.peak);
    }

    // ---- launches ----
    fn grid1(n: usize) u32 {
        return @intCast(@min((n + 255) / 256, 1 << 16));
    }

    fn launch(self: *const Tower, f: cuda.Function, grid: cuda.Dim3, block: cuda.Dim3, shared: u32, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = grid, .block = block, .shared = shared }, self.stream, a);
    }

    /// Records a CUDA event named for the work queued since the previous mark (only when `Options.profile`).
    fn mark(self: *Tower, name: []const u8) !void {
        if (!self.opts.profile) return;
        var ev = try cuda.Event.init(self.d, true);
        errdefer ev.deinit();
        try ev.record(self.stream);
        try self.marks.append(self.gpa, .{ .name = name, .ev = ev });
    }

    /// Adds the marks' durations to `prof_ms` (per name) and drops them; call after the stream has finished.
    pub fn profileFlush(self: *Tower) !void {
        var prev: ?cuda.Event = null;
        for (self.marks.items) |*m| {
            if (prev) |p| {
                const ms = try cuda.Event.elapsedMs(p, m.ev);
                const e = try self.prof_ms.getOrPut(self.gpa, m.name);
                if (!e.found_existing) e.value_ptr.* = 0;
                e.value_ptr.* += ms;
            }
            prev = m.ev;
        }
        for (self.marks.items) |*m| m.ev.deinit();
        self.marks.clearRetainingCapacity();
    }

    fn hook(self: *const Tower, name: []const u8, ptr: u64, count: usize, dtype: StageDType) !void {
        if (self.opts.stage_hook) |h| {
            try self.stream.synchronize();
            try h.call(h.ctx, name, ptr, count, dtype);
        }
    }

    // ---- stages (public so a check can run each one from a reference input) ----

    /// Patch matrix of a pass: bf16 (and fp32 when `patches_f32` is set) from frames already on the device.
    pub fn stagePatches(self: *Tower, frames: u64, frame_stride: u64, pair0: usize, pairs: usize, gh: usize, gw: usize, h: usize, w: usize, bf_out: u64, f32_out: u64) !void {
        var a: cuda.Args = .{};
        a.add(frames);
        a.add(@as(i64, @intCast(frame_stride)));
        a.add(@as(i32, @intCast(pair0)));
        a.add(@as(i32, @intCast(pairs)));
        a.add(@as(i32, @intCast(gh)));
        a.add(@as(i32, @intCast(gw)));
        a.add(@as(i32, @intCast(h)));
        a.add(@as(i32, @intCast(w)));
        // (x - mean * 255) / (std * 255): the scaled constants are fp32 products, as torch forms them
        for (mean_rgb) |m| a.add(m * 255.0);
        for (std_rgb) |s| a.add(s * 255.0);
        a.add(bf_out);
        a.add(f32_out);
        try self.launch(self.f.patches, .{ .x = grid1(pairs * gh * gw * 1176) }, .{ .x = 256 }, 0, &a);
    }

    /// conv3d patch embedding: bf16 patches [R, 1176] -> x [R, 1024]; the bias is a second bf16 add.
    pub fn stageEmbed(self: *Tower, patches: u64, out: u64, R: usize, with_bias: bool) !void {
        const bias = if (with_bias) self.w.embed_b else 0;
        if (self.opts.conv == .cudnn) if (self.cudnn) |*c| {
            try c.run(cudnn_mod.Spec.embed(R), patches, self.w.embed_w, out);
            if (with_bias) try self.biasAdd(out, bias, 1024, R * 1024);
            return;
        };
        switch (self.opts.embed_kind) {
            .mma => try self.chain(patches, 1176, self.w.embed_w, 1176, self.maps[0].ptr, self.maps[1].ptr, self.kpad, bias, out, R, 1024, 0),
            .seq => {
                var a: cuda.Args = .{};
                a.add(patches);
                a.add(@as(i64, 1176));
                a.add(self.w.embed_w);
                a.add(@as(i64, 1176));
                a.add(self.maps[0].ptr);
                a.add(self.maps[1].ptr);
                a.add(@as(i32, @intCast(self.kpad)));
                a.add(bias);
                a.add(out);
                a.add(@as(i32, @intCast(R)));
                a.add(@as(i32, 1024));
                try self.launch(self.f.seq, .{ .x = 16, .y = @intCast((R + 63) / 64) }, .{ .x = 256 }, 0, &a);
            },
        }
    }

    pub fn chain(self: *Tower, a_ptr: u64, lda: usize, w_ptr: u64, ldw: usize, amap: u64, wmap: u64, kpad: usize, bias: u64, out: u64, M: usize, N: usize, fresh: u32) !void {
        var a: cuda.Args = .{};
        a.add(a_ptr);
        a.add(@as(i64, @intCast(lda)));
        a.add(w_ptr);
        a.add(@as(i64, @intCast(ldw)));
        a.add(amap);
        a.add(wmap);
        a.add(@as(i32, @intCast(kpad)));
        a.add(bias);
        a.add(out);
        a.add(@as(i32, @intCast(M)));
        a.add(@as(i32, @intCast(N)));
        a.add(@as(i32, @intCast(fresh)));
        try self.launch(self.f.chain, .{ .x = @intCast((N + 63) / 64), .y = @intCast((M + 63) / 64) }, .{ .x = 128 }, 0, &a);
    }

    pub fn stageRope(self: *Tower, gh: usize, gw: usize) !void {
        var a: cuda.Args = .{};
        a.add(self.b.rope_cos.ptr);
        a.add(self.b.rope_sin.ptr);
        a.add(@as(i32, @intCast(gh)));
        a.add(@as(i32, @intCast(gw)));
        a.add(self.cfg.theta);
        try self.launch(self.f.rope, .{ .x = grid1(gh * gw * 64) }, .{ .x = 256 }, 0, &a);
    }

    pub fn stageRms(self: *Tower, x: u64, w: u64, out: u64, R: usize) !void {
        // ATen's mean-reduce block follows the row count: 16 or more rows -> 32 lanes per row (a (32, 16) block)
        var bh: u32 = 1;
        while (bh * 2 <= @min(R, 16)) bh *= 2;
        const bw: u32 = @min(256, 512 / bh);
        const per_block: u32 = @max(1, 128 / bw);
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(out);
        a.add(@as(i32, @intCast(R)));
        a.add(self.cfg.eps);
        try self.launch(self.f.rms, .{ .x = @intCast((R + per_block - 1) / per_block) }, .{ .x = bw, .y = per_block }, bw * per_block * 4, &a);
    }

    pub fn stageQkNormRope(self: *Tower, qkv: u64, qw: u64, kw: u64, q_out: u64, k_out: u64, R: usize, S: usize) !void {
        var a: cuda.Args = .{};
        a.add(qkv);
        a.add(qw);
        a.add(kw);
        a.add(self.b.rope_cos.ptr);
        a.add(self.b.rope_sin.ptr);
        a.add(q_out);
        a.add(k_out);
        a.add(@as(i32, @intCast(R)));
        a.add(@as(i32, @intCast(S)));
        a.add(self.cfg.eps);
        try self.launch(self.f.qk, .{ .x = @intCast((R * 32 * 32 + 255) / 256) }, .{ .x = 256 }, 0, &a); // one warp a (row, q or k, head): not a grid-stride loop
    }

    pub fn stageAttention(self: *Tower, q: u64, k: u64, v: u64, out: u64, R: usize, S: usize) !void {
        switch (self.opts.attention) {
            .flash2 => {
                var a: cuda.Args = .{};
                a.add(q);
                a.add(k);
                a.add(v);
                a.add(@as(i64, 3072));
                a.add(out);
                a.add(@as(i32, @intCast(S)));
                // scale * log2(e) in double, then rounded: flash-attention's params.scale_softmax_log2
                a.add(@as(f32, @floatCast(0.125 * 1.4426950408889634)));
                a.add(@as(i32, @intCast(self.opts.flash_variant)));
                try self.launch(self.f.flash, .{ .x = @intCast((S + 63) / 64), .y = 16, .z = @intCast(R / S) }, .{ .x = 128 }, 0, &a);
            },
            .plain => {
                var a: cuda.Args = .{};
                a.add(q);
                a.add(k);
                a.add(v);
                a.add(@as(i64, 3072));
                a.add(out);
                a.add(@as(i32, @intCast(S)));
                a.add(@as(i32, @intCast(R)));
                try self.launch(self.f.plain, .{ .x = @intCast((R * 16 * 32 + 255) / 256) }, .{ .x = 256 }, 0, &a);
            },
        }
    }

    pub fn stageAdd(self: *Tower, a_ptr: u64, b_ptr: u64, out: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(a_ptr);
        a.add(b_ptr);
        a.add(out);
        a.add(@as(i64, @intCast(n)));
        try self.launch(self.f.add, .{ .x = grid1(n) }, .{ .x = 256 }, 0, &a);
    }

    pub fn stageSwiglu(self: *Tower, gate: u64, up: u64, out: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(gate);
        a.add(up);
        a.add(out);
        a.add(@as(i64, @intCast(n)));
        a.add(self.cfg.limit);
        try self.launch(self.f.swiglu, .{ .x = grid1(n) }, .{ .x = 256 }, 0, &a);
    }

    pub fn stageLnGelu(self: *Tower, x: u64, ln_out: u64, out: u64, R: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(self.w.merger_norm_w);
        a.add(self.w.merger_norm_b);
        a.add(ln_out);
        a.add(out);
        a.add(@as(f32, 1e-5));
        a.add(@as(i32, @intCast(self.opts.ln_variant)));
        a.add(self.geluLut());
        const by = self.opts.ln_rows;
        try self.launch(self.f.ln_gelu, .{ .x = @intCast(R) }, .{ .x = 32, .y = by }, by * 3 / 2 * 4 + 64, &a);
    }

    fn geluLut(self: *const Tower) u64 {
        return if (self.opts.gelu_table) self.gelu_lut.?.ptr else 0;
    }

    pub fn stageGelu(self: *Tower, x: u64, out: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(out);
        a.add(@as(i64, @intCast(n)));
        a.add(self.geluLut());
        try self.launch(self.f.gelu, .{ .x = grid1(n) }, .{ .x = 256 }, 0, &a);
    }

    /// One transformer block on x [R, 1024] (in place), groups of S tokens attending within themselves.
    pub fn stageBlock(self: *Tower, i: usize, R: usize, S: usize) !void {
        const w = self.w.blocks[i];
        const b = &self.b;
        const bl = &self.blas;
        const first = i == 0;
        try self.stageRms(b.x.ptr, w.norm1, b.h.ptr, R);
        try self.mark("block: rms norms");
        if (first) try self.hook("b0_norm1", b.h.ptr, R * 1024, .bf16);
        try bl.linear(self.stream, b.h.ptr, w.qkv_w, w.qkv_b, b.qkv.ptr, R, 3072, 1024);
        try self.mark("block: qkv linear");
        if (first) try self.hook("b0_qkv", b.qkv.ptr, R * 3072, .bf16);
        try self.stageQkNormRope(b.qkv.ptr, w.q_norm, w.k_norm, b.q.ptr, b.k.ptr, R, S);
        try self.mark("block: q/k norm + rotary");
        if (first) {
            try self.hook("b0_qr", b.q.ptr, R * 1024, .bf16);
            try self.hook("b0_kr", b.k.ptr, R * 1024, .bf16);
        }
        try self.stageAttention(b.q.ptr, b.k.ptr, b.qkv.ptr + 2048 * 2, b.o.ptr, R, S);
        try self.mark("block: attention");
        if (first) try self.hook("b0_attn", b.o.ptr, R * 1024, .bf16);
        try bl.linear(self.stream, b.o.ptr, w.proj_w, w.proj_b, b.q.ptr, R, 1024, 1024); // q is free now
        try self.mark("block: proj linear");
        if (first) try self.hook("b0_proj", b.q.ptr, R * 1024, .bf16);
        try self.stageAdd(b.x.ptr, b.q.ptr, b.x.ptr, R * 1024);
        try self.mark("block: rms norms");
        if (first) try self.hook("b0_res1", b.x.ptr, R * 1024, .bf16);
        try self.stageRms(b.x.ptr, w.norm2, b.h.ptr, R);
        if (first) try self.hook("b0_norm2", b.h.ptr, R * 1024, .bf16);
        const inter = self.cfg.inter;
        try bl.linear(self.stream, b.h.ptr, w.gate_w, w.gate_b, b.gate.ptr, R, inter, 1024);
        try bl.linear(self.stream, b.h.ptr, w.up_w, w.up_b, b.up.ptr, R, inter, 1024);
        try self.mark("block: gate/up linears");
        if (first) {
            try self.hook("b0_gate_raw", b.gate.ptr, R * inter, .bf16);
            try self.hook("b0_up_raw", b.up.ptr, R * inter, .bf16);
        }
        try self.stageSwiglu(b.gate.ptr, b.up.ptr, b.gate.ptr, R * inter);
        try self.mark("block: swiglu");
        if (first) try self.hook("b0_act", b.gate.ptr, R * inter, .bf16);
        try bl.linear(self.stream, b.gate.ptr, w.down_w, w.down_b, b.h.ptr, R, 1024, inter);
        try self.mark("block: down linear");
        try self.stageAdd(b.x.ptr, b.h.ptr, b.x.ptr, R * 1024);
        try self.mark("block: rms norms");
    }

    /// post norm, downsample and the merger on x [R, 1024] -> out [R/4, 4096].
    pub fn stageHead(self: *Tower, R: usize, out: u64) !void {
        const b = &self.b;
        const bl = &self.blas;
        const M = R / 4;
        try self.stageRms(b.x.ptr, self.w.post_norm, b.h.ptr, R);
        try self.hook("post_norm", b.h.ptr, R * 1024, .bf16);
        // downsample: a 2x2 conv, stride 2, on rows already in merge-block order: K = (kh, kw, c) in memory order
        try self.stageDownsample(b.h.ptr, b.o.ptr, M, true);
        try self.hook("downsample", b.o.ptr, M * 4096, .bf16);
        try bl.linear(self.stream, b.o.ptr, self.w.merger_proj, 0, b.q.ptr, M, 4096, 4096);
        try self.hook("merger_proj", b.q.ptr, M * 4096, .bf16);
        // b.k holds the LayerNorm result, b.q the GELU output (both [M, 4096] bf16 fit in the q/k/o/h rows used here)
        try self.stageLnGelu(b.q.ptr, b.k.ptr, b.q.ptr, M);
        try self.hook("merger_ln", b.k.ptr, M * 4096, .bf16);
        try self.hook("merger_gelu", b.q.ptr, M * 4096, .bf16);
        const pi = self.cfg.proj_inter;
        try bl.linear(self.stream, b.q.ptr, self.w.merger_gate, 0, b.gate.ptr, M, pi, 4096);
        try bl.linear(self.stream, b.q.ptr, self.w.merger_up, 0, b.up.ptr, M, pi, 4096);
        try self.stageSwiglu(b.gate.ptr, b.up.ptr, b.gate.ptr, M * pi);
        try bl.linear(self.stream, b.gate.ptr, self.w.merger_down, 0, out, M, 4096, pi);
        try self.hook("features", out, M * 4096, .bf16);
    }

    pub fn biasAdd(self: *Tower, y: u64, bias: u64, cols: usize, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(y);
        a.add(bias);
        a.add(@as(i32, @intCast(cols)));
        a.add(@as(i64, @intCast(n)));
        try self.launch(self.f.bias_add, .{ .x = grid1(n) }, .{ .x = 256 }, 0, &a);
    }

    pub fn stageDownsample(self: *Tower, x: u64, out: u64, M: usize, with_bias: bool) !void {
        if (self.opts.conv == .cudnn) if (self.cudnn) |*c| {
            try c.run(cudnn_mod.Spec.downsample(M), x, self.down_w.?.ptr, out);
            if (with_bias) try self.biasAdd(out, self.w.down_b, 4096, M * 4096);
            return;
        };
        // W [4096, 1024, 2, 2] (c, kh, kw) -> K order (kh, kw, c): wmap[k] = c * 4 + kh * 2 + kw
        try self.chain(x, 4096, self.w.down_w, 4096, 0, self.dsmap(), 4096, if (with_bias) self.w.down_b else 0, out, M, 4096, self.opts.downsample_fresh);
    }

    fn dsmap(self: *Tower) u64 {
        return self.ds_map_ptr;
    }

    // ---- the pass loop ----
    /// Encodes the items to `out` (bf16, [sum of rows, 4096]) in item order, pass by pass as GlmVisionTower.features.
    pub fn encode(self: *Tower, items: []const Item, out: u64) !void {
        try self.ensureDownsampleMap();
        var at: usize = 0;
        for (items) |it| {
            if (!it.valid()) return error.BadItem;
            const gh = it.h / 14;
            const gw = it.w / 14;
            const S = gh * gw;
            const frames_total = if (it.frames == 1) 2 else it.frames;
            const pairs_total = frames_total / 2;
            const per = @max(1, self.opts.pass_patches / S);
            var start: usize = 0;
            while (start < pairs_total) : (start += per) {
                const n = @min(per, pairs_total - start);
                const R = n * S;
                const frame_bytes = 3 * it.h * it.w;
                const upload = if (it.frames == 1) frame_bytes else 2 * n * frame_bytes;
                try self.ensure(R, S, upload);
                const src = if (it.frames == 1) it.pixels else it.pixels[2 * start * frame_bytes ..][0..upload];
                try self.b.frames.upload(0, src);
                try self.runPass(self.b.frames.ptr, if (it.frames == 1) 0 else frame_bytes, 0, n, gh, gw, it.h, it.w, out + at * 4096 * 2);
                at += R / 4;
            }
        }
        try self.stream.synchronize();
    }

    fn runPass(self: *Tower, frames: u64, frame_stride: u64, pair0: usize, n: usize, gh: usize, gw: usize, h: usize, w: usize, out: u64) !void {
        const S = gh * gw;
        const R = n * S;
        const b = &self.b;
        try self.mark("start");
        try self.stagePatches(frames, frame_stride, pair0, n, gh, gw, h, w, b.patches.ptr, b.patches_f32.ptr);
        if (self.opts.stage_hook != null) try self.hook("patches", b.patches_f32.ptr, R * 1176, .f32);
        try self.mark("patches");
        try self.stageEmbed(b.patches.ptr, b.x.ptr, R, true);
        try self.mark("embed conv");
        try self.hook("embed", b.x.ptr, R * 1024, .bf16);
        try self.stageRope(gh, gw);
        if (self.opts.stage_hook != null) {
            try self.hook("rope_cos", b.rope_cos.ptr, S * 64, .f32);
            try self.hook("rope_sin", b.rope_sin.ptr, S * 64, .f32);
        }
        var name: [16]u8 = undefined;
        for (0..self.cfg.depth) |i| {
            try self.stageBlock(i, R, S);
            if (self.opts.stage_hook != null) {
                const nm = try std.fmt.bufPrint(&name, "block{d:0>2}", .{i});
                try self.hook(nm, b.x.ptr, R * 1024, .bf16);
            }
        }
        try self.stageHead(R, out);
        try self.mark("head (post norm, downsample, merger)");
    }

    fn ensureDownsampleMap(self: *Tower) !void {
        if (self.ds_map != null) return;
        var m: [4096]i32 = undefined;
        for (0..4096) |k| {
            const kh = k / 2048;
            const kw = (k / 1024) % 2;
            const c = k % 1024;
            m[k] = @intCast(c * 4 + kh * 2 + kw);
        }
        self.ds_map = try cuda.DeviceBuffer.fromHost(self.d, std.mem.sliceAsBytes(&m));
        self.ds_map_ptr = self.ds_map.?.ptr;
    }

    fn uploadMaps(self: *Tower, order: EmbedOrder) !usize {
        const m = try embedMaps(self.gpa, order);
        defer self.gpa.free(m.a);
        defer self.gpa.free(m.w);
        self.maps[0] = try cuda.DeviceBuffer.fromHost(self.d, std.mem.sliceAsBytes(m.a));
        errdefer self.maps[0].free();
        self.maps[1] = try cuda.DeviceBuffer.fromHost(self.d, std.mem.sliceAsBytes(m.w));
        return m.a.len;
    }
};

/// The patch embedding as an implicit GEMM: K positions of the accumulation chain mapped to indices in a patch row /
/// weight row (both laid out (channel, time, y, x), 1176 values); -1 is a zero pad. The order is the hypothesis for
/// cuDNN's kernel; the report says which one reproduces torch.
pub fn embedMaps(gpa: std.mem.Allocator, order: EmbedOrder) !struct { a: []i32, w: []i32 } {
    const slots: usize = switch (order) {
        .ctyx => 1176,
        .tyxc, .yxct => 1176,
        .yxtc => 1176,
        .tyxc4 => 4 * 392,
        .tyxc8 => 8 * 392,
        .yxtc8 => 8 * 196,
    };
    const kpad = (slots + 15) / 16 * 16;
    const a = try gpa.alloc(i32, kpad);
    errdefer gpa.free(a);
    @memset(a, -1);
    switch (order) {
        .ctyx => for (0..1176) |k| {
            a[k] = @intCast(k);
        },
        .tyxc, .tyxc4, .tyxc8 => {
            const cs: usize = switch (order) {
                .tyxc => 3,
                .tyxc4 => 4,
                else => 8,
            };
            for (0..2) |t| for (0..14) |y| for (0..14) |x| for (0..3) |c| {
                a[((t * 14 + y) * 14 + x) * cs + c] = @intCast(((c * 2 + t) * 14 + y) * 14 + x);
            };
        },
        .yxtc, .yxtc8 => {
            const ts: usize = if (order == .yxtc) 6 else 8;
            for (0..14) |y| for (0..14) |x| for (0..2) |t| for (0..3) |c| {
                a[(y * 14 + x) * ts + t * 3 + c] = @intCast(((c * 2 + t) * 14 + y) * 14 + x);
            };
        },
        .yxct => for (0..14) |y| for (0..14) |x| for (0..3) |c| for (0..2) |t| {
            a[((y * 14 + x) * 3 + c) * 2 + t] = @intCast(((c * 2 + t) * 14 + y) * 14 + x);
        },
    }
    const w = try gpa.dupe(i32, a);
    return .{ .a = a, .w = w };
}

fn loadWeights(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, model_dir: []const u8, cfg: Config) !Weights {
    // names -> shard files from the index
    const ipath = try std.fs.path.join(gpa, &.{ model_dir, "model.safetensors.index.json" });
    defer gpa.free(ipath);
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, ipath, gpa, .limited(1 << 28));
    defer gpa.free(raw);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, raw, .{});
    defer parsed.deinit();
    const wm = (parsed.value.object.get("weight_map") orelse return error.BadIndex).object;

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const al = arena.allocator();
    var files: std.ArrayList(st.File) = .empty;
    var file_names: std.ArrayList([]const u8) = .empty;
    defer {
        for (files.items) |*f| f.close(io);
        files.deinit(gpa);
        file_names.deinit(gpa);
    }

    const Local = struct {
        fn find(gpa_: std.mem.Allocator, io_: std.Io, dir: []const u8, wm_: std.json.ObjectMap, files_: *std.ArrayList(st.File), names_: *std.ArrayList([]const u8), al_: std.mem.Allocator, name: []const u8) !st.Tensor {
            const fv = wm_.get(name) orelse {
                std.log.err("vision tower: {s} is not in the checkpoint", .{name});
                return error.MissingVisionTensor;
            };
            const fname = fv.string;
            for (names_.items, 0..) |n, i| if (std.mem.eql(u8, n, fname)) return files_.items[i].get(name) orelse error.MissingVisionTensor;
            const p = try std.fs.path.join(al_, &.{ dir, fname });
            var f = try st.File.open(gpa_, io_, p);
            errdefer f.close(io_);
            try files_.append(gpa_, f);
            try names_.append(gpa_, try al_.dupe(u8, fname));
            return files_.items[files_.items.len - 1].get(name) orelse error.MissingVisionTensor;
        }
    };
    var total: usize = 0;
    const Req = struct {
        name: []const u8,
        shape: [5]usize,
        rank: usize,
    };
    var reqs: std.ArrayList(Req) = .empty;
    defer reqs.deinit(gpa);
    const add = struct {
        fn f(r: *std.ArrayList(Req), gpa_: std.mem.Allocator, al_: std.mem.Allocator, comptime fmt: []const u8, args: anytype, shape: []const usize) !void {
            var req: Req = .{ .name = try std.fmt.allocPrint(al_, fmt, args), .shape = @splat(1), .rank = shape.len };
            @memcpy(req.shape[0..shape.len], shape);
            try r.append(gpa_, req);
        }
    }.f;
    const V = "model.visual.";
    try add(&reqs, gpa, al, V ++ "patch_embed.proj.weight", .{}, &.{ 1024, 3, 2, 14, 14 });
    try add(&reqs, gpa, al, V ++ "patch_embed.proj.bias", .{}, &.{1024});
    for (0..cfg.depth) |i| {
        try add(&reqs, gpa, al, V ++ "blocks.{d}.norm1.weight", .{i}, &.{1024});
        try add(&reqs, gpa, al, V ++ "blocks.{d}.attn.qkv.weight", .{i}, &.{ 3072, 1024 });
        try add(&reqs, gpa, al, V ++ "blocks.{d}.attn.qkv.bias", .{i}, &.{3072});
        try add(&reqs, gpa, al, V ++ "blocks.{d}.attn.q_norm.weight", .{i}, &.{64});
        try add(&reqs, gpa, al, V ++ "blocks.{d}.attn.k_norm.weight", .{i}, &.{64});
        try add(&reqs, gpa, al, V ++ "blocks.{d}.attn.proj.weight", .{i}, &.{ 1024, 1024 });
        try add(&reqs, gpa, al, V ++ "blocks.{d}.attn.proj.bias", .{i}, &.{1024});
        try add(&reqs, gpa, al, V ++ "blocks.{d}.norm2.weight", .{i}, &.{1024});
        try add(&reqs, gpa, al, V ++ "blocks.{d}.mlp.gate_proj.weight", .{i}, &.{ cfg.inter, 1024 });
        try add(&reqs, gpa, al, V ++ "blocks.{d}.mlp.gate_proj.bias", .{i}, &.{cfg.inter});
        try add(&reqs, gpa, al, V ++ "blocks.{d}.mlp.up_proj.weight", .{i}, &.{ cfg.inter, 1024 });
        try add(&reqs, gpa, al, V ++ "blocks.{d}.mlp.up_proj.bias", .{i}, &.{cfg.inter});
        try add(&reqs, gpa, al, V ++ "blocks.{d}.mlp.down_proj.weight", .{i}, &.{ 1024, cfg.inter });
        try add(&reqs, gpa, al, V ++ "blocks.{d}.mlp.down_proj.bias", .{i}, &.{1024});
    }
    try add(&reqs, gpa, al, V ++ "post_layernorm.weight", .{}, &.{1024});
    try add(&reqs, gpa, al, V ++ "downsample.weight", .{}, &.{ 4096, 1024, 2, 2 });
    try add(&reqs, gpa, al, V ++ "downsample.bias", .{}, &.{4096});
    try add(&reqs, gpa, al, V ++ "merger.proj.weight", .{}, &.{ 4096, 4096 });
    try add(&reqs, gpa, al, V ++ "merger.post_projection_norm.weight", .{}, &.{4096});
    try add(&reqs, gpa, al, V ++ "merger.post_projection_norm.bias", .{}, &.{4096});
    try add(&reqs, gpa, al, V ++ "merger.gate_proj.weight", .{}, &.{ cfg.proj_inter, 4096 });
    try add(&reqs, gpa, al, V ++ "merger.up_proj.weight", .{}, &.{ cfg.proj_inter, 4096 });
    try add(&reqs, gpa, al, V ++ "merger.down_proj.weight", .{}, &.{ 4096, cfg.proj_inter });

    var offsets = try gpa.alloc(usize, reqs.items.len);
    defer gpa.free(offsets);
    var tensors = try gpa.alloc(st.Tensor, reqs.items.len);
    defer gpa.free(tensors);
    for (reqs.items, 0..) |r, i| {
        const t = try Local.find(gpa, io, model_dir, wm, &files, &file_names, al, r.name);
        if (!t.is(.bf16, r.shape[0..r.rank])) {
            std.log.err("vision tower: {s} is not bf16 with the expected shape", .{r.name});
            return error.BadVisionTensor;
        }
        tensors[i] = t;
        offsets[i] = total;
        total += (t.bytes.len + 255) / 256 * 256;
    }
    var buf = try cuda.DeviceBuffer.alloc(d, total);
    errdefer buf.free();
    for (tensors, offsets) |t, off| try buf.upload(off, t.bytes);

    const blocks = try gpa.alloc(Block, cfg.depth);
    errdefer gpa.free(blocks);
    var at: usize = 0;
    const P = struct {
        fn next(b: cuda.DeviceBuffer, o: []const usize, i: *usize) u64 {
            const v = b.ptr + o[i.*];
            i.* += 1;
            return v;
        }
    };
    var w: Weights = undefined;
    w.buf = buf;
    w.bytes = total;
    w.embed_w = P.next(buf, offsets, &at);
    w.embed_b = P.next(buf, offsets, &at);
    for (blocks) |*b| {
        b.norm1 = P.next(buf, offsets, &at);
        b.qkv_w = P.next(buf, offsets, &at);
        b.qkv_b = P.next(buf, offsets, &at);
        b.q_norm = P.next(buf, offsets, &at);
        b.k_norm = P.next(buf, offsets, &at);
        b.proj_w = P.next(buf, offsets, &at);
        b.proj_b = P.next(buf, offsets, &at);
        b.norm2 = P.next(buf, offsets, &at);
        b.gate_w = P.next(buf, offsets, &at);
        b.gate_b = P.next(buf, offsets, &at);
        b.up_w = P.next(buf, offsets, &at);
        b.up_b = P.next(buf, offsets, &at);
        b.down_w = P.next(buf, offsets, &at);
        b.down_b = P.next(buf, offsets, &at);
    }
    w.blocks = blocks;
    w.post_norm = P.next(buf, offsets, &at);
    w.down_w = P.next(buf, offsets, &at);
    w.down_b = P.next(buf, offsets, &at);
    w.merger_proj = P.next(buf, offsets, &at);
    w.merger_norm_w = P.next(buf, offsets, &at);
    w.merger_norm_b = P.next(buf, offsets, &at);
    w.merger_gate = P.next(buf, offsets, &at);
    w.merger_up = P.next(buf, offsets, &at);
    w.merger_down = P.next(buf, offsets, &at);
    std.debug.assert(at == reqs.items.len);
    return w;
}

fn freeWeights(gpa: std.mem.Allocator, w: *Weights) void {
    gpa.free(w.blocks);
    w.buf.free();
}

test "item geometry" {
    const it: Item = .{ .frames = 1, .h = 56, .w = 84, .pixels = &.{} };
    try std.testing.expectEqual(@as(usize, 6), it.rows());
    const clip: Item = .{ .frames = 4, .h = 56, .w = 56, .pixels = &.{} };
    try std.testing.expectEqual(@as(usize, 8), clip.rows());
}

test "embed maps cover the 1176 patch values once" {
    const gpa = std.testing.allocator;
    for (std.enums.values(EmbedOrder)) |ord| {
        const m = try embedMaps(gpa, ord);
        defer gpa.free(m.a);
        defer gpa.free(m.w);
        var seen: [1176]u8 = @splat(0);
        for (m.a) |v| if (v >= 0) {
            seen[@intCast(v)] += 1;
        };
        for (seen) |c| try std.testing.expectEqual(@as(u8, 1), c);
        try std.testing.expect(m.a.len % 16 == 0);
    }
}
