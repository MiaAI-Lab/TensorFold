//! Prepared GLM decode weights: glm5_next/cuda/weights.py load, qmm.py quantize4_mse/make_f8.
//! Ash Hart and TensorFold contributors. Rank slicing uses cuda_weights.Plan, no second slicing policy.
//! MSE clipping reduction parity is P2/P3's outstanding oracle qualification; caller selects a qualified quantizer.
const std = @import("std");
const cuda = @import("cuda");
const plan_mod = @import("cuda_weights.zig");
const quant = @import("cuda_quantize.zig");
const forward = @import("cuda_forward.zig");
const mm = @import("cuda_matmuls.zig");
const SliceSpans = @import("cuda_upload.zig").SliceSpans;
pub const Quantizer = *const fn ([]const u16, usize, usize, []u32, []u16, []u16) anyerror!void;
/// qmm.py:draft_quantize4's minmax default, for the proposal-only head.
pub fn minmax(x: []const u16, n: usize, k: usize, w: []u32, s: []u16, b: []u16) !void {
    quant.quantize4(x, n, k, w, s, b, false);
}
/// Explicit opt-in for qualification only; not silently used by the production loader.
pub fn candidateMse(x: []const u16, n: usize, k: usize, w: []u32, s: []u16, b: []u16) !void {
    quant.quantize4(x, n, k, w, s, b, true);
}
fn addNs(field: *u64, t0: std.Io.Timestamp, io: std.Io) void {
    _ = @atomicRmw(u64, field, .Add, @intCast(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds), .monotonic);
}
threadlocal var single: bool = false;
/// Number of load threads: TF_GLM_LOAD_THREADS, else min(cpus, 16).
pub fn loadThreads() usize {
    if (single) return 1;
    if (std.c.getenv("TF_GLM_LOAD_THREADS")) |v| {
        if (std.fmt.parseInt(usize, std.mem.span(v), 10)) |n| return @max(n, 1) else |_| {}
    }
    return @min(std.Thread.getCpuCount() catch 4, 16);
}
/// Runs `f(ctx, i)` for i in 0..n on up to `threads` threads (dynamic claiming). Returns the first error.
/// `d` (optional) makes the caller's CUDA context current on each worker.
fn parallelFor(comptime Ctx: type, ctx: *Ctx, n: usize, threads: usize, d: ?*const cuda.Driver, comptime f: fn (*Ctx, usize) anyerror!void) !void {
    const Shared = struct {
        ctx: *Ctx,
        n: usize,
        next: std.atomic.Value(usize) = .init(0),
        failed: std.atomic.Value(u32) = .init(0),
        first: anyerror = error.Unexpected,
        cuda_ctx: cuda.abi.Context = null,
        d: ?*const cuda.Driver,
        fn work(sh: *@This()) void {
            if (sh.d) |dd| _ = dd.api.cuCtxSetCurrent(sh.cuda_ctx);
            while (sh.failed.load(.monotonic) == 0) {
                const i = sh.next.fetchAdd(1, .monotonic);
                if (i >= sh.n) return;
                f(sh.ctx, i) catch |e| {
                    if (sh.failed.cmpxchgStrong(0, 1, .monotonic, .monotonic) == null) sh.first = e;
                    return;
                };
            }
        }
    };
    var sh: Shared = .{ .ctx = ctx, .n = n, .d = d };
    if (d) |dd| _ = dd.api.cuCtxGetCurrent(&sh.cuda_ctx);
    const count = @min(threads, n);
    if (count <= 1) {
        sh.d = null;
        sh.work();
    } else {
        var pool: [64]?std.Thread = @splat(null);
        const extra = @min(count - 1, pool.len);
        for (pool[0..extra]) |*t| t.* = std.Thread.spawn(.{}, Shared.work, .{&sh}) catch null;
        sh.work();
        for (pool[0..extra]) |t| if (t) |th| th.join();
    }
    if (sh.failed.load(.monotonic) != 0) return sh.first;
}
/// candidateMse split across rows on all load threads; every 64-wide group is independent, so the bits are identical.
pub fn parallelMse(x: []const u16, n: usize, k: usize, w: []u32, s: []u16, b: []u16) !void {
    const rows_per = 8;
    const Job = struct {
        x: []const u16,
        n: usize,
        k: usize,
        w: []u32,
        s: []u16,
        b: []u16,
        fn run(j: *@This(), c: usize) anyerror!void {
            const r0 = c * rows_per;
            const r1 = @min(j.n, r0 + rows_per);
            const kg = j.k / 64;
            quant.quantize4(j.x[r0 * j.k .. r1 * j.k], r1 - r0, j.k, j.w[r0 * j.k / 8 .. r1 * j.k / 8], j.s[r0 * kg .. r1 * kg], j.b[r0 * kg .. r1 * kg], true);
        }
    };
    var job: Job = .{ .x = x, .n = n, .k = k, .w = w, .s = s, .b = b };
    try parallelFor(Job, &job, (n + rows_per - 1) / rows_per, loadThreads(), null, Job.run);
}
pub const Prepared = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    plan: *const plan_mod.Plan,
    dir: []const u8,
    tensors: std.StringHashMapUnmanaged(usize) = .empty,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    weights: forward.Weights = undefined,
    quantizer: Quantizer,
    ns_read: u64 = 0,
    lock: std.atomic.Value(u32) = .init(0),
    ns_quant: u64 = 0,
    ns_pack: u64 = 0,
    ns_qread: u64 = 0,
    ns_head: u64 = 0,
    ns_put: u64 = 0,
    shared_layers: std.atomic.Value(u32) = .init(0),
    fn acquire(p: *Prepared) void {
        while (p.lock.cmpxchgWeak(0, 1, .acquire, .monotonic) != null) std.atomic.spinLoopHint();
    }
    fn release(p: *Prepared) void {
        p.lock.store(0, .release);
    }
    pub fn deinit(p: *Prepared) void {
        for (p.buffers.items) |*b| b.free();
        p.buffers.deinit(p.gpa);
        p.tensors.deinit(p.gpa);
    }
    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, plan: *const plan_mod.Plan, dir: []const u8, quantizer: ?Quantizer) !Prepared {
        var p: Prepared = .{ .gpa = gpa, .io = io, .d = d, .plan = plan, .dir = dir, .quantizer = quantizer orelse return error.MseQuantizerNotQualified };
        errdefer p.deinit();
        p.weights.mtp = null;
        for (plan.items.items, 0..) |record, i| if (record.src.source == .model) try p.tensors.put(gpa, record.src.name, i);
        p.weights.embed = try p.tensor("embed_tokens.weight", .bf16);
        p.weights.norm = try p.tensor("norm.weight", .bf16);
        const th = std.Io.Timestamp.now(io, .awake);
        const head_item = try p.item("lm_head.weight");
        const head = try p.bf16(head_item);
        defer gpa.free(head);
        const codes = try gpa.alloc(u8, head.len);
        defer gpa.free(codes);
        const scales = try gpa.alloc(f32, head.len / 128);
        defer gpa.free(scales);
        quant.makeF8(head, codes, scales);
        p.weights.head = try p.put(codes);
        p.weights.head_scale = try p.put(std.mem.sliceAsBytes(scales));
        p.ns_head = @intCast(th.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds);
        const g = try @import("cuda_geometry.zig").Geometry.init(plan.world, plan.rank);
        const Layers = struct {
            p: *Prepared,
            g: @import("cuda_geometry.zig").Geometry,
            fn run(j: *@This(), li: usize) anyerror!void {
                // inside a layer worker the nested loops (experts, MSE, pack) run on this thread
                single = true;
                defer single = false;
                try j.p.loadLayer(j.g, li);
            }
        };
        var layers: Layers = .{ .p = &p, .g = g };
        try parallelFor(Layers, &layers, p.weights.layers.len, loadThreads(), d, Layers.run);
        std.log.info("experts rotate each row's input once (shared gate/up scale) in {d} MoE layers", .{p.shared_layers.load(.monotonic)});
        if (plan.mtp) {
            p.weights.mtp = .{ .enorm = try p.tensor("layers.45.enorm.weight", .bf16), .hnorm = try p.tensor("layers.45.hnorm.weight", .bf16), .eh = try p.q4("layers.45.", &.{"eh_proj"}), .norm = try p.tensor("layers.45.shared_head.norm.weight", .bf16), .layer = undefined, .head = undefined };
            try p.loadLayer(g, 45);
            const mode = if (std.c.getenv("TF_GLM_DRAFT_QUANT")) |v| std.mem.span(v) else "minmax";
            if (mode.len != 0 and !std.mem.eql(u8, mode, "minmax") and !std.mem.eql(u8, mode, "mse")) return error.BadDraftQuant;
            const saved = p.quantizer;
            p.quantizer = if (std.mem.eql(u8, mode, "mse")) saved else minmax;
            defer p.quantizer = saved;
            p.weights.mtp.?.head = try p.q4("", &.{"lm_head"});
        }
        return p;
    }
    fn loadLayer(p: *Prepared, g: @import("cuda_geometry.zig").Geometry, li: usize) !void {
        const gpa = p.gpa;
        const layer = if (li == 45) &p.weights.mtp.?.layer else &p.weights.layers[li];
        var prefix_buf: [96]u8 = undefined;
        const prefix = try std.fmt.bufPrint(&prefix_buf, "layers.{d}.", .{li});
        layer.attn = if (li == 45) .{ .fn_w = 0, .base = 0, .scale = 0, .norm = try p.at(prefix, "input_layernorm.weight", .bf16) } else try p.hc(prefix, "attn", "input_layernorm.weight");
        layer.ffn = if (li == 45) .{ .fn_w = 0, .base = 0, .scale = 0, .norm = try p.at(prefix, "post_attention_layernorm.weight", .bf16) } else try p.hc(prefix, "ffn", "post_attention_layernorm.weight");
        var attn_buf: [128]u8 = undefined;
        const attn = try std.fmt.bufPrint(&attn_buf, "{s}self_attn.", .{prefix});
        if (li != 45 and li % 4 != 3) {
            layer.mixer = .{ .kda = .{
                .proj = try p.q4(attn, &.{ "q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj" }),
                .fb = try p.q4(attn, &.{"f_b_proj"}),
                .gb = try p.q4(attn, &.{"g_b_proj"}),
                .o = try p.q4(attn, &.{"o_proj"}),
                .fa_off = 3 * g.lin_heads * 128,
                .ga_off = 3 * g.lin_heads * 128 + 128,
                .b_off = 3 * g.lin_heads * 128 + 256,
                .conv = try p.stackTensors(attn, &.{ "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight" }),
                .a_log = try p.at(attn, "A_log", .f32),
                .dt_bias = try p.at(attn, "dt_bias", .f32),
                .norm = try p.at(attn, "o_norm.weight", .bf16),
            } };
        } else {
            const kv_item = try p.namedItem(attn, "kv_b_proj.weight");
            const kv = try p.bf16(kv_item);
            defer gpa.free(kv);
            if (kv.len != g.heads * 512 * 512) return error.BadWeightShape;
            const wk = try gpa.alloc(u16, kv.len / 2);
            defer gpa.free(wk);
            const wv = try gpa.alloc(u16, kv.len / 2);
            defer gpa.free(wv);
            for (0..g.heads) |h| {
                @memcpy(wk[h * 256 * 512 ..][0 .. 256 * 512], kv[h * 512 * 512 ..][0 .. 256 * 512]);
                @memcpy(wv[h * 256 * 512 ..][0 .. 256 * 512], kv[h * 512 * 512 + 256 * 512 ..][0 .. 256 * 512]);
            }
            layer.mixer = .{ .dsa = .{
                .proj = try p.q4(attn, &.{ "q_a_proj", "kv_a_proj_with_mqa" }),
                .qb = try p.q4(attn, &.{"q_b_proj"}),
                .o = try p.q4(attn, &.{"o_proj"}),
                .kw = try p.q4(attn, &.{ "indexer.wk", "indexer.weights_proj" }),
                .iqb = try p.q4(attn, &.{"indexer.wq_b"}),
                .q_norm = try p.at(attn, "q_a_layernorm.weight", .bf16),
                .kv_norm = try p.at(attn, "kv_a_layernorm.weight", .bf16),
                .wk = try p.put(std.mem.sliceAsBytes(wk)),
                .wv = try p.put(std.mem.sliceAsBytes(wv)),
                .gate = try p.at(attn, "indexer.index_kpool_compress_gate", .bf16),
                .ln_w = try p.at(attn, "indexer.k_norm.weight", .bf16),
                .ln_b = try p.at(attn, "indexer.k_norm.bias", .bf16),
                .ape = try p.at(attn, "indexer.index_kpool_compress_ape", .bf16),
            } };
        }
        var mlp_buf: [128]u8 = undefined;
        const mlp_prefix = try std.fmt.bufPrint(&mlp_buf, "{s}mlp.", .{prefix});
        if (li < 3) layer.mlp = .{ .dense = try p.mlp(mlp_prefix) } else {
            var shared_buf: [160]u8 = undefined;
            const shared_prefix = try std.fmt.bufPrint(&shared_buf, "{s}shared_experts.", .{mlp_prefix});
            const suh_g = try p.expertStackN(mlp_prefix, "gate_proj.suh");
            const suh_u = try p.expertStackN(mlp_prefix, "up_proj.suh");
            layer.mlp = .{ .moe = .{
                .router = try p.at(mlp_prefix, "gate.weight", .bf16),
                .bias = try p.at(mlp_prefix, "gate.e_score_correction_bias", .f32),
                .shared = try p.mlp(shared_prefix),
                .experts = .{
                    .gt = try p.expertStack(mlp_prefix, "gate_proj.trellis"),
                    .ut = try p.expertStack(mlp_prefix, "up_proj.trellis"),
                    .dt = try p.expertStack(mlp_prefix, "down_proj.trellis"),
                    .suh_g = suh_g.ptr,
                    .suh_u = suh_u.ptr,
                    .shared_suh = if (shareSuh()) try p.sharedSuh(suh_g, suh_u) else 0,
                    .svh_g = try p.expertStack(mlp_prefix, "gate_proj.svh"),
                    .svh_u = try p.expertStack(mlp_prefix, "up_proj.svh"),
                    .suh_d = try p.expertStack(mlp_prefix, "down_proj.suh"),
                    .svh_d = try p.expertStack(mlp_prefix, "down_proj.svh"),
                    .width = g.moe_width,
                    .count = 288,
                },
            } };
            // shared_suh=0 uses the exact per-pair rotation; detect-and-share is a later load optimization.
        }
    }
    const Type = enum { bf16, f32 };
    fn item(p: *const Prepared, name: []const u8) !plan_mod.Item {
        if (p.tensors.get(name)) |i| return p.plan.items.items[i];
        var buf: [256]u8 = undefined;
        const full = try std.fmt.bufPrint(&buf, "model.language_model.{s}", .{name});
        const i = p.tensors.get(full) orelse return error.MissingWeight;
        return p.plan.items.items[i];
    }
    fn namedItem(p: *const Prepared, prefix: []const u8, name: []const u8) !plan_mod.Item {
        var buf: [256]u8 = undefined;
        return p.item(try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, name }));
    }
    fn read(p: *const Prepared, it: plan_mod.Item, out: []u8) !void {
        const spans = try SliceSpans.init(it);
        if (out.len != it.slice.device_bytes) return error.BadWeightShape;
        const path = try std.fs.path.join(p.gpa, &.{ p.dir, it.src.file });
        defer p.gpa.free(path);
        const file = try std.Io.Dir.cwd().openFile(p.io, path, .{});
        defer file.close(p.io);
        const bytes: usize = @intCast(spans.bytes);
        const window = try p.gpa.alloc(u8, @intCast(@min(@as(u64, 32 << 20), @max(spans.count -| 1, 0) * spans.file_stride + spans.bytes)));
        defer p.gpa.free(window);
        var i: u64 = 0;
        while (i < spans.count) {
            if (bytes > out.len / 2 or spans.count == 1) {
                if (try file.readPositionalAll(p.io, out[i * bytes ..][0..bytes], spans.file_begin + i * spans.file_stride) != bytes) return error.TruncatedWeight;
                i += 1;
            } else i += try spans.readRun(p.io, file, i, out[i * bytes ..], window);
        }
    }
    fn bf16(p: *const Prepared, it: plan_mod.Item) ![]u16 {
        if (!std.mem.eql(u8, it.src.dtype, "BF16")) return error.ExpectedBf16Weight;
        const out = try p.gpa.alloc(u16, @intCast(it.slice.device_bytes / 2));
        errdefer p.gpa.free(out);
        try p.read(it, std.mem.sliceAsBytes(out));
        return out;
    }
    fn put(p: *Prepared, bytes: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.fromHost(p.d, bytes);
        errdefer b.free();
        p.acquire();
        defer p.release();
        try p.buffers.append(p.gpa, b);
        return b.ptr;
    }
    fn tensor(p: *Prepared, name: []const u8, ty: Type) !u64 {
        const it = try p.item(name);
        if (ty == .bf16) {
            const raw = try p.bf16(it);
            defer p.gpa.free(raw);
            return p.put(std.mem.sliceAsBytes(raw));
        }
        if (std.mem.eql(u8, it.src.dtype, "F32")) {
            const raw = try p.gpa.alloc(u8, @intCast(it.slice.device_bytes));
            defer p.gpa.free(raw);
            try p.read(it, raw);
            return p.put(raw);
        }
        const raw = try p.bf16(it);
        defer p.gpa.free(raw);
        const floats = try p.gpa.alloc(f32, raw.len);
        defer p.gpa.free(floats);
        for (raw, floats) |b, *f| f.* = quant.bf16ToF32(b);
        return p.put(std.mem.sliceAsBytes(floats));
    }
    fn at(p: *Prepared, prefix: []const u8, name: []const u8, ty: Type) !u64 {
        var buf: [256]u8 = undefined;
        return p.tensor(try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, name }), ty);
    }
    fn hc(p: *Prepared, prefix: []const u8, site: []const u8, norm: []const u8) !@import("cuda_hc.zig").Weights {
        var buf: [64]u8 = undefined;
        const fn_name = try std.fmt.bufPrint(&buf, "hc_{s}_fn", .{site});
        const fn_w = try p.at(prefix, fn_name, .bf16);
        const base_name = try std.fmt.bufPrint(&buf, "hc_{s}_base", .{site});
        const base = try p.at(prefix, base_name, .f32);
        const scale_name = try std.fmt.bufPrint(&buf, "hc_{s}_scale", .{site});
        const scale = try p.at(prefix, scale_name, .f32);
        return .{ .fn_w = fn_w, .base = base, .scale = scale, .norm = try p.at(prefix, norm, .bf16) };
    }
    fn q4(p: *Prepared, prefix: []const u8, names: []const []const u8) !mm.Q4 {
        var count: usize = 0;
        var k: usize = 0;
        for (names) |name| {
            var buf: [128]u8 = undefined;
            const it = try p.namedItem(prefix, try std.fmt.bufPrint(&buf, "{s}.weight", .{name}));
            if (it.slice.rank != 2 or (k != 0 and k != it.slice.shape[1])) return error.BadWeightShape;
            k = it.slice.shape[1];
            count += it.slice.shape[0];
        }
        if (k % 64 != 0) return error.BadWeightShape;
        const x = try p.gpa.alloc(u16, count * k);
        defer p.gpa.free(x);
        var at_row: usize = 0;
        const tr = std.Io.Timestamp.now(p.io, .awake);
        defer addNs(&p.ns_qread, tr, p.io);
        for (names) |name| {
            var buf: [128]u8 = undefined;
            const it = try p.namedItem(prefix, try std.fmt.bufPrint(&buf, "{s}.weight", .{name}));
            if (!std.mem.eql(u8, it.src.dtype, "BF16")) return error.ExpectedBf16Weight;
            const n = it.slice.shape[0];
            try p.read(it, std.mem.sliceAsBytes(x[at_row * k ..][0 .. n * k]));
            at_row += n;
        }
        const w = try p.gpa.alloc(u32, count * k / 8);
        defer p.gpa.free(w);
        const scales = try p.gpa.alloc(u16, count * k / 64);
        defer p.gpa.free(scales);
        const biases = try p.gpa.alloc(u16, scales.len);
        defer p.gpa.free(biases);
        const tq = std.Io.Timestamp.now(p.io, .awake);
        try p.quantizer(x, count, k, w, scales, biases);
        addNs(&p.ns_quant, tq, p.io);
        const npad = (count + 127) / 128 * 128;
        const packed_words = try p.gpa.alloc(u32, npad * k / 8);
        defer p.gpa.free(packed_words);
        const ps = try p.gpa.alloc(u16, npad * k / 64);
        defer p.gpa.free(ps);
        const pb = try p.gpa.alloc(u16, ps.len);
        defer p.gpa.free(pb);
        const tp = std.Io.Timestamp.now(p.io, .awake);
        @memset(packed_words, 0);
        @memset(ps, 0);
        @memset(pb, 0);
        {
            const Job = struct {
                w: []const u32,
                s: []const u16,
                b: []const u16,
                n: usize,
                k: usize,
                o: []u32,
                so: []u16,
                bo: []u16,
                fn run(j: *@This(), c: usize) anyerror!void {
                    quant.packRows(j.w, j.s, j.b, j.n, j.k, j.o, j.so, j.bo, c * 16, @min(j.n, c * 16 + 16));
                }
            };
            var job: Job = .{ .w = w, .s = scales, .b = biases, .n = count, .k = k, .o = packed_words, .so = ps, .bo = pb };
            try parallelFor(Job, &job, (count + 15) / 16, loadThreads(), null, Job.run);
        }
        addNs(&p.ns_pack, tp, p.io);
        return .{ .w = try p.put(std.mem.sliceAsBytes(packed_words)), .s = try p.put(std.mem.sliceAsBytes(ps)), .b = try p.put(std.mem.sliceAsBytes(pb)), .n = count, .k = k };
    }
    fn mlp(p: *Prepared, prefix: []const u8) !forward.MlpW {
        return .{ .gu = try p.q4(prefix, &.{ "gate_proj", "up_proj" }), .down = try p.q4(prefix, &.{"down_proj"}) };
    }
    fn stackTensors(p: *Prepared, prefix: []const u8, names: []const []const u8) !u64 {
        var bytes: usize = 0;
        for (names) |name| bytes += @intCast((try p.namedItem(prefix, name)).slice.device_bytes);
        const raw = try p.gpa.alloc(u8, bytes);
        defer p.gpa.free(raw);
        var offset: usize = 0;
        for (names) |name| {
            const it = try p.namedItem(prefix, name);
            const n: usize = @intCast(it.slice.device_bytes);
            try p.read(it, raw[offset..][0..n]);
            offset += n;
        }
        return p.put(raw);
    }
    fn expertStack(p: *Prepared, prefix: []const u8, part: []const u8) !u64 {
        return (try p.expertStackN(prefix, part)).ptr;
    }
    /// The one input scale vector (device address) when every expert's gate and up `suh` are bitwise equal, else 0
    /// (glm5_next/cuda/exl3_mm.py shared_suh): the experts then rotate each row's input once, not once a pair.
    fn sharedSuh(p: *Prepared, g: Stack, u: Stack) !u64 {
        if (g.n != u.n or g.n == 0) return 0;
        const total = g.n * 288;
        const gb = try p.gpa.alloc(u8, total);
        defer p.gpa.free(gb);
        const ub = try p.gpa.alloc(u8, total);
        defer p.gpa.free(ub);
        try (cuda.DeviceBuffer{ .d = p.d, .ptr = g.ptr, .len = total }).download(0, gb);
        try (cuda.DeviceBuffer{ .d = p.d, .ptr = u.ptr, .len = total }).download(0, ub);
        const one = sharedRow(gb, ub, g.n) orelse return 0;
        _ = p.shared_layers.fetchAdd(1, .monotonic);
        return p.put(one);
    }
    const Stack = struct { ptr: u64, n: usize };
    fn expertStackN(p: *Prepared, prefix: []const u8, part: []const u8) !Stack {
        var buf: [128]u8 = undefined;
        const first = try p.namedItem(prefix, try std.fmt.bufPrint(&buf, "experts.0.{s}", .{part}));
        const n: usize = @intCast(first.slice.device_bytes);
        var b = try cuda.DeviceBuffer.alloc(p.d, n * 288);
        errdefer b.free();
        const te = std.Io.Timestamp.now(p.io, .awake);
        defer addNs(&p.ns_put, te, p.io);
        const Job = struct {
            p: *const Prepared,
            prefix: []const u8,
            part: []const u8,
            n: usize,
            b: cuda.DeviceBuffer,
            fn run(j: *@This(), e: usize) anyerror!void {
                var nb: [128]u8 = undefined;
                const it = try j.p.namedItem(j.prefix, try std.fmt.bufPrint(&nb, "experts.{d}.{s}", .{ e, j.part }));
                if (it.slice.device_bytes != j.n) return error.BadWeightShape;
                const raw = try j.p.gpa.alloc(u8, j.n);
                defer j.p.gpa.free(raw);
                try j.p.read(it, raw);
                try j.b.upload(e * j.n, raw);
            }
        };
        var job: Job = .{ .p = p, .prefix = prefix, .part = part, .n = n, .b = b };
        try parallelFor(Job, &job, 288, loadThreads(), p.d, Job.run);
        p.acquire();
        defer p.release();
        try p.buffers.append(p.gpa, b);
        return .{ .ptr = b.ptr, .n = n };
    }
};
/// exl3_mm.py shared_suh: the first expert's row when every expert's gate row and up row equal it bit for bit (-0 is not
/// +0, NaN payloads count), else null. `gate` and `up` are [experts, row_bytes].
pub fn sharedRow(gate: []const u8, up: []const u8, row_bytes: usize) ?[]const u8 {
    if (row_bytes == 0 or gate.len != up.len or gate.len % row_bytes != 0 or gate.len == 0) return null;
    if (!std.mem.eql(u8, gate, up)) return null;
    const first = gate[0..row_bytes];
    var at: usize = row_bytes;
    while (at < gate.len) : (at += row_bytes) if (!std.mem.eql(u8, gate[at..][0..row_bytes], first)) return null;
    return first;
}
test "shared suh needs every expert's gate and up equal bit for bit" {
    const a = [_]u8{ 1, 2, 3, 4 };
    var g: [12]u8 = undefined;
    for (0..3) |i| @memcpy(g[i * 4 ..][0..4], &a);
    var u = g;
    try std.testing.expectEqualSlices(u8, &a, sharedRow(&g, &u, 4).?);
    u[9] ^= 0x80; // one bit of the last expert's up scale (a -0 against a +0 in fp16)
    try std.testing.expect(sharedRow(&g, &u, 4) == null);
    u = g;
    g[5] ^= 1;
    u[5] ^= 1; // gate and up agree but the experts differ
    try std.testing.expect(sharedRow(&g, &u, 4) == null);
    try std.testing.expect(sharedRow(&g, &u, 5) == null);
    try std.testing.expect(sharedRow(&.{}, &.{}, 4) == null);
}
test "unqualified MSE loader fails before IO or device allocation" {
    try std.testing.expectError(error.MseQuantizerNotQualified, Prepared.load(std.testing.allocator, undefined, undefined, undefined, "", null));
    if (std.c.getenv("TF_GLM_P6_COMPILE_PROBE") != null) {
        var p = try Prepared.load(std.testing.allocator, undefined, undefined, undefined, "", candidateMse);
        p.deinit();
    }
}

/// TF_GLM_SHARED_SUH=0 keeps the per-pair input rotation (A/B).
fn shareSuh() bool {
    const v = std.c.getenv("TF_GLM_SHARED_SUH") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}
