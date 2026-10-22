//! GPU self-check of the vision tower against the Python reference's stages (zig/tests/vision/tower/gen.py writes the
//! fixtures): the full chain stage by stage, and each stage run from the reference's own input ("teacher forced"), which
//! places a first difference inside a stage instead of letting it spread through the 24 blocks.

const std = @import("std");
const cuda = @import("cuda");
const vision = @import("cuda_vision.zig");
const kernels = @import("cuda_kernels.zig");

pub const Stats = struct {
    n: usize = 0,
    identical: usize = 0,
    max_abs: f64 = 0,
    sum_abs: f64 = 0,
    sum_ref: f64 = 0,
    bad_nan: usize = 0,
    first: [4]usize = .{ 0, 0, 0, 0 }, // indices of the first four differing values
    nfirst: usize = 0,

    pub fn identicalFrac(s: Stats) f64 {
        return if (s.n == 0) 1 else @as(f64, @floatFromInt(s.identical)) / @as(f64, @floatFromInt(s.n));
    }
    pub fn meanRel(s: Stats) f64 {
        return if (s.sum_ref == 0) 0 else s.sum_abs / s.sum_ref;
    }
};

fn bf(u: u16) f32 {
    return @bitCast(@as(u32, u) << 16);
}

pub fn compare(dtype: vision.StageDType, got: []const u8, ref: []const u8) Stats {
    var s: Stats = .{};
    switch (dtype) {
        .bf16 => {
            const a = std.mem.bytesAsSlice(u16, got);
            const b = std.mem.bytesAsSlice(u16, ref);
            s.n = @min(a.len, b.len);
            for (a[0..s.n], b[0..s.n], 0..) |x, y, idx16| {
                if (x == y) s.identical += 1 else if (s.nfirst < 4) {
                    s.first[s.nfirst] = idx16;
                    s.nfirst += 1;
                }
                const fx: f64 = bf(x);
                const fy: f64 = bf(y);
                if (std.math.isNan(fx) or std.math.isNan(fy)) {
                    s.bad_nan += 1;
                    continue;
                }
                const dd = @abs(fx - fy);
                s.max_abs = @max(s.max_abs, dd);
                s.sum_abs += dd;
                s.sum_ref += @abs(fy);
            }
        },
        .f32 => {
            const a = std.mem.bytesAsSlice(f32, got);
            const b = std.mem.bytesAsSlice(f32, ref);
            s.n = @min(a.len, b.len);
            for (a[0..s.n], b[0..s.n]) |x, y| {
                if (@as(u32, @bitCast(x)) == @as(u32, @bitCast(y))) s.identical += 1;
                const dd = @abs(@as(f64, x) - @as(f64, y));
                if (std.math.isNan(dd)) {
                    s.bad_nan += 1;
                    continue;
                }
                s.max_abs = @max(s.max_abs, dd);
                s.sum_abs += dd;
                s.sum_ref += @abs(@as(f64, y));
            }
        },
    }
    return s;
}

pub const Fixture = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    dir: []const u8,
    parsed: std.json.Parsed(std.json.Value),

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Fixture {
        const path = try std.fs.path.join(gpa, &.{ dir, "manifest.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
        defer gpa.free(text);
        return .{ .gpa = gpa, .io = io, .dir = dir, .parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{}) };
    }
    pub fn deinit(f: *Fixture) void {
        f.parsed.deinit();
    }
    fn obj(f: Fixture) std.json.ObjectMap {
        return f.parsed.value.object;
    }
    pub fn int(f: Fixture, name: []const u8) usize {
        return @intCast(f.obj().get(name).?.integer);
    }
    pub fn has(f: Fixture, stage: []const u8) bool {
        for (f.obj().get("stages").?.array.items) |s| if (std.mem.eql(u8, s.object.get("name").?.string, stage)) return true;
        return false;
    }
    pub fn dtype(f: Fixture, stage: []const u8) ?vision.StageDType {
        for (f.obj().get("stages").?.array.items) |s| if (std.mem.eql(u8, s.object.get("name").?.string, stage)) {
            return if (std.mem.eql(u8, s.object.get("dtype").?.string, "float32")) .f32 else .bf16;
        };
        return null;
    }
    pub fn read(f: Fixture, stage: []const u8) ![]u8 {
        const fname = try std.fmt.allocPrint(f.gpa, "{s}.bin", .{stage});
        defer f.gpa.free(fname);
        const path = try std.fs.path.join(f.gpa, &.{ f.dir, fname });
        defer f.gpa.free(path);
        return std.Io.Dir.cwd().readFileAlloc(f.io, path, f.gpa, .limited(1 << 31));
    }
};

pub const Report = struct {
    out: *std.Io.Writer,
    pub fn line(r: Report, label: []const u8, s: Stats) void {
        r.out.print("{s:<34} n={d:<9} identical={d:>9.6} max_abs={e:>10.3} mean_rel={e:>10.3}{s}\n", .{ label, s.n, s.identicalFrac(), s.max_abs, s.meanRel(), if (s.bad_nan > 0) " NaN!" else "" }) catch {};
        r.out.flush() catch {};
    }
};

const ChainCtx = struct {
    c: *Checker,
    pass: usize = 0,
    prefix: []const u8 = "",
};

pub const Checker = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    fx: Fixture,
    rep: Report,
    scratch: std.ArrayList(u8) = .empty,
    worst_block: usize = 0,
    sweep: bool = false,
    detail: bool = false,

    fn download(c: *Checker, ptr: u64, bytes: usize) ![]u8 {
        try c.scratch.resize(c.gpa, bytes);
        const view: cuda.DeviceBuffer = .{ .d = c.d, .ptr = ptr, .len = bytes };
        try view.download(0, c.scratch.items);
        return c.scratch.items;
    }

    /// Compares a device array with a fixture stage and prints one table line.
    pub fn check(c: *Checker, label: []const u8, stage: []const u8, ptr: u64, count: usize, dtype: vision.StageDType) !?Stats {
        if (!c.fx.has(stage)) return null;
        const ref = try c.fx.read(stage);
        defer c.gpa.free(ref);
        const esz: usize = if (dtype == .f32) 4 else 2;
        const got = try c.download(ptr, @min(count * esz, ref.len));
        const s = compare(dtype, got, ref);
        c.rep.line(label, s);
        if (c.detail and s.nfirst > 0 and dtype == .bf16) {
            const a = std.mem.bytesAsSlice(u16, got);
            const b = std.mem.bytesAsSlice(u16, ref);
            for (s.first[0..s.nfirst]) |i| c.rep.out.print("    index {d} (row {d}, col {d} of 1024): got 0x{x:0>4} ref 0x{x:0>4}\n", .{ i, i / 1024, i % 1024, a[i], b[i] }) catch {};
        }
        return s;
    }

    fn hookCall(ctx: ?*anyopaque, name: []const u8, ptr: u64, count: usize, dtype: vision.StageDType) anyerror!void {
        const h: *ChainCtx = @ptrCast(@alignCast(ctx.?));
        if (std.mem.eql(u8, name, "patches")) h.pass += 1;
        var buf: [96]u8 = undefined;
        const stage = try std.fmt.bufPrint(&buf, "p{d}_{s}", .{ h.pass - 1, name });
        _ = try h.c.check(stage, stage, ptr, count, dtype);
    }

    /// The whole tower through `encode`, every stage compared as it passes.
    pub fn chain(c: *Checker, model_dir: []const u8, opts: vision.Options, canvas: []const u8, frames: usize, h: usize, w: usize) !void {
        var ctx: ChainCtx = .{ .c = c };
        var o = opts;
        o.stage_hook = .{ .ctx = &ctx, .call = hookCall };
        var t = try vision.Tower.init(c.gpa, c.io, c.d, c.stream, model_dir, o);
        defer t.deinit();
        const item: vision.Item = .{ .frames = frames, .h = h, .w = w, .pixels = canvas };
        var out = try cuda.DeviceBuffer.alloc(c.d, item.rows() * 4096 * 2);
        defer out.free();
        try t.encode(&.{item}, out.ptr);
        if (c.fx.has("features_all")) _ = try c.check("FINAL features vs features_all", "features_all", out.ptr, item.rows() * 4096, .bf16);
    }

    fn up(c: *Checker, stage: []const u8, ptr: u64) !void {
        const raw = try c.fx.read(stage);
        defer c.gpa.free(raw);
        const view: cuda.DeviceBuffer = .{ .d = c.d, .ptr = ptr, .len = raw.len };
        try view.upload(0, raw);
    }

    fn fmtz(buf: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
        return std.fmt.bufPrint(buf, fmt, args) catch unreachable;
    }

    /// Block `i`'s steps, each run from the reference's input to that step (the fixture stores b{i}_*).
    fn innerBlock(c: *Checker, t: *vision.Tower, i: usize, R: usize, S: usize) !void {
        const b = &t.b;
        const bl = &t.blas;
        const w0 = t.w.blocks[i];
        var sb: [48]u8 = undefined;
        var lb: [64]u8 = undefined;
        const block_in = if (i == 0) "p0_embed" else fmtz(&sb, "p0_block{d:0>2}", .{i - 1});
        const pre = fmtz(&lb, "p0_b{d}_", .{i});
        var s1: [64]u8 = undefined;
        var s2: [64]u8 = undefined;
        var l1: [96]u8 = undefined;
        // the step names
        try c.up(block_in, b.x.ptr);
        try t.stageRms(b.x.ptr, w0.norm1, b.h.ptr, R);
        try c.stream.synchronize();
        _ = try c.check(fmtz(&l1, "b{d} norm1", .{i}), fmtz(&s1, "{s}norm1", .{pre}), b.h.ptr, R * 1024, .bf16);
        try c.up(fmtz(&s1, "{s}norm1", .{pre}), b.h.ptr);
        try bl.linear(c.stream, b.h.ptr, w0.qkv_w, w0.qkv_b, b.qkv.ptr, R, 3072, 1024);
        try c.stream.synchronize();
        _ = try c.check(fmtz(&l1, "b{d} qkv linear (+bias)", .{i}), fmtz(&s1, "{s}qkv", .{pre}), b.qkv.ptr, R * 3072, .bf16);
        try c.up(fmtz(&s1, "{s}qkv", .{pre}), b.qkv.ptr);
        try t.stageQkNormRope(b.qkv.ptr, w0.q_norm, w0.k_norm, b.q.ptr, b.k.ptr, R, S);
        try c.stream.synchronize();
        _ = try c.check(fmtz(&l1, "b{d} q norm+rope", .{i}), fmtz(&s1, "{s}qr", .{pre}), b.q.ptr, R * 1024, .bf16);
        _ = try c.check(fmtz(&l1, "b{d} k norm+rope", .{i}), fmtz(&s1, "{s}kr", .{pre}), b.k.ptr, R * 1024, .bf16);
        try c.up(fmtz(&s1, "{s}qr", .{pre}), b.q.ptr);
        try c.up(fmtz(&s1, "{s}kr", .{pre}), b.k.ptr);
        const keep = t.opts.attention;
        for ([_]vision.Attention{ .flash2, .plain }) |att| {
            if (att == .plain and i != 0) continue;
            t.opts.attention = att;
            try t.stageAttention(b.q.ptr, b.k.ptr, b.qkv.ptr + 2048 * 2, b.o.ptr, R, S);
            try c.stream.synchronize();
            _ = try c.check(fmtz(&l1, "b{d} attention {t}", .{ i, att }), fmtz(&s1, "{s}attn", .{pre}), b.o.ptr, R * 1024, .bf16);
        }
        t.opts.attention = keep;
        if (c.sweep and (i == 0 or i == 23)) {
            const keep_v = t.opts.flash_variant;
            var v: u32 = 1;
            while (v < 32) : (v += 1) {
                t.opts.flash_variant = v;
                try t.stageAttention(b.q.ptr, b.k.ptr, b.qkv.ptr + 2048 * 2, b.o.ptr, R, S);
                try c.stream.synchronize();
                _ = try c.check(fmtz(&l1, "b{d} attention flash2 variant {d}", .{ i, v }), fmtz(&s1, "{s}attn", .{pre}), b.o.ptr, R * 1024, .bf16);
            }
            t.opts.flash_variant = keep_v;
        }
        try c.up(fmtz(&s1, "{s}attn", .{pre}), b.o.ptr);
        try bl.linear(c.stream, b.o.ptr, w0.proj_w, w0.proj_b, b.q.ptr, R, 1024, 1024);
        try c.stream.synchronize();
        _ = try c.check(fmtz(&l1, "b{d} proj linear (+bias)", .{i}), fmtz(&s1, "{s}proj", .{pre}), b.q.ptr, R * 1024, .bf16);
        try c.up(block_in, b.x.ptr);
        try c.up(fmtz(&s1, "{s}proj", .{pre}), b.q.ptr);
        try t.stageAdd(b.x.ptr, b.q.ptr, b.x.ptr, R * 1024);
        try c.stream.synchronize();
        _ = try c.check(fmtz(&l1, "b{d} residual add", .{i}), fmtz(&s1, "{s}res1", .{pre}), b.x.ptr, R * 1024, .bf16);
        try t.stageRms(b.x.ptr, w0.norm2, b.h.ptr, R);
        try c.stream.synchronize();
        _ = try c.check(fmtz(&l1, "b{d} norm2", .{i}), fmtz(&s1, "{s}norm2", .{pre}), b.h.ptr, R * 1024, .bf16);
        try c.up(fmtz(&s1, "{s}norm2", .{pre}), b.h.ptr);
        try bl.linear(c.stream, b.h.ptr, w0.gate_w, w0.gate_b, b.gate.ptr, R, 4096, 1024);
        try bl.linear(c.stream, b.h.ptr, w0.up_w, w0.up_b, b.up.ptr, R, 4096, 1024);
        try c.stream.synchronize();
        _ = try c.check(fmtz(&l1, "b{d} gate linear (+bias)", .{i}), fmtz(&s1, "{s}gate_raw", .{pre}), b.gate.ptr, R * 4096, .bf16);
        _ = try c.check(fmtz(&l1, "b{d} up linear (+bias)", .{i}), fmtz(&s2, "{s}up_raw", .{pre}), b.up.ptr, R * 4096, .bf16);
        try c.up(fmtz(&s1, "{s}gate_raw", .{pre}), b.gate.ptr);
        try c.up(fmtz(&s2, "{s}up_raw", .{pre}), b.up.ptr);
        try t.stageSwiglu(b.gate.ptr, b.up.ptr, b.gate.ptr, R * 4096);
        try c.stream.synchronize();
        _ = try c.check(fmtz(&l1, "b{d} swiglu", .{i}), fmtz(&s1, "{s}act", .{pre}), b.gate.ptr, R * 4096, .bf16);
        try c.up(fmtz(&s1, "{s}act", .{pre}), b.gate.ptr);
        try bl.linear(c.stream, b.gate.ptr, w0.down_w, w0.down_b, b.h.ptr, R, 1024, 4096);
        try c.up(fmtz(&s1, "{s}res1", .{pre}), b.x.ptr);
        try t.stageAdd(b.x.ptr, b.h.ptr, b.x.ptr, R * 1024);
        try c.stream.synchronize();
        const out_name = if (i == 23) "p0_block23" else fmtz(&s1, "p0_block{d:0>2}", .{i});
        _ = try c.check(fmtz(&l1, "b{d} down linear + add = block output", .{i}), out_name, b.x.ptr, R * 1024, .bf16);
    }

    /// Each stage of pass 0 from the reference's input to that stage.
    pub fn teacher(c: *Checker, model_dir: []const u8, opts: vision.Options, canvas: []const u8, frames: usize, h: usize, w: usize, hypotheses: bool) !void {
        var t = try vision.Tower.init(c.gpa, c.io, c.d, c.stream, model_dir, opts);
        defer t.deinit();
        const gh = h / 14;
        const gw = w / 14;
        const S = gh * gw;
        const pairs: usize = if (frames == 1) 1 else frames / 2;
        const per = @max(1, opts.pass_patches / S);
        const n = @min(per, pairs);
        const R = n * S;
        try t.ensure(R, S, canvas.len);
        const b = &t.b;
        const rep = c.rep;
        rep.out.print("-- teacher-forced stages, pass 0: {d} rows, grid {d}x{d}\n", .{ R, gh, gw }) catch {};
        // patches
        {
            try b.frames.upload(0, canvas[0 .. if (frames == 1) canvas.len else 2 * n * 3 * h * w]);
            var tmp = try cuda.DeviceBuffer.alloc(c.d, R * 1176 * 4);
            defer tmp.free();
            try t.stagePatches(b.frames.ptr, if (frames == 1) 0 else 3 * h * w, 0, n, gh, gw, h, w, b.patches.ptr, tmp.ptr);
            try c.stream.synchronize();
            _ = try c.check("patches (fp32)", "p0_patches", tmp.ptr, R * 1176, .f32);
        }
        // the rope tables from scratch
        try t.stageRope(gh, gw);
        try c.stream.synchronize();
        _ = try c.check("rope cos", "p0_rope_cos", b.rope_cos.ptr, S * 64, .f32);
        _ = try c.check("rope sin", "p0_rope_sin", b.rope_sin.ptr, S * 64, .f32);
        // embed: the fp32 patches rounded to bf16 are the conv input
        {
            const order_names = [_]vision.EmbedOrder{ .ctyx, .tyxc, .tyxc4, .tyxc8, .yxtc, .yxtc8, .yxct };
            var out = try cuda.DeviceBuffer.alloc(c.d, R * 1024 * 2);
            defer out.free();
            const list: []const vision.EmbedOrder = if (hypotheses) &order_names else &.{opts.embed_order};
            const keep_conv = t.opts.conv;
            t.opts.conv = .own;
            for ([_]vision.EmbedKind{ .seq, .mma }) |kind| {
                if (!hypotheses) break;
                t.opts.embed_kind = kind;
                for (list) |ord| {
                    try t.setEmbedOrder(ord);
                    try t.stageEmbed(b.patches.ptr, out.ptr, R, false);
                    try c.stream.synchronize();
                    var lab: [64]u8 = undefined;
                    _ = try c.check(try std.fmt.bufPrint(&lab, "embed (no bias) {t} order={t}", .{ kind, ord }), "p0_embed_nobias", out.ptr, R * 1024, .bf16);
                }
            }
            t.opts.embed_kind = opts.embed_kind;
            t.opts.conv = keep_conv;
            try t.setEmbedOrder(opts.embed_order);
            try t.stageEmbed(b.patches.ptr, b.x.ptr, R, true);
            try c.stream.synchronize();
            var lab2: [96]u8 = undefined;
            _ = try c.check(fmtz(&lab2, "embed (+bias) conv={t} cuDNN engine {d}", .{ t.opts.conv, try t.convEngine(false, R) }), "p0_embed", b.x.ptr, R * 1024, .bf16);
        }
        // block 0 (and any block whose inner steps are stored) piece by piece, each from the reference's input
        var bi: usize = 0;
        while (bi < t.cfg.depth) : (bi += 1) {
            var nbuf: [32]u8 = undefined;
            if (c.fx.has(try std.fmt.bufPrint(&nbuf, "p0_b{d}_norm1", .{bi}))) try c.innerBlock(&t, bi, R, S);
        }
        const bl = &t.blas;
        // blocks, each from the reference's previous output
        const all = c.fx.has("p0_block01");
        const last: usize = if (all) 23 else 0;
        var i: usize = 0;
        var buf: [32]u8 = undefined;
        while (i <= last) : (i += 1) {
            const src = if (i == 0) "p0_embed" else try std.fmt.bufPrint(&buf, "p0_block{d:0>2}", .{i - 1});
            try c.up(src, b.x.ptr);
            try t.stageBlock(i, R, S);
            try c.stream.synchronize();
            var nbuf: [32]u8 = undefined;
            const stage = try std.fmt.bufPrint(&nbuf, "p0_block{d:0>2}", .{i});
            var lab: [48]u8 = undefined;
            _ = try c.check(try std.fmt.bufPrint(&lab, "block {d:0>2} from reference input", .{i}), stage, b.x.ptr, R * 1024, .bf16);
        }
        if (!all) {
            for ([_]usize{ 12, 23 }) |blk| {
                var nbuf: [32]u8 = undefined;
                var nbuf2: [32]u8 = undefined;
                const src = try std.fmt.bufPrint(&nbuf, "p0_block{d:0>2}", .{blk - 1});
                if (!c.fx.has(src)) continue; // only blocks 0, 12 and 23 are stored
                try c.up(src, b.x.ptr);
                try t.stageBlock(blk, R, S);
                try c.stream.synchronize();
                _ = try c.check("block from reference input", try std.fmt.bufPrint(&nbuf2, "p0_block{d:0>2}", .{blk}), b.x.ptr, R * 1024, .bf16);
            }
        }
        // the head
        try c.up("p0_block23", b.x.ptr);
        const M = R / 4;
        try t.stageRms(b.x.ptr, t.w.post_norm, b.h.ptr, R);
        try c.stream.synchronize();
        _ = try c.check("post norm", "p0_post_norm", b.h.ptr, R * 1024, .bf16);
        try c.up("p0_post_norm", b.h.ptr);
        {
            var lab: [96]u8 = undefined;
            try t.stageDownsample(b.h.ptr, b.o.ptr, M, false);
            try c.stream.synchronize();
            _ = try c.check(fmtz(&lab, "downsample (no bias) conv={t} cuDNN engine {d}", .{ t.opts.conv, try t.convEngine(true, M) }), "p0_downsample_nobias", b.o.ptr, M * 4096, .bf16);
        }
        if (hypotheses) {
            const keep_conv = t.opts.conv;
            t.opts.conv = .own;
            for ([_]u32{ 0, 16, 32, 64, 128, 256, 512 }) |fresh| {
                t.opts.downsample_fresh = fresh;
                try t.stageDownsample(b.h.ptr, b.o.ptr, M, false);
                try c.stream.synchronize();
                var lab: [64]u8 = undefined;
                _ = try c.check(try std.fmt.bufPrint(&lab, "downsample own chain partial={d}", .{fresh}), "p0_downsample_nobias", b.o.ptr, M * 4096, .bf16);
            }
            t.opts.conv = keep_conv;
        }
        t.opts.downsample_fresh = opts.downsample_fresh;
        try t.stageDownsample(b.h.ptr, b.o.ptr, M, true);
        try c.stream.synchronize();
        _ = try c.check("downsample (+bias)", "p0_downsample", b.o.ptr, M * 4096, .bf16);
        try c.up("p0_downsample", b.o.ptr);
        try bl.linear(c.stream, b.o.ptr, t.w.merger_proj, 0, b.q.ptr, M, 4096, 4096);
        try c.stream.synchronize();
        _ = try c.check("merger proj (cublasGemmEx)", "p0_merger_proj", b.q.ptr, M * 4096, .bf16);
        try c.up("p0_merger_proj", b.q.ptr);
        const ly = [_]u32{ 1, 2, 4, 8, 16 };
        for (ly) |rows| {
            t.opts.ln_rows = rows;
            const nv: u32 = if (c.sweep) 16 else 1;
            var v: u32 = 0;
            while (v < nv) : (v += 1) {
                t.opts.ln_variant = v;
                try t.stageLnGelu(b.q.ptr, b.k.ptr, b.o.ptr, M);
                try c.stream.synchronize();
                var lab: [64]u8 = undefined;
                _ = try c.check(try std.fmt.bufPrint(&lab, "merger layer_norm (BY={d} variant {d})", .{ rows, v }), "p0_merger_ln", b.k.ptr, M * 4096, .bf16);
                if (v == 0) _ = try c.check(try std.fmt.bufPrint(&lab, "merger gelu (BY={d})", .{rows}), "p0_merger_gelu", b.o.ptr, M * 4096, .bf16);
            }
        }
        t.opts.ln_variant = opts.ln_variant;
        t.opts.ln_rows = opts.ln_rows;
        try c.up("p0_merger_ln", b.k.ptr);
        try t.stageGelu(b.k.ptr, b.o.ptr, M * 4096);
        try c.stream.synchronize();
        _ = try c.check("gelu alone, from reference layer_norm", "p0_merger_gelu", b.o.ptr, M * 4096, .bf16);
        try c.up("p0_merger_gelu", b.q.ptr);
        const pi = t.cfg.proj_inter;
        try bl.linear(c.stream, b.q.ptr, t.w.merger_gate, 0, b.gate.ptr, M, pi, 4096);
        try bl.linear(c.stream, b.q.ptr, t.w.merger_up, 0, b.up.ptr, M, pi, 4096);
        try t.stageSwiglu(b.gate.ptr, b.up.ptr, b.gate.ptr, M * pi);
        try bl.linear(c.stream, b.gate.ptr, t.w.merger_down, 0, b.o.ptr, M, 4096, pi);
        try c.stream.synchronize();
        _ = try c.check("merger swiglu + down -> features", "p0_features", b.o.ptr, M * 4096, .bf16);
    }

    /// Time and device memory of `encode` on this input (no hook).
    pub fn bench(c: *Checker, model_dir: []const u8, opts: vision.Options, canvas: []const u8, frames: usize, h: usize, w: usize) !void {
        var o = opts;
        o.stage_hook = null;
        _ = cuda.usage(true);
        var t = try vision.Tower.init(c.gpa, c.io, c.d, c.stream, model_dir, o);
        defer t.deinit();
        const item: vision.Item = .{ .frames = frames, .h = h, .w = w, .pixels = canvas };
        var out = try cuda.DeviceBuffer.alloc(c.d, item.rows() * 4096 * 2);
        defer out.free();
        const weights = cuda.usage(false).device;
        try t.encode(&.{item}, out.ptr);
        var a = try cuda.Event.init(c.d, true);
        defer a.deinit();
        var b = try cuda.Event.init(c.d, true);
        defer b.deinit();
        try a.record(c.stream);
        const runs: usize = 3;
        for (0..runs) |_| try t.encode(&.{item}, out.ptr);
        try b.record(c.stream);
        try b.synchronize();
        const ms = try cuda.Event.elapsedMs(a, b) / @as(f32, @floatFromInt(runs));
        // a second, profiled pass
        t.opts.profile = true;
        try t.encode(&.{item}, out.ptr);
        try t.profileFlush();
        t.opts.profile = false;
        var it = t.prof_ms.iterator();
        while (it.next()) |e| c.rep.out.print("  profile {s:<44} {d:>8.2} ms\n", .{ e.key_ptr.*, e.value_ptr.* }) catch {};
        const u = cuda.usage(false);
        c.rep.out.print("bench: {d:.1} ms per encode ({d} rows); device bytes: weights+tower {d:.1} MiB at start, peak {d:.1} MiB (peak beyond the weights {d:.1} MiB, the weights are {d:.1} MiB)\n", .{ ms, item.rows(), @as(f64, @floatFromInt(weights)) / 1048576, @as(f64, @floatFromInt(u.peak)) / 1048576, @as(f64, @floatFromInt(u.peak - t.w.bytes)) / 1048576, @as(f64, @floatFromInt(t.w.bytes)) / 1048576 }) catch {};
        c.rep.out.flush() catch {};
    }
};

/// Candidate formulas for torch's bf16 GELU and SiLU against tables of torch's own results at every bf16 value
/// (tests/vision/tower/gen_luts.py): prints how many of the 65536 inputs each candidate gets wrong.
pub fn actCandidates(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, stream: cuda.Stream, out: *std.Io.Writer, luts_path: []const u8) !void {
    const lut = try std.Io.Dir.cwd().readFileAlloc(io, luts_path, gpa, .limited(1 << 20));
    defer gpa.free(lut);
    if (lut.len != 4 * 65536) return error.BadLuts;
    const table = std.mem.bytesAsSlice(u16, lut);
    var mod = try kernels.load(d, .vision);
    defer mod.unload();
    const f = try mod.function("vis_act_candidates");
    var buf = try cuda.DeviceBuffer.alloc(d, 65536 * 2);
    defer buf.free();
    const got = try gpa.alloc(u16, 65536);
    defer gpa.free(got);
    const names = [_][]const u8{ "gelu fp32", "gelu erf(double) of a double argument", "gelu double, fp32 argument", "silu fp32", "silu double", "gelu 0.5x erfc(-x/sqrt2)", "gelu x normcdf(x)", "gelu fma(0.5x, erf, 0.5x)", "gelu unfused", "gelu mixed fp32 erf, double product" };
    for (names, 0..) |name, mode| {
        var a: cuda.Args = .{};
        a.add(buf.ptr);
        a.add(@as(i32, @intCast(mode)));
        try cuda.launch.launch(f, .{ .grid = .{ .x = 256 }, .block = .{ .x = 256 } }, stream, &a);
        try stream.synchronize();
        try buf.download(0, std.mem.sliceAsBytes(got));
        const ref = if (mode < 3 or mode > 4) table[65536..] else table[0..65536];
        var bad: usize = 0;
        var shown: usize = 0;
        for (got, ref, 0..) |g, r, i| {
            if (g == r) continue;
            // NaN payloads and the sign of zero are not what the model sees; count them apart
            bad += 1;
            if (shown < 4) {
                out.print("    x=0x{x:0>4} ({e}): got 0x{x:0>4} ref 0x{x:0>4}\n", .{ i, bf(@intCast(i)), g, r }) catch {};
                shown += 1;
            }
        }
        out.print("act candidate {s:<42} wrong on {d} of 65536 inputs\n", .{ name, bad }) catch {};
        out.flush() catch {};
    }
}
