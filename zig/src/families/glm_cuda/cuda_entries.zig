//! Plain launches for TensorFold glm5_next/cuda/exl3.cu and cuda/kernels/qmm*.cu.
//! Device math authored by Ash Hart and TensorFold contributors; launch pattern by jschmied (PR #443).
const std = @import("std");
const cuda = @import("cuda");
pub const Entries = struct {
    modules: [4]cuda.Module,
    functions: [variants.len]cuda.Function,
    pub fn load(ctx: *const cuda.Context) !Entries {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var e: Entries = undefined;
        var loaded: usize = 0;
        errdefer for (e.modules[0..loaded]) |*m| m.unload();
        for ([_][]const u8{ cuda.kernels.glm_qmm, cuda.kernels.glm_qmm_prefill, cuda.kernels.glm_exl3, cuda.kernels.experts }, 0..) |img, i| {
            e.modules[i] = try cuda.Module.load(ctx.d, img);
            loaded += 1;
        }
        for (variants, &e.functions) |v, *f| {
            f.* = try e.modules[v.image].function(v.symbol);
            if (v.shared > 0) try f.allowDynamicShared(v.shared);
        }
        return e;
    }
    pub fn deinit(e: *Entries) void {
        for (&e.modules) |*m| m.unload();
    }
    pub fn launch(e: *const Entries, image: usize, symbol: [:0]const u8, config: cuda.launch.Config, stream: cuda.Stream, args: *cuda.Args) !void {
        for (variants, 0..) |v, i| if (v.image == image and std.mem.eql(u8, v.symbol, symbol)) {
            if (config.shared > v.shared) return error.Invalid;
            return cuda.launch.launch(e.functions[i], config, stream, args);
        };
        return error.Invalid;
    }
    pub fn glm_qmm_16_0_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_16_0_0", config, stream, &a);
    }
    pub fn glm_qmm_32_0_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_32_0_0", config, stream, &a);
    }
    pub fn glm_qmm_64_0_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_64_0_0", config, stream, &a);
    }
    pub fn glm_qmm_cfg2_0_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_cfg2_0_0", config, stream, &a);
    }
    pub fn glm_qmm_cfg4_0_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_cfg4_0_0", config, stream, &a);
    }
    pub fn glm_qmm_16_0_1(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_16_0_1", config, stream, &a);
    }
    pub fn glm_qmm_32_0_1(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_32_0_1", config, stream, &a);
    }
    pub fn glm_qmm_64_0_1(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_64_0_1", config, stream, &a);
    }
    pub fn glm_qmm_cfg2_0_1(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_cfg2_0_1", config, stream, &a);
    }
    pub fn glm_qmm_cfg4_0_1(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_cfg4_0_1", config, stream, &a);
    }
    pub fn glm_qmm_16_1_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_16_1_0", config, stream, &a);
    }
    pub fn glm_qmm_32_1_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_32_1_0", config, stream, &a);
    }
    pub fn glm_qmm_64_1_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_64_1_0", config, stream, &a);
    }
    pub fn glm_qmm_cfg2_1_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_cfg2_1_0", config, stream, &a);
    }
    pub fn glm_qmm_cfg4_1_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_cfg4_1_0", config, stream, &a);
    }
    pub fn glm_qmm_16_1_1(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_16_1_1", config, stream, &a);
    }
    pub fn glm_qmm_32_1_1(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_32_1_1", config, stream, &a);
    }
    pub fn glm_qmm_64_1_1(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_64_1_1", config, stream, &a);
    }
    pub fn glm_qmm_cfg2_1_1(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_cfg2_1_1", config, stream, &a);
    }
    pub fn glm_qmm_cfg4_1_1(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, xs: u64, w: u64, scales: u64, biases: u64, out: u64, part: u64, M: c_int, N: c_int, K: c_int, SK: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(part);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(SK);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(0, "glm_qmm_cfg4_1_1", config, stream, &a);
    }
    pub fn glm_qmm_reduce_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, part: u64, out: u64, total: i64, SK: c_int) !void {
        var a: cuda.Args = .{};
        a.add(part);
        a.add(out);
        a.add(total);
        a.add(SK);
        try e.launch(0, "glm_qmm_reduce_0", config, stream, &a);
    }
    pub fn glm_qmm_reduce_1(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, part: u64, out: u64, total: i64, SK: c_int) !void {
        var a: cuda.Args = .{};
        a.add(part);
        a.add(out);
        a.add(total);
        a.add(SK);
        try e.launch(0, "glm_qmm_reduce_1", config, stream, &a);
    }
    pub fn glm_qmm_prefill_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, w: u64, scales: u64, biases: u64, out: u64, M: c_int, N: c_int, K: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(1, "glm_qmm_prefill_0", config, stream, &a);
    }
    pub fn glm_qmm_prefill_1(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, w: u64, scales: u64, biases: u64, out: u64, M: c_int, N: c_int, K: c_int, npad: c_int, ldx: c_int, group: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(scales);
        a.add(biases);
        a.add(out);
        a.add(M);
        a.add(N);
        a.add(K);
        a.add(npad);
        a.add(ldx);
        a.add(group);
        try e.launch(1, "glm_qmm_prefill_1", config, stream, &a);
    }
    pub fn glm_exl3_rot_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, x_stride: c_int, pick: u64, suh0: u64, suh1: u64, out0: u64, out1: u64, K: c_int, slots: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(x_stride);
        a.add(pick);
        a.add(suh0);
        a.add(suh1);
        a.add(out0);
        a.add(out1);
        a.add(K);
        a.add(slots);
        try e.launch(2, "glm_exl3_rot_0", config, stream, &a);
    }
    pub fn glm_exl3_rot_1(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, x: u64, x_stride: c_int, pick: u64, suh0: u64, suh1: u64, out0: u64, out1: u64, K: c_int, slots: c_int) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(x_stride);
        a.add(pick);
        a.add(suh0);
        a.add(suh1);
        a.add(out0);
        a.add(out1);
        a.add(K);
        a.add(slots);
        try e.launch(2, "glm_exl3_rot_1", config, stream, &a);
    }
    pub fn glm_exl3_gateup(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, Z: u64, pick: u64, svh_g: u64, svh_u: u64, suh_d: u64, xd: u64, P: c_int, N: c_int, SK: c_int, slots: c_int, limit: f32) !void {
        var a: cuda.Args = .{};
        a.add(Z);
        a.add(pick);
        a.add(svh_g);
        a.add(svh_u);
        a.add(suh_d);
        a.add(xd);
        a.add(P);
        a.add(N);
        a.add(SK);
        a.add(slots);
        a.add(limit);
        try e.launch(2, "glm_exl3_gateup", config, stream, &a);
    }
    pub fn glm_exl3_down(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, Z: u64, pick: u64, svh_d: u64, y: u64, P: c_int, D: c_int, SK: c_int, slots: c_int) !void {
        var a: cuda.Args = .{};
        a.add(Z);
        a.add(pick);
        a.add(svh_d);
        a.add(y);
        a.add(P);
        a.add(D);
        a.add(SK);
        a.add(slots);
        try e.launch(2, "glm_exl3_down", config, stream, &a);
    }
    pub fn glm_exl3_grouped(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, X0: u64, X1: u64, T0: u64, T1: u64, items: u64, counts: u64, members: u64, Z: u64, K: c_int, N: c_int, P: c_int, SK: c_int, E: c_int) !void {
        var a: cuda.Args = .{};
        a.add(X0);
        a.add(X1);
        a.add(T0);
        a.add(T1);
        a.add(items);
        a.add(counts);
        a.add(members);
        a.add(Z);
        a.add(K);
        a.add(N);
        a.add(P);
        a.add(SK);
        a.add(E);
        try e.launch(2, "glm_exl3_grouped", config, stream, &a);
    }
    pub fn glm_exl3_grouped_seq(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, X0: u64, X1: u64, T0: u64, T1: u64, items: u64, counts: u64, members: u64, Z: u64, K: c_int, N: c_int, P: c_int, SK: c_int, E: c_int) !void {
        var a: cuda.Args = .{};
        a.add(X0);
        a.add(X1);
        a.add(T0);
        a.add(T1);
        a.add(items);
        a.add(counts);
        a.add(members);
        a.add(Z);
        a.add(K);
        a.add(N);
        a.add(P);
        a.add(SK);
        a.add(E);
        try e.launch(2, "glm_exl3_grouped_seq", config, stream, &a);
    }
    pub fn glm_exl3_grouped_nf(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, X0: u64, X1: u64, T0: u64, T1: u64, items: u64, counts: u64, members: u64, Z: u64, K: c_int, N: c_int, P: c_int, SK: c_int, E: c_int) !void {
        var a: cuda.Args = .{};
        a.add(X0);
        a.add(X1);
        a.add(T0);
        a.add(T1);
        a.add(items);
        a.add(counts);
        a.add(members);
        a.add(Z);
        a.add(K);
        a.add(N);
        a.add(P);
        a.add(SK);
        a.add(E);
        try e.launch(2, "glm_exl3_grouped_nf", config, stream, &a);
    }
    pub fn glm_exl3_dec_8_0_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, X0: u64, X1: u64, T0: u64, T1: u64, items: u64, counts: u64, members: u64, Z: u64, K: c_int, N: c_int, P: c_int, SK: c_int, E: c_int, slots: c_int, sv0: u64, sv1: u64, su2: u64, out: u64, limit: f32, done: u64) !void {
        var a: cuda.Args = .{};
        a.add(X0);
        a.add(X1);
        a.add(T0);
        a.add(T1);
        a.add(items);
        a.add(counts);
        a.add(members);
        a.add(Z);
        a.add(K);
        a.add(N);
        a.add(P);
        a.add(SK);
        a.add(E);
        a.add(slots);
        a.add(sv0);
        a.add(sv1);
        a.add(su2);
        a.add(out);
        a.add(limit);
        a.add(done);
        try e.launch(2, "glm_exl3_dec_8_0_0", config, stream, &a);
    }
    pub fn glm_exl3_dec_8_1_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, X0: u64, X1: u64, T0: u64, T1: u64, items: u64, counts: u64, members: u64, Z: u64, K: c_int, N: c_int, P: c_int, SK: c_int, E: c_int, slots: c_int, sv0: u64, sv1: u64, su2: u64, out: u64, limit: f32, done: u64) !void {
        var a: cuda.Args = .{};
        a.add(X0);
        a.add(X1);
        a.add(T0);
        a.add(T1);
        a.add(items);
        a.add(counts);
        a.add(members);
        a.add(Z);
        a.add(K);
        a.add(N);
        a.add(P);
        a.add(SK);
        a.add(E);
        a.add(slots);
        a.add(sv0);
        a.add(sv1);
        a.add(su2);
        a.add(out);
        a.add(limit);
        a.add(done);
        try e.launch(2, "glm_exl3_dec_8_1_0", config, stream, &a);
    }
    pub fn glm_exl3_dec_10_0_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, X0: u64, X1: u64, T0: u64, T1: u64, items: u64, counts: u64, members: u64, Z: u64, K: c_int, N: c_int, P: c_int, SK: c_int, E: c_int, slots: c_int, sv0: u64, sv1: u64, su2: u64, out: u64, limit: f32, done: u64) !void {
        var a: cuda.Args = .{};
        a.add(X0);
        a.add(X1);
        a.add(T0);
        a.add(T1);
        a.add(items);
        a.add(counts);
        a.add(members);
        a.add(Z);
        a.add(K);
        a.add(N);
        a.add(P);
        a.add(SK);
        a.add(E);
        a.add(slots);
        a.add(sv0);
        a.add(sv1);
        a.add(su2);
        a.add(out);
        a.add(limit);
        a.add(done);
        try e.launch(2, "glm_exl3_dec_10_0_0", config, stream, &a);
    }
    pub fn glm_exl3_dec_10_1_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, X0: u64, X1: u64, T0: u64, T1: u64, items: u64, counts: u64, members: u64, Z: u64, K: c_int, N: c_int, P: c_int, SK: c_int, E: c_int, slots: c_int, sv0: u64, sv1: u64, su2: u64, out: u64, limit: f32, done: u64) !void {
        var a: cuda.Args = .{};
        a.add(X0);
        a.add(X1);
        a.add(T0);
        a.add(T1);
        a.add(items);
        a.add(counts);
        a.add(members);
        a.add(Z);
        a.add(K);
        a.add(N);
        a.add(P);
        a.add(SK);
        a.add(E);
        a.add(slots);
        a.add(sv0);
        a.add(sv1);
        a.add(su2);
        a.add(out);
        a.add(limit);
        a.add(done);
        try e.launch(2, "glm_exl3_dec_10_1_0", config, stream, &a);
    }
    pub fn glm_exl3_dec_12_0_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, X0: u64, X1: u64, T0: u64, T1: u64, items: u64, counts: u64, members: u64, Z: u64, K: c_int, N: c_int, P: c_int, SK: c_int, E: c_int, slots: c_int, sv0: u64, sv1: u64, su2: u64, out: u64, limit: f32, done: u64) !void {
        var a: cuda.Args = .{};
        a.add(X0);
        a.add(X1);
        a.add(T0);
        a.add(T1);
        a.add(items);
        a.add(counts);
        a.add(members);
        a.add(Z);
        a.add(K);
        a.add(N);
        a.add(P);
        a.add(SK);
        a.add(E);
        a.add(slots);
        a.add(sv0);
        a.add(sv1);
        a.add(su2);
        a.add(out);
        a.add(limit);
        a.add(done);
        try e.launch(2, "glm_exl3_dec_12_0_0", config, stream, &a);
    }
    pub fn glm_exl3_dec_12_1_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, X0: u64, X1: u64, T0: u64, T1: u64, items: u64, counts: u64, members: u64, Z: u64, K: c_int, N: c_int, P: c_int, SK: c_int, E: c_int, slots: c_int, sv0: u64, sv1: u64, su2: u64, out: u64, limit: f32, done: u64) !void {
        var a: cuda.Args = .{};
        a.add(X0);
        a.add(X1);
        a.add(T0);
        a.add(T1);
        a.add(items);
        a.add(counts);
        a.add(members);
        a.add(Z);
        a.add(K);
        a.add(N);
        a.add(P);
        a.add(SK);
        a.add(E);
        a.add(slots);
        a.add(sv0);
        a.add(sv1);
        a.add(su2);
        a.add(out);
        a.add(limit);
        a.add(done);
        try e.launch(2, "glm_exl3_dec_12_1_0", config, stream, &a);
    }
    pub fn glm_exl3_dec_16_0_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, X0: u64, X1: u64, T0: u64, T1: u64, items: u64, counts: u64, members: u64, Z: u64, K: c_int, N: c_int, P: c_int, SK: c_int, E: c_int, slots: c_int, sv0: u64, sv1: u64, su2: u64, out: u64, limit: f32, done: u64) !void {
        var a: cuda.Args = .{};
        a.add(X0);
        a.add(X1);
        a.add(T0);
        a.add(T1);
        a.add(items);
        a.add(counts);
        a.add(members);
        a.add(Z);
        a.add(K);
        a.add(N);
        a.add(P);
        a.add(SK);
        a.add(E);
        a.add(slots);
        a.add(sv0);
        a.add(sv1);
        a.add(su2);
        a.add(out);
        a.add(limit);
        a.add(done);
        try e.launch(2, "glm_exl3_dec_16_0_0", config, stream, &a);
    }
    pub fn glm_exl3_dec_16_1_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, X0: u64, X1: u64, T0: u64, T1: u64, items: u64, counts: u64, members: u64, Z: u64, K: c_int, N: c_int, P: c_int, SK: c_int, E: c_int, slots: c_int, sv0: u64, sv1: u64, su2: u64, out: u64, limit: f32, done: u64) !void {
        var a: cuda.Args = .{};
        a.add(X0);
        a.add(X1);
        a.add(T0);
        a.add(T1);
        a.add(items);
        a.add(counts);
        a.add(members);
        a.add(Z);
        a.add(K);
        a.add(N);
        a.add(P);
        a.add(SK);
        a.add(E);
        a.add(slots);
        a.add(sv0);
        a.add(sv1);
        a.add(su2);
        a.add(out);
        a.add(limit);
        a.add(done);
        try e.launch(2, "glm_exl3_dec_16_1_0", config, stream, &a);
    }
    pub fn glm_exl3_dec_16_0_1(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, X0: u64, X1: u64, T0: u64, T1: u64, items: u64, counts: u64, members: u64, Z: u64, K: c_int, N: c_int, P: c_int, SK: c_int, E: c_int, slots: c_int, sv0: u64, sv1: u64, su2: u64, out: u64, limit: f32, done: u64) !void {
        var a: cuda.Args = .{};
        a.add(X0);
        a.add(X1);
        a.add(T0);
        a.add(T1);
        a.add(items);
        a.add(counts);
        a.add(members);
        a.add(Z);
        a.add(K);
        a.add(N);
        a.add(P);
        a.add(SK);
        a.add(E);
        a.add(slots);
        a.add(sv0);
        a.add(sv1);
        a.add(su2);
        a.add(out);
        a.add(limit);
        a.add(done);
        try e.launch(2, "glm_exl3_dec_16_0_1", config, stream, &a);
    }
    pub fn glm_exl3_order(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, items: u64, counts: u64, order: u64, cap: c_int, E: c_int) !void {
        var a: cuda.Args = .{};
        a.add(items);
        a.add(counts);
        a.add(order);
        a.add(cap);
        a.add(E);
        try e.launch(2, "glm_exl3_order", config, stream, &a);
    }
    pub fn glm_exl3_prompt_gu_0(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, Xg: u64, Xu: u64, Tg: u64, Tu: u64, items: u64, counts: u64, members: u64, svh_g: u64, svh_u: u64, suh_d: u64, xd: u64, K: c_int, N: c_int, E: c_int, limit: f32, slots: c_int, order: u64) !void {
        var a: cuda.Args = .{};
        a.add(Xg);
        a.add(Xu);
        a.add(Tg);
        a.add(Tu);
        a.add(items);
        a.add(counts);
        a.add(members);
        a.add(svh_g);
        a.add(svh_u);
        a.add(suh_d);
        a.add(xd);
        a.add(K);
        a.add(N);
        a.add(E);
        a.add(limit);
        a.add(slots);
        a.add(order);
        try e.launch(2, "glm_exl3_prompt_gu_0", config, stream, &a);
    }
    pub fn glm_exl3_prompt_gu_1(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, Xg: u64, Xu: u64, Tg: u64, Tu: u64, items: u64, counts: u64, members: u64, svh_g: u64, svh_u: u64, suh_d: u64, xd: u64, K: c_int, N: c_int, E: c_int, limit: f32, slots: c_int, order: u64) !void {
        var a: cuda.Args = .{};
        a.add(Xg);
        a.add(Xu);
        a.add(Tg);
        a.add(Tu);
        a.add(items);
        a.add(counts);
        a.add(members);
        a.add(svh_g);
        a.add(svh_u);
        a.add(suh_d);
        a.add(xd);
        a.add(K);
        a.add(N);
        a.add(E);
        a.add(limit);
        a.add(slots);
        a.add(order);
        try e.launch(2, "glm_exl3_prompt_gu_1", config, stream, &a);
    }
    pub fn glm_exl3_prompt_down(e: *const Entries, config: cuda.launch.Config, stream: cuda.Stream, Xd: u64, Td: u64, items: u64, counts: u64, members: u64, svh_d: u64, y: u64, K: c_int, D: c_int, E: c_int, order: u64) !void {
        var a: cuda.Args = .{};
        a.add(Xd);
        a.add(Td);
        a.add(items);
        a.add(counts);
        a.add(members);
        a.add(svh_d);
        a.add(y);
        a.add(K);
        a.add(D);
        a.add(E);
        a.add(order);
        try e.launch(2, "glm_exl3_prompt_down", config, stream, &a);
    }
};
pub const variants = [_]struct { image: usize, symbol: [:0]const u8, shared: u32 }{
    .{ .image = 0, .symbol = "glm_qmm_16_0_0", .shared = 17664 },
    .{ .image = 0, .symbol = "glm_qmm_32_0_0", .shared = 26112 },
    .{ .image = 0, .symbol = "glm_qmm_64_0_0", .shared = 43008 },
    .{ .image = 0, .symbol = "glm_qmm_cfg2_0_0", .shared = 26880 },
    .{ .image = 0, .symbol = "glm_qmm_cfg4_0_0", .shared = 40320 },
    .{ .image = 0, .symbol = "glm_qmm_16_0_1", .shared = 17664 },
    .{ .image = 0, .symbol = "glm_qmm_32_0_1", .shared = 26112 },
    .{ .image = 0, .symbol = "glm_qmm_64_0_1", .shared = 43008 },
    .{ .image = 0, .symbol = "glm_qmm_cfg2_0_1", .shared = 26880 },
    .{ .image = 0, .symbol = "glm_qmm_cfg4_0_1", .shared = 40320 },
    .{ .image = 0, .symbol = "glm_qmm_16_1_0", .shared = 17664 },
    .{ .image = 0, .symbol = "glm_qmm_32_1_0", .shared = 26112 },
    .{ .image = 0, .symbol = "glm_qmm_64_1_0", .shared = 43008 },
    .{ .image = 0, .symbol = "glm_qmm_cfg2_1_0", .shared = 26880 },
    .{ .image = 0, .symbol = "glm_qmm_cfg4_1_0", .shared = 40320 },
    .{ .image = 0, .symbol = "glm_qmm_16_1_1", .shared = 17664 },
    .{ .image = 0, .symbol = "glm_qmm_32_1_1", .shared = 26112 },
    .{ .image = 0, .symbol = "glm_qmm_64_1_1", .shared = 43008 },
    .{ .image = 0, .symbol = "glm_qmm_cfg2_1_1", .shared = 26880 },
    .{ .image = 0, .symbol = "glm_qmm_cfg4_1_1", .shared = 40320 },
    .{ .image = 0, .symbol = "glm_qmm_reduce_0", .shared = 0 },
    .{ .image = 0, .symbol = "glm_qmm_reduce_1", .shared = 0 },
    .{ .image = 1, .symbol = "glm_qmm_prefill_0", .shared = 41984 },
    .{ .image = 1, .symbol = "glm_qmm_prefill_1", .shared = 41984 },
    .{ .image = 2, .symbol = "glm_exl3_rot_0", .shared = 0 },
    .{ .image = 2, .symbol = "glm_exl3_rot_1", .shared = 0 },
    .{ .image = 2, .symbol = "glm_exl3_gateup", .shared = 0 },
    .{ .image = 2, .symbol = "glm_exl3_down", .shared = 0 },
    .{ .image = 2, .symbol = "glm_exl3_grouped", .shared = 0 },
    .{ .image = 2, .symbol = "glm_exl3_grouped_seq", .shared = 0 },
    .{ .image = 2, .symbol = "glm_exl3_grouped_nf", .shared = 0 },
    .{ .image = 2, .symbol = "glm_exl3_dec_8_0_0", .shared = 0 },
    .{ .image = 2, .symbol = "glm_exl3_dec_8_1_0", .shared = 0 },
    .{ .image = 2, .symbol = "glm_exl3_dec_10_0_0", .shared = 0 },
    .{ .image = 2, .symbol = "glm_exl3_dec_10_1_0", .shared = 0 },
    .{ .image = 2, .symbol = "glm_exl3_dec_12_0_0", .shared = 0 },
    .{ .image = 2, .symbol = "glm_exl3_dec_12_1_0", .shared = 0 },
    .{ .image = 2, .symbol = "glm_exl3_dec_16_0_0", .shared = 0 },
    .{ .image = 2, .symbol = "glm_exl3_dec_16_1_0", .shared = 0 },
    .{ .image = 2, .symbol = "glm_exl3_dec_16_0_1", .shared = 0 },
    .{ .image = 2, .symbol = "glm_exl3_order", .shared = 0 },
    .{ .image = 2, .symbol = "glm_exl3_prompt_gu_0", .shared = 43008 },
    .{ .image = 2, .symbol = "glm_exl3_prompt_gu_1", .shared = 33792 },
    .{ .image = 2, .symbol = "glm_exl3_prompt_down", .shared = 43008 },
    .{ .image = 3, .symbol = "_ZN10tf_experts11plan_kernelEPKiiiiPiS2_S2_", .shared = 0 },
    .{ .image = 3, .symbol = "_ZN10tf_experts9plan_rankEPKiiiPiS2_", .shared = 0 },
    .{ .image = 3, .symbol = "_ZN10tf_experts12plan_offsetsEiiiPiS0_S0_", .shared = 0 },
    .{ .image = 3, .symbol = "_ZN10tf_experts12plan_scatterEPKiiiS1_S1_Pi", .shared = 0 },
};
