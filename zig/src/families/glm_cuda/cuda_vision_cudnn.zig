//! The tower's two convolutions through cuDNN's backend (graph) API, called the way PyTorch 2.13 calls it, so that cuDNN's
//! heuristics pick the kernel PyTorch's `F.conv3d` / `F.conv2d` run -- and with it the accumulation order, which varies
//! with the number of rows (split-K for some, plain chains for others) and cannot be reproduced by one fixed kernel.
//!
//! PyTorch (aten/src/ATen/native/cudnn/Conv_v8.cpp) builds, per call shape: tensor descriptors (data type bf16, the dims
//! and strides of the tensors it passes, byte alignment 32), a convolution descriptor (fp32 compute, cross-correlation),
//! a forward-convolution operation (alpha 1, beta 0), an operation graph, asks the INSTANT heuristic for engine
//! configurations, builds an execution plan for the first and runs it with a variant pack. cuDNN's own API trace
//! (CUDNN_LOGINFO_DBG) shows exactly these calls for the shapes of the tower; this file issues the same ones. The
//! convolution's bias is not part of the cuDNN call: torch adds it in a separate bf16 add (see vis_bias_add).
//!
//! libcudnn.so.9 is opened at run time like cuBLASLt (the Zig side has no build dependency on it).

const std = @import("std");
const cuda = @import("cuda");

pub const Error = error{ LibraryUnavailable, MissingSymbol, CudnnFailed, NoPlan, Invalid, OutOfMemory };

const Status = c_int;
const Handle = ?*anyopaque;
const Desc = ?*anyopaque;

// cudnn_graph_v9.h values
const T_HANDLE: c_int = 0;
const T_DATA_TYPE: c_int = 1;
const T_INT64: c_int = 3;
const T_FLOAT: c_int = 4;
const T_VOID_PTR: c_int = 6;
const T_CONV_MODE: c_int = 7;
const T_HEUR_MODE: c_int = 8;
const T_DESC: c_int = 15;

const D_CONV: c_int = 1;
const D_ENGINE: c_int = 2;
const D_ENGINECFG: c_int = 3;
const D_HEUR: c_int = 4;
const D_PLAN: c_int = 5;
const D_OP_CONV_FWD: c_int = 10;
const D_GRAPH: c_int = 15;
const D_VARPACK: c_int = 16;
const D_TENSOR: c_int = 17;

const A_TENSOR_ALIGN: c_int = 900;
const A_TENSOR_DTYPE: c_int = 901;
const A_TENSOR_DIMS: c_int = 902;
const A_TENSOR_STRIDES: c_int = 903;
const A_TENSOR_UID: c_int = 906;
const A_CONV_COMP: c_int = 100;
const A_CONV_MODE: c_int = 101;
const A_CONV_DILATIONS: c_int = 102;
const A_CONV_FSTRIDES: c_int = 103;
const A_CONV_POST: c_int = 104;
const A_CONV_PRE: c_int = 105;
const A_CONV_SPATIAL: c_int = 106;
const A_OP_ALPHA: c_int = 700;
const A_OP_BETA: c_int = 701;
const A_OP_CONV: c_int = 702;
const A_OP_W: c_int = 703;
const A_OP_X: c_int = 704;
const A_OP_Y: c_int = 705;
const A_GRAPH_HANDLE: c_int = 800;
const A_GRAPH_OPS: c_int = 801;
const A_HEUR_MODE: c_int = 200;
const A_HEUR_GRAPH: c_int = 201;
const A_HEUR_RESULTS: c_int = 202;
const A_CFG_ENGINE: c_int = 300;
const A_ENGINE_INDEX: c_int = 1301;
const A_PLAN_CFG: c_int = 401;
const A_PLAN_WORKSPACE: c_int = 402;
const A_VP_UIDS: c_int = 1000;
const A_VP_PTRS: c_int = 1001;
const A_VP_WORKSPACE: c_int = 1003;

const data_bf16: c_int = 9;
const data_f32: c_int = 0;
const cross_correlation: c_int = 1;
const heur_instant: c_int = 0;

const Api = struct {
    cudnnCreate: *const fn (*Handle) callconv(.c) Status,
    cudnnDestroy: *const fn (Handle) callconv(.c) Status,
    cudnnSetStream: *const fn (Handle, cuda.abi.Stream) callconv(.c) Status,
    cudnnGetVersion: *const fn () callconv(.c) usize,
    cudnnBackendCreateDescriptor: *const fn (c_int, *Desc) callconv(.c) Status,
    cudnnBackendDestroyDescriptor: *const fn (Desc) callconv(.c) Status,
    cudnnBackendSetAttribute: *const fn (Desc, c_int, c_int, i64, ?*const anyopaque) callconv(.c) Status,
    cudnnBackendGetAttribute: *const fn (Desc, c_int, c_int, i64, *i64, ?*anyopaque) callconv(.c) Status,
    cudnnBackendFinalize: *const fn (Desc) callconv(.c) Status,
    cudnnBackendExecute: *const fn (Handle, Desc, Desc) callconv(.c) Status,
};

/// A convolution as PyTorch describes it to cuDNN: x [N, C, spatial...], w [K, C, filter...], y [N, K, 1...].
pub const Spec = struct {
    rank: u8, // 5 (conv3d) or 4 (conv2d)
    xdims: [5]i64,
    xstr: [5]i64,
    wdims: [5]i64,
    wstr: [5]i64,
    ydims: [5]i64,
    ystr: [5]i64,
    fstride: [3]i64,

    /// conv3d patch embedding: [rows, 3, 2, 14, 14] * [1024, 3, 2, 14, 14], stride (2, 14, 14), contiguous tensors.
    pub fn embed(rows: usize) Spec {
        const r: i64 = @intCast(rows);
        return .{ .rank = 5, .xdims = .{ r, 3, 2, 14, 14 }, .xstr = .{ 1176, 392, 196, 14, 1 }, .wdims = .{ 1024, 3, 2, 14, 14 }, .wstr = .{ 1176, 392, 196, 14, 1 }, .ydims = .{ r, 1024, 1, 1, 1 }, .ystr = .{ 1024, 1, 1, 1, 1 }, .fstride = .{ 2, 14, 14 } };
    }
    /// conv2d downsample: channels-last [rows, 1024, 2, 2] * channels-last [4096, 1024, 2, 2], stride 2.
    pub fn downsample(rows: usize) Spec {
        const r: i64 = @intCast(rows);
        return .{ .rank = 4, .xdims = .{ r, 1024, 2, 2, 1 }, .xstr = .{ 4096, 1, 2048, 1024, 1 }, .wdims = .{ 4096, 1024, 2, 2, 1 }, .wstr = .{ 4096, 1, 2048, 1024, 1 }, .ydims = .{ r, 4096, 1, 1, 1 }, .ystr = .{ 4096, 1, 4096, 4096, 1 }, .fstride = .{ 2, 2, 1 } };
    }
};

pub const Plan = struct {
    plan: Desc,
    workspace: usize,
    engine: i64,
    spec: Spec,
};

pub const Cudnn = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    lib: std.DynLib,
    api: Api,
    handle: Handle,
    plans: std.AutoHashMapUnmanaged(Key, Plan) = .empty,
    workspace: ?cuda.DeviceBuffer = null,
    keep: std.ArrayList(Desc) = .empty, // graphs, heuristics and configurations the plans refer to
    version: usize,

    const Key = struct { rank: u8, rows: i64 };

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, stream: cuda.Stream) Error!Cudnn {
        var self: Cudnn = undefined;
        self.gpa = gpa;
        self.d = d;
        self.plans = .empty;
        self.keep = .empty;
        self.workspace = null;
        self.lib = std.DynLib.open("libcudnn.so.9") catch std.DynLib.open("libcudnn.so") catch return error.LibraryUnavailable;
        errdefer self.lib.close();
        inline for (@typeInfo(Api).@"struct".field_names, @typeInfo(Api).@"struct".field_types) |name, T| {
            @field(self.api, name) = self.lib.lookup(T, name) orelse return error.MissingSymbol;
        }
        self.version = self.api.cudnnGetVersion();
        if (self.api.cudnnCreate(&self.handle) != 0) return error.CudnnFailed;
        errdefer _ = self.api.cudnnDestroy(self.handle);
        if (self.api.cudnnSetStream(self.handle, stream.handle) != 0) return error.CudnnFailed;
        return self;
    }

    pub fn deinit(self: *Cudnn) void {
        var it = self.plans.valueIterator();
        while (it.next()) |p| _ = self.api.cudnnBackendDestroyDescriptor(p.plan);
        self.plans.deinit(self.gpa);
        for (self.keep.items) |k| _ = self.api.cudnnBackendDestroyDescriptor(k);
        self.keep.deinit(self.gpa);
        if (self.workspace) |*w| w.free();
        _ = self.api.cudnnDestroy(self.handle);
        self.lib.close();
        self.* = undefined;
    }

    fn ck(self: *const Cudnn, s: Status, what: []const u8) Error!void {
        _ = self;
        if (s == 0) return;
        std.log.err("cuDNN {s}: status {d}", .{ what, s });
        return error.CudnnFailed;
    }

    fn create(self: *Cudnn, kind: c_int) Error!Desc {
        var d: Desc = null;
        try self.ck(self.api.cudnnBackendCreateDescriptor(kind, &d), "create descriptor");
        return d;
    }
    fn destroy(self: *Cudnn, d: Desc) void {
        _ = self.api.cudnnBackendDestroyDescriptor(d);
    }
    fn set(self: *Cudnn, d: Desc, attr: c_int, ty: c_int, count: i64, p: anytype) Error!void {
        const ptr: ?*const anyopaque = @ptrCast(p);
        try self.ck(self.api.cudnnBackendSetAttribute(d, attr, ty, count, ptr), "set attribute");
    }
    fn fin(self: *Cudnn, d: Desc) Error!void {
        try self.ck(self.api.cudnnBackendFinalize(d), "finalize");
    }

    fn tensor(self: *Cudnn, dims: []const i64, strides: []const i64, uid: i64) Error!Desc {
        const t = try self.create(D_TENSOR);
        errdefer self.destroy(t);
        const dt: c_int = data_bf16;
        const align_: i64 = 32;
        try self.set(t, A_TENSOR_DTYPE, T_DATA_TYPE, 1, &dt);
        try self.set(t, A_TENSOR_DIMS, T_INT64, @intCast(dims.len), dims.ptr);
        try self.set(t, A_TENSOR_STRIDES, T_INT64, @intCast(strides.len), strides.ptr);
        try self.set(t, A_TENSOR_UID, T_INT64, 1, &uid);
        try self.set(t, A_TENSOR_ALIGN, T_INT64, 1, &align_);
        try self.fin(t);
        return t;
    }

    const uid_x: i64 = 120;
    const uid_w: i64 = 119;
    const uid_y: i64 = 121;

    /// The plan for a convolution shape, built once (graph, INSTANT heuristic, first configuration that builds).
    pub fn plan(self: *Cudnn, spec: Spec) Error!*Plan {
        const key: Key = .{ .rank = spec.rank, .rows = spec.xdims[0] };
        if (self.plans.getPtr(key)) |p| return p;
        const n: usize = spec.rank;
        const nsp: usize = n - 2;
        const x = try self.tensor(spec.xdims[0..n], spec.xstr[0..n], uid_x);
        defer self.destroy(x);
        const y = try self.tensor(spec.ydims[0..n], spec.ystr[0..n], uid_y);
        defer self.destroy(y);
        const w = try self.tensor(spec.wdims[0..n], spec.wstr[0..n], uid_w);
        defer self.destroy(w);

        const conv = try self.create(D_CONV);
        defer self.destroy(conv);
        {
            const comp: c_int = data_f32;
            const mode: c_int = cross_correlation;
            const sp: i64 = @intCast(nsp);
            const zeros = [3]i64{ 0, 0, 0 };
            const ones = [3]i64{ 1, 1, 1 };
            try self.set(conv, A_CONV_COMP, T_DATA_TYPE, 1, &comp);
            try self.set(conv, A_CONV_MODE, T_CONV_MODE, 1, &mode);
            try self.set(conv, A_CONV_SPATIAL, T_INT64, 1, &sp);
            try self.set(conv, A_CONV_PRE, T_INT64, @intCast(nsp), &zeros);
            try self.set(conv, A_CONV_POST, T_INT64, @intCast(nsp), &zeros);
            try self.set(conv, A_CONV_DILATIONS, T_INT64, @intCast(nsp), &ones);
            try self.set(conv, A_CONV_FSTRIDES, T_INT64, @intCast(nsp), &spec.fstride);
            try self.fin(conv);
        }
        const op = try self.create(D_OP_CONV_FWD);
        defer self.destroy(op);
        {
            const one: f32 = 1;
            const zero: f32 = 0;
            try self.set(op, A_OP_X, T_DESC, 1, &x);
            try self.set(op, A_OP_W, T_DESC, 1, &w);
            try self.set(op, A_OP_Y, T_DESC, 1, &y);
            try self.set(op, A_OP_CONV, T_DESC, 1, &conv);
            try self.set(op, A_OP_ALPHA, T_FLOAT, 1, &one);
            try self.set(op, A_OP_BETA, T_FLOAT, 1, &zero);
            try self.fin(op);
        }
        const graph = try self.create(D_GRAPH);
        try self.keep.append(self.gpa, graph);
        {
            try self.set(graph, A_GRAPH_OPS, T_DESC, 1, &op);
            try self.set(graph, A_GRAPH_HANDLE, T_HANDLE, 1, &self.handle);
            try self.fin(graph);
        }
        const heur = try self.create(D_HEUR);
        try self.keep.append(self.gpa, heur);
        {
            const mode: c_int = heur_instant;
            try self.set(heur, A_HEUR_GRAPH, T_DESC, 1, &graph);
            try self.set(heur, A_HEUR_MODE, T_HEUR_MODE, 1, &mode);
            try self.fin(heur);
        }
        var count: i64 = 0;
        try self.ck(self.api.cudnnBackendGetAttribute(heur, A_HEUR_RESULTS, T_DESC, 0, &count, null), "heuristic count");
        if (count <= 0) return error.NoPlan;
        const cfgs = try self.gpa.alloc(Desc, @intCast(count));
        defer self.gpa.free(cfgs);
        for (cfgs) |*c| c.* = try self.create(D_ENGINECFG);
        // every configuration descriptor stays with the plans (the plan refers to its configuration)
        for (cfgs) |c| try self.keep.append(self.gpa, c);
        var got: i64 = 0;
        try self.ck(self.api.cudnnBackendGetAttribute(heur, A_HEUR_RESULTS, T_DESC, count, &got, @ptrCast(cfgs.ptr)), "heuristic results");
        for (cfgs[0..@intCast(got)]) |cfg| {
            const p = try self.create(D_PLAN);
            self.set(p, A_PLAN_CFG, T_DESC, 1, &cfg) catch {
                self.destroy(p);
                continue;
            };
            self.fin(p) catch {
                self.destroy(p);
                continue;
            };
            var ws: i64 = 0;
            var c1: i64 = 0;
            try self.ck(self.api.cudnnBackendGetAttribute(p, A_PLAN_WORKSPACE, T_INT64, 1, &c1, @ptrCast(&ws)), "workspace size");
            const eng = self.engineIndex(cfg) catch -1;
            try self.plans.put(self.gpa, key, .{ .plan = p, .workspace = @intCast(@max(ws, 0)), .engine = eng, .spec = spec });
            return self.plans.getPtr(key).?;
        }
        return error.NoPlan;
    }

    fn engineIndex(self: *Cudnn, cfg: Desc) Error!i64 {
        const e = try self.create(D_ENGINE);
        defer self.destroy(e);
        var n: i64 = 0;
        var arr = [1]Desc{e};
        try self.ck(self.api.cudnnBackendGetAttribute(cfg, A_CFG_ENGINE, T_DESC, 1, &n, @ptrCast(&arr)), "engine of config");
        var idx: i64 = 0;
        try self.ck(self.api.cudnnBackendGetAttribute(e, A_ENGINE_INDEX, T_INT64, 1, &n, @ptrCast(&idx)), "engine index");
        return idx;
    }

    /// y = conv(x, w) on the stream given to `init`.
    pub fn run(self: *Cudnn, spec: Spec, x: u64, w: u64, y: u64) Error!void {
        const p = try self.plan(spec);
        if (p.workspace > 0) {
            if (self.workspace == null or self.workspace.?.len < p.workspace) {
                if (self.workspace) |*old| old.free();
                self.workspace = cuda.DeviceBuffer.alloc(self.d, p.workspace) catch return error.CudnnFailed;
            }
        }
        const vp = try self.create(D_VARPACK);
        defer self.destroy(vp);
        const uids = [3]i64{ uid_x, uid_y, uid_w };
        const ptrs = [3]u64{ x, y, w };
        const ws: u64 = if (self.workspace) |b| b.ptr else 0;
        try self.set(vp, A_VP_UIDS, T_INT64, 3, &uids);
        try self.set(vp, A_VP_PTRS, T_VOID_PTR, 3, &ptrs);
        try self.set(vp, A_VP_WORKSPACE, T_VOID_PTR, 1, &ws);
        try self.fin(vp);
        try self.ck(self.api.cudnnBackendExecute(self.handle, p.plan, vp), "execute");
    }
};
