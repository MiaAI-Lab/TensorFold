//! The vision tower's matrix products, issued the way PyTorch 2.13 issues them so that cuBLAS picks the same kernels:
//!  * F.linear with a bias (addmm on a 2-D input, bf16): cublasLtMatmul with CUBLASLT_EPILOGUE_BIAS, a 1 MiB workspace,
//!    256-byte alignment preferences and the first heuristic result (`gemm_and_bias` in ATen/cuda/CUDABlas.cpp);
//!  * F.linear without a bias (mm): cublasGemmEx, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP, default math mode,
//!    a 32 MiB workspace set on the handle.
//! Both libraries are opened at run time (libcublasLt.so.13 / libcublas.so.13), like the rest of this runtime.
//! Written for this port from the CUDA headers and the call sequences cuBLAS's own logging shows for PyTorch's calls.

const std = @import("std");
const cuda = @import("cuda");
const lt = cuda.cublaslt;

pub const Error = lt.Error || error{GemmFailed};

pub const lt_workspace_bytes: usize = 1 << 20;
pub const gemm_workspace_bytes: usize = 32 << 20;

const bias_epilogue: c_int = 4; // CUBLASLT_EPILOGUE_BIAS
const attr_epilogue: c_int = 7;
const attr_bias_pointer: c_int = 8;
const pref_align_a: c_int = 5;

const Plan = struct { desc: lt.MatmulDesc, a: lt.Layout, b: lt.Layout, c: lt.Layout, pref: lt.Preference, algo: lt.Algo, ws: usize };
const Key = struct { m: u32, n: u32, k: u32, bias: bool };

pub const CublasApi = struct {
    cublasCreate_v2: *const fn (*?*anyopaque) callconv(.c) c_int,
    cublasDestroy_v2: *const fn (?*anyopaque) callconv(.c) c_int,
    cublasSetStream_v2: *const fn (?*anyopaque, cuda.abi.Stream) callconv(.c) c_int,
    cublasSetWorkspace_v2: *const fn (?*anyopaque, u64, usize) callconv(.c) c_int,
    cublasSetMathMode: *const fn (?*anyopaque, c_int) callconv(.c) c_int,
    cublasGemmEx: *const fn (?*anyopaque, c_int, c_int, c_int, c_int, c_int, *const anyopaque, u64, c_int, c_int, u64, c_int, c_int, *const anyopaque, u64, c_int, c_int, c_int, c_int) callconv(.c) c_int,
};

pub const Blas = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    ltlib: lt.Library,
    handle: lt.Handle,
    plans: std.AutoHashMapUnmanaged(Key, Plan) = .empty,
    workspace: cuda.DeviceBuffer,
    gemm_ws: cuda.DeviceBuffer,
    cublas_lib: std.DynLib,
    cublas_api: CublasApi,
    cublas: ?*anyopaque,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, stream: cuda.Stream) !Blas {
        var self: Blas = undefined;
        self.gpa = gpa;
        self.d = d;
        self.plans = .empty;
        self.ltlib = try lt.Library.open();
        errdefer self.ltlib.close();
        try self.ltlib.check(self.ltlib.api.cublasLtCreate(&self.handle), "cublasLtCreate");
        errdefer _ = self.ltlib.api.cublasLtDestroy(self.handle);
        self.workspace = try cuda.DeviceBuffer.alloc(d, lt_workspace_bytes);
        errdefer self.workspace.free();
        self.gemm_ws = try cuda.DeviceBuffer.alloc(d, gemm_workspace_bytes);
        errdefer self.gemm_ws.free();
        self.cublas_lib = std.DynLib.open("libcublas.so.13") catch std.DynLib.open("libcublas.so") catch return error.LibraryUnavailable;
        errdefer self.cublas_lib.close();
        inline for (@typeInfo(CublasApi).@"struct".field_names, @typeInfo(CublasApi).@"struct".field_types) |name, T| {
            @field(self.cublas_api, name) = self.cublas_lib.lookup(T, name) orelse return error.MissingSymbol;
        }
        if (self.cublas_api.cublasCreate_v2(&self.cublas) != 0) return error.GemmFailed;
        errdefer _ = self.cublas_api.cublasDestroy_v2(self.cublas);
        if (self.cublas_api.cublasSetStream_v2(self.cublas, stream.handle) != 0) return error.GemmFailed;
        if (self.cublas_api.cublasSetWorkspace_v2(self.cublas, self.gemm_ws.ptr, gemm_workspace_bytes) != 0) return error.GemmFailed;
        if (self.cublas_api.cublasSetMathMode(self.cublas, 0) != 0) return error.GemmFailed;
        return self;
    }

    pub fn deinit(self: *Blas) void {
        const api = self.ltlib.api;
        var it = self.plans.valueIterator();
        while (it.next()) |p| {
            _ = api.cublasLtMatmulPreferenceDestroy(p.pref);
            _ = api.cublasLtMatrixLayoutDestroy(p.c);
            _ = api.cublasLtMatrixLayoutDestroy(p.b);
            _ = api.cublasLtMatrixLayoutDestroy(p.a);
            _ = api.cublasLtMatmulDescDestroy(p.desc);
        }
        self.plans.deinit(self.gpa);
        _ = self.cublas_api.cublasDestroy_v2(self.cublas);
        self.cublas_lib.close();
        _ = api.cublasLtDestroy(self.handle);
        self.ltlib.close();
        self.gemm_ws.free();
        self.workspace.free();
        self.* = undefined;
    }

    fn plan(self: *Blas, m: usize, n: usize, k: usize, bias: u64) !*Plan {
        const key: Key = .{ .m = @intCast(m), .n = @intCast(n), .k = @intCast(k), .bias = bias != 0 };
        if (self.plans.getPtr(key)) |p| return p;
        const l = &self.ltlib;
        const api = l.api;
        var p: Plan = undefined;
        try l.check(api.cublasLtMatmulDescCreate(&p.desc, lt.compute_32f, .f32), "cublasLtMatmulDescCreate");
        errdefer _ = api.cublasLtMatmulDescDestroy(p.desc);
        const ta: c_int = 1;
        const tb: c_int = 0;
        try l.check(api.cublasLtMatmulDescSetAttribute(p.desc, lt.desc_transa, &ta, 4), "transa");
        try l.check(api.cublasLtMatmulDescSetAttribute(p.desc, lt.desc_transb, &tb, 4), "transb");
        if (bias != 0) {
            const ep: c_int = bias_epilogue;
            try l.check(api.cublasLtMatmulDescSetAttribute(p.desc, attr_epilogue, &ep, 4), "epilogue");
            try l.check(api.cublasLtMatmulDescSetAttribute(p.desc, attr_bias_pointer, &bias, 8), "bias pointer");
        }
        // the column-major view PyTorch hands cuBLAS: D^T [n, m] = W [k, n]^T . X^T [k, m]
        try l.check(api.cublasLtMatrixLayoutCreate(&p.a, .bf16, k, n, @intCast(k)), "layout A");
        errdefer _ = api.cublasLtMatrixLayoutDestroy(p.a);
        try l.check(api.cublasLtMatrixLayoutCreate(&p.b, .bf16, k, m, @intCast(k)), "layout B");
        errdefer _ = api.cublasLtMatrixLayoutDestroy(p.b);
        try l.check(api.cublasLtMatrixLayoutCreate(&p.c, .bf16, n, m, @intCast(n)), "layout C");
        errdefer _ = api.cublasLtMatrixLayoutDestroy(p.c);
        try l.check(api.cublasLtMatmulPreferenceCreate(&p.pref), "preference");
        errdefer _ = api.cublasLtMatmulPreferenceDestroy(p.pref);
        const limit: u64 = lt_workspace_bytes;
        try l.check(api.cublasLtMatmulPreferenceSetAttribute(p.pref, lt.pref_max_workspace_bytes, &limit, 8), "workspace");
        const alignment: u32 = 256;
        for (0..4) |i| try l.check(api.cublasLtMatmulPreferenceSetAttribute(p.pref, pref_align_a + @as(c_int, @intCast(i)), &alignment, 4), "alignment");
        var found: [1]lt.Heuristic = undefined;
        var count: c_int = 0;
        try l.check(api.cublasLtMatmulAlgoGetHeuristic(self.handle, p.desc, p.a, p.b, p.c, p.c, p.pref, 1, &found, &count), "heuristic");
        if (count < 1 or found[0].state != 0) return error.NoAlgorithm;
        p.algo = found[0].algo;
        p.ws = found[0].workspace_size;
        try self.plans.put(self.gpa, key, p);
        return self.plans.getPtr(key).?;
    }

    /// out[m, n] = x[m, k] . w[n, k]^T (+ bias[n] in the epilogue when `bias` != 0), bf16 throughout, fp32 accumulation.
    pub fn linear(self: *Blas, stream: cuda.Stream, x: u64, w: u64, bias: u64, out: u64, m: usize, n: usize, k: usize) !void {
        if (bias == 0) return self.gemm(x, w, out, m, n, k);
        const p = try self.plan(m, n, k, bias);
        const l = &self.ltlib;
        try l.check(l.api.cublasLtMatmulDescSetAttribute(p.desc, attr_bias_pointer, &bias, 8), "bias pointer");
        const one: f32 = 1;
        const zero: f32 = 0;
        try l.check(l.api.cublasLtMatmul(self.handle, p.desc, &one, w, p.a, x, p.b, &zero, out, p.c, out, p.c, &p.algo, self.workspace.ptr, lt_workspace_bytes, stream.handle), "cublasLtMatmul");
    }

    /// The same product through cublasGemmEx (torch's mm).
    pub fn gemm(self: *Blas, x: u64, w: u64, out: u64, m: usize, n: usize, k: usize) !void {
        const one: f32 = 1;
        const zero: f32 = 0;
        const st = self.cublas_api.cublasGemmEx(self.cublas, 1, 0, @intCast(n), @intCast(m), @intCast(k), &one, w, 14, @intCast(k), x, 14, @intCast(k), &zero, out, 14, @intCast(n), 68, 99);
        if (st != 0) {
            std.log.err("cublasGemmEx failed: status {d} (m={d} n={d} k={d})", .{ st, m, n, k });
            return error.GemmFailed;
        }
    }
};
