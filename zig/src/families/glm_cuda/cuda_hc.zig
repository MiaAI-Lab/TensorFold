//! glm5_next/cuda/hc.cu hc_partial_cuda and glue.py hc_pre/hc_post.
//! Ash Hart and TensorFold contributors; CUDA launch pattern by jschmied (PR #443).
const cuda = @import("cuda");
const Tri = @import("cuda_triton.zig").Tri;
const Buffers = @import("cuda_state.zig").Buffers;
pub const Weights = struct { fn_w: u64, base: u64, scale: u64, norm: u64 };
pub const HC = struct {
    mod: cuda.Module,
    partial: cuda.Function,
    pub fn init(d: *const cuda.Driver) !HC {
        var m = try @import("cuda_kernels.zig").load(d, .hc);
        errdefer m.unload();
        return .{ .mod = m, .partial = try m.function("hc_partial_kernel") };
    }
    pub fn deinit(h: *HC) void {
        h.mod.unload();
    }
    pub fn runPartial(h: HC, stream: cuda.Stream, x: u64, fn_w: u64, part: u64, rows: usize) !void {
        var args: cuda.Args = .{};
        args.add(x);
        args.add(fn_w);
        args.add(part);
        args.add(@as(i32, @intCast(rows)));
        try cuda.launch.launch(h.partial, .{ .grid = .{ .x = 16, .y = 3, .z = @intCast((rows + 3) / 4) }, .block = .{ .x = 288 } }, stream, &args);
    }
    pub fn pre(h: HC, t: Tri, b: Buffers, w: Weights, rows: usize) !void {
        try h.runPartial(t.s, b.x, w.fn_w, b.hc_part, rows);
        try t.hc_finish(.{ rows, 1, 1 }, b.x, b.hc_part, w.base, w.scale, w.norm, b.normed, b.xs, b.post, b.comb, 1e-5, 1e-6, .{ .D = 4096, .S = 4, .NB = 16, .ITERS = 20, .BLOCK = 4096 });
    }
    pub fn post(t: Tri, b: Buffers, gathered: u64, rows: usize, world: usize) !void {
        try t.hc_post(.{ rows, 4, 1 }, b.x, b.x, gathered, b.post, b.comb, @intCast(rows * 4096), .{ .D = 4096, .S = 4, .WORLD = @intCast(world), .BLOCK = 1024, .ROUND = 0 });
    }
};
