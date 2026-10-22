//! P9 owned drafter construction; TensorFold glm5_next/cuda/dflash2.py:Drafter.__init__.
//! Ash Hart and TensorFold contributors; jschmied CUDA patterns. Checkpoint weights CC BY-NC-ND.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("cuda_dflash_config.zig");
const weights = @import("cuda_dflash_weights.zig");
const forward = @import("cuda_dflash_forward.zig");
const multi = @import("cuda_dflash.zig");
const wp = @import("cuda_weights.zig");
pub const Dependencies = struct {
    driver: *const cuda.Driver,
    stream: cuda.Stream,
    tri: @import("cuda_triton.zig").Tri,
    torch: @import("cuda_torch_ops.zig").Torch,
    entries: *const @import("cuda_matmuls.zig").Entries,
    comm: ?@import("flashnext").mirror.Coll = null,
    lt: *const cuda.cublaslt.Library,
    embed: u64,
};
/// The supplied plan must include DFlash2. Optional MTP weights belong to the target loader.
/// P6 owns Dependencies and the plan; retain them until this package is destroyed.
pub const Package = struct {
    gpa: std.mem.Allocator,
    loader: weights.Loader,
    forward: forward.Forward,
    multi: multi.Multi,
    pred: []f32,
    succ: []f32,
    pub fn open(gpa: std.mem.Allocator, io: std.Io, plan: *const wp.Plan, model_dir: []const u8, draft_dir: []const u8, streams: usize, capacity: usize, deps: Dependencies) !*Package {
        if (deps.tri.rec != null or deps.tri.set == null) return error.MissingTritonSet;
        if (deps.tri.s.handle != deps.stream.handle or deps.torch.s.handle != deps.stream.handle) return error.BadDraftStream;
        const cfg = try config.Config.read(gpa, io, draft_dir);
        const rank = try config.Rank.init(plan.world, plan.rank);
        const p = try gpa.create(Package);
        errdefer gpa.destroy(p);
        p.gpa = gpa;
        p.loader = .{ .gpa = gpa, .io = io, .driver = deps.driver, .model_dir = model_dir, .draft_dir = draft_dir, .plan = plan };
        errdefer p.loader.deinit();
        const w = try p.loader.load(rank);
        const pi = try p.loader.item(.dflash2, "candidate_selector.predecessor_codebook");
        const si = try p.loader.item(.dflash2, "candidate_selector.successor_codebook");
        if (pi.slice.shape[0] != 154880 or pi.slice.shape[1] != 256 or si.slice.shape[0] != 154880 or si.slice.shape[1] != 256) return error.BadSelector;
        p.pred = try weights.readCodebook(gpa, io, draft_dir, pi);
        errdefer gpa.free(p.pred);
        p.succ = try weights.readCodebook(gpa, io, draft_dir, si);
        errdefer gpa.free(p.succ);
        p.forward = try forward.Forward.init(gpa, deps.driver, deps.stream, cfg, rank, streams, w, deps.embed, deps.tri, deps.torch, deps.entries, deps.comm, deps.lt);
        errdefer p.forward.deinit();
        p.multi = try multi.Multi.init(gpa, &p.forward, .{ .pred = p.pred, .succ = p.succ }, capacity);
        return p;
    }
    pub fn deinit(p: *Package) void {
        p.multi.deinit();
        p.forward.deinit();
        p.loader.deinit();
        p.gpa.free(p.pred);
        p.gpa.free(p.succ);
        p.gpa.destroy(p);
    }
};
