//! GLM-5.3-Flash CUDA foundations. Python: families/glm5_next/cuda/ (TensorFold v0.6.0).
//! Ash Hart and TensorFold contributors; jschmied's CUDA family registry (PR #443).
pub const native = @import("cuda_native.zig");
pub const geometry = @import("cuda_geometry.zig");
pub const state = @import("cuda_state.zig");
pub const pool = @import("cuda_pool.zig");
pub const shard = @import("cuda_shard.zig"); // TF_GLM_KV_SHARD (0118): sizes, addresses, glue, the history fetch (host side)
pub const segments = @import("cuda_segments.zig");
pub const kda = @import("cuda_kda.zig"); // P4: kda.cu / kda_chunk.cu, the segment table, replay, conv shift
pub const triton = @import("cuda_triton.zig");
pub const kernels = @import("cuda_kernels.zig");
test {
    _ = native;
    _ = geometry;
    _ = state;
    _ = pool;
    _ = shard;
    _ = @import("cuda_shard_test.zig");
    _ = segments;
    _ = kda;
    _ = triton;
    _ = kernels;
}

test "analyze device allocation and view entry points without running CUDA" {
    _ = state.Slots.init;
    _ = state.Slots.deinit;
    _ = state.Caches.init;
    _ = state.Caches.deinit;
    _ = state.Buffers.init;
    _ = state.Buffers.deinit;
    _ = segments.SegRows.initDevice;
    _ = segments.SegRows.stage;
    _ = kernels.load;
}

pub const sampler = @import("cuda_sampler.zig");
pub const nucleus = @import("cuda_nucleus.zig");
pub const torch_ops = @import("cuda_torch_ops.zig");
pub const rounds = @import("cuda_rounds.zig");
pub const backend = @import("cuda_lanes.zig");
pub const mirror = @import("cuda_mirror.zig");
pub const grammar = @import("cuda_grammar.zig");
test {
    _ = grammar;
    _ = sampler;
    _ = nucleus;
    _ = torch_ops;
    _ = rounds;
    _ = backend;
    _ = mirror;
}

pub const serving = @import("cuda_serving.zig");
test {
    _ = serving;
}

pub const forward = @import("cuda_forward.zig");
pub const kda_interface = @import("cuda_kda_interface.zig");
test {
    _ = forward;
    _ = kda_interface;
}

pub const stage = @import("cuda_stage.zig");
test {
    _ = stage;
}

pub const verify = @import("cuda_verify.zig");
test {
    _ = verify;
}

pub const upload = @import("cuda_upload.zig");
test {
    _ = upload;
}

pub const layer_check = @import("cuda_layer_check.zig");
test {
    _ = layer_check;
}

pub const model_weights = @import("cuda_model_weights.zig");
test {
    _ = model_weights;
}

pub const hc = @import("cuda_hc.zig");

pub const dflash = @import("cuda_dflash.zig");
pub const dflash_config = @import("cuda_dflash_config.zig");
pub const dflash_weights = @import("cuda_dflash_weights.zig");
pub const dflash_forward = @import("cuda_dflash_forward.zig");
pub const draft_policy = @import("cuda_draft_policy.zig");
pub const copy_drafts = @import("cuda_copy_drafts.zig");
test {
    _ = dflash;
    _ = dflash_config;
    _ = dflash_weights;
    _ = dflash_forward;
    _ = draft_policy;
    _ = copy_drafts;
}
test "analyze P9 CUDA loader, forward, graphs and commit without opening driver" {
    const std = @import("std");
    std.mem.doNotOptimizeAway(&dflash_weights.Loader.load);
    std.mem.doNotOptimizeAway(&dflash_forward.Forward.init);
    std.mem.doNotOptimizeAway(&dflash_forward.Forward.capture);
    std.mem.doNotOptimizeAway(&dflash.Multi.commitDrafts);
    std.mem.doNotOptimizeAway(&dflash.Multi.addTaps);
    std.mem.doNotOptimizeAway(&dflash.Multi.propose);
    _ = dflash_weights.Loader.load;
    _ = dflash_forward.Forward.init;
    _ = dflash_forward.Forward.capture;
    _ = dflash.Multi.commitDrafts;
    _ = dflash.Multi.addTaps;
    _ = dflash.Multi.propose;
}
test {
    _ = @import("p9_test.zig");
}
pub const dflash_package = @import("cuda_dflash_package.zig");
test "compile owned P9 package construction and cleanup" {
    const std = @import("std");
    std.mem.doNotOptimizeAway(&dflash_package.Package.open);
    std.mem.doNotOptimizeAway(&dflash_package.Package.deinit);
}
pub const mla = @import("cuda_mla.zig");
pub const matmuls = @import("cuda_matmuls.zig");
pub const weights_pkg = @import("glm_cuda.zig");
pub const engine = @import("cuda_engine.zig");
pub const entries = @import("cuda_entries.zig");
test {
    _ = mla;
    _ = matmuls;
    _ = weights_pkg;
    _ = engine;
    _ = entries;
}
pub const prefill = @import("cuda_prefill.zig");
pub const prefill_grid = @import("cuda_prefill_grid.zig");
pub const prefill_loop = @import("cuda_prefill_loop.zig");
pub const prefill_check = @import("cuda_prefill_check.zig");
test {
    _ = prefill;
    _ = prefill_grid;
    _ = prefill_loop;
    _ = prefill_check;
}

pub const generate = @import("cuda_generate.zig");

test {
    _ = generate;
}
pub const side = @import("cuda_side.zig");
test {
    _ = side;
}

pub const prompt_micro = @import("cuda_prompt_micro.zig");
test {
    _ = prompt_micro;
}

pub const prefill_split = @import("cuda_prefill_split.zig");
test {
    _ = prefill_split;
}

pub const ring_exchange = @import("cuda_ring_exchange.zig");
test {
    _ = ring_exchange;
}

pub const snapshot = @import("cuda_snapshot.zig"); // kept prompt states: save and restore of a sequence's state
test {
    _ = snapshot;
}

pub const cache_plan = @import("cuda_cache_plan.zig"); // the recipe's kept-prompt settings and memory plan
test {
    _ = cache_plan;
}

pub const spill = @import("cuda_spill.zig"); // the spill tier's disk side
test {
    _ = spill;
}

pub const l2pf = @import("cuda_l2pf.zig");
test {
    _ = l2pf;
}

pub const display_kv = @import("cuda_display_kv.zig");
pub const pool_capacity = @import("cuda_pool_capacity.zig");
test {
    _ = display_kv;
    _ = pool_capacity;
}

pub const vision = @import("cuda_vision.zig");
test {
    _ = vision;
}
pub const vision_check = @import("cuda_vision_check.zig");
test {
    _ = vision_check;
}
pub const vision_cudnn = @import("cuda_vision_cudnn.zig");
test {
    _ = vision_cudnn;
}
