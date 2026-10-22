//! Flash Next (qwen4_exp) on CUDA in Zig: the Python 0.6.5 engine's kernels and layouts for NVIDIA's ModelOpt NVFP4
//! export, so tokens match it bit for bit on one GPU, and the same arithmetic split over two ranks (work/PLAN.md).
//! The Metal family's host files (config.zig, affine.zig, ...) reach core by relative paths, so this module reads the
//! checkpoint through its own cuda_*.zig files and the `core` module.

pub const native = @import("cuda_native.zig");
pub const ngram = @import("cuda_ngram.zig");
pub const rope = @import("cuda_rope.zig");
pub const sampler = @import("cuda_sampler.zig");
pub const link = @import("cuda_link.zig");
pub const comm = @import("cuda_comm.zig");
pub const kernels = @import("cuda_kernels.zig");
pub const state = @import("cuda_state.zig");
pub const decode = @import("cuda_decode.zig");
pub const triton = @import("cuda_triton.zig");
pub const torch_ops = @import("cuda_torch_ops.zig");
pub const config = @import("cuda_config.zig");
pub const layouts = @import("cuda_layouts.zig");
pub const weights = @import("cuda_weights.zig");
pub const forward = @import("cuda_forward.zig");
pub const vmm = @import("cuda_vmm.zig");
pub const nucleus = @import("cuda_nucleus.zig");
pub const mtp = @import("cuda_mtp.zig");
pub const mtp_q4 = @import("cuda_mtp_q4.zig");
pub const engine = @import("cuda_engine.zig");
pub const prompt = @import("cuda_prompt.zig");
pub const fp4_serial = @import("cuda_fp4_serial.zig");
pub const prof = @import("cuda_prof.zig");
pub const moe_prompt = @import("cuda_moe_prompt.zig");
pub const fp8 = @import("cuda_fp8.zig");
pub const int4 = @import("cuda_int4.zig");
pub const int4_check = @import("cuda_int4_check.zig");
pub const Engine = engine.Engine;

test {
    _ = native;
    _ = ngram;
    _ = rope;
    _ = sampler;
    _ = link;
    _ = comm;
    _ = kernels;
    _ = state;
    _ = decode;
    _ = triton;
    _ = torch_ops;
    _ = fp4_serial;
    _ = config;
    _ = layouts;
    _ = weights;
    _ = forward;
    _ = vmm;
    _ = nucleus;
    _ = mtp;
    _ = engine;
    _ = prompt;
    _ = prof;
    _ = moe_prompt;
    _ = fp8;
    _ = int4;
    _ = int4_check;
}

pub const mirror = @import("cuda_mirror.zig");
pub const lane_backend = @import("cuda_lanes.zig");
