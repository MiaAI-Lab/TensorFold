//! The CUDA half of the root build: kernel fatbins with each Python extension's nvcc flags, the runtime, Nemotron, the CLI.

const std = @import("std");
const vision_build = @import("vision.zig");

/// Each .cu in zig/kernels/cuda (`src`, else `name`) with the flags its Python extension passes in `extra_cuda_cflags`.
const Kernel = struct { name: []const u8, flags: []const []const u8, src: ?[]const u8 = null, optional: bool = false, arch_specific: bool = false };

/// The torch-op replacements' qualification flags (runs/006): no contraction, no flush to zero.
const torch_ops = &[_][]const u8{ "-O3", "--fmad=false", "--ftz=false" };

/// IEEE division and square root, no contraction or flush: host glue references match bit for bit.
const glue = &[_][]const u8{ "-O3", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true" };

const kernels = [_]Kernel{
    // GLM package images: optional until their owning packages supply the CUDA sources.
    .{ .name = "glm_ops", .flags = torch_ops },
    .{ .name = "glm_foundations", .flags = &.{"-O3"} },
    .{ .name = "glm_kda", .optional = true, .flags = &.{ "-O3", "--fmad=false" } },
    .{ .name = "glm_kda_chunk", .optional = true, .flags = &.{ "-O3", "--fmad=false" } },
    .{ .name = "glm_hc", .optional = true, .flags = &.{"-O3"} },
    .{ .name = "glm_vision", .optional = true, .flags = &.{"-O3"} }, // the vision tower's norms, rotary, LayerNorm+GELU, conv GEMM chain, flash-attention order
    .{ .name = "glm_exl3", .optional = true, .flags = &.{"-O3"} },
    .{ .name = "glm_qmm", .optional = true, .flags = &.{"-O3"} },
    .{ .name = "glm_qmm_prefill", .optional = true, .flags = &.{"-O3"} },
    .{ .name = "gdn", .flags = &.{ "-O3", "--fmad=false" } }, // cuda/kernels/gdn.py, tensorfold_gdn_v2
    .{ .name = "probe", .flags = &.{"-O3"} },
    .{ .name = "qmm_group", .flags = &.{"-O3"} }, // cuda/kernels/qmm.py, tensorfold_qmm_v5
    .{ .name = "qmm_prefill", .flags = &.{"-O3"} },
    .{ .name = "experts", .flags = &.{"-O3"} }, // cuda/experts.py, tensorfold_experts_v7
    .{ .name = "experts_prefill", .flags = &.{"-O3"} },
    .{ .name = "experts_pack", .flags = &.{"-O3"} },
    .{ .name = "prefill_attention", .flags = &.{ "-O3", "--fmad=false" } }, // tensorfold_prefill_attention_v1
    .{ .name = "scan_rows", .flags = &.{ "-O3", "--fmad=false" } }, // nemotron_h/cuda/mamba.py
    .{ .name = "nemotron_ops", .flags = &.{"-O3"} }, // ours: Nemotron's serial feed, routed plan and rests
    .{ .name = "affine4_pack", .flags = &.{"-O3"} }, // ours: the MLX affine-4 repack into qlinear's tiles
    .{ .name = "fp8_experts", .flags = &.{"-O3"} }, // fp8/experts.cu's device code, tensorfold_fp8_experts_v6
    .{ .name = "nvfp4_experts", .flags = &.{"-O3"} }, // nvfp4/experts.cu's device code, the NVFP4 expert kernels
    .{ .name = "lane_gemv", .flags = &.{"-O3"} }, // ours: qmm_group's arithmetic, a column tile's K slices in one CTA
    .{ .name = "sample", .flags = &.{ "-O3", "--fmad=false", "--ftz=false" } }, // ours: the Metal engine's keyed draws
    .{ .name = "train", .flags = &.{"-O3"} }, // ours: Sliding Weights' learning step (train.metal's kernels)
    .{ .name = "nemotron_norms", .flags = glue }, // ours, each with a host reference (glue_ref.zig): Nemotron's glue
    .{ .name = "nemotron_route", .flags = glue },
    .{ .name = "nemotron_mamba", .flags = glue },
    .{ .name = "nemotron_attention", .flags = glue },
    .{ .name = "nemotron_keyed", .flags = glue },
    .{ .name = "qmmf", .flags = &.{"-O3"} }, // nvfp4/qmmf.cu's device code (FP8G, FP4), tensorfold_nvfp4_v3
    .{ .name = "prompt16", .flags = &.{"-O3"} }, // nvfp4/prompt.cu's FP4 and FP8 tile 4, tensorfold_nvfp4_prompt_v1
    .{ .name = "modelopt_pack", .flags = &.{"-O3"} }, // ours: ModelOpt tensors into the lane, prompt and expert layouts
    .{ .name = "torch_argmax", .src = "torch_ops/argmax", .flags = torch_ops },
    .{ .name = "torch_topk", .src = "torch_ops/topk", .flags = torch_ops },
    .{ .name = "torch_pointwise", .src = "torch_ops/pointwise", .flags = torch_ops },
    .{ .name = "torch_indexing", .src = "torch_ops/indexing", .flags = torch_ops },
    .{ .name = "torch_movement", .src = "torch_ops/movement", .flags = torch_ops },
    .{ .name = "torch_nemotron_constants", .src = "torch_ops/nemotron_constants", .flags = torch_ops },
    // Flash Next (qwen4_exp): copies of its extensions' device code (zig/tests/cuda/copies.py), the same flags
    .{ .name = "fn_gdn", .flags = &.{ "-O3", "--fmad=false" } }, // qwen4_exp/cuda/gdn.py, tensorfold_qwen4_exp_gdn
    .{ .name = "fn_gdn_io", .flags = &.{ "-O3", "--fmad=false" } }, // qwen4_exp/cuda/gdn_io.py, tensorfold_qwen4_exp_gdn_io
    .{ .name = "fn_gdn_prefill", .flags = &.{ "-O3", "--fmad=false" } }, // cuda/kernels/gdn.py, tensorfold_gdn_v2
    .{ .name = "fn_gdn_tree", .flags = &.{ "-O3", "--fmad=false" } }, // cuda/kernels/gdn.py, tensorfold_gdn_v2
    .{ .name = "fn_nvfp4_experts", .flags = &.{"-O3"} }, // cuda/nvfp4/linear.py, tensorfold_nvfp4_v3
    .{ .name = "fn_qmm", .flags = &.{"-O3"} }, // cuda/kernels/qmm.py, tensorfold_qmm_v5
    .{ .name = "fn_qmm_prefill", .flags = &.{"-O3"} }, // cuda/kernels/qmm.py, tensorfold_qmm_v5
    .{ .name = "fn_pack", .flags = &.{"-O3"} }, // ours: the n-gram table's GPU gather (flashnext/cuda_weights.zig)
    .{ .name = "torch_fn_ops", .src = "torch_ops/fn_ops", .flags = torch_ops }, // ours: Flash Next torch ops (SwiGLU, sampler packing)
    .{ .name = "torch_fn_logsumexp", .src = "torch_ops/fn_logsumexp", .flags = torch_ops }, // ours: torch.logsumexp in ATen's order
    .{ .name = "fn_experts_prompt", .flags = &.{"-O3"} }, // ours: routed NVFP4 experts reading weights once a prompt call (fn_nvfp4_experts' bits)
    // INT4-AutoRound checkpoint (GPTQ int4 experts and head, 128x128-block FP8 dense linears)
    .{ .name = "fn_qmmf", .flags = &.{"-O3"} }, // cuda/nvfp4/qmmf.cu (FP8G lane matmul), tensorfold_nvfp4_v3
    .{ .name = "fn_int4", .flags = &.{"-O3"} }, // ours: GPTQ int4 g128 routed experts and lm_head (W4A16, fp32 sums)
    .{ .name = "fn_nvfp4_shape", .flags = &.{"-O3"} }, // ours (decode D3): fn_nvfp4_experts' unit arithmetic in other launch shapes
    .{ .name = "fn_qmm_cluster", .flags = &.{"-O3"} }, // ours (decode D5): qmm_kernel's K-slice cluster forms for the MTP head in 4-bit
    .{ .name = "fn_roce", .flags = &.{"-O3"} }, // ours (decode D1): the one-shot RoCE all-gather's GPU half (cuda/roce.zig)
    // GLM-5.3-Flash (glm5_next), P5 MLA/DSA: the only CUDA in the attention path (the rest is Triton, replayed from cubins)
    .{ .name = "glm_msa", .flags = &.{ "-O3", "-lineinfo" } }, // glm5_next/cuda/msa.py, tensorfold_glm_flash_msa_v2 (TF_GLM_MSA=1)
};

/// torch.utils.cpp_extension's own nvcc flags (torch 2.13): C++20 and which half/bf16 operators the headers define.
const torch_flags = [_][]const u8{
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-std=c++20",
};

/// core/stagger.zig, the segment schedule the Metal and CUDA runners share, as its own module (no imports).
fn stagger(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/core/stagger.zig"), .target = target, .optimize = optimize });
}

/// The runtime module for `target`; `with_kernels` false builds it host-only (empty images).
fn runtime(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, images: []const ?std.Build.LazyPath) *std.Build.Module {
    const options = b.addOptions();
    var with = images.len > 0;
    if (images.len > 0) for (kernels, images) |k, i| {
        if (!k.optional) with = with and i != null;
    };
    for (kernels, 0..) |k, i| if (k.optional) {
        options.addOption(bool, k.name, images.len > 0 and images[i] != null);
    };
    options.addOption(bool, "with_kernels", with);
    const cuda = b.createModule(.{ .root_source_file = b.path("zig/src/cuda/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    cuda.addOptions("kernel_options", options);
    // the one-shot RoCE all-gather's proxy (decode D1): plain C over libibverbs' header, the library opened at run time
    // (roce_proxy_stub.c, which refuses every open, where /usr/include has no infiniband/verbs.h)
    const verbs = if (std.Io.Dir.accessAbsolute(b.graph.io, "/usr/include/infiniband/verbs.h", .{})) |_| true else |_| false;
    cuda.addCSourceFile(.{ .file = b.path(if (verbs) "zig/src/cuda/roce_proxy.c" else "zig/src/cuda/roce_proxy_stub.c"), .flags = &.{ "-std=gnu11", "-O2", "-idirafter", "/usr/include" } });
    cuda.addImport("stagger", stagger(b, target, optimize));
    if (images.len > 0) for (kernels, images) |k, image| {
        if (image) |file| cuda.addAnonymousImport(b.fmt("fatbin_{s}", .{k.name}), .{ .root_source_file = file });
    };
    return cuda;
}

fn family(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, draft_ids: *std.Build.Module) struct { core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, flashnext: *std.Build.Module, glm_cuda: *std.Build.Module, glm: *std.Build.Module, tokenizer: *std.Build.Module, heat: *std.Build.Module } {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("tokenizer", tokenizer);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const heat = b.createModule(.{ .root_source_file = b.path("zig/src/core/heat.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const nemotron = b.createModule(.{ .root_source_file = b.path("zig/src/families/nemotron/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    nemotron.addImport("cuda", cuda);
    nemotron.addImport("core", core);
    nemotron.addImport("lanes", lanes);
    nemotron.addImport("heat", heat);
    nemotron.addImport("nemotron_draft_ids", draft_ids);
    // Flash Next on CUDA: its root sits beside the Metal family's host files (config.zig), which it shares
    const flashnext = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    flashnext.addImport("cuda", cuda);
    flashnext.addImport("core", core);
    flashnext.addImport("lanes", lanes);
    const glm_cuda = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_cuda/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    glm_cuda.addImport("cuda", cuda);
    glm_cuda.addImport("core", core);
    glm_cuda.addImport("lanes", lanes);
    glm_cuda.addImport("flashnext", flashnext);
    // cuda_mla.zig stays a root so its host tests keep running; the family root re-exports it as mla.
    const glm = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_cuda/cuda_mla.zig"), .target = target, .optimize = optimize, .link_libc = true });
    glm.addImport("cuda", cuda);
    return .{ .glm_cuda = glm_cuda, .glm = glm, .core = core, .lanes = lanes, .nemotron = nemotron, .flashnext = flashnext, .tokenizer = tokenizer, .heat = heat };
}

/// Linux targets: fatbins (-Dnvcc builds them, -Dfatbins embeds prebuilt ones), `tensorfold` and `tf-cuda-test`.
pub fn targets(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options) void {
    const nvcc = b.option([]const u8, "nvcc", "nvcc (or a wrapper) that builds the CUDA kernel fatbins");
    const prebuilt = b.option([]const u8, "fatbins", "absolute directory of prebuilt <name>.fatbin files to embed");
    const sms = b.option([]const u8, "sm", "SASS targets, comma separated (default 121; 80, 86, 89, 120 and 121 build)") orelse "121";
    const strip = b.option(bool, "strip", "No debug info in the CUDA executables: no build machine paths leave with them") orelse false;
    // the compiler's version text is an input of every fatbin, so a new nvcc rebuilds them all
    const version: ?std.Build.LazyPath = if (prebuilt == null and nvcc != null) blk: {
        const run = b.addSystemCommand(&.{ nvcc.?, "--version" });
        run.has_side_effects = true;
        break :blk run.captureStdOut(.{});
    } else null;
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    const fatbin_step = b.step("fatbins", "Build and install the CUDA kernel fatbins alone");
    for (kernels, &images) |k, *image| {
        if (prebuilt) |dir| {
            const path = b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) });
            if (!k.optional or exists(b, path)) image.* = b.graph.cwdRelativePath(path);
        } else if (nvcc) |tool| {
            if (!k.optional or exists(b, b.fmt("zig/kernels/cuda/{s}.cu", .{k.src orelse k.name}))) image.* = fatbin(b, tool, version.?, k, sms);
        }
        if (image.*) |file| fatbin_step.dependOn(&b.addInstallFile(file, b.fmt("fatbin/{s}.fatbin", .{k.name})).step);
    }
    const cuda = runtime(b, target, optimize, if (nvcc != null or prebuilt != null) &images else &.{});
    const glm_matmuls = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_cuda/cuda_matmuls.zig"), .target = target, .optimize = optimize, .link_libc = true });
    glm_matmuls.addImport("cuda", cuda);
    const mods = family(b, target, optimize, cuda, draft_ids);
    const glm_check = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_p23_check.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm_cuda", .module = mods.glm_cuda } } });
    b.step("glm-matmuls", "Build tf-glm-matmuls: byte-check P2/P3 Python fixtures (GPU)").dependOn(&b.addInstallArtifact(b.addExecutable(.{ .name = "tf-glm-matmuls", .root_module = glm_check }), .{}).step);
    const glm_p1 = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_p1/replay.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm_cuda", .module = mods.glm_cuda } } });
    b.step("glm-p1-replay", "Build tf-glm-p1-replay: byte-check captured Python Triton pointer spans (GPU)").dependOn(&b.addInstallArtifact(b.addExecutable(.{ .name = "tf-glm-p1-replay", .root_module = glm_p1 }), .{}).step);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    cli.addImport("flashnext", mods.flashnext);
    cli.addImport("glm_cuda", mods.glm_cuda);
    const native = engines(b, target, optimize, cuda, mods.lanes, mods.nemotron, mods.flashnext, mods.glm_cuda);
    cli.addOptions("build_options", build_options);
    cli.addImport("checkpoint_cli", checkpointCli(b, target, optimize, strip, native.engines));
    b.installArtifact(b.addExecutable(.{ .name = "tensorfold", .root_module = cli }));
    const runner = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/main.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip });
    runner.addImport("cuda", cuda);
    runner.addImport("lanes", mods.lanes);
    runner.addImport("nemotron", mods.nemotron);
    b.installArtifact(b.addExecutable(.{ .name = "tf-cuda-test", .root_module = runner }));
    nativeServer(b, target, optimize, cuda, mods.lanes, mods.nemotron, mods.flashnext, mods.glm_cuda, mods.tokenizer, build_options, true).root_module.strip = strip;
    // Flash Next's two-rank transport test (zig/tests/cuda/flashnext/box/comm.sh): its own step, not in `install`
    const comm_test = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/flashnext/comm_test.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "flashnext", .module = mods.flashnext } } });
    b.step("flashnext-comm", "tf-flashnext-comm RANK MASTER PORT: two-rank all-gathers, exact and timed").dependOn(&b.addInstallArtifact(b.addExecutable(.{ .name = "tf-flashnext-comm", .root_module = comm_test }), .{}).step);
    // Flash Next's weights against the oracle's digests (zig/tests/cuda/flashnext/weights_check.zig): its own step
    const weights_check = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/flashnext/weights_check.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "flashnext", .module = mods.flashnext } } });
    const hc_check = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_hc_check.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm_cuda", .module = mods.glm_cuda } } });
    b.step("glm-hc-check", "Build tf-glm-hc-check: HC partial byte oracle (GPU)").dependOn(&b.addInstallArtifact(b.addExecutable(.{ .name = "tf-glm-hc-check", .root_module = hc_check }), .{}).step);
    // GLM vision tower: the GPU tower against the Python reference's stages (zig/tests/vision/tower)
    const vision_check = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_vision_check.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm_cuda", .module = mods.glm_cuda } } });
    b.step("glm-vision-check", "Build tf-glm-vision-check: the vision tower stage by stage against the Python tower (GPU)").dependOn(&b.addInstallArtifact(b.addExecutable(.{ .name = "tf-glm-vision-check", .root_module = vision_check }), .{}).step);
    // GLM P5: glm_msa.cu on the GPU against the host attention (zig/tests/cuda/glm_p5/msa_check.zig): its own step
    const msa_check = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_p5/msa_check.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm_cuda", .module = mods.glm_cuda } } });
    b.step("glm-p5-msa", "tf-glm-p5-msa [ROWS] [CACHE_ROWS]: glm_msa.cu against the host attention, row independence").dependOn(&b.addInstallArtifact(b.addExecutable(.{ .name = "tf-glm-p5-msa", .root_module = msa_check }), .{}).step);
    b.step("flashnext-weights", "tf-flashnext-weights SNAPSHOT [WEIGHTS.json]: the loaded weights' digests against the oracle's").dependOn(&b.addInstallArtifact(b.addExecutable(.{ .name = "tf-flashnext-weights", .root_module = weights_check }), .{}).step);
    // GLM KDA: chain vs wide vs replay, and a segmented window vs a solo run of each segment (needs the fatbins)
    const kda_check = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_kda_check.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm", .module = mods.glm_cuda } } });
    b.step("glm-kda", "tf-glm-kda: KDA chain vs replay, segments vs solo, chunk launch (GPU)").dependOn(&b.addInstallArtifact(b.addExecutable(.{ .name = "tf-glm-kda", .root_module = kda_check }), .{}).step);
}

/// The CUDA engines a native server opens (native/cuda.zig), over the given runtime and families.
fn engines(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, flashnext: *std.Build.Module, glm_cuda: *std.Build.Module) struct { api: *std.Build.Module, engines: *std.Build.Module } {
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const mod = b.createModule(.{
        .root_source_file = b.path("zig/src/native/cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes }, .{ .name = "nemotron", .module = nemotron }, .{ .name = "flashnext", .module = flashnext }, .{ .name = "glm_cuda", .module = glm_cuda } },
    });
    return .{ .api = api, .engines = mod };
}

/// The checkpoint subcommands' module: `models`, `info` and `pull` over the CUDA families.
fn checkpointCli(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, strip: ?bool, engines_mod: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = optimize, .link_libc = true, .strip = strip, .imports = &.{.{ .name = "native_engines", .module = engines_mod }} });
}

/// `zig build native`: tensorfold-native with the CUDA engines into zig-out/native/bin, as the Metal build makes it.
fn nativeServer(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, flashnext: *std.Build.Module, glm_cuda: *std.Build.Module, tokenizer: *std.Build.Module, build_options: *std.Build.Step.Options, install_native: bool) *std.Build.Step.Compile {
    const m = engines(b, target, optimize, cuda, lanes, nemotron, flashnext, glm_cuda);
    // the HTTP side keeps its safety checks; the engine below it runs at `optimize` (the tokenizer is the family's)
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/server/main.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = m.api }, .{ .name = "tokenizer", .module = tokenizer }, .{ .name = "template", .module = template }, .{ .name = "native_engines", .module = m.engines }, .{ .name = "checkpoint_cli", .module = checkpointCli(b, target, .ReleaseSafe, null, m.engines) } },
    }) });
    exe.root_module.addOptions("build_options", build_options);
    vision_build.attach(b, exe.root_module, target, .ReleaseSafe);
    if (!install_native) {
        exe.root_module.strip = true;
        return exe;
    }
    const install = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "native/bin" } } });
    b.step("native", "tensorfold-native with the CUDA engines into zig-out/native/bin").dependOn(&install.step);
    return exe;
}

/// Host unit tests of the CUDA runtime, the backend-neutral core, the lane core and the CUDA family (no GPU), on any host.
pub fn hostTests(b: *std.Build, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, all: *std.Build.Step) void {
    const step = b.step("test-cuda-host", "The CUDA side's host unit tests alone (no GPU work)");
    all.dependOn(step);
    const host = b.graph.host;
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/build/check_fatbins.zig"), .target = host }) })).step);
    const cuda = runtime(b, host, .debug, &.{});
    const mods = family(b, host, .debug, cuda, draft_ids);
    const layer_tool = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_layer_check.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "glm_cuda", .module = mods.glm_cuda }} });
    b.step("glm-layer-check", "Build tf-glm-layer-check REFERENCE_DIR ACTUAL_DIR (CPU byte oracle)").dependOn(&b.addInstallArtifact(b.addExecutable(.{ .name = "tf-glm-layer-check", .root_module = layer_tool }), .{}).step);
    const prefill_tool = b.createModule(.{ .root_source_file = b.path("zig/tests/cuda/glm_prefill_check.zig"), .target = host, .optimize = .debug, .link_libc = true, .imports = &.{.{ .name = "glm_cuda", .module = mods.glm_cuda }} });
    b.step("glm-prefill-check", "Build tf-glm-prefill-check REFERENCE_DIR ACTUAL_DIR (prompt chunk byte oracle)").dependOn(&b.addInstallArtifact(b.addExecutable(.{ .name = "tf-glm-prefill-check", .root_module = prefill_tool }), .{}).step);
    const glm_test = b.addRunArtifact(b.addTest(.{ .root_module = mods.glm_cuda }));
    b.step("test-glm-cuda-host", "GLM CUDA CPU contracts (foundations, packed sampling, rounds, lanes)").dependOn(&glm_test.step);
    const serve_test = b.step("test-glm-serve", "GLM serving and drafted-versus-serial CPU contracts");
    serve_test.dependOn(&glm_test.step);
    const cli_tests = b.createModule(.{
        .root_source_file = b.path("zig/src/cli/glm_cuda.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "glm_cuda", .module = mods.glm_cuda } },
    });
    const cli_test = b.addRunArtifact(b.addTest(.{ .root_module = cli_tests }));
    step.dependOn(&cli_test.step);
    serve_test.dependOn(&cli_test.step);
    b.step("test-glm-p9", "GLM DFlash2 and copy draft CPU contracts").dependOn(&glm_test.step);
    b.step("test-glm-p7", "GLM sampling/commit/rounds CPU fixtures").dependOn(&glm_test.step);
    b.step("test-glm-layer-check", "P6 forward and layer byte oracle host contracts").dependOn(&glm_test.step);
    b.step("test-glm-prefill-check", "P8 prompt chunk planner and layer byte oracle host contracts").dependOn(&glm_test.step);
    step.dependOn(&glm_test.step);
    const native_modules = engines(b, host, .debug, cuda, mods.lanes, mods.nemotron, mods.flashnext, mods.glm_cuda);
    const native = native_modules.engines;
    // the HTTP server's unit tests (zig/src/server/root.zig), which the macOS build runs, on Linux hosts too
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = host, .optimize = .debug, .link_libc = true });
    const server_tests = b.createModule(.{
        .root_source_file = b.path("zig/src/server/root.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = native_modules.api }, .{ .name = "tokenizer", .module = mods.tokenizer }, .{ .name = "template", .module = template } },
    });
    vision_build.attach(b, server_tests, host, .debug);
    const server_test = b.addRunArtifact(b.addTest(.{ .root_module = server_tests }));
    step.dependOn(&server_test.step);
    serve_test.dependOn(&server_test.step);
    b.step("test-server-cpu", "The HTTP server's unit tests (routes, templates, tool parsing) without a GPU").dependOn(&server_test.step);
    const glm_weights_mod = b.createModule(.{
        .root_source_file = b.path("zig/src/families/glm_cuda/glm_cuda.zig"),
        .target = host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "core", .module = mods.core }},
    });
    const glm_weights = b.addRunArtifact(b.addTest(.{ .root_module = glm_weights_mod }));
    b.step("test-glm-weights", "GLM-5.3-Flash CUDA config and EXL3 weight-plan tests (CPU, headers only)").dependOn(&glm_weights.step);
    step.dependOn(&glm_weights.step);
    for ([_]*std.Build.Module{ cuda, mods.core, mods.lanes, mods.nemotron, mods.flashnext, mods.glm_cuda, mods.glm, mods.heat, native, stagger(b, host, .debug) }) |m| step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = m })).step);
    const glm_mm = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm_cuda/cuda_matmuls.zig"), .target = host, .optimize = .debug, .link_libc = true });
    glm_mm.addImport("cuda", cuda);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = glm_mm })).step);
    const cli = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cuda_main.zig"), .target = host, .optimize = .debug, .link_libc = true });
    cli.addImport("cuda", cuda);
    cli.addImport("core", mods.core);
    cli.addImport("lanes", mods.lanes);
    cli.addImport("nemotron", mods.nemotron);
    cli.addImport("flashnext", mods.flashnext);
    cli.addImport("glm_cuda", mods.glm_cuda);
    cli.addOptions("build_options", build_options);
    cli.addImport("checkpoint_cli", checkpointCli(b, host, .debug, false, native_modules.engines));
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = cli })).step);
}

/// nvcc -fatbin with torch's flags, the kernel's own and one -gencode per SASS target, as the Python build passes them.
fn fatbin(b: *std.Build, nvcc: []const u8, version: std.Build.LazyPath, k: Kernel, sms: []const u8) std.Build.LazyPath {
    const run = b.addSystemCommand(&.{ nvcc, "-fatbin" });
    run.addFileInput(version);
    run.addArgs(&torch_flags);
    run.addArgs(k.flags);
    const a = if (k.arch_specific) "a" else "";
    var it = std.mem.tokenizeScalar(u8, sms, ',');
    while (it.next()) |sm| run.addArg(b.fmt("-gencode=arch=compute_{s}{s},code=sm_{s}{s}", .{ sm, a, sm, a }));
    run.addArgs(&.{ "-MD", "-MF" });
    _ = run.addDepFileOutputArg2(b.fmt("{s}.d", .{k.name}), .{});
    run.addArg("-o");
    const out = run.addOutputFileArg(b.fmt("{s}.fatbin", .{k.name}));
    run.addFileArg(b.path(b.fmt("zig/kernels/cuda/{s}.cu", .{k.src orelse k.name})));
    return out;
}

/// Cross-build a release server, embedding the same fatbins on either host CPU.
pub fn distServer(b: *std.Build, target: std.Build.ResolvedTarget, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, prebuilt: ?[]const u8) *std.Build.Step.Compile {
    var images: [kernels.len]?std.Build.LazyPath = @splat(null);
    if (prebuilt) |dir| for (kernels, &images) |k, *image| {
        const path = b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) });
        if (!k.optional or exists(b, path)) image.* = b.graph.cwdRelativePath(path);
    };
    const cuda = runtime(b, target, .fast, if (prebuilt != null) &images else &.{});
    const mods = family(b, target, .fast, cuda, draft_ids);
    return nativeServer(b, target, .fast, cuda, mods.lanes, mods.nemotron, mods.flashnext, mods.glm_cuda, mods.tokenizer, build_options, false);
}

/// SASS targets every release fatbin holds: Ampere (sm_86), RTX 50 and RTX PRO Blackwell (sm_120), GB10 (sm_121).
pub const release_sms = "86,120,121";

/// Validate the complete named input set, including images unused by today's server, and each file's release targets.
pub fn checkDistFatbins(b: *std.Build, dir: []const u8) *std.Build.Step {
    const tool = b.addExecutable(.{ .name = "check-fatbins", .root_module = b.createModule(.{ .root_source_file = b.path("zig/build/check_fatbins.zig"), .target = b.graph.host }) });
    const check = b.addRunArtifact(tool);
    check.addArg(release_sms);
    for (kernels) |k| {
        if (k.optional) continue;
        check.addFileArg(b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{k.name}) })));
    }
    return &check.step;
}

fn exists(b: *std.Build, path: []const u8) bool {
    std.Io.Dir.cwd().access(b.graph.io, path, .{}) catch return false;
    return true;
}
