Real GLM TP=2 capture and replay (TensorFold/Ash Hart reference)

The recipe worktree is read only. The capture launcher sources its config, mounts
these hooks through PYTHONPATH, and uses its docker GPU/RDMA/network privileges.
Only uniquely named glm-p1-* containers created by the launcher are removed.

    flock /tmp/claude-1000/gpu.lock env GLM_GPU_LOCKED=1 \
      python3 tools/zig/glm_capture.py --out glm-capture
    python3 tools/zig/glm_aot.py --capture glm-capture \
      --out zig/tests/cuda/glm_p1/aot \
      --zig zig/src/families/glm_cuda/cuda_triton_generated.zig
    ZIG_GLOBAL_CACHE_DIR=$PWD/../.zig-global-cache nice -n 19 zig build glm-p1-replay -j2
    flock /tmp/claude-1000/gpu.lock zig-out/bin/tf-glm-p1-replay \
      zig/tests/cuda/glm_p1/aot glm-capture/rank0
    flock /tmp/claude-1000/gpu.lock zig-out/bin/tf-glm-p1-replay \
      zig/tests/cuda/glm_p1/aot glm-capture/rank1

GPU-FREE must exist. The lock covers both reference ranks and cleanup. Startup,
eager decode and eager prefill launches retain the compiled signature, constexprs,
argument alignment, source, call sites, grid, compiler options and cached cubin.
Fixtures retain the union of tensor pointer spans per storage, rounded down to a
16-byte base, preserving offset pointers and aliasing. Each span has before/after
raw little-endian bytes. The replay restores the complete before image and checks
the complete after image, including untouched bytes and padding.

The large checkpoint pointer spans live in ignored glm-capture/, not in Git.
Small wrapper metadata and captured cubins can be kept separately. A snapshot
size limit is explicit in fixtures.json; skipped snapshots are never reported as
passing. Graph replays do not reenter JITFunction.run, so fixtures require eager
batched verify/drafter execution. Captured specializations cover this workload;
missing variants fail through aot.Set.find instead of approximating arithmetic.
