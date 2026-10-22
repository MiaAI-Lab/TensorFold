# Package J: GLM display KV and startup pool sizing

Ash Hart (`ashhart`) and TensorFold contributors authored the reference and Zig engine;
[jschmied](https://github.com/jschmied) authored native CUDA serving and the family registry
([TensorFold PR #443](https://github.com/ashhart/TensorFold/pull/443));
[b12x / local-inference-lab](https://github.com/local-inference-lab) authored RoCEnante (Apache-2.0).
Existing credits are retained. New modules identify the Python functions they port.
No banned-source code was read or used. No network, GPU, docker or ssh command was run.

## Reference and behavior

Read `../zig-notes/BRIEF-recipe.md` and `BRIEF.md` first, then `r107-gap.md` and `HANDOFF-ZIG.md`.
Read the recipe from the local git object database with exactly:

```sh
git -C /home/mia/newMod/GLM-5.3-Flash-EXL3-TensorFold show origin/main:scripts/config.sh
```

Read-only Python sources: `../eng-v111/src/tensorfold/families/glm5_next/cuda/`
`display_kv.py`, `engine.py`, `forward.py`, `pool.py`, `multi.py`, plus shared
`tensorfold/cuda/capacity.py` and `geometry.py`.

* `DISPLAY_KV_MIB` exports `TF_GLM_DISPLAY_KV_MIB`: default 0; nonzero must be a multiple
  of 16, at most 2032 MiB. It adds GB10 firmware display-reservation bytes excluded from
  host `MemAvailable`, above the ordinary cache cap, for `--parallel > 1`.
* `DISPLAY_KV_BACKEND` exports `TF_GLM_DISPLAY_KV_BACKEND`: `drm` by default, or `dispram`.
  `TF_GLM_DISPLAY_KV_CARD` defaults to `/dev/dri/card0`. Both refuse connected outputs
  found in `/sys/class/drm/<card>-*/status`.
* The fewest first DSA latent planes sufficient to hold the credited bytes are carved
  from a contiguous span: ordinary memory, followed by the display reservation.
  The ordinary part is `count * plane_bytes - display_bytes`; FP8 rows stay 528 bytes.
  Pooled keys, index rings, values and model arithmetic retain their existing formats.
* DRM creates/maps a dumb buffer and registers the ordinary and display halves with CUDA
  flags DEVICEMAP and DEVICEMAP|IOMEMORY respectively. The device pointers must be adjacent.
  Missing devices, short mappings, connected displays, registration errors and nonadjacent
  pointers stop startup; there is no ordinary-memory fallback for credited display bytes.
* `dispram` calls the installed kindling client's `available()` / `map_glued(total, device)`
  in this same process via a dynamically loaded shared CPython library. It checks the returned
  pointer and tail credit. No client implementation is copied. Requires its Python dependencies,
  `PYTHONPATH=/opt/kindling/dispram/python`, `/run/dispram`, and a shared CPython 3.10–3.13 library.
  As in the reference, dispram allocations/client/library live for the process lifetime.
* `MEMORY_RESERVE_GIB` exports `TENSORFOLD_MEMORY_RESERVE_GIB`; recipe default 14.5 GiB,
  increased by the launcher for parallelism/window size. `KV_POOL_GIB` exports `TF_GLM_CACHE_GIB`
  (recipe 12.5 GiB for TP=2, Python engine default 3 GiB). Canonical engine env names take
  precedence; direct recipe names also work as fallbacks. Reserve defaults without a recipe
  export follow TensorFold's integrated/discrete memory floors and shares.

## Implemented

`cuda_display_kv.zig` implements parsing, headless refusal, plane selection, DRM registration,
unwinding and the optional installed dispram client bridge. `cuda_state.zig` places the selected
latent planes in the span and owns DRM cleanup; duplicate live display allocations are refused.

`cuda_pool_capacity.zig` reads unified-memory host availability from streamed `/proc/meminfo`
(discrete devices use CUDA counts), subtracts the reserve and caps ordinary surplus by
`TF_GLM_CACHE_GIB`. The GLM engine sizes the pool at startup after allocating its fixed arenas
and drafter, aligns to 2048-token extents, credits display bytes above the cap, and checks the
actual Zig cache/decode-scratch layouts against the budget. TP=2 takes the smaller rank capacity
before final allocation and verify graph capture. Canonical reserve/cache/display settings
participate in startup setting agreement.

`Engine.pool` now owns `cuda_pool.zig` extents. New sequences reserve their requested prompt/reply
limit instead of a fixed full-context slice per slot; free/cancel releases the extent.
`Engine.poolStatus()` returns total tokens and an atomic published free count, so HTTP threads
never traverse the lane thread's mutable extent list.

Server hookup: `cuda_native.Loaded.read_status` -> native CUDA `Host` ->
`LaneHost.extra_status` -> `Status.pool_tokens` / `pool_free_tokens` -> `/health`.
This worktree did **not** contain package F's fields or `zig/src/server/glm_health.zig`, so it adds
those minimal optional fields and a small `glm_health.put` helper here. When merging F, retain
one copy of the fields/helper and this engine callback; unrelated families omit unknown counts.

The TP=2 server qualification launcher now passes the memory/display environment to both ranks,
passes the DRM card when needed, and mounts the dispram client/socket for that backend.
`GLM_POOL_CHECK=1` runs `check_displaykv.py` and stops both containers while still holding the lock.
The check observes allocated/free pool tokens during eight requests, compares their greedy replies
and completion counts to solo, and checks that all extents return after completion.

## Exactness and remaining limits

No floating-point model kernel, reduction or quantizer changed. **GPU/token parity is unqualified.**
Pool token counts are intentionally based on Zig's live allocations/layouts rather than Python's
weight/workspace estimates; they are **not claimed identical** to Python on the same host.
Python's kept-state entry charges are absent because this branch has no kept-prompt implementation.
That integration belongs with package E. The recipe's startup context auto-refit and prompt-cache
accounting are not added here. Display placement is ignored for a single stream, as in Python's
pool setup; enabled display settings are parsed/validated at engine startup.

Captured Triton variants are keyed on selection buckets and pointer alignment. Larger automatic
pools and display-backed addresses may require new AOT captures. A `MissingTritonVariant` remains
fatal; this change does not substitute approximate kernels. Verify kernels at the pool size logged
at startup before claiming production readiness. DRM success, dispram imports/VMM, real free-memory
accounting, TP rank agreement, CUDA graph capture/replay and cancellation need hardware checks.

## CPU validation

From this worktree root, the requested commands were used:

# Package I: GLM CUDA checkpoint MTP drafter

Implemented in this worktree; CPU build/tests pass. GPU qualification is outstanding.
No GPU, Docker, SSH or network command was executed. No cubins were invented or
captured, no other worktree was edited, and nothing was pushed.

Ash Hart and the TensorFold contributors authored the reference Python MTP,
weight loading, sampling and Triton arithmetic. jschmied authored the Zig CUDA
serving path (TensorFold PR #443). Existing transport credits are retained:
b12x by local-inference-lab authored RoCEnante (Apache-2.0). New source headers
identify the Python functions ported. Sources were the requested local reference
and recipe git objects; the banned sources were not used.

## Done

- Read `../zig-notes/BRIEF-recipe.md`, `BRIEF.md`, `r107-gap.md`, `HANDOFF-ZIG.md`
  and applicable ancestor instructions before implementing. Read the recipe via
  `git -C /home/mia/newMod/GLM-5.3-Flash-EXL3-TensorFold show origin/main:scripts/config.sh`.
- Added `DRAFTER=mtp` selection and `TF_GLM_MTP=auto|0|1`. DFlash2 stays the
  default. MTP drafting requires parallel=1; missing/disabled heads fail closed.
  Explicit `1` also prepares the optional head beside single-stream DFlash2 or
  drafts-off. Rank settings agreement includes the drafter, MTP and draft quantizer.
- Prepared checkpoint layer 45: enorm, hnorm, EH projection, plain input/post
  norms, DSA/indexer, shared/routed MoE experts, shared-head norm, and a separate
  TP-sliced Q4 vocabulary head (`TF_GLM_DRAFT_QUANT=minmax|mse`). Reused the
  existing weight slicing, Q4 packing and GLM kernels. Target weights/head retain
  their existing arithmetic.
- Added independent MTP FP8 latent/pooled caches and index rings. Prompt absorption
  pairs final-normalized target rows with their next prompt tokens, saves the last
  target row for the first draw, and uses prefill arithmetic. Position zero's
  token embedding is zeroed before normalization.
- Decode concatenates separately normalized embedding/hidden rows, projects EH,
  runs plain DSA and MoE residuals, normalizes with shared_head.norm and computes
  proposal-only logits. Residual replay preserves rank-order fp32 addition,
  bf16 branch rounding, residual addition and final bf16 rounding. Chained steps
  consume the MTP normalized output; the next absorb rolls back all chained rows.
- Wired MTP into the existing engine DraftReq/P7 verification contract for CLI
  and serving. CLI commits publish accepted target hidden rows for re-absorption.
  Copy proposals still absorb committed rows into MTP and undergo target verification.
  Both proposal sources emit only target draws.
- Added CPU selection/error tests, synthetic TP weight fixtures, launch-ABI recorder
  tests, rollback/chain/copy fakes and drafted-versus-serial tests with intentional
  mismatches, EOS and output limits. Added 36 NumPy reference confidence fixtures;
  only `_probability` was extracted by AST, without importing Torch or CUDA.
- Added `--drafter mtp` to capture tooling, a capture inventory and a future GPU
  qualification script: `tools/zig/glm_mtp_capture.md`, `tools/zig/e2e/mtp_check.sh`.

## Commands and results

From the worktree root:
```sh
export ZIG_GLOBAL_CACHE_DIR=$PWD/.zig-global-cache
nice -n 10 zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=fast -j6 native install
nice -n 10 zig build test -j6 --summary all
bash -n tools/zig/e2e/serve2.sh
python3 -m py_compile tools/zig/e2e/check_displaykv.py
git diff --check
```

The build/test commands were rerun after correcting compiler diagnostics. Final native build:
**passed (exit 0)**; existing nvcc unused-variable warnings only. Final CPU test summary: **53/53 steps succeeded; 141/141 tests passed**;
remaining test artifacts were cached from preceding successful suite runs. The preceding complete
changed-module pass reported 274/274 tests passed. Expected BadMaster-test warning output does not
change the successful build summary/exit status. Syntax and diff checks passed.
Logs are `.validation/build.log` and `.validation/tests.log` (ignored). Initial build logs were
redirected to `/tmp/displaykv-{build,tests}.log`; they were moved into this worktree and later
runs redirected directly here. No task scratch files remain at those `/tmp` paths.
The sandbox failed to initialize its loopback interface, so approved local commands ran outside
that failing sandbox; their scope remained local reads, this worktree, and its git metadata.

CPU tests cover display numeric/backend boundaries, fewest-plane rounding, connected-card refusal,
every DRM driver-step failure and full unwind, short/nonadjacent device spans, zero ordinary bytes,
reserve/meminfo validation, display credit and integer extent alignment, pool allocation/release
publication, and optional versus known-zero health counts. The GLM CUDA test root is now explicitly
included in `zig build test` so these tests run.

## Exact GPU checks — prepared, NOT run

Only a GPU-authorized operator should run these. Each command holds the shared lock through
startup, requests and cleanup. Use headless Sparks with card0 available, the captured AOT set,
the checkpoint already installed on both nodes, and this worktree's built artifacts.

```sh
WT=/home/mia/newMod/GLM-5.3-Flash-EXL3-TensorFold/work/zig-glm-cx-displaykv
for mib in 0 1792; do
  flock /tmp/claude-1000/gpu.lock env GLM_GPU_LOCKED=1 GLM_POOL_CHECK=1 LOAD_TIMEOUT=180 \
    WT="$WT" TF_GLM_CACHE_GIB=12.5 TENSORFOLD_MEMORY_RESERVE_GIB=14.5 \
    TF_GLM_DISPLAY_KV_MIB="$mib" TF_GLM_DISPLAY_KV_BACKEND=drm \
    bash "$WT/tools/zig/e2e/serve2.sh" "pkg-j-drm-$mib"
done
# Repeat on kindling nodes with the installed client + daemon:
flock /tmp/claude-1000/gpu.lock env GLM_GPU_LOCKED=1 GLM_POOL_CHECK=1 LOAD_TIMEOUT=180 \
  WT="$WT" TF_GLM_CACHE_GIB=12.5 TENSORFOLD_MEMORY_RESERVE_GIB=14.5 \
  TF_GLM_DISPLAY_KV_MIB=1792 TF_GLM_DISPLAY_KV_BACKEND=dispram \
  bash "$WT/tools/zig/e2e/serve2.sh" pkg-j-dispram
# Raised reserve, same cap/display: compare logged capacity to baseline.
flock /tmp/claude-1000/gpu.lock env GLM_GPU_LOCKED=1 GLM_POOL_CHECK=1 LOAD_TIMEOUT=180 \
  WT="$WT" TF_GLM_CACHE_GIB=12.5 TENSORFOLD_MEMORY_RESERVE_GIB=18.5 \
  TF_GLM_DISPLAY_KV_MIB=0 \
  bash "$WT/tools/zig/e2e/serve2.sh" pkg-j-reserve
```

The HTTP checker requires positive extent-aligned totals, free <= total, a drop in free tokens while
requests run, solo/concurrent equality, and complete release afterward. Compare each run's
`/tmp/claude-1000/zig-serve/<label>/pool-check.json`: display-enabled pools should grow beyond the
ordinary cap; raising reserve should reduce capacity when the available-memory bound is active.
Observe host `MemAvailable` on both ranks through startup/prefill: adding display credit must not
consume the credited amount of ordinary RAM. Also test a connected output (startup must fail with
`DisplayConnected`), missing DRM/client/socket (must fail), and unequal rank settings (must fail).

For the known exact **token** digests (HTTP text equality alone is insufficient), prepare an
in-worktree copy of the existing runner with its hard-coded worktree fixed:

```sh
sed "s#^WT=.*#WT=$WT#" "$WT/tools/zig/e2e/run2.sh" > "$WT/.validation/run2-display.sh"
for mib in 0 1792; do
  for mode in serial drafts; do
    args="--parallel 8 --context 131072 --tokens-file /zig/prompts/short.ids,/zig/prompts/long.ids --max-tokens 200"
    if [ "$mode" = drafts ]; then
      args="$args --drafts --drafter /hf/hub/models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a"
    fi
    label="pkg-j-digest-$mib-$mode"
    flock /tmp/claude-1000/gpu.lock env TIMEOUT=240 \
      ENVSTR="--device /dev/dri/card0 -e TF_GLM_DISPLAY_KV_MIB=$mib -e TF_GLM_DISPLAY_KV_BACKEND=drm -e TF_GLM_CACHE_GIB=12.5 -e TENSORFOLD_MEMORY_RESERVE_GIB=14.5" \
      bash "$WT/.validation/run2-display.sh" "$label" "$args"
    rg TOKEN_SHA "/tmp/claude-1000/zig-e2e/$label/rank0.log"
  done
done
```

Require rank0 exit 0 and both existing digests **bfe49e793813** (short) and **33325d640a7b**
(long) for every run, with display off/on and drafts off/on. A new pool bucket may fail before
inference until the corresponding AOT variants are captured; record that as an unqualified shape,
not token parity. Repeat the server check with `DRAFTER_REL=models--incoai--GLM-5.3-Flash-DFlash2/snapshots/bf582e4eacc1810f76656d1811693ff6c6737d2a`,
and cancellation/client disconnect tests from package A, to verify released pool extents with drafts.

zig build test -j6 --summary all
```

Native/install succeeded (exit 0), including the final dense-MTP path. Compiler
output contains existing NVCC unused-variable warnings; no compilation errors.
Final native output is retained locally in ignored `mtp-native.log`.

The first full test run passed 714/714 tests, 51/51 steps. After adding the
weight-plan and CLI hidden-publication regressions, the final run passed 51/51
steps: 260/260 rerun tests, remaining test runs cached. Exit 0; the final output
is in ignored `mtp-tests.log`. An intermediate synthetic Catalog fixture failed
compilation because its arena initializer was missing; fixed before the final
passing run. The printed `failed command` beneath the expected bad-master warning
also appears in the passing Zig summary; it is not a failed test (exit 0 and all
steps succeeded).

Additional CPU commands:

```sh
python3 tools/zig/glm_mtp_cpu_fixtures.py
python3 -m py_compile tools/zig/glm_mtp_cpu_fixtures.py tools/zig/glm_capture.py
bash -n tools/zig/e2e/mtp_check.sh
python3 tools/zig/glm_capture.py --help
python3 tools/zig/glm_capture.py --out unused --drafter mtp --parallel 2
# Expected exit 2: "MTP capture requires --parallel 1", before external operations.
git diff --check
```

All expected results passed. Fixture generator produced 36 cases using locally
installed NumPy 2.5.3, from `../eng-v111/src/tensorfold/families/glm5_next/cuda/decode.py`.
Fixture SHA256: `abc07625187f2fc9e06fd6803a2914749f920be140dd4d8dfa175f969658299c`.
Confidence comparisons use absolute tolerance 1e-15; this does not claim bit-exact
NumPy/libm exponentiation.

## GPU checks left (not executed)

First capture/install real MTP and parallel=1 target variants. New requirements
are `_residual_add`, strided `_rmsnorm` SUMS=0 with bf16 XS aliases, COPIES=1
embedding, K=8192 group sums, dense merge NCH=1..5, and prompt final-tail shapes.
The full inventory and capture/manifest instructions are in
`tools/zig/glm_mtp_capture.md`. Preserve existing target/segmented variants; a
short capture is not exhaustive. Existing prompt `_select_split` OUT `*i64`
coverage limits also remain. Never enable `TF_GLM_TRI_LENIENT`.

Authorized GPU qualifier, after installing those variants and the existing token
files, should run exactly:

```sh
flock /tmp/claude-1000/gpu.lock env GLM_GPU_LOCKED=1 \
  bash tools/zig/e2e/mtp_check.sh r107-mtp
flock /tmp/claude-1000/gpu.lock env GLM_GPU_LOCKED=1 MTP_CHECK_COPY=1 \
  bash tools/zig/e2e/mtp_check.sh r107-mtp-copy
```

Each checks both ranks, MTP-on versus drafts-off, and the established short/long
TOKEN_SHA values `bfe49e793813` / `33325d640a7b` (parallel=1, context=131072).
The script uses this worktree's binaries, a temporary adapted runner and the
existing owned-container lifecycle. Its prerequisite is actual parallel=1 AOT
coverage; the old parallel=8-only set is insufficient.

Full forward arithmetic needs byte comparisons against Python at the two input
norm halves, EH output, both residuals, shared-head norm and Q4 logits, plus
MTP cursor/cache comparisons after rejection and re-absorption. Include prompt
chunks/tails, zero embedding, positions 2050/2051, four-token pool boundaries,
partial acceptance, EOS, sequential requests and both draft quantizers. Also
qualify TF_GLM_MTP=1 beside DFlash2 and drafts-off, and memory usage at startup.

## Remaining limitations

The CPU implementation and verification contract are complete; actual GPU
arithmetic/token parity is unqualified until those checks pass. MTP uses eager
launches and synchronous host staging; graph capture/tuning remains performance
work. Existing host MSE clipping versus ATen and sorted top-k tie membership
qualification limits apply. Sparse decode reuses Python's documented
per-segment DSA equivalence, which still needs the GPU byte check here. Copy
settings retain the existing Settings defaults (8-token match, 15 maximum,
16-token reply match); TF_GLM_COPY_DRAFTS=0 is honored. Broader recipe copy-policy
knob mapping, prompt cache/snapshots, TP=3/4 and other packages remain outside
this change.