# The Flash Next CUDA box

The capture pipeline that makes the Python Flash Next engine (`tensorfold.families.qwen4_exp.cuda`, TP=1) the Zig
engine's oracle, the same steps the Nemotron port takes (`zig/tests/cuda/nemotron/box`). One work dir (`TF_FN`) holds
the pieces: `TensorFold/` is this repository (mounted read-only at `/tensorfold`), `out/RUN/` takes each run's logs and
artifacts, `aot/RUN/` the Triton cache, the extension builds and the packed kernel set. Every GPU step runs under the
GPU lock (`flock $TF_FN/gpu.lock ...`); capture.sh refuses to start while a compute app holds the GPU, while
MemAvailable is under 100 GiB or `/` has under 110 GB free, and a watchdog stops its container (`fnz-capture-RUN`)
when MemAvailable falls under `TF_MIN_AVAILABLE_GIB` (default 8). The container runs as the calling user, with the
Hugging Face cache mounted read-only at `/hf`.

## 1. capture.sh: the oracle run (GPU, checkpoint)

`TF_FN=<work> TF_MODEL=<snapshot under ~/.cache/huggingface> flock <work>/gpu.lock bash -u capture.sh RUN [capture.py options]`

Runs `zig/tests/cuda/flashnext/capture.py` in `nvcr.io/nvidia/pytorch:26.07-py3` (`TF_ZIG_IMAGE`), the engine
built as `tensorfold serve` builds it (`cuda_engine`, serve's default context unless `--context N`), with
`tools/zig/triton_aot_manifest.py`'s Recorder installed and every extension function wrapped. Phases:
`startup-ext` (the extension loads), `startup-load` (weights), `startup` (admission, n-gram tables),
`startup-graphs` (decode graph captures, each logged in detail), `startup-warm` (prompt warm-up), `weights-digest`,
`teacher` (48 one-row eager steps from an empty state over the long prompt's first 48 tokens, the MTP head absorbing
each row), `prefill-code` / `prefill-long` (eager prompt chunks with dumps), then `serial-*` and `drafted-*` for
greedy and sampled (seed 1234, temperature 1, top-k 20, top-p 0.95) decoding of every prompt, twice, and `profile`.

Files in `out/RUN/`:

| file | holds |
| --- | --- |
| `prompts.json`, `prompts_text.json` | token ids and rendered text (the checkpoint's chat template, thinking off) of code, story, facts, long (~3k tokens: two prompt chunks), huge (~9k: five chunks, past the 8192-key decode bucket) |
| `launches.json` | the Recorder: every Triton specialization launched (source, signature, constexprs, attrs, metadata, cache files), launch counts per phase for Triton and extension calls, grids and call sites per kernel, and the detail log (each launch's arguments: dtype, shape, strides, alignment, scalar values) for startup-ext, startup-load, graph captures, startup-warm, teacher steps 0-1 and both prefill phases |
| `jit.json` | each JIT function's parameter names and its do-not-specialize lists |
| `weights.json` | sha256 and shape of every device tensor under `Weights`, by dotted dataclass/attribute name (the Zig loader must reproduce these bytes); `weights_host.json` the host-side tables |
| `teacher.json` | the 48 teacher tokens, each step's greedy token and logits sha256, the MTP head's next token and logits sha256 per step, the prefill phases' first tokens and last-stream digests |
| `teacher/stepNNN/` | `digests.json` (sha256, shape, dtype of every dumped tensor, every step) and raw `.bin` files for steps 0, 1, 47: per layer `LNN_h_in` (streams entering the layer), `LNN_hc_a` (the attention HC read-out), `LNN_gdn_mK` / `LNN_attn_mK` (the mixer branch, write-back mode K), `LNN_hc_m` (the MoE input), `LNN_router_{logits,pick,wts}`, `LNN_moe_y_mK`, `L02_ple_emb` / `L02_h_ple`, `final_streams`, `final_mixed`, `logits`; `mtp_*` the same for the MTP layer and `mtp_head_logits` |
| `prefill/{code,long}/` | the same names prefixed `cNN_` (prompt chunk): all in full for code, digests for all and full streams of layers 0, 2, 3, 47 in the first chunk for long |
| `draws.json` | every MTP draft draw: phase, position, token, temperature-1 probability, the draft logits' sha256 |
| `graphs.json` | each graph capture's key (rows, DeltaNet parity, context bucket) and replays per phase |
| `results.json` | engine info (capacity plan, window, rows, extensions loaded and their functions), every run's tokens, sha, timings, rounds, acceptance, `repeats_same`, `drafted==serial` per prompt and sampling |
| `profile/` | torch.profiler tables of a serial run, a drafted run and the long prompt's prefill: every CUDA kernel, torch's own ops included |
| `manifest.json` | step 2 |
| `kernels.md` | step 4 |

## 2. triton_aot_manifest.py: the manifest (host, no GPU)

capture.sh runs `tools/zig/triton_aot_manifest.py --cache aot/RUN/triton --launches out/RUN/launches.json --mount
/aot/triton --out out/RUN/manifest.json`: each launched specialization mapped to its cubin, metadata and ABI.

## 3. tools/zig/aot_pack.py: the kernel set (host, no GPU)

capture.sh then runs `tools/zig/aot_pack.py --manifest out/RUN/manifest.json --cache aot/RUN/triton --jit
out/RUN/jit.json --out aot/RUN`: `aot.json` plus `cubins/<hash>.cubin`, what `zig/src/cuda/aot.zig` loads
(`TENSORFOLD_CUDA_KERNELS=aot/RUN`). The family-neutral form of `zig/tests/cuda/nemotron/aot_pack.py` (same fields,
plus the Python function and source of each kernel; non-numeric constexprs named instead of refused).

## 4. kernels_md.py: the inventory (host)

`python3 -B zig/tests/cuda/flashnext/kernels_md.py --run out/RUN` writes `out/RUN/kernels.md`: every Triton
function and extension function by role (embed, n-gram/PLE, HC, GDN, attention, indexer/QSA, router, experts, shared
expert, dense bf16 matmul, norm, head, sampler, MTP, commit), with constexpr variants, grids, sites, launch counts per
phase group, extension argument variants, graph replays and the torch ops the profiler saw.

## 5. tools/zig/flashnext_aot.py: the kernel set without the checkpoint (no GPU)

The spec `zig/tests/cuda/flashnext/kernels.json` is made from a capture and checked in:

```
python3 -B tools/zig/flashnext_aot.py extract --launches out/RUN/launches.json --cache aot/RUN/triton --out spec-tp1.json
python  -B tools/zig/flashnext_aot.py derive --spec spec-tp1.json --out zig/tests/cuda/flashnext/kernels.json
python3 -B tools/zig/flashnext_aot.py cover --spec zig/tests/cuda/flashnext/kernels.json \
  --fixtures zig/src/families/flashnext/fixtures_cuda_triton.json --out zig/tests/cuda/flashnext/kernels.json
python  -B tools/zig/flashnext_aot.py build --spec zig/tests/cuda/flashnext/kernels.json --out <dir> --tp 1,2 \
  [--check out/RUN/manifest.json] [--jit out/RUN/jit.json]
```

Each entry is what Triton's JIT hands `compile` (function, signature, typed constexprs, divisibility attributes,
options) with `"tp"` = the ranks that launch it; `derive` adds the TP=2 shapes of PLAN.md's split (`tp1_hash` and
`rule` name the captured entry each comes from; the K splits, tiles and warps follow the Python wrappers' own rules
for the new shapes) and the TP=2-only `_moe_partial` and mode-3 `_hc_writeback`; it also expands every entry to each row bucket the
wrappers pick (`_b16mm` BM 16/128, `_fp4mm` and `_router` BM 16/32/64/128 with their tiles) and each Triton form of
the row-count int (`M`, `R`, `keep`, `total`, `NB`: value 1 as a constexpr, a multiple of 16, neither), and `_select`
to every width up to its register limit plus `_select_tiles`, so prompts of any length and every context bucket find
their variant. `cover` then checks every launch in W5's replay fixtures (`fixtures_cuda_triton.json`) with
`aot.zig`'s matching rule and adds what is still missing from a template of the same kernel (for example
`_moe_partial` with fp32 slots, the MTP head's buffers at TP=2); run it after every `derive`. `build` runs `triton.compile` for
sm_121 (no GPU, no checkpoint) and writes `aot.json` + `cubins/`; with `--check` every TP=1 entry's Triton hash and
cubin sha256 must equal the capture's. Run derive and build in the PyTorch image with this tree at `/tensorfold`
(`-v <tree>:/tensorfold:ro -e PYTHONPATH=/tensorfold/src`): Triton's line info puts the source path into each
cubin, so only the capture's path (`source_root` in the spec) reproduces its bytes, and `build` refuses another.
