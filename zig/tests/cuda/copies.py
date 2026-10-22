"""Checks (or rewrites) zig/kernels/cuda copies of the Python engine's CUDA device code: same lines, same bits."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# Upstream 1.0 removed the Python runtime. Supply its qualified source tree explicitly.
SOURCE_ROOT = ROOT

GDN_FOOTER = (
    "// The instantiations the Python wrappers launch (gdn_replay_cuda, dispatch_tree for bf16 keys).\n"
    "#define TF_REPLAY(QK) template __global__ void tf_gdn::replay_kernel<QK, 8, 4>(const long long*, int, const int*, int, \\\n"
    "    const int*, int, float*, int, int, int);\n"
    "TF_REPLAY(__nv_bfloat16)\n"
    "TF_REPLAY(float)\n"
    "#define TF_TREE(S, R, W, C) template __global__ void tf_gdn::tree_kernel<__nv_bfloat16, S, R, W, C>( \\\n"
    "    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, const float*, \\\n"
    "    const long long*, const int*, const int*, int, __nv_bfloat16*, int, int, int, tf_gdn::Pending<__nv_bfloat16>, \\\n"
    "    float*, const long long*, bool);\n"
    "TF_TREE(0, 8, 4, true)\n"
    "TF_TREE(1, 8, 4, false)\n"
    "TF_TREE(2, 8, 2, false)\n"
    "TF_TREE(2, 4, 4, false)\n"
    "TF_TREE(4, 2, 4, false)\n"
    "TF_TREE(8, 2, 4, false)\n"
    "TF_TREE(16, 2, 2, false)\n"
    "TF_TREE(32, 2, 1, false)\n"
)

FN_GDN_FOOTER = (
    "// The instantiations gdn_chain_cuda and gdn_replay_cuda launch: one GPU (16, 48) and a TP=2 rank (8, 24).\n"
    "#define TF_CHAIN(NK, NV, AHEAD) template __global__ void tf_fn_gdn::chain_kernel<NK, NV, AHEAD>( \\\n"
    "    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, const float*, \\\n"
    "    const __nv_bfloat16*, float, int, __nv_bfloat16*, float*, float*, float*, __nv_bfloat16*, float*, float*);\n"
    "TF_CHAIN(16, 48, true)\n"
    "TF_CHAIN(16, 48, false)\n"
    "TF_CHAIN(8, 24, true)\n"
    "TF_CHAIN(8, 24, false)\n"
    "#define TF_REPLAY(NK, NV) template __global__ void tf_fn_gdn::replay_kernel<NK, NV>(const float*, const float*, \\\n"
    "    const __nv_bfloat16*, const float*, const float*, int, float*);\n"
    "TF_REPLAY(16, 48)\n"
    "TF_REPLAY(8, 24)\n"
)

FN_GDN_IO_FOOTER = (
    "// The instantiations gdn_front_cuda and gdn_back_cuda launch: one GPU (16, 48) and a TP=2 rank (8, 24).\n"
    "#define TF_FRONT(NK, NV) template __global__ void tf_fn_gdn_io::front_kernel<NK, NV>(const __nv_bfloat16*, \\\n"
    "    const long long*, const int*, const int*, const __nv_bfloat16*, const float*, const float*, float*, float*, \\\n"
    "    __nv_bfloat16*, float*, float*);\n"
    "#define TF_BACK(NK, NV) template __global__ void tf_fn_gdn_io::back_kernel<NK, NV>(const __nv_bfloat16*, \\\n"
    "    const __nv_bfloat16*, const __nv_bfloat16*, float, __nv_bfloat16*, float*);\n"
    "TF_FRONT(16, 48)\n"
    "TF_FRONT(8, 24)\n"
    "TF_BACK(16, 48)\n"
    "TF_BACK(8, 24)\n"
)

FN_GDN_PREFILL_FOOTER = (
    "// The instantiations gdn_prefill_cuda launches (32 steps a stage): fp32 keys (Flash Next's front) and bf16,\n"
    "// 128 value rows a block when the value heads fill the SMs, else 64.\n"
    "#define TF_PREFILL(QK, ROWS) template __global__ void tf_fn_gdn_prefill::chain_kernel<QK, ROWS, 32>(const QK*, \\\n"
    "    const QK*, const __nv_bfloat16*, const float*, const float*, const float*, float*, __nv_bfloat16*, int, int, int);\n"
    "TF_PREFILL(float, 128)\n"
    "TF_PREFILL(float, 64)\n"
    "TF_PREFILL(__nv_bfloat16, 128)\n"
    "TF_PREFILL(__nv_bfloat16, 64)\n"
)

FN_GDN_TREE_FOOTER = (
    "// fp32 keys (Flash Next's front): every tree_kernel dispatch_tree<float> picks, and replay_kernel<float>.\n"
    "#define TF_TREE(S, R, W, C) template __global__ void tf_fn_gdn_tree::tree_kernel<float, S, R, W, C>( \\\n"
    "    const float*, const float*, const __nv_bfloat16*, const float*, const float*, const float*, \\\n"
    "    const long long*, const int*, const int*, int, __nv_bfloat16*, int, int, int, tf_fn_gdn_tree::Pending<float>, \\\n"
    "    float*, const long long*, bool);\n"
    "TF_TREE(0, 8, 4, true)\n"
    "TF_TREE(1, 8, 4, false)\n"
    "TF_TREE(2, 8, 2, false)\n"
    "TF_TREE(2, 4, 4, false)\n"
    "TF_TREE(4, 2, 4, false)\n"
    "TF_TREE(8, 2, 4, false)\n"
    "TF_TREE(16, 2, 2, false)\n"
    "TF_TREE(32, 2, 1, false)\n"
    "template __global__ void tf_fn_gdn_tree::replay_kernel<float, 8, 4>(const long long*, int, const int*, int,\n"
    "    const int*, int, float*, int, int, int);\n"
)

FN_NVFP4_FOOTER = (
    "// The instantiations nvfp4_experts_cuda launches: gate/up SwiGLU (M 2, epilogue 2), down fp32 (0) and bf16 (3).\n"
    "#define TF_NVFP4(M, EPI) template __global__ void tf_fn_nvfp4_experts::nvfp4_expert_kernel<M, EPI, 4>( \\\n"
    "    const __nv_bfloat16*, int, int, const uint4*, const float*, int, int, const int*, const int*, const int*, \\\n"
    "    void*, int, float, int);\n"
    "TF_NVFP4(2, 2)\n"
    "TF_NVFP4(1, 0)\n"
    "TF_NVFP4(1, 3)\n"
)

FN_QMM_FOOTER = (
    "// Groups of 32 (the MTP draft head's 4-bit copy of lm_head rows): row tiles 16, 32, 64, bf16 or fp32 out, one K\n"
    "// slice (no cluster); the launches qmm_cuda's dispatch<32, F32, false> makes.\n"
    "#define TF_QMM(BM, F32) template __global__ void tf_fn_qmm::qmm_kernel<32, BM, 64, 1, 4, 4, F32, false, false>( \\\n"
    "    const __nv_bfloat16*, const float*, const uint32_t*, const __nv_bfloat16*, const __nv_bfloat16*, void*, float*, \\\n"
    "    int, int, int, int, int, int, int);\n"
    "TF_QMM(16, false)\n"
    "TF_QMM(32, false)\n"
    "TF_QMM(64, false)\n"
    "TF_QMM(16, true)\n"
    "TF_QMM(32, true)\n"
    "TF_QMM(64, true)\n"
)

FN_QMM_PREFILL_FOOTER = (
    "// Groups of 32 (the MTP draft head's prompt rows), tile 0: 128x128 on 2x4 warps, 3 stages; bf16 or fp32 out.\n"
    "#define TF_QMM_PREFILL(F32) template __global__ void tf_fn_qmm_prefill::prefill_kernel<32, 128, 128, 2, 4, 3, \\\n"
    "    F32>(const __nv_bfloat16*, const uint32_t*, const __nv_bfloat16*, const __nv_bfloat16*, void*, int, int, int, \\\n"
    "    int, int, int);\n"
    "TF_QMM_PREFILL(false)\n"
    "TF_QMM_PREFILL(true)\n"
)

FN_QMMF_FOOTER = (
    "// Block FP8 (FP8G, an fp32 scale per 64 inputs and column: Fp8BlockLinear) on sm_121: row tiles 16, 32, 64 with\n"
    "// one K slice (no cluster) or 2-8 slices in a cluster, and the fused prompt form (bm 0); bf16 or fp32 out. The\n"
    "// reduce never runs: split_k stays <= 8 and sm_90+ adds those slices in the cluster.\n"
    "#define TF_QMMF(BM, F32, CLUSTER, FUSE) template __global__ void tf_fn_qmmf::qmmf_kernel<3, BM, 64, 1, 4, 4, F32, \\\n"
    "    CLUSTER, FUSE>(const __nv_bfloat16*, const unsigned char*, const uint8_t*, float, void*, float*, int, int, int, \\\n"
    "    int, int, int, int);\n"
    "TF_QMMF(16, false, false, false)\n"
    "TF_QMMF(32, false, false, false)\n"
    "TF_QMMF(64, false, false, false)\n"
    "TF_QMMF(16, true, false, false)\n"
    "TF_QMMF(32, true, false, false)\n"
    "TF_QMMF(64, true, false, false)\n"
    "TF_QMMF(16, false, true, false)\n"
    "TF_QMMF(32, false, true, false)\n"
    "TF_QMMF(64, false, true, false)\n"
    "TF_QMMF(16, true, true, false)\n"
    "TF_QMMF(32, true, true, false)\n"
    "TF_QMMF(64, true, true, false)\n"
    "TF_QMMF(64, false, false, true)\n"
    "TF_QMMF(64, true, false, true)\n"
)

# name: source, first and last line taken, lines dropped (ATen includes), (namespace line, new name), instantiations
COPIES = {
    "gdn.cu": ("src/tensorfold/cuda/kernels/gdn.cu", 1, 319, (3, 4), (10, "tf_gdn"), GDN_FOOTER),
    "qmm_frag.cuh": ("src/tensorfold/cuda/kernels/qmm_frag.cuh", 1, 118, (), None, ""),
    "experts.cuh": ("src/tensorfold/cuda/experts.cuh", 1, 105, (), None, ""),
    "qmm_group.cu": (
        "src/tensorfold/cuda/kernels/qmm_group.cu", 1, 348, (3, 5), (14, "tf_qmm_group"),
        "// The instantiations Nemotron's windows launch on sm_121 (tile 2: rows <= 16, bf16 out).\n"
        "template __global__ void tf_qmm_group::group_kernel<64, 16, 64, 1, 4, 8, false, false, false, false>(\n"
        "    const __nv_bfloat16*, const float*, const __grid_constant__ tf_qmm_group::Parts, int, int, int, int, int);\n",
    ),
    "qmm_prefill.cu": (
        "src/tensorfold/cuda/kernels/qmm_prefill.cu", 1, 151, (3, 5), (11, "tf_qmm_prefill"),
        "// The instantiation prefill_matmul launches (tile 0: 128x128 on 2x4 warps, 3 stages), bf16 out.\n"
        "template __global__ void tf_qmm_prefill::prefill_kernel<64, 128, 128, 2, 4, 3, false>(\n"
        "    const __nv_bfloat16*, const uint32_t*, const __nv_bfloat16*, const __nv_bfloat16*, void*, int, int, int,\n"
        "    int, int, int);\n",
    ),
    "experts.cu": (
        "src/tensorfold/cuda/experts.cu", 1, 286, (3, 4, 7), (11, "tf_experts"),
        "// Decode form for groups of 64: up with relu^2 (epilogue 1), down to fp32 (epilogue 0).\n"
        "#define TF_EXPERT(EPI) template __global__ void tf_experts::expert_kernel<64, 1, EPI, 2, 4>(const __nv_bfloat16*, \\\n"
        "    int, int, const uint4*, int, int, const int*, const int*, const int*, void*, int, float);\n"
        "TF_EXPERT(1)\n"
        "TF_EXPERT(0)\n",
    ),
    "experts_prefill.cu": (
        "src/tensorfold/cuda/experts_prefill.cu", 1, 153, (3, 4, 7), (11, "tf_experts_prefill"),
        "// Prefill form for groups of 64: up with relu^2 (epilogue 1), down to bf16 (epilogue 3).\n"
        "#define TF_PREFILL(EPI) template __global__ void tf_experts_prefill::prefill_kernel<64, 1, EPI, 2, 2, 4>( \\\n"
        "    const __nv_bfloat16*, int, int, const uint4*, int, int, const int*, const int*, const int*, void*, int, float);\n"
        "TF_PREFILL(1)\n"
        "TF_PREFILL(3)\n",
    ),
    "experts_pack.cu": (
        "src/tensorfold/cuda/experts_pack.cu", 1, 42, (3, 4, 6), (8, "tf_experts_pack"),
        "// Groups of 64 inputs.\n"
        "template __global__ void tf_experts_pack::pack_kernel<2>(const uint32_t*, const uint16_t*, const uint16_t*,\n"
        "    uint32_t*, int, int, int, int);\n",
    ),
    "prefill_attention.cu": (
        "src/tensorfold/cuda/kernels/prefill_attention.cu", 1, 250, (3, 6), (11, "tf_prefill_attention"),
        "// Head dim 128, eight warps, eight query heads a block, eight staging slots (Nemotron's 16 heads a KV head).\n"
        "template __global__ void tf_prefill_attention::pattn_kernel<128, 8, 8, 8>(const __nv_bfloat16*,\n"
        "    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, int, float);\n",
    ),
    "scan_rows.cu": ("src/tensorfold/families/nemotron_h/cuda/scan_rows.cu", 1, 90, (3, 4), (8, "tf_scan_rows"), ""),
    # Flash Next (qwen4_exp) on the NVFP4 checkpoint: the extensions build_kernels loads and the engine launches.
    "fn_gdn.cu": ("src/tensorfold/families/qwen4_exp/cuda/gdn.cu", 1, 213, (10, 11), (15, "tf_fn_gdn"), FN_GDN_FOOTER),
    "fn_gdn_io.cu": (
        "src/tensorfold/families/qwen4_exp/cuda/gdn_io.cu", 1, 103, (3, 4), (8, "tf_fn_gdn_io"), FN_GDN_IO_FOOTER),
    "fn_gdn_prefill.cu": (
        "src/tensorfold/cuda/kernels/gdn_prefill.cu", 1, 144, (3, 4), (9, "tf_fn_gdn_prefill"), FN_GDN_PREFILL_FOOTER),
    "fn_gdn_tree.cu": ("src/tensorfold/cuda/kernels/gdn.cu", 1, 319, (3, 4), (10, "tf_fn_gdn_tree"), FN_GDN_TREE_FOOTER),
    "fn_nvfp4_experts.cu": (
        "src/tensorfold/cuda/nvfp4/experts.cu", 1, 148, (4, 5, 9), (13, "tf_fn_nvfp4_experts"), FN_NVFP4_FOOTER),
    "fn_qmm.cu": ("src/tensorfold/cuda/kernels/qmm.cu", 1, 230, (3, 5), (12, "tf_fn_qmm"), FN_QMM_FOOTER),
    "fn_qmm_prefill.cu": (
        "src/tensorfold/cuda/kernels/qmm_prefill.cu", 1, 151, (3, 5), (11, "tf_fn_qmm_prefill"), FN_QMM_PREFILL_FOOTER),
    # INT4-AutoRound checkpoint: its 128x128-block FP8 dense linears (Fp8BlockLinear, qmmf FP8G)
    "fn_qmmf.cu": ("src/tensorfold/cuda/nvfp4/qmmf.cu", 1, 298, (4, 6), (14, "tf_fn_qmmf"), FN_QMMF_FOOTER),
}

# name: {line: (source text, copy text)}: includes whose path differs from the copy's directory (same header bytes)
EDITS = {
    "fn_nvfp4_experts.cu": {11: ('#include "../experts.cuh"\n', '#include "experts.cuh"\n')},
    "fn_qmmf.cu": {12: ('#include "../kernels/qmm_frag.cuh"\n', '#include "qmm_frag.cuh"\n')},
}


def render(name: str) -> tuple[str, str]:
    """The copy's expected text and the sha256 of the exact source lines it came from."""

    src, first, last, drop, ns, footer = COPIES[name]
    lines = (SOURCE_ROOT / src).read_text().splitlines(keepends=True)
    if last > len(lines):
        raise SystemExit(f"{src} has {len(lines)} lines, fewer than the copy's {last}; update copies.py")
    taken = lines[first - 1:last]
    digest = hashlib.sha256("".join(taken).encode()).hexdigest()
    note = ", comments and ATen includes dropped" if drop else ", comments dropped"
    out = [f"// Device code of {src} (lines {first}-{last}{note}), checked by zig/tests/cuda/copies.py.\n"]
    for number, line in enumerate(taken, start=first):
        if line.lstrip().startswith("//"):
            continue                # comments stay in the source; the copy keeps code only (same SASS)
        if number in drop:
            if "ATen" not in line and "c10" not in line and "torch/" not in line:
                raise SystemExit(f"{src}:{number} is no longer an ATen include; update copies.py")
            continue
        if number in EDITS.get(name, {}):
            was, now = EDITS[name][number]
            if line != was:
                raise SystemExit(f"{src}:{number} is no longer {was.strip()!r}; update copies.py")
            line = now
        if ns is not None and number == ns[0]:
            if line != "namespace {\n":
                raise SystemExit(f"{src}:{number} is no longer the anonymous namespace; update copies.py")
            line = f"namespace {ns[1]} {{\n"
        out.append(line)
    if ns is not None:
        out.append(f"}} // namespace {ns[1]}\n\n")
    out.append(footer)
    return "".join(out), digest


def main() -> int:
    global SOURCE_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT, help="Python reference checkout containing src/tensorfold")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    SOURCE_ROOT = args.source_root.resolve()
    write = args.write
    bad = 0
    for name in COPIES:
        text, digest = render(name)
        path = ROOT / "zig/kernels/cuda" / name
        if write:
            path.write_text(text)
            print(f"wrote {path.relative_to(ROOT)} from source lines sha256 {digest}")
        elif not path.exists() or path.read_text() != text:
            print(f"DRIFT {path.relative_to(ROOT)}: differs from its source; rerun with --write and re-prove its bits")
            bad += 1
        else:
            print(f"ok {name} (source lines sha256 {digest})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
