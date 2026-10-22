#!/bin/bash
# Flash Next's CUDA extension copies: builds the Zig engine with the PyTorch image's nvcc, proves each fn_* fatbin's
# SASS equals the Python extension's object, and checks every entry point flashnext/cuda_kernels.zig resolves exists.
# Usage: TF_FN=<work dir: src/ out/ aot/> TF_ZIG=<zig binary> bash -u build.sh RUN [EXT_DIR]
# EXT_DIR holds the Python extension builds (TORCH_EXTENSIONS_DIR); default ${TF_FN}/aot/w3-ext, built here when absent
# (no GPU: the image's torch reports capability 12.1, so build.py's arch_flags give -gencode=...compute_121,code=sm_121).
set -u
RUN="${1:?run id}"
TF="${TF_FN:?set TF_FN}"
ZIG="${TF_ZIG:?set TF_ZIG}"
EXT="${2:-${TF:?}/aot/w3-ext}"
IMAGE="${TF_ZIG_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
OUT="${TF:?}/out/${RUN:?}"
SRC="${TF:?}/src"
mkdir -p "${OUT:?}" "${EXT:?}" "${TF:?}/cache/zig-local" "${TF:?}/cache/zig-global"
rc=0
python3 -B "${SRC:?}/zig/tests/cuda/copies.py" || rc=1
if [ ! -f "${EXT:?}/tensorfold_nvfp4_v3/experts.cuda.o" ]; then
  timeout 3000 docker run --rm --name "fnz-ext-${RUN}" --network none --memory 16g -v "${TF:?}:${TF:?}" \
    -e PYTHONPATH="${SRC:?}/src" -e TORCH_EXTENSIONS_DIR="${EXT:?}" -e TORCH_CUDA_ARCH_LIST=12.1 -e MAX_JOBS=8 \
    -w /tmp --entrypoint python "$IMAGE" -B -c '
import torch
torch.cuda.get_device_capability = lambda *a, **k: (12, 1)
torch.cuda.get_device_name = lambda *a, **k: "GB10 (build only)"
from tensorfold.families.qwen4_exp.cuda import gdn, gdn_io
from tensorfold.cuda.kernels import gdn as shared_gdn, qmm
from tensorfold.cuda import experts
from tensorfold.cuda.nvfp4 import linear
for load in (gdn._ext, gdn_io._ext, shared_gdn._ext, qmm._ext, experts._ext, linear._ext):
    load()
print("python extensions built")' > "${OUT:?}/ext.log" 2>&1 || { rc=1; tail -30 "${OUT:?}/ext.log"; }
fi
timeout 1800 docker run --rm --name "fnz-build-${RUN}" --network none --memory 16g -v "${TF:?}:${TF:?}" \
  -v "$(dirname "${ZIG:?}"):/zigtool:ro" -w "${SRC:?}" \
  -e ZIG_LOCAL_CACHE_DIR="${TF:?}/cache/zig-local" -e ZIG_GLOBAL_CACHE_DIR="${TF:?}/cache/zig-global" \
  --entrypoint /zigtool/zig "$IMAGE" build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=safe \
  --prefix "${OUT:?}/zig-out" -j16 ${TF_BUILD_STEPS:-fatbins install native} > "${OUT:?}/build.log" 2>&1 || { rc=1; tail -60 "${OUT:?}/build.log"; }
timeout 600 docker run --rm --network none -v "${OUT:?}:/out" -v "${EXT:?}:/ext:ro" -v "${SRC:?}:/tensorfold:ro" \
  --entrypoint bash "$IMAGE" -c '
  set -u; bad=0
  for pair in fn_gdn:tensorfold_qwen4_exp_gdn/gdn fn_gdn_io:tensorfold_qwen4_exp_gdn_io/gdn_io \
              fn_gdn_prefill:tensorfold_gdn_v2/gdn_prefill fn_gdn_tree:tensorfold_gdn_v2/gdn \
              fn_nvfp4_experts:tensorfold_nvfp4_v3/experts fn_qmm:tensorfold_qmm_v5/qmm \
              fn_qmm_prefill:tensorfold_qmm_v5/qmm_prefill experts:tensorfold_experts_v7/experts; do
    zig=${pair%%:*}; py=${pair#*:}
    [ -f /out/zig-out/fatbin/$zig.fatbin ] || { echo "MISSING $zig"; bad=1; continue; }
    cuobjdump -sass /ext/$py.cuda.o > /out/sass-python-$zig.txt
    cuobjdump -sass /out/zig-out/fatbin/$zig.fatbin > /out/sass-zig-$zig.txt
    cuobjdump -symbols /out/zig-out/fatbin/$zig.fatbin > /out/symbols-$zig.txt
    echo "--- $zig"; python -B /tensorfold/zig/tests/cuda/nemotron/sass_compare.py /out/sass-python-$zig.txt /out/sass-zig-$zig.txt || bad=1
  done
  exit $bad' > "${OUT:?}/sass.log" 2>&1 || rc=1
cat "${OUT:?}/sass.log"
grep -h 'STO_ENTRY' "${OUT:?}"/symbols-*.txt | awk '{print $NF}' | sort -u > "${OUT:?}/entries.txt"
# every name cuda_kernels.zig resolves: its literals, and the tree_kernel names it formats (tree_variants)
K="${SRC:?}/zig/src/families/flashnext/cuda_kernels.zig"
{ grep -o '"_ZN[^"]*"' "$K" | tr -d '"' | grep -v '{'
  for v in 0:8:4:1 1:8:4:0 2:8:2:0 2:4:4:0 4:2:4:0 8:2:4:0 16:2:2:0 32:2:1:0; do
    IFS=: read -r s r w c <<< "$v"
    echo "_ZN14tf_fn_gdn_tree11tree_kernelIfLi${s}ELi${r}ELi${w}ELb${c}EEEvPKT_S3_PK13__nv_bfloat16PKfS8_S8_PKxPKiSC_iPS4_iiiNS_7PendingIS1_EEPfSA_b"
  done; } | sort -u > "${OUT:?}/resolved.txt"
missing=$(comm -23 "${OUT:?}/resolved.txt" "${OUT:?}/entries.txt")
if [ -n "$missing" ]; then echo "UNRESOLVED (named in cuda_kernels.zig, absent from the fatbins):"; echo "$missing"; rc=1; fi
echo "resolved $(wc -l < "${OUT:?}/resolved.txt") entry points; table ${OUT:?}/resolved.txt"
echo "build rc=$rc"
exit $rc
