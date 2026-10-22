#!/bin/sh
# Run a fixture generator in the Python reference image (CPU only):  run.sh <script.py> [args...]
# Mounts the reference engine tree (read-only) at /src and the fixture directory at /fix.
HERE=$(cd "$(dirname "$0")" && pwd)
FIX=$(cd "$HERE/.." && pwd)
SRC=${TF_ENGINE_SRC:-/home/mia/newMod/GLM-5.3-Flash-EXL3-TensorFold/work/eng-v111/src}
CKPT=${GLM_CKPT:-$HOME/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold/snapshots/078455ffe6472f9a52fbc1139f58b9db2881b25c}
script=$1; shift
exec docker run --rm -u "$(id -u):$(id -g)" --entrypoint python3 -e PYTHONPATH=/src -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 -e PYTHONFAULTHANDLER=1 \
  -v "$SRC":/src:ro -v "$HERE":/gen:ro -v "$FIX":/fix -v "$FIX/../../src/vision":/zsrc -v "$CKPT":/ckpt:ro \
  tensorfold-glm53:r107ref "/gen/$script" "$@"
