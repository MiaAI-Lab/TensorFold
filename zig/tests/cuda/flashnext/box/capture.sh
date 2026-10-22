#!/bin/bash
# The Python Flash Next engine's oracle run for the Zig port (TP=1): launches, dumps, digests, tokens, then the manifest
# and the packed kernel set. Usage: TF_FN=<work dir: TensorFold/ out/ aot/> TF_MODEL=<snapshot dir> \
#   flock <work>/gpu.lock bash -u capture.sh RUN [capture.py options]
set -u
RUN="${1:?run id}"
shift
MORE=("$@")
TF="${TF_FN:?set TF_FN}"
MODEL="${TF_MODEL:?set TF_MODEL}"
SRC="${TF:?}/TensorFold"
OUT="${TF:?}/out/${RUN:?}"
CACHE="${TF:?}/aot/${RUN:?}"
IMAGE="${TF_ZIG_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
NAME="fnz-capture-${RUN:?}"
MIN_KB=$(( ${TF_MIN_AVAILABLE_GIB:-8} * 1024 * 1024 ))
HF="${HF_HOME_HOST:-$HOME/.cache/huggingface}"
mkdir -p "${OUT:?}" "${CACHE:?}/triton" "${CACHE:?}/torch_ext" "${CACHE:?}/cuda_cache"

apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
if [ -n "$apps" ]; then echo "PREFLIGHT-FAIL: compute apps present: $apps"; exit 3; fi
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then echo "PREFLIGHT-FAIL: $NAME exists"; exit 3; fi
avail=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
if [ "$avail" -lt $(( 100 * 1024 * 1024 )) ]; then echo "PREFLIGHT-FAIL: MemAvailable ${avail} kB < 100 GiB"; exit 3; fi
free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
if [ "$free_gb" -lt 110 ]; then echo "PREFLIGHT-FAIL: ${free_gb} GB free on /"; exit 3; fi
{ git -C "$SRC" rev-parse HEAD; git -C "$SRC" status --short src; } > "${OUT:?}/tree.txt" 2>&1
case "$MODEL" in "$HF"/*) INNER="/hf/${MODEL#"$HF"/}";; *) echo "model must be under $HF"; exit 2;; esac

# TF_CAPTURE_ENV="NAME=value ...": extra engine settings for the container (e.g. TF_FLASHNEXT_YARN=4.0)
envs=()
for kv in ${TF_CAPTURE_ENV:-}; do envs+=(-e "$kv"); done
echo "${TF_CAPTURE_ENV:-}" > "${OUT:?}/capture_env.txt"

cleanup() { docker stop --time 20 "$NAME" >/dev/null 2>&1; docker rm -f "$NAME" >/dev/null 2>&1; }
trap 'cleanup; kill $WATCH 2>/dev/null; exit 143' TERM INT

date -u +"start %FT%TZ" > "${OUT:?}/mem.log"
docker run -d --name "$NAME" --user "$(id -u):$(id -g)" --gpus all --ipc host --network none --shm-size 16g --ulimit memlock=-1 \
  --ulimit stack=67108864 --read-only --tmpfs /tmp:size=8g \
  -v "$HF:/hf:ro" -e HF_HOME=/hf -v "${SRC:?}:/tensorfold:ro" -v "${OUT:?}:/out" -v "${CACHE:?}:/aot" \
  -e HOME=/tmp -e PYTHONPATH=/tensorfold/src -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
  "${envs[@]}" -e HF_HUB_OFFLINE=1 -e TENSORFOLD_NO_UPDATE_CHECK=1 -e TORCH_CUDA_ARCH_LIST=12.1 \
  -e TRITON_CACHE_DIR=/aot/triton -e TORCH_EXTENSIONS_DIR=/aot/torch_ext -e CUDA_CACHE_PATH=/aot/cuda_cache \
  -w /tensorfold --entrypoint python "$IMAGE" -B /tensorfold/zig/tests/cuda/flashnext/capture.py \
  --model "$INNER" --out /out --tools /tensorfold/tools/zig "${MORE[@]}" > /dev/null || { cleanup; exit 4; }

# the memory watchdog: GB10's unified memory freezes the machine when exhausted
( while docker ps -q --filter "name=^${NAME}$" | grep -q .; do
    kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    echo "$(date -u +%T) MemAvailable_kB=$kb" >> "${OUT:?}/mem.log"
    if [ "$kb" -lt "$MIN_KB" ]; then echo "WATCHDOG: stopping $NAME at $kb kB" >> "${OUT:?}/mem.log"; docker stop --time 5 "$NAME"; fi
    sleep 5
  done ) &
WATCH=$!
rc=$(timeout "${TF_TIMEOUT:-5400}" docker wait "$NAME" || echo 124)
docker logs "$NAME" > "${OUT:?}/capture.log" 2>&1
kill $WATCH 2>/dev/null
cleanup
echo "container rc $rc"
if [ "$rc" = 0 ]; then
  python3 -B "${SRC:?}/tools/zig/triton_aot_manifest.py" --cache "${CACHE:?}/triton" --launches "${OUT:?}/launches.json" \
    --mount /aot/triton --out "${OUT:?}/manifest.json" || rc=5
fi
if [ "$rc" = 0 ]; then
  python3 -B "${SRC:?}/tools/zig/aot_pack.py" --manifest "${OUT:?}/manifest.json" --cache "${CACHE:?}/triton" \
    --jit "${OUT:?}/jit.json" --out "${CACHE:?}" > "${OUT:?}/pack.log" 2>&1 || { rc=6; cat "${OUT:?}/pack.log"; }
fi
echo "rc $rc"
exit "$rc"
