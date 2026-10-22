#!/bin/bash
# Flash Next's two-rank transport on a pair of Sparks: builds tf-flashnext-comm (zig/tests/cuda/flashnext/comm_test.zig)
# in the PyTorch image (no GPU, no network), copies it to both boxes, and runs rank 0 on HOST0 and rank 1 on HOST1 in
# that image with the recipe's NCCL settings (scripts/nodes.sh: the link's netdev, both PCIe rails of the cabled CX7
# port, the RoCE v2 GID index, 4 channels), each under its box's GPU lock.
# Usage: TF_FN=<work dir: src/ out/ cache/> TF_ZIG=<zig binary> bash -u comm.sh RUN HOST0 HOST1 PEER [PORT] [ITERS] [WARMUP]
#   HOST0, HOST1: ssh targets (user@address) or "local" for this machine; PEER: HOST1's address on the CX7 link (the
#   link and MASTER, HOST0's link address, are found from the route to it, as nodes.sh's detect_link does).
# Env: TF_NODES_SH (default $TF_FN/../scripts/nodes.sh), TF_ZIG_IMAGE, TF_SKIP_BUILD=1, NCCL_RAILS=1 (one rail),
#   NCCL_CHANNELS (4), TF_NCCL_DEBUG (INFO), TF_LOCK0 / TF_LOCK1 (the GPU lock: ~/fnz-gpu.lock; on spark1 work/gpu.lock),
#   TF_NCCL_EXTRA (more docker -e settings for both ranks, e.g. "-e NCCL_PROTO=LL -e NCCL_BUFFSIZE=1048576").
# Writes out/RUN/{build.log,link.txt,rank0.log,rank1.log}. Containers fnz-comm-RUN-r0/-r1, removed after.
set -u
RUN="${1:?run id}"; HOST0="${2:?host of rank 0}"; HOST1="${3:?host of rank 1}"; PEER="${4:?rank 1 address on the link}"
PORT="${5:-29561}"; ITERS="${6:-1000}"; WARMUP="${7:-100}"
TF="${TF_FN:?set TF_FN}"
IMAGE="${TF_ZIG_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
NODES="${TF_NODES_SH:-${TF:?}/../scripts/nodes.sh}"
OUT="${TF:?}/out/${RUN:?}"
SRC="${TF:?}/src"
BIN="${OUT:?}/zig-out/bin/tf-flashnext-comm"
REMOTE_DIR='fnz-work/comm'
mkdir -p "${OUT:?}"
[[ -w "${OUT:?}" ]] || { echo "${OUT} is not writable (made by a container as root?): use another RUN"; exit 1; }

on() { # on <host> <command>: run on a box (local: here)
  local h=$1; shift
  if [[ "$h" == local ]]; then bash -c "$*"; else ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 "$h" "$*"; fi
}

if [[ "${TF_SKIP_BUILD:-0}" != 1 ]]; then
  ZIG="${TF_ZIG:?set TF_ZIG}"
  timeout 1800 docker run --rm --name "fnz-build-${RUN}" --network none --memory 16g -v "${TF:?}:${TF:?}" \
    -v "$(dirname "${ZIG:?}"):/zigtool:ro" -w "${SRC:?}" \
    -e ZIG_LOCAL_CACHE_DIR="${TF:?}/cache/zig-local" -e ZIG_GLOBAL_CACHE_DIR="${TF:?}/cache/zig-global" \
    --entrypoint /zigtool/zig "$IMAGE" build -Doptimize=safe --prefix "${OUT:?}/zig-out" -j16 flashnext-comm \
    > "${OUT:?}/build.log" 2>&1 || { tail -60 "${OUT:?}/build.log"; echo "build failed"; exit 1; }
fi
[[ -x "$BIN" ]] || { echo "no $BIN"; exit 1; }

# the link, with nodes.sh's own functions run on each box: "<address> <netdev> <rdma device> <gid index>", then rails
source "$NODES" || exit 1
read -r A0 DEV0 HCA0 GID0 <<<"$(on "$HOST0" "$(declare -f link_info); link_info $PEER")"
[[ -n "${A0:-}" && "${HCA0:--}" != - && "${GID0:--}" != - ]] || { echo "rank 0: no RoCE link toward $PEER ($A0 $DEV0 $HCA0 $GID0)"; exit 1; }
read -r A1 DEV1 HCA1 GID1 <<<"$(on "$HOST1" "$(declare -f link_info); link_info $A0")"
[[ -n "${A1:-}" && "${HCA1:--}" != - && "${GID1:--}" != - ]] || { echo "rank 1: no RoCE link toward $A0 ($A1 $DEV1 $HCA1 $GID1)"; exit 1; }
HCAS0=$(on "$HOST0" "$(declare -f rails); rails $DEV0 $GID0"); HCAS1=$(on "$HOST1" "$(declare -f rails); rails $DEV1 $GID1")
[[ "$(tr ',' '\n' <<<"$HCAS0" | wc -l)" == "$(tr ',' '\n' <<<"$HCAS1" | wc -l)" ]] || { HCAS0=$HCA0; HCAS1=$HCA1; }
[[ "${NCCL_RAILS:-2}" == 1 ]] && { HCAS0=$HCA0; HCAS1=$HCA1; }
ENV0="$(nccl_env "$DEV0" "$HCAS0" "$GID0") -e NCCL_DEBUG=${TF_NCCL_DEBUG:-INFO} -e NCCL_DEBUG_SUBSYS=INIT,NET,ENV ${TF_NCCL_EXTRA:-}"
ENV1="$(nccl_env "$DEV1" "$HCAS1" "$GID1") -e NCCL_DEBUG=${TF_NCCL_DEBUG:-INFO} -e NCCL_DEBUG_SUBSYS=INIT,NET,ENV ${TF_NCCL_EXTRA:-}"
{ echo "rank 0 $HOST0: $A0 $DEV0 $HCAS0 gid $GID0"; echo "rank 1 $HOST1: $A1 $DEV1 $HCAS1 gid $GID1"
  echo "env0 $ENV0"; echo "env1 $ENV1"; echo "master $A0:$PORT iters $ITERS warmup $WARMUP"; } | tee "${OUT:?}/link.txt"

# preflight on both: the image, no other compute app, memory
for h in "$HOST0" "$HOST1"; do
  on "$h" "docker image inspect '$IMAGE' >/dev/null 2>&1" || { echo "$h has no $IMAGE"; exit 1; }
  on "$h" "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; awk '/MemAvailable/ {print \"MemAvailable\", int(\$2/1048576), \"GiB\"}' /proc/meminfo" | sed "s|^|$h: |"
  avail=$(on "$h" "awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo")
  (( avail >= 20 )) || { echo "$h: only $avail GiB available"; exit 1; }
done

# the binary to both boxes
for h in "$HOST0" "$HOST1"; do
  if [[ "$h" == local ]]; then mkdir -p "$HOME/$REMOTE_DIR" && cp "$BIN" "$HOME/$REMOTE_DIR/"
  else on "$h" "mkdir -p ~/$REMOTE_DIR" && scp -q -o BatchMode=yes "$BIN" "$h:$REMOTE_DIR/"; fi || { echo "copy to $h failed"; exit 1; }
done

cleanup() {
  on "$HOST0" "docker rm -f fnz-comm-${RUN}-r0 >/dev/null 2>&1" ; on "$HOST1" "docker rm -f fnz-comm-${RUN}-r1 >/dev/null 2>&1"
}
trap cleanup EXIT

launch() { # launch <rank> <host> <env> <lock>
  local r=$1 h=$2 env=$3 lock=$4
  on "$h" "flock -w 900 $lock timeout 900 docker run --rm --name fnz-comm-${RUN}-r$r --gpus all --network host \
    --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --shm-size 1g $env \
    -v \$HOME/$REMOTE_DIR:/comm:ro --entrypoint /comm/tf-flashnext-comm '$IMAGE' $r $A0 $PORT $ITERS $WARMUP" \
    > "${OUT:?}/rank$r.log" 2>&1
}
launch 0 "$HOST0" "$ENV0" "${TF_LOCK0:-\$HOME/fnz-gpu.lock}" & P0=$!
launch 1 "$HOST1" "$ENV1" "${TF_LOCK1:-\$HOME/fnz-gpu.lock}" & P1=$!
wait $P0; rc0=$?; wait $P1; rc1=$?
# NCCL's INFO lines share the output, so a line of ours may start mid-line
grep -ohE '(RESULT|PASS|FAIL|INFO rank) .*' "${OUT:?}/rank0.log" "${OUT:?}/rank1.log"
grep -h -E 'NCCL INFO (NCCL version|NET/IB : Using|Using network|Channel 0./0 .*send|Connected all|NCCL_IB_GID)' "${OUT:?}/rank0.log" | head -12
echo "rank0 rc=$rc0 rank1 rc=$rc1"
(( rc0 == 0 && rc1 == 0 ))
