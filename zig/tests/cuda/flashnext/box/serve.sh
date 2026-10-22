#!/bin/bash
# tensorfold-native serving Flash Next on Sparks, then the API checks: /v1/models, a greedy chat completion with
# thinking off, the same with "draft": false (equal text), the same streamed (equal text), a tool call. One rank
# (HOST1 "-") or two (rank 0 on HOST0 serving, rank 1 on HOST1 following over the CX7 link toward PEER).
# Usage: TF_FN=<work dir> bash -u serve.sh RUN BIN KSET HOST0 [HOST1 PEER]
#   BIN: a built tensorfold-native (box/build.sh's out/<build>/zig-out/native/bin/tensorfold-native); KSET: the kernel
#   set dir (aot.json + cubins/). HOST*: ssh targets. The checkpoint is the box's read-only volume glm53-hf-ablit.
#   A host may be "local" (this machine). Per rank R (0, 1): TF_LOCKR (the GPU lock, default ~/fnz-gpu.lock; spark1:
#   work/gpu.lock; held for the whole run, taken in the order spark1 < spark2 < spark3 < spark4), TF_HFR (the Hugging Face cache mounted read-only: a docker volume, default glm53-hf-ablit, or a path
#   such as $HOME/.cache/huggingface). TF_ENV: extra docker -e flags for both ranks (-e TF_FLASHNEXT_YARN=4.0).
# Env: TF_CLI (a built `tensorfold`: after the server, its `run` on the greedy prompt's ids, drafted and serial, must
#   give the server's token sha), TF_ZIG_IMAGE, TF_CONTEXT (32768), TF_PARALLEL (1), TF_PORT (8888), TF_MASTER_PORT (29551), TF_EXTRA (serve
#   flags), TF_HOLD_S (0: seconds to keep serving after the checks), TF_AFTER (a command run while serving), TF_MIN_AVAILABLE_GIB (8: the watchdog's floor).
# Writes out/RUN/{rank0.log,rank1.log,checks/*.json,checks.txt}. Containers fnz-serve-RUN-r0/-r1, removed after.
set -u
RUN="${1:?run id}"; BIN="${2:?tensorfold-native}"; KSET="${3:?kernel set dir}"; HOST0="${4:?host of rank 0}"
HOST1="${5:--}"; PEER="${6:-}"
TF="${TF_FN:?set TF_FN}"
IMAGE="${TF_ZIG_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
# TF_SNAP: another snapshot under the mounted cache (INT4-AutoRound: models--azampatti--...-INT4-AutoRound/snapshots/1464274...)
SNAP=${TF_SNAP:-/root/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47}
NAME=${TF_NAME:-Qwen3.8-Flash-Next}
PORT="${TF_PORT:-8888}"; MPORT="${TF_MASTER_PORT:-29551}"
OUT="${TF:?}/out/${RUN:?}"
NODES="${TF_NODES_SH:-${TF:?}/../scripts/nodes.sh}"
REMOTE="fnz-work/serve-${RUN}"
mkdir -p "${OUT:?}/checks"
[[ -x "$BIN" && -f "$KSET/aot.json" ]] || { echo "need $BIN and $KSET/aot.json"; exit 1; }

on() { if [[ "$1" == local ]]; then bash -c "${*:2}"; else ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 "$1" "${@:2}"; fi; }
put() { # put <host> <src> <dir under ~> [rsync flags]
  if [[ "$1" == local ]]; then mkdir -p "$HOME/$3" && rsync -a ${4:-} "$2" "$HOME/$3"; else rsync -a ${4:-} -e 'ssh -o BatchMode=yes' "$2" "$1:$3"; fi
}
get() { if [[ "$1" == local ]]; then cp "$HOME/$2" "$3"; else scp -q -o BatchMode=yes "$1:$2" "$3"; fi; }
HF=("${TF_HF0:-glm53-hf-ablit}" "${TF_HF1:-glm53-hf-ablit}")
LOCK=("${TF_LOCK0:-\$HOME/fnz-gpu.lock}" "${TF_LOCK1:-\$HOME/fnz-gpu.lock}")

TP=1; HOSTS=("$HOST0")
[[ "$HOST1" != - ]] && { TP=2; HOSTS+=("$HOST1"); }

# PLAN.md "Box locking": every box's GPU lock is held for the whole run (load, checks, stop), taken in the fixed
# order spark1 < spark2 < spark3 < spark4; a holder process keeps each lock until release_locks
order_of() { case "$1" in local) echo 1 ;; *10.0.22.2*) echo 2 ;; *10.0.0.3*) echo 3 ;; *10.0.41.4* | *10.0.0.4*) echo 4 ;; *) echo 9 ;; esac; }
LOCKPIDS=()
take_locks() {
  local i h
  for i in $(for j in "${!HOSTS[@]}"; do echo "$(order_of "${HOSTS[$j]}") $j"; done | sort -n | awk '{print $2}'); do
    h=${HOSTS[$i]}
    on "$h" "mkdir -p ~/$REMOTE && rm -f ~/$REMOTE/unlock" || exit 1
    on "$h" "flock ${LOCK[$i]} bash -c 'echo locked; while [ ! -f ~/$REMOTE/unlock ]; do sleep 1; done'" > "$OUT/lock$i.txt" 2>&1 &
    LOCKPIDS+=($!)
    echo "$h: waiting for its GPU lock (${LOCK[$i]})"
    until grep -q locked "$OUT/lock$i.txt" 2>/dev/null; do
      kill -0 "${LOCKPIDS[-1]}" 2>/dev/null || { echo "$h: the lock holder ended: $(cat "$OUT/lock$i.txt")"; exit 1; }
      sleep 2
    done
  done
}
release_locks() {
  for h in "${HOSTS[@]}"; do on "$h" "touch ~/$REMOTE/unlock" 2>/dev/null; done
  for p in "${LOCKPIDS[@]}"; do wait "$p" 2>/dev/null; done
  LOCKPIDS=()
}
trap release_locks EXIT
take_locks

for i in "${!HOSTS[@]}"; do
  h=${HOSTS[$i]}; hf=${HF[$i]}
  on "$h" "docker image inspect '$IMAGE' >/dev/null && { [[ '$hf' == /* ]] && [ -d '$hf/hub' ] || docker volume inspect '$hf' >/dev/null; }" || { echo "$h: no image or $hf"; exit 1; }
  # one rank of two holds about half the weights: TF_MIN_START_GIB (110 one rank, 80 a rank of two); TF_ALLOW_APPS=1
  # starts beside small compute apps that are not ours (spark3's services); a box running another fnz- job is
  # waited for (TF_WAIT_S, 1800 s)
  waited=0
  while true; do
    apps=$(on "$h" "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l")
    avail=$(on "$h" "awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo")
    ours=$(on "$h" "docker ps --format '{{.Names}}' | grep '^fnz-' | grep -vc '^fnz-nfs$'")  # fnz-nfs: spark4's NFS export
    (( (apps == 0 || ${TF_ALLOW_APPS:-0} == 1) && ours == 0 && avail >= ${TF_MIN_START_GIB:-$(( TP == 2 ? 80 : 110 ))} )) && break
    (( waited >= ${TF_WAIT_S:-1800} )) && { echo "$h is busy (apps $apps, fnz jobs $ours, MemAvailable $avail GiB)"; exit 1; }
    sleep 20; waited=$(( waited + 20 ))
  done
  echo "$h: compute apps $apps, MemAvailable $avail GiB (waited ${waited} s)"
  on "$h" "mkdir -p ~/$REMOTE/bin ~/$REMOTE/kernels" && put "$h" "$BIN" "$REMOTE/bin/" &&
    put "$h" "$KSET/" "$REMOTE/kernels/" --delete || { echo "copy to $h failed"; exit 1; }
done

ENV0=""; ENV1=""; TPFLAGS0=""; TPFLAGS1=""; DEVS=""
if (( TP == 2 )); then
  source "$NODES" || exit 1
  read -r A0 DEV0 HCA0 GID0 <<<"$(on "$HOST0" "$(declare -f link_info); link_info $PEER")"
  read -r A1 DEV1 HCA1 GID1 <<<"$(on "$HOST1" "$(declare -f link_info); link_info $A0")"
  HCAS0=$(on "$HOST0" "$(declare -f rails); rails $DEV0 $GID0"); HCAS1=$(on "$HOST1" "$(declare -f rails); rails $DEV1 $GID1")
  ENV0="$(nccl_env "$DEV0" "$HCAS0" "$GID0")"; ENV1="$(nccl_env "$DEV1" "$HCAS1" "$GID1")"
  echo "rank 0 $HOST0 $A0 $DEV0 $HCAS0 gid $GID0 | rank 1 $HOST1 $A1 $DEV1 $HCAS1 gid $GID1" | tee "$OUT/link.txt"
  TPFLAGS0="--tp 2 --rank 0 --master $A0 --master-port $MPORT"; TPFLAGS1="--tp 2 --rank 1 --master $A0 --master-port $MPORT"
  DEVS="--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1"
fi

# one session a box under its GPU lock: start, watch memory, wait for the stop file, remove the container
session() { # session <rank> <host> <env> <tp flags>
  local r=$1 h=$2 env="$3 ${TF_ENV:-}" tpf=$4 c="fnz-serve-${RUN}-r$1" hf=${HF[$1]} lock=${LOCK[$1]}
  on "$h" "cat > ~/$REMOTE/session.sh" <<EOF
set -u
rm -f ~/$REMOTE/stop$r ~/$REMOTE/exited$r
docker rm -f $c >/dev/null 2>&1
docker run -d --name $c --gpus all --network host --shm-size 1g $DEVS $env \\
  -v $hf:/root/.cache/huggingface:ro -v \$HOME/$REMOTE:/srv:ro -e TENSORFOLD_CUDA_KERNELS=/srv/kernels \\
  --entrypoint /srv/bin/tensorfold-native '$IMAGE' serve $SNAP --name $NAME --host 0.0.0.0 --port $PORT \\
  --context ${TF_CONTEXT:-32768} --parallel ${TF_PARALLEL:-1} ${TF_EXTRA:-} $tpf >/dev/null || exit 1
while [ ! -f ~/$REMOTE/stop$r ]; do
  [ "\$(docker inspect -f '{{.State.Running}}' $c 2>/dev/null)" = true ] || {
    docker inspect -f '{{.State.ExitCode}} {{.State.FinishedAt}}' $c > ~/$REMOTE/exited$r; echo "rank $r container exited"; break; }
  a=\$(awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo)
  if [ "\$a" -lt ${TF_MIN_AVAILABLE_GIB:-8} ]; then echo "watchdog: MemAvailable \$a GiB, stopping"; docker stop -t 5 $c; break; fi
  sleep 1
done
docker stop -t 30 $c >/dev/null 2>&1 # SIGTERM first: a rank stops its peer and frees its GPU itself
docker logs $c > ~/$REMOTE/rank$r.log 2>&1
docker rm -f $c >/dev/null 2>&1
EOF
  on "$h" "bash ~/$REMOTE/session.sh" & # under the lock this run holds
  SESSIONS+=($!)
}
SESSIONS=()
stop_all() {
  # rank 0 first (it stops rank 1); rank 1 gets 30 s to leave on its own before it is stopped too
  on "$HOST0" "touch ~/$REMOTE/stop0"
  if (( TP == 2 )); then
    for _ in $(seq 30); do on "$HOST1" "docker inspect -f '{{.State.Running}}' fnz-serve-${RUN}-r1 2>/dev/null" | grep -q true || break; sleep 1; done
    on "$HOST1" "touch ~/$REMOTE/stop1"
  fi
  for p in "${SESSIONS[@]}"; do wait "$p" 2>/dev/null; done
  SESSIONS=()
  for i in "${!HOSTS[@]}"; do get "${HOSTS[$i]}" "$REMOTE/rank$i.log" "$OUT/" 2>/dev/null; done
}
trap 'stop_all; release_locks' EXIT
(( TP == 2 )) && session 1 "$HOST1" "$ENV1" "$TPFLAGS1"
session 0 "$HOST0" "$ENV0" "$TPFLAGS0"

URL="http://127.0.0.1:$PORT"
t0=$(date +%s)
until on "$HOST0" "curl -sf $URL/health >/dev/null"; do
  (( $(date +%s) - t0 > 1500 )) && { echo "no /health after 25 min"; exit 1; }
  on "$HOST0" "docker inspect -f '{{.State.Running}}' fnz-serve-${RUN}-r0 2>/dev/null" | grep -q true || {
    (( $(date +%s) - t0 > 30 )) && { echo "rank 0 exited"; on "$HOST0" "docker logs --tail 40 fnz-serve-${RUN}-r0" 2>&1; exit 1; }; }
  sleep 10
done
echo "serving after $(( $(date +%s) - t0 )) s"

if [[ "${TF_CHECKS:-1}" == 1 ]]; then
post() { # post <name> <json>: the response into checks/<name>.json
  on "$HOST0" "curl -s -m 600 -H 'Content-Type: application/json' -d @- $URL/v1/chat/completions" <<<"$2" > "$OUT/checks/$1.json"
}
on "$HOST0" "curl -s $URL/v1/models" > "$OUT/checks/models.json"
MSG='"messages": [{"role": "user", "content": "Write a Python function that returns the nth Fibonacci number, with a docstring."}]'
BASE="\"model\": \"$NAME\", $MSG, \"max_tokens\": 160, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false}"
post greedy "{ $BASE }"
post greedy_nodraft "{ $BASE, \"draft\": false }"
post greedy_again "{ $BASE }"
on "$HOST0" "curl -sN -m 600 -H 'Content-Type: application/json' -d @- $URL/v1/chat/completions" <<<"{ $BASE, \"stream\": true }" > "$OUT/checks/stream.sse"
TOOLS='"tools": [{"type": "function", "function": {"name": "get_weather", "description": "The weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]'
post tools "{ \"model\": \"$NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"What is the weather in Zurich right now?\"}], $TOOLS, \"max_tokens\": 200, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false} }"
post sampled "{ $BASE, \"temperature\": 0.7, \"seed\": 1234 }"
post prose "{ \"model\": \"$NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Write a short story of about 300 words about a lighthouse keeper who finds a message in a bottle.\"}], \"max_tokens\": 400, \"temperature\": 0, \"chat_template_kwargs\": {\"enable_thinking\": false} }"
post sampled_nodraft "{ $BASE, \"temperature\": 0.7, \"seed\": 1234, \"draft\": false }"
# a long prompt (its prefill call is past a record: rank 1 reads it from the link) with a needle in its middle
python3 -B - "${TF_LONG_WORDS:-6000}" "$NAME" > "$OUT/checks/long.req" <<'PY'
import json, sys
n, name = int(sys.argv[1]), sys.argv[2]
words = ("the quiet harbor kept its boats in rows while gulls argued over nets and the tide came slowly in " * (n // 20 + 1)).split()[:n]
words.insert(n // 2, "The secret code is MAPLE-4417.")
print(json.dumps({"model": name, "max_tokens": 40, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                  "messages": [{"role": "user", "content": " ".join(words) + "\n\nWhat is the secret code in the text above? Answer with the code only."}]}))
PY
on "$HOST0" "curl -s -m 1200 -H 'Content-Type: application/json' -d @- $URL/v1/chat/completions" < "$OUT/checks/long.req" > "$OUT/checks/long.json"
on "$HOST0" "curl -s $URL/metrics" > "$OUT/checks/metrics.txt" 2>/dev/null
on "$HOST0" "curl -s -H 'Content-Type: application/json' -d @- $URL/tokenize" <<<"{ $BASE }" > "$OUT/checks/tokenize.json"

python3 -B - "$OUT/checks" <<'EOF' | tee "$OUT/checks.txt"
import json, sys, os
d = sys.argv[1]
def load(n):
    try: return json.load(open(os.path.join(d, n + ".json")))
    except Exception as e: return {"error": f"unreadable: {e}"}
def text(r):
    try: return r["choices"][0]["message"].get("content") or ""
    except Exception: return None
ok = True
def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f": {detail}" if detail else ""))
m = load("models")
check("/v1/models names the model", any(x.get("id") == os.environ.get("TF_NAME", "Qwen3.8-Flash-Next") for x in m.get("data", [])), json.dumps(m)[:200])
g, nd, again = load("greedy"), load("greedy_nodraft"), load("greedy_again")
tg = text(g)
check("greedy chat completion", bool(tg), json.dumps(g)[:300] if not tg else repr(tg[:120]))
check("draft false equals drafted", tg is not None and tg == text(nd), repr((text(nd) or "")[:120]))
check("a repeat equals the first", tg == text(again))
sse = open(os.path.join(d, "stream.sse")).read().splitlines()
parts = []
for line in sse:
    if line.startswith("data: ") and line[6:].strip() != "[DONE]":
        try:
            delta = json.loads(line[6:])["choices"][0].get("delta", {})
            parts.append(delta.get("content") or "")
        except Exception: pass
check("streaming equals the reply", "".join(parts) == tg, f"{len(parts)} chunks")
t = load("tools")
try: calls = t["choices"][0]["message"].get("tool_calls") or []
except Exception: calls = []
check("tool call get_weather", any(c["function"]["name"] == "get_weather" and "urich" in c["function"]["arguments"] for c in calls), json.dumps(t)[:300])
check("sampled draft false equals drafted", text(load("sampled")) == text(load("sampled_nodraft")) and text(load("sampled")))
lg = load("long")
check("no stop token in the replies", all("<|im_end|>" not in (text(load(n)) or "") for n in ("greedy", "long", "sampled")))
check("long prompt finds the needle", "MAPLE-4417" in (text(lg) or ""), f"prompt {lg.get('usage', {}).get('prompt_tokens')} tokens: {(text(lg) or json.dumps(lg))[:120]!r}")
for n in ("greedy", "greedy_nodraft"):
    print(n, "usage", json.dumps(load(n).get("usage")))
for n in ("greedy", "prose", "sampled", "greedy_nodraft"):
    r = load(n); t = r.get("tensorfold", {}); sp = r.get("speculative", {})
    print(f"speed {n}: {t.get('tokens_per_second', 0):.1f} tok/s, {sp.get('rounds')} rounds, {sp.get('accepted')}/{sp.get('drafted')} kept, {r.get('usage', {}).get('completion_tokens')} tokens")
print("ALL PASS" if ok else "SOME FAILED")
EOF
fi

# concurrency: TF_CONC="2 4 8" requests at once must reply exactly as each alone (greedy and sampled)
if [[ -n "${TF_CONC:-}" ]]; then
  on "$HOST0" "python3 - $PORT $NAME ${TF_CONC_TOKENS:-160} ${TF_CONC}" > "$OUT/conc.txt" 2>&1 <<'PY'
import json, sys, threading, time, urllib.request
port, name, max_tokens, ns = sys.argv[1], sys.argv[2], int(sys.argv[3]), [int(x) for x in sys.argv[4:]]
prompts = ["Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
           "Explain in four sentences why the sky looks blue during the day and red at sunset.",
           "List the first twelve prime numbers, then give their sum and show the addition.",
           "Translate into French: 'The old bridge was repaired before the winter storms came to the valley.'",
           "Write a short JSON object describing a library book with title, author, year and three subject tags.",
           "A train leaves at 09:40 and arrives at 13:15. How long is the trip? Show the steps.",
           "Write a haiku about a lighthouse, then explain its imagery in two sentences.",
           "Give three tips for writing clear commit messages, each with a one-line example."]
def ask(i, sampled):
    body = {"model": name, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompts[i % len(prompts)]}],
            "chat_template_kwargs": {"enable_thinking": False}, "temperature": 0.7 if sampled else 0, "seed": 1000 + i}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return r["choices"][0]["message"].get("content"), r.get("tensorfold", {}).get("token_sha"), r["usage"]["completion_tokens"]
ok = True
for sampled in (False, True):
    mode = "sampled" if sampled else "greedy"
    top = max(ns)
    alone = [ask(i, sampled) for i in range(top)]
    for n in ns:
        got = [None] * n
        def run(i):
            got[i] = ask(i, sampled)
        t0 = time.time()
        ts = [threading.Thread(target=run, args=(i,)) for i in range(n)]
        [t.start() for t in ts]; [t.join() for t in ts]
        dt = time.time() - t0
        same = [got[i][:2] == alone[i][:2] for i in range(n)]
        toks = sum(g[2] for g in got)
        ok &= all(same)
        print(f"{'PASS' if all(same) else 'FAIL'} {mode} x{n}: {sum(same)}/{n} equal to alone; {toks} tokens in {dt:.1f} s = {toks/dt:.1f} tok/s together")
print("CONC ALL PASS" if ok else "CONC SOME FAILED")
PY
  cat "$OUT/conc.txt" | tee -a "$OUT/checks.txt"
fi

# prompt reuse: TF_REUSE=1 runs the recipe's tools/prompt_reuse.py (TF_REUSE_ARGS, default "30000 4 --other") and a
# 3-turn conversation whose turns' token shas go to reuse.txt (compare a run with TF_EXTRA="--prompt-cache-gib 0")
if [[ "${TF_REUSE:-0}" == 1 ]]; then
  TOOLS="${TF_TOOLS:-$TF/../tools}"
  put "$HOST0" "$TOOLS/" "$REMOTE/tools/" || exit 1
  on "$HOST0" "cd ~/$REMOTE/tools && PORT=$PORT MODEL=$NAME timeout 3000 python3 -B prompt_reuse.py ${TF_REUSE_ARGS:-30000 4 --other}" > "$OUT/reuse_tool.txt" 2>&1
  echo "prompt_reuse.py exit $?" >> "$OUT/reuse_tool.txt"
  on "$HOST0" "cd ~/$REMOTE/tools && PORT=$PORT MODEL=$NAME python3 -B -" > "$OUT/reuse.txt" 2>&1 <<'PY'
import client
filler = " ".join(f"Entry {i}: the harbor log records {i % 7} boats, wind {i % 5} knots, tide {'rising' if i % 2 else 'falling'}." for i in range(700))
convo = [{"role": "system", "content": "You are a careful assistant. Answer in one sentence."},
         {"role": "user", "content": "Here is a harbor log.\n\n" + filler + "\n\nWhat is the most common wind speed?"}]
for turn in range(3):
    r = client.chat(convo, max_tokens=40, thinking=False, temperature=0.0, seed=7)
    print(f"turn {turn}: prompt {r.prompt_tokens} cached {r.cached_tokens} sha {r.tensorfold.get('token_sha')} prefill {r.prefill_seconds}")
    convo += [{"role": "assistant", "content": r.content}, {"role": "user", "content": f"And on day {turn + 2}, what changed? Answer briefly."}]
PY
  cat "$OUT/reuse_tool.txt" "$OUT/reuse.txt" | tee -a "$OUT/checks.txt"
fi

# peer loss: kill rank TF_KILL (idle, or with TF_KILL_BUSY=1 inside a long prompt's collectives) and time the other's exit
if [[ -n "${TF_KILL:-}" ]] && (( TP == 2 )); then
  k=$TF_KILL; o=$(( 1 - k )); hk=${HOSTS[$k]}; ho=${HOSTS[$o]}; what=idle
  if [[ "${TF_KILL_BUSY:-0}" == 1 ]]; then
    what=busy
    python3 -B - "${TF_KILL_WORDS:-24000}" "$NAME" > "$OUT/kill.req" <<'PY'
import json, sys
n, name = int(sys.argv[1]), sys.argv[2]
words = ("the quiet harbor kept its boats in rows while gulls argued over nets and the tide came slowly in " * (n // 20 + 1)).split()[:n]
print(json.dumps({"model": name, "max_tokens": 400, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                  "messages": [{"role": "user", "content": " ".join(words) + "\n\nSummarize the text above."}]}))
PY
    on "$HOST0" "curl -s -m 600 -H 'Content-Type: application/json' -d @- $URL/v1/chat/completions" < "$OUT/kill.req" > "$OUT/kill.json" 2>&1 &
    CURLPID=$!
    sleep "${TF_KILL_AFTER_S:-6}"
  fi
  t0=$(date +%s.%N)
  sig=${TF_KILL_SIGNAL:-KILL}; what="$what, SIG$sig"
  on "$hk" "docker kill --signal $sig fnz-serve-${RUN}-r$k >/dev/null"
  gone=""
  for _ in $(seq 240); do
    on "$ho" "docker inspect -f '{{.State.Running}}' fnz-serve-${RUN}-r$o 2>/dev/null" | grep -q true || { gone=$(date +%s.%N); break; }
    sleep 0.25
  done
  if [[ -n "$gone" ]]; then
    printf 'PASS kill rank %d (%s): rank %d exited %.1f s later\n' "$k" "$what" "$o" "$(echo "$gone - $t0" | bc)" | tee -a "$OUT/checks.txt"
  else
    echo "FAIL kill rank $k ($what): rank $o still running after 60 s" | tee -a "$OUT/checks.txt"
  fi
  [[ -n "${CURLPID:-}" ]] && wait "$CURLPID" 2>/dev/null # the busy request (not the lock holders)
  trap release_locks EXIT; stop_all
  for i in 0 1; do get "${HOSTS[$i]}" "$REMOTE/exited$i" "$OUT/" 2>/dev/null; done
  echo "rank $k (signalled): exit $(cat "$OUT/exited$k" 2>/dev/null); rank $o: exit $(cat "$OUT/exited$o" 2>/dev/null)"
  for i in $k $o; do echo "rank $i's last words:"; grep -E "rank [01]|stopped|stop signal|exit" "$OUT/rank$i.log" | grep -v "NCCL INFO" | tail -2; done
fi
# TF_AFTER: a command run here while the server is still up (more checks, benches), then the hold
if [[ -n "${TF_AFTER:-}" ]]; then echo "running TF_AFTER"; bash -c "$TF_AFTER" 2>&1 | tee "$OUT/after.txt"; fi
if [[ "${TF_HOLD_S:-0}" != 0 ]]; then echo "holding $TF_HOLD_S s"; sleep "$TF_HOLD_S"; fi
if [[ -n "${TF_CLI:-}" ]]; then
  trap release_locks EXIT; stop_all
  ids=$(python3 -c 'import json,sys; print(",".join(map(str, json.load(open(sys.argv[1]))["tokens"])))' "$OUT/checks/tokenize.json")
  put "$HOST0" "$TF_CLI" "$REMOTE/bin/" || exit 1
  for mode in drafted serial; do
    flag=""; [[ $mode == serial ]] && flag="--no-drafts"
    on "$HOST0" "docker run --rm --name fnz-serve-${RUN}-cli --gpus all --network none ${TF_ENV:-} \
      -v ${HF[0]}:/root/.cache/huggingface:ro -v \$HOME/$REMOTE:/srv -e TENSORFOLD_CUDA_KERNELS=/srv/kernels \
      --entrypoint /srv/bin/tensorfold '$IMAGE' run $SNAP --tokens $ids --max-tokens 160 --context ${TF_CONTEXT:-32768} \
      --kernels /srv/kernels --report /srv/cli-$mode.json $flag" > "$OUT/cli-$mode.log" 2>&1
    tail -1 "$OUT/cli-$mode.log"
    get "$HOST0" "$REMOTE/cli-$mode.json" "$OUT/"
  done
  want=$(grep -m1 -o 'sha=[0-9a-f]*' "$OUT/rank0.log" | cut -d= -f2)
  for mode in drafted serial; do
    # the server's sha: sha256 of the reply's ids joined by commas, 12 hex digits (chat.zig tokenSha)
    got=$(python3 -c 'import json,sys,hashlib; t=json.load(open(sys.argv[1]))["tokens"]; print(hashlib.sha256(",".join(map(str,t)).encode()).hexdigest()[:12])' "$OUT/cli-$mode.json")
    if [[ -n "$want" && "$got" == "$want" ]]; then echo "PASS engine CLI $mode sha $got == server's"; else echo "FAIL engine CLI $mode sha '$got' vs server's '$want'"; fi
  done | tee -a "$OUT/checks.txt"
fi
