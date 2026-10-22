#!/usr/bin/env python3
"""Flash Next's Python CUDA engine (qwen4_exp, one GPU) as the Zig engine's oracle: every Triton launch and extension
call by phase, weight digests, per-layer dumps, teacher-forced logits, MTP draws, tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

TEXTS = {
    "code": "Write a Python function that merges two sorted lists, with a docstring and three tests.",
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "facts": "Explain how a refrigerator moves heat out of its cabinet, step by step.",
}
LONG = {"long": ("textwrap", 3000),      # ~3k tokens: a full 2048-row prompt chunk and a partial one
        "huge": ("inspect", 9000)}       # past the 8192-key decode bucket: five chunks, 16384-key graphs
HEAD = "Review this module and list its public functions:\n\n"
TEACHER = 48                             # teacher-forced one-row steps from an empty state
TEACHER_PROMPT = "long"                  # its first 48 tokens (the short prompts have ~30)
DUMP_STEPS = (0, 1, TEACHER - 1)         # steps whose tensors are written in full (all steps keep digests)
DETAIL_STEPS = (0, 1)                    # teacher steps whose launches are logged one by one
FULL_LAYERS_LONG = (0, 2, 3, 47)         # the long prompt's first chunk: these layers' streams in full
SAMPLED = "1234,1.0,20,0.95,0.0"         # the served sampling defaults with a fixed seed


def render(model: Path) -> tuple[dict[str, list[int]], dict[str, str]]:
    """Every prompt through the checkpoint's chat template (thinking off) and tokenizer, as the server renders."""

    import importlib

    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate

    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    tpl = ChatTemplate(model)
    users = dict(TEXTS)
    for name, (module, size) in LONG.items():
        words = Path(importlib.import_module(module).__file__).read_text()
        ids = tok.encode(words * (1 + size // max(1, len(words) // 4)), add_special_tokens=False).ids[:size]
        users[name] = HEAD + tok.decode(ids)
    texts = {n: tpl.render([{"role": "user", "content": u}], tools=None, enable_thinking=False)
             for n, u in users.items()}
    return {n: [int(t) for t in tok.encode(t, add_special_tokens=False).ids] for n, t in texts.items()}, texts


def sha12(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps([int(t) for t in tokens]).encode()).hexdigest()[:12]


CHUNK = 256 << 20


def digest(t) -> str:
    """sha256 of a tensor's bytes (contiguous, little-endian), read back in 256 MiB pieces."""

    import torch

    raw = t.detach()
    if not raw.is_contiguous():
        raw = raw.contiguous()
    h = hashlib.sha256()
    if raw.numel():
        flat = raw.reshape(-1).view(torch.uint8)
        for i in range(0, flat.numel(), CHUNK):
            h.update(flat[i:i + CHUNK].cpu().numpy())
    return h.hexdigest()


def kind(t) -> str:
    return "x".join(map(str, t.shape)) + ":" + str(t.dtype).replace("torch.", "")


class Dumps:
    """Per-layer tensors: a digest of each (``digests.json``), the raw bytes of those ``full`` selects."""

    def __init__(self) -> None:
        self.dir: Path | None = None
        self.full = lambda name: True
        self.digests: dict[str, dict] = {}
        self.prefix = ""
        self.mtp = False

    def begin(self, d: Path | None, full=True) -> None:
        self.end()
        self.dir, self.digests, self.prefix = d, {}, ""
        self.full = full if callable(full) else (lambda name, f=bool(full): f)
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)

    def end(self) -> None:
        if self.dir is not None:
            (self.dir / "digests.json").write_text(json.dumps(self.digests, indent=0, sort_keys=True) + "\n")
        self.dir = None

    def put(self, name: str, t) -> None:
        import torch

        if self.dir is None or t is None or not isinstance(t, torch.Tensor):
            return
        name = self.prefix + ("mtp_" if self.mtp else "") + name
        base, n = name, 1
        while name in self.digests:
            n += 1
            name = f"{base}#{n}"
        torch.cuda.synchronize()
        raw = t.detach().contiguous()
        self.digests[name] = {"sha256": digest(raw), "shape": list(raw.shape),
                              "dtype": str(raw.dtype).replace("torch.", "")}
        if self.full(base):
            raw.view(torch.uint8).cpu().numpy().tofile(self.dir / f"{name.replace('#', '_')}.bin")
            self.digests[name]["file"] = f"{name.replace('#', '_')}.bin"


class Hooks:
    """Wraps the engine's module functions: dumps, phases inside startup, graph keys, MTP draws."""

    def __init__(self, rec, dumps: Dumps) -> None:
        self.rec, self.dumps = rec, dumps
        self.graphs: dict[str, Counter] = defaultdict(Counter)    # phase -> graph key -> replays
        self.captured: list[dict] = []
        self.draws: list[dict] = []
        self.key = None

    def phase(self):
        return self.rec.phase if self.rec is not None else "run"

    def scope(self, name: str, detail: bool = False):
        return self.rec.scope(name, detail) if self.rec is not None else nullcontext()

    def install(self) -> None:
        from tensorfold.families.qwen4_exp.cuda import decode as D
        from tensorfold.families.qwen4_exp.cuda import forward as F
        from tensorfold.families.qwen4_exp.cuda import graphs as G
        from tensorfold.families.qwen4_exp.cuda import mtp as M
        from tensorfold.families.qwen4_exp.cuda import weights as W

        d, hooks = self.dumps, self

        def tag(layer) -> str:
            return f"L{layer.index:02d}_"

        layer_forward = F.layer_forward

        def traced_layer(layer, w, segs, b, R, pending, **kw):
            d.put(tag(layer) + "h_in", b.h[:R])
            out = layer_forward(layer, w, segs, b, R, pending, **kw)
            return out

        F.layer_forward = traced_layer
        M.layer_forward = traced_layer

        ple_block = F.ple_block

        def traced_ple(layer, w, segs, b, R):
            ple_block(layer, w, segs, b, R)
            d.put(tag(layer) + "ple_emb", b.ple_emb[:R])
            d.put(tag(layer) + "h_ple", b.h[:R])

        F.ple_block = traced_ple

        def mixer(fn, what):
            def traced(layer, w, segs, b, R, *a, **kw):
                d.put(tag(layer) + "hc_a", b.mixed[:R])
                mode, branch = fn(layer, w, segs, b, R, *a, **kw)
                d.put(tag(layer) + f"{what}_m{mode}", branch)
                return mode, branch
            return traced

        F.gdn_block = mixer(F.gdn_block, "gdn")
        F.attn_block = mixer(F.attn_block, "attn")
        moe_block = F.moe_block

        def traced_moe(layer, w, b, R):
            d.put(tag(layer) + "hc_m", b.mixed[:R])
            out = moe_block(layer, w, b, R)
            m = b.moe
            for name in ("logits", "pick", "wts"):
                t = getattr(m, name, None)
                d.put(tag(layer) + "router_" + name, t[:R] if t is not None else None)
            d.put(tag(layer) + f"moe_y_m{out[0]}", out[1])
            return out

        F.moe_block = traced_moe
        finish = F.finish

        def traced_finish(w, mixer_hc, b, R, pending, logits=True, ends=()):
            out = finish(w, mixer_hc, b, R, pending, logits=logits, ends=ends)
            d.put("final_streams", b.streams[:R])
            d.put("final_mixed", b.mixed[:max(1, len(ends)) if b.prefill else R])
            d.put("logits", out)
            return out

        F.finish = traced_finish
        M.finish = traced_finish
        mtp_compute = M.mtp_compute

        def traced_mtp(w, segs, b, **kw):
            was, d.mtp = d.mtp, True
            try:
                out = mtp_compute(w, segs, b, **kw)
                d.put("head_logits", out)
                return out
            finally:
                d.mtp = was

        M.mtp_compute = traced_mtp
        G.mtp_compute = traced_mtp
        chunk = D.prefill_chunk

        def traced_chunk(e, prompt, start, **kw):
            d.prefix = f"c{start // e.prefill_rows:02d}_"
            try:
                return chunk(e, prompt, start, **kw)
            finally:
                d.prefix = ""

        D.prefill_chunk = traced_chunk

        sample_draft = D.Engine.sample_draft

        def traced_draft(eng, logits, position, sampling):
            tok, p = sample_draft(eng, logits, position, sampling)
            hooks.draws.append({"phase": hooks.phase(), "position": int(position), "token": int(tok),
                                "p": float(p), "logits_sha256": digest(logits[:1])})
            return tok, p

        D.Engine.sample_draft = traced_draft

        # phases inside the engine's constructor
        load = W.load

        def traced_load(*a, **kw):
            with hooks.scope("startup-load", detail=True):
                return load(*a, **kw)

        W.load = traced_load
        warm = D.warm

        def traced_warm(e):
            with hooks.scope("startup-warm", detail=True):
                return warm(e)

        D.warm = traced_warm
        gwarm = G.Graphs.warm

        def traced_gwarm(g, rows=None):
            with hooks.scope("startup-graphs"):
                return gwarm(g, rows)

        G.Graphs.warm = traced_gwarm
        gforward, gmtp, capture = G.Graphs.forward, G.Graphs.mtp_forward, G.Graphs._capture

        def traced_gforward(g, tokens):
            st, R = g.e.st, len(tokens)
            hooks.key = f"main R{R} parity{st.cur[0] if st.cur else 0} ctx{g._bucket(st.pos + R)}"
            hooks.graphs[hooks.phase()][hooks.key if R <= g.max_rows else f"eager R{R}"] += 1
            return gforward(g, tokens)

        def traced_gmtp(g, next_tokens, streams):
            st, n = g.e.st, len(next_tokens)
            hooks.key = f"mtp n{n} ctx{g._bucket(st.mtp_len + n)}"
            hooks.graphs[hooks.phase()][hooks.key if n <= g.max_rows else f"mtp-eager n{n}"] += 1
            return gmtp(g, next_tokens, streams)

        def traced_capture(g, fn):
            hooks.captured.append({"phase": hooks.phase(), "key": hooks.key})
            if hooks.rec is None:
                return capture(g, fn)
            hooks.rec._entry({"kind": "mark", "what": "graph capture", "key": hooks.key})
            with hooks.rec.scope(hooks.rec.phase, True):
                return capture(g, fn)

        G.Graphs.forward, G.Graphs.mtp_forward, G.Graphs._capture = traced_gforward, traced_gmtp, traced_capture


def extensions(rec) -> dict[str, list[str]]:
    """Load every extension the NVFP4 engine uses (``build_kernels`` with nvfp4 and solo) and log all calls."""

    from tensorfold.cuda import experts
    from tensorfold.cuda.kernels import gdn as shared_gdn
    from tensorfold.cuda.kernels import qmm
    from tensorfold.cuda.nvfp4 import checkpoint, linear
    from tensorfold.families.qwen4_exp.cuda import gdn, gdn_io

    mods = {"experts": experts._ext, "gdn": shared_gdn._ext, "qmm": qmm._ext, "gdn_io": gdn_io._ext,
            "fn_gdn": gdn._ext, "nvfp4": linear._ext, "nvfp4_prompt": linear._prompt_ext,
            "nvfp4_ck": checkpoint._ext}
    out = {}
    for label, load in mods.items():
        try:
            mod = load()
        except Exception as exc:                      # noqa: BLE001  (recorded: an extension this GPU refuses)
            out[label] = [f"LOAD FAILED: {exc!r}"]
            continue
        names = sorted(n for n in dir(mod) if not n.startswith("_") and callable(getattr(mod, n)))
        out[label] = names + [f"module {getattr(mod, '__name__', '?')} {getattr(mod, '__file__', '?')}"]
        if rec is not None:
            rec.wrap(mod, tuple(names), label)
    return out


def weight_digests(w) -> tuple[dict[str, str], dict]:
    """sha256 of every device tensor the engine's weights hold, by a dotted name; host-side tables described."""

    import numpy as np
    import torch

    out: dict[str, str] = {}
    host: dict[str, dict] = {}
    seen: set[int] = set()

    def walk(prefix: str, v) -> None:
        if isinstance(v, torch.Tensor):
            if v.is_cuda:
                out[prefix] = digest(v)
                out[prefix + ".shape"] = kind(v)
            else:
                host[prefix] = {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": digest(v)
                                if v.numel() * v.element_size() < 1 << 26 else None}
            return
        if isinstance(v, np.ndarray):
            host[prefix] = {"shape": list(v.shape), "dtype": str(v.dtype),
                            "sha256": hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                            if v.nbytes < 1 << 26 else None}
            return
        if v is None or isinstance(v, (int, float, str, bool, bytes)):
            return
        if id(v) in seen:
            return
        seen.add(id(v))
        if isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                walk(f"{prefix}.{i}" if prefix else str(i), x)
        elif isinstance(v, dict):
            for k, x in v.items():
                walk(f"{prefix}.{k}" if prefix else str(k), x)
        elif hasattr(v, "__dataclass_fields__") or type(v).__module__.startswith("tensorfold"):
            names = list(v.__dataclass_fields__) if hasattr(v, "__dataclass_fields__") else []
            names += [n for n in getattr(v, "__dict__", {}) if n not in names]
            for name in names:
                if name in ("comm", "cfg"):
                    continue
                walk(f"{prefix}.{name}" if prefix else name, getattr(v, name, None))
            if type(v).__name__ in ("HostTable", "SSDTable", "BF16Table", "NGram"):
                host[prefix + ".__class__"] = {"class": f"{type(v).__module__}.{type(v).__qualname__}"}

    walk("", w)
    return out, host


def specialization() -> dict:
    """Each JIT function's parameter names and which ones Triton never specializes."""

    import gc

    from triton.runtime.jit import JITFunction

    out = {}
    for fn in gc.get_objects():
        if isinstance(fn, JITFunction):
            name = f"{fn.fn.__module__}.{fn.fn.__qualname__}"
            out[name] = {"params": [p.name for p in fn.params], "kind": type(fn).__name__,
                         "do_not_specialize": [p.name for p in fn.params if p.do_not_specialize],
                         "no_align": [p.name for p in fn.params if p.do_not_specialize_on_alignment]}
    return out


def profile(out: Path, name: str, fn) -> None:
    """torch.profiler over ``fn``: every CUDA kernel by name (Triton, extensions and torch's own ops)."""

    import torch
    from torch.profiler import ProfilerActivity, profile as prof

    torch.cuda.synchronize()
    with prof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    ev = p.key_averages()
    (out / f"profile-{name}.txt").write_text(ev.table(sort_by="cuda_time_total", row_limit=400) + "\n")
    rows = []
    for e in ev:
        dev = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
        rows.append({"name": e.key, "count": e.count, "device_us": dev, "cpu_us": e.cpu_time_total})
    (out / f"profile-{name}.json").write_text(json.dumps(rows, indent=0) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tools", default="", help="folder of triton_aot_manifest.py: record every Triton launch")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--context", type=int, default=0, help="--context N as serve takes it (0: serve's default)")
    ap.add_argument("--no-weights", action="store_true", help="skip weights.json (75 GiB of digests)")
    ap.add_argument("--only", default="", help="comma list of prompt names (default all)")
    ap.add_argument("--digests-only", action="store_true", help="prompt-chunk dumps as digests only (no .bin)")
    ap.add_argument("--no-profile", action="store_true", help="skip the torch.profiler runs")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = None
    if a.tools:
        sys.path.insert(0, a.tools)
        import triton_aot_manifest as aot

        rec = aot.Recorder().install()

    import torch

    from tensorfold.engine.exact_sampling import Sampling

    model = Path(a.model)
    ids, texts = render(model)
    if a.only:
        ids = {k: v for k, v in ids.items() if k in a.only.split(",")}
    (out / "prompts.json").write_text(json.dumps(ids) + "\n")
    (out / "prompts_text.json").write_text(json.dumps(texts, indent=1, ensure_ascii=False) + "\n")
    print({k: len(v) for k, v in ids.items()}, flush=True)
    dumps = Dumps()
    hooks = Hooks(rec, dumps)
    hooks.install()
    scope = hooks.scope

    t0 = time.perf_counter()
    with scope("startup-ext", detail=True):
        ext = extensions(rec)
    from tensorfold.families.qwen4_exp import cuda_engine

    with scope("startup"):
        kw = {"context": a.context, "context_explicit": True} if a.context else {"context_explicit": False}
        eng = cuda_engine(model, **kw)
    e = eng.e
    plan = {k: v for k, v in eng.capacity_plan.items() if isinstance(v, (int, float, str, bool, list))}
    info = {"startup_s": round(time.perf_counter() - t0, 2), "max_len": eng.max_len,
            "context_window": eng.context_window, "depth": eng.depth, "confidence": eng.confidence,
            "max_rows": e.rows, "prefill_rows": e.prefill_rows, "capacity": e.capacity, "eos": list(eng.eos),
            "graphs": None if e.graphs is None else len(e.graphs.main) + len(e.graphs.mtp),
            "torch": torch.__version__, "triton": __import__("triton").__version__,
            "gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability()),
            "context_arg": a.context or None, "capacity_plan": plan, "extensions": ext,
            "env": {k: v for k, v in os.environ.items() if k.startswith(("TENSORFOLD", "TF_", "TRITON", "TORCH",
                                                                         "CUDA", "PYTORCH"))},
            "meta": {k: v for k, v in e.w.meta.items() if isinstance(v, (int, float, str, bool))}}
    print(json.dumps(info, default=str), flush=True)
    if not a.no_weights:
        t = time.perf_counter()
        with scope("weights-digest"):
            digests, host = weight_digests(e.w)
        (out / "weights.json").write_text(json.dumps(digests, indent=0, sort_keys=True) + "\n")
        (out / "weights_host.json").write_text(json.dumps(host, indent=0, sort_keys=True) + "\n")
        info["weights_digest_s"] = round(time.perf_counter() - t, 1)
        print(f"weights.json: {len(digests) // 2} tensors in {info['weights_digest_s']}s", flush=True)

    def partial_dump():
        """Write what exists so far, so the other work items can start before the run ends."""

        if rec is not None:
            rec.dump(out / "launches.json")
            (out / "jit.json").write_text(json.dumps(specialization(), indent=1, sort_keys=True) + "\n")

    # 1. teacher-forced one-row steps from an empty state (eager: every launch in Python), with the MTP head absorbing
    from tensorfold.families.qwen4_exp.cuda.decode import sample_mapped
    from tensorfold.families.qwen4_exp.cuda.forward import commit
    from tensorfold.families.qwen4_exp.cuda.mtp import mtp_forward

    graphs, e.graphs = e.graphs, None
    if TEACHER_PROMPT not in ids:
        raise SystemExit(f"the teacher steps read the {TEACHER_PROMPT!r} prompt: keep it in --only")
    tf = ids[TEACHER_PROMPT][:TEACHER]
    st, w = e.st, e.w
    sampled, logit_sha, mtp_tokens, mtp_sha = [], [], [], []
    e.reset()
    for i, t in enumerate(tf):
        with scope("teacher", detail=i in DETAIL_STEPS):
            dumps.begin(out / "teacher" / f"step{i:03d}", full=i in DUMP_STEPS)
            logits = e.forward([t])
            sampled.append(e.sample(logits[:1], [st.pos + 1], None)[0])
            logit_sha.append(digest(logits[:1]))
            if i in DUMP_STEPS:
                torch.cuda.synchronize()
            if i + 1 < len(tf) and w.mtp is not None:      # the MTP head absorbs (streams of row i, token i+1)
                ml = mtp_forward(w, st, e.mbuf, [tf[i + 1]], e.buf.streams[:1])
                st.set_mtp_len(st.mtp_len + 1)
                mtp_tokens.append(sample_mapped(ml[:1], [st.pos + 2], None, w.draft_ids)[0]
                                  if w.draft_ids is not None else int(ml[:1].argmax()))
                mtp_sha.append(digest(ml[:1]))
            commit(w, st, e.buf, 1, 1)
            dumps.end()
    teacher = {"tokens": tf, "sampled": sampled, "logits_sha256": logit_sha, "mtp_next": mtp_tokens,
               "mtp_logits_sha256": mtp_sha, "dump_steps": list(DUMP_STEPS), "detail_steps": list(DETAIL_STEPS),
               "note": "step i: forward([tokens[i]]) at pos i, greedy sample, then mtp_forward([tokens[i+1]], "
                       "buf.streams[:1]) at mtp pos i, then commit(1, 1)"}
    e.reset()
    e.graphs = graphs

    # 2. prompt chunks with per-layer dumps (eager, prefill buffers), the long one crossing a chunk boundary
    from tensorfold.families.qwen4_exp.cuda.decode import prefill

    pre_out = {}
    for name in ("code", "long"):
        if name not in ids:
            continue
        full = False if a.digests_only else True if name == "code" else (lambda n: n.startswith("c00_") and any(
            n.startswith(f"c00_L{x:02d}_h") for x in FULL_LAYERS_LONG) or "logits" in n)
        with scope(f"prefill-{name}", detail=True):
            dumps.begin(out / "prefill" / name, full=full)
            first = prefill(e, ids[name], None)
            dumps.put("last_streams", e.last_streams)
            dumps.end()
        pre_out[name] = {"first": first, "rows": len(ids[name]), "pos": e.st.pos, "mtp_len": e.st.mtp_len,
                         "last_streams": digest(e.last_streams)}
        print(f"prefill {name}: {len(ids[name])} rows first {first}", flush=True)
    e.reset()
    teacher["prefill"] = pre_out
    (out / "teacher.json").write_text(json.dumps(teacher) + "\n")
    partial_dump()
    (out / "results.json").write_text(json.dumps({"info": info, "partial": True}, indent=1, default=str) + "\n")
    print("teacher.json written", flush=True)

    # 3. serial and MTP-drafted decoding, greedy and sampled, every prompt
    results: dict[str, dict] = {}
    seed, temperature, top_k, top_p, min_p = SAMPLED.split(",")
    samplings = {"greedy": None, "sampled": Sampling(int(seed), float(temperature), int(top_k), float(top_p),
                                                     float(min_p))}
    for sname, sampling in samplings.items():
        for draft in (False, True):
            label = f"{'drafted' if draft else 'serial'}-{sname}"
            for name, prompt in ids.items():
                runs = []
                for r in range(a.repeats):
                    toks: list[int] = []
                    first_draw = len(hooks.draws)
                    with scope(f"{label}-{name}"):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        stats = eng.generate(prompt, a.max_tokens, sampling,
                                             lambda new: toks.extend(map(int, new)) and False, draft=draft,
                                             stop_eos=True)
                        torch.cuda.synchronize()
                        wall = time.perf_counter() - start
                    runs.append({"tokens": toks, "sha": sha12(toks), "wall_s": round(wall, 4),
                                 "draws": [first_draw, len(hooks.draws)], **stats})
                last = runs[-1]
                same = all(x["tokens"] == last["tokens"] for x in runs)
                step = last.get("decode_s", 0.0) / max(1, len(last["tokens"]) - 1) * 1e3
                results[f"{label}/{name}"] = {**last, "repeats_same": same, "repeat_shas": [x["sha"] for x in runs],
                                              "ms_per_token": round(step, 4)}
                print(f"{label} {name}: {len(last['tokens'])} tokens sha {last['sha']} decode {last.get('decode_s')}s "
                      f"prefill {last.get('prefill_s')}s {step:.2f} ms/token rounds {last.get('rounds')} "
                      f"accepted {last.get('accepted')} same {same}", flush=True)
        for name in ids:
            s, d = results[f"serial-{sname}/{name}"], results[f"drafted-{sname}/{name}"]
            results[f"drafted==serial-{sname}/{name}"] = {"equal": s["tokens"] == d["tokens"]}
            print(f"drafted == serial {sname} {name}: {s['tokens'] == d['tokens']}", flush=True)

    # 4. every CUDA kernel by name (torch's own ops included) for one serial step run, one drafted run, one prompt
    if "code" in ids and not a.no_profile:
        prof_dir = out / "profile"
        prof_dir.mkdir(exist_ok=True)
        with scope("profile"):
            profile(prof_dir, "serial", lambda: eng.generate(ids["code"], 16, None, lambda new: False, draft=False))
            profile(prof_dir, "drafted", lambda: eng.generate(ids["code"], 32, None, lambda new: False, draft=True))
            profile(prof_dir, "prefill-long", lambda: prefill(e, ids.get("long", ids["code"]), None))
            e.reset()

    tree = {}
    try:
        tree["head"] = subprocess.run(["git", "-C", "/tensorfold", "rev-parse", "HEAD"], capture_output=True,
                                      text=True).stdout.strip()
        tree["src_status"] = subprocess.run(["git", "-C", "/tensorfold", "status", "--short", "src"],
                                            capture_output=True, text=True).stdout.splitlines()
    except Exception as exc:                               # noqa: BLE001
        tree["error"] = repr(exc)
    (out / "draws.json").write_text(json.dumps(hooks.draws) + "\n")
    (out / "graphs.json").write_text(json.dumps({"captured": hooks.captured,
                                                 "replays": {p: dict(c) for p, c in hooks.graphs.items()}},
                                                indent=1) + "\n")
    (out / "results.json").write_text(json.dumps({"info": info, "tree": tree, "results": results}, indent=1,
                                                 default=str) + "\n")
    partial_dump()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
