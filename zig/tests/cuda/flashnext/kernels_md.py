#!/usr/bin/env python3
"""A capture's kernel inventory as kernels.md: every Triton kernel (function, constexpr variants, grids, sites, launches
per phase) and every extension call (function, scalar/template arguments, shapes), grouped by role; graph replays and
the torch ops the profiler saw."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROLES = ("embed", "ngram/PLE", "HC", "GDN", "attention", "indexer/QSA", "router", "experts", "shared expert",
         "dense bf16 matmul", "norm", "head", "sampler", "MTP", "commit", "weight prep (startup)", "other")

# a kernel's own name decides first (module suffix, function)
BY_FUNCTION = [
    (("cuda.moe._router", "cuda.moe._topk_rows"), "router"),
    (("attention._pool", "attention._pool_block", "attention._select", "attention._select_tiles",
      "attention._scores", "attention._block_keys", "attn_multi._pool_multi"), "indexer/QSA"),
    (("glue._embed", "exl3_mm._embed", "affine_kernels.embed"), "embed"),
    (("glue._ple_conv", "glue._ple_embed", "glue._ple_embed_bf16", "glue._ple_gate", "exl3_mm._ple_rows"), "ngram/PLE"),
    (("glue._hc_act", "glue._hc_mix", "glue._hc_normed", "glue._hc_reduce_act", "glue._hc_writeback",
      "hc_fused._write_norm", "hc_upmix._upmix", "qmm._qmm_hcdown", "qmm._qmm_upmix"), "HC"),
    (("forward._shift_windows",), "commit"),
    (("glue._rmsnorm",), "norm"),
    (("glue._add_streams",), "MTP"),
    (("glue._attn_prep", "glue._attn_gate", "glue._prep_row"), "attention"),
    (("glue._moe_partial", "cuda.moe._combine", "experts.plan", "experts.run", "experts.prefill",
      "nvfp4.experts"), "experts"),
    (("fn_gdn.replay", "gdn.replay"), "commit"),
    (("fn_gdn.chain", "gdn.prefill", "gdn.tree", "gdn_io.front", "gdn_io.back"), "GDN"),
    (("qmm.qmm", "qmm.qmm_prefill", "qmm.qmm_group"), "MTP"),
]
# else the innermost frame of the engine that names a block decides
BY_SITE = [
    (("_shared", "shared_act"), "shared expert"),
    (("sample", "sample_rows", "sample_draft", "sample_mapped", "candidates", "nucleus_rows", "choose_rows"), "sampler"),
    (("commit", "replay"), "commit"),
    (("_readout", "_readout_b16", "_readout_plain", "_readout_fused", "_down_act", "hc_block", "_writeback"), "HC"),
    (("ple_block", "stage_ple_rows"), "ngram/PLE"),
    (("gdn_block", "_prefill_chain", "chain", "front", "back"), "GDN"),
    (("qsa_pool", "qsa_rows", "qsa_select"), "indexer/QSA"),
    (("attn_block", "attention"), "attention"),
    (("moe_block", "moe", "routed", "_exl3_moe"), "experts"),
    (("finish",), "head"),
    (("mtp_compute",), "MTP"),
    (("load", "_load", "load_layer", "pack", "_pack", "make_b16"), "weight prep (startup)"),
]


def clean(site: list[str]) -> list[str]:
    """A recorded site without the capture's own wrappers."""

    return [f for f in site if "tests/cuda/flashnext/capture.py" not in f]


def frames(site: str) -> list[str]:
    """``file:line func <- file:line func ...`` -> function names, innermost first."""

    return [part.strip().split(" ")[-1] for part in site.split(" <- ") if part.strip()]


SHARED = ("bf16._b16mm", "bf16._reduce", "nvfp4.qmmf")


def site_role(site: str) -> str:
    """The role of one launch site alone (for kernels that serve several blocks)."""

    fr = frames(site)
    mtp = "mtp_compute" in fr
    for fn in fr:
        for names, role in BY_SITE:
            if fn in names:
                return role + (" (MTP layer)" if mtp and role not in ("MTP", "head") else "")
    return "other"


def role_of(function: str, sites: list[str]) -> str:
    if any(function.endswith(n) for n in SHARED):
        return "dense bf16 matmul"
    for names, role in BY_FUNCTION:
        if any(function.endswith(n) for n in names):
            return role
    best = None
    for site in sites:
        for depth, fn in enumerate(frames(site)):
            for names, role in BY_SITE:
                if fn in names and (best is None or depth < best[0]):
                    best = (depth, role)
                    break
            else:
                continue
            break
    if best is not None:
        return best[1]
    if "bf16" in function or "b16" in function:
        return "dense bf16 matmul"
    return "other"


def phase_group(phase: str) -> str:
    for p in ("startup", "teacher", "prefill", "serial", "drafted", "profile", "weights"):
        if phase.startswith(p):
            return p
    return phase


def fmt_const(v) -> str:
    if isinstance(v, dict) and "float" in v:
        return f"{v['float']}"
    return json.dumps(v)


def arg_sig(a) -> str:
    if isinstance(a, dict) and "dtype" in a:
        return f"{a['dtype']}{a['shape']}" + ("" if a.get("align16", True) else "!a16")
    if isinstance(a, list):
        return "[" + ", ".join(arg_sig(x) for x in a) + "]"
    if isinstance(a, dict) and "float" in a:
        return f"{a['float']:g}"
    return json.dumps(a)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="the capture's out dir (launches.json, manifest.json, ...)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    run = Path(a.run)
    rec = json.loads((run / "launches.json").read_text())
    man = json.loads((run / "manifest.json").read_text()) if (run / "manifest.json").exists() else {"kernels": []}
    by_hash = {k["hash"]: k for k in man["kernels"]}
    detail_sites: dict[str, set] = defaultdict(set)
    site_roles: dict[str, Counter] = defaultdict(Counter)        # hash or ext name -> site role -> detail launches
    ext_detail: dict[str, Counter] = defaultdict(Counter)
    ext_sites: dict[str, set] = defaultdict(set)
    for row in rec.get("log", []):
        if row.get("kind") == "triton":
            detail_sites[row["hash"]].add(" <- ".join(clean(row.get("site", []))[:4]))
            site_roles[row["hash"]][site_role(" <- ".join(clean(row.get("site", []))))] += 1
        elif row.get("kind") == "ext":
            ext_detail[row["name"]][" ".join(arg_sig(x) for x in row["args"])] += 1
            ext_sites[row["name"]].add(" <- ".join(clean(row.get("site", []))[:4]))
            site_roles[row["name"]][site_role(" <- ".join(clean(row.get("site", []))))] += 1
    counts: dict[str, Counter] = defaultdict(Counter)       # key -> phase group -> launches
    for phase, keys in rec["phases"].items():
        for key, n in keys.items():
            counts[key][phase_group(phase)] += n
    groups = sorted({g for c in counts.values() for g in c})

    # Triton: one entry per Python function, its variants (hashes) beneath
    fns: dict[str, list[str]] = defaultdict(list)
    for h, info in rec["kernels"].items():
        fns[info["function"]].append(h)
    by_role: dict[str, list] = defaultdict(list)
    for fn, hashes in fns.items():
        sites = sorted({" <- ".join(clean(s.split(" <- "))) for h in hashes
                        for s in rec["sites"].get(h, []) + sorted(detail_sites.get(h, ()))})
        by_role[role_of(fn, sites)].append(("triton", fn, hashes, sites))
    ext_keys = sorted(k for k in counts if k not in rec["kernels"])
    for key in ext_keys:
        sites = sorted(ext_sites.get(key, ()))
        by_role[role_of(key, sites)].append(("ext", key, [], sites))

    out = []
    total_t = sum(len(h) for h in fns.values())
    out.append(f"# Kernel inventory: {run.name}\n")
    out.append(f"Generated by `zig/tests/cuda/flashnext/kernels_md.py` from `{run}/launches.json` and `manifest.json`. "
               f"{len(fns)} Triton functions in {total_t} specializations, {len(ext_keys)} extension functions. "
               "Launch counts are Python-side calls per phase group (graph replays are not calls: see the graph "
               f"section). Phase groups: {', '.join(groups)}.\n")
    out.append("| role | Triton functions | specializations | extension functions |\n| --- | ---: | ---: | ---: |")
    for role in ROLES:
        items = by_role.get(role, [])
        if not items:
            continue
        t = [x for x in items if x[0] == "triton"]
        out.append(f"| {role} | {len(t)} | {sum(len(x[2]) for x in t)} | {len(items) - len(t)} |")
    out.append("")
    if (run / "notes.md").exists():                    # the run's hand-written findings, kept across regenerations
        out.append((run / "notes.md").read_text().rstrip() + "\n")
    for role in ROLES:
        items = by_role.get(role, [])
        if not items:
            continue
        out.append(f"## {role}\n")
        for kind_, name, hashes, sites in sorted(items, key=lambda x: (x[0], x[1])):
            c = Counter()
            for h in hashes or [name]:
                c.update(counts.get(h, {}))
            per = ", ".join(f"{g} {c[g]}" for g in groups if c[g])
            if kind_ == "triton":
                info = rec["kernels"][hashes[0]]
                src = info["source"]
                out.append(f"### Triton `{name}`\n")
                out.append(f"- source `{src['file'].split('/tensorfold/', 1)[-1]}:{src['line']}`; kernel name "
                           f"`{info['name']}`; params {', '.join(info['params'])}")
                out.append(f"- launches: {per or 'none counted'}")
                keys = sorted({k for h in hashes for k in rec['kernels'][h]['constexprs']})
                same = {k for k in keys if len({fmt_const(rec['kernels'][h]['constexprs'].get(k)) for h in hashes}) == 1}
                if same:
                    k0 = rec["kernels"][hashes[0]]["constexprs"]
                    out.append("- constexprs (all variants): " + ", ".join(f"{k}={fmt_const(k0.get(k))}" for k in sorted(same)))
                out.append(f"- {len(hashes)} variant(s):\n")
                out.append("  | hash | differing constexprs | divisible-by-16 args | warps | regs | spills | shared | grids | launches |")
                out.append("  | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
                for h in sorted(hashes):
                    k = rec["kernels"][h]
                    m = by_hash.get(h, {})
                    diff = ", ".join(f"{x}={fmt_const(k['constexprs'].get(x))}" for x in keys if x not in same)
                    div = ",".join(n for n, v in k["attrs"].items() if v)
                    grids = rec["grids"].get(h, [])
                    gtxt = "; ".join("x".join(map(str, g)) for g in grids[:6]) + (f" (+{len(grids) - 6})" if len(grids) > 6 else "")
                    lc = ", ".join(f"{g} {n}" for g, n in sorted(counts.get(h, {}).items()))
                    out.append(f"  | `{h[:12]}` | {diff} | {div} | {k['metadata'].get('num_warps')} | {k.get('n_regs')} | "
                               f"{k.get('n_spills')} | {m.get('dynamic_shared_bytes', k['metadata'].get('shared'))} | {gtxt} | {lc} |")
                out.append("")
            else:
                out.append(f"### extension `{name}`\n")
                out.append(f"- calls: {per or 'none'}")
                variants = ext_detail.get(name, Counter())
                if variants:
                    out.append(f"- argument variants seen in detail phases ({len(variants)}; top 12 by count):")
                    for sig, n in variants.most_common(12):
                        out.append(f"  - {n}x `{sig[:400]}`")
                out.append("")
            sr = Counter()
            for h in hashes or [name]:
                sr.update(site_roles.get(h, {}))
            if role == "dense bf16 matmul" or len({r.split(" (")[0] for r in sr if r != "other"}) > 1:
                out.append("- by site role (detail-phase launches): " + ", ".join(f"{r} {n}" for r, n in sr.most_common()))
                if hashes and len(hashes) > 1:
                    for h in sorted(hashes):
                        if site_roles.get(h):
                            out.append(f"  - `{h[:12]}`: " + ", ".join(f"{r} {n}" for r, n in site_roles[h].most_common()))
                out.append("")
            if sites:
                out.append("  sites: " + " | ".join(f"`{s}`" for s in sites[:6]) + (f" (+{len(sites) - 6})" if len(sites) > 6 else ""))
                out.append("")
    g = run / "graphs.json"
    if g.exists():
        gj = json.loads(g.read_text())
        out.append("## CUDA graphs\n")
        out.append(f"{len(gj['captured'])} captures: " + ", ".join(f"`{c['key']}` ({c['phase']})" for c in gj["captured"]))
        out.append("")
        tot = Counter()
        for phase, keys in gj["replays"].items():
            for k, n in keys.items():
                tot[(phase_group(phase), k)] += n
        out.append("| phase group | graph | replays |\n| --- | --- | ---: |")
        for (p, k), n in sorted(tot.items()):
            out.append(f"| {p} | {k} | {n} |")
        out.append("")
    prof = run / "profile"
    if prof.exists():
        out.append("## Every CUDA kernel the profiler saw (torch's own ops included)\n")
        triton_names = {info["name"] for info in rec["kernels"].values()}
        for f in sorted(prof.glob("profile-*.json")):
            rows = json.loads(f.read_text())
            dev = [r for r in rows if r.get("device_us") and not r["name"].startswith(("tf::", "aten::", "cuda", "Memcpy",
                                                                                        "Memset", "ProfilerStep"))]
            torchy = [r for r in dev if r["name"] not in triton_names and ("at::" in r["name"] or "native" in r["name"]
                                                                          or "cub" in r["name"] or "elementwise" in r["name"]
                                                                          or "reduce" in r["name"].lower())]
            out.append(f"### {f.stem}: {len(dev)} kernels, torch's own: {len(torchy)}\n")
            for r in sorted(torchy, key=lambda r: -r["device_us"])[:40]:
                out.append(f"- {r['count']}x {r['device_us']:.0f} us `{r['name'][:180]}`")
            out.append("")
            aten = [r for r in rows if r["name"].startswith("aten::")]
            out.append("  aten ops: " + ", ".join(f"{r['name']} {r['count']}" for r in sorted(aten, key=lambda r: -r["count"])[:40]))
            out.append("")
    text = "\n".join(out) + "\n"
    Path(a.out or run / "kernels.md").write_text(text)
    print(f"{len(fns)} functions, {total_t} specializations, {len(ext_keys)} extension functions -> {a.out or run / 'kernels.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
