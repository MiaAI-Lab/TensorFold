"""Fixtures for the Zig vision tower (zig/src/families/glm_cuda/cuda_vision.zig): the Python reference tower's
inputs and every stage, as raw little-endian .bin files plus manifest.json.

Run in the reference image (needs the GPU; the tower is the one the engine ships):
  docker run --rm --gpus all --entrypoint python3 -v $HOME/.cache/huggingface:/hf:ro -v <this dir>:/fx \
      -v <out dir>:/out tensorfold-glm53:r107ref /fx/gen.py /out [--model DIR] [--big] [--only NAME]

Stages per case (pass p0 first, then p1 ...): canvas (uint8 input), patches (fp32), embed (after conv3d), block00..,
post_norm, downsample, merger_ln (layer_norm), merger_gelu, features (bf16 [rows, 4096]).  Block 0 also records its
inner steps (b0_*), so a mismatch can be placed inside the block.
"""
import json, sys, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tensorfold.vision import glm as G

STAGE_FILTER = None   # --stages a,b,c: record only these stages (debugging)
INNER_ALL = False   # --inner-all: every block records its inner steps too (debugging, not committed)
MODEL = "/hf/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-4bpw-TensorFold/snapshots/078455ffe6472f9a52fbc1139f58b9db2881b25c"


def canvas(h, w, seed, frames=None):
    """Deterministic uint8 pictures: smooth gradients, a few edges and noise (no RNG library state involved)."""
    rng = np.random.default_rng(seed)
    n = 1 if frames is None else frames
    out = np.empty((n, 3, h, w), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    for f in range(n):
        for c in range(3):
            base = 127 + 100 * np.sin((xx * (0.05 + 0.02 * c) + f * 0.7) + yy * 0.03 * (c + 1))
            edge = ((xx // 17 + yy // 23 + c + f) % 3 == 0) * 40.0
            noise = rng.integers(-12, 13, (h, w))
            out[f, c] = np.clip(base + edge + noise, 0, 255).astype(np.uint8)
    return out[0] if frames is None else out


CASES = {
    "img56x84": dict(kind="image", h=56, w=84, seed=1, all_blocks=True),
    "img112x140": dict(kind="image", h=112, w=140, seed=2),
    "img196x196": dict(kind="image", h=196, w=196, seed=3),     # 196 patches: two key blocks of 128
    "clip4x56x56": dict(kind="video", h=56, w=56, frames=4, seed=4),
    "clip8x56x56_pass32": dict(kind="video", h=56, w=56, frames=8, seed=5, pass_patches=32),
}
# more shapes, only their final features are kept (--extra --stages features): other row counts make cuDNN and cuBLAS pick
# other kernels, other sequence lengths use other numbers of key blocks
EXTRA = {
    "x_img56x56": dict(kind="image", h=56, w=56, seed=11),
    "x_img84x84": dict(kind="image", h=84, w=84, seed=12),
    "x_img140x224": dict(kind="image", h=140, w=224, seed=13),
    "x_img280x196": dict(kind="image", h=280, w=196, seed=14),
    "x_img448x448": dict(kind="image", h=448, w=448, seed=15),
    "x_img224x1344": dict(kind="image", h=224, w=1344, seed=16),
    "x_img28x1400": dict(kind="image", h=28, w=1400, seed=17),
    "x_img1008x56": dict(kind="image", h=1008, w=56, seed=18),
    "x_clip2x56x84": dict(kind="video", h=56, w=84, frames=2, seed=19),
    "x_clip6x84x84": dict(kind="video", h=84, w=84, frames=6, seed=20),
    "x_clip16x56x56": dict(kind="video", h=56, w=56, frames=16, seed=21),
    "x_clip8x196x196": dict(kind="video", h=196, w=196, frames=8, seed=22),
    "x_clip40x84x84": dict(kind="video", h=84, w=84, frames=40, seed=23, pass_patches=288),
}
BIG = {"big8192": dict(kind="image", h=896, w=1792, seed=9, big=True)}   # 64 x 128 = 8192 patches


def save(dirp, name, t):
    a = t.detach().cpu().contiguous()
    raw = a.view(torch.uint8).numpy().tobytes() if a.dtype != torch.bfloat16 else a.view(torch.int16).numpy().tobytes()
    (dirp / f"{name}.bin").write_bytes(raw)
    return dict(name=name, dtype=str(a.dtype).replace("torch.", ""), shape=list(a.shape), sha256=hashlib.sha256(raw).hexdigest()[:16])


class Rec:
    def __init__(self, dirp, prefix, all_blocks, keep):
        self.d, self.p, self.all, self.keep, self.stages, self.note = dirp, prefix, all_blocks, keep, [], {}

    def put(self, name, t):
        if STAGE_FILTER is not None and name not in STAGE_FILTER:
            return
        if self.keep:
            self.stages.append(save(self.d, f"{self.p}{name}", t))


def staged_encode(tw, rec, patches, groups, gh, gw, all_blocks, c_big=False):
    """GlmVisionTower.encode, step for step, recording the stages."""
    t = tw.t
    S = gh * gw
    rec.put("patches", patches)
    x = F.conv3d(patches.to(torch.bfloat16).view(-1, 3, tw.temporal, tw.patch, tw.patch), t["patch_embed.proj.weight"],
                 t["patch_embed.proj.bias"], stride=(tw.temporal, tw.patch, tw.patch))
    if rec.keep:
        nb = F.conv3d(patches.to(torch.bfloat16).view(-1, 3, tw.temporal, tw.patch, tw.patch), t["patch_embed.proj.weight"], None,
                      stride=(tw.temporal, tw.patch, tw.patch))
        rec.put("embed_nobias", nb)
        rec.note["embed_is_nobias_plus_bias_bf16"] = bool(torch.equal(nb.view(x.shape) + t["patch_embed.proj.bias"].view(1, -1, 1, 1, 1), x))
    x = x.view(groups, S, tw.hidden)
    rec.put("embed", x)
    cos, sin = tw._rope(gh, gw)
    if rec.keep:
        rec.put("rope_cos", cos.float())
        rec.put("rope_sin", sin.float())
    for i in range(tw.depth):
        if (i == 0 or INNER_ALL) and rec.keep:
            x = block0(tw, rec, x, cos, sin, i)
        else:
            x = tw._block(i, x, cos, sin)
        if rec.keep and (all_blocks or i in (0, 12, 23) or c_big):
            rec.put(f"block{i:02d}", x)
    x = tw._rms(x, t["post_layernorm.weight"])
    rec.put("post_norm", x)
    x = x.reshape(-1, tw.merge, tw.merge, tw.hidden)
    if rec.keep:
        nb = F.conv2d(x.permute(0, 3, 1, 2), t["downsample.weight"], None, stride=tw.merge).view(-1, tw.out)
        rec.put("downsample_nobias", nb)
    x = F.conv2d(x.permute(0, 3, 1, 2), t["downsample.weight"], t["downsample.bias"], stride=tw.merge).view(-1, tw.out)
    rec.put("downsample", x)
    if rec.keep:
        rec.note["downsample_is_nobias_plus_bias_bf16"] = bool(torch.equal(nb + t["downsample.bias"], x))
    x = F.linear(x, t["merger.proj.weight"])
    rec.put("merger_proj", x)
    if rec.keep:   # the kernel's own fp32 statistics, to pin down its Welford order
        _, mu, rs = torch.native_layer_norm(x, (tw.out,), t["merger.post_projection_norm.weight"],
                                            t["merger.post_projection_norm.bias"], 1e-5)
        rec.put("merger_ln_mean", mu)
        rec.put("merger_ln_rstd", rs)
    x = F.layer_norm(x, (tw.out,), t["merger.post_projection_norm.weight"], t["merger.post_projection_norm.bias"])
    rec.put("merger_ln", x)
    x = F.gelu(x)
    rec.put("merger_gelu", x)
    gate = F.linear(x, t["merger.gate_proj.weight"]).clamp(max=tw.limit)
    up = F.linear(x, t["merger.up_proj.weight"]).clamp(-tw.limit, tw.limit)
    out = F.linear(F.silu(gate) * up, t["merger.down_proj.weight"])
    rec.put("features", out)
    return out


def block0(tw, rec, x, cos, sin, i=0):
    t, L = tw.t, tw.limit
    g = f"blocks.{i}."
    pre = f"b{i}_"
    B, S, D = x.shape
    h = tw._rms(x, t[g + "norm1.weight"])
    rec.put(pre + "norm1", h)
    qkv = F.linear(h, t[g + "attn.qkv.weight"], t[g + "attn.qkv.bias"])
    rec.put(pre + "qkv", qkv)
    q, k, v = qkv.view(B, S, 3, tw.heads, tw.head_dim).unbind(2)
    q = tw._rms(q, t[g + "attn.q_norm.weight"]); k = tw._rms(k, t[g + "attn.k_norm.weight"])
    rec.put(pre + "qn", q); rec.put(pre + "kn", k)
    q = tw._rotate(q, cos, sin); k = tw._rotate(k, cos, sin)
    rec.put(pre + "qr", q); rec.put(pre + "kr", k)
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=tw.head_dim ** -0.5)
    o = o.transpose(1, 2).reshape(B, S, D)
    rec.put(pre + "attn", o)
    pr = F.linear(o, t[g + "attn.proj.weight"], t[g + "attn.proj.bias"])
    rec.put(pre + "proj", pr)
    x = x + pr
    rec.put(pre + "res1", x)
    h = tw._rms(x, t[g + "norm2.weight"])
    rec.put(pre + "norm2", h)
    gate = F.linear(h, t[g + "mlp.gate_proj.weight"], t[g + "mlp.gate_proj.bias"])
    up = F.linear(h, t[g + "mlp.up_proj.weight"], t[g + "mlp.up_proj.bias"])
    rec.put(pre + "gate_raw", gate); rec.put(pre + "up_raw", up)
    gate = gate.clamp(max=L); up = up.clamp(-L, L)
    act = F.silu(gate) * up
    rec.put(pre + "act", act)
    return x + F.linear(act, t[g + "mlp.down_proj.weight"], t[g + "mlp.down_proj.bias"])


def run_case(tw, name, c, outdir, dev):
    d = outdir / name
    d.mkdir(parents=True, exist_ok=True)
    keep = True
    pp = c.get("pass_patches", G.PASS_PATCHES)
    if c["kind"] == "image":
        cv = canvas(c["h"], c["w"], c["seed"])
        frames = torch.from_numpy(cv).to(dev)
        frames = frames.unsqueeze(0).expand(2, -1, -1, -1)
        item = ("image", torch.from_numpy(cv).to(dev))
    else:
        cv = canvas(c["h"], c["w"], c["seed"], c["frames"])
        frames = torch.from_numpy(cv).to(dev)
        item = ("video", frames)
    man = dict(name=name, kind=c["kind"], pass_patches=pp, h=c["h"], w=c["w"], frames=c.get("frames", 2),
               canvas_shape=list(cv.shape), stages=[])
    if keep:
        (d / "canvas.bin").write_bytes(cv.tobytes())
    all_blocks = c.get("all_blocks", False) or INNER_ALL
    T, _, H, W = frames.shape
    gh, gw = H // 14, W // 14
    per = max(1, pp // (gh * gw))
    outs = []
    for pi, start in enumerate(range(0, T // 2, per)):
        pairs = frames[2 * start:2 * min(T // 2, start + per)]
        rec = Rec(d, f"p{pi}_", all_blocks, keep)
        with torch.no_grad():
            x = staged_encode(tw, rec, tw.patches(pairs), pairs.shape[0] // 2, gh, gw, all_blocks, False)
        outs.append(x)
        man["stages"] += rec.stages
        man.update(rec.note)
    # the engine's own entry point must give the same bytes as the staged copy
    pp_old = G.PASS_PATCHES
    G.PASS_PATCHES = pp
    with torch.no_grad():
        ref = tw.features([item])
    G.PASS_PATCHES = pp_old
    cat = torch.cat(outs)
    man["staged_equals_features"] = bool(torch.equal(cat, ref))
    man["rows"] = int(ref.shape[0])
    if keep:
        man["stages"].append(save(d, "features_all", ref))
        (d / "manifest.json").write_text(json.dumps(man, indent=1))
    torch.cuda.synchronize()
    if c.get("big"):
        import time
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.time()
        for _ in range(3):
            with torch.no_grad():
                tw.features([item])
        torch.cuda.synchronize()
        man["python_ms_per_run"] = (time.time() - t0) / 3 * 1000
        man["python_peak_extra_MiB"] = (torch.cuda.max_memory_allocated() - base) / 2**20
        print("big timing", man["python_ms_per_run"], "ms; extra peak MiB", man["python_peak_extra_MiB"], flush=True)
        (d / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def main():
    outdir = Path(sys.argv[1])
    args = sys.argv[2:]
    model = Path(args[args.index("--model") + 1]) if "--model" in args else Path(MODEL)
    only = args[args.index("--only") + 1] if "--only" in args else None
    global INNER_ALL, STAGE_FILTER
    INNER_ALL = "--inner-all" in args
    if "--stages" in args:
        STAGE_FILTER = set(args[args.index("--stages") + 1].split(","))
    cases = dict(BIG) if "--big" in args else dict(EXTRA) if "--extra" in args else dict(CASES)
    dev = torch.device("cuda")
    tw = G.GlmVisionTower(model, dev)
    for name, c in cases.items():
        if only and name != only:
            continue
        m = run_case(tw, name, c, outdir, dev)
        print(name, "rows", m["rows"], "staged==features", m["staged_equals_features"], flush=True)
    print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(),
          torch.cuda.get_device_name())


if __name__ == "__main__":
    main()
