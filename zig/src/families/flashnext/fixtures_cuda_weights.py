"""Digests of the Python engine's weight layouts on small deterministic inputs, for cuda_weights.zig's host tests.

Run in the pytorch image with the TensorFold source (src/tensorfold) and a GPU (the float steps run where Python runs
them, on CUDA):  python -I fixtures_cuda_weights.py <src/tensorfold> <out.json> [<checkpoint dir>]

Only the layout modules are loaded, by path, under stub parent packages (no package __init__ runs): bf16.py, nvfp4.py,
qmm.py (qwen4_exp/cuda), cuda/kernels/qmm.py, cuda/nvfp4/{experts,linear,format}.py, cuda/experts.py, all authored by
Ash Hart (ashhart). The FP8 n-gram LUT and the FP8_PB_WO dequantization below restate host_table.FP8Table (tournierjc,
ashhart) and weights.weight_bf16 (Jürgen Schmied) line for line, so the digests are those functions' outputs.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import struct
import sys
import types
from pathlib import Path

import numpy as np
import torch

SRC = Path(sys.argv[1])
for name, sub in (("tensorfold", ""), ("tensorfold.cuda", "cuda"), ("tensorfold.cuda.kernels", "cuda/kernels"),
                  ("tensorfold.cuda.nvfp4", "cuda/nvfp4"), ("tensorfold.families", "families"),
                  ("tensorfold.families.qwen4_exp", "families/qwen4_exp"),
                  ("tensorfold.families.qwen4_exp.cuda", "families/qwen4_exp/cuda")):
    m = types.ModuleType(name)
    m.__path__ = [str(SRC / sub)]
    sys.modules[name] = m

bf16 = importlib.import_module("tensorfold.families.qwen4_exp.cuda.bf16")
nvfp4 = importlib.import_module("tensorfold.families.qwen4_exp.cuda.nvfp4")
nvx = importlib.import_module("tensorfold.cuda.nvfp4.experts")
linear = importlib.import_module("tensorfold.cuda.nvfp4.linear")
fmt = importlib.import_module("tensorfold.cuda.nvfp4.format")

DEV = "cuda" if torch.cuda.is_available() else "cpu"
U = np.uint64


def rnd(seed: int, n: int) -> np.ndarray:
    """splitmix64 of seed * 2**32 + i (cuda_weights.zig `rnd`)."""

    with np.errstate(over="ignore"):
        x = np.arange(n, dtype=U) + U(seed) * U(1 << 32) + U(0x9E3779B97F4A7C15)
        x = (x ^ (x >> U(30))) * U(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> U(27))) * U(0x94D049BB133111EB)
        return x ^ (x >> U(31))


def bf16_bits(seed: int, n: int) -> np.ndarray:
    r = rnd(seed, n)
    return (((r >> U(63)) << U(15)) | ((U(112) + (r >> U(40)) % U(16)) << U(7)) | ((r >> U(8)) & U(0x7F))).astype(np.uint16)


def u8(seed: int, n: int) -> np.ndarray:
    return (rnd(seed, n) & U(0xFF)).astype(np.uint8)


def t16(a: np.ndarray, shape) -> torch.Tensor:
    return torch.from_numpy(a.view(np.int16).reshape(shape).copy()).view(torch.bfloat16).to(DEV)


def sha(t) -> str:
    raw = t.detach().contiguous()
    return hashlib.sha256(raw.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


out: dict[str, object] = {"device": DEV, "torch": torch.__version__}

# A. routed experts: nvx.make (pack), one GPU and rank 1 of 2 (weights.moe_nvfp4's slices)
E, NI, D = 3, 64, 96
def fp4(seed, n, k):
    w = torch.from_numpy(u8(seed, E * n * k // 2).reshape(E, n, k // 2)).to(DEV)
    s = torch.from_numpy(u8(seed + 1, E * n * k // 16).reshape(E, n, k // 16)).view(torch.float8_e4m3fn).to(DEV)
    s2 = torch.from_numpy(((rnd(seed + 2, E) % U(1000) + U(1)).astype(np.float32) / np.float32(1024))).to(DEV)
    return w, s, s2
gate, up, down = fp4(1, NI, D), fp4(4, NI, D), fp4(7, D, NI)
for tag, (g, u, d) in {"w1": (gate, up, down),
                       "w2r1": ((gate[0][:, 32:64], gate[1][:, 32:64], gate[2]), (up[0][:, 32:64], up[1][:, 32:64], up[2]),
                                (down[0][:, :, 16:32], down[1][:, :, 2:4], down[2]))}.items():
    ex = nvx.make(g, u, d)
    out[f"experts.{tag}"] = {k: sha(getattr(ex, k)) for k in ("up", "down", "up_scale", "down_scale")}

# B. the shared expert as identity-scaled FP4 tables (nvfp4_moe.expert4_from_bf16)
NI, D = 128, 128
sg, su, sd = t16(bf16_bits(10, NI * D), (NI, D)), t16(bf16_bits(11, NI * D), (NI, D)), t16(bf16_bits(12, D * NI), (D, NI))
for tag, (g, u, d) in {"w1": (sg, su, sd), "w2r1": (sg[64:128], su[64:128], sd[:, 64:128])}.items():
    gu = nvfp4.fp4_from_bf16(torch.cat([g, u], dim=0).contiguous())
    dn = nvfp4.fp4_from_bf16(d.contiguous())
    out[f"shared.{tag}"] = {"gu.weight": sha(gu.weight), "gu.scale": sha(gu.scale), "gu.scale2": sha(gu.scale2),
                            "down.weight": sha(dn.weight), "down.scale": sha(dn.scale), "down.scale2": sha(dn.scale2)}

# C. the draft head: bf16.quantize4 (MLX 4-bit, groups of 32) then qmm.make_q4's frag pack
N, K = 200, 128
w = bf16_bits(13, N * K).reshape(N, K)
w[7] = w[7, 0]                                   # a constant row: max == min, the 1e-8 floor
w[8, ::2] = w[8, 0]                              # two values a group: codes 0 and 15 only
q = bf16.quantize4(t16(w, (N, K)))
out["quantize4"] = {"weight": sha(q.weight), "scales": sha(q.scales), "biases": sha(q.biases),
                    "shape": list(q.weight.shape)}

# D. nvx.quantize (ModelOpt's recipe, the MTP experts at load)
E, N, K = 3, 64, 64
x = bf16_bits(14, E * N * K).reshape(E, N, K)
x[2] = 0                                         # a zero expert: the 1e-30 floor
x[1, 0, :16] = 0                                 # a zero block
words, scales, g = nvx.quantize(t16(x, (E, N, K)))
out["nvx_quantize"] = {"words": sha(words), "scales": sha(scales), "g": sha(g)}

# E. FP8_PB_WO dequantization (weights.weight_bf16 with 128x128-block weight_scale_inv)
N, K = 256, 256
codes = u8(15, N * K)
codes[(codes & 0x7F) == 0x7F] -= 1               # no NaN codes in a weight
codes_t = torch.from_numpy(codes.reshape(N, K)).view(torch.float8_e4m3fn).to(DEV)
inv = (bf16_bits(16, 4) & np.uint16(0x7FFF)).reshape(2, 2)
inv_t = t16(inv, (2, 2))
cols = linear.Fp8BlockLinear.column_scales(inv_t, N, K)
deq = torch.empty((N, K), dtype=torch.bfloat16, device=codes_t.device)
for r in range(0, N, 16384):
    blk = codes_t[r:r + 16384].float().view(-1, K // 64, 64) * cols[r:r + 16384, :, None]
    deq[r:r + 16384] = blk.view(-1, K).to(torch.bfloat16)
out["fp8_block"] = {"bf16": sha(deq)}
# the chain the MTP experts take: dequantized rows of two experts, stacked, quantized, packed
both = torch.stack([deq[:64, :], deq[64:128, :]])
qw, qs, qg = nvx.quantize(both)
ex = nvx.make((qw, qs, qg), (qw, qs, qg), nvx.quantize(both.transpose(1, 2).contiguous()))
out["fp8_chain"] = {k: sha(getattr(ex, k)) for k in ("up", "down", "up_scale", "down_scale")}

# F. the FP8 n-gram table's LUT (host_table.FP8Table): bf16_rne(e4m3 x scale), NaN codes 0x7FC0
def lut(scale: float) -> np.ndarray:
    f32 = (fmt.e4m3(np.arange(256)) * np.float32(scale)).astype(np.float32).view(np.uint32).astype(np.uint64)
    t = ((f32 + 0x7FFF + ((f32 >> 16) & 1)) >> 16).astype(np.uint16)
    t[(np.arange(256) & 0x7F) == 0x7F] = 0x7FC0
    return t
scales = {"3c23": float(torch.tensor([0x3C23], dtype=torch.int16).view(torch.bfloat16).float())}
if len(sys.argv) > 3:                            # the checkpoint's own table scale
    ck = Path(sys.argv[3])
    where = json.loads((ck / "model.safetensors.index.json").read_text())["weight_map"]
    name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
    with open(ck / where[name], "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        f.seek(8 + n + h[name]["data_offsets"][0])
        bits = struct.unpack("<H", f.read(2))[0]
    scales[f"{bits:04x}"] = float(torch.tensor([bits], dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float())
out["lut"] = {k: lut(v).tobytes().hex() for k, v in scales.items()}

# G. inv_freq (weights.load): theta ** (-arange(half) / half) in fp64 on the CPU, stored fp32
inv = torch.tensor(1e7, dtype=torch.float64) ** (-torch.arange(0, 32, dtype=torch.float64) / 32)
out["inv_freq"] = inv.to(torch.float32).numpy().view(np.uint32).tolist()

Path(sys.argv[2]).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "lut"})[:2000])
