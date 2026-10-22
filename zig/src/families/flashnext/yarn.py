"""YaRN on Flash Next's 64 rotary dims: Qwen's recipe past the native 262,144 tokens (rope_type yarn, factor 4).

The semantics are Transformers' ``_compute_yarn_parameters``: each rotary pair's frequency blends the native one
(extrapolation) and the native one over ``factor`` (interpolation) along a linear ramp over the pairs, from the
correction dim of ``beta_fast`` (32) rotations to that of ``beta_slow`` (1) rotation in
``original_max_position_embeddings`` positions (floor and ceil with ``truncate``, the default); and the cos/sin of
the rotary dims are multiplied by the attention factor, 0.1 ln(factor) + 1 unless the config gives one.

Here the frequencies are computed in fp64 and stored fp32, like the default ones (``weights.load``), and the
attention factor is folded into the fp32 gammas of the four RMSNorms whose outputs are rotated (q_norm, k_norm and
the indexer's q_layernorm and k_layernorm, in every attention layer and the MTP layer): RoPE is linear, so scaling a
normalized head's rotary dims before the rotation is scaling cos and sin. The main attention, the QSA indexer (its
queries and its pooled keys) and the MTP layer share the one rotary embedding, as in Transformers and vLLM.

Off unless the config's ``rope_parameters`` say yarn or ``TF_FLASHNEXT_YARN`` gives a factor (``0``/``off`` turns a
config's yarn off). The Zig engine computes the same bits (zig/src/families/flashnext/cuda_rope.zig).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

ENV = "TF_FLASHNEXT_YARN"                         # a factor > 1 (4.0: 1,048,576 tokens), or 0 / off
ENV_RAMP = "TF_FLASHNEXT_YARN_RAMP_POSITIONS"     # the ramp's positions (default original_max_position_embeddings;
                                                  # vLLM's MRotaryEmbedding passes 4x that)
TEXT_TYPES = ("qwen4_exp", "qwen4_exp_text")
OFF = ("0", "off", "no", "false", "none", "default")


def _mscale(scale: float, mscale: float = 1.0) -> float:
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


@dataclass(frozen=True)
class Yarn:
    factor: float
    original: int                    # original_max_position_embeddings: the trained window
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    attention: float = 0.0           # the attention factor (0: derived from factor)
    truncate: bool = True
    ramp_positions: int = 0          # the positions the correction dims count rotations in (0: original)

    @property
    def window(self) -> int:
        return int(self.factor * self.original)

    @property
    def scale(self) -> float:
        """The attention factor multiplying cos and sin of the rotary dims (fp64)."""

        return float(self.attention) if self.attention else _mscale(self.factor)

    def bounds(self, rotary_dim: int, theta: float) -> tuple[float, float]:
        """The ramp's first and last rotary pair (Transformers' find_correction_range)."""

        positions = self.ramp_positions or self.original

        def dim(rotations: float) -> float:
            return rotary_dim * math.log(positions / (rotations * 2 * math.pi)) / (2 * math.log(theta))

        low, high = dim(self.beta_fast), dim(self.beta_slow)
        if self.truncate:
            low, high = math.floor(low), math.ceil(high)
        return max(low, 0), min(high, rotary_dim - 1)


def read(text: dict) -> Yarn | None:
    """The YaRN settings of a text config (``rope_parameters``), with ``TF_FLASHNEXT_YARN`` over them."""

    rope = dict(text.get("rope_parameters") or {})
    native = int(text.get("max_position_embeddings") or 0)
    env = os.environ.get(ENV, "").strip().lower()
    if env in OFF:
        return None
    if env:
        try:
            factor = float(env)
        except ValueError:
            raise ValueError(f"{ENV}={env!r}: a YaRN factor above 1 (4.0 for 1,048,576 tokens), or off") from None
        rope = {**rope, "rope_type": "yarn", "factor": factor}
    if str(rope.get("rope_type", rope.get("type", "default"))) != "yarn":
        return None
    original = int(rope.get("original_max_position_embeddings") or native)
    factor = rope.get("factor")
    factor = float(factor) if factor is not None else (native / original if original else 0.0)
    if not factor > 1.0 or original <= 0:
        raise ValueError(f"YaRN needs a factor above 1 and original_max_position_embeddings; got factor {factor}, "
                         f"original {original}")
    attention = rope.get("attention_factor")
    mscale, mscale_all = rope.get("mscale"), rope.get("mscale_all_dim")
    if attention is None and mscale and mscale_all:
        attention = _mscale(factor, float(mscale)) / _mscale(factor, float(mscale_all))
    ramp = os.environ.get(ENV_RAMP, "").strip()
    return Yarn(factor=factor, original=original, beta_fast=float(rope.get("beta_fast") or 32),
                beta_slow=float(rope.get("beta_slow") or 1), attention=float(attention or 0.0),
                truncate=bool(rope.get("truncate", True)), ramp_positions=int(ramp) if ramp else 0)


def window(text: dict, limit: int) -> int:
    """A Flash Next config's window: ``factor * original`` under YaRN, else ``limit`` (max_position_embeddings)."""

    if str(text.get("model_type", "")) not in TEXT_TYPES:
        return limit
    yarn = read(text)
    return max(int(limit or 0), yarn.window) if yarn is not None else limit


def inv_freq(theta: float, rotary_dim: int, yarn: Yarn | None):
    """The fp32 frequencies of the rotary pairs, computed in fp64 (the default ones are weights.load's bits)."""

    import torch

    half = rotary_dim // 2
    native = torch.tensor(theta, dtype=torch.float64) ** (-torch.arange(0, half, dtype=torch.float64) / half)
    if yarn is None:
        return native.to(torch.float32)
    low, high = yarn.bounds(rotary_dim, theta)
    if low == high:
        high += 0.001
    ramp = ((torch.arange(half, dtype=torch.float64) - low) / (high - low)).clamp(0, 1)
    keep = 1 - ramp                                   # 1: the native frequency, 0: the native one over factor
    return ((native / yarn.factor) * (1 - keep) + native * keep).to(torch.float32)


def scale32(yarn: Yarn | None) -> float:
    """The attention factor rounded to fp32, as it multiplies the fp32 gammas."""

    import numpy as np

    return 1.0 if yarn is None else float(np.float32(yarn.scale))


def fold(gamma, rotary_dim: int, scale: float) -> None:
    """gamma[:rotary_dim] = fp32(gamma * scale) in place: one IEEE fp32 multiply (the fp64 product is exact)."""

    import torch

    head = gamma[:rotary_dim]
    head.copy_((head.to(torch.float64) * float(scale)).to(torch.float32))


def apply(w):
    """YaRN on loaded weights: the frequencies and the attention factor folded into the rotated norms' gammas."""

    cfg = w.cfg
    yarn = getattr(cfg, "yarn", None)
    if yarn is None:
        return w
    import torch

    if w.inv_freq.dtype != torch.float32 or w.inv_freq.numel() != cfg.rotary_dim // 2:
        raise ValueError("YaRN expects the fp32 rotary frequencies of the loaded weights")
    w.inv_freq = inv_freq(cfg.rope_theta, cfg.rotary_dim, yarn).to(w.inv_freq.device)
    m = scale32(yarn)
    layers = [layer for layer in w.layers if layer.attn is not None]
    if w.mtp is not None and w.mtp.layer.attn is not None:
        layers.append(w.mtp.layer)
    seen: set[int] = set()
    for layer in layers:
        a = layer.attn
        for gamma in (a.q_scale, a.k_scale, a.iq_scale, a.ik_scale):
            if gamma.dtype != torch.float32 or gamma.numel() < cfg.rotary_dim:
                raise ValueError("YaRN folds its attention factor into fp32 norm gammas of the rotary dims")
            if gamma.data_ptr() in seen:              # a gamma shared between heads is folded once
                continue
            seen.add(gamma.data_ptr())
            fold(gamma, cfg.rotary_dim, m)
    low, high = yarn.bounds(cfg.rotary_dim, cfg.rope_theta)
    w.meta["yarn"] = dict(factor=yarn.factor, original=yarn.original, window=yarn.window, attention=m,
                          ramp=(low, high), ramp_positions=yarn.ramp_positions or yarn.original)
    print(f"[tensorfold] Flash Next YaRN: factor {yarn.factor:g} over {yarn.original} positions "
          f"({yarn.window}-token window), ramp pairs {low}..{high}, attention factor {m:.9g} folded into the "
          f"q/k and indexer norms of {len(layers)} attention layers", flush=True)
    return w
