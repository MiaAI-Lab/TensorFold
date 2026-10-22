import sys
sys.path.insert(0, "/gen")
from imgcommon import *

rng = np.random.default_rng(7)
def content(kind, w, h, alpha=False):
    if kind == "noise": a = noise(w, h, 4 if alpha else 3, w * 7 + h)
    elif kind == "grad": a = gradient(w, h, 4 if alpha else 3, 3)
    elif kind == "text": a = texty(w, h, 2)
    elif kind == "blocks": a = blocks(w, h, 5)
    elif kind == "pal2":
        a = (rng.integers(0, 2, (h, w, 1)) * np.array([[[250, 10, 30]]]) + (1 - rng.integers(0, 2, (h, w, 1))) * np.array([[[10, 200, 60]]])).astype(np.uint8)
    elif kind == "pal5":
        pal = rng.integers(0, 256, (5, 3)); a = pal[rng.integers(0, 5, (h, w))].astype(np.uint8)
    elif kind == "pal12":
        pal = rng.integers(0, 256, (12, 3)); a = pal[rng.integers(0, 12, (h, w))].astype(np.uint8)
    elif kind == "pal100":
        pal = rng.integers(0, 256, (100, 3)); a = pal[rng.integers(0, 100, (h, w))].astype(np.uint8)
    elif kind == "flat": a = np.full((h, w, 3), 77, np.uint8)
    elif kind == "repeat":
        tile = rng.integers(0, 256, (4, 4, 3)); a = np.tile(tile, (h // 4 + 1, w // 4 + 1, 1))[:h, :w].astype(np.uint8)
    if alpha and a.shape[2] == 3:
        al = (gradient(w, h, 1, 1)[..., 0:1]).astype(np.uint8)
        al = np.where(((np.arange(h)[:, None] + np.arange(w)[None, :]) % 7 == 0)[..., None], 0, al)
        a = np.concatenate([a, al], axis=2)
    return a

