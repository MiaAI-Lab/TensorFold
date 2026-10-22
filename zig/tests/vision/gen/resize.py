import hashlib, random, sys
import numpy as np, torch
sys.path.insert(0, "/gen")
from common import lcg_bytes
from torchvision.transforms.v2 import functional as tvF

rnd = random.Random(99)
cases = []
def add(h, w, oh, ow):
    if (h, w) != (oh, ow) or True:
        cases.append((h, w, oh, ow))
small = [1, 2, 3, 4, 5, 7, 13, 14, 27, 28, 29, 56, 57, 100]
for _ in range(180):
    add(rnd.choice(small), rnd.choice(small), rnd.choice(small), rnd.choice(small))
for _ in range(120):
    add(rnd.randint(1, 300), rnd.randint(1, 300), rnd.randint(1, 400), rnd.randint(1, 400))
for (h, w, oh, ow) in [(1080, 1920, 392, 700), (1080, 1920, 1080, 700), (1080, 1920, 392, 1920), (720, 1280, 336, 588), (2000, 3000, 560, 840),
                       (8192, 64, 1400, 28), (64, 8192, 28, 1400), (4000, 5, 1000, 28), (5, 4000, 28, 1000), (1, 4096, 28, 2048), (4096, 1, 2048, 28),
                       (100, 100, 2000, 2000), (50, 70, 1400, 1960), (3, 3, 56, 56), (480, 640, 480, 224), (480, 640, 168, 640), (7680, 4320, 504, 896)]:
    add(h, w, oh, ow)
out = []
for i, (h, w, oh, ow) in enumerate(cases):
    seed = 5000 + i
    raw = lcg_bytes(seed, h * w * 3)
    t = torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()).permute(2, 0, 1)
    if (h, w) == (oh, ow):
        r = t
    else:
        r = tvF.resize(t, [oh, ow], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    arr = r.permute(1, 2, 0).contiguous().numpy()
    out.append(f"{seed} {h} {w} {oh} {ow} {hashlib.sha256(arr.tobytes()).hexdigest()}")
open("/fix/resize.txt", "w").write("\n".join(out) + "\n")
print(len(out))
