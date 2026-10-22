import hashlib, sys, torch
sys.path.insert(0, "/gen")
from common import lcg_bytes
from tensorfold.vision.glm import MEAN, STD

mean = torch.tensor(MEAN).view(3, 1, 1) * 255
std = torch.tensor(STD).view(3, 1, 1) * 255

def patches(frames, patch=14, merge=2):
    T, C, H, W = frames.shape
    p, m = patch, merge
    x = (frames.float() - mean) / std
    x = x.view(T // 2, 2, C, H // p // m, m, p, W // p // m, m, p)
    x = x.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return x.reshape(T // 2 * (H // p) * (W // p), C * 2 * p * p)

out = []
for seed, (T, H, W) in enumerate([(2, 28, 28), (2, 28, 56), (4, 56, 28), (2, 84, 112), (6, 56, 56), (2, 140, 28)]):
    raw = lcg_bytes(seed + 100, T * 3 * H * W)
    f = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(T, 3, H, W)
    r = patches(f).contiguous()
    out.append(f"patches {seed + 100} {T} {H} {W} {tuple(r.shape)[0]} {hashlib.sha256(r.numpy().tobytes()).hexdigest()}")
open("/fix/patches.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
