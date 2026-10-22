# Geometry fixtures: smart_resize, content_size, sample_frames from the reference glm.py.
import random, sys
from tensorfold.vision.glm import smart_resize, content_size, sample_frames, GlmVisionLimits
from tensorfold.vision.images_http import ImageInputError

rnd = random.Random(1234)
out = []
sizes = [1, 2, 3, 13, 14, 27, 28, 29, 55, 56, 57, 100, 224, 480, 720, 1000, 1080, 1920, 2048, 4000, 4096, 8191, 8192]
def pick():
    return rnd.choice(sizes) if rnd.random() < .6 else rnd.randint(1, 8192)
for _ in range(1500):
    frames = rnd.choice([2, 2, 2, 2, 4, 6, 8, 16, 64, 128, 3, 5])
    h, w = pick(), pick()
    mx = rnd.choice([16, 17, 64, 100, 327, 2048, 4096, 8000, 16384, 240000, 1])
    mn = 16
    try:
        ch, cw = smart_resize(frames, h, w, mx, mn)
        c = content_size(frames, h, w, (ch, cw), mn)
        out.append(f"smart {frames} {h} {w} {mx} {mn} {ch} {cw} {c[0]} {c[1]}")
    except ImageInputError as e:
        out.append(f"smart {frames} {h} {w} {mx} {mn} ERR {e}")
open("/fix/geometry_smart.txt", "w").write("\n".join(out) + "\n")

out = []
L = GlmVisionLimits()
cases = []
for _ in range(500):
    total = rnd.choice([1, 2, 3, 5, 10, 24, 30, 60, 100, 127, 128, 129, 250, 256, 300, 1000, 5000, 7200, 36000]) if rnd.random() < .5 else rnd.randint(1, 20000)
    fps = rnd.choice([1.0, 5.0, 12.0, 15.0, 23.976023976023978, 24.0, 25.0, 29.97002997002997, 30.0, 50.0, 59.94005994005994, 60.0, 120.0, 0.5, 1000.0]) if rnd.random() < .8 else rnd.uniform(0.1, 200)
    cases.append((total, fps))
for total, fps in cases:
    vf = rnd.choice([2, 8, 32, 128, 500, 2048])
    lim = GlmVisionLimits(video_frames=vf)
    r = sample_frames(total, fps, total / fps, lim)
    out.append(f"sample {total} {fps!r} {(total / fps)!r} {vf} {len(r)} " + " ".join(map(str, r)))
# duration == 0 branch
for total, fps in cases[:100]:
    r = sample_frames(total, fps, 0.0, L)
    out.append(f"sample {total} {fps!r} 0.0 128 {len(r)} " + " ".join(map(str, r)))
open("/fix/geometry_sample.txt", "w").write("\n".join(out) + "\n")
print("ok")

# f"{s:.1f}" of frame-pair times i / fps
out = []
for _ in range(3000):
    fps = rnd.choice([1.0, 5.0, 12.0, 15.0, 23.976023976023978, 24.0, 25.0, 29.97002997002997, 30.0, 50.0, 59.94005994005994, 60.0, 0.5])
    i = rnd.randint(0, 200000)
    s = i / fps
    out.append(f"{s!r} {s:.1f}")
for s in [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 1.25, 2.675, 0.0, 1e-300, 5e-324, 123456789.25]:
    out.append(f"{s!r} {s:.1f}")
open("/fix/format1.txt", "w").write("\n".join(out) + "\n")
