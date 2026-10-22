# What the recipe's decode_clip makes of each clip under a few limits (shape, frame-pair times, content hash).
import json, os, sys, time
sys.path.insert(0, "/gen")
from dataclasses import replace
import tensorfold.vision.glm as G
from tensorfold.vision.images_http import ImageInputError

out = []
variants = [("default", G.GlmVisionLimits()),
            ("tiny", replace(G.GlmVisionLimits(), video_tokens=64, video_frames=8)),
            ("two", replace(G.GlmVisionLimits(), video_frames=2)),
            ("wide", replace(G.GlmVisionLimits(), video_tokens=240, video_frames=64))]
for f in sorted(os.listdir("/fix/vid")):
    if f == "expected.json" or f.endswith(".json"): continue
    data = open("/fix/vid/" + f, "rb").read()
    for name, lim in variants:
        e = {"file": f, "variant": name, "video_tokens": lim.video_tokens, "video_frames": lim.video_frames}
        try:
            c = G.decode_clip(data, lim, time.monotonic() + 1000)
            e["shape"] = list(c.frames.shape); e["seconds"] = [repr(s) for s in c.seconds]; e["hash"] = c.content_hash
        except ImageInputError as ex:
            e["error"] = str(ex)
        out.append(e)
# damaged inputs
good = open("/fix/vid/h264_420_64x48.mp4", "rb").read()
for name, data in [("trunc", good[: len(good) // 2]), ("garbage", bytes(range(256)) * 20), ("empty", b""), ("png", open("/fix/img/pil_rgb.png", "rb").read()), ("flip", bytes(b ^ (0x55 if i == len(good) // 2 else 0) for i, b in enumerate(good)))]:
    open(f"/fix/vid/bad_{name}.bin", "wb").write(data)
    e = {"file": f"bad_{name}.bin", "variant": "default", "video_tokens": 16384, "video_frames": 128}
    try:
        c = G.decode_clip(data, G.GlmVisionLimits(), time.monotonic() + 1000)
        e["shape"] = list(c.frames.shape); e["seconds"] = [repr(s) for s in c.seconds]; e["hash"] = c.content_hash
    except ImageInputError as ex:
        e["error"] = str(ex)
    out.append(e)
json.dump(out, open("/fix/vid/clips.json", "w"))
print(len(out), sum("error" in e for e in out))
for e in out:
    if "error" in e: print(e["file"], e["variant"], e["error"])
