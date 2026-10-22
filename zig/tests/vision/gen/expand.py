# prepare()/escape_quoted() fixtures from the reference GlmVision (no tower: the object is built bare).
import hashlib, json, sys
from collections import OrderedDict
import torch
sys.path.insert(0, "/gen")
from common import lcg_bytes
import tensorfold.vision.glm as G
from tensorfold.vision.images_http import ImageInputError

IMG, VID = G.IMAGE_SPAN, G.VIDEO_SPAN

LAST = []

class Tok:                      # stands in for the checkpoint's tokenizer: <|image|> -> 154854, else the bytes
    def encode_batch_fast(self, texts, add_special_tokens=False):
        t = texts[0]
        LAST.append(t)
        ids, pos = [], 0
        while pos < len(t):
            if t.startswith(G.IMAGE, pos):
                ids.append(154854); pos += len(G.IMAGE)
            else:
                ids.extend(t[pos].encode("utf-8")); pos += 1
        class R: pass
        r = R(); r.ids = ids
        return [r]

def bare(cache: bool):
    v = G.GlmVision.__new__(G.GlmVision)
    v.limits = G.GlmVisionLimits()
    v.image_token = 154854
    v.tok = Tok()
    v._pictures = OrderedDict(); v._canvases = OrderedDict()
    G.PICTURE_CACHE = cache
    return v

def canvas(seed, h, w):
    return torch.frombuffer(bytearray(lcg_bytes(seed, 3 * h * w)), dtype=torch.uint8).view(3, h, w).clone()

class Dec:  # a decoded image, only its size matters
    def __init__(self, w, h): self.width, self.height = w, h

def run(case_images, case_videos, text, cache, max_prompt=None):
    v = bare(cache)
    images = []
    for im in case_images:
        c = canvas(im["seed"], im["h"], im["w"])
        v._picture = lambda image, tokens, c=c: c
        if cache:
            images.append(v._remember((b"k", "auto", 2048, 16), b"data", "auto", 2048, Dec(im["pw"], im["ph"]), 32 * 1024 * 1024))
        else:
            images.append(c)
    videos = []
    for vd in case_videos:
        frames = torch.frombuffer(bytearray(lcg_bytes(vd["seed"], vd["t"] * 3 * vd["h"] * vd["w"])), dtype=torch.uint8).view(vd["t"], 3, vd["h"], vd["w"]).clone()
        videos.append(G.GlmClip(frames, tuple(vd["seconds"])))
    out = {"text": text, "images": case_images, "videos": case_videos, "cache": cache, "max_prompt": max_prompt}
    try:
        p = v.prepare(text, images, videos=videos, max_prompt_tokens=max_prompt)
        out["ids_len"] = len(p.token_ids)
        out["positions"] = p.positions
        out["content_hash"] = p.content_hash
        out["rows"] = p.item_rows
        out["keys"] = p.item_keys
        out["expanded"] = LAST[-1]
    except (ImageInputError, ValueError) as e:
        out["error"] = str(e)
    return out

cases = []
seconds = [0.0, 0.5, 1.0, 7.949999999999999, 33.333333333333336, 12.25, 100000.05, 0.05, 0.25, 0.35, 2.675, 1e-9]
zw = "​"
texts = [
    "hi " + IMG + " bye",
    "a " + IMG + " b " + VID + " c " + IMG + " d",
    "<|image|> quoted and <|begin_of_image|> and <|end_of_video|> and <|video|> " + IMG,
    "no media at all but <|image|>",
    IMG + IMG,
    VID + " " + IMG,
    "x" + IMG + "<|begin_of_image|>y<|end_of_image|>",
    "ünïcode ​ " + IMG + " <|begin_of_video|>",
]
imgs = [{"seed": 1, "h": 56, "w": 84, "pw": 100, "ph": 70}, {"seed": 2, "h": 28, "w": 28, "pw": 20, "ph": 20}, {"seed": 3, "h": 84, "w": 28, "pw": 10, "ph": 30}]
vids = [{"seed": 9, "t": 4, "h": 28, "w": 56, "seconds": seconds[:2]}, {"seed": 10, "t": 6, "h": 56, "w": 28, "seconds": seconds[2:5]}]
for cache in (True, False):
    cases.append(run(imgs[:1], [], texts[0], cache))
    cases.append(run(imgs[:2], vids[:1], texts[1], cache))
    cases.append(run(imgs[:1], [], texts[2], cache))
    cases.append(run([], [], texts[3], cache))
    cases.append(run(imgs[:2], [], texts[4], cache))
    cases.append(run(imgs[:1], vids[:2], texts[5], cache))
    cases.append(run(imgs[:1], [], texts[6], cache))
    cases.append(run(imgs[:1], [], texts[7], cache))
    cases.append(run(imgs[:1], [], texts[0], cache, max_prompt=40))
    cases.append(run(imgs[:1], [], texts[0], cache, max_prompt=100000))
    cases.append(run(imgs, vids, "A" + IMG + "B" + VID + "C" + IMG + "D" + IMG + "E" + VID, cache))
    cases.append(run(imgs[:1], [], texts[0] + IMG, cache))   # count mismatch
open("/fix/expand.json", "w").write(json.dumps(cases))

# escape_quoted fixtures
v = bare(True)
eq = []
def esc(text, ends):
    return {"text": text, "ends": ends, "out": v.escape_quoted(text, ends)}
t1 = "pre <|image|> mid " + IMG + " post <|begin_of_video|> " + VID + " <|end_of_image|>"
starts = []
i = t1.find(IMG)
eq.append(esc(t1, []))
eq.append(esc(t1, [i]))
j = t1.find(VID)
eq.append(esc(t1, [i, j]))
eq.append(esc("plain text", []))
eq.append(esc("a<|image|>b<|video|>c", [3]))
# utf-8 text: byte offsets differ from code point offsets, so the fixture gives the offsets in bytes and the
# Python run uses the code point ones
t2 = "ééé " + IMG + " é <|image|> " + VID
cp = t2.find(IMG); cp2 = t2.find(VID)
r = v.escape_quoted(t2, [cp])
eq.append({"text": t2, "ends": [len(t2[:cp].encode())], "out": r})
r = v.escape_quoted(t2, [cp, cp2])
eq.append({"text": t2, "ends": [len(t2[:cp].encode()), len(t2[:cp2].encode())], "out": r})
open("/fix/escape.json", "w").write(json.dumps(eq))
print(len(cases), "cases;", sum("error" in c for c in cases), "errors")
for c in cases[:14]:
    print({k: c.get(k) for k in ("error", "ids_len", "rows", "keys")})
