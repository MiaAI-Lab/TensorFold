import sys
sys.path.insert(0, "/gen")
from imgcommon import *
from jpeg_enc import encode

rng = np.random.default_rng(11)
def pic(w, h, kind):
    if kind == "noise": return noise(w, h, 3, w * 5 + h)
    if kind == "text": return texty(w, h, 1)
    return gradient(w, h, 3, 2)
combos = {
    "h1v2": [(1, 2), (1, 1), (1, 1)], "h2v1": [(2, 1), (1, 1), (1, 1)], "h2v2": [(2, 2), (1, 1), (1, 1)], "h4v1": [(4, 1), (1, 1), (1, 1)],
    "h4v2": [(4, 2), (1, 1), (1, 1)], "h1v4": [(1, 4), (1, 1), (1, 1)], "h2v4": [(2, 4), (1, 1), (1, 1)],
    "mix_cb_h2v1": [(2, 2), (1, 2), (2, 2)], "mix_cr_v": [(2, 2), (2, 1), (1, 2)], "full": [(1, 1), (1, 1), (1, 1)],
    "chroma_big": [(1, 1), (2, 2), (2, 2)], "h3v1": [(3, 1), (1, 1), (1, 1)], "h3v3_c11": [(3, 3), (3, 1), (1, 3)],
}
sizes = [(1, 1), (2, 3), (7, 9), (17, 33), (33, 17), (63, 65), (100, 37)]
n = 0
for name, samp in combos.items():
    for (w, h) in sizes:
        if (n % 2) and (w, h) in ((1, 1), (2, 3)): n += 1; continue
        kind = ["noise", "text", "grad"][n % 3]
        add(f"je_{name}_{w}x{h}_{kind}.jpg", encode(pic(w, h, kind), samp)); n += 1
# restart intervals, with each layout
for name in ("h2v2", "h1v2", "full", "h4v1"):
    for r in (1, 2, 5):
        add(f"je_restart{r}_{name}.jpg", encode(pic(63, 65, "text"), combos[name], restart=r))
# non-interleaved scans
for name in ("h2v2", "h1v2", "full", "h2v1"):
    for (w, h) in ((17, 33), (63, 65)):
        add(f"je_sep_{name}_{w}x{h}.jpg", encode(pic(w, h, "noise"), combos[name], scans="separate"))
        add(f"je_sep_restart_{name}_{w}x{h}.jpg", encode(pic(w, h, "text"), combos[name], scans="separate", restart=3))
# colour space heuristics
full = combos["full"]
for label, kw in {
    "ids123_jfif": dict(ids=(1, 2, 3), jfif=True), "ids123_nojfif": dict(ids=(1, 2, 3), jfif=False), "idsRGB_nojfif": dict(ids=(82, 71, 66), jfif=False, ycc=False),
    "idsRGB_jfif": dict(ids=(82, 71, 66), jfif=True, ycc=False), "ids789_nojfif": dict(ids=(7, 8, 9), jfif=False), "adobe0_rgb": dict(ids=(1, 2, 3), jfif=False, adobe=0, ycc=False),
    "adobe1_ycc": dict(ids=(1, 2, 3), jfif=False, adobe=1), "adobe0_jfif": dict(ids=(1, 2, 3), jfif=True, adobe=0, ycc=False), "adobe2_3comp": dict(ids=(1, 2, 3), jfif=False, adobe=2),
    "ids012": dict(ids=(0, 1, 2), jfif=False),
}.items():
    add(f"je_cs_{label}.jpg", encode(pic(33, 17, "noise"), full, **kw))
    add(f"je_cs_{label}_420.jpg", encode(pic(33, 17, "grad"), combos["h2v2"], **kw))
# grayscale
add("je_gray.jpg", encode(pic(33, 17, "noise"), [(1, 1)], gray=True, ids=(1, 2, 3)))
add("je_gray_restart.jpg", encode(pic(63, 65, "text"), [(1, 1)], gray=True, restart=4))
write_manifest("img_jpeg_enc.txt")
print(len(manifest), "cases", sum(" ERR " in m for m in manifest), "errors")
for m in manifest:
    if " ERR " in m: print(m)
