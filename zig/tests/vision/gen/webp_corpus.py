import io, struct, sys
sys.path.insert(0, "/gen")
from imgcommon import *
from PIL import Image

sizes = [(1, 1), (2, 2), (3, 5), (16, 16), (17, 33), (63, 65), (100, 37), (250, 130), (33, 17)]
from webp_corpus_content import content

def webp(name, arr, mode=None, **kw):
    im = Image.fromarray(arr, mode) if mode else Image.fromarray(arr)
    b = io.BytesIO(); im.save(b, "WEBP", **kw)
    add(name, b.getvalue()); return b.getvalue()

kinds = ["noise", "grad", "text", "blocks", "pal2", "pal5", "pal12", "pal100", "flat", "repeat"]
n = 0
for (w, h) in sizes:
    for kind in kinds:
        if (n % 3) == 0 and w * h > 5000 and kind in ("noise",): n += 1; continue
        webp(f"wl_{kind}_{w}x{h}.webp", content(kind, w, h), lossless=True, quality=100 if n % 2 else 40, method=n % 7)
        n += 1
for (w, h) in sizes:
    for kind in ("noise", "grad", "text", "pal5", "blocks"):
        webp(f"wla_{kind}_{w}x{h}.webp", content(kind, w, h, True), "RGBA", lossless=True, quality=70, method=n % 7)
        n += 1
        webp(f"wla_exact_{kind}_{w}x{h}.webp", content(kind, w, h, True), "RGBA", lossless=True, exact=True, method=n % 7)
for (w, h) in sizes:
    for kind in ("noise", "grad", "text", "blocks"):
        for q in (1, 30, 75, 100):
            if (n % 2) and q in (30,): n += 1; continue
            webp(f"wy_{kind}_q{q}_{w}x{h}.webp", content(kind, w, h), quality=q, method=n % 7)
            n += 1
for (w, h) in sizes:
    for kind in ("grad", "text", "noise"):
        for aq in (100, 60):
            webp(f"wya_{kind}_aq{aq}_{w}x{h}.webp", content(kind, w, h, True), "RGBA", quality=70, alpha_quality=aq, method=n % 7)
            n += 1
# lossless images with big dimensions to exercise backward references and the colour cache
webp("wl_big_text.webp", content("text", 400, 300), lossless=True, quality=100, method=6)
webp("wl_big_grad.webp", content("grad", 400, 300), lossless=True, quality=100, method=6)
webp("wl_big_repeat.webp", content("repeat", 500, 300), lossless=True, quality=100, method=6)
webp("wy_big_text.webp", content("text", 400, 300), quality=80, method=6)
webp("wy_big_grad.webp", content("grad", 640, 480), quality=60, method=4)
webp("wy_big_noise.webp", content("noise", 300, 200), quality=90, method=6)
# EXIF / XMP orientation
base = content("noise", 17, 33)
for o in range(1, 9):
    ex = Image.Exif(); ex[0x0112] = o
    for lossless in (False, True):
        webp(f"exif_o{o}_{'ll' if lossless else 'ly'}.webp", base, lossless=lossless, exif=ex.tobytes())
xmp = b'<x:xmpmeta><rdf:Description tiff:Orientation="6"/></x:xmpmeta>'
webp("xmp_o6.webp", base, lossless=True, xmp=xmp)
# animated: refused
frames = [Image.fromarray(content("noise", 20, 20)), Image.fromarray(content("grad", 20, 20))]
b = io.BytesIO(); frames[0].save(b, "WEBP", save_all=True, append_images=frames[1:], duration=100, lossless=True); add("anim2.webp", b.getvalue())
# damaged
good = webp("tmp_good.webp", content("grad", 40, 40), quality=80)
good_ll = webp("tmp_good_ll.webp", content("text", 40, 40), lossless=True)
add("bad_trunc_lossy.webp", good[: len(good) // 2])
add("bad_trunc_lossless.webp", good_ll[: len(good_ll) // 2])
add("bad_riff_size.webp", good[:4] + struct.pack("<I", 9999) + good[8:])
add("bad_garbage.webp", b"RIFF\x20\x00\x00\x00WEBPVP8 " + bytes(range(100)))
add("bad_empty.webp", b"RIFF\x04\x00\x00\x00WEBP")
c = bytearray(good); c[len(c) // 2] ^= 0xaa
add("bad_flip_lossy.webp", bytes(c))
c = bytearray(good_ll); c[len(c) // 2] ^= 0xaa
add("bad_flip_lossless.webp", bytes(c))
manifest[:] = [m for m in manifest if not m.startswith("tmp_")]
import os
for f in ("tmp_good.webp", "tmp_good_ll.webp"): os.remove(os.path.join("/fix/img", f))
write_manifest("img_webp.txt")
print(len(manifest), "webp cases", sum(" ERR " in m for m in manifest), "errors")
