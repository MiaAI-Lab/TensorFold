import io, struct, sys
sys.path.insert(0, "/gen")
from imgcommon import *
from imgcommon import _chunk
from PIL import Image

rng = np.random.default_rng(42)
sizes = [(1, 1), (17, 33), (63, 65), (8, 8), (100, 37)]
combos = [(0, 1), (0, 2), (0, 4), (0, 8), (0, 16), (2, 8), (2, 16), (3, 1), (3, 2), (3, 4), (3, 8), (4, 8), (4, 16), (6, 8), (6, 16)]
nchan = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
n = 0
for ct, depth in combos:
    for inter in (0, 1):
        for si in ((0, 1, 2, 3, 4) if inter else (1, 2, 4)):
            if (si + ct + depth + inter) % 3 == 0 and si not in (0, 1): continue
            w, h = sizes[si]
            smax = (1 << depth) - 1
            pal = None
            if ct == 3:
                k = min(1 << depth, int(rng.integers(max(2, (1 << depth) // 2), (1 << depth) + 1)))
                pal = rng.integers(0, 256, k * 3, dtype=np.uint8).tobytes()
                data = rng.integers(0, k, (h, w, 1))
            else:
                data = rng.integers(0, smax + 1, (h, w, nchan[ct]))
            add(f"png_ct{ct}_d{depth}_i{inter}_{w}x{h}.png", png(data, w, h, ct, depth, inter, palette=pal, seed=n))
            n += 1

# tRNS
for (ct, depth, w, h) in [(0, 1, 17, 33), (0, 2, 17, 33), (0, 4, 17, 33), (0, 8, 63, 65), (0, 8, 8, 8), (0, 16, 17, 33), (2, 8, 63, 65), (2, 16, 17, 33), (2, 8, 8, 8)]:
    smax = (1 << depth) - 1
    if ct == 0:
        tv = int(rng.integers(0, smax + 1))
        data = rng.integers(0, min(smax + 1, 4), (h, w, 1))
        data = np.where(data == 3, tv, data) % (smax + 1)
        trns = struct.pack(">H", tv)
        for inter in (0, 1):
            add(f"png_trns_gray_d{depth}_i{inter}_{w}x{h}.png", png(data, w, h, 0, depth, inter, trns=trns, seed=n)); n += 1
        # the transparent sample appears everywhere
        data2 = np.full((h, w, 1), tv)
        data2[::2, ::3] = (tv + 1) % (smax + 1)
        add(f"png_trns_gray_d{depth}_hit_{w}x{h}.png", png(data2, w, h, 0, depth, 0, trns=trns, seed=n)); n += 1
    else:
        tv = tuple(int(v) for v in rng.integers(0, smax + 1, 3))
        data = rng.integers(0, 3, (h, w, 3)) * (smax // 2)
        data[::2, ::2] = tv
        trns = struct.pack(">HHH", *tv)
        add(f"png_trns_rgb_d{depth}_{w}x{h}.png", png(data, w, h, 2, depth, 0, trns=trns, seed=n)); n += 1
for depth in (1, 2, 4, 8):
    w, h = 31, 17
    k = min(1 << depth, 16)
    pal = rng.integers(0, 256, k * 3, dtype=np.uint8).tobytes()
    data = rng.integers(0, k, (h, w, 1))
    for j, tl in enumerate((1, k // 2 + 1, k)):
        trns = bytes(rng.integers(0, 256, tl, dtype=np.uint8).tolist())
        add(f"png_trns_pal_d{depth}_t{j}_n{tl}.png", png(data, w, h, 3, depth, depth % 2, palette=pal, trns=trns, seed=n)); n += 1
# alpha extremes in RGBA / LA
for ct, depth in ((6, 8), (6, 16), (4, 8), (4, 16)):
    w, h = 40, 30
    d = rng.integers(0, (1 << depth), (h, w, nchan[ct]))
    d[..., -1] = rng.choice([0, 1, 127, 128, 254, (1 << depth) - 1], (h, w)) if depth == 8 else rng.choice([0, 255, 256, 32768, 65535, 12345], (h, w))
    add(f"png_alpha_ct{ct}_d{depth}.png", png(d, w, h, ct, depth, 0, seed=n)); n += 1

# split IDAT, compression levels, fixed filters
data = rng.integers(0, 256, (65, 63, 3))
add("png_idat_split.png", png(data, 63, 65, 2, 8, 0, split=50, seed=1))
for ft in range(5):
    add(f"png_filter{ft}.png", png(gradient(40, 40, 3), 40, 40, 2, 8, 0, fixed=ft, seed=1))
    add(f"png_filter{ft}_rgba.png", png(noise(33, 21, 4, ft), 33, 21, 6, 8, 0, fixed=ft, seed=1))
    add(f"png_filter{ft}_gray16.png", png(rng.integers(0, 65536, (21, 33, 1)), 33, 21, 0, 16, 1, fixed=ft, seed=1))
add("png_level0.png", png(noise(40, 40, 3, 1), 40, 40, 2, 8, 0, level=0))

# written by Pillow
def pil_png(name, im, **kw):
    b = io.BytesIO(); im.save(b, "PNG", **kw); add(name, b.getvalue())
a = noise(63, 65, 3, 7)
pil_png("pil_rgb.png", Image.fromarray(a))
pil_png("pil_rgb_opt.png", Image.fromarray(a), optimize=True, compress_level=9)
pil_png("pil_rgba.png", Image.fromarray(noise(63, 65, 4, 8), "RGBA"))
pil_png("pil_la.png", Image.fromarray(noise(63, 65, 2, 9), "LA"))
pil_png("pil_l.png", Image.fromarray(noise(63, 65, 1, 10)[..., 0], "L"))
pil_png("pil_1.png", Image.fromarray(noise(63, 65, 1, 11)[..., 0] > 127))
pil_png("pil_i16.png", Image.fromarray((noise(63, 65, 2, 12).view(">u2")[..., 0]).astype(np.uint16), "I;16"))
p = Image.fromarray(texty(100, 60)).quantize(16)
pil_png("pil_p16.png", p, optimize=True)
pil_png("pil_p16_trns.png", p, transparency=3)
pil_png("pil_p16_trnsb.png", p, transparency=bytes(range(0, 256, 17)))
pil_png("pil_l_trns.png", Image.fromarray(noise(30, 30, 1, 13)[..., 0] // 8 * 8, "L"), transparency=8)
pil_png("pil_rgb_trns.png", Image.fromarray(blocks(40, 40, 1)), transparency=tuple(int(v) for v in blocks(40, 40, 1)[0, 0]))
pil_png("pil_text.png", Image.fromarray(texty(250, 130)))
pil_png("pil_grad.png", Image.fromarray(gradient(250, 130, 3)))
pil_png("pil_rgb_srgb.png", Image.fromarray(a), icc_profile=b"")
# EXIF orientation on PNG (eXIf chunk)
base = noise(17, 33, 3, 21)
for o in range(1, 9):
    ex = Image.Exif(); ex[0x0112] = o
    pil_png(f"exif_o{o}.png", Image.fromarray(base), exif=ex)
ex = Image.Exif(); ex[0x0112] = 6
pil_png("exif_o6_rgba.png", Image.fromarray(noise(17, 33, 4, 22), "RGBA"), exif=ex)
ex = Image.Exif(); ex[0x0112] = 9
pil_png("exif_o9.png", Image.fromarray(base), exif=ex)
ex = Image.Exif(); ex[0x0112] = 0
pil_png("exif_o0.png", Image.fromarray(base), exif=ex)

# invalid files
good = png(noise(20, 20, 3, 5), 20, 20, 2, 8, 0)
add("bad_signature.png", b"\x89PNG\r\n\x1a\x0b" + good[8:])
add("bad_trunc_half.png", good[: len(good) // 2])
add("bad_no_iend.png", good[:-12])
c = bytearray(good); c[40] ^= 1
add("bad_crc_idat.png", bytes(c))
c = bytearray(good); c[20] ^= 1
add("bad_crc_ihdr.png", bytes(c))
add("bad_empty.png", b"")
add("bad_garbage.png", bytes(range(256)) * 4)
add("bad_zero_dim.png", png(np.zeros((1, 1, 3), int), 1, 1, 2, 8, 0).replace(struct.pack(">II", 1, 1), struct.pack(">II", 0, 1), 1))
z = zlib.compress(b"\x00" + bytes(10), 6)   # too few rows
add("bad_short_data.png", b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", 20, 20, 8, 2, 0, 0, 0)) + _chunk(b"IDAT", z) + _chunk(b"IEND", b""))
write_manifest("img_png.txt")
print(len(manifest), "png cases", sum(" ERR " in m for m in manifest), "errors")
