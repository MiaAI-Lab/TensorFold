import io, struct, sys
sys.path.insert(0, "/gen")
from imgcommon import *
from PIL import Image

contents = {
    "noise": lambda w, h: noise(w, h, 3, w * 31 + h),
    "grad": lambda w, h: gradient(w, h, 3, 5),
    "text": lambda w, h: texty(w, h, 3),
    "blocks": lambda w, h: blocks(w, h, 4),
}
sizes = [(1, 1), (2, 2), (3, 5), (8, 8), (16, 16), (17, 33), (63, 65), (100, 37), (250, 130), (33, 17), (65, 63)]

def jpg(name, im, **kw):
    b = io.BytesIO()
    im.save(b, "JPEG", **kw)
    add(name, b.getvalue())
    return b.getvalue()

n = 0
for sub in (0, 1, 2, "4:4:0", "4:1:1"):
    for q in (5, 50, 75, 95, 100):
        w, h = sizes[n % len(sizes)]; n += 1
        c = list(contents)[n % 4]
        try:
            jpg(f"j_sub{str(sub).replace(':','')}_q{q}_{c}_{w}x{h}.jpg", Image.fromarray(contents[c](w, h)), quality=q, subsampling=sub)
        except Exception as e:
            print("skip", sub, e)
for sub in (0, 1, 2, "4:4:0"):
    for (w, h) in [(17, 33), (63, 65), (250, 130), (1, 1), (8, 8)]:
        for c in ("noise", "text"):
            try:
                jpg(f"j_all_sub{str(sub).replace(':','')}_{c}_{w}x{h}.jpg", Image.fromarray(contents[c](w, h)), quality=85, subsampling=sub)
            except Exception as e:
                print("skip", sub, e)
for (w, h) in sizes:
    for prog in (False, True):
        for opt in (False, True):
            if prog is False and opt is False and (w + h) % 2: continue
            jpg(f"j_prog{int(prog)}_opt{int(opt)}_{w}x{h}.jpg", Image.fromarray(contents["text"](w, h)), quality=80, progressive=prog, optimize=opt)
for (w, h) in [(63, 65), (250, 130), (17, 33)]:
    for sub in (0, 2):
        jpg(f"j_progressive_noise_sub{sub}_{w}x{h}.jpg", Image.fromarray(contents["noise"](w, h)), quality=90, progressive=True, subsampling=sub)
        jpg(f"j_progressive_grad_sub{sub}_{w}x{h}.jpg", Image.fromarray(contents["grad"](w, h)), quality=60, progressive=True, subsampling=sub)
# gray
for (w, h) in [(1, 1), (17, 33), (63, 65), (250, 130)]:
    for prog in (False, True):
        jpg(f"j_gray_p{int(prog)}_{w}x{h}.jpg", Image.fromarray(noise(w, h, 1, w)[..., 0], "L"), quality=75, progressive=prog)
# CMYK
for (w, h) in [(17, 33), (63, 65)]:
    cm = Image.fromarray(noise(w, h, 4, 3), "CMYK")
    jpg(f"j_cmyk_{w}x{h}.jpg", cm, quality=85)
    jpg(f"j_cmyk_prog_{w}x{h}.jpg", cm, quality=85, progressive=True)
# keep RGB
for (w, h) in [(17, 33), (63, 65)]:
    try:
        jpg(f"j_keeprgb_{w}x{h}.jpg", Image.fromarray(contents["noise"](w, h)), quality=90, keep_rgb=True)
    except Exception as e:
        print("skip keep_rgb", e)
# restart markers
for (w, h) in [(63, 65), (250, 130)]:
    for blocks_n in (1, 3, 7):
        try:
            jpg(f"j_restart{blocks_n}_{w}x{h}.jpg", Image.fromarray(contents["text"](w, h)), quality=85, restart_marker_blocks=blocks_n)
            jpg(f"j_restart_rows{blocks_n}_{w}x{h}.jpg", Image.fromarray(contents["noise"](w, h)), quality=85, restart_marker_rows=blocks_n, subsampling=2)
        except Exception as e:
            print("skip restart", e)
# EXIF orientation
base = contents["noise"](17, 33)
for o in range(0, 10):
    ex = Image.Exif(); ex[0x0112] = o
    jpg(f"exif_o{o}.jpg", Image.fromarray(base), exif=ex, quality=90)
ex = Image.Exif(); ex[0x0112] = 6
jpg("exif_o6_gray.jpg", Image.fromarray(base[..., 0], "L"), exif=ex)
jpg("exif_o8_sub2_big.jpg", Image.fromarray(contents["text"](100, 37)), exif=ex, subsampling=2)
# EXIF in unusual forms (hand written TIFF blocks)
def exif_block(order, tag_type, count, value_bytes, big=False):
    e = ">" if big else "<"
    head = (b"MM\x00*" if big else b"II*\x00") + struct.pack(e + "I", 8)
    ifd = struct.pack(e + "H", 1) + struct.pack(e + "HHI", 0x0112, tag_type, count) + value_bytes + struct.pack(e + "I", 0)
    return b"Exif\x00\x00" + head + ifd
def with_app1(jpeg_bytes, payload, ident=None):
    seg = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    return jpeg_bytes[:2] + seg + jpeg_bytes[2:]
plain = io.BytesIO(); Image.fromarray(base).save(plain, "JPEG", quality=90); plain = plain.getvalue()
# strip JFIF so we control the APP1 position: keep as is
for label, payload in [
    ("be_short6", exif_block(0, 3, 1, struct.pack(">HH", 6, 0), big=True)),
    ("le_long8", exif_block(0, 4, 1, struct.pack("<I", 8))),
    ("le_byte3", exif_block(0, 1, 1, struct.pack("<BBBB", 3, 0, 0, 0))),
    ("le_short_count2", exif_block(0, 3, 2, struct.pack("<HH", 6, 6))),
    ("le_short0", exif_block(0, 3, 1, struct.pack("<HH", 0, 0))),
    ("le_short99", exif_block(0, 3, 1, struct.pack("<HH", 99, 0))),
    ("trunc", exif_block(0, 3, 1, struct.pack("<HH", 6, 0))[:20]),
    ("noheader_ii", exif_block(0, 3, 1, struct.pack("<HH", 6, 0))[6:]),
    ("bad_magic", b"Exif\x00\x00" + b"XX\x00*" + b"\x00" * 20),
]:
    add(f"exifraw_{label}.jpg", with_app1(plain, payload))
# XMP orientation
xmp = b"http://ns.adobe.com/xap/1.0/\x00" + b'<x:xmpmeta><rdf:Description tiff:Orientation="6"/></x:xmpmeta>'
add("xmp_o6_attr.jpg", with_app1(plain, xmp))
xmp2 = b"http://ns.adobe.com/xap/1.0/\x00" + b"<tiff:Orientation>8</tiff:Orientation>"
add("xmp_o8_elem.jpg", with_app1(plain, xmp2))
# two APP1s: EXIF first wins
add("exif_two_app1.jpg", with_app1(with_app1(plain, exif_block(0, 3, 1, struct.pack("<HH", 3, 0))), exif_block(0, 3, 1, struct.pack("<HH", 6, 0))))

# damaged files
good = plain
add("bad_trunc_half.jpg", good[: len(good) // 2])
add("bad_trunc_noeoi.jpg", good[:-2])
add("bad_trunc_tail10.jpg", good[:-10])
add("bad_empty.jpg", b"\xff\xd8\xff")
add("bad_soi_only.jpg", b"\xff\xd8\xff\xd9")
add("bad_garbage.jpg", b"\xff\xd8\xff" + bytes(range(256)) * 3)
c = bytearray(good); c[len(c) // 2] ^= 0x55
add("bad_flip_mid.jpg", bytes(c))
pj = io.BytesIO(); Image.fromarray(contents["text"](63, 65)).save(pj, "JPEG", progressive=True, quality=80); pj = pj.getvalue()
add("bad_prog_trunc.jpg", pj[: len(pj) * 3 // 4])
add("bad_prog_noeoi.jpg", pj[:-2])
# a marker inside the entropy data (corrupt): zero fill
c = bytearray(good); i = len(c) // 2; c[i:i + 2] = b"\xff\xd0"
add("bad_stray_rst.jpg", bytes(c))
write_manifest("img_jpeg.txt")
print(len(manifest), "jpeg cases", sum(" ERR " in m for m in manifest), "errors")
