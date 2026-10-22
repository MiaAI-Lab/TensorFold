# WebP streams with encoder settings Pillow does not expose (simple filter, partitions, segments, sharpness, ALPH
# filters / raw alpha), made by calling libwebp 1.6.0's advanced API through ctypes.
import ctypes as C, io, sys
sys.path.insert(0, "/gen")
from imgcommon import *
from webp_corpus_content import content

lib = C.CDLL("/usr/local/lib/python3.12/dist-packages/pillow.libs/libwebp-bac0fc6b.so.7.2.0")

class Config(C.Structure):
    _fields_ = [(n, C.c_int) for n in ["lossless"]] + [("quality", C.c_float)] + [(n, C.c_int) for n in ["method", "image_hint", "target_size"]] + [("target_PSNR", C.c_float)] + [(n, C.c_int) for n in ["segments", "sns_strength", "filter_strength", "filter_sharpness", "filter_type", "autofilter", "alpha_compression", "alpha_filtering", "alpha_quality", "pass_", "show_compressed", "preprocessing", "partitions", "partition_limit", "emulate_jpeg_size", "thread_level", "low_memory", "near_lossless", "exact", "use_delta_palette", "use_sharp_yuv", "qmin", "qmax"]]

class Picture(C.Structure):
    _fields_ = [("use_argb", C.c_int), ("colorspace", C.c_int), ("width", C.c_int), ("height", C.c_int),
                ("y", C.c_void_p), ("u", C.c_void_p), ("v", C.c_void_p), ("y_stride", C.c_int), ("uv_stride", C.c_int),
                ("a", C.c_void_p), ("a_stride", C.c_int), ("pad1", C.c_uint32 * 2), ("argb", C.c_void_p), ("argb_stride", C.c_int),
                ("pad2", C.c_uint32 * 3), ("writer", C.c_void_p), ("custom_ptr", C.c_void_p), ("extra_info_type", C.c_int),
                ("extra_info", C.c_void_p), ("stats", C.c_void_p), ("error_code", C.c_int), ("progress_hook", C.c_void_p),
                ("user_data", C.c_void_p), ("pad3", C.c_uint32 * 3), ("pad4", C.c_void_p), ("pad5", C.c_void_p),
                ("pad6", C.c_uint32 * 8), ("memory_", C.c_void_p), ("memory_argb_", C.c_void_p), ("pad7", C.c_void_p * 2)]

class Writer(C.Structure):
    _fields_ = [("mem", C.c_void_p), ("size", C.c_size_t), ("max_size", C.c_size_t), ("pad", C.c_uint32)]

lib.WebPMemoryWrite  # exists
ABI = 0x020f

def encode(arr, **opts):
    h, w, c = arr.shape
    cfg = Config()
    assert lib.WebPConfigInitInternal(C.byref(cfg), 0, C.c_float(opts.pop("quality", 75.0)), ABI)
    for k, v in opts.items():
        setattr(cfg, "pass_" if k == "pass" else k, v)
    assert lib.WebPValidateConfig(C.byref(cfg)), opts
    pic = Picture()
    assert lib.WebPPictureInitInternal(C.byref(pic), ABI)
    pic.width, pic.height = w, h
    pic.use_argb = 1 if cfg.lossless else 0
    wr = Writer()
    lib.WebPMemoryWriterInit(C.byref(wr))
    pic.writer = C.cast(lib.WebPMemoryWrite, C.c_void_p).value
    pic.custom_ptr = C.addressof(wr)
    rgba = np.ascontiguousarray(arr if c == 4 else np.concatenate([arr, np.full((h, w, 1), 255, np.uint8)], axis=2))
    assert lib.WebPPictureImportRGBA(C.byref(pic), rgba.ctypes.data_as(C.c_void_p), w * 4)
    ok = lib.WebPEncode(C.byref(cfg), C.byref(pic))
    assert ok, pic.error_code
    out = C.string_at(wr.mem, wr.size)
    lib.WebPPictureFree(C.byref(pic))
    lib.WebPMemoryWriterClear(C.byref(wr))
    return out

n = 0
sizes = [(17, 33), (63, 65), (100, 37), (250, 130), (16, 16), (1, 1), (33, 17), (64, 48)]
def add_enc(name, arr, **opts):
    add(name, encode(arr, **opts))
cases = []
for fs in (0, 1):   # filter type: 0 = simple, 1 = strong
    for strength in (0, 20, 60, 100):
        for sharp in (0, 3, 7):
            cases.append(dict(filter_type=fs, filter_strength=strength, filter_sharpness=sharp))
cases += [dict(segments=s, sns_strength=sns) for s in (1, 2, 3, 4) for sns in (0, 50, 100)]
cases += [dict(partitions=p) for p in (1, 2, 3)]
cases += [dict(autofilter=1, filter_strength=50), dict(method=0, filter_type=0), dict(method=6, segments=4, partitions=3, filter_strength=80, filter_sharpness=5)]
kinds = ["text", "noise", "grad", "blocks"]
for i, o in enumerate(cases):
    w, h = sizes[i % len(sizes)]
    kind = kinds[i % 4]
    q = [20, 50, 80, 95][i % 4]
    tag = "_".join(f"{k}{v}" for k, v in o.items())
    add_enc(f"wadv_{tag}_q{q}_{kind}_{w}x{h}.webp", content(kind, w, h), quality=float(q), **o)
# a few with larger pictures for partitions / filters over many macroblocks
for i, o in enumerate([dict(partitions=3, filter_type=0, filter_strength=70), dict(partitions=2, filter_strength=40, filter_sharpness=4), dict(segments=4, sns_strength=100, partitions=1)]):
    add_enc(f"wadv_big{i}.webp", content("text", 300, 200), quality=60.0, **o)
    add_enc(f"wadv_bigg{i}.webp", content("grad", 320, 240), quality=40.0, **o)
# alpha: compression off / on, filters, quality levels
for comp in (0, 1):
    for filt in (0, 1, 2, 3):
        for aq in (100, 70, 30):
            if comp == 0 and aq != 100: continue
            w, h = sizes[(comp * 7 + filt * 3 + aq) % len(sizes)]
            add_enc(f"wadv_alpha_c{comp}_f{filt}_aq{aq}_{w}x{h}.webp", content("grad" if filt % 2 else "text", w, h, True), quality=60.0, alpha_compression=comp, alpha_filtering=filt, alpha_quality=aq)
# lossless variants
for i, o in enumerate([dict(lossless=1, quality=0.0, method=0), dict(lossless=1, quality=100.0, method=6), dict(lossless=1, near_lossless=40), dict(lossless=1, use_delta_palette=1, exact=1), dict(lossless=1, quality=50.0, method=3, exact=1)]):
    q = o.pop("quality", 75.0)
    for kind in ("text", "pal12", "noise", "grad"):
        w, h = sizes[(i + len(kind)) % len(sizes)]
        a = content(kind, w, h, True) if i % 2 else content(kind, w, h)
        add_enc(f"wadv_ll{i}_{kind}_{w}x{h}.webp", a, quality=q, **o)
write_manifest("img_webp_adv.txt")
print(len(manifest), "cases", sum(" ERR " in m for m in manifest), "errors")
