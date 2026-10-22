import hashlib, io, os, struct, sys, zlib
import numpy as np
sys.path.insert(0, "/gen")
from PIL import Image, ImageDraw

OUT = "/fix/img"

def noise(w, h, c, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, c), dtype=np.uint8)

def gradient(w, h, c, seed=0):
    ys, xs = np.mgrid[0:h, 0:w]
    ch = [((xs * 255) // max(w - 1, 1) + k * 60 + seed) % 256 for k in range(c)]
    ch[-1] = (ys * 255) // max(h - 1, 1) if c > 1 else ch[-1]
    return np.stack(ch, axis=-1).astype(np.uint8)

def texty(w, h, seed=0):
    im = Image.new("RGB", (w, h), (250, 248, 240))
    d = ImageDraw.Draw(im)
    rng = np.random.default_rng(seed)
    for i in range(0, h, 12):
        d.text((2, i), "The quick brown fox %d" % rng.integers(0, 999), fill=(10, 20, 30))
    for _ in range(max(2, w * h // 800)):
        x, y = rng.integers(0, w), rng.integers(0, h)
        d.rectangle([x, y, x + rng.integers(1, 20), y + rng.integers(1, 12)], outline=tuple(int(v) for v in rng.integers(0, 255, 3)))
    return np.asarray(im)

def blocks(w, h, seed=0):
    rng = np.random.default_rng(seed)
    a = np.zeros((h, w, 3), np.uint8)
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            a[y:y + 8, x:x + 8] = rng.integers(0, 256, 3)
    return a

def save(name, data):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "wb") as f:
        f.write(data)

def reference(data):
    """What the recipe's images._decode makes of the bytes, or the error text."""
    from tensorfold.vision.images import _decode, DEFAULT_LIMITS
    from tensorfold.vision.images_http import ImageInputError
    try:
        r = _decode(data, "auto", DEFAULT_LIMITS, DEFAULT_LIMITS.max_total_pixels)
        return r.width, r.height, r.pixels
    except ImageInputError as e:
        return str(e)

manifest = []
def add(name, data):
    save(name, data)
    r = reference(data)
    if isinstance(r, str):
        manifest.append(f"{name} ERR {r}")
    else:
        w, h, px = r
        manifest.append(f"{name} {w} {h} {hashlib.sha256(px).hexdigest()}")

def write_manifest(fname):
    with open(os.path.join("/fix", fname), "w") as f:
        f.write("\n".join(manifest) + "\n")

# ---- a PNG writer covering every colour type, bit depth, interlace and filter ----
def _chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)

def _pack_row(samples, depth):
    """samples: list of ints for one row (each sample `depth` bits); returns bytes."""
    if depth == 8:
        return bytes(samples)
    if depth == 16:
        return b"".join(struct.pack(">H", s) for s in samples)
    out = bytearray(); acc = 0; n = 0
    for s in samples:
        acc = (acc << depth) | s; n += depth
        if n == 8:
            out.append(acc); acc = 0; n = 0
    if n:
        out.append(acc << (8 - n))
    return bytes(out)

def _paeth(a, b, c):
    p = a + b - c; pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)

def _filter_rows(rows, bpp, rng, fixed=None):
    out = bytearray(); prev = bytes(len(rows[0]))
    for row in rows:
        ft = fixed if fixed is not None else int(rng.integers(0, 5))
        out.append(ft)
        r = bytearray(row)
        for i in range(len(row)):
            a = row[i - bpp] if i >= bpp else 0
            b = prev[i]; c = prev[i - bpp] if i >= bpp else 0
            if ft == 0: v = row[i]
            elif ft == 1: v = row[i] - a
            elif ft == 2: v = row[i] - b
            elif ft == 3: v = row[i] - ((a + b) >> 1)
            else: v = row[i] - _paeth(a, b, c)
            out.append(v & 255)
        prev = row
    return bytes(out)

ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]

def png(samples, w, h, ctype, depth, interlace=0, palette=None, trns=None, seed=0, extra=(), fixed=None, level=6, split=0):
    """samples: array [h, w, nchan] of ints (palette index / gray / RGB / ...)."""
    rng = np.random.default_rng(seed)
    nch = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    bpp = max(1, nch * depth // 8)
    def rows_of(arr):
        return [_pack_row([int(v) for v in arr[y].reshape(-1)], depth) for y in range(arr.shape[0])]
    raw = bytearray()
    if interlace:
        for (x0, y0, dx, dy) in ADAM7:
            sub = samples[y0::dy, x0::dx]
            if sub.shape[0] == 0 or sub.shape[1] == 0: continue
            raw += _filter_rows(rows_of(sub), bpp, rng, fixed)
    else:
        raw += _filter_rows(rows_of(samples), bpp, rng, fixed)
    comp = zlib.compress(bytes(raw), level)
    out = b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, interlace))
    if palette is not None: out += _chunk(b"PLTE", bytes(palette))
    if trns is not None: out += _chunk(b"tRNS", trns)
    for t, d in extra: out += _chunk(t, d)
    if split:
        for i in range(0, len(comp), split): out += _chunk(b"IDAT", comp[i:i + split])
    else:
        out += _chunk(b"IDAT", comp)
    return out + _chunk(b"IEND", b"")
