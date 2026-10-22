# A small baseline JPEG writer with free sampling factors, scan layouts, colour-space markers and restart
# intervals: the shapes Pillow cannot write (4:4:0, 4:1:0, odd per-component factors, non-interleaved scans,
# RGB component ids, missing JFIF).
import io, struct, sys
import numpy as np

ZZ = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
      35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]
C = np.array([[np.sqrt(1 / 8) if k == 0 else np.sqrt(2 / 8) * np.cos((2 * n + 1) * k * np.pi / 16) for n in range(8)] for k in range(8)])

def fdct(block):
    return C @ block @ C.T

class Bits:
    def __init__(self): self.out = bytearray(); self.acc = 0; self.n = 0
    def put(self, code, length):
        self.acc = (self.acc << length) | code; self.n += length
        while self.n >= 8:
            b = (self.acc >> (self.n - 8)) & 0xff; self.out.append(b)
            if b == 0xff: self.out.append(0)
            self.n -= 8
        self.acc &= (1 << self.n) - 1
    def flush(self):
        if self.n:
            self.put((1 << (8 - self.n)) - 1, 8 - self.n)
        self.acc = 0; self.n = 0

def cat(v):
    return int(abs(v)).bit_length()

def enc_val(bits, v, s):
    bits.put(v if v >= 0 else v + (1 << s) - 1, s) if s else None

# tables: DC: 12 symbols of length 5 (codes 0..11); AC: 256 possible (run,size) symbols, the 162 valid ones of length 8
AC_SYMS = [0x00, 0xf0] + [(r << 4) | s for r in range(16) for s in range(1, 11)]
AC_SYMS.sort()
AC_CODE = {s: i for i, s in enumerate(AC_SYMS)}

def encode_block(bits, q, pred, blk, quant):
    f = fdct(blk - 128.0)
    coef = np.rint(f / quant).astype(int)
    z = [coef.reshape(-1)[ZZ[i]] for i in range(64)]
    diff = z[0] - pred
    s = cat(diff)
    bits.put(s, 5); enc_val(bits, diff, s)
    run = 0
    last = max([i for i in range(1, 64) if z[i] != 0], default=0)
    for i in range(1, last + 1):
        if z[i] == 0: run += 1; continue
        while run > 15:
            bits.put(AC_CODE[0xf0], 8); run -= 16
        s = cat(z[i]); bits.put(AC_CODE[(run << 4) | s], 8); enc_val(bits, z[i], s); run = 0
    if last < 63: bits.put(AC_CODE[0x00], 8)
    return z[0]

def seg(marker, payload):
    return b"\xff" + bytes([marker]) + struct.pack(">H", len(payload) + 2) + payload

def encode(rgb, samp, ids=(1, 2, 3), jfif=True, adobe=None, restart=0, scans="interleaved", quant_v=3, ycc=True, gray=False):
    """rgb: [h, w, 3] uint8. samp: [(h, v)] per component."""
    H, W, _ = rgb.shape
    img = rgb.astype(float)
    if gray:
        planes = [0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]]
    elif ycc:
        planes = [0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2],
                  -0.168736 * img[..., 0] - 0.331264 * img[..., 1] + 0.5 * img[..., 2] + 128,
                  0.5 * img[..., 0] - 0.418688 * img[..., 1] - 0.081312 * img[..., 2] + 128]
    else:
        planes = [img[..., 0], img[..., 1], img[..., 2]]
    n = len(planes)
    hmax = max(h for h, v in samp); vmax = max(v for h, v in samp)
    mx = -(-W // (8 * hmax)); my = -(-H // (8 * vmax))
    comps = []
    for (h, v), pl in zip(samp, planes):
        fh, fv = hmax // h, vmax // v
        full = np.pad(pl, ((0, my * 8 * vmax - H), (0, mx * 8 * hmax - W)), mode="edge")
        sub = full.reshape(full.shape[0] // fv, fv, full.shape[1] // fh, fh).mean(axis=(1, 3))
        comps.append((h, v, sub))
    out = bytearray(b"\xff\xd8")
    if jfif: out += seg(0xe0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    if adobe is not None: out += seg(0xee, b"Adobe" + struct.pack(">HHHB", 100, 0, 0, adobe))
    quant = np.full((8, 8), float(quant_v))
    out += seg(0xdb, bytes([0]) + bytes([quant_v] * 64))
    out += seg(0xc0, struct.pack(">BHHB", 8, H, W, n) + b"".join(bytes([ids[i], (comps[i][0] << 4) | comps[i][1], 0]) for i in range(n)))
    dc_counts = [0] * 4 + [12] + [0] * 11
    out += seg(0xc4, bytes([0x00]) + bytes(dc_counts) + bytes(range(12)))
    ac_counts = [0] * 7 + [len(AC_SYMS)] + [0] * 8
    out += seg(0xc4, bytes([0x10]) + bytes(ac_counts) + bytes(AC_SYMS))
    if restart: out += seg(0xdd, struct.pack(">H", restart))
    def blocks_of(ci, by, bx):
        h, v, sub = comps[ci]
        return sub[by * 8:(by + 1) * 8, bx * 8:(bx + 1) * 8]
    def scan(comp_ids, interleaved):
        sos = bytes([len(comp_ids)]) + b"".join(bytes([ids[i], 0x00]) for i in comp_ids) + bytes([0, 63, 0])
        s = bytearray(seg(0xda, sos))
        bits = Bits(); pred = [0] * n; count = 0; rst = 0
        if interleaved:
            units = [(m_y, m_x) for m_y in range(my) for m_x in range(mx)]
        else:
            h0, v0, sub = comps[comp_ids[0]]
            # a single-component scan covers the component's own blocks (real size, not MCU padded)
            cw = -(-(-(-W * h0 // hmax)) // 8); ch = -(-(-(-H * v0 // vmax)) // 8)
            units = [(y, x) for y in range(ch) for x in range(cw)]
        for u in units:
            if restart and count and count % restart == 0:
                bits.flush(); s += bits.out; bits = Bits(); s += bytes([0xff, 0xd0 + rst]); rst = (rst + 1) & 7; pred = [0] * n
            count += 1
            for ci in comp_ids:
                h, v, sub = comps[ci]
                if interleaved:
                    for by in range(v):
                        for bx in range(h):
                            pred[ci] = encode_block(bits, None, pred[ci], blocks_of(ci, u[0] * v + by, u[1] * h + bx), quant)
                else:
                    pred[ci] = encode_block(bits, None, pred[ci], blocks_of(ci, u[0], u[1]), quant)
        bits.flush(); s += bits.out
        return bytes(s)
    if scans == "interleaved":
        out += scan(list(range(n)), True if n > 1 else False) if n > 1 else scan([0], False)
    else:
        for ci in range(n): out += scan([ci], False)
    out += b"\xff\xd9"
    return bytes(out)
