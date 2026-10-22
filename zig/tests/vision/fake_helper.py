"""A stand-in for tools/zig/vision/glm_native_helper.py (no torch): the same frames, deterministic answers, for the Zig
server's CPU tests of the helper protocol. Features are row r, column c = bf16 pattern (r * 7 + c) & 0x7fff."""
import json
import struct
import sys

out = sys.stdout.buffer


def write(header, payload=()):
    head = json.dumps(header, separators=(",", ":")).encode()
    out.write(struct.pack("<I", len(head)) + head + struct.pack("<Q", sum(len(p) for p in payload)))
    for p in payload:
        out.write(p)
    out.flush()


def read(n):
    d = b""
    while len(d) < n:
        c = sys.stdin.buffer.read(n - len(d))
        if not c:
            return None
        d += c
    return d


width = 8
write({"ready": True, "pid": 1, "videos": True, "tower_bytes": 1 << 20, "width": width, "versions": {}})
while True:
    raw = read(4)
    if raw is None:
        break
    req = json.loads(read(struct.unpack("<I", raw)[0]))
    if "FAIL" in req["prompt"]:
        write({"ok": False, "status": 400, "error": "image bytes are invalid or unsupported; use JPEG, PNG or WebP"})
        continue
    media = req["media"]
    # one placeholder row per media item at 1, 3, 5 ...; the prompt is rows[-1] + 3 tokens, 9 at the rows
    rows = [1 + 2 * k for k in range(len(media))]
    ids = [100 + i for i in range(rows[-1] + 3)]
    for r in rows:
        ids[r] = 9
    marks = req["prompt"].count("\x00tf-media-" + req["nonce"])
    ids[0] = marks  # the number of marks the server rendered
    feats = b"".join(struct.pack("<H", (r * 7 + c) & 0x7FFF) for r in range(len(rows)) for c in range(width))
    tokens = struct.pack("<%dI" % len(ids), *ids)
    positions = struct.pack("<%dI" % len(rows), *rows)
    write({"ok": True, "tokens": len(ids), "rows": len(rows), "width": width, "images": len(media), "videos": 0,
           "visual_tokens": len(rows), "features_sha256": "00" * 32, "prepare_s": 0.0, "encode_s": 0.0, "peak": 0},
          [tokens, positions, feats])
