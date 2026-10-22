import numpy as np

MASK = (1 << 64) - 1

def lcg_bytes(seed: int, n: int) -> bytes:
    """Deterministic test bytes (the same generator is in zig/src/vision/fixtures.zig)."""
    x = seed & MASK
    out = bytearray(n)
    for i in range(n):
        x = (x * 6364136223846793005 + 1442695040888963407) & MASK
        out[i] = x >> 56
    return bytes(out)
