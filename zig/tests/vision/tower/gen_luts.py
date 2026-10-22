"""F.silu and F.gelu (exact erf) of every bf16 value, as torch computes them on the GPU: silu_gelu.bin = 65536 silu
results then 65536 gelu results, little-endian uint16 bf16 bit patterns indexed by the input's bit pattern.
Both functions of a bf16 tensor are functions of the 16-bit input, so a table reproduces torch's bits exactly whatever
libm the kernel was compiled with.   python3 gen_luts.py OUT.bin"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
x = bits.view(torch.bfloat16).cuda()
silu = F.silu(x).view(torch.int16).cpu().numpy().astype(np.uint16)
gelu = F.gelu(x).view(torch.int16).cpu().numpy().astype(np.uint16)
open(sys.argv[1], "wb").write(silu.tobytes() + gelu.tobytes())
print("wrote", sys.argv[1], silu.shape, gelu.shape)
