"""Does the nvcc that built torch (13.3) give torch's GELU bits with the plain fp32 formula? Compiles the candidate kernel
here and compares it with F.gelu / F.silu over every bf16 value.  python3 lab_gelu.py KERNEL.cu"""
import ctypes, subprocess, sys, os
import numpy as np
import torch
import torch.nn.functional as F
src = sys.argv[1]
os.makedirs("/tmp/lab", exist_ok=True)
wrap = r'''
#include "%s"
extern "C" void run_act(void* out, int mode, void* stream) { vis_act_candidates<<<256, 256, 0, (cudaStream_t)stream>>>((__nv_bfloat16*)out, mode); }
''' % os.path.abspath(src)
open("/tmp/lab/wrap.cu", "w").write(wrap)
subprocess.check_call(["nvcc", "-O3", "-arch=sm_121", "-shared", "-Xcompiler", "-fPIC", "-o", "/tmp/lab/act.so", "/tmp/lab/wrap.cu"])
lib = ctypes.CDLL("/tmp/lab/act.so")
bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
x = bits.view(torch.bfloat16).cuda()
gelu = F.gelu(x).view(torch.int16); silu = F.silu(x).view(torch.int16)
out = torch.empty(65536, dtype=torch.bfloat16, device="cuda")
for mode, name, ref in ((0, "gelu fp32", gelu), (3, "silu fp32", silu), (5, "gelu erfc", gelu), (8, "gelu unfused", gelu)):
    lib.run_act(ctypes.c_void_p(out.data_ptr()), mode, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    bad = (out.view(torch.int16) != ref).sum().item()
    print(f"{name}: wrong on {bad} of 65536 inputs (nvcc built by the container)")
