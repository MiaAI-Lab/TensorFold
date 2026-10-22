"""probe_conv.py over many row counts, with the kernel names torch/cuDNN runs and a one-line fingerprint of the K steps.
  python3 probe_sweep.py OUTDIR"""
import json, sys, collections
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
import probe_conv as P

out = sys.argv[1]
cases = {"conv3d": [16, 24, 32, 48, 64, 80, 96, 128, 196, 256, 512, 1024, 2048, 4096, 8192, 8000],
         "conv2d": [4, 6, 8, 12, 16, 20, 24, 36, 49, 64, 100, 128, 196, 256, 400, 512, 784, 1024, 1568, 2048]}
dev = torch.device("cuda")
for kind, Rs in cases.items():
    for R in Rs:
        P.setup(kind, R)
        x = torch.randn(R, P.K, device=dev).to(torch.bfloat16)
        w = torch.randn(P.OC, P.K, device=dev).to(torch.bfloat16)
        P.run(x, w)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            P.run(x, w)
            torch.cuda.synchronize()
        names = [e.name[:140] for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        groups = P.analyse()
        sizes = collections.Counter(len(g) for g in groups)
        print(kind, R, "kernels:", names, "| steps", len(groups), "sizes", dict(sizes), flush=True)
        json.dump({"kind": kind, "rows": R, "groups": groups}, open(f"{out}/{kind}_{R}.json", "w"))
