"""Finds the accumulation steps of torch/cuDNN's convolutions on this GPU, by probing.

A probe has three nonzero products: +B at K position p, -B at the fixed position z, +1 at j (B = 2**30). The tensor-core
chain returns 1 only when j is added in a later step than p and z (inside one mma instruction the 1 is truncated away
next to 2**30; added before p or z it is absorbed by fp32): After[p][j] = [step(j) > step(p)] for z in the first step.
The input row carries the signs (+B on the set A, -B at z, 1 elsewhere), each output channel selects (p, j) with weights.

  python3 probe_conv.py conv3d|conv2d ROWS OUT.json

Prints the steps found (positions, in the order the kernel adds them) and writes them to OUT.json.
"""
import json, sys, time
import numpy as np
import torch
import torch.nn.functional as F

B = float(2 ** 30)
dev = torch.device("cuda")
bf = torch.bfloat16
kind = R = None
Z = 0
K = OC = 0
run = None


def setup(kind_, R_):
    global kind, R, K, OC, run
    kind, R = kind_, R_
    if kind == "conv3d":
        K, OC = 1176, 1024

        def run_(x, w):   # x [R, K] patch rows (c,t,y,x), w [OC, K]
            return F.conv3d(x.view(-1, 3, 2, 14, 14), w.view(OC, 3, 2, 14, 14), None, stride=(2, 14, 14)).view(-1, OC)
    else:
        K, OC = 4096, 4096

        def run_(x, w):   # positions u = (kh, kw, c) in the input's memory order; the weight tensor is [oc, c, kh, kw]
            xin = x.view(-1, 2, 2, 1024).permute(0, 3, 1, 2)
            wt = w.view(OC, 2, 2, 1024).permute(0, 3, 1, 2).contiguous()
            return F.conv2d(xin, wt, None, stride=2).view(-1, OC)
    run = run_

def after_matrix(A, T, z):
    """After[a, t] for p = A[a], j = T[t] (A, T disjoint position lists, z in neither)."""
    x = torch.ones(K, dtype=bf)
    x[A] = B
    x[z] = -B
    xs = x.expand(R, K).contiguous().to(dev)
    nA, nT = len(A), len(T)
    pa = torch.tensor(A, device=dev).repeat_interleave(nT)
    jt = torch.tensor(T, device=dev).repeat(nA)
    total = nA * nT
    res = np.zeros(total, np.int8)
    for s in range(0, total, OC):
        n = min(OC, total - s)
        w = torch.zeros(OC, K, dtype=bf, device=dev)
        idx = torch.arange(n, device=dev)
        w[idx, pa[s:s + n]] = 1.0
        w[idx, jt[s:s + n]] = 1.0
        w[:, z] = 1.0
        out = run(xs, w)[0, :n].float()
        res[s:s + n] = out.round().to(torch.int8).cpu().numpy()
    return res.reshape(nA, nT)

def analyse():
    global Z
    t0 = time.time()
    pos = [k for k in range(K) if k != Z]
    even = [k for k in pos if k % 2 == 0]
    odd = [k for k in pos if k % 2 == 1]
    # After[p][j]: rows p odd, cols j even; and rows p even, cols j odd
    A_oe = after_matrix(odd, even, Z)
    A_eo = after_matrix(even, odd, Z)
    vals = set(np.unique(A_oe).tolist()) | set(np.unique(A_eo).tolist())
    print("probe values seen:", sorted(vals), "in", round(time.time() - t0, 1), "s", flush=True)
    # rank of each position = number of opposite-parity positions strictly before it
    rank = {}
    for ji, j in enumerate(even):
        rank[j] = int(A_oe[:, ji].sum())      # number of odd p with step(p) < step(j)
    for ji, j in enumerate(odd):
        rank[j] = int(A_eo[:, ji].sum())
    # link an even j and an odd p of one step: neither is after the other
    nb = {}
    for ai, p in enumerate(odd):
        row_after = A_oe[ai]                    # j after p?
        for ji, j in enumerate(even):
            if row_after[ji] == 0 and A_eo[ji][ai] == 0:   # p not before j, and j (as p of the other matrix) not before p
                nb.setdefault(p, []).append(j)
                nb.setdefault(j, []).append(p)
    # union-find over the "same step" links
    parent = {k: k for k in pos}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, bs in nb.items():
        for b_ in bs:
            parent[find(a)] = find(b_)
    comp = {}
    for k in pos:
        comp.setdefault(find(k), []).append(k)
    groups = list(comp.values())
    # order a group by the opposite-parity count of the members (even members' rank counts odd predecessors)
    def key(g):
        ev = [rank[k] for k in g if k % 2 == 0]
        od = [rank[k] for k in g if k % 2 == 1]
        return (np.mean(ev) if ev else 0) + (np.mean(od) if od else 0)
    groups.sort(key=key)
    groups = [sorted(g) for g in groups]
    groups[0] = sorted(groups[0] + [Z])
    print("z =", Z, "steps:", len(groups), "sizes:", sorted(set(len(g) for g in groups)), "first:", groups[:2], "last:", groups[-1])
    print("covered", sum(len(g) for g in groups) + 1, "of", K)
    return groups



if __name__ == "__main__":
    setup(sys.argv[1], int(sys.argv[2]))
    groups = analyse()
    json.dump({"kind": kind, "rows": R, "K": K, "z": Z, "groups": groups}, open(sys.argv[3], "w"))
