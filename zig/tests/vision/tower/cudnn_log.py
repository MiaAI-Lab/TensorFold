"""One conv3d / conv2d at the tower's shapes, for CUDNN_LOGINFO_DBG logs (which engine cuDNN picks per shape)."""
import torch, torch.nn.functional as F
dev = torch.device("cuda")
for R in (80, 8192):
    x = torch.randn(R, 1176, device=dev).to(torch.bfloat16); w = torch.randn(1024, 1176, device=dev).to(torch.bfloat16)
    F.conv3d(x.view(-1, 3, 2, 14, 14), w.view(1024, 3, 2, 14, 14), None, stride=(2, 14, 14)); torch.cuda.synchronize()
for M in (20, 2048):
    x = torch.randn(M, 4096, device=dev).to(torch.bfloat16); w = torch.randn(4096, 4096, device=dev).to(torch.bfloat16)
    F.conv2d(x.view(-1, 2, 2, 1024).permute(0, 3, 1, 2), w.view(4096, 2, 2, 1024).permute(0, 3, 1, 2).contiguous(), None, stride=2); torch.cuda.synchronize()
