"""The CUDA engines' tests: collected only where PyTorch sees an NVIDIA GPU (DGX Spark, in NVIDIA's container)."""

import importlib.util
import os

# GLM's engines here load the MTP head beside the DFlash2 drafter (TF_GLM_MTP=1, the default off beside a drafter),
# so both drafters and the per-round choice between them stay under test; TF_GLM_MTP's own tests set it per engine
os.environ.setdefault("TF_GLM_MTP", "1")


def _cuda() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available()


collect_ignore_glob = [] if _cuda() else ["test_*.py"]
