import os
import random
import numpy as np
import torch
from omegaconf import DictConfig


def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def set_precision(precision: str) -> None:
    """Configure floating point precision."""
    if "bf16" in precision:
        torch.set_float32_matmul_precision("medium")
    elif "16" in precision:
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("highest")
