import random
import numpy as np
import torch
from torch.optim import AdamW

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_optimizer(model, learning_rate: float = 2e-5, weight_decay: float = 0.01):
    """Create AdamW optimizer"""
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer