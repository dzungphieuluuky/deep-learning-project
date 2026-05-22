from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models.
    Enforces a consistent interface across experiments.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            batch: Dictionary containing model inputs
        Returns:
            Dictionary containing model outputs (must include 'logits' or 'loss')
        """
        ...

    @abstractmethod
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss from model outputs and batch.

        Returns:
            Dict with at least {'loss': scalar_tensor}
            Can include additional loss components for logging.
        """
        ...

    def configure_optimizers(
        self,
        optimizer_cfg: DictConfig,
        scheduler_cfg: Optional[DictConfig] = None,
    ) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        """Default optimizer configuration. Override for custom behavior."""
        from hydra.utils import instantiate

        # Allow weight decay exclusion for certain parameter groups
        no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
        params = [
            {
                "params": [p for n, p in self.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": optimizer_cfg.get("weight_decay", 0.01),
            },
            {
                "params": [p for n, p in self.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = instantiate(optimizer_cfg, params=params)
        scheduler = instantiate(scheduler_cfg, optimizer=optimizer) \
            if scheduler_cfg else None

        return optimizer, scheduler

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def get_model_summary(self) -> str:
        params = self.count_parameters()
        return (
            f"Model: {self.__class__.__name__}\n"
            f"  Total params:     {params['total']:,}\n"
            f"  Trainable params: {params['trainable']:,}\n"
        )
