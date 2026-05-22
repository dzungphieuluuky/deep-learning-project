from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig


class BaseWorkspace:
    """Owns model, optimizer, scheduler, and training state for a run."""

    SERIALIZED_STATE_KEYS = (
        "current_epoch",
        "global_step",
        "best_metric",
        "patience_counter",
    )

    def __init__(self, cfg: DictConfig, model: nn.Module):
        self.cfg = cfg
        self.model = model

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        self.optimizer, self.scheduler = self._configure_optimizers()

        # Mutable state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")
        self.patience_counter = 0

    def _configure_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        optimizer, scheduler = self.model.configure_optimizers(
            self.cfg.optimizer,
            self.cfg.get("scheduler"),
        )
        return optimizer, scheduler

    def state_dict(self) -> Dict[str, Any]:
        state = {k: getattr(self, k) for k in self.SERIALIZED_STATE_KEYS}
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "cfg": self.cfg,
            "state": state,
        }

    def load_state_dict(self, checkpoint: Dict[str, Any]) -> None:
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)

        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is not None and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)

        state = checkpoint.get("state", {})
        for key in self.SERIALIZED_STATE_KEYS:
            if key in state:
                setattr(self, key, state[key])
