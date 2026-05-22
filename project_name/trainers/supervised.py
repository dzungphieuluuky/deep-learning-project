from typing import Any, Dict

import torch
from omegaconf import DictConfig

from .base import BaseTrainer
from project_name.workspace.base import BaseWorkspace
from project_name.metrics.registry import Accuracy, MetricCollection


class SupervisedTrainer(BaseTrainer):
    """Standard supervised classification/regression trainer."""

    def __init__(self, cfg, workspace: BaseWorkspace, data_module, logger):
        super().__init__(cfg, workspace, data_module, logger)

        self.train_metrics = MetricCollection({
            "acc@1": Accuracy(top_k=1),
            "acc@5": Accuracy(top_k=5),
        })
        self.val_metrics = MetricCollection({
            "acc@1": Accuracy(top_k=1),
            "acc@5": Accuracy(top_k=5),
        })

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outputs = self.model(batch)
        loss_dict = self.model.compute_loss(outputs, batch)
        self.train_metrics.update(outputs, batch)
        return loss_dict

    def val_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        outputs = self.model(batch)
        loss_dict = self.model.compute_loss(outputs, batch)
        self.val_metrics.update(outputs, batch)

        # Compute and reset metrics at end of val
        metric_values = self.val_metrics.compute()
        self.val_metrics.reset()
        return {**loss_dict, **metric_values}
