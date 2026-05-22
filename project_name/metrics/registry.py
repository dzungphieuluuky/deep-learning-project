from typing import Any, Dict, List, Optional

import torch
from project_name.models.registry import METRIC_REGISTRY


class MetricCollection:
    """Manages a collection of metrics for train/val/test."""

    def __init__(self, metrics: Dict[str, Any]):
        self.metrics = metrics

    def update(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> None:
        for metric in self.metrics.values():
            metric.update(outputs, batch)

    def compute(self) -> Dict[str, float]:
        results = {}
        for name, metric in self.metrics.items():
            value = metric.compute()
            if isinstance(value, dict):
                results.update({f"{name}/{k}": v for k, v in value.items()})
            else:
                results[name] = value
        return results

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()


class Accuracy:
    """Top-k accuracy metric."""

    def __init__(self, top_k: int = 1):
        self.top_k = top_k
        self.correct = 0
        self.total = 0

    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
    ) -> None:
        logits = outputs["logits"]
        labels = batch["labels"]
        _, predicted = logits.topk(self.top_k, dim=-1)
        correct = predicted.eq(labels.unsqueeze(-1).expand_as(predicted))
        self.correct += correct.any(dim=-1).sum().item()
        self.total += labels.size(0)

    def compute(self) -> float:
        return self.correct / max(self.total, 1)

    def reset(self) -> None:
        self.correct = 0
        self.total = 0


METRIC_REGISTRY.register("accuracy")(Accuracy)
