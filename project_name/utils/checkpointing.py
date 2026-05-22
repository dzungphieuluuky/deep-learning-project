import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    """Manages saving and loading of model checkpoints."""

    def __init__(
        self,
        checkpoint_dir: str,
        monitor: str = "val/loss",
        mode: str = "min",
        save_top_k: int = 3,
        save_last: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self._best_value = float("inf") if mode == "min" else float("-inf")
        self._saved_checkpoints: list = []

    def is_better(self, value: float) -> bool:
        if self.mode == "min":
            return value < self._best_value
        return value > self._best_value

    def save(
        self,
        state: Dict[str, Any],
        metric_value: float,
        epoch: int,
        step: int,
    ) -> Optional[Path]:
        """Save checkpoint if metric improved."""
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "metric_name": self.monitor,
            "metric_value": metric_value,
            **state,
        }

        # Always save last
        if self.save_last:
            last_path = self.checkpoint_dir / "last.pt"
            torch.save(checkpoint, last_path)

        # Save if best
        if self.is_better(metric_value):
            self._best_value = metric_value
            best_path = self.checkpoint_dir / f"best_epoch{epoch:04d}_step{step:07d}.pt"
            torch.save(checkpoint, best_path)
            self._saved_checkpoints.append((metric_value, best_path))
            self._saved_checkpoints.sort(key=lambda x: x[0],
                                          reverse=(self.mode == "max"))

            # Remove old checkpoints beyond top-k
            while len(self._saved_checkpoints) > self.save_top_k:
                _, old_path = self._saved_checkpoints.pop()
                if old_path.exists():
                    old_path.unlink()

            # Symlink best
            best_link = self.checkpoint_dir / "best.pt"
            if best_link.exists() or best_link.is_symlink():
                best_link.unlink()
            best_link.symlink_to(best_path.name)

            return best_path
        return None

    def find_last_checkpoint(self) -> Optional[Path]:
        last_path = self.checkpoint_dir / "last.pt"
        if last_path.exists():
            return last_path
        return None

    @staticmethod
    def load(path: str, map_location: str = "cpu") -> Dict[str, Any]:
        return torch.load(path, map_location=map_location)

    def get_best_value(self) -> float:
        return self._best_value
