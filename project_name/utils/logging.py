import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import wandb
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.logging import RichHandler


console = Console()


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get a rich-formatted logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    return logging.getLogger(name)


class ExperimentLogger:
    """Unified logger that wraps W&B, TensorBoard, and console."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = get_logger(__name__, cfg.logging.console_level)
        self._wandb_run = None
        self._step = 0

        if cfg.logging.use_wandb and not cfg.debug:
            self._init_wandb()

    def _init_wandb(self) -> None:
        self._wandb_run = wandb.init(
            project=self.cfg.logging.wandb.project,
            entity=self.cfg.logging.wandb.entity,
            name=self.cfg.run_name,
            config=OmegaConf.to_container(self.cfg, resolve=True),
            tags=self.cfg.logging.wandb.tags,
            notes=self.cfg.logging.wandb.notes,
        )
        self.logger.info(f"W&B run initialized: {self._wandb_run.url}")

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        """Log metrics to all enabled backends."""
        step = step or self._step
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Console
        metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()
                                  if isinstance(v, (int, float)))
        self.logger.info(f"Step {step} | {metrics_str}")

        # W&B
        if self._wandb_run is not None:
            wandb.log(metrics, step=step)

        self._step += 1

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        if self._wandb_run is not None:
            wandb.config.update(params)

    def log_artifact(self, path: str, name: str, artifact_type: str) -> None:
        if self._wandb_run is not None:
            artifact = wandb.Artifact(name=name, type=artifact_type)
            artifact.add_file(path)
            self._wandb_run.log_artifact(artifact)

    def finish(self) -> None:
        if self._wandb_run is not None:
            wandb.finish()
