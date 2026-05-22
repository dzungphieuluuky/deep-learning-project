from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.cuda.amp import GradScaler

from project_name.utils.checkpointing import CheckpointManager
from project_name.utils.logging import ExperimentLogger
from project_name.workspace.base import BaseWorkspace


class BaseTrainer(ABC):
    """
    Abstract trainer — handles the training loop boilerplate.
    Subclass this and implement train_step / val_step.
    """

    def __init__(
        self,
        cfg: DictConfig,
        workspace: BaseWorkspace,
        data_module: Any,
        logger: ExperimentLogger,
    ):
        self.cfg = cfg
        self.workspace = workspace
        self.data_module = data_module
        self.exp_logger = logger

        self.device = self.workspace.device
        self.model = self.workspace.model
        self.optimizer = self.workspace.optimizer
        self.scheduler = self.workspace.scheduler

        # Mixed precision
        self.use_amp = "16" in cfg.training.precision
        self.scaler = GradScaler() if self.use_amp else None

        # Checkpointing
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=cfg.paths.checkpoints,
            monitor=cfg.training.early_stopping.monitor,
            mode=cfg.training.early_stopping.mode,
            save_top_k=3,
        )

        # State is stored in workspace

    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Single training step. Return dict with at least {'loss': tensor}."""
        ...

    @abstractmethod
    def val_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Single validation step. Return dict with metrics."""
        ...

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_metrics: Dict[str, float] = {}

        train_loader = self.data_module.train_dataloader()
        for step, batch in enumerate(train_loader):
            batch = self._to_device(batch)

            # Gradient accumulation
            accumulate = self.cfg.training.accumulate_grad_batches
            is_accumulation_step = (step + 1) % accumulate != 0

            with torch.autocast(
                device_type=self.device.type,
                enabled=self.use_amp,
            ):
                outputs = self.train_step(batch)

            loss = outputs["loss"] / accumulate

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if not is_accumulation_step:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.training.gradient_clip,
                )

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                self.workspace.global_step += 1

            # Logging
            if self.workspace.global_step % self.cfg.training.log_every_n_steps == 0:
                log_metrics = {k: v.item() if isinstance(v, torch.Tensor) else v
                               for k, v in outputs.items()}
                self.exp_logger.log(log_metrics, self.workspace.global_step, prefix="train")

        return epoch_metrics

    @torch.no_grad()
    def val_epoch(self) -> Dict[str, float]:
        self.model.eval()
        all_outputs = []

        for batch in self.data_module.val_dataloader():
            batch = self._to_device(batch)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.val_step(batch)
            all_outputs.append(outputs)

        # Aggregate metrics
        aggregated = {}
        for key in all_outputs[0]:
            values = [o[key] for o in all_outputs]
            if isinstance(values[0], torch.Tensor):
                aggregated[key] = torch.stack(values).mean().item()
            else:
                aggregated[key] = sum(values) / len(values)

        self.exp_logger.log(aggregated, self.workspace.global_step, prefix="val")
        return aggregated

    def fit(self) -> None:
        """Main training loop."""
        self.data_module.setup("fit")
        self.exp_logger.logger.info(
            f"Starting training on {self.device} for "
            f"{self.cfg.training.max_epochs} epochs"
        )
        self.exp_logger.logger.info(self.model.get_model_summary())

        for epoch in range(self.cfg.training.max_epochs):
            self.workspace.current_epoch = epoch
            self.exp_logger.logger.info(f"Epoch {epoch + 1}/{self.cfg.training.max_epochs}")

            train_metrics = self.train_epoch()
            val_metrics = self.val_epoch()

            if self.scheduler:
                self.scheduler.step()

            # Checkpoint
            monitor_value = val_metrics.get(
                self.cfg.training.early_stopping.monitor.split("/")[-1],
                float("inf"),
            )
            self.checkpoint_manager.save(
                state=self.workspace.state_dict(),
                metric_value=monitor_value,
                epoch=epoch,
                step=self.workspace.global_step,
            )

            # Early stopping
            if self.cfg.training.early_stopping.enabled:
                if self._check_early_stopping(monitor_value):
                    self.exp_logger.logger.info(
                        f"Early stopping triggered at epoch {epoch + 1}"
                    )
                    break

        self.exp_logger.finish()

    def _check_early_stopping(self, value: float) -> bool:
        mode = self.cfg.training.early_stopping.mode
        improved = (value < self.workspace.best_metric) if mode == "min" \
               else (value > self.workspace.best_metric)

        if improved:
            self.workspace.best_metric = value
            self.workspace.patience_counter = 0
        else:
            self.workspace.patience_counter += 1

        return self.workspace.patience_counter >= self.cfg.training.early_stopping.patience

    def _to_device(self, batch: Any) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._to_device(x) for x in batch)
        return batch
