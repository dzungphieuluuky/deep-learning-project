from typing import Any, Dict, Tuple
import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from hydra.utils import instantiate

class ResearchLitModule(pl.LightningModule):
    """
    Standard LightningModule.
    Decouples architecture (self.net) from training logic.
    """

    def __init__(
        self, 
        net: torch.nn.Module, 
        optimizer_config: Dict[str, Any],
        scheduler_config: Dict[str, Any]
    ):
        """
        :param net: The pure PyTorch model (instantiated by Hydra).
        :param optimizer_config: Config for the optimizer.
        :param scheduler_config: Config for the scheduler.
        """
        super().__init__()
        # Saves arguments to 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        
        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        # Lightning handles logging automatically to W&B if configured
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)

    def configure_optimizers(self):
        """
        Instantiate optimizers/schedulers from Hydra config.
        """
        # Partial instantiation allows passing model params
        opt = instantiate(self.hparams.optimizer_config, params=self.parameters())
        sched = instantiate(self.hparams.scheduler_config, optimizer=opt)
        
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }