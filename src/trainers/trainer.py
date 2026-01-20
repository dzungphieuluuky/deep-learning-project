import lightning as L
from typing import Optional

class Trainer:
    def __init__(self, max_epochs: int = 10, gpus: int = 1, strategy: str = "auto"):
        self.max_epochs = max_epochs
        self.gpus = gpus
        self.strategy = strategy
    
    def train(self, model: L.LightningModule, datamodule: L.LightningDataModule, callbacks: Optional[list] = None):
        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            devices=self.gpus,
            accelerator="auto",
            strategy=self.strategy,
            callbacks=callbacks or [],
            enable_progress_bar=True,
            log_every_n_steps=10
        )
        trainer.fit(model, datamodule=datamodule)
        return trainer
    
    def test(self, model: L.LightningModule, datamodule: L.LightningDataModule):
        trainer = L.Trainer(devices=self.gpus, accelerator="auto")
        return trainer.test(model, datamodule=datamodule)