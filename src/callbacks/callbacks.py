import lightning as L
from lightning import Callback

class CustomCallback(Callback):
    def on_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        current_epoch = trainer.current_epoch
        if (current_epoch + 1) % 5 == 0:
            print(f"Completed epoch {current_epoch + 1}")
    
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        print("Training started!")
    
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        print("Training completed!")