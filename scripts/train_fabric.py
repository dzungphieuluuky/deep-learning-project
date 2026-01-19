import hydra
from omegaconf import DictConfig
import lightning as L
import torch
from hydra.utils import instantiate

@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def run_fabric(cfg: DictConfig):
    # 1. Setup Fabric (Handles devices, precision, DDP)
    fabric = L.Fabric(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision
    )
    fabric.launch()

    # 2. Manual Instantiation
    # Note: We instantiate the inner 'net', not the LightningModule
    model = instantiate(cfg.model.net)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.model.optimizer_config.lr)
    
    # 3. Fabric setup (magic happens here)
    model, optimizer = fabric.setup(model, optimizer)
    
    datamodule = instantiate(cfg.data)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    # Fabric setup dataloader for distributed sampling
    train_loader = fabric.setup_dataloaders(train_loader)

    model.train()
    
    # 4. Manual Loop
    for epoch in range(cfg.trainer.max_epochs):
        for batch in train_loader:
            x, y = batch
            
            # No need for .to(device), Fabric handles it
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            
            # Manual backward with Fabric
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            if fabric.is_global_zero:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    run_fabric()