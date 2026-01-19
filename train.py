import hydra
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from hydra.utils import instantiate
import torch

@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig):
    """
    Main training pipeline.
    1. Instantiates DataModule
    2. Instantiates Model
    3. Instantiates Trainer (with W&B and Callbacks)
    4. Fits
    """
    
    # 1. Set Seed for reproducibility
    pl.seed_everything(cfg.seed)

    # 2. Instantiate DataModule
    print(f"Instantiating DataModule <{cfg.data._target_}>")
    datamodule: pl.LightningDataModule = instantiate(cfg.data)

    # 3. Instantiate Model
    print(f"Instantiating Model <{cfg.model._target_}>")
    model: pl.LightningModule = instantiate(cfg.model)

    # 4. Set up W&B Logger
    logger = WandbLogger(
        project=cfg.project_name,
        name=cfg.get("run_name"),
        tags=cfg.get("tags"),
        log_model=True
    )

    # 5. Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.paths.output_dir,
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            filename="epoch_{epoch:03d}"
        ),
        EarlyStopping(monitor="val/loss", patience=5, mode="min")
    ]

    # 6. Instantiate Trainer
    print(f"Instantiating Trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks
    )

    # 7. Train
    print("Starting training...")
    trainer.fit(model=model, datamodule=datamodule)
    
    print("Training finished.")

if __name__ == "__main__":
    main()