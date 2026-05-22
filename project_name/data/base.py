from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split


class BaseDataset(Dataset, ABC):
    """Abstract base dataset."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]: ...


class BaseDataModule(ABC):
    """
    Abstract data module — wraps train/val/test datasets
    and their respective dataloaders.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None

    @abstractmethod
    def setup(self, stage: str = "fit") -> None:
        """
        Called before dataloaders are created.
        stage: "fit" | "test" | "predict"
        """
        ...

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
