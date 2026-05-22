from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchvision
import torchvision.transforms as T
from omegaconf import DictConfig

from .base import BaseDataModule, BaseDataset
from project_name.models.registry import DATAMODULE_REGISTRY


def build_transforms(augmentation_cfg: List[Dict]) -> T.Compose:
    """Dynamically build torchvision transforms from config."""
    transforms = []
    for aug in augmentation_cfg:
        aug = dict(aug)
        name = aug.pop("name")
        transform_cls = getattr(T, name)
        transforms.append(transform_cls(**aug))
    return T.Compose(transforms)


@DATAMODULE_REGISTRY.register("cifar10")
class CIFAR10DataModule(BaseDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.train_transform = build_transforms(cfg.augmentation.train)
        self.val_transform = build_transforms(cfg.augmentation.val)

    def setup(self, stage: str = "fit") -> None:
        if stage == "fit":
            full_train = torchvision.datasets.CIFAR10(
                root=self.cfg.root,
                train=True,
                download=True,
                transform=self.train_transform,
            )
            val_size = int(len(full_train) * self.cfg.val_split)
            train_size = len(full_train) - val_size
            self._train_dataset, self._val_dataset = torch.utils.data.random_split(
                full_train, [train_size, val_size]
            )
            # Override val transform
            self._val_dataset.dataset.transform = self.val_transform

        elif stage == "test":
            self._test_dataset = torchvision.datasets.CIFAR10(
                root=self.cfg.root,
                train=False,
                download=True,
                transform=self.val_transform,
            )


class CustomDataset(BaseDataset):
    """
    Template for custom datasets.
    Replace with your own data loading logic.
    """

    def __init__(self, data_path: str, transform=None):
        self.data_path = Path(data_path)
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Any]:
        # Implement your data loading here
        # e.g., read from CSV, HDF5, JSON, etc.
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        # Process and return sample
        if self.transform:
            sample = self.transform(sample)
        return sample
