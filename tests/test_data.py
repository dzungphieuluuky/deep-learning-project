import pytest
from omegaconf import OmegaConf
from project_name.data.datasets import CIFAR10DataModule


@pytest.fixture
def data_cfg(tmp_path):
    return OmegaConf.create({
        "root": str(tmp_path),
        "batch_size": 4,
        "num_workers": 0,
        "pin_memory": False,
        "val_split": 0.1,
        "augmentation": {
            "train": [
                {"name": "ToTensor"},
            ],
            "val": [
                {"name": "ToTensor"},
            ],
        },
    })


def test_datamodule_setup(data_cfg):
    dm = CIFAR10DataModule(data_cfg)
    dm.setup("fit")
    assert dm._train_dataset is not None
    assert dm._val_dataset is not None


def test_dataloader_batch_shape(data_cfg):
    dm = CIFAR10DataModule(data_cfg)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    # CIFAR10 returns (image, label)
    assert len(batch) == 2
