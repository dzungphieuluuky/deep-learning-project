from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
from omegaconf import DictConfig
from hydra.utils import instantiate

from project_name.utils.logging import get_logger
from project_name.utils.checkpointing import CheckpointManager
from project_name.workspace.base import BaseWorkspace

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def evaluate(cfg: DictConfig) -> None:
    # Load checkpoint
    checkpoint_path = cfg.get("checkpoint_path", None)
    assert checkpoint_path, "Must provide checkpoint_path"

    checkpoint = CheckpointManager.load(checkpoint_path)
    log.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
             f"step {checkpoint['step']}")

    # Instantiate model
    model = instantiate(cfg.model)
    workspace = BaseWorkspace(cfg, model)
    workspace.load_state_dict(checkpoint)
    model = workspace.model
    model.eval()

    # Instantiate data
    data_module = instantiate(cfg.data)
    data_module.setup("test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Evaluate
    all_metrics = []
    with torch.no_grad():
        for batch in data_module.test_dataloader():
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            outputs = model(batch)
            all_metrics.append(outputs)

    log.info("Evaluation complete.")


if __name__ == "__main__":
    evaluate()
