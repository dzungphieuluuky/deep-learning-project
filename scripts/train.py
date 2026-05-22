from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from project_name.utils.reproducibility import set_seed, set_precision
from project_name.utils.logging import ExperimentLogger, get_logger
from project_name.utils.checkpointing import CheckpointManager
from project_name.trainers.supervised import SupervisedTrainer
from project_name.workspace.base import BaseWorkspace

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def train(cfg: DictConfig) -> None:
    # ── Setup ────────────────────────────────────────────────────
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    set_seed(cfg.seed)
    set_precision(cfg.training.precision)

    # ── Instantiate Components ───────────────────────────────────
    model = instantiate(cfg.model)
    data_module = instantiate(cfg.data)
    exp_logger = ExperimentLogger(cfg)

    workspace = BaseWorkspace(cfg, model)

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=cfg.paths.checkpoints,
        monitor=cfg.training.early_stopping.monitor,
        mode=cfg.training.early_stopping.mode,
        save_top_k=3,
    )

    resume_path = cfg.training.get("resume_from", None)
    if resume_path is None and cfg.training.get("auto_resume", True):
        resume_path = checkpoint_manager.find_last_checkpoint()

    if resume_path:
        checkpoint = checkpoint_manager.load(str(resume_path))
        workspace.load_state_dict(checkpoint)
        log.info(f"Auto-resumed from {resume_path}")

    log.info(f"Model: {model.__class__.__name__}")
    log.info(f"Data:  {data_module.__class__.__name__}")

    # ── Train ───────────────────────────────────────────────────
    trainer = SupervisedTrainer(
        cfg=cfg,
        workspace=workspace,
        data_module=data_module,
        logger=exp_logger,
    )
    trainer.fit()


if __name__ == "__main__":
    train()
