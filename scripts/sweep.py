"""
Example W&B sweep configuration and runner.

sweep_config.yaml:
    program: scripts/train.py
    method: bayes
    metric:
        name: val/loss
        goal: minimize
    parameters:
        model.hidden_dim:
            values: [128, 256, 512]
        experiment.optimizer.lr:
            distribution: log_uniform_values
            min: 1e-5
            max: 1e-2
        model.num_layers:
            values: [4, 6, 8]
        model.dropout:
            values: [0.0, 0.1, 0.2]
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import subprocess
import wandb


def run_sweep():
    import yaml
    with open("sweep_config.yaml") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_config, project="ml_research")

    def agent_fn():
        with wandb.init() as run:
            config = run.config
            cmd = [
                "python", "scripts/train.py",
                f"run_name=sweep_{run.id}",
                f"model.hidden_dim={config.get('model.hidden_dim', 256)}",
                f"experiment.optimizer.lr={config.get('experiment.optimizer.lr', 3e-4)}",
            ]
            subprocess.run(cmd)

    wandb.agent(sweep_id, function=agent_fn, count=20)


if __name__ == "__main__":
    run_sweep()
