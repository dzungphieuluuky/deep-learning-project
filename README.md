# deep_learning_project

Light-weight PyTorch Lightning / Fabric training template with YAML-driven configuration.

## Overview
Minimal project structure for experimenting with deep learning models, data modules, callbacks and trainer configuration. Configuration lives in `configs/` and defaults are provided under `configs/{data,model,trainer}`. Source code is under `src/`.

## Quickstart

1. Create virtual environment and install dependencies (uses `pyproject.toml`):
    - With Poetry:
      ```
      poetry install
      poetry run python train.py --config configs/train.yaml
      ```
    - With pip:
      ```
      python -m venv .venv
      source .venv/bin/activate   # or .venv\Scripts\activate on Windows
      python -m pip install -e .
      python train.py --config configs/train.yaml
      ```

2. Optionally set environment variables in `.env` (e.g., CUDA device, experiment name).

3. Run alternative Fabric script:
    ```
    python scripts/train_fabric.py --config configs/train.yaml
    ```

## Configuration
- Top-level experiment config: `configs/train.yaml`
- Default component configs:
  - `configs/data/default_data.yaml`
  - `configs/model/default_model.yaml`
  - `configs/trainer/default_trainer.yaml`
- Adjust YAMLs to change datasets, architecture, optimizer, scheduler, training hyperparameters, callbacks, and logging.

## Important files
- `train.py` — main entrypoint for training (loads config, builds components, runs training).
- `scripts/train_fabric.py` — alternate training script using PyTorch-Fabric (if present).
- `src/` — package source:
  - `src/data/datamodule_py` — data loading and preprocessing hooks.
  - `src/models/lit_module.py` — LightningModule wrapper (training/validation/test steps).
  - `src/models/components/simple_net.py` — example model implementation.
  - `src/utils/` — utilities and helpers.
- `pyproject.toml` — project metadata and dependencies.
- `.env` — environment overrides (not checked in).

## Extending the template
- Add new models under `src/models/components/` and reference them in `configs/model`.
- Add datamodules under `src/data/` and reference in `configs/data`.
- Add callbacks in `configs/callbacks/` and implement under `src/utils` or `src/callbacks`.

## Running experiments
- Use separate config files for each experiment (copy `configs/train.yaml` and override).
- Track experiments with your preferred logger (integrate in trainer config).
- Resume training by pointing to a checkpoint in trainer config.

## Contributing
- Follow repository layout and add tests for new components.
- Keep configs declarative and small; prefer composition of default YAMLs.

## License
Specify project license in `pyproject.toml` or add a `LICENSE` file.
