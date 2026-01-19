# 🚀 deep_learning_project

A lightweight PyTorch Lightning / Fabric training template with YAML-driven configuration — minimal, composable, and ready for experiments.

## ✨ Features
- Declarative configs with Hydra (configs/{data,model,trainer})
- Clear src layout for models, datamodules, and utils
- Easy to extend callbacks, loggers, and training scripts
- Supports both Lightning and Fabric for flexible training

---

## ⚡ Quickstart
1. Create env & install:
   - Poetry:
     ```
     poetry install
     poetry run python train.py --config configs/train.yaml
     ```
   - pip (Windows):
     ```
     python -m venv .venv
     .venv\Scripts\activate
     python -m pip install -e .
     python train.py --config configs/train.yaml
     ```
2. Set secrets / overrides in `.env` (e.g., WANDB_API_KEY, CUDA_VISIBLE_DEVICES).
3. Alternate Fabric runner:
   ```
   python scripts/train_fabric.py --config configs/train.yaml
   ```

---

## 🧭 Project Layout
- **configs/** — Hydra configs & defaults (data, model, trainer, callbacks)
- **src/**
  - **src/data/** — datamodules & loaders (e.g., `datamodule.py`)
  - **src/models/** — model implementations & Lightning wrappers (e.g., `lit_module.py`, `components/simple_net.py`)
  - **src/utils/** — instantiation, logging, helpers
- **scripts/** — auxiliary scripts (e.g., Fabric loops)
- **train.py** — main training entrypoint
- **pyproject.toml** — deps & metadata
- **.env** — local overrides (not checked in)

---

## 🛠️ Extending the Template
- Add models under `src/models/components/` and reference in `configs/model`
- Add datamodules under `src/data/` and reference in `configs/data`
- Keep configs small and composable; compose defaults into experiment YAMLs

---

## 📚 Inspiration & Further Reading
- Lightning (training primitives): https://github.com/Lightning-AI/lightning
- Hydra (config composition): https://github.com/facebookresearch/hydra
- lightning-hydra-template (example template): https://github.com/ashleve/lightning-hydra-template
- Transformers (project structure & best practices): https://github.com/huggingface/transformers

---

## 📝 Contributing
- Follow the structure, add tests for new components, and keep configs declarative.

© Specify license in pyproject.toml or add a LICENSE file.
