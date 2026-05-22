# AI Agent Guidelines for Deep Learning Project

This is a **modular, Hydra-based PyTorch research framework** with composable components via registries. Focus on config-driven development, ABC inheritance, and registry-based instantiation.

## Project Overview

- **Core Packages**: `project_name/` (models, data, trainers, metrics, utils, workspace)
- **Configs**: `configs/` with Hydra YAML composition (base + model/data/experiment groups)
- **Entry Points**: `scripts/train.py`, `scripts/evaluate.py`, `scripts/sweep.py`
- **Tests**: pytest-based test suite with fixtures

## Quick Setup & Commands

```bash
# Setup (one-time)
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
pip install -e .

# Training with Hydra overrides
python scripts/train.py experiment=baseline training.max_epochs=50 model.hidden_dim=512

# Testing
pytest tests/

# All dependencies in requirements.txt (PyTorch 2.0+, Hydra 1.3+, wandb, transformers, accelerate)
```

## Core Architecture Patterns

### **Registry Pattern** (All Components)
Every major component uses dynamic registration with decorators:

```python
@MODEL_REGISTRY.register("transformer")
class TransformerModel(BaseModel): ...

@DATAMODULE_REGISTRY.register("cifar10")
class CIFAR10DataModule(BaseDataModule): ...
```

**Registries**: `MODEL_REGISTRY`, `DATAMODULE_REGISTRY`, `LOSS_REGISTRY`, `OPTIMIZER_REGISTRY`, `METRIC_REGISTRY`

**When adding a new component**: Always use `@REGISTRY.register("name")` + inherit from appropriate base class.

### **ABC Inheritance Hierarchy**
Every component has a required interface:

| Base Class | File | Required Methods | Key Pattern |
|-----------|------|-----------------|-------------|
| `BaseModel` | `models/base.py` | `forward(batch)`, `compute_loss()`, `configure_optimizers()`, `count_parameters()` | Returns `Dict[str, Tensor]` always |
| `BaseDataModule` | `data/base.py` | `setup(stage)`, `train_dataloader()`, `val_dataloader()`, `test_dataloader()` | Configurable workers + pin memory |
| `BaseTrainer` | `trainers/base.py` | `train_step()`, `val_step()`, `train_epoch()` | Mixed precision (AMP) + gradient accumulation built-in |
| `BaseMetric` | Implemented in test utils | `update()`, `compute()`, `reset()` | Accumulates across batches |

### **Hydra Configuration System**
Composition-based config management:

```yaml
# configs/base.yaml (default for all experiments)
defaults:
  - model: transformer      # Selects configs/model/transformer.yaml
  - data: cifar10           # Selects configs/data/cifar10.yaml
  - experiment: baseline    # Selects configs/experiment/baseline.yaml

training:
  max_epochs: 100
  batch_size: 32

# configs/experiment/baseline.yaml (overrides using @package _global_)
# @package _global_
training:
  max_epochs: 50
  lr: 1e-3
```

**Key behaviors**:
- Command-line overrides have highest priority: `python scripts/train.py model.hidden_dim=512`
- Use `@package _global_` in experiment configs to override top-level keys
- Never hardcode values—pull from `cfg` (DictConfig from Hydra)

### **State Management via Workspace**
`BaseWorkspace` owns model, optimizer, scheduler, and training state:

```python
from project_name.workspace.base import BaseWorkspace
workspace = BaseWorkspace.from_config(cfg)
workspace.train()  # Sets model to train mode
checkpoint = workspace.state_dict()  # Serializable state
workspace.load_state_dict(checkpoint)  # Restore
```

## Module Structure & Interfaces

### `project_name/models/`
- **`base.py`**: `BaseModel(nn.Module)` with `compute_loss()`, `configure_optimizers()`
  - Weight decay exclusion: No decay for bias/norm layers (see `configure_optimizers()`)
- **`registry.py`**: `MODEL_REGISTRY` with `build()`, `get()`, `list()`, `__contains__()`
- **`transformer.py`**: Example—Flash Attention support + multi-head attention blocks

**Pattern for new models**: Inherit from `BaseModel`, implement required methods, register with decorator.

### `project_name/data/`
- **`base.py`**: `BaseDataset`, `BaseDataModule` (standard PyTorch interfaces)
- **`datasets.py`**: Concrete implementations (CIFAR10, etc.)

**Pattern for new datasets**: Register with decorator, implement `setup(stage)` and dataloader methods, use config for transforms/split.

### `project_name/trainers/`
- **`base.py`**: `BaseTrainer` (ABC) + `SupervisedTrainer` (concrete)
  - `train_step(batch)` → single batch loss
  - `train_epoch()` → full loop with AMP, gradient accumulation, checkpointing
  - Early stopping via `CheckpointManager`
- **`supervised.py`**: Example trainer for classification

**Pattern for new trainers**: Inherit from `BaseTrainer`, implement `train_step()` and `val_step()`, use workspace for state.

### `project_name/utils/`
- **`logging.py`**: `ExperimentLogger` (wraps wandb + rich console)
  - Usage: `exp_logger.log(metrics, step, prefix="train")`
- **`checkpointing.py`**: `CheckpointManager` (saves best-k models, auto-cleanup)
- **`reproducibility.py`**: `set_seed()`, `set_precision()` for determinism
- **`logging.py`** / **`checkpointing.py`**: Key utilities for common tasks

### `project_name/metrics/`
- **`registry.py`**: `METRIC_REGISTRY` for dynamic metric instantiation

## Common Development Tasks

### Adding a New Model
1. Create file `project_name/models/new_model.py`
2. Inherit from `BaseModel`, implement required methods:
   ```python
   @MODEL_REGISTRY.register("new_model")
   class NewModel(BaseModel):
       def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
           # Return dict with at least 'logits' key
           return {"logits": output}
       def compute_loss(self, outputs, batch):
           # Return dict with 'loss' key
           return {"loss": F.cross_entropy(...)}
       def configure_optimizers(self):
           # Return optimizer, no decay for bias/norm
           return ...
   ```
3. Add config in `configs/model/new_model.yaml`:
   ```yaml
   _target_: project_name.models.new_model.NewModel
   hidden_dim: 256
   num_layers: 4
   ```
4. Use in training: `python scripts/train.py model=new_model`

### Adding a New Dataset
1. Create file `project_name/data/new_dataset.py`
2. Inherit from `BaseDataModule`:
   ```python
   @DATAMODULE_REGISTRY.register("new_dataset")
   class NewDataModule(BaseDataModule):
       def setup(self, stage: str):
           # Initialize train/val/test datasets
           pass
       def train_dataloader(self):
           return DataLoader(self.train_dataset, ...)
   ```
3. Add config in `configs/data/new_dataset.yaml`
4. Use: `python scripts/train.py data=new_dataset`

### Running Experiments with Hydra Overrides
```bash
# Single override
python scripts/train.py training.max_epochs=100

# Multiple overrides
python scripts/train.py experiment=baseline model.hidden_dim=512 data.batch_size=64

# List available options (if script supports)
python scripts/train.py --help  # or use tab completion
```

### Configuration Best Practices
- All hyperparameters in YAML (`configs/` hierarchy)
- Use `instantiate(cfg.model)` to dynamically create components
- Device handling: Access via `workspace.device`, not hardcoded
- Batch format: Always `Dict[str, Tensor]` through pipeline
- Logging: Use `exp_logger.log()` with step-based tracking

## Important Files to Know

| File/Dir | Purpose | Learn From |
|----------|---------|-----------|
| [scripts/train.py](scripts/train.py) | Main entry point | Hydra integration, component instantiation |
| [project_name/models/base.py](project_name/models/base.py) | Model interface | Weight decay exclusion, optimizer config |
| [project_name/trainers/base.py](project_name/trainers/base.py) | Training loop | AMP, gradient accumulation, early stopping |
| [configs/base.yaml](configs/base.yaml) | Config structure | Hydra composition pattern |
| [project_name/utils/checkpointing.py](project_name/utils/checkpointing.py) | Checkpoint management | Save/restore strategy |
| [project_name/models/registry.py](project_name/models/registry.py) | Registry pattern | Reusable for all components |
| [tests/conftest.py](tests/conftest.py) | pytest fixtures | Test setup patterns |

## Testing Pattern

Tests use pytest with fixtures:
```bash
pytest tests/                    # Run all tests
pytest tests/test_models.py -v   # Verbose, single file
pytest tests/ -k "transformer"   # Filter by name
```

**Key**: Fixture-based configs in `conftest.py` provide consistent test setup.

## Conventions Summary

| Type | Convention | Example |
|------|-----------|---------|
| Classes | PascalCase | `TransformerModel`, `CIFAR10DataModule` |
| Functions | snake_case | `set_seed()`, `train_step()` |
| Config keys | snake_case | `max_epochs`, `hidden_dim` |
| Files | snake_case | `base.py`, `registry.py` |
| Registry names | lowercase + underscore | `@MODEL_REGISTRY.register("my_model")` |
| Imports | Full path | `from project_name.models.base import BaseModel` |

## Key Concepts to Remember

1. **Everything is in config** — No hyperparameter hardcoding; pull from `cfg`
2. **Registry + ABC = Modularity** — Register new components, inherit from base classes
3. **Batch format is Dict[str, Tensor]** — Consistent across models, trainers, metrics
4. **State via Workspace** — Don't manage model/optimizer/scheduler separately
5. **Hydra composition** — Use `defaults:` list for config hierarchy
6. **Device abstraction** — Access via `workspace.device`, never hardcode CUDA/CPU
7. **Logging centralized** — Use `ExperimentLogger` for wandb + console output

## Next Steps When Confused

1. Check the relevant base class (`models/base.py`, `data/base.py`, `trainers/base.py`)
2. Look at a concrete example (e.g., `TransformerModel`, `CIFAR10DataModule`)
3. Verify config usage in `scripts/train.py`
4. Check test files for usage patterns (`tests/conftest.py`, `tests/test_models.py`)


## 0. Mental Model First

```
A good ML research codebase answers YES to all of these:

  Can a new teammate run your experiments in < 30 minutes?
  Can you reproduce any past result from 6 months ago?
  Can you swap model / data / optimizer without touching other files?
  Can you tell what every experiment changed vs the baseline?
  Can you trust your evaluation numbers?
  Can you scale from 1 GPU to 8 GPUs without rewriting?
```

---

## 1. Project Structure

```
[ ] Flat and predictable directory layout
[ ] No deeply nested folders (max 3 levels)
[ ] Clear separation of concerns per directory
[ ] README at every major directory level

Recommended layout:
  project/
  ├── configs/              ← all hyperparameters live here
  │   ├── base.yaml
  │   ├── model/
  │   ├── data/
  │   └── experiment/
  ├── src/                  ← importable source code
  │   ├── models/
  │   ├── data/
  │   ├── trainers/
  │   ├── losses/
  │   ├── metrics/
  │   └── utils/
  ├── scripts/              ← entry points only (thin wrappers)
  │   ├── train.py
  │   ├── evaluate.py
  │   └── sweep.py
  ├── experiments/          ← auto-generated results
  │   └── {run_name}/
  │       ├── config.yaml   ← exact config used
  │       ├── logs/
  │       ├── checkpoints/
  │       └── metrics/
  ├── notebooks/            ← exploration only, not pipeline
  ├── tests/
  ├── data/                 ← raw data (gitignored if large)
  ├── docs/
  ├── requirements.txt
  ├── setup.py / pyproject.toml
  └── README.md

[ ] src/ is installable as a package (setup.py or pyproject.toml)
[ ] scripts/ contains only argument parsing + function calls
[ ] notebooks/ are for exploration only, never imported by src/
[ ] experiments/ is gitignored or tracked with DVC
[ ] data/ is gitignored (tracked separately)
```

---

## 2. Configuration Management

```
[ ] All hyperparameters live in config files, not hardcoded
[ ] Config is hierarchical (base → model → data → experiment)
[ ] Any config value can be overridden from CLI
[ ] Full config is saved alongside every experiment run
[ ] Config has type annotations / validation
[ ] No magic numbers anywhere in src/
[ ] Secrets / API keys in .env, never in config files

Config system options:
  [ ] Hydra + OmegaConf       ← recommended for research
  [ ] simple-parsing
  [ ] argparse + yaml         ← minimum viable

Anti-patterns to avoid:
  [ ] No hardcoded paths in source code
  [ ] No if/else blocks switching on string flags inside model code
  [ ] No default argument mutation between runs
  [ ] No config scattered across multiple locations

Config validation checklist:
  [ ] Unknown keys raise errors (not silently ignored)
  [ ] Type mismatches raise errors
  [ ] Required fields marked explicitly
  [ ] Config diff logged between runs
```

---

## 3. Reproducibility

```
[ ] Random seeds set for all sources:
    [ ] Python random
    [ ] NumPy
    [ ] PyTorch / JAX
    [ ] CUDA (manual_seed_all)
    [ ] Environment seed
[ ] Seed logged to experiment tracker
[ ] torch.backends.cudnn.deterministic documented
    (tradeoff: determinism vs speed — document your choice)

[ ] Exact package versions pinned:
    [ ] requirements.txt with == versions
    [ ] OR pyproject.toml with locked deps
    [ ] OR conda environment.yml
[ ] Python version pinned (.python-version or Dockerfile)

[ ] Hardware documented per experiment:
    [ ] GPU model
    [ ] CUDA version
    [ ] Number of GPUs

[ ] Data versioning:
    [ ] Dataset version / hash logged
    [ ] Preprocessing steps versioned
    [ ] Train/val/test splits fixed and saved
    [ ] No data leakage between splits

[ ] Code versioning:
    [ ] Git commit hash logged with every run
    [ ] Git status (clean/dirty) logged
    [ ] Tag releases that correspond to paper results
```

---

## 4. Experiment Tracking

```
[ ] Every run has a unique, human-readable name
[ ] Run name includes: date + experiment_name + seed
    Example: 2024-01-15_transformer_baseline_s42

[ ] Per-run logging includes:
    [ ] Full config (all hyperparameters)
    [ ] Git commit hash
    [ ] Hardware info
    [ ] Start time / end time / duration
    [ ] All metrics at each step
    [ ] Best metric value + at which step
    [ ] System metrics (GPU util, memory)

[ ] Experiment tracker chosen and consistent:
    [ ] Weights & Biases (recommended)
    [ ] MLflow
    [ ] TensorBoard
    [ ] Neptune
    [ ] ClearML

[ ] Logging is non-blocking (async where possible)
[ ] Failed runs are marked as failed (not just abandoned)
[ ] Experiment groups / tags used for organization
[ ] Notes field used to explain what changed

[ ] Metric naming is consistent across all runs:
    [ ] train/loss  not  loss_train or training_loss
    [ ] val/acc     not  accuracy or val_accuracy
    [ ] Standard prefix: train/ val/ test/

[ ] Log artifacts:
    [ ] Best checkpoint
    [ ] Final config
    [ ] Sample predictions / visualizations
    [ ] Confusion matrix (classification)
    [ ] Loss curves

Anti-patterns:
    [ ] Never log to a file called "results.txt" manually
    [ ] Never track experiments in a spreadsheet by hand
    [ ] Never rely on console output alone
```

---

## 5. Codebase Modularity

```
[ ] Registry pattern for swappable components:
    [ ] Models registered by name
    [ ] Losses registered by name
    [ ] Metrics registered by name
    [ ] Optimizers registered by name
    [ ] Data modules registered by name

Example:
    @MODEL_REGISTRY.register("transformer")
    class TransformerModel(BaseModel): ...

    # In config:  model: transformer
    # Anywhere:   model = MODEL_REGISTRY.build("transformer", **cfg)

[ ] Abstract base classes define required interfaces:
    [ ] BaseModel    — forward(), compute_loss()
    [ ] BaseDataset  — __len__(), __getitem__()
    [ ] BaseTrainer  — train_step(), val_step(), fit()
    [ ] BaseMetric   — update(), compute(), reset()

[ ] No circular imports
[ ] Components are independently testable
[ ] No global mutable state
[ ] No implicit dependencies between modules

[ ] Dependency injection over hard imports:
    [ ] Trainer receives model, not imports it
    [ ] DataModule receives config, not reads files directly
    [ ] Loss receives predictions and targets, nothing else

[ ] Feature flags for experimental code:
    [ ] use_flash_attention: true/false
    [ ] use_compile: true/false
    Not: commenting/uncommenting code blocks
```

---

## 6. Model Code Quality

```
[ ] Model input/output shapes documented in docstring
[ ] Forward pass has no side effects
[ ] Model does not own the optimizer
[ ] Model does not own the data loader
[ ] Model does not perform logging directly

[ ] Shape annotations on all tensors:
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, D)
        # returns: (B, num_classes)

[ ] No magic numbers inside model:
    [ ] Dimensions come from config
    [ ] Numerical constants named and explained
    [ ] Dropout rates in config

[ ] Weight initialization is explicit:
    [ ] init strategy documented
    [ ] Not relying on framework defaults silently

[ ] Model has:
    [ ] count_parameters() method
    [ ] __repr__ showing architecture
    [ ] save() / load() methods or compatible state_dict

[ ] Numerical stability:
    [ ] No division by zero (add epsilon)
    [ ] Log-sum-exp trick where relevant
    [ ] Gradient clipping configured
    [ ] NaN/Inf checks in debug mode
```

---

## 7. Data Pipeline

```
[ ] Data loading is completely separate from model code
[ ] Dataset class handles one sample, DataLoader handles batching
[ ] All preprocessing is deterministic and documented
[ ] Augmentation is configurable (on/off, strength)
[ ] Train and val/test transforms are different and explicit

[ ] DataModule interface:
    [ ] setup(stage)         — called once before training
    [ ] train_dataloader()   — shuffled, augmented
    [ ] val_dataloader()     — not shuffled, no augmentation
    [ ] test_dataloader()    — not shuffled, no augmentation

[ ] Performance:
    [ ] num_workers > 0 for CPU-bound loading
    [ ] pin_memory=True when using GPU
    [ ] prefetch_factor set appropriately
    [ ] persistent_workers=True for small datasets
    [ ] Data fits in RAM? → cache it

[ ] Data validation:
    [ ] Check dataset size at startup
    [ ] Check label distribution
    [ ] Detect empty batches
    [ ] Detect NaN in inputs
    [ ] Check class imbalance

[ ] Data splits:
    [ ] Splits are fixed (saved indices or fixed seed)
    [ ] No overlap between train/val/test
    [ ] Split sizes logged
    [ ] Stratified split for imbalanced datasets

[ ] Collate function handles variable-length sequences
[ ] Custom collate documented if used
```

---

## 8. Training Loop

```
[ ] Training loop is in a Trainer class, not in main()
[ ] main() only parses args and calls trainer.fit()
[ ] Training loop handles:
    [ ] Mixed precision (AMP)
    [ ] Gradient accumulation
    [ ] Gradient clipping
    [ ] Learning rate scheduling
    [ ] Early stopping
    [ ] Checkpointing
    [ ] Metric logging
    [ ] train() / eval() mode switching

[ ] Gradient accumulation:
    [ ] Effective batch size = batch_size * accumulation_steps
    [ ] Effective batch size logged
    [ ] Loss scaled by accumulation steps

[ ] Scheduler step timing documented:
    [ ] After optimizer.step()
    [ ] Per step vs per epoch (explicit)

[ ] Validation:
    [ ] torch.no_grad() or inference_mode() always used
    [ ] Model set to eval() before validation
    [ ] Model set back to train() after validation
    [ ] val metrics computed over full val set, not per-batch

[ ] Checkpointing strategy:
    [ ] Save best checkpoint (monitored metric)
    [ ] Save last checkpoint (resume)
    [ ] Save top-k checkpoints
    [ ] Checkpoint includes: model, optimizer, scheduler, epoch, config
    [ ] Checkpoint can resume training exactly

[ ] Resume from checkpoint:
    [ ] model state
    [ ] optimizer state
    [ ] scheduler state
    [ ] epoch / step counter
    [ ] RNG state (for full reproducibility)
    [ ] Best metric value (for early stopping)
```

---

## 9. Evaluation and Metrics

```
[ ] Evaluation is a separate script from training
[ ] Metrics are computed over the entire test set
[ ] No metric is computed on training data and reported as test metric
[ ] Metric implementation is tested against a known baseline

[ ] For each metric:
    [ ] update(preds, targets) accumulates
    [ ] compute() returns final value
    [ ] reset() clears state
    [ ] Handles edge cases (empty batch, all-same class)

[ ] Statistical rigor:
    [ ] Multiple seeds run (at least 3)
    [ ] Mean ± std reported, not single run
    [ ] Confidence intervals for important results
    [ ] Significance tests for comparisons

[ ] Evaluation checklist:
    [ ] Test set touched only once (final evaluation)
    [ ] Validation set used for all hyperparameter decisions
    [ ] No test set peeking during development
    [ ] Evaluation is deterministic (fixed seed, no dropout)

[ ] Baselines:
    [ ] Random baseline implemented and evaluated
    [ ] Simple heuristic baseline implemented
    [ ] Prior work baseline reproduced or cited with numbers
    [ ] All baselines evaluated on same test set

[ ] Results table:
    [ ] All compared methods use same data split
    [ ] All compared methods evaluated same way
    [ ] Variance reported
    [ ] Compute cost reported (runtime, GPU-hours)
```

---

## 10. Testing

```
[ ] Tests exist and pass before every commit
[ ] Tests run in CI (GitHub Actions / GitLab CI)
[ ] Test coverage > 70% for src/

Unit tests:
  [ ] Every model: forward pass with dummy input
  [ ] Every model: output shape is correct
  [ ] Every model: loss is not NaN
  [ ] Every dataset: __len__ and __getitem__ work
  [ ] Every metric: known input → known output
  [ ] Every config: loads without error
  [ ] Every loss: gradient flows (loss.backward() works)

Integration tests:
  [ ] One full training step (forward + backward + optimizer)
  [ ] One full validation loop
  [ ] Checkpoint save → load → resume produces same result
  [ ] Data pipeline produces expected shapes end-to-end

Regression tests:
  [ ] Known input → known output for key functions
  [ ] Model with fixed seed → fixed output
  [ ] Metric values match hand-computed examples

Performance tests (optional but useful):
  [ ] Training step time < threshold
  [ ] Memory usage < threshold
  [ ] Data loading is not the bottleneck

Test structure:
  tests/
  ├── test_models.py
  ├── test_data.py
  ├── test_metrics.py
  ├── test_losses.py
  ├── test_trainer.py
  ├── test_config.py
  └── conftest.py          ← shared fixtures

[ ] Tests use small models and tiny datasets
[ ] Tests do not require GPU (use CPU)
[ ] Tests run in < 2 minutes total
[ ] Fixtures shared via conftest.py
[ ] No test depends on another test
```

---

## 11. Hyperparameter Management

```
[ ] All hyperparameters in config files
[ ] Hyperparameter search is scriptable
[ ] Search space is documented and versioned
[ ] Search results are logged (not just best result)

[ ] Sweep strategy chosen:
    [ ] Grid search (small spaces)
    [ ] Random search (medium spaces)
    [ ] Bayesian optimization (expensive evaluations)
    [ ] Population-based training (if using W&B PBT)

[ ] For each hyperparameter, document:
    [ ] Default value and why
    [ ] Search range and why
    [ ] Sensitivity (is it critical or not)
    [ ] Interaction with other hyperparameters

[ ] Learning rate:
    [ ] Warm-up configured
    [ ] Schedule type documented
    [ ] LR range test run
    [ ] Final LR logged

[ ] Batch size:
    [ ] Effective batch size = batch * accumulation * num_gpus logged
    [ ] LR scaled with batch size if changed
    [ ] Memory usage at this batch size documented

[ ] Ablation studies:
    [ ] Each component tested independently
    [ ] Ablation results in a table
    [ ] Control experiments clearly labeled
    [ ] Same seed across ablations for fair comparison
```

---

## 12. Compute and Scalability

```
[ ] Single-GPU training works cleanly first
[ ] Multi-GPU added as an optional extension, not a requirement

Single GPU checklist:
  [ ] Mixed precision (AMP / bfloat16) enabled
  [ ] torch.compile used where beneficial
  [ ] Memory profiled (no unnecessary tensor retention)
  [ ] Data loading not the bottleneck
  [ ] GPU utilization > 80% during training

Multi-GPU checklist:
  [ ] DistributedDataParallel (DDP) or FSDP used
  [ ] Metrics aggregated across GPUs (dist.all_reduce)
  [ ] Logging done only on rank 0
  [ ] Checkpointing done only on rank 0
  [ ] Seed different per rank (for data sampling)
  [ ] Batch size scales linearly with num GPUs
  [ ] LR scaled with effective batch size

Memory optimization:
  [ ] Gradient checkpointing for large models
  [ ] del intermediate tensors where needed
  [ ] optimizer.zero_grad(set_to_none=True) used
  [ ] No Python objects holding references to tensors

Profiling tools:
  [ ] torch.profiler used at least once
  [ ] GPU memory tracked (torch.cuda.memory_summary())
  [ ] Bottleneck identified (CPU/GPU/data/memory)
  [ ] Profile results saved as artifact
```

---

## 13. Code Style and Documentation

```
[ ] Code formatter configured and applied:
    [ ] black (Python)
    [ ] isort (imports)
    [ ] ruff (linting)

[ ] Type annotations on all public functions
[ ] Docstrings on all public classes and functions
[ ] Docstring format consistent (Google / NumPy / reStructuredText)

Docstring must include:
  [ ] What the function does (one line)
  [ ] Args with types and descriptions
  [ ] Returns with types and descriptions
  [ ] Raises (if applicable)
  [ ] Example (for non-trivial functions)

[ ] Comments explain WHY, not WHAT
    Bad:  # multiply by 2
    Good: # scale by 2 because loss is averaged over 2 GPUs

[ ] No commented-out code in main branch
[ ] No TODO left > 1 week without a ticket
[ ] Magic numbers named with ALL_CAPS constants
[ ] Long functions (> 50 lines) split into subfunctions

[ ] pre-commit hooks configured:
    [ ] black
    [ ] isort
    [ ] ruff / flake8
    [ ] mypy (optional but recommended)
    [ ] no large files committed accidentally
```

---

## 14. Version Control

```
[ ] .gitignore covers:
    [ ] data/              ← raw datasets
    [ ] experiments/       ← results
    [ ] artifacts/         ← checkpoints
    [ ] __pycache__/
    [ ] .env               ← secrets
    [ ] *.pyc
    [ ] wandb/
    [ ] .DS_Store

[ ] Branching strategy:
    [ ] main / master      ← stable, paper-ready code
    [ ] dev                ← integration branch
    [ ] feature/*          ← new features
    [ ] experiment/*       ← experimental ideas

[ ] Commit message convention:
    feat:  add transformer encoder
    fix:   correct off-by-one in positional encoding
    exp:   try cosine annealing schedule
    refac: extract loss computation to separate module
    docs:  add docstrings to data module
    test:  add unit tests for metric computation

[ ] Tag releases:
    [ ] v1.0  ← first submission / paper version
    [ ] v1.1  ← camera-ready revision

[ ] Every significant experiment referenced by commit hash
[ ] No large binary files in git history
[ ] Git LFS for necessary large files
```

---

## 15. Documentation

```
README.md must contain:
  [ ] One-sentence description of the project
  [ ] Installation instructions (exact commands)
  [ ] Quickstart: reproduce main result in one command
  [ ] Project structure overview
  [ ] Config explanation
  [ ] How to run training
  [ ] How to run evaluation
  [ ] How to add a new model / dataset
  [ ] Citation / paper link
  [ ] License

docs/ should contain:
  [ ] Design decisions and why
  [ ] Experiment log (what was tried, what worked)
  [ ] Known issues and limitations
  [ ] Data format specification
  [ ] API reference (auto-generated from docstrings)

Experiment log format:
  ## 2024-01-15 — Baseline transformer
  Config: configs/experiment/baseline.yaml
  Commit: abc1234
  Result: val/acc=72.3 ± 1.2 (3 seeds)
  Notes:  used cosine LR with 10% warmup

  ## 2024-01-18 — Added pre-norm
  Config: configs/experiment/prenorm.yaml
  Commit: def5678
  Change: LayerNorm before attention (not after)
  Result: val/acc=74.1 ± 0.8 (3 seeds)
  Notes:  +1.8% improvement, training more stable
```

---

## 16. Environment and Dependencies

```
[ ] Python version pinned (.python-version file)
[ ] All dependencies in requirements.txt with exact versions
[ ] Dev dependencies separate (requirements-dev.txt)

requirements.txt:
  torch==2.1.0
  numpy==1.24.3
  ...

requirements-dev.txt:
  pytest==7.4.0
  black==23.9.1
  mypy==1.5.1
  ...

[ ] Docker / container option:
    [ ] Dockerfile provided
    [ ] Base image pinned (not :latest)
    [ ] GPU support (nvidia/cuda base)
    [ ] Build instructions in README

[ ] Virtual environment instructions provided:
    [ ] venv setup commands
    [ ] conda env creation commands
    [ ] uv install command

[ ] Installation tested on clean environment
[ ] Installation time documented
[ ] Minimum hardware requirements documented
```

---

## 17. Releasing and Sharing

```
[ ] Paper results are reproducible from released code
[ ] Pretrained weights available (HuggingFace Hub / Zenodo)
[ ] Weights include config used to produce them
[ ] Inference-only script provided (no training deps needed)

[ ] Submission / deployment checklist:
    [ ] Remove all debug prints
    [ ] Remove all hardcoded paths
    [ ] Remove all internal comments
    [ ] Test on a clean machine
    [ ] Test with CPU only
    [ ] Test with GPU
    [ ] Measure inference speed

[ ] Model card if releasing weights:
    [ ] Training data description
    [ ] Evaluation results
    [ ] Intended use
    [ ] Limitations
    [ ] Compute used
```

---

## 18. Quick Self-Audit

```
Run through this every few weeks:

Reproducibility audit:
  [ ] Pick any past run from W&B
  [ ] Find its config and commit hash
  [ ] Re-run it
  [ ] Does it produce the same result? ± small numerical noise

Onboarding audit:
  [ ] Ask a colleague to run your code from scratch
  [ ] Measure how long it takes
  [ ] Note every question they had
  [ ] Fix the README / docs

Debt audit:
  [ ] How many TODOs are in the codebase?
  [ ] How many files have no tests?
  [ ] How many experiments are not logged?
  [ ] Are there hardcoded paths anywhere?

Performance audit:
  [ ] Run torch.profiler for one training step
  [ ] Is GPU utilization > 80%?
  [ ] Is data loading the bottleneck?
  [ ] Is there room for torch.compile?
```

---

## Summary Priority Order

```
Week 1 — must have:
  ✓ Installable src/ package
  ✓ Config-driven (no hardcoded values)
  ✓ Seeds set everywhere
  ✓ Experiment tracker (W&B or MLflow)
  ✓ Checkpoint save/load
  ✓ Basic unit tests

Week 2 — should have:
  ✓ Registry pattern for models/losses/metrics
  ✓ Abstract base classes
  ✓ Full config saved per run
  ✓ Git commit hash logged
  ✓ Evaluation script separate from training
  ✓ CI running tests

Month 1 — nice to have:
  ✓ Sweep / hyperparameter search script
  ✓ Docker / reproducible environment
  ✓ Multi-GPU support
  ✓ Data versioning (DVC)
  ✓ Performance profiling
  ✓ Full documentation

Before paper submission:
  ✓ Multiple seeds for all results
  ✓ Ablation studies logged
  ✓ Baseline comparisons on same data
  ✓ Released code reproduces paper numbers
  ✓ Pretrained weights released
  ✓ README with quickstart
```