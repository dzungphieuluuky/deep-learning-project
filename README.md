# ML Research Project

A **modular, production-ready deep learning framework** for machine learning research experiments. Built with PyTorch, Transformers, and Hydra for configuration management. Perfect for experimenting with models, datasets, and training strategies at scale.

## Features

- 🏗️ **Modular Architecture** - Separated concerns for models, data, trainers, and utilities
- ⚙️ **Hydra Configuration** - Declarative experiment configuration with composition and overrides
- 🤖 **Transformer Models** - Pre-built transformer implementations with optimizations (Flash Attention support)
- 📊 **Experiment Tracking** - Integration with Weights & Biases (wandb) for logging and visualization
- 🔄 **Distributed Training** - Built-in support for multi-GPU training via Accelerate
- 💾 **Checkpointing** - Automatic checkpoint management with early stopping and model selection
- 📈 **Metrics Registry** - Extensible metric computation framework
- 🔐 **Reproducibility** - Seed management and precision control for consistent results
- ✅ **Testing** - Comprehensive test suite with pytest

## Quick Start

### Installation

1. **Clone and setup the environment:**
   ```bash
   cd deep-learning-project
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the project package:**
   ```bash
   pip install -e .
   ```

### Basic Training

Train a model using the default configuration:
```bash
python scripts/train.py
```

Override configuration parameters:
```bash
python scripts/train.py \
  experiment=baseline \
  training.max_epochs=50 \
  model.hidden_dim=512 \
  data.batch_size=32
```

## Project Structure

```
project_name/
├── data/              # Data loading and preprocessing
│   ├── base.py       # Base dataset classes
│   └── datasets.py   # Dataset implementations
├── models/            # Model architectures
│   ├── base.py       # Base model class
│   ├── registry.py   # Model registry for dynamic instantiation
│   └── transformer.py # Transformer implementations
├── metrics/           # Evaluation metrics
│   └── registry.py   # Metrics registry
├── trainers/          # Training loops
│   ├── base.py       # Base trainer
│   └── supervised.py # Supervised learning trainer
├── utils/             # Utilities
│   ├── checkpointing.py    # Checkpoint management
│   ├── logging.py          # Experiment logging
│   └── reproducibility.py  # Seed and precision control
└── workspace/         # Workspace management
    └── base.py       # Experiment workspace

configs/
├── base.yaml          # Base configuration
├── data/              # Data configuration presets
├── model/             # Model configuration presets
└── experiment/        # Experiment presets

scripts/
├── train.py           # Training script
├── evaluate.py        # Evaluation script
└── sweep.py           # Hyperparameter sweep

tests/                 # Test suite
```

## Configuration

Configuration is managed through YAML files in `configs/` using Hydra. 

**Key concepts:**
- **Base config** (`base.yaml`) - Default settings for all experiments
- **Experiment configs** - Predefined combinations in `configs/experiment/`
- **Overrides** - Command-line overrides take precedence

Example configuration structure:
```yaml
# Base settings
project: "ml_research"
seed: 42

# Paths
paths:
  data: ./data
  checkpoints: ./experiments/checkpoints

# Training
training:
  max_epochs: 100
  batch_size: 32
  learning_rate: 1e-4

# Model
model:
  _target_: project_name.models.transformer.TransformerModel
  hidden_dim: 768
  num_layers: 12
```

## Usage Examples

### Training a Model

```bash
# Train with default config
python scripts/train.py

# Train a specific experiment
python scripts/train.py experiment=baseline

# Override multiple parameters
python scripts/train.py \
  training.max_epochs=200 \
  training.batch_size=64 \
  model.hidden_dim=1024
```

### Evaluation

```bash
python scripts/evaluate.py checkpoint=/path/to/model.pt
```

### Hyperparameter Sweep

```bash
python scripts/sweep.py -m \
  model.hidden_dim=512,768,1024 \
  training.learning_rate=1e-4,1e-3,1e-2
```

## Dependencies

Key dependencies include:
- **PyTorch** (>=2.0.0) - Deep learning framework
- **Transformers** (>=4.35.0) - Pre-trained models and utilities
- **Hydra** (>=1.3.0) - Configuration management
- **Accelerate** (>=0.24.0) - Distributed training
- **Weights & Biases** (>=0.16.0) - Experiment tracking
- **Einops** (>=0.7.0) - Tensor manipulation
- **PyTest** (>=7.4.0) - Testing

See `requirements.txt` for complete list.

## Testing

Run the test suite:
```bash
pytest tests/
```

Run tests with verbose output:
```bash
pytest tests/ -v
```

Run a specific test file:
```bash
pytest tests/test_models.py
```

## Experiment Tracking

This project integrates with **Weights & Biases (wandb)** for experiment tracking. 

Configure in your experiment config:
```yaml
logging:
  wandb:
    enabled: true
    project: "ml_research"
    entity: "your-entity"
```

View your experiments at [wandb.ai](https://wandb.ai)

## Reproducibility

Ensure reproducible results:
- Seed is set at the start of training (default: 42)
- Floating-point precision is controlled
- Deterministic operations where possible

Override seed:
```bash
python scripts/train.py seed=123
```

## Contributing

Contributions are welcome! Please:
1. Add tests for new features
2. Follow the modular structure
3. Update configurations as needed
4. Ensure all tests pass before submitting

## License

[Add your license here]

## Contact

[Add contact information if needed]
