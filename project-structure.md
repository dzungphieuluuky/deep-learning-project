deep_learning_project/
├── configs/                  # Hydra configuration files
│   ├── callbacks/            # Checkpointing, EarlyStopping
│   ├── data/                 # DataModule configs
│   ├── model/                # Model architecture configs
│   ├── trainer/              # Lightning Trainer flags
│   └── train.yaml            # Main config entry point
├── src/
│   ├── data/                 # Data loading logic
│   ├── models/               # Model architectures & LightningWrappers
│   ├── utils/                # Utilities (instantiation, logging)
│   └── __init__.py
├── scripts/                  # Standalone scripts (e.g., manual Accelerate loops)
├── pyproject.toml            # Dependencies
├── train.py                  # Main entry point (Lightning)
└── .env                      # API Keys (WANDB_API_KEY)