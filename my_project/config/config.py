# config.py

import os
from dataclasses import dataclass, field
from typing import Optional
import yaml

@dataclass
class Config:
    # Model Config
    model: str = "STT1"
    id: Optional[str] = None

    # Device Config
    default: str = "cuda"
    devices: str = "0"

    # Job Config
    name: str = "model_saved_0"

    # Distributed Config
    rank: int = 0
    socket: str = "12455"

    # Data Loading Config
    num_workers: int = 2
    batch_size: int = 32
    max_epochs: int = 10
    folder_train: str = "/data/prabajcn/projects/TranSia/embeddingsWbatchNPY/dataset0/train/"
    folder_val: str = "/data/prabajcn/projects/TranSia/embeddingsWbatchNPY/dataset0/val/"
    folder_test: str = "/data/prabajcn/projects/TranSia/embeddingsWbatchNPY/dataset0/test/"

    # Compatibility Config
    mock: int = 0

    # Sampling Config
    sample_fraction: Optional[float] = None
    add_noise: float = 0.0

    # Tokenizer Config
    stride: int = 1
    window: int = 1
    shift: int = 1

    # Hyperparameters Config
    seed: int = 1111
    max_seq_len: int = 1201
    embd_dim: int = 512
    hidn_dim: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.10
    clip: float = 0.15
    lr: float = 0.0008
    pad_token: int = 0
    log_interval: int = 100

    # Optimizer Config
    optimizer_type: str = "AdamW"

    # Scheduler Config
    scheduler_type: str = "CosineAnnealingWarmRestarts"

    # Resume / Restart Config
    resume_checkpoint: Optional[str] = None
    restart_epoch: int = 0
    restart_log: int = 0

    # Vocabulary Config
    n_token: int = 27

    # Save Config
    save_dir: str = "model_saved_0"

    # ----------------------
    # W&B CONFIG FIELDS
    # ----------------------
    wandb_project: str = "my_project"
    wandb_entity: Optional[str] = None
    wandb_run_name: str = "SNN_finetune"
    # ----------------------

def get_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)