from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class PPOConfig:
    obs_dim: tuple | int
    act_dim: int
    actor_hidden_size: int
    critic_hidden_size: int

    T: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epsilon: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    minibatch_size: int = 64
    epochs_per_batch: int = 4
    entropy_coefficient: float = 0.01
    entropy_decay: bool = True
    device: str = "cuda" if torch.cuda.is_available() \
                else 'mps' if torch.backends.mps.is_available() \
                else "cpu"

    
    # Wandb
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    video_log_freq: Optional[int] = None
    
    # Checkpointing
    save_freq: Optional[int] = None  # Save checkpoint every N steps (None = disabled)
    save_dir: str = "checkpoints"    # Directory to save checkpoints

