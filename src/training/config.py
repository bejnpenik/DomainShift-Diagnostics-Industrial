from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Tuple
from dataclasses import dataclass
import torch.nn as nn

class TrainerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_epochs: int = Field(default=2000, gt=0)
    optimizer_name: Literal["adamw", "sgd"] = "adamw"
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    momentum: float = Field(default=0.9, ge=0)
    device: str = "cuda"
    early_stopping: Tuple[int, float] | None = (10, 0.0)
    min_epochs: int = Field(default=100, ge=0)
    noise: Tuple[float, float] | None = (0.1, 0.1)
    verbose_level: int = Field(default=1, ge=0)


@dataclass
class TrainResult:
    """Output from a training run."""
    model: nn.Module
    epochs_run: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float