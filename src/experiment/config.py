from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple

from representation import ProcessorConfig
from training.config import TrainerConfig
from model.config import ModelConfig
from experiment.sampling import FileSamplingProtocol



@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    processor_config: ProcessorConfig
    model_config: ModelConfig
    trainer_config: TrainerConfig = field(default_factory=TrainerConfig)
    
    # Data
    file_sampling: FileSamplingProtocol | None = None
    normalization: Literal["sample", "dataset", "pretrained", "none"] = "none"
    normalization_vals: tuple | None = None
    train_val_split_ratio: float = 0.33
    random_seed: int = 42

    @property
    def processor_name(self) -> str:
        """Get processor name from processor_config."""
        return self.processor_config.name
    
    @property
    def model_name(self) -> str:
        """Get model name from model_config"""
        return self.model_config.name
    @property
    def model_class_name(self) -> str:
        """Get model_class.__name__ from model_config"""
        return self.model_config.model_class.__name__