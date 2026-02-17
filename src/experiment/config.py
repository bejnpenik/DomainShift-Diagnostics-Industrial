from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple

from representation.signal import SignalProcessorConfig
from training.config import TrainerConfig

from experiment.sampling import FileSamplingProtocol




@dataclass(frozen=True)
class ExperimentConfig:
    """Complete specification for one experiment.

    Composes pipeline, training, and data configs.
    Name must be unique within a study.
    """
    name: str

    # Representation
    processor_config: SignalProcessorConfig

    # Model
    model_name: str
    model_class: type
    model_params: dict = field(default_factory=dict)

    # Training
    trainer_config: TrainerConfig = field(default_factory=TrainerConfig)

    # Data
    file_sampling: FileSamplingProtocol | None = None
    normalization: Literal["sample", "dataset", "pretrained"] = "sample"
    normalization_vals: Tuple | None = None
    train_val_split_ratio: float = 0.33

    random_seed: int = 42


    @property
    def processor_name(self) -> str:
        """Get processor name from processor_config."""
        return self.processor_config.name