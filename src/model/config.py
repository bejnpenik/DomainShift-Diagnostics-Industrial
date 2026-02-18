# model/config.py
from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

@dataclass(frozen=True)
class ModelConfig:
    """Specification for model architecture."""
    name: str
    model_class: type
    params: dict = field(default_factory=dict)

    def create_model(self, num_classes:int) -> nn.Module:
        """Generate new model instance"""
        return self.model_class(
            num_classes=num_classes,
            **self.params
        )