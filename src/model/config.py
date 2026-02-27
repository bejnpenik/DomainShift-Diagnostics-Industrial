# model/config.py
from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

from typing import Any

from pathlib import Path

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
    
    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> ModelConfig:
        """Create ModelConfig from a parsed YAML dict.

        The returned config uses the YAML builder as its model_class.

        """
        from model.builder import build_model
        name = cfg.get("name", "unnamed")

        return cls(
            name=name,
            model_class=_YAMLModelFactory(cfg),
            params={},
        )
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        """Create ModelConfig from a YAML file."""
        import yaml

        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls.from_dict(cfg)


class _YAMLModelFactory:
    """Callable that wraps a YAML config dict for use as model_class.

    Stores the config and calls build_model when invoked.
    Satisfies the model_class(num_classes=N, **params) interface.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg

    def __call__(self, num_classes: int, **kwargs: Any) -> nn.Module:
        from model.builder import build_model
        return build_model(self._cfg, num_classes)

    def __repr__(self) -> str:
        return f"_YAMLModelFactory(name={self._cfg.get('name', '?')})"
