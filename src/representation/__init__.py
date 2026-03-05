# representation/__init__.py

from __future__ import annotations

from typing import Protocol
import numpy.typing as npt
import torch

from ..collection import Metadata


class Processor(Protocol):
    @property
    def name(self) -> str:
        ...

    def __call__(self, data: npt.ArrayLike, metadata: Metadata) -> torch.Tensor:
        ...


class ProcessorConfig(Protocol):
    @property
    def name(self) -> str:
        ...


def create_processor(config) -> Processor:
    """Create a Processor from any config type.

    Accepts:
        - SignalProcessorConfig (or any config with create_processor())
        - A dict (parsed YAML) — dispatched through the builder
        - A str or Path — loaded as YAML file, then built
    """
    from pathlib import Path

    # String or Path -> load YAML file
    if isinstance(config, (str, Path)):
        from .builder import build_processor_config_from_yaml
        config = build_processor_config_from_yaml(config)

    # Dict -> build from dict
    if isinstance(config, dict):
        from representation.builder import build_processor_config
        config = build_processor_config(config)

    # SignalProcessorConfig (or anything with create_processor)
    from .signal.config import SignalProcessorConfig
    if isinstance(config, SignalProcessorConfig):
        from .signal.processor import SignalProcessor
        return SignalProcessor(config)

    raise ValueError(f"Unknown config type: {type(config)}")