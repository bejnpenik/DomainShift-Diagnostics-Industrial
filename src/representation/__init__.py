# representation/__init__.py
from __future__ import annotations

from typing import Protocol
import numpy.typing as npt
import torch

from collection import Metadata


class Processor(Protocol):
    @property
    def name(self) -> str: 
        ...
    def __call__(self, data: npt.ArrayLike, metadata: Metadata) -> torch.Tensor: 
        ...

class ProcessorConfig(Protocol):
    @property
    def name(self)->str:
        ...




def create_processor(config) -> Processor:
    from representation.signal.config import SignalProcessorConfig
    from representation.signal.processor import SignalProcessor

    if isinstance(config, SignalProcessorConfig):
        return SignalProcessor(config)

    raise ValueError(f"Unknown config type: {type(config)}")