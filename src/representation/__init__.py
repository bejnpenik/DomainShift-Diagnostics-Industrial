# representation/__init__.py

from __future__ import annotations

from typing import Protocol
import numpy.typing as npt
import torch


class Processor(Protocol):
    """Single-channel processor protocol.

    DomainDataset calls __call__(signal, sampling_rate) for processors
    that satisfy this interface.
    """

    @property
    def name(self) -> str:
        ...

    def __call__(self, data: npt.ArrayLike, sampling_rate: int) -> torch.Tensor:
        ...


class MultiChannelProcessor(Protocol):
    """Extended protocol for processors that require multiple raw input channels.

    DomainDataset detects this via hasattr(processor, 'required_reader_channels')
    and calls process(channels) instead of __call__(signal, sampling_rate).

    Implementors must also satisfy the Processor protocol (name + __call__)
    so they are valid wherever a Processor is expected; __call__ should raise
    NotImplementedError to prevent accidental single-channel use.
    """

    @property
    def name(self) -> str:
        ...

    @property
    def required_reader_channels(self) -> frozenset[str]:
        """Reader channel names that DomainDataset must preload from disk."""
        ...

    def process(self, channels: dict) -> torch.Tensor:
        """Transform a dict of {reader_channel_name: np.ndarray} into a tensor."""
        ...

    def __call__(self, data: npt.ArrayLike, sampling_rate: int) -> torch.Tensor:
        ...


class ProcessorConfig(Protocol):
    @property
    def name(self) -> str:
        ...


def create_processor(config) -> Processor:
    """Create a Processor from any config type.

    Accepts:
        - SignalProcessorConfig  → SignalProcessor
        - OrderTrackingProcessorConfig → OrderTrackingProcessor
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
        from .builder import build_processor_config
        config = build_processor_config(config)

    # OrderTrackingProcessorConfig
    from .order.config import OrderTrackingProcessorConfig
    if isinstance(config, OrderTrackingProcessorConfig):
        from .order.processor import OrderTrackingProcessor
        return OrderTrackingProcessor(config)

    # SignalProcessorConfig
    from .signal.config import SignalProcessorConfig
    if isinstance(config, SignalProcessorConfig):
        from .signal.processor import SignalProcessor
        return SignalProcessor(config)

    raise ValueError(f"Unknown config type: {type(config)}")
