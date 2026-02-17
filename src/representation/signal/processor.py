from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

from collection import Metadata
from representation.signal.config import SignalProcessorConfig, RawViewConfig, STFTViewConfig
from representation.signal.resampling import Resampler
from representation.signal.segmentation import SignalSegmenter


class SignalProcessor:
    """Implement ψ for 1D signals: resample → segment → view.
    
    Args:
        config: Complete signal pipeline specification.
    """

    def __init__(self, config: SignalProcessorConfig) -> None:
        self._config = config

        self._resampler = Resampler(config.max_signal_bandwidth_factor)

        self._segmenter = SignalSegmenter(
            window_duration=config.window_duration,
            overlap=config.window_overlap,
        )

        self._view = config.view.create_view()

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def config(self) -> SignalProcessorConfig:
        return self._config

    def __call__(self, signal: npt.ArrayLike, metadata: Metadata) -> torch.Tensor:
        """Transform raw signal into model-ready tensor.

        Args:
            signal: 1D numpy array.
            metadata: Must contain 'sampling_rate'.

        Returns:
            (N, 1, L) for raw or (N, 1, F, T) for STFT.
        """
        sampling_rate = metadata.sampling_rate

        # Resample if needed
        if sampling_rate != self._config.target_sampling_rate:
            signal = self._resampler(signal, sampling_rate, self._config.target_sampling_rate)

        # Segment into windows
        windows = self._segmenter(signal, self._config.target_sampling_rate)

        # Apply view
        return self._view(windows)