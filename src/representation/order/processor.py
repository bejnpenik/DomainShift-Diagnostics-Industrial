from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

from .angular_resampler import AngularResampler
from .config import OrderTrackingProcessorConfig


class OrderTrackingProcessor:
    """Angular-domain processor using order tracking.

    Converts vibration from the time domain to the angular domain by:
        1. Integrating the RPM signal → cumulative shaft angle
        2. Resampling vibration onto a uniform angular grid
        3. Segmenting into fixed-revolution windows
        4. Applying a view transform (raw angular or order spectrum)

    This processor requires two raw channels (vibration + RPM) and
    implements the multi-channel processor protocol:

        required_reader_channels  →  frozenset of reader channel names to preload
        process(channels)         →  the actual transform

    DomainDataset detects this via hasattr(processor, 'required_reader_channels')
    and calls process() instead of __call__().

    V1 limitation: segment_raw() is not implemented, so conditioning channels
    alongside an OrderTrackingProcessor are not supported. DomainDataset raises
    ValueError at init if pipeline.conditioning is non-empty with this processor.
    """

    def __init__(self, config: OrderTrackingProcessorConfig) -> None:
        self._config = config
        self._resampler = AngularResampler()
        self._view = config.view.create_view()

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def config(self) -> OrderTrackingProcessorConfig:
        return self._config

    @property
    def required_reader_channels(self) -> frozenset[str]:
        """Reader channel names that must be loaded before calling process()."""
        return frozenset({
            self._config.vibration_reader_channel,
            self._config.rpm_reader_channel,
        })

    def process(self, channels: dict[str, np.ndarray]) -> torch.Tensor:
        """Transform raw vibration + RPM into a model-ready tensor.

        Args:
            channels: Dict mapping reader channel names to raw numpy arrays.
                Must contain both vibration_reader_channel and rpm_reader_channel.

        Returns:
            (N, 1, L) for raw_order view or (N, 1, O) for order_spectrum view.

        Raises:
            ValueError: If the recording spans less than one shaft revolution,
                or if the angular signal is shorter than one window.
        """
        cfg = self._config
        vib = np.asarray(channels[cfg.vibration_reader_channel], dtype=np.float64)
        rpm = np.asarray(channels[cfg.rpm_reader_channel], dtype=np.float64)

        cumangle = self._resampler.integrate_rpm(rpm, cfg.rpm_sampling_rate)
        angular = self._resampler.resample_to_angular(
            vib,
            cfg.vibration_sampling_rate,
            cumangle,
            cfg.rpm_sampling_rate,
            cfg.target_orders,
        )
        windows = self._resampler.segment_angular(
            angular,
            cfg.target_orders,
            cfg.window_revolutions,
            cfg.window_overlap,
        )
        t = torch.tensor(windows, dtype=torch.float32)  # (N, window_samples)
        return self._view(t)

    def __call__(self, data: npt.ArrayLike, sampling_rate: int) -> torch.Tensor:
        raise NotImplementedError(
            "OrderTrackingProcessor requires both vibration and RPM channels. "
            "DomainDataset should call process(channels) via required_reader_channels — "
            "do not call __call__ directly."
        )
