from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch

from .angular_resampler import AngularResampler
from .config import OrderTrackingProcessorConfig
from collection.metadata import Metadata


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

    def process(
        self, channels: dict[str, np.ndarray], metadata: Metadata | None = None,
    ) -> torch.Tensor:
        """Transform raw vibration + RPM into a model-ready tensor.

        Args:
            channels: Dict mapping reader channel names to raw numpy arrays.
                Must contain both vibration_reader_channel and rpm_reader_channel.
                A length-1 rpm array is treated as a metadata-sourced nominal
                speed and broadcast to a constant-speed profile.
            metadata: Optional sample metadata. If both metadata and
                config.nominal_speed_metadata_path are set, the measured
                integrated revolutions are sanity-checked against the nominal
                RPM found at that metadata path.

        Returns:
            (N, 1, L) for raw_order view or (N, 1, O) for order_spectrum view.

        Raises:
            ValueError: If the recording spans less than one shaft revolution,
                if the angular signal is shorter than one window, if the
                cumulative shaft angle is corrupted/non-monotonic, or if the
                measured revolutions deviate from nominal_speed_metadata_path
                by more than 20%.
        """
        cfg = self._config
        vib = np.asarray(channels[cfg.vibration_reader_channel], dtype=np.float64)
        rpm = np.asarray(channels[cfg.rpm_reader_channel], dtype=np.float64)

        # Scalar RPM (metadata-sourced nominal speed, no tachometer trace) —
        # broadcast to a constant-speed profile spanning the vibration
        # recording's duration so tachometer-less datasets can still use
        # order tracking.
        if len(rpm) == 1:
            n = max(2, int(round(len(vib) / cfg.vibration_sampling_rate * cfg.rpm_sampling_rate)))
            rpm = np.full(n, rpm[0])

        cumangle = self._resampler.integrate_rpm(rpm, cfg.rpm_sampling_rate)

        if cfg.nominal_speed_metadata_path is not None and metadata is not None:
            nominal_rpm = self._resolve_metadata_value(cfg.nominal_speed_metadata_path, metadata)
            duration_s = len(rpm) / cfg.rpm_sampling_rate
            expected_rev = nominal_rpm * duration_s / 60.0
            measured_rev = cumangle[-1]
            rel_error = abs(measured_rev - expected_rev) / expected_rev
            if rel_error > 0.20:
                raise ValueError(
                    f"Measured shaft revolutions ({measured_rev:.2f}) deviate from "
                    f"the nominal-speed expectation ({expected_rev:.2f}, from "
                    f"{cfg.nominal_speed_metadata_path}={nominal_rpm}) by "
                    f"{rel_error:.0%} (> 20% tolerance). Likely causes: wrong "
                    "rpm_reader_channel, or speed recorded in Hz rather than rpm."
                )

        angular = self._resampler.resample_to_angular(
            vib,
            cfg.vibration_sampling_rate,
            cumangle,
            cfg.rpm_sampling_rate,
            cfg.target_orders,
            anti_alias=cfg.anti_alias,
        )
        windows = self._resampler.segment_angular(
            angular,
            cfg.target_orders,
            cfg.window_revolutions,
            cfg.window_overlap,
        )
        t = torch.tensor(windows, dtype=torch.float32)  # (N, window_samples)
        return self._view(t)

    def _resolve_metadata_value(self, path: str, metadata) -> float:
        """Dot-path metadata lookup, mirroring
        DomainDataset._resolve_metadata_value (src/experiment/dataset.py)."""
        val = metadata
        for part in path.split('.'):
            val = val[part]
        if isinstance(val, dict):
            if 'value' not in val:
                raise ValueError(
                    f"Metadata path '{path}' resolved to a dict without a 'value' key: {val}"
                )
            val = val['value']
        return float(val)

    def __call__(self, data: npt.ArrayLike, sampling_rate: int) -> torch.Tensor:
        raise NotImplementedError(
            "OrderTrackingProcessor requires both vibration and RPM channels. "
            "DomainDataset should call process(channels) via required_reader_channels — "
            "do not call __call__ directly."
        )
