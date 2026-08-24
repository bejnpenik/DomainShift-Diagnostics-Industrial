from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field

from .view import OrderTrackingView, OrderSpectrogramView, OrderSpectrumView


class OrderTrackingViewConfig(BaseModel):
    """Config for raw angular-domain output: (N, 1, L)."""
    type: Literal["raw_order"] = "raw_order"

    def create_view(self) -> OrderTrackingView:
        return OrderTrackingView()


class OrderSpectrogramViewConfig(BaseModel):
    """Config for order-spectrogram output: (N, 1, F, T).

    Applies STFT to angular windows, producing a 2D order-vs-revolution image
    compatible with 2D CNN encoders. F = n_fft//2+1, T depends on window length
    and hop_length.

    For the default 2560-sample window (512 orders/rev × 5 rev):
        n_fft=256, hop_length=96  →  F=129, T=27
    """
    type: Literal["order_spectrogram"] = "order_spectrogram"
    n_fft: int = Field(default=256, gt=0)
    hop_length: int = Field(default=96, gt=0)
    win_length: int = Field(default=256, gt=0)

    def create_view(self) -> OrderSpectrogramView:
        return OrderSpectrogramView(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )


class OrderSpectrumViewConfig(BaseModel):
    """Config for order-spectrum output: (N, 1, O)."""
    type: Literal["order_spectrum"] = "order_spectrum"
    n_orders: int = Field(default=256, gt=0)
    window_function: Literal["none", "hann"] = "none"

    def create_view(self) -> OrderSpectrumView:
        return OrderSpectrumView(n_orders=self.n_orders, window_function=self.window_function)


OrderViewConfig = Annotated[
    OrderTrackingViewConfig | OrderSpectrogramViewConfig | OrderSpectrumViewConfig,
    Discriminator("type"),
]


class OrderTrackingProcessorConfig(BaseModel):
    """Full specification for an order-tracking processor.

    Fields:
        name: Identifier used in result exports.
        vibration_reader_channel: Key in the reader's output dict for vibration.
        rpm_reader_channel: Key in the reader's output dict for RPM.
        vibration_sampling_rate: Vibration signal sampling rate in Hz.
        rpm_sampling_rate: RPM signal sampling rate in Hz.
        target_orders: Angular sampling rate — samples per shaft revolution
            on the uniform angular grid.
        window_revolutions: Window duration in shaft revolutions.
        window_overlap: Fractional overlap between consecutive windows [0, 1).
        anti_alias: If True (default), apply a zero-phase Butterworth lowpass
            filter before angular resampling, with a cutoff derived from the
            slowest 1st-percentile instantaneous shaft speed in the recording.
        nominal_speed_metadata_path: Optional dot-path into sample metadata
            (e.g. "condition.speed") used to sanity-check the measured
            integrated revolutions against a nominal RPM. None disables the
            check.
        view: Output representation — raw_order, order_spectrum, or
            order_spectrogram.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    vibration_reader_channel: str = "vibration"
    rpm_reader_channel: str = "rpm"
    vibration_sampling_rate: int = Field(gt=0)
    rpm_sampling_rate: int = Field(gt=0)
    target_orders: int = Field(default=512, gt=0)
    window_revolutions: float = Field(default=5.0, gt=0)
    window_overlap: float = Field(default=0.2, ge=0, lt=1)
    anti_alias: bool = True
    nominal_speed_metadata_path: str | None = None
    view: OrderViewConfig
