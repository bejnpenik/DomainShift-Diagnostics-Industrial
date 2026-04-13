from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field

from .view import OrderTrackingView, OrderSpectrumView


class OrderTrackingViewConfig(BaseModel):
    """Config for raw angular-domain output: (N, 1, L)."""
    type: Literal["raw_order"] = "raw_order"

    def create_view(self) -> OrderTrackingView:
        return OrderTrackingView()


class OrderSpectrumViewConfig(BaseModel):
    """Config for order-spectrum output: (N, 1, O)."""
    type: Literal["order_spectrum"] = "order_spectrum"
    n_orders: int = Field(default=256, gt=0)

    def create_view(self) -> OrderSpectrumView:
        return OrderSpectrumView(n_orders=self.n_orders)


OrderViewConfig = Annotated[
    OrderTrackingViewConfig | OrderSpectrumViewConfig,
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
        view: Output representation — raw_order or order_spectrum.
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
    view: OrderViewConfig
