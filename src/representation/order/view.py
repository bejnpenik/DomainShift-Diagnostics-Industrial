from __future__ import annotations

import torch
import numpy.typing as npt

from ..signal.view import BaseView


class OrderTrackingView(BaseView):
    """Pass angular-domain windows through as-is, adding a channel dimension.

    Input:  (N, L) tensor of angular-domain windows.
    Output: (N, 1, L) tensor — directly usable by 1D CNNs.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1)


class OrderSpectrumView(BaseView):
    """Convert angular-domain windows to log-magnitude order spectra.

    Applies a real FFT along the angular (last) dimension, computes
    log1p of the magnitude, and returns the first n_orders bins.

    Input:  (N, L) tensor of angular-domain windows.
    Output: (N, 1, O) tensor where O = n_orders — usable by 2D CNNs
            treating the order axis as the frequency axis.

    Note: No window function is applied before the FFT. Angular windows
    span whole revolutions by construction (window_revolutions is an
    integer or near-integer multiple of 1/target_orders), so leakage is
    minimal for periodic fault harmonics.
    """

    def __init__(self, n_orders: int = 256) -> None:
        self._n_orders = n_orders

    def __call__(self, x: npt.ArrayLike) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        spectrum = torch.fft.rfft(x, dim=-1)          # (N, L//2+1) complex
        magnitude = torch.abs(spectrum)                # (N, L//2+1) real
        magnitude = torch.log1p(magnitude)             # log-compress
        magnitude = magnitude[:, :self._n_orders]      # (N, O)
        return magnitude.unsqueeze(1)                  # (N, 1, O)
