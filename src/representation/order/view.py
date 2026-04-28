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


class OrderSpectrogramView(BaseView):
    """Convert angular-domain windows to log-magnitude order spectrograms via STFT.

    Applies STFT along the angular (revolution) axis of each window, producing a
    2D time-frequency image where the axes are order (fault harmonic number) and
    revolution position. This preserves both spectral content and its evolution
    over the revolution axis, making it compatible with 2D CNN encoders.

    Input:  (N, L) tensor of angular-domain windows.
    Output: (N, 1, F, T) tensor where F = n_fft//2+1 order bins and
            T = number of revolution frames.

    For the default processor (512 orders/rev × 5 rev = 2560 samples/window):
        n_fft=256, hop_length=96  →  (N, 1, 129, 27)
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 96,
        win_length: int = 256,
    ) -> None:
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._win_length = win_length

    def __call__(self, x: npt.ArrayLike) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        win = torch.hann_window(self._win_length, device=x.device)
        spec = torch.stft(
            x,
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            win_length=self._win_length,
            window=win,
            return_complex=True,
        )                                   # (N, F, T) complex
        magnitude = torch.abs(spec)         # (N, F, T) real
        magnitude = torch.log1p(magnitude)  # log-compress
        return magnitude.unsqueeze(1)       # (N, 1, F, T)


class OrderSpectrumView(BaseView):
    """Convert angular-domain windows to log-magnitude order spectra.

    Applies an optional window function, then a real FFT along the angular
    (last) dimension, computes log1p of the magnitude, and returns the first
    n_orders bins.

    Input:  (N, L) tensor of angular-domain windows.
    Output: (N, 1, O) tensor where O = n_orders — usable by 1D CNNs treating
            the order axis as the feature axis.

    Args:
        n_orders: Number of order bins to retain (output length O).
        window_function: Optional window to apply before FFT. "hann" reduces
            spectral leakage when window_revolutions is non-integer. "none"
            (default) applies no windowing — leakage is minimal for integer
            window_revolutions since fault harmonics land on exact FFT bins.
    """

    def __init__(self, n_orders: int = 256, window_function: str = "none") -> None:
        self._n_orders = n_orders
        if window_function not in ("none", "hann"):
            raise ValueError(f"window_function must be 'none' or 'hann', got '{window_function}'")
        self._window_function = window_function

    def __call__(self, x: npt.ArrayLike) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        if self._window_function == "hann":
            win = torch.hann_window(x.shape[-1], device=x.device)
            x = x * win
        spectrum = torch.fft.rfft(x, dim=-1)          # (N, L//2+1) complex
        magnitude = torch.abs(spectrum)                # (N, L//2+1) real
        magnitude = torch.log1p(magnitude)             # log-compress
        magnitude = magnitude[:, :self._n_orders]      # (N, O)
        return magnitude.unsqueeze(1)                  # (N, 1, O)
