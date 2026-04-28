import numpy.typing as npt
import numpy as np
import torch

from abc import ABC, abstractmethod


class BaseView(ABC):
    @abstractmethod
    def __call__(self, x: npt.ArrayLike) -> torch.Tensor:
        ...


class RawSignalView(BaseView):
    """Pass segmented windows through as-is, adding a channel dimension.

    Input:  (N, L) tensor of signal windows.
    Output: (N, 1, L) tensor.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1)


class STFTSignalView(BaseView):
    """Convert segmented windows to log-magnitude spectrograms via STFT.

    Input:  (N, L) tensor of signal windows.
    Output: (N, 1, F, T) tensor where F = n_fft//2+1 and T depends on hop_length.
    """

    def __init__(self, n_fft: int = 256, hop_length: int = 128, win_length: int = 256):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)

    def __call__(self, x: npt.ArrayLike) -> torch.Tensor:
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device),
            return_complex=True,
        )
        x = torch.abs(x)
        x = torch.log1p(x)
        return x.unsqueeze(1)
