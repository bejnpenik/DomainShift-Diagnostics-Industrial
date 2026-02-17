import numpy.typing as npt
import numpy as np
import torch

from abc import ABC, abstractmethod


class BaseView(ABC):
    """"""
    @abstractmethod
    def __call__(self, x:npt.ArrayLike)->torch.Tensor:
        """
        Docstring for __call__
        
        :param self: Description
        :param x: Description
        :type x: npt.ArrayLike
        :return: Description
        :rtype: Tensor
        """
        ...

class RawSignalView(BaseView):
    """
    Docstring for RawSignalRepresentation

    """
    def __call__(self, x:torch.Tensor)->torch.Tensor:
        """
        Docstring for __call__
        
        :param self: Description
        :param x: Description
        :type x: npt.ArrayLike
        :return: Description
        :rtype: Tensor
        """
        return x.unsqueeze(1)

class STFTSignalView(BaseView):
    """
    Docstring for SpectrogramRepresentation
 
    """
    def __init__(self, n_fft:int=256, hop_length:int=128, win_length:int=256):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)
    
    def __call__(self, x:npt.ArrayLike)->torch.Tensor:
        """
        Docstring for __call__
        
        :param self: Description
        :param x: Description
        :type x: npt.ArrayLike
        :return: Description
        :rtype: Tensor
        """
        
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )

        x = torch.abs(x)

        x = torch.log1p(x)

        return x.unsqueeze(1)