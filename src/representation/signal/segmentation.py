import numpy as np
import numpy.typing as npt
import torch

class SignalSegmenter:
    """
    Docstring for SignalSegmenter
    """
    def __init__(self, window_duration: float = 0.05, overlap: float = 0.5):
        self._window_duration = window_duration
        self._overlap = overlap
    
    def __call__(self, x: npt.ArrayLike, sampling_rate: int) -> npt.ArrayLike:
        """
        Docstring for __call__
        
        :param self: Description
        :param x: Description
        :type x: npt.ArrayLike
        :param sampling_rate: Description
        :type sampling_rate: int
        :return: Description
        :rtype: ArrayLike
        """
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        
        window_samples = int(self._window_duration * sampling_rate)
        overlap_samples = int(window_samples * self._overlap)
        step = window_samples - overlap_samples
        
        return x.unfold(0, window_samples, step)