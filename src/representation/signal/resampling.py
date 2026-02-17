import numpy as np
import numpy.typing as npt
from fractions import Fraction
from scipy.signal import decimate, resample, resample_poly
from math import floor, ceil

from typing import Tuple

class Resampler:
    def __init__(self, max_signal_bandwidth_factor:float=0.5):
        self._max_signal_bandwidth_factor = max_signal_bandwidth_factor

    def _decimate(self, x:npt.ArrayLike, sampling_rate:int, target_sampling_rate:int) -> Tuple:
        q = sampling_rate / target_sampling_rate
        q = floor(q) if abs(floor(q)-q) < abs(ceil(q) - q) else ceil(q)

        return decimate(x, q, ftype = 'iir', zero_phase = True), self._sampling_rate / q
    
    def _resample(self, x:npt.ArrayLike, sampling_rate:int, target_sampling_rate:int)-> Tuple:
        n = x.shape[0]
        n_out = n * target_sampling_rate / sampling_rate

        if n_out.is_integer():
            return resample(x, int(n_out)), target_sampling_rate
        else:
            n_out = ceil(n_out)
            return resample(np.append(x, 0.0), n_out), target_sampling_rate
    def _resample_poly_fast(self, x:npt.ArrayLike, sampling_rate:int, target_sampling_rate:int)->npt.ArrayLike:
        q_max = max(1, floor(sampling_rate / (2 * self._max_signal_bandwidth_factor * target_sampling_rate)))
        q_best, up_best, down_best = 1, 1, 1
        cost_best = float('inf')
        for q in range(1, q_max+1):
            intermediate_sampling_rate = sampling_rate // q
            r = Fraction(target_sampling_rate, intermediate_sampling_rate).limit_denominator(1000)
            cost = max(r.numerator, r.denominator)
            if cost < cost_best:
                cost_best = cost
                q_best = q
                up_best = r.numerator
                down_best = r.denominator

        if q_best > 1:
            x = decimate(x, q_best, ftype = 'iir', zero_phase = False)
        if up_best > 1 or down_best > 1:
            x = resample_poly(x, up=up_best, down=down_best)
        return x

    def __call__(self, x:npt.ArrayLike, sampling_rate:int, target_sampling_rate:int)->npt.ArrayLike:
        return self._resample_poly_fast(x, sampling_rate, target_sampling_rate)