from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.signal
from scipy.integrate import cumulative_trapezoid


class AngularResampler:
    """Convert vibration signals from the time domain to the angular domain.

    Pipeline:
        integrate_rpm  →  resample_to_angular  →  segment_angular

    integrate_rpm integrates the RPM signal via trapezoidal quadrature into
    cumulative shaft angle (revolutions), starting at exactly 0.0.

    resample_to_angular applies a zero-phase Butterworth anti-aliasing
    lowpass (scipy.signal.sosfiltfilt) before interpolating vibration onto
    the uniform angular grid, with a cutoff derived from the slowest
    1st-percentile instantaneous shaft speed in the recording. Disable via
    anti_alias=False. A strictly decreasing cumulative shaft angle (corrupted
    or negative RPM) raises ValueError before interpolation.

    Limitations (V1):
        - Interpolation onto both the vibration and the angular grid is
          linear (np.interp). Upgrade to cubic via scipy.interpolate.interp1d
          for better accuracy at high orders.
    """

    def integrate_rpm(
        self, rpm_signal: npt.ArrayLike, rpm_sr: int
    ) -> np.ndarray:
        """Integrate RPM → cumulative shaft angle in revolutions (trapezoidal).

        Args:
            rpm_signal: 1D array of instantaneous speed in RPM, sampled at rpm_sr.
            rpm_sr: Sampling rate of the RPM signal in Hz.

        Returns:
            Monotonically non-decreasing 1D array of cumulative shaft angle in
            revolutions, same length as rpm_signal, starting at exactly 0.0
            (element [0] == 0.0). Element [n] is the total angle reached at
            time n / rpm_sr.
        """
        rpm = np.asarray(rpm_signal, dtype=np.float64)
        return cumulative_trapezoid(rpm / 60.0, dx=1.0 / rpm_sr, initial=0.0)

    def resample_to_angular(
        self,
        vibration: npt.ArrayLike,
        vib_sr: int,
        cumulative_angle_at_rpm_times: np.ndarray,
        rpm_sr: int,
        target_orders: int,
        anti_alias: bool = True,
    ) -> np.ndarray:
        """Resample vibration from the time domain to a uniform angular grid.

        Args:
            vibration: 1D vibration signal sampled at vib_sr Hz.
            vib_sr: Vibration sampling rate in Hz.
            cumulative_angle_at_rpm_times: Cumulative shaft angle (revolutions)
                at each RPM sample, as returned by integrate_rpm.
            rpm_sr: Sampling rate of the RPM signal in Hz (used to build its
                time axis).
            target_orders: Number of angular samples per revolution on the
                uniform output grid (analogous to samples-per-second in time).
            anti_alias: If True (default), apply a zero-phase Butterworth
                lowpass (scipy.signal.sosfiltfilt) before interpolation, with
                cutoff derived from the slowest 1st-percentile instantaneous
                shaft speed. Skipped automatically if that cutoff is already
                at or above the vibration Nyquist frequency. Set False to
                disable — angular sampling above target_orders/2 * f_shaft
                will then alias, with the alias location depending on speed.

        Returns:
            1D array of the vibration signal resampled onto a uniform angular
            grid, with exact 1/target_orders sample spacing. Length ≈
            total_revolutions × target_orders (may be slightly less due to
            boundary alignment).

        Raises:
            ValueError: If the vibration signal spans less than one revolution,
                if the cumulative shaft angle is strictly decreasing anywhere
                (corrupted or negative RPM data), or (when anti_alias=True) if
                the estimated minimum shaft speed is non-positive.
        """
        vib = np.asarray(vibration, dtype=np.float64)
        n_vib = len(vib)
        n_rpm = len(cumulative_angle_at_rpm_times)

        # Time axes
        t_rpm = np.arange(n_rpm) / rpm_sr
        t_vib = np.arange(n_vib) / vib_sr

        # Interpolate cumulative angle onto the vibration time axis
        angle_at_vib = np.interp(t_vib, t_rpm, cumulative_angle_at_rpm_times)

        total_revolutions = angle_at_vib[-1]
        if total_revolutions < 1.0:
            raise ValueError(
                f"Signal spans only {total_revolutions:.3f} revolutions — "
                "need at least 1 revolution for angular resampling. "
                "Increase recording length or reduce window_revolutions."
            )

        # Monotonicity guard: strict "< 0", not "<= 0". np.interp legitimately
        # produces exactly-flat (diff == 0) runs at the tail whenever
        # vib_sr > rpm_sr, since t_vib.max() > t_rpm.max() by construction and
        # the tail is extrapolated flat — that is benign. Only a strictly
        # decreasing angle indicates corrupted/negative RPM data.
        if np.any(np.diff(angle_at_vib) < 0):
            raise ValueError(
                "Cumulative shaft angle is strictly decreasing somewhere in "
                "the recording — this indicates corrupted or negative RPM "
                "data (a non-positive speed region). Angular resampling "
                "requires a non-decreasing cumulative shaft angle."
            )

        if anti_alias:
            f_inst = np.gradient(angle_at_vib, 1.0 / vib_sr)
            f_min = np.percentile(f_inst, 1.0)
            if f_min <= 0:
                raise ValueError(
                    f"Estimated minimum instantaneous shaft speed is "
                    f"non-positive ({f_min:.4f} rev/s) — cannot design an "
                    "anti-aliasing filter. Check for stationary/near-zero-RPM "
                    "regions in the recording."
                )
            cutoff_hz = 0.45 * target_orders * f_min
            if cutoff_hz < vib_sr / 2:
                sos = scipy.signal.butter(8, cutoff_hz, btype="low", fs=vib_sr, output="sos")
                vib = scipy.signal.sosfiltfilt(sos, vib)

        # Uniform angular grid: target_orders samples per revolution, with
        # exact 1/target_orders spacing (arange avoids the sub-sample drift
        # linspace(..., endpoint=False) accumulates over long recordings).
        n_angular = int(total_revolutions * target_orders)
        uniform_angle = np.arange(n_angular) / target_orders

        # Interpolate vibration onto the uniform angular grid
        return np.interp(uniform_angle, angle_at_vib, vib)

    def segment_angular(
        self,
        angular_signal: npt.ArrayLike,
        target_orders: int,
        window_revolutions: float,
        window_overlap: float,
    ) -> np.ndarray:
        """Segment an angular-domain signal into overlapping windows.

        Args:
            angular_signal: 1D array in the angular domain (uniform angular grid).
            target_orders: Samples per revolution (the angular sampling rate).
            window_revolutions: Window duration in shaft revolutions.
            window_overlap: Fractional overlap between consecutive windows [0, 1).

        Returns:
            2D array of shape (N, window_samples) where
            window_samples = int(target_orders * window_revolutions).

        Raises:
            ValueError: If the signal is shorter than one window.
        """
        signal = np.asarray(angular_signal, dtype=np.float64)
        window_samples = int(target_orders * window_revolutions)
        if window_samples == 0:
            raise ValueError(
                f"window_samples=0: target_orders={target_orders} × "
                f"window_revolutions={window_revolutions} must be > 0."
            )
        step = max(1, int(window_samples * (1.0 - window_overlap)))

        if len(signal) < window_samples:
            raise ValueError(
                f"Angular signal ({len(signal)} samples) is shorter than one window "
                f"({window_samples} samples = {window_revolutions} rev × "
                f"{target_orders} orders/rev). "
                "Increase recording length or reduce window_revolutions."
            )

        windows = np.lib.stride_tricks.sliding_window_view(signal, window_samples)
        # sliding_window_view returns shape (len-window+1, window_samples)
        # take every `step` row
        return np.ascontiguousarray(windows[::step])
