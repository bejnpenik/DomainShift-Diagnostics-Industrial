from __future__ import annotations

import numpy as np
import numpy.typing as npt


class AngularResampler:
    """Convert vibration signals from the time domain to the angular domain.

    Pipeline:
        integrate_rpm  →  resample_to_angular  →  segment_angular

    Limitations (V1):
        - RPM = 0 at any sample makes the cumulative angle flat at that point.
          np.interp handles this gracefully (duplicate x values return the first
          matching y), but the angular grid will over-sample that region.
          Avoid using recordings with stationary intervals for order tracking.
        - Interpolation is linear (np.interp). Upgrade to cubic via
          scipy.interpolate.interp1d for better accuracy at high orders.
    """

    def integrate_rpm(
        self, rpm_signal: npt.ArrayLike, rpm_sr: int
    ) -> np.ndarray:
        """Integrate RPM → cumulative shaft angle in revolutions.

        Args:
            rpm_signal: 1D array of instantaneous speed in RPM, sampled at rpm_sr.
            rpm_sr: Sampling rate of the RPM signal in Hz.

        Returns:
            Monotonically non-decreasing 1D array of cumulative shaft angle in
            revolutions, same length as rpm_signal. Element [n] is the total
            angle reached at time n / rpm_sr.
        """
        rpm = np.asarray(rpm_signal, dtype=np.float64)
        # Each sample contributes rpm[n] revolutions per minute over 1/rpm_sr seconds
        # → rpm[n] / rpm_sr / 60 revolutions per sample
        increments = rpm / (rpm_sr * 60.0)
        return np.cumsum(increments)

    def resample_to_angular(
        self,
        vibration: npt.ArrayLike,
        vib_sr: int,
        cumulative_angle_at_rpm_times: np.ndarray,
        rpm_sr: int,
        target_orders: int,
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

        Returns:
            1D array of the vibration signal resampled onto a uniform angular
            grid. Length ≈ total_revolutions × target_orders (may be slightly
            less due to boundary alignment).

        Raises:
            ValueError: If the vibration signal spans less than one revolution.
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

        # Uniform angular grid: target_orders samples per revolution
        n_angular = int(total_revolutions * target_orders)
        uniform_angle = np.linspace(0.0, total_revolutions, n_angular, endpoint=False)

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
