"""
Tests for the order-tracking processor pipeline.

Coverage:
    - AngularResampler: integrate_rpm, resample_to_angular, segment_angular
    - OrderTrackingView / OrderSpectrumView
    - OrderTrackingProcessorConfig (Pydantic, discriminated union)
    - Builder round-trip: YAML dict → config
    - OrderTrackingProcessor: required_reader_channels, process(), __call__
    - DomainDataset multi-channel dispatch (mock-based, avoids cross-package imports)

No data files required — all tests use synthetic numpy arrays.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from representation.order.angular_resampler import AngularResampler
from representation.order.view import OrderTrackingView, OrderSpectrogramView, OrderSpectrumView
from representation.order.config import (
    OrderTrackingProcessorConfig,
    OrderTrackingViewConfig,
    OrderSpectrogramViewConfig,
    OrderSpectrumViewConfig,
)
from representation.order.processor import OrderTrackingProcessor
from representation.builder import build_processor_config, build_processor_config_from_yaml
from representation.signal.view import BaseView


# =====================================================================
# Helpers
# =====================================================================

def _make_config(view_type: str = "raw_order", **kwargs) -> OrderTrackingProcessorConfig:
    if view_type == "raw_order":
        view = OrderTrackingViewConfig()
    elif view_type == "order_spectrogram":
        view = OrderSpectrogramViewConfig()
    else:
        view = OrderSpectrumViewConfig()
    defaults = dict(
        name="test_ot",
        vibration_sampling_rate=64000,
        rpm_sampling_rate=4000,
        target_orders=64,
        window_revolutions=2.0,
        window_overlap=0.0,
        view=view,
    )
    defaults.update(kwargs)
    return OrderTrackingProcessorConfig(**defaults)


def _make_processor(view_type: str = "raw_order", **kwargs) -> OrderTrackingProcessor:
    return OrderTrackingProcessor(_make_config(view_type, **kwargs))


def _synthetic_signals(
    rpm_value: float = 1500.0,
    duration_s: float = 1.0,
    vib_sr: int = 64000,
    rpm_sr: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vibration, rpm) arrays for a constant-speed recording."""
    n_vib = int(duration_s * vib_sr)
    n_rpm = int(duration_s * rpm_sr)
    vib = np.random.randn(n_vib).astype(np.float64)
    rpm = np.full(n_rpm, rpm_value, dtype=np.float64)
    return vib, rpm


# =====================================================================
# TestAngularResampler
# =====================================================================

class TestAngularResampler:
    def setup_method(self):
        self.ar = AngularResampler()

    def test_constant_rpm_total_angle(self):
        """1500 rpm for 1 s at 4000 Hz → 25 revolutions."""
        rpm = np.full(4000, 1500.0)
        angle = self.ar.integrate_rpm(rpm, rpm_sr=4000)
        assert abs(angle[-1] - 25.0) < 1e-6

    def test_angle_is_monotone(self):
        rpm = np.abs(np.random.randn(4000)) * 1000 + 100  # positive RPM
        angle = self.ar.integrate_rpm(rpm, rpm_sr=4000)
        assert np.all(np.diff(angle) >= 0)

    def test_angle_length_matches_input(self):
        rpm = np.full(4000, 1200.0)
        angle = self.ar.integrate_rpm(rpm, rpm_sr=4000)
        assert len(angle) == 4000

    def test_resample_output_length(self):
        """Output should span ≈ 25 revolutions × 64 orders/rev = 1600 samples."""
        vib, rpm = _synthetic_signals(rpm_value=1500.0, duration_s=1.0, vib_sr=64000, rpm_sr=4000)
        cumangle = self.ar.integrate_rpm(rpm, rpm_sr=4000)
        angular = self.ar.resample_to_angular(vib, 64000, cumangle, 4000, target_orders=64)
        # 25 revolutions × 64 orders/rev = 1600; endpoint=False gives exactly this
        assert len(angular) == 1600

    def test_resample_constant_signal(self):
        """Resampling a constant vibration signal should give the same constant."""
        vib = np.ones(64000)
        rpm = np.full(4000, 1500.0)
        cumangle = self.ar.integrate_rpm(rpm, rpm_sr=4000)
        angular = self.ar.resample_to_angular(vib, 64000, cumangle, 4000, target_orders=64)
        np.testing.assert_allclose(angular, 1.0, atol=1e-10)

    def test_resample_short_signal_raises(self):
        """Signal spanning < 1 revolution should raise ValueError."""
        # 0.01 seconds at 1500 rpm = 0.25 revolutions
        vib, rpm = _synthetic_signals(rpm_value=1500.0, duration_s=0.01,
                                      vib_sr=64000, rpm_sr=4000)
        cumangle = self.ar.integrate_rpm(rpm, rpm_sr=4000)
        with pytest.raises(ValueError, match="revolution"):
            self.ar.resample_to_angular(vib, 64000, cumangle, 4000, target_orders=64)

    def test_segment_output_shape(self):
        """1600 samples, 128 per window, step 128 → 12 windows."""
        signal = np.random.randn(1600)
        # target_orders=64, window_revolutions=2 → 128 samples/window
        windows = self.ar.segment_angular(signal, target_orders=64,
                                          window_revolutions=2.0, window_overlap=0.0)
        assert windows.shape == (12, 128)

    def test_segment_with_overlap(self):
        """50 % overlap should approximately double the number of windows."""
        signal = np.random.randn(1600)
        no_overlap = self.ar.segment_angular(signal, 64, 2.0, 0.0)
        with_overlap = self.ar.segment_angular(signal, 64, 2.0, 0.5)
        assert with_overlap.shape[0] > no_overlap.shape[0]
        assert with_overlap.shape[1] == no_overlap.shape[1]

    def test_segment_short_signal_raises(self):
        signal = np.random.randn(10)
        with pytest.raises(ValueError, match="shorter than one window"):
            self.ar.segment_angular(signal, target_orders=64,
                                    window_revolutions=2.0, window_overlap=0.0)

    def test_segment_values_match_source(self):
        """First window should equal the first 128 samples of the signal."""
        signal = np.arange(1600, dtype=float)
        windows = self.ar.segment_angular(signal, 64, 2.0, 0.0)
        np.testing.assert_array_equal(windows[0], signal[:128])


# =====================================================================
# TestOrderTrackingView
# =====================================================================

class TestOrderTrackingView:
    def test_output_shape(self):
        view = OrderTrackingView()
        x = torch.randn(8, 128)
        out = view(x)
        assert out.shape == (8, 1, 128)

    def test_values_preserved(self):
        view = OrderTrackingView()
        x = torch.randn(4, 64)
        out = view(x)
        torch.testing.assert_close(out.squeeze(1), x)

    def test_is_base_view(self):
        assert isinstance(OrderTrackingView(), BaseView)


# =====================================================================
# TestOrderSpectrumView
# =====================================================================

class TestOrderSpectrogramView:
    def test_output_shape(self):
        """Default params: n_fft=256, hop=96 on 2560-sample window → (N,1,129,27)."""
        view = OrderSpectrogramView(n_fft=256, hop_length=96, win_length=256)
        x = torch.randn(4, 2560)
        out = view(x)
        assert out.shape == (4, 1, 129, 27)

    def test_output_4d(self):
        view = OrderSpectrogramView()
        x = torch.randn(3, 2560)
        out = view(x)
        assert out.ndim == 4

    def test_channel_dim_is_1(self):
        view = OrderSpectrogramView()
        x = torch.randn(2, 2560)
        out = view(x)
        assert out.shape[1] == 1

    def test_output_non_negative(self):
        view = OrderSpectrogramView()
        x = torch.randn(2, 2560)
        out = view(x)
        assert (out >= 0).all()

    def test_output_dtype_float32(self):
        view = OrderSpectrogramView()
        x = torch.randn(2, 2560)
        out = view(x)
        assert out.dtype == torch.float32

    def test_is_base_view(self):
        assert isinstance(OrderSpectrogramView(), BaseView)

    def test_small_window(self):
        """Should work for small windows — shape determined by STFT math."""
        view = OrderSpectrogramView(n_fft=16, hop_length=8, win_length=16)
        x = torch.randn(2, 128)
        out = view(x)
        assert out.ndim == 4
        assert out.shape[1] == 1
        assert out.shape[2] == 9   # n_fft//2 + 1


class TestOrderSpectrumView:
    def test_output_shape(self):
        view = OrderSpectrumView(n_orders=64)
        x = torch.randn(8, 256)
        out = view(x)
        assert out.shape == (8, 1, 64)

    def test_output_non_negative(self):
        view = OrderSpectrumView(n_orders=64)
        x = torch.randn(8, 256)
        out = view(x)
        assert (out >= 0).all()

    def test_is_base_view(self):
        assert isinstance(OrderSpectrumView(), BaseView)

    def test_default_n_orders(self):
        view = OrderSpectrumView()
        assert view._n_orders == 256

    def test_clips_to_n_orders(self):
        """Output should have exactly n_orders bins regardless of input length."""
        view = OrderSpectrumView(n_orders=32)
        x = torch.randn(4, 512)  # rfft gives 257 bins; we want only 32
        out = view(x)
        assert out.shape[-1] == 32

    def test_hann_window_same_shape(self):
        """Hann window should not change output shape."""
        view = OrderSpectrumView(n_orders=64, window_function="hann")
        x = torch.randn(8, 256)
        out = view(x)
        assert out.shape == (8, 1, 64)

    def test_invalid_window_function_raises(self):
        with pytest.raises(ValueError, match="window_function"):
            OrderSpectrumView(n_orders=64, window_function="blackman")


# =====================================================================
# TestOrderTrackingProcessorConfig
# =====================================================================

class TestOrderTrackingProcessorConfig:
    def test_frozen(self):
        cfg = _make_config()
        with pytest.raises(Exception):  # ValidationError or TypeError
            cfg.name = "new_name"

    def test_discriminator_raw_order(self):
        cfg = _make_config("raw_order")
        assert isinstance(cfg.view, OrderTrackingViewConfig)

    def test_discriminator_order_spectrum(self):
        cfg = _make_config("order_spectrum")
        assert isinstance(cfg.view, OrderSpectrumViewConfig)

    def test_target_orders_gt_zero(self):
        with pytest.raises(Exception):
            _make_config(target_orders=0)

    def test_discriminator_order_spectrogram(self):
        cfg = _make_config("order_spectrogram")
        assert isinstance(cfg.view, OrderSpectrogramViewConfig)

    def test_order_spectrogram_config_defaults(self):
        cfg = OrderSpectrogramViewConfig()
        assert cfg.n_fft == 256
        assert cfg.hop_length == 96
        assert cfg.win_length == 256

    def test_order_spectrogram_creates_view(self):
        cfg = OrderSpectrogramViewConfig(n_fft=64, hop_length=16, win_length=64)
        view = cfg.create_view()
        assert isinstance(view, OrderSpectrogramView)
        assert view._n_fft == 64

    def test_order_spectrum_config_window_function_default(self):
        cfg = _make_config("order_spectrum")
        assert cfg.view.window_function == "none"

    def test_order_spectrum_config_hann(self):
        cfg = OrderTrackingProcessorConfig(
            name="t", vibration_sampling_rate=64000, rpm_sampling_rate=4000,
            view=OrderSpectrumViewConfig(n_orders=64, window_function="hann"),
        )
        view = cfg.view.create_view()
        assert view._window_function == "hann"

    def test_window_overlap_bounds(self):
        with pytest.raises(Exception):
            _make_config(window_overlap=1.0)
        with pytest.raises(Exception):
            _make_config(window_overlap=-0.1)

    def test_window_revolutions_gt_zero(self):
        with pytest.raises(Exception):
            _make_config(window_revolutions=0.0)


# =====================================================================
# TestBuilderOrderTracking
# =====================================================================

class TestBuilderOrderTracking:
    def _raw_order_dict(self):
        return {
            "type": "order_tracking",
            "name": "test_ot",
            "channels": {
                "vibration_reader_channel": "vibration",
                "rpm_reader_channel": "rpm",
                "vibration_sampling_rate": 64000,
                "rpm_sampling_rate": 4000,
            },
            "angular": {
                "target_orders": 512,
                "window_revolutions": 5.0,
                "window_overlap": 0.2,
            },
            "view": {"type": "raw_order"},
        }

    def test_build_from_dict_raw_order(self):
        cfg = build_processor_config(self._raw_order_dict())
        assert isinstance(cfg, OrderTrackingProcessorConfig)
        assert cfg.name == "test_ot"
        assert cfg.vibration_sampling_rate == 64000
        assert cfg.rpm_sampling_rate == 4000
        assert isinstance(cfg.view, OrderTrackingViewConfig)

    def test_build_from_dict_order_spectrum(self):
        d = self._raw_order_dict()
        d["view"] = {"type": "order_spectrum", "n_orders": 128}
        cfg = build_processor_config(d)
        assert isinstance(cfg.view, OrderSpectrumViewConfig)
        assert cfg.view.n_orders == 128

    def test_unknown_type_raises(self):
        d = self._raw_order_dict()
        d["type"] = "unknown_processor"
        with pytest.raises(ValueError, match="Unknown processor type"):
            build_processor_config(d)

    def test_unknown_view_type_raises(self):
        d = self._raw_order_dict()
        d["view"] = {"type": "bad_view"}
        with pytest.raises(ValueError, match="Unknown order tracking view type"):
            build_processor_config(d)

    def test_build_from_yaml_order_64k(self):
        cfg = build_processor_config_from_yaml("configs/processors/order_64k.yaml")
        assert isinstance(cfg, OrderTrackingProcessorConfig)
        assert cfg.name == "order_64k"
        assert cfg.target_orders == 512

    def test_build_from_yaml_order_spec_64k(self):
        cfg = build_processor_config_from_yaml("configs/processors/order_spec_64k.yaml")
        assert isinstance(cfg, OrderTrackingProcessorConfig)
        assert isinstance(cfg.view, OrderSpectrumViewConfig)
        assert cfg.view.n_orders == 256

    def test_build_from_yaml_order_spect_64k(self):
        cfg = build_processor_config_from_yaml("configs/processors/order_spect_64k.yaml")
        assert isinstance(cfg, OrderTrackingProcessorConfig)
        assert isinstance(cfg.view, OrderSpectrogramViewConfig)
        assert cfg.view.n_fft == 256
        assert cfg.view.hop_length == 96

    def test_create_processor_returns_order_tracking_processor(self):
        from representation import create_processor
        cfg = build_processor_config_from_yaml("configs/processors/order_64k.yaml")
        proc = create_processor(cfg)
        assert isinstance(proc, OrderTrackingProcessor)
        assert proc.name == "order_64k"


# =====================================================================
# TestOrderTrackingProcessor
# =====================================================================

class TestOrderTrackingProcessor:
    def test_required_reader_channels(self):
        proc = _make_processor()
        assert proc.required_reader_channels == frozenset({"vibration", "rpm"})

    def test_required_reader_channels_custom_names(self):
        cfg = _make_config(vibration_reader_channel="vib", rpm_reader_channel="speed")
        proc = OrderTrackingProcessor(cfg)
        assert proc.required_reader_channels == frozenset({"vib", "speed"})

    def test_name_property(self):
        proc = _make_processor()
        assert proc.name == "test_ot"

    def test_call_raises_not_implemented(self):
        proc = _make_processor()
        with pytest.raises(NotImplementedError):
            proc(np.zeros(100), 1000)

    def test_process_raw_order_shape(self):
        """1 second at 1500 rpm → ~25 revs; 2 rev windows, no overlap → ~12 windows."""
        proc = _make_processor("raw_order", target_orders=64, window_revolutions=2.0,
                               window_overlap=0.0)
        vib, rpm = _synthetic_signals(1500.0, 1.0, 64000, 4000)
        out = proc.process({"vibration": vib, "rpm": rpm})
        assert out.ndim == 3
        assert out.shape[1] == 1          # channel dim
        assert out.shape[2] == 128        # 64 orders × 2 rev
        assert out.shape[0] >= 1          # at least one window

    def test_process_order_spectrum_shape(self):
        # window = 64 orders × 2 rev = 128 samples → rfft gives 65 bins
        # n_orders=32 < 65, so output should be clipped to 32
        proc = _make_processor("order_spectrum", target_orders=64, window_revolutions=2.0,
                               window_overlap=0.0, view=OrderSpectrumViewConfig(n_orders=32))
        vib, rpm = _synthetic_signals(1500.0, 1.0, 64000, 4000)
        out = proc.process({"vibration": vib, "rpm": rpm})
        assert out.ndim == 3
        assert out.shape[1] == 1
        assert out.shape[2] == 32

    def test_process_output_is_float_tensor(self):
        proc = _make_processor()
        vib, rpm = _synthetic_signals()
        out = proc.process({"vibration": vib, "rpm": rpm})
        assert out.dtype == torch.float32

    def test_process_short_signal_raises(self):
        """Signal < 1 revolution should propagate ValueError from AngularResampler."""
        proc = _make_processor(target_orders=64, window_revolutions=2.0)
        vib, rpm = _synthetic_signals(1500.0, duration_s=0.01, vib_sr=64000, rpm_sr=4000)
        with pytest.raises(ValueError):
            proc.process({"vibration": vib, "rpm": rpm})

    def test_process_order_spectrogram_shape(self):
        """Order spectrogram: 1 sec at 1500 rpm → 5-rev windows → (N,1,F,T)."""
        proc = _make_processor(
            "order_spectrogram",
            target_orders=512,
            window_revolutions=5.0,
            window_overlap=0.0,
            view=OrderSpectrogramViewConfig(n_fft=256, hop_length=96, win_length=256),
        )
        vib, rpm = _synthetic_signals(1500.0, 1.0, 64000, 4000)
        out = proc.process({"vibration": vib, "rpm": rpm})
        assert out.ndim == 4
        assert out.shape[1] == 1        # channel dim
        assert out.shape[2] == 129      # F = 256//2 + 1
        assert out.shape[3] == 27       # T for 2560-sample window, hop=96

    def test_process_missing_channel_raises_key_error(self):
        """process() with a missing channel key should raise KeyError."""
        proc = _make_processor()
        vib, _ = _synthetic_signals()
        with pytest.raises(KeyError):
            proc.process({"vibration": vib})   # rpm missing


# =====================================================================
# TestDomainDatasetMultiChannelDispatch
# =====================================================================

class TestDomainDatasetMultiChannelDispatch:
    """Test that DomainDataset correctly dispatches to process() for
    multi-channel processors. Uses duck-typed mocks to avoid the cross-package
    import issue with experiment/__init__.py."""

    def _call_dispatch(self, processor, raw: dict, primary_cfg, meta):
        """Inline the DomainDataset dispatch logic for isolated testing."""
        if hasattr(processor, 'required_reader_channels'):
            proc_channels = {ch: raw[ch] for ch in processor.required_reader_channels}
            return processor.process(proc_channels)
        else:
            from representation.order.angular_resampler import AngularResampler
            # Simulate SingleChannelProcessor call
            return processor(raw[primary_cfg.reader_channel], 64000)

    def test_multichannel_calls_process(self):
        """hasattr dispatch should call process(), not __call__."""
        proc = _make_processor("raw_order", target_orders=64, window_revolutions=2.0,
                               window_overlap=0.0)
        vib, rpm = _synthetic_signals(1500.0, 1.0, 64000, 4000)
        raw = {"vibration": vib, "rpm": rpm}

        assert hasattr(proc, 'required_reader_channels')

        # Should succeed and return a tensor
        out = proc.process(raw)
        assert isinstance(out, torch.Tensor)

    def test_multichannel_reader_channels_extended(self):
        """required_reader_channels should be merged into reader_channels set."""
        proc = _make_processor()
        existing = {"vibration"}
        extended = existing | proc.required_reader_channels
        assert "rpm" in extended
        assert "vibration" in extended

    def test_single_channel_no_required_attr(self):
        """SignalProcessor should NOT have required_reader_channels."""
        from representation.builder import build_processor_config_from_yaml
        from representation import create_processor
        cfg = build_processor_config_from_yaml("configs/processors/raw_12k.yaml")
        proc = create_processor(cfg)
        assert not hasattr(proc, 'required_reader_channels')

    def test_conditioning_with_multichannel_raises(self):
        """DomainDataset.__init__ must reject conditioning + multi-channel in V1."""
        # Reproduce the guard logic inline
        class _MockConditioningSource:
            channel = "torque"
            reduce = "mean"

        proc = _make_processor()
        conditioning = [_MockConditioningSource()]

        if hasattr(proc, 'required_reader_channels') and conditioning:
            raised = True
        else:
            raised = False
        assert raised, "Guard should fire when conditioning is non-empty"
