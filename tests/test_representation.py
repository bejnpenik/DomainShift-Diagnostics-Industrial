"""
Tests for representation.signal package.

Covers: Resampler, SignalSegmenter, RawSignalView, STFTSignalView,
        RawViewConfig, STFTViewConfig, SignalPipelineConfig, SignalPipeline
"""

from __future__ import annotations

import pytest
import numpy as np
import torch

from representation.signal.resampling import Resampler
from representation.signal.segmentation import SignalSegmenter
from representation.signal.view import BaseView, RawSignalView, STFTSignalView
from representation.signal.config import (
    RawViewConfig,
    STFTViewConfig,
    SignalProcessorConfig,
)
from representation.signal.processor import SignalProcessor
from collection import Metadata


# =====================================================================
# Resampler
# =====================================================================

class TestResampler:
    def test_downsample_length(self):
        r = Resampler()
        x = np.random.randn(48000).astype(np.float32)
        result = r(x, 48000, 12000)
        assert abs(len(result) - 12000) < 100

    def test_upsample_length(self):
        r = Resampler()
        x = np.random.randn(12000).astype(np.float32)
        result = r(x, 12000, 48000)
        assert abs(len(result) - 48000) < 100

    def test_same_rate_approximate_identity(self):
        r = Resampler()
        x = np.random.randn(12000).astype(np.float32)
        result = r(x, 12000, 12000)
        assert abs(len(result) - 12000) < 10

    def test_output_is_numpy(self):
        r = Resampler()
        x = np.random.randn(24000).astype(np.float32)
        result = r(x, 24000, 12000)
        assert isinstance(result, np.ndarray)


# =====================================================================
# SignalSegmenter
# =====================================================================

class TestSignalSegmenter:
    def test_basic_windowing(self):
        seg = SignalSegmenter(window_duration=0.01, overlap=0.5)
        data = np.random.randn(1000).astype(np.float32)
        result = seg(data, sampling_rate=10000)
        # window=100, step=50, windows = (1000-100)//50 + 1 = 19
        assert result.shape == (19, 100)

    def test_no_overlap(self):
        seg = SignalSegmenter(window_duration=0.01, overlap=0.0)
        data = np.random.randn(1000).astype(np.float32)
        result = seg(data, sampling_rate=10000)
        # window=100, step=100, windows=10
        assert result.shape == (10, 100)

    def test_output_is_torch(self):
        seg = SignalSegmenter(window_duration=0.05, overlap=0.2)
        data = np.random.randn(12000).astype(np.float32)
        result = seg(data, 12000)
        assert isinstance(result, torch.Tensor)

    def test_window_content(self):
        """First window should match first window_samples of input."""
        seg = SignalSegmenter(window_duration=0.01, overlap=0.0)
        data = np.arange(1000, dtype=np.float32)
        result = seg(data, sampling_rate=10000)
        expected = torch.arange(100, dtype=torch.float32)
        torch.testing.assert_close(result[0], expected)

    def test_unfold_equivalence(self):
        """Must produce same result as manual torch.unfold."""
        seg = SignalSegmenter(window_duration=0.05, overlap=0.2)
        data = np.random.RandomState(42).randn(12000).astype(np.float32)
        result = seg(data, 12000)

        x = torch.from_numpy(data)
        ws = int(0.05 * 12000)
        os = int(ws * 0.2)
        step = ws - os
        expected = x.unfold(0, ws, step)
        torch.testing.assert_close(result, expected)


# =====================================================================
# Views
# =====================================================================

class TestRawSignalView:
    def test_shape(self):
        view = RawSignalView()
        x = torch.randn(10, 600)
        result = view(x)
        assert result.shape == (10, 1, 600)

    def test_values_preserved(self):
        view = RawSignalView()
        x = torch.randn(5, 100)
        result = view(x)
        torch.testing.assert_close(result.squeeze(1), x)

    def test_is_base_view(self):
        assert isinstance(RawSignalView(), BaseView)


class TestSTFTSignalView:
    def test_shape_4d(self):
        view = STFTSignalView(n_fft=256, hop_length=64, win_length=256)
        x = torch.randn(10, 600)
        result = view(x)
        assert result.ndim == 4
        assert result.shape[0] == 10
        assert result.shape[1] == 1

    def test_positive_values(self):
        """log1p(abs(...)) should be >= 0."""
        view = STFTSignalView(n_fft=256, hop_length=64, win_length=256)
        x = torch.randn(5, 600)
        result = view(x)
        assert (result >= 0).all()

    def test_matches_manual_computation(self):
        n_fft, hop, win = 256, 64, 256
        view = STFTSignalView(n_fft=n_fft, hop_length=hop, win_length=win)

        torch.manual_seed(42)
        x = torch.randn(10, 600)
        result = view(x)

        window = torch.hann_window(win)
        manual = torch.stft(
            x, n_fft=n_fft, hop_length=hop, win_length=win,
            window=window, return_complex=True,
        )
        manual = torch.log1p(torch.abs(manual)).unsqueeze(1)
        torch.testing.assert_close(result, manual)

    def test_is_base_view(self):
        assert isinstance(STFTSignalView(n_fft=256, hop_length=64, win_length=256), BaseView)


# =====================================================================
# View Configs
# =====================================================================

class TestRawViewConfig:
    def test_type_field(self):
        cfg = RawViewConfig()
        assert cfg.type == "raw"

    def test_frozen(self):
        cfg = RawViewConfig()
        with pytest.raises(Exception):
            cfg.type = "stft"

    def test_create_view(self):
        cfg = RawViewConfig()
        view = cfg.create_view()
        assert isinstance(view, RawSignalView)


class TestSTFTViewConfig:
    def test_defaults(self):
        cfg = STFTViewConfig()
        assert cfg.n_fft == 256
        assert cfg.hop_length == 128
        assert cfg.win_length == 256

    def test_custom_values(self):
        cfg = STFTViewConfig(n_fft=512, hop_length=64, win_length=512)
        assert cfg.n_fft == 512

    def test_frozen(self):
        cfg = STFTViewConfig()
        with pytest.raises(Exception):
            cfg.n_fft = 512

    def test_invalid_n_fft(self):
        with pytest.raises(Exception):
            STFTViewConfig(n_fft=0)

    def test_create_view(self):
        cfg = STFTViewConfig(n_fft=256, hop_length=64, win_length=256)
        view = cfg.create_view()
        assert isinstance(view, STFTSignalView)


# =====================================================================
# SignalProcessorConfig
# =====================================================================

class TestSignalProcessorConfig:
    def test_raw_config(self):
        cfg = SignalProcessorConfig(name="test", view=RawViewConfig())
        assert cfg.name == "test"
        assert cfg.target_sampling_rate == 12000
        assert isinstance(cfg.view, RawViewConfig)

    def test_stft_config(self):
        cfg = SignalProcessorConfig(
            name="test",
            view=STFTViewConfig(n_fft=512),
        )
        assert isinstance(cfg.view, STFTViewConfig)
        assert cfg.view.n_fft == 512

    def test_frozen(self):
        cfg = SignalProcessorConfig(name="test", view=RawViewConfig())
        with pytest.raises(Exception):
            cfg.name = "changed"

    def test_invalid_overlap(self):
        with pytest.raises(Exception):
            SignalProcessorConfig(name="bad", view=RawViewConfig(), window_overlap=1.5)

    def test_negative_overlap(self):
        with pytest.raises(Exception):
            SignalProcessorConfig(name="bad", view=RawViewConfig(), window_overlap=-0.1)

    def test_invalid_duration(self):
        with pytest.raises(Exception):
            SignalProcessorConfig(name="bad", view=RawViewConfig(), window_duration=-1)

    def test_invalid_sampling_rate(self):
        with pytest.raises(Exception):
            SignalProcessorConfig(name="bad", view=RawViewConfig(), target_sampling_rate=0)

    def test_serialization_roundtrip(self):
        cfg = SignalProcessorConfig(
            name="test",
            target_sampling_rate=48000,
            window_duration=0.1,
            view=STFTViewConfig(n_fft=512, hop_length=64, win_length=512),
        )
        json_str = cfg.model_dump_json()
        restored = SignalProcessorConfig.model_validate_json(json_str)
        assert restored == cfg
        assert isinstance(restored.view, STFTViewConfig)
        assert restored.view.n_fft == 512

    def test_discriminator_raw(self):
        """Pydantic should parse view from dict with type='raw'."""
        cfg = SignalProcessorConfig.model_validate({
            "name": "test",
            "view": {"type": "raw"},
        })
        assert isinstance(cfg.view, RawViewConfig)

    def test_discriminator_stft(self):
        """Pydantic should parse view from dict with type='stft'."""
        cfg = SignalProcessorConfig.model_validate({
            "name": "test",
            "view": {"type": "stft", "n_fft": 512},
        })
        assert isinstance(cfg.view, STFTViewConfig)
        assert cfg.view.n_fft == 512


# =====================================================================
# SignalProcessor
# =====================================================================

class TestSignalProcessor:
    def _meta(self, sr: int = 12000) -> Metadata:
        return Metadata({"sampling_rate": sr})

    def test_raw_pipeline_shape(self):
        cfg = SignalProcessorConfig(
            name="raw_12k",
            target_sampling_rate=12000,
            window_duration=0.05,
            window_overlap=0.5,
            view=RawViewConfig(),
        )
        pipe = SignalProcessor(cfg)
        signal = np.random.randn(12000).astype(np.float32)
        result = pipe(signal, self._meta(12000))
        assert result.ndim == 3
        assert result.shape[1] == 1
        assert result.shape[2] == 600  # 0.05 * 12000

    def test_stft_pipeline_shape(self):
        cfg = SignalProcessorConfig(
            name="stft_12k",
            target_sampling_rate=12000,
            window_duration=0.05,
            window_overlap=0.5,
            view=STFTViewConfig(n_fft=256, hop_length=16, win_length=256),
        )
        pipe = SignalProcessor(cfg)
        signal = np.random.randn(12000).astype(np.float32)
        result = pipe(signal, self._meta(12000))
        assert result.ndim == 4
        assert result.shape[1] == 1

    def test_resampling(self):
        cfg = SignalProcessorConfig(
            name="raw_12k",
            target_sampling_rate=12000,
            window_duration=0.05,
            window_overlap=0.5,
            view=RawViewConfig(),
        )
        pipe = SignalProcessor(cfg)
        signal = np.random.randn(48000).astype(np.float32)
        result = pipe(signal, self._meta(48000))
        assert result.shape[2] == 600  # resampled to 12k, 0.05*12000

    def test_no_resampling_when_same_rate(self):
        cfg = SignalProcessorConfig(
            name="raw_12k",
            target_sampling_rate=12000,
            view=RawViewConfig(),
        )
        pipe = SignalProcessor(cfg)
        signal = np.random.randn(12000).astype(np.float32)
        result = pipe(signal, self._meta(12000))
        # Should work without error — resampler not called
        assert result.ndim == 3

    def test_name_property(self):
        cfg = SignalProcessorConfig(name="my_pipeline", view=RawViewConfig())
        pipe = SignalProcessor(cfg)
        assert pipe.name == "my_pipeline"

    def test_config_property(self):
        cfg = SignalProcessorConfig(name="test", view=RawViewConfig())
        pipe = SignalProcessor(cfg)
        assert pipe.config is cfg

    def test_different_configs_different_outputs(self):
        signal = np.random.randn(12000).astype(np.float32)
        meta = self._meta(12000)

        raw_pipe = SignalProcessor(SignalProcessorConfig(
            name="raw", view=RawViewConfig(),
            window_duration=0.05, window_overlap=0.5,
        ))
        stft_pipe = SignalProcessor(SignalProcessorConfig(
            name="stft", view=STFTViewConfig(n_fft=256, hop_length=16, win_length=256),
            window_duration=0.05, window_overlap=0.5,
        ))

        raw_result = raw_pipe(signal, meta)
        stft_result = stft_pipe(signal, meta)

        assert raw_result.ndim == 3
        assert stft_result.ndim == 4
        assert raw_result.shape[0] == stft_result.shape[0]  # same number of windows