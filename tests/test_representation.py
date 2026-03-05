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


"""
Tests for representation.builder — processor YAML builder.

Covers:
    - build_processor_config from dict (raw, stft)
    - build_processor_config_from_yaml roundtrip
    - Defaults (missing optional fields)
    - Validation (unknown type, unknown view, missing type)
    - Auto-naming when name is omitted
    - Config protocol satisfaction (name property, create_processor)
    - Equivalence with direct Python construction
    - validate_processor_yaml helper
"""

#from __future__ import annotations

import pytest
import tempfile
from pathlib import Path


# =====================================================================
# Helper: write YAML to temp file
# =====================================================================

def _write_yaml(cfg_dict: dict) -> Path:
    import yaml
    path = Path(tempfile.mktemp(suffix=".yaml"))
    with open(path, "w") as f:
        yaml.dump(cfg_dict, f)
    return path


# =====================================================================
# 1. Build from dict — raw
# =====================================================================

class TestBuildRawFromDict:

    def test_basic_raw(self):
        from representation.builder import build_processor_config
        from representation.signal.config import SignalProcessorConfig, RawViewConfig

        cfg = {
            "type": "signal",
            "name": "raw_12k",
            "resampling": {"target_sampling_rate": 12000},
            "segmentation": {"window_duration": 0.05, "window_overlap": 0.2},
            "view": {"type": "raw"},
        }
        result = build_processor_config(cfg)

        assert isinstance(result, SignalProcessorConfig)
        assert result.name == "raw_12k"
        assert result.target_sampling_rate == 12000
        assert result.window_duration == 0.05
        assert result.window_overlap == 0.2
        assert isinstance(result.view, RawViewConfig)

    def test_raw_with_bandwidth_factor(self):
        from representation.builder import build_processor_config

        cfg = {
            "type": "signal",
            "name": "raw_48k",
            "resampling": {
                "target_sampling_rate": 48000,
                "max_bandwidth_factor": 0.3,
            },
            "segmentation": {"window_duration": 0.1, "window_overlap": 0.5},
            "view": {"type": "raw"},
        }
        result = build_processor_config(cfg)

        assert result.target_sampling_rate == 48000
        assert result.max_signal_bandwidth_factor == 0.3
        assert result.window_duration == 0.1
        assert result.window_overlap == 0.5

    def test_raw_different_rates(self):
        from representation.builder import build_processor_config

        for sr in [12000, 48000, 64000]:
            cfg = {
                "type": "signal",
                "name": f"raw_{sr // 1000}k",
                "resampling": {"target_sampling_rate": sr},
                "segmentation": {"window_duration": 0.05, "window_overlap": 0.2},
                "view": {"type": "raw"},
            }
            result = build_processor_config(cfg)
            assert result.target_sampling_rate == sr
            assert result.name == f"raw_{sr // 1000}k"


# =====================================================================
# 2. Build from dict — STFT
# =====================================================================

class TestBuildSTFTFromDict:

    def test_basic_stft(self):
        from representation.builder import build_processor_config
        from representation.signal.config import SignalProcessorConfig, STFTViewConfig

        cfg = {
            "type": "signal",
            "name": "spec_12k",
            "resampling": {"target_sampling_rate": 12000},
            "segmentation": {"window_duration": 0.05, "window_overlap": 0.2},
            "view": {
                "type": "stft",
                "n_fft": 256,
                "hop_length": 16,
                "win_length": 256,
            },
        }
        result = build_processor_config(cfg)

        assert isinstance(result, SignalProcessorConfig)
        assert result.name == "spec_12k"
        assert isinstance(result.view, STFTViewConfig)
        assert result.view.n_fft == 256
        assert result.view.hop_length == 16
        assert result.view.win_length == 256

    def test_stft_different_hop_lengths(self):
        from representation.builder import build_processor_config

        for hop in [16, 64, 96, 128]:
            cfg = {
                "type": "signal",
                "name": f"spec_hop{hop}",
                "resampling": {"target_sampling_rate": 48000},
                "segmentation": {"window_duration": 0.05, "window_overlap": 0.2},
                "view": {"type": "stft", "n_fft": 256, "hop_length": hop, "win_length": 256},
            }
            result = build_processor_config(cfg)
            assert result.view.hop_length == hop


# =====================================================================
# 3. Build from YAML file (roundtrip)
# =====================================================================

class TestBuildFromYAML:

    def test_raw_yaml_roundtrip(self):
        from representation.builder import build_processor_config_from_yaml
        from representation.signal.config import RawViewConfig

        cfg = {
            "type": "signal",
            "name": "raw_12k",
            "resampling": {"target_sampling_rate": 12000, "max_bandwidth_factor": 0.5},
            "segmentation": {"window_duration": 0.05, "window_overlap": 0.2},
            "view": {"type": "raw"},
        }
        path = _write_yaml(cfg)
        try:
            result = build_processor_config_from_yaml(path)
            assert result.name == "raw_12k"
            assert result.target_sampling_rate == 12000
            assert isinstance(result.view, RawViewConfig)
        finally:
            path.unlink(missing_ok=True)

    def test_stft_yaml_roundtrip(self):
        from representation.builder import build_processor_config_from_yaml
        from representation.signal.config import STFTViewConfig

        cfg = {
            "type": "signal",
            "name": "spec_48k",
            "resampling": {"target_sampling_rate": 48000},
            "segmentation": {"window_duration": 0.05, "window_overlap": 0.2},
            "view": {"type": "stft", "n_fft": 256, "hop_length": 64, "win_length": 256},
        }
        path = _write_yaml(cfg)
        try:
            result = build_processor_config_from_yaml(path)
            assert result.name == "spec_48k"
            assert isinstance(result.view, STFTViewConfig)
            assert result.view.hop_length == 64
        finally:
            path.unlink(missing_ok=True)

    def test_file_not_found(self):
        from representation.builder import build_processor_config_from_yaml

        with pytest.raises(FileNotFoundError):
            build_processor_config_from_yaml("/nonexistent/path.yaml")


# =====================================================================
# 4. Defaults
# =====================================================================

class TestDefaults:

    def test_missing_name_auto_generates(self):
        from representation.builder import build_processor_config

        cfg = {
            "type": "signal",
            "resampling": {"target_sampling_rate": 12000},
            "view": {"type": "raw"},
        }
        result = build_processor_config(cfg)
        assert result.name == "raw_12000"

    def test_missing_name_stft(self):
        from representation.builder import build_processor_config

        cfg = {
            "type": "signal",
            "resampling": {"target_sampling_rate": 48000},
            "view": {"type": "stft"},
        }
        result = build_processor_config(cfg)
        assert result.name == "stft_48000"

    def test_missing_resampling_defaults(self):
        from representation.builder import build_processor_config

        cfg = {
            "type": "signal",
            "name": "defaults_test",
            "view": {"type": "raw"},
        }
        result = build_processor_config(cfg)
        assert result.target_sampling_rate == 12000
        assert result.max_signal_bandwidth_factor == 0.5

    def test_missing_segmentation_defaults(self):
        from representation.builder import build_processor_config

        cfg = {
            "type": "signal",
            "name": "defaults_test",
            "resampling": {"target_sampling_rate": 12000},
            "view": {"type": "raw"},
        }
        result = build_processor_config(cfg)
        assert result.window_duration == 0.05
        assert result.window_overlap == 0.5

    def test_missing_view_defaults_to_raw(self):
        from representation.builder import build_processor_config
        from representation.signal.config import RawViewConfig

        cfg = {
            "type": "signal",
            "name": "no_view",
            "resampling": {"target_sampling_rate": 12000},
        }
        result = build_processor_config(cfg)
        assert isinstance(result.view, RawViewConfig)

    def test_stft_view_defaults(self):
        from representation.builder import build_processor_config

        cfg = {
            "type": "signal",
            "name": "stft_defaults",
            "view": {"type": "stft"},
        }
        result = build_processor_config(cfg)
        assert result.view.n_fft == 256
        assert result.view.hop_length == 128
        assert result.view.win_length == 256

    def test_bandwidth_factor_default(self):
        from representation.builder import build_processor_config

        cfg = {
            "type": "signal",
            "name": "bw_default",
            "resampling": {"target_sampling_rate": 48000},
            "view": {"type": "raw"},
        }
        result = build_processor_config(cfg)
        assert result.max_signal_bandwidth_factor == 0.5


# =====================================================================
# 5. Validation errors
# =====================================================================

class TestValidation:

    def test_missing_type_field(self):
        from representation.builder import build_processor_config

        with pytest.raises(ValueError, match="must have a 'type' field"):
            build_processor_config({"name": "no_type", "view": {"type": "raw"}})

    def test_unknown_processor_type(self):
        from representation.builder import build_processor_config

        with pytest.raises(ValueError, match="Unknown processor type"):
            build_processor_config({"type": "image", "name": "test"})

    def test_unknown_view_type(self):
        from representation.builder import build_processor_config

        with pytest.raises(ValueError, match="Unknown view type"):
            build_processor_config({
                "type": "signal",
                "name": "bad_view",
                "view": {"type": "wavelet"},
            })


# =====================================================================
# 6. Protocol satisfaction
# =====================================================================

class TestProtocolSatisfaction:

    def test_has_name_property(self):
        from representation.builder import build_processor_config

        cfg = {
            "type": "signal",
            "name": "proto_test",
            "resampling": {"target_sampling_rate": 12000},
            "view": {"type": "raw"},
        }
        result = build_processor_config(cfg)
        assert hasattr(result, "name")
        assert result.name == "proto_test"

    def test_create_processor_works(self):
        from representation.builder import build_processor_config
        from representation import create_processor

        cfg = {
            "type": "signal",
            "name": "proto_test",
            "resampling": {"target_sampling_rate": 12000},
            "segmentation": {"window_duration": 0.05, "window_overlap": 0.2},
            "view": {"type": "raw"},
        }
        config = build_processor_config(cfg)
        processor = create_processor(config)

        assert hasattr(processor, "name")
        assert processor.name == "proto_test"
        assert callable(processor)

    def test_create_processor_stft(self):
        from representation.builder import build_processor_config
        from representation import create_processor

        cfg = {
            "type": "signal",
            "name": "spec_test",
            "resampling": {"target_sampling_rate": 12000},
            "segmentation": {"window_duration": 0.05, "window_overlap": 0.2},
            "view": {"type": "stft", "n_fft": 256, "hop_length": 128, "win_length": 256},
        }
        config = build_processor_config(cfg)
        processor = create_processor(config)
        assert processor.name == "spec_test"


# =====================================================================
# 7. Equivalence with direct Python construction
# =====================================================================

class TestEquivalenceWithPython:

    def test_raw_equivalence(self):
        from representation.builder import build_processor_config
        from representation.signal.config import SignalProcessorConfig, RawViewConfig

        yaml_cfg = {
            "type": "signal",
            "name": "raw_12k",
            "resampling": {"target_sampling_rate": 12000, "max_bandwidth_factor": 0.5},
            "segmentation": {"window_duration": 0.05, "window_overlap": 0.5},
            "view": {"type": "raw"},
        }
        from_yaml = build_processor_config(yaml_cfg)

        from_python = SignalProcessorConfig(
            name="raw_12k",
            target_sampling_rate=12000,
            window_duration=0.05,
            window_overlap=0.5,
            view=RawViewConfig(),
            max_signal_bandwidth_factor=0.5,
        )

        assert from_yaml.name == from_python.name
        assert from_yaml.target_sampling_rate == from_python.target_sampling_rate
        assert from_yaml.window_duration == from_python.window_duration
        assert from_yaml.window_overlap == from_python.window_overlap
        assert from_yaml.max_signal_bandwidth_factor == from_python.max_signal_bandwidth_factor
        assert type(from_yaml.view) == type(from_python.view)

    def test_stft_equivalence(self):
        from representation.builder import build_processor_config
        from representation.signal.config import SignalProcessorConfig, STFTViewConfig

        yaml_cfg = {
            "type": "signal",
            "name": "spec_12k",
            "resampling": {"target_sampling_rate": 12000, "max_bandwidth_factor": 0.5},
            "segmentation": {"window_duration": 0.05, "window_overlap": 0.5},
            "view": {"type": "stft", "n_fft": 256, "hop_length": 128, "win_length": 256},
        }
        from_yaml = build_processor_config(yaml_cfg)

        from_python = SignalProcessorConfig(
            name="spec_12k",
            target_sampling_rate=12000,
            window_duration=0.05,
            window_overlap=0.5,
            view=STFTViewConfig(n_fft=256, hop_length=128, win_length=256),
            max_signal_bandwidth_factor=0.5,
        )

        assert from_yaml.name == from_python.name
        assert from_yaml.target_sampling_rate == from_python.target_sampling_rate
        assert type(from_yaml.view) == type(from_python.view)
        assert from_yaml.view.n_fft == from_python.view.n_fft
        assert from_yaml.view.hop_length == from_python.view.hop_length
        assert from_yaml.view.win_length == from_python.view.win_length


# =====================================================================
# 8. validate_processor_yaml helper
# =====================================================================

class TestValidateHelper:

    def test_valid_raw_yaml(self):
        from representation.builder import validate_processor_yaml

        cfg = {
            "type": "signal",
            "name": "raw_12k",
            "resampling": {"target_sampling_rate": 12000},
            "view": {"type": "raw"},
        }
        path = _write_yaml(cfg)
        try:
            result = validate_processor_yaml(path)
            assert result["type"] == "signal"
            assert result["view"]["type"] == "raw"
        finally:
            path.unlink(missing_ok=True)

    def test_validate_missing_type(self):
        from representation.builder import validate_processor_yaml

        cfg = {"name": "no_type", "view": {"type": "raw"}}
        path = _write_yaml(cfg)
        try:
            with pytest.raises(ValueError, match="missing 'type'"):
                validate_processor_yaml(path)
        finally:
            path.unlink(missing_ok=True)

    def test_validate_unknown_view(self):
        from representation.builder import validate_processor_yaml

        cfg = {"type": "signal", "view": {"type": "wavelet"}}
        path = _write_yaml(cfg)
        try:
            with pytest.raises(ValueError, match="unknown view type"):
                validate_processor_yaml(path)
        finally:
            path.unlink(missing_ok=True)


# =====================================================================
# 9. Fresh instances (frozen dataclass)
# =====================================================================

class TestFreshInstances:

    def test_two_configs_are_independent(self):
        from representation.builder import build_processor_config

        cfg1 = {
            "type": "signal",
            "name": "raw_12k",
            "resampling": {"target_sampling_rate": 12000},
            "view": {"type": "raw"},
        }
        cfg2 = {
            "type": "signal",
            "name": "raw_48k",
            "resampling": {"target_sampling_rate": 48000},
            "view": {"type": "raw"},
        }
        r1 = build_processor_config(cfg1)
        r2 = build_processor_config(cfg2)

        assert r1.name != r2.name
        assert r1.target_sampling_rate != r2.target_sampling_rate
        assert r1 is not r2

    def test_processors_are_independent(self):
        from representation.builder import build_processor_config
        from representation import create_processor
        cfg = {
            "type": "signal",
            "name": "test",
            "resampling": {"target_sampling_rate": 12000},
            "segmentation": {"window_duration": 0.05, "window_overlap": 0.2},
            "view": {"type": "raw"},
        }
        config = build_processor_config(cfg)
        p1 = create_processor(config)
        p2 = create_processor(config)
        assert p1 is not p2
        assert p1.name == p2.name