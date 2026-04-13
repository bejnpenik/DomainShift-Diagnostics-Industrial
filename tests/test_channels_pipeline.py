"""
Tests for collection channels parsing, PipelineConfig, and DomainDataset internals.

No data files needed — uses synthetic configs and direct YAML parsing where needed.

Note: DomainDataset integration tests are not included here because the
experiment package uses cross-package relative imports that require the
project to be installed (not just src/ on sys.path). Those are covered
at runtime by the study pipeline.
"""

import pytest
import yaml
import torch

from collection.channels import (
    SignalChannelConfig,
    MetadataChannelConfig,
    parse_channel_config,
    parse_all_channels,
)
from study.pipeline import PipelineConfig, ConditioningSource
from collection.metadata import Metadata


# =====================================================================
# Channel config parsing
# =====================================================================

class TestParseChannelConfig:
    def test_signal_channel_static_rate(self):
        raw = {"reader_channel": "vibration", "sampling_rate": 64000, "unit": "m/s²"}
        cfg = parse_channel_config("vibration", raw)
        assert isinstance(cfg, SignalChannelConfig)
        assert cfg.sampling_rate == 64000
        assert cfg.reader_channel == "vibration"

    def test_signal_channel_dynamic_rate_needs_key(self):
        raw = {"reader_channel": "vibration", "sampling_rate": "dynamic"}
        with pytest.raises(ValueError, match="sampling_rate_key"):
            parse_channel_config("vibration", raw)

    def test_signal_channel_dynamic_rate_with_key(self):
        raw = {
            "reader_channel": "vibration",
            "sampling_rate": "dynamic",
            "sampling_rate_key": "sampling_rate",
        }
        cfg = parse_channel_config("vibration", raw)
        assert isinstance(cfg, SignalChannelConfig)
        assert cfg.sampling_rate == "dynamic"
        assert cfg.sampling_rate_key == "sampling_rate"

    def test_metadata_channel(self):
        raw = {"source": "metadata", "metadata_path": "condition.speed", "unit": "rpm"}
        cfg = parse_channel_config("rpm", raw)
        assert isinstance(cfg, MetadataChannelConfig)
        assert cfg.metadata_path == "condition.speed"

    def test_parse_all_channels_mixed(self):
        raw = {
            "vibration": {"reader_channel": "vibration", "sampling_rate": 64000},
            "rpm": {"source": "metadata", "metadata_path": "condition.speed"},
        }
        channels = parse_all_channels(raw)
        assert isinstance(channels["vibration"], SignalChannelConfig)
        assert isinstance(channels["rpm"], MetadataChannelConfig)


# =====================================================================
# PipelineConfig
# =====================================================================

class TestPipelineConfig:
    def test_from_dict_minimal(self):
        p = PipelineConfig.from_dict({"primary": "vibration"})
        assert p.primary == "vibration"
        assert p.conditioning == []

    def test_from_dict_with_conditioning_dict(self):
        p = PipelineConfig.from_dict({
            "primary": "vibration",
            "conditioning": [{"channel": "rpm", "reduce": "mean"}],
        })
        assert p.primary == "vibration"
        assert len(p.conditioning) == 1
        assert p.conditioning[0].channel == "rpm"
        assert p.conditioning[0].reduce == "mean"

    def test_from_dict_with_conditioning_string(self):
        p = PipelineConfig.from_dict({
            "primary": "vibration",
            "conditioning": ["rpm"],
        })
        assert p.conditioning[0].channel == "rpm"
        assert p.conditioning[0].reduce == "mean"  # default

    def test_from_dict_missing_primary_raises(self):
        with pytest.raises(ValueError, match="primary"):
            PipelineConfig.from_dict({})

    def test_from_dict_empty_raises(self):
        with pytest.raises(ValueError):
            PipelineConfig.from_dict({"conditioning": []})

    def test_conditioning_names(self):
        p = PipelineConfig.from_dict({
            "primary": "vibration",
            "conditioning": [{"channel": "rpm"}, {"channel": "torque"}],
        })
        assert p.conditioning_names == ["rpm", "torque"]


# =====================================================================
# Collection YAML channels sections
# =====================================================================

class TestCollectionChannelsYAML:
    """Parse channels directly from YAML without loading DatasetCollection
    (which triggers reader imports incompatible with test sys.path setup)."""

    def _load_channels(self, yaml_path: str) -> dict:
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        return parse_all_channels(raw.get("channels", {}))

    def test_cwru_vibration_channel(self):
        channels = self._load_channels("configs/collections/cwru.yaml")
        assert "vibration" in channels
        cfg = channels["vibration"]
        assert isinstance(cfg, SignalChannelConfig)
        assert cfg.sampling_rate == "dynamic"
        assert cfg.sampling_rate_key == "sampling_rate"
        assert cfg.reader_channel == "vibration"

    def test_cwru_rpm_channel(self):
        channels = self._load_channels("configs/collections/cwru.yaml")
        assert "rpm" in channels
        cfg = channels["rpm"]
        assert isinstance(cfg, MetadataChannelConfig)
        assert cfg.metadata_path == "condition.speed"

    def test_paderborn_vibration_channel(self):
        channels = self._load_channels("configs/collections/paderborn.yaml")
        assert "vibration" in channels
        cfg = channels["vibration"]
        assert isinstance(cfg, SignalChannelConfig)
        assert cfg.sampling_rate == 64000

    def test_paderborn_all_channels_present(self):
        channels = self._load_channels("configs/collections/paderborn.yaml")
        for expected in ("vibration", "rpm", "torque", "phase_current"):
            assert expected in channels, f"Missing channel: {expected}"

    def test_paderborn_rpm_sampling_rate(self):
        channels = self._load_channels("configs/collections/paderborn.yaml")
        assert channels["rpm"].sampling_rate == 4000

    def test_paderborn_torque_sampling_rate(self):
        channels = self._load_channels("configs/collections/paderborn.yaml")
        assert channels["torque"].sampling_rate == 4000


# =====================================================================
# Study YAML pipeline sections
# =====================================================================

class TestStudyYAMLPipeline:
    """Verify all study YAMLs have a pipeline.primary entry."""

    STUDY_YAMLS = [
        "configs/study/cwru_study.yaml",
        "configs/study/cwru_study_testing.yaml",
        "configs/study/paderborn_study.yaml",
        "configs/study/paderborn_study_testing.yaml",
    ]

    @pytest.mark.parametrize("path", STUDY_YAMLS)
    def test_pipeline_primary_present(self, path):
        with open(path) as f:
            raw = yaml.safe_load(f)
        pipeline_raw = raw.get("grid", {}).get("independent", {}).get("pipeline")
        assert pipeline_raw is not None, f"No pipeline in grid.independent of {path}"
        p = PipelineConfig.from_dict(pipeline_raw)
        assert p.primary == "vibration"


# =====================================================================
# _resolve_sampling_rate logic (tested inline, no package imports)
# =====================================================================

class TestResolveSamplingRateLogic:
    """Unit-test the sampling rate resolution logic directly."""

    def _resolve(self, ch_cfg, metadata):
        """Inline copy of DomainDataset._resolve_sampling_rate."""
        sr = ch_cfg.sampling_rate
        if isinstance(sr, int):
            return sr
        if sr == "dynamic":
            entry = metadata[ch_cfg.sampling_rate_key]
            return int(entry["value"] if isinstance(entry, dict) else entry)
        raise ValueError(f"Unknown sampling_rate spec: {sr}")

    def test_static_rate(self):
        cfg = SignalChannelConfig(reader_channel="vibration", sampling_rate=64000)
        meta = Metadata({})
        assert self._resolve(cfg, meta) == 64000

    def test_dynamic_rate_dict_entry(self):
        cfg = SignalChannelConfig(
            reader_channel="vibration",
            sampling_rate="dynamic",
            sampling_rate_key="sampling_rate",
        )
        meta = Metadata({"sampling_rate": {"value": 12000, "name": "12k"}})
        assert self._resolve(cfg, meta) == 12000

    def test_dynamic_rate_plain_int(self):
        cfg = SignalChannelConfig(
            reader_channel="vibration",
            sampling_rate="dynamic",
            sampling_rate_key="sampling_rate",
        )
        meta = Metadata({"sampling_rate": 48000})
        assert self._resolve(cfg, meta) == 48000

    def test_unknown_spec_raises(self):
        cfg = SignalChannelConfig(reader_channel="vibration", sampling_rate="unknown")
        meta = Metadata({})
        with pytest.raises(ValueError, match="Unknown sampling_rate spec"):
            self._resolve(cfg, meta)


# =====================================================================
# _resolve_metadata_value logic (tested inline)
# =====================================================================

class TestResolveMetadataValueLogic:
    """Unit-test the metadata path resolution logic directly."""

    def _resolve(self, path, metadata):
        """Inline copy of DomainDataset._resolve_metadata_value."""
        val = metadata
        for part in path.split("."):
            val = val[part]
        if isinstance(val, dict):
            if "value" not in val:
                raise ValueError(
                    f"Metadata path '{path}' resolved to a dict without a 'value' key: {val}"
                )
            val = val["value"]
        return float(val)

    def test_nested_int_leaf(self):
        meta = Metadata({"condition": {"speed": 1797, "load": 0}})
        assert self._resolve("condition.speed", meta) == 1797.0

    def test_top_level_int(self):
        meta = Metadata({"speed": 1797})
        assert self._resolve("speed", meta) == 1797.0

    def test_dict_with_value_key(self):
        meta = Metadata({"speed": {"value": 1797, "unit": "rpm"}})
        assert self._resolve("speed", meta) == 1797.0

    def test_dict_without_value_key_raises(self):
        meta = Metadata({"condition": {"name": "0HP", "speed": 1797}})
        with pytest.raises(ValueError, match="'value' key"):
            self._resolve("condition", meta)

    def test_string_value_after_float_cast_raises(self):
        meta = Metadata({"label": {"value": "C1"}})
        with pytest.raises((ValueError, TypeError)):
            self._resolve("label", meta)
