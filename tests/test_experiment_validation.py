"""
Tests for experiment.validation (fail-fast cross-collection transfer checks)
and Experiment.load_plan_arrays.

Uses mocks for collection/reader, following tests/test_experiment.py's style.
"""

from __future__ import annotations

import pytest
import torch
from unittest.mock import MagicMock, patch

from experiment.config import ExperimentConfig
from experiment.experiment import Experiment
from experiment.validation import (
    resolve_class_aliases_to_names,
    validate_transfer_setup,
)
from training.config import TrainerConfig
from model.config import ModelConfig
from collection import Metadata, SampleGroup, DatasetPlan
from collection.channels import SignalChannelConfig
from study.pipeline import PipelineConfig


# =====================================================================
# Helpers / Fixtures (mirrors tests/test_experiment.py)
# =====================================================================

def _make_processor_config():
    from representation.signal.config import SignalProcessorConfig, RawViewConfig
    return SignalProcessorConfig(name="raw_12k", view=RawViewConfig())


def _make_trainer_config(**overrides):
    defaults = dict(max_epochs=5, device="cpu", early_stopping=None, noise=None, verbose_level=0)
    defaults.update(overrides)
    return TrainerConfig(**defaults)


def _simple_model_factory(num_classes, **kwargs):
    import torch.nn as nn
    return nn.Sequential(nn.Flatten(), nn.Linear(600, num_classes))


def _make_model_config(**overrides):
    defaults = dict(name="simple", model_class=_simple_model_factory, params={})
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_experiment_config(**overrides):
    defaults = dict(
        name="test_exp",
        processor_config=_make_processor_config(),
        model_config=_make_model_config(),
        trainer_config=_make_trainer_config(),
        normalization="none",
        train_val_split_ratio=0.33,
        random_seed=42,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _make_mock_collection(name, header, channels=None):
    """MagicMock collection whose get_filter_value_from_description mirrors
    the real DatasetCollection implementation exactly (alias/name lookup)."""
    def get_filter_value_from_description(field, description):
        for code, desc in header[field].items():
            if isinstance(desc, dict):
                if desc.get("alias") == description or desc.get("name") == description:
                    return code
            elif desc == description:
                return code
        raise ValueError(f"Filter '{field}' value '{description}' not found in header.")

    collection = MagicMock()
    collection.name = name
    collection.header = header
    collection.channels = channels or {}
    collection.get_filter_value_from_description = MagicMock(side_effect=get_filter_value_from_description)
    return collection


def _cwru_like_header(extra_ball_class=True):
    fault_element = {
        0: {"name": "normal", "alias": "NR"},
        1: {"name": "inner ring", "alias": "IR"},
        2: {"name": "outer ring", "alias": "OR"},
    }
    if extra_ball_class:
        fault_element[3] = {"name": "ball", "alias": "BA"}
    return {"fault_element": fault_element}


def _paderborn_like_header():
    return {
        "fault_element": {
            0: {"name": "normal", "alias": "NR"},
            1: {"name": "inner ring", "alias": "IR"},
            2: {"name": "outer ring", "alias": "OR"},
        }
    }


def _make_probe_plan(name, class_names):
    groups = {
        cls: SampleGroup(codes={i: [f"{name}_{i}.mat"]}, metadata={i: Metadata({})})
        for i, cls in enumerate(class_names)
    }
    return DatasetPlan(dataset_name=name, label="probe", sample_groups=groups)


def _fake_load_returns(shape=(1, 600)):
    def fake(self, plan):
        return torch.randn(2, *shape), torch.zeros(2, dtype=torch.long), {}, None
    return fake


# =====================================================================
# Experiment.load_plan_arrays
# =====================================================================

class TestLoadPlanArrays:
    def test_forwards_to_domain_dataset_with_none_normalisator(self):
        config = _make_experiment_config(pipeline=PipelineConfig(primary="vibration"), random_seed=7)
        collection = MagicMock()
        collection.channels = {"vibration": SignalChannelConfig(reader_channel="vibration", sampling_rate=12000)}
        experiment = Experiment(collection, MagicMock(), config)

        sentinel = (torch.zeros(1), torch.zeros(1), {"a": 0}, None)
        experiment._domain_dataset = MagicMock(return_value=sentinel)

        plan = MagicMock()
        result = experiment.load_plan_arrays(plan)

        experiment._domain_dataset.assert_called_once_with(plan, None, 7)
        assert result is sentinel


# =====================================================================
# sorted() class-index-mapping pin (guarantees cross-collection alignment)
# =====================================================================

class TestSortedClassIndexPin:
    def test_cls_labels_independent_of_sample_groups_insertion_order(self):
        config = _make_experiment_config(pipeline=PipelineConfig(primary="vibration"))
        collection = MagicMock()
        collection.channels = {"vibration": SignalChannelConfig(reader_channel="vibration", sampling_rate=12000)}

        def fake_reader(path, metadata, channels):
            import numpy as np
            return {"vibration": np.random.randn(2000).astype(np.float32)}

        experiment = Experiment(collection, MagicMock(side_effect=fake_reader), config)

        def make_plan(order):
            groups = {
                "normal": SampleGroup(codes={1: ["a.mat"]}, metadata={1: Metadata({})}),
                "inner ring": SampleGroup(codes={2: ["b.mat"]}, metadata={2: Metadata({})}),
                "outer ring": SampleGroup(codes={3: ["c.mat"]}, metadata={3: Metadata({})}),
            }
            return DatasetPlan(dataset_name="test", label="probe", sample_groups={k: groups[k] for k in order})

        plan_a = make_plan(["normal", "inner ring", "outer ring"])
        plan_b = make_plan(["outer ring", "normal", "inner ring"])

        _, _, cls_labels_a, _ = experiment.load_plan_arrays(plan_a)
        _, _, cls_labels_b, _ = experiment.load_plan_arrays(plan_b)

        expected = {"inner ring": 0, "normal": 1, "outer ring": 2}
        assert cls_labels_a == expected
        assert cls_labels_b == expected


# =====================================================================
# resolve_class_aliases_to_names
# =====================================================================

class TestResolveClassAliasesToNames:
    def test_resolves_only_requested_aliases(self):
        collection = _make_mock_collection("cwru", _cwru_like_header(extra_ball_class=True))
        names = resolve_class_aliases_to_names(collection, "fault_element", ("IR", "NR", "OR"))
        assert names == {"IR": "inner ring", "NR": "normal", "OR": "outer ring"}
        # BA is present in the header but was never asked for -- not resolved.
        assert "BA" not in names

    def test_unknown_alias_raises(self):
        collection = _make_mock_collection("cwru", _cwru_like_header())
        with pytest.raises(ValueError, match="not found"):
            resolve_class_aliases_to_names(collection, "fault_element", ("XX",))


# =====================================================================
# validate_transfer_setup
# =====================================================================

class TestValidateTransferSetup:
    def _collections(self, cwru_extra_ball=True, cwru_sr=12000, pad_sr=64000):
        cwru = _make_mock_collection(
            "cwru", _cwru_like_header(extra_ball_class=cwru_extra_ball),
            channels={"vibration": SignalChannelConfig(reader_channel="vibration", sampling_rate=cwru_sr)},
        )
        paderborn = _make_mock_collection(
            "paderborn", _paderborn_like_header(),
            channels={"vibration": SignalChannelConfig(reader_channel="vibration", sampling_rate=pad_sr)},
        )
        return {"cwru": cwru, "paderborn": paderborn}

    def _readers(self):
        return {"cwru": MagicMock(), "paderborn": MagicMock()}

    def test_extra_class_outside_class_aliases_validates_cleanly(self):
        """CWRU-like mock with an extra 'ball' class + class_aliases=[IR,NR,OR]
        must validate with no errors -- collections legitimately have
        different, larger header sets than the shared class_aliases."""
        collections = self._collections(cwru_extra_ball=True)
        probe_plans = {
            "cwru": _make_probe_plan("cwru", ["normal", "inner ring", "outer ring", "ball"]),
            "paderborn": _make_probe_plan("paderborn", ["normal", "inner ring", "outer ring"]),
        }
        config = _make_experiment_config(pipeline=PipelineConfig(primary="vibration"))

        with patch.object(Experiment, "load_plan_arrays", _fake_load_returns()):
            report = validate_transfer_setup(
                collections, ("IR", "NR", "OR"), "fault_element", config, self._readers(), probe_plans,
            )
        assert report.probe_shapes == {"cwru": (1, 600), "paderborn": (1, 600)}

    def test_alias_resolving_to_different_names_raises_naming_alias_and_both_names(self):
        collections = self._collections()
        collections["paderborn"].header["fault_element"][1]["name"] = "inner-ring-different"
        probe_plans = {
            "cwru": _make_probe_plan("cwru", ["normal", "inner ring", "outer ring", "ball"]),
            "paderborn": _make_probe_plan("paderborn", ["normal", "inner-ring-different", "outer ring"]),
        }
        config = _make_experiment_config(pipeline=PipelineConfig(primary="vibration"))

        with pytest.raises(ValueError) as exc:
            validate_transfer_setup(
                collections, ("IR", "NR", "OR"), "fault_element", config, self._readers(), probe_plans,
            )
        message = str(exc.value)
        assert "IR" in message
        assert "inner ring" in message
        assert "inner-ring-different" in message

    def test_missing_alias_in_probe_plan_names_collection_and_aliases(self):
        collections = self._collections()
        probe_plans = {
            "cwru": _make_probe_plan("cwru", ["normal", "inner ring", "outer ring", "ball"]),
            "paderborn": _make_probe_plan("paderborn", ["normal", "outer ring"]),  # missing inner ring
        }
        config = _make_experiment_config(pipeline=PipelineConfig(primary="vibration"))

        with pytest.raises(ValueError) as exc:
            validate_transfer_setup(
                collections, ("IR", "NR", "OR"), "fault_element", config, self._readers(), probe_plans,
            )
        message = str(exc.value)
        assert "paderborn" in message
        assert "IR" in message

    def test_missing_primary_channel_raises(self):
        collections = self._collections()
        collections["paderborn"].channels = {}
        probe_plans = {
            "cwru": _make_probe_plan("cwru", ["normal", "inner ring", "outer ring", "ball"]),
            "paderborn": _make_probe_plan("paderborn", ["normal", "inner ring", "outer ring"]),
        }
        config = _make_experiment_config(pipeline=PipelineConfig(primary="vibration"))

        with pytest.raises(ValueError, match="not in collection"):
            validate_transfer_setup(
                collections, ("IR", "NR", "OR"), "fault_element", config, self._readers(), probe_plans,
            )

    def test_native_rate_below_target_warns_not_raises(self):
        collections = self._collections(cwru_sr=6000)  # default processor target is 12000
        probe_plans = {
            "cwru": _make_probe_plan("cwru", ["normal", "inner ring", "outer ring", "ball"]),
            "paderborn": _make_probe_plan("paderborn", ["normal", "inner ring", "outer ring"]),
        }
        config = _make_experiment_config(pipeline=PipelineConfig(primary="vibration"))

        with patch.object(Experiment, "load_plan_arrays", _fake_load_returns()):
            report = validate_transfer_setup(
                collections, ("IR", "NR", "OR"), "fault_element", config, self._readers(), probe_plans,
            )
        assert any("cwru" in w and "6000" in w for w in report.warnings)
        assert not any("paderborn" in w for w in report.warnings)

    def test_mismatched_probe_shapes_raise(self):
        collections = self._collections()
        probe_plans = {
            "cwru": _make_probe_plan("cwru", ["normal", "inner ring", "outer ring", "ball"]),
            "paderborn": _make_probe_plan("paderborn", ["normal", "inner ring", "outer ring"]),
        }
        config = _make_experiment_config(pipeline=PipelineConfig(primary="vibration"))

        def fake_load(self, plan):
            shape = (1, 600) if self._collection.name == "cwru" else (1, 500)
            return torch.randn(2, *shape), torch.zeros(2, dtype=torch.long), {}, None

        with patch.object(Experiment, "load_plan_arrays", fake_load):
            with pytest.raises(ValueError, match="inconsistent shapes"):
                validate_transfer_setup(
                    collections, ("IR", "NR", "OR"), "fault_element", config, self._readers(), probe_plans,
                )

    def test_hard_failures_aggregated_into_one_message(self):
        """Two independent hard failures (missing channel + missing alias in
        plan) must both appear in a single raised message, not just the first."""
        collections = self._collections()
        collections["paderborn"].channels = {}
        probe_plans = {
            "cwru": _make_probe_plan("cwru", ["normal", "inner ring", "outer ring", "ball"]),
            "paderborn": _make_probe_plan("paderborn", ["normal", "outer ring"]),  # missing inner ring too
        }
        config = _make_experiment_config(pipeline=PipelineConfig(primary="vibration"))

        with pytest.raises(ValueError) as exc:
            validate_transfer_setup(
                collections, ("IR", "NR", "OR"), "fault_element", config, self._readers(), probe_plans,
            )
        message = str(exc.value)
        assert "not in collection" in message
        assert "class_aliases missing from probe plan" in message
