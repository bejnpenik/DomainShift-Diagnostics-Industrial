"""
Tests for study.transfer_builder, study.transfer_study, and main.py's
transfer-study dispatch.

Builder tests use check_files=False against the real study/collection/task
YAMLs (Pin 3: no data files needed to pass on a fresh clone). TransferStudy
and run_dry_run tests use mocks, following tests/test_transfer.py's style,
to stay independent of real data too.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from collection import Task, Metadata, SampleGroup, DatasetPlan
from collection.channels import SignalChannelConfig
from experiment.config import ExperimentConfig
from experiment.experiment import Experiment
from experiment.transfer import TransferSpec
from experiment.sampling import FileSamplingProtocol
from model.config import ModelConfig
from study.builder import is_transfer_study
from study.transfer_builder import build_transfer_study_design_from_yaml, TransferStudyDesign
from study.transfer_study import TransferStudy, run_dry_run
from study.storage import StorageConfig
from study.pipeline import PipelineConfig
from training.config import TrainerConfig, TrainResult
from training import Trainer

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRANSFER_YAML = _REPO_ROOT / "configs/study/cwru_pu_transfer_study.yaml"
_SINGLE_YAML = _REPO_ROOT / "configs/study/cwru_study.yaml"


# =====================================================================
# is_transfer_study
# =====================================================================

class TestIsTransferStudy:
    def test_transfer_yaml_returns_true(self):
        assert is_transfer_study(_TRANSFER_YAML) is True

    def test_single_collection_yaml_returns_false(self):
        assert is_transfer_study(_SINGLE_YAML) is False


# =====================================================================
# build_transfer_study_design_from_yaml -- real YAML, check_files=False
# =====================================================================

class TestBuildTransferStudyDesignFromYaml:
    def test_real_yaml_resolves_expected_spec_count(self):
        design, collections = build_transfer_study_design_from_yaml(_TRANSFER_YAML, check_files=False)
        assert design.num_configs == 4  # 1 model_type x 4 model_variant x 1 normalization
        assert design.class_aliases == ("IR", "NR", "OR")
        assert design.target == "fault_element"
        assert set(collections) == {"cwru", "paderborn"}
        assert design.seeds == (11, 32, 52)

    def test_processor_config_identical_across_all_grid_points(self):
        design, _ = build_transfer_study_design_from_yaml(_TRANSFER_YAML, check_files=False)
        names = {cfg.processor_config.name for cfg in design.experiment_configs}
        assert names == {"raw_12k"}

    def test_lr_optimizer_match_cwru_study_adamw_branch(self):
        """Consistency check requested before finalizing the YAML: transfer
        results must stay comparable with the single-collection studies."""
        design, _ = build_transfer_study_design_from_yaml(_TRANSFER_YAML, check_files=False)
        for cfg in design.experiment_configs:
            assert cfg.trainer_config.optimizer_name == "adamw"
            assert cfg.trainer_config.lr == 0.001


class TestProcessorSharedInvariantRejection:
    def test_processor_as_grid_factor_raises(self, tmp_path):
        raw = yaml.safe_load(open(_TRANSFER_YAML))
        raw["grid"]["factors"]["processor"] = ["configs/processors/raw_12k.yaml"]
        bad_path = tmp_path / "bad_factor.yaml"
        with open(bad_path, "w") as f:
            yaml.dump(raw, f)

        with pytest.raises(ValueError, match="varying grid factor"):
            build_transfer_study_design_from_yaml(bad_path, check_files=False)

    def test_processor_as_grid_dependent_raises(self, tmp_path):
        raw = yaml.safe_load(open(_TRANSFER_YAML))
        raw["grid"]["dependent"]["processor"] = {
            "depends_on": "model_type",
            "mapping": {"1d": "configs/processors/raw_12k.yaml"},
        }
        bad_path = tmp_path / "bad_dependent.yaml"
        with open(bad_path, "w") as f:
            yaml.dump(raw, f)

        with pytest.raises(ValueError, match="grid.dependent mapping"):
            build_transfer_study_design_from_yaml(bad_path, check_files=False)

    def test_real_yaml_accepted(self):
        """The real study YAML must NOT trip either rejection above."""
        design, _ = build_transfer_study_design_from_yaml(_TRANSFER_YAML, check_files=False)
        assert design.num_configs > 0


# =====================================================================
# Mock helpers for TransferStudy / run_dry_run (no real data/YAML needed)
# =====================================================================

def _make_processor_config():
    from representation.signal.config import SignalProcessorConfig, RawViewConfig
    return SignalProcessorConfig(name="raw_12k", view=RawViewConfig())


def _make_trainer_config(**overrides):
    defaults = dict(max_epochs=2, device="cpu", early_stopping=None, noise=None, verbose_level=0)
    defaults.update(overrides)
    return TrainerConfig(**defaults)


def _simple_model_factory(num_classes, **kwargs):
    return nn.Sequential(nn.Flatten(), nn.Linear(600, num_classes))


def _make_experiment_config(name="cfg", **overrides):
    defaults = dict(
        name=name,
        processor_config=_make_processor_config(),
        model_config=ModelConfig(name="simple", model_class=_simple_model_factory, params={}),
        trainer_config=_make_trainer_config(),
        normalization="none",
        train_val_split_ratio=0.33,
        random_seed=42,
        pipeline=PipelineConfig(primary="vibration"),
        file_sampling=FileSamplingProtocol(max_files_per_code=1),
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


_FAULT_ELEMENT_HEADER = {
    "fault_element": {
        0: {"name": "normal", "alias": "NR"},
        1: {"name": "inner ring", "alias": "IR"},
        2: {"name": "outer ring", "alias": "OR"},
    }
}


def _make_plan(dataset_name, label, classes):
    groups = {
        cls: SampleGroup(codes={i: [f"{label}_{cls}_{i}.mat"]}, metadata={i: Metadata({})})
        for i, cls in enumerate(classes)
    }
    return DatasetPlan(dataset_name=dataset_name, label=label, sample_groups=groups)


def _make_mock_collection(name, plans_by_filters):
    collection = MagicMock()
    collection.name = name
    collection.channels = {"vibration": SignalChannelConfig(reader_channel="vibration", sampling_rate=12000)}
    collection.header = _FAULT_ELEMENT_HEADER

    def get_filter_value_from_description(field, description):
        for code, desc in collection.header[field].items():
            if isinstance(desc, dict):
                if desc.get("alias") == description or desc.get("name") == description:
                    return code
            elif desc == description:
                return code
        raise ValueError(f"Filter '{field}' value '{description}' not found in header.")

    collection.get_filter_value_from_description = MagicMock(side_effect=get_filter_value_from_description)

    def construct_dataset_plan(task, **filters):
        key = frozenset(filters.items())
        if key not in plans_by_filters:
            raise AssertionError(f"No mock plan registered for filters {filters}")
        return plans_by_filters[key]

    collection.construct_dataset_plan = MagicMock(side_effect=construct_dataset_plan)
    return collection


def _fake_reader(path, metadata, channels):
    return {"vibration": np.random.randn(2000).astype(np.float32)}


def _make_train_result(model=None):
    if model is None:
        model = nn.Linear(2, 2)
    return TrainResult(model=model, epochs_run=1, train_loss=0.1, train_acc=90.0, val_loss=0.2, val_acc=85.0)


def _make_mock_design_and_collections(num_configs=1, seeds=(11,)):
    task = Task(target="fault_element", domain_factors=("fault_size",))
    cwru = _make_mock_collection(
        "cwru", {frozenset({"fault_size": 1}.items()): _make_plan("cwru", "fault_element-fault_size=1", ["normal", "inner ring"])},
    )
    paderborn = _make_mock_collection(
        "paderborn", {frozenset({"fault_size": 1}.items()): _make_plan("paderborn", "fault_element-fault_size=1", ["normal", "inner ring"])},
    )
    collections = {
        "cwru": (cwru, MagicMock(side_effect=_fake_reader)),
        "paderborn": (paderborn, MagicMock(side_effect=_fake_reader)),
    }
    source_specs = (
        TransferSpec("cwru", task, {"fault_size": 1}),
        TransferSpec("paderborn", task, {"fault_size": 1}),
    )
    experiment_configs = tuple(_make_experiment_config(name=f"cfg{i}") for i in range(num_configs))
    design = TransferStudyDesign(
        name="mock_transfer_study",
        class_aliases=("NR", "IR"),
        target="fault_element",
        source_specs=source_specs,
        target_specs=source_specs,  # self-eval for both collections
        experiment_configs=experiment_configs,
        seeds=seeds,
    )
    return design, collections


_NO_ARTIFACTS = StorageConfig(save_model_weights=False, save_config_snapshot=False, save_study_design=False)


# =====================================================================
# Pin 1 -- TransferStudy.run validates before any training
# =====================================================================

class TestTransferStudyValidatesBeforeTraining:
    def test_validator_called_once_before_any_training(self, tmp_path):
        design, collections = _make_mock_design_and_collections(num_configs=1, seeds=(11,))
        study = TransferStudy(collections, results_dir=tmp_path, storage_config=_NO_ARTIFACTS)

        from experiment.transfer import validate_transfer_study_setup as real_validate
        call_order = []

        def spy_validate(*args, **kwargs):
            call_order.append("validate")
            return real_validate(*args, **kwargs)

        def spy_fit(self, model, train_data, val_data):
            call_order.append("fit")
            # Return the REAL model being trained (shape-correct for the
            # subsequent real evaluate_on_plan calls below) rather than a
            # throwaway placeholder -- a mismatched fake model here would
            # crash evaluation with a shape error, not exercise the thing
            # this test actually checks (call order/count).
            return _make_train_result(model=model)

        with patch("study.transfer_study.validate_transfer_study_setup", side_effect=spy_validate) as mock_validate, \
             patch.object(Trainer, "fit", spy_fit):
            study.run(design, verbose=False)

        assert mock_validate.call_count == 1
        assert call_order[0] == "validate"
        assert "fit" in call_order

    def test_validator_raising_aborts_with_zero_training_calls(self, tmp_path):
        design, collections = _make_mock_design_and_collections(num_configs=2, seeds=(11, 32))
        study = TransferStudy(collections, results_dir=tmp_path, storage_config=_NO_ARTIFACTS)

        with patch("study.transfer_study.validate_transfer_study_setup", side_effect=ValueError("boom")), \
             patch.object(Trainer, "fit") as mock_fit:
            with pytest.raises(ValueError, match="boom"):
                study.run(design, verbose=False)
            mock_fit.assert_not_called()


# =====================================================================
# Pin 4 -- dry-run determinism and provenance
# =====================================================================

class TestRunDryRun:
    def test_file_sampling_mismatch_across_configs_raises(self):
        design, collections = _make_mock_design_and_collections(num_configs=2, seeds=(11,))
        mismatched = dataclasses.replace(
            design.experiment_configs[1], file_sampling=FileSamplingProtocol(max_files_per_code=99)
        )
        design = dataclasses.replace(design, experiment_configs=(design.experiment_configs[0], mismatched))

        with pytest.raises(ValueError, match="file_sampling"):
            run_dry_run(design, collections)

    def test_prints_which_seed_counts_reflect_and_completes(self, capsys):
        design, collections = _make_mock_design_and_collections(num_configs=1, seeds=(11, 32))
        exit_code = run_dry_run(design, collections)
        out = capsys.readouterr().out

        assert exit_code == 0
        assert "seed=11" in out
        assert "Dry run complete" in out
        assert "segments per class" in out
        assert "imbalance ratio" in out


# =====================================================================
# main.py -- --dry-run compatibility predicate (Pin 5) + type dispatch
# =====================================================================

class TestDryRunRequiresTransferPredicate:
    def test_predicate(self):
        from main import _dry_run_requires_transfer
        assert _dry_run_requires_transfer(False) == "--dry-run currently supports transfer studies only"
        assert _dry_run_requires_transfer(True) is None


class TestMainTypeDispatch:
    """Proves old (type:-absent) studies are provably untouched: the
    existing single-collection builder is what actually runs, and the
    transfer builder is never invoked, and vice versa."""

    def test_type_absent_uses_single_collection_builder_only(self, monkeypatch):
        import main as main_module

        class _Marker(Exception):
            pass

        monkeypatch.setattr(sys, "argv", [
            "main.py", "--collection", "configs/collections/cwru.yaml", "--study", str(_SINGLE_YAML),
        ])

        with patch("study.builder.build_study_design_from_yaml", side_effect=_Marker("single called")) as mock_single, \
             patch("study.transfer_builder.build_transfer_study_design_from_yaml") as mock_transfer, \
             patch("main.DatasetCollection") as mock_collection_cls:
            mock_collection_cls.return_value.reader = MagicMock()
            mock_collection_cls.return_value.name = "cwru"
            with pytest.raises(_Marker):
                main_module.main()

        mock_single.assert_called_once()
        mock_transfer.assert_not_called()

    def test_transfer_type_uses_transfer_builder_only(self, monkeypatch):
        import main as main_module

        class _Marker(Exception):
            pass

        monkeypatch.setattr(sys, "argv", ["main.py", "--study", str(_TRANSFER_YAML)])

        with patch("study.transfer_builder.build_transfer_study_design_from_yaml", side_effect=_Marker("transfer called")) as mock_transfer, \
             patch("study.builder.build_study_design_from_yaml") as mock_single:
            with pytest.raises(_Marker):
                main_module.main()

        mock_transfer.assert_called_once()
        mock_single.assert_not_called()

    def test_dry_run_with_non_transfer_study_errors_clearly(self, monkeypatch, capsys):
        import main as main_module

        monkeypatch.setattr(sys, "argv", [
            "main.py", "--collection", "configs/collections/cwru.yaml",
            "--study", str(_SINGLE_YAML), "--dry-run",
        ])
        with pytest.raises(SystemExit):
            main_module.main()
        err = capsys.readouterr().err
        assert "--dry-run currently supports transfer studies only" in err
