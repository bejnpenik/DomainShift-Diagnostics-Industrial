"""
Tests for experiment.transfer (cross-collection transfer orchestration).

Uses mocks for collection/reader, following tests/test_experiment.py's
style. Experiment.train_on_plan / evaluate_on_plan are patched at the class
level for orchestration-focused tests (label qualification, chokepoint
routing, self-eval preflight, pooling, sanitization) -- these test
TransferExperiment's own logic, not Experiment's training/data-loading
correctness. The cross-experiment cls_labels-mismatch test is the
exception: it needs evaluate_on_plan's REAL internals, so it uses a working
reader+processor stack for that one collection instead of mocking it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from collection import Task, Metadata, SampleGroup, DatasetPlan
from collection.channels import SignalChannelConfig
from experiment.config import ExperimentConfig
from experiment.experiment import Experiment, ExperimentTrainResult
from experiment.transfer import (
    TransferExperiment,
    TransferSpec,
    sanitize_label_for_filename,
    _pooled_label,
)
from training.config import TrainerConfig, TrainResult
from model.config import ModelConfig
from normalization import Normalisator
from study.pipeline import PipelineConfig


# =====================================================================
# Helpers
# =====================================================================

def _make_processor_config():
    from representation.signal.config import SignalProcessorConfig, RawViewConfig
    return SignalProcessorConfig(name="raw_12k", view=RawViewConfig())


def _make_trainer_config(**overrides):
    defaults = dict(max_epochs=5, device="cpu", early_stopping=None, noise=None, verbose_level=0)
    defaults.update(overrides)
    return TrainerConfig(**defaults)


def _simple_model_factory(num_classes, **kwargs):
    return nn.Sequential(nn.Flatten(), nn.Linear(600, num_classes))


def _make_model_config(**overrides):
    defaults = dict(name="simple", model_class=_simple_model_factory, params={})
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_experiment_config(**overrides):
    defaults = dict(
        name="test_transfer",
        processor_config=_make_processor_config(),
        model_config=_make_model_config(),
        trainer_config=_make_trainer_config(),
        normalization="none",
        train_val_split_ratio=0.33,
        random_seed=42,
        pipeline=PipelineConfig(primary="vibration"),
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _make_plan(dataset_name, label, classes):
    groups = {
        cls: SampleGroup(codes={i: [f"{label}_{cls}_{i}.mat"]}, metadata={i: Metadata({})})
        for i, cls in enumerate(classes)
    }
    return DatasetPlan(dataset_name=dataset_name, label=label, sample_groups=groups)


def _make_mock_collection(name, plans_by_filters, reader_side_effect=None):
    """plans_by_filters: {frozenset(filters.items()): DatasetPlan}."""
    collection = MagicMock()
    collection.name = name
    collection.channels = {"vibration": SignalChannelConfig(reader_channel="vibration", sampling_rate=12000)}

    def construct_dataset_plan(task, **filters):
        key = frozenset(filters.items())
        if key not in plans_by_filters:
            raise AssertionError(f"No mock plan registered for filters {filters}")
        return plans_by_filters[key]

    collection.construct_dataset_plan = MagicMock(side_effect=construct_dataset_plan)
    return collection


def _make_task(domain_factors=("fault_size",)):
    return Task(target="fault_element", domain_factors=tuple(domain_factors))


def _make_train_result(model=None):
    if model is None:
        model = nn.Linear(2, 2)
    return TrainResult(model=model, epochs_run=1, train_loss=0.1, train_acc=90.0, val_loss=0.2, val_acc=85.0)


def _make_exp_train_result(cls_labels, model=None):
    return ExperimentTrainResult(
        train_result=_make_train_result(model),
        normalisator=Normalisator(mode="none"),
        cls_labels=cls_labels,
        dataset_label="unused",
    )


def _fake_confusion_matrix(n=2):
    return np.eye(n, dtype=int) * 5


def _fake_reader(path, metadata, channels):
    return {"vibration": np.random.randn(2000).astype(np.float32)}


# =====================================================================
# Basic run_transfer: qualified labels, self-eval present
# =====================================================================

class TestRunTransferBasic:
    def test_source_and_two_targets_qualified_labels(self):
        cwru = _make_mock_collection("cwru", {
            frozenset({"fault_size": 1}.items()): _make_plan("cwru", "fault_element-fault_size=1", ["normal", "inner ring"]),
        })
        paderborn = _make_mock_collection("paderborn", {
            frozenset({"fault_size": 1}.items()): _make_plan("paderborn", "fault_element-fault_size=1", ["normal", "inner ring"]),
        })
        config = _make_experiment_config()
        te = TransferExperiment({"cwru": (cwru, MagicMock()), "paderborn": (paderborn, MagicMock())}, config)
        task = _make_task()

        source_specs = (TransferSpec("cwru", task, {"fault_size": 1}),)
        target_specs = (
            TransferSpec("cwru", task, {"fault_size": 1}),
            TransferSpec("paderborn", task, {"fault_size": 1}),
        )

        cls_labels = {"inner ring": 0, "normal": 1}
        with patch.object(Experiment, "train_on_plan", return_value=_make_exp_train_result(cls_labels)), \
             patch.object(Experiment, "evaluate_on_plan", return_value=(_fake_confusion_matrix(), "unused")):
            mds = te.run_transfer(source_specs, target_specs)

        assert len(mds.domain_solutions) == 1
        ds = mds.domain_solutions[0]
        assert ds.train_dataset_name == "cwru:fault_element-fault_size=1"
        assert set(ds.confusion_matrices.keys()) == {
            "cwru:fault_element-fault_size=1",
            "paderborn:fault_element-fault_size=1",
        }


# =====================================================================
# Pin 1 — self-eval presence checked before any training
# =====================================================================

class TestSelfEvalPreflightCheck:
    def test_missing_self_eval_target_raises_before_training(self):
        cwru = _make_mock_collection("cwru", {
            frozenset({"fault_size": 1}.items()): _make_plan("cwru", "fault_element-fault_size=1", ["normal"]),
        })
        paderborn = _make_mock_collection("paderborn", {
            frozenset({"fault_size": 1}.items()): _make_plan("paderborn", "fault_element-fault_size=1", ["normal"]),
        })
        config = _make_experiment_config()
        te = TransferExperiment({"cwru": (cwru, MagicMock()), "paderborn": (paderborn, MagicMock())}, config)
        task = _make_task()

        source_specs = (TransferSpec("cwru", task, {"fault_size": 1}),)
        target_specs = (TransferSpec("paderborn", task, {"fault_size": 1}),)  # omits cwru's own plan

        with patch.object(Experiment, "train_on_plan") as mock_train:
            with pytest.raises(ValueError, match="cwru"):
                te.run_transfer(source_specs, target_specs)
            mock_train.assert_not_called()


# =====================================================================
# Pin 2 — deterministic pooled labels
# =====================================================================

class TestPooledLabel:
    def test_order_independent(self):
        task = _make_task(domain_factors=("fault_size", "condition"))
        combos_a = [{"fault_size": 1, "condition": 1}, {"fault_size": 2, "condition": 1}]
        combos_b = list(reversed(combos_a))
        assert _pooled_label(task, tuple(combos_a)) == _pooled_label(task, tuple(combos_b))

    def test_distinct_from_single_domain_label(self):
        task = _make_task(domain_factors=("fault_size", "condition"))
        single = task.label(fault_size=1, condition=1)
        pooled = _pooled_label(task, ({"fault_size": 1, "condition": 1},))
        assert pooled != single
        assert pooled.endswith("-pooled")


# =====================================================================
# Pin 3 — cross-experiment cls_labels mismatch raises the existing
# runtime guard (pins the real evaluate_on_plan path, not a mock)
# =====================================================================

class TestCrossExperimentLabelMismatchRuntimeCheck:
    def test_target_plan_with_different_classes_raises_runtime_error(self):
        cwru_plan = _make_plan("cwru", "fault_element-fault_size=1", ["normal", "inner ring"])
        paderborn_plan = _make_plan("paderborn", "fault_element-fault_size=1", ["normal", "outer ring"])

        cwru = _make_mock_collection("cwru", {frozenset({"fault_size": 1}.items()): cwru_plan})
        paderborn = _make_mock_collection("paderborn", {frozenset({"fault_size": 1}.items()): paderborn_plan})

        config = _make_experiment_config()
        te = TransferExperiment(
            {
                "cwru": (cwru, MagicMock()),
                "paderborn": (paderborn, MagicMock(side_effect=_fake_reader)),
            },
            config,
        )
        task = _make_task()

        source_cls_labels = {"inner ring": 0, "normal": 1}  # matches cwru_plan's classes
        with patch.object(Experiment, "train_on_plan", return_value=_make_exp_train_result(source_cls_labels)):
            with pytest.raises(RuntimeError, match="Train/Test labels mismatch"):
                te.run_transfer(
                    source_specs=(TransferSpec("cwru", task, {"fault_size": 1}),),
                    # paderborn (mismatched) evaluated before cwru's self-eval,
                    # so cwru's reader (an unconfigured MagicMock) is never reached.
                    target_specs=(
                        TransferSpec("paderborn", task, {"fault_size": 1}),
                        TransferSpec("cwru", task, {"fault_size": 1}),
                    ),
                )


# =====================================================================
# Pin 4 — empty filter tuple guard
# =====================================================================

class TestEmptyFilterGuard:
    def test_get_plan_empty_tuple_raises(self):
        cwru = _make_mock_collection("cwru", {})
        config = _make_experiment_config()
        te = TransferExperiment({"cwru": (cwru, MagicMock())}, config)
        task = _make_task()
        with pytest.raises(ValueError, match="at least one filter combination"):
            te._get_plan("cwru", task, ())


# =====================================================================
# Adjustment 5 — target specs accept pooled OR explicit single-domain
# =====================================================================

class TestTargetSpecGenerality:
    def test_pooled_source_with_pooled_and_single_domain_targets(self):
        task = _make_task(domain_factors=("fault_size",))
        # "normal" reuses the same code+files across domains (benign
        # duplicate, mirrors CWRU's NR class reusing one baseline recording);
        # "inner ring" gets distinct codes+files per domain (mirrors
        # genuinely new data per domain) -- using _make_plan's enumerate-based
        # codes for both would collide "inner ring" across domains too.
        cwru_d1 = DatasetPlan(
            dataset_name="cwru", label="unused1",
            sample_groups={
                "normal": SampleGroup(codes={0: ["normal_0.mat"]}, metadata={0: Metadata({})}),
                "inner ring": SampleGroup(codes={10: ["ir_d1.mat"]}, metadata={10: Metadata({})}),
            },
        )
        cwru_d2 = DatasetPlan(
            dataset_name="cwru", label="unused2",
            sample_groups={
                "normal": SampleGroup(codes={0: ["normal_0.mat"]}, metadata={0: Metadata({})}),
                "inner ring": SampleGroup(codes={20: ["ir_d2.mat"]}, metadata={20: Metadata({})}),
            },
        )
        cwru = _make_mock_collection("cwru", {
            frozenset({"fault_size": 1}.items()): cwru_d1,
            frozenset({"fault_size": 2}.items()): cwru_d2,
        })
        pad_single = _make_plan("paderborn", "fault_element-fault_size=1", ["normal", "inner ring"])
        paderborn = _make_mock_collection("paderborn", {
            frozenset({"fault_size": 1}.items()): pad_single,
        })
        config = _make_experiment_config()
        te = TransferExperiment({"cwru": (cwru, MagicMock()), "paderborn": (paderborn, MagicMock())}, config)

        pooled_filters = ({"fault_size": 1}, {"fault_size": 2})
        source_specs = (TransferSpec("cwru", task, pooled_filters),)
        target_specs = (
            TransferSpec("cwru", task, pooled_filters),          # pooled self-eval
            TransferSpec("paderborn", task, {"fault_size": 1}),  # explicit single-domain
        )

        cls_labels = {"inner ring": 0, "normal": 1}
        with patch.object(Experiment, "train_on_plan", return_value=_make_exp_train_result(cls_labels)), \
             patch.object(Experiment, "evaluate_on_plan", return_value=(_fake_confusion_matrix(), "unused")):
            mds = te.run_transfer(source_specs, target_specs)

        ds = mds.domain_solutions[0]
        expected_pooled_label = f"cwru:{_pooled_label(task, pooled_filters)}"
        assert expected_pooled_label in ds.confusion_matrices
        assert "paderborn:fault_element-fault_size=1" in ds.confusion_matrices


# =====================================================================
# Adjustment 3 — filesystem-safe artifact names
# =====================================================================

class TestSanitizeLabelForFilename:
    def test_colon_replaced(self):
        assert sanitize_label_for_filename("cwru:label") == "cwru__label"

    def test_unsafe_chars_replaced(self):
        assert sanitize_label_for_filename("cwru:a b/c") == "cwru__a-b-c"

    def test_two_realistic_labels_are_distinct(self):
        a = sanitize_label_for_filename("cwru:fault_element-fault_size=[1, 2]-pooled")
        b = sanitize_label_for_filename("paderborn:fault_element-fault_size=[1, 2]-pooled")
        assert a != b


class TestModelSaveDirUsesSanitizedNames:
    def test_saved_model_file_uses_sanitized_name_labels_stay_qualified(self, tmp_path):
        cwru = _make_mock_collection("cwru", {
            frozenset({"fault_size": 1}.items()): _make_plan("cwru", "fault_element-fault_size=1", ["normal", "inner ring"]),
        })
        config = _make_experiment_config()
        te = TransferExperiment({"cwru": (cwru, MagicMock())}, config)
        task = _make_task()

        model = nn.Linear(2, 2)
        cls_labels = {"inner ring": 0, "normal": 1}
        with patch.object(Experiment, "train_on_plan", return_value=_make_exp_train_result(cls_labels, model=model)), \
             patch.object(Experiment, "evaluate_on_plan", return_value=(_fake_confusion_matrix(), "unused")):
            mds = te.run_transfer(
                source_specs=(TransferSpec("cwru", task, {"fault_size": 1}),),
                target_specs=(TransferSpec("cwru", task, {"fault_size": 1}),),
                model_save_dir=tmp_path,
            )

        qualified_label = "cwru:fault_element-fault_size=1"
        expected_file = tmp_path / f"{sanitize_label_for_filename(qualified_label)}.pt"
        assert expected_file.exists()
        state_dict = torch.load(expected_file)
        assert set(state_dict.keys()) == set(model.state_dict().keys())

        # containers/CSV-facing labels still carry the colon form
        ds = mds.domain_solutions[0]
        assert ds.train_dataset_name == qualified_label
        assert qualified_label in ds.confusion_matrices
