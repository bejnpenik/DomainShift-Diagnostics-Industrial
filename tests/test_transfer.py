"""
Tests for experiment.transfer (cross-collection transfer orchestration).

Uses mocks for collection/reader, following tests/test_experiment.py's
style. Experiment.train_on_plan / evaluate_on_plan are patched at the class
level for orchestration-focused tests (label qualification, chokepoint
routing, self-eval preflight, pooling, sanitization) -- these test
TransferExperiment's own logic, not Experiment's training/data-loading
correctness. Tests that need to prove data actually flowed a particular way
(cross-experiment label mismatch, restriction dropping a class end-to-end,
normalizer contamination) use a working reader+processor stack instead.
"""

from __future__ import annotations

from pathlib import Path
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
    restrict_to_classes,
    _pooled_label,
)
from training.config import TrainerConfig, TrainResult
from training import Trainer
from model.config import ModelConfig
from normalization import Normalisator
from study.pipeline import PipelineConfig


# =====================================================================
# Helpers
# =====================================================================

_REPO_ROOT = Path(__file__).resolve().parent.parent

_FAULT_ELEMENT_HEADER = {
    "fault_element": {
        0: {"name": "normal", "alias": "NR"},
        1: {"name": "inner ring", "alias": "IR"},
        2: {"name": "outer ring", "alias": "OR"},
    }
}

_CWRU_FAULT_ELEMENT_HEADER_WITH_BALL = {
    "fault_element": {
        0: {"name": "normal", "alias": "NR"},
        1: {"name": "inner ring", "alias": "IR"},
        2: {"name": "outer ring", "alias": "OR"},
        3: {"name": "ball", "alias": "BA"},
    }
}


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


def _make_mock_collection(name, plans_by_filters, header=None):
    """plans_by_filters: {frozenset(filters.items()): DatasetPlan}."""
    collection = MagicMock()
    collection.name = name
    collection.channels = {"vibration": SignalChannelConfig(reader_channel="vibration", sampling_rate=12000)}
    collection.header = header or {}

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


def _make_task(domain_factors=("fault_size",), target="fault_element"):
    return Task(target=target, domain_factors=tuple(domain_factors))


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
        cwru = _make_mock_collection(
            "cwru",
            {frozenset({"fault_size": 1}.items()): _make_plan("cwru", "fault_element-fault_size=1", ["normal", "inner ring"])},
            header=_FAULT_ELEMENT_HEADER,
        )
        paderborn = _make_mock_collection(
            "paderborn",
            {frozenset({"fault_size": 1}.items()): _make_plan("paderborn", "fault_element-fault_size=1", ["normal", "inner ring"])},
            header=_FAULT_ELEMENT_HEADER,
        )
        config = _make_experiment_config()
        te = TransferExperiment(
            {"cwru": (cwru, MagicMock()), "paderborn": (paderborn, MagicMock())},
            config, class_aliases=("NR", "IR"), target="fault_element",
        )
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
# Pin 1 (Phase 2) — self-eval presence checked before any training
# =====================================================================

class TestSelfEvalPreflightCheck:
    def test_missing_self_eval_target_raises_before_training(self):
        cwru = _make_mock_collection(
            "cwru", {frozenset({"fault_size": 1}.items()): _make_plan("cwru", "fault_element-fault_size=1", ["normal"])},
            header=_FAULT_ELEMENT_HEADER,
        )
        paderborn = _make_mock_collection(
            "paderborn", {frozenset({"fault_size": 1}.items()): _make_plan("paderborn", "fault_element-fault_size=1", ["normal"])},
            header=_FAULT_ELEMENT_HEADER,
        )
        config = _make_experiment_config()
        te = TransferExperiment(
            {"cwru": (cwru, MagicMock()), "paderborn": (paderborn, MagicMock())},
            config, class_aliases=("NR",), target="fault_element",
        )
        task = _make_task()

        source_specs = (TransferSpec("cwru", task, {"fault_size": 1}),)
        target_specs = (TransferSpec("paderborn", task, {"fault_size": 1}),)  # omits cwru's own plan

        with patch.object(Experiment, "train_on_plan") as mock_train:
            with pytest.raises(ValueError, match="cwru"):
                te.run_transfer(source_specs, target_specs)
            mock_train.assert_not_called()


# =====================================================================
# Pin 2 (Phase 2) — deterministic pooled labels
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
# Pin 3 (Phase 2) — cross-experiment cls_labels mismatch raises the
# existing runtime guard (pins the real evaluate_on_plan path)
# =====================================================================

class TestCrossExperimentLabelMismatchRuntimeCheck:
    def test_target_plan_with_different_classes_raises_runtime_error(self):
        """Restriction now guarantees a single collection's plan always
        contains exactly its own resolved class_aliases names, so a
        same-collection mismatch can no longer happen post-restriction.
        The remaining way this guard can still fire is exactly what it's
        for: a misconfigured setup where the SAME alias resolves to
        DIFFERENT names across collections (what validate_transfer_setup's
        check (a) is meant to catch beforehand) -- this pins the runtime
        backstop for when that earlier check was skipped or regresses.
        """
        cwru_header = _FAULT_ELEMENT_HEADER  # IR -> "inner ring"
        paderborn_header = {
            "fault_element": {
                0: {"name": "normal", "alias": "NR"},
                1: {"name": "outer ring", "alias": "IR"},  # misconfigured: IR -> "outer ring" here
            }
        }
        cwru_plan = _make_plan("cwru", "fault_element-fault_size=1", ["normal", "inner ring"])
        paderborn_plan = _make_plan("paderborn", "fault_element-fault_size=1", ["normal", "outer ring"])

        cwru = _make_mock_collection("cwru", {frozenset({"fault_size": 1}.items()): cwru_plan}, header=cwru_header)
        paderborn = _make_mock_collection("paderborn", {frozenset({"fault_size": 1}.items()): paderborn_plan}, header=paderborn_header)

        config = _make_experiment_config()
        te = TransferExperiment(
            {"cwru": (cwru, MagicMock()), "paderborn": (paderborn, MagicMock(side_effect=_fake_reader))},
            config, class_aliases=("NR", "IR"), target="fault_element",
        )
        task = _make_task()

        source_cls_labels = {"inner ring": 0, "normal": 1}  # matches cwru_plan's restricted classes
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
# Pin 4 (Phase 2) — empty filter tuple guard
# =====================================================================

class TestEmptyFilterGuard:
    def test_get_plan_empty_tuple_raises(self):
        cwru = _make_mock_collection("cwru", {}, header=_FAULT_ELEMENT_HEADER)
        config = _make_experiment_config()
        te = TransferExperiment({"cwru": (cwru, MagicMock())}, config, class_aliases=("NR",), target="fault_element")
        task = _make_task()
        with pytest.raises(ValueError, match="at least one filter combination"):
            te._get_plan("cwru", task, ())


# =====================================================================
# Adjustment 5 (Phase 2) — target specs accept pooled OR explicit
# single-domain filters
# =====================================================================

class TestTargetSpecGenerality:
    def test_pooled_source_with_pooled_and_single_domain_targets(self):
        task = _make_task(domain_factors=("fault_size",))
        # "normal" reuses the same code+files across domains (benign
        # duplicate, mirrors CWRU's NR class); "inner ring" gets distinct
        # codes+files per domain (mirrors genuinely new data per domain).
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
        cwru = _make_mock_collection(
            "cwru", {frozenset({"fault_size": 1}.items()): cwru_d1, frozenset({"fault_size": 2}.items()): cwru_d2},
            header=_FAULT_ELEMENT_HEADER,
        )
        pad_single = _make_plan("paderborn", "fault_element-fault_size=1", ["normal", "inner ring"])
        paderborn = _make_mock_collection("paderborn", {frozenset({"fault_size": 1}.items()): pad_single}, header=_FAULT_ELEMENT_HEADER)
        config = _make_experiment_config()
        te = TransferExperiment(
            {"cwru": (cwru, MagicMock()), "paderborn": (paderborn, MagicMock())},
            config, class_aliases=("NR", "IR"), target="fault_element",
        )

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
# Adjustment 3 (Phase 2) — filesystem-safe artifact names
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
        cwru = _make_mock_collection(
            "cwru", {frozenset({"fault_size": 1}.items()): _make_plan("cwru", "fault_element-fault_size=1", ["normal", "inner ring"])},
            header=_FAULT_ELEMENT_HEADER,
        )
        config = _make_experiment_config()
        te = TransferExperiment({"cwru": (cwru, MagicMock())}, config, class_aliases=("NR", "IR"), target="fault_element")
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

        ds = mds.domain_solutions[0]
        assert ds.train_dataset_name == qualified_label
        assert qualified_label in ds.confusion_matrices


# =====================================================================
# Phase 3 pin 2 — task.target must match the constructor's target
# =====================================================================

class TestTargetGuard:
    def test_mismatched_task_target_raises_naming_both(self):
        cwru = _make_mock_collection(
            "cwru", {frozenset({"fault_size": 1}.items()): _make_plan("cwru", "label", ["normal", "inner ring"])},
            header=_FAULT_ELEMENT_HEADER,
        )
        config = _make_experiment_config()
        te = TransferExperiment({"cwru": (cwru, MagicMock())}, config, class_aliases=("NR", "IR"), target="fault_element")

        wrong_target_task = Task(target="bearing_position", domain_factors=("fault_size",))
        with pytest.raises(ValueError) as exc:
            te._get_plan("cwru", wrong_target_task, {"fault_size": 1})
        message = str(exc.value)
        assert "fault_element" in message
        assert "bearing_position" in message


# =====================================================================
# Phase 3 pin 3 — constructor guards
# =====================================================================

class TestConstructorGuards:
    def test_empty_class_aliases_raises(self):
        cwru = _make_mock_collection("cwru", {}, header=_FAULT_ELEMENT_HEADER)
        config = _make_experiment_config()
        with pytest.raises(ValueError, match="non-empty"):
            TransferExperiment({"cwru": (cwru, MagicMock())}, config, class_aliases=(), target="fault_element")

    def test_aliases_collapsing_to_same_name_raises(self):
        header = {
            "fault_element": {
                0: {"name": "normal", "alias": "NR"},
                1: {"name": "normal", "alias": "N2"},  # different alias, SAME name -> collapse
            }
        }
        cwru = _make_mock_collection("cwru", {}, header=header)
        config = _make_experiment_config()
        with pytest.raises(ValueError, match="collapse"):
            TransferExperiment({"cwru": (cwru, MagicMock())}, config, class_aliases=("NR", "N2"), target="fault_element")


# =====================================================================
# Phase 3 pin 4 — full run_transfer under ACTIVE restriction
# =====================================================================

class TestRunTransferUnderActiveRestriction:
    def test_extra_class_dropped_from_full_run_transfer_result(self):
        """No mocking of train_on_plan/evaluate_on_plan here -- the extra
        'ball' class must never reach the real training/eval path at all,
        proving the whole train->eval flow under restriction, not just the
        chokepoint in isolation."""
        cwru_plan = _make_plan("cwru", "label", ["normal", "inner ring", "ball"])
        cwru = _make_mock_collection(
            "cwru", {frozenset({"fault_size": 1}.items()): cwru_plan},
            header=_CWRU_FAULT_ELEMENT_HEADER_WITH_BALL,
        )
        config = _make_experiment_config()
        te = TransferExperiment(
            {"cwru": (cwru, MagicMock(side_effect=_fake_reader))},
            config, class_aliases=("NR", "IR"), target="fault_element",
        )
        task = _make_task()

        mds = te.run_transfer(
            source_specs=(TransferSpec("cwru", task, {"fault_size": 1}),),
            target_specs=(TransferSpec("cwru", task, {"fault_size": 1}),),
        )

        ds = mds.domain_solutions[0]
        # class_labels is {name: index} (matches run_pairwise's existing
        # convention: DomainSolution.class_labels is populated directly
        # from ExperimentTrainResult.cls_labels, name-keyed).
        assert set(ds.class_labels.keys()) == {"normal", "inner ring"}
        assert "ball" not in ds.class_labels
        for cm in ds.confusion_matrices.values():
            assert cm.shape == (2, 2)  # not 3x3 -- "ball" dimension never appears


# =====================================================================
# Phase 3 pin 5 — chokepoint spy
# =====================================================================

class TestChokepointSpy:
    def test_every_plan_used_passes_through_get_plan(self):
        cwru = _make_mock_collection(
            "cwru", {frozenset({"fault_size": 1}.items()): _make_plan("cwru", "fault_element-fault_size=1", ["normal", "inner ring"])},
            header=_FAULT_ELEMENT_HEADER,
        )
        paderborn = _make_mock_collection(
            "paderborn", {frozenset({"fault_size": 1}.items()): _make_plan("paderborn", "fault_element-fault_size=1", ["normal", "inner ring"])},
            header=_FAULT_ELEMENT_HEADER,
        )
        config = _make_experiment_config()
        te = TransferExperiment(
            {"cwru": (cwru, MagicMock()), "paderborn": (paderborn, MagicMock())},
            config, class_aliases=("NR", "IR"), target="fault_element",
        )
        task = _make_task()

        source_specs = (TransferSpec("cwru", task, {"fault_size": 1}),)
        target_specs = (
            TransferSpec("cwru", task, {"fault_size": 1}),
            TransferSpec("paderborn", task, {"fault_size": 1}),
        )

        cls_labels = {"inner ring": 0, "normal": 1}
        with patch.object(te, "_get_plan", wraps=te._get_plan) as spy:
            with patch.object(Experiment, "train_on_plan", return_value=_make_exp_train_result(cls_labels)), \
                 patch.object(Experiment, "evaluate_on_plan", return_value=(_fake_confusion_matrix(), "unused")):
                te.run_transfer(source_specs, target_specs)

        # Expected count = len(sources) + len(targets): Phase 2's design
        # resolves every plan exactly ONCE up front (targets first, then
        # sources) and reuses the resolved plan/label for both the
        # self-eval check and the actual train/eval calls -- it never
        # re-resolves a plan per source. If a future change switches to
        # per-source re-resolution of targets, this count will change;
        # that's a design decision to revisit deliberately, not a silent
        # regression to chase.
        assert spy.call_count == len(source_specs) + len(target_specs)


# =====================================================================
# restrict_to_classes — tests (a) identity-preserving drop, (d) missing
# declared class raises, plus label preservation
# =====================================================================

class TestRestrictToClasses:
    def test_drops_correct_classes_keeps_survivors_intact(self):
        codes_normal = {0: ["normal_0.mat"]}
        meta_normal = {0: Metadata({"x": 1})}
        codes_ir = {1: ["ir_0.mat"]}
        meta_ir = {1: Metadata({"x": 2})}
        plan = DatasetPlan(
            dataset_name="cwru", label="fault_element-fault_size=1",
            sample_groups={
                "normal": SampleGroup(codes=codes_normal, metadata=meta_normal),
                "inner ring": SampleGroup(codes=codes_ir, metadata=meta_ir),
                "ball": SampleGroup(codes={2: ["ball_0.mat"]}, metadata={2: Metadata({})}),
            },
        )
        restricted = restrict_to_classes(plan, frozenset({"normal", "inner ring"}))

        assert set(restricted.sample_groups) == {"normal", "inner ring"}
        assert restricted.sample_groups["normal"].codes is codes_normal
        assert restricted.sample_groups["normal"].metadata is meta_normal
        assert restricted.sample_groups["inner ring"].codes is codes_ir
        assert restricted.sample_groups["inner ring"].metadata is meta_ir

    def test_preserves_original_label(self):
        plan = _make_plan("cwru", "fault_element-fault_size=1-pooled", ["normal", "inner ring", "ball"])
        restricted = restrict_to_classes(plan, frozenset({"normal", "inner ring"}))
        assert restricted.label == plan.label

    def test_missing_declared_class_raises(self):
        plan = _make_plan("cwru", "label", ["normal"])
        with pytest.raises(ValueError, match="inner ring"):
            restrict_to_classes(plan, frozenset({"normal", "inner ring"}))


# =====================================================================
# test (b) — cls_labels from a restricted plan are exactly the resolved
# class_aliases, sorted-order indexed
# =====================================================================

class TestRestrictedPlanClsLabels:
    def test_cls_labels_exactly_class_aliases_sorted(self):
        cwru = _make_mock_collection(
            "cwru", {frozenset({"fault_size": 1}.items()): _make_plan("cwru", "label", ["normal", "inner ring", "ball"])},
            header=_CWRU_FAULT_ELEMENT_HEADER_WITH_BALL,
        )
        config = _make_experiment_config()
        te = TransferExperiment(
            {"cwru": (cwru, MagicMock(side_effect=_fake_reader))},
            config, class_aliases=("NR", "IR"), target="fault_element",
        )
        task = _make_task()

        plan = te._get_plan("cwru", task, {"fault_size": 1})
        _, _, cls_labels, _ = te._experiments["cwru"].load_plan_arrays(plan)

        assert cls_labels == {"inner ring": 0, "normal": 1}


# =====================================================================
# test (c) — dataset-mode normalizer fit through the restricted path is
# unaffected by the excluded, wildly-different-magnitude class
# =====================================================================

class TestNormalizerNotContaminatedByExcludedClass:
    def test_dataset_mode_normalizer_stats_unaffected_by_excluded_class(self):
        def magnitude_reader(path, metadata, channels):
            if "ball" in path:
                signal = (np.random.randn(2000).astype(np.float32) * 50.0) + 5000.0
            else:
                signal = np.random.randn(2000).astype(np.float32)
            return {"vibration": signal}

        cwru_plan = DatasetPlan(
            dataset_name="cwru", label="fault_element-fault_size=1",
            sample_groups={
                "normal": SampleGroup(codes={0: ["normal_0.mat"]}, metadata={0: Metadata({})}),
                "inner ring": SampleGroup(codes={1: ["inner_ring_0.mat"]}, metadata={1: Metadata({})}),
                "outer ring": SampleGroup(codes={2: ["outer_ring_0.mat"]}, metadata={2: Metadata({})}),
                "ball": SampleGroup(codes={3: ["ball_0.mat"]}, metadata={3: Metadata({})}),
            },
        )
        cwru = _make_mock_collection(
            "cwru", {frozenset({"fault_size": 1}.items()): cwru_plan},
            header=_CWRU_FAULT_ELEMENT_HEADER_WITH_BALL,
        )

        config = _make_experiment_config(normalization="dataset")
        te = TransferExperiment(
            {"cwru": (cwru, MagicMock(side_effect=magnitude_reader))},
            config, class_aliases=("NR", "IR", "OR"), target="fault_element",
        )
        task = _make_task()

        restricted_plan = te._get_plan("cwru", task, {"fault_size": 1})
        assert "ball" not in restricted_plan.sample_groups  # sanity: restriction actually happened

        _, _, _, train_norm = te._experiments["cwru"]._prepare_data_splits(restricted_plan)

        assert train_norm.mean.abs().max().item() < 5.0
        assert (train_norm.std - 1.0).abs().max().item() < 5.0


# =====================================================================
# Empty-after-restriction companion runtime check
# =====================================================================

class TestEmptyAfterRestriction:
    def test_kept_class_with_zero_codes_raises_naming_empty_classes(self):
        plan = DatasetPlan(
            dataset_name="cwru", label="label",
            sample_groups={
                "normal": SampleGroup(codes={0: ["normal_0.mat"]}, metadata={0: Metadata({})}),
                "inner ring": SampleGroup(codes={}, metadata={}),  # kept class, but empty
            },
        )
        cwru = _make_mock_collection("cwru", {frozenset({"fault_size": 1}.items()): plan}, header=_FAULT_ELEMENT_HEADER)
        config = _make_experiment_config()
        te = TransferExperiment({"cwru": (cwru, MagicMock())}, config, class_aliases=("NR", "IR"), target="fault_element")
        task = _make_task()

        with pytest.raises(ValueError, match="inner ring"):
            te._get_plan("cwru", task, {"fault_size": 1})


# =====================================================================
# _build_pooled_plan — rigorous dedup/collision tests (Adjustment 1)
# =====================================================================

class TestBuildPooledPlanMergeSemantics:
    def test_repeated_identical_code_deduplicates_not_sums(self):
        """Mirrors CWRU's 'normal' class: same code+files legitimately
        repeats across per-domain plans -- pooled count is the deduplicated
        union, not the raw sum."""
        cwru = _make_mock_collection("cwru", {}, header=_FAULT_ELEMENT_HEADER)
        config = _make_experiment_config()
        te = TransferExperiment({"cwru": (cwru, MagicMock())}, config, class_aliases=("NR",), target="fault_element")
        task = _make_task()

        plan_a = DatasetPlan(
            dataset_name="cwru", label="a",
            sample_groups={"normal": SampleGroup(codes={0: ["normal_0.mat"]}, metadata={0: Metadata({})})},
        )
        plan_b = DatasetPlan(
            dataset_name="cwru", label="b",
            sample_groups={"normal": SampleGroup(codes={0: ["normal_0.mat"]}, metadata={0: Metadata({})})},
        )
        cwru.construct_dataset_plan = MagicMock(side_effect=[plan_a, plan_b])

        pooled = te._build_pooled_plan(cwru, task, ({"fault_size": 1}, {"fault_size": 2}))

        assert len(pooled.sample_groups["normal"].codes) == 1  # deduplicated, not 2

    def test_same_code_different_files_raises(self):
        cwru = _make_mock_collection("cwru", {}, header=_FAULT_ELEMENT_HEADER)
        config = _make_experiment_config()
        te = TransferExperiment({"cwru": (cwru, MagicMock())}, config, class_aliases=("NR",), target="fault_element")
        task = _make_task()

        plan_a = DatasetPlan(
            dataset_name="cwru", label="a",
            sample_groups={"normal": SampleGroup(codes={0: ["a.mat"]}, metadata={0: Metadata({})})},
        )
        plan_b = DatasetPlan(
            dataset_name="cwru", label="b",
            sample_groups={"normal": SampleGroup(codes={0: ["b.mat"]}, metadata={0: Metadata({})})},
        )
        cwru.construct_dataset_plan = MagicMock(side_effect=[plan_a, plan_b])

        with pytest.raises(ValueError, match="Pooling conflict"):
            te._build_pooled_plan(cwru, task, ({"fault_size": 1}, {"fault_size": 2}))


# =====================================================================
# Phase 3 pin 1 — real CWRU/Paderborn collection metadata integration
# (no file I/O: check_files=False -- only path lists, never file content)
# =====================================================================

class TestRealCollectionMetadataIntegration:
    def test_real_cwru_and_paderborn_restrict_to_matching_class_sets(self):
        from collection.collection import DatasetCollection
        from collection.task_builder import build_task_and_filters_from_yaml

        cwru_collection = DatasetCollection(_REPO_ROOT / "configs/collections/cwru.yaml", check_files=False)
        pad_collection = DatasetCollection(_REPO_ROOT / "configs/collections/paderborn.yaml", check_files=False)

        cwru_task, cwru_filters = build_task_and_filters_from_yaml(
            _REPO_ROOT / "configs/tasks/cwru_fault_element.yaml", cwru_collection
        )
        pad_task, pad_filters = build_task_and_filters_from_yaml(
            _REPO_ROOT / "configs/tasks/paderborn_fault_element.yaml", pad_collection
        )

        config = _make_experiment_config()
        te = TransferExperiment(
            {"cwru": (cwru_collection, MagicMock()), "paderborn": (pad_collection, MagicMock())},
            config, class_aliases=("NR", "IR", "OR"), target="fault_element",
        )

        cwru_plan = te._get_plan("cwru", cwru_task, cwru_filters[0])
        pad_plan = te._get_plan("paderborn", pad_task, pad_filters[0])

        expected = {"normal", "inner ring", "outer ring"}
        assert set(cwru_plan.sample_groups) == expected
        assert set(pad_plan.sample_groups) == expected

        cwru_cls_labels = {cls: i for i, cls in enumerate(sorted(cwru_plan.sample_groups))}
        pad_cls_labels = {cls: i for i, cls in enumerate(sorted(pad_plan.sample_groups))}
        assert cwru_cls_labels == pad_cls_labels


# =====================================================================
# Phase 4 — TransferExperiment.train_on_plans
# =====================================================================

def _make_two_source_te(config=None, cwru_n=4, paderborn_n=3, aux=False):
    """Two collections, each with one valid plan, both restricted to
    {normal, inner ring}. aux=True gives every source a (n, 1) conditioning
    tensor from a patched Experiment.load_plan_arrays."""
    cwru = _make_mock_collection(
        "cwru", {frozenset({"fault_size": 1}.items()): _make_plan("cwru", "fault_element-fault_size=1", ["normal", "inner ring"])},
        header=_FAULT_ELEMENT_HEADER,
    )
    paderborn = _make_mock_collection(
        "paderborn", {frozenset({"fault_size": 1}.items()): _make_plan("paderborn", "fault_element-fault_size=1", ["normal", "inner ring"])},
        header=_FAULT_ELEMENT_HEADER,
    )
    config = config or _make_experiment_config()
    te = TransferExperiment(
        {"cwru": (cwru, MagicMock()), "paderborn": (paderborn, MagicMock())},
        config, class_aliases=("NR", "IR"), target="fault_element",
    )
    return te


_TWO_SOURCE_TASK = _make_task()
_TWO_SOURCE_SPECS = (
    TransferSpec("cwru", _TWO_SOURCE_TASK, {"fault_size": 1}),
    TransferSpec("paderborn", _TWO_SOURCE_TASK, {"fault_size": 1}),
)
_TWO_SOURCE_CLS_LABELS = {"inner ring": 0, "normal": 1}


def _capturing_trainer_fit(capture: dict):
    def fake_fit(self, model, train_data, val_data):
        capture["train_data"] = train_data
        capture["val_data"] = val_data
        return _make_train_result()
    return fake_fit


class TestTrainOnPlansBasic:
    def test_returns_experiment_train_result_with_combined_label(self):
        te = _make_two_source_te()

        def fake_load(self, plan):
            n = 4 if self._collection.name == "cwru" else 3
            return torch.randn(n, 1, 600), torch.zeros(n, dtype=torch.long), _TWO_SOURCE_CLS_LABELS, None

        with patch.object(Experiment, "load_plan_arrays", fake_load), patch.object(Trainer, "fit", _capturing_trainer_fit({})):
            result = te.train_on_plans(_TWO_SOURCE_SPECS)

        assert isinstance(result, ExperimentTrainResult)
        assert result.cls_labels == _TWO_SOURCE_CLS_LABELS
        assert result.dataset_label == "cwru:fault_element-fault_size=1+paderborn:fault_element-fault_size=1"

    def test_empty_sources_raises(self):
        te = _make_two_source_te()
        with pytest.raises(ValueError, match="at least one source"):
            te.train_on_plans(())

    def test_merged_sample_count_equals_sum_of_sources(self):
        te = _make_two_source_te()

        def fake_load(self, plan):
            n = 4 if self._collection.name == "cwru" else 3
            return torch.randn(n, 1, 600), torch.zeros(n, dtype=torch.long), _TWO_SOURCE_CLS_LABELS, None

        capture = {}
        with patch.object(Experiment, "load_plan_arrays", fake_load), patch.object(Trainer, "fit", _capturing_trainer_fit(capture)):
            te.train_on_plans(_TWO_SOURCE_SPECS)

        total = capture["train_data"][1].shape[0] + capture["val_data"][1].shape[0]
        assert total == 4 + 3


class TestTrainOnPlansValidation:
    def test_mismatched_cls_labels_raises(self):
        """Restriction guarantees a single collection's plan always has
        exactly its own resolved class_aliases names, so (as in Phase 3's
        cross-experiment test) the only way this can legitimately arise is
        the same alias resolving to different names across collections."""
        cwru_header = _FAULT_ELEMENT_HEADER  # IR -> "inner ring"
        paderborn_header = {
            "fault_element": {
                0: {"name": "normal", "alias": "NR"},
                1: {"name": "outer ring", "alias": "IR"},  # misconfigured
            }
        }
        cwru = _make_mock_collection(
            "cwru", {frozenset({"fault_size": 1}.items()): _make_plan("cwru", "l1", ["normal", "inner ring"])}, header=cwru_header,
        )
        paderborn = _make_mock_collection(
            "paderborn", {frozenset({"fault_size": 1}.items()): _make_plan("paderborn", "l2", ["normal", "outer ring"])}, header=paderborn_header,
        )
        config = _make_experiment_config()
        te = TransferExperiment(
            {"cwru": (cwru, MagicMock(side_effect=_fake_reader)), "paderborn": (paderborn, MagicMock(side_effect=_fake_reader))},
            config, class_aliases=("NR", "IR"), target="fault_element",
        )

        with pytest.raises(ValueError, match="cls_labels mismatch"):
            te.train_on_plans(_TWO_SOURCE_SPECS)

    def test_mismatched_feature_shapes_raises(self):
        te = _make_two_source_te()

        def fake_load(self, plan):
            shape = (1, 600) if self._collection.name == "cwru" else (1, 500)
            return torch.randn(2, *shape), torch.zeros(2, dtype=torch.long), _TWO_SOURCE_CLS_LABELS, None

        with patch.object(Experiment, "load_plan_arrays", fake_load):
            with pytest.raises(ValueError, match="Feature shape mismatch"):
                te.train_on_plans(_TWO_SOURCE_SPECS)


class TestTrainOnPlansAuxChannelGuard:
    """Pin 1: aux channels are all-or-none across sources. Mixed presence
    would otherwise make torch.cat produce an aux tensor shorter than X,
    silently misaligning rows with their signals -- no exception."""

    def test_mixed_aux_presence_raises(self):
        te = _make_two_source_te()

        def fake_load(self, plan):
            if self._collection.name == "cwru":
                return torch.randn(4, 1, 600), torch.zeros(4, dtype=torch.long), _TWO_SOURCE_CLS_LABELS, torch.randn(4, 1)
            return torch.randn(3, 1, 600), torch.zeros(3, dtype=torch.long), _TWO_SOURCE_CLS_LABELS, None

        with patch.object(Experiment, "load_plan_arrays", fake_load):
            with pytest.raises(ValueError, match="aux"):
                te.train_on_plans(_TWO_SOURCE_SPECS)

    def test_both_present_merged_aux_length_matches_x(self):
        te = _make_two_source_te()

        def fake_load(self, plan):
            n = 4 if self._collection.name == "cwru" else 3
            return torch.randn(n, 1, 600), torch.zeros(n, dtype=torch.long), _TWO_SOURCE_CLS_LABELS, torch.randn(n, 1)

        capture = {}
        with patch.object(Experiment, "load_plan_arrays", fake_load), patch.object(Trainer, "fit", _capturing_trainer_fit(capture)):
            te.train_on_plans(_TWO_SOURCE_SPECS)

        X_train, Y_train, aux_train = capture["train_data"]
        X_val, Y_val, aux_val = capture["val_data"]
        assert aux_train is not None and aux_val is not None
        assert aux_train.shape[0] == X_train.shape[0] == Y_train.shape[0]
        assert aux_val.shape[0] == X_val.shape[0] == Y_val.shape[0]
        assert aux_train.shape[0] + aux_val.shape[0] == 4 + 3


class TestTrainOnPlansDeterminism:
    """Mirrors TestBuilderDeterminism's build-twice-compare pattern."""

    def _run_capturing_splits(self, specs):
        te = _make_two_source_te()

        def fake_load(self, plan):
            n = 4 if self._collection.name == "cwru" else 3
            return torch.randn(n, 1, 600), torch.zeros(n, dtype=torch.long), _TWO_SOURCE_CLS_LABELS, None

        capture = {}
        with patch.object(Experiment, "load_plan_arrays", fake_load), patch.object(Trainer, "fit", _capturing_trainer_fit(capture)):
            te.train_on_plans(specs)
        return capture["train_data"][1].clone(), capture["val_data"][1].clone()

    def test_same_seed_produces_same_split(self):
        y1 = self._run_capturing_splits(_TWO_SOURCE_SPECS)
        y2 = self._run_capturing_splits(_TWO_SOURCE_SPECS)
        assert torch.equal(y1[0], y2[0])
        assert torch.equal(y1[1], y2[1])

    def test_order_invariant_same_seed_opposite_source_order(self):
        """Pin 2: train_on_plans canonicalizes source order by qualified
        label before seeding/loading, so swapped input order must not
        change the result."""
        specs_reversed = tuple(reversed(_TWO_SOURCE_SPECS))
        y_forward = self._run_capturing_splits(_TWO_SOURCE_SPECS)
        y_reversed = self._run_capturing_splits(specs_reversed)
        assert torch.equal(y_forward[0], y_reversed[0])
        assert torch.equal(y_forward[1], y_reversed[1])


class TestTrainOnPlansNormalizerFitScope:
    """Pin 4: pins 'normalizer fit on merged train only' directly, since
    it's the headline semantic of train_on_plans' docstring."""

    def test_dataset_mode_fits_exactly_once_on_merged_train_split(self):
        te = _make_two_source_te(config=_make_experiment_config(normalization="dataset"))

        def fake_load(self, plan):
            n = 4 if self._collection.name == "cwru" else 3
            return torch.randn(n, 1, 600), torch.zeros(n, dtype=torch.long), _TWO_SOURCE_CLS_LABELS, None

        capture = {}
        fit_calls = []
        real_fit = Normalisator.fit

        def spy_fit(self, x):
            fit_calls.append(x)
            return real_fit(self, x)

        with patch.object(Experiment, "load_plan_arrays", fake_load), \
             patch.object(Trainer, "fit", _capturing_trainer_fit(capture)), \
             patch.object(Normalisator, "fit", spy_fit):
            te.train_on_plans(_TWO_SOURCE_SPECS)

        assert len(fit_calls) == 1
        assert fit_calls[0].shape[0] == capture["train_data"][1].shape[0]

    @pytest.mark.parametrize("mode", ["sample", "none"])
    def test_non_dataset_modes_never_call_fit(self, mode):
        te = _make_two_source_te(config=_make_experiment_config(normalization=mode))

        def fake_load(self, plan):
            n = 4 if self._collection.name == "cwru" else 3
            return torch.randn(n, 1, 600), torch.zeros(n, dtype=torch.long), _TWO_SOURCE_CLS_LABELS, None

        with patch.object(Experiment, "load_plan_arrays", fake_load), \
             patch.object(Trainer, "fit", _capturing_trainer_fit({})), \
             patch.object(Normalisator, "fit") as mock_fit:
            te.train_on_plans(_TWO_SOURCE_SPECS)

        mock_fit.assert_not_called()
