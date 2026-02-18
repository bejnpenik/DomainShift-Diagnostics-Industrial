"""
Tests for study package.

Covers: ExperimentSpec, StudyDesign, Study, StudyAnalyzer,
        StudyGridBuilder, DependentFactor

Uses synthetic DomainSolution/MultiDomainSolution/StudySolution
for analyzer tests.

Note: SignalPipelineConfig has been renamed to SignalProcessorConfig.
      The Protocol is ProcessorConfig from representation.
"""

from __future__ import annotations

import pytest
import numpy as np
import torch.nn as nn
from unittest.mock import MagicMock, patch
from pathlib import Path
import pickle
import tempfile

from study.design import ExperimentSpec, StudyDesign
from study.study import Study
from study.grid import StudyGridBuilder, DependentFactor

from experiment.config import ExperimentConfig
from experiment.sampling import FileSamplingProtocol
from training.config import TrainerConfig
from model.config import ModelConfig
from collection import Task, Rule
from results import (
    DomainSolution,
    MultiDomainSolution,
    RepeatedMultiDomainSolution,
    StudySolution,
    StudySolutionBuilder,
)


# =====================================================================
# Helpers
# =====================================================================

def _make_processor_config(name="raw_12k"):
    from representation.signal.config import SignalProcessorConfig, RawViewConfig
    return SignalProcessorConfig(name=name, view=RawViewConfig())


def _make_trainer_config(**overrides):
    defaults = dict(
        max_epochs=5, device="cpu", early_stopping=None,
        noise=None, verbose_level=0,
    )
    defaults.update(overrides)
    return TrainerConfig(**defaults)


def _simple_model_factory(num_classes, **kwargs):
    return nn.Sequential(nn.Flatten(), nn.Linear(600, num_classes))


def _make_model_config(name="simple", **overrides):
    defaults = dict(name=name, model_class=_simple_model_factory, params={})
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_experiment_config(name="exp1", **overrides):
    defaults = dict(
        name=name,
        processor_config=_make_processor_config(),
        model_config=_make_model_config(),
        trainer_config=_make_trainer_config(),
        normalization="none",
        random_seed=42,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _make_task():
    return Task(
        target="fault_element",
        domain_factors=("condition",),
        defaults=Rule(fixed={"fault_size": 0}),
    )


def _make_filter_combinations():
    return ({"condition": 1}, {"condition": 2}, {"condition": 3})


def _make_experiment_spec(name="exp1", **config_overrides):
    return ExperimentSpec(
        task=_make_task(),
        filter_combinations=_make_filter_combinations(),
        config=_make_experiment_config(name=name, **config_overrides),
    )


def _make_confusion_matrix(n_classes=3, accuracy=0.9):
    """Create a synthetic confusion matrix with approximate accuracy."""
    total = 100
    correct = int(total * accuracy)
    wrong = total - correct
    cm = np.zeros((n_classes, n_classes), dtype=int)
    per_class = correct // n_classes
    per_class_wrong = wrong // (n_classes * (n_classes - 1)) if n_classes > 1 else 0
    for i in range(n_classes):
        cm[i, i] = per_class
        for j in range(n_classes):
            if i != j:
                cm[i, j] = per_class_wrong
    return cm


def _make_domain_solution(train_name="domain_A", test_names=None, seed=42, accuracy=0.9):
    if test_names is None:
        test_names = ["domain_A", "domain_B"]
    cms = {}
    for tn in test_names:
        acc = accuracy if tn == train_name else accuracy * 0.8
        cms[tn] = _make_confusion_matrix(accuracy=acc)
    return DomainSolution(
        train_dataset_name=train_name,
        class_labels={0: "healthy", 1: "faulty", 2: "severe"},
        seed=seed,
        train_metadata={"epochs_run": 10, "train_loss": 0.05, "train_acc": 95.0,
                        "val_loss": 0.1, "val_acc": 92.0},
        confusion_matrices=cms,
    )


def _make_multi_domain_solution(config_name="exp1", seed=42):
    domains = ["domain_A", "domain_B"]
    solutions = []
    for train_name in domains:
        solutions.append(_make_domain_solution(
            train_name=train_name, test_names=domains, seed=seed,
        ))
    return MultiDomainSolution(
        config_name=config_name,
        domain_solutions=solutions,
        processor_name="raw_12k",
    )


def _make_study_solution(config_names=("exp1", "exp2"), seeds=(42, 99)):
    repeated = []
    for cfg_name in config_names:
        mds_list = [_make_multi_domain_solution(cfg_name, seed) for seed in seeds]
        repeated.append(RepeatedMultiDomainSolution(multi_domain_solutions=mds_list))
    return StudySolution(
        study_name="test_study",
        timestamp="20260218_120000",
        repeated_solutions=repeated,
    )


# =====================================================================
# ExperimentSpec
# =====================================================================

class TestExperimentSpec:
    def test_basic(self):
        spec = _make_experiment_spec()
        assert spec.name == "exp1"
        assert spec.num_domains == 3

    def test_name_from_config(self):
        spec = _make_experiment_spec(name="my_exp")
        assert spec.name == "my_exp"

    def test_label(self):
        spec = _make_experiment_spec(name="cnn1d_raw")
        label = spec.label()
        assert "cnn1d_raw" in label
        assert "fault_element" in label

    def test_processor_name(self):
        spec = _make_experiment_spec()
        assert spec.processor_name == "raw_12k"

    def test_frozen(self):
        spec = _make_experiment_spec()
        with pytest.raises(AttributeError):
            spec.task = None


# =====================================================================
# StudyDesign
# =====================================================================

class TestStudyDesign:
    def test_basic(self):
        design = StudyDesign(
            name="test_study",
            experiment_specs=(_make_experiment_spec("exp1"), _make_experiment_spec("exp2")),
            seeds=(42, 99, 123),
        )
        assert design.num_configs == 2
        assert design.num_seeds == 3

    def test_total_runs(self):
        design = StudyDesign(
            name="test",
            experiment_specs=(_make_experiment_spec("a"), _make_experiment_spec("b")),
            seeds=(42, 99),
        )
        # 2 specs * 3 domains each * 2 seeds = 12
        assert design.total_runs == 2 * 3 * 2

    def test_duplicate_names_raises(self):
        with pytest.raises(ValueError, match="unique names"):
            StudyDesign(
                name="bad",
                experiment_specs=(_make_experiment_spec("dup"), _make_experiment_spec("dup")),
                seeds=(42,),
            )

    def test_duplicate_seeds_raises(self):
        with pytest.raises(ValueError, match="unique"):
            StudyDesign(
                name="bad",
                experiment_specs=(_make_experiment_spec("exp1"),),
                seeds=(42, 42),
            )

    def test_summary(self):
        design = StudyDesign(
            name="bearing_study",
            experiment_specs=(_make_experiment_spec("exp1"),),
            seeds=(42, 99),
            description="Test study",
        )
        summary = design.summary()
        assert "bearing_study" in summary
        assert "exp1" in summary

    def test_description_and_metadata(self):
        design = StudyDesign(
            name="test",
            experiment_specs=(_make_experiment_spec("exp1"),),
            seeds=(42,),
            description="My study",
            metadata={"version": "1.0"},
        )
        assert design.description == "My study"
        assert design.metadata["version"] == "1.0"


# =====================================================================
# Study (runner)
# =====================================================================

class TestStudy:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            study_solution = _make_study_solution()
            design = StudyDesign(
                name="test",
                experiment_specs=(_make_experiment_spec("exp1"),),
                seeds=(42,),
            )

            runner = Study(
                collection=MagicMock(),
                reader=MagicMock(),
                results_dir=Path(tmpdir),
            )

            save_path = runner.save("test", study_solution, design)
            assert save_path.exists()
            assert (save_path / "results.pkl").exists()
            assert (save_path / "design.pkl").exists()
            assert (save_path / "metadata.txt").exists()

            loaded_solution, loaded_design = Study.load(save_path)
            assert loaded_solution.study_name == "test_study"
            assert loaded_design is not None

    def test_save_without_design(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            study_solution = _make_study_solution()
            runner = Study(
                collection=MagicMock(),
                reader=MagicMock(),
                results_dir=Path(tmpdir),
            )

            save_path = runner.save("test", study_solution)
            loaded_solution, loaded_design = Study.load(save_path)
            assert loaded_solution.study_name == "test_study"
            assert loaded_design is None

    def test_list_saved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = Study(
                collection=MagicMock(),
                reader=MagicMock(),
                results_dir=Path(tmpdir),
            )
            study_solution = _make_study_solution()
            runner.save("study_a", study_solution)
            runner.save("study_b", study_solution)

            saved = runner.list_saved()
            assert len(saved) == 2

    def test_results_dir_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "nested" / "results"
            runner = Study(
                collection=MagicMock(),
                reader=MagicMock(),
                results_dir=results_path,
            )
            assert results_path.exists()


# =====================================================================
# DependentFactor
# =====================================================================

class TestDependentFactor:
    def test_simple_resolve(self):
        dep = DependentFactor(
            depends_on="optimizer",
            mapping={"adamw": 1e-3, "sgd": 1e-2},
        )
        assert dep.resolve({"optimizer": "adamw"}) == 1e-3
        assert dep.resolve({"optimizer": "sgd"}) == 1e-2

    def test_default_value(self):
        dep = DependentFactor(
            depends_on="optimizer",
            mapping={"adamw": 1e-3},
            default=5e-4,
        )
        assert dep.resolve({"optimizer": "unknown"}) == 5e-4

    def test_missing_key_returns_default(self):
        dep = DependentFactor(
            depends_on="optimizer",
            mapping={"adamw": 1e-3},
            default=None,
        )
        assert dep.resolve({}) is None

    def test_nested_resolve(self):
        dep = DependentFactor(
            depends_on=("model", "rate"),
            mapping={
                "CNN1D": {12000: "raw_12k", 48000: "raw_48k"},
                "CNN2D": {12000: "stft_12k", 48000: "stft_48k"},
            },
        )
        assert dep.resolve({"model": "CNN1D", "rate": 12000}) == "raw_12k"
        assert dep.resolve({"model": "CNN2D", "rate": 48000}) == "stft_48k"

    def test_nested_missing_returns_default(self):
        dep = DependentFactor(
            depends_on=("a", "b"),
            mapping={"x": {"y": "found"}},
            default="fallback",
        )
        assert dep.resolve({"a": "x", "b": "z"}) == "fallback"
        assert dep.resolve({"a": "missing", "b": "y"}) == "fallback"


# =====================================================================
# StudyGridBuilder
# =====================================================================

class TestStudyGridBuilder:
    def test_single_factor(self):
        builder = StudyGridBuilder()
        builder.set_factors(lr=(1e-3, 1e-2))
        configs = builder.build()
        assert len(configs) == 2

    def test_two_factors_cartesian(self):
        builder = StudyGridBuilder()
        builder.set_factors(
            lr=(1e-3, 1e-2),
            optimizer=("adamw", "sgd"),
        )
        configs = builder.build()
        assert len(configs) == 4

    def test_three_factors(self):
        builder = StudyGridBuilder()
        builder.set_factors(
            a=(1, 2),
            b=(3, 4),
            c=(5, 6),
        )
        configs = builder.build()
        assert len(configs) == 8

    def test_independent_factors(self):
        builder = StudyGridBuilder()
        builder.set_factors(lr=(1e-3, 1e-2))
        builder.set_independent(device="cpu", max_epochs=100)
        configs = builder.build()
        for cfg in configs:
            assert cfg["device"] == "cpu"
            assert cfg["max_epochs"] == 100

    def test_dependent_factor(self):
        builder = StudyGridBuilder()
        builder.set_factors(optimizer=("adamw", "sgd"))
        builder.set_dependent(
            "lr",
            depends_on="optimizer",
            mapping={"adamw": 1e-3, "sgd": 1e-2},
        )
        configs = builder.build()
        adamw_cfg = [c for c in configs if c["optimizer"] == "adamw"][0]
        sgd_cfg = [c for c in configs if c["optimizer"] == "sgd"][0]
        assert adamw_cfg["lr"] == 1e-3
        assert sgd_cfg["lr"] == 1e-2

    def test_nested_dependent_factor(self):
        builder = StudyGridBuilder()
        builder.set_factors(
            model=("CNN1D", "CNN2D"),
            rate=(12000, 48000),
        )
        builder.set_dependent(
            "processor",
            depends_on=("model", "rate"),
            mapping={
                "CNN1D": {12000: "raw_12k", 48000: "raw_48k"},
                "CNN2D": {12000: "stft_12k", 48000: "stft_48k"},
            },
        )
        configs = builder.build()
        assert len(configs) == 4
        cnn1d_12k = [c for c in configs if c["model"] == "CNN1D" and c["rate"] == 12000][0]
        assert cnn1d_12k["processor"] == "raw_12k"

    def test_name_generation_default(self):
        builder = StudyGridBuilder()
        builder.set_factors(optimizer=("adamw", "sgd"))
        configs = builder.build()
        names = [c["name"] for c in configs]
        assert "adamw" in names[0] or "sgd" in names[0]
        assert len(set(names)) == 2  # unique names

    def test_name_template_string(self):
        builder = StudyGridBuilder()
        builder.set_factors(optimizer=("adamw", "sgd"))
        builder.set_name_template("{optimizer}_config")
        configs = builder.build()
        assert configs[0]["name"] == "adamw_config"
        assert configs[1]["name"] == "sgd_config"

    def test_name_template_callable(self):
        builder = StudyGridBuilder()
        builder.set_factors(a=(1, 2))
        builder.set_name_template(lambda f: f"custom_{f['a']}")
        configs = builder.build()
        assert configs[0]["name"] == "custom_1"
        assert configs[1]["name"] == "custom_2"

    def test_num_combinations(self):
        builder = StudyGridBuilder()
        builder.set_factors(a=(1, 2, 3), b=(4, 5))
        assert builder.num_combinations == 6

    def test_num_combinations_no_factors(self):
        builder = StudyGridBuilder()
        assert builder.num_combinations == 1

    def test_chaining(self):
        builder = StudyGridBuilder()
        result = builder.set_factors(a=(1, 2)).set_independent(b=3).set_dependent(
            "c", depends_on="a", mapping={1: "x", 2: "y"}
        )
        assert result is builder

    def test_invalid_factor_not_tuple(self):
        builder = StudyGridBuilder()
        with pytest.raises(ValueError):
            builder.set_factors(lr=0.001)

    def test_no_factors_yields_one_config(self):
        builder = StudyGridBuilder()
        builder.set_independent(device="cpu")
        configs = builder.build()
        assert len(configs) == 1
        assert configs[0]["device"] == "cpu"

    def test_iter_combinations(self):
        builder = StudyGridBuilder()
        builder.set_factors(x=(1, 2))
        builder.set_independent(y=10)
        combos = list(builder.iter_combinations())
        assert len(combos) == 2
        assert all(c["y"] == 10 for c in combos)

    def test_build_experiment_configs(self):
        builder = StudyGridBuilder()
        builder.set_factors(
            model_config=(
                _make_model_config("small"),
                _make_model_config("large"),
            ),
        )
        builder.set_independent(
            processor_config=_make_processor_config(),
            trainer_config=_make_trainer_config(),
            normalization="none",
            random_seed=42,
        )
        configs = builder.build_experiment_configs()
        assert len(configs) == 2
        assert all(isinstance(c, ExperimentConfig) for c in configs)
        names = [c.name for c in configs]
        assert len(set(names)) == 2  # unique names

    def test_build_study_design(self):
        builder = StudyGridBuilder()
        builder.set_factors(
            model_config=(
                _make_model_config("small"),
                _make_model_config("large"),
            ),
        )
        builder.set_independent(
            processor_config=_make_processor_config(),
            trainer_config=_make_trainer_config(),
            normalization="none",
            random_seed=42,
        )
        design = builder.build_study_design(
            study_name="grid_test",
            task=_make_task(),
            filter_combinations=_make_filter_combinations(),
            seeds=(42, 99),
            description="Grid test study",
        )
        assert isinstance(design, StudyDesign)
        assert design.name == "grid_test"
        assert design.num_configs == 2
        assert design.num_seeds == 2

    def test_summary(self):
        builder = StudyGridBuilder()
        builder.set_factors(a=(1, 2))
        builder.set_independent(b=3)
        builder.set_dependent("c", depends_on="a", mapping={1: "x", 2: "y"})
        summary = builder.summary()
        assert "Total combinations: 2" in summary
        assert "Varying factors" in summary
        assert "Independent factors" in summary
        assert "Dependent factors" in summary


# =====================================================================
# StudySolutionBuilder
# =====================================================================

class TestStudySolutionBuilder:
    def test_basic_build(self):
        builder = StudySolutionBuilder("test_study")
        mds1 = _make_multi_domain_solution("cfg1", seed=42)
        mds2 = _make_multi_domain_solution("cfg1", seed=99)
        builder.add_multi_domain_solution(mds1)
        builder.add_multi_domain_solution(mds2)

        result = builder.build()
        assert result.study_name == "test_study"
        assert result.num_configs == 1
        assert len(result.repeated_solutions[0].multi_domain_solutions) == 2

    def test_multiple_configs(self):
        builder = StudySolutionBuilder("test")
        builder.add_multi_domain_solution(_make_multi_domain_solution("cfg1", 42))
        builder.add_multi_domain_solution(_make_multi_domain_solution("cfg2", 42))
        result = builder.build()
        assert result.num_configs == 2
        assert set(result.config_names) == {"cfg1", "cfg2"}

    def test_metadata(self):
        builder = StudySolutionBuilder("test")
        builder.add_multi_domain_solution(_make_multi_domain_solution("cfg1", 42))
        builder.set_metadata("description", "A test study")
        builder.set_metadata("version", "1.0")
        result = builder.build()
        assert result.study_metadata["description"] == "A test study"
        assert result.study_metadata["version"] == "1.0"

    def test_chaining(self):
        builder = StudySolutionBuilder("test")
        result = builder.add_multi_domain_solution(
            _make_multi_domain_solution("cfg1", 42)
        ).set_metadata("key", "val").set_timestamp("20260101_000000")
        assert result is builder

    def test_custom_timestamp(self):
        builder = StudySolutionBuilder("test")
        builder.add_multi_domain_solution(_make_multi_domain_solution("cfg1", 42))
        builder.set_timestamp("custom_time")
        result = builder.build()
        assert result.timestamp == "custom_time"

    def test_auto_timestamp(self):
        builder = StudySolutionBuilder("test")
        builder.add_multi_domain_solution(_make_multi_domain_solution("cfg1", 42))
        result = builder.build()
        assert result.timestamp is not None
        assert len(result.timestamp) > 0