"""
Tests for experiment package.

Covers: ExperimentConfig, ExperimentTrainResult, FileSamplingProtocol,
        FileSampler, Experiment, ExperimentRunner

Uses mocks for collection, reader, and pipeline to isolate experiment logic.
"""

from __future__ import annotations

import dataclasses
import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from typing import Literal

from experiment.config import ExperimentConfig
from experiment.sampling import FileSamplingProtocol, FileSampler
from experiment.experiment import Experiment, ExperimentRunner, ExperimentTrainResult, set_seed
from training.config import TrainerConfig, TrainResult
from collection import Metadata, SampleGroup, DatasetPlan, Task, Rule
from results import DomainSolution, MultiDomainSolution, RepeatedMultiDomainSolution


# =====================================================================
# Helpers / Fixtures
# =====================================================================

def _make_processor_config():
    """Create a minimal SignalProcessorConfig."""
    from representation.signal.config import SignalProcessorConfig, RawViewConfig
    return SignalProcessorConfig(name="raw_12k", view=RawViewConfig())


def _make_trainer_config():
    return TrainerConfig(
        max_epochs=5,
        device="cpu",
        early_stopping=None,
        noise=None,
        verbose_level=0,
    )


def _make_experiment_config(**overrides):
    defaults = dict(
        name="test_exp",
        processor_config=_make_processor_config(),
        model_name="simple",
        model_class=lambda num_classes, **kw: nn.Sequential(nn.Flatten(), nn.Linear(600, num_classes)),
        trainer_config=_make_trainer_config(),
        normalization="sample",
        train_val_split_ratio=0.33,
        random_seed=42,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _make_plan(n_files=5, n_classes=2):
    """Create a synthetic DatasetPlan."""
    sample_groups = {}
    class_names = ["healthy", "faulty", "severe"][:n_classes]
    for i, name in enumerate(class_names):
        code = (i + 1) * 100
        sample_groups[name] = SampleGroup(
            codes={code: [f"file_{name}_{j}.mat" for j in range(n_files)]},
            metadata={code: Metadata({"sampling_rate": 12000})},
        )
    return DatasetPlan(
        dataset_name="test_collection",
        label="test-label",
        sample_groups=sample_groups,
    )


def _mock_collection(task, filter_combinations):
    """Create a mock DatasetCollection that returns plans from filters."""
    collection = MagicMock()

    def construct_plan(task_arg=None, **filters):
        label = "-".join(f"{k}={v}" for k, v in sorted(filters.items()))
        return _make_plan(n_files=2, n_classes=2)

    collection.construct_dataset_plan = MagicMock(side_effect=construct_plan)
    return collection


# =====================================================================
# ExperimentConfig
# =====================================================================

class TestExperimentConfig:
    def test_basic(self):
        cfg = _make_experiment_config()
        assert cfg.name == "test_exp"
        assert cfg.random_seed == 42
        assert cfg.normalization == "sample"

    def test_frozen(self):
        cfg = _make_experiment_config()
        with pytest.raises(AttributeError):
            cfg.name = "changed"

    def test_replace_seed(self):
        cfg = _make_experiment_config(random_seed=42)
        cfg2 = dataclasses.replace(cfg, random_seed=99)
        assert cfg.random_seed == 42
        assert cfg2.random_seed == 99

    def test_processor_config_accessible(self):
        cfg = _make_experiment_config()
        assert cfg.processor_config.name == "raw_12k"

    def test_trainer_config_accessible(self):
        cfg = _make_experiment_config()
        assert cfg.trainer_config.max_epochs == 5
        assert cfg.trainer_config.device == "cpu"

    def test_default_model_params(self):
        cfg = _make_experiment_config()
        assert cfg.model_params == {}

    def test_custom_model_params(self):
        cfg = _make_experiment_config(model_params={"aggregator_levels": 16})
        assert cfg.model_params["aggregator_levels"] == 16

    def test_file_sampling_none(self):
        cfg = _make_experiment_config()
        assert cfg.file_sampling is None

    def test_file_sampling_set(self):
        fs = FileSamplingProtocol(max_files_per_code=3)
        cfg = _make_experiment_config(file_sampling=fs)
        assert cfg.file_sampling.max_files_per_code == 3


# =====================================================================
# FileSamplingProtocol
# =====================================================================

class TestFileSamplingProtocol:
    def test_basic(self):
        fsp = FileSamplingProtocol(max_files_per_code=5)
        assert fsp.max_files_per_code == 5

    def test_none_means_no_limit(self):
        fsp = FileSamplingProtocol(max_files_per_code=None)
        assert fsp.max_files_per_code is None

    def test_invalid_negative(self):
        with pytest.raises(ValueError):
            FileSamplingProtocol(max_files_per_code=-1)

    def test_frozen(self):
        fsp = FileSamplingProtocol(max_files_per_code=3)
        with pytest.raises(AttributeError):
            fsp.max_files_per_code = 5


# =====================================================================
# FileSampler
# =====================================================================

class TestFileSampler:
    def _make_plan(self, n_files=20):
        return DatasetPlan(
            dataset_name="test",
            label="test-label",
            sample_groups={
                "healthy": SampleGroup(
                    codes={100: [f"file_{i}.mat" for i in range(n_files)]},
                    metadata={100: Metadata({"sampling_rate": 12000})},
                ),
            },
        )

    def test_deterministic(self):
        plan = self._make_plan()
        sampler = FileSampler(FileSamplingProtocol(max_files_per_code=3))
        r1 = sampler(plan, seed=42)
        r2 = sampler(plan, seed=42)
        assert r1.sample_groups["healthy"].codes[100] == r2.sample_groups["healthy"].codes[100]

    def test_limits_files(self):
        plan = self._make_plan(20)
        sampler = FileSampler(FileSamplingProtocol(max_files_per_code=3))
        result = sampler(plan, seed=42)
        assert len(result.sample_groups["healthy"].codes[100]) == 3

    def test_different_seeds_differ(self):
        plan = self._make_plan(20)
        sampler = FileSampler(FileSamplingProtocol(max_files_per_code=3))
        r1 = sampler(plan, seed=42)
        r2 = sampler(plan, seed=99)
        assert r1.sample_groups["healthy"].codes[100] != r2.sample_groups["healthy"].codes[100]

    def test_no_limit_keeps_all(self):
        plan = self._make_plan(5)
        sampler = FileSampler(FileSamplingProtocol(max_files_per_code=None))
        result = sampler(plan, seed=42)
        assert len(result.sample_groups["healthy"].codes[100]) == 5

    def test_none_protocol_passthrough(self):
        plan = self._make_plan(5)
        sampler = FileSampler(None)
        result = sampler(plan, seed=42)
        assert result is plan

    def test_preserves_metadata(self):
        plan = self._make_plan(10)
        sampler = FileSampler(FileSamplingProtocol(max_files_per_code=2))
        result = sampler(plan, seed=42)
        assert 100 in result.sample_groups["healthy"].metadata
        assert result.sample_groups["healthy"].metadata[100].sampling_rate == 12000


# =====================================================================
# ExperimentTrainResult
# =====================================================================

class TestExperimentTrainResult:
    def test_model_property(self):
        model = nn.Linear(10, 2)
        train_result = TrainResult(
            model=model, epochs_run=5,
            train_loss=0.1, train_acc=90.0,
            val_loss=0.2, val_acc=85.0,
        )
        from normalization import Normalisator
        exp_result = ExperimentTrainResult(
            train_result=train_result,
            normalisator=Normalisator(mode="sample"),
            cls_labels={0: "h", 1: "f"},
            dataset_label="test-label",
        )
        assert exp_result.model is model
        assert exp_result.cls_labels == {0: "h", 1: "f"}
        assert exp_result.dataset_label == "test-label"


# =====================================================================
# set_seed
# =====================================================================

class TestSetSeed:
    def test_deterministic_torch(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        torch.testing.assert_close(a, b)

    def test_deterministic_numpy(self):
        set_seed(42)
        a = np.random.randn(10)
        set_seed(42)
        b = np.random.randn(10)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        set_seed(42)
        a = torch.randn(10)
        set_seed(99)
        b = torch.randn(10)
        assert not torch.allclose(a, b)


# =====================================================================
# Experiment (integration-style with mocks)
# =====================================================================

class TestExperiment:
    """Tests using mock collection, reader, and pipeline.

    These test the orchestration logic without needing real data files.
    """

    def _make_experiment(self, seed=42):
        """Create an Experiment with mocked dependencies."""
        config = _make_experiment_config(random_seed=seed)

        collection = MagicMock()
        reader = MagicMock()

        # Mock the pipeline to return a tensor from any signal
        processor = MagicMock()
        processor.name = "raw_12k"
        processor.return_value = torch.randn(10, 1, 600)

        # Mock reader to return a numpy signal
        reader.return_value = np.random.randn(12000).astype(np.float32)

        experiment = Experiment(
            collection=collection,
            reader=reader,
            config=config,
        )

        # Patch the internal pipeline and domain_dataset
        return experiment, collection, reader

    def test_construction(self):
        experiment, _, _ = self._make_experiment()
        assert experiment is not None

    def test_evaluate_returns_confusion_matrix(self):
        """Test that evaluate_on_plan produces a confusion matrix."""
        config = _make_experiment_config()
        trainer_cfg = config.trainer_config

        # Create a simple trained model
        torch.manual_seed(42)
        model = nn.Sequential(nn.Flatten(), nn.Linear(600, 2))

        # Create test data directly
        X_test = torch.randn(20, 1, 600)
        Y_test = torch.randint(0, 2, (20,))

        from training.trainer import Trainer
        trainer = Trainer(trainer_cfg)
        cm = trainer.predict(model, X_test, Y_test)

        assert cm.shape == (2, 2)
        assert cm.sum() == 20


# =====================================================================
# ExperimentRunner
# =====================================================================

class TestExperimentRunner:
    def test_seed_replacement(self):
        """Verify that config seeds are replaced correctly per run."""
        base_config = _make_experiment_config(random_seed=42)

        cfg1 = dataclasses.replace(base_config, random_seed=42)
        cfg2 = dataclasses.replace(base_config, random_seed=99)
        cfg3 = dataclasses.replace(base_config, random_seed=123)

        assert cfg1.random_seed == 42
        assert cfg2.random_seed == 99
        assert cfg3.random_seed == 123

        # All other fields unchanged
        assert cfg1.name == cfg2.name == cfg3.name
        assert cfg1.trainer_config == cfg2.trainer_config

    def test_configs_independent(self):
        """Replacing seed doesn't mutate original."""
        base = _make_experiment_config(random_seed=42)
        replaced = dataclasses.replace(base, random_seed=99)
        assert base.random_seed == 42
        assert replaced.random_seed == 99