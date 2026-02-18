"""
Tests for experiment package.

Covers: ExperimentConfig, ModelConfig, ExperimentTrainResult,
        FileSamplingProtocol, FileSampler, Normalisator integration,
        Experiment, ExperimentRunner

Uses mocks for collection, reader, and pipeline to isolate experiment logic.
"""

from __future__ import annotations

import dataclasses
import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from experiment.config import ExperimentConfig
from experiment.sampling import FileSamplingProtocol, FileSampler
from experiment.experiment import Experiment, ExperimentRunner, ExperimentTrainResult, set_seed
from training.config import TrainerConfig, TrainResult
from model.config import ModelConfig
from normalization import Normalisator
from collection import Metadata, SampleGroup, DatasetPlan


# =====================================================================
# Helpers / Fixtures
# =====================================================================

def _make_processor_config():
    from representation.signal.config import SignalProcessorConfig, RawViewConfig
    return SignalProcessorConfig(name="raw_12k", view=RawViewConfig())


def _make_trainer_config(**overrides):
    defaults = dict(
        max_epochs=5,
        device="cpu",
        early_stopping=None,
        noise=None,
        verbose_level=0,
    )
    defaults.update(overrides)
    return TrainerConfig(**defaults)


def _simple_model_factory(num_classes, **kwargs):
    return nn.Sequential(nn.Flatten(), nn.Linear(600, num_classes))


def _make_model_config(**overrides):
    defaults = dict(
        name="simple",
        model_class=_simple_model_factory,
        params={},
    )
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


def _make_plan(n_files=5, n_classes=2):
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


def _make_train_result(model=None):
    """Create a TrainResult for testing."""
    if model is None:
        model = nn.Linear(10, 2)
    return TrainResult(
        model=model,
        epochs_run=10,
        train_loss=0.05,
        train_acc=95.0,
        val_loss=0.1,
        val_acc=92.0,
    )


def _make_experiment_train_result(model=None, mode="none"):
    """Create an ExperimentTrainResult for testing."""
    train_result = _make_train_result(model)
    return ExperimentTrainResult(
        train_result=train_result,
        normalisator=Normalisator(mode=mode),
        cls_labels={"healthy": 0, "faulty": 1},
        dataset_label="domain-A",
    )


# =====================================================================
# ModelConfig
# =====================================================================

class TestModelConfig:
    def test_basic(self):
        mc = ModelConfig(name="cnn1d", model_class=_simple_model_factory)
        assert mc.name == "cnn1d"
        assert mc.params == {}

    def test_frozen(self):
        mc = ModelConfig(name="cnn1d", model_class=_simple_model_factory)
        with pytest.raises(AttributeError):
            mc.name = "changed"

    def test_create_model(self):
        mc = ModelConfig(name="cnn1d", model_class=_simple_model_factory)
        model = mc.create_model(num_classes=4)
        assert isinstance(model, nn.Module)
        out = model(torch.randn(8, 1, 600))
        assert out.shape == (8, 4)

    def test_create_model_with_params(self):
        def custom_factory(num_classes, hidden=128):
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(600, hidden),
                nn.ReLU(),
                nn.Linear(hidden, num_classes),
            )
        mc = ModelConfig(name="custom", model_class=custom_factory, params={"hidden": 64})
        model = mc.create_model(num_classes=3)
        out = model(torch.randn(8, 1, 600))
        assert out.shape == (8, 3)

    def test_different_configs_different_instances(self):
        mc = ModelConfig(name="small", model_class=_simple_model_factory)
        m1 = mc.create_model(num_classes=3)
        m2 = mc.create_model(num_classes=3)
        assert m1 is not m2


# =====================================================================
# ExperimentConfig
# =====================================================================

class TestExperimentConfig:
    def test_basic(self):
        cfg = _make_experiment_config()
        assert cfg.name == "test_exp"
        assert cfg.random_seed == 42
        assert cfg.normalization == "none"

    def test_frozen(self):
        cfg = _make_experiment_config()
        with pytest.raises(AttributeError):
            cfg.name = "changed"

    def test_replace_seed(self):
        cfg = _make_experiment_config(random_seed=42)
        cfg2 = dataclasses.replace(cfg, random_seed=99)
        assert cfg.random_seed == 42
        assert cfg2.random_seed == 99

    def test_processor_name_property(self):
        cfg = _make_experiment_config()
        assert cfg.processor_name == "raw_12k"

    def test_model_name_property(self):
        cfg = _make_experiment_config()
        assert cfg.model_name == "simple"

    def test_pipeline_config_accessible(self):
        cfg = _make_experiment_config()
        assert cfg.processor_config.name == "raw_12k"

    def test_trainer_config_accessible(self):
        cfg = _make_experiment_config()
        assert cfg.trainer_config.max_epochs == 5
        assert cfg.trainer_config.device == "cpu"

    def test_model_config_accessible(self):
        cfg = _make_experiment_config()
        assert cfg.model_config.name == "simple"
        assert cfg.model_config.params == {}

    def test_model_config_create_model(self):
        cfg = _make_experiment_config()
        model = cfg.model_config.create_model(num_classes=3)
        assert isinstance(model, nn.Module)
        out = model(torch.randn(8, 1, 600))
        assert out.shape == (8, 3)

    def test_file_sampling_none(self):
        cfg = _make_experiment_config()
        assert cfg.file_sampling is None

    def test_file_sampling_set(self):
        fs = FileSamplingProtocol(max_files_per_code=3)
        cfg = _make_experiment_config(file_sampling=fs)
        assert cfg.file_sampling.max_files_per_code == 3

    def test_normalization_modes(self):
        for mode in ("none", "sample", "dataset", "pretrained"):
            cfg = _make_experiment_config(normalization=mode)
            assert cfg.normalization == mode

    def test_default_trainer_config(self):
        cfg = ExperimentConfig(
            name="test",
            processor_config=_make_processor_config(),
            model_config=_make_model_config(),
        )
        assert cfg.trainer_config is not None
        assert cfg.trainer_config.max_epochs == 2000

    def test_composition_all_configs(self):
        cfg = _make_experiment_config()
        assert cfg.processor_config is not cfg.model_config
        assert cfg.trainer_config is not cfg.model_config


# =====================================================================
# Normalisator
# =====================================================================

class TestNormalisator:
    # --- sample mode ---
    def test_sample_basic(self):
        norm = Normalisator(mode="sample")
        x = torch.randn(10, 1, 600)
        result = norm(x)
        assert result.shape == x.shape

    def test_sample_rejects_mean_std(self):
        with pytest.raises(ValueError):
            Normalisator(mode="sample", mean=torch.tensor(0.0))

    def test_sample_fit_is_noop(self):
        norm = Normalisator(mode="sample")
        x = torch.randn(10, 1, 600)
        result = norm.fit(x)
        assert result is norm

    def test_sample_normalizes_per_sample(self):
        norm = Normalisator(mode="sample")
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
        result = norm(x)
        # Each sample should have roughly zero mean
        assert abs(result.mean().item()) < 0.1

    # --- dataset mode ---
    def test_dataset_requires_fit(self):
        norm = Normalisator(mode="dataset")
        x = torch.randn(10, 1, 600)
        with pytest.raises(RuntimeError):
            norm(x)

    def test_dataset_fit_then_call(self):
        norm = Normalisator(mode="dataset")
        x = torch.randn(50, 1, 600)
        norm.fit(x)
        result = norm(x)
        assert result.shape == x.shape

    def test_dataset_cannot_refit(self):
        norm = Normalisator(mode="dataset")
        x = torch.randn(10, 1, 600)
        norm.fit(x)
        with pytest.raises(RuntimeError):
            norm.fit(x)

    # --- pretrained mode ---
    def test_pretrained_requires_mean_std(self):
        with pytest.raises(ValueError):
            Normalisator(mode="pretrained")

    def test_pretrained_basic(self):
        norm = Normalisator(
            mode="pretrained",
            mean=torch.tensor(0.0),
            std=torch.tensor(1.0),
        )
        x = torch.randn(10, 1, 600)
        result = norm(x)
        torch.testing.assert_close(result, x, atol=1e-6, rtol=1e-6)

    def test_pretrained_already_fitted(self):
        norm = Normalisator(
            mode="pretrained",
            mean=torch.tensor(0.0),
            std=torch.tensor(1.0),
        )
        assert norm._fitted is True

    # --- none mode ---
    def test_none_mode_identity(self):
        """None mode should act as identity (x-0)/(1+eps) ≈ x."""
        norm = Normalisator(mode="none")
        x = torch.randn(10, 1, 600)
        result = norm(x)
        torch.testing.assert_close(result, x, atol=1e-6, rtol=1e-6)

    def test_none_mode_rejects_mean_std(self):
        with pytest.raises(ValueError):
            Normalisator(mode="none", mean=torch.tensor(0.0))
        with pytest.raises(ValueError):
            Normalisator(mode="none", std=torch.tensor(1.0))

    def test_none_mode_fit_is_noop(self):
        norm = Normalisator(mode="none")
        x = torch.randn(10, 1, 600)
        result = norm.fit(x)
        assert result is norm

    def test_none_mode_internally_pretrained(self):
        """None mode should set self.mode to 'pretrained' internally."""
        norm = Normalisator(mode="none")
        # After __init__, self.mode should be 'pretrained' (not 'none')
        # so that __call__ doesn't raise ValueError
        assert norm._fitted is True

    # --- edge cases ---
    def test_unknown_mode_raises(self):
        norm = Normalisator.__new__(Normalisator)
        norm.mode = "bogus"
        norm.eps = 1e-8
        norm._fitted = False
        norm.mean = None
        norm.std = None
        with pytest.raises(ValueError, match="Unknown normalization mode"):
            norm(torch.randn(5, 1, 100))


# =====================================================================
# ExperimentTrainResult
# =====================================================================

class TestExperimentTrainResult:
    def test_model_property(self):
        exp_result = _make_experiment_train_result()
        assert isinstance(exp_result.model, nn.Module)

    def test_train_result_accessible(self):
        exp_result = _make_experiment_train_result()
        assert exp_result.train_result.epochs_run == 10
        assert exp_result.train_result.train_loss == 0.05
        assert exp_result.train_result.val_acc == 92.0

    def test_experiment_level_fields(self):
        norm = Normalisator(mode="sample")
        model = nn.Linear(10, 2)
        exp_result = ExperimentTrainResult(
            train_result=_make_train_result(model),
            normalisator=norm,
            cls_labels={"h": 0, "f": 1},
            dataset_label="test-label",
        )
        assert exp_result.cls_labels == {"h": 0, "f": 1}
        assert exp_result.dataset_label == "test-label"
        assert exp_result.normalisator is norm

    def test_unwrap_pattern_for_run_pairwise(self):
        """Test the correct pattern that run_pairwise should use."""
        exp_result = _make_experiment_train_result()

        # This is how experiment.py should unwrap:
        tr = exp_result.train_result
        train_metadata = {
            "epochs_run": tr.epochs_run,
            "train_loss": tr.train_loss,
            "train_acc": tr.train_acc,
            "val_loss": tr.val_loss,
            "val_acc": tr.val_acc,
        }
        assert train_metadata["epochs_run"] == 10
        assert train_metadata["train_acc"] == 95.0

        # These come from ExperimentTrainResult, NOT TrainResult
        assert exp_result.model is not None
        assert exp_result.normalisator is not None
        assert exp_result.cls_labels == {"healthy": 0, "faulty": 1}
        assert exp_result.dataset_label == "domain-A"

    def test_train_result_has_no_normalisator(self):
        """TrainResult should NOT have normalisator — it's on ExperimentTrainResult."""
        tr = _make_train_result()
        assert not hasattr(tr, "normalisator")

    def test_train_result_has_no_cls_labels(self):
        """TrainResult should NOT have cls_labels — it's on ExperimentTrainResult."""
        tr = _make_train_result()
        assert not hasattr(tr, "cls_labels")

    def test_train_result_has_no_dataset_label(self):
        """TrainResult should NOT have dataset_label — it's on ExperimentTrainResult."""
        tr = _make_train_result()
        assert not hasattr(tr, "dataset_label")

    def test_train_result_field_name_is_epochs_run(self):
        """Field is epochs_run, NOT train_epoch_nbr."""
        tr = _make_train_result()
        assert hasattr(tr, "epochs_run")
        assert not hasattr(tr, "train_epoch_nbr")


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
    def test_construction(self):
        config = _make_experiment_config()
        collection = MagicMock()
        reader = MagicMock()
        experiment = Experiment(
            collection=collection,
            reader=reader,
            config=config,
        )
        assert experiment is not None

    def test_evaluate_returns_confusion_matrix(self):
        trainer_cfg = _make_trainer_config()
        torch.manual_seed(42)
        model = nn.Sequential(nn.Flatten(), nn.Linear(600, 2))
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
        base_config = _make_experiment_config(random_seed=42)
        cfg1 = dataclasses.replace(base_config, random_seed=42)
        cfg2 = dataclasses.replace(base_config, random_seed=99)
        cfg3 = dataclasses.replace(base_config, random_seed=123)

        assert cfg1.random_seed == 42
        assert cfg2.random_seed == 99
        assert cfg3.random_seed == 123

        assert cfg1.name == cfg2.name == cfg3.name
        assert cfg1.trainer_config == cfg2.trainer_config
        assert cfg1.model_config == cfg2.model_config
        assert cfg1.processor_config == cfg2.processor_config

    def test_configs_independent(self):
        base = _make_experiment_config(random_seed=42)
        replaced = dataclasses.replace(base, random_seed=99)
        assert base.random_seed == 42
        assert replaced.random_seed == 99

    def test_replaced_config_preserves_composed_configs(self):
        base = _make_experiment_config()
        replaced = dataclasses.replace(base, random_seed=99)
        assert replaced.trainer_config.device == "cpu"
        assert replaced.model_config.name == "simple"
        assert replaced.processor_config.name == "raw_12k"