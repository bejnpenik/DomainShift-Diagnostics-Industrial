"""
Tests for training package.

Covers: TrainerConfig, TrainResult, Trainer, EarlyStopper, IllStopper
"""

from __future__ import annotations

import pytest
import numpy as np
import torch
import torch.nn as nn

from training.config import TrainerConfig, TrainResult
from training.trainer import Trainer
from training.early_stopping import EarlyStopper, IllStopper


# =====================================================================
# TrainerConfig
# =====================================================================

class TestTrainerConfig:
    def test_defaults(self):
        cfg = TrainerConfig()
        assert cfg.max_epochs == 2000
        assert cfg.optimizer_name == "adamw"
        assert cfg.lr == 1e-3
        assert cfg.device == "cuda"
        assert cfg.early_stopping == (10, 0.0)
        assert cfg.noise == (0.1, 0.1)

    def test_custom_values(self):
        cfg = TrainerConfig(
            max_epochs=500,
            optimizer_name="sgd",
            lr=0.01,
            momentum=0.95,
            device="cpu",
        )
        assert cfg.max_epochs == 500
        assert cfg.optimizer_name == "sgd"
        assert cfg.momentum == 0.95

    def test_frozen(self):
        cfg = TrainerConfig()
        with pytest.raises(Exception):
            cfg.lr = 0.1

    def test_invalid_optimizer(self):
        with pytest.raises(Exception):
            TrainerConfig(optimizer_name="rmsprop")

    def test_invalid_lr(self):
        with pytest.raises(Exception):
            TrainerConfig(lr=-1)

    def test_invalid_max_epochs(self):
        with pytest.raises(Exception):
            TrainerConfig(max_epochs=0)

    def test_invalid_weight_decay(self):
        with pytest.raises(Exception):
            TrainerConfig(weight_decay=-0.1)

    def test_no_early_stopping(self):
        cfg = TrainerConfig(early_stopping=None)
        assert cfg.early_stopping is None

    def test_no_noise(self):
        cfg = TrainerConfig(noise=None)
        assert cfg.noise is None

    def test_serialization_roundtrip(self):
        cfg = TrainerConfig(max_epochs=500, lr=0.01, optimizer_name="sgd")
        json_str = cfg.model_dump_json()
        restored = TrainerConfig.model_validate_json(json_str)
        assert restored == cfg


# =====================================================================
# TrainResult
# =====================================================================

class TestTrainResult:
    def test_fields(self):
        model = nn.Linear(10, 2)
        result = TrainResult(
            model=model,
            epochs_run=50,
            train_loss=0.1,
            train_acc=95.0,
            val_loss=0.2,
            val_acc=90.0,
        )
        assert result.epochs_run == 50
        assert result.train_loss == 0.1
        assert result.val_acc == 90.0
        assert result.model is model


# =====================================================================
# EarlyStopper
# =====================================================================

class TestEarlyStopper:
    def test_no_stop_when_improving(self):
        stopper = EarlyStopper(patience=3)
        model = nn.Linear(10, 2)
        assert not stopper.step(1.0, model)
        assert not stopper.step(0.9, model)
        assert not stopper.step(0.8, model)
        assert not stopper.step(0.7, model)

    def test_stop_after_patience(self):
        stopper = EarlyStopper(patience=3)
        model = nn.Linear(10, 2)
        assert not stopper.step(1.0, model)
        assert not stopper.step(0.5, model)  # best
        assert not stopper.step(0.6, model)  # worse, count=1
        assert not stopper.step(0.7, model)  # worse, count=2
        assert not stopper.step(0.8, model)  # worse, count=3
        assert stopper.step(0.9, model)      # worse, count=4 > patience

    def test_resets_on_improvement(self):
        stopper = EarlyStopper(patience=2)
        model = nn.Linear(10, 2)
        stopper.step(1.0, model)
        stopper.step(1.1, model)  # count=1
        stopper.step(0.5, model)  # improved, count resets
        stopper.step(0.6, model)  # count=1
        assert not stopper.step(0.7, model)  # count=2
        assert stopper.step(0.8, model)      # count=3 > patience

    def test_min_delta(self):
        stopper = EarlyStopper(patience=2, min_delta=0.1)
        model = nn.Linear(10, 2)
        stopper.step(1.0, model)
        # Improvement of 0.05 < min_delta=0.1, doesn't count
        assert not stopper.step(0.95, model)  # count=1
        assert not stopper.step(0.94, model)  # count=2
        assert stopper.step(0.93, model)      # count=3 > patience

    def test_saves_best_state(self):
        stopper = EarlyStopper(patience=2)
        model = nn.Linear(10, 2)
        stopper.step(1.0, model)
        stopper.step(0.5, model)  # best
        best = stopper.best_state()
        assert best is not None
        assert isinstance(best, dict)
        assert "weight" in list(best.keys())[0]

    def test_best_state_is_copy(self):
        """Modifying model after step shouldn't affect saved state."""
        stopper = EarlyStopper(patience=2)
        model = nn.Linear(10, 2)
        stopper.step(0.5, model)
        saved_weight = stopper.best_state()["weight"].clone()
        # Modify model
        with torch.no_grad():
            model.weight.fill_(999.0)
        # Saved state should be unchanged
        torch.testing.assert_close(stopper.best_state()["weight"], saved_weight)


# =====================================================================
# IllStopper
# =====================================================================

class TestIllStopper:
    def test_no_stop_when_improving(self):
        stopper = IllStopper(patience=3)
        model = nn.Linear(10, 2)
        assert not stopper.step(1.0, model)
        assert not stopper.step(0.9, model)
        assert not stopper.step(0.8, model)

    def test_stop_after_patience(self):
        stopper = IllStopper(patience=3)
        model = nn.Linear(10, 2)
        assert not stopper.step(0.5, model)  # best
        assert not stopper.step(0.6, model)  # count=1
        assert not stopper.step(0.7, model)  # count=2
        assert not stopper.step(0.8, model)  # count=3
        assert stopper.step(0.9, model)      # count=4 > patience

    def test_saves_best_state(self):
        stopper = IllStopper(patience=2)
        model = nn.Linear(10, 2)
        stopper.step(1.0, model)
        stopper.step(0.5, model)
        assert stopper.best_state() is not None


# =====================================================================
# Trainer
# =====================================================================

def _make_data(n_samples: int, input_dim: int, n_classes: int):
    """Helper to create random train/val data."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, 1, input_dim)
    Y = torch.randint(0, n_classes, (n_samples,))
    return X, Y


class TestTrainer:
    def test_fit_basic(self):
        cfg = TrainerConfig(
            max_epochs=5,
            device="cpu",
            early_stopping=None,
            noise=None,
            verbose_level=0,
        )
        trainer = Trainer(cfg)
        model = nn.Sequential(nn.Flatten(), nn.Linear(600, 3))
        train_data = _make_data(50, 600, 3)
        val_data = _make_data(20, 600, 3)

        result = trainer.fit(model, train_data, val_data)

        assert isinstance(result, TrainResult)
        assert result.epochs_run == 5
        assert result.train_loss > 0
        assert 0 <= result.train_acc <= 100
        assert 0 <= result.val_acc <= 100

    def test_fit_with_early_stopping(self):
        cfg = TrainerConfig(
            max_epochs=1000,
            device="cpu",
            early_stopping=(3, 0.0),
            min_epochs=0,
            noise=None,
            verbose_level=0,
        )
        trainer = Trainer(cfg)
        model = nn.Sequential(nn.Flatten(), nn.Linear(600, 3))
        train_data = _make_data(50, 600, 3)
        val_data = _make_data(20, 600, 3)

        result = trainer.fit(model, train_data, val_data)

        # Should stop before max_epochs
        assert result.epochs_run < 1000

    def test_fit_with_noise(self):
        cfg = TrainerConfig(
            max_epochs=5,
            device="cpu",
            early_stopping=None,
            noise=(0.5, 0.1),
            verbose_level=0,
        )
        trainer = Trainer(cfg)
        model = nn.Sequential(nn.Flatten(), nn.Linear(600, 3))
        train_data = _make_data(50, 600, 3)
        val_data = _make_data(20, 600, 3)

        result = trainer.fit(model, train_data, val_data)
        assert result.epochs_run == 5

    def test_fit_sgd(self):
        cfg = TrainerConfig(
            max_epochs=5,
            optimizer_name="sgd",
            lr=0.01,
            momentum=0.9,
            device="cpu",
            early_stopping=None,
            noise=None,
            verbose_level=0,
        )
        trainer = Trainer(cfg)
        model = nn.Sequential(nn.Flatten(), nn.Linear(600, 3))
        train_data = _make_data(50, 600, 3)
        val_data = _make_data(20, 600, 3)

        result = trainer.fit(model, train_data, val_data)
        assert result.epochs_run == 5

    def test_predict_confusion_matrix(self):
        cfg = TrainerConfig(device="cpu", verbose_level=0)
        trainer = Trainer(cfg)

        torch.manual_seed(42)
        model = nn.Sequential(nn.Flatten(), nn.Linear(600, 3))
        X = torch.randn(30, 1, 600)
        Y = torch.randint(0, 3, (30,))

        cm = trainer.predict(model, X, Y)
        assert cm.shape == (3, 3)
        assert cm.sum() == 30

    def test_predict_perfect_model(self):
        """A model that memorizes should get diagonal confusion matrix."""
        cfg = TrainerConfig(
            max_epochs=200,
            device="cpu",
            early_stopping=None,
            noise=None,
            lr=0.01,
            verbose_level=0,
        )
        trainer = Trainer(cfg)

        torch.manual_seed(42)
        # Simple linearly separable data
        X = torch.zeros(60, 1, 10)
        Y = torch.tensor([0]*20 + [1]*20 + [2]*20)
        X[:20, 0, 0] = 1.0
        X[20:40, 0, 1] = 1.0
        X[40:, 0, 2] = 1.0

        model = nn.Sequential(nn.Flatten(), nn.Linear(10, 3))
        result = trainer.fit(model, (X, Y), (X, Y))

        cm = trainer.predict(result.model, X, Y)
        # Should be near-perfect
        assert cm.trace() >= 55  # at least 55/60 correct

    def test_model_returned_on_device(self):
        cfg = TrainerConfig(
            max_epochs=2,
            device="cpu",
            early_stopping=None,
            noise=None,
            verbose_level=0,
        )
        trainer = Trainer(cfg)
        model = nn.Sequential(nn.Flatten(), nn.Linear(600, 3))
        train_data = _make_data(50, 600, 3)
        val_data = _make_data(20, 600, 3)

        result = trainer.fit(model, train_data, val_data)
        # Model parameters should be on cpu
        param_device = next(result.model.parameters()).device
        assert str(param_device) == "cpu"

    def test_verbose_prints(self, capsys):
        cfg = TrainerConfig(
            max_epochs=3,
            device="cpu",
            early_stopping=None,
            noise=None,
            verbose_level=1,
        )
        trainer = Trainer(cfg)
        model = nn.Sequential(nn.Flatten(), nn.Linear(600, 3))
        train_data = _make_data(50, 600, 3)
        val_data = _make_data(20, 600, 3)

        trainer.fit(model, train_data, val_data)
        captured = capsys.readouterr()
        assert "Epoch" in captured.out
        assert "train loss" in captured.out

    def test_silent_mode(self, capsys):
        cfg = TrainerConfig(
            max_epochs=3,
            device="cpu",
            early_stopping=None,
            noise=None,
            verbose_level=0,
        )
        trainer = Trainer(cfg)
        model = nn.Sequential(nn.Flatten(), nn.Linear(600, 3))
        train_data = _make_data(50, 600, 3)
        val_data = _make_data(20, 600, 3)

        trainer.fit(model, train_data, val_data)
        captured = capsys.readouterr()
        assert captured.out == ""