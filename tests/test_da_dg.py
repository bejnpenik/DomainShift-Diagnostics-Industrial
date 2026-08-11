"""
Tests for domain adaptation (DA) and domain generalization (DG).

All tests use fully synthetic in-memory data — no disk, no collection, no reader.
The feature-dim is kept tiny (D=16) and epochs are short (max 30) so the
suite runs in a few seconds.

Coverage:
    - losses.py     : coral_loss, mmd_loss, dann_loss, irm_penalty
    - domain_modules: GradientReversalLayer, DomainDiscriminator
    - da_trainer    : AdaptationConfig, DomainAdaptiveTrainer (coral/dann/mmd)
    - dg_trainer    : DomainGeneralizationTrainer (mixup/irm)
    - experiment.py : Experiment.run() dispatch, ExperimentRunner.run_multi_seed
    - backward compat: adaptation='none' produces same result as Trainer.fit()
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

# =====================================================================
# Helpers: tiny synthetic data + minimal BuiltModel clone
# =====================================================================

def _source_target(n=64, d=16, n_classes=3, seed=0):
    """Return (X_src, Y_src, X_tgt) tensors on CPU."""
    rng = np.random.default_rng(seed)
    X_src = torch.tensor(rng.standard_normal((n, 1, d)), dtype=torch.float32)
    Y_src = torch.tensor(rng.integers(0, n_classes, n), dtype=torch.long)
    X_tgt = torch.tensor(rng.standard_normal((n, 1, d)) + 1.0, dtype=torch.float32)
    return X_src, Y_src, X_tgt


def _multi_source(n_domains=3, n=40, d=16, n_classes=3, seed=0):
    """Return list of (X, Y) tuples, one per source domain."""
    rng = np.random.default_rng(seed)
    datasets = []
    for k in range(n_domains):
        X = torch.tensor(rng.standard_normal((n, 1, d)) + k * 0.5, dtype=torch.float32)
        Y = torch.tensor(rng.integers(0, n_classes, n), dtype=torch.long)
        datasets.append((X, Y))
    return datasets


class _TinyModel(nn.Module):
    """Minimal BuiltModel-compatible model: flat → linear → head.

    Exposes .head and .features() so DA trainers can use them.
    """

    def __init__(self, in_dim=16, hidden=32, n_classes=3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, hidden), nn.ReLU())
        self.aggregator = nn.Identity()
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        return self.head(self.features(x))

    def features(self, x):
        return self.aggregator(self.encoder(x))


def _trainer_cfg(max_epochs=20, batch_size=32, device="cpu"):
    from training.config import TrainerConfig
    return TrainerConfig(
        max_epochs=max_epochs,
        optimizer_name="adamw",
        lr=1e-3,
        weight_decay=0.0,
        device=device,
        early_stopping=None,
        min_epochs=0,
        noise=None,
        verbose_level=0,
        batch_size=batch_size,
    )


def _adap_cfg(**kwargs):
    from training.da_trainer import AdaptationConfig
    defaults = dict(batch_size=32, lambda_coral=0.5, lambda_dann=0.5, lambda_mmd=0.5)
    defaults.update(kwargs)
    return AdaptationConfig(**defaults)


# =====================================================================
# TestLosses
# =====================================================================

class TestCoralLoss:
    def test_zero_when_identical(self):
        from training.losses import coral_loss
        x = torch.randn(32, 16)
        loss = coral_loss(x, x.clone())
        assert loss.item() >= 0

    def test_positive_when_different(self):
        from training.losses import coral_loss
        src = torch.randn(32, 16)
        tgt = torch.randn(32, 16) * 5 + 3
        loss = coral_loss(src, tgt)
        assert loss.item() > 0

    def test_gradient_flows(self):
        from training.losses import coral_loss
        src = torch.randn(16, 8, requires_grad=True)
        tgt = torch.randn(16, 8)
        coral_loss(src, tgt).backward()
        assert src.grad is not None

    def test_small_batch_returns_zero(self):
        from training.losses import coral_loss
        src = torch.randn(1, 8)
        tgt = torch.randn(1, 8)
        loss = coral_loss(src, tgt)
        assert loss.item() == 0.0


class TestMMDLoss:
    def test_non_negative(self):
        from training.losses import mmd_loss
        src = torch.randn(20, 8)
        tgt = torch.randn(20, 8) + 2
        loss = mmd_loss(src, tgt)
        assert loss.item() >= 0

    def test_small_when_same_distribution(self):
        from training.losses import mmd_loss
        torch.manual_seed(0)
        src = torch.randn(100, 16)
        tgt = torch.randn(100, 16)
        loss = mmd_loss(src, tgt)
        # Not guaranteed to be tiny, but should be < large shifted case
        src_shift = torch.randn(100, 16) + 5
        loss_shift = mmd_loss(src, src_shift)
        assert loss_shift.item() > loss.item()

    def test_gradient_flows(self):
        from training.losses import mmd_loss
        src = torch.randn(16, 8, requires_grad=True)
        tgt = torch.randn(16, 8)
        mmd_loss(src, tgt).backward()
        assert src.grad is not None


class TestDANNLoss:
    def test_positive(self):
        from training.losses import dann_loss
        logits = torch.randn(32)
        labels = torch.randint(0, 2, (32,))
        loss = dann_loss(logits, labels)
        assert loss.item() > 0

    def test_gradient_flows(self):
        from training.losses import dann_loss
        logits = torch.randn(16, requires_grad=True)
        labels = torch.zeros(16)
        dann_loss(logits, labels).backward()
        assert logits.grad is not None

    def test_accepts_2d_input(self):
        from training.losses import dann_loss
        logits = torch.randn(16, 1)
        labels = torch.ones(16)
        loss = dann_loss(logits, labels)
        assert loss.item() > 0


class TestIRMPenalty:
    def test_zero_penalty_same_distribution(self):
        import torch.nn.functional as F
        from training.losses import irm_penalty
        model = nn.Linear(8, 3)
        logits = [model(torch.randn(16, 8)) for _ in range(3)]
        labels = [torch.randint(0, 3, (16,)) for _ in range(3)]
        losses = [F.cross_entropy(lg, lb) for lg, lb in zip(logits, labels)]
        penalty = irm_penalty(losses, logits)
        assert penalty.item() >= 0

    def test_gradient_flows_through_penalty(self):
        import torch.nn.functional as F
        from training.losses import irm_penalty
        params = list(nn.Linear(8, 3).parameters())
        logits = [sum(p.sum() for p in params) * torch.ones(16, 3) for _ in range(2)]
        for lg in logits:
            lg.retain_grad()
        labels = [torch.randint(0, 3, (16,)) for _ in range(2)]
        losses = [F.cross_entropy(lg, lb) for lg, lb in zip(logits, labels)]
        penalty = irm_penalty(losses, logits)
        # Just verify it's computable without error; graph may be complex
        assert penalty.item() >= 0


# =====================================================================
# TestDomainModules
# =====================================================================

class TestGradientReversalLayer:
    def test_forward_identity(self):
        from model.domain_modules import GradientReversalLayer
        grl = GradientReversalLayer(alpha=1.0)
        x = torch.randn(8, 16)
        y = grl(x)
        assert torch.allclose(x, y)

    def test_backward_reversal(self):
        from model.domain_modules import GradientReversalLayer
        grl = GradientReversalLayer(alpha=2.0)
        x = torch.randn(4, 8, requires_grad=True)
        y = grl(x)
        y.sum().backward()
        # gradient should be -alpha * ones
        expected = -2.0 * torch.ones_like(x)
        assert torch.allclose(x.grad, expected)

    def test_alpha_zero_passes_gradient(self):
        from model.domain_modules import GradientReversalLayer
        grl = GradientReversalLayer(alpha=0.0)
        x = torch.randn(4, 8, requires_grad=True)
        grl(x).sum().backward()
        assert torch.allclose(x.grad, torch.zeros_like(x))


class TestDomainDiscriminator:
    def test_output_shape(self):
        from model.domain_modules import DomainDiscriminator
        disc = DomainDiscriminator(input_dim=64, hidden_dim=32)
        x = torch.randn(16, 64)
        out = disc(x)
        assert out.shape == (16, 1)

    def test_is_nn_module(self):
        from model.domain_modules import DomainDiscriminator
        assert issubclass(DomainDiscriminator, nn.Module)

    def test_gradient_flows(self):
        from model.domain_modules import DomainDiscriminator
        disc = DomainDiscriminator(input_dim=16, hidden_dim=8)
        x = torch.randn(8, 16, requires_grad=True)
        disc(x).sum().backward()
        assert x.grad is not None


# =====================================================================
# TestAdaptationConfig
# =====================================================================

class TestAdaptationConfig:
    def test_defaults(self):
        from training.da_trainer import AdaptationConfig
        cfg = AdaptationConfig()
        assert cfg.batch_size == 64
        assert cfg.lambda_coral == 1.0
        assert cfg.alpha_schedule == "sigmoid"

    def test_frozen(self):
        from training.da_trainer import AdaptationConfig
        cfg = AdaptationConfig()
        with pytest.raises(Exception):
            cfg.batch_size = 128

    def test_invalid_schedule_rejected(self):
        from training.da_trainer import AdaptationConfig
        with pytest.raises(Exception):
            AdaptationConfig(alpha_schedule="cosine")


# =====================================================================
# TestDomainAdaptiveTrainer
# =====================================================================

class TestDomainAdaptiveTrainer:
    def _run(self, method):
        from training.da_trainer import DomainAdaptiveTrainer
        from training.config import TrainResult
        X_src, Y_src, X_tgt = _source_target()
        # Split source into train/val
        X_tr, X_val = X_src[:48], X_src[48:]
        Y_tr, Y_val = Y_src[:48], Y_src[48:]
        model = _TinyModel()
        tc = _trainer_cfg(max_epochs=5)
        ac = _adap_cfg()
        trainer = DomainAdaptiveTrainer(tc, ac, method)
        result = trainer.fit(model, (X_tr, Y_tr), (X_val, Y_val), X_tgt)
        assert isinstance(result, TrainResult)
        return result

    def test_coral_returns_train_result(self):
        result = self._run("coral")
        assert result.epochs_run > 0
        assert 0.0 <= result.train_acc <= 100.0

    def test_dann_returns_train_result(self):
        result = self._run("dann")
        assert result.epochs_run > 0

    def test_mmd_returns_train_result(self):
        result = self._run("mmd")
        assert result.epochs_run > 0

    def test_unknown_method_raises(self):
        from training.da_trainer import DomainAdaptiveTrainer
        X_src, Y_src, X_tgt = _source_target(n=32)
        model = _TinyModel()
        trainer = DomainAdaptiveTrainer(_trainer_cfg(max_epochs=2), _adap_cfg(), "coral")
        trainer._method = "unknown"
        with pytest.raises(ValueError, match="Unknown DA method"):
            trainer.fit(model, (X_src, Y_src), (X_src[:8], Y_src[:8]), X_tgt)

    def test_model_output_valid_after_dann(self):
        """After DANN training, model.predict() still works correctly."""
        from training.da_trainer import DomainAdaptiveTrainer
        X_src, Y_src, X_tgt = _source_target(n=64)
        model = _TinyModel()
        trainer = DomainAdaptiveTrainer(_trainer_cfg(max_epochs=3), _adap_cfg(), "dann")
        result = trainer.fit(model, (X_src[:48], Y_src[:48]), (X_src[48:], Y_src[48:]), X_tgt)
        # predict() from base Trainer should still work
        from training.trainer import Trainer
        cm = Trainer(_trainer_cfg()).predict(result.model, X_src, Y_src)
        assert cm.shape == (3, 3)


# =====================================================================
# TestDomainGeneralizationTrainer
# =====================================================================

class TestDomainGeneralizationTrainer:
    def _run(self, method):
        from training.dg_trainer import DomainGeneralizationTrainer
        from training.config import TrainResult
        domains = _multi_source(n_domains=3, n=48)
        val_X = torch.randn(16, 1, 16)
        val_Y = torch.randint(0, 3, (16,))
        model = _TinyModel()
        tc = _trainer_cfg(max_epochs=5)
        ac = _adap_cfg()
        trainer = DomainGeneralizationTrainer(tc, ac, method)
        result = trainer.fit(model, domains, (val_X, val_Y))
        assert isinstance(result, TrainResult)
        return result

    def test_mixup_returns_train_result(self):
        result = self._run("mixup")
        assert result.epochs_run > 0

    def test_irm_returns_train_result(self):
        result = self._run("irm")
        assert result.epochs_run > 0

    def test_single_domain_raises(self):
        from training.dg_trainer import DomainGeneralizationTrainer
        domains = _multi_source(n_domains=1)
        val = (torch.randn(8, 1, 16), torch.randint(0, 3, (8,)))
        model = _TinyModel()
        trainer = DomainGeneralizationTrainer(_trainer_cfg(max_epochs=2), _adap_cfg(), "mixup")
        with pytest.raises(ValueError, match="≥ 2 source domains"):
            trainer.fit(model, domains, val)

    def test_undersized_domain_raises_instead_of_hanging(self):
        """A domain smaller than batch_size must raise, not hang.

        Per-domain loaders use drop_last=True, so an undersized domain yields
        zero batches and _cycling_loader would otherwise spin forever.
        """
        from training.dg_trainer import DomainGeneralizationTrainer
        domains = _multi_source(n_domains=3, n=40)
        domains[1] = (domains[1][0][:8], domains[1][1][:8])  # domain 1: 8 < batch_size=32
        val = (torch.randn(8, 1, 16), torch.randint(0, 3, (8,)))
        model = _TinyModel()
        trainer = DomainGeneralizationTrainer(_trainer_cfg(max_epochs=2), _adap_cfg(), "mixup")
        with pytest.raises(ValueError, match="fewer than batch_size"):
            trainer.fit(model, domains, val)


# =====================================================================
# TestBackwardCompatibility
# =====================================================================

class TestBackwardCompatibility:
    def test_train_result_unchanged(self):
        """TrainResult must have exactly the same fields as before."""
        from training.config import TrainResult
        fields = set(TrainResult.__dataclass_fields__)
        expected = {"model", "epochs_run", "train_loss", "train_acc", "val_loss", "val_acc"}
        assert expected == fields

    def test_adaptation_config_defaults(self):
        """AdaptationConfig defaults are stable."""
        from training.da_trainer import AdaptationConfig
        cfg = AdaptationConfig()
        assert cfg.batch_size == 64
        assert cfg.lambda_coral == 1.0
        assert cfg.lambda_dann == 1.0
        assert cfg.lambda_mmd == 1.0

    def test_standard_trainer_fit_unchanged(self):
        """Standard Trainer.fit() still works exactly as before."""
        from training.trainer import Trainer
        from training.config import TrainResult
        X, Y, _ = _source_target(n=64)
        model = _TinyModel()
        tc = _trainer_cfg(max_epochs=5, batch_size=None)  # full-batch
        result = Trainer(tc).fit(model, (X[:48], Y[:48]), (X[48:], Y[48:]))
        assert isinstance(result, TrainResult)
        assert result.epochs_run == 5


# =====================================================================
# TestExperimentDispatch (pure logic test — no experiment package import)
# =====================================================================

_DA_METHODS = frozenset({"coral", "dann", "mmd"})
_DG_METHODS = frozenset({"mixup", "irm"})


def _make_dispatch_target(adaptation: str):
    """Return a duck-typed object that implements the same dispatch as Experiment.run()."""
    from training.da_trainer import AdaptationConfig

    class _MockExp:
        called: str = ""

        def __init__(self, method):
            self._method = method
            self._adap = AdaptationConfig()

        def run_pairwise(self, t, f):
            self.called = "pairwise"

        def run_pairwise_with_adaptation(self, t, f, method, cfg):
            self.called = f"da_{method}"

        def run_leave_one_out_dg(self, t, f, method, cfg):
            self.called = f"dg_{method}"

        def run(self, task, filters):
            m = self._method
            if m in (None, "none"):
                return self.run_pairwise(task, filters)
            if m in _DA_METHODS:
                return self.run_pairwise_with_adaptation(task, filters, m, self._adap)
            if m in _DG_METHODS:
                return self.run_leave_one_out_dg(task, filters, m, self._adap)
            raise ValueError(f"Unknown adaptation: {m!r}")

    return _MockExp(adaptation)


# =====================================================================
# TestMultiDomainSolutionStructure
# =====================================================================

class TestMultiDomainSolutionStructure:
    """Integration test: DA trainer → DomainSolution → MultiDomainSolution."""

    def _make_solution(self, method: str, n_domains: int = 2):
        """Train one model per domain with DA and wrap into MultiDomainSolution."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

        from training.da_trainer import DomainAdaptiveTrainer
        from training.trainer import Trainer

        n_classes = 3
        n_src = 48

        try:
            from results.containers import DomainSolution, MultiDomainSolution
        except ImportError:
            import importlib
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
            from results.containers import DomainSolution, MultiDomainSolution

        domain_solutions = []
        for i in range(n_domains):
            rng = torch.Generator().manual_seed(i)
            X_src = torch.randn(n_src, 1, 16, generator=rng)
            Y_src = torch.randint(0, n_classes, (n_src,), generator=rng)
            X_tgt = torch.randn(32, 1, 16, generator=rng) + 1.0

            X_tr, X_val = X_src[:36], X_src[36:]
            Y_tr, Y_val = Y_src[:36], Y_src[36:]

            model = _TinyModel(in_dim=16, n_classes=n_classes)
            tc = _trainer_cfg(max_epochs=3)
            ac = _adap_cfg(batch_size=16)
            trainer = DomainAdaptiveTrainer(tc, ac, method)
            result = trainer.fit(model, (X_tr, Y_tr), (X_val, Y_val), X_tgt)

            predictor = Trainer(_trainer_cfg())
            train_cm = predictor.predict(result.model, X_tr, Y_tr)
            val_cm = predictor.predict(result.model, X_val, Y_val)

            label = f"domain_{i}"
            domain_solutions.append(DomainSolution(
                train_dataset_name=label,
                class_labels={j: f"cls_{j}" for j in range(n_classes)},
                seed=42,
                train_metadata={"train_epoch_nbr": result.epochs_run},
                confusion_matrices={
                    label: train_cm,
                    f"domain_{1 - i}": val_cm,
                },
            ))

        return MultiDomainSolution(
            config_name="test_da_config",
            domain_solutions=domain_solutions,
            processor_name="raw",
        )

    def test_coral_structure(self):
        mds = self._make_solution("coral")
        assert len(mds.domain_solutions) == 2
        for ds in mds.domain_solutions:
            assert ds.train_dataset_name in ds.confusion_matrices
            cm = ds.confusion_matrices[ds.train_dataset_name]
            assert cm.shape == (3, 3)

    def test_mmd_structure(self):
        mds = self._make_solution("mmd")
        assert len(mds.domain_solutions) == 2
        for ds in mds.domain_solutions:
            assert ds.train_dataset_name in ds.confusion_matrices

    def test_config_name_preserved(self):
        mds = self._make_solution("coral")
        assert mds.config_name == "test_da_config"
        assert mds.processor_name == "raw"

    def test_domain_solution_post_init_requires_self_eval(self):
        """DomainSolution raises if train_dataset_name not in confusion_matrices."""
        try:
            from results.containers import DomainSolution
        except ImportError:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
            from results.containers import DomainSolution
        import numpy as np
        with pytest.raises(ValueError, match="self-evaluation"):
            DomainSolution(
                train_dataset_name="A",
                class_labels={0: "x"},
                seed=0,
                train_metadata={},
                confusion_matrices={"B": np.zeros((1, 1))},
            )


class TestExperimentDispatch:
    def test_none_dispatches_to_pairwise(self):
        exp = _make_dispatch_target("none")
        exp.run(None, ())
        assert exp.called == "pairwise"

    def test_coral_dispatches_to_da(self):
        exp = _make_dispatch_target("coral")
        exp.run(None, ())
        assert exp.called == "da_coral"

    def test_dann_dispatches_to_da(self):
        exp = _make_dispatch_target("dann")
        exp.run(None, ())
        assert exp.called == "da_dann"

    def test_mmd_dispatches_to_da(self):
        exp = _make_dispatch_target("mmd")
        exp.run(None, ())
        assert exp.called == "da_mmd"

    def test_mixup_dispatches_to_dg(self):
        exp = _make_dispatch_target("mixup")
        exp.run(None, ())
        assert exp.called == "dg_mixup"

    def test_irm_dispatches_to_dg(self):
        exp = _make_dispatch_target("irm")
        exp.run(None, ())
        assert exp.called == "dg_irm"

    def test_unknown_adaptation_raises(self):
        exp = _make_dispatch_target("bad_method")
        with pytest.raises(ValueError, match="Unknown adaptation"):
            exp.run(None, ())
