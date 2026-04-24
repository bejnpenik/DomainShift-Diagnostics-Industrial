"""
Domain Generalization Trainer — multi-source training without target access.

Supports two strategies:
    mixup  -- Domain-level interpolation: mix samples from two source domains
    irm    -- Invariant Risk Minimization: penalise gradient variance across domains

Usage::

    cfg = AdaptationConfig(mixup_alpha=0.2, batch_size=64)
    trainer = DomainGeneralizationTrainer(trainer_config, cfg, method="mixup")
    result = trainer.fit(model, source_datasets, val_data)

``source_datasets`` is a list of ``(X, Y)`` tuples, one per source domain.
``val_data``        is a ``(X, Y)`` or ``(X, Y, aux)`` tuple from any held-in domain.

The returned TrainResult is identical to the standard Trainer output.
"""

from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from .trainer import Trainer
from .config import TrainerConfig, TrainResult
from .early_stopping import EarlyStopper
from .da_trainer import AdaptationConfig
from .losses import irm_penalty


class DomainGeneralizationTrainer(Trainer):
    """Train a model on multiple source domains without access to the target.

    Inherits Trainer._create_optimizer, _inject_noise, _validate, predict.
    Overrides fit() to implement Mixup or IRM across source domains.

    Args:
        trainer_config:    Standard TrainerConfig.
        adaptation_config: DG hyperparameters (mixup_alpha, irm_lambda, batch_size).
        method:            'mixup' or 'irm'.
    """

    def __init__(
        self,
        trainer_config: TrainerConfig,
        adaptation_config: AdaptationConfig,
        method: Literal["mixup", "irm"],
    ) -> None:
        super().__init__(trainer_config)
        self._adap_cfg = adaptation_config
        self._method = method

    def fit(
        self,
        model: nn.Module,
        source_datasets: list[tuple[torch.Tensor, torch.Tensor]],
        val_data: tuple,
    ) -> TrainResult:
        """Train across multiple source domains.

        Args:
            model:           BuiltModel.
            source_datasets: List of (X, Y) tuples, one per source domain.
                             All X must have the same shape except the batch dim.
            val_data:        (X, Y) or (X, Y, aux) validation tensors.

        Returns:
            TrainResult.
        """
        if len(source_datasets) < 2:
            raise ValueError(
                f"DomainGeneralizationTrainer needs ≥ 2 source domains, "
                f"got {len(source_datasets)}."
            )

        cfg = self._config
        adap = self._adap_cfg
        batch_size = adap.batch_size

        model = model.to(cfg.device)
        optimizer = self._create_optimizer(model.parameters())
        criterion = nn.CrossEntropyLoss()
        stopper = EarlyStopper(*cfg.early_stopping) if cfg.early_stopping else None

        # Build per-domain DataLoaders
        loaders = [
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X, Y),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )
            for (X, Y) in source_datasets
        ]

        verbosity = -1
        train_loss = train_acc = val_loss = val_acc = 0.0
        epochs_run = 0

        for epoch in range(cfg.max_epochs):
            model.train()
            total_loss = total_correct = total_n = 0

            iters = [iter(ld) for ld in loaders]
            n_domains = len(iters)

            while True:
                # Draw a batch from every domain; stop when any domain is exhausted
                try:
                    batches = [next(it) for it in iters]
                except StopIteration:
                    break

                if self._method == "mixup":
                    loss, correct, n = self._mixup_step(
                        model, batches, criterion, cfg.device, adap.mixup_alpha
                    )
                else:  # irm
                    loss, correct, n = self._irm_step(
                        model, batches, criterion, cfg.device, adap.irm_lambda
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * n
                total_correct += correct
                total_n += n

            if total_n == 0:
                break

            train_loss = total_loss / total_n
            train_acc = 100.0 * total_correct / total_n
            val_loss, val_acc = self._validate(model, val_data, criterion)
            epochs_run = epoch + 1

            if cfg.verbose_level > 0 and epoch // cfg.verbose_level > verbosity:
                print(
                    f"[DG-{self._method}] Epoch {epochs_run:04d} | "
                    f"train loss {train_loss:.5f} | train acc {train_acc:.2f}% | "
                    f"val loss {val_loss:.5f} | val acc {val_acc:.2f}%"
                )
                verbosity = epoch // cfg.verbose_level

            if stopper and epoch >= cfg.min_epochs:
                if stopper.step(val_loss, model):
                    if cfg.verbose_level > 0:
                        print(f"Early stopping at epoch {epochs_run}")
                    model.load_state_dict(stopper.best_state())
                    break

        return TrainResult(
            model=model,
            epochs_run=epochs_run,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )

    # ------------------------------------------------------------------
    # Private per-step helpers
    # ------------------------------------------------------------------

    def _mixup_step(
        self,
        model: nn.Module,
        batches: list[tuple],
        criterion: nn.Module,
        device: str,
        alpha: float,
    ):
        """One Mixup step: mix samples from two randomly chosen source domains."""
        n_domains = len(batches)
        i, j = random.sample(range(n_domains), 2)

        xa = self._inject_noise(batches[i][0].to(device))
        ya = batches[i][1].to(device)
        xb = self._inject_noise(batches[j][0].to(device))
        yb = batches[j][1].to(device)

        lam = float(np.random.beta(alpha, alpha))
        n = min(xa.shape[0], xb.shape[0])

        x_mix = lam * xa[:n] + (1.0 - lam) * xb[:n]
        logits = model(x_mix)

        # Soft label loss: λ·CE(logits, y_a) + (1-λ)·CE(logits, y_b)
        loss = lam * criterion(logits, ya[:n]) + (1.0 - lam) * criterion(logits, yb[:n])
        preds = logits.max(1)[1]
        # Accuracy counts a correct prediction if it matches either mixed label
        correct = (
            lam * preds.eq(ya[:n]).sum().item()
            + (1.0 - lam) * preds.eq(yb[:n]).sum().item()
        )
        return loss, correct, n

    def _irm_step(
        self,
        model: nn.Module,
        batches: list[tuple],
        criterion: nn.Module,
        device: str,
        irm_lambda: float,
    ):
        """One IRM step: per-domain ERM + invariance penalty."""
        per_domain_logits = []
        per_domain_labels = []
        per_domain_losses = []

        for batch in batches:
            xb = self._inject_noise(batch[0].to(device))
            yb = batch[1].to(device)
            logits = model(xb)
            domain_loss = criterion(logits, yb)
            per_domain_logits.append(logits)
            per_domain_labels.append(yb)
            per_domain_losses.append(domain_loss)

        erm_loss = torch.stack(per_domain_losses).mean()
        penalty = irm_penalty(per_domain_logits, per_domain_labels)
        loss = erm_loss + irm_lambda * penalty

        # Accuracy on combined batch
        all_logits = torch.cat(per_domain_logits, dim=0)
        all_labels = torch.cat(per_domain_labels, dim=0)
        correct = all_logits.max(1)[1].eq(all_labels).sum().item()
        n = all_labels.shape[0]
        return loss, correct, n
