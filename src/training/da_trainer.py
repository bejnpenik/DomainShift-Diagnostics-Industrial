"""
Domain Adaptive Trainer — source + unlabeled target co-training.

Supports three distribution alignment methods:
    coral  -- Correlation Alignment: minimise covariance distance
    dann   -- Domain-Adversarial NN:  gradient reversal + domain discriminator
    mmd    -- Maximum Mean Discrepancy: kernel mean embedding distance

Usage::

    cfg = AdaptationConfig(lambda_coral=1.0, batch_size=64)
    trainer = DomainAdaptiveTrainer(trainer_config, cfg, method="coral")
    result = trainer.fit(model, source_train, source_val, target_x)

The returned TrainResult is identical to the standard Trainer output so all
downstream result containers work unchanged.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.utils.data
from pydantic import BaseModel, ConfigDict, Field

from .trainer import Trainer
from .config import TrainerConfig, TrainResult
from .early_stopping import EarlyStopper
from .losses import coral_loss, mmd_loss, dann_loss
# GradientReversalLayer and DomainDiscriminator are imported lazily inside
# the DANN branch of fit() to avoid the cross-package relative import
# (..model) failing when src/ is the top-level directory on sys.path.


# =====================================================================
# AdaptationConfig
# =====================================================================

class AdaptationConfig(BaseModel):
    """Hyperparameters shared by all domain adaptation / generalization methods.

    Fields used by method:
        coral  -- lambda_coral
        dann   -- lambda_dann, alpha_schedule, disc_hidden_dim
        mmd    -- lambda_mmd
        mixup  -- mixup_alpha
        irm    -- irm_lambda
        all    -- batch_size (mandatory for DA/DG; overrides trainer batch_size)
    """
    model_config = ConfigDict(frozen=True)

    # Loss weights
    lambda_coral: float = Field(default=1.0, ge=0.0)
    lambda_dann: float = Field(default=1.0, ge=0.0)
    lambda_mmd: float = Field(default=1.0, ge=0.0)

    # DANN-specific
    alpha_schedule: Literal["sigmoid", "linear", "constant"] = "sigmoid"
    disc_hidden_dim: int = Field(default=256, gt=0)
    disc_lr: float = Field(default=3e-4, gt=0)

    # DG-specific
    mixup_alpha: float = Field(default=0.2, gt=0.0)
    irm_lambda: float = Field(default=1.0, ge=0.0)

    # Mini-batch size (required for DA/DG)
    batch_size: int = Field(default=64, gt=0)


# =====================================================================
# DomainAdaptiveTrainer
# =====================================================================

class DomainAdaptiveTrainer(Trainer):
    """Train a model using labeled source data + unlabeled target features.

    Inherits Trainer._create_optimizer, _inject_noise, _validate, predict.
    Overrides fit() to add the adaptation loss.

    Args:
        trainer_config: Standard TrainerConfig (batch_size is overridden by
                        AdaptationConfig.batch_size for mini-batch iteration).
        adaptation_config: DA hyperparameters.
        method: One of 'coral', 'dann', 'mmd'.
    """

    def __init__(
        self,
        trainer_config: TrainerConfig,
        adaptation_config: AdaptationConfig,
        method: Literal["coral", "dann", "mmd"],
    ) -> None:
        super().__init__(trainer_config)
        self._adap_cfg = adaptation_config
        self._method = method

    def _alpha(self, epoch: int, max_epochs: int) -> float:
        """Compute DANN reversal strength alpha for current epoch."""
        cfg = self._adap_cfg
        p = epoch / max(max_epochs, 1)
        if cfg.alpha_schedule == "sigmoid":
            return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
        elif cfg.alpha_schedule == "linear":
            return p
        else:  # constant
            return 1.0

    def fit(
        self,
        model: nn.Module,
        source_train: tuple,
        source_val: tuple,
        target_x: torch.Tensor,
    ) -> TrainResult:
        """Train with source labels + target distribution alignment.

        Args:
            model: BuiltModel with a .features() method and .head attribute.
            source_train: (X, Y) or (X, Y, aux) labeled source tensors.
            source_val:   (X, Y) or (X, Y, aux) for validation (source only).
            target_x:     (N_t, ...) unlabeled target signal tensor.

        Returns:
            TrainResult identical in structure to standard Trainer.fit().
        """
        cfg = self._config
        adap = self._adap_cfg
        batch_size = adap.batch_size

        model = model.to(cfg.device)
        optimizer = self._create_optimizer(model.parameters())
        criterion = nn.CrossEntropyLoss()
        stopper = EarlyStopper(*cfg.early_stopping) if cfg.early_stopping else None

        # GRL is created eagerly (no feat_dim needed); discriminator is lazy (feat_dim unknown)
        if self._method == "dann":
            try:
                from model.domain_modules import GradientReversalLayer, DomainDiscriminator
            except ImportError:
                from model.domain_modules import GradientReversalLayer, DomainDiscriminator
            grl: GradientReversalLayer | None = GradientReversalLayer()
        else:
            grl = None
        discriminator = None
        disc_optimizer = None

        # Build source DataLoader (drop_last=False to use all samples)
        src_tensors = [source_train[0], source_train[1]]
        if len(source_train) > 2 and source_train[2] is not None:
            src_tensors.append(source_train[2])
        src_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*src_tensors),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Build target DataLoader; restarted (reshuffled) whenever source outruns it
        tgt_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(target_x),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        verbosity = -1
        train_loss = train_acc = val_loss = val_acc = 0.0
        epochs_run = 0

        for epoch in range(cfg.max_epochs):
            model.train()
            total_loss = total_correct = total_n = 0

            src_iter = iter(src_loader)
            tgt_iter = iter(tgt_loader)  # fresh shuffle each epoch
            alpha = self._alpha(epoch, cfg.max_epochs)
            if grl is not None:
                grl.alpha = alpha  # set once per epoch, not per batch

            step = 0
            while True:
                try:
                    src_batch = next(src_iter)
                except StopIteration:
                    break
                try:
                    tgt_batch = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_loader)  # restart with a new shuffle
                    tgt_batch = next(tgt_iter)

                xb = src_batch[0].to(cfg.device)
                yb = src_batch[1].to(cfg.device)
                x_tgt = tgt_batch[0].to(cfg.device)
                xb = self._inject_noise(xb, epoch, step)
                step += 1

                # Feature extraction (shared encoder + aggregator)
                feat_src = model.features(xb)       # (N_s, D)
                feat_tgt = model.features(x_tgt)    # (N_t, D)

                # Classification loss (source only)
                logits = model.head(feat_src)        # (N_s, C)
                cls_loss = criterion(logits, yb)

                # Adaptation loss
                if self._method == "coral":
                    adap_loss = coral_loss(feat_src, feat_tgt)
                    loss = cls_loss + adap.lambda_coral * adap_loss

                elif self._method == "mmd":
                    adap_loss = mmd_loss(feat_src, feat_tgt)
                    loss = cls_loss + adap.lambda_mmd * adap_loss

                elif self._method == "dann":
                    # Lazy init of discriminator on first batch (feat_dim unknown until here)
                    if discriminator is None:
                        feat_dim = feat_src.shape[1]
                        discriminator = DomainDiscriminator(
                            feat_dim, hidden_dim=adap.disc_hidden_dim
                        ).to(cfg.device)
                        disc_optimizer = torch.optim.Adam(
                            discriminator.parameters(), lr=adap.disc_lr
                        )

                    discriminator.train()
                    n_src = feat_src.shape[0]
                    n_tgt = feat_tgt.shape[0]

                    dom_logits = discriminator(
                        torch.cat([grl(feat_src), grl(feat_tgt)], dim=0)
                    )
                    dom_labels = torch.cat([
                        torch.zeros(n_src, device=cfg.device),
                        torch.ones(n_tgt, device=cfg.device),
                    ])
                    adap_loss = dann_loss(dom_logits, dom_labels)
                    loss = cls_loss + adap.lambda_dann * adap_loss
                else:
                    raise ValueError(f"Unknown DA method: '{self._method}'")

                optimizer.zero_grad()
                if disc_optimizer is not None:
                    disc_optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if disc_optimizer is not None:
                    disc_optimizer.step()

                total_loss += loss.item() * yb.size(0)
                total_correct += logits.max(1)[1].eq(yb).sum().item()
                total_n += yb.size(0)

            if total_n == 0:
                break

            train_loss = total_loss / total_n
            train_acc = 100.0 * total_correct / total_n
            val_loss, val_acc = self._validate(model, source_val, criterion)
            epochs_run = epoch + 1

            if cfg.verbose_level > 0 and epoch // cfg.verbose_level > verbosity:
                print(
                    f"[DA-{self._method}] Epoch {epochs_run:04d} | "
                    f"train loss {train_loss:.5f} | train acc {train_acc:.2f}% | "
                    f"val loss {val_loss:.5f} | val acc {val_acc:.2f}% | α={alpha:.3f}"
                )
                verbosity = epoch // cfg.verbose_level

            if self._handle_early_stopping(stopper, val_loss, model, epoch, epochs_run):
                break

        return TrainResult(
            model=model,
            epochs_run=epochs_run,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )
