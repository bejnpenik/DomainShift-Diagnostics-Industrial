from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import numpy.typing as npt

from training.early_stopping import EarlyStopper
from training.config import TrainerConfig, TrainResult



class Trainer:
    """Train and evaluate classification models.

    Args:
        max_epochs: Maximum training epochs.
        optimizer_name: 'adamw' or 'sgd'.
        lr: Learning rate.
        weight_decay: Weight decay.
        momentum: Momentum for SGD.
        device: 'cuda' or 'cpu'.
        early_stopping: (patience, min_delta) or None.
        min_epochs: Minimum epochs before early stopping activates.
        noise: (noise_prob, noise_std) or None for no noise.
        verbose_level: Print every N epochs. 0 for silent.
    """

    def __init__(
        self,
        config: TrainerConfig
    ) -> None:
        self._config = config

    def _create_optimizer(self, params) -> optim.Optimizer:
        cfg = self._config
        if cfg.optimizer_name == "adamw":
            return optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer_name == "sgd":
            return optim.SGD(
                params, lr=cfg.lr, momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
            )
        raise ValueError(f"Unknown optimizer: {self._optimizer_name}")

    def _inject_noise(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self._config
        if cfg.noise is None:
            return x
        prob, std = cfg.noise
        if prob <= 0:
            return x
        mask = torch.rand(x.shape[0], device=x.device) < prob
        noise = torch.randn_like(x) * std
        x_noisy = x.clone()
        x_noisy[mask] = x[mask] + noise[mask]
        return x_noisy

    def _validate(
        self, model: nn.Module, val_data: tuple[torch.Tensor, torch.Tensor],
        criterion: nn.Module,
    ) -> tuple[float, float]:
        cfg = self._config
        model.eval()
        x, y = val_data[0].to(cfg.device), val_data[1].to(cfg.device)
        with torch.no_grad():
            out = model(x)
            loss = criterion(out, y)
            correct = out.max(1)[1].eq(y).sum().item()
        return loss.item(), 100 * correct / y.size(0)

    def fit(
        self,
        model: nn.Module,
        train_data: tuple[torch.Tensor, torch.Tensor],
        val_data: tuple[torch.Tensor, torch.Tensor],
    ) -> TrainResult:
        """Train the model.

        Args:
            model: PyTorch model.
            train_data: (X_train, Y_train) tensors.
            val_data: (X_val, Y_val) tensors.

        Returns:
            TrainResult with trained model and final metrics.
        """
        cfg = self._config
        model = model.to(cfg.device)
        optimizer = self._create_optimizer(model.parameters())
        criterion = nn.CrossEntropyLoss()

        x, y = train_data[0].to(cfg.device), train_data[1].to(cfg.device)

        stopper = None
        if cfg.early_stopping:
            stopper = EarlyStopper(cfg.early_stopping[0], cfg.early_stopping[1])

        verbosity = -1
        train_loss = train_acc = val_loss = val_acc = 0.0
        epochs_run = 0

        for epoch in range(cfg.max_epochs):
            model.train()
            optimizer.zero_grad()

            xt = self._inject_noise(x)
            out = model(xt)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss = loss.item()
            correct = out.max(1)[1].eq(y).sum().item()
            train_acc = 100 * correct / y.size(0)

            val_loss, val_acc = self._validate(model, val_data, criterion)
            epochs_run = epoch + 1

            if cfg.verbose_level > 0 and epoch // cfg.verbose_level > verbosity:
                print(
                    f"Epoch {epochs_run:02d} | "
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

    def predict(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> npt.NDArray:
        """Predict and return confusion matrix.

        Args:
            model: Trained model.
            x: Input tensor.
            y: True labels.

        Returns:
            Confusion matrix as numpy array.
        """
        cfg = self._config
        model.to(cfg.device)
        model.eval()
        with torch.no_grad():
            x, y = x.to(cfg.device), y.to(cfg.device)
            preds = model(x).max(1)[1]
        return confusion_matrix(y.cpu().numpy(), preds.cpu().numpy())