from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from sklearn.metrics import confusion_matrix
import numpy.typing as npt

from .early_stopping import EarlyStopper
from .config import TrainerConfig, TrainResult


class _FullBatchIter:
    """Wrap pre-loaded GPU tensors as a single-batch iterable."""
    def __init__(self, x: torch.Tensor, y: torch.Tensor, aux: torch.Tensor | None = None) -> None:
        self._batch = (x, y, aux) if aux is not None else (x, y)

    def __iter__(self):
        yield self._batch



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
        raise ValueError(f"Unknown optimizer: {cfg.optimizer_name}")

    def _inject_noise(self, x: torch.Tensor, epoch: int = 0, step: int = 0) -> torch.Tensor:
        cfg = self._config
        if cfg.noise is None:
            return x
        prob, std = cfg.noise
        if prob <= 0:
            return x
        seed = cfg.random_seed ^ (epoch * 100_003 + step)
        gen = torch.Generator(device=x.device).manual_seed(seed & 0xFFFF_FFFF)
        mask = torch.rand(x.shape[0], generator=gen, device=x.device) < prob
        noise = torch.randn(x.shape, generator=gen, device=x.device) * std
        x_noisy = x.clone()
        x_noisy[mask] = x[mask] + noise[mask]
        return x_noisy

    def _handle_early_stopping(
        self,
        stopper: EarlyStopper | None,
        val_loss: float,
        model: nn.Module,
        epoch: int,
        epochs_run: int,
    ) -> bool:
        """Check early stopping and restore the best checkpoint if triggered.

        Shared by Trainer, DomainAdaptiveTrainer, and DomainGeneralizationTrainer
        so the "no best checkpoint was ever recorded" case (e.g. val_loss was
        NaN/Inf for the whole run) is handled once instead of independently in
        three fit() loops.

        Returns:
            True if the caller should break out of the training loop.
        """
        cfg = self._config
        if not stopper or epoch < cfg.min_epochs:
            return False
        if not stopper.step(val_loss, model):
            return False
        if cfg.verbose_level > 0:
            print(f"Early stopping at epoch {epochs_run}")
        best_state = stopper.best_state()
        if best_state is not None:
            model.load_state_dict(best_state)
        elif cfg.verbose_level > 0:
            print(
                "Warning: no improved checkpoint was ever recorded (validation "
                "loss was never finite) — keeping current model weights instead "
                "of restoring a best checkpoint."
            )
        return True

    def _validate(
        self, model: nn.Module, val_data: tuple,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        cfg = self._config
        model.eval()
        x = val_data[0].to(cfg.device)
        y = val_data[1].to(cfg.device)
        aux = val_data[2].to(cfg.device) if len(val_data) > 2 and val_data[2] is not None else None
        with torch.no_grad():
            out = model(x) if aux is None else model(x, aux)
            loss = criterion(out, y)
            correct = out.max(1)[1].eq(y).sum().item()
        return loss.item(), 100 * correct / y.size(0)

    def fit(
        self,
        model: nn.Module,
        train_data: tuple,
        val_data: tuple,
    ) -> TrainResult:
        """Train the model.

        Args:
            model: PyTorch model.
            train_data: (X_train, Y_train) or (X_train, Y_train, aux_train) tensors.
            val_data: (X_val, Y_val) or (X_val, Y_val, aux_val) tensors.

        Returns:
            TrainResult with trained model and final metrics.
        """
        cfg = self._config
        model = model.to(cfg.device)
        optimizer = self._create_optimizer(model.parameters())
        criterion = nn.CrossEntropyLoss()
        stopper = EarlyStopper(*cfg.early_stopping) if cfg.early_stopping else None

        aux_train = train_data[2] if len(train_data) > 2 else None

        if cfg.batch_size is None:
            x = train_data[0].to(cfg.device)
            y = train_data[1].to(cfg.device)
            aux = aux_train.to(cfg.device) if aux_train is not None else None
            data_iter = _FullBatchIter(x, y, aux)
        else:
            tensors = [train_data[0], train_data[1]]
            if aux_train is not None:
                tensors.append(aux_train)
            dataset = torch.utils.data.TensorDataset(*tensors)
            data_iter = torch.utils.data.DataLoader(
                dataset, batch_size=cfg.batch_size, shuffle=True
            )

        verbosity = -1
        train_loss = train_acc = val_loss = val_acc = 0.0
        epochs_run = 0

        for epoch in range(cfg.max_epochs):
            model.train()
            total_loss = total_correct = total_n = 0

            for step, batch in enumerate(data_iter):
                xb = batch[0].to(cfg.device)
                yb = batch[1].to(cfg.device)
                auxb = batch[2].to(cfg.device) if len(batch) > 2 else None
                optimizer.zero_grad()
                xb = self._inject_noise(xb, epoch, step)
                out = model(xb) if auxb is None else model(xb, auxb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * yb.size(0)
                total_correct += out.max(1)[1].eq(yb).sum().item()
                total_n += yb.size(0)

            train_loss = total_loss / total_n
            train_acc = 100 * total_correct / total_n
            val_loss, val_acc = self._validate(model, val_data, criterion)
            epochs_run = epoch + 1

            if cfg.verbose_level > 0 and epoch // cfg.verbose_level > verbosity:
                print(
                    f"Epoch {epochs_run:04d} | "
                    f"train loss {train_loss:.5f} | train acc {train_acc:.2f}% | "
                    f"val loss {val_loss:.5f} | val acc {val_acc:.2f}%"
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

    def predict(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        aux: torch.Tensor | None = None,
    ) -> npt.NDArray:
        """Predict and return confusion matrix.

        Args:
            model: Trained model.
            x: Input tensor.
            y: True labels.
            aux: Optional conditioning tensor.

        Returns:
            Confusion matrix as numpy array.
        """
        cfg = self._config
        model.to(cfg.device)
        model.eval()
        with torch.no_grad():
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            if aux is not None:
                aux = aux.to(cfg.device)
            preds = (model(x) if aux is None else model(x, aux)).max(1)[1]
        return confusion_matrix(y.cpu().numpy(), preds.cpu().numpy())