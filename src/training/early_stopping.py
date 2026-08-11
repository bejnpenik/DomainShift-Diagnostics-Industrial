import math
import torch

class EarlyStopper:
    def __init__(self, patience:int=10, min_delta:float=0.0):
        self._patience = patience
        self._min_delta = min_delta
        self._best_loss = float('inf')
        self._counter = 0
        self._best_state = None

    def step(self, val_loss:float, model:torch.nn.Module):
        # NaN/Inf comparisons are always False, so `val_loss < self._best_loss`
        # would silently treat a diverged loss as "no improvement" without ever
        # explaining why — and if this happens before any real improvement is
        # ever recorded, _best_state stays None. Handle it explicitly so the
        # caller can tell "never improved" apart from "diverged".
        if not math.isfinite(val_loss):
            self._counter += 1
            return self._counter > self._patience
        if val_loss < self._best_loss - self._min_delta:
            self._best_loss = val_loss
            self._counter = 0
            self._best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            return False
        else:
            self._counter += 1
            return self._counter > self._patience

    def best_state(self):
        return self._best_state