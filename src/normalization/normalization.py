from __future__ import annotations

import torch

class Normalisator:
    def __init__(
        self,
        mode:str = "sample",
        mean:torch.Tensor|None = None,
        std:torch.Tensor|None = None,
        eps:float = 1e-8,
    ):
        self.mode = mode
        self.eps = eps
        self.mean = mean
        self.std = std
        self._fitted = False

        if mode == "pretrained":
            if mean is None or std is None:
                raise ValueError("Pretrained normalization requires mean/std")
            self._fitted = True 

        if mode == "sample" and (mean is not None or std is not None):
            raise ValueError("Sample normalization does not use mean/std")

    def fit(self, x: torch.Tensor):
        if self.mode != "dataset":
            return self
        if self._fitted:
            raise RuntimeError("Normaliser already fitted, cannot refit")
        self.std, self.mean = torch.std_mean(x, dim=0, keepdim=True)
        self._fitted = True
        return self

    def __call__(self, x: torch.Tensor)->torch.Tensor:
        if self.mode == "sample":
            std, mean = torch.std_mean(x, dim=tuple(range(1, x.ndim)), keepdim=True)
            return (x - mean) / (std + self.eps)

        if self.mode == "dataset":
            if not self._fitted:
                raise RuntimeError("Normaliser must be fitted for dataset normalisation!")
            return (x - self.mean) / (self.std + self.eps)

        if self.mode == "pretrained":
            return (x - self.mean) / (self.std + self.eps)

        raise ValueError(f"Unknown normalization mode {self.mode}")