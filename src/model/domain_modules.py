"""
Domain adaptation neural network modules.

GradientReversalLayer  -- autograd Function + nn.Module wrapper for DANN
DomainDiscriminator    -- small MLP that classifies source vs target features
"""

from __future__ import annotations

import torch
import torch.nn as nn


# =====================================================================
# Gradient Reversal Layer
# =====================================================================

class _GRL(torch.autograd.Function):
    """Gradient reversal: identity in forward, scaled negation in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps the GRL autograd function as an nn.Module.

    During forward pass: identity.
    During backward pass: multiplies gradient by -alpha.

    Args:
        alpha: Reversal strength. Set to 0 at start of training, ramp to 1
               using the sigmoidal schedule in DomainAdaptiveTrainer.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRL.apply(x, self.alpha)


# =====================================================================
# Domain Discriminator
# =====================================================================

class DomainDiscriminator(nn.Module):
    """Binary domain classifier for DANN.

    Three-layer MLP: input_dim → hidden_dim → hidden_dim → 1 (raw logit).
    Uses BatchNorm + ReLU on hidden layers, no activation on output.

    Args:
        input_dim: Dimensionality of feature vectors from the feature extractor.
        hidden_dim: Width of hidden layers. Default 256.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw domain logit (single value per sample).

        Args:
            x: (N, D) feature tensor.

        Returns:
            (N, 1) raw logit — 0 = source, 1 = target (after sigmoid).
        """
        return self.net(x)
