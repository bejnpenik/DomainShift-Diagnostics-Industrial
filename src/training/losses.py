"""
Domain adaptation and generalization loss functions.

All functions operate on batches of feature vectors (2D tensors) and return
scalar tensors with gradients attached for use in a training loop.

Functions:
    coral_loss      -- CORAL: covariance alignment between source and target
    mmd_loss        -- MMD:   kernel mean embedding distance
    dann_loss       -- DANN:  domain classification BCE (use with GRL)
    irm_penalty     -- IRM:   invariance penalty across source domains
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """CORAL loss: squared Frobenius distance between source/target covariances.

    Args:
        source: (N_s, D) feature tensor from source domain.
        target: (N_t, D) feature tensor from target domain.

    Returns:
        Scalar tensor.  Divide by 4D² as in the original paper so the loss
        is independent of feature dimensionality.
    """
    if source.shape[0] < 2 or target.shape[0] < 2:
        return source.new_zeros(1).squeeze()
    d = source.shape[1]
    cs = torch.cov(source.T)   # (D, D)
    ct = torch.cov(target.T)   # (D, D)
    return (cs - ct).pow(2).sum() / (4.0 * d * d)


def mmd_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    bandwidths: list[float] | None = None,
) -> torch.Tensor:
    """MMD loss with multi-scale Gaussian (RBF) kernel.

    Uses the unbiased estimator of the squared MMD:
        MMD² = E[k(s,s')] + E[k(t,t')] - 2·E[k(s,t)]

    Args:
        source: (N_s, D) feature tensor from source domain.
        target: (N_t, D) feature tensor from target domain.
        bandwidths: List of RBF bandwidth values σ². Defaults to [0.1, 1.0, 10.0].

    Returns:
        Scalar tensor (clamped to ≥ 0 for stability).
    """
    if bandwidths is None:
        bandwidths = [0.1, 1.0, 10.0]

    def _rbf(x: torch.Tensor, y: torch.Tensor, bw: float) -> torch.Tensor:
        # ||x - y||² matrix via (a-b)² = a² + b² - 2ab
        xx = (x * x).sum(1, keepdim=True)
        yy = (y * y).sum(1, keepdim=True)
        dist = xx + yy.T - 2.0 * x @ y.T   # (N_x, N_y)
        return torch.exp(-dist / (2.0 * bw))

    kss = sum(_rbf(source, source, b).mean() for b in bandwidths)
    ktt = sum(_rbf(target, target, b).mean() for b in bandwidths)
    kst = sum(_rbf(source, target, b).mean() for b in bandwidths)
    return (kss + ktt - 2.0 * kst).clamp(min=0.0)


def dann_loss(
    domain_logits: torch.Tensor,
    domain_labels: torch.Tensor,
) -> torch.Tensor:
    """DANN domain classification loss (binary cross-entropy).

    The GRL upstream ensures feature extractor gradients are reversed.
    This function is just BCE — the adversarial behaviour comes from GRL.

    Args:
        domain_logits: (N,) or (N,1) raw logits from DomainDiscriminator.
        domain_labels: (N,) binary labels — 0=source, 1=target.

    Returns:
        Scalar BCE loss tensor.
    """
    return F.binary_cross_entropy_with_logits(
        domain_logits.squeeze(dim=-1),
        domain_labels.float(),
    )


def irm_penalty(
    losses_per_domain: list[torch.Tensor],
    logits_per_domain: list[torch.Tensor],
) -> torch.Tensor:
    """IRM v1 invariance penalty.

    Measures how much the optimal classifier on top of the features differs
    across source domains.  Uses the IRM v1 formulation: for each domain,
    compute ||∇_w CE(w·logits, y)||² at w=1, then sum over domains.

    By the chain rule, ∇_w CE(w·logits, y)|_{w=1} = (∇_logits CE) · logits,
    so the per-domain CE losses are reused rather than recomputed with a dummy w.
    create_graph=True is required so the outer loss.backward() can differentiate
    through this gradient back to model parameters (IRM v1 is second-order by design).
    retain_graph defaults to True when create_graph=True, so the CE graphs survive
    for the erm_loss path in the caller.

    Args:
        losses_per_domain: Per-domain CE loss scalars (already computed for ERM).
        logits_per_domain: List of (N_k, C) logit tensors, one per source domain.

    Returns:
        Scalar penalty tensor (differentiable w.r.t. model parameters via logits).
    """
    penalty = logits_per_domain[0].new_zeros(1)
    for loss, logits in zip(losses_per_domain, logits_per_domain):
        grad_logits = torch.autograd.grad(loss, logits, create_graph=True)[0]
        penalty = penalty + (grad_logits * logits).sum().pow(2)
    return penalty
