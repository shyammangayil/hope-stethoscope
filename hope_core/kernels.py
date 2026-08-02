"""Closed-form kernel evaluations for HOPE (arXiv:2607.21366, Mobahi & Bartlett 2026).

A "neuron" in HOPE is treated as a rank-1 Hilbert-Schmidt operator. Two quantities
drive everything downstream (capacity, pruning cost, merge cost, the optimal
merged parent): the self-kernel K(i, i) and the cross-kernel K(i, j) of a pair
of ReLU neurons under the paper's Gaussian surrogate input distribution.

Both formulas below are for ReLU specifically (the paper derives closed forms
for ReLU; the general PH-1 family is only closed-form for ReLU in the main text).

Self-kernel (paper eq. 3) is exactly E[ReLU(Z)^2] for Z ~ N(beta, gamma^2):
    K(i,i) = (gamma_i^2 + beta_i^2) * Phi(beta_i / |gamma_i|)
             + beta_i * |gamma_i| * phi(beta_i / |gamma_i|)
This is a standard, independently verifiable closed form (not something that
can only be checked against the paper) -- see test_kernels.py for a Monte
Carlo cross-check against the definition itself.

Cross-kernel (paper eq. 4-5) is an approximation that assumes zero bias
(beta_i, beta_j ~ 0) to sidestep evaluating a bivariate normal CDF per pair,
which the paper's own text calls computationally prohibitive at scale. This
is a documented approximation, not exact -- see NOTES.md for why that's an
acceptable engineering trade rather than a silent shortcut.
"""

from __future__ import annotations

import math

import torch

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
_EPS = 1e-12


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF, Phi(x)."""
    return torch.special.ndtr(x)


def normal_pdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal PDF, phi(x)."""
    return _INV_SQRT_2PI * torch.exp(-0.5 * x * x)


def self_kernel(gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """K(i, i) for a ReLU neuron with effective BN scale/shift (gamma, beta).

    Equivalent to E[ReLU(Z)^2] for Z ~ N(beta, gamma^2). Always >= 0.

    Args:
        gamma: effective BN scale, any shape.
        beta: effective BN shift, broadcastable to gamma's shape.

    Returns:
        Tensor of the same broadcast shape, self-kernel values (>= 0).
    """
    abs_gamma = gamma.abs().clamp_min(_EPS)
    ratio = beta / abs_gamma
    k = (gamma.pow(2) + beta.pow(2)) * normal_cdf(ratio) + beta * abs_gamma * normal_pdf(ratio)
    # Numerical floor: the closed form is a sum of two non-negative-in-theory
    # terms, but floating point can push it very slightly negative near 0.
    return k.clamp_min(0.0)


def _effective_direction(w_eff_in: torch.Tensor) -> torch.Tensor:
    """Row-wise unit vectors of w_eff_in (each row is one neuron's effective input weight)."""
    norms = w_eff_in.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return w_eff_in / norms


def cosine_similarity_matrix(w_eff_in: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity of effective input weight vectors, one row per neuron."""
    directions = _effective_direction(w_eff_in)
    return directions @ directions.T


def cross_kernel(
    gamma: torch.Tensor,
    w_eff_in: torch.Tensor,
    self_k: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pairwise cross-kernel K(i, j) under the paper's zero-bias approximation (eq. 4-5).

    Args:
        gamma: (N,) effective BN scale per neuron.
        w_eff_in: (N, D) effective input weight vectors, one row per neuron.
        self_k: optional precomputed (N,) self-kernel values (avoids recompute
            when the caller already has them cached).

    Returns:
        (N, N) symmetric matrix of approximate cross-kernel values. Diagonal
        equals self_k (by construction of the approximation at rho_eff -> 1).
    """
    n = w_eff_in.shape[0]
    if self_k is None:
        # zero-bias self-kernel reduces to gamma^2 * Phi(0) = gamma^2 / 2... but
        # callers should pass the *true* self_k (with real beta) whenever
        # available; this branch only exists for standalone testing.
        self_k = gamma.pow(2) * 0.5

    w_norm = w_eff_in.norm(dim=-1).clamp_min(_EPS)  # (N,)
    rho_eff = cosine_similarity_matrix(w_eff_in)  # (N, N), in [-1, 1]
    rho_eff = rho_eff.clamp(-1.0 + 1e-6, 1.0 - 1e-6)  # keep 1-rho^2 well away from 0

    gamma_over_norm = (gamma.abs() / w_norm)  # (N,)
    outer_scale = gamma_over_norm.unsqueeze(1) * gamma_over_norm.unsqueeze(0)  # (N, N)

    kappa = (rho_eff / (1.0 - rho_eff.pow(2))) * outer_scale
    rho_hat = (2.0 * kappa) / (1.0 + torch.sqrt(1.0 + 4.0 * kappa.pow(2)))
    rho_hat = rho_hat.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    self_outer = torch.sqrt(self_k.clamp_min(0.0)).unsqueeze(1) * torch.sqrt(self_k.clamp_min(0.0)).unsqueeze(0)

    term = torch.sqrt((1.0 - rho_hat.pow(2)).clamp_min(0.0)) + (math.pi - torch.arccos(rho_hat)) * rho_hat
    k_ij = (1.0 / math.pi) * term * self_outer

    k_ij = k_ij.clamp_min(0.0)
    # Force exact symmetry and an exact diagonal (the approximation is only
    # asymptotically exact at rho_eff=1; floating point drift on the diagonal
    # would otherwise slightly disagree with self_k).
    k_ij = 0.5 * (k_ij + k_ij.T)
    k_ij.diagonal().copy_(self_k.clamp_min(0.0))
    return k_ij
