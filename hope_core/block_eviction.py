"""Macro block eviction (paper section 8, eq. 19-20): forces an entire
residual pathway F(X) -> 0, collapsing a block to a pure identity Y=X,
competing against granular prune/merge within the same distortion-rate
selection criterion.

Practical realization of F(X) -> 0: zero the final BN's affine parameters
(gamma, beta) in the block. This is the same "zero-gamma" trick already
standard practice for residual-block initialization (He et al. 2016, Goyal
et al. 2017 -- both cited by the paper directly), so it's guaranteed
well-behaved rather than a novel risk: BN(conv(...)) with gamma=beta=0
outputs exactly zero regardless of what the internal conv layers compute,
which is a clean, trivially-reversible way to force Y = X + 0 = X without
needing to touch W1/W2/W3's actual weights.
"""

from __future__ import annotations

import torch
from torch import nn


def ambient_identity_energy(gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """E_identity = sum_k sqrt(gamma_k^2 + beta_k^2) over the block's final
    (pre-addition) BN -- the "parallel survival capacity" that keeps the
    macro-state's projection stable (section 8.2)."""
    return torch.sqrt(gamma.pow(2) + beta.pow(2)).sum()


def block_eviction_cost(internal_n_active: list[int], internal_e_active: list[torch.Tensor], identity_gamma: torch.Tensor, identity_beta: torch.Tensor) -> torch.Tensor:
    """J_evict = sum_l (N_active^(l) * E_active^(l)) / E_identity (eq. 20).

    Args:
        internal_n_active: active neuron counts for each internal layer (W1, W2
            in the canonical ResNet bottleneck -- NOT the final layer W3, which
            can't be granularly touched; see module docstring / paper section 8.1).
        internal_e_active: matching active capacities for those same layers.
        identity_gamma, identity_beta: the block's final BN affine params.
    """
    if len(internal_n_active) != len(internal_e_active):
        raise ValueError("internal_n_active and internal_e_active must have matching length")
    e_identity = ambient_identity_energy(identity_gamma, identity_beta).clamp_min(1e-8)
    numerator = sum(n * e for n, e in zip(internal_n_active, internal_e_active))
    return numerator / e_identity


def apply_block_eviction(final_bn: nn.BatchNorm2d | nn.BatchNorm1d) -> None:
    """Zero the block's final BN affine params in place, forcing F(X) -> 0
    (Y = X + 0 = X). Trivially reversible: the pre-eviction (gamma, beta)
    aren't destroyed by this module -- callers should snapshot them first
    (e.g. via the audit trail) if reversibility matters for their workflow.
    """
    with torch.no_grad():
        final_bn.weight.zero_()
        final_bn.bias.zero_()


def is_evicted(final_bn: nn.BatchNorm2d | nn.BatchNorm1d, tol: float = 1e-8) -> bool:
    """Check whether a block's final BN has already been evicted (both
    affine params ~0), useful for report generation and idempotency checks."""
    return bool(final_bn.weight.abs().max() < tol and final_bn.bias.abs().max() < tol)
