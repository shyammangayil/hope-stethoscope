"""Distortion cost functionals J_prune and J_merge (paper section 6.3, eq. 6).

Both costs share the same shape: a projection-error numerator (how far the
network moves in Hilbert space) over a capacity-remaining denominator (so
depleting a nearly-exhausted layer gets increasingly expensive -- this is what
gives HOPE its "connectivity preservation" guarantee instead of hollowing out
one layer to zero while ignoring an obviously-redundant one elsewhere).

The action-selection criterion used by the encoding loop is the *distortion
rate* J_k / delta_P_k^init (eq. 23) -- lower is better, and delta_P_init is a
static (pre-computed once) parameter-yield estimate, deliberately not
recomputed as the layer shrinks (paper section 9 explains why: using the live
shrinking parameter count breaks Dantzig's item-independence assumption and
traps the optimizer in a fragmented state).
"""

from __future__ import annotations

import torch

_EPS = 1e-8


def hilbert_inner_product(k_ij: torch.Tensor, w_out_i: torch.Tensor, w_out_j: torch.Tensor) -> torch.Tensor:
    """<f_i, f_j>_H = K(i, j) * <w_out_i, w_out_j> (paper eq. "Kernel Formulation").

    Args:
        k_ij: scalar or (...,) cross-kernel value(s).
        w_out_i, w_out_j: (D_out,) or (..., D_out) outgoing weight vectors.
    """
    return k_ij * (w_out_i * w_out_j).sum(dim=-1)


def prune_cost(capacity_i: torch.Tensor, n_active: int, e_active: torch.Tensor) -> torch.Tensor:
    """J_prune = N * ||f_i||_H / (E_a - ||f_i||_H).

    Blows up (correctly) as capacity_i approaches e_active -- pruning the last
    surviving neuron in a layer should be maximally expensive (Axiom 2,
    Connectivity Preservation).
    """
    denom = (e_active - capacity_i).clamp_min(_EPS)
    return n_active * capacity_i / denom


def merge_distortion_squared(
    capacity_i: torch.Tensor,
    capacity_j: torch.Tensor,
    capacity_p: torch.Tensor,
    inner_ip: torch.Tensor,
    inner_jp: torch.Tensor,
) -> torch.Tensor:
    """D^2 = ||f_i - f_p||_H^2 + ||f_j - f_p||_H^2, expanded via the HS inner product."""
    dist_ip_sq = (capacity_i.pow(2) + capacity_p.pow(2) - 2.0 * inner_ip).clamp_min(0.0)
    dist_jp_sq = (capacity_j.pow(2) + capacity_p.pow(2) - 2.0 * inner_jp).clamp_min(0.0)
    return dist_ip_sq + dist_jp_sq


def merge_cost(
    capacity_i: torch.Tensor,
    capacity_j: torch.Tensor,
    capacity_p: torch.Tensor,
    distortion_sq: torch.Tensor,
    n_active: int,
    e_active: torch.Tensor,
) -> torch.Tensor:
    """J_merge = N * sqrt(D^2) / (E_a - ||f_i|| - ||f_j|| + ||f_p||)."""
    denom = (e_active - capacity_i - capacity_j + capacity_p).clamp_min(_EPS)
    return n_active * torch.sqrt(distortion_sq.clamp_min(0.0)) / denom


def distortion_rate(cost: torch.Tensor, delta_p_init: torch.Tensor) -> torch.Tensor:
    """DR = J / delta_P^init (eq. 23) -- the greedy selection criterion, lower is better."""
    return cost / delta_p_init.clamp_min(1.0)
