"""Effective parameters and capacity for a bank of neurons in one prunable layer.

A "NeuronBank" is HOPE's atomic unit: all neurons that live at the same point
in the network (e.g. all output channels of one Conv2d+BN, or all units of one
Linear+BN), sharing a downstream consumer. Capacity, pruning cost, and merge
cost are all computed within one bank at a time (paper section 3, eq. 1-2 and
the Hilbert-Schmidt norm in section 5).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from hope_core.kernels import self_kernel

_EPS_DEFAULT = 1e-5


@dataclass
class NeuronBank:
    """A layer's worth of neurons, in HOPE's (weight, BN) parameterization.

    Shapes (N = number of neurons in this bank, D_in = flattened fan-in,
    D_out = fan-out into whatever consumes this bank's output):
        w_raw:        (N, D_in)  raw incoming weight, pre-BN
        gamma, beta:  (N,)       BN affine scale / shift
        running_mean: (N,)       BN running mean
        running_var:  (N,)       BN running variance (>= 0)
        w_out:        (N, D_out) outgoing weight (this bank's contribution to
                                  the next layer's input)
        bn_eps:       scalar, BN's own numerical-stability epsilon
        name:         human-readable label for reports/logging
    """

    w_raw: torch.Tensor
    gamma: torch.Tensor
    beta: torch.Tensor
    running_mean: torch.Tensor
    running_var: torch.Tensor
    w_out: torch.Tensor
    bn_eps: float = _EPS_DEFAULT
    name: str = "layer"

    def __post_init__(self) -> None:
        n = self.w_raw.shape[0]
        for field_name in ("gamma", "beta", "running_mean", "running_var"):
            t = getattr(self, field_name)
            if t.shape != (n,):
                raise ValueError(f"{self.name}: {field_name} has shape {tuple(t.shape)}, expected ({n},)")
        if self.w_out.shape[0] != n:
            raise ValueError(f"{self.name}: w_out has {self.w_out.shape[0]} rows, expected {n}")
        if torch.any(self.running_var < 0):
            raise ValueError(f"{self.name}: running_var must be non-negative")

    @property
    def n_neurons(self) -> int:
        return self.w_raw.shape[0]

    def effective_input(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (w_eff_in, b_eff) per eq. 1 -- absorbs BN into the incoming weight.

        w_eff_in = (gamma / sqrt(running_var + eps)) * w_raw
        b_eff    = beta - gamma * running_mean / sqrt(running_var + eps)
        """
        denom = torch.sqrt(self.running_var + self.bn_eps)
        scale = self.gamma / denom  # (N,)
        w_eff_in = scale.unsqueeze(-1) * self.w_raw  # (N, D_in)
        b_eff = self.beta - self.gamma * self.running_mean / denom  # (N,)
        return w_eff_in, b_eff

    def self_kernel(self) -> torch.Tensor:
        """K(i, i) for every neuron in this bank -- depends only on (gamma, beta)."""
        return self_kernel(self.gamma, self.beta)

    def capacity(self) -> torch.Tensor:
        """||f_i||_H = ||w_out_i||_2 * sqrt(K(i, i)) for every neuron (paper section 5)."""
        k_self = self.self_kernel()
        w_out_norm = self.w_out.flatten(1).norm(dim=-1) if self.w_out.dim() > 2 else self.w_out.norm(dim=-1)
        return w_out_norm * torch.sqrt(k_self.clamp_min(0.0))

    def layer_capacity(self) -> torch.Tensor:
        """E(Phi) = sum_k ||f_k||_H -- the unique invariant-to-partitioning layer capacity (section 6)."""
        return self.capacity().sum()

    def drop(self, index: int) -> "NeuronBank":
        """Return a new bank with neuron `index` removed (used after a prune/merge is applied)."""
        keep = torch.ones(self.n_neurons, dtype=torch.bool)
        keep[index] = False
        return self._select(keep)

    def drop_pair_append(self, i: int, j: int, parent: "ParentNeuron") -> "NeuronBank":
        """Return a new bank with neurons i, j removed and the merged parent appended."""
        keep = torch.ones(self.n_neurons, dtype=torch.bool)
        keep[i] = False
        keep[j] = False
        base = self._select(keep)
        return NeuronBank(
            w_raw=torch.cat([base.w_raw, parent.w_raw.unsqueeze(0)], dim=0),
            gamma=torch.cat([base.gamma, parent.gamma.unsqueeze(0)]),
            beta=torch.cat([base.beta, parent.beta.unsqueeze(0)]),
            running_mean=torch.cat([base.running_mean, parent.running_mean.unsqueeze(0)]),
            running_var=torch.cat([base.running_var, parent.running_var.unsqueeze(0)]),
            w_out=torch.cat([base.w_out, parent.w_out.unsqueeze(0)], dim=0),
            bn_eps=self.bn_eps,
            name=self.name,
        )

    def _select(self, mask: torch.Tensor) -> "NeuronBank":
        return NeuronBank(
            w_raw=self.w_raw[mask],
            gamma=self.gamma[mask],
            beta=self.beta[mask],
            running_mean=self.running_mean[mask],
            running_var=self.running_var[mask],
            w_out=self.w_out[mask],
            bn_eps=self.bn_eps,
            name=self.name,
        )

    def clone(self) -> "NeuronBank":
        return replace(
            self,
            w_raw=self.w_raw.clone(),
            gamma=self.gamma.clone(),
            beta=self.beta.clone(),
            running_mean=self.running_mean.clone(),
            running_var=self.running_var.clone(),
            w_out=self.w_out.clone(),
        )


@dataclass
class ParentNeuron:
    """Physical parameters of a neuron synthesized by merging two others (eq. 15-18)."""

    w_raw: torch.Tensor
    gamma: torch.Tensor
    beta: torch.Tensor
    running_mean: torch.Tensor
    running_var: torch.Tensor
    w_out: torch.Tensor
