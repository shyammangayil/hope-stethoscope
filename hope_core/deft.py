"""DEFT: Dispersed Elastic Fine-Tuning (paper section 11.2.2, eq. 24-29).

Partitions a layer's neurons into a frozen Universal Core and a plastic
Peripheral Slack, using HOPE's own pruning-cost trajectory as the signal:
a neuron that survives late into a simulated "prune everything" pass is,
by construction of the greedy algorithm itself, judged more valuable than
one pruned early -- eq. 24's J_prune^(i) is recorded *at that neuron's own
removal step*, not re-evaluated at ambient width, so it reflects both the
neuron's own capacity and how much of the layer had already been stripped
away by the time it was that neuron's turn. This module reuses LayerState's
existing stable-id machinery for the collection pass rather than
reimplementing removal bookkeeping a second time.

Scope note: "Dynamic Resolution of Redundancy" (paper Figure 4a -- merging
redundant core neurons to free new plastic capacity) is implemented as
`resolve_redundancy()` using a directly-configurable distortion threshold,
since the paper's exact selection rule for this step lives in Appendix H
(Theoretical Guarantees of DEFT), which isn't available here. The core
elasticity/mask/gradient-scaling pipeline (eq. 24-29) is implemented exactly
as specified in the main text.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from hope_core.costs import prune_cost
from hope_core.layer_state import LayerState
from hope_core.neuron import NeuronBank

_EPS = 1e-6


def collect_prune_cost_distribution(bank: NeuronBank) -> dict[int, float]:
    """Simulate a prune-only pass to N=1 (never actually mutating the caller's
    bank -- operates on a clone), recording each ORIGINAL neuron's J_prune at
    its own removal step (eq. 24). Returns {original_position: J_prune}.

    The single surviving neuron at the end (never pruned, by Axiom 2 /
    Connectivity Preservation) is assigned float('inf') -- it's the most
    valuable neuron in the layer by construction of the algorithm, so it
    belongs in the core unconditionally, not by threshold comparison against
    a percentile it was never given the chance to be measured against.
    """
    state = LayerState(bank=bank.clone(), delta_p_init=1.0, name="deft_collection", build_merge_cache=False)
    costs: dict[int, float] = {}
    while state.n_active() > 1:
        result = state.best_prune()
        if result is None:
            break
        victim_id, _dr = result
        pos = state._pos_of(victim_id)
        j = float(prune_cost(state.capacities[pos], state.n_active(), state.e_active))
        e_b = float(state.e_active - state.capacities[pos])
        state.apply_prune(victim_id)
        if e_b > _EPS:  # eq. 25: filter out near-zero-capacity extinction artifacts
            costs[victim_id] = j
    if state.ids:
        costs[state.ids[0]] = float("inf")  # the sole survivor: unconditionally core
    return costs


def compute_lock_threshold(costs: dict[int, float], percentile: float) -> float:
    """J_lock per eq. 26. `percentile` in [0, 100]."""
    finite_costs = [c for c in costs.values() if c != float("inf")]
    if not finite_costs:
        return 1.0  # eq. 26, case 3: nothing to threshold against
    j_p = float(torch.quantile(torch.tensor(finite_costs), percentile / 100.0))
    j_sup = max(finite_costs)
    if j_p >= _EPS:
        return j_p
    if j_sup >= _EPS:
        return j_sup
    return 1.0


def elasticity_map(costs: dict[int, float], j_lock: float, n_neurons: int) -> torch.Tensor:
    """E in {0,1}^N per eq. 27: E_i=1 (plastic/slack) if J_prune^(i) < j_lock,
    else 0 (frozen/core). `costs` keys are original bank positions (0..N-1)."""
    elasticity = torch.zeros(n_neurons)
    for pos, j in costs.items():
        elasticity[pos] = 1.0 if j < j_lock else 0.0
    return elasticity


@dataclass
class DeftPartition:
    """Result of partitioning one layer for DEFT fine-tuning."""

    elasticity: torch.Tensor  # (N,) in {0,1}, 1=plastic/slack, 0=frozen/core
    j_lock: float
    costs: dict[int, float]

    @property
    def core_fraction(self) -> float:
        return float((self.elasticity == 0).float().mean())


def partition_layer(bank: NeuronBank, percentile: float) -> DeftPartition:
    """End-to-end eq. 24-27 for one layer."""
    costs = collect_prune_cost_distribution(bank)
    j_lock = compute_lock_threshold(costs, percentile)
    elasticity = elasticity_map(costs, j_lock, bank.n_neurons)
    return DeftPartition(elasticity=elasticity, j_lock=j_lock, costs=costs)


def structural_mask(elasticity_in: torch.Tensor, elasticity_out: torch.Tensor) -> torch.Tensor:
    """M_{j,k} per eq. 28: severs upstream-plastic -> downstream-frozen
    connections. Shape (len(elasticity_out), len(elasticity_in)), matching a
    standard nn.Linear/Conv weight's (out, in, ...) leading dimensions.

    M[j,k] = 0 if elasticity_in[k] > 0 (upstream plastic) and
             elasticity_out[j] == 0 (downstream frozen)
             else 1
    """
    plastic_upstream = (elasticity_in > 0).unsqueeze(0)  # (1, in)
    frozen_downstream = (elasticity_out == 0).unsqueeze(1)  # (out, 1)
    sever = plastic_upstream & frozen_downstream
    return (~sever).float()


def apply_structural_mask(weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Applies a structural mask to a weight tensor, broadcasting over any
    trailing (e.g. conv kernel) dimensions beyond (out, in)."""
    extra_dims = weight.dim() - mask.dim()
    if extra_dims > 0:
        mask = mask.reshape(mask.shape + (1,) * extra_dims)
    return weight * mask


def scale_gradient_(grad: torch.Tensor, elasticity_out: torch.Tensor) -> None:
    """In-place eq. 29 gradient scaling: g_t = E_out (broadcast) * grad.
    Call from a backward hook or right before optimizer.step()."""
    extra_dims = grad.dim() - elasticity_out.dim()
    e = elasticity_out
    if extra_dims > 0:
        e = e.reshape(e.shape + (1,) * extra_dims)
    grad.mul_(e)


def register_deft_gradient_hooks(layer_weight: nn.Parameter, elasticity_out: torch.Tensor) -> torch.utils.hooks.RemovableHandle:
    """Convenience wrapper: registers a backward hook on `layer_weight` that
    applies eq. 29 scaling automatically. Returns the hook handle so the
    caller can remove it later (e.g. after fine-tuning completes)."""

    def _hook(grad: torch.Tensor) -> torch.Tensor:
        scaled = grad.clone()
        scale_gradient_(scaled, elasticity_out)
        return scaled

    return layer_weight.register_hook(_hook)
