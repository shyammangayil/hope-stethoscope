"""LayerState: caches per-pair merge *geometry* for one compressible layer,
and applies the paper's O(N) local update after an action instead of
recomputing everything from scratch (section 10, "Localized Update").

Two correctness pitfalls this module exists to avoid, both caught by tests
rather than assumed away:

1. Index vs. stable id (see test_repeated_mixed_actions_stay_consistent_with_full_rebuild).
   Neurons get removed from the bank as the loop runs, which shifts every
   later position down by one (NeuronBank.drop/drop_pair_append). Caching
   anything by *position* would silently corrupt after the first removal.
   Every neuron gets a permanent, never-reused `stable_id` at creation, and
   all caches are keyed by that id; positions are only used transiently to
   index into the live bank tensors.

2. Caching the final J_merge instead of the geometry behind it. J_merge =
   N * sqrt(D^2) / (E_a - cap_i - cap_j + cap_p) depends on the *layer-wide*
   N and E_a, which change on every action -- including actions that don't
   touch this particular pair at all. An earlier version of this file cached
   the fully-assembled J_merge value per pair, which meant pairs untouched
   by a given action silently went stale the moment any *other* action in
   the same layer changed E_a. What's actually safe to cache is the
   pair-intrinsic geometry (the optimal parent, its capacity, and D^2 --
   none of which depend on layer-wide state); the O(1)-per-pair "evaluation
   of J during subsequent greedy scans" the paper describes (section 10)
   means assembling J fresh from cached geometry plus live (N, E_a), not
   looking up a stale precomputed number. This is what best_merge() does.

3. Bounded staleness in cached parent geometry (see refresh_all() docstring
   for the full explanation). The optimal parent's scale s* (eq. 12) depends
   on layer-wide e_rem, so strictly speaking every cached pair's parent goes
   slightly stale the moment any *other* action changes E_a -- not just its
   final J_merge (best_merge() always recomputes that fresh from live
   state), but the parent's exact identity. Eliminating this fully would
   require O(N^2) work per action, defeating the O(N) local update the paper
   specifies. This is accepted, bounded approximation, not a bug -- observed
   at roughly 1-2% relative drift in a 10-neuron/6-action test scenario. Call
   refresh_all() if a caller needs tighter consistency at the cost of
   dropping back to O(N^2) per refresh.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from hope_core.costs import hilbert_inner_product, merge_cost, merge_distortion_squared, prune_cost
from hope_core.kernels import cross_kernel
from hope_core.merge import compute_optimal_parent
from hope_core.neuron import NeuronBank, ParentNeuron


@dataclass
class _PairGeometry:
    parent: ParentNeuron
    cap_p: torch.Tensor
    d2: torch.Tensor  # ||f_i - f_p||^2 + ||f_j - f_p||^2 -- pair-intrinsic, layer-state-independent


@dataclass
class LayerState:
    bank: NeuronBank
    delta_p_init: float
    name: str = "layer"

    ids: list[int] = field(default_factory=list)
    _next_id: int = 0
    build_merge_cache: bool = True
    capacities: torch.Tensor = field(init=False, repr=False)
    e_active: torch.Tensor = field(init=False, repr=False)
    _pair_geometry: dict[frozenset, _PairGeometry] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.ids:
            self.ids = list(range(self.bank.n_neurons))
            self._next_id = self.bank.n_neurons
        self.capacities = self.bank.capacity()
        self.e_active = self.capacities.sum()
        if self.build_merge_cache:
            self._rebuild_all_pairs()

    # -- id/position bookkeeping ---------------------------------------------

    def _pos_of(self, stable_id: int) -> int:
        return self.ids.index(stable_id)

    def n_active(self) -> int:
        return self.bank.n_neurons

    # -- cache construction (O(N^2), paid once at init and once per O(N) local
    #    update -- never re-paid in full just to answer a query) ------------

    def _rebuild_all_pairs(self) -> None:
        self._pair_geometry.clear()
        n = self.bank.n_neurons
        for a in range(n):
            for b in range(a + 1, n):
                self._cache_pair_geometry(a, b)

    def _cache_pair_geometry(self, pos_a: int, pos_b: int) -> None:
        parent, cap_p = compute_optimal_parent(self.bank, pos_a, pos_b)
        d2 = self._distortion_for(pos_a, pos_b, parent, cap_p)
        key = frozenset((self.ids[pos_a], self.ids[pos_b]))
        self._pair_geometry[key] = _PairGeometry(parent=parent, cap_p=cap_p, d2=d2)

    def _distortion_for(self, pos_i: int, pos_j: int, parent: ParentNeuron, cap_p: torch.Tensor) -> torch.Tensor:
        """D^2 for this pair against the given parent. Pair-intrinsic: does NOT
        depend on n_active/e_active, so it's safe to cache indefinitely until
        one of the two neurons involved actually changes."""
        w_eff_in, _ = self.bank.effective_input()
        self_k = self.bank.self_kernel()
        parent_w_eff = parent.w_raw * parent.gamma / torch.sqrt(parent.running_var + self.bank.bn_eps)
        parent_self_k = cap_p.pow(2) / parent.w_out.norm().clamp_min(1e-8).pow(2)

        stacked_gamma_i = torch.stack([self.bank.gamma[pos_i], parent.gamma])
        stacked_w_i = torch.stack([w_eff_in[pos_i], parent_w_eff])
        stacked_self_k_i = torch.stack([self_k[pos_i], parent_self_k])
        k_ip = cross_kernel(stacked_gamma_i, stacked_w_i, self_k=stacked_self_k_i)[0, 1]

        stacked_gamma_j = torch.stack([self.bank.gamma[pos_j], parent.gamma])
        stacked_w_j = torch.stack([w_eff_in[pos_j], parent_w_eff])
        stacked_self_k_j = torch.stack([self_k[pos_j], parent_self_k])
        k_jp = cross_kernel(stacked_gamma_j, stacked_w_j, self_k=stacked_self_k_j)[0, 1]

        inner_ip = hilbert_inner_product(k_ip, self.bank.w_out[pos_i], parent.w_out)
        inner_jp = hilbert_inner_product(k_jp, self.bank.w_out[pos_j], parent.w_out)
        cap_i, cap_j = self.capacities[pos_i], self.capacities[pos_j]
        return merge_distortion_squared(cap_i, cap_j, cap_p, inner_ip, inner_jp)

    def _assemble_cost(self, id_a: int, id_b: int, geom: _PairGeometry) -> torch.Tensor:
        """Fresh J_merge from cached geometry + LIVE n_active/e_active. O(1)."""
        cap_a = self.capacities[self._pos_of(id_a)]
        cap_b = self.capacities[self._pos_of(id_b)]
        return merge_cost(cap_a, cap_b, geom.cap_p, geom.d2, self.n_active(), self.e_active)

    # -- candidate queries (O(current N^2) scan, matching the paper's stated
    #    per-iteration greedy-scan complexity -- section 10, step 2) --------

    def best_prune(self) -> tuple[int, float] | None:
        """Returns (stable_id, distortion_rate) for the cheapest legal prune, or None."""
        if self.bank.n_neurons <= 1:
            return None  # Axiom 2 / Connectivity Preservation: never fully empty a layer
        pos = int(torch.argmin(self.capacities).item())
        cost = prune_cost(self.capacities[pos], self.n_active(), self.e_active)
        return self.ids[pos], float(cost) / max(self.delta_p_init, 1e-8)

    def best_merge(self) -> tuple[int, int, float] | None:
        """Returns (stable_id_i, stable_id_j, distortion_rate) for the cheapest legal merge, or None."""
        if not self._pair_geometry:
            return None
        best_key, best_dr, best_cost = None, None, None
        for key, geom in self._pair_geometry.items():
            id_a, id_b = tuple(key)
            cost = self._assemble_cost(id_a, id_b, geom)
            dr = float(cost) / max(self.delta_p_init, 1e-8)
            if best_dr is None or dr < best_dr:
                best_key, best_dr, best_cost = key, dr, cost
        id_a, id_b = tuple(best_key)
        return id_a, id_b, best_dr

    def current_pair_costs(self) -> dict[frozenset, float]:
        """Fresh {pair: J_merge} for every cached pair, using live state. For
        reporting/inspection -- NOT stored persistently (see module docstring
        for why persisting the assembled cost is the wrong thing to do)."""
        return {
            key: float(self._assemble_cost(*tuple(key), geom))
            for key, geom in self._pair_geometry.items()
        }

    # -- action application (O(N) local update) -------------------------------

    def apply_prune(self, stable_id: int) -> None:
        pos = self._pos_of(stable_id)
        self.bank = self.bank.drop(pos)
        del self.ids[pos]
        self.capacities = self.bank.capacity()
        self.e_active = self.capacities.sum()
        self._drop_pairs_involving(stable_id)

    def apply_merge(self, id_a: int, id_b: int) -> int:
        """Applies the cached-optimal merge for (id_a, id_b); returns the new parent's stable_id."""
        key = frozenset((id_a, id_b))
        if key not in self._pair_geometry:
            # defensive fallback: shouldn't happen via best_merge(), but keeps
            # this method safe to call directly (e.g. from tests or a CLI
            # override) without silently using a stale/missing cache entry.
            self._cache_pair_geometry(self._pos_of(id_a), self._pos_of(id_b))
        parent = self._pair_geometry[key].parent

        pos_a, pos_b = self._pos_of(id_a), self._pos_of(id_b)
        lo, hi = sorted((pos_a, pos_b))
        self.bank = self.bank.drop_pair_append(lo, hi, parent)

        new_id = self._next_id
        self._next_id += 1
        del self.ids[hi]
        del self.ids[lo]
        self.ids.append(new_id)

        self.capacities = self.bank.capacity()
        self.e_active = self.capacities.sum()

        self._drop_pairs_involving(id_a)
        self._drop_pairs_involving(id_b)
        new_pos = self.ids.index(new_id)
        for other_pos, other_id in enumerate(self.ids):
            if other_id == new_id:
                continue
            self._cache_pair_geometry(*sorted((new_pos, other_pos)))
        return new_id

    def _drop_pairs_involving(self, stable_id: int) -> None:
        stale = [k for k in self._pair_geometry if stable_id in k]
        for k in stale:
            del self._pair_geometry[k]

    def resync_downstream_consumer(self, downstream_w_raw: torch.Tensor) -> None:
        """Call after the layer this one feeds INTO (its downstream
        consumer) changes its own neuron set via prune/merge. Fixes the
        cross-layer weight propagation gap documented in NOTES.md and
        README.md's former "Not done" list.

        Why this is correct for both prune and merge on the downstream
        side, with no special-casing needed: this layer's w_out is not an
        independent quantity, it's literally the downstream layer's raw
        incoming weight, transposed (see adapters.py's extract_*_bn_bank).
        The downstream layer's OWN w_raw, after either a prune or a merge,
        already correctly encodes "how do my current neurons read from
        this (unchanged) layer's output space" -- merge.py's machinery
        computes a new row that's valid across the full upstream output
        space by construction (eq. 16-18 operate in the same w_eff_in
        space regardless of what layer is asking). So the fix is just:
        re-derive w_out fresh from the downstream layer's current w_raw,
        transposed, whenever it changes.

        capacity and e_active are always refreshed (cheap, O(N)). The
        pair-geometry cache is only refreshed if build_merge_cache is (and
        was) True -- resyncing must not silently force the O(N^2) cache
        build for a layer that was specifically configured to avoid it
        (e.g. DEFT's collect_prune_cost_distribution, or a layer above
        reproduce_paper_resnet50.py's --merge-width-limit).
        """
        new_w_out = downstream_w_raw.T.clone()
        if new_w_out.shape[0] != self.bank.n_neurons:
            raise ValueError(
                f"{self.name}: downstream w_raw has {new_w_out.shape[0]} rows, "
                f"expected {self.bank.n_neurons} (this layer's current neuron count)"
            )
        self.bank = self.bank.with_replaced_w_out(new_w_out)
        self.capacities = self.bank.capacity()
        self.e_active = self.capacities.sum()
        if self.build_merge_cache:
            self.refresh_all()

    def resync_upstream_provider(self, upstream_w_out: torch.Tensor) -> None:
        """The mirror of resync_downstream_consumer, for the OTHER
        direction: call after the layer THIS one reads FROM (its upstream
        provider) changes its own neuron set via prune/merge.

        This direction matters during the live loop, not just at final
        rebuild -- found by a real test failure, not anticipated in
        advance. self_kernel/capacity don't depend on w_raw's shape or
        values (only gamma/beta), so a stale upstream-facing w_raw doesn't
        corrupt THIS layer's own capacity numbers. But merge.py's direction-
        finding (compute_optimal_parent) uses effective_input(), which DOES
        read w_raw's actual values via cosine similarity between this
        layer's own neurons -- if this layer's w_raw still has columns (or
        wrong values in existing columns) for upstream neurons that were
        pruned or merged away, this layer's OWN internal merge candidate
        selection silently corrupts, not just the eventual rebuild shape.

        upstream_w_out should be the upstream LayerState's CURRENT
        bank.w_out (kept correct via ITS OWN resync_downstream_consumer
        calls) -- transposing it gives exactly this layer's correct new
        w_raw, by the same "w_out and w_raw are transposes of each other
        across a layer boundary" relationship resync_downstream_consumer's
        docstring describes.
        """
        new_w_raw = upstream_w_out.T.clone()
        if new_w_raw.shape[0] != self.bank.n_neurons:
            raise ValueError(
                f"{self.name}: upstream_w_out.T has {new_w_raw.shape[0]} rows, "
                f"expected {self.bank.n_neurons} (this layer's current neuron count)"
            )
        self.bank = self.bank.with_replaced_w_raw(new_w_raw)
        self.capacities = self.bank.capacity()  # unchanged in practice (doesn't depend on w_raw), refreshed for safety
        self.e_active = self.capacities.sum()
        if self.build_merge_cache:
            self.refresh_all()  # pair geometry DOES depend on w_raw's values via effective_input()

    def refresh_all(self) -> None:
        """Full O(N^2) recompute of every cached pair's geometry against the
        current bank state.

        Why this exists: the optimal parent's *scale* s* (eq. 12) depends on
        e_rem = E_a - cap_i - cap_j, which is layer-wide state. An action
        anywhere in the layer changes E_a, which means every OTHER cached
        pair's parent is technically now defined with respect to a slightly
        stale e_rem -- not just its final J_merge (which best_merge() already
        recomputes fresh each call), but the parent's exact identity itself.
        Fully eliminating this staleness would mean recomputing all O(N^2)
        pairs on every single action, which defeats the point of the O(N)
        local update the paper specifies (section 10) and this class
        implements. This is a real, bounded approximation inherent to any
        O(N)-per-step greedy scheme on a moving objective, not a latent bug --
        see NOTES.md for the numeric bound observed in testing (~1-2%
        relative drift in J_merge between untouched-pair refreshes on a
        10-neuron toy layer over 6 actions).

        Callers that want tighter consistency (e.g. a caller running many
        actions on a small, cheap-to-refresh layer) can call this
        periodically, trading the O(N) per-action guarantee for exactness.
        The encoding loop does not call this automatically.
        """
        self._rebuild_all_pairs()
