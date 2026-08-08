"""ProgressiveEncodingLoop: the greedy, receding-horizon compression loop
(paper section 9-10). Scans every compressible layer for its cheapest legal
action, executes the single globally-cheapest one (by distortion rate,
eq. 23), updates state, and repeats until a target is reached.

This is intentionally a thin orchestrator over LayerState -- all the actual
math lives in kernels.py/neuron.py/costs.py/merge.py, all the per-layer
bookkeeping lives in layer_state.py. This module's only job is cross-layer
selection and producing an audit trail, which is also the thing the
Stethoscope report (report.py) consumes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from hope_core.layer_state import LayerState


@dataclass
class ActionRecord:
    """One entry in the audit trail -- every projection, in order, with enough
    detail to reconstruct why the optimizer chose it."""

    step: int
    layer_name: str
    action: Literal["prune", "merge"]
    involved_ids: tuple[int, ...]
    result_id: int | None  # new parent's stable id, for merge; None for prune
    distortion_rate: float
    params_freed: float
    total_params_after: float
    wall_time_seconds: float


@dataclass
class ProgressiveEncodingLoop:
    layers: dict[str, LayerState]
    initial_total_params: float = field(init=False)
    total_params: float = field(init=False)
    history: list[ActionRecord] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.initial_total_params = sum(
            state.delta_p_init * state.n_active() for state in self.layers.values()
        )
        self.total_params = self.initial_total_params

    @property
    def current_density(self) -> float:
        if self.initial_total_params <= 0:
            return 0.0
        return self.total_params / self.initial_total_params

    def _find_best_global_action(self) -> tuple[str, str, tuple] | None:
        """Scan every layer's best prune and best merge candidate; return the
        single globally-cheapest (layer_name, action_type, payload) or None
        if no layer has any legal action left."""
        best = None  # (dr, layer_name, action_type, payload)
        for layer_name, state in self.layers.items():
            prune = state.best_prune()
            if prune is not None:
                victim_id, dr = prune
                if best is None or dr < best[0]:
                    best = (dr, layer_name, "prune", (victim_id,))
            merge = state.best_merge()
            if merge is not None:
                id_a, id_b, dr = merge
                if best is None or dr < best[0]:
                    best = (dr, layer_name, "merge", (id_a, id_b))
        if best is None:
            return None
        _dr, layer_name, action_type, payload = best
        return layer_name, action_type, payload

    def step(self) -> ActionRecord | None:
        """Execute exactly one action (the single globally-cheapest one across
        all layers). Returns the ActionRecord, or None if no legal action
        remains anywhere in the network."""
        choice = self._find_best_global_action()
        if choice is None:
            return None
        layer_name, action_type, payload = choice
        state = self.layers[layer_name]
        t0 = time.perf_counter()

        if action_type == "prune":
            (victim_id,) = payload
            _id, dr = state.best_prune()
            state.apply_prune(victim_id)
            result_id = None
            involved = (victim_id,)
        else:
            id_a, id_b = payload
            _a, _b, dr = state.best_merge()
            result_id = state.apply_merge(id_a, id_b)
            involved = (id_a, id_b)

        self.total_params -= state.delta_p_init
        record = ActionRecord(
            step=len(self.history),
            layer_name=layer_name,
            action=action_type,
            involved_ids=involved,
            result_id=result_id,
            distortion_rate=dr,
            params_freed=state.delta_p_init,
            total_params_after=self.total_params,
            wall_time_seconds=time.perf_counter() - t0,
        )
        self.history.append(record)
        return record

    def run(
        self,
        target_density: float | None = None,
        max_steps: int | None = None,
    ) -> list[ActionRecord]:
        """Run until target_density is reached, max_steps is hit, or no legal
        action remains -- whichever comes first. At least one of
        target_density/max_steps should be set, or this runs until every
        layer is down to a single neuron."""
        if target_density is None and max_steps is None:
            raise ValueError("run() needs at least one of target_density or max_steps to avoid an unbounded loop")

        new_records = []
        while True:
            if target_density is not None and self.current_density <= target_density:
                break
            if max_steps is not None and len(new_records) >= max_steps:
                break
            record = self.step()
            if record is None:
                break  # every layer is down to its last neuron; nothing left to do
            new_records.append(record)
        return new_records


@dataclass
class ChainAwareEncodingLoop:
    """Wraps a ProgressiveEncodingLoop and automatically keeps adjacent
    layers' shared boundary weight in sync in BOTH directions, fixing the
    cross-layer weight propagation gap documented in NOTES.md.

    Composition, not inheritance or modification: ProgressiveEncodingLoop's
    own step()/run() are used exactly as-is and stay exactly as tested by
    tests/test_encoding_loop.py. This class only adds a post-step hook.

    Both directions matter, not just one -- found the hard way, not
    anticipated correctly on the first attempt at this class:
      - downstream changes -> upstream.resync_downstream_consumer(): keeps
        upstream's w_out (used for upstream's OWN capacity) correct.
      - upstream changes -> downstream.resync_upstream_provider(): keeps
        downstream's w_raw (used for downstream's OWN merge-direction
        finding via effective_input(), not just eventual rebuild shape)
        correct. Missing this direction is what a first version of this
        class's own test suite caught: a joint-compression run failed with
        a real shape-mismatch error from resync_downstream_consumer's own
        validation, the moment upstream got pruned before downstream did.

    `chain` maps downstream_layer_name -> upstream_layer_name, i.e. the
    same direction data flows: chain={"L2": "L1"} means L1's output feeds
    L2's input. An action on L2 triggers L1's resync; an action on L1
    triggers L2's resync -- both directions are derived from this one dict.
    """

    loop: ProgressiveEncodingLoop
    chain: dict[str, str] = field(default_factory=dict)
    _reverse_chain: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        unknown = set(self.chain.values()) | set(self.chain.keys())
        missing = unknown - set(self.loop.layers.keys())
        if missing:
            raise ValueError(f"chain references layer(s) not in loop.layers: {missing}")
        # chain: downstream -> upstream. reverse: upstream -> downstream.
        self._reverse_chain = {upstream: downstream for downstream, upstream in self.chain.items()}

    @property
    def current_density(self) -> float:
        return self.loop.current_density

    @property
    def history(self) -> list[ActionRecord]:
        return self.loop.history

    def step(self) -> ActionRecord | None:
        record = self.loop.step()
        if record is None:
            return None

        if record.layer_name in self.chain:
            upstream_name = self.chain[record.layer_name]
            upstream_state = self.loop.layers[upstream_name]
            downstream_state = self.loop.layers[record.layer_name]
            upstream_state.resync_downstream_consumer(downstream_state.bank.w_raw)

        if record.layer_name in self._reverse_chain:
            downstream_name = self._reverse_chain[record.layer_name]
            downstream_state = self.loop.layers[downstream_name]
            upstream_state = self.loop.layers[record.layer_name]
            downstream_state.resync_upstream_provider(upstream_state.bank.w_out)

        return record

    def run(self, target_density: float | None = None, max_steps: int | None = None) -> list[ActionRecord]:
        """Same contract as ProgressiveEncodingLoop.run(), but calling THIS
        class's step() (with the resync hooks) on every iteration rather
        than delegating to loop.run(), which would skip them entirely."""
        if target_density is None and max_steps is None:
            raise ValueError("run() needs at least one of target_density or max_steps to avoid an unbounded loop")

        new_records = []
        while True:
            if target_density is not None and self.current_density <= target_density:
                break
            if max_steps is not None and len(new_records) >= max_steps:
                break
            record = self.step()
            if record is None:
                break
            new_records.append(record)
        return new_records
