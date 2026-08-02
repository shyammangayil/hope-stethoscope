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
