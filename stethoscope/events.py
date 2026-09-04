"""stethoscope/events.py
=========================
ObservationEvent: standardized observation event/data structure
(architectural-evolution spec §9, `docs/ARCHITECTURE_EVOLUTION_SPEC.md`).

This module intentionally contains a data structure only -- no Observer,
CapturePolicy, or storage backend. Those are explicitly Phase 2 in the
spec's own phase breakdown (§26), and building them now, before Phase 2
has a real producer/consumer to validate the shape against, would be
speculative (see the architecture delta report §13). For the same
reason, this module does not import `stethoscope.registry` for a
`ObservationEvent.for_component(...)` convenience constructor, tempting
as that is -- nothing produces an ObservationEvent yet, so there is
nothing to validate that convenience API against. Add it in Phase 2, once
the Observer exists and has an opinion about the ergonomics it actually
wants.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ObservationPhase(str, Enum):
    """Spec §7's phases: the first five are the "must support at least"
    list, the last three are "where technically possible.\""""

    INPUT = "input"
    PRE_FORWARD = "pre_forward"
    INTERNAL_STATE = "internal_state"
    OUTPUT = "output"
    POST_FORWARD = "post_forward"
    BACKWARD = "backward"
    GRADIENT = "gradient"
    PARAMETER_UPDATE = "parameter_update"


@dataclass
class ObservationEvent:
    """Spec §9's exact field set.

    `tensor_reference` is deliberately typed `Any` and is NOT guaranteed
    to be a full tensor -- spec §9 explicitly says "Do not necessarily
    store every full tensor permanently." What it holds (a full tensor, a
    sampled slice, or nothing with only `statistics` populated instead)
    is a Phase-2 CapturePolicy decision; this dataclass just has a slot
    for it and does not interpret it.
    """

    experiment_id: str
    model_id: str
    # str(ComponentID), not a ComponentID -- keeps this dataclass
    # trivially JSON-serializable without a custom encoder for the ID
    # type; see stethoscope.ids.ComponentID.parse() to go back the other
    # way when needed.
    component_id: str
    component_type: str  # a ComponentType's .value
    phase: str  # an ObservationPhase's .value

    layer: Optional[int] = None
    token_positions: Optional[tuple[int, ...]] = None
    timestep: Optional[int] = None

    tensor_metadata: Optional[dict] = None  # cheap, always-safe-to-capture facts: shape, dtype, device
    tensor_reference: Optional[Any] = None  # see class docstring -- not necessarily the full tensor
    statistics: Optional[dict] = None  # e.g. {"l2_norm": ..., "mean": ...}
    provenance: dict = field(default_factory=dict)

    def to_dict(self, *, include_tensor_reference: bool = False) -> dict:
        """JSON-safe dict. `tensor_reference` is excluded by default: it
        may hold a live tensor object (not JSON-safe), and per spec §9 it
        isn't guaranteed to even be a full tensor -- serializing it
        meaningfully is a capture-policy-specific concern for Phase 2,
        not something this method should guess at."""
        d = {
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "component_id": self.component_id,
            "component_type": self.component_type,
            "phase": self.phase,
            "layer": self.layer,
            "token_positions": list(self.token_positions) if self.token_positions is not None else None,
            "timestep": self.timestep,
            "tensor_metadata": self.tensor_metadata,
            "statistics": self.statistics,
            "provenance": self.provenance,
        }
        if include_tensor_reference:
            d["tensor_reference"] = self.tensor_reference
        return d
