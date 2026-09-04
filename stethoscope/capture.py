"""stethoscope/capture.py
=========================
CaptureMode, CapturePolicy: what to capture and how much detail to keep
(architectural-evolution spec §22 "Memory/performance architecture" and
§23 "Do not instrument everything by default",
`docs/ARCHITECTURE_EVOLUTION_SPEC.md`).

Two independent axes, deliberately kept separate:

  - SCOPE: which components/layers/tokens are eligible to be observed at
    all (`components`, `layers`, `tokens` below) -- applies the same way
    regardless of mode, including NONE (which makes scope irrelevant by
    forcing everything out).
  - DETAIL: how much information is kept for an in-scope event (`mode`
    below) -- NONE keeps nothing, SUMMARY thins each kept event down to
    a headline statistic, SAMPLED keeps a systematic subset of events at
    full detail, SELECTED and FULL both keep full detail (the difference
    between them is caller intent -- SELECTED signals "only within these
    filters", FULL signals "everything, in full" -- not a difference in
    per-event content).

This module is deliberately pure: no dependency on `core/`,
`compression/`, `harness/`, `integrations/`, or `hope_core/`, and no
mutable/counting state (sampling's "which Nth event is this" counter
lives on `Observer` in `observer.py`, not here) -- a `CapturePolicy` is a
plain, comparable, declarative value, not a coordinator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from stethoscope.ids import ComponentType


class CaptureMode(str, Enum):
    """Spec §22's five capture modes, verbatim."""

    NONE = "none"
    SUMMARY = "summary"
    SAMPLED = "sampled"
    SELECTED = "selected"
    FULL = "full"


@dataclass
class CapturePolicy:
    """What to capture. Defaults to a lightweight mode (SUMMARY, no
    filters = "every component/layer/token, headline stats only") rather
    than FULL -- spec §23: "The default mode should be lightweight."
    """

    mode: CaptureMode = CaptureMode.SUMMARY

    # Scope filters. None means "no restriction on this axis" -- NOT
    # "nothing allowed"; that's what mode=NONE is for. An empty tuple
    # WOULD mean "nothing allowed on this axis" if you passed one, but
    # there's no reason to construct a CapturePolicy that way instead of
    # just using mode=NONE, so this is noted rather than special-cased.
    components: Optional[tuple[ComponentType, ...]] = None
    layers: Optional[tuple[int, ...]] = None
    tokens: Optional[tuple[int, ...]] = None
    # Forward-compatible scope filters for Phase 4/5 (attention heads,
    # MLP neurons) -- accepted here now so CapturePolicy's shape doesn't
    # need to change when those phases land, but nothing in Phase 2's
    # KV-cache bridge reads them yet (KV events aren't per-head or
    # per-neuron).
    heads: Optional[tuple[int, ...]] = None
    neurons: Optional[tuple[int, ...]] = None

    # Required when mode=SAMPLED (validated in __post_init__ below):
    # the fraction of in-scope events to keep, systematically (every
    # Nth), not randomly -- randomness would make tests non-deterministic
    # for no real benefit here. N = round(1/sample_rate).
    sample_rate: Optional[float] = None

    def __post_init__(self) -> None:
        if self.mode == CaptureMode.SAMPLED:
            if self.sample_rate is None:
                raise ValueError(
                    "CapturePolicy(mode=SAMPLED) requires an explicit sample_rate -- "
                    "there is no principled default rate to silently assume (spec §30: "
                    "never guess). Pass e.g. sample_rate=0.1 to keep roughly 1 in 10 events."
                )
            if not (0.0 < self.sample_rate <= 1.0):
                raise ValueError(f"sample_rate must be in (0, 1], got {self.sample_rate!r}")

    def allows_component_type(self, component_type: ComponentType) -> bool:
        if self.mode == CaptureMode.NONE:
            return False
        if self.components is None:
            return True
        return component_type in self.components

    def allows_layer(self, layer: Optional[int]) -> bool:
        if self.mode == CaptureMode.NONE:
            return False
        if self.layers is None or layer is None:
            return True
        return layer in self.layers

    def allows_token(self, token_pos: Optional[int]) -> bool:
        if self.mode == CaptureMode.NONE:
            return False
        if self.tokens is None or token_pos is None:
            return True
        return token_pos in self.tokens

    @property
    def keeps_full_detail(self) -> bool:
        """False for SUMMARY (thinned per-event detail) and for NONE
        (which keeps nothing at all, so "full detail" is a contradiction
        in terms for it, not merely a moot question) -- everything that
        reaches this point for SAMPLED/SELECTED/FULL keeps full per-event
        statistics; the difference between those three is which/how many
        events reach this point at all, not how much of each is kept."""
        return self.mode not in (CaptureMode.SUMMARY, CaptureMode.NONE)

    def sample_keep_every(self) -> Optional[int]:
        """For SAMPLED mode: keep every Nth in-scope event. None for
        every other mode (this question doesn't apply to them)."""
        if self.mode != CaptureMode.SAMPLED:
            return None
        assert self.sample_rate is not None  # guaranteed by __post_init__
        return max(1, round(1.0 / self.sample_rate))
