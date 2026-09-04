"""stethoscope/observer.py
==========================
Observer: coordinates observation of a model's components against a
CapturePolicy, producing ObservationEvents (architectural-evolution spec
§26 Phase 2, `docs/ARCHITECTURE_EVOLUTION_SPEC.md`).

DELIBERATE BOUNDARY NOTE, read before touching this file: every other
module under `stethoscope/` (Phase 1's `ids.py`/`manifest.py`/
`registry.py`/`events.py`/`adapters/*`, and Phase 2's own `capture.py`/
`storage.py`) imports nothing from `core/`, `compression/`, `harness/`,
`integrations/`, or `hope_core/` -- that was a deliberate, structural
guarantee for Phase 1 (see the architecture delta report §10/§13): if
Phase 1 code can't reach those packages at all, it can't regress them,
full stop. THIS module is the one, documented exception, and that's by
design, not scope creep: spec §26 Phase 2's own job description is
"Migrate KV instrumentation onto this abstraction while preserving
compatibility" -- Phase 2 is explicitly where the KV-cache subsystem and
the new observability layer are supposed to start talking to each other.

What "preserving compatibility" means concretely here: this module
imports `core.hooks.CacheTap` and `core.storage.EventSink`/
`InMemoryEventSink` and calls them exactly as an existing caller already
does (`CacheTap(cache, sink, run_id=...)`) -- it does not edit a single
line of `core/hooks.py` or `core/storage.py` to make this work. The
bridge lives entirely on this side: `Observer.kv_cache_sink()` returns an
object that satisfies `core.storage.EventSink`'s Protocol (so `CacheTap`
can't tell the difference from a plain `SQLiteEventSink`) and, on every
`write(KVEvent)` call, ALSO translates that event into an
`ObservationEvent` and feeds it through this Observer's `CapturePolicy`.
Nothing about the existing `CacheTap`/`KVEvent`/`EventSink` code path
changes; an existing caller who ignores this module entirely sees no
difference at all. This is verified, not just claimed -- see
`tests/test_stethoscope_observer.py`'s "does not affect KVEvent writing"
tests, and the unmodified `core/hooks.py` tests in `tests/test_core.py`,
which stay green throughout (rerun as part of every full-suite check in
`TASKS.md`).

What this module does NOT do, on purpose: it does not implement
`with stethoscope.observe(model, components=[...]): output = model(...)`
(spec §7's exact pseudocode) as automatic, no-setup sugar. That would
require either auto-discovering which cache object an arbitrary
`model.generate()`/`forward()` call uses internally, or the Observer
injecting one -- both are real, unsolved, framework-specific design
questions (how a cache gets created/passed differs across HF generation
paths), and building speculative sugar around them now, before Phase
3/4/5 give a fuller picture of what other component types need from the
same API, risks exactly the "introduce arbitrary abstractions without
corresponding tests" failure spec §34 warns against. What IS built here
is the concrete, validated mechanism underneath: the caller constructs
their own cache (exactly as they already do for a plain `CacheTap` call
today), asks `Observer` for a sink, and passes it to `CacheTap`
themselves -- one extra line over today's existing usage, not a rewrite
of it. Residual/attention/MLP observation (Phase 3/4/5) is not
implemented here either -- `Observer.kv_cache_sink()` is the only capture
entry point this phase provides; requesting anything else raises
`NotImplementedError` rather than silently doing nothing.
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from stethoscope.capture import CapturePolicy
from stethoscope.events import ObservationEvent, ObservationPhase
from stethoscope.ids import ComponentID, ComponentType
from stethoscope.storage import InMemoryObservationEventSink, ObservationEventSink

if TYPE_CHECKING:  # pragma: no cover - type checking only, no runtime import
    from core.storage import EventSink, KVEvent


def _observation_event_from_kv_event(
    kv_event: "KVEvent", *, model_id: str, experiment_id: str, keep_full_detail: bool,
) -> ObservationEvent:
    """The actual KVEvent -> ObservationEvent field mapping. Lossless:
    every KVEvent field ends up somewhere in the result (mostly
    `statistics`/`provenance`), never silently dropped -- see the field
    comments below for where each one lands and why.
    """
    if keep_full_detail:
        statistics = {
            "l2_norm": kv_event.l2_norm,
            "attn_entropy": kv_event.attn_entropy,
            "variance": kv_event.variance,
            "latency_ms": kv_event.latency_ms,
            "kv_bytes": kv_event.kv_bytes,
        }
    else:
        # SUMMARY mode: thin to the headline stat only (spec §22).
        statistics = {"l2_norm": kv_event.l2_norm}

    return ObservationEvent(
        experiment_id=experiment_id,
        model_id=model_id,
        component_id=str(ComponentID.kv_cache(kv_event.layer_idx)),
        component_type=ComponentType.KV_CACHE.value,
        # CacheTap captures inside Cache.update(), after the real append
        # but mid-forward-pass (one update() call per layer, not one per
        # whole-model forward pass) -- INTERNAL_STATE, not PRE_/POST_FORWARD,
        # which are whole-model boundaries this event isn't one of.
        phase=ObservationPhase.INTERNAL_STATE.value,
        layer=kv_event.layer_idx,
        token_positions=(kv_event.token_pos,),
        timestep=None,  # KVEvent has no distinct notion of "generation step" separate from token_pos; left honestly unset rather than overloading token_pos to mean two things
        tensor_metadata=None,  # KVEvent never carried shape/dtype/device either -- nothing to report here that isn't already fabricated
        tensor_reference=None,
        statistics=statistics,
        provenance={
            "token_source": kv_event.token_source,
            "ts": kv_event.ts,
            "head_idx": kv_event.head_idx,
            "run_id": kv_event.run_id,
            **kv_event.extra,
        },
    )


class _KVEventBridgeSink:
    """An object satisfying `core.storage.EventSink`'s Protocol
    (write/read_all/close), constructed only by `Observer.kv_cache_sink()`
    -- not meant to be built directly. Forwards every `KVEvent` unchanged
    to `inner` (so existing KV-cache storage/querying is completely
    unaffected -- nothing is silently dropped or redirected away from
    where a caller already expects it), and also feeds each one to
    `observer` for `ObservationEvent` translation."""

    def __init__(self, *, inner: "EventSink", observer: "Observer") -> None:
        self._inner = inner
        self._observer = observer

    def write(self, event: "KVEvent") -> None:
        self._inner.write(event)
        self._observer._accept_kv_event(event)

    def read_all(self, run_id: Optional[str] = None) -> list:
        return self._inner.read_all(run_id)

    def close(self) -> None:
        self._inner.close()


class Observer:
    """Coordinates observation against a `CapturePolicy`, collecting
    `ObservationEvent`s into `self.events` (and, if `sink=` is given,
    also persisting them there via `stethoscope.storage`).

    Phase 2's actual, validated scope: KV-cache observation only, via
    `kv_cache_sink()` below -- see this module's docstring for why
    residual/attention/MLP observation (Phase 3/4/5) and the
    fully-automatic `with stethoscope.observe(model, ...)` context
    manager (spec §7) are both deliberately not built yet.
    """

    def __init__(
        self,
        *,
        model_id: str,
        experiment_id: Optional[str] = None,
        policy: Optional[CapturePolicy] = None,
        sink: Optional[ObservationEventSink] = None,
    ) -> None:
        self.model_id = model_id
        self.experiment_id = experiment_id or uuid.uuid4().hex[:8]
        self.policy = policy or CapturePolicy()
        self.sink = sink
        self._events: list[ObservationEvent] = []
        self._kv_candidate_count = 0  # for SAMPLED mode's "keep every Nth" counter -- counts events that already passed the scope filters, so the rate is "1 in N of what would otherwise be captured", not "1 in N of everything CacheTap ever sees"

    @property
    def events(self) -> list[ObservationEvent]:
        """A snapshot list -- mutating the returned list does not affect
        this Observer's internal state."""
        return list(self._events)

    def kv_cache_sink(self, inner_sink: Optional["EventSink"] = None) -> "EventSink":
        """Returns an `EventSink`-conforming object, ready to pass
        directly to `core.hooks.CacheTap(cache, this_sink, ...)` exactly
        as an existing caller already passes a plain
        `core.storage` sink -- the only change from today's usage is
        passing THIS sink instead of (or wrapping) their own.

        `inner_sink`: where `KVEvent`s themselves still go, completely
        unaffected by anything this Observer does -- defaults to a fresh
        `core.storage.InMemoryEventSink()` if not given, never silently
        drops the existing KVEvent stream. Pass your own (e.g. a real
        `core.storage.SQLiteEventSink`) to keep KV-cache persistence
        exactly as it already was in your existing code.
        """
        from core.storage import InMemoryEventSink  # local import: see module docstring -- core/ stays out of every OTHER stethoscope module's dependency surface

        inner = inner_sink if inner_sink is not None else InMemoryEventSink()
        return _KVEventBridgeSink(inner=inner, observer=self)

    def observe(self, component_type: ComponentType, **kwargs) -> "EventSink":
        """Generic entry point matching component_type to the right
        concrete capture mechanism. Phase 2 only has one:
        ComponentType.KV_CACHE -> kv_cache_sink(). Any other component
        type raises NotImplementedError -- explicitly, per spec §30
        ("never silently fabricate data... report UNAVAILABLE"), rather
        than returning a sink that would just never produce anything."""
        if component_type == ComponentType.KV_CACHE:
            return self.kv_cache_sink(**kwargs)
        raise NotImplementedError(
            f"Observer has no capture mechanism for {component_type.value} yet -- "
            f"only ComponentType.KV_CACHE is implemented in Phase 2. See this "
            f"module's docstring and TASKS.md for what Phase 3/4/5 need to add."
        )

    def _accept_kv_event(self, kv_event: "KVEvent") -> None:
        """Called by `_KVEventBridgeSink` for every `KVEvent` CacheTap
        writes. Applies this Observer's `CapturePolicy` (scope filters,
        then detail/sampling), and if the event survives, translates and
        stores it."""
        if not self.policy.allows_component_type(ComponentType.KV_CACHE):
            return
        if not self.policy.allows_layer(kv_event.layer_idx):
            return
        if not self.policy.allows_token(kv_event.token_pos):
            return

        keep_every = self.policy.sample_keep_every()
        if keep_every is not None:
            self._kv_candidate_count += 1
            if (self._kv_candidate_count - 1) % keep_every != 0:
                return  # sampled out -- but it DID pass scope filters, so it still counted above

        observation = _observation_event_from_kv_event(
            kv_event,
            model_id=self.model_id,
            experiment_id=self.experiment_id,
            keep_full_detail=self.policy.keeps_full_detail,
        )
        self._events.append(observation)
        if self.sink is not None:
            self.sink.write(observation)
