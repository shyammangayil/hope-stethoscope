"""Tests for stethoscope.observer -- Observer + the KV-cache bridge
(architectural-evolution spec §26 Phase 2).

This tier uses `core.hooks.CacheTap` against the SYNTHETIC model's real
`NumpyKVCache` (`examples/synthetic_model.py`) -- genuinely the same
`CacheTap` class real Gemma3 observation uses (see
`test_stethoscope_observer_real.py`), not a mock or a reimplementation,
because `NumpyKVCache.update()` matches the exact interface CacheTap
wraps (confirmed directly in its own docstring: "This is the object
core.hooks.CacheTap wraps"). `core.hooks`/`core.storage` themselves don't
import torch/numpy... at module scope for hooks, or torch/numpy at all
for storage (see each module's own docstring) -- so, in an environment
that genuinely lacks torch/transformers, this whole tier would still
import and run; this sandbox happens to have both installed regardless,
so that claim is stated from reading the code, not independently proven
by uninstalling anything here.
"""
from __future__ import annotations

from core.hooks import CacheTap
from core.storage import InMemoryEventSink

from examples.synthetic_model import build_demo_model

from stethoscope.capture import CaptureMode, CapturePolicy
from stethoscope.ids import ComponentID, ComponentType
from stethoscope.observer import Observer
from stethoscope.storage import InMemoryObservationEventSink


def _wrap_synthetic_model_cache(model, observer: Observer, inner_sink=None):
    """Wires CacheTap onto `model`'s own cache using `observer`'s bridge
    sink, and reassigns `model.cache` to the (in-place-mutated) traced
    object -- exactly the pattern `examples/run_demo.py` already uses for
    plain CacheTap, with one extra step (asking the Observer for the
    sink instead of constructing a plain EventSink directly)."""
    bridge_sink = observer.kv_cache_sink(inner_sink)
    model.cache = CacheTap(model.cache, bridge_sink, run_id=observer.experiment_id)
    return bridge_sink


# ------------------------------------------------------- basic end-to-end

def test_observing_a_synthetic_generate_call_produces_kv_cache_events():
    model = build_demo_model(seed=0)  # 4 layers x 4 heads x 16 head_dim
    observer = Observer(model_id="synthetic-test", policy=CapturePolicy(mode=CaptureMode.FULL))
    _wrap_synthetic_model_cache(model, observer)

    model.generate(prompt_ids=[1, 2, 3], n_new_tokens=2)

    events = observer.events
    assert len(events) > 0
    assert all(e.component_type == ComponentType.KV_CACHE.value for e in events)
    assert all(e.experiment_id == observer.experiment_id for e in events)
    assert all(e.model_id == "synthetic-test" for e in events)
    assert {e.layer for e in events} == {0, 1, 2, 3}  # all 4 layers observed
    # 3 prompt tokens + 2 generated tokens = 5 token positions per layer
    assert len({e.token_positions[0] for e in events if e.layer == 0}) == 5


def test_component_ids_match_the_registry_convention_from_phase_1():
    """The observer's translated events use the exact same ComponentID
    grammar Phase 1's ComponentRegistry already established -- not a
    parallel, independently-invented ID scheme."""
    model = build_demo_model(seed=0)
    observer = Observer(model_id="x")
    _wrap_synthetic_model_cache(model, observer)
    model.generate(prompt_ids=[1], n_new_tokens=1)

    layer2_events = [e for e in observer.events if e.layer == 2]
    assert layer2_events
    assert layer2_events[0].component_id == str(ComponentID.kv_cache(2))


# ---------------------------------------------- does not affect KVEvent path

def test_bridge_does_not_change_what_the_inner_kvevent_sink_receives():
    """The core compatibility claim from observer.py's module docstring,
    checked directly: an inner EventSink wired through the bridge sees
    IDENTICAL KVEvents to what it would see from a plain, un-bridged
    CacheTap call, given the same model weights and the same generation
    call. `.generate()`'s default RNG is deterministic (`rng=None` ->
    `np.random.default_rng(0)` inside `SyntheticHookableModel.generate`),
    so two independently-built models from the same seed produce
    identical token choices and therefore identical KVEvents."""
    plain_sink = InMemoryEventSink()
    plain_model = build_demo_model(seed=7)
    plain_model.cache = CacheTap(plain_model.cache, plain_sink, run_id="same-run")
    plain_model.generate(prompt_ids=[4, 5], n_new_tokens=3)
    plain_rows = plain_sink.read_all()

    bridged_inner = InMemoryEventSink()
    observer = Observer(model_id="x")
    bridged_model = build_demo_model(seed=7)  # same seed -> identical weights -> identical generation
    _wrap_synthetic_model_cache(bridged_model, observer, inner_sink=bridged_inner)
    bridged_model.generate(prompt_ids=[4, 5], n_new_tokens=3)
    bridged_rows = bridged_inner.read_all()

    assert len(plain_rows) == len(bridged_rows) > 0
    # Compare the fields that should be identical regardless of run_id
    # (run_id differs on purpose -- "same-run" vs. the Observer's own
    # generated experiment_id -- everything else must match exactly).
    for p, b in zip(plain_rows, bridged_rows):
        assert p["layer_idx"] == b["layer_idx"]
        assert p["token_pos"] == b["token_pos"]
        assert p["l2_norm"] == b["l2_norm"]
        assert p["kv_bytes"] == b["kv_bytes"]


# -------------------------------------------------------- CapturePolicy

def test_none_mode_captures_nothing():
    model = build_demo_model(seed=0)
    observer = Observer(model_id="x", policy=CapturePolicy(mode=CaptureMode.NONE))
    _wrap_synthetic_model_cache(model, observer)
    model.generate(prompt_ids=[1, 2], n_new_tokens=2)
    assert observer.events == []


def test_layer_filter_restricts_which_layers_are_observed():
    model = build_demo_model(seed=0)
    observer = Observer(model_id="x", policy=CapturePolicy(mode=CaptureMode.SELECTED, layers=(0, 2)))
    _wrap_synthetic_model_cache(model, observer)
    model.generate(prompt_ids=[1, 2], n_new_tokens=1)
    assert {e.layer for e in observer.events} == {0, 2}


def test_token_filter_restricts_which_token_positions_are_observed():
    model = build_demo_model(seed=0)
    observer = Observer(model_id="x", policy=CapturePolicy(mode=CaptureMode.SELECTED, tokens=(0,)))
    _wrap_synthetic_model_cache(model, observer)
    model.generate(prompt_ids=[1, 2, 3], n_new_tokens=2)  # positions 0..4 exist
    assert {e.token_positions[0] for e in observer.events} == {0}


def test_component_type_filter_excluding_kv_cache_captures_nothing():
    model = build_demo_model(seed=0)
    observer = Observer(model_id="x", policy=CapturePolicy(mode=CaptureMode.SELECTED, components=(ComponentType.ATTENTION,)))
    _wrap_synthetic_model_cache(model, observer)
    model.generate(prompt_ids=[1], n_new_tokens=1)
    assert observer.events == []


def test_summary_mode_thins_statistics_to_headline_stat_only():
    model = build_demo_model(seed=0)
    observer = Observer(model_id="x", policy=CapturePolicy(mode=CaptureMode.SUMMARY))
    _wrap_synthetic_model_cache(model, observer)
    model.generate(prompt_ids=[1], n_new_tokens=1)
    assert observer.events
    for e in observer.events:
        assert set(e.statistics.keys()) == {"l2_norm"}


def test_full_mode_keeps_every_kv_statistic():
    model = build_demo_model(seed=0)
    observer = Observer(model_id="x", policy=CapturePolicy(mode=CaptureMode.FULL))
    _wrap_synthetic_model_cache(model, observer)
    model.generate(prompt_ids=[1], n_new_tokens=1)
    assert observer.events
    for e in observer.events:
        assert set(e.statistics.keys()) == {"l2_norm", "attn_entropy", "variance", "latency_ms", "kv_bytes"}


def test_sampled_mode_keeps_every_nth_in_scope_event():
    model = build_demo_model(seed=0)
    # 1 layer only, to make the counting unambiguous; sample_rate=0.5 -> keep every 2nd
    observer = Observer(
        model_id="x",
        policy=CapturePolicy(mode=CaptureMode.SAMPLED, layers=(0,), sample_rate=0.5),
    )
    _wrap_synthetic_model_cache(model, observer)
    model.generate(prompt_ids=[1, 2, 3, 4], n_new_tokens=4)  # 8 token positions on layer 0
    kept_positions = sorted(e.token_positions[0] for e in observer.events)
    assert kept_positions == [0, 2, 4, 6]  # every 2nd of the 8 in-scope events, 1-indexed-then-halved


# --------------------------------------------------------------- storage

def test_observer_writes_to_an_observation_event_sink_when_given_one():
    model = build_demo_model(seed=0)
    obs_sink = InMemoryObservationEventSink()
    observer = Observer(model_id="x", sink=obs_sink)
    _wrap_synthetic_model_cache(model, observer)
    model.generate(prompt_ids=[1], n_new_tokens=1)

    assert len(obs_sink.read_all()) == len(observer.events) > 0
    assert obs_sink.read_all(experiment_id=observer.experiment_id) == obs_sink.read_all()


def test_observer_events_is_a_snapshot_not_a_live_view():
    model = build_demo_model(seed=0)
    observer = Observer(model_id="x")
    _wrap_synthetic_model_cache(model, observer)
    model.generate(prompt_ids=[1], n_new_tokens=1)

    snapshot = observer.events
    snapshot.clear()
    assert len(observer.events) > 0  # unaffected by mutating the returned list


# ----------------------------------------------------------------- observe()

def test_observe_dispatches_kv_cache_to_kv_cache_sink():
    observer = Observer(model_id="x")
    sink_via_observe = observer.observe(ComponentType.KV_CACHE)
    assert hasattr(sink_via_observe, "write") and hasattr(sink_via_observe, "read_all")


def test_observe_raises_not_implemented_for_unbuilt_component_types():
    observer = Observer(model_id="x")
    for component_type in (ComponentType.ATTENTION, ComponentType.MLP, ComponentType.RESIDUAL):
        try:
            observer.observe(component_type)
            raise AssertionError(f"expected NotImplementedError for {component_type}")
        except NotImplementedError:
            pass
