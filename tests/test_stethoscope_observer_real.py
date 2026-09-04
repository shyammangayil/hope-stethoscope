"""Real, offline-constructed-model validation for stethoscope.observer
(architectural-evolution spec §24's "Real Hugging Face model" tier).

Reuses the exact `_tiny_real_gemma3()` construction and
`model.generate(..., past_key_values=cache, ...)` call pattern
`tests/test_real_gemma3_offline.py` already established and proved out,
with one line changed: the sink passed to `CacheTap` comes from
`Observer.kv_cache_sink()` instead of a plain `core.storage` sink.
Skipped, not failed, if `transformers` isn't installed -- same
convention as every other real-model tier in this repo.
"""
from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")
import torch  # noqa: E402

from core.hooks import CacheTap, verify_capture_integrity  # noqa: E402
from core.storage import InMemoryEventSink  # noqa: E402

from stethoscope.capture import CaptureMode, CapturePolicy  # noqa: E402
from stethoscope.ids import ComponentID  # noqa: E402
from stethoscope.observer import Observer  # noqa: E402


def _tiny_real_gemma3(seed: int = 0):
    """Identical construction to tests/test_real_gemma3_offline.py's own
    _tiny_real_gemma3() (and stethoscope's own Phase 1 real-model test) --
    reused, not reimplemented."""
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    config = Gemma3TextConfig(
        vocab_size=200, hidden_size=32, intermediate_size=64,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, sliding_window=16,
    )
    torch.manual_seed(seed)
    model = Gemma3ForCausalLM(config)
    model.eval()
    return model, config


def test_observing_a_real_gemma3_generate_call_produces_kv_cache_events():
    from transformers import DynamicCache

    model, config = _tiny_real_gemma3(seed=1)
    input_ids = torch.randint(0, config.vocab_size, (1, 6), generator=torch.Generator().manual_seed(1))

    observer = Observer(model_id="tiny-gemma3", policy=CapturePolicy(mode=CaptureMode.FULL))
    inner = InMemoryEventSink()
    cache = CacheTap(DynamicCache(config=config), observer.kv_cache_sink(inner), run_id=observer.experiment_id)

    with torch.no_grad():
        out = model.generate(input_ids=input_ids, past_key_values=cache, max_new_tokens=5, do_sample=False, use_cache=True)

    assert out.shape == (1, 11)  # 6 prompt + 5 generated, matching test_real_gemma3_offline.py's own check

    # The real cache is still a real DynamicCache; CacheTap's own
    # integrity check (against its OWN get_seq_length()) still passes --
    # the bridge changes nothing about CacheTap's own guarantees.
    integrity = verify_capture_integrity(cache, inner, observer.experiment_id, config.num_hidden_layers)
    assert integrity["all_match"], integrity

    events = observer.events
    assert len(events) > 0
    assert {e.layer for e in events} == set(range(8))
    assert all(e.model_id == "tiny-gemma3" for e in events)


def test_real_gemma3_component_ids_include_the_sliding_global_layer_split():
    """Confirms the Observer's translated events sit on the exact same
    ComponentID grammar for a REAL heterogeneous-attention model, not
    just the uniform synthetic one -- layer 5 is "full_attention" per
    Gemma3TextConfig's own layer_types (see the Phase 1 real-model tests
    for the source of that fact); this test doesn't re-derive it, just
    confirms the KV-cache events for that layer use the right ID."""
    from transformers import DynamicCache

    model, config = _tiny_real_gemma3(seed=2)
    input_ids = torch.randint(0, config.vocab_size, (1, 4), generator=torch.Generator().manual_seed(2))

    observer = Observer(model_id="x")
    cache = CacheTap(DynamicCache(config=config), observer.kv_cache_sink(), run_id=observer.experiment_id)

    with torch.no_grad():
        model.generate(input_ids=input_ids, past_key_values=cache, max_new_tokens=2, do_sample=False, use_cache=True)

    layer5_events = [e for e in observer.events if e.layer == 5]
    assert layer5_events
    assert layer5_events[0].component_id == str(ComponentID.kv_cache(5))


def test_capture_policy_layer_filter_works_against_a_real_model():
    from transformers import DynamicCache

    model, config = _tiny_real_gemma3(seed=3)
    input_ids = torch.randint(0, config.vocab_size, (1, 4), generator=torch.Generator().manual_seed(3))

    observer = Observer(model_id="x", policy=CapturePolicy(mode=CaptureMode.SELECTED, layers=(0, 5)))
    cache = CacheTap(DynamicCache(config=config), observer.kv_cache_sink(), run_id=observer.experiment_id)

    with torch.no_grad():
        model.generate(input_ids=input_ids, past_key_values=cache, max_new_tokens=2, do_sample=False, use_cache=True)

    assert {e.layer for e in observer.events} == {0, 5}
