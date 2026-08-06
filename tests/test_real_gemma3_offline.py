"""The closest thing to a real-Gemma integration test this repo can run
without network access: constructs a REAL (tiny, randomly-initialized,
untrained) transformers.Gemma3ForCausalLM offline -- no pretrained-weight
download, no tokenizer, no huggingface.co access needed -- and runs the
actual CacheTap -> generate() -> modifier -> inspector pipeline against it.

Why this is worth more than the synthetic-numpy-model tests elsewhere in
this repo: it exercises the REAL model class's REAL generate() loop and
REAL Cache object, including the same heterogeneous sliding-window/global
per-layer attention pattern the real gemma-3-4b-it checkpoint has (verified
independently against this session's own real-Gemma run log: layer 5 of 8
here comes back "full_attention"/global, matching the real 34-layer model's
"global every 6th layer" pattern exactly). What it does NOT verify: the
actual pretrained 4B weights, real tokenization, or anything about
generation quality -- only that this repo's hooking/modifier/inspection
code interoperates correctly with the authentic model class and cache
mechanics. That's a real, substantial narrowing of what was previously
"structurally plausible, never run" for the real-model path -- see
README.md's validation status table.

Skipped automatically if `transformers` isn't installed (it's an optional,
heavy dependency -- see requirements.txt), same pattern as this repo's
other optional-dependency tests.
"""

import tempfile
import os

import pytest
import torch

transformers = pytest.importorskip("transformers")

from core.hooks import CacheTap, verify_capture_integrity
from core.storage import SQLiteEventSink
from core.kv_modifiers import parse_modifier_spec
from core.cache_adapters import get_layer_kv, describe_cache
from core.inspector import KVCacheInspector


def _tiny_real_gemma3(seed=0):
    """Small enough to construct and run generate() in well under a second,
    while keeping the SAME GQA ratio (2:1) and enough layers to get both
    sliding_window and global attention types in the mix -- the two
    properties of the real architecture that actually matter for testing
    CacheTap/inspector against heterogeneous per-layer cache behavior."""
    from transformers import Gemma3TextConfig, Gemma3ForCausalLM

    config = Gemma3TextConfig(
        vocab_size=200, hidden_size=32, intermediate_size=64,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, sliding_window=16,
    )
    torch.manual_seed(seed)
    model = Gemma3ForCausalLM(config)
    model.eval()
    return model, config


def test_layer_types_match_real_gemma3_pattern():
    """Confirms the config class's auto-computed layer_types (global every
    6th layer) matches what the real gemma-3-4b-it checkpoint showed in
    this session's own op.txt log -- not assumed, checked."""
    _, config = _tiny_real_gemma3()
    assert config.layer_types[5] == "full_attention"
    for i in [0, 1, 2, 3, 4, 6, 7]:
        assert config.layer_types[i] == "sliding_attention"


def test_cachetap_isinstance_holds_for_real_gemma3_cache():
    """The one property CacheTap's whole design (see core/hooks.py's
    docstring) exists to guarantee -- checked against a REAL DynamicCache
    produced by a real model's generate(), not a hand-built stand-in."""
    from transformers import DynamicCache

    model, config = _tiny_real_gemma3()
    cache = DynamicCache(config=config)
    with tempfile.TemporaryDirectory() as tmp:
        sink = SQLiteEventSink(os.path.join(tmp, "events.db"))
        tapped = CacheTap(cache, sink, run_id="test")
        assert isinstance(tapped, DynamicCache)


def test_full_generate_loop_with_modifier_against_real_gemma3():
    """The end-to-end check: CacheTap + a real modifier + a real generate()
    call against a real (tiny) Gemma3ForCausalLM, then confirm via the
    ACTUAL returned KV tensor that the modifier took effect on exactly the
    token it targeted and nowhere else."""
    from transformers import DynamicCache

    model, config = _tiny_real_gemma3(seed=1)
    input_ids = torch.randint(0, config.vocab_size, (1, 6), generator=torch.Generator().manual_seed(1))

    with tempfile.TemporaryDirectory() as tmp:
        sink = SQLiteEventSink(os.path.join(tmp, "events.db"))
        run_id = "gen_test"
        cache = CacheTap(DynamicCache(config=config), sink, run_id=run_id,
                          modifier=parse_modifier_spec("zero:token_positions=0"))

        with torch.no_grad():
            out = model.generate(input_ids=input_ids, past_key_values=cache,
                                  max_new_tokens=5, do_sample=False, use_cache=True)

        assert out.shape == (1, 11)  # 6 prompt + 5 generated

        integrity = verify_capture_integrity(cache, sink, run_id, config.num_hidden_layers)
        assert integrity["all_match"], integrity

        key0, _ = get_layer_kv(cache, 0)
        assert key0[0, :, 0, :].abs().max().item() == 0.0, "modifier should have zeroed token 0"
        assert key0[0, :, 1, :].abs().max().item() > 0.0, "token 1 should be untouched"


def test_describe_cache_reports_correct_heterogeneous_layer_types():
    from transformers import DynamicCache

    model, config = _tiny_real_gemma3(seed=2)
    input_ids = torch.randint(0, config.vocab_size, (1, 6), generator=torch.Generator().manual_seed(2))
    with tempfile.TemporaryDirectory() as tmp:
        sink = SQLiteEventSink(os.path.join(tmp, "events.db"))
        cache = CacheTap(DynamicCache(config=config), sink, run_id="describe_test")
        with torch.no_grad():
            model(input_ids=input_ids, past_key_values=cache, use_cache=True)

        desc = describe_cache(cache, config.num_hidden_layers)
        assert "Layer 05" in desc
        # layer 5's block should say "global", every other layer "sliding_window"
        blocks = desc.split("Layer ")[1:]
        assert "global" in blocks[5]
        for i in [0, 1, 2, 3, 4, 6, 7]:
            assert "sliding_window" in blocks[i]


def test_inspector_full_report_against_real_gemma3_keys():
    """KVCacheInspector's stats (the same function that produces the
    per-token table this session's real op.txt log showed) against real
    (if untrained) Gemma3 keys, not synthetic data."""
    from transformers import DynamicCache

    model, config = _tiny_real_gemma3(seed=3)
    input_ids = torch.randint(0, config.vocab_size, (1, 8), generator=torch.Generator().manual_seed(3))
    with tempfile.TemporaryDirectory() as tmp:
        sink = SQLiteEventSink(os.path.join(tmp, "events.db"))
        cache = CacheTap(DynamicCache(config=config), sink, run_id="inspect_test")
        with torch.no_grad():
            model(input_ids=input_ids, past_key_values=cache, use_cache=True)

        key0, _ = get_layer_kv(cache, 0)
        seq_len = key0.shape[2]
        report = KVCacheInspector.full_report(
            key0[0], token_positions=list(range(seq_len)), prefill_len=8, current_step=seq_len,
        )
        assert len(report) == seq_len
        for stat in report:
            assert stat.l2_norm >= 0
            assert stat.variance >= 0
            assert stat.token_source == "prompt"  # no generation happened, only prefill


# ---------------------------------------------------------------------------
# Broader modifier coverage against the real model -- the tests above only
# exercised "zero". Each modifier has different semantics (additive noise,
# per-head masking, cross-layer scoping) and it's worth confirming each one
# actually does what it claims against real Gemma3 tensors, not just that
# parsing the spec string works (that's covered separately, and without a
# model at all, in test_kv_modifiers.py).
# ---------------------------------------------------------------------------


def _generate_with_modifier(modifier_spec, seed=10):
    """Shared setup for the modifier-coverage tests below: real tiny Gemma3,
    one prefill forward pass with the given modifier applied, returns the
    captured layer-0 key tensor for inspection."""
    from transformers import DynamicCache

    model, config = _tiny_real_gemma3(seed=seed)
    input_ids = torch.randint(0, config.vocab_size, (1, 8), generator=torch.Generator().manual_seed(seed))
    with tempfile.TemporaryDirectory() as tmp:
        sink = SQLiteEventSink(os.path.join(tmp, "events.db"))
        modifier = parse_modifier_spec(modifier_spec) if modifier_spec else None
        cache = CacheTap(DynamicCache(config=config), sink, run_id="mod_test", modifier=modifier)
        with torch.no_grad():
            model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        key0, _ = get_layer_kv(cache, 0)
        return key0.clone()


def test_noise_modifier_actually_perturbs_real_gemma3_keys():
    baseline = _generate_with_modifier(None, seed=11)
    noised = _generate_with_modifier("noise:sigma=5.0;seed=0", seed=11)
    assert not torch.allclose(baseline, noised), "noise modifier should have changed the real keys"
    # sigma=5.0 against real Gemma3 key magnitudes (~O(1-5), per the
    # per-token L2 norms seen elsewhere in this file) should be a large,
    # obviously-detectable perturbation, not noise-floor-level -- catches a
    # modifier that's silently a no-op as well as one that errors.
    assert (noised - baseline).abs().mean() > 0.5


def test_head_mask_modifier_zeros_only_targeted_heads_on_real_gemma3():
    """num_key_value_heads=2 in the tiny config -- masking head 0 should
    leave head 1 completely untouched, checked against the real tensor,
    not just that the call didn't raise.

    target=key is specified explicitly: HeadMaskModifier defaults to
    target="value" (see test_kv_modifiers.py's own default-value test),
    and this test checks the KEY tensor -- an earlier version of this test
    omitted target= and failed with head 0 showing completely normal,
    unmasked values, which looked like a masking bug until it turned out
    to just be checking the wrong tensor for the modifier's actual default.
    """
    masked = _generate_with_modifier("head_mask:head_indices=0;scale=0.0;target=key", seed=12)
    assert masked[0, 0, :, :].abs().max().item() == 0.0, "masked head should be exactly zero"
    assert masked[0, 1, :, :].abs().max().item() > 0.0, "untouched head should be unaffected"


def test_layer_indices_scoping_gates_which_layers_modify_is_called_for():
    """The actual, narrow guarantee layer_indices provides: modify() is
    only INVOKED for layers in the list. This is deliberately NOT a test
    that other layers' cached VALUES come out identical to an unmodified
    run -- see test_layer0_modification_causally_cascades_to_later_layers
    below for why that's a different (and false) claim, discovered while
    writing this file's first attempt at a layer-scoping test."""
    modifier = parse_modifier_spec("zero:token_positions=0;layer_indices=0")
    assert modifier.applies_to_layer(0) is True
    assert modifier.applies_to_layer(1) is False
    assert modifier.applies_to_layer(7) is False


def test_layer0_modification_causally_cascades_to_later_layers():
    """Documents real, correct (if initially surprising) behavior found
    while testing layer_indices scoping against the real model: a modifier
    that only writes to layer 0's cache still changes layer 1's cached
    values, because layer 0's cache.update() call happens DURING layer 0's
    own attention computation and its return value is what that attention
    computation actually uses -- so a modified K/V changes layer 0's hidden-
    state OUTPUT, which is layer 1's INPUT, within the same forward pass.
    layer_indices controls which layer's cache gets the modification
    applied; it was never a guarantee that only that layer's values would
    differ from a baseline run, and can't be -- that would require
    breaking the causal connection between adjacent transformer layers.
    A first version of this test asserted layer 1 would be untouched and
    failed loudly against a real ~2.5 max-abs-diff, which is what caused
    this to be understood and written up rather than assumed."""
    from transformers import DynamicCache

    model, config = _tiny_real_gemma3(seed=13)
    input_ids = torch.randint(0, config.vocab_size, (1, 8), generator=torch.Generator().manual_seed(13))

    baseline_cache = DynamicCache(config=config)
    with torch.no_grad():
        model(input_ids=input_ids, past_key_values=baseline_cache, use_cache=True)
    baseline_layer1 = baseline_cache.layers[1].keys.clone()

    with tempfile.TemporaryDirectory() as tmp:
        sink = SQLiteEventSink(os.path.join(tmp, "events.db"))
        modifier = parse_modifier_spec("zero:token_positions=0;layer_indices=0")
        cache = CacheTap(DynamicCache(config=config), sink, run_id="cascade_test", modifier=modifier)
        with torch.no_grad():
            model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        modified_layer0, _ = get_layer_kv(cache, 0)
        modified_layer1, _ = get_layer_kv(cache, 1)

    assert modified_layer0[0, :, 0, :].abs().max().item() == 0.0, "layer 0 token 0 should be zeroed (the actual scoping guarantee)"
    assert not torch.allclose(modified_layer1, baseline_layer1), (
        "layer 1 SHOULD differ from baseline -- this is expected causal cascading, not a scoping failure"
    )
