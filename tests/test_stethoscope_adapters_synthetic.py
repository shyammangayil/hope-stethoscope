"""Tests for stethoscope.adapters.synthetic -- SyntheticAdapter, wrapping
the existing `examples/synthetic_model.py` fixture (architectural-
evolution spec §24's "Synthetic Transformer" validation tier).

Imports `examples.synthetic_model` bare, the same way `core.hooks` etc.
are imported bare throughout the rest of `tests/` -- `examples/` is a
real package (`examples/__init__.py` exists) and the repo root is on
`sys.path` whenever pytest is invoked as `python -m pytest` from the repo
root, exactly like every other existing test file already relies on.
"""
from __future__ import annotations

from examples.synthetic_model import SyntheticTransformer, build_demo_model

from stethoscope.adapters.synthetic import SyntheticAdapter
from stethoscope.ids import ComponentType
from stethoscope.registry import ComponentRegistry


def test_manifest_matches_synthetic_transformer_constructor_args():
    t = SyntheticTransformer(num_layers=3, num_heads=5, head_dim=7, vocab_size=250, seed=1)
    m = SyntheticAdapter(t, model_id="unit-test-synth").build_manifest()

    assert m.model_id == "unit-test-synth"
    assert m.framework == "synthetic"
    assert m.num_layers == 3
    assert m.attention.num_query_heads == 5
    assert m.attention.num_kv_heads == 5  # MHA -- verified against project_qkv/attend's construction, not assumed
    assert m.attention.head_dim == 7
    assert m.attention.kind == "mha"
    assert m.embedding.vocab_size == 250
    assert m.embedding.hidden_dim == 5 * 7
    assert m.dtype == "float32"
    assert m.device == "cpu"


def test_manifest_honestly_reports_no_mlp_no_norm_no_positional_encoding():
    m = SyntheticAdapter(SyntheticTransformer()).build_manifest()
    assert m.mlp.kind is None
    assert m.normalization.kind is None
    assert m.attention.positional_scheme == "none"  # confirmed absent, not "unknown"
    assert len(m.notes) >= 2  # explains why, rather than leaving it silently blank


def test_adapter_accepts_hookable_model_wrapper_not_just_bare_transformer():
    hookable = build_demo_model(seed=0)  # SyntheticHookableModel, not SyntheticTransformer
    m = SyntheticAdapter(hookable).build_manifest()
    # build_demo_model's own documented defaults: 4 layers x 4 heads x 16 head_dim, vocab 1000
    assert m.num_layers == 4
    assert m.attention.num_query_heads == 4
    assert m.attention.head_dim == 16
    assert m.embedding.vocab_size == 1000
    assert m.cache.implementation == "NumpyKVCache"  # this model DOES have a real cache, unlike MLP/norm


def test_parameter_count_matches_actual_weight_array_sizes():
    t = SyntheticTransformer(num_layers=2, num_heads=2, head_dim=4, vocab_size=10, seed=0)
    m = SyntheticAdapter(t).build_manifest()
    hidden = 2 * 4
    expected = (10 * hidden) + (hidden * 10) + 2 * (4 * hidden * hidden)  # embed + unembed + 2 layers * (Wq,Wk,Wv,Wo)
    assert m.parameter_count == expected


def test_registry_from_synthetic_manifest_has_attention_and_cache_but_no_mlp_or_norm():
    m = SyntheticAdapter(build_demo_model(seed=0)).build_manifest()
    reg = ComponentRegistry.from_manifest(m)
    types_present = {c.type for c in reg}
    assert ComponentType.MLP not in types_present
    assert ComponentType.NORM not in types_present
    assert ComponentType.ATTENTION in types_present
    assert ComponentType.KV_CACHE in types_present  # NumpyKVCache genuinely exists on this model
    assert ComponentType.RESIDUAL in types_present


def test_build_registry_convenience_method_matches_from_manifest():
    adapter = SyntheticAdapter(SyntheticTransformer(num_layers=1, num_heads=2, head_dim=4))
    via_convenience = adapter.build_registry()
    via_explicit = ComponentRegistry.from_manifest(adapter.build_manifest())
    assert {str(c.id) for c in via_convenience} == {str(c.id) for c in via_explicit}
