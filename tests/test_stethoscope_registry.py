"""Tests for stethoscope.registry -- ComponentRegistry
(architectural-evolution spec §6).
"""
from __future__ import annotations

import pytest

from stethoscope.ids import ComponentID, ComponentType
from stethoscope.manifest import (
    AttentionManifest,
    CacheManifest,
    EmbeddingManifest,
    MLPManifest,
    ModelManifest,
    NormalizationManifest,
)
from stethoscope.registry import ComponentInfo, ComponentRegistry


def _full_manifest(num_layers: int = 2) -> ModelManifest:
    """A manifest with every enumeration-gating field positively known,
    so the registry should emit attention, kv_cache, mlp (gated), norm
    (pre), and residual components for every layer."""
    return ModelManifest(
        model_id="test-model",
        framework="test",
        num_layers=num_layers,
        attention=AttentionManifest(kind="mha", num_query_heads=4, num_kv_heads=4, head_dim=8),
        mlp=MLPManifest(kind="gated", gated=True, hidden_dim=32, intermediate_dim=64),
        normalization=NormalizationManifest(kind="rmsnorm", position="pre"),
        embedding=EmbeddingManifest(vocab_size=100, hidden_dim=32),
        cache=CacheManifest(implementation="DynamicCache", num_layers=num_layers, strategy="dynamic"),
    )


def _expected_ids_for_full_manifest(num_layers: int = 2) -> set[str]:
    ids = {"model", "model/embedding", "model/lm_head", "model/logits"}
    for i in range(num_layers):
        ids.add(f"model/layer/{i}")
        ids.add(f"model/layer/{i}/attention")
        for part in ("q", "k", "v", "scores", "probs", "context", "o"):
            ids.add(f"model/layer/{i}/attention/{part}")
        ids.add(f"model/layer/{i}/kv_cache")
        ids.add(f"model/layer/{i}/mlp")
        for part in ("gate", "up", "activation", "down"):
            ids.add(f"model/layer/{i}/mlp/{part}")
        ids.add(f"model/layer/{i}/norm/pre_attention")
        ids.add(f"model/layer/{i}/norm/pre_mlp")
        for point in ("pre_attention", "post_attention", "pre_mlp", "post_mlp"):
            ids.add(f"model/layer/{i}/residual/{point}")
    return ids


def test_from_manifest_produces_exact_expected_component_set():
    reg = ComponentRegistry.from_manifest(_full_manifest(num_layers=2))
    actual = {str(c.id) for c in reg}
    expected = _expected_ids_for_full_manifest(num_layers=2)
    assert actual == expected
    # Pinned count so a future change to the enumeration logic is visible
    # as a failing assertion here, not just "the two sets still agree
    # with each other."
    assert len(reg) == len(expected) == 46


def test_from_manifest_omits_mlp_norm_kv_cache_when_absent_synthetic_style():
    # Matches SyntheticAdapter's honest reporting for a model that
    # genuinely has none of these -- mlp/normalization/cache left at
    # their all-None defaults, not guessed.
    m = ModelManifest(
        model_id="synthetic-like",
        framework="synthetic",
        num_layers=1,
        attention=AttentionManifest(kind="mha", num_query_heads=4, num_kv_heads=4, head_dim=16),
    )
    reg = ComponentRegistry.from_manifest(m)
    types_present = {c.type for c in reg}
    assert ComponentType.MLP not in types_present
    assert ComponentType.MLP_GATE not in types_present
    assert ComponentType.NORM not in types_present
    assert ComponentType.KV_CACHE not in types_present
    # but attention, residual, block, model, embedding, lm_head, logits
    # are all still architecturally real for this model and must remain
    assert ComponentType.ATTENTION in types_present
    assert ComponentType.RESIDUAL in types_present
    assert ComponentType.BLOCK in types_present
    assert ComponentID.model() in reg
    assert ComponentID.lm_head() in reg


def test_dense_mlp_omits_gate_but_keeps_up_activation_down():
    m = _full_manifest(num_layers=1)
    m.mlp.gated = False
    m.mlp.kind = "dense"
    reg = ComponentRegistry.from_manifest(m)
    ids = {str(c.id) for c in reg}
    assert "model/layer/0/mlp/gate" not in ids
    assert "model/layer/0/mlp/up" in ids
    assert "model/layer/0/mlp/activation" in ids
    assert "model/layer/0/mlp/down" in ids


def test_post_norm_uses_post_attention_and_post_mlp_positions():
    m = _full_manifest(num_layers=1)
    m.normalization.position = "post"
    reg = ComponentRegistry.from_manifest(m)
    ids = {str(c.id) for c in reg}
    assert "model/layer/0/norm/post_attention" in ids
    assert "model/layer/0/norm/post_mlp" in ids
    assert "model/layer/0/norm/pre_attention" not in ids


def test_unknown_norm_position_emits_one_generic_unqualified_norm():
    m = _full_manifest(num_layers=1)
    m.normalization.position = None  # kind is still known ("rmsnorm"), just not pre/post
    reg = ComponentRegistry.from_manifest(m)
    ids = {str(c.id) for c in reg}
    assert "model/layer/0/norm" in ids
    assert "model/layer/0/norm/pre_attention" not in ids
    assert "model/layer/0/norm/post_attention" not in ids


def test_attention_flavor_recorded_in_extra_for_heterogeneous_layers():
    m = _full_manifest(num_layers=2)
    m.attention.layer_types = ("sliding_attention", "full_attention")
    reg = ComponentRegistry.from_manifest(m)
    assert reg.get(ComponentID.attention(0)).extra["attention_flavor"] == "sliding_attention"
    assert reg.get(ComponentID.attention(1)).extra["attention_flavor"] == "full_attention"
    assert reg.get(ComponentID.kv_cache(0)).extra["attention_flavor"] == "sliding_attention"


def test_get_raises_key_error_for_missing_component():
    reg = ComponentRegistry.from_manifest(_full_manifest(num_layers=1))
    with pytest.raises(KeyError):
        reg.get(ComponentID.layer(999))


def test_contains_len_and_iter_agree():
    reg = ComponentRegistry.from_manifest(_full_manifest(num_layers=1))
    assert ComponentID.model() in reg
    assert ComponentID.layer(999) not in reg
    assert len(reg) == len(list(iter(reg))) > 0


def test_by_type_and_by_layer_filter_correctly():
    reg = ComponentRegistry.from_manifest(_full_manifest(num_layers=2))
    attentions = reg.by_type(ComponentType.ATTENTION)
    assert len(attentions) == 2
    assert {c.layer for c in attentions} == {0, 1}

    layer0_components = reg.by_layer(0)
    assert layer0_components  # non-empty
    assert all(c.layer == 0 for c in layer0_components)
    # global components (model/embedding/lm_head/logits) have layer=None
    # and must never appear under a specific layer's filter
    assert ComponentID.model() not in {c.id for c in layer0_components}


def test_registry_rejects_duplicate_component_ids():
    dup = ComponentID.model()
    with pytest.raises(ValueError):
        ComponentRegistry(
            [
                ComponentInfo(id=dup, type=ComponentType.MODEL),
                ComponentInfo(id=dup, type=ComponentType.MODEL),
            ]
        )


def test_registry_records_source_manifest_for_provenance():
    m = _full_manifest(num_layers=1)
    reg = ComponentRegistry.from_manifest(m)
    assert reg.manifest is m
