"""Tests for stethoscope.manifest -- ModelManifest and sub-manifests
(architectural-evolution spec §5).
"""
from __future__ import annotations

import json

from stethoscope.manifest import (
    AttentionManifest,
    CacheManifest,
    EmbeddingManifest,
    MLPManifest,
    ModelManifest,
    NormalizationManifest,
)


def test_minimal_manifest_has_all_optional_fields_none():
    m = ModelManifest(model_id="x", framework="test")
    assert m.architecture is None
    assert m.num_layers is None
    assert m.parameter_count is None
    assert m.dtype is None
    assert m.device is None
    assert m.context_length is None
    assert m.tokenizer is None
    assert m.notes == []
    assert m.modalities == ("text",)


def test_sub_manifests_default_to_all_fields_none():
    m = ModelManifest(model_id="x", framework="test")
    assert m.attention == AttentionManifest()
    assert m.mlp == MLPManifest()
    assert m.normalization == NormalizationManifest()
    assert m.embedding == EmbeddingManifest()
    assert m.cache == CacheManifest()


def test_to_dict_round_trips_through_json_when_fully_populated():
    m = ModelManifest(
        model_id="x",
        framework="huggingface",
        architecture="GemmaForCausalLM",
        num_layers=8,
        parameter_count=1000,
        dtype="float32",
        device="cpu",
        context_length=2048,
        tokenizer="gemma-tokenizer",
        attention=AttentionManifest(
            kind="gqa",
            num_query_heads=4,
            num_kv_heads=2,
            head_dim=8,
            positional_scheme="rope",
            sliding_window=16,
            layer_types=("sliding_attention", "full_attention"),
        ),
        mlp=MLPManifest(kind="gated", hidden_dim=32, intermediate_dim=64, activation="gelu", gated=True),
        normalization=NormalizationManifest(kind="rmsnorm", position="pre", epsilon=1e-6),
        embedding=EmbeddingManifest(vocab_size=1000, hidden_dim=32, tied_embeddings=True),
        cache=CacheManifest(implementation="DynamicCache", num_layers=8, strategy="dynamic"),
        notes=["a note"],
    )
    encoded = json.dumps(m.to_dict())  # must not raise -- the actual "is this JSON-safe" test
    decoded = json.loads(encoded)
    assert decoded["model_id"] == "x"
    assert decoded["attention"]["layer_types"] == ["sliding_attention", "full_attention"]  # tuple -> list
    assert decoded["mlp"]["gated"] is True
    assert decoded["notes"] == ["a note"]


def test_to_dict_round_trips_through_json_when_minimal():
    m = ModelManifest(model_id="x", framework="test")
    json.dumps(m.to_dict())  # must not raise even with every optional field at its default


def test_unavailable_fields_lists_every_none_field_by_dotted_name():
    m = ModelManifest(model_id="x", framework="test")
    fields = m.unavailable_fields
    assert "architecture" in fields
    assert "num_layers" in fields
    assert "parameter_count" in fields
    assert "attention.kind" in fields
    assert "attention.num_query_heads" in fields
    assert "mlp.kind" in fields
    assert "normalization.kind" in fields
    assert "embedding.vocab_size" in fields
    assert "cache.implementation" in fields
    assert "notes" not in fields  # notes is explanatory prose, not itself an unavailable fact
    assert "model_id" not in fields  # required field, always set
    assert "framework" not in fields  # required field, always set


def test_unavailable_fields_empty_when_every_field_is_set():
    m = ModelManifest(
        model_id="x",
        framework="huggingface",
        architecture="A",
        num_layers=1,
        parameter_count=1,
        dtype="float32",
        device="cpu",
        context_length=1,
        tokenizer="t",
        attention=AttentionManifest(
            kind="mha", num_query_heads=1, num_kv_heads=1, head_dim=1,
            positional_scheme="rope", sliding_window=0, layer_types=("full_attention",),
        ),
        mlp=MLPManifest(kind="dense", hidden_dim=1, intermediate_dim=1, activation="relu", gated=False),
        normalization=NormalizationManifest(kind="layernorm", position="pre", epsilon=1e-5),
        embedding=EmbeddingManifest(vocab_size=1, hidden_dim=1, tied_embeddings=False),
        cache=CacheManifest(implementation="x", num_layers=1, strategy="dynamic"),
    )
    assert m.unavailable_fields == []
