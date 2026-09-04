"""Tests for stethoscope.adapters.huggingface -- fast unit tests against
plain mock config objects. No `transformers`/`torch` import anywhere in
this file, on purpose: `manifest_from_config()` only needs attribute
access, and this file is the evidence for that claim. The real,
offline-constructed-model tier (Gemma3 + Llama) lives in
`test_stethoscope_adapters_huggingface_real.py`, gated behind
`pytest.importorskip("transformers")` the same way
`tests/test_real_gemma3_offline.py` already gates its own real-model
tests. Architectural-evolution spec §4.1.
"""
from __future__ import annotations

import json

from stethoscope.adapters.huggingface import manifest_from_config
from stethoscope.ids import ComponentID
from stethoscope.registry import ComponentRegistry


class _MockConfig:
    """A plain object with HF-config-shaped attributes -- deliberately
    NOT a `transformers.PretrainedConfig`, to prove `manifest_from_config()`
    only needs attribute access, nothing transformers-specific."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _llama_shaped_config(**overrides) -> _MockConfig:
    defaults = dict(
        vocab_size=32000, hidden_size=64, intermediate_size=172,
        num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=8,
        hidden_act="silu", rms_norm_eps=1e-5, max_position_embeddings=2048,
        rope_theta=10000.0, tie_word_embeddings=False,
        architectures=["LlamaForCausalLM"],
    )
    defaults.update(overrides)
    return _MockConfig(**defaults)


def test_basic_field_extraction_from_llama_shaped_mock():
    m = manifest_from_config(_llama_shaped_config(), model_id="mock-llama")
    assert m.model_id == "mock-llama"
    assert m.framework == "huggingface"
    assert m.architecture == "LlamaForCausalLM"
    assert m.num_layers == 4
    assert m.attention.num_query_heads == 8
    assert m.attention.num_kv_heads == 8
    assert m.attention.kind == "mha"
    assert m.attention.head_dim == 64 // 8  # not explicit on this mock -- derived from hidden_size/num_heads
    assert m.attention.positional_scheme == "rope"  # rope_theta present
    assert m.normalization.kind == "rmsnorm"  # rms_norm_eps present, not layer_norm_eps
    assert m.normalization.epsilon == 1e-5
    assert m.embedding.vocab_size == 32000
    assert m.embedding.tied_embeddings is False
    assert m.context_length == 2048


def test_gqa_detected_from_differing_kv_heads():
    cfg = _llama_shaped_config(num_attention_heads=8, num_key_value_heads=2)
    m = manifest_from_config(cfg, model_id="mock-gqa")
    assert m.attention.kind == "gqa"
    assert m.attention.num_kv_heads == 2


def test_mqa_detected_when_single_kv_head():
    cfg = _llama_shaped_config(num_attention_heads=8, num_key_value_heads=1)
    m = manifest_from_config(cfg, model_id="mock-mqa")
    assert m.attention.kind == "mqa"


def test_missing_num_key_value_heads_falls_back_to_mha_not_unknown():
    # Plenty of real configs (GPT2-family) have no separate KV-head-count
    # field at all -- absence should mean "assume MHA" (the only
    # architecturally sound reading of "no separate KV head count
    # exists"), not "unknown".
    cfg = _llama_shaped_config()
    del cfg.num_key_value_heads
    m = manifest_from_config(cfg, model_id="mock-no-kv-heads")
    assert m.attention.num_kv_heads == m.attention.num_query_heads == 8
    assert m.attention.kind == "mha"


def test_gpt2_style_naming_variants_are_understood():
    cfg = _MockConfig(
        n_layer=6, n_head=4, n_embd=32, n_inner=128,
        activation_function="gelu_new", layer_norm_epsilon=1e-5,
        n_positions=1024, vocab_size=500, architectures=["GPT2LMHeadModel"],
    )
    m = manifest_from_config(cfg, model_id="mock-gpt2")
    assert m.num_layers == 6
    assert m.attention.num_query_heads == 4
    assert m.mlp.hidden_dim == 32
    assert m.mlp.intermediate_dim == 128
    assert m.mlp.activation == "gelu_new"
    assert m.normalization.kind == "layernorm"  # layer_norm_epsilon present, not rms_norm_eps
    assert m.normalization.epsilon == 1e-5
    assert m.context_length == 1024


def test_mlp_gated_left_none_without_override_and_explained_in_notes():
    m = manifest_from_config(_llama_shaped_config(), model_id="x")
    assert m.mlp.gated is None
    assert m.mlp.kind is None
    assert any("mlp.gated" in note for note in m.notes)


def test_mlp_gated_override_sets_kind_and_gated_consistently():
    m = manifest_from_config(_llama_shaped_config(), model_id="x", mlp_gated=True)
    assert m.mlp.gated is True
    assert m.mlp.kind == "gated"
    assert not any("mlp.gated" in note for note in m.notes)  # no longer unexplained -- an override was given

    m2 = manifest_from_config(_llama_shaped_config(), model_id="x", mlp_gated=False)
    assert m2.mlp.gated is False
    assert m2.mlp.kind == "dense"


def test_get_text_config_is_called_when_present_on_a_composite_config():
    text_cfg = _llama_shaped_config()

    class _CompositeConfig:
        """Stands in for a multimodal composite config (e.g. the real
        Gemma3Config) whose top-level object does NOT expose
        num_hidden_layers etc. directly -- only via get_text_config()."""

        def get_text_config(self):
            return text_cfg

    m = manifest_from_config(_CompositeConfig(), model_id="mock-composite")
    assert m.num_layers == text_cfg.num_hidden_layers
    assert m.attention.num_query_heads == text_cfg.num_attention_heads
    # architecture name comes from the TEXT config's `architectures` (via
    # get_text_config()), not the composite wrapper's own class name
    assert m.architecture == "LlamaForCausalLM"


def test_plain_config_without_get_text_config_still_works():
    # Most configs don't define get_text_config() at all in older
    # transformers versions -- the hasattr() guard must fall back to
    # using the config object directly rather than erroring.
    cfg = _llama_shaped_config()
    assert not hasattr(cfg, "get_text_config")
    m = manifest_from_config(cfg, model_id="x")
    assert m.num_layers == 4


def test_layer_types_preserved_as_tuple_not_list():
    cfg = _llama_shaped_config(
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "sliding_attention"]
    )
    m = manifest_from_config(cfg, model_id="x")
    assert m.attention.layer_types == (
        "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
    )
    assert isinstance(m.attention.layer_types, tuple)


def test_registry_uses_layer_types_from_config_for_attention_flavor():
    cfg = _llama_shaped_config(
        num_hidden_layers=2, layer_types=["sliding_attention", "full_attention"], sliding_window=16,
    )
    m = manifest_from_config(cfg, model_id="x")
    reg = ComponentRegistry.from_manifest(m)
    assert reg.get(ComponentID.attention(0)).extra["attention_flavor"] == "sliding_attention"
    assert reg.get(ComponentID.attention(1)).extra["attention_flavor"] == "full_attention"


def test_missing_fields_stay_none_not_guessed():
    m = manifest_from_config(_MockConfig(), model_id="empty")  # nothing set at all
    assert m.num_layers is None
    assert m.attention.num_query_heads is None
    assert m.attention.kind is None  # can't derive mha/gqa/mqa without head counts
    assert m.embedding.vocab_size is None
    assert m.normalization.kind is None
    assert m.architecture == "_MockConfig"  # falls back to the config's own class name, not a guessed "real" name


def test_manifest_from_mock_config_is_json_serializable():
    m = manifest_from_config(_llama_shaped_config(), model_id="x")
    json.dumps(m.to_dict())  # must not raise
