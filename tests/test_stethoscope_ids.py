"""Tests for stethoscope.ids -- ComponentID / ComponentType
(architectural-evolution spec §6).
"""
from __future__ import annotations

import pytest

from stethoscope.ids import ATTENTION_PART_TYPES, MLP_PART_TYPES, ComponentID, ComponentType


def test_model_id_is_single_part():
    assert str(ComponentID.model()) == "model"


def test_layer_id_matches_spec_example():
    assert str(ComponentID.layer(17)) == "model/layer/17"


def test_attention_part_ids_match_spec_examples():
    assert str(ComponentID.attention(17)) == "model/layer/17/attention"
    assert str(ComponentID.attention_part(17, "q")) == "model/layer/17/attention/q"
    assert str(ComponentID.attention_part(17, "k")) == "model/layer/17/attention/k"
    assert str(ComponentID.attention_part(17, "v")) == "model/layer/17/attention/v"


def test_mlp_id_matches_spec_example():
    assert str(ComponentID.mlp(17)) == "model/layer/17/mlp"


def test_mlp_neuron_id_matches_spec_example():
    assert str(ComponentID.neuron(17, 183)) == "model/layer/17/mlp/neuron/183"


def test_residual_id_matches_spec_example():
    assert str(ComponentID.residual(17, "post_attention")) == "model/layer/17/residual/post_attention"


def test_parse_round_trips_for_every_factory():
    candidates = [
        ComponentID.model(),
        ComponentID.embedding(),
        ComponentID.layer(0),
        ComponentID.layer(17),
        ComponentID.attention(3),
        ComponentID.attention_part(3, "q"),
        ComponentID.attention_part(3, "scores"),
        ComponentID.kv_cache(3),
        ComponentID.mlp(3),
        ComponentID.mlp_part(3, "gate"),
        ComponentID.neuron(3, 42),
        ComponentID.norm(3),
        ComponentID.norm(3, "pre_attention"),
        ComponentID.residual(3, "pre_mlp"),
        ComponentID.lm_head(),
        ComponentID.logits(),
    ]
    for cid in candidates:
        assert ComponentID.parse(str(cid)) == cid


def test_equality_and_hashing():
    a = ComponentID.attention_part(5, "q")
    b = ComponentID.attention_part(5, "q")
    c = ComponentID.attention_part(5, "k")
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert len({a, b, c}) == 2  # a and b collapse in a set


def test_parent_strips_one_string_segment_at_a_time_not_one_semantic_level():
    # .parent is structural truncation (like pathlib.PurePath.parent), not
    # semantic-boundary walking -- factory methods add a different number
    # of parts at each step (layer() adds 2, attention() adds 1), so
    # chained .parent calls don't land on another canonical, factory-
    # producible ID at every step. See ids.py's ComponentID.parent
    # docstring, and use .layer_id (tested below) for "enclosing layer".
    cid = ComponentID.attention_part(5, "q")  # model/layer/5/attention/q (5 parts)
    assert cid.parent == ComponentID.attention(5)  # model/layer/5/attention (4 parts)
    assert cid.parent.parent == ComponentID.layer(5)  # model/layer/5 (3 parts)
    grandparent = cid.parent.parent.parent  # model/layer (2 parts) -- strips just the "5"
    assert str(grandparent) == "model/layer"
    assert grandparent != ComponentID.model()
    assert cid.parent.parent.parent.parent == ComponentID.model()  # model (1 part)
    assert cid.parent.parent.parent.parent.parent is None


def test_layer_id_reaches_enclosing_layer_in_one_step_regardless_of_id_kind():
    assert ComponentID.attention_part(5, "q").layer_id == ComponentID.layer(5)
    assert ComponentID.neuron(5, 42).layer_id == ComponentID.layer(5)
    assert ComponentID.residual(5, "pre_mlp").layer_id == ComponentID.layer(5)
    assert ComponentID.layer(5).layer_id == ComponentID.layer(5)  # a layer ID is its own enclosing layer
    assert ComponentID.model().layer_id is None
    assert ComponentID.embedding().layer_id is None
    assert ComponentID.lm_head().layer_id is None


def test_layer_index_extracted_correctly():
    assert ComponentID.attention_part(17, "q").layer_index == 17
    assert ComponentID.neuron(3, 999).layer_index == 3
    assert ComponentID.model().layer_index is None
    assert ComponentID.embedding().layer_index is None
    assert ComponentID.lm_head().layer_index is None


def test_rejects_empty_parts_tuple():
    with pytest.raises(ValueError):
        ComponentID(())


def test_rejects_part_containing_slash():
    with pytest.raises(ValueError):
        ComponentID(("model", "layer/17"))


def test_rejects_empty_string_part():
    with pytest.raises(ValueError):
        ComponentID(("model", ""))


def test_attention_part_rejects_unknown_part_name():
    with pytest.raises(ValueError):
        ComponentID.attention_part(0, "not_a_real_part")


def test_mlp_part_rejects_unknown_part_name():
    with pytest.raises(ValueError):
        ComponentID.mlp_part(0, "not_a_real_part")


def test_residual_rejects_unknown_point_name():
    with pytest.raises(ValueError):
        ComponentID.residual(0, "not_a_real_point")


def test_component_type_values_match_spec_section_6_exactly():
    # Spec §6's canonical vocabulary, verbatim -- a tripwire test: if
    # someone "cleans up" a member name or value later, this fails loudly
    # instead of silently drifting from the spec.
    expected = {
        "MODEL": "Model", "EMBEDDING": "Embedding", "BLOCK": "Block",
        "NORM": "Norm", "RESIDUAL": "Residual", "ATTENTION": "Attention",
        "ATTENTION_Q": "Attention.Q", "ATTENTION_K": "Attention.K",
        "ATTENTION_V": "Attention.V", "ATTENTION_SCORES": "Attention.Scores",
        "ATTENTION_PROBABILITY": "Attention.Probability",
        "ATTENTION_CONTEXT": "Attention.Context", "ATTENTION_O": "Attention.O",
        "KV_CACHE": "KVCache", "MLP": "MLP", "MLP_GATE": "MLP.Gate",
        "MLP_UP": "MLP.Up", "MLP_ACTIVATION": "MLP.Activation",
        "MLP_DOWN": "MLP.Down", "NEURON": "Neuron", "LM_HEAD": "LMHead",
        "LOGITS": "Logits",
    }
    for name, value in expected.items():
        assert ComponentType[name].value == value
    assert {m.name for m in ComponentType} == set(expected)


def test_component_type_is_str_subclass_for_json_friendliness():
    assert ComponentType.ATTENTION_Q == "Attention.Q"
    assert isinstance(ComponentType.ATTENTION_Q, str)


def test_attention_and_mlp_part_type_maps_cover_all_parts():
    assert set(ATTENTION_PART_TYPES) == {"q", "k", "v", "scores", "probs", "context", "o"}
    assert set(MLP_PART_TYPES) == {"gate", "up", "activation", "down"}
