"""Tests for stethoscope.events -- ObservationEvent / ObservationPhase
(architectural-evolution spec §9).
"""
from __future__ import annotations

import json

from stethoscope.events import ObservationEvent, ObservationPhase


def test_observation_phase_values_match_spec_section_7():
    assert ObservationPhase.INPUT.value == "input"
    assert ObservationPhase.PRE_FORWARD.value == "pre_forward"
    assert ObservationPhase.INTERNAL_STATE.value == "internal_state"
    assert ObservationPhase.OUTPUT.value == "output"
    assert ObservationPhase.POST_FORWARD.value == "post_forward"
    assert ObservationPhase.BACKWARD.value == "backward"
    assert ObservationPhase.GRADIENT.value == "gradient"
    assert ObservationPhase.PARAMETER_UPDATE.value == "parameter_update"


def test_minimal_event_construction_leaves_optionals_unset():
    e = ObservationEvent(
        experiment_id="exp-1",
        model_id="m",
        component_id="model/layer/0/attention/q",
        component_type="Attention.Q",
        phase=ObservationPhase.INTERNAL_STATE.value,
    )
    assert e.layer is None
    assert e.token_positions is None
    assert e.tensor_metadata is None
    assert e.tensor_reference is None
    assert e.statistics is None
    assert e.provenance == {}


def test_to_dict_excludes_tensor_reference_by_default():
    e = ObservationEvent(
        experiment_id="exp-1",
        model_id="m",
        component_id="c",
        component_type="t",
        phase="internal_state",
        tensor_reference=object(),  # a deliberately non-JSON-safe sentinel
    )
    d = e.to_dict()
    assert "tensor_reference" not in d
    json.dumps(d)  # must not raise -- would raise if tensor_reference leaked in


def test_to_dict_includes_tensor_reference_when_explicitly_requested():
    sentinel = object()
    e = ObservationEvent(
        experiment_id="exp-1", model_id="m", component_id="c", component_type="t",
        phase="internal_state", tensor_reference=sentinel,
    )
    d = e.to_dict(include_tensor_reference=True)
    assert d["tensor_reference"] is sentinel


def test_to_dict_converts_token_positions_tuple_to_json_safe_list():
    e = ObservationEvent(
        experiment_id="exp-1", model_id="m", component_id="c", component_type="t",
        phase="internal_state", token_positions=(0, 1, 2),
    )
    d = e.to_dict()
    assert d["token_positions"] == [0, 1, 2]
    json.dumps(d)  # tuples aren't JSON-native; the converted list is


def test_to_dict_full_round_trip_through_json():
    e = ObservationEvent(
        experiment_id="exp-1",
        model_id="gemma-3-4b",
        component_id="model/layer/17/residual/post_attention",
        component_type="Residual",
        phase=ObservationPhase.INTERNAL_STATE.value,
        layer=17,
        token_positions=(4, 5),
        timestep=0,
        tensor_metadata={"shape": [2, 8], "dtype": "float32", "device": "cpu"},
        statistics={"l2_norm": 1.23},
        provenance={"git_commit": "abc123"},
    )
    decoded = json.loads(json.dumps(e.to_dict()))
    assert decoded["layer"] == 17
    assert decoded["token_positions"] == [4, 5]
    assert decoded["statistics"]["l2_norm"] == 1.23
    assert decoded["provenance"]["git_commit"] == "abc123"
