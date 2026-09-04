"""Tests for stethoscope.capture -- CaptureMode / CapturePolicy
(architectural-evolution spec §22, §23).
"""
from __future__ import annotations

import pytest

from stethoscope.capture import CaptureMode, CapturePolicy
from stethoscope.ids import ComponentType


def test_default_policy_is_lightweight_summary_mode_no_filters():
    p = CapturePolicy()
    assert p.mode == CaptureMode.SUMMARY
    assert p.components is None
    assert p.layers is None
    assert p.tokens is None
    assert not p.keeps_full_detail


def test_none_mode_disallows_everything_regardless_of_filters():
    p = CapturePolicy(mode=CaptureMode.NONE, components=(ComponentType.KV_CACHE,), layers=(0,), tokens=(0,))
    assert not p.allows_component_type(ComponentType.KV_CACHE)  # even though it's explicitly listed
    assert not p.allows_layer(0)
    assert not p.allows_token(0)


def test_no_filters_means_unrestricted_not_restricted():
    p = CapturePolicy(mode=CaptureMode.FULL)
    assert p.allows_component_type(ComponentType.KV_CACHE)
    assert p.allows_component_type(ComponentType.ATTENTION)
    assert p.allows_layer(0)
    assert p.allows_layer(999)
    assert p.allows_token(0)
    assert p.allows_token(999)


def test_explicit_filters_restrict_correctly():
    p = CapturePolicy(mode=CaptureMode.SELECTED, components=(ComponentType.KV_CACHE,), layers=(0, 2), tokens=(5,))
    assert p.allows_component_type(ComponentType.KV_CACHE)
    assert not p.allows_component_type(ComponentType.ATTENTION)
    assert p.allows_layer(0)
    assert p.allows_layer(2)
    assert not p.allows_layer(1)
    assert p.allows_token(5)
    assert not p.allows_token(6)


def test_layer_and_token_filters_ignore_none_values_rather_than_rejecting():
    # A component with no meaningful layer (e.g. embedding, layer=None)
    # or no meaningful token position shouldn't be rejected just because
    # a layers=/tokens= filter happens to be set for OTHER components.
    p = CapturePolicy(mode=CaptureMode.SELECTED, layers=(0,), tokens=(5,))
    assert p.allows_layer(None)
    assert p.allows_token(None)


def test_keeps_full_detail_true_for_every_mode_except_summary():
    assert CapturePolicy(mode=CaptureMode.FULL).keeps_full_detail
    assert CapturePolicy(mode=CaptureMode.SELECTED).keeps_full_detail
    assert CapturePolicy(mode=CaptureMode.SAMPLED, sample_rate=0.5).keeps_full_detail
    assert not CapturePolicy(mode=CaptureMode.SUMMARY).keeps_full_detail
    assert not CapturePolicy(mode=CaptureMode.NONE).keeps_full_detail  # doesn't matter -- nothing reaches this point anyway


def test_sampled_mode_requires_sample_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        CapturePolicy(mode=CaptureMode.SAMPLED)


def test_sampled_mode_rejects_out_of_range_rate():
    with pytest.raises(ValueError):
        CapturePolicy(mode=CaptureMode.SAMPLED, sample_rate=0.0)
    with pytest.raises(ValueError):
        CapturePolicy(mode=CaptureMode.SAMPLED, sample_rate=1.5)


def test_sample_keep_every_matches_rate():
    assert CapturePolicy(mode=CaptureMode.SAMPLED, sample_rate=1.0).sample_keep_every() == 1
    assert CapturePolicy(mode=CaptureMode.SAMPLED, sample_rate=0.5).sample_keep_every() == 2
    assert CapturePolicy(mode=CaptureMode.SAMPLED, sample_rate=0.1).sample_keep_every() == 10
    assert CapturePolicy(mode=CaptureMode.SAMPLED, sample_rate=0.34).sample_keep_every() == 3  # round(1/0.34) == 3


def test_sample_keep_every_is_none_for_non_sampled_modes():
    assert CapturePolicy(mode=CaptureMode.FULL).sample_keep_every() is None
    assert CapturePolicy(mode=CaptureMode.SUMMARY).sample_keep_every() is None
    assert CapturePolicy(mode=CaptureMode.NONE).sample_keep_every() is None
