"""stethoscope -- model-wide Transformer observability, intervention, and
causal-experiment framework.

This package is the "common architecture" the KV-cache subsystem
(`core/`, `compression/`, `harness/`, `integrations/`) and the
weight-compression subsystem (`hope_core/`) both sit above, per
`docs/ARCHITECTURE_EVOLUTION_SPEC.md` §2. It is built in phases; see
`TASKS.md` at the repo root for current status.

Phase 1 (component identity and architecture discovery):

    stethoscope.ids       ComponentID, ComponentType
    stethoscope.manifest  ModelManifest and sub-manifests
    stethoscope.registry  ComponentRegistry
    stethoscope.adapters  ModelAdapter implementations (synthetic, HF)

Phase 2 (generic observation layer -- KV-cache only so far; see
stethoscope.observer's module docstring for exactly what is and isn't
built yet):

    stethoscope.events    ObservationEvent, ObservationPhase
    stethoscope.capture   CaptureMode, CapturePolicy
    stethoscope.storage   ObservationEventSink (InMemory / SQLite)
    stethoscope.observer  Observer -- bridges core.hooks.CacheTap's
                           existing KVEvent stream into ObservationEvents

Import boundary: every module above except `stethoscope.observer`
imports nothing from `core`, `compression`, `harness`, `integrations`, or
`hope_core`, and nothing in those packages imports from `stethoscope` --
see the architecture delta report §10/§13 for why that was a deliberate,
structural guarantee in Phase 1. `stethoscope.observer` is the one,
documented exception: Phase 2's own job (spec §26) is bridging into the
existing KV-cache mechanism, and it does so by calling
`core.hooks.CacheTap`/`core.storage.EventSink` exactly as an existing
caller already does, without editing either module -- see
`stethoscope/observer.py`'s module docstring for the full reasoning.
Importing `stethoscope` itself still does not require `core`/`torch`/
`transformers` to be installed or attach anything to any model; that
import only happens lazily, inside `Observer.kv_cache_sink()`, the
moment it's actually called.

Naming note: `hope_core.report.StethoscopeReport` is a different,
pre-existing, unrelated class (a static weight-compression report). See
that module's docstring and the delta report §12 for the disambiguation.
"""
from __future__ import annotations

__all__ = [
    "ComponentID",
    "ComponentType",
    "ModelManifest",
    "AttentionManifest",
    "MLPManifest",
    "NormalizationManifest",
    "EmbeddingManifest",
    "CacheManifest",
    "ComponentInfo",
    "ComponentRegistry",
    "ObservationPhase",
    "ObservationEvent",
    "ModelAdapter",
    "SyntheticAdapter",
    "HFTransformersAdapter",
    "manifest_from_config",
    "CaptureMode",
    "CapturePolicy",
    "ObservationEventSink",
    "InMemoryObservationEventSink",
    "SQLiteObservationEventSink",
    "Observer",
]

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
from stethoscope.events import ObservationEvent, ObservationPhase
from stethoscope.adapters.base import ModelAdapter
from stethoscope.adapters.synthetic import SyntheticAdapter
from stethoscope.adapters.huggingface import HFTransformersAdapter, manifest_from_config
from stethoscope.capture import CaptureMode, CapturePolicy
from stethoscope.storage import InMemoryObservationEventSink, ObservationEventSink, SQLiteObservationEventSink
from stethoscope.observer import Observer
