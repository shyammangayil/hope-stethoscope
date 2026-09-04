"""stethoscope/adapters/base.py
================================
ModelAdapter: the abstraction that lets the Stethoscope core work across
multiple Transformer implementations (architectural-evolution spec §4.1,
`docs/ARCHITECTURE_EVOLUTION_SPEC.md`).

Two responsibilities only, in this phase: produce a ModelManifest, and
produce a ComponentRegistry from it. Binding either to a live model's
actual `nn.Module` tree -- so that Phase 2's Observer can attach to a
specific component -- is deliberately not part of this interface yet; see
`stethoscope.registry`'s module docstring for why that split is a Phase 1
vs. Phase 2 boundary, not an oversight.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from stethoscope.manifest import ModelManifest
from stethoscope.registry import ComponentRegistry


class ModelAdapter(ABC):
    """Base class for a framework/family-specific adapter.

    Subclasses implement `build_manifest()`. `build_registry()` has a
    correct default (`ComponentRegistry.from_manifest(self.build_manifest())`)
    and should only be overridden for a concrete reason (e.g. caching the
    manifest to avoid recomputing it on repeated calls) -- neither
    concrete adapter in this phase needs to.
    """

    @abstractmethod
    def build_manifest(self) -> ModelManifest:
        """Produce a ModelManifest describing the wrapped model.

        Must never guess a field it cannot determine: leave it `None`,
        and if it's worth explaining why, add a line to
        `manifest.notes` (spec §30 -- report UNAVAILABLE, don't
        fabricate).
        """
        raise NotImplementedError

    def build_registry(self) -> ComponentRegistry:
        return ComponentRegistry.from_manifest(self.build_manifest())
