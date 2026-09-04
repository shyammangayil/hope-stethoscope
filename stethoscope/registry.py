"""stethoscope/registry.py
===========================
ComponentRegistry: canonical catalog of a model's semantically meaningful
components (architectural-evolution spec §6,
`docs/ARCHITECTURE_EVOLUTION_SPEC.md`).

`ComponentRegistry.from_manifest(manifest)` is a **pure function of a
ModelManifest** -- it does not touch a live model or `nn.Module` tree at
all. This is the one real design choice in this module worth being
explicit about (see the architecture delta report §13): "what components
does this architecture have" (this module, Phase 1) is deliberately
decoupled from "how do I find the actual `nn.Module` that implements
component X at runtime" (Phase 2's Observer, not built yet). That split
sidesteps the `named_modules()` fragility spec §34 explicitly warns
against, for this phase, at the cost of a documented, accepted
trade-off: a registry built this way describes what the *manifest*
implies exists, which is not a runtime guarantee that every one of those
components is physically present in exactly that shape on a given
checkpoint (delta report §12, risk 4). Binding a registered ComponentID
to a live tensor -- and failing loudly with UNAVAILABLE (spec §30) when
that binding doesn't hold -- is Phase 2's job, not this module's.

Individual neurons are deliberately NOT pre-populated (spec §22/23: a
real model's `intermediate_size` can be in the thousands, so eagerly
enumerating one ComponentID per neuron per layer would bloat the
registry for a component nothing consumes yet). Use
`ComponentID.neuron(layer, idx)` on demand instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Optional

from stethoscope.ids import ATTENTION_PART_TYPES, MLP_PART_TYPES, ComponentID, ComponentType
from stethoscope.manifest import ModelManifest

# The four canonical residual-stream tap points per layer, matching the
# common pre-norm decoder block structure (input -> attn -> add -> norm ->
# mlp -> add). Spec §10 names residual instrumentation a priority
# capability; this is a documented Phase 1 default scope, not an attempt
# to auto-discover arbitrary architecture-specific boundaries -- see the
# delta report §13.
_RESIDUAL_POINTS = ("pre_attention", "post_attention", "pre_mlp", "post_mlp")


@dataclass(frozen=True)
class ComponentInfo:
    """One catalog entry: a ComponentID plus its type and any metadata
    worth attaching at discovery time (e.g. which attention flavor a
    layer uses, for Gemma3-style heterogeneous sliding/global layers)."""

    id: ComponentID
    type: ComponentType
    layer: Optional[int] = None
    extra: Mapping[str, Any] = field(default_factory=dict)


class ComponentRegistry:
    """An enumerable, queryable catalog of ComponentInfo, keyed by
    ComponentID. Construct via `from_manifest()`, not directly, unless
    you're building a registry by hand for a test."""

    def __init__(
        self,
        components: Iterable[ComponentInfo],
        *,
        manifest: Optional[ModelManifest] = None,
    ) -> None:
        by_id: dict[ComponentID, ComponentInfo] = {}
        for c in components:
            if c.id in by_id:
                raise ValueError(f"duplicate ComponentID in registry: {c.id}")
            by_id[c.id] = c
        self._by_id = by_id
        # Kept for provenance/debuggability -- "what manifest produced
        # this registry" -- not required for the registry to function.
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[ComponentInfo]:
        return iter(self._by_id.values())

    def __contains__(self, component_id: ComponentID) -> bool:
        return component_id in self._by_id

    def get(self, component_id: ComponentID) -> ComponentInfo:
        try:
            return self._by_id[component_id]
        except KeyError:
            raise KeyError(f"no such component in registry: {component_id}") from None

    def by_type(self, component_type: ComponentType) -> list[ComponentInfo]:
        return [c for c in self._by_id.values() if c.type == component_type]

    def by_layer(self, layer: int) -> list[ComponentInfo]:
        return [c for c in self._by_id.values() if c.layer == layer]

    @classmethod
    def from_manifest(cls, manifest: ModelManifest) -> "ComponentRegistry":
        components: list[ComponentInfo] = []

        components.append(ComponentInfo(id=ComponentID.model(), type=ComponentType.MODEL))
        components.append(
            ComponentInfo(
                id=ComponentID.embedding(),
                type=ComponentType.EMBEDDING,
                extra={
                    "vocab_size": manifest.embedding.vocab_size,
                    "hidden_dim": manifest.embedding.hidden_dim,
                },
            )
        )

        num_layers = manifest.num_layers or 0
        layer_types = manifest.attention.layer_types
        has_cache = manifest.cache.implementation is not None
        mlp_parts = ("gate", "up", "activation", "down") if manifest.mlp.gated else ("up", "activation", "down")

        for layer_idx in range(num_layers):
            components.append(
                ComponentInfo(id=ComponentID.block(layer_idx), type=ComponentType.BLOCK, layer=layer_idx)
            )

            layer_flavor = layer_types[layer_idx] if layer_types and layer_idx < len(layer_types) else None
            flavor_extra = {"attention_flavor": layer_flavor} if layer_flavor else {}

            components.append(
                ComponentInfo(
                    id=ComponentID.attention(layer_idx),
                    type=ComponentType.ATTENTION,
                    layer=layer_idx,
                    extra=flavor_extra,
                )
            )
            for part, ctype in ATTENTION_PART_TYPES.items():
                components.append(
                    ComponentInfo(id=ComponentID.attention_part(layer_idx, part), type=ctype, layer=layer_idx)
                )

            if has_cache:
                components.append(
                    ComponentInfo(
                        id=ComponentID.kv_cache(layer_idx),
                        type=ComponentType.KV_CACHE,
                        layer=layer_idx,
                        extra=flavor_extra,
                    )
                )

            # Gated on `mlp.kind` being known at all -- not just on
            # `gated` -- because "no information" and "confirmed absent"
            # are different claims (spec §30). The synthetic fixture in
            # examples/synthetic_model.py is the concrete case this
            # matters for: it is attention+residual only, by design, with
            # no MLP and no normalization layers at all (see its module
            # docstring). An adapter for it must leave `mlp.kind`/
            # `normalization.kind` as None -- not guess "dense" or
            # "rmsnorm" -- and this registry must then correctly emit no
            # MLP/Norm components for it, rather than fabricate an
            # up/activation/down block that doesn't exist on that model.
            if manifest.mlp.kind is not None:
                components.append(
                    ComponentInfo(id=ComponentID.mlp(layer_idx), type=ComponentType.MLP, layer=layer_idx)
                )
                for part in mlp_parts:
                    components.append(
                        ComponentInfo(
                            id=ComponentID.mlp_part(layer_idx, part), type=MLP_PART_TYPES[part], layer=layer_idx
                        )
                    )

            # Position-qualified norms only when the manifest states
            # pre/post explicitly; one generic, unqualified per-layer Norm
            # when normalization is known to exist but its position isn't;
            # nothing at all when normalization's presence itself is
            # unknown/absent (see the MLP comment above -- same reasoning).
            if manifest.normalization.kind is not None:
                if manifest.normalization.position == "pre":
                    norm_positions: tuple[Optional[str], ...] = ("pre_attention", "pre_mlp")
                elif manifest.normalization.position == "post":
                    norm_positions = ("post_attention", "post_mlp")
                else:
                    norm_positions = (None,)
                for position in norm_positions:
                    components.append(
                        ComponentInfo(id=ComponentID.norm(layer_idx, position), type=ComponentType.NORM, layer=layer_idx)
                    )

            for point in _RESIDUAL_POINTS:
                components.append(
                    ComponentInfo(id=ComponentID.residual(layer_idx, point), type=ComponentType.RESIDUAL, layer=layer_idx)
                )

        components.append(ComponentInfo(id=ComponentID.lm_head(), type=ComponentType.LM_HEAD))
        components.append(ComponentInfo(id=ComponentID.logits(), type=ComponentType.LOGITS))

        return cls(components, manifest=manifest)
