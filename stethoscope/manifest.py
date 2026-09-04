"""stethoscope/manifest.py
==========================
ModelManifest: canonical, machine-readable architecture representation
(architectural-evolution spec §5, `docs/ARCHITECTURE_EVOLUTION_SPEC.md`).

Every field below is Optional. A field is None when the adapter that
built this manifest could not determine it -- never a guess. `notes`
records *why*, for anything worth explaining (why `sliding_window` is
None on a model that plausibly has one; why `parameter_count` wasn't
computed for a config-only manifest with no live model attached). This
mirrors the honesty discipline `harness.evaluator.InfrastructureUnavailable`
and the shape-changing compression plugins in `compression/plugins.py`
already use elsewhere in this repository for a related problem (see the
architecture delta report §7) -- applied here to architecture discovery
instead of evaluation.

Serialization: `to_dict()` on every dataclass here returns a plain,
JSON-safe dict (Enums become their `.value`, tuples become lists). A
manifest is meant to become part of a reproducibility record eventually
(spec §19 -- model identifier, framework versions, etc. travel with an
experiment), so JSON-round-trippability is a real requirement here, not
a convenience.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


def _asdict_safe(obj: Any) -> Any:
    """Like dataclasses.asdict(), but Enum members become their `.value`
    (plain dataclasses.asdict() leaves them as Enum instances, which
    json.dumps rejects by default) and tuples become lists (JSON has no
    tuple type)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _asdict_safe(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [_asdict_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _asdict_safe(v) for k, v in obj.items()}
    return obj


@dataclass
class AttentionManifest:
    """Spec §5 "Attention" fields. `kind`, when known, is one of "mha"
    (num_query_heads == num_kv_heads), "gqa" (num_kv_heads > 1 and <
    num_query_heads), or "mqa" (num_kv_heads == 1)."""

    kind: Optional[str] = None
    num_query_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    # "rope" | "absolute" | "alibi" | "none" (confirmed: no positional
    # encoding at all -- a real, meaningful value, not a stand-in for
    # "unknown") | None (unknown -- adapter could not determine this).
    positional_scheme: Optional[str] = None
    # None here is genuinely ambiguous between "confirmed global
    # attention, no window" and "unknown" -- there is no int sentinel for
    # "not applicable" distinct from "not determined". An adapter that
    # can positively confirm global-only attention should say so in
    # `ModelManifest.notes` rather than rely on readers inferring it from
    # this field alone.
    sliding_window: Optional[int] = None
    # Per-layer attention "flavor" when heterogeneous across layers (e.g.
    # Gemma 3's alternating sliding/global layers) -- one entry per layer,
    # or None if uniform/unknown. Length, when set, must equal num_layers.
    layer_types: Optional[tuple[str, ...]] = None

    def to_dict(self) -> dict:
        return _asdict_safe(self)


@dataclass
class MLPManifest:
    """Spec §5 "MLP" fields. `kind` is "gated" (SwiGLU/GeGLU-style, has a
    separate gate projection) or "dense" (a single up-projection),
    matching `gated` below (kept as a separate bool for convenience;
    `kind` and `gated` are always consistent when both are known)."""

    kind: Optional[str] = None
    hidden_dim: Optional[int] = None
    intermediate_dim: Optional[int] = None
    activation: Optional[str] = None
    gated: Optional[bool] = None

    def to_dict(self) -> dict:
        return _asdict_safe(self)


@dataclass
class NormalizationManifest:
    """Spec §5 "Normalization" fields. `position` is "pre" (norm before
    the sublayer, the modern default) or "post" (norm after, the
    original Transformer paper's convention)."""

    kind: Optional[str] = None  # "layernorm" | "rmsnorm" | None
    position: Optional[str] = None  # "pre" | "post" | None
    epsilon: Optional[float] = None

    def to_dict(self) -> dict:
        return _asdict_safe(self)


@dataclass
class EmbeddingManifest:
    """Spec §5 "Embedding/output" fields (the output/LM-head projection
    dimensions are implied by `hidden_dim`/`vocab_size` plus
    `tied_embeddings`; there is no separate output-only manifest since a
    tied LM head has no independent shape to record)."""

    vocab_size: Optional[int] = None
    hidden_dim: Optional[int] = None
    tied_embeddings: Optional[bool] = None

    def to_dict(self) -> dict:
        return _asdict_safe(self)


@dataclass
class CacheManifest:
    """Spec §5 "Cache" fields."""

    implementation: Optional[str] = None
    num_layers: Optional[int] = None
    strategy: Optional[str] = None  # "dynamic" | "static" | None

    def to_dict(self) -> dict:
        return _asdict_safe(self)


@dataclass
class ModelManifest:
    """Spec §5's top-level manifest. `model_id` and `framework` are the
    only required fields -- every architectural fact below them is
    Optional by design (see module docstring)."""

    model_id: str
    framework: str  # "synthetic" | "huggingface" | "pytorch" | ...

    architecture: Optional[str] = None
    modalities: tuple[str, ...] = ("text",)
    num_layers: Optional[int] = None
    parameter_count: Optional[int] = None
    dtype: Optional[str] = None
    device: Optional[str] = None
    context_length: Optional[int] = None
    tokenizer: Optional[str] = None

    attention: AttentionManifest = field(default_factory=AttentionManifest)
    mlp: MLPManifest = field(default_factory=MLPManifest)
    normalization: NormalizationManifest = field(default_factory=NormalizationManifest)
    embedding: EmbeddingManifest = field(default_factory=EmbeddingManifest)
    cache: CacheManifest = field(default_factory=CacheManifest)

    # Why any of the above are None, when that's worth explaining (spec
    # §30's UNAVAILABLE discipline, applied here as free-text alongside
    # the structured None). Never used to smuggle in a value that
    # couldn't be determined -- see unavailable_fields below for the
    # structured version of the same idea.
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return _asdict_safe(self)

    @property
    def unavailable_fields(self) -> list[str]:
        """Dotted names of every top-level or sub-manifest field
        currently None -- e.g. ["dtype", "attention.sliding_window"]. A
        quick, structured way to check "what couldn't be determined"
        without parsing `notes` prose. Empty on a fully-determined
        manifest."""
        out: list[str] = []
        for f in dataclasses.fields(self):
            if f.name == "notes":
                continue
            val = getattr(self, f.name)
            if val is None:
                out.append(f.name)
            elif dataclasses.is_dataclass(val):
                for sub_f in dataclasses.fields(val):
                    if getattr(val, sub_f.name) is None:
                        out.append(f"{f.name}.{sub_f.name}")
        return out
