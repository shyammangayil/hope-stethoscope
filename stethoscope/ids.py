"""stethoscope/ids.py
=====================
Stable, canonical identifiers for model components (architectural-evolution
spec §6, `docs/ARCHITECTURE_EVOLUTION_SPEC.md`).

A ComponentID names a *logical* place in a transformer's architecture --
"layer 17's attention Q projection" -- independent of any particular
nn.Module object, its Python class name, or its position in
`named_modules()`. Per spec §6: "Do not make identifiers dependent solely
on Python object names." Concretely here, that means IDs are built from
the canonical factory methods below rather than free-form string
concatenation scattered across call sites -- this module is the one place
the ID grammar is defined.

IDs are immutable, hashable (usable as dict keys / in sets), and
string-round-trippable: `ComponentID.parse(str(cid)) == cid` always holds
for any ComponentID this module can construct (see
tests/test_stethoscope_ids.py).

Deliberately NOT in this module: any binding from a ComponentID to a live
`nn.Module`/tensor. That is a runtime *observation* concern (Phase 2's
Observer), not an *identity* concern (Phase 1). A ComponentID can be
constructed and reasoned about even for a component that was never
observed, or that turns out not to exist on a particular checkpoint --
see ComponentRegistry's module docstring for how that distinction is
enforced.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional, Union

# Component kinds a caller may pass to attention_part()/mlp_part(), kept as
# plain string literals (not their own enum) because they are sub-selectors
# within one ComponentType, not component types themselves -- see
# ComponentType below for the actual controlled vocabulary.
_ATTENTION_PARTS = ("q", "k", "v", "scores", "probs", "context", "o")
_MLP_PARTS = ("gate", "up", "activation", "down")
_RESIDUAL_POINTS = ("pre_attention", "post_attention", "pre_mlp", "post_mlp")


class ComponentType(str, Enum):
    """Canonical component vocabulary, verbatim from spec §6.

    A `str` subclass so values compare/serialize as their spec-exact
    strings (``ComponentType.ATTENTION_Q == "Attention.Q"`` is True, and
    ``json.dumps({"type": ComponentType.ATTENTION_Q})`` needs no custom
    encoder) without giving up enum membership/exhaustiveness checks.
    """

    MODEL = "Model"
    EMBEDDING = "Embedding"
    BLOCK = "Block"
    NORM = "Norm"
    RESIDUAL = "Residual"
    ATTENTION = "Attention"
    ATTENTION_Q = "Attention.Q"
    ATTENTION_K = "Attention.K"
    ATTENTION_V = "Attention.V"
    ATTENTION_SCORES = "Attention.Scores"
    ATTENTION_PROBABILITY = "Attention.Probability"
    ATTENTION_CONTEXT = "Attention.Context"
    ATTENTION_O = "Attention.O"
    KV_CACHE = "KVCache"
    MLP = "MLP"
    MLP_GATE = "MLP.Gate"
    MLP_UP = "MLP.Up"
    MLP_ACTIVATION = "MLP.Activation"
    MLP_DOWN = "MLP.Down"
    NEURON = "Neuron"
    LM_HEAD = "LMHead"
    LOGITS = "Logits"


# ComponentType members that (a) are meaningfully per-token and (b) exist
# per-layer -- used by ComponentRegistry to decide what to enumerate.
# Kept here, next to the vocabulary itself, rather than in registry.py, so
# the two never drift apart silently.
ATTENTION_PART_TYPES = {
    "q": ComponentType.ATTENTION_Q,
    "k": ComponentType.ATTENTION_K,
    "v": ComponentType.ATTENTION_V,
    "scores": ComponentType.ATTENTION_SCORES,
    "probs": ComponentType.ATTENTION_PROBABILITY,
    "context": ComponentType.ATTENTION_CONTEXT,
    "o": ComponentType.ATTENTION_O,
}
MLP_PART_TYPES = {
    "gate": ComponentType.MLP_GATE,
    "up": ComponentType.MLP_UP,
    "activation": ComponentType.MLP_ACTIVATION,
    "down": ComponentType.MLP_DOWN,
}


class ComponentID:
    """Immutable, hashable, string-round-trippable component identifier.

    Construct via the canonical factory classmethods (`.model()`,
    `.layer()`, `.attention()`, `.attention_part()`, ...) rather than
    `ComponentID(("a", "b"))` directly at call sites. The raw constructor
    and `.parse()` stay available (registries and storage need to
    reconstruct IDs from strings), but application code should read as
    `ComponentID.attention_part(17, "q")`, not string-built by hand --
    that's what keeps every caller's IDs consistent with each other.
    """

    __slots__ = ("_parts",)

    def __init__(self, parts: tuple[str, ...]) -> None:
        if not parts:
            raise ValueError("ComponentID requires at least one part")
        clean = tuple(str(p) for p in parts)
        for p in clean:
            if not p or "/" in p:
                raise ValueError(f"invalid ComponentID part: {p!r} (must be non-empty, and not contain '/')")
        self._parts = clean

    # -- core value-type protocol ----------------------------------------
    @property
    def parts(self) -> tuple[str, ...]:
        return self._parts

    def __str__(self) -> str:
        return "/".join(self._parts)

    def __repr__(self) -> str:
        return f"ComponentID({str(self)!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ComponentID) and self._parts == other._parts

    def __hash__(self) -> int:
        return hash(self._parts)

    def __truediv__(self, part: Union[str, int]) -> "ComponentID":
        """``cid / "q"`` appends one part. Used internally by the factory
        methods below; also available to callers building a component ID
        this module doesn't have a named factory for yet."""
        return ComponentID(self._parts + (str(part),))

    @classmethod
    def parse(cls, s: str) -> "ComponentID":
        if not s:
            raise ValueError("cannot parse empty string as ComponentID")
        return cls(tuple(s.split("/")))

    @property
    def parent(self) -> Optional["ComponentID"]:
        """Structural truncation: removes exactly the last part, the way
        `pathlib.PurePath.parent` removes exactly the last path segment.

        This is NOT guaranteed to land on another "canonical",
        factory-producible component ID. The part-tuples this module's
        factories build are not a uniform number of parts apart from
        each other -- `.layer(17)` adds two parts ("layer", "17") on top
        of `.model()`, but `.attention(17)` adds only one ("attention")
        on top of `.layer(17)`, and `.neuron(17, 3)` adds two ("neuron",
        "3") on top of `.mlp(17)`. So `.parent` on an attention-part ID
        reaches `.attention(17)` in one step, but reaching `.layer(17)`
        needs a second `.parent` call, and reaching `.model()` needs a
        third -- there is no fixed number of `.parent` calls that always
        lands on "the enclosing layer" or "the model root" for every
        kind of ID. Use `.layer_id` below for "the enclosing layer's own
        ID", which does not have this problem.
        """
        if len(self._parts) <= 1:
            return None
        return ComponentID(self._parts[:-1])

    @property
    def layer_id(self) -> Optional["ComponentID"]:
        """The enclosing layer's own ComponentID (e.g. `model/layer/17`
        for any ID belonging to layer 17), or None for an ID with no
        layer_index (model-level components: embedding, lm_head,
        logits, or the model root itself). Unlike chaining `.parent`
        (see its docstring for why), this always lands on a real,
        `ComponentID.layer(...)`-producible ID in one step, regardless
        of how many parts separate this ID from its layer."""
        idx = self.layer_index
        return ComponentID.layer(idx) if idx is not None else None

    @property
    def layer_index(self) -> Optional[int]:
        """The layer index this ID belongs to, if any. Parsed by finding a
        "layer" segment followed by an integer segment -- not assumed to
        sit at a fixed position, since not every ID has a layer at all
        (e.g. `model/embedding`)."""
        parts = self._parts
        for i in range(len(parts) - 1):
            if parts[i] == "layer":
                try:
                    return int(parts[i + 1])
                except ValueError:
                    return None
        return None

    # -- canonical factories (spec §6's own examples) ---------------------
    @classmethod
    def model(cls) -> "ComponentID":
        return cls(("model",))

    @classmethod
    def embedding(cls) -> "ComponentID":
        return cls.model() / "embedding"

    @classmethod
    def layer(cls, layer_idx: int) -> "ComponentID":
        return cls.model() / "layer" / layer_idx

    @classmethod
    def attention(cls, layer_idx: int) -> "ComponentID":
        return cls.layer(layer_idx) / "attention"

    @classmethod
    def attention_part(cls, layer_idx: int, part: str) -> "ComponentID":
        if part not in _ATTENTION_PARTS:
            raise ValueError(f"unknown attention part {part!r}; expected one of {_ATTENTION_PARTS}")
        return cls.attention(layer_idx) / part

    @classmethod
    def kv_cache(cls, layer_idx: int) -> "ComponentID":
        return cls.layer(layer_idx) / "kv_cache"

    @classmethod
    def mlp(cls, layer_idx: int) -> "ComponentID":
        return cls.layer(layer_idx) / "mlp"

    @classmethod
    def mlp_part(cls, layer_idx: int, part: str) -> "ComponentID":
        if part not in _MLP_PARTS:
            raise ValueError(f"unknown MLP part {part!r}; expected one of {_MLP_PARTS}")
        return cls.mlp(layer_idx) / part

    @classmethod
    def neuron(cls, layer_idx: int, neuron_idx: int) -> "ComponentID":
        """Deliberately not pre-populated in ComponentRegistry (spec
        §22/23 -- a real model's intermediate_size can be in the
        thousands, so eagerly enumerating one ComponentID per neuron per
        layer would bloat the registry for a component nothing consumes
        yet). Callers construct this on demand once Phase 5 needs it."""
        return cls.mlp(layer_idx) / "neuron" / neuron_idx

    @classmethod
    def norm(cls, layer_idx: int, position: Optional[str] = None) -> "ComponentID":
        base = cls.layer(layer_idx) / "norm"
        return base / position if position else base

    @classmethod
    def residual(cls, layer_idx: int, point: str) -> "ComponentID":
        if point not in _RESIDUAL_POINTS:
            raise ValueError(f"unknown residual point {point!r}; expected one of {_RESIDUAL_POINTS}")
        return cls.layer(layer_idx) / "residual" / point

    @classmethod
    def block(cls, layer_idx: int) -> "ComponentID":
        return cls.layer(layer_idx)

    @classmethod
    def lm_head(cls) -> "ComponentID":
        return cls.model() / "lm_head"

    @classmethod
    def logits(cls) -> "ComponentID":
        return cls.model() / "logits"
