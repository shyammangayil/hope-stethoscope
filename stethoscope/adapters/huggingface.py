"""stethoscope/adapters/huggingface.py
========================================
HFTransformersAdapter: a config-driven ModelAdapter for Hugging Face
`transformers` causal-LM models (architectural-evolution spec §4.1,
`docs/ARCHITECTURE_EVOLUTION_SPEC.md`).

Deliberately NOT Gemma-specific (spec §4.1: "The core must not become
hardcoded to Gemma"; spec §34: "do not hardcode Gemma-specific
assumptions into the core"). Field extraction uses a multi-name-tolerant
lookup across the attribute names different model families use for the
same underlying fact (e.g. `num_hidden_layers` vs `n_layer`), rather than
assuming one family's exact config shape. See
tests/test_stethoscope_adapters_huggingface.py for real, offline
validation against both a Gemma3 config (GQA, heterogeneous sliding/
global layers) and a Llama config -- concrete evidence for spec §24's
"test against more than one architecture", not a promise deferred to
Phase 9.

Composite-config handling: calls `config.get_text_config()` first when
present, before reading any field -- the same lesson `examples/
run_gemma.py` already applies for real (delta report §3): multimodal
composite configs (e.g. the full `Gemma3Config`, as opposed to the
text-only `Gemma3TextConfig`) don't expose fields like
`num_hidden_layers` on the top-level object directly. `get_text_config()`
is defined on HF's base `PretrainedConfig` and is safe to call on an
already-text-only config too (returns itself), so this is applied
unconditionally rather than trying to detect "is this composite" first.

`transformers`/`torch` are never imported at module scope -- matching the
lazy-import discipline already used throughout `core/` and `harness/` in
this repository -- so importing `stethoscope`, or even this module, does
not require either package to be installed unless an adapter instance is
actually built and used against a live model.

Known, deliberate limitation -- read before assuming a gap is a bug:
`mlp.gated`/`mlp.kind` cannot be reliably determined from config alone
across arbitrary HF model families. Unlike layer/head counts or norm
epsilon, "does this MLP have a separate gate projection" is not
consistently encoded as a named config field across families (Llama/
Gemma/Mistral/Qwen-style configs expose no `is_gated`-shaped flag; it is
architectural, baked into the modeling code's MLP class, not the
config). Guessing it from the activation-function name is unreliable
(Gemma's gated MLP uses a GELU variant, not SiLU, so "gelu implies
dense" would be wrong specifically for Gemma), and inferring it from
actual submodule names (checking for a `gate_proj` attribute) is exactly
the `named_modules()`-shaped fragility spec §34 warns against -- and
would also break the "pure function of a manifest, no live-module
access" property Phase 1 deliberately preserves (see
`stethoscope.registry`'s module docstring). Pass `mlp_gated=True/False`
explicitly when the caller knows this fact about the family being
adapted (as the tests here do for Gemma3 and Llama, both genuinely
gated); otherwise it is honestly left `None`, with a note explaining
why, rather than guessed.
"""
from __future__ import annotations

from typing import Any, Optional

from stethoscope.adapters.base import ModelAdapter
from stethoscope.manifest import (
    AttentionManifest,
    CacheManifest,
    EmbeddingManifest,
    MLPManifest,
    ModelManifest,
    NormalizationManifest,
)

# Candidate attribute names tried, in order, for each fact -- covers
# GPT2/Llama/Gemma/Mistral/Qwen-family naming without hardcoding any one
# family's exact shape. A field stays None if none of its candidates are
# present (or all are None) on the given config -- never guessed from an
# unrelated field.
_NUM_LAYERS_NAMES = ("num_hidden_layers", "n_layer", "num_layers")
_NUM_Q_HEADS_NAMES = ("num_attention_heads", "n_head")
_NUM_KV_HEADS_NAMES = ("num_key_value_heads", "num_kv_heads")
_HEAD_DIM_NAMES = ("head_dim",)
_HIDDEN_SIZE_NAMES = ("hidden_size", "n_embd", "d_model")
_INTERMEDIATE_SIZE_NAMES = ("intermediate_size", "n_inner", "ffn_dim")
_ACTIVATION_NAMES = ("hidden_activation", "hidden_act", "activation_function")
_RMS_EPS_NAMES = ("rms_norm_eps",)
_LAYERNORM_EPS_NAMES = ("layer_norm_eps", "layer_norm_epsilon")
_MAX_POS_NAMES = ("max_position_embeddings", "n_positions", "n_ctx")
_ROPE_SIGNAL_NAMES = ("rope_theta", "rope_parameters", "rope_scaling")
_SLIDING_WINDOW_NAMES = ("sliding_window",)
_VOCAB_SIZE_NAMES = ("vocab_size",)
_TIE_EMBEDDINGS_NAMES = ("tie_word_embeddings",)
_LAYER_TYPES_NAMES = ("layer_types",)
_CACHE_STRATEGY_NAMES = ("cache_implementation",)
_ARCHITECTURES_NAMES = ("architectures",)


def _first_attr(obj: Any, names: tuple, default: Any = None) -> Any:
    """First non-None attribute among `names` present on `obj`.

    Deliberately does not distinguish "attribute missing entirely" from
    "attribute present but None" -- either way there is nothing usable
    to report, and the caller ends up with `default` (normally None,
    i.e. UNAVAILABLE) either way.
    """
    for name in names:
        val = getattr(obj, name, None)
        if val is not None:
            return val
    return default


def manifest_from_config(
    config: Any,
    *,
    model_id: str,
    framework: str = "huggingface",
    mlp_gated: Optional[bool] = None,
) -> ModelManifest:
    """The pure, torch/transformers-independent core of this adapter:
    reads plain attributes off any config-like object -- a real HF
    `PretrainedConfig`, or a plain mock object with the same attribute
    names, for fast unit tests that don't need to construct a real
    model (see tests/test_stethoscope_adapters_huggingface.py). Public
    (not underscore-prefixed) because it is independently useful: an
    "inspect the architecture without loading any weights" workflow --
    e.g. from a bare `AutoConfig.from_pretrained(..., local_files_only=True)`
    -- needs exactly this and nothing else from `HFTransformersAdapter`.
    """
    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config

    num_layers = _first_attr(text_config, _NUM_LAYERS_NAMES)
    num_q_heads = _first_attr(text_config, _NUM_Q_HEADS_NAMES)
    num_kv_heads = _first_attr(text_config, _NUM_KV_HEADS_NAMES, default=num_q_heads)
    hidden_size = _first_attr(text_config, _HIDDEN_SIZE_NAMES)
    head_dim = _first_attr(text_config, _HEAD_DIM_NAMES)
    if head_dim is None and hidden_size is not None and num_q_heads:
        head_dim = hidden_size // num_q_heads  # the standard convention when a config doesn't state head_dim explicitly

    if num_q_heads is not None and num_kv_heads is not None:
        if num_kv_heads == num_q_heads:
            attn_kind = "mha"
        elif num_kv_heads == 1:
            attn_kind = "mqa"
        else:
            attn_kind = "gqa"
    else:
        attn_kind = None

    layer_types = _first_attr(text_config, _LAYER_TYPES_NAMES)
    if layer_types is not None:
        layer_types = tuple(layer_types)

    attention = AttentionManifest(
        kind=attn_kind,
        num_query_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        positional_scheme="rope" if _first_attr(text_config, _ROPE_SIGNAL_NAMES) is not None else None,
        sliding_window=_first_attr(text_config, _SLIDING_WINDOW_NAMES),
        layer_types=layer_types,
    )

    mlp = MLPManifest(
        kind=("gated" if mlp_gated else "dense") if mlp_gated is not None else None,
        hidden_dim=hidden_size,
        intermediate_dim=_first_attr(text_config, _INTERMEDIATE_SIZE_NAMES),
        activation=_first_attr(text_config, _ACTIVATION_NAMES),
        gated=mlp_gated,
    )

    rms_eps = _first_attr(text_config, _RMS_EPS_NAMES)
    ln_eps = _first_attr(text_config, _LAYERNORM_EPS_NAMES)
    # rms_norm_eps / layer_norm_eps are field NAMES HF config authors
    # chose specifically to encode which norm kind is used -- reading
    # which one is present is treated as direct evidence, not a guess
    # (contrast with mlp.gated above, which has no equivalent).
    if rms_eps is not None:
        norm_kind, norm_eps = "rmsnorm", rms_eps
    elif ln_eps is not None:
        norm_kind, norm_eps = "layernorm", ln_eps
    else:
        norm_kind, norm_eps = None, None
    normalization = NormalizationManifest(kind=norm_kind, position=None, epsilon=norm_eps)

    embedding = EmbeddingManifest(
        vocab_size=_first_attr(text_config, _VOCAB_SIZE_NAMES),
        hidden_dim=hidden_size,
        tied_embeddings=_first_attr(text_config, _TIE_EMBEDDINGS_NAMES),
    )

    cache_strategy = _first_attr(text_config, _CACHE_STRATEGY_NAMES, default="dynamic")
    cache = CacheManifest(
        # "transformers.Cache" names the actual abstract interface every
        # HF causal LM's forward() accepts as past_key_values -- the same
        # authoritative-interface terminology spec §3 and
        # core/cache_adapters.py already use ("Cache.update()"). This is
        # not a claim about which concrete subclass (DynamicCache,
        # StaticCache, ...) a specific call will instantiate -- that
        # depends on generation-time arguments this adapter never sees.
        implementation="transformers.Cache",
        num_layers=num_layers,
        strategy=cache_strategy,
    )

    architectures = _first_attr(text_config, _ARCHITECTURES_NAMES)
    architecture = architectures[0] if architectures else type(config).__name__

    notes = []
    if mlp_gated is None:
        notes.append(
            "mlp.gated/mlp.kind could not be determined from config alone "
            "(no HF config field reliably encodes this across families) "
            "and no mlp_gated= override was supplied -- see this module's "
            "docstring for why this is a documented limitation, not an "
            "oversight."
        )
    if norm_kind is not None:
        notes.append(
            "normalization.position (pre/post) is not set generically: "
            "some architectures (e.g. Gemma2/3's sandwich norm) have both "
            "pre- and post-sublayer norms, so a blanket pre/post guess "
            "would be wrong for them. ComponentRegistry falls back to one "
            "generic, unqualified per-layer Norm component when position "
            "is None but kind is known."
        )
    if _first_attr(text_config, _CACHE_STRATEGY_NAMES) is None:
        notes.append(
            "cache.strategy defaults to 'dynamic' (this config has no "
            "explicit cache_implementation field) -- matching the "
            "DynamicCache-by-default behavior this repo's own "
            "tests/test_real_gemma3_offline.py already relies on for "
            "generation, not a fresh guess about this specific model."
        )

    return ModelManifest(
        model_id=model_id,
        framework=framework,
        architecture=architecture,
        modalities=("text",),
        num_layers=num_layers,
        parameter_count=None,  # filled in by HFTransformersAdapter.build_manifest() when a live model is available; this function only ever sees a config
        dtype=None,
        device=None,
        context_length=_first_attr(text_config, _MAX_POS_NAMES),
        tokenizer=None,
        attention=attention,
        mlp=mlp,
        normalization=normalization,
        embedding=embedding,
        cache=cache,
        notes=notes,
    )


class HFTransformersAdapter(ModelAdapter):
    """Wraps a live `transformers.PreTrainedModel`-like object (anything
    exposing a `.config`). For architecture-only inspection without a
    live model, call `manifest_from_config()` above directly instead of
    constructing this class."""

    def __init__(self, model: Any, *, model_id: Optional[str] = None, mlp_gated: Optional[bool] = None) -> None:
        self.model = model
        self._model_id = model_id
        self.mlp_gated = mlp_gated

    def _resolve_model_id(self, config: Any) -> str:
        if self._model_id:
            return self._model_id
        name = getattr(config, "name_or_path", None) or getattr(config, "_name_or_path", None)
        return name or type(config).__name__

    def build_manifest(self) -> ModelManifest:
        config = self.model.config
        manifest = manifest_from_config(
            config, model_id=self._resolve_model_id(config), mlp_gated=self.mlp_gated
        )
        # The live model's own class name is strictly more authoritative
        # than anything derivable from config alone: a freshly-constructed
        # config (Gemma3TextConfig(...), not loaded via from_pretrained())
        # has no populated `architectures` field at all, so
        # manifest_from_config() falls back to the *config* class's own
        # name (e.g. "Gemma3TextConfig") -- correct as far as it goes with
        # only a config to look at, but wrong once a live model is
        # available, since the model class (e.g. "Gemma3ForCausalLM") is
        # what a caller actually means by "architecture". Found via the
        # real-model tests in test_stethoscope_adapters_huggingface_real.py,
        # not anticipated in advance -- the mock-config tests didn't catch
        # it because their mocks set `architectures=[...]` explicitly,
        # matching a from_pretrained()-loaded config's shape rather than a
        # freshly-constructed one's.
        manifest.architecture = type(self.model).__name__
        manifest.parameter_count = self._count_parameters(self.model)
        manifest.dtype = self._infer_dtype(self.model)
        manifest.device = self._infer_device(self.model)
        return manifest

    @staticmethod
    def _count_parameters(model: Any) -> Optional[int]:
        try:
            return int(sum(p.numel() for p in model.parameters()))
        except Exception:
            return None

    @staticmethod
    def _infer_dtype(model: Any) -> Optional[str]:
        try:
            return str(next(model.parameters()).dtype).replace("torch.", "")
        except Exception:
            return None

    @staticmethod
    def _infer_device(model: Any) -> Optional[str]:
        try:
            return str(next(model.parameters()).device)
        except Exception:
            return None
