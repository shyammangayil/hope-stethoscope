"""stethoscope/adapters/synthetic.py
=====================================
SyntheticAdapter: wraps the existing `examples/synthetic_model.py`
fixture (`SyntheticTransformer` / `SyntheticHookableModel`) as a
ModelAdapter.

Does not reimplement the synthetic model -- per the spec's own
confidentiality/boundaries note ("Do not replace existing project-
specific ideas with generic implementations merely because a generic
library exists"), this adapter reads facts off the *existing*,
already-validated fixture the rest of this repo's test suite and
`examples/run_demo.py` already rely on (spec §24's validation
progression starts with exactly this: "Unit test -> Synthetic
Transformer -> ...").

Honesty note (see `stethoscope.registry`'s module docstring for why this
matters downstream): `SyntheticTransformer` is attention+residual only,
by design -- no MLP, no normalization layers, no positional encoding
(see its own module docstring). This adapter reports that honestly
(`mlp.kind=None`, `normalization.kind=None`,
`attention.positional_scheme="none"`, with explanatory notes) rather
than guessing a shape none of those actually have.
"""
from __future__ import annotations

from typing import Any

from stethoscope.adapters.base import ModelAdapter
from stethoscope.manifest import AttentionManifest, CacheManifest, EmbeddingManifest, ModelManifest


class SyntheticAdapter(ModelAdapter):
    """Wraps a `SyntheticTransformer` or a `SyntheticHookableModel`
    wrapping one (`examples/synthetic_model.py`)."""

    def __init__(self, model: Any, *, model_id: str = "synthetic") -> None:
        # Accept either a bare SyntheticTransformer or a
        # SyntheticHookableModel (which holds one at `.t`) -- both show
        # up across this repo's own examples/tests. Only the
        # transformer's static shape is needed here, not cache state.
        self._transformer = getattr(model, "t", model)
        self.model_id = model_id

    def build_manifest(self) -> ModelManifest:
        t = self._transformer
        notes = [
            "SyntheticTransformer (examples/synthetic_model.py) is "
            "attention + residual-add only, by design -- no MLP and no "
            "normalization layers exist on this model at all. mlp.kind "
            "and normalization.kind are left None (not guessed at "
            "'dense'/'rmsnorm'), per spec §30; ComponentRegistry "
            "correctly emits no MLP/Norm components for this model as a "
            "result -- see stethoscope.registry's module docstring.",
            "No positional encoding of any kind is added to token "
            "embeddings -- attention is purely content-based plus a "
            "causal mask (see SyntheticTransformer.attend). "
            "attention.positional_scheme is set to the confirmed value "
            "'none', not left unset, since this is a determined fact "
            "about the model, not an unknown.",
            "context_length is left None: no fixed context window is "
            "enforced anywhere in this fixture (the KV cache grows "
            "unbounded via np.concatenate every step).",
        ]
        return ModelManifest(
            model_id=self.model_id,
            framework="synthetic",
            architecture="SyntheticTransformer",
            modalities=("text",),
            num_layers=t.num_layers,
            parameter_count=self._count_parameters(t),
            dtype="float32",  # every weight array is explicitly .astype(np.float32) at construction
            device="cpu",  # numpy has no GPU tensors in this repo's usage
            context_length=None,
            tokenizer=None,  # operates on raw integer token ids; no tokenizer object exists
            attention=AttentionManifest(
                kind="mha",  # Wq/Wk/Wv all produce num_heads heads -- no separate KV head count exists anywhere in project_qkv/attend
                num_query_heads=t.num_heads,
                num_kv_heads=t.num_heads,
                head_dim=t.head_dim,
                positional_scheme="none",
                sliding_window=None,
                layer_types=None,  # uniform causal self-attention on every layer
            ),
            embedding=EmbeddingManifest(
                vocab_size=t.vocab_size,
                hidden_dim=t.hidden_dim,
                tied_embeddings=False,  # embed and unembed are two independently-initialized random matrices
            ),
            cache=CacheManifest(
                implementation="NumpyKVCache",
                num_layers=t.num_layers,
                strategy="dynamic",  # grows every step via np.concatenate; no static pre-allocation
            ),
            notes=notes,
        )

    @staticmethod
    def _count_parameters(t: Any) -> int:
        """embed + unembed + per-layer (Wq, Wk, Wv, Wo). A real count
        from the live weight arrays, not an estimate from the shape
        parameters -- cheap and exact for a numpy model, so there is no
        reason to approximate it."""
        n = t.embed.size + t.unembed.size
        for layer in t.layers:
            n += layer.Wq.size + layer.Wk.size + layer.Wv.size + layer.Wo.size
        return int(n)
