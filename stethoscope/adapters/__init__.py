"""stethoscope.adapters -- ModelAdapter implementations.

Two adapters ship in Phase 1:

    stethoscope.adapters.synthetic     wraps examples/synthetic_model.py
    stethoscope.adapters.huggingface   config-driven, multi-family

See stethoscope.adapters.base.ModelAdapter for the interface both
implement, and the architecture delta report §13 for why "produce a
manifest/registry" is the whole interface in this phase -- no live-model
binding yet, that's Phase 2.
"""
