# Project Status

Last updated: 2026-08-06, in response to "is everything done?" after a
successful CLI test run against a throwaway, untrained example model. Short
answer: **no.** This document exists so that question has a durable, honest
answer that doesn't depend on scrolling back through a chat history to
reconstruct.

**Update, same session:** the cross-layer weight propagation gap flagged
below as the top real-code limitation has been fixed and verified (see its
row in the table). This document is being kept current rather than treated
as a one-time snapshot — if you're reading this later and something below
looks stale, trust the tests over the prose.

Read this before deciding to rely on either subsystem for anything beyond
what's explicitly marked done below.

## The one-sentence version

Both subsystems' *code* is real, tested, and has caught and fixed genuine
bugs along the way (see NOTES.md and CHANGELOG.md). Neither subsystem's
*core scientific/practical claim* has been confirmed at real scale by
anyone. Those are different kinds of "done," and only the first one is true.

## Weight compression (`hope_core/`)

| Item | Status |
|---|---|
| Formulas match the paper (arXiv:2607.21366) | ✅ 78 tests, including a Monte Carlo check of the self-kernel against its own mathematical definition (not just the paper's transcription), and the paper's central theoretical claim (scale invariance under weight rescaling) checked directly, not assumed |
| Merge optimality | ✅ Verified against naive strawmen (parameter averaging, reusing one child) on the paper's own distortion objective |
| Real PyTorch round-trip | ✅ `adapters.py` tested against actual `nn.Module` forward passes, both Linear and Conv2d |
| CLI | ✅ Confirmed working end-to-end on a second, independent machine (Windows) |
| **The paper's own ResNet-50/ImageNet result** | ❌ **Never run, by anyone.** `scripts/reproduce_paper_resnet50.py` is written and structurally verified against the real ResNet-50 architecture (random weights, no network needed for that part) -- the actual accuracy numbers, which are the only thing that would confirm HOPE works at real scale, have never been produced. This is the single most important open item in the whole project. |
| Toy reproduction (synthetic data) | ⚠️ Runs and shows graceful, non-catastrophic degradation -- a real code-sanity signal. Also honestly shows HOPE was *not* clearly better than magnitude pruning across 3 seeds. Do not read this as either a pass or a fail on HOPE's actual merit; it's evidence about the implementation, not the method. |
| Cross-layer joint compression | ✅ Fixed and verified — `ChainAwareEncodingLoop` + `LayerState.resync_downstream_consumer()`/`resync_upstream_provider()` (`hope_core/layer_state.py`, `hope_core/encoding_loop.py`). Tested against a real 3-module `nn.Sequential` chain: two adjacent layers compressed *jointly* (not sequentially), rebuilt via `adapters.py`, run through a real forward pass. The fix needed BOTH directions of propagation, not one — a first attempt only handled one and failed loudly (a real shape-mismatch error) the moment the untested direction was exercised, which is how the gap was found to be incomplete rather than assumed fixed. `scripts/toy_reproduction.py` still uses sequential single-boundary compression as of this document — updating it to use the new joint-compression path is real, straightforward follow-up work, not done yet. |
| Transformer/GELU/SwiGLU applicability | ❌ Untested. The exact kernel formulas are derived for PH-1 (ReLU-family) activations; most current LLM MLPs use GELU/SiLU, which fall outside that exact guarantee. Unknown whether this degrades gracefully or needs new derivation. |
| O(N^2) merge-cache scaling | ⚠️ Confirmed real bottleneck at ResNet-50 layer widths (full prune+merge didn't finish in 300s on a single ~128-channel boundary). Mitigated (prune-only fallback above a width threshold), not solved. |
| Two math simplifications in `merge.py` | ⚠️ Sign resolution and the `gamma_p^2 < eps` boundary case -- reasonable, internally consistent, validated indirectly (optimality tests pass), but the paper defers the exact formulas to appendices this implementation never had access to |
| DEFT's "Dynamic Resolution of Redundancy" | ❌ Not implemented (the core elasticity/mask/gradient-scaling pipeline is) |

## KV-cache introspection (`core/`, `compression/`, `harness/`)

| Item | Status |
|---|---|
| Core mechanics (hooks, modifiers, inspector) | ✅ 96 tests, including against a REAL (tiny, offline-constructed, untrained) `Gemma3ForCausalLM` -- not just synthetic numpy data. Confirmed the modifier system correctly targets/scopes real KV tensors, and documented a real, non-obvious finding about layer-scoping semantics along the way (`core/kv_modifiers.py`'s `KVModifier` docstring). |
| Real Gemma-3-4b-it, cache **observation** | ✅ Confirmed working, dated 2026-07-29 (CHANGELOG.md) |
| Real Gemma-3-4b-it, causal **editing** (`--modifier` + `--compare-baseline`, real pretrained weights) | ❌ Attempted once; output never confirmed/shared. Status is "need to check more," not "done." |
| CLI unification | ✅ Essentially complete -- `parse-modifier`/`benchmark`/`demo` reimplemented and tested; `gemma` is a verified byte-identical passthrough, deliberately not a reimplementation |
| `hook_and_inspector_overhead_under_10pct` | ❌ Real, measured failure (+92% overhead), reproduced independently multiple times, unresolved |
| GEPA / Autoresearch integrations | ⚠️ Explicitly "reference implementations against the interface the spec described" (the repo's own words) -- never verified against the real GEPA or Autoresearch systems |

## Platform-level (the merge itself)

| Item | Status |
|---|---|
| Structural merge (one repo, one dependency list, one test run) | ✅ Verified -- both suites run independently before merging, then together after, specifically to catch import breakage. None found. |
| Shared event-log format | ❌ Not done, on purpose -- the two subsystems converged on a similar instinct (append-only audit trail) but extracting a shared abstraction now, before a third subsystem exists to check it against, would repeat the premature-generalization mistake this project explicitly set out to avoid |
| Unified debugger | ❌ Not done |

## What "done" would actually require, in priority order

1. **Phase 0.** Run `scripts/reproduce_paper_resnet50.py` against real ImageNet. Nothing else on this list matters as much -- if HOPE's core claim doesn't hold at real scale, the rest is a well-tested implementation of something that doesn't work as well as claimed, and that needs to be known plainly, not discovered later.
2. **Real-Gemma causal editing, captured as a repeatable result** -- not just "I ran it once," but the actual before/after output, ideally checked into this repo the way `CHANGELOG.md` already does for the observation path.
3. Everything else in the tables above, roughly in the order listed.

Nothing about this document is a request to stop using either subsystem --
it's a request to use them with accurate expectations about what's actually
been checked.
