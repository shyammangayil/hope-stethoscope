# HOPE Stethoscope

An observability-and-intervention platform for neural networks, currently
made of two independently-validated subsystems that got merged into one
repository, plus a third that's just begun (see "Model-wide observability"
below):

- **Weight-space compression** (`hope_core/`) — an implementation of HOPE
  (arXiv:2607.21366), modeling neurons as rank-1 Hilbert-Schmidt operators to
  prune and merge redundant capacity in trained weights.
- **KV-cache introspection & compression** (`core/`, `compression/`,
  `harness/`, `integrations/`) — hooks into a running HuggingFace model's KV
  cache during generation, to observe, benchmark-compress, and (experimentally)
  causally edit it.
- **Model-wide observability framework** (`stethoscope/`, Phase 2 of 9 —
  new) — the planned generalization of the KV-cache work above into
  model-wide observation, intervention, and causal experimentation
  (residual stream, attention Q/K/V, MLP/neuron activations, not just
  the KV cache). Phase 2 bridges the existing, unmodified `CacheTap`
  mechanism into the new framework, so KV-cache observation now works
  through both APIs at once. See
  [`docs/ARCHITECTURE_DELTA_REPORT.md`](docs/ARCHITECTURE_DELTA_REPORT.md)
  for the full plan and [`TASKS.md`](TASKS.md) for live status — residual/
  attention/MLP observation, interventions, and causal experiments don't
  exist yet.

Full detail on each lives in its own doc — **read those, not just this file,
before trusting either subsystem**: [`docs/weight_compression.md`](docs/weight_compression.md)
and [`docs/kv_cache.md`](docs/kv_cache.md). This README is the map, not the
territory.

**For a direct answer to "is this done" at any point, read
[`PROJECT_STATUS.md`](PROJECT_STATUS.md)** — an honest, kept-up-to-date
ledger of what's actually validated versus what just runs without error.
The two are not the same claim, and this repo has repeatedly found that
out the hard way.

## The one principle that shaped this merge

**Shared infrastructure, not shared algorithms.** These two subsystems now
live in one repo, one `requirements.txt`, one test run — but weight-space
Hilbert operator math and KV-cache tensor manipulation are genuinely
different problems, and forcing them behind one shared abstraction (a
`RepresentationOperator` base class, a unified "compress anything" API, etc.)
before there's a real third method to generalize from would repeat the exact
premature-abstraction mistake this project explicitly avoided when it was
just `hope_core`. So: this merge is structural (one directory tree, one
dependency list, one place to look), not architectural. Where the two
subsystems genuinely share a shape — see "What's actually shared" below —
that's called out explicitly, not implied by proximity.

## Validation status (read before trusting any number from either side)

Both subsystems were built the same way: real code, real tests run against
real (if sometimes synthetic) inputs, honest reporting of what failed. Both
have the same category of open gap — validated in miniature, not yet
confirmed at the scale that would make either one's own paper/claims stick:

| Subsystem | What's independently verified | What's still open |
|---|---|---|
| Weight compression | 78 tests incl. a Monte Carlo check of the self-kernel formula against its own definition, and an end-to-end merge-optimality check against naive baselines. Toy reproduction (synthetic data) runs and shows sane, graceful degradation. | The paper's own ResNet-50/ImageNet numbers — needs real ImageNet + real pretrained weights, neither reachable from the sandbox this was built in. `scripts/reproduce_paper_resnet50.py` is written and structurally verified against the real ResNet-50 architecture, but never run against real data. |
| KV-cache introspection | 96 tests (was 80). Full demo pipeline (`examples/run_demo.py`) re-run fresh in this session, including a real, un-hidden failing criterion (`hook_and_inspector_overhead_under_10pct`, measured +92% overhead) and a real infrastructure-unavailable guard firing correctly instead of fabricating a perplexity number. Confirmed (per `CHANGELOG.md`) working against real `google/gemma-3-4b-it` on 2026-07-29 for cache observation. **New**: `tests/test_real_gemma3_offline.py` constructs a REAL (tiny, untrained) `transformers.Gemma3ForCausalLM` with no network access at all, and runs the actual `CacheTap` → `generate()` → modifier → inspector pipeline against it — confirming the mechanics (not just the interface) against the authentic model class and its real heterogeneous sliding-window/global cache pattern, across 4 of the 7 modifier types. | The causal-editing path against the real *pretrained* Gemma checkpoint specifically (tokenizer, real weights, `--compare-baseline`) — still "some manual runs happened, not yet a repeatable, checked-in result" against real weights, though the underlying mechanics it depends on are now much more thoroughly verified than before. The `hook_and_inspector_overhead_under_10pct` failure is real and unresolved. |
| Model-wide observability (`stethoscope/`, Phase 2 of 9) | 106 tests (71 Phase 1 + 35 Phase 2). `ComponentRegistry` enumeration is exact-set-tested. `HFTransformersAdapter` validated against two REAL, offline-constructed models (Gemma3 GQA + heterogeneous layers, and Llama, a second family). **New**: `Observer`/`CapturePolicy` bridge the existing `CacheTap` mechanism into `ObservationEvent`s without editing `core/hooks.py` at all (verified via `git diff`) — validated against a real `CacheTap` running on BOTH the synthetic model's `NumpyKVCache` and a real offline Gemma3's `DynamicCache`. See `TASKS.md` §2/§3 for what was found and decided along the way. | Residual/attention/MLP observation (Phase 3-5), interventions (Phase 6), causal experiments (Phase 7), HOPE integration (Phase 8), and the fully-automatic `with stethoscope.observe(model, ...):` sugar from the spec — deliberately deferred, see `stethoscope/observer.py`'s module docstring for why. No CLI wiring yet. |

**Test count over time: 158 (initial merge) → 175 (CLI update) → 187 (real-Gemma3-offline work) → 204 (later layer_state/encoding_loop/neuron growth, per `git log`'s Aug 8 commit — not previously called out in this file) → 275 (`stethoscope/` Phase 1) → 310 (`stethoscope/` Phase 2, see above).**
(`pytest tests/ -v` from the repo root) — the two suites were run independently before the merge
and once more together immediately after, specifically to catch any
import-path breakage from moving files around. None found.

## What's actually shared (small, deliberate list)

- One `requirements.txt`, one `pytest.ini`, one place to run `pytest tests/`.
- Both subsystems independently converged on the same reporting instinct —
  an append-only, timestamped record of every action taken (`ActionRecord`/
  `StethoscopeReport` on the weight side, `KVEvent`/`ExperienceDB` on the KV
  side) rather than only a final summary. That convergence is a real, useful
  signal about what a future shared "observation" layer should look like —
  but it hasn't been extracted into one yet. Doing that now, before a third
  subsystem exists to check the abstraction against, would be exactly the
  premature move this README just argued against. Noted as the most
  promising concrete candidate for future unification, not acted on yet.
- Nothing else. No shared base classes between `hope_core/merge.py` and
  `compression/plugins.py`. No unified CLI yet (see "Not done" below) — they
  currently have separate entry points, and that's an honest reflection of
  not having merged their actual interfaces, not an oversight.

## Directory layout

```
hope-stethoscope/
├── hope_core/              # weight-space: kernels, capacity, merge, DEFT, etc.
├── cli.py                  # weight-space CLI: hope inspect/compress/deft-setup
│
├── core/                   # KV-cache: hooks (CacheTap), storage, inspector, modifiers
├── compression/             # KV-cache: fp8/int4/low-rank/window/eviction/pruning plugins
├── harness/                  # KV-cache: evaluator, retry, budget, experiment router
├── integrations/              # KV-cache: reference GEPA + Autoresearch adapters
├── debugger/                   # KV-cache: Streamlit event-log explorer
├── config/                      # KV-cache: models.yaml, budgets.yaml
│
├── stethoscope/             # model-wide observability, Phase 2 of 9 (new)
│   ├── ids.py                # ComponentID, ComponentType
│   ├── manifest.py            # ModelManifest + sub-manifests
│   ├── registry.py             # ComponentRegistry
│   ├── events.py                # ObservationEvent, ObservationPhase
│   ├── capture.py                # CaptureMode, CapturePolicy
│   ├── storage.py                 # ObservationEventSink (InMemory / SQLite)
│   ├── observer.py                 # Observer -- bridges core.hooks.CacheTap, unmodified
│   └── adapters/                    # SyntheticAdapter, HFTransformersAdapter
│
├── scripts/                # weight-space toy + real (ResNet-50) reproduction
├── examples/                # BOTH: quickstart.py (weights), run_demo.py /
│                            #   run_gemma.py / run_mutation_demo.py (KV cache)
├── tests/                   # ALL THREE subsystems' tests, 310 total
├── docs/
│   ├── weight_compression.md   # full README for hope_core (was the top-level README)
│   ├── kv_cache.md              # full README for the KV-cache subsystem
│   ├── ARCHITECTURE_EVOLUTION_SPEC.md  # verbatim task spec for the stethoscope/ evolution
│   └── ARCHITECTURE_DELTA_REPORT.md     # Phase 0 audit + Phase 1 plan for stethoscope/
├── TASKS.md                  # live status/resumption tracker for the stethoscope/ work
├── NOTES.md                 # weight-space: paper-equation-to-code traceability
├── CHANGELOG.md               # KV-cache subsystem's development history
└── success_criteria.md         # KV-cache subsystem's own stated scorecard
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# For real-model modes (see the validation table above for what that
# unlocks and what it doesn't):
#   pip install torchvision                      # weight-space real repro
#   pip install transformers accelerate           # KV-cache real Gemma, and stethoscope/'s HFTransformersAdapter
python3 -m pytest tests/ -v   # expect 310 passed
```

## Quick start

**Weight-space compression** — see [`docs/weight_compression.md`](docs/weight_compression.md)
for the full CLI/API reference:
```bash
python3 examples/quickstart.py

# Try the CLI hands-on with a real example model (no model of your own needed yet):
python3 examples/make_example_model.py
python3 cli.py inspect model.pt --layer-config layer.json --target-density 0.5 --report-dir out/
```
**Before pointing this at your own model**: `torch.save(model, path)` needs
the whole object (not a state dict), *and* your model class needs to live
in its own importable module, not defined inline in the script that calls
`torch.save()` — see `docs/weight_compression.md`'s CLI section for why
(a real bug, found by running this on a second machine) and a working
pattern to copy (`examples/example_model_def.py`).

**KV-cache introspection** — see [`docs/kv_cache.md`](docs/kv_cache.md) for
the full command reference, modifier syntax, and debugger instructions.
Network-free operations are wrapped into the unified `cli.py` as of this
update; the real-Gemma path is not (see "Not done" below):
```bash
python3 cli.py kv demo                                   # full synthetic-model pipeline, no network needed
python3 cli.py kv parse-modifier "zero:token_positions=0,4"   # validate a modifier spec before a slow real run
python3 cli.py kv benchmark --plugins fp8,int4            # compression benchmark table, synthetic or --npy data

python3 -m examples.run_gemma --model google/gemma-3-4b-it --prompt "..." --max-new-tokens 5   # real model, or:
python3 cli.py kv gemma --model google/gemma-3-4b-it --prompt "..." --max-new-tokens 5          # identical, thin CLI passthrough
streamlit run debugger/app_streamlit.py
```
`hope kv gemma` forwards every argument to `run_gemma.py` unchanged (verified
byte-identical output) — it exists purely for discoverability under one
entry point, not as a reimplementation. See `docs/kv_cache.md` for the full
`--modifier` syntax, and `tests/test_real_gemma3_offline.py` for what's
independently verified about the underlying mechanics without needing
network access.

**Model-wide observability (`stethoscope/`, Phase 2)** — no CLI wiring yet
(see `TASKS.md`); this phase is a library only, for now:
```python
from stethoscope.adapters.huggingface import HFTransformersAdapter
from stethoscope.registry import ComponentRegistry
# mlp_gated=True here because Gemma3 IS a gated (GeGLU) architecture -- a
# documented fact about the family, not something the adapter guesses on
# its own; see stethoscope/adapters/huggingface.py's module docstring.
manifest = HFTransformersAdapter(my_gemma_model, mlp_gated=True).build_manifest()
registry = ComponentRegistry.from_manifest(manifest)
print(len(registry), "components discovered")           # attention/MLP/norm/residual/KV-cache per layer
```
For the synthetic (numpy-only, no torch needed) fixture instead:
```python
from stethoscope.adapters.synthetic import SyntheticAdapter
from examples.synthetic_model import build_demo_model
manifest = SyntheticAdapter(build_demo_model()).build_manifest()
```
**New in Phase 2**: observing a real KV-cache generation, one extra line
over plain `CacheTap` usage — the existing `core.hooks.CacheTap` mechanism
runs completely unmodified underneath:
```python
from transformers import DynamicCache
from core.hooks import CacheTap
from stethoscope.observer import Observer
from stethoscope.capture import CapturePolicy, CaptureMode

observer = Observer(model_id="my-gemma3", policy=CapturePolicy(mode=CaptureMode.FULL))
cache = CacheTap(DynamicCache(config=my_gemma_model.config), observer.kv_cache_sink(), run_id=observer.experiment_id)
my_gemma_model.generate(input_ids=input_ids, past_key_values=cache, max_new_tokens=20)
for event in observer.events:
    print(event.component_id, event.layer, event.statistics["l2_norm"])
```
See `docs/ARCHITECTURE_DELTA_REPORT.md` and `TASKS.md` for what this can
and can't do yet — KV-cache observation works end to end; there is no
residual/attention/MLP observation, no intervention framework, and no
causal experiment engine in this phase, and the fully-automatic
`with stethoscope.observe(model, ...):` form from the architecture spec
is deliberately not built (see `stethoscope/observer.py`'s module
docstring for why).

## Not done (explicitly — this remains a work in progress, not a finished platform)

- **CLI unification is now close to complete, with one deliberate exception.**
  `cli.py`'s `kv` group has `parse-modifier`, `benchmark`, `demo` (fully
  reimplemented against tested library code) and `gemma` (a thin, verified
  byte-identical passthrough to `examples/run_gemma.py` — not a
  reimplementation, on purpose: that script's ~400 lines of tightly-
  sequential real-model logic are confirmed working against real
  `google/gemma-3-4b-it`, and refactoring them into something wrappable
  here would mean changing tested code without being able to re-verify the
  one part that actually matters — the network-dependent path this CLI has
  no access to). `core.kv_modifiers.parse_modifier_spec` was promoted out
  of a private helper in `run_gemma.py` into `core/kv_modifiers.py` itself,
  with real test coverage it didn't have before. Separately,
  `tests/test_real_gemma3_offline.py` substantially de-risked the
  mechanics `run_gemma.py` depends on (see the validation table above) by
  testing them against a real, tiny, offline-constructible
  `Gemma3ForCausalLM` — without needing to touch `run_gemma.py`'s own code
  at all.
- **No shared event-log format**, despite the convergent instinct noted
  above — `StethoscopeReport` and `KVEvent`/`ExperienceDB` are structurally
  similar but not interchangeable today.
- **No unified debugger.** `debugger/app_streamlit.py` only visualizes
  KV-cache events; it has no view into weight-compression's `NOTES.md`/audit
  trail.
- **Both subsystems' real-scale validation gaps are still open** (see the
  table above) — none of the CLI or merge work closed either one.
- **`stethoscope/` is Phase 2 of 9** (component identity + architecture
  discovery + KV-cache observation only) — no residual/attention/MLP
  observation, no interventions, no causal experiments, no CLI wiring,
  no unification with `core.kv_modifiers` yet (that's explicitly Phase 6
  — see `TASKS.md`). `mlp.gated`/`mlp.kind` cannot be determined from an
  HF config alone across arbitrary model families and must be supplied
  explicitly (`mlp_gated=True/False`) when known — see
  `stethoscope/adapters/huggingface.py`'s module docstring for why this
  is a documented limitation, not an oversight.

**310 tests total** as of this update (`pytest tests/ -v` from the repo root).

Also as of this update: `tests/test_real_gemma3_offline.py` (see the
validation table above), and an important, non-obvious finding it surfaced
about `layer_indices` scoping -- a modifier scoped to one layer still
visibly changes LATER layers' cached values within the same forward pass,
because an earlier layer's modified cache write changes that layer's own
hidden-state output, which is the next layer's input. That's expected causal
behavior, not a scoping bug -- documented in `KVModifier`'s own docstring
(`core/kv_modifiers.py`) since it's the kind of thing anyone running live
modifier experiments (e.g. via `hope kv gemma` / `examples/run_gemma.py`)
could reasonably assume works the other way, the way an earlier version of
this repo's own test suite briefly did.

## Citation

Weight-space compression implements HOPE (arXiv:2607.21366, Mobahi &
Bartlett, 2026) — see `docs/weight_compression.md` for the full citation.
This implementation is unofficial and has no affiliation with the paper's
authors.
