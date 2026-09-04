# HOPE Stethoscope — Phase 0 Architecture Delta Report

Produced against `docs/ARCHITECTURE_EVOLUTION_SPEC.md` §36, before any
Phase 1 code was written. Every claim below was checked against the
repository itself in this session — either by reading the source directly
or by actually running it (a fresh venv was built, the real dependency set
installed including `torch`+`transformers`, and `pytest tests/`,
`python -m examples.run_demo`, `python -m examples.run_mutation_demo`, and
`tests/test_real_gemma3_offline.py` in isolation were all executed, not
just read about). Where a number appears below, it was produced by a tool
call in this session, not carried over from the repo's own docs uncritically
— in every case checked, the repo's own docs turned out to be accurate or
conservative, which is itself a finding (see §6, §7).

---

## 1. Current architecture

The repository is not one system; it is **two independent subsystems that
share infrastructure but not algorithms**, plus a thin CLI that dispatches
between them. This matters for everything that follows — the spec's
"common architecture" ask (§2) is about giving these two subsystems (and a
third, not-yet-built one) a shared *skeleton*, not about merging their
internals.

```
                              cli.py  (Click, two command groups)
                              /                              \
                    hope <cmd>                          hope kv <cmd>
                    (weight compression)                (KV-cache introspection)
                             |                                    |
                        hope_core/                    core/ + compression/ + harness/
                        (11 files, 1689 lines)         + integrations/ + debugger/
                                                        (19 files, 2597 lines)
                             |                                    |
                    scripts/, examples/quickstart.py    examples/run_demo.py,
                    (paper reproduction)                run_gemma.py, run_mutation_demo.py,
                                                          synthetic_model.py
```

- **Weight-compression subsystem** (`hope_core/`): implements a specific
  published algorithm (BN-fused neuron pruning/merging under a
  Hilbert-space capacity metric — see `docs/weight_compression.md` and the
  paper-section citations throughout `hope_core/*.py`'s docstrings). This
  is the git history's starting point ("Initial Commit").
- **KV-cache subsystem** (`core/`, `compression/`, `harness/`,
  `integrations/`, `debugger/`): added in the second commit ("Starting the
  dev work pre-Phase0, integrating with KV"), organized around six
  self-numbered "Subproblems" (Runtime Hook Layer, KV Cache Inspector,
  Compression Plugins, Timeline & Memory Debugger, Integration/
  Meta-Harness, Success Criteria) that `docs/kv_cache.md` and
  `success_criteria.md` both use as their organizing structure.
- **Bridge points between the two**: `cli.py` (one process, two command
  groups); `config/models.yaml` and `config/budgets.yaml` (read only by
  the KV side); nothing in `hope_core/` imports from `core/`/
  `compression/`/`harness/` or vice versa. They are genuinely decoupled
  at the code level today.

`config/models.yaml` is worth quoting directly, because it states the
existing model-agnosticism claim explicitly, in the repo's own words:

> `active_target: synthetic` is what `examples/run_demo.py` actually uses
> in this sandbox... The `gemma-3-*` entries show the real config shape
> for when you have GPU + huggingface.co access: swap `active_target`,
> supply HF credentials, and nothing else in this repo needs to change —
> `core/hooks.py`, `core/inspector.py`, `compression/*`, and `harness/*`
> are all already model-agnostic.

That claim is basically true for the KV-cache-shaped slice of the
architecture (see §3), which is good news for Phase 1: the new
`ModelAdapter` layer is not fighting deeply Gemma-hardcoded plumbing, it
is *formalizing an abstraction the KV side was already informally
honoring*.

---

## 2. Existing components

| Package | Files | Lines | Role |
|---|---:|---:|---|
| `core/` | 7 | 1,706 | Runtime hooks (`CacheTap`, `KVHookManager`), cache-format normalization (`cache_adapters.py`), KV statistics (`inspector.py`), KV interventions (`kv_modifiers.py`), persistence (`storage.py`: SQLite event sink, ExperienceDB, KnowledgeGraphDB, MutationDB), matplotlib visualization (`visualize.py`) |
| `compression/` | 3 | 377 | Plugin contract (`base.py`) + six numpy-only KV compression algorithms (`plugins.py`): FP8, INT4, low-rank SVD, sliding window, token eviction, head pruning |
| `harness/` | 5 | 380 | Experiment orchestration: `Router` (dispatch+retry+logging), `Evaluator` (compression ratio/latency/reconstruction error, plus real-model PPL/accuracy behind an explicit `InfrastructureUnavailable` guard), `RetryPolicy`, `BudgetEnforcer` |
| `integrations/` | 3 | 243 | `GEPAAdapter` (experience → reflective-prompt feedback), `Planner`/`Reflector` (rule-based "what to try next" / "which heads are unused" — deliberately not LLM-backed, so they're deterministic and testable) |
| `hope_core/` | 11 | 1,689 | The HOPE algorithm itself: `NeuronBank`/`ParentNeuron` (§4 below), `kernels.py`/`costs.py`/`merge.py` (the paper's math), `layer_state.py` (O(N) local-update orchestration + the cross-layer resync fix), `deft.py` (elasticity-based fine-tuning prep), `block_eviction.py`, `adapters.py` (Linear+BN / Conv+BN ↔ NeuronBank extraction), `encoding_loop.py` (the greedy prune/merge driver, plain + chain-aware), `report.py` (`StethoscopeReport`) |
| `debugger/` | 1 | 170 | Streamlit timeline+memory dashboard over `core.storage.SQLiteEventSink` output. **No automated test imports this file or `streamlit`/`pandas`/`plotly`** — confirmed by grep across `tests/*.py`; it's exercised manually, not in CI. |
| `examples/` | 8 | 1,336 | `synthetic_model.py` (pure-numpy transformer + `SyntheticHookableModel`, the shared test fixture), `run_demo.py` (full synthetic pipeline, all 6 subproblems in sequence), `run_gemma.py` (real HF Gemma pipeline), `run_mutation_demo.py` (causal KV-intervention demo), `quickstart.py` (HOPE walkthrough), `example_model_def.py`/`make_example_model.py` (CLI demo fixtures) |
| `scripts/` | 2 | 659 | `toy_reproduction.py` (small synthetic HOPE reproduction, actually run — see `toy_reproduction_results.json`), `reproduce_paper_resnet50.py` (real ImageNet/ResNet-50 reproduction — gated behind a `--i-have-data-and-time` flag and explicitly documented as never having been run in this sandbox) |
| `cli.py` | 1 | 369 | Click CLI, two command groups: `hope inspect/compress/deft-setup` (weight side) and `hope kv parse-modifier/benchmark/gemma/demo` (KV side) |
| `tests/` | 14 | 3,399 | 204 collected tests (see §6) |

Total: 55 Python files, 10,328 lines, git history of 5 commits (2026-07 to
2026-08-08), MIT-licensed. Working tree is CRLF throughout (git blobs are
LF — pure line-ending normalization noise from however this zip was
produced, confirmed via `git diff --stat` showing 1:1 insertions/deletions
on every tracked file; not a content difference, not something to "fix").

---

## 3. Existing KV architecture

This is the subsystem the spec calls "the first mature implementation of
the general abstraction," and it earns that description.

**`core/hooks.py` (455 lines) — the two capture mechanisms:**

- `CacheTap`: retroactively changes a live HF `Cache` instance's class
  (`instance.__class__ = new_subclass`) so its own `update()` method is
  intercepted *without* replacing the object identity, subclassing, or
  monkey-patching `Cache.update` globally. This exists because an earlier,
  more conventional approach (subclass instantiation, module-name-matching
  for attention modules) broke on Gemma 3's dual-path architecture
  (`Gemma3TextConfig` inside a multimodal `Gemma3Config`, separate
  `sliding_attention`/`full_attention` cache regions per
  `layer_types[i]`). This is the concrete instance of the spec's own
  §3 lesson ("module-name-based attention identification is insufficient
  for multimodal architectures such as Gemma 3... prefer the authoritative
  semantic interface") — it isn't a principle the spec is introducing to a
  naive codebase, it's a principle this codebase already independently
  discovered and encoded, under real pressure from a real model. Verified
  directly: `tests/test_real_gemma3_offline.py`'s 9 tests, which construct
  a real (tiny, offline) `Gemma3ForCausalLM` and exercise `CacheTap`
  against it, all pass in this session.
- `KVHookManager`: a conventional `register_forward_hook`-based path for
  attention *weights* specifically (not cache state), used where a
  semantic interface isn't available. Its module discovery
  (`looks_like_attention_module`, `find_attention_modules`) is
  class-name + leaf-path-segment + duck-typing based, with explicit
  exclusions for nested projection children and vision-tower modules
  (Gemma 3 is multimodal) — i.e., exactly the "module-name matching" the
  spec warns is fragile, but scoped deliberately to the one place
  (attention weights) where no better interface exists, and covered by
  6 dedicated tests (`test_looks_like_attention_module_*`,
  `test_find_attention_modules_*`) including two negative tests that
  encode the multimodal lesson directly.

**`core/cache_adapters.py` (181 lines):** normalizes four different cache
"shapes" seen across HF versions/models (legacy tuple-of-tuples,
`DynamicCache.layers[i]`, a `{key_cache: [...], value_cache: [...]}`
convention, and a bare `{k: ..., v: ...}` per-layer convention) behind one
`get_layer_kv`/`set_layer_kv`/`describe_cache_format` interface. This is
already, in miniature, exactly the kind of format-abstraction the new
`ModelAdapter` layer needs to do at the whole-architecture level.

**`core/kv_modifiers.py` (360 lines):** seven interventions (`Null`,
`Zero`, `Noise`, `SwapTokens`, `HeadMask`, `MemoryDecay`, `LowRank`), each
with a `describe()` that round-trips through JSON, plus
`parse_modifier_spec()` for a CLI-friendly string grammar
(`name:key=val;key2=val2`). This is a real, tested (28 tests), reversible
intervention framework — just scoped to KV tensors only. It is the
concrete prior art the spec's §16 generic `Intervention` abstraction
should generalize, not replace.

**`core/inspector.py` (164 lines):** pure-numpy statistics (L2 norms,
cosine similarity, duplicate-token clustering, token-source split,
per-head importance from attention-weight entropy) — deliberately
torch-independent so it's directly unit-testable without a model. This is
the measurement layer the spec's abstraction chain
(`OBSERVE → MEASURE → COMPARE`) already has a template for.

**`core/storage.py` (416 lines):** `SQLiteEventSink`/`JSONLEventSink`/
`InMemoryEventSink` behind one `EventSink` protocol; `ExperienceDB`
(compression-experiment log, consumed by the GEPA/Autoresearch
integrations); `KnowledgeGraphDB` (generic entity/relationship tables —
already shaped close to the spec's §20 ontology, just populated with KV/
attention-head facts today, not the full Model→Layer→Attention→Head/Q/K/V
graph the spec describes); `MutationDB` (before/after KV-mutation log with
output-diff detection — the direct ancestor of the spec's §18 causal
experiment engine, currently scoped to "did the generated tokens change,"
not internal-state propagation).

**Compression/harness/integrations layer:** six plugins behind one
contract (`compress`/`decompress`/`run`, `CompressedResult` with
`compression_ratio`/`latency_ms`/`metadata`), a `Router` that adds
retry+budget+logging around any plugin call, an `Evaluator` that reports
`InfrastructureUnavailable` rather than a fabricated number when no real
model/dataset is wired in (verified directly: real PPL/accuracy
computation code exists and is correct, but is genuinely gated — this
sandbox has neither GPU nor huggingface.co network access, so those two
methods were read but not independently re-executed with real data). This
`InfrastructureUnavailable` pattern is exactly the discipline the spec's
§30 ("never silently fabricate data... report UNAVAILABLE") is asking
for, applied one level up (evaluation results, not observation results) —
again, prior art to extend, not a gap to fill from scratch.

**Real-model integration (`examples/run_gemma.py`, 320+ lines):** wires
`CacheTap` onto a real `Gemma3ForCausalLM.forward()`, correctly uses
`model.config.get_text_config()` when present (composite multimodal
configs don't expose `num_hidden_layers` etc. directly on the top-level
config — this is the second concrete instance of the "authoritative
interface over guessing" lesson, at the config layer this time). No
residual-stream, attention Q/K/V, or MLP/neuron hook exists anywhere in
this file or `core/` — confirmed by reading the full generation loop.
Everything the spec's Phases 3–5 ask for is genuinely new territory, not
a refactor of something that already exists under a different name.

---

## 4. Existing HOPE architecture

`hope_core/` implements a specific algorithm end-to-end, independent of
the KV-cache subsystem:

- `NeuronBank` (182 lines): the atomic unit — all neurons at one
  prunable point (a Linear/Conv2d + BatchNorm pair), parameterized as
  `(w_raw, gamma, beta, running_mean, running_var, w_out)`.
  `effective_input()` absorbs BN into the incoming weight (paper eq. 1);
  `capacity()` computes each neuron's Hilbert-space norm (paper §5).
  `w_out` is explicitly documented as *not* an independent quantity — it
  is the downstream layer's incoming weight, transposed — which is the
  root of a real, found-by-testing bug class (see below).
- `kernels.py`/`costs.py`/`merge.py`: the paper's actual math (self-kernel,
  cross-kernel, prune/merge cost, optimal-parent computation for a merged
  neuron).
- `layer_state.py` (316 lines): orchestrates prune/merge decisions with an
  O(N) local update per action instead of an O(N²) full recompute (paper
  §10), with three documented, test-verified correctness pitfalls in its
  own module docstring: (1) stable IDs vs. shifting positions after a
  removal, (2) caching pair *geometry* rather than the final assembled
  cost (which depends on layer-wide state that changes every action), and
  (3) a bounded, *accepted* staleness in cached parent geometry (~1–2%
  relative drift observed in a 10-neuron/6-action test scenario), with a
  `refresh_all()` escape hatch documented as trading the O(N) guarantee
  for exactness when a caller needs it. This is unusually careful,
  self-aware engineering — the kind of thing an adversarial review would
  normally have to go looking for problems to find, already written down
  by the original author.
- `resync_downstream_consumer()`/`resync_upstream_provider()`: fixes a
  real cross-layer bug (documented in `NOTES.md` and the git log's Aug 8
  commit) where pruning/merging one layer silently left an adjacent
  layer's cached `w_out`/`w_raw` stale. This was found by a test failure,
  not anticipated in advance — worth noting because it's a direct preview
  of exactly the kind of cross-component consistency bug the new
  Stethoscope's causal-propagation tracking (spec §18) will need to get
  right for residual/attention/MLP state, not just HOPE's own weights.
- `deft.py`/`block_eviction.py`: elasticity-based fine-tuning prep and
  final-block eviction, both built on the same `NeuronBank`.
- `adapters.py`: `extract_linear_bn_bank`/`extract_conv_bn_bank` — the
  only place `hope_core` touches real `nn.Module` objects.
- `report.py`: `StethoscopeReport.from_loop(...)` — turns an
  `encoding_loop` audit trail into a report (capacity retained, distortion
  incurred, actions taken). **Naming note, carried forward from this
  audit into the Phase 1 design (see §12, §13):** this class is the
  origin of the word "Stethoscope" already being a live, tested,
  documented Python identifier in this codebase (`hope_core.report.
  StethoscopeReport`, used in `cli.py` and `examples/quickstart.py`, 8
  dedicated tests in `tests/test_report.py`). It is unrelated to the
  spec's new unified instrumentation API — a static, weight-space,
  post-hoc compression report, not a runtime observability facade — but
  the name overlap is real and worth a reader's attention once a new
  `stethoscope/` package exists alongside it.

**Known, self-documented limitation directly relevant to Phase 5 later:**
`docs/weight_compression.md`'s own "Known limitations" section states the
capacity/merge math is derived for ReLU-family activations; GELU/SiLU/
SwiGLU (i.e., every MLP in a modern decoder-only transformer, including
Gemma) changes the exact capacity number, though the qualitative
pruning/merging behavior is expected to still be directionally useful.
This is exactly the boundary the spec's §15 ("HOPE integration... Which
neurons can be pruned?... What happens when the neuron is removed?") will
eventually need to confront directly — flagged here so Phase 5/8 doesn't
rediscover it late.

---

## 5. Existing experiment infrastructure

Three persistence layers exist today, all in `core/storage.py`, all
SQLite-backed with an in-process fallback:

1. **`EventSink` family** — append-only, one row per (layer, token) scalar
   observation. This is what `debugger/app_streamlit.py` reads, and
   structurally the closest existing thing to the spec's §9
   `ObservationEvent` — but scoped to KV-cache scalars only today
   (position, cache size, latency — not component/phase/statistics in the
   generalized sense).
2. **`ExperienceDB`** — one row per compression *experiment*
   (plugin, config, compression ratio, PPL delta, latency, pass/fail, a
   rule-based one-line "lesson"). This is the compression-tuning log the
   `GEPAAdapter`/`Planner`/`Reflector` read and write; it is not a
   general causal-experiment record.
3. **`KnowledgeGraphDB`** — generic `upsert_entity`/`upsert_relationship`
   over two SQLite tables. Currently populated with `Head` entities and
   underused/keep verdicts by `Reflector.analyze_head_usage`. The schema
   is already generic enough to hold the spec's §20 ontology
   (Model/Layer/Attention/Head/Q/K/V/O/MLP/Neuron/Norm/Residual/Token/
   Parameter/Experiment nodes and the `represented_by`/`attended_to`/
   `activated_by`/`transforms`/`caused_change_in` edges) — it just isn't
   populated with most of that yet, because most of those component types
   don't exist as observable things yet.
4. **`MutationDB`** — before/after KV-mutation records with an
   output-token-diff check (`detects_unchanged_output`, tested). This is
   the direct, narrow ancestor of the spec's §18 causal experiment engine:
   it already has the right shape (baseline vs. intervention, did the
   output change) but only for KV mutations, and only at the output level
   — no internal-state propagation tracking (first-divergent-layer,
   maximum-divergence-layer, recovery) exists anywhere yet.

`examples/run_mutation_demo.py`, run directly in this session, reproduces
the CHANGELOG's documented memory-decay compounding figure exactly
(0.5314) — independent confirmation that this whole chain (modifier →
mutation record → output-diff detection) genuinely works end-to-end
against a real forward pass today, at the KV-only scope it was built for.

---

## 6. Existing tests

**Ground truth, established by direct execution in this session** (a
fresh venv, `numpy`/`scipy`/`torch`/`transformers`/`matplotlib`/`click`/
`pyyaml` installed — deliberately *not* `pandas`/`scikit-learn`/`plotly`/
`streamlit`, confirmed unnecessary by grep: nothing under `tests/`
imports any of those four, or `debugger.app_streamlit`, at all):

```
pytest tests/  →  204 passed, 0 failed, 0 skipped, 0 errors  (35.3s)
```

This is **more** than `README.md`'s own stated count of 187 — not a
regression, the opposite: the repo has grown since the docs were last
synced (the Aug 8 "changes and bugfixes" commit added
`test_encoding_loop.py`, `test_layer_state.py` growth, and
`test_neuron.py` growth, per `git log --stat`, and `PROJECT_STATUS.md` is
dated 2026-08-06, two days before that commit). The 204-vs-199-raw-`def
test_`-count discrepancy is fully explained by one
`@pytest.mark.parametrize("plugin_name", [...6 plugins...])` on
`test_plugin_round_trip_shape` in `test_core.py` (1 function → 6
collected tests; 199 − 1 + 6 = 204). Not a mystery, checked.

`tests/test_real_gemma3_offline.py`'s 9 tests were also run in isolation
and verified individually passing, against a real (not mocked)
`transformers.Gemma3ForCausalLM` built offline with a tiny config
(2 layers, 4 query heads / 2 KV heads — GQA, sliding-window layer 0 /
global layer 1) — this is the single most valuable existing test fixture
in the repo for Phase 1 validation, because it's a genuine, offline,
network-free way to construct a real HF model with the exact
architectural properties (GQA, mixed attention types, composite config)
that make Gemma the hard case. It's reused directly rather than
reconstructed — see §13.

`python -m examples.run_demo` was also run directly. It reproduces every
number in `success_criteria.md`'s scorecard, including the one documented
failure (`hook_and_inspector_overhead_under_10pct` — hook overhead on a
tiny synthetic model is dominated by Python-level dict/list bookkeeping
per call, not the underlying algorithm; this is stated plainly in the
repo's own docs and confirmed here rather than taken on faith).

**Coverage breadth** (function names surveyed directly, not inferred):
56 tests in `test_core.py` alone span hook mechanics (`CacheTap`
object-identity preservation, isinstance-safety, per-token event counts,
absolute-position tracking across prefill/decode, capture-integrity
verification), all four cache-format adapters, inspector statistics, all
six compression plugins (parametrized + individual edge cases), retry/
budget policies, and all three DB types. `test_kv_modifiers.py` (28
tests) covers all seven modifiers plus the CLI-string parser exhaustively,
including boundary cases (`out_of_range_position_is_ignored_not_an_error`,
`rejects_bad_rate`). This is a genuinely well-tested codebase, not one
with a thin veneer of tests over untested internals.

**Gaps, stated plainly:**
- `debugger/app_streamlit.py` has zero automated coverage (manual-only,
  per its own module docstring's "Design note").
- `core/visualize.py` (matplotlib rendering) has zero direct test
  coverage either — `run_demo.py` calls it and doesn't crash, which is
  weak evidence of correctness, not a real test.
- `scripts/reproduce_paper_resnet50.py` has never been run in any
  environment this project has had access to (no GPU, no ImageNet) —
  stated as such in the script's own header and in `PROJECT_STATUS.md`'s
  scorecard. This is the one place in the repo where "validated on real
  model" cannot honestly be claimed at any scale beyond the toy
  reproduction.

---

## 7. What should be preserved

Everything in §3–§6 that is described above as tested and working. In
priority order, because Phase 2+ will build directly on top of these and
regressing any of them would be a real cost, not a cosmetic one:

1. **`CacheTap`'s retroactive-subclassing mechanism and the lesson behind
   it.** This is the one piece of the existing codebase the spec calls
   out by name (§3) as something not to regress from. It should become
   the reference pattern for how Phase 2's `Observer` attaches to any
   *other* authoritative interface a future component type turns out to
   have (there is no reason to assume attention/MLP/residual will each
   have as clean an interface as `Cache.update()` — see §9 below).
2. **`core.cache_adapters`'s format-normalization approach** — the
   template for how `ModelManifest`/`ComponentRegistry` should treat
   "the same logical thing, several possible physical shapes" generally.
3. **The `InfrastructureUnavailable` / honest-failure discipline** in
   `harness/evaluator.py`, and the "report what's discarded, don't invent
   zeros" discipline in `compression/plugins.py`'s shape-changing
   plugins. Both are directly generalizable to the spec's §30 `UNAVAILABLE`
   requirement — this codebase already has the right instinct, twice.
4. **All 204 passing tests, unmodified**, as the regression floor for
   every phase from here forward.
5. **`hope_core`'s independence from the KV-cache subsystem.** The spec's
   §15 explicitly warns against redefining the Stethoscope as a HOPE
   compression framework; the current code decoupling (no imports either
   direction) is the correct starting state for that boundary and should
   stay that way until Phase 8 gives a specific, tested reason to add a
   connection.
6. **The two-command-group CLI shape**, `config/*.yaml`-driven target
   selection, and the `data/`/`outputs/` gitignored-and-regenerated
   convention — none of these need to change for Phase 1, and changing
   them without a concrete driver would be exactly the kind of
   "replace proven mechanisms... for aesthetic consistency" the spec's
   §34 rules out.

---

## 8. What should be refactored

Deliberately a short list. Nothing found in this audit needs to change
*to make Phase 1 possible* — Phase 1 is additive by construction (see
§13). What follows is refactoring that later phases will genuinely need,
named now so it isn't rediscovered under time pressure:

1. **KV-specific interventions/events will need to become instances of
   the generic `Intervention`/`ObservationEvent` types, not be replaced
   by them.** The spec is explicit about this (§13: "Do not replace
   working KV functionality merely to make it 'generic'"; Phase 2's own
   description: "Migrate KV instrumentation onto this abstraction *while
   preserving compatibility*"). Concretely, this later means
   `core.kv_modifiers.KVModifier` subclasses should each be expressible
   as one `Intervention(component=ComponentType.KV_CACHE, operation=...)`
   without `core/kv_modifiers.py`'s existing public functions changing
   signature — an adapter/wrapper relationship, not a rewrite. Not
   attempted in this pass; flagged for Phase 6.
2. **`KVEvent`/`core.storage.EventSink` vs. the new `ObservationEvent`.**
   Same relationship as above — Phase 2's job, not Phase 1's. The two
   should end up either (a) `ObservationEvent` gains a KV-specific view/
   projection that the existing `EventSink` schema can still ingest
   unchanged, or (b) `EventSink` becomes one storage backend among
   several that Phase 2's generic observer can write to. Which one is
   right depends on decisions Phase 2 hasn't made yet (see `TASKS.md`
   §4's open questions) — deferred deliberately rather than guessed at
   here.
3. **`pyproject.toml`'s package `include` list** needs the new top-level
   package added when it exists (mechanical, one line — see §13/`TASKS.md`
   §2). Not urgent, but a real, easy-to-forget gap: `examples`,
   `debugger`, and `scripts` are *also* not in that list today, which is
   fine for them (they're not meant to be pip-installed as a library),
   but would silently break `pip install -e .` for a new `stethoscope`
   package if forgotten.
4. **Documentation currency** — `README.md`'s "187 tests" line and
   `PROJECT_STATUS.md`'s 2026-08-06 date are both two commits stale
   relative to the actual tree (see §6). Not a code defect, but worth
   fixing in the same pass that touches those files for the Phase 1
   status update anyway (see `TASKS.md` §3, item 3), rather than as a
   separate effort.

**Explicitly not recommended:** collapsing `compression.plugins`'s
offline `compress()`/`decompress()` contract and `core.kv_modifiers`'s
live `KVModifier.apply()` contract into one interface. They solve
different problems (batch transform-and-measure vs. live single-call
mutation with `describe()`-based provenance) and the repo's own
`weight_compression.md` explicitly considered and rejected an analogous
unification (a shared `RepresentationOperator` between `hope_core.merge`
and `compression.plugins`) for the same reason: forcing two genuinely
different operation shapes through one interface tends to produce an
interface that fits neither well. The same restraint should apply here.

---

## 9. What is missing

Mapped directly against spec §1's question list and §35's deliverable
list — everything below is confirmed absent by reading, not assumed
absent because it wasn't mentioned in the docs:

- **No `ModelAdapter`, `ModelManifest`, or `ComponentRegistry` in any
  form.** `config/models.yaml` describes targets as flat key/value
  entries interpreted ad hoc by `run_gemma.py`/`run_demo.py`; there is no
  machine-readable, queryable representation of "what components does
  this model have."
- **No stable `ComponentID` scheme.** Nothing in the repo names a
  component more precisely than a bare layer index (an `int`) passed
  around as a function argument. There is no `"model/layer/17/attention/
  q"`-shaped identifier anywhere.
- **No residual-stream access at all** (spec §10, called out as one of
  the highest-priority capabilities). Nothing in `core/` or `examples/
  run_gemma.py` reads a layer's hidden-state output; only cache K/V and
  (via `KVHookManager`) attention weights are ever touched.
- **No logit lens** (spec §11). No code anywhere applies the LM head to
  an intermediate layer's representation.
- **No Q/K/V/scores/probabilities/context access as first-class,
  per-component observations** (spec §12). `KVHookManager` captures
  attention *weights* (the post-softmax probabilities) via forward hooks
  for entropy/importance statistics — genuinely useful, but it is one
  scalar-derived signal from one point in the attention computation, not
  the six-stage breakdown (Q, K, V, scores, probabilities, context, O)
  the spec asks for.
- **No MLP or neuron *activation* instrumentation** (spec §14). To be
  precise about a real ambiguity here: `hope_core.NeuronBank` models a
  "neuron" as a static, weight-space object (fused Linear/Conv+BN
  parameters, used for capacity/pruning math) — a genuinely different
  concept from the spec's runtime, per-token, per-forward-pass
  *activation* neuron. Neither this repo nor the spec's Phase 5 has
  built the activation-side concept yet; the two are related (Phase 8's
  entire job is connecting them) but must not be conflated as the same
  thing already existing under a different name. They aren't.
- **No generic `Intervention` abstraction, `Observer`, or `CapturePolicy`**
  (spec §7, §16, §22–23). Interventions today are KV-modifier-specific
  (`core.kv_modifiers`); there is no notion of "intervention" that could
  target a head, a neuron, or a residual-stream position.
- **No causal experiment engine beyond KV-mutation output-diffing**
  (spec §18) — see §5 above; `MutationDB` is real, narrow prior art, not
  a generalized version of what's asked for.
- **No `stethoscope.observe(...)`/`.intervene(...)`/`.experiment(...)`/
  `.compare(...)` API surface, or anything resembling it** — confirmed by
  grep: no top-level `stethoscope` package/module exists anywhere in the
  repository today (the only existing hit for the string "stethoscope" as
  a Python identifier is `hope_core.report.StethoscopeReport` — see §4,
  §12).
- **No architecture-portability test against a second model family.**
  Every real-model test and example targets Gemma specifically. The spec's
  §24/§34 "at minimum, design the system so it can eventually support
  different families" is not yet exercised anywhere, even lightly.

---

## 10. Proposed target architecture

This is the spec's own §2/§4 diagram, annotated with what already exists
(✅), what Phase 1 adds (🔵, this pass), and what's genuinely still open
(⬜):

```
                              HOPE STETHOSCOPE
                                     │
                    ┌────────────────┼─────────────────┐
                    │                │                 │
                    ▼                ▼                 ▼
              Observation      Intervention        Experiment
                Layer ⬜           Layer ⬜            Layer ⬜
             (Phase 2)     (Phase 6, generalizes    (Phase 7, generalizes
                            core.kv_modifiers ✅)     core.storage.MutationDB ✅)
                    │                │                 │
                    └────────────────┼─────────────────┘
                                     ▼
                          Measurement / Analysis
                        (core.inspector ✅ for KV;
                         residual/attn/MLP-level ⬜)
                                     │
                                     ▼
                          Evidence / Provenance
                     (core.storage: EventSink/ExperienceDB/
                      KnowledgeGraphDB/MutationDB ✅ for KV;
                      generalized ontology ⬜, Phase 2/7)
                                     │
                                     ▼
                       Visualization / Reporting
                    (core.visualize ✅ for KV, untested;
                     debugger/ ✅ manual-only; new views ⬜)


                    Model Adapter Layer 🔵 (Phase 1, this pass)
                              │
              ┌────────────────┼──────────────────┐
              │                │                  │
         HuggingFace 🔵      Synthetic 🔵        Future ⬜
          Adapter            Adapter             (PyTorch-generic,
        (config-driven,    (wraps existing       non-HF frameworks)
         multi-family,      examples/
         not Gemma-only)    synthetic_model.py)
              │
    ModelManifest 🔵 → ComponentRegistry 🔵 → ComponentID 🔵
                              │
                    (feeds Phase 2's Observer,
                     not built yet — registry
                     names components, doesn't
                     bind to live tensors)
```

The key structural decision this audit is making explicit (elaborated in
§13): **Phase 1 sits entirely underneath and to the side of the existing
KV architecture, touching nothing in it.** The Model Adapter Layer's job
in this pass is *discovery* — producing a `ModelManifest` and a
`ComponentRegistry` that describe what components a model has and what
their canonical IDs are — not *observation*. Binding a `ComponentID` to a
live tensor at runtime (attaching a hook, reading a cache, wiring up
`CacheTap`) is Phase 2's `Observer`, explicitly out of scope here. This
split is not just spec-compliance box-ticking; it's the thing that makes
"without breaking existing KV behavior" verifiable by inspection rather
than by hope — if Phase 1 never imports `core`, `compression`, `harness`,
or `hope_core`, there is no code path by which it could regress them, and
the 204 existing tests passing unchanged is a direct proof of that rather
than a coincidence.

---

## 11. Migration plan

Phase-by-phase, cross-referenced to spec §26, with the concrete
dependency of each phase on what already exists:

| Phase | Depends on (existing) | Produces | Existing code touched |
|---|---|---|---|
| **1** (this pass) | Nothing from `core`/`hope_core` — deliberately decoupled | `stethoscope/{ids,manifest,registry,events}.py` + 2 adapters | None (see §13) |
| **2** | `core.storage.EventSink` (as a template, possibly as a backend), `core.hooks.CacheTap` (as the reference pattern for a semantic-interface `Observer`) | `Observer`, `CapturePolicy`, storage backend; **migrates KV onto `ObservationEvent`** | `core/hooks.py` gains a Phase-1-`ComponentRegistry`-aware path, additively — existing `CacheTap`/`KVHookManager` call sites keep working |
| **3** | Phase 2's `Observer`; needs a real per-layer hidden-state hook, which needs the HF adapter's layer-module discovery — new territory, `looks_like_attention_module`'s pattern is a template, not directly reusable (residual stream isn't attention-shaped) | Residual capture + trajectory/logit-lens analysis | New module(s) under `stethoscope/`; no existing file touched |
| **4** | Phase 2, 3; `core.hooks.KVHookManager`'s existing attention-weight capture as one *input* to the fuller Q/K/V/scores/probs/context breakdown | Generalized attention instrumentation | Likely wraps/extends `KVHookManager`, additively |
| **5** | Phase 2; `hope_core.NeuronBank` as a *distinct*, weight-space companion concept, not a dependency | MLP/neuron *activation* instrumentation | None required; `hope_core` stays decoupled until Phase 8 |
| **6** | Phase 2's `Intervention` shape; `core.kv_modifiers`'s 7 existing modifiers as the migration target | Generic `Intervention` framework; KV modifiers become instances of it | `core/kv_modifiers.py` — adapter layer added, public functions preserved (see §8, item 1) |
| **7** | Phase 6; `core.storage.MutationDB` as the direct, narrow ancestor | Causal experiment engine (baseline/intervention/compare/propagation) | `core/storage.py` likely gains a generalized experiment table alongside `MutationDB`, not replacing it |
| **8** | Phase 5, 7; `hope_core.NeuronBank`/`encoding_loop`/`report.StethoscopeReport` as the *target* system being connected, not modified | HOPE ↔ Stethoscope bridge | `hope_core/` ideally untouched — a new integration module calls into it, per spec §15's explicit "not a separate universe" framing meaning *connected*, not *merged* |
| **9** | All of the above | Multi-family validation, remove any Gemma-only assumptions that crept in | Test-only, in principle |

Each phase boundary above is also a regression-test boundary: the
existing 204 tests (growing as prior phases add their own) must stay
green before a phase is considered complete, per spec §28's regression
requirement — not deferred to the end.

---

## 12. Risks

In order of how much they could actually cost if mishandled, not in the
order they were discovered:

1. **Scope creep disguised as "just generalizing."** The spec itself
   names this risk repeatedly (§25, §34) and the existing codebase has
   already shown the discipline to resist it once (the rejected
   `RepresentationOperator` unification, §8 above). The concrete failure
   mode to watch for: touching `core/kv_modifiers.py`'s or
   `core/hooks.py`'s *public* behavior "while I'm in there" during a
   phase whose actual job is additive. Mitigated in this pass by the
   Phase 1 design not importing those modules at all (§10) — a structural
   guard, not just a promise.
2. **The residual-stream/attention/MLP hook-attachment problem (Phases
   3–5) has no existing template as clean as `CacheTap`'s.**
   `Cache.update()` is an unusually good authoritative interface — a
   single, always-called method with a stable signature. There is no
   equivalent single method for "read the residual stream after layer
   17's attention block" across arbitrary HF architectures; it will most
   likely mean forward hooks on the *decoder layer* module itself (a much
   more standardized point across families than attention submodules
   are), which is a real, not-yet-validated design bet. Flagged now,
   deliberately not resolved in Phase 1 — resolving it prematurely, before
   Phase 3 needs a real answer, would be exactly the "introduce arbitrary
   abstractions without corresponding tests" failure mode spec §34 warns
   against.
3. **The `stethoscope` naming decision (§4, §9 above).** Not a technical
   collision — `hope_core.report.StethoscopeReport` and a new top-level
   `stethoscope` package are different importable paths and Python
   doesn't care. It is a genuine human-readability risk: a new
   contributor grepping for "stethoscope" will find two unrelated things.
   Decision made in this pass (see §13): proceed with `stethoscope/` as
   the new package name anyway, because it's the name the entire
   repository (and the spec's own pseudocode, dozens of times) already
   uses for exactly this concept — avoiding it would trade a
   well-justified name for a manufactured one, for a cost (a docstring
   cross-reference) that's cheap to pay directly. Mitigation: a short,
   explicit cross-reference note added to `hope_core/report.py`'s
   docstring (see `TASKS.md` §2) — one place, once, rather than a
   different name everywhere forever.
4. **`ComponentRegistry` correctness silently drifting from reality.**
   Because Phase 1's registry is built from `ModelManifest` (i.e., from
   `config`) rather than from live module introspection (§13), there's a
   real risk it describes components a specific checkpoint's *actual*
   module tree doesn't have in exactly that shape (e.g., a fine-tuned
   variant with a modified MLP). This is an accepted, documented
   trade-off for Phase 1 specifically (config-driven discovery sidesteps
   the `named_modules()` fragility spec §34 warns against), but Phase 2's
   `Observer` — which *does* need to bind to live modules — must not
   silently assume the registry is always physically accurate. It should
   fail loudly (`UNAVAILABLE`, per spec §30) when a registered component
   can't actually be found on a given live model, rather than assume
   config and reality always agree. Flagged for Phase 2's design, not
   solved here.
5. **`pip install torch` pulling ~2.7GB of unneeded CUDA/nvidia/triton
   packages** in a CPU-only, disk-constrained environment (observed
   directly in this session: 87% disk utilization after one `pip install
   torch` with no CUDA hardware present, no `download.pytorch.org` access
   available to select a CPU-only wheel). Not a code risk, an
   environment/onboarding risk — worth a one-line README note for anyone
   reproducing this setup on a constrained machine, not worth solving in
   this repository.
6. **`hope_core`'s ReLU-family capacity-math assumption (§4 above)**
   surfacing as a real accuracy problem the moment Phase 8 actually
   connects HOPE to a GELU/SiLU/SwiGLU-activated real transformer MLP,
   rather than the ResNet/BN-Linear settings it was validated against. Not
   a Phase 1 risk, but cheap to name now so Phase 8 doesn't spend time
   rediscovering a limitation the weight-compression docs already state.

---

## 13. Minimal first implementation milestone

**Decision, locked in for this pass:** implement spec §26 Phase 1 exactly
as scoped — `ModelAdapter`, `ModelManifest`, `ComponentRegistry`,
`ComponentID`, `ObservationEvent` — as a new, additive, top-level
`stethoscope/` package that imports nothing from `core/`, `compression/`,
`harness/`, `integrations/`, or `hope_core/`, and that none of those
existing packages import from either. This is the concrete answer to
"without breaking existing KV behavior": there is no import edge by which
Phase 1 code can affect existing behavior, so the regression check is a
structural guarantee, confirmed by (not just asserted before) rerunning
the full existing test suite after Phase 1 lands.

**Scope, concretely:**

- `ComponentID` — stable, hashable, string-round-trippable identifier
  (`"model/layer/17/attention/q"`-shaped), built from canonical factory
  methods rather than free-form string concatenation at call sites, per
  spec §6's "not dependent solely on Python object names."
- `ComponentType` — the exact controlled vocabulary from spec §6
  (`Model`, `Embedding`, `Block`, `Norm`, `Residual`, `Attention` +
  `.Q`/`.K`/`.V`/`.Scores`/`.Probability`/`.Context`/`.O`, `KVCache`,
  `MLP` + `.Gate`/`.Up`/`.Activation`/`.Down`, `Neuron`, `LMHead`,
  `Logits`).
- `ModelManifest` and five sub-manifests (attention, MLP, normalization,
  embedding, cache) — every field `Optional`, with an explicit `notes:
  list[str]` for *why* something couldn't be determined, following the
  `InfrastructureUnavailable`/shape-changing-plugin honesty pattern
  already established in this codebase (§7) rather than inventing a new
  convention.
- `ComponentRegistry.from_manifest(manifest)` — **a pure function of a
  manifest, not of a live model.** This is the one real design choice in
  this milestone worth calling out on its own: it means "what components
  does this architecture have" (Phase 1) is fully decoupled from "how do
  I find the actual `nn.Module` that implements component X at runtime"
  (Phase 2). It sidesteps the `named_modules()` fragility spec §34 warns
  against entirely for this phase, at the cost of the drift risk named in
  §12, item 4 — an explicit, accepted trade-off, not an oversight.
  Individual neurons are *not* pre-populated into the registry (would be
  thousands of entries per layer, for a component nothing consumes yet);
  `ComponentID.neuron(layer, idx)` exists as an on-demand factory instead,
  matching spec §22/23's "do not instrument everything by default"
  applied to registry construction, not just capture.
- `ObservationEvent` — the exact field set from spec §9, as a standalone,
  serializable data structure. **No `Observer`, `CapturePolicy`, or
  storage backend in this pass** — those are explicitly Phase 2 per the
  spec's own phase breakdown, and building them now, before Phase 2 has a
  real producer/consumer to validate the shape against, would be
  speculative.
- Two `ModelAdapter` implementations: `SyntheticAdapter` (wraps the
  *existing* `examples/synthetic_model.py` fixture — not a reimplementation)
  and `HFTransformersAdapter` (config-driven — reads `model.config`,
  calling `get_text_config()` first when present, the same composite-config
  handling `run_gemma.py` already established — with multi-name-tolerant
  field lookup so it isn't Gemma-specific by construction: e.g.
  `num_hidden_layers`/`n_layer`, `num_attention_heads`/`n_head`,
  `hidden_act`/`activation_function`, etc.). `transformers` is imported
  lazily inside methods, not at module scope, matching the lazy-import
  discipline already used throughout `core/` and `harness/`.

**Validation plan, following spec §24's exact progression rather than
skipping straight to "real model":**

1. Unit tests against hand-built manifests/mock configs (fast, no model
   construction).
2. `SyntheticAdapter` against the repo's own existing synthetic fixture
   (`num_layers=4, num_heads=4, head_dim=16` from `config/models.yaml`'s
   `synthetic` target / `examples/synthetic_model.py`'s actual defaults).
3. `HFTransformersAdapter` against a **real**, offline-constructed
   `Gemma3ForCausalLM`, reusing `tests/test_real_gemma3_offline.py`'s
   proven construction pattern directly (GQA: 4 query / 2 KV heads,
   head_dim 8, mixed sliding/global `layer_types`) — genuine re-validation
   against the hardest real case this repo already has offline access to,
   not a new synthetic stand-in for it.
4. A second family — a tiny, offline `LlamaForCausalLM`/`LlamaConfig` —
   specifically to give spec §24's "test against more than one
   architecture" and §34's "do not hardcode Gemma" concrete, checked
   evidence in this very first milestone, rather than a promise deferred
   to Phase 9.
5. Full existing suite (`pytest tests/`) rerun afterward, confirmed
   204+N passed / 0 failed — the actual regression check, executed, not
   assumed.

**Two small, deliberately deferred, existing-file edits** (not made until
the new package's own tests are green, so a mid-work interruption never
leaves an existing file half-changed): add `"stethoscope*"` to
`pyproject.toml`'s package `include` list, and a short docstring
cross-reference in `hope_core/report.py` disambiguating `StethoscopeReport`
from the new package (§12, item 3). Both are mechanical and reversible.

This milestone is deliberately narrower than everything spec §35 lists as
final deliverables — it is Phase 1 of 9, not the finished system.
Continuation status, exact file list, and what's next live in `TASKS.md`
at the repo root, updated as each piece lands rather than at the end.
