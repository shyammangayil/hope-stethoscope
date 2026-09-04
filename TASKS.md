# TASKS — HOPE Stethoscope architectural evolution

**Read this file first in any new session.** It is the single source of
truth for "what has actually been done, verified, and decided" on the
KV-cache → model-wide-observability evolution described in
`docs/ARCHITECTURE_EVOLUTION_SPEC.md` (verbatim copy of the task spec —
read that first for the *requirements*; this file tracks *status* against
them, and `docs/ARCHITECTURE_DELTA_REPORT.md` holds the full Phase 0
audit + Phase 1 design rationale). Update the "Last updated" line and the
relevant section every time you make a milestone-sized change — not at
the end, as you go — since there is no guarantee of session/sandbox
continuity.

**Last updated: 2026-09-04. Phases 0, 1, and 2 are complete and
verified.** Phase 3 (residual-stream instrumentation) has not been
started. See §1 for the phase table, §2 for exactly what exists and how
it was validated, §3 for design rationale worth knowing before touching
this code, §4 for what Phase 3 should read first.

---

## 0. How to resume cold

1. Read, in this order: this file → `docs/ARCHITECTURE_EVOLUTION_SPEC.md`
   (requirements) → `docs/ARCHITECTURE_DELTA_REPORT.md` (Phase 0 audit +
   Phase 1 design rationale — Phase 2's own rationale is in §3 below,
   this file, since it postdates that report).
2. The repo has a real `.git` history (5 commits through "changes and
   bugfixes" on 2026-08-08) that predates this work. Everything from
   Phase 0/1/2 is uncommitted on top of that unless committed since —
   `git status`/`git log` first, don't assume.
3. Environment: Python 3.12.
   ```
   python3 -m venv venv
   pip install numpy scipy pytest click tqdm pyyaml torch transformers matplotlib
   pip install -e . --no-deps
   ```
   (`pip install torch` on plain PyPI pulls the CUDA build — ~2.7GB of
   unneeded nvidia/triton packages on a CPU-only box; prefer a CPU-only
   index if one is reachable.) `pandas`/`scikit-learn`/`plotly`/
   `streamlit` are not required for `pytest tests/`.
4. Ground truth, independently verified by actually running it:
   - `pytest tests/` → **310 passed, 0 failed, 0 skipped** (204
     pre-existing + 71 Phase 1 + 35 Phase 2). If you see fewer,
     something regressed or didn't collect — investigate before
     continuing, don't assume it's fine.
   - `tests/test_stethoscope_observer_real.py` and
     `tests/test_stethoscope_adapters_huggingface_real.py` both pass
     against real, offline-constructed Gemma3 (both) and Llama (the
     latter only).
5. **Only four existing files were ever edited, all additively, all
   individually verified via `git diff --ignore-space-at-eol` to
   contain exactly the intended change and nothing else:**
   `pyproject.toml` (one line), `hope_core/report.py` (one docstring
   paragraph), `README.md` and `PROJECT_STATUS.md` (status-table
   additions, in each document's own existing format). Every file under
   `core/`, `compression/`, `harness/`, `integrations/`, `hope_core/`
   (besides `report.py`'s docstring), `debugger/`, `cli.py`, and every
   existing test, is byte-for-byte what it was before this work. Re-run
   `git diff --ignore-space-at-eol --stat` yourself before trusting this
   claim if you're not sure it's still true.

---

## 1. Status by phase (per spec §26)

| Phase | Description | Status |
|---|---|---|
| 0 | Repository audit | **Done.** `docs/ARCHITECTURE_DELTA_REPORT.md`, all 13 required sections. |
| 1 | ModelAdapter, ModelManifest, ComponentRegistry, ComponentID, ObservationEvent | **Done.** §2 below. |
| 2 | Generic observation layer (Observer, CapturePolicy, storage backend); migrate KV onto it | **Done, for KV-cache only** — see §2/§3. Residual/attention/MLP observation is explicitly Phase 3/4/5, not attempted here. |
| 3 | Residual-stream instrumentation | Not started. §4 has notes for whoever starts this. |
| 4 | Attention instrumentation (Q/K/V/scores/probs/context/O) | Not started. |
| 5 | MLP/neuron instrumentation | Not started. |
| 6 | Generic intervention framework; migrate KV modifiers onto it | Not started. |
| 7 | Causal experiment engine | Not started. |
| 8 | HOPE integration | Not started. |
| 9 | Architecture portability (multi-family) | Not started, though Phase 1 already has real evidence for two families (Gemma3, Llama), reconfirmed at the Observer level in Phase 2. |

---

## 2. What exists now

### Documentation
- `docs/ARCHITECTURE_EVOLUTION_SPEC.md` — verbatim task spec.
- `docs/ARCHITECTURE_DELTA_REPORT.md` — Phase 0 deliverable (13
  sections) + detailed Phase 1 design rationale. Phase 2's own design
  rationale is in §3 below (this file), not retrofitted into that
  report, since the report is explicitly a Phase-0-dated artifact.
- `TASKS.md` — this file.
- `README.md` / `PROJECT_STATUS.md` — updated with honest, scorecard-
  style entries for both new-subsystem milestones, in each document's
  own existing format.

### Code: `stethoscope/` package (15 files)
Phase 1 (unchanged since its own completion):
```
stethoscope/__init__.py                    public re-exports; phase-roadmap docstring
stethoscope/ids.py                         ComponentType, ComponentID
stethoscope/manifest.py                    ModelManifest + 5 sub-manifests
stethoscope/registry.py                    ComponentInfo, ComponentRegistry.from_manifest()
stethoscope/events.py                      ObservationPhase, ObservationEvent
stethoscope/adapters/base.py               ModelAdapter ABC
stethoscope/adapters/synthetic.py          SyntheticAdapter
stethoscope/adapters/huggingface.py        HFTransformersAdapter, manifest_from_config()
```
Phase 2 (new):
```
stethoscope/capture.py                     CaptureMode enum, CapturePolicy (scope filters +
                                            detail/sampling — pure, no core/ dependency)
stethoscope/storage.py                     ObservationEventSink protocol, InMemory + SQLite
                                            implementations (mirrors core.storage's own design
                                            precisely — pure, no core/ dependency)
stethoscope/observer.py                    Observer + the KV-cache bridge -- THE ONE MODULE
                                            THAT IMPORTS core.hooks/core.storage, deliberately
                                            and documented (see its own module docstring for
                                            the full reasoning, and §3 below)
```
`stethoscope/observer.py` is the sole, intentional exception to "Phase 1
imports nothing from core/" — Phase 2's own spec-defined job is bridging
into the existing KV mechanism. It calls `core.hooks.CacheTap`/
`core.storage.EventSink`/`InMemoryEventSink` exactly as an existing
caller already does; it does not edit either module (verified via
`git diff`, not assumed — see §0.5 above).

### Tests: 106 new total (71 Phase 1 + 35 Phase 2), all passing
Phase 2 additions: `test_stethoscope_capture.py` (10),
`test_stethoscope_storage.py` (8), `test_stethoscope_observer.py` (14,
synthetic-model tier — real `CacheTap` against the real `NumpyKVCache`,
not a mock), `test_stethoscope_observer_real.py` (3, real offline
Gemma3, skipped not failed if transformers is absent).

Full suite: `pytest tests/` → **310 passed, 0 failed, 0 skipped.**

### The four existing-file edits (all additive, all individually diffed and confirmed exact)
1. `pyproject.toml` — `"stethoscope*"` added to
   `[tool.setuptools.packages.find] include`. Verified with a real
   `pip install -e . --no-deps`.
2. `hope_core/report.py` — one docstring paragraph disambiguating
   `StethoscopeReport` from the new package. No code change.
3. `README.md` — new-subsystem bullet, validation-table row, directory
   tree entry, quick-start snippet (verified to actually run, byte for
   byte as printed), test-count fixes (was stale at 187; now 310).
4. `PROJECT_STATUS.md` — new-subsystem status table, dated update note.

### Real findings from Phase 1 (carried forward, still relevant)
See the previous version of this file (in git history once committed)
or `docs/ARCHITECTURE_DELTA_REPORT.md` for full detail. Headlines:
`ComponentID.parent` does structural, not semantic, truncation — use
`.layer_id`. `mlp.gated` can't be reliably derived from HF config alone
across families — pass `mlp_gated=` explicitly. The installed
`transformers` (5.16.1) uses `rope_parameters`/`rope_scaling`, not a flat
`rope_theta` — `_ROPE_SIGNAL_NAMES` tries all three.
`HFTransformersAdapter.build_manifest()` trusts the live model's own
class name over `config.architectures`, which a freshly-constructed
config never populates.

### Real findings from Phase 2
- **`CacheTap` is hard-wired to write `KVEvent`s directly to ONE
  `EventSink` at construction time — there is no callback/hook point.**
  This is why `Observer.kv_cache_sink()` works by *impersonating* an
  `EventSink` (satisfying its Protocol: `write`/`read_all`/`close`) and
  forwarding every `KVEvent` to a real inner sink while also translating
  it, rather than by modifying `CacheTap` to accept a callback. Confirmed
  by reading `core/hooks.py` directly before designing anything — not
  assumed from the module docstring alone.
- **`NumpyKVCache` (the synthetic model's cache) already implements the
  exact interface `CacheTap` wraps** — confirmed in its own docstring
  ("This is the object core.hooks.CacheTap wraps") and by directly
  running `CacheTap` against it, unmodified, in
  `test_stethoscope_observer.py`. This is why Phase 2's KV-cache bridge
  has real dual-tier validation (synthetic AND real Gemma3) using the
  literal same `CacheTap` class for both, not two different capture
  paths that happen to produce similar-looking output.
- **`keeps_full_detail` initially returned `True` for `CaptureMode.NONE`**
  (only excluded `SUMMARY`) — internally inconsistent (a mode that
  captures nothing claiming to "keep full detail" of it), caught by a
  test that itself needed a second look to realize the test's
  expectation, not the code, was the more defensible one. Fixed by
  excluding `NONE` too, not by loosening the test.
- **What Phase 2 deliberately does NOT build:** the fully-automatic
  `with stethoscope.observe(model, components=[...]): output = model(...)`
  context-manager sugar from spec §7. Building it now would mean either
  auto-discovering which cache object an arbitrary `model.generate()`
  call uses internally, or having the Observer inject one — both real,
  unsolved, framework-specific questions that Phase 3/4/5's own component
  types will have more to say about before that API's shape should be
  locked in. What Phase 2 built instead: the validated mechanism
  underneath (`Observer.kv_cache_sink()`), usable with one extra line
  over today's existing `CacheTap(cache, sink)` call. See
  `stethoscope/observer.py`'s module docstring for the full reasoning —
  it's deliberately written at length there because this is exactly the
  kind of decision a future session might otherwise "fix" by building
  speculative sugar, which is precisely what spec §34 warns against.

---

## 3. Design decisions locked in during Phase 2 (why, not just what)

- **The KV-cache bridge wraps `CacheTap` by impersonating an `EventSink`,
  not by modifying `CacheTap` itself.** `_KVEventBridgeSink` in
  `observer.py` satisfies `core.storage.EventSink`'s Protocol
  (`write`/`read_all`/`close`) so `CacheTap` cannot tell the difference
  from a plain `SQLiteEventSink` — this is the concrete mechanism behind
  "preserving compatibility": zero lines of `core/hooks.py` change, and
  an existing caller who never touches `stethoscope` sees no difference
  at all (verified directly by
  `test_bridge_does_not_change_what_the_inner_kvevent_sink_receives`).
- **`ObservationEventSink` is a new, separate storage backend, not a
  KV-specific projection bolted onto `core.storage.EventSink`.** This
  was an open question at the end of Phase 1 (see the old TASKS.md §4);
  resolved by building the thing `ObservationEvent`'s richer shape
  (three dict-valued fields vs. `KVEvent`'s one) actually needed, mirroring
  `core.storage`'s proven design (Protocol + InMemory + SQLite,
  JSON-serialized extras, the same `read_all()`-without-a-filter footgun
  documented the same way) rather than forcing a shape mismatch to avoid
  writing a second, structurally similar module.
- **`CapturePolicy`'s five modes split into two independent axes**: which
  components/layers/tokens are in scope (filters, apply regardless of
  mode) vs. how much detail survives for an in-scope event (`mode`
  itself). `SUMMARY` thins per-event detail; `SAMPLED` thins event
  *count* (deterministically — every Nth in-scope event, not randomly,
  so tests stay reproducible); `SELECTED` and `FULL` both keep full
  detail and differ only in caller intent, not behavior.
- **Sampling is deterministic, not random.** A random sampler would make
  "did SAMPLED mode work" tests flaky or require seeding gymnastics for
  no real benefit at this milestone — "every Nth" is simpler, fully
  testable, and just as legitimately a "sample" for spec §22's purposes.
- **`Observer.observe(component_type)` raises `NotImplementedError` for
  anything except `ComponentType.KV_CACHE`**, rather than silently
  returning a sink that would just never produce events. Spec §30: never
  silently fabricate (or, here, silently omit) — a caller asking to
  observe attention today should get a clear error, not quiet nothing.

---

## 4. Where Phase 3 should start

Phase 3 (spec §26): residual-stream capture and analysis — spec §10
calls this one of the highest-priority capabilities. Concretely, read
first:

- **This is genuinely new territory, unlike Phase 2.** Phase 2 had
  `CacheTap` as an existing, proven, authoritative-interface mechanism to
  bridge into. Nothing in `core/` reads a layer's hidden-state output
  today (confirmed in the Phase 0 audit, delta report §9 — re-confirm
  it's still true before assuming otherwise, in case something changed
  since). There is no equivalent "just impersonate an existing sink"
  trick available for residual capture.
- **The real design bet, flagged but not resolved in Phase 0/1** (delta
  report §12, risk 2): residual-stream capture most likely means forward
  hooks on the *decoder layer* module itself (`register_forward_hook` on
  `model.model.layers[i]`, roughly) rather than on any attention/MLP
  submodule specifically — decoder-layer-as-a-whole is more standardized
  across HF families than attention/MLP internals are, but this has not
  been validated against a real model yet. Do that validation BEFORE
  building the general abstraction around it, the same way Phase 2 first
  confirmed `CacheTap`'s exact interface (by reading `core/hooks.py`
  directly) before designing the bridge against it.
- **`ObservationEvent`/`CapturePolicy`/`ObservationEventSink` should not
  need to change shape for this** — they were designed generally, not
  KV-specific, precisely so Phase 3 could reuse them. `ComponentID.residual(layer,
  point)` and the four canonical residual tap points
  (`pre_attention`/`post_attention`/`pre_mlp`/`post_mlp`) already exist
  in `stethoscope/ids.py` and are already populated in
  `ComponentRegistry` (Phase 1) — Phase 3's job is *binding* those
  existing IDs to live tensors, not inventing new ones.
- **Follow the same validation discipline**: mock/pure unit tests first,
  then the synthetic model (its `attend()` method already computes a
  residual add — `x = x + out` in `SyntheticTransformer.attend`, so a
  synthetic residual-capture test is possible without torch), then real
  offline Gemma3, then a second family (Llama) if the first pass doesn't
  already cover it. Don't skip straight to "real model" — Phase 2's own
  mock/synthetic tiers caught real bugs (the architecture-name bug in
  Phase 1, the `keeps_full_detail` bug in Phase 2) that the real-model
  tier alone would have made much slower to isolate.

Not Phase 3's job, still open, not urgent: the spec §7 context-manager
sugar (`with stethoscope.observe(...)`) — see §2 above for why it's
deliberately deferred, and reconsider once Phase 3/4/5 give a fuller
picture of what the API actually needs to support before locking in its
shape. `hope_core.NeuronBank` vs. the future `ComponentType.NEURON` —
still Phase 8's job, not sooner.
