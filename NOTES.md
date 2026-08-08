# NOTES.md — paper-to-code traceability

Source of truth: arXiv:2607.21366v1 (Mobahi & Bartlett, Google DeepMind / UC
Berkeley, submitted 23 Jul 2026), fetched and cross-checked directly against
the primary text during implementation (not reconstructed from memory).

This file exists so that "which equation does this function implement" is
answerable without re-deriving anything, and so the two intentional
simplifications in the codebase are documented in exactly one place rather
than scattered across commit messages.

## Verified exact matches (implemented as literally specified in the main text)

| Code | Paper |
|---|---|
| `neuron.py: effective_input()` | eq. 1: `w_eff_in = (gamma/sqrt(sigma^2+eps)) * w_raw`, `b = beta - gamma*mu/sqrt(sigma^2+eps)` |
| `kernels.py: self_kernel()` | eq. 3, "Self-Kernel of ReLU Neurons" |
| `kernels.py: cross_kernel()` | eq. 4-5, "Cross-Kernel of ReLU Neurons" (paper's own documented zero-bias approximation, footnote a) |
| `neuron.py: capacity()` | section 5, "Kernel Formulation": `\|\|f_i\|\|_H = \|\|w_out,i\|\|_2 * sqrt(K(i,i))` |
| `costs.py: prune_cost()` | eq. 6, `J_prune = N \|\|f_i\|\|_H / (E_a - \|\|f_i\|\|_H)` |
| `costs.py: merge_distortion_squared(), merge_cost()` | eq. 6, `J_merge`, where `D^2 = \|\|f_i-f_p\|\|^2 + \|\|f_j-f_p\|\|^2` |
| `merge.py: _optimal_direction_2d_restricted()` | eq. 14, rank-2-restricted eigenproblem for `û` |
| `merge.py` scale computation | eq. 12, `s* = (a + b*E_rem) / (2*E_rem + b)`, section 7.1.2 |
| `merge.py` effective-parameter split | eq. 15, Frobenius-ratio-preserving `(w_tilde_p, w_out_p)` split |
| `merge.py` physical parameter recovery | eq. 16-18 exactly, active regime (`gamma_p^2 >= eps`) |
| `costs.py: distortion_rate()`, action selection | eq. 23, `k* = argmin J_k / DeltaP_init_k` (static surrogate, not live parameter count — see section 9's "Dantzig axiom of item independence" discussion for why) |

**Important correction during implementation:** `D^2` in eq. 6 is `\|\|f_i-f_p\|\|^2
+ \|\|f_j-f_p\|\|^2` — distance from the parent to *each child individually* —
not `\|\|f_i+f_j-f_p\|\|^2` (sum reconstruction). Confirmed independently by
Appendix A.6 and the related-work section: merging is a **constrained
rank-1 projection of the pair** `[f_i, f_j]`, forced to use the identical
parent in both slots (`alpha=beta=1` in the Eckart-Young-Mirsky formulation),
not a sum-preserving reconstruction. Concretely: merging two *exactly*
identical neurons has zero-distortion optimum `f_p = f_i` (capacity
unchanged), not `f_p = 2*f_i`. An earlier draft of `test_merge.py` asserted
the sum-reconstruction property and failed at ~50% relative error — the
bug was in the test's oracle, not the implementation; see the git-less
history in this session if that ever needs re-confirming, or just re-derive
from eq. 6 above.

## Documented simplifications (paper defers to an appendix not available here)

1. **Sign resolution for `û` (`merge.py`, eq. 10-11).** The paper resolves the
   `±û` ambiguity by evaluating the *exact* (non-linearized) alignment
   objective at both signs. That requires `K(u, w_tilde_k_in)` for an
   arbitrary candidate direction `u`, which in general needs the full
   per-layer surrogate covariance `Sigma_hat_x` (section 4) — expensive, and
   whose closed-form reduction is in Appendix E.3 (not available). This
   implementation instead treats the candidate `u` as a linear combination
   `c1*w_tilde_i + c2*w_tilde_j` (the same reasoning the paper itself uses
   for eq. 17-18) to get an implied `(gamma_u, beta_u)`, then reuses the
   already-tested `cross_kernel()` to evaluate the same comparison. This is
   internally consistent with how the paper treats linear combinations of
   real neurons elsewhere, and is validated indirectly by
   `test_merge_beats_naive_alternatives_on_paper_objective` (the resulting
   parent beats naive strawmen on the paper's actual D^2 objective) rather
   than validated directly against Appendix E.3, which we don't have.

2. **Inactive regime `gamma_p^2 < eps` (`merge.py`, eq. 16-18 boundary case).**
   The paper explicitly defers the exact boundary-safe formulation and this
   edge case to Appendix B.5 / D. This implementation floors `gamma_p` and
   zeroes `sigma_p` instead of guessing the paper's exact formula. This path
   only triggers for a merge that's nearly degenerate (near-zero implied
   variance); the greedy loop should rarely if ever select such a merge
   since it would show high distortion for near-zero capacity, but the
   fallback exists so the implementation never NaNs if it does.

## Cross-layer weight propagation gap -- fixed

Originally found while building `scripts/toy_reproduction.py`: compressing
two adjacent layers (conv1 and conv2) independently in the same run
silently broke, because nothing propagated one layer's neuron-set changes
into the other's stored weight. Fixed in `hope_core/layer_state.py` and
`hope_core/encoding_loop.py`; this section is the equation-level account of
why the fix works, referenced from docs/weight_compression.md.

**The insight that made this tractable**: a layer's `w_out` is not an
independent, cacheable quantity. Per `adapters.py`'s `extract_*_bn_bank`,
`bank_L.w_out` is constructed as `next_layer.weight.T` -- literally the
downstream layer's own raw incoming weight, transposed. Symmetrically,
`bank_{L+1}.w_raw`'s fan-in dimension *is* `bank_L`'s neuron count. These
two tensors are transposes of each other across the layer boundary by
construction, not by coincidence -- so "propagating a compression action
across the boundary" reduces to "keep re-deriving each side's boundary
tensor from whatever the other side currently is," not to designing some
bespoke transformation per action type.

**Why this works uniformly for prune AND merge, no special-casing needed**:
after layer L+1 is pruned or merged, `bank_{L+1}.w_raw` already correctly
encodes "how do my current neurons read from L's (unchanged) output space"
-- eq. 16-18's merge machinery produces a new row that's valid across L's
full output space by construction, the same way a pruned row is simply
absent. So `bank_L`'s new `w_out` is always just `bank_{L+1}.w_raw.T`,
regardless of which action produced the current `bank_{L+1}`.

**The two primitives** (`hope_core/neuron.py`):
- `NeuronBank.with_replaced_w_out(new_w_out)` -- for the "downstream
  changed" direction.
- `NeuronBank.with_replaced_w_raw(new_w_raw)` -- the mirror, for "upstream
  changed."

**The two `LayerState` methods** (`hope_core/layer_state.py`):
- `resync_downstream_consumer(downstream_w_raw)` -- call when the layer
  this one feeds INTO changes. Refreshes `w_out`, capacities, `e_active`,
  and (if `build_merge_cache`) the full pair-geometry cache, since merge
  candidates' optimal-parent computation depends on `w_out` via `v*`
  (section 7.1.2's `best_numer_vec`).
- `resync_upstream_provider(upstream_w_out)` -- the mirror, call when the
  layer this one reads FROM changes. Refreshes `w_raw` and (if
  `build_merge_cache`) the pair-geometry cache -- NOT capacities, since
  `self_kernel`/capacity depend only on `(gamma, beta)`, never on `w_raw`.
  Pair geometry still needs refreshing because `compute_optimal_parent`'s
  direction-finding uses `effective_input()`, which reads `w_raw`'s actual
  values via cosine similarity between this layer's own neurons -- a stale
  fan-in here silently corrupts this layer's OWN internal merge-candidate
  selection, not just the eventual rebuild shape. This was the harder half
  of the bug to find: a first attempt at this fix only implemented the
  downstream-consumer direction, and a test compressing two real adjacent
  layers jointly failed with a genuine shape-mismatch `ValueError` the
  moment the layer got pruned *before* its consumer did -- which is what
  revealed the upstream-provider direction was still missing, rather than
  it being anticipated in the original design.

**The orchestration layer** (`hope_core/encoding_loop.py`):
`ChainAwareEncodingLoop` wraps a `ProgressiveEncodingLoop` by composition
(not inheritance/modification -- the wrapped loop's own `step()`/`run()`
are used exactly as-is, unmodified from what `test_encoding_loop.py`
already verified). Given a `chain` dict (`{downstream_name: upstream_name}`),
it calls the appropriate resync method on the *other* layer after every
action, in whichever direction is relevant. Verified end-to-end
(`tests/test_encoding_loop.py::test_real_two_layer_joint_compression_end_to_end`)
against a real 3-module `nn.Sequential` chain: two adjacent Linear+BN
layers compressed *jointly* (interleaved actions on both, driven by the
same global greedy selection the paper specifies -- not the sequential,
one-boundary-at-a-time workaround `scripts/toy_reproduction.py` still
uses), rebuilt via `adapters.py`, and run through a real forward pass.

**Not yet done**: `scripts/toy_reproduction.py` and
`scripts/reproduce_paper_resnet50.py` haven't been updated to use
`ChainAwareEncodingLoop` -- they still compress one boundary at a time,
sequentially, which is correct but not the paper's single joint pass. Real,
scoped follow-up work, not attempted in this session.

## Formulas confirmed but not yet wired into code (next: block eviction, DEFT)

- eq. 19-20 (block eviction): `J_evict = sum_{l=1}^{2} N_active^(l) * (E_active^(l) / E_identity)`,
  `E_identity = sum_k sqrt(gamma_k^2 + beta_k^2)` over the ambient/skip-path BN parameters.
- eq. 24-29 (DEFT): percentile-threshold elasticity map, structural mask
  `M_{j,k} = 0 if E_in,k>0 and E_out,j=0 else 1`, gradient scaling
  `g_t = E_out ⊙ grad`.
- Complexity, confirmed directly (section 10, step 2): building the initial
  merge-candidate cache is `O(L*N^2)`; each subsequent step is an `O(N)`
  local update. Matches what was flagged as a scaling risk before any code
  was written.

## Empirically confirmed: the O(N^2) merge-cache cost is real at ResNet-50 scale

This was flagged as a risk before any code was written (see the very first
architecture discussion this project started from) and confirmed empirically
while building `scripts/reproduce_paper_resnet50.py`: building the full
pairwise merge-candidate cache (`LayerState`'s `O(N^2)` `compute_optimal_parent`
calls at construction, plus `O(N)` new-pair calls per subsequent action) did
not finish within 300s for a single ~128-channel ResNet-50 boundary. Running
prune-only (`build_merge_cache=False`) across ALL 32 compressible boundaries
in the full network (widths 64-512) completed in ~14s on CPU. The
reproduction script defaults to prune-only for exactly this reason
(`--merge-width-limit`, default 32 -- below every standard ResNet-50
bottleneck's internal width, so merge is off by default; raise it if you
have the time budget). The underlying fix -- approximate candidate
generation (LSH / clustering / top-k prefiltering instead of an exact
all-pairs scan) -- is real, identified follow-up work, not implemented here.

## A genuine (not a bug) approximation discovered during implementation

`LayerState`'s O(N) local update (`hope_core/layer_state.py`) caches each
candidate pair's optimal parent, capacity, and D^2. Those turn out to *not*
be fully layer-state-independent: the optimal parent's scale `s*` (eq. 12)
depends on `e_rem = E_a - cap_i - cap_j`, i.e. on the layer's current total
capacity. An action anywhere in the layer changes `E_a`, which means every
*other* cached pair's parent is technically defined against a now-slightly-
stale `e_rem` — not just its final `J_merge` (which is always recomputed
fresh from live state at query time), but the parent's exact identity.

Fully eliminating this would require recomputing all `O(N^2)` pairs on every
single action, which is exactly the cost the paper's `O(N)` local update is
designed to avoid — so some staleness between touch-points is an inherent
property of *any* correct `O(N)`-per-step implementation of this algorithm,
not a defect in this one. Measured at roughly 1-2% relative drift in
`J_merge` on a 10-neuron/6-action toy scenario in `test_layer_state.py`.
`LayerState.refresh_all()` is provided as an explicit, documented escape
hatch for callers who want exactness back at the cost of a full `O(N^2)`
refresh; the encoding loop doesn't call it automatically.

