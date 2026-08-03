# hope-stethoscope

A from-scratch implementation of **HOPE** (Hilbert Operator for Progressive
Encoding) — arXiv:[2607.21366](https://arxiv.org/abs/2607.21366), Hossein
Mobahi and Peter L. Bartlett, Google DeepMind / UC Berkeley, posted 23 Jul
2026 — a data-free-for-BatchNorm framework that models neurons as rank-1
Hilbert-Schmidt operators, unifying pruning and neuron-merging as low-rank
subspace projection, plus **DEFT** (Dispersed Elastic Fine-Tuning), the
paper's own capacity-informed fine-tuning method.

**Read this whole README before trusting any result from this repo.** It's
written to tell you exactly what's been verified, how, and what hasn't —
not to sell the library. The single most important fact about this codebase
is in the next section.

## What's actually been validated (read this first)

This was built and tested inside a sandboxed environment with no access to
ImageNet, CIFAR, SVHN, or any pretrained-weight host. That materially limits
what could be checked before handing this to you:

| Claim | Status |
|---|---|
| The code correctly implements the paper's formulas | **Yes, high confidence.** 78 automated tests, including a Monte Carlo cross-check of the self-kernel against its own mathematical definition (not just against the paper's text), an exact-forward-pass round-trip check for every PyTorch adapter, and an end-to-end numerical check that a computed merge parent actually beats naive parameter-averaging on the paper's own distortion objective. See "What's tested" below. |
| The paper's own published ResNet-50/ImageNet numbers reproduce | **Unverified.** This needs `scripts/reproduce_paper_resnet50.py` run against real ImageNet, which requires data and network access this build environment didn't have. The script is written and its *structural* correctness was verified against the real ResNet-50 architecture (see below) — but the actual accuracy numbers have never been produced. Run it yourself; treat that as the real gate before trusting this for anything beyond a toy. |
| HOPE-guided compression beats magnitude/random pruning | **Not demonstrated, and that's an honest finding, not a hedge.** `scripts/toy_reproduction.py` runs entirely in-sandbox on synthetic data and shows graceful degradation with HOPE roughly competitive with (not clearly superior to) baselines, noisy across seeds — see "Toy reproduction results" below for the actual numbers. This is real evidence the *implementation* is sane, not evidence the *method* wins. |

If you only read one more section, read **"Known limitations"** below —
one of them (cross-layer weight propagation) will silently produce a wrong
model if you don't know about it.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate   # or your preferred env tool
pip install torch numpy scipy pytest click tqdm matplotlib
# Only needed for scripts/reproduce_paper_resnet50.py:
pip install torchvision
```

Tested against Python 3.12 and PyTorch 2.13 (CPU). No GPU is required for
anything except a full local ResNet-50/ImageNet run, and even there the
compression step itself is CPU-bound linear algebra on weight tensors, not
GPU-bound — only training/inference benefits from a GPU.

Verify the install:

```bash
python3 -m pytest tests/ -v
```

You should see 78 passed. If anything fails, stop here and check your torch
version before going further — every number in this README assumes a clean
run of this suite.

## Quick start

### Python API

```python
import torch
from hope_core.adapters import extract_linear_bn_bank, apply_bank_to_linear_bn
from hope_core.layer_state import LayerState
from hope_core.encoding_loop import ProgressiveEncodingLoop
from hope_core.report import StethoscopeReport

# model is any nn.Sequential(Linear, BatchNorm1d, ReLU, Linear, ...)
bank = extract_linear_bn_bank(model[0], model[1], model[3])
initial_bank = bank.clone()

state = LayerState(bank=bank, delta_p_init=bank.w_raw.shape[1] + bank.w_out.shape[1] + 4)
loop = ProgressiveEncodingLoop(layers={"fc1": state})
loop.run(target_density=0.5)

report = StethoscopeReport.from_loop(loop, {"fc1": initial_bank})
print(report.to_dict())

new_linear, new_bn, new_next_weight = apply_bank_to_linear_bn(state.bank, linear_bias=None, next_bias=None)
```

### CLI

```bash
# 1. Save a full model object (not just a state_dict) so the CLI can load it directly:
python3 -c "
import torch
from my_model import MyModel
m = MyModel(); m.load_state_dict(torch.load('weights.pt')); torch.save(m, 'model.pt')
"

# 2. Describe the one compressible layer you're targeting:
cat > layer.json << 'EOF'
{"kind": "linear", "linear_name": "fc1", "bn_name": "bn1", "next_linear_name": "fc2"}
EOF

# 3. Inspect (no model saved, just a report):
python3 cli.py inspect model.pt --layer-config layer.json --target-density 0.5 --report-dir out/

# 4. Actually compress and save:
python3 cli.py compress model.pt --layer-config layer.json --target-density 0.5 \
    --out compressed.pt --mask-out core_mask.json

# 5. DEFT: compute a core/slack elasticity map (wiring it into a training loop is on you --
#    see hope_core/deft.py's structural_mask/scale_gradient_/register_deft_gradient_hooks):
python3 cli.py deft-setup model.pt --layer-config layer.json --percentile 70 --out elasticity.json
```

Full command reference and the `layer.json` schema (including the `"kind":
"conv"` case) are in `cli.py`'s module docstring.

### Toy reproduction (runs immediately, no data needed)

```bash
python3 scripts/toy_reproduction.py
```

Trains a small synthetic-data CNN (~2 minutes on CPU), compresses it with
this repo's actual HOPE implementation, and compares against random and
magnitude-pruning baselines across 3 random seeds. See "Toy reproduction
results" below for what a real run showed.

### Real paper reproduction (needs data you have to supply)

```bash
pip install torchvision
python3 scripts/reproduce_paper_resnet50.py --imagenet-val /path/to/imagenet/val
```

Requires a local ImageNet validation set in `torchvision.datasets.ImageFolder`
layout — get this via [image-net.org](https://image-net.org)'s own
registration process; nothing in this repo can download it for you. See
"What's actually been validated" above for exactly what this script has and
hasn't been checked against.

## Package layout

```
hope-stethoscope/
├── hope_core/
│   ├── kernels.py         # self-/cross-kernel closed forms (eq. 3-5)
│   ├── neuron.py          # NeuronBank: effective params + capacity (eq. 1, section 5)
│   ├── costs.py            # J_prune, J_merge distortion functionals (eq. 6)
│   ├── merge.py            # optimal parent-neuron computation (eq. 9-18) -- the hardest math
│   ├── adapters.py         # NeuronBank <-> real nn.Linear/Conv2d+BN, round-trip tested
│   ├── layer_state.py      # per-layer capacity/pair cache, O(N) local updates (section 10)
│   ├── encoding_loop.py    # ProgressiveEncodingLoop: greedy cross-layer selection + audit trail
│   ├── block_eviction.py   # residual-block identity collapse (eq. 19-20)
│   ├── deft.py              # elasticity map, structural mask, gradient scaling (eq. 24-29)
│   └── report.py            # StethoscopeReport: survival ranking, core/slack, baseline comparison
├── cli.py                   # hope inspect / compress / deft-setup
├── scripts/
│   ├── toy_reproduction.py            # synthetic data, runs immediately, see results below
│   └── reproduce_paper_resnet50.py    # the real Phase-0 gate -- needs your ImageNet + network access
├── tests/                    # 78 tests -- see "What's tested"
└── NOTES.md                  # paper-equation-to-code traceability; read this for any "why" question
```

## What's tested

78 tests across 9 files, run with `pytest tests/ -v`. Categories, roughly by
what they'd actually catch:

- **Against first principles, not just the paper's text**: `test_kernels.py`
  Monte Carlo-simulates `E[ReLU(Z)^2]` directly and checks the closed form
  against the simulation, independent of whether the paper's formula was
  transcribed correctly.
- **The paper's central theoretical claim**: `test_neuron.py`'s scale-invariance
  test directly verifies that capacity is unchanged under the raw-weight
  rescaling that fools magnitude-based pruning — this is HOPE's whole reason
  to exist over a magnitude baseline, and it's checked, not assumed.
- **End-to-end numerical agreement, not just formula plumbing**:
  `test_merge.py` checks that a computed merge parent actually approximates
  each child individually (the paper's real objective — a rank-1 projection
  of the pair, not sum reconstruction, a distinction that took an actual
  debugging session to get right; see NOTES.md) and beats naive strawmen
  (parameter averaging, reusing one child) on the paper's own distortion
  metric.
- **Real forward passes, not just shapes**: `test_adapters.py` round-trips
  actual `nn.Conv2d`/`nn.Linear`+BN modules and compares logits, which is
  what actually catches an axis-order bug in the Conv2d flatten/unflatten
  convention (shapes alone wouldn't).
- **Bookkeeping under mutation**: `test_layer_state.py` specifically chains
  prune and merge actions and rebuilds from scratch to check stable-id
  tracking didn't get confused with tensor positions after removals — this
  class of bug (index vs. identity) doesn't show up in single-action tests.
- **Real subprocesses**: `test_cli.py` uses Click's `CliRunner` to actually
  invoke the CLI against saved model files, for both the Conv2d and Linear
  code paths, including an error-path test.

Two real bugs were caught and fixed by this suite during development (not
hypothetically — this is what actually happened, recorded in more detail in
NOTES.md): a caching bug where `LayerState` stored the final assembled
`J_merge` instead of the layer-independent geometry behind it (pairs not
touched by the most recent action silently went stale), and a wrong test
oracle that assumed merging was sum-reconstruction when the paper's actual
objective is a rank-1 projection of the pair.

## Toy reproduction results

A real run of `scripts/toy_reproduction.py` (3 seeds, synthetic 8-class
image task, `TinyConvNet` with a 32-channel `conv1`), mean ± std:

```
Dense (uncompressed) accuracy: 0.8621 +/- 0.0116

 density |            HOPE |          random |       magnitude
--------------------------------------------------------------
   0.844 | 0.8963+/-0.0476 | 0.8892+/-0.0255 | 0.9012+/-0.0358
   0.688 | 0.8350+/-0.0119 | 0.8163+/-0.0411 | 0.8388+/-0.0383
   0.531 | 0.7296+/-0.0543 | 0.6438+/-0.1027 | 0.7221+/-0.0265
   0.375 | 0.5829+/-0.1070 | 0.5408+/-0.0313 | 0.5325+/-0.0390
   0.250 | 0.3858+/-0.0733 | 0.4175+/-0.0776 | 0.3913+/-0.0022
```

**Honest read:** graceful, non-catastrophic degradation as density drops —
that's a real code-correctness signal (a broken implementation is much more
likely to produce nonsense or a cliff-edge collapse than a smooth curve).
HOPE is roughly competitive with magnitude pruning (both usually ahead of
random in the middle of the range) and the ordering between all three
methods is noisy and flips at the most aggressive compression. **This is
not evidence HOPE beats the baselines** — one 32-channel layer on a
synthetic task, three seeds, is nowhere near enough signal for that claim,
and it isn't what this script is for. It's here to show the implementation
does something sane on data it's never seen, which it does.

A significant, unplanned part of getting this result required adding a
BatchNorm recalibration step after every rebuild: without it, ALL THREE
methods (HOPE, random, magnitude alike) collapsed to near-chance accuracy
after removing even a handful of channels, because the downstream layer's
BN running statistics were still calibrated for the pre-pruning channel
count. This is standard, expected behavior for structured pruning in
general — not a HOPE-specific issue — but it's worth knowing about if you
extend this script or write your own evaluation harness.

## Known limitations

Ordered roughly by how likely each is to actually bite you.

1. **Cross-layer weight propagation is one-directional.** Compressing layer
   L correctly updates layer L+1's *input*-facing weight (tested, used
   throughout `cli.py` and both reproduction scripts). It does **not**
   propagate layer L+1's own compression *backward* into layer L's
   *outgoing* weight. Compressing two adjacent layers independently in the
   same run will silently produce shape-mismatched or wrong weights for a
   merge (a merge has no simple "which original column survived" mapping
   the way a prune does). `scripts/toy_reproduction.py` and
   `scripts/reproduce_paper_resnet50.py` both work around this by
   compressing one boundary at a time, sequentially, re-extracting fresh
   from the current model state each time — which is correct, but is not
   the paper's single joint cross-layer greedy pass. A real fix needs an
   explicit model-graph abstraction; see NOTES.md.
2. **The O(N²) merge-candidate cache is a real bottleneck above ~100-200
   neurons per layer**, confirmed empirically (not just theoretically) while
   building the ResNet-50 script: a single ~128-channel boundary with merge
   enabled didn't finish in 300s, while prune-only across the *entire*
   network (32 boundaries, widths 64-512) took ~14s. `LayerState`'s
   `build_merge_cache=False` option and the reproduction script's
   `--merge-width-limit` are the current mitigation, not a fix. The actual
   fix — approximate candidate generation instead of an exact all-pairs
   scan — is identified, real follow-up work.
3. **PH-1 activation assumption.** The exact self-/cross-kernel formulas are
   derived for ReLU-family (positively-homogeneous-degree-1) activations.
   GELU/SiLU/SwiGLU — what most current transformer MLPs actually use —
   fall outside that exact guarantee. Not relevant to this repo's current
   scope (BN + ReLU-family, matching the paper's own validated envelope),
   but relevant if you're tempted to point this at a transformer.
4. **Two documented, narrow simplifications in `merge.py`** where the paper
   defers to an appendix this implementation doesn't have access to (sign
   resolution for the optimal merge direction, and the `gamma_p^2 < eps`
   boundary case). Both are flagged inline in the code, validated indirectly
   (the merge optimality tests pass), and detailed in NOTES.md.
5. **DEFT's "Dynamic Resolution of Redundancy"** (merging redundant core
   neurons to free new plastic capacity, paper Figure 4a) is not
   implemented — the core elasticity/mask/gradient-scaling pipeline (eq.
   24-29) is.

## Citation

```
@article{mobahi2026hope,
  title={Hilbert Operator for Progressive Encoding (HOPE): A Mathematical Framework for Deconstructing Learned Representations in Deep Networks},
  author={Mobahi, Hossein and Bartlett, Peter L.},
  journal={arXiv preprint arXiv:2607.21366},
  year={2026}
}
```

This implementation is unofficial and has no affiliation with the paper's
authors. Report implementation issues against this repo, not against them.
