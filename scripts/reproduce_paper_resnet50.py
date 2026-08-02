"""Phase 0: reproduce the paper's own ResNet-50/ImageNet compression result.

THIS IS THE SCRIPT THAT ACTUALLY VALIDATES THE SCIENCE. Everything else in
this repo (65+ unit/integration tests, the toy reproduction) validates that
the CODE correctly implements the formulas as written. This script is the
only thing that checks whether the resulting numbers look like what the
paper reports on real data at real scale -- which is the entire point of
"Phase 0" from the original build plan: run this FIRST, before trusting
anything built on top of it, and compare the printed density-vs-accuracy
curve against the paper's own Figure/Table for ResNet-50 pruning.

CANNOT BE RUN IN THE BUILD SANDBOX THIS REPO WAS ASSEMBLED IN: it needs
(a) network access to download a pretrained ResNet-50 checkpoint, and
(b) a local copy of the ImageNet validation set, which requires registration
and isn't something any sandbox auto-downloads. This script's code has been
reviewed carefully but NOT executed end-to-end -- run it yourself and treat
the first run as the real correctness gate, per the original build plan's
Phase 0 -> Phase 1 sequencing. If the density-vs-accuracy curve doesn't
roughly track the paper's own numbers, stop and debug here before trusting
anything else in this repo built on top of hope_core.

SCOPE: compresses each bottleneck block's two INTERNAL conv layers
(conv1->conv2 and conv2->conv3), matching the paper's own l in {1,2}
convention for what granular compression can reach (section 8.1) -- NOT the
block's final expansion conv (conv3) or any downsample path, both of which
are constrained by the residual-addition shape requirement. Layers are
compressed sequentially, one boundary at a time (extract fresh -> compress
-> write back -> move on), not as one joint cross-layer greedy pass -- see
NOTES.md and cli.py's docstring for why (hope_core's cross-layer weight
propagation is currently one-directional; joint multi-layer optimization is
identified future work, not silently assumed to work here).

Usage:
    pip install torchvision
    # Point --imagenet-val at a local ImageNet validation directory in the
    # standard torchvision.datasets.ImageFolder layout (one subdirectory per
    # class). Get this by following https://image-net.org 's own
    # registration/download process -- not something this script can do for you.
    python3 scripts/reproduce_paper_resnet50.py --imagenet-val /path/to/imagenet/val
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hope_core.adapters import apply_bank_to_conv_bn, extract_conv_bn_bank
from hope_core.encoding_loop import ProgressiveEncodingLoop
from hope_core.layer_state import LayerState
from hope_core.neuron import NeuronBank


def load_pretrained_resnet50() -> nn.Module:
    import torchvision

    try:
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    except AttributeError:  # older torchvision API
        return torchvision.models.resnet50(pretrained=True).eval()
    model = torchvision.models.resnet50(weights=weights).eval()
    return model


def get_imagenet_val_loader(val_dir: str, batch_size: int = 128):
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision import transforms

    # Standard ImageNet eval preprocessing, matching torchvision's own
    # ResNet50_Weights.IMAGENET1K_V2 recommended transform.
    preprocess = transforms.Compose([
        transforms.Resize(232),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.ImageFolder(val_dir, transform=preprocess)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)


@torch.no_grad()
def evaluate_top1(model: nn.Module, loader, max_batches: int | None = None, device: str = "cpu") -> float:
    model.eval().to(device)
    correct, total = 0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=-1)
        correct += int((preds == y).sum())
        total += y.shape[0]
    return correct / max(total, 1)


@torch.no_grad()
def recalibrate_bn(model: nn.Module, loader, n_batches: int = 20, device: str = "cpu") -> None:
    """Standard post-structural-pruning BN recalibration -- see the extensive
    discussion in scripts/toy_reproduction.py for why this is necessary for
    ANY structured pruning method, not specific to HOPE. Uses batches from
    the same loader (val split) purely for statistics, not for any accuracy
    computation -- swap in a held-out calibration/train split if you have one
    available; using val data for BN stats only (not for selecting which
    neurons to keep) is a common, defensible simplification when a separate
    calibration split isn't at hand, but a train-split calibration set is
    the more rigorous choice if you have one.
    """
    original_momenta = {}
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            original_momenta[m] = m.momentum
            m.momentum = None  # cumulative moving average across all calibration batches
            m.reset_running_stats()
    model.train().to(device)
    for i, (x, _y) in enumerate(loader):
        if i >= n_batches:
            break
        model(x.to(device))
    model.eval()
    for m, momentum in original_momenta.items():
        m.momentum = momentum


def _compress_conv_boundary(
    model: nn.Module, conv_name: str, bn_name: str, next_conv_name: str, target_density: float,
    merge_width_limit: int = 32,
) -> nn.Module:
    """Extract -> compress -> write back ONE conv+bn -> next_conv boundary,
    in place on a deep-copied model. Sequential/single-boundary by design --
    see module docstring's SCOPE note.

    merge_width_limit: layers wider than this use prune-only (LayerState's
    build_merge_cache=False). This is a real, deliberate trade-off, not an
    oversight: building the merge candidate cache is O(N^2)
    compute_optimal_parent calls (NOTES.md flagged this exact scaling risk
    before any code was written, and it was confirmed empirically while
    testing this script -- ResNet-50's layer3/layer4 boundaries are 256-512
    channels wide, where the full prune+merge cache took long enough to be
    impractical for a sandboxed test run). Pruning alone is O(N) per layer
    and doesn't hit this wall. If you have the time budget (this can mean
    hours, not minutes, at these widths) or want full prune+merge fidelity
    throughout, pass a higher merge_width_limit or float('inf').
    """

    def get(m, name):
        return dict(m.named_modules())[name]

    new_model = copy.deepcopy(model)
    conv, bn, next_conv = get(new_model, conv_name), get(new_model, bn_name), get(new_model, next_conv_name)

    bank = extract_conv_bn_bank(conv, bn, next_conv, name=f"{conv_name}")
    delta_p = bank.w_raw.shape[1] + bank.w_out.shape[1] + 4
    use_merge = bank.n_neurons <= merge_width_limit
    state = LayerState(bank=bank, delta_p_init=delta_p, name=conv_name, build_merge_cache=use_merge)
    if not use_merge:
        print(f"    ({conv_name}: {bank.n_neurons} channels > merge_width_limit={merge_width_limit}, using prune-only)")
    loop = ProgressiveEncodingLoop(layers={conv_name: state})
    loop.run(target_density=target_density)

    new_conv, new_bn, new_next_weight = apply_bank_to_conv_bn(
        state.bank,
        conv_kernel_size=tuple(conv.kernel_size),
        next_conv_kernel_size=tuple(next_conv.kernel_size),
        next_conv_out_channels=next_conv.out_channels,
        conv_stride=conv.stride,
        conv_padding=conv.padding,
    )
    setattr(new_model.get_submodule(conv_name.rsplit(".", 1)[0]) if "." in conv_name else new_model, conv_name.rsplit(".", 1)[-1], new_conv)
    setattr(new_model.get_submodule(bn_name.rsplit(".", 1)[0]) if "." in bn_name else new_model, bn_name.rsplit(".", 1)[-1], new_bn)
    next_conv_parent = new_model.get_submodule(next_conv_name.rsplit(".", 1)[0]) if "." in next_conv_name else new_model
    next_conv_attr = next_conv_name.rsplit(".", 1)[-1]
    with torch.no_grad():
        next_conv.weight = torch.nn.Parameter(new_next_weight.to(next_conv.weight.dtype))
    next_conv.in_channels = state.bank.n_neurons
    setattr(next_conv_parent, next_conv_attr, next_conv)
    return new_model


def compress_resnet50(model: nn.Module, per_layer_target_density: float, merge_width_limit: int = 32) -> nn.Module:
    """Sequentially compress every bottleneck block's conv1->conv2 and
    conv2->conv3 boundaries across layer1..layer4. See module docstring's
    SCOPE note: sequential, not a joint cross-layer greedy pass. See
    _compress_conv_boundary's docstring for merge_width_limit."""
    current = model
    block_groups = ["layer1", "layer2", "layer3", "layer4"]
    for group_name in block_groups:
        group = getattr(current, group_name)
        for block_idx in range(len(group)):
            prefix = f"{group_name}.{block_idx}"
            print(f"  compressing {prefix}.conv1 -> {prefix}.conv2 ...")
            current = _compress_conv_boundary(
                current, f"{prefix}.conv1", f"{prefix}.bn1", f"{prefix}.conv2", per_layer_target_density,
                merge_width_limit=merge_width_limit,
            )
            print(f"  compressing {prefix}.conv2 -> {prefix}.conv3 ...")
            current = _compress_conv_boundary(
                current, f"{prefix}.conv2", f"{prefix}.bn2", f"{prefix}.conv3", per_layer_target_density,
                merge_width_limit=merge_width_limit,
            )
    return current


def magnitude_prune_conv_boundary(model: nn.Module, conv_name: str, bn_name: str, next_conv_name: str, target_density: float) -> nn.Module:
    """L1-norm baseline for the same conv boundary, for direct comparison
    against HOPE at a matched density (paper's own baseline family, section 11.1)."""

    def get(m, name):
        return dict(m.named_modules())[name]

    new_model = copy.deepcopy(model)
    conv, bn, next_conv = get(new_model, conv_name), get(new_model, bn_name), get(new_model, next_conv_name)
    bank = extract_conv_bn_bank(conv, bn, next_conv, name=conv_name)

    delta_p = bank.w_raw.shape[1] + bank.w_out.shape[1] + 4
    n_original = bank.n_neurons
    target_count = max(1, int(round(n_original * target_density)))

    magnitudes = bank.w_out.norm(dim=-1)
    keep = torch.zeros(n_original, dtype=torch.bool)
    keep[torch.topk(magnitudes, target_count, largest=True).indices] = True
    compressed_bank = bank._select(keep)

    new_conv, new_bn, new_next_weight = apply_bank_to_conv_bn(
        compressed_bank, conv_kernel_size=tuple(conv.kernel_size), next_conv_kernel_size=tuple(next_conv.kernel_size),
        next_conv_out_channels=next_conv.out_channels, conv_stride=conv.stride, conv_padding=conv.padding,
    )
    conv_parent = new_model.get_submodule(conv_name.rsplit(".", 1)[0]) if "." in conv_name else new_model
    setattr(conv_parent, conv_name.rsplit(".", 1)[-1], new_conv)
    bn_parent = new_model.get_submodule(bn_name.rsplit(".", 1)[0]) if "." in bn_name else new_model
    setattr(bn_parent, bn_name.rsplit(".", 1)[-1], new_bn)
    with torch.no_grad():
        next_conv.weight = torch.nn.Parameter(new_next_weight.to(next_conv.weight.dtype))
    next_conv.in_channels = compressed_bank.n_neurons
    return new_model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imagenet-val", required=True, help="Path to ImageNet validation set (ImageFolder layout)")
    parser.add_argument("--densities", type=float, nargs="+", default=[0.9, 0.7, 0.5, 0.3],
                         help="Per-layer target densities to sweep (paper compares a full curve)")
    parser.add_argument("--eval-batches", type=int, default=50, help="Batches per accuracy evaluation (use all for a full run)")
    parser.add_argument("--recalib-batches", type=int, default=20)
    parser.add_argument("--merge-width-limit", type=int, default=32,
                         help="Layers wider than this use prune-only. Default is conservative and based on "
                              "measured timings, not a guess: full prune+merge on a 128-channel boundary did "
                              "not finish in 300s in testing, while prune-only across ALL of ResNet-50's 32 "
                              "boundaries (widths 64-512) completed in ~14s on CPU. Every standard ResNet-50 "
                              "bottleneck's internal width is >=64, so the default effectively means "
                              "prune-only throughout -- raise this (e.g. to 64 or 128) to get merge on "
                              "narrower boundaries if you have the time budget; expect it to cost minutes "
                              "to hours depending on how high you set it and your hardware, not seconds. "
                              "This is the real, current cost of the O(N^2) merge-cache construction "
                              "(NOTES.md flagged this scaling risk before any code was written); a proper "
                              "fix is approximate candidate generation (LSH/clustering) instead of the exact "
                              "all-pairs scan, which is real follow-up work, not implemented here.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("Loading pretrained ResNet-50 (ImageNet weights)...")
    model = load_pretrained_resnet50()

    print(f"Loading ImageNet validation set from {args.imagenet_val} ...")
    loader = get_imagenet_val_loader(args.imagenet_val)

    print("\nEvaluating dense baseline...")
    t0 = time.time()
    dense_acc = evaluate_top1(model, loader, max_batches=args.eval_batches, device=args.device)
    print(f"Dense top-1 accuracy: {dense_acc:.4f} ({time.time() - t0:.1f}s)")

    results = []
    for density in args.densities:
        print(f"\n=== target per-layer density: {density} ===")

        t0 = time.time()
        hope_model = compress_resnet50(model, density, merge_width_limit=args.merge_width_limit)
        print(f"HOPE compression done in {time.time() - t0:.1f}s, recalibrating BN...")
        recalibrate_bn(hope_model, loader, n_batches=args.recalib_batches, device=args.device)
        hope_acc = evaluate_top1(hope_model, loader, max_batches=args.eval_batches, device=args.device)
        print(f"HOPE top-1 @ density={density}: {hope_acc:.4f}")

        # magnitude baseline over the same set of boundaries, for a direct comparison
        mag_model = model
        for group_name in ["layer1", "layer2", "layer3", "layer4"]:
            group = getattr(mag_model, group_name)
            for block_idx in range(len(group)):
                prefix = f"{group_name}.{block_idx}"
                mag_model = magnitude_prune_conv_boundary(mag_model, f"{prefix}.conv1", f"{prefix}.bn1", f"{prefix}.conv2", density)
                mag_model = magnitude_prune_conv_boundary(mag_model, f"{prefix}.conv2", f"{prefix}.bn2", f"{prefix}.conv3", density)
        recalibrate_bn(mag_model, loader, n_batches=args.recalib_batches, device=args.device)
        mag_acc = evaluate_top1(mag_model, loader, max_batches=args.eval_batches, device=args.device)
        print(f"Magnitude top-1 @ density={density}: {mag_acc:.4f}")

        results.append({"density": density, "hope_acc": hope_acc, "magnitude_acc": mag_acc})

    print("\n" + "=" * 60)
    print(f"{'density':>10} | {'HOPE':>8} | {'magnitude':>10}")
    print("-" * 36)
    print(f"{'1.000 (dense)':>10} | {dense_acc:>8.4f} | {'--':>10}")
    for r in results:
        print(f"{r['density']:>10.3f} | {r['hope_acc']:>8.4f} | {r['magnitude_acc']:>10.4f}")
    print("\nCompare this curve against the paper's own ResNet-50/ImageNet")
    print("figure (arXiv:2607.21366, section 11.1). If it doesn't roughly")
    print("track, stop here and debug hope_core before trusting anything")
    print("built on top of it -- this is Phase 0's entire purpose.")


if __name__ == "__main__":
    main()
