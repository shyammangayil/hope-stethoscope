"""hope-stethoscope CLI.

SCOPE (read this before filing a bug about an unsupported model): this CLI
operates on ONE explicitly-named compressible layer at a time (a Conv2d+BN
or Linear+BN, matching hope_core/adapters.py's two tested cases), not an
automatically-traced arbitrary architecture. Automatic multi-layer graph
detection for arbitrary PyTorch/HuggingFace models is real, useful, future
work -- deliberately not attempted here. hope_core's cross-layer weight
propagation gap (once a real risk for auto-detection specifically) is now
fixed for an explicitly-specified chain of layers (see
hope_core/layer_state.py's resync_downstream_consumer/resync_upstream_provider
and NOTES.md), but automatic graph tracing across an ARBITRARY model's full
layer topology is separate, larger, unattempted scope on its own -- "just
auto-detect every layer and compress them all together" still needs that
tracing infrastructure built from scratch, propagation fix or not.

Models must be saved as a full object (`torch.save(model, path)`, not
`torch.save(model.state_dict(), path)`), since this CLI has no way to
construct an arbitrary model class from a state dict alone. Beyond that,
your model class must live in its own importable module (e.g.
`my_project/models.py`), not be defined inline in whatever script calls
`torch.save()` -- torch.save pickles a *reference* to the class (its
module + name), not the class body, so a class defined in a directly-run
script gets recorded as belonging to `__main__`, which won't resolve when
a *different* script (this CLI) tries to load it later. This is an
inherent property of Python pickling, not something this CLI can paper
over. Run `python3 examples/make_example_model.py` for a working example
of the correct pattern (and see `examples/example_model_def.py`'s
docstring) before troubleshooting your own model against it -- this exact
mistake was made and caught once already while building this CLI.

Commands:
    hope inspect  MODEL.pt --layer-config layer.json --target-density 0.5 --report-dir out/
    hope compress MODEL.pt --layer-config layer.json --target-density 0.5 --out compressed.pt --mask-out mask.json
    hope deft-setup MODEL.pt --layer-config layer.json --percentile 70 --out elasticity.json

layer.json (Conv2d+BN example):
    {
      "kind": "conv",
      "conv_name": "conv1", "bn_name": "bn1", "next_conv_name": "conv2",
      "kernel_size": [3, 3], "next_kernel_size": [3, 3],
      "next_out_channels": 32, "padding": 1
    }
layer.json (Linear+BN example):
    {
      "kind": "linear",
      "linear_name": "fc1", "bn_name": "bn1", "next_linear_name": "fc2"
    }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hope_core.adapters import (
    apply_bank_to_conv_bn,
    apply_bank_to_linear_bn,
    extract_conv_bn_bank,
    extract_linear_bn_bank,
)
from hope_core.deft import compute_lock_threshold, elasticity_map, collect_prune_cost_distribution
from hope_core.encoding_loop import ProgressiveEncodingLoop
from hope_core.layer_state import LayerState
from hope_core.neuron import NeuronBank
from hope_core.report import StethoscopeReport

# KV-cache subsystem imports -- deliberately kept separate from the
# hope_core imports above rather than interleaved, matching this whole
# repo's "shared infrastructure, not shared algorithms" principle (see
# README.md): these two import blocks correspond to genuinely different
# subsystems that happen to share one CLI entry point, not one shared
# module graph.
from core.kv_modifiers import MODIFIER_REGISTRY, parse_modifier_spec
from compression.plugins import PLUGIN_REGISTRY, get_plugin
from harness.evaluator import Evaluator


def _get_module(model: torch.nn.Module, name: str) -> torch.nn.Module:
    module = dict(model.named_modules()).get(name)
    if module is None:
        raise click.ClickException(f"no module named '{name}' in this model (check layer-config names)")
    return module


def _load_layer_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _extract_bank(model: torch.nn.Module, config: dict) -> NeuronBank:
    if config["kind"] == "conv":
        conv = _get_module(model, config["conv_name"])
        bn = _get_module(model, config["bn_name"])
        next_conv = _get_module(model, config["next_conv_name"])
        return extract_conv_bn_bank(conv, bn, next_conv, name=config["conv_name"])
    elif config["kind"] == "linear":
        linear = _get_module(model, config["linear_name"])
        bn = _get_module(model, config["bn_name"])
        next_linear = _get_module(model, config["next_linear_name"])
        return extract_linear_bn_bank(linear, bn, next_linear, name=config["linear_name"])
    raise click.ClickException(f"unknown layer kind '{config['kind']}' (expected 'conv' or 'linear')")


def _write_back_bank(model: torch.nn.Module, config: dict, bank: NeuronBank) -> torch.nn.Module:
    """Returns a NEW model with the compressed layer + correctly re-sliced
    next-layer input swapped in. The next layer's other attributes (bias,
    stride, etc. beyond the input-facing weight) are preserved from the
    original model -- only the input-facing weight tensor changes size."""
    import copy

    new_model = copy.deepcopy(model)
    if config["kind"] == "conv":
        new_conv, new_bn, new_next_weight = apply_bank_to_conv_bn(
            bank,
            conv_kernel_size=tuple(config["kernel_size"]),
            next_conv_kernel_size=tuple(config["next_kernel_size"]),
            next_conv_out_channels=config["next_out_channels"],
            conv_padding=config.get("padding", 0),
        )
        setattr(new_model, config["conv_name"], new_conv)
        setattr(new_model, config["bn_name"], new_bn)
        next_conv_ref = _get_module(new_model, config["next_conv_name"])
        with torch.no_grad():
            next_conv_ref.weight = torch.nn.Parameter(new_next_weight.to(next_conv_ref.weight.dtype))
        next_conv_ref.in_channels = bank.n_neurons
    else:
        new_linear, new_bn, new_next_weight = apply_bank_to_linear_bn(
            bank,
            linear_bias=getattr(_get_module(model, config["linear_name"]), "bias", None),
            next_bias=None,
        )
        setattr(new_model, config["linear_name"], new_linear)
        setattr(new_model, config["bn_name"], new_bn)
        next_linear_ref = _get_module(new_model, config["next_linear_name"])
        with torch.no_grad():
            next_linear_ref.weight = torch.nn.Parameter(new_next_weight.to(next_linear_ref.weight.dtype))
        next_linear_ref.in_features = bank.n_neurons
    return new_model


def _run_compression(model: torch.nn.Module, config: dict, target_density: float) -> tuple[ProgressiveEncodingLoop, NeuronBank, dict]:
    original_bank = _extract_bank(model, config)
    delta_p = original_bank.w_raw.shape[1] + original_bank.w_out.shape[1] + 4
    layer_name = config.get("conv_name") or config.get("linear_name")
    layer_state = LayerState(bank=original_bank.clone(), delta_p_init=delta_p, name=layer_name)
    loop = ProgressiveEncodingLoop(layers={layer_name: layer_state})
    loop.run(target_density=target_density)
    initial_banks = {layer_name: original_bank}
    return loop, layer_state.bank, initial_banks


@click.group()
def cli():
    """hope-stethoscope: HOPE-based neural network compression (arXiv:2607.21366)."""


@cli.command()
@click.argument("model_path")
@click.option("--layer-config", required=True, help="Path to layer.json describing the compressible layer.")
@click.option("--target-density", type=float, required=True, help="Target parameter density in (0, 1].")
@click.option("--report-dir", default="hope_report", help="Directory to write report.json into.")
def inspect(model_path, layer_config, target_density, report_dir):
    """Run HOPE's encoding loop and emit a StethoscopeReport WITHOUT saving a compressed model."""
    model = torch.load(model_path, weights_only=False)
    model.eval()
    config = _load_layer_config(layer_config)

    loop, _final_bank, initial_banks = _run_compression(model, config, target_density)
    report = StethoscopeReport.from_loop(loop, initial_banks)

    Path(report_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(report_dir) / "report.json"
    report.save_json(str(report_path))

    click.echo(f"Final density: {report.final_density:.4f}")
    click.echo(f"Actions: {report.action_counts()}")
    click.echo(f"Report written to {report_path}")


@cli.command()
@click.argument("model_path")
@click.option("--layer-config", required=True)
@click.option("--target-density", type=float, required=True)
@click.option("--out", required=True, help="Output path for the compressed model.")
@click.option("--mask-out", default=None, help="Optional output path for a portable core mask (JSON).")
def compress(model_path, layer_config, target_density, out, mask_out):
    """Compress a model's specified layer to a target density and save it."""
    model = torch.load(model_path, weights_only=False)
    model.eval()
    config = _load_layer_config(layer_config)

    loop, final_bank, initial_banks = _run_compression(model, config, target_density)
    new_model = _write_back_bank(model, config, final_bank)
    torch.save(new_model, out)
    click.echo(f"Compressed model ({loop.current_density:.4f} density) written to {out}")

    if mask_out:
        layer_name = config.get("conv_name") or config.get("linear_name")
        surviving_original_ids = [i for i in range(initial_banks[layer_name].n_neurons)
                                   if i not in {x for r in loop.history for x in r.involved_ids if r.action == "prune"}]
        with open(mask_out, "w") as f:
            json.dump({"layer": layer_name, "original_width": initial_banks[layer_name].n_neurons,
                       "final_width": final_bank.n_neurons, "surviving_original_positions": surviving_original_ids}, f, indent=2)
        click.echo(f"Core mask written to {mask_out}")


@cli.command(name="deft-setup")
@click.argument("model_path")
@click.option("--layer-config", required=True)
@click.option("--percentile", type=float, required=True, help="Freeze/plasticity percentile threshold P in [0, 100].")
@click.option("--out", required=True, help="Output path for the elasticity map (JSON).")
def deft_setup(model_path, layer_config, percentile, out):
    """Compute a DEFT elasticity map (core/slack partition) for a layer, WITHOUT running any training.

    This only produces the partition; wiring structural_mask()/scale_gradient_()
    into an actual training loop is left to the caller (see README.md's DEFT
    section for a worked example), since this CLI has no opinion about your
    dataset, optimizer, or training loop.
    """
    model = torch.load(model_path, weights_only=False)
    model.eval()
    config = _load_layer_config(layer_config)
    bank = _extract_bank(model, config)

    costs = collect_prune_cost_distribution(bank)
    j_lock = compute_lock_threshold(costs, percentile)
    elasticity = elasticity_map(costs, j_lock, bank.n_neurons)

    result = {
        "layer": config.get("conv_name") or config.get("linear_name"),
        "percentile": percentile,
        "j_lock": j_lock,
        "elasticity": elasticity.tolist(),
        "core_count": int((elasticity == 0).sum()),
        "slack_count": int((elasticity == 1).sum()),
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    click.echo(f"Elasticity map ({result['core_count']} core / {result['slack_count']} slack) written to {out}")


@cli.group()
def kv():
    """KV-cache introspection and compression (core/, compression/, harness/).

    Deliberately a separate command group, not folded into the top-level
    commands above -- see README.md's "shared infrastructure, not shared
    algorithms" principle. parse-modifier/benchmark/demo are fully
    reimplemented here against tested library code; gemma is a thin,
    verified byte-identical passthrough to `examples/run_gemma.py` rather
    than a reimplementation -- see that command's own docstring for why.
    """


@kv.command(name="parse-modifier")
@click.argument("spec")
def kv_parse_modifier(spec):
    """Validate and echo a modifier spec, e.g. 'zero:token_positions=0,4'.

    Useful for checking syntax before committing to a slow real-model run --
    this only imports core.kv_modifiers (no torch/transformers needed).
    """
    try:
        modifier = parse_modifier_spec(spec)
    except ValueError as e:
        raise click.ClickException(str(e))
    click.echo(json.dumps(modifier.describe(), indent=2))


@kv.command(name="benchmark")
@click.option("--npy", default=None, help="Path to a saved KV block (.npy, shape [heads, seq, head_dim] "
                                            "or [batch, heads, seq, head_dim]). Omit to use synthetic data.")
@click.option("--plugins", default=None, help="Comma-separated plugin names to run (default: all). "
                                                f"Available: {', '.join(PLUGIN_REGISTRY)}")
@click.option("--seed", type=int, default=0, help="Seed for synthetic data (ignored if --npy is given).")
@click.option("--out", default=None, help="Optional path to write the results as JSON.")
def kv_benchmark(npy, plugins, seed, out):
    """Run the compression-plugin benchmark table against a KV block.

    Pure numpy end to end (compression/plugins.py + harness/evaluator.py) --
    no torch, no network, works the same whether the block came from a real
    model (saved separately via `np.save`) or synthetic data.
    """
    import numpy as np

    if npy:
        kv_block = np.load(npy)
    else:
        rng = np.random.default_rng(seed)
        kv_block = rng.normal(scale=20.0, size=(4, 10, 256)).astype(np.float32)  # matches real Gemma's shape, see op.txt
        click.echo(f"No --npy given, using synthetic KV block shape={kv_block.shape} (seed={seed})")

    plugin_names = [p.strip() for p in plugins.split(",")] if plugins else list(PLUGIN_REGISTRY)
    unknown = set(plugin_names) - set(PLUGIN_REGISTRY)
    if unknown:
        raise click.ClickException(f"Unknown plugin(s) {sorted(unknown)}. Available: {list(PLUGIN_REGISTRY)}")

    evaluator = Evaluator()
    results = []
    click.echo(f"\n{'plugin':>16} | {'ratio':>7} | {'latency_ms':>10} | {'max_err':>10} | status")
    click.echo("-" * 66)
    for name in plugin_names:
        report = evaluator.benchmark_plugin(get_plugin(name), kv_block)
        status = "PASS" if report.passed else "below threshold"
        err_str = f"{report.reconstruction_max_abs_err:.5f}" if report.reconstruction_max_abs_err is not None else "N/A"
        click.echo(f"{name:>16} | {report.compression_ratio:>6.2f}x | {report.latency_ms:>10.4f} | {err_str:>10} | {status}")
        results.append({
            "plugin": name, "compression_ratio": report.compression_ratio,
            "latency_ms": report.latency_ms, "max_abs_err": report.reconstruction_max_abs_err,
            "passed": report.passed,
        })

    if out:
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        click.echo(f"\nResults written to {out}")


@kv.command(name="gemma", context_settings={"ignore_unknown_options": True}, add_help_option=False)
@click.argument("gemma_args", nargs=-1, type=click.UNPROCESSED)
def kv_gemma(gemma_args):
    """Real-model KV inspection/modification against a HuggingFace model
    (needs torch+transformers+network access). Thin passthrough to
    examples/run_gemma.py for discoverability -- deliberately NOT a
    reimplementation. That script's ~400 lines of tightly-sequential logic
    (model loading, cache setup, generation, --compare-baseline,
    --cross-check) are confirmed working against real google/gemma-3-4b-it
    (see CHANGELOG.md), and refactoring them into something wrappable here
    would mean changing tested code without being able to re-verify the one
    part that actually matters -- the network-dependent path this CLI has
    no access to. See tests/test_real_gemma3_offline.py for what IS
    independently verified about the underlying mechanics (CacheTap,
    modifiers, generate() integration) without touching this script.

    All arguments after `gemma` are forwarded as-is, e.g.:
        hope kv gemma --model google/gemma-3-4b-it --prompt "..." --modifier "zero:token_positions=0"
    is exactly:
        python -m examples.run_gemma --model google/gemma-3-4b-it --prompt "..." --modifier "zero:token_positions=0"
    """
    import subprocess

    repo_root = Path(__file__).resolve().parent
    result = subprocess.run(
        [sys.executable, "-m", "examples.run_gemma", *gemma_args],
        cwd=repo_root,
    )
    sys.exit(result.returncode)


@kv.command(name="demo")
def kv_demo():
    """Run the full KV-cache pipeline (hooks -> inspector -> compression ->
    harness -> storage) against a synthetic model. No network, no torch
    required. Thin wrapper around examples/run_demo.py's own main() --
    reuses it directly rather than reimplementing the pipeline here."""
    from examples.run_demo import main as run_demo_main

    run_demo_main()


if __name__ == "__main__":
    cli()
