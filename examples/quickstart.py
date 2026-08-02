"""Minimal runnable example of the hope_core Python API on a tiny model you
already have in memory -- no CLI, no saved checkpoints, no dataset.

Run: python3 examples/quickstart.py
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hope_core.adapters import apply_bank_to_linear_bn, extract_linear_bn_bank
from hope_core.encoding_loop import ProgressiveEncodingLoop
from hope_core.layer_state import LayerState
from hope_core.report import StethoscopeReport

torch.manual_seed(0)

# Any Linear -> BatchNorm1d -> ReLU -> Linear chain works. Real BN running
# stats matter (a freshly-constructed BN's default mean=0/var=1 is a
# degenerate case) -- normally these come from training; here we fake
# plausible ones directly since this example isn't about training a model.
model = nn.Sequential(nn.Linear(20, 16, bias=False), nn.BatchNorm1d(16), nn.ReLU(), nn.Linear(16, 5, bias=False))
model.eval()
with torch.no_grad():
    model[1].running_mean.copy_(torch.randn(16) * 0.2)
    model[1].running_var.copy_(torch.rand(16) * 0.6 + 0.3)
    model[1].weight.copy_(torch.randn(16).abs() + 0.2)
    model[1].bias.copy_(torch.randn(16) * 0.3)

print("1. Extract a NeuronBank from the hidden Linear+BN layer")
bank = extract_linear_bn_bank(model[0], model[1], model[3], name="hidden")
initial_bank = bank.clone()
print(f"   {bank.n_neurons} neurons, capacities: {bank.capacity().tolist()}")

print("\n2. Run the progressive encoding loop to 50% parameter density")
delta_p = bank.w_raw.shape[1] + bank.w_out.shape[1] + 4  # rough per-neuron param footprint
state = LayerState(bank=bank, delta_p_init=delta_p, name="hidden")
loop = ProgressiveEncodingLoop(layers={"hidden": state})
loop.run(target_density=0.5)
print(f"   {state.n_active()} neurons survive, density={loop.current_density:.3f}")
for record in loop.history:
    print(f"   step {record.step}: {record.action} {record.involved_ids} (distortion_rate={record.distortion_rate:.4f})")

print("\n3. Generate a StethoscopeReport")
report = StethoscopeReport.from_loop(loop, {"hidden": initial_bank})
core, slack = report.core_slack_partition()
print(f"   {len(core)} core survivors, {len(slack)} slack survivors")
print(f"   magnitude-baseline agreement: {report.compare_to_magnitude_baseline()}")

print("\n4. Rebuild a smaller, real nn.Linear/BatchNorm1d from the compressed bank")
new_hidden, new_bn, new_out_weight = apply_bank_to_linear_bn(state.bank, linear_bias=None, next_bias=None)
print(f"   new hidden layer: {new_hidden.in_features} -> {new_hidden.out_features}")

print("\n5. Confirm the rebuilt model actually runs")
compressed_model = nn.Sequential(new_hidden, new_bn, nn.ReLU())
compressed_model.eval()
x = torch.randn(4, 20)
with torch.no_grad():
    hidden_out = compressed_model(x)
    final_out = hidden_out @ new_out_weight.T
print(f"   output shape: {tuple(final_out.shape)} (expected (4, 5))")
assert final_out.shape == (4, 5)
assert torch.isfinite(final_out).all()
print("\nDone -- see README.md for the CLI equivalent of this same workflow.")
