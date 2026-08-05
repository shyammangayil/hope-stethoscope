"""Creates a real model.pt + layer.json in the current directory, so the CLI
examples in README.md are actually runnable rather than placeholder names.

This is what was missing before: cli.py inspect/compress need a model *you*
saved yourself (torch.save(model, path) -- the whole object, not a state
dict) plus a small JSON describing which layer to compress. There's nothing
Gemma-specific here or anywhere in the weight-compression CLI -- Gemma only
exists on the KV-cache side (examples/run_gemma.py), accessed by HuggingFace
model id, never a local .pt file. This script exists purely so you have
something real to point the weight-compression CLI at on the first try,
before pointing it at your own model.

IMPORTANT if you're adapting this for your own model: the model class must
live in its own importable module (like examples/example_model_def.py
does), not defined inline in whatever script calls torch.save(). See that
module's docstring for why -- it's an inherent property of how Python
pickles full objects, not specific to this repo, and it's exactly the bug
an earlier version of this script had.

Run: python3 examples/make_example_model.py
Then, from the same directory:
    python3 cli.py inspect model.pt --layer-config layer.json --target-density 0.5 --report-dir out
    python3 cli.py compress model.pt --layer-config layer.json --target-density 0.5 --out compressed.pt
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.example_model_def import build_example_model


def main():
    model = build_example_model(seed=0)

    torch.save(model, "model.pt")  # the whole object -- cli.py can't construct
                                     # your model class from a state_dict alone

    layer_config = {"kind": "linear", "linear_name": "fc1", "bn_name": "bn1", "next_linear_name": "fc2"}
    with open("layer.json", "w") as f:
        json.dump(layer_config, f, indent=2)

    print("Wrote model.pt and layer.json to the current directory.")
    print("Try:")
    print("  python3 cli.py inspect model.pt --layer-config layer.json --target-density 0.5 --report-dir out")
    print("  python3 cli.py compress model.pt --layer-config layer.json --target-density 0.5 --out compressed.pt")
    print()
    print("model.pt is a throwaway, untrained example model -- its compression numbers")
    print("aren't meaningful on their own, this is purely to have something real to run")
    print("the CLI against. Point --layer-config at your own model's real layer names")
    print("(any Linear+BatchNorm1d or Conv2d+BatchNorm2d pair, see cli.py's docstring")
    print("for the 'kind': 'conv' schema) once you're past the first try -- and see")
    print("examples/example_model_def.py's docstring for an important pickling gotcha")
    print("if your model class isn't in its own importable module.")


if __name__ == "__main__":
    main()
