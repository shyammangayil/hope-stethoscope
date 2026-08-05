"""The example model class, kept in its own module on purpose.

torch.save(model, path) pickles a REFERENCE to the class (its __module__ +
__qualname__), not the class body itself. If ExampleModel were defined
inline in make_example_model.py, pickle would record it as
'__main__.ExampleModel' when that script runs (since the script IS
__main__ when executed directly) -- but cli.py is a DIFFERENT __main__
when IT runs, with no ExampleModel in it, so torch.load() would fail with
`AttributeError: Can't get attribute 'ExampleModel' on <module '__main__'>`.
This was caught by actually running the save step and the CLI's load step
as two separate process invocations, not by any test in this repo (every
existing test saves and loads within the same process/module, which
doesn't hit this).

The general lesson for anyone pointing this CLI at their OWN model: your
model class needs to live in a module that's importable by name from
wherever you invoke cli.py (not defined inline in a `__main__` script) --
this is an inherent property of Python pickling full objects, not
something specific to this repo. See README.md.
"""

import torch
from torch import nn


class ExampleModel(nn.Module):
    def __init__(self, d_in=20, hidden=16, d_out=5):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, d_out, bias=False)

    def forward(self, x):
        return self.fc2(self.relu(self.bn1(self.fc1(x))))


def build_example_model(seed: int = 0) -> "ExampleModel":
    """Realistic-looking (not trained, not meaningful) BN running stats --
    a freshly-constructed BatchNorm1d's degenerate mean=0/var=1 defaults
    would make for a boring/misleading first look at what capacity/pruning
    output looks like."""
    torch.manual_seed(seed)
    model = ExampleModel()
    with torch.no_grad():
        model.bn1.running_mean.copy_(torch.randn(16) * 0.2)
        model.bn1.running_var.copy_(torch.rand(16) * 0.6 + 0.3)
        model.bn1.weight.copy_(torch.randn(16).abs() + 0.2)
        model.bn1.bias.copy_(torch.randn(16) * 0.3)
    model.eval()
    return model
