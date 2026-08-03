"""End-to-end tests for cli.py using Click's CliRunner -- these actually
invoke the CLI commands as subprocesses-in-process would, against real
saved model files, rather than testing the underlying functions in
isolation (which is already covered by tests/test_adapters.py etc.)."""

import json
import os
import tempfile

import torch
from click.testing import CliRunner
from torch import nn

from cli import cli

torch.manual_seed(0)


class _LinearBNModel(nn.Module):
    def __init__(self, d_in=10, n=12, d_out=5):
        super().__init__()
        self.fc1 = nn.Linear(d_in, n, bias=False)
        self.bn1 = nn.BatchNorm1d(n)
        self.fc2 = nn.Linear(n, d_out, bias=False)

    def forward(self, x):
        return self.fc2(torch.relu(self.bn1(self.fc1(x))))


class _ConvBNModel(nn.Module):
    def __init__(self, c_in=3, n=10, c_out=4):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, n, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n)
        self.conv2 = nn.Conv2d(n, c_out, 3, padding=1, bias=False)

    def forward(self, x):
        return self.conv2(torch.relu(self.bn1(self.conv1(x))))


def _init_bn_realistically(model: nn.Module) -> nn.Module:
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.running_mean.copy_(torch.randn_like(m.running_mean) * 0.2)
                m.running_var.copy_(torch.rand_like(m.running_var) + 0.3)
                m.weight.copy_(torch.randn_like(m.weight).abs() + 0.2)
                m.bias.copy_(torch.randn_like(m.bias) * 0.3)
    model.eval()
    return model


def test_inspect_command_on_linear_model():
    model = _init_bn_realistically(_LinearBNModel())
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.pt")
        config_path = os.path.join(tmp, "layer.json")
        report_dir = os.path.join(tmp, "report")
        torch.save(model, model_path)
        with open(config_path, "w") as f:
            json.dump({"kind": "linear", "linear_name": "fc1", "bn_name": "bn1", "next_linear_name": "fc2"}, f)

        result = runner.invoke(cli, ["inspect", model_path, "--layer-config", config_path,
                                      "--target-density", "0.5", "--report-dir", report_dir])
        assert result.exit_code == 0, result.output
        assert os.path.exists(os.path.join(report_dir, "report.json"))
        with open(os.path.join(report_dir, "report.json")) as f:
            report = json.load(f)
        assert report["final_density"] <= 0.5 + 1e-6


def test_compress_command_produces_a_loadable_smaller_model():
    model = _init_bn_realistically(_LinearBNModel(d_in=10, n=12, d_out=5))
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.pt")
        config_path = os.path.join(tmp, "layer.json")
        out_path = os.path.join(tmp, "compressed.pt")
        mask_path = os.path.join(tmp, "mask.json")
        torch.save(model, model_path)
        with open(config_path, "w") as f:
            json.dump({"kind": "linear", "linear_name": "fc1", "bn_name": "bn1", "next_linear_name": "fc2"}, f)

        result = runner.invoke(cli, ["compress", model_path, "--layer-config", config_path,
                                      "--target-density", "0.4", "--out", out_path, "--mask-out", mask_path])
        assert result.exit_code == 0, result.output
        assert os.path.exists(out_path)
        assert os.path.exists(mask_path)

        compressed = torch.load(out_path, weights_only=False)
        assert compressed.fc1.out_features < 12
        assert compressed.bn1.num_features == compressed.fc1.out_features
        assert compressed.fc2.in_features == compressed.fc1.out_features

        # the rebuilt model should actually run without shape errors
        x = torch.randn(3, 10)
        compressed.eval()
        out = compressed(x)
        assert out.shape == (3, 5)
        assert torch.isfinite(out).all()

        with open(mask_path) as f:
            mask = json.load(f)
        assert mask["final_width"] == compressed.fc1.out_features


def test_compress_command_on_conv_model():
    model = _init_bn_realistically(_ConvBNModel(c_in=3, n=10, c_out=4))
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.pt")
        config_path = os.path.join(tmp, "layer.json")
        out_path = os.path.join(tmp, "compressed.pt")
        torch.save(model, model_path)
        with open(config_path, "w") as f:
            json.dump({
                "kind": "conv", "conv_name": "conv1", "bn_name": "bn1", "next_conv_name": "conv2",
                "kernel_size": [3, 3], "next_kernel_size": [3, 3], "next_out_channels": 4, "padding": 1,
            }, f)

        result = runner.invoke(cli, ["compress", model_path, "--layer-config", config_path,
                                      "--target-density", "0.5", "--out", out_path])
        assert result.exit_code == 0, result.output

        compressed = torch.load(out_path, weights_only=False)
        assert compressed.conv1.out_channels < 10
        assert compressed.conv2.in_channels == compressed.conv1.out_channels

        x = torch.randn(2, 3, 8, 8)
        compressed.eval()
        out = compressed(x)
        assert out.shape == (2, 4, 8, 8)
        assert torch.isfinite(out).all()


def test_deft_setup_command():
    model = _init_bn_realistically(_LinearBNModel(d_in=10, n=12, d_out=5))
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.pt")
        config_path = os.path.join(tmp, "layer.json")
        out_path = os.path.join(tmp, "elasticity.json")
        torch.save(model, model_path)
        with open(config_path, "w") as f:
            json.dump({"kind": "linear", "linear_name": "fc1", "bn_name": "bn1", "next_linear_name": "fc2"}, f)

        result = runner.invoke(cli, ["deft-setup", model_path, "--layer-config", config_path,
                                      "--percentile", "50", "--out", out_path])
        assert result.exit_code == 0, result.output

        with open(out_path) as f:
            result_json = json.load(f)
        assert result_json["core_count"] + result_json["slack_count"] == 12
        assert len(result_json["elasticity"]) == 12
        assert set(result_json["elasticity"]) <= {0.0, 1.0}


def test_inspect_reports_missing_module_name_clearly():
    model = _init_bn_realistically(_LinearBNModel())
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.pt")
        config_path = os.path.join(tmp, "layer.json")
        torch.save(model, model_path)
        with open(config_path, "w") as f:
            json.dump({"kind": "linear", "linear_name": "does_not_exist", "bn_name": "bn1", "next_linear_name": "fc2"}, f)

        result = runner.invoke(cli, ["inspect", model_path, "--layer-config", config_path, "--target-density", "0.5"])
        assert result.exit_code != 0
        assert "does_not_exist" in result.output


# ---------------------------------------------------------------------------
# KV-cache subsystem commands (hope kv ...). Only the network-free paths
# (parse-modifier, benchmark, demo) are wrapped in the CLI so far -- see
# cli.py's `kv` group docstring for why the real-Gemma path isn't yet.
# ---------------------------------------------------------------------------


def test_kv_parse_modifier_valid_spec():
    runner = CliRunner()
    result = runner.invoke(cli, ["kv", "parse-modifier", "zero:token_positions=0,4"])
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    assert parsed["name"] == "zero"
    assert parsed["token_positions"] == [0, 4]


def test_kv_parse_modifier_invalid_name_fails_clearly():
    runner = CliRunner()
    result = runner.invoke(cli, ["kv", "parse-modifier", "not_a_real_modifier:foo=1"])
    assert result.exit_code != 0
    assert "Unknown modifier" in result.output


def test_kv_benchmark_synthetic_default():
    runner = CliRunner()
    result = runner.invoke(cli, ["kv", "benchmark"])
    assert result.exit_code == 0, result.output
    assert "synthetic KV block" in result.output
    for plugin_name in ["fp8", "int4", "low_rank", "sliding_window", "token_eviction", "head_pruning"]:
        assert plugin_name in result.output


def test_kv_benchmark_specific_plugins_and_json_output():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "results.json")
        result = runner.invoke(cli, ["kv", "benchmark", "--plugins", "fp8,int4", "--out", out_path])
        assert result.exit_code == 0, result.output
        assert "sliding_window" not in result.output  # only requested plugins should run
        with open(out_path) as f:
            results = json.load(f)
        assert {r["plugin"] for r in results} == {"fp8", "int4"}
        assert all(r["compression_ratio"] > 0 for r in results)


def test_kv_benchmark_unknown_plugin_fails_clearly():
    runner = CliRunner()
    result = runner.invoke(cli, ["kv", "benchmark", "--plugins", "not_a_real_plugin"])
    assert result.exit_code != 0
    assert "Unknown plugin" in result.output


def test_kv_benchmark_is_reproducible_given_same_seed():
    """Same seed -> same synthetic data -> same compression_ratio/max_err,
    so this command is actually useful for regression-checking a plugin
    change. latency_ms is real wall-clock timing and correctly excluded
    from this comparison -- it was never going to be reproducible run to
    run, and asserting full output equality here would be testing the
    system clock, not this command."""
    import json as _json
    import re

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        out1, out2 = os.path.join(tmp, "r1.json"), os.path.join(tmp, "r2.json")
        r1 = runner.invoke(cli, ["kv", "benchmark", "--plugins", "fp8", "--seed", "7", "--out", out1])
        r2 = runner.invoke(cli, ["kv", "benchmark", "--plugins", "fp8", "--seed", "7", "--out", out2])
        assert r1.exit_code == 0 and r2.exit_code == 0

        with open(out1) as f:
            results1 = _json.load(f)
        with open(out2) as f:
            results2 = _json.load(f)
        assert results1[0]["compression_ratio"] == results2[0]["compression_ratio"]
        assert results1[0]["max_abs_err"] == results2[0]["max_abs_err"]
        assert results1[0]["passed"] == results2[0]["passed"]
        # sanity: latency really does vary (or this test isn't checking what it claims to)
        # -- not asserted strictly (could coincidentally match), just documented via the
        # field being present and read, not silently dropped from the comparison above.
        assert "latency_ms" in results1[0]


def test_kv_benchmark_against_real_npy_file():
    """Exercises the --npy path with a hand-shaped array, matching real
    Gemma's confirmed (kv_heads, seq_len, head_dim) shape (see op.txt)."""
    import numpy as np

    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmp:
        npy_path = os.path.join(tmp, "kv_block.npy")
        rng = np.random.default_rng(0)
        np.save(npy_path, rng.normal(scale=25.0, size=(4, 12, 256)).astype(np.float32))

        result = runner.invoke(cli, ["kv", "benchmark", "--npy", npy_path, "--plugins", "fp8"])
        assert result.exit_code == 0, result.output
        assert "synthetic" not in result.output  # should report using the real file, not synthetic data


def test_kv_demo_runs_end_to_end_through_cli():
    """Slower (full pipeline), but this is the one test that would catch the
    CLI wrapper itself breaking something run_demo.py's own execution
    wouldn't (e.g. a Click context/argument-parsing issue)."""
    runner = CliRunner()
    result = runner.invoke(cli, ["kv", "demo"])
    assert result.exit_code == 0, result.output
    assert "SUCCESS CRITERIA" in result.output
    assert "DONE" in result.output
