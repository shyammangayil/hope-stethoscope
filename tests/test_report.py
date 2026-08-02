import json
import os
import tempfile

import torch

from hope_core.encoding_loop import ProgressiveEncodingLoop
from hope_core.layer_state import LayerState
from hope_core.neuron import NeuronBank
from hope_core.report import StethoscopeReport

torch.manual_seed(0)


def _random_bank(n, d_in=8, d_out=4, seed=0) -> NeuronBank:
    g = torch.Generator().manual_seed(seed)
    return NeuronBank(
        w_raw=torch.randn(n, d_in, generator=g),
        gamma=torch.randn(n, generator=g).abs() + 0.2,
        beta=torch.randn(n, generator=g) * 0.5,
        running_mean=torch.randn(n, generator=g) * 0.3,
        running_var=torch.randn(n, generator=g).abs() + 0.4,
        w_out=torch.randn(n, d_out, generator=g),
        bn_eps=1e-5,
        name=f"layer_{seed}",
    )


def _run_small_loop(n=10, target_density=0.5, seed=0):
    bank = _random_bank(n, seed=seed)
    initial_banks = {"layer0": bank.clone()}
    loop = ProgressiveEncodingLoop(layers={"layer0": LayerState(bank=bank, delta_p_init=10.0, name="layer0")})
    loop.run(target_density=target_density)
    return loop, initial_banks


def test_survival_table_covers_every_original_neuron_exactly_once():
    loop, initial_banks = _run_small_loop(n=10, seed=1)
    report = StethoscopeReport.from_loop(loop, initial_banks)
    assert len(report.survival_table) == 10
    positions = sorted(e.original_position for e in report.survival_table)
    assert positions == list(range(10))


def test_survivors_plus_removed_matches_action_history():
    loop, initial_banks = _run_small_loop(n=10, seed=2, target_density=0.4)
    report = StethoscopeReport.from_loop(loop, initial_banks)
    n_removed = sum(1 for e in report.survival_table if not e.survived)
    n_original_ids_touched = sum(len(r.involved_ids) for r in loop.history)
    assert n_removed == n_original_ids_touched


def test_core_slack_partition_sizes():
    loop, initial_banks = _run_small_loop(n=10, seed=3, target_density=0.3)
    report = StethoscopeReport.from_loop(loop, initial_banks)
    core, slack = report.core_slack_partition(top_fraction=0.5)
    survivors = [e for e in report.survival_table if e.survived]
    assert len(core) + len(slack) == len(survivors)
    if core and slack:
        assert min(e.capacity for e in core) >= max(e.capacity for e in slack) - 1e-5


def test_survival_ranking_is_sorted_descending():
    loop, initial_banks = _run_small_loop(n=8, seed=4)
    report = StethoscopeReport.from_loop(loop, initial_banks)
    ranking = report.survival_ranking()
    caps = [e.capacity for e in ranking]
    assert caps == sorted(caps, reverse=True)


def test_action_counts_matches_history():
    loop, initial_banks = _run_small_loop(n=10, seed=5, target_density=0.4)
    report = StethoscopeReport.from_loop(loop, initial_banks)
    counts = report.action_counts()
    assert counts["prune"] + counts["merge"] == len(loop.history)
    assert counts["prune"] == sum(1 for r in loop.history if r.action == "prune")
    assert counts["merge"] == sum(1 for r in loop.history if r.action == "merge")


def test_magnitude_baseline_comparison_returns_valid_fraction():
    loop, initial_banks = _run_small_loop(n=12, seed=6, target_density=0.5)
    report = StethoscopeReport.from_loop(loop, initial_banks)
    result = report.compare_to_magnitude_baseline()
    if result["compared_count"] > 0:
        assert 0.0 <= result["agreement_fraction"] <= 1.0


def test_to_dict_and_save_json_round_trip():
    loop, initial_banks = _run_small_loop(n=10, seed=7, target_density=0.4)
    report = StethoscopeReport.from_loop(loop, initial_banks)
    d = report.to_dict()
    assert 0.0 <= d["final_density"] <= 1.0
    assert d["n_survivors"] + d["n_removed"] == 10

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "report.json")
        report.save_json(path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == d


def test_capacity_heatmap_data_shapes():
    loop, initial_banks = _run_small_loop(n=9, seed=8)
    report = StethoscopeReport.from_loop(loop, initial_banks)
    heatmap = report.capacity_heatmap_data()
    assert list(heatmap.keys()) == ["layer0"]
    assert len(heatmap["layer0"]) == 9
