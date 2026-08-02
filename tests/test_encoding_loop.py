"""Tests for hope_core.encoding_loop.ProgressiveEncodingLoop."""

import torch

from hope_core.encoding_loop import ProgressiveEncodingLoop
from hope_core.layer_state import LayerState
from hope_core.neuron import NeuronBank

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
        name=f"bank_{seed}",
    )


def test_initial_total_params_is_sum_across_layers():
    loop = ProgressiveEncodingLoop(layers={
        "a": LayerState(bank=_random_bank(6, seed=1), delta_p_init=12.0, name="a"),
        "b": LayerState(bank=_random_bank(4, seed=2), delta_p_init=20.0, name="b"),
    })
    assert loop.initial_total_params == 6 * 12.0 + 4 * 20.0
    assert loop.total_params == loop.initial_total_params
    assert loop.current_density == 1.0


def test_run_requires_a_termination_condition():
    loop = ProgressiveEncodingLoop(layers={"a": LayerState(bank=_random_bank(4, seed=3), delta_p_init=10.0)})
    try:
        loop.run()
        assert False, "expected ValueError when neither target_density nor max_steps is given"
    except ValueError:
        pass


def test_single_layer_run_reaches_target_density():
    loop = ProgressiveEncodingLoop(layers={"a": LayerState(bank=_random_bank(10, seed=4), delta_p_init=10.0)})
    records = loop.run(target_density=0.5)
    assert loop.current_density <= 0.5
    assert len(records) == len(loop.history)
    assert len(records) > 0


def test_density_decreases_monotonically():
    loop = ProgressiveEncodingLoop(layers={"a": LayerState(bank=_random_bank(12, seed=5), delta_p_init=10.0)})
    densities = [1.0]
    for _ in range(8):
        rec = loop.step()
        if rec is None:
            break
        densities.append(loop.current_density)
    assert all(densities[i] >= densities[i + 1] for i in range(len(densities) - 1))


def test_max_steps_caps_the_run():
    loop = ProgressiveEncodingLoop(layers={"a": LayerState(bank=_random_bank(15, seed=6), delta_p_init=10.0)})
    records = loop.run(max_steps=3)
    assert len(records) == 3
    assert len(loop.history) == 3


def test_run_stops_when_no_actions_remain():
    """Every layer down to 1 neuron: further steps should return None, and
    run() should terminate rather than loop forever."""
    loop = ProgressiveEncodingLoop(layers={
        "a": LayerState(bank=_random_bank(2, seed=7), delta_p_init=10.0),
    })
    records = loop.run(max_steps=100)  # far more than needed to exhaust a 2-neuron layer
    assert len(records) < 100  # must have stopped early via the None-return path
    assert loop.layers["a"].n_active() == 1
    assert loop.step() is None


def test_cross_layer_selection_prefers_globally_cheapest_action():
    """Construct two layers where one has an obviously cheap prune candidate
    (near-zero capacity neuron) and the other doesn't; the loop's first
    action should come from the cheap layer, not just layer registration
    order."""
    torch.manual_seed(42)
    cheap_bank = _random_bank(6, seed=8)
    # zero out one neuron's output weight entirely -> capacity ~0 -> trivially cheap to prune
    with torch.no_grad():
        cheap_bank.w_out[2].zero_()

    expensive_bank = _random_bank(6, seed=9)  # all neurons have "normal" capacity

    loop = ProgressiveEncodingLoop(layers={
        "expensive": LayerState(bank=expensive_bank, delta_p_init=10.0, name="expensive"),
        "cheap": LayerState(bank=cheap_bank, delta_p_init=10.0, name="cheap"),
    })
    first_action = loop.step()
    assert first_action is not None
    assert first_action.layer_name == "cheap", (
        f"expected the near-zero-capacity neuron's layer to win first, got {first_action.layer_name}"
    )


def test_audit_trail_records_are_well_formed():
    loop = ProgressiveEncodingLoop(layers={"a": LayerState(bank=_random_bank(8, seed=10), delta_p_init=10.0, name="a")})
    loop.run(max_steps=4)
    assert len(loop.history) == 4
    for i, record in enumerate(loop.history):
        assert record.step == i
        assert record.layer_name == "a"
        assert record.action in ("prune", "merge")
        assert record.distortion_rate >= 0
        assert record.params_freed == 10.0
        assert record.wall_time_seconds >= 0
        if record.action == "prune":
            assert len(record.involved_ids) == 1
            assert record.result_id is None
        else:
            assert len(record.involved_ids) == 2
            assert record.result_id is not None
