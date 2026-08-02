import torch
from torch import nn

from hope_core.deft import (
    apply_structural_mask,
    collect_prune_cost_distribution,
    compute_lock_threshold,
    elasticity_map,
    partition_layer,
    register_deft_gradient_hooks,
    scale_gradient_,
    structural_mask,
)
from hope_core.neuron import NeuronBank

torch.manual_seed(0)


def _bank_with_one_dominant_neuron(n=8, d_in=6, d_out=4, seed=0) -> NeuronBank:
    """Neuron 0 has much higher capacity than everyone else -- should end up
    in the core; the low-capacity ones should end up as slack."""
    g = torch.Generator().manual_seed(seed)
    w_out = torch.randn(n, d_out, generator=g) * 0.3
    w_out[0] *= 15.0  # make neuron 0's capacity dominate
    return NeuronBank(
        w_raw=torch.randn(n, d_in, generator=g),
        gamma=torch.randn(n, generator=g).abs() * 0.5 + 0.3,
        beta=torch.randn(n, generator=g) * 0.2,
        running_mean=torch.randn(n, generator=g) * 0.2,
        running_var=torch.randn(n, generator=g).abs() + 0.3,
        w_out=w_out,
        bn_eps=1e-5,
    )


def test_collect_prune_cost_distribution_covers_all_positions():
    bank = _bank_with_one_dominant_neuron(n=6, seed=1)
    costs = collect_prune_cost_distribution(bank)
    assert set(costs.keys()) <= set(range(6))
    n_infinite = sum(1 for v in costs.values() if v == float("inf"))
    assert n_infinite == 1  # exactly one survivor


def test_collect_prune_cost_distribution_does_not_mutate_input_bank():
    bank = _bank_with_one_dominant_neuron(n=5, seed=2)
    original_w_raw = bank.w_raw.clone()
    collect_prune_cost_distribution(bank)
    assert torch.allclose(bank.w_raw, original_w_raw)
    assert bank.n_neurons == 5


def test_compute_lock_threshold_hand_computed():
    costs = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0, 4: float("inf")}
    # 50th percentile of [1,2,3,4] should sit between 2 and 3
    threshold = compute_lock_threshold(costs, percentile=50)
    assert 2.0 <= threshold <= 3.0


def test_compute_lock_threshold_empty_costs_returns_one():
    assert compute_lock_threshold({}, percentile=50) == 1.0


def test_elasticity_map_thresholds_correctly():
    costs = {0: 0.5, 1: 5.0, 2: 10.0, 3: float("inf")}
    elasticity = elasticity_map(costs, j_lock=5.0, n_neurons=4)
    assert elasticity[0] == 1.0  # 0.5 < 5.0 -> plastic
    assert elasticity[1] == 0.0  # 5.0 is not < 5.0 -> frozen (core)
    assert elasticity[2] == 0.0  # 10.0 >= 5.0 -> frozen
    assert elasticity[3] == 0.0  # inf -> frozen


def test_partition_layer_puts_dominant_neuron_in_core():
    bank = _bank_with_one_dominant_neuron(n=8, seed=3)
    partition = partition_layer(bank, percentile=50)
    assert partition.elasticity[0] == 0.0, "the dominant-capacity neuron should be frozen (core)"
    assert 0.0 < partition.core_fraction < 1.0


def test_structural_mask_hand_computed():
    # 2 downstream neurons (out=2), 3 upstream neurons (in=3)
    e_in = torch.tensor([1.0, 0.0, 1.0])  # neuron 0,2 plastic; neuron 1 frozen
    e_out = torch.tensor([0.0, 1.0])  # neuron 0 frozen; neuron 1 plastic
    mask = structural_mask(e_in, e_out)
    assert mask.shape == (2, 3)
    # row 0 (frozen downstream): sever any plastic-upstream connection -> cols 0,2 = 0; col 1 (frozen upstream) = 1
    assert torch.equal(mask[0], torch.tensor([0.0, 1.0, 0.0]))
    # row 1 (plastic downstream): nothing severed regardless of upstream
    assert torch.equal(mask[1], torch.tensor([1.0, 1.0, 1.0]))


def test_apply_structural_mask_broadcasts_over_conv_kernel_dims():
    weight = torch.ones(2, 3, 3, 3)  # (out=2, in=3, kH=3, kW=3), conv-shaped
    e_in = torch.tensor([1.0, 0.0, 1.0])
    e_out = torch.tensor([0.0, 1.0])
    mask = structural_mask(e_in, e_out)
    masked = apply_structural_mask(weight, mask)
    assert masked.shape == weight.shape
    assert torch.all(masked[0, 0] == 0)  # severed
    assert torch.all(masked[0, 1] == 1)  # frozen upstream -> kept
    assert torch.all(masked[0, 2] == 0)  # severed
    assert torch.all(masked[1] == 1)  # plastic downstream row: untouched


def test_scale_gradient_zeroes_frozen_rows():
    grad = torch.ones(4, 5)
    elasticity_out = torch.tensor([0.0, 1.0, 1.0, 0.0])
    scale_gradient_(grad, elasticity_out)
    assert torch.all(grad[0] == 0)
    assert torch.all(grad[1] == 1)
    assert torch.all(grad[2] == 1)
    assert torch.all(grad[3] == 0)


def test_register_deft_gradient_hooks_scales_real_backward_pass():
    linear = nn.Linear(4, 3, bias=False)
    elasticity_out = torch.tensor([0.0, 1.0, 1.0])  # neuron 0 frozen
    handle = register_deft_gradient_hooks(linear.weight, elasticity_out)

    x = torch.randn(5, 4)
    out = linear(x)
    loss = out.sum()
    loss.backward()

    assert torch.all(linear.weight.grad[0] == 0), "frozen output row's gradient should be zeroed"
    assert not torch.all(linear.weight.grad[1] == 0), "plastic output row should have a real gradient"
    handle.remove()
