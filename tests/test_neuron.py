"""Tests for hope_core.neuron.NeuronBank.

test_scale_invariance_under_bn_reparameterization is the important one: it
directly checks the paper's central claim (section 5, "Neuron Scale
Invariance") that capacity is immune to the raw-weight rescaling that fools
magnitude-based pruning. If this test ever fails, HOPE's core value
proposition over magnitude/BN-scale baselines is broken for this
implementation, regardless of what everything else in the repo says.
"""

import pytest
import torch

from hope_core.neuron import NeuronBank

torch.manual_seed(0)


def _toy_bank(n=5, d_in=8, d_out=4, seed=0) -> NeuronBank:
    g = torch.Generator().manual_seed(seed)
    return NeuronBank(
        w_raw=torch.randn(n, d_in, generator=g),
        gamma=torch.randn(n, generator=g).abs() + 0.2,
        beta=torch.randn(n, generator=g),
        running_mean=torch.randn(n, generator=g),
        running_var=torch.randn(n, generator=g).abs() + 0.3,
        w_out=torch.randn(n, d_out, generator=g),
        name="toy",
    )


def test_effective_input_hand_computed():
    """Single neuron, hand-computed expected effective weight/bias."""
    bank = NeuronBank(
        w_raw=torch.tensor([[1.0, 2.0, -1.0]]),
        gamma=torch.tensor([2.0]),
        beta=torch.tensor([0.5]),
        running_mean=torch.tensor([1.0]),
        running_var=torch.tensor([3.0]),
        w_out=torch.tensor([[1.0]]),
        bn_eps=0.0,
        name="hand",
    )
    w_eff, b_eff = bank.effective_input()
    denom = 3.0 ** 0.5
    expected_w = torch.tensor([[1.0, 2.0, -1.0]]) * (2.0 / denom)
    expected_b = 0.5 - 2.0 * 1.0 / denom
    assert torch.allclose(w_eff, expected_w, atol=1e-6)
    assert torch.allclose(b_eff, torch.tensor([expected_b]), atol=1e-6)


def test_capacity_shape_and_nonnegative():
    bank = _toy_bank()
    cap = bank.capacity()
    assert cap.shape == (5,)
    assert torch.all(cap >= 0)
    assert torch.all(torch.isfinite(cap))


def test_layer_capacity_is_sum_of_neuron_capacities():
    bank = _toy_bank()
    assert torch.isclose(bank.layer_capacity(), bank.capacity().sum())


def test_scale_invariance_under_bn_reparameterization():
    """Paper section 5: rescaling (w_raw, running_mean, running_var) by (lam, lam, lam^2)
    represents the *same* underlying function and must leave capacity unchanged.
    This is the property that makes HOPE robust to the scale-symmetry artifacts
    that fool magnitude-based pruning baselines."""
    bank = _toy_bank(seed=7)
    # use a tiny bn_eps so the eps-vs-variance approximation in the derivation
    # doesn't itself introduce a real (rather than numerical) discrepancy
    bank.bn_eps = 1e-10

    original_capacity = bank.capacity().clone()

    for lam in [0.01, 0.5, 2.0, 37.0]:
        rescaled = NeuronBank(
            w_raw=bank.w_raw * lam,
            gamma=bank.gamma.clone(),
            beta=bank.beta.clone(),
            running_mean=bank.running_mean * lam,
            running_var=bank.running_var * (lam ** 2),
            w_out=bank.w_out.clone(),
            bn_eps=bank.bn_eps,
            name="rescaled",
        )
        rescaled_capacity = rescaled.capacity()
        assert torch.allclose(rescaled_capacity, original_capacity, rtol=1e-3, atol=1e-5), (
            f"capacity changed under lambda={lam}: {original_capacity} vs {rescaled_capacity}"
        )


def test_shape_invariance_independent_of_fan_in():
    """Capacity depends only on (gamma, beta, w_out) -- two neurons with different
    fan-in but identical (gamma, beta, w_out) must have identical capacity
    (paper footnote 3: capacity is decoupled from the physical width of the input)."""
    bank_small = NeuronBank(
        w_raw=torch.randn(1, 4),
        gamma=torch.tensor([1.3]),
        beta=torch.tensor([-0.4]),
        running_mean=torch.tensor([0.1]),
        running_var=torch.tensor([0.9]),
        w_out=torch.tensor([[0.7, -0.2]]),
        name="small",
    )
    bank_wide = NeuronBank(
        w_raw=torch.randn(1, 4000),  # 1000x wider fan-in
        gamma=torch.tensor([1.3]),
        beta=torch.tensor([-0.4]),
        running_mean=torch.tensor([0.1]),
        running_var=torch.tensor([0.9]),
        w_out=torch.tensor([[0.7, -0.2]]),
        name="wide",
    )
    assert torch.isclose(bank_small.capacity()[0], bank_wide.capacity()[0], atol=1e-6)


def test_drop_removes_correct_neuron():
    bank = _toy_bank(n=4, seed=3)
    original_w_raw = bank.w_raw.clone()
    reduced = bank.drop(2)
    assert reduced.n_neurons == 3
    assert torch.allclose(reduced.w_raw[0], original_w_raw[0])
    assert torch.allclose(reduced.w_raw[1], original_w_raw[1])
    assert torch.allclose(reduced.w_raw[2], original_w_raw[3])  # neuron 3 shifted into slot 2


def test_clone_is_independent():
    bank = _toy_bank()
    cloned = bank.clone()
    cloned.w_raw[0, 0] = 999.0
    assert bank.w_raw[0, 0] != 999.0


def test_with_replaced_w_out_updates_capacity_correctly():
    bank = _toy_bank(n=4, d_out=3, seed=9)
    original_capacity = bank.capacity().clone()

    new_w_out = torch.ones(4, 5) * 2.0  # different shape (d_out 3->5) and different values
    updated = bank.with_replaced_w_out(new_w_out)

    assert updated.w_out.shape == (4, 5)
    assert not torch.allclose(updated.capacity(), original_capacity), "capacity must reflect the new w_out"
    # self-kernel (depends only on gamma/beta, not w_out) must be unchanged
    assert torch.allclose(updated.self_kernel(), bank.self_kernel())


def test_with_replaced_w_out_does_not_mutate_original():
    bank = _toy_bank(n=3, seed=10)
    original_w_out = bank.w_out.clone()
    new_w_out = torch.zeros(3, 4)
    bank.with_replaced_w_out(new_w_out)
    assert torch.allclose(bank.w_out, original_w_out), "original bank must be untouched"


def test_with_replaced_w_out_wrong_row_count_raises():
    bank = _toy_bank(n=4, seed=11)
    with pytest.raises(ValueError, match="n_neurons"):
        bank.with_replaced_w_out(torch.zeros(3, 5))  # 3 rows, bank has 4 neurons
