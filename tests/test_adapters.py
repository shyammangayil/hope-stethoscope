"""Tests for hope_core.adapters.

The round-trip tests are the important ones: they don't just check shapes,
they run an actual forward pass through the original nn.Sequential and
through the rebuilt-from-bank modules and assert numerical agreement. This
is what actually validates the flatten/unflatten conventions (especially
for Conv2d, where a wrong axis order would still produce the right *shapes*
but silently wrong numbers).
"""

import torch
from torch import nn

from hope_core.adapters import (
    apply_bank_to_conv_bn,
    apply_bank_to_linear_bn,
    extract_conv_bn_bank,
    extract_linear_bn_bank,
)

torch.manual_seed(0)


def _make_linear_bn_net(d_in=10, n=8, d_out=5):
    net = nn.Sequential(nn.Linear(d_in, n, bias=False), nn.BatchNorm1d(n), nn.ReLU(), nn.Linear(n, d_out, bias=False))
    net.eval()
    # BN needs realistic (non-default) running stats to be a meaningful test
    with torch.no_grad():
        net[1].running_mean.copy_(torch.randn(n) * 0.3)
        net[1].running_var.copy_(torch.rand(n) * 2 + 0.2)
        net[1].weight.copy_(torch.randn(n).abs() + 0.2)
        net[1].bias.copy_(torch.randn(n) * 0.5)
    return net


def test_linear_bn_roundtrip_matches_original_forward():
    net = _make_linear_bn_net()
    linear, bn, _, next_linear = net[0], net[1], net[2], net[3]

    bank = extract_linear_bn_bank(linear, bn, next_linear)
    new_linear, new_bn, new_next_weight = apply_bank_to_linear_bn(bank, linear_bias=None, next_bias=None)
    new_bn.eval()

    x = torch.randn(64, linear.in_features)
    with torch.no_grad():
        original_out = next_linear(torch.relu(bn(linear(x))))
        rebuilt_out = torch.relu(new_bn(new_linear(x))) @ new_next_weight.T

    assert torch.allclose(original_out, rebuilt_out, atol=1e-4), (original_out - rebuilt_out).abs().max()


def test_linear_bn_bank_shapes_after_pruning_one_neuron():
    net = _make_linear_bn_net(d_in=10, n=8, d_out=5)
    linear, bn, next_linear = net[0], net[1], net[3]
    bank = extract_linear_bn_bank(linear, bn, next_linear)

    pruned_bank = bank.drop(3)
    assert pruned_bank.n_neurons == 7

    new_linear, new_bn, new_next_weight = apply_bank_to_linear_bn(pruned_bank, linear_bias=None, next_bias=None)
    assert new_linear.weight.shape == (7, 10)
    assert new_bn.num_features == 7
    assert new_next_weight.shape == (5, 7)

    x = torch.randn(4, 10)
    new_bn.eval()
    out = torch.relu(new_bn(new_linear(x))) @ new_next_weight.T
    assert out.shape == (4, 5)
    assert torch.isfinite(out).all()


def _make_conv_bn_net(c_in=4, n=6, c_out=3, hw=8):
    net = nn.Sequential(
        nn.Conv2d(c_in, n, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(n),
        nn.ReLU(),
        nn.Conv2d(n, c_out, kernel_size=3, padding=1, bias=False),
    )
    net.eval()
    with torch.no_grad():
        net[1].running_mean.copy_(torch.randn(n) * 0.3)
        net[1].running_var.copy_(torch.rand(n) * 2 + 0.2)
        net[1].weight.copy_(torch.randn(n).abs() + 0.2)
        net[1].bias.copy_(torch.randn(n) * 0.5)
    return net


def test_conv_bn_roundtrip_matches_original_forward():
    net = _make_conv_bn_net()
    conv, bn, next_conv = net[0], net[1], net[3]

    bank = extract_conv_bn_bank(conv, bn, next_conv)
    new_conv, new_bn, new_next_weight = apply_bank_to_conv_bn(
        bank,
        conv_kernel_size=(3, 3),
        next_conv_kernel_size=(3, 3),
        next_conv_out_channels=next_conv.out_channels,
        conv_padding=1,
    )
    new_bn.eval()

    x = torch.randn(2, conv.in_channels, 8, 8)
    with torch.no_grad():
        original_out = next_conv(torch.relu(bn(conv(x))))
        hidden = torch.relu(new_bn(new_conv(x)))
        rebuilt_out = nn.functional.conv2d(hidden, new_next_weight, padding=1)

    assert torch.allclose(original_out, rebuilt_out, atol=1e-4), (original_out - rebuilt_out).abs().max()


def test_conv_bn_shapes_after_merging_two_channels():
    from hope_core.merge import compute_optimal_parent

    net = _make_conv_bn_net(c_in=4, n=6, c_out=3, hw=8)
    conv, bn, next_conv = net[0], net[1], net[3]
    bank = extract_conv_bn_bank(conv, bn, next_conv)

    parent, _ = compute_optimal_parent(bank, 0, 1)
    merged_bank = bank.drop_pair_append(0, 1, parent)
    assert merged_bank.n_neurons == 5

    new_conv, new_bn, new_next_weight = apply_bank_to_conv_bn(
        merged_bank, conv_kernel_size=(3, 3), next_conv_kernel_size=(3, 3),
        next_conv_out_channels=3, conv_padding=1,
    )
    assert new_conv.out_channels == 5
    assert new_bn.num_features == 5
    assert new_next_weight.shape == (3, 5, 3, 3)

    x = torch.randn(2, 4, 8, 8)
    new_bn.eval()
    hidden = torch.relu(new_bn(new_conv(x)))
    out = nn.functional.conv2d(hidden, new_next_weight, padding=1)
    assert out.shape == (2, 3, 8, 8)
    assert torch.isfinite(out).all()
