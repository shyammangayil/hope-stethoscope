"""Tests for hope_core.block_eviction.

test_apply_eviction_collapses_a_real_residual_block_to_identity is the
important one: it builds an actual small residual block (conv->bn->relu
twice, added to a skip connection) and checks that after eviction, the
block's output exactly equals its input on real random data -- not just
that the BN params became zero.
"""

import torch
from torch import nn

from hope_core.block_eviction import (
    ambient_identity_energy,
    apply_block_eviction,
    block_eviction_cost,
    is_evicted,
)

torch.manual_seed(0)


def test_ambient_identity_energy_hand_computed():
    gamma = torch.tensor([3.0, 0.0, 1.0])
    beta = torch.tensor([4.0, 5.0, 0.0])
    # sqrt(9+16) + sqrt(0+25) + sqrt(1+0) = 5 + 5 + 1 = 11
    energy = ambient_identity_energy(gamma, beta)
    assert torch.isclose(energy, torch.tensor(11.0))


def test_block_eviction_cost_matches_hand_computation():
    n_active = [4, 3]
    e_active = [torch.tensor(2.0), torch.tensor(1.5)]
    gamma = torch.tensor([1.0, 1.0])
    beta = torch.tensor([0.0, 0.0])
    # numerator = 4*2.0 + 3*1.5 = 8 + 4.5 = 12.5; e_identity = sqrt(1)+sqrt(1) = 2.0
    cost = block_eviction_cost(n_active, e_active, gamma, beta)
    assert torch.isclose(cost, torch.tensor(12.5 / 2.0))


def test_block_eviction_cost_mismatched_lengths_raises():
    try:
        block_eviction_cost([1, 2, 3], [torch.tensor(1.0)], torch.tensor([1.0]), torch.tensor([0.0]))
        assert False, "expected ValueError for mismatched list lengths"
    except ValueError:
        pass


def test_block_eviction_cost_lower_when_internal_layers_already_depleted():
    """A block whose internal layers have already been heavily pruned
    (low N_active, low E_active) should be cheaper to evict than a block
    that's still fully intact -- eviction should look attractive once
    granular pruning has already hollowed out W1/W2."""
    gamma = torch.tensor([1.0, 1.0, 1.0])
    beta = torch.tensor([0.2, 0.2, 0.2])

    intact_cost = block_eviction_cost([8, 8], [torch.tensor(5.0), torch.tensor(5.0)], gamma, beta)
    depleted_cost = block_eviction_cost([2, 2], [torch.tensor(0.5), torch.tensor(0.5)], gamma, beta)

    assert depleted_cost < intact_cost


class _ToyResidualBlock(nn.Module):
    """Minimal but real residual block: X -> [conv,bn,relu] x2 -> +X."""

    def __init__(self, channels=6):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)  # the "final" BN, right before the residual add
        self.relu = nn.ReLU()

    def forward(self, x):
        f = self.relu(self.bn1(self.conv1(x)))
        f = self.bn2(self.conv2(f))
        return x + f


def _random_init_block(block: _ToyResidualBlock) -> _ToyResidualBlock:
    with torch.no_grad():
        for bn in (block.bn1, block.bn2):
            bn.weight.copy_(torch.randn_like(bn.weight).abs() + 0.3)
            bn.bias.copy_(torch.randn_like(bn.bias) * 0.5)
            bn.running_mean.copy_(torch.randn_like(bn.running_mean) * 0.2)
            bn.running_var.copy_(torch.rand_like(bn.running_var) + 0.5)
    block.eval()
    return block


def test_apply_eviction_collapses_a_real_residual_block_to_identity():
    block = _random_init_block(_ToyResidualBlock(channels=6))
    x = torch.randn(3, 6, 10, 10)

    with torch.no_grad():
        pre_eviction_out = block(x)
    # sanity: before eviction, the block should NOT already be an identity map
    assert not torch.allclose(pre_eviction_out, x, atol=1e-3)

    apply_block_eviction(block.bn2)

    with torch.no_grad():
        post_eviction_out = block(x)
    assert torch.allclose(post_eviction_out, x, atol=1e-5), (
        "after eviction, block(x) should exactly equal x (pure identity mapping)"
    )


def test_is_evicted_detects_state_correctly():
    block = _random_init_block(_ToyResidualBlock(channels=4))
    assert not is_evicted(block.bn2)
    apply_block_eviction(block.bn2)
    assert is_evicted(block.bn2)
    assert not is_evicted(block.bn1)  # only bn2 was evicted
