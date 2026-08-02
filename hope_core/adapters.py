"""Extract a NeuronBank from real PyTorch layers, and write a compressed bank
back as real, smaller, loadable nn.Module weights.

This is the boundary between HOPE's math (hope_core/*.py, which only knows
about tensors) and an actual model on disk. Two layer families are
supported, matching what the paper's own experiments use (per Appendix B.1,
the fully-connected derivation "seamlessly extends to convolutional
networks", and the ResNet-50 experiments in section 11 rely on exactly this
adaptation):

    - nn.Linear + nn.BatchNorm1d, consumed by a following nn.Linear.
    - nn.Conv2d (1x1 or KxK) + nn.BatchNorm2d, consumed by a following
      nn.Conv2d. Each output channel is treated as one "neuron"; its raw
      input weight is the flattened (in_channels * kH * kW) kernel slice,
      and its outgoing weight is the flattened slice of the *next* layer's
      kernel that reads from this channel.

Both directions (extract / apply) round-trip losslessly when the bank is
unmodified -- see test_adapters.py's round-trip test, which is really a
correctness check on the whole flatten/unflatten convention, not just a
shape check.
"""

from __future__ import annotations

import torch
from torch import nn

from hope_core.neuron import NeuronBank

_EPS_DEFAULT = 1e-5


def extract_linear_bn_bank(linear: nn.Linear, bn: nn.BatchNorm1d, next_linear: nn.Linear, name: str = "layer") -> NeuronBank:
    """Build a NeuronBank from Linear -> BN -> (implicit activation) -> next_linear."""
    if linear.out_features != bn.num_features:
        raise ValueError(f"{name}: linear.out_features={linear.out_features} != bn.num_features={bn.num_features}")
    if next_linear.in_features != linear.out_features:
        raise ValueError(
            f"{name}: next_linear.in_features={next_linear.in_features} != linear.out_features={linear.out_features}"
        )
    return NeuronBank(
        w_raw=linear.weight.detach().clone(),
        gamma=bn.weight.detach().clone(),
        beta=bn.bias.detach().clone(),
        running_mean=bn.running_mean.detach().clone(),
        running_var=bn.running_var.detach().clone(),
        w_out=next_linear.weight.detach().clone().T,  # (out_features, in_features) -> (in_features==N, out_features)
        bn_eps=bn.eps,
        name=name,
    )


def apply_bank_to_linear_bn(
    bank: NeuronBank, linear_bias: torch.Tensor | None, next_bias: torch.Tensor | None
) -> tuple[nn.Linear, nn.BatchNorm1d, torch.Tensor]:
    """Rebuild (Linear, BatchNorm1d, next_layer_weight) from a (possibly compressed) bank.

    Returns the new next-layer WEIGHT (not a full nn.Linear, since the caller
    already owns the next layer's identity/bias/output-side connections and
    only its input dimension changes here).
    """
    n, d_in = bank.w_raw.shape
    d_out = bank.w_out.shape[1]

    new_linear = nn.Linear(d_in, n, bias=linear_bias is not None)
    with torch.no_grad():
        new_linear.weight.copy_(bank.w_raw)
        if linear_bias is not None:
            # raw linear bias is independent of BN's own (beta, mean) and is
            # preserved as-is; pruning/merging operate on BN-absorbed capacity,
            # not on this bias, so it's carried through unchanged per surviving row.
            new_linear.bias.copy_(linear_bias)

    new_bn = nn.BatchNorm1d(n, eps=bank.bn_eps)
    with torch.no_grad():
        new_bn.weight.copy_(bank.gamma)
        new_bn.bias.copy_(bank.beta)
        new_bn.running_mean.copy_(bank.running_mean)
        new_bn.running_var.copy_(bank.running_var)

    new_next_weight = bank.w_out.T.clone()  # (in_features==N, out_features) -> (out_features, in_features)
    return new_linear, new_bn, new_next_weight


def extract_conv_bn_bank(conv: nn.Conv2d, bn: nn.BatchNorm2d, next_conv: nn.Conv2d, name: str = "conv_layer") -> NeuronBank:
    """Build a NeuronBank from Conv2d -> BN -> (implicit activation) -> next_conv.

    Each output channel is one neuron. w_raw for channel i is the flattened
    (in_channels * kH * kW) kernel; w_out for channel i is the flattened
    slice of next_conv's kernel that reads channel i, i.e.
    next_conv.weight[:, i, :, :] flattened over (out_channels' * kH' * kW').
    """
    if conv.out_channels != bn.num_features:
        raise ValueError(f"{name}: conv.out_channels={conv.out_channels} != bn.num_features={bn.num_features}")
    if next_conv.in_channels != conv.out_channels:
        raise ValueError(f"{name}: next_conv.in_channels={next_conv.in_channels} != conv.out_channels={conv.out_channels}")

    n = conv.out_channels
    w_raw = conv.weight.detach().reshape(n, -1).clone()  # (N, C_in*kH*kW)
    # next_conv.weight: (C_out', C_in==N, kH', kW') -> per-input-channel slice, flattened
    w_out = next_conv.weight.detach().permute(1, 0, 2, 3).reshape(n, -1).clone()  # (N, C_out'*kH'*kW')

    return NeuronBank(
        w_raw=w_raw,
        gamma=bn.weight.detach().clone(),
        beta=bn.bias.detach().clone(),
        running_mean=bn.running_mean.detach().clone(),
        running_var=bn.running_var.detach().clone(),
        w_out=w_out,
        bn_eps=bn.eps,
        name=name,
    )


def apply_bank_to_conv_bn(
    bank: NeuronBank,
    conv_kernel_size: tuple[int, int],
    next_conv_kernel_size: tuple[int, int],
    next_conv_out_channels: int,
    conv_stride=1,
    conv_padding=0,
) -> tuple[nn.Conv2d, nn.BatchNorm2d, torch.Tensor]:
    """Rebuild (Conv2d, BatchNorm2d, next_conv_weight) from a (possibly compressed) bank.

    Like apply_bank_to_linear_bn, returns the next layer's raw WEIGHT tensor
    rather than a full module, since the caller owns the next conv's other
    attributes (stride, padding, bias, etc.) and only its in_channels changes.
    """
    n = bank.w_raw.shape[0]
    kh, kw = conv_kernel_size
    c_in = bank.w_raw.shape[1] // (kh * kw)
    if c_in * kh * kw != bank.w_raw.shape[1]:
        raise ValueError(f"w_raw fan-in {bank.w_raw.shape[1]} doesn't factor as C_in*{kh}*{kw}")

    new_conv = nn.Conv2d(c_in, n, kernel_size=(kh, kw), stride=conv_stride, padding=conv_padding, bias=False)
    with torch.no_grad():
        new_conv.weight.copy_(bank.w_raw.reshape(n, c_in, kh, kw))

    new_bn = nn.BatchNorm2d(n, eps=bank.bn_eps)
    with torch.no_grad():
        new_bn.weight.copy_(bank.gamma)
        new_bn.bias.copy_(bank.beta)
        new_bn.running_mean.copy_(bank.running_mean)
        new_bn.running_var.copy_(bank.running_var)

    nkh, nkw = next_conv_kernel_size
    # w_out rows are (N, C_out' * kH' * kW') -> next conv weight is (C_out', N, kH', kW')
    new_next_weight = bank.w_out.reshape(n, next_conv_out_channels, nkh, nkw).permute(1, 0, 2, 3).clone()
    return new_conv, new_bn, new_next_weight
