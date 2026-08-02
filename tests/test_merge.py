"""Tests for hope_core.merge.compute_optimal_parent.

test_merge_of_near_duplicates_approximates_sum is the load-bearing test in
this file: it's the only check in the whole repo that validates the *entire*
merge pipeline end-to-end (direction, sign, scale, effective params, physical
BN-parameter recovery) against something outside the formulas themselves --
actual forward-pass numerical agreement. Everything else here is a plumbing
or robustness check.
"""

import torch

from hope_core.costs import hilbert_inner_product, merge_distortion_squared
from hope_core.kernels import cross_kernel
from hope_core.merge import compute_optimal_parent
from hope_core.neuron import NeuronBank

torch.manual_seed(0)


def _bank_with_correlated_pair(d_in=12, d_out=6, noise=0.02, seed=0) -> NeuronBank:
    g = torch.Generator().manual_seed(seed)
    n = 5
    w_raw = torch.randn(n, d_in, generator=g)
    gamma = torch.randn(n, generator=g).abs() + 0.3
    beta = torch.randn(n, generator=g) * 0.5
    running_mean = torch.randn(n, generator=g) * 0.3
    running_var = torch.randn(n, generator=g).abs() + 0.5
    w_out = torch.randn(n, d_out, generator=g)

    # Make neuron 1 a near-duplicate of neuron 0 (small perturbation) so we
    # have a genuinely highly-correlated pair to merge.
    w_raw[1] = w_raw[0] + noise * torch.randn(d_in, generator=g)
    gamma[1] = gamma[0] + noise * torch.randn(1, generator=g).item()
    beta[1] = beta[0] + noise * torch.randn(1, generator=g).item()
    running_mean[1] = running_mean[0] + noise * torch.randn(1, generator=g).item()
    running_var[1] = running_var[0] + noise * torch.randn(1, generator=g).abs().item()
    w_out[1] = w_out[0] + noise * torch.randn(d_out, generator=g)

    return NeuronBank(
        w_raw=w_raw, gamma=gamma, beta=beta, running_mean=running_mean,
        running_var=running_var, w_out=w_out, bn_eps=1e-5, name="corr",
    )


def _forward_contribution(w_eff_in, b_eff, w_out, x):
    """f(x) = w_out * ReLU(w_eff_in . x + b_eff), batched over x: (B, D_in) -> (B, D_out)."""
    pre_act = x @ w_eff_in + b_eff
    act = torch.relu(pre_act)
    return act.unsqueeze(-1) * w_out.unsqueeze(0)


def test_parent_shapes_correct():
    bank = _bank_with_correlated_pair()
    parent, cap_p = compute_optimal_parent(bank, 0, 1)
    assert parent.w_raw.shape == (bank.w_raw.shape[1],)
    assert parent.w_out.shape == (bank.w_out.shape[1],)
    assert parent.gamma.dim() == 0
    assert parent.beta.dim() == 0
    assert torch.isfinite(cap_p)
    assert cap_p >= 0


def test_capacity_self_consistent_via_public_api():
    """Rebuild a NeuronBank containing only the parent and check that
    .capacity() (computed from raw+BN fields through the normal public path)
    agrees with the capacity compute_optimal_parent returned directly --
    this validates the raw/BN parameter *recovery* step, not just the
    effective-parameter math."""
    bank = _bank_with_correlated_pair(seed=1)
    parent, cap_p_returned = compute_optimal_parent(bank, 0, 1)

    solo_bank = NeuronBank(
        w_raw=parent.w_raw.unsqueeze(0),
        gamma=parent.gamma.unsqueeze(0),
        beta=parent.beta.unsqueeze(0),
        running_mean=parent.running_mean.unsqueeze(0),
        running_var=parent.running_var.unsqueeze(0),
        w_out=parent.w_out.unsqueeze(0),
        bn_eps=bank.bn_eps,
    )
    cap_p_recomputed = solo_bank.capacity()[0]
    assert torch.isclose(cap_p_returned, cap_p_recomputed, rtol=1e-3, atol=1e-5), (
        f"returned={cap_p_returned}, recomputed via raw params={cap_p_recomputed}"
    )


def test_merge_of_near_duplicates_approximates_each_child():
    """The core end-to-end check -- corrected oracle.

    HOPE defines merging as a *rank-1 projection of the pair* [f_i, f_j]
    (paper section 6.3, eq. 6 and Appendix A.6): the constrained
    Eckart-Young-Mirsky objective forces the SAME parent into both slots
    (alpha=beta=1 in W'_joint=[f_p, f_p]), so D^2 = ||f_i-f_p||^2 + ||f_j-f_p||^2
    measures distance to *each child individually* -- this is a centroid /
    common-representative objective, not a sum-reconstruction objective.
    For two near-duplicate neurons the zero-distortion answer is f_p ~= f_i
    ~= f_j individually (capacity unchanged, NOT doubled). An earlier version
    of this test asserted f_p should approximate f_i+f_j, which is a
    different (and for this paper, incorrect) objective -- see NOTES.md.
    """
    torch.manual_seed(123)
    bank = _bank_with_correlated_pair(d_in=16, d_out=8, noise=0.01, seed=2)
    parent, _ = compute_optimal_parent(bank, 0, 1)

    w_eff_in, b_eff = bank.effective_input()
    solo_bank = NeuronBank(
        w_raw=parent.w_raw.unsqueeze(0), gamma=parent.gamma.unsqueeze(0),
        beta=parent.beta.unsqueeze(0), running_mean=parent.running_mean.unsqueeze(0),
        running_var=parent.running_var.unsqueeze(0), w_out=parent.w_out.unsqueeze(0),
        bn_eps=bank.bn_eps,
    )
    w_eff_in_p, b_eff_p = solo_bank.effective_input()

    x = torch.randn(2000, bank.w_raw.shape[1])
    f_i = _forward_contribution(w_eff_in[0], b_eff[0], bank.w_out[0], x)
    f_j = _forward_contribution(w_eff_in[1], b_eff[1], bank.w_out[1], x)
    f_parent = _forward_contribution(w_eff_in_p[0], b_eff_p[0], parent.w_out, x)

    err_i = (f_i - f_parent).norm() / f_i.norm().clamp_min(1e-8)
    err_j = (f_j - f_parent).norm() / f_j.norm().clamp_min(1e-8)

    assert err_i < 0.05, f"parent should approximate child i individually; relative_error={err_i:.4f}"
    assert err_j < 0.05, f"parent should approximate child j individually; relative_error={err_j:.4f}"


def test_merge_beats_naive_alternatives_on_paper_objective():
    """Optimality check: our computed parent should achieve lower D^2 (eq. 6)
    than two naive strawman "parents" -- plain parameter averaging, and just
    reusing child i outright. If HOPE's optimizer isn't beating these, the
    optimization is not doing its job even if it runs without error."""
    torch.manual_seed(55)
    bank = _bank_with_correlated_pair(noise=0.15, seed=8)  # correlated but not near-identical
    parent, cap_p = compute_optimal_parent(bank, 0, 1)

    w_eff_in, _ = bank.effective_input()
    self_k = bank.self_kernel()
    cap_i, cap_j = bank.capacity()[0], bank.capacity()[1]

    def _d2_for_candidate(gamma_c, w_eff_in_c, w_out_c, cap_c):
        stacked_gamma = torch.stack([bank.gamma[0], bank.gamma[1], gamma_c])
        stacked_w = torch.stack([w_eff_in[0], w_eff_in[1], w_eff_in_c])
        stacked_self_k = torch.stack([self_k[0], self_k[1], cap_c.pow(2) / w_out_c.norm().clamp_min(1e-8).pow(2)])
        k = cross_kernel(stacked_gamma, stacked_w, self_k=stacked_self_k)
        inner_ip = hilbert_inner_product(k[0, 2], bank.w_out[0], w_out_c)
        inner_jp = hilbert_inner_product(k[1, 2], bank.w_out[1], w_out_c)
        return merge_distortion_squared(cap_i, cap_j, cap_c, inner_ip, inner_jp)

    d2_hope = _d2_for_candidate(
        parent.gamma, parent.w_raw * parent.gamma / torch.sqrt(parent.running_var + bank.bn_eps), parent.w_out, cap_p
    )

    # Strawman 1: plain average of the two children's raw parameters
    avg_gamma = (bank.gamma[0] + bank.gamma[1]) / 2
    avg_w_eff = (w_eff_in[0] + w_eff_in[1]) / 2
    avg_w_out = (bank.w_out[0] + bank.w_out[1]) / 2
    avg_cap = avg_w_out.norm() * torch.sqrt(
        cross_kernel(torch.stack([avg_gamma, avg_gamma]), torch.stack([avg_w_eff, avg_w_eff]))[0, 0].clamp_min(0.0)
    )
    d2_avg = _d2_for_candidate(avg_gamma, avg_w_eff, avg_w_out, avg_cap)

    # Strawman 2: just reuse child i as the "parent" outright
    d2_reuse_i = _d2_for_candidate(bank.gamma[0], w_eff_in[0], bank.w_out[0], cap_i)

    assert d2_hope <= d2_avg + 1e-4, f"HOPE parent D^2={d2_hope} should beat naive averaging D^2={d2_avg}"
    assert d2_hope <= d2_reuse_i + 1e-4, f"HOPE parent D^2={d2_hope} should beat reusing child i D^2={d2_reuse_i}"


def test_merge_distortion_small_for_near_duplicates_large_for_orthogonal():
    """Sanity-check that J_merge's ingredients (capacity_p, K(i,p), K(j,p))
    computed via compute_optimal_parent give low distortion for a highly
    correlated pair and clearly higher distortion for two unrelated neurons."""
    corr_bank = _bank_with_correlated_pair(noise=0.01, seed=3)
    parent_corr, cap_p_corr = compute_optimal_parent(corr_bank, 0, 1)

    torch.manual_seed(9)
    n, d_in, d_out = 5, 10, 5
    orth_bank = NeuronBank(
        w_raw=torch.randn(n, d_in), gamma=torch.randn(n).abs() + 0.3,
        beta=torch.randn(n) * 0.1, running_mean=torch.randn(n) * 0.1,
        running_var=torch.randn(n).abs() + 0.5, w_out=torch.randn(n, d_out),
        bn_eps=1e-5, name="orth",
    )
    parent_orth, cap_p_orth = compute_optimal_parent(orth_bank, 0, 1)

    def _distortion(bank, i, j, parent, cap_p):
        w_eff_in, _ = bank.effective_input()
        self_k = bank.self_kernel()
        cap_i, cap_j = bank.capacity()[i], bank.capacity()[j]

        stacked_gamma = torch.stack([bank.gamma[i], parent.gamma])
        stacked_w = torch.stack([w_eff_in[i], parent.w_raw * parent.gamma / torch.sqrt(parent.running_var + bank.bn_eps)])
        stacked_self_k = torch.stack([self_k[i], cap_p.pow(2) / parent.w_out.norm().clamp_min(1e-8).pow(2)])
        k_ip = cross_kernel(stacked_gamma, stacked_w, self_k=stacked_self_k)[0, 1]
        inner_ip = hilbert_inner_product(k_ip, bank.w_out[i], parent.w_out)

        stacked_gamma_j = torch.stack([bank.gamma[j], parent.gamma])
        stacked_w_j = torch.stack([w_eff_in[j], parent.w_raw * parent.gamma / torch.sqrt(parent.running_var + bank.bn_eps)])
        stacked_self_k_j = torch.stack([self_k[j], cap_p.pow(2) / parent.w_out.norm().clamp_min(1e-8).pow(2)])
        k_jp = cross_kernel(stacked_gamma_j, stacked_w_j, self_k=stacked_self_k_j)[0, 1]
        inner_jp = hilbert_inner_product(k_jp, bank.w_out[j], parent.w_out)

        return merge_distortion_squared(cap_i, cap_j, cap_p, inner_ip, inner_jp)

    d2_corr = _distortion(corr_bank, 0, 1, parent_corr, cap_p_corr)
    d2_orth = _distortion(orth_bank, 0, 1, parent_orth, cap_p_orth)

    assert d2_corr < d2_orth, f"correlated distortion {d2_corr} should be less than orthogonal distortion {d2_orth}"


def test_merge_does_not_crash_or_nan_on_diverse_inputs():
    """Robustness sweep: merge shouldn't NaN/crash across a range of random banks."""
    for seed in range(15):
        torch.manual_seed(seed)
        n, d_in, d_out = 6, 9, 4
        bank = NeuronBank(
            w_raw=torch.randn(n, d_in), gamma=torch.randn(n).abs() + 0.05,
            beta=torch.randn(n) * 2.0, running_mean=torch.randn(n),
            running_var=torch.randn(n).abs() + 0.1, w_out=torch.randn(n, d_out),
            bn_eps=1e-5, name=f"seed{seed}",
        )
        i, j = seed % n, (seed + 1) % n
        parent, cap_p = compute_optimal_parent(bank, i, j)
        assert torch.isfinite(cap_p), f"seed={seed}: capacity is not finite"
        assert torch.isfinite(parent.w_raw).all(), f"seed={seed}: w_raw has non-finite values"
        assert torch.isfinite(parent.w_out).all(), f"seed={seed}: w_out has non-finite values"
