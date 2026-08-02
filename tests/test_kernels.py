"""Tests for hope_core.kernels.

The self-kernel test doesn't just check the implementation against itself --
it Monte Carlo simulates the actual definition (E[ReLU(Z)^2] for Z ~
N(beta, gamma^2)) and checks the closed form against that simulation. This is
the one place in the whole library where we can verify a HOPE formula against
first principles rather than against the paper's own text.
"""

import math

import torch

from hope_core.kernels import cross_kernel, normal_cdf, normal_pdf, self_kernel

torch.manual_seed(0)


def test_normal_cdf_pdf_sane_values():
    x = torch.tensor([-3.0, 0.0, 3.0])
    cdf = normal_cdf(x)
    pdf = normal_pdf(x)
    assert torch.allclose(cdf, torch.tensor([0.00135, 0.5, 0.99865]), atol=1e-4)
    assert torch.allclose(pdf, torch.tensor([0.004432, 0.398942, 0.004432]), atol=1e-4)


def test_self_kernel_matches_monte_carlo():
    """E[ReLU(Z)^2] for Z ~ N(beta, gamma^2), checked against direct simulation."""
    torch.manual_seed(42)
    gamma = torch.tensor([0.5, 1.0, 2.0, 0.1, -1.5])  # negative gamma must still work (uses |gamma|)
    beta = torch.tensor([0.0, 1.0, -1.0, 3.0, -0.5])

    closed_form = self_kernel(gamma, beta)

    n_samples = 4_000_000
    z = torch.randn(n_samples, gamma.shape[0]) * gamma.abs() + beta
    relu_sq = torch.relu(z).pow(2)
    monte_carlo = relu_sq.mean(dim=0)

    # Monte Carlo with 4M samples has non-trivial variance for the larger
    # gamma/beta combos, so allow a generous but still meaningful tolerance.
    rel_err = (closed_form - monte_carlo).abs() / monte_carlo.clamp_min(1e-3)
    assert torch.all(rel_err < 0.02), f"closed_form={closed_form}, monte_carlo={monte_carlo}, rel_err={rel_err}"


def test_self_kernel_zero_beta_special_case():
    """At beta=0, E[ReLU(Z)^2] for Z~N(0,gamma^2) has a simple known form: gamma^2/2."""
    gamma = torch.tensor([1.0, 2.0, 0.3])
    beta = torch.zeros_like(gamma)
    k = self_kernel(gamma, beta)
    expected = gamma.pow(2) / 2.0
    assert torch.allclose(k, expected, atol=1e-5)


def test_self_kernel_nonnegative_and_finite():
    torch.manual_seed(1)
    gamma = torch.randn(200) * 3.0
    beta = torch.randn(200) * 3.0
    k = self_kernel(gamma, beta)
    assert torch.all(k >= 0)
    assert torch.all(torch.isfinite(k))


def test_cross_kernel_diagonal_equals_self_kernel():
    torch.manual_seed(2)
    n, d = 6, 10
    w_eff_in = torch.randn(n, d)
    gamma = torch.randn(n).abs() + 0.1
    beta = torch.randn(n)
    sk = self_kernel(gamma, beta)
    ck = cross_kernel(gamma, w_eff_in, self_k=sk)
    assert torch.allclose(ck.diagonal(), sk, atol=1e-5)


def test_cross_kernel_symmetric():
    torch.manual_seed(3)
    n, d = 8, 16
    w_eff_in = torch.randn(n, d)
    gamma = torch.randn(n).abs() + 0.1
    beta = torch.randn(n)
    sk = self_kernel(gamma, beta)
    ck = cross_kernel(gamma, w_eff_in, self_k=sk)
    assert torch.allclose(ck, ck.T, atol=1e-6)


def test_cross_kernel_identical_neurons_matches_self_kernel():
    """Two rows that are exact duplicates should have K(i,j) very close to K(i,i)."""
    torch.manual_seed(4)
    w_eff_in = torch.randn(1, 12).repeat(2, 1)  # neuron 0 and 1 are identical
    gamma = torch.tensor([1.0, 1.0])
    beta = torch.tensor([0.0, 0.0])  # zero-bias approximation is exact here
    sk = self_kernel(gamma, beta)
    ck = cross_kernel(gamma, w_eff_in, self_k=sk)
    assert torch.isclose(ck[0, 1], sk[0], atol=1e-3)


def test_cross_kernel_bounded_by_cauchy_schwarz():
    """|K(i,j)| should never exceed sqrt(K(i,i) * K(j,j)) by more than numerical slack."""
    torch.manual_seed(5)
    n, d = 15, 20
    w_eff_in = torch.randn(n, d)
    gamma = torch.randn(n).abs() + 0.1
    beta = torch.randn(n) * 0.01  # near-zero bias, where the approximation is most faithful
    sk = self_kernel(gamma, beta)
    ck = cross_kernel(gamma, w_eff_in, self_k=sk)
    bound = torch.sqrt(sk.unsqueeze(1) * sk.unsqueeze(0)) + 1e-4
    assert torch.all(ck <= bound), f"violation: {(ck - bound).max()}"
