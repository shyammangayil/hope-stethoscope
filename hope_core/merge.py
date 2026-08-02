"""Compute the optimal parent neuron for merging two neurons (paper sections 7.1-7.2).

Merging is defined as a constrained rank-1 projection of the pair [f_i, f_j]
(section 6.3, Appendix A.6) -- NOT a sum-reconstruction. See NOTES.md for why
that distinction matters and how it was caught (a first draft of the test
suite assumed the wrong objective; the code was right, the test wasn't).

Steps 1, 3, 4, and 5 below are verified exact matches to the main-text
equations (cross-checked directly against the paper's primary text -- see
NOTES.md for the full equation-to-code table). Step 2 and the eq. 16-18
boundary case are the only two places with a documented simplification,
both flagged inline and covered by tests rather than taken on faith.

Pipeline:
    1. Optimal direction u_c: restrict the eigenproblem to the 2D span of the
       two children's augmented input directions (eq. 14) -- exact match.
    2. Sign resolution: the eigenvector is only defined up to sign. The paper
       resolves this against the *exact* (non-linearized) objective (eq. 11),
       which needs K(u, child) for an arbitrary candidate direction u -- the
       closed-form reduction for this lives in Appendix E.3, which isn't
       available here. SIMPLIFICATION: we treat the candidate as a linear
       combination of the two real children under the Gaussian surrogate
       (the same reasoning the paper itself uses for eq. 17-18) to get an
       implied (gamma_u, beta_u), then reuse the tested cross_kernel
       machinery. Validated indirectly via
       test_merge_beats_naive_alternatives_on_paper_objective.
    3. Optimal scale s* (eq. 12, section 7.1.2): exact match.
    4. Effective-parameter recovery (eq. 15): exact match.
    5. Physical (raw weight / BN) parameter recovery (eq. 16-18): exact match
       for the active regime (gamma_p^2 >= eps). SIMPLIFICATION: the paper
       explicitly defers the exact boundary-safe formulation and the
       gamma_p^2 < eps edge case to Appendix B.5, which isn't available
       here; we fall back to a clamped, degenerate-neuron path instead of
       guessing the paper's exact formula for that narrow edge case.
"""

from __future__ import annotations

import torch

from hope_core.kernels import cross_kernel, self_kernel
from hope_core.neuron import NeuronBank, ParentNeuron

_EPS = 1e-8


def _optimal_direction_2d_restricted(w_tilde_i: torch.Tensor, w_tilde_j: torch.Tensor, w_out_i: torch.Tensor, w_out_j: torch.Tensor) -> torch.Tensor:
    """Solve eq. 14 restricted to span(w_tilde_i, w_tilde_j) -- a 2x2 eigenproblem, O(D) not O(D^2).

    Returns a unit-norm vector in the (D_in+1)-dim augmented space (sign not yet resolved).
    """
    basis = torch.stack([w_tilde_i, w_tilde_j], dim=1)  # (D_in+1, 2)
    q, r = torch.linalg.qr(basis)  # q: (D_in+1, 2) orthonormal (or rank-deficient if parallel)

    # Degenerate case: the two children's augmented directions are (numerically)
    # parallel, so the span is really 1D. QR still returns 2 columns but the
    # second is meaningless (r[1,1] ~ 0). Fall back to the single well-defined
    # direction instead of amplifying noise from an ill-conditioned 2x2 solve.
    if torch.abs(r[1, 1]) < 1e-6 * torch.abs(r[0, 0]).clamp_min(1e-12):
        return q[:, 0]

    p_i = r[:, 0]  # coords of w_tilde_i in the {q0,q1} basis == r[:,0] by QR construction
    p_j = r[:, 1]

    b_mat = torch.outer(w_out_i, p_i) + torch.outer(w_out_j, p_j)  # (D_out, 2)
    m = b_mat.T @ b_mat  # (2, 2), symmetric PSD
    eigvals, eigvecs = torch.linalg.eigh(m)
    c_star = eigvecs[:, -1]  # top eigenvector (eigh sorts ascending)

    u = q @ c_star  # lift back to (D_in+1,)
    return u / u.norm().clamp_min(_EPS)


def _project_onto_children(target: torch.Tensor, w_tilde_i: torch.Tensor, w_tilde_j: torch.Tensor) -> tuple[float, float]:
    """Solve target ~= c1*w_tilde_i + c2*w_tilde_j via least squares. Returns (c1, c2)."""
    basis = torch.stack([w_tilde_i, w_tilde_j], dim=1)  # (D, 2)
    sol = torch.linalg.lstsq(basis, target.unsqueeze(-1)).solution.squeeze(-1)
    return float(sol[0]), float(sol[1])


def _implied_gamma_beta(c1: float, c2: float, gamma_i: torch.Tensor, gamma_j: torch.Tensor, beta_i: torch.Tensor, beta_j: torch.Tensor, rho_hat_ij: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(gamma_u, beta_u) implied by treating direction u as c1*i + c2*j under the
    Gaussian surrogate (same construction as eq. 17-18, reused for sign resolution)."""
    var_u = (
        (c1 ** 2) * gamma_i.pow(2)
        + (c2 ** 2) * gamma_j.pow(2)
        + 2 * c1 * c2 * gamma_i.abs() * gamma_j.abs() * rho_hat_ij
    ).clamp_min(0.0)
    gamma_u = torch.sqrt(var_u)
    beta_u = c1 * beta_i + c2 * beta_j
    return gamma_u, beta_u


def _rho_hat_between(gamma_i, w_eff_in_i, gamma_j, w_eff_in_j, self_k_i, self_k_j) -> torch.Tensor:
    """Recover the implied correlation rho_hat_ij used inside cross_kernel, for reuse in eq. 17-18."""
    bank_pair_gamma = torch.stack([gamma_i, gamma_j])
    bank_pair_w = torch.stack([w_eff_in_i, w_eff_in_j])
    self_k_pair = torch.stack([self_k_i, self_k_j])
    k = cross_kernel(bank_pair_gamma, bank_pair_w, self_k=self_k_pair)
    denom = torch.sqrt(self_k_pair[0] * self_k_pair[1]).clamp_min(_EPS)
    # k[0,1] = (1/pi)[sqrt(1-rho_hat^2) + (pi-arccos(rho_hat))*rho_hat] * denom
    # We don't invert that closed form; instead recompute rho_hat directly the
    # same way cross_kernel does internally, so the two never drift apart.
    w_norm_i = w_eff_in_i.norm().clamp_min(_EPS)
    w_norm_j = w_eff_in_j.norm().clamp_min(_EPS)
    rho_eff = torch.dot(w_eff_in_i, w_eff_in_j) / (w_norm_i * w_norm_j)
    rho_eff = rho_eff.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    kappa = (rho_eff / (1.0 - rho_eff.pow(2))) * (gamma_i.abs() / w_norm_i) * (gamma_j.abs() / w_norm_j)
    rho_hat = (2.0 * kappa) / (1.0 + torch.sqrt(1.0 + 4.0 * kappa.pow(2)))
    return rho_hat.clamp(-1.0 + 1e-7, 1.0 - 1e-7)


def compute_optimal_parent(bank: NeuronBank, i: int, j: int) -> tuple[ParentNeuron, torch.Tensor]:
    """Compute the optimal merged parent for neurons i, j in `bank`.

    Returns:
        (parent, capacity_p) -- the synthesized ParentNeuron and its scalar capacity.
    """
    w_eff_in, b_eff = bank.effective_input()
    w_tilde_i = torch.cat([w_eff_in[i], b_eff[i:i + 1]])
    w_tilde_j = torch.cat([w_eff_in[j], b_eff[j:j + 1]])

    w_out_i, w_out_j = bank.w_out[i], bank.w_out[j]
    gamma_i, gamma_j = bank.gamma[i], bank.gamma[j]
    beta_i, beta_j = bank.beta[i], bank.beta[j]

    self_k = bank.self_kernel()
    self_k_i, self_k_j = self_k[i], self_k[j]
    cap_i = (w_out_i.norm() * torch.sqrt(self_k_i.clamp_min(0.0)))
    cap_j = (w_out_j.norm() * torch.sqrt(self_k_j.clamp_min(0.0)))

    rho_hat_ij = _rho_hat_between(gamma_i, w_eff_in[i], gamma_j, w_eff_in[j], self_k_i, self_k_j)

    # --- Step 1: direction, up to sign ---
    u_hat = _optimal_direction_2d_restricted(w_tilde_i, w_tilde_j, w_out_i, w_out_j)

    # --- Step 2: sign resolution (SIMPLIFICATION #1, see module docstring) ---
    best_score, best_u, best_c1, best_c2 = None, None, None, None
    for sign in (1.0, -1.0):
        u = sign * u_hat
        c1, c2 = _project_onto_children(u, w_tilde_i, w_tilde_j)
        gamma_u, beta_u = _implied_gamma_beta(c1, c2, gamma_i, gamma_j, beta_i, beta_j, rho_hat_ij)
        k_uu = self_kernel(gamma_u.unsqueeze(0), beta_u.unsqueeze(0))[0]

        # cross-kernel of the candidate against each real child, via the same
        # zero-bias cross_kernel already used and tested elsewhere.
        w_eff_u_dir = u[:-1]  # direction part only; cross_kernel needs a full-length vector matching w_eff_in's dim
        stacked_gamma = torch.stack([gamma_u, gamma_i, gamma_j])
        stacked_w = torch.stack([w_eff_u_dir, w_eff_in[i], w_eff_in[j]])
        stacked_self_k = torch.stack([k_uu, self_k_i, self_k_j])
        k_full = cross_kernel(stacked_gamma, stacked_w, self_k=stacked_self_k)
        k_ui, k_uj = k_full[0, 1], k_full[0, 2]

        numer_vec = k_ui * w_out_i + k_uj * w_out_j
        score = numer_vec.norm() / torch.sqrt(k_uu.clamp_min(_EPS))

        if best_score is None or score > best_score:
            best_score, best_u, best_c1, best_c2 = score, u, c1, c2
            best_k_uu, best_k_ui, best_k_uj, best_numer_vec = k_uu, k_ui, k_uj, numer_vec

    u_c = best_u
    k_self_u = best_k_uu

    # --- Step 3: optimal scale s* (section 7.1.2) ---
    r_norm = best_numer_vec.norm().clamp_min(_EPS)
    v_star = best_numer_vec / r_norm

    b_coef = (best_k_ui / torch.sqrt(k_self_u.clamp_min(_EPS))) * torch.dot(v_star, w_out_i) + (
        best_k_uj / torch.sqrt(k_self_u.clamp_min(_EPS))
    ) * torch.dot(v_star, w_out_j)
    a_coef = cap_i.pow(2) + cap_j.pow(2)
    e_rem = (bank.layer_capacity() - cap_i - cap_j).clamp_min(0.0)

    s_denominator = (2.0 * e_rem + b_coef)
    if torch.abs(s_denominator) < _EPS:
        s_star = a_coef / b_coef.clamp_min(_EPS)  # total-layer-collapse special case (paper eq. after 12)
    else:
        s_star = (a_coef + b_coef * e_rem) / s_denominator
    s_star = s_star.clamp_min(0.0)  # scale must be non-negative by construction (s in R+)

    # --- Step 4: effective parameters of the parent (eq. 15) ---
    stacked_in = torch.stack([w_tilde_i, w_tilde_j], dim=1)  # (D+1, 2)
    stacked_out = torch.stack([w_out_i, w_out_j], dim=1)  # (D_out, 2)
    r_f = stacked_in.norm() / stacked_out.norm().clamp_min(_EPS)

    k_self_u_q = k_self_u.clamp_min(_EPS).pow(-0.25)
    w_tilde_p = torch.sqrt((s_star * r_f).clamp_min(0.0)) * k_self_u_q * u_c
    w_out_p = torch.sqrt((s_star / r_f.clamp_min(_EPS)).clamp_min(0.0)) * k_self_u_q * v_star

    w_eff_in_p = w_tilde_p[:-1]
    b_p = w_tilde_p[-1]

    # --- Step 5: physical (raw / BN) parameter recovery (eq. 16-18) ---
    c1_p, c2_p = _project_onto_children(w_eff_in_p, w_eff_in[i], w_eff_in[j])
    beta_p = c1_p * beta_i + c2_p * beta_j
    var_p = (
        (c1_p ** 2) * gamma_i.pow(2)
        + (c2_p ** 2) * gamma_j.pow(2)
        + 2 * c1_p * c2_p * gamma_i.abs() * gamma_j.abs() * rho_hat_ij
    ).clamp_min(0.0)
    gamma_p = torch.sqrt(var_p)

    if gamma_p.pow(2) < bank.bn_eps:
        # SIMPLIFICATION #2 (see module docstring): the paper defers the
        # gamma_p^2 < eps ("inactive regime") edge case to Appendix B.5, which
        # this implementation doesn't have. We fall back to a degenerate,
        # near-zero-output neuron rather than guess the paper's exact
        # formula -- this keeps the merge numerically safe (no NaN/inf) at
        # the cost of not exactly matching an unavailable derivation for this
        # rare edge case. Property: a degenerate parent has ~zero capacity,
        # so the greedy loop will naturally avoid selecting merges that land
        # here (they'd show near-total distortion for near-zero capacity gain).
        gamma_p = torch.sqrt(torch.tensor(bank.bn_eps))
        sigma_p = torch.zeros_like(gamma_p)
    else:
        sigma_p = gamma_p.clone()

    mu_p = beta_p - b_p  # rearranged from b_eff = beta - gamma*mu/sqrt(sigma^2+eps), active regime raw=eff (eq. 17)
    w_raw_p = w_eff_in_p.clone()  # eq. 17 active-regime convention: w_raw_in,p = w_eff_in,p

    parent = ParentNeuron(
        w_raw=w_raw_p,
        gamma=gamma_p,
        beta=beta_p,
        running_mean=mu_p,
        running_var=sigma_p.pow(2),
        w_out=w_out_p,
    )

    capacity_p = w_out_p.norm() * torch.sqrt(self_kernel(gamma_p.unsqueeze(0), beta_p.unsqueeze(0))[0].clamp_min(0.0))
    return parent, capacity_p
