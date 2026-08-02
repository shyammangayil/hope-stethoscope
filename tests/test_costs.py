import torch

from hope_core.costs import distortion_rate, hilbert_inner_product, merge_cost, merge_distortion_squared, prune_cost


def test_prune_cost_blows_up_near_layer_exhaustion():
    e_active = torch.tensor(10.0)
    low = prune_cost(torch.tensor(1.0), n_active=5, e_active=e_active)
    high = prune_cost(torch.tensor(9.99), n_active=5, e_active=e_active)
    assert high > low
    assert high > 100 * low  # should blow up, not just increase mildly


def test_prune_cost_monotonic_in_capacity():
    e_active = torch.tensor(20.0)
    caps = torch.tensor([1.0, 5.0, 10.0, 15.0])
    costs = torch.stack([prune_cost(c, n_active=8, e_active=e_active) for c in caps])
    assert torch.all(costs[1:] > costs[:-1])


def test_merge_distortion_zero_for_perfect_duplicate_parent():
    """If i and j are identical and the parent exactly reproduces them,
    merging should cost (near) nothing -- no information is lost."""
    cap = torch.tensor(3.0)
    inner = cap * cap  # <f_i, f_p> = ||f_i|| * ||f_p|| when perfectly aligned
    d2 = merge_distortion_squared(cap, cap, cap, inner, inner)
    assert torch.isclose(d2, torch.tensor(0.0), atol=1e-5)


def test_merge_distortion_positive_for_orthogonal_neurons():
    """Merging two orthogonal (unrelated) neurons should be expensive -- the
    parent can't simultaneously represent two independent directions."""
    cap_i = torch.tensor(2.0)
    cap_j = torch.tensor(2.0)
    cap_p = torch.tensor(2.0)
    inner_ip = torch.tensor(0.0)  # orthogonal: no shared inner product
    inner_jp = torch.tensor(0.0)
    d2 = merge_distortion_squared(cap_i, cap_j, cap_p, inner_ip, inner_jp)
    assert d2 > 10.0  # should be clearly nonzero and large relative to capacities


def test_hilbert_inner_product_matches_definition():
    k = torch.tensor(2.5)
    w_i = torch.tensor([1.0, 2.0, 3.0])
    w_j = torch.tensor([0.5, -1.0, 2.0])
    expected = 2.5 * (1.0 * 0.5 + 2.0 * -1.0 + 3.0 * 2.0)
    got = hilbert_inner_product(k, w_i, w_j)
    assert torch.isclose(got, torch.tensor(expected))


def test_distortion_rate_is_simple_ratio():
    cost = torch.tensor(4.0)
    delta_p = torch.tensor(2.0)
    assert torch.isclose(distortion_rate(cost, delta_p), torch.tensor(2.0))


def test_merge_cost_favors_correlated_over_orthogonal_pairs():
    """Sanity check end-to-end: given equal capacities, a highly-correlated
    (cheap to merge) pair should have lower J_merge than an orthogonal pair."""
    e_active = torch.tensor(50.0)
    cap = torch.tensor(3.0)

    # correlated pair: parent nearly reproduces both children
    d2_correlated = merge_distortion_squared(cap, cap, cap, inner_ip=torch.tensor(8.9), inner_jp=torch.tensor(8.9))
    j_correlated = merge_cost(cap, cap, cap, d2_correlated, n_active=10, e_active=e_active)

    # orthogonal pair: parent captures neither child well
    d2_orthogonal = merge_distortion_squared(cap, cap, cap, inner_ip=torch.tensor(0.0), inner_jp=torch.tensor(0.0))
    j_orthogonal = merge_cost(cap, cap, cap, d2_orthogonal, n_active=10, e_active=e_active)

    assert j_correlated < j_orthogonal
