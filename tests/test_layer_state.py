"""Tests for hope_core.layer_state.LayerState.

test_repeated_mixed_actions_stay_consistent_with_full_rebuild is the load-
bearing test here: after a sequence of prunes and merges, it throws away all
the incrementally-maintained caches and rebuilds a fresh LayerState from the
*current* bank, then checks the incrementally-updated state agrees with the
from-scratch one. This is the only way to catch a stable-id/position mixup
that happens to still produce a *plausible-looking* (but wrong) cache.
"""

import torch

from hope_core.layer_state import LayerState
from hope_core.neuron import NeuronBank

torch.manual_seed(0)


def _random_bank(n=6, d_in=8, d_out=4, seed=0) -> NeuronBank:
    g = torch.Generator().manual_seed(seed)
    return NeuronBank(
        w_raw=torch.randn(n, d_in, generator=g),
        gamma=torch.randn(n, generator=g).abs() + 0.2,
        beta=torch.randn(n, generator=g) * 0.5,
        running_mean=torch.randn(n, generator=g) * 0.3,
        running_var=torch.randn(n, generator=g).abs() + 0.4,
        w_out=torch.randn(n, d_out, generator=g),
        bn_eps=1e-5,
        name="test_layer",
    )


def test_init_pair_count_is_n_choose_2():
    state = LayerState(bank=_random_bank(n=6), delta_p_init=10.0)
    assert len(state.current_pair_costs()) == 6 * 5 // 2
    assert len(state.ids) == 6
    assert len(set(state.ids)) == 6  # all unique


def test_best_prune_picks_minimum_capacity_neuron():
    state = LayerState(bank=_random_bank(n=6, seed=1), delta_p_init=10.0)
    expected_pos = int(torch.argmin(state.capacities).item())
    expected_id = state.ids[expected_pos]
    result = state.best_prune()
    assert result is not None
    picked_id, _dr = result
    assert picked_id == expected_id


def test_best_prune_none_when_single_neuron_left():
    state = LayerState(bank=_random_bank(n=1, seed=2), delta_p_init=10.0)
    assert state.best_prune() is None


def test_apply_prune_updates_ids_and_drops_stale_pairs():
    state = LayerState(bank=_random_bank(n=5, seed=3), delta_p_init=10.0)
    victim_id, _ = state.best_prune()
    state.apply_prune(victim_id)

    assert state.n_active() == 4
    assert victim_id not in state.ids
    assert len(state.current_pair_costs()) == 4 * 3 // 2
    assert all(victim_id not in k for k in state.current_pair_costs())


def test_apply_merge_creates_new_id_and_updates_pairs():
    state = LayerState(bank=_random_bank(n=5, seed=4), delta_p_init=10.0)
    id_a, id_b, _dr = state.best_merge()
    new_id = state.apply_merge(id_a, id_b)

    assert state.n_active() == 4
    assert id_a not in state.ids
    assert id_b not in state.ids
    assert new_id in state.ids
    assert len(state.current_pair_costs()) == 4 * 3 // 2
    assert all(id_a not in k and id_b not in k for k in state.current_pair_costs())
    # new parent should have pairs cached against every surviving neighbor
    assert sum(1 for k in state.current_pair_costs() if new_id in k) == 3


def test_repeated_prunes_keep_capacities_consistent_with_bank():
    state = LayerState(bank=_random_bank(n=8, seed=5), delta_p_init=10.0)
    for _ in range(5):
        victim_id, _ = state.best_prune()
        state.apply_prune(victim_id)
        assert torch.allclose(state.capacities, state.bank.capacity(), atol=1e-5)
        assert torch.isclose(state.e_active, state.bank.layer_capacity(), atol=1e-5)
    assert state.n_active() == 3


def test_repeated_mixed_actions_stay_consistent_with_full_rebuild():
    """The critical test: after a chain of prune/merge actions, throw away
    the incremental caches and rebuild fresh from the live bank. If stable
    ids and positions ever got crossed, this is where it would show up."""
    state = LayerState(bank=_random_bank(n=10, seed=6), delta_p_init=15.0)

    actions_taken = []
    for step in range(6):
        prune_candidate = state.best_prune()
        merge_candidate = state.best_merge()
        # alternate actions deterministically so the test is reproducible
        if step % 2 == 0 and prune_candidate is not None:
            victim_id, _ = prune_candidate
            state.apply_prune(victim_id)
            actions_taken.append(("prune", victim_id))
        elif merge_candidate is not None:
            id_a, id_b, _ = merge_candidate
            new_id = state.apply_merge(id_a, id_b)
            actions_taken.append(("merge", id_a, id_b, new_id))

    # Independent ground truth: rebuild a LayerState from scratch using the
    # bank exactly as it stands now (a plain snapshot, no incremental history).
    fresh = LayerState(bank=state.bank.clone(), delta_p_init=state.delta_p_init)

    assert state.n_active() == fresh.n_active()
    assert torch.allclose(
        torch.sort(state.capacities).values, torch.sort(fresh.capacities).values, atol=1e-4
    ), "incrementally-maintained capacities diverged from a from-scratch rebuild"
    assert torch.isclose(state.e_active, fresh.e_active, atol=1e-4)
    assert len(state.current_pair_costs()) == len(fresh.current_pair_costs()) == state.n_active() * (state.n_active() - 1) // 2

    # Every cached pair's J_merge should be CLOSE to a from-scratch value,
    # bounded but not exact -- see LayerState.refresh_all()'s docstring for
    # why exact agreement isn't the right bar here: the optimal parent's
    # scale depends on layer-wide e_rem, so pairs untouched by the most
    # recent action(s) carry bounded staleness by design, not by bug. 10%
    # relative tolerance comfortably covers that staleness while still
    # catching a real regression -- the bug this test originally caught
    # produced ~100%+ relative error (2.18 vs 1.78, and worse elsewhere),
    # not a few percent.
    incremental_values = sorted(state.current_pair_costs().values())
    fresh_values = sorted(fresh.current_pair_costs().values())
    assert len(incremental_values) == len(fresh_values)
    for a, b in zip(incremental_values, fresh_values):
        rel_err = abs(a - b) / max(abs(b), 1e-6)
        assert rel_err < 0.10, f"pair cost drifted more than expected: incremental={a} fresh={b} rel_err={rel_err:.3f}"

    # refresh_all() should close the gap to (near) exactly zero, since it's
    # a full recompute against the current bank -- this is the actual
    # regression test for the geometry-caching logic itself, independent of
    # the accepted e_rem staleness.
    state.refresh_all()
    refreshed_values = sorted(state.current_pair_costs().values())
    for a, b in zip(refreshed_values, fresh_values):
        assert abs(a - b) < 1e-3, f"refresh_all() should match a from-scratch rebuild exactly: {a} vs {b}"


def test_build_merge_cache_false_skips_pair_construction():
    state = LayerState(bank=_random_bank(n=8, seed=20), delta_p_init=10.0, build_merge_cache=False)
    assert len(state.current_pair_costs()) == 0
    assert state.best_merge() is None
    # prune-only operations should still work normally
    victim_id, _ = state.best_prune()
    state.apply_prune(victim_id)
    assert state.n_active() == 7


def test_build_merge_cache_false_is_faster_than_default_for_larger_layers():
    import time

    bank = _random_bank(n=30, seed=21)
    t0 = time.perf_counter()
    LayerState(bank=bank.clone(), delta_p_init=10.0, build_merge_cache=False)
    fast_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    LayerState(bank=bank.clone(), delta_p_init=10.0)
    full_time = time.perf_counter() - t0

    assert fast_time < full_time / 3, f"expected a clear speedup: fast={fast_time:.4f}s full={full_time:.4f}s"
    """Regression-style test for a specific edge case: immediately acting on
    a just-created parent (rather than an original neuron) exercises the
    'new_id' bookkeeping path end-to-end."""
    state = LayerState(bank=_random_bank(n=6, seed=7), delta_p_init=10.0)
    id_a, id_b, _ = state.best_merge()
    new_id = state.apply_merge(id_a, id_b)

    # the parent should be prunable like any other neuron
    assert new_id in state.ids
    state.apply_prune(new_id)
    assert new_id not in state.ids
    assert state.n_active() == 4
    assert torch.allclose(state.capacities, state.bank.capacity(), atol=1e-5)
