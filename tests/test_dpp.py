"""Correctness tests for the DPP kernel and greedy MAP inference."""

import numpy as np

from robocurate.dpp import build_dpp_kernel, greedy_map


def test_kernel_is_psd_and_symmetric():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    q = rng.uniform(0.1, 1.0, size=20)
    L = build_dpp_kernel(X, q)
    assert np.allclose(L, L.T, atol=1e-10)
    eigs = np.linalg.eigvalsh(L)
    assert eigs.min() > -1e-8  # PSD up to numerical noise


def test_negative_quality_rejected():
    X = np.random.default_rng(1).normal(size=(5, 3))
    q = np.array([0.5, -0.1, 0.2, 0.3, 0.4])
    try:
        build_dpp_kernel(X, q)
    except ValueError:
        return
    raise AssertionError("expected ValueError for negative quality")


def test_greedy_returns_distinct_indices():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 6))
    q = rng.uniform(0.1, 1.0, size=30)
    L = build_dpp_kernel(X, q)
    sel = greedy_map(L, 10)
    assert len(sel) == 10
    assert len(set(sel)) == 10
    assert all(0 <= i < 30 for i in sel)


def test_greedy_gain_product_equals_subdeterminant():
    """The product of per-step gains must equal det(L_S): validates the
    incremental Cholesky recurrence independently of optimality."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(25, 4))
    q = rng.uniform(0.2, 1.0, size=25)
    L = build_dpp_kernel(X, q)
    sel, gains = greedy_map(L, 8, return_gains=True)
    sub = L[np.ix_(sel, sel)]
    logdet = np.linalg.slogdet(sub)[1]
    assert np.isclose(np.sum(np.log(gains)), logdet, atol=1e-6)


def test_greedy_beats_random_on_logdet():
    """Greedy MAP should achieve a larger log-det than random subsets on average."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 5))
    q = rng.uniform(0.2, 1.0, size=40)
    L = build_dpp_kernel(X, q)
    k = 8
    greedy_logdet = np.linalg.slogdet(L[np.ix_(*(2 * [greedy_map(L, k)]))])[1]
    rand_logdets = []
    for _ in range(50):
        idx = list(rng.choice(40, size=k, replace=False))
        rand_logdets.append(np.linalg.slogdet(L[np.ix_(idx, idx)])[1])
    assert greedy_logdet > np.mean(rand_logdets)


def test_greedy_is_deterministic():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 6))
    q = rng.uniform(0.1, 1.0, size=30)
    L = build_dpp_kernel(X, q)
    assert greedy_map(L, 12) == greedy_map(L, 12)
