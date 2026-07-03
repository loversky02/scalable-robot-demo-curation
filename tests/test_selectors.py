"""Behavioural tests for the four selectors."""

import numpy as np

from robocurate import (
    select_diversity_only,
    select_dpp,
    select_quality_only,
    select_random,
)


def test_random_distinct_and_in_range():
    rng = np.random.default_rng(0)
    sel = select_random(50, 12, rng)
    assert len(sel) == 12
    assert len(set(sel)) == 12
    assert all(0 <= i < 50 for i in sel)


def test_random_caps_at_n():
    rng = np.random.default_rng(0)
    sel = select_random(5, 20, rng)
    assert len(sel) == 5
    assert len(set(sel)) == 5


def test_quality_only_picks_top_quality():
    q = np.array([0.1, 0.9, 0.5, 0.95, 0.2])
    sel = select_quality_only(q, 2)
    assert set(sel) == {3, 1}          # 0.95 and 0.9
    assert sel[0] == 3                 # highest first


def test_diversity_only_distinct():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 6))
    sel = select_diversity_only(X, 10)
    assert len(sel) == 10
    assert len(set(sel)) == 10


def test_dpp_distinct_and_capped():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(15, 4))
    q = rng.uniform(0.1, 1.0, size=15)
    sel = select_dpp(X, q, 30)         # k > n
    assert len(set(sel)) == len(sel)
    assert len(sel) <= 15


def test_dpp_prefers_high_quality_when_similar():
    """Two near-identical directions with different quality: DPP should keep the
    higher-quality one before adding redundant low-quality copies."""
    X = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.001],   # near-duplicate of row 0
            [0.0, 1.0],     # orthogonal, distinct direction
        ]
    )
    q = np.array([0.95, 0.2, 0.9])
    sel = select_dpp(X, q, 2)
    # should pick the two distinct high-quality directions (0 and 2), not the dup
    assert set(sel) == {0, 2}
