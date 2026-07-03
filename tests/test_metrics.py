"""Tests for the evaluation metrics."""

import numpy as np

from robocurate import (
    coverage_radius,
    diversity_spread,
    expert_fraction,
    selected_mean_quality,
)


def test_selected_mean_quality():
    q = np.array([0.2, 0.8, 0.5, 1.0])
    assert np.isclose(selected_mean_quality([1, 3], q), 0.9)
    assert selected_mean_quality([], q) == 0.0


def test_expert_fraction():
    is_expert = np.array([True, False, True, True])
    assert np.isclose(expert_fraction([0, 1], is_expert), 0.5)
    assert np.isclose(expert_fraction([0, 2, 3], is_expert), 1.0)


def test_diversity_spread_of_identical_is_zero():
    X = np.ones((5, 3))
    assert diversity_spread([0, 1, 2], X) == 0.0


def test_diversity_spread_orthogonal_is_positive():
    X = np.eye(4)
    assert diversity_spread([0, 1, 2, 3], X) > 0.5


def test_coverage_radius_full_selection_is_zero():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    assert np.isclose(coverage_radius(list(range(10)), X), 0.0, atol=1e-9)


def test_coverage_radius_decreases_with_more_points():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 5))
    few = coverage_radius([0, 1], X)
    many = coverage_radius(list(range(30)), X)
    assert many < few
