"""Tests for the reward-free kinematic proxy features (offline; no lerobot)."""

import numpy as np

from robocurate.pusht import compute_kinematic_features


def test_smooth_straight_trajectory_scores_high():
    actions = np.tile([0.5, 0.0], (20, 1))     # constant action -> straight, no jerk
    f = compute_kinematic_features(actions)
    assert f["smoothness"] > 0.9
    assert f["efficiency"] > 0.9
    assert f["stability"] > 0.9
    assert f["duration"] == 20.0


def test_zigzag_trajectory_is_low_stability_and_efficiency():
    actions = np.array([[1.0, 0.0] if i % 2 == 0 else [-1.0, 0.0] for i in range(20)])
    f = compute_kinematic_features(actions)
    straight = compute_kinematic_features(np.tile([1.0, 0.0], (20, 1)))
    assert f["stability"] < 0.3
    assert straight["efficiency"] > f["efficiency"]
    assert straight["smoothness"] > f["smoothness"]


def test_uses_states_when_provided():
    actions = np.zeros((10, 2))
    states = np.cumsum(np.tile([1.0, 1.0], (10, 1)), axis=0)   # straight diagonal path
    f = compute_kinematic_features(actions, states=states)
    assert f["efficiency"] > 0.9
    assert f["stability"] > 0.9


def test_handles_short_trajectory():
    f = compute_kinematic_features(np.array([[1.0, 0.0]]))
    assert 0.0 <= f["stability"] <= 1.0
    assert f["duration"] == 1.0
