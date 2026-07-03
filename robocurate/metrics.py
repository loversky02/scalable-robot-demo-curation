"""Metrics for evaluating a selected demonstration subset.

Two axes matter for "usable data from limited human effort":

* ``selected_mean_quality`` -- are the picked demos actually good?  (want HIGH)
* ``coverage_radius``       -- do the picked demos represent the whole pool?
                              mean distance from every demo to its nearest
                              selected demo                       (want LOW)

A good curator scores high quality AND low coverage radius. ``expert_fraction``
is a diagnostic used only with labeled synthetic pools to make the
"diversity-only grabs the noise" failure mode explicit.
"""

from __future__ import annotations

import numpy as np

from .dpp import cosine_similarity_matrix


def selected_mean_quality(selected: list[int], quality: np.ndarray) -> float:
    q = np.asarray(quality, dtype=float)
    if len(selected) == 0:
        return 0.0
    return float(np.mean(q[selected]))


def coverage_radius(selected: list[int], embeddings: np.ndarray) -> float:
    """Mean distance from each demo to the nearest *selected* demo (lower better)."""
    if len(selected) == 0:
        return float("inf")
    D = 1.0 - cosine_similarity_matrix(embeddings)
    nearest = D[:, selected].min(axis=1)
    return float(nearest.mean())


def diversity_spread(selected: list[int], embeddings: np.ndarray) -> float:
    """Mean pairwise cosine distance within the selected set (higher = more diverse)."""
    idx = np.asarray(selected)
    if len(idx) < 2:
        return 0.0
    D = 1.0 - cosine_similarity_matrix(embeddings)
    sub = D[np.ix_(idx, idx)]
    k = len(idx)
    return float(sub.sum() / (k * (k - 1)))


def expert_fraction(selected: list[int], is_expert: np.ndarray) -> float:
    """Diagnostic (labeled pools only): fraction of picks that are true experts."""
    if len(selected) == 0:
        return 0.0
    return float(np.mean(np.asarray(is_expert, dtype=bool)[selected]))
