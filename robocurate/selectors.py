"""The four curation strategies compared in the Milestone-1 ablation.

    random          -- uniformly sample K demos (the do-nothing baseline)
    quality_only    -- top-K by task-quality proxy   (ignores diversity)
    diversity_only  -- k-center / farthest-point      (ignores quality)
    dpp             -- quality x diversity DPP MAP     (couples both)

The ablation is designed to isolate each factor: if `dpp` beats both
`quality_only` (more diverse) and `diversity_only` (higher quality), then
*informativeness needs both* -- the thesis of the project.
"""

from __future__ import annotations

import numpy as np

from .dpp import cosine_similarity_matrix, select_dpp

__all__ = [
    "select_random",
    "select_quality_only",
    "select_diversity_only",
    "select_dpp",
    "SELECTORS",
]


def select_random(n: int, k: int, rng: np.random.Generator) -> list[int]:
    k = int(min(k, n))
    return list(rng.choice(n, size=k, replace=False))


def select_quality_only(quality: np.ndarray, k: int) -> list[int]:
    q = np.asarray(quality, dtype=float)
    k = int(min(k, len(q)))
    # stable: highest quality first
    return list(np.argsort(-q, kind="stable")[:k])


def select_diversity_only(embeddings: np.ndarray, k: int) -> list[int]:
    """k-center greedy (farthest-point sampling) on cosine distance.

    Starts from the medoid (neutral, reproducible) then repeatedly adds the point
    farthest from the current set. This is the canonical coverage/diversity
    selector -- and on a noisy pool it is exactly what grabs low-quality
    *outliers* first, which is the failure mode the DPP is meant to avoid.
    """
    X = np.asarray(embeddings, dtype=float)
    n = len(X)
    k = int(min(k, n))
    D = 1.0 - cosine_similarity_matrix(X)          # cosine distance in [0, 2]

    centroid = X.mean(axis=0, keepdims=True)
    medoid = int(np.argmin(np.linalg.norm(X - centroid, axis=1)))
    selected = [medoid]
    min_dist = D[medoid].copy()
    while len(selected) < k:
        min_dist[selected] = -np.inf
        j = int(np.argmax(min_dist))
        selected.append(j)
        min_dist = np.minimum(min_dist, D[j])
    return selected


# name -> callable metadata for the ablation driver
SELECTORS = ("random", "quality_only", "diversity_only", "dpp")
