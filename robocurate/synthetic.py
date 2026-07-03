"""A controlled, labeled *mixed-quality* demonstration pool for offline verification.

This synthetic pool is NOT evidence for the research claim -- it is a unit-test
fixture that reproduces the structure we expect from low-cost, non-expert data:

* ``expert`` demos  -> tightly clustered around a few task "modes" (low novelty),
                       high task reward.
* ``novice`` demos  -> erratic, spread across gap directions (high novelty),
                       low task reward.

The DPP kernel uses cosine similarity, which depends only on *direction*, so the
novice demos are made *directional* outliers (random directions filling the gaps
between the few expert modes). By construction the novice demos are BOTH
low-quality AND far-from-the-clusters, so a pure farthest-point (diversity-only)
selector chases them first, while a quality x diversity DPP avoids them. The real
research signal comes from the same pipeline on ``lerobot/pusht`` and
``lerobot/aloha_sim_insertion_human`` (see :mod:`robocurate.pusht`).
"""

from __future__ import annotations

import numpy as np


def make_mixed_pool(
    n_expert: int = 150,
    n_novice: int = 56,
    dim: int = 8,
    n_modes: int = 4,
    mode_quality_bias: float = 0.12,
    seed: int = 0,
):
    """Return ``(embeddings, raw_quality, is_expert)`` for a mixed-quality pool.

    ``mode_quality_bias`` makes one mode slightly higher-reward so that a
    *quality-only* selector over-concentrates on it -- exposing its lack of
    diversity (a realistic "the best demos all look alike" scenario).
    """
    rng = np.random.default_rng(seed)

    # A few well-separated mode *directions* (cosine space cares about direction).
    centers = rng.normal(size=(n_modes, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    modes = rng.integers(0, n_modes, size=n_expert)
    Xe = centers[modes] + rng.normal(0.0, 0.10, size=(n_expert, dim))
    qe = rng.uniform(0.70, 0.95, size=n_expert)
    qe = np.clip(qe + mode_quality_bias * (modes == 0), 0.0, 1.0)

    # novice: random directions over the sphere (fill the gaps), low reward.
    Xn = rng.normal(size=(n_novice, dim))
    qn = rng.uniform(0.0, 0.30, size=n_novice)

    X = np.vstack([Xe, Xn])
    q_raw = np.concatenate([qe, qn])
    is_expert = np.concatenate(
        [np.ones(n_expert, dtype=bool), np.zeros(n_novice, dtype=bool)]
    )

    perm = rng.permutation(len(X))
    return X[perm], q_raw[perm], is_expert[perm]


def make_labeled_pool(
    n_expert: int = 150,
    n_novice: int = 56,
    dim: int = 8,
    n_modes: int = 4,
    mode_quality_bias: float = 0.12,
    seed: int = 0,
):
    """Richer pool for the non-circular curation check (Milestone 2.5).

    A latent per-demo *skill* drives BOTH the task ``reward`` AND -- through
    INDEPENDENT noise -- cheap kinematic quality signals (``smoothness``,
    ``efficiency``). This lets us score demos with a *reward-free* proxy quality
    and then evaluate the selection against the held-out reward: the proxy never
    sees the metric, so a win is not circular. It also mirrors the deployable
    setting, where low-cost non-expert collection often has no reward function.

    Returns a dict with keys: ``embeddings, reward, success, smoothness,
    efficiency, skill, is_expert``.
    """
    rng = np.random.default_rng(seed)

    skill = np.concatenate(
        [
            rng.uniform(0.70, 0.98, size=n_expert),   # experts: high skill
            rng.uniform(0.02, 0.30, size=n_novice),   # novices: low skill
        ]
    )
    n = len(skill)

    centers = rng.normal(size=(n_modes, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    modes = rng.integers(0, n_modes, size=n_expert)
    Xe = centers[modes] + rng.normal(0.0, 0.10, size=(n_expert, dim))
    Xn = rng.normal(size=(n_novice, dim))           # novice: gap directions
    X = np.vstack([Xe, Xn])

    is_mode0 = np.concatenate([modes == 0, np.zeros(n_novice, dtype=bool)])
    reward = np.clip(skill + mode_quality_bias * is_mode0
                     + rng.normal(0.0, 0.05, n), 0.0, 1.0)
    success = (reward >= 0.5).astype(float)
    # reward-INDEPENDENT kinematic proxies: share the latent skill, different (larger)
    # noise -> a *noisy* proxy, as real kinematic quality signals are.
    smoothness = np.clip(skill + rng.normal(0.0, 0.22, n), 0.0, 1.0)
    efficiency = np.clip(skill + rng.normal(0.0, 0.22, n), 0.0, 1.0)

    is_expert = np.concatenate(
        [np.ones(n_expert, dtype=bool), np.zeros(n_novice, dtype=bool)]
    )
    perm = rng.permutation(n)
    return {
        "embeddings": X[perm],
        "reward": reward[perm],
        "success": success[perm],
        "smoothness": smoothness[perm],
        "efficiency": efficiency[perm],
        "skill": skill[perm],
        "is_expert": is_expert[perm],
    }
