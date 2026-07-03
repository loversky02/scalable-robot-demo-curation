"""Controlled, labeled demonstration pools for offline verification.

These synthetic pools are NOT evidence for the research claim -- they are
unit-test fixtures that reproduce the structure we expect from low-cost,
non-expert data and let us debug the DPP / proxy-q logic offline. The research
signal comes from running the SAME curation logic on the public `lerobot/pusht`
and `lerobot/aloha_sim_insertion_human` datasets (see :mod:`robocurate.pusht`).
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
    """Return ``(embeddings, raw_quality, is_expert)`` for the M1 ablation.

    Experts cluster in a few mode *directions* (low novelty, high reward); novices
    are random gap directions (high novelty, low reward). A pure farthest-point
    selector therefore chases the novices, while a quality x diversity DPP avoids
    them. ``mode_quality_bias`` makes mode 0 slightly higher-reward so quality-only
    over-concentrates there.
    """
    rng = np.random.default_rng(seed)

    centers = rng.normal(size=(n_modes, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    modes = rng.integers(0, n_modes, size=n_expert)
    Xe = centers[modes] + rng.normal(0.0, 0.10, size=(n_expert, dim))
    qe = rng.uniform(0.70, 0.95, size=n_expert)
    qe = np.clip(qe + mode_quality_bias * (modes == 0), 0.0, 1.0)

    Xn = rng.normal(size=(n_novice, dim))
    qn = rng.uniform(0.0, 0.30, size=n_novice)

    X = np.vstack([Xe, Xn])
    q_raw = np.concatenate([qe, qn])
    is_expert = np.concatenate(
        [np.ones(n_expert, dtype=bool), np.zeros(n_novice, dtype=bool)]
    )
    perm = rng.permutation(len(X))
    return X[perm], q_raw[perm], is_expert[perm]


# tier codes for the labeled pool
EXPERT, CLUMSY, NOISY = 0, 1, 2
TIER_NAMES = ("expert", "clumsy", "noisy")


def make_labeled_pool(
    n_expert: int = 100,
    n_clumsy: int = 50,
    n_noisy: int = 56,
    dim: int = 8,
    n_modes: int = 4,
    mode_quality_bias: float = 0.12,
    seed: int = 0,
):
    """A three-tier (expert / clumsy / noisy) mixed-quality pool.

    A latent per-demo *skill* drives BOTH the task ``reward`` AND -- through
    INDEPENDENT noise -- cheap REWARD-FREE kinematic signals (``smoothness``,
    ``efficiency``, ``stability``). Scoring the selection with those kinematics and
    then evaluating it against the held-out ``reward`` is therefore non-circular:
    the selector never reads the metric. This also mirrors the deployable setting,
    where low-cost non-expert collection often has no reward function at all.

    Returns a dict: ``embeddings, reward, success, smoothness, efficiency,
    stability, skill, tier, is_expert``.
    """
    rng = np.random.default_rng(seed)

    skill = np.concatenate(
        [
            rng.uniform(0.75, 0.98, size=n_expert),   # expert
            rng.uniform(0.40, 0.65, size=n_clumsy),   # clumsy but recoverable
            rng.uniform(0.00, 0.25, size=n_noisy),    # noisy / erratic
        ]
    )
    n = len(skill)

    centers = rng.normal(size=(n_modes, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    modes_e = rng.integers(0, n_modes, size=n_expert)
    modes_c = rng.integers(0, n_modes, size=n_clumsy)
    Xe = centers[modes_e] + rng.normal(0.0, 0.10, size=(n_expert, dim))   # tight
    Xc = centers[modes_c] + rng.normal(0.0, 0.35, size=(n_clumsy, dim))   # looser
    Xn = rng.normal(size=(n_noisy, dim))                                  # gap dirs
    X = np.vstack([Xe, Xc, Xn])

    mode = np.concatenate([modes_e, modes_c, np.full(n_noisy, -1)])
    reward = np.clip(
        skill + mode_quality_bias * (mode == 0) + rng.normal(0.0, 0.05, n), 0.0, 1.0
    )
    success = (reward >= 0.5).astype(float)

    # reward-INDEPENDENT kinematic proxies (share skill, separate noisy draws)
    smoothness = np.clip(skill + rng.normal(0.0, 0.22, n), 0.0, 1.0)
    efficiency = np.clip(skill + rng.normal(0.0, 0.22, n), 0.0, 1.0)
    stability = np.clip(skill + rng.normal(0.0, 0.22, n), 0.0, 1.0)

    tier = np.concatenate(
        [np.full(n_expert, EXPERT), np.full(n_clumsy, CLUMSY), np.full(n_noisy, NOISY)]
    )
    is_expert = tier == EXPERT

    perm = rng.permutation(n)
    return {
        "embeddings": X[perm],
        "reward": reward[perm],
        "success": success[perm],
        "smoothness": smoothness[perm],
        "efficiency": efficiency[perm],
        "stability": stability[perm],
        "skill": skill[perm],
        "tier": tier[perm],
        "is_expert": is_expert[perm],
    }
