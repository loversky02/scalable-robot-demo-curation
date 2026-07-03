"""Per-demonstration quality proxy.

For robot demonstrations we do NOT have a ground-truth "informativeness" label.
We use a *task-quality proxy* -- for PushT this is the per-episode reward/coverage
(``next.reward`` / ``next.success`` are available in ``lerobot/pusht``). Higher
task reward => the operator actually accomplished (more of) the task => the demo
is more likely to be useful supervision.

This is deliberately a proxy, not the objective itself: a high-reward demo can
still be redundant. That is precisely why curation needs *diversity* on top of
quality (see :mod:`robocurate.dpp`).
"""

from __future__ import annotations

import numpy as np


def normalize_quality(raw: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Min-max normalize raw task rewards to ``[floor, 1]``.

    ``floor`` keeps every quality strictly positive so no demo is structurally
    unselectable and the DPP kernel keeps full rank.
    """
    r = np.asarray(raw, dtype=float)
    lo, hi = float(r.min()), float(r.max())
    if hi - lo < 1e-12:
        q = np.ones_like(r)
    else:
        q = (r - lo) / (hi - lo)
    return np.clip(q, floor, 1.0)


def proxy_quality(*features: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """Combine reward-INDEPENDENT quality signals into one score (geometric mean).

    Intended inputs are cheap kinematic features -- action smoothness, action
    efficiency, episode duration, jerk -- that do NOT use the task reward. This is
    the *deployable* quality proxy: real low-cost, non-expert collection often has
    no reward function, and scoring the selection this way keeps any downstream
    reward/success evaluation non-circular.
    """
    if not features:
        raise ValueError("proxy_quality needs at least one feature")
    norm = np.stack(
        [normalize_quality(np.asarray(f, dtype=float), floor=floor) for f in features],
        axis=0,
    )
    q = np.exp(np.mean(np.log(norm), axis=0))   # geometric mean, stays in (0, 1]
    return np.clip(q, floor, 1.0)
