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
