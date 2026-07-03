"""Loader for a collected/built mixed-quality PushT pool (.npz).

The pool stores, per episode, the raw ``states`` and ``actions`` (real gym_pusht /
real human dynamics), the peak ``ep_reward`` (a REAL simulator signal),
``ep_success``, and an ``operator_mode`` tier (0=expert, 1=clumsy, 2=noisy).

We represent every demo by the agent trajectory (first two state dims) + action,
so human demos (2-d state) and scripted demos (5-d state) are comparable, and
compute the same reward-free kinematic proxy used everywhere else.
"""

from __future__ import annotations

import numpy as np

from .pusht import _temporal_pool, compute_kinematic_features


def load_collected_pool(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    states, actions = d["states"], d["actions"]
    reward = d["ep_reward"].astype(float)
    success = d["ep_success"].astype(float)
    mode = d["operator_mode"].astype(int)

    emb, sm, ef, st, du, keep = [], [], [], [], [], []
    for i, (s, a) in enumerate(zip(states, actions)):
        s = np.asarray(s, dtype=float)
        a = np.asarray(a, dtype=float)
        if len(a) < 3 or len(s) < 3:         # drop degenerate (0-2 step) episodes
            continue
        sxy = s[:, :2]                       # agent trajectory (comparable across sources)
        emb.append(np.concatenate([_temporal_pool(sxy), _temporal_pool(a)]))
        k = compute_kinematic_features(a, states=sxy)
        sm.append(k["smoothness"]); ef.append(k["efficiency"])
        st.append(k["stability"]); du.append(k["duration"])
        keep.append(i)

    keep = np.asarray(keep, dtype=int)
    emb = np.nan_to_num(np.asarray(emb, dtype=float))
    # z-score features (pixel-scale states/actions) so the cosine kernel is
    # well-conditioned -- avoids near-duplicate numerical blow-ups in greedy MAP.
    emb = (emb - emb.mean(axis=0)) / (emb.std(axis=0) + 1e-8)
    return {
        "embeddings": emb,
        "reward": reward[keep],
        "success": success[keep],
        "smoothness": np.asarray(sm),
        "efficiency": np.asarray(ef),
        "stability": np.asarray(st),
        "duration": np.asarray(du),
        "tier": mode[keep],
        "is_expert": mode[keep] == 0,
        "reward_available": True,
        "name": f"collected mixed-quality pool ({path})",
    }
