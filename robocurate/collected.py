"""Loader for a collected/built mixed-quality PushT pool (.npz).

The pool stores, per episode, the raw ``states`` and ``actions`` (real gym_pusht /
real human dynamics), the peak ``ep_reward`` (a REAL simulator signal),
``ep_success``, and an ``operator_mode`` tier (0=expert, 1=clumsy, 2=noisy).

We represent every demo by the agent trajectory (first two state dims) + action,
so demos are comparable, and compute the same reward-free kinematic proxy used
everywhere else. The filtered raw ``states``/``actions`` are also returned so a
downstream BC policy (M3-lite) can train on them.
"""

from __future__ import annotations

import numpy as np

from .pusht import _temporal_pool, compute_kinematic_features


def pool_from_arrays(states, actions, reward, success, mode) -> dict:
    """Build a curation pool dict from per-episode raw arrays (drops degenerate
    <3-step episodes; z-scores embeddings for a well-conditioned cosine kernel)."""
    reward = np.asarray(reward, dtype=float)
    success = np.asarray(success, dtype=float)
    mode = np.asarray(mode, dtype=int)

    emb, sm, ef, st, du, keep = [], [], [], [], [], []
    kept_states, kept_actions = [], []
    for i, (s, a) in enumerate(zip(states, actions)):
        s = np.asarray(s, dtype=float)
        a = np.asarray(a, dtype=float)
        if len(a) < 3 or len(s) < 3:
            continue
        sxy = s[:, :2]
        emb.append(np.concatenate([_temporal_pool(sxy), _temporal_pool(a)]))
        k = compute_kinematic_features(a, states=sxy)
        sm.append(k["smoothness"]); ef.append(k["efficiency"])
        st.append(k["stability"]); du.append(k["duration"])
        keep.append(i); kept_states.append(s); kept_actions.append(a)

    keep = np.asarray(keep, dtype=int)
    emb = np.nan_to_num(np.asarray(emb, dtype=float))
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
        "states": kept_states,
        "actions": kept_actions,
    }


def load_collected_pool(path: str) -> dict:
    d = np.load(path, allow_pickle=True)
    p = pool_from_arrays(d["states"], d["actions"], d["ep_reward"],
                         d["ep_success"], d["operator_mode"])
    p["name"] = f"collected mixed-quality pool ({path})"
    return p


def load_and_merge(paths) -> dict:
    """Concatenate several collected .npz pools (e.g. human experts + scripted)."""
    S, A, R, SU, M = [], [], [], [], []
    for path in paths:
        d = np.load(path, allow_pickle=True)
        S += list(d["states"]); A += list(d["actions"])
        R += list(d["ep_reward"]); SU += list(d["ep_success"]); M += list(d["operator_mode"])
    p = pool_from_arrays(S, A, np.asarray(R), np.asarray(SU), np.asarray(M))
    p["name"] = "merged pool (" + ", ".join(paths) + ")"
    return p
