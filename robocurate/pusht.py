"""Real-data adapter: turn a LeRobot dataset into a (embeddings, quality) pool.

This is the bridge from the offline-verified curation engine to real robot
demonstrations. It is intentionally dependency-guarded: the core library and the
synthetic ablation run with only numpy + matplotlib, and ``lerobot`` (plus torch)
is required *only* here.

Per demonstration episode we compute:
  * quality proxy -- the per-episode task reward (``next.reward`` max; for
    sparse-reward datasets we fall back to the success rate / final reward).
  * embedding     -- a fixed-length temporal pooling of ``observation.state``
    and ``action`` (mean/std/min/max/first/last). This is deliberately simple;
    a visual/foundation-model encoder (DINOv2, R3M) is a documented upgrade.

Run:
    python -m robocurate.pusht --repo lerobot/pusht --limit 50
"""

from __future__ import annotations

import importlib

import numpy as np


def _import_lerobot_dataset():
    last_err = None
    for path in (
        "lerobot.datasets.lerobot_dataset",
        "lerobot.common.datasets.lerobot_dataset",
    ):
        try:
            return importlib.import_module(path).LeRobotDataset
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise ImportError(
        "Could not import LeRobotDataset. Install the real-data extras:\n"
        "    pip install 'lerobot' torch\n"
        f"(last import error: {last_err})"
    )


def _temporal_pool(seq: np.ndarray) -> np.ndarray:
    """Pool a (T, d) trajectory into a fixed 6*d feature vector."""
    seq = np.asarray(seq, dtype=float)
    if seq.ndim == 1:
        seq = seq[:, None]
    return np.concatenate(
        [seq.mean(0), seq.std(0), seq.min(0), seq.max(0), seq[0], seq[-1]]
    )


def compute_kinematic_features(actions: np.ndarray, states: np.ndarray | None = None):
    """Cheap, REWARD-FREE per-episode quality signals from a trajectory.

    Uses ONLY kinematics (never task reward / success / goal distance), so scoring
    a selection with these and evaluating against reward stays non-circular:

      * smoothness -- low action jerk  -> ``1 / (1 + mean|jerk|)``   (higher better)
      * efficiency -- path straightness ``net_displacement / path_length``
      * stability  -- fraction of non-reversing velocity steps
      * duration   -- number of frames (raw; normalize across the pool)
    """
    a = np.asarray(actions, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    T = len(a)

    jerk = np.linalg.norm(np.diff(a, n=2, axis=0), axis=1).mean() if T >= 3 else 0.0
    smoothness = 1.0 / (1.0 + float(jerk))

    path = np.asarray(states, dtype=float) if states is not None else np.cumsum(a, axis=0)
    if path.ndim == 1:
        path = path[:, None]
    vel = np.diff(path, axis=0)
    vnorm = np.linalg.norm(vel, axis=1)
    path_len = float(vnorm.sum())
    net = float(np.linalg.norm(path[-1] - path[0]))
    efficiency = net / path_len if path_len > 1e-9 else 0.0

    if len(vel) >= 2:
        u = vel / np.where(vnorm[:, None] < 1e-9, 1.0, vnorm[:, None])
        cos = (u[1:] * u[:-1]).sum(axis=1)
        stability = float(np.mean(cos >= 0))
    else:
        stability = 1.0

    return {
        "smoothness": smoothness,
        "efficiency": float(efficiency),
        "stability": stability,
        "duration": float(T),
    }


def load_lerobot_pool(
    repo_id: str = "lerobot/pusht",
    state_key: str = "observation.state",
    action_key: str = "action",
    reward_key: str = "next.reward",
    success_key: str = "next.success",
    limit: int | None = None,
    return_kinematics: bool = False,
):
    """Return ``(embeddings[N, d], raw_quality[N])`` for one demonstration pool.

    With ``return_kinematics=True`` returns a dict that additionally carries the
    per-episode REWARD-FREE proxy signals (smoothness/efficiency/stability/
    duration) and ``success`` -- everything the M1.5 / M2.5 proxy-q study needs on
    real data. ``limit`` caps episodes (quick smoke test); column names are
    configurable because they vary across LeRobot versions.
    """
    LeRobotDataset = _import_lerobot_dataset()
    ds = LeRobotDataset(repo_id)
    hf = ds.hf_dataset.with_format("numpy")

    ep = np.asarray(hf["episode_index"]).reshape(-1)
    state = np.asarray(hf[state_key])
    action = np.asarray(hf[action_key]) if action_key in hf.column_names else None
    reward = np.asarray(hf[reward_key]).reshape(-1) if reward_key in hf.column_names else None
    success = (
        np.asarray(hf[success_key]).reshape(-1).astype(float)
        if success_key in hf.column_names
        else None
    )

    episodes = np.unique(ep)
    if limit is not None:
        episodes = episodes[:limit]

    embeddings, quality, kin, succ = [], [], [], []
    for e in episodes:
        m = ep == e
        feats = [_temporal_pool(state[m])]
        if action is not None:
            feats.append(_temporal_pool(action[m]))
        embeddings.append(np.concatenate(feats))

        if reward is not None and np.ptp(reward[m]) > 0:
            quality.append(float(reward[m].max()))
        elif success is not None and np.ptp(success[m]) > 0:
            quality.append(float(success[m].mean()))
        elif reward is not None:
            quality.append(float(reward[m][-1]))
        elif return_kinematics:
            quality.append(float("nan"))   # reward-free dataset (e.g. ALOHA): proxy-q only
        else:
            raise KeyError(
                f"no usable reward/success column in {repo_id} "
                f"(columns: {hf.column_names})"
            )

        if return_kinematics:
            kin.append(
                compute_kinematic_features(
                    action[m] if action is not None else state[m], states=state[m]
                )
            )
            succ.append(float(success[m].max()) if success is not None else float("nan"))

    X = np.asarray(embeddings, dtype=float)
    q = np.asarray(quality, dtype=float)
    if not return_kinematics:
        return X, q
    return {
        "embeddings": X,
        "reward": q,
        "success": np.asarray(succ, dtype=float),
        "smoothness": np.array([k["smoothness"] for k in kin]),
        "efficiency": np.array([k["efficiency"] for k in kin]),
        "stability": np.array([k["stability"] for k in kin]),
        "duration": np.array([k["duration"] for k in kin]),
    }


def _main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="lerobot/pusht")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    X, q = load_lerobot_pool(args.repo, limit=args.limit)
    print(f"{args.repo}: {len(X)} episodes, embedding dim {X.shape[1]}, "
          f"quality range [{q.min():.3f}, {q.max():.3f}], mean {q.mean():.3f}")


if __name__ == "__main__":
    _main()
