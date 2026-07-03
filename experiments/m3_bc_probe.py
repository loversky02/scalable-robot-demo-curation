"""M3-lite: a downstream imitation-policy probe (NOT a full Diffusion Policy).

Trains a small behaviour-cloning MLP on curated-K vs random-K vs diversity-K
subsets of a collected pool, then evaluates each policy by ROLLING OUT in the
real gym_pusht env. Because the selector is judged by downstream rollout reward
(a different quantity from the offline reward used to curate), this is the
cleanest *non-circular* evidence in the project.

Requires all demos to share one observation/action schema, so use a pool
collected entirely through gym_pusht (5-D state) -- e.g. human mouse-teleop demos
+ scripted low-skill demos. Do NOT use the build_mixed_pusht.py pool (its human
tier comes from lerobot/pusht with a 2-D observation).

Usage:
    python experiments/m3_bc_probe.py --paths data/human_pusht.npz data/collected_pusht.npz
    python experiments/m3_bc_probe.py --paths data/collected_pusht.npz   # pipeline smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

np.seterr(all="ignore")   # benign fp noise from BLAS on near-rank-deficient kernels

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robocurate import (  # noqa: E402
    normalize_quality,
    proxy_quality,
    select_diversity_only,
    select_dpp,
    select_random,
)
from robocurate.collected import load_and_merge  # noqa: E402


def build_bc(obs, act, epochs, seed):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    om, os_ = obs.mean(0), obs.std(0) + 1e-6
    X = torch.tensor((obs - om) / os_, dtype=torch.float32)
    Y = torch.tensor(act / 512.0, dtype=torch.float32)
    model = nn.Sequential(nn.Linear(obs.shape[1], 128), nn.ReLU(),
                          nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, act.shape[1]))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(X), Y)
        loss.backward()
        opt.step()
    return model, om, os_


def make_policy(model, om, os_):
    import torch

    def act(obs):
        xn = (np.asarray(obs, dtype=float) - om) / os_
        with torch.no_grad():
            y = model(torch.tensor(xn, dtype=torch.float32)).numpy()
        return np.clip(y * 512.0, 0.0, 512.0).astype(np.float32)

    return act


def rollout(policy, n_eps, max_steps, seed0=5000):
    import gymnasium as gym
    import gym_pusht  # noqa: F401

    env = gym.make("gym_pusht/PushT-v0", obs_type="state")
    rewards, succ = [], []
    for i in range(n_eps):
        obs, _ = env.reset(seed=seed0 + i)
        best, ok = 0.0, False
        for _ in range(max_steps):
            obs, r, term, trunc, info = env.step(policy(np.asarray(obs, dtype=float)))
            best = max(best, float(r))
            ok = ok or bool(info.get("is_success", False))
            if term or trunc:
                break
        rewards.append(best); succ.append(float(ok))
    env.close()
    return float(np.mean(rewards)), float(np.mean(succ))


def _pairs(states, actions, idx):
    obs, act = [], []
    for e in idx:
        s, a = np.asarray(states[e], float), np.asarray(actions[e], float)
        t = min(len(s) - 1, len(a))
        obs.append(s[:t]); act.append(a[:t])
    return np.concatenate(obs), np.concatenate(act)


def run(paths, k, epochs, eval_eps, max_steps, seeds):
    pool = load_and_merge(paths)
    emb, reward, states, actions = (pool["embeddings"], pool["reward"],
                                    pool["states"], pool["actions"])
    n = len(states)
    k = min(k, n - 1)
    q_proxy = proxy_quality(pool["smoothness"], pool["efficiency"], pool["stability"])
    q_reward = normalize_quality(reward)
    rng = np.random.default_rng(0)

    subsets = {
        "random-K": select_random(n, k, rng),
        "diversity-only-K": select_diversity_only(emb, k),
        "curated-K: DPP proxy-q": select_dpp(emb, q_proxy, k),
        "curated-K: DPP reward-q": select_dpp(emb, q_reward, k),
    }

    print(f"[data] {pool['name']}: N={n} demos, K={k}, "
          f"pool mean reward {reward.mean():.3f} ({seeds} seeds)")
    print(f"{'training subset':26s} {'subset rew':>10s} {'rollout reward (mean±std)':>26s}")
    rows = []
    for name, idx0 in subsets.items():
        rr, srs = [], []
        for s in range(seeds):
            # random baseline draws a fresh subset each seed (averaged over draws);
            # curated/diversity subsets are deterministic.
            idx = (select_random(n, k, np.random.default_rng(100 + s))
                   if name == "random-K" else idx0)
            obs, act = _pairs(states, actions, idx)
            model, om, os_ = build_bc(obs, act, epochs=epochs, seed=s)
            mr, _ = rollout(make_policy(model, om, os_), eval_eps, max_steps,
                            seed0=5000 + 137 * s)
            rr.append(mr); srs.append(float(np.mean(reward[np.asarray(idx)])))
        mean, std = float(np.mean(rr)), float(np.std(rr))
        rows.append((name, float(np.mean(srs)), mean, std))
        print(f"{name:26s} {np.mean(srs):10.3f} {mean:16.3f} ± {std:.3f}")

    _plot(rows, k, pool["name"])
    return rows


def _plot(rows, k, name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(__file__).resolve().parents[1] / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    labels = [r[0].replace(": ", "\n") for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.bar(range(len(rows)), [r[2] for r in rows], yerr=[r[3] for r in rows],
           capsize=5, color=["tab:gray", "tab:blue", "tab:red", "tab:green"],
           edgecolor="k")
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("rollout reward of trained BC policy (mean ± std)")
    ax.set_title(f"M3-lite: downstream policy vs training subset (K={k})\n{name}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "m3_bc_probe.png", dpi=130)
    plt.close(fig)
    print(f"[out] wrote {out}/m3_bc_probe.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", default=["data/collected_pusht.npz"])
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--eval-eps", type=int, default=15)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    run(args.paths, args.k, args.epochs, args.eval_eps, args.max_steps, args.seeds)


if __name__ == "__main__":
    main()
