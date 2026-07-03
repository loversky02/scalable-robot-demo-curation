"""Milestone-2.5: non-circular curation check on a labeled mixed-quality pool.

Scores demonstrations with a REWARD-FREE proxy quality (kinematic
smoothness x action efficiency), selects with the quality x diversity DPP, then
evaluates the picks against the HELD-OUT task reward. Because the proxy never
sees the reward, a win is not circular.

  * reward-q  -> oracle upper bound (uses the true reward signal)
  * proxy-q   -> deployable setting  (real non-expert collection has no reward fn)

Usage:
    python experiments/proxy_vs_oracle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robocurate import (  # noqa: E402
    expert_fraction,
    make_labeled_pool,
    normalize_quality,
    proxy_quality,
    select_diversity_only,
    select_dpp,
    select_random,
)


def run(k: int = 30, out_dir: Path | None = None):
    out_dir = Path(out_dir or Path(__file__).resolve().parents[1] / "outputs")
    p = make_labeled_pool(seed=0)
    X, reward = p["embeddings"], p["reward"]

    q_reward = normalize_quality(reward)                        # oracle
    q_proxy = proxy_quality(p["smoothness"], p["efficiency"])   # deployable, reward-free

    corr = float(np.corrcoef(q_proxy, reward)[0, 1])
    print(f"[check] proxy-vs-reward correlation = {corr:.3f} "
          f"(informative but NOT identical -> non-circular)")

    rng = np.random.default_rng(0)
    picks = {
        "random": select_random(len(X), k, rng),
        "diversity_only": select_diversity_only(X, k),
        "DPP (proxy-q, deployable)": select_dpp(X, q_proxy, k),
        "DPP (reward-q, oracle)": select_dpp(X, q_reward, k),
    }

    rows = []
    print(f"\n=== held-out reward of selected demos (K={k}) ===")
    print(f"{'method':28s} {'mean_reward':>11s} {'expert_frac':>12s}")
    for m, sel in picks.items():
        mr = float(np.mean(reward[sel]))
        ef = expert_fraction(sel, p["is_expert"])
        rows.append((m, mr, ef))
        print(f"{m:28s} {mr:11.3f} {ef:12.3f}")

    _plot(rows, k, corr, out_dir)
    return rows


def _plot(rows, k, corr, out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = ["tab:gray", "tab:blue", "tab:red", "tab:green"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(range(len(names)), vals, color=colors, edgecolor="k", zorder=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylabel("held-out mean task reward of selected demos")
    ax.set_title(
        f"Non-circular check (K={k}): a reward-free proxy still recovers "
        f"high-reward demos\n(proxy-vs-reward corr = {corr:.2f})"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "proxy_vs_oracle_synthetic.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[out] wrote {path}")


if __name__ == "__main__":
    run()
