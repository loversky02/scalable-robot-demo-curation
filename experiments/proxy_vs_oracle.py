"""M1.5 / M2.5: reward-free proxy-q curation study (synthetic OR real datasets).

Compares six selectors and evaluates every selection against the HELD-OUT task
reward/success -- a signal the proxy-q selectors never read (so a proxy-q win is
not circular):

    random
    quality-only (reward-q)   -- oracle, quality axis only
    quality-only (proxy-q)    -- deployable, quality axis only
    diversity-only            -- the failure-mode candidate
    DPP (proxy-q)             -- deployable quality x diversity
    DPP (reward-q)            -- oracle quality x diversity

Sources:
    synthetic  -- 3-tier expert/clumsy/noisy fixture (offline, $0)
    pusht      -- lerobot/pusht                       (needs lerobot+torch)
    aloha      -- lerobot/aloha_sim_insertion_human   (needs lerobot+torch)

For real datasets there are no ground-truth tiers, so the composition diagnostic
bins demos into reward terciles (used only for the plot, never for selection).

Usage:
    python experiments/proxy_vs_oracle.py --source synthetic
    python experiments/proxy_vs_oracle.py --source pusht
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robocurate import (  # noqa: E402
    diversity_spread,
    make_labeled_pool,
    normalize_quality,
    proxy_quality,
    select_diversity_only,
    select_dpp,
    select_quality_only,
    select_random,
)
from robocurate.synthetic import CLUMSY, EXPERT, NOISY  # noqa: E402

STYLE = {
    "random": ("tab:gray", "o"),
    "quality-only (reward-q)": ("tab:olive", "s"),
    "quality-only (proxy-q)": ("tab:orange", "s"),
    "diversity-only": ("tab:blue", "^"),
    "DPP (proxy-q, deployable)": ("tab:red", "*"),
    "DPP (reward-q, oracle)": ("tab:green", "*"),
}


def load_pool(source: str, limit: int | None):
    """Return a dict with embeddings, reward, success, kinematic proxies, and a
    (name, color, mask) group spec for the composition diagnostic."""
    if source == "synthetic":
        p = make_labeled_pool(seed=0)
        p["name"] = "synthetic 3-tier pool"
        p["groups"] = [
            ("expert", "tab:green", p["tier"] == EXPERT),
            ("clumsy", "gold", p["tier"] == CLUMSY),
            ("noisy", "tab:red", p["tier"] == NOISY),
        ]
        p["bad_label"] = "noisy%"
        p["bad_mask"] = p["tier"] == NOISY
        return p

    from robocurate.pusht import load_lerobot_pool  # guarded import

    repo = "lerobot/pusht" if source == "pusht" else "lerobot/aloha_sim_insertion_human"
    d = load_lerobot_pool(repo, limit=limit, return_kinematics=True)
    r = d["reward"]
    lo, hi = np.quantile(r, [1 / 3, 2 / 3])
    tercile = np.digitize(r, [lo, hi])  # 0 low, 1 mid, 2 high
    d["name"] = repo
    d["groups"] = [
        ("high-reward", "tab:green", tercile == 2),
        ("mid-reward", "gold", tercile == 1),
        ("low-reward", "tab:red", tercile == 0),
    ]
    d["bad_label"] = "low-reward%"
    d["bad_mask"] = tercile == 0
    return d


def run(source: str = "synthetic", k: int = 30, limit: int | None = None,
        out_dir: Path | None = None):
    out_dir = Path(out_dir or Path(__file__).resolve().parents[1] / "outputs")
    p = load_pool(source, limit)
    X, reward, success = p["embeddings"], p["reward"], p["success"]
    n = len(X)
    k = min(k, n - 1)

    q_reward = normalize_quality(reward)
    q_proxy = proxy_quality(p["smoothness"], p["efficiency"], p["stability"])
    corr = float(np.corrcoef(q_proxy, reward)[0, 1])

    rng = np.random.default_rng(0)
    picks = {
        "random": select_random(n, k, rng),
        "quality-only (reward-q)": select_quality_only(q_reward, k),
        "quality-only (proxy-q)": select_quality_only(q_proxy, k),
        "diversity-only": select_diversity_only(X, k),
        "DPP (proxy-q, deployable)": select_dpp(X, q_proxy, k),
        "DPP (reward-q, oracle)": select_dpp(X, q_reward, k),
    }

    M = {}
    for name, sel in picks.items():
        sel = np.asarray(sel)
        M[name] = dict(
            reward=float(np.mean(reward[sel])),
            success=float(np.nanmean(success[sel])) if success.size else float("nan"),
            diversity=diversity_spread(list(sel), X),
            bad=float(np.mean(p["bad_mask"][sel])),
            comp=[float(np.mean(mask[sel])) for _, _, mask in p["groups"]],
        )

    _print(M, p, k, corr)
    _plot(M, p, source, k, corr, out_dir)
    return M


def _print(M, p, k, corr):
    print(f"[data] {p['name']}: N={len(p['embeddings'])} demos, "
          f"embedding dim={p['embeddings'].shape[1]}")
    print(f"[check] proxy-vs-reward correlation = {corr:.3f} "
          f"(informative but NOT identical -> non-circular)\n")
    print(f"=== held-out metrics of selected demos (K={k}) ===")
    print(f"{'selector':30s} {'reward':>7s} {'success':>8s} {'divers':>7s} "
          f"{p['bad_label']:>11s}")
    for name, m in M.items():
        print(f"{name:30s} {m['reward']:7.3f} {m['success']:8.3f} "
              f"{m['diversity']:7.3f} {m['bad']:11.2f}")


def _plot(M, p, source, k, corr, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(M.keys())
    x = np.arange(len(names))
    short = [n.replace(" (", "\n(") for n in names]

    # 1) composition (stacked)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bottom = np.zeros(len(names))
    for gi, (gname, gcolor, _) in enumerate(p["groups"]):
        vals = np.array([M[n]["comp"][gi] for n in names])
        ax.bar(x, vals, bottom=bottom, label=gname, color=gcolor, edgecolor="k")
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel("fraction of selected demos")
    ax.set_title(f"Selected-set composition (K={k}) — {p['name']}")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / f"selector_composition_{source}.png", dpi=130)
    plt.close(fig)

    # 2) held-out reward bar
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    ax.bar(x, [M[n]["reward"] for n in names],
           color=[STYLE[n][0] for n in names], edgecolor="k", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel("held-out mean task reward of selected demos")
    ax.set_title(f"Reward-free proxy-q recovers high-reward demos "
                 f"(K={k}, corr={corr:.2f}) — {p['name']}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"reward_free_proxy_vs_reward_{source}.png", dpi=130)
    plt.close(fig)

    # 3) quality-diversity pareto
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, m in M.items():
        c, mk = STYLE[name]
        ax.scatter(m["diversity"], m["reward"], c=c, marker=mk, s=220,
                   edgecolor="k", label=name, zorder=3)
    ax.set_xlabel("diversity (mean pairwise distance in selected)  → more diverse")
    ax.set_ylabel("held-out mean task reward  → better")
    ax.set_title(f"Quality–diversity trade-off (K={k}) — {p['name']}")
    ax.legend(fontsize=8, loc="lower center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"quality_diversity_pareto_{source}.png", dpi=130)
    plt.close(fig)
    print(f"\n[out] wrote 3 figures to {out_dir}/ (suffix _{source})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic",
                    choices=["synthetic", "pusht", "aloha"])
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run(args.source, args.k, args.limit, Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
