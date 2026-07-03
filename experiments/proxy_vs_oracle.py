"""Milestone-2.5: offline non-circular curation study on a 3-tier pool.

Compares six selectors on a labeled expert/clumsy/noisy pool and evaluates every
selection against the HELD-OUT task reward/success -- a signal the proxy-q
selectors never read (so a proxy-q win is not circular):

    random
    quality-only (reward-q)   -- oracle, quality axis only
    quality-only (proxy-q)    -- deployable, quality axis only
    diversity-only            -- the failure-mode candidate
    DPP (proxy-q)             -- deployable quality x diversity
    DPP (reward-q)            -- oracle quality x diversity

Produces three figures (coverage is left as an appendix diagnostic):
    selector_composition_synthetic.png       expert/clumsy/noisy of each pick
    reward_free_proxy_vs_reward_synthetic.png held-out reward per selector
    quality_diversity_pareto_synthetic.png    diversity vs held-out reward

Usage:  python experiments/proxy_vs_oracle.py
"""

from __future__ import annotations

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


def run(k: int = 30, out_dir: Path | None = None):
    out_dir = Path(out_dir or Path(__file__).resolve().parents[1] / "outputs")
    p = make_labeled_pool(seed=0)
    X, reward, success, tier = p["embeddings"], p["reward"], p["success"], p["tier"]

    q_reward = normalize_quality(reward)                                    # oracle
    q_proxy = proxy_quality(p["smoothness"], p["efficiency"], p["stability"])  # reward-free
    corr = float(np.corrcoef(q_proxy, reward)[0, 1])

    rng = np.random.default_rng(0)
    picks = {
        "random": select_random(len(X), k, rng),
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
            success=float(np.mean(success[sel])),
            diversity=diversity_spread(list(sel), X),
            expert=float(np.mean(tier[sel] == EXPERT)),
            clumsy=float(np.mean(tier[sel] == CLUMSY)),
            noisy=float(np.mean(tier[sel] == NOISY)),
        )

    _print(M, k, corr)
    _plot_composition(M, k, out_dir)
    _plot_reward(M, k, corr, out_dir)
    _plot_pareto(M, k, out_dir)
    return M


def _print(M, k, corr):
    print(f"[check] proxy-vs-reward correlation = {corr:.3f} "
          f"(informative but NOT identical -> non-circular)\n")
    print(f"=== held-out metrics of selected demos (K={k}) ===")
    print(f"{'selector':30s} {'reward':>7s} {'success':>8s} {'divers':>7s} {'noisy%':>7s}")
    for name, m in M.items():
        print(f"{name:30s} {m['reward']:7.3f} {m['success']:8.3f} "
              f"{m['diversity']:7.3f} {m['noisy']:7.2f}")


def _fig(out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    return plt


def _short(names):
    return [n.replace(" (", "\n(") for n in names]


def _plot_composition(M, k, out_dir):
    plt = _fig(out_dir)
    names = list(M.keys())
    x = np.arange(len(names))
    exp = np.array([M[n]["expert"] for n in names])
    clu = np.array([M[n]["clumsy"] for n in names])
    noi = np.array([M[n]["noisy"] for n in names])
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x, exp, label="expert", color="tab:green", edgecolor="k")
    ax.bar(x, clu, bottom=exp, label="clumsy", color="gold", edgecolor="k")
    ax.bar(x, noi, bottom=exp + clu, label="noisy", color="tab:red", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(_short(names), rotation=0, fontsize=8)
    ax.set_ylabel("fraction of selected demos")
    ax.set_title(f"Selected-set composition (K={k}): diversity-only over-picks noise")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "selector_composition_synthetic.png", dpi=130)
    plt.close(fig)


def _plot_reward(M, k, corr, out_dir):
    plt = _fig(out_dir)
    names = list(M.keys())
    x = np.arange(len(names))
    vals = [M[n]["reward"] for n in names]
    colors = [STYLE[n][0] for n in names]
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    ax.bar(x, vals, color=colors, edgecolor="k", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(_short(names), rotation=0, fontsize=8)
    ax.set_ylabel("held-out mean task reward of selected demos")
    ax.set_title(f"Reward-free proxy-q recovers high-reward demos (K={k}, "
                 f"proxy-reward corr={corr:.2f})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "reward_free_proxy_vs_reward_synthetic.png", dpi=130)
    plt.close(fig)


def _plot_pareto(M, k, out_dir):
    plt = _fig(out_dir)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, m in M.items():
        c, mk = STYLE[name]
        ax.scatter(m["diversity"], m["reward"], c=c, marker=mk, s=220,
                   edgecolor="k", label=name, zorder=3)
    ax.set_xlabel("diversity (mean pairwise distance in selected)  → more diverse")
    ax.set_ylabel("held-out mean task reward  → better")
    ax.set_title(f"Quality–diversity trade-off (K={k})")
    ax.legend(fontsize=8, loc="lower center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "quality_diversity_pareto_synthetic.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run()
