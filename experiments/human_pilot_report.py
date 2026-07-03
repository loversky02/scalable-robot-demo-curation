"""Human-pilot report — the collection-first evidence.

Takes a small real mouse-teleop pool (collect_pusht.py --mode human) and shows the
core story of the RA topic: a low-barrier interface lets non-experts contribute,
the resulting data is MIXED-QUALITY, and reward-free quality assessment + DPP
curation recover the useful part. Generates three figures:

    human_pilot_quality_distribution.png   reward vs proxy, coloured by operator
    human_pilot_selector_composition.png   what each selector keeps, by operator
    human_pilot_pareto.png                 reward vs diversity of the curated set

This is a PILOT to validate the pipeline, not a statistically powered user study.

Run:  python experiments/human_pilot_report.py --path data/human_pilot_pusht.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

np.seterr(all="ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robocurate import (  # noqa: E402
    diversity_spread,
    normalize_quality,
    proxy_quality,
    select_diversity_only,
    select_dpp,
    select_quality_only,
    select_random,
)
from robocurate.collected import load_and_merge  # noqa: E402

MODE_NAMES = ("careful", "normal", "rushed")           # operator_mode 0 / 1 / 2
MODE_COLORS = ("tab:green", "gold", "tab:red")


def _mode(m):
    return MODE_NAMES[m] if 0 <= m < len(MODE_NAMES) else f"mode{m}"


def run(paths, k):
    pool = load_and_merge(paths)
    X, reward, tier = pool["embeddings"], pool["reward"], pool["tier"]
    q_proxy = proxy_quality(pool["smoothness"], pool["efficiency"], pool["stability"])
    n = len(X)
    k = min(k, n - 1)

    print(f"[pilot] {len(paths)} file(s): {n} demos, K={k}")
    print(f"{'operator':10s} {'n':>3s} {'mean reward':>11s} {'mean smooth':>11s} {'mean dur':>9s}")
    for m in np.unique(tier):
        s = tier == m
        print(f"{_mode(m):10s} {int(s.sum()):3d} {reward[s].mean():11.3f} "
              f"{pool['smoothness'][s].mean():11.3f} {pool['duration'][s].mean():9.1f}")

    rng = np.random.default_rng(0)
    picks = {
        "random": select_random(n, k, rng),
        "quality-only": select_quality_only(q_proxy, k),
        "diversity-only": select_diversity_only(X, k),
        "DPP proxy-q": select_dpp(X, q_proxy, k),
    }
    for name, idx in picks.items():
        idx = np.asarray(idx)
        print(f"  {name:16s} selected mean reward {reward[idx].mean():.3f}, "
              f"careful-fraction {np.mean(tier[idx] == 0):.2f}")

    _plots(pool, reward, tier, q_proxy, picks, k)


def _plots(pool, reward, tier, q_proxy, picks, k):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(__file__).resolve().parents[1] / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    # 1) quality distribution — the "non-expert data is mixed-quality" evidence
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for m in np.unique(tier):
        s = tier == m
        ax.scatter(q_proxy[s], reward[s], c=MODE_COLORS[m % 3], s=60,
                   edgecolor="k", label=_mode(m))
    ax.set_xlabel("reward-free proxy quality (smoothness×efficiency×stability)")
    ax.set_ylabel("task reward")
    ax.set_title("Human pilot: low-cost non-expert data is mixed-quality")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out / "human_pilot_quality_distribution.png", dpi=130); plt.close(fig)

    # 2) selector composition — what curation keeps, by operator mode
    names = list(picks)
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    for m in np.unique(tier):
        vals = np.array([np.mean(tier[np.asarray(picks[nm])] == m) for nm in names])
        ax.bar(x, vals, bottom=bottom, color=MODE_COLORS[m % 3], edgecolor="k",
               label=_mode(m))
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("fraction of selected demos")
    ax.set_title(f"Human pilot: selected-set composition (K={k})")
    ax.legend(loc="lower left"); fig.tight_layout()
    fig.savefig(out / "human_pilot_selector_composition.png", dpi=130); plt.close(fig)

    # 3) quality-diversity pareto
    style = {"random": ("tab:gray", "o"), "quality-only": ("tab:orange", "s"),
             "diversity-only": ("tab:blue", "^"), "DPP proxy-q": ("tab:red", "*")}
    fig, ax = plt.subplots(figsize=(6, 4.7))
    for nm, idx in picks.items():
        idx = np.asarray(idx)
        c, mk = style[nm]
        ax.scatter(diversity_spread(list(idx), pool["embeddings"]),
                   float(np.mean(reward[idx])), c=c, marker=mk, s=210,
                   edgecolor="k", label=nm)
    ax.set_xlabel("diversity of selected  → more diverse")
    ax.set_ylabel("mean task reward of selected  → better")
    ax.set_title(f"Human pilot: quality–diversity trade-off (K={k})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out / "human_pilot_pareto.png", dpi=130); plt.close(fig)
    print(f"[out] wrote human_pilot_*.png to {out}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", default=["data/human_pilot_pusht.npz"])
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    run(args.paths, args.k)


if __name__ == "__main__":
    main()
