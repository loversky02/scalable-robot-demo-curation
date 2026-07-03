"""Milestone-1 ablation: random vs quality-only vs diversity-only vs DPP.

Runs the four curation strategies over a range of budgets K on a mixed-quality
demonstration pool and reports, for each:

    mean_quality     (want HIGH)   -- did we pick good demos?
    diversity_spread (want HIGH)   -- did we pick *different* demos?
    expert_fraction  (labeled only) -- diagnostic for "diversity-only grabs noise"

The headline figure is a quality-vs-diversity scatter: the DPP should sit in the
top-right (high quality AND high diversity), quality-only top-left (redundant),
diversity-only bottom-right (noisy). Informativeness needs BOTH axes.

Usage:
    python experiments/run_ablation.py                     # synthetic (offline, $0)
    python experiments/run_ablation.py --source pusht      # real lerobot/pusht
    python experiments/run_ablation.py --source aloha      # cross-dataset hedge
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robocurate import (  # noqa: E402
    coverage_radius,
    diversity_spread,
    expert_fraction,
    make_mixed_pool,
    normalize_quality,
    select_diversity_only,
    select_dpp,
    select_quality_only,
    select_random,
)

METHOD_STYLE = {
    "random": ("tab:gray", "o"),
    "quality_only": ("tab:orange", "s"),
    "diversity_only": ("tab:blue", "^"),
    "dpp": ("tab:red", "*"),
}


def load_pool(source: str, seed: int):
    """Return (embeddings, raw_quality, is_expert|None, name)."""
    if source == "synthetic":
        X, q_raw, is_expert = make_mixed_pool(seed=seed)
        return X, q_raw, is_expert, "synthetic mixed-quality pool"
    if source in ("pusht", "aloha"):
        from robocurate.pusht import load_lerobot_pool  # guarded import

        repo = "lerobot/pusht" if source == "pusht" else "lerobot/aloha_sim_insertion_human"
        X, q_raw = load_lerobot_pool(repo)
        return X, q_raw, None, repo
    raise ValueError(f"unknown source: {source}")


def run(source: str, ks, n_random_seeds: int, out_dir: Path):
    X, q_raw, is_expert, name = load_pool(source, seed=0)
    q = normalize_quality(q_raw)
    n = len(X)
    ks = [k for k in ks if k <= n]
    print(f"[data] {name}: N={n} demos, dim={X.shape[1]}, "
          f"quality range [{q.min():.3f}, {q.max():.3f}]")

    rows = []
    for k in ks:
        picks = {
            "quality_only": select_quality_only(q, k),
            "diversity_only": select_diversity_only(X, k),
            "dpp": select_dpp(X, q, k),
        }
        # random averaged over seeds
        rng = np.random.default_rng(1234)
        rand_mq, rand_sp, rand_ef, rand_cov = [], [], [], []
        for _ in range(n_random_seeds):
            r = select_random(n, k, rng)
            rand_mq.append(float(np.mean(q[r])))
            rand_sp.append(diversity_spread(r, X))
            rand_cov.append(coverage_radius(r, X))
            if is_expert is not None:
                rand_ef.append(expert_fraction(r, is_expert))
        picks_metrics = {
            "random": dict(
                mean_quality=float(np.mean(rand_mq)),
                diversity_spread=float(np.mean(rand_sp)),
                coverage_radius=float(np.mean(rand_cov)),
                expert_fraction=(float(np.mean(rand_ef)) if rand_ef else None),
            )
        }
        for m, sel in picks.items():
            picks_metrics[m] = dict(
                mean_quality=float(np.mean(q[sel])),
                diversity_spread=diversity_spread(sel, X),
                coverage_radius=coverage_radius(sel, X),
                expert_fraction=(
                    expert_fraction(sel, is_expert) if is_expert is not None else None
                ),
            )
        for m, mm in picks_metrics.items():
            rows.append(dict(k=k, method=m, **mm))

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out_dir / f"ablation_{source}.csv")
    _plot(rows, ks, source, name, is_expert is not None, out_dir)
    _print_summary(rows, ks)
    return rows


def _write_csv(rows, path: Path):
    fields = ["k", "method", "mean_quality", "diversity_spread",
              "coverage_radius", "expert_fraction"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[out] wrote {path}")


def _plot(rows, ks, source, name, labeled, out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(method, key):
        return [next(r[key] for r in rows if r["k"] == k and r["method"] == method)
                for k in ks]

    # Figure 1: quality vs diversity Pareto scatter at the largest K.
    k_star = ks[len(ks) // 2]
    fig, ax = plt.subplots(figsize=(6, 5))
    for m, (c, mk) in METHOD_STYLE.items():
        mq = next(r["mean_quality"] for r in rows if r["k"] == k_star and r["method"] == m)
        sp = next(r["diversity_spread"] for r in rows if r["k"] == k_star and r["method"] == m)
        ax.scatter(sp, mq, c=c, marker=mk, s=200, label=m, edgecolor="k", zorder=3)
    ax.set_xlabel("diversity (mean pairwise distance in selected)  → more diverse")
    ax.set_ylabel("mean task quality of selected  → better")
    ax.set_title(f"Quality vs diversity of curated set (K={k_star})\n{name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"pareto_quality_diversity_{source}.png", dpi=130)
    plt.close(fig)

    # Figure 2: mean quality vs K.
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, (c, mk) in METHOD_STYLE.items():
        ax.plot(ks, series(m, "mean_quality"), c=c, marker=mk, label=m)
    ax.set_xlabel("demonstration budget K")
    ax.set_ylabel("mean task quality of selected")
    ax.set_title(f"Selected-set quality vs budget\n{name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"quality_vs_k_{source}.png", dpi=130)
    plt.close(fig)

    # Figure 3: expert fraction vs K (labeled pools only) -- the failure mode.
    if labeled:
        fig, ax = plt.subplots(figsize=(6, 4))
        for m, (c, mk) in METHOD_STYLE.items():
            ax.plot(ks, series(m, "expert_fraction"), c=c, marker=mk, label=m)
        ax.set_xlabel("demonstration budget K")
        ax.set_ylabel("fraction of picks that are true experts")
        ax.set_title("Diversity-only chases the noise; DPP does not")
        ax.set_ylim(0, 1.02)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"expert_fraction_{source}.png", dpi=130)
        plt.close(fig)
    print(f"[out] wrote plots to {out_dir}/")


def _print_summary(rows, ks):
    k_star = ks[len(ks) // 2]
    print(f"\n=== summary at K={k_star} ===")
    print(f"{'method':16s} {'mean_quality':>13s} {'diversity':>10s} {'expert_frac':>12s}")
    for m in ("random", "quality_only", "diversity_only", "dpp"):
        r = next(r for r in rows if r["k"] == k_star and r["method"] == m)
        ef = "-" if r["expert_fraction"] is None else f"{r['expert_fraction']:.3f}"
        print(f"{m:16s} {r['mean_quality']:13.3f} {r['diversity_spread']:10.3f} {ef:>12s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic",
                    choices=["synthetic", "pusht", "aloha"])
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 30, 50])
    ap.add_argument("--random-seeds", type=int, default=20)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "outputs"))
    args = ap.parse_args()
    run(args.source, args.ks, args.random_seeds, Path(args.out))


if __name__ == "__main__":
    main()
