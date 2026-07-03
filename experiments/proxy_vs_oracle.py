"""M1.5 / M2.5: reward-free proxy-q curation study (synthetic OR real datasets).

Compares curation selectors and evaluates each selection against the HELD-OUT
task reward -- a signal the proxy-q selectors never read (so a proxy-q win is not
circular):

    random
    quality-only (reward-q)   -- oracle, quality axis only   (reward datasets only)
    quality-only (proxy-q)    -- deployable, quality axis only
    diversity-only            -- the failure-mode candidate
    DPP (proxy-q)             -- deployable quality x diversity
    DPP (reward-q)            -- oracle quality x diversity    (reward datasets only)

Sources:
    synthetic  -- 3-tier expert/clumsy/noisy fixture       (offline, $0)
    pusht      -- lerobot/pusht                             (has reward)
    aloha      -- lerobot/aloha_sim_insertion_human         (NO reward -> proxy-only,
                  task-agnostic + diversity validation of the pipeline)

Usage:
    python experiments/proxy_vs_oracle.py --source synthetic
    python experiments/proxy_vs_oracle.py --source pusht
    python experiments/proxy_vs_oracle.py --source aloha
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


def load_pool(source: str, limit: int | None, path: str | None = None):
    if source in ("synthetic", "collected"):
        if source == "synthetic":
            p = make_labeled_pool(seed=0)
            p["name"] = "synthetic 3-tier pool"
            p["reward_available"] = True
        else:
            from robocurate.collected import load_collected_pool
            p = load_collected_pool(path)
        tier = p["tier"]
        p["groups"] = [
            ("expert", "tab:green", tier == EXPERT),
            ("clumsy", "gold", tier == CLUMSY),
            ("noisy", "tab:red", tier == NOISY),
        ]
        p["bad_label"] = "noisy%"
        p["bad_mask"] = tier == NOISY
        return p

    from robocurate.pusht import load_lerobot_pool  # guarded import

    repo = "lerobot/pusht" if source == "pusht" else "lerobot/aloha_sim_insertion_human"
    d = load_lerobot_pool(repo, limit=limit, return_kinematics=True)
    d["name"] = repo
    d["reward_available"] = bool(np.isfinite(d["reward"]).any())
    return d


def run(source: str = "synthetic", k: int = 30, limit: int | None = None,
        path: str | None = None, out_dir: Path | None = None):
    out_dir = Path(out_dir or Path(__file__).resolve().parents[1] / "outputs")
    p = load_pool(source, limit, path)
    X, reward, success = p["embeddings"], p["reward"], p["success"]
    n = len(X)
    k = min(k, n - 1)
    reward_avail = p["reward_available"]

    q_proxy = proxy_quality(p["smoothness"], p["efficiency"], p["stability"])
    # evaluation score: the true reward when available, else the proxy itself
    # (for reward-free datasets this is a structural/diversity validation only).
    escore = reward if reward_avail else q_proxy
    p["escore_name"] = "held-out reward" if reward_avail else "proxy score (no reward in dataset)"

    # composition groups (terciles of the evaluation score) when not preset by the pool
    if "groups" not in p:
        lo, hi = np.nanquantile(escore, [1 / 3, 2 / 3])
        terc = np.digitize(escore, [lo, hi])
        hi_name = "high-reward" if reward_avail else "high-proxy"
        mid_name = "mid-reward" if reward_avail else "mid-proxy"
        lo_name = "low-reward" if reward_avail else "low-proxy"
        p["groups"] = [(hi_name, "tab:green", terc == 2),
                       (mid_name, "gold", terc == 1),
                       (lo_name, "tab:red", terc == 0)]
        p["bad_label"] = lo_name + "%"
        p["bad_mask"] = terc == 0

    corr = (float(np.corrcoef(q_proxy, reward)[0, 1])
            if reward_avail and np.std(q_proxy) > 1e-9 and np.nanstd(reward) > 1e-9
            else float("nan"))
    succ_ok = np.isfinite(success).any() and float(np.nanstd(success)) > 1e-9

    rng = np.random.default_rng(0)
    picks = {
        "random": select_random(n, k, rng),
        "quality-only (proxy-q)": select_quality_only(q_proxy, k),
        "diversity-only": select_diversity_only(X, k),
        "DPP (proxy-q, deployable)": select_dpp(X, q_proxy, k),
    }
    if reward_avail:
        q_reward = normalize_quality(reward)
        picks["quality-only (reward-q)"] = select_quality_only(q_reward, k)
        picks["DPP (reward-q, oracle)"] = select_dpp(X, q_reward, k)

    M = {}
    for name, sel in picks.items():
        sel = np.asarray(sel)
        M[name] = dict(
            escore=float(np.nanmean(escore[sel])),
            success=float(np.nanmean(success[sel])) if succ_ok else float("nan"),
            diversity=diversity_spread(list(sel), X),
            bad=float(np.mean(p["bad_mask"][sel])),
            comp=[float(np.mean(mask[sel])) for _, _, mask in p["groups"]],
        )

    _print(M, p, k, corr, succ_ok, reward_avail)
    _plot(M, p, source, k, corr)
    return M


def _print(M, p, k, corr, succ_ok, reward_avail):
    def f(v, w=8):
        return f"{'n/a':>{w}}" if not np.isfinite(v) else f"{v:{w}.3f}"

    print(f"[data] {p['name']}: N={len(p['embeddings'])} demos, "
          f"embedding dim={p['embeddings'].shape[1]}")
    if reward_avail:
        tag = ("weak on this homogeneous pool" if np.isfinite(corr) and abs(corr) < 0.3
               else "informative but NOT identical")
        print(f"[check] proxy-vs-reward correlation = {f(corr, 5).strip()} ({tag})")
    else:
        print("[note] this dataset has NO reward labels -> reward-q selectors and "
              "held-out reward evaluation are N/A; proxy-only structural validation.")
    if reward_avail and not succ_ok:
        print("[note] success labels absent/degenerate -> using reward only")
    score_hdr = "reward" if reward_avail else "proxy"
    print(f"\n=== held-out metrics of selected demos (K={k}) ===")
    print(f"{'selector':30s} {score_hdr:>7s} {'success':>8s} {'divers':>7s} "
          f"{p['bad_label']:>12s}")
    for name, m in M.items():
        print(f"{name:30s} {f(m['escore'],7)} {f(m['success'])} "
              f"{f(m['diversity'],7)} {m['bad']:12.2f}")


def _plot(M, p, source, k, corr):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(M.keys())
    x = np.arange(len(names))
    short = [nm.replace(" (", "\n(") for nm in names]
    ylab = p["escore_name"]

    # 1) composition (stacked)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bottom = np.zeros(len(names))
    for gi, (gname, gcolor, _) in enumerate(p["groups"]):
        vals = np.array([M[nm]["comp"][gi] for nm in names])
        ax.bar(x, vals, bottom=bottom, label=gname, color=gcolor, edgecolor="k")
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel("fraction of selected demos")
    ax.set_title(f"Selected-set composition (K={k}) — {p['name']}")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / f"selector_composition_{source}.png", dpi=130)
    plt.close(fig)

    # 2) evaluation-score bar
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    ax.bar(x, [M[nm]["escore"] for nm in names],
           color=[STYLE[nm][0] for nm in names], edgecolor="k", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel(ylab)
    title2 = (f"Reward-free proxy-q recovers high-reward demos (K={k}, corr={corr:.2f})"
              if np.isfinite(corr) else f"Selected-set score (K={k})")
    ax.set_title(f"{title2} — {p['name']}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"reward_free_proxy_vs_reward_{source}.png", dpi=130)
    plt.close(fig)

    # 3) score-diversity pareto
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, m in M.items():
        c, mk = STYLE[name]
        ax.scatter(m["diversity"], m["escore"], c=c, marker=mk, s=220,
                   edgecolor="k", label=name, zorder=3)
    ax.set_xlabel("diversity (mean pairwise distance in selected)  → more diverse")
    ax.set_ylabel(f"{ylab}  → better")
    ax.set_title(f"Quality–diversity trade-off (K={k}) — {p['name']}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"quality_diversity_pareto_{source}.png", dpi=130)
    plt.close(fig)
    print(f"\n[out] wrote 3 figures to {out_dir}/ (suffix _{source})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic",
                    choices=["synthetic", "pusht", "aloha", "collected"])
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--path", default="data/mixed_pusht.npz",
                    help="pool file for --source collected")
    args = ap.parse_args()
    run(args.source, args.k, args.limit, args.path)


if __name__ == "__main__":
    main()
