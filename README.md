# scalable-robot-demo-curation

**Low-cost human demonstration collection and informativeness-aware curation for robot learning.**

> **Thesis.** Low-cost, non-expert demonstrations are *scalable but noisy*. This
> project studies whether **quality-weighted diversity** curation can recover
> useful demonstrations from such pools — reducing the human burden of data
> collection without sacrificing policy-relevant data.

**Research question:** *Can cheap, non-expert human demonstrations be made more
useful through automatic informativeness-aware curation?*

---

## The idea in one figure

Not all demonstrations are equally useful, and **the two obvious heuristics both
fail on low-cost data**:

- **Quality-only** (keep the highest-reward demos) → redundant: the best demos
  often look alike, so you re-collect the same skill.
- **Diversity-only** (maximize coverage / farthest-point) → *actively harmful* on
  non-expert data: **noise looks novel**, so a coverage-maximizer preferentially
  selects the erratic, low-quality demos.

We therefore define

```
informativeness  =  quality  ×  diversity
```

and realize it with a **quality-weighted Determinantal Point Process** (DPP),
using the Kulesza–Taskar decomposition:

```
L = diag(q) · S · diag(q)          q = per-demo task-quality proxy
                                    S = cosine similarity of demo embeddings
```

A DPP MAP selection under `L` keeps demonstrations that are **both high-quality
and mutually diverse** — exactly the trade-off scalable collection needs.

## Offline result (synthetic mixed-quality pool, `K=30`, $0, no GPU)

| method            | mean quality ↑ | diversity ↑ | % true experts ↑ |
|-------------------|:-------------:|:-----------:|:----------------:|
| random            | 0.637         | 0.894       | 0.71             |
| quality-only      | **0.961**     | 0.813       | 1.00             |
| diversity-only    | 0.214 ⚠️      | **1.026**   | **0.10** ⚠️       |
| **DPP (ours)**    | **0.901**     | 0.829       | **1.00**         |

Diversity-only collapses to the noise (quality `0.214`, only `10%` real experts).
The DPP recovers near-top quality **and** stays more diverse than quality-only —
the only method strong on *both* axes. See `outputs/`:

- `pareto_quality_diversity_synthetic.png` — the quality-vs-diversity trade-off
- `quality_vs_k_synthetic.png` — selected-set quality across budgets
- `expert_fraction_synthetic.png` — diversity-only chasing the noise

> ⚠️ The synthetic pool is a **controlled unit-test fixture**, not evidence for
> the claim. It verifies the method and reproduces the expected failure mode
> offline. The research signal comes from running the same pipeline on real
> `lerobot/pusht` and a second dataset (below).

## Quickstart

```bash
pip install -r requirements.txt          # numpy + matplotlib only

# 1) offline verification — runs the test suite + synthetic ablation ($0, no net)
bash scripts/verify_offline.sh

# 2) real data (needs the optional extras: pip install lerobot torch)
python experiments/run_ablation.py --source pusht
python experiments/run_ablation.py --source aloha   # cross-dataset hedge
```

## Method details

- **Quality proxy** (`robocurate/quality.py`) — per-episode task reward
  (`next.reward` / `next.success` in `lerobot/pusht`), min-max normalized to
  `[ε, 1]`. This is a *task-quality* proxy, **not** ground-truth informativeness.
- **Embedding** (`robocurate/pusht.py`) — fixed-length temporal pooling
  (mean/std/min/max/first/last) of `observation.state` and `action`. A visual /
  foundation-model encoder (DINOv2, R3M) is a documented upgrade.
- **Selection** (`robocurate/dpp.py`) — greedy DPP MAP (Chen et al. 2018) with an
  incremental Cholesky update; correctness checked against sub-determinants.

## Roadmap (milestones)

- [x] **M1 — curation engine + ablation** (this repo): quality×diversity DPP,
      offline-verified, real-data adapters for PushT and ALOHA-sim.
- [ ] **M2 — low-barrier collector**: keyboard/mouse PushT teleoperation so a
      *non-expert* can contribute demos, logged in LeRobot format with reward
      labels → feeds a real mixed-quality pool into M1.
- [ ] **M3 — downstream policy**: train Diffusion Policy on `curated-K` vs
      `random-K` (same budget, multiple seeds, error bars). Honest about null
      results — PushT is small and Diffusion Policy is robust.

## Honest limitations

- Single-task benchmarks (PushT; one ALOHA-sim task as a task-agnostic sanity
  check). We do **not** claim cross-platform generalization.
- `q` is a task-quality proxy; a high-reward demo can still be redundant (which
  is exactly why diversity is needed).
- Greedy DPP MAP is an approximation (no optimality guarantee).
- Simple pooled embeddings; richer encoders are future work.

## Relation to prior work

- **Data Scaling Laws in Imitation Learning** (Lin et al., ICLR 2025) — motivates
  efficient collection; we study *which* demos to keep.
- **Robot Data Curation with Mutual Information Estimators** (2025) — a parallel
  take on demonstration informativeness.
- **Universal Manipulation Interface / UMI** (Chi et al., 2024) — low-cost,
  robot-free collection; our M2 collector is the same spirit at PushT scale.

*Design note:* the quality×diversity DPP is a natural fit for demonstration
selection — the determinantal *diversity* term is the discrete cousin of
maintaining diverse solutions via Stein-variational repulsion, and here it is
coupled with a task-quality term so diversity never means "keep the noise".

## Repo layout

```
robocurate/         core library (offline: numpy only)
  dpp.py            quality×diversity kernel + greedy MAP
  selectors.py      random / quality-only / diversity-only / dpp
  quality.py        reward → quality proxy
  metrics.py        mean quality, diversity spread, coverage, expert fraction
  synthetic.py      controlled mixed-quality pool (test fixture)
  pusht.py          LeRobot adapter (PushT / ALOHA-sim) — guarded import
experiments/run_ablation.py   the M1 ablation driver (CSV + plots)
tests/              23 tests, offline, ~0.1s
outputs/            generated CSV + figures
```
