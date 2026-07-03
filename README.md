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
> offline. The research signal comes from running the same pipeline on the
> public `lerobot/pusht` and `lerobot/aloha_sim_insertion_human` datasets (below).

### Avoiding reward circularity: a reward-free proxy still recovers good demos

Scoring the DPP with `q = reward` and then judging it by *selected-set reward*
would be circular. So we also score with a **reward-free** proxy quality
(kinematic `smoothness × efficiency`) and evaluate against the **held-out**
reward — a signal the selector never saw:

| method (K=30)                 | held-out mean reward ↑ | % experts ↑ |
|-------------------------------|:----------------------:|:-----------:|
| random                        | 0.645                  | 0.67        |
| diversity-only                | 0.226 ⚠️               | 0.13 ⚠️      |
| **DPP (proxy-q, deployable)** | **0.840**              | 0.90        |
| DPP (reward-q, oracle)        | 0.947                  | 1.00        |

The reward-free proxy recovers most of the oracle's advantage while never
touching the reward (proxy–reward correlation ≈ 0.91 — a *good* proxy, not the
metric itself). This mirrors the **deployable** setting: real low-cost, non-expert
collection often has no reward function at all. Regenerate with
`python experiments/proxy_vs_oracle.py` → `outputs/proxy_vs_oracle_synthetic.png`.

> Note: `reward-q` is an oracle upper bound; `proxy-q` is the deployable setting.
> The cleanest non-circular evidence of all is **M3** (downstream policy success),
> which measures a different quantity from the reward used to select.

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

- **Quality proxy** (`robocurate/quality.py`) — two settings: `normalize_quality`
  uses the per-episode task reward (`next.reward` / `next.success`) as an *oracle*
  signal; `proxy_quality` combines **reward-free** kinematic features (smoothness,
  action efficiency, duration) for the *deployable* setting. Neither is
  ground-truth informativeness — a high-reward demo can still be redundant, which
  is why diversity is needed.
- **Embedding** (`robocurate/pusht.py`) — fixed-length temporal pooling
  (mean/std/min/max/first/last) of `observation.state` and `action`. A visual /
  foundation-model encoder (DINOv2, R3M) is a documented upgrade.
- **Selection** (`robocurate/dpp.py`) — greedy DPP MAP (Chen et al. 2018) with an
  incremental Cholesky update; correctness checked against sub-determinants.

## Roadmap (milestones)

- [x] **M1 — curation engine + ablation** (this repo): quality×diversity DPP,
      offline-verified; guarded adapters for public robot-learning datasets.
- [x] **M2.5 engine — non-circular check** (offline, done here): reward-free proxy
      quality + oracle/deployable comparison, evaluated against a held-out reward
      the selector never used. To be re-applied to the M2 pool.
- [ ] **M1.5 — public-dataset validation**: run the ablation on `lerobot/pusht`
      and `lerobot/aloha_sim_insertion_human` (needs `lerobot`+`torch`, no GPU).
      Report selected-set reward/success, diversity, and composition; coverage is
      a *diagnostic only*, not the headline.
- [ ] **M2 — low-barrier collection**: mouse teleoperation of PushT (its native
      control) so a *non-expert* can contribute demos. Self-collect an
      `expert / clumsy / noisy` pool — *a simulated non-expert pool, not a user
      study* — logged in LeRobot format with `reward, success, smoothness,
      duration, operator_mode`.
- [ ] **M3 — downstream policy**: train Diffusion Policy on the *mixed-quality*
      pool, `random-K` vs `diversity-only-K` vs `DPP-K` (K=10/25/50, same budget,
      multiple seeds, error bars; RunPod preferred near a deadline). Honest about
      null results on clean pools — curation earns its keep on noisy ones.

## Honest limitations

- Single-task benchmarks (PushT; one ALOHA-sim task as a task-agnostic sanity
  check). We do **not** claim cross-platform generalization.
- `q` is a task-quality proxy; a high-reward demo can still be redundant (which
  is exactly why diversity is needed).
- In the synthetic fixture a latent skill drives *both* reward and the kinematic
  proxy, so their correlation is high by construction. Genuine reward/kinematic
  decoupling is a claim to validate on real M2 data, not on the fixture.
- Greedy DPP MAP is an approximation (no optimality guarantee).
- Simple pooled embeddings; richer encoders (DINOv2, R3M) are future work.

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
  quality.py        reward proxy (oracle) + reward-free kinematic proxy
  metrics.py        mean quality, diversity spread, coverage, expert fraction
  synthetic.py      controlled mixed-quality pools (test fixtures)
  pusht.py          LeRobot adapter (PushT / ALOHA-sim) — guarded import
experiments/
  run_ablation.py       the M1 ablation driver (CSV + plots)
  proxy_vs_oracle.py    the M2.5 non-circular check (reward-free proxy)
tests/              27 tests, offline, ~0.1s
outputs/            generated CSV + figures
```
