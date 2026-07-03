# scalable-robot-demo-curation

**Low-cost human demonstration collection and informativeness-aware curation for robot learning.**

> **Thesis.** Low-cost, non-expert demonstrations are *scalable but noisy*. Naive
> diversity selection can over-select unusual-but-poor demonstrations, so
> informativeness should combine **reward-free quality proxies with diversity**.
> This project implements a **quality-weighted DPP** curation pipeline to recover
> useful demonstrations from mixed-quality human data — reducing the human burden
> of data collection without sacrificing policy-relevant data.

**Research question:** *Can cheap, non-expert human demonstrations be made more
useful through automatic informativeness-aware curation?*

## Status

| stage | what | state |
|-------|------|-------|
| **M1**   | quality×diversity DPP engine + ablation            | ✅ done, offline-verified (31 tests) |
| **M2.5** | reward-free proxy-q, non-circular study            | ✅ done, offline (synthetic) |
| **M1.5** | run on public datasets (PushT, ALOHA-sim)          | ✅ done — real results in `outputs/`, see below |
| **M2**   | mouse-teleop non-expert collection                 | ⬜ planned |
| **M3**   | downstream Diffusion Policy (curated vs random)    | ⬜ planned |

> ⚠️ The results below are a **controlled synthetic sanity check** — they verify
> the method and reproduce the expected failure modes, and are **not**
> robot-learning evidence. Public-dataset (PushT / ALOHA-sim) results are the
> validation step (M1.5).

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
(kinematic `smoothness × efficiency × stability`) and evaluate against the
**held-out** reward/success — signals the selector never saw. Six selectors on a
three-tier `expert / clumsy / noisy` pool (`K=30`):

| selector                       | held-out reward ↑ | success ↑ | diversity ↑ | noisy % ↓ |
|--------------------------------|:-----------------:|:---------:|:-----------:|:---------:|
| random                         | 0.618             | 0.70      | 0.808       | 0.23      |
| quality-only (reward-q)        | 0.989             | 1.00      | 0.638 ⚠️    | 0.00      |
| quality-only (proxy-q)         | 0.930             | 1.00      | 0.664 ⚠️    | 0.00      |
| diversity-only                 | 0.279 ⚠️          | 0.30 ⚠️   | **1.022**   | 0.67 ⚠️   |
| **DPP (proxy-q, deployable)**  | **0.797**         | **1.00**  | **0.793**   | **0.00**  |
| DPP (reward-q, oracle)         | 0.849             | 0.97      | 0.752       | 0.00      |

The **reward-free** proxy-DPP reaches 100% held-out success with 0% noisy picks
while staying diverse — it never reads the reward (proxy–reward correlation ≈ 0.94:
a *good* proxy, not the metric itself). Quality-only maxes reward but is the least
diverse (redundant); diversity-only is the most diverse but collapses to the noise
(67% noisy, 30% success). This mirrors the **deployable** setting: real low-cost,
non-expert collection often has no reward function at all. Figures in `outputs/`
(regenerate with `python experiments/proxy_vs_oracle.py`):
`selector_composition_synthetic.png`, `reward_free_proxy_vs_reward_synthetic.png`,
`quality_diversity_pareto_synthetic.png`.

> Note: `reward-q` is an oracle upper bound; `proxy-q` is the deployable setting.
> The cleanest non-circular evidence of all is **M3** (downstream policy success),
> which measures a different quantity from the reward used to select.

## Real-data findings (M1.5): PushT + ALOHA-sim

The same pipeline runs unchanged on two public LeRobot datasets, on CPU.

**`lerobot/pusht`** (206 human demos, 24-d embedding). Reward-based curation *does*
pick measurably higher-reward demos than random — but the margin is small because
PushT ships only expert teleop (per-episode reward 0.81–0.95, a narrow band):

| selector (K=30)          | held-out reward ↑ | diversity ↑ | low-reward % ↓ |
|--------------------------|:-----------------:|:-----------:|:--------------:|
| random                   | 0.893             | 0.055       | 0.27           |
| quality-only (reward-q)  | **0.928**         | 0.057       | 0.00           |
| DPP (reward-q)           | 0.917             | 0.069       | 0.00           |
| DPP (proxy-q)            | 0.889             | 0.082       | 0.37           |

The **reward-free proxy is uninformative here** (proxy–reward correlation ≈ 0.09):
homogeneous expert demos give kinematics little skill variance to capture.
`next.success` is absent in this dataset → reward only.

**`lerobot/aloha_sim_insertion_human`** (50 demos, 14-DoF, 168-d embedding) ships
**no reward labels at all** — the pure deployable setting. We can only validate that
the pipeline is *task-agnostic* (loads, embeds, computes reward-free kinematics, and
curates); demos are near-identical (diversity ≈ 0.03), so little separates them.

**Takeaway — this is the point.** On clean, homogeneous *expert* datasets, curation
has little to gain and reward-free proxies are uninformative. The mixed-quality,
skill-heterogeneous regime the method targets does **not** exist in public
demonstration datasets — it has to be *collected* (M2). Running on real data thus
empirically motivates the low-barrier non-expert collection step — an honest signal
the synthetic sanity check cannot provide. Figures: `outputs/*_pusht.png`,
`outputs/*_aloha.png`.

## Quickstart

```bash
pip install -r requirements.txt          # numpy + matplotlib only

# 1) offline verification — runs the test suite + synthetic ablation ($0, no net)
bash scripts/verify_offline.sh

# 2) real data (needs the optional extras: pip install lerobot torch)
python experiments/run_ablation.py --source pusht          # M1.5: 4-selector ablation
python experiments/proxy_vs_oracle.py --source pusht        # M1.5: 6-selector proxy-q study
python experiments/proxy_vs_oracle.py --source aloha        # cross-dataset hedge
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
  synthetic.py      controlled pools: M1 mixed pool + 3-tier labeled pool
  pusht.py          LeRobot adapter + reward-free kinematic features (guarded)
experiments/
  run_ablation.py       the M1 ablation driver (CSV + plots)
  proxy_vs_oracle.py    the M2.5 6-selector non-circular study (3 figures)
tests/              31 tests, offline, ~0.1s
outputs/            generated CSV + figures
```
