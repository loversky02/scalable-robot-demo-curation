# scalable-robot-demo-curation

**A low-cost human demonstration collection + curation prototype for robot learning.**

> **Thesis.** Expert teleoperation is expensive, so *scaling* demonstration data
> means letting **non-experts** contribute through low-barrier interfaces — but that
> data is **mixed-quality and noisy**. This is a small end-to-end pipeline for that
> regime: a **low-cost mouse-teleoperation collector** for PushT, plus **reward-free
> quality assessment** and **quality-weighted DPP curation** to recover the useful
> demonstrations from a noisy non-expert pool. Curation is the quality-control
> *module* — the focus is the low-barrier **collection → recovery** pipeline.

**Research question:** *When low-barrier interfaces let non-experts contribute
demonstrations, can reward-free quality assessment and quality-weighted diversity
recover useful data from noisy human inputs?*

## Status

| stage | what | state |
|-------|------|-------|
| **M1**   | quality×diversity DPP engine + ablation            | ✅ done, offline-verified (31 tests) |
| **M2.5** | reward-free proxy-q, non-circular study            | ✅ done, offline (synthetic) |
| **M1.5** | run on public datasets (PushT, ALOHA-sim)          | ✅ done — real results in `outputs/`, see below |
| **M2**   | low-barrier collection + real-env mixed pool       | ✅ collector + real-env result (below); human mode ready |
| **M3-lite** | downstream BC-MLP probe (gym_pusht rollout)      | ✅ done — non-circular result (below): quality×diversity ≈ 2.2× random |
| **M3-full** | downstream Diffusion Policy                     | ⬜ future (needs image-obs PushT + GPU) |
| **Pilot** | small **real** mouse-teleop human collection      | ⏳ collector + report ready — run after collecting ~15 demos |

> ⚠️ The synthetic results are a **controlled sanity check** (they verify the method
> and reproduce the failure modes); the public-dataset, real-env, and downstream
> results are the evidence. The **human pilot** is the collection-first validation
> and is the one remaining step (needs ~15 min of mouse demos).

---

## The pipeline (collection-first)

1. **Motivation** — expert teleoperation needs skilled operators, special hardware,
   and time. Non-expert collection is scalable but **noisy**.
2. **System** — a **mouse-teleoperation** interface for PushT
   (`experiments/collect_pusht.py --mode human`): the agent follows the cursor, so a
   non-expert contributes a demo with a mouse. Each episode logs
   `reward, success, duration, operator_mode` (careful / normal / rushed) and, at
   load time, reward-free kinematics (smoothness / efficiency / stability).
3. **Data issue** — the pool is **mixed-quality** (careful vs rushed attempts differ).
4. **Quality control** — a **reward-free proxy** scores each demo from kinematics
   only (deployable when there is no reward function) — the same signal family as
   [DQAF (Closing the Loop in Teleoperation, 2026)](https://arxiv.org/abs/2605.26349).
5. **Curation** — a **quality-weighted DPP** keeps demos that are both good *and*
   diverse (the module below).
6. **Evidence** — public datasets (M1.5), a real-env mixed pool (M2), and a
   downstream BC rollout check (M3-lite).
7. **Human pilot** — a small real mouse-teleop collection validates it end-to-end.

---

## Curation module: informativeness = quality × diversity

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

## M2 result: curation pays off on a real-env mixed-quality pool

Public expert datasets are too clean (M1.5), so we *build* the mixed-quality regime
the method targets, through the real `gym_pusht` simulator.
`experiments/collect_pusht.py` has a **mouse-teleoperation** mode (the low-barrier
non-expert interface) and a **scripted** mode. Naive scripted heuristics can't push
PushT well (it needs a learned policy), so `experiments/build_mixed_pusht.py` forms
an honest pool with a large *real* reward gap: **real human expert demos**
(`lerobot/pusht`, reward ≈ 0.90) + **scripted low-skill** demos (real gym_pusht,
reward ≈ 0.15). On this 90-demo pool the thesis holds with real simulator reward:

| selector (K=30)              | held-out reward ↑ | diversity ↑ | noisy % ↓ |
|------------------------------|:-----------------:|:-----------:|:---------:|
| random                       | 0.439             | 1.019       | 0.37      |
| diversity-only               | 0.439 ⚠️          | 1.018       | 0.33 ⚠️   |
| quality-only (reward-q)      | 0.908             | 0.701       | 0.00      |
| **DPP (proxy-q, deployable)**| **0.830**         | **0.826**   | **0.00**  |
| DPP (reward-q, oracle)       | 0.759             | 0.977       | 0.00      |

Here the reward-free proxy is **informative** (proxy–reward correlation ≈ 0.90, vs
0.09 on clean PushT): heterogeneous skill gives kinematics something to capture.
Diversity-only gains nothing over random (it chases the scattered low-skill demos);
quality-only recovers reward but is least diverse; **DPP — even reward-free
proxy-q — is the only method high on both axes**, with 0% noisy picks. Figures:
`outputs/*_collected.png`.

> Honesty: the expert tier is real human teleop; the low-skill tiers are scripted
> (a *simulated* non-expert pool). The `--mode human` collector gathers genuine
> non-expert demos — swap them in and re-run the same study unchanged.

## M3-lite result: curation helps the *downstream policy* (non-circular)

The offline results (M2) show curation *picks* good, diverse demos. M3 asks the
real question: does a policy **trained** on curated data actually perform better?
`experiments/m3_bc_probe.py` trains a small behaviour-cloning MLP on curated-K /
random-K / diversity-K subsets and scores each by **rolling out in gym_pusht**.
Rollout reward is a different quantity from the offline reward used to curate, so
this is the cleanest **non-circular** evidence in the project.

On a 90-demo pool of scripted PushT attempts (real gym_pusht reward, naturally
spread 0.0–0.81 — the "most attempts miss" structure of low-cost collection),
K=30, 5 training seeds (random re-drawn per seed):

| training subset (K=30)          | subset reward | rollout reward (mean ± std) |
|---------------------------------|:-------------:|:---------------------------:|
| random-K                        | 0.166         | 0.113 ± 0.028               |
| diversity-only-K                | 0.234         | 0.105 ± 0.016               |
| reward-only-K (DPP reward-q)    | **0.401**     | 0.107 ± 0.019               |
| **quality×diversity (DPP proxy-q)** | 0.178     | **0.247 ± 0.032**           |

The finding is sharp and **non-obvious**: selecting the *highest-reward* demos
(reward-q, subset reward 0.40) gives **no** downstream benefit (0.107 ≈ random) —
they are redundant, so the BC policy overfits. Diversity alone also fails (0.105).
Only the **reward-free quality×diversity** selection trains a policy that
generalizes — ≈ 2.2× random, gap well beyond seed variance. *Diversity is what the
downstream policy needs.* Figure: `outputs/m3_bc_probe.png`.

> Honesty: a lightweight BC-MLP probe on *scripted* attempts (PushT is too hard to
> script into clean experts, so we use the natural reward spread of many attempts).
> Rollout *success* is 0 for all (imperfect demos) — the metric is rollout coverage
> reward. Real human demos (`--mode human`) and a stronger policy are future work;
> the pipeline is unchanged.

## Human pilot — the collection-first validation

Everything above uses public datasets or scripted attempts. The one step that makes
this touch *human* demonstrations is a **small real pilot** through the mouse
collector — the standard validation in this space (RoboCrowd situates a collector in
public; DQAF validates with a 3-novice pilot), **not** a large user study. Collect
~15 demos at a couple of effort levels, then one command builds the report:

```bash
# on your Mac (needs a display); one run per operator mode:
python experiments/collect_pusht.py --mode human --operator-mode careful --n 8 --out data/pilot_careful.npz
python experiments/collect_pusht.py --mode human --operator-mode rushed  --n 8 --out data/pilot_rushed.npz
# quality distribution + selector composition + quality-diversity pareto:
python experiments/human_pilot_report.py --paths data/pilot_careful.npz data/pilot_rushed.npz
```

Outputs (`outputs/`): `human_pilot_quality_distribution.png` (non-expert data is
mixed-quality), `human_pilot_selector_composition.png` (what curation keeps, per
operator), `human_pilot_pareto.png` (quality vs diversity of the curated set).

> Honesty: a **pilot to validate the collection pipeline, not a statistically
> powered user study**.

## Quickstart

```bash
pip install -r requirements.txt          # numpy + matplotlib only

# 1) offline verification — runs the test suite + synthetic ablation ($0, no net)
bash scripts/verify_offline.sh

# 2) real data (needs the optional extras: pip install lerobot torch)
python experiments/run_ablation.py --source pusht          # M1.5: 4-selector ablation
python experiments/proxy_vs_oracle.py --source pusht        # M1.5: 6-selector proxy-q study
python experiments/proxy_vs_oracle.py --source aloha        # cross-dataset hedge

# 3) M2 -- build a real-env mixed-quality pool and run the study on it
#    (collector extras: pip install gym_pusht pygame "pymunk<7")
python experiments/collect_pusht.py --mode scripted --per-tier 25   # real gym_pusht low-skill demos
python experiments/build_mixed_pusht.py --n-expert 40               # + real human experts -> mixed pool
python experiments/proxy_vs_oracle.py --source collected --path data/mixed_pusht.npz
# python experiments/collect_pusht.py --mode human --operator-mode expert  # mouse teleop (local display)

# 4) M3-lite -- downstream BC-MLP probe, evaluated by gym_pusht rollout
python experiments/m3_bc_probe.py --paths data/collected_pusht.npz             # pipeline smoke test
# with real human demos: --paths data/human_pusht.npz data/collected_pusht.npz
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
- [x] **M1.5 — public-dataset validation** (done): ran on `lerobot/pusht` (206
      demos) and `lerobot/aloha_sim_insertion_human` (50 demos, no reward column).
      Finding: public data is homogeneous expert, so curation gains are small and
      the reward-free proxy is uninformative — which empirically motivates M2.
      See **Real-data findings** above; figures in `outputs/*_pusht.png`, `*_aloha.png`.
- [x] **M2 — low-barrier collection + real-env mixed pool** (done): `gym_pusht`
      mouse-teleop collector (`--mode human`) + scripted tiers; built a real
      mixed-quality pool (human experts + scripted low-skill) where curation pays
      off and the reward-free proxy becomes informative (see *M2 result* above).
- [x] **M3-lite — downstream BC probe** (done): a small BC-MLP trained on
      `curated-K` vs `random-K` vs `diversity-K` subsets, scored by **gym_pusht
      rollout** (`experiments/m3_bc_probe.py`) — non-circular (rollout reward ≠ the
      reward used to curate). Result: reward-free quality×diversity curation ≈ 2.2×
      random downstream, while reward-only and diversity-only do not beat random
      (see *M3-lite result*). Swap in real human demos (`--mode human`) to
      strengthen it.
- [ ] **M3-full — Diffusion Policy** (future): needs image-observation PushT + GPU;
      the state-only `lerobot/pusht` (2-D agent obs) is not policy-trainable alone.

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

**Collection (the topic's core).**
- **UMI / FastUMI / DexWild** (Chi et al. 2024; 2025) — low-cost, robot-free,
  portable interfaces for non-expert collection; our mouse-teleop collector is the
  same spirit at PushT scale.
- **RoboCrowd / RoboTurk** — crowdsourcing demonstrations from non-experts, and the
  incentive/quality problems that come with it.

**Quality control + curation (this repo's module).**
- **DQAF — Closing the Loop in Teleoperation** (2026) — scores teleop episodes from
  smoothness / stalls / kinematic limits; our reward-free proxy uses the same signal
  family, for *selection* rather than operator feedback.
- **Robomimic** — demonstrator proficiency and diversity matter as much as raw scale.
- **Data Scaling Laws** (Lin et al., ICLR 2025) and **Robot Data Curation with MI
  Estimators** (2025) — efficient collection / demonstration informativeness.

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
  collected.py      loader for an M2 collected/built mixed-quality pool (.npz)
experiments/
  run_ablation.py       the M1 ablation driver (CSV + plots)
  proxy_vs_oracle.py    the 6-selector study (synthetic / pusht / aloha / collected)
  collect_pusht.py      collector: mouse-teleop (human) + scripted, real gym_pusht
  human_pilot_report.py human pilot: quality distribution + composition + pareto
  build_mixed_pusht.py  M2: human experts + scripted low-skill -> mixed pool
  m3_bc_probe.py        M3-lite: BC-MLP on curated vs random -> gym_pusht rollout
tests/              31 tests, offline, ~0.1s
data/               collected/built pools (.npz)
outputs/            generated CSV + figures
```
