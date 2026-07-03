"""The thesis test: on a mixed-quality pool, informativeness needs BOTH
quality and diversity.

This is an offline correctness fixture, not evidence for the research claim
(that comes from the same pipeline on lerobot/pusht + aloha). It asserts that:

  * diversity-only chases the noise      -> low quality, few experts
  * quality-only is the most redundant   -> highest quality, lowest diversity
  * the DPP recovers quality (near quality-only) while being MORE diverse than
    quality-only                          -> dominates both on the trade-off
"""

import numpy as np

from robocurate import (
    diversity_spread,
    expert_fraction,
    make_mixed_pool,
    normalize_quality,
    select_diversity_only,
    select_dpp,
    select_quality_only,
)

K = 30


def _run(seed=0):
    X, q_raw, is_expert = make_mixed_pool(seed=seed)
    q = normalize_quality(q_raw)
    sel = {
        "quality_only": select_quality_only(q, K),
        "diversity_only": select_diversity_only(X, K),
        "dpp": select_dpp(X, q, K),
    }
    mq = {m: float(np.mean(q[s])) for m, s in sel.items()}
    sp = {m: diversity_spread(s, X) for m, s in sel.items()}
    ef = {m: expert_fraction(s, is_expert) for m, s in sel.items()}
    return mq, sp, ef, float(q.mean())


def test_diversity_only_chases_noise():
    mq, sp, ef, _ = _run()
    assert mq["diversity_only"] < 0.35        # low quality
    assert ef["diversity_only"] < 0.25        # grabs the novice outliers


def test_dpp_recovers_quality():
    mq, sp, ef, pool_mean = _run()
    assert mq["dpp"] > 0.85                    # near the quality-only ceiling
    assert mq["dpp"] > pool_mean + 0.2         # much better than picking blindly
    assert ef["dpp"] > 0.95                    # essentially avoids all noise


def test_dpp_more_diverse_than_quality_only():
    mq, sp, ef, _ = _run()
    # the whole point: DPP is not just top-quality, it spreads out too
    assert sp["dpp"] > sp["quality_only"]


def test_quality_diversity_tradeoff_shape():
    """Sanity on the ablation geometry: quality-only maximizes quality,
    diversity-only maximizes diversity, DPP dominates the trade-off."""
    mq, sp, ef, _ = _run()
    assert mq["quality_only"] >= mq["dpp"]         # quality-only is the quality ceiling
    assert sp["diversity_only"] >= sp["dpp"]       # diversity-only is the diversity ceiling
    # DPP beats diversity-only on quality by a wide margin
    assert mq["dpp"] - mq["diversity_only"] > 0.4


def test_thesis_holds_across_seeds():
    for seed in (0, 1, 2, 3):
        mq, sp, ef, pool_mean = _run(seed)
        assert mq["dpp"] - mq["diversity_only"] > 0.3
        assert ef["dpp"] > ef["diversity_only"]
        assert mq["dpp"] > pool_mean
