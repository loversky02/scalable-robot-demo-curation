"""Non-circular curation check (Milestone 2.5).

The DPP is scored with a REWARD-FREE proxy quality (kinematic smoothness x
efficiency) and the selection is then evaluated against the held-out task
reward. Because the proxy never sees the reward, a win is not circular -- it
shows that cheap, reward-independent quality signals, coupled with diversity,
recover high-reward demonstrations and avoid the diversity-only failure mode.
"""

import numpy as np

from robocurate import (
    expert_fraction,
    make_labeled_pool,
    normalize_quality,
    proxy_quality,
    select_diversity_only,
    select_dpp,
    select_random,
)

K = 30


def test_proxy_is_informative_but_not_the_reward():
    """The proxy must be correlated with reward (useful) yet NOT identical to it
    (otherwise the evaluation would be circular)."""
    p = make_labeled_pool(seed=0)
    q_proxy = proxy_quality(p["smoothness"], p["efficiency"])
    c = float(np.corrcoef(q_proxy, p["reward"])[0, 1])
    assert 0.5 < c < 0.98


def test_reward_free_proxy_recovers_high_reward_demos():
    p = make_labeled_pool(seed=0)
    X, reward = p["embeddings"], p["reward"]
    q_proxy = proxy_quality(p["smoothness"], p["efficiency"])   # never sees reward

    sel_proxy = select_dpp(X, q_proxy, K)
    sel_div = select_diversity_only(X, K)
    sel_rand = select_random(len(X), K, np.random.default_rng(0))

    def mean_reward(sel):
        return float(np.mean(reward[sel]))

    # proxy-q DPP beats the diversity-only failure mode by a wide margin ...
    assert mean_reward(sel_proxy) - mean_reward(sel_div) > 0.3
    # ... and beats picking blindly at random.
    assert mean_reward(sel_proxy) > mean_reward(sel_rand) + 0.1
    # and it mostly avoids the noisy novice demos.
    assert expert_fraction(sel_proxy, p["is_expert"]) > 0.75


def test_deployable_proxy_approaches_oracle_reward_q():
    """The deployable reward-free proxy should come close to the oracle setting
    that scores with the true reward."""
    p = make_labeled_pool(seed=0)
    X, reward = p["embeddings"], p["reward"]
    q_proxy = proxy_quality(p["smoothness"], p["efficiency"])
    q_reward = normalize_quality(reward)

    def mean_reward(sel):
        return float(np.mean(reward[sel]))

    proxy_r = mean_reward(select_dpp(X, q_proxy, K))
    oracle_r = mean_reward(select_dpp(X, q_reward, K))
    assert proxy_r >= oracle_r - 0.20


def test_holds_across_seeds():
    for seed in (0, 1, 2):
        p = make_labeled_pool(seed=seed)
        X, reward = p["embeddings"], p["reward"]
        q_proxy = proxy_quality(p["smoothness"], p["efficiency"])
        r_proxy = float(np.mean(reward[select_dpp(X, q_proxy, K)]))
        r_div = float(np.mean(reward[select_diversity_only(X, K)]))
        assert r_proxy - r_div > 0.25
