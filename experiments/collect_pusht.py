"""M2 -- low-barrier PushT demonstration collection into a mixed-quality pool.

Two modes, both writing the same on-disk format (real gym_pusht dynamics, so the
per-episode reward is a REAL simulator signal, not synthetic):

  scripted  -- heuristic controllers at three skill tiers
                 expert : a simple goal-directed pusher
                 clumsy : expert action + moderate noise
                 noisy  : mostly random actions
               Runs headless -> reproducible mixed-quality pool with no human.

  human     -- pygame mouse teleoperation: the agent follows the cursor. This is
               the actual "lower the barrier" interface -- a non-expert contributes
               demos with a mouse. Run it locally (needs a display). Tag each run
               with --operator-mode so you can build a heterogeneous pool by hand.

Output: an .npz pool (states/actions per episode + reward/success/operator_mode),
loadable by robocurate.collected.load_collected_pool and runnable through
experiments/proxy_vs_oracle.py --source collected --path <file>.

Usage:
    python experiments/collect_pusht.py --mode scripted --per-tier 25 --out data/collected_pusht.npz
    python experiments/collect_pusht.py --mode human --n 10 --operator-mode expert --out data/human_pusht.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TIERS = {"expert": 0, "clumsy": 1, "noisy": 2}
GOAL = np.array([256.0, 256.0], dtype=float)


def expert_action(obs: np.ndarray) -> np.ndarray:
    """A deliberately simple goal-directed pusher (not an oracle).

    Position the agent behind the block (relative to the goal), then push the
    block toward the goal. Good enough to beat noise on average -- which is all we
    need to create a real reward gap between the skill tiers.
    """
    agent, block = obs[:2], obs[2:4]
    to_goal = GOAL - block
    d = np.linalg.norm(to_goal)
    if d < 1e-3:
        return GOAL.astype(np.float32)
    push_dir = to_goal / d
    behind = block - push_dir * 40.0
    if np.linalg.norm(agent - behind) > 30.0:
        target = behind                    # get behind the block first
    else:
        target = block + push_dir * 60.0   # push through it toward the goal
    return np.clip(target, 0.0, 512.0).astype(np.float32)


def tier_action(tier: int, obs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    e = expert_action(obs)
    if tier == 0:
        return e
    if tier == 1:
        return np.clip(e + rng.normal(0.0, 45.0, size=2), 0.0, 512.0).astype(np.float32)
    return rng.uniform(0.0, 512.0, size=2).astype(np.float32)   # noisy


def _rollout(env, tier, seed, max_steps, rng):
    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=float)
    states, actions, rewards, success = [obs.copy()], [], [], False
    for _ in range(max_steps):
        a = tier_action(tier, obs, rng)
        obs, r, term, trunc, info = env.step(a)
        obs = np.asarray(obs, dtype=float)
        actions.append(np.asarray(a, dtype=float))
        states.append(obs.copy())
        rewards.append(float(r))
        success = success or bool(info.get("is_success", False))
        if term or trunc:
            break
    return (np.asarray(states), np.asarray(actions),
            float(np.max(rewards)) if rewards else 0.0, float(success))


def collect_scripted(per_tier: int, max_steps: int, out: Path, seed: int = 0):
    import gymnasium as gym
    import gym_pusht  # noqa: F401

    env = gym.make("gym_pusht/PushT-v0", obs_type="state")
    rng = np.random.default_rng(seed)
    states, actions, ep_reward, ep_success, operator_mode = [], [], [], [], []
    ep = 0
    for tier_name, tier in TIERS.items():
        for i in range(per_tier):
            s, a, r, ok = _rollout(env, tier, seed=1000 * tier + i, max_steps=max_steps, rng=rng)
            states.append(s); actions.append(a)
            ep_reward.append(r); ep_success.append(ok); operator_mode.append(tier)
            ep += 1
        rs = [ep_reward[j] for j in range(len(operator_mode)) if operator_mode[j] == tier]
        print(f"  {tier_name:7s}: {per_tier} demos, mean peak-reward {np.mean(rs):.3f} "
              f"[{np.min(rs):.2f}, {np.max(rs):.2f}]")
    env.close()
    _save(out, states, actions, ep_reward, ep_success, operator_mode)
    print(f"[out] wrote {ep} episodes -> {out}")


def collect_human(n: int, operator_mode: str, max_steps: int, out: Path):
    """pygame mouse teleoperation -- run locally (needs a display)."""
    import gymnasium as gym
    import gym_pusht  # noqa: F401
    import pygame

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode="human")
    tier = TIERS[operator_mode]
    win = 512
    states, actions, ep_reward, ep_success, modes = [], [], [], [], []
    for i in range(n):
        obs, _ = env.reset(seed=10_000 + i)
        obs = np.asarray(obs, dtype=float)
        s, a, rs, ok = [obs.copy()], [], [], False
        for _ in range(max_steps):
            pygame.event.pump()
            mx, my = pygame.mouse.get_pos()
            act = np.clip([mx / max(pygame.display.get_surface().get_width(), 1) * win,
                           my / max(pygame.display.get_surface().get_height(), 1) * win],
                          0, 512).astype(np.float32)
            obs, r, term, trunc, info = env.step(act)
            obs = np.asarray(obs, dtype=float)
            a.append(np.asarray(act, dtype=float)); s.append(obs.copy()); rs.append(float(r))
            ok = ok or bool(info.get("is_success", False))
            env.render()
            if term or trunc:
                break
        states.append(np.asarray(s)); actions.append(np.asarray(a))
        ep_reward.append(float(np.max(rs)) if rs else 0.0)
        ep_success.append(float(ok)); modes.append(tier)
        print(f"  demo {i + 1}/{n} ({operator_mode}) peak-reward {ep_reward[-1]:.3f}")
    env.close()
    _save(out, states, actions, ep_reward, ep_success, modes)
    print(f"[out] wrote {n} human demos -> {out}")


def _save(out: Path, states, actions, ep_reward, ep_success, operator_mode):
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        states=np.array(states, dtype=object),
        actions=np.array(actions, dtype=object),
        ep_reward=np.asarray(ep_reward, dtype=float),
        ep_success=np.asarray(ep_success, dtype=float),
        operator_mode=np.asarray(operator_mode, dtype=int),
        allow_pickle=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["scripted", "human"], default="scripted")
    ap.add_argument("--per-tier", type=int, default=25)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--operator-mode", choices=list(TIERS), default="expert")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--out", default="data/collected_pusht.npz")
    args = ap.parse_args()
    out = Path(args.out)
    if args.mode == "scripted":
        print("[collect] scripted mixed-quality pool (real gym_pusht dynamics)")
        collect_scripted(args.per_tier, args.max_steps, out)
    else:
        collect_human(args.n, args.operator_mode, args.max_steps, out)


if __name__ == "__main__":
    main()
