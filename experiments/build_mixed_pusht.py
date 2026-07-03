"""Build a REAL mixed-quality PushT pool: human experts + scripted low-skill.

Scripted heuristics can't push PushT well (it needs a learned policy), so instead
of a fake "expert" tier we combine:

  * expert tier -> real human teleop demos from ``lerobot/pusht`` (reward 0.8-0.95)
  * clumsy/noisy -> the scripted low-skill demos from collect_pusht.py (real
                    gym_pusht dynamics, reward ~0.1-0.2)

All rewards are REAL simulator signals; the quality gap is real and large. This is
the mixed-quality regime that public expert datasets lack (see the M1.5 finding),
and the pool where curation should actually pay off.

Usage:
    python experiments/build_mixed_pusht.py --n-expert 40 \
        --scripted data/collected_pusht.npz --out data/mixed_pusht.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_human_experts(n_expert: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("lerobot/pusht")
    hf = ds.hf_dataset.with_format("numpy")
    ep = np.asarray(hf["episode_index"]).reshape(-1)
    state = np.asarray(hf["observation.state"])
    action = np.asarray(hf["action"])
    reward = np.asarray(hf["next.reward"]).reshape(-1)

    states, actions, rew = [], [], []
    for e in np.unique(ep)[:n_expert]:
        m = ep == e
        states.append(state[m].astype(float))
        actions.append(action[m].astype(float))
        rew.append(float(reward[m].max()))
    return states, actions, rew


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-expert", type=int, default=40)
    ap.add_argument("--scripted", default="data/collected_pusht.npz")
    ap.add_argument("--out", default="data/mixed_pusht.npz")
    args = ap.parse_args()

    # scripted low-skill demos (keep clumsy=1 and noisy=2; drop the weak scripted 'expert')
    sc = np.load(args.scripted, allow_pickle=True)
    keep = sc["operator_mode"].astype(int) != 0
    s_states = list(sc["states"][keep])
    s_actions = list(sc["actions"][keep])
    s_reward = list(sc["ep_reward"][keep].astype(float))
    s_success = list(sc["ep_success"][keep].astype(float))
    s_mode = list(sc["operator_mode"][keep].astype(int))

    # real human experts -> tier 0
    h_states, h_actions, h_reward = load_human_experts(args.n_expert)
    h_success = [0.0] * len(h_states)     # dataset has no success labels
    h_mode = [0] * len(h_states)

    states = h_states + s_states
    actions = h_actions + s_actions
    ep_reward = h_reward + s_reward
    ep_success = h_success + s_success
    operator_mode = h_mode + s_mode

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        states=np.array(states, dtype=object),
        actions=np.array(actions, dtype=object),
        ep_reward=np.asarray(ep_reward, dtype=float),
        ep_success=np.asarray(ep_success, dtype=float),
        operator_mode=np.asarray(operator_mode, dtype=int),
        allow_pickle=True,
    )
    for name, t in (("expert(human)", 0), ("clumsy(scripted)", 1), ("noisy(scripted)", 2)):
        rs = [ep_reward[i] for i in range(len(operator_mode)) if operator_mode[i] == t]
        if rs:
            print(f"  {name:18s}: {len(rs):3d} demos, mean reward {np.mean(rs):.3f} "
                  f"[{np.min(rs):.2f}, {np.max(rs):.2f}]")
    print(f"[out] wrote {len(states)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
