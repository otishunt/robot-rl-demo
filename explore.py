"""
STAGE 1 - Explore the environment before training anything.

Answers: what does the robot observe? what can it do? what reward does it get?

    python explore.py                      # print spaces, run a random policy
    python explore.py --watch              # same, in a live MuJoCo window
    python explore.py --preset speed_only  # see how a different reward scores it
"""
import argparse
import numpy as np
from common import make_env, parse_weights

p = argparse.ArgumentParser()
p.add_argument("--env", default="Hopper-v5")
p.add_argument("--preset", default="default")
p.add_argument("--w", nargs="*", help="override weights, e.g. --w tilt=0.5")
p.add_argument("--steps", type=int, default=300)
p.add_argument("--watch", action="store_true", help="open a live viewer window")
args = p.parse_args()

weights = parse_weights(args.preset, args.w)
env = make_env(args.env, weights, render_mode="human" if args.watch else None)

print(f"\nEnvironment: {args.env}")
print(f"Observation space: {env.observation_space}  (what the policy sees)")
print(f"Action space:      {env.action_space}  (joint torques it outputs)")
print(f"Reward weights:    {weights}\n")

obs, info = env.reset(seed=0)
print(f"First observation (first 6 values): {np.round(obs[:6], 3)}\n")
print(f"{'step':>5} {'reward':>8}  per-term contributions")

episodes, ep_return = 0, 0.0
for t in range(args.steps):
    action = env.action_space.sample()   # a random policy: the "untrained" baseline
    obs, reward, terminated, truncated, info = env.step(action)
    ep_return += reward
    if t < 15 or terminated or truncated:
        terms = "  ".join(f"{k}={v:+.3f}" for k, v in info["reward_terms"].items())
        print(f"{t:>5} {reward:>8.3f}  {terms}")
    if terminated or truncated:
        s = info["episode_stats"]
        print(f"  -> episode ended after {s['length']} steps | return {s['return']:.1f} "
              f"| distance {s['distance']:.2f} m | fell={s['fell']}\n")
        episodes += 1
        obs, info = env.reset()
env.close()
print(f"Random policy finished {episodes} episodes in {args.steps} steps. "
      "Training should make episodes longer and returns higher.")
