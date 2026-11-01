"""
STAGE 3 - Observe how a trained policy performs.

    python evaluate.py runs/hopper_default_ppo_s0                 # numbers
    python evaluate.py runs/hopper_default_ppo_s0 --watch         # live MuJoCo window
    python evaluate.py runs/hopper_default_ppo_s0 --video out.mp4 # save a video
    python evaluate.py runs/hopper_default_ppo_s0 --checkpoint 20 --watch
                                       # watch the policy from 20% of training
    python evaluate.py runs/hopper_default_ppo_s0 --friction 0.6  # slippery world
    python evaluate.py runs/hopper_default_ppo_s0 --robustness    # sweep friction & mass

It reports two different kinds of numbers on purpose:
  * the REWARD the policy earned (what it was trained to maximize), and
  * TASK METRICS (distance, falls) that measure what we actually wanted.
When those disagree, your reward design has a problem.
"""
import argparse, json, os, pickle
import numpy as np
from stable_baselines3 import PPO, SAC
from common import make_env

p = argparse.ArgumentParser()
p.add_argument("run_dir")
p.add_argument("--checkpoint", type=int, default=None, help="percent, e.g. 20 (default: final)")
p.add_argument("--episodes", type=int, default=10)
p.add_argument("--friction", type=float, default=1.0)
p.add_argument("--mass", type=float, default=1.0)
p.add_argument("--robustness", action="store_true", help="sweep friction and mass")
p.add_argument("--watch", action="store_true")
p.add_argument("--video", default=None, help="save the first episode as an mp4")
args = p.parse_args()

cfg = json.load(open(os.path.join(args.run_dir, "config.json")))
if args.checkpoint is None:
    model_path = os.path.join(args.run_dir, "model.zip")
    norm_path = os.path.join(args.run_dir, "vecnormalize.pkl")
else:
    base = os.path.join(args.run_dir, "checkpoints", f"at_{args.checkpoint:03d}pct")
    model_path, norm_path = base + ".zip", base + "_vecnormalize.pkl"

Algo = PPO if cfg["algo"] == "ppo" else SAC
model = Algo.load(model_path, device="cpu")
with open(norm_path, "rb") as f:
    norm = pickle.load(f)          # observation statistics seen during training
norm.training = False


def run(friction, mass, episodes, render_mode=None, video=None):
    env = make_env(cfg["env"], cfg["weights"], friction=friction, mass=mass,
                   render_mode=render_mode)
    results, frames = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=1000 + ep)   # fixed seeds -> comparable across runs
        done = False
        while not done:
            action, _ = model.predict(norm.normalize_obs(obs), deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            if video and ep == 0:
                frames.append(env.render())
        results.append(info["episode_stats"])
    env.close()
    if video and frames:
        import imageio
        imageio.mimsave(video, frames, fps=int(1 / env.unwrapped.dt))
        print(f"Saved video: {video}")
    return results


def summarize(res, label=""):
    ret = np.array([r["return"] for r in res])
    dist = np.array([r["distance"] for r in res])
    fell = np.mean([r["fell"] for r in res])
    length = np.mean([r["length"] for r in res])
    print(f"{label}return {ret.mean():8.1f} ± {ret.std():6.1f} | distance {dist.mean():6.2f} m "
          f"± {dist.std():4.2f} | fell {100*fell:4.0f}% | length {length:6.0f}")
    return ret, dist, fell


print(f"\nPolicy: {model_path}\nTrained with reward weights: {cfg['weights']}\n")

if args.robustness:
    print("Robustness to physics the policy never saw in training:")
    for fr in [0.5, 0.75, 1.0, 1.25, 1.5]:
        summarize(run(fr, 1.0, args.episodes), f"  friction x{fr:<4} ")
    for m in [0.7, 0.85, 1.15, 1.3]:
        summarize(run(1.0, m, args.episodes), f"  mass     x{m:<4} ")
else:
    mode = "human" if args.watch else ("rgb_array" if args.video else None)
    res = run(args.friction, args.mass, args.episodes, render_mode=mode, video=args.video)
    summarize(res, "Overall: ")
    print("\nWhere the reward came from (average per episode):")
    term_keys = [k for k in res[0] if k.startswith("term_")]
    for k in term_keys:
        print(f"  {k[5:]:>8}: {np.mean([r[k] for r in res]):+9.1f}")
