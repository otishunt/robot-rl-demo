"""
STAGE 2 - Train a policy with the reward you designed.

    python train.py                                   # default reward, PPO
    python train.py --preset survive_only             # a reward that gets hacked
    python train.py --preset default --w tilt=0.5     # tweak one term
    python train.py --seed 1 --steps 500000           # another seed, longer run

Everything for a run goes into runs/<name>/:
    config.json        exactly what was trained (env, reward weights, seed...)
    episodes.csv       one row per training episode: return, distance, fell,
                       and how much each reward term contributed
    checkpoints/       snapshots at 20%, 40%, ... of training (watch it improve)
    model.zip          final policy
    vecnormalize.pkl   observation-normalization statistics (needed to evaluate)
"""
import argparse, csv, json, os, time
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from common import make_env, parse_weights

p = argparse.ArgumentParser()
p.add_argument("--env", default="Hopper-v5")
p.add_argument("--preset", default="default",
               help="reward preset from common.py, or 'none' to use the env's own reward")
p.add_argument("--w", nargs="*", help="override reward weights, e.g. --w ctrl=0.01 tilt=0.5")
p.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
p.add_argument("--steps", type=int, default=300_000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n-envs", type=int, default=4, help="parallel copies of the env (PPO)")
p.add_argument("--name", default=None, help="run folder name (auto if omitted)")
args = p.parse_args()

weights = None if args.preset == "none" else parse_weights(args.preset, args.w)
tag = args.preset + ("_" + "_".join(args.w) if args.w else "")
name = args.name or f"{args.env.split('-')[0].lower()}_{tag}_{args.algo}_s{args.seed}"
run_dir = os.path.join("runs", name)
os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

def git_version():
    """Record which commit of the code produced this run (None outside a git repo).
    'dirty' means there were uncommitted edits, so the commit alone can't reproduce it."""
    import subprocess
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "*.py"],
                                        stderr=subprocess.DEVNULL, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


config = {**vars(args), "weights": weights, "run_dir": run_dir, "git_commit": git_version()}
with open(os.path.join(run_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
print(f"Run: {run_dir}\nReward weights: {weights}")


class EpisodeLogger(BaseCallback):
    """Writes one CSV row per finished episode, and saves periodic checkpoints."""

    def __init__(self, path, total_steps, n_checkpoints=5):
        super().__init__()
        self.path, self.rows = path, []
        self.marks = [int(total_steps * (i + 1) / n_checkpoints) for i in range(n_checkpoints)]
        self.t0 = time.time()

    def _on_step(self):
        for info in self.locals["infos"]:
            stats = info.get("episode_stats")
            if stats is None and "episode" in info:          # env's own reward
                stats = {"return": info["episode"]["r"], "length": info["episode"]["l"]}
            if stats is not None:
                self.rows.append({"timesteps": self.num_timesteps, **stats})
        if self.marks and self.num_timesteps >= self.marks[0]:
            pct = int(100 * self.marks.pop(0) / self.locals["total_timesteps"])
            ckpt = os.path.join(run_dir, "checkpoints", f"at_{pct:03d}pct")
            self.model.save(ckpt)
            self.training_env.save(ckpt + "_vecnormalize.pkl")
            recent = self.rows[-20:]
            if recent:
                r = np.mean([x["return"] for x in recent])
                d = np.mean([x.get("distance", np.nan) for x in recent])
                print(f"[{pct:3d}%] {self.num_timesteps:>8} steps | "
                      f"recent return {r:8.1f} | distance {d:6.2f} m | "
                      f"{time.time() - self.t0:5.0f}s elapsed")
        return True

    def _on_training_end(self):
        keys = sorted({k for r in self.rows for k in r}, key=lambda k: (k != "timesteps", k))
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.rows)


n_envs = args.n_envs if args.algo == "ppo" else 1
venv = make_vec_env(lambda: make_env(args.env, weights), n_envs=n_envs, seed=args.seed)
# Normalize observations (helps learning); rewards are left raw so they stay readable.
venv = VecNormalize(venv, norm_obs=True, norm_reward=False)

if args.algo == "ppo":
    model = PPO("MlpPolicy", venv, seed=args.seed, verbose=0,
                n_steps=1024, batch_size=256, gae_lambda=0.95, gamma=0.99,
                n_epochs=10, learning_rate=3e-4, clip_range=0.2)
else:
    model = SAC("MlpPolicy", venv, seed=args.seed, verbose=0, learning_starts=10_000)

model.learn(total_timesteps=args.steps,
            callback=EpisodeLogger(os.path.join(run_dir, "episodes.csv"), args.steps))
model.save(os.path.join(run_dir, "model"))
venv.save(os.path.join(run_dir, "vecnormalize.pkl"))
print(f"Done. Next: python evaluate.py {run_dir}")
