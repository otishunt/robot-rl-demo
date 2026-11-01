# Robot RL pipeline: a small, hands-on example

A minimal end-to-end reinforcement learning loop on MuJoCo's **Hopper**, a one-legged
robot that has to learn to hop forward without falling. The point isn't a state-of-the-art
policy. It's to make every moving piece of the pipeline visible, with **reward design**
front and center.

```
 explore.py ──► train.py ──► evaluate.py ──► plot.py
 what does the   learn a      watch it, measure   compare runs:
 robot see/do/   policy from  it, stress-test it  reward vs. what the
 get rewarded?   YOUR reward  under new physics   robot actually did
      ▲                                                  │
      └──────── change the reward in common.py ◄─────────┘
```

## Setup (laptop or PC, about 5 minutes)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python explore.py --watch                              # a MuJoCo window should open
```

A training run (300–400k steps) takes roughly 3–10 minutes on a laptop CPU. No GPU needed.

**Trained policies are included** in `runs/`, so you can watch them right away without training:

```bash
python evaluate.py runs/hopper_survive_only_ppo_s0 --watch     # the reward hack
python evaluate.py runs/hopper_default_long_ppo_s0 --watch     # the best one here
```

## The pieces

| File | Pipeline stage | What to look at |
|---|---|---|
| `common.py` | **Environment + reward** | `TERMS` (reward ingredients), `REWARD_PRESETS` (recipes), `PhysicsShift` (change the world) |
| `explore.py` | Understand the problem | Observation/action spaces; per-step reward from a random policy |
| `train.py` | Learn a policy (PPO) | Progress printed at 20/40/60/80/100%; saves checkpoints along the way |
| `evaluate.py` | Observe performance | Live viewer, mp4 videos, reward-vs-task metrics, robustness sweep |
| `plot.py` | Compare runs | Learning curves across rewards and seeds; per-term breakdown |

### 1. Explore

```bash
python explore.py                          # spaces + random-policy reward, step by step
python explore.py --preset speed_only      # same actions, scored by a different reward
```

The Hopper sees 11 numbers (joint angles and velocities) and outputs 3 torques (thigh,
leg, foot). A random policy falls over in about 15–20 steps.

### 2. Design a reward

Rewards are built from named **terms** in `common.py`, each multiplied by a weight:

| Term | Meaning | Typical role |
|---|---|---|
| `forward` | forward velocity | the actual task |
| `alive` | +1 per step while upright | "don't fall" |
| `ctrl` | −(sum of squared torques) | energy efficiency |
| `tilt` | −(torso angle)² | posture / style |
| `smooth` | −(change in torques)² | smooth motion, matters on real hardware |
| `height` | torso height above start | a deliberately tempting term |

Presets:

| Preset | Weights | What you should expect |
|---|---|---|
| `default` | forward 1, alive 1, ctrl 0.001 | Gymnasium's own reward; hops forward |
| `survive_only` | alive 1 | **Reward hacking:** learns to stand still. Perfect reward, zero progress |
| `speed_only` | forward 1 | Falling costs nothing, so it may lunge forward and fall |
| `smooth` | default + tilt, smooth, more ctrl | Slower but steadier? You find out |
| `jumpy` | default + height 3 | Does it bounce in place instead of travelling? |

Change any weight from the command line without editing code, or set it to 0 to remove a term:

```bash
python train.py --preset default --w ctrl=0.01 tilt=0.5
python train.py --preset default --w alive=0          # what happens without the alive bonus?
```

To invent a **new term**, write a 2-line function in `common.py`, add it to `TERMS`, and
give it a weight. The `ctx` dictionary already has velocity, torso angle, height, and
current/previous action.

### 3. Train

```bash
python train.py                                   # default reward, seed 0, 300k steps
python train.py --preset survive_only
python train.py --preset default --seed 1         # same reward, different seed
```

Each run gets its own folder in `runs/`, with `config.json` recording exactly what was
trained. That record is what makes runs comparable later.

### 4. Observe

```bash
python evaluate.py runs/hopper_default_ppo_s0 --watch               # live 3D viewer
python evaluate.py runs/hopper_default_ppo_s0 --checkpoint 20 --watch   # early in training
python evaluate.py runs/hopper_default_ppo_s0 --video hop.mp4
python evaluate.py runs/hopper_default_ppo_s0 --robustness          # friction & mass sweep
```

`evaluate.py` deliberately reports two kinds of numbers:

- **Reward** is what the policy was trained to maximize.
- **Task metrics** (distance travelled, fall rate) are what you actually wanted.

**When those two disagree, the reward is the problem, not the learning algorithm.** That is
the core lesson of reward design.

### 5. Compare and adjust

```bash
python plot.py runs/* --out compare.png
```

- `compare.png`: reward, distance, and fall rate over training for every run. Runs that
  differ only by seed are grouped: the line is the mean and the band is the range across seeds.
- `compare_terms.png`: how much each reward term contributed over training. It shows
  *which part* of the reward the policy learned to exploit.

Then change the reward, retrain, and compare again. That's the loop.

## What the included demo showed (`demo_outputs/`)

Three reward presets, 2 seeds each, 400k PPO steps (~3 min per run on a laptop CPU):

| Preset | Reward earned | Distance | Fell | What the robot learned |
|---|---|---|---|---|
| `default` | ~585 | ~3.3 m | 100% | Lunges and scrambles forward, then falls |
| `speed_only` | ~410 | ~3.3 m | 100% | Practically the same lunge |
| `survive_only` | **1000 (max)** | **0.01 m** | 0% | Stands perfectly still for 1000 steps |

Things to notice in `compare.png`:

- **Reward hacking, caught in the act.** `survive_only` first moves forward (distance peaks
  ~2 m around 100k steps), then discovers that standing still is the safest way to collect
  `alive`. Its reward hits the ceiling exactly as its distance collapses to zero.
- **The `alive` term wasn't doing its job yet.** `default` and `speed_only` end up with the
  same behavior and both fall 100% of the time (their lines overlap at 100% in the right
  panel). The extra ~170 reward from `alive` came from the fall happening a bit later, not
  from staying up. Reward totals differ, behavior doesn't.
- **More training changes the picture.** The same `default` reward trained for 1.5M steps
  (`runs/hopper_default_long_ppo_s0`, ~14 min) reaches ~8 m and survives ~325 steps instead
  of ~175, with `alive` contributing ~325. It still falls eventually, so "is my reward wrong,
  or did I just not train long enough?" is a real question you have to answer with experiments.
  Compare `--checkpoint 20` with the final model in that run to see the difference.
- **Seeds agree here** (narrow bands), so these differences are real rather than seed noise.
- **Fragility:** `python evaluate.py runs/hopper_default_ppo_s0 --robustness` showed that
  25% less friction cut distance from 3.3 m to 0.55 m. The policy is tuned to one exact world.

## Experiments to try, roughly in order

1. **Watch learning happen:** `--watch` the 20%, 60% and 100% checkpoints of one run.
2. **See reward hacking:** train `default` and `survive_only`, then compare their reward
   curves and their distance curves.
3. **Seed variance, your noise floor:** train `default` with seeds 0–4 and plot them.
   How wide is the band? Any reward change smaller than that band can't be distinguished
   from noise. This is the "noise floor" idea from the research plan.
4. **One-knob sensitivity:** sweep `--w ctrl=` across 0.0001, 0.001, 0.01, 0.1 with 3 seeds
   each. Plot distance against the weight.
5. **Reward vs. robustness:** run `--robustness` on `default` and on `smooth`. Does a
   "nicer" gait survive slippery floors or heavier bodies better?
6. **Your own term:** e.g. reward foot clearance, or penalize hopping backwards. Predict
   what will happen *before* you train, then check.
7. **Swap the robot:** `--env Walker2d-v5` (two legs) works with the same reward terms.

## Notes

- **macOS:** the `--watch` window should work with plain `python`. If it doesn't, use
  `--video` instead and open the mp4.
- **Headless machines (TIDE, servers):** there's no screen, so use `--video` and set
  `MUJOCO_GL=egl` (GPU nodes) or `MUJOCO_GL=osmesa` (CPU only) before the command.
- **Training length:** 300–400k steps learns a forward lunge; 1.5M steps travels much
  further. A smooth, never-falling hop typically needs more steps and some hyperparameter
  tuning. Pass `--steps`.
- **Speed:** about 2,000 steps/s for one run on a laptop CPU. Running two at once on a
  2-core machine roughly halves each one's speed.
- **Algorithms:** `--algo sac` is more sample-efficient but slower per step on a CPU.
- The `forward` term sums per-step velocities, so its episode total equals distance ÷ 0.008 s.
  That's why it can read ~250 while distance reads ~2 m.
