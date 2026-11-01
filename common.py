"""
Shared building blocks for the pipeline:

  1. RewardDesigner  - replaces the environment's built-in reward with one YOU
                       assemble from named terms and weights. Logs every term.
  2. PhysicsShift    - scales friction / mass so you can test a trained policy
                       in a "different world" (a sim-to-sim stand-in for reality).
  3. make_env        - builds an environment with both of the above applied.

The reward is the main thing to play with. Edit REWARD_PRESETS or pass
--w name=value on the command line to change weights without touching code.
"""
import numpy as np
import gymnasium as gym

# --------------------------------------------------------------------------
# Reward terms. Each one is a raw, unweighted signal computed every step.
# Sign convention: positive = good, penalties are already negative.
# To add your own term, add a function here and a weight in a preset.
# --------------------------------------------------------------------------
def term_forward(ctx):   # forward velocity of the torso (m/s)
    return ctx["x_vel"]

def term_alive(ctx):     # 1 per step while the robot is "healthy" (upright)
    return 1.0 if ctx["healthy"] else 0.0

def term_ctrl(ctx):      # energy penalty: big torques cost reward
    return -float(np.sum(np.square(ctx["action"])))

def term_tilt(ctx):      # penalize torso leaning away from vertical
    return -float(ctx["torso_angle"] ** 2)

def term_smooth(ctx):    # penalize jerky changes in action between steps
    return -float(np.sum(np.square(ctx["action"] - ctx["prev_action"])))

def term_height(ctx):    # reward torso height above its starting height
    return float(ctx["z"] - ctx["z0"])

TERMS = {
    "forward": term_forward,
    "alive": term_alive,
    "ctrl": term_ctrl,
    "tilt": term_tilt,
    "smooth": term_smooth,
    "height": term_height,
}

# --------------------------------------------------------------------------
# Presets. "default" reproduces Gymnasium's own Hopper reward, so you have a
# known-good baseline. The others are deliberately flawed or opinionated so
# you can SEE what changing the reward does.
# --------------------------------------------------------------------------
REWARD_PRESETS = {
    # Gymnasium's built-in Hopper/Walker2d reward
    "default":      {"forward": 1.0, "alive": 1.0, "ctrl": 1e-3},
    # Only reward staying upright -> expect the robot to learn to stand still
    "survive_only": {"alive": 1.0},
    # Only reward speed -> expect lunging/diving, falling is "free"
    "speed_only":   {"forward": 1.0},
    # Default plus style terms -> slower but steadier, smoother gait?
    "smooth":       {"forward": 1.0, "alive": 1.0, "ctrl": 1e-2,
                     "tilt": 0.5, "smooth": 0.1},
    # Default plus a height bonus -> does it learn to bounce instead of walk?
    "jumpy":        {"forward": 1.0, "alive": 1.0, "ctrl": 1e-3, "height": 3.0},
}

# Environments whose state layout the reward terms understand
# (qpos[0]=x, qpos[1]=torso height, qpos[2]=torso angle).
DESIGNABLE_ENVS = {"Hopper-v5", "Walker2d-v5"}


class RewardDesigner(gym.Wrapper):
    """Replaces env reward with sum_i w_i * term_i, and records each term.

    info["reward_terms"]   -> this step's weighted contribution of each term
    info["episode_stats"]  -> on the last step of an episode: totals per term,
                              plus task metrics that are NOT part of the reward
                              (distance travelled, whether it fell, length).
    """

    def __init__(self, env, weights):
        super().__init__(env)
        unknown = set(weights) - set(TERMS)
        if unknown:
            raise ValueError(f"Unknown reward terms {unknown}; choose from {list(TERMS)}")
        self.weights = dict(weights)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        data = self.env.unwrapped.data
        self.x0, self.z0 = float(data.qpos[0]), float(data.qpos[1])
        self.prev_action = np.zeros(self.action_space.shape)
        self.ep_terms = {k: 0.0 for k in self.weights}
        self.ep_return, self.ep_len = 0.0, 0
        return obs, info

    def step(self, action):
        u = self.env.unwrapped
        x_before = float(u.data.qpos[0])
        obs, _env_reward, terminated, truncated, info = self.env.step(action)
        ctx = {
            "x_vel": (float(u.data.qpos[0]) - x_before) / u.dt,
            "healthy": bool(u.is_healthy),
            "action": np.asarray(action),
            "prev_action": self.prev_action,
            "torso_angle": float(u.data.qpos[2]),
            "z": float(u.data.qpos[1]),
            "z0": self.z0,
        }
        contrib = {k: w * TERMS[k](ctx) for k, w in self.weights.items()}
        reward = float(sum(contrib.values()))
        self.prev_action = np.asarray(action).copy()

        for k, v in contrib.items():
            self.ep_terms[k] += v
        self.ep_return += reward
        self.ep_len += 1
        info["reward_terms"] = contrib

        if terminated or truncated:
            info["episode_stats"] = {
                "return": self.ep_return,
                "length": self.ep_len,
                # Task metrics: what we actually care about, measured
                # independently of whatever reward we chose.
                "distance": float(u.data.qpos[0]) - self.x0,
                "fell": bool(terminated),
                **{f"term_{k}": v for k, v in self.ep_terms.items()},
            }
        return obs, reward, terminated, truncated, info


class PhysicsShift(gym.Wrapper):
    """Scale ground/body friction and body masses once, at construction.

    friction=0.7 means 30% more slippery; mass=1.2 means 20% heavier.
    Train at 1.0/1.0, evaluate at other values to see how robust a policy is.
    """

    def __init__(self, env, friction=1.0, mass=1.0):
        super().__init__(env)
        model = self.env.unwrapped.model
        model.geom_friction[:, 0] *= friction   # sliding friction
        model.body_mass[:] *= mass


def parse_weights(preset, overrides):
    """Start from a preset, then apply --w name=value overrides."""
    if preset not in REWARD_PRESETS:
        raise ValueError(f"Unknown preset '{preset}'; choose from {list(REWARD_PRESETS)}")
    weights = dict(REWARD_PRESETS[preset])
    for item in overrides or []:
        name, value = item.split("=")
        value = float(value)
        if value == 0:
            weights.pop(name, None)
        else:
            weights[name] = value
    return weights


def make_env(env_id="Hopper-v5", weights=None, friction=1.0, mass=1.0,
             render_mode=None):
    kwargs = {}
    if render_mode == "rgb_array" and env_id in DESIGNABLE_ENVS:
        kwargs["camera_name"] = "track"      # follow the robot in saved videos
    env = gym.make(env_id, render_mode=render_mode, **kwargs)
    if friction != 1.0 or mass != 1.0:
        env = PhysicsShift(env, friction=friction, mass=mass)
    if weights is not None:
        if env_id not in DESIGNABLE_ENVS:
            raise ValueError(f"Custom rewards support {DESIGNABLE_ENVS}; "
                             f"use --preset none for {env_id}")
        env = RewardDesigner(env, weights)
    return env
