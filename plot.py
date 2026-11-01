"""
STAGE 4 - Compare runs (this is where reward design becomes visible).

    python plot.py runs/hopper_default_ppo_s0 runs/hopper_survive_only_ppo_s0
    python plot.py runs/*                       # everything you've trained
    python plot.py runs/* --out compare.png

Produces two figures:
  <out>             learning curves: reward return, distance travelled, fall rate
  <out>_terms.png   for each run, how much each reward term contributed over training
                    (shows WHICH part of the reward the policy learned to exploit)

Runs that differ only by seed (..._s0, ..._s1, ...) are grouped: the line is the
mean across seeds and the band is the min-max range across seeds.
"""
import argparse, json, os, re, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # empty bins when runs differ in length
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from common import TERMS

K = FuncFormatter(lambda x, _: f"{x/1e6:g}M" if x >= 1e6 else f"{x/1e3:g}k")

# Fixed categorical order (color follows the run group, never its rank).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

p = argparse.ArgumentParser()
p.add_argument("runs", nargs="+")
p.add_argument("--out", default="compare.png")
p.add_argument("--bins", type=int, default=40, help="timestep bins for smoothing")
args = p.parse_args()


def load(run):
    df = pd.read_csv(os.path.join(run, "episodes.csv"))
    cfg = json.load(open(os.path.join(run, "config.json")))
    return df, cfg


def binned(df, col, edges):
    idx = np.digitize(df["timesteps"], edges)
    return np.array([df[col][idx == i].mean() if (idx == i).any() else np.nan
                     for i in range(1, len(edges))])


runs = [r.rstrip("/") for r in args.runs if os.path.exists(os.path.join(r, "episodes.csv"))]
groups = {}
for r in runs:
    groups.setdefault(re.sub(r"_s\d+$", "", os.path.basename(r)), []).append(r)
if len(groups) > len(PALETTE):
    raise SystemExit(f"{len(groups)} groups; plot at most {len(PALETTE)} at a time.")

max_t = max(load(r)[0]["timesteps"].max() for r in runs)
edges = np.linspace(0, max_t, args.bins + 1)
centers = (edges[:-1] + edges[1:]) / 2

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK})
panels = [("return", "Reward return per episode", None),
          ("distance", "Distance travelled (m)", None),
          ("fell", "Episodes ending in a fall (%)", 100)]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, (col, title, scale) in zip(axes, panels):
    for color, (gname, members) in zip(PALETTE, groups.items()):
        curves = []
        for r in members:
            df, _ = load(r)
            if col in df:
                curves.append(binned(df, col, edges) * (scale or 1))
        if not curves:
            continue
        c = np.vstack(curves)
        ax.plot(centers, np.nanmean(c, 0), color=color, lw=2,
                label=f"{gname} (n={len(curves)} seed{'s' if len(curves) > 1 else ''})")
        if len(curves) > 1:
            ax.fill_between(centers, np.nanmin(c, 0), np.nanmax(c, 0), color=color, alpha=0.15, lw=0)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("training timesteps")
    ax.xaxis.set_major_formatter(K)
    ax.grid(True, color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 3), frameon=False,
           fontsize=9, bbox_to_anchor=(0.5, 0))
fig.suptitle("Same robot, different rewards: what the reward says vs. what the robot does",
             x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(args.out, dpi=130)
print(f"Saved {args.out}")

# ---- Per-term breakdown: one small panel per group -------------------------
fig, axes = plt.subplots(1, len(groups), figsize=(4.6 * len(groups), 3.8), squeeze=False)
for ax, (gname, members) in zip(axes[0], groups.items()):
    df, cfg = load(members[0])
    terms = [c for c in df.columns if c.startswith("term_")]
    for t in terms:
        color = PALETTE[list(TERMS).index(t[5:]) % len(PALETTE)]
        y = np.nanmean([binned(load(r)[0], t, edges) for r in members], 0)
        ax.plot(centers, y, color=color, lw=2, label=t[5:])
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title(gname, loc="left", fontsize=10)
    ax.set_xlabel("training timesteps")
    ax.xaxis.set_major_formatter(K)
    ax.grid(True, color=GRID, lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
axes[0][0].set_ylabel("contribution to episode reward")
fig.suptitle("Where the reward comes from, per term", x=0.01, ha="left", fontsize=12, color=INK)
fig.tight_layout()
terms_out = os.path.splitext(args.out)[0] + "_terms.png"
fig.savefig(terms_out, dpi=130)
print(f"Saved {terms_out}")
