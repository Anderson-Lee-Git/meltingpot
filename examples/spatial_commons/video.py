"""Render a paired (unfiltered vs filtered) rollout as a side-by-side MP4.

The environment is deterministic and policies are seeded, so this replays
exactly the episodes that evaluate.py runs for the same population and seed.

From the repository root:
  python -m examples.spatial_commons.video --policies greedy stay hungry random \
      --seed 0 --rounds 300 --output runs/<date>-<slug>/mixed_seed0.mp4

A frozen PPO checkpoint replays an evaluate_trained.py episode exactly:
  python -m examples.spatial_commons.video \
      --checkpoint runs/.../cooperative/seed_2/final.pt --start energy_02 \
      --rounds 1000 --output runs/<date>-<slug>/cooperative.mp4
"""

import argparse
import dataclasses
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import animation
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
import numpy as np

from examples.spatial_commons import environment
from examples.spatial_commons import model
from examples.spatial_commons import policies

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / ".agents/skills/paper-figure/scaffold"))
try:
  from paper_style import apply_style, PALETTE  # pylint: disable=g-import-not-at-top
except ImportError:  # Plain matplotlib style if the figure scaffold is absent.
  PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]

  def apply_style():
    pass


@dataclasses.dataclass(frozen=True)
class Frame:
  state: model.State
  overridden: tuple[bool, ...]  # In the round that produced this state.
  apples: int  # Cumulative apples eaten by the population.


def rollout(config, make, *, shield, rounds, horizon, initial_state=None):
  """Frames 0..T, where frame k is the state after k rounds."""
  population = make()
  env = environment.SpatialCommonsEnvironment(
      config,
      shield=shield,
      horizon=horizon,
      episode_length=rounds,
      initial_state=initial_state,
  )
  timestep = env.reset()
  frames = [Frame(env.state, (False,) * config.num_agents, 0)]
  while not timestep.last():
    timestep = env.step(
        tuple(p(o) for p, o in zip(population, timestep.observation))
    )
    info = env.last_info
    frames.append(
        Frame(info.after, info.overridden, frames[-1].apples + sum(info.eaten))
    )
  return frames


def board_rgb(config, state, ripe_color, unripe_color, growing_color):
  """Ripe tiles solid; unripe tiles shaded from grey towards pale green."""
  frac = state.ripeness / config.ripe_units
  lo, mid, hi = map(
      np.array, map(to_rgb, (unripe_color, growing_color, ripe_color))
  )
  img = lo + (mid - lo) * frac[..., None]
  img[frac == 1] = hi
  return img


def render(
    config,
    make,
    title,
    *,
    rounds,
    horizon,
    output,
    fps,
    dpi,
    initial_state=None,
):
  apply_style()
  runs = {
      shield: rollout(
          config,
          make,
          shield=shield,
          rounds=rounds,
          horizon=horizon,
          initial_state=initial_state,
      )
      for shield in (False, True)
  }
  n = config.grid_size
  ripe, unripe, growing = PALETTE[2], "#E6E6E6", "#BFE3D3"
  override_color, agent_color = PALETTE[3], "#222222"
  labels = {False: "Unfiltered", True: "Filtered (Backup-Set Filter)"}
  heads = {False: PALETTE[1], True: PALETTE[0]}

  fig, axes = plt.subplots(1, 2, figsize=(11, 6.6))
  fig.subplots_adjust(left=0.03, right=0.97, top=0.83, bottom=0.17, wspace=0.12)
  fig.suptitle(title, fontsize=16)
  artists = {}
  for ax, shield in zip(axes, (False, True)):
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for side in ax.spines.values():
      side.set_visible(False)
    image = ax.imshow(
        board_rgb(config, runs[shield][0].state, ripe, unripe, growing),
        interpolation="nearest",
    )
    for x in np.arange(-0.5, n, 1):
      ax.axhline(x, color="white", lw=1.0)
      ax.axvline(x, color="white", lw=1.0)
    ax.set_title(labels[shield], color=heads[shield], fontsize=15, pad=26)
    status = ax.text(
        0.5,
        1.015,
        "",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )
    energy = ax.text(
        0.5,
        -0.03,
        "",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
    )
    rings, dots, tags = [], [], []
    for i in range(config.num_agents):
      rings.append(
          plt.Circle(
              (0, 0),
              0.47,
              fill=False,
              lw=3,
              color=override_color,
              zorder=3,
              visible=False,
          )
      )
      dots.append(plt.Circle((0, 0), 0.34, color=agent_color, zorder=4))
      ax.add_patch(rings[-1])
      ax.add_patch(dots[-1])
      tags.append(
          ax.text(
              0,
              0,
              str(i + 1),
              color="white",
              ha="center",
              va="center",
              fontsize=11,
              zorder=5,
          )
      )
    banner = ax.text(
        0.5,
        0.5,
        "",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=22,
        color="white",
        zorder=6,
        bbox=dict(
            boxstyle="round,pad=0.4", fc=PALETTE[3], ec="none", alpha=0.9
        ),
        visible=False,
    )
    artists[shield] = (image, status, energy, rings, dots, tags, banner)

  fig.text(
      0.5,
      0.035,
      "Green: Ripe Apple   Pale Green: Ripening   Grey: Unripe   "
      "Red Ring: Move Overridden by Filter This Round",
      ha="center",
      fontsize=12,
  )

  total = max(len(f) for f in runs.values())

  def draw(k):
    changed = []
    for shield, frames in runs.items():
      image, status, energy, rings, dots, tags, banner = artists[shield]
      frame = frames[min(k, len(frames) - 1)]
      done = k >= len(frames) - 1 and frame.state.failed
      s = frame.state
      image.set_data(board_rgb(config, s, ripe, unripe, growing))
      for (r, c), ring, dot, tag, over in zip(
          s.positions, rings, dots, tags, frame.overridden
      ):
        ring.center = dot.center = (c, r)
        tag.set_position((c, r))
        ring.set_visible(over and not done)
      overrides = sum(frame.overridden)
      status.set_text(
          f"Round {s.round_index}   Apples Eaten {frame.apples}"
          + (f"   Overrides {overrides}" if shield else "")
      )
      energy.set_text(
          "Energy   "
          + "   ".join(f"{i + 1}: {e}" for i, e in enumerate(s.energies))
      )
      if frame.state.failed:
        starved = [i + 1 for i, e in enumerate(s.energies) if e == 0]
        banner.set_text(
            f"Agent {', '.join(map(str, starved))} Starved at Round"
            f" {s.round_index}"
        )
        banner.set_visible(True)
      changed += [image, status, energy, banner, *rings, *dots, *tags]
    return changed

  output.parent.mkdir(parents=True, exist_ok=True)
  anim = animation.FuncAnimation(fig, draw, frames=total, blit=False)
  anim.save(
      output,
      writer=animation.FFMpegWriter(fps=fps, bitrate=2400),
      dpi=dpi,
  )
  plt.close(fig)
  return {
      shield: {
          "rounds_executed": len(frames) - 1,
          "failed": frames[-1].state.failed,
          "apples": frames[-1].apples,
          "overridden_agent_steps": sum(sum(f.overridden) for f in frames),
      }
      for shield, frames in runs.items()
  }


def main(argv=None):
  defaults = model.Config()
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--policies",
      nargs="+",
      choices=policies.POLICY_NAMES,
      default=["greedy", "stay", "hungry", "random"],
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--rounds", type=int, default=300)
  parser.add_argument("--horizon", type=int, default=model.DEFAULT_HORIZON)
  parser.add_argument(
      "--growth-rate", type=model.as_fraction, default=defaults.growth_rate
  )
  parser.add_argument(
      "--checkpoint",
      type=Path,
      help="Frozen PPO final.pt; replaces --policies and --growth-rate.",
  )
  parser.add_argument(
      "--start",
      default="default",
      help="Evaluation start name from evaluate_trained (e.g. energy_02).",
  )
  parser.add_argument("--stochastic", action="store_true")
  parser.add_argument("--eval-seed", type=int, default=532)
  parser.add_argument("--fps", type=int, default=10)
  parser.add_argument("--dpi", type=int, default=150)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args(argv)
  initial_state = None
  if args.checkpoint is not None:
    # Same starts and RNG seeds as evaluate_trained.evaluate_checkpoint.
    from examples.spatial_commons import evaluate_trained  # pylint: disable=g-import-not-at-top
    from examples.spatial_commons import ppo  # pylint: disable=g-import-not-at-top

    prototype = ppo.FrozenPolicy(args.checkpoint)
    config = prototype.config
    starts = evaluate_trained.evaluation_starts(config, 4, args.eval_seed)
    index = [name for name, _ in starts].index(args.start)
    initial_state = starts[index][1]
    case_seed = args.eval_seed + index

    def make():
      frozen = ppo.FrozenPolicy(
          args.checkpoint, stochastic=args.stochastic, seed=case_seed
      )
      return frozen.population(seed=case_seed)

    mode = "Stochastic" if args.stochastic else "Deterministic"
    experiment = prototype.experiment.replace("_", " ").title()
    title = (
        f"PPO {experiment}, Training Seed {prototype.metadata['seed']}"
        f" ({mode}, Start {args.start.replace('_', ' ').title()})"
    )
    names = (str(args.checkpoint),)
  else:
    config = model.Config(growth_rate=args.growth_rate)
    names = tuple(args.policies)
    if len(names) == 1:
      names = names * config.num_agents

    def make():
      return policies.make_population(names, config, seed=args.seed)

    title = (
        "Spatial Commons Rollout: "
        + ", ".join(n_.title() for n_ in names)
        + f" (Seed {args.seed})"
    )
  stats = render(
      config,
      make,
      title,
      rounds=args.rounds,
      horizon=args.horizon,
      output=args.output,
      fps=args.fps,
      dpi=args.dpi,
      initial_state=initial_state,
  )
  meta = {
      "video": args.output.name,
      "policies": names,
      "start": args.start if args.checkpoint else "default",
      "stochastic": bool(args.checkpoint and args.stochastic),
      "seed": args.seed,
      "rounds": args.rounds,
      "fps": args.fps,
      "backup_horizon": args.horizon,
      "config": {
          **dataclasses.asdict(config),
          "homes": [list(h) for h in config.homes],
          "growth_rate": str(config.growth_rate),
      },
      "unfiltered": stats[False],
      "filtered": stats[True],
      "command": (
          "python -m examples.spatial_commons.video "
          + " ".join(sys.argv[1:] if argv is None else argv)
      ),
  }
  args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
  print(json.dumps({"unfiltered": stats[False], "filtered": stats[True]}))
  print(f"Wrote {args.output}")


if __name__ == "__main__":
  main()
