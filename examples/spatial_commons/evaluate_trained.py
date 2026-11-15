"""Evaluate frozen unshielded-trained checkpoints with and without the filter.

Every checkpoint is played from the same starts, in deterministic (argmax) and
stochastic (sampled) modes, with the backup-set filter off and on. Weights are
frozen; nothing here updates the policy.
"""

import argparse
import csv
import json
from pathlib import Path

from examples.spatial_commons import evaluate
from examples.spatial_commons import model
from examples.spatial_commons import ppo
import numpy as np
import torch


def evaluation_starts(config, random_starts, seed):
  """Default start, held-out energies at home, held-out positions.

  Every start is in Omega_B, so the shielded condition is well defined.
  """
  base = model.default_state(config)
  states = [("default", base)]
  rng = np.random.default_rng(seed)
  cells = [
      (r, c) for r in range(config.grid_size) for c in range(config.grid_size)
  ]
  while len(states) < 1 + random_starts:
    energies = tuple(
        int(e)
        for e in rng.integers(10, config.max_energy + 1, config.num_agents)
    )
    states.append((
        f"energy_{len(states) - 1:02}",
        model.State(base.positions, energies, base.ripeness),
    ))
  count = 0
  while count < random_starts:
    picks = rng.choice(len(cells), size=config.num_agents, replace=False)
    positions = tuple(cells[int(k)] for k in picks)
    energies = tuple(
        int(e)
        for e in rng.integers(15, config.max_energy + 1, config.num_agents)
    )
    state = model.State(positions, energies, base.ripeness)
    if model.in_backup_set(config, state):
      states.append((f"position_{count:02}", state))
      count += 1
  return states


def evaluate_checkpoint(path, output, steps=1000, random_starts=4, seed=532):
  torch.set_num_threads(1)
  prototype = ppo.FrozenPolicy(path)
  if prototype.centralized:
    raise ValueError(
        "Centralized checkpoints act on the joint state; evaluate them with"
        " `python -m examples.spatial_commons.baselines evaluate`."
    )
  config = prototype.config
  output.mkdir(parents=True, exist_ok=True)
  summaries = []
  starts = evaluation_starts(config, random_starts, seed)
  with (output / "trajectories.csv").open("w", newline="") as stream:
    writer = None
    for selection in ("deterministic", "stochastic"):
      for case_index, (name, state) in enumerate(starts):
        for shield in (False, True):
          # Fresh instances with matching RNG seeds pair the two conditions.
          frozen = ppo.FrozenPolicy(
              path, stochastic=selection == "stochastic", seed=seed + case_index
          )
          result = evaluate.run_episode(
              config,
              frozen.population(seed=seed + case_index),
              shield=shield,
              steps=steps,
              initial_state=state,
          )
          label = f"{selection}_{name}"
          summaries.append(
              dict(
                  result.summary,
                  population=label,
                  selection=selection,
                  scenario=name,
                  episode=case_index,
                  seed=seed + case_index,
                  experiment=frozen.experiment,
                  training_seed=frozen.metadata["seed"],
                  training_shield=False,
              )
          )
          for info in result.transitions:
            row = evaluate._trace_row(label, case_index, seed + case_index, config, info)  # pylint: disable=protected-access
            if writer is None:
              writer = csv.DictWriter(stream, fieldnames=list(row))
              writer.writeheader()
            writer.writerow(row)
  payload = {
      "checkpoint": str(path.resolve()),
      "config": ppo.config_to_dict(config),
      "training_metadata": prototype.metadata,
      "training_shield": False,
      "evaluation_weights_frozen": True,
      "episodes": summaries,
  }
  (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
  for selection in ("deterministic", "stochastic"):
    for shield in (False, True):
      cases = [
          s
          for s in summaries
          if s["selection"] == selection and s["shield"] == shield
      ]
      total = np.mean([s["apples_per_horizon_step"] for s in cases])
      focal = np.mean([s["apples_by_agent"][0] / steps for s in cases])
      print(
          f"{prototype.experiment} seed={prototype.metadata['seed']}"
          f" {selection} shield={shield}"
          f" survived={sum(s['survived'] for s in cases)}/{len(cases)}"
          f" apples/round={total:.3f} agent0/round={focal:.3f}",
          flush=True,
      )
  return payload


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--input", type=Path, default=Path("examples/results/spatial_commons/ppo")
  )
  parser.add_argument(
      "--checkpoint",
      type=Path,
      help="Evaluate one final.pt only (for one job of a Slurm array).",
  )
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--random-starts", type=int, default=4)
  parser.add_argument("--seed", type=int, default=532)
  args = parser.parse_args(argv)
  if args.random_starts < 0 or args.steps < 1:
    parser.error("steps must be positive; random-starts must be nonnegative")
  if args.checkpoint is not None:
    evaluate_checkpoint(
        args.checkpoint,
        args.checkpoint.parent / "evaluation",
        args.steps,
        args.random_starts,
        args.seed,
    )
    return
  # Combine only a completed sweep: the manifest lists the exact checkpoints.
  manifest_path = args.input / "training_manifest.json"
  if not manifest_path.exists():
    parser.error("Finish training first: training_manifest.json is missing")
  manifest = json.loads(manifest_path.read_text())
  combined = []
  for item in manifest["checkpoints"]:
    checkpoint = args.input / item
    summary_path = checkpoint.parent / "evaluation" / "summary.json"
    if summary_path.exists():
      result = json.loads(summary_path.read_text())
    else:
      result = evaluate_checkpoint(
          checkpoint,
          checkpoint.parent / "evaluation",
          args.steps,
          args.random_starts,
          args.seed,
      )
    combined.extend(result["episodes"])
  (args.input / "evaluation_summary.json").write_text(
      json.dumps(
          {
              "training_shield": False,
              "evaluation_weights_frozen": True,
              "evaluation_seed": args.seed,
              "horizon": args.steps,
              "random_starts": args.random_starts,
              "episodes": combined,
          },
          indent=2,
      )
      + "\n"
  )
  print(f"Combined {len(combined)} episodes.", flush=True)


if __name__ == "__main__":
  main()
