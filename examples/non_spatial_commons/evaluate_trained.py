"""Evaluate frozen unshielded-trained checkpoints with and without shielding."""

import argparse
import csv
import dataclasses
import json
from pathlib import Path

from examples.non_spatial_commons import evaluate
from examples.non_spatial_commons import model
from examples.non_spatial_commons import ppo
import numpy as np
import torch


def evaluation_starts(config, random_starts, seed):
  """Matched starts: two fixed examples plus held-out viable states."""
  states = [
      ("default", model.default_state(config)),
      ("recovery", model.State(13.3, (0.0,) * config.num_agents)),
  ]
  rng = np.random.default_rng(seed)
  for i in range(random_starts):
    energies = tuple(rng.uniform(0, config.max_energy, size=config.num_agents))
    b = sum(model.metabolic_floors(config, model.State(0, energies)))
    lower = config.protected_escapement + b + 0.01
    state = model.State(
        float(rng.uniform(lower, config.carrying_capacity)), energies
    )
    states.append((f"heldout_{i:02}", state))
  return states


def evaluate_checkpoint(path, output, steps=1000, random_starts=10, seed=532):
  torch.set_num_threads(2)
  prototype = ppo.FrozenPolicy(path)
  config = prototype.config
  output.mkdir(parents=True, exist_ok=True)
  summaries = []
  starts = evaluation_starts(config, random_starts, seed)
  with (output / "trajectories.csv").open("w", newline="") as stream:
    writer = None
    for selection in ("deterministic", "stochastic"):
      for case_index, (name, state) in enumerate(starts):
        for shield in (False, True):
          # New instances with matching evaluation RNGs prevent training updates
          # and preserve pairing across modes, even when episode lengths differ.
          frozen = ppo.FrozenPolicy(
              path, stochastic=selection == "stochastic", seed=seed + case_index
          )
          result = evaluate.run_episode(
              config,
              frozen.population(),
              shield=shield,
              steps=steps,
              initial_state=state,
          )
          label = f"{selection}_{name}"
          summary = dict(
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
          summaries.append(summary)
          for info in result.transitions:
            row = evaluate._trace_row(label, case_index, seed + case_index, config, info)  # pylint: disable=protected-access
            if writer is None:
              writer = csv.DictWriter(stream, fieldnames=list(row))
              writer.writeheader()
            writer.writerow(row)
  payload = {
      "checkpoint": str(path.resolve()),
      "config": dataclasses.asdict(config),
      "critical_escapement": config.critical_escapement,
      "protected_escapement": config.protected_escapement,
      "msy": config.msy,
      "msy_stock": config.msy_stock,
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
      total = sum(s["harvest_per_horizon_step"] for s in cases) / len(cases)
      focal = sum(s["return_by_agent"][0] / steps for s in cases) / len(cases)
      print(
          f"{prototype.experiment} seed={prototype.metadata['seed']}"
          f" {selection} shield={shield}"
          f" survived={sum(s['survived'] for s in cases)}/{len(cases)}"
          f" total_harvest/step={total:.3f} learner_harvest/step={focal:.3f}",
          flush=True,
      )
  return payload


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--input",
      type=Path,
      default=Path("examples/results/non_spatial_commons/ppo"),
  )
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--random-starts", type=int, default=10)
  parser.add_argument("--seed", type=int, default=532)
  args = parser.parse_args(argv)
  checkpoints = sorted(args.input.glob("*/seed_*/final.pt"))
  if not checkpoints:
    parser.error("No final.pt checkpoints found")
  if args.random_starts < 0 or args.steps < 1:
    parser.error("steps must be positive; random-starts must be nonnegative")
  # Do not evaluate a partial training sweep. A manifest is written only after
  # every requested run finishes, and lists the exact checkpoints to evaluate.
  manifest_path = args.input / "training_manifest.json"
  if not manifest_path.exists():
    parser.error(
        "Finish the training sweep first: training_manifest.json is missing"
    )
  manifest = json.loads(manifest_path.read_text())
  expected = {
      str((args.input / item).resolve()) for item in manifest["checkpoints"]
  }
  if {str(p.resolve()) for p in checkpoints} != expected:
    parser.error(
        "Final checkpoint set differs from the completed training manifest"
    )
  combined = []
  for checkpoint in checkpoints:
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
  print(
      f"Completed frozen-policy evaluation: {len(combined)} episodes.",
      flush=True,
  )


if __name__ == "__main__":
  main()
