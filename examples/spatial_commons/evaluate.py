"""Compare fixed populations with/without the backup-set filter.

From the repository root:
  python -m examples.spatial_commons.evaluate --suite --steps 1000
"""

import argparse
from collections.abc import Sequence
import csv
import dataclasses
import json
from pathlib import Path

from examples.spatial_commons import environment
from examples.spatial_commons import model
from examples.spatial_commons import policies


@dataclasses.dataclass(frozen=True)
class EpisodeResult:
  summary: dict
  transitions: tuple[environment.StepInfo, ...]


def run_episode(
    config: model.Config,
    population: Sequence[policies.Policy],
    *,
    shield: bool = True,
    horizon: int = model.DEFAULT_HORIZON,
    steps: int = 1000,
    initial_state: model.State | None = None,
) -> EpisodeResult:
  """Shared evaluation loop for fixed policies or future trained callables."""
  if len(population) != config.num_agents:
    raise ValueError("Expected one policy per agent")
  env = environment.SpatialCommonsEnvironment(
      config,
      shield=shield,
      horizon=horizon,
      episode_length=steps,
      initial_state=initial_state,
  )
  timestep = env.reset()
  initial = env.state
  records = []
  minimum_energy = min(initial.energies)
  while not timestep.last():
    proposals = tuple(
        policy(obs) for policy, obs in zip(population, timestep.observation)
    )
    timestep = env.step(proposals)
    info = env.last_info
    records.append(info)
    minimum_energy = min(minimum_energy, *info.after.energies)
  final = env.state
  returns = tuple(
      sum(info.eaten[i] for info in records) for i in range(config.num_agents)
  )
  corrected_rounds = sum(info.intervention_count > 0 for info in records)
  summary = {
      "shield": shield,
      "survived": not final.failed,
      "failure_round": final.round_index if final.failed else None,
      "steps_executed": len(records),
      "horizon": steps,
      "initial_energies": initial.energies,
      "final_energies": final.energies,
      "minimum_energy": minimum_energy,
      "initial_ripe": int(initial.ripe(config).sum()),
      "final_ripe": int(final.ripe(config).sum()),
      "apples_by_agent": returns,
      "total_apples": sum(returns),
      # Failed runs contribute zero for the rest of the horizon.
      "apples_per_horizon_step": sum(returns) / steps,
      "corrected_rounds": corrected_rounds,
      "corrected_round_fraction": corrected_rounds / len(records),
      "overridden_agent_steps": sum(
          info.intervention_count for info in records
      ),
      "final_in_core": model.in_core(config, final),
  }
  env.close()
  return EpisodeResult(summary, tuple(records))


def _trace_row(label, episode, seed, config, info):
  row = {
      "population": label,
      "episode": episode,
      "seed": seed,
      "shield": info.shielded,
      "round": info.after.round_index,
      "ripe_after": int(info.after.ripe(config).sum()),
      "apples_eaten": sum(info.eaten),
      "minimum_energy": min(info.after.energies),
      "failed": info.after.failed,
      "overridden_agents": info.intervention_count,
      "rounds_to_core": info.rounds_to_core,
  }
  for i in range(config.num_agents):
    row[f"proposed_{i}"] = model.ACTION_NAMES[info.proposed[i]]
    row[f"executed_{i}"] = model.ACTION_NAMES[info.executed[i]]
    row[f"row_{i}"], row[f"col_{i}"] = info.after.positions[i]
    row[f"energy_{i}"] = info.after.energies[i]
  return row


def _parser():
  defaults = model.Config()
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--grid-size", type=int, default=defaults.grid_size)
  parser.add_argument(
      "--growth-rate",
      type=model.as_fraction,
      default=defaults.growth_rate,
      help="r as a decimal or fraction, e.g. 0.015 or 3/200",
  )
  parser.add_argument(
      "--metabolic-cost", type=int, default=defaults.metabolic_cost
  )
  parser.add_argument("--apple-energy", type=int, default=defaults.apple_energy)
  parser.add_argument("--max-energy", type=int, default=defaults.max_energy)
  parser.add_argument(
      "--initial-energy", type=int, default=defaults.initial_energy
  )
  parser.add_argument(
      "--core-k",
      type=int,
      default=defaults.core_k,
      help="Neighbour threshold k of Omega_core (1-8).",
  )
  parser.add_argument("--horizon", type=int, default=model.DEFAULT_HORIZON)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--episodes", type=int, default=5)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--shield", choices=("both", "on", "off"), default="both")
  group = parser.add_mutually_exclusive_group()
  group.add_argument("--suite", action="store_true")
  group.add_argument("--policies", nargs="+", choices=policies.POLICY_NAMES)
  parser.add_argument(
      "--output", type=Path, default=Path("examples/results/spatial_commons")
  )
  return parser


def main(argv=None):
  parser = _parser()
  args = parser.parse_args(argv)
  if args.episodes < 1 or args.seed < 0:
    parser.error("episodes must be positive and seed nonnegative")
  try:
    config = model.Config(
        grid_size=args.grid_size,
        metabolic_cost=args.metabolic_cost,
        apple_energy=args.apple_energy,
        max_energy=args.max_energy,
        initial_energy=args.initial_energy,
        growth_rate=args.growth_rate,
        core_k=args.core_k,
    )
    environment.SpatialCommonsEnvironment(
        config,
        shield=args.shield != "off",
        horizon=args.horizon,
        episode_length=args.steps,
    )
    populations = (
        {name: (name,) for name in policies.POLICY_NAMES}
        if args.suite
        else {"custom": tuple(args.policies or ["greedy"])}
    )
    if args.suite:
      mix = ("greedy", "stay", "hungry", "random")
      populations["mixed"] = tuple(
          mix[i % len(mix)] for i in range(config.num_agents)
      )
    for names in populations.values():
      policies.make_population(names, config, seed=args.seed)
  except ValueError as error:
    parser.error(str(error))

  modes = (False, True) if args.shield == "both" else (args.shield == "on",)
  args.output.mkdir(parents=True, exist_ok=True)
  summaries = []
  with (args.output / "trajectories.csv").open("w", newline="") as trace:
    writer = None
    for label, names in populations.items():
      for episode in range(args.episodes):
        seed = args.seed + episode
        for shield in modes:
          population = policies.make_population(names, config, seed=seed)
          result = run_episode(
              config,
              population,
              shield=shield,
              horizon=args.horizon,
              steps=args.steps,
          )
          summaries.append(
              dict(
                  result.summary,
                  population=label,
                  policy_names=names,
                  episode=episode,
                  seed=seed,
              )
          )
          for info in result.transitions:
            row = _trace_row(label, episode, seed, config, info)
            if writer is None:
              writer = csv.DictWriter(trace, fieldnames=list(row))
              writer.writeheader()
            writer.writerow(row)
      for shield in modes:
        cases = [
            s
            for s in summaries
            if s["population"] == label and s["shield"] == shield
        ]
        survived = sum(s["survived"] for s in cases)
        apples = sum(s["apples_per_horizon_step"] for s in cases) / len(cases)
        corrected = sum(s["corrected_round_fraction"] for s in cases) / len(
            cases
        )
        print(
            f"{label:8} shield={str(shield):5}"
            f"  survived={survived}/{len(cases)}"
            f"  apples/step={apples:.3f}"
            f"  corrected={corrected:.1%}"
        )
  config_dict = dataclasses.asdict(config)
  config_dict["growth_rate"] = str(config.growth_rate)
  payload = {
      "config": config_dict,
      "core_cycle_T": config.core_cycle,
      "core_certified": config.core_certified,
      "backup_horizon": args.horizon,
      "reward": "apples_eaten",
      "movement": "simultaneous, vacating allowed, cycles blocked",
      "episodes": summaries,
  }
  (args.output / "summary.json").write_text(
      json.dumps(payload, indent=2) + "\n"
  )
  print(f"Saved summary.json and trajectories.csv to {args.output}")


if __name__ == "__main__":
  main()
