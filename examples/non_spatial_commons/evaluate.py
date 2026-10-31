"""Compare fixed populations with/without shielding and save auditable traces.

From the repository root:
  python -m examples.non_spatial_commons.evaluate --suite --steps 1000
"""

import argparse
from collections.abc import Sequence
import csv
import dataclasses
import json
import math
from pathlib import Path

from examples.non_spatial_commons import environment
from examples.non_spatial_commons import model
from examples.non_spatial_commons import policies


@dataclasses.dataclass(frozen=True)
class EpisodeResult:
  summary: dict
  transitions: tuple[environment.StepInfo, ...]


def run_episode(
    config: model.Config,
    population: Sequence[policies.Policy],
    *,
    shield: bool = True,
    steps: int = 1000,
    initial_state: model.State | None = None,
) -> EpisodeResult:
  """Shared evaluation loop for fixed policies or future trained callables."""
  if len(population) != config.num_agents:
    raise ValueError("Expected one policy per agent")
  env = environment.CommonsEnvironment(
      config, shield=shield, episode_length=steps, initial_state=initial_state
  )
  timestep = env.reset()
  records = []
  minimum_energy = min(env.state.energies)
  minimum_margin = model.core_margin(config, env.state)
  while not timestep.last():
    requests = tuple(
        policy(obs) for policy, obs in zip(population, timestep.observation)
    )
    timestep = env.step(requests)
    info = env.last_info
    records.append(info)
    minimum_energy = min(minimum_energy, *info.after.energies)
    minimum_margin = min(minimum_margin, model.core_margin(config, info.after))
  returns = tuple(
      math.fsum(info.executed[i] for info in records)
      for i in range(config.num_agents)
  )
  corrected_rounds = sum(info.intervention_count > 0 for info in records)
  summary = {
      "shield": shield,
      "survived": not env.state.failed,
      "failure_round": env.state.round_index if env.state.failed else None,
      "steps_executed": len(records),
      "horizon": steps,
      "initial_resource": records[0].before.resource,
      "initial_energies": records[0].before.energies,
      "final_resource": env.state.resource,
      "final_energies": env.state.energies,
      "minimum_energy": minimum_energy,
      "minimum_core_margin": minimum_margin,
      "return_by_agent": returns,
      "total_harvest": math.fsum(returns),
      # After failure, the remaining horizon contributes zero. Do not compare
      # survivor-only averages with the full-horizon average of shielded agents.
      "harvest_per_horizon_step": math.fsum(returns) / steps,
      "mean_harvest_per_executed_step": math.fsum(returns) / len(records),
      "corrected_rounds": corrected_rounds,
      "corrected_round_fraction": corrected_rounds / len(records),
      "mean_correction_l2": (
          math.fsum(info.correction_l2 for info in records) / len(records)
      ),
      "forced_upward_agent_steps": sum(
          c > model.CORE_TOLERANCE for info in records for c in info.corrections
      ),
      "downward_agent_steps": sum(
          c < -model.CORE_TOLERANCE
          for info in records
          for c in info.corrections
      ),
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
      "resource_before": info.before.resource,
      "resource_after": info.after.resource,
      "escapement": info.escapement,
      "harvest": math.fsum(info.executed),
      "requested_harvest": math.fsum(info.requested),
      "cap": info.cap,
      "minimum_energy": min(info.after.energies),
      "core_margin": model.core_margin(config, info.after),
      "failed": info.after.failed,
      "corrected_agents": info.intervention_count,
      "correction_l2": info.correction_l2,
  }
  for i in range(config.num_agents):
    row[f"request_{i}"] = info.requested[i]
    row[f"executed_{i}"] = info.executed[i]
    row[f"energy_{i}"] = info.after.energies[i]
  return row


def _parser():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--num-agents", type=int, default=4)
  parser.add_argument("--capacity", type=float, default=100.0)
  parser.add_argument("--growth-rate", type=float, default=0.5)
  parser.add_argument("--metabolic-cost", type=float, default=1.0)
  parser.add_argument("--max-energy", type=float, default=10.0)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--episodes", type=int, default=5)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--initial-resource", type=float)
  parser.add_argument("--initial-energy", type=float)
  parser.add_argument("--shield", choices=("both", "on", "off"), default="both")
  group = parser.add_mutually_exclusive_group()
  group.add_argument("--suite", action="store_true")
  group.add_argument("--policies", nargs="+", choices=policies.POLICY_NAMES)
  parser.add_argument(
      "--output",
      type=Path,
      default=Path("examples/results/non_spatial_commons"),
  )
  return parser


def main(argv=None):
  parser = _parser()
  args = parser.parse_args(argv)
  if args.episodes < 1 or args.seed < 0:
    parser.error("episodes must be positive and seed nonnegative")
  try:
    config = model.Config(
        args.num_agents,
        args.capacity,
        args.growth_rate,
        args.metabolic_cost,
        args.max_energy,
    )
    initial = model.default_state(config)
    initial = model.State(
        initial.resource
        if args.initial_resource is None
        else args.initial_resource,
        initial.energies
        if args.initial_energy is None
        else ((args.initial_energy,) * config.num_agents),
    )
    # Validate before creating output files or running any condition.
    environment.CommonsEnvironment(
        config,
        shield=args.shield != "off",
        episode_length=args.steps,
        initial_state=initial,
    )
    populations = (
        {name: (name,) for name in policies.POLICY_NAMES}
        if args.suite
        else {"custom": tuple(args.policies or ["greedy"])}
    )
    if args.suite:
      mix = ("greedy", "maintenance", "msy_share", "reserve_target")
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
          # Recreate each population so paired runs use the same random stream.
          population = policies.make_population(names, config, seed=seed)
          result = run_episode(
              config,
              population,
              shield=shield,
              steps=args.steps,
              initial_state=initial,
          )
          summary = dict(
              result.summary,
              population=label,
              policy_names=names,
              episode=episode,
              seed=seed,
          )
          summaries.append(summary)
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
        yield_mean = math.fsum(
            s["harvest_per_horizon_step"] for s in cases
        ) / len(cases)
        correction = math.fsum(
            s["corrected_round_fraction"] for s in cases
        ) / len(cases)
        print(
            f"{label:15} shield={str(shield):5} "
            f" survived={survived}/{len(cases)} "
            f" harvest/horizon-step={yield_mean:.3f} "
            f" corrected={correction:.1%}"
        )
  payload = {
      "config": dataclasses.asdict(config),
      "critical_escapement": config.critical_escapement,
      "protected_escapement": config.protected_escapement,
      "msy": config.msy,
      "msy_stock": config.msy_stock,
      "reward": "executed_harvest",
      "unshielded_allocation": (
          "Euclidean projection onto the physical resource budget"
      ),
      "episodes": summaries,
  }
  (args.output / "summary.json").write_text(
      json.dumps(payload, indent=2) + "\n"
  )
  print(f"Saved summary.json and trajectories.csv to {args.output}")


if __name__ == "__main__":
  main()
