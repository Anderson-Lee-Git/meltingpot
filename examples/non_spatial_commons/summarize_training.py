"""Summarize completed frozen PPO evaluations and prepare example trajectories."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from examples.non_spatial_commons import ppo


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--input",
      type=Path,
      default=Path("examples/results/non_spatial_commons/ppo"),
  )
  args = parser.parse_args(argv)
  data = json.loads((args.input / "evaluation_summary.json").read_text())
  records = []
  lines = [
      "# Unshielded PPO training: frozen-policy evaluation",
      "",
      (
          "Nine runs: three experiments x three seeds, each with 200,704"
          " training transitions."
      ),
      (
          "All training was unshielded; final checkpoints were selected by the"
          " fixed budget, then frozen."
      ),
      "The uv-created isolated .venv uses Python 3.13.3 and CPU PyTorch 2.9.1.",
      "The implementation has 67 passing targeted tests.",
      "",
      (
          "Evaluation: 1,000-round horizon, two fixed starts plus ten held-out"
          " certified starts per checkpoint."
      ),
      (
          "Each selection/mode cell contains 36 episodes (three training seeds"
          " x twelve starts)."
      ),
      (
          "Stochastic evaluations use matching action-noise streams for the two"
          " shielding modes."
      ),
      "",
  ]
  for selection in ("deterministic", "stochastic"):
    lines += [
        "## " + selection.title() + " actions",
        "",
        (
            "| Experiment | Shield | Survival | Population harvest/round |"
            " Agent 0 harvest/round | Corrected rounds |"
        ),
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for experiment in ppo.EXPERIMENTS:
      for shield in (False, True):
        cases = [
            s
            for s in data["episodes"]
            if s["experiment"] == experiment
            and s["selection"] == selection
            and s["shield"] == shield
        ]
        mean = lambda key: sum(s[key] for s in cases) / len(cases)
        focal = sum(
            s["return_by_agent"][0] / s["horizon"] for s in cases
        ) / len(cases)
        row = {
            "experiment": experiment,
            "selection": selection,
            "shield": shield,
            "episodes": len(cases),
            "survived": sum(s["survived"] for s in cases),
            "population_harvest_per_round": mean("harvest_per_horizon_step"),
            "agent_0_harvest_per_round": focal,
            "corrected_round_fraction": mean("corrected_round_fraction"),
        }
        records.append(row)
        lines.append(
            f"| {experiment} | {'on' if shield else 'off'} |"
            f" {row['survived']}/{len(cases)} |"
            f" {row['population_harvest_per_round']:.3f} | {focal:.3f} |"
            f" {row['corrected_round_fraction']:.1%} |"
        )
    lines.append("")
  with (args.input / "metrics.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
  metadata = [
      json.loads(p.read_text())
      for p in args.input.glob("*/seed_*/metadata.json")
  ]
  total_time = sum(m["elapsed_seconds"] for m in metadata)
  lines += [
      "## Interpretation and limits",
      "",
      (
          "- All 216 shielded episodes survived. Unshielded survival was 66/108"
          " for deterministic policies and 26/108 for stochastic policies."
      ),
      (
          "- Training increased harvest returns but did not produce reliably"
          " safe policies across starts and action noise."
      ),
      (
          "- Individual policies optimize agent 0 only; their fixed partners"
          " retain their original requests. The cooperative policy optimizes"
          " total population harvest, not fairness."
      ),
      (
          "- This is a first fixed-budget PPO baseline. No optimality,"
          " equilibrium convergence, or statistical significance is claimed."
      ),
      (
          "- The final policies were used even where late training was"
          " unstable; there was no best-on-evaluation checkpoint selection."
      ),
      (
          "- The training episode limit was 256 rounds; evaluation extends to"
          " 1,000 rounds and includes initial energies outside the training"
          " range."
      ),
      (
          "- No reward penalty or shield was used during training. Unshielded"
          " execution still rations harvest to available stock."
      ),
      (
          "- Corrected rounds in unshielded mode mean physical rationing. In"
          " shielded mode they include safety corrections, both upward and"
          " downward."
      ),
      (
          "- Harvest/round uses the full requested evaluation horizon; rounds"
          " after failure contribute zero."
      ),
      (
          "- Finite rollouts are empirical evidence. The analytic safety"
          " argument depends on the invariant-core assumptions and the"
          " numerical guard documented in README.md."
      ),
      "",
      (
          f"The nine training runs used {sum(m['steps'] for m in metadata):,}"
          f" transitions and {total_time:.1f} seconds of summed training wall"
          " time on this Mac (excluding setup, tests, and evaluation)."
      ),
      "",
      "## Reproduce",
      "",
      "See TRAINING.md. Run training first, then evaluate_trained, then:",
      "",
      "```sh",
      ".venv/bin/python -m examples.non_spatial_commons.summarize_training",
      (
          ".venv/bin/python -m examples.non_spatial_commons.plot_training"
          " --input examples/results/non_spatial_commons/ppo"
      ),
      "```",
      "",
      (
          "Each seed folder retains initial.pt, final.pt, metadata.json,"
          " training.csv, and the full evaluation traces."
      ),
      (
          "The showcase plots use training seed 0, deterministic actions, and"
          " one specified starting state; they are not averages over seeds."
      ),
  ]
  (args.input / "RESULTS.md").write_text("\n".join(lines) + "\n")
  for scenario, episode in (("default", 0), ("recovery", 1)):
    folder = args.input / f"showcase_{scenario}"
    folder.mkdir(exist_ok=True)
    summaries, trace_rows = [], []
    for experiment in ppo.EXPERIMENTS:
      evaluation = args.input / experiment / "seed_0" / "evaluation"
      source = json.loads((evaluation / "summary.json").read_text())
      for item in source["episodes"]:
        if (
            item["selection"] == "deterministic"
            and item["scenario"] == scenario
        ):
          summaries.append(dict(item, population=experiment))
      with (evaluation / "trajectories.csv").open() as stream:
        for item in csv.DictReader(stream):
          if item["population"] == f"deterministic_{scenario}":
            trace_rows.append(dict(item, population=experiment))
    with (folder / "trajectories.csv").open("w", newline="") as stream:
      writer = csv.DictWriter(stream, fieldnames=list(trace_rows[0]))
      writer.writeheader()
      writer.writerows(trace_rows)
    payload = {
        key: source[key]
        for key in (
            "config",
            "critical_escapement",
            "protected_escapement",
            "msy",
            "msy_stock",
        )
    }
    payload["episodes"] = summaries
    (folder / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
  hashes = {
      str(p.relative_to(args.input)): hashlib.sha256(p.read_bytes()).hexdigest()
      for p in sorted(args.input.glob("*/seed_*/*.pt"))
  }
  (args.input / "checkpoint_sha256.json").write_text(
      json.dumps(hashes, indent=2) + "\n"
  )
  print(
      "Saved RESULTS.md, metrics.csv, checkpoint hashes, and seed-0 showcase"
      f" traces to {args.input}"
  )


if __name__ == "__main__":
  main()
