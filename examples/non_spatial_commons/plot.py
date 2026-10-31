"""Plot paired trajectories emitted by evaluate.py (requires matplotlib)."""

import argparse
import csv
import json
from pathlib import Path


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--output", type=Path)
  parser.add_argument(
      "--populations", nargs="+", default=["greedy", "abstain", "mixed"]
  )
  parser.add_argument("--episode", type=int, default=0)
  parser.add_argument("--max-rounds", type=int, default=100)
  parser.add_argument(
      "--title",
      default=(
          "Non-spatial commons: fixed policies with and without the safety"
          " filter"
      ),
  )
  args = parser.parse_args(argv)
  if args.max_rounds < 1:
    parser.error("max-rounds must be positive")
  payload = json.loads((args.input / "summary.json").read_text())
  with (args.input / "trajectories.csv").open() as stream:
    rows = [
        row
        for row in csv.DictReader(stream)
        if int(row["episode"]) == args.episode
        and int(row["round"]) <= args.max_rounds
    ]
  available = {row["population"] for row in rows}
  if not set(args.populations) <= available:
    parser.error(
        f"Requested populations not in episode; available: {sorted(available)}"
    )

  import matplotlib  # pylint: disable=g-import-not-at-top

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

  plt.rcParams.update(
      {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
  )
  fig, axes = plt.subplots(
      3,
      len(args.populations),
      squeeze=False,
      figsize=(4.5 * len(args.populations), 9),
      sharex=True,
      layout="constrained",
  )
  fields = ("resource_after", "minimum_energy", "harvest")
  labels = ("Resource stock", "Minimum agent energy", "Aggregate harvest")
  for column, name in enumerate(args.populations):
    title = (
        "MSY share" if name == "msy_share" else name.replace("_", " ").title()
    )
    axes[0, column].set_title(title, weight="bold")
    for shield, color, label in [
        (False, "#c4543b", "Unshielded"),
        (True, "#237a69", "Shielded"),
    ]:
      series = [
          row
          for row in rows
          if row["population"] == name and (row["shield"] == "True") == shield
      ]
      if not series:
        continue
      summary = next(
          s
          for s in payload["episodes"]
          if s["population"] == name
          and s["episode"] == args.episode
          and s["shield"] == shield
      )
      times = [int(row["round"]) for row in series]
      for i, field in enumerate(fields):
        x = list(times)
        values = [float(row[field]) for row in series]
        if i < 2:
          x.insert(0, 0)
          values.insert(
              0,
              summary["initial_resource"]
              if i == 0
              else min(summary["initial_energies"]),
          )
        axes[i, column].plot(x, values, color=color, label=label, linewidth=1.8)
        if series[-1]["failed"] == "True":
          axes[i, column].plot(
              x[-1], values[-1], "x", color=color, markersize=7
          )
    axes[0, column].axhline(
        payload["msy_stock"],
        color="#777777",
        linestyle=":",
        linewidth=1,
        label="MSY stock",
    )
    axes[1, column].axhline(0, color="#777777", linestyle=":", linewidth=1)
    axes[2, column].axhline(
        payload["msy"], color="#777777", linestyle=":", linewidth=1, label="MSY"
    )
    for i in range(3):
      axes[i, column].grid(alpha=0.18)
      axes[i, column].set_xlim(0, args.max_rounds)
      if column == 0:
        axes[i, column].set_ylabel(labels[i])
    axes[2, column].set_xlabel("Round")
    axes[0, column].legend(fontsize=8, loc="best")
  fig.suptitle(
      args.title,
      fontsize=14,
      weight="bold",
  )
  output = args.output or args.input / "comparison.png"
  output.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output, dpi=170, bbox_inches="tight", pad_inches=0.15)
  plt.close(fig)
  print(f"Saved {output}")


if __name__ == "__main__":
  main()
