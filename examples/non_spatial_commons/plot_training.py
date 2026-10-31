"""Plot PPO training curves; bands show the range over training seeds."""

import argparse
import csv
from pathlib import Path

from examples.non_spatial_commons import ppo
import numpy as np


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args(argv)
  import matplotlib  # pylint: disable=g-import-not-at-top

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top

  fig, axes = plt.subplots(
      3, 3, figsize=(13, 9), layout="constrained", sharex=True
  )
  for column, experiment in enumerate(ppo.EXPERIMENTS):
    runs = []
    for path in sorted((args.input / experiment).glob("seed_*/training.csv")):
      with path.open() as stream:
        runs.append(list(csv.DictReader(stream)))
    if not runs:
      continue
    count = min(map(len, runs))
    steps = [int(row["steps"]) for row in runs[0][:count]]
    title = {
        "individual_maintenance": "Individual / maintenance partners",
        "individual_msy": "Individual / MSY partners",
        "cooperative": "Cooperative joint policy",
    }[experiment]
    axes[0, column].set_title(title, fontsize=11, weight="bold")
    for row, (key, label) in enumerate([
        ("recent_mean_return", "Episode harvest return"),
        ("recent_mean_length", "Episode length"),
        ("recent_failure_fraction", "Failure fraction"),
    ]):
      values = np.asarray(
          [[float(item[key]) for item in run[:count]] for run in runs]
      )
      for i, run in enumerate(runs):
        for j, item in enumerate(run[:count]):
          if int(item["completed_episodes"]) == 0:
            values[i, j] = np.nan
      valid = np.any(np.isfinite(values), axis=0)
      values = values[:, valid]
      plotted_steps = np.asarray(steps)[valid]
      axes[row, column].plot(
          plotted_steps, np.nanmean(values, axis=0), color="#237a69"
      )
      axes[row, column].fill_between(
          plotted_steps,
          np.nanmin(values, axis=0),
          np.nanmax(values, axis=0),
          color="#237a69",
          alpha=0.2,
      )
      axes[row, column].grid(alpha=0.2)
      if column == 0:
        axes[row, column].set_ylabel(label)
    axes[2, column].set_ylim(-0.02, 1.02)
    axes[2, column].set_xlabel("Environment transitions")
    axes[2, column].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
  fig.suptitle(
      "PPO training without shielding — mean and range over three"
      " seeds\nIndividual returns measure agent 0; cooperative returns measure"
      " total harvest",
      fontsize=13,
      weight="bold",
  )
  output = args.output or args.input / "training_curves.png"
  output.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output, dpi=170, bbox_inches="tight", pad_inches=0.15)
  plt.close(fig)
  print(f"Saved {output}")


if __name__ == "__main__":
  main()
