"""Tests of the start sampler and the baseline comparison runner."""

import csv

from examples.spatial_commons import baselines
from examples.spatial_commons import model
from examples.spatial_commons import starts
import numpy as np


def test_sampler_is_reproducible_and_in_backup_set():
  config = model.Config()
  a, stats = starts.sample_starts(config, 5, 7)
  b, _ = starts.sample_starts(config, 5, 7)
  assert a == b and stats.accepted == 5 and stats.attempts >= 5
  assert all(model.in_backup_set(config, s) for s in a)
  assert any(s.positions != config.homes for s in a)
  c, _ = starts.sample_starts(config, 5, 8)
  assert a != c


def test_starts_round_trip(tmp_path):
  config = model.Config()
  states, stats = starts.sample_starts(config, 2, 0)
  starts.save_starts(tmp_path / "s.json", states, stats, {"seed": 0})
  loaded, meta = starts.load_starts(tmp_path / "s.json")
  assert loaded == states and meta["accepted"] == 2


def test_evaluate_and_summarize(tmp_path):
  baselines.main(
      ["starts", "--count", "2", "--output", str(tmp_path / "starts.json")]
  )
  for method in ("random", "greedy"):
    baselines.main([
        "evaluate",
        "--method",
        method,
        "--steps",
        "30",
        "--starts",
        str(tmp_path / "starts.json"),
        "--output",
        str(tmp_path / "ep" / f"episodes_{method}.csv"),
    ])  # pyformat: disable
  with (tmp_path / "ep" / "episodes_random.csv").open() as stream:
    rows = list(csv.DictReader(stream))
  assert len(rows) == 4
  assert [r["shield"] for r in rows] == ["0", "1", "0", "1"]
  assert rows[0]["pair_seed"] == rows[1]["pair_seed"]
  for row in rows:
    if row["shield"] == "1":
      assert row["survived"] == "1"
    else:
      assert float(row["override_action_rate"]) == 0
  baselines.main(["summarize", "--input", str(tmp_path / "ep")])
  assert (tmp_path / "ep" / "table.tex").exists()
  summary, paired = baselines.aggregate(rows)
  assert len(summary) == 2 and len(paired) == 1
  assert np.isclose(
      summary[0]["total_apples_mean"],
      np.mean([float(r["total_apples"]) for r in rows if r["shield"] == "0"]),
  )


def test_sampler_min_energy():
  config = model.Config()
  states, _ = starts.sample_starts(config, 5, 3, min_energy=15)
  assert all(min(s.energies) >= 15 for s in states)


def test_ppo_ci_averages_seeds_per_start():
  rows = []
  for seed, offset in ((0, 0.0), (1, 10.0)):
    for j, value in enumerate((1.0, 3.0, 5.0)):
      for shield in (0, 1):
        rows.append({
            "method": "ippo_shared",
            "training_seed": seed,
            "start_index": j,
            "shield": shield,
            "total_apples": value + offset + 2 * shield,
            "apples_per_round": 0,
            "survived": 1,
            "override_action_rate": 0,
            "override_round_rate": 0,
            "failure_round": "",
        })  # pyformat: disable
  summary, paired = baselines.aggregate(rows)
  off = summary[0]
  # Seed-averaged per start: 6, 8, 10, so the mean is 8.
  assert np.isclose(off["total_apples_mean"], 8.0)
  assert np.isclose(off["total_apples_se"], np.std([6, 8, 10], ddof=1) / 3**0.5)
  assert off["per_seed"]["total_apples"] == {"0": 3.0, "1": 13.0}
  assert np.isclose(paired[0]["delta_total_apples_mean"], 2.0)
  assert paired[0]["starts"] == 3
