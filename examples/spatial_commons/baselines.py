"""Baselines with the safety filter off and on, from shared random starts.

Subcommands (run from the repository root with PYTHONPATH=.):

  starts     Sample the shared evaluation starts in Omega_B^H (starts.py) and
             save them to JSON.
  evaluate   Play one method (one checkpoint for PPO) from a range of the
             saved starts, filter off and on, and write per-episode rows.
  summarize  Combine every episode file under a directory into
             episodes.csv, summary.csv, summary.json, table.md and table.tex.

Methods:
  random       Each agent proposes one of the 5 moves uniformly at random
               (independent seeded stream per agent).
  greedy       policies.GreedyPolicy: head for the ripe tile nearest in
               Manhattan distance that no other agent stands on (ties: lowest
               row, then column), vertical moves first; stay if none.
               Ignores its energy. Deterministic.
  ippo_shared  Independent PPO with parameter sharing: one network acts for
               every agent on that agent's own observation (decentralised
               execution), each agent rewarded for its own apples
               (ppo experiment `selfish_shared`).
  ippo_shared_team  As ippo_shared, but every agent is rewarded with the
               team reward, total apples / M (`cooperative`).
  ippo_hetero  Independent PPO with one network per agent, own-apple reward
               (`selfish_independent`).
  ippo_hetero_team  One network per agent, team reward
               (`cooperative_independent`).
  central_ppo  Centralized PPO: one network on the joint state outputs the
               joint action with a factorised per-agent head over a shared
               encoder, trained on the team reward (`centralized`).
  PPO checkpoints act stochastically (sampled) by default.

Pairing: start j uses policy seed `seed + j` in both filter conditions, so
the filter-off and filter-on episodes see the same start, the same policy
weights and the same initial RNG state.

Per-episode metrics (horizon = --steps rounds; an episode stops early only
on starvation):
  total_apples        apples eaten by all agents = sum of environment rewards
                      (the undiscounted episode return of the team).
  apples_agent_i      apples eaten by agent i (its own undiscounted return).
  apples_per_round    total_apples / horizon (failed episodes score zero for
                      the remaining rounds).
  override_action_rate  overridden agent-actions / (M * rounds executed).
  override_round_rate   rounds with >= 1 overridden agent / rounds executed.
                      Both are 0 with the filter off.
  survived            1 if no agent starved within the horizon.
  failure_round       round index of the first failed state, else empty.
"""

import argparse
import collections
import csv
import json
import math
from pathlib import Path
import time

from examples.spatial_commons import environment
from examples.spatial_commons import model
from examples.spatial_commons import policies
from examples.spatial_commons import starts as starts_lib
import numpy as np

FIXED_METHODS = ("random", "greedy")
PPO_METHODS = {
    "ippo_shared": "selfish_shared",
    "ippo_shared_team": "cooperative",
    "ippo_hetero": "selfish_independent",
    "ippo_hetero_team": "cooperative_independent",
    "central_ppo": "centralized",
}
METHODS = FIXED_METHODS + tuple(PPO_METHODS)
METHOD_LABELS = {
    "random": "Random",
    "greedy": "Greedy",
    "ippo_shared": "IPPO shared (own)",
    "ippo_shared_team": "IPPO shared (team)",
    "ippo_hetero": "IPPO hetero (own)",
    "ippo_hetero_team": "IPPO hetero (team)",
    "central_ppo": "Centralized PPO",
}
METRICS = (
    "total_apples",
    "apples_per_round",
    "survived",
    "override_action_rate",
    "override_round_rate",
)
# Two-sided 95% Student t quantiles; normal beyond 30 degrees of freedom.
_T975 = (
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)  # pyformat: disable


def t975(df):
  if df < 1:
    return float("nan")
  return _T975[df - 1] if df <= len(_T975) else 1.96


# ----------------------------------------------------------------------------
# Joint policies: (observations, state) -> joint proposal.


class PerAgent:
  """Decentralised execution: agent i acts on its own observation only."""

  def __init__(self, population):
    self.population = population

  def __call__(self, observations, state):
    del state
    return tuple(p(o) for p, o in zip(self.population, observations))


class Centralized:
  """Joint-state policy (the centralized PPO controller)."""

  def __init__(self, frozen):
    self.frozen = frozen

  def __call__(self, observations, state):
    del observations
    return self.frozen.act_joint(state)


def make_joint_policy(method, config, seed, frozen=None):
  """A fresh joint policy whose randomness is fully determined by `seed`."""
  if method in FIXED_METHODS:
    return PerAgent(policies.make_population((method,), config, seed=seed))
  frozen.reseed(seed)
  if frozen.centralized:
    return Centralized(frozen)
  return PerAgent(frozen.population(seed=seed))


# ----------------------------------------------------------------------------
# Episodes.


def run_episode(config, joint_policy, initial_state, *, shield, horizon, steps):
  env = environment.SpatialCommonsEnvironment(
      config,
      shield=shield,
      horizon=horizon,
      episode_length=steps,
      initial_state=initial_state,
  )
  timestep = env.reset()
  m = config.num_agents
  apples = [0] * m
  rounds = override_actions = override_rounds = 0
  minimum_energy = min(initial_state.energies)
  start = time.perf_counter()
  while not timestep.last():
    timestep = env.step(joint_policy(timestep.observation, env.state))
    info = env.last_info
    rounds += 1
    for i, h in enumerate(info.eaten):
      apples[i] += h
    override_actions += info.intervention_count
    override_rounds += info.intervention_count > 0
    minimum_energy = min(minimum_energy, *info.after.energies)
  final = env.state
  row = {
      "shield": int(shield),
      "survived": int(not final.failed),
      "failure_round": final.round_index if final.failed else "",
      "rounds_executed": rounds,
      "total_apples": sum(apples),
      "apples_per_round": sum(apples) / steps,
      "override_action_rate": override_actions / (m * rounds),
      "override_round_rate": override_rounds / rounds,
      "override_actions": override_actions,
      "override_rounds": override_rounds,
      "minimum_energy": minimum_energy,
      "final_in_core": int(model.in_core(config, final)),
      "seconds": round(time.perf_counter() - start, 3),
  }
  for i in range(m):
    row[f"apples_agent_{i}"] = apples[i]
  return row


def evaluate(args):
  states, meta = starts_lib.load_starts(args.starts)
  config = model.Config()
  frozen = None
  training_seed = ""
  if args.method in PPO_METHODS:
    if args.checkpoint is None:
      raise SystemExit("--checkpoint is required for PPO methods")
    # pylint: disable-next=g-import-not-at-top
    from examples.spatial_commons import ppo
    import torch  # pylint: disable=g-import-not-at-top

    torch.set_num_threads(1)
    frozen = ppo.FrozenPolicy(
        args.checkpoint,
        stochastic=args.selection == "stochastic",
        allow_shielded=True,
    )
    if frozen.experiment != PPO_METHODS[args.method]:
      raise SystemExit(
          f"{args.checkpoint} is a {frozen.experiment} checkpoint, not"
          f" {PPO_METHODS[args.method]}"
      )
    config = frozen.config
    training_seed = frozen.metadata["seed"]
  if ppo_config_mismatch(config, meta):
    raise SystemExit("Starts were sampled for a different Config")
  first, last = args.first_start, min(
      args.last_start or len(states), len(states)
  )
  shields = {"both": (False, True), "off": (False,), "on": (True,)}[args.shield]
  args.output.parent.mkdir(parents=True, exist_ok=True)
  rows = []
  with args.output.open("w", newline="") as stream:
    writer = None
    for j in range(first, last):
      for shield in shields:
        pair_seed = args.seed + j
        policy = make_joint_policy(args.method, config, pair_seed, frozen)
        row = {
            "method": args.method,
            "selection": (
                args.selection if args.method in PPO_METHODS else "fixed"
            ),
            "training_seed": training_seed,
            "training_shield": (
                int(frozen.training_shield) if frozen is not None else ""
            ),
            "start_index": j,
            "pair_seed": pair_seed,
            **run_episode(
                config,
                policy,
                states[j],
                shield=shield,
                horizon=args.horizon,
                steps=args.steps,
            ),
        }
        rows.append(row)
        if writer is None:
          writer = csv.DictWriter(stream, fieldnames=list(row))
          writer.writeheader()
        writer.writerow(row)
        stream.flush()
        print(
            f"{args.method} seed={training_seed} start={j}"
            f" shield={shield} survived={row['survived']}"
            f" apples={row['total_apples']}"
            f" overrides={row['override_round_rate']:.1%}"
            f" ({row['seconds']:.1f}s)",
            flush=True,
        )
  return rows


def ppo_config_mismatch(config, meta):
  stored = meta.get("config")
  return stored is not None and stored != config_to_dict(config)


def config_to_dict(config):
  return {
      "grid_size": config.grid_size,
      "homes": [list(h) for h in config.homes],
      "metabolic_cost": config.metabolic_cost,
      "apple_energy": config.apple_energy,
      "max_energy": config.max_energy,
      "growth_rate": str(config.growth_rate),
      "core_k": config.core_k,
  }


def make_starts(args):
  config = model.Config()
  states, stats = starts_lib.sample_starts(
      config,
      args.count,
      args.seed,
      ripe_prob=args.ripe_prob,
      min_energy=args.min_energy,
      horizon=args.horizon,
  )
  starts_lib.save_starts(
      args.output,
      states,
      stats,
      {
          "config": config_to_dict(config),
          "seed": args.seed,
          "ripe_prob": args.ripe_prob,
          "min_energy": args.min_energy,
          "horizon": args.horizon,
          "distribution": (
              "positions: distinct uniform tiles; energies: uniform"
              " {min_energy..E_max}; ripeness: ripe w.p. ripe_prob else uniform"
              " unripe units; accepted iff in Omega_B^H"
          ),
      },
  )
  print(
      f"Accepted {stats.accepted}/{stats.attempts} draws"
      f" (rate {stats.acceptance_rate:.3f}); wrote {args.output}"
  )


# ----------------------------------------------------------------------------
# Aggregation.


def _mean_se(values):
  values = np.asarray(values, dtype=float)
  n = len(values)
  if n == 0:
    return float("nan"), float("nan"), 0
  se = values.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
  return float(values.mean()), float(se), n


def _per_start(cases, value):
  """{start_index: mean over training seeds of value(row)}."""
  by_start = collections.defaultdict(list)
  for row in cases:
    by_start[int(row["start_index"])].append(value(row))
  return {j: float(np.mean(v)) for j, v in sorted(by_start.items())}


def _ci(units):
  mean, se, n = _mean_se(units)
  return mean, se, t975(n - 1) * se if n > 1 else float("nan"), n


def aggregate(rows):
  """Mean, SE and 95% CI per (method, shield), plus paired differences.

  The unit is the start for every method. For PPO, each start's metric is
  first averaged over the training seeds (all seeds share the starts), then
  the t-CI is taken across starts. Per-seed means (over starts) are kept for
  reference under `per_seed`.
  """
  groups = collections.defaultdict(list)
  for row in rows:
    groups[(row["method"], int(row["shield"]))].append(row)
  summary = []
  for (method, shield), cases in sorted(
      groups.items(), key=lambda kv: (METHODS.index(kv[0][0]), kv[0][1])
  ):
    seeds = sorted({row["training_seed"] for row in cases})
    out = {
        "method": method,
        "label": METHOD_LABELS[method],
        "shield": shield,
        "episodes": len(cases),
        "training_seeds": len(seeds) if method in PPO_METHODS else 0,
        "starts": len({row["start_index"] for row in cases}),
        "ci_unit": "start (seed-averaged)",
    }
    metrics = list(METRICS) + [
        k for k in cases[0] if k.startswith("apples_agent_")
    ]
    per_seed = {}
    for metric in metrics:
      units = list(_per_start(cases, lambda r: float(r[metric])).values())
      mean, se, ci, _ = _ci(units)
      out[f"{metric}_mean"] = mean
      out[f"{metric}_se"] = se
      out[f"{metric}_ci95"] = ci
      if method in PPO_METHODS:
        per_seed[metric] = {
            str(seed): float(
                np.mean([
                    float(r[metric])
                    for r in cases
                    if r["training_seed"] == seed
                ])
            )
            for seed in seeds
        }  # pyformat: disable
    failures = [
        int(row["failure_round"]) for row in cases if row["failure_round"] != ""
    ]
    out["failures"] = len(failures)
    out["failure_round_mean"] = (
        float(np.mean(failures)) if failures else float("nan")
    )
    out["failure_round_median"] = (
        float(np.median(failures)) if failures else float("nan")
    )
    if per_seed:
      out["per_seed"] = per_seed
    summary.append(out)
  # Paired filter-on minus filter-off differences on the same start and seed,
  # averaged over seeds per start, CI across starts.
  pairs = collections.defaultdict(dict)
  for row in rows:
    key = (row["method"], row["training_seed"], int(row["start_index"]))
    pairs[key][int(row["shield"])] = row
  diffs = collections.defaultdict(list)
  for (method, seed, j), pair in pairs.items():
    if 0 in pair and 1 in pair:
      diffs[method].append({
          "training_seed": seed,
          "start_index": j,
          "delta": (
              float(pair[1]["total_apples"]) - float(pair[0]["total_apples"])
          ),
      })
  paired = []
  for method in sorted(diffs, key=METHODS.index):
    items = diffs[method]
    units = list(_per_start(items, lambda d: d["delta"]).values())
    mean, se, ci, n = _ci(units)
    entry = {
        "method": method,
        "pairs": len(items),
        "starts": n,
        "ci_unit": "start (seed-averaged)",
        "delta_total_apples_mean": mean,
        "delta_total_apples_se": se,
        "delta_total_apples_ci95": ci,
    }
    if method in PPO_METHODS:
      by_seed = collections.defaultdict(list)
      for d in items:
        by_seed[str(d["training_seed"])].append(d["delta"])
      entry["per_seed"] = {k: float(np.mean(v)) for k, v in by_seed.items()}
    paired.append(entry)
  return summary, paired


def _fmt(mean, ci, digits=1, percent=False):
  if math.isnan(mean):
    return "--"
  scale = 100 if percent else 1
  text = f"{mean * scale:.{digits}f}"
  if not math.isnan(ci):
    text += f" ± {ci * scale:.{digits}f}"
  return text


def tables(summary, paired):
  """Markdown and LaTeX tables (mean ± 95% CI)."""
  header = [
      "Method",
      "Filter",
      "N",
      "Total apples",
      "Apples/round",
      "Survival (%)",
      "Overridden actions (%)",
      "Rounds w/ override (%)",
      "Mean failure round",
  ]
  body = []
  for s in summary:
    body.append([
        s["label"],
        "on" if s["shield"] else "off",
        str(s["episodes"]),
        _fmt(s["total_apples_mean"], s["total_apples_ci95"]),
        _fmt(s["apples_per_round_mean"], s["apples_per_round_ci95"], 3),
        _fmt(s["survived_mean"], s["survived_ci95"], 0, True),
        _fmt(
            s["override_action_rate_mean"],
            s["override_action_rate_ci95"],
            1,
            True,
        ),
        _fmt(
            s["override_round_rate_mean"],
            s["override_round_rate_ci95"],
            1,
            True,
        ),
        "--"
        if math.isnan(s["failure_round_mean"])
        else f"{s['failure_round_mean']:.0f} ({s['failures']})",
    ])
  md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
  md += ["| " + " | ".join(r) + " |" for r in body]
  md.append("")
  md.append("| Method | Pairs | Δ total apples (on − off) |")
  md.append("|---|---|---|")
  for p in paired:
    md.append(
        f"| {METHOD_LABELS[p['method']]} | {p['pairs']} |"
        f" {_fmt(p['delta_total_apples_mean'], p['delta_total_apples_ci95'])} |"
    )

  def tex(cell):
    return (
        cell.replace("±", r"$\pm$")
        .replace("%", r"\%")
        .replace("Δ", r"$\Delta$")
        .replace("−", "$-$")
    )

  lines = [
      "% Generated by examples/spatial_commons/baselines.py summarize.",
      "% Mean ± 95% t-CI across starts; PPO metrics are first averaged over",
      "% training seeds per start. Failure round: mean over failed episodes",
      "% (count).",
      r"\begin{tabular}{llrrrrrrr}",
      r"\toprule",
      " & ".join(tex(h) for h in header) + r" \\",
      r"\midrule",
  ]
  previous = None
  for r in body:
    if previous is not None and r[0] != previous:
      lines.append(r"\midrule")
    lines.append(" & ".join(tex(c) for c in r) + r" \\")
    previous = r[0]
  lines += [r"\bottomrule", r"\end{tabular}", ""]
  return "\n".join(md) + "\n", "\n".join(lines)


def summarize(args):
  rows = []
  files = sorted(args.input.rglob("episodes_*.csv"))
  if not files:
    raise SystemExit(f"No episodes_*.csv files under {args.input}")
  for path in files:
    with path.open() as stream:
      rows.extend(csv.DictReader(stream))
  output = args.output or args.input
  output.mkdir(parents=True, exist_ok=True)
  fields = list(rows[0])
  for row in rows:
    fields += [k for k in row if k not in fields]
  with (output / "episodes.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
  summary, paired = aggregate(rows)
  with (output / "summary.csv").open("w", newline="") as stream:
    # Nested per-seed numbers go to summary.json only.
    flat = [{k: v for k, v in s.items() if k != "per_seed"} for s in summary]
    writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
    writer.writeheader()
    writer.writerows(flat)
  (output / "summary.json").write_text(
      json.dumps(
          {
              "episode_files": [str(p) for p in files],
              "summary": summary,
              "paired_filter_effect": paired,
          },
          indent=2,
      )
      + "\n"
  )
  md, tex = tables(summary, paired)
  (output / "table.md").write_text(md)
  (output / "table.tex").write_text(tex)
  print(md)
  print(
      "Wrote episodes.csv, summary.csv, summary.json, table.md, table.tex"
      f" to {output}"
  )


def main(argv=None):
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  sub = parser.add_subparsers(dest="command", required=True)

  p = sub.add_parser("starts", help="Sample the shared evaluation starts.")
  p.add_argument("--count", type=int, default=100)
  p.add_argument("--seed", type=int, default=532)
  p.add_argument(
      "--ripe-prob", type=float, default=starts_lib.DEFAULT_RIPE_PROB
  )
  p.add_argument(
      "--min-energy",
      type=int,
      default=starts_lib.DEFAULT_MIN_ENERGY,
      help="Initial energies are uniform in {min_energy, ..., E_max}.",
  )
  p.add_argument("--horizon", type=int, default=model.DEFAULT_HORIZON)
  p.add_argument("--output", type=Path, required=True)

  p = sub.add_parser("evaluate", help="Evaluate one method, filter off/on.")
  p.add_argument("--method", choices=METHODS, required=True)
  p.add_argument("--checkpoint", type=Path)
  p.add_argument("--starts", type=Path, required=True)
  p.add_argument("--first-start", type=int, default=0)
  p.add_argument("--last-start", type=int, help="Exclusive; default all.")
  p.add_argument("--steps", type=int, default=1000)
  p.add_argument("--horizon", type=int, default=model.DEFAULT_HORIZON)
  p.add_argument("--seed", type=int, default=0, help="Pair seed base.")
  p.add_argument("--shield", choices=("both", "off", "on"), default="both")
  p.add_argument(
      "--selection",
      choices=("stochastic", "deterministic"),
      default="stochastic",
  )
  p.add_argument(
      "--output",
      type=Path,
      required=True,
      help="CSV path; name it episodes_*.csv for summarize.",
  )

  p = sub.add_parser("summarize", help="Aggregate episode CSVs into tables.")
  p.add_argument("--input", type=Path, required=True)
  p.add_argument("--output", type=Path)

  args = parser.parse_args(argv)
  if args.command == "starts":
    make_starts(args)
  elif args.command == "evaluate":
    evaluate(args)
  else:
    summarize(args)


if __name__ == "__main__":
  main()
