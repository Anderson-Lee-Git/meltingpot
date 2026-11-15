"""Randomized numerical checks of the Section 3 safety claims.

From the repository root:
  python -m examples.spatial_commons.verify_safety --quick

Checks, for several (homes, k, r, eta, delta, E_max) satisfying Assumption 1,
including border homes and homes at inf-distance exactly 2:

  core      Proposition 1: random s in Omega_core stay in Omega_core under
            all-stay (checked for every round of a multi-round rollout).
  rollout   The fast model.rounds_to_core equals the plain State-based
            rounds_to_core_reference on random (mostly non-core) states.
  backup    Proposition 2(3): random s in Omega_B^H satisfy
            f(s, pi_B(s)) in Omega_B^H, for H in {0, 5, 20, 64}.
  filter    Theorem 1: from random s_0 in Omega_B^H, the filtered closed loop
            stays in Omega_B^H (hence never fails) under random, greedy and
            adversarial proposals.

Any counterexample is printed and makes the script exit nonzero.
"""

import argparse
from fractions import Fraction
import itertools
import json
import sys
import time

from examples.spatial_commons import environment
from examples.spatial_commons import model
from examples.spatial_commons import policies
from examples.spatial_commons import starts
import numpy as np

S = model.STAY


def _layouts(n):
  """Named home layouts on an n x n grid (n >= 10)."""
  return {
      "default": model.DEFAULT_HOMES,
      "corners": ((0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)),
      "edges_dist2": ((0, 3), (0, 5), (2, 4), (n - 1, 4), (5, 0)),
      "lattice_dist2": tuple(
          (r, c) for r in range(0, n, 2) for c in range(0, n, 2)
      )[:9],
      "dense_dist2": ((2, 2), (2, 4), (4, 2), (4, 4), (6, 6), (6, 8)),
  }


def random_layout(rng, n, m):
  """m random homes with pairwise inf-distance >= 2 (border allowed)."""
  while True:
    homes = []
    for q in rng.permutation(n * n):
      h = (int(q) // n, int(q) % n)
      if all(max(abs(h[0] - a), abs(h[1] - b)) >= 2 for a, b in homes):
        homes.append(h)
        if len(homes) == m:
          return tuple(homes)


def parameter_settings():
  """(k, r, eta, delta, E_max) tuples satisfying Assumption 1's (2)-(3)."""
  out = []
  for k, r, eta, delta, e_max in [
      (4, Fraction(3, 200), 20, 1, 40),  # defaults, T = 17
      (4, Fraction(1, 76), 19, 1, 19),  # T*delta = eta, E_max = T
      (1, Fraction(1, 10), 10, 1, 10),  # T = 10
      (2, Fraction(1, 20), 12, 1, 15),  # T = 10
      (3, Fraction(1, 9), 6, 2, 5),  # T = 3, delta = 2, E_max = 2(T-1)+1
      (5, Fraction(1, 35), 14, 2, 30),  # T = 7
      (8, Fraction(1, 64), 24, 3, 22),  # T = 8, E_max = 3*7+1
      (6, Fraction(1, 6), 1, 1, 1),  # T = 1
      (3, Fraction(3, 100), 33, 3, 40),  # T = 12, delta*T = 36 > 33 (bad)
  ]:
    config = model.Config(
        homes=((0, 0),),
        growth_rate=r,
        apple_energy=eta,
        metabolic_cost=delta,
        max_energy=e_max,
        initial_energy=1,
        core_k=k,
    )
    if config.core_certified:
      out.append((k, r, eta, delta, e_max))
  return out


def make_config(homes, setting, n=10):
  k, r, eta, delta, e_max = setting
  return model.Config(
      grid_size=n,
      homes=homes,
      growth_rate=r,
      apple_energy=eta,
      metabolic_cost=delta,
      max_energy=e_max,
      initial_energy=1,
      core_k=k,
  )


def neighbours(config, h):
  n = config.grid_size
  r, c = h
  return [
      (r + dr, c + dc)
      for dr in (-1, 0, 1)
      for dc in (-1, 0, 1)
      if (dr or dc) and 0 <= r + dr < n and 0 <= c + dc < n
  ]


def random_core_state(config, rng, ripe_prob):
  """A random member of Omega_core, or None if Omega_core is empty."""
  n, full, k = config.grid_size, config.ripe_units, config.core_k
  if any(len(neighbours(config, h)) < k for h in config.homes):
    return None
  z = np.where(
      rng.random((n, n)) < ripe_prob, full, rng.integers(0, full, (n, n))
  ).astype(np.int64)
  for h in config.homes:
    nb = neighbours(config, h)
    count = int(rng.integers(k, len(nb) + 1))
    for j in rng.choice(len(nb), size=count, replace=False):
      z[nb[int(j)]] = full
  # Ripe tiles may make other homes' counts larger, never smaller. Home tiles
  # get any ripeness, then energies uniform on (delta * tau, E_max].
  energies = []
  for h in config.homes:
    for _ in range(100):
      z[h] = full if rng.random() < 0.3 else int(rng.integers(0, full))
      low = config.metabolic_cost * model.burn_rounds(config, z[h]) + 1
      if low <= config.max_energy:
        break
    else:
      return None
    energies.append(int(rng.integers(low, config.max_energy + 1)))
  state = model.State(config.homes, tuple(energies), z)
  assert model.in_core(config, state), "constructed state not in core"
  return state


class Failure(Exception):
  pass


def random_backup_states(config, rng, count, horizon, max_draws=20_000):
  """Random members of Omega_B^H from a mixture of three proposals.

  (i) the start sampler's proposal, (ii) a random core state, (iii) a random
  core state followed by 1-15 unfiltered rounds of uniform random moves.
  Each draw is kept iff it is in Omega_B^H.
  """
  out = []
  for _ in range(max_draws):
    if len(out) == count:
      break
    kind = rng.integers(3)
    if kind == 0:
      state = starts.propose(config, rng, float(rng.uniform(0.1, 0.6)))
    else:
      state = random_core_state(config, rng, float(rng.uniform(0, 0.5)))
      if state is None:
        continue
      if kind == 2:
        for _ in range(int(rng.integers(1, 16))):
          if state.failed:
            break
          moves = rng.integers(model.NUM_ACTIONS, size=config.num_agents)
          state = model.transition(config, state, tuple(int(a) for a in moves))
        state = model.State(state.positions, state.energies, state.ripeness)
    if not state.failed and model.in_backup_set(config, state, horizon):
      out.append(state)
  return out


def check_core(rng, configs, samples, rounds):
  checked = 0
  stay_cache = {}
  for config in configs:
    stay = stay_cache.setdefault(config.num_agents, (S,) * config.num_agents)
    for _ in range(samples):
      state = random_core_state(config, rng, float(rng.uniform(0, 0.6)))
      if state is None:
        continue
      for t in range(rounds):
        nxt = model.transition(config, state, stay)
        checked += 1
        if not model.in_core(config, nxt):
          raise Failure(
              f"core: all-stay left Omega_core at t={t} for {config}:"
              f" {starts.state_to_dict(state)}"
          )
        state = nxt
  return {"core_transitions_checked": checked}


def check_rollout(rng, configs, samples):
  agree, reached = 0, 0
  for config in configs:
    for horizon in (0, 3, 20, 64):
      for _ in range(samples):
        state = starts.propose(config, rng, float(rng.uniform(0, 0.7)))
        if rng.random() < 0.3:
          core = random_core_state(config, rng, 0.2)
          if core is not None:
            state = core
        fast = model.rounds_to_core(config, state, horizon)
        slow = model.rounds_to_core_reference(config, state, horizon)
        if fast != slow:
          raise Failure(
              f"rollout: fast={fast} reference={slow} for {config},"
              f" H={horizon}: {starts.state_to_dict(state)}"
          )
        agree += 1
        reached += fast is not None
  return {"rollout_pairs_agree": agree, "rollout_pairs_in_backup": reached}


def check_backup(rng, configs, samples):
  checked = 0
  for config in configs:
    for horizon in (0, 5, 20, 64):
      for state in random_backup_states(config, rng, samples, horizon):
        nxt = model.transition(
            config, state, model.backup_actions(config, state)
        )
        checked += 1
        if not model.in_backup_set(config, nxt, horizon):
          raise Failure(
              f"backup: f(s, pi_B(s)) left Omega_B^{horizon} for {config}:"
              f" {starts.state_to_dict(state)}"
          )
  return {"backup_transitions_checked": checked}


class AwayPolicy:
  """Adversary: moves away from its home (reverse of pi_B), else random."""

  def __init__(self, config, seed):
    self.config = config
    self.rng = np.random.default_rng(seed)

  def __call__(self, observation):
    i = int(observation["PLAYER_INDEX"])
    r, c = (int(v) for v in observation["POSITIONS"][i])
    hr, hc = self.config.homes[i]
    options = []
    if r <= hr:
      options.append(model.UP)
    if r >= hr:
      options.append(model.DOWN)
    if c <= hc:
      options.append(model.LEFT)
    if c >= hc:
      options.append(model.RIGHT)
    return int(self.rng.choice(options))


def check_filter(rng, configs, episodes, rounds):
  rounds_checked, overrides = 0, 0
  for config in configs:
    horizon = int(rng.choice([8, 20, 64]))
    states = random_backup_states(config, rng, episodes, horizon)
    for e, state in enumerate(states):
      kind = ("random", "greedy", "away")[e % 3]
      seed = int(rng.integers(2**31))
      if kind == "away":
        population = [
            AwayPolicy(config, seed + i) for i in range(config.num_agents)
        ]
      else:
        population = policies.make_population((kind,), config, seed=seed)
      env = environment.SpatialCommonsEnvironment(
          config,
          shield=True,
          horizon=horizon,
          episode_length=rounds,
          initial_state=state,
      )
      timestep = env.reset()
      while not timestep.last():
        timestep = env.step(
            tuple(p(o) for p, o in zip(population, timestep.observation))
        )
        rounds_checked += 1
        overrides += env.last_info.intervention_count
        if env.state.failed or not model.in_backup_set(
            config, env.state, horizon
        ):
          raise Failure(
              f"filter: closed loop left Omega_B^{horizon} ({kind}) for"
              f" {config} at round {env.state.round_index} from"
              f" {starts.state_to_dict(state)}"
          )
  return {
      "filter_rounds_checked": rounds_checked,
      "filter_agent_overrides": overrides,
  }


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--quick", action="store_true")
  parser.add_argument(
      "--only", nargs="+", choices=("core", "rollout", "backup", "filter")
  )
  args = parser.parse_args(argv)
  rng = np.random.default_rng(args.seed)
  scale = 1 if args.quick else 5
  n = 10
  layouts = dict(_layouts(n))
  for j in range(3):
    layouts[f"random_{j}"] = random_layout(rng, n, int(rng.integers(2, 7)))
  settings = parameter_settings()
  configs = []
  for (name, homes), setting in itertools.product(layouts.items(), settings):
    config = make_config(homes, setting, n)
    if config.core_certified:
      configs.append(config)
  # The filter check is expensive; use the 4-6 agent layouts only.
  small = [c for c in configs if c.num_agents <= 6]
  summary = {
      "layouts": {k: [list(h) for h in v] for k, v in layouts.items()},
      "settings_k_r_eta_delta_emax": [
          [k, str(r), eta, d, e] for k, r, eta, d, e in settings
      ],
      "configs": len(configs),
  }
  checks = {
      "core": lambda: check_core(rng, configs, 40 * scale, 60),
      "rollout": lambda: check_rollout(rng, configs, 10 * scale),
      "backup": lambda: check_backup(rng, configs, 10 * scale),
      "filter": lambda: check_filter(rng, small, 3 * scale, 60),
  }
  ok = True
  for name, check in checks.items():
    if args.only and name not in args.only:
      continue
    start = time.perf_counter()
    try:
      result = check()
      result[f"{name}_seconds"] = round(time.perf_counter() - start, 1)
      summary.update(result)
      print(name, "OK", result, flush=True)
    except Failure as error:
      ok = False
      summary[f"{name}_counterexample"] = str(error)
      print(name, "COUNTEREXAMPLE", error, flush=True)
  print(json.dumps({k: v for k, v in summary.items() if k != "layouts"}))
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
