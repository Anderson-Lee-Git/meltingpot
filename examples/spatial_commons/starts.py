"""Random initial states in Omega_B^H (Assumption 2), by rejection sampling.

Proposal distribution, for a fixed Config (homes and constants fixed):

* Positions: M distinct tiles drawn uniformly without replacement from the
  whole grid, assigned to agents in draw order. Agents may start anywhere,
  including on another agent's home.
* Energies: independent uniform integers in {min_energy, ..., E_max}
  (default min_energy = 15).
* Ripeness: every tile independently is ripe (z = 1) with probability
  `ripe_prob`; otherwise its integer ripeness units are uniform in
  {0, ..., den - 1} (r = num / den), i.e. z uniform on the unripe grid
  {0, 1/den, ..., 1 - 1/den}. Tiles around homes are not special-cased.

A draw is accepted iff the backup rollout reaches Omega_core within `horizon`
rounds without starvation (model.in_backup_set). Accepted states are
therefore exactly the proposal distribution conditioned on Omega_B^H.

Everything is driven by one numpy Generator seeded with `seed`, so the list of
starts is a deterministic function of (config, count, seed, ripe_prob,
horizon, min_energy), and every method and filter condition can share it.
"""

import dataclasses
import json
from pathlib import Path

from examples.spatial_commons import model
import numpy as np

DEFAULT_RIPE_PROB = 0.2  # Same density as the default board (20 / 100).
DEFAULT_MIN_ENERGY = 15


@dataclasses.dataclass(frozen=True)
class SampleStats:
  accepted: int
  attempts: int

  @property
  def acceptance_rate(self) -> float:
    return self.accepted / self.attempts if self.attempts else float("nan")


def propose(
    config: model.Config,
    rng: np.random.Generator,
    ripe_prob: float,
    min_energy: int = 1,
) -> model.State:
  """One draw from the proposal distribution (not yet conditioned)."""
  if not 1 <= min_energy <= config.max_energy:
    raise ValueError("min_energy must lie in [1, E_max]")
  n, m, full = config.grid_size, config.num_agents, config.ripe_units
  cells = rng.choice(n * n, size=m, replace=False)
  positions = tuple((int(q) // n, int(q) % n) for q in cells)
  energies = tuple(
      int(e) for e in rng.integers(min_energy, config.max_energy + 1, size=m)
  )
  ripe = rng.random((n, n)) < ripe_prob
  units = rng.integers(0, full, size=(n, n))
  ripeness = np.where(ripe, full, units).astype(np.int64)
  return model.State(positions, energies, ripeness)


def sample_starts(
    config: model.Config,
    count: int,
    seed: int,
    *,
    ripe_prob: float = DEFAULT_RIPE_PROB,
    min_energy: int = DEFAULT_MIN_ENERGY,
    horizon: int = model.DEFAULT_HORIZON,
    max_attempts: int | None = None,
) -> tuple[list[model.State], SampleStats]:
  """Rejection-samples `count` states in Omega_B^H."""
  if count < 0 or not 0 <= ripe_prob <= 1:
    raise ValueError("count must be nonnegative and ripe_prob in [0, 1]")
  rng = np.random.default_rng(seed)
  max_attempts = max_attempts or 1000 * max(count, 1)
  states, attempts = [], 0
  while len(states) < count:
    if attempts >= max_attempts:
      raise RuntimeError(
          f"Only {len(states)}/{count} starts after {attempts} draws"
      )
    attempts += 1
    state = propose(config, rng, ripe_prob, min_energy)
    if model.in_backup_set(config, state, horizon):
      states.append(state)
  return states, SampleStats(len(states), attempts)


class StartStream:
  """Endless accepted starts from one seeded generator (for training)."""

  def __init__(
      self,
      config: model.Config,
      seed: int,
      *,
      ripe_prob: float = DEFAULT_RIPE_PROB,
      min_energy: int = DEFAULT_MIN_ENERGY,
      horizon: int = model.DEFAULT_HORIZON,
  ):
    self.config = config
    self.ripe_prob = ripe_prob
    self.min_energy = min_energy
    self.horizon = horizon
    self.rng = np.random.default_rng(seed)
    self.accepted = 0
    self.attempts = 0

  def __call__(self) -> model.State:
    while True:
      self.attempts += 1
      state = propose(self.config, self.rng, self.ripe_prob, self.min_energy)
      if model.in_backup_set(self.config, state, self.horizon):
        self.accepted += 1
        return state


def state_to_dict(state: model.State) -> dict:
  return {
      "positions": [list(p) for p in state.positions],
      "energies": list(state.energies),
      "ripeness_units": state.ripeness.tolist(),
  }


def state_from_dict(payload: dict) -> model.State:
  return model.State(
      tuple(tuple(p) for p in payload["positions"]),
      tuple(payload["energies"]),
      np.asarray(payload["ripeness_units"], dtype=np.int64),
  )


def save_starts(path: Path, states, stats: SampleStats, metadata: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(
      json.dumps(
          {
              **metadata,
              "accepted": stats.accepted,
              "attempts": stats.attempts,
              "acceptance_rate": stats.acceptance_rate,
              "starts": [state_to_dict(s) for s in states],
          },
          indent=1,
      )
      + "\n"
  )


def load_starts(path: Path) -> tuple[list[model.State], dict]:
  payload = json.loads(Path(path).read_text())
  states = [state_from_dict(s) for s in payload.pop("starts")]
  return states, payload
