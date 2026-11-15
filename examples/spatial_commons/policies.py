"""Fixed per-agent policies; trained callables can use the same API."""

import dataclasses
from typing import Protocol

from examples.spatial_commons import environment
from examples.spatial_commons import model
import numpy as np

POLICY_NAMES = ("stay", "greedy", "hungry", "random")


class Policy(Protocol):
  """One observation in, one proposed move index out."""

  def __call__(self, observation: environment.Observation) -> int:
    ...


def step_towards(position, target) -> int:
  """Vertical first, then horizontal; STAY when already there."""
  (r, c), (tr, tc) = position, target
  if r < tr:
    return model.DOWN
  if r > tr:
    return model.UP
  if c < tc:
    return model.RIGHT
  if c > tc:
    return model.LEFT
  return model.STAY


def nearest_free_apple(observation: environment.Observation):
  """Closest ripe tile (Manhattan) not held by another agent, or None."""
  me = int(observation["PLAYER_INDEX"])
  positions = [tuple(p) for p in np.asarray(observation["POSITIONS"])]
  r, c = positions[me]
  others = {p for i, p in enumerate(positions) if i != me}
  rows, cols = np.nonzero(np.asarray(observation["APPLES"]))
  best = None
  for q in zip(rows.tolist(), cols.tolist()):
    if q in others:
      continue
    key = (abs(q[0] - r) + abs(q[1] - c), q)
    if best is None or key < best:
      best = key
  return None if best is None else best[1]


@dataclasses.dataclass(frozen=True)
class StayPolicy:
  """Resting baseline: never moves, eats whenever its own tile ripens."""

  def __call__(self, observation: environment.Observation) -> int:
    del observation
    return model.STAY


@dataclasses.dataclass(frozen=True)
class GreedyPolicy:
  """Always heads for the nearest ripe apple; stays if none is left.

  With `hunger_threshold`, it only forages while its energy is at or below the
  threshold and otherwise stays.
  """

  hunger_threshold: int | None = None

  def __call__(self, observation: environment.Observation) -> int:
    if (
        self.hunger_threshold is not None
        and int(observation["ENERGY"]) > self.hunger_threshold
    ):
      return model.STAY
    target = nearest_free_apple(observation)
    if target is None:
      return model.STAY
    me = int(observation["PLAYER_INDEX"])
    return step_towards(tuple(observation["POSITIONS"][me]), target)


class RandomPolicy:
  """Uniform random moves with an independent seeded stream."""

  def __init__(self, seed: int):
    self._rng = np.random.default_rng(seed)

  def __call__(self, observation: environment.Observation) -> int:
    del observation
    return int(self._rng.integers(model.NUM_ACTIONS))


def make_policy(name: str, config: model.Config, *, seed: int = 0) -> Policy:
  if name == "stay":
    return StayPolicy()
  if name == "greedy":
    return GreedyPolicy()
  if name == "hungry":
    return GreedyPolicy(hunger_threshold=config.max_energy // 2)
  if name == "random":
    return RandomPolicy(seed)
  raise ValueError(f"Unknown policy {name!r}; choose from {POLICY_NAMES}")


def make_population(
    names: tuple[str, ...], config: model.Config, *, seed: int = 0
) -> tuple[Policy, ...]:
  """A single name is broadcast; otherwise give exactly one name per agent."""
  if len(names) == 1:
    names = names * config.num_agents
  if len(names) != config.num_agents:
    raise ValueError("Use one policy name or exactly num_agents policy names")
  seeds = np.random.SeedSequence(seed).spawn(config.num_agents)
  return tuple(
      make_policy(name, config, seed=int(child.generate_state(1)[0]))
      for name, child in zip(names, seeds)
  )
