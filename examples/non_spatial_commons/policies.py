"""Fixed per-agent policy baselines; trained callables can use the same API."""

import dataclasses
from typing import Protocol

from examples.non_spatial_commons import environment
from examples.non_spatial_commons import model
import numpy as np

POLICY_NAMES = (
    "maintenance",
    "msy_share",
    "greedy",
    "reserve_target",
    "random",
    "abstain",
)


class Policy(Protocol):
  """One observation in, one uncertified scalar harvest in [0,K] out."""

  def __call__(self, observation: environment.Observation) -> float:
    ...


@dataclasses.dataclass(frozen=True)
class ConstantPolicy:
  amount: float

  def __call__(self, observation: environment.Observation) -> float:
    del observation
    return self.amount


@dataclasses.dataclass(frozen=True)
class ReserveTargetPolicy:
  """Replenishes reserves towards E_max/2 rather than maximizing harvest."""

  config: model.Config
  gain: float = 0.5

  def __call__(self, observation: environment.Observation) -> float:
    energy = float(observation["ENERGY"])
    amount = self.config.metabolic_cost + self.gain * (
        self.config.max_energy / 2 - energy
    )
    return float(np.clip(amount, 0, self.config.carrying_capacity))


class RandomPolicy:
  """Seeded exploratory baseline with an independent generator per agent."""

  def __init__(self, capacity: float, seed: int):
    self._capacity = capacity
    self._rng = np.random.default_rng(seed)

  def __call__(self, observation: environment.Observation) -> float:
    del observation
    return float(self._rng.uniform(0, self._capacity))


def make_policy(name: str, config: model.Config, *, seed: int = 0) -> Policy:
  if name == "maintenance":
    return ConstantPolicy(config.metabolic_cost)
  if name == "msy_share":
    return ConstantPolicy(config.msy / config.num_agents)
  if name == "greedy":
    return ConstantPolicy(config.carrying_capacity)
  if name == "abstain":
    return ConstantPolicy(0.0)
  if name == "reserve_target":
    return ReserveTargetPolicy(config)
  if name == "random":
    return RandomPolicy(config.carrying_capacity, seed)
  raise ValueError(f"Unknown policy {name!r}; choose from {POLICY_NAMES}")


def make_population(
    names: tuple[str, ...], config: model.Config, *, seed: int = 0
) -> tuple[Policy, ...]:
  """A single name is broadcast; otherwise give exactly one name per agent."""
  if len(names) == 1:
    names = names * config.num_agents
  if len(names) != config.num_agents:
    raise ValueError("Use one policy name or exactly num_agents policy names")
  # SeedSequence avoids overlapping ad hoc streams and supports reproducibility.
  seeds = np.random.SeedSequence(seed).spawn(config.num_agents)
  return tuple(
      make_policy(name, config, seed=int(child.generate_state(1)[0]))
      for name, child in zip(names, seeds)
  )
