"""Known ecological/energy dynamics and the report's invariant-core filter.

All harvests and energies use the same resource units (efficiency is one).
The model stops at first starvation. Negative energy is never clamped away.
"""

from collections.abc import Sequence
import dataclasses
import math

import numpy as np

# Absolute reporting tolerance in resource units. Core membership does not use
# unsafe slack; the projection returns the feasible side of its bisection.
CORE_TOLERANCE = 1e-10


@dataclasses.dataclass(frozen=True)
class Config:
  """Parameters satisfying Assumption 1 of non_spatial.tex."""

  num_agents: int = 4
  carrying_capacity: float = 100.0
  growth_rate: float = 0.5
  metabolic_cost: float = 1.0
  max_energy: float = 10.0

  def __post_init__(self):
    if (
        isinstance(self.num_agents, bool)
        or not isinstance(self.num_agents, int)
        or self.num_agents < 1
    ):
      raise ValueError("num_agents must be a positive integer")
    for name in (
        "carrying_capacity",
        "growth_rate",
        "metabolic_cost",
        "max_energy",
    ):
      value = getattr(self, name)
      if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if self.growth_rate > 2:
      raise ValueError("The proved core requires 0 < growth_rate <= 2")
    if not self.maintenance_demand < self.msy:
      raise ValueError("MSY must strictly exceed total metabolic demand")
    # Fail early if growth/maintenance differences cannot be resolved in float64.
    _ = self.protected_escapement

  @property
  def maintenance_demand(self) -> float:
    return self.num_agents * self.metabolic_cost

  @property
  def msy(self) -> float:
    return self.carrying_capacity * (self.growth_rate / 4)

  @property
  def msy_stock(self) -> float:
    return self.carrying_capacity / 2 + self.msy

  @property
  def critical_escapement(self) -> float:
    # Rationalized smaller quadratic root avoids cancellation for small demand.
    root = math.sqrt(1 - self.maintenance_demand / self.msy)
    return (2 * self.maintenance_demand / self.growth_rate) / (1 + root)

  @property
  def protected_escapement(self) -> float:
    """Slightly tighter S_crit so the invariant boundary survives roundoff.

    rho'(S_crit) = r * sqrt(1 - D/MSY). Choose a stock buffer that creates
    roughly 64 ulps of K in regenerative surplus, not an arbitrary energy floor.
    The tighter core remains an analytic invariant subset because this threshold
    is below K/2 and its regeneration exceeds maintenance demand.
    """
    critical = self.critical_escapement
    slope = self.growth_rate * math.sqrt(1 - self.maintenance_demand / self.msy)
    buffer = 64 * math.ulp(self.carrying_capacity) / slope
    if (
        not math.isfinite(buffer)
        or buffer >= (self.carrying_capacity / 2 - critical) / 4
    ):
      raise ValueError(
          "Insufficient float64 resolution for the growth/maintenance gap"
      )
    return math.nextafter(critical + buffer, math.inf)


@dataclasses.dataclass(frozen=True)
class State:
  """Immutable pre-harvest state, also used for terminated observations."""

  resource: float
  energies: tuple[float, ...]
  round_index: int = 0

  def __post_init__(self):
    object.__setattr__(self, "resource", float(self.resource))
    object.__setattr__(self, "energies", tuple(float(e) for e in self.energies))

  @property
  def failed(self) -> bool:
    return any(energy < 0 for energy in self.energies)


def validate_state(config: Config, state: State) -> None:
  """Validates physical bounds, not membership in the invariant safe core."""
  if not math.isfinite(state.resource) or not (
      0 <= state.resource <= config.carrying_capacity
  ):
    raise ValueError("resource must be finite and within [0, K]")
  if len(state.energies) != config.num_agents:
    raise ValueError("There must be one energy reserve per agent")
  if any(
      not math.isfinite(e)
      or not -config.metabolic_cost <= e <= config.max_energy
      for e in state.energies
  ):
    raise ValueError("Energies must be finite and within [-delta, E_max]")
  if (
      not isinstance(state.round_index, int)
      or isinstance(state.round_index, bool)
      or state.round_index < 0
  ):
    raise ValueError("round_index must be a nonnegative integer")


def default_state(config: Config) -> State:
  """Starts at the MSY resource stock with half-full energy reserves."""
  return State(config.msy_stock, (config.max_energy / 2,) * config.num_agents)


def metabolic_floors(config: Config, state: State) -> np.ndarray:
  return np.maximum(0.0, config.metabolic_cost - np.asarray(state.energies))


def core_margin(config: Config, state: State) -> float:
  """Margin for the numerically buffered core; energies checked separately."""
  return state.resource - (
      config.protected_escapement + math.fsum(metabolic_floors(config, state))
  )


def in_safe_core(config: Config, state: State) -> bool:
  validate_state(config, state)
  return not state.failed and core_margin(config, state) >= 0


def harvest_cap(config: Config, state: State) -> float:
  return max(
      math.fsum(metabolic_floors(config, state)),
      state.resource - config.carrying_capacity / 2,
  )


def project_harvest(
    requests: Sequence[float], floors: Sequence[float], budget: float
) -> np.ndarray:
  """Euclidean projection onto {u_i >= floors_i, sum_i u_i <= budget}.

  Uses a feasible-side scalar bisection, not proportional rationing. At a
  saturated cap all agents above their floors receive the same downward shift.
  """
  requests = np.asarray(requests, dtype=np.float64)
  floors = np.asarray(floors, dtype=np.float64)
  if requests.ndim != 1 or requests.size == 0 or requests.shape != floors.shape:
    raise ValueError(
        "requests and floors must be nonempty vectors of equal size"
    )
  if not np.all(np.isfinite(requests)) or not np.all(np.isfinite(floors)):
    raise ValueError("Requests and floors must be finite")
  if np.any(requests < 0) or np.any(floors < 0):
    raise ValueError("Requests and floors must be nonnegative")
  if not math.isfinite(budget) or budget < 0:
    raise ValueError("budget must be finite and nonnegative")
  floor_total = math.fsum(floors)
  if floor_total > budget:
    raise ValueError(
        "Infeasible projection: metabolic floors exceed the budget"
    )
  if floor_total == budget:
    return floors.copy()
  raised = np.maximum(requests, floors)
  if math.fsum(raised) <= budget:
    return raised

  low = 0.0
  high = float(np.max(requests))
  # high is feasible, low is infeasible. Keep that invariant through rounding.
  for _ in range(80):
    middle = low + (high - low) / 2
    if middle == low or middle == high:
      break
    candidate = np.maximum(requests - middle, floors)
    if math.fsum(candidate) > budget:
      low = middle
    else:
      high = middle
  return np.maximum(requests - high, floors)


def validate_requests(config: Config, requests: Sequence[float]) -> np.ndarray:
  requests = np.asarray(requests, dtype=np.float64)
  if requests.shape != (config.num_agents,):
    raise ValueError("Expected one scalar harvest request per agent")
  if (
      not np.all(np.isfinite(requests))
      or np.any(requests < 0)
      or np.any(requests > config.carrying_capacity)
  ):
    raise ValueError("Harvest requests must be finite and within [0, K]")
  return requests


class SafetyFilter:
  """Section 3's analytic filter; no learned critic or training is needed."""

  def __init__(self, config: Config):
    self.config = config

  def __call__(self, state: State, requests: Sequence[float]) -> np.ndarray:
    requests = validate_requests(self.config, requests)
    if not in_safe_core(self.config, state):
      raise ValueError("Safety certification requires a state in Omega_core")
    return project_harvest(
        requests,
        metabolic_floors(self.config, state),
        harvest_cap(self.config, state),
    )


def transition(config: Config, state: State, harvest: Sequence[float]) -> State:
  """Applies one physical transition, without silently correcting actions."""
  validate_state(config, state)
  if state.failed:
    raise ValueError("Cannot transition after starvation; reset the episode")
  harvest = validate_requests(config, harvest)
  total = math.fsum(harvest)
  if total > state.resource:
    raise ValueError("Executed harvest exceeds available resource")
  escapement = state.resource - total
  growth = (
      config.growth_rate
      * escapement
      * (1 - escapement / config.carrying_capacity)
  )
  resource = min(config.carrying_capacity, escapement + growth)
  energies = tuple(
      min(config.max_energy, energy - config.metabolic_cost + float(amount))
      for energy, amount in zip(state.energies, harvest)
  )
  return State(resource, energies, state.round_index + 1)
