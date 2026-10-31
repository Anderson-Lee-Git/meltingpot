"""Tests of the physical dynamics, projection optimum, and invariant core."""

import itertools
import math

from examples.non_spatial_commons import model
import numpy as np
import pytest


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_agents": 0},
        {"num_agents": True},
        {"growth_rate": 0},
        {"growth_rate": 2.1},
        {"growth_rate": math.nan},
        {"carrying_capacity": math.inf},
        {"max_energy": 0},
        {"num_agents": 4, "metabolic_cost": 3.125},
    ],
)
def test_invalid_parameters(kwargs):
  with pytest.raises(ValueError):
    model.Config(**kwargs)


def test_critical_escapement_offsets_maintenance():
  config = model.Config()
  s = config.critical_escapement
  assert 0 < s < config.carrying_capacity / 2
  assert config.growth_rate * s * (
      1 - s / config.carrying_capacity
  ) == pytest.approx(config.maintenance_demand)


def test_harvest_then_growth_and_storage_saturation():
  config = model.Config(num_agents=2)
  state = model.State(40, (0, 9.5))
  successor = model.transition(config, state, (1, 3))
  assert successor.resource == pytest.approx(36 + 0.5 * 36 * (1 - 36 / 100))
  assert successor.energies == (0, 10)
  assert successor.round_index == 1
  with pytest.raises(ValueError, match="available resource"):
    model.transition(config, state, (30, 30))


def test_truncation_and_nonmonotone_growth_above_one():
  config = model.Config(num_agents=1, growth_rate=2)
  successor = model.transition(config, model.State(75, (5,)), (0,))
  assert successor.resource == config.carrying_capacity


def test_failure_is_recorded_not_clamped_and_zero_stock_cannot_regrow():
  config = model.Config(num_agents=1)
  state = model.State(0, (0.5,))
  assert not state.failed
  assert not model.in_safe_core(config, state)
  successor = model.transition(config, state, (0,))
  assert successor.resource == 0
  assert successor.energies == (-0.5,)
  assert successor.failed
  with pytest.raises(ValueError, match="after starvation"):
    model.transition(config, successor, (0,))


@pytest.mark.parametrize(
    "requests,floors,budget,expected",
    [
        ([1, 2], [0, 0], 5, [1, 2]),
        ([0, 0], [1, 2], 5, [1, 2]),
        ([8, 1], [0, 1], 5, [4, 1]),
        ([8, 6], [0, 0], 6, [4, 2]),
        ([100, 0], [1, 2], 3, [1, 2]),
        ([0, 0], [0, 0], 0, [0, 0]),
    ],
)
def test_projection_examples(requests, floors, budget, expected):
  result = model.project_harvest(requests, floors, budget)
  np.testing.assert_allclose(result, expected, atol=1e-12)
  assert math.fsum(result) <= budget


@pytest.mark.parametrize(
    "requests,floors,budget",
    [
        ([1], [2], 1),
        ([math.nan], [0], 1),
        ([1], [-1], 1),
        ([1, 2], [0], 1),
        ([[1]], [[0]], 1),
        ([1], [0], math.inf),
        ([-1], [0], 1),
        ([], [], 0),
    ],
)
def test_invalid_projection(requests, floors, budget):
  with pytest.raises(ValueError):
    model.project_harvest(requests, floors, budget)


def _reference_projection(requests, floors, budget):
  """Independent exhaustive active-set solver for small test problems."""
  raised = np.maximum(requests, floors)
  if sum(raised) <= budget:
    return raised
  for free_flags in itertools.product([False, True], repeat=len(requests)):
    free = np.array(free_flags)
    if not np.any(free):
      continue
    theta = (sum(requests[free]) + sum(floors[~free]) - budget) / sum(free)
    if theta < 0:
      continue
    if np.all(requests[free] - theta >= floors[free] - 1e-12) and np.all(
        requests[~free] - theta <= floors[~free] + 1e-12
    ):
      return np.where(free, requests - theta, floors)
  raise AssertionError("Reference solver found no solution")


def test_projection_matches_independent_active_set_optimum():
  rng = np.random.default_rng(24)
  for _ in range(200):
    requests = rng.uniform(0, 10, size=4)
    floors = rng.uniform(0, 2, size=4)
    budget = math.fsum(floors) + rng.uniform(0.001, 15)
    result = model.project_harvest(requests, floors, budget)
    expected = _reference_projection(requests, floors, budget)
    np.testing.assert_allclose(result, expected, atol=1e-11, rtol=1e-11)
    assert np.all(result >= floors)
    assert math.fsum(result) <= budget


def test_projection_uses_feasible_side_near_floor_budget():
  floors = np.array([0.1, 0.2, 0.3])
  budget = np.nextafter(math.fsum(floors), math.inf)
  result = model.project_harvest([10, 20, 30], floors, budget)
  assert np.all(result >= floors)
  assert math.fsum(result) <= budget


def test_boundary_floors_reenter_core():
  config = model.Config(max_energy=0.25)
  energies = (0.0, 0.1, 0.2, 0.25)
  floors = np.maximum(0, config.metabolic_cost - np.asarray(energies))
  state = model.State(config.protected_escapement + math.fsum(floors), energies)
  action = model.SafetyFilter(config)(state, (0,) * config.num_agents)
  after = model.transition(config, state, action)
  assert min(after.energies) >= 0
  assert model.in_safe_core(config, after)


def test_safe_core_invariance_across_parameters_and_arbitrary_requests():
  rng = np.random.default_rng(532)
  for trial in range(40):
    n = int(rng.integers(1, 7))
    capacity = float(rng.uniform(10, 200))
    growth = float(rng.uniform(0.05, 2))
    delta = capacity * growth / (4 * n) * float(rng.uniform(0.05, 0.95))
    config = model.Config(n, capacity, growth, delta, max_energy=delta * 2)
    energies = tuple(rng.uniform(0, config.max_energy, size=n))
    b = math.fsum(model.metabolic_floors(config, model.State(0, energies)))
    lower = config.protected_escapement + b
    # Half the trials start exactly on the coupled boundary, including r > 1.
    resource = lower if trial % 2 == 0 else float(rng.uniform(lower, capacity))
    state = model.State(resource, energies)
    safety_filter = model.SafetyFilter(config)
    for _ in range(100):
      action = safety_filter(state, rng.uniform(0, capacity, size=n))
      assert np.all(action >= model.metabolic_floors(config, state))
      assert math.fsum(action) <= model.harvest_cap(config, state)
      state = model.transition(config, state, action)
      assert min(state.energies) >= 0
      assert model.in_safe_core(config, state)


def test_nonviable_state_not_certified_despite_large_reserves():
  config = model.Config(
      num_agents=1,
      carrying_capacity=8,
      growth_rate=1,
      metabolic_cost=1,
      max_energy=3,
  )
  state = model.State(0, (3,))
  assert not state.failed
  with pytest.raises(ValueError, match="Omega_core"):
    model.SafetyFilter(config)(state, (0,))


def test_operational_core_has_positive_regenerative_roundoff_margin():
  config = model.Config()
  protected = config.protected_escapement
  assert config.critical_escapement < protected < config.carrying_capacity / 2
  assert (
      config.growth_rate
      * protected
      * (1 - protected / config.carrying_capacity)
      > config.maintenance_demand
  )
  state = model.State(
      np.nextafter(protected, 0), (config.max_energy,) * config.num_agents
  )
  assert not model.in_safe_core(config, state)  # No unsafe tolerance slack.


def test_unresolvable_growth_gap_rejected():
  with pytest.raises(ValueError, match="float64"):
    model.Config(growth_rate=1e-30, metabolic_cost=1e-32)
