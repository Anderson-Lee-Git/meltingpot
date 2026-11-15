"""Tests of movement, dynamics, the k-neighbour core and the filter."""

from fractions import Fraction

from examples.spatial_commons import environment
from examples.spatial_commons import model
from examples.spatial_commons import policies
from examples.spatial_commons import starts
import numpy as np
import pytest

S, U, D, L, R = model.STAY, model.UP, model.DOWN, model.LEFT, model.RIGHT


# --------------------------------------------------------------------------
# Movement.


@pytest.mark.parametrize(
    "positions, actions, expected",
    [
        # Off the grid executes as stay.
        (((0, 0),), (U,), ((0, 0),)),
        (((0, 0),), (L,), ((0, 0),)),
        # Same target: lowest index wins regardless of direction.
        (((1, 0), (1, 2)), (R, L), ((1, 1), (1, 2))),
        (((1, 2), (1, 0)), (L, R), ((1, 1), (1, 0))),
        # Blocked by an agent that stays.
        (((1, 1), (1, 2)), (R, S), ((1, 1), (1, 2))),
        # Vacating allowed, in either index order.
        (((1, 1), (1, 2)), (R, R), ((1, 2), (1, 3))),
        (((1, 2), (1, 1)), (R, R), ((1, 3), (1, 2))),
        # Swap is a 2-cycle: blocked.
        (((1, 1), (1, 2)), (R, L), ((1, 1), (1, 2))),
        # 2x2 rotation is a 4-cycle: blocked.
        (
            ((1, 1), (1, 2), (2, 2), (2, 1)),
            (R, D, L, U),
            ((1, 1), (1, 2), (2, 2), (2, 1)),
        ),
        # Chain blocked at its head propagates backwards.
        (
            ((1, 1), (1, 2), (1, 3)),
            (R, R, S),
            ((1, 1), (1, 2), (1, 3)),
        ),
        # Chain into a tile vacated by a same-target winner.
        (
            ((1, 1), (1, 2), (0, 3)),
            (R, R, D),
            ((1, 2), (1, 3), (0, 3)),
        ),
        # Agent moving into the tail of a swap cycle is blocked too.
        (
            ((1, 1), (1, 2), (1, 0)),
            (R, L, R),
            ((1, 1), (1, 2), (1, 0)),
        ),
    ],
)
def test_resolve_moves(positions, actions, expected):
  assert model.resolve_moves(5, positions, actions) == expected


def test_same_target_loser_blocks_follower():
  # Agent 0 wins (1,3); agent 1 loses it, so agent 2 cannot enter (1,2).
  positions = ((0, 3), (1, 2), (1, 1))
  assert model.resolve_moves(5, positions, (D, R, R)) == (
      (1, 3),
      (1, 2),
      (1, 1),
  )


def test_random_moves_never_collide():
  rng = np.random.default_rng(0)
  for _ in range(2000):
    cells = rng.choice(16, size=4, replace=False)
    positions = tuple((int(c) // 4, int(c) % 4) for c in cells)
    actions = tuple(int(a) for a in rng.integers(5, size=4))
    out = model.resolve_moves(4, positions, actions)
    assert len(set(out)) == 4
    for p, a, q in zip(positions, actions, out):
      assert q == p or q == (
          p[0] + model.DISPLACEMENTS[a][0],
          p[1] + model.DISPLACEMENTS[a][1],
      )


# --------------------------------------------------------------------------
# Dynamics.


def small_config(**kwargs):
  defaults = dict(
      grid_size=5,
      homes=((2, 2),),
      metabolic_cost=1,
      apple_energy=5,
      max_energy=10,
      initial_energy=5,
      growth_rate=Fraction(1, 4),
  )
  defaults.update(kwargs)
  return model.Config(**defaults)


def test_growth_rate_parsing():
  assert model.Config(growth_rate=0.015).growth_rate == Fraction(3, 200)
  assert model.Config(growth_rate="3/200").growth_rate == Fraction(3, 200)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metabolic_cost": 0},
        {"apple_energy": 1.5},
        {"growth_rate": 0},
        {"growth_rate": 2},
        {"initial_energy": 50},
        {"homes": ((2, 2), (2, 2))},
        {"homes": ((10, 0),)},
    ],
)
def test_invalid_config(kwargs):
  with pytest.raises(ValueError):
    model.Config(**kwargs)


def test_harvest_and_same_round_regrowth():
  config = small_config()
  state = model.default_state(config)
  out = model.step(config, state, (S,))
  assert out.eaten == (1,)
  assert out.next_state.energies == (5 - 1 + 5,)
  # The eaten centre regrows in the same round from its 4 ripe neighbours.
  full = config.ripe_units
  assert out.next_state.ripeness[2, 2] == min(full, 4 * config.growth_units)
  # Diagonal (1,1) sees ripe (1,2),(2,1); the eaten centre no longer counts.
  assert out.next_state.ripeness[1, 1] == 2 * config.growth_units
  # Ripe neighbours that are not eaten stay ripe.
  assert out.next_state.ripeness[1, 2] == full


def test_energy_clamps():
  config = small_config(max_energy=6, initial_energy=6)
  state = model.default_state(config)
  assert model.transition(config, state, (S,)).energies == (6,)
  empty = model.State(((0, 0),), (1,), np.zeros((5, 5)))
  after = model.transition(config, empty, (S,))
  assert after.energies == (0,) and after.failed
  with pytest.raises(ValueError):
    model.transition(config, after, (S,))


def test_isolated_tiles_never_regrow():
  config = small_config()
  z = np.zeros((5, 5))
  state = model.State(((0, 0),), (5,), z)
  for _ in range(5):
    state = model.transition(config, state, (S,))
  assert state.ripeness.sum() == 0 and state.failed


# --------------------------------------------------------------------------
# k-neighbour staying core.


def test_default_config_is_certified():
  config = model.Config()
  assert config.core_cycle == 17
  assert config.core_certified
  assert model.in_core(config, model.default_state(config))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"growth_rate": Fraction(1, 100)},  # T = 25 > eta = 20
        {"max_energy": 16, "initial_energy": 16},  # E_max <= (T-1) delta
        {"homes": ((2, 2), (3, 3), (7, 2), (7, 7))},  # 8-neighbours
        {"homes": ((2, 2), (2, 3), (7, 2), (7, 7))},  # 8-neighbours
        {"core_k": 2},  # T_2 = 34 > eta = 20
    ],
)
def test_uncertified_configs(kwargs):
  config = model.Config(**kwargs)
  assert not config.core_certified
  with pytest.raises(ValueError):
    model.SafetyFilter(config)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"homes": ((0, 0), (2, 7), (7, 2), (9, 9))},  # border homes
        {"homes": ((2, 2), (2, 4), (4, 2), (4, 4))},  # inf-distance 2
        {"core_k": 8, "growth_rate": Fraction(1, 160)},  # T_8 = 20
        {"core_k": 1, "growth_rate": Fraction(1, 20)},  # T_1 = 20
    ],
)
def test_certified_configs(kwargs):
  config = model.Config(**kwargs)
  assert config.core_certified, config.core_condition_violations()
  model.SafetyFilter(config)


def test_core_counts_ripe_8_neighbours():
  config = model.Config(homes=((0, 0), (2, 7), (7, 2), (7, 7)))
  ripe = config.ripe_units
  state = model.default_state(config)
  z = np.array(state.ripeness)
  z[0, 0] = ripe  # home tile itself does not count
  z[0, 1] = z[1, 0] = 0
  z[1, 1] = ripe  # only 1 of the 3 clipped neighbours of the corner is ripe
  energies = (config.max_energy,) * 4
  state = model.State(config.homes, energies, z)
  assert model.home_ripe_counts(config, state)[0] == 1
  assert not model.in_core(config, state)
  z[0, 1] = z[1, 0] = ripe
  state = model.State(config.homes, energies, z)
  assert model.home_ripe_counts(config, state) == (3, 4, 4, 4)
  assert not model.in_core(config, state)
  k3 = model.Config(homes=config.homes, core_k=3, apple_energy=23)  # T_3=23
  assert k3.core_certified, k3.core_condition_violations()
  assert model.in_core(k3, state)
  # Diagonal neighbours count, so a diagonal-only ring certifies k = 4.
  z2 = np.zeros_like(z)
  for r, c in model.DEFAULT_HOMES:
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
      z2[r + dr, c + dc] = ripe
  diag = model.State(model.DEFAULT_HOMES, energies, z2)
  assert model.in_core(model.Config(), diag)


def _all_stay_never_leaves_core(config, steps):
  state = model.default_state(config)
  stay = (S,) * config.num_agents
  for _ in range(steps):
    assert model.in_core(config, state)
    state = model.transition(config, state, stay)
  assert model.in_core(config, state)


def test_core_invariant_under_all_stay_default():
  _all_stay_never_leaves_core(model.Config(), 1000)


@pytest.mark.parametrize(
    "growth_rate, eta, e_max",
    [
        (Fraction(1, 76), 19, 19),  # boundary T*delta = eta, E_max = T
        (Fraction(1, 63), 16, 16),
        (Fraction(1, 40), 10, 10),
        (Fraction(1, 4), 1, 1),  # T = 1
    ],
)
def test_core_invariant_at_parameter_boundaries(growth_rate, eta, e_max):
  config = model.Config(
      growth_rate=growth_rate,
      apple_energy=eta,
      max_energy=e_max,
      initial_energy=1,
  )
  assert config.core_certified, config.core_condition_violations()
  _all_stay_never_leaves_core(config, 600)


def test_core_minimal_energy_is_tight():
  # One energy unit below the core bound starves under all-stay.
  config = model.Config()
  state = model.default_state(config)
  state = model.transition(config, state, (S,) * 4)  # centres just eaten
  for _ in range(3):
    state = model.transition(config, state, (S,) * 4)
  r, c = config.homes[0]
  tau = model.burn_rounds(config, state.ripeness[r, c])
  energies = list(state.energies)
  energies[0] = tau  # E = delta * tau: outside the core
  edge = model.State(state.positions, energies, state.ripeness)
  assert not model.in_core(config, edge)
  energies[0] = tau + 1
  assert model.in_core(
      config, model.State(state.positions, energies, state.ripeness)
  )


# --------------------------------------------------------------------------
# Backup set and filter.


def test_core_is_in_backup_set():
  config = model.Config()
  assert model.rounds_to_core(config, model.default_state(config)) == 0


def test_backup_policy_walks_home():
  config = model.Config()
  state = model.default_state(config)
  moved = model.State(
      ((0, 0), (2, 7), (7, 2), (9, 9)), state.energies, state.ripeness
  )
  assert model.backup_actions(config, moved) == (D, S, S, U)


def test_filter_passes_safe_proposals_unchanged():
  config = model.Config()
  f = model.SafetyFilter(config)
  state = model.default_state(config)
  result = f(state, (S, S, S, S))
  assert result.actions == (S, S, S, S) and result.intervention_count == 0


def test_filter_search_order():
  f = model.SafetyFilter(model.Config())
  cands = f.candidates((U, U, U, U), (S, S, S, S))
  assert len(cands) == 5**4 and len(set(cands)) == 5**4
  assert cands[0] == (U, U, U, U)
  # Single overrides with the backup action come first, agent 0 first.
  assert cands[1:5] == [(S, U, U, U), (U, S, U, U), (U, U, S, U), (U, U, U, S)]


def test_filter_rejects_states_outside_backup_set():
  config = model.Config()
  state = model.default_state(config)
  starving = model.State(
      ((0, 0), (2, 7), (7, 2), (7, 7)), (1, 20, 20, 20), state.ripeness
  )
  assert not model.in_backup_set(config, starving)
  with pytest.raises(ValueError):
    model.SafetyFilter(config)(starving, (S,) * 4)


@pytest.mark.parametrize("names", [("greedy",), ("random",), ("mixed",)])
@pytest.mark.parametrize("seed", [0, 1])
def test_filter_keeps_state_in_backup_set(names, seed):
  config = model.Config()
  if names == ("mixed",):
    names = ("greedy", "random", "stay", "hungry")
  population = policies.make_population(names, config, seed=seed)
  env = environment.SpatialCommonsEnvironment(config, episode_length=150)
  timestep = env.reset()
  while not timestep.last():
    proposals = tuple(p(o) for p, o in zip(population, timestep.observation))
    timestep = env.step(proposals)
    assert not env.state.failed
    assert env.last_info.rounds_to_core is not None
    assert model.in_backup_set(config, env.state)
  assert timestep.discount == 1.0  # truncated, not terminated


def test_unshielded_greedy_starves():
  config = model.Config()
  population = policies.make_population(("greedy",), config)
  env = environment.SpatialCommonsEnvironment(
      config, shield=False, episode_length=1000
  )
  timestep = env.reset()
  while not timestep.last():
    timestep = env.step(
        tuple(p(o) for p, o in zip(population, timestep.observation))
    )
  assert env.state.failed and timestep.discount == 0.0


@pytest.mark.parametrize("seed", range(4))
def test_fast_rollout_matches_reference(seed):
  rng = np.random.default_rng(seed)
  config = model.Config(
      homes=((0, 0), (2, 2), (5, 9), (8, 4)), core_k=int(rng.integers(1, 5))
  )
  config = model.Config(
      homes=config.homes,
      core_k=config.core_k,
      growth_rate=Fraction(1, 5 * config.core_k),  # T_k = 5
      apple_energy=5,
      max_energy=10,
      initial_energy=10,
  )
  assert config.core_certified
  for _ in range(50):
    state = starts.propose(config, rng, float(rng.uniform(0, 0.8)))
    for horizon in (0, 7, 40):
      assert model.rounds_to_core(
          config, state, horizon
      ) == model.rounds_to_core_reference(config, state, horizon)
