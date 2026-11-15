"""A multi-agent dm_env interface for the spatial commons."""

from collections.abc import Mapping, Sequence
import dataclasses

import dm_env
from dm_env import specs
from examples.spatial_commons import model
import numpy as np

Observation = Mapping[str, np.ndarray]


@dataclasses.dataclass(frozen=True, eq=False)
class StepInfo:
  """Proposed versus executed actions, kept out of the observations."""

  before: model.State
  after: model.State
  proposed: tuple[int, ...]
  executed: tuple[int, ...]
  eaten: tuple[int, ...]
  shielded: bool
  rounds_to_core: int | None

  @property
  def overridden(self) -> tuple[bool, ...]:
    return tuple(a != b for a, b in zip(self.proposed, self.executed))

  @property
  def intervention_count(self) -> int:
    return sum(self.overridden)


class SpatialCommonsEnvironment(dm_env.Environment):
  """Fully observed grid game with an optional backup-set safety filter.

  Actions are one integer per agent (see model.ACTION_NAMES). Each agent
  observes all positions, the ripe-apple map and its own energy (Section 2.2);
  other agents' energies and the continuous ripeness stay hidden. Reward is
  apples eaten this round (0 or 1). Starvation terminates with discount 0; the
  time limit truncates with discount 1.
  """

  def __init__(
      self,
      config: model.Config = model.Config(),
      *,
      shield: bool = True,
      horizon: int = model.DEFAULT_HORIZON,
      episode_length: int = 1000,
      initial_state: model.State | None = None,
  ):
    if (
        isinstance(episode_length, bool)
        or not isinstance(episode_length, int)
        or not 1 <= episode_length <= np.iinfo(np.int32).max
    ):
      raise ValueError("episode_length must be a positive int32 integer")
    self.config = config
    self.shield = shield
    self.episode_length = episode_length
    self._filter = model.SafetyFilter(config, horizon) if shield else None
    self._initial_state = initial_state or model.default_state(config)
    self._validate_initial_state(self._initial_state)
    self._state = self._initial_state
    self._reset_next_step = True
    self._elapsed_steps = 0
    self._last_info = None

  def _validate_initial_state(self, state: model.State) -> None:
    model.validate_state(self.config, state)
    if state.failed or state.round_index != 0:
      raise ValueError(
          "Initial state must be non-failed and have round_index=0"
      )
    if self.shield and not model.in_backup_set(
        self.config, state, self._filter.horizon
    ):
      raise ValueError("Shielded initialization must be in Omega_B")

  @property
  def state(self) -> model.State:
    return self._state

  @property
  def last_info(self) -> StepInfo | None:
    return self._last_info

  def reset(self, *, state: model.State | None = None) -> dm_env.TimeStep:
    initial = self._initial_state if state is None else state
    self._validate_initial_state(initial)
    self._state = initial
    self._elapsed_steps = 0
    self._reset_next_step = False
    self._last_info = None
    return dm_env.restart(self._observations())

  def step(self, actions: Sequence[int]) -> dm_env.TimeStep:
    if self._reset_next_step:
      return self.reset()
    proposed = model.validate_actions(self.config, actions)
    before = self._state
    if self._filter is not None:
      result = self._filter(before, proposed)
      executed, rounds_to_core = result.actions, result.rounds_to_core
    else:
      executed, rounds_to_core = proposed, None
    outcome = model.step(self.config, before, executed)
    after = outcome.next_state
    self._last_info = StepInfo(
        before,
        after,
        proposed,
        executed,
        outcome.eaten,
        self.shield,
        rounds_to_core,
    )
    self._state = after
    self._elapsed_steps += 1
    rewards = tuple(np.float64(h) for h in outcome.eaten)
    observations = self._observations()
    if after.failed:
      self._reset_next_step = True
      return dm_env.termination(rewards, observations)
    if self._elapsed_steps >= self.episode_length:
      self._reset_next_step = True
      return dm_env.truncation(rewards, observations, discount=1.0)
    return dm_env.transition(rewards, observations, discount=1.0)

  def _observations(self) -> tuple[Observation, ...]:
    positions = np.array(self._state.positions, dtype=np.int32)
    apples = self._state.ripe(self.config).astype(np.uint8)
    return tuple(
        {
            "POSITIONS": positions.copy(),
            "APPLES": apples.copy(),
            "ENERGY": np.asarray(self._state.energies[i], dtype=np.int32),
            "PLAYER_INDEX": np.asarray(i, dtype=np.int32),
            "ROUND": np.asarray(self._elapsed_steps, dtype=np.int32),
        }
        for i in range(self.config.num_agents)
    )

  def action_spec(self):
    return tuple(
        specs.DiscreteArray(model.NUM_ACTIONS, dtype=np.int64, name="move")
        for _ in range(self.config.num_agents)
    )

  def observation_spec(self):
    n, m = self.config.grid_size, self.config.num_agents
    return tuple(
        {
            "POSITIONS": specs.BoundedArray((m, 2), np.int32, 0, n - 1),
            "APPLES": specs.BoundedArray((n, n), np.uint8, 0, 1),
            "ENERGY": specs.BoundedArray(
                (), np.int32, 0, self.config.max_energy
            ),
            "PLAYER_INDEX": specs.BoundedArray((), np.int32, 0, m - 1),
            "ROUND": specs.BoundedArray((), np.int32, 0, self.episode_length),
        }
        for _ in range(m)
    )

  def reward_spec(self):
    return tuple(
        specs.BoundedArray((), np.float64, 0, 1)
        for _ in range(self.config.num_agents)
    )

  def close(self):
    pass
