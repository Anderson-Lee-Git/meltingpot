"""A lightweight multi-agent dm_env interface for the non-spatial commons."""

from collections.abc import Mapping, Sequence
import dataclasses
import math

import dm_env
from dm_env import specs
from examples.non_spatial_commons import model
import numpy as np

Observation = Mapping[str, np.ndarray]


@dataclasses.dataclass(frozen=True)
class StepInfo:
  """Requested versus executed actions, kept out of the current observation."""

  before: model.State
  after: model.State
  requested: tuple[float, ...]
  executed: tuple[float, ...]
  cap: float
  shielded: bool

  @property
  def corrections(self) -> tuple[float, ...]:
    """Signed interventions; positive values mean forced additional harvest."""
    return tuple(u - a for u, a in zip(self.executed, self.requested))

  @property
  def intervention_count(self) -> int:
    return sum(abs(c) > model.CORE_TOLERANCE for c in self.corrections)

  @property
  def correction_l2(self) -> float:
    return math.sqrt(math.fsum(c * c for c in self.corrections))

  @property
  def escapement(self) -> float:
    return self.before.resource - math.fsum(self.executed)


class CommonsEnvironment(dm_env.Environment):
  """Fully observed continuous-action game with optional runtime shielding.

  Actions and rewards are tuples of scalar harvests, one per player. Reward is
  executed consumption, with no sanction or death penalty. Terminal failure has
  discount 0; the evaluation/training time limit truncates with discount 1.

  With shield=False, proposals are still projected onto the physical resource
  budget (zero floors, no ecological reserve). This is an unshielded comparison,
  not an environment allowed to harvest nonexistent resource.
  """

  def __init__(
      self,
      config: model.Config = model.Config(),
      *,
      shield: bool = True,
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
    self._initial_state = initial_state or model.default_state(config)
    self._validate_initial_state(self._initial_state)
    self._filter = model.SafetyFilter(config)
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
    if self.shield and not model.in_safe_core(self.config, state):
      raise ValueError("Shielded initialization must be in Omega_core")

  @property
  def state(self) -> model.State:
    return self._state

  @property
  def last_info(self) -> StepInfo | None:
    return self._last_info

  def reset(self, *, state: model.State | None = None) -> dm_env.TimeStep:
    initial = self._initial_state if state is None else state
    self._validate_initial_state(initial)
    # Canonicalize to immutable Python scalars, independent of caller arrays.
    self._state = model.State(
        float(initial.resource), tuple(float(e) for e in initial.energies)
    )
    self._elapsed_steps = 0
    self._reset_next_step = False
    self._last_info = None
    return dm_env.restart(self._observations())

  def step(self, actions: Sequence[float]) -> dm_env.TimeStep:
    # Follow dm_env's standard automatic reset after construction/termination.
    if self._reset_next_step:
      return self.reset()
    requests = model.validate_requests(self.config, actions)
    before = self._state
    if self.shield:
      cap = model.harvest_cap(self.config, before)
      executed = self._filter(before, requests)
    else:
      cap = before.resource
      executed = model.project_harvest(
          requests, np.zeros(self.config.num_agents), cap
      )
    after = model.transition(self.config, before, executed)
    if self.shield and not model.in_safe_core(self.config, after):
      raise RuntimeError(
          "Numerical safety check failed: successor left Omega_core"
      )
    self._last_info = StepInfo(
        before,
        after,
        tuple(float(a) for a in requests),
        tuple(float(u) for u in executed),
        cap,
        self.shield,
    )
    self._state = after
    self._elapsed_steps += 1
    rewards = tuple(np.float64(u) for u in executed)
    observations = self._observations()
    if after.failed:
      self._reset_next_step = True
      return dm_env.termination(rewards, observations)
    if self._elapsed_steps >= self.episode_length:
      self._reset_next_step = True
      return dm_env.truncation(rewards, observations, discount=1.0)
    return dm_env.transition(rewards, observations, discount=1.0)

  def _observations(self) -> tuple[Observation, ...]:
    return tuple(
        {
            "RESOURCE": np.asarray(self._state.resource, dtype=np.float64),
            "ENERGY": np.asarray(self._state.energies[i], dtype=np.float64),
            "ALL_ENERGIES": np.array(self._state.energies, dtype=np.float64),
            "PLAYER_INDEX": np.asarray(i, dtype=np.int32),
            "ROUND": np.asarray(self._elapsed_steps, dtype=np.int32),
        }
        for i in range(self.config.num_agents)
    )

  def action_spec(self):
    return tuple(
        specs.BoundedArray(
            (),
            np.float64,
            0,
            self.config.carrying_capacity,
            name="harvest_request",
        )
        for _ in range(self.config.num_agents)
    )

  def observation_spec(self):
    return tuple(
        {
            "RESOURCE": specs.BoundedArray(
                (), np.float64, 0, self.config.carrying_capacity
            ),
            "ENERGY": specs.BoundedArray(
                (),
                np.float64,
                -self.config.metabolic_cost,
                self.config.max_energy,
            ),
            "ALL_ENERGIES": specs.BoundedArray(
                (self.config.num_agents,),
                np.float64,
                -self.config.metabolic_cost,
                self.config.max_energy,
            ),
            "PLAYER_INDEX": specs.BoundedArray(
                (), np.int32, 0, self.config.num_agents - 1
            ),
            "ROUND": specs.BoundedArray((), np.int32, 0, self.episode_length),
        }
        for _ in range(self.config.num_agents)
    )

  def reward_spec(self):
    return tuple(
        specs.BoundedArray((), np.float64, 0, self.config.carrying_capacity)
        for _ in range(self.config.num_agents)
    )

  def close(self):
    pass


def policy_features(
    observation: Observation, config: model.Config
) -> np.ndarray:
  """Normalized full state + one-hot player ID for future learned policies.

  Shape is (2 * num_agents + 1,). The vector contains R/K, every E_i/E_max,
  and the acting player's one-hot ID. The action remains a raw scalar in [0,K].
  """
  player_id = np.zeros(config.num_agents, dtype=np.float64)
  player_id[int(observation["PLAYER_INDEX"])] = 1
  return np.concatenate((
      np.asarray([float(observation["RESOURCE"]) / config.carrying_capacity]),
      np.asarray(observation["ALL_ENERGIES"]) / config.max_energy,
      player_id,
  ))
