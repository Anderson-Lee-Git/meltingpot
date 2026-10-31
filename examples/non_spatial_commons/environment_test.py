"""Tests of the multi-agent API, terminal semantics, and baseline behavior."""

import csv
import json

from examples.non_spatial_commons import environment
from examples.non_spatial_commons import evaluate
from examples.non_spatial_commons import model
from examples.non_spatial_commons import policies
import numpy as np
import pytest


def _validate_observations(env, timestep):
  assert len(timestep.observation) == env.config.num_agents
  for observation, spec in zip(timestep.observation, env.observation_spec()):
    for key in spec:
      spec[key].validate(observation[key])


def test_specs_rewards_and_normalized_policy_features():
  config = model.Config()
  env = environment.CommonsEnvironment(config, episode_length=2)
  first = env.reset()
  assert first.first()
  assert first.reward is None
  _validate_observations(env, first)
  features = environment.policy_features(first.observation[2], config)
  assert features.shape == (2 * config.num_agents + 1,)
  np.testing.assert_array_equal(features[-config.num_agents :], [0, 0, 1, 0])
  for spec in env.action_spec():
    spec.validate(np.asarray(config.msy / config.num_agents, np.float64))
  timestep = env.step([config.msy / config.num_agents] * config.num_agents)
  _validate_observations(env, timestep)
  for reward, spec in zip(timestep.reward, env.reward_spec()):
    spec.validate(np.asarray(reward))
  assert timestep.mid()
  np.testing.assert_allclose(timestep.reward, env.last_info.executed)
  last = env.step([0] * config.num_agents)
  assert last.last()
  assert last.discount == 1.0  # Time limit is truncation, not starvation.
  assert env.step([0] * config.num_agents).first()


def test_starvation_terminates_with_zero_discount():
  config = model.Config(num_agents=1)
  env = environment.CommonsEnvironment(
      config, shield=False, initial_state=model.State(0, (0.5,))
  )
  env.reset()
  last = env.step([0])
  assert last.last()
  assert last.discount == 0.0
  assert float(last.observation[0]["ENERGY"]) == -0.5
  _validate_observations(env, last)


def test_observations_do_not_alias_state_or_other_players():
  env = environment.CommonsEnvironment()
  timestep = env.reset()
  timestep.observation[0]["ALL_ENERGIES"][:] = -1
  timestep.observation[0]["RESOURCE"][...] = 0
  assert env.state.resource > 0
  assert min(env.state.energies) > 0
  assert np.all(timestep.observation[1]["ALL_ENERGIES"] > 0)


def test_shield_can_raise_requests_to_metabolic_floors():
  config = model.Config(num_agents=2)
  env = environment.CommonsEnvironment(
      config, initial_state=model.State(config.msy_stock, (0, 0))
  )
  env.reset()
  env.step([0, 0])
  assert env.last_info.executed == (1, 1)
  assert env.last_info.corrections == (1, 1)
  assert env.last_info.intervention_count == 2
  assert env.state.energies == (0, 0)


def test_reset_outside_core_rejected_only_when_shielded():
  state = model.State(0, (3,))
  config = model.Config(num_agents=1)
  with pytest.raises(ValueError, match="Omega_core"):
    environment.CommonsEnvironment(config, initial_state=state)
  environment.CommonsEnvironment(config, shield=False, initial_state=state)


@pytest.mark.parametrize(
    "actions",
    [
        [1],
        [1, 2, 3, np.nan],
        [1, 2, 3, -1],
        [1, 2, 3, 101],
        [[1], [2], [3], [4]],
    ],
)
def test_invalid_actions_do_not_change_state(actions):
  env = environment.CommonsEnvironment()
  env.reset()
  before = env.state
  with pytest.raises(ValueError):
    env.step(actions)
  assert env.state == before
  assert env.last_info is None


@pytest.mark.parametrize("name", ["maintenance", "msy_share"])
def test_sustainable_baselines_need_no_intervention_at_msy_stock(name):
  config = model.Config()
  result = evaluate.run_episode(
      config, policies.make_population((name,), config), steps=100
  )
  assert result.summary["survived"]
  assert result.summary["corrected_rounds"] == 0
  if name == "msy_share":
    assert result.summary["final_resource"] == pytest.approx(config.msy_stock)
    assert result.summary["harvest_per_horizon_step"] == pytest.approx(
        config.msy
    )


@pytest.mark.parametrize("name", policies.POLICY_NAMES)
def test_all_fixed_policies_survive_when_shielded(name):
  config = model.Config()
  result = evaluate.run_episode(
      config, policies.make_population((name,), config, seed=4), steps=100
  )
  assert result.summary["survived"]
  assert result.summary["minimum_energy"] >= 0
  assert result.summary["minimum_core_margin"] >= -model.CORE_TOLERANCE


@pytest.mark.parametrize("name", ["greedy", "abstain"])
def test_unshielded_pathological_policies_fail(name):
  config = model.Config()
  result = evaluate.run_episode(
      config, policies.make_population((name,), config), shield=False, steps=100
  )
  assert not result.summary["survived"]
  assert result.summary["failure_round"] < 100
  assert result.summary["minimum_energy"] < 0
  assert result.summary["total_harvest"] / 100 == pytest.approx(
      result.summary["harvest_per_horizon_step"]
  )


def test_near_boundary_greedy_population_reaches_msy():
  config = model.Config()
  initial = model.State(
      config.critical_escapement + config.maintenance_demand + 0.5,
      (0,) * config.num_agents,
  )
  result = evaluate.run_episode(
      config,
      policies.make_population(("greedy",), config),
      steps=200,
      initial_state=initial,
  )
  assert result.summary["survived"]
  assert result.summary["final_resource"] == pytest.approx(config.msy_stock)
  assert sum(result.transitions[-1].executed) == pytest.approx(config.msy)


def test_seeded_random_policies_reproduce_evaluation():
  config = model.Config()
  runs = [
      evaluate.run_episode(
          config,
          policies.make_population(("random",), config, seed=17),
          steps=10,
      )
      for _ in range(2)
  ]
  assert runs[0] == runs[1]
  assert len(set(runs[0].transitions[0].requested)) == config.num_agents


def test_mixed_population_and_policy_errors():
  config = model.Config()
  names = ("greedy", "maintenance", "msy_share", "reserve_target")
  result = evaluate.run_episode(
      config, policies.make_population(names, config), steps=20
  )
  assert result.summary["survived"]
  with pytest.raises(ValueError):
    policies.make_population(("maintenance", "greedy"), config)
  with pytest.raises(ValueError):
    policies.make_policy("not_a_policy", config)


def test_cli_emits_auditable_paired_traces(tmp_path):
  evaluate.main([
      "--steps",
      "20",
      "--episodes",
      "1",
      "--policies",
      "greedy",
      "--output",
      str(tmp_path),
  ])
  payload = json.loads((tmp_path / "summary.json").read_text())
  assert len(payload["episodes"]) == 2
  assert [s["survived"] for s in payload["episodes"]] == [False, True]
  with (tmp_path / "trajectories.csv").open() as stream:
    rows = list(csv.DictReader(stream))
  assert len(rows) == sum(s["steps_executed"] for s in payload["episodes"])
  assert {r["shield"] for r in rows} == {"False", "True"}
  assert "energy_3" in rows[0]
