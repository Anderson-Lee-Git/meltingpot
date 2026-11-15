"""Tests of spatial PPO: features, unshielded pool, checkpoint round trips."""

from examples.spatial_commons import environment
from examples.spatial_commons import evaluate_trained
from examples.spatial_commons import model
from examples.spatial_commons import ppo
from examples.spatial_commons import train
import numpy as np
import pytest
import torch


def test_features_encode_own_and_other_positions():
  config = model.Config()
  env = environment.SpatialCommonsEnvironment(config, shield=False)
  obs = env.reset().observation
  x = ppo.features(obs[1], config)
  n2 = config.grid_size**2
  assert x.shape == (ppo.feature_dim(config),)
  assert x[:n2].sum() == 20  # four ripe crosses
  own = x[n2 : 2 * n2].reshape(config.grid_size, -1)
  others = x[2 * n2 : 3 * n2].reshape(config.grid_size, -1)
  assert own[2, 7] == 1 and own.sum() == 1
  assert others.sum() == 3 and others[2, 7] == 0
  assert x[3 * n2] == pytest.approx(0.5)
  np.testing.assert_array_equal(x[-4:], [0, 1, 0, 0])


def test_gae_bootstraps_truncation_but_not_death():
  rewards = np.array([[1, 1], [100, 100]], dtype=np.float32)
  values = np.zeros_like(rewards)
  next_values = np.array([[10, 10], [0, 0]], dtype=np.float32)
  terminated = np.array([[True, False], [True, True]])
  ended = np.ones_like(terminated)
  advantage, _ = ppo.generalized_advantage(
      rewards, values, next_values, terminated, ended, 0.9, 0.95
  )
  np.testing.assert_allclose(advantage[0], [1, 10])


@pytest.mark.parametrize("experiment", ppo.EXPERIMENTS)
def test_training_pool_is_unshielded(experiment):
  config = model.Config()
  pool = ppo.TrainingPool(config, experiment, 2, 3, 0)
  assert all(not env.shield for env in pool.envs)
  rows = pool.num_rows
  assert rows == (8 if experiment in ppo.MULTI_AGENT else 2)
  if experiment in ppo.CENTRALIZED:
    dim, actions = ppo.joint_feature_dim(config), np.full((rows, 4), model.UP)
  else:
    dim, actions = ppo.feature_dim(config), np.full(rows, model.UP)
  assert pool.current_features().shape == (rows, dim)
  out = pool.step(actions)
  assert all(a.shape[0] == rows for a in out[:4])
  assert all(not env.last_info.shielded for env in pool.envs)


@pytest.mark.parametrize("experiment", ["individual_greedy", "cooperative"])
def test_smoke_training_and_frozen_round_trip(tmp_path, experiment):
  training = train.TrainingConfig(
      total_steps=16,
      num_envs=2,
      rollout_steps=4,
      epochs=1,
      minibatch_size=8,
      episode_length=4,
  )
  checkpoint = train.train_one(
      model.Config(), training, experiment, 3, tmp_path
  )
  policy = ppo.FrozenPolicy(checkpoint)
  assert policy.config == model.Config()
  assert not any(p.requires_grad for p in policy.network.parameters())
  assert policy.metadata["steps"] == 16
  env = environment.SpatialCommonsEnvironment(policy.config)
  obs = env.reset().observation
  actions = [p(o) for p, o in zip(policy.population(), obs)]
  assert all(0 <= a < model.NUM_ACTIONS for a in actions)
  again = ppo.FrozenPolicy(checkpoint)
  assert actions == [p(o) for p, o in zip(again.population(), obs)]
  a = ppo.FrozenPolicy(checkpoint, stochastic=True, seed=1)
  b = ppo.FrozenPolicy(checkpoint, stochastic=True, seed=1)
  assert [a(obs[0]) for _ in range(20)] == [b(obs[0]) for _ in range(20)]


def test_evaluation_starts_are_in_backup_set():
  config = model.Config()
  starts = evaluate_trained.evaluation_starts(config, 3, 532)
  assert len(starts) == 7
  assert all(model.in_backup_set(config, s) for _, s in starts)


def test_selfish_rewards_are_per_agent_and_cooperative_shared():
  config = model.Config()
  # From the default start, all-stay eats each home centre in round one.
  selfish = ppo.TrainingPool(config, "selfish_shared", 1, 5, 0)
  _, rewards, _, _, _ = selfish.step(np.full(4, model.STAY))
  np.testing.assert_array_equal(rewards, [1, 1, 1, 1])
  # Agent 0 steps off its centre onto a ripe neighbour; the others stay.
  pool = ppo.TrainingPool(config, "selfish_shared", 1, 5, 0)
  _, rewards, _, _, _ = pool.step([model.UP, 0, 0, 0])
  np.testing.assert_array_equal(rewards, [1, 1, 1, 1])
  coop = ppo.TrainingPool(config, "cooperative", 1, 5, 0)
  _, rewards, _, _, _ = coop.step(np.full(4, model.STAY))
  np.testing.assert_array_equal(rewards, [1, 1, 1, 1])
  assert selfish.num_networks == 1 and coop.num_networks == 1
  assert (
      ppo.TrainingPool(config, "selfish_independent", 1, 5, 0).num_networks == 4
  )


def test_selfish_reward_differs_from_team_reward():
  config = model.Config()
  state = model.default_state(config)
  ripeness = np.array(state.ripeness)
  ripeness[2, 2] = 0  # agent 0's centre is empty; the others' are ripe
  start = model.State(state.positions, state.energies, ripeness)
  for name, expected in [
      ("selfish_shared", [0, 1, 1, 1]),
      ("cooperative", [0.75] * 4),
  ]:
    pool = ppo.TrainingPool(config, name, 1, 5, 0)
    pool.observations[0] = pool.envs[0].reset(state=start).observation
    _, rewards, _, _, _ = pool.step(np.full(4, model.STAY))
    np.testing.assert_allclose(rewards, expected)


def test_independent_checkpoint_has_one_network_per_agent(tmp_path):
  training = train.TrainingConfig(
      total_steps=16,
      num_envs=2,
      rollout_steps=4,
      epochs=1,
      minibatch_size=8,
      episode_length=4,
  )
  checkpoint = train.train_one(
      model.Config(), training, "selfish_independent", 0, tmp_path
  )
  policy = ppo.FrozenPolicy(checkpoint)
  assert len(policy.networks) == 4
  first = policy.networks[0].actor[-1].weight
  assert not torch.equal(first, policy.networks[1].actor[-1].weight)
  env = environment.SpatialCommonsEnvironment(policy.config)
  obs = env.reset().observation
  assert all(0 <= p(o) < 5 for p, o in zip(policy.population(), obs))


def test_centralized_checkpoint_acts_jointly(tmp_path):
  training = train.TrainingConfig(
      total_steps=16,
      num_envs=2,
      rollout_steps=4,
      epochs=1,
      minibatch_size=8,
      episode_length=4,
      starts="random",
  )
  checkpoint = train.train_one(
      model.Config(), training, "centralized", 0, tmp_path
  )
  policy = ppo.FrozenPolicy(checkpoint, stochastic=True, seed=3)
  state = model.default_state(policy.config)
  joint = policy.act_joint(state)
  assert len(joint) == 4 and all(0 <= a < 5 for a in joint)
  policy.reseed(3)
  assert policy.act_joint(state) == joint
  with pytest.raises(TypeError):
    policy.population()
  # Factorised head: the joint log-probability is the sum over agents.
  x = torch.as_tensor(ppo.joint_features(state, policy.config))[None]
  dist = policy.network.distribution(x)
  action = torch.tensor([joint])
  logits = policy.network.actor(x).view(4, 5).log_softmax(-1)
  expected = sum(logits[i, a] for i, a in enumerate(joint))
  torch.testing.assert_close(dist.log_prob(action)[0], expected)


def test_shielded_training_pool_and_random_starts():
  config = model.Config()
  pool = ppo.TrainingPool(
      config, "selfish_shared", 2, 5, 0, shield=True, starts="random"
  )
  assert all(env.shield for env in pool.envs)
  for _ in range(5):
    pool.step(np.full(pool.num_rows, model.UP))
    assert not any(env.state.failed for env in pool.envs)
