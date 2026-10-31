"""Tests of requested-action PPO, reset masking, and checkpoint round trips."""

from examples.non_spatial_commons import environment
from examples.non_spatial_commons import model
from examples.non_spatial_commons import ppo
from examples.non_spatial_commons import train
import numpy as np
import pytest
import torch


def test_gae_bootstraps_truncation_but_not_death_or_reset():
  rewards = np.array([[1, 1], [100, 100]], dtype=np.float32)
  values = np.zeros_like(rewards)
  next_values = np.array([[10, 10], [0, 0]], dtype=np.float32)
  terminated = np.array([[True, False], [True, True]])
  ended = np.ones_like(terminated)
  advantage, _ = ppo.generalized_advantage(
      rewards, values, next_values, terminated, ended, 0.9, 0.95
  )
  np.testing.assert_allclose(advantage[0], [1, 10])
  np.testing.assert_allclose(advantage[1], [100, 100])


def test_policy_sample_has_correct_bounds_and_probability_ratio():
  torch.manual_seed(0)
  network = ppo.ActorCritic(9, 4, 100, 1)
  features = torch.zeros(16, 9)
  actions, latent, old_logprob, _ = network.sample(features)
  logprob, _, _ = network.evaluate(features, latent)
  assert actions.shape == (16, 4)
  assert torch.all((actions >= 0) & (actions <= 100))
  torch.testing.assert_close((logprob - old_logprob).exp(), torch.ones(16))
  (-logprob.mean()).backward()
  assert network.actor[-1].weight.grad is not None


@pytest.mark.parametrize("experiment", ppo.EXPERIMENTS)
def test_training_pool_is_always_unshielded(experiment):
  config = model.Config()
  pool = ppo.TrainingPool(config, experiment, 2, 2, 0)
  assert all(not env.shield for env in pool.envs)
  actions = np.full((2, config.num_agents if pool.cooperative else 1), 100.0)
  _, _, _, _, _ = pool.step(actions)
  assert all(
      env.last_info is not None and not env.last_info.shielded
      for env in pool.envs
  )


@pytest.mark.parametrize(
    "experiment", ["individual_maintenance", "cooperative"]
)
def test_smoke_training_and_frozen_policy_round_trip(tmp_path, experiment):
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
  assert not policy.network.training
  assert all(
      not parameter.requires_grad for parameter in policy.network.parameters()
  )
  assert policy.metadata["steps"] == 16
  env = environment.CommonsEnvironment(policy.config)
  timestep = env.reset()
  actions = [
      p(obs) for p, obs in zip(policy.population(), timestep.observation)
  ]
  assert len(actions) == 4
  assert all(0 <= a <= 100 for a in actions)
  second = ppo.FrozenPolicy(checkpoint)
  second_actions = [
      p(obs) for p, obs in zip(second.population(), timestep.observation)
  ]
  np.testing.assert_array_equal(actions, second_actions)


def test_joint_stochastic_policy_uses_one_draw_for_all_agents(tmp_path):
  config = model.Config()
  torch.manual_seed(1)
  network = ppo.ActorCritic(5, 4, 100, 1)
  checkpoint = tmp_path / "test.pt"
  ppo.save_checkpoint(checkpoint, network, config, "cooperative", {"seed": 1})
  first = ppo.FrozenPolicy(checkpoint, stochastic=True, seed=42)
  second = ppo.FrozenPolicy(checkpoint, stochastic=True, seed=42)
  observations = environment.CommonsEnvironment(config).reset().observation
  a = [p(o) for p, o in zip(first.population(), observations)]
  b = [p(o) for p, o in zip(second.population(), observations)]
  np.testing.assert_array_equal(a, b)
  assert first(observations[0]) == a[0]


def test_frozen_evaluation_records_both_modes_without_updating_weights(
    tmp_path,
):
  from examples.non_spatial_commons import evaluate_trained

  config = model.Config()
  network = ppo.ActorCritic(5, 4, 100, 1)
  checkpoint = tmp_path / "test.pt"
  ppo.save_checkpoint(checkpoint, network, config, "cooperative", {"seed": 0})
  before = checkpoint.read_bytes()
  payload = evaluate_trained.evaluate_checkpoint(
      checkpoint, tmp_path / "evaluation", steps=2, random_starts=0
  )
  assert checkpoint.read_bytes() == before
  assert payload["training_shield"] is False
  assert payload["evaluation_weights_frozen"] is True
  assert len(payload["episodes"]) == 8
  assert {case["shield"] for case in payload["episodes"]} == {False, True}


def test_evaluation_requires_completed_training_sweep(tmp_path):
  from examples.non_spatial_commons import evaluate_trained

  folder = tmp_path / "cooperative" / "seed_0"
  folder.mkdir(parents=True)
  (folder / "final.pt").write_bytes(b"not a checkpoint")
  with pytest.raises(SystemExit):
    evaluate_trained.main(["--input", str(tmp_path)])
