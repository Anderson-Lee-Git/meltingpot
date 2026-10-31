"""Small continuous-action CPU PPO and frozen-policy adapters.

Policies sample Gaussian latent requests z, then request K*sigmoid(z). PPO uses
latent log probabilities: the fixed transformation Jacobian cancels in the
importance ratio. Actions retained for learning are proposals, never the
resource-rationed or shield-certified harvests.
"""

import dataclasses
import math
from pathlib import Path

from examples.non_spatial_commons import environment
from examples.non_spatial_commons import model
from examples.non_spatial_commons import policies
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

EXPERIMENTS = ("individual_maintenance", "individual_msy", "cooperative")


def features(observation, config, cooperative):
  if cooperative:
    return np.concatenate((
        [float(observation["RESOURCE"]) / config.carrying_capacity],
        observation["ALL_ENERGIES"] / config.max_energy,
    ))
  return environment.policy_features(observation, config)


class ActorCritic(nn.Module):
  """Separate two-layer actor and critic, with bounded continuous requests."""

  def __init__(
      self, input_dim, action_dim, capacity, initial_harvest, hidden=64
  ):
    super().__init__()
    self.capacity = capacity
    self.actor = nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, action_dim),
    )
    self.critic = nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 1),
    )
    self.log_std = nn.Parameter(torch.full((action_dim,), math.log(0.3)))
    for net in (self.actor, self.critic):
      for layer in net:
        if isinstance(layer, nn.Linear):
          nn.init.orthogonal_(layer.weight, math.sqrt(2))
          nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(self.actor[-1].weight, 0.01)
    probability = initial_harvest / capacity
    nn.init.constant_(
        self.actor[-1].bias, math.log(probability / (1 - probability))
    )
    nn.init.orthogonal_(self.critic[-1].weight, 1.0)

  def distribution(self, x):
    return Normal(self.actor(x), self.log_std.clamp(-3, 0.8).exp())

  def value(self, x):
    return self.critic(x).squeeze(-1)

  def sample(self, x):
    distribution = self.distribution(x)
    latent = distribution.sample()
    return (
        self.capacity * torch.sigmoid(latent),
        latent,
        distribution.log_prob(latent).sum(-1),
        self.value(x),
    )

  def evaluate(self, x, latent):
    distribution = self.distribution(x)
    return (
        distribution.log_prob(latent).sum(-1),
        distribution.entropy().sum(-1),
        self.value(x),
    )


def generalized_advantage(
    rewards, values, next_values, terminated, ended, gamma=0.99, gae_lambda=0.95
):
  """Bootstrap time-limit truncations but stop GAE across every reset."""
  advantages = np.zeros_like(rewards)
  running = np.zeros(rewards.shape[1], dtype=np.float32)
  for t in reversed(range(rewards.shape[0])):
    delta = rewards[t] + gamma * (~terminated[t]) * next_values[t] - values[t]
    running = delta + gamma * gae_lambda * (~ended[t]) * running
    advantages[t] = running
  return advantages, advantages + values


class TrainingPool:
  """Independent unshielded environments; no shielded training option exists."""

  def __init__(self, config, experiment, num_envs, episode_length, seed):
    if experiment not in EXPERIMENTS:
      raise ValueError(f"Unknown experiment {experiment}")
    self.config = config
    self.experiment = experiment
    self.cooperative = experiment == "cooperative"
    self.rng = np.random.default_rng(seed)
    self.envs = [
        environment.CommonsEnvironment(
            config, shield=False, episode_length=episode_length
        )
        for _ in range(num_envs)
    ]
    partner_name = (
        "msy_share" if experiment == "individual_msy" else "maintenance"
    )
    self.partners = policies.make_population((partner_name,), config)
    self.observations = [self._reset(env) for env in self.envs]
    self.episode_returns = np.zeros(num_envs)
    self.episode_lengths = np.zeros(num_envs, dtype=int)

  def _reset(self, env):
    # Randomized viable starts, but no action filtering or intervention. The
    # depleted E=0 recovery scenario is deliberately held out for evaluation.
    state = model.State(
        float(self.rng.uniform(0.2, 1.0) * self.config.carrying_capacity),
        tuple(
            self.rng.uniform(0.2, 0.8, size=self.config.num_agents)
            * self.config.max_energy
        ),
    )
    return env.reset(state=state).observation

  def current_features(self):
    return np.asarray(
        [
            features(obs[0], self.config, self.cooperative)
            for obs in self.observations
        ],
        dtype=np.float32,
    )

  def step(self, actions):
    next_features, rewards, terminated, ended, completed = [], [], [], [], []
    for i, env in enumerate(self.envs):
      if self.cooperative:
        requests = actions[i]
      else:
        requests = [
            partner(obs)
            for partner, obs in zip(self.partners, self.observations[i])
        ]
        requests[0] = float(actions[i, 0])
      timestep = env.step(requests)
      reward = sum(timestep.reward) if self.cooperative else timestep.reward[0]
      scale = (
          self.config.msy
          if self.cooperative
          else self.config.msy / self.config.num_agents
      )
      rewards.append(reward / scale)
      next_features.append(
          features(timestep.observation[0], self.config, self.cooperative)
      )
      terminated.append(timestep.last() and timestep.discount == 0)
      ended.append(timestep.last())
      self.episode_returns[i] += reward
      self.episode_lengths[i] += 1
      if timestep.last():
        completed.append({
            "return": float(self.episode_returns[i]),
            "length": int(self.episode_lengths[i]),
            "failed": env.state.failed,
        })
        self.observations[i] = self._reset(env)
        self.episode_returns[i] = 0
        self.episode_lengths[i] = 0
      else:
        self.observations[i] = timestep.observation
    return (
        np.asarray(next_features, np.float32),
        np.asarray(rewards, np.float32),
        np.asarray(terminated, bool),
        np.asarray(ended, bool),
        completed,
    )


def save_checkpoint(path, network, config, experiment, metadata):
  torch.save(
      {
          "format_version": 1,
          "config": dataclasses.asdict(config),
          "experiment": experiment,
          "hidden": network.actor[0].out_features,
          "training_shield": False,
          "state_dict": network.state_dict(),
          "metadata": metadata,
      },
      path,
  )


class FrozenPolicy:
  """One immutable checkpoint; joint decisions are cached across agent calls."""

  def __init__(self, path: Path, *, stochastic=False, seed=0):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload["format_version"] != 1
        or payload["training_shield"] is not False
    ):
      raise ValueError("Expected a version-1 unshielded-training checkpoint")
    self.config = model.Config(**payload["config"])
    self.experiment = payload["experiment"]
    self.cooperative = self.experiment == "cooperative"
    self.stochastic = stochastic
    self.generator = torch.Generator().manual_seed(seed)
    self.network = ActorCritic(
        self.config.num_agents + 1
        if self.cooperative
        else 2 * self.config.num_agents + 1,
        self.config.num_agents if self.cooperative else 1,
        self.config.carrying_capacity,
        self.config.metabolic_cost,
        payload["hidden"],
    )
    self.network.load_state_dict(payload["state_dict"])
    self.network.eval()
    self.network.requires_grad_(False)
    self.metadata = payload["metadata"]
    self._cache_key = None
    self._cache_actions = None

  def __call__(self, observation):
    key = (
        float(observation["RESOURCE"]),
        tuple(observation["ALL_ENERGIES"]),
        int(observation["ROUND"]),
    )
    if not self.cooperative or key != self._cache_key:
      x = torch.as_tensor(
          features(observation, self.config, self.cooperative),
          dtype=torch.float32,
      )
      with torch.no_grad():
        latent = self.network.actor(x)
        if self.stochastic:
          noise = torch.randn(latent.shape, generator=self.generator)
          latent = latent + self.network.log_std.clamp(-3, 0.8).exp() * noise
        self._cache_actions = (
            self.config.carrying_capacity * torch.sigmoid(latent)
        ).numpy()
      self._cache_key = key
    index = int(observation["PLAYER_INDEX"]) if self.cooperative else 0
    return float(self._cache_actions[index])

  def population(self):
    if self.cooperative:
      return (self,) * self.config.num_agents
    partner = (
        "msy_share" if self.experiment == "individual_msy" else "maintenance"
    )
    result = list(policies.make_population((partner,), self.config))
    result[0] = self
    return tuple(result)
