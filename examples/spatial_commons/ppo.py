"""Small CPU PPO for the spatial commons, with frozen-policy adapters.

The policy is categorical over the five moves. Training is unshielded by
default: the actions stored for learning are the agents' proposals, and no
filter runs. With `shield=True` the filter runs inside the training
environments (rewards and transitions follow the filtered actions, while the
log-probabilities are those of the proposals). The safety filter is otherwise
applied only at evaluation time to frozen checkpoints.

Training starts are either the default board (`starts="default"`, energies
uniform in [10, 30]) or fresh draws from the Omega_B^H start sampler
(`starts="random"`, see starts.py), from a seed stream disjoint from the
evaluation starts.

Experiments:
  individual_stay    Agent 0 learns; agents 1-3 always stay at home.
  individual_greedy  Agent 0 learns; agents 1-3 run the greedy policy.
  cooperative        One shared policy controls all four agents
                     (decentralised, parameter sharing) and is trained on the
                     team reward, i.e. total apples / num_agents.
  selfish_shared     One shared policy controls all four agents, but each
                     agent is rewarded only for its own apples.
  selfish_independent  Four separate networks (independent PPO), each trained
                     on its own agent's apples.
  cooperative_independent  Four separate networks, each trained on the team
                     reward.
  centralized        One network observes the joint state (ripe map, every
                     agent's position and energy) and outputs the joint
                     action through a factorised head: one categorical per
                     agent over a shared encoder, so log pi(u|s) is the sum
                     of the per-agent log-probabilities. One team value,
                     trained on the team reward.

In every experiment an episode ends for everyone when any agent starves.
"""

import dataclasses
from fractions import Fraction
import math
from pathlib import Path

from examples.spatial_commons import environment
from examples.spatial_commons import model
from examples.spatial_commons import policies
from examples.spatial_commons import starts as starts_lib
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.distributions import Independent

EXPERIMENTS = (
    "individual_stay",
    "individual_greedy",
    "cooperative",
    "selfish_shared",
    "selfish_independent",
    "cooperative_independent",
    "centralized",
)
MULTI_AGENT = (
    "cooperative",
    "selfish_shared",
    "selfish_independent",
    "cooperative_independent",
)
INDEPENDENT = ("selfish_independent", "cooperative_independent")
TEAM_REWARD = ("cooperative", "cooperative_independent", "centralized")
CENTRALIZED = ("centralized",)
START_KINDS = ("default", "random")
# Training-start seeds are offset so they never coincide with the
# evaluation-start seeds used by baselines.py.
TRAINING_START_SEED_OFFSET = 1_000_003
PARTNERS = {"individual_stay": "stay", "individual_greedy": "greedy"}
FORMAT_VERSION = 1


def feature_dim(config):
  return 3 * config.grid_size**2 + 1 + config.num_agents


def features(observation, config):
  """Ripe map, own position, other agents, own energy, player one-hot."""
  n, m = config.grid_size, config.num_agents
  i = int(observation["PLAYER_INDEX"])
  positions = observation["POSITIONS"]
  own = np.zeros((n, n), np.float32)
  own[positions[i, 0], positions[i, 1]] = 1
  others = np.zeros((n, n), np.float32)
  for j in range(m):
    if j != i:
      others[positions[j, 0], positions[j, 1]] = 1
  index = np.zeros(m, np.float32)
  index[i] = 1
  return np.concatenate((
      observation["APPLES"].astype(np.float32).ravel(),
      own.ravel(),
      others.ravel(),
      [float(observation["ENERGY"]) / config.max_energy],
      index,
  )).astype(np.float32)


def joint_feature_dim(config):
  return (1 + config.num_agents) * config.grid_size**2 + config.num_agents


def joint_features(state, config):
  """Ripe map, one position plane per agent, every agent's energy."""
  n, m = config.grid_size, config.num_agents
  planes = np.zeros((1 + m, n, n), np.float32)
  planes[0] = state.ripeness == config.ripe_units
  for i, (r, c) in enumerate(state.positions):
    planes[1 + i, r, c] = 1
  energies = np.asarray(state.energies, np.float32) / config.max_energy
  return np.concatenate((planes.ravel(), energies)).astype(np.float32)


class ActorCritic(nn.Module):
  """Separate two-layer tanh actor and critic; near-uniform initial policy."""

  def __init__(self, input_dim, hidden=128, num_actions=model.NUM_ACTIONS):
    super().__init__()

    def mlp(out):
      return nn.Sequential(
          nn.Linear(input_dim, hidden),
          nn.Tanh(),
          nn.Linear(hidden, hidden),
          nn.Tanh(),
          nn.Linear(hidden, out),
      )

    self.actor = mlp(num_actions)
    self.critic = mlp(1)
    for net in (self.actor, self.critic):
      for layer in net:
        if isinstance(layer, nn.Linear):
          nn.init.orthogonal_(layer.weight, math.sqrt(2))
          nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(self.actor[-1].weight, 0.01)
    nn.init.orthogonal_(self.critic[-1].weight, 1.0)

  def distribution(self, x):
    return Categorical(logits=self.actor(x))

  def value(self, x):
    return self.critic(x).squeeze(-1)

  def sample(self, x):
    distribution = self.distribution(x)
    action = distribution.sample()
    return action, distribution.log_prob(action), self.value(x)

  def evaluate(self, x, action):
    distribution = self.distribution(x)
    return (
        distribution.log_prob(action),
        distribution.entropy(),
        self.value(x),
    )


class CentralizedActorCritic(ActorCritic):
  """Joint-state actor with one 5-way categorical head per agent."""

  def __init__(self, input_dim, num_agents, hidden=128):
    super().__init__(input_dim, hidden, num_agents * model.NUM_ACTIONS)
    self.num_agents = num_agents

  def distribution(self, x):
    logits = self.actor(x).unflatten(-1, (self.num_agents, model.NUM_ACTIONS))
    return Independent(Categorical(logits=logits), 1)


def build_network(config, experiment, hidden=128):
  if experiment in CENTRALIZED:
    return CentralizedActorCritic(
        joint_feature_dim(config), config.num_agents, hidden
    )
  return ActorCritic(feature_dim(config), hidden)


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


def training_start(config, rng):
  """Default board with homes, energies drawn uniformly from [10, 30]."""
  state = model.default_state(config)
  energies = tuple(int(e) for e in rng.integers(10, 31, config.num_agents))
  return model.State(state.positions, energies, state.ripeness)


class TrainingPool:
  """Independent training environments. Rows are learner decisions.

  Individual experiments have one row per environment (agent 0); the
  decentralised multi-agent experiments have num_agents rows per environment;
  the centralized experiment has one row per environment whose action is the
  joint action.
  """

  def __init__(
      self,
      config,
      experiment,
      num_envs,
      episode_length,
      seed,
      *,
      shield=False,
      starts="default",
      start_ripe_prob=starts_lib.DEFAULT_RIPE_PROB,
      start_min_energy=starts_lib.DEFAULT_MIN_ENERGY,
  ):
    if experiment not in EXPERIMENTS:
      raise ValueError(f"Unknown experiment {experiment}")
    if starts not in START_KINDS:
      raise ValueError(f"Unknown start kind {starts}")
    self.config = config
    self.experiment = experiment
    self.cooperative = experiment in TEAM_REWARD
    self.centralized = experiment in CENTRALIZED
    self.multi_agent = experiment in MULTI_AGENT
    self.learners = config.num_agents if self.multi_agent else 1
    self.num_networks = config.num_agents if experiment in INDEPENDENT else 1
    self.rng = np.random.default_rng(seed)
    self.start_stream = (
        starts_lib.StartStream(
            config,
            TRAINING_START_SEED_OFFSET + seed,
            ripe_prob=start_ripe_prob,
            min_energy=start_min_energy,
        )
        if starts == "random"
        else None
    )
    self.envs = [
        environment.SpatialCommonsEnvironment(
            config, shield=shield, episode_length=episode_length
        )
        for _ in range(num_envs)
    ]
    self.partners = (
        None
        if self.multi_agent or self.centralized
        else policies.make_population((PARTNERS[experiment],), config)
    )
    self.observations = [self._reset(env) for env in self.envs]
    self.episode_returns = np.zeros(num_envs)
    self.episode_lengths = np.zeros(num_envs, dtype=int)

  @property
  def num_rows(self):
    return len(self.envs) * self.learners

  def _reset(self, env):
    if self.start_stream is not None:
      return env.reset(state=self.start_stream()).observation
    return env.reset(state=training_start(self.config, self.rng)).observation

  def current_features(self):
    if self.centralized:
      return np.asarray(
          [joint_features(env.state, self.config) for env in self.envs]
      )
    return np.asarray(
        [
            features(obs[i], self.config)
            for obs in self.observations
            for i in range(self.learners)
        ],
        dtype=np.float32,
    )

  def step(self, actions):
    """actions: (num_rows,) ints, or (num_rows, M) for centralized."""
    width = self.config.num_agents if self.centralized else self.learners
    actions = np.asarray(actions).reshape(len(self.envs), width)
    rewards, terminated, ended, completed, next_features = [], [], [], [], []
    for k, env in enumerate(self.envs):
      if self.multi_agent or self.centralized:
        joint = tuple(int(a) for a in actions[k])
      else:
        joint = [p(o) for p, o in zip(self.partners, self.observations[k])]
        joint[0] = int(actions[k, 0])
      timestep = env.step(tuple(joint))
      eaten = env.last_info.eaten
      if self.cooperative:
        row_rewards = [sum(eaten) / self.config.num_agents] * self.learners
      else:  # Selfish and individual learners: own apples only.
        row_rewards = list(eaten[: self.learners])
      # Logged return: population apples for multi-agent runs, else agent 0.
      self.episode_returns[k] += (
          sum(eaten) if self.multi_agent or self.centralized else eaten[0]
      )
      self.episode_lengths[k] += 1
      # The bootstrap value uses the successor before any reset.
      if self.centralized:
        next_features.append(joint_features(env.state, self.config))
      else:
        next_features.extend(
            features(timestep.observation[i], self.config)
            for i in range(self.learners)
        )
      rewards += row_rewards
      terminated += [timestep.last() and timestep.discount == 0] * self.learners
      ended += [timestep.last()] * self.learners
      if timestep.last():
        completed.append({
            "return": float(self.episode_returns[k]),
            "length": int(self.episode_lengths[k]),
            "failed": env.state.failed,
        })
        self.observations[k] = self._reset(env)
        self.episode_returns[k] = 0
        self.episode_lengths[k] = 0
      else:
        self.observations[k] = timestep.observation
    return (
        np.asarray(next_features, np.float32),
        np.asarray(rewards, np.float32),
        np.asarray(terminated, bool),
        np.asarray(ended, bool),
        completed,
    )


def config_to_dict(config):
  return {
      **dataclasses.asdict(config),
      "homes": [list(h) for h in config.homes],
      "growth_rate": str(config.growth_rate),
  }


def config_from_dict(payload):
  return model.Config(**{
      **payload,
      "homes": tuple(tuple(h) for h in payload["homes"]),
      "growth_rate": Fraction(payload["growth_rate"]),
  })


def save_checkpoint(
    path, networks, config, experiment, metadata, training_shield=False
):
  """`networks` is one ActorCritic or a list (one per agent if independent)."""
  if isinstance(networks, ActorCritic):
    networks = [networks]
  torch.save(
      {
          "format_version": FORMAT_VERSION,
          "config": config_to_dict(config),
          "experiment": experiment,
          "hidden": networks[0].actor[0].out_features,
          "training_shield": bool(training_shield),
          "state_dicts": [n.state_dict() for n in networks],
          "metadata": metadata,
      },
      path,
  )


class FrozenPolicy:
  """One immutable checkpoint used as a per-agent observation -> move map.

  Centralized checkpoints instead map the full state to the joint action
  (`act_joint`); `population()` is then not available.
  """

  def __init__(
      self, path: Path, *, stochastic=False, seed=0, allow_shielded=False
  ):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format_version"] != FORMAT_VERSION:
      raise ValueError("Unknown checkpoint format")
    if payload["training_shield"] and not allow_shielded:
      raise ValueError("Expected an unshielded-training checkpoint")
    self.training_shield = bool(payload["training_shield"])
    self.config = config_from_dict(payload["config"])
    self.experiment = payload["experiment"]
    self.centralized = self.experiment in CENTRALIZED
    self.multi_agent = self.experiment in MULTI_AGENT
    self.stochastic = stochastic
    self.generator = torch.Generator().manual_seed(seed)
    state_dicts = payload.get("state_dicts") or [payload["state_dict"]]
    self.networks = []
    for state_dict in state_dicts:
      network = build_network(self.config, self.experiment, payload["hidden"])
      network.load_state_dict(state_dict)
      network.eval()
      network.requires_grad_(False)
      self.networks.append(network)
    self.network = self.networks[0]
    self.metadata = payload["metadata"]

  def reseed(self, seed):
    """Restart the sampling stream (pairs filter-off/on episodes)."""
    self.generator = torch.Generator().manual_seed(seed)

  def act_joint(self, state):
    """Centralized checkpoints: full state -> joint action."""
    x = torch.as_tensor(joint_features(state, self.config))
    with torch.no_grad():
      logits = self.network.actor(x).view(
          self.config.num_agents, model.NUM_ACTIONS
      )
      if self.stochastic:
        probs = torch.softmax(logits, -1)
        picks = torch.multinomial(probs, 1, generator=self.generator)
        return tuple(int(a) for a in picks.squeeze(-1))
      return tuple(int(a) for a in logits.argmax(-1))

  def __call__(self, observation):
    if self.centralized:
      raise TypeError("Centralized checkpoints act on the joint state")
    x = torch.as_tensor(features(observation, self.config))
    index = int(observation["PLAYER_INDEX"]) % len(self.networks)
    with torch.no_grad():
      logits = self.networks[index].actor(x)
      if self.stochastic:
        probs = torch.softmax(logits, -1)
        return int(torch.multinomial(probs, 1, generator=self.generator))
      return int(logits.argmax())

  def population(self, seed=0):
    if self.centralized:
      raise TypeError("Centralized checkpoints have no per-agent population")
    if self.multi_agent:
      return (self,) * self.config.num_agents
    result = list(
        policies.make_population(
            (PARTNERS[self.experiment],), self.config, seed=seed
        )
    )
    result[0] = self
    return tuple(result)
