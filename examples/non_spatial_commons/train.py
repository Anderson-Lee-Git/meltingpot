"""Train individual and cooperative PPO policies, exclusively unshielded."""

import argparse
import collections
import csv
import dataclasses
import json
import math
from pathlib import Path
import time

from examples.non_spatial_commons import model
from examples.non_spatial_commons import ppo
import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
  total_steps: int = 200704
  num_envs: int = 16
  rollout_steps: int = 128
  episode_length: int = 256
  epochs: int = 6
  minibatch_size: int = 256
  learning_rate: float = 3e-4
  gamma: float = 0.99
  gae_lambda: float = 0.95
  clip_ratio: float = 0.2
  target_kl: float = 0.03

  def __post_init__(self):
    for name in (
        "total_steps",
        "num_envs",
        "rollout_steps",
        "episode_length",
        "epochs",
        "minibatch_size",
    ):
      if getattr(self, name) < 1:
        raise ValueError(f"{name} must be positive")
    if not 0 < self.gamma < 1 or not 0 <= self.gae_lambda <= 1:
      raise ValueError("Invalid discount or GAE parameter")
    if self.learning_rate <= 0 or self.clip_ratio <= 0 or self.target_kl <= 0:
      raise ValueError("Optimizer hyperparameters must be positive")


def train_one(config, training, experiment, seed, output):
  """No shielded evaluation or checkpoint selection occurs during training."""
  torch.manual_seed(seed)
  rng = np.random.default_rng(seed)
  torch.set_num_threads(2)
  torch.use_deterministic_algorithms(True)
  output.mkdir(parents=True, exist_ok=True)
  pool = ppo.TrainingPool(
      config, experiment, training.num_envs, training.episode_length, seed
  )
  initial_features = pool.current_features()
  action_dim = config.num_agents if pool.cooperative else 1
  network = ppo.ActorCritic(
      initial_features.shape[1],
      action_dim,
      config.carrying_capacity,
      config.metabolic_cost,
  )
  optimizer = torch.optim.Adam(
      network.parameters(), lr=training.learning_rate, eps=1e-5
  )
  batch_size = training.num_envs * training.rollout_steps
  updates = math.ceil(training.total_steps / batch_size)
  metadata = {
      "seed": seed,
      "training": dataclasses.asdict(training),
      "steps": 0,
      "device": "cpu",
      "reward": "total_harvest" if pool.cooperative else "agent_0_harvest",
  }
  ppo.save_checkpoint(
      output / "initial.pt", network, config, experiment, metadata
  )
  history = collections.deque(maxlen=100)
  all_episodes = 0
  start = time.perf_counter()
  with (output / "training.csv").open("w", newline="") as stream:
    writer = None
    for update in range(updates):
      optimizer.param_groups[0]["lr"] = training.learning_rate * (
          1 - update / updates
      )
      storage = collections.defaultdict(list)
      for _ in range(training.rollout_steps):
        x = torch.as_tensor(pool.current_features())
        with torch.no_grad():
          actions, latent, log_prob, value = network.sample(x)
        next_x, reward, terminated, ended, completed = pool.step(
            actions.numpy().astype(np.float64)
        )
        with torch.no_grad():
          next_value = network.value(torch.as_tensor(next_x))
        for key, array in [
            ("features", x.numpy()),
            ("latent", latent.numpy()),
            ("log_prob", log_prob.numpy()),
            ("value", value.numpy()),
            ("next_value", next_value.numpy()),
            ("reward", reward),
            ("terminated", terminated),
            ("ended", ended),
        ]:
          storage[key].append(array)
        history.extend(completed)
        all_episodes += len(completed)
      arrays = {key: np.asarray(value) for key, value in storage.items()}
      advantage, returns = ppo.generalized_advantage(
          arrays["reward"],
          arrays["value"],
          arrays["next_value"],
          arrays["terminated"],
          arrays["ended"],
          training.gamma,
          training.gae_lambda,
      )
      x = torch.as_tensor(arrays["features"].reshape(batch_size, -1))
      latent = torch.as_tensor(arrays["latent"].reshape(batch_size, -1))
      old_log_prob = torch.as_tensor(arrays["log_prob"].reshape(-1))
      advantages = torch.as_tensor(advantage.reshape(-1))
      advantages = (advantages - advantages.mean()) / (
          advantages.std(unbiased=False) + 1e-8
      )
      targets = torch.as_tensor(returns.reshape(-1))
      losses, kls = [], []
      stop = False
      for _ in range(training.epochs):
        permutation = rng.permutation(batch_size)
        for offset in range(0, batch_size, training.minibatch_size):
          ix = permutation[offset : offset + training.minibatch_size]
          log_prob, _, value = network.evaluate(x[ix], latent[ix])
          log_ratio = log_prob - old_log_prob[ix]
          ratio = log_ratio.exp()
          approx_kl = ((ratio - 1) - log_ratio).mean()
          if approx_kl.item() > training.target_kl:
            stop = True
            break
          actor_loss = -torch.minimum(
              ratio * advantages[ix],
              ratio.clamp(1 - training.clip_ratio, 1 + training.clip_ratio)
              * advantages[ix],
          ).mean()
          critic_loss = 0.5 * (value - targets[ix]).square().mean()
          loss = actor_loss + critic_loss
          optimizer.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(
              list(network.actor.parameters()) + [network.log_std], 0.5
          )
          torch.nn.utils.clip_grad_norm_(network.critic.parameters(), 0.5)
          optimizer.step()
          losses.append(float(loss.detach()))
          kls.append(float(approx_kl.detach()))
        if stop:
          break
      steps = (update + 1) * batch_size
      row = {
          "update": update + 1,
          "steps": steps,
          "elapsed_seconds": time.perf_counter() - start,
          "completed_episodes": all_episodes,
          "recent_mean_return": (
              np.mean([e["return"] for e in history]) if history else 0
          ),
          "recent_mean_length": (
              np.mean([e["length"] for e in history]) if history else 0
          ),
          "recent_failure_fraction": (
              np.mean([e["failed"] for e in history]) if history else 0
          ),
          "loss": np.mean(losses) if losses else 0,
          "approx_kl": np.mean(kls) if kls else 0,
          "latent_std": float(
              network.log_std.detach().clamp(-3, 0.8).exp().mean()
          ),
      }
      if not all(math.isfinite(float(v)) for v in row.values()):
        raise RuntimeError(f"Nonfinite training metric: {row}")
      if writer is None:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
      writer.writerow(row)
      stream.flush()
      if update == 0 or (update + 1) % 10 == 0 or update + 1 == updates:
        print(
            f"{experiment} seed={seed} steps={steps}/{updates*batch_size}"
            f" return={row['recent_mean_return']:.1f}"
            f" length={row['recent_mean_length']:.1f}"
            f" failure={row['recent_failure_fraction']:.0%}"
            f" fps={steps/row['elapsed_seconds']:.0f}",
            flush=True,
        )
  metadata.update(
      steps=updates * batch_size,
      elapsed_seconds=time.perf_counter() - start,
      completed_episodes=all_episodes,
  )
  ppo.save_checkpoint(
      output / "final.pt", network, config, experiment, metadata
  )
  (output / "metadata.json").write_text(
      json.dumps(
          {
              "config": dataclasses.asdict(config),
              "experiment": experiment,
              "training_shield": False,
              **metadata,
          },
          indent=2,
      )
      + "\n"
  )
  for env in pool.envs:
    env.close()
  return output / "final.pt"


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--experiments",
      nargs="+",
      choices=ppo.EXPERIMENTS,
      default=list(ppo.EXPERIMENTS),
  )
  parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
  parser.add_argument("--total-steps", type=int, default=200704)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--rollout-steps", type=int, default=128)
  parser.add_argument(
      "--episode-length",
      type=int,
      default=256,
      help="Maximum rounds per training episode (default: 256).",
  )
  parser.add_argument("--epochs", type=int, default=6)
  parser.add_argument(
      "--output",
      type=Path,
      default=Path("examples/results/non_spatial_commons/ppo"),
  )
  args = parser.parse_args(argv)
  training = TrainingConfig(
      total_steps=args.total_steps,
      num_envs=args.num_envs,
      rollout_steps=args.rollout_steps,
      episode_length=args.episode_length,
      epochs=args.epochs,
  )
  checkpoints = []
  for experiment in args.experiments:
    for seed in args.seeds:
      checkpoint = train_one(
          model.Config(),
          training,
          experiment,
          seed,
          args.output / experiment / f"seed_{seed}",
      )
      checkpoints.append(str(checkpoint.relative_to(args.output)))
  (args.output / "training_manifest.json").write_text(
      json.dumps(
          {
              "training_shield": False,
              "experiments": args.experiments,
              "seeds": args.seeds,
              "checkpoints": checkpoints,
              "training": dataclasses.asdict(training),
          },
          indent=2,
      )
      + "\n"
  )
  print(
      "All unshielded training runs complete. Final checkpoints are ready for"
      " evaluation.",
      flush=True,
  )


if __name__ == "__main__":
  main()
