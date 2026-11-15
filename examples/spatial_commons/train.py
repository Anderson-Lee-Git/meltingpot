"""Train spatial-commons PPO policies (unshielded unless --train-shield)."""

import argparse
import collections
import csv
import dataclasses
import json
import math
from pathlib import Path
import time

from examples.spatial_commons import model
from examples.spatial_commons import ppo
import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
  total_steps: int = 1_000_448
  num_envs: int = 16
  rollout_steps: int = 128
  episode_length: int = 256
  epochs: int = 4
  minibatch_size: int = 256
  learning_rate: float = 3e-4
  gamma: float = 0.99
  gae_lambda: float = 0.95
  clip_ratio: float = 0.2
  target_kl: float = 0.03
  entropy_coef: float = 0.01
  starts: str = "default"
  start_min_energy: int = 15
  shield: bool = False

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
    if self.starts not in ppo.START_KINDS:
      raise ValueError(f"starts must be one of {ppo.START_KINDS}")


def train_one(config, training, experiment, seed, output):
  """No evaluation or checkpoint selection occurs during training.

  The filter runs in the training environments only if training.shield.
  """
  torch.manual_seed(seed)
  rng = np.random.default_rng(seed)
  torch.set_num_threads(2)
  torch.use_deterministic_algorithms(True)
  output.mkdir(parents=True, exist_ok=True)
  pool = ppo.TrainingPool(
      config,
      experiment,
      training.num_envs,
      training.episode_length,
      seed,
      shield=training.shield,
      starts=training.starts,
      start_min_energy=training.start_min_energy,
  )
  # One network, or one per agent for independent PPO. Rows are ordered
  # environment-major, agent-minor, so network j owns rows j::K.
  k = pool.num_networks
  networks = [ppo.build_network(config, experiment) for _ in range(k)]
  optimizers = [
      torch.optim.Adam(n.parameters(), lr=training.learning_rate, eps=1e-5)
      for n in networks
  ]
  env_steps_per_update = training.num_envs * training.rollout_steps
  updates = math.ceil(training.total_steps / env_steps_per_update)
  metadata = {
      "seed": seed,
      "training": dataclasses.asdict(training),
      "steps": 0,
      "device": "cpu",
      "reward": (
          {
              "cooperative": "team_apples_per_agent",
              "selfish_shared": "own_apples",
              "selfish_independent": "own_apples",
              "cooperative_independent": "team_apples_per_agent",
              "centralized": "team_apples_per_agent",
          }.get(experiment, "agent_0_apples")
      ),
      "networks": k,
  }
  ppo.save_checkpoint(
      output / "initial.pt",
      networks,
      config,
      experiment,
      metadata,
      training.shield,
  )
  history = collections.deque(maxlen=100)
  all_episodes = 0
  start = time.perf_counter()

  def forward(x, method):
    """Apply each network to its own rows; returns arrays in row order."""
    outputs = None
    for j, network in enumerate(networks):
      result = method(network, x[j::k])
      if outputs is None:
        outputs = [
            torch.empty((x.shape[0],) + r.shape[1:], dtype=r.dtype)
            for r in result
        ]
      for out, r in zip(outputs, result):
        out[j::k] = r
    return outputs

  with (output / "training.csv").open("w", newline="") as stream:
    writer = None
    for update in range(updates):
      for optimizer in optimizers:
        optimizer.param_groups[0]["lr"] = training.learning_rate * (
            1 - update / updates
        )
      storage = collections.defaultdict(list)
      for _ in range(training.rollout_steps):
        x = torch.as_tensor(pool.current_features())
        with torch.no_grad():
          actions, log_prob, value = forward(x, lambda n, v: n.sample(v))
        next_x, reward, terminated, ended, completed = pool.step(
            actions.numpy()
        )
        with torch.no_grad():
          (next_value,) = forward(
              torch.as_tensor(next_x), lambda n, v: (n.value(v),)
          )
        for key, array in [
            ("features", x.numpy()),
            ("action", actions.numpy()),
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
      losses, kls, entropies = [], [], []
      for j, (network, optimizer) in enumerate(zip(networks, optimizers)):
        # Time x rows -> this network's columns, flattened.
        def mine(a):
          a = a[:, j::k]
          return torch.as_tensor(a.reshape((-1,) + a.shape[2:]))

        x, action, old_log_prob = (
            mine(arrays["features"]),
            mine(arrays["action"]),
            mine(arrays["log_prob"]),
        )
        advantages = mine(advantage)
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
        targets = mine(returns)
        batch_size = x.shape[0]
        stop = False
        for _ in range(training.epochs):
          permutation = rng.permutation(batch_size)
          for offset in range(0, batch_size, training.minibatch_size):
            ix = permutation[offset : offset + training.minibatch_size]
            log_prob, entropy, value = network.evaluate(x[ix], action[ix])
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
            loss = (
                actor_loss
                + critic_loss
                - training.entropy_coef * entropy.mean()
            )
            entropies.append(float(entropy.mean().detach()))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(network.critic.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach()))
            kls.append(float(approx_kl.detach()))
          if stop:
            break
      steps = (update + 1) * env_steps_per_update
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
          "entropy": np.mean(entropies) if entropies else 0,
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
            f"{experiment} seed={seed}"
            f" steps={steps}/{updates*env_steps_per_update}"
            f" return={row['recent_mean_return']:.1f}"
            f" length={row['recent_mean_length']:.1f}"
            f" failure={row['recent_failure_fraction']:.0%}"
            f" fps={steps/row['elapsed_seconds']:.0f}",
            flush=True,
        )
  metadata.update(
      steps=updates * env_steps_per_update,
      elapsed_seconds=time.perf_counter() - start,
      completed_episodes=all_episodes,
  )
  ppo.save_checkpoint(
      output / "final.pt",
      networks,
      config,
      experiment,
      metadata,
      training.shield,
  )
  (output / "metadata.json").write_text(
      json.dumps(
          {
              "config": ppo.config_to_dict(config),
              "experiment": experiment,
              "training_shield": training.shield,
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
  parser.add_argument("--total-steps", type=int, default=1_000_448)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--rollout-steps", type=int, default=128)
  parser.add_argument(
      "--episode-length",
      type=int,
      default=256,
      help="Maximum rounds per training episode (default: 256).",
  )
  parser.add_argument("--epochs", type=int, default=4)
  parser.add_argument(
      "--starts",
      choices=ppo.START_KINDS,
      default="default",
      help=(
          "Training starts: the default board, or random draws in Omega_B^H"
          " (starts.py)."
      ),
  )
  parser.add_argument(
      "--start-min-energy",
      type=int,
      default=15,
      help="Lowest initial energy of random starts (--starts random).",
  )
  parser.add_argument(
      "--train-shield",
      action="store_true",
      help="Run the safety filter inside the training environments.",
  )
  parser.add_argument(
      "--output",
      type=Path,
      default=Path("examples/results/spatial_commons/ppo"),
  )
  parser.add_argument(
      "--no-manifest",
      action="store_true",
      help="Skip the manifest (for one job of a Slurm array).",
  )
  parser.add_argument(
      "--manifest-only",
      action="store_true",
      help="Only write the manifest after all array jobs finished.",
  )
  args = parser.parse_args(argv)
  training = TrainingConfig(
      total_steps=args.total_steps,
      num_envs=args.num_envs,
      rollout_steps=args.rollout_steps,
      episode_length=args.episode_length,
      epochs=args.epochs,
      starts=args.starts,
      start_min_energy=args.start_min_energy,
      shield=args.train_shield,
  )
  if args.manifest_only:
    write_manifest(args, training)
    return
  for experiment in args.experiments:
    for seed in args.seeds:
      checkpoint = train_one(
          model.Config(),
          training,
          experiment,
          seed,
          args.output / experiment / f"seed_{seed}",
      )
      print(f"Wrote {checkpoint}", flush=True)
  if not args.no_manifest:
    write_manifest(args, training)


def write_manifest(args, training):
  """Written only once every requested run has a final checkpoint."""
  checkpoints = []
  for experiment in args.experiments:
    for seed in args.seeds:
      path = args.output / experiment / f"seed_{seed}" / "final.pt"
      if not path.exists():
        raise FileNotFoundError(f"Missing {path}; training is incomplete")
      checkpoints.append(str(path.relative_to(args.output)))
  (args.output / "training_manifest.json").write_text(
      json.dumps(
          {
              "training_shield": training.shield,
              "experiments": args.experiments,
              "seeds": args.seeds,
              "checkpoints": checkpoints,
              "training": dataclasses.asdict(training),
          },
          indent=2,
      )
      + "\n"
  )
  print(f"Wrote {args.output / 'training_manifest.json'}", flush=True)


if __name__ == "__main__":
  main()
