# Spatial commons harvest with a backup-set safety filter

The spatial harvesting game from the ECE532 write-up ("Safety-Critical
Commons Harvest", Sections 2 and 4.1). A central regulator filters joint moves
so that no agent ever starves. Like `examples/non_spatial_commons`, this is a
standalone Python `dm_env` experiment that needs only NumPy and dm-env. It does
not import Lab2D and is not registered with `meltingpot.substrate.build`.

## Run

From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r examples/spatial_commons/requirements-dev.txt
.venv/bin/python -m pytest -n 8 examples/spatial_commons
.venv/bin/python -m examples.spatial_commons.evaluate --suite --episodes 3 --steps 1000
```

Results go to `examples/results/spatial_commons/` (git-ignored): a
`summary.json` file and a per-round `trajectories.csv` file.
`--shield on|off|both` selects the conditions. `--policies` takes one name for
everyone, or one name per agent. Every constant is a flag. `--growth-rate`
accepts `0.015` or `3/200`.

## Dynamics

| Item | Choice |
| --- | --- |
| Grid | 10x10, 0-indexed, no obstacles |
| Agents | 4, starting on homes (2,2), (2,7), (7,2), (7,7) |
| Initial apples | The five tiles of each home's cross are ripe; all other tiles start at 0 |
| Actions | `stay, up, down, left, right` with displacements (0,0), (-1,0), (1,0), (0,-1), (0,1) |
| Harvest | Eat if the tile an agent ends on was ripe at the start of the round |
| Energy | `E' = max(0, min(E_max, E - delta + eta*h))`. Failure (end of episode) when any `E = 0` |
| Regrowth | `z' = min(1, z+ + r * #ripe 8-neighbours in the post-harvest map)` (Eq. 7). An eaten tile regrows in the same round |
| Reward | Apples eaten this round (0/1) per agent. No sanction or death penalty |
| Observation | All positions, the ripe-apple map, own energy, player index, round |
| Defaults | `delta=1, eta=20, E_max=40, E_init=20, r=3/200` |

**Movement** (resolves what the write-up left unspecified): moves are
simultaneous, and entering a tile that is being vacated is allowed.

1. A move off the grid executes as `stay`.
2. If several agents move to the same tile, the lowest index wins and the
   others stay.
3. A mover whose target holds an agent that does not move (it chose to stay or
   was blocked itself) stays. This rule repeats until nothing changes, so
   blocked chains propagate backwards.
4. Chains resolve, but cycles are blocked: swaps and 2x2 rotations.

Ripeness is stored exactly as integers: with `r = num/den`, a tile holds
`z*den`. All core and filter checks are therefore exact, with no float
tolerance.

## Safety

**k-neighbour staying core** (write-up Section 3). Fix `k = Config.core_k`
(default 4), `T_k = ceil(1/(k r))` and `tau_k(z) = ceil((1-z)/(k r))`.
`Omega_core` is the set of states in which, for every agent:

- the agent is on its home tile,
- at least `k` of the home's 8 neighbours are ripe (neighbourhoods are
  clipped at the border; the home tile itself does not count), and
- `E_i > delta * tau_k(z(home_i))`.

Assumption 1 (`Config.core_condition_violations()`): homes pairwise at
inf-distance `>= 2` (border homes allowed), `T_k*delta <= eta` and
`E_max > (T_k-1)*delta`. Under it, all-stay keeps `Omega_core` invariant. The
filter refuses to run without it. The defaults give `T_4 = 17 <= 20`, and the
default start (ripe crosses) is in `Omega_core` for `k <= 4`.

**Backup set.** The backup policy `pi_B` walks each agent home, making vertical
moves first and then horizontal moves, and then stays. `Omega_B` (`Omega_B^H`
in the write-up) is the set of states from which rolling out `pi_B` reaches
`Omega_core` within `horizon` rounds (default `H = 64`) with no starvation.
`model.rounds_to_core` runs this rollout on raw arrays and is checked against
the plain `rounds_to_core_reference`; it stops early (outside `Omega_B`) once
no tile is ripe, since nothing can ever ripen again. Properties:

- `Omega_core ⊆ Omega_B`.
- `Omega_B` is controlled invariant under `pi_B`, because the rollout from the
  successor is the tail of the rollout from the current state.
- `Omega_B` is contained in the viability kernel.

Blocking, deadlocks and eaten crosses are all handled by simulating the real
dynamics. States where the backup policy would fail are simply excluded,
which makes the set conservative but never unsound.

**Filter** (`model.SafetyFilter`): return the joint action that overrides the
fewest agents such that the successor is in `Omega_B`. Ties are broken in this
order:

1. overridden agents get their backup action;
2. subsets of agents in lexicographic order;
3. actions in `ACTION_NAMES` order.

A safe proposal passes unchanged. `pi_B(s)` is always a candidate, so the
filter can always find a safe action. Candidates that resolve to the same
positions share one rollout, and the successor the filter certified last
round is not re-checked.

**Numerical checks.** `python -m examples.spatial_commons.verify_safety`
samples random core states, random `Omega_B^H` states and filtered closed
loops (random, greedy and move-away-from-home proposals) for several
`(homes, k, r, eta, delta, E_max)` satisfying Assumption 1, including border
homes and homes at inf-distance exactly 2, and exits nonzero on any
counterexample. `--quick` takes about 7 minutes.

## Random starts (Assumption 2)

`starts.py` rejection-samples initial states in `Omega_B^H` for the fixed
homes and constants:

- positions: `M` distinct tiles uniform over the whole grid;
- energies: independent uniform integers in `{min_energy, ..., E_max}`
  (`--min-energy`, default 15);
- ripeness: each tile is ripe with probability `ripe_prob` (default 0.2, the
  density of the default board), otherwise uniform over the unripe units
  `{0, 1/den, ..., 1 - 1/den}`. Tiles near homes are not special-cased.

A draw is kept iff `model.in_backup_set` holds. At the defaults about 22%
of draws are accepted (1000/4532 for seed 1). Everything comes from one
seeded generator, so the same `(seed, count)` gives the same starts.

## Fixed policies

| Policy | Behaviour |
| --- | --- |
| `stay` | Resting baseline (Section 4.2) |
| `greedy` | Head for the nearest free ripe apple (Manhattan distance); stay if none |
| `hungry` | Greedy while `E <= E_max/2`, otherwise stay |
| `random` | Uniform random move, with a seeded stream per agent |

## Known limitations

- The admissible range of `r` is narrow. The bound `r < 1/63` (so that greedy
  agents can deplete the board) comes from a regrowth maximum `G = 252`. That
  value was found by numerical search on the 10x10 board, not proved.
- The filter costs one rollout of up to `horizon` rounds per distinct
  candidate outcome. At the defaults, that is about 1 ms per round for random
  proposals and 15-50 ms per round for greedy ones, which are overridden in
  most rounds.
- The lexicographic tie-break overrides lower-index agents first. A fairness
  tie-break is a design choice that has not been made yet.

## PPO policies

`ppo.py`, `train.py` and `evaluate_trained.py` mirror the non-spatial PPO
pipeline. They need `torch==2.9.1` (CPU build).

- **Training is always unshielded.** The filter never runs during training;
  it is applied only when frozen checkpoints are evaluated.
- **Policy:** categorical over the 5 moves, with a separate 2x128 tanh actor
  and critic.
- **Features:** ripe map, own position, other agents' positions, own energy,
  player one-hot.
- **Optimisation:** PPO (clip 0.2, entropy 0.01, target KL 0.03), 16 envs x
  128-step rollouts, episodes of up to 256 rounds.
- **Training starts:** the default board with energies drawn from [10, 30].

| Experiment | Learner | Partners | Reward |
| --- | --- | --- | --- |
| `individual_stay` | agent 0 | stay | agent 0's apples |
| `individual_greedy` | agent 0 | greedy | agent 0's apples |
| `cooperative` | all four (shared weights) | none | team apples / 4 |
| `selfish_shared` | all four (shared weights) | none | own apples |
| `selfish_independent` | all four (4 separate networks) | none | own apples |

In every experiment, the episode ends for everyone as soon as any agent
starves.

Evaluation plays each frozen checkpoint for 1,000 rounds from 9 starts, all in
`Omega_B`:

- the default start;
- 4 starts with held-out energies at the homes;
- 4 starts with held-out positions.

Each start is played deterministically (argmax) and stochastically (sampled),
with the filter both off and on.

```sh
.venv/bin/python -m examples.spatial_commons.train --experiments cooperative --seeds 0
.venv/bin/python -m examples.spatial_commons.evaluate_trained --checkpoint \
    examples/results/spatial_commons/ppo/cooperative/seed_0/final.pt
```

On the Slurm cluster, see `runs/2026-11-04-spatial-ppo/*.sbatch`.

Additional experiments and options:

| Experiment | Learner | Reward |
| --- | --- | --- |
| `cooperative_independent` | 4 separate networks | team apples / 4 |
| `centralized` | one network on the joint state (ripe map, every position and energy), factorised head: one 5-way categorical per agent over a shared encoder | team apples / 4 |

`train.py --starts random` draws training starts from the `Omega_B^H` sampler
(seed stream disjoint from the evaluation starts). `--train-shield` runs the
filter inside the training environments; the stored actions and
log-probabilities are then the proposals, while rewards and transitions follow
the filtered actions. Both default off, which reproduces earlier checkpoints.

## Baseline comparison (filter off vs on)

`baselines.py` evaluates five methods from the same random starts, each with
the filter off and on, with paired seeds (start `j` uses policy seed
`seed + j` in both conditions):

| Method | Definition |
| --- | --- |
| `random` | each agent picks one of the 5 moves uniformly |
| `greedy` | `GreedyPolicy`: step (vertical first) toward the nearest ripe tile no other agent stands on (Manhattan; ties lowest row, then column); stay if none; ignores energy |
| `ippo_shared` | PPO `selfish_shared`: one network for all agents, own observation, own-apple reward |
| `ippo_shared_team` | PPO `cooperative`: as `ippo_shared`, team reward (total apples / 4) |
| `ippo_hetero` | PPO `selfish_independent`: one network per agent, own-apple reward |
| `ippo_hetero_team` | PPO `cooperative_independent`: one network per agent, team reward |
| `central_ppo` | PPO `centralized`: joint state to joint action, team reward |

PPO is trained without the filter, from random starts (`--starts random`,
same `min_energy`) on 1000-round episodes (`--episode-length 1000`, the
evaluation horizon), and evaluated with sampled actions. Metrics per episode over a 1000-round horizon: total and
per-agent apples eaten (the undiscounted sum of environment rewards),
apples per round, the fraction of agent-actions overridden, the fraction of
rounds with at least one override, survival and the first failure round.
`summarize` reports mean ± 95% t-CI across starts per method and filter
condition; for PPO each start's metric is first averaged over the training
seeds (all seeds share the starts). The paired filter-on minus filter-off
difference in apples is treated the same way. Per-seed means are kept in
`summary.json` (`per_seed`).

```sh
bash examples/spatial_commons/slurm/smoke.sh   # ~2 min end to end on CPU
bash examples/spatial_commons/slurm/submit.sh  # full run on Slurm
```

The smoke test writes to `examples/results/spatial_commons/baselines_smoke/`.
The full run writes to `runs/2026-11-06-spatial-baselines/`: `starts.json`,
`ppo/`, `episodes/`, and the combined `episodes.csv`, `summary.csv`,
`summary.json`, `table.md` and `table.tex`. Settings live in
`slurm/common.sh` and can be overridden by environment variables.
