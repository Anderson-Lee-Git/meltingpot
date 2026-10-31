# Non-spatial ecological commons

This experiment implements Sections 1–3 of `non_spatial.tex`: known resource
and energy dynamics, a conservative controlled-invariant core, and an online
projection filter. It includes fixed policies and paired evaluations, with fixed-policy baselines and unshielded PPO training for individual and
cooperative policies. See [TRAINING.md](TRAINING.md) for the training workflow.

This is a Python continuous-action experiment inside the Melting Pot repo. It
is separate from the spatial `commons_harvest` Lab2D substrate and is not
registered with `meltingpot.substrate.build`. Importing it does not import the
full `meltingpot` package or load TensorFlow/Lab2D. Existing Melting Pot bot
policies use discrete action IDs; they need an adapter or a new continuous head
before they can act in this environment.

## Run

From the repository root, with Python 3.11 or newer:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r examples/non_spatial_commons/requirements.txt
.venv/bin/python -m examples.non_spatial_commons.evaluate \
  --suite --episodes 5 --steps 1000
```

If a repository-local environment already exists, use it instead of recreating
it. The working environment used for initial validation is Python 3.13; the
repo's `.python-version` is not required by this standalone experiment. The
minimal dependency set is NumPy and dm-env; a full Melting Pot installation is
not needed. Generated results default to the existing ignored
`examples/results/` directory.

For a particular mixed population:

```sh
.venv/bin/python -m examples.non_spatial_commons.evaluate \
  --policies greedy maintenance msy_share reserve_target \
  --episodes 5 --steps 1000 --shield both \
  --output examples/results/non_spatial_commons/mixed
```

One policy name is repeated across all agents. Otherwise provide exactly one
name per agent. `--shield on`, `off`, or `both` selects the comparison.
`--num-agents`, `--capacity`, `--growth-rate`, `--metabolic-cost`, and
`--max-energy` configure the model. Defaults are N=4, K=100, r=0.5, delta=1,
and E_max=10. Default initialization is R=R_MSY=62.5 and every E_i=5.

To test recovery from a depleted but certified starting state:

```sh
.venv/bin/python -m examples.non_spatial_commons.evaluate \
  --suite --initial-resource 13.3 --initial-energy 0 \
  --episodes 5 --steps 1000 \
  --output examples/results/non_spatial_commons/recovery
```

## Dynamics and filter

Every policy proposes a scalar `a_i` in [0,K]. The regulator receives the full
state and all proposals. The executed harvests must be nonnegative and sum to
at most the available resource. After harvesting:

```text
C = sum_i u_i
S = R - C
R_next = min(K, S + r*S*(1 - S/K))
E_i_next = min(E_max, E_i - delta + u_i)
```

Failure occurs at the first negative energy. Zero energy is still alive.
Negative energy is recorded and terminates the episode; it is never clamped
to zero. There is no inter-agent transfer of energy or resource replenishment
from zero stock. Energy conversion efficiency is one.

The analytic core requires 0 < r <= 2 and D=N*delta < MSY=r*K/4. Define:

```text
b_i(s) = max(0, delta - E_i)
b(s) = sum_i b_i(s)
S_crit = K/2 * (1 - sqrt(1 - 4*D/(r*K)))
Omega_core = {E_i >= 0 for all i, R >= S_crit + b(s)}
C(s) = max(b(s), R - K/2)
```

`model.SafetyFilter` solves the Euclidean projection of the requests subject to
`u_i >= b_i(s)` for every agent and `sum_i u_i <= C(s)`. It can **increase**
a request below an agent's metabolic floor. It returns feasible requests
unchanged. The cap and floors ensure recursive feasibility inside the core;
no learned safety critic is used. The implementation uses a scalar threshold
and feasible-side bisection rather than a QP dependency. The cap-equals-total-
floors case is handled directly.

### Numerical certification

At the exact mathematical boundary, roundoff can accumulate because the
lower regenerative equilibrium is unstable. For execution we use a slightly
tighter core: replace S_crit by `config.protected_escapement`, a threshold above
the analytic root by approximately
`64*ulp(K) / (r*sqrt(1-D/MSY))`. This creates a small positive regenerative
margin while remaining below K/2. The transition equations and harvest cap
are unchanged. `in_safe_core` checks this tighter bound without accepting
negative margin. Both thresholds are saved in `summary.json`.

Parameters too poorly resolved in float64 to construct this margin are
rejected. Initial states outside the numerically certified core are rejected
in shielded mode; they are not declared inevitably unsafe. For such states,
model predictive recovery/shielding would be a separate extension.

In **unshielded** mode we still project proposals onto the physical stock
budget, with zero floors and no ecological reserve. This prevents harvesting
nonexistent resource. All modes use the same Euclidean allocation rule; the
shield adds the metabolic floors and ecological cap. Corrections in an
unshielded trace are physical rationing, not safety-filter interventions.

## Fixed policies

| Policy | Request | Purpose |
| --- | --- | --- |
| `maintenance` | delta | Sustainable low-yield reference at the default initial state |
| `msy_share` | MSY/N | Cooperative maximum-yield reference at the default initial state |
| `greedy` | K | Overharvesting stress case |
| `reserve_target` | clip(delta + 0.5*(E_max/2 - E_i), 0, K) | Replenish individual reserves towards a target |
| `random` | Uniform[0,K] with an independent seeded stream per agent | Exploratory stress case |
| `abstain` | 0 | Underconsumption stress case; tests upward correction |

These are fixed decision rules. A state-dependent rule such as `reserve_target`
is still a fixed policy because its parameters do not learn. `--suite` runs
every homogeneous population plus a repeating greedy/maintenance/MSY/target
mixed population. Paired runs recreate the same random streams for each seed.

## Environment and future learned policies

`environment.CommonsEnvironment` implements `dm_env.Environment` with one
continuous scalar action per agent. Each player receives a dictionary with:

- `RESOURCE`: global resource stock, scalar float64.
- `ENERGY`: the player's energy, scalar float64.
- `ALL_ENERGIES`: full energy vector, shape (N,), float64.
- `PLAYER_INDEX`: zero-based scalar int32 player ID.
- `ROUND`: scalar int32 episode round.

Full-state observations match the report's fully observed model. Each
observation is an independent copy. Action, observation, and reward specs are
provided. Rewards are **executed harvests**, one per agent: no sanction,
quadratic penalty, death penalty, or automatic MSY-convergence objective has
been added. Failure terminates with discount 0; reaching `episode_length`
truncates with discount 1 so a learner can bootstrap. As in standard dm_env,
stepping after termination automatically resets and does not execute that
call's action.

A policy is any callable `policy(observation) -> float` returning a request in
[0,K]. `environment.policy_features(obs, config)` supplies a normalized vector
of length 2*N+1: R/K, all E_i/E_max, and a one-hot player ID. A trained actor can
use these same features and scale its bounded output to [0,K].

```python
from examples.non_spatial_commons import environment, evaluate, model, policies

config = model.Config()
population = list(policies.make_population(("maintenance",), config))
# Later: population[0] = trained_policy_callable
result = evaluate.run_episode(config, population, shield=True, steps=1000)
print(result.summary)
```

For later RL, retain the requested action as the learner's action and use the
reward/next state produced by execution of the filtered action. An evaluation
or replay pipeline should also log the certified action and correction. This
models the policy interacting with the shielded environment, including cases
where many proposals execute the same harvest. Reset recurrent policy state
and exploration RNGs at the appropriate episode boundary; the runner does not
implicitly reset arbitrary trained callables.

The initial PPO experiments train one agent against fixed maintenance or
MSY-share partners and a cooperative joint controller, then evaluate frozen
policies with and without shielding. Training multiple independently learning
agents can follow after this baseline. Unshielded individual and cooperative PPO training is now implemented; see
[TRAINING.md](TRAINING.md) for the uv environment, checkpoints, and paired
post-training evaluations.

## Results and diagnostics

Each evaluation writes:

- `summary.json`: configuration, both escapement thresholds, per-episode
  survival/failure round, per-agent returns, resource/energy statistics, and
  signed-correction counts.
- `trajectories.csv`: per-round requested and executed harvests, all individual
  energies, resource, escapement, cap, core margin, failure, and correction size.

`harvest_per_horizon_step` divides by the requested horizon, counting the
remaining horizon after failure as zero harvest. The separately named
`mean_harvest_per_executed_step` averages only rounds actually executed; it
must not be mistaken for comparable full-horizon productivity. Correction
statistics include physical rationing in unshielded mode.

The baseline runs documented in `BASELINE_RESULTS.md` cover both the default
and low-resource recovery initialization. They demonstrate behavior under
these conditions; the safety argument comes from the invariant core, not
from finite rollouts. Safety does not imply MSY: maintenance and abstention
can survive with harvest below MSY.

Optional comparison plots:

```sh
.venv/bin/python -m pip install matplotlib
.venv/bin/python -m examples.non_spatial_commons.plot \
  --input examples/results/non_spatial_commons/default --max-rounds 100
```

Plots show the selected episode, not an average over seeds. An x marks a
starvation endpoint; that trajectory stops there. Select populations with
`--populations greedy abstain mixed` or choose other names from the suite.

## Tests

```sh
.venv/bin/python -m pip install -r examples/non_spatial_commons/requirements-dev.txt
.venv/bin/python -m pytest -n 0 examples/non_spatial_commons -q
.venv/bin/python -m pyink --check examples/non_spatial_commons
```

Tests check physical dynamics and failure semantics, the projection against an
independent active-set optimum, invariant-core preservation across parameters
and arbitrary requests, exact operational-boundary initialization, interface
specs, reproducibility, the fixed baselines, and emitted evaluation data. The
full spatial Melting Pot suite is not required by this experiment.

Unshielded PPO results are recorded in [RL_RESULTS.md](RL_RESULTS.md).
