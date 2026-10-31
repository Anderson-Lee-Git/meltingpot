# Unshielded PPO training: frozen-policy evaluation

Nine runs: three experiments x three seeds, each with 200,704 training transitions.
All training was unshielded; final checkpoints were selected by the fixed budget, then frozen.
The uv-created isolated .venv uses Python 3.13.3 and CPU PyTorch 2.9.1.
The implementation has 67 passing targeted tests.

Evaluation: 1,000-round horizon, two fixed starts plus ten held-out certified starts per checkpoint.
Each selection/mode cell contains 36 episodes (three training seeds x twelve starts).
Stochastic evaluations use matching action-noise streams for the two shielding modes.

## Deterministic actions

| Experiment | Shield | Survival | Population harvest/round | Agent 0 harvest/round | Corrected rounds |
| --- | --- | --- | --- | --- | --- |
| individual_maintenance | off | 21/36 | 7.141 | 5.383 | 21.6% |
| individual_maintenance | on | 36/36 | 12.072 | 9.075 | 0.3% |
| individual_msy | off | 21/36 | 7.279 | 1.796 | 26.1% |
| individual_msy | on | 36/36 | 12.419 | 3.065 | 0.3% |
| cooperative | off | 24/36 | 7.955 | 1.817 | 21.0% |
| cooperative | on | 36/36 | 11.893 | 2.717 | 0.3% |

## Stochastic actions

| Experiment | Shield | Survival | Population harvest/round | Agent 0 harvest/round | Corrected rounds |
| --- | --- | --- | --- | --- | --- |
| individual_maintenance | off | 8/36 | 4.423 | 3.335 | 22.5% |
| individual_maintenance | on | 36/36 | 11.962 | 8.976 | 2.6% |
| individual_msy | off | 14/36 | 6.142 | 1.513 | 26.3% |
| individual_msy | on | 36/36 | 12.431 | 3.130 | 10.6% |
| cooperative | off | 4/36 | 4.238 | 0.944 | 23.7% |
| cooperative | on | 36/36 | 12.204 | 2.769 | 6.2% |

## Interpretation and limits

- All 216 shielded episodes survived. Unshielded survival was 66/108 for deterministic policies and 26/108 for stochastic policies.
- Training increased harvest returns but did not produce reliably safe policies across starts and action noise.
- Individual policies optimize agent 0 only; their fixed partners retain their original requests. The cooperative policy optimizes total population harvest, not fairness.
- This is a first fixed-budget PPO baseline. No optimality, equilibrium convergence, or statistical significance is claimed.
- The final policies were used even where late training was unstable; there was no best-on-evaluation checkpoint selection.
- The training episode limit was 256 rounds; evaluation extends to 1,000 rounds and includes initial energies outside the training range.
- No reward penalty or shield was used during training. Unshielded execution still rations harvest to available stock.
- Corrected rounds in unshielded mode mean physical rationing. In shielded mode they include safety corrections, both upward and downward.
- Harvest/round uses the full requested evaluation horizon; rounds after failure contribute zero.
- Finite rollouts are empirical evidence. The analytic safety argument depends on the invariant-core assumptions and the numerical guard documented in README.md.

The nine training runs used 1,806,336 transitions and 172.8 seconds of summed training wall time on this Mac (excluding setup, tests, and evaluation).

## Reproduce

See TRAINING.md. Run training first, then evaluate_trained, then:

```sh
.venv/bin/python -m examples.non_spatial_commons.summarize_training
.venv/bin/python -m examples.non_spatial_commons.plot_training --input examples/results/non_spatial_commons/ppo
```

Each seed folder retains initial.pt, final.pt, metadata.json, training.csv, and the full evaluation traces.
The showcase plots use training seed 0, deterministic actions, and one specified starting state; they are not averages over seeds.
