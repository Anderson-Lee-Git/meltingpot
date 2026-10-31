# Fixed-policy baseline results

Validated with Python 3.13, 57 passing tests, and paired 1,000-round rollouts.
Each cell reports three episodes (seeds 0, 1, 2); deterministic policies repeat the same trajectory.

The default state is R=62.5 and E_i=5. The recovery state is R=13.3 and E_i=0.
Parameters: N=4, K=100, r=0.5, delta=1, E_max=10; MSY=12.5.

## Default

| Population | Unshielded survival | Shielded survival | Unshielded harvest/round | Shielded harvest/round | Shielded corrected rounds |
| --- | --- | --- | --- | --- | --- |
| maintenance | 3/3 | 3/3 | 4.000 | 4.000 | 0.0% |
| msy_share | 3/3 | 3/3 | 12.500 | 12.500 | 0.0% |
| greedy | 0/3 | 3/3 | 0.062 | 12.500 | 100.0% |
| reserve_target | 3/3 | 3/3 | 4.000 | 4.000 | 0.0% |
| random | 0/3 | 3/3 | 0.062 | 12.500 | 100.0% |
| abstain | 0/3 | 3/3 | 0.000 | 3.980 | 99.5% |
| mixed | 0/3 | 3/3 | 0.062 | 12.500 | 100.0% |

Analytic S_crit: 8.76894374382339. Numerically protected threshold: 8.7689437438256.

## Recovery

| Population | Unshielded survival | Shielded survival | Unshielded harvest/round | Shielded harvest/round | Shielded corrected rounds |
| --- | --- | --- | --- | --- | --- |
| maintenance | 3/3 | 3/3 | 4.000 | 4.000 | 0.0% |
| msy_share | 0/3 | 3/3 | 0.014 | 12.373 | 1.5% |
| greedy | 0/3 | 3/3 | 0.013 | 12.373 | 100.0% |
| reserve_target | 0/3 | 3/3 | 0.013 | 4.020 | 1.6% |
| random | 0/3 | 3/3 | 0.013 | 12.373 | 100.0% |
| abstain | 0/3 | 3/3 | 0.000 | 4.000 | 100.0% |
| mixed | 0/3 | 3/3 | 0.013 | 12.373 | 100.0% |

Analytic S_crit: 8.76894374382339. Numerically protected threshold: 8.7689437438256.

## Interpretation

- All 42 shielded episodes survived to the 1,000-round time limit.
- Maintenance and MSY-share policies need no correction at the default operating point.
- Greedy, random, abstaining, and mixed populations fail unshielded in both initializations.
- A fixed MSY-share request is only sustainable at suitable resource stocks: it fails from the depleted initialization unless the filter temporarily reduces harvest.
- Shielded maintenance and abstention survive below MSY; safety alone does not force maximum yield.
- Reserve-target policies also need intervention during depleted-stock recovery.

Harvest/round uses the full requested horizon; after starvation the rest of that horizon contributes zero.
Correction statistics include both downward changes and forced upward harvest. Unshielded trajectories still ration requests to the physical stock budget.
The mixed population contains greedy, maintenance, MSY-share, and reserve-target agents in that order.

Reproduce these runs with the commands in README.md using `--episodes 3`.
Plots show seed 0 only. Raw CSV traces and JSON summaries are generated under `examples/results/non_spatial_commons/{default,recovery}/`.
