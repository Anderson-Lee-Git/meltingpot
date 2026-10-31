# Longer Cooperative Training Episodes

Three fresh centralized PPO runs (seeds 0, 1, 2), each trained entirely unshielded.
The episode limit changed from 256 to 1,000 rounds. Each run retained the original
200,704-transition budget, reward, initial-state distribution, and PPO hyperparameters.
Final checkpoints were used; no held-out evaluation informed checkpoint selection.

Training used 602,112 transitions and 50.4 seconds of summed training time.

Evaluation: 1,000 rounds; 12 identical starts per seed; deterministic and stochastic
actions; with and without shielding. All training completed before evaluation.
Checkpoint hashes were unchanged by evaluation. Old baseline artifacts were preserved.

| Training Episode Limit | Actions | Shield | Survival | Population Harvest / Round |
| --- | --- | --- | --- | --- |
| 256 | deterministic | Off | 24/36 | 7.955 |
| 256 | deterministic | On | 36/36 | 11.893 |
| 256 | stochastic | Off | 4/36 | 4.238 |
| 256 | stochastic | On | 36/36 | 12.204 |
| 1000 | deterministic | Off | 24/36 | 7.916 |
| 1000 | deterministic | On | 36/36 | 11.834 |
| 1000 | stochastic | Off | 1/36 | 3.289 |
| 1000 | stochastic | On | 36/36 | 12.188 |

Harvest / round divides total harvest by the full 1,000-round evaluation horizon;
rounds after failure count as zero. Each cell has 36 episodes, representing three
trained policies on 12 starts; these are not 36 independent training runs.

## Training Diagnostics

| Seed | Final Recent Failure Fraction | Final Recent Episode Length |
| --- | --- | --- |
| seed_0 | 75% | 422.8 |
| seed_1 | 88% | 341.6 |
| seed_2 | 95% | 261.0 |

Training diagnostics summarize the recent 100 completed episodes, which span
different numbers of transitions across horizons. Raw training episode returns
cannot be directly compared across 256- and 1,000-round limits.

Longer episodes alone did not eliminate late-training instability. This experiment
changes the episode limit at a fixed transition budget; it does not test a larger
training budget, different discount, or validation-based checkpoint selection.

Reproduction commands are in examples/non_spatial_commons/TRAINING.md.
