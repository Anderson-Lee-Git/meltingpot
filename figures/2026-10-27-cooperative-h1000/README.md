# Cooperative PPO Trained With 1,000-Round Episodes

Four matched shielded/unshielded trajectory figures use the completed frozen-policy
evaluation of the three new cooperative checkpoints. Training used a maximum of
1,000 rounds per episode, 200,704 transitions per seed, and no shielding.
The original evaluation ran all 144 cases after all training completed:
3 seeds × 12 starts × 2 action-selection modes × 2 shielding modes.
The figure build verifies that checkpoint hashes remain unchanged.

| Figure | Actions | Initial State |
| --- | --- | --- |
| `deterministic_default` | Sigmoid of Gaussian latent mean | Resource 62.5; each energy 5 |
| `deterministic_recovery` | Sigmoid of Gaussian latent mean | Resource 13.3; each energy 0 |
| `stochastic_default` | Sampled Gaussian latent actions | Resource 62.5; each energy 5 |
| `stochastic_recovery` | Sampled Gaussian latent actions | Resource 13.3; each energy 0 |

Each column is one training seed, not a different learning method. Each trajectory
is one episode; these figures are not averages. Rows show resource stock, minimum
energy across four agents, aggregate executed harvest, and cumulative population
return. Cumulative return is the undiscounted running sum of all four agents'
executed harvests, starting at zero. After failure, it stays constant through the
remaining horizon because no further rewards accrue; the cross remains at the
actual failure round. Orange dashed lines
are unshielded and blue solid lines are shielded. Coincident curves appear as
orange dashes over blue. Crosses mark failure; trajectories stop at termination.
Dotted references are MSY stock (62.5), zero energy, and MSY harvest (12.5).

All figures show the full 1,000-round horizon, matching the earlier PPO plots;
early failures from depleted starts are consequently near the left edge.

| Start | Actions | Unshielded Failure Rounds (Seeds 0, 1, 2) | Shielded Survival |
| --- | --- | --- | --- |
| Default | Deterministic | None; all survive | 3/3 |
| Depleted | Deterministic | 7, 7, 6 | 3/3 |
| Default | Stochastic | 635, 309, 637 | 3/3 |
| Depleted | Stochastic | 4, 1, 4 | 3/3 |

Across all 12 starts per seed, unshielded survival is 24/36 deterministic and
1/36 stochastic. Shielded survival is 36/36 in each mode. These single-start
figures illustrate behavior; the full evaluation provides the aggregate result.

## Reproduce

From the repository root:

```sh
uv run --no-project --python .venv/bin/python python figures/2026-10-27-cooperative-h1000/prepare.py
.agents/skills/paper-figure/scaffold/build_pdf.sh figures/2026-10-27-cooperative-h1000/deterministic_default.tex
```

Run each figure's `.py` to render only that figure; repeat the PDF command for
its `.tex` file. `prepare.py` selects saved evaluation traces without rerunning
training or evaluation. `render_common.py` provides the same renderer used for
the earlier comparisons. Each `.tex` independently plots numeric tables in
`data/`; `.provenance.json` files record original and selected-data hashes.
Both the matplotlib PNGs and rendered TikZ PDF previews were visually checked.

In the exported bundle, `bash scaffold/build_pdf.sh deterministic_default.tex`
recompiles a PDF. Python rebuilding uses the original repository data paths.
