# Non-Spatial Commons Figures

Recreated with the repository's paper-figure skill from saved experimental data.
No training or evaluation was rerun. Each figure has a matplotlib PNG and an
independent pgfplots/TikZ vector PDF, with Carlito fonts and Okabe–Ito colors.

| Figure | Selection |
| --- | --- |
| `fixed_default` | Greedy, abstain, mixed; episode/seed 0; first 100 rounds |
| `fixed_recovery` | MSY share, reserve target, mixed; episode/seed 0; first 100 rounds |
| `ppo_training` | Three seeds per experiment; mean and min–max across seeds |
| `ppo_default` | Frozen PPO; training seed 0; deterministic actions; 1,000 rounds |
| `ppo_recovery` | Same frozen policies; depleted initial state; 1,000 rounds |

Training bands are seed ranges, not confidence intervals. Training metrics are
recent-100-completed-episode statistics; observations before any completed
episode are excluded. Individual return is agent 0's harvest; cooperative return
is total population harvest. Their top-row scales differ accordingly.

Trajectory rows are resource stock, minimum energy across the four agents, and
aggregate executed harvest. Initial states are prepended only to stock and
energy. Orange dashed curves are unshielded; blue solid curves are shielded.
Identical traces appear as orange dashes over a blue line. Crosses mark terminal
failure. Dotted references are MSY stock, zero energy, and MSY harvest, by row.

## Rebuild

Run from the repository root, with the existing uv-managed `.venv`:

```sh
uv run --no-project --python .venv/bin/python python figures/2026-10-27-non-spatial-commons/render.py
# Or run one figure's .py entry point.
.agents/skills/paper-figure/scaffold/build_pdf.sh figures/2026-10-27-non-spatial-commons/ppo_training.tex
```

The Python renderer reads the original CSV/JSON files under
`examples/results/non_spatial_commons`, writes numeric tables under `data/`, and
renders PNGs. Each `.tex` independently plots those tables and can be edited and
recompiled without Python. Each `.provenance.json` records input SHA-256 hashes.
Shared rendering code is in `render.py`; shared font/style and the LaTeX wrapper
are in `.agents/skills/paper-figure/scaffold`.

Both formats were visually inspected. All five PDFs compile without shell
escape; source hashes are checked against the saved results before delivery.
