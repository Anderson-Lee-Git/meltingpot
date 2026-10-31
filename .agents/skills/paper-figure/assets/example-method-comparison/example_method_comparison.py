"""Example paper figure (matplotlib -> PNG source).

Grouped bar chart comparing three methods across two benchmarks. This is
the reference for the matplotlib half of the paper-figure workflow.

Run from the repo root:
    uv run --no-project --python .venv/bin/python python \
        figures/example-method-comparison/example_method_comparison.py
"""
import sys
from pathlib import Path

# Make the shared style importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]
                       / ".agents/skills/paper-figure/scaffold"))
from paper_style import apply_style, PALETTE  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

apply_style()

methods = ["Baseline", "Ours (small)", "Ours (large)"]
benchmarks = ["GSM8K", "MATH"]
# Illustrative upstream example values, not Melting Pot measurements.
scores = np.array([
    [62.1, 34.5],   # Baseline
    [71.8, 41.2],   # Ours (small)
    [78.3, 48.9],   # Ours (large)
])

x = np.arange(len(benchmarks))
width = 0.25

fig, ax = plt.subplots(figsize=(6.0, 3.6))
for i, method in enumerate(methods):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, scores[i], width, label=method,
                  color=PALETTE[i], edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=10)

ax.set_ylabel("Accuracy (%)")
ax.set_title("Method Comparison Across Reasoning Benchmarks")
ax.set_xticks(x)
ax.set_xticklabels(benchmarks)
ax.set_ylim(0, 90)
ax.legend(loc="upper right", ncol=1)
ax.grid(axis="x", visible=False)

out = Path(__file__).with_suffix(".png")
fig.savefig(out)
print(f"Wrote {out}")
