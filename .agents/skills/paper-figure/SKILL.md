---
name: paper-figure
description: Create or revise paper-ready research plots and diagrams with a matplotlib PNG and an independently compiled LaTeX/TikZ vector PDF, using Carlito, readable labels, and a color-blind-safe palette. Use for academic figures and publication exports.
---

# Paper Figure

Produce matching figures from two editable sources:

- `<name>.py` renders `<name>.png` with matplotlib.
- `<name>.tex` contains one `tikzpicture` and compiles to `<name>.pdf` with the
  shared wrapper. A matplotlib PDF export is not the independent TikZ source.

Use both formats by default; honor an explicit request for a narrower output.
Inspect the matplotlib PNG and a rasterized preview of the compiled PDF. Fix
clipping, collisions, font substitutions, or inconsistent data, then rebuild and
inspect again. Successful compilation alone does not establish visual quality.

## Repository Environment

Run commands from the Melting Pot repository root. Use the isolated `.venv`
managed by `uv`. Avoid a plain `uv run` here: the root `pyproject.toml` belongs to
Melting Pot and can trigger installation of the full Lab2D/TensorFlow stack.

```sh
# Verify the existing plotting environment.
uv run --no-project --python .venv/bin/python python -c 'import matplotlib'
# Only if matplotlib is missing:
uv pip install --python .venv/bin/python matplotlib

# Check the font, TeX wrapper dependencies, and PDF rasterizer.
kpsewhich carlito.sty standalone.cls fontaxes.sty Carlito-Regular.ttf
command -v pdflatex
command -v pdftoppm || command -v pdftocairo || command -v gs
```

The shared Python style registers Carlito from TeX Live; no system font install
is necessary when those files are present. A fallback-font warning means the
Carlito style requirement has not passed. Resolve missing dependencies within
the task's permissions; report any remaining build limitation.

For a writable matplotlib cache in sandboxed runs, set `MPLCONFIGDIR` to an
existing writable temporary or task-local directory. Do not hard-code a user's
home or this machine's absolute paths into generated figure sources.

## Figure Layout and Build

Create `figures/<sortable-task-slug>/` for each figure task. Reuse its directory
when revising that figure; do not overwrite an unrelated task's figures.

```text
figures/<slug>/
  <name>.py
  <name>.tex
  <name>.png
  <name>.pdf
  <name>.pdf-preview.png
  <name>.log
```

Put the preamble only in the shared `scaffold/main.tex`. The figure's `.tex`
contains `\begin{tikzpicture} ... \end{tikzpicture}`, not a standalone document.
Additional local data tables are appropriate when both sources use the same
measurements. Preserve the data, aggregation, uncertainty definition, and
experimental labels when restyling existing figures. Restyling does not require
rerunning training or changing experiment settings.

Import the shared style at the top of a figure script located two levels below
the repository root:

```python
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / '.agents/skills/paper-figure/scaffold'))
from paper_style import apply_style, PALETTE

apply_style()
# Build the figure and save it next to this script:
# fig.savefig(Path(__file__).with_suffix('.png'))
```

Build both sources:

```sh
uv run --no-project --python .venv/bin/python python figures/<slug>/<name>.py
.agents/skills/paper-figure/scaffold/build_pdf.sh figures/<slug>/<name>.tex
```

The build helper uses `pdflatex` without shell escape and renders a preview via
Poppler or Ghostscript. It retains the log for diagnostics. View the two PNGs
with the available image/file tools, such as `view_image`, or open them in the
app. Inspect the PDF through its rendered preview; do not substitute the
matplotlib PNG for that inspection.

## Visual Checklist

- **Font:** Carlito throughout, including legends; no fallback warning.
- **Size:** readable at the intended paper-column width; titles at least as
  large as axis labels, labels at least as large as ticks.
- **Layout:** no clipped bars/markers or text collisions; legends do not cover
  the data; sufficient margins and clean axes with light grids.
- **Color:** use the shared Okabe-Ito `PALETTE` in Python and `c0`-`c5` in TeX.
  Match each series color across formats and related panels.
- **Casing:** Title Case for ordinary titles, labels, and legend text, while
  preserving canonical method names and acronyms such as PPO and MSY. See
  [references/casing.md](references/casing.md) when naming datasets or models.
- **Parity:** identical underlying values, labels, series, units, and statistical
  definitions in both formats, with consistent proportions and colors.

In this repo, fixed-policy and PPO traces are under
`examples/results/non_spatial_commons/`. Distinguish deterministic from
stochastic evaluation, individual from population returns, and training-seed
ranges from confidence intervals. Label a single-episode trajectory as such.

## Shared Resources

- `scaffold/paper_style.py`: `apply_style()` and the six-color `PALETTE`.
- `scaffold/main.tex`: Carlito/sans-math wrapper, TikZ/pgfplots libraries, palette.
- `scaffold/build_pdf.sh`: compile a picture and rasterize the resulting PDF.
- `assets/example-method-comparison/`: upstream example `.py` and `.tex`. Copy
  both into a new `figures/<slug>/` directory before running. Its scores are
  illustrative; replace them with actual inputs for research figures.

Read [references/troubleshooting.md](references/troubleshooting.md) for grouped
bar spacing, pgfplots axis placement, font-map issues, missing rasterizers, or
cluster TeX setup. Do not load those environment-specific procedures for every
ordinary plotting task. Shared scaffolding is common infrastructure; make
per-figure changes in the figure sources.

## Upstream Provenance

Ported from Anderson-Lee-Git/research-template's
[paper-figure skill](https://github.com/Anderson-Lee-Git/research-template/tree/2f4a04c8d1a98d027e1daf2d13c64cfba99d41cb/.claude/skills/paper-figure)
at commit `2f4a04c8d1a98d027e1daf2d13c64cfba99d41cb`.
Codex adaptations: repository-scoped `.agents/skills` paths, Codex image-viewing
instructions, isolated `uv` execution, bundled example sources, and progressive
references. The PDF helper explicitly selects its shared wrapper, disables shell
escape, and retains logs for inspection.
