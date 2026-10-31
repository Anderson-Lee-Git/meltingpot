# Figure build and layout troubleshooting

## Gotchas (learned by looking at the output)

- **pgfplots `symbolic x coords` silently ignores `enlarge x limits`.** With
  few groups the first group crowds against the y-axis and the leftmost bar
  overlaps the tick labels — no amount of `enlarge x limits` fixes it. Use
  **numeric x positions + explicit `xmin`/`xmax`** (`xtick`/`xticklabels`
  for the category names) for deterministic margins. See the example `.tex`.
- **Grouped bars clip at the axis boundary.** Outer bars that extend past
  the outermost coordinate get cut off (only their value labels remain).
  Set `clip=false` *and* leave margin via `xmin`/`xmax`.
- **Use `axis lines*=left`** (starred) for the L-shaped spine — the
  non-starred `axis y line=left` can place the y-axis at x=0 instead of the
  plot boundary, putting it through your bars.
- **Carlito's digit `0` reads like `O`** at small sizes in the raster
  preview — that's the font, not a rendering bug.
- The matplotlib `<name>.png` and the LaTeX `<name>.pdf-preview.png` are
  different files. The `.png` is the matplotlib deliverable; the
  `-preview.png` is only a raster of the PDF for your eyes.

## Troubleshooting

- `WARNING: Carlito not found` from a Python run → the TeX Live Carlito TTFs
  weren't located. Check `kpsewhich Carlito-Regular.ttf`; install the
  `carlito` TeX package if missing.
- `pdflatex: ... carlito.sty not found` → install the `carlito` package
  (`tlmgr install carlito` on TeX Live).
- `build_pdf.sh: need one of pdftoppm, pdftocairo, or gs` → install a
  rasterizer: `brew install poppler` (macOS) / `apt-get install
  poppler-utils` (Linux with root). No root? `gs` is usually already
  present; otherwise `conda install -c conda-forge poppler`.
- LaTeX `! Package pgfplots Error: ... compat` → bump `compat=` in
  `main.tex` to your pgfplots version (or lower it).
- `pdfTeX error: ... Font Crlt-... not found` (tries `mktexpk`/`gsftopk`) →
  the active `pdftex.map` doesn't know your Carlito. Register it:
  `updmap-user --enable Map=carlito.map` (user) or `--sys` (root). If
  `updmap` aborts on an unrelated missing map (e.g. a stale `morisawa5.map`
  in a system config you can't edit), clear it from your user view first:
  `updmap-user --disable <that>.map`, then re-run.

### Cluster setup (no root, no internet on compute nodes)

On a shared cluster you typically can't `tlmgr install` (it's often
distro-patched to refuse, and the system tree is read-only) and compute
nodes have no internet. Install the missing TeX packages into your **user
tree** (`TEXMFHOME`, usually `~/texmf`) from a **login node**, once:

1. Grab the runtime files. CTAN serves TDS-ready zips for some packages
   (`https://mirrors.ctan.org/install/<path>.tds.zip`); for the rest pull
   the per-package tarball from the matching TeX Live historic tlnet
   archive (`.../historic/systems/texlive/<year>/tlnet-final/archive/<pkg>.tar.xz`).
   `paper-figure` needs `standalone`, `carlito`, and `fontaxes`.
2. Extract the TDS dirs (`tex/`, `fonts/`, `doc/`) into `~/texmf`, then
   refresh the filename DB: `mktexlsr ~/texmf`.
3. Carlito embeds Type1 fonts via a map — register it once:
   `updmap-user --enable Map=carlito.map` (see the `Crlt-... not found`
   note above if `updmap` balks).
4. Verify: `kpsewhich carlito.sty standalone.cls fontaxes.sty
   Carlito-Regular.ttf`. Match the package versions to your TeX Live year
   so font/`.tfm`/map names line up.
