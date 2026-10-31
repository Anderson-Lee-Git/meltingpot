"""Shared matplotlib style for paper-ready figures.

Import and call ``apply_style()`` at the top of a figure script, then use
the Okabe-Ito ``PALETTE`` for series colors. The goal is parity with the
LaTeX/TikZ output: Carlito font, large fonts, color-blind-safe palette.

    from paper_style import apply_style, PALETTE
    apply_style()

Carlito is registered from the local TeX Live install (Google's Carlito
TTFs ship with the `carlito` package), so no system font install is
needed. Falls back to DejaVu Sans with a warning if it can't be found.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# Okabe-Ito color-blind-safe palette. Matches c0..c5 in scaffold/main.tex.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def _register_carlito() -> str:
    """Register Carlito TTFs with matplotlib; return the family name to use."""
    # 1) Already known to fontconfig / matplotlib?
    for name in fm.fontManager.ttflist:
        if name.name == "Carlito":
            return "Carlito"
    # 2) Pull the TTFs out of the TeX Live tree (carlito package).
    candidates: list[Path] = []
    try:
        out = subprocess.run(
            ["kpsewhich", "Carlito-Regular.ttf"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if out:
            candidates.append(Path(out).parent)
    except Exception:
        pass
    # Common TeX tree locations as a fallback if kpsewhich is unavailable:
    # macOS / manual TeX Live, Linux distro system trees, and the per-user
    # texmf tree (TEXMFHOME) where a no-root cluster install lands.
    carlito_rel = "fonts/truetype/google/carlito"
    candidates += list(Path("/usr/local/texlive").glob(f"*/texmf-dist/{carlito_rel}"))
    candidates += [
        Path("/usr/share/texlive/texmf-dist") / carlito_rel,
        Path("/usr/share/texmf-dist") / carlito_rel,
        Path("/usr/share/texmf") / carlito_rel,
        Path.home() / "texmf" / carlito_rel,
    ]
    for d in candidates:
        ttfs = list(d.glob("Carlito-*.ttf"))
        if ttfs:
            for ttf in ttfs:
                fm.fontManager.addfont(str(ttf))
            return "Carlito"
    print("WARNING: Carlito not found; falling back to DejaVu Sans.",
          file=sys.stderr)
    return "DejaVu Sans"


def apply_style() -> None:
    family = _register_carlito()
    matplotlib.rcParams.update({
        "font.family": family,
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 16,
        # Clean, paper-ready axes.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Embed real fonts (TrueType) in PDFs, not Type-3 bitmaps.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })
