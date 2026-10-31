#!/usr/bin/env bash
# Compile a single-tikzpicture figure into a cropped, paper-ready PDF,
# then rasterize it to a PNG so the result can be visually inspected.
#
#   build_pdf.sh <figure_dir>/<figure_name>.tex
#
# Produces, next to the .tex:
#   <figure_name>.pdf           the deliverable (vector, LaTeX/TikZ source)
#   <figure_name>.pdf-preview.png   raster preview for `look at it` checks
#
# The <figure_name>.tex must contain ONLY a single \begin{tikzpicture}
# ... \end{tikzpicture}. The preamble lives in scaffold/main.tex.
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: build_pdf.sh <path/to/figure_name.tex>" >&2
  exit 2
fi

TEX_PATH="$1"
DIR="$(cd "$(dirname "$TEX_PATH")" && pwd)"
NAME="$(basename "$TEX_PATH" .tex)"
SCAFFOLD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$DIR"
# TEXINPUTS lets pdflatex find main.tex (in scaffold) and the figure (here).
TEXINPUTS="$SCAFFOLD:$DIR:" pdflatex \
  -no-shell-escape -interaction=nonstopmode -halt-on-error \
  -jobname="$NAME" "\def\figfile{$NAME.tex}\input{$SCAFFOLD/main.tex}"

# Keep the log for font/overfull-box checks; remove only the auxiliary file.
rm -f "$NAME.aux"

# Rasterize for visual inspection (single page -> <name>.pdf-preview.png).
# pdftoppm/pdftocairo (poppler) give the cleanest output but are often
# absent on clusters; fall back to Ghostscript, which is near-universal.
PREVIEW="$NAME.pdf-preview.png"
if command -v pdftoppm >/dev/null 2>&1; then
  pdftoppm -png -r 200 -singlefile "$NAME.pdf" "$NAME.pdf-preview"
elif command -v pdftocairo >/dev/null 2>&1; then
  pdftocairo -png -r 200 -singlefile "$NAME.pdf" "$NAME.pdf-preview"
elif command -v gs >/dev/null 2>&1; then
  gs -q -dSAFER -dBATCH -dNOPAUSE -dUseCropBox -r200 \
     -sDEVICE=pngalpha -sOutputFile="$PREVIEW" "$NAME.pdf"
else
  echo "build_pdf.sh: need one of pdftoppm, pdftocairo, or gs to rasterize the PDF." >&2
  echo "  Linux (no root): 'conda install -c conda-forge poppler', or load a ghostscript/poppler module." >&2
  echo "  macOS: 'brew install poppler'." >&2
  exit 3
fi

echo "Built: $DIR/$NAME.pdf"
echo "Preview: $DIR/$NAME.pdf-preview.png"
