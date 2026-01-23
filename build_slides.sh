#!/bin/bash
# Build Marp slides from markdown files in slides/

set -e

echo "Building Marp slides with mila theme..."

# Build HTML slides with custom theme
npx @marp-team/marp-cli slides/linear_models.md \
    --theme slides/mila.css \
    --html \
    --allow-local-files \
    -o _build/slides/linear_models.html

# Copy images
mkdir -p _build/slides/_static
cp _static/*.gif _build/slides/_static/ 2>/dev/null || true
cp _static/*.svg _build/slides/_static/ 2>/dev/null || true

echo ""
echo "✓ Slides built to _build/slides/linear_models.html"
echo ""
echo "To view:"
echo "  open _build/slides/linear_models.html"
echo ""
echo "For PDF export:"
echo "  npx @marp-team/marp-cli slides/linear_models.md --theme slides/mila.css --pdf -o _build/slides/linear_models.pdf"
