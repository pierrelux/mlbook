#!/bin/bash
# Build Marp slides from markdown files in slides/

set -e

echo "Building Marp slides with mila theme..."

# Create output directory
mkdir -p _build/slides/_static

# Build HTML slides with custom theme
npx @marp-team/marp-cli --no-stdin slides/linear_models.md \
    --theme slides/mila.css \
    --html \
    --allow-local-files \
    -o _build/slides/linear_models.html

npx @marp-team/marp-cli --no-stdin slides/probabilistic_models.md \
    --theme slides/mila.css \
    --html \
    --allow-local-files \
    -o _build/slides/probabilistic_models.html

npx @marp-team/marp-cli --no-stdin slides/projet_energie.md \
    --theme slides/mila.css \
    --html \
    --allow-local-files \
    -o _build/slides/projet_energie.html

npx @marp-team/marp-cli --no-stdin slides/bradley_terry.md \
    --theme slides/mila.css \
    --html \
    --allow-local-files \
    -o _build/slides/bradley_terry.html

npx @marp-team/marp-cli --no-stdin slides/latent_variables.md \
    --theme slides/mila.css \
    --html \
    --allow-local-files \
    -o _build/slides/latent_variables.html

npx @marp-team/marp-cli --no-stdin slides/nonparametric_methods.md \
    --theme slides/mila.css \
    --html \
    --allow-local-files \
    -o _build/slides/nonparametric_methods.html

# Copy images
cp _static/*.gif _build/slides/_static/ 2>/dev/null || true
cp _static/*.svg _build/slides/_static/ 2>/dev/null || true
cp _static/*.png _build/slides/_static/ 2>/dev/null || true

echo ""
echo "✓ Slides built to _build/slides/"
echo ""
echo "To view:"
echo "  open _build/slides/linear_models.html"
echo "  open _build/slides/probabilistic_models.html"
echo "  open _build/slides/projet_energie.html"
echo "  open _build/slides/bradley_terry.html"
echo "  open _build/slides/latent_variables.html"
echo "  open _build/slides/nonparametric_methods.html"
echo ""
echo "For PDF export:"
echo "  npx @marp-team/marp-cli --no-stdin slides/FILE.md --theme slides/mila.css --pdf -o _build/slides/FILE.pdf"
