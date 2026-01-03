#!/bin/bash
# Build with BASE_URL=/mlbook for subdirectory deployment
BASE_URL=/mlbook uv run jupyter-book build --html --execute
# Deploy to GitHub Pages
ghp-import -n -p -f _build/html

