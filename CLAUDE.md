# Manuel d'apprentissage machine (IFT3395/IFT6390)

## Commandes

Servir le livre avec rechargement automatique (développement):
```bash
uv run jupyter-book start --execute --port 3000
```

Compiler le site (sans serveur):
```bash
uv run jupyter-book build --site
```

Compiler avec exécution des notebooks:
```bash
uv run jupyter-book build --site --execute
```

## Lignes directrices

Lors de la rédaction ou révision de contenu, appliquer:
- **WRITING_GUIDELINES.md** — ton, style, langue française, typographie
- **PEDAGOGICAL_GUIDELINES.md** — structure pédagogique, gestion de l'anxiété, exercices

Les skills `/writing` et `/pedagogical` contiennent des versions condensées de ces guides.

## Structure du projet

- `ch*.md` — chapitres du livre (MyST Markdown)
- `_toc.yml` — table des matières
- `_config.yml` — configuration Jupyter Book
- `references.bib` — bibliographie BibTeX
