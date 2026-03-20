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

Publier sur GitHub Pages:
```bash
source publish.sh
```

## Structure du projet

- `ch*.md` — chapitres du livre (MyST Markdown)
- `_toc.yml` — table des matières
- `_config.yml` — configuration Jupyter Book
- `references.bib` — bibliographie BibTeX

# Style d'écriture

## Contexte

Rédaction scientifique en français : manuel d'apprentissage machine pour IFT3395/IFT6390, Université de Montréal. L'objectif premier est la clarté.

## Principes

1. Clarté et facilité de compréhension
2. Enchaînement logique des idées : chaque phrase découle de la précédente et prépare la suivante
3. Explication progressive (du simple vers le complexe)
4. Ton honnête et direct, sans figures de style grandiloquentes

## Éviter

- Les anglicismes (voir tableau dans WRITING_GUIDELINES.md)
- Le tiret cadratin (em-dash)
- Les constructions rhétoriques de LLM : « non seulement X, mais Y », phrases courtes isolées pour effet dramatique, ton LinkedIn
- Les phrases creuses : « Il est important de noter que... », « fondamentalement », « essentiellement »
- Le gras décoratif dans les paragraphes

## Références

- `WRITING_GUIDELINES.md` pour le ton, le style et les conventions typographiques
- `PEDAGOGICAL_GUIDELINES.md` pour la structure du matériel et les principes pédagogiques