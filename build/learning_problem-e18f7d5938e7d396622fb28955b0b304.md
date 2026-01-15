---
kernelspec:
  name: python3
  display_name: Python 3
---

# Le problème d'apprentissage

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Définir formellement le problème d'apprentissage supervisé
- Distinguer les tâches de classification et de régression
- Définir le risque et le risque empirique
- Expliquer le principe de minimisation du risque empirique
- Dériver l'estimateur du maximum de vraisemblance
- Relier le maximum de vraisemblance à la divergence de Kullback-Leibler
- Identifier les sources d'écart entre performance mesurée et performance réelle
```

Dans le chapitre précédent, nous avons vu les k plus proches voisins, une méthode intuitive qui prédit en consultant directement les données d'entraînement. Cette approche est simple à comprendre et à implémenter, mais elle a un coût: les données doivent être conservées en mémoire, et chaque prédiction requiert de parcourir l'ensemble d'entraînement. Ce chapitre développe une approche différente: plutôt que de garder les données, nous cherchons à les *résumer* dans un ensemble de **paramètres**. L'apprentissage devient alors un problème d'**optimisation**.

## Apprentissage supervisé

Une ingénieure automobile mesure la distance de freinage d'un véhicule à différentes vitesses. Ses données ressemblent à ceci:

| Vitesse (mph) | Distance (ft) |
|---------------|---------------|
| 4 | 2 |
| 7 | 4 |
| 12 | 20 |
| 18 | 56 |
| 24 | 93 |

Elle veut prédire la distance de freinage à 30 mph sans faire l'essai. Pour cela, elle cherche une fonction $f$ telle que $f(\text{vitesse}) \approx \text{distance}$ sur ses observations. Si la fonction capture la relation sous-jacente, elle devrait donner une prédiction raisonnable pour des vitesses non mesurées.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Données de freinage (Ezekiel, 1930): vitesse (mph) vs distance d'arrêt (ft)
speed = np.array([4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14,
                  14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19,
                  20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25], dtype=float)
dist = np.array([2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46,
                 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46,
                 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85], dtype=float)

plt.figure(figsize=(6, 4))
plt.scatter(speed, dist, alpha=0.7, label='Observations')

# Fit quadratic
coeffs = np.polyfit(speed, dist, 2)
speed_grid = np.linspace(0, 30, 100)
dist_pred = np.polyval(coeffs, speed_grid)
plt.plot(speed_grid, dist_pred, 'k--', alpha=0.6, label='Fonction ajustée')

# Prediction at 30 mph
pred_30 = np.polyval(coeffs, 30)
plt.scatter([30], [pred_30], marker='x', s=80, color='C1', zorder=5, label=f'Prédiction à 30 mph: {pred_30:.0f} ft')

plt.xlabel('Vitesse (mph)')
plt.ylabel('Distance de freinage (ft)')
plt.legend()
plt.tight_layout()
```

Ce processus est l'ajustement de courbe (*curve fitting*). Nous avons des paires (entrée, sortie), nous ajustons une fonction, et nous utilisons cette fonction pour prédire. L'apprentissage supervisé généralise cette idée: les entrées peuvent être des vecteurs de dimension quelconque, les sorties peuvent être continues ou discrètes, et les fonctions candidates peuvent être bien plus complexes qu'un polynôme.

Formellement, nous disposons d'un jeu de données $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ composé de $N$ paires, où chaque $\mathbf{x}_i \in \mathcal{X}$ est une entrée et $y_i \in \mathcal{Y}$ est la sortie correspondante. L'objectif est de trouver une fonction $f: \mathcal{X} \to \mathcal{Y}$ qui approxime bien la relation entre entrées et sorties, y compris pour des exemples que nous n'avons pas encore observés.

Dans de nombreuses applications, les entrées sont des vecteurs de caractéristiques. Chaque exemple $\mathbf{x}_i \in \mathbb{R}^d$ est un vecteur de dimension $d$, où chaque composante représente une mesure ou un attribut. Pour prédire le prix d'une maison, les entrées pourraient être la superficie, le nombre de chambres et l'âge du bâtiment. Pour filtrer les pourriels, les entrées pourraient être des fréquences de mots. Pour diagnostiquer une maladie, les entrées pourraient être des résultats d'analyses sanguines.

Lorsque la sortie est une valeur continue, nous parlons de **régression**: $f: \mathbb{R}^d \to \mathbb{R}$ pour une sortie scalaire, ou $f: \mathbb{R}^d \to \mathbb{R}^p$ pour une sortie vectorielle. La distance de freinage, le prix d'une maison, la concentration d'un médicament dans le sang sont des exemples de régression.

Lorsque la sortie est une catégorie parmi un ensemble fini, nous parlons de **classification**. Pour la classification binaire, $f: \mathbb{R}^d \to \{0, 1\}$. Pour la classification multiclasse avec $m$ catégories, $f: \mathbb{R}^d \to \{0, \ldots, m-1\}$. Déterminer si un courriel est un pourriel, diagnostiquer une maladie, ou reconnaître un chiffre manuscrit sont des exemples de classification.

## Mesurer l'erreur

Pour choisir entre deux fonctions candidates, nous avons besoin d'un critère qui quantifie la qualité des prédictions. Une **fonction de perte** $\ell: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_+$ mesure l'écart entre une prédiction $\hat{y}$ et la vraie valeur $y$. Une perte de zéro indique une prédiction parfaite; plus la perte est grande, plus l'erreur est importante.

Pour la régression, nous utilisons généralement la **perte quadratique**:

$$
\ell_2(y, \hat{y}) = (y - \hat{y})^2
$$

Cette perte pénalise les grandes erreurs de manière quadratique. Une erreur de 2 coûte quatre fois plus qu'une erreur de 1.

Reprenons les données de freinage. Supposons que notre fonction prédise 50 ft pour une vitesse où la vraie distance est 56 ft. La perte quadratique est $(56 - 50)^2 = 36$. Si elle prédit 70 ft, la perte est $(56 - 70)^2 = 196$. La perte quadratique pénalise sévèrement les grandes erreurs.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Données de freinage
speed = np.array([4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14,
                  14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19,
                  20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25], dtype=float)
dist = np.array([2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46,
                 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46,
                 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85], dtype=float)

coeffs = np.polyfit(speed, dist, 2)
predictions = np.polyval(coeffs, speed)
residuals = dist - predictions

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: predictions vs observations
ax = axes[0]
ax.scatter(speed, dist, alpha=0.7, label='Observations')
speed_grid = np.linspace(4, 25, 100)
ax.plot(speed_grid, np.polyval(coeffs, speed_grid), 'k--', alpha=0.6, label='Prédictions')
for i in range(len(speed)):
    ax.plot([speed[i], speed[i]], [dist[i], predictions[i]], 'C1-', alpha=0.5)
ax.set_xlabel('Vitesse (mph)')
ax.set_ylabel('Distance (ft)')
ax.legend()
ax.set_title('Résidus: écarts entre observations et prédictions')

# Right: histogram of squared residuals
ax = axes[1]
ax.hist(residuals**2, bins=15, edgecolor='black', alpha=0.7)
ax.set_xlabel(r'Perte quadratique $(y - \hat{y})^2$')
ax.set_ylabel('Fréquence')
ax.set_title(f'MSE = {np.mean(residuals**2):.1f}')

plt.tight_layout()
```

Pour la classification, un choix naturel est la **perte 0-1**:

$$
\ell_{0-1}(y, \hat{y}) = \mathbb{1}_{y \neq \hat{y}} = \begin{cases} 0 & \text{si } y = \hat{y} \\ 1 & \text{si } y \neq \hat{y} \end{cases}
$$

Cette perte compte simplement les erreurs: elle vaut 1 pour une mauvaise prédiction, 0 sinon.

Le choix de la fonction de perte dépend du problème. En diagnostic médical, manquer une maladie grave (faux négatif) peut avoir des conséquences bien plus importantes que de prescrire un test supplémentaire à un patient sain (faux positif). Une perte asymétrique refléterait cette différence. En régression, si les grandes erreurs sont particulièrement problématiques, la perte quadratique est appropriée; si nous voulons être robustes aux valeurs aberrantes, la perte absolue $|y - \hat{y}|$ est préférable.

## Le risque

La perte évalue une seule prédiction. Pour évaluer un modèle dans son ensemble, nous voulons mesurer sa performance moyenne sur toutes les données possibles, pas seulement sur les exemples que nous avons observés.

### Pourquoi des variables aléatoires?

Une question naturelle se pose: si nous ajustons une fonction déterministe $f$ à des données, pourquoi avons-nous besoin de variables aléatoires et d'espérances? La fonction obtenue n'est-elle pas simplement une courbe fixe?

La réponse tient en un mot: **généralisation**. Nous ne nous intéressons pas vraiment à la performance sur les données d'entraînement car ces points sont déjà connus. Ce qui compte, c'est la performance sur des données *futures* que nous n'avons pas encore observées.

Considérons l'exemple de la distance de freinage. Les 50 mesures dans notre tableau sont un *échantillon* de toutes les mesures possibles. Si nous retournions sur le terrain et mesurions à nouveau, nous obtiendrions des valeurs légèrement différentes. En effet, le même véhicule à 20 mph ne s'arrête pas exactement à la même distance à chaque essai. Il y a de la variabilité intrinsèque: état de la route, température des freins, réflexes du conducteur.

Cette variabilité est capturée par une distribution de probabilité $p(x, y)$. Nos 50 points sont des tirages de cette distribution. La question fondamentale devient alors:

> *Notre modèle $f$ prédira-t-il bien sur de **nouveaux** tirages de cette même distribution?*

La fonction $f$ elle-même est déterministe une fois entraînée. Mais son *évaluation*, savoir si elle prédit bien ou mal, dépend de quelles données futures elle rencontrera. Et ces données futures sont incertaines: elles seront tirées de $p(x, y)$, mais nous ne savons pas lesquelles.

Le **risque** formalise cette idée: c'est la perte moyenne que subira notre modèle $f$ lorsqu'il sera confronté à des données tirées de $p(x, y)$. C'est une mesure de performance *prospective*, tournée vers le futur.

```{admonition} Modèles déterministes vs stochastiques (et pourquoi on s'en fiche un peu)
:class: note

Il existe deux façons de raconter la même histoire.

- **Modèle déterministe**: on suppose qu'il existe une relation $y \approx f^\star(x)$, et que les écarts proviennent de facteurs non modélisés (mesure bruitée, variabilité du monde réel). Ici, $f$ est une fonction déterministe; l'aléatoire vit dans les données que l'on observe et dans celles que l'on observera demain.

- **Modèle stochastique**: on suppose plutôt que $Y$ est une variable aléatoire conditionnellement à $X=x$, via une distribution $p(y\mid x)$. La \"bonne\" prédiction devient alors une question de moyenne/quantile/probabilité, selon la perte.

Dans la pratique, ces deux points de vue sont surtout des **langages** différents. Le formalisme probabiliste est souvent plus commode: il permet d'exprimer simplement \"la performance moyenne sur des données futures\" via une espérance. Ce chapitre adopte ce langage parce qu'il rend la généralisation et les garanties mathématiques plus propres, sans changer l'objectif final: produire une règle de prédiction utile.
```

### Définition formelle

Le **risque** d'une fonction $f$ est l'espérance de la perte sur la distribution des données:

$$
\mathcal{R}(f) = \mathbb{E}_{(\mathbf{X},Y) \sim p}\left[\ell(Y, f(\mathbf{X}))\right] = \int \ell(y, f(\mathbf{x})) \, p(\mathbf{x}, y) \, d\mathbf{x} \, dy
$$

Décomposons cette formule étape par étape:

1. **$\mathbb{E}_{(\mathbf{X},Y) \sim p}$**: L'espérance mathématique signifie "moyenne sur tous les exemples possibles". La notation $(\mathbf{X},Y) \sim p$ indique que nous tirons les paires $(\mathbf{x}, y)$ selon la distribution $p(\mathbf{x}, y)$ de la nature.

2. **$\ell(Y, f(\mathbf{X}))$**: Pour chaque exemple aléatoire $(\mathbf{X}, Y)$, nous calculons la perte entre la vraie valeur $Y$ et la prédiction $f(\mathbf{X})$ du modèle.

3. **L'intégrale $\int \ell(y, f(\mathbf{x})) \, p(\mathbf{x}, y) \, d\mathbf{x} \, dy$**: Cette intégrale calcule une moyenne pondérée. Pour chaque paire possible $(\mathbf{x}, y)$, nous multiplions la perte $\ell(y, f(\mathbf{x}))$ par la probabilité $p(\mathbf{x}, y)$ que cette paire apparaisse dans la nature, puis nous sommons (intégrons) sur toutes les paires possibles.

### Exemple concret

Considérons un problème de classification binaire en 2D. Supposons que $\mathbf{x} \in [0, 1]^2$ et $y \in \{0, 1\}$. Pour calculer le risque, nous devrions:

1. Diviser l'espace $[0,1]^2$ en une grille fine (par exemple, $1000 \times 1000$ points)
2. Pour chaque point $\mathbf{x}$ de la grille, considérer les deux valeurs possibles de $y$ (0 et 1)
3. Pour chaque combinaison $(\mathbf{x}, y)$, calculer:
   - La probabilité $p(\mathbf{x}, y)$ que cette combinaison apparaisse
   - La perte $\ell(y, f(\mathbf{x}))$ si notre modèle prédit $f(\mathbf{x})$
4. Faire la somme pondérée: $\sum_{\mathbf{x}} \sum_{y \in \{0,1\}} \ell(y, f(\mathbf{x})) \cdot p(\mathbf{x}, y)$

En pratique, pour un espace continu, cette somme devient une intégrale sur un domaine continu, ce qui est encore plus complexe à calculer.

Visualisons ceci concrètement. La figure suivante montre un problème de classification binaire où chaque classe suit une distribution gaussienne en 2D. Les contours représentent la densité $p(x|y)$ pour chaque classe. La ligne pointillée est la frontière de décision d'un classificateur linéaire. Les régions ombrées indiquent où le classificateur fait des erreurs: la région rouge correspond aux points de classe 0 classés comme classe 1, et la région bleue correspond aux points de classe 1 classés comme classe 0.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Paramètres du mélange gaussien (classification binaire 2D)
mu0 = np.array([0.0, 0.0])
mu1 = np.array([2.0, 1.0])
cov = np.array([[1.0, 0.3], [0.3, 1.0]])

def gaussian_pdf(x, mu, cov):
    """PDF d'une gaussienne multivariée."""
    d = len(mu)
    diff = x - mu
    cov_inv = np.linalg.inv(cov)
    mahal = np.einsum('...i,ij,...j->...', diff, cov_inv, diff)
    return np.exp(-0.5 * mahal) / np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))

# Create grid for visualization
x_range = np.linspace(-3, 5, 200)
y_range = np.linspace(-3, 4, 200)
X_grid, Y_grid = np.meshgrid(x_range, y_range)
pos = np.dstack([X_grid, Y_grid])

# Compute class-conditional densities
p_x_given_0 = gaussian_pdf(pos, mu0, cov)
p_x_given_1 = gaussian_pdf(pos, mu1, cov)

# Joint densities (with equal priors)
prior = 0.5
p_x_y0 = p_x_given_0 * (1 - prior)
p_x_y1 = p_x_given_1 * prior

# Linear decision boundary (Bayes optimal for equal covariances)
# w^T x + b = 0 where w = Sigma^{-1}(mu1 - mu0)
cov_inv = np.linalg.inv(cov)
w = cov_inv @ (mu1 - mu0)
b = -0.5 * (mu1 @ cov_inv @ mu1 - mu0 @ cov_inv @ mu0)

# Decision boundary: w[0]*x + w[1]*y + b = 0  =>  y = -(w[0]*x + b)/w[1]
x_boundary = np.linspace(-3, 5, 100)
y_boundary = -(w[0] * x_boundary + b) / w[1]

# Classifier prediction: classify as 1 if w^T x + b > 0
predictions = (w[0] * X_grid + w[1] * Y_grid + b) > 0

# Misclassification regions
# Class 0 misclassified as 1: true class is 0, but prediction is 1
misclass_0 = predictions  # region where we predict 1
# Class 1 misclassified as 0: true class is 1, but prediction is 0
misclass_1 = ~predictions  # region where we predict 0

fig, ax = plt.subplots(figsize=(8, 6))

# Plot class-conditional densities as contours
levels = [0.01, 0.05, 0.1, 0.15]
ax.contour(X_grid, Y_grid, p_x_given_0, levels=levels, colors='C0', alpha=0.7)
ax.contour(X_grid, Y_grid, p_x_given_1, levels=levels, colors='C1', alpha=0.7)

# Shade misclassification regions weighted by probability
# Red: class 0 points incorrectly classified as 1
error_region_0 = np.where(misclass_0, p_x_y0, 0)
# Blue: class 1 points incorrectly classified as 0  
error_region_1 = np.where(misclass_1, p_x_y1, 0)

ax.contourf(X_grid, Y_grid, error_region_0, levels=[0.001, 0.01, 0.05, 0.1], 
            colors=['#ff000010', '#ff000030', '#ff000050'], extend='max')
ax.contourf(X_grid, Y_grid, error_region_1, levels=[0.001, 0.01, 0.05, 0.1],
            colors=['#0000ff10', '#0000ff30', '#0000ff50'], extend='max')

# Decision boundary
ax.plot(x_boundary, y_boundary, 'k--', linewidth=2, label='Frontière de décision')

# Class centers
ax.scatter(*mu0, s=100, c='C0', marker='x', linewidths=3, zorder=5, label='Centre classe 0')
ax.scatter(*mu1, s=100, c='C1', marker='x', linewidths=3, zorder=5, label='Centre classe 1')

ax.set_xlim(-3, 5)
ax.set_ylim(-3, 4)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.legend(loc='upper left')
ax.set_aspect('equal')

plt.tight_layout()
```

Le risque est l'intégrale de la perte sur tout l'espace, pondérée par $p(\mathbf{x}, y)$. Les régions ombrées contribuent au risque: chaque point dans ces régions est mal classé, et sa contribution dépend de la densité de probabilité à cet endroit. Les régions denses proches de la frontière contribuent le plus au risque.

### Pourquoi le risque est important

Le risque mesure ce que nous obtiendrons en moyenne si nous appliquons $f$ à de nouvelles données tirées de la même distribution. Un modèle avec un faible risque fait de bonnes prédictions en général, pas seulement sur les exemples d'entraînement. C'est exactement ce que nous voulons optimiser: un modèle qui performe bien sur des données jamais vues, pas seulement sur celles qu'il a déjà observées.

Cette quantité est ce que nous voulons minimiser. Le problème fondamental est que nous ne connaissons pas la distribution $p(\mathbf{x}, y)$ de la nature. Nous n'y avons accès qu'indirectement, via un échantillon fini $\mathcal{D}$.

## Le risque empirique

Puisque le risque est inaccessible, nous l'approximons par une moyenne sur les données disponibles. Le **risque empirique** est:

$$
\hat{\mathcal{R}}(f, \mathcal{D}) = \frac{1}{N} \sum_{i=1}^{N} \ell(y_i, f(\mathbf{x}_i))
$$

Cette quantité est calculable: c'est la moyenne des pertes sur l'échantillon d'entraînement. Pour la perte 0-1, le risque empirique est le taux d'erreur sur les données d'entraînement. Pour la perte quadratique, c'est l'erreur quadratique moyenne.

### Pourquoi le risque est-il inaccessible?

La nécessité d'utiliser le risque empirique découle de deux obstacles fondamentaux, l'un conceptuel et l'autre computationnel.

#### Obstacle 1: La distribution $p(\mathbf{x}, y)$ est inconnue

La nature possède une distribution $p(\mathbf{x}, y)$ qui génère les données, mais nous ne la connaissons pas. Nous n'observons qu'un échantillon fini $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ tiré de cette distribution. C'est comme si nous regardions quelques gouttes d'eau d'un océan: nous pouvons analyser ces gouttes, mais nous ne connaissons pas la composition complète de l'océan.

Même si nous tentions d'estimer $p(\mathbf{x}, y)$ à partir des données (par exemple, via des techniques d'estimation de densité comme les mélanges de gaussiennes ou les estimateurs à noyau), nous n'obtiendrions qu'une approximation $\hat{p}(\mathbf{x}, y)$ de la vraie distribution. Cette approximation serait elle-même imparfaite et dépendrait de nos hypothèses sur la forme de la distribution.

La figure suivante illustre ce problème. À gauche, la vraie distribution $p(\mathbf{x}, y)$ que la nature utilise pour générer les données (que nous ne connaissons pas). À droite, un échantillon de $N = 50$ points tirés de cette distribution (ce que nous observons).

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Paramètres du mélange gaussien
mu0, mu1 = np.array([0.0, 0.0]), np.array([2.0, 1.0])
cov = np.array([[1.0, 0.3], [0.3, 1.0]])

def gaussian_pdf(x, mu, cov):
    d = len(mu)
    diff = x - mu
    cov_inv = np.linalg.inv(cov)
    mahal = np.einsum('...i,ij,...j->...', diff, cov_inv, diff)
    return np.exp(-0.5 * mahal) / np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))

# Générer un échantillon
rng = np.random.default_rng(42)
n = 50
X0 = rng.multivariate_normal(mu0, cov, n // 2)
X1 = rng.multivariate_normal(mu1, cov, n // 2)
X = np.vstack([X0, X1])
y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])

# Grille pour visualisation
x_range = np.linspace(-3, 5, 150)
y_range = np.linspace(-3, 4, 150)
X_grid, Y_grid = np.meshgrid(x_range, y_range)
pos = np.dstack([X_grid, Y_grid])

p_x_given_0 = gaussian_pdf(pos, mu0, cov)
p_x_given_1 = gaussian_pdf(pos, mu1, cov)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: True distribution (what nature knows)
ax = axes[0]
ax.contourf(X_grid, Y_grid, p_x_given_0, levels=15, cmap='Blues', alpha=0.6)
ax.contourf(X_grid, Y_grid, p_x_given_1, levels=15, cmap='Oranges', alpha=0.6)
ax.contour(X_grid, Y_grid, p_x_given_0, levels=5, colors='C0', alpha=0.8)
ax.contour(X_grid, Y_grid, p_x_given_1, levels=5, colors='C1', alpha=0.8)
ax.scatter(*mu0, s=100, c='C0', marker='x', linewidths=3, zorder=5)
ax.scatter(*mu1, s=100, c='C1', marker='x', linewidths=3, zorder=5)
ax.set_xlim(-3, 5)
ax.set_ylim(-3, 4)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Distribution vraie $p(x, y)$\n(inconnue)')
ax.set_aspect('equal')

# Right: Finite sample (what we observe)
ax = axes[1]
ax.scatter(X[y == 0, 0], X[y == 0, 1], c='C0', alpha=0.7, s=50, label='Classe 0')
ax.scatter(X[y == 1, 0], X[y == 1, 1], c='C1', alpha=0.7, s=50, label='Classe 1')
ax.set_xlim(-3, 5)
ax.set_ylim(-3, 4)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title(f'Échantillon observé $\\mathcal{{D}}$\n($N = {len(X)}$ points)')
ax.legend()
ax.set_aspect('equal')

plt.tight_layout()
```

Nous ne voyons que les points à droite. La structure continue à gauche, incluant les contours, les densités, ainsi que les régions de haute et basse probabilité, nous est cachée. C'est à partir de ces quelques points que nous devons estimer la performance de notre modèle.

#### Obstacle 2: L'intégration est computationnellement intractable

Supposons, par un miracle, que nous connaissions exactement $p(\mathbf{x}, y)$. Pourrions-nous alors calculer le risque $\mathcal{R}(f) = \int \ell(y, f(\mathbf{x})) \, p(\mathbf{x}, y) \, d\mathbf{x} \, dy$?

La réponse est généralement non, pour plusieurs raisons:

**Pour les espaces continus**: L'intégrale est une intégrale de grande dimension. Si $\mathbf{x} \in \mathbb{R}^d$ avec $d$ grand (par exemple, $d = 1000$ pour des images ou $d = 10^6$ pour des données textuelles), nous devons intégrer sur un espace de dimension $d+1$.

Pour vous rappeler l'idée, en calcul on approche une intégrale en 1D par une somme: on découpe l'intervalle en petites tranches et on additionne des aires de rectangles ou de trapèzes. Par exemple, sur $[a,b]$:

$$
\int_a^b g(x)\,dx \;\approx\; \sum_{m=1}^{M} g(x_m)\,\Delta x
$$

Cette idée générale, qui consiste à remplacer une intégrale par une somme pondérée de valeurs de $g$ évaluées à des points $x_m$, s'appelle **l'intégration numérique** (ou **quadrature**).

Le problème en apprentissage est que notre intégrale n'est pas en 1D. Si on applique le même raisonnement en dimension $d$ en mettant, disons, $M$ points par dimension, on obtient une grille de taille $M^d$ (et ici $d$ peut être très grand). Le nombre de points à évaluer explose donc exponentiellement avec $d$. C'est exactement la **malédiction de la dimensionnalité**.

**Pour les espaces discrets**: Si $\mathbf{x}$ et $y$ sont discrets mais prennent de nombreuses valeurs, la somme $\sum_{\mathbf{x}} \sum_y \ell(y, f(\mathbf{x})) \cdot p(\mathbf{x}, y)$ peut avoir un nombre exponentiel de termes. Par exemple, si $\mathbf{x}$ est un vecteur binaire de dimension $d$, il y a $2^d$ valeurs possibles pour $\mathbf{x}$. Pour $d = 100$, cela fait déjà $2^{100} \approx 10^{30}$ termes à sommer, ce qui est computationnellement impossible.

**Intégration de Monte Carlo**: On pourrait penser utiliser l'intégration de Monte Carlo: tirer des échantillons $(\mathbf{x}, y)$ selon $p(\mathbf{x}, y)$ et estimer l'intégrale par la moyenne empirique. Mais pour obtenir une estimation précise du risque, nous aurions besoin d'un très grand nombre d'échantillons (potentiellement infini pour une précision parfaite). De plus, cela nécessiterait de pouvoir échantillonner efficacement depuis $p(\mathbf{x}, y)$, ce qui est lui-même un problème difficile si la distribution est complexe.

La figure suivante illustre la malédiction de la dimensionnalité. Avec seulement 10 points par dimension pour une quadrature numérique, le nombre total de points d'évaluation explose rapidement.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Number of grid points per dimension
points_per_dim = 10

# Dimensions to consider
dimensions = np.array([1, 2, 3, 5, 10, 20, 50, 100])

# Total grid points = points_per_dim^d
total_points = points_per_dim ** dimensions.astype(float)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(range(len(dimensions)), total_points, color='steelblue', edgecolor='black')

# Add reference lines
ax.axhline(y=1e9, color='C1', linestyle='--', alpha=0.7, label='1 milliard (limite pratique)')
ax.axhline(y=1e80, color='C3', linestyle=':', alpha=0.7, label='$10^{80}$ (atomes dans l\'univers)')

ax.set_yscale('log')
ax.set_xticks(range(len(dimensions)))
ax.set_xticklabels([f'd={d}' for d in dimensions])
ax.set_xlabel('Dimension de l\'espace des entrées')
ax.set_ylabel('Nombre de points de grille')
ax.set_title(f'Points nécessaires pour l\'intégration numérique\n({points_per_dim} points par dimension)')
ax.legend(loc='upper left')

# Annotate a few bars
for i, (d, n) in enumerate(zip(dimensions, total_points)):
    if d <= 5:
        ax.annotate(f'$10^{{{d}}}$', (i, n), ha='center', va='bottom', fontsize=9)
    elif d == 10:
        ax.annotate(f'$10^{{{10}}}$', (i, n), ha='center', va='bottom', fontsize=9)
    elif d == 100:
        ax.annotate(f'$10^{{{100}}}$', (i, n), ha='center', va='bottom', fontsize=9)

ax.set_ylim(1, 1e105)

plt.tight_layout()
```

En dimension 10, il faut déjà $10^{10}$ points, soit dix milliards. En dimension 100, il en faut $10^{100}$, un nombre qui dépasse le nombre d'atomes dans l'univers observable. L'intégration numérique directe est donc impossible pour les problèmes de haute dimension, même si nous connaissions $p(\mathbf{x}, y)$ exactement.

### Le risque empirique comme seule option pratique

Face à ces obstacles, le risque empirique est notre seule option calculable. Mais il y a une bonne nouvelle: le risque empirique est une forme d'**intégration de Monte Carlo**, et Monte Carlo a une propriété remarquable.

| Méthode | Complexité | Exigence |
|---------|------------|----------|
| Quadrature (règles trapézoïdales, etc.) | $O(M^d)$ | Connaître $p(x,y)$ exactement |
| Monte Carlo | $O(N)$ | Avoir des échantillons de $p(x,y)$ |

La complexité de Monte Carlo est **indépendante de la dimension** $d$. Elle ne dépend que du nombre d'échantillons $N$. C'est cette propriété qui rend l'apprentissage possible en haute dimension. De plus, nous n'avons même pas besoin de connaître la valeur numérique de $p(x,y)$; il suffit de pouvoir tirer des échantillons de cette distribution. C'est exactement ce que nos données d'entraînement nous fournissent.

Le risque empirique remplace l'intégrale sur la distribution inconnue par une moyenne sur l'échantillon fini que nous possédons:

$$
\hat{\mathcal{R}}(f, \mathcal{D}) = \frac{1}{N} \sum_{i=1}^{N} \ell(y_i, f(x_i))
$$

Cette formule est simple à calculer: nous parcourons nos $N$ exemples d'entraînement, calculons la perte pour chacun, et faisons la moyenne.

Reprenons les données de freinage. Divisons-les en deux parties: les mesures à vitesses faibles (4-19 mph) pour l'entraînement, et les mesures à vitesses élevées (20-25 mph) pour le test. Le risque empirique sur l'ensemble d'entraînement mesure la qualité de l'ajustement. Le risque empirique sur l'ensemble de test estime la performance sur des vitesses non vues pendant l'entraînement.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Données de freinage
speed = np.array([4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14,
                  14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19,
                  20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25], dtype=float)
dist = np.array([2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46,
                 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46,
                 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85], dtype=float)

# Split: train on low speeds, test on high speeds
train_mask = speed < 20
test_mask = speed >= 20

speed_train, dist_train = speed[train_mask], dist[train_mask]
speed_test, dist_test = speed[test_mask], dist[test_mask]

# Fit on training data
coeffs = np.polyfit(speed_train, dist_train, 2)

# Compute MSE on train and test
pred_train = np.polyval(coeffs, speed_train)
pred_test = np.polyval(coeffs, speed_test)
mse_train = np.mean((dist_train - pred_train)**2)
mse_test = np.mean((dist_test - pred_test)**2)

plt.figure(figsize=(7, 4))
plt.scatter(speed_train, dist_train, alpha=0.7, label=f'Entraînement (MSE={mse_train:.1f})')
plt.scatter(speed_test, dist_test, alpha=0.7, marker='s', label=f'Test (MSE={mse_test:.1f})')

speed_grid = np.linspace(4, 28, 100)
plt.plot(speed_grid, np.polyval(coeffs, speed_grid), 'k--', alpha=0.6, label='Fonction ajustée')

plt.axvline(x=20, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('Vitesse (mph)')
plt.ylabel('Distance (ft)')
plt.legend()
plt.tight_layout()
```

Dans cet exemple, le MSE sur l'ensemble de test est plus élevé que sur l'ensemble d'entraînement. Cet écart est typique: la fonction a été optimisée pour les données d'entraînement, pas pour les données de test.

Sous l'hypothèse que les exemples $(x_i, y_i)$ sont tirés indépendamment et identiquement distribués (i.i.d.) selon $p(x, y)$, le risque empirique est un estimateur non biaisé du vrai risque: $\mathbb{E}[\hat{\mathcal{R}}(f, \mathcal{D})] = \mathcal{R}(f)$. Cela signifie qu'en moyenne, sur tous les échantillons possibles, le risque empirique est égal au vrai risque.

Par la loi des grands nombres, lorsque $N \to \infty$, le risque empirique converge vers le vrai risque (presque sûrement). Avec suffisamment de données, si l'échantillon est représentatif de la distribution, le risque empirique devrait être proche du risque.

La figure suivante illustre cette convergence. Nous utilisons le problème de classification gaussienne pour lequel nous pouvons calculer le vrai risque analytiquement. Chaque courbe montre l'évolution du risque empirique pour un échantillon de taille croissante. Toutes les courbes convergent vers le vrai risque (ligne pointillée), mais avec des fluctuations qui diminuent à mesure que $N$ augmente.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Paramètres du mélange gaussien
mu0, mu1 = np.array([0.0, 0.0]), np.array([2.0, 1.0])
cov = np.array([[1.0, 0.3], [0.3, 1.0]])

def gaussian_pdf(x, mu, cov):
    d = len(mu)
    diff = x - mu
    cov_inv = np.linalg.inv(cov)
    mahal = np.einsum('...i,ij,...j->...', diff, cov_inv, diff)
    return np.exp(-0.5 * mahal) / np.sqrt((2 * np.pi) ** d * np.linalg.det(cov))

# Compute Bayes-optimal classifier error rate (true risk)
# For Gaussian classes with equal covariance, the Bayes error is:
# P(error) = Phi(-d/2) where d is the Mahalanobis distance between means
cov_inv = np.linalg.inv(cov)
d_squared = (mu1 - mu0) @ cov_inv @ (mu1 - mu0)
d = np.sqrt(d_squared)
true_risk = norm.cdf(-d / 2)

# Simulate empirical risk for different sample sizes
sample_sizes = np.arange(10, 1001, 10)
n_runs = 20

fig, ax = plt.subplots(figsize=(9, 5))

# Store all runs for confidence band
all_risks = np.zeros((n_runs, len(sample_sizes)))

for run in range(n_runs):
    empirical_risks = []
    # Generate a large dataset and compute cumulative empirical risk
    rng = np.random.default_rng(run)
    n_total = 1000
    X0 = rng.multivariate_normal(mu0, cov, n_total // 2)
    X1 = rng.multivariate_normal(mu1, cov, n_total // 2)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_total // 2), np.ones(n_total // 2)])
    perm = rng.permutation(n_total)
    X, y = X[perm], y[perm]
    
    # Bayes-optimal classifier: predict 1 if w^T x + b > 0
    w = cov_inv @ (mu1 - mu0)
    b = -0.5 * (mu1 @ cov_inv @ mu1 - mu0 @ cov_inv @ mu0)
    
    for n in sample_sizes:
        X_n, y_n = X[:n], y[:n]
        predictions = (X_n @ w + b > 0).astype(float)
        emp_risk = np.mean(predictions != y_n)
        empirical_risks.append(emp_risk)
    
    all_risks[run] = empirical_risks
    ax.plot(sample_sizes, empirical_risks, 'C0-', alpha=0.15, linewidth=0.8)

# Mean and confidence bands
mean_risk = np.mean(all_risks, axis=0)
std_risk = np.std(all_risks, axis=0)
ax.fill_between(sample_sizes, mean_risk - 2*std_risk, mean_risk + 2*std_risk, 
                alpha=0.3, color='C0', label='Intervalle ± 2 écarts-types')
ax.plot(sample_sizes, mean_risk, 'C0-', linewidth=2, label='Moyenne empirique')

# True risk
ax.axhline(y=true_risk, color='C3', linestyle='--', linewidth=2, 
           label=f'Vrai risque = {true_risk:.3f}')

ax.set_xlabel('Taille de l\'échantillon $N$')
ax.set_ylabel('Risque empirique (taux d\'erreur)')
ax.set_title('Convergence du risque empirique vers le vrai risque')
ax.legend(loc='upper right')
ax.set_xlim(0, 1000)
ax.set_ylim(0, 0.35)

plt.tight_layout()
```

Avec $N = 50$, le risque empirique peut facilement varier de 0.10 à 0.25 selon l'échantillon. Avec $N = 500$, la variabilité est beaucoup plus faible. C'est la loi des grands nombres en action: plus l'échantillon est grand, plus l'estimation est précise.

### Le compromis fondamental

Cette situation crée un compromis fondamental en apprentissage automatique:

- **Ce que nous voulons minimiser**: Le risque $\mathcal{R}(f)$, qui mesure la performance sur toutes les données possibles
- **Ce que nous pouvons minimiser**: Le risque empirique $\hat{\mathcal{R}}(f, \mathcal{D})$, qui mesure la performance sur nos données d'entraînement

L'écart entre ces deux quantités est au cœur de l'apprentissage automatique. Un modèle peut avoir un risque empirique très faible (il performe bien sur les données d'entraînement) tout en ayant un risque élevé (il performe mal sur de nouvelles données). C'est le problème du surapprentissage, que nous explorerons plus en détail dans le chapitre sur la généralisation.

La question de savoir quand et à quelle vitesse l'approximation du risque par le risque empirique est fiable relève de la théorie de la généralisation, que nous aborderons au chapitre suivant.

## Minimisation du risque empirique

Nous avons maintenant les éléments pour formuler l'apprentissage comme un problème d'optimisation. Nous cherchons la fonction $f$ dans une classe $\mathcal{F}$ qui minimise le risque:

$$
f^\star = \arg\min_{f \in \mathcal{F}} \mathcal{R}(f)
$$

Puisque le risque est inaccessible, nous le remplaçons par le risque empirique:

$$
\hat{f} = \arg\min_{f \in \mathcal{F}} \hat{\mathcal{R}}(f, \mathcal{D})
$$

Ce principe est la **minimisation du risque empirique** (MRE). L'idée est simple: choisir la fonction qui fait le moins d'erreurs sur les données d'entraînement, en espérant que cette performance se transfère aux nouvelles données.

La classe $\mathcal{F}$ est notre **classe d'hypothèses**. Elle représente l'ensemble des fonctions que nous sommes prêts à considérer. Par exemple, si $\mathcal{F}$ est l'ensemble des fonctions linéaires, nous cherchons la meilleure fonction linéaire. Si $\mathcal{F}$ est l'ensemble des polynômes de degré au plus $k$, nous cherchons le meilleur polynôme de ce degré. Le choix de $\mathcal{F}$ encode nos hypothèses sur la forme de la relation entre entrées et sorties.

La question centrale de la théorie de l'apprentissage est: quand le minimiseur du risque empirique a-t-il un faible risque? Si $\hat{f}$ minimise $\hat{\mathcal{R}}$ et $f^\star$ minimise $\mathcal{R}$, nous voulons que $\mathcal{R}(\hat{f})$ soit proche de $\mathcal{R}(f^\star)$. Cette question dépend de la taille de l'échantillon $N$, de la complexité de la classe $\mathcal{F}$, et de propriétés de la distribution $p$.

## Fonctions de perte de substitution

La perte 0-1 pose un problème pratique. Pour trouver le minimiseur du risque empirique, nous utilisons généralement des méthodes d'optimisation itératives comme la descente de gradient. Ces méthodes requièrent que la fonction objectif soit différentiable, or la perte 0-1 est constante par morceaux: sa dérivée est nulle presque partout et indéfinie aux points de discontinuité.

Nous contournons ce problème en utilisant des **fonctions de perte de substitution**: des approximations convexes et différentiables de la perte originale.

Pour la classification binaire avec $y \in \{-1, +1\}$, la **perte logistique** est:

$$
\ell_{\text{log}}(y, s) = \log(1 + e^{-y \cdot s})
$$

où $s = f(x)$ est le score produit par le modèle. Cette fonction est convexe et différentiable partout. Lorsque $y$ et $s$ ont le même signe (prédiction correcte avec confiance), la perte est faible. Lorsqu'ils ont des signes opposés (erreur), la perte croît linéairement avec l'amplitude de l'erreur.

La **perte à charnière** (hinge loss) est utilisée dans les machines à vecteurs de support:

$$
\ell_{\text{hinge}}(y, s) = \max(0, 1 - y \cdot s)
$$

Cette fonction est convexe mais non différentiable au point $y \cdot s = 1$. Elle est nulle lorsque la prédiction est correcte avec une marge suffisante ($y \cdot s \geq 1$), et croît linéairement sinon.

Ces deux fonctions majorent la perte 0-1: pour tout $y$ et $s$, nous avons $\ell_{0-1} \leq \ell_{\text{log}}$ et $\ell_{0-1} \leq \ell_{\text{hinge}}$. Minimiser ces substituts garantit donc un certain contrôle sur la perte originale.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Margin: y * s (positive = correct prediction, negative = error)
margin = np.linspace(-3, 3, 500)

# 0-1 loss: 1 if margin < 0, else 0
loss_01 = (margin < 0).astype(float)

# Logistic loss: log(1 + exp(-margin))
loss_log = np.log(1 + np.exp(-margin))

# Hinge loss: max(0, 1 - margin)
loss_hinge = np.maximum(0, 1 - margin)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(margin, loss_01, 'k-', linewidth=2, label='Perte 0-1')
ax.plot(margin, loss_log, 'C0-', linewidth=2, label='Perte logistique')
ax.plot(margin, loss_hinge, 'C1-', linewidth=2, label='Perte à charnière')

ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
ax.axhline(1, color='gray', linestyle=':', alpha=0.3)

ax.set_xlabel(r'Marge $y \cdot s$')
ax.set_ylabel('Perte')
ax.set_xlim(-3, 3)
ax.set_ylim(-0.1, 4)
ax.legend()
ax.set_title('Fonctions de perte de substitution comme bornes supérieures convexes')

# Annotate regions
ax.text(-1.5, 3.5, 'Erreur\n(prédiction incorrecte)', ha='center', fontsize=9, color='gray')
ax.text(1.5, 0.3, 'Correct\n(prédiction juste)', ha='center', fontsize=9, color='gray')

plt.tight_layout()
```

La figure montre les trois fonctions de perte en fonction de la marge $y \cdot s$. Une marge positive indique une prédiction correcte (le signe de $s$ correspond à $y$), une marge négative indique une erreur. La perte 0-1 est discontinue au point $y \cdot s = 0$. Les pertes logistique et à charnière sont continues et convexes, ce qui permet d'utiliser des méthodes d'optimisation par gradient. Elles majorent partout la perte 0-1.

## Maximum de vraisemblance

Jusqu'ici, nous avons choisi des fonctions de perte de manière ad hoc: la perte quadratique semble raisonnable pour la régression, la perte logistique pour la classification. Mais d'où viennent ces choix? Existe-t-il un principe unificateur?

Le **maximum de vraisemblance** offre une réponse: plutôt que de choisir une perte arbitraire, nous modélisons explicitement comment les données ont été générées, puis nous cherchons les paramètres qui rendent nos observations les plus probables.

### Construction de la vraisemblance

Supposons que nous avons un modèle paramétrique $p(y|x; \theta)$ qui, pour chaque entrée $x$ et choix de paramètres $\theta$, définit une distribution sur les sorties possibles $y$. Par exemple, en régression, ce pourrait être une gaussienne centrée sur $f(x; \theta)$.

Considérons un seul exemple $(x_1, y_1)$. Pour des paramètres $\theta$ fixés, nous pouvons évaluer $p(y_1 | x_1; \theta)$: la probabilité (ou densité) que le modèle assigne à l'observation $y_1$. Si cette valeur est élevée, les paramètres $\theta$ "expliquent bien" cette observation. Si elle est faible, $y_1$ est une valeur improbable sous ce modèle.

Avec deux exemples indépendants $(x_1, y_1)$ et $(x_2, y_2)$, la probabilité conjointe est le produit:

$$
p(y_1, y_2 | x_1, x_2; \theta) = p(y_1 | x_1; \theta) \cdot p(y_2 | x_2; \theta)
$$

Avec $N$ exemples indépendants, nous obtenons la **vraisemblance**:

$$
\mathcal{L}(\theta) = \prod_{i=1}^N p(y_i | x_i; \theta)
$$

Cette quantité est une fonction de $\theta$. Elle répond à la question: pour ce choix de paramètres, quelle est la probabilité d'avoir observé exactement ces données?

### Pourquoi maximiser?

Si $\mathcal{L}(\theta_A) > \mathcal{L}(\theta_B)$, alors les données observées sont plus probables sous $\theta_A$ que sous $\theta_B$. Les paramètres $\theta_A$ rendent les observations moins "surprenantes".

L'**estimateur du maximum de vraisemblance** (EMV) choisit les paramètres qui maximisent cette probabilité:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \mathcal{L}(\theta) = \arg\max_\theta \prod_{i=1}^N p(y_i | x_i; \theta)
$$

C'est le choix de paramètres sous lequel nos données sont les plus "attendues".

### Du produit à la somme

En pratique, multiplier $N$ probabilités (souvent petites) pose des problèmes numériques: le résultat devient rapidement trop petit pour être représenté par un ordinateur. Le logarithme résout ce problème: il transforme le produit en somme et, comme c'est une fonction croissante, il ne change pas le maximiseur:

$$
\log \mathcal{L}(\theta) = \sum_{i=1}^N \log p(y_i | x_i; \theta)
$$

Pour l'optimisation, nous préférons minimiser plutôt que maximiser (par convention). La **log-vraisemblance négative** (negative log-likelihood, NLL) est notre fonction objectif:

$$
\text{NLL}(\theta) = -\sum_{i=1}^N \log p(y_i | x_i; \theta)
$$

Remarquez la structure: c'est une somme sur les exemples d'une quantité $-\log p(y_i | x_i; \theta)$ qui dépend de chaque observation. Cette quantité joue le rôle d'une fonction de perte. Le maximum de vraisemblance est donc un cas particulier de la minimisation du risque empirique, où la perte est définie par le modèle probabiliste lui-même.

### Régression avec bruit gaussien: d'où vient la perte quadratique?

Appliquons ce principe à la régression. Le modèle de génération des données est simple: la sortie observée est la prédiction "vraie" du modèle, corrompue par un bruit aléatoire gaussien:

$$
y = f(x; \theta) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)
$$

Ce modèle dit que si nous connaissions les vrais paramètres $\theta$ et que nous mesurions $y$ pour un $x$ donné, nous obtiendrions $f(x; \theta)$ plus ou moins $\sigma$ la plupart du temps.

La distribution conditionnelle qui en découle est:

$$
p(y|x; \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - f(x; \theta))^2}{2\sigma^2}\right)
$$

Calculons la log-vraisemblance négative:

$$
\text{NLL}(\theta) = -\sum_{i=1}^N \log p(y_i | x_i; \theta) = \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - f(x_i; \theta))^2 + \frac{N}{2}\log(2\pi\sigma^2)
$$

Le second terme ne dépend pas de $\theta$. Minimiser la NLL revient donc exactement à minimiser la somme des erreurs quadratiques.

C'est un résultat fondamental: **la perte quadratique n'est pas un choix arbitraire**. Elle découle naturellement de l'hypothèse que les erreurs de mesure suivent une loi gaussienne. Le maximum de vraisemblance sous bruit gaussien coïncide avec les moindres carrés.

Dans ce modèle, nous avons supposé que la variance $\sigma^2$ est constante pour toutes les entrées $x$. C'est ce qu'on appelle la **régression homoscédastique** (du grec *homos*, même, et *skedasis*, dispersion). C'est l'hypothèse standard en régression linéaire.

En pratique, l'incertitude peut varier selon l'entrée. Par exemple, les mesures à haute vitesse peuvent être plus bruitées que celles à basse vitesse. La **régression hétéroscédastique** modélise cette variation en faisant dépendre la variance de $x$:

$$
p(y|x; \theta) = \mathcal{N}(y | f_\mu(x; \theta), f_\sigma(x; \theta)^2)
$$

où $f_\mu$ prédit la moyenne et $f_\sigma$ prédit l'écart-type. Ce modèle est plus flexible mais requiert d'apprendre des paramètres supplémentaires.

### Exemple: pharmacocinétique

Le MLE s'applique à des modèles non linéaires. Considérons la concentration d'un médicament dans le sang après administration orale. Les données suivantes proviennent d'une étude sur la théophylline, un bronchodilatateur:

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Données pharmacocinétiques: théophylline, sujet 1 (Boeckmann et al., 1994)
time = np.array([0, 0.25, 0.57, 1.12, 2.02, 3.82, 5.10, 7.03, 9.05, 12.12, 24.37])
conc = np.array([0.74, 2.84, 6.57, 10.50, 9.66, 8.58, 8.36, 7.47, 6.89, 5.94, 3.28])

# Model: C(t) = C0 * exp(-k * t) for t > t_peak
# We'll fit on the decay phase (after peak)
peak_idx = np.argmax(conc)
t_decay = time[peak_idx:]
c_decay = conc[peak_idx:]

# MLE: minimize NLL under Gaussian noise
def neg_log_likelihood(params, t, c):
    C0, k, sigma = params
    if sigma <= 0 or k <= 0 or C0 <= 0:
        return np.inf
    pred = C0 * np.exp(-k * (t - t[0]))
    nll = 0.5 * len(t) * np.log(2 * np.pi * sigma**2)
    nll += 0.5 * np.sum((c - pred)**2) / sigma**2
    return nll

# Initial guess and optimization
x0 = [c_decay[0], 0.1, 1.0]
result = minimize(neg_log_likelihood, x0, args=(t_decay, c_decay), method='Nelder-Mead')
C0_mle, k_mle, sigma_mle = result.x

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: data and fit
ax = axes[0]
ax.scatter(time, conc, s=50, zorder=5, label='Observations')
t_grid = np.linspace(time[peak_idx], 25, 100)
ax.plot(t_grid, C0_mle * np.exp(-k_mle * (t_grid - time[peak_idx])), 'k--', 
        label=f'MLE: $C_0$={C0_mle:.1f}, $k$={k_mle:.2f}')
ax.axvline(time[peak_idx], color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Temps (h)')
ax.set_ylabel('Concentration (mg/L)')
ax.legend()
ax.set_title('Concentration plasmatique de théophylline')

# Right: residuals
ax = axes[1]
pred_decay = C0_mle * np.exp(-k_mle * (t_decay - t_decay[0]))
residuals = c_decay - pred_decay
ax.stem(t_decay, residuals, basefmt=' ')
ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax.set_xlabel('Temps (h)')
ax.set_ylabel('Résidu (mg/L)')
ax.set_title(rf'$\sigma$ estimé: {sigma_mle:.2f} mg/L')

plt.tight_layout()
```

Le modèle $C(t) = C_0 e^{-kt}$ décrit la décroissance exponentielle après le pic de concentration. Les paramètres $C_0$ (concentration initiale) et $k$ (constante d'élimination) sont estimés par maximum de vraisemblance sous l'hypothèse d'un bruit gaussien. Cette approche est identique à celle des moindres carrés, mais elle fournit également une estimation de l'écart-type du bruit $\sigma$.

### Classification binaire

Rappelons que la perte 0-1 pour la classification est discontinue, ce qui empêche l'utilisation de méthodes de gradient. La fonction **sigmoïde** $\sigma(z) = 1/(1 + e^{-z})$ offre une solution élégante: c'est une **approximation lisse de la fonction échelon** (step function). Elle transforme n'importe quel score réel en une valeur dans l'intervalle $(0, 1)$, que nous pouvons interpréter comme une probabilité.

Cette interprétation probabiliste n'est pas qu'une astuce numérique. Elle correspond exactement à modéliser $Y | X$ par une distribution de **Bernoulli** dont le paramètre dépend de l'entrée.

Pour la classification binaire avec $y \in \{0, 1\}$, nous modélisons la probabilité de la classe positive par:

$$
p(y = 1 | x; \theta) = \sigma(f(x; \theta)) = \frac{1}{1 + e^{-f(x; \theta)}}
$$

où $\sigma$ est la fonction sigmoïde et $f(x; \theta)$ est le **logit** (ou log-odds), le score brut du modèle avant transformation. La distribution conditionnelle suit une loi de Bernoulli:

$$
p(y|x; \theta) = \sigma(f(x; \theta))^y (1 - \sigma(f(x; \theta)))^{1-y}
$$

La log-vraisemblance négative est:

$$
\text{NLL}(\theta) = -\sum_{i=1}^N \left[ y_i \log \sigma(f(x_i; \theta)) + (1-y_i) \log(1 - \sigma(f(x_i; \theta))) \right]
$$

Cette quantité est l'**entropie croisée binaire**. Elle correspond à la perte logistique, à une reparamétrisation près.

## Lien entre MRE et maximum de vraisemblance

La minimisation du risque empirique et l'estimation par maximum de vraisemblance sont deux formulations du même problème lorsque nous choisissons la perte logarithmique $\ell(y, f(x)) = -\log p(y | f(x))$.

Le risque empirique avec cette perte est:

$$
\hat{\mathcal{R}}(\theta) = \frac{1}{N} \sum_{i=1}^N -\log p(y_i | x_i; \theta) = \frac{1}{N} \text{NLL}(\theta)
$$

Le minimiseur du risque empirique est donc l'estimateur du maximum de vraisemblance. Les deux approches, l'une fondée sur la théorie de la décision et l'autre sur l'inférence statistique, convergent vers le même algorithme.

## Interprétation informationnelle

Nous pouvons interpréter le maximum de vraisemblance du point de vue de la théorie de l'information. L'EMV trouve le modèle paramétrique le plus proche de la distribution empirique des données.

La **distribution empirique** place une masse $1/N$ sur chaque observation:

$$
p_{\mathcal{D}}(y) = \frac{1}{N} \sum_{i=1}^N \delta(y - y_i)
$$

La **divergence de Kullback-Leibler** mesure la dissimilarité entre deux distributions:

$$
D_{\text{KL}}(p \| q) = \sum_y p(y) \log \frac{p(y)}{q(y)}
$$

Cette quantité est toujours positive ou nulle, et vaut zéro si et seulement si $p = q$. Elle n'est pas symétrique: $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$ en général.

En posant $p = p_{\mathcal{D}}$ et $q = p(\cdot | \theta)$:

$$
D_{\text{KL}}(p_{\mathcal{D}} \| p(\cdot|\theta)) = -\mathbb{H}(p_{\mathcal{D}}) - \frac{1}{N} \sum_{i=1}^N \log p(y_i | \theta)
$$

Le premier terme, l'entropie de la distribution empirique, ne dépend pas de $\theta$. Minimiser la divergence KL revient à minimiser la log-vraisemblance négative:

$$
\arg\min_\theta D_{\text{KL}}(p_{\mathcal{D}} \| p(\cdot|\theta)) = \arg\min_\theta \text{NLL}(\theta)
$$

L'EMV trouve les paramètres qui rendent le modèle aussi proche que possible de la distribution empirique au sens de la divergence KL.

## Classes de modèles

Nous n'avons pas encore précisé la forme des fonctions $f$ que nous considérons. Le choix de la classe d'hypothèses $\mathcal{F}$ détermine ce que le modèle peut représenter.

Le modèle le plus simple est la **régression linéaire**:

$$
f(x; w, b) = w^\top x + b = \sum_{j=1}^d w_j x_j + b
$$

Les paramètres sont le vecteur de poids $w \in \mathbb{R}^d$ et le biais $b \in \mathbb{R}$. Ce modèle suppose que la sortie est une combinaison linéaire des entrées.

### Trois familles de modèles

Avant de détailler les modèles linéaires, situons-les dans une hiérarchie plus large. Nous distinguons trois familles de modèles, de complexité croissante:

1. **Modèles linéaires**: $f(x; \theta) = \theta^\top x + b$. La sortie est une combinaison linéaire des entrées. Simple, interprétable, mais limité aux relations linéaires.

2. **Modèles à expansion de caractéristiques**: $f(x; \theta) = \theta^\top \phi(x) + b$, où $\phi: \mathbb{R}^d \to \mathbb{R}^D$ est une transformation non linéaire fixée à l'avance (par exemple, polynomiale). Le modèle reste linéaire dans les paramètres $\theta$, ce qui facilite l'optimisation, mais peut capturer des relations non linéaires en $x$. L'espace de redescription a souvent une dimension $D \gg d$.

3. **Réseaux de neurones**: $f(x; \theta) = f_K(f_{K-1}(\cdots f_1(x; \theta_1); \theta_{K-1}); \theta_K)$. Une composition de $K$ fonctions non linéaires, chacune avec ses propres paramètres. Contrairement aux modèles à expansion fixe, les réseaux de neurones **apprennent la représentation** $\phi$ en même temps que les paramètres $\theta$.

Cette progression capture l'évolution historique du domaine: des modèles linéaires classiques aux méthodes à noyaux (expansion implicite), puis aux réseaux profonds qui apprennent leurs propres représentations. Nous verrons les réseaux de neurones en détail dans les chapitres suivants; concentrons-nous ici sur les deux premières familles.

### Expansion de caractéristiques

Pour capturer des relations non linéaires tout en gardant un modèle linéaire dans les paramètres, nous pouvons transformer les entrées. En **régression polynomiale**, nous appliquons une fonction $\phi: \mathbb{R} \to \mathbb{R}^{k+1}$:

$$
\phi(x) = [1, x, x^2, \ldots, x^k]
$$

La prédiction devient $f(x; w) = w^\top \phi(x)$. Le modèle est polynomial en $x$ mais linéaire en $w$, ce qui permet d'utiliser les mêmes algorithmes d'optimisation.

Le degré $k$ contrôle la **capacité** du modèle: sa capacité à représenter des fonctions complexes. Avec $k = 1$, nous avons une droite. Avec $k$ élevé, le polynôme peut osciller pour passer par tous les points d'entraînement. Avec $k = N - 1$, nous pouvons interpoler exactement les $N$ points: le risque empirique atteint zéro. Mais un polynôme qui passe exactement par les points d'entraînement n'a aucune raison de bien prédire les nouveaux points.

Illustrons ce phénomène avec les données de freinage. Nous ajustons des polynômes de degrés 1, 2, 5 et 15, et comparons leurs erreurs sur les ensembles d'entraînement et de test.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress polyfit warnings for high-degree polynomials (expected for this demo)
warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

# Données de freinage
speed = np.array([4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14,
                  14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19,
                  20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25], dtype=float)
dist = np.array([2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46,
                 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46,
                 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85], dtype=float)

# Train/test split
np.random.seed(42)
indices = np.random.permutation(len(speed))
train_idx, test_idx = indices[:35], indices[35:]
speed_train, dist_train = speed[train_idx], dist[train_idx]
speed_test, dist_test = speed[test_idx], dist[test_idx]

degrees_to_plot = [1, 2, 5, 15]
degrees_eval = range(1, 16)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Pre-compute all errors for the summary plot later
all_train_errors = []
all_test_errors = []
for deg in degrees_eval:
    coeffs = np.polyfit(speed_train, dist_train, deg)
    all_train_errors.append(np.mean((dist_train - np.polyval(coeffs, speed_train))**2))
    all_test_errors.append(np.mean((dist_test - np.polyval(coeffs, speed_test))**2))

for ax, deg in zip(axes.flat, degrees_to_plot):
    # Fit polynomial
    coeffs = np.polyfit(speed_train, dist_train, deg)
    
    # Predictions
    pred_train = np.polyval(coeffs, speed_train)
    pred_test = np.polyval(coeffs, speed_test)
    
    # MSE
    mse_train = np.mean((dist_train - pred_train)**2)
    mse_test = np.mean((dist_test - pred_test)**2)
    
    # Plot
    ax.scatter(speed_train, dist_train, alpha=0.6, s=30, label='Train')
    ax.scatter(speed_test, dist_test, alpha=0.6, s=30, marker='s', label='Test')
    
    speed_grid = np.linspace(3, 26, 200)
    pred_grid = np.polyval(coeffs, speed_grid)
    # Clip extreme predictions for visualization
    pred_grid = np.clip(pred_grid, -50, 200)
    ax.plot(speed_grid, pred_grid, 'k-', alpha=0.7)
    
    ax.set_xlim(3, 26)
    ax.set_ylim(-20, 150)
    ax.set_xlabel('Vitesse (mph)')
    ax.set_ylabel('Distance (ft)')
    ax.set_title(f'Degré {deg}: Train MSE={mse_train:.1f}, Test MSE={mse_test:.1f}')
    if deg == 1:
        ax.legend()

plt.tight_layout()
```

Le polynôme de degré 1 (droite) ne capture pas la courbure des données: c'est du sous-apprentissage. Le polynôme de degré 2 capture bien la relation quadratique. Le polynôme de degré 5 commence à osciller. Le polynôme de degré 15 passe près de tous les points d'entraînement, mais ses oscillations produisent des prédictions absurdes entre les points: c'est du surapprentissage.

```{code-cell} python
:tags: [hide-input]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(degrees_eval, all_train_errors, 'o-', linewidth=2, label='Erreur entraînement')
ax.plot(degrees_eval, all_test_errors, 's-', linewidth=2, label='Erreur test')

# Utiliser une échelle logarithmique car l'erreur de test explose
ax.set_yscale('log')

ax.set_xlabel('Degré du polynôme (complexité)')
ax.set_ylabel('MSE (échelle log)')
ax.set_xticks(range(1, 16, 2))
ax.grid(True, which="both", ls="-", alpha=0.2)
ax.legend()

ax.set_title('Compromis biais-variance (Échelle logarithmique)')
plt.tight_layout()
```

L'erreur d'entraînement diminue avec le degré du polynôme. L'erreur de test diminue d'abord (quand le modèle gagne en expressivité), puis augmente (quand le modèle commence à mémoriser le bruit). Le meilleur modèle se trouve à l'intersection de ces deux tendances.

## Régularisation

Une alternative à la réduction du degré du polynôme consiste à pénaliser la complexité du modèle directement dans la fonction objectif. Le **risque empirique régularisé** est:

$$
\hat{\mathcal{R}}_\lambda(\theta) = \hat{\mathcal{R}}(\theta) + \lambda \, C(\theta)
$$

où $C(\theta)$ mesure la complexité du modèle et $\lambda \geq 0$ contrôle l'intensité de la pénalisation. Un choix courant est la **régularisation $\ell_2$** (ou *weight decay*):

$$
C(\theta) = \|\theta\|_2^2 = \sum_j \theta_j^2
$$

Cette pénalisation pousse les paramètres vers zéro, ce qui a pour effet de lisser la fonction apprise. En régression linéaire, l'ajout de cette pénalité donne la **régression ridge**:

$$
\hat{w}_{\text{ridge}} = \arg\min_w \frac{1}{N}\sum_{i=1}^N (y_i - w^\top x_i)^2 + \lambda \|w\|_2^2
$$

Illustrons l'effet de la régularisation sur le même problème de régression polynomiale. Avec un polynôme de degré 15 et différentes valeurs de $\lambda$:

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Données de freinage
speed = np.array([4, 4, 7, 7, 8, 9, 10, 10, 10, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14,
                  14, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19,
                  20, 20, 20, 20, 20, 22, 23, 24, 24, 24, 24, 25], dtype=float)
dist = np.array([2, 10, 4, 22, 16, 10, 18, 26, 34, 17, 28, 14, 20, 24, 28, 26, 34, 34, 46,
                 26, 36, 60, 80, 20, 26, 54, 32, 40, 32, 40, 50, 42, 56, 76, 84, 36, 46,
                 68, 32, 48, 52, 56, 64, 66, 54, 70, 92, 93, 120, 85], dtype=float)

# Train/test split
np.random.seed(42)
indices = np.random.permutation(len(speed))
train_idx, test_idx = indices[:35], indices[35:]
speed_train, dist_train = speed[train_idx], dist[train_idx]
speed_test, dist_test = speed[test_idx], dist[test_idx]

# Build polynomial features (degree 15)
degree = 15
def poly_features(x, deg):
    return np.vstack([x**i for i in range(deg+1)]).T

X_train = poly_features(speed_train, degree)
X_test = poly_features(speed_test, degree)

# Ridge regression for different lambda values
lambdas = [0, 1e-6, 1e-3, 1]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for ax, lam in zip(axes.flat, lambdas):
    # Ridge solution: (X^T X + lambda I)^{-1} X^T y
    I = np.eye(X_train.shape[1])
    I[0, 0] = 0  # Don't regularize bias
    w = np.linalg.solve(X_train.T @ X_train + lam * I, X_train.T @ dist_train)
    
    # Predictions
    pred_train = X_train @ w
    pred_test = X_test @ w
    mse_train = np.mean((dist_train - pred_train)**2)
    mse_test = np.mean((dist_test - pred_test)**2)
    
    # Plot
    ax.scatter(speed_train, dist_train, alpha=0.6, s=30, label='Train')
    ax.scatter(speed_test, dist_test, alpha=0.6, s=30, marker='s', label='Test')
    
    speed_grid = np.linspace(3, 26, 200)
    X_grid = poly_features(speed_grid, degree)
    pred_grid = X_grid @ w
    pred_grid = np.clip(pred_grid, -50, 200)
    ax.plot(speed_grid, pred_grid, 'k-', alpha=0.7)
    
    ax.set_xlim(3, 26)
    ax.set_ylim(-20, 150)
    ax.set_xlabel('Vitesse (mph)')
    ax.set_ylabel('Distance (ft)')
    ax.set_title(f'$\\lambda$ = {lam}: Train MSE={mse_train:.1f}, Test MSE={mse_test:.1f}')
    if lam == 0:
        ax.legend()

plt.tight_layout()
```

Sans régularisation ($\lambda = 0$), le polynôme de degré 15 oscille fortement. Avec une régularisation modérée ($\lambda = 10^{-3}$), les oscillations sont atténuées et l'erreur de test diminue. Avec une régularisation trop forte ($\lambda = 1$), le modèle devient trop contraint et sous-apprend.

### Interprétation bayésienne

La régularisation $\ell_2$ admet une interprétation probabiliste. Si nous plaçons un a priori gaussien sur les paramètres $p(w) = \mathcal{N}(w | 0, \sigma_w^2 I)$, l'estimateur du maximum a posteriori (MAP) est:

$$
\hat{w}_{\text{MAP}} = \arg\max_w \log p(\mathcal{D}|w) + \log p(w)
$$

En développant:

$$
\hat{w}_{\text{MAP}} = \arg\min_w \text{NLL}(w) + \frac{1}{2\sigma_w^2}\|w\|_2^2
$$

L'estimation MAP avec un a priori gaussien coïncide avec la régression ridge, avec $\lambda = 1/(2\sigma_w^2)$. Un a priori avec une petite variance (forte croyance que les paramètres sont proches de zéro) correspond à une grande valeur de $\lambda$.

### Choix de l'hyperparamètre

La valeur de $\lambda$ est un **hyperparamètre** qui doit être choisi avant l'entraînement. Ce choix se fait généralement par **validation**: on entraîne le modèle pour plusieurs valeurs de $\lambda$ et on retient celle qui minimise l'erreur sur un ensemble de validation.

## Généralisation

La différence entre le risque et le risque empirique est l'**écart de généralisation**:

$$
\text{Écart} = \mathcal{R}(f) - \hat{\mathcal{R}}(f; \mathcal{D}_{\text{train}})
$$

Un modèle qui minimise le risque empirique peut avoir un risque élevé si cet écart est grand. Ce phénomène est le **surapprentissage**: le modèle s'ajuste aux particularités de l'échantillon d'entraînement, y compris le bruit, plutôt qu'aux régularités sous-jacentes. L'erreur d'entraînement est faible, mais l'erreur sur de nouvelles données est élevée.

À l'inverse, un modèle trop simple peut avoir un risque empirique et un risque tous deux élevés. C'est le **sous-apprentissage**: le modèle n'a pas la capacité de capturer la structure des données.

### Extrapolation

Un cas particulier de mauvaise généralisation est l'**extrapolation**: prédire pour des entrées en dehors de la plage des données d'entraînement. Même un modèle bien ajusté peut échouer spectaculairement lorsqu'on lui demande de prédire au-delà de ce qu'il a vu.

Considérons des essais en soufflerie pour mesurer la portance d'une aile à différentes vitesses. Les tests sont effectués entre 20 et 60 m/s. L'ingénieur veut prédire la portance à 100 m/s.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Données de portance aérodynamique (simulées)
# Équation de portance: L = 0.5 * rho * v^2 * S * C_L
np.random.seed(42)
rho, S, C_L = 1.225, 20.0, 0.5  # densité air, surface aile, coeff. portance
v_train = np.linspace(20, 60, 8)
L_true_train = 0.5 * rho * v_train**2 * S * C_L
L_train = L_true_train + np.random.normal(0, 400, len(v_train))

# Fit quadratic (correct model) and higher-degree polynomial
coeffs_2 = np.polyfit(v_train, L_train, 2)
coeffs_5 = np.polyfit(v_train, L_train, 5)

# Extrapolation range
v_extrap = np.linspace(15, 110, 200)
L_true_extrap = 0.5 * rho * v_extrap**2 * S * C_L

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, coeffs, deg in zip(axes, [coeffs_2, coeffs_5], [2, 5]):
    ax.scatter(v_train, L_train, s=50, zorder=5, label='Observations')
    ax.plot(v_extrap, L_true_extrap, 'g-', alpha=0.5, label='Vraie relation')
    
    L_pred = np.polyval(coeffs, v_extrap)
    ax.plot(v_extrap, L_pred, 'k--', label=f'Polynôme degré {deg}')
    
    # Mark extrapolation region
    ax.axvline(60, color='gray', linestyle=':', alpha=0.5)
    ax.axvspan(60, 110, alpha=0.1, color='red')
    
    ax.set_xlabel('Vitesse (m/s)')
    ax.set_ylabel('Portance (N)')
    ax.set_title(f'Degré {deg}')
    ax.legend(loc='upper left')
    ax.set_ylim(-5000, 80000)
    ax.text(85, 5000, 'Extrapolation', ha='center', fontsize=10, color='red', alpha=0.7)

plt.tight_layout()
```

Le polynôme de degré 2 (qui correspond au vrai modèle physique $L \propto v^2$) extrapole correctement. Le polynôme de degré 5, bien qu'il ajuste aussi bien les données d'entraînement, diverge complètement en dehors de la plage observée.

En pratique, nous estimons le risque par le risque empirique sur un **ensemble de test** $\mathcal{D}_{\text{test}}$ disjoint de l'ensemble d'entraînement. Un troisième ensemble, l'**ensemble de validation**, sert à choisir parmi plusieurs modèles ou à régler des hyperparamètres. L'ensemble de test doit rester intact jusqu'à l'évaluation finale, pour fournir une estimation non biaisée.

Cette séparation est importante. Si nous utilisons l'ensemble de test pour faire des choix (quel modèle garder, quelle valeur d'hyperparamètre utiliser), l'estimation de performance sur ce même ensemble devient optimiste. Nous aurions alors besoin d'un quatrième ensemble pour obtenir une estimation fiable.

```{admonition} Mise en garde: la fuite d'information
:class: warning

Les outils modernes de génération de code peuvent produire des pipelines d'apprentissage machine complets en quelques minutes. Mais ces pipelines peuvent contenir des erreurs subtiles qui mènent à des résultats trop beaux pour être vrais.

Un exemple vécu: un praticien utilise un assistant de programmation pour construire un modèle prédictif. L'erreur d'entraînement passe de 0.20 à 0.01 en quelques itérations, un résultat spectaculaire. Mais en examinant le code de plus près, il découvre que le modèle utilise des caractéristiques qui ne seraient pas disponibles au moment du déploiement. Par exemple, dans un problème de prédiction temporelle, le modèle avait accès à des informations futures. Il s'agissait de données qui existent dans l'ensemble d'entraînement historique, mais qui n'existeront pas quand le modèle sera utilisé en production.

Ce phénomène s'appelle la **fuite d'information** (*data leakage*). Le modèle ne généralise pas: il triche. Les métriques d'entraînement sont excellentes, mais le modèle échouera en déploiement.

**Votre rôle** dans un monde où le code s'écrit facilement: auditer les pipelines, vérifier que les caractéristiques utilisées seront disponibles en production, et maintenir une séparation stricte entre les données d'entraînement et de test.
```

## Biais inductifs

Il n'existe pas de modèle universel qui fonctionne optimalement pour tous les problèmes. Ce résultat, connu sous le nom de **théorème du no free lunch**, affirme qu'un algorithme d'apprentissage qui performe bien sur une classe de problèmes performe nécessairement moins bien sur d'autres.

Tout modèle encode des **biais inductifs**: des hypothèses implicites ou explicites sur la structure du problème. La régression linéaire suppose que la relation entre entrées et sorties est linéaire. Les k plus proches voisins supposent que les points proches dans l'espace des entrées ont des sorties similaires. Les réseaux convolutifs supposent que les motifs locaux dans une image sont informatifs indépendamment de leur position.

Ces hypothèses sont nécessaires pour que l'apprentissage soit possible. Sans elles, nous n'aurions aucune raison de croire que la performance sur l'échantillon d'entraînement prédit la performance sur de nouvelles données. Le choix du modèle et de ses hypothèses est une décision que l'algorithme ne peut pas prendre seul; elle requiert une connaissance du domaine.

## Résumé

Ce chapitre a établi le cadre formel de l'apprentissage supervisé. Nous avons défini le risque comme la mesure de performance que nous voulons optimiser, et le risque empirique comme son approximation calculable. Le principe de minimisation du risque empirique consiste à choisir le modèle qui minimise cette approximation.

L'estimation par maximum de vraisemblance offre une perspective complémentaire, fondée sur l'inférence statistique. Les deux approches coïncident pour la perte logarithmique. L'interprétation en termes de divergence KL montre que le maximum de vraisemblance trouve le modèle le plus proche de la distribution empirique.

La question centrale que ce chapitre laisse ouverte est celle de la généralisation: quand le risque empirique est-il un bon indicateur du vrai risque? Cette question dépend de la taille de l'échantillon, de la complexité du modèle, et de la distribution des données. Le chapitre suivant développe les outils pour y répondre.

## Exercices

````{admonition} Exercice 1: Usure d'outil
:class: hint dropdown

Un machiniste mesure l'usure d'un outil de coupe (en mm) à différents temps de coupe (en minutes):

```python
import numpy as np

# Données d'usure d'outil (simulées selon une loi de puissance avec bruit)
time = np.array([2.0, 5.1, 8.2, 11.3, 14.4, 17.6, 20.7, 23.8, 26.9, 30.0])
wear = np.array([0.08, 0.14, 0.17, 0.21, 0.22, 0.25, 0.27, 0.28, 0.31, 0.32])
```

L'outil doit être remplacé lorsque l'usure atteint 0.4 mm.

1. **Visualisation.** Tracez les données. Quelle forme de relation observez-vous?

2. **Ajustement.** Ajustez un modèle linéaire $w(t) = at + b$ et un modèle en loi de puissance $w(t) = at^b$ aux données. Pour le second modèle, utilisez une transformation logarithmique: $\log w = \log a + b \log t$.

3. **Comparaison.** Calculez le MSE de chaque modèle sur les données. Lequel ajuste mieux?

4. **Prédiction.** Selon chaque modèle, à quel moment l'usure atteindra-t-elle 0.4 mm? Les deux modèles donnent-ils la même réponse?

5. **Extrapolation.** Si vous n'aviez mesuré que jusqu'à $t = 15$ min, vos prédictions changeraient-elles? Discutez du risque d'extrapolation.
````

```{admonition} Solution Exercice 1
:class: dropdown

1. **Visualisation.** Les données montrent une relation non linéaire, concave: l'usure augmente rapidement au début puis ralentit. Cela suggère une loi de puissance avec exposant $b < 1$.

2. **Ajustement.**
   - Modèle linéaire: `coeffs = np.polyfit(time, wear, 1)` donne $a \approx 0.006$, $b \approx 0.05$.
   - Loi de puissance: en posant $\log w = \log a + b \log t$, on ajuste une droite dans l'espace log-log: `coeffs = np.polyfit(np.log(time), np.log(wear), 1)`. On obtient $b \approx 0.5$ et $a = \exp(\text{intercept}) \approx 0.05$.

3. **Comparaison.** Le MSE du modèle linéaire est typiquement plus élevé car il ne capture pas la courbure. Le modèle en loi de puissance ajuste mieux les données.

4. **Prédiction.** Pour trouver $t$ tel que $w(t) = 0.4$:
   - Linéaire: $t = (0.4 - b) / a$
   - Puissance: $t = (0.4 / a)^{1/b}$
   
   Les réponses diffèrent significativement car les modèles extrapolent différemment.

5. **Extrapolation.** Avec moins de données, les estimations des paramètres changent, et les prédictions au-delà des données observées deviennent plus incertaines. L'extrapolation est risquée car le comportement futur peut ne pas suivre le modèle ajusté sur les données passées.
```

````{admonition} Exercice 2: Risque et risque empirique
:class: hint dropdown

Soit un problème de classification binaire avec la perte 0-1. Un classificateur $f$ fait 3 erreurs sur 20 exemples d'entraînement.

1. Quel est le risque empirique de $f$ sur l'ensemble d'entraînement?

2. Peut-on en déduire le vrai risque $\mathcal{R}(f)$? Pourquoi ou pourquoi pas?

3. Si nous avions 1000 exemples de test et que $f$ fait 45 erreurs, quelle serait notre meilleure estimation du vrai risque?
````

```{admonition} Solution Exercice 2
:class: dropdown

1. **Risque empirique:** $\hat{\mathcal{R}}(f) = \frac{3}{20} = 0.15$ (soit 15% d'erreur).

2. **Non, on ne peut pas en déduire le vrai risque.** Le risque empirique sur l'entraînement est une estimation biaisée du vrai risque car:
   - Le modèle $f$ a été choisi/optimisé pour bien performer sur ces mêmes données
   - Il y a surapprentissage potentiel: $f$ peut avoir mémorisé des particularités de l'entraînement qui ne généralisent pas
   - Le risque empirique sur l'entraînement sous-estime généralement le vrai risque

3. **Estimation sur le test:** $\hat{\mathcal{R}}(f) = \frac{45}{1000} = 0.045$ (soit 4.5% d'erreur). Cette estimation est plus fiable car:
   - Les données de test n'ont pas été utilisées pour construire $f$
   - Avec 1000 exemples, l'estimation est plus précise (écart-type $\approx \sqrt{0.045 \times 0.955 / 1000} \approx 0.007$)
```

````{admonition} Exercice 3: Maximum de vraisemblance
:class: hint dropdown

Soit $\{y_1, \ldots, y_N\}$ un échantillon i.i.d. d'une distribution exponentielle de paramètre $\lambda > 0$:

$$
p(y | \lambda) = \lambda e^{-\lambda y}, \quad y \geq 0
$$

1. Écrivez la vraisemblance $\mathcal{L}(\lambda)$ et la log-vraisemblance $\log \mathcal{L}(\lambda)$.

2. Dérivez l'estimateur du maximum de vraisemblance $\hat{\lambda}_{\text{MLE}}$.

3. Si les observations sont $y = \{0.5, 1.2, 0.8, 2.1, 0.3\}$, calculez $\hat{\lambda}_{\text{MLE}}$.
````

```{admonition} Solution Exercice 3
:class: dropdown

1. **Vraisemblance et log-vraisemblance:**

   $$\mathcal{L}(\lambda) = \prod_{i=1}^N \lambda e^{-\lambda y_i} = \lambda^N \exp\left(-\lambda \sum_{i=1}^N y_i\right)$$
   
   $$\log \mathcal{L}(\lambda) = N \log \lambda - \lambda \sum_{i=1}^N y_i$$

2. **Dérivation du MLE:**

   On dérive par rapport à $\lambda$ et on égale à zéro:
   
   $$\frac{d}{d\lambda} \log \mathcal{L}(\lambda) = \frac{N}{\lambda} - \sum_{i=1}^N y_i = 0$$
   
   D'où:
   
   $$\hat{\lambda}_{\text{MLE}} = \frac{N}{\sum_{i=1}^N y_i} = \frac{1}{\bar{y}}$$
   
   L'estimateur MLE est l'inverse de la moyenne empirique.

3. **Application numérique:**

   $\bar{y} = \frac{0.5 + 1.2 + 0.8 + 2.1 + 0.3}{5} = \frac{4.9}{5} = 0.98$
   
   $\hat{\lambda}_{\text{MLE}} = \frac{1}{0.98} \approx 1.02$
```

````{admonition} Exercice 4: Fonctions de perte
:class: hint dropdown

Soit $y = 1$ (classe positive) et un score $s = f(x) = 2$.

1. Calculez la perte 0-1, la perte logistique, et la perte à charnière.

2. Répétez pour $s = -0.5$ (prédiction incorrecte).

3. Tracez les trois fonctions de perte en fonction de $y \cdot s$ pour $y \cdot s \in [-3, 3]$. Vérifiez que les pertes de substitution majorent la perte 0-1.
````

````{admonition} Solution Exercice 4
:class: dropdown

1. **Pour $y = 1$ et $s = 2$** (prédiction correcte, marge $y \cdot s = 2$):

   - Perte 0-1: $\mathbb{1}[\text{sign}(s) \neq y] = \mathbb{1}[1 \neq 1] = 0$
   - Perte logistique: $\log(1 + e^{-y \cdot s}) = \log(1 + e^{-2}) \approx \log(1.135) \approx 0.127$
   - Perte à charnière: $\max(0, 1 - y \cdot s) = \max(0, 1 - 2) = \max(0, -1) = 0$

2. **Pour $y = 1$ et $s = -0.5$** (prédiction incorrecte, marge $y \cdot s = -0.5$):

   - Perte 0-1: $\mathbb{1}[\text{sign}(-0.5) \neq 1] = \mathbb{1}[-1 \neq 1] = 1$
   - Perte logistique: $\log(1 + e^{-(-0.5)}) = \log(1 + e^{0.5}) \approx \log(2.649) \approx 0.974$
   - Perte à charnière: $\max(0, 1 - (-0.5)) = \max(0, 1.5) = 1.5$

3. **Vérification graphique:**

   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   
   margin = np.linspace(-3, 3, 100)
   loss_01 = (margin < 0).astype(float)
   loss_log = np.log(1 + np.exp(-margin))
   loss_hinge = np.maximum(0, 1 - margin)
   
   plt.plot(margin, loss_01, label='0-1')
   plt.plot(margin, loss_log, label='Logistique')
   plt.plot(margin, loss_hinge, label='Charnière')
   plt.legend()
   ```
   
   On vérifie que pour tout $m$: $\ell_{\text{log}}(m) \geq \ell_{0-1}(m)$ et $\ell_{\text{hinge}}(m) \geq \ell_{0-1}(m)$.
````