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
- Expliquer pourquoi le risque est inaccessible et nécessite une approximation
- Formuler le principe de minimisation du risque empirique
- Dériver le prédicteur de Bayes optimal pour différentes fonctions de perte
- Expliquer pourquoi le risque de Bayes est la limite théorique de performance
- Formuler le principe du maximum de vraisemblance et son lien avec les fonctions de perte
```

Le chapitre précédent a présenté les méthodes non paramétriques, comme les k plus proches voisins, qui conservent les données d'entraînement et les consultent au moment de la prédiction. Cette approche est intuitive, mais elle a un coût: les données doivent rester en mémoire, et chaque prédiction requiert de parcourir l'ensemble d'entraînement. Ce chapitre développe une approche différente: plutôt que de garder les données, nous cherchons à les *résumer* dans un ensemble fixe de **paramètres**. L'apprentissage devient alors un problème d'**optimisation**.

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

# Configuration pour des figures haute résolution
%config InlineBackend.figure_format = 'retina'

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
ax.set_title(f'EQM = {np.mean(residuals**2):.1f}')

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

- **Modèle stochastique**: on suppose plutôt que $Y$ est une variable aléatoire conditionnellement à $X=x$, via une distribution $p(y\mid x)$. La "bonne" prédiction devient alors une question de moyenne/quantile/probabilité, selon la perte.

Dans la pratique, ces deux points de vue sont surtout des **langages** différents. Le formalisme probabiliste est souvent plus commode: il permet d'exprimer simplement "la performance moyenne sur des données futures" via une espérance. Ce chapitre adopte ce langage parce qu'il rend la généralisation et les garanties mathématiques plus propres, sans changer l'objectif final: produire une règle de prédiction utile.
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

La nature possède une distribution $p(\mathbf{x}, y)$ qui génère les données, mais nous ne la connaissons pas. Nous n'observons qu'un échantillon fini $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ tiré de cette distribution. 

L'ensemble $\mathcal{D}$ est une **variable aléatoire**: si nous répétions l'expérience de collecte de données, nous obtiendrions un échantillon différent. Cette perspective, adoptée notamment dans {cite:t}`murphy2022probabilistic`, rappelle que nos conclusions dépendent de l'échantillon particulier que nous avons observé. C'est comme si nous regardions quelques gouttes d'eau d'un océan: nous pouvons analyser ces gouttes, mais un autre prélèvement donnerait des gouttes différentes.

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

#### Obstacle 2: L'intégration est computationnellement difficile

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
| Quadrature (règles trapézoïdales, etc.) | $O(M^d)$ | Connaître $p(\mathbf{x},y)$ exactement |
| Monte Carlo | $O(N)$ | Avoir des échantillons de $p(\mathbf{x},y)$ |

La complexité de Monte Carlo est **indépendante de la dimension** $d$. Elle ne dépend que du nombre d'échantillons $N$. C'est cette propriété qui rend l'apprentissage possible en haute dimension. De plus, nous n'avons pas besoin de connaître la valeur numérique de $p(\mathbf{x},y)$: nous avons seulement besoin de pouvoir tirer des échantillons de cette distribution. C'est exactement ce que nos données d'entraînement nous fournissent.

Le risque empirique remplace l'intégrale sur la distribution inconnue par une moyenne sur l'échantillon fini que nous possédons:

$$
\hat{\mathcal{R}}(f, \mathcal{D}) = \frac{1}{N} \sum_{i=1}^{N} \ell(y_i, f(\mathbf{x}_i))
$$

Cette formule est directe à évaluer: nous parcourons nos $N$ exemples d'entraînement, calculons la perte pour chacun, et faisons la moyenne.

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
plt.scatter(speed_train, dist_train, alpha=0.7, label=f'Entraînement (EQM={mse_train:.1f})')
plt.scatter(speed_test, dist_test, alpha=0.7, marker='s', label=f'Test (EQM={mse_test:.1f})')

speed_grid = np.linspace(4, 28, 100)
plt.plot(speed_grid, np.polyval(coeffs, speed_grid), 'k--', alpha=0.6, label='Fonction ajustée')

plt.axvline(x=20, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('Vitesse (mph)')
plt.ylabel('Distance (ft)')
plt.legend()
plt.tight_layout()
```

Dans cet exemple, l'EQM (*MSE*, erreur quadratique moyenne) sur l'ensemble de test est plus élevé que sur l'ensemble d'entraînement. Cet écart est typique: la fonction a été optimisée pour les données d'entraînement, pas pour les données de test.

Sous l'hypothèse que les exemples $(\mathbf{x}_i, y_i)$ sont tirés indépendamment et identiquement distribués (i.i.d.) selon $p(\mathbf{x}, y)$, le risque empirique est un estimateur non biaisé du vrai risque: $\mathbb{E}[\hat{\mathcal{R}}(f, \mathcal{D})] = \mathcal{R}(f)$. Cela signifie qu'en moyenne, sur tous les échantillons possibles, le risque empirique est égal au vrai risque.

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

L'écart entre ces deux quantités est au cœur de l'apprentissage automatique. Un modèle peut avoir un risque empirique très faible (il performe bien sur les données d'entraînement) tout en ayant un risque élevé (il performe mal sur de nouvelles données). C'est le problème du **surapprentissage**, que nous explorerons dans le chapitre sur la [généralisation](ch4_generalization.md).

## Minimisation du risque empirique

Nous avons maintenant les éléments pour formuler l'apprentissage comme un problème d'optimisation. Nous cherchons la fonction $f$ dans une classe $\mathcal{F}$ qui minimise le risque:

$$
f^\star = \arg\min_{f \in \mathcal{F}} \mathcal{R}(f)
$$

Puisque le risque est inaccessible, nous le remplaçons par le risque empirique:

$$
\hat{f} = \arg\min_{f \in \mathcal{F}} \hat{\mathcal{R}}(f, \mathcal{D})
$$

Ce principe est la **minimisation du risque empirique** (MRE): choisir la fonction qui fait le moins d'erreurs sur les données d'entraînement, en espérant que cette performance se transfère aux nouvelles données.

La classe $\mathcal{F}$ est notre **classe d'hypothèses**. Elle représente l'ensemble des fonctions que nous sommes prêts à considérer. Le choix de $\mathcal{F}$ encode nos hypothèses sur la forme de la relation entre entrées et sorties. Par exemple, si nous choisissons la classe des fonctions linéaires, nous supposons que la relation est (approximativement) linéaire.

Mais comment résoudre ce problème d'optimisation concrètement? Et quand pouvons-nous espérer que le minimiseur du risque empirique aura un faible risque? Avant d'aborder ces questions, établissons d'abord la cible théorique: le meilleur prédicteur possible si nous connaissions la vraie distribution.

## Le prédicteur optimal et le risque de Bayes

Si nous connaissions la **vraie** distribution conjointe $p(\mathbf{x}, y)$, quelle fonction $f$ minimiserait le risque?

$$
\mathcal{R}(f) = \mathbb{E}_{p(\mathbf{x}, y)}[\ell(y, f(\mathbf{x}))] = \int \int \ell(y, f(\mathbf{x})) \, p(y | \mathbf{x}) \, p(\mathbf{x}) \, dy \, d\mathbf{x}
$$

Puisque $p(\mathbf{x})$ est toujours positif, minimiser cette intégrale revient à minimiser, pour chaque $\mathbf{x}$, l'espérance conditionnelle de la perte. Le **prédicteur de Bayes optimal** est donc:

$$
f^*(\mathbf{x}) = \arg\min_{\hat{y}} \mathbb{E}_{p(y|\mathbf{x})}[\ell(y, \hat{y})]
$$

La réponse dépend de la fonction de perte choisie. Chaque perte définit son propre prédicteur optimal.

### Perte quadratique: la moyenne conditionnelle

Pour la **perte quadratique** $\ell(y, \hat{y}) = (y - \hat{y})^2$, développons l'espérance:

$$
\mathbb{E}[(y - \hat{y})^2 | \mathbf{x}] = \mathbb{E}[y^2 | \mathbf{x}] - 2\hat{y}\mathbb{E}[y | \mathbf{x}] + \hat{y}^2
$$

C'est une fonction quadratique en $\hat{y}$. En dérivant et en posant la dérivée égale à zéro:

$$
\frac{\partial}{\partial \hat{y}} \mathbb{E}[(y - \hat{y})^2 | \mathbf{x}] = -2\mathbb{E}[y | \mathbf{x}] + 2\hat{y} = 0 \quad \Rightarrow \quad \hat{y}^* = \mathbb{E}[y | \mathbf{x}]
$$

Le prédicteur optimal pour la perte quadratique est la **moyenne conditionnelle**. Cette observation justifie l'utilisation de la perte quadratique en régression: si notre objectif est de prédire la valeur moyenne de $y$ sachant $\mathbf{x}$, c'est la bonne perte à utiliser.

### Perte 0-1: le mode conditionnel

Pour la **perte 0-1** en classification $\ell(y, \hat{y}) = \mathbf{1}[y \neq \hat{y}]$, l'espérance est:

$$
\mathbb{E}[\mathbf{1}[y \neq \hat{y}] | \mathbf{x}] = P(y \neq \hat{y} | \mathbf{x}) = 1 - P(y = \hat{y} | \mathbf{x})
$$

Minimiser cette quantité revient à maximiser $P(y = \hat{y} | \mathbf{x})$, donc à choisir la classe la plus probable:

$$
\hat{y}^* = \arg\max_c \, p(y = c | \mathbf{x})
$$

Le prédicteur optimal pour la perte 0-1 est le **mode conditionnel**: la classe ayant la plus grande probabilité sachant $\mathbf{x}$.

### Le risque de Bayes: la limite théorique

Le risque du prédicteur optimal s'appelle le **risque de Bayes**:

$$
\mathcal{R}^* = \mathcal{R}(f^*) = \mathbb{E}_{p(\mathbf{x})}[\min_{\hat{y}} \mathbb{E}_{p(y|\mathbf{x})}[\ell(y, \hat{y})]]
$$

Ce risque est un **repère théorique**: aucun algorithme ne peut faire mieux, car il suppose l'accès à la vraie distribution. La différence entre le risque d'un prédicteur appris $\hat{f}$ et ce risque de Bayes mesure ce que nous perdons en ne connaissant pas la vraie distribution:

$$
\mathcal{R}(\hat{f}) - \mathcal{R}^* \geq 0
$$

Cette différence, appelée **excès de risque**, se décompose en deux parties que nous étudierons dans le chapitre sur la [généralisation](ch4_generalization.md): l'erreur d'approximation (notre classe $\mathcal{F}$ ne contient peut-être pas $f^*$) et l'erreur d'estimation (nous n'avons qu'un échantillon fini pour choisir dans $\mathcal{F}$).

## Le modèle probabiliste et la vraisemblance

Nous avons vu comment minimiser le risque empirique pour une fonction de perte donnée. Mais d'où vient le choix de la perte? Pourquoi la perte quadratique pour la régression et non une autre? Cette section introduit une perspective qui justifie ces choix de manière principielle.

### Du prédicteur au modèle probabiliste

Jusqu'ici, notre modèle produit une prédiction déterministe $\hat{y} = f(\mathbf{x}; \boldsymbol{\theta})$. Une vue plus riche est de modéliser une **distribution de probabilité** sur les sorties possibles:

$$
p(y | \mathbf{x}; \boldsymbol{\theta})
$$

Ce modèle probabiliste répond à la question: pour une entrée $\mathbf{x}$ et des paramètres $\boldsymbol{\theta}$, quelle est la probabilité (ou densité) de chaque valeur possible de $y$?

Par exemple, en régression, nous pourrions supposer que $y$ suit une gaussienne centrée sur notre prédiction:

$$
p(y | \mathbf{x}; \boldsymbol{\theta}) = \mathcal{N}(y \,|\, f(\mathbf{x}; \boldsymbol{\theta}), \sigma^2)
$$

En classification binaire, nous pourrions supposer que $y$ suit une distribution de Bernoulli avec une probabilité qui dépend de $\mathbf{x}$:

$$
p(y | \mathbf{x}; \boldsymbol{\theta}) = \text{Ber}(y \,|\, \mu(\mathbf{x}; \boldsymbol{\theta}))
$$

Cette perspective probabiliste unifie régression et classification dans un même cadre.

### La vraisemblance et l'hypothèse i.i.d.

Comment évaluer si des paramètres $\boldsymbol{\theta}$ sont bons? Une idée naturelle: les bons paramètres devraient rendre nos observations **probables**. Si le modèle assigne une probabilité élevée aux données que nous avons effectivement observées, c'est un signe qu'il capture bien le phénomène sous-jacent.

Pour un seul exemple $(\mathbf{x}_1, y_1)$, nous évaluons $p(y_1 | \mathbf{x}_1; \boldsymbol{\theta})$. Mais nous avons $N$ exemples. Comment combiner leurs probabilités?

C'est ici qu'intervient l'**hypothèse i.i.d.** (indépendants et identiquement distribués): nous supposons que les exemples sont tirés indépendamment de la même distribution. Sous cette hypothèse fondamentale, la probabilité conjointe se factorise en un **produit**:

$$
p(y_1, \ldots, y_N | \mathbf{x}_1, \ldots, \mathbf{x}_N; \boldsymbol{\theta}) = \prod_{i=1}^N p(y_i | \mathbf{x}_i; \boldsymbol{\theta})
$$

Cette quantité, vue comme fonction de $\boldsymbol{\theta}$, s'appelle la **vraisemblance**:

$$
\mathcal{L}(\boldsymbol{\theta}; \mathcal{D}) = \prod_{i=1}^N p(y_i | \mathbf{x}_i; \boldsymbol{\theta})
$$

Elle répond à la question: pour ce choix de paramètres, quelle est la probabilité d'avoir observé exactement ces données?

### Le maximum de vraisemblance

Le principe du **maximum de vraisemblance** (EMV, ou MLE en anglais) consiste à choisir les paramètres qui maximisent cette probabilité:

$$
\hat{\boldsymbol{\theta}}_{\text{EMV}} = \arg\max_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}; \mathcal{D}) = \arg\max_{\boldsymbol{\theta}} \prod_{i=1}^N p(y_i | \mathbf{x}_i; \boldsymbol{\theta})
$$

En pratique, nous travaillons avec le logarithme. Comme le logarithme est une fonction croissante, maximiser la vraisemblance équivaut à maximiser la log-vraisemblance, ou de manière équivalente, à **minimiser la log-vraisemblance négative** (LVN):

$$
\text{LVN}(\boldsymbol{\theta}) = -\log \mathcal{L}(\boldsymbol{\theta}) = -\sum_{i=1}^N \log p(y_i | \mathbf{x}_i; \boldsymbol{\theta})
$$

Le logarithme transforme le produit en somme, ce qui est plus stable numériquement et plus facile à optimiser.

### Lien avec les fonctions de perte

Voici le résultat clé: **la forme de la LVN dépend du modèle probabiliste choisi**, et cette forme définit une fonction de perte naturelle.

- **Modèle gaussien** (régression): Si $p(y|\mathbf{x}; \boldsymbol{\theta}) = \mathcal{N}(y | f(\mathbf{x}; \boldsymbol{\theta}), \sigma^2)$, alors la LVN est proportionnelle à la **somme des carrés des résidus**. Le maximum de vraisemblance donne exactement les moindres carrés.

- **Modèle de Bernoulli** (classification binaire): Si $p(y|\mathbf{x}; \boldsymbol{\theta}) = \mu^y (1-\mu)^{1-y}$ avec $\mu = \sigma(\boldsymbol{\theta}^\top \mathbf{x})$, alors la LVN est l'**entropie croisée binaire**.

Les chapitres suivants développent ces connexions en détail: le [chapitre 2](ch2_linear_regression.md) pour la régression et le [chapitre 3](ch3_classification.md) pour la classification. Le [chapitre 5](ch5_probabilistic.md) étend ce cadre avec l'inférence bayésienne et le maximum a posteriori.

Le chapitre suivant applique ce cadre au cas concret de la **régression linéaire**, en dérivant des solutions analytiques et en explorant les outils pour résoudre le problème d'optimisation.

## Résumé

Ce chapitre a posé les bases formelles de l'apprentissage supervisé:

- L'**apprentissage supervisé** cherche une fonction $f$ qui prédit $y$ à partir de $\mathbf{x}$, en distinguant **régression** (sortie continue) et **classification** (sortie discrète).

- La **fonction de perte** $\ell(y, \hat{y})$ quantifie l'erreur d'une prédiction. Les choix courants sont la perte quadratique (régression) et la perte 0-1 (classification).

- Le **risque** $\mathcal{R}(f) = \mathbb{E}[\ell(Y, f(\mathbf{X}))]$ mesure la performance attendue sur de nouvelles données. C'est ce que nous voulons minimiser.

- Le risque est **inaccessible** car la distribution $p(\mathbf{x}, y)$ est inconnue et l'intégration est computationnellement difficile en haute dimension.

- Le **risque empirique** $\hat{\mathcal{R}}(f, \mathcal{D})$ est une approximation calculable, qui converge vers le vrai risque quand $N \to \infty$.

- La **minimisation du risque empirique** (MRE) est le principe fondamental: choisir $\hat{f} = \arg\min_{f \in \mathcal{F}} \hat{\mathcal{R}}(f, \mathcal{D})$.

- Le **prédicteur de Bayes optimal** $f^*$ minimise le risque si la vraie distribution est connue. Pour la perte quadratique, c'est la moyenne conditionnelle $\mathbb{E}[y|\mathbf{x}]$; pour la perte 0-1, c'est le mode conditionnel. Le **risque de Bayes** $\mathcal{R}^*$ est la limite théorique de performance.

- Le **maximum de vraisemblance** (EMV) justifie le choix des fonctions de perte de manière principielle: sous l'hypothèse i.i.d., nous choisissons les paramètres qui rendent les données observées les plus probables. Le modèle probabiliste (gaussien, Bernoulli, etc.) détermine la forme de la perte.

Le chapitre suivant applique ce cadre au cas concret de la **régression linéaire**, en dérivant des solutions analytiques et en explorant les outils pour résoudre le problème d'optimisation.

## Exercices

````{admonition} Exercice 1: Usure d'outil ★
:class: hint dropdown

Un machiniste mesure l'usure d'un outil de coupe (en mm) à différents temps de coupe (en minutes):

```python
import numpy as np

# Données d'usure d'outil (simulées selon une loi de puissance avec bruit)
time = np.array([2.0, 5.1, 8.2, 11.3, 14.4, 17.6, 20.7, 23.8, 26.9, 30.0])
wear = np.array([0.08, 0.14, 0.17, 0.21, 0.22, 0.25, 0.27, 0.28, 0.31, 0.32])
```

L'outil doit être remplacé lorsque l'usure atteint 0,4 mm.

1. **Visualisation.** Tracez les données. Quelle forme de relation observez-vous?

2. **Ajustement.** Ajustez un modèle linéaire $w(t) = at + b$ et un modèle en loi de puissance $w(t) = at^b$ aux données. Pour le second modèle, utilisez une transformation logarithmique: $\log w = \log a + b \log t$.

3. **Comparaison.** Calculez l'EQM de chaque modèle sur les données. Lequel ajuste mieux?

4. **Prédiction.** Selon chaque modèle, à quel moment l'usure atteindra-t-elle 0,4 mm? Les deux modèles donnent-ils la même réponse?

5. **Extrapolation.** Si vous n'aviez mesuré que jusqu'à $t = 15$ min, vos prédictions changeraient-elles? Discutez du risque d'extrapolation.
````

```{admonition} Solution Exercice 1
:class: dropdown

1. **Visualisation.** Les données montrent une relation non linéaire, concave: l'usure augmente rapidement au début puis ralentit. Cela suggère une loi de puissance avec exposant $b < 1$.

2. **Ajustement.**
   - Modèle linéaire: `coeffs = np.polyfit(time, wear, 1)` donne $a \approx 0.006$, $b \approx 0.05$.
   - Loi de puissance: en posant $\log w = \log a + b \log t$, on ajuste une droite dans l'espace log-log: `coeffs = np.polyfit(np.log(time), np.log(wear), 1)`. On obtient $b \approx 0.5$ et $a = \exp(\text{intercept}) \approx 0.05$.

3. **Comparaison.** L'EQM du modèle linéaire est typiquement plus élevé car il ne capture pas la courbure. Le modèle en loi de puissance ajuste mieux les données.

4. **Prédiction.** Pour trouver $t$ tel que $w(t) = 0.4$:
   - Linéaire: $t = (0.4 - b) / a$
   - Puissance: $t = (0.4 / a)^{1/b}$
   
   Les réponses diffèrent significativement car les modèles extrapolent différemment.

5. **Extrapolation.** Avec moins de données, les estimations des paramètres changent, et les prédictions au-delà des données observées deviennent plus incertaines. L'extrapolation est risquée car le comportement futur peut ne pas suivre le modèle ajusté sur les données passées.
```

````{admonition} Exercice 2: Risque et risque empirique ★
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

3. **Estimation sur le test:** $\hat{\mathcal{R}}(f) = \frac{45}{1000} = 0.045$ (soit 4,5% d'erreur). Cette estimation est plus fiable car:
   - Les données de test n'ont pas été utilisées pour construire $f$
   - Avec 1000 exemples, l'estimation est plus précise (écart-type $\approx \sqrt{0.045 \times 0.955 / 1000} \approx 0.007$)
```

````{admonition} Exercice 3: Fonctions de perte ★
:class: hint dropdown

Soit $y = 1$ (classe positive) et un score $s = f(x) = 2$.

1. Calculez la perte 0-1, la perte logistique, et la perte à charnière.

2. Répétez pour $s = -0.5$ (prédiction incorrecte).

3. Tracez les trois fonctions de perte en fonction de $y \cdot s$ pour $y \cdot s \in [-3, 3]$. Vérifiez que les pertes de substitution majorent la perte 0-1.
````

````{admonition} Solution Exercice 3
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

````{admonition} Exercice 4: Prédicteur de Bayes optimal (perte quadratique) ★★
:class: hint dropdown

Le prédicteur de Bayes optimal minimise le risque pour une perte donnée, en supposant que la vraie distribution $p(y|x)$ est connue.

Considérons la distribution conditionnelle suivante pour un $x$ donné:

$$
p(y|x) = 0.3 \cdot \mathcal{N}(y | 1, 0.5^2) + 0.7 \cdot \mathcal{N}(y | 4, 1^2)
$$

C'est un mélange de deux gaussiennes.

1. Tracez cette distribution $p(y|x)$.

2. Calculez l'espérance $\mathbb{E}[y|x]$ (utiliser la linéarité de l'espérance).

3. Pour la perte quadratique, le prédicteur optimal est la moyenne conditionnelle. Quelle est donc la prédiction optimale $\hat{y}^*$?

4. Calculez le risque de Bayes (l'erreur minimale atteignable): $\mathcal{R}^* = \mathbb{E}[(y - \hat{y}^*)^2 | x]$.

5. Si vous prédisiez le mode (la valeur la plus probable) au lieu de la moyenne, quelle serait votre prédiction? Quel serait le risque?
````

````{admonition} Solution Exercice 4
:class: dropdown

1. **Distribution:**

   ```python
   y = np.linspace(-2, 8, 200)
   p = 0.3 * scipy.stats.norm.pdf(y, 1, 0.5) + 0.7 * scipy.stats.norm.pdf(y, 4, 1)
   plt.plot(y, p)
   plt.xlabel('y')
   plt.ylabel('p(y|x)')
   ```
   
   La distribution est bimodale avec un petit pic à $y = 1$ et un grand pic à $y = 4$.

2. **Espérance:**

   Par linéarité: $\mathbb{E}[y|x] = 0.3 \times 1 + 0.7 \times 4 = 0.3 + 2.8 = 3.1$

3. **Prédiction optimale:**

   Pour la perte quadratique, $\hat{y}^* = \mathbb{E}[y|x] = 3.1$

4. **Risque de Bayes:**

   $$
   \mathcal{R}^* = \mathbb{E}[(y - 3.1)^2 | x] = \text{Var}(y|x)
   $$
   
   Pour un mélange: $\text{Var}(y) = \mathbb{E}[\text{Var}(y|k)] + \text{Var}(\mathbb{E}[y|k])$
   
   où $k$ est la composante.
   
   - $\mathbb{E}[\text{Var}] = 0.3 \times 0.25 + 0.7 \times 1 = 0.075 + 0.7 = 0.775$
   - $\text{Var}(\mathbb{E}) = 0.3 \times (1 - 3.1)^2 + 0.7 \times (4 - 3.1)^2 = 0.3 \times 4.41 + 0.7 \times 0.81 = 1.323 + 0.567 = 1.89$
   - $\mathcal{R}^* = 0.775 + 1.89 = 2.665$

5. **Prédiction par le mode:**

   Le mode est le pic le plus haut, soit $y = 4$ (puisque le poids 0.7 > 0.3).
   
   Risque: $\mathbb{E}[(y - 4)^2 | x] = 0.3 \times [(1-4)^2 + 0.25] + 0.7 \times [(4-4)^2 + 1]$
   $= 0.3 \times 9.25 + 0.7 \times 1 = 2.775 + 0.7 = 3.475$
   
   Le mode donne un risque plus élevé (3.475 > 2.665) pour la perte quadratique.
````

````{admonition} Exercice 5: Prédicteur de Bayes optimal (perte 0-1) ★★
:class: hint dropdown

Pour la classification avec perte 0-1, le prédicteur de Bayes optimal est le **mode conditionnel** (la classe la plus probable).

Considérons un problème de classification à 3 classes avec les probabilités conditionnelles suivantes pour un $\mathbf{x}$ donné:

$$
p(y = 0 | \mathbf{x}) = 0.25, \quad p(y = 1 | \mathbf{x}) = 0.45, \quad p(y = 2 | \mathbf{x}) = 0.30
$$

1. Quel est le prédicteur de Bayes optimal $\hat{y}^*$?

2. Calculez le risque de Bayes $\mathcal{R}^* = P(\hat{y}^* \neq y | \mathbf{x})$.

3. Supposons qu'un classificateur prédit la classe 0 pour cet $\mathbf{x}$. Quel est son risque?

4. **Situation asymétrique**: Supposons que se tromper sur la classe 1 (maladie) coûte 10 fois plus cher que les autres erreurs. Définissez une matrice de coût et trouvez la prédiction optimale.

5. Montrez que pour la perte 0-1, aucun classificateur ne peut avoir un risque inférieur au risque de Bayes.
````

```{admonition} Solution Exercice 5
:class: dropdown

1. **Prédicteur optimal:**

   Pour la perte 0-1, $\hat{y}^* = \arg\max_c p(y = c | \mathbf{x})$.
   
   Ici, la classe 1 a la probabilité maximale (0.45), donc $\hat{y}^* = 1$.

2. **Risque de Bayes:**

   $$
   \mathcal{R}^* = P(y \neq 1 | \mathbf{x}) = 1 - P(y = 1 | \mathbf{x}) = 1 - 0.45 = 0.55
   $$
   
   Même le meilleur classificateur possible se trompe 55% du temps pour ce $\mathbf{x}$.

3. **Risque du classificateur sous-optimal:**

   Si on prédit la classe 0:
   $$
   P(y \neq 0 | \mathbf{x}) = 1 - 0.25 = 0.75
   $$
   
   Ce classificateur a un risque plus élevé (0.75 > 0.55).

4. **Coûts asymétriques:**

   Matrice de coût $C_{ij}$ = coût de prédire $i$ quand la vraie classe est $j$:
   
   |  | $y=0$ | $y=1$ | $y=2$ |
   |--|-------|-------|-------|
   | $\hat{y}=0$ | 0 | 10 | 1 |
   | $\hat{y}=1$ | 1 | 0 | 1 |
   | $\hat{y}=2$ | 1 | 10 | 0 |
   
   Risque espéré pour chaque prédiction:
   - Prédire 0: $0 \times 0.25 + 10 \times 0.45 + 1 \times 0.30 = 4.80$
   - Prédire 1: $1 \times 0.25 + 0 \times 0.45 + 1 \times 0.30 = 0.55$
   - Prédire 2: $1 \times 0.25 + 10 \times 0.45 + 0 \times 0.30 = 4.75$
   
   Prédiction optimale: classe 1 (coût minimal 0.55).

5. **Optimalité:**

   Le risque est $\mathbb{E}[\mathbf{1}[y \neq \hat{y}] | \mathbf{x}] = 1 - p(y = \hat{y} | \mathbf{x})$.
   
   Pour minimiser cette quantité, il faut maximiser $p(y = \hat{y} | \mathbf{x})$, donc choisir $\hat{y} = \arg\max_c p(y = c | \mathbf{x})$.
   
   Tout autre choix donne un risque plus élevé. Le prédicteur de Bayes est donc optimal par construction.
```
