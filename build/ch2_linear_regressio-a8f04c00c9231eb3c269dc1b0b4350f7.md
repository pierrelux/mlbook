---
kernelspec:
  name: python3
  display_name: Python 3
---

# Régression linéaire

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Dériver la solution analytique des moindres carrés ordinaires (MCO)
- Interpréter la solution MCO via la décomposition en valeurs singulières (SVD)
- Expliquer les problèmes de la colinéarité et du conditionnement
- Comprendre la régularisation Ridge et son interprétation géométrique
- Distinguer la variance des données de la variance d'estimation
```

Le [chapitre précédent](ch1_learning_problem.md) a posé le problème d'apprentissage supervisé comme un problème d'optimisation: minimiser le risque empirique sur une classe de fonctions. Ce chapitre applique ce cadre au cas le plus simple et le plus fondamental: la **régression linéaire**. Nous verrons comment résoudre analytiquement ce problème, comment interpréter la solution géométriquement, et comment la régularisation permet de contrôler la complexité du modèle.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Configuration pour des figures haute résolution
%config InlineBackend.figure_format = 'retina'
```

## Les moindres carrés ordinaires

Rappelons le problème. Un **modèle linéaire** suppose que la sortie est une combinaison linéaire des entrées:

$$
f(\mathbf{x}; \boldsymbol{\theta}) = \theta_0 + \sum_{j=1}^d \theta_j x_j = \boldsymbol{\theta}^\top \mathbf{x}
$$

où $\mathbf{x} \in \mathbb{R}^{d+1}$ est le vecteur d'entrée augmenté d'un 1 pour le biais ($x_0 = 1$), et $\boldsymbol{\theta} \in \mathbb{R}^{d+1}$ est le vecteur de paramètres.

Avec la perte quadratique, l'objectif est de minimiser la **somme des carrés des résidus** (RSS, ou SCR en français):

$$
\text{RSS}(\boldsymbol{\theta}) = \sum_{i=1}^N (y_i - \boldsymbol{\theta}^\top \mathbf{x}_i)^2 = \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2
$$

où $\mathbf{X}$ est la matrice $N \times (d+1)$ des entrées (avec une colonne de 1 pour le biais) et $\mathbf{y}$ est le vecteur des sorties.

### Dérivation de la solution analytique

En développant et en calculant le gradient:

$$
\nabla_{\boldsymbol{\theta}} \text{RSS}(\boldsymbol{\theta}) = -2\mathbf{X}^\top \mathbf{y} + 2\mathbf{X}^\top \mathbf{X} \boldsymbol{\theta}
$$

En posant le gradient égal à zéro, nous obtenons les **équations normales**:

$$
\mathbf{X}^\top \mathbf{X} \boldsymbol{\theta} = \mathbf{X}^\top \mathbf{y}
$$

Si la matrice $\mathbf{X}^\top \mathbf{X}$ est inversible, la solution unique est:

$$
\hat{\boldsymbol{\theta}}_{\text{MCO}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$

Cette solution porte le nom d'estimateur des **moindres carrés ordinaires** (MCO, ou *ordinary least squares*, OLS). Elle peut être calculée directement sans itération, ce qui en fait un exemple classique de solution analytique.

### Interprétation probabiliste: MCO = EMV sous bruit gaussien

Pourquoi minimiser la somme des carrés? Le [chapitre 1](ch1_learning_problem.md) a introduit le maximum de vraisemblance comme principe pour choisir les paramètres. Appliquons-le à la régression.

Supposons que les observations suivent le modèle:

$$
y = \boldsymbol{\theta}^\top \mathbf{x} + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)
$$

Le bruit $\varepsilon$ est gaussien, de moyenne nulle et de variance $\sigma^2$ constante. Ce modèle implique que, pour chaque observation:

$$
p(y | \mathbf{x}; \boldsymbol{\theta}) = \mathcal{N}(y \,|\, \boldsymbol{\theta}^\top \mathbf{x}, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - \boldsymbol{\theta}^\top \mathbf{x})^2}{2\sigma^2}\right)
$$

Sous l'hypothèse i.i.d., la log-vraisemblance négative est:

$$
\text{LVN}(\boldsymbol{\theta}) = -\sum_{i=1}^N \log p(y_i | \mathbf{x}_i; \boldsymbol{\theta}) = \frac{N}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - \boldsymbol{\theta}^\top \mathbf{x}_i)^2
$$

Le premier terme est une constante (ne dépend pas de $\boldsymbol{\theta}$). Minimiser la LVN revient donc à minimiser:

$$
\sum_{i=1}^N (y_i - \boldsymbol{\theta}^\top \mathbf{x}_i)^2 = \text{RSS}(\boldsymbol{\theta})
$$

Nous retrouvons exactement la somme des carrés des résidus. Ainsi, **les moindres carrés ordinaires sont l'estimateur du maximum de vraisemblance sous l'hypothèse de bruit gaussien**. Ce n'est pas un choix arbitraire: c'est la solution optimale si nous croyons que le bruit suit une distribution normale.

### Généralisation et surapprentissage

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
np.random.seed(42)
rho, S, C_L = 1.225, 20.0, 0.5
v_train = np.linspace(20, 60, 8)
L_true_train = 0.5 * rho * v_train**2 * S * C_L
L_train = L_true_train + np.random.normal(0, 400, len(v_train))

coeffs_2 = np.polyfit(v_train, L_train, 2)
coeffs_5 = np.polyfit(v_train, L_train, 5)

v_extrap = np.linspace(15, 110, 200)
L_true_extrap = 0.5 * rho * v_extrap**2 * S * C_L

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, coeffs, deg in zip(axes, [coeffs_2, coeffs_5], [2, 5]):
    ax.scatter(v_train, L_train, s=50, zorder=5, label='Observations')
    ax.plot(v_extrap, L_true_extrap, 'g-', alpha=0.5, label='Vraie relation')
    L_pred = np.polyval(coeffs, v_extrap)
    ax.plot(v_extrap, L_pred, 'k--', label=f'Polynôme degré {deg}')
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

## Régularisation Ridge

Une manière de contrôler le surapprentissage consiste à pénaliser la complexité du modèle directement dans la fonction objectif. Le **risque empirique régularisé** est:

$$
\hat{\mathcal{R}}_\lambda(\boldsymbol{\theta}) = \hat{\mathcal{R}}(\boldsymbol{\theta}) + \lambda \, C(\boldsymbol{\theta})
$$

où $C(\boldsymbol{\theta})$ mesure la complexité du modèle et $\lambda \geq 0$ contrôle l'intensité de la pénalisation. Un choix courant est la **régularisation $\ell_2$** (ou *weight decay*):

$$
C(\boldsymbol{\theta}) = \|\boldsymbol{\theta}\|_2^2 = \sum_j \theta_j^2
$$

Cette pénalisation pousse les paramètres vers zéro, ce qui a pour effet de lisser la fonction apprise. En régression linéaire, l'ajout de cette pénalité donne la **régularisation Ridge**:

$$
\hat{\boldsymbol{\theta}}_{\text{ridge}} = \arg\min_{\boldsymbol{\theta}} \frac{1}{N}\sum_{i=1}^N (y_i - \boldsymbol{\theta}^\top \mathbf{x}_i)^2 + \lambda \|\boldsymbol{\theta}\|_2^2
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
    ax.scatter(speed_train, dist_train, alpha=0.6, s=30, label='Entraînement')
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
    ax.set_title(f'$\\lambda$ = {lam}: Entr. EQM={mse_train:.1f}, Test EQM={mse_test:.1f}')
    if lam == 0:
        ax.legend()

plt.tight_layout()
```

Sans régularisation ($\lambda = 0$), le polynôme de degré 15 oscille fortement. Avec une régularisation modérée ($\lambda = 10^{-3}$), les oscillations sont atténuées et l'erreur de test diminue. Avec une régularisation trop forte ($\lambda = 1$), le modèle devient trop contraint et sous-apprend.

### Solution analytique de Ridge

Comme pour les moindres carrés ordinaires, Ridge admet une solution analytique. L'objectif régularisé est:

$$
\text{RSS}_\lambda(\boldsymbol{\theta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2 + \lambda \|\boldsymbol{\theta}\|_2^2
$$

En développant et en calculant le gradient:

$$
\nabla_{\boldsymbol{\theta}} \text{RSS}_\lambda(\boldsymbol{\theta}) = -2\mathbf{X}^\top \mathbf{y} + 2\mathbf{X}^\top \mathbf{X} \boldsymbol{\theta} + 2\lambda \boldsymbol{\theta} = -2\mathbf{X}^\top \mathbf{y} + 2(\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}) \boldsymbol{\theta}
$$

En posant le gradient égal à zéro, nous obtenons les **équations normales régularisées**:

$$
(\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}) \boldsymbol{\theta} = \mathbf{X}^\top \mathbf{y}
$$

La solution est:

$$
\hat{\boldsymbol{\theta}}_{\text{ridge}} = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}
$$

Comparons avec la solution MCO: $\hat{\boldsymbol{\theta}}_{\text{MCO}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$. La seule différence est l'ajout du terme $\lambda \mathbf{I}$ à la matrice $\mathbf{X}^\top \mathbf{X}$.

## Interprétation via la SVD

Cette section présente une autre façon d'exprimer les solutions MCO et Ridge, en utilisant la **décomposition en valeurs singulières** (SVD). Si vous n'avez jamais rencontré la SVD, ne vous inquiétez pas: nous allons l'introduire progressivement. Cette approche n'est pas strictement nécessaire pour comprendre Ridge, mais elle offre une interprétation géométrique très éclairante qui révèle *pourquoi* la régularisation fonctionne.

### Qu'est-ce que la SVD?

Si vous avez déjà rencontré la **décomposition en valeurs propres**, la SVD en est une généralisation. Pour une matrice carrée symétrique $\mathbf{A}$, la décomposition en valeurs propres s'écrit $\mathbf{A} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^\top$, où $\mathbf{Q}$ contient les vecteurs propres et $\boldsymbol{\Lambda}$ les valeurs propres. La SVD généralise cette idée à **n'importe quelle matrice**, même rectangulaire.

Pour une matrice $\mathbf{X}$ de données, la SVD la réécrit comme le produit de trois matrices:

$$
\mathbf{X} = \mathbf{U} \mathbf{D} \mathbf{V}^\top
$$

**Lien avec la décomposition en valeurs propres**: Les colonnes de $\mathbf{V}$ sont les **vecteurs propres** de $\mathbf{X}^\top \mathbf{X}$, et les valeurs singulières $d_j$ sont les **racines carrées des valeurs propres** de $\mathbf{X}^\top \mathbf{X}$. Autrement dit, si $\mathbf{X}^\top \mathbf{X} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^\top$ est la décomposition en valeurs propres de $\mathbf{X}^\top \mathbf{X}$, alors $d_j = \sqrt{\lambda_j}$ où $\lambda_j$ sont les valeurs propres. De même, les colonnes de $\mathbf{U}$ sont les vecteurs propres de $\mathbf{X} \mathbf{X}^\top$.

Cette connexion est utile car $\mathbf{X}^\top \mathbf{X}$ apparaît naturellement dans la régression (c'est la matrice que nous inversons pour MCO). Les valeurs singulières $d_j$ nous renseignent donc directement sur le "conditionnement" de cette matrice: si certaines valeurs singulières sont très petites, alors $\mathbf{X}^\top \mathbf{X}$ est proche d'être singulière (non inversible).

### Interprétation géométrique

- **$\mathbf{V}$** contient les **directions principales** dans l'espace des caractéristiques (les colonnes $\mathbf{v}_j$ sont orthonormales). Ces directions correspondent aux axes le long desquels la matrice $\mathbf{X}$ transforme les vecteurs de manière la plus efficace.
- **$\mathbf{D}$** est une matrice diagonale contenant les **valeurs singulières** $d_1 \geq d_2 \geq \cdots \geq d_d \geq 0$, ordonnées du plus grand au plus petit. Chaque valeur singulière $d_j$ mesure l'**amplitude** de la transformation le long de la direction $\mathbf{v}_j$. Une grande valeur singulière signifie que la transformation est forte dans cette direction; une petite valeur singulière signifie que la transformation est faible.
- **$\mathbf{U}$** contient les directions correspondantes dans l'espace des observations (les colonnes $\mathbf{u}_j$ sont orthonormales). Chaque $\mathbf{u}_j$ indique comment les observations se projettent sur la direction principale $\mathbf{v}_j$.

### Solution MCO via SVD

En utilisant cette décomposition, la solution MCO peut s'écrire:

$$
\hat{\boldsymbol{\theta}}_{\text{MCO}} = \sum_{j=1}^d \frac{\mathbf{u}_j^\top \mathbf{y}}{d_j} \mathbf{v}_j
$$

Cette formule décompose la solution en une somme de contributions le long de chaque direction principale $\mathbf{v}_j$. Le terme $\frac{\mathbf{u}_j^\top \mathbf{y}}{d_j}$ mesure combien la sortie $\mathbf{y}$ s'aligne avec la direction $\mathbf{u}_j$, divisé par l'amplitude $d_j$ de cette direction. Notez que diviser par une petite valeur singulière $d_j$ peut amplifier le bruit, ce qui explique pourquoi MCO peut être instable quand certaines directions ont de petites valeurs singulières.

### Solution Ridge via SVD

Pour Ridge, la solution devient:

$$
\hat{\boldsymbol{\theta}}_{\text{ridge}} = \sum_{j=1}^d \frac{d_j^2}{d_j^2 + \lambda} \frac{\mathbf{u}_j^\top \mathbf{y}}{d_j} \mathbf{v}_j
$$

La différence avec MCO est le facteur de rétrécissement $\frac{d_j^2}{d_j^2 + \lambda}$ qui multiplie chaque terme. Ce facteur est toujours inférieur à 1, ce qui "rétrécit" chaque composante vers zéro. L'effet clé est que ce rétrécissement est **différencié**: les directions avec de petites valeurs singulières sont rétrécies plus fortement que celles avec de grandes valeurs singulières.

**Avantages numériques**: Au-delà de l'interprétation, la SVD offre aussi des avantages pratiques. Elle est plus stable numériquement que l'inversion directe de $\mathbf{X}^\top \mathbf{X}$, surtout quand cette matrice est mal conditionnée (c'est-à-dire quand certaines valeurs singulières sont très petites). Les algorithmes SVD gèrent mieux ces cas délicats.

### Visualisation: ellipse des données et vecteurs singuliers

Pour rendre ces concepts concrets, visualisons ce que la SVD capture sur un nuage de données 2D. Générons des points suivant une distribution gaussienne avec une covariance non triviale (les deux variables sont corrélées).

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

np.random.seed(42)

# Générer des données gaussiennes corrélées
n_points = 200
mean = [0, 0]
cov = [[2.0, 1.2], [1.2, 1.0]]  # Covariance non diagonale
X = np.random.multivariate_normal(mean, cov, n_points)

# Centrer les données
X_centered = X - X.mean(axis=0)

# SVD de la matrice de données centrée
U, d, Vt = np.linalg.svd(X_centered, full_matrices=False)
V = Vt.T

# Les valeurs singulières sont liées aux écarts-types: d_j / sqrt(N-1)
# Pour l'ellipse, nous utilisons les écarts-types dans chaque direction
std_1 = d[0] / np.sqrt(n_points - 1)
std_2 = d[1] / np.sqrt(n_points - 1)

# Créer la figure
fig, ax = plt.subplots(figsize=(8, 6))

# Tracer les points
ax.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.5, s=20, c='tab:blue', label='Données')

# Tracer les vecteurs singuliers (directions principales)
origin = [0, 0]
scale = 2  # Facteur d'échelle pour la visualisation

# Premier vecteur singulier (direction de plus grande variance)
ax.annotate('', xy=V[:, 0] * std_1 * scale, xytext=origin,
            arrowprops=dict(arrowstyle='->', color='tab:red', lw=2.5))
ax.annotate('', xy=-V[:, 0] * std_1 * scale, xytext=origin,
            arrowprops=dict(arrowstyle='->', color='tab:red', lw=2.5))

# Deuxième vecteur singulier (direction de plus petite variance)
ax.annotate('', xy=V[:, 1] * std_2 * scale, xytext=origin,
            arrowprops=dict(arrowstyle='->', color='tab:orange', lw=2.5))
ax.annotate('', xy=-V[:, 1] * std_2 * scale, xytext=origin,
            arrowprops=dict(arrowstyle='->', color='tab:orange', lw=2.5))

# Ellipse de confiance (2 écarts-types)
angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
ellipse = Ellipse(xy=(0, 0), width=4*std_1, height=4*std_2, angle=angle,
                  fill=False, edgecolor='gray', linestyle='--', linewidth=1.5)
ax.add_patch(ellipse)

# Annotations
ax.text(V[0, 0] * std_1 * scale * 1.15, V[1, 0] * std_1 * scale * 1.15, 
        f'$\\mathbf{{v}}_1$ ($d_1 = {d[0]:.1f}$)', fontsize=11, color='tab:red')
ax.text(V[0, 1] * std_2 * scale * 1.3, V[1, 1] * std_2 * scale * 1.3, 
        f'$\\mathbf{{v}}_2$ ($d_2 = {d[1]:.1f}$)', fontsize=11, color='tab:orange')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Nuage gaussien et directions principales (SVD)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlim(-4, 4)
ax.set_ylim(-3, 3)

plt.tight_layout()
```

La figure montre un nuage de 200 points tirés d'une gaussienne 2D. Les flèches représentent les **vecteurs singuliers** $\mathbf{v}_1$ et $\mathbf{v}_2$:

- Le vecteur $\mathbf{v}_1$ (rouge) pointe dans la direction de **plus grande variance**. La valeur singulière $d_1$ mesure l'amplitude de la dispersion dans cette direction.
- Le vecteur $\mathbf{v}_2$ (orange) pointe dans la direction de **plus petite variance**, perpendiculaire à $\mathbf{v}_1$. La valeur singulière $d_2$ est plus petite.

L'ellipse en pointillés représente la région contenant environ 95% des données si elles suivent exactement la distribution gaussienne. Ses axes coïncident avec les vecteurs singuliers, et les longueurs des demi-axes sont proportionnelles aux valeurs singulières.

Cette visualisation illustre pourquoi la SVD est si utile: elle identifie automatiquement les **axes naturels** des données. Dans le contexte de la régression, si les caractéristiques forment un nuage allongé (valeurs singulières très différentes), alors certaines directions contiennent beaucoup d'information (grandes valeurs singulières) tandis que d'autres en contiennent peu (petites valeurs singulières).

## Deux variances: données vs estimation

```{margin}
**Attention**: Le mot « variance » désigne deux quantités distinctes selon le contexte. Ne pas les confondre!
```

Avant d'aller plus loin, clarifions une source fréquente de confusion. Le mot **variance** apparaît dans deux contextes très différents lorsqu'on parle de SVD et de régularisation:

1. **Variance des données** (dispersion): mesure l'étalement des données le long d'une direction $\mathbf{v}_j$. Elle est proportionnelle à $d_j^2$. Une grande valeur singulière $d_j$ signifie que les données sont très dispersées dans cette direction.

2. **Variance d'estimation** (incertitude): mesure l'incertitude sur notre estimé $\hat{\theta}_j$ du paramètre correspondant à la direction $\mathbf{v}_j$. Elle est proportionnelle à $1/d_j^2$. Une petite valeur singulière $d_j$ signifie que notre estimé est très incertain.

Ces deux variances sont **inversement reliées**:

| Valeur singulière $d_j$ | Variance des données | Variance d'estimation | Interprétation |
|-------------------------|---------------------|----------------------|----------------|
| Grande | Élevée (données étalées) | Faible (estimé précis) | Beaucoup d'information |
| Petite | Faible (données concentrées) | Élevée (estimé incertain) | Peu d'information |

**Intuition**: Imaginez estimer une pente à partir de données. Si les points sont très étalés horizontalement (grande variance des données en $x$), la pente est facile à déterminer avec précision (faible variance d'estimation). Si les points sont tous regroupés (petite variance des données), la pente est très incertaine (grande variance d'estimation).

C'est cette relation inverse qui explique le comportement de Ridge:
- Ridge rétrécit les directions où $d_j$ est **petit** (faible variance des données)
- Ce sont précisément les directions où la **variance d'estimation** est grande
- En rétrécissant ces directions, Ridge réduit la variance d'estimation au prix d'un biais

Ainsi, quand nous disons que « Ridge contrôle la variance », nous parlons de la **variance d'estimation** des paramètres, pas de la variance des données. La régularisation n'affecte pas la dispersion des données; elle réduit l'incertitude de nos estimés en les « tirant » vers zéro.

## Spectre des valeurs singulières et rang effectif

En pratique, les données réelles ont souvent des dizaines ou des centaines de dimensions. Comment se comportent les valeurs singulières dans ce cas? Examinons un exemple avec des données de dimension plus élevée.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

# Simuler des données avec une structure de rang bas + bruit
n_samples = 100
n_features = 30

# Vraie structure: combinaison de 5 facteurs latents
n_latent = 5
latent = np.random.randn(n_samples, n_latent)
loadings = np.random.randn(n_latent, n_features)
X_signal = latent @ loadings

# Ajouter du bruit
noise_level = 0.5
X = X_signal + noise_level * np.random.randn(n_samples, n_features)

# Centrer
X_centered = X - X.mean(axis=0)

# SVD
U, d, Vt = np.linalg.svd(X_centered, full_matrices=False)

# Variance expliquée
variance_explained = d**2 / np.sum(d**2)
cumulative_variance = np.cumsum(variance_explained)

# Créer la figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panneau gauche: spectre des valeurs singulières
ax1 = axes[0]
ax1.semilogy(range(1, len(d)+1), d, 'o-', markersize=6, linewidth=1.5, color='tab:blue')
ax1.axhline(d[n_latent], color='tab:red', linestyle='--', linewidth=1.5, 
            label=f'Seuil (rang effectif = {n_latent})')
ax1.fill_between(range(1, n_latent+1), d[:n_latent], alpha=0.3, color='tab:green', label='Signal')
ax1.fill_between(range(n_latent+1, len(d)+1), d[n_latent:], alpha=0.3, color='tab:orange', label='Bruit')
ax1.set_xlabel('Indice $j$')
ax1.set_ylabel('Valeur singulière $d_j$ (échelle log)')
ax1.set_title('Spectre des valeurs singulières')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3, which='both')
ax1.set_xlim(0.5, len(d)+0.5)

# Panneau droit: variance expliquée cumulative
ax2 = axes[1]
ax2.plot(range(1, len(d)+1), cumulative_variance * 100, 'o-', markersize=6, 
         linewidth=1.5, color='tab:blue')
ax2.axhline(95, color='gray', linestyle='--', linewidth=1, label='Seuil 95%')
ax2.axvline(n_latent, color='tab:red', linestyle='--', linewidth=1.5)

# Trouver k pour 95% de variance
k_95 = np.searchsorted(cumulative_variance, 0.95) + 1
ax2.scatter([k_95], [cumulative_variance[k_95-1]*100], s=100, color='tab:red', zorder=5)
ax2.annotate(f'{k_95} composantes\npour 95%', xy=(k_95, cumulative_variance[k_95-1]*100),
             xytext=(k_95+5, cumulative_variance[k_95-1]*100-10), fontsize=10,
             arrowprops=dict(arrowstyle='->', color='gray'))

ax2.set_xlabel('Nombre de composantes $k$')
ax2.set_ylabel('Variance expliquée cumulative (%)')
ax2.set_title('Variance expliquée')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.5, len(d)+0.5)
ax2.set_ylim(0, 105)

plt.tight_layout()
```

Le panneau de gauche montre le **spectre des valeurs singulières** en échelle logarithmique. On observe un schéma typique:

- Les premières valeurs singulières sont grandes: elles correspondent aux **directions du signal**, la vraie structure sous-jacente des données.
- Après un certain point (ici, autour de $j = 5$), les valeurs singulières chutent et forment un "plancher": ce sont les **directions du bruit**.

Le **rang effectif** est le nombre de valeurs singulières significativement au-dessus du plancher de bruit. Dans cet exemple, nous avons simulé 5 facteurs latents, et le spectre révèle bien cette structure: les 5 premières valeurs singulières dominent.

Le panneau de droite montre la **variance expliquée cumulative**. C'est un outil pratique pour choisir combien de composantes retenir:

- **Critère du seuil**: Retenir assez de composantes pour expliquer 95% (ou 99%) de la variance.
- **Critère du coude**: Chercher le "coude" dans le spectre où les valeurs singulières cessent de décroître rapidement.
- **Critère du gap**: Si le spectre présente un saut net (comme ici entre $d_5$ et $d_6$), c'est un bon point de coupure.

### De la troncature à la réduction de dimension (ACP)

L'analyse ci-dessus suggère une idée: si les dernières directions ne contiennent que du bruit, pourquoi ne pas simplement les ignorer? Au lieu de rétrécir les coefficients comme Ridge, nous pourrions **tronquer** la représentation en ne gardant que les $k$ premières directions principales.

C'est exactement l'idée de l'**analyse en composantes principales** (ACP). Au lieu de travailler avec les $d$ caractéristiques originales, nous projetons les données sur les $k$ premiers vecteurs singuliers:

$$
\mathbf{z}_n = \mathbf{V}_k^\top (\mathbf{x}_n - \bar{\mathbf{x}}) \in \mathbb{R}^k
$$

où $\mathbf{V}_k$ contient les $k$ premiers vecteurs singuliers (les colonnes de $\mathbf{V}$ correspondant aux $k$ plus grandes valeurs singulières).

Cette projection préserve au mieux la variance des données: les $k$ composantes principales capturent la direction où les données varient le plus. La reconstruction à partir de cette représentation compressée s'écrit:

$$
\hat{\mathbf{x}}_n = \mathbf{V}_k \mathbf{z}_n + \bar{\mathbf{x}}
$$

L'erreur de reconstruction est minimale parmi toutes les projections linéaires sur un sous-espace de dimension $k$.

**Lien entre Ridge et ACP**: Les deux approches traitent le même problème (les directions à faible valeur singulière sont bruitées) mais différemment:

| Approche | Traitement des directions bruitées | Type de régularisation |
|----------|-----------------------------------|------------------------|
| **Ridge** | Rétrécit (soft thresholding) | Continue: garde tout, pénalise |
| **ACP** | Élimine (hard thresholding) | Discrète: garde $k$, ignore le reste |

Ridge est appropriée pour la régression supervisée, où même les petites directions peuvent contenir du signal utile pour prédire $\mathbf{y}$. L'ACP est appropriée pour la réduction de dimension non supervisée, où nous voulons une représentation compacte des données elles-mêmes.

```{margin} ACP en détail
L'analyse en composantes principales mérite un traitement approfondi. Nous y reviendrons dans un chapitre dédié à l'apprentissage non supervisé, où nous explorerons aussi les méthodes de clustering et les modèles génératifs.
```

## Pourquoi λI aide: trois perspectives

Le terme diagonal $\lambda \mathbf{I}$ ajouté à $\mathbf{X}^\top \mathbf{X}$ a plusieurs effets bénéfiques:

### 1. Amélioration du conditionnement

La matrice $\mathbf{X}^\top \mathbf{X}$ peut être mal conditionnée (ses valeurs propres varient sur plusieurs ordres de grandeur) ou même singulière. L'ajout de $\lambda \mathbf{I}$ augmente toutes les valeurs propres de $\lambda$, rendant la matrice inversible et mieux conditionnée.

### 2. Rétrécissement des coefficients

Comme nous l'avons vu dans la section SVD ci-dessus, la solution Ridge s'écrit:

$$
\hat{\boldsymbol{\theta}}_{\text{ridge}} = \sum_{j=1}^d \frac{d_j^2}{d_j^2 + \lambda} \frac{\mathbf{u}_j^\top \mathbf{y}}{d_j} \mathbf{v}_j
$$

Le facteur de rétrécissement $\frac{d_j^2}{d_j^2 + \lambda}$ est toujours inférieur à 1, ce qui "rétrécit" chaque composante vers zéro. L'effet est différencié selon les directions:

- Pour une grande valeur singulière $d_j$ (fort signal), le facteur $\frac{d_j^2}{d_j^2 + \lambda}$ reste proche de 1 même pour des valeurs modérées de $\lambda$. La direction est peu affectée.
- Pour une petite valeur singulière $d_j$ (faible signal), le facteur $\frac{d_j^2}{d_j^2 + \lambda}$ décroît rapidement avec $\lambda$. La direction est fortement pénalisée.

### 3. Stabilité numérique

Quand $\mathbf{X}^\top \mathbf{X}$ est presque singulière, de petites perturbations dans les données causent de grandes variations dans $\hat{\boldsymbol{\theta}}_{\text{MCO}}$. La régularisation réduit cette sensibilité.

## Géométrie de la régularisation Ridge

Pour visualiser le rétrécissement et comprendre son effet, examinons un exemple concret. L'animation suivante montre simultanément trois perspectives sur la régularisation Ridge: les données et la droite ajustée, le paysage de perte avec la contrainte, et les facteurs de rétrécissement.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from IPython.display import Image

# Générer des données de régression simple
np.random.seed(42)
n = 30

# Une seule caractéristique pour visualisation claire
x = np.random.uniform(-2, 2, n)
# Relation linéaire avec bruit
theta_true = 1.5
y = theta_true * x + np.random.normal(0, 0.8, n)

# Ajouter une caractéristique corrélée (pour créer de la colinéarité)
x2 = 0.9 * x + 0.3 * np.random.randn(n)

# Matrice de design avec les deux caractéristiques
X = np.column_stack([x, x2])

# Solution MCO
theta_ols = np.linalg.lstsq(X, y, rcond=None)[0]

# SVD pour analyse
U, d_svd, Vt = np.linalg.svd(X, full_matrices=False)
V = Vt.T

# Fonction pour calculer la solution Ridge
def ridge_solution(X, y, lam):
    n_features = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(n_features), X.T @ y)

# Préparer la grille pour les contours RSS
theta1_range = np.linspace(-0.5, 3, 100)
theta2_range = np.linspace(-1.5, 2, 100)
T1, T2 = np.meshgrid(theta1_range, theta2_range)

# Calculer RSS pour chaque point de la grille
RSS = np.zeros_like(T1)
for i in range(T1.shape[0]):
    for j in range(T1.shape[1]):
        theta = np.array([T1[i, j], T2[i, j]])
        residuals = y - X @ theta
        RSS[i, j] = np.sum(residuals**2)

# Créer la figure avec trois panneaux
fig = plt.figure(figsize=(15, 5))

# === Panneau 1: Données et droite ajustée ===
ax1 = fig.add_subplot(1, 3, 1)

# Données
ax1.scatter(x, y, c='tab:blue', s=50, alpha=0.7, label='Données', zorder=3)

# Grille pour tracer les droites
x_grid = np.linspace(-2.5, 2.5, 100)

# Droite MCO (fixe) - on utilise seulement theta1 car x et x2 sont très corrélés
# La prédiction effective est environ (theta1 + 0.9*theta2) * x
slope_ols = theta_ols[0] + 0.9 * theta_ols[1]  # Pente effective
y_ols = slope_ols * x_grid
ax1.plot(x_grid, y_ols, 'k-', linewidth=2, alpha=0.7, label='MCO')

# Droite Ridge (animée)
line_ridge, = ax1.plot([], [], '-', color='tab:orange', linewidth=2.5, label='Ridge')

# Ligne horizontale (prédiction = moyenne, lambda infini)
y_mean = np.mean(y)
ax1.axhline(y_mean, color='gray', linestyle=':', alpha=0.5, label=f'Moyenne ($\\lambda \\to \\infty$)')

ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
ax1.set_title('Données et droite de régression')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-4, 5)

# Texte pour les coefficients
coef_text = ax1.text(0.98, 0.02, '', transform=ax1.transAxes, fontsize=10, 
                     ha='right', va='bottom',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# === Panneau 2: Paysage de perte ===
ax2 = fig.add_subplot(1, 3, 2)

# Contours RSS (ellipses centrées sur OLS)
levels = np.percentile(RSS.flatten(), [5, 15, 30, 50, 70, 85, 95])
contours = ax2.contour(T1, T2, RSS, levels=levels, colors='gray', alpha=0.6)
ax2.clabel(contours, inline=True, fontsize=8, fmt='%.0f')

# Solution MCO (fixe)
ax2.plot(theta_ols[0], theta_ols[1], 'ko', markersize=12, label='MCO', zorder=5)
ax2.annotate('MCO', xy=(theta_ols[0], theta_ols[1]), 
             xytext=(theta_ols[0] + 0.2, theta_ols[1] + 0.2),
             fontsize=11, ha='left')

# Origine = coefficients nuls (prédiction constante)
ax2.plot(0, 0, 'k+', markersize=15, markeredgewidth=2, zorder=4)
ax2.annotate('$\\boldsymbol{\\theta} = 0$\n(pente nulle)', xy=(0, 0), 
             xytext=(-0.4, -1.2), fontsize=9, ha='center', color='gray')

# Cercle de contrainte Ridge (animé)
circle_ridge = Circle((0, 0), radius=np.linalg.norm(theta_ols), 
                       fill=False, edgecolor='tab:orange', linewidth=2.5, 
                       linestyle='-', alpha=0.8, zorder=3)
ax2.add_patch(circle_ridge)

# Solution Ridge (animée)
point_ridge, = ax2.plot([], [], 'o', color='tab:orange', markersize=10, 
                        label='Ridge', zorder=6)

# Chemin de régularisation
lambda_path = np.logspace(-3, 1.5, 50)
theta_path = np.array([ridge_solution(X, y, l) for l in lambda_path])
ax2.plot(theta_path[:, 0], theta_path[:, 1], 'tab:orange', linewidth=1.5, 
         alpha=0.4, linestyle='--', label='Chemin')

ax2.set_xlabel('$\\theta_1$')
ax2.set_ylabel('$\\theta_2$')
ax2.set_title('Paysage RSS et chemin de régularisation')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.5, 3)
ax2.set_ylim(-1.5, 2)
ax2.set_aspect('equal')

# Texte pour lambda
lambda_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes, fontsize=11, 
                       va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# === Panneau 3: Facteurs de rétrécissement ===
ax3 = fig.add_subplot(1, 3, 3)

lambda_range = np.linspace(0, 10, 200)
shrink1_curve = d_svd[0]**2 / (d_svd[0]**2 + lambda_range)
shrink2_curve = d_svd[1]**2 / (d_svd[1]**2 + lambda_range)

ax3.plot(lambda_range, shrink1_curve, 'b-', linewidth=2, 
         label=f'Direction forte ($d_1={d_svd[0]:.1f}$)')
ax3.plot(lambda_range, shrink2_curve, 'r-', linewidth=2, 
         label=f'Direction faible ($d_2={d_svd[1]:.2f}$)')

# Zone de surapprentissage et sous-apprentissage
ax3.axvspan(0, 0.5, alpha=0.1, color='red', label='Surapprentissage')
ax3.axvspan(5, 10, alpha=0.1, color='blue', label='Sous-apprentissage')

point_shrink1, = ax3.plot([], [], 'bo', markersize=10, zorder=3)
point_shrink2, = ax3.plot([], [], 'ro', markersize=10, zorder=3)

ax3.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
ax3.set_xlabel('$\\lambda$')
ax3.set_ylabel('Facteur de rétrécissement')
ax3.set_title('Rétrécissement par direction SVD')
ax3.legend(loc='center right', fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 1.1)

shrink_text = ax3.text(0.02, 0.5, '', transform=ax3.transAxes, fontsize=10, 
                       va='center', ha='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

# Fonction d'animation
def animate(frame):
    if frame < 80:
        lam = (frame / 80) * 10
    else:
        lam = 10
    
    # Solution Ridge
    theta_ridge = ridge_solution(X, y, lam)
    
    # Panneau 1: Mettre à jour la droite
    slope_ridge = theta_ridge[0] + 0.9 * theta_ridge[1]
    y_ridge = slope_ridge * x_grid
    line_ridge.set_data(x_grid, y_ridge)
    coef_text.set_text(f'Pente MCO: {slope_ols:.2f}\nPente Ridge: {slope_ridge:.2f}')
    
    # Panneau 2: Mettre à jour le cercle et le point
    norm_ridge = np.linalg.norm(theta_ridge)
    circle_ridge.set_radius(norm_ridge)
    point_ridge.set_data([theta_ridge[0]], [theta_ridge[1]])
    lambda_text.set_text(f'$\\lambda = {lam:.1f}$')
    
    # Panneau 3: Mettre à jour les points de rétrécissement
    shrink1 = d_svd[0]**2 / (d_svd[0]**2 + lam)
    shrink2 = d_svd[1]**2 / (d_svd[1]**2 + lam)
    point_shrink1.set_data([lam], [shrink1])
    point_shrink2.set_data([lam], [shrink2])
    shrink_text.set_text(f'Facteur dir. 1: {shrink1:.2f}\nFacteur dir. 2: {shrink2:.2f}')
    
    return (line_ridge, point_ridge, circle_ridge, lambda_text, 
            point_shrink1, point_shrink2, coef_text, shrink_text)

# Créer l'animation
anim = FuncAnimation(fig, animate, frames=90, interval=80, blit=False, repeat=True)
anim.save('_static/ridge_geometry.gif', writer='pillow', fps=12, dpi=100)
plt.close()

# Afficher le GIF
Image(filename='_static/ridge_geometry.gif')
```

L'animation relie trois perspectives sur la régularisation Ridge lorsque $\lambda$ augmente de 0 à 10:

**Panneau de gauche (données et ajustement)**: Les points bleus sont les données d'entraînement. La droite noire est l'ajustement MCO ($\lambda = 0$), la droite orange est l'ajustement Ridge. À mesure que $\lambda$ augmente, la pente de la droite Ridge **diminue**, se rapprochant de la ligne horizontale (prédiction constante égale à la moyenne). C'est le **rétrécissement vers zéro**: Ridge "tire" les coefficients vers l'origine, ce qui réduit la pente.

**Panneau central (paysage de perte)**: Chaque point de ce plan représente un choix de coefficients $(\theta_1, \theta_2)$. Les contours gris montrent la fonction de coût RSS: plus on est proche du point noir (MCO), plus l'erreur sur les données d'entraînement est faible. L'ellipse est allongée car $x_1$ et $x_2$ sont corrélées (colinéarité). L'origine $\boldsymbol{\theta} = (0, 0)$ correspond à une **pente nulle** (prédiction constante). Le cercle orange montre la norme de la solution Ridge courante $\|\hat{\boldsymbol{\theta}}_{\text{ridge}}\|_2$. La formulation pénalisée $\text{RSS} + \lambda\|\boldsymbol{\theta}\|^2$ est équivalente à la formulation contrainte $\min \text{RSS}$ sous $\|\boldsymbol{\theta}\|^2 \leq t$, où $\lambda$ joue le rôle du multiplicateur de Lagrange: pour chaque $\lambda$, il existe un $t$ tel que les deux problèmes ont la même solution. À mesure que $\lambda$ augmente, la solution se déplace le long du **chemin de régularisation** vers l'origine.

**Panneau de droite (rétrécissement différencié)**: La direction "forte" (grande valeur singulière $d_1$, où les données sont dispersées) est peu affectée par la régularisation. La direction "faible" (petite valeur singulière $d_2$, direction de colinéarité) est **rétrécit beaucoup plus rapidement**. C'est le cœur de l'effet Ridge: pénaliser davantage les directions où le signal est faible et l'estimation instable.

L'intuition géométrique est la suivante: quand les données sont colinéaires, l'ellipse RSS est très allongée. De petites perturbations dans les données causent de grands déplacements de la solution MCO le long de l'axe allongé. La pénalité Ridge ajoute un terme $\lambda\|\boldsymbol{\theta}\|^2$ qui « tire » la solution vers l'origine. Dans la formulation contrainte équivalente, cela correspond à chercher le minimum de RSS à l'intérieur d'une boule de rayon $\sqrt{t}$. Plus la boule est petite (plus $\lambda$ est grand), plus la solution est proche de l'origine et donc plus stable.

## Résumé

Ce chapitre a développé les outils fondamentaux pour la régression linéaire:

- Les **moindres carrés ordinaires** (MCO) minimisent la somme des carrés des résidus et admettent une solution analytique: $\hat{\boldsymbol{\theta}}_{\text{MCO}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$.

- La **décomposition en valeurs singulières** (SVD) offre une interprétation géométrique: MCO amplifie le bruit le long des directions de faibles valeurs singulières.

- La **régularisation Ridge** ajoute une pénalité $\ell_2$ qui **rétrécit** les coefficients vers zéro, avec un rétrécissement différencié: les directions faibles sont pénalisées plus fortement.

- Il faut distinguer deux types de **variance**: la variance des données (dispersion, $\propto d_j^2$) et la variance d'estimation (incertitude, $\propto 1/d_j^2$). Ridge réduit la variance d'estimation.

- Le **spectre des valeurs singulières** révèle la structure des données et permet de distinguer signal et bruit.

Nous avons vu comment résoudre la régression. Mais la régression n'est qu'un type de problème supervisé. Le [chapitre suivant](ch3_classification.md) aborde la **classification linéaire**, où la sortie est une catégorie plutôt qu'un nombre réel.

## Exercices

````{admonition} Exercice 1: Dérivation des moindres carrés ordinaires ★
:class: hint dropdown

Soit un problème de régression linéaire simple avec $N$ observations:

$$
y_i = \theta_0 + \theta_1 x_i + \epsilon_i, \quad i = 1, \ldots, N
$$

1. Écrivez la somme des carrés des résidus $\text{RSS}(\theta_0, \theta_1) = \sum_{i=1}^N (y_i - \theta_0 - \theta_1 x_i)^2$.

2. Calculez les dérivées partielles $\frac{\partial \text{RSS}}{\partial \theta_0}$ et $\frac{\partial \text{RSS}}{\partial \theta_1}$.

3. En posant ces dérivées égales à zéro, résolvez le système d'équations pour obtenir les estimateurs $\hat{\theta}_0$ et $\hat{\theta}_1$.

4. **Application numérique**: Pour les données $(x, y) = \{(1, 2), (2, 4), (3, 5), (4, 4), (5, 5)\}$, calculez les coefficients MCO $\hat{\theta}_0$ et $\hat{\theta}_1$, puis la prédiction pour $x = 6$.
````

```{admonition} Solution Exercice 1
:class: dropdown

1. **Somme des carrés des résidus:**

$$
\text{RSS}(\theta_0, \theta_1) = \sum_{i=1}^N (y_i - \theta_0 - \theta_1 x_i)^2
$$

2. **Dérivées partielles:**

$$
\frac{\partial \text{RSS}}{\partial \theta_0} = -2 \sum_{i=1}^N (y_i - \theta_0 - \theta_1 x_i) = -2 \sum_{i=1}^N y_i + 2N\theta_0 + 2\theta_1 \sum_{i=1}^N x_i
$$

$$
\frac{\partial \text{RSS}}{\partial \theta_1} = -2 \sum_{i=1}^N x_i(y_i - \theta_0 - \theta_1 x_i) = -2 \sum_{i=1}^N x_i y_i + 2\theta_0 \sum_{i=1}^N x_i + 2\theta_1 \sum_{i=1}^N x_i^2
$$

3. **Résolution du système:**

En posant les dérivées égales à zéro et en simplifiant:

$$
N\theta_0 + \theta_1 \sum x_i = \sum y_i \quad \text{(équation 1)}
$$

$$
\theta_0 \sum x_i + \theta_1 \sum x_i^2 = \sum x_i y_i \quad \text{(équation 2)}
$$

De l'équation 1: $\theta_0 = \bar{y} - \theta_1 \bar{x}$ où $\bar{x} = \frac{1}{N}\sum x_i$ et $\bar{y} = \frac{1}{N}\sum y_i$.

En substituant dans l'équation 2 et en simplifiant:

$$
\hat{\theta}_1 = \frac{\sum_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^N (x_i - \bar{x})^2} = \frac{\text{Cov}(x, y)}{\text{Var}(x)}
$$

$$
\hat{\theta}_0 = \bar{y} - \hat{\theta}_1 \bar{x}
$$

4. **Application numérique:**

   - $\bar{x} = (1+2+3+4+5)/5 = 3$, $\bar{y} = (2+4+5+4+5)/5 = 4$
   - $\sum (x_i - \bar{x})(y_i - \bar{y}) = (-2)(-2) + (-1)(0) + (0)(1) + (1)(0) + (2)(1) = 4 + 0 + 0 + 0 + 2 = 6$
   - $\sum (x_i - \bar{x})^2 = 4 + 1 + 0 + 1 + 4 = 10$
   - $\hat{\theta}_1 = 6/10 = 0.6$
   - $\hat{\theta}_0 = 4 - 0.6 \times 3 = 2.2$
   - Prédiction pour $x=6$: $\hat{y} = 2.2 + 0.6 \times 6 = 5.8$
```

````{admonition} Exercice 2: Régression ridge et colinéarité ★★
:class: hint dropdown

La colinéarité entre les caractéristiques rend la matrice $\mathbf{X}^\top \mathbf{X}$ mal conditionnée, ce qui peut déstabiliser la solution MCO.

1. Générez des données avec deux caractéristiques presque colinéaires:

   ```python
   np.random.seed(42)
   n = 30
   x1 = np.random.randn(n)
   x2 = x1 + 0.01 * np.random.randn(n)  # x2 ≈ x1
   y = 2*x1 + 3*x2 + 0.5*np.random.randn(n)
   ```

2. Calculez le nombre de conditionnement de $\mathbf{X}^\top \mathbf{X}$ (avec `np.linalg.cond`). Que signifie un grand nombre de conditionnement?

3. Ajustez un modèle MCO. Les coefficients $\hat{\theta}_1$ et $\hat{\theta}_2$ sont-ils proches des vraies valeurs (2 et 3)?

4. Ajustez des modèles Ridge pour $\lambda \in \{0.01, 0.1, 1, 10\}$. Comment les coefficients évoluent-ils?

5. Tracez le «chemin de régularisation»: les coefficients en fonction de $\log(\lambda)$.
````

````{admonition} Solution Exercice 2
:class: dropdown

1. **Génération des données:** (code fourni dans l'énoncé)

2. **Nombre de conditionnement:**

   ```python
   X = np.column_stack([np.ones(n), x1, x2])
   cond = np.linalg.cond(X.T @ X)
   print(f"Conditionnement: {cond:.0f}")
   ```
   
   Le nombre de conditionnement est très élevé (de l'ordre de $10^6$ ou plus). Cela signifie que de petites perturbations dans les données peuvent causer de grandes variations dans la solution. La matrice est proche d'être singulière.

3. **Modèle MCO:**

   ```python
   theta_ols = np.linalg.solve(X.T @ X, X.T @ y)
   ```
   
   Les coefficients MCO sont très instables: $\hat{\theta}_1$ et $\hat{\theta}_2$ peuvent être très différents de 2 et 3, et parfois de signes opposés avec de grandes magnitudes. Le modèle «distribue» l'effet entre les deux variables de manière arbitraire.

4. **Modèles Ridge:**

   ```python
   from sklearn.linear_model import Ridge
   for lam in [0.01, 0.1, 1, 10]:
       model = Ridge(alpha=lam, fit_intercept=True)
       model.fit(np.column_stack([x1, x2]), y)
       print(f"λ={lam}: θ1={model.coef_[0]:.2f}, θ2={model.coef_[1]:.2f}")
   ```
   
   Avec $\lambda$ croissant, les coefficients se rapprochent de zéro et deviennent plus stables. Les deux coefficients convergent vers des valeurs similaires (autour de 2,5 chacun), ce qui reflète mieux la symétrie du problème.

5. **Chemin de régularisation:**

   ```python
   lambdas = np.logspace(-3, 2, 50)
   coefs = []
   for lam in lambdas:
       model = Ridge(alpha=lam, fit_intercept=True)
       model.fit(np.column_stack([x1, x2]), y)
       coefs.append(model.coef_)
   coefs = np.array(coefs)
   
   plt.plot(np.log10(lambdas), coefs[:, 0], label='θ1')
   plt.plot(np.log10(lambdas), coefs[:, 1], label='θ2')
   plt.xlabel('log10(λ)')
   plt.ylabel('Coefficients')
   plt.legend()
   ```
   
   On observe que pour $\lambda$ petit, les coefficients sont instables et peuvent être extrêmes. Pour $\lambda$ grand, ils convergent vers zéro. Il existe une zone intermédiaire où les coefficients sont raisonnables.
````

````{admonition} Exercice 3: Homoscédasticité et hétéroscédasticité ★★
:class: hint dropdown

En régression, l'**homoscédasticité** suppose que la variance du bruit est constante: $\text{Var}(\epsilon | x) = \sigma^2$. L'**hétéroscédasticité** suppose que la variance dépend de $x$.

1. Générez deux jeux de données ($N = 100$, $x \in [0, 10]$):
   - **Homoscédastique**: $y = 2x + \epsilon$ avec $\epsilon \sim \mathcal{N}(0, 1)$
   - **Hétéroscédastique**: $y = 2x + \epsilon$ avec $\epsilon \sim \mathcal{N}(0, (0.3x)^2)$

2. Visualisez les deux jeux de données. Quelle différence observez-vous?

3. Ajustez un modèle linéaire sur chaque jeu. Les coefficients sont-ils similaires?

4. Tracez les résidus $r_i = y_i - \hat{y}_i$ en fonction de $x_i$ pour les deux cas. Que remarquez-vous?

5. Pourquoi l'hétéroscédasticité peut-elle être problématique pour l'inférence statistique (intervalles de confiance, tests)?
````

````{admonition} Solution Exercice 3
:class: dropdown

1. **Génération des données:**

   ```python
   np.random.seed(42)
   x = np.random.uniform(0, 10, 100)
   
   # Homoscédastique
   y_homo = 2*x + np.random.normal(0, 1, 100)
   
   # Hétéroscédastique
   y_hetero = 2*x + np.random.normal(0, 0.3*x, 100)
   ```

2. **Visualisation:**

   Dans le cas homoscédastique, les points sont dispersés uniformément autour de la droite sur toute la plage de $x$. Dans le cas hétéroscédastique, la dispersion augmente avec $x$: les points sont serrés près de $x = 0$ et très dispersés pour les grandes valeurs de $x$.

3. **Coefficients:**

   Les coefficients MCO sont similaires dans les deux cas (proches de $\theta_1 = 2$, $\theta_0 = 0$). MCO reste non biaisé sous hétéroscédasticité, mais n'est plus optimal (pas de variance minimale).

4. **Résidus:**

   - **Homoscédastique**: les résidus sont répartis uniformément autour de zéro, avec une dispersion constante.
   - **Hétéroscédastique**: les résidus montrent un «cône» ou «éventail» (*fan shape*): la dispersion augmente avec $x$. C'est le signe classique d'hétéroscédasticité.

5. **Problèmes d'inférence:**

   - Les **erreurs standard** des coefficients sont incorrectes: elles supposent une variance constante.
   - Les **intervalles de confiance** et **tests t** ne sont pas valides.
   - Les **tests de significativité** peuvent être trop optimistes ou trop pessimistes.
   - Solution: utiliser des erreurs standard robustes (Huber-White) ou des moindres carrés pondérés.
````

````{admonition} Exercice 4: SVD et facteurs de rétrécissement ★★★
:class: hint dropdown

La décomposition en valeurs singulières (SVD) de $\mathbf{X}$ révèle pourquoi Ridge «rétrécit» les coefficients de manière différenciée.

1. Pour la matrice de données suivante, calculez la SVD $\mathbf{X} = \mathbf{U} \mathbf{D} \mathbf{V}^\top$:

   $$
   \mathbf{X} = \begin{pmatrix} 2 & 1 \\ 2 & 2 \\ 2 & 3 \end{pmatrix}
   $$

2. Vérifiez que $\mathbf{X}^\top \mathbf{X} = \mathbf{V} \mathbf{D}^2 \mathbf{V}^\top$ (les colonnes de $\mathbf{V}$ sont les vecteurs propres de $\mathbf{X}^\top \mathbf{X}$).

3. Pour $\lambda = 1$, calculez les facteurs de rétrécissement $s_j = \frac{d_j^2}{d_j^2 + \lambda}$ pour chaque direction $j$.

4. Expliquez pourquoi la direction avec la plus petite valeur singulière est plus fortement rétrécée.

5. Tracez les facteurs de rétrécissement $s_1$ et $s_2$ en fonction de $\lambda$ pour $\lambda \in [0, 10]$.
````

````{admonition} Solution Exercice 4
:class: dropdown

1. **SVD:**

   ```python
   X = np.array([[2, 1], [2, 2], [2, 3]])
   U, d, Vt = np.linalg.svd(X, full_matrices=False)
   V = Vt.T
   D = np.diag(d)
   ```
   
   Résultat (approximatif):
   - $d_1 \approx 4.58$, $d_2 \approx 0.77$
   - $\mathbf{v}_1 \approx [0.58, 0.82]^\top$, $\mathbf{v}_2 \approx [0.82, -0.58]^\top$

2. **Vérification:**

   ```python
   XtX = X.T @ X
   VD2Vt = V @ D**2 @ V.T
   np.allclose(XtX, VD2Vt)  # True
   ```
   
   On peut aussi vérifier que les valeurs propres de $\mathbf{X}^\top \mathbf{X}$ sont $d_1^2 \approx 21$ et $d_2^2 \approx 0.6$.

3. **Facteurs de rétrécissement pour λ = 1:**

   $$
   s_1 = \frac{d_1^2}{d_1^2 + 1} = \frac{21}{22} \approx 0.95
   $$
   
   $$
   s_2 = \frac{d_2^2}{d_2^2 + 1} = \frac{0.6}{1.6} \approx 0.37
   $$

4. **Explication:**

   La direction 2 a une petite valeur singulière ($d_2 \approx 0.77$), ce qui signifie que les données varient peu dans cette direction. L'information est donc «faible» et potentiellement bruitée. Ridge pénalise plus fortement cette direction ($s_2 = 0.37$ vs $s_1 = 0.95$) pour éviter d'ajuster le bruit.
   
   En termes de conditionnement: le rapport $d_1/d_2 \approx 6$ indique que la matrice est mal conditionnée. Ridge améliore ce conditionnement en réduisant l'effet des petites valeurs singulières.

5. **Visualisation:**

   ```python
   lambdas = np.linspace(0, 10, 100)
   s1 = d[0]**2 / (d[0]**2 + lambdas)
   s2 = d[1]**2 / (d[1]**2 + lambdas)
   
   plt.plot(lambdas, s1, label=f's1 (d1={d[0]:.2f})')
   plt.plot(lambdas, s2, label=f's2 (d2={d[1]:.2f})')
   plt.xlabel('λ')
   plt.ylabel('Facteur de rétrécissement')
   plt.legend()
   ```
   
   On observe que $s_1$ reste proche de 1 même pour $\lambda$ modéré, tandis que $s_2$ décroît rapidement. C'est le rétrécissement différencié de Ridge.
````

````{admonition} Exercice 5: Conditionnement et stabilité numérique ★★★
:class: hint dropdown

Le **nombre de conditionnement** $\kappa(\mathbf{A}) = \|\mathbf{A}\| \cdot \|\mathbf{A}^{-1}\|$ mesure la sensibilité de la solution d'un système linéaire aux perturbations.

Pour une matrice symétrique définie positive, $\kappa(\mathbf{A}) = \lambda_{\max} / \lambda_{\min}$ où $\lambda$ sont les valeurs propres.

1. Calculez le nombre de conditionnement de $\mathbf{A} = \mathbf{X}^\top \mathbf{X}$ pour:

   $$
   \mathbf{X} = \begin{pmatrix} 1 & 1 \\ 1 & 1.001 \\ 1 & 0.999 \end{pmatrix}
   $$

2. Résolvez le système $\mathbf{A} \boldsymbol{\theta} = \mathbf{X}^\top \mathbf{y}$ pour $\mathbf{y} = [1, 2, 3]^\top$.

3. Perturbez légèrement $\mathbf{y}$ en $\mathbf{y}' = [1.01, 2, 3]^\top$ et résolvez à nouveau. Comment change la solution?

4. Montrez que pour Ridge, $\kappa(\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}) = \frac{\lambda_{\max} + \lambda}{\lambda_{\min} + \lambda}$.

5. Calculez le conditionnement de la matrice Ridge pour $\lambda = 0.1$. Comparez avec le cas MCO.
````

````{admonition} Solution Exercice 5
:class: dropdown

1. **Conditionnement de X'X:**

   ```python
   X = np.array([[1, 1], [1, 1.001], [1, 0.999]])
   A = X.T @ X
   eigvals = np.linalg.eigvalsh(A)
   kappa = eigvals.max() / eigvals.min()
   print(f"Conditionnement: {kappa:.0f}")
   ```
   
   Le conditionnement est très élevé (de l'ordre de $10^6$) car les colonnes sont presque colinéaires.

2. **Solution MCO:**

   ```python
   y = np.array([1, 2, 3])
   theta = np.linalg.solve(A, X.T @ y)
   ```
   
   La solution peut être numériquement instable.

3. **Perturbation:**

   ```python
   y_perturb = np.array([1.01, 2, 3])  # 1% de perturbation sur y[0]
   theta_perturb = np.linalg.solve(A, X.T @ y_perturb)
   print(f"Changement: {np.linalg.norm(theta_perturb - theta)}")
   ```
   
   Une perturbation de 1% sur $\mathbf{y}$ peut causer un changement de plusieurs centaines de % sur $\boldsymbol{\theta}$. C'est le signe d'un système mal conditionné.

4. **Conditionnement Ridge:**

   La matrice Ridge est $\mathbf{A} + \lambda \mathbf{I}$. Ses valeurs propres sont $\lambda_j + \lambda$ (où $\lambda_j$ sont les valeurs propres de $\mathbf{A}$).
   
   $$
   \kappa(\mathbf{A} + \lambda \mathbf{I}) = \frac{\max_j(\lambda_j + \lambda)}{\min_j(\lambda_j + \lambda)} = \frac{\lambda_{\max} + \lambda}{\lambda_{\min} + \lambda}
   $$
   
   Pour $\lambda > 0$, le numérateur et le dénominateur sont tous deux augmentés, mais le dénominateur relativement plus (puisque $\lambda_{\min}$ est petit). Le conditionnement diminue.

5. **Comparaison:**

   ```python
   lambda_reg = 0.1
   A_ridge = A + lambda_reg * np.eye(2)
   eigvals_ridge = np.linalg.eigvalsh(A_ridge)
   kappa_ridge = eigvals_ridge.max() / eigvals_ridge.min()
   print(f"Conditionnement MCO: {kappa:.0f}")
   print(f"Conditionnement Ridge: {kappa_ridge:.0f}")
   ```
   
   Le conditionnement Ridge est beaucoup plus faible (quelques dizaines au lieu de millions), ce qui rend le système numériquement stable.
````
