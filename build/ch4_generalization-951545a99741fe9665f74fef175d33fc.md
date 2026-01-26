---
kernelspec:
  name: python3
  display_name: Python 3
---

# Généralisation et sélection de modèles

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Distinguer trois familles de modèles: linéaires, à expansion de caractéristiques, et réseaux de neurones
- Comprendre comment l'expansion de caractéristiques transforme un problème non linéaire en problème linéaire
- Dériver et interpréter la décomposition biais-variance
- Utiliser la validation croisée pour choisir les hyperparamètres
- Expliquer pourquoi les fonctions de perte de substitution sont nécessaires en classification
```

Les chapitres précédents ont développé les deux modèles linéaires fondamentaux: la [régression](ch2_linear_regression.md) (MCO, Ridge) et la [classification](ch3_classification.md) (régression logistique, entropie croisée). Mais un problème fondamental demeure: comment savoir si notre modèle généralisera bien à de nouvelles données? Ce chapitre explore cette question à travers le **compromis biais-variance**, les techniques d'**expansion de caractéristiques**, et les méthodes de **sélection de modèle**.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Configuration pour des figures haute résolution
%config InlineBackend.figure_format = 'retina'
```

## Classes de modèles et expansion de caractéristiques

Les exemples de régularisation du chapitre précédent utilisaient des **caractéristiques polynomiales**: au lieu de prédire $y$ directement à partir de $x$, nous avons construit des caractéristiques $[1, x, x^2, \ldots, x^{15}]$ et appliqué un modèle linéaire dans cet espace étendu. Cette technique s'appelle l'**expansion de caractéristiques** et mérite d'être formalisée.

### Trois familles de modèles

Situons les modèles linéaires dans une hiérarchie plus large. Nous distinguons trois familles de complexité croissante:

1. **Modèles linéaires**: $f(\mathbf{x}; \boldsymbol{\theta}) = \boldsymbol{\theta}^\top \mathbf{x} + b$. La sortie est une combinaison linéaire des entrées. Simple, interprétable, mais limité aux relations linéaires.

2. **Modèles à expansion de caractéristiques**: $f(\mathbf{x}; \boldsymbol{\theta}) = \boldsymbol{\theta}^\top \boldsymbol{\phi}(\mathbf{x}) + b$, où $\boldsymbol{\phi}: \mathbb{R}^d \to \mathbb{R}^D$ est une transformation non linéaire fixée à l'avance (par exemple, polynomiale). Le modèle reste linéaire dans les paramètres $\boldsymbol{\theta}$, ce qui facilite l'optimisation, mais peut capturer des relations non linéaires en $\mathbf{x}$. L'espace de redescription a souvent une dimension $D \gg d$.

3. **Réseaux de neurones**: $f(\mathbf{x}; \boldsymbol{\theta}) = f_K(f_{K-1}(\cdots f_1(\mathbf{x}; \boldsymbol{\theta}_1); \boldsymbol{\theta}_{K-1}); \boldsymbol{\theta}_K)$. Une composition de $K$ fonctions non linéaires, chacune avec ses propres paramètres. Contrairement aux modèles à expansion fixe, les réseaux de neurones **apprennent la représentation** $\boldsymbol{\phi}$ en même temps que les paramètres $\boldsymbol{\theta}$.

Cette progression capture l'évolution historique du domaine: des modèles linéaires classiques aux méthodes à noyaux (expansion implicite), puis aux réseaux profonds qui apprennent leurs propres représentations. Nous verrons les réseaux de neurones en détail dans les chapitres suivants; concentrons-nous ici sur les deux premières familles.

### Expansion de caractéristiques

Pour capturer des relations non linéaires tout en gardant un modèle linéaire dans les paramètres, nous transformons les entrées. En **régression polynomiale**, nous appliquons une fonction $\phi: \mathbb{R} \to \mathbb{R}^{k+1}$:

$$
\phi(x) = [1, x, x^2, \ldots, x^k]
$$

La prédiction devient $f(x; \boldsymbol{\theta}) = \boldsymbol{\theta}^\top \phi(x)$. Le modèle est polynomial en $x$ mais linéaire en $\boldsymbol{\theta}$, ce qui permet d'utiliser les mêmes algorithmes d'optimisation (MCO, Ridge).

Le degré $k$ contrôle la **capacité** du modèle. Avec $k = 1$, nous avons une droite. Avec $k$ élevé, le polynôme peut osciller pour passer par tous les points d'entraînement. Avec $k = N - 1$, nous pouvons interpoler exactement les $N$ points: le risque empirique atteint zéro. Mais un polynôme qui passe exactement par les points d'entraînement n'a aucune raison de bien prédire les nouveaux points.

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
    ax.scatter(speed_train, dist_train, alpha=0.6, s=30, label='Entraînement')
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
    ax.set_title(f'Degré {deg}: Entr. EQM={mse_train:.1f}, Test EQM={mse_test:.1f}')
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
ax.set_ylabel('EQM (échelle log)')
ax.set_xticks(range(1, 16, 2))
ax.grid(True, which="both", ls="-", alpha=0.2)
ax.legend()

ax.set_title('Compromis biais-variance')
plt.tight_layout()
```

L'erreur d'entraînement diminue avec le degré du polynôme. L'erreur de test diminue d'abord (quand le modèle gagne en expressivité), puis augmente (quand le modèle commence à mémoriser le bruit). Le meilleur modèle se trouve à l'intersection de ces deux tendances. La régularisation Ridge, vue précédemment, est une alternative au choix du degré: elle permet d'utiliser un modèle de haute capacité tout en contrôlant le surapprentissage.

## Décomposition biais-variance

Ce compromis peut être formalisé mathématiquement. Supposons que les données suivent le modèle $y = f^*(\mathbf{x}) + \epsilon$, où $f^*$ est la vraie fonction (le prédicteur de Bayes optimal pour la perte quadratique), et $\epsilon$ est un bruit de moyenne nulle et de variance $\sigma^2$.

**Clarification sur les variables aléatoires**: Dans cette analyse, il y a trois sources d'aléa qu'il faut bien distinguer:

- **$f^*(\mathbf{x})$** est une valeur **déterministe** (fixe) pour chaque $\mathbf{x}$. C'est la vraie fonction sous-jacente, qui ne dépend d'aucun tirage aléatoire.
- **$\epsilon$** est une variable aléatoire représentant le bruit d'observation. C'est l'aléa intrinsèque aux données.
- **$\hat{f}$** est notre estimateur, une **fonction apprise** à partir de l'échantillon d'entraînement $\mathcal{D}$. Comme $\mathcal{D}$ est tiré aléatoirement, différents tirages de $\mathcal{D}$ produisent différentes fonctions $\hat{f}$. Pour un point test $\mathbf{x}$ fixé, la **prédiction** $\hat{f}(\mathbf{x})$ est donc une variable aléatoire (un scalaire qui varie selon $\mathcal{D}$), même si $\hat{f}$ elle-même est une fonction.

Autrement dit, l'espérance $\mathbb{E}[\hat{f}(\mathbf{x})]$ moyenne sur tous les échantillons d'entraînement possibles: si nous pouvions répéter l'expérience d'apprentissage un grand nombre de fois avec différents $\mathcal{D}$, quelle serait la prédiction moyenne pour ce $\mathbf{x}$?

### Dérivation mathématique

L'erreur quadratique moyenne, moyennée sur les échantillons d'entraînement possibles et le bruit, se décompose comme suit:

$$
\mathbb{E}_{\mathcal{D}, \epsilon}[(\hat{f}(\mathbf{x}) - y)^2] = \mathbb{E}[(\hat{f}(\mathbf{x}) - f^*(\mathbf{x}) - \epsilon)^2]
$$

En développant le carré et utilisant $\mathbb{E}[\epsilon] = 0$ ainsi que l'indépendance entre $\epsilon$ et $\hat{f}$:

$$
= \mathbb{E}[(\hat{f}(\mathbf{x}) - f^*(\mathbf{x}))^2] + \sigma^2
$$

Pour le premier terme, ajoutons et retranchons $\mathbb{E}[\hat{f}(\mathbf{x})]$:

$$
\mathbb{E}[(\hat{f}(\mathbf{x}) - f^*(\mathbf{x}))^2] = \mathbb{E}[(\hat{f}(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})] + \mathbb{E}[\hat{f}(\mathbf{x})] - f^*(\mathbf{x}))^2]
$$

En développant et utilisant $\mathbb{E}[\hat{f}(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})]] = 0$:

$$
= \underbrace{\mathbb{E}[(\hat{f}(\mathbf{x}) - \mathbb{E}[\hat{f}(\mathbf{x})])^2]}_{\text{Var}(\hat{f}(\mathbf{x}))} + \underbrace{(\mathbb{E}[\hat{f}(\mathbf{x})] - f^*(\mathbf{x}))^2}_{\text{Biais}^2(\hat{f}(\mathbf{x}))}
$$

Nous obtenons la **décomposition biais-variance**:

$$
\boxed{\mathbb{E}[(\hat{f}(\mathbf{x}) - y)^2] = \text{Biais}^2(\hat{f}(\mathbf{x})) + \text{Var}(\hat{f}(\mathbf{x})) + \sigma^2}
$$

### Interprétation des termes

Chaque terme a une interprétation précise:

- **Biais²**: L'écart entre la prédiction moyenne de notre estimateur et la vraie fonction $f^*$. Un modèle trop simple (classe $\mathcal{H}$ trop restrictive) aura un biais élevé car il ne peut pas approcher $f^*$.

- **Variance**: La sensibilité de notre estimateur à l'échantillon d'entraînement particulier. Un modèle trop complexe aura une variance élevée car de petites variations dans les données causent de grandes variations dans les prédictions.

- **$\sigma^2$**: Le bruit irréductible, inhérent aux données. Aucun estimateur ne peut faire mieux que cette erreur.

Cette décomposition explique pourquoi l'erreur de test a une forme en U: à faible complexité, le biais domine; à haute complexité, la variance domine. Le minimum se trouve au point où la somme des deux est minimale.

### Lien avec Ridge

Cette décomposition éclaire aussi pourquoi Ridge fonctionne. Rappelons que Ridge rétrécit les coefficients vers zéro, avec un rétrécissement différencié selon les valeurs singulières:

- **Effet sur le biais**: En rétrécissant, Ridge introduit un **biais** (l'estimé s'éloigne de la vraie valeur)
- **Effet sur la variance**: En rétrécissant, Ridge **réduit la variance** (l'estimé devient moins sensible aux fluctuations des données)

Le coefficient $\lambda$ contrôle ce compromis. Un $\lambda$ trop petit n'aide pas assez (variance élevée). Un $\lambda$ trop grand introduit trop de biais. Le $\lambda$ optimal minimise l'erreur totale.

## Intuition géométrique: pourquoi la dimension supérieure aide

L'expansion de caractéristiques semble être un simple changement de variables, mais elle cache une idée géométrique profonde. Pour comprendre pourquoi projeter les données dans un espace de dimension supérieure permet de capturer des relations non linéaires, examinons d'abord le cas de la régression, puis celui de la classification.

### Le plan caché derrière la parabole

Considérons une régression quadratique: $f(x) = \theta_0 + \theta_1 x + \theta_2 x^2$. Cette fonction est **non linéaire en $x$** (c'est une parabole), mais **linéaire dans les paramètres** $\boldsymbol{\theta} = (\theta_0, \theta_1, \theta_2)$. Que signifie cette distinction géométriquement?

Introduisons l'espace des caractéristiques $\phi(x) = (1, x, x^2)$. Chaque valeur de $x$ correspond à un point dans $\mathbb{R}^3$. Ces points ne sont pas dispersés arbitrairement: ils vivent sur une courbe particulière, la **courbe des moments** (*moment curve*), qui ressemble à une rampe tordue.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Créer la courbe des moments: (x, x², x³) pour visualisation
# On utilise (1, x, x²) mais on projette sur (x, x², y) pour la visualisation
fig = plt.figure(figsize=(12, 5))

# Gauche: les fonctions de base
ax1 = fig.add_subplot(121)
x = np.linspace(-2, 2, 100)
ax1.plot(x, np.ones_like(x), 'b-', linewidth=2, label=r'$\phi_0(x) = 1$')
ax1.plot(x, x, 'orange', linewidth=2, label=r'$\phi_1(x) = x$')
ax1.plot(x, x**2, 'g-', linewidth=2, label=r'$\phi_2(x) = x^2$')
ax1.axhline(0, color='gray', linewidth=0.5)
ax1.axvline(0, color='gray', linewidth=0.5)
ax1.set_xlabel('$x$')
ax1.set_ylabel(r'$\phi_j(x)$')
ax1.set_title('Les fonctions de base')
ax1.legend()
ax1.set_ylim(-2.5, 4.5)
ax1.grid(True, alpha=0.3)

# Droite: combinaisons linéaires
ax2 = fig.add_subplot(122)
x = np.linspace(-2, 2, 100)

# Différentes combinaisons de coefficients
combinations = [
    ((1, 0, 0), 'Constante: $1$'),
    ((0, 1, 0), 'Linéaire: $x$'),
    ((0, 0, 1), 'Quadratique: $x^2$'),
    ((1, -0.5, 0.5), 'Combinaison: $1 - 0.5x + 0.5x^2$'),
]

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for (theta0, theta1, theta2), label in combinations:
    y = theta0 + theta1 * x + theta2 * x**2
    ax2.plot(x, y, linewidth=2, label=label)

ax2.axhline(0, color='gray', linewidth=0.5)
ax2.axvline(0, color='gray', linewidth=0.5)
ax2.set_xlabel('$x$')
ax2.set_ylabel('$f(x)$')
ax2.set_title('Combinaisons linéaires des fonctions de base')
ax2.legend(loc='upper center')
ax2.set_ylim(-2.5, 4.5)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
```

Les fonctions de base $\{1, x, x^2\}$ sont les "ingrédients" du modèle. La régression polynomiale cherche les coefficients $\theta_0, \theta_1, \theta_2$ qui mélangent ces ingrédients de façon optimale. Chaque combinaison produit une courbe différente, mais toutes sont des paraboles (ou des cas dégénérés: droites, constantes).

Voici l'insight géométrique clé: dans l'espace $(x, x^2, y)$, le modèle $y = \theta_0 + \theta_1 x + \theta_2 x^2$ définit un **plan**. La parabole que nous voyons dans le graphique $(x, y)$ est simplement la **projection** de ce plan sur notre espace de visualisation.

### De 1D à 2D: séparer l'inséparable

Considérons des points sur une droite, répartis en deux classes: les points bleus au centre, les points orange aux extrémités. Aucun seuil unique ne peut séparer ces deux classes.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Classe bleue: points au centre
x_blue = np.random.uniform(-0.5, 0.5, 15)
# Classe orange: points aux extrémités
x_orange = np.concatenate([np.random.uniform(-1.5, -0.8, 8), 
                           np.random.uniform(0.8, 1.5, 8)])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Gauche: espace original avec tentative de séparation
ax = axes[0]
ax.scatter(x_blue, np.zeros_like(x_blue), c='tab:blue', s=80, zorder=3)
ax.scatter(x_orange, np.zeros_like(x_orange), c='tab:orange', s=80, zorder=3)
ax.axhline(0, color='gray', linewidth=0.5, zorder=1)
ax.axvline(0.3, color='red', linestyle='--', linewidth=2, label='Seuil?')
ax.set_xlim(-2, 2)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel('$x$')
ax.set_yticks([])
ax.set_title('Espace original: pas de séparation linéaire')
ax.legend()

# Droite: espace transformé
ax = axes[1]
ax.scatter(x_blue, x_blue**2, c='tab:blue', s=80, zorder=3, label='Classe A')
ax.scatter(x_orange, x_orange**2, c='tab:orange', s=80, zorder=3, label='Classe B')

# Ligne de séparation dans l'espace transformé
x_line = np.linspace(-2, 2, 100)
threshold = 0.6
ax.axhline(threshold, color='green', linestyle='-', linewidth=2, label='Frontière linéaire')
ax.fill_between(x_line, 0, threshold, alpha=0.1, color='blue')
ax.fill_between(x_line, threshold, 2.5, alpha=0.1, color='orange')

# Parabole de référence
x_curve = np.linspace(-1.6, 1.6, 100)
ax.plot(x_curve, x_curve**2, 'k--', alpha=0.3, linewidth=1)

ax.set_xlim(-2, 2)
ax.set_ylim(-0.1, 2.5)
ax.set_xlabel('$x$')
ax.set_ylabel('$x^2$')
ax.set_title(r'Espace transformé $\phi(x) = (x, x^2)$: séparation linéaire!')
ax.legend()

plt.tight_layout()
```

Dans l'espace transformé, une simple droite horizontale sépare les deux classes. Cette droite correspond, dans l'espace original, à **deux seuils**: $x^2 < 0.6$, soit $|x| < \sqrt{0.6} \approx 0.77$. L'expansion de caractéristiques a transformé une frontière de décision non linéaire (un intervalle) en une frontière linéaire (une droite).

### De 2D à 3D: soulever pour séparer

Passons à un exemple plus visuel. Considérons deux classes disposées en cercles concentriques: la classe bleue forme un disque central, la classe orange forme un anneau extérieur. Aucune droite ne peut séparer ces deux régions.

Appliquons l'expansion $\phi(x_1, x_2) = (x_1, x_2, x_1^2 + x_2^2)$. La troisième coordonnée est le carré de la distance à l'origine: $z = r^2$. Les points proches du centre (petit $r$) sont "soulevés" moins haut que les points éloignés (grand $r$).

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

# Classe bleue: disque central
n_blue = 50
r_blue = np.random.uniform(0, 0.7, n_blue)
theta_blue = np.random.uniform(0, 2*np.pi, n_blue)
x_blue = r_blue * np.cos(theta_blue)
y_blue = r_blue * np.sin(theta_blue)

# Classe orange: anneau extérieur
n_orange = 70
r_orange = np.random.uniform(1.0, 1.5, n_orange)
theta_orange = np.random.uniform(0, 2*np.pi, n_orange)
x_orange = r_orange * np.cos(theta_orange)
y_orange = r_orange * np.sin(theta_orange)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Gauche: vue 2D originale
ax = axes[0]
ax.scatter(x_blue, y_blue, c='tab:blue', s=40, alpha=0.7, label='Classe A')
ax.scatter(x_orange, y_orange, c='tab:orange', s=40, alpha=0.7, label='Classe B')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_aspect('equal')
ax.legend()
ax.set_title('Cercles concentriques: pas de séparation linéaire en 2D')

# Droite: vue 2D avec frontière circulaire
ax = axes[1]
ax.scatter(x_blue, y_blue, c='tab:blue', s=40, alpha=0.7, label='Classe A')
ax.scatter(x_orange, y_orange, c='tab:orange', s=40, alpha=0.7, label='Classe B')

# Cercle de décision (projection du plan z = 0.75)
theta_circle = np.linspace(0, 2*np.pi, 100)
r_decision = np.sqrt(0.75)
ax.plot(r_decision * np.cos(theta_circle), r_decision * np.sin(theta_circle), 
        'g-', linewidth=2.5, label=f'Frontière: $r = {r_decision:.2f}$')

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_aspect('equal')
ax.legend()
ax.set_title('Frontière de décision projetée en 2D')

plt.tight_layout()
```

Dans l'espace 3D $(x_1, x_2, x_1^2 + x_2^2)$, un plan horizontal $z = 0.75$ sépare les deux classes. Ce plan correspond, dans l'espace original 2D, à un **cercle** de rayon $\sqrt{0.75} \approx 0.87$. La frontière de décision linéaire en 3D devient une frontière circulaire en 2D.

### Le principe unificateur

Ces exemples de régression et de classification illustrent le même principe géométrique:

| | Régression | Classification |
|---|---|---|
| **Objectif** | Ajuster les données | Séparer les classes |
| **Dans l'espace original** | Courbe (parabole, etc.) | Frontière courbe (cercle, etc.) |
| **Dans l'espace des caractéristiques** | Hyperplan d'ajustement | Hyperplan séparateur |
| **La courbe/frontière est...** | L'intersection du plan avec la surface $\phi(x)$ | La projection de l'hyperplan |

L'expansion de caractéristiques transforme un problème non linéaire en un problème linéaire dans un espace de dimension supérieure. Les modèles linéaires, simples à optimiser et à analyser, deviennent alors suffisants pour capturer des structures complexes.

En augmentant la dimension de l'espace de représentation, nous augmentons la **capacité** du modèle. Mais cette flexibilité a un coût: plus l'espace est grand, plus le modèle risque de mémoriser les particularités des données d'entraînement plutôt que d'apprendre la structure sous-jacente. C'est le **compromis biais-variance**, et c'est pourquoi la régularisation est si importante pour les modèles à haute capacité.

## Évaluation et choix de modèle

En pratique, nous estimons le risque par le risque empirique sur un **ensemble de test** $\mathcal{D}_{\text{test}}$ disjoint de l'ensemble d'entraînement. Un troisième ensemble, l'**ensemble de validation**, sert à choisir parmi plusieurs modèles ou à régler des hyperparamètres. L'ensemble de test doit rester intact jusqu'à l'évaluation finale, pour fournir une estimation non biaisée.

Cette séparation est importante. Si nous utilisons l'ensemble de test pour faire des choix (quel modèle garder, quelle valeur d'hyperparamètre utiliser), l'estimation de performance sur ce même ensemble devient optimiste.

### Hyperparamètres et validation

De nombreux modèles ont des **hyperparamètres**: des choix qui doivent être faits avant l'entraînement et qui ne sont pas appris à partir des données. Le degré $k$ d'un polynôme, le nombre de voisins dans les $k$ plus proches voisins, ou le coefficient de régularisation $\lambda$ sont des exemples d'hyperparamètres.

Un hyperparamètre mal choisi peut mener au surapprentissage (modèle trop complexe) ou au sous-apprentissage (modèle trop simple). La méthode standard pour choisir un hyperparamètre est la **validation**: on réserve une partie des données (typiquement 20%) comme ensemble de validation, on entraîne le modèle pour plusieurs valeurs de l'hyperparamètre, et on retient celle qui minimise l'erreur sur l'ensemble de validation.

Plus formellement, pour un hyperparamètre $h$, définissons le **risque de validation**:

$$
\hat{\mathcal{R}}^{\text{val}}_h = \hat{\mathcal{R}}\left(\hat{f}_h(\mathcal{D}_{\text{train}}), \mathcal{D}_{\text{valid}}\right)
$$

où $\hat{f}_h(\mathcal{D}_{\text{train}})$ est le modèle entraîné avec l'hyperparamètre $h$. La **recherche par grille** consiste à évaluer ce risque pour un ensemble de valeurs candidates et à retenir:

$$
h^* = \arg\min_{h \in \{h_1, \ldots, h_K\}} \hat{\mathcal{R}}^{\text{val}}_h
$$

Une fois $h^*$ choisi, on peut ré-entraîner le modèle sur l'ensemble des données (entraînement + validation) pour obtenir le modèle final.

### Validation croisée

Quand les données sont peu nombreuses, réserver 20% pour la validation peut être coûteux. La **validation croisée** offre une alternative.

L'idée est de partitionner les données en $K$ **blocs**. Pour chaque bloc $k$, on entraîne le modèle sur les $K-1$ autres blocs et on évalue sur le bloc $k$. Le **risque de validation croisée** est la moyenne des $K$ évaluations:

$$
\hat{\mathcal{R}}^{\text{cv}}_h = \frac{1}{K} \sum_{k=1}^K \hat{\mathcal{R}}\left(\hat{f}_h(\mathcal{D}_{-k}), \mathcal{D}_k\right)
$$

où $\mathcal{D}_{-k}$ désigne toutes les données sauf le bloc $k$.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 3))

K = 5
for fold in range(K):
    for k in range(K):
        if k == fold:
            ax.barh(fold, 1, left=k, color='C1', edgecolor='black', linewidth=1, label='Validation' if fold == 0 else '')
        else:
            ax.barh(fold, 1, left=k, color='C0', edgecolor='black', linewidth=1, label='Entraînement' if fold == 0 and k == 0 else '')

ax.set_yticks(range(K))
ax.set_yticklabels([f'Itération {k+1}' for k in range(K)])
ax.set_xticks(np.arange(K) + 0.5)
ax.set_xticklabels([f'Bloc {k+1}' for k in range(K)])
ax.set_xlabel('Blocs de données')
ax.legend(loc='upper right')
ax.set_title(f'Validation croisée à {K} blocs')
ax.set_xlim(0, K)

plt.tight_layout()
```

Le cas particulier $K = N$ (un bloc par exemple) est appelé **validation croisée leave-one-out**. Elle utilise au maximum les données disponibles, mais est coûteuse en calcul. En pratique, $K = 5$ ou $K = 10$ offrent un bon compromis.

```{admonition} Mise en garde: la fuite d'information
:class: warning

Les outils modernes de génération de code peuvent produire des pipelines d'apprentissage automatique complets en quelques minutes. Mais ces pipelines peuvent contenir des erreurs subtiles qui mènent à des résultats trop beaux pour être vrais.

Un exemple: un praticien utilise un assistant de programmation pour construire un modèle prédictif. L'erreur d'entraînement passe de 0.20 à 0.01 en quelques itérations. Mais en examinant le code, il découvre que le modèle utilise des caractéristiques qui ne seraient pas disponibles au moment du déploiement.

Ce phénomène s'appelle la **fuite d'information**. Le modèle ne généralise pas: il triche. Les métriques d'entraînement sont excellentes, mais le modèle échouera en déploiement.

**Votre rôle**: auditer les pipelines, vérifier que les caractéristiques utilisées seront disponibles en production, et maintenir une séparation stricte entre les données d'entraînement et de test.
```

### Biais inductifs

Il n'existe pas de modèle universel qui fonctionne optimalement pour tous les problèmes. Ce résultat, connu sous le nom de **théorème du no free lunch**, affirme qu'un algorithme d'apprentissage qui performe bien sur une classe de problèmes performe nécessairement moins bien sur d'autres.

Tout modèle encode des **biais inductifs**: des hypothèses implicites ou explicites sur la structure du problème. La régression linéaire suppose que la relation entre entrées et sorties est linéaire. Les k plus proches voisins supposent que les points proches dans l'espace des entrées ont des sorties similaires. Les modèles plus complexes, comme les réseaux de neurones, encodent d'autres hypothèses sur la structure des données.

Ces hypothèses sont nécessaires pour que l'apprentissage soit possible. Sans elles, nous n'aurions aucune raison de croire que la performance sur l'échantillon d'entraînement prédit la performance sur de nouvelles données. Le choix du modèle et de ses hypothèses est une décision que l'algorithme ne peut pas prendre seul; elle requiert une connaissance du domaine.

## Fonctions de perte de substitution

La perte 0-1 pose un problème pratique. Les méthodes d'optimisation itératives, comme la descente de gradient, requièrent que la fonction objectif soit différentiable. Or la perte 0-1 est constante par morceaux: sa dérivée est nulle presque partout et indéfinie aux points de discontinuité.

Nous contournons ce problème en utilisant des **fonctions de perte de substitution**: des approximations convexes et différentiables de la perte originale. L'entropie croisée, que nous avons dérivée dans le [chapitre précédent](ch3_classification.md) à partir du maximum de vraisemblance, en est un exemple central. Examinons ici une vue unifiée des pertes de substitution les plus courantes.

Pour la classification binaire, plutôt que de prédire directement une classe, les modèles produisent souvent un **score** $s = f(\mathbf{x})$ (un nombre réel). La prédiction de classe se fait ensuite en prenant le signe de ce score: si $s > 0$, on prédit la classe $+1$; si $s < 0$, on prédit la classe $-1$. La valeur absolue de $s$ mesure la confiance: plus $|s|$ est grand, plus le modèle est confiant dans sa prédiction.

Avec la convention $y \in \{-1, +1\}$ (plutôt que $\{0, 1\}$), la **perte logistique** s'écrit:

$$
\ell_{\text{log}}(y, s) = \log(1 + e^{-y \cdot s})
$$

où $s = f(\mathbf{x})$ est le score produit par le modèle. Cette formulation est équivalente à l'entropie croisée binaire du [chapitre 3](ch3_classification.md), mais avec la convention $y \in \{-1, +1\}$ plutôt que $y \in \{0, 1\}$. La fonction est convexe et différentiable partout. Lorsque $y$ et $s$ ont le même signe (prédiction correcte avec confiance), la perte est faible. Lorsqu'ils ont des signes opposés (erreur), la perte croît linéairement avec l'amplitude de l'erreur.

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

## Résumé

Ce chapitre a exploré les fondements théoriques et pratiques de la généralisation:

- L'**expansion de caractéristiques** transforme un problème non linéaire en problème linéaire dans un espace de dimension supérieure, au prix d'une capacité accrue.

- La **décomposition biais-variance** formalise le compromis: modèles simples = biais élevé, variance faible; modèles complexes = biais faible, variance élevée. L'erreur totale est minimale au point où la somme est minimale.

- La **validation croisée** permet de choisir les hyperparamètres (comme $\lambda$ ou le degré polynomial) en estimant l'erreur de généralisation sans toucher à l'ensemble de test.

- Les **fonctions de perte de substitution** (logistique, hinge) remplacent la perte 0-1 non différentiable par des approximations convexes optimisables par descente de gradient.

- Les **biais inductifs** (hypothèses sur la structure des données) sont inévitables et nécessaires à l'apprentissage.

Mais d'où viennent ces choix de fonctions de perte? Pourquoi la perte quadratique pour la régression et l'entropie croisée pour la classification? Le [chapitre suivant](ch5_probabilistic.md) répond à ces questions en présentant le **cadre probabiliste**, qui unifie ces approches via le maximum de vraisemblance et le maximum a posteriori.

## Exercices

````{admonition} Exercice 1: Expansion de caractéristiques ★
:class: hint dropdown

Considérez un problème de régression où la relation entre $x$ et $y$ est quadratique:

$$
y = 3 - 2x + 0.5x^2 + \epsilon, \quad \epsilon \sim \mathcal{N}(0, 0.5^2)
$$

1. Générez $N = 50$ points selon ce modèle pour $x \in [-2, 4]$.

2. Ajustez un modèle linéaire $y = \theta_0 + \theta_1 x$ aux données. Visualisez le résultat et calculez l'EQM.

3. Définissez la transformation $\phi(x) = [1, x, x^2]^\top$. Montrez que le problème devient linéaire en $\boldsymbol{\theta}$ dans cet espace étendu.

4. Ajustez un modèle linéaire dans l'espace étendu: $y = \theta_0 + \theta_1 x + \theta_2 x^2$. Comparez l'EQM avec le modèle précédent.

5. Que se passe-t-il si vous utilisez $\phi(x) = [1, x, x^2, x^3, x^4, x^5]$? Discutez du risque de surapprentissage.
````

````{admonition} Solution Exercice 1
:class: dropdown

1. **Génération des données:**

   ```python
   import numpy as np
   np.random.seed(42)
   x = np.random.uniform(-2, 4, 50)
   y = 3 - 2*x + 0.5*x**2 + np.random.normal(0, 0.5, 50)
   ```

2. **Modèle linéaire simple:**

   ```python
   coeffs_lin = np.polyfit(x, y, 1)
   y_pred_lin = np.polyval(coeffs_lin, x)
   mse_lin = np.mean((y - y_pred_lin)**2)
   ```
   
   L'EQM est élevé car le modèle linéaire ne peut pas capturer la courbure. Visuellement, la droite ne suit pas la tendance parabolique des données.

3. **Transformation en problème linéaire:**

   En définissant $\phi(x) = [1, x, x^2]^\top$, le modèle s'écrit:
   
   $$
   y = \boldsymbol{\theta}^\top \phi(x) = \theta_0 \cdot 1 + \theta_1 \cdot x + \theta_2 \cdot x^2
   $$
   
   C'est linéaire en $\boldsymbol{\theta} = [\theta_0, \theta_1, \theta_2]^\top$, même si c'est non-linéaire en $x$. On peut appliquer les équations normales dans cet espace.

4. **Modèle quadratique:**

   ```python
   coeffs_quad = np.polyfit(x, y, 2)
   y_pred_quad = np.polyval(coeffs_quad, x)
   mse_quad = np.mean((y - y_pred_quad)**2)
   ```
   
   L'EQM est beaucoup plus faible (proche de $\sigma^2 = 0.25$). Les coefficients récupérés sont proches des vrais: $\theta_2 \approx 0.5$, $\theta_1 \approx -2$, $\theta_0 \approx 3$.

5. **Expansion de degré 5:**

   Avec plus de termes, le modèle peut s'ajuster encore mieux aux données d'entraînement (EQM très faible), mais il risque de capturer le bruit plutôt que le signal. Sur de nouvelles données, les performances se dégradent. C'est le **surapprentissage**: le modèle a trop de capacité par rapport à la complexité réelle de la relation.
````

````{admonition} Exercice 2: Validation croisée pour le choix de λ ★★
:class: hint dropdown

La validation croisée permet de choisir l'hyperparamètre $\lambda$ de Ridge sans utiliser de données de test.

1. Générez un jeu de données de régression polynomiale ($N = 50$):

   ```python
   np.random.seed(42)
   x = np.random.uniform(-3, 3, 50)
   y = 0.5*x**3 - x**2 + 2*x + np.random.normal(0, 2, 50)
   ```

2. Créez une matrice de caractéristiques polynomiales de degré 10: $\phi(x) = [1, x, x^2, \ldots, x^{10}]$.

3. Implémentez la validation croisée à 5 plis (*5-fold CV*):
   - Divisez les données en 5 groupes
   - Pour chaque $\lambda \in \{10^{-4}, 10^{-3}, \ldots, 10^{2}\}$:
     - Entraînez sur 4 plis, évaluez sur le 5ème
     - Calculez l'EQM moyen sur les 5 plis

4. Tracez l'EQM de validation en fonction de $\log_{10}(\lambda)$. Quel $\lambda$ choisiriez-vous?

5. Comparez les performances (sur un ensemble de test séparé) de MCO ($\lambda = 0$) et de Ridge avec le $\lambda$ optimal.
````

````{admonition} Solution Exercice 2
:class: dropdown

1-2. **Génération et matrice de caractéristiques:**

   ```python
   from sklearn.preprocessing import PolynomialFeatures
   
   np.random.seed(42)
   x = np.random.uniform(-3, 3, 50)
   y = 0.5*x**3 - x**2 + 2*x + np.random.normal(0, 2, 50)
   
   poly = PolynomialFeatures(degree=10, include_bias=True)
   X = poly.fit_transform(x.reshape(-1, 1))
   ```

3. **Validation croisée:**

   ```python
   from sklearn.model_selection import KFold
   from sklearn.linear_model import Ridge
   
   lambdas = np.logspace(-4, 2, 20)
   kf = KFold(n_splits=5, shuffle=True, random_state=42)
   
   cv_scores = []
   for lam in lambdas:
       fold_scores = []
       for train_idx, val_idx in kf.split(X):
           model = Ridge(alpha=lam, fit_intercept=False)
           model.fit(X[train_idx], y[train_idx])
           y_pred = model.predict(X[val_idx])
           mse = np.mean((y[val_idx] - y_pred)**2)
           fold_scores.append(mse)
       cv_scores.append(np.mean(fold_scores))
   ```

4. **Visualisation et choix de λ:**

   ```python
   plt.plot(np.log10(lambdas), cv_scores)
   plt.xlabel('log10(λ)')
   plt.ylabel('EQM de validation')
   best_idx = np.argmin(cv_scores)
   best_lambda = lambdas[best_idx]
   ```
   
   La courbe montre typiquement:
   - EQM élevé pour $\lambda$ très petit (surapprentissage)
   - EQM minimal pour $\lambda$ intermédiaire
   - EQM qui remonte pour $\lambda$ grand (sous-apprentissage)
   
   Le $\lambda$ optimal se situe au minimum de la courbe (souvent autour de $10^{-1}$ à $10^0$).

5. **Comparaison finale:**

   MCO avec degré 10 surapprend fortement et a un EQM de test élevé. Ridge avec $\lambda$ optimal a un EQM de test beaucoup plus faible car la régularisation empêche les coefficients d'exploser.
````

````{admonition} Exercice 3: Décomposition biais-variance empirique ★★
:class: hint dropdown

Objectif: Visualiser le compromis biais-variance expérimentalement.

1. Générez 100 jeux de données indépendants à partir du modèle $y = \sin(x) + \epsilon$ où $\epsilon \sim \mathcal{N}(0, 0.3^2)$ et $x \in [0, 2\pi]$ avec $N = 20$ points par jeu.

2. Pour chaque degré polynomial $k \in \{1, 3, 5, 10, 15\}$:
   - Ajustez un polynôme sur chacun des 100 jeux de données
   - Pour un point test fixé $x_0 = \pi$, collectez les 100 prédictions $\hat{y}_0^{(1)}, \ldots, \hat{y}_0^{(100)}$

3. Calculez pour chaque degré $k$:
   - Le **biais²** : $(\bar{\hat{y}}_0 - \sin(x_0))^2$ où $\bar{\hat{y}}_0 = \frac{1}{100}\sum_{j=1}^{100} \hat{y}_0^{(j)}$
   - La **variance** : $\frac{1}{100}\sum_{j=1}^{100} (\hat{y}_0^{(j)} - \bar{\hat{y}}_0)^2$
   - L'**erreur totale** : biais² + variance

4. Tracez biais², variance, et erreur totale en fonction du degré $k$.

5. Vérifiez que le degré minimisant l'erreur totale correspond à celui qui généralise le mieux.
````

````{admonition} Solution Exercice 3
:class: dropdown

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_datasets = 100
n_points = 20
sigma = 0.3
x_test = np.pi  # point test fixé
true_value = np.sin(x_test)

degrees = [1, 3, 5, 10, 15]
bias_sq = []
variance = []

for deg in degrees:
    predictions = []
    for _ in range(n_datasets):
        # Générer un jeu de données
        x = np.random.uniform(0, 2*np.pi, n_points)
        y = np.sin(x) + np.random.normal(0, sigma, n_points)
        
        # Ajuster le polynôme
        coeffs = np.polyfit(x, y, deg)
        y_pred = np.polyval(coeffs, x_test)
        predictions.append(y_pred)
    
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions)
    
    # Biais²
    bias_sq.append((mean_pred - true_value)**2)
    
    # Variance
    variance.append(np.var(predictions))

bias_sq = np.array(bias_sq)
variance = np.array(variance)
total_error = bias_sq + variance

plt.figure(figsize=(8, 5))
plt.plot(degrees, bias_sq, 'o-', label='Biais²')
plt.plot(degrees, variance, 's-', label='Variance')
plt.plot(degrees, total_error, '^-', label='Erreur totale')
plt.axhline(sigma**2, color='gray', linestyle='--', label=f'Bruit σ²={sigma**2:.2f}')
plt.xlabel('Degré du polynôme')
plt.ylabel('Erreur')
plt.legend()
plt.title('Décomposition biais-variance')
```

**Observations:**
- Le **biais** diminue avec le degré: les modèles complexes approchent mieux $\sin(x)$
- La **variance** augmente avec le degré: plus de paramètres → plus sensible aux fluctuations
- L'**erreur totale** a une forme en U: le minimum est autour de degré 3-5
- Le degré optimal n'est ni trop simple (biais) ni trop complexe (variance)
````

````{admonition} Exercice 4: Courbes d'apprentissage ★
:class: hint dropdown

Objectif: Observer la convergence du risque empirique vers le risque.

1. Générez $N = 1000$ points de régression linéaire: $y = 2x + 3 + \epsilon$ avec $\epsilon \sim \mathcal{N}(0, 1)$ et $x \sim \mathcal{U}(0, 10)$.

2. Réservez 500 points pour le test (données fraîches, jamais utilisées pour l'entraînement).

3. Pour des tailles d'entraînement $n \in \{10, 20, 50, 100, 200, 500\}$:
   - Entraînez un modèle linéaire sur les $n$ premiers points
   - Calculez l'EQM sur l'ensemble d'entraînement (erreur train)
   - Calculez l'EQM sur l'ensemble de test (erreur test)

4. Tracez les deux courbes d'erreur en fonction de $n$.

5. À quelle taille les deux courbes se rejoignent-elles approximativement? Que signifie cet écart?
````

````{admonition} Solution Exercice 4
:class: dropdown

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 1000

# Générer toutes les données
x_all = np.random.uniform(0, 10, N)
y_all = 2*x_all + 3 + np.random.normal(0, 1, N)

# Séparer train et test
x_test, y_test = x_all[500:], y_all[500:]

train_sizes = [10, 20, 50, 100, 200, 500]
train_errors = []
test_errors = []

for n in train_sizes:
    x_train, y_train = x_all[:n], y_all[:n]
    
    # Ajuster le modèle linéaire
    X_train = np.column_stack([np.ones(n), x_train])
    theta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
    
    # Erreur d'entraînement
    y_pred_train = X_train @ theta
    train_errors.append(np.mean((y_train - y_pred_train)**2))
    
    # Erreur de test
    X_test = np.column_stack([np.ones(500), x_test])
    y_pred_test = X_test @ theta
    test_errors.append(np.mean((y_test - y_pred_test)**2))

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_errors, 'o-', label='Erreur train')
plt.plot(train_sizes, test_errors, 's-', label='Erreur test')
plt.axhline(1.0, color='gray', linestyle='--', label='Bruit σ²=1')
plt.xlabel('Taille d\'entraînement n')
plt.ylabel('EQM')
plt.legend()
plt.title('Courbes d\'apprentissage')
```

**Observations:**
- L'erreur d'entraînement **augmente** légèrement avec $n$ (moins de surapprentissage)
- L'erreur de test **diminue** avec $n$ (meilleure généralisation)
- Les courbes convergent vers $\sigma^2 = 1$ (le bruit irréductible)
- L'écart entre les deux courbes est l'**écart de généralisation**
- Avec $n \approx 100-200$, les courbes sont proches: le modèle généralise bien
````

````{admonition} Exercice 5: Choix du degré polynomial ★★
:class: hint dropdown

Objectif: Pratiquer la sélection de modèle avec un ensemble de validation.

1. Générez $N = 50$ points selon $y = 0.5x^3 - x^2 + 2x + \epsilon$ avec $\epsilon \sim \mathcal{N}(0, 2)$ et $x \in [-3, 3]$.

2. Divisez les données: 70% entraînement, 30% validation.

3. Pour chaque degré $k \in \{1, 2, 3, \ldots, 10\}$:
   - Ajustez un polynôme de degré $k$ sur l'ensemble d'entraînement
   - Calculez l'EQM sur l'ensemble de validation

4. Tracez l'EQM de validation en fonction du degré. Quel degré minimise l'erreur de validation?

5. Comparez le degré sélectionné avec le vrai degré (3). Discutez des cas où ils pourraient différer.
````

````{admonition} Solution Exercice 5
:class: dropdown

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 50
x = np.random.uniform(-3, 3, N)
y = 0.5*x**3 - x**2 + 2*x + np.random.normal(0, 2, N)

# Split train/validation
n_train = int(0.7 * N)
indices = np.random.permutation(N)
train_idx, val_idx = indices[:n_train], indices[n_train:]
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

degrees = range(1, 11)
val_errors = []

for deg in degrees:
    coeffs = np.polyfit(x_train, y_train, deg)
    y_pred_val = np.polyval(coeffs, x_val)
    val_errors.append(np.mean((y_val - y_pred_val)**2))

best_degree = degrees[np.argmin(val_errors)]

plt.figure(figsize=(8, 5))
plt.plot(degrees, val_errors, 'o-')
plt.axvline(best_degree, color='r', linestyle='--', label=f'Meilleur: {best_degree}')
plt.axvline(3, color='g', linestyle=':', label='Vrai degré: 3')
plt.xlabel('Degré du polynôme')
plt.ylabel('EQM de validation')
plt.legend()
plt.title('Sélection du degré par validation')
```

**Discussion:**
- Le degré sélectionné est souvent proche de 3, mais pas toujours exactement 3
- Avec peu de données (N=50), la variance de l'estimation est élevée
- Un degré légèrement supérieur (4 ou 5) peut être sélectionné si le bruit crée des patterns
- Un degré inférieur peut être sélectionné si les données ne montrent pas clairement la cubique
- La validation croisée (au lieu d'un simple split) réduirait cette variabilité
````

````{admonition} Exercice 6: Double descente (optionnel) ★★★
:class: hint dropdown

Objectif: Explorer le régime d'interpolation moderne.

Le phénomène de **double descente** contredit l'intuition classique: au-delà d'un certain seuil de complexité, l'erreur de test peut *diminuer* à nouveau.

1. Générez $N = 20$ points selon $y = \sin(x) + \epsilon$ avec $\epsilon \sim \mathcal{N}(0, 0.2)$ et $x$ uniformément espacé sur $[0, 2\pi]$.

2. Pour chaque degré $k \in \{1, 2, \ldots, 30\}$:
   - Ajustez un polynôme de degré $k$ (utilisez `np.polyfit` avec `full=False`)
   - Calculez l'EQM sur un ensemble de test de 100 points

3. Tracez l'EQM de test en fonction du degré.

4. Identifiez le **point d'interpolation**: le degré à partir duquel le modèle peut passer exactement par tous les points d'entraînement.

5. Observez la forme de la courbe: y a-t-il une remontée puis une redescente de l'erreur?

6. Discutez des implications pour le deep learning, où les réseaux ont souvent plus de paramètres que de données.
````

````{admonition} Solution Exercice 6
:class: dropdown

```python
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
N = 20
x_train = np.linspace(0, 2*np.pi, N)
y_train = np.sin(x_train) + np.random.normal(0, 0.2, N)

# Ensemble de test
x_test = np.linspace(0, 2*np.pi, 100)
y_test = np.sin(x_test)

degrees = range(1, 31)
test_errors = []
train_errors = []

for deg in degrees:
    try:
        coeffs = np.polyfit(x_train, y_train, deg)
        y_pred_train = np.polyval(coeffs, x_train)
        y_pred_test = np.polyval(coeffs, x_test)
        train_errors.append(np.mean((y_train - y_pred_train)**2))
        test_errors.append(np.mean((y_test - y_pred_test)**2))
    except:
        train_errors.append(np.nan)
        test_errors.append(np.nan)

plt.figure(figsize=(10, 5))
plt.semilogy(degrees, test_errors, 'o-', label='Erreur test')
plt.semilogy(degrees, train_errors, 's-', label='Erreur train', alpha=0.5)
plt.axvline(N-1, color='r', linestyle='--', label=f'Interpolation (d={N-1})')
plt.xlabel('Degré du polynôme')
plt.ylabel('EQM (échelle log)')
plt.legend()
plt.title('Phénomène de double descente')
```

**Observations:**
- **Point d'interpolation**: degré $k = N - 1 = 19$. À ce degré, le polynôme a exactement $N$ coefficients et peut passer par tous les points.
- **Forme de la courbe**:
  - Degré 1-5: erreur test diminue (gain d'expressivité)
  - Degré 5-19: erreur test augmente (surapprentissage classique)
  - Degré > 19: erreur test peut diminuer! (double descente)
- L'erreur d'entraînement atteint 0 au point d'interpolation.

**Implications pour le deep learning:**
- Les réseaux profonds sont souvent dans le régime «sur-paramétrisé» (plus de paramètres que de données)
- Contrairement à l'intuition classique, ils peuvent bien généraliser même en interpolant les données
- La régularisation implicite (SGD, early stopping) et la structure du réseau jouent un rôle clé
- Ce phénomène explique en partie pourquoi les très grands modèles peuvent surpasser les modèles de taille «optimale»
````
