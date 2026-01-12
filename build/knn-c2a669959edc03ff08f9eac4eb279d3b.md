---
kernelspec:
  name: python3
  display_name: Python 3
---

# K plus proches voisins

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Expliquer le fonctionnement de l'algorithme des k plus proches voisins
- Définir et appliquer différentes fonctions de distance
- Analyser l'effet du paramètre k sur le compromis biais-variance
- Expliquer le fléau de la dimensionnalité et ses conséquences
- Implémenter l'algorithme k-ppv pour la classification et la régression
```

## L'idée de base

Soit $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ un ensemble d'entraînement avec $x_i \in \mathbb{R}^d$ et $y_i \in \{1, \ldots, C\}$. Nous voulons prédire l'étiquette d'un nouveau point $x$. L'approche la plus simple consiste à regarder les exemples connus qui ressemblent à $x$ et à prédire la même chose.

Les k plus proches voisins (k-ppv) formalisent cette intuition. Pour classifier $x$, nous identifions les $k$ points de $\mathcal{D}$ les plus proches de $x$ et prenons un vote majoritaire sur leurs étiquettes. La méthode ne fait aucune hypothèse sur la forme de la relation entre $x$ et $y$. Elle se contente de consulter les données au moment de la prédiction.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Generate simple 2D data
np.random.seed(42)
n_per_class = 15

# Class 0: cluster around (-1, -1)
X0 = np.random.randn(n_per_class, 2) * 0.6 + np.array([-1, -1])
# Class 1: cluster around (1, 1)
X1 = np.random.randn(n_per_class, 2) * 0.6 + np.array([1, 1])

X_train = np.vstack([X0, X1])
y_train = np.array([0] * n_per_class + [1] * n_per_class)

# Query point
x_query = np.array([0.3, 0.2])
k = 5

# Compute distances and find k nearest
distances = np.sqrt(np.sum((X_train - x_query)**2, axis=1))
k_nearest_idx = np.argsort(distances)[:k]

fig, ax = plt.subplots(figsize=(6, 5))

# Plot training points
ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], 
           c='C0', s=60, label='Classe 0', zorder=2)
ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
           c='C1', s=60, label='Classe 1', zorder=2)

# Highlight k nearest neighbors
for idx in k_nearest_idx:
    ax.plot([x_query[0], X_train[idx, 0]], [x_query[1], X_train[idx, 1]], 
            'k--', alpha=0.4, linewidth=1, zorder=1)
    ax.scatter(X_train[idx, 0], X_train[idx, 1], 
               s=150, facecolors='none', edgecolors='black', linewidths=2, zorder=3)

# Plot query point
ax.scatter(x_query[0], x_query[1], c='red', s=120, marker='*', 
           label='Point requête', zorder=4)

# Count votes
votes = y_train[k_nearest_idx]
n_class0 = np.sum(votes == 0)
n_class1 = np.sum(votes == 1)
prediction = 0 if n_class0 > n_class1 else 1

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.legend(loc='upper left')
ax.set_title(f'$k = {k}$: votes = [{n_class0} classe 0, {n_class1} classe 1] → prédiction: classe {prediction}')
ax.set_aspect('equal')
plt.tight_layout()
```

Soit $\mathcal{N}_k(x)$ l'ensemble des indices des $k$ plus proches voisins de $x$. La prédiction est:

$$
\hat{y} = \arg\max_{c} \sum_{i \in \mathcal{N}_k(x)} \mathbb{1}_{y_i = c}
$$

La somme compte combien de voisins appartiennent à chaque classe $c$, et nous retenons la classe la plus fréquente.

Cette formulation admet une interprétation probabiliste. La proportion de voisins appartenant à la classe $c$ estime la probabilité conditionnelle:

$$
p(y = c \mid x, \mathcal{D}) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} \mathbb{1}_{y_i = c}
$$

La prédiction déterministe correspond au mode de cette distribution empirique.

## Fonctions de distance

L'algorithme repose sur la capacité à mesurer la proximité entre points. Une **fonction de distance** $d: \mathcal{X} \times \mathcal{X} \to [0, \infty)$ doit satisfaire trois axiomes: $d(x, y) = 0$ si et seulement si $x = y$ (identité), $d(x, y) = d(y, x)$ (symétrie), et $d(x, z) \leq d(x, y) + d(y, z)$ (inégalité triangulaire).

La **distance euclidienne** est le choix le plus courant:

$$
d_2(x, y) = \sqrt{\sum_{j=1}^{d} (x_j - y_j)^2} = \|x - y\|_2
$$

La **distance de Manhattan** suit les axes plutôt que la ligne droite:

$$
d_1(x, y) = \sum_{j=1}^{d} |x_j - y_j| = \|x - y\|_1
$$

Ces deux distances appartiennent à la famille des normes $\ell_p$, définies par $\|x\|_p = \left(\sum_j |x_j|^p\right)^{1/p}$.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

# Unit balls for different norms
theta = np.linspace(0, 2*np.pi, 1000)

# L1 norm (diamond)
ax = axes[0]
t = np.linspace(0, 1, 250)
x_l1 = np.concatenate([t, 1-t, -t, -1+t])
y_l1 = np.concatenate([1-t, -t, -1+t, t])
ax.fill(x_l1, y_l1, alpha=0.3, color='C0')
ax.plot(x_l1, y_l1, 'C0-', linewidth=2)
ax.set_title(r'Norme $\ell_1$ (Manhattan)')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

# L2 norm (circle)
ax = axes[1]
x_l2 = np.cos(theta)
y_l2 = np.sin(theta)
ax.fill(x_l2, y_l2, alpha=0.3, color='C1')
ax.plot(x_l2, y_l2, 'C1-', linewidth=2)
ax.set_title(r'Norme $\ell_2$ (Euclidienne)')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

# L-infinity norm (square)
ax = axes[2]
x_linf = np.array([1, 1, -1, -1, 1])
y_linf = np.array([1, -1, -1, 1, 1])
ax.fill(x_linf, y_linf, alpha=0.3, color='C2')
ax.plot(x_linf, y_linf, 'C2-', linewidth=2)
ax.set_title(r'Norme $\ell_\infty$')
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')

plt.tight_layout()
```

La distance euclidienne traite toutes les dimensions de manière égale. Si les variables ont des échelles différentes, certaines domineront le calcul. La **distance de Mahalanobis** corrige ce problème:

$$
d_M(x, y) = \sqrt{(x - y)^\top M (x - y)}
$$

où $M$ est une matrice définie positive. Avec $M = I$, on retrouve la distance euclidienne. Avec $M = \Sigma^{-1}$ où $\Sigma$ est la covariance des données, les variables sont normalisées et décorrélées.

Le choix de la distance encode des hypothèses sur ce que signifie "similaire" dans le contexte du problème. Des données de natures différentes appellent des distances différentes.

## L'effet du paramètre k

Le paramètre $k$ contrôle la complexité du modèle. Avec $k = 1$, chaque point est classifié selon son plus proche voisin. La frontière de décision est très irrégulière et s'adapte étroitement aux données. L'erreur d'entraînement est exactement zéro: chaque point est son propre plus proche voisin. Mais cette adaptation excessive aux données d'entraînement nuit à la généralisation.

Avec un grand $k$, la prédiction moyenne sur plus de voisins et la frontière devient plus lisse. Le cas extrême $k = N$ prédit toujours la classe majoritaire globale, ignorant complètement l'entrée.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Generate 2D classification data
np.random.seed(0)
n_samples = 100

# Two interleaved half circles
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=42)

def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for x in X_test:
        distances = np.sqrt(np.sum((X_train - x)**2, axis=1))
        k_nearest_idx = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_nearest_idx]
        predictions.append(np.round(np.mean(k_nearest_labels)))
    return np.array(predictions)

# Create mesh for decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                     np.linspace(y_min, y_max, 150))
X_mesh = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
k_values = [1, 5, 50]
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

for ax, k in zip(axes, k_values):
    Z = knn_predict(X, y, X_mesh, k)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap_light)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='C0', s=30, edgecolors='k', linewidths=0.5)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='C1', s=30, edgecolors='k', linewidths=0.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f'$k = {k}$')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

plt.tight_layout()
```

Entre ces deux extrêmes se trouve le compromis biais-variance. Un petit $k$ donne un modèle à faible biais mais haute variance: les prédictions sont sensibles aux fluctuations des données. Un grand $k$ donne un modèle à haute biais mais faible variance: les prédictions sont stables mais peuvent manquer des structures locales.

Le choix de $k$ se fait par validation. On trace l'erreur sur un ensemble de validation en fonction de $k$ et on retient la valeur qui minimise cette erreur. Des valeurs impaires évitent les égalités dans les votes.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.25, random_state=42)

# Split into train and test
n_train = 140
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for x in X_test:
        distances = np.sqrt(np.sum((X_train - x)**2, axis=1))
        k_nearest_idx = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_nearest_idx]
        predictions.append(np.round(np.mean(k_nearest_labels)))
    return np.array(predictions)

# Compute train and test error for different k
k_values = range(1, 61)
train_errors = []
test_errors = []

for k in k_values:
    y_pred_train = knn_predict(X_train, y_train, X_train, k)
    y_pred_test = knn_predict(X_train, y_train, X_test, k)
    train_errors.append(np.mean(y_pred_train != y_train))
    test_errors.append(np.mean(y_pred_test != y_test))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(k_values, train_errors, 'o-', markersize=3, label='Erreur entraînement')
ax.plot(k_values, test_errors, 's-', markersize=3, label='Erreur test')

# Mark optimal k
best_k = k_values[np.argmin(test_errors)]
ax.axvline(best_k, color='gray', linestyle='--', alpha=0.7)
ax.text(best_k + 1, max(test_errors) * 0.9, f'$k^* = {best_k}$', fontsize=10)

ax.set_xlabel('$k$')
ax.set_ylabel('Taux d\'erreur')
ax.legend()
ax.set_xlim(0, 61)
ax.set_ylim(0, max(max(train_errors), max(test_errors)) * 1.1)
plt.tight_layout()
```

## Diagramme de Voronoï

Le cas $k = 1$ induit une partition de l'espace en cellules. La cellule $V_i$ associée au point $x_i$ contient tous les points plus proches de $x_i$ que de tout autre point d'entraînement:

$$
V_i = \{x \in \mathbb{R}^d : d(x, x_i) \leq d(x, x_j) \text{ pour tout } j \neq i\}
$$

Cette partition s'appelle le **diagramme de Voronoï**. Les frontières entre cellules sont des hyperplans en dimension $d$. Avec le 1-ppv, la frontière de décision suit exactement ce diagramme. Un point mal étiqueté dans l'ensemble d'entraînement crée une cellule entière où les prédictions seront incorrectes.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Generate points
np.random.seed(123)
n_points = 20
X = np.random.randn(n_points, 2) * 1.5
y = (X[:, 0] + X[:, 1] + np.random.randn(n_points) * 0.5 > 0).astype(int)

# Compute Voronoi diagram
vor = Voronoi(X)

fig, ax = plt.subplots(figsize=(7, 6))

# Plot Voronoi regions with colors based on class
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# Create colored regions
for region_idx, point_idx in enumerate(vor.point_region):
    region = vor.regions[point_idx]
    if -1 not in region and len(region) > 0:
        polygon = [vor.vertices[i] for i in region]
        poly = Polygon(polygon, alpha=0.3, 
                      facecolor='C0' if y[region_idx] == 0 else 'C1',
                      edgecolor='gray', linewidth=0.5)
        ax.add_patch(poly)

# Plot Voronoi edges
voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False, 
                line_colors='gray', line_width=1, line_alpha=0.6)

# Plot points
ax.scatter(X[y == 0, 0], X[y == 0, 1], c='C0', s=80, edgecolors='k', 
           linewidths=1, zorder=5, label='Classe 0')
ax.scatter(X[y == 1, 0], X[y == 1, 1], c='C1', s=80, edgecolors='k', 
           linewidths=1, zorder=5, label='Classe 1')

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.legend(loc='upper left')
ax.set_aspect('equal')
plt.tight_layout()
```

## Le fléau de la dimensionnalité

Les k-ppv fonctionnent bien en basse dimension mais souffrent en haute dimension. Ce phénomène, le **fléau de la dimensionnalité**, affecte toutes les méthodes basées sur la localité.

Considérons des points uniformément distribués dans $[0, 1]^d$. Pour capturer une fraction $p$ des points dans un hypercube, le côté doit être $r = p^{1/d}$. En dimension 1, capturer 10% des points requiert un intervalle de longueur 0.1. En dimension 100, il faut un hypercube de côté $0.1^{1/100} \approx 0.98$, couvrant presque tout l'espace.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

p = 0.1  # fraction of points to capture
dimensions = np.arange(1, 101)
side_length = p ** (1 / dimensions)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(dimensions, side_length, 'C0-', linewidth=2)
ax.axhline(1, color='gray', linestyle='--', alpha=0.5)

# Mark specific points
for d in [1, 2, 10, 50, 100]:
    r = p ** (1/d)
    ax.plot(d, r, 'ko', markersize=6)
    if d == 1:
        ax.annotate(f'd={d}\nr={r:.2f}', (d, r), textcoords='offset points', 
                   xytext=(10, -15), fontsize=9)
    elif d == 100:
        ax.annotate(f'd={d}\nr={r:.2f}', (d, r), textcoords='offset points', 
                   xytext=(-40, -20), fontsize=9)

ax.set_xlabel('Dimension $d$')
ax.set_ylabel('Côté de l\'hypercube $r$')
ax.set_title(f'Côté nécessaire pour capturer {int(p*100)}% des points: $r = p^{{1/d}}$')
ax.set_xlim(0, 105)
ax.set_ylim(0, 1.1)
plt.tight_layout()
```

En haute dimension, les "voisins" ne sont plus locaux. La distance au plus proche voisin augmente et tous les points deviennent approximativement équidistants. Pour maintenir une densité constante, le nombre d'exemples requis croît exponentiellement avec la dimension.

La réduction de dimension et les distances adaptatives peuvent atténuer le problème, mais fondamentalement, les méthodes de voisinage perdent leur efficacité quand $d$ est grand.

## Régression

L'algorithme s'adapte à la régression en remplaçant le vote majoritaire par une moyenne:

$$
\hat{y} = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} y_i
$$

Cette moyenne locale estime $\mathbb{E}[Y \mid X = x]$.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Generate 1D regression data
np.random.seed(42)
n = 50
X_train = np.sort(np.random.uniform(0, 10, n))
y_train = np.sin(X_train) + np.random.randn(n) * 0.3

def knn_regression(X_train, y_train, X_test, k):
    predictions = []
    for x in X_test:
        distances = np.abs(X_train - x)
        k_nearest_idx = np.argsort(distances)[:k]
        predictions.append(np.mean(y_train[k_nearest_idx]))
    return np.array(predictions)

X_test = np.linspace(0, 10, 200)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
k_values = [1, 5, 15]

for ax, k in zip(axes, k_values):
    y_pred = knn_regression(X_train, y_train, X_test, k)
    
    ax.scatter(X_train, y_train, c='C0', s=30, alpha=0.6, label='Données')
    ax.plot(X_test, y_pred, 'C1-', linewidth=2, label=f'k-ppv ($k={k}$)')
    ax.plot(X_test, np.sin(X_test), 'k--', alpha=0.5, label=r'$\sin(x)$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'$k = {k}$')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, 10)

plt.tight_layout()
```

Une variante pondère les voisins par l'inverse de leur distance:

$$
\hat{y} = \frac{\sum_{i \in \mathcal{N}_k(x)} w_i \, y_i}{\sum_{i \in \mathcal{N}_k(x)} w_i}, \quad w_i = \frac{1}{d(x, x_i)}
$$

Les voisins plus proches ont alors plus d'influence sur la prédiction.

## Complexité

L'entraînement consiste à stocker les données: $O(N)$. L'inférence requiert de calculer la distance à tous les points et d'identifier les $k$ plus proches: $O(Nd)$ par requête. Pour de grands ensembles, des structures comme les arbres k-d ou le hachage sensible à la localité (LSH) réduisent ce coût.

## Implémentation

```python
import numpy as np

def knn_classify(X_train, y_train, X_test, k=3):
    predictions = []
    for x in X_test:
        distances = np.sqrt(np.sum((X_train - x)**2, axis=1))
        k_nearest_idx = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_nearest_idx]
        unique, counts = np.unique(k_nearest_labels, return_counts=True)
        predictions.append(unique[np.argmax(counts)])
    return np.array(predictions)

def knn_regression(X_train, y_train, X_test, k=3):
    predictions = []
    for x in X_test:
        distances = np.sqrt(np.sum((X_train - x)**2, axis=1))
        k_nearest_idx = np.argsort(distances)[:k]
        predictions.append(np.mean(y_train[k_nearest_idx]))
    return np.array(predictions)
```

## Méthodes paramétriques et non paramétriques

Les k-ppv sont une méthode **non paramétrique**: les données sont le modèle. Il n'y a pas de paramètres appris; les prédictions consultent directement l'ensemble d'entraînement. La complexité du modèle croît avec $N$.

| | Non paramétrique | Paramétrique |
|--|------------------|--------------|
| **Modèle** | Les données | Un vecteur $\theta \in \mathbb{R}^p$ |
| **Complexité** | Croît avec $N$ | Fixe |
| **Inférence** | Requiert les données | Requiert seulement $\theta$ |

Les méthodes **paramétriques** distillent l'information dans un vecteur de paramètres de taille fixe. Un réseau de neurones entraîné sur des milliards d'exemples n'a besoin que de ses poids pour faire des prédictions, pas des données d'entraînement.

## Résumé

Les k plus proches voisins classifient un point par vote majoritaire parmi ses $k$ voisins les plus proches. Le paramètre $k$ contrôle le compromis biais-variance. Le choix de la distance encode les hypothèses sur la similarité. Le fléau de la dimensionnalité limite l'efficacité en haute dimension.

La méthode illustre la tension entre mémorisation et généralisation: avec $k=1$, l'erreur d'entraînement est nulle mais la généralisation est mauvaise. Elle illustre aussi la distinction entre approches non paramétriques (les données sont le modèle) et paramétriques (un vecteur de paramètres résume les données).

Le chapitre suivant développe l'approche paramétrique: l'apprentissage comme problème d'optimisation, où nous cherchons les paramètres qui minimisent une fonction de perte.
