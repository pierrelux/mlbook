---
kernelspec:
  name: python3
  display_name: Python 3
---

# Méthodes non paramétriques

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Distinguer les méthodes paramétriques et non paramétriques
- Expliquer le fonctionnement de l'algorithme des k plus proches voisins
- Définir et appliquer différentes fonctions de distance
- Analyser l'effet du paramètre k sur le compromis biais-variance
- Expliquer le fléau de la dimensionnalité et ses conséquences
- Implémenter l'algorithme k-ppv pour la classification et la régression
```

En apprentissage automatique, deux grandes familles d'approches s'opposent. Les **méthodes paramétriques** résument les données d'entraînement dans un ensemble fixe de paramètres: une fois l'apprentissage terminé, les données peuvent être jetées. Les **méthodes non paramétriques** conservent les données et les consultent directement au moment de la prédiction. Ce chapitre présente cette seconde famille, dont les k plus proches voisins sont l'exemple le plus simple.

## L'idée de base

Soit $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ un ensemble d'entraînement avec $\mathbf{x}_i \in \mathbb{R}^d$ et $y_i \in \{1, \ldots, C\}$. Nous voulons prédire l'étiquette d'un nouveau point $\mathbf{x}$. L'approche la plus simple consiste à regarder les exemples connus qui ressemblent à $\mathbf{x}$ et à prédire la même chose.

Les **k plus proches voisins** (k-ppv) formalisent cette intuition. Pour classifier $\mathbf{x}$, nous identifions les $k$ points de $\mathcal{D}$ les plus proches de $\mathbf{x}$ et prenons un vote majoritaire sur leurs étiquettes. La méthode ne fait aucune hypothèse sur la forme de la relation entre $\mathbf{x}$ et $y$. Elle se contente de consulter les données au moment de la prédiction.

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

Soit $\mathcal{N}_k(\mathbf{x})$ l'ensemble des indices des $k$ plus proches voisins de $\mathbf{x}$. La prédiction est:

$$
\hat{y} = \arg\max_{c} \sum_{i \in \mathcal{N}_k(\mathbf{x})} \mathbb{1}_{y_i = c}
$$

La somme compte combien de voisins appartiennent à chaque classe $c$, et nous retenons la classe la plus fréquente.

Cette formulation admet une interprétation probabiliste. La proportion de voisins appartenant à la classe $c$ estime la probabilité conditionnelle:

$$
p(y = c \mid \mathbf{x}, \mathcal{D}) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(\mathbf{x})} \mathbb{1}_{y_i = c}
$$

La prédiction déterministe correspond au mode de cette distribution empirique.

## Fonctions de distance

L'algorithme repose sur la capacité à mesurer la proximité entre points. Une **fonction de distance** $d: \mathcal{X} \times \mathcal{X} \to [0, \infty)$ doit satisfaire trois axiomes: $d(\mathbf{x}, \mathbf{y}) = 0$ si et seulement si $\mathbf{x} = \mathbf{y}$ (identité), $d(\mathbf{x}, \mathbf{y}) = d(\mathbf{y}, \mathbf{x})$ (symétrie), et $d(\mathbf{x}, \mathbf{z}) \leq d(\mathbf{x}, \mathbf{y}) + d(\mathbf{y}, \mathbf{z})$ (inégalité triangulaire).

La **distance euclidienne** est le choix le plus courant:

$$
d_2(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{j=1}^{d} (x_j - y_j)^2} = \|\mathbf{x} - \mathbf{y}\|_2
$$

La **distance de Manhattan** suit les axes plutôt que la ligne droite:

$$
d_1(\mathbf{x}, \mathbf{y}) = \sum_{j=1}^{d} |x_j - y_j| = \|\mathbf{x} - \mathbf{y}\|_1
$$

Ces deux distances appartiennent à la famille des normes $\ell_p$, définies par $\|\mathbf{x}\|_p = \left(\sum_j |x_j|^p\right)^{1/p}$. Le cas limite $p \to \infty$ donne la norme $\ell_\infty$:

$$
\|\mathbf{x}\|_\infty = \max_j |x_j|
$$

### Boules unité et géométrie des normes

Pour comprendre comment une norme mesure les distances, on trace sa **boule unité**. Formellement, la boule unité d'une norme $\|\cdot\|$ est l'ensemble:

$$
B = \{\mathbf{x} \in \mathbb{R}^d : \|\mathbf{x}\| \leq 1\}
$$

et sa frontière, la **sphère unité**, est $S = \{\mathbf{x} : \|\mathbf{x}\| = 1\}$. Tous les points sur cette sphère sont à distance exactement 1 de l'origine. La forme de la boule révèle ce que la norme considère comme "équidistant".

- **Norme $\ell_2$ (cercle)**: Le point $(1, 0)$ et le point $(0.71, 0.71)$ sont à la même distance de l'origine. Se déplacer en diagonale coûte autant que suivre un axe. C'est notre intuition géométrique habituelle.

- **Norme $\ell_1$ (losange)**: Le point $(1, 0)$ est à distance 1, mais $(0.71, 0.71)$ est à distance $0.71 + 0.71 = 1.42$. Se déplacer en diagonale coûte plus cher, comme un taxi qui ne peut tourner qu'aux intersections.

- **Norme $\ell_\infty$ (carré)**: Seule la plus grande coordonnée compte. Les points $(1, 0)$, $(1, 0.5)$ et $(1, 1)$ sont tous à distance 1. C'est la distance du joueur d'échecs (le roi peut se déplacer d'une case dans n'importe quelle direction).

Pour les k-ppv, la forme de la boule détermine quels points sont considérés voisins. Avec $\ell_1$, les voisins forment un losange autour de la requête; avec $\ell_\infty$, un carré.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(11, 4))

# Unit balls for different norms
theta = np.linspace(0, 2*np.pi, 1000)

# Key points to highlight
p1 = (1, 0)
p2 = (1/np.sqrt(2), 1/np.sqrt(2))  # ≈ (0.71, 0.71)

# L1 norm (diamond)
ax = axes[0]
t = np.linspace(0, 1, 250)
x_l1 = np.concatenate([t, 1-t, -t, -1+t])
y_l1 = np.concatenate([1-t, -t, -1+t, t])
ax.fill(x_l1, y_l1, alpha=0.3, color='C0')
ax.plot(x_l1, y_l1, 'C0-', linewidth=2)

# Show points - (1,0) is on boundary, (0.71, 0.71) is OUTSIDE
ax.scatter([p1[0]], [p1[1]], s=80, c='black', zorder=5)
ax.scatter([p2[0]], [p2[1]], s=80, c='red', zorder=5)
ax.annotate(f'$(1, 0)$\n$d=1$', p1, textcoords='offset points', 
            xytext=(5, 10), fontsize=9)
ax.annotate(f'$(0.71, 0.71)$\n$d=1.42$', p2, textcoords='offset points', 
            xytext=(5, 10), fontsize=9, color='red')

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

# Both points are on the boundary for L2
ax.scatter([p1[0]], [p1[1]], s=80, c='black', zorder=5)
ax.scatter([p2[0]], [p2[1]], s=80, c='black', zorder=5)
ax.annotate(f'$(1, 0)$\n$d=1$', p1, textcoords='offset points', 
            xytext=(5, 10), fontsize=9)
ax.annotate(f'$(0.71, 0.71)$\n$d=1$', p2, textcoords='offset points', 
            xytext=(5, 10), fontsize=9)

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

# For L-inf: (1,0), (1,0.5), (1,1) all have distance 1
p_inf = [(1, 0), (1, 0.5), (1, 1)]
for i, p in enumerate(p_inf):
    ax.scatter([p[0]], [p[1]], s=80, c='black', zorder=5)
ax.annotate('$(1, 0)$\n$d=1$', p_inf[0], textcoords='offset points', 
            xytext=(-45, -5), fontsize=9)
ax.annotate('$(1, 0.5)$\n$d=1$', p_inf[1], textcoords='offset points', 
            xytext=(5, -5), fontsize=9)
ax.annotate('$(1, 1)$\n$d=1$', p_inf[2], textcoords='offset points', 
            xytext=(5, 5), fontsize=9)

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

### Données numériques: normalisation et corrélation

La distance euclidienne traite toutes les dimensions de manière égale. Si les variables ont des échelles différentes, certaines domineront le calcul. Imaginons un problème où $x_1$ est l'âge (0-100) et $x_2$ est le revenu annuel (0-500000). Sans normalisation, la différence de revenu écrasera la différence d'âge.

**Solution pratique**: normaliser les variables (soustraire la moyenne, diviser par l'écart-type) avant d'appliquer les k-ppv. C'est presque toujours nécessaire pour des données tabulaires.

La **distance de Mahalanobis** va plus loin en tenant compte des corrélations:

$$
d_M(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x} - \mathbf{y})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \mathbf{y})}
$$

où $\boldsymbol{\Sigma}$ est la matrice de covariance des données. Pour comprendre cette formule, décomposons-la.

**La matrice de covariance $\boldsymbol{\Sigma}$.** Cette matrice $d \times d$ capture deux informations: sur la diagonale, les variances de chaque variable; hors diagonale, les covariances (corrélations) entre variables. Si $\boldsymbol{\Sigma} = \begin{pmatrix} 4 & 0 \\ 0 & 1 \end{pmatrix}$, la première variable a une variance 4 fois plus grande que la seconde, et elles sont indépendantes.

**Calcul concret.** Soit $\mathbf{X} \in \mathbb{R}^{N \times d}$ la matrice des données (chaque ligne est un exemple). On centre d'abord les données en soustrayant la moyenne de chaque colonne:

$$
\bar{x}_j = \frac{1}{N} \sum_{i=1}^{N} x_{ij}, \quad \tilde{\mathbf{X}} = \mathbf{X} - \mathbf{1} \bar{\mathbf{x}}^\top
$$

La covariance empirique est alors:

$$
\boldsymbol{\Sigma} = \frac{1}{N-1} \tilde{\mathbf{X}}^\top \tilde{\mathbf{X}}
$$

L'élément $(j, k)$ de cette matrice est $\Sigma_{jk} = \frac{1}{N-1} \sum_{i=1}^{N} (x_{ij} - \bar{x}_j)(x_{ik} - \bar{x}_k)$. En Python:

```python
X_centered = X - X.mean(axis=0)
Sigma = (X_centered.T @ X_centered) / (len(X) - 1)
# ou directement: Sigma = np.cov(X.T)
```

**L'inverse $\boldsymbol{\Sigma}^{-1}$.** Multiplier par l'inverse de la covariance "blanchit" les données: les directions de forte variance sont comprimées, les corrélations sont supprimées. Après cette transformation, les données ressemblent à un nuage sphérique de variance unitaire.

**Interprétation géométrique.** La distance de Mahalanobis mesure "à combien d'écarts-types" un point se trouve d'un autre, en tenant compte de la forme du nuage de données. Deux points éloignés dans une direction de forte variance sont considérés plus proches que deux points également éloignés dans une direction de faible variance.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

np.random.seed(42)

# Generate correlated 2D data
n = 200
mean = [0, 0]
cov = [[2, 1.5], [1.5, 1.5]]  # Correlated, different variances
X = np.random.multivariate_normal(mean, cov, n)

# Compute sample covariance
Sigma = np.cov(X.T)
Sigma_inv = np.linalg.inv(Sigma)

# A reference point and a query point
ref = np.array([0, 0])
query1 = np.array([2, 0])    # Along high-variance direction
query2 = np.array([-0.5, 1]) # Along low-variance direction

# Compute distances
def euclidean(a, b):
    return np.sqrt(np.sum((a - b)**2))

def mahalanobis(a, b, Sigma_inv):
    diff = a - b
    return np.sqrt(diff @ Sigma_inv @ diff)

d_euc1 = euclidean(ref, query1)
d_euc2 = euclidean(ref, query2)
d_mah1 = mahalanobis(ref, query1, Sigma_inv)
d_mah2 = mahalanobis(ref, query2, Sigma_inv)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Euclidean view
ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], alpha=0.3, s=20, c='gray')
ax.scatter([ref[0]], [ref[1]], s=100, c='black', zorder=5, label='Référence')
ax.scatter([query1[0]], [query1[1]], s=100, c='C0', zorder=5, marker='s')
ax.scatter([query2[0]], [query2[1]], s=100, c='C1', zorder=5, marker='^')

# Draw circles for Euclidean distance
circle1 = plt.Circle(ref, d_euc1, fill=False, color='C0', linestyle='--', linewidth=2)
circle2 = plt.Circle(ref, d_euc2, fill=False, color='C1', linestyle='--', linewidth=2)
ax.add_patch(circle1)
ax.add_patch(circle2)

ax.annotate(f'$d_{{euc}} = {d_euc1:.2f}$', query1, textcoords='offset points', 
            xytext=(10, 10), fontsize=10, color='C0')
ax.annotate(f'$d_{{euc}} = {d_euc2:.2f}$', query2, textcoords='offset points', 
            xytext=(10, 10), fontsize=10, color='C1')

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Distance Euclidienne\n(ignore la structure des données)')

# Right: Mahalanobis view
ax = axes[1]
ax.scatter(X[:, 0], X[:, 1], alpha=0.3, s=20, c='gray')
ax.scatter([ref[0]], [ref[1]], s=100, c='black', zorder=5, label='Référence')
ax.scatter([query1[0]], [query1[1]], s=100, c='C0', zorder=5, marker='s')
ax.scatter([query2[0]], [query2[1]], s=100, c='C1', zorder=5, marker='^')

# Draw ellipses for Mahalanobis distance (iso-distance contours)
# Eigendecomposition for ellipse orientation
eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))

for d_mah, color in [(d_mah1, 'C0'), (d_mah2, 'C1')]:
    width = 2 * d_mah * np.sqrt(eigenvalues[1])
    height = 2 * d_mah * np.sqrt(eigenvalues[0])
    ellipse = Ellipse(ref, width, height, angle=angle, fill=False, 
                      color=color, linestyle='--', linewidth=2)
    ax.add_patch(ellipse)

ax.annotate(f'$d_{{mah}} = {d_mah1:.2f}$', query1, textcoords='offset points', 
            xytext=(10, 10), fontsize=10, color='C0')
ax.annotate(f'$d_{{mah}} = {d_mah2:.2f}$', query2, textcoords='offset points', 
            xytext=(10, 10), fontsize=10, color='C1')

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Distance de Mahalanobis\n(tient compte de la covariance)')

plt.tight_layout()
```

Dans cet exemple, le point bleu (carré) est plus loin en distance euclidienne, mais plus proche en distance de Mahalanobis. Cela s'explique par le fait qu'il se trouve dans la direction où les données varient naturellement. Le point orange (triangle), bien que plus proche en euclidien, est "surprenant" par rapport à la distribution et donc plus loin en Mahalanobis.

En pratique, on utilise rarement Mahalanobis directement pour les k-ppv. La normalisation standard (centrer et réduire chaque variable) capture l'essentiel. Mahalanobis devient utile quand les corrélations entre variables sont fortes et informatives.

### Au-delà des vecteurs numériques

Les k-ppv ne se limitent pas aux vecteurs dans $\mathbb{R}^d$. Toute fonction de distance valide permet d'appliquer l'algorithme.

**Chaînes de caractères et ADN.** La **distance d'édition** (ou distance de Levenshtein) compte le nombre minimum d'opérations (insertion, suppression, substitution) pour transformer une chaîne en une autre:

$$
d_{\text{edit}}(\texttt{"chat"}, \texttt{"chien"}) = 3
$$

Cette distance est utilisée pour la correction orthographique, l'alignement de séquences ADN, et la détection de plagiat. Pour comparer des séquences génétiques, on peut aussi utiliser des distances spécialisées qui tiennent compte de la biologie des mutations.

**Documents et texte.** Pour comparer des documents, on les représente souvent comme des vecteurs de fréquences de mots (bag-of-words). La **similarité cosinus** mesure l'angle entre ces vecteurs:

$$
\text{sim}_{\cos}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}
$$

On convertit en distance par $d = 1 - \text{sim}_{\cos}$. Cette mesure ignore la longueur des documents et se concentre sur leur contenu thématique. C'est le choix standard pour la recherche d'information et la classification de texte.

**Ensembles et données binaires.** Pour des données représentées comme des ensembles (mots-clés, tags, gènes exprimés), la **distance de Jaccard** mesure le chevauchement:

$$
d_{\text{Jaccard}}(A, B) = 1 - \frac{|A \cap B|}{|A \cup B|}
$$

Deux documents partageant 80% de leurs mots-clés ont une distance de 0.2. Pour des vecteurs binaires (présence/absence), la **distance de Hamming** compte les positions différentes.

**Images.** Les pixels bruts donnent des distances peu informatives. Par exemple, deux images du même objet décalé d'un pixel seraient très "différentes". En pratique, on extrait des représentations (embeddings) via des réseaux de neurones pré-entraînés, puis on applique la distance euclidienne ou cosinus dans cet espace de représentation.

### Guide pratique: quelle distance choisir?

| Type de données | Distance recommandée | Pourquoi |
|-----------------|---------------------|----------|
| Vecteurs numériques | Euclidienne (après normalisation) | Simple, efficace, interprétable |
| Données avec corrélations fortes | Mahalanobis ou PCA + Euclidienne | Tient compte de la structure |
| Texte / documents | Cosinus sur TF-IDF ou embeddings | Invariant à la longueur |
| Séquences (ADN, protéines) | Distance d'édition ou alignement | Capture les mutations/insertions |
| Ensembles, tags | Jaccard | Mesure le chevauchement |
| Vecteurs binaires | Hamming | Compte les différences |
| Images | Cosinus sur embeddings CNN | Les pixels bruts sont peu informatifs |

**Le choix de la distance encode vos hypothèses.** Si deux clients ayant acheté les mêmes produits sont "similaires", utilisez Jaccard sur les paniers. Si deux clients ayant dépensé des montants similaires sont "proches", utilisez la distance euclidienne sur les dépenses. La distance définit ce que "voisin" signifie: c'est une décision de modélisation, pas un détail technique.

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

np.random.seed(42)

# Generate a larger dataset for smoother curves
n_samples = 500
X = np.random.randn(n_samples, 2)
# Create a non-linear boundary
y = ((X[:, 0]**2 + X[:, 1]**2) > 1.5).astype(int)
# Add label noise
noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
y[noise_idx] = 1 - y[noise_idx]

# Split into train and test
n_train = 350
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
# Use 1/k on x-axis (model complexity decreases as k increases)
k_values = list(range(1, 100, 2))  # Odd values to avoid ties
train_errors = []
test_errors = []

for k in k_values:
    y_pred_train = knn_predict(X_train, y_train, X_train, k)
    y_pred_test = knn_predict(X_train, y_train, X_test, k)
    train_errors.append(np.mean(y_pred_train != y_train))
    test_errors.append(np.mean(y_pred_test != y_test))

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.plot(k_values, train_errors, 'C0-', linewidth=2, label='Erreur entraînement')
ax.plot(k_values, test_errors, 'C1-', linewidth=2, label='Erreur test')

# Mark optimal k
best_idx = np.argmin(test_errors)
best_k = k_values[best_idx]
ax.scatter([best_k], [test_errors[best_idx]], s=100, c='C1', zorder=5, edgecolors='black')
ax.axvline(best_k, color='gray', linestyle='--', alpha=0.5)

# Add annotations for the regions - positioned to avoid overlapping curves
ax.text(10, 0.45, 'Surapprentissage\n(haute variance)', fontsize=10, 
        ha='center', va='top', color='gray', alpha=0.8)
ax.text(85, 0.1, 'Sous-apprentissage\n(haut biais)', fontsize=10, 
        ha='center', va='bottom', color='gray', alpha=0.8)

# Mark optimal k with an arrow from above
ax.annotate(f'Meilleur $k = {best_k}$', xy=(best_k, test_errors[best_idx]), 
            xytext=(best_k, test_errors[best_idx] + 0.1),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black'))

ax.set_xlabel('$k$ (nombre de voisins)')
ax.set_ylabel('Taux d\'erreur')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
ax.set_xlim(0, 100)
ax.set_ylim(0, 0.5)

# Add complexity arrow at the bottom
ax.text(0.5, -0.15, r'$\longleftarrow$ complexité du modèle', transform=ax.transAxes, 
        ha='center', va='top', fontsize=9, color='gray')

plt.tight_layout()
```

## Diagramme de Voronoï

Le cas $k = 1$ induit une partition de l'espace en cellules. La cellule $V_i$ associée au point $\mathbf{x}_i$ contient tous les points plus proches de $\mathbf{x}_i$ que de tout autre point d'entraînement:

$$
V_i = \{\mathbf{x} \in \mathbb{R}^d : d(\mathbf{x}, \mathbf{x}_i) \leq d(\mathbf{x}, \mathbf{x}_j) \text{ pour tout } j \neq i\}
$$

Cette partition s'appelle le **diagramme de Voronoï**. Les frontières entre cellules sont des hyperplans en dimension $d$. Avec le 1-ppv, la frontière de décision suit exactement ce diagramme.

### Stabilité et marge géométrique

Le diagramme de Voronoï donne une lecture géométrique de la **stabilité** des prédictions. Tant qu'une requête $\mathbf{x}$ reste dans la même cellule $V_i$, sa prédiction ne change pas. La distance à la frontière de la cellule mesure donc la robustesse de la prédiction.

Pour quantifier cette robustesse, considérons l'écart entre les distances au premier et au deuxième plus proche voisin:

$$
\rho(\mathbf{x}) = \frac{1}{2}\left(d(\mathbf{x}, \mathbf{x}_{(2)}) - d(\mathbf{x}, \mathbf{x}_{(1)})\right)
$$

où $\mathbf{x}_{(1)}$ et $\mathbf{x}_{(2)}$ sont les premier et deuxième plus proches voisins. Si $\rho(\mathbf{x})$ est grand, le point $\mathbf{x}$ est loin de toute frontière: sa prédiction est stable. Si $\rho(\mathbf{x})$ est proche de zéro, $\mathbf{x}$ est sur une frontière: la moindre perturbation (bruit de mesure, erreur d'arrondi) peut changer la prédiction.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Generate points with clear class structure
np.random.seed(42)
n_per_class = 8
X0 = np.random.randn(n_per_class, 2) * 0.8 + np.array([-1.5, 0])
X1 = np.random.randn(n_per_class, 2) * 0.8 + np.array([1.5, 0])
X = np.vstack([X0, X1])
y = np.array([0] * n_per_class + [1] * n_per_class)

# Compute Voronoi diagram
vor = Voronoi(X)

# Create test points and compute their margin
np.random.seed(123)
test_points = np.random.randn(5, 2) * 1.5

def compute_margin(x, X_train):
    """Compute margin: half the difference between 2nd and 1st neighbor distances."""
    distances = np.sqrt(np.sum((X_train - x)**2, axis=1))
    sorted_dist = np.sort(distances)
    return 0.5 * (sorted_dist[1] - sorted_dist[0])

margins = [compute_margin(p, X) for p in test_points]

fig, ax = plt.subplots(figsize=(8, 6))

# Plot Voronoi regions with colors based on class
from matplotlib.patches import Polygon, Circle

for region_idx, point_idx in enumerate(vor.point_region):
    region = vor.regions[point_idx]
    if -1 not in region and len(region) > 0:
        polygon = [vor.vertices[i] for i in region]
        poly = Polygon(polygon, alpha=0.2, 
                      facecolor='C0' if y[region_idx] == 0 else 'C1',
                      edgecolor='gray', linewidth=0.5)
        ax.add_patch(poly)

# Plot Voronoi edges
voronoi_plot_2d(vor, ax=ax, show_vertices=False, show_points=False, 
                line_colors='gray', line_width=1, line_alpha=0.6)

# Plot training points
ax.scatter(X[y == 0, 0], X[y == 0, 1], c='C0', s=80, edgecolors='k', 
           linewidths=1, zorder=5, label='Classe 0')
ax.scatter(X[y == 1, 0], X[y == 1, 1], c='C1', s=80, edgecolors='k', 
           linewidths=1, zorder=5, label='Classe 1')

# Plot test points with margin circles
for i, (p, m) in enumerate(zip(test_points, margins)):
    # Color based on stability: green = stable (large margin), red = unstable
    color = 'green' if m > 0.3 else 'orange' if m > 0.1 else 'red'
    ax.scatter(p[0], p[1], c=color, s=100, marker='*', edgecolors='k', 
               linewidths=0.5, zorder=6)
    # Draw circle showing margin
    circle = Circle(p, m, fill=False, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    ax.add_patch(circle)

ax.set_xlim(-4, 4)
ax.set_ylim(-3, 3)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.legend(loc='upper left')
ax.set_aspect('equal')
ax.set_title(r'Marge $\rho(x)$: distance à la frontière (cercles pointillés)')
plt.tight_layout()
```

Les étoiles représentent des points de test. Le cercle pointillé autour de chaque point montre sa marge $\rho(x)$: tant que le point reste dans ce cercle, sa prédiction est garantie stable. Les points verts ont une grande marge (prédiction robuste), les points rouges sont près d'une frontière (prédiction fragile).

### Quels points définissent la frontière?

La frontière de décision du 1-ppv est constituée des **faces de Voronoï séparant des points d'étiquettes différentes**. Les faces entre points de même classe ne contribuent pas à la frontière de décision.

Cette observation a une conséquence pratique importante: seuls les points proches de la frontière "comptent vraiment" pour le classifieur. Les points bien à l'intérieur d'une région homogène (entourés uniquement de points de même classe) pourraient être retirés sans changer significativement les prédictions.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

np.random.seed(42)
n_per_class = 8
X0 = np.random.randn(n_per_class, 2) * 0.8 + np.array([-1.5, 0])
X1 = np.random.randn(n_per_class, 2) * 0.8 + np.array([1.5, 0])
X = np.vstack([X0, X1])
y = np.array([0] * n_per_class + [1] * n_per_class)

vor = Voronoi(X)

fig, ax = plt.subplots(figsize=(8, 6))

# Find which edges are on the decision boundary (between different classes)
# For each ridge, check if the two points have different labels
boundary_edges = []
internal_edges = []

for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
    if v1 >= 0 and v2 >= 0:  # Both vertices are finite
        if y[p1] != y[p2]:
            boundary_edges.append((vor.vertices[v1], vor.vertices[v2]))
        else:
            internal_edges.append((vor.vertices[v1], vor.vertices[v2]))

# Plot internal edges (light gray)
for v1, v2 in internal_edges:
    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'gray', linewidth=0.8, alpha=0.4)

# Plot boundary edges (thick black)
for v1, v2 in boundary_edges:
    ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 'k-', linewidth=2.5)

# Identify boundary vs interior points
# A point is on the boundary if it shares a Voronoi edge with a point of different class
boundary_points = set()
for p1, p2 in vor.ridge_points:
    if y[p1] != y[p2]:
        boundary_points.add(p1)
        boundary_points.add(p2)

interior_points = set(range(len(X))) - boundary_points

# Plot points: boundary points larger, interior points smaller and faded
for i in range(len(X)):
    if i in boundary_points:
        color = 'C0' if y[i] == 0 else 'C1'
        ax.scatter(X[i, 0], X[i, 1], c=color, s=120, edgecolors='k', 
                   linewidths=2, zorder=5)
    else:
        color = 'C0' if y[i] == 0 else 'C1'
        ax.scatter(X[i, 0], X[i, 1], c=color, s=60, edgecolors='gray', 
                   linewidths=1, alpha=0.5, zorder=4)

ax.set_xlim(-4, 4)
ax.set_ylim(-3, 3)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_aspect('equal')
ax.set_title('Points définissant la frontière (grands) vs points intérieurs (petits, estompés)')
plt.tight_layout()
```

Cette idée motive les variantes **condensées** et **éditées** des k-ppv:
- **Condensed Nearest Neighbor**: ne garder que les points nécessaires pour préserver la frontière de décision
- **Edited Nearest Neighbor**: retirer les points mal classifiés par leurs voisins (probablement du bruit)

Ces techniques réduisent la taille du jeu de données stocké, accélérant l'inférence sans dégrader significativement la performance.

### Du concept au calcul

Le diagramme de Voronoï est la structure géométrique *idéale* pour répondre à la question "quel est le plus proche voisin?". En dimension 2 ou 3, on peut le construire efficacement et l'utiliser pour des requêtes en temps logarithmique.

Mais en grande dimension ($d > 20$), la construction devient prohibitive: le nombre de faces croît exponentiellement avec $d$. C'est pourquoi nous utilisons des **structures approximatives** (arbres k-d, graphes de proximité, hachage) qui renoncent à la perfection géométrique en échange de l'efficacité. Le diagramme de Voronoï reste utile conceptuellement: il nous dit ce que nous *voudrions* calculer, même si nous devons nous contenter d'approximations.

## Le fléau de la dimensionnalité

Les k-ppv fonctionnent bien en basse dimension mais souffrent en haute dimension. Ce phénomène, le **fléau de la dimensionnalité**, affecte toutes les méthodes basées sur la localité. Le problème n'est pas principalement calculatoire: le coût $O(Nd)$ croît linéairement en $d$, mais il est fondamentalement *statistique*.

### La notion de voisinage s'effondre

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

### Concentration des distances

En haute dimension, un phénomène contre-intuitif se produit: les distances entre points se concentrent autour d'une même valeur. Le ratio entre la distance au plus proche voisin et au plus éloigné tend vers 1:

$$
\frac{d_{\min}}{d_{\max}} \xrightarrow{d \to \infty} 1
$$

Tous les points deviennent approximativement équidistants. La notion même de "plus proche voisin" perd son sens: si tous les points sont à la même distance, lequel choisir?

### Conséquences sur le biais

Rappelons que les k-ppv estiment $\mathbb{E}[Y \mid X = x]$ par une moyenne locale. En haute dimension, cette moyenne n'est plus locale car elle inclut des points qui, bien que "voisins" au sens de la distance, peuvent être très différents de $x$ dans l'espace d'entrée. Le biais augmente: nous moyennons sur des régions trop vastes pour capturer les variations locales de la fonction cible.

Pour maintenir une densité constante de voisins dans une boule de rayon fixe, le nombre d'exemples requis croît exponentiellement avec la dimension: $N \propto r^{-d}$. Avec 1000 points en dimension 10, la densité locale est déjà très faible.

### Remèdes

La réduction de dimension (PCA, autoencodeurs) projette les données dans un espace de plus basse dimension avant d'appliquer les k-ppv. Les distances adaptatives comme Mahalanobis peuvent aider si certaines dimensions sont plus informatives. Mais fondamentalement, les méthodes de voisinage ne sont pas adaptées aux problèmes en très haute dimension, ce qui motive l'étude des méthodes paramétriques dans les chapitres suivants.

## Régression

En classification, nous avons combiné les étiquettes des voisins par vote majoritaire. Pour la régression, où $y_i \in \mathbb{R}$, la combinaison naturelle est une moyenne:

$$
\hat{y} = \frac{1}{k} \sum_{i \in \mathcal{N}_k(\mathbf{x})} y_i
$$

Cette moyenne locale estime l'espérance conditionnelle $\mathbb{E}[Y \mid \mathbf{X} = \mathbf{x}]$. L'intuition est simple: si nous voulons prédire la température demain et que nous avons des données historiques, regarder les jours passés qui ressemblaient à aujourd'hui et moyenner leurs températures du lendemain semble raisonnable.

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

Avec $k = 1$, la prédiction saute d'un point à l'autre, créant une fonction en escalier. Augmenter $k$ lisse la prédiction, mais un $k$ trop grand écrase les variations locales.

## Estimation de densité par noyaux

Avant d'aller plus loin dans la régression, introduisons un outil fondamental: l'**estimation de densité par noyaux** (kernel density estimation, ou fenêtres de Parzen).

Supposons que nous observons des points $x_1, \ldots, x_N$ tirés d'une densité inconnue $p(x)$. Comment estimer cette densité? Une approche naïve serait de construire un histogramme, mais les histogrammes dépendent du choix arbitraire des intervalles et produisent des estimations discontinues.

L'idée des fenêtres de Parzen est de placer un petit "noyau" sur chaque observation et de sommer ces contributions:

$$
\hat{p}(x) = \frac{1}{N} \sum_{i=1}^{N} K_\lambda(x - x_i)
$$

Le **noyau** $K_\lambda$ est une fonction qui satisfait $\int K_\lambda(u) \, du = 1$ et $K_\lambda(u) = K_\lambda(-u)$. Le paramètre $\lambda$ contrôle la **largeur de bande** (bandwidth): plus $\lambda$ est grand, plus le noyau est étalé, plus l'estimation est lisse.

Le noyau le plus courant est le noyau gaussien:

$$
K_\lambda(u) = \frac{1}{\lambda \sqrt{2\pi}} \exp\left(-\frac{u^2}{2\lambda^2}\right)
$$

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Generate samples from a mixture of Gaussians
np.random.seed(42)
n = 100
samples = np.concatenate([
    np.random.randn(n//2) * 0.5 + 2,
    np.random.randn(n//2) * 0.8 + 5
])

def gaussian_kernel(u, bandwidth):
    return np.exp(-u**2 / (2 * bandwidth**2)) / (bandwidth * np.sqrt(2 * np.pi))

def kde(x_query, samples, bandwidth):
    density = np.zeros_like(x_query)
    for xi in samples:
        density += gaussian_kernel(x_query - xi, bandwidth)
    return density / len(samples)

x = np.linspace(-1, 9, 500)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
bandwidths = [0.2, 0.5, 1.5]

for ax, bw in zip(axes, bandwidths):
    density = kde(x, samples, bw)
    
    ax.fill_between(x, density, alpha=0.3, color='C0')
    ax.plot(x, density, 'C0-', linewidth=2, label=f'KDE ($\\lambda={bw}$)')
    ax.scatter(samples, np.zeros_like(samples) - 0.02, c='k', s=10, alpha=0.5, marker='|')
    ax.set_xlabel('$x$')
    ax.set_ylabel(r'$\hat{p}(x)$')
    ax.set_title(f'Largeur de bande $\\lambda = {bw}$')
    ax.set_xlim(-1, 9)
    ax.set_ylim(-0.05, 0.6)

plt.tight_layout()
```

Avec une petite largeur de bande ($\lambda = 0.2$), chaque observation crée un pic distinct et l'estimation est très variable. Avec une grande largeur de bande ($\lambda = 1.5$), les détails sont perdus et la structure bimodale des données est masquée. Le choix de $\lambda$ incarne encore une fois le compromis biais-variance.

## Régression de Nadaraya-Watson

L'estimation de densité par noyaux mène naturellement à une forme de régression plus souple que les k-ppv. Plutôt que de sélectionner exactement $k$ voisins et de les traiter également, nous pouvons pondérer *tous* les points selon leur proximité à la requête.

L'**estimateur de Nadaraya-Watson** définit la prédiction comme une moyenne pondérée:

$$
\hat{y}(\mathbf{x}) = \frac{\sum_{i=1}^{N} K_\lambda(\mathbf{x} - \mathbf{x}_i) \, y_i}{\sum_{i=1}^{N} K_\lambda(\mathbf{x} - \mathbf{x}_i)} = \sum_{i=1}^{N} w_i(\mathbf{x}) \, y_i
$$

où les poids sont normalisés:

$$
w_i(\mathbf{x}) = \frac{K_\lambda(\mathbf{x} - \mathbf{x}_i)}{\sum_{j=1}^{N} K_\lambda(\mathbf{x} - \mathbf{x}_j)}
$$

Chaque point d'entraînement contribue à la prédiction, mais les points éloignés ont un poids négligeable. Le noyau agit comme une fenêtre qui détermine l'influence locale.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Generate 1D regression data
np.random.seed(42)
n = 50
X_train = np.sort(np.random.uniform(0, 10, n))
y_train = np.sin(X_train) + np.random.randn(n) * 0.3

def gaussian_kernel(u, bandwidth):
    return np.exp(-u**2 / (2 * bandwidth**2))

def nadaraya_watson(X_train, y_train, X_test, bandwidth):
    predictions = []
    for x in X_test:
        weights = gaussian_kernel(X_train - x, bandwidth)
        if np.sum(weights) > 1e-10:
            predictions.append(np.sum(weights * y_train) / np.sum(weights))
        else:
            predictions.append(0)
    return np.array(predictions)

X_test = np.linspace(0, 10, 200)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
bandwidths = [0.2, 0.5, 1.5]

for ax, bw in zip(axes, bandwidths):
    y_pred = nadaraya_watson(X_train, y_train, X_test, bw)
    
    ax.scatter(X_train, y_train, c='C0', s=30, alpha=0.6, label='Données')
    ax.plot(X_test, y_pred, 'C1-', linewidth=2, label=f'Nadaraya-Watson')
    ax.plot(X_test, np.sin(X_test), 'k--', alpha=0.5, label=r'$\sin(x)$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'$\\lambda = {bw}$')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, 10)

plt.tight_layout()
```

Comparé aux k-ppv, Nadaraya-Watson produit des prédictions plus lisses car la transition entre voisins est graduelle plutôt qu'abrupte. Le paramètre $\lambda$ joue un rôle analogue à $k$: une petite largeur de bande donne une courbe qui suit de près les données (haute variance), une grande largeur de bande lisse excessivement (haut biais).

Les deux approches, k-ppv et Nadaraya-Watson, sont des méthodes à **moyennes locales**. Elles estiment $\mathbb{E}[Y \mid \mathbf{X} = \mathbf{x}]$ en faisant une moyenne pondérée des $y_i$ pour les points $\mathbf{x}_i$ proches de $\mathbf{x}$. La différence réside dans la définition de "proche": les k-ppv utilisent une frontière nette (les $k$ plus proches), tandis que Nadaraya-Watson utilise une pondération douce (le noyau).

```{admonition} Lien avec le mécanisme d'attention
:class: note

La formule de Nadaraya-Watson ressemble étonnamment au **mécanisme d'attention** qui a révolutionné l'apprentissage profond. Dans les deux cas, la sortie est une moyenne pondérée où les poids sont normalisés (somme égale à 1):

$$
\text{Nadaraya-Watson:} \quad \hat{y}(\mathbf{x}) = \sum_{i} \frac{K(\mathbf{x}, \mathbf{x}_i)}{\sum_j K(\mathbf{x}, \mathbf{x}_j)} y_i
$$

En 2014, Bahdanau, Cho et Bengio, alors à Mila (Montréal), ont introduit le mécanisme d'attention pour la traduction automatique neuronale. L'idée: plutôt que de compresser toute une phrase source en un vecteur fixe, le décodeur peut "regarder" différentes parties de la phrase source à chaque étape, avec des poids d'attention appris. Ce mécanisme a ensuite été généralisé dans l'architecture **Transformer** (Vaswani et al., 2017), qui est à la base des grands modèles de langage comme GPT et Claude.

La connexion n'est pas que superficielle: Nadaraya-Watson et l'attention résolvent le même problème fondamental. Comment agréger de l'information provenant de plusieurs sources (voisins ou tokens) de manière différenciée selon leur pertinence pour la requête?
```

## Complexité

L'entraînement consiste à stocker les données: $O(N)$. L'inférence requiert de calculer la distance à tous les points et d'identifier les $k$ plus proches: $O(Nd)$ par requête. Pour de grands ensembles, des structures comme les arbres k-d ou le hachage sensible à la localité (LSH) réduisent ce coût.

## Passage à l'échelle

L'implémentation naïve des k-ppv calcule la distance entre la requête et chaque point d'entraînement: $O(Nd)$ par requête. Pour un million de points en dimension 100, chaque prédiction requiert 100 millions d'opérations. Cette complexité linéaire en $N$ rend la méthode impraticable pour de grands ensembles de données.

Comment accélérer la recherche? L'idée clé est d'**éviter de calculer des distances inutiles**. Si nous pouvons éliminer rapidement des régions entières de l'espace qui ne peuvent pas contenir le plus proche voisin, nous gagnons du temps.

### Arbres k-d

Un **arbre k-d** est une structure d'indexation qui découpe l'espace en boîtes. Lors d'une requête, on peut **élaguer** (ignorer) toute boîte dont la distance minimale à la requête dépasse la meilleure distance trouvée jusqu'ici.

### Élaguer par boîtes englobantes

Chaque nœud de l'arbre correspond à une région rectangulaire (une "boîte" alignée sur les axes). Pour décider si une boîte peut contenir un point plus proche que notre meilleur candidat actuel, on calcule la **distance minimale** entre la requête et la boîte.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: box that CAN contain a closer point
ax = axes[0]
query = np.array([6, 4])
best_dist = 3.5
box1 = {'xmin': 1, 'xmax': 4, 'ymin': 2, 'ymax': 6}

# Draw box
rect = Rectangle((box1['xmin'], box1['ymin']), 
                 box1['xmax'] - box1['xmin'], box1['ymax'] - box1['ymin'],
                 facecolor='C0', alpha=0.3, edgecolor='C0', linewidth=2)
ax.add_patch(rect)

# Draw query and search radius
ax.scatter(*query, s=150, c='red', marker='*', zorder=5, label='Requête')
circle = plt.Circle(query, best_dist, fill=False, color='C1', linewidth=2, linestyle='--', label=f'Rayon = {best_dist}')
ax.add_patch(circle)

# Closest point on box to query
closest_on_box = np.array([box1['xmax'], query[1]])  # right edge of box
ax.plot([query[0], closest_on_box[0]], [query[1], closest_on_box[1]], 'k-', linewidth=1.5)
dist_to_box = np.linalg.norm(query - closest_on_box)
ax.scatter(*closest_on_box, s=80, c='black', zorder=5)

ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_aspect('equal')
ax.set_title(f'Distance à la boîte = {dist_to_box:.1f} < {best_dist}\n→ Explorer cette région')
ax.legend(loc='upper right')

# Right: box that CANNOT contain a closer point
ax = axes[1]
box2 = {'xmin': 0, 'xmax': 2, 'ymin': 0, 'ymax': 2}

rect = Rectangle((box2['xmin'], box2['ymin']), 
                 box2['xmax'] - box2['xmin'], box2['ymax'] - box2['ymin'],
                 facecolor='gray', alpha=0.3, edgecolor='gray', linewidth=2)
ax.add_patch(rect)

ax.scatter(*query, s=150, c='red', marker='*', zorder=5, label='Requête')
circle = plt.Circle(query, best_dist, fill=False, color='C1', linewidth=2, linestyle='--', label=f'Rayon = {best_dist}')
ax.add_patch(circle)

# Closest point on box to query (corner)
closest_on_box2 = np.array([box2['xmax'], box2['ymax']])
ax.plot([query[0], closest_on_box2[0]], [query[1], closest_on_box2[1]], 'k-', linewidth=1.5)
dist_to_box2 = np.linalg.norm(query - closest_on_box2)
ax.scatter(*closest_on_box2, s=80, c='black', zorder=5)

ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_aspect('equal')
ax.set_title(f'Distance à la boîte = {dist_to_box2:.1f} > {best_dist}\n→ Élaguer (ignorer)')
ax.legend(loc='upper right')

plt.tight_layout()
```

**Règle d'élagage**: Si la distance minimale de la requête à une boîte dépasse la meilleure distance trouvée, aucun point dans cette boîte ne peut être plus proche. On peut ignorer toute la région.

### Structure de l'arbre

L'arbre k-d partitionne récursivement l'espace. À chaque nœud, on choisit une dimension et on divise les points selon leur coordonnée sur cette dimension (au niveau de la médiane). On alterne les dimensions à chaque niveau.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

np.random.seed(0)
# Use fewer points for clarity
points = np.array([[2, 7], [5, 4], [9, 6], [4, 2], [8, 1], [7, 8]])
labels = ['A', 'B', 'C', 'D', 'E', 'F']

# Build tree structure for visualization
class KDNode:
    def __init__(self, point_idx, dim, split_val, left=None, right=None, bounds=None):
        self.point_idx = point_idx
        self.dim = dim
        self.split_val = split_val
        self.left = left
        self.right = right
        self.bounds = bounds  # (x_min, x_max, y_min, y_max)

def build_kdtree(point_indices, depth, x_min, x_max, y_min, y_max):
    if len(point_indices) == 0:
        return None
    if len(point_indices) == 1:
        return KDNode(point_indices[0], None, None, bounds=(x_min, x_max, y_min, y_max))
    
    dim = depth % 2
    pts = points[point_indices]
    sorted_order = np.argsort(pts[:, dim])
    median_pos = len(sorted_order) // 2
    median_idx = point_indices[sorted_order[median_pos]]
    split_val = points[median_idx, dim]
    
    left_indices = point_indices[sorted_order[:median_pos]]
    right_indices = point_indices[sorted_order[median_pos+1:]]
    
    if dim == 0:
        left = build_kdtree(left_indices, depth+1, x_min, split_val, y_min, y_max)
        right = build_kdtree(right_indices, depth+1, split_val, x_max, y_min, y_max)
    else:
        left = build_kdtree(left_indices, depth+1, x_min, x_max, y_min, split_val)
        right = build_kdtree(right_indices, depth+1, x_min, x_max, split_val, y_max)
    
    return KDNode(median_idx, dim, split_val, left, right, (x_min, x_max, y_min, y_max))

root = build_kdtree(np.arange(len(points)), 0, 0, 10, 0, 10)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: spatial view
ax = axes[0]
colors = ['C0', 'C1', 'C2']

def draw_splits(node, depth, ax):
    if node is None or node.dim is None:
        return
    x_min, x_max, y_min, y_max = node.bounds
    color = colors[min(depth, len(colors)-1)]
    if node.dim == 0:
        ax.plot([node.split_val, node.split_val], [y_min, y_max], 
                color=color, linewidth=2.5, alpha=0.8)
    else:
        ax.plot([x_min, x_max], [node.split_val, node.split_val], 
                color=color, linewidth=2.5, alpha=0.8)
    draw_splits(node.left, depth+1, ax)
    draw_splits(node.right, depth+1, ax)

draw_splits(root, 0, ax)
ax.scatter(points[:, 0], points[:, 1], s=120, c='black', zorder=5)
for i, (p, label) in enumerate(zip(points, labels)):
    ax.annotate(label, p, xytext=(5, 5), textcoords='offset points', fontsize=12, fontweight='bold')

ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 10.5)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_aspect('equal')
ax.set_title('Partition de l\'espace')

from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=colors[i], linewidth=2.5, 
                          label=f'Profondeur {i} ({"$x_1$" if i%2==0 else "$x_2$"})') 
                   for i in range(3)]
ax.legend(handles=legend_elements, loc='lower right')

# Right: tree structure
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Structure de l\'arbre')

# Draw tree manually with positions
node_positions = {}
def assign_positions(node, x, y, dx, positions):
    if node is None:
        return
    positions[node.point_idx] = (x, y)
    if node.left:
        assign_positions(node.left, x - dx, y - 2, dx * 0.5, positions)
    if node.right:
        assign_positions(node.right, x + dx, y - 2, dx * 0.5, positions)

assign_positions(root, 5, 9, 2.2, node_positions)

# Draw edges first
def draw_edges(node, ax, positions):
    if node is None:
        return
    x, y = positions[node.point_idx]
    if node.left:
        lx, ly = positions[node.left.point_idx]
        ax.plot([x, lx], [y-0.4, ly+0.4], 'k-', linewidth=1.5, zorder=1)
        ax.text((x+lx)/2 - 0.3, (y+ly)/2, 'G', fontsize=9, color='gray')
    if node.right:
        rx, ry = positions[node.right.point_idx]
        ax.plot([x, rx], [y-0.4, ry+0.4], 'k-', linewidth=1.5, zorder=1)
        ax.text((x+rx)/2 + 0.2, (y+ry)/2, 'D', fontsize=9, color='gray')
    draw_edges(node.left, ax, positions)
    draw_edges(node.right, ax, positions)

draw_edges(root, ax, node_positions)

# Draw nodes
def draw_nodes(node, depth, ax, positions):
    if node is None:
        return
    x, y = positions[node.point_idx]
    color = colors[min(depth, len(colors)-1)] if node.dim is not None else 'lightgray'
    
    # Node box
    bbox = FancyBboxPatch((x-0.5, y-0.35), 1, 0.7, boxstyle="round,pad=0.05",
                          facecolor=color, edgecolor='black', linewidth=1.5, 
                          alpha=0.8, zorder=3)
    ax.add_patch(bbox)
    ax.text(x, y, labels[node.point_idx], ha='center', va='center', 
            fontsize=14, fontweight='bold', zorder=4)
    
    # Split info
    if node.dim is not None:
        dim_name = '$x_1$' if node.dim == 0 else '$x_2$'
        ax.text(x, y-0.6, f'{dim_name} < {node.split_val:.0f}?', ha='center', 
                va='top', fontsize=9, color='black')
    
    draw_nodes(node.left, depth+1, ax, positions)
    draw_nodes(node.right, depth+1, ax, positions)

draw_nodes(root, 0, ax, node_positions)

plt.tight_layout()
```

Chaque nœud interne correspond à une question "$x_j < t$?". Les branches gauche (G) et droite (D) correspondent aux réponses "oui" et "non". Les feuilles (en gris) contiennent les points.

### Déroulement d'une requête

Voici comment se déroule la recherche du plus proche voisin:

1. **Descente**: Suivre l'arbre jusqu'à une feuille contenant la requête (comme dans un arbre binaire de recherche)
2. **Initialisation**: Le point dans cette feuille devient notre meilleur candidat
3. **Remontée avec élagage**: En remontant, pour chaque nœud ancêtre:
   - Si l'autre côté de la division *pourrait* contenir un point plus proche (test de la boîte), l'explorer
   - Sinon, l'élaguer

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

np.random.seed(42)
points = np.random.rand(15, 2) * 10
query = np.array([7.5, 3.0])

distances = np.sqrt(np.sum((points - query)**2, axis=1))
nn_idx = np.argmin(distances)
nn_dist = distances[nn_idx]

splits = []
def build_kdtree_splits(points_idx, depth, x_min, x_max, y_min, y_max):
    if len(points_idx) <= 1:
        return
    dim = depth % 2
    pts = points[points_idx]
    sorted_idx = np.argsort(pts[:, dim])
    median_pos = len(sorted_idx) // 2
    median_val = pts[sorted_idx[median_pos], dim]
    splits.append((dim, median_val, depth, x_min, x_max, y_min, y_max))
    left_idx = points_idx[sorted_idx[:median_pos]]
    right_idx = points_idx[sorted_idx[median_pos:]]
    if dim == 0:
        build_kdtree_splits(left_idx, depth + 1, x_min, median_val, y_min, y_max)
        build_kdtree_splits(right_idx, depth + 1, median_val, x_max, y_min, y_max)
    else:
        build_kdtree_splits(left_idx, depth + 1, x_min, x_max, y_min, median_val)
        build_kdtree_splits(right_idx, depth + 1, x_min, x_max, median_val, y_max)

build_kdtree_splits(np.arange(len(points)), 0, 0, 10, 0, 10)

fig, ax = plt.subplots(figsize=(8, 6))

# Pruned region (left side)
rect = Rectangle((0, 0), 4.5, 10, facecolor='gray', alpha=0.25, edgecolor='none')
ax.add_patch(rect)

for dim, val, depth, x_min, x_max, y_min, y_max in splits:
    alpha = 0.6
    if dim == 0:
        ax.plot([val, val], [y_min, y_max], color='gray', linewidth=1.5, alpha=alpha)
    else:
        ax.plot([x_min, x_max], [val, val], color='gray', linewidth=1.5, alpha=alpha)

circle = Circle(query, nn_dist, fill=False, color='C1', linewidth=2, linestyle='--')
ax.add_patch(circle)

ax.scatter(points[:, 0], points[:, 1], s=80, c='C0', zorder=5, label='Points')
ax.scatter(points[nn_idx, 0], points[nn_idx, 1], s=120, c='C2', zorder=6, 
           edgecolors='black', linewidths=2, label='Plus proche voisin')
ax.scatter(query[0], query[1], s=150, c='red', marker='*', zorder=7, label='Requête')
ax.plot([query[0], points[nn_idx, 0]], [query[1], points[nn_idx, 1]], 
        'C2--', linewidth=1.5, alpha=0.7)

ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 10.5)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_aspect('equal')
ax.legend(loc='upper left')
ax.annotate('Région élaguée\n(distance min > rayon)', xy=(2, 5), fontsize=10, 
            ha='center', color='gray')
ax.set_title('Le cercle pointillé montre le rayon de recherche actuel')
plt.tight_layout()
```

### Limites en haute dimension

```{admonition} Règle pratique
:class: tip

- **Fonctionne bien**: dimension $d \lesssim 10$ à $20$, données bien distribuées
- **Perd son avantage**: dimension $d > 20$, on finit par visiter presque tous les nœuds
- **Alternative en haute dimension**: méthodes approximatives (section suivante)
```

Pourquoi l'arbre k-d échoue en haute dimension? Le test d'élagage compare la distance à une boîte avec le rayon de recherche. En haute dimension, les boîtes deviennent des "hyper-rectangles" avec beaucoup de faces. La distance minimale à une boîte reste souvent faible même quand la boîte est "loin" intuitivement, car il suffit qu'une seule coordonnée soit proche. Résultat: peu de régions sont élaguées, et on retombe sur une complexité proche de $O(N)$.

### En pratique

En scikit-learn:

```python
from sklearn.neighbors import KNeighborsClassifier

# algorithm='auto' choisit automatiquement entre brute, kd_tree, ball_tree
clf = KNeighborsClassifier(n_neighbors=5, algorithm='auto')

# Pour forcer un algorithme spécifique:
clf_kd = KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree')
clf_brute = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
```

L'option `algorithm='auto'` choisit la meilleure stratégie selon la taille et la dimension des données.

### Recherche approximative

Quand la recherche exacte est trop coûteuse, on peut accepter des voisins *approximatifs*. Les méthodes de **recherche approximative des plus proches voisins** (approximate nearest neighbors, ANN) garantissent de trouver des points qui sont proches, sans garantir qu'ils soient les plus proches.

Le **hachage sensible à la localité** (locality-sensitive hashing, LSH) projette les points dans un espace de hachage où les points proches ont une forte probabilité de collision. Plusieurs tables de hachage avec des fonctions différentes permettent d'atteindre un bon rappel. La complexité devient sous-linéaire en $N$, au prix d'une approximation.

Des bibliothèques comme **FAISS** (Facebook AI Similarity Search) et **Annoy** (Approximate Nearest Neighbors Oh Yeah) implémentent ces algorithmes et permettent de chercher parmi des milliards de vecteurs. Ces outils sont essentiels pour les systèmes de recommandation et la recherche sémantique à grande échelle, où les représentations vectorielles (embeddings) de documents, images ou produits sont comparées par similarité.

```{admonition} Bases de données vectorielles et RAG
:class: tip

Le terme **base de données vectorielle** (*vector database*) est devenu courant dans l'industrie. Il désigne essentiellement un système de recherche approximative des plus proches voisins optimisé pour les embeddings de haute dimension. Sous le capot, ce sont les mêmes algorithmes: LSH, graphes de proximité (HNSW), ou quantification de produits.

Ces systèmes sont au cœur des architectures **RAG** (Retrieval-Augmented Generation) utilisées avec les grands modèles de langage. Le principe: plutôt que de tout mémoriser dans les paramètres du modèle, on stocke des documents dans une base vectorielle. À chaque requête, on retrouve les documents pertinents par similarité cosinus sur leurs embeddings, puis on les fournit au modèle comme contexte.

La similarité cosinus $\text{sim}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}$ est particulièrement adaptée aux embeddings de texte car elle ignore la magnitude des vecteurs et se concentre sur leur direction, capturant ainsi la similarité sémantique plutôt que la longueur des documents.
```

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)

# Generate points in 2D
n_points = 40
points = np.random.randn(n_points, 2)

# Query point
query = np.array([0.5, 0.3])

# Find true 5 nearest neighbors
k = 5
distances = np.sqrt(np.sum((points - query)**2, axis=1))
true_nn_idx = np.argsort(distances)[:k]

# Simulate approximate nearest neighbors (miss one, include one wrong)
approx_nn_idx = np.array([true_nn_idx[0], true_nn_idx[1], true_nn_idx[2], 
                          true_nn_idx[4], np.argsort(distances)[k+1]])

# Points that are in both
common_idx = np.intersect1d(true_nn_idx, approx_nn_idx)
# Only in true
only_true = np.setdiff1d(true_nn_idx, approx_nn_idx)
# Only in approx
only_approx = np.setdiff1d(approx_nn_idx, true_nn_idx)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Left: Exact search
ax = axes[0]
ax.scatter(points[:, 0], points[:, 1], s=50, c='lightgray', alpha=0.7)
ax.scatter(points[true_nn_idx, 0], points[true_nn_idx, 1], s=100, c='C2', 
           edgecolors='black', linewidths=1.5, label=f'{k} plus proches (vrais)')
ax.scatter(query[0], query[1], s=200, c='red', marker='*', zorder=10, label='Requête')

# Draw circle for k-th distance
kth_dist = distances[true_nn_idx[-1]]
circle = plt.Circle(query, kth_dist, fill=False, color='C2', linestyle='--', linewidth=1.5)
ax.add_patch(circle)

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Recherche exacte')
ax.legend(loc='upper left', fontsize=9)

# Right: Approximate search
ax = axes[1]
ax.scatter(points[:, 0], points[:, 1], s=50, c='lightgray', alpha=0.7)

# Common points (found by both)
ax.scatter(points[common_idx, 0], points[common_idx, 1], s=100, c='C2', 
           edgecolors='black', linewidths=1.5, label='Trouvés (corrects)')

# Missed by approximate
ax.scatter(points[only_true, 0], points[only_true, 1], s=100, c='C3', 
           edgecolors='black', linewidths=1.5, marker='s', label='Manqués')

# False positives from approximate
ax.scatter(points[only_approx, 0], points[only_approx, 1], s=100, c='C1', 
           edgecolors='black', linewidths=1.5, marker='^', label='Faux positifs')

ax.scatter(query[0], query[1], s=200, c='red', marker='*', zorder=10, label='Requête')

ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title(f'Recherche approximative (rappel = {len(common_idx)}/{k})')
ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
```

### Compromis précision-vitesse

Le choix entre recherche exacte et approximative dépend de l'application. Pour un diagnostic médical, une erreur dans l'identification des cas similaires peut avoir des conséquences graves: la recherche exacte est préférable. Pour suggérer des vidéos similaires sur une plateforme de streaming, quelques voisins manqués importent peu si les suggestions restent pertinentes.

| Méthode | Complexité requête | Exacte | Dimension |
|---------|-------------------|--------|-----------|
| Force brute | $O(Nd)$ | Oui | Toute |
| Arbre k-d | $O(\log N)$ à $O(N)$ | Oui | $d \lesssim 20$ |
| LSH | $O(1)$ à $O(N^{\rho})$ | Non | Haute |
| Graphes de proximité | $O(\log N)$ | Non | Haute |

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Simulated query times for different methods
N = np.logspace(2, 7, 50)  # 100 to 10 million points

# Brute force: O(N)
brute_force = N * 1e-7

# k-d tree: O(log N) in low dim, degrades to O(N) in high dim
# We show the ideal low-dim case
kd_tree = np.log2(N) * 1e-5

# ANN (e.g., HNSW): nearly constant with slight log factor
ann = np.log2(N) * 5e-6 + 1e-4

fig, ax = plt.subplots(figsize=(8, 5))

ax.loglog(N, brute_force, 'C0-', linewidth=2, label='Force brute $O(N)$')
ax.loglog(N, kd_tree, 'C1-', linewidth=2, label=r'Arbre k-d $O(\log N)$')
ax.loglog(N, ann, 'C2-', linewidth=2, label=r'ANN $O(\log N)$')

# Add shaded regions for practical regimes
ax.axvspan(100, 1e4, alpha=0.1, color='C0')
ax.axvspan(1e4, 1e6, alpha=0.1, color='C1')
ax.axvspan(1e6, 1e7, alpha=0.1, color='C2')

ax.set_xlabel('Taille de l\'ensemble $N$')
ax.set_ylabel('Temps par requête (s)')
ax.legend(loc='upper left')
ax.set_xlim(100, 1e7)
ax.grid(True, alpha=0.3, which='both')

# Annotations
ax.annotate('Petit\nensemble', xy=(500, 1e-3), fontsize=9, ha='center', alpha=0.7)
ax.annotate('Moyen', xy=(1e5, 1e-3), fontsize=9, ha='center', alpha=0.7)
ax.annotate('Grande\néchelle', xy=(3e6, 1e-3), fontsize=9, ha='center', alpha=0.7)

plt.tight_layout()
```

La figure illustre comment le temps de requête évolue avec la taille de l'ensemble. La force brute devient rapidement prohibitive. Les méthodes approximatives maintiennent des temps de réponse acceptables même pour des millions de points.

## Méthodes paramétriques et non paramétriques

Les k-ppv sont une méthode **non paramétrique**: les données sont le modèle. Il n'y a pas de paramètres appris; les prédictions consultent directement l'ensemble d'entraînement. La complexité du modèle croît avec $N$.

| | Non paramétrique | Paramétrique |
|--|------------------|--------------|
| **Modèle** | Les données | Un vecteur $\boldsymbol{\theta} \in \mathbb{R}^p$ |
| **Complexité** | Croît avec $N$ | Fixe |
| **Inférence** | Requiert les données | Requiert seulement $\boldsymbol{\theta}$ |

Les méthodes **paramétriques** distillent l'information dans un vecteur de paramètres de taille fixe. Un réseau de neurones entraîné sur des milliards d'exemples n'a besoin que de ses poids pour faire des prédictions, pas des données d'entraînement.

## Résumé

Les k plus proches voisins classifient un point par vote majoritaire parmi ses $k$ voisins les plus proches. Le paramètre $k$ contrôle le compromis biais-variance. Le choix de la distance encode les hypothèses sur la similarité. Le fléau de la dimensionnalité limite l'efficacité en haute dimension.

La méthode illustre la tension entre mémorisation et généralisation: avec $k=1$, l'erreur d'entraînement est nulle mais la généralisation est mauvaise. Elle illustre aussi la distinction entre approches non paramétriques (les données sont le modèle) et paramétriques (un vecteur de paramètres résume les données).

Le chapitre suivant développe l'approche paramétrique: l'apprentissage comme problème d'optimisation, où nous cherchons les paramètres qui minimisent une fonction de perte.

## Exercices

````{admonition} Exercice 1: Classification manuelle
:class: hint dropdown

Considérez les points d'entraînement suivants en 2D:

| Point | $x_1$ | $x_2$ | Classe |
|-------|-------|-------|--------|
| A | 0 | 0 | 0 |
| B | 1 | 0 | 0 |
| C | 0 | 1 | 1 |
| D | 2 | 2 | 1 |
| E | 3 | 1 | 1 |

1. Pour le point requête $\mathbf{x} = (1, 1)$, identifiez les 3 plus proches voisins avec la distance euclidienne. Quelle est la prédiction du 3-ppv?

2. Répétez avec $k = 1$ et $k = 5$. Les prédictions changent-elles?

3. Calculez les distances avec la norme $\ell_1$ (Manhattan). Les 3 plus proches voisins sont-ils les mêmes?
````

```{admonition} Solution Exercice 1
:class: dropdown

1. **Distances euclidiennes depuis $\mathbf{x} = (1, 1)$:**

   | Point | Distance $\ell_2$ | Classe |
   |-------|-------------------|--------|
   | A | $\sqrt{(1-0)^2 + (1-0)^2} = \sqrt{2} \approx 1.41$ | 0 |
   | B | $\sqrt{(1-1)^2 + (1-0)^2} = 1$ | 0 |
   | C | $\sqrt{(1-0)^2 + (1-1)^2} = 1$ | 1 |
   | D | $\sqrt{(1-2)^2 + (1-2)^2} = \sqrt{2} \approx 1.41$ | 1 |
   | E | $\sqrt{(1-3)^2 + (1-1)^2} = 2$ | 1 |
   
   Les 3 plus proches: B (0), C (1), puis A ou D (égalité). 
   - Si on prend B, C, A: votes = {0: 2, 1: 1} → **prédiction: classe 0**
   - Si on prend B, C, D: votes = {0: 1, 1: 2} → **prédiction: classe 1**

2. **Effet de $k$:**
   - $k = 1$: Plus proche = B ou C (égalité à distance 1). Prédiction dépend du choix.
   - $k = 5$: Tous les points. Votes = {0: 2, 1: 3} → **prédiction: classe 1**

3. **Distances Manhattan ($\ell_1$):**

   | Point | Distance $\ell_1$ |
   |-------|-------------------|
   | A | $|1-0| + |1-0| = 2$ |
   | B | $|1-1| + |1-0| = 1$ |
   | C | $|1-0| + |1-1| = 1$ |
   | D | $|1-2| + |1-2| = 2$ |
   | E | $|1-3| + |1-1| = 2$ |
   
   Les 3 plus proches avec $\ell_1$: B, C, puis A/D/E (égalité). Les deux plus proches (B, C) sont identiques, mais le troisième peut différer selon le critère de départage.
```

````{admonition} Exercice 2: Effet de la normalisation
:class: hint dropdown

Un jeu de données contient deux variables: l'âge (entre 20 et 70 ans) et le revenu annuel (entre 20 000 et 200 000 dollars).

1. Calculez la distance euclidienne entre les points $\mathbf{x}_1 = (30, 50000)$ et $\mathbf{x}_2 = (35, 51000)$.

2. Calculez la distance entre $\mathbf{x}_1 = (30, 50000)$ et $\mathbf{x}_3 = (31, 150000)$.

3. Laquelle des deux paires est "plus proche"? Ce résultat est-il intuitivement raisonnable?

4. Proposez une transformation des données qui rendrait les deux variables comparables. Recalculez les distances après transformation.
````

```{admonition} Solution Exercice 2
:class: dropdown

1. **Distance $\mathbf{x}_1$ à $\mathbf{x}_2$:**
   $$d(\mathbf{x}_1, \mathbf{x}_2) = \sqrt{(35-30)^2 + (51000-50000)^2} = \sqrt{25 + 1000000} \approx 1000$$

2. **Distance $\mathbf{x}_1$ à $\mathbf{x}_3$:**
   $$d(\mathbf{x}_1, \mathbf{x}_3) = \sqrt{(31-30)^2 + (150000-50000)^2} = \sqrt{1 + 10^{10}} \approx 100000$$

3. **Comparaison:** Selon la distance euclidienne, $(\mathbf{x}_1, \mathbf{x}_2)$ est 100 fois plus proche que $(\mathbf{x}_1, \mathbf{x}_3)$. 
   
   **Ce n'est pas raisonnable:** la différence de revenu domine complètement. La paire $(\mathbf{x}_1, \mathbf{x}_2)$ diffère de 5 ans et 1000$, tandis que $(\mathbf{x}_1, \mathbf{x}_3)$ diffère de 1 an et 100000$. Intuitivement, on pourrait argumenter que 5 ans de différence d'âge est plus significatif qu'1 an.

4. **Normalisation (standardisation):**
   
   Soit $\mu_{\text{âge}} = 45$, $\sigma_{\text{âge}} = 15$ et $\mu_{\text{revenu}} = 110000$, $\sigma_{\text{revenu}} = 60000$ (valeurs approximatives).
   
   Transformation: $z = (x - \mu) / \sigma$
   
   - $\mathbf{x}_1' = ((30-45)/15, (50000-110000)/60000) = (-1, -1)$
   - $\mathbf{x}_2' = ((35-45)/15, (51000-110000)/60000) = (-0.67, -0.98)$
   - $\mathbf{x}_3' = ((31-45)/15, (150000-110000)/60000) = (-0.93, 0.67)$
   
   Nouvelles distances:
   - $d(\mathbf{x}_1', \mathbf{x}_2') = \sqrt{(-1+0.67)^2 + (-1+0.98)^2} \approx 0.33$
   - $d(\mathbf{x}_1', \mathbf{x}_3') = \sqrt{(-1+0.93)^2 + (-1-0.67)^2} \approx 1.68$
   
   Après normalisation, les deux variables contribuent équitablement.
```

````{admonition} Exercice 3: Compromis biais-variance
:class: hint dropdown

Soit un problème de régression 1D où la vraie fonction est $f(x) = \sin(2\pi x)$ et les observations sont bruitées: $y = f(x) + \varepsilon$ avec $\varepsilon \sim \mathcal{N}(0, 0.1)$.

1. Générez 50 points d'entraînement uniformément répartis sur $[0, 1]$.

2. Implémentez la régression k-ppv et tracez les prédictions pour $k = 1, 5, 20, 50$.

3. Calculez l'erreur quadratique moyenne (MSE) sur un ensemble de test de 200 points pour chaque valeur de $k$.

4. Tracez le MSE en fonction de $k$. Quelle valeur de $k$ minimise l'erreur de test?

5. Que se passe-t-il quand $k = N$ (nombre total de points d'entraînement)?
````

````{admonition} Solution Exercice 3
:class: dropdown

```python
import numpy as np

# 1. Génération des données
np.random.seed(42)
X_train = np.sort(np.random.uniform(0, 1, 50))
y_train = np.sin(2 * np.pi * X_train) + np.random.randn(50) * 0.1

X_test = np.linspace(0, 1, 200)
y_test_true = np.sin(2 * np.pi * X_test)

# 2. Implémentation k-ppv régression
def knn_regression(X_train, y_train, X_test, k):
    predictions = []
    for x in X_test:
        distances = np.abs(X_train - x)
        k_nearest = np.argsort(distances)[:k]
        predictions.append(np.mean(y_train[k_nearest]))
    return np.array(predictions)

# 3. & 4. Calcul du MSE pour différents k
k_values = range(1, 51)
mse_values = []
for k in k_values:
    y_pred = knn_regression(X_train, y_train, X_test, k)
    mse = np.mean((y_pred - y_test_true)**2)
    mse_values.append(mse)
```

**Résultats typiques:**
- $k = 1$: MSE élevé (haute variance, la prédiction saute entre points)
- $k = 5$ à $10$: MSE minimal (bon compromis)
- $k = 50$: MSE élevé (haut biais)

**5. Quand $k = N = 50$:**
La prédiction est la moyenne de tous les $y_i$, soit $\hat{y}(x) = \bar{y} \approx 0$ (car $\sin$ est symétrique sur $[0,1]$). Le modèle ignore complètement $x$ et prédit une constante. C'est le cas extrême de sous-apprentissage (biais maximal, variance nulle).
````

````{admonition} Exercice 4: Fléau de la dimensionnalité
:class: hint dropdown

Considérez $N = 1000$ points uniformément distribués dans l'hypercube $[0, 1]^d$.

1. Pour $d = 1, 2, 5, 10, 20, 50, 100$, calculez la distance moyenne au plus proche voisin parmi ces points. Utilisez la simulation Monte Carlo.

2. Tracez cette distance en fonction de $d$. Que constatez-vous?

3. Pour capturer les 10 plus proches voisins (soit 1% des données), quel est le rayon de la boule centrée sur un point arbitraire? Calculez ce rayon pour différentes dimensions.

4. Expliquez pourquoi les k-ppv deviennent inefficaces en haute dimension, même avec beaucoup de données.
````

````{admonition} Solution Exercice 4
:class: dropdown

**1. & 2. Simulation Monte Carlo:**

```python
import numpy as np
from scipy.spatial.distance import cdist

dims = [1, 2, 5, 10, 20, 50, 100]
N = 1000
mean_nn_distances = []

for d in dims:
    X = np.random.uniform(0, 1, (N, d))
    dists = cdist(X, X)
    np.fill_diagonal(dists, np.inf)  # ignorer distance à soi-même
    nn_dists = dists.min(axis=1)
    mean_nn_distances.append(nn_dists.mean())
```

**Résultats typiques:**

| $d$ | Distance moyenne au plus proche voisin |
|-----|----------------------------------------|
| 1 | ~0.001 |
| 2 | ~0.02 |
| 5 | ~0.15 |
| 10 | ~0.35 |
| 20 | ~0.55 |
| 50 | ~0.75 |
| 100 | ~0.85 |

La distance au plus proche voisin augmente avec $d$ et tend vers la diagonale de l'hypercube.

**3. Rayon pour capturer 1% des points:**

Pour une distribution uniforme dans $[0,1]^d$, le volume d'une boule de rayon $r$ doit contenir 1% du volume total. Le volume d'une $d$-boule est $V_d(r) = C_d \cdot r^d$ où $C_d$ est une constante. Pour capturer une fraction $p$ du volume unitaire: $r = p^{1/d}$.

| $d$ | $r = 0.01^{1/d}$ |
|-----|------------------|
| 1 | 0.01 |
| 2 | 0.10 |
| 10 | 0.63 |
| 100 | 0.955 |

En dimension 100, il faut un rayon couvrant 95.5% de chaque axe pour capturer 1% des données!

**4. Pourquoi k-ppv échoue en haute dimension:**

- Les "voisins" ne sont plus locaux: ils couvrent presque tout l'espace
- Tous les points deviennent approximativement équidistants (concentration des distances)
- La moyenne locale devient une moyenne globale, perdant toute information sur la structure locale de $f(x)$
````

````{admonition} Exercice 5: Distances pour texte
:class: hint dropdown

Considérez trois documents représentés par leurs vecteurs de fréquence de mots (bag-of-words) sur un vocabulaire de 5 mots:

| Document | chat | chien | maison | voiture | arbre |
|----------|------|-------|--------|---------|-------|
| $d_1$ | 3 | 0 | 1 | 0 | 2 |
| $d_2$ | 2 | 1 | 0 | 0 | 1 |
| $d_3$ | 0 | 0 | 2 | 3 | 0 |

1. Calculez la distance euclidienne entre chaque paire de documents.

2. Calculez la similarité cosinus entre chaque paire, puis convertissez en distance ($d = 1 - \text{sim}$).

3. Selon chaque mesure, quels sont les deux documents les plus similaires?

4. Pourquoi la similarité cosinus est-elle souvent préférée pour les documents textuels?
````

```{admonition} Solution Exercice 5
:class: dropdown

**Vecteurs:** $d_1 = (3, 0, 1, 0, 2)$, $d_2 = (2, 1, 0, 0, 1)$, $d_3 = (0, 0, 2, 3, 0)$

**1. Distances euclidiennes:**

$$d_E(d_1, d_2) = \sqrt{(3-2)^2 + (0-1)^2 + (1-0)^2 + 0 + (2-1)^2} = \sqrt{1+1+1+1} = 2$$

$$d_E(d_1, d_3) = \sqrt{9 + 0 + 1 + 9 + 4} = \sqrt{23} \approx 4.80$$

$$d_E(d_2, d_3) = \sqrt{4 + 1 + 4 + 9 + 1} = \sqrt{19} \approx 4.36$$

**2. Similarités cosinus:**

$\|d_1\| = \sqrt{9+0+1+0+4} = \sqrt{14}$, $\|d_2\| = \sqrt{4+1+0+0+1} = \sqrt{6}$, $\|d_3\| = \sqrt{0+0+4+9+0} = \sqrt{13}$

$$\text{sim}(d_1, d_2) = \frac{d_1 \cdot d_2}{\|d_1\| \|d_2\|} = \frac{6+0+0+0+2}{\sqrt{14}\sqrt{6}} = \frac{8}{\sqrt{84}} \approx 0.87$$

$$\text{sim}(d_1, d_3) = \frac{0+0+2+0+0}{\sqrt{14}\sqrt{13}} = \frac{2}{\sqrt{182}} \approx 0.15$$

$$\text{sim}(d_2, d_3) = \frac{0+0+0+0+0}{\sqrt{6}\sqrt{13}} = 0$$

**Distances cosinus:** $d_C = 1 - \text{sim}$

| Paire | $d_E$ | $d_C$ |
|-------|-------|-------|
| $(d_1, d_2)$ | 2.00 | 0.13 |
| $(d_1, d_3)$ | 4.80 | 0.85 |
| $(d_2, d_3)$ | 4.36 | 1.00 |

**3. Documents les plus similaires:**
- Distance euclidienne: $(d_1, d_2)$ ✓
- Distance cosinus: $(d_1, d_2)$ ✓

Les deux métriques s'accordent ici.

**4. Pourquoi cosinus pour le texte:**
- **Invariance à la longueur:** Un document 2x plus long a des fréquences 2x plus grandes, mais le même contenu thématique. Cosinus ignore la norme.
- **Focus sur l'orientation:** Cosinus mesure l'angle entre vecteurs, pas leur magnitude. Deux documents sur le même sujet pointent dans la même direction.
- **Documents creux:** Les vecteurs textuels sont très creux (la plupart des mots ont fréquence 0). Cosinus gère bien cette sparsité.
```

````{admonition} Exercice 6: Nadaraya-Watson
:class: hint dropdown

Soit les données d'entraînement: $x = [0, 1, 2, 3]$ et $y = [1, 2, 1.5, 3]$.

1. Avec un noyau gaussien $K_\lambda(u) = \exp(-u^2 / 2\lambda^2)$ et $\lambda = 0.5$, calculez manuellement la prédiction de Nadaraya-Watson pour $x^* = 1.5$.

2. Répétez avec $\lambda = 2$. Comment la prédiction change-t-elle?

3. Implémentez l'estimateur et tracez les prédictions pour $\lambda = 0.2, 0.5, 1, 2$ sur l'intervalle $[0, 3]$.

4. Comparez visuellement avec la régression k-ppv pour $k = 1, 2, 3$. Quelle méthode produit des prédictions plus lisses?
````

```{admonition} Solution Exercice 6
:class: dropdown

**1. Nadaraya-Watson pour $x^* = 1.5$ avec $\lambda = 0.5$:**

Distances: $u_i = x^* - x_i = [1.5, 0.5, -0.5, -1.5]$

Poids du noyau: $K_\lambda(u) = \exp(-u^2 / 2\lambda^2) = \exp(-u^2 / 0.5)$

| $i$ | $x_i$ | $y_i$ | $u_i$ | $K(u_i)$ |
|-----|-------|-------|-------|----------|
| 0 | 0 | 1 | 1.5 | $e^{-4.5} \approx 0.011$ |
| 1 | 1 | 2 | 0.5 | $e^{-0.5} \approx 0.607$ |
| 2 | 2 | 1.5 | -0.5 | $e^{-0.5} \approx 0.607$ |
| 3 | 3 | 3 | -1.5 | $e^{-4.5} \approx 0.011$ |

Somme des poids: $\sum K = 0.011 + 0.607 + 0.607 + 0.011 = 1.236$

Prédiction:
$$\hat{y} = \frac{\sum K(u_i) y_i}{\sum K(u_i)} = \frac{0.011 \times 1 + 0.607 \times 2 + 0.607 \times 1.5 + 0.011 \times 3}{1.236}$$
$$= \frac{0.011 + 1.214 + 0.911 + 0.033}{1.236} = \frac{2.169}{1.236} \approx 1.75$$

**2. Avec $\lambda = 2$:**

Poids: $K(u) = \exp(-u^2 / 8)$

| $i$ | $K(u_i)$ |
|-----|----------|
| 0 | $e^{-0.28} \approx 0.756$ |
| 1 | $e^{-0.03} \approx 0.969$ |
| 2 | $e^{-0.03} \approx 0.969$ |
| 3 | $e^{-0.28} \approx 0.756$ |

$$\hat{y} = \frac{0.756 \times 1 + 0.969 \times 2 + 0.969 \times 1.5 + 0.756 \times 3}{0.756 + 0.969 + 0.969 + 0.756}$$
$$= \frac{0.756 + 1.938 + 1.454 + 2.268}{3.45} = \frac{6.416}{3.45} \approx 1.86$$

Avec $\lambda$ plus grand, tous les points contribuent plus également, et la prédiction se rapproche de la moyenne globale $\bar{y} = 1.875$.

**3. & 4.** Nadaraya-Watson produit des courbes plus lisses car la transition entre voisins est graduelle (poids continus) plutôt qu'abrupte (k-ppv: poids 0 ou 1).
```

````{admonition} Exercice 7: Complexité computationnelle
:class: hint dropdown

Vous développez un système de recommandation pour une plateforme avec 10 millions d'utilisateurs. Chaque utilisateur est représenté par un vecteur de 100 dimensions (embeddings).

1. Combien d'opérations (multiplications et additions) faut-il pour trouver le plus proche voisin d'un utilisateur par force brute?

2. Si chaque opération prend 1 nanoseconde, quel est le temps de réponse pour une requête?

3. Si vous devez traiter 1000 requêtes par seconde, cette approche est-elle viable?

4. Un arbre k-d réduit la complexité à $O(\log N)$ en basse dimension. Pourquoi cette structure n'est-elle pas efficace pour $d = 100$?

5. Proposez une stratégie pour ce problème à grande échelle.
````

```{admonition} Solution Exercice 7
:class: dropdown

**1. Nombre d'opérations:**

Pour calculer la distance euclidienne entre deux vecteurs de dimension $d$:
- $d$ soustractions
- $d$ multiplications (carrés)
- $d-1$ additions
- 1 racine carrée

Soit environ $3d$ opérations par distance.

Pour $N = 10^7$ utilisateurs et $d = 100$:
$$\text{Opérations} = N \times 3d = 10^7 \times 300 = 3 \times 10^9$$

**2. Temps de réponse:**

À 1 ns par opération: $3 \times 10^9 \times 10^{-9}$ s $= 3$ secondes par requête.

**3. Viabilité pour 1000 req/s:**

Non! Il faudrait $3000$ secondes de calcul par seconde, soit $3000$ cœurs CPU dédiés juste pour tenir la charge. Ce n'est pas viable économiquement.

**4. Pourquoi k-d tree échoue en $d = 100$:**

- En haute dimension, les hyperplans de séparation deviennent inefficaces: la plupart des régions de l'espace sont "proches" de la requête
- L'élagage des branches devient rare: on finit par visiter presque tous les nœuds
- La complexité dégénère vers $O(N)$, comme la force brute
- Règle empirique: k-d trees efficaces seulement pour $d \lesssim 20$

**5. Stratégies à grande échelle:**

- **Recherche approximative (ANN):** Utiliser FAISS, Annoy, ou ScaNN. Ces bibliothèques utilisent:
  - **Quantification de produit (PQ):** Compresser les vecteurs pour réduire la mémoire et accélérer les calculs
  - **Graphes de proximité (HNSW):** Naviguer un graphe où les nœuds proches sont connectés
  - **LSH:** Hacher les vecteurs pour que les voisins aient des hachages similaires

- **Réduction de dimension:** Projeter les embeddings de 100 à 32 dimensions via PCA avant la recherche

- **Filtrage en deux étapes:** 
  1. Recherche approximative rapide pour obtenir 1000 candidats
  2. Réordonnancement exact sur ces 1000 candidats

- **Mise en cache:** Pré-calculer et stocker les k-NN des utilisateurs les plus actifs

Avec FAISS optimisé, on peut atteindre ~1ms par requête sur 10M vecteurs, soit 1000 req/s sur un seul serveur.
````

````{admonition} Exercice 8: Élagage dans un arbre k-d
:class: hint dropdown

Cet exercice teste la compétence clé des arbres k-d: décider si une région peut être élaguée.

**Contexte**: Vous cherchez le plus proche voisin d'une requête $q = (5, 3)$. Votre meilleur candidat actuel est à distance $r = 2.5$.

**Question**: Pour chaque boîte ci-dessous, calculez la distance minimale de $q$ à la boîte et décidez si vous pouvez élaguer (ignorer) cette région.

| Boîte | $x_1$ min | $x_1$ max | $x_2$ min | $x_2$ max |
|-------|-----------|-----------|-----------|-----------|
| A | 0 | 2 | 0 | 2 |
| B | 6 | 8 | 2 | 5 |
| C | 3 | 7 | 4 | 6 |
| D | 8 | 10 | 0 | 1 |

**Rappel**: La distance minimale d'un point $q$ à une boîte alignée sur les axes se calcule ainsi:
- Pour chaque dimension $j$: si $q_j < \text{min}_j$, contribution $= (\text{min}_j - q_j)^2$; si $q_j > \text{max}_j$, contribution $= (q_j - \text{max}_j)^2$; sinon contribution $= 0$
- Distance minimale $= \sqrt{\text{somme des contributions}}$
````

````{admonition} Solution Exercice 8
:class: dropdown

Requête: $q = (5, 3)$, rayon actuel: $r = 2.5$

**Boîte A** $[0,2] \times [0,2]$:
- Dimension 1: $q_1 = 5 > 2 = \text{max}$, contribution $= (5-2)^2 = 9$
- Dimension 2: $q_2 = 3 > 2 = \text{max}$, contribution $= (3-2)^2 = 1$
- Distance min $= \sqrt{9 + 1} = \sqrt{10} \approx 3.16 > 2.5$ → **Élaguer** ✓

**Boîte B** $[6,8] \times [2,5]$:
- Dimension 1: $q_1 = 5 < 6 = \text{min}$, contribution $= (6-5)^2 = 1$
- Dimension 2: $q_2 = 3 \in [2,5]$, contribution $= 0$
- Distance min $= \sqrt{1 + 0} = 1 < 2.5$ → **Explorer** (peut contenir un point plus proche)

**Boîte C** $[3,7] \times [4,6]$:
- Dimension 1: $q_1 = 5 \in [3,7]$, contribution $= 0$
- Dimension 2: $q_2 = 3 < 4 = \text{min}$, contribution $= (4-3)^2 = 1$
- Distance min $= \sqrt{0 + 1} = 1 < 2.5$ → **Explorer**

**Boîte D** $[8,10] \times [0,1]$:
- Dimension 1: $q_1 = 5 < 8 = \text{min}$, contribution $= (8-5)^2 = 9$
- Dimension 2: $q_2 = 3 > 1 = \text{max}$, contribution $= (3-1)^2 = 4$
- Distance min $= \sqrt{9 + 4} = \sqrt{13} \approx 3.61 > 2.5$ → **Élaguer** ✓

**Résumé**: On élague A et D, on explore B et C.
````
