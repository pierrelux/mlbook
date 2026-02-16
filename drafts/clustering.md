---
kernelspec:
  name: python3
  display_name: Python 3
---

# Partitionnement

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Formuler le problème de partitionnement comme un problème d'optimisation
- Implémenter l'algorithme k-moyennes et expliquer sa convergence
- Dériver les mises à jour de l'algorithme EM pour les mélanges gaussiens (GMM)
- Expliquer le lien entre k-moyennes et les GMM à covariance sphérique
- Choisir un type de covariance adapté à la géométrie des données
- Utiliser le BIC ou la méthode du coude pour sélectionner le nombre de groupes
```

Jusqu'ici, nous avons travaillé dans un cadre **supervisé**: chaque exemple du jeu d'entraînement était accompagné d'une étiquette. Mais que faire quand ces étiquettes manquent? Si l'on dispose d'un ensemble de mesures sans savoir à quelles catégories elles correspondent, peut-on tout de même découvrir une structure dans les données?

Le **partitionnement** (*clustering*) répond à cette question. L'objectif est de regrouper des observations similaires en **groupes** (*clusters*), sans supervision. Cette tâche apparaît dans de nombreux contextes: segmenter une clientèle selon ses habitudes d'achat, compresser une image en regroupant les pixels de couleur proche, ou identifier des familles de gènes aux fonctions similaires.

Dans ce chapitre, nous commençons par un exemple concret pour fixer les idées, puis nous formalisons le problème avec l'algorithme k-moyennes. Nous verrons ensuite que k-moyennes fait des hypothèses implicites sur la forme des groupes, ce qui nous mènera naturellement aux modèles de mélange gaussien et à l'algorithme EM. Le chapitre se termine par des considérations pratiques sur le choix du nombre de groupes.

```{admonition} Prérequis
:class: note

Ce chapitre suppose une familiarité avec:
- Les distributions gaussiennes multivariées (chapitre 5)
- Le maximum de vraisemblance (chapitre 5)
- Les modèles génératifs et le théorème de Bayes (chapitre 6)
```

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

%config InlineBackend.figure_format = 'retina'
```

## Un exemple pour fixer les idées

Considérons six mesures prises sur une droite: $\{1, 2, 3, 10, 11, 12\}$. En les traçant, on voit spontanément deux groupes: les valeurs proches de 2 et celles proches de 11. Notre intuition s'appuie sur la **proximité**: les points d'un même groupe sont plus proches entre eux qu'ils ne le sont des points de l'autre groupe.

```{code-cell} python
:tags: [hide-input]

points = np.array([1, 2, 3, 10, 11, 12])
fig, ax = plt.subplots(figsize=(8, 1.5))
ax.scatter(points, np.zeros_like(points), c=['steelblue']*3 + ['coral']*3, s=100, zorder=3)
for p in points:
    ax.annotate(str(p), (p, 0.03), ha='center', fontsize=12)
ax.set_xlim(-1, 14)
ax.set_ylim(-0.1, 0.15)
ax.set_xlabel('$x$')
ax.set_yticks([])
ax.set_title('Six points en 1D: deux groupes naturels')
ax.axhline(0, color='gray', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
```

Comment formaliser cette intuition? On pourrait chercher deux «centres» $\mu_1$ et $\mu_2$, puis assigner chaque point au centre le plus proche. Le meilleur choix de centres serait celui qui minimise la somme des distances au carré entre chaque point et son centre. C'est exactement ce que fait l'algorithme k-moyennes.

## K-moyennes

### Formulation du problème

Supposons que l'on veuille partitionner $N$ points $\boldsymbol{x}_1, \ldots, \boldsymbol{x}_N$ en $K$ groupes. On introduit des **centroïdes** $\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_K$ et des **assignations** $r_{nk} \in \{0, 1\}$ (avec $\sum_k r_{nk} = 1$: chaque point appartient à exactement un groupe). L'objectif est de minimiser la **distorsion**:

$$
\mathcal{L} = \sum_{n=1}^N \sum_{k=1}^K r_{nk} \|\boldsymbol{x}_n - \boldsymbol{\mu}_k\|^2
$$

Cette fonction mesure la somme des distances au carré entre chaque point et le centroïde de son groupe. Plus les groupes sont compacts, plus $\mathcal{L}$ est faible.

Minimiser $\mathcal{L}$ simultanément sur les assignations $r_{nk}$ et les centroïdes $\boldsymbol{\mu}_k$ est un problème combinatoire difficile ($K^N$ assignations possibles). L'algorithme k-moyennes contourne cette difficulté en alternant entre deux étapes.

### L'algorithme

**Étape d'assignation.** On fixe les centroïdes et on optimise les assignations. Chaque point est assigné au centroïde le plus proche:

$$
r_{nk} = \begin{cases} 1 & \text{si } k = \arg\min_{k'} \|\boldsymbol{x}_n - \boldsymbol{\mu}_{k'}\|^2 \\ 0 & \text{sinon} \end{cases}
$$

**Étape de mise à jour.** On fixe les assignations et on optimise les centroïdes. La solution est la moyenne des points assignés à chaque groupe:

$$
\boldsymbol{\mu}_k = \frac{\sum_{n=1}^N r_{nk} \, \boldsymbol{x}_n}{\sum_{n=1}^N r_{nk}}
$$

On peut vérifier ce résultat en dérivant $\mathcal{L}$ par rapport à $\boldsymbol{\mu}_k$ et en posant le gradient à zéro.

L'algorithme commence par une initialisation (aléatoire ou par une stratégie plus élaborée, voir plus bas), puis alterne les deux étapes jusqu'à ce que les assignations ne changent plus.

### Exemple travaillé

Reprenons nos six points $\{1, 2, 3, 10, 11, 12\}$ avec $K = 2$, et initialisons les centroïdes à $\mu_1 = 1$ et $\mu_2 = 2$.

**Itération 1, assignation.** On calcule la distance de chaque point aux deux centroïdes:

| $x_n$ | $\|x_n - \mu_1\|^2$ | $\|x_n - \mu_2\|^2$ | Groupe |
|--------|---------------------|---------------------|--------|
| 1      | 0                   | 1                   | 1      |
| 2      | 1                   | 0                   | 2      |
| 3      | 4                   | 1                   | 2      |
| 10     | 81                  | 64                  | 2      |
| 11     | 100                 | 81                  | 2      |
| 12     | 121                 | 100                 | 2      |

Le groupe 1 contient $\{1\}$ et le groupe 2 contient $\{2, 3, 10, 11, 12\}$.

**Itération 1, mise à jour.** Les nouveaux centroïdes sont:
$$\mu_1 = 1, \qquad \mu_2 = \frac{2 + 3 + 10 + 11 + 12}{5} = 7{,}6$$

**Itération 2, assignation.** Avec $\mu_1 = 1$ et $\mu_2 = 7{,}6$:
- Les points $\{1, 2, 3\}$ sont plus proches de $\mu_1 = 1$ (le seuil de bascule est à $\frac{1 + 7{,}6}{2} = 4{,}3$)
- Les points $\{10, 11, 12\}$ sont plus proches de $\mu_2 = 7{,}6$

**Itération 2, mise à jour:**
$$\mu_1 = \frac{1 + 2 + 3}{3} = 2, \qquad \mu_2 = \frac{10 + 11 + 12}{3} = 11$$

**Itération 3.** Les assignations ne changent plus: l'algorithme a convergé. La distorsion finale est $(1-2)^2 + (2-2)^2 + (3-2)^2 + (10-11)^2 + (11-11)^2 + (12-11)^2 = 4$.

Cet exemple montre que même avec une initialisation maladroite ($\mu_1 = 1$, $\mu_2 = 2$, tous deux dans le premier groupe), l'algorithme retrouve les deux groupes naturels en quelques itérations. Mais ce n'est pas toujours le cas: l'initialisation peut mener à des solutions sous-optimales.

```{code-cell} python
:tags: [hide-input]

points = np.array([1., 2., 3., 10., 11., 12.])
fig, axes = plt.subplots(1, 3, figsize=(14, 2.5))

history = [
    {'mu': [1.0, 2.0], 'groups': [[0], [1, 2, 3, 4, 5]]},
    {'mu': [1.0, 7.6], 'groups': [[0, 1, 2], [3, 4, 5]]},
    {'mu': [2.0, 11.0], 'groups': [[0, 1, 2], [3, 4, 5]]},
]

colors = ['steelblue', 'coral']
for i, (ax, state) in enumerate(zip(axes, history)):
    for k, grp in enumerate(state['groups']):
        ax.scatter(points[grp], np.zeros(len(grp)), c=colors[k], s=100, zorder=3)
    for k, mu in enumerate(state['mu']):
        ax.axvline(mu, color=colors[k], linestyle='--', linewidth=2, alpha=0.7)
        ax.plot(mu, 0.05, marker='v', color=colors[k], markersize=12, zorder=4)
    ax.set_xlim(-1, 14)
    ax.set_ylim(-0.1, 0.15)
    ax.set_yticks([])
    ax.set_xlabel('$x$')
    ax.set_title(f'Itération {i+1}: $\\mu_1={state["mu"][0]:.1f}$, $\\mu_2={state["mu"][1]:.1f}$')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
```

### Convergence et minima locaux

K-moyennes converge toujours. À chaque étape d'assignation, la distorsion diminue ou reste constante (chaque point est réassigné au centroïde le plus proche). À chaque étape de mise à jour, la distorsion diminue ou reste constante (la moyenne minimise la somme des distances au carré). Comme la distorsion est bornée inférieurement par zéro et que le nombre d'assignations possibles est fini, l'algorithme atteint un point fixe en un nombre fini d'itérations.

Cependant, ce point fixe est un **minimum local**, pas nécessairement le minimum global. Avec nos six points, une initialisation $\mu_1 = 2$, $\mu_2 = 3$ (les deux centroïdes dans le premier groupe) convergerait vers une solution différente de celle avec $\mu_1 = 1$, $\mu_2 = 12$.

En pratique, on lance k-moyennes plusieurs fois avec des initialisations aléatoires différentes et on garde la solution ayant la distorsion la plus faible.

### K-means++: une initialisation plus efficace

L'algorithme **k-means++** améliore l'initialisation en choisissant les centroïdes de manière à les espacer:

1. Choisir le premier centroïde uniformément au hasard parmi les données.
2. Pour chaque centroïde suivant, choisir un point avec une probabilité proportionnelle au carré de sa distance au centroïde le plus proche.
3. Répéter jusqu'à avoir $K$ centroïdes.

Cette stratégie tend à placer les centroïdes initiaux dans des régions différentes de l'espace. On peut montrer qu'elle garantit une distorsion au plus $O(\log K)$ fois la distorsion optimale, ce qui est un résultat remarquable pour un algorithme aussi simple.

```{code-cell} python
:tags: [hide-input]

np.random.seed(42)

# Données 2D avec 3 groupes
centers_true = [[-3, -3], [0, 4], [4, 0]]
X_demo = np.vstack([
    np.random.randn(60, 2) * 0.8 + c for c in centers_true
])

def run_kmeans(X, K, max_iter=20, seed=0):
    rng = np.random.RandomState(seed)
    mu = X[rng.choice(len(X), K, replace=False)]
    history = []
    for it in range(max_iter):
        dists = np.linalg.norm(X[:, None] - mu[None], axis=2)
        labels = np.argmin(dists, axis=1)
        history.append((mu.copy(), labels.copy()))
        new_mu = np.array([X[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else mu[k] for k in range(K)])
        if np.allclose(new_mu, mu):
            break
        mu = new_mu
    return history

history_demo = run_kmeans(X_demo, 3, seed=10)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
colors_map = ['steelblue', 'coral', 'forestgreen']

iters_to_show = [0, min(1, len(history_demo)-1), len(history_demo)-1]
for ax, it_idx in zip(axes, iters_to_show):
    mu, labels = history_demo[it_idx]
    for k in range(3):
        mask = labels == k
        ax.scatter(X_demo[mask, 0], X_demo[mask, 1], c=colors_map[k], alpha=0.5, s=20)
    for k in range(3):
        ax.plot(mu[k, 0], mu[k, 1], marker='X', color=colors_map[k],
                markersize=14, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(f'Itération {it_idx + 1}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

fig.suptitle('K-moyennes sur des données 2D (les croix marquent les centroïdes)', fontsize=12, y=1.02)
plt.tight_layout()
```

### Limites de k-moyennes

K-moyennes utilise la distance euclidienne pour mesurer la proximité. Cela revient à supposer que les groupes sont des **sphères** de même rayon dans l'espace des données. Quand les groupes ont des formes allongées, des tailles différentes ou des orientations variées, k-moyennes peut échouer à les retrouver.

```{code-cell} python
:tags: [hide-input]

np.random.seed(42)

# Données avec deux ellipses d'orientations différentes
cov1 = np.array([[4.0, 1.5], [1.5, 1.0]])
cov2 = np.array([[1.0, -0.8], [-0.8, 3.0]])
X_ell = np.vstack([
    np.random.multivariate_normal([0, 0], cov1, 100),
    np.random.multivariate_normal([4, 4], cov2, 100)
])
y_ell = np.array([0]*100 + [1]*100)

history_ell = run_kmeans(X_ell, 2, seed=0)
mu_final, labels_km = history_ell[-1]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(X_ell[y_ell == 0, 0], X_ell[y_ell == 0, 1], c='steelblue', alpha=0.5, s=20, label='Groupe 1')
ax.scatter(X_ell[y_ell == 1, 0], X_ell[y_ell == 1, 1], c='coral', alpha=0.5, s=20, label='Groupe 2')
ax.set_title('Groupes réels (ellipsoïdes)')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.legend()
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

ax = axes[1]
for k, c in enumerate(['steelblue', 'coral']):
    mask = labels_km == k
    ax.scatter(X_ell[mask, 0], X_ell[mask, 1], c=c, alpha=0.5, s=20)
    ax.plot(mu_final[k, 0], mu_final[k, 1], marker='X', color=c,
            markersize=14, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
ax.set_title('Résultat de k-moyennes (frontière linéaire)')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
```

La figure montre que k-moyennes coupe l'espace par un hyperplan perpendiculaire au segment reliant les deux centroïdes. Il ne peut pas s'adapter aux formes elliptiques des groupes. Pour dépasser cette limite, il faut un modèle plus riche: les modèles de mélange gaussien.

## Modèles de mélange gaussien

### Du partitionnement dur au partitionnement souple

K-moyennes assigne chaque point à exactement un groupe. Mais pour un point situé entre deux groupes, cette décision tranchée ignore l'ambiguïté. Un modèle probabiliste permet d'exprimer cette incertitude: au lieu d'une assignation binaire, chaque point se voit attribuer une **probabilité** d'appartenir à chaque groupe.

Un **modèle de mélange gaussien** (GMM, *Gaussian Mixture Model*) suppose que les données sont générées par un processus en deux étapes:

1. On tire un composant $k$ selon les **poids du mélange** $\pi_k$ (avec $\sum_k \pi_k = 1$ et $\pi_k \geq 0$).
2. On tire une observation $\boldsymbol{x}$ selon la gaussienne du composant choisi: $\boldsymbol{x} \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$.

La densité marginale de $\boldsymbol{x}$ est alors un mélange de gaussiennes:

$$
p(\boldsymbol{x} \mid \boldsymbol{\theta}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

Si vous avez étudié l'analyse discriminante gaussienne au [chapitre 6](ch6_probabilistic_models.md), ce modèle vous semblera familier. La différence est que les «classes» sont maintenant inconnues: ce sont des **variables latentes** $z_n \in \{1, \ldots, K\}$ qu'on ne peut pas observer directement. Le partitionnement consiste à inférer ces variables latentes à partir des données.

### Responsabilités

Étant donné un jeu de paramètres $\boldsymbol{\theta} = (\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)_{k=1}^K$, on peut calculer la probabilité a posteriori que l'observation $\boldsymbol{x}_n$ provienne du composant $k$. Cette quantité s'appelle la **responsabilité** du composant $k$ pour le point $n$:

$$
r_{nk} \triangleq p(z_n = k \mid \boldsymbol{x}_n, \boldsymbol{\theta}) = \frac{\pi_k \, \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{k'=1}^K \pi_{k'} \, \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_{k'}, \boldsymbol{\Sigma}_{k'})}
$$

Contrairement aux assignations de k-moyennes (qui valent 0 ou 1), les responsabilités sont des valeurs continues dans $[0, 1]$. Un point situé exactement entre deux composants aura des responsabilités proches de $0{,}5$ pour chacun, exprimant l'ambiguïté de son appartenance.

### L'algorithme EM

Pour estimer les paramètres $\boldsymbol{\theta}$, on voudrait maximiser la log-vraisemblance des données observées:

$$
\ell(\boldsymbol{\theta}) = \sum_{n=1}^N \log \left( \sum_{k=1}^K \pi_k \, \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right)
$$

La somme à l'intérieur du logarithme rend ce problème difficile: il n'existe pas de solution analytique. L'**algorithme EM** (*Expectation-Maximization*) contourne cette difficulté en alternant deux étapes, de manière analogue à k-moyennes.

**Étape E (espérance).** On fixe les paramètres $\boldsymbol{\theta}^{(t)}$ et on calcule les responsabilités:

$$
r_{nk}^{(t)} = \frac{\pi_k^{(t)} \, \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k^{(t)}, \boldsymbol{\Sigma}_k^{(t)})}{\sum_{k'=1}^K \pi_{k'}^{(t)} \, \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_{k'}^{(t)}, \boldsymbol{\Sigma}_{k'}^{(t)})}
$$

**Étape M (maximisation).** On fixe les responsabilités et on met à jour les paramètres. Notons $N_k = \sum_n r_{nk}^{(t)}$ le «nombre effectif» de points dans le composant $k$. Les mises à jour sont:

$$
\pi_k^{(t+1)} = \frac{N_k}{N}
$$

$$
\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{N_k} \sum_{n=1}^N r_{nk}^{(t)} \, \boldsymbol{x}_n
$$

$$
\boldsymbol{\Sigma}_k^{(t+1)} = \frac{1}{N_k} \sum_{n=1}^N r_{nk}^{(t)} \, (\boldsymbol{x}_n - \boldsymbol{\mu}_k^{(t+1)})(\boldsymbol{x}_n - \boldsymbol{\mu}_k^{(t+1)})^\top
$$

Ces formules ressemblent aux estimateurs du maximum de vraisemblance pour une seule gaussienne, mais avec des **moyennes pondérées** par les responsabilités au lieu de simples moyennes. Si les responsabilités étaient toutes 0 ou 1, on retrouverait exactement les formules de k-moyennes.

On peut montrer que chaque itération de EM augmente (ou maintient) la log-vraisemblance $\ell(\boldsymbol{\theta})$. Comme $\ell$ est bornée supérieurement, l'algorithme converge. Cependant, comme pour k-moyennes, la convergence se fait vers un maximum local, pas nécessairement global. Il est donc recommandé de lancer EM plusieurs fois et de garder la solution avec la meilleure log-vraisemblance.

Cette partie est plus technique que les précédentes. Si les détails des mises à jour semblent lourds au premier abord, retenez le principe: EM alterne entre «deviner à quel groupe appartient chaque point» (étape E) et «mettre à jour les paramètres de chaque groupe» (étape M).

```{code-cell} python
:tags: [hide-input]

np.random.seed(42)

# Données de mélange 2D
true_mus = [np.array([-2, 0]), np.array([2, 3]), np.array([3, -2])]
true_covs = [
    np.array([[1.0, 0.3], [0.3, 0.8]]),
    np.array([[0.6, -0.4], [-0.4, 1.2]]),
    np.array([[1.5, 0.0], [0.0, 0.5]])
]
true_pis = [0.3, 0.4, 0.3]
X_gmm = np.vstack([
    np.random.multivariate_normal(mu, cov, int(200 * pi))
    for mu, cov, pi in zip(true_mus, true_covs, true_pis)
])
np.random.shuffle(X_gmm)

def gmm_e_step(X, pis, mus, covs):
    N, K = len(X), len(pis)
    resp = np.zeros((N, K))
    for k in range(K):
        resp[:, k] = pis[k] * stats.multivariate_normal.pdf(X, mus[k], covs[k])
    resp /= resp.sum(axis=1, keepdims=True)
    return resp

def gmm_m_step(X, resp):
    N, D = X.shape
    K = resp.shape[1]
    Nk = resp.sum(axis=0)
    pis = Nk / N
    mus = [(resp[:, k:k+1] * X).sum(axis=0) / Nk[k] for k in range(K)]
    covs = []
    for k in range(K):
        diff = X - mus[k]
        covs.append((resp[:, k:k+1] * diff).T @ diff / Nk[k] + 1e-6 * np.eye(D))
    return pis, mus, covs

def run_gmm_em(X, K, n_iter=30, seed=0):
    rng = np.random.RandomState(seed)
    N, D = X.shape
    idx = rng.choice(N, K, replace=False)
    mus = [X[i].copy() for i in idx]
    covs = [np.eye(D) for _ in range(K)]
    pis = np.ones(K) / K
    history = []
    for it in range(n_iter):
        resp = gmm_e_step(X, pis, mus, covs)
        pis, mus, covs = gmm_m_step(X, resp)
        history.append((pis.copy(), [m.copy() for m in mus], [c.copy() for c in covs], resp.copy()))
    return history

hist_gmm = run_gmm_em(X_gmm, 3, n_iter=30, seed=5)

from matplotlib.patches import Ellipse

def draw_ellipse(ax, mu, cov, color, n_std=2):
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 1e-8))
    ell = Ellipse(mu, w, h, angle=angle, fill=False, color=color, linewidth=2, linestyle='--')
    ax.add_patch(ell)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
colors_gmm = ['steelblue', 'coral', 'forestgreen']
iters_gmm = [0, 4, len(hist_gmm)-1]

for ax, it in zip(axes, iters_gmm):
    pis_t, mus_t, covs_t, resp_t = hist_gmm[it]
    labels_t = np.argmax(resp_t, axis=1)

    for k in range(3):
        mask = labels_t == k
        ax.scatter(X_gmm[mask, 0], X_gmm[mask, 1], c=colors_gmm[k], alpha=0.4, s=15)
        draw_ellipse(ax, mus_t[k], covs_t[k], colors_gmm[k])
        ax.plot(mus_t[k][0], mus_t[k][1], marker='X', color=colors_gmm[k],
                markersize=12, markeredgecolor='black', markeredgewidth=1.5, zorder=5)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(f'Itération {it + 1}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6, 7)
    ax.set_ylim(-5, 7)

fig.suptitle('EM pour un mélange de 3 gaussiennes (les ellipses montrent les covariances)', fontsize=12, y=1.02)
plt.tight_layout()
```

La figure montre l'évolution de l'algorithme EM. Au départ, les composants sont mal placés et leurs covariances ne correspondent pas aux données. Au fil des itérations, les ellipses (qui représentent les courbes d'iso-densité) s'ajustent aux groupes naturels des données. Contrairement à k-moyennes, les groupes retrouvés ne sont pas des sphères mais des ellipsoïdes adaptés à la géométrie locale.

### Partitionnement dur et souple

La distinction entre k-moyennes et GMM illustre la différence entre **partitionnement dur** et **partitionnement souple**. Avec k-moyennes, chaque point appartient à un seul groupe. Avec un GMM, chaque point a une distribution de probabilité sur les groupes. Le partitionnement souple est plus informatif: il identifie les points ambigus et quantifie l'incertitude de l'assignation.

Si l'on souhaite un partitionnement dur à partir d'un GMM, on assigne chaque point au composant ayant la plus grande responsabilité: $\hat{z}_n = \arg\max_k r_{nk}$.

```{code-cell} python
:tags: [hide-input]

pis_f, mus_f, covs_f, resp_f = hist_gmm[-1]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Partitionnement dur
ax = axes[0]
labels_hard = np.argmax(resp_f, axis=1)
for k in range(3):
    mask = labels_hard == k
    ax.scatter(X_gmm[mask, 0], X_gmm[mask, 1], c=colors_gmm[k], alpha=0.5, s=20)
ax.set_title('Partitionnement dur ($\\hat{z}_n = \\arg\\max_k r_{nk}$)')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# Partitionnement souple
ax = axes[1]
rgb_colors = np.array([[0.27, 0.51, 0.71], [0.99, 0.50, 0.31], [0.13, 0.55, 0.13]])
point_colors = resp_f @ rgb_colors
ax.scatter(X_gmm[:, 0], X_gmm[:, 1], c=point_colors, alpha=0.6, s=20)
ax.set_title('Partitionnement souple (couleur = mélange des responsabilités)')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
```

## Types de covariance et complexité du modèle

La forme des groupes découverts par un GMM dépend directement de la structure imposée aux matrices de covariance $\boldsymbol{\Sigma}_k$. Plus la covariance est flexible, plus le modèle peut capturer des formes variées, mais plus le nombre de paramètres à estimer est élevé. Avec peu de données, un modèle trop flexible risque d'être instable.

Le cas le plus général est la **covariance complète**: chaque composant a sa propre matrice $\boldsymbol{\Sigma}_k$ sans contrainte (hormis la symétrie et la définie-positivité). Les groupes peuvent être des ellipsoïdes d'orientation et de taille quelconques. En dimension $D$, chaque matrice de covariance a $D(D+1)/2$ paramètres libres, soit $K \times D(D+1)/2$ au total pour les $K$ composants.

On peut réduire le nombre de paramètres en imposant que tous les composants partagent la même covariance, $\boldsymbol{\Sigma}_k = \boldsymbol{\Sigma}$ pour tout $k$. C'est la **covariance partagée** (*tied*). Les groupes ont alors la même forme et la même orientation, mais des centres différents. Cette hypothèse est l'analogue, pour le partitionnement, de celle que fait LDA en classification.

Une restriction plus forte est la **covariance diagonale**: $\boldsymbol{\Sigma}_k$ est diagonale pour chaque composant, ce qui suppose que les dimensions sont indépendantes au sein de chaque groupe. Les ellipsoïdes sont alors alignés avec les axes de coordonnées. Le nombre de paramètres tombe à $K \times D$.

Le cas extrême est la **covariance sphérique**: $\boldsymbol{\Sigma}_k = \sigma_k^2 \mathbf{I}$. Les groupes sont des sphères. Chaque composant n'a qu'un seul paramètre de dispersion. C'est avec cette hypothèse que le lien avec k-moyennes devient direct, comme nous allons le voir.

```{code-cell} python
:tags: [hide-input]

fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

mu_ex = np.array([0, 0])
covs_types = [
    ('Complète', np.array([[2.0, 1.2], [1.2, 1.5]])),
    ('Partagée (tied)', np.array([[1.5, 0.5], [0.5, 1.5]])),
    ('Diagonale', np.array([[2.5, 0.0], [0.0, 0.8]])),
    ('Sphérique', np.array([[1.2, 0.0], [0.0, 1.2]])),
]

for ax, (name, cov_ex) in zip(axes, covs_types):
    samples = np.random.multivariate_normal(mu_ex, cov_ex, 150)
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, c='steelblue')
    draw_ellipse(ax, mu_ex, cov_ex, 'steelblue', n_std=2)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title(name)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

plt.tight_layout()
```

Le choix du type de covariance dépend de la quantité de données disponibles et de la connaissance que l'on a de la géométrie des groupes. Avec beaucoup de données par rapport à la dimension, la covariance complète est souvent le meilleur choix. Avec peu de données ou en haute dimension, une covariance diagonale ou sphérique évite les problèmes numériques.

## K-moyennes comme cas limite de GMM

L'algorithme k-moyennes et les GMM ne sont pas deux méthodes indépendantes: k-moyennes est un cas particulier de GMM dans une limite précise.

Considérons un GMM avec des covariances sphériques identiques $\boldsymbol{\Sigma}_k = \sigma^2 \mathbf{I}$ et des poids uniformes $\pi_k = 1/K$. La responsabilité du composant $k$ pour le point $\boldsymbol{x}_n$ devient:

$$
r_{nk} = \frac{\exp(-\|\boldsymbol{x}_n - \boldsymbol{\mu}_k\|^2 / 2\sigma^2)}{\sum_{k'} \exp(-\|\boldsymbol{x}_n - \boldsymbol{\mu}_{k'}\|^2 / 2\sigma^2)}
$$

Quand $\sigma^2$ est grand, les responsabilités sont proches de $1/K$ pour tous les composants: chaque point appartient presque autant à tous les groupes (les gaussiennes sont très étalées et se ressemblent partout). Quand $\sigma^2$ est petit, l'exponentielle avec la plus petite distance domine toutes les autres, et les responsabilités deviennent:

$$
r_{nk} \to \begin{cases} 1 & \text{si } k = \arg\min_{k'} \|\boldsymbol{x}_n - \boldsymbol{\mu}_{k'}\|^2 \\ 0 & \text{sinon} \end{cases}
$$

Ce sont exactement les assignations de k-moyennes. Les mises à jour des moyennes deviennent alors les mises à jour classiques de k-moyennes. Autrement dit, k-moyennes est l'algorithme EM pour un GMM à covariance sphérique dans la limite où la variance tend vers zéro.

Ce lien explique pourquoi k-moyennes produit des groupes sphériques: c'est la seule forme compatible avec son hypothèse implicite de covariance. Pour des groupes de forme elliptique, il faut un GMM avec covariance complète ou diagonale.

## Considérations pratiques

### Choix du nombre de groupes $K$

Ni k-moyennes ni EM ne déterminent automatiquement le nombre de groupes. Trois approches sont courantes.

La **méthode du coude** consiste à tracer la distorsion (pour k-moyennes) ou la log-vraisemblance (pour GMM) en fonction de $K$, et à chercher un «coude» dans la courbe: un point au-delà duquel ajouter des groupes n'apporte plus d'amélioration notable. Cette approche est simple mais subjective.

Les **critères d'information** offrent un compromis plus formel entre adéquation aux données et complexité du modèle. Le BIC (*Bayesian Information Criterion*) et l'AIC (*Akaike Information Criterion*) pénalisent la log-vraisemblance par le nombre de paramètres:

$$
\text{BIC} = -2\ell(\hat{\boldsymbol{\theta}}) + p \log N, \qquad \text{AIC} = -2\ell(\hat{\boldsymbol{\theta}}) + 2p
$$

où $p$ est le nombre de paramètres du modèle. On choisit le $K$ qui minimise le critère. Le BIC pénalise plus fortement les modèles complexes et tend à choisir un $K$ plus petit que l'AIC.

Le **score de silhouette** mesure, pour chaque point, la cohésion au sein de son groupe et la séparation avec les autres groupes. Ce score ne nécessite pas de modèle probabiliste et peut être utilisé avec k-moyennes.

### Problèmes numériques

Avec un GMM à covariance complète, un composant qui ne «contient» que très peu de points effectifs ($N_k$ petit) peut avoir une matrice de covariance presque singulière. Cela produit des densités arbitrairement élevées et fait diverger la log-vraisemblance. Deux solutions sont courantes: ajouter un petit terme de régularisation $\epsilon \mathbf{I}$ à chaque matrice de covariance, ou utiliser un type de covariance plus contraint (diagonale ou partagée).

Il est aussi recommandé d'initialiser EM avec le résultat de k-moyennes, ce qui fournit un bon point de départ et accélère la convergence.

## Résumé

Ce chapitre a introduit le problème du partitionnement en partant d'un exemple simple en 1D. L'algorithme k-moyennes formalise l'intuition de regrouper les points proches en minimisant une distorsion; il alterne entre assignation des points et mise à jour des centroïdes, et converge toujours vers un minimum local.

Nous avons vu que k-moyennes fait l'hypothèse implicite de groupes sphériques de même taille. Les modèles de mélange gaussien dépassent cette limite en modélisant chaque groupe par une gaussienne avec sa propre covariance. L'algorithme EM estime les paramètres en alternant entre le calcul de responsabilités (des assignations souples) et la mise à jour des paramètres par des moyennes pondérées.

Le lien entre les deux approches est direct: k-moyennes est le cas limite de EM pour un GMM à covariance sphérique quand la variance tend vers zéro. Passer de l'un à l'autre revient à relâcher des hypothèses sur la forme des groupes.

Le choix du nombre de groupes $K$ reste un défi pratique pour les deux méthodes; les critères d'information (BIC, AIC) offrent un guide raisonnable, mais demandent de comparer plusieurs valeurs de $K$.

## Exercices

````{admonition} Exercice 1: Vérification des formules de k-moyennes ★
:class: hint dropdown

Considérez les quatre points en 2D: $\boldsymbol{x}_1 = (0, 0)$, $\boldsymbol{x}_2 = (1, 0)$, $\boldsymbol{x}_3 = (5, 5)$, $\boldsymbol{x}_4 = (6, 5)$, avec $K = 2$.

1. Initialisez les centroïdes à $\boldsymbol{\mu}_1 = (0, 0)$ et $\boldsymbol{\mu}_2 = (6, 5)$. Effectuez une itération complète de k-moyennes (assignation puis mise à jour).
2. L'algorithme a-t-il convergé? Justifiez.
3. Calculez la distorsion $\mathcal{L}$ après convergence.
````

```{admonition} Solution Exercice 1
:class: dropdown

1. **Assignation.** On calcule les distances:
   - $\boldsymbol{x}_1 = (0,0)$: $\|\boldsymbol{x}_1 - \boldsymbol{\mu}_1\|^2 = 0$, $\|\boldsymbol{x}_1 - \boldsymbol{\mu}_2\|^2 = 61$ → groupe 1
   - $\boldsymbol{x}_2 = (1,0)$: $\|\boldsymbol{x}_2 - \boldsymbol{\mu}_1\|^2 = 1$, $\|\boldsymbol{x}_2 - \boldsymbol{\mu}_2\|^2 = 50$ → groupe 1
   - $\boldsymbol{x}_3 = (5,5)$: $\|\boldsymbol{x}_3 - \boldsymbol{\mu}_1\|^2 = 50$, $\|\boldsymbol{x}_3 - \boldsymbol{\mu}_2\|^2 = 1$ → groupe 2
   - $\boldsymbol{x}_4 = (6,5)$: $\|\boldsymbol{x}_4 - \boldsymbol{\mu}_1\|^2 = 61$, $\|\boldsymbol{x}_4 - \boldsymbol{\mu}_2\|^2 = 0$ → groupe 2

   **Mise à jour:**
   $$\boldsymbol{\mu}_1 = \frac{(0,0) + (1,0)}{2} = (0{,}5,\; 0), \qquad \boldsymbol{\mu}_2 = \frac{(5,5) + (6,5)}{2} = (5{,}5,\; 5)$$

2. **Convergence.** On vérifie: avec les nouveaux centroïdes, les assignations restent les mêmes (les distances intergroupes sont grandes). L'algorithme a convergé en une seule itération.

3. **Distorsion:**
   $$\mathcal{L} = \|(0,0) - (0{,}5, 0)\|^2 + \|(1,0) - (0{,}5, 0)\|^2 + \|(5,5) - (5{,}5, 5)\|^2 + \|(6,5) - (5{,}5, 5)\|^2$$
   $$= 0{,}25 + 0{,}25 + 0{,}25 + 0{,}25 = 1$$
```

````{admonition} Exercice 2: K-moyennes en 1D ★
:class: hint dropdown

Reprenez les six points $\{1, 2, 3, 10, 11, 12\}$ avec $K = 2$.

1. Si les centroïdes initiaux sont $\mu_1 = 5$ et $\mu_2 = 6$, effectuez les itérations de k-moyennes jusqu'à convergence.
2. Comparez avec l'initialisation $\mu_1 = 1$, $\mu_2 = 12$. La solution finale est-elle la même?
3. Calculez la distorsion dans les deux cas. Quel résultat est meilleur?
````

```{admonition} Solution Exercice 2
:class: dropdown

1. **Initialisation $\mu_1 = 5$, $\mu_2 = 6$:**

   *Itération 1, assignation.* Le seuil est à $(5+6)/2 = 5{,}5$. Points $\leq 5{,}5$ → groupe 1: $\{1,2,3\}$. Points $> 5{,}5$ → groupe 2: $\{10,11,12\}$.

   *Itération 1, mise à jour.* $\mu_1 = (1+2+3)/3 = 2$, $\mu_2 = (10+11+12)/3 = 11$.

   *Itération 2.* Le seuil est à $(2+11)/2 = 6{,}5$. Les assignations ne changent pas. Convergence.

2. **Initialisation $\mu_1 = 1$, $\mu_2 = 12$:**

   *Itération 1, assignation.* Le seuil est à $6{,}5$. Même partition: $\{1,2,3\}$ et $\{10,11,12\}$.

   *Itération 1, mise à jour.* $\mu_1 = 2$, $\mu_2 = 11$. Convergence immédiate.

   Les deux initialisations donnent la même solution finale.

3. **Distorsion:** $(1-2)^2 + (2-2)^2 + (3-2)^2 + (10-11)^2 + (11-11)^2 + (12-11)^2 = 4$ dans les deux cas. C'est le minimum global pour $K = 2$ sur ces données.
```

````{admonition} Exercice 3: Responsabilités d'un GMM ★★
:class: hint dropdown

Un GMM à 2 composants en 1D a les paramètres $\pi_1 = 0{,}3$, $\mu_1 = 0$, $\sigma_1^2 = 1$ et $\pi_2 = 0{,}7$, $\mu_2 = 3$, $\sigma_2^2 = 1$.

1. Pour l'observation $x = 1$, calculez les densités $p(x \mid z = 1)$ et $p(x \mid z = 2)$, puis les responsabilités $r_1$ et $r_2$.
2. Faites le même calcul pour $x = 1{,}5$ (le point milieu entre les deux moyennes). Pourquoi les responsabilités ne sont-elles pas $0{,}5$ et $0{,}5$?
3. Trouvez la valeur $x^*$ où $r_1 = r_2$.

*Rappel:* la densité gaussienne 1D est $\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$.
````

```{admonition} Solution Exercice 3
:class: dropdown

1. **Pour $x = 1$:**

   $$p(x=1 \mid z=1) = \frac{1}{\sqrt{2\pi}} e^{-1/2} \approx 0{,}242$$
   $$p(x=1 \mid z=2) = \frac{1}{\sqrt{2\pi}} e^{-4/2} = \frac{1}{\sqrt{2\pi}} e^{-2} \approx 0{,}054$$

   $$r_1 = \frac{0{,}3 \times 0{,}242}{0{,}3 \times 0{,}242 + 0{,}7 \times 0{,}054} = \frac{0{,}073}{0{,}073 + 0{,}038} \approx 0{,}66$$
   $$r_2 \approx 0{,}34$$

   Le point $x = 1$ est plus probablement issu du composant 1.

2. **Pour $x = 1{,}5$:**

   Les deux densités sont égales (par symétrie: $|1{,}5 - 0| = |1{,}5 - 3| = 1{,}5$):
   $$p(x = 1{,}5 \mid z = 1) = p(x = 1{,}5 \mid z = 2) \approx 0{,}130$$

   Mais les responsabilités ne sont pas égales car les poids $\pi_k$ diffèrent:
   $$r_1 = \frac{0{,}3 \times 0{,}130}{0{,}3 \times 0{,}130 + 0{,}7 \times 0{,}130} = \frac{0{,}3}{1{,}0} = 0{,}3, \qquad r_2 = 0{,}7$$

   Au point milieu, les densités se compensent et les responsabilités reflètent directement les poids du mélange.

3. **Point $x^*$ où $r_1 = r_2$:**

   On résout $\pi_1 \mathcal{N}(x^* \mid 0, 1) = \pi_2 \mathcal{N}(x^* \mid 3, 1)$:
   $$0{,}3 \exp\left(-\frac{(x^*)^2}{2}\right) = 0{,}7 \exp\left(-\frac{(x^*-3)^2}{2}\right)$$

   En prenant le logarithme et en simplifiant:
   $$\log(0{,}3) - \frac{(x^*)^2}{2} = \log(0{,}7) - \frac{(x^*-3)^2}{2}$$
   $$\log\frac{0{,}3}{0{,}7} = 3x^* - 4{,}5$$
   $$x^* = \frac{4{,}5 + \log(0{,}3/0{,}7)}{3} \approx \frac{4{,}5 - 0{,}847}{3} \approx 1{,}22$$

   Le point d'égale responsabilité est décalé vers le composant de plus faible poids ($\mu_1 = 0$), car le composant 2 a un poids plus élevé et «attire» davantage les points vers lui.
```

````{admonition} Exercice 4: Nombre de paramètres d'un GMM ★★
:class: hint dropdown

Pour un GMM à $K$ composants en dimension $D$:

1. Comptez le nombre total de paramètres libres avec des covariances complètes. N'oubliez pas les poids $\pi_k$ (attention: combien sont libres?), les moyennes, et les covariances.
2. Faites le même calcul avec des covariances diagonales, puis sphériques.
3. Pour $D = 100$ et $K = 10$, calculez ces trois nombres. Que constatez-vous?
````

```{admonition} Solution Exercice 4
:class: dropdown

1. **Covariance complète:**
   - Poids: $K - 1$ paramètres libres (ils somment à 1)
   - Moyennes: $K \times D$
   - Covariances: chaque matrice $D \times D$ symétrique a $D(D+1)/2$ paramètres libres, donc $K \times D(D+1)/2$

   Total: $(K-1) + KD + K \cdot D(D+1)/2$

2. **Covariance diagonale:** chaque $\boldsymbol{\Sigma}_k$ a $D$ paramètres. Total: $(K-1) + KD + KD = (K-1) + 2KD$.

   **Covariance sphérique:** chaque $\boldsymbol{\Sigma}_k = \sigma_k^2 \mathbf{I}$ a 1 paramètre. Total: $(K-1) + KD + K$.

3. **Pour $D = 100$, $K = 10$:**
   - Complète: $9 + 1000 + 10 \times 5050 = 51\,009$
   - Diagonale: $9 + 2000 = 2\,009$
   - Sphérique: $9 + 1000 + 10 = 1\,019$

   Le nombre de paramètres avec covariance complète est 25 fois plus élevé qu'avec covariance diagonale. Avec peu de données en haute dimension, la covariance complète sera très difficile à estimer de manière fiable.
```

````{admonition} Exercice 5: Étape M de l'algorithme EM ★★
:class: hint dropdown

Soit un GMM à 2 composants en 1D avec les données $\{1, 2, 4, 5\}$ et les responsabilités suivantes après l'étape E:

| $x_n$ | $r_{n1}$ | $r_{n2}$ |
|--------|----------|----------|
| 1      | 0,9      | 0,1      |
| 2      | 0,8      | 0,2      |
| 4      | 0,2      | 0,8      |
| 5      | 0,1      | 0,9      |

1. Calculez $N_1$ et $N_2$.
2. Calculez les nouvelles moyennes $\mu_1^{(t+1)}$ et $\mu_2^{(t+1)}$.
3. Calculez les nouvelles variances $(\sigma_1^2)^{(t+1)}$ et $(\sigma_2^2)^{(t+1)}$.
4. Calculez les nouveaux poids $\pi_1^{(t+1)}$ et $\pi_2^{(t+1)}$.
````

```{admonition} Solution Exercice 5
:class: dropdown

1. **Nombres effectifs:**
   $$N_1 = 0{,}9 + 0{,}8 + 0{,}2 + 0{,}1 = 2{,}0, \qquad N_2 = 0{,}1 + 0{,}2 + 0{,}8 + 0{,}9 = 2{,}0$$

2. **Moyennes:**
   $$\mu_1^{(t+1)} = \frac{0{,}9 \times 1 + 0{,}8 \times 2 + 0{,}2 \times 4 + 0{,}1 \times 5}{2{,}0} = \frac{0{,}9 + 1{,}6 + 0{,}8 + 0{,}5}{2{,}0} = \frac{3{,}8}{2{,}0} = 1{,}9$$
   $$\mu_2^{(t+1)} = \frac{0{,}1 \times 1 + 0{,}2 \times 2 + 0{,}8 \times 4 + 0{,}9 \times 5}{2{,}0} = \frac{0{,}1 + 0{,}4 + 3{,}2 + 4{,}5}{2{,}0} = \frac{8{,}2}{2{,}0} = 4{,}1$$

3. **Variances:**
   $$(\sigma_1^2)^{(t+1)} = \frac{0{,}9(1-1{,}9)^2 + 0{,}8(2-1{,}9)^2 + 0{,}2(4-1{,}9)^2 + 0{,}1(5-1{,}9)^2}{2{,}0}$$
   $$= \frac{0{,}729 + 0{,}008 + 0{,}882 + 0{,}961}{2{,}0} = \frac{2{,}58}{2{,}0} = 1{,}29$$

   $$(\sigma_2^2)^{(t+1)} = \frac{0{,}1(1-4{,}1)^2 + 0{,}2(2-4{,}1)^2 + 0{,}8(4-4{,}1)^2 + 0{,}9(5-4{,}1)^2}{2{,}0}$$
   $$= \frac{0{,}961 + 0{,}882 + 0{,}008 + 0{,}729}{2{,}0} = \frac{2{,}58}{2{,}0} = 1{,}29$$

   Les variances sont symétriques car les responsabilités sont symétriques.

4. **Poids:**
   $$\pi_1^{(t+1)} = \frac{N_1}{N} = \frac{2{,}0}{4} = 0{,}5, \qquad \pi_2^{(t+1)} = \frac{N_2}{N} = \frac{2{,}0}{4} = 0{,}5$$
```

````{admonition} Exercice 6: K-moyennes et covariance sphérique ★★★
:class: hint dropdown

*Cet exercice est optionnel pour IFT3395.*

1. Montrez que, pour un GMM avec $\boldsymbol{\Sigma}_k = \sigma^2 \mathbf{I}$ et $\pi_k = 1/K$, la responsabilité $r_{nk}$ peut s'écrire sous la forme d'un softmax:
   $$r_{nk} = \frac{\exp(-\|\boldsymbol{x}_n - \boldsymbol{\mu}_k\|^2 / 2\sigma^2)}{\sum_{k'} \exp(-\|\boldsymbol{x}_n - \boldsymbol{\mu}_{k'}\|^2 / 2\sigma^2)}$$

2. Montrez que dans la limite $\sigma^2 \to 0$, les responsabilités deviennent les assignations de k-moyennes.

3. L'algorithme EM pour un GMM converge vers un maximum local de la log-vraisemblance. Quelle quantité k-moyennes converge-t-il vers un minimum local?

4. Pourquoi pourrait-on préférer EM (avec $\sigma^2$ petit mais non nul) à k-moyennes, même si l'on ne s'intéresse qu'au partitionnement dur?
````

```{admonition} Solution Exercice 6
:class: dropdown

1. **Écriture softmax.** Avec $\boldsymbol{\Sigma}_k = \sigma^2 \mathbf{I}$ et $\pi_k = 1/K$:
   $$\mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \sigma^2 \mathbf{I}) = \frac{1}{(2\pi\sigma^2)^{D/2}} \exp\left(-\frac{\|\boldsymbol{x}_n - \boldsymbol{\mu}_k\|^2}{2\sigma^2}\right)$$

   Le facteur $(2\pi\sigma^2)^{-D/2}$ et les poids $1/K$ sont identiques pour tous les composants et s'annulent dans la fraction:
   $$r_{nk} = \frac{\frac{1}{K} \cdot (2\pi\sigma^2)^{-D/2} \exp(-\|\boldsymbol{x}_n - \boldsymbol{\mu}_k\|^2 / 2\sigma^2)}{\sum_{k'} \frac{1}{K} \cdot (2\pi\sigma^2)^{-D/2} \exp(-\|\boldsymbol{x}_n - \boldsymbol{\mu}_{k'}\|^2 / 2\sigma^2)} = \frac{\exp(-\|\boldsymbol{x}_n - \boldsymbol{\mu}_k\|^2 / 2\sigma^2)}{\sum_{k'} \exp(-\|\boldsymbol{x}_n - \boldsymbol{\mu}_{k'}\|^2 / 2\sigma^2)}$$

2. **Limite $\sigma^2 \to 0$.** Le terme $-\|\boldsymbol{x}_n - \boldsymbol{\mu}_k\|^2 / 2\sigma^2$ tend vers $-\infty$ pour tous les $k$, sauf pour celui qui minimise $\|\boldsymbol{x}_n - \boldsymbol{\mu}_k\|^2$. Pour ce $k^*$, le rapport des exponentielles tend vers 1; pour les autres, vers 0. On retrouve $r_{nk} \to \mathbb{1}(k = \arg\min_{k'} \|\boldsymbol{x}_n - \boldsymbol{\mu}_{k'}\|^2)$.

3. **Quantité optimisée.** K-moyennes minimise la distorsion $\mathcal{L} = \sum_n \sum_k r_{nk} \|\boldsymbol{x}_n - \boldsymbol{\mu}_k\|^2$ avec des assignations dures.

4. **Avantage de EM sur k-moyennes.** Avec $\sigma^2$ petit mais non nul, les responsabilités sont presque binaires mais pas exactement. Cela rend la surface d'optimisation plus lisse et permet d'éviter certains minima locaux (un point peut «glisser» progressivement d'un groupe à l'autre au lieu de basculer brutalement). EM avec $\sigma^2$ fini peut aussi identifier les points frontières dont l'assignation est ambiguë.
```

````{admonition} Exercice 7: Implémenter k-moyennes ★★
:class: hint dropdown

Implémentez l'algorithme k-moyennes en Python:

```python
def kmeans(X, K, max_iter=100, seed=42):
    """
    Args:
        X: données (N, D)
        K: nombre de groupes
    Returns:
        labels: assignations (N,), valeurs dans {0, ..., K-1}
        centroids: centroïdes (K, D)
    """
    rng = np.random.RandomState(seed)
    # Initialisez K centroïdes au hasard parmi les données
    # Alternez assignation et mise à jour jusqu'à convergence
    pass
```

Testez sur les données suivantes et tracez le résultat:

```python
np.random.seed(0)
X_test = np.vstack([
    np.random.randn(50, 2) + [0, 0],
    np.random.randn(50, 2) + [5, 5],
    np.random.randn(50, 2) + [5, 0]
])
```
````

````{admonition} Solution Exercice 7
:class: dropdown

```python
def kmeans(X, K, max_iter=100, seed=42):
    rng = np.random.RandomState(seed)
    N, D = X.shape

    idx = rng.choice(N, K, replace=False)
    centroids = X[idx].copy()

    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)  # (N, K)
        labels = np.argmin(dists, axis=1)

        new_centroids = np.array([
            X[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centroids[k]
            for k in range(K)
        ])

        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return labels, centroids

np.random.seed(0)
X_test = np.vstack([
    np.random.randn(50, 2) + [0, 0],
    np.random.randn(50, 2) + [5, 5],
    np.random.randn(50, 2) + [5, 0]
])

labels, centroids = kmeans(X_test, 3)

plt.figure(figsize=(6, 5))
for k in range(3):
    mask = labels == k
    plt.scatter(X_test[mask, 0], X_test[mask, 1], alpha=0.6, s=30)
    plt.plot(centroids[k, 0], centroids[k, 1], 'kX', markersize=14)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Résultat de k-moyennes')
plt.grid(True, alpha=0.3)
plt.tight_layout()
```
````
