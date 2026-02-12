---
marp: true
theme: mila
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Méthodes non paramétriques
## Des k-ppv à la régression locale (LOESS)

*Pierre-Luc Bacon*
IFT6390 – Fondements de l'apprentissage machine

---

## Plan de la présentation

1. **Paramétrique vs non paramétrique** : deux philosophies
2. **k plus proches voisins** : classification et régression par voisinage
3. **Lissage par noyaux** : Nadaraya-Watson et pondération douce
4. **LOESS** : régression locale — combiner voisinage et régression linéaire
5. **Comparaison** : k-ppv, Nadaraya-Watson, LOESS

---

<!-- _class: lead -->

# Paramétrique vs non paramétrique
## Deux façons de résumer les données

---

<!-- footer: "📖 Méthodes non paramétriques" -->

## Deux familles d'approches

| | Paramétrique | Non paramétrique |
|--|--------------|------------------|
| **Modèle** | $f(\mathbf{x}; \boldsymbol{\theta})$ | Les données elles-mêmes |
| **Entraînement** | Estimer $\boldsymbol{\theta} \in \mathbb{R}^p$ | Stocker $\mathcal{D}$ |
| **Complexité** | Fixe ($p$ paramètres) | Croît avec $N$ |
| **Prédiction** | Évaluer $f(\mathbf{x}; \hat{\boldsymbol{\theta}})$ | Consulter les voisins |
| **Exemple** | Régression linéaire | k-ppv |

Les méthodes paramétriques font une hypothèse sur la forme de $f$. Les méthodes non paramétriques laissent les données « parler d'elles-mêmes ».

---

## Quand utiliser quoi?

| Situation | Paramétrique | Non paramétrique |
|-----------|-------------|------------------|
| Beaucoup de données, basse dimension | Bon | Bon |
| Peu de données | Bon (si le modèle est correct) | Risqué |
| Haute dimension ($d > 20$) | Nécessaire | Fléau de la dimensionnalité |
| Relation inconnue, basse dimension | Risqué | Bon |
| Besoin d'interprétabilité | Coefficients $\boldsymbol{\theta}$ | Moins direct |

En basse dimension avec suffisamment de données, les méthodes non paramétriques sont séduisantes : aucune hypothèse sur la forme de $f$.

---

<!-- _class: lead -->

# Les k plus proches voisins
## Prédire par voisinage

---

## Classification par k-ppv

**Idée** : Pour classifier $\mathbf{x}$, trouver les $k$ points d'entraînement les plus proches et voter.

$$\hat{y} = \arg\max_c \sum_{i \in \mathcal{N}_k(\mathbf{x})} \mathbb{1}_{y_i = c}$$

| $k$ | Frontière | Biais | Variance |
|-----|-----------|-------|----------|
| 1 | Très irrégulière | Faible | Élevée |
| $k$ moyen | Lisse | Moyen | Moyenne |
| $N$ | Classe majoritaire | Élevé | Nulle |

Le paramètre $k$ contrôle le compromis biais-variance, comme le degré du polynôme en régression ou $\lambda$ en Ridge.

---

## Régression par k-ppv

Pour la régression ($y \in \mathbb{R}$), le vote majoritaire devient une **moyenne** :

$$\hat{y}(\mathbf{x}) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(\mathbf{x})} y_i$$

Cette moyenne locale estime $\mathbb{E}[Y \mid \mathbf{X} = \mathbf{x}]$ en utilisant les $k$ exemples les plus proches.

```python
def knn_regression(X_train, y_train, x_query, k):
    distances = np.sqrt(np.sum((X_train - x_query)**2, axis=1))
    k_nearest = np.argsort(distances)[:k]
    return np.mean(y_train[k_nearest])
```

---

## Visualisation : régression k-ppv en 1D

```python
np.random.seed(42)
X_train = np.sort(np.random.uniform(0, 10, 50))
y_train = np.sin(X_train) + np.random.randn(50) * 0.3
X_test = np.linspace(0, 10, 200)

for k in [1, 5, 15]:
    y_pred = [knn_regression(X_train, y_train, x, k) for x in X_test]
    plt.plot(X_test, y_pred, label=f'k = {k}')
```

| $k$ | Comportement |
|-----|-------------|
| $k = 1$ | Fonction en escalier, saute d'un point à l'autre |
| $k = 5$ | Lissage modéré, capture la tendance |
| $k = 15$ | Trop lisse, écrase les variations locales |

---

## Deux problèmes des k-ppv en régression

**1. Prédictions discontinues** : la fonction $\hat{y}(\mathbf{x})$ change brusquement quand un point entre ou sort du voisinage.

**2. Pondération uniforme** : tous les $k$ voisins contribuent également, qu'ils soient très proches ou à la limite du voisinage.

| Problème | Cause | Solution |
|----------|-------|----------|
| Discontinuités | Frontière dure du voisinage | Pondération douce (noyaux) |
| Poids uniformes | Traitement égal des voisins | Poids décroissant avec la distance |

Ces deux problèmes motivent le passage au **lissage par noyaux**.

---

<!-- _class: lead -->

# Lissage par noyaux
## De la frontière dure à la pondération douce

---

## Noyau gaussien : pondérer par proximité

Un **noyau** $K_\lambda$ assigne un poids à chaque point selon sa distance à la requête.

Le noyau gaussien est le choix le plus courant :

$$K_\lambda(u) = \frac{1}{\lambda \sqrt{2\pi}} \exp\left(-\frac{u^2}{2\lambda^2}\right)$$

| $\lambda$ petit | $\lambda$ grand |
|-----------------|-----------------|
| Noyau concentré, influence locale | Noyau étalé, influence globale |
| Haute variance, faible biais | Faible variance, haut biais |

Le paramètre $\lambda$ (**largeur de bande**) joue le rôle de $k$ dans les k-ppv : il contrôle le compromis biais-variance.

---

## Estimateur de Nadaraya-Watson

Au lieu de sélectionner $k$ voisins, nous pondérons **tous** les points par proximité :

$$\boxed{\hat{y}(\mathbf{x}) = \frac{\sum_{i=1}^N K_\lambda(\mathbf{x} - \mathbf{x}_i) \, y_i}{\sum_{i=1}^N K_\lambda(\mathbf{x} - \mathbf{x}_i)} = \sum_{i=1}^N w_i(\mathbf{x}) \, y_i}$$

avec les poids normalisés :

$$w_i(\mathbf{x}) = \frac{K_\lambda(\mathbf{x} - \mathbf{x}_i)}{\sum_{j=1}^N K_\lambda(\mathbf{x} - \mathbf{x}_j)}, \quad \sum_i w_i(\mathbf{x}) = 1$$

Chaque point contribue, mais les points éloignés ont un poids négligeable.

---

## Code : Nadaraya-Watson

```python
def gaussian_kernel(u, bandwidth):
    return np.exp(-u**2 / (2 * bandwidth**2))

def nadaraya_watson(X_train, y_train, x_query, bandwidth):
    weights = gaussian_kernel(X_train - x_query, bandwidth)
    return np.sum(weights * y_train) / np.sum(weights)
```

| k-ppv | Nadaraya-Watson |
|-------|-----------------|
| $k$ voisins avec poids $1/k$ | Tous les points, poids $w_i(\mathbf{x})$ |
| Transition brusque | Transition graduelle |
| Prédiction en escalier | Prédiction lisse |

---

## Comparaison visuelle : k-ppv vs Nadaraya-Watson

```python
X_test = np.linspace(0, 10, 200)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# k-ppv avec k=5
y_knn = [knn_regression(X_train, y_train, x, k=5) for x in X_test]
axes[0].plot(X_test, y_knn, 'C1-', linewidth=2)
axes[0].set_title('k-ppv (k = 5)')

# Nadaraya-Watson avec lambda=0.5
y_nw = [nadaraya_watson(X_train, y_train, x, 0.5) for x in X_test]
axes[1].plot(X_test, y_nw, 'C1-', linewidth=2)
axes[1].set_title(r'Nadaraya-Watson ($\lambda = 0.5$)')
```

Nadaraya-Watson produit des prédictions plus lisses car la transition entre voisins est graduelle plutôt qu'abrupte.

---

## Le problème du biais aux bords

Nadaraya-Watson est une **moyenne locale pondérée** : il ajuste localement une constante. Cela pose un problème aux **bords** des données et dans les régions de **forte pente**.

| Situation | Problème |
|-----------|----------|
| Bord des données | Le noyau déborde d'un seul côté → biais |
| Forte pente locale | La moyenne locale « retarde » → biais |
| Plateau | Pas de problème, la constante suffit |

La moyenne pondérée ne peut pas capturer une tendance locale. Si la vraie fonction monte avec une pente positive, la moyenne locale tire la prédiction **en dessous** de la vraie valeur.

---

<!-- _class: lead -->

# LOESS
## Régression locale : combiner noyaux et régression linéaire

---

## L'idée de LOESS

LOESS (*LOcally Estimated Scatterplot Smoothing*) remplace la **constante locale** de Nadaraya-Watson par une **régression linéaire locale**.

| Méthode | Ajustement local | Biais aux bords |
|---------|-----------------|-----------------|
| k-ppv | Constante (moyenne) | Élevé |
| Nadaraya-Watson | Constante (moyenne pondérée) | Élevé |
| **LOESS** | **Droite (régression pondérée)** | **Réduit** |

Pour prédire en un point $\mathbf{x}_0$, LOESS ajuste une droite (ou un polynôme de faible degré) aux données voisines, en pondérant chaque point selon sa distance à $\mathbf{x}_0$.

---

## Formulation mathématique

Pour prédire en $\mathbf{x}_0$, résoudre le problème de **moindres carrés pondérés** :

$$\boxed{\hat{\boldsymbol{\beta}}(\mathbf{x}_0) = \arg\min_{\boldsymbol{\beta}} \sum_{i=1}^N K_\lambda(\mathbf{x}_0 - \mathbf{x}_i) \left(y_i - \beta_0 - \boldsymbol{\beta}_1^\top \mathbf{x}_i\right)^2}$$

La prédiction est :

$$\hat{y}(\mathbf{x}_0) = \hat{\beta}_0(\mathbf{x}_0) + \hat{\boldsymbol{\beta}}_1(\mathbf{x}_0)^\top \mathbf{x}_0$$

Les coefficients $\hat{\boldsymbol{\beta}}$ changent en fonction du point de prédiction $\mathbf{x}_0$.

---

## Solution en forme fermée

En posant $\mathbf{W}(\mathbf{x}_0) = \text{diag}(K_\lambda(\mathbf{x}_0 - \mathbf{x}_1), \ldots, K_\lambda(\mathbf{x}_0 - \mathbf{x}_N))$, la solution des moindres carrés pondérés est :

$$\hat{\boldsymbol{\beta}}(\mathbf{x}_0) = (\tilde{\mathbf{X}}^\top \mathbf{W}(\mathbf{x}_0) \tilde{\mathbf{X}})^{-1} \tilde{\mathbf{X}}^\top \mathbf{W}(\mathbf{x}_0) \mathbf{y}$$

où $\tilde{\mathbf{X}}$ est la matrice de conception augmentée d'une colonne de 1.

| Composante | Rôle |
|------------|------|
| $\mathbf{W}(\mathbf{x}_0)$ | Poids du noyau : points proches pèsent plus |
| $\tilde{\mathbf{X}}$ | Matrice de conception avec intercept |
| $\mathbf{y}$ | Cibles |

C'est une régression linéaire classique (MCO) avec une matrice de poids.

---

## Algorithme LOESS

Pour chaque point de prédiction $\mathbf{x}_0$ :

1. **Calculer les poids** : $w_i = K_\lambda(\mathbf{x}_0 - \mathbf{x}_i)$ pour $i = 1, \ldots, N$
2. **Résoudre les MCO pondérés** : ajuster $y \approx \beta_0 + \boldsymbol{\beta}_1^\top \mathbf{x}$ avec les poids $w_i$
3. **Prédire** : $\hat{y}(\mathbf{x}_0) = \hat{\beta}_0 + \hat{\boldsymbol{\beta}}_1^\top \mathbf{x}_0$

**Coût** : une régression linéaire par point de prédiction. Plus cher que Nadaraya-Watson, mais réduit le biais.

---

## Code : LOESS en 1D

```python
def loess_predict(X_train, y_train, x_query, bandwidth):
    """LOESS: régression linéaire locale pondérée."""
    # Poids du noyau
    weights = gaussian_kernel(X_train - x_query, bandwidth)

    # Matrice de conception locale [1, x]
    X_design = np.column_stack([np.ones_like(X_train), X_train])

    # Moindres carrés pondérés: (X'WX)^{-1} X'Wy
    W = np.diag(weights)
    XtWX = X_design.T @ W @ X_design
    XtWy = X_design.T @ W @ y_train
    beta = np.linalg.solve(XtWX, XtWy)

    # Prédiction au point de requête
    return beta[0] + beta[1] * x_query
```

---

## Visualisation : LOESS vs Nadaraya-Watson

```python
X_test = np.linspace(0, 10, 200)

y_nw = [nadaraya_watson(X_train, y_train, x, 0.8) for x in X_test]
y_loess = [loess_predict(X_train, y_train, x, 0.8) for x in X_test]

plt.plot(X_test, y_nw, 'C0-', label='Nadaraya-Watson')
plt.plot(X_test, y_loess, 'C1-', label='LOESS')
plt.plot(X_test, np.sin(X_test), 'k--', alpha=0.5, label=r'$\sin(x)$')
plt.scatter(X_train, y_train, c='gray', s=20, alpha=0.5)
plt.legend()
```

Aux bords de l'intervalle, LOESS suit mieux la tendance que Nadaraya-Watson, qui est tiré vers la moyenne.

---

## Pourquoi LOESS réduit le biais aux bords

Aux bords, le noyau est **asymétrique** : les données ne sont disponibles que d'un côté.

| Méthode | Bord gauche | Explication |
|---------|-------------|-------------|
| Nadaraya-Watson | Biais vers le haut/bas | La moyenne pondérée ne suit pas la pente |
| LOESS | Biais réduit | La droite locale s'adapte à la pente |

Nadaraya-Watson estime une constante locale $\hat{y} = c$, qui ne peut pas capturer $f'(\mathbf{x}_0) \neq 0$. LOESS estime une droite locale $\hat{y} = a + bx$, qui s'adapte à la pente et corrige naturellement le biais aux bords.

---

## Le paramètre $\lambda$ : largeur de bande

| $\lambda$ | Effet local | Biais | Variance | Résultat |
|-----------|------------|-------|----------|----------|
| Petit | Peu de points influents | Faible | Élevée | Courbe irrégulière |
| Grand | Beaucoup de points | Élevé | Faible | Courbe trop lisse |

Le choix de $\lambda$ se fait par **validation croisée**, comme pour $k$ en k-ppv ou $\lambda$ en Ridge.

En pratique, on utilise souvent une fraction des données (ex: les 30% les plus proches) plutôt qu'un $\lambda$ fixe. Cette variante est appelée LOESS avec voisinage adaptatif.

---

## Noyau tri-cubique (variante classique de LOESS)

La formulation classique de LOESS utilise le noyau **tri-cubique** à support compact :

$$K(u) = \begin{cases} (1 - |u|^3)^3 & \text{si } |u| < 1 \\ 0 & \text{sinon} \end{cases}$$

| Noyau | Support | Propriété |
|-------|---------|-----------|
| Gaussien | Infini | Tous les points contribuent |
| Tri-cubique | $[-1, 1]$ | Seuls les voisins proches contribuent |

Le support compact rend le calcul plus efficace : les points lointains ont un poids exactement nul.

---

<!-- _class: lead -->

# Comparaison des trois méthodes
## k-ppv, Nadaraya-Watson, LOESS

---

## Synthèse : trois niveaux de sophistication

| | k-ppv | Nadaraya-Watson | LOESS |
|--|-------|-----------------|-------|
| **Ajustement local** | Constante | Constante | Linéaire |
| **Pondération** | Uniforme ($1/k$) | Noyau (douce) | Noyau (douce) |
| **Continuité** | Non | Oui | Oui |
| **Biais aux bords** | Élevé | Élevé | Réduit |
| **Complexité/requête** | $O(N \log N)$ | $O(N)$ | $O(Nd^2)$ |
| **Hyperparamètre** | $k$ | $\lambda$ | $\lambda$ |

Chaque méthode ajoute un ingrédient :
- k-ppv → + pondération douce = **Nadaraya-Watson**
- Nadaraya-Watson → + régression locale = **LOESS**

---

## Lien avec le mécanisme d'attention

La formule de Nadaraya-Watson a une structure familière :

$$\hat{y}(\mathbf{x}) = \sum_i \frac{K(\mathbf{x}, \mathbf{x}_i)}{\sum_j K(\mathbf{x}, \mathbf{x}_j)} y_i$$

C'est une **moyenne pondérée normalisée**, exactement comme le mécanisme d'attention dans les Transformers :

$$\text{Attention}(\mathbf{q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}$$

| Nadaraya-Watson | Attention |
|-----------------|-----------|
| Requête $\mathbf{x}$ | Vecteur requête $\mathbf{q}$ |
| Points d'entraînement $\mathbf{x}_i$ | Clés $\mathbf{k}_i$ |
| Cibles $y_i$ | Valeurs $\mathbf{v}_i$ |
| Noyau $K_\lambda$ | Produit scalaire + softmax |

---

## Le fléau de la dimensionnalité

Toutes les méthodes locales souffrent en **haute dimension** :

$$r = p^{1/d}$$

Pour capturer une fraction $p$ des données dans un hypercube de côté $r$ :

| Dimension $d$ | Côté $r$ pour $p = 10\%$ |
|--------------|--------------------------|
| 1 | 0,10 |
| 2 | 0,32 |
| 10 | 0,79 |
| 100 | 0,98 |

En dimension 100, il faut couvrir 98% de chaque axe pour capturer 10% des points. Le voisinage n'est plus « local ».

---

## En pratique avec scikit-learn

scikit-learn fournit les k-ppv mais pas LOESS directement. La bibliothèque `statsmodels` offre une implémentation complète :

```python
# k-ppv régression
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# LOESS via statsmodels
import statsmodels.api as sm
lowess = sm.nonparametric.lowess
y_pred_loess = lowess(y_train, X_train, frac=0.3,
                      xvals=X_test)
```

Le paramètre `frac` de LOESS contrôle la fraction des données utilisées pour chaque ajustement local (analogue à $\lambda$).

---

<!-- footer: "" -->

## Résumé

| Concept | Idée clé |
|---------|----------|
| **k-ppv** | Moyenne (vote) sur les $k$ voisins les plus proches |
| **Noyau gaussien** | Pondération douce décroissant avec la distance |
| **Nadaraya-Watson** | Moyenne pondérée par noyau (constante locale) |
| **LOESS** | Régression linéaire pondérée par noyau (droite locale) |
| **Biais aux bords** | LOESS réduit le biais par l'ajustement linéaire local |
| **Fléau de la dimensionnalité** | Les méthodes locales souffrent en haute dimension |

Les méthodes non paramétriques sont adaptées quand la forme de $f$ est inconnue et la dimension est modérée. En haute dimension, les méthodes paramétriques (régression linéaire, réseaux de neurones) sont préférables.

---

<!-- _class: lead -->

# Questions?

**Exercices recommandés** :
- Exercice (knn) : Comparer k-ppv et Nadaraya-Watson sur $\sin(x) + \varepsilon$
- Exercice (knn) : Implémenter LOESS et comparer avec `statsmodels.lowess`
- Exercice (knn) : Montrer le biais aux bords de Nadaraya-Watson vs LOESS
- Exercice (knn) : Fléau de la dimensionnalité : distance au plus proche voisin
