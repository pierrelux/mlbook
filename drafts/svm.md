# Machines à vecteurs de support

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Formuler le problème de classification à marge maximale
- Dériver les formulations primale et duale des SVM
- Expliquer le concept de marge souple et les variables de jeu
- Appliquer l'astuce du noyau pour des frontières non linéaires
- Définir les noyaux de Mercer et la matrice de Gram
- Énoncer le théorème du représentant
- Implémenter un SVM simple avec descente de sous-gradient
```

## Introduction

Les machines à vecteurs de support (SVM, de l'anglais *support vector machines*) sont des classifieurs qui cherchent l'hyperplan séparateur maximisant la marge entre les classes. Cette approche géométrique conduit à des propriétés de généralisation remarquables et à l'utilisation de noyaux pour des frontières non linéaires.

## Classifieurs à marge maximale

### Motivation

Considérons un problème de classification binaire avec des données linéairement séparables. Il existe alors une infinité d'hyperplans séparateurs possibles. Lequel choisir?

L'intuition géométrique suggère de choisir l'hyperplan qui maximise la **marge**, c'est-à-dire la distance au point le plus proche de chaque classe. Un classifieur à grande marge est plus robuste aux perturbations et devrait mieux généraliser.

### Formulation géométrique

Considérons un classifieur linéaire:

$$
h(x) = \text{sign}(f(x)), \quad f(x) = w^\top x + w_0
$$

La frontière de décision est l'hyperplan $\{x : w^\top x + w_0 = 0\}$. Le vecteur $w$ est normal à cet hyperplan.

Pour un point $x$, nous pouvons l'écrire comme:

$$
x = x_\perp + r \frac{w}{\|w\|}
$$

où $x_\perp$ est la projection orthogonale de $x$ sur l'hyperplan et $r$ est la distance signée à l'hyperplan.

En substituant dans $f(x)$:

$$
f(x) = w^\top x_\perp + w_0 + r\|w\| = r\|w\|
$$

car $w^\top x_\perp + w_0 = 0$ par définition de l'hyperplan. La distance signée est donc:

$$
r = \frac{f(x)}{\|w\|}
$$

### Contraintes de marge

Pour que les exemples soient correctement classifiés avec une marge, nous exigeons:

$$
\tilde{y}_n f(x_n) \geq 1 \quad \text{pour tout } n
$$

où $\tilde{y}_n \in \{-1, +1\}$ est l'étiquette (avec cette convention plutôt que $\{0, 1\}$). La marge géométrique est alors:

$$
\gamma = \min_n \frac{\tilde{y}_n f(x_n)}{\|w\|} = \frac{1}{\|w\|}
$$

car nous avons imposé $\tilde{y}_n f(x_n) \geq 1$.

### Problème d'optimisation primal

Maximiser la marge $1/\|w\|$ équivaut à minimiser $\|w\|^2$. Le problème de classification à **marge dure** (hard margin) est:

$$
\min_{w, w_0} \frac{1}{2}\|w\|^2 \quad \text{tel que} \quad \tilde{y}_n(w^\top x_n + w_0) \geq 1, \quad n = 1, \ldots, N
$$

C'est un problème d'optimisation quadratique convexe avec $d + 1$ variables et $N$ contraintes linéaires.

## Marge souple

### Motivation

Si les données ne sont pas linéairement séparables, les contraintes $\tilde{y}_n f(x_n) \geq 1$ ne peuvent pas être satisfaites simultanément. Nous introduisons des **variables de jeu** (slack variables) $\xi_n \geq 0$ qui permettent de violer les contraintes.

### Formulation

Le problème de classification à **marge souple** (soft margin) est:

$$
\min_{w, w_0, \xi} \frac{1}{2}\|w\|^2 + C \sum_{n=1}^{N} \xi_n
$$

sous les contraintes:

$$
\xi_n \geq 0, \quad \tilde{y}_n(w^\top x_n + w_0) \geq 1 - \xi_n
$$

Le paramètre $C > 0$ contrôle le compromis entre maximiser la marge et minimiser les violations.

### Interprétation des variables de jeu

- $\xi_n = 0$: le point est correctement classifié avec une marge suffisante
- $0 < \xi_n < 1$: le point est dans la marge mais du bon côté de l'hyperplan
- $\xi_n \geq 1$: le point est mal classifié

La somme $\sum_n \xi_n$ est une borne supérieure sur le nombre d'erreurs.

## Formulation duale

### Lagrangien

Le lagrangien du problème à marge souple est:

$$
\mathcal{L}(w, w_0, \xi, \alpha, \mu) = \frac{1}{2}w^\top w + C\sum_{n=1}^N \xi_n - \sum_{n=1}^N \alpha_n\left(\tilde{y}_n(w^\top x_n + w_0) - 1 + \xi_n\right) - \sum_{n=1}^N \mu_n \xi_n
$$

où $\alpha_n \geq 0$ et $\mu_n \geq 0$ sont les multiplicateurs de Lagrange.

### Conditions d'optimalité

En annulant les dérivées par rapport aux variables primales:

$$
\frac{\partial \mathcal{L}}{\partial w} = w - \sum_{n=1}^N \alpha_n \tilde{y}_n x_n = 0 \implies w = \sum_{n=1}^N \alpha_n \tilde{y}_n x_n
$$

$$
\frac{\partial \mathcal{L}}{\partial w_0} = -\sum_{n=1}^N \alpha_n \tilde{y}_n = 0
$$

$$
\frac{\partial \mathcal{L}}{\partial \xi_n} = C - \alpha_n - \mu_n = 0 \implies \alpha_n \leq C
$$

### Problème dual

En substituant ces conditions dans le lagrangien, nous obtenons le problème dual:

$$
\max_\alpha \sum_{n=1}^N \alpha_n - \frac{1}{2}\sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j \tilde{y}_i \tilde{y}_j x_i^\top x_j
$$

sous les contraintes:

$$
0 \leq \alpha_n \leq C, \quad \sum_{n=1}^N \alpha_n \tilde{y}_n = 0
$$

### Vecteurs de support

Les conditions de Karush-Kuhn-Tucker (KKT) impliquent que $\alpha_n > 0$ seulement si la contrainte correspondante est active:

$$
\alpha_n(\tilde{y}_n f(x_n) - 1 + \xi_n) = 0
$$

Les exemples avec $\alpha_n > 0$ sont les **vecteurs de support**. Ils se trouvent sur la frontière de la marge ($\xi_n = 0$) ou à l'intérieur de la marge ($\xi_n > 0$).

La solution optimale ne dépend que des vecteurs de support, d'où le nom de la méthode. En pratique, seule une fraction des exemples d'entraînement sont des vecteurs de support.

### Prédiction

La fonction de prédiction s'exprime en fonction des vecteurs de support:

$$
f(x) = \sum_{n \in \mathcal{S}} \alpha_n \tilde{y}_n x_n^\top x + w_0
$$

où $\mathcal{S}$ est l'ensemble des indices des vecteurs de support.

Le biais $w_0$ se calcule à partir des vecteurs de support sur la marge ($0 < \alpha_n < C$):

$$
w_0 = \frac{1}{|\mathcal{M}|} \sum_{n \in \mathcal{M}} \left(\tilde{y}_n - \sum_{m \in \mathcal{S}} \alpha_m \tilde{y}_m x_m^\top x_n\right)
$$

## Lien avec la perte à charnière

Le problème SVM à marge souple peut se reformuler comme:

$$
\min_w \sum_{n=1}^N \ell_{\text{hinge}}(\tilde{y}_n, f(x_n)) + \lambda\|w\|^2
$$

où $\lambda = 1/(2C)$ et la **perte à charnière** est:

$$
\ell_{\text{hinge}}(\tilde{y}, \eta) = \max(0, 1 - \tilde{y}\eta) = [1 - \tilde{y}\eta]_+
$$

Cette formulation est analogue à la régression logistique régularisée, avec la perte à charnière remplaçant la perte logistique.

## L'astuce du noyau

### Motivation

La formulation duale ne dépend des données qu'à travers les produits scalaires $x_i^\top x_j$. Nous pouvons remplacer ces produits par une fonction de noyau:

$$
\mathcal{K}(x_i, x_j) = \phi(x_i)^\top \phi(x_j)
$$

où $\phi: \mathbb{R}^d \to \mathbb{R}^m$ est une transformation (potentiellement vers un espace de dimension infinie).

### Problème dual avec noyau

$$
\max_\alpha \sum_{n=1}^N \alpha_n - \frac{1}{2}\sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j \tilde{y}_i \tilde{y}_j \mathcal{K}(x_i, x_j)
$$

La fonction de prédiction devient:

$$
f(x) = \sum_{n \in \mathcal{S}} \alpha_n \tilde{y}_n \mathcal{K}(x_n, x) + w_0
$$

L'astuce est que nous n'avons jamais besoin de calculer $\phi(x)$ explicitement. Nous évaluons seulement le noyau.

## Noyaux de Mercer

### Définition

Un noyau $\mathcal{K}: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ est un **noyau de Mercer** (ou noyau défini positif) si pour tout ensemble de points $\{x_1, \ldots, x_N\}$ et tout vecteur $c \in \mathbb{R}^N$:

$$
\sum_{i=1}^N \sum_{j=1}^N c_i c_j \mathcal{K}(x_i, x_j) \geq 0
$$

De manière équivalente, la **matrice de Gram**:

$$
K_{ij} = \mathcal{K}(x_i, x_j)
$$

doit être semi-définie positive pour tout ensemble de points.

### Théorème de Mercer

Si $\mathcal{K}$ est un noyau de Mercer, alors il existe un espace de Hilbert $\mathcal{H}$ et une fonction $\phi: \mathcal{X} \to \mathcal{H}$ tels que:

$$
\mathcal{K}(x, x') = \langle\phi(x), \phi(x')\rangle_{\mathcal{H}}
$$

Le noyau est donc bien un produit scalaire dans un certain espace de caractéristiques.

### Exemples de noyaux

**Noyau linéaire**:
$$
\mathcal{K}(x, x') = x^\top x'
$$

**Noyau polynomial**:
$$
\mathcal{K}(x, x') = (x^\top x' + c)^d
$$

où $d$ est le degré et $c \geq 0$.

**Noyau gaussien (RBF)**:
$$
\mathcal{K}(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)
$$

où $\ell$ est le paramètre de bande passante. Ce noyau correspond à un espace de caractéristiques de dimension infinie.

**Noyau périodique**:
$$
\mathcal{K}(r) = \exp\left(-\frac{2}{\ell^2}\sin^2\left(\pi\frac{r}{p}\right)\right)
$$

où $r = |x - x'|$ et $p$ est la période.

### Composition de noyaux

De nouveaux noyaux valides peuvent être construits par:

- **Multiplication par un scalaire**: $c \cdot \mathcal{K}(x, x')$ pour $c > 0$
- **Somme**: $\mathcal{K}_1(x, x') + \mathcal{K}_2(x, x')$
- **Produit**: $\mathcal{K}_1(x, x') \cdot \mathcal{K}_2(x, x')$
- **Exponentielle**: $\exp(\mathcal{K}(x, x'))$
- **Polynôme**: $q(\mathcal{K}(x, x'))$ pour tout polynôme $q$ à coefficients non négatifs

### Exemple: le noyau quadratique

Pour $x = (x_1, x_2) \in \mathbb{R}^2$, le noyau quadratique $\mathcal{K}(x, x') = (x^\top x')^2$ se développe:

$$
\mathcal{K}(x, x') = (x_1 x_1' + x_2 x_2')^2 = x_1^2 (x_1')^2 + 2x_1 x_2 x_1' x_2' + x_2^2 (x_2')^2
$$

Ceci correspond à $\phi(x) = [x_1^2, \sqrt{2}x_1 x_2, x_2^2]^\top \in \mathbb{R}^3$.

Le noyau quadratique en 2D équivaut à un produit scalaire dans un espace de caractéristiques en 3D.

## Le théorème du représentant

### Énoncé

Soit $\mathcal{H}_\mathcal{K}$ un espace de Hilbert à noyau reproduisant (RKHS) associé au noyau $\mathcal{K}$. Pour tout problème de la forme:

$$
\min_{f \in \mathcal{H}_\mathcal{K}} \sum_{n=1}^N L(y_n, f(x_n)) + \lambda\|f\|_{\mathcal{H}_\mathcal{K}}^2
$$

la solution optimale s'écrit:

$$
f(x) = \sum_{n=1}^N \alpha_n \mathcal{K}(x, x_n)
$$

### Implications

Ce théorème est fondamental car:

1. Il réduit un problème de dimension infinie (optimiser sur toutes les fonctions de $\mathcal{H}_\mathcal{K}$) à un problème de dimension finie ($N$ coefficients $\alpha_n$)
2. Il justifie l'utilisation de méthodes à noyaux dans de nombreux contextes au-delà des SVM

### Application: régression ridge à noyau

La régression ridge:

$$
\min_w \|y - Xw\|^2 + \lambda\|w\|^2
$$

a pour forme duale:

$$
\min_\alpha (y - K\alpha)^\top(y - K\alpha) + \lambda \alpha^\top K \alpha
$$

La solution est $\hat{\alpha} = (K + \lambda I)^{-1}y$ et les prédictions sont $\hat{f} = K\hat{\alpha}$.

## Optimisation par sous-gradient

La perte à charnière n'est pas différentiable en $\tilde{y}\eta = 1$. Nous utilisons la méthode du **sous-gradient**.

### Sous-gradient

Pour une fonction convexe $f$, un vecteur $g$ est un **sous-gradient** de $f$ en $x$ si:

$$
f(z) \geq f(x) + g^\top(z - x) \quad \text{pour tout } z
$$

L'ensemble des sous-gradients est le **sous-différentiel** $\partial f(x)$.

Pour la perte à charnière $\ell(\tilde{y}, \eta) = [1 - \tilde{y}\eta]_+$:

$$
\partial_\eta \ell(\tilde{y}, \eta) = \begin{cases}
0 & \text{si } \tilde{y}\eta > 1 \\
[-\tilde{y}, 0] & \text{si } \tilde{y}\eta = 1 \\
-\tilde{y} & \text{si } \tilde{y}\eta < 1
\end{cases}
$$

### Descente de sous-gradient

La mise à jour pour le SVM linéaire avec perte à charnière est:

$$
w_{t+1} = w_t - \eta_t \left(\lambda w_t + \frac{1}{N}\sum_{n: \tilde{y}_n f(x_n) < 1} -\tilde{y}_n x_n\right)
$$

Les exemples contribuant au gradient sont ceux dont la marge est insuffisante.

## Implémentation

```python
import numpy as np

def hinge_loss(y, scores):
    """Perte à charnière."""
    return np.maximum(0, 1 - y * scores)

def svm_sgd(X, y, C=1.0, lr=0.01, n_epochs=100):
    """SVM linéaire par descente de sous-gradient.
    
    Args:
        X: Matrice de design (N x d)
        y: Étiquettes dans {-1, +1} (N,)
        C: Paramètre de régularisation
        lr: Taux d'apprentissage initial
        n_epochs: Nombre d'époques
        
    Returns:
        Vecteur de poids (d,), biais
    """
    N, d = X.shape
    w = np.zeros(d)
    b = 0.0
    
    for epoch in range(n_epochs):
        for i in np.random.permutation(N):
            eta = lr / (1 + epoch)  # Décroissance du taux
            
            score = np.dot(X[i], w) + b
            
            if y[i] * score < 1:
                # Exemple avec marge insuffisante
                w = (1 - eta / C) * w + eta * y[i] * X[i]
                b = b + eta * y[i]
            else:
                # Exemple bien classifié avec marge
                w = (1 - eta / C) * w
                
    return w, b

def predict(X, w, b):
    """Prédictions SVM."""
    return np.sign(X @ w + b)
```

## Résumé

Ce chapitre a présenté les machines à vecteurs de support:

- Les **SVM** maximisent la marge géométrique entre les classes
- Le problème **primal** minimise $\|w\|^2$ sous contraintes de marge
- Les **variables de jeu** permettent de traiter les données non séparables
- Le problème **dual** ne dépend que des produits scalaires, permettant l'**astuce du noyau**
- Les **noyaux de Mercer** définissent des produits scalaires implicites dans des espaces de grande dimension
- Le **théorème du représentant** justifie que la solution est une combinaison linéaire de noyaux
- La **perte à charnière** relie les SVM à la régularisation

Le chapitre suivant introduit les réseaux de neurones, qui étendent les modèles linéaires par composition de transformations non linéaires.
