# Régression linéaire

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Formuler le problème de régression linéaire
- Dériver la solution des moindres carrés ordinaires (MCO)
- Relier les MCO à l'estimateur du maximum de vraisemblance
- Distinguer régression homoscédastique et hétéroscédastique
- Étendre le modèle à la régression polynomiale
- Appliquer la régression ridge pour la régularisation
- Implémenter la régression linéaire par formule analytique et gradient
```

## Introduction

La régression linéaire est le modèle fondamental de l'apprentissage supervisé pour les problèmes à sortie continue. Malgré sa simplicité, elle reste omniprésente en pratique: comme premier modèle de base, comme composante de modèles plus complexes, et comme outil d'analyse et d'interprétation.

Ce chapitre développe la régression linéaire sous trois angles complémentaires: géométrique (projection), statistique (maximum de vraisemblance), et algorithmique (optimisation).

## Formulation du problème

### Le modèle linéaire

En régression linéaire, nous modélisons la cible comme une combinaison linéaire des caractéristiques:

$$
f(x; w, b) = w^\top x + b = \sum_{j=1}^{d} w_j x_j + b
$$

où:
- $x \in \mathbb{R}^d$ est le vecteur de caractéristiques
- $w \in \mathbb{R}^d$ est le vecteur de poids
- $b \in \mathbb{R}$ est le biais (ou ordonnée à l'origine)

Pour simplifier la notation, nous absorbons souvent le biais dans les poids en ajoutant une caractéristique constante égale à 1:

$$
\tilde{x} = [1, x_1, \ldots, x_d]^\top \in \mathbb{R}^{d+1}, \quad \tilde{w} = [b, w_1, \ldots, w_d]^\top
$$

La prédiction devient alors $f(x; \tilde{w}) = \tilde{w}^\top \tilde{x}$.

### Notation matricielle

Pour un ensemble de $N$ exemples, nous formons la matrice de design $X \in \mathbb{R}^{N \times d}$ dont chaque ligne est un exemple:

$$
X = \begin{bmatrix}
x_1^\top \\
x_2^\top \\
\vdots \\
x_N^\top
\end{bmatrix}, \quad
y = \begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_N
\end{bmatrix}
$$

Les prédictions pour tous les exemples s'écrivent $\hat{y} = Xw$.

## Moindres carrés ordinaires

### Fonction objectif

L'objectif est de trouver les poids qui minimisent la somme des carrés des résidus (SCR, en anglais *residual sum of squares*, RSS):

$$
\text{RSS}(w) = \sum_{n=1}^{N} (y_n - w^\top x_n)^2 = \|y - Xw\|_2^2
$$

Le **résidu** $r_n = y_n - w^\top x_n$ est l'écart entre la valeur observée et la prédiction.

En développant la norme:

$$
\text{RSS}(w) = (y - Xw)^\top (y - Xw) = y^\top y - 2w^\top X^\top y + w^\top X^\top X w
$$

### Dérivation de la solution

Pour trouver le minimum, nous annulons le gradient par rapport à $w$:

$$
\nabla_w \text{RSS}(w) = -2X^\top y + 2X^\top X w = 0
$$

Ce qui donne les **équations normales**:

$$
X^\top X w = X^\top y
$$

Si $X^\top X$ est inversible (ce qui requiert que $X$ soit de rang plein), la solution est:

$$
\hat{w}_{\text{MCO}} = (X^\top X)^{-1} X^\top y
$$

Cette solution porte le nom d'estimateur des **moindres carrés ordinaires** (MCO, en anglais *ordinary least squares*, OLS).

### Interprétation géométrique

La solution des MCO a une interprétation géométrique élégante. Les colonnes de $X$ définissent un sous-espace de $\mathbb{R}^N$. Le vecteur $\hat{y} = X\hat{w}$ est la projection orthogonale de $y$ sur ce sous-espace.

Le résidu $r = y - \hat{y}$ est orthogonal à l'espace colonnes de $X$:

$$
X^\top (y - X\hat{w}) = X^\top y - X^\top X \hat{w} = 0
$$

La matrice $H = X(X^\top X)^{-1}X^\top$ est appelée **matrice chapeau** (hat matrix) car $\hat{y} = Hy$. C'est une matrice de projection.

## Interprétation probabiliste

### Modèle génératif

Nous pouvons interpréter la régression linéaire dans un cadre probabiliste. Supposons que les données soient générées selon:

$$
y = w^\top x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

Le bruit $\epsilon$ capture les variations non expliquées par le modèle linéaire. Ceci implique que la distribution conditionnelle de $y$ sachant $x$ est:

$$
p(y | x; w, \sigma^2) = \mathcal{N}(y | w^\top x, \sigma^2)
$$

### Maximum de vraisemblance

Sous l'hypothèse que les exemples sont indépendants et identiquement distribués, la vraisemblance est:

$$
p(\mathcal{D} | w, \sigma^2) = \prod_{n=1}^{N} \mathcal{N}(y_n | w^\top x_n, \sigma^2)
$$

La log-vraisemblance négative (NLL) est:

$$
\text{NLL}(w) = -\sum_{n=1}^{N} \log \mathcal{N}(y_n | w^\top x_n, \sigma^2)
$$

En développant l'expression de la gaussienne:

$$
\text{NLL}(w) = \frac{N}{2} \log(2\pi\sigma^2) + \frac{1}{2\sigma^2} \sum_{n=1}^{N} (y_n - w^\top x_n)^2
$$

Le premier terme est constant en $w$. Minimiser la NLL équivaut donc à minimiser la somme des carrés des résidus:

$$
\hat{w}_{\text{MLE}} = \arg\min_w \text{NLL}(w) = \arg\min_w \text{RSS}(w) = \hat{w}_{\text{MCO}}
$$

**Conclusion importante**: l'estimateur des moindres carrés ordinaires coïncide avec l'estimateur du maximum de vraisemblance sous un modèle gaussien.

### Estimation de la variance

Une fois $\hat{w}$ obtenu, nous pouvons estimer la variance du bruit:

$$
\hat{\sigma}^2_{\text{MLE}} = \frac{1}{N} \sum_{n=1}^{N} (y_n - \hat{w}^\top x_n)^2 = \frac{\text{RSS}(\hat{w})}{N}
$$

Cette estimation est biaisée. L'estimateur non biaisé utilise $N - d$ au dénominateur pour tenir compte des degrés de liberté:

$$
\hat{\sigma}^2 = \frac{\text{RSS}(\hat{w})}{N - d}
$$

## Métriques d'évaluation

### Erreur quadratique moyenne

L'**erreur quadratique moyenne** (EQM, en anglais *mean squared error*, MSE) est la RSS divisée par $N$:

$$
\text{MSE}(w) = \frac{1}{N} \text{RSS}(w) = \frac{1}{N} \sum_{n=1}^{N} (y_n - w^\top x_n)^2
$$

### Racine de l'erreur quadratique moyenne

La **racine de l'EQM** (REQM, en anglais *root mean squared error*, RMSE) est dans les mêmes unités que la cible:

$$
\text{RMSE}(w) = \sqrt{\text{MSE}(w)}
$$

### Coefficient de détermination

Le **coefficient de détermination** $R^2$ mesure la proportion de variance expliquée par le modèle:

$$
R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = 1 - \frac{\sum_n (y_n - \hat{y}_n)^2}{\sum_n (y_n - \bar{y})^2}
$$

où TSS (total sum of squares) est la variance totale autour de la moyenne $\bar{y}$.

- $R^2 = 1$: le modèle explique toute la variance
- $R^2 = 0$: le modèle n'est pas meilleur que prédire la moyenne
- $R^2 < 0$: le modèle est pire que prédire la moyenne (possible sur l'ensemble de test)

## Homoscédasticité et hétéroscédasticité

### Modèle homoscédastique

Le modèle standard suppose une **variance constante** (homoscédasticité):

$$
p(y | x; w, \sigma^2) = \mathcal{N}(y | w^\top x, \sigma^2)
$$

La variance $\sigma^2$ ne dépend pas de $x$.

### Modèle hétéroscédastique

Dans un modèle **hétéroscédastique**, la variance dépend aussi de l'entrée:

$$
p(y | x; w_\mu, w_\sigma) = \mathcal{N}(y | w_\mu^\top x, \sigma_+(w_\sigma^\top x))
$$

où $\sigma_+(a) = \log(1 + e^a)$ est la fonction softplus qui garantit une variance positive.

Ce modèle exprime l'**incertitude aleatoire** (ou incertitude irréductible): pour certaines valeurs de $x$, les observations sont intrinsèquement plus variables. Par exemple, prédire le prix de maisons de luxe peut être plus incertain que celui de maisons modestes.

La log-vraisemblance négative devient:

$$
\text{NLL}(w_\mu, w_\sigma) = \sum_{n=1}^{N} \left[\log \sigma_+(w_\sigma^\top x_n) + \frac{(y_n - w_\mu^\top x_n)^2}{2\sigma_+(w_\sigma^\top x_n)}\right]
$$

Cette expression n'a pas de solution analytique et doit être optimisée par gradient.

## Régression polynomiale

### Extension non linéaire

La régression linéaire suppose une relation linéaire entre caractéristiques et cible. Pour capturer des relations non linéaires, nous pouvons transformer les caractéristiques.

En **régression polynomiale**, nous appliquons une fonction de redescription $\phi: \mathbb{R} \to \mathbb{R}^{k+1}$:

$$
\phi(x) = [1, x, x^2, \ldots, x^k]
$$

Le modèle devient:

$$
f(x; w) = w^\top \phi(x) = w_0 + w_1 x + w_2 x^2 + \cdots + w_k x^k
$$

Le modèle reste **linéaire dans les paramètres** $w$, même s'il est non linéaire dans l'entrée $x$.

### Généralisation aux fonctions de base

Plus généralement, nous pouvons utiliser n'importe quel ensemble de **fonctions de base** $\phi_1, \ldots, \phi_m$:

$$
f(x; w) = \sum_{j=1}^{m} w_j \phi_j(x)
$$

Exemples:
- Polynômes: $\phi_j(x) = x^{j-1}$
- Fonctions de base radiales: $\phi_j(x) = \exp(-\|x - c_j\|^2 / 2\sigma^2)$
- Fonctions trigonométriques: $\sin(j\omega x), \cos(j\omega x)$

La solution des MCO s'applique directement avec la matrice de design transformée $\Phi$ dont l'entrée $(n, j)$ est $\phi_j(x_n)$.

### Surapprentissage

Un degré polynomial trop élevé mène au surapprentissage. Avec $k = N - 1$ paramètres, nous pouvons interpoler parfaitement les $N$ points d'entraînement, mais le polynôme oscillera violemment entre les points.

Le choix du degré est un problème de sélection de modèle, résolu par validation croisée ou régularisation.

## Régression ridge

### Motivation

La solution MCO $(X^\top X)^{-1} X^\top y$ peut être instable si:
- $X^\top X$ est mal conditionnée (proche de singulière)
- Le nombre de caractéristiques $d$ est proche ou supérieur au nombre d'exemples $N$
- Les caractéristiques sont fortement corrélées (multicolinéarité)

La **régression ridge** ajoute une pénalité sur la norme des poids.

### Formulation

L'objectif de la régression ridge est:

$$
\hat{w}_{\text{ridge}} = \arg\min_w \left[\|y - Xw\|_2^2 + \lambda \|w\|_2^2\right]
$$

Le terme $\lambda \|w\|_2^2$ pénalise les poids de grande magnitude. Le coefficient de régularisation $\lambda > 0$ contrôle la force de cette pénalité.

### Solution analytique

En annulant le gradient:

$$
\nabla_w \left[\|y - Xw\|_2^2 + \lambda \|w\|_2^2\right] = -2X^\top(y - Xw) + 2\lambda w = 0
$$

La solution est:

$$
\hat{w}_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y
$$

L'ajout de $\lambda I$ à $X^\top X$ garantit l'inversibilité et améliore le conditionnement.

### Interprétation bayésienne

La régression ridge correspond à l'estimation MAP avec un a priori gaussien sur les poids:

$$
p(w) = \mathcal{N}(w | 0, \tau^2 I)
$$

Le coefficient de régularisation est relié à la variance de l'a priori par $\lambda = \sigma^2 / \tau^2$.

### Effet de la régularisation

- $\lambda = 0$: pas de régularisation, solution MCO
- $\lambda \to \infty$: les poids tendent vers zéro
- $\lambda$ optimal: compromis biais-variance

La régularisation introduit un biais (les poids sont "rétrécis" vers zéro) mais réduit la variance. La valeur optimale de $\lambda$ se détermine par validation croisée.

## Implémentation

### Solution analytique

```python
import numpy as np

def linear_regression_closed(X, y, reg_lambda=0):
    """Régression linéaire par solution analytique.
    
    Args:
        X: Matrice de design (N x d)
        y: Vecteur cible (N,)
        reg_lambda: Coefficient de régularisation (0 pour MCO)
        
    Returns:
        Vecteur de poids (d,)
    """
    d = X.shape[1]
    # Ajouter régularisation à la diagonale
    A = X.T @ X + reg_lambda * np.eye(d)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    return w
```

### Descente de gradient

Pour les grands ensembles de données, la solution analytique peut être coûteuse ($O(d^3)$ pour l'inversion). La descente de gradient offre une alternative:

```python
def linear_regression_gd(X, y, lr=0.01, n_iters=1000, reg_lambda=0):
    """Régression linéaire par descente de gradient.
    
    Args:
        X: Matrice de design (N x d)
        y: Vecteur cible (N,)
        lr: Taux d'apprentissage
        n_iters: Nombre d'itérations
        reg_lambda: Coefficient de régularisation
        
    Returns:
        Vecteur de poids (d,), historique des pertes
    """
    N, d = X.shape
    w = np.zeros(d)
    losses = []
    
    for i in range(n_iters):
        # Prédictions
        y_pred = X @ w
        
        # Gradient de la MSE + régularisation
        grad = (2/N) * X.T @ (y_pred - y) + 2 * reg_lambda * w
        
        # Mise à jour
        w = w - lr * grad
        
        # Enregistrer la perte
        loss = np.mean((y_pred - y)**2) + reg_lambda * np.sum(w**2)
        losses.append(loss)
        
    return w, losses
```

### Utilisation avec scikit-learn

```python
from sklearn.linear_model import LinearRegression, Ridge

# Régression linéaire standard
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Régression ridge
ridge = Ridge(alpha=1.0)  # alpha correspond à lambda
ridge.fit(X_train, y_train)
predictions_ridge = ridge.predict(X_test)
```

## Résumé

Ce chapitre a présenté la régression linéaire sous plusieurs angles:

- Le modèle linéaire prédit $\hat{y} = w^\top x + b$
- Les **moindres carrés ordinaires** minimisent la somme des carrés des résidus
- La solution analytique est $\hat{w} = (X^\top X)^{-1} X^\top y$
- Sous un modèle gaussien, les MCO coïncident avec le **maximum de vraisemblance**
- La **régression polynomiale** étend le modèle à des relations non linéaires
- La **régression ridge** régularise par pénalité L2, équivalent au MAP bayésien
- Le modèle **hétéroscédastique** permet une variance dépendant de l'entrée

Le chapitre suivant étend ces idées à la classification avec les classifieurs linéaires.
