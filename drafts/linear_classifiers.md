# Classifieurs linéaires

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Formuler la régression logistique binaire et multiclasse
- Dériver le gradient de la log-vraisemblance négative
- Expliquer la fonction softmax et l'entropie croisée
- Implémenter la descente de gradient et ses variantes (SGD, momentum, Adam)
- Distinguer modèles génératifs et discriminatifs
- Décrire le perceptron et ses limitations
- Interpréter géométriquement la frontière de décision linéaire
```

## Introduction

Les classifieurs linéaires prédisent une classe à partir d'une combinaison linéaire des caractéristiques. Malgré leur simplicité, ils forment la base de nombreuses méthodes plus avancées et restent utiles en pratique lorsque les données sont approximativement séparables linéairement.

Ce chapitre présente la régression logistique, le modèle discriminatif de classification par excellence, ainsi que les méthodes d'optimisation par gradient nécessaires à son apprentissage.

## Modèles génératifs vs discriminatifs

Avant d'aborder la régression logistique, distinguons deux approches fondamentales de la classification probabiliste.

### Modèles génératifs

Un **modèle génératif** modélise la distribution jointe $p(x, y)$ ou, de manière équivalente, la vraisemblance $p(x | y)$ et l'a priori $p(y)$. Par le théorème de Bayes:

$$
p(y = c | x) = \frac{p(x | y = c) p(y = c)}{\sum_{c'} p(x | y = c') p(y = c')}
$$

Le classifieur naïf bayésien et l'analyse discriminante gaussienne sont des exemples de modèles génératifs.

**Avantages**:
- Parfois plus faciles à apprendre (comptage, moyennes)
- Gèrent naturellement les caractéristiques manquantes
- Peuvent générer de nouvelles instances
- Permettent d'ajouter de nouvelles classes sans réentraîner

**Inconvénients**:
- Suppositions de modélisation parfois irréalistes
- Estimation de $p(x | y)$ peut être complexe en haute dimension

### Modèles discriminatifs

Un **modèle discriminatif** modélise directement la distribution conditionnelle $p(y | x)$ sans passer par la vraisemblance. La régression logistique en est l'exemple canonique.

**Avantages**:
- Meilleure précision prédictive (objectif directement optimisé)
- Compatibles avec le prétraitement des caractéristiques
- Probabilités souvent mieux calibrées

**Inconvénients**:
- Ne peuvent pas générer de nouvelles instances
- Requièrent une optimisation numérique

En pratique, les modèles discriminatifs dominent lorsque l'objectif est uniquement la classification.

## Régression logistique binaire

### Modèle probabiliste

En régression logistique binaire, nous modélisons la probabilité de la classe positive par une fonction sigmoïde appliquée à une combinaison linéaire:

$$
p(y = 1 | x; \theta) = \sigma(w^\top x + b) = \frac{1}{1 + e^{-(w^\top x + b)}}
$$

où $\sigma: \mathbb{R} \to (0, 1)$ est la **fonction sigmoïde**:

$$
\sigma(a) = \frac{1}{1 + e^{-a}}
$$

Le modèle complet s'écrit:

$$
p(y | x; \theta) = \text{Ber}(y | \sigma(w^\top x + b))
$$

où $\text{Ber}(y | \mu) = \mu^y (1 - \mu)^{1-y}$ est la distribution de Bernoulli.

### Logit et log-odds

La quantité $a = w^\top x + b$ est appelée **logit** ou **pré-activation**. Elle est reliée aux probabilités par:

$$
a = \log \frac{p(y = 1 | x)}{p(y = 0 | x)} = \log \frac{p(y = 1 | x)}{1 - p(y = 1 | x)}
$$

Cette quantité est le **logarithme de la cote** (log-odds). La fonction sigmoïde transforme ce log-odds en probabilité.

### Frontière de décision

La frontière de décision est l'ensemble des points où les deux classes sont équiprobables:

$$
p(y = 1 | x) = p(y = 0 | x) = 0.5 \iff w^\top x + b = 0
$$

Cette équation définit un **hyperplan** de vecteur normal $w$ et de décalage $b$. La classification s'effectue par:

$$
\hat{y} = \mathbb{1}(w^\top x + b > 0)
$$

Les données sont dites **linéairement séparables** si une frontière de décision linéaire peut les séparer parfaitement.

## Estimation par maximum de vraisemblance

### Log-vraisemblance négative

La log-vraisemblance négative (NLL) pour la régression logistique est:

$$
\text{NLL}(w) = -\frac{1}{N} \sum_{n=1}^{N} \left[y_n \log \mu_n + (1 - y_n) \log(1 - \mu_n)\right]
$$

où $\mu_n = \sigma(w^\top x_n + b)$ est la probabilité prédite pour l'exemple $n$.

Cette expression est l'**entropie croisée binaire** moyennée:

$$
\text{NLL}(w) = \frac{1}{N} \sum_{n=1}^{N} \mathbb{H}_{\text{ce}}(y_n, \mu_n)
$$

où $\mathbb{H}_{\text{ce}}(p, q) = -[p \log q + (1-p) \log(1-q)]$.

### Calcul du gradient

Le gradient de la NLL a une forme élégante:

$$
\nabla_w \text{NLL}(w) = \frac{1}{N} \sum_{n=1}^{N} (\mu_n - y_n) x_n
$$

**Démonstration**: Pour un exemple $(x_n, y_n)$, la perte est:
$$
\ell_n = -y_n \log \mu_n - (1 - y_n) \log(1 - \mu_n)
$$

En utilisant $\frac{d\sigma}{da} = \sigma(a)(1 - \sigma(a)) = \mu(1-\mu)$:

$$
\frac{\partial \ell_n}{\partial w} = \left(-\frac{y_n}{\mu_n} + \frac{1 - y_n}{1 - \mu_n}\right) \mu_n(1 - \mu_n) x_n = (\mu_n - y_n) x_n
$$

Le gradient est donc une moyenne pondérée des entrées, où le poids est l'erreur de prédiction $e_n = \mu_n - y_n$.

### Convexité

La NLL de la régression logistique est **strictement convexe**. La matrice hessienne est:

$$
H(w) = \frac{1}{N} X^\top S X
$$

où $S = \text{diag}(\mu_1(1-\mu_1), \ldots, \mu_N(1-\mu_N))$ est une matrice diagonale positive.

Pour tout vecteur $v \neq 0$:

$$
v^\top X^\top S X v = \|S^{1/2} X v\|_2^2 > 0
$$

donc $H$ est définie positive. La NLL admet donc un unique minimum global.

## Régression logistique multiclasse

### Fonction softmax

Pour la classification à $C$ classes, nous généralisons la sigmoïde par la fonction **softmax**:

$$
p(y = c | x; \theta) = \text{softmax}(a)_c = \frac{e^{a_c}}{\sum_{c'=1}^{C} e^{a_{c'}}}
$$

où $a = Wx + b$ est le vecteur de logits, $W \in \mathbb{R}^{C \times d}$ est la matrice de poids, et $b \in \mathbb{R}^C$ est le vecteur de biais.

La fonction softmax transforme un vecteur de scores arbitraires en distribution de probabilité: les sorties sont positives et somment à 1.

### Entropie croisée catégorielle

La log-vraisemblance négative devient:

$$
\text{NLL}(\theta) = -\frac{1}{N} \sum_{n=1}^{N} \sum_{c=1}^{C} y_{nc} \log \mu_{nc} = \frac{1}{N} \sum_{n=1}^{N} \mathbb{H}_{\text{ce}}(y_n, \mu_n)
$$

où $y_{nc} = \mathbb{1}(y_n = c)$ est l'encodage one-hot et $\mu_{nc} = p(y = c | x_n)$.

L'entropie croisée catégorielle est:

$$
\mathbb{H}_{\text{ce}}(p, q) = -\sum_{c=1}^{C} p_c \log q_c
$$

## Optimisation par gradient

Il n'existe pas de solution analytique pour la régression logistique. Nous devons recourir à l'optimisation numérique.

### Descente de gradient

La **descente de gradient** met à jour itérativement les paramètres dans la direction opposée au gradient:

$$
\theta_{t+1} = \theta_t - \eta_t \nabla \mathcal{L}(\theta_t)
$$

où $\eta_t > 0$ est le **taux d'apprentissage** (ou pas d'apprentissage).

Un vecteur $d$ est une **direction de descente** si $d^\top \nabla \mathcal{L} < 0$. Le gradient négatif est la direction de descente la plus rapide (pour la norme euclidienne).

### Descente de gradient stochastique

La **descente de gradient stochastique** (SGD) remplace le gradient exact par une estimation basée sur un sous-ensemble des données:

$$
\theta_{t+1} = \theta_t - \eta_t \nabla \mathcal{L}(\theta_t, z_t)
$$

où $z_t$ est un exemple (ou mini-lot) échantillonné aléatoirement.

**Mini-lots**: En pratique, nous utilisons des mini-lots de taille $B$:

$$
g_t = \frac{1}{B} \sum_{n \in \mathcal{B}_t} \nabla \ell(y_n, f(x_n; \theta_t))
$$

Cette estimation est non biaisée et réduit la variance par rapport au SGD pur ($B = 1$).

**Avantages du SGD**:
- Plus rapide en pratique que le gradient complet
- Régularisation implicite par le bruit
- Permet l'apprentissage en ligne (streaming)

### Momentum

L'algorithme de **momentum** (Polyak, 1964) accumule les gradients passés pour atténuer les oscillations:

$$
\begin{aligned}
m_t &= \beta m_{t-1} + g_{t-1} \\
\theta_t &= \theta_{t-1} - \eta_t m_t
\end{aligned}
$$

Le terme $m_t$ est une moyenne mobile exponentielle des gradients:

$$
m_t = \sum_{\tau=0}^{t-1} \beta^\tau g_{t-\tau-1}
$$

Le paramètre $\beta \in [0, 1)$ contrôle la mémoire (typiquement $\beta = 0.9$).

### Taux d'apprentissage adaptatifs

**RMSProp** adapte le taux d'apprentissage à chaque paramètre:

$$
\begin{aligned}
s_{t+1,d} &= \beta s_{t,d} + (1 - \beta) g_{t,d}^2 \\
\theta_{t+1} &= \theta_t - \frac{\eta_t}{\sqrt{s_t + \epsilon}} \odot g_t
\end{aligned}
$$

où $\odot$ désigne le produit élément par élément.

**Adam** combine momentum et RMSProp:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
s_t &= \beta_2 s_{t-1} + (1 - \beta_2) g_t^2 \\
\theta_{t+1} &= \theta_t - \frac{\eta_t}{\sqrt{s_t} + \epsilon} m_t
\end{aligned}
$$

avec correction de biais pour les premières itérations:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{s}_t = \frac{s_t}{1 - \beta_2^t}
$$

Valeurs typiques: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

### Planification du taux d'apprentissage

Le taux d'apprentissage peut varier au cours de l'entraînement. Les conditions de Robbins-Monro garantissent la convergence:

$$
\sum_{t=1}^{\infty} \eta_t = \infty, \quad \sum_{t=1}^{\infty} \eta_t^2 < \infty
$$

Exemples de planifications:
- **Décroissance polynomiale**: $\eta_t = \eta_0 (\beta t + 1)^{-\alpha}$
- **Décroissance exponentielle**: $\eta_t = \eta_0 e^{-\lambda t}$
- **Constante par morceaux**: réduction par facteur à des époques fixées

La planification en racine carrée $\eta_t = \eta_0 / \sqrt{t+1}$ est un choix courant.

## Le perceptron

Le **perceptron** (Rosenblatt, 1957) est un précurseur de la régression logistique:

$$
f(x; w, b) = \mathbb{1}(w^\top x + b > 0)
$$

La règle de mise à jour est:

$$
w_{t+1} = w_t - \eta_t (\hat{y}_n - y_n) x_n
$$

où $\hat{y}_n = f(x_n)$ est la prédiction (0 ou 1).

### Comparaison avec la régression logistique

| Aspect | Perceptron | Régression logistique |
|--------|------------|----------------------|
| Sortie | Binaire $\{0, 1\}$ | Probabilité $[0, 1]$ |
| Fonction | Marche d'escalier | Sigmoïde |
| Différentiabilité | Non | Oui |
| Convergence | Seulement si séparable | Toujours |

Le perceptron converge en un nombre fini d'itérations si les données sont linéairement séparables (théorème de convergence du perceptron). Sinon, il oscille indéfiniment.

## Transformation non linéaire

Lorsque les données ne sont pas linéairement séparables, nous pouvons les transformer dans un espace de dimension supérieure où elles le deviennent.

Par exemple, l'expansion polynomiale de degré 2 transforme $x = (x_1, x_2)$ en:

$$
\phi(x) = [1, x_1, x_2, x_1^2, x_2^2, x_1 x_2]
$$

La frontière de décision dans l'espace original devient quadratique.

Cette idée est généralisée par les méthodes à noyau (chapitre SVM), où la transformation peut être implicite.

## Implémentation

### Régression logistique binaire

```python
import numpy as np

def sigmoid(a):
    """Fonction sigmoïde numériquement stable."""
    return np.where(a >= 0,
                    1 / (1 + np.exp(-a)),
                    np.exp(a) / (1 + np.exp(a)))

def logistic_regression_sgd(X, y, lr=0.01, n_epochs=100, batch_size=32):
    """Régression logistique par SGD.
    
    Args:
        X: Matrice de design (N x d)
        y: Étiquettes binaires (N,)
        lr: Taux d'apprentissage
        n_epochs: Nombre d'époques
        batch_size: Taille des mini-lots
        
    Returns:
        Vecteur de poids (d,)
    """
    N, d = X.shape
    w = np.zeros(d)
    
    for epoch in range(n_epochs):
        # Mélanger les données
        indices = np.random.permutation(N)
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_idx = indices[start:end]
            
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            # Prédictions
            mu = sigmoid(X_batch @ w)
            
            # Gradient
            grad = X_batch.T @ (mu - y_batch) / len(batch_idx)
            
            # Mise à jour
            w = w - lr * grad
            
    return w

def predict_proba(X, w):
    """Probabilités de la classe positive."""
    return sigmoid(X @ w)

def predict(X, w, threshold=0.5):
    """Prédictions binaires."""
    return (predict_proba(X, w) >= threshold).astype(int)
```

### Régression logistique multiclasse

```python
def softmax(a):
    """Fonction softmax numériquement stable."""
    a_max = np.max(a, axis=1, keepdims=True)
    exp_a = np.exp(a - a_max)
    return exp_a / np.sum(exp_a, axis=1, keepdims=True)

def multiclass_logistic_sgd(X, y, n_classes, lr=0.01, n_epochs=100, batch_size=32):
    """Régression logistique multiclasse par SGD.
    
    Args:
        X: Matrice de design (N x d)
        y: Étiquettes (N,) avec valeurs dans {0, ..., C-1}
        n_classes: Nombre de classes C
        lr: Taux d'apprentissage
        n_epochs: Nombre d'époques
        batch_size: Taille des mini-lots
        
    Returns:
        Matrice de poids (C x d)
    """
    N, d = X.shape
    W = np.zeros((n_classes, d))
    
    # Encodage one-hot
    Y_onehot = np.eye(n_classes)[y]
    
    for epoch in range(n_epochs):
        indices = np.random.permutation(N)
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_idx = indices[start:end]
            
            X_batch = X[batch_idx]
            Y_batch = Y_onehot[batch_idx]
            
            # Prédictions
            logits = X_batch @ W.T
            probs = softmax(logits)
            
            # Gradient
            grad = (probs - Y_batch).T @ X_batch / len(batch_idx)
            
            # Mise à jour
            W = W - lr * grad
            
    return W
```

## Résumé

Ce chapitre a présenté les classifieurs linéaires discriminatifs:

- La **régression logistique** modélise $p(y|x)$ directement via une sigmoïde
- La **NLL** équivaut à l'**entropie croisée**, objectif convexe avec minimum unique
- Le gradient a une forme simple: $\sum_n (\mu_n - y_n) x_n$
- La **fonction softmax** généralise la sigmoïde à plusieurs classes
- **SGD** et ses variantes (momentum, Adam) permettent l'optimisation à grande échelle
- Le **perceptron** est un cas limite avec prédictions binaires
- Les **transformations non linéaires** permettent des frontières non linéaires

Le chapitre suivant étend ces idées aux machines à vecteurs de support, qui optimisent la marge de classification.
