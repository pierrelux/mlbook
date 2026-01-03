# Boosting

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Expliquer le principe du boosting et de la modélisation additive
- Dériver l'algorithme AdaBoost à partir de la perte exponentielle
- Comprendre le gradient boosting comme descente de gradient fonctionnelle
- Implémenter le gradient tree boosting
- Analyser les fonctions de perte appropriées pour le boosting
- Comparer le boosting au bagging en termes de biais et variance
```

## Modèles additifs

### Structure générale

Les ensembles d'arbres (et plus généralement, les méthodes d'ensemble) correspondent à un modèle de la forme:

$$
f(\boldsymbol{x}; \boldsymbol{\theta}) = \sum_{m=1}^M \beta_m F_m(\boldsymbol{x}; \boldsymbol{\theta}_m)
$$

où $F_m$ est le $m$-ième modèle de base et $\beta_m$ est son poids. Ce type de modèle est appelé un **modèle additif**.

Dans le bagging, nous avons $\beta_m = 1/M$ et les modèles sont entraînés indépendamment. Le boosting adopte une approche différente: les modèles sont entraînés **séquentiellement**, chaque nouveau modèle corrigeant les erreurs des précédents.

### Objectif d'optimisation

Nous cherchons à minimiser:
$$
\mathcal{L}(f) = \sum_{i=1}^N \ell(y_i, f(\boldsymbol{x}_i))
$$

où $\ell$ est une fonction de perte. Le défi est que l'espace des fonctions est de dimension infinie.

## Modélisation additive itérative

### Principe

En **modélisation additive itérative en avant** (*forward stagewise additive modeling*), nous construisons le modèle séquentiellement. À l'étape $m$, nous résolvons:

$$
(\beta_m, \boldsymbol{\theta}_m) = \arg\min_{\beta, \boldsymbol{\theta}} \sum_{i=1}^N \ell\left(y_i, f_{m-1}(\boldsymbol{x}_i) + \beta F(\boldsymbol{x}_i; \boldsymbol{\theta})\right)
$$

Puis nous mettons à jour le modèle:
$$
f_m(\boldsymbol{x}) = f_{m-1}(\boldsymbol{x}) + \beta_m F_m(\boldsymbol{x})
$$

**Important**: nous ne réajustons pas les paramètres des modèles précédents. Chaque modèle est ajouté de manière **incrémentale**.

### Interprétation

Intuitivement:
1. Le modèle $f_m$ tente de corriger les **résidus** de $f_{m-1}$
2. Chaque nouveau modèle se concentre sur les exemples difficiles
3. L'ensemble devient progressivement plus puissant

## Boosting des moindres carrés

### Cas de la perte quadratique

Avec la perte d'erreur quadratique $\ell(y, \hat{y}) = (y - \hat{y})^2$, l'objectif à l'étape $m$ devient:

$$
\ell(y_i, f_{m-1}(\boldsymbol{x}_i) + \beta F(\boldsymbol{x}_i; \boldsymbol{\theta})) = (y_i - f_{m-1}(\boldsymbol{x}_i) - \beta F(\boldsymbol{x}_i; \boldsymbol{\theta}))^2 = (r_{im} - \beta F(\boldsymbol{x}_i; \boldsymbol{\theta}))^2
$$

où $r_{im} = y_i - f_{m-1}(\boldsymbol{x}_i)$ est le **résidu** du modèle actuel sur l'exemple $i$.

### Interprétation des résidus

Le modèle faible $F_m$ apprend à **prédire les résidus**. Ainsi:
- Si $f_{m-1}$ sous-estime $y_i$, le résidu $r_{im}$ est positif
- $F_m$ apprend à produire des corrections positives pour ces exemples
- La somme $f_{m-1} + F_m$ s'approche de $y$

Cette approche s'appelle le **boosting des moindres carrés** (*least squares boosting*).

## AdaBoost

### Contexte: classification binaire

Considérons la classification binaire avec $\tilde{y}_i \in \{-1, +1\}$. Nous modélisons la probabilité comme:

$$
p(y = 1 \mid \boldsymbol{x}) = \frac{e^{F(\boldsymbol{x})}}{e^{-F(\boldsymbol{x})} + e^{F(\boldsymbol{x})}} = \frac{1}{1 + e^{-2F(\boldsymbol{x})}}
$$

où $F(\boldsymbol{x})$ représente la moitié du logarithme des cotes (*log-odds*).

### Fonction de perte exponentielle

La **perte exponentielle** est définie par:
$$
\ell(\tilde{y}, F(\boldsymbol{x})) = \exp(-\tilde{y} F(\boldsymbol{x}))
$$

Propriétés importantes:
1. C'est une **borne supérieure lisse** sur la perte 0-1
2. Elle a la même solution optimale que la perte logarithmique
3. Elle est plus facile à optimiser dans le cadre du boosting

La perte exponentielle est **sensible aux valeurs aberrantes** car elle pénalise exponentiellement les erreurs.

### Dérivation d'AdaBoost

À l'étape $m$, nous minimisons:
$$
L_m(F) = \sum_{i=1}^N \exp\left[-\tilde{y}_i\left(f_{m-1}(\boldsymbol{x}_i) + \beta F(\boldsymbol{x}_i)\right)\right] = \sum_{i=1}^N \omega_{im} \exp\left(-\beta \tilde{y}_i F(\boldsymbol{x}_i)\right)
$$

où $\omega_{im} \triangleq \exp(-\tilde{y}_i f_{m-1}(\boldsymbol{x}_i))$ sont les **poids** des exemples.

Pour un classificateur binaire $F \in \{-1, +1\}$, nous pouvons réécrire:
$$
L_m = (e^\beta - e^{-\beta}) \sum_{i=1}^N \omega_{im} \mathbb{I}(\tilde{y}_i \neq F(\boldsymbol{x}_i)) + e^{-\beta} \sum_{i=1}^N \omega_{im}
$$

### Choix du modèle de base

Le modèle de base optimal minimise l'erreur pondérée:
$$
F_m = \arg\min_F \sum_{i=1}^N \omega_{im} \mathbb{I}(\tilde{y}_i \neq F(\boldsymbol{x}_i))
$$

Nous entraînons donc un classificateur sur une version **pondérée** des données.

### Choix du coefficient

Le coefficient optimal est:
$$
\beta_m = \frac{1}{2} \log \frac{1 - \text{err}_m}{\text{err}_m}
$$

où l'erreur pondérée est:
$$
\text{err}_m = \frac{\sum_{i=1}^N \omega_{im} \mathbb{I}(\tilde{y}_i \neq F_m(\boldsymbol{x}_i))}{\sum_{i=1}^N \omega_{im}}
$$

Remarques:
- Si $\text{err}_m = 0$, alors $\beta_m \to +\infty$ (confiance totale)
- Si $\text{err}_m = 0.5$, alors $\beta_m = 0$ (pas mieux que le hasard)
- Si $\text{err}_m > 0.5$, alors $\beta_m < 0$ (inversé!)

### Mise à jour des poids

Les poids pour l'itération suivante sont:
$$
\omega_{i,m+1} = \omega_{im} \exp(-\tilde{y}_i \beta_m F_m(\boldsymbol{x}_i))
$$

En pratique, cela signifie:
- Les exemples **mal classifiés** ($\tilde{y}_i \neq F_m(\boldsymbol{x}_i)$) voient leur poids **augmenter** exponentiellement
- Les exemples **bien classifiés** voient leur poids diminuer

### Algorithme AdaBoost.M1

```
Entrée: Données {(x_i, y_i)}, nombre d'itérations M
1. Initialiser ω_i = 1/N pour tout i
2. Pour m = 1 à M:
   a. Apprendre F_m sur les données pondérées par ω
   b. Calculer err_m = Σ ω_i I(ỹ_i ≠ F_m(x_i)) / Σ ω_i
   c. Calculer α_m = log[(1 - err_m) / err_m]
   d. Mettre à jour: ω_i ← ω_i exp[α_m I(ỹ_i ≠ F_m(x_i))]
3. Retourner f(x) = signe[Σ α_m F_m(x)]
```

## LogitBoost

### Motivation

AdaBoost augmente exponentiellement les poids des exemples mal classifiés, ce qui le rend **très sensible aux valeurs aberrantes** et au bruit.

**LogitBoost** utilise plutôt la perte logarithmique (entropie croisée):
$$
L_m(F) = \sum_{i=1}^N \log\left[1 + \exp\left(-2\tilde{y}_i(f_{m-1}(\boldsymbol{x}) + F(\boldsymbol{x}_i))\right)\right]
$$

Cette perte croît linéairement (au lieu d'exponentiellement) avec la marge négative, ce qui donne une meilleure robustesse aux outliers.

### Optimisation

Contrairement à AdaBoost, la mise à jour de LogitBoost n'a pas de forme analytique. On utilise une approximation par la **méthode de Newton** (Newton boosting).

## Gradient Boosting

### Idée générale

Le **gradient boosting** est une approche générique qui résout:
$$
\hat{f} = \arg\min_f \mathcal{L}(f)
$$

en effectuant une **descente de gradient dans l'espace des fonctions**.

### Représentation finie

Puisque les fonctions sont des objets de dimension infinie, nous les représentons par leurs valeurs sur l'ensemble d'entraînement:
$$
\boldsymbol{f} = (f(\boldsymbol{x}_1), \ldots, f(\boldsymbol{x}_N))
$$

Le gradient de $\mathcal{L}$ par rapport à $\boldsymbol{f}$ au point $f_{m-1}$ est:
$$
g_{im} = \left[\frac{\partial \ell(y_i, f(\boldsymbol{x}_i))}{\partial f(\boldsymbol{x}_i)}\right]_{f = f_{m-1}}
$$

### Descente de gradient fonctionnelle

La mise à jour serait:
$$
\boldsymbol{f}_m = \boldsymbol{f}_{m-1} - \beta_m \boldsymbol{g}_m
$$

avec:
$$
\beta_m = \arg\min_\beta \mathcal{L}(\boldsymbol{f}_{m-1} - \beta \boldsymbol{g}_m)
$$

**Problème**: cette mise à jour ne définit $f_m$ que sur les points d'entraînement. Elle ne généralise pas aux nouveaux exemples.

### Solution: approximer le gradient

Le gradient boosting résout ce problème en apprenant un modèle faible $F_m$ qui **approxime le gradient négatif**:
$$
F_m = \arg\min_F \sum_{i=1}^N (-g_{im} - F(\boldsymbol{x}_i))^2
$$

Ce modèle peut ensuite être évalué sur n'importe quel point $\boldsymbol{x}$.

### Algorithme général

```
1. Initialiser f_0(x) = argmin_F Σ ℓ(y_i, F(x_i))
2. Pour m = 1 à M:
   a. Calculer les résidus: r_im = -∂ℓ(y_i, f(x_i))/∂f(x_i)|_{f=f_{m-1}}
   b. Apprendre F_m sur les couples (x_i, r_im) par régression
   c. Mettre à jour: f_m(x) = f_{m-1}(x) + ν F_m(x)
3. Retourner f_M(x)
```

Le paramètre $0 < \nu \leq 1$ est le **taux d'apprentissage** (ou facteur de rétrécissement, *shrinkage*). Des valeurs petites ($\nu \approx 0.1$) améliorent la généralisation au prix de plus d'itérations.

## Gradient Tree Boosting

### Arbres comme modèles de base

Les arbres de décision sont des modèles de base naturels pour le gradient boosting:
$$
F_m(\boldsymbol{x}) = \sum_{j=1}^{J_m} w_{jm} \mathbb{I}(\boldsymbol{x} \in R_{jm})
$$

où $w_{jm}$ est la prédiction de la région $R_{jm}$.

### Procédure en deux étapes

1. **Trouver les régions**: Construire un arbre de régression sur les résidus $\{-g_{im}\}$ pour déterminer les régions $R_{jm}$

2. **Optimiser les poids**: Pour chaque feuille $j$, trouver le poids optimal:
$$
\hat{w}_{jm} = \arg\min_w \sum_{\boldsymbol{x}_i \in R_{jm}} \ell(y_i, f_{m-1}(\boldsymbol{x}_i) + w)
$$

Pour la perte quadratique, le poids optimal est simplement la moyenne des résidus dans la feuille.

### XGBoost

**XGBoost** (*eXtreme Gradient Boosting*) est une implémentation très populaire du gradient tree boosting. Ses innovations principales:

1. **Régularisation explicite**:
$$
\mathcal{L}_m(F_m) = \sum_{i=1}^N \ell(y_i, f_{m-1}(\boldsymbol{x}_i) + F_m(\boldsymbol{x}_i)) + \Omega(F_m)
$$

où $\Omega(F_m) = \gamma J + \frac{\lambda}{2} \sum_{j=1}^J w_j^2$ pénalise le nombre de feuilles et les poids.

2. **Approximation de second ordre**: Taylor de second ordre de la perte:
$$
\mathcal{L}_m \approx \sum_{i=1}^N \left[g_{im} F_m(\boldsymbol{x}_i) + \frac{1}{2} h_{im} F_m^2(\boldsymbol{x}_i)\right] + \Omega(F_m)
$$

où $h_{im}$ est la dérivée seconde (Hessienne).

3. **Critère de gain optimisé**:
$$
\text{gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma
$$

## Importance des caractéristiques

### Définition pour un arbre

Pour un arbre $T$, l'**importance de la caractéristique** $k$ est:
$$
R_k(T) = \sum_{j=1}^{J-1} G_j \mathbb{I}(v_j = k)
$$

où:
- La somme est sur tous les nœuds internes
- $G_j$ est le gain (réduction de coût) au nœud $j$
- $v_j = k$ si le nœud $j$ utilise la caractéristique $k$

### Pour un ensemble

Pour un ensemble de $M$ arbres, nous moyennons:
$$
R_k = \frac{1}{M} \sum_{m=1}^M R_k(T_m)
$$

Ces scores peuvent être normalisés dans l'intervalle $[0, 1]$.

Cette mesure d'importance permet d'interpréter les ensembles d'arbres, même si les arbres individuels ne sont plus interprétables.

## Comparaison: Bagging vs Boosting

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| Réduction | Variance | Biais |
| Entraînement | Parallèle | Séquentiel |
| Pondération | Uniforme | Adaptative |
| Modèles de base | Haute variance | Faible complexité |
| Surapprentissage | Résistant | Possible |
| Interprétation | Importance des variables | Importance des variables |

### Quand utiliser quoi?

- **Bagging/Forêts aléatoires**: données bruitées, prévention du surapprentissage
- **Boosting**: maximiser la performance prédictive, arbres peu profonds suffisants

## Résumé

Le boosting est une méthode puissante qui construit un ensemble **séquentiellement**:

- Chaque modèle corrige les **erreurs** des précédents
- **AdaBoost** utilise la perte exponentielle et des poids adaptatifs
- **Gradient boosting** généralise à n'importe quelle perte différentiable
- Les arbres de décision sont des modèles de base efficaces
- **XGBoost** ajoute régularisation et optimisations

Le boosting réduit principalement le **biais** (contrairement au bagging qui réduit la variance), ce qui en fait un complément naturel aux forêts aléatoires.

## Exercices

```{admonition} Exercice 1: Poids AdaBoost
:class: tip

Considérez AdaBoost avec 3 exemples initialement pondérés uniformément ($\omega_i = 1/3$). Le premier classificateur faible a une erreur pondérée de $\text{err}_1 = 0.25$.

1. Calculez $\alpha_1 = \log[(1 - \text{err}_1) / \text{err}_1]$.
2. Si l'exemple 1 est mal classifié et les exemples 2 et 3 sont bien classifiés, calculez les nouveaux poids $\omega_i$.
3. Normalisez les poids pour qu'ils somment à 1.
```

```{admonition} Exercice 2: Résidus
:class: tip

Pour le boosting des moindres carrés avec les données $\{(1, 2), (2, 4), (3, 3)\}$:

1. Si $f_0(x) = 3$ (constante), calculez les résidus $r_{i1}$.
2. Si $F_1(x) = 0.5x - 1$, calculez $f_1(x) = f_0(x) + F_1(x)$.
3. Calculez les nouveaux résidus $r_{i2}$.
```

```{admonition} Exercice 3: Gradient de la perte
:class: tip

Pour la perte logistique $\ell(y, f) = \log(1 + e^{-yf})$ où $y \in \{-1, +1\}$:

1. Calculez le gradient $\frac{\partial \ell}{\partial f}$.
2. Interprétez le signe du gradient en fonction de $y$ et $f$.
3. Que représentent les résidus $-g_{im}$ dans le gradient boosting avec cette perte?
```

```{admonition} Exercice 4: Régularisation
:class: tip

Dans XGBoost avec la régularisation $\Omega(F) = \gamma J + \frac{\lambda}{2} \sum_j w_j^2$:

1. Expliquez l'effet de $\gamma$ sur la structure de l'arbre.
2. Expliquez l'effet de $\lambda$ sur les prédictions.
3. Si $\gamma = 0$ et $\lambda = 0$, que devient le critère de gain?
```
