# Annexe: Révision d'optimisation

Cette annexe révise les concepts d'optimisation utilisés tout au long du livre.

## Formulation générale

Un problème d'optimisation s'écrit:
$$
\min_{\boldsymbol{x} \in \mathcal{X}} f(\boldsymbol{x})
$$

où:
- $f: \mathbb{R}^D \to \mathbb{R}$ est la **fonction objectif**
- $\mathcal{X} \subseteq \mathbb{R}^D$ est l'**ensemble réalisable**
- $\boldsymbol{x}^*$ est un **minimiseur** (ou optimum)

**Note**: Maximiser $f$ équivaut à minimiser $-f$.

## Convexité

### Ensembles convexes

Un ensemble $\mathcal{C}$ est **convexe** si pour tout $\boldsymbol{x}, \boldsymbol{y} \in \mathcal{C}$ et $\alpha \in [0, 1]$:
$$
\alpha \boldsymbol{x} + (1-\alpha)\boldsymbol{y} \in \mathcal{C}
$$

Intuition: le segment entre deux points de l'ensemble reste dans l'ensemble.

### Fonctions convexes

Une fonction $f$ est **convexe** si son domaine est convexe et:
$$
f(\alpha \boldsymbol{x} + (1-\alpha)\boldsymbol{y}) \leq \alpha f(\boldsymbol{x}) + (1-\alpha) f(\boldsymbol{y})
$$

pour tout $\alpha \in [0, 1]$.

**Interprétation géométrique**: la fonction est en-dessous de toute corde.

### Caractérisations

Pour une fonction $f$ deux fois différentiable:

| Condition | Signification |
|-----------|---------------|
| $f$ convexe | $\nabla^2 f(\boldsymbol{x}) \succeq 0$ (Hessienne semi-définie positive) |
| $f$ strictement convexe | $\nabla^2 f(\boldsymbol{x}) \succ 0$ (Hessienne définie positive) |
| $f$ concave | $-f$ convexe |

### Exemples de fonctions convexes

- Fonctions affines: $f(\boldsymbol{x}) = \boldsymbol{a}^\top \boldsymbol{x} + b$
- Normes: $f(\boldsymbol{x}) = \|\boldsymbol{x}\|_p$ pour $p \geq 1$
- Formes quadratiques: $f(\boldsymbol{x}) = \boldsymbol{x}^\top \mathbf{A} \boldsymbol{x}$ si $\mathbf{A} \succeq 0$
- Exponentielle: $f(x) = e^x$
- Entropie négative: $f(x) = x \log x$ pour $x > 0$

### Propriété fondamentale

Pour une fonction convexe, **tout minimum local est un minimum global**.

C'est pourquoi les problèmes convexes sont plus faciles à résoudre.

## Conditions d'optimalité

### Condition nécessaire du premier ordre

Si $\boldsymbol{x}^*$ est un minimum local et $f$ est différentiable:
$$
\nabla f(\boldsymbol{x}^*) = \mathbf{0}
$$

Un point où $\nabla f = \mathbf{0}$ est appelé **point stationnaire**.

### Condition nécessaire du second ordre

Si $\boldsymbol{x}^*$ est un minimum local et $f$ est deux fois différentiable:
$$
\nabla f(\boldsymbol{x}^*) = \mathbf{0} \quad \text{et} \quad \nabla^2 f(\boldsymbol{x}^*) \succeq 0
$$

### Condition suffisante du second ordre

Si $\nabla f(\boldsymbol{x}^*) = \mathbf{0}$ et $\nabla^2 f(\boldsymbol{x}^*) \succ 0$, alors $\boldsymbol{x}^*$ est un **minimum local strict**.

### Points-selle

Un **point-selle** satisfait $\nabla f = \mathbf{0}$ mais n'est ni un minimum ni un maximum:
- Minimum dans certaines directions
- Maximum dans d'autres directions

La Hessienne a des valeurs propres de signes différents.

## Descente de gradient

### Algorithme de base

Mise à jour itérative:
$$
\boldsymbol{x}^{(t+1)} = \boldsymbol{x}^{(t)} - \eta \nabla f(\boldsymbol{x}^{(t)})
$$

où $\eta > 0$ est le **taux d'apprentissage** (learning rate).

**Intuition**: on se déplace dans la direction de plus forte descente.

### Choix du taux d'apprentissage

- $\eta$ trop petit: convergence lente
- $\eta$ trop grand: oscillations, divergence possible

**Règle pratique**: commencer avec $\eta$ modéré, observer la perte.

### Convergence

Pour une fonction $L$-lisse ($\|\nabla f(\boldsymbol{x}) - \nabla f(\boldsymbol{y})\| \leq L\|\boldsymbol{x} - \boldsymbol{y}\|$) et convexe:
- Avec $\eta \leq 1/L$: convergence en $O(1/T)$ itérations
- Si fortement convexe: convergence linéaire en $O(\kappa \log(1/\epsilon))$

où $\kappa$ est le **conditionnement** (rapport des valeurs propres extrêmes de la Hessienne).

## Descente de gradient stochastique (SGD)

### Motivation

Pour une fonction de la forme:
$$
f(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^N f_i(\boldsymbol{\theta})
$$

Le gradient exact coûte $O(N)$ par itération.

### Algorithme SGD

À chaque itération, on utilise un **estimateur non biaisé** du gradient:
$$
\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \eta_t \nabla f_i(\boldsymbol{\theta}^{(t)})
$$

où $i$ est tiré uniformément au hasard.

**Mini-batch SGD**: moyenne sur $B$ exemples:
$$
\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \frac{\eta_t}{B} \sum_{i \in \mathcal{B}} \nabla f_i(\boldsymbol{\theta}^{(t)})
$$

### Variance et convergence

SGD introduit du bruit (variance). Pour converger:
- Le taux d'apprentissage doit décroître: $\sum_t \eta_t = \infty$ et $\sum_t \eta_t^2 < \infty$
- Exemple: $\eta_t = \eta_0 / \sqrt{t}$

## Méthodes avec momentum

### Momentum classique

$$
\boldsymbol{v}^{(t+1)} = \beta \boldsymbol{v}^{(t)} + \nabla f(\boldsymbol{x}^{(t)})
$$
$$
\boldsymbol{x}^{(t+1)} = \boldsymbol{x}^{(t)} - \eta \boldsymbol{v}^{(t+1)}
$$

où $\beta \in [0, 1)$ est le coefficient de momentum (typiquement 0.9).

**Avantages**:
- Accélère la convergence dans les directions cohérentes
- Amortit les oscillations

### Momentum de Nesterov

$$
\boldsymbol{v}^{(t+1)} = \beta \boldsymbol{v}^{(t)} + \nabla f(\boldsymbol{x}^{(t)} - \eta \beta \boldsymbol{v}^{(t)})
$$
$$
\boldsymbol{x}^{(t+1)} = \boldsymbol{x}^{(t)} - \eta \boldsymbol{v}^{(t+1)}
$$

**Idée**: on regarde le gradient "en avance".

## Méthodes adaptatives

### AdaGrad

Adapte le taux d'apprentissage par coordonnée:
$$
g_i^{(t)} = \nabla_i f(\boldsymbol{x}^{(t)})
$$
$$
G_{ii}^{(t)} = G_{ii}^{(t-1)} + (g_i^{(t)})^2
$$
$$
x_i^{(t+1)} = x_i^{(t)} - \frac{\eta}{\sqrt{G_{ii}^{(t)} + \epsilon}} g_i^{(t)}
$$

**Problème**: le taux effectif décroît trop vite.

### RMSProp

Utilise une moyenne mobile exponentielle:
$$
v_i^{(t)} = \gamma v_i^{(t-1)} + (1-\gamma)(g_i^{(t)})^2
$$
$$
x_i^{(t+1)} = x_i^{(t)} - \frac{\eta}{\sqrt{v_i^{(t)} + \epsilon}} g_i^{(t)}
$$

### Adam

Combine momentum et adaptation:
$$
m^{(t)} = \beta_1 m^{(t-1)} + (1-\beta_1) g^{(t)} \quad \text{(premier moment)}
$$
$$
v^{(t)} = \beta_2 v^{(t-1)} + (1-\beta_2) (g^{(t)})^2 \quad \text{(second moment)}
$$
$$
\hat{m}^{(t)} = \frac{m^{(t)}}{1-\beta_1^t}, \quad \hat{v}^{(t)} = \frac{v^{(t)}}{1-\beta_2^t} \quad \text{(correction de biais)}
$$
$$
x^{(t+1)} = x^{(t)} - \frac{\eta}{\sqrt{\hat{v}^{(t)}} + \epsilon} \hat{m}^{(t)}
$$

**Hyperparamètres par défaut**: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

## Optimisation sous contraintes

### Formulation

$$
\min_{\boldsymbol{x}} f(\boldsymbol{x}) \quad \text{s.t.} \quad g_i(\boldsymbol{x}) \leq 0, \quad h_j(\boldsymbol{x}) = 0
$$

où:
- $g_i$: contraintes d'inégalité
- $h_j$: contraintes d'égalité

### Multiplicateurs de Lagrange (contraintes d'égalité)

Pour le problème $\min f(\boldsymbol{x})$ s.t. $h(\boldsymbol{x}) = 0$:

Le **Lagrangien**:
$$
\mathcal{L}(\boldsymbol{x}, \boldsymbol{\lambda}) = f(\boldsymbol{x}) + \boldsymbol{\lambda}^\top h(\boldsymbol{x})
$$

Conditions d'optimalité:
$$
\nabla_{\boldsymbol{x}} \mathcal{L} = \nabla f(\boldsymbol{x}) + \sum_j \lambda_j \nabla h_j(\boldsymbol{x}) = \mathbf{0}
$$
$$
\nabla_{\boldsymbol{\lambda}} \mathcal{L} = h(\boldsymbol{x}) = \mathbf{0}
$$

**Interprétation**: au point optimal, $\nabla f$ est une combinaison linéaire des $\nabla h_j$.

### Conditions KKT (Karush-Kuhn-Tucker)

Pour le problème général avec inégalités:

Le Lagrangien:
$$
\mathcal{L}(\boldsymbol{x}, \boldsymbol{\mu}, \boldsymbol{\lambda}) = f(\boldsymbol{x}) + \sum_i \mu_i g_i(\boldsymbol{x}) + \sum_j \lambda_j h_j(\boldsymbol{x})
$$

**Conditions KKT** (nécessaires pour l'optimalité):

1. **Stationnarité**: $\nabla_{\boldsymbol{x}} \mathcal{L} = \mathbf{0}$
2. **Réalisabilité primale**: $g_i(\boldsymbol{x}^*) \leq 0$, $h_j(\boldsymbol{x}^*) = 0$
3. **Réalisabilité duale**: $\mu_i \geq 0$
4. **Complémentarité**: $\mu_i g_i(\boldsymbol{x}^*) = 0$

La condition de complémentarité signifie:
- Si $g_i(\boldsymbol{x}^*) < 0$ (contrainte inactive): $\mu_i = 0$
- Si $\mu_i > 0$: $g_i(\boldsymbol{x}^*) = 0$ (contrainte active)

### Dualité

Le **problème primal**:
$$
p^* = \min_{\boldsymbol{x}} \max_{\boldsymbol{\mu} \geq 0, \boldsymbol{\lambda}} \mathcal{L}(\boldsymbol{x}, \boldsymbol{\mu}, \boldsymbol{\lambda})
$$

Le **problème dual**:
$$
d^* = \max_{\boldsymbol{\mu} \geq 0, \boldsymbol{\lambda}} \min_{\boldsymbol{x}} \mathcal{L}(\boldsymbol{x}, \boldsymbol{\mu}, \boldsymbol{\lambda})
$$

**Dualité faible**: $d^* \leq p^*$ (toujours vraie)

**Dualité forte**: $d^* = p^*$ (vraie pour les problèmes convexes satisfaisant une condition de régularité)

### Application aux SVM

Le problème SVM à marge souple:
$$
\min_{\boldsymbol{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\boldsymbol{w}\|^2 + C\sum_i \xi_i
$$
$$
\text{s.t.} \quad y_i(\boldsymbol{w}^\top \boldsymbol{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

Par la dualité de Lagrange, on obtient le problème dual (plus facile à résoudre avec l'astuce du noyau).

## Méthodes de Newton

### Newton classique

Utilise l'information de second ordre:
$$
\boldsymbol{x}^{(t+1)} = \boldsymbol{x}^{(t)} - [\nabla^2 f(\boldsymbol{x}^{(t)})]^{-1} \nabla f(\boldsymbol{x}^{(t)})
$$

**Avantages**:
- Convergence quadratique près de l'optimum
- Invariant aux transformations affines

**Inconvénients**:
- Coût $O(D^3)$ pour l'inversion
- Peut diverger si la Hessienne n'est pas définie positive

### Quasi-Newton (BFGS, L-BFGS)

Approxime la Hessienne sans la calculer explicitement.

**L-BFGS** (Limited-memory BFGS):
- Stocke seulement les $m$ dernières différences de gradient
- Coût $O(mD)$ par itération
- Très utilisé pour l'optimisation de grande dimension

## Résumé des méthodes

| Méthode | Coût/itération | Convergence | Usage |
|---------|----------------|-------------|-------|
| Gradient | $O(D)$ | Sous-linéaire | Grande dimension |
| SGD | $O(D)$ | Bruitée | Très grands datasets |
| Adam | $O(D)$ | Adaptative | Deep learning |
| Newton | $O(D^3)$ | Quadratique | Petite dimension |
| L-BFGS | $O(mD)$ | Super-linéaire | Dimension moyenne |
