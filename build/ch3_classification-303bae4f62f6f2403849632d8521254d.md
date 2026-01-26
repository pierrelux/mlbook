---
kernelspec:
  name: python3
  display_name: Python 3
---

# Classification linéaire

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Formuler la régression logistique comme un modèle probabiliste
- Dériver l'entropie croisée à partir du maximum de vraisemblance
- Calculer le gradient et comprendre la convexité de l'objectif
- Généraliser au cas multiclasse avec la fonction softmax
- Appliquer la descente de gradient stochastique pour l'optimisation
- Interpréter géométriquement la frontière de décision linéaire
```

Le [chapitre précédent](ch2_linear_regression.md) a traité la régression: prédire une valeur continue $y \in \mathbb{R}$ à partir d'entrées $\mathbf{x}$. Ce chapitre aborde le problème de **classification**: prédire une catégorie $y \in \{0, 1, \ldots, C-1\}$ parmi un ensemble fini de classes.

La régression logistique est le pendant de la régression linéaire pour la classification. Malgré son nom, c'est bien un modèle de classification, pas de régression. Comme pour les moindres carrés, nous cherchons une combinaison linéaire des entrées, mais nous transformons cette combinaison pour produire des probabilités de classe.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Configuration pour des figures haute résolution
%config InlineBackend.figure_format = 'retina'
```

## La régression logistique binaire

### Du score à la probabilité

En régression, le modèle produit directement une prédiction $\hat{y} = \boldsymbol{\theta}^\top \mathbf{x}$. En classification binaire, nous voulons plutôt une probabilité: quelle est la chance que l'exemple appartienne à la classe positive?

Une première idée serait de prendre $p(y=1|\mathbf{x}) = \boldsymbol{\theta}^\top \mathbf{x}$, mais cela pose problème: une combinaison linéaire peut produire n'importe quelle valeur réelle, pas seulement une probabilité entre 0 et 1.

La solution est d'appliquer une fonction de transformation. La **fonction sigmoïde** (ou fonction logistique) transforme n'importe quel nombre réel en une valeur dans l'intervalle $(0, 1)$:

$$
\sigma(a) = \frac{1}{1 + e^{-a}}
$$

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(-6, 6, 200)
sigma = 1 / (1 + np.exp(-a))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(a, sigma, 'C0-', linewidth=2)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

ax.set_xlabel('$a = \\boldsymbol{\\theta}^\\top \\mathbf{x}$')
ax.set_ylabel('$\\sigma(a) = P(y=1|\\mathbf{x})$')
ax.set_title('La fonction sigmoïde transforme un score en probabilité')
ax.set_xlim(-6, 6)
ax.set_ylim(-0.05, 1.05)

# Annotations
ax.annotate('$a < 0$: classe 0 plus probable', xy=(-4, 0.1), fontsize=10, color='C1')
ax.annotate('$a > 0$: classe 1 plus probable', xy=(1.5, 0.9), fontsize=10, color='C0')
ax.annotate('$a = 0$: équiprobable', xy=(0.2, 0.55), fontsize=10, color='gray')

plt.tight_layout()
```

### Le modèle probabiliste

La **régression logistique binaire** modélise la probabilité de la classe positive comme:

$$
p(y = 1 | \mathbf{x}; \boldsymbol{\theta}) = \sigma(\boldsymbol{\theta}^\top \mathbf{x}) = \frac{1}{1 + e^{-\boldsymbol{\theta}^\top \mathbf{x}}}
$$

où $\boldsymbol{\theta}$ inclut le biais (en ajoutant un 1 aux entrées). La probabilité de la classe négative est simplement le complément:

$$
p(y = 0 | \mathbf{x}; \boldsymbol{\theta}) = 1 - \sigma(\boldsymbol{\theta}^\top \mathbf{x}) = \sigma(-\boldsymbol{\theta}^\top \mathbf{x})
$$

En notation compacte, la distribution sur $y \in \{0, 1\}$ est une **distribution de Bernoulli**:

$$
p(y | \mathbf{x}; \boldsymbol{\theta}) = \mu^y (1 - \mu)^{1-y}
$$

où $\mu = \sigma(\boldsymbol{\theta}^\top \mathbf{x})$ est la probabilité prédite pour la classe positive.

### Le logit et le rapport de cotes

La quantité $a = \boldsymbol{\theta}^\top \mathbf{x}$ avant la transformation sigmoïde s'appelle le **logit**. Pourquoi ce nom? En inversant la relation $\mu = \sigma(a)$, on obtient:

$$
a = \sigma^{-1}(\mu) = \log \frac{\mu}{1 - \mu} = \log \frac{p(y = 1 | \mathbf{x})}{p(y = 0 | \mathbf{x})}
$$

Cette quantité est le logarithme du **rapport de cotes** (odds ratio en anglais). La **cote** (odds) est le rapport entre la probabilité de succès et la probabilité d'échec: si $\mu = 0.75$, la cote est $0.75/0.25 = 3$, ce qui signifie que le succès est trois fois plus probable que l'échec.

Si $a = 0$, les deux classes sont équiprobables ($\mu = 0.5$). Si $a > 0$, la classe 1 est plus probable; si $a < 0$, c'est la classe 0. La valeur absolue de $a$ mesure la confiance: plus $|a|$ est grand, plus le modèle est certain de sa prédiction.

### Interprétation des coefficients

Cette formulation via le logit a une conséquence pratique importante: elle permet d'interpréter directement les coefficients $\theta_j$.

Puisque $a = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots$, une augmentation de $x_j$ d'une unité (toutes les autres variables restant constantes) augmente le logit de $\theta_j$:

$$
\Delta a = \theta_j \cdot \Delta x_j
$$

En termes de cotes, si nous notons $\text{cote} = \frac{\mu}{1-\mu} = e^a$, alors:

$$
\frac{\text{cote après}}{\text{cote avant}} = e^{\theta_j}
$$

Le coefficient $\theta_j$ représente donc le **changement dans le log-odds** pour une augmentation unitaire de $x_j$, et $e^{\theta_j}$ est le **facteur multiplicatif sur la cote**.

```{admonition} Exemple: prédiction d'admission
:class: tip

Supposons un modèle de régression logistique prédisant l'admission à un programme, avec:
- $\theta_{\text{GPA}} = 1.5$ (moyenne cumulative)
- $\theta_{\text{experience}} = 0.3$ (années d'expérience)

**Interprétation de $\theta_{\text{GPA}} = 1.5$:**
- $e^{1.5} \approx 4.5$
- Chaque point de GPA supplémentaire multiplie les chances d'admission par 4.5

**Interprétation de $\theta_{\text{experience}} = 0.3$:**
- $e^{0.3} \approx 1.35$
- Chaque année d'expérience supplémentaire augmente les chances de 35%

Cette interprétation est largement utilisée en médecine, sciences sociales et marketing pour quantifier l'effet de chaque variable.
```

### La frontière de décision

Pour classifier, nous prédisons la classe la plus probable:

$$
\hat{y} = \begin{cases} 1 & \text{si } p(y=1|\mathbf{x}) > 0.5 \\ 0 & \text{sinon} \end{cases} = \mathbb{1}(\boldsymbol{\theta}^\top \mathbf{x} > 0)
$$

La **frontière de décision** est l'ensemble des points où les deux classes sont équiprobables:

$$
\{\mathbf{x} : \boldsymbol{\theta}^\top \mathbf{x} = 0\}
$$

C'est un **hyperplan** dans l'espace des entrées. En 2D, c'est une droite; en 3D, un plan. Le vecteur $\boldsymbol{\theta}$ est perpendiculaire à cet hyperplan et pointe vers la région de classe 1.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Générer des données linéairement séparables
np.random.seed(42)
n_per_class = 50

# Classe 0: centrée en (-1, -1)
X0 = np.random.randn(n_per_class, 2) * 0.8 + np.array([-1, -1])
# Classe 1: centrée en (1, 1)
X1 = np.random.randn(n_per_class, 2) * 0.8 + np.array([1, 1])

X = np.vstack([X0, X1])
y = np.array([0] * n_per_class + [1] * n_per_class)

# Ajustement simple (pour illustration)
# Supposons theta = [0, 1, 1] (avec biais)
theta = np.array([1, 1])  # sans biais pour simplifier

fig, ax = plt.subplots(figsize=(8, 6))

# Points
ax.scatter(X0[:, 0], X0[:, 1], c='C1', label='Classe 0', alpha=0.7, edgecolor='white')
ax.scatter(X1[:, 0], X1[:, 1], c='C0', label='Classe 1', alpha=0.7, edgecolor='white')

# Frontière de décision: theta[0]*x1 + theta[1]*x2 = 0 => x2 = -x1
x_line = np.linspace(-3, 3, 100)
y_line = -x_line  # Pour theta = [1, 1], frontière est x1 + x2 = 0
ax.plot(x_line, y_line, 'k-', linewidth=2, label='Frontière de décision')

# Vecteur normal (theta)
ax.arrow(0, 0, 0.8, 0.8, head_width=0.15, head_length=0.1, fc='green', ec='green')
ax.text(0.9, 0.6, '$\\boldsymbol{\\theta}$', fontsize=14, color='green')

# Régions
ax.fill_between(x_line, y_line, 3, alpha=0.1, color='C0')
ax.fill_between(x_line, y_line, -3, alpha=0.1, color='C1')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Frontière de décision linéaire')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.legend()

plt.tight_layout()
```

Les données sont dites **linéairement séparables** si une frontière linéaire peut les séparer parfaitement. Dans ce cas, il existe des paramètres $\boldsymbol{\theta}$ tels que toutes les prédictions sont correctes. En pratique, les données réelles sont rarement parfaitement séparables.

## Estimation par maximum de vraisemblance

### Construction de la vraisemblance

Comment trouver les paramètres $\boldsymbol{\theta}$? Le [chapitre 1](ch1_learning_problem.md) a introduit le principe du maximum de vraisemblance: choisir les paramètres qui rendent les données observées les plus probables. Nous avons vu au [chapitre 2](ch2_linear_regression.md) que ce principe, appliqué à un modèle gaussien, donne les moindres carrés. Appliquons maintenant ce même principe à la classification.

Sous l'hypothèse i.i.d. (rappel: les exemples sont tirés indépendamment de la même distribution), la vraisemblance est le produit des probabilités pour chaque exemple:

$$
\mathcal{L}(\boldsymbol{\theta}; \mathcal{D}) = \prod_{i=1}^N p(y_i | \mathbf{x}_i; \boldsymbol{\theta}) = \prod_{i=1}^N \mu_i^{y_i} (1 - \mu_i)^{1-y_i}
$$

où $\mu_i = \sigma(\boldsymbol{\theta}^\top \mathbf{x}_i)$.

### La log-vraisemblance négative est l'entropie croisée

Pour l'optimisation, nous travaillons avec la log-vraisemblance négative (LVN):

$$
\text{LVN}(\boldsymbol{\theta}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log \mu_i + (1 - y_i) \log(1 - \mu_i) \right]
$$

Cette expression porte un nom: c'est l'**entropie croisée binaire**. Pour un seul exemple:

$$
\mathbb{H}_{\text{ce}}(y, \mu) = -\left[ y \log \mu + (1-y) \log(1-\mu) \right]
$$

Ainsi, minimiser la LVN revient à minimiser l'entropie croisée moyenne sur les données d'entraînement:

$$
\text{LVN}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^N \mathbb{H}_{\text{ce}}(y_i, \mu_i)
$$

C'est un résultat fondamental: **l'entropie croisée n'est pas un choix arbitraire de fonction de perte**. Elle découle directement du maximum de vraisemblance appliqué au modèle de Bernoulli. Nous retrouvons le même schéma qu'au [chapitre 2](ch2_linear_regression.md): le choix du modèle probabiliste (gaussien ou Bernoulli) détermine la fonction de perte optimale (quadratique ou entropie croisée).

### Calcul du gradient

Le gradient de la LVN a une forme remarquablement simple:

$$
\nabla_{\boldsymbol{\theta}} \text{LVN}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^N (\mu_i - y_i) \mathbf{x}_i
$$

```{admonition} Dérivation du gradient
:class: tip dropdown

Pour un exemple $(\mathbf{x}_i, y_i)$, la perte est:

$$
\ell_i = -y_i \log \mu_i - (1 - y_i) \log(1 - \mu_i)
$$

La dérivée de la sigmoïde a une propriété utile: $\frac{d\sigma}{da} = \sigma(a)(1 - \sigma(a)) = \mu(1-\mu)$.

En appliquant la règle de la chaîne:

$$
\frac{\partial \ell_i}{\partial \boldsymbol{\theta}} = \left( -\frac{y_i}{\mu_i} + \frac{1 - y_i}{1 - \mu_i} \right) \cdot \mu_i(1 - \mu_i) \cdot \mathbf{x}_i
$$

Simplifions le terme entre parenthèses multiplié par $\mu_i(1-\mu_i)$:

$$
\left( -\frac{y_i}{\mu_i} + \frac{1 - y_i}{1 - \mu_i} \right) \mu_i(1 - \mu_i) = -y_i(1-\mu_i) + (1-y_i)\mu_i = \mu_i - y_i
$$

D'où le gradient: $\nabla_{\boldsymbol{\theta}} \ell_i = (\mu_i - y_i) \mathbf{x}_i$.
```

Le gradient est une moyenne pondérée des entrées, où le poids est **l'erreur de prédiction** $e_i = \mu_i - y_i$:
- Si $y_i = 1$ et $\mu_i$ est proche de 1, l'erreur est faible et la contribution au gradient est petite.
- Si $y_i = 1$ et $\mu_i$ est proche de 0, l'erreur est grande (-1) et le gradient pousse $\boldsymbol{\theta}$ dans la direction de $\mathbf{x}_i$.

Cette forme est analogue au gradient des moindres carrés: $\nabla_{\boldsymbol{\theta}} \text{RSS} = \mathbf{X}^\top(\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$.

### Convexité de l'objectif

La LVN de la régression logistique est **strictement convexe**. La matrice hessienne est:

$$
\mathbf{H}(\boldsymbol{\theta}) = \frac{1}{N} \mathbf{X}^\top \mathbf{S} \mathbf{X}
$$

où $\mathbf{S} = \text{diag}(\mu_1(1-\mu_1), \ldots, \mu_N(1-\mu_N))$ est une matrice diagonale avec des éléments positifs (car $\mu_i \in (0, 1)$).

Pour tout vecteur $\mathbf{v} \neq \mathbf{0}$:

$$
\mathbf{v}^\top \mathbf{X}^\top \mathbf{S} \mathbf{X} \mathbf{v} = \|\mathbf{S}^{1/2} \mathbf{X} \mathbf{v}\|_2^2 \geq 0
$$

La hessienne est donc semi-définie positive (définie positive si $\mathbf{X}$ est de rang plein). La LVN admet un unique minimum global, ce qui garantit que l'optimisation convergera vers la même solution peu importe l'initialisation.

## Régression logistique multiclasse

### La fonction softmax

Pour la classification à $C$ classes, nous généralisons la sigmoïde par la fonction **softmax**:

$$
p(y = c | \mathbf{x}; \boldsymbol{\Theta}) = \text{softmax}(\mathbf{a})_c = \frac{e^{a_c}}{\sum_{c'=1}^{C} e^{a_{c'}}}
$$

où $\mathbf{a} = \boldsymbol{\Theta} \mathbf{x}$ est le vecteur de **logits** (un score par classe), et $\boldsymbol{\Theta} \in \mathbb{R}^{C \times d}$ est la matrice de paramètres.

La fonction softmax a deux propriétés essentielles:
1. Les sorties sont positives: $\text{softmax}(\mathbf{a})_c > 0$ pour tout $c$
2. Les sorties somment à 1: $\sum_{c=1}^C \text{softmax}(\mathbf{a})_c = 1$

C'est donc une distribution de probabilité valide sur les $C$ classes.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

def softmax(a):
    a_max = np.max(a)
    exp_a = np.exp(a - a_max)
    return exp_a / np.sum(exp_a)

# Exemple avec 3 classes
logits = np.array([2.0, 1.0, 0.5])
probs = softmax(logits)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Logits
ax = axes[0]
bars = ax.bar(['Classe 0', 'Classe 1', 'Classe 2'], logits, color=['C0', 'C1', 'C2'])
ax.set_ylabel('Logit $a_c$')
ax.set_title('Scores bruts (logits)')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

# Probabilities
ax = axes[1]
bars = ax.bar(['Classe 0', 'Classe 1', 'Classe 2'], probs, color=['C0', 'C1', 'C2'])
ax.set_ylabel('Probabilité $p(y=c|\\mathbf{x})$')
ax.set_title('Après softmax: distribution de probabilité')
ax.set_ylim(0, 1)

# Add probability values on bars
for bar, p in zip(bars, probs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{p:.2f}', ha='center', fontsize=11)

plt.tight_layout()
```

### Entropie croisée catégorielle

La log-vraisemblance négative pour le cas multiclasse utilise l'**entropie croisée catégorielle**:

$$
\text{LVN}(\boldsymbol{\Theta}) = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^{C} y_{ic} \log \mu_{ic}
$$

où $y_{ic} = \mathbb{1}(y_i = c)$ est l'**encodage one-hot** de l'étiquette (1 pour la vraie classe, 0 ailleurs) et $\mu_{ic} = p(y = c | \mathbf{x}_i)$.

Puisque $y_{ic}$ est nul sauf pour la vraie classe, cette somme se simplifie:

$$
\text{LVN}(\boldsymbol{\Theta}) = -\frac{1}{N} \sum_{i=1}^N \log p(y = y_i | \mathbf{x}_i)
$$

C'est simplement la moyenne du logarithme négatif de la probabilité assignée à la vraie classe.

### Cas binaire comme cas particulier

Pour $C = 2$ classes, la fonction softmax se réduit à la sigmoïde. En effet, si $\mathbf{a} = (a_0, a_1)$:

$$
\text{softmax}(\mathbf{a})_1 = \frac{e^{a_1}}{e^{a_0} + e^{a_1}} = \frac{1}{1 + e^{-(a_1 - a_0)}} = \sigma(a_1 - a_0)
$$

En posant $a = a_1 - a_0$ (la différence de scores), on retrouve exactement le modèle binaire.

## Optimisation par descente de gradient

Au chapitre 2, nous avons résolu la régression linéaire en posant le gradient égal à zéro et en obtenant une formule explicite pour $\hat{\boldsymbol{\theta}}$. Cette approche fonctionnait parce que le gradient de la somme des carrés est linéaire en $\boldsymbol{\theta}$, ce qui donne un système d'équations linéaires.

Pour la régression logistique, le gradient de la LVN contient des termes $\mu_i = \sigma(\boldsymbol{\theta}^\top \mathbf{x}_i)$ qui dépendent de $\boldsymbol{\theta}$ de manière non linéaire. Poser le gradient égal à zéro ne donne pas un système que nous pouvons résoudre algébriquement. Nous devons donc trouver le minimum par une autre méthode.

### L'idée de la descente de gradient

Imaginons que nous cherchions le point le plus bas d'un terrain vallonné, mais dans le brouillard: nous ne voyons que notre voisinage immédiat. Une stratégie naturelle serait de regarder dans quelle direction le terrain descend le plus rapidement, puis de faire un pas dans cette direction. En répétant ce processus, nous devrions nous rapprocher progressivement du fond de la vallée.

Le gradient $\nabla_{\boldsymbol{\theta}} \text{LVN}(\boldsymbol{\theta})$ indique précisément la direction de plus forte montée de la fonction. Pour descendre, nous allons donc dans la direction opposée. La **descente de gradient** formalise cette idée: à chaque itération, nous mettons à jour les paramètres selon

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla_{\boldsymbol{\theta}} \text{LVN}(\boldsymbol{\theta}_t)
$$

où $\eta > 0$ est le **taux d'apprentissage**, qui contrôle la longueur du pas. Un pas trop petit rend la progression lente; un pas trop grand peut faire osciller autour du minimum ou même diverger.

Pour la régression logistique, nous avons calculé que le gradient est $\frac{1}{N} \sum_{i=1}^N (\mu_i - y_i) \mathbf{x}_i$. La mise à jour devient donc:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{N} \sum_{i=1}^N (\mu_i - y_i) \mathbf{x}_i
$$

Chaque itération ajuste $\boldsymbol{\theta}$ dans une direction qui réduit l'erreur moyenne. Les exemples mal classés (où $\mu_i$ est loin de $y_i$) contribuent davantage à la mise à jour.

### Passage à l'échelle: la descente de gradient stochastique

Calculer le gradient exact requiert de parcourir tous les $N$ exemples à chaque itération. Pour un jeu de données de millions d'exemples, cela devient prohibitif. La **descente de gradient stochastique** (SGD) contourne ce problème en utilisant une estimation du gradient basée sur un sous-ensemble des données.

À chaque itération, nous échantillonnons un **mini-lot** $\mathcal{B}_t$ de quelques dizaines d'exemples et calculons le gradient sur ce sous-ensemble uniquement:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} (\mu_i - y_i) \mathbf{x}_i
$$

Cette estimation est non biaisée: en espérance sur le choix du mini-lot, elle égale le vrai gradient. La variance de l'estimation diminue avec la taille du mini-lot, mais même avec quelques dizaines d'exemples, l'estimation est suffisamment précise pour progresser vers le minimum.

La SGD présente plusieurs avantages pratiques. Chaque itération est beaucoup plus rapide puisqu'elle ne traite qu'une fraction des données. De plus, le bruit introduit par l'échantillonnage peut aider l'algorithme à s'échapper de minima locaux peu profonds. Enfin, la SGD permet de traiter des données qui arrivent en flux continu, sans avoir besoin de stocker tout le jeu de données en mémoire.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Générer des données
np.random.seed(42)
n = 200
X = np.random.randn(n, 2)
X = np.hstack([np.ones((n, 1)), X])  # Ajouter biais
true_theta = np.array([0, 2, -1])
probs = 1 / (1 + np.exp(-X @ true_theta))
y = (np.random.rand(n) < probs).astype(float)

def sigmoid(a):
    return np.where(a >= 0, 1 / (1 + np.exp(-a)), np.exp(a) / (1 + np.exp(a)))

def compute_loss(theta, X, y):
    mu = sigmoid(X @ theta)
    mu = np.clip(mu, 1e-10, 1 - 1e-10)
    return -np.mean(y * np.log(mu) + (1 - y) * np.log(1 - mu))

def compute_gradient(theta, X, y):
    mu = sigmoid(X @ theta)
    return X.T @ (mu - y) / len(y)

# Descente de gradient complète
theta_gd = np.zeros(3)
losses_gd = [compute_loss(theta_gd, X, y)]
for _ in range(100):
    grad = compute_gradient(theta_gd, X, y)
    theta_gd = theta_gd - 0.5 * grad
    losses_gd.append(compute_loss(theta_gd, X, y))

# SGD avec mini-lots
theta_sgd = np.zeros(3)
losses_sgd = [compute_loss(theta_sgd, X, y)]
batch_size = 32
for epoch in range(100):
    indices = np.random.permutation(n)
    for start in range(0, n, batch_size):
        batch_idx = indices[start:start+batch_size]
        grad = compute_gradient(theta_sgd, X[batch_idx], y[batch_idx])
        theta_sgd = theta_sgd - 0.5 * grad
    losses_sgd.append(compute_loss(theta_sgd, X, y))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(losses_gd, 'C0-', linewidth=2, label='Gradient complet')
ax.plot(losses_sgd, 'C1-', linewidth=2, alpha=0.7, label=f'SGD (mini-lots de {batch_size})')
ax.set_xlabel('Époque')
ax.set_ylabel('Entropie croisée')
ax.set_title('Convergence de la descente de gradient')
ax.legend()
ax.set_xlim(0, 100)

plt.tight_layout()
```

### Choix du taux d'apprentissage

Le taux d'apprentissage $\eta$ contrôle la vitesse de convergence et requiert un réglage soigneux. Un taux trop petit rend la progression très lente: l'algorithme fait de minuscules pas et peut nécessiter des milliers d'itérations pour converger. Un taux trop grand cause des oscillations autour du minimum, voire une divergence si les mises à jour dépassent systématiquement le minimum.

Une stratégie pratique consiste à commencer avec un taux modéré puis à le réduire au cours de l'entraînement. Au début, de grands pas permettent de progresser rapidement vers la région du minimum; ensuite, de petits pas permettent de s'en approcher avec précision. La décroissance en racine carrée $\eta_t = \eta_0 / \sqrt{t+1}$ est un choix classique qui garantit la convergence théorique tout en conservant une vitesse raisonnable.

## Implémentation

Traduisons maintenant ces idées en code. L'implémentation ci-dessous suit fidèlement les équations que nous avons dérivées: la fonction `sigmoid` calcule $\sigma(a)$, le gradient est $\mathbf{X}^\top(\boldsymbol{\mu} - \mathbf{y}) / N$, et la mise à jour soustrait ce gradient multiplié par le taux d'apprentissage.

Un détail pratique mérite attention: le calcul de la sigmoïde. Pour de grandes valeurs négatives de $a$, l'exponentielle $e^{-a}$ peut déborder numériquement. La formulation $\sigma(a) = e^a / (1 + e^a)$ est équivalente mathématiquement mais numériquement stable pour $a < 0$. Notre implémentation utilise l'une ou l'autre forme selon le signe de $a$.

```{code-cell} python
:tags: [hide-input]

import numpy as np

def sigmoid(a):
    """Fonction sigmoïde numériquement stable."""
    return np.where(a >= 0,
                    1 / (1 + np.exp(-a)),
                    np.exp(a) / (1 + np.exp(a)))

def logistic_regression_sgd(X, y, lr=0.1, n_epochs=100, batch_size=32):
    """Régression logistique par descente de gradient stochastique.
    
    Args:
        X: Matrice de design (N x d), inclut le biais si nécessaire
        y: Étiquettes binaires (N,)
        lr: Taux d'apprentissage
        n_epochs: Nombre d'époques
        batch_size: Taille des mini-lots
        
    Returns:
        theta: Vecteur de paramètres (d,)
        losses: Historique des pertes
    """
    N, d = X.shape
    theta = np.zeros(d)
    losses = []
    
    for epoch in range(n_epochs):
        # Mélanger les données à chaque époque
        indices = np.random.permutation(N)
        
        for start in range(0, N, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            
            # Prédictions
            mu = sigmoid(X_batch @ theta)
            
            # Gradient
            grad = X_batch.T @ (mu - y_batch) / len(batch_idx)
            
            # Mise à jour
            theta = theta - lr * grad
        
        # Calculer la perte après chaque époque
        mu_all = sigmoid(X @ theta)
        mu_all = np.clip(mu_all, 1e-10, 1 - 1e-10)
        loss = -np.mean(y * np.log(mu_all) + (1 - y) * np.log(1 - mu_all))
        losses.append(loss)
    
    return theta, losses

def predict_proba(X, theta):
    """Probabilités de la classe positive."""
    return sigmoid(X @ theta)

def predict(X, theta, threshold=0.5):
    """Prédictions binaires."""
    return (predict_proba(X, theta) >= threshold).astype(int)
```

Appliquons cette implémentation à un problème synthétique. Nous générons deux groupes de points en 2D, chacun centré en un point différent, avec un certain chevauchement. Ce type de données permet de visualiser clairement la frontière de décision apprise.

```{code-cell} python
:tags: [hide-input]

# Générer des données
np.random.seed(123)
n_samples = 300

# Deux classes avec centres différents
X0 = np.random.randn(n_samples // 2, 2) + np.array([-1, -1])
X1 = np.random.randn(n_samples // 2, 2) + np.array([1.5, 1.5])
X_data = np.vstack([X0, X1])
y_data = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

# Ajouter colonne de 1 pour le biais
X_with_bias = np.hstack([np.ones((n_samples, 1)), X_data])

# Entraîner
theta_learned, loss_history = logistic_regression_sgd(
    X_with_bias, y_data, lr=0.5, n_epochs=50, batch_size=32
)

print(f"Paramètres appris: biais={theta_learned[0]:.2f}, θ₁={theta_learned[1]:.2f}, θ₂={theta_learned[2]:.2f}")

# Précision
y_pred = predict(X_with_bias, theta_learned)
accuracy = np.mean(y_pred == y_data)
print(f"Précision: {accuracy:.1%}")
```

La figure ci-dessous montre les résultats. À gauche, les points des deux classes et la frontière de décision apprise: c'est la droite où $\boldsymbol{\theta}^\top \mathbf{x} = 0$, soit $\theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0$. Les points au-dessus de cette droite sont classés comme positifs, ceux en dessous comme négatifs. À droite, la courbe d'entropie croisée montre que l'algorithme converge rapidement vers le minimum.

```{code-cell} python
:tags: [hide-input]

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Données et frontière de décision
ax = axes[0]
ax.scatter(X0[:, 0], X0[:, 1], c='C1', label='Classe 0', alpha=0.6, edgecolor='white')
ax.scatter(X1[:, 0], X1[:, 1], c='C0', label='Classe 1', alpha=0.6, edgecolor='white')

# Frontière: theta[0] + theta[1]*x1 + theta[2]*x2 = 0
x1_range = np.linspace(-4, 5, 100)
x2_boundary = -(theta_learned[0] + theta_learned[1] * x1_range) / theta_learned[2]
ax.plot(x1_range, x2_boundary, 'k-', linewidth=2, label='Frontière apprise')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Classification par régression logistique')
ax.set_xlim(-4, 5)
ax.set_ylim(-4, 5)
ax.legend()

# Courbe de perte
ax = axes[1]
ax.plot(loss_history, 'C0-', linewidth=2)
ax.set_xlabel('Époque')
ax.set_ylabel('Entropie croisée')
ax.set_title('Convergence de l\'entraînement')

plt.tight_layout()
```

## Résumé

Ce chapitre a introduit la classification linéaire:

- La **régression logistique** modélise $p(y=1|\mathbf{x}) = \sigma(\boldsymbol{\theta}^\top \mathbf{x})$ où $\sigma$ est la fonction sigmoïde.

- Le **maximum de vraisemblance** appliqué au modèle de Bernoulli donne l'**entropie croisée** comme fonction de perte. Ce n'est pas un choix arbitraire.

- Le gradient a une forme simple: $\frac{1}{N} \sum_i (\mu_i - y_i) \mathbf{x}_i$, analogue aux moindres carrés.

- L'objectif est **convexe**, garantissant un unique minimum global.

- La **fonction softmax** généralise la sigmoïde au cas multiclasse, avec l'**entropie croisée catégorielle** comme perte.

- La **descente de gradient stochastique** (SGD) permet l'optimisation efficace, en utilisant des mini-lots pour estimer le gradient.

- La **frontière de décision** est un hyperplan dans l'espace des entrées.

Nous avons maintenant deux modèles linéaires fondamentaux: la régression (chapitre 2) et la classification (ce chapitre). Le [chapitre suivant](ch4_generalization.md) étudie comment ces modèles généralisent à de nouvelles données, en explorant le **compromis biais-variance** et les techniques de **sélection de modèle**.

## Exercices

````{admonition} Exercice 1: Dérivation de la sigmoïde ★
:class: hint dropdown

1. Montrez que $\sigma(-a) = 1 - \sigma(a)$.

2. Montrez que $\frac{d\sigma}{da} = \sigma(a)(1 - \sigma(a))$.

3. Utilisez ces propriétés pour simplifier $\log \sigma(a) + \log(1 - \sigma(a))$.
````

```{admonition} Solution Exercice 1
:class: dropdown

1. **$\sigma(-a) = 1 - \sigma(a)$:**

   $$\sigma(-a) = \frac{1}{1 + e^{-(-a)}} = \frac{1}{1 + e^a} = \frac{e^{-a}}{e^{-a} + 1} = 1 - \frac{1}{1 + e^{-a}} = 1 - \sigma(a)$$

2. **Dérivée:**

   $$\frac{d\sigma}{da} = \frac{d}{da} (1 + e^{-a})^{-1} = -(-e^{-a})(1 + e^{-a})^{-2} = \frac{e^{-a}}{(1 + e^{-a})^2}$$
   
   On peut réécrire: $\frac{e^{-a}}{(1 + e^{-a})^2} = \frac{1}{1 + e^{-a}} \cdot \frac{e^{-a}}{1 + e^{-a}} = \sigma(a) \cdot (1 - \sigma(a))$

3. **Simplification:**

   $$\log \sigma(a) + \log(1 - \sigma(a)) = \log[\sigma(a)(1-\sigma(a))]$$
   
   En utilisant $\sigma(a) = \frac{1}{1+e^{-a}}$ et $1-\sigma(a) = \frac{e^{-a}}{1+e^{-a}}$:
   
   $$\sigma(a)(1-\sigma(a)) = \frac{e^{-a}}{(1+e^{-a})^2}$$
   
   Donc: $\log \sigma(a) + \log(1-\sigma(a)) = -a - 2\log(1 + e^{-a})$
```

````{admonition} Exercice 2: Régularisation de la régression logistique ★★
:class: hint dropdown

Comme pour Ridge, nous pouvons régulariser la régression logistique en ajoutant une pénalité $\ell_2$:

$$
\text{LVN}_\lambda(\boldsymbol{\theta}) = \text{LVN}(\boldsymbol{\theta}) + \frac{\lambda}{2} \|\boldsymbol{\theta}\|_2^2
$$

1. Calculez le gradient de l'objectif régularisé.

2. Modifiez l'implémentation `logistic_regression_sgd` pour inclure la régularisation.

3. Sur un jeu de données avec peu d'exemples, comparez les frontières de décision avec et sans régularisation.
````

````{admonition} Solution Exercice 2
:class: dropdown

1. **Gradient régularisé:**

   $$\nabla \text{LVN}_\lambda(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^N (\mu_i - y_i) \mathbf{x}_i + \lambda \boldsymbol{\theta}$$

2. **Implémentation:**

   ```python
   def logistic_regression_sgd_regularized(X, y, lr=0.1, n_epochs=100, 
                                            batch_size=32, lambda_reg=0.01):
       N, d = X.shape
       theta = np.zeros(d)
       
       for epoch in range(n_epochs):
           indices = np.random.permutation(N)
           for start in range(0, N, batch_size):
               batch_idx = indices[start:start + batch_size]
               X_batch, y_batch = X[batch_idx], y[batch_idx]
               
               mu = sigmoid(X_batch @ theta)
               grad = X_batch.T @ (mu - y_batch) / len(batch_idx)
               grad += lambda_reg * theta  # Terme de régularisation
               
               theta = theta - lr * grad
       
       return theta
   ```

3. Avec peu d'exemples, la régularisation empêche les coefficients de devenir trop grands, produisant une frontière plus lisse et moins sensible au bruit.
````

````{admonition} Exercice 3: Softmax et température ★★
:class: hint dropdown

La fonction softmax avec **température** $T > 0$ est définie par:

$$
\text{softmax}_T(\mathbf{a})_c = \frac{e^{a_c / T}}{\sum_{c'} e^{a_{c'} / T}}
$$

1. Que se passe-t-il quand $T \to 0$? (Indice: quelle classe domine?)

2. Que se passe-t-il quand $T \to \infty$?

3. Implémentez `softmax_temperature(a, T)` et visualisez l'effet de $T$ sur un vecteur $\mathbf{a} = [2, 1, 0.5]$.
````

````{admonition} Solution Exercice 3
:class: dropdown

1. **$T \to 0$:** La distribution devient de plus en plus concentrée sur la classe avec le logit maximal. À la limite, c'est un vecteur one-hot: $[0, \ldots, 1, \ldots, 0]$ avec 1 à la position $\arg\max_c a_c$.

2. **$T \to \infty$:** Les logits divisés par $T$ tendent tous vers 0. La distribution devient uniforme: $[1/C, \ldots, 1/C]$.

3. **Implémentation:**

   ```python
   def softmax_temperature(a, T):
       a_scaled = a / T
       a_max = np.max(a_scaled)
       exp_a = np.exp(a_scaled - a_max)
       return exp_a / np.sum(exp_a)
   
   a = np.array([2, 1, 0.5])
   for T in [0.1, 0.5, 1.0, 2.0, 10.0]:
       probs = softmax_temperature(a, T)
       print(f"T={T}: {probs}")
   ```
   
   Résultat: avec $T=0.1$, presque [1, 0, 0]; avec $T=10$, proche de [0.37, 0.33, 0.30].
````

````{admonition} Exercice 4: Comparaison avec scikit-learn ★
:class: hint dropdown

Utilisez `sklearn.linear_model.LogisticRegression` pour entraîner un modèle sur les mêmes données que l'exemple du chapitre.

1. Comparez les coefficients appris avec votre implémentation.

2. Comparez les temps d'exécution pour différentes tailles de données.

3. Explorez les options de régularisation de scikit-learn (`penalty`, `C`).
````
