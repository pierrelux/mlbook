---
marp: true
theme: mila
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Réseaux de neurones
## et apprentissage profond

*Pierre-Luc Bacon*
IFT3395/IFT6390 – Apprentissage machine

---

## Plan de la présentation

1. **Cadre unifié** : maximum de vraisemblance et caractéristiques apprises
2. **Le perceptron** : origines, limites structurelles
3. **Le problème XOR** : pourquoi les modèles linéaires ne suffisent pas
4. **Anatomie d'un MLP** : couches, activations, expressivité
5. **Rétropropagation** : règle de la chaîne et VJP
6. **Optimisation** : SGD, momentum, Adam
7. **Différentiation automatique** : graphes de calcul, traceur
8. **Pratique** : gradient qui disparaît, initialisation, batch norm, dropout

---

<!-- _class: lead -->

# 1. Cadre unifié
## Maximum de vraisemblance et caractéristiques apprises

---

## Prédire les paramètres d'une distribution

Dans tous les modèles vus jusqu'ici, le problème est le même :

> Étant donné $\mathbf{x}$, prédire les paramètres d'une distribution $p(y|\mathbf{x};\boldsymbol{\theta})$

| Problème | Distribution | Sortie $\mu(\mathbf{x})$ | Perte |
|----------|-------------|--------------------------|-------|
| Régression | Gaussienne $\mathcal{N}(\mu, \sigma^2)$ | Moyenne | Moindres carrés |
| Classif. binaire | Bernoulli | $p(y=1|\mathbf{x})$ | Entropie croisée |
| Classif. multi | Catégorielle | $\text{softmax}(\cdot)$ | Entropie croisée |

La question centrale : **quelle forme donner à $\mu(\mathbf{x})$?**

---

## Des modèles linéaires aux caractéristiques apprises

**Modèles linéaires** (ch. 2–3) : $\mu(\mathbf{x}) = \boldsymbol{\theta}^\top \mathbf{x}$

**Expansion de caractéristiques** (ch. 4) : $\mu(\mathbf{x}) = \boldsymbol{\theta}^\top \boldsymbol{\phi}(\mathbf{x})$
— mais $\boldsymbol{\phi}$ est **fixée à l'avance** par le praticien

**Réseaux de neurones** : on *apprend* $\boldsymbol{\phi}$ à partir des données

$$\mu(\mathbf{x}) = \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x};\boldsymbol{\theta}_\phi)$$

Les paramètres $\boldsymbol{\theta}_\phi$ contrôlent la transformation des entrées.
On optimise **tout** simultanément : représentation + prédicteur.

---

<!-- _class: lead -->

# 2. Le perceptron
## Origines et limites structurelles

---

## Un modèle inspiré du neurone biologique

**McCulloch & Pitts (1943)** : le neurone comme unité de calcul binaire

$$\hat{y} = \mathbf{1}[\boldsymbol{\theta}^\top \mathbf{x} \geq 0]$$

**Rosenblatt (1958)** : le perceptron, avec un algorithme d'apprentissage

| | Régression logistique | Perceptron |
|---|---|---|
| Activation | $\sigma(z)$ (sigmoïde) | $\mathbf{1}[z \geq 0]$ (échelon) |
| Sortie | probabilité $\in (0,1)$ | décision $\in \{0,1\}$ |
| Frontière | $\boldsymbol{\theta}^\top \mathbf{x} = 0$ | $\boldsymbol{\theta}^\top \mathbf{x} = 0$ |

**Même hyperplan séparateur** — deux lectures différentes du même modèle.

---

## Perceptron vs régression logistique

![w:900](../slides/_static/ch7_perceptron_vs_logistic.png)

La sigmoïde est une version **lisse et probabiliste** de la fonction échelon.

---

## Règle d'apprentissage et convergence

La perte perceptron (avec $y_i \in \{-1, +1\}$) :

$$\mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^n \max(0,\; -y_i \cdot \boldsymbol{\theta}^\top \mathbf{x}_i)$$

Mise à jour pour chaque exemple **mal classé** :

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \eta\, y_i \mathbf{x}_i \qquad \text{si } y_i \boldsymbol{\theta}^\top \mathbf{x}_i \leq 0$$

**Théorème de convergence** (Novikoff, 1962) : si les données sont linéairement séparables, l'algorithme converge en au plus $(R/\gamma)^2$ mises à jour.

Si les données **ne sont pas** linéairement séparables → peut cycler indéfiniment.

---

## Limite structurelle

**Minsky & Papert (1969)** : aucun perceptron simple ne peut apprendre XOR.

Toutes ces variantes partagent la même contrainte :

> La frontière de décision est un **hyperplan**.
> On ne peut séparer que des classes linéairement séparables.

Mais leur livre contenait aussi la solution :
**empiler deux couches de perceptrons suffit à résoudre XOR.**

---

<!-- _class: lead -->

# 3. Le problème XOR
## Pourquoi les modèles linéaires ne suffisent pas

---

## XOR : une fonction simple, impossible pour un linéaire

| $x_1$ | $x_2$ | $y = x_1 \oplus x_2$ |
|:-----:|:-----:|:--------------------:|
| 0     | 0     | 0                    |
| 0     | 1     | 1                    |
| 1     | 0     | 1                    |
| 1     | 1     | 0                    |

Les points de la classe 0 sont sur une diagonale, ceux de la classe 1 sur l'autre.
**Aucune droite ne peut séparer ces deux groupes.**

---

## La couche cachée transforme l'espace

![w:900](../slides/_static/ch7_xor_problem.png)

Deux neurones ReLU : $h_1 = \text{ReLU}(x_1+x_2-0{,}5)$, $h_2 = \text{ReLU}(x_1+x_2-1{,}5)$

Dans l'espace $(h_1, h_2)$, les classes deviennent **linéairement séparables**.

---

## Le MLP appris sur XOR

![w:500 center](../slides/_static/ch7_xor_mlp.png)

Le réseau découvre **lui-même** la transformation — aucune ingénierie manuelle.

---

## Ce que l'exemple révèle

**1. La couche cachée crée des caractéristiques**
$h_1$ détecte «au moins une entrée active»; $h_2$ détecte «les deux actives».

**2. La non-linéarité est indispensable**
Sans $\varphi$, composer deux couches linéaires donne une seule couche linéaire :
$$W_L(\cdots W_1 \mathbf{x}) = W' \mathbf{x}$$

**3. L'apprentissage trouve la bonne transformation**
En pratique, on n'invente pas $W_1$, $\mathbf{b}_1$ à la main : l'optimisation les découvre.

---

<!-- _class: lead -->

# 4. Anatomie d'un MLP
## Couches, activations, expressivité

---

## Un neurone : transformation affine + non-linéarité

$$h = \varphi(\underbrace{\mathbf{w}^\top \mathbf{x} + b}_{\text{pré-activation } a})$$

Une **couche** de $m$ neurones, sous forme matricielle :

$$\mathbf{h} = \varphi(W\mathbf{x} + \mathbf{b}), \quad W \in \mathbb{R}^{m \times d}$$

- $W$, $\mathbf{b}$ : **paramètres** (appris)
- $\varphi$ : **fonction d'activation** (non linéaire, appliquée élément par élément)
- $\mathbf{h}$ : **activations** (caractéristiques apprises)

---

## Fonctions d'activation

![w:900](../slides/_static/ch7_activation_functions.png)

La dérivée de la sigmoïde est bornée par $0{,}25$ → **saturation** → gradient qui disparaît.
**ReLU** n'a pas ce problème pour les valeurs positives.

---

## Fonctions d'activation : résumé

| Activation | Formule | Avantages | Inconvénients |
|------------|---------|-----------|---------------|
| Sigmoïde | $\sigma(a) = \frac{1}{1+e^{-a}}$ | Sortie en $(0,1)$ | Sature, gradient $\leq 0{,}25$ |
| Tanh | $\tanh(a)$ | Centrée en 0 | Sature aussi |
| **ReLU** | $\max(0, a)$ | Simple, rapide | Neurones morts |
| GELU | $a \cdot \Phi(a)$ | Lisse, performante | Plus coûteuse |

ReLU est aujourd'hui la fonction **par défaut** dans la plupart des architectures.

---

## Le perceptron multicouche (MLP)

Composition de $L$ couches :

$$\mathbf{z}_0 = \mathbf{x}, \quad \mathbf{a}_\ell = W_\ell \mathbf{z}_{\ell-1} + \mathbf{b}_\ell, \quad \mathbf{z}_\ell = \varphi(\mathbf{a}_\ell)$$

Les couches cachées $\mathbf{z}_1, \ldots, \mathbf{z}_{L-1}$ sont les **caractéristiques apprises**.

**Couche de sortie** (selon le problème) :

| Problème | Sortie |
|----------|--------|
| Régression | Linéaire : $\mathbf{w}^\top \mathbf{z}_{L-1} + b$ |
| Classif. binaire | Sigmoïde : $\sigma(\mathbf{w}^\top \mathbf{z}_{L-1} + b)$ |
| Classif. multi | Softmax : $\text{softmax}(W_L \mathbf{z}_{L-1} + \mathbf{b}_L)$ |

Un MLP pour la classification = régression logistique sur **caractéristiques apprises**.

---

## Théorème d'approximation universelle

> Un MLP avec **une seule couche cachée** suffisamment large peut approximer toute fonction continue sur un ensemble compact (Hornik, 1989).

Mais la largeur requise peut croître **exponentiellement** avec la complexité.

Les réseaux **profonds** représentent certaines fonctions de façon beaucoup plus compacte que les réseaux larges mais peu profonds :

- 1 couche, 40 neurones ↔ 4 couches, 8 neurones
- La profondeur permet des représentations **hiérarchiques**

---

<!-- _class: lead -->

# 5. Rétropropagation
## Règle de la chaîne et produits VJP

---

## Le problème du gradient

Pour un réseau avec paramètres $\{W_\ell, \mathbf{b}_\ell\}_{\ell=1}^L$, on doit calculer :

$$\nabla_{W_\ell} \mathcal{L}, \quad \nabla_{\mathbf{b}_\ell} \mathcal{L} \quad \text{pour } \ell = 1, \ldots, L$$

La perte dépend des paramètres de la couche $\ell$ à travers **toutes les couches suivantes**.

La **rétropropagation** résout ce problème en appliquant systématiquement la **règle de la chaîne**.

---

## JVP vs VJP : deux façons de propager

Pour une composition $f = f_L \circ \cdots \circ f_1$ :

$$\mathbf{J}_f = \mathbf{J}_{f_L} \cdots \mathbf{J}_{f_1}$$

![w:900](../slides/_static/ch7_jvp_vjp.png)

---

## Pourquoi le mode arrière?

| | Mode avant (JVP) | Mode arrière (VJP) |
|---|---|---|
| Direction | Gauche → droite | Droite → gauche |
| Coût par passe | $O(mn)$ | $O(mn)$ |
| Passes nécessaires | **$n$ passes** (une par paramètre) | **1 passe** |

Pour $n$ paramètres et une perte scalaire :

> Le mode arrière calcule **tout le gradient** en une seule passe.
> C'est la rétropropagation.

---

## Règles VJP : sans jacobienne explicite

| Opération | $f(\mathbf{x})$ | Règle VJP : $\mathbf{J}_f^\top \mathbf{u}$ |
|-----------|-----------------|---------------------------------------------|
| Couche affine (entrée $\mathbf{z}$) | $W\mathbf{z} + \mathbf{b}$ | $W^\top \mathbf{u}$ |
| Couche affine (poids $W$) | $W\mathbf{z} + \mathbf{b}$ | $\mathbf{u}\mathbf{z}^\top$ |
| Couche affine (biais $\mathbf{b}$) | $W\mathbf{z} + \mathbf{b}$ | $\mathbf{u}$ |
| Activation élémentaire | $\varphi(\mathbf{a})$ | $\mathbf{u} \odot \varphi'(\mathbf{a})$ |

Jamais besoin de former la jacobienne $\mathbf{J}_f$ explicitement.
Pour l'activation : $O(m)$ au lieu de $O(m^2)$.

---

## Exemple : MLP à une couche cachée

**Passe avant** :
$$\mathbf{a}_1 = W_1 \mathbf{x} + \mathbf{b}_1, \quad \mathbf{z}_1 = \varphi(\mathbf{a}_1), \quad \hat{y} = \mathbf{w}_2^\top \mathbf{z}_1 + b_2, \quad \mathcal{L} = \tfrac{1}{2}(y-\hat{y})^2$$

**Passe arrière** :
$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = \hat{y} - y, \quad
\frac{\partial \mathcal{L}}{\partial \mathbf{w}_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \mathbf{z}_1, \quad
\frac{\partial \mathcal{L}}{\partial b_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \mathbf{w}_2, \quad
\frac{\partial \mathcal{L}}{\partial \mathbf{a}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} \odot \varphi'(\mathbf{a}_1)$$

$$\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}_1} \mathbf{x}^\top, \quad
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}_1}$$

---

<!-- _class: lead -->

# 6. Optimisation
## SGD, momentum, Adam

---

## Descente de gradient stochastique par mini-lots

Le gradient exact sur $N$ exemples est coûteux. On estime sur $B$ exemples :

$$\hat{\nabla}_{\boldsymbol{\theta}} \mathcal{L} = \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla_{\boldsymbol{\theta}} \ell(\mathbf{x}_i, y_i; \boldsymbol{\theta})$$

Mise à jour :
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\, \hat{\nabla}_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$$

**Époque** = un passage complet sur le jeu de données ($N/B$ mises à jour).

En pratique : $B \in [32, 512]$. Plus $B$ est petit → plus de bruit → meilleure généralisation (souvent).

---

## Momentum : accumuler de la vitesse

**Problème de SGD** : zigzag dans les vallées étroites et allongées.

**Momentum** (Polyak, 1964) : accumuler une vitesse dans les directions stables

$$\mathbf{m}_{t+1} = \beta \mathbf{m}_t + \hat{\mathbf{g}}_t, \qquad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{m}_{t+1}$$

$\beta \approx 0{,}9$ : la mise à jour courante contribue 10% à la vitesse.

![w:900](../slides/_static/ch7_sgd_vs_momentum.png)

---

## RMSProp et Adam

**RMSProp** : taux d'apprentissage adaptatif par dimension

$$s_{t+1,j} = \beta s_{t,j} + (1-\beta) g_{t,j}^2, \qquad \theta_{t+1,j} = \theta_{t,j} - \frac{\eta}{\sqrt{s_{t+1,j} + \epsilon}} g_{t,j}$$

**Adam** (Kingma & Ba, 2014) : momentum + RMSProp + correction de biais

$$\hat{\mathbf{m}} = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{s}} = \frac{\mathbf{s}_t}{1-\beta_2^t}, \quad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{s}}} + \epsilon}$$

Valeurs par défaut : $\beta_1 = 0{,}9$, $\beta_2 = 0{,}999$, $\epsilon = 10^{-8}$, $\eta = 10^{-3}$.

**Adam est aujourd'hui l'optimiseur par défaut** pour les réseaux de neurones.

---

<!-- _class: lead -->

# 7. Différentiation automatique
## Graphes de calcul et traceur

---

## Trois approches pour calculer des gradients

| Approche | Précision | Coût | Portée |
|----------|-----------|------|--------|
| Différences finies | Approximative | $O(n)$ évaluations | Toute fonction |
| Symbolique | Exacte | Peut exploser | Expressions fermées |
| **Auto-différentiation** | **Exacte** | **$O(1)$ passe** | **Tout programme** |

La rétropropagation = **différentiation automatique en mode arrière**.

La bande (*tape*) enregistre les opérations durant la passe avant,
puis les rejoue à rebours en appelant chaque règle VJP.

---

## La programmation différentiable

```python
import jax
import jax.numpy as jnp

def predict(params, x):
    W1, b1, W2, b2 = params
    h = jnp.tanh(W1 @ x + b1)   # couche cachée
    return W2 @ h + b2            # couche de sortie

def loss_fn(params, x, y):
    return 0.5 * jnp.sum((predict(params, x) - y) ** 2)

# Une seule ligne suffit pour obtenir le gradient
grad_fn = jax.grad(loss_fn)
grads   = grad_fn(params, x, y)
```

On écrit la **passe avant** comme un programme ordinaire.
JAX calcule automatiquement les gradients par rapport à tous les paramètres.

---

<!-- _class: lead -->

# 8. Considérations pratiques
## Gradient, initialisation, régularisation

---

## Le gradient qui disparaît

Dans un réseau profond, le gradient est un **produit de jacobiennes** :

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_L} \prod_{\ell=2}^{L} \frac{\partial \mathbf{z}_\ell}{\partial \mathbf{z}_{\ell-1}}$$

Si chaque jacobienne a un rayon spectral $< 1$ → gradient **disparaît** exponentiellement.
Si rayon spectral $> 1$ → gradient **explose**.

![w:800](../slides/_static/ch7_vanishing_gradient.png)

---

## Solutions au gradient instable

| Problème | Solution |
|----------|----------|
| Gradient qui disparaît | **ReLU** (dérivée = 1 pour $a > 0$) |
| Gradient qui disparaît | **Connexions résiduelles** |
| Gradient qui explose | **Écrêtage** : $\mathbf{g}' = \min(1, c/\|\mathbf{g}\|)\mathbf{g}$ |
| Activation saturée | **Initialisation de Glorot/He** |
| Distributions instables | **Normalisation par lots** |

**Connexions résiduelles** (He et al., 2016) :
$$\mathbf{z}_{\ell+1} = \mathbf{z}_\ell + f(\mathbf{z}_\ell;\boldsymbol{\theta}_\ell), \quad \frac{\partial \mathbf{z}_{\ell+1}}{\partial \mathbf{z}_\ell} = I + \frac{\partial f}{\partial \mathbf{z}_\ell}$$

Le terme identité garantit que le gradient ne s'annule pas.

---

## Initialisation des poids

L'objectif : maintenir la **variance des activations et des gradients** stable à travers les couches.

**Initialisation de Glorot** (Xavier, pour sigmoïde/tanh) :
$$W_{ij} \sim \mathcal{N}\!\left(0, \frac{2}{n + m}\right)$$

**Initialisation de He** (pour ReLU) :
$$W_{ij} \sim \mathcal{N}\!\left(0, \frac{2}{n}\right)$$

Le facteur 2 compense le fait que ReLU annule ~50% des activations.

![w:900](../slides/_static/ch7_weight_init.png)

---

## Normalisation par lots

**Batch normalization** (Ioffe & Szegedy, 2015) : normaliser les pré-activations à chaque couche

$$\hat{a}_j = \frac{a_j - \bar{a}_j}{\sqrt{s_j^2 + \epsilon}}, \qquad \tilde{a}_j = \gamma_j \hat{a}_j + \beta_j$$

où $\bar{a}_j$, $s_j^2$ sont la moyenne et variance empiriques sur le mini-lot.

**Effets** :
- Stabilise l'entraînement
- Permet des taux d'apprentissage plus élevés
- Agit comme régularisateur implicite

---

## Régularisation : décroissance des poids et dropout

**Décroissance des poids** (L2) : $\mathcal{L}_{\text{rég}} = \mathcal{L} + \frac{\lambda}{2}\|\boldsymbol{\theta}\|^2$

$$\boldsymbol{\theta}_{t+1} = (1 - \eta\lambda)\boldsymbol{\theta}_t - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}$$

Équivalent à un prior gaussien sur les paramètres (régularisation MAP).

**Dropout** (Srivastava et al., 2014) : désactiver aléatoirement des neurones à l'entraînement

$$\tilde{\mathbf{z}}_\ell = \frac{1}{1-p}(\boldsymbol{\epsilon} \odot \mathbf{z}_\ell), \quad \boldsymbol{\epsilon} \sim \text{Ber}(1-p)^{\otimes m}$$

Le facteur $\frac{1}{1-p}$ (inverted dropout) préserve l'espérance des activations.

---

## Dropout : visualisation

![w:900](../slides/_static/ch7_dropout.png)

À l'inférence, dropout est désactivé et le réseau complet est utilisé sans rescaling.

---

## Résumé

| Concept | Idée clé |
|---------|----------|
| MLP | Compositions de couches affines + non-linéarités |
| Rétropropagation | Règle de la chaîne en mode arrière (VJP) |
| SGD / Adam | Estimation stochastique du gradient |
| Auto-différentiation | Trace d'exécution + rejeu inverse |
| ReLU | Évite la saturation, simple à calculer |
| Connexions résiduelles | Autoroute pour le gradient |
| Batch norm | Stabilise les distributions d'activations |
| Dropout | Régularisation par bruit structuré |

**Un réseau de neurones** = régression logistique sur des caractéristiques **apprises conjointement** avec le prédicteur.
