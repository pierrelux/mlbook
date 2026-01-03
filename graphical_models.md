# Modèles graphiques probabilistes

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Représenter des distributions de probabilité avec des graphes
- Distinguer les modèles graphiques dirigés et non-dirigés
- Lire l'indépendance conditionnelle à partir de la structure du graphe
- Décrire l'inférence et l'apprentissage dans des modèles graphiques
- Appliquer l'algorithme d'élimination de variables
- Comprendre l'inférence variationnelle et l'ELBO
- Expliquer les auto-encodeurs variationnels (VAE)
```

## Introduction

Les **modèles graphiques probabilistes** (PGM) utilisent un graphe pour représenter une distribution de probabilité conjointe et ses relations d'indépendance conditionnelle. Ils fournissent un cadre unifiant pour modéliser l'incertitude et raisonner sur des systèmes complexes.

## Rappels de probabilités

### Règles fondamentales

La **distribution marginale** d'une variable aléatoire:
$$
p(X=x) = \sum_y p(X=x, Y=y)
$$

Cette relation est aussi appelée la **règle de la somme** ou **règle de la probabilité totale**.

La **probabilité conditionnelle**:
$$
p(Y=y \mid X=x) = \frac{p(X=x, Y=y)}{p(X=x)}
$$

Ce qui donne la **règle du produit**:
$$
p(x, y) = p(y \mid x) p(x)
$$

### Règle de la chaîne

En étendant la règle du produit à $D$ variables, nous obtenons la **règle de la chaîne de probabilité**:
$$
p(x_{1:D}) = p(x_1) p(x_2 \mid x_1) p(x_3 \mid x_1, x_2) \cdots p(x_D \mid x_{1:D-1})
$$

Cette règle permet d'exprimer une distribution conjointe de grande dimension à partir de distributions conditionnelles.

## Indépendance

### Indépendance marginale

$X$ et $Y$ sont **indépendants marginalement** (noté $X \perp Y$) si:
$$
X \perp Y \Longleftrightarrow p(X, Y) = p(X) p(Y)
$$

Un ensemble de variables $X_1, \ldots, X_n$ est **mutuellement indépendant** si pour tous les sous-ensembles $\{X_1, \ldots, X_m\} \subseteq \{X_1, \ldots, X_n\}$:
$$
p(X_1, \ldots, X_m) = \prod_{i=1}^m p(X_i)
$$

### Exemple

Pour que $X_1, X_2, X_3$ soient mutuellement indépendantes, il faut:
- $p(X_1, X_2, X_3) = p(X_1) p(X_2) p(X_3)$
- $p(X_1, X_2) = p(X_1) p(X_2)$
- $p(X_2, X_3) = p(X_2) p(X_3)$
- $p(X_1, X_3) = p(X_1) p(X_3)$

L'indépendance marginale est rare, car la plupart des variables peuvent influencer d'autres variables.

### Indépendance conditionnelle

L'influence entre des variables se fait généralement par l'entremise d'autres variables (**médiation**).

$X$ et $Y$ sont **conditionnellement indépendants étant donné** $Z$ si:
$$
X \perp Y \mid Z \Longleftrightarrow p(X, Y \mid Z) = p(X \mid Z) p(Y \mid Z)
$$

Intuitivement: $X$ n'informe pas $Y$ si $Z$ est observé (et vice versa). Toute l'information nécessaire est déjà contenue dans $Z$.

## Modèles graphiques probabilistes

### Définition

Un **modèle graphique probabiliste** (PGM) utilise un graphe pour représenter une distribution de probabilité conjointe et ses relations d'indépendance conditionnelle.

Dans le graphe:
- Chaque **nœud** représente une variable aléatoire
- Chaque **arête** représente une dépendance directe
- Chaque **absence d'arête** représente une indépendance conditionnelle

### Types de modèles graphiques

| Type | Graphe | Exemple |
|------|--------|---------|
| **Réseau bayésien** | Graphe acyclique dirigé (DAG) | Classification naïve de Bayes |
| **Champ de Markov** | Graphe non-dirigé | Modèles de Boltzmann |

## Réseaux bayésiens

### Factorisation

Pour un graphe acyclique dirigé (DAG), chaque nœud est conditionnellement indépendant de tous ses prédécesseurs étant donné ses parents:
$$
Y_i \perp \mathbf{Y}_{\text{pred}(i) \setminus \text{pa}(i)} \mid \mathbf{Y}_{\text{pa}(i)}
$$

où $\text{pa}(i)$ sont les parents du nœud $i$ et $\text{pred}(i)$ sont ses prédécesseurs dans l'ordre topologique.

La probabilité conjointe se factorise:
$$
p(Y_{1:N_G}) = \prod_{i=1}^{N_G} p(Y_i \mid \mathbf{Y}_{\text{pa}(i)})
$$

où $N_G$ est le nombre de nœuds dans le graphe.

### Exemple: Réseau de l'arroseur automatique

Considérons les variables:
1. $C$: saison nuageuse ou non
2. $R$: pluie ou non
3. $S$: arroseur automatique allumé ou non
4. $W$: herbe mouillée ou non

La probabilité conjointe:
$$
p(C, S, R, W) = p(C) p(S \mid C) p(R \mid C) p(W \mid S, R)
$$

Les relations de CI simplifient la factorisation:
- $S \perp R \mid C$ (l'arroseur et la pluie sont indépendants étant donné la météo)

### Distribution de probabilité conditionnelle (CPD)

Chaque $p(Y_i \mid \mathbf{Y}_{\text{pa}(i)})$ est appelée la **distribution de probabilité conditionnelle** (CPD) pour le nœud $i$.

Pour des variables catégorielles, on utilise un **tableau de probabilité conditionnelle** (CPT):
$$
\theta_{ijk} \triangleq p(Y_i = k \mid \mathbf{Y}_{\text{pa}(i)} = j)
$$

où:
- $i$ indexe les nœuds, $i \in [N_G]$
- $k$ indexe les états du nœud, $k \in [K_i]$
- $j$ indexe les configurations des parents, $j \in [J_i]$ avec $J_i = \prod_{p \in \text{pa}(i)} K_p$

Les contraintes sont: $0 \leq \theta_{ijk} \leq 1$ et $\sum_{k=1}^{K_i} \theta_{ijk} = 1$.

### Notation de plaque

Pour éviter les encombrements visuels dans les graphes avec des variables répétées, on utilise la **notation de plaque**: une boîte (plaque) entoure les variables qui se répètent $N$ fois.

## Modèles de séquences

### Modèles autorégressifs

Pour une séquence $y_{1:T}$, la règle de la chaîne donne:
$$
p(y_{1:T}) = \prod_{t=1}^T p(y_t \mid y_{1:t-1})
$$

Le nombre de paramètres pour $p(y_t \mid y_{1:t-1})$ augmente exponentiellement avec $t$.

### Chaînes de Markov

Si nous faisons l'hypothèse que **l'avenir est indépendant du passé étant donné le présent** (condition de Markov d'ordre 1):
$$
p(y_{1:T}) = p(y_1) \prod_{t=2}^T p(y_t \mid y_{t-1})
$$

Ce modèle est appelé **chaîne de Markov** ou **modèle autorégressif d'ordre 1**.

### Noyau de transition

La fonction $p(y_t \mid y_{t-1})$ est le **noyau de transition** ou **noyau de Markov**:
$$
p(y_t \mid y_{t-1}) \geq 0, \quad \sum_{k=1}^K p(y_t = k \mid y_{t-1} = j) = 1
$$

Le CPT de ce modèle est contenu dans une **matrice stochastique** de transition:
$$
A_{jk} = p(y_t = k \mid y_{t-1} = j)
$$

où chaque ligne somme à 1. Si cette matrice est la même pour toutes les étapes, le modèle est dit **homogène** ou **invariant dans le temps**.

### Modèles d'ordre supérieur

L'hypothèse de Markov peut être généralisée à des dépendances sur les $M$ observations précédentes:
$$
p(y_{1:T}) = p(y_{1:M}) \prod_{t=M+1}^T p(y_t \mid y_{t-M:t-1})
$$

En modélisation du langage:
- **Bigramme**: modèle de Markov d'ordre 1 ($M=1$)
- **Trigramme**: modèle de Markov d'ordre 2 ($M=2$)

## Échantillonnage et génération

### Échantillonnage ancestral

Pour générer des échantillons d'un réseau bayésien:
1. Visiter les nœuds dans l'**ordre topologique** (parents avant enfants)
2. Échantillonner une valeur pour chaque nœud en fonction de ses parents

Cette technique s'appelle **l'échantillonnage ancestral** et permet d'obtenir des échantillons indépendants de la distribution conjointe:
$$
(x_1, \ldots, x_{N_G}) \sim p(\boldsymbol{x} \mid \boldsymbol{\theta})
$$

## Apprentissage dans les PGM

Si les paramètres des CPD sont inconnus, nous pouvons les traiter comme des variables à inférer:
$$
\begin{aligned}
\boldsymbol{\theta} &\sim p(\boldsymbol{\theta}) \\
\boldsymbol{y}_n &\sim p(\boldsymbol{y} \mid \boldsymbol{\theta})
\end{aligned}
$$

où $p(\boldsymbol{\theta})$ est un a priori sur les paramètres.

La distribution conjointe:
$$
p(\mathcal{D}, \boldsymbol{\theta}) = p(\boldsymbol{\theta}) p(\mathcal{D} \mid \boldsymbol{\theta})
$$

Sous l'hypothèse i.i.d., la vraisemblance:
$$
p(\mathcal{D} \mid \boldsymbol{\theta}) = \prod_{n=1}^N p(\boldsymbol{y}_n \mid \boldsymbol{\theta})
$$

## Inférence

### Le problème d'inférence

En inférence, nous calculons le a posteriori sur un ensemble de nœuds de **requête** $Q$ étant donné les valeurs observées pour un ensemble de nœuds **visibles** $V$. Nous devons marginaliser sur les variables de **nuisance** $R$:
$$
p_{\boldsymbol{\theta}}(Q \mid V) = \frac{p_{\boldsymbol{\theta}}(Q, V)}{p_{\boldsymbol{\theta}}(V)} = \frac{\sum_R p_{\boldsymbol{\theta}}(Q, V, R)}{p_{\boldsymbol{\theta}}(V)}
$$

**En général, l'inférence est NP-difficile.**

### Élimination de variables

L'**élimination de variables** est un algorithme exact qui exploite la structure du graphe pour calculer efficacement les marginales.

L'idée clé est d'**intégrer les sommes dans les produits**:
$$
p(J) = \sum_{L,S,G,H,I,D,C} p(C,D,I,G,S,L,J,H)
$$

Au lieu d'énumérer toutes les $2^7$ configurations, on procède par étapes:

1. **Multiplier** les facteurs dans la portée de la somme la plus interne
2. **Marginaliser** pour créer un nouveau facteur
3. **Répéter** jusqu'à ce que toutes les variables de nuisance soient éliminées

**Exemple**:
$$
\tau_1(D) = \sum_C \psi_C(C) \psi_D(D, C)
$$
$$
\tau_2(G, I) = \sum_D \psi_G(G, I, D) \tau_1(D)
$$

Et ainsi de suite.

## Inférence variationnelle

### Motivation

Considérons un modèle avec:
- Variables latentes (inconnues): $\boldsymbol{z}$
- Variables observées: $\boldsymbol{x}$
- Paramètres fixes: $\boldsymbol{\theta}$

L'a posteriori exact:
$$
p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x}) = \frac{p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})}{p_{\boldsymbol{\theta}}(\boldsymbol{x})}
$$

La normalisation $p_{\boldsymbol{\theta}}(\boldsymbol{x}) = \int p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z}) d\boldsymbol{z}$ est souvent **impossible à calculer**.

### Approximation variationnelle

L'idée est de trouver une approximation $q(\boldsymbol{z})$ de l'a posteriori en minimisant:
$$
q^* = \underset{q \in \mathcal{Q}}{\text{argmin}} \, D_{\text{KL}}(q(\boldsymbol{z}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x}))
$$

### Borne inférieure de l'évidence (ELBO)

En développant la divergence KL:
$$
D_{\text{KL}}(q_{\boldsymbol{\psi}}(\boldsymbol{z}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z} \mid \boldsymbol{x})) = \mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\psi} \mid \boldsymbol{x}) + \log p_{\boldsymbol{\theta}}(\boldsymbol{x})
$$

où:
$$
\mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\psi} \mid \boldsymbol{x}) = \mathbb{E}_{q_{\boldsymbol{\psi}}(\boldsymbol{z})}[-\log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z}) + \log q_{\boldsymbol{\psi}}(\boldsymbol{z})]
$$

Puisque $D_{\text{KL}} \geq 0$:
$$
-\mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\psi} \mid \boldsymbol{x}) \leq \log p_{\boldsymbol{\theta}}(\boldsymbol{x})
$$

Le négatif de $\mathcal{L}$ est la **borne inférieure de l'évidence** (ELBO):
$$
\text{ELBO} = \mathbb{E}_{q_{\boldsymbol{\psi}}(\boldsymbol{z})}[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z}) - \log q_{\boldsymbol{\psi}}(\boldsymbol{z})]
$$

**Maximiser l'ELBO** équivaut à **minimiser la divergence KL**.

### Interprétations de l'ELBO

**Forme 1**: Énergie et entropie
$$
\text{ELBO} = \mathbb{E}_{q_{\boldsymbol{\psi}}(\boldsymbol{z})}[\log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})] + \mathbb{H}(q_{\boldsymbol{\psi}})
$$

$$
= \text{log-probabilité conjointe attendue} + \text{entropie}
$$

**Forme 2**: Vraisemblance et régularisation
$$
\text{ELBO} = \mathbb{E}_{q_{\boldsymbol{\psi}}(\boldsymbol{z})}[\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})] - D_{\text{KL}}(q_{\boldsymbol{\psi}}(\boldsymbol{z}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z}))
$$

$$
= \text{vraisemblance attendue} - \text{régularisation}
$$

Le terme KL empêche l'a posteriori approché de s'éloigner trop de l'a priori.

## Inférence variationnelle par gradient

### Gradient par rapport à $\boldsymbol{\theta}$

Le gradient par rapport aux paramètres du modèle est facile car la dépendance est **structurelle**:
$$
\nabla_{\boldsymbol{\theta}} \text{ELBO} = \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z})}[\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z})] \approx \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(\boldsymbol{x}, \boldsymbol{z}^s)
$$

où $\boldsymbol{z}^s \sim q_{\boldsymbol{\phi}}(\boldsymbol{z})$.

### Gradient par rapport à $\boldsymbol{\phi}$

Le gradient par rapport aux paramètres variationnels est plus difficile car la dépendance est **distributionnelle**:
$$
\nabla_{\boldsymbol{\phi}} \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z})}[f(\boldsymbol{z})] \neq \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z})}[\nabla_{\boldsymbol{\phi}} f(\boldsymbol{z})]
$$

### L'astuce de reparamétrisation

La solution est de réécrire $\boldsymbol{z} \sim q_{\boldsymbol{\phi}}(\boldsymbol{z})$ comme une transformation d'une variable aléatoire indépendante de $\boldsymbol{\phi}$.

Pour une gaussienne:
$$
\boldsymbol{z} \sim \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma})) \Longleftrightarrow \boldsymbol{z} = \boldsymbol{\mu} + \boldsymbol{\epsilon} \odot \boldsymbol{\sigma}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Maintenant:
$$
\mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z})}[f(\boldsymbol{z})] = \mathbb{E}_{p(\boldsymbol{\epsilon})}[f(g(\boldsymbol{\phi}, \boldsymbol{\epsilon}))]
$$

Et le gradient devient:
$$
\nabla_{\boldsymbol{\phi}} \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z})}[f(\boldsymbol{z})] = \mathbb{E}_{p(\boldsymbol{\epsilon})}[\nabla_{\boldsymbol{\phi}} f(g(\boldsymbol{\phi}, \boldsymbol{\epsilon}))]
$$

## Auto-encodeurs variationnels (VAE)

### Modèles de variables latentes profondes

Un **modèle de variables latentes profondes** (DLVM) est un modèle génératif:
$$
\begin{aligned}
\boldsymbol{z} &\sim p_{\boldsymbol{\theta}}(\boldsymbol{z}) \\
\boldsymbol{x} \mid \boldsymbol{z} &\sim \text{Expfam}(\boldsymbol{x} \mid d_{\boldsymbol{\theta}}(\boldsymbol{z}))
\end{aligned}
$$

où:
- $p_{\boldsymbol{\theta}}(\boldsymbol{z})$ est un a priori sur le code latent (souvent $\mathcal{N}(\mathbf{0}, \mathbf{I})$)
- $d_{\boldsymbol{\theta}}(\boldsymbol{z})$ est un réseau de neurones profond appelé **décodeur**

### Architecture du VAE

Un **auto-encodeur variationnel** (VAE) combine:

**Modèle génératif (décodeur)**:
$$
p_{\boldsymbol{\theta}}(\boldsymbol{z}, \boldsymbol{x}) = p_{\boldsymbol{\theta}}(\boldsymbol{z}) p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})
$$

Pour des observations binaires:
$$
p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z}) = \prod_{d=1}^D \text{Ber}(x_d \mid \sigma(d_{\boldsymbol{\theta}}(\boldsymbol{z})))
$$

**Modèle de reconnaissance (encodeur)**:
$$
q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) = \mathcal{N}(\boldsymbol{z} \mid \boldsymbol{\mu}, \text{diag}(\exp(\boldsymbol{\ell})))
$$

où $(\boldsymbol{\mu}, \boldsymbol{\ell}) = e_{\boldsymbol{\phi}}(\boldsymbol{x})$ sont les sorties de l'encodeur.

### Entraînement du VAE

L'ELBO pour un VAE:
$$
\mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\phi}) = \mathbb{E}_{q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x})}[\log p_{\boldsymbol{\theta}}(\boldsymbol{x} \mid \boldsymbol{z})] - D_{\text{KL}}(q_{\boldsymbol{\phi}}(\boldsymbol{z} \mid \boldsymbol{x}) \| p_{\boldsymbol{\theta}}(\boldsymbol{z}))
$$

- **Premier terme**: reconstruction (l'encodage permet-il de reconstruire $\boldsymbol{x}$?)
- **Second terme**: régularisation (l'a posteriori reste proche de l'a priori)

L'astuce de reparamétrisation permet d'entraîner les deux réseaux par descente de gradient.

### Inférence variationnelle stochastique

Pour les grands ensembles de données, on utilise des mini-lots:
$$
\mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\phi} \mid \mathcal{D}) \approx \frac{N}{B} \sum_{\boldsymbol{x}_n \in \mathcal{B}} \mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\phi} \mid \boldsymbol{x}_n)
$$

Cette technique est appelée **inférence variationnelle stochastique** (SVI).

## Résumé

Les modèles graphiques probabilistes offrent un cadre puissant pour:

| Concept | Description |
|---------|-------------|
| **Représentation** | Factoriser les distributions jointes via les indépendances |
| **Échantillonnage** | Générer des données via l'échantillonnage ancestral |
| **Inférence** | Calculer les marginales et conditionnelles |
| **Apprentissage** | Estimer les paramètres des CPD |

L'inférence variationnelle permet d'approximer des a posteriori intractables, et les VAE combinent cette approche avec des réseaux de neurones profonds pour apprendre des représentations latentes.

## Exercices

```{admonition} Exercice 1: Factorisation
:class: tip

Considérez un réseau bayésien avec les arêtes: $A \to B$, $A \to C$, $B \to D$, $C \to D$.

1. Écrivez la factorisation de la distribution conjointe $p(A, B, C, D)$.
2. Quelles sont les relations d'indépendance conditionnelle impliquées par ce graphe?
3. Combien de paramètres sont nécessaires si toutes les variables sont binaires?
```

```{admonition} Exercice 2: Élimination de variables
:class: tip

Pour le réseau $A \to B \to C$:
1. Calculez $p(C)$ par élimination de variables.
2. Montrez les facteurs intermédiaires créés à chaque étape.
```

```{admonition} Exercice 3: ELBO
:class: tip

Pour un VAE avec a priori $p(\boldsymbol{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ et a posteriori approché $q_{\boldsymbol{\phi}}(\boldsymbol{z}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$:

1. Calculez analytiquement $D_{\text{KL}}(q_{\boldsymbol{\phi}}(\boldsymbol{z}) \| p(\boldsymbol{z}))$.
2. Pourquoi cette forme analytique est-elle utile pour l'entraînement?
```

```{admonition} Exercice 4: Reparamétrisation
:class: tip

Montrez comment appliquer l'astuce de reparamétrisation pour:
1. Une distribution uniforme $z \sim \text{Unif}(a, b)$
2. Une distribution exponentielle $z \sim \text{Exp}(\lambda)$
```
