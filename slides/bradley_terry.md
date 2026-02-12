---
marp: true
theme: mila
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Modèle de Bradley-Terry
## Des préférences aux variables latentes

*Pierre-Luc Bacon*
IFT6390 -- Fondements de l'apprentissage machine

---

## Plan de la présentation

1. **Préférences et comparaisons par paires** : motivation, format des données
2. **Le modèle de Bradley-Terry** : scores latents et sigmoïde
3. **Lien avec la régression logistique** : matrice de conception et estimation
4. **Perspective à variables latentes** : modèle de Thurstone, bruit de Gumbel, processus génératif
5. **Extension bayésienne** : a priori sur les scores et régularisation
6. **Connexions** : parallèle avec les GMM et le fil conducteur du cours

---

<!-- _class: lead -->

# Préférences et comparaisons par paires
## Transformer des jugements qualitatifs en scores

---

<!-- footer: "📖 Chapitre 3 : Classification" -->

## Pourquoi modéliser des préférences?

Dans de nombreuses situations, nous n'observons pas de mesures absolues, mais des **comparaisons relatives** :

| Domaine | Comparaison | Résultat |
|---------|-------------|----------|
| Échecs | Joueur A vs Joueur B | Qui gagne? |
| RLHF | Réponse A vs Réponse B | Laquelle est meilleure? |
| Acoustique | Son A vs Son B | Lequel est préféré? |
| Recherche web | Résultat A vs Résultat B | Lequel est cliqué? |

Le défi : convertir ces jugements qualitatifs en **scores numériques** utilisables pour l'optimisation.

---

## Format des données de préférence

Chaque observation est un triplet : deux objets et un résultat.

**Exemple** : Quatre joueurs, cinq matchs observés.

| Comparaison | Gagnant | $y$ |
|-------------|---------|-----|
| 0 vs 2 | 0 | 1 |
| 1 vs 3 | 1 | 1 |
| 0 vs 1 | 1 | 0 |
| 2 vs 3 | 2 | 1 |
| 1 vs 2 | 1 | 1 |

Nous observons des victoires et des défaites, mais pas la « force » de chaque joueur. Comment inférer ces forces à partir des résultats?

---

## Des jugements qualitatifs aux scores quantitatifs

Nous cherchons un **score** $s_k$ pour chaque objet $k$ tel que :

- Un score élevé = un objet de meilleure qualité
- La **différence** de scores prédit le résultat d'une comparaison
- Les scores permettent de classer tous les objets sur une échelle unique

```
  Préférences observées         Scores inférés
  ┌──────────────────┐         ┌──────────────┐
  │  A > B           │         │  s_A = 2.1   │
  │  A > C           │  ───►   │  s_B = 0.8   │
  │  B > C           │         │  s_C = -0.3  │
  │  C > D           │         │  s_D = -1.5  │
  └──────────────────┘         └──────────────┘
```

Le modèle de Bradley-Terry formalise cette idée.

---

<!-- _class: lead -->

# Le modèle de Bradley-Terry
## Scores latents et probabilités de préférence

---

<!-- footer: "📖 Chapitre 3 : Classification" -->

## Formulation du modèle

Chaque objet $k$ possède un score latent $s_k$. La probabilité que $A$ soit préféré à $B$ est :

$$\boxed{P(A \succ B) = \sigma(s_A - s_B) = \frac{1}{1 + e^{-(s_A - s_B)}}}$$

La sigmoïde transforme la différence de scores en probabilité :

| Différence $s_A - s_B$ | $\sigma(s_A - s_B)$ | Interprétation |
|-------------------------|---------------------|----------------|
| $+\infty$ | $\approx 1$ | $A$ gagne presque certainement |
| $0$ | $0{,}5$ | Pile ou face |
| $-\infty$ | $\approx 0$ | $B$ gagne presque certainement |

Les scores sont dits **latents** car nous ne les observons pas directement.

---

## Identifiabilité des scores

Les scores sont définis à une **constante additive** près.

Ajouter $c$ à tous les scores ne change pas les probabilités :

$$\sigma((s_A + c) - (s_B + c)) = \sigma(s_A - s_B)$$

| Propriété | Conséquence |
|-----------|-------------|
| Définis à une constante près | $s_k + c$ donne les mêmes probabilités |
| Interprétables par différence | $s_A - s_B = 1$ implique environ 73% pour $A$ |
| Log-cote de préférence | $s_A - s_B = \log \frac{P(A \succ B)}{P(B \succ A)}$ |

En pratique, on centre les scores : $\sum_k s_k = 0$. C'est analogue au fait que le potentiel électrique est défini à une constante près.

---

## Le classement Elo aux échecs

Le système **Elo**, utilisé aux échecs depuis les années 1960, repose sur le même modèle.

$$P(A \succ B) = \frac{1}{1 + 10^{-(R_A - R_B)/400}}$$

| Écart de classement | $P(A \succ B)$ |
|---------------------|-----------------|
| $+400$ | $\approx 91\%$ |
| $+200$ | $\approx 76\%$ |
| $0$ | $50\%$ |
| $-200$ | $\approx 24\%$ |

C'est le modèle de Bradley-Terry avec une base 10 et un facteur d'échelle de 400.

---

<!-- _class: lead -->

# Lien avec la régression logistique
## Une matrice de conception astucieuse

---

## Construction de la matrice de conception

Pour $K$ objets, construisons un vecteur $\mathbf{x}_{ij} \in \mathbb{R}^K$ par comparaison :

$$x_{ij,k} = \begin{cases} +1 & \text{si } k = i \\ -1 & \text{si } k = j \\ 0 & \text{sinon} \end{cases}$$

Le produit scalaire avec le vecteur de scores $\mathbf{s}$ donne :

$$\mathbf{s}^\top \mathbf{x}_{ij} = s_i - s_j$$

Le modèle prédit $P(y=1 \mid \mathbf{x}_{ij}) = \sigma(\mathbf{s}^\top \mathbf{x}_{ij})$ : c'est une **régression logistique** sans intercept. Les coefficients sont directement les scores.

---

## Exemple : 4 joueurs, 5 matchs

$$\mathbf{X} = \begin{pmatrix} 1 & 0 & -1 & 0 \\ 0 & 1 & 0 & -1 \\ 1 & -1 & 0 & 0 \\ 0 & 0 & 1 & -1 \\ 0 & 1 & -1 & 0 \end{pmatrix}, \quad \mathbf{y} = \begin{pmatrix} 1 \\ 1 \\ 0 \\ 1 \\ 1 \end{pmatrix}$$

Chaque ligne encode un match : $+1$ pour le premier joueur, $-1$ pour le second. L'étiquette $y = 1$ indique que le premier joueur a gagné, $y = 0$ que le second a gagné.

$$\boldsymbol{\theta}^\top \mathbf{x}_{ij} = \theta_i - \theta_j = s_i - s_j$$

$$\boxed{\theta_k = s_k \quad \text{pour tout } k}$$

---

## Estimation par maximum de vraisemblance

La log-vraisemblance est celle de la régression logistique :

$$\ell(\mathbf{s}) = \sum_{m=1}^M \left[ y_m \log \sigma(s_{i_m} - s_{j_m}) + (1-y_m) \log(1 - \sigma(s_{i_m} - s_{j_m})) \right]$$

| Propriété | Détail |
|-----------|--------|
| **Convexe** | Un unique maximum global |
| **Pas de solution analytique** | Optimisation itérative nécessaire |
| **Identifiabilité** | Scores définis à une constante près |

En pratique : `LogisticRegression(fit_intercept=False)` de scikit-learn.

---

## Code : estimation des scores

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Matrice de conception (+1/-1) et étiquettes
X = np.array([[ 1, 0,-1, 0],   # 0 vs 2
              [ 0, 1, 0,-1],   # 1 vs 3
              [ 1,-1, 0, 0],   # 0 vs 1
              [ 0, 0, 1,-1],   # 2 vs 3
              [ 0, 1,-1, 0]])  # 1 vs 2
y = np.array([1, 1, 0, 1, 1])

model = LogisticRegression(fit_intercept=False, C=1e6)
model.fit(X, y)
scores = model.coef_[0]
scores -= scores.mean()  # Centrer les scores
```

Avec plus de matchs, les scores estimés convergent vers les vrais scores.

---

## Application : RLHF pour les modèles de langage

Le RLHF utilise Bradley-Terry pour entraîner un **modèle de récompense** :

| Étape | Description |
|-------|-------------|
| 1. Collecte | Des annotateurs comparent des paires de réponses |
| 2. Récompense | Bradley-Terry estime un score par réponse |
| 3. Optimisation | Le modèle est entraîné à maximiser ce score |

Au lieu d'un score par objet, un réseau $r_\phi(\text{réponse})$ attribue un score à chaque réponse :

$$P(A \succ B) = \sigma(r_\phi(A) - r_\phi(B))$$

Le vecteur de scores $\mathbf{s}$ est remplacé par une **fonction paramétrique**.

---

<!-- _class: lead -->

# Perspective à variables latentes
## Le modèle de Thurstone et le bruit de Gumbel

---

<!-- footer: "📖 Chapitres 5-6 : Cadre probabiliste et modèles génératifs" -->

## Variables latentes : une idée récurrente

Au chapitre 6, nous avons vu les GMM et leur interprétation à variables latentes :

| Élément | GMM | Bradley-Terry |
|---------|-----|---------------|
| Observé | Donnée $\mathbf{x}$ | Résultat du match |
| Latent | Composant $z \in \{1, \ldots, K\}$ | ??? |

Pour les GMM, la variable latente $z$ indique de quel composant provient chaque observation. En la marginalisant, on obtient la densité de mélange.

Peut-on trouver une structure analogue pour Bradley-Terry? Quelles sont les variables latentes derrière les résultats de comparaisons?

---

## Le modèle de Thurstone (1927)

L.L. Thurstone a proposé cette idée : chaque objet $i$ possède un score moyen $s_i$, mais sa **performance** à chaque comparaison est bruitée :

$$Z_i = s_i + \epsilon_i$$

où $\epsilon_i$ est un bruit aléatoire indépendant, tiré à chaque comparaison.

Le joueur $i$ bat le joueur $j$ quand sa performance dépasse celle de $j$ :

$$i \succ j \iff Z_i > Z_j$$

Les performances $Z_i, Z_j$ sont les **variables latentes**. Nous n'observons que le résultat ($i$ gagne ou $j$ gagne), jamais les performances elles-mêmes.

---

## Choix du bruit : Gumbel vs gaussien

La distribution du bruit $\epsilon$ détermine le modèle de préférence :

| Bruit $\epsilon$ | Modèle | $P(i \succ j)$ |
|-------------------|--------|-----------------|
| Gumbel$(0, 1)$ | **Bradley-Terry** | $\sigma(s_i - s_j)$ |
| $\mathcal{N}(0, \sigma^2)$ | Thurstone Case V | $\Phi\!\left(\frac{s_i - s_j}{\sigma\sqrt{2}}\right)$ |

Avec le bruit de Gumbel, on retrouve **exactement** le modèle de Bradley-Terry.

Avec le bruit gaussien, on obtient un modèle probit. En pratique, les deux donnent des résultats très similaires : la sigmoïde et la CDF gaussienne ont des formes proches.

---

## Pourquoi le bruit de Gumbel donne la sigmoïde

Si $\epsilon_i, \epsilon_j \overset{\text{i.i.d.}}{\sim} \text{Gumbel}(0, 1)$, alors la différence $\epsilon_i - \epsilon_j$ suit une **distribution logistique** :

$$P(\epsilon_i - \epsilon_j \leq t) = \sigma(t) = \frac{1}{1 + e^{-t}}$$

La probabilité que $i$ batte $j$ :

$$P(Z_i > Z_j) = P(s_i + \epsilon_i > s_j + \epsilon_j) = P(\epsilon_j - \epsilon_i < s_i - s_j)$$

$$\boxed{P(i \succ j) = \sigma(s_i - s_j)}$$

La sigmoïde du modèle de Bradley-Terry provient de la différence de deux variables de Gumbel.

---

## La distribution de Gumbel

La distribution de Gumbel (type I, ou distribution des valeurs extrêmes) :

$$f(\epsilon) = e^{-\epsilon - e^{-\epsilon}}, \qquad F(\epsilon) = e^{-e^{-\epsilon}}$$

| Propriété | Valeur |
|-----------|--------|
| Espérance | $\gamma \approx 0{,}577$ (constante d'Euler) |
| Variance | $\pi^2/6 \approx 1{,}645$ |
| Mode | $0$ |
| Queue | Asymétrique, queue droite lourde |

Cette distribution apparaît naturellement dans la théorie des valeurs extrêmes et dans les modèles de choix discret (logit multinomial).

---

## Le processus génératif

Le modèle de Thurstone avec bruit de Gumbel définit un processus de génération :

**Pour chaque comparaison entre $i$ et $j$ :**
1. Tirer une performance $Z_i = s_i + \epsilon_i$, avec $\epsilon_i \sim \text{Gumbel}(0,1)$
2. Tirer une performance $Z_j = s_j + \epsilon_j$, avec $\epsilon_j \sim \text{Gumbel}(0,1)$
3. Observer le résultat : $y = \mathbb{1}[Z_i > Z_j]$

Comparons avec le processus génératif du GMM :

**Pour chaque observation $\mathbf{x}_n$ :**
1. Tirer un composant $z_n \sim \text{Catégorielle}(\boldsymbol{\pi})$
2. Tirer $\mathbf{x}_n \sim \mathcal{N}(\boldsymbol{\mu}_{z_n}, \boldsymbol{\Sigma}_{z_n})$

Dans les deux cas, les variables latentes rendent les observations faciles à modéliser.

---

## Marginaliser les variables latentes

La même structure dans les deux modèles :

**GMM** : marginaliser $z$ donne la densité de mélange

$$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

**Bradley-Terry** : marginaliser les performances donne la probabilité de préférence

$$P(i \succ j) = \int\!\!\int \mathbb{1}[z_i > z_j] \, p(z_i \mid s_i) \, p(z_j \mid s_j) \, dz_i \, dz_j = \sigma(s_i - s_j)$$

Dans les deux cas, la marginalisation des latentes produit la vraisemblance observée. La difficulté d'estimation vient de cette marginalisation.

---

## Le parallèle structurel

| Aspect | GMM | Bradley-Terry |
|--------|-----|---------------|
| Variable latente | $z \in \{1, \ldots, K\}$ | Performances $Z_i, Z_j \in \mathbb{R}$ |
| Nature de la latente | Discrète | Continue |
| Observation | $\mathbf{x} \in \mathbb{R}^D$ | $y \in \{0, 1\}$ (qui gagne) |
| Paramètres | $\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, \pi_k$ | Scores $s_1, \ldots, s_K$ |
| Marginalisation | $\sum_k$ (somme sur composants) | $\iint$ (intégrale sur performances) |
| Résultat | Densité de mélange | Sigmoïde de la différence |

La même idée -- marginaliser des variables latentes pour obtenir la vraisemblance -- apparaît dans des contextes très différents.

---

## La vraisemblance des données complètes

Si nous observions les performances $(Z_i, Z_j)$ en plus du résultat, l'estimation serait simple.

La log-vraisemblance des données complètes se factorise :

$$\log p(Z_i, Z_j, y \mid \mathbf{s}) = \log p(Z_i \mid s_i) + \log p(Z_j \mid s_j) + \log p(y \mid Z_i, Z_j)$$

Le dernier terme est déterministe : $y = \mathbb{1}[Z_i > Z_j]$. Les deux premiers sont des densités de Gumbel. L'estimation se réduit à estimer les paramètres de position de distributions de Gumbel.

**Mais nous n'observons pas les performances!** C'est le même problème que dans le GMM : les latentes manquent.

---

## EM pour Bradley-Terry

L'algorithme EM s'applique naturellement :

**Étape E** : Calculer la distribution a posteriori des performances, sachant les résultats observés et les scores courants

$$p(Z_i, Z_j \mid y, \mathbf{s}^{(t)}) \propto p(Z_i \mid s_i^{(t)}) \, p(Z_j \mid s_j^{(t)}) \, \mathbb{1}[y = \mathbb{1}(Z_i > Z_j)]$$

**Étape M** : Mettre à jour les scores en maximisant l'espérance de la log-vraisemblance complète sous la distribution E

| Approche | GMM | Bradley-Terry |
|----------|-----|---------------|
| EM (E) | Responsabilités $r_{nk}$ | Distribution a posteriori de $(Z_i, Z_j)$ |
| EM (M) | Moyennes/covariances pondérées | Mise à jour des scores |
| En pratique | EM est la méthode standard | Descente de gradient suffit |

---

<!-- _class: lead -->

# Extension bayésienne
## A priori sur les scores et régularisation

---

<!-- footer: "📖 Chapitre 5 : Le cadre probabiliste" -->

## L'approche bayésienne pour les scores

Le chapitre 5 a montré que l'a priori gaussien mène à la régularisation Ridge. La même idée s'applique ici.

**A priori** : $p(\mathbf{s}) = \mathcal{N}(\mathbf{0}, \tau^2 \mathbf{I})$

**MAP** : $\hat{\mathbf{s}}_{\text{MAP}} = \arg\max_{\mathbf{s}} \left[\log p(\text{données} \mid \mathbf{s}) + \log p(\mathbf{s})\right]$

$$= \arg\max_{\mathbf{s}} \left[ \ell(\mathbf{s}) - \frac{1}{2\tau^2} \|\mathbf{s}\|^2 \right]$$

L'a priori gaussien pénalise les scores extrêmes, exactement comme Ridge régularise les coefficients de régression.

---

## Effet de la régularisation

| $\tau^2$ | Régularisation | Effet sur les scores |
|----------|----------------|----------------------|
| Grand | Faible | Scores proches de l'EMV |
| Petit | Forte | Scores proches de zéro |

**Sans régularisation** : Si un joueur gagne toutes ses parties, son score tend vers $+\infty$ (séparation complète, comme en régression logistique).

**Avec régularisation** : Les scores restent bornés, même avec peu de comparaisons.

En RLHF, cette régularisation empêche le modèle de récompense d'attribuer des scores extrêmes, ce qui stabilise l'entraînement du modèle de langage.

---

## Incertitude sur les scores

L'approche bayésienne complète donne une **distribution a posteriori** sur les scores :

$$p(\mathbf{s} \mid \text{données}) \propto p(\text{données} \mid \mathbf{s}) \, p(\mathbf{s})$$

Cette distribution n'a pas de forme analytique, mais on peut l'approximer par :
- L'**approximation de Laplace** (gaussienne centrée au MAP)
- Des **méthodes de Monte Carlo** (MCMC)

L'incertitude est utile : un joueur avec 2 matchs aura une large incertitude sur son score, un joueur avec 200 matchs une incertitude faible. Cela guide les comparaisons futures : comparer des joueurs incertains est plus informatif.

---

<!-- _class: lead -->

# Connexions et résumé
## Le fil conducteur des variables latentes

---

<!-- footer: "" -->

## Le fil conducteur : variables latentes

Un thème récurrent du cours : des variables **non observées** structurent les données.

| Modèle | Latent | Observé | Lien |
|--------|--------|---------|------|
| **Régression** (ch2) | Bruit $\epsilon \sim \mathcal{N}(0, \sigma^2)$ | $(x, y)$ | $y = f(x) + \epsilon$ |
| **GMM** (ch6) | Composant $z$ | $\mathbf{x}$ | $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_z, \boldsymbol{\Sigma}_z)$ |
| **Bradley-Terry** | Performances $Z_i, Z_j$ | Qui gagne | $y = \mathbb{1}[Z_i > Z_j]$ |

La marginalisation des variables latentes produit des modèles probabilistes dont l'estimation nécessite des outils adaptés (EM, descente de gradient, MCMC).

---

## Trois perspectives sur Bradley-Terry

| Perspective | Formulation | Chapitre |
|-------------|-------------|----------|
| **Régression logistique** | $P(A \succ B) = \sigma(\mathbf{s}^\top \mathbf{x})$ avec matrice $\pm 1$ | Ch. 3 |
| **Variable latente** | Performances bruitées $Z_i = s_i + \epsilon_i$, marginalisation | Ch. 6 |
| **Bayésienne** | A priori $p(\mathbf{s})$, a posteriori $p(\mathbf{s} \mid \text{données})$ | Ch. 5 |

Ces perspectives se complètent :
- La vue logistique donne un **algorithme** d'estimation efficace
- La vue latente donne une **interprétation** du processus de génération des données
- La vue bayésienne donne des **outils** pour la régularisation et la quantification de l'incertitude

---

## Résumé : le modèle de Bradley-Terry

| Concept | Formule |
|---------|---------|
| Probabilité de préférence | $P(A \succ B) = \sigma(s_A - s_B)$ |
| Matrice de conception | $+1$ pour $A$, $-1$ pour $B$, $0$ ailleurs |
| Estimation | Régression logistique sans intercept |
| Identifiabilité | Scores définis à une constante près |

$$\boxed{P(A \succ B) = \sigma(s_A - s_B) = \frac{1}{1 + e^{-(s_A - s_B)}}}$$

---

## Résumé : perspective à variables latentes

| Concept | Formule |
|---------|---------|
| Performance latente | $Z_i = s_i + \epsilon_i$, $\epsilon_i \sim \text{Gumbel}(0, 1)$ |
| Résultat observé | $y = \mathbb{1}[Z_i > Z_j]$ |
| Marginalisation | $P(Z_i > Z_j) = \sigma(s_i - s_j)$ |
| Régularisation MAP | $\hat{\mathbf{s}}_{\text{MAP}} = \arg\max [\ell(\mathbf{s}) - \frac{1}{2\tau^2}\|\mathbf{s}\|^2]$ |

Le modèle de Thurstone avec bruit de Gumbel donne une justification probabiliste du modèle de Bradley-Terry, en parallèle direct avec les modèles à variables latentes du chapitre 6.

---

<!-- _class: lead -->

# Questions?

**Lectures recommandées** :
- Chapitre 3 : Section « Préférences et le modèle de Bradley-Terry »
- Chapitre 5 : Le cadre probabiliste (EMV, MAP, divergence KL)
- Chapitre 6 : Modèles probabilistes génératifs (GMM, algorithme EM)
