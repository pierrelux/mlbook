---
marp: true
theme: mila
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Variables latentes et représentations
## Bradley-Terry, ACP probabiliste et autoencodeurs

*Pierre-Luc Bacon*
IFT6390 – Fondements de l'apprentissage machine

---

## Plan de la présentation

1. **Modèle de Bradley-Terry** : préférences, scores latents et régression logistique
2. **ACP : rappel géométrique** : reconstruction, variance et valeurs propres
3. **ACP probabiliste** : modèle génératif à variables latentes
4. **Autoencodeurs** : de la projection linéaire aux représentations apprises

---

<!-- _class: lead -->

# Modèle de Bradley-Terry
## Des comparaisons par paires aux scores latents

---

<!-- footer: "📖 Chapitre 3 : Classification" -->

## Le problème des préférences

Dans plusieurs situations, nous n'observons pas de mesure absolue mais des **comparaisons relatives** :

| Domaine | Comparaison |
|---------|-------------|
| Échecs | Joueur A bat joueur B |
| Acoustique | Son A préféré au son B |
| Moteurs de recherche | Clic sur le résultat A plutôt que B |
| Modèles de langage (RLHF) | Réponse A jugée meilleure que B |

Comment convertir ces jugements qualitatifs en **scores numériques**?

---

## Scores latents et sigmoïde

L'idée est d'attribuer un score $s_k$ à chaque objet $k$, puis de modéliser la probabilité que $A$ soit préféré à $B$ par

$$P(A \succ B) = \sigma(s_A - s_B) = \frac{1}{1 + e^{-(s_A - s_B)}}$$

| Différence $s_A - s_B$ | $P(A \succ B)$ | Interprétation |
|-------------------------|-----------------|----------------|
| $\gg 0$ | $\approx 1$ | $A$ domine |
| $\approx 0$ | $\approx 0{,}5$ | Résultat incertain |
| $\ll 0$ | $\approx 0$ | $B$ domine |

Les scores $s_k$ sont **latents** : on ne les observe pas directement, on les infère à partir des résultats des comparaisons.

---

## Lien avec la régression logistique

Pour $K$ objets, construisons un vecteur $\mathbf{x}_{ij} \in \mathbb{R}^K$ qui encode la comparaison entre $i$ et $j$ :

$$x_{ij,k} = \begin{cases} +1 & \text{si } k = i \\ -1 & \text{si } k = j \\ 0 & \text{sinon} \end{cases}$$

Le produit scalaire avec le vecteur de scores donne :

$$\mathbf{s}^\top \mathbf{x}_{ij} = s_i - s_j$$

C'est exactement le logit d'une **régression logistique** sans biais, avec les scores comme paramètres.

$$\boxed{P(A \succ B) = \sigma(\boldsymbol{\theta}^\top \mathbf{x}_{ij}), \quad \text{où } \theta_k = s_k}$$

---

## Matrice de conception : exemple concret

Quatre joueurs, cinq comparaisons :

| Match | Gagnant | $y$ |
|-------|---------|-----|
| 0 vs 2 | 0 | 1 |
| 1 vs 3 | 1 | 1 |
| 0 vs 1 | 1 | 0 |
| 2 vs 3 | 2 | 1 |
| 1 vs 2 | 1 | 1 |

$$\mathbf{X} = \begin{pmatrix} 1 & 0 & -1 & 0 \\ 0 & 1 & 0 & -1 \\ 1 & -1 & 0 & 0 \\ 0 & 0 & 1 & -1 \\ 0 & 1 & -1 & 0 \end{pmatrix}, \quad \mathbf{y} = \begin{pmatrix} 1 \\ 1 \\ 0 \\ 1 \\ 1 \end{pmatrix}$$

---

## Estimation et identifiabilité

L'estimation se fait par maximum de vraisemblance, avec la même log-vraisemblance que la régression logistique :

$$\hat{\mathbf{s}} = \arg\min_{\mathbf{s}} -\sum_{n=1}^{N} \left[ y_n \log \sigma(\mathbf{s}^\top \mathbf{x}_n) + (1-y_n) \log(1 - \sigma(\mathbf{s}^\top \mathbf{x}_n)) \right]$$

Les scores sont définis **à une constante additive près** : ajouter $c$ à tous les scores ne change pas les probabilités, car seule la différence $s_i - s_j$ intervient.

| Convention | Résultat |
|------------|----------|
| Centrer : $\sum_k s_k = 0$ | Le plus courant |
| Fixer un score de référence : $s_1 = 0$ | Aussi valable |

---

## Implémentation avec scikit-learn

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Matrice de conception (chaque ligne : +1 pour i, -1 pour j)
X = np.array([[ 1, 0,-1, 0],
              [ 0, 1, 0,-1],
              [ 1,-1, 0, 0],
              [ 0, 0, 1,-1],
              [ 0, 1,-1, 0]])
y = np.array([1, 1, 0, 1, 1])

model = LogisticRegression(fit_intercept=False, C=1e6)
model.fit(X, y)

scores = model.coef_[0]
scores -= scores.mean()  # Centrer les scores
```

Les coefficients $\hat{\boldsymbol{\theta}}$ sont directement les scores $\hat{\mathbf{s}}$ des objets.

---

## Le système Elo

Le classement Elo, utilisé aux échecs depuis les années 1960, repose sur le même modèle.

$$P(\text{A gagne}) = \frac{1}{1 + 10^{-(R_A - R_B)/400}}$$

La base 10 et le facteur 400 sont des conventions historiques. La structure est identique :

| Écart de classement | Probabilité de victoire |
|---------------------|------------------------|
| 0 points | 50% |
| 200 points | 76% |
| 400 points | 91% |

Après chaque partie, les classements sont mis à jour par une règle de gradient :
$$R_A \leftarrow R_A + K(y - P(\text{A gagne}))$$

C'est une descente de gradient stochastique avec un seul exemple.

---

## RLHF : aligner les modèles de langage

Le RLHF (*Reinforcement Learning from Human Feedback*) utilise Bradley-Terry pour entraîner une **fonction de récompense** :

1. Un modèle de langage génère deux réponses $A$ et $B$ à la même question
2. Un annotateur humain indique sa préférence ($A \succ B$ ou $B \succ A$)
3. On entraîne un modèle de récompense $r_\phi(\text{réponse})$ par Bradley-Terry :
   $$P(A \succ B) = \sigma(r_\phi(A) - r_\phi(B))$$
4. Le modèle de langage est optimisé pour maximiser cette récompense

Le score $r_\phi$ joue le rôle du score latent $s_k$, mais il est paramétré par un réseau de neurones appliqué au texte de la réponse.

---

## Application : préférences acoustiques

Des auditeurs comparent des champs sonores dans une salle de concert ({cite:t}`kousgaard1984sound`).

Trois facteurs binaires → $2^3 = 8$ configurations :

| Config. | Son direct | Réflexion | Réverbération |
|---------|------------|-----------|---------------|
| 000 | Non | Faible | Faible |
| 111 | Oui | Élevée | Élevée |
| ... | ... | ... | ... |

La matrice de conception est construite comme avant : $+1$ pour la configuration A, $-1$ pour B. Les scores estimés classent les 8 configurations par qualité perçue.

La régression logistique standard suffit pour résoudre ce problème d'acoustique psychologique.

---

## Bradley-Terry : résumé

| Élément | Détail |
|---------|--------|
| Modèle | $P(A \succ B) = \sigma(s_A - s_B)$ |
| Estimation | Régression logistique sans biais |
| Matrice de conception | $+1$ pour $i$, $-1$ pour $j$, $0$ ailleurs |
| Identifiabilité | Scores définis à une constante près |
| Applications | Elo, RLHF, recommandation, acoustique |

La variable latente ici est le **score** de chaque objet, inféré à partir des comparaisons observées.

Cette idée de variable latente va réapparaître sous une forme différente avec l'ACP probabiliste.

---

<!-- _class: lead -->

# ACP : rappel géométrique
## Reconstruction, variance et valeurs propres

---

<!-- footer: "📖 Chapitre sur l'ACP" -->

## Le problème de la réduction de dimensionnalité

Les données vivent souvent dans un espace de dimension $D$, mais leur **structure** occupe un sous-espace de dimension bien plus petite.

| Situation | $D$ (dimension ambiante) | Structure réelle |
|-----------|--------------------------|------------------|
| Images 28×28 | 784 | Variété de basse dimension |
| Texte (vocabulaire) | 50 000+ | Thèmes, sujets |
| Capteurs redondants | $D$ | Quelques facteurs physiques |

L'ACP cherche un sous-espace linéaire de dimension $L \ll D$ qui **préserve au mieux** l'information contenue dans les données.

---

## L'ACP comme autoencodeur linéaire

Étant donné des données centrées $\mathbf{x}_n \in \mathbb{R}^D$, l'ACP cherche :

| Étape | Opération | Dimension |
|-------|-----------|-----------|
| Encodage | $\mathbf{z}_n = \mathbf{W}^\top \mathbf{x}_n$ | $\mathbb{R}^D \to \mathbb{R}^L$ |
| Décodage | $\hat{\mathbf{x}}_n = \mathbf{W} \mathbf{z}_n$ | $\mathbb{R}^L \to \mathbb{R}^D$ |

La matrice $\mathbf{W} \in \mathbb{R}^{D \times L}$ satisfait $\mathbf{W}^\top \mathbf{W} = \mathbf{I}_L$ (colonnes orthonormales).

L'erreur de reconstruction à minimiser :

$$\mathcal{L}(\mathbf{W}) = \frac{1}{N} \sum_{n=1}^N \|\mathbf{x}_n - \mathbf{W}\mathbf{W}^\top \mathbf{x}_n\|^2$$

---

## Solution : vecteurs propres de la covariance

La matrice de covariance empirique est $\hat{\boldsymbol{\Sigma}} = \frac{1}{N} \sum_n \mathbf{x}_n \mathbf{x}_n^\top$.

Le problème se ramène à un problème de valeurs propres :

$$\hat{\boldsymbol{\Sigma}} \mathbf{w}_k = \lambda_k \mathbf{w}_k$$

La solution optimale : prendre les $L$ vecteurs propres associés aux $L$ **plus grandes** valeurs propres $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_D$.

| Propriété | Formule |
|-----------|---------|
| Variance projetée sur $\mathbf{w}_k$ | $\lambda_k$ |
| Erreur de reconstruction | $\sum_{k=L+1}^{D} \lambda_k$ |
| Variance expliquée | $\frac{\sum_{k=1}^L \lambda_k}{\sum_{k=1}^D \lambda_k}$ |

---

## Deux formulations équivalentes

$$\underbrace{\min_{\mathbf{W}} \frac{1}{N} \sum_n \|\mathbf{x}_n - \mathbf{W}\mathbf{W}^\top\mathbf{x}_n\|^2}_{\text{Minimiser l'erreur de reconstruction}} \iff \underbrace{\max_{\mathbf{W}} \sum_{k=1}^L \mathbf{w}_k^\top \hat{\boldsymbol{\Sigma}} \mathbf{w}_k}_{\text{Maximiser la variance projetée}}$$

sous $\mathbf{W}^\top \mathbf{W} = \mathbf{I}_L$.

L'ACP trouve les directions le long desquelles les données **varient le plus**.

Rappel du lien avec la DVS : si $\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$, les colonnes de $\mathbf{V}$ sont les vecteurs propres de $\mathbf{X}^\top\mathbf{X}$, et $\lambda_k = d_k^2 / N$.

---

## Ridge vs ACP : seuillage doux vs dur

Deux façons de traiter les directions à faible signal (petites valeurs singulières) :

| Approche | Traitement | Type |
|----------|------------|------|
| Ridge | Rétrécit par $\frac{d_j^2}{d_j^2 + \lambda}$ | Seuillage doux (garde tout) |
| ACP | Élimine si $j > L$ | Seuillage dur (coupe net) |

Ridge convient à la **régression supervisée** : on veut prédire $y$.

L'ACP convient à la **réduction de dimension non supervisée** : on veut comprimer $\mathbf{x}$.

Peut-on combiner les avantages? Un modèle probabiliste de l'ACP va nous donner ce cadre.

---

<!-- _class: lead -->

# ACP probabiliste
## Un modèle génératif à variables latentes

---

<!-- footer: "📖 Chapitre sur l'ACP – Perspective probabiliste" -->

## D'un algorithme à un modèle

L'ACP classique est un **algorithme d'optimisation** : minimiser la reconstruction, prendre les vecteurs propres.

Changeons de perspective. Posons-nous la question : existe-t-il un **modèle probabiliste** dont l'ACP serait la solution par maximum de vraisemblance?

| Perspective | Question |
|-------------|----------|
| ACP classique | Quel sous-espace minimise l'erreur? |
| ACP probabiliste | Quel processus génératif explique les données? |

Cette reformulation va nous donner des outils supplémentaires : gestion du bruit, données manquantes, sélection de modèle, et surtout une **voie vers les autoencodeurs**.

---

## Le modèle génératif

L'ACP probabiliste (PPCA) suppose que les données sont générées par le processus suivant :

**1. Variable latente** (cause cachée) :
$$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_L)$$

**2. Observation** (mesure bruitée) :
$$\mathbf{x} \mid \mathbf{z} \sim \mathcal{N}(\mathbf{W}\mathbf{z} + \boldsymbol{\mu}, \sigma^2 \mathbf{I}_D)$$

| Paramètre | Rôle | Dimension |
|-----------|------|-----------|
| $\mathbf{W}$ | Matrice de chargement | $D \times L$ |
| $\boldsymbol{\mu}$ | Moyenne des données | $D \times 1$ |
| $\sigma^2$ | Variance du bruit d'observation | scalaire |

---

## Processus de génération

Pour générer une observation $\mathbf{x}$ :

1. Tirer une **cause latente** $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_L)$ dans l'espace de dimension $L$
2. Projeter dans l'espace de dimension $D$ : $\mathbf{W}\mathbf{z} + \boldsymbol{\mu}$
3. Ajouter du **bruit d'observation** $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_D)$

$$\mathbf{x} = \mathbf{W}\mathbf{z} + \boldsymbol{\mu} + \boldsymbol{\epsilon}$$

La donnée observée est une **version bruitée** de la projection linéaire d'un vecteur latent de basse dimension. Le bruit $\sigma^2$ capture tout ce que le sous-espace de dimension $L$ ne peut pas expliquer.

---

## Distribution marginale des observations

Puisque $\mathbf{z}$ et $\boldsymbol{\epsilon}$ sont gaussiens, la distribution marginale de $\mathbf{x}$ est aussi gaussienne :

$$p(\mathbf{x}) = \int p(\mathbf{x} \mid \mathbf{z}) \, p(\mathbf{z}) \, d\mathbf{z} = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \mathbf{C})$$

avec la matrice de covariance :

$$\boxed{\mathbf{C} = \mathbf{W}\mathbf{W}^\top + \sigma^2 \mathbf{I}_D}$$

| Terme | Source | Structure |
|-------|--------|-----------|
| $\mathbf{W}\mathbf{W}^\top$ | Signal (rang $L$) | Matrice de rang faible |
| $\sigma^2 \mathbf{I}_D$ | Bruit | Isotrope |

Le modèle dit que la covariance des données est la somme d'une **composante de rang faible** (le signal) et d'un **bruit isotrope**.

---

## Inférence : retrouver le code latent

Étant donné une observation $\mathbf{x}$, quelle est la cause latente la plus probable?

Par Bayes, la **distribution a posteriori** est aussi gaussienne :

$$p(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\mathbf{z} \mid \mathbf{m}, \boldsymbol{\Sigma}_z)$$

avec :

$$\mathbf{M} = \mathbf{W}^\top \mathbf{W} + \sigma^2 \mathbf{I}_L$$
$$\mathbf{m} = \mathbf{M}^{-1} \mathbf{W}^\top (\mathbf{x} - \boldsymbol{\mu})$$
$$\boldsymbol{\Sigma}_z = \sigma^2 \mathbf{M}^{-1}$$

L'inversion porte sur une matrice $L \times L$, pas $D \times D$ : cela reste praticable même quand $D$ est grand.

---

## De l'inférence à l'encodage

Comparons l'encodage ACP classique et l'encodage PPCA :

| Méthode | Encodage | Effet du bruit |
|---------|----------|----------------|
| ACP classique | $\mathbf{z} = \mathbf{W}^\top \mathbf{x}$ | Ignoré |
| PPCA (moyenne a posteriori) | $\mathbf{z} = \mathbf{M}^{-1}\mathbf{W}^\top (\mathbf{x} - \boldsymbol{\mu})$ | Pris en compte |

La matrice $\mathbf{M}^{-1} = (\mathbf{W}^\top\mathbf{W} + \sigma^2\mathbf{I})^{-1}$ agit comme un **rétrécissement** : elle pondère chaque direction par le rapport signal/bruit.

Quand $\sigma^2 \to 0$, on retrouve $\mathbf{M}^{-1}\mathbf{W}^\top \to (\mathbf{W}^\top\mathbf{W})^{-1}\mathbf{W}^\top = \mathbf{W}^\top$ (puisque $\mathbf{W}$ est orthonormale dans ce cas).

---

## Estimation par maximum de vraisemblance

La log-vraisemblance du modèle est :

$$\log p(\mathbf{X} \mid \mathbf{W}, \boldsymbol{\mu}, \sigma^2) = -\frac{N}{2} \left[ D \log(2\pi) + \log|\mathbf{C}| + \text{tr}(\mathbf{C}^{-1}\hat{\boldsymbol{\Sigma}}) \right]$$

où $\hat{\boldsymbol{\Sigma}} = \frac{1}{N}\sum_n (\mathbf{x}_n - \boldsymbol{\mu})(\mathbf{x}_n - \boldsymbol{\mu})^\top$ est la covariance empirique.

La solution EMV admet une **forme fermée** :

$$\boxed{\hat{\mathbf{W}}_{\text{EMV}} = \mathbf{U}_L (\boldsymbol{\Lambda}_L - \sigma^2 \mathbf{I})^{1/2} \mathbf{R}}$$

où $\mathbf{U}_L$ contient les $L$ premiers vecteurs propres de $\hat{\boldsymbol{\Sigma}}$, $\boldsymbol{\Lambda}_L = \text{diag}(\lambda_1, \ldots, \lambda_L)$, et $\mathbf{R}$ est une rotation arbitraire.

---

## Estimation du bruit

La variance du bruit estimée par EMV est la moyenne des valeurs propres **résiduelles** :

$$\boxed{\hat{\sigma}^2 = \frac{1}{D - L} \sum_{j=L+1}^{D} \lambda_j}$$

| Quantité | Interprétation |
|----------|----------------|
| $\lambda_1, \ldots, \lambda_L$ | Variance expliquée par le signal |
| $\lambda_{L+1}, \ldots, \lambda_D$ | Variance résiduelle (bruit) |
| $\hat{\sigma}^2$ | Variance moyenne dans les directions ignorées |

Le modèle sépare automatiquement le **signal** (capturé par $\mathbf{W}$) du **bruit** (capturé par $\sigma^2$).

---

## Limite sans bruit : retour à l'ACP classique

Quand $\sigma^2 \to 0$ :

$$\hat{\mathbf{W}}_{\text{EMV}} \to \mathbf{U}_L \boldsymbol{\Lambda}_L^{1/2} \mathbf{R}$$

Le sous-espace engendré par les colonnes de $\hat{\mathbf{W}}$ est le **même** que celui de l'ACP classique (les $L$ premiers vecteurs propres de $\hat{\boldsymbol{\Sigma}}$).

$$\text{ACP classique} = \lim_{\sigma^2 \to 0} \text{PPCA}$$

La rotation $\mathbf{R}$ est libre : l'EMV détermine le **sous-espace**, pas une base particulière dans ce sous-espace. L'ACP classique fait le choix supplémentaire d'aligner les axes sur les vecteurs propres.

---

## Analyse factorielle : généralisation

L'analyse factorielle généralise PPCA en permettant un **bruit non isotrope** :

$$\mathbf{x} \mid \mathbf{z} \sim \mathcal{N}(\mathbf{W}\mathbf{z} + \boldsymbol{\mu}, \boldsymbol{\Psi})$$

| Modèle | Bruit $\boldsymbol{\Psi}$ | Paramètres de bruit |
|--------|--------------------------|---------------------|
| PPCA | $\sigma^2 \mathbf{I}$ | 1 (isotrope) |
| Analyse factorielle | $\text{diag}(\psi_1, \ldots, \psi_D)$ | $D$ (par dimension) |

L'analyse factorielle permet à chaque variable observée d'avoir son propre niveau de bruit. L'estimation requiert alors l'algorithme EM, car il n'y a pas de solution analytique directe.

---

## Pourquoi un modèle probabiliste?

| Avantage | Explication |
|----------|-------------|
| Estimation du bruit | $\hat{\sigma}^2$ quantifie ce que le modèle n'explique pas |
| Données manquantes | L'inférence bayésienne s'adapte naturellement |
| Sélection de $L$ | Comparer les vraisemblances pour différentes dimensions |
| Génération | On peut échantillonner de nouvelles données |
| Incertitude | La distribution a posteriori $p(\mathbf{z} \mid \mathbf{x})$ quantifie l'incertitude sur le code |

L'ACP classique ne donne qu'un point $\mathbf{z} = \mathbf{W}^\top\mathbf{x}$. PPCA donne une **distribution** $p(\mathbf{z} \mid \mathbf{x})$ sur les codes possibles.

---

## Résumé : trois niveaux de lecture de l'ACP

| Niveau | Formulation | Résultat |
|--------|-------------|----------|
| Géométrique | Minimiser la reconstruction | Vecteurs propres de $\hat{\boldsymbol{\Sigma}}$ |
| Statistique | Maximiser la variance projetée | Mêmes vecteurs propres |
| Probabiliste | Maximum de vraisemblance de PPCA | Même sous-espace + estimation du bruit |

Les trois niveaux donnent le même sous-espace, mais la perspective probabiliste offre un **cadre plus riche** pour la modélisation, l'inférence et les extensions.

---

<!-- _class: lead -->

# Des variables latentes aux représentations
## Autoencodeurs et apprentissage de représentations

---

<!-- footer: "📖 Vers les réseaux de neurones" -->

## Le schéma encodeur-décodeur

PPCA suit un schéma **encodeur-décodeur** :

$$\underbrace{\mathbf{x}}_{\text{observation}} \xrightarrow{\text{encodeur}} \underbrace{\mathbf{z}}_{\text{représentation}} \xrightarrow{\text{décodeur}} \underbrace{\hat{\mathbf{x}}}_{\text{reconstruction}}$$

Dans PPCA, les deux transformations sont **linéaires** :

| Rôle | PPCA |
|------|------|
| Encodeur | $\mathbf{z} = \mathbf{M}^{-1}\mathbf{W}^\top(\mathbf{x} - \boldsymbol{\mu})$ |
| Décodeur | $\hat{\mathbf{x}} = \mathbf{W}\mathbf{z} + \boldsymbol{\mu}$ |
| Code $\mathbf{z}$ | Gaussien, dimension $L$ |

Que se passe-t-il si on remplace les transformations linéaires par des **fonctions non linéaires**?

---

## L'autoencodeur non linéaire

Un **autoencodeur** remplace les projections linéaires par des réseaux de neurones :

| Rôle | PPCA (linéaire) | Autoencodeur (non linéaire) |
|------|------------------|-----------------------------|
| Encodeur | $\mathbf{W}^\top \mathbf{x}$ | $\mathbf{z} = f_\phi(\mathbf{x})$ |
| Décodeur | $\mathbf{W}\mathbf{z}$ | $\hat{\mathbf{x}} = g_\psi(\mathbf{z})$ |
| Paramètres | $\mathbf{W}$ (partagé) | $\phi, \psi$ (distincts) |

L'objectif reste la minimisation de la reconstruction :

$$\min_{\phi, \psi} \frac{1}{N} \sum_{n=1}^N \|\mathbf{x}_n - g_\psi(f_\phi(\mathbf{x}_n))\|^2$$

L'encodeur et le décodeur sont entraînés conjointement par rétropropagation.

---

## Architecture d'un autoencodeur

```
Entrée x ∈ ℝ^D
    ↓
┌─────────────────────┐
│   Encodeur f_ϕ      │   Couches de neurones
│   D → ... → L       │   avec non-linéarités
└─────────────────────┘
    ↓
Code z ∈ ℝ^L              ← goulot d'étranglement
    ↓
┌─────────────────────┐
│   Décodeur g_ψ      │   Couches de neurones
│   L → ... → D       │   avec non-linéarités
└─────────────────────┘
    ↓
Reconstruction x̂ ∈ ℝ^D
```

Le **goulot d'étranglement** ($L \ll D$) force le réseau à apprendre une représentation comprimée. Si l'encodeur et le décodeur sont linéaires (sans non-linéarités), on retrouve l'ACP.

---

## Trois familles de modèles

| Famille | Encodeur | Décodeur | Représentation |
|---------|----------|----------|----------------|
| ACP / PPCA | Linéaire | Linéaire | Fixée par les données |
| Autoencodeur | Réseau de neurones | Réseau de neurones | Apprise, déterministe |
| Autoencodeur variationnel (VAE) | Réseau → $\mu, \sigma$ | Réseau de neurones | Apprise, probabiliste |

L'évolution suit une progression :

- ACP → autoencodeur : on passe du **linéaire au non linéaire**
- Autoencodeur → VAE : on ajoute un **modèle probabiliste** (comme PPCA généralisait l'ACP)

---

## L'autoencodeur variationnel (VAE)

Le VAE combine l'architecture neuronale avec le modèle génératif de PPCA :

| Composante | PPCA | VAE |
|------------|------|-----|
| A priori | $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ | $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ |
| Décodeur | $p(\mathbf{x} \mid \mathbf{z}) = \mathcal{N}(\mathbf{W}\mathbf{z}, \sigma^2\mathbf{I})$ | $p_\psi(\mathbf{x} \mid \mathbf{z}) = \mathcal{N}(g_\psi(\mathbf{z}), \sigma^2\mathbf{I})$ |
| Encodeur (inférence) | $p(\mathbf{z} \mid \mathbf{x})$ exact (gaussien) | $q_\phi(\mathbf{z} \mid \mathbf{x})$ approché (réseau) |

Le passage à un décodeur non linéaire rend l'inférence exacte intraitable. L'encodeur $q_\phi$ est une **approximation variationnelle** de la vraie distribution a posteriori.

---

## Apprentissage de représentations

L'ACP, les autoencodeurs et les VAE partagent un objectif commun : trouver une **représentation** $\mathbf{z}$ des données qui soit :

| Propriété | Signification |
|-----------|---------------|
| De basse dimension | $L \ll D$, compression |
| Informative | Permet de reconstruire $\mathbf{x}$ |
| Structurée | Capture les facteurs de variation des données |

En apprentissage supervisé, les caractéristiques $\boldsymbol{\phi}(\mathbf{x})$ sont choisies à l'avance. Les réseaux de neurones **apprennent** la représentation $\mathbf{z}$ en même temps que le prédicteur, ce qui permet de capturer des structures que des caractéristiques prédéfinies manqueraient.

---

## Le fil conducteur : variables latentes

Nous avons vu les variables latentes sous trois formes :

| Modèle | Variable latente | Rôle |
|--------|-----------------|------|
| Bradley-Terry | Score $s_k \in \mathbb{R}$ | Qualité d'un objet |
| GMM | $z \in \{1, \ldots, K\}$ | Appartenance à un groupe |
| PPCA / Autoencodeur | $\mathbf{z} \in \mathbb{R}^L$ | Représentation comprimée |

Dans chaque cas, les variables latentes **ne sont pas observées directement**. On les infère à partir des données, soit par maximum de vraisemblance (Bradley-Terry), soit par l'étape E de l'algorithme EM (GMM), soit par l'encodeur (autoencodeur).

---

<!-- footer: "" -->

## Résumé : ACP probabiliste

| Concept | Détail |
|---------|--------|
| Modèle génératif | $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, $\mathbf{x} \mid \mathbf{z} \sim \mathcal{N}(\mathbf{W}\mathbf{z} + \boldsymbol{\mu}, \sigma^2\mathbf{I})$ |
| Marginale | $p(\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \mathbf{W}\mathbf{W}^\top + \sigma^2\mathbf{I})$ |
| EMV | $\hat{\mathbf{W}} = \mathbf{U}_L(\boldsymbol{\Lambda}_L - \sigma^2\mathbf{I})^{1/2}\mathbf{R}$ |
| Limite $\sigma^2 \to 0$ | ACP classique |
| Estimation du bruit | $\hat{\sigma}^2 = \frac{1}{D-L}\sum_{j>L} \lambda_j$ |

---

## Résumé : vers les autoencodeurs

| Modèle | Encodeur / Décodeur | Entraînement |
|--------|---------------------|--------------|
| ACP classique | Linéaire / Linéaire | Valeurs propres |
| PPCA | Linéaire / Linéaire | EMV (forme fermée) |
| Autoencodeur | Réseau / Réseau | Rétropropagation |
| VAE | Réseau / Réseau | ELBO + rétropropagation |

La progression ACP → PPCA → autoencodeur → VAE montre comment enrichir un modèle de base par des non-linéarités et un cadre probabiliste.

---

<!-- _class: lead -->

# Questions?

**Exercices recommandés** :
- Exercice (ch3) : Construire la matrice de conception de Bradley-Terry
- Exercice (ch3) : Scores latents et classement Elo
- Exercice (ACP) : Calcul manuel des composantes principales
- Exercice (ACP) : Variance expliquée et choix de $L$
- Exercice (ACP) : Marginale du modèle PPCA
