# Analyse en composantes principales

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Formuler l'ACP comme minimisation de l'erreur de reconstruction
- Formuler l'ACP comme maximisation de la variance
- Dériver l'ACP à partir de la décomposition en valeurs propres
- Comprendre le lien entre l'ACP et l'analyse factorielle
- Appliquer l'ACP pour la réduction de dimensionnalité
- Choisir le nombre de composantes à retenir
```

## Motivation

L'**analyse en composantes principales** (ACP, ou *PCA* en anglais) est une technique fondamentale de réduction de dimensionnalité. Elle transforme des données de haute dimension en une représentation de plus basse dimension tout en préservant autant d'information que possible.

### Pourquoi réduire la dimensionnalité?

1. **Visualisation**: projeter des données en 2D ou 3D pour les explorer visuellement
2. **Compression**: représenter les données avec moins de variables
3. **Débruitage**: éliminer les composantes de faible variance (souvent du bruit)
4. **Prétraitement**: améliorer les performances d'algorithmes en aval
5. **Découverte de structure**: identifier les directions principales de variation

### L'idée de l'autoencodeur linéaire

L'ACP peut être vue comme un **autoencodeur linéaire**: nous cherchons à transformer les données $\boldsymbol{x} \in \mathbb{R}^D$ en un sous-espace de dimension réduite $\boldsymbol{z} \in \mathbb{R}^L$ (où $L < D$), de sorte que la représentation soit une "bonne approximation" des données d'origine.

Le processus se fait en deux étapes:
1. **Encodage**: $\boldsymbol{z} = \mathbf{W}^\top \boldsymbol{x}$ (projection vers l'espace latent)
2. **Décodage**: $\hat{\boldsymbol{x}} = \mathbf{W} \boldsymbol{z}$ (reconstruction)

L'objectif est de trouver $\mathbf{W}$ tel que $\hat{\boldsymbol{x}}$ soit proche de $\boldsymbol{x}$.

## Formulation du problème

### Notation

Soit un ensemble de données centrées $\mathcal{D} = \{\boldsymbol{x}_n : n = 1, \ldots, N\}$ où $\boldsymbol{x}_n \in \mathbb{R}^D$ et:
$$
\bar{\boldsymbol{x}} = \frac{1}{N} \sum_{n=1}^N \boldsymbol{x}_n = \mathbf{0}
$$

Ces données sont organisées dans une matrice $\mathbf{X}$ de taille $N \times D$ (chaque ligne est un exemple).

### Représentation en termes de fonctions de base

Nous supposons que chaque $\boldsymbol{x}_n$ peut être "expliqué" comme une combinaison pondérée de $L$ fonctions de base $\boldsymbol{w}_1, \ldots, \boldsymbol{w}_L$ où chaque $\boldsymbol{w}_k \in \mathbb{R}^D$:
$$
\boldsymbol{x}_n \approx \sum_{k=1}^L z_{nk} \boldsymbol{w}_k
$$

Le vecteur $\boldsymbol{z}_n \in \mathbb{R}^L$ est la représentation de basse dimension, appelée **vecteur latent** ou **embedding**.

### Erreur de reconstruction

L'erreur de reconstruction s'écrit:
$$
\mathcal{L}(\mathbf{W}, \mathbf{Z}) = \frac{1}{N} \|\mathbf{X} - \mathbf{Z}\mathbf{W}^\top\|_F^2 = \frac{1}{N} \sum_{n=1}^N \|\boldsymbol{x}_n - \mathbf{W}\boldsymbol{z}_n\|^2
$$

où:
- $\mathbf{Z}$ est une matrice $N \times L$ dont les lignes sont les $\boldsymbol{z}_n$
- $\mathbf{W}$ est une matrice $D \times L$ dont les colonnes sont les $\boldsymbol{w}_k$
- $\|\cdot\|_F$ est la norme de Frobenius

Nous voulons minimiser cette erreur sous la contrainte que $\mathbf{W}$ soit **orthonormale**: $\mathbf{W}^\top \mathbf{W} = \mathbf{I}_L$.

## Dérivation de l'ACP

### Cas d'une seule composante ($L = 1$)

Commençons par trouver la meilleure représentation en 1D. L'erreur de reconstruction est:
$$
\mathcal{L}(\boldsymbol{w}_1, \tilde{\boldsymbol{z}}_1) = \frac{1}{N} \sum_{n=1}^N \|\boldsymbol{x}_n - z_{n1} \boldsymbol{w}_1\|^2
$$

En développant:
$$
\mathcal{L} = \frac{1}{N} \sum_{n=1}^N \left[\boldsymbol{x}_n^\top \boldsymbol{x}_n - 2z_{n1} \boldsymbol{w}_1^\top \boldsymbol{x}_n + z_{n1}^2 \boldsymbol{w}_1^\top \boldsymbol{w}_1\right]
$$

Puisque $\boldsymbol{w}_1^\top \boldsymbol{w}_1 = 1$ (contrainte d'orthonormalité):
$$
\mathcal{L} = \frac{1}{N} \sum_{n=1}^N \left[\boldsymbol{x}_n^\top \boldsymbol{x}_n - 2z_{n1} \boldsymbol{w}_1^\top \boldsymbol{x}_n + z_{n1}^2\right]
$$

### Optimisation par rapport à $z_{n1}$

En dérivant par rapport à $z_{n1}$ et en égalant à zéro:
$$
\frac{\partial \mathcal{L}}{\partial z_{n1}} = \frac{1}{N}\left[-2\boldsymbol{w}_1^\top \boldsymbol{x}_n + 2z_{n1}\right] = 0 \quad \Rightarrow \quad z_{n1}^* = \boldsymbol{w}_1^\top \boldsymbol{x}_n
$$

Le coefficient optimal est la **projection orthogonale** de $\boldsymbol{x}_n$ sur $\boldsymbol{w}_1$.

### Optimisation par rapport à $\boldsymbol{w}_1$

En réinsérant $z_{n1}^*$ dans l'erreur:
$$
\mathcal{L}(\boldsymbol{w}_1) = \frac{1}{N} \sum_{n=1}^N \left[\boldsymbol{x}_n^\top \boldsymbol{x}_n - z_{n1}^2\right] = \text{const} - \frac{1}{N} \sum_{n=1}^N z_{n1}^2
$$

Or:
$$
\frac{1}{N} \sum_{n=1}^N z_{n1}^2 = \frac{1}{N} \sum_{n=1}^N \boldsymbol{w}_1^\top \boldsymbol{x}_n \boldsymbol{x}_n^\top \boldsymbol{w}_1 = \boldsymbol{w}_1^\top \hat{\boldsymbol{\Sigma}} \boldsymbol{w}_1
$$

où $\hat{\boldsymbol{\Sigma}} = \frac{1}{N} \sum_{n=1}^N \boldsymbol{x}_n \boldsymbol{x}_n^\top$ est la **matrice de covariance empirique**.

### Problème contraint

Nous voulons maximiser $\boldsymbol{w}_1^\top \hat{\boldsymbol{\Sigma}} \boldsymbol{w}_1$ sous la contrainte $\|\boldsymbol{w}_1\| = 1$.

En utilisant les multiplicateurs de Lagrange:
$$
\tilde{\mathcal{L}}(\boldsymbol{w}_1) = \boldsymbol{w}_1^\top \hat{\boldsymbol{\Sigma}} \boldsymbol{w}_1 - \lambda_1(\boldsymbol{w}_1^\top \boldsymbol{w}_1 - 1)
$$

En dérivant et égalant à zéro:
$$
\hat{\boldsymbol{\Sigma}} \boldsymbol{w}_1 = \lambda_1 \boldsymbol{w}_1
$$

C'est l'équation aux **valeurs propres**! Le vecteur $\boldsymbol{w}_1$ doit être un vecteur propre de $\hat{\boldsymbol{\Sigma}}$.

### Choix de la valeur propre

En multipliant à gauche par $\boldsymbol{w}_1^\top$:
$$
\boldsymbol{w}_1^\top \hat{\boldsymbol{\Sigma}} \boldsymbol{w}_1 = \lambda_1
$$

Puisque nous voulons **maximiser** cette quantité, nous choisissons le vecteur propre correspondant à la **plus grande valeur propre**.

### Extension à $L$ composantes

Par un raisonnement similaire (en utilisant l'orthogonalité avec les composantes précédentes), on montre que:
- La $k$-ième composante principale est le $k$-ième vecteur propre de $\hat{\boldsymbol{\Sigma}}$
- Ordonnés par valeurs propres décroissantes: $\lambda_1 \geq \lambda_2 \geq \ldots \geq \lambda_D$

La solution optimale est:
$$
\mathbf{W} = \mathbf{U}_L
$$
où $\mathbf{U}_L$ contient les $L$ premiers vecteurs propres de $\hat{\boldsymbol{\Sigma}}$.

## Interprétation: maximisation de la variance

### Variance des données projetées

Puisque les données sont centrées ($\mathbb{E}[\boldsymbol{x}_n] = \mathbf{0}$), la moyenne des projections est:
$$
\mathbb{E}[z_{n1}] = \mathbb{E}[\boldsymbol{w}_1^\top \boldsymbol{x}_n] = \boldsymbol{w}_1^\top \mathbb{E}[\boldsymbol{x}_n] = 0
$$

La variance des données projetées est:
$$
\mathbb{V}[\tilde{\boldsymbol{z}}_1] = \mathbb{E}[z_{n1}^2] - (\mathbb{E}[z_{n1}])^2 = \frac{1}{N} \sum_{n=1}^N z_{n1}^2 = \boldsymbol{w}_1^\top \hat{\boldsymbol{\Sigma}} \boldsymbol{w}_1 = \lambda_1
$$

### Équivalence des formulations

Minimiser l'erreur de reconstruction est **équivalent** à maximiser la variance des données projetées:
$$
\arg\min_{\boldsymbol{w}_1} \mathcal{L}(\boldsymbol{w}_1) = \arg\max_{\boldsymbol{w}_1} \mathbb{V}[\tilde{\boldsymbol{z}}_1]
$$

C'est pourquoi on dit que l'ACP trouve les **directions de variance maximale**.

### Interprétation géométrique

- La première composante principale $\boldsymbol{w}_1$ est la direction le long de laquelle les données varient le plus
- La deuxième composante $\boldsymbol{w}_2$ est la direction de variance maximale parmi celles orthogonales à $\boldsymbol{w}_1$
- Et ainsi de suite...

## Algorithme et implémentation

### Calcul via la décomposition en valeurs propres

```
Entrée: Données X (N × D), dimension cible L
1. Centrer les données: X_c = X - moyenne(X)
2. Calculer la covariance: Σ = (1/N) X_c^T X_c
3. Calculer les valeurs/vecteurs propres de Σ
4. Trier par valeurs propres décroissantes
5. W = les L premiers vecteurs propres
6. Z = X_c W  (projections)
Sortie: W (D × L), Z (N × L)
```

### Calcul via la SVD

La **décomposition en valeurs singulières** (SVD) de $\mathbf{X}_c$ donne:
$$
\mathbf{X}_c = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top
$$

Les colonnes de $\mathbf{V}$ sont les vecteurs propres de $\mathbf{X}_c^\top \mathbf{X}_c$, donc les composantes principales.

Cette approche est numériquement plus stable et plus efficace pour les grandes matrices.

## Variance expliquée

### Proportion de variance

La variance totale des données est:
$$
\text{Var}_{\text{total}} = \text{tr}(\hat{\boldsymbol{\Sigma}}) = \sum_{k=1}^D \lambda_k
$$

La variance expliquée par les $L$ premières composantes est:
$$
\text{Var}_L = \sum_{k=1}^L \lambda_k
$$

La **proportion de variance expliquée** est:
$$
\frac{\text{Var}_L}{\text{Var}_{\text{total}}} = \frac{\sum_{k=1}^L \lambda_k}{\sum_{k=1}^D \lambda_k}
$$

### Choix de $L$

Plusieurs critères pour choisir le nombre de composantes:

1. **Seuil de variance**: garder assez de composantes pour expliquer (par exemple) 95% de la variance

2. **Règle du coude**: tracer les valeurs propres et chercher un "coude" (changement de pente)

3. **Critère de Kaiser**: garder les composantes avec $\lambda_k > 1$ (après standardisation)

4. **Validation croisée**: évaluer l'erreur de reconstruction sur des données de test

## Analyse en composantes principales probabiliste (PPCA)

### Modèle génératif

L'ACP peut être formulée comme un modèle probabiliste. L'**analyse factorielle** suppose:
$$
\begin{aligned}
p(\boldsymbol{z}) &= \mathcal{N}(\boldsymbol{z} \mid \mathbf{0}, \mathbf{I}) \\
p(\boldsymbol{x} \mid \boldsymbol{z}, \boldsymbol{\theta}) &= \mathcal{N}(\boldsymbol{x} \mid \mathbf{W}\boldsymbol{z} + \boldsymbol{\mu}, \boldsymbol{\Psi})
\end{aligned}
$$

où $\mathbf{W}$ est la **matrice de chargement des facteurs** et $\boldsymbol{\Psi}$ est la covariance du bruit.

### PPCA comme cas particulier

L'**ACP probabiliste** (PPCA) est un cas particulier où:
1. $\mathbf{W}$ a des colonnes orthonormales
2. $\boldsymbol{\Psi} = \sigma^2 \mathbf{I}$ (bruit isotrope)
3. $\boldsymbol{\mu} = \mathbf{0}$

### Lien avec l'ACP classique

Dans la **limite sans bruit** ($\sigma^2 \to 0$), la solution MLE de PPCA converge vers la solution de l'ACP classique:
$$
\mathbf{W}_{\text{MLE}} = \mathbf{U}_L \mathbf{L}_L^{1/2}
$$

où $\mathbf{U}_L$ contient les $L$ premiers vecteurs propres et $\mathbf{L}_L$ les valeurs propres correspondantes.

### Avantages de la formulation probabiliste

- Estimation du bruit $\sigma^2$
- Gestion des données manquantes
- Critère de sélection de modèle (vraisemblance)
- Base pour des extensions (mélange de PPCA, etc.)

## Résumé

L'analyse en composantes principales est une technique fondamentale qui:

- **Réduit la dimensionnalité** en projetant sur un sous-espace de dimension $L < D$
- **Minimise l'erreur de reconstruction** (équivalent à maximiser la variance)
- Se calcule via la **décomposition en valeurs propres** de la matrice de covariance
- Les **valeurs propres** indiquent la variance expliquée par chaque composante
- Admet une interprétation **probabiliste** (PPCA)

L'ACP est un outil essentiel pour la visualisation, la compression et le prétraitement des données.

## Exercices

```{admonition} Exercice 1: Calcul manuel
:class: tip

Considérez les données centrées suivantes en 2D:
$\boldsymbol{x}_1 = (2, 1)$, $\boldsymbol{x}_2 = (-2, -1)$, $\boldsymbol{x}_3 = (1, 2)$, $\boldsymbol{x}_4 = (-1, -2)$

1. Calculez la matrice de covariance empirique $\hat{\boldsymbol{\Sigma}}$.
2. Trouvez les valeurs propres et vecteurs propres.
3. Quelle proportion de variance est expliquée par la première composante?
```

```{admonition} Exercice 2: Reconstruction
:class: tip

Soit $\mathbf{W} = \begin{pmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{pmatrix}$ la première composante principale.

Pour le point $\boldsymbol{x} = (3, 1)$:
1. Calculez le coefficient latent $z = \mathbf{W}^\top \boldsymbol{x}$.
2. Calculez la reconstruction $\hat{\boldsymbol{x}} = \mathbf{W} z$.
3. Calculez l'erreur de reconstruction $\|\boldsymbol{x} - \hat{\boldsymbol{x}}\|^2$.
```

```{admonition} Exercice 3: Variance expliquée
:class: tip

Les valeurs propres d'une matrice de covariance sont $\lambda_1 = 5, \lambda_2 = 3, \lambda_3 = 1.5, \lambda_4 = 0.5$.

1. Quelle est la variance totale?
2. Quelle proportion de variance est expliquée par les 2 premières composantes?
3. Combien de composantes faut-il pour expliquer au moins 90% de la variance?
```

```{admonition} Exercice 4: Centrage
:class: tip

Expliquez pourquoi il est important de centrer les données avant d'appliquer l'ACP.

Que se passe-t-il si on n'effectue pas le centrage?
```
