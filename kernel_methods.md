# Méthodes à noyau

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Définir un noyau et ses propriétés mathématiques
- Comprendre le lissage par noyau et ses applications
- Implémenter l'estimateur de densité de Parzen-Rosenblatt
- Appliquer la régression de Nadaraya-Watson
- Expliquer l'astuce du noyau et le théorème du représentant
- Relier les différentes méthodes à noyau vues dans ce livre
```

## Introduction aux noyaux

### Qu'est-ce qu'un noyau?

Un **noyau** $K$ est une fonction de pondération qui mesure la similarité ou la proximité entre deux points. Il existe plusieurs utilisations distinctes des noyaux en apprentissage automatique:

1. **Noyaux de lissage**: fonctions de pondération pour les méthodes non-paramétriques
2. **Noyaux de Mercer**: fonctions de similarité pour l'astuce du noyau

Ce chapitre unifie ces concepts et montre comment ils sont reliés aux méthodes vues précédemment.

## Noyaux de lissage

### Définition

Un **noyau de lissage** (ou noyau de densité) est une fonction $K: \mathbb{R} \to \mathbb{R}$ satisfaisant:
1. $K(u) \geq 0$ pour tout $u$
2. $\int K(u) \, du = 1$
3. $K(u) = K(-u)$ (symétrie)

Ces propriétés font de $K$ une densité de probabilité symétrique.

### Exemples de noyaux de lissage

**Noyau uniforme (boxcar)**:
$$
K(u) = \frac{1}{2} \mathbb{I}(|u| \leq 1)
$$

**Noyau gaussien**:
$$
K(u) = \frac{1}{\sqrt{2\pi}} e^{-u^2/2}
$$

**Noyau d'Epanechnikov**:
$$
K(u) = \frac{3}{4}(1 - u^2) \mathbb{I}(|u| \leq 1)
$$

**Noyau tri-cube**:
$$
K(u) = \frac{70}{81}(1 - |u|^3)^3 \mathbb{I}(|u| \leq 1)
$$

### Paramètre de largeur

Pour contrôler l'étendue du noyau, on introduit un paramètre de **largeur** (ou *bandwidth*) $\lambda > 0$:
$$
K_\lambda(u) = \frac{1}{\lambda} K\left(\frac{u}{\lambda}\right)
$$

- Grand $\lambda$: beaucoup de lissage, modèle simple
- Petit $\lambda$: peu de lissage, modèle complexe

Ce paramètre joue un rôle analogue à $k$ dans k-NN ou à $\lambda$ en régularisation.

## Estimation de densité par noyau

### Estimateur de Parzen-Rosenblatt

L'**estimateur de densité à noyau** (ou estimateur de Parzen-Rosenblatt) estime la densité de probabilité $p(x)$ en plaçant un noyau sur chaque observation:
$$
\hat{p}(x \mid \mathcal{D}) = \frac{1}{N} \sum_{i=1}^N K_\lambda(x - x_i)
$$

### Interprétation

Intuitivement, cet estimateur:
1. Place une "bosse" sur chaque point de données
2. Somme toutes ces bosses (normalisées)
3. Le résultat est une approximation lisse de la densité

### Effet du paramètre de largeur

- **$\lambda$ trop petit**: l'estimateur est trop "piquant", chaque observation crée un pic
- **$\lambda$ trop grand**: l'estimateur est trop lisse, tous les détails sont perdus
- **$\lambda$ optimal**: compromis entre biais et variance

### Extension multidimensionnelle

Pour des données en dimension $D$, on utilise souvent un noyau produit:
$$
K_\lambda(\boldsymbol{x} - \boldsymbol{x}_i) = \prod_{d=1}^D K_\lambda(x_d - x_{id})
$$

ou un noyau gaussien multivarié:
$$
K_\mathbf{H}(\boldsymbol{x} - \boldsymbol{x}_i) = \frac{1}{(2\pi)^{D/2}|\mathbf{H}|^{1/2}} \exp\left(-\frac{1}{2}(\boldsymbol{x} - \boldsymbol{x}_i)^\top \mathbf{H}^{-1}(\boldsymbol{x} - \boldsymbol{x}_i)\right)
$$

où $\mathbf{H}$ est la matrice de largeur de bande.

## Régression par noyau

### Estimateur de Nadaraya-Watson

Pour la régression, nous voulons estimer $\mathbb{E}[Y \mid \boldsymbol{x}]$. L'estimateur de **Nadaraya-Watson** utilise une moyenne pondérée par noyau:
$$
\hat{f}(\boldsymbol{x}) = \sum_{i=1}^N w_i(\boldsymbol{x}) \, y_i
$$

où les poids sont normalisés:
$$
w_i(\boldsymbol{x}) = \frac{K_\lambda(\boldsymbol{x} - \boldsymbol{x}_i)}{\sum_{j=1}^N K_\lambda(\boldsymbol{x} - \boldsymbol{x}_j)}
$$

### Dérivation

En utilisant l'estimation de densité par noyau pour $p(x, y)$:
$$
\hat{p}(x, y) = \frac{1}{N} \sum_{i=1}^N K_\lambda(x - x_i) K_\lambda(y - y_i)
$$

On peut montrer que:
$$
\mathbb{E}[Y \mid x] = \frac{\int y \, p(x, y) \, dy}{\int p(x, y) \, dy} \approx \sum_{i=1}^N w_i(x) \, y_i
$$

### Propriétés

- **Non-paramétrique**: pas de forme fonctionnelle supposée
- **Local**: la prédiction dépend surtout des voisins proches
- **Lisse**: produit une fonction continue (contrairement à k-NN)

## Lien avec k-NN

### K-NN comme cas particulier

Le k-NN peut être vu comme une méthode à noyau avec un noyau **adaptatif**:
$$
K(x, x_i) = \begin{cases} 1 & \text{si } x_i \in \mathcal{N}_k(x) \\ 0 & \text{sinon} \end{cases}
$$

où $\mathcal{N}_k(x)$ est l'ensemble des $k$ plus proches voisins de $x$.

La largeur du noyau s'adapte à la densité locale:
- Dans les régions denses: le voisinage est petit
- Dans les régions éparses: le voisinage est grand

### Comparaison

| Aspect | Noyau fixe | K-NN |
|--------|-----------|------|
| Largeur | Fixe ($\lambda$) | Adaptative |
| Nombre de voisins | Variable | Fixe ($k$) |
| Poids | Continus | Uniformes |
| Discontinuités | Non | Oui (aux frontières) |

## L'astuce du noyau

### Motivation

Dans de nombreux algorithmes, les données n'apparaissent qu'à travers des **produits scalaires** $\langle \boldsymbol{x}_i, \boldsymbol{x}_j \rangle$.

L'**astuce du noyau** consiste à remplacer ces produits scalaires par une fonction noyau:
$$
\langle \boldsymbol{x}_i, \boldsymbol{x}_j \rangle \to \mathcal{K}(\boldsymbol{x}_i, \boldsymbol{x}_j)
$$

### Noyaux de Mercer

Un noyau $\mathcal{K}$ est un **noyau de Mercer** s'il correspond à un produit scalaire dans un espace de redescription:
$$
\mathcal{K}(\boldsymbol{x}, \boldsymbol{x}') = \langle \boldsymbol{\phi}(\boldsymbol{x}), \boldsymbol{\phi}(\boldsymbol{x}') \rangle
$$

où $\boldsymbol{\phi}: \mathbb{R}^D \to \mathcal{H}$ est une transformation vers un espace de caractéristiques $\mathcal{H}$ (possiblement de dimension infinie).

### Condition de Mercer

Un noyau est de Mercer si et seulement si sa **matrice de Gram** est semi-définie positive pour tout ensemble de points:
$$
\mathbf{K} = \begin{pmatrix}
\mathcal{K}(\boldsymbol{x}_1, \boldsymbol{x}_1) & \cdots & \mathcal{K}(\boldsymbol{x}_1, \boldsymbol{x}_N) \\
\vdots & \ddots & \vdots \\
\mathcal{K}(\boldsymbol{x}_N, \boldsymbol{x}_1) & \cdots & \mathcal{K}(\boldsymbol{x}_N, \boldsymbol{x}_N)
\end{pmatrix} \succeq 0
$$

### Exemples de noyaux de Mercer

**Noyau linéaire**:
$$
\mathcal{K}(\boldsymbol{x}, \boldsymbol{x}') = \boldsymbol{x}^\top \boldsymbol{x}'
$$

**Noyau polynomial**:
$$
\mathcal{K}(\boldsymbol{x}, \boldsymbol{x}') = (1 + \boldsymbol{x}^\top \boldsymbol{x}')^p
$$

**Noyau gaussien (RBF)**:
$$
\mathcal{K}(\boldsymbol{x}, \boldsymbol{x}') = \exp\left(-\frac{\|\boldsymbol{x} - \boldsymbol{x}'\|^2}{2\ell^2}\right)
$$

Le noyau gaussien correspond à un espace de caractéristiques de dimension **infinie**.

## Le théorème du représentant

### Énoncé

Le **théorème du représentant** stipule que pour tout problème d'optimisation de la forme:
$$
\min_{f \in \mathcal{H}_K} \left[\sum_{i=1}^N L(y_i, f(x_i)) + \lambda \|f\|_{\mathcal{H}_K}^2\right]
$$

où $\mathcal{H}_K$ est un espace de Hilbert à noyau reproduisant (RKHS), la solution optimale s'écrit:
$$
f^*(x) = \sum_{i=1}^N \alpha_i \mathcal{K}(x, x_i)
$$

### Implications

1. Même si $\mathcal{H}_K$ est de dimension infinie, la solution optimale est une **combinaison linéaire finie** des noyaux centrés sur les données
2. Le problème de dimension infinie devient un problème d'optimisation sur $N$ paramètres $\alpha_i$
3. C'est le fondement théorique des SVM et de la régression ridge à noyau

## Applications

### Régression ridge à noyau

En appliquant le théorème du représentant à la régression ridge:
$$
\min_{\boldsymbol{\alpha}} \|\mathbf{y} - \mathbf{K}\boldsymbol{\alpha}\|^2 + \lambda \boldsymbol{\alpha}^\top \mathbf{K} \boldsymbol{\alpha}
$$

La solution est:
$$
\boldsymbol{\alpha} = (\mathbf{K} + \lambda \mathbf{I})^{-1} \mathbf{y}
$$

### SVM à noyau

Les SVM utilisent l'astuce du noyau dans leur forme duale pour créer des frontières de décision non-linéaires (voir chapitre SVM).

## Résumé

Les méthodes à noyau forment une famille puissante de techniques:

- Les **noyaux de lissage** pondèrent les observations par distance
- L'**estimation de densité de Parzen** utilise des noyaux pour estimer $p(x)$
- La **régression de Nadaraya-Watson** fait de la régression locale pondérée
- L'**astuce du noyau** permet de travailler implicitement dans des espaces de grande dimension
- Le **théorème du représentant** garantit des solutions finies même dans des espaces infinis

Ces concepts unifient k-NN, les SVM, la régression ridge à noyau et de nombreuses autres méthodes.

## Exercices

```{admonition} Exercice 1: Estimation de densité
:class: tip

Soit les données $\{-1, 0, 2\}$ en 1D.

1. Calculez l'estimateur de Parzen avec un noyau gaussien ($\sigma = 1$) en $x = 0$.
2. Calculez-le en $x = 1$.
3. Esquissez la forme générale de $\hat{p}(x)$.
```

```{admonition} Exercice 2: Nadaraya-Watson
:class: tip

Données: $(x, y) = \{(0, 1), (1, 2), (2, 1)\}$.

Avec un noyau gaussien de $\sigma = 0.5$:
1. Calculez les poids pour prédire en $x = 0.5$.
2. Calculez la prédiction $\hat{f}(0.5)$.
```

```{admonition} Exercice 3: Noyau polynomial
:class: tip

Montrez que le noyau polynomial en 2D $\mathcal{K}(\boldsymbol{x}, \boldsymbol{x}') = (\boldsymbol{x}^\top \boldsymbol{x}')^2$ correspond à la transformation:
$$
\boldsymbol{\phi}(x_1, x_2) = (x_1^2, \sqrt{2}x_1 x_2, x_2^2)
$$

(Calculez explicitement $\boldsymbol{\phi}(\boldsymbol{x})^\top \boldsymbol{\phi}(\boldsymbol{x}')$.)
```

```{admonition} Exercice 4: Matrice de Gram
:class: tip

Pour les points $\{0, 1, 2\}$ et le noyau gaussien avec $\ell = 1$:
1. Calculez la matrice de Gram $\mathbf{K}$.
2. Vérifiez qu'elle est symétrique.
3. Calculez ses valeurs propres pour vérifier qu'elle est semi-définie positive.
```
