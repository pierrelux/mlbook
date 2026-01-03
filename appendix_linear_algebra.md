# Annexe: Révision d'algèbre linéaire

Cette annexe révise les concepts d'algèbre linéaire utilisés tout au long du livre.

## Vecteurs et matrices

### Notation

Un **vecteur** $\boldsymbol{x} \in \mathbb{R}^D$ est un tableau de $D$ nombres réels:
$$
\boldsymbol{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_D \end{pmatrix}
$$

Par convention, les vecteurs sont des vecteurs colonnes. Le transposé $\boldsymbol{x}^\top$ est un vecteur ligne.

Une **matrice** $\mathbf{A} \in \mathbb{R}^{M \times N}$ a $M$ lignes et $N$ colonnes:
$$
\mathbf{A} = \begin{pmatrix}
a_{11} & a_{12} & \cdots & a_{1N} \\
a_{21} & a_{22} & \cdots & a_{2N} \\
\vdots & \vdots & \ddots & \vdots \\
a_{M1} & a_{M2} & \cdots & a_{MN}
\end{pmatrix}
$$

### Opérations de base

**Produit scalaire** (dot product):
$$
\boldsymbol{x}^\top \boldsymbol{y} = \sum_{i=1}^D x_i y_i
$$

**Produit matrice-vecteur**: $\mathbf{A}\boldsymbol{x}$ où $\mathbf{A} \in \mathbb{R}^{M \times N}$ et $\boldsymbol{x} \in \mathbb{R}^N$ donne un vecteur $\in \mathbb{R}^M$.

**Produit matriciel**: $\mathbf{AB}$ où $\mathbf{A} \in \mathbb{R}^{M \times K}$ et $\mathbf{B} \in \mathbb{R}^{K \times N}$ donne $\mathbf{C} \in \mathbb{R}^{M \times N}$.
$$
C_{ij} = \sum_{k=1}^K A_{ik} B_{kj}
$$

### Matrices spéciales

| Type | Définition |
|------|------------|
| **Identité** $\mathbf{I}$ | $I_{ij} = 1$ si $i=j$, 0 sinon |
| **Diagonale** | $A_{ij} = 0$ si $i \neq j$ |
| **Symétrique** | $\mathbf{A} = \mathbf{A}^\top$ |
| **Orthogonale** | $\mathbf{A}^\top \mathbf{A} = \mathbf{I}$ |
| **Définie positive** | $\boldsymbol{x}^\top \mathbf{A} \boldsymbol{x} > 0$ pour tout $\boldsymbol{x} \neq \mathbf{0}$ |
| **Semi-définie positive** | $\boldsymbol{x}^\top \mathbf{A} \boldsymbol{x} \geq 0$ pour tout $\boldsymbol{x}$ |

## Normes vectorielles et matricielles

### Normes vectorielles

La **norme $\ell_p$** d'un vecteur:
$$
\|\boldsymbol{x}\|_p = \left(\sum_{i=1}^D |x_i|^p\right)^{1/p}
$$

Cas particuliers:
- **Norme $\ell_1$** (Manhattan): $\|\boldsymbol{x}\|_1 = \sum_i |x_i|$
- **Norme $\ell_2$** (euclidienne): $\|\boldsymbol{x}\|_2 = \sqrt{\sum_i x_i^2}$
- **Norme $\ell_\infty$**: $\|\boldsymbol{x}\|_\infty = \max_i |x_i|$

### Normes matricielles

**Norme de Frobenius**:
$$
\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\text{tr}(\mathbf{A}^\top \mathbf{A})}
$$

**Norme spectrale** (norme induite par $\ell_2$):
$$
\|\mathbf{A}\|_2 = \sigma_{\max}(\mathbf{A})
$$

où $\sigma_{\max}$ est la plus grande valeur singulière.

## Valeurs propres et vecteurs propres

### Définition

Un vecteur $\boldsymbol{v} \neq \mathbf{0}$ est un **vecteur propre** de $\mathbf{A}$ avec **valeur propre** $\lambda$ si:
$$
\mathbf{A}\boldsymbol{v} = \lambda \boldsymbol{v}
$$

Pour une matrice $n \times n$, il existe au plus $n$ valeurs propres distinctes.

### Propriétés

Pour une matrice **symétrique** $\mathbf{A} = \mathbf{A}^\top$:
- Toutes les valeurs propres sont **réelles**
- Les vecteurs propres associés à des valeurs propres distinctes sont **orthogonaux**
- $\mathbf{A}$ est diagonalisable: $\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^\top$

### Diagonalisation

Si $\mathbf{A}$ a $n$ vecteurs propres linéairement indépendants:
$$
\mathbf{A} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^{-1}
$$

où:
- $\mathbf{V}$ a les vecteurs propres en colonnes
- $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_n)$

Pour une matrice symétrique: $\mathbf{V}^{-1} = \mathbf{V}^\top$ (matrice orthogonale).

## Décompositions matricielles

### Décomposition en valeurs singulières (SVD)

Toute matrice $\mathbf{A} \in \mathbb{R}^{M \times N}$ peut s'écrire:
$$
\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top
$$

où:
- $\mathbf{U} \in \mathbb{R}^{M \times M}$ est orthogonale (vecteurs singuliers gauches)
- $\boldsymbol{\Sigma} \in \mathbb{R}^{M \times N}$ est diagonale (valeurs singulières $\sigma_i \geq 0$)
- $\mathbf{V} \in \mathbb{R}^{N \times N}$ est orthogonale (vecteurs singuliers droits)

**SVD tronquée**: approximation de rang $k$:
$$
\mathbf{A}_k = \sum_{i=1}^k \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^\top
$$

C'est la meilleure approximation de rang $k$ au sens de la norme de Frobenius.

### Décomposition de Cholesky

Si $\mathbf{A}$ est symétrique définie positive:
$$
\mathbf{A} = \mathbf{L}\mathbf{L}^\top
$$

où $\mathbf{L}$ est triangulaire inférieure avec des éléments diagonaux positifs.

Utile pour:
- Résoudre efficacement $\mathbf{A}\boldsymbol{x} = \boldsymbol{b}$
- Calculer $\mathbf{A}^{-1}$
- Échantillonner d'une gaussienne multivariée

### Décomposition QR

$$
\mathbf{A} = \mathbf{Q}\mathbf{R}
$$

où $\mathbf{Q}$ est orthogonale et $\mathbf{R}$ est triangulaire supérieure.

Utile pour résoudre les problèmes de moindres carrés.

## Inverses et pseudo-inverses

### Matrice inverse

Si $\mathbf{A}$ est carrée et inversible:
$$
\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} = \mathbf{I}
$$

Propriétés:
- $(\mathbf{AB})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$
- $(\mathbf{A}^\top)^{-1} = (\mathbf{A}^{-1})^\top$

### Pseudo-inverse de Moore-Penrose

Pour toute matrice $\mathbf{A}$, la pseudo-inverse $\mathbf{A}^+$ satisfait:
$$
\mathbf{A}\mathbf{A}^+\mathbf{A} = \mathbf{A}
$$

Si $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ (SVD):
$$
\mathbf{A}^+ = \mathbf{V}\boldsymbol{\Sigma}^+\mathbf{U}^\top
$$

où $\Sigma^+_{ii} = 1/\sigma_i$ si $\sigma_i > 0$.

**Application**: solution des moindres carrés:
$$
\boldsymbol{x}^* = \mathbf{A}^+ \boldsymbol{b}
$$

## Trace et déterminant

### Trace

La **trace** d'une matrice carrée est la somme des éléments diagonaux:
$$
\text{tr}(\mathbf{A}) = \sum_i A_{ii}
$$

Propriétés:
- $\text{tr}(\mathbf{A} + \mathbf{B}) = \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B})$
- $\text{tr}(\mathbf{AB}) = \text{tr}(\mathbf{BA})$ (propriété cyclique)
- $\text{tr}(\mathbf{A}) = \sum_i \lambda_i$ (somme des valeurs propres)

### Déterminant

Le **déterminant** mesure le changement de volume d'une transformation linéaire.

Propriétés:
- $\det(\mathbf{AB}) = \det(\mathbf{A})\det(\mathbf{B})$
- $\det(\mathbf{A}^{-1}) = 1/\det(\mathbf{A})$
- $\det(\mathbf{A}) = \prod_i \lambda_i$ (produit des valeurs propres)
- $\mathbf{A}$ est inversible $\Leftrightarrow$ $\det(\mathbf{A}) \neq 0$

## Dérivées matricielles

### Gradient

Le gradient de $f: \mathbb{R}^D \to \mathbb{R}$ par rapport à $\boldsymbol{x}$:
$$
\nabla_{\boldsymbol{x}} f = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_D} \end{pmatrix}
$$

### Jacobienne

Pour $\boldsymbol{f}: \mathbb{R}^N \to \mathbb{R}^M$, la jacobienne:
$$
\mathbf{J} = \frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}} \in \mathbb{R}^{M \times N}, \quad J_{ij} = \frac{\partial f_i}{\partial x_j}
$$

### Hessienne

Pour $f: \mathbb{R}^D \to \mathbb{R}$, la hessienne:
$$
\mathbf{H} = \nabla^2 f, \quad H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

### Identités utiles

| Expression | Gradient |
|------------|----------|
| $\boldsymbol{a}^\top \boldsymbol{x}$ | $\boldsymbol{a}$ |
| $\boldsymbol{x}^\top \mathbf{A} \boldsymbol{x}$ | $(\mathbf{A} + \mathbf{A}^\top)\boldsymbol{x}$ |
| $\|\boldsymbol{x}\|_2^2$ | $2\boldsymbol{x}$ |
| $\|\mathbf{A}\boldsymbol{x} - \boldsymbol{b}\|_2^2$ | $2\mathbf{A}^\top(\mathbf{A}\boldsymbol{x} - \boldsymbol{b})$ |

Pour la matrice $\mathbf{X}$:

| Expression | Gradient par rapport à $\mathbf{X}$ |
|------------|-------------------------------------|
| $\text{tr}(\mathbf{AX})$ | $\mathbf{A}^\top$ |
| $\text{tr}(\mathbf{X}^\top\mathbf{A})$ | $\mathbf{A}$ |
| $\log\det(\mathbf{X})$ | $\mathbf{X}^{-\top}$ |
