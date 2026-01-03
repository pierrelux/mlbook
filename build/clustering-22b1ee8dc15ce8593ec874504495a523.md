# Partitionnement

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Définir le problème de partitionnement (clustering)
- Implémenter l'algorithme k-moyennes
- Dériver l'algorithme EM pour les mélanges gaussiens (GMM)
- Expliquer le lien entre k-moyennes et GMM
- Comprendre les différents types de covariance dans les GMM
- Appliquer ces méthodes à des problèmes réels
```

## Motivation

Le **partitionnement** (ou *clustering*) est une tâche d'apprentissage **non supervisé** qui consiste à regrouper des exemples similaires ensemble, sans disposer d'étiquettes. C'est l'une des techniques les plus utilisées en analyse de données.

### Applications

- **Segmentation de clientèle**: regrouper des clients ayant des comportements similaires
- **Compression d'images**: réduire le nombre de couleurs en regroupant les pixels similaires
- **Détection d'anomalies**: identifier les points qui n'appartiennent à aucun groupe
- **Prétraitement**: découvrir une structure dans les données avant une analyse supervisée
- **Bioinformatique**: regrouper des gènes ou des protéines ayant des fonctions similaires

### Types de partitionnement

- **Dur** (*hard clustering*): chaque point appartient à exactement un groupe
- **Souple** (*soft clustering*): chaque point a une probabilité d'appartenir à chaque groupe

## K-moyennes

### Formulation

L'algorithme **k-moyennes** (*k-means*) partitionne $N$ points en $K$ groupes en minimisant la **distorsion intra-groupe**:

$$
\mathcal{L} = \sum_{n=1}^N \sum_{k=1}^K r_{nk} \|\boldsymbol{x}_n - \boldsymbol{\mu}_k\|^2
$$

où:
- $r_{nk} \in \{0, 1\}$ est l'**assignation** du point $n$ au groupe $k$ (avec $\sum_k r_{nk} = 1$)
- $\boldsymbol{\mu}_k$ est le **centroïde** du groupe $k$

### Algorithme

K-moyennes alterne entre deux étapes:

**Étape E (assignation)**: Fixer les centroïdes, optimiser les assignations
$$
r_{nk} = \begin{cases} 1 & \text{si } k = \arg\min_{k'} \|\boldsymbol{x}_n - \boldsymbol{\mu}_{k'}\|^2 \\ 0 & \text{sinon} \end{cases}
$$

En d'autres termes, chaque point est assigné au centroïde le plus proche.

**Étape M (mise à jour)**: Fixer les assignations, optimiser les centroïdes
$$
\boldsymbol{\mu}_k = \frac{\sum_{n=1}^N r_{nk} \boldsymbol{x}_n}{\sum_{n=1}^N r_{nk}}
$$

Chaque centroïde est la moyenne des points qui lui sont assignés.

### Pseudocode

```
Entrée: Données X, nombre de groupes K
1. Initialiser les centroïdes μ_1, ..., μ_K (aléatoirement)
2. Répéter jusqu'à convergence:
   a. Pour chaque point n:
      - Assigner x_n au centroïde le plus proche
   b. Pour chaque groupe k:
      - μ_k ← moyenne des points assignés à k
Sortie: Centroïdes et assignations
```

### Propriétés

**Convergence**: K-moyennes converge toujours vers un minimum local car:
1. L'étape E réduit ou maintient la distorsion (assignation optimale)
2. L'étape M réduit ou maintient la distorsion (centroïde optimal)
3. La distorsion est bornée inférieurement par 0

**Sensibilité à l'initialisation**: Le minimum trouvé dépend de l'initialisation. Solutions:
- Exécuter plusieurs fois avec des initialisations différentes
- Utiliser **k-means++** pour une meilleure initialisation

### K-means++

Initialisation intelligente qui choisit les centroïdes initiaux de manière à les espacer:
1. Choisir le premier centroïde uniformément au hasard parmi les données
2. Pour chaque centroïde suivant, choisir un point avec probabilité proportionnelle au carré de la distance au centroïde le plus proche
3. Répéter jusqu'à avoir $K$ centroïdes

Cette stratégie garantit une solution $O(\log K)$-compétitive avec l'optimum.

## Modèles de mélange gaussien (GMM)

### Formulation probabiliste

Un **modèle de mélange gaussien** (GMM, *Gaussian Mixture Model*) suppose que les données sont générées par un mélange de $K$ distributions gaussiennes:

$$
p(\boldsymbol{x} \mid \boldsymbol{\theta}) = \sum_{k=1}^K \pi_k \mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

où:
- $\pi_k$ est le **poids** du composant $k$ (avec $\sum_k \pi_k = 1$ et $\pi_k \geq 0$)
- $\boldsymbol{\mu}_k$ est la **moyenne** du composant $k$
- $\boldsymbol{\Sigma}_k$ est la **matrice de covariance** du composant $k$

### Variables latentes

Pour chaque observation $\boldsymbol{x}_n$, nous introduisons une variable latente $z_n \in \{1, \ldots, K\}$ indiquant le composant dont elle provient:

$$
p(z_n = k) = \pi_k
$$
$$
p(\boldsymbol{x}_n \mid z_n = k) = \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

### Responsabilités

La **responsabilité** du composant $k$ pour le point $n$ est la probabilité a posteriori:

$$
r_{nk} \triangleq p(z_n = k \mid \boldsymbol{x}_n, \boldsymbol{\theta}) = \frac{\pi_k \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{k'=1}^K \pi_{k'} \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_{k'}, \boldsymbol{\Sigma}_{k'})}
$$

Contrairement à k-moyennes, les responsabilités sont des valeurs continues dans $[0, 1]$.

## Algorithme EM pour GMM

### L'algorithme Espérance-Maximisation

L'**algorithme EM** (*Expectation-Maximization*) est une méthode générale pour l'estimation de paramètres dans des modèles avec variables latentes. Il maximise itérativement une **borne inférieure** de la log-vraisemblance.

### Étape E

Calculer les responsabilités avec les paramètres actuels $\boldsymbol{\theta}^{(t)}$:

$$
r_{nk}^{(t)} = \frac{\pi_k^{(t)} \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k^{(t)}, \boldsymbol{\Sigma}_k^{(t)})}{\sum_{k'=1}^K \pi_{k'}^{(t)} \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_{k'}^{(t)}, \boldsymbol{\Sigma}_{k'}^{(t)})}
$$

### Étape M

Maximiser la log-vraisemblance des données complètes attendue. Les mises à jour sont:

**Poids du mélange**:
$$
\pi_k^{(t+1)} = \frac{1}{N} \sum_{n=1}^N r_{nk}^{(t)} = \frac{r_k^{(t)}}{N}
$$

où $r_k^{(t)} = \sum_n r_{nk}^{(t)}$ est le nombre effectif de points dans le groupe $k$.

**Moyennes**:
$$
\boldsymbol{\mu}_k^{(t+1)} = \frac{\sum_{n=1}^N r_{nk}^{(t)} \boldsymbol{x}_n}{r_k^{(t)}}
$$

**Covariances**:
$$
\boldsymbol{\Sigma}_k^{(t+1)} = \frac{\sum_{n=1}^N r_{nk}^{(t)} (\boldsymbol{x}_n - \boldsymbol{\mu}_k^{(t+1)})(\boldsymbol{x}_n - \boldsymbol{\mu}_k^{(t+1)})^\top}{r_k^{(t)}}
$$

### Pseudocode

```
Entrée: Données X, nombre de composants K
1. Initialiser les paramètres θ = (π, μ, Σ)
2. Répéter jusqu'à convergence:
   a. Étape E: Calculer les responsabilités r_nk
   b. Étape M: Mettre à jour π, μ, Σ
Sortie: Paramètres θ et responsabilités r
```

## Types de covariance

### Covariance complète

Chaque composant a sa propre matrice de covariance $\boldsymbol{\Sigma}_k$, qui peut avoir n'importe quelle forme (ellipsoïdes orientés arbitrairement).

**Paramètres**: $K \times D(D+1)/2$ pour les covariances

### Covariance partagée (tied)

Tous les composants partagent la même matrice de covariance $\boldsymbol{\Sigma}$:
$$
\boldsymbol{\Sigma} = \frac{1}{N} \sum_{k=1}^K \sum_{n=1}^N r_{nk} (\boldsymbol{x}_n - \boldsymbol{\mu}_k)(\boldsymbol{x}_n - \boldsymbol{\mu}_k)^\top
$$

### Covariance diagonale

Chaque $\boldsymbol{\Sigma}_k$ est diagonale: les dimensions sont indépendantes au sein de chaque groupe.

**Paramètres**: $K \times D$

### Covariance sphérique

Chaque $\boldsymbol{\Sigma}_k = \sigma_k^2 \mathbf{I}$ est un multiple de l'identité: les groupes sont des sphères.

**Paramètres**: $K$ scalaires

## Lien entre k-moyennes et GMM

### K-moyennes comme cas particulier

K-moyennes peut être vu comme un GMM avec:
1. **Covariances sphériques partagées**: $\boldsymbol{\Sigma}_k = \sigma^2 \mathbf{I}$ pour tout $k$
2. **Limite $\sigma^2 \to 0$**: les responsabilités deviennent dures (0 ou 1)
3. **Poids uniformes**: $\pi_k = 1/K$

Dans cette limite:
$$
r_{nk} \to \mathbb{I}(k = \arg\min_{k'} \|\boldsymbol{x}_n - \boldsymbol{\mu}_{k'}\|^2)
$$

### Regroupement dur vs souple

| Aspect | K-moyennes | GMM |
|--------|-----------|-----|
| Assignation | Dure (0 ou 1) | Souple (probabilités) |
| Forme des groupes | Sphériques | Ellipsoïdales |
| Modèle probabiliste | Non | Oui |
| Incertitude | Non | Oui |
| Génératif | Non | Oui |

## Considérations pratiques

### Choix de $K$

Le nombre de groupes est un hyperparamètre difficile à choisir:

1. **Méthode du coude**: tracer la distorsion en fonction de $K$ et chercher un coude

2. **Critères d'information**:
   - BIC: $-2\log p(\mathcal{D}|\hat{\boldsymbol{\theta}}) + p \log N$
   - AIC: $-2\log p(\mathcal{D}|\hat{\boldsymbol{\theta}}) + 2p$
   
   où $p$ est le nombre de paramètres

3. **Silhouette**: mesure de cohésion et séparation des groupes

### Problèmes potentiels

**Singularités**: Si un composant GMM ne contient qu'un point, sa covariance peut devenir singulière. Solutions:
- Ajouter une régularisation diagonale
- Utiliser des covariances partagées ou diagonales
- Réinitialiser les composants problématiques

**Minima locaux**: EM converge vers un minimum local. Solutions:
- Plusieurs exécutions avec initialisations différentes
- Initialiser avec k-moyennes

## Résumé

Le partitionnement regroupe des données non étiquetées:

- **K-moyennes** minimise la distorsion en alternant assignation et mise à jour des centroïdes
- Les **GMM** modélisent les données comme un mélange de gaussiennes
- **EM** estime les paramètres des GMM en alternant étape E (responsabilités) et étape M (paramètres)
- K-moyennes est un cas limite de GMM avec covariances sphériques
- Le choix de $K$ reste un défi pratique

Ces méthodes forment la base de nombreuses techniques d'apprentissage non supervisé plus avancées.

## Exercices

```{admonition} Exercice 1: K-moyennes manuel
:class: tip

Considérez les points suivants en 1D: $\{1, 2, 3, 10, 11, 12\}$ avec $K=2$.

1. Si les centroïdes initiaux sont $\mu_1 = 1$ et $\mu_2 = 2$, effectuez une itération de k-moyennes.
2. Après combien d'itérations l'algorithme converge-t-il?
3. Quelle est la distorsion finale?
```

```{admonition} Exercice 2: Responsabilités GMM
:class: tip

Un GMM à 2 composants en 1D a les paramètres:
- $\pi_1 = 0.3$, $\mu_1 = 0$, $\sigma_1^2 = 1$
- $\pi_2 = 0.7$, $\mu_2 = 3$, $\sigma_2^2 = 1$

Pour l'observation $x = 1$:
1. Calculez $p(x | z=1)$ et $p(x | z=2)$.
2. Calculez les responsabilités $r_1$ et $r_2$.
3. À quel composant ce point "appartient-il" le plus?
```

```{admonition} Exercice 3: Nombre de paramètres
:class: tip

Pour un GMM avec $K$ composants en dimension $D$:
1. Combien de paramètres y a-t-il avec covariances complètes?
2. Et avec covariances diagonales?
3. Pour $D = 100$ et $K = 10$, calculez ces nombres.
```

```{admonition} Exercice 4: Convergence EM
:class: tip

Expliquez pourquoi l'algorithme EM est garanti de converger (au sens où la log-vraisemblance ne décroît jamais).

Pourquoi cela ne garantit-il pas de trouver le maximum global?
```
