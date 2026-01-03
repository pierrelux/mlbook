# Arbres de décision

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Décrire la structure des arbres de décision et leur fonctionnement
- Expliquer les critères de division (impureté de Gini, entropie)
- Implémenter un algorithme glouton d'apprentissage d'arbre
- Discuter de l'élagage et de la régularisation
- Appliquer les arbres à la classification et à la régression
- Comprendre pourquoi les arbres sont des modèles de base idéaux pour les ensembles
```

## Motivation

Les **arbres de décision** sont parmi les modèles d'apprentissage automatique les plus intuitifs et interprétables. Ils partitionnent l'espace des caractéristiques en régions avec des prédictions simples, imitant ainsi le processus de décision humain sous forme d'une série de questions.

Par exemple, pour prédire si un objet est un fruit comestible, un arbre pourrait poser successivement les questions:
1. "La couleur est-elle rouge ou jaune?" 
2. "La forme est-elle ronde?"
3. "La taille est-elle supérieure à 5 cm?"

Chaque question divise l'espace en deux régions, et les réponses successives mènent à une prédiction finale.

### Avantages des arbres de décision

Les arbres de décision sont populaires pour plusieurs raisons:

1. **Interprétabilité**: on peut visualiser et expliquer facilement les règles de décision
2. **Gestion des types mixtes**: ils traitent naturellement les caractéristiques discrètes et continues
3. **Invariance aux transformations monotones**: les seuils sont basés sur le rang des valeurs, pas leur magnitude
4. **Sélection automatique de variables**: les caractéristiques non informatives ne sont pas utilisées
5. **Robustesse aux valeurs aberrantes**: les seuils ne sont pas affectés par les valeurs extrêmes
6. **Scalabilité**: ils sont rapides à entraîner et s'adaptent bien aux grands ensembles de données
7. **Gestion des valeurs manquantes**: des stratégies existent pour les entrées incomplètes

### Inconvénients

Cependant, les arbres ont aussi des limitations:
- **Surapprentissage facile**: sans régularisation, ils mémorisent les données
- **Instabilité**: de petits changements dans les données peuvent modifier drastiquement l'arbre
- **Frontières axiales**: les divisions sont parallèles aux axes, limitant l'expressivité

Cette instabilité est précisément ce qui rend les arbres excellents pour les **méthodes d'ensemble**, que nous verrons au chapitre suivant.

## Structure d'un arbre de décision

### Composants

Un arbre de décision est composé de:
- **Nœuds internes**: posent des questions sur les caractéristiques
- **Branches**: représentent les réponses possibles
- **Feuilles**: contiennent les prédictions finales

### Arbres de régression

Pour la régression, un arbre partitionne l'espace d'entrée en $J$ régions disjointes $R_1, R_2, \ldots, R_J$ et associe une valeur constante $w_j$ à chaque région:

$$
f(\boldsymbol{x}; \boldsymbol{\theta}) = \sum_{j=1}^J w_j \, \mathbb{I}(\boldsymbol{x} \in R_j)
$$

où $\boldsymbol{\theta} = \{(R_j, w_j) : j = 1, \ldots, J\}$ représente les paramètres de l'arbre.

Chaque région $R_j$ est définie par une conjonction de conditions. Par exemple:
$$
R_1 = \{\boldsymbol{x} : x_1 \leq t_1 \text{ et } x_2 \leq t_2\}
$$

La prédiction pour cette région est la moyenne des réponses des exemples qui y tombent:
$$
w_j = \frac{\sum_{n=1}^N y_n \, \mathbb{I}(\boldsymbol{x}_n \in R_j)}{\sum_{n=1}^N \mathbb{I}(\boldsymbol{x}_n \in R_j)}
$$

Le résultat est une **surface constante par morceaux**.

### Arbres de classification

Pour la classification, chaque feuille contient une **distribution sur les étiquettes de classe** plutôt qu'une valeur unique. La distribution empirique au nœud $i$ est:

$$
\hat{\pi}_{ic} = \frac{1}{|\mathcal{D}_i|} \sum_{n \in \mathcal{D}_i} \mathbb{I}(y_n = c)
$$

où $\mathcal{D}_i$ est l'ensemble des exemples atteignant le nœud $i$.

La prédiction peut être:
- La classe majoritaire: $\hat{y} = \arg\max_c \hat{\pi}_{ic}$
- La distribution complète $\hat{\boldsymbol{\pi}}_i$ pour une estimation probabiliste

## Apprentissage d'un arbre

### Le problème d'optimisation

Idéalement, nous voudrions minimiser:
$$
\mathcal{L}(\boldsymbol{\theta}) = \sum_{n=1}^N \ell(y_n, f(\boldsymbol{x}_n; \boldsymbol{\theta})) = \sum_{j=1}^J \sum_{\boldsymbol{x}_n \in R_j} \ell(y_n, w_j)
$$

Cependant, cette fonction n'est pas différentiable car nous devons chercher dans l'espace **discret** des structures d'arbres. Trouver la partition optimale est un problème **NP-complet**.

### Algorithme glouton

En pratique, nous utilisons une procédure **gloutonne** (greedy) qui développe l'arbre itérativement, un nœud à la fois. Les algorithmes classiques (CART, C4.5, ID3) suivent cette approche.

À chaque nœud $i$, soit $\mathcal{D}_i = \{(\boldsymbol{x}_n, y_n) : n \in N_i\}$ l'ensemble des exemples qui atteignent ce nœud. Nous cherchons la meilleure **division** pour minimiser l'erreur dans les sous-arbres enfants.

### Types de divisions

**Pour une caractéristique continue** $j$: on compare avec un seuil $t$
- Enfant gauche: $\mathcal{D}_i^L(j, t) = \{(\boldsymbol{x}_n, y_n) \in \mathcal{D}_i : x_{nj} \leq t\}$
- Enfant droit: $\mathcal{D}_i^R(j, t) = \{(\boldsymbol{x}_n, y_n) \in \mathcal{D}_i : x_{nj} > t\}$

L'ensemble des seuils candidats $\mathcal{T}_j$ est obtenu en triant les valeurs uniques de $\{x_{nj}\}$.

**Pour une caractéristique catégorielle** avec $K_j$ valeurs: on teste chaque valeur possible, créant $K_j$ divisions binaires.

### Critère de division

Nous choisissons la meilleure caractéristique $j_i$ et le meilleur seuil $t_i$ en minimisant:

$$
(j_i, t_i) = \arg\min_{j \in \{1, \ldots, D\}} \min_{t \in \mathcal{T}_j} \left[ \frac{|\mathcal{D}_i^L(j, t)|}{|\mathcal{D}_i|} c(\mathcal{D}_i^L(j, t)) + \frac{|\mathcal{D}_i^R(j, t)|}{|\mathcal{D}_i|} c(\mathcal{D}_i^R(j, t)) \right]
$$

où $c(\cdot)$ est une fonction de **coût** mesurant l'impureté d'un nœud.

## Critères d'impureté

### Pour la régression: erreur quadratique

En régression, nous utilisons l'erreur quadratique moyenne:
$$
\text{cost}(\mathcal{D}_i) = \frac{1}{|\mathcal{D}_i|} \sum_{n \in \mathcal{D}_i} (y_n - \bar{y}_i)^2
$$

où $\bar{y}_i = \frac{1}{|\mathcal{D}_i|} \sum_{n \in \mathcal{D}_i} y_n$ est la moyenne des réponses au nœud $i$.

### Pour la classification: indice de Gini

L'**indice de Gini** mesure l'impureté d'un nœud:
$$
G_i = \sum_{c=1}^C \hat{\pi}_{ic}(1 - \hat{\pi}_{ic}) = 1 - \sum_{c=1}^C \hat{\pi}_{ic}^2
$$

Propriétés:
- $G_i = 0$ si le nœud est **pur** (tous les exemples de la même classe)
- $G_i$ est maximal quand les classes sont équiprobables
- $G_i \in [0, 1 - 1/C]$ pour $C$ classes

L'indice de Gini peut être interprété comme la probabilité de mal classifier un exemple choisi au hasard si on lui assigne une classe selon la distribution $\hat{\boldsymbol{\pi}}_i$.

### Pour la classification: entropie

Alternativement, nous pouvons utiliser l'**entropie** (aussi appelée **déviance**):
$$
H_i = \mathbb{H}(\hat{\boldsymbol{\pi}}_i) = -\sum_{c=1}^C \hat{\pi}_{ic} \log \hat{\pi}_{ic}
$$

Propriétés similaires:
- $H_i = 0$ pour un nœud pur
- $H_i$ est maximal pour une distribution uniforme
- $H_i \in [0, \log C]$

Le **gain d'information** est la réduction d'entropie obtenue par une division:
$$
\text{IG}(j, t) = H_i - \left[ \frac{|\mathcal{D}_i^L|}{|\mathcal{D}_i|} H_L + \frac{|\mathcal{D}_i^R|}{|\mathcal{D}_i|} H_R \right]
$$

### Comparaison des critères

En pratique, l'indice de Gini et l'entropie donnent des résultats similaires. Le Gini est légèrement plus rapide à calculer car il évite le logarithme. L'entropie a l'avantage d'être liée à la théorie de l'information.

## Régularisation et élagage

### Le problème du surapprentissage

Si nous laissons l'arbre croître jusqu'à ce que chaque feuille contienne un seul exemple, nous obtenons une erreur d'entraînement de 0 mais une très mauvaise généralisation.

### Stratégies de régularisation

**Arrêt précoce (pre-pruning)**: on stoppe la croissance selon des heuristiques:
- Profondeur maximale atteinte
- Nombre minimal d'exemples par feuille
- Gain d'information insuffisant

**Élagage (post-pruning)**: on fait croître l'arbre complet, puis on fusionne les sous-arbres avec leur parent:
- Élagage par coût-complexité (CART)
- Élagage par erreur réduite
- Élagage pessimiste (C4.5)

L'élagage post-construction est généralement préféré car il évite de stopper prématurément une division qui pourrait être bénéfique plus tard.

### Régularisation par coût-complexité

CART utilise un critère de coût-complexité:
$$
C_\alpha(T) = \sum_{j=1}^{|T|} \sum_{\boldsymbol{x}_n \in R_j} \ell(y_n, w_j) + \alpha |T|
$$

où $|T|$ est le nombre de feuilles et $\alpha \geq 0$ est un hyperparamètre de complexité. Plus $\alpha$ est grand, plus l'arbre sera petit.

## L'instabilité des arbres

### Sensibilité aux données

Les arbres de décision sont **instables**: de petits changements dans les données d'entraînement peuvent produire des arbres très différents.

Cela vient du processus de construction glouton: si une division précoce change, toute la structure en aval est affectée. Cette propriété est parfois appelée **haute variance**.

### Illustration

Considérons les données Iris. En entraînant un arbre sur l'ensemble complet, nous obtenons une certaine frontière de décision. Si nous omettons un seul point (proche de la frontière), l'arbre résultant peut avoir une structure complètement différente.

Cette instabilité est un **inconvénient** pour un modèle unique, mais un **avantage** pour les méthodes d'ensemble: des arbres divers peuvent être combinés pour obtenir des prédictions plus robustes.

## Arbres vs autres modèles

### Comparaison avec les modèles linéaires

| Aspect | Arbres de décision | Modèles linéaires |
|--------|-------------------|-------------------|
| Frontières | Parallèles aux axes | Hyperplans obliques |
| Interactions | Capturées naturellement | Nécessitent des termes explicites |
| Interprétabilité | Règles si-alors | Coefficients |
| Extrapolation | Constante (dernière région) | Linéaire |

### Comparaison avec k-NN

| Aspect | Arbres de décision | k-NN |
|--------|-------------------|------|
| Entraînement | Lent (construction de l'arbre) | Aucun |
| Prédiction | Rapide (parcours de l'arbre) | Lent (calcul des distances) |
| Mémoire | Modèle compact | Stocke toutes les données |
| Interprétabilité | Règles explicites | Exemples similaires |

## Résumé

Les arbres de décision sont des modèles puissants et interprétables qui:

- **Partitionnent** l'espace d'entrée en régions avec des prédictions simples
- Utilisent des critères d'**impureté** (Gini, entropie, MSE) pour guider la construction
- Sont appris par un algorithme **glouton** car l'optimisation exacte est NP-complète
- Nécessitent une **régularisation** (élagage, profondeur maximale) pour éviter le surapprentissage
- Sont **instables** mais cette propriété les rend idéaux pour les méthodes d'ensemble

Au chapitre suivant, nous verrons comment combiner plusieurs arbres pour créer des modèles plus robustes et performants: le **bagging** et les **forêts aléatoires**.

## Exercices

```{admonition} Exercice 1: Construction manuelle
:class: tip

Considérez les données suivantes pour un problème de classification binaire:

| $x_1$ | $x_2$ | $y$ |
|-------|-------|-----|
| 1 | 2 | + |
| 2 | 1 | + |
| 3 | 3 | - |
| 4 | 2 | - |

1. Calculez l'entropie initiale du nœud racine.
2. Pour la caractéristique $x_1$ avec seuil $t = 2.5$, calculez l'entropie des deux enfants et le gain d'information.
3. Faites de même pour $x_2$ avec $t = 2.5$.
4. Quelle division est préférable?
```

```{admonition} Exercice 2: Indice de Gini
:class: tip

Un nœud contient 60 exemples de classe A et 40 exemples de classe B.
1. Calculez l'indice de Gini de ce nœud.
2. Si une division produit un enfant gauche avec (50 A, 10 B) et un enfant droit avec (10 A, 30 B), calculez l'indice de Gini pondéré après division.
3. Cette division réduit-elle l'impureté? De combien?
```

```{admonition} Exercice 3: Régression
:class: tip

Considérez un arbre de régression avec les données $\{(1, 2), (2, 4), (3, 3), (4, 8)\}$ où la première coordonnée est $x$ et la seconde $y$.
1. Calculez l'erreur quadratique du nœud racine (avant toute division).
2. Évaluez l'erreur après division à $x = 2.5$.
3. Évaluez l'erreur après division à $x = 3.5$.
4. Quelle division est meilleure?
```

```{admonition} Exercice 4: Profondeur et complexité
:class: tip

Un arbre binaire complet de profondeur $d$ a combien de:
1. Feuilles?
2. Nœuds internes?
3. Nœuds au total?

Si chaque nœud interne teste une caractéristique parmi $D$ avec un seuil parmi $N$ valeurs, combien de paramètres faut-il stocker pour représenter l'arbre?
```
