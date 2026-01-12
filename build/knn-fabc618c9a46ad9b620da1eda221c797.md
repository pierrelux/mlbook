# K plus proches voisins

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Expliquer le fonctionnement de l'algorithme des k plus proches voisins
- Définir et appliquer différentes fonctions de distance
- Interpréter le diagramme de Voronoï
- Expliquer le fléau de la dimensionnalité
- Analyser l'effet du paramètre k sur le compromis biais-variance
- Implémenter l'algorithme k-ppv pour la classification et la régression
```

## Introduction

Les k plus proches voisins (k-ppv, en anglais *k-nearest neighbors*, k-NN) constituent l'une des méthodes d'apprentissage les plus simples et les plus intuitives. L'idée fondamentale est que des exemples similaires devraient avoir des étiquettes similaires. Pour prédire l'étiquette d'un nouvel exemple, nous identifions ses voisins les plus proches dans l'ensemble d'entraînement et combinons leurs étiquettes.

Cette méthode est dite **non paramétrique**: elle ne suppose pas de forme fonctionnelle particulière pour la relation entre entrées et sorties. Elle est également **à mémoire** (en anglais *memory-based* ou *instance-based*): l'entraînement consiste simplement à stocker les données, et le travail se fait au moment de l'inférence.

## L'algorithme k-ppv

### Formulation

Soit $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ un ensemble d'entraînement. Pour classifier un nouveau point $x$, nous procédons en deux étapes:

1. **Trouver les voisins**: Identifier les $k$ points de $\mathcal{D}$ les plus proches de $x$ selon une métrique donnée
2. **Agréger les votes**: Prédire la classe majoritaire parmi ces $k$ voisins

Formellement, soit $\mathcal{N}_k(x)$ l'ensemble des indices des $k$ plus proches voisins de $x$. La prédiction est:

$$
\hat{y} = \arg\max_{c} \sum_{i \in \mathcal{N}_k(x)} \mathbb{1}_{y_i = c}
$$

où la somme compte le nombre de voisins appartenant à chaque classe $c$.

### Version probabiliste

Plutôt que de retourner uniquement la classe prédite, nous pouvons estimer la distribution sur les classes:

$$
p(y = c | x, \mathcal{D}) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} \mathbb{1}_{y_i = c}
$$

Cette probabilité est simplement la proportion de voisins appartenant à la classe $c$. La prédiction déterministe correspond au mode de cette distribution.

### Exemple

Considérons le jeu de données Iris avec trois classes: Setosa, Versicolor et Virginica. Pour un nouveau point $x$ avec $k = 5$, supposons que les 5 plus proches voisins aient les étiquettes:

| Voisin | Étiquette |
|--------|-----------|
| 1 | Setosa |
| 2 | Setosa |
| 3 | Versicolor |
| 4 | Setosa |
| 5 | Versicolor |

Les probabilités estimées sont:
- $p(\text{Setosa} | x) = 3/5 = 0.6$
- $p(\text{Versicolor} | x) = 2/5 = 0.4$
- $p(\text{Virginica} | x) = 0/5 = 0$

La classe prédite est Setosa.

## Fonctions de distance

Le choix de la distance détermine la notion de similarité entre exemples. Ce choix encode nos hypothèses sur la structure du problème.

### Définition formelle

Une **fonction de distance** (ou métrique) $d: \mathcal{X} \times \mathcal{X} \to [0, \infty)$ doit satisfaire trois axiomes:

1. **Identité**: $d(x, y) = 0 \Leftrightarrow x = y$
2. **Symétrie**: $d(x, y) = d(y, x)$
3. **Inégalité triangulaire**: $d(x, z) \leq d(x, y) + d(y, z)$

Ces conditions impliquent la non-négativité: $d(x, y) \geq 0$ pour tous $x, y$.

### Distance euclidienne

La **distance euclidienne** (ou distance L2) est le choix le plus courant:

$$
d_2(x, y) = \sqrt{\sum_{j=1}^{d} (x_j - y_j)^2} = \|x - y\|_2
$$

Cette distance correspond à la longueur du segment de droite entre les deux points, la distance "à vol d'oiseau".

### Distance de Manhattan

La **distance de Manhattan** (ou distance L1, taxicab distance) est:

$$
d_1(x, y) = \sum_{j=1}^{d} |x_j - y_j| = \|x - y\|_1
$$

Elle mesure la distance parcourue en suivant une grille, comme dans les rues d'une ville. Contrairement à la distance euclidienne, elle favorise les chemins alignés avec les axes.

### Distance de Mahalanobis

La **distance de Mahalanobis** prend en compte la corrélation entre les variables:

$$
d_M(x, y) = \sqrt{(x - y)^\top M (x - y)}
$$

où $M$ est une matrice définie positive. Si $M = I$, nous retrouvons la distance euclidienne. Si $M = \Sigma^{-1}$ où $\Sigma$ est la matrice de covariance des données, la distance de Mahalanobis normalise les variables et supprime les corrélations.

### Normes et distances

Dans un espace vectoriel, toute norme $\|\cdot\|$ induit une distance:

$$
d(x, y) = \|x - y\|
$$

La norme $\ell_p$ est définie par:

$$
\|x\|_p = \left(\sum_{j=1}^{d} |x_j|^p\right)^{1/p}
$$

Les cas $p = 1$ et $p = 2$ correspondent aux distances de Manhattan et euclidienne respectivement. La limite $p \to \infty$ donne la norme infinie $\|x\|_\infty = \max_j |x_j|$.

## Diagramme de Voronoï

Le cas $k = 1$ induit une structure géométrique particulière appelée **diagramme de Voronoï** (ou tesselation de Voronoï).

### Définition

Pour un ensemble de points $\{x_1, \ldots, x_N\}$ appelés **germes**, le diagramme de Voronoï partitionne l'espace en **cellules**. La cellule $V_i$ associée au germe $x_i$ contient tous les points plus proches de $x_i$ que de tout autre germe:

$$
V_i = \{x \in \mathbb{R}^d : d(x, x_i) \leq d(x, x_j) \text{ pour tout } j \neq i\}
$$

Les frontières entre cellules sont des hyperplans (en dimension $d > 2$) ou des segments (en dimension 2).

### Interprétation

Avec le 1-ppv, la frontière de décision suit exactement le diagramme de Voronoï. Tout point dans la cellule $V_i$ est classifié selon l'étiquette de $x_i$.

Pour $k = 1$, l'erreur d'entraînement est exactement zéro: chaque point d'entraînement est son propre plus proche voisin et est donc correctement classifié. Cette interpolation parfaite des données d'entraînement peut mener à un surapprentissage sévère.

## Effet du paramètre k

Le paramètre $k$ contrôle la complexité du modèle et influence le compromis biais-variance.

### Petites valeurs de k

Avec un petit $k$ (notamment $k = 1$):
- La frontière de décision est très irrégulière
- Le modèle s'adapte étroitement aux données d'entraînement
- La variance est élevée: de petits changements dans les données produisent des prédictions très différentes
- Le biais est faible: le modèle peut capturer des structures complexes
- Risque de surapprentissage

### Grandes valeurs de k

Avec un grand $k$ (jusqu'à $k = N$):
- La frontière de décision devient plus lisse
- Le modèle "moyenne" sur de nombreux exemples
- La variance est faible: les prédictions sont stables
- Le biais est élevé: le modèle peut manquer des structures locales
- Risque de sous-apprentissage

Le cas extrême $k = N$ prédit toujours la classe majoritaire globale, ignorant complètement l'entrée $x$.

### Choix de k

Le choix optimal de $k$ dépend du problème et se fait typiquement par validation croisée. En pratique:
- Des valeurs impaires évitent les égalités dans les votes
- Une règle empirique suggère $k \approx \sqrt{N}$
- La courbe d'erreur de validation en fonction de $k$ guide le choix

## Fléau de la dimensionnalité

Les méthodes basées sur la distance souffrent particulièrement du **fléau de la dimensionnalité** (en anglais *curse of dimensionality*). En haute dimension, les distances perdent leur pouvoir discriminant.

### Phénomène

Considérons des points uniformément distribués dans un hypercube $[0, 1]^d$. Pour contenir une fraction $p$ des points, nous avons besoin d'un hypercube de côté $r = p^{1/d}$.

| Dimension $d$ | Côté $r$ pour $p = 0.1$ |
|---------------|-------------------------|
| 1 | 0.10 |
| 2 | 0.32 |
| 5 | 0.63 |
| 10 | 0.79 |
| 100 | 0.98 |

En dimension 100, pour capturer 10% des points, nous avons besoin d'un hypercube couvrant 98% de chaque dimension. Les "voisins" ne sont plus vraiment proches.

### Conséquences

1. **Distance minimale croissante**: La distance au plus proche voisin augmente avec la dimension. Les voisins deviennent tous approximativement équidistants.

2. **Volume des hypersphères**: Le volume d'une hypersphère de rayon fixe décroît exponentiellement avec la dimension. La plupart du volume d'un hypercube se concentre près de sa surface.

3. **Densité des données**: Pour maintenir une densité constante de points, le nombre d'exemples requis croît exponentiellement avec la dimension.

### Atténuation

Plusieurs stratégies atténuent le fléau de la dimensionnalité:
- **Réduction de dimension**: PCA, sélection de variables
- **Distances adaptatives**: Distance de Mahalanobis, apprentissage de métrique
- **Régularisation**: Contraintes sur le modèle

## Régression par k-ppv

L'algorithme k-ppv s'adapte naturellement à la régression. Plutôt que de prendre un vote majoritaire, nous moyennons les valeurs cibles des voisins:

$$
\hat{y} = \frac{1}{k} \sum_{i \in \mathcal{N}_k(x)} y_i
$$

Cette moyenne locale estime l'espérance conditionnelle $\mathbb{E}[Y | X = x]$.

### Pondération

Une variante pondère les contributions des voisins selon leur distance:

$$
\hat{y} = \frac{\sum_{i \in \mathcal{N}_k(x)} w_i y_i}{\sum_{i \in \mathcal{N}_k(x)} w_i}
$$

où $w_i = 1/d(x, x_i)$ ou $w_i = \exp(-d(x, x_i)^2/\sigma^2)$. Les voisins plus proches ont plus d'influence sur la prédiction.

## Complexité computationnelle

### Entraînement

L'entraînement est trivial: $O(1)$ si nous comptons simplement stocker les données, ou $O(N)$ pour les copier.

### Inférence

L'inférence naïve requiert $O(Nd)$ opérations par requête:
- Calculer la distance à chaque exemple: $O(d)$ par exemple
- Trouver les $k$ plus petites: $O(N)$ ou $O(N \log k)$ avec un tas

Pour de grands ensembles de données, c'est prohibitif. Des structures de données accélèrent la recherche:
- **Arbres k-d**: $O(\log N)$ en moyenne pour la recherche du plus proche voisin
- **Hachage sensible à la localité** (LSH): recherche approximative en temps sous-linéaire
- **Indexation par graphe**: HNSW et autres méthodes modernes

## Implémentation

Voici une implémentation simple en Python:

```python
import numpy as np

def knn_classify(X_train, y_train, X_test, k=3):
    """Classification par k plus proches voisins.
    
    Args:
        X_train: Matrice d'entraînement (N x d)
        y_train: Étiquettes d'entraînement (N,)
        X_test: Points à classifier (M x d)
        k: Nombre de voisins
        
    Returns:
        Prédictions (M,)
    """
    predictions = []
    for x in X_test:
        # Calculer les distances à tous les points d'entraînement
        distances = np.sqrt(np.sum((X_train - x)**2, axis=1))
        
        # Trouver les indices des k plus proches
        k_nearest_idx = np.argsort(distances)[:k]
        
        # Vote majoritaire
        k_nearest_labels = y_train[k_nearest_idx]
        unique, counts = np.unique(k_nearest_labels, return_counts=True)
        predictions.append(unique[np.argmax(counts)])
        
    return np.array(predictions)
```

```python
def knn_regression(X_train, y_train, X_test, k=3):
    """Régression par k plus proches voisins.
    
    Args:
        X_train: Matrice d'entraînement (N x d)
        y_train: Cibles d'entraînement (N,)
        X_test: Points à prédire (M x d)
        k: Nombre de voisins
        
    Returns:
        Prédictions (M,)
    """
    predictions = []
    for x in X_test:
        distances = np.sqrt(np.sum((X_train - x)**2, axis=1))
        k_nearest_idx = np.argsort(distances)[:k]
        k_nearest_values = y_train[k_nearest_idx]
        predictions.append(np.mean(k_nearest_values))
        
    return np.array(predictions)
```

## Avantages et limitations

### Avantages

- **Simplicité**: Algorithme intuitif, facile à comprendre et implémenter
- **Pas d'hypothèse paramétrique**: S'adapte à des frontières de décision complexes
- **Pas d'entraînement**: Nouvelles données ajoutées sans réentraînement
- **Interprétabilité locale**: Les prédictions s'expliquent par les exemples voisins

### Limitations

- **Coût d'inférence**: $O(Nd)$ par requête dans le cas naïf
- **Stockage**: Tout l'ensemble d'entraînement doit être conservé
- **Sensibilité aux échelles**: Les variables doivent être normalisées
- **Fléau de la dimensionnalité**: Performance dégradée en haute dimension
- **Sensibilité au bruit**: Un seul exemple mal étiqueté affecte les voisinages

### Méthodes non paramétriques vs paramétriques

Les k-ppv sont une méthode **non paramétrique**: la complexité du modèle croît avec la taille des données. Lors du déploiement, il faut conserver l'ensemble d'entraînement en mémoire — les données *sont* le modèle.

À l'opposé, les méthodes **paramétriques** distillent l'information des données dans un vecteur de paramètres $\theta$ de taille fixe. Un réseau de neurones, par exemple, peut être entraîné sur des milliards d'exemples, mais lors de l'inférence, seuls les poids du réseau sont nécessaires — pas les données d'entraînement.

| | Non paramétrique | Paramétrique |
|--|------------------|--------------|
| **Modèle** | Les données elles-mêmes | Un vecteur $\theta \in \mathbb{R}^p$ |
| **Complexité** | Croît avec $N$ | Fixe (indépendante de $N$) |
| **Inférence** | Requiert les données | Requiert seulement $\theta$ |
| **Exemples** | k-ppv, processus gaussiens | Régression linéaire, réseaux de neurones |

Cette distinction a des implications pratiques importantes. Un modèle de langage comme GPT est **paramétrique**: bien qu'entraîné sur une portion significative d'internet, il ne garde pas ces textes en mémoire. Lors d'une requête, seuls ses paramètres (plusieurs milliards de nombres) sont consultés. Déployer un tel modèle sur un téléphone serait impossible s'il fallait stocker toutes les données d'entraînement.

## Résumé

Ce chapitre a présenté l'algorithme des k plus proches voisins:

- Le **k-ppv** classifie un exemple par vote majoritaire parmi ses $k$ plus proches voisins
- Le choix de la **distance** encode les hypothèses sur la similarité
- Le **paramètre k** contrôle le compromis biais-variance
- Le **diagramme de Voronoï** décrit la partition de l'espace pour $k = 1$
- Le **fléau de la dimensionnalité** limite l'efficacité en haute dimension
- L'algorithme s'adapte naturellement à la **régression** par moyennage local

Les k-ppv illustrent parfaitement la tension entre mémorisation et généralisation: avec $k=1$, le modèle mémorise les données et atteint une erreur d'entraînement nulle, mais généralise mal. Cette méthode est **non paramétrique** — les données *sont* le modèle, ce qui implique un coût mémoire et computationnel à l'inférence qui croît avec $N$.

Ces limitations motivent une approche différente. Plutôt que de garder toutes les données en mémoire, nous pouvons chercher à *distiller* l'information des données dans un ensemble fixe de **paramètres**. Le chapitre suivant formalise cette idée: l'apprentissage devient un problème d'**optimisation** où nous cherchons les paramètres qui minimisent une fonction de perte sur les données d'entraînement.
