# Méthodes d'ensemble

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Expliquer le principe de l'apprentissage en ensemble
- Analyser le compromis biais-variance
- Décrire le bagging et l'échantillonnage bootstrap
- Implémenter les forêts aléatoires
- Comprendre les méthodes de comité et le stacking
- Justifier pourquoi les arbres de décision sont des modèles de base idéaux
```

## Le principe des ensembles

Une façon simple de **réduire la variance** d'un modèle est de faire la moyenne de plusieurs modèles:

$$
f(y \mid \boldsymbol{x}) = \frac{1}{|\mathcal{M}|} \sum_{m \in \mathcal{M}} f_m(y \mid \boldsymbol{x})
$$

où $f_m$ est le $m$-ième **modèle de base** (ou *apprenant faible*). L'ensemble aura:
- Un **biais similaire** aux modèles de base
- Une **variance plus faible** grâce à l'effet de moyennage

Cette idée simple mais puissante est à la base de certains des algorithmes les plus performants en apprentissage automatique.

## Le compromis biais-variance

### Décomposition de l'erreur

Soit $\hat{\theta} = \hat{\theta}(\mathcal{D})$ un estimateur et $\bar{\theta} = \mathbb{E}[\hat{\theta}]$ son espérance (sous la distribution des données $p(\mathcal{D} \mid \theta^*)$). L'erreur quadratique moyenne se décompose comme:

$$
\begin{aligned}
\mathbb{E}\left[(\hat{\theta} - \theta^*)^2\right] &= \mathbb{E}\left[\left[(\hat{\theta} - \bar{\theta}) + (\bar{\theta} - \theta^*)\right]^2\right] \\
&= \mathbb{E}\left[(\hat{\theta} - \bar{\theta})^2\right] + (\bar{\theta} - \theta^*)^2 \\
&= \mathbb{V}[\hat{\theta}] + \text{biais}^2(\hat{\theta})
\end{aligned}
$$

Ainsi:
$$
\text{MSE} = \text{variance} + \text{biais}^2
$$

### Interprétation

- **Biais**: erreur systématique due aux hypothèses du modèle
- **Variance**: sensibilité aux fluctuations des données d'entraînement

Un modèle simple (comme la régression linéaire) a typiquement:
- **Biais élevé**: ne peut pas capturer les relations complexes
- **Variance faible**: stable face aux variations des données

Un modèle complexe (comme un arbre profond) a typiquement:
- **Biais faible**: peut approximer des fonctions complexes
- **Variance élevée**: très sensible aux données d'entraînement

Les méthodes d'ensemble visent à **réduire la variance** tout en conservant un biais faible.

## Méthodes de comité

### Vote majoritaire

Pour la classification, plutôt que de moyenner les probabilités, on peut utiliser un **vote majoritaire**:

$$
\hat{y} = \arg\max_c \sum_{m=1}^M \mathbb{I}(f_m(\boldsymbol{x}) = c)
$$

Cette approche est parfois appelée une **méthode de comité**.

### Analyse théorique

Supposons que chaque modèle de base est un classificateur binaire avec une précision $\theta > 0.5$, et que les erreurs sont **indépendantes**. Soit:
- $Y_m \in \{0, 1\}$ la prédiction du $m$-ième modèle
- $S = \sum_{m=1}^M Y_m$ le nombre de votes pour la classe 1 (supposée correcte)

Le prédicteur final prédit la classe 1 si $S > M/2$.

La probabilité que l'ensemble soit correct est:
$$
p = \Pr(S > M/2) = 1 - B(M/2, M, \theta)
$$

où $B(x, M, \theta)$ est la fonction de répartition de la loi binomiale.

### Exemple numérique

Avec $\theta = 0.51$ (chaque modèle légèrement meilleur que le hasard):
- $M = 1000$ modèles: $p \approx 0.73$
- $M = 10000$ modèles: $p \approx 0.97$

La **sagesse de la foule** (wisdom of crowds) en action!

### Limitation

Cette amélioration repose sur l'hypothèse que les modèles font des **erreurs indépendantes**, ce qui n'est généralement pas le cas en pratique. Nous devons donc chercher des méthodes pour **diversifier** les prédictions des modèles de base.

## Bagging

### Principe

Le **bagging** (pour *Bootstrap AGGregatING*) crée de la diversité en entraînant chaque modèle sur un **sous-ensemble différent** des données, échantillonné avec remplacement.

Cette technique s'appelle l'**échantillonnage bootstrap**: on tire $N$ exemples uniformément avec remplacement parmi les $N$ exemples originaux.

### Propriétés du bootstrap

En moyenne, chaque modèle de base voit seulement **63%** des données. En effet, la probabilité qu'un exemple donné ne soit *pas* sélectionné dans $N$ tirages est:

$$
\left(1 - \frac{1}{N}\right)^N \xrightarrow{N \to \infty} e^{-1} \approx 0.37
$$

Les 37% d'exemples non utilisés par un modèle sont appelés les **instances hors sac** (*out-of-bag*, OOB). Elles permettent une forme de validation croisée gratuite.

### Algorithme

```
Entrée: Données D = {(x_n, y_n)}, nombre de modèles M
Pour m = 1 à M:
    D_m ← échantillon bootstrap de D (N tirages avec remplacement)
    f_m ← entraîner un modèle sur D_m
Sortie: f(x) = (1/M) Σ_m f_m(x)  [régression]
        f(x) = vote_majoritaire({f_m(x)})  [classification]
```

### Avantages du bagging

1. **Robustesse**: empêche l'ensemble de donner trop d'importance à un exemple particulier
2. **Généralisation**: réduit le surapprentissage
3. **Parallélisable**: chaque modèle peut être entraîné indépendamment
4. **Estimation OOB**: pas besoin d'ensemble de validation séparé

### Quand le bagging fonctionne

Le bagging fonctionne bien quand les modèles de base sont **instables** — c'est-à-dire que l'omission de certaines données change significativement le modèle.

**Bons candidats**:
- Arbres de décision (très instables)
- Réseaux de neurones (avec petits ensembles)

**Mauvais candidats**:
- k-NN (très stable: changer quelques données ne change pas beaucoup les voisins)
- Régression linéaire (relativement stable)

## Forêts aléatoires

### Au-delà du bagging

Les **forêts aléatoires** (*Random Forests*) étendent le bagging en ajoutant une source supplémentaire de randomisation: à chaque division d'un arbre, seul un **sous-ensemble aléatoire de caractéristiques** est considéré.

### Algorithme

À chaque nœud $i$, au lieu de chercher la meilleure division parmi toutes les caractéristiques $\{1, \ldots, D\}$, on tire un sous-ensemble aléatoire $S_i \subset \{1, \ldots, D\}$ et on optimise:

$$
(j_i, t_i) = \arg\min_{j \in S_i} \min_{t \in \mathcal{T}_j} \left[ \frac{|\mathcal{D}_i^L(j, t)|}{|\mathcal{D}_i|} c(\mathcal{D}_i^L(j, t)) + \frac{|\mathcal{D}_i^R(j, t)|}{|\mathcal{D}_i|} c(\mathcal{D}_i^R(j, t)) \right]
$$

La taille typique du sous-ensemble est:
- $|S_i| = \sqrt{D}$ pour la classification
- $|S_i| = D/3$ pour la régression

### Pourquoi ça fonctionne

L'échantillonnage de caractéristiques **décorrèle** les arbres:
- Dans le bagging simple, les arbres tendent à avoir des structures similaires (mêmes caractéristiques importantes aux premières divisions)
- Avec les forêts aléatoires, les arbres sont plus divers

Si les arbres sont moins corrélés, la réduction de variance par moyennage est plus efficace.

### Importance des variables

Un avantage des forêts aléatoires est de fournir une mesure d'**importance des variables**:

1. **Importance par diminution d'impureté**: somme des réductions d'impureté à chaque division utilisant une variable, moyennée sur tous les arbres

2. **Importance par permutation**: mesure la dégradation de la performance OOB quand on permute aléatoirement les valeurs d'une variable

Ces mesures aident à l'interprétation et à la sélection de caractéristiques.

## Stacking

### Combinaison apprise

Au lieu d'utiliser une moyenne non pondérée ou un vote, on peut **apprendre** comment combiner les modèles:

$$
f(y \mid \boldsymbol{x}) = \sum_{m \in \mathcal{M}} w_m f_m(y \mid \boldsymbol{x})
$$

Cette technique s'appelle le **stacking** (*stacked generalization*).

### Procédure

1. **Niveau 0**: entraîner les modèles de base $f_1, \ldots, f_M$
2. **Création des méta-caractéristiques**: pour chaque exemple, créer un vecteur $(f_1(\boldsymbol{x}), \ldots, f_M(\boldsymbol{x}))$
3. **Niveau 1**: entraîner un méta-apprenant sur ces nouvelles caractéristiques

**Important**: les poids $w_m$ doivent être entraînés sur un **ensemble séparé**, sinon le stacking mettrait tout le poids sur le modèle de base le plus performant sur l'ensemble d'entraînement.

### Avantages

- Permet de combiner des modèles de types différents
- Peut apprendre des interactions complexes entre modèles
- Souvent utilisé dans les compétitions de ML

## Pourquoi les arbres sont idéaux pour les ensembles

### Haute variance, faible biais

Les arbres de décision profonds ont:
- **Faible biais**: peuvent approximer n'importe quelle fonction
- **Haute variance**: très sensibles aux données d'entraînement

C'est exactement ce que les méthodes d'ensemble savent exploiter: réduire la variance tout en conservant le faible biais.

### Instabilité

L'instabilité des arbres signifie que le bagging produit des modèles véritablement différents, ce qui améliore la diversité de l'ensemble.

### Efficacité computationnelle

- Les arbres s'entraînent rapidement
- Pas d'optimisation itérative (comme la descente de gradient)
- Facilement parallélisables

### Pas de pré-traitement

- Pas besoin de normaliser les données
- Gèrent les caractéristiques mixtes (continues et catégorielles)
- Robustes aux valeurs aberrantes

## Comparaison: Bagging vs Forêts aléatoires

| Aspect | Bagging | Forêts aléatoires |
|--------|---------|-------------------|
| Randomisation | Données (bootstrap) | Données + caractéristiques |
| Corrélation entre arbres | Plus élevée | Plus faible |
| Réduction de variance | Bonne | Meilleure |
| Coût computationnel | Légèrement plus élevé | Légèrement plus bas par arbre |
| Interprétabilité | Difficile | Importance des variables |

En pratique, les forêts aléatoires sont presque toujours préférées au bagging simple.

## Résumé

Les méthodes d'ensemble exploitent le principe que **plusieurs modèles sont meilleurs qu'un seul**:

- Le **compromis biais-variance** explique pourquoi: le moyennage réduit la variance
- Le **bagging** crée de la diversité par échantillonnage bootstrap
- Les **forêts aléatoires** ajoutent l'échantillonnage de caractéristiques
- Le **stacking** apprend à combiner optimalement les prédictions
- Les **arbres de décision** sont des modèles de base idéaux grâce à leur haute variance

Au chapitre suivant, nous verrons le **boosting**, une approche complémentaire qui réduit le **biais** plutôt que la variance.

## Exercices

```{admonition} Exercice 1: Probabilité de sélection bootstrap
:class: tip

1. Calculez exactement $(1 - 1/N)^N$ pour $N = 10, 100, 1000$.
2. Vérifiez que cela converge vers $e^{-1} \approx 0.368$.
3. Si $N = 1000$, en moyenne combien d'exemples uniques sont présents dans un échantillon bootstrap?
```

```{admonition} Exercice 2: Vote majoritaire
:class: tip

Considérez un ensemble de 7 classificateurs binaires, chacun avec une précision de 60%.

1. Calculez la probabilité que le vote majoritaire soit correct (en supposant l'indépendance des erreurs).
2. Combien de classificateurs doivent voter correctement pour que l'ensemble soit correct?
3. Que se passe-t-il si la précision individuelle tombe à 49%?
```

```{admonition} Exercice 3: Variance de l'ensemble
:class: tip

Soit $\hat{f}_m$ des estimateurs i.i.d. avec variance $\sigma^2$. L'estimateur de l'ensemble est $\hat{f} = \frac{1}{M} \sum_{m=1}^M \hat{f}_m$.

1. Calculez $\mathbb{V}[\hat{f}]$.
2. Que se passe-t-il quand $M \to \infty$?
3. Si les estimateurs sont corrélés avec $\text{Cov}(\hat{f}_i, \hat{f}_j) = \rho \sigma^2$ pour $i \neq j$, que devient la variance de l'ensemble?
```

```{admonition} Exercice 4: Forêt aléatoire
:class: tip

Un jeu de données a $D = 100$ caractéristiques. Pour une forêt aléatoire de classification:

1. Combien de caractéristiques sont typiquement considérées à chaque division?
2. Si un arbre a une profondeur de 10, combien de caractéristiques différentes peut-il utiliser au maximum?
3. Expliquez pourquoi échantillonner les caractéristiques décorrèle les arbres.
```
