# Introduction

## Qu'est-ce qu'un modèle?

Le mot *modèle* est omniprésent en apprentissage machine, mais sa signification précise est rarement explicitée. Un modèle est une représentation simplifiée d'un phénomène — une abstraction qui capture certains aspects de la réalité tout en en ignorant d'autres. Cette idée n'est pas propre à l'informatique: les physiciens utilisent des modèles (la mécanique newtonienne ignore les effets relativistes), les économistes aussi (l'homo economicus ignore l'irrationalité humaine), et les statisticiens depuis toujours.

En apprentissage machine, un modèle est typiquement une **fonction paramétrée** $f_\theta: \mathcal{X} \to \mathcal{Y}$ qui associe des entrées $x$ à des sorties $y$. Les paramètres $\theta$ déterminent le comportement de cette fonction. *Apprendre*, c'est trouver les valeurs de $\theta$ qui font que $f_\theta$ se comporte de façon utile — par exemple, qu'elle prédise correctement des étiquettes sur de nouvelles données.

Prenons un exemple concret. Supposons qu'on veuille prédire le prix d'une maison à partir de sa superficie. Un modèle linéaire simple serait:

$$
f_\theta(x) = \theta_0 + \theta_1 x
$$

où $x$ est la superficie, $\theta_0$ est l'ordonnée à l'origine, et $\theta_1$ est la pente. Ce modèle fait une hypothèse forte: la relation entre superficie et prix est linéaire. Cette hypothèse est probablement fausse — les très grandes maisons ne suivent pas la même dynamique que les petites — mais elle peut être *utile* si elle capture l'essentiel de la variation pour les maisons qui nous intéressent.

C'est là une tension fondamentale: un modèle trop simple ignore des régularités importantes (sous-apprentissage), mais un modèle trop complexe risque de capturer du bruit plutôt que du signal (surapprentissage). Tout le livre tourne autour de cette tension.

### Modèle vs algorithme d'apprentissage

Il est important de distinguer le **modèle** (la famille de fonctions $\{f_\theta : \theta \in \Theta\}$) de l'**algorithme d'apprentissage** (la procédure qui, étant donné des données, choisit un $\theta$ particulier). Le même modèle peut être entraîné par différents algorithmes; le même algorithme peut être appliqué à différents modèles.

Par exemple, pour un modèle linéaire:
- L'algorithme des **moindres carrés** trouve le $\theta$ qui minimise la somme des erreurs au carré
- La **descente de gradient** trouve (approximativement) le même $\theta$ par itérations successives
- L'**inférence bayésienne** produit une distribution sur les $\theta$ possibles plutôt qu'un point unique

Ces algorithmes ont des propriétés différentes (rapidité, stabilité, interprétabilité), mais ils opèrent tous sur le même espace de modèles.

## Pourquoi apprendre l'apprentissage machine à l'ère des LLMs?

Aujourd'hui, un assistant de programmation peut écrire un pipeline d'apprentissage machine complet en quelques secondes. Il peut charger des données, définir un modèle, l'entraîner, afficher des courbes d'apprentissage, et rapporter des métriques de performance. Pourquoi, alors, passer un semestre à étudier les fondements théoriques et algorithmiques de l'AM?

La réponse courte: **parce que les LLMs sont eux-mêmes des systèmes d'apprentissage machine, et ils utilisent constamment l'AM comme sous-routine**.

### Les LLMs sont construits sur ces fondements

Un grand modèle de langage comme GPT ou Claude est, fondamentalement:
- Une **architecture de réseau de neurones** (le transformer, couvert au chapitre sur les réseaux récurrents et l'attention)
- Entraînée par **descente de gradient stochastique** sur une fonction de perte (la cross-entropie sur la prédiction du prochain token)
- Avec des techniques de **régularisation** pour éviter le surapprentissage (dropout, weight decay)
- Et des considérations de **généralisation** pour que le modèle fonctionne sur des textes jamais vus

Comprendre les LLMs, c'est comprendre l'AM. Un praticien qui ne connaît pas la descente de gradient ne peut pas diagnostiquer pourquoi un fine-tuning diverge. Un praticien qui ne comprend pas le surapprentissage ne saura pas interpréter les courbes de validation. Un praticien qui ignore la notion de distribution des données ne comprendra pas les échecs de généralisation hors domaine.

### Les LLMs utilisent l'AM comme sous-routine

Quand vous demandez à un assistant de programmation d'analyser des données ou de construire un modèle prédictif, il génère du code qui appelle des algorithmes d'AM classiques: régression logistique, forêts aléatoires, réseaux de neurones, validation croisée. Le LLM ne *fait* pas l'apprentissage — il écrit du code qui le fait.

Et ce code peut être faux.

Un LLM peut écrire un pipeline où:
- Les données de test sont utilisées pour choisir les hyperparamètres (fuite d'information)
- Les caractéristiques incluent des variables qui ne seront pas disponibles en production (variables privilégiées)
- Le modèle mémorise les exemples d'entraînement plutôt que d'apprendre des régularités (surapprentissage)
- La métrique optimisée ne correspond pas à l'objectif métier réel (erreur de spécification)

Dans tous ces cas, les métriques rapportées par le pipeline seront excellentes, mais le modèle échouera en déploiement. **Savoir détecter ces erreurs requiert une compréhension des fondements.**

### La compétence centrale: auditer un pipeline

À l'ère où le code s'écrit facilement, la compétence rare n'est plus d'écrire du code — c'est de savoir si le code fait ce qu'il prétend faire. Un praticien compétent doit pouvoir:

1. **Lire** un pipeline d'AM et identifier sa structure (quel modèle? quelle perte? quel algorithme d'optimisation?)
2. **Évaluer** si le protocole expérimental est valide (les données de test sont-elles vraiment indépendantes? la métrique est-elle pertinente?)
3. **Diagnostiquer** les modes d'échec (surapprentissage? sous-apprentissage? fuite d'information?)
4. **Corriger** en utilisant les bons outils (régularisation, validation croisée, augmentation de données)

Ce livre vise à développer ces compétences. Chaque chapitre introduit des concepts qui permettent de poser des questions précises sur un système d'apprentissage: *Quelle est la classe d'hypothèses? Quel est le risque que nous minimisons? Comment savons-nous que le modèle généralisera?*

```{admonition} Compétence centrale
:class: warning

Savoir entraîner un modèle ne suffit plus. Il faut savoir **inspecter, évaluer et critiquer** les artefacts produits par du code — qu'il soit écrit par un humain ou généré par un LLM.
```

## Types d'apprentissage

Les problèmes d'apprentissage machine se divisent en plusieurs catégories selon la nature des données disponibles et l'objectif visé.

### Apprentissage supervisé

Dans l'apprentissage supervisé, nous disposons de paires d'entrées et de sorties: un ensemble $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ où chaque $x_i$ est une entrée et $y_i$ est la sortie correspondante (l'*étiquette* ou la *cible*). L'objectif est d'apprendre une fonction $f$ telle que $f(x) \approx y$ pour de nouvelles paires $(x, y)$ jamais vues.

Selon la nature de la sortie:
- **Classification**: $y \in \{0, 1, \ldots, K-1\}$ (un nombre fini de classes). Exemple: déterminer si un courriel est un spam.
- **Régression**: $y \in \mathbb{R}$ (une valeur continue). Exemple: prédire le prix d'une maison.

### Apprentissage non supervisé

Dans l'apprentissage non supervisé, nous n'avons que des entrées $\{x_1, \ldots, x_n\}$ sans étiquettes associées. L'objectif est de découvrir une structure cachée dans les données:
- **Partitionnement** (clustering): regrouper les données en clusters similaires
- **Réduction de dimensionnalité**: trouver une représentation compacte des données
- **Estimation de densité**: modéliser la distribution $p(x)$ des données

### Apprentissage par renforcement

Dans l'apprentissage par renforcement, un *agent* interagit avec un *environnement* et apprend à prendre des *actions* qui maximisent une *récompense* cumulative. Ce paradigme s'applique aux jeux (AlphaGo), à la robotique, et aux systèmes de recommandation. Ce livre ne couvre pas l'apprentissage par renforcement en détail.

## Prérequis et ressources

Ce livre suppose une familiarité avec:
- **Algèbre linéaire**: vecteurs, matrices, produits, valeurs propres
- **Probabilités**: distributions, espérance, variance, théorème de Bayes
- **Calcul différentiel**: dérivées, gradients, règle de la chaîne
- **Programmation**: Python, NumPy, matplotlib

Les annexes fournissent des révisions de ce matériel.

### Ouvrages de référence

- Murphy, K. *Probabilistic Machine Learning: An Introduction* (2022)
- Hastie, Tibshirani, Friedman. *The Elements of Statistical Learning* (2009)
- Bishop, C. *Pattern Recognition and Machine Learning* (2006)
- Goodfellow, Bengio, Courville. *Deep Learning* (2016)

## Notation

| Symbole | Signification |
|---------|---------------|
| $x, \boldsymbol{x}$ | Scalaire, vecteur |
| $\boldsymbol{X}$ | Matrice |
| $\theta, \boldsymbol{\theta}$ | Paramètres du modèle |
| $\mathcal{D} = \{(x_i, y_i)\}$ | Ensemble de données |
| $\mathcal{H}$ | Classe d'hypothèses |
| $\ell(y, \hat{y})$ | Fonction de perte |
| $\mathcal{R}(f)$ | Risque (vrai) |
| $\hat{\mathcal{R}}(f)$ | Risque empirique |
| $\mathbb{E}[\cdot]$ | Espérance |
| $\mathbb{P}(\cdot)$ | Probabilité |
| $\mathbb{1}_A$ | Indicatrice de l'événement $A$ |
