# Introduction

L'apprentissage machine étudie les algorithmes qui s'améliorent par l'expérience. Plutôt que de suivre des instructions explicites programmées à l'avance, un algorithme d'apprentissage utilise les données pour découvrir des régularités, faire des prédictions et informer des décisions. Ce livre introduit les fondements mathématiques et les algorithmes pratiques qui rendent un tel apprentissage possible.

## Perspective historique

L'idée d'apprendre à partir des données n'est pas récente. En 1693, Edmund Halley tentait de découvrir des régularités dans les naissances et mortalités de la ville de Breslau. Son objectif était de dériver une formule pour prédire les montants espérés en taxes qui seraient collectés pour chaque individu. Cette table de mortalité représente l'une des premières tentatives systématiques d'utiliser des données pour faire des prédictions.

La cybernétique, discipline introduite par Norbert Wiener en 1947, a unifié les théories de l'information et de la commande optimale. Cette approche ambitieuse visait à expliquer le cerveau à l'aide des ordinateurs, à comprendre les organisations sociales et les interactions entre humains et machines. Ces travaux ont posé les bases conceptuelles de l'apprentissage automatique moderne.

En 1957, Frank Rosenblatt a introduit le perceptron pour classifier des images. Utilisant les idées de McCulloch et Pitts ainsi que de Donald Hebb sur la formalisation du calcul neuronal, le perceptron représente l'ancêtre des réseaux de neurones modernes. Cette machine pouvait apprendre à distinguer différentes formes à partir d'exemples, démontrant pour la première fois qu'une machine pouvait véritablement apprendre.

## Types d'apprentissage

Les problèmes d'apprentissage machine se divisent en plusieurs catégories selon la nature des données disponibles et l'objectif visé.

### Apprentissage supervisé

Dans l'apprentissage supervisé, nous disposons de paires d'entrées et de sorties. L'objectif est d'identifier automatiquement une relation entre ces paires pour pouvoir prédire la sortie associée à de nouvelles entrées. Nous supposons qu'il existe une vraie fonction $f: \mathcal{X} \to \mathcal{Y}$ pouvant expliquer les données, et nous cherchons à l'approximer.

Soit $d$ la dimension du vecteur d'entrée. Selon la nature de la sortie, nous distinguons:

- La **classification binaire**: $\hat{f}: \mathbb{R}^d \to \{0, 1\}$
- La **classification multiclasse**: $\hat{f}: \mathbb{R}^d \to \{0, \ldots, m\}$ 
- La **régression**: $\hat{f}: \mathbb{R}^d \to \mathbb{R}^p$

Les problèmes de classification ont trait à des questions de nature qualitative, alors que la régression traite de questions quantitatives. Par exemple, déterminer si un courriel est un pourriel relève de la classification, tandis que prédire le prix d'une maison relève de la régression.

### Apprentissage non supervisé

Dans l'apprentissage non supervisé, nous n'avons accès qu'aux données d'entrée, sans étiquettes associées. L'objectif est de découvrir la structure cachée dans les données. L'estimation de densité, qui consiste à estimer la fonction de densité d'une variable aléatoire à partir d'observations, constitue un exemple classique. Le partitionnement (clustering), qui vise à regrouper des exemples similaires, en est un autre.

### Apprentissage auto-supervisé

L'apprentissage auto-supervisé génère automatiquement des cibles utiles à partir des données elles-mêmes. Par exemple, on peut masquer une partie d'une image et entraîner un modèle à la reconstruire, ou prédire le mot suivant dans une phrase. Cette approche permet d'exploiter de grandes quantités de données non étiquetées.

### Apprentissage par renforcement

L'apprentissage par renforcement traite des problèmes d'actions plutôt que de prédictions. Un agent interagit avec un environnement et apprend à maximiser une récompense cumulative. Ce paradigme s'applique aux jeux, à la robotique et aux systèmes de recommandation.

## Aperçu du cours

Ce livre couvre les concepts et méthodes fondamentaux de l'apprentissage machine. Nous commençons par la formalisation du problème d'apprentissage: le risque, le risque empirique, et ce que signifie pour un algorithme de généraliser. Ce cadre théorique guide notre étude des algorithmes.

Nous étudions d'abord les méthodes non paramétriques comme les k plus proches voisins et les méthodes à noyau. Ces approches simples illustrent les concepts fondamentaux sans imposer de forme paramétrique forte aux données.

Les modèles linéaires forment ensuite le coeur du livre: régression linéaire, régression logistique, et machines à vecteurs de support. Ces méthodes, bien que conceptuellement simples, sont puissantes et constituent la base des approches plus complexes.

Les réseaux de neurones étendent les modèles linéaires en composant des transformations simples. Nous couvrons les perceptrons multicouches, les réseaux convolutifs pour les images, et les réseaux récurrents pour les séquences.

Les méthodes d'ensemble combinent plusieurs modèles: le bagging réduit la variance, le boosting réduit le biais. Les forêts aléatoires et le gradient boosting comptent parmi les algorithmes les plus performants en pratique.

L'apprentissage non supervisé occupe la dernière partie: l'analyse en composantes principales pour la réduction de dimensionnalité, et les mélanges gaussiens pour le partitionnement. Nous concluons par une introduction aux modèles graphiques probabilistes.

## Prérequis

Ce cours suppose une familiarité avec les domaines suivants.

**Algèbre linéaire.** Vecteurs, matrices, produits matriciels, valeurs propres et vecteurs propres, décompositions matricielles. La capacité à manipuler des expressions vectorielles et matricielles est essentielle.

**Probabilités et statistiques.** Variables aléatoires, distributions de probabilité, espérance, variance, covariance. Les distributions de Bernoulli, catégorique et gaussienne apparaissent fréquemment.

**Calcul différentiel.** Dérivées partielles, gradients, règle de la chaîne. L'optimisation par descente de gradient est omniprésente en apprentissage machine.

**Programmation.** Capacité à implémenter des algorithmes en Python, avec les bibliothèques NumPy et Matplotlib. Les exemples de code utilisent également scikit-learn.

Les annexes fournissent des révisions brèves du matériel prérequis pour référence.

## Comment utiliser ce livre

Chaque chapitre commence par une liste d'objectifs d'apprentissage formulés comme des capacités: "À la fin de ce chapitre, vous serez en mesure de..." Ces objectifs guident la lecture et correspondent à ce qui est évalué dans les exercices.

Les concepts sont introduits par des exemples concrets avant les définitions formelles. Un exemple simple en deux dimensions précède souvent la formulation générale. Après avoir présenté la théorie, nous revenons à l'exemple initial pour montrer comment il s'inscrit dans le cadre abstrait.

Les exemples travaillés montrent les étapes intermédiaires des calculs. Nous indiquons explicitement quelles hypothèses ou lemmes sont utilisés, et pourquoi chaque étape est effectuée.

Les exercices varient en difficulté: rappel de définitions, applications directes, preuves conceptuelles, et implémentations. Les exercices de codage demandent d'implémenter les algorithmes à partir de zéro avant d'utiliser les fonctions de bibliothèque.

## Ressources complémentaires

Les ouvrages suivants complètent ce livre:

- *Probabilistic Machine Learning: An Introduction* par Kevin P. Murphy, disponible gratuitement en ligne
- *The Elements of Statistical Learning* par Hastie, Tibshirani et Friedman
- *Pattern Recognition and Machine Learning* par Christopher M. Bishop
- *Deep Learning* par Goodfellow, Bengio et Courville

## Notation

Nous adoptons les conventions suivantes tout au long du livre:

| Symbole | Signification |
|---------|---------------|
| $\mathcal{D}$ | Ensemble de données |
| $\boldsymbol{x}$ | Vecteur d'entrée |
| $y$ | Cible ou étiquette |
| $\boldsymbol{\theta}$ | Paramètres du modèle |
| $\mathcal{R}$ | Risque (vrai risque) |
| $\hat{\mathcal{R}}$ | Risque empirique |
| $\ell$ | Fonction de perte |
| $\mathcal{H}$ | Classe d'hypothèses |
| $\mathbb{E}[\cdot]$ | Espérance |
| $\mathbb{P}(\cdot)$ | Probabilité |
| $\mathbb{1}_A$ | Fonction indicatrice de l'ensemble $A$ |

Les vecteurs sont notés en gras minuscule ($\boldsymbol{x}$), les matrices en gras majuscule ($\boldsymbol{X}$), et les scalaires en italique ($x$).
