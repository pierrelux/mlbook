# Introduction

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Expliquer pourquoi la maîtrise des fondements de l'AM reste essentielle à l'ère des LLMs
- Identifier les sources classiques d'écart entre performance rapportée et performance réelle
- Distinguer les différents types d'apprentissage (supervisé, non supervisé, par renforcement)
- Relier un protocole d'évaluation à la notion de risque
```

## Pourquoi l'apprentissage machine à l'ère des LLMs?

Aujourd'hui, il est devenu facile d'obtenir du code d'apprentissage machine en quelques instructions. Un assistant de programmation peut écrire un script complet, afficher des graphiques, et même rapporter des métriques qui semblent convaincantes. Cette facilité change profondément la pratique: **la difficulté n'est plus d'écrire du code "qui tourne", mais de savoir si ce code mesure réellement ce qu'il prétend mesurer**, et si le modèle apprendra quelque chose qui sera encore vrai au moment du déploiement.

Le danger le plus courant est que des résultats impressionnants cachent un problème méthodologique. Un modèle peut sembler excellent tout en:

- **Surapprenant**: mémoriser les exemples d'entraînement plutôt qu'apprendre des régularités généralisables
- **Exploitant une fuite d'information**: utiliser indirectement des données qui ne seront pas disponibles au moment du test
- **S'appuyant sur des variables privilégiées**: des caractéristiques présentes dans le jeu de données mais non observables dans l'environnement réel

Dans tous ces cas, les métriques "auto-rapportées" sont trompeuses: elles décrivent surtout la capacité du pipeline à se convaincre lui-même.

```{admonition} Compétence centrale: inspection et évaluation
:class: warning

Savoir entraîner un modèle ne suffit plus. Il faut savoir **lire et critiquer** les artefacts produits par du code généré: jeux d'entraînement/validation/test, construction des variables, protocole d'évaluation, choix de métriques, et signes de surapprentissage ou de fuite d'information.

Ce livre introduit le langage formel — risque, risque empirique, ERM, MLE — qui permet de poser ces questions proprement.
```

C'est pourquoi ce livre insiste sur une idée simple: **apprendre l'AM, c'est apprendre à définir un problème, spécifier une procédure d'évaluation crédible, et relier un objectif d'optimisation à une notion de performance qui généralise**. Les modèles modernes rendent certaines tâches plus accessibles, mais ils rendent la rigueur *encore plus essentielle*, car il devient plus facile de produire des artefacts convaincants qui sont néanmoins faux.

## Qu'est-ce que l'apprentissage machine?

L'apprentissage machine étudie les algorithmes qui s'améliorent par l'expérience. Plutôt que de suivre des instructions explicites programmées à l'avance, un algorithme d'apprentissage utilise les données pour découvrir des régularités, faire des prédictions et informer des décisions.

### Perspective historique

L'idée d'apprendre à partir des données n'est pas récente. En 1693, Edmund Halley tentait de découvrir des régularités dans les naissances et mortalités de la ville de Breslau pour prédire les montants espérés en taxes. La cybernétique de Norbert Wiener (1947) a unifié les théories de l'information et de la commande optimale, posant les bases conceptuelles de l'apprentissage automatique moderne. En 1957, Frank Rosenblatt a introduit le perceptron, ancêtre des réseaux de neurones modernes, démontrant pour la première fois qu'une machine pouvait véritablement apprendre à partir d'exemples.

### Types d'apprentissage

Les problèmes d'apprentissage machine se divisent en plusieurs catégories selon la nature des données disponibles et l'objectif visé.

**Apprentissage supervisé.** Nous disposons de paires d'entrées et de sorties. L'objectif est d'identifier une relation pour prédire la sortie associée à de nouvelles entrées. Selon la nature de la sortie:
- **Classification**: $\hat{f}: \mathbb{R}^d \to \{0, 1, \ldots, m\}$ (questions qualitatives)
- **Régression**: $\hat{f}: \mathbb{R}^d \to \mathbb{R}^p$ (questions quantitatives)

**Apprentissage non supervisé.** Nous n'avons accès qu'aux données d'entrée, sans étiquettes. L'objectif est de découvrir la structure cachée: estimation de densité, partitionnement (clustering), réduction de dimensionnalité.

**Apprentissage auto-supervisé.** On génère automatiquement des cibles à partir des données elles-mêmes (masquer une partie d'une image et la reconstruire, prédire le mot suivant). Cette approche exploite de grandes quantités de données non étiquetées.

**Apprentissage par renforcement.** Un agent interagit avec un environnement et apprend à maximiser une récompense cumulative. Ce paradigme s'applique aux jeux, à la robotique et aux systèmes de recommandation.

## Structure du livre

Ce livre est organisé en huit parties qui progressent des fondements vers les méthodes avancées.

**Partie I: Fondements.** Nous formalisons le problème d'apprentissage: le risque, le risque empirique, et ce que signifie pour un algorithme de généraliser. Ce cadre théorique est essentiel pour comprendre *pourquoi* certains modèles échouent malgré de bonnes métriques d'entraînement.

**Partie II: Méthodes non paramétriques.** Les k plus proches voisins et les méthodes à noyau illustrent les concepts fondamentaux sans imposer de forme paramétrique forte.

**Partie III: Modèles linéaires.** Régression linéaire, régression logistique, et modèles génératifs. Ces méthodes, bien que conceptuellement simples, sont puissantes et constituent la base des approches plus complexes.

**Partie IV: Machines à vecteurs de support.** L'optimisation sous contraintes et l'astuce du noyau permettent des frontières de décision non linéaires.

**Partie V: Réseaux de neurones.** Les perceptrons multicouches, réseaux convolutifs et récurrents étendent les modèles linéaires en composant des transformations.

**Partie VI: Méthodes d'ensemble.** Arbres de décision, bagging, boosting. Les forêts aléatoires et le gradient boosting comptent parmi les algorithmes les plus performants en pratique.

**Partie VII: Apprentissage non supervisé.** Analyse en composantes principales et mélanges gaussiens.

**Partie VIII: Modèles probabilistes.** Introduction aux modèles graphiques et à l'inférence variationnelle.

## Prérequis

Ce cours suppose une familiarité avec:

- **Algèbre linéaire**: vecteurs, matrices, valeurs propres, décompositions matricielles
- **Probabilités**: variables aléatoires, distributions, espérance, variance
- **Calcul différentiel**: gradients, règle de la chaîne
- **Programmation**: Python, NumPy, Matplotlib, scikit-learn

Les annexes fournissent des révisions du matériel prérequis.

## Comment utiliser ce livre

Chaque chapitre commence par des **objectifs d'apprentissage** formulés comme des capacités. Ces objectifs guident la lecture et correspondent à ce qui est évalué.

Les concepts sont introduits par des **exemples concrets** avant les définitions formelles. Un exemple simple en deux dimensions précède souvent la formulation générale.

Les **exercices** varient en difficulté: rappel de définitions, applications directes, preuves, et implémentations. Les exercices de codage demandent d'implémenter les algorithmes à partir de zéro avant d'utiliser les fonctions de bibliothèque.

## Notation

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
| $\mathbb{1}_A$ | Fonction indicatrice |

Les vecteurs sont notés en gras minuscule ($\boldsymbol{x}$), les matrices en gras majuscule ($\boldsymbol{X}$), et les scalaires en italique ($x$).

## Ressources complémentaires

- *Probabilistic Machine Learning: An Introduction* — Kevin P. Murphy
- *The Elements of Statistical Learning* — Hastie, Tibshirani, Friedman
- *Pattern Recognition and Machine Learning* — Christopher M. Bishop
- *Deep Learning* — Goodfellow, Bengio, Courville
