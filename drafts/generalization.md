# Généralisation

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Expliquer le phénomène de surapprentissage et de sous-apprentissage
- Relier la régularisation à l'estimation MAP
- Dériver la régression ridge comme cas particulier de régularisation L2
- Implémenter la validation croisée pour le choix d'hyperparamètres
- Énoncer et prouver une borne de généralisation PAC
- Appliquer les inégalités de Hoeffding et de Boole
```

## Le problème de la généralisation

L'objectif de l'apprentissage n'est pas de minimiser l'erreur sur les données d'entraînement, mais de faire de bonnes prédictions sur de nouvelles données. Cette capacité à bien performer au-delà des exemples observés est la **généralisation**.

Rappelons que le risque est défini comme l'espérance de la perte sur la distribution des données:

$$
\mathcal{R}(\theta) = \mathbb{E}_{(X,Y) \sim p}\left[\ell(Y, f(X; \theta))\right]
$$

Le risque empirique, calculé sur un ensemble de données $\mathcal{D}$, est:

$$
\hat{\mathcal{R}}(\theta; \mathcal{D}) = \frac{1}{|\mathcal{D}|} \sum_{i=1}^{|\mathcal{D}|} \ell(y_i, f(x_i; \theta))
$$

L'**écart de généralisation** est la différence entre ces deux quantités:

$$
\mathcal{R}(\theta) - \hat{\mathcal{R}}(\theta; \mathcal{D}_{\text{train}})
$$

Un modèle qui minimise parfaitement le risque empirique peut avoir un écart de généralisation important. L'erreur d'entraînement est alors un estimateur trop **optimiste** du risque de la population.

### Surapprentissage et sous-apprentissage

Le **surapprentissage** (en anglais *overfitting*) se produit lorsque le modèle mémorise les particularités des données d'entraînement plutôt que d'apprendre les régularités sous-jacentes. L'erreur d'entraînement est faible, mais l'erreur sur de nouvelles données est élevée.

Le **sous-apprentissage** (en anglais *underfitting*) est le problème inverse: le modèle est trop simple pour capturer la structure des données. Les erreurs d'entraînement et de test sont toutes deux élevées.

Considérons la régression polynomiale comme exemple. Soit $f(x; w) = \sum_{d=0}^D w_d x^d$ un polynôme de degré $D$. Si $D = 1$, le modèle ne peut représenter que des droites, ce qui est insuffisant si la relation entre $x$ et $y$ est non linéaire. À l'inverse, si $D = N - 1$ où $N$ est le nombre d'exemples, nous avons autant de paramètres que d'observations et pouvons interpoler parfaitement les données. L'erreur d'entraînement atteint zéro, mais le polynôme oscille violemment entre les points et généralise mal.

L'erreur de test suit typiquement une **courbe en U** en fonction de la complexité du modèle. Elle est élevée pour les modèles trop simples (sous-apprentissage), diminue jusqu'à un minimum optimal, puis augmente à nouveau pour les modèles trop complexes (surapprentissage).

## Régularisation

La régularisation combat le surapprentissage en pénalisant la complexité du modèle. Plutôt que de minimiser uniquement le risque empirique, nous minimisons une combinaison du risque empirique et d'un terme de pénalité:

$$
\hat{\mathcal{R}}_{\lambda}(\theta) = \frac{1}{N} \sum_{n=1}^{N} \ell(y_n, f(x_n; \theta)) + \lambda C(\theta)
$$

Le terme $C(\theta)$ mesure la complexité du modèle, et le coefficient de régularisation $\lambda \geq 0$ contrôle l'importance relative de cette pénalité. Cette approche découle du **principe de minimisation du risque structurel** (en anglais *structural risk minimization*).

### Régularisation et estimation MAP

La régularisation a une interprétation probabiliste naturelle. Si nous choisissons $C(\theta) = -\log p(\theta)$ où $p(\theta)$ est une distribution a priori, le risque régularisé devient:

$$
\hat{\mathcal{R}}_{\lambda}(\theta) = -\frac{1}{N} \sum_{n=1}^{N} \log p(y_n | x_n, \theta) - \lambda \log p(\theta)
$$

Pour $\lambda = 1$ et en ignorant le facteur $1/N$, minimiser cette expression équivaut à maximiser le log a posteriori:

$$
\log p(\theta | \mathcal{D}) = \log p(\mathcal{D} | \theta) + \log p(\theta) - \text{constante}
$$

Le minimiseur du risque empirique régularisé coïncide donc avec l'**estimateur du maximum a posteriori** (MAP):

$$
\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} \log p(\theta | \mathcal{D}) = \arg\max_{\theta} \left[\log p(\mathcal{D} | \theta) + \log p(\theta)\right]
$$

L'a priori $p(\theta)$ encode nos croyances sur les valeurs plausibles des paramètres avant d'observer les données. La régularisation incorpore cette connaissance préalable dans l'apprentissage.

### Statistique bayésienne

L'estimation MAP provient de la statistique bayésienne, qui s'intéresse à caractériser l'incertitude sur les paramètres. Le théorème de Bayes relie l'a posteriori à la vraisemblance et l'a priori:

$$
p(\theta | \mathcal{D}) = \frac{p(\theta) p(\mathcal{D} | \theta)}{p(\mathcal{D})} = \frac{p(\theta) p(\mathcal{D} | \theta)}{\int p(\theta') p(\mathcal{D} | \theta') d\theta'}
$$

où:
- $p(\theta | \mathcal{D})$ est la **distribution a posteriori**
- $p(\theta)$ est la **distribution a priori**
- $p(\mathcal{D} | \theta)$ est la **vraisemblance**
- $p(\mathcal{D})$ est la **vraisemblance marginale** (ou évidence)

L'approche bayésienne complète utilise la **distribution prédictive a posteriori**:

$$
p(y | x, \mathcal{D}) = \int p(y | x, \theta) p(\theta | \mathcal{D}) d\theta
$$

Cette distribution moyenne les prédictions sur tous les modèles possibles, pondérés par leur probabilité a posteriori. L'estimation MAP est une approximation qui utilise uniquement le mode de l'a posteriori.

## Régularisation L2 et régression ridge

Un choix courant d'a priori est une gaussienne centrée à zéro:

$$
p(w) = \mathcal{N}(w | 0, \tau^2 I)
$$

Cet a priori favorise les paramètres de petite magnitude. Le log a priori est proportionnel à la norme L2 au carré:

$$
\log p(w) \propto -\frac{1}{2\tau^2} \|w\|_2^2
$$

L'estimateur MAP devient:

$$
\hat{w}_{\text{MAP}} = \arg\min_{w} \text{NLL}(w) + \lambda \|w\|_2^2
$$

où $\|w\|_2^2 = \sum_{d=1}^D w_d^2$. Cette pénalisation est appelée **régularisation L2** ou **dégradation des pondérations** (en anglais *weight decay*).

### Régression ridge

Dans le contexte de la régression linéaire, la régularisation L2 donne la **régression ridge** (ou régression par crêtes). L'objectif est:

$$
\hat{w}_{\text{ridge}} = \arg\min_{w} \|Xw - y\|_2^2 + \lambda \|w\|_2^2
$$

La solution analytique s'obtient en annulant le gradient:

$$
\nabla_w \left[\|Xw - y\|_2^2 + \lambda \|w\|_2^2\right] = 2X^\top(Xw - y) + 2\lambda w = 0
$$

Ce qui donne:

$$
\hat{w}_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y
$$

Comparée à la solution des moindres carrés ordinaires $(X^\top X)^{-1} X^\top y$, la régression ridge ajoute $\lambda I$ à la matrice $X^\top X$. Cet ajout a plusieurs effets bénéfiques:

1. La matrice devient inversible même si $X^\top X$ est singulière
2. Les valeurs propres sont augmentées de $\lambda$, améliorant la stabilité numérique
3. Les coefficients sont "rétrécis" vers zéro, réduisant la variance

### Exemple: régularisation L2 pour Bernoulli

Considérons l'estimation du paramètre d'une loi de Bernoulli. Avec seulement trois observations "face" ($N_1 = 3, N_0 = 0$), l'EMV donne:

$$
\hat{\theta}_{\text{MLE}} = \frac{N_1}{N_0 + N_1} = \frac{3}{3 + 0} = 1
$$

Ce modèle prédit que l'événement "pile" est impossible, ce qui est peu plausible intuitivement.

Utilisons plutôt l'estimateur MAP avec un a priori Beta:

$$
p(\theta) = \text{Beta}(\theta | a, b)
$$

Pour $a, b > 1$, cet a priori favorise des valeurs de $\theta$ proches de $a/(a+b)$. Le log a posteriori est:

$$
\ell(\theta) = \left[N_1 \log \theta + N_0 \log(1-\theta)\right] + \left[(a-1) \log \theta + (b-1) \log(1-\theta)\right]
$$

En annulant la dérivée, nous obtenons:

$$
\hat{\theta}_{\text{MAP}} = \frac{N_1 + a - 1}{N_1 + N_0 + a + b - 2}
$$

Le choix $a = b = 2$ favorise les valeurs proches de $0.5$ et donne:

$$
\hat{\theta}_{\text{MAP}} = \frac{N_1 + 1}{N_1 + N_0 + 2} = \frac{3 + 1}{3 + 0 + 2} = 0.8
$$

Cette technique est le **lissage de Laplace** (ou *add-one smoothing*), particulièrement utile lorsque certains événements n'ont jamais été observés.

## Choix des hyperparamètres

Le coefficient de régularisation $\lambda$ est un **hyperparamètre**: il n'est pas appris par minimisation du risque empirique mais doit être choisi autrement. Une grande valeur de $\lambda$ met l'emphase sur la proximité à l'a priori, ce qui peut causer du sous-apprentissage. Une petite valeur favorise la minimisation du risque empirique, risquant le surapprentissage.

### Ensemble de validation

La méthode standard consiste à diviser les données en trois parties:

1. **Ensemble d'entraînement** $\mathcal{D}_{\text{train}}$: utilisé pour apprendre les paramètres
2. **Ensemble de validation** $\mathcal{D}_{\text{valid}}$: utilisé pour choisir les hyperparamètres
3. **Ensemble de test** $\mathcal{D}_{\text{test}}$: utilisé une seule fois pour l'évaluation finale

Typiquement, nous gardons 60-80% des données pour l'entraînement, 10-20% pour la validation, et 10-20% pour le test.

Le **risque de validation** est:

$$
\hat{\mathcal{R}}^{\text{val}}_{\lambda} = \hat{\mathcal{R}}_0\left(\hat{\theta}_{\lambda}(\mathcal{D}_{\text{train}}), \mathcal{D}_{\text{valid}}\right)
$$

où $\hat{\theta}_{\lambda}(\mathcal{D}_{\text{train}})$ sont les paramètres appris sur l'ensemble d'entraînement avec régularisation $\lambda$, et l'évaluation se fait sans régularisation ($\lambda = 0$) sur l'ensemble de validation.

### Recherche par quadrillage

La **recherche par quadrillage** (en anglais *grid search*) évalue systématiquement un ensemble discret d'hyperparamètres. Les valeurs sont souvent choisies sur une échelle logarithmique, par exemple $\lambda \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1, 10\}$.

```{prf:algorithm} Recherche par quadrillage
:label: grid-search

**Entrée**: Ensemble d'hyperparamètres candidats $\mathcal{S} = \{\lambda_0, \ldots, \lambda_K\}$

**Sortie**: Meilleur hyperparamètre $\lambda^*$

1. Pour chaque $\lambda \in \mathcal{S}$:
   1. Entraîner le modèle: $\hat{\theta}_{\lambda} = \arg\min_{\theta} \hat{\mathcal{R}}_{\lambda}(\theta, \mathcal{D}_{\text{train}})$
   2. Évaluer sur la validation: $v_{\lambda} = \hat{\mathcal{R}}_0(\hat{\theta}_{\lambda}, \mathcal{D}_{\text{valid}})$
2. Retourner $\lambda^* = \arg\min_{\lambda \in \mathcal{S}} v_{\lambda}$
```

Une fois $\lambda^*$ choisi, nous réentraînons le modèle sur l'union $\mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{valid}}$ pour obtenir les paramètres finaux.

### Validation croisée

Lorsque les données sont limitées, réserver un ensemble de validation réduit significativement les exemples disponibles pour l'entraînement. La **validation croisée** (en anglais *cross-validation*) offre une alternative.

Dans la validation croisée à $K$ blocs, nous partitionnons les données en $K$ sous-ensembles de taille approximativement égale. Pour chaque bloc $k$:

1. Nous entraînons le modèle sur tous les blocs sauf le $k$-ième
2. Nous évaluons sur le bloc $k$

Le **risque de validation croisée** est la moyenne des erreurs sur les $K$ blocs:

$$
\hat{\mathcal{R}}^{\text{cv}}_{\lambda} = \frac{1}{K} \sum_{k=1}^{K} \hat{\mathcal{R}}_0\left(\hat{\theta}_{\lambda}(\mathcal{D}_{-k}), \mathcal{D}_k\right)
$$

où $\mathcal{D}_{-k}$ désigne toutes les données sauf le bloc $k$.

Le cas particulier $K = N$ (un bloc par exemple) est la **validation croisée tout-sauf-un** (en anglais *leave-one-out cross-validation*, LOOCV). Cette méthode utilise le maximum de données pour l'entraînement mais est coûteuse en calcul.

```{prf:algorithm} Validation croisée à K blocs
:label: k-fold-cv

**Entrée**: Données $\mathcal{D}$, nombre de blocs $K$

**Sortie**: Estimation du risque

1. Partitionner $\mathcal{D}$ en $K$ blocs: $\mathcal{D}_1, \ldots, \mathcal{D}_K$
2. Pour chaque $k = 1, \ldots, K$:
   1. Former l'ensemble d'entraînement $\mathcal{D}_{-k} = \mathcal{D} \setminus \mathcal{D}_k$
   2. Entraîner le modèle: $\hat{\theta}_k = \arg\min_{\theta} \hat{\mathcal{R}}(\theta, \mathcal{D}_{-k})$
   3. Évaluer sur le bloc $k$: $e_k = \hat{\mathcal{R}}(\hat{\theta}_k, \mathcal{D}_k)$
3. Retourner $\bar{e} = \frac{1}{K} \sum_{k=1}^{K} e_k$
```

Les valeurs $K = 5$ et $K = 10$ sont couramment utilisées. Elles offrent un bon compromis entre le biais de l'estimateur (qui diminue avec $K$) et sa variance (qui augmente avec $K$).

## Effet de la taille de l'ensemble d'entraînement

Pour un modèle de complexité fixe, l'augmentation de la taille de l'ensemble d'entraînement réduit les chances de surapprentissage. Avec plus de données, le risque empirique devient un meilleur estimateur du vrai risque.

Intuitivement, si nous avons peu de données, un modèle flexible peut facilement les mémoriser. Avec beaucoup de données, la seule façon d'obtenir une faible erreur d'entraînement est de capturer les vraies régularités.

La courbe d'apprentissage, qui trace l'erreur en fonction de la taille de l'ensemble d'entraînement, est un outil diagnostique utile. Si l'erreur d'entraînement et l'erreur de validation sont toutes deux élevées, le modèle est en sous-apprentissage. Si l'écart entre les deux erreurs est grand, le modèle est en surapprentissage et bénéficierait de plus de données ou de plus de régularisation.

## Théorie de l'apprentissage statistique

La théorie de l'apprentissage statistique fournit des garanties formelles sur la généralisation. Ces résultats quantifient quand et pourquoi la minimisation du risque empirique mène à un faible risque de population.

### Apprentissage PAC

Un problème est dit **PAC-apprenable** (pour *Probably Approximately Correct*) s'il existe un algorithme qui, avec haute probabilité, trouve une hypothèse approximativement correcte. Formellement, pour tout $\epsilon > 0$ (précision) et $\delta > 0$ (confiance), l'algorithme doit produire une hypothèse $h$ telle que:

$$
\mathbb{P}\left[\mathcal{R}(h) \leq \min_{h' \in \mathcal{H}} \mathcal{R}(h') + \epsilon\right] \geq 1 - \delta
$$

en utilisant un nombre polynomial d'exemples en $1/\epsilon$ et $1/\delta$.

### Borne de généralisation

Le théorème suivant donne une borne sur la probabilité que l'écart de généralisation dépasse un seuil donné.

```{prf:theorem} Borne de généralisation pour une classe finie
:label: gen-bound

Soit $\mathcal{H}$ une classe d'hypothèses de dimension finie $|\mathcal{H}|$. Pour toute distribution $p^*$ sur les données et tout ensemble $\mathcal{D}$ de taille $N$ échantillonné i.i.d. de $p^*$, la probabilité que l'écart de généralisation d'un classifieur binaire dépasse $\epsilon$ est bornée par:

$$
\mathbb{P}\left(\max_{h \in \mathcal{H}} |\mathcal{R}(h) - \hat{\mathcal{R}}(h, \mathcal{D})| > \epsilon\right) \leq 2 |\mathcal{H}| e^{-2N\epsilon^2}
$$
```

Cette borne nous dit que:
- L'écart de généralisation **augmente** avec la taille de la classe d'hypothèses $|\mathcal{H}|$
- L'écart de généralisation **diminue** avec le nombre d'exemples $N$

### Preuve

La preuve utilise deux inégalités fondamentales.

**Inégalité de Hoeffding.** Si $E_1, \ldots, E_N \sim \text{Ber}(\theta)$ sont des variables aléatoires i.i.d., alors pour tout $\epsilon > 0$:

$$
\mathbb{P}(|\bar{E} - \theta| > \epsilon) \leq 2e^{-2N\epsilon^2}
$$

où $\bar{E} = \frac{1}{N} \sum_{i=1}^{N} E_i$ est la moyenne empirique.

Cette inégalité borne la probabilité que la moyenne d'un échantillon s'éloigne de la vraie moyenne d'une quantité $\epsilon$.

**Inégalité de Boole (union bound).** Pour des événements $A_1, \ldots, A_d$:

$$
\mathbb{P}\left(\bigcup_{i=1}^{d} A_i\right) \leq \sum_{i=1}^{d} \mathbb{P}(A_i)
$$

En combinant ces inégalités:

$$
\begin{aligned}
\mathbb{P}\left(\max_{h \in \mathcal{H}} |\mathcal{R}(h) - \hat{\mathcal{R}}(h, \mathcal{D})| > \epsilon\right) 
&= \mathbb{P}\left(\bigcup_{h \in \mathcal{H}} \{|\mathcal{R}(h) - \hat{\mathcal{R}}(h, \mathcal{D})| > \epsilon\}\right) \\
&\leq \sum_{h \in \mathcal{H}} \mathbb{P}(|\mathcal{R}(h) - \hat{\mathcal{R}}(h, \mathcal{D})| > \epsilon) \\
&\leq \sum_{h \in \mathcal{H}} 2e^{-2N\epsilon^2} \\
&= 2|\mathcal{H}| e^{-2N\epsilon^2}
\end{aligned}
$$

La première inégalité utilise l'union bound. La deuxième applique Hoeffding à chaque hypothèse individuellement, en notant que le risque empirique est la moyenne de pertes i.i.d.

### Interprétation

Cette borne peut être réarrangée pour donner le nombre d'exemples nécessaires. Si nous voulons que l'écart de généralisation soit au plus $\epsilon$ avec probabilité au moins $1 - \delta$, nous posons:

$$
2|\mathcal{H}| e^{-2N\epsilon^2} \leq \delta
$$

En résolvant pour $N$:

$$
N \geq \frac{1}{2\epsilon^2} \ln\left(\frac{2|\mathcal{H}|}{\delta}\right)
$$

La complexité de l'échantillon croît logarithmiquement avec la taille de la classe d'hypothèses, mais polynomialement avec $1/\epsilon$.

## Résumé

Ce chapitre a traité de la généralisation, le concept central de l'apprentissage machine:

- Le **surapprentissage** et le **sous-apprentissage** sont les deux écueils à éviter. Le premier survient avec des modèles trop complexes, le second avec des modèles trop simples.

- La **régularisation** pénalise la complexité du modèle. Elle a une interprétation probabiliste comme estimation **MAP** avec un a priori sur les paramètres.

- La **régularisation L2** (dégradation des pondérations) correspond à un a priori gaussien. En régression linéaire, elle donne la **régression ridge**.

- La **validation croisée** permet de choisir les hyperparamètres en réutilisant les données de manière efficace.

- La **théorie de l'apprentissage statistique** fournit des bornes sur l'écart de généralisation. Ces bornes dépendent de la taille de la classe d'hypothèses et du nombre d'exemples.

Le prochain chapitre étudie les méthodes non paramétriques, en commençant par les k plus proches voisins.
