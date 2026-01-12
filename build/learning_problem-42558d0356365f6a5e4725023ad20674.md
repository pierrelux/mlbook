# Le problème d'apprentissage

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Définir formellement le problème d'apprentissage supervisé
- Distinguer les tâches de classification et de régression
- Définir le risque et le risque empirique
- Expliquer le principe de minimisation du risque empirique
- Dériver l'estimateur du maximum de vraisemblance (EMV)
- Relier le EMV à la divergence de Kullback-Leibler
- Décrire les fonctions de perte courantes et leurs propriétés
```

## Formalisation de l'apprentissage supervisé

L'apprentissage supervisé consiste à identifier automatiquement une relation entre des paires d'entrées et de sorties. Nous disposons d'un jeu de données $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ composé de $N$ paires, où chaque $x_i \in \mathcal{X}$ est une entrée et $y_i \in \mathcal{Y}$ est la sortie correspondante.

L'objectif est de trouver une fonction $\hat{f}: \mathcal{X} \to \mathcal{Y}$ qui approxime bien les données. Nous supposons implicitement qu'il existe une vraie fonction $f: \mathcal{X} \to \mathcal{Y}$ pouvant expliquer les données, et nous cherchons à l'estimer. La capacité à faire de bonnes prédictions au-delà des données déjà observées est ce que nous appelons la **généralisation**.

### Données tabulaires et caractéristiques

Dans de nombreuses applications, les entrées prennent la forme de vecteurs de caractéristiques. Chaque exemple $x_i \in \mathbb{R}^d$ est un vecteur de dimension $d$, où chaque composante représente une **caractéristique** (ou **trait**, en anglais *feature*) choisie pour sa valeur prédictive.

Considérons le jeu de données Iris, un exemple classique en apprentissage machine. Chaque fleur est décrite par quatre caractéristiques: la longueur et la largeur du sépale, ainsi que la longueur et la largeur du pétale. L'objectif est de prédire l'espèce parmi trois catégories: Setosa, Versicolor et Virginica.

| Longueur sépale (cm) | Largeur sépale (cm) | Longueur pétale (cm) | Largeur pétale (cm) |
|---------------------|---------------------|---------------------|---------------------|
| 5.1 | 3.5 | 1.4 | 0.2 |
| 4.9 | 3.0 | 1.4 | 0.2 |
| 4.7 | 3.2 | 1.3 | 0.2 |

Parfois, nous transformons les caractéristiques d'origine dans un autre espace, plus riche et expressif. Cet espace transformé est appelé **espace de redescription** (en anglais *feature space*). Par exemple, nous pourrions ajouter des termes quadratiques ou des interactions entre variables.

### Classification et régression

Selon la nature de l'espace de sortie $\mathcal{Y}$, nous distinguons deux types de problèmes.

En **classification**, la sortie est une catégorie parmi un ensemble fini. Pour la classification binaire, $\hat{f}: \mathbb{R}^d \to \{0, 1\}$. Pour la classification multiclasse avec $m$ catégories, $\hat{f}: \mathbb{R}^d \to \{0, \ldots, m-1\}$. Les problèmes de classification ont trait à des questions de nature qualitative: déterminer si un courriel est un pourriel, diagnostiquer une maladie, ou reconnaître un chiffre manuscrit.

En **régression**, la sortie est une valeur continue. Pour une sortie scalaire, $\hat{f}: \mathbb{R}^d \to \mathbb{R}$. Pour une sortie vectorielle, $\hat{f}: \mathbb{R}^d \to \mathbb{R}^p$. Les problèmes de régression concernent des questions quantitatives: prédire le prix d'une maison, estimer la température demain, ou prévoir la demande d'électricité.

## Risque et risque empirique

La qualité d'un modèle se mesure par sa capacité à faire de bonnes prédictions. Pour quantifier cette notion, nous introduisons les concepts de fonction de perte et de risque.

### Fonction de perte

Une **fonction de perte** $\ell: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_+$ mesure l'erreur commise lorsque nous prédisons $\hat{y}$ alors que la vraie valeur est $y$. Une perte de zéro indique une prédiction parfaite; plus la perte est grande, plus l'erreur est importante.

Pour la classification, un choix naturel est la **perte 0-1**:

$$
\ell_{0-1}(y, \hat{y}) = \begin{cases}
0 & \text{si } y = \hat{y} \\
1 & \text{si } y \neq \hat{y}
\end{cases} = \mathbb{1}_{y \neq \hat{y}}
$$

Cette perte vaut 0 pour une prédiction correcte et 1 pour une erreur. Le risque empirique correspondant est le taux d'erreur de classification.

Pour la régression, nous utilisons généralement la **perte quadratique**:

$$
\ell_2(y, \hat{y}) = (y - \hat{y})^2
$$

Cette perte pénalise les grandes erreurs de manière quadratique.

### Le risque

Le **risque** (ou **risque de la population**) d'un modèle $f$ est l'espérance de la fonction de perte sur la distribution intrinsèque des données:

$$
\mathcal{R}(f) = \mathbb{E}_{(X,Y) \sim p}\left[\ell(Y, f(X))\right] = \int \ell(y, f(x)) \, p(x, y) \, dx \, dy
$$

Le risque mesure la performance moyenne du modèle sur toutes les données possibles, pas seulement celles que nous avons observées. Un modèle avec un faible risque fait de bonnes prédictions en général.

Le problème fondamental est que nous ne connaissons pas la distribution $p(x, y)$. Nous n'y avons accès qu'indirectement, via un échantillon fini $\mathcal{D}$.

### Le risque empirique

Le **risque empirique** est une approximation du vrai risque calculée sur l'échantillon disponible:

$$
\hat{\mathcal{R}}(f, \mathcal{D}) = \frac{1}{|\mathcal{D}|} \sum_{i=1}^{|\mathcal{D}|} \ell(y_i, f(x_i))
$$

Le risque empirique est la moyenne des pertes sur les exemples d'entraînement. C'est une quantité que nous pouvons calculer directement.

### Principe de minimisation du risque empirique

Vapnik (1992) propose un cadre théorique pour formaliser l'apprentissage. L'idée centrale est de voir l'apprentissage comme un problème d'estimation de fonction. Nous cherchons la fonction qui minimise le risque:

$$
\min_{f \in \mathcal{F}} \mathcal{R}(f)
$$

où $\mathcal{F}$ est une famille de fonctions (notre **classe d'hypothèses**). Puisque le vrai risque est inaccessible, nous le remplaçons par le risque empirique:

$$
\min_{f \in \mathcal{F}} \hat{\mathcal{R}}(f, \mathcal{D})
$$

Ce principe est appelé **minimisation du risque empirique** (MRE, en anglais *empirical risk minimization*, ERM).

La question centrale de la théorie de l'apprentissage statistique est: quand le minimum du risque empirique s'approche-t-il du minimum du vrai risque? Formellement, si $f^\star$ minimise $\mathcal{R}$ et $\hat{f}_N$ minimise $\hat{\mathcal{R}}$ sur $N$ exemples:

- Est-ce que $\mathcal{R}(\hat{f}_N)$ converge vers $\mathcal{R}(f^\star)$ lorsque $N \to \infty$?
- À quelle vitesse cette convergence se produit-elle?

Ces questions relèvent de la **cohérence statistique** et des bornes de généralisation, que nous étudierons au chapitre suivant.

## Fonctions de perte de substitution

La perte 0-1 pose un problème pratique: elle n'est pas différentiable. Les méthodes d'optimisation par gradient, omniprésentes en apprentissage machine, requièrent des fonctions lisses. Nous utilisons donc des **fonctions de perte de substitution** (en anglais *surrogate loss functions*).

Une bonne fonction de substitution est une borne supérieure convexe de la perte originale. Elle est plus facile à optimiser tout en conservant des garanties théoriques.

### Perte logistique

Dans un contexte probabiliste de classification binaire où $y \in \{-1, +1\}$, nous modélisons la probabilité de la classe positive par une fonction sigmoïde:

$$
p(y = 1 | x) = \sigma(f(x)) = \frac{1}{1 + e^{-f(x)}}
$$

où $f(x)$ est appelé le **logit** (ou log-odds). La **perte logistique** (ou log-vraisemblance négative) est:

$$
\ell_{\text{log}}(y, f(x)) = \log(1 + e^{-y \cdot f(x)})
$$

Cette fonction est convexe et différentiable partout. Elle constitue une borne supérieure de la perte 0-1.

### Perte à charnière

La **perte à charnière** (en anglais *hinge loss*) est utilisée dans les machines à vecteurs de support:

$$
\ell_{\text{hinge}}(y, f(x)) = \max(0, 1 - y \cdot f(x))
$$

Cette fonction est convexe mais différentiable seulement par morceaux. Elle pénalise non seulement les erreurs de classification, mais aussi les prédictions correctes avec une marge insuffisante.

Les deux fonctions majorent la perte 0-1: pour tout $y$ et $f(x)$, $\ell_{0-1} \leq \ell_{\text{log}}$ et $\ell_{0-1} \leq \ell_{\text{hinge}}$. Minimiser ces substituts garantit donc un contrôle sur la perte originale.

## Maximum de vraisemblance

L'approche probabiliste de l'apprentissage modélise explicitement les incertitudes. Nous supposons que les données sont générées par une distribution paramétrique $p(y|x; \theta)$, et nous cherchons les paramètres $\theta$ qui expliquent le mieux les observations.

### Vraisemblance et log-vraisemblance

La **vraisemblance** des paramètres $\theta$ étant données les données $\mathcal{D}$ est:

$$
p(\mathcal{D}|\theta) = \prod_{i=1}^N p(y_i | x_i, \theta)
$$

Le produit découle de l'hypothèse que les exemples sont indépendants et identiquement distribués (i.i.d.). L'**estimateur du maximum de vraisemblance** (EMV, en anglais *maximum likelihood estimator*, MLE) est:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta p(\mathcal{D}|\theta)
$$

En pratique, nous maximisons le logarithme de la vraisemblance, ce qui transforme le produit en somme:

$$
\log p(\mathcal{D}|\theta) = \sum_{i=1}^N \log p(y_i | x_i, \theta)
$$

Pour l'optimisation, nous minimisons la **log-vraisemblance négative** (NLL):

$$
\text{NLL}(\theta) = -\sum_{i=1}^N \log p(y_i | x_i, \theta)
$$

### Lien avec le risque empirique

Le principe de minimisation du risque empirique et l'estimation par maximum de vraisemblance coïncident lorsque nous choisissons la perte logarithmique $\ell(y, f(x)) = -\log p(y | f(x))$. Le risque empirique devient alors:

$$
\hat{\mathcal{R}}(\theta) = \frac{1}{N} \sum_{i=1}^N -\log p(y_i | x_i, \theta) = \frac{1}{N} \text{NLL}(\theta)
$$

Le minimiseur du risque empirique avec perte logarithmique est donc l'estimateur du maximum de vraisemblance.

### Application à la régression

En régression, nous modélisons souvent la distribution conditionnelle par une gaussienne:

$$
p(y|x; \theta) = \mathcal{N}(y | f(x; \theta), \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - f(x; \theta))^2}{2\sigma^2}\right)
$$

La log-vraisemblance négative est alors:

$$
\text{NLL}(\theta) = \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - f(x_i; \theta))^2 + \text{constante}
$$

Le terme dominant est l'**erreur quadratique moyenne** (EQM, en anglais *mean squared error*, MSE):

$$
\text{MSE}(\theta) = \frac{1}{N} \sum_{i=1}^N (y_i - f(x_i; \theta))^2
$$

Minimiser la NLL sous un modèle gaussien équivaut donc à minimiser l'erreur quadratique moyenne. Cette connexion justifie l'utilisation de la perte quadratique en régression d'un point de vue probabiliste.

## Justification informationnelle du MLE

Nous pouvons justifier l'estimation par maximum de vraisemblance du point de vue de la théorie de l'information. L'EMV trouve la distribution paramétrique qui s'approche le plus de la distribution empirique des données.

### Distribution empirique

La **distribution empirique** des données est:

$$
p_{\mathcal{D}}(y) = \frac{1}{N} \sum_{i=1}^N \delta(y - y_i)
$$

où $\delta$ est la fonction delta de Dirac. Cette distribution place une masse de probabilité $1/N$ sur chaque observation.

### Divergence de Kullback-Leibler

La **divergence de Kullback-Leibler** mesure la dissimilarité entre deux distributions:

$$
D_{\text{KL}}(p \| q) = \sum_y p(y) \log \frac{p(y)}{q(y)} = -\mathbb{H}(p) - \sum_y p(y) \log q(y)
$$

Le premier terme est l'entropie négative de $p$. Le second est l'**entropie croisée** entre $p$ et $q$. La divergence KL satisfait $D_{\text{KL}}(p \| q) \geq 0$ avec égalité si et seulement si $p = q$.

### Connexion avec le MLE

En posant $p = p_{\mathcal{D}}$ et $q = p(\cdot | \theta)$, nous obtenons:

$$
D_{\text{KL}}(p_{\mathcal{D}} \| p(\cdot|\theta)) = -\mathbb{H}(p_{\mathcal{D}}) - \frac{1}{N} \sum_{i=1}^N \log p(y_i | \theta)
$$

Le premier terme est une constante indépendante de $\theta$. Minimiser la divergence KL revient donc à minimiser la log-vraisemblance négative:

$$
\arg\min_\theta D_{\text{KL}}(p_{\mathcal{D}} \| p(\cdot|\theta)) = \arg\min_\theta \text{NLL}(\theta) = \hat{\theta}_{\text{MLE}}
$$

L'EMV trouve les paramètres qui rendent le modèle aussi proche que possible de la distribution empirique au sens de la divergence KL.

## Modèles de régression linéaire

Le modèle de régression le plus simple est la **régression linéaire**, où la prédiction est une combinaison linéaire des caractéristiques:

$$
f(x; w, b) = w^\top x + b = \sum_{j=1}^d w_j x_j + b
$$

Les **paramètres** du modèle sont le vecteur de poids $w \in \mathbb{R}^d$ et le biais $b \in \mathbb{R}$. Nous regroupons souvent tous les paramètres sous le symbole $\theta = (w, b)$.

### Régression polynomiale

Pour capturer des relations non linéaires, nous pouvons d'abord transformer les entrées. En **régression polynomiale**, nous appliquons une fonction de redescription $\phi: \mathbb{R} \to \mathbb{R}^{k+1}$:

$$
\phi(x) = [1, x, x^2, \ldots, x^k]
$$

La prédiction devient $f(x; w) = w^\top \phi(x)$, ce qui reste linéaire dans les paramètres mais polynomial dans l'entrée originale.

Le degré $k$ du polynôme contrôle la **capacité** du modèle. Avec $k = N - 1$, nous avons autant de paramètres que d'exemples et pouvons **interpoler** parfaitement les données d'entraînement. L'EQM sur l'ensemble d'entraînement atteint zéro, mais ce modèle ne généralise généralement pas bien aux nouvelles données.

## Vocabulaire de la généralisation

La différence entre le risque et le risque empirique sur l'ensemble d'entraînement est l'**écart de généralisation**:

$$
\text{Écart de généralisation} = \mathcal{R}(\theta) - \hat{\mathcal{R}}(\theta; \mathcal{D}_{\text{train}})
$$

En pratique, nous estimons le risque par le risque empirique sur un **ensemble de test** $\mathcal{D}_{\text{test}}$ disjoint de l'ensemble d'entraînement.

Un modèle peut souffrir de deux maux opposés. Le **surapprentissage** (en anglais *overfitting*) se produit lorsque le modèle mémorise les données d'entraînement sans apprendre les régularités sous-jacentes. L'erreur d'entraînement est faible, mais l'erreur de test est élevée. À l'inverse, le **sous-apprentissage** (en anglais *underfitting*) se produit lorsque le modèle est trop simple pour capturer la structure des données. Les erreurs d'entraînement et de test sont toutes deux élevées.

Un troisième ensemble, l'**ensemble de validation**, sert à la sélection de modèles et au réglage des hyperparamètres. L'ensemble de test doit être gardé intact jusqu'à l'évaluation finale, pour fournir une estimation non biaisée de la performance.

## Biais inductifs

Il n'existe pas de modèle universel qui fonctionne optimalement pour tous les problèmes. Ce résultat, formalisé par Wolpert (1996), est le **théorème du no free lunch**.

L'apprentissage dépend des **biais inductifs** exprimés dans nos modèles, c'est-à-dire des suppositions implicites ou explicites que nous faisons. Par exemple, la régression linéaire suppose que la relation entre entrées et sorties est linéaire. Les k plus proches voisins supposent que les points proches ont des étiquettes similaires.

Ces biais inductifs sont spécifiques à une classe de problèmes. Un biais approprié pour un problème peut être inapproprié pour un autre. Le choix du modèle encode donc notre connaissance a priori sur la structure du problème.

## Résumé

Ce chapitre a établi le cadre formel de l'apprentissage supervisé:

- L'**apprentissage supervisé** consiste à trouver une fonction qui prédit les sorties à partir des entrées, en se basant sur des exemples étiquetés.

- Le **risque** mesure la performance attendue d'un modèle; le **risque empirique** est son approximation sur un échantillon.

- Le principe de **minimisation du risque empirique** guide l'apprentissage: nous choisissons le modèle qui minimise la perte moyenne sur les données d'entraînement.

- L'**estimateur du maximum de vraisemblance** coïncide avec la minimisation du risque empirique pour la perte logarithmique.

- Les **fonctions de perte de substitution** remplacent les pertes non différentiables par des approximations convexes.

- Le MLE trouve la distribution paramétrique qui minimise la **divergence KL** avec la distribution empirique.

Le chapitre suivant étudie la généralisation: comment contrôler l'écart entre le risque empirique et le vrai risque, et comment choisir la complexité appropriée pour un modèle.
