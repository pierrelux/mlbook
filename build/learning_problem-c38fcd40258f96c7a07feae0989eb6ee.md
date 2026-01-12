# Le problème d'apprentissage

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Définir formellement le problème d'apprentissage supervisé
- Distinguer les tâches de classification et de régression
- Définir le risque et le risque empirique
- Expliquer le principe de minimisation du risque empirique
- Dériver l'estimateur du maximum de vraisemblance
- Relier le maximum de vraisemblance à la divergence de Kullback-Leibler
- Identifier les sources d'écart entre performance mesurée et performance réelle
```

## Apprentissage supervisé

L'apprentissage supervisé consiste à trouver une fonction qui prédit des sorties à partir d'entrées, en se basant sur des exemples. Nous disposons d'un jeu de données $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ composé de $N$ paires, où chaque $x_i \in \mathcal{X}$ est une entrée et $y_i \in \mathcal{Y}$ est la sortie correspondante. L'objectif est de trouver une fonction $f: \mathcal{X} \to \mathcal{Y}$ qui approxime bien la relation entre entrées et sorties, y compris pour des exemples que nous n'avons pas encore observés.

Dans de nombreuses applications, les entrées sont des vecteurs de caractéristiques. Chaque exemple $x_i \in \mathbb{R}^d$ est un vecteur de dimension $d$, où chaque composante représente une mesure ou un attribut. Le jeu de données Iris illustre cette structure: chaque fleur est décrite par quatre mesures (longueur et largeur du sépale, longueur et largeur du pétale), et l'objectif est de prédire l'espèce parmi trois catégories.

| Longueur sépale | Largeur sépale | Longueur pétale | Largeur pétale | Espèce |
|-----------------|----------------|-----------------|----------------|--------|
| 5.1 | 3.5 | 1.4 | 0.2 | Setosa |
| 7.0 | 3.2 | 4.7 | 1.4 | Versicolor |
| 6.3 | 3.3 | 6.0 | 2.5 | Virginica |

D'autres problèmes ont une structure similaire. Pour prédire le prix d'une maison, les entrées pourraient être la superficie, le nombre de chambres et le code postal. Pour filtrer les pourriels, les entrées pourraient être des comptages de mots ou des indicateurs de présence de certains termes. La nature de la sortie varie selon le problème.

Lorsque la sortie est une catégorie parmi un ensemble fini, nous parlons de **classification**. Pour la classification binaire, $f: \mathbb{R}^d \to \{0, 1\}$. Pour la classification multiclasse avec $m$ catégories, $f: \mathbb{R}^d \to \{0, \ldots, m-1\}$. Lorsque la sortie est une valeur continue, nous parlons de **régression**: $f: \mathbb{R}^d \to \mathbb{R}$ pour une sortie scalaire, ou $f: \mathbb{R}^d \to \mathbb{R}^p$ pour une sortie vectorielle.

## Mesurer l'erreur

Pour choisir entre deux fonctions candidates, nous avons besoin d'un critère qui quantifie la qualité des prédictions. Une **fonction de perte** $\ell: \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_+$ mesure l'écart entre une prédiction $\hat{y}$ et la vraie valeur $y$. Une perte de zéro indique une prédiction parfaite; plus la perte est grande, plus l'erreur est importante.

Pour la classification, un choix naturel est la **perte 0-1**:

$$
\ell_{0-1}(y, \hat{y}) = \mathbb{1}_{y \neq \hat{y}} = \begin{cases} 0 & \text{si } y = \hat{y} \\ 1 & \text{si } y \neq \hat{y} \end{cases}
$$

Cette perte compte simplement les erreurs: elle vaut 1 pour une mauvaise prédiction, 0 sinon.

Pour la régression, nous utilisons généralement la **perte quadratique**:

$$
\ell_2(y, \hat{y}) = (y - \hat{y})^2
$$

Cette perte pénalise les grandes erreurs de manière quadratique. Une erreur de 2 coûte quatre fois plus qu'une erreur de 1.

Le choix de la fonction de perte dépend du problème. En diagnostic médical, manquer une maladie grave (faux négatif) peut avoir des conséquences bien plus importantes que de prescrire un test supplémentaire à un patient sain (faux positif). Une perte asymétrique refléterait cette différence. En régression, si les grandes erreurs sont particulièrement problématiques, la perte quadratique est appropriée; si nous voulons être robustes aux valeurs aberrantes, la perte absolue $|y - \hat{y}|$ est préférable.

## Le risque

La perte évalue une seule prédiction. Pour évaluer un modèle dans son ensemble, nous voulons mesurer sa performance moyenne sur toutes les données possibles, pas seulement sur les exemples que nous avons observés.

### Interprétation algorithmique

Imaginez que vous puissiez générer une infinité d'exemples $(x, y)$ en les tirant de la vraie distribution $p(x, y)$ de la nature. Pour chaque exemple, vous calculez la perte $\ell(y, f(x))$ entre la vraie valeur $y$ et la prédiction $f(x)$ de votre modèle. Le risque est simplement la moyenne de toutes ces pertes, calculée sur cet ensemble infini.

En termes algorithmiques, si nous avions accès à un générateur infini d'exemples de la vraie distribution, nous pourrions approximer le risque ainsi:

```python
# Pseudocode conceptuel (impossible en pratique)
total_loss = 0
count = 0
while True:  # Boucle infinie
    x, y = sample_from_true_distribution()  # Tirer de p(x,y)
    loss = loss_function(y, f(x))
    total_loss += loss
    count += 1
    risk = total_loss / count  # Converge vers le vrai risque
```

Bien sûr, nous ne pouvons pas exécuter cette boucle infinie. Mais cette intuition nous guide: le risque est ce que nous mesurerions si nous avions accès à toutes les données possibles.

### Définition formelle

Le **risque** d'une fonction $f$ est l'espérance de la perte sur la distribution des données:

$$
\mathcal{R}(f) = \mathbb{E}_{(X,Y) \sim p}\left[\ell(Y, f(X))\right] = \int \ell(y, f(x)) \, p(x, y) \, dx \, dy
$$

Décomposons cette formule étape par étape:

1. **$\mathbb{E}_{(X,Y) \sim p}$**: L'espérance mathématique signifie "moyenne sur tous les exemples possibles". La notation $(X,Y) \sim p$ indique que nous tirons les paires $(x, y)$ selon la distribution $p(x, y)$ de la nature.

2. **$\ell(Y, f(X))$**: Pour chaque exemple aléatoire $(X, Y)$, nous calculons la perte entre la vraie valeur $Y$ et la prédiction $f(X)$ du modèle.

3. **L'intégrale $\int \ell(y, f(x)) \, p(x, y) \, dx \, dy$**: Cette intégrale calcule une moyenne pondérée. Pour chaque paire possible $(x, y)$, nous multiplions la perte $\ell(y, f(x))$ par la probabilité $p(x, y)$ que cette paire apparaisse dans la nature, puis nous sommons (intégrons) sur toutes les paires possibles.

### Exemple concret

Considérons un problème de classification binaire en 2D. Supposons que $x \in [0, 1]^2$ et $y \in \{0, 1\}$. Pour calculer le risque, nous devrions:

1. Diviser l'espace $[0,1]^2$ en une grille fine (par exemple, $1000 \times 1000$ points)
2. Pour chaque point $x$ de la grille, considérer les deux valeurs possibles de $y$ (0 et 1)
3. Pour chaque combinaison $(x, y)$, calculer:
   - La probabilité $p(x, y)$ que cette combinaison apparaisse
   - La perte $\ell(y, f(x))$ si notre modèle prédit $f(x)$
4. Faire la somme pondérée: $\sum_{x} \sum_{y \in \{0,1\}} \ell(y, f(x)) \cdot p(x, y)$

En pratique, pour un espace continu, cette somme devient une intégrale sur un domaine continu, ce qui est encore plus complexe à calculer.

### Pourquoi le risque est important

Le risque mesure ce que nous obtiendrons en moyenne si nous appliquons $f$ à de nouvelles données tirées de la même distribution. Un modèle avec un faible risque fait de bonnes prédictions en général, pas seulement sur les exemples d'entraînement. C'est exactement ce que nous voulons optimiser: un modèle qui performe bien sur des données jamais vues, pas seulement sur celles qu'il a déjà observées.

Cette quantité est ce que nous voulons minimiser. Le problème fondamental est que nous ne connaissons pas la distribution $p(x, y)$ de la nature. Nous n'y avons accès qu'indirectement, via un échantillon fini $\mathcal{D}$.

## Le risque empirique

Puisque le risque est inaccessible, nous l'approximons par une moyenne sur les données disponibles. Le **risque empirique** est:

$$
\hat{\mathcal{R}}(f, \mathcal{D}) = \frac{1}{N} \sum_{i=1}^{N} \ell(y_i, f(x_i))
$$

Cette quantité est calculable: c'est la moyenne des pertes sur l'échantillon d'entraînement. Pour la perte 0-1, le risque empirique est le taux d'erreur sur les données d'entraînement. Pour la perte quadratique, c'est l'erreur quadratique moyenne.

Le risque empirique est une estimation du vrai risque. Avec suffisamment de données, si l'échantillon est représentatif de la distribution, le risque empirique devrait être proche du risque. La question de savoir quand et à quelle vitesse cette approximation est fiable relève de la théorie de la généralisation, que nous aborderons au chapitre suivant.

## Minimisation du risque empirique

Nous avons maintenant les éléments pour formuler l'apprentissage comme un problème d'optimisation. Nous cherchons la fonction $f$ dans une classe $\mathcal{F}$ qui minimise le risque:

$$
f^\star = \arg\min_{f \in \mathcal{F}} \mathcal{R}(f)
$$

Puisque le risque est inaccessible, nous le remplaçons par le risque empirique:

$$
\hat{f} = \arg\min_{f \in \mathcal{F}} \hat{\mathcal{R}}(f, \mathcal{D})
$$

Ce principe est la **minimisation du risque empirique** (MRE). L'idée est simple: choisir la fonction qui fait le moins d'erreurs sur les données d'entraînement, en espérant que cette performance se transfère aux nouvelles données.

La classe $\mathcal{F}$ est notre **classe d'hypothèses**. Elle représente l'ensemble des fonctions que nous sommes prêts à considérer. Par exemple, si $\mathcal{F}$ est l'ensemble des fonctions linéaires, nous cherchons la meilleure fonction linéaire. Si $\mathcal{F}$ est l'ensemble des polynômes de degré au plus $k$, nous cherchons le meilleur polynôme de ce degré. Le choix de $\mathcal{F}$ encode nos hypothèses sur la forme de la relation entre entrées et sorties.

La question centrale de la théorie de l'apprentissage est: quand le minimiseur du risque empirique a-t-il un faible risque? Si $\hat{f}$ minimise $\hat{\mathcal{R}}$ et $f^\star$ minimise $\mathcal{R}$, nous voulons que $\mathcal{R}(\hat{f})$ soit proche de $\mathcal{R}(f^\star)$. Cette question dépend de la taille de l'échantillon $N$, de la complexité de la classe $\mathcal{F}$, et de propriétés de la distribution $p$.

## Fonctions de perte de substitution

La perte 0-1 pose un problème pratique. Pour trouver le minimiseur du risque empirique, nous utilisons généralement des méthodes d'optimisation itératives comme la descente de gradient. Ces méthodes requièrent que la fonction objectif soit différentiable, or la perte 0-1 est constante par morceaux: sa dérivée est nulle presque partout et indéfinie aux points de discontinuité.

Nous contournons ce problème en utilisant des **fonctions de perte de substitution**: des approximations convexes et différentiables de la perte originale.

Pour la classification binaire avec $y \in \{-1, +1\}$, la **perte logistique** est:

$$
\ell_{\text{log}}(y, s) = \log(1 + e^{-y \cdot s})
$$

où $s = f(x)$ est le score produit par le modèle. Cette fonction est convexe et différentiable partout. Lorsque $y$ et $s$ ont le même signe (prédiction correcte avec confiance), la perte est faible. Lorsqu'ils ont des signes opposés (erreur), la perte croît linéairement avec l'amplitude de l'erreur.

La **perte à charnière** (hinge loss) est utilisée dans les machines à vecteurs de support:

$$
\ell_{\text{hinge}}(y, s) = \max(0, 1 - y \cdot s)
$$

Cette fonction est convexe mais non différentiable au point $y \cdot s = 1$. Elle est nulle lorsque la prédiction est correcte avec une marge suffisante ($y \cdot s \geq 1$), et croît linéairement sinon.

Ces deux fonctions majorent la perte 0-1: pour tout $y$ et $s$, nous avons $\ell_{0-1} \leq \ell_{\text{log}}$ et $\ell_{0-1} \leq \ell_{\text{hinge}}$. Minimiser ces substituts garantit donc un certain contrôle sur la perte originale.

## Maximum de vraisemblance

Une autre approche consiste à modéliser explicitement la distribution conditionnelle $p(y|x)$ et à trouver les paramètres qui rendent les données observées les plus probables.

Supposons que notre modèle définisse une distribution $p(y|x; \theta)$ paramétrée par $\theta$. La **vraisemblance** des paramètres étant donnée les données est:

$$
\mathcal{L}(\theta) = p(\mathcal{D}|\theta) = \prod_{i=1}^N p(y_i | x_i; \theta)
$$

Le produit découle de l'hypothèse que les exemples sont indépendants. L'**estimateur du maximum de vraisemblance** (EMV) est le $\theta$ qui maximise cette quantité:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \mathcal{L}(\theta)
$$

En pratique, nous maximisons le logarithme de la vraisemblance. Le logarithme transforme le produit en somme et ne change pas le maximiseur:

$$
\log \mathcal{L}(\theta) = \sum_{i=1}^N \log p(y_i | x_i; \theta)
$$

Pour l'optimisation, nous minimisons la **log-vraisemblance négative**:

$$
\text{NLL}(\theta) = -\sum_{i=1}^N \log p(y_i | x_i; \theta)
$$

### Régression avec bruit gaussien

En régression, nous modélisons souvent la sortie comme la prédiction du modèle plus un bruit gaussien:

$$
y = f(x; \theta) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)
$$

La distribution conditionnelle est donc:

$$
p(y|x; \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - f(x; \theta))^2}{2\sigma^2}\right)
$$

La log-vraisemblance négative devient:

$$
\text{NLL}(\theta) = \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - f(x_i; \theta))^2 + \frac{N}{2}\log(2\pi\sigma^2)
$$

Le second terme est constant par rapport à $\theta$. Minimiser la NLL revient donc à minimiser la somme des erreurs quadratiques. Sous l'hypothèse d'un bruit gaussien, le maximum de vraisemblance coïncide avec les moindres carrés.

### Classification binaire

Pour la classification binaire avec $y \in \{0, 1\}$, nous modélisons la probabilité de la classe positive par:

$$
p(y = 1 | x; \theta) = \sigma(f(x; \theta)) = \frac{1}{1 + e^{-f(x; \theta)}}
$$

où $\sigma$ est la fonction sigmoïde. La distribution conditionnelle suit une loi de Bernoulli:

$$
p(y|x; \theta) = \sigma(f(x; \theta))^y (1 - \sigma(f(x; \theta)))^{1-y}
$$

La log-vraisemblance négative est:

$$
\text{NLL}(\theta) = -\sum_{i=1}^N \left[ y_i \log \sigma(f(x_i; \theta)) + (1-y_i) \log(1 - \sigma(f(x_i; \theta))) \right]
$$

Cette quantité est l'**entropie croisée binaire**. Elle correspond à la perte logistique, à une reparamétrisation près.

## Lien entre MRE et maximum de vraisemblance

La minimisation du risque empirique et l'estimation par maximum de vraisemblance sont deux formulations du même problème lorsque nous choisissons la perte logarithmique $\ell(y, f(x)) = -\log p(y | f(x))$.

Le risque empirique avec cette perte est:

$$
\hat{\mathcal{R}}(\theta) = \frac{1}{N} \sum_{i=1}^N -\log p(y_i | x_i; \theta) = \frac{1}{N} \text{NLL}(\theta)
$$

Le minimiseur du risque empirique est donc l'estimateur du maximum de vraisemblance. Les deux approches, l'une fondée sur la théorie de la décision et l'autre sur l'inférence statistique, convergent vers le même algorithme.

## Interprétation informationnelle

Nous pouvons interpréter le maximum de vraisemblance du point de vue de la théorie de l'information. L'EMV trouve le modèle paramétrique le plus proche de la distribution empirique des données.

La **distribution empirique** place une masse $1/N$ sur chaque observation:

$$
p_{\mathcal{D}}(y) = \frac{1}{N} \sum_{i=1}^N \delta(y - y_i)
$$

La **divergence de Kullback-Leibler** mesure la dissimilarité entre deux distributions:

$$
D_{\text{KL}}(p \| q) = \sum_y p(y) \log \frac{p(y)}{q(y)}
$$

Cette quantité est toujours positive ou nulle, et vaut zéro si et seulement si $p = q$. Elle n'est pas symétrique: $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$ en général.

En posant $p = p_{\mathcal{D}}$ et $q = p(\cdot | \theta)$:

$$
D_{\text{KL}}(p_{\mathcal{D}} \| p(\cdot|\theta)) = -\mathbb{H}(p_{\mathcal{D}}) - \frac{1}{N} \sum_{i=1}^N \log p(y_i | \theta)
$$

Le premier terme, l'entropie de la distribution empirique, ne dépend pas de $\theta$. Minimiser la divergence KL revient à minimiser la log-vraisemblance négative:

$$
\arg\min_\theta D_{\text{KL}}(p_{\mathcal{D}} \| p(\cdot|\theta)) = \arg\min_\theta \text{NLL}(\theta)
$$

L'EMV trouve les paramètres qui rendent le modèle aussi proche que possible de la distribution empirique au sens de la divergence KL.

## Classes de modèles

Nous n'avons pas encore précisé la forme des fonctions $f$ que nous considérons. Le choix de la classe d'hypothèses $\mathcal{F}$ détermine ce que le modèle peut représenter.

Le modèle le plus simple est la **régression linéaire**:

$$
f(x; w, b) = w^\top x + b = \sum_{j=1}^d w_j x_j + b
$$

Les paramètres sont le vecteur de poids $w \in \mathbb{R}^d$ et le biais $b \in \mathbb{R}$. Ce modèle suppose que la sortie est une combinaison linéaire des entrées.

Pour capturer des relations non linéaires tout en gardant un modèle linéaire dans les paramètres, nous pouvons transformer les entrées. En **régression polynomiale**, nous appliquons une fonction $\phi: \mathbb{R} \to \mathbb{R}^{k+1}$:

$$
\phi(x) = [1, x, x^2, \ldots, x^k]
$$

La prédiction devient $f(x; w) = w^\top \phi(x)$. Le modèle est polynomial en $x$ mais linéaire en $w$, ce qui permet d'utiliser les mêmes algorithmes d'optimisation.

Le degré $k$ contrôle la **capacité** du modèle: sa capacité à représenter des fonctions complexes. Avec $k = 1$, nous avons une droite. Avec $k$ élevé, le polynôme peut osciller pour passer par tous les points d'entraînement. Avec $k = N - 1$, nous pouvons interpoler exactement les $N$ points: le risque empirique atteint zéro. Mais un polynôme qui passe exactement par les points d'entraînement n'a aucune raison de bien prédire les nouveaux points.

## Généralisation

La différence entre le risque et le risque empirique est l'**écart de généralisation**:

$$
\text{Écart} = \mathcal{R}(f) - \hat{\mathcal{R}}(f; \mathcal{D}_{\text{train}})
$$

Un modèle qui minimise le risque empirique peut avoir un risque élevé si cet écart est grand. Ce phénomène est le **surapprentissage**: le modèle s'ajuste aux particularités de l'échantillon d'entraînement, y compris le bruit, plutôt qu'aux régularités sous-jacentes. L'erreur d'entraînement est faible, mais l'erreur sur de nouvelles données est élevée.

À l'inverse, un modèle trop simple peut avoir un risque empirique et un risque tous deux élevés. C'est le **sous-apprentissage**: le modèle n'a pas la capacité de capturer la structure des données.

En pratique, nous estimons le risque par le risque empirique sur un **ensemble de test** $\mathcal{D}_{\text{test}}$ disjoint de l'ensemble d'entraînement. Un troisième ensemble, l'**ensemble de validation**, sert à choisir parmi plusieurs modèles ou à régler des hyperparamètres. L'ensemble de test doit rester intact jusqu'à l'évaluation finale, pour fournir une estimation non biaisée.

Cette séparation est importante. Si nous utilisons l'ensemble de test pour faire des choix (quel modèle garder, quelle valeur d'hyperparamètre utiliser), l'estimation de performance sur ce même ensemble devient optimiste. Nous aurions alors besoin d'un quatrième ensemble pour obtenir une estimation fiable.

## Biais inductifs

Il n'existe pas de modèle universel qui fonctionne optimalement pour tous les problèmes. Ce résultat, connu sous le nom de **théorème du no free lunch**, affirme qu'un algorithme d'apprentissage qui performe bien sur une classe de problèmes performe nécessairement moins bien sur d'autres.

Tout modèle encode des **biais inductifs**: des hypothèses implicites ou explicites sur la structure du problème. La régression linéaire suppose que la relation entre entrées et sorties est linéaire. Les k plus proches voisins supposent que les points proches dans l'espace des entrées ont des sorties similaires. Les réseaux convolutifs supposent que les motifs locaux dans une image sont informatifs indépendamment de leur position.

Ces hypothèses sont nécessaires pour que l'apprentissage soit possible. Sans elles, nous n'aurions aucune raison de croire que la performance sur l'échantillon d'entraînement prédit la performance sur de nouvelles données. Le choix du modèle et de ses hypothèses est une décision que l'algorithme ne peut pas prendre seul; elle requiert une connaissance du domaine.

## Résumé

Ce chapitre a établi le cadre formel de l'apprentissage supervisé. Nous avons défini le risque comme la mesure de performance que nous voulons optimiser, et le risque empirique comme son approximation calculable. Le principe de minimisation du risque empirique consiste à choisir le modèle qui minimise cette approximation.

L'estimation par maximum de vraisemblance offre une perspective complémentaire, fondée sur l'inférence statistique. Les deux approches coïncident pour la perte logarithmique. L'interprétation en termes de divergence KL montre que le maximum de vraisemblance trouve le modèle le plus proche de la distribution empirique.

La question centrale que ce chapitre laisse ouverte est celle de la généralisation: quand le risque empirique est-il un bon indicateur du vrai risque? Cette question dépend de la taille de l'échantillon, de la complexité du modèle, et de la distribution des données. Le chapitre suivant développe les outils pour y répondre.
