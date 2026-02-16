---
kernelspec:
  name: python3
  display_name: Python 3
---

# Modèles probabilistes génératifs

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Distinguer les approches générative et discriminative pour la classification
- Dériver l'estimateur du maximum de vraisemblance pour le classifieur naïf bayésien
- Expliquer pourquoi l'hypothèse «naïve» d'indépendance conditionnelle fonctionne souvent bien
- Appliquer l'analyse discriminante gaussienne (LDA et QDA)
- Comprendre les modèles de mélange gaussien (GMM) comme généralisation de k-moyennes
- Décrire l'algorithme EM et l'appliquer aux GMM
- Relier EM à l'inférence variationnelle et à la maximisation de l'ELBO (borne inférieure de l'évidence)
```

Le [chapitre précédent](ch5_probabilistic.md) a présenté le cadre bayésien et montré comment le maximum de vraisemblance découle de principes probabilistes. Ce chapitre exploite ce cadre pour construire des **modèles génératifs**: des modèles qui décrivent comment les données sont produites. Cette perspective ouvre de nouvelles possibilités pour la classification et le partitionnement.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Configuration pour des figures haute résolution
%config InlineBackend.figure_format = 'retina'
```

## Approches générative et discriminative

Le [chapitre 3](ch3_classification.md) a introduit la régression logistique, qui modélise directement la probabilité qu'une observation appartienne à chaque classe:

$$
p(y = c \mid \mathbf{x}; \boldsymbol{\theta}) = \frac{\exp(f_c(\mathbf{x}; \boldsymbol{\theta}))}{\sum_{c'} \exp(f_{c'}(\mathbf{x}; \boldsymbol{\theta}))}
$$

Cette approche est dite **discriminative**: elle apprend à distinguer les classes sans modéliser comment les données de chaque classe sont distribuées. Le modèle répond à la question «étant donné cette observation, quelle est sa classe probable?» sans se demander «à quoi ressemblent les observations de chaque classe?».

L'approche **générative** procède différemment. Au lieu de modéliser $p(y \mid \mathbf{x})$ directement, elle modélise:

1. La **distribution a priori** des classes: $p(y = c)$
2. La **vraisemblance conditionnelle** de chaque classe: $p(\mathbf{x} \mid y = c)$

Le théorème de Bayes permet ensuite de calculer la probabilité a posteriori:

$$
p(y = c \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid y = c) \, p(y = c)}{\sum_{c'} p(\mathbf{x} \mid y = c') \, p(y = c')}
$$

Le terme «génératif» vient du fait que ce modèle décrit un processus de génération des données: d'abord tirer une classe $c$ selon $p(y)$, puis générer une observation $\mathbf{x}$ selon $p(\mathbf{x} \mid y = c)$. Nous pouvons utiliser ce processus pour créer des données synthétiques.

```{code-cell} python
:tags: [hide-input]

# Illustration: génératif vs discriminatif
np.random.seed(42)

# Générer des données de deux classes
n_per_class = 100
mu0, mu1 = np.array([0, 0]), np.array([2.5, 2.5])
cov = np.array([[1, 0.5], [0.5, 1]])

X0 = np.random.multivariate_normal(mu0, cov, n_per_class)
X1 = np.random.multivariate_normal(mu1, cov, n_per_class)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Gauche: vue générative (modélise chaque classe séparément)
ax = axes[0]
ax.scatter(X0[:, 0], X0[:, 1], c='steelblue', alpha=0.6, label='Classe 0', s=30)
ax.scatter(X1[:, 0], X1[:, 1], c='coral', alpha=0.6, label='Classe 1', s=30)

# Contours des distributions
x_grid = np.linspace(-3, 6, 100)
y_grid = np.linspace(-3, 6, 100)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
pos = np.dstack((X_grid, Y_grid))

rv0 = stats.multivariate_normal(mu0, cov)
rv1 = stats.multivariate_normal(mu1, cov)

ax.contour(X_grid, Y_grid, rv0.pdf(pos), levels=3, colors='steelblue', alpha=0.7, linestyles='--')
ax.contour(X_grid, Y_grid, rv1.pdf(pos), levels=3, colors='coral', alpha=0.7, linestyles='--')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Approche générative\nModélise $p(\\mathbf{x} \\mid y)$ pour chaque classe')
ax.legend(loc='upper left')
ax.set_xlim(-3, 6)
ax.set_ylim(-3, 6)

# Droite: vue discriminative (modélise la frontière)
ax = axes[1]
ax.scatter(X0[:, 0], X0[:, 1], c='steelblue', alpha=0.6, label='Classe 0', s=30)
ax.scatter(X1[:, 0], X1[:, 1], c='coral', alpha=0.6, label='Classe 1', s=30)

# Frontière de décision (pour LDA avec covariance partagée)
# La frontière est là où p(y=0|x) = p(y=1|x)
cov_inv = np.linalg.inv(cov)
w = cov_inv @ (mu1 - mu0)
b = -0.5 * (mu1 @ cov_inv @ mu1 - mu0 @ cov_inv @ mu0)
# Frontière: w'x + b = 0
x_line = np.linspace(-3, 6, 100)
y_line = -(w[0] * x_line + b) / w[1]

ax.plot(x_line, y_line, 'k-', linewidth=2, label='Frontière de décision')
ax.fill_between(x_line, y_line, 6, alpha=0.1, color='coral')
ax.fill_between(x_line, -3, y_line, alpha=0.1, color='steelblue')

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Approche discriminative\nModélise $p(y \\mid \\mathbf{x})$ directement')
ax.legend(loc='upper left')
ax.set_xlim(-3, 6)
ax.set_ylim(-3, 6)

plt.tight_layout()
```

La figure illustre les deux perspectives. À gauche, l'approche générative modélise la distribution de chaque classe (les ellipses montrent les contours de densité). À droite, l'approche discriminative se concentre sur la frontière qui sépare les classes. Les deux approches peuvent donner la même frontière de décision, mais elles y arrivent par des chemins différents.

### Avantages et limites

Chaque approche a ses forces. L'approche discriminative optimise directement ce qui nous intéresse: la capacité à distinguer les classes. Elle fait moins d'hypothèses sur la forme des distributions et atteint souvent une meilleure précision prédictive.

L'approche générative offre d'autres avantages:

- **Génération de données**: nous pouvons créer des exemples synthétiques, utiles pour l'augmentation de données ou la visualisation
- **Données manquantes**: si certaines caractéristiques sont absentes, nous pouvons marginaliser sur les valeurs manquantes
- **Apprentissage par classe**: nous pouvons ajouter une nouvelle classe sans réentraîner les autres
- **Apprentissage avec peu de données**: les hypothèses du modèle génératif peuvent aider quand les exemples sont rares

La suite de ce chapitre présente trois modèles génératifs: le classifieur naïf bayésien, l'analyse discriminante gaussienne, et les modèles de mélange gaussien.

## Le classifieur naïf bayésien

### L'hypothèse d'indépendance conditionnelle

Le classifieur naïf bayésien (*Naive Bayes*) est un modèle génératif simple mais efficace. Son nom vient d'une hypothèse qui simplifie considérablement le modèle: les caractéristiques sont **conditionnellement indépendantes** étant donné la classe.

Pour comprendre l'impact de cette hypothèse, considérons d'abord le cas général. Par la règle de chaîne, la vraisemblance conditionnelle de classe se décompose en:

$$
p(\mathbf{x} \mid y = c) = p(x_1 \mid y = c)\, p(x_2 \mid x_1, y = c) \cdots p(x_D \mid x_1, \ldots, x_{D-1}, y = c)
$$

Chaque facteur dépend de toutes les caractéristiques précédentes. Si chaque caractéristique prend $K$ valeurs, le dernier facteur à lui seul nécessite de spécifier une distribution conditionnelle pour chacune des $K^{D-1}$ combinaisons possibles des caractéristiques précédentes. Au total, la distribution conjointe $p(\mathbf{x} \mid y = c)$ requiert $K^D - 1$ paramètres par classe, un nombre qui explose exponentiellement avec la dimension.

L'hypothèse d'indépendance conditionnelle élimine toutes ces dépendances. Sachant la classe, chaque caractéristique est supposée indépendante des autres:

$$
p(\mathbf{x} \mid y = c) = \prod_{d=1}^D p(x_d \mid y = c)
$$

Cette factorisation réduit drastiquement le nombre de paramètres: nous n'avons plus que $D(K - 1)$ paramètres par classe. Par exemple, avec $D = 20$ caractéristiques binaires ($K = 2$), le modèle général nécessiterait $2^{20} - 1 \approx 10^6$ paramètres par classe, alors que le naïf bayésien n'en utilise que 20.

Concrètement, considérons un problème de classification de courriels (pourriel ou non) avec des caractéristiques binaires indiquant la présence de certains mots. L'hypothèse d'indépendance conditionnelle suppose que, sachant qu'un courriel est un pourriel, la présence du mot «gratuit» n'influence pas la probabilité de présence du mot «urgent». Chaque mot apparaît indépendamment selon sa propre probabilité conditionnelle à la classe.

```{margin} Pourquoi «naïf»?
Le terme «naïf» ne signifie pas que le modèle est stupide ou simpliste. Il indique que l'hypothèse d'indépendance conditionnelle est rarement vraie en pratique. Dans notre exemple de courriels, les mots «gratuit» et «offre» apparaissent souvent ensemble dans les pourriels, alors qu'ils ne sont pas vraiment indépendants. Pourtant, le classifieur fonctionne bien malgré cette violation de l'hypothèse. Cette robustesse fait du naïf bayésien un outil pratique, pas une méthode à éviter.
```

### Modèle complet et classification

Pour tout modèle génératif, le théorème de Bayes donne la probabilité a posteriori d'une classe:

$$
p(y = c \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid y = c)\, p(y = c)}{\sum_{c'} p(\mathbf{x} \mid y = c')\, p(y = c')}
$$

Le modèle naïf bayésien spécifie:

1. Un a priori sur les classes: $p(y = c) = \pi_c$ avec $\sum_c \pi_c = 1$
2. Pour chaque caractéristique $d$ et chaque classe $c$, une distribution $p(x_d \mid y = c; \boldsymbol{\theta}_{dc})$

En substituant l'hypothèse d'indépendance conditionnelle dans le théorème de Bayes, la probabilité a posteriori devient:

$$
p(y = c \mid \mathbf{x}) = \frac{\pi_c \prod_{d=1}^D p(x_d \mid y = c)}{\sum_{c'} \pi_{c'} \prod_{d=1}^D p(x_d \mid y = c')}
$$

Pour classifier, nous choisissons la classe qui maximise le numérateur (le dénominateur est constant pour toutes les classes):

$$
\hat{y} = \arg\max_c \, \pi_c \prod_{d=1}^D p(x_d \mid y = c)
$$

En pratique, nous travaillons avec le logarithme pour éviter les problèmes de sous-dépassement numérique (*underflow*):

$$
\hat{y} = \arg\max_c \left[ \log \pi_c + \sum_{d=1}^D \log p(x_d \mid y = c) \right]
$$

### Estimation par maximum de vraisemblance

Un atout du naïf bayésien est que l'estimation des paramètres admet des formules fermées. La log-vraisemblance se factorise en termes indépendants:

$$
\log p(\mathcal{D} \mid \boldsymbol{\theta}) = \underbrace{\sum_{n=1}^N \log p(y_n \mid \boldsymbol{\pi})}_{\text{terme des classes}} + \sum_{d=1}^D \underbrace{\sum_{n=1}^N \log p(x_{nd} \mid y_n; \boldsymbol{\theta}_d)}_{\text{terme de la caractéristique } d}
$$

Cette factorisation permet d'optimiser chaque terme séparément.

**A priori de classe.** L'EMV des probabilités de classe est simplement la fréquence empirique:

$$
\hat{\pi}_c = \frac{N_c}{N}
$$

où $N_c$ est le nombre d'exemples de classe $c$.

**Caractéristiques catégorielles.** Si la caractéristique $d$ prend des valeurs parmi $\{1, \ldots, K\}$, l'EMV est:

$$
\hat{\theta}_{dck} = \frac{N_{dck}}{N_c}
$$

où $N_{dck}$ compte les exemples de classe $c$ où la caractéristique $d$ vaut $k$.

**Caractéristiques binaires.** Pour des caractéristiques binaires (présent/absent), nous utilisons une distribution de Bernoulli:

$$
\hat{\theta}_{dc} = \frac{N_{dc}}{N_c}
$$

où $N_{dc}$ compte les exemples de classe $c$ où la caractéristique $d$ est présente.

**Caractéristiques continues.** Pour des caractéristiques continues, nous supposons souvent une distribution gaussienne et estimons la moyenne et la variance par classe:

$$
\hat{\mu}_{dc} = \frac{1}{N_c} \sum_{n: y_n = c} x_{nd}, \qquad \hat{\sigma}^2_{dc} = \frac{1}{N_c} \sum_{n: y_n = c} (x_{nd} - \hat{\mu}_{dc})^2
$$

### Le problème des probabilités nulles et le lissage de Laplace

Un problème survient quand une combinaison caractéristique-classe n'apparaît jamais dans les données d'entraînement. Si le mot «gratuit» n'apparaît dans aucun courriel légitime, nous avons $\hat{\theta}_{dc} = 0$. Lors de la classification d'un nouveau courriel contenant «gratuit», le produit $\prod_d p(x_d \mid y = \text{légitime})$ devient nul, quelle que soit la valeur des autres caractéristiques. Un seul mot peut ainsi dominer entièrement la décision.

Le **lissage de Laplace** (*add-one smoothing*) résout ce problème en ajoutant des pseudo-observations:

$$
\hat{\theta}_{dck} = \frac{N_{dck} + 1}{N_c + K}
$$

où $K$ est le nombre de valeurs possibles. Cette formule garantit que toutes les probabilités restent strictement positives.

Le lissage de Laplace a une interprétation bayésienne: c'est l'estimateur MAP avec un a priori uniforme (Beta(1,1) pour le cas binaire, Dirichlet(1,...,1) pour le cas catégoriel). Nous retrouvons ici le lien entre régularisation et a priori établi au [chapitre 5](ch5_probabilistic.md).

```{code-cell} python
:tags: [hide-input]

# Exemple: effet du lissage de Laplace
fig, ax = plt.subplots(figsize=(9, 4))

# Scénario: 10 pourriels, 0 avec le mot "gratuit"
N_spam = 10
N_gratuit_spam = 0
K = 2  # binaire

# EMV
theta_mle = N_gratuit_spam / N_spam

# Avec lissage de Laplace
alphas = [0, 0.1, 0.5, 1, 2, 5]
thetas = [(N_gratuit_spam + alpha) / (N_spam + K * alpha) for alpha in alphas]

bars = ax.bar(range(len(alphas)), thetas, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xticks(range(len(alphas)))
ax.set_xticklabels([f'$\\alpha = {a}$' for a in alphas])
ax.set_ylabel('$\\hat{\\theta}$ (probabilité estimée)')
ax.set_xlabel('Paramètre de lissage')
ax.set_title('Effet du lissage sur $p(\\text{«gratuit»} \\mid \\text{pourriel})$\n(0 occurrence sur 10 exemples)')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='A priori uniforme')
ax.legend()
ax.set_ylim(0, 0.6)

# Annoter les valeurs
for bar, theta in zip(bars, thetas):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{theta:.2f}', ha='center', fontsize=9)

plt.tight_layout()
```

La figure montre comment le lissage affecte l'estimation. Sans lissage ($\alpha = 0$), l'EMV est zéro, ce qui pose problème. Avec $\alpha = 1$ (lissage de Laplace standard), l'estimation devient $1/12 \approx 0{,}08$, reflétant notre incertitude face à l'absence de données.

### Pourquoi le naïf bayésien fonctionne-t-il?

L'hypothèse d'indépendance conditionnelle est presque toujours violée en pratique. Pourtant, le classifieur naïf bayésien obtient souvent de bonnes performances. Comment expliquer ce paradoxe?

La réponse tient au fait que nous utilisons le modèle pour **classifier**, pas pour estimer des probabilités précises. Pour classifier correctement, nous n'avons besoin que de la classe la plus probable, pas des probabilités exactes. Même si les probabilités estimées sont biaisées, l'**ordre** des classes peut rester correct.

Plus précisément, les dépendances entre caractéristiques peuvent affecter les probabilités absolues sans changer quelle classe domine. Si les mots «gratuit» et «offre» sont corrélés dans les pourriels, ignorer cette corrélation surestime la «surprise» de voir les deux ensemble; cette surestimation s'applique toutefois à toutes les classes et peut s'annuler dans la comparaison.

Cette observation a une conséquence pratique: les probabilités retournées par un naïf bayésien sont souvent mal calibrées (trop proches de 0 ou 1). Si vous avez besoin de probabilités fiables et pas seulement de classifications, d'autres méthodes comme la régression logistique sont préférables.

```{code-cell} python
:tags: [hide-input]

# Démonstration: Naive Bayes sur un exemple de classification de texte
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Données d'exemple (classification de sentiment)
texts = [
    "ce film est excellent vraiment superbe",
    "quelle merveille un chef-d'oeuvre",
    "j'ai adoré ce film magnifique",
    "film ennuyeux et long très décevant",
    "terrible je n'ai pas aimé du tout",
    "mauvais film vraiment nul"
]
labels = [1, 1, 1, 0, 0, 0]  # 1 = positif, 0 = négatif

# Vectorisation (comptage des mots)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Entraînement du Naive Bayes
clf = MultinomialNB(alpha=1.0)  # alpha=1 = lissage de Laplace
clf.fit(X, labels)

# Test sur de nouveaux exemples
test_texts = ["ce film est superbe", "film terrible et ennuyeux"]
X_test = vectorizer.transform(test_texts)
predictions = clf.predict(X_test)
probas = clf.predict_proba(X_test)

fig, ax = plt.subplots(figsize=(9, 4))

x_pos = np.arange(len(test_texts))
width = 0.35

bars1 = ax.bar(x_pos - width/2, probas[:, 0], width, label='$p(\\text{négatif} \\mid \\mathbf{x})$', 
               color='coral', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x_pos + width/2, probas[:, 1], width, label='$p(\\text{positif} \\mid \\mathbf{x})$', 
               color='steelblue', alpha=0.7, edgecolor='black')

ax.set_ylabel('Probabilité a posteriori')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'«{t[:25]}...»' if len(t) > 25 else f'«{t}»' for t in test_texts], fontsize=9)
ax.legend()
ax.set_ylim(0, 1.1)
ax.set_title('Classification de sentiment avec Naive Bayes')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.2f}', 
                ha='center', fontsize=9)

plt.tight_layout()
```

## Analyse discriminante gaussienne

### Modèle

L'**analyse discriminante gaussienne** (GDA, *Gaussian Discriminant Analysis*) est un cas particulier de modèle génératif où les vraisemblances conditionnelles de classe sont des distributions gaussiennes:

$$
p(\mathbf{x} \mid y = c) = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)
$$

Chaque classe $c$ est caractérisée par:
- Un vecteur moyenne $\boldsymbol{\mu}_c \in \mathbb{R}^D$
- Une matrice de covariance $\boldsymbol{\Sigma}_c \in \mathbb{R}^{D \times D}$

Le modèle complet inclut aussi les probabilités a priori $\pi_c = p(y = c)$.

### La fonction discriminante

Pour classifier, nous calculons la probabilité a posteriori de chaque classe et choisissons la plus grande. En prenant le logarithme:

$$
\log p(y = c \mid \mathbf{x}) = \log \pi_c + \log \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c) - \log p(\mathbf{x})
$$

Le terme $\log p(\mathbf{x})$ est constant pour toutes les classes et peut être ignoré pour la classification. La **fonction discriminante** pour la classe $c$ est:

$$
\delta_c(\mathbf{x}) = \log \pi_c - \frac{1}{2}\log|\boldsymbol{\Sigma}_c| - \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_c)^\top \boldsymbol{\Sigma}_c^{-1}(\mathbf{x} - \boldsymbol{\mu}_c)
$$

Le terme $(\mathbf{x} - \boldsymbol{\mu}_c)^\top \boldsymbol{\Sigma}_c^{-1}(\mathbf{x} - \boldsymbol{\mu}_c)$ est la **distance de Mahalanobis** entre $\mathbf{x}$ et $\boldsymbol{\mu}_c$. Cette distance tient compte de la forme de la distribution: un point éloigné dans une direction de grande variance est moins «surprenant» qu'un point éloigné dans une direction de faible variance.

### Analyse discriminante quadratique (QDA)

Quand chaque classe a sa propre matrice de covariance $\boldsymbol{\Sigma}_c$, la fonction discriminante contient un terme quadratique en $\mathbf{x}$. La frontière de décision entre deux classes (là où $\delta_c(\mathbf{x}) = \delta_{c'}(\mathbf{x})$) est une **quadrique** (une ellipse, une hyperbole ou une parabole selon la configuration). Cette méthode s'appelle **QDA** (*Quadratic Discriminant Analysis*).

### Analyse discriminante linéaire (LDA)

Si toutes les classes partagent la **même matrice de covariance** $\boldsymbol{\Sigma}$, les termes quadratiques se simplifient:

$$
\delta_c(\mathbf{x}) = \log \pi_c - \frac{1}{2}\boldsymbol{\mu}_c^\top \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_c + \mathbf{x}^\top \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_c
$$

Cette expression est **linéaire** en $\mathbf{x}$. La frontière de décision entre deux classes devient un hyperplan. Cette méthode s'appelle **LDA** (*Linear Discriminant Analysis*).

La différence entre LDA et QDA est analogue à celle entre un modèle linéaire et un modèle quadratique en régression: LDA est plus simple et moins sujet au surapprentissage, mais QDA peut capturer des frontières plus complexes.

```{code-cell} python
:tags: [hide-input]

# Comparaison LDA vs QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

np.random.seed(42)

# Générer des données avec covariances différentes
n_samples = 150
mu0 = np.array([0, 0])
mu1 = np.array([3, 3])
cov0 = np.array([[2, 0.5], [0.5, 0.5]])
cov1 = np.array([[0.5, -0.3], [-0.3, 2]])

X0 = np.random.multivariate_normal(mu0, cov0, n_samples)
X1 = np.random.multivariate_normal(mu1, cov1, n_samples)
X = np.vstack([X0, X1])
y = np.array([0]*n_samples + [1]*n_samples)

# Entraîner LDA et QDA
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()
lda.fit(X, y)
qda.fit(X, y)

# Grille pour les frontières de décision
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, clf, title in [(axes[0], lda, 'LDA (covariance partagée)'), 
                        (axes[1], qda, 'QDA (covariances différentes)')]:
    Z = clf.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.contour(xx, yy, Z, colors='black', linewidths=1, levels=[0.5])
    
    ax.scatter(X0[:, 0], X0[:, 1], c='steelblue', alpha=0.6, label='Classe 0', s=20)
    ax.scatter(X1[:, 0], X1[:, 1], c='coral', alpha=0.6, label='Classe 1', s=20)
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title)
    ax.legend(loc='upper left')

plt.tight_layout()
```

La figure montre la différence entre LDA et QDA sur des données où les classes ont des covariances différentes. LDA impose une frontière linéaire qui ne peut pas s'adapter aux formes elliptiques différentes. QDA capture mieux la structure des données avec une frontière courbe.

### Estimation des paramètres

L'EMV des paramètres de GDA a des formules fermées:

**A priori de classe:**
$$
\hat{\pi}_c = \frac{N_c}{N}
$$

**Moyenne par classe:**
$$
\hat{\boldsymbol{\mu}}_c = \frac{1}{N_c} \sum_{n: y_n = c} \mathbf{x}_n
$$

**Covariance par classe (QDA):**
$$
\hat{\boldsymbol{\Sigma}}_c = \frac{1}{N_c} \sum_{n: y_n = c} (\mathbf{x}_n - \hat{\boldsymbol{\mu}}_c)(\mathbf{x}_n - \hat{\boldsymbol{\mu}}_c)^\top
$$

**Covariance partagée (LDA):**
$$
\hat{\boldsymbol{\Sigma}} = \frac{1}{N} \sum_{c=1}^C \sum_{n: y_n = c} (\mathbf{x}_n - \hat{\boldsymbol{\mu}}_c)(\mathbf{x}_n - \hat{\boldsymbol{\mu}}_c)^\top
$$

Ces formules sont des moyennes et des covariances empiriques, calculables efficacement sans optimisation itérative.

### Lien avec la régression logistique

LDA et la régression logistique partagent la même forme de frontière de décision (linéaire), mais diffèrent dans leurs hypothèses. LDA suppose que les données de chaque classe suivent une distribution gaussienne avec covariance partagée. La régression logistique ne fait pas d'hypothèse sur la distribution des données.

Quand l'hypothèse gaussienne est correcte, LDA peut être plus efficace avec peu de données car elle exploite cette structure. Quand l'hypothèse est incorrecte, la régression logistique est généralement plus robuste. En pratique, la régression logistique domine souvent car l'hypothèse gaussienne est rarement satisfaite exactement.

## Modèles de mélange gaussien

### De la classification au partitionnement

Les modèles précédents supposent que nous connaissons les classes des exemples d'entraînement. En pratique, les étiquettes sont souvent absentes. Un commerce cherche à identifier des profils de clientèle à partir de données de transactions; un généticien veut découvrir des sous-types de maladies à partir de profils d'expression génétique; un système de sécurité doit repérer des comportements atypiques sans exemples préalables d'attaques.

Le **partitionnement** (*clustering*) regroupe automatiquement les observations en groupes homogènes, sans supervision. Nous allons aborder ce problème en deux temps. D'abord, l'algorithme k-moyennes donne une solution simple et intuitive. Ensuite, nous verrons que k-moyennes fait des hypothèses implicites sur la forme des groupes, ce qui nous mènera aux modèles de mélange gaussien et à l'algorithme EM.

### K-moyennes: un premier algorithme

L'idée de k-moyennes est de représenter chaque groupe par un **centroïde** $\boldsymbol{\mu}_k$ (sa moyenne), puis d'assigner chaque observation au centroïde le plus proche. L'algorithme minimise la **distorsion**, c'est-à-dire la somme des distances au carré entre chaque point et le centroïde de son groupe:

$$
\mathcal{L} = \sum_{n=1}^N \sum_{k=1}^K r_{nk} \|\mathbf{x}_n - \boldsymbol{\mu}_k\|^2
$$

où $r_{nk} \in \{0, 1\}$ est l'assignation du point $n$ au groupe $k$ (avec $\sum_k r_{nk} = 1$). Ce problème d'optimisation porte à la fois sur des variables continues (les centroïdes $\boldsymbol{\mu}_k$) et des variables discrètes (les assignations $r_{nk} \in \{0,1\}$). Les assignations discrètes rendent la distorsion non différentiable par rapport aux $r_{nk}$: on ne peut pas calculer un gradient et «descendre» dans la direction des meilleures assignations. De plus, le nombre total d'assignations possibles est $K^N$ (chacun des $N$ points peut aller dans l'un des $K$ groupes), ce qui exclut toute recherche exhaustive dès que $N$ dépasse quelques dizaines.

On pourrait être tenté de relâcher la contrainte $r_{nk} \in \{0,1\}$ et d'autoriser des assignations continues $r_{nk} \in [0,1]$, par exemple via un softmax, pour rendre le problème différentiable et appliquer la descente de gradient. C'est une bonne intuition: elle mène directement à l'algorithme EM que nous verrons plus loin, où les responsabilités $r_{nk}$ sont précisément des assignations souples dans $[0,1]$. Mais k-moyennes choisit de rester dans le monde discret: il alterne entre deux étapes, chacune ayant une solution simple:

**Assignation.** On fixe les centroïdes et on assigne chaque point au plus proche:
$$r_{nk} = \begin{cases} 1 & \text{si } k = \arg\min_{k'} \|\mathbf{x}_n - \boldsymbol{\mu}_{k'}\|^2 \\ 0 & \text{sinon} \end{cases}$$

**Mise à jour.** On fixe les assignations et on recalcule les centroïdes:
$$\boldsymbol{\mu}_k = \frac{\sum_n r_{nk} \, \mathbf{x}_n}{\sum_n r_{nk}}$$

Chaque centroïde est la moyenne des points qui lui sont assignés. On répète ces deux étapes jusqu'à ce que les assignations ne changent plus. Chaque étape réduit (ou maintient) la distorsion, et comme le nombre d'assignations possibles est fini, l'algorithme converge toujours vers un minimum local.

```{code-cell} python
:tags: [hide-input]

# Animation de l'algorithme k-moyennes
from matplotlib.animation import FuncAnimation
from IPython.display import Image

np.random.seed(42)
n_samples = 300
X_kmeans = np.vstack([
    np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_samples // 3),
    np.random.multivariate_normal([4, 0], [[0.5, 0.3], [0.3, 0.5]], n_samples // 3),
    np.random.multivariate_normal([2, 3], [[0.8, -0.4], [-0.4, 0.8]], n_samples // 3)
])

def run_kmeans(X, K, seed=123, max_iter=20):
    rng = np.random.RandomState(seed)
    mu = X[rng.choice(len(X), K, replace=False)].copy()
    history = []
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None] - mu[None], axis=2)
        labels = np.argmin(dists, axis=1)
        history.append((mu.copy(), labels.copy()))
        new_mu = np.array([X[labels == k].mean(axis=0) for k in range(K)])
        if np.allclose(new_mu, mu):
            break
        mu = new_mu
    return history

hist_km = run_kmeans(X_kmeans, 3, seed=123)

fig, ax = plt.subplots(figsize=(8, 6))
colors_km = ['steelblue', 'coral', 'seagreen']

def animate_km(frame):
    ax.clear()
    mu_t, labels_t = hist_km[frame]
    for k in range(3):
        mask = labels_t == k
        ax.scatter(X_kmeans[mask, 0], X_kmeans[mask, 1], c=colors_km[k], alpha=0.5, s=15)
        ax.plot(mu_t[k, 0], mu_t[k, 1], marker='X', color=colors_km[k],
                markersize=14, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
    ax.set_xlim(-3, 7)
    ax.set_ylim(-4, 6)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(f'K-moyennes, itération {frame}')
    return []

anim_km = FuncAnimation(fig, animate_km, frames=len(hist_km), interval=800, blit=True)
anim_km.save('_static/kmeans_convergence.gif', writer='pillow', fps=2, dpi=100)
plt.close()

Image(filename='_static/kmeans_convergence.gif')
```

L'animation montre la convergence de k-moyennes. Les croix marquent les centroïdes, et la couleur de chaque point indique son assignation au centroïde le plus proche. À chaque itération, les centroïdes migrent vers le centre de masse de leur groupe, et les assignations se réorganisent en conséquence.

K-moyennes est rapide et simple, mais il a une limite structurelle. Puisque chaque point est assigné au centroïde le plus proche au sens de la distance euclidienne, la séparation entre deux groupes adjacents est toujours la **médiatrice** du segment reliant leurs centroïdes, c'est-à-dire une droite perpendiculaire passant par le milieu. L'ensemble de ces médiatrices forme un diagramme de Voronoï dont les cellules sont des polygones convexes. Les groupes retrouvés sont donc nécessairement **sphériques**: k-moyennes ne peut pas capturer des groupes allongés, inclinés ou de tailles différentes.

```{code-cell} python
:tags: [hide-input]

# Illustration: k-moyennes (médiatrice) vs GMM (séparation adaptée)
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

np.random.seed(42)

# Données avec deux groupes elliptiques d'orientations différentes
cov_a = np.array([[3.0, 1.8], [1.8, 1.5]])
cov_b = np.array([[1.0, -0.7], [-0.7, 2.5]])
mu_a, mu_b = np.array([-1, -1]), np.array([3, 3])
X_a = np.random.multivariate_normal(mu_a, cov_a, 120)
X_b = np.random.multivariate_normal(mu_b, cov_b, 120)
X_ell = np.vstack([X_a, X_b])
y_true = np.array([0]*120 + [1]*120)

# K-moyennes
hist_ell = run_kmeans(X_ell, 2, seed=0)
mu_f, labels_f = hist_ell[-1]

# GMM
gmm_ell = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm_ell.fit(X_ell)

# Grille pour les frontières de décision
x_min, x_max = -6, 8
y_min, y_max = -5, 8
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

def draw_ellipse_on(ax, mu, cov, color, n_std=2):
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 1e-8))
    ell = Ellipse(mu, w, h, angle=angle, fill=False, color=color,
                  linewidth=2, linestyle='--')
    ax.add_patch(ell)

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
colors_2 = ['steelblue', 'coral']

# --- Gauche: k-moyennes ---
ax = axes[0]

# Régions d'assignation k-moyennes (colorier la grille)
dists_grid = np.linalg.norm(grid[:, None] - mu_f[None], axis=2)
Z_km = np.argmin(dists_grid, axis=1).reshape(xx.shape)
ax.contourf(xx, yy, Z_km, levels=[-0.5, 0.5, 1.5], colors=[colors_2[0], colors_2[1]], alpha=0.08)
ax.contour(xx, yy, Z_km, levels=[0.5], colors='black', linewidths=2.5)

# Points colorés par groupe réel
ax.scatter(X_a[:, 0], X_a[:, 1], c=colors_2[0], alpha=0.5, s=20, edgecolors='none')
ax.scatter(X_b[:, 0], X_b[:, 1], c=colors_2[1], alpha=0.5, s=20, edgecolors='none')

# Centroïdes et segment
ax.plot(mu_f[0, 0], mu_f[0, 1], marker='X', color=colors_2[0],
        markersize=14, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
ax.plot(mu_f[1, 0], mu_f[1, 1], marker='X', color=colors_2[1],
        markersize=14, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
ax.plot([mu_f[0, 0], mu_f[1, 0]], [mu_f[0, 1], mu_f[1, 1]],
        'k--', linewidth=1.5, alpha=0.4)

# Ellipses des vrais groupes
draw_ellipse_on(ax, mu_a, cov_a, colors_2[0])
draw_ellipse_on(ax, mu_b, cov_b, colors_2[1])

# Points mal classés par k-moyennes
misclassified = labels_f != y_true
ax.scatter(X_ell[misclassified, 0], X_ell[misclassified, 1],
           facecolors='none', edgecolors='black', s=60, linewidths=1.2, zorder=4)

n_errors_km = misclassified.sum()
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title(f'K-moyennes: médiatrice ({n_errors_km} erreurs)')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# --- Droite: GMM ---
ax = axes[1]

# Régions d'assignation GMM
Z_gmm = gmm_ell.predict(grid).reshape(xx.shape)
ax.contourf(xx, yy, Z_gmm, levels=[-0.5, 0.5, 1.5], colors=[colors_2[0], colors_2[1]], alpha=0.08)
ax.contour(xx, yy, Z_gmm, levels=[0.5], colors='black', linewidths=2.5)

# Points colorés par groupe réel
ax.scatter(X_a[:, 0], X_a[:, 1], c=colors_2[0], alpha=0.5, s=20, edgecolors='none')
ax.scatter(X_b[:, 0], X_b[:, 1], c=colors_2[1], alpha=0.5, s=20, edgecolors='none')

# Ellipses estimées par le GMM
for k in range(2):
    draw_ellipse_on(ax, gmm_ell.means_[k], gmm_ell.covariances_[k], colors_2[k])

labels_gmm = gmm_ell.predict(X_ell)
misclassified_gmm = labels_gmm != y_true
if misclassified_gmm.sum() > misclassified_gmm.size / 2:
    misclassified_gmm = ~misclassified_gmm
n_errors_gmm = misclassified_gmm.sum()
ax.scatter(X_ell[misclassified_gmm, 0], X_ell[misclassified_gmm, 1],
           facecolors='none', edgecolors='black', s=60, linewidths=1.2, zorder=4)

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title(f'GMM: séparation adaptée ({n_errors_gmm} erreurs)')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
```

Les deux panneaux montrent les mêmes données (colorées selon leur vrai groupe d'origine) avec les ellipses de covariance à 2 écarts-types; les points mal assignés sont cerclés de noir. À gauche, k-moyennes sépare les groupes par la médiatrice du segment reliant les deux centroïdes: cette droite coupe à travers les ellipses et assigne au mauvais groupe les points qui se trouvent du côté «interdit» de la perpendiculaire. À droite, le GMM ajuste une covariance propre à chaque composant; la séparation entre les deux régions d'assignation épouse la forme elliptique des groupes et réduit le nombre d'erreurs.

### Du partitionnement dur au modèle probabiliste

Comment dépasser cette limitation? Au lieu d'assigner chaque point à un seul groupe (0 ou 1), on peut lui attribuer une **probabilité** d'appartenir à chaque groupe. Et au lieu de groupes sphériques, on peut modéliser chaque groupe par une gaussienne avec sa propre matrice de covariance, capable de capturer des formes elliptiques.

C'est exactement ce que fait un **modèle de mélange gaussien** (GMM, *Gaussian Mixture Model*). Il suppose que les données sont générées par un mélange de $K$ distributions gaussiennes:

$$
p(\mathbf{x} \mid \boldsymbol{\theta}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

où $\pi_k$ est le **poids du mélange** pour le composant $k$ (avec $\sum_k \pi_k = 1$ et $\pi_k \geq 0$). Nous pouvons interpréter ce modèle avec une **variable latente** $z \in \{1, \ldots, K\}$ qui indique de quel composant provient chaque observation:

$$
p(z = k) = \pi_k, \qquad p(\mathbf{x} \mid z = k) = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

Le processus de génération est: tirer un composant $z \sim \text{Cat}(\boldsymbol{\pi})$, puis tirer une observation $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_z, \boldsymbol{\Sigma}_z)$. Ce cadre généralise l'analyse discriminante gaussienne au cas non supervisé: la même structure probabiliste s'applique, mais les «classes» sont maintenant inconnues.

### Responsabilités

Pour une observation $\mathbf{x}_n$, la **responsabilité** du composant $k$ est la probabilité a posteriori que cette observation provienne du composant $k$:

$$
r_{nk} = p(z_n = k \mid \mathbf{x}_n) = \frac{\pi_k \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}
$$

Les responsabilités sont des valeurs continues dans $[0, 1]$ qui somment à 1 pour chaque observation. Un point situé exactement entre deux composants aura des responsabilités proches de $0{,}5$ pour chacun, exprimant l'ambiguïté de son appartenance. C'est un **partitionnement souple** (*soft clustering*), en contraste avec les assignations binaires de k-moyennes.

Le lien entre les deux est direct. Supposons que tous les composants partagent une covariance sphérique $\boldsymbol{\Sigma}_k = \sigma^2 \mathbf{I}$ et des poids uniformes $\pi_k = 1/K$. Les facteurs de normalisation $(2\pi\sigma^2)^{-D/2}$ et les poids $1/K$ sont identiques pour tous les composants et s'annulent dans la fraction. Il ne reste que les exponentielles des distances, et les responsabilités prennent la forme d'un **softmax**, la même transformation que nous avions vue en régression logistique pour convertir des scores arbitraires en probabilités valides ($\geq 0$, sommant à 1):

$$
r_{nk} = \frac{\exp(-\|\mathbf{x}_n - \boldsymbol{\mu}_k\|^2 / 2\sigma^2)}{\sum_{k'} \exp(-\|\mathbf{x}_n - \boldsymbol{\mu}_{k'}\|^2 / 2\sigma^2)}
$$

Quand $\sigma^2$ est grand, les gaussiennes sont très étalées et les responsabilités sont proches de $1/K$ partout. Quand $\sigma^2$ diminue, l'exponentielle associée au centroïde le plus proche domine de plus en plus. À la limite $\sigma^2 \to 0$, les responsabilités deviennent binaires et on retrouve exactement l'assignation de k-moyennes. Passer de k-moyennes à un GMM revient donc à relâcher l'hypothèse de groupes sphériques et à remplacer les assignations dures par des probabilités d'appartenance.

```{code-cell} python
:tags: [hide-input]

# Comparaison: partitionnement dur (k-moyennes) vs souple (GMM)
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_kmeans)
responsibilities = gmm.predict_proba(X_kmeans)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Gauche: k-moyennes (assignations dures)
ax = axes[0]
mu_final, labels_km = hist_km[-1]
colors_km_list = ['steelblue', 'coral', 'seagreen']
for k in range(3):
    mask = labels_km == k
    ax.scatter(X_kmeans[mask, 0], X_kmeans[mask, 1], c=colors_km_list[k], alpha=0.6, s=20, label=f'Groupe {k+1}')
    ax.plot(mu_final[k, 0], mu_final[k, 1], marker='X', color=colors_km_list[k],
            markersize=12, markeredgecolor='black', markeredgewidth=1.5, zorder=5)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('K-moyennes (assignation dure)')
ax.legend()
ax.set_xlim(-3, 7)
ax.set_ylim(-3, 6)

# Droite: GMM (responsabilités souples + ellipses)
ax = axes[1]
rgb = responsibilities @ np.array([[0.27, 0.51, 0.71],
                                    [1.0, 0.5, 0.31],
                                    [0.18, 0.55, 0.34]])
ax.scatter(X_kmeans[:, 0], X_kmeans[:, 1], c=rgb, alpha=0.6, s=20)

from matplotlib.patches import Ellipse
for k in range(3):
    mean = gmm.means_[k]
    cov = gmm.covariances_[k]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    for n_std in [1, 2]:
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        ellipse = Ellipse(mean, width, height, angle=angle, fill=False,
                         color=colors_km_list[k], linewidth=2, linestyle='--')
        ax.add_patch(ellipse)

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('GMM (responsabilités souples, ellipses de covariance)')
ax.set_xlim(-3, 7)
ax.set_ylim(-3, 6)

plt.tight_layout()
```

La figure met en regard les deux approches sur les mêmes données. À gauche, k-moyennes assigne chaque point à un seul groupe; les frontières sont rectilignes. À droite, le GMM exprime l'incertitude par un dégradé de couleurs et capture la forme elliptique de chaque composant grâce aux matrices de covariance.

Ce parallèle entre k-moyennes et GMM est aussi la clé pour comprendre l'algorithme EM: la même stratégie d'alternance (assigner les points, puis mettre à jour les paramètres) s'applique aux deux, mais avec des assignations souples au lieu de dures.

## L'algorithme EM

### Le problème d'estimation

K-moyennes alternait entre assigner les points et recalculer les centroïdes, et chaque étape avait une solution simple. Peut-on faire la même chose pour un GMM? La difficulté vient de la log-vraisemblance:

$$
\log p(\mathbf{X} \mid \boldsymbol{\theta}) = \sum_{n=1}^N \log \left( \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right)
$$

La somme à l'intérieur du logarithme empêche d'isoler la contribution de chaque composant. Il n'y a pas de solution analytique comme pour le naïf bayésien ou LDA.

Si nous connaissions les assignations $z_n$ de chaque point, le problème serait simple: nous estimerions séparément les paramètres de chaque composant à partir des points qui lui sont assignés, comme le fait k-moyennes. Mais les $z_n$ sont inconnus: ce sont des variables latentes. L'algorithme **Espérance-Maximisation** (EM) résout ce dilemme en reprenant la stratégie d'alternance de k-moyennes, mais avec des assignations souples.

### L'intuition: k-moyennes avec des responsabilités

Dans k-moyennes, chaque itération fait deux choses: assigner les points (étape d'assignation), puis recalculer les centroïdes (étape de mise à jour). EM fait exactement la même chose, mais au lieu d'assigner chaque point à un seul groupe, il calcule des **responsabilités**, c'est-à-dire la probabilité que chaque point appartienne à chaque composant. Les mises à jour des paramètres deviennent alors des moyennes pondérées par ces responsabilités, plutôt que des moyennes simples sur les points assignés.

```{margin} Pourquoi EM converge-t-il?
EM maximise itérativement une borne inférieure de la log-vraisemblance (l'ELBO, défini dans la section « Inférence variationnelle et EM »). À chaque itération, cette borne augmente ou reste stable, ce qui garantit la convergence vers un maximum local.
```

### Les étapes de l'algorithme

**Étape E (Espérance).** Fixer les paramètres $\boldsymbol{\theta}^{(t)}$ et calculer les responsabilités:

$$
r_{nk}^{(t)} = \frac{\pi_k^{(t)} \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k^{(t)}, \boldsymbol{\Sigma}_k^{(t)})}{\sum_{j=1}^K \pi_j^{(t)} \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j^{(t)}, \boldsymbol{\Sigma}_j^{(t)})}
$$

**Étape M (Maximisation).** Fixer les responsabilités et mettre à jour les paramètres. Définissons $N_k = \sum_{n=1}^N r_{nk}$ le «nombre effectif» de points dans le composant $k$.

*Poids du mélange:*
$$
\pi_k^{(t+1)} = \frac{N_k^{(t)}}{N}
$$

*Moyennes:*
$$
\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{N_k^{(t)}} \sum_{n=1}^N r_{nk}^{(t)} \mathbf{x}_n
$$

*Covariances:*
$$
\boldsymbol{\Sigma}_k^{(t+1)} = \frac{1}{N_k^{(t)}} \sum_{n=1}^N r_{nk}^{(t)} (\mathbf{x}_n - \boldsymbol{\mu}_k^{(t+1)})(\mathbf{x}_n - \boldsymbol{\mu}_k^{(t+1)})^\top
$$

Ces formules sont des versions pondérées des estimateurs classiques. Au lieu de compter chaque point une fois, nous le pondérons par sa responsabilité envers le composant.

### Pseudocode

```
Entrée: Données X, nombre de composants K
1. Initialiser les paramètres θ = (π, μ, Σ)
2. Répéter jusqu'à convergence:
   a. Étape E: calculer les responsabilités r_nk
   b. Étape M: mettre à jour π, μ, Σ
3. Calculer la log-vraisemblance et vérifier la convergence
Sortie: Paramètres θ et responsabilités r
```

### Visualisation de la convergence

```{code-cell} python
:tags: [hide-input]

# Animation de l'algorithme EM sur un GMM
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

np.random.seed(42)

# Données
n_samples = 300
true_means = [np.array([0, 0]), np.array([4, 0]), np.array([2, 3])]
true_covs = [np.array([[1, 0], [0, 1]]), 
             np.array([[0.5, 0.3], [0.3, 0.5]]), 
             np.array([[0.8, -0.4], [-0.4, 0.8]])]
X_em = np.vstack([np.random.multivariate_normal(m, c, n_samples // 3) 
                  for m, c in zip(true_means, true_covs)])

# Fonction pour calculer la densité gaussienne
def gaussian_pdf(x, mean, cov):
    d = len(mean)
    diff = x - mean
    return np.exp(-0.5 * diff @ np.linalg.inv(cov) @ diff) / np.sqrt((2*np.pi)**d * np.linalg.det(cov))

# Initialisation (mauvaise, pour montrer la convergence)
K = 3
np.random.seed(123)
means = [np.random.randn(2) * 2 for _ in range(K)]
covs = [np.eye(2) * 2 for _ in range(K)]
weights = np.ones(K) / K

# Stocker l'historique
history = {'means': [means.copy()], 'covs': [covs.copy()], 'weights': [weights.copy()]}

# Exécuter EM
for iteration in range(15):
    # Étape E
    responsibilities = np.zeros((len(X_em), K))
    for n, x in enumerate(X_em):
        for k in range(K):
            responsibilities[n, k] = weights[k] * gaussian_pdf(x, means[k], covs[k])
        responsibilities[n] /= responsibilities[n].sum()
    
    # Étape M
    N_k = responsibilities.sum(axis=0)
    weights = N_k / len(X_em)
    for k in range(K):
        means[k] = (responsibilities[:, k:k+1] * X_em).sum(axis=0) / N_k[k]
        diff = X_em - means[k]
        covs[k] = (responsibilities[:, k:k+1] * diff).T @ diff / N_k[k]
    
    history['means'].append([m.copy() for m in means])
    history['covs'].append([c.copy() for c in covs])
    history['weights'].append(weights.copy())

# Créer la figure
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['steelblue', 'coral', 'seagreen']

def draw_ellipse(ax, mean, cov, color, alpha=0.3):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    for n_std in [1, 2]:
        width, height = 2 * n_std * np.sqrt(np.maximum(eigenvalues, 1e-6))
        ellipse = Ellipse(mean, width, height, angle=angle, fill=True, 
                         facecolor=color, alpha=alpha*0.5, edgecolor=color, linewidth=2)
        ax.add_patch(ellipse)

def animate(frame):
    ax.clear()
    ax.scatter(X_em[:, 0], X_em[:, 1], c='gray', alpha=0.3, s=10)
    
    for k in range(K):
        mean = history['means'][frame][k]
        cov = history['covs'][frame][k]
        draw_ellipse(ax, mean, cov, colors[k])
        ax.plot(mean[0], mean[1], 'o', color=colors[k], markersize=10, markeredgecolor='black')
    
    ax.set_xlim(-4, 7)
    ax.set_ylim(-4, 6)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(f'Algorithme EM - Itération {frame}')
    return []

anim = FuncAnimation(fig, animate, frames=len(history['means']), interval=500, blit=True)
anim.save('_static/em_convergence.gif', writer='pillow', fps=2, dpi=100)
plt.close()

from IPython.display import Image
Image(filename='_static/em_convergence.gif')
```

L'animation montre la convergence de l'algorithme EM. Les ellipses représentent les composants gaussiens (contours à 1 et 2 écarts-types), et les points colorés sont les moyennes. À partir d'une initialisation arbitraire, l'algorithme ajuste progressivement les paramètres pour mieux couvrir les données.

### Considérations pratiques

**Initialisation.** EM converge vers un maximum local, et le résultat dépend de l'initialisation. Stratégies courantes:
- Exécuter EM plusieurs fois avec des initialisations aléatoires différentes
- Initialiser avec k-moyennes (rapide et donne souvent un bon point de départ)
- Utiliser k-means++ pour une initialisation plus robuste

**Choix de $K$.** Le nombre de composants est un hyperparamètre. Des critères comme le BIC (*Bayesian Information Criterion*) ou l'AIC (*Akaike Information Criterion*) pénalisent la complexité du modèle et peuvent guider ce choix.

**Singularités.** Si un composant contient un seul point, sa covariance estimée peut être singulière. Solutions:
- Ajouter une régularisation diagonale: $\boldsymbol{\Sigma}_k \leftarrow \boldsymbol{\Sigma}_k + \epsilon \mathbf{I}$
- Utiliser des covariances contraintes (diagonales ou partagées)
- Réinitialiser les composants problématiques

## Inférence variationnelle et EM

Jusqu'ici, nous avons présenté EM comme une recette: calculer les responsabilités, mettre à jour les paramètres, répéter. Mais pourquoi cette alternance converge-t-elle? Et y a-t-il un objectif que chaque itération améliore? Cette section répond à ces questions en construisant une borne inférieure de la log-vraisemblance, puis en montrant qu'EM la maximise par alternance. La dérivation qui suit est un peu plus formelle que le reste du chapitre. Prenons le temps de la parcourir étape par étape; si certains passages semblent abstraits en première lecture, le point à retenir est résumé à la fin de la section.

### Le problème: la somme dans le logarithme

Revenons à notre GMM. La log-vraisemblance que nous voulons maximiser est:

$$
\log p(\mathbf{X} \mid \boldsymbol{\theta}) = \sum_{n=1}^N \log \left( \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right).
$$

La somme sur les composants $k$ se trouve à l'intérieur du logarithme. Si nous connaissions l'assignation $z_n$ de chaque point, le log passerait directement sur chaque gaussienne et le problème serait séparable. Comme les $z_n$ sont inconnus, il nous faut un moyen de « pousser » le logarithme à travers cette somme. L'idée est de construire un objectif auxiliaire, une borne inférieure de $\log p(\mathbf{X} \mid \boldsymbol{\theta})$, que l'on peut optimiser même sans connaître les $z_n$.

### Construire une borne inférieure

Introduisons une distribution auxiliaire $q(\mathbf{Z})$ sur les variables latentes (les assignations de tous les points). Cette distribution est un outil de calcul: nous sommes libres de la choisir comme nous voulons. Partons de la règle de Bayes, qui relie la conjointe et l'a posteriori:

$$
p(\mathbf{X} \mid \boldsymbol{\theta}) = \frac{p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})}.
$$

Cette égalité est vraie pour tout $\mathbf{Z}$. En prenant le logarithme des deux côtés:

$$
\log p(\mathbf{X} \mid \boldsymbol{\theta}) = \log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta}) - \log p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}).
$$

Le membre de gauche ne dépend pas de $\mathbf{Z}$: c'est une constante par rapport aux latentes. Si nous prenons la moyenne de cette égalité sous notre distribution auxiliaire $q(\mathbf{Z})$, le membre de gauche ne change pas, et nous obtenons:

$$
\log p(\mathbf{X} \mid \boldsymbol{\theta}) = \mathbb{E}_{q(\mathbf{Z})}\!\big[ \log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta}) \big] - \mathbb{E}_{q(\mathbf{Z})}\!\big[ \log p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}) \big].
$$

Maintenant, ajoutons et retranchons $\mathbb{E}_q[\log q(\mathbf{Z})]$ dans le membre de droite. Cela revient à réarranger les termes, sans rien changer à l'égalité. En regroupant:

$$
\log p(\mathbf{X} \mid \boldsymbol{\theta}) = \underbrace{\mathbb{E}_{q(\mathbf{Z})}\!\big[ \log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta}) - \log q(\mathbf{Z}) \big]}_{\text{ELBO}(q, \boldsymbol{\theta})} \;+\; \underbrace{\mathbb{E}_{q(\mathbf{Z})}\!\left[ \log \frac{q(\mathbf{Z})}{p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})} \right]}_{D_{\mathrm{KL}}(q \,\|\, p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}))}.
$$

Le second terme est la divergence de Kullback-Leibler entre $q$ et l'a posteriori exact. Cette divergence mesure à quel point $q$ diffère de la vraie distribution des latentes, et elle est toujours $\geq 0$ (elle vaut zéro uniquement quand $q$ coïncide exactement avec l'a posteriori).

Puisque la log-vraisemblance est la somme de ces deux termes et que la KL est positive, le premier terme est nécessairement plus petit (ou égal) à la log-vraisemblance:

$$
\log p(\mathbf{X} \mid \boldsymbol{\theta}) \;\geq\; \mathrm{ELBO}(q, \boldsymbol{\theta}).
$$

Ce premier terme porte le nom de **borne inférieure de l'évidence** (en anglais *Evidence Lower Bound*, d'où l'acronyme ELBO). L'« évidence » désigne ici $p(\mathbf{X} \mid \boldsymbol{\theta})$, la vraisemblance marginale des données. L'ELBO est une borne inférieure de cette quantité, et la borne est serrée (l'égalité est atteinte) quand $q(\mathbf{Z}) = p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})$.

```{admonition} Si cette dérivation vous semble dense
:class: tip

C'est normal. Beaucoup d'étudiants ont besoin de la relire deux ou trois fois. L'essentiel à retenir est le résultat: pour toute distribution auxiliaire $q$, l'ELBO est un objectif que l'on peut calculer et qui ne dépasse jamais la vraie log-vraisemblance. Plus $q$ ressemble à l'a posteriori, plus la borne est serrée.
```

### EM maximise l'ELBO par alternance

Nous avons maintenant un objectif, l'ELBO, qui dépend de deux choses: la distribution auxiliaire $q$ et les paramètres $\boldsymbol{\theta}$. Comment le maximiser?

**Optimiser $q$ à $\boldsymbol{\theta}$ fixé.** Si les paramètres du modèle sont fixés, quel $q$ donne le meilleur ELBO? L'égalité $\log p(\mathbf{X} \mid \boldsymbol{\theta}) = \mathrm{ELBO}(q, \boldsymbol{\theta}) + D_{\mathrm{KL}}(q \| p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}))$ nous le dit: puisque la log-vraisemblance ne dépend pas de $q$, augmenter l'ELBO revient exactement à diminuer la KL. Le maximum est atteint quand la KL vaut zéro, c'est-à-dire quand $q(\mathbf{Z}) = p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})$. Pour le GMM, cet a posteriori correspond aux responsabilités $r_{nk}$ que nous calculons à l'étape E.

**Optimiser $\boldsymbol{\theta}$ à $q$ fixé.** Maintenant que $q$ est fixée, l'ELBO se simplifie: $\mathbb{E}_q[\log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})]$ moins une constante (le terme $\mathbb{E}_q[\log q]$ ne dépend pas de $\boldsymbol{\theta}$). Maximiser l'ELBO en $\boldsymbol{\theta}$ revient à maximiser la log-vraisemblance conjointe pondérée par les responsabilités. Ce sont exactement les formules de l'étape M (moyennes pondérées, covariances pondérées, etc.).

L'alternance E/M est donc une maximisation par coordonnées de l'ELBO: l'étape E ajuste $q$ pour serrer la borne au maximum, puis l'étape M ajuste $\boldsymbol{\theta}$ pour pousser cette borne vers le haut. À chaque demi-pas, l'ELBO augmente (ou reste stable), et comme il reste toujours inférieur à la log-vraisemblance, la log-vraisemblance elle-même ne peut pas diminuer. Cela garantit la convergence vers un maximum local.

En résumé: EM n'est pas une heuristique ad hoc, mais une procédure d'optimisation bien fondée. Chaque itération améliore un objectif précis (l'ELBO), et la convergence découle de cette monotonie.

### L'inférence variationnelle: au-delà d'EM

Le raisonnement que nous venons de mener, maximiser l'ELBO par rapport à une distribution auxiliaire $q$ et des paramètres $\boldsymbol{\theta}$, porte un nom: l'**inférence variationnelle**. EM en est le cas le plus favorable: l'a posteriori $p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})$ est calculable (les responsabilités du GMM ont une formule fermée), et nous pouvons choisir $q$ exactement égal à cet a posteriori.

```{admonition} Quand l'a posteriori est intractable (optionnel IFT3395)
:class: note

Dans beaucoup de modèles plus complexes (modèles de thèmes, autoencodeurs variationnels, etc.), l'a posteriori $p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})$ n'a pas de formule fermée. On ne peut donc pas poser $q = p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})$ comme en EM. La solution est de restreindre $q$ à une **famille variationnelle** $\mathcal{Q}$ de distributions plus simples (par exemple des gaussiennes factorisées, ou des distributions paramétrées par un réseau de neurones), puis de maximiser l'ELBO par descente de gradient sur les paramètres de $q$. La borne ne sera plus serrée, mais elle restera un objectif utile. Ces extensions sont abordées dans des cours dédiés aux modèles génératifs profonds.
```

## Mélange d'experts

### Du partitionnement à la prédiction

Un GMM partitionne l'espace des observations en groupes, mais il ne fait pas de prédiction. Pourtant, l'idée de «diviser un problème entre plusieurs spécialistes» s'applique naturellement à la régression et à la classification. Considérons des données où la relation entre l'entrée $x$ et la sortie $y$ change selon la région: par exemple, un système physique qui se comporte différemment à basse et haute température, ou un marché dont la dynamique varie selon la conjoncture. Un seul modèle linéaire ne peut pas capturer ces régimes. Mais si l'on dispose de plusieurs modèles linéaires (un par régime) et d'un mécanisme pour aiguiller chaque observation vers le bon modèle, on obtient une prédiction flexible à partir de composants simples.

```{code-cell} python
:tags: [hide-input]

# Données de régression par morceaux: deux régimes linéaires
np.random.seed(42)
n_moe = 200
x_moe = np.random.uniform(-3, 3, n_moe)
noise_moe = np.random.normal(0, 0.3, n_moe)
y_moe = np.where(x_moe < 0, 2 * x_moe + 1, -1.5 * x_moe + 2) + noise_moe

# Régression linéaire unique
X_moe = np.column_stack([np.ones(n_moe), x_moe])
w_lin = np.linalg.lstsq(X_moe, y_moe, rcond=None)[0]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.scatter(x_moe, y_moe, c='gray', alpha=0.5, s=20)
x_grid_moe = np.linspace(-3, 3, 200)
ax.plot(x_grid_moe, w_lin[0] + w_lin[1] * x_grid_moe, 'k-', linewidth=2, label='Régression linéaire')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Un seul modèle linéaire')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
mask_neg = x_moe < 0
mask_pos = ~mask_neg
ax.scatter(x_moe[mask_neg], y_moe[mask_neg], c='steelblue', alpha=0.5, s=20, label='Régime 1')
ax.scatter(x_moe[mask_pos], y_moe[mask_pos], c='coral', alpha=0.5, s=20, label='Régime 2')
ax.plot(x_grid_moe[x_grid_moe < 0], 2 * x_grid_moe[x_grid_moe < 0] + 1,
        'steelblue', linewidth=2, label='Expert 1')
ax.plot(x_grid_moe[x_grid_moe >= 0], -1.5 * x_grid_moe[x_grid_moe >= 0] + 2,
        'coral', linewidth=2, label='Expert 2')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Deux experts, un par régime')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
```

La figure illustre le problème. À gauche, une seule droite de régression ne capture ni la pente ascendante pour $x < 0$ ni la pente descendante pour $x > 0$: elle passe au milieu sans bien prédire aucune des deux régions. À droite, deux experts linéaires se partagent le travail et chacun s'ajuste à son régime. Le **mélange d'experts** (*Mixture of Experts*, MoE) formalise cette idée.

### Le modèle

Un mélange d'experts modélise la distribution conditionnelle $p(y \mid \mathbf{x})$ comme un mélange de $K$ modèles simples, les **experts**, dont les poids dépendent de l'entrée:

$$
p(y \mid \mathbf{x}) = \sum_{k=1}^K g_k(\mathbf{x}) \, p(y \mid \mathbf{x}, z = k)
$$

La structure rappelle le GMM, mais avec deux différences. D'abord, c'est un modèle **supervisé**: on prédit $y$ à partir de $\mathbf{x}$, au lieu de modéliser la densité de $\mathbf{x}$ seul. Ensuite, les poids du mélange ne sont plus des constantes $\pi_k$: ils sont calculés par un **réseau de routage** (ou *gating network*) $g_k(\mathbf{x})$ qui dépend de l'entrée.

Le réseau de routage utilise un softmax sur des scores linéaires:

$$
g_k(\mathbf{x}) = \frac{\exp(\mathbf{v}_k^\top \mathbf{x})}{\sum_{j=1}^K \exp(\mathbf{v}_j^\top \mathbf{x})}
$$

où les $\mathbf{v}_k$ sont des paramètres appris. Les sorties $g_k(\mathbf{x})$ sont positives et somment à 1: elles représentent la probabilité que l'expert $k$ soit responsable de l'observation $(\mathbf{x}, y)$. Si $\mathbf{v}_1^\top \mathbf{x}$ est beaucoup plus grand que les autres scores, le routage attribue presque tout le poids à l'expert 1.

Chaque expert $k$ est un modèle de régression linéaire gaussien:

$$
p(y \mid \mathbf{x}, z = k) = \mathcal{N}(y \mid \mathbf{w}_k^\top \mathbf{x}, \sigma_k^2)
$$

L'expert $k$ prédit $\hat{y}_k = \mathbf{w}_k^\top \mathbf{x}$ avec une incertitude $\sigma_k^2$. La prédiction globale du MoE est la moyenne pondérée des prédictions de chaque expert:

$$
\hat{y} = \sum_{k=1}^K g_k(\mathbf{x}) \, \mathbf{w}_k^\top \mathbf{x}
$$

### EM pour le mélange d'experts

L'estimation des paramètres suit la même logique EM que pour le GMM. La variable latente $z_n \in \{1, \ldots, K\}$ indique quel expert est responsable de l'observation $n$.

**Étape E.** On calcule les responsabilités, soit la probabilité a posteriori que l'expert $k$ soit responsable de $(\mathbf{x}_n, y_n)$:

$$
r_{nk} = \frac{g_k(\mathbf{x}_n) \, \mathcal{N}(y_n \mid \mathbf{w}_k^\top \mathbf{x}_n, \sigma_k^2)}{\sum_{j=1}^K g_j(\mathbf{x}_n) \, \mathcal{N}(y_n \mid \mathbf{w}_j^\top \mathbf{x}_n, \sigma_j^2)}
$$

La structure est identique à celle du GMM: le numérateur est le produit du poids (ici $g_k(\mathbf{x}_n)$ au lieu de $\pi_k$) par la vraisemblance du point sous le composant $k$. La différence est que la vraisemblance porte sur $y_n$ conditionné à $\mathbf{x}_n$, et que les poids dépendent de $\mathbf{x}_n$.

**Étape M.** On met à jour les paramètres en deux temps.

Pour les **experts**, la mise à jour de $\mathbf{w}_k$ est une régression linéaire pondérée par les responsabilités. Au lieu de minimiser $\sum_n (y_n - \mathbf{w}_k^\top \mathbf{x}_n)^2$ comme en régression ordinaire, on minimise $\sum_n r_{nk} (y_n - \mathbf{w}_k^\top \mathbf{x}_n)^2$: chaque point contribue proportionnellement à la responsabilité de l'expert $k$ pour ce point. La solution en forme fermée est:

$$
\mathbf{w}_k = (\mathbf{X}^\top \mathbf{R}_k \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{R}_k \mathbf{y}
$$

où $\mathbf{R}_k = \text{diag}(r_{1k}, \ldots, r_{Nk})$. La variance se met à jour de manière analogue:

$$
\sigma_k^2 = \frac{\sum_n r_{nk} (y_n - \mathbf{w}_k^\top \mathbf{x}_n)^2}{\sum_n r_{nk}}
$$

Pour le **routage**, on maximise $\sum_n \sum_k r_{nk} \log g_k(\mathbf{x}_n)$ par rapport aux paramètres $\mathbf{v}_k$. Contrairement aux experts, il n'y a pas de formule fermée pour le softmax; on utilise quelques itérations de montée de gradient avec le gradient:

$$
\nabla_{\mathbf{v}_k} = \sum_{n=1}^N (r_{nk} - g_k(\mathbf{x}_n)) \, \mathbf{x}_n
$$

Ce gradient a une interprétation simple: si la responsabilité $r_{nk}$ est plus grande que le poids actuel $g_k(\mathbf{x}_n)$, le gradient pousse le routage à donner plus de poids à l'expert $k$ pour les entrées proches de $\mathbf{x}_n$.

```{code-cell} python
:tags: [hide-input]

# Implémentation complète du MoE et visualisation
def softmax_stable(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)

def moe_fit(X, y, K=2, n_iter=40, seed=42):
    """Ajuste un MoE par EM."""
    N, D = X.shape
    rng = np.random.RandomState(seed)

    # Initialisation
    V = rng.randn(K, D) * 0.1
    w_list = [np.linalg.lstsq(X, y, rcond=None)[0] + rng.randn(D) * 0.3 for _ in range(K)]
    sig2 = np.ones(K) * 0.5
    ll_hist = []

    for _ in range(n_iter):
        # Étape E
        g = softmax_stable(X @ V.T)
        weighted = np.zeros((N, K))
        for k in range(K):
            mu_k = X @ w_list[k]
            weighted[:, k] = g[:, k] * np.exp(-0.5 * (y - mu_k)**2 / sig2[k]) / np.sqrt(2 * np.pi * sig2[k])
        r = weighted / (weighted.sum(axis=1, keepdims=True) + 1e-300)

        # Étape M : experts (régression pondérée)
        for k in range(K):
            rk = r[:, k]
            Xr = np.sqrt(rk)[:, None] * X
            yr = np.sqrt(rk) * y
            w_list[k] = np.linalg.lstsq(Xr, yr, rcond=None)[0]
            pred_k = X @ w_list[k]
            sig2[k] = np.sum(rk * (y - pred_k)**2) / (rk.sum() + 1e-10) + 1e-6

        # Étape M : routage (montée de gradient)
        for _ in range(30):
            g = softmax_stable(X @ V.T)
            for k in range(K):
                V[k] += 0.05 * X.T @ (r[:, k] - g[:, k])

        # Log-vraisemblance
        g = softmax_stable(X @ V.T)
        ll = 0
        for k in range(K):
            mu_k = X @ w_list[k]
            ll += g[:, k] * np.exp(-0.5 * (y - mu_k)**2 / sig2[k]) / np.sqrt(2 * np.pi * sig2[k])
        ll_hist.append(np.sum(np.log(ll + 1e-300)))

    return V, w_list, sig2, r, ll_hist

V_fit, w_fit, sig_fit, r_fit, ll_fit = moe_fit(X_moe, y_moe, K=2, n_iter=40)

# Grille de prédiction
x_g = np.linspace(-3.2, 3.2, 300)
X_g = np.column_stack([np.ones(300), x_g])
g_g = softmax_stable(X_g @ V_fit.T)
mu_experts = np.column_stack([X_g @ w_fit[k] for k in range(2)])
y_pred_moe = (g_g * mu_experts).sum(axis=1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel 1: experts et prédiction globale
ax = axes[0]
ax.scatter(x_moe, y_moe, c='gray', alpha=0.4, s=15)
ax.plot(x_g, mu_experts[:, 0], color='steelblue', linewidth=2, alpha=0.7, label='Expert 1')
ax.plot(x_g, mu_experts[:, 1], color='coral', linewidth=2, alpha=0.7, label='Expert 2')
ax.plot(x_g, y_pred_moe, 'k-', linewidth=2.5, label='Prédiction MoE')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Prédictions des experts')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: réseau de routage
ax = axes[1]
ax.plot(x_g, g_g[:, 0], color='steelblue', linewidth=2.5, label='$g_1(x)$ (expert 1)')
ax.plot(x_g, g_g[:, 1], color='coral', linewidth=2.5, label='$g_2(x)$ (expert 2)')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('$x$')
ax.set_ylabel('Poids du routage')
ax.set_title('Réseau de routage')
ax.legend(fontsize=9)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

# Panel 3: points colorés par responsabilité
ax = axes[2]
rgb_moe = r_fit @ np.array([[0.27, 0.51, 0.71], [0.99, 0.50, 0.31]])
ax.scatter(x_moe, y_moe, c=rgb_moe, alpha=0.7, s=25)
ax.plot(x_g, y_pred_moe, 'k-', linewidth=2)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Responsabilités (couleur = assignation souple)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
```

Le panneau de gauche montre les deux experts (droites colorées) et la prédiction globale du MoE (courbe noire), qui passe d'un expert à l'autre dans la zone de transition. Le panneau central montre le réseau de routage: $g_1(x)$ domine pour $x < 0$ et $g_2(x)$ domine pour $x > 0$, avec une transition sigmoïdale autour de zéro. Le panneau de droite colore chaque point selon ses responsabilités: les points bleus sont gérés par l'expert 1, les points orangés par l'expert 2, et les points dans la zone de transition ont une couleur intermédiaire.

L'animation suivante montre comment EM fait converger le MoE. Au départ, les deux experts sont mal orientés et le routage est presque uniforme. Au fil des itérations, chaque expert se spécialise sur son régime et le routage apprend à les séparer.

```{code-cell} python
:tags: [hide-input]

# Animation de la convergence EM pour le MoE
from matplotlib.animation import FuncAnimation
from IPython.display import Image as IPImage

def moe_fit_history(X, y, K=2, n_iter=40, seed=42):
    """Ajuste un MoE par EM et stocke l'historique de chaque itération."""
    N, D = X.shape
    rng = np.random.RandomState(seed)
    V = rng.randn(K, D) * 0.1
    w_list = [np.linalg.lstsq(X, y, rcond=None)[0] + rng.randn(D) * 0.3 for _ in range(K)]
    sig2 = np.ones(K) * 0.5
    history = []
    # Stocker état initial
    history.append({'V': V.copy(), 'w': [w.copy() for w in w_list],
                    'sig2': sig2.copy(), 'r': np.ones((N, K)) / K})
    for _ in range(n_iter):
        g = softmax_stable(X @ V.T)
        weighted = np.zeros((N, K))
        for k in range(K):
            mu_k = X @ w_list[k]
            weighted[:, k] = g[:, k] * np.exp(-0.5 * (y - mu_k)**2 / sig2[k]) / np.sqrt(2 * np.pi * sig2[k])
        r = weighted / (weighted.sum(axis=1, keepdims=True) + 1e-300)
        for k in range(K):
            rk = r[:, k]
            Xr = np.sqrt(rk)[:, None] * X
            yr = np.sqrt(rk) * y
            w_list[k] = np.linalg.lstsq(Xr, yr, rcond=None)[0]
            pred_k = X @ w_list[k]
            sig2[k] = np.sum(rk * (y - pred_k)**2) / (rk.sum() + 1e-10) + 1e-6
        for _ in range(30):
            g = softmax_stable(X @ V.T)
            for k in range(K):
                V[k] += 0.05 * X.T @ (r[:, k] - g[:, k])
        history.append({'V': V.copy(), 'w': [w.copy() for w in w_list],
                        'sig2': sig2.copy(), 'r': r.copy()})
    return history

hist_moe = moe_fit_history(X_moe, y_moe, K=2, n_iter=25)

x_anim = np.linspace(-3.2, 3.2, 200)
X_anim = np.column_stack([np.ones(200), x_anim])
colors_ex = ['steelblue', 'coral']
rgb_basis = np.array([[0.27, 0.51, 0.71], [0.99, 0.50, 0.31]])

fig_moe, axes_moe = plt.subplots(1, 3, figsize=(15, 4.5))

def animate_moe(frame):
    state = hist_moe[frame]
    V_t, w_t, sig_t, r_t = state['V'], state['w'], state['sig2'], state['r']
    g_t = softmax_stable(X_anim @ V_t.T)
    mu_t = np.column_stack([X_anim @ w_t[k] for k in range(2)])
    y_pred_t = (g_t * mu_t).sum(axis=1)

    # Panel 1: experts + prédiction
    ax = axes_moe[0]
    ax.clear()
    ax.scatter(x_moe, y_moe, c='gray', alpha=0.3, s=12)
    ax.plot(x_anim, mu_t[:, 0], color=colors_ex[0], linewidth=2, alpha=0.7, label='Expert 1')
    ax.plot(x_anim, mu_t[:, 1], color=colors_ex[1], linewidth=2, alpha=0.7, label='Expert 2')
    ax.plot(x_anim, y_pred_t, 'k-', linewidth=2.5, label='Prédiction MoE')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'Experts, itération {frame}')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-7, 6)
    ax.grid(True, alpha=0.3)

    # Panel 2: routage
    ax = axes_moe[1]
    ax.clear()
    ax.plot(x_anim, g_t[:, 0], color=colors_ex[0], linewidth=2.5, label='$g_1(x)$')
    ax.plot(x_anim, g_t[:, 1], color=colors_ex[1], linewidth=2.5, label='$g_2(x)$')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('Poids du routage')
    ax.set_title(f'Routage, itération {frame}')
    ax.legend(fontsize=8)
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Panel 3: responsabilités
    ax = axes_moe[2]
    ax.clear()
    pt_colors = r_t @ rgb_basis
    ax.scatter(x_moe, y_moe, c=pt_colors, alpha=0.7, s=20)
    ax.plot(x_anim, y_pred_t, 'k-', linewidth=2)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'Responsabilités, itération {frame}')
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-7, 6)
    ax.grid(True, alpha=0.3)
    return []

anim_moe = FuncAnimation(fig_moe, animate_moe,
                          frames=len(hist_moe), interval=600, blit=True)
anim_moe.save('_static/moe_convergence.gif', writer='pillow', fps=2, dpi=100)
plt.close()

IPImage(filename='_static/moe_convergence.gif')
```

Au départ, les deux experts ont des pentes proches et le routage est presque plat à $0{,}5$: le modèle ne distingue pas les deux régimes. Itération après itération, un expert adopte la pente positive (régime $x < 0$) et l'autre la pente négative (régime $x > 0$), pendant que le réseau de routage apprend une transition sigmoïdale autour de $x = 0$. La couleur des points passe progressivement de gris mélangé à bleu franc ou orange franc, reflétant la spécialisation croissante.

### Pourquoi le routage dépendant de l'entrée est nécessaire

La différence entre un GMM et un MoE tient à un seul changement: les poids du mélange. Dans un GMM, les poids $\pi_k$ sont des constantes. Dans un MoE, les poids $g_k(\mathbf{x})$ dépendent de l'entrée. Ce changement a des conséquences profondes.

Avec des poids constants, le modèle ne peut pas aiguiller une entrée vers un expert plutôt qu'un autre. Chaque expert contribue toujours dans les mêmes proportions, quel que soit $\mathbf{x}$. Pour des données de régression par morceaux comme celles de notre exemple, cela empêche la spécialisation: les deux experts tentent de s'ajuster à l'ensemble des données, au lieu de se partager le travail.

```{code-cell} python
:tags: [hide-input]

# Comparaison: MoE (routage adaptatif) vs poids fixes
# Simulation d'un "GMM-régression" avec poids constants
def moe_fixed_weights(X, y, K=2, n_iter=40, seed=42):
    """MoE avec poids constants pi_k (pas de routage)."""
    N, D = X.shape
    rng = np.random.RandomState(seed)
    pi = np.ones(K) / K
    w_list = [np.linalg.lstsq(X, y, rcond=None)[0] + rng.randn(D) * 0.3 for _ in range(K)]
    sig2 = np.ones(K) * 0.5

    for _ in range(n_iter):
        weighted = np.zeros((N, K))
        for k in range(K):
            mu_k = X @ w_list[k]
            weighted[:, k] = pi[k] * np.exp(-0.5 * (y - mu_k)**2 / sig2[k]) / np.sqrt(2 * np.pi * sig2[k])
        r = weighted / (weighted.sum(axis=1, keepdims=True) + 1e-300)
        for k in range(K):
            rk = r[:, k]
            Xr = np.sqrt(rk)[:, None] * X
            yr = np.sqrt(rk) * y
            w_list[k] = np.linalg.lstsq(Xr, yr, rcond=None)[0]
            pred_k = X @ w_list[k]
            sig2[k] = np.sum(rk * (y - pred_k)**2) / (rk.sum() + 1e-10) + 1e-6
        pi = r.mean(axis=0)

    return w_list, sig2, pi, r

w_fixed, sig_fixed, pi_fixed, r_fixed = moe_fixed_weights(X_moe, y_moe, K=2)

mu_fixed = np.column_stack([X_g @ w_fixed[k] for k in range(2)])
y_pred_fixed = sum(pi_fixed[k] * X_g @ w_fixed[k] for k in range(2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

ax = axes[0]
ax.scatter(x_moe, y_moe, c='gray', alpha=0.4, s=15)
ax.plot(x_g, mu_fixed[:, 0], color='steelblue', linewidth=2, alpha=0.7, label='Expert 1')
ax.plot(x_g, mu_fixed[:, 1], color='coral', linewidth=2, alpha=0.7, label='Expert 2')
ax.plot(x_g, y_pred_fixed, 'k-', linewidth=2.5, label='Prédiction')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Poids constants $\\pi_k$ (pas de routage)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(x_moe, y_moe, c='gray', alpha=0.4, s=15)
ax.plot(x_g, mu_experts[:, 0], color='steelblue', linewidth=2, alpha=0.7, label='Expert 1')
ax.plot(x_g, mu_experts[:, 1], color='coral', linewidth=2, alpha=0.7, label='Expert 2')
ax.plot(x_g, y_pred_moe, 'k-', linewidth=2.5, label='Prédiction MoE')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Routage $g_k(\\mathbf{x})$ dépendant de l\'entrée')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
```

La comparaison est parlante. Avec des poids constants (à gauche), les experts peinent à se spécialiser: chacun tente de couvrir l'ensemble des données, et la prédiction globale reste un compromis insatisfaisant. Avec un routage dépendant de l'entrée (à droite), chaque expert se concentre sur la région où il excelle, et la prédiction suit la structure par morceaux des données.

Le mélange d'experts illustre comment la machinerie EM (responsabilités, moyennes pondérées, alternance E/M) s'adapte au cadre supervisé. Ces notions ne sont pas qu'un exercice de cours: elles réapparaissent en **apprentissage profond** et dans les **grands modèles de langage** (LLM). Dans ces contextes, une «couche MoE» désigne un ensemble d'experts (souvent des sous-réseaux) et un routage qui aiguille chaque entrée vers un petit nombre d'experts actifs; le but reste le même, soit spécialiser des composants et n'utiliser que ceux qui conviennent à l'entrée. Nous n'entrerons pas ici dans les détails des architectures (transformers, etc.), que nous aborderons plus tard, mais retenir l'idée (routage dépendant de l'entrée, experts locaux, combinaison pondérée) facilitera la lecture lorsque vous rencontrerez des modèles comme Mixtral ou des discussions sur le *sparse MoE*.

## Résumé

Ce chapitre a présenté les modèles génératifs pour la classification et le partitionnement, puis montré comment l'algorithme EM s'étend au cadre supervisé.

L'approche générative modélise comment les données sont produites: une classe est d'abord tirée selon un a priori, puis une observation est générée selon la distribution de cette classe. Le théorème de Bayes permet ensuite de calculer la probabilité a posteriori des classes.

Le classifieur naïf bayésien simplifie ce cadre en supposant que les caractéristiques sont conditionnellement indépendantes étant donné la classe. Cette hypothèse, rarement vraie en pratique, permet néanmoins une estimation efficace (formules fermées) et donne souvent de bons résultats en classification. Le lissage de Laplace évite les probabilités nulles et correspond à un estimateur MAP avec a priori uniforme.

L'analyse discriminante gaussienne suppose que chaque classe suit une distribution gaussienne. LDA (covariance partagée) donne des frontières linéaires, QDA (covariances différentes) des frontières quadratiques.

K-moyennes partitionne les données en groupes sphériques en alternant assignation et mise à jour des centroïdes. Les modèles de mélange gaussien généralisent cette approche en remplaçant les assignations dures par des responsabilités souples et les sphères par des ellipsoïdes. L'algorithme EM estime les paramètres en alternant le calcul des responsabilités (étape E) et la mise à jour des paramètres par des moyennes pondérées (étape M). EM maximise en fait une borne inférieure de la log-vraisemblance (l'ELBO); vue comme inférence variationnelle, EM correspond au cas où la distribution approchée $q$ sur les latentes est choisie égale à l'a posteriori exact à chaque itération.

Le mélange d'experts transpose cette logique au cadre supervisé: les experts se spécialisent dans différentes régions de l'espace d'entrée, et un réseau de routage apprend à aiguiller chaque observation vers l'expert approprié. L'algorithme EM s'y applique avec le même schéma d'alternance.

## Exercices

````{admonition} Exercice 1: Naive Bayes sur des données binaires ★
:class: hint dropdown

Un classifieur naïf bayésien est entraîné pour détecter les pourriels. Les données d'entraînement comprennent deux caractéristiques binaires: la présence du mot «gratuit» ($x_1$) et la présence du mot «urgent» ($x_2$).

Données:
- Pourriels (10 courriels): 8 contiennent «gratuit», 6 contiennent «urgent»
- Légitimes (20 courriels): 2 contiennent «gratuit», 4 contiennent «urgent»

1. Calculez les estimateurs EMV de tous les paramètres du modèle.
2. Classifiez un courriel contenant «gratuit» mais pas «urgent».
3. Appliquez le lissage de Laplace ($\alpha = 1$) et recalculez la classification.
````

```{admonition} Solution Exercice 1
:class: dropdown

1. **Estimateurs EMV:**

   *A priori de classe:*
   $$\hat{\pi}_{\text{pourriel}} = \frac{10}{30} = \frac{1}{3}, \quad \hat{\pi}_{\text{légitime}} = \frac{20}{30} = \frac{2}{3}$$
   
   *Probabilités conditionnelles (sans lissage):*
   $$\hat{\theta}_{\text{gratuit}|\text{pourriel}} = \frac{8}{10} = 0{,}8, \quad \hat{\theta}_{\text{urgent}|\text{pourriel}} = \frac{6}{10} = 0{,}6$$
   $$\hat{\theta}_{\text{gratuit}|\text{légitime}} = \frac{2}{20} = 0{,}1, \quad \hat{\theta}_{\text{urgent}|\text{légitime}} = \frac{4}{20} = 0{,}2$$

2. **Classification sans lissage:**

   Pour $\mathbf{x} = (1, 0)$ (gratuit présent, urgent absent):
   
   $$p(\text{pourriel}) \cdot p(x_1=1|\text{pourriel}) \cdot p(x_2=0|\text{pourriel}) = \frac{1}{3} \times 0{,}8 \times 0{,}4 = 0{,}107$$
   
   $$p(\text{légitime}) \cdot p(x_1=1|\text{légitime}) \cdot p(x_2=0|\text{légitime}) = \frac{2}{3} \times 0{,}1 \times 0{,}8 = 0{,}053$$
   
   Puisque $0{,}107 > 0{,}053$, le courriel est classé comme **pourriel**.

3. **Avec lissage de Laplace:**

   $$\hat{\theta}_{\text{gratuit}|\text{pourriel}} = \frac{8+1}{10+2} = \frac{9}{12} = 0{,}75$$
   $$\hat{\theta}_{\text{urgent}|\text{pourriel}} = \frac{6+1}{10+2} = \frac{7}{12} \approx 0{,}58$$
   $$\hat{\theta}_{\text{gratuit}|\text{légitime}} = \frac{2+1}{20+2} = \frac{3}{22} \approx 0{,}14$$
   $$\hat{\theta}_{\text{urgent}|\text{légitime}} = \frac{4+1}{20+2} = \frac{5}{22} \approx 0{,}23$$
   
   Recalcul:
   $$\text{Score pourriel} = \frac{1}{3} \times 0{,}75 \times (1 - 0{,}58) = \frac{1}{3} \times 0{,}75 \times 0{,}42 \approx 0{,}105$$
   $$\text{Score légitime} = \frac{2}{3} \times 0{,}14 \times (1 - 0{,}23) = \frac{2}{3} \times 0{,}14 \times 0{,}77 \approx 0{,}072$$
   
   La classification reste **pourriel**.
```

````{admonition} Exercice 2: Distance de Mahalanobis ★
:class: hint dropdown

Soit une classe avec $\boldsymbol{\mu} = (0, 0)$ et $\boldsymbol{\Sigma} = \begin{pmatrix} 4 & 0 \\ 0 & 1 \end{pmatrix}$.

1. Calculez la distance de Mahalanobis du point $(2, 0)$ à $\boldsymbol{\mu}$.
2. Calculez la distance de Mahalanobis du point $(0, 2)$ à $\boldsymbol{\mu}$.
3. Ces points ont la même distance euclidienne à l'origine. Pourquoi leurs distances de Mahalanobis diffèrent-elles?
4. Dessinez l'ellipse des points à distance de Mahalanobis 1 de l'origine.
````

```{admonition} Solution Exercice 2
:class: dropdown

La distance de Mahalanobis est $d_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$.

Avec $\boldsymbol{\Sigma}^{-1} = \begin{pmatrix} 1/4 & 0 \\ 0 & 1 \end{pmatrix}$:

1. **Point $(2, 0)$:**
   $$d_M^2 = (2, 0) \begin{pmatrix} 1/4 & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 2 \\ 0 \end{pmatrix} = 2 \times \frac{1}{4} \times 2 = 1$$
   
   Donc $d_M(2, 0) = 1$.

2. **Point $(0, 2)$:**
   $$d_M^2 = (0, 2) \begin{pmatrix} 1/4 & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 0 \\ 2 \end{pmatrix} = 2 \times 1 \times 2 = 4$$
   
   Donc $d_M(0, 2) = 2$.

3. **Interprétation:** Les deux points sont à distance euclidienne 2 de l'origine. Mais la distribution a une grande variance (4) dans la direction $x_1$ et une petite variance (1) dans la direction $x_2$. Un écart de 2 dans la direction $x_1$ correspond à 1 écart-type, tandis qu'un écart de 2 dans la direction $x_2$ correspond à 2 écarts-types. La distance de Mahalanobis mesure les écarts en «unités d'écart-type» dans chaque direction.

4. **Ellipse:** Les points à distance de Mahalanobis 1 satisfont:
   $$\frac{x_1^2}{4} + x_2^2 = 1$$
   
   C'est une ellipse avec demi-grand axe 2 (direction $x_1$) et demi-petit axe 1 (direction $x_2$).
```

````{admonition} Exercice 3: LDA vs régression logistique ★★
:class: hint dropdown

Considérez deux classes en 1D:
- Classe 0: moyenne $\mu_0 = 0$, 50 exemples
- Classe 1: moyenne $\mu_1 = 2$, 50 exemples
- Covariance partagée: $\sigma^2 = 1$
- A priori égaux: $\pi_0 = \pi_1 = 0{,}5$

1. Écrivez la fonction discriminante LDA pour chaque classe.
2. Trouvez le seuil de décision (la valeur $x^*$ où les deux classes sont équiprobables).
3. Pour la régression logistique avec $p(y=1|x) = \sigma(\theta_0 + \theta_1 x)$, montrez que le seuil de décision est $x^* = -\theta_0/\theta_1$.
4. Dans quelles situations LDA sera-t-il meilleur que la régression logistique? Et vice versa?
````

```{admonition} Solution Exercice 3
:class: dropdown

1. **Fonctions discriminantes LDA:**

   Pour LDA avec covariance partagée $\sigma^2$:
   $$\delta_c(x) = \log \pi_c - \frac{\mu_c^2}{2\sigma^2} + \frac{\mu_c}{\sigma^2} x$$
   
   Avec nos paramètres:
   $$\delta_0(x) = \log(0{,}5) - \frac{0}{2} + 0 \cdot x = -\log 2$$
   $$\delta_1(x) = \log(0{,}5) - \frac{4}{2} + 2x = -\log 2 - 2 + 2x$$

2. **Seuil de décision:**

   On cherche $x^*$ tel que $\delta_0(x^*) = \delta_1(x^*)$:
   $$-\log 2 = -\log 2 - 2 + 2x^*$$
   $$0 = -2 + 2x^*$$
   $$x^* = 1$$
   
   Le seuil est au milieu entre les deux moyennes (car les a priori sont égaux).

3. **Régression logistique:**

   $p(y=1|x) = 0{,}5$ quand $\sigma(\theta_0 + \theta_1 x) = 0{,}5$, donc $\theta_0 + \theta_1 x = 0$, d'où $x^* = -\theta_0/\theta_1$.

4. **Comparaison:**
   
   *LDA meilleur:*
   - Quand les données suivent effectivement une distribution gaussienne
   - Avec peu de données (les hypothèses fortes aident)
   - Quand les classes ont des covariances similaires
   
   *Régression logistique meilleure:*
   - Quand les distributions ne sont pas gaussiennes
   - Avec beaucoup de données (les hypothèses deviennent moins nécessaires)
   - Quand on veut des probabilités bien calibrées
   - Pour des données non continues (ex: caractéristiques catégorielles)
```

````{admonition} Exercice 4: Responsabilités GMM ★★
:class: hint dropdown

Un GMM à 2 composants en 1D a les paramètres:
- $\pi_1 = 0{,}3$, $\mu_1 = 0$, $\sigma_1^2 = 1$
- $\pi_2 = 0{,}7$, $\mu_2 = 3$, $\sigma_2^2 = 1$

1. Pour l'observation $x = 1$, calculez les responsabilités $r_1$ et $r_2$.
2. Pour l'observation $x = 1{,}5$, calculez les responsabilités.
3. Trouvez la valeur $x^*$ où $r_1 = r_2 = 0{,}5$.
4. Pourquoi $x^*$ n'est-il pas au milieu entre $\mu_1$ et $\mu_2$?
````

```{admonition} Solution Exercice 4
:class: dropdown

La densité gaussienne 1D est $\mathcal{N}(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$.

Avec $\sigma = 1$: $\mathcal{N}(x|\mu, 1) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2}\right)$.

1. **Pour $x = 1$:**
   
   $$\mathcal{N}(1|0, 1) = \frac{1}{\sqrt{2\pi}} e^{-0{,}5} \approx 0{,}242$$
   $$\mathcal{N}(1|3, 1) = \frac{1}{\sqrt{2\pi}} e^{-2} \approx 0{,}054$$
   
   $$r_1 = \frac{0{,}3 \times 0{,}242}{0{,}3 \times 0{,}242 + 0{,}7 \times 0{,}054} = \frac{0{,}073}{0{,}073 + 0{,}038} = \frac{0{,}073}{0{,}111} \approx 0{,}66$$
   $$r_2 \approx 0{,}34$$

2. **Pour $x = 1{,}5$:**
   
   $$\mathcal{N}(1{,}5|0, 1) \approx 0{,}130$$
   $$\mathcal{N}(1{,}5|3, 1) \approx 0{,}130$$
   
   $$r_1 = \frac{0{,}3 \times 0{,}130}{0{,}3 \times 0{,}130 + 0{,}7 \times 0{,}130} = \frac{0{,}3}{0{,}3 + 0{,}7} = 0{,}3$$
   $$r_2 = 0{,}7$$
   
   Au point équidistant des moyennes, les densités sont égales, donc les responsabilités sont proportionnelles aux poids $\pi_k$.

3. **Point où $r_1 = r_2$:**
   
   On cherche $x^*$ tel que $\pi_1 \mathcal{N}(x^*|\mu_1, 1) = \pi_2 \mathcal{N}(x^*|\mu_2, 1)$:
   $$0{,}3 \exp\left(-\frac{(x^*)^2}{2}\right) = 0{,}7 \exp\left(-\frac{(x^*-3)^2}{2}\right)$$
   
   En prenant le logarithme:
   $$\log(0{,}3) - \frac{(x^*)^2}{2} = \log(0{,}7) - \frac{(x^*-3)^2}{2}$$
   $$\log\frac{0{,}3}{0{,}7} = \frac{(x^*)^2 - (x^*-3)^2}{2} = \frac{6x^* - 9}{2} = 3x^* - 4{,}5$$
   $$-0{,}847 = 3x^* - 4{,}5$$
   $$x^* = \frac{4{,}5 - 0{,}847}{3} \approx 1{,}22$$

4. **Interprétation:** Le point $x^* \approx 1{,}22$ est plus proche de $\mu_1 = 0$ que du milieu ($1{,}5$) car le composant 2 a un poids plus élevé ($\pi_2 = 0{,}7$). Pour que les responsabilités soient égales, il faut que le point soit plus proche du composant de plus faible poids.
```

````{admonition} Exercice 5: Étape M de l'algorithme EM ★★
:class: hint dropdown

Soit un GMM à 2 composants en 1D avec les données $\{1, 2, 4, 5\}$ et les responsabilités suivantes après l'étape E:

| $x_n$ | $r_{n1}$ | $r_{n2}$ |
|-------|----------|----------|
| 1 | 0,9 | 0,1 |
| 2 | 0,8 | 0,2 |
| 4 | 0,2 | 0,8 |
| 5 | 0,1 | 0,9 |

1. Calculez $N_1$ et $N_2$ (les «nombres effectifs» de points par composant).
2. Calculez les nouvelles moyennes $\mu_1$ et $\mu_2$.
3. Calculez les nouvelles variances $\sigma_1^2$ et $\sigma_2^2$.
4. Calculez les nouveaux poids $\pi_1$ et $\pi_2$.
````

```{admonition} Solution Exercice 5
:class: dropdown

1. **Nombres effectifs:**
   $$N_1 = 0{,}9 + 0{,}8 + 0{,}2 + 0{,}1 = 2{,}0$$
   $$N_2 = 0{,}1 + 0{,}2 + 0{,}8 + 0{,}9 = 2{,}0$$

2. **Moyennes:**
   $$\mu_1 = \frac{1}{N_1}\sum_n r_{n1} x_n = \frac{0{,}9 \times 1 + 0{,}8 \times 2 + 0{,}2 \times 4 + 0{,}1 \times 5}{2{,}0}$$
   $$= \frac{0{,}9 + 1{,}6 + 0{,}8 + 0{,}5}{2{,}0} = \frac{3{,}8}{2{,}0} = 1{,}9$$
   
   $$\mu_2 = \frac{0{,}1 \times 1 + 0{,}2 \times 2 + 0{,}8 \times 4 + 0{,}9 \times 5}{2{,}0}$$
   $$= \frac{0{,}1 + 0{,}4 + 3{,}2 + 4{,}5}{2{,}0} = \frac{8{,}2}{2{,}0} = 4{,}1$$

3. **Variances:**
   $$\sigma_1^2 = \frac{1}{N_1}\sum_n r_{n1}(x_n - \mu_1)^2$$
   $$= \frac{0{,}9(1-1{,}9)^2 + 0{,}8(2-1{,}9)^2 + 0{,}2(4-1{,}9)^2 + 0{,}1(5-1{,}9)^2}{2{,}0}$$
   $$= \frac{0{,}9 \times 0{,}81 + 0{,}8 \times 0{,}01 + 0{,}2 \times 4{,}41 + 0{,}1 \times 9{,}61}{2{,}0}$$
   $$= \frac{0{,}729 + 0{,}008 + 0{,}882 + 0{,}961}{2{,}0} = \frac{2{,}58}{2{,}0} = 1{,}29$$
   
   $$\sigma_2^2 = \frac{0{,}1(1-4{,}1)^2 + 0{,}2(2-4{,}1)^2 + 0{,}8(4-4{,}1)^2 + 0{,}9(5-4{,}1)^2}{2{,}0}$$
   $$= \frac{0{,}1 \times 9{,}61 + 0{,}2 \times 4{,}41 + 0{,}8 \times 0{,}01 + 0{,}9 \times 0{,}81}{2{,}0}$$
   $$= \frac{0{,}961 + 0{,}882 + 0{,}008 + 0{,}729}{2{,}0} = \frac{2{,}58}{2{,}0} = 1{,}29$$

4. **Poids:**
   $$\pi_1 = \frac{N_1}{N} = \frac{2{,}0}{4} = 0{,}5$$
   $$\pi_2 = \frac{N_2}{N} = \frac{2{,}0}{4} = 0{,}5$$
```

````{admonition} Exercice 6: Convergence de k-moyennes et EM ★★★
:class: hint dropdown

1. Expliquez pourquoi k-moyennes converge toujours vers un minimum local de la distorsion.

2. Montrez que k-moyennes est un cas particulier de l'algorithme EM pour un GMM avec:
   - Covariances sphériques identiques: $\boldsymbol{\Sigma}_k = \sigma^2 \mathbf{I}$
   - $\sigma^2 \to 0$

3. Dans la limite $\sigma^2 \to 0$, que deviennent les responsabilités?

4. Pourquoi EM peut-il être préférable à k-moyennes même quand on ne veut qu'un partitionnement dur?
````

```{admonition} Solution Exercice 6
:class: dropdown

1. **Convergence de k-moyennes:**
   
   K-moyennes minimise la distorsion $J = \sum_n \sum_k r_{nk} \|\mathbf{x}_n - \boldsymbol{\mu}_k\|^2$ où $r_{nk} \in \{0,1\}$.
   
   - **Étape d'assignation:** Pour $\boldsymbol{\mu}$ fixé, assigner chaque point au centroïde le plus proche minimise $J$ (car on choisit le terme avec la plus petite distance).
   - **Étape de mise à jour:** Pour $r$ fixé, la moyenne minimise la somme des distances carrées (c'est un résultat classique de statistique).
   
   Chaque étape réduit ou maintient $J$. Comme $J \geq 0$ et le nombre d'assignations possibles est fini, l'algorithme converge.

2. **K-moyennes comme GMM limite:**
   
   Avec $\boldsymbol{\Sigma}_k = \sigma^2 \mathbf{I}$, la densité gaussienne est:
   $$\mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k, \sigma^2\mathbf{I}) \propto \exp\left(-\frac{\|\mathbf{x} - \boldsymbol{\mu}_k\|^2}{2\sigma^2}\right)$$
   
   Les responsabilités sont:
   $$r_{nk} = \frac{\pi_k \exp(-\|\mathbf{x}_n - \boldsymbol{\mu}_k\|^2 / 2\sigma^2)}{\sum_j \pi_j \exp(-\|\mathbf{x}_n - \boldsymbol{\mu}_j\|^2 / 2\sigma^2)}$$

3. **Limite $\sigma^2 \to 0$:**
   
   Quand $\sigma^2 \to 0$, l'exponentielle avec la plus petite distance domine:
   $$r_{nk} \to \begin{cases} 1 & \text{si } k = \arg\min_j \|\mathbf{x}_n - \boldsymbol{\mu}_j\| \\ 0 & \text{sinon} \end{cases}$$
   
   Les responsabilités deviennent binaires: c'est l'assignation de k-moyennes.

4. **Avantages de EM sur k-moyennes:**
   
   - **Détection des cas ambigus:** Les responsabilités souples identifient les points mal assignés
   - **Formes des groupes:** GMM capture des ellipses, k-moyennes ne fait que des sphères
   - **Initialisation plus robuste:** EM est moins sensible à l'initialisation grâce au ramollissement
   - **Critère de sélection de modèle:** La vraisemblance permet de comparer différents $K$
   - **Incertitude quantifiée:** Utile pour l'analyse en aval
```

````{admonition} Exercice 7: Générer des données avec un modèle génératif ★★★
:class: hint dropdown

Vous avez entraîné un GMM à 3 composants sur des données 2D et obtenu les paramètres suivants:

| $k$ | $\pi_k$ | $\boldsymbol{\mu}_k$ | $\boldsymbol{\Sigma}_k$ |
|-----|---------|----------------------|-------------------------|
| 1 | 0,2 | $(0, 0)$ | $\mathbf{I}$ |
| 2 | 0,5 | $(3, 3)$ | $2\mathbf{I}$ |
| 3 | 0,3 | $(0, 4)$ | $\begin{pmatrix} 1 & 0{,}5 \\ 0{,}5 & 1 \end{pmatrix}$ |

1. Décrivez l'algorithme pour générer $N$ nouveaux échantillons à partir de ce modèle.

2. Implémentez cet algorithme en Python (sans utiliser `sklearn.mixture`).

3. Générez 500 points et visualisez-les. Les groupes sont-ils visibles?

4. Calculez la log-vraisemblance moyenne de vos données générées. Est-ce cohérent?
````

```{admonition} Solution Exercice 7
:class: dropdown

1. **Algorithme de génération:**
   
   Pour générer un échantillon:
   1. Tirer un composant $k \sim \text{Catégorielle}(\boldsymbol{\pi})$
   2. Tirer $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$
   
   Répéter $N$ fois.

2. **Implémentation Python:**

   ```python
   import numpy as np
   
   # Paramètres
   pis = [0.2, 0.5, 0.3]
   mus = [np.array([0, 0]), np.array([3, 3]), np.array([0, 4])]
   sigmas = [np.eye(2), 2*np.eye(2), np.array([[1, 0.5], [0.5, 1]])]
   
   def generate_gmm_samples(n_samples, pis, mus, sigmas):
       K = len(pis)
       samples = []
       labels = []
       
       for _ in range(n_samples):
           # Étape 1: tirer un composant
           k = np.random.choice(K, p=pis)
           labels.append(k)
           
           # Étape 2: tirer de la gaussienne correspondante
           x = np.random.multivariate_normal(mus[k], sigmas[k])
           samples.append(x)
       
       return np.array(samples), np.array(labels)
   
   X, z = generate_gmm_samples(500, pis, mus, sigmas)
   ```

3. **Visualisation:** Les trois groupes devraient être visibles, avec le groupe 2 plus étalé (variance 2) et le groupe 3 légèrement allongé (corrélation positive).

4. **Log-vraisemblance:** Elle devrait être proche de celle des données d'entraînement originales, car les échantillons viennent de la même distribution. Une valeur typique serait autour de $-3$ à $-4$ par point (dépend de la normalisation).
```
