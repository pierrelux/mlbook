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

Que signifie cette hypothèse? Considérons un problème de classification de courriels (pourriel ou non) avec des caractéristiques binaires indiquant la présence de certains mots. L'hypothèse d'indépendance conditionnelle suppose que, sachant qu'un courriel est un pourriel, la présence du mot «gratuit» n'influence pas la probabilité de présence du mot «urgent». Chaque mot apparaît indépendamment selon sa propre probabilité conditionnelle à la classe.

Formellement, pour une observation $\mathbf{x} = (x_1, \ldots, x_D)$ avec $D$ caractéristiques:

$$
p(\mathbf{x} \mid y = c) = \prod_{d=1}^D p(x_d \mid y = c)
$$

Cette factorisation réduit drastiquement le nombre de paramètres. Sans elle, modéliser $p(\mathbf{x} \mid y)$ pour des caractéristiques discrètes à $K$ valeurs nécessiterait $O(K^D)$ paramètres — un nombre qui explose avec la dimension. Avec l'hypothèse d'indépendance, nous n'avons besoin que de $O(KD)$ paramètres.

```{margin} Pourquoi «naïf»?
Le terme «naïf» ne signifie pas que le modèle est stupide ou simpliste. Il indique que l'hypothèse d'indépendance conditionnelle est rarement vraie en pratique. Dans notre exemple de courriels, les mots «gratuit» et «offre» apparaissent souvent ensemble dans les pourriels — ils ne sont pas vraiment indépendants. Pourtant, le classifieur fonctionne bien malgré cette violation de l'hypothèse. Cette robustesse fait du naïf bayésien un outil pratique, pas une méthode à éviter.
```

### Modèle complet et classification

Le modèle naïf bayésien spécifie:

1. Un a priori sur les classes: $p(y = c) = \pi_c$ avec $\sum_c \pi_c = 1$
2. Pour chaque caractéristique $d$ et chaque classe $c$, une distribution $p(x_d \mid y = c; \boldsymbol{\theta}_{dc})$

La probabilité a posteriori d'une classe devient:

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

Plus précisément, les dépendances entre caractéristiques peuvent affecter les probabilités absolues sans changer quelle classe domine. Si les mots «gratuit» et «offre» sont corrélés dans les pourriels, ignorer cette corrélation surestime la «surprise» de voir les deux ensemble — mais cette surestimation s'applique à toutes les classes et peut s'annuler dans la comparaison.

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

Quand chaque classe a sa propre matrice de covariance $\boldsymbol{\Sigma}_c$, la fonction discriminante contient un terme quadratique en $\mathbf{x}$. La frontière de décision entre deux classes (là où $\delta_c(\mathbf{x}) = \delta_{c'}(\mathbf{x})$) est une **quadrique** — une ellipse, une hyperbole, ou une parabole selon la configuration. Cette méthode s'appelle **QDA** (*Quadratic Discriminant Analysis*).

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

Les modèles précédents supposent que nous connaissons les classes des exemples d'entraînement. Que faire quand nous n'avons pas d'étiquettes, mais voulons découvrir une structure de groupes dans les données?

Le **partitionnement** (*clustering*) répond à cette question. Parmi les méthodes de partitionnement, les **modèles de mélange gaussien** (GMM, *Gaussian Mixture Models*) offrent une approche probabiliste qui généralise l'analyse discriminante gaussienne au cas non supervisé.

### Formulation

Un GMM suppose que les données sont générées par un mélange de $K$ distributions gaussiennes:

$$
p(\mathbf{x} \mid \boldsymbol{\theta}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

où:
- $\pi_k$ est le **poids du mélange** pour le composant $k$ (avec $\sum_k \pi_k = 1$ et $\pi_k \geq 0$)
- $\boldsymbol{\mu}_k$ est la moyenne du composant $k$
- $\boldsymbol{\Sigma}_k$ est la matrice de covariance du composant $k$

Nous pouvons interpréter ce modèle avec une **variable latente** $z \in \{1, \ldots, K\}$ qui indique de quel composant provient chaque observation:

$$
p(z = k) = \pi_k, \qquad p(\mathbf{x} \mid z = k) = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

Le processus de génération est: tirer un composant $z \sim \text{Cat}(\boldsymbol{\pi})$, puis tirer une observation $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_z, \boldsymbol{\Sigma}_z)$.

### Responsabilités

Pour une observation $\mathbf{x}_n$, la **responsabilité** du composant $k$ est la probabilité a posteriori que cette observation provienne du composant $k$:

$$
r_{nk} = p(z_n = k \mid \mathbf{x}_n) = \frac{\pi_k \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}
$$

Les responsabilités sont des valeurs continues dans $[0, 1]$ qui somment à 1 pour chaque observation. C'est un **partitionnement souple** (*soft clustering*): chaque point «appartient» partiellement à plusieurs composants.

```{code-cell} python
:tags: [hide-input]

# Illustration d'un GMM et des responsabilités
from sklearn.mixture import GaussianMixture

np.random.seed(42)

# Générer des données de 3 composants
n_samples = 300
X_gmm = np.vstack([
    np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_samples // 3),
    np.random.multivariate_normal([4, 0], [[0.5, 0.3], [0.3, 0.5]], n_samples // 3),
    np.random.multivariate_normal([2, 3], [[0.8, -0.4], [-0.4, 0.8]], n_samples // 3)
])

# Ajuster un GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_gmm)
responsibilities = gmm.predict_proba(X_gmm)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Gauche: assignations dures
ax = axes[0]
labels = gmm.predict(X_gmm)
colors = ['steelblue', 'coral', 'seagreen']
for k in range(3):
    mask = labels == k
    ax.scatter(X_gmm[mask, 0], X_gmm[mask, 1], c=colors[k], alpha=0.6, s=20, label=f'Composant {k+1}')

# Dessiner les ellipses des composants
from matplotlib.patches import Ellipse
for k in range(3):
    mean = gmm.means_[k]
    cov = gmm.covariances_[k]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    for n_std in [1, 2]:
        width, height = 2 * n_std * np.sqrt(eigenvalues)
        ellipse = Ellipse(mean, width, height, angle=angle, fill=False, 
                         color=colors[k], linewidth=2, linestyle='--')
        ax.add_patch(ellipse)

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Partitionnement dur (groupe le plus probable)')
ax.legend()
ax.set_xlim(-3, 7)
ax.set_ylim(-3, 6)

# Droite: responsabilités souples (couleur = mélange)
ax = axes[1]
# Couleur RGB basée sur les responsabilités
rgb = responsibilities @ np.array([[0.27, 0.51, 0.71],  # steelblue
                                    [1.0, 0.5, 0.31],   # coral
                                    [0.18, 0.55, 0.34]]) # seagreen
ax.scatter(X_gmm[:, 0], X_gmm[:, 1], c=rgb, alpha=0.6, s=20)

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Partitionnement souple (couleur = responsabilités)')
ax.set_xlim(-3, 7)
ax.set_ylim(-3, 6)

plt.tight_layout()
```

La figure compare le partitionnement dur (chaque point assigné à un seul composant) et le partitionnement souple (la couleur de chaque point est un mélange reflétant ses responsabilités). Les points dans les zones de chevauchement ont des couleurs intermédiaires, indiquant une appartenance incertaine.

### Lien avec k-moyennes

L'algorithme k-moyennes peut être vu comme un cas limite de GMM avec:
- Covariances sphériques identiques: $\boldsymbol{\Sigma}_k = \sigma^2 \mathbf{I}$ pour tout $k$
- Variance tendant vers zéro: $\sigma^2 \to 0$

Quand $\sigma^2 \to 0$, les responsabilités deviennent binaires: chaque point est assigné entièrement au composant dont la moyenne est la plus proche. C'est exactement l'assignation de k-moyennes.

GMM généralise k-moyennes en permettant:
- Des covariances différentes par composant (formes elliptiques variées)
- Des assignations souples (incertitude quantifiée)
- Des poids de mélange non uniformes

## L'algorithme EM

### Le problème d'estimation

Comment estimer les paramètres d'un GMM par maximum de vraisemblance? La log-vraisemblance est:

$$
\log p(\mathbf{X} \mid \boldsymbol{\theta}) = \sum_{n=1}^N \log \left( \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right)
$$

Le logarithme d'une somme rend l'optimisation difficile. Il n'y a pas de solution analytique comme pour le naïf bayésien ou LDA.

Si nous connaissions les assignations $z_n$ de chaque point, le problème serait simple: nous estimerions séparément les paramètres de chaque composant à partir des points qui lui sont assignés. Mais les $z_n$ sont inconnus — ce sont des variables latentes.

### L'intuition de l'algorithme EM

L'algorithme **Espérance-Maximisation** (EM) contourne cette difficulté par une stratégie d'alternance:

1. **Si nous connaissions les paramètres**, nous pourrions calculer les responsabilités $r_{nk} = p(z_n = k \mid \mathbf{x}_n, \boldsymbol{\theta})$
2. **Si nous connaissions les responsabilités**, nous pourrions estimer les paramètres par des moyennes pondérées

EM alterne entre ces deux étapes jusqu'à convergence. C'est une forme de **descente de coordonnées**: optimiser alternativement les responsabilités et les paramètres.

```{margin} Pourquoi EM converge-t-il?
EM maximise itérativement une **borne inférieure** de la log-vraisemblance, appelée ELBO (*Evidence Lower Bound*). À chaque itération, cette borne augmente (ou reste stable), garantissant la convergence vers un maximum local. La dérivation complète utilise l'inégalité de Jensen et sort du cadre de ce cours, mais l'intuition est que chaque étape améliore l'objectif sans jamais le dégrader.
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

## Résumé

Ce chapitre a présenté les modèles génératifs pour la classification et le partitionnement.

L'approche générative modélise comment les données sont produites: une classe est d'abord tirée selon un a priori, puis une observation est générée selon la distribution de cette classe. Le théorème de Bayes permet ensuite de calculer la probabilité a posteriori des classes.

Le classifieur naïf bayésien simplifie ce cadre en supposant que les caractéristiques sont conditionnellement indépendantes étant donné la classe. Cette hypothèse, rarement vraie en pratique, permet néanmoins une estimation efficace (formules fermées) et donne souvent de bons résultats en classification. Le lissage de Laplace évite les probabilités nulles et correspond à un estimateur MAP avec a priori uniforme.

L'analyse discriminante gaussienne suppose que chaque classe suit une distribution gaussienne. LDA (covariance partagée) donne des frontières linéaires, QDA (covariances différentes) des frontières quadratiques. Ces méthodes admettent aussi des solutions analytiques.

Les modèles de mélange gaussien étendent ce cadre au cas non supervisé, où les classes sont inconnues. L'algorithme EM estime les paramètres en alternant entre le calcul des responsabilités (probabilités d'appartenance souples) et la mise à jour des paramètres par des moyennes pondérées. K-moyennes est un cas limite de GMM avec assignations dures.

Ces modèles génératifs offrent des avantages uniques — génération de données, gestion des valeurs manquantes, interprétabilité — qui complètent les approches discriminatives vues aux chapitres précédents.

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
