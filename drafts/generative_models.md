# Modèles génératifs

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Distinguer les approches générative et discriminative
- Dériver l'estimateur du maximum de vraisemblance pour le classifieur naïf bayésien
- Appliquer l'analyse discriminante gaussienne (LDA et QDA)
- Calculer les estimateurs EMV pour ces modèles
- Expliquer les avantages et inconvénients des modèles génératifs
```

## Approches générative vs discriminative

### Rappel: le théorème de Bayes

Pour classifier un exemple $\boldsymbol{x}$, nous voulons calculer la probabilité a posteriori:
$$
p(y = c \mid \boldsymbol{x}; \boldsymbol{\theta}) = \frac{p(\boldsymbol{x} \mid y = c; \boldsymbol{\theta}) \, p(y = c; \boldsymbol{\theta})}{\sum_{c'} p(\boldsymbol{x} \mid y = c'; \boldsymbol{\theta}) \, p(y = c'; \boldsymbol{\theta})}
$$

où:
- $p(\boldsymbol{x} \mid y = c; \boldsymbol{\theta})$ est la **vraisemblance** (densité conditionnelle de classe)
- $p(y = c; \boldsymbol{\theta})$ est l'**a priori** sur les classes

### Modèles génératifs

Dans l'approche **générative**, nous modélisons explicitement:
1. La distribution des entrées pour chaque classe: $p(\boldsymbol{x} \mid y = c)$
2. La distribution a priori sur les classes: $p(y = c)$

Le nom "génératif" vient du fait que nous pouvons **générer** de nouvelles données:
1. Échantillonner une classe $c \sim p(y)$
2. Échantillonner une observation $\boldsymbol{x} \sim p(\boldsymbol{x} \mid y = c)$

### Modèles discriminatifs

Dans l'approche **discriminative**, nous modélisons directement:
$$
p(y = c \mid \boldsymbol{x}; \boldsymbol{\theta})
$$

sans passer par $p(\boldsymbol{x} \mid y)$. Exemples: régression logistique, SVM, réseaux de neurones.

### Comparaison

| Aspect | Génératif | Discriminatif |
|--------|-----------|---------------|
| Ce qui est modélisé | $p(\boldsymbol{x}, y)$ ou $p(\boldsymbol{x} \mid y) p(y)$ | $p(y \mid \boldsymbol{x})$ |
| Peut générer des données | Oui | Non |
| Données manquantes | Peut les gérer | Difficile |
| Ajout de classes | Possible séparément | Nécessite réentraînement |
| Précision prédictive | Souvent inférieure | Souvent supérieure |
| Hypothèses | Plus fortes | Plus faibles |

## Classifieur naïf bayésien

### Hypothèse d'indépendance conditionnelle

Le **classifieur naïf bayésien** (*Naive Bayes*) suppose que les caractéristiques sont **conditionnellement indépendantes** étant donné la classe:
$$
p(\boldsymbol{x} \mid y = c, \boldsymbol{\theta}) = \prod_{d=1}^D p(x_d \mid y = c, \boldsymbol{\theta}_{dc})
$$

Cette hypothèse est "naïve" car elle est rarement vraie en pratique, mais le classifieur fonctionne souvent bien malgré tout.

### A posteriori

Le a posteriori sur les classes devient:
$$
p(y = c \mid \boldsymbol{x}, \boldsymbol{\theta}) = \frac{p(y = c \mid \boldsymbol{\pi}) \prod_{d=1}^D p(x_d \mid y = c, \boldsymbol{\theta}_{dc})}{\sum_{c'} p(y = c' \mid \boldsymbol{\pi}) \prod_{d=1}^D p(x_d \mid y = c', \boldsymbol{\theta}_{dc'})}
$$

où $\pi_c = p(y = c)$ est l'a priori de classe, et les paramètres sont $\boldsymbol{\theta} = (\boldsymbol{\pi}, \{\boldsymbol{\theta}_{dc}\})$.

### EMV pour Naive Bayes

La vraisemblance des données s'écrit:
$$
p(\mathcal{D} \mid \boldsymbol{\theta}) = \prod_{n=1}^N \text{Cat}(y_n \mid \boldsymbol{\pi}) \prod_{d=1}^D p(x_{nd} \mid y_n, \boldsymbol{\theta}_d)
$$

La log-vraisemblance se factorise:
$$
\log p(\mathcal{D} \mid \boldsymbol{\theta}) = \log p(\mathcal{D}_y \mid \boldsymbol{\pi}) + \sum_c \sum_d \log p(\mathcal{D}_{dc} \mid \boldsymbol{\theta}_{dc})
$$

où $\mathcal{D}_y = \{y_n\}$ sont les étiquettes et $\mathcal{D}_{dc} = \{x_{nd} : y_n = c\}$ sont les valeurs du trait $d$ pour les exemples de classe $c$.

Cela permet d'optimiser chaque terme **séparément**.

### Cas des caractéristiques catégorielles

Si les traits prennent des valeurs discrètes parmi $K$ possibilités, l'EMV est:
$$
\hat{\theta}_{dck} = \frac{N_{dck}}{N_c}
$$

où $N_{dck} = \sum_{n=1}^N \mathbb{I}(x_{nd} = k, y_n = c)$ est le nombre de fois que le trait $d$ vaut $k$ parmi les exemples de classe $c$.

### Cas des caractéristiques binaires

Si les traits sont binaires (présent/absent), on utilise la loi de Bernoulli:
$$
\hat{\theta}_{dc} = \frac{N_{dc}}{N_c}
$$

où $N_{dc}$ est le nombre de fois que le trait $d$ est présent dans la classe $c$.

### Lissage de Laplace

Un problème survient si un trait n'est jamais observé avec une certaine classe: $N_{dc} = 0$ implique $\hat{\theta}_{dc} = 0$, ce qui rend $p(\boldsymbol{x} \mid y = c) = 0$ même si un seul trait est absent.

Le **lissage de Laplace** (*add-one smoothing*) résout ce problème:
$$
\hat{\theta}_{dc} = \frac{N_{dc} + 1}{N_c + K}
$$

où $K$ est le nombre de valeurs possibles. Cela correspond à un a priori uniforme dans le cadre MAP.

## Analyse discriminante gaussienne

### Modèle

L'**analyse discriminante gaussienne** (GDA, *Gaussian Discriminant Analysis*) suppose que les densités conditionnelles de classe sont gaussiennes:
$$
p(\boldsymbol{x} \mid y = c, \boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)
$$

Le a posteriori sur les classes est:
$$
p(y = c \mid \boldsymbol{x}, \boldsymbol{\theta}) \propto \pi_c \mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)
$$

### Log a posteriori et fonction discriminante

Le logarithme du a posteriori (appelé **fonction discriminante**) est:
$$
\log p(y = c \mid \boldsymbol{x}, \boldsymbol{\theta}) = \log \pi_c - \frac{1}{2}\log|2\pi\boldsymbol{\Sigma}_c| - \frac{1}{2}(\boldsymbol{x} - \boldsymbol{\mu}_c)^\top \boldsymbol{\Sigma}_c^{-1}(\boldsymbol{x} - \boldsymbol{\mu}_c) + \text{const}
$$

Le terme quadratique $(\boldsymbol{x} - \boldsymbol{\mu}_c)^\top \boldsymbol{\Sigma}_c^{-1}(\boldsymbol{x} - \boldsymbol{\mu}_c)$ est la **distance de Mahalanobis** entre $\boldsymbol{x}$ et $\boldsymbol{\mu}_c$.

### Analyse discriminante quadratique (QDA)

Quand chaque classe a sa propre matrice de covariance $\boldsymbol{\Sigma}_c$, la frontière de décision est **quadratique**. Cette méthode s'appelle **QDA** (*Quadratic Discriminant Analysis*).

Les lignes de niveau sont des **ellipsoïdes** dont l'orientation et les axes sont déterminés par les vecteurs propres de $\boldsymbol{\Sigma}_c$.

### Analyse discriminante linéaire (LDA)

Si toutes les classes partagent la **même covariance** $\boldsymbol{\Sigma}$, le terme quadratique se simplifie:
$$
\log p(y = c \mid \boldsymbol{x}, \boldsymbol{\theta}) = \underbrace{\log \pi_c - \frac{1}{2}\boldsymbol{\mu}_c^\top \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_c}_{\gamma_c} + \boldsymbol{x}^\top \underbrace{\boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_c}_{\boldsymbol{\beta}_c} + \kappa
$$

où $\kappa$ ne dépend pas de $c$. La fonction discriminante est **linéaire** en $\boldsymbol{x}$:
$$
\log p(y = c \mid \boldsymbol{x}, \boldsymbol{\theta}) = \gamma_c + \boldsymbol{x}^\top \boldsymbol{\beta}_c + \kappa
$$

Cette méthode s'appelle **LDA** (*Linear Discriminant Analysis*).

### Estimation des paramètres

Pour la GDA, l'EMV des paramètres est:

**A priori de classe**:
$$
\hat{\pi}_c = \frac{N_c}{N}
$$

**Moyenne par classe**:
$$
\hat{\boldsymbol{\mu}}_c = \frac{1}{N_c} \sum_{n: y_n = c} \boldsymbol{x}_n
$$

**Covariance par classe** (QDA):
$$
\hat{\boldsymbol{\Sigma}}_c = \frac{1}{N_c} \sum_{n: y_n = c} (\boldsymbol{x}_n - \hat{\boldsymbol{\mu}}_c)(\boldsymbol{x}_n - \hat{\boldsymbol{\mu}}_c)^\top
$$

**Covariance partagée** (LDA):
$$
\hat{\boldsymbol{\Sigma}} = \frac{1}{N} \sum_{c=1}^C \sum_{n: y_n = c} (\boldsymbol{x}_n - \hat{\boldsymbol{\mu}}_c)(\boldsymbol{x}_n - \hat{\boldsymbol{\mu}}_c)^\top
$$

## Avantages des modèles génératifs

1. **Apprentissage parfois plus simple**: Naive Bayes s'apprend par comptage et moyennes, sans optimisation itérative

2. **Gestion des caractéristiques manquantes**: On peut marginaliser sur les valeurs manquantes

3. **Apprentissage incrémental par classe**: On peut ajouter de nouvelles classes sans réentraîner les existantes

4. **Apprentissage semi-supervisé**: On peut utiliser des données non étiquetées pour améliorer l'estimation de $p(\boldsymbol{x})$

5. **Robustesse aux caractéristiques factices**: Modéliser $p(\boldsymbol{x} \mid y)$ peut capturer les mécanismes causaux sous-jacents

## Inconvénients des modèles génératifs

1. **Hypothèses plus fortes**: Le modèle génératif doit spécifier $p(\boldsymbol{x} \mid y)$, ce qui est plus contraignant

2. **Précision souvent inférieure**: Les modèles discriminatifs optimisent directement ce qui nous intéresse

3. **Difficultés avec les caractéristiques prétraitées**: Après extraction de caractéristiques, les nouvelles variables peuvent avoir des corrélations complexes difficiles à modéliser

4. **Probabilités mal calibrées**: Les hypothèses fortes (comme l'indépendance) peuvent donner des probabilités biaisées

## Résumé

Les modèles génératifs modélisent la distribution jointe des entrées et des étiquettes:

- Le **classifieur naïf bayésien** suppose l'indépendance conditionnelle des caractéristiques
- L'**analyse discriminante gaussienne** modélise les classes par des gaussiennes
- **LDA** utilise une covariance partagée (frontière linéaire)
- **QDA** permet des covariances différentes (frontière quadratique)

Ces modèles sont simples à entraîner (formules fermées pour l'EMV), interprétables, et permettent de générer des données synthétiques.

## Exercices

```{admonition} Exercice 1: Naive Bayes binaire
:class: tip

Un classifieur Naive Bayes est entraîné sur des emails (spam vs non-spam) avec 2 traits binaires: présence du mot "gratuit" ($x_1$) et présence du mot "urgent" ($x_2$).

Données d'entraînement:
- Spam (10 emails): 8 avec "gratuit", 6 avec "urgent"
- Non-spam (20 emails): 2 avec "gratuit", 4 avec "urgent"

1. Calculez les EMV de tous les paramètres.
2. Classifiez un email contenant "gratuit" mais pas "urgent".
3. Appliquez le lissage de Laplace et recalculez.
```

```{admonition} Exercice 2: LDA en 1D
:class: tip

Deux classes en 1D:
- Classe 0: $\mu_0 = 0$, exemples: $\{-1, 0, 1\}$
- Classe 1: $\mu_1 = 3$, exemples: $\{2, 3, 4\}$

Avec une covariance partagée $\sigma^2 = 1$ et des a priori égaux:
1. Écrivez la fonction discriminante pour chaque classe.
2. Trouvez le seuil de décision.
3. Classifiez $x = 1.5$.
```

```{admonition} Exercice 3: QDA vs LDA
:class: tip

Expliquez géométriquement pourquoi:
1. LDA produit des frontières de décision linéaires
2. QDA produit des frontières de décision quadratiques
3. Donnez un exemple de données où QDA serait nettement meilleur que LDA
```

```{admonition} Exercice 4: Distance de Mahalanobis
:class: tip

Soit $\boldsymbol{\mu} = (0, 0)$ et $\boldsymbol{\Sigma} = \begin{pmatrix} 4 & 0 \\ 0 & 1 \end{pmatrix}$.

1. Calculez la distance de Mahalanobis de $(2, 0)$ à $\boldsymbol{\mu}$.
2. Calculez la distance de Mahalanobis de $(0, 2)$ à $\boldsymbol{\mu}$.
3. Pourquoi ces distances sont-elles différentes malgré la même distance euclidienne?
```
