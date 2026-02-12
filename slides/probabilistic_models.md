---
marp: true
theme: mila
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Modèles probabilistes
## Naïf bayésien, analyse discriminante, MMG et k-moyennes

*Pierre-Luc Bacon*
IFT6390 – Fondements de l'apprentissage machine

---

## Plan de la présentation

1. **Stabilité numérique** : l'astuce log-sum-exp
2. **Théorie de l'information** : entropie, divergence KL, entropie croisée
3. **Classifieur naïf bayésien** : approche générative pour la classification
4. **Modèles de mélange gaussien** : partitionnement probabiliste et algorithme EM

---

<!-- _class: lead -->

# Stabilité numérique
## L'astuce log-sum-exp

---

<!-- footer: "📖 Chapitre 3 : Classification" -->

## Le problème : débordement numérique

La fonction **softmax** transforme des scores en probabilités :

$$\text{softmax}(\mathbf{a})_c = \frac{e^{a_c}}{\sum_{c'} e^{a_{c'}}}$$

**Problème** : Si les logits sont grands, l'exponentielle déborde.

| Logit $a_c$ | $e^{a_c}$ | Résultat |
|-------------|-----------|----------|
| 10 | 22 026 | OK |
| 100 | $2.7 \times 10^{43}$ | OK |
| 1000 | $\infty$ | **Débordement!** |

En Python, `np.exp(1000)` retourne `inf`, rendant le calcul inutilisable.

---

## L'astuce log-sum-exp

**Idée** : Soustraire le maximum avant d'exponencier.

$$\text{softmax}(\mathbf{a})_c = \frac{e^{a_c - a_{\max}}}{\sum_{c'} e^{a_{c'} - a_{\max}}}$$

où $a_{\max} = \max_c a_c$.

**Pourquoi ça fonctionne?** Le facteur $e^{-a_{\max}}$ apparaît au numérateur et au dénominateur :

$$\frac{e^{a_c - a_{\max}}}{\sum_{c'} e^{a_{c'} - a_{\max}}} = \frac{e^{a_c} \cdot e^{-a_{\max}}}{\sum_{c'} e^{a_{c'}} \cdot e^{-a_{\max}}} = \frac{e^{a_c}}{\sum_{c'} e^{a_{c'}}}$$

Le résultat est identique, mais le plus grand exposant vaut maintenant **0**.

---

## Implémentation stable

```python
def softmax_naive(a):
    """Version naïve - déborde pour grands logits"""
    return np.exp(a) / np.sum(np.exp(a))

def softmax_stable(a):
    """Version stable - soustrait le max"""
    a_max = np.max(a)
    exp_a = np.exp(a - a_max)
    return exp_a / np.sum(exp_a)
```

| Entrée | `softmax_naive` | `softmax_stable` |
|--------|-----------------|------------------|
| `[1, 2, 3]` | `[0.09, 0.24, 0.67]` | `[0.09, 0.24, 0.67]` |
| `[1000, 1001, 1002]` | `[nan, nan, nan]` | `[0.09, 0.24, 0.67]` |

En pratique, utilisez `scipy.special.softmax` qui gère automatiquement la stabilité.

---

## Visualisation du softmax

![w:850](_static/softmax_transform.png)

Le softmax transforme des scores arbitraires en probabilités valides ($\sum_c \mu_c = 1$, $\mu_c > 0$).

---

## La fonction log-sum-exp

Pour calculer $\log \sum_c e^{a_c}$ de manière stable :

$$\boxed{\text{logsumexp}(\mathbf{a}) = a_{\max} + \log \sum_{c} e^{a_c - a_{\max}}}$$

**Applications** : softmax, entropie croisée, GMM, responsabilités EM.

---

## Entraînement : logsumexp suffit!

L'entropie croisée pour la vraie classe $y$ est $-\log \text{softmax}(\mathbf{a})_y$. Développons :

$$-\log \frac{e^{a_y}}{\sum_c e^{a_c}} = -a_y + \log \sum_c e^{a_c} = \boxed{-a_y + \text{logsumexp}(\mathbf{a})}$$

**Pas besoin de calculer le softmax!** La perte se calcule directement à partir des logits.

```python
def cross_entropy_stable(logits, y):
    """Entropie croisée sans calculer softmax"""
    return -logits[y] + logsumexp(logits)
```

C'est ce que fait `torch.nn.CrossEntropyLoss` : il prend des **logits**, pas des probabilités.

---

<!-- _class: lead -->

# Théorie de l'information
## Entropie, divergence KL et entropie croisée

---

<!-- footer: "📖 Chapitre 5 : Le cadre probabiliste" -->

## L'entropie : mesurer l'incertitude

Considérons une pièce de monnaie :
- Pièce équilibrée ($p = 0{,}5$) : incertitude **maximale**
- Pièce truquée ($p = 0{,}99$) : incertitude **faible**

L'**entropie** quantifie cette incertitude :

$$\mathbb{H}(p) = -\sum_y p(y) \log p(y)$$

| Distribution | Entropie | Interprétation |
|--------------|----------|----------------|
| $p = [0{,}5, 0{,}5]$ | 1 bit | Incertitude maximale |
| $p = [0{,}99, 0{,}01]$ | 0,08 bits | Presque certain |
| $p = [1, 0]$ | 0 bits | Déterministe |

L'entropie est maximale pour la distribution **uniforme**.

---

## Pourquoi des bits? La base du logarithme

L'entropie = **nombre minimal de questions oui/non** pour identifier un résultat.

- Pièce équilibrée : 1 question → 1 bit
- Dé à 6 faces : $\log_2 6 \approx 2{,}58$ questions

| Base | Unité | Usage |
|------|-------|-------|
| $\log_2$ | **bits** | Théorie de l'information, compression |
| $\ln$ | **nats** | Apprentissage automatique (gradients) |

**En ML** : on utilise $\ln$ (dérivée simple), mais l'interprétation reste la même.

Conversion : $1 \text{ nat} = \frac{1}{\ln 2} \approx 1{,}44 \text{ bits}$

---

## Entropie de Bernoulli

Pour une variable binaire avec $p(Y=1) = \theta$ :

$$\mathbb{H}(\theta) = -\theta \log_2 \theta - (1-\theta) \log_2 (1-\theta)$$

![w:650](_static/entropy_bernoulli.png)

L'entropie est **symétrique** autour de $\theta = 0{,}5$ et atteint son maximum (1 bit) quand les deux résultats sont équiprobables.

---

## Entropie croisée

Supposons que les données suivent $p$, mais nous utilisons $q$ pour prédire.

L'**entropie croisée** mesure la surprise moyenne :

$$\mathbb{H}_{\text{ce}}(p, q) = -\sum_y p(y) \log q(y)$$

| Relation | Signification |
|----------|---------------|
| $q = p$ | $\mathbb{H}_{\text{ce}}(p, q) = \mathbb{H}(p)$ |
| $q \neq p$ | $\mathbb{H}_{\text{ce}}(p, q) > \mathbb{H}(p)$ |

Utiliser le « mauvais » modèle augmente toujours la surprise moyenne.

---

## Divergence de Kullback-Leibler

La **divergence KL** mesure la différence entre deux distributions :

$$D_{\text{KL}}(p \| q) = \sum_y p(y) \log \frac{p(y)}{q(y)} = \mathbb{E}_{p}\left[\log \frac{p(y)}{q(y)}\right]$$

**Propriétés** :
- $D_{\text{KL}}(p \| q) \geq 0$ toujours (inégalité de Gibbs)
- $D_{\text{KL}}(p \| q) = 0$ si et seulement si $p = q$
- **Non symétrique** : $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$

L'asymétrie a du sens : la surprise de quelqu'un qui croit en $q$ mais observe $p$ diffère de l'inverse.

---

## Décomposition fondamentale

$$\boxed{D_{\text{KL}}(p \| q) = \mathbb{H}_{\text{ce}}(p, q) - \mathbb{H}(p)}$$

![w:900](_static/kl_divergence.png)

En apprentissage, nous ne pouvons pas réduire $\mathbb{H}(p)$ (irréductible), mais nous pouvons minimiser $D_{\text{KL}}$ en améliorant notre modèle.

---

## La distribution empirique

Qu'est-ce que « les données » en termes de distribution?

**Exemple:** Lancer un dé 6 fois → résultats: 3, 1, 3, 5, 3, 2

| Face | 1 | 2 | 3 | 4 | 5 | 6 |
|------|---|---|---|---|---|---|
| Occurrences | 1 | 1 | 3 | 0 | 1 | 0 |
| Fréquence | 1/6 | 1/6 | 1/2 | 0 | 1/6 | 0 |

La **distribution empirique** $p_{\mathcal{D}}$ place une masse $1/N$ sur chaque observation :

$$p_{\mathcal{D}}(y) = \frac{1}{N} \sum_{i=1}^N \mathbb{1}(y_i = y) = \frac{\#\{i : y_i = y\}}{N}$$

C'est notre meilleure représentation des données sous forme de distribution.

---

## Convergence de la distribution empirique

Avec plus de données, $p_{\mathcal{D}}$ converge vers la vraie distribution $p$ :

| $N$ | Distribution empirique | Divergence KL |
|-----|------------------------|---------------|
| 20 | Bruitée, fluctuations | Élevée |
| 100 | Moins variable | Modérée |
| 1000 | Presque identique à $p$ | ≈ 0 |

C'est la **loi des grands nombres** : $p_{\mathcal{D}}(y) \xrightarrow{N \to \infty} p(y)$

---

## L'EMV minimise la divergence KL

**Objectif :** Trouver $p_{\boldsymbol{\theta}}$ proche de la distribution empirique $p_{\mathcal{D}}$.

$$D_{\text{KL}}(p_{\mathcal{D}} \| p_{\boldsymbol{\theta}}) = \underbrace{\mathbb{H}_{\text{ce}}(p_{\mathcal{D}}, p_{\boldsymbol{\theta}})}_{\text{dépend de } \boldsymbol{\theta}} - \underbrace{\mathbb{H}(p_{\mathcal{D}})}_{\text{constant}}$$

L'entropie croisée avec la distribution empirique est exactement la LVN :

$$\mathbb{H}_{\text{ce}}(p_{\mathcal{D}}, p_{\boldsymbol{\theta}}) = -\sum_y p_{\mathcal{D}}(y) \log p_{\boldsymbol{\theta}}(y) = -\frac{1}{N} \sum_{i=1}^N \log p(y_i \mid \boldsymbol{\theta})$$

$$\boxed{\text{Minimiser LVN} \iff \text{Minimiser } D_{\text{KL}}(p_{\mathcal{D}} \| p_{\boldsymbol{\theta}})}$$

---

## Interprétation géométrique

Le maximum de vraisemblance cherche, parmi toutes les distributions de la famille $\{p_{\boldsymbol{\theta}}\}$, celle qui est la **plus proche** de $p_{\mathcal{D}}$ au sens de la divergence KL.

![w:700](_static/kl_geometric_interpretation.png)

---

## Trois perspectives unifiées

Le même algorithme, trois interprétations :

| Perspective | Objectif | Résultat |
|-------------|----------|----------|
| **Décisionnelle** | Minimiser le risque empirique | Fonction de perte |
| **Probabiliste** | Maximiser la vraisemblance | Modèle génératif |
| **Informationnelle** | Minimiser la divergence KL | Distance aux données |

Pour la **régression** : bruit gaussien → perte quadratique
Pour la **classification** : Bernoulli/catégoriel → entropie croisée

Le choix du modèle probabiliste détermine la perte optimale.

---

<!-- _class: lead -->

# Classifieur naïf bayésien
## Classification générative avec indépendance conditionnelle

---

<!-- footer: "📖 Chapitre 6 : Modèles probabilistes génératifs" -->

## Approches générative vs discriminative

![w:900](_static/generative_vs_discriminative.png)

| Approche | Modélise | Question posée |
|----------|----------|----------------|
| **Générative** | $p(\mathbf{x} \mid y)$ et $p(y)$ | À quoi ressemblent les données de chaque classe? |
| **Discriminative** | $p(y \mid \mathbf{x})$ | Quelle classe pour cette observation? |

---

## Frontière de décision linéaire

La régression logistique prédit $\hat{y} = 1$ lorsque $p(y = 1 \mid \mathbf{x}) > \tfrac{1}{2}$, soit :

$$\sigma(\mathbf{w}^\top \mathbf{x} + b) > \tfrac{1}{2} \iff \mathbf{w}^\top \mathbf{x} + b > 0$$

Or $\mathbf{w}^\top \mathbf{x} + b = \log \frac{p(y=1 \mid \mathbf{x})}{p(y=0 \mid \mathbf{x})}$ est le log du rapport des probabilités a posteriori. La **frontière de décision** est l'ensemble des $\mathbf{x}$ où ce rapport vaut zéro, c'est-à-dire où les deux classes sont équiprobables. C'est un hyperplan perpendiculaire à $\mathbf{w}$.

Les modèles génératifs que nous allons voir (le classifieur naïf bayésien et l'analyse discriminante linéaire) mènent aussi à cette forme.

---

## Frontière linéaire et probabilités a posteriori

![w:900](_static/linear_decision_boundary.png)

À gauche : l'hyperplan $\mathbf{w}^\top \mathbf{x} + b = 0$ sépare les deux classes; $\mathbf{w}$ est perpendiculaire à cette frontière. À droite : $p(y = 1 \mid \mathbf{x})$ varie en sigmoïde le long de la direction $\mathbf{w}$.

---

## L'hypothèse d'indépendance conditionnelle

En général, la vraisemblance de classe se décompose par la **règle de chaîne** :

$$p(\mathbf{x} \mid y = c) = \prod_{d=1}^D p(x_d \mid x_1, \ldots, x_{d-1}, y = c) = p(x_1 \mid y = c)\, p(x_2 \mid x_1, y = c) \cdots p(x_D \mid x_1, \ldots, x_{D-1}, y = c)$$

Le classifieur **naïf bayésien** suppose l'**indépendance conditionnelle** :

$$p(\mathbf{x} \mid y = c) = \prod_{d=1}^D p(x_d \mid y = c)$$

| | Paramètres par classe | $D = 20$, $K = 2$ |
|---|---|---|
| Sans indépendance | $K^D - 1$ | $\approx 10^6$ |
| Avec indépendance | $D(K - 1)$ | $20$ |

---

## Effet de l'indépendance conditionnelle

![w:900](_static/naive_bayes_independence.png)

À gauche : le modèle général peut capturer les corrélations entre $x_1$ et $x_2$ (contours inclinés). À droite : l'hypothèse d'indépendance force $p(x_1, x_2 \mid y) = p(x_1 \mid y)\, p(x_2 \mid y)$, ce qui élimine toute corrélation (contours alignés avec les axes).

---

## Modèles graphiques probabilistes

Tous les modèles de ce chapitre se représentent comme des **graphes orientés** où les nœuds sont des variables (gris = observée, blanc = latente) et les flèches indiquent les dépendances :

![w:950](_static/pgm_models.png)

La structure du graphe détermine comment la distribution jointe se factorise, et donc comment estimer les paramètres.

---

## Le modèle complet

Par le **théorème de Bayes**, la probabilité a posteriori d'une classe est :

$$p(y = c \mid \mathbf{x}) = \frac{p(\mathbf{x} \mid y = c)\, p(y = c)}{\sum_{c'} p(\mathbf{x} \mid y = c')\, p(y = c')}$$

En substituant l'hypothèse d'indépendance avec $p(y = c) = \pi_c$ :

$$p(y = c \mid \mathbf{x}) = \frac{\pi_c \prod_{d=1}^D p(x_d \mid y = c)}{\sum_{c'} \pi_{c'} \prod_{d=1}^D p(x_d \mid y = c')}$$

**Classification** : choisir la classe qui maximise le numérateur.

---

## Forme logarithmique (stable)

Pour éviter les sous-dépassements numériques (*underflow*), on travaille en log :

$$\hat{y} = \arg\max_c \left[ \log \pi_c + \sum_{d=1}^D \log p(x_d \mid y = c) \right]$$

| Avantage | Explication |
|----------|-------------|
| Pas de sous-dépassement | Produit de petites probabilités → somme de logs |
| Plus rapide | Additions au lieu de multiplications |
| Numériquement stable | Pas de problème avec $10^{-300}$ |

C'est une application de l'astuce log-sum-exp!

---

## Estimation par maximum de vraisemblance

La log-vraisemblance du modèle naïf bayésien est:

$$\ell(\boldsymbol{\theta}) = \sum_{n=1}^N \left[ \log \pi_{y_n} + \sum_{d=1}^D \log p(x_{nd} \mid y_n) \right]$$

Grâce à l'indépendance conditionnelle, cette expression se décompose en termes séparés pour $\pi_c$ et pour chaque $p(x_d \mid y = c)$, ce qui permet de les estimer indépendamment.

**A priori de classe** : simplement la fréquence empirique :
$$\hat{\pi}_c = \frac{N_c}{N}$$

---

## Naïf bayésien avec caractéristiques binaires

Il reste à choisir la forme de $p(x_d \mid y = c)$. Pour des caractéristiques binaires (présence/absence), on utilise le modèle de Bernoulli :

$$p(x_d = 1 \mid y = c) = \theta_{dc} \qquad \Rightarrow \qquad \hat{\theta}_{dc} = \frac{\text{nb. d'exemples de classe } c \text{ où } x_d = 1}{N_c}$$

En classification de courriels, chaque $x_d$ indique la présence d'un mot dans le message. On estime la probabilité que le mot $d$ apparaisse dans un pourriel vs un courriel légitime, simplement en comptant les occurrences dans chaque classe.

---

## Naïf bayésien avec caractéristiques continues

Pour des caractéristiques continues, on modélise chaque dimension par une gaussienne univariée :

$$p(x_d \mid y = c) = \mathcal{N}(x_d \mid \mu_{dc},\, \sigma^2_{dc})$$
$$\hat{\mu}_{dc} = \frac{1}{N_c} \sum_{n: y_n = c} x_{nd}, \qquad \hat{\sigma}^2_{dc} = \frac{1}{N_c} \sum_{n: y_n = c} (x_{nd} - \hat{\mu}_{dc})^2$$

Par exemple, pour distinguer des espèces de fleurs à partir de mesures de pétales, on estime la moyenne et la variance de chaque mesure dans chaque espèce.

---

## De naïf bayésien à l'analyse discriminante

Le naïf bayésien gaussien suppose $\boldsymbol{\Sigma}_c$ diagonale. Si on lève cette hypothèse avec $p(\mathbf{x} \mid y = c) = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)$, le log du rapport a posteriori devient :

$$\log \frac{p(y=1 \mid \mathbf{x})}{p(y=0 \mid \mathbf{x})} = \underbrace{-\tfrac{1}{2}\mathbf{x}^\top(\boldsymbol{\Sigma}_1^{-1} - \boldsymbol{\Sigma}_0^{-1})\mathbf{x}}_{\text{terme quadratique}} + \mathbf{w}^\top \mathbf{x} + b$$

Rappel : la frontière de décision est l'ensemble des $\mathbf{x}$ où ce rapport vaut zéro (les deux classes sont équiprobables).

- **ADQ** (*QDA*), $\boldsymbol{\Sigma}_c$ libre : le terme quadratique subsiste → frontière quadratique
- **ADL** (*LDA*), $\boldsymbol{\Sigma}_c = \boldsymbol{\Sigma}$ : $\boldsymbol{\Sigma}_1^{-1} - \boldsymbol{\Sigma}_0^{-1} = \mathbf{0}$, le terme quadratique disparaît → frontière linéaire

---

## Récapitulatif : prédiction

Pour classer une nouvelle observation $\mathbf{x}$ avec un modèle génératif entraîné :

1. Pour chaque classe $c$, calculer la vraisemblance $p(\mathbf{x} \mid y = c)$
2. Multiplier par l'a priori $\pi_c$
3. Prédire la classe qui maximise : $\hat{y} = \arg\max_c \; \pi_c \, p(\mathbf{x} \mid y = c)$

Ce qui change d'un modèle à l'autre, c'est **l'étape 1** :

| Modèle | On calcule... |
|--------|---------------|
| Naïf bayésien | le produit $\prod_d p(x_d \mid y = c)$ (chaque caractéristique séparément) |
| ADL (*LDA*) | la densité $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_c, \boldsymbol{\Sigma})$ (covariance partagée) |
| ADQ (*QDA*) | la densité $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)$ (covariance par classe) |

---

## Récapitulatif : estimation

Tous les paramètres s'estiment par des **formules fermées** :

| Modèle | Paramètres estimés |
|--------|--------------------|
| Naïf bayésien | $\hat{\pi}_c = N_c/N$, puis fréquences (binaire) ou $\hat{\mu}_{dc}, \hat{\sigma}^2_{dc}$ (continu) par classe |
| ADL (*LDA*) | $\hat{\pi}_c$, $\hat{\boldsymbol{\mu}}_c$ par classe, $\hat{\boldsymbol{\Sigma}}$ combinée sur toutes les classes |
| ADQ (*QDA*) | $\hat{\pi}_c$, $\hat{\boldsymbol{\mu}}_c$, $\hat{\boldsymbol{\Sigma}}_c$ par classe |

---

## Le problème des probabilités nulles

Si le mot « gratuit » n'apparaît dans aucun courriel légitime : $\hat{\theta} = 0$

$$p(\text{légitime} \mid \mathbf{x}) \propto \pi_{\text{lég}} \times \ldots \times \underbrace{p(\text{gratuit} \mid \text{lég})}_{= 0} \times \ldots = 0$$

Un seul mot peut dominer entièrement la décision!

**Solution : lissage de Laplace** (*add-one smoothing*)

$$\hat{\theta}_{dck} = \frac{N_{dck} + 1}{N_c + K}$$

C'est le MAP avec un a priori uniforme (Beta(1,1) ou Dirichlet).

---

## Effet du lissage

![w:700](_static/laplace_smoothing.png)

Le lissage ajoute des « pseudo-observations » : comme si nous avions vu chaque événement au moins une fois avant de commencer.

---

## Pourquoi ça fonctionne?

L'hypothèse d'indépendance est presque toujours **violée** en pratique.

Pourtant, Naive Bayes fonctionne souvent bien. Pourquoi?

| Observation | Explication |
|-------------|-------------|
| On classe, on n'estime pas | Seul l'ordre des probabilités compte |
| Erreurs qui s'annulent | Surestimation dans toutes les classes |
| Régularisation implicite | Modèle simple = moins de surapprentissage |

**Attention** : Les probabilités retournées sont souvent mal calibrées (trop proches de 0 ou 1). Pour des probabilités fiables, préférez la régression logistique.

---

<!-- _class: lead -->

# Modèles de mélange gaussien (MMG)
## Partitionnement probabiliste et algorithme EM

---

<!-- footer: "📖 Chapitre 6 : Modèles probabilistes génératifs" -->

## Pourquoi le partitionnement?

Jusqu'ici, nous avons supposé que les étiquettes de classe sont connues. En pratique, elles sont souvent absentes : un biologiste mesure des fleurs sans connaître l'espèce, un commerce enregistre des transactions sans profil client, un généticien séquence des tumeurs sans sous-type défini.

Le partitionnement (*clustering*) regroupe automatiquement les observations en groupes homogènes, sans supervision.

---

## Le partitionnement en pratique

![w:950](_static/clustering_motivation.png)

Le jeu de données Iris : 150 fleurs décrites par la longueur et la largeur du pétale. À gauche, les données brutes sans étiquettes. À droite, un modèle de mélange gaussien découvre trois groupes correspondant aux espèces.

---

## Formulation du GMM

Un GMM suppose que les données proviennent d'un mélange de $K$ gaussiennes :

$$p(\mathbf{x} \mid \boldsymbol{\theta}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

![w:900](_static/gmm_formulation.png)

---

## Variable latente et processus génératif

On peut interpréter le GMM avec une variable latente $z \in \{1, \ldots, K\}$ :

$$p(z = k) = \pi_k, \quad p(\mathbf{x} \mid z = k) = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

![w:950](_static/gmm_generative_process.png)

---

## Responsabilités : partitionnement souple

La **responsabilité** du composant $k$ pour l'observation $\mathbf{x}_n$ :

$$r_{nk} = p(z_n = k \mid \mathbf{x}_n) = \frac{\pi_k \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

C'est un **partitionnement souple** : chaque point « appartient » partiellement à plusieurs composants ($r_{nk} \in [0, 1]$, $\sum_k r_{nk} = 1$).

---

## Partitionnement dur vs souple

![w:900](_static/gmm_clustering.png)

À gauche : chaque point assigné au composant le plus probable. À droite : la couleur reflète les responsabilités (mélange = incertitude).

---

<!-- _class: lead -->

# L'algorithme des k-moyennes
## Partitionnement dur par centroïdes

---

## L'algorithme des k-moyennes

Un **centroïde** $\boldsymbol{\mu}_k$ est la moyenne des observations assignées au groupe $k$ :

$$\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x}_n \in C_k} \mathbf{x}_n$$

![w:850](_static/kmeans_centroids.png)

---

## Les étapes des k-moyennes

1. Initialiser $K$ centroïdes $\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_K$
2. **Assignation** : $z_n = \arg\min_k \|\mathbf{x}_n - \boldsymbol{\mu}_k\|^2$
3. **Mise à jour** : $\boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{n: z_n = k} \mathbf{x}_n$
4. Répéter 2–3 jusqu'à convergence

Chaque point appartient à exactement un groupe : c'est un partitionnement dur.

---

## Du partitionnement dur au partitionnement souple

Les k-moyennes sont un cas limite du MMG avec covariances sphériques identiques $\boldsymbol{\Sigma}_k = \sigma^2 \mathbf{I}$ et $\sigma^2 \to 0$.

Le MMG les généralise sur trois axes :

- Chaque point peut appartenir **partiellement** à plusieurs groupes (assignation souple)
- Les groupes ont des formes **elliptiques** ($\boldsymbol{\Sigma}_k$ quelconque)
- Les groupes peuvent avoir des **poids différents** ($\pi_k$ variables)

---

## Le problème d'estimation

La log-vraisemblance du GMM est :

$$\log p(\mathbf{X} \mid \boldsymbol{\theta}) = \sum_{n=1}^N \log \left( \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right)$$

Le $\log$ d'une somme empêche de décomposer $\ell$ en une moyenne empirique sur les observations, contrairement à la MRE supervisée. On ne peut pas isoler la contribution de chaque composant.

On pourrait utiliser la descente de gradient, mais les paramètres ne sont pas libres dans $\mathbb{R}^n$ :

- $\boldsymbol{\pi} \in \Delta_K$ : les poids vivent sur le **simplexe** ($\pi_k \geq 0$, $\sum_k \pi_k = 1$)
- $\boldsymbol{\Sigma}_k \succ 0$ : chaque covariance doit rester **définie positive**
- La surface est **non convexe** : plusieurs optima locaux

L'algorithme EM contourne ces difficultés : si on connaissait les assignations $z_n$, on pourrait estimer chaque paramètre par des formules fermées; et réciproquement.

---

## L'algorithme Espérance-Maximisation (EM)

EM contourne le problème par **alternance** :

**Étape E (Espérance)** : Fixer les paramètres, calculer les responsabilités

$$r_{nk}^{(t)} = \frac{\pi_k^{(t)} \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k^{(t)}, \boldsymbol{\Sigma}_k^{(t)})}{\sum_{j} \pi_j^{(t)} \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j^{(t)}, \boldsymbol{\Sigma}_j^{(t)})}$$

**Étape M (Maximisation)** : Fixer les responsabilités, mettre à jour les paramètres

C'est une forme de **descente de coordonnées**.

---

## Étape M : mise à jour des paramètres

Soit $N_k = \sum_{n=1}^N r_{nk}$ le « nombre effectif » de points dans le composant $k$.

**Poids du mélange** :
$$\pi_k^{(t+1)} = \frac{N_k^{(t)}}{N}$$

**Moyennes** :
$$\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{N_k^{(t)}} \sum_{n=1}^N r_{nk}^{(t)} \mathbf{x}_n$$

**Covariances** :
$$\boldsymbol{\Sigma}_k^{(t+1)} = \frac{1}{N_k^{(t)}} \sum_{n=1}^N r_{nk}^{(t)} (\mathbf{x}_n - \boldsymbol{\mu}_k^{(t+1)})(\mathbf{x}_n - \boldsymbol{\mu}_k^{(t+1)})^\top$$

---

## Convergence de l'algorithme EM

![w:800](_static/em_convergence.gif)

À partir d'une initialisation arbitraire, EM ajuste progressivement les composants. Les ellipses représentent les contours à 1 et 2 écarts-types.

---

## Résumé de l'algorithme EM

**Entrée** : données $\mathbf{X}$, nombre de composants $K$

1. Initialiser $\boldsymbol{\theta}^{(0)} = (\boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma})$
2. Répéter jusqu'à convergence de $\ell(\boldsymbol{\theta})$ :

   **E** : $\quad r_{nk} \leftarrow \dfrac{\pi_k \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \pi_j \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$

   **M** : $\quad \pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k \leftarrow$ formules fermées (slide précédent)

La log-vraisemblance augmente (ou reste stable) à chaque itération.

---

## Considérations pratiques

| Aspect | Problème | Solution |
|--------|----------|----------|
| **Initialisation** | Maximum local | Plusieurs essais, k-means++ |
| **Choix de $K$** | Hyperparamètre | BIC, AIC, validation croisée |
| **Singularités** | Covariance dégénérée | Régularisation $\boldsymbol{\Sigma}_k + \epsilon \mathbf{I}$ |
| **Convergence** | Lente parfois | Critère d'arrêt sur $\Delta$LL |

**Initialisation recommandée** : Exécuter k-moyennes d'abord, puis utiliser les centroïdes comme moyennes initiales.

---

<!-- _class: lead -->

# Le modèle de Bradley-Terry
## Variables latentes et comparaisons par paires

---

<!-- footer: "📖 Chapitre 3 : Classification" -->

## Préférences et comparaisons par paires

Dans de nombreuses situations, nous n'observons pas de mesures absolues, mais des **comparaisons relatives** : qui gagne aux échecs, quelle réponse est préférée (RLHF), quel résultat de recherche est cliqué.

Le défi : convertir ces jugements en **scores numériques**. On cherche un score $s_k$ par objet tel que la différence $s_i - s_j$ prédise le résultat.

---

## Le modèle de Bradley-Terry

Chaque objet $k$ possède un score latent $s_k$. La probabilité que $i$ soit préféré à $j$ :

$$\boxed{P(i \succ j) = \sigma(s_i - s_j) = \frac{1}{1 + e^{-(s_i - s_j)}}}$$

- Si $s_i \gg s_j$ → $P(i \succ j) \approx 1$
- Si $s_i = s_j$ → $P(i \succ j) = 0{,}5$ (pile ou face)
- Si $s_i \ll s_j$ → $P(i \succ j) \approx 0$

Les scores sont **latents** : on ne les observe pas, seulement qui gagne.

---

## Lien avec la régression logistique

Pour $K$ objets, on construit un vecteur $\mathbf{x}_{ij} \in \mathbb{R}^K$ par comparaison :

$$x_{ij,k} = \begin{cases} +1 & \text{si } k = i \\ -1 & \text{si } k = j \\ 0 & \text{sinon} \end{cases} \qquad \Rightarrow \qquad \mathbf{s}^\top \mathbf{x}_{ij} = s_i - s_j$$

Le modèle prédit $P(y = 1 \mid \mathbf{x}_{ij}) = \sigma(\mathbf{s}^\top \mathbf{x}_{ij})$ : c'est une **régression logistique** sans ordonnée à l'origine, dont les coefficients sont directement les scores.

---

## Exemple : 4 joueurs, 5 matchs

$$\mathbf{X} = \begin{pmatrix} 1 & 0 & -1 & 0 \\ 0 & 1 & 0 & -1 \\ 1 & -1 & 0 & 0 \\ 0 & 0 & 1 & -1 \\ 0 & 1 & -1 & 0 \end{pmatrix}, \quad \mathbf{y} = \begin{pmatrix} 1 \\ 1 \\ 0 \\ 1 \\ 1 \end{pmatrix}$$

Chaque ligne encode un match : $+1$ pour le premier joueur, $-1$ pour le second. L'étiquette $y = 1$ indique que le premier joueur a gagné.

Le produit $\mathbf{X}\mathbf{s}$ donne les différences de scores : $\mathbf{X}\mathbf{s} = (s_0 - s_2,\; s_1 - s_3,\; s_0 - s_1,\; s_2 - s_3,\; s_1 - s_2)^\top$.

---

## Deux perspectives, un même modèle

| | Discriminative (rég. logistique) | Générative (Thurstone) |
|---|---|---|
| **Modélise** | $P(y \mid \mathbf{x}_{ij})$ directement | Les performances latentes $Z_i, Z_j$ |
| **Formule** | $\sigma(\mathbf{s}^\top \mathbf{x}_{ij})$ | $P(Z_i > Z_j)$ après marginalisation |
| **Estimation** | Maximiser la log-vraisemblance conditionnelle | Marginaliser, puis maximiser |
| **Avantage** | Algorithme simple (rég. logistique) | Interprétation du processus de génération |

Les deux perspectives donnent **exactement la même formule** $P(i \succ j) = \sigma(s_i - s_j)$. Le choix de perspective influence l'interprétation, pas le résultat.

---

<!-- footer: "📖 Chapitre 6 : Modèles probabilistes génératifs" -->

## Perspective à variables latentes

L.L. Thurstone (1927) : chaque objet a un score moyen $s_i$, mais sa **performance** à chaque comparaison est bruitée :

$$Z_i = s_i + \epsilon_i, \quad \epsilon_i \sim \text{Gumbel}(0,1)$$

Le joueur $i$ bat $j$ quand $Z_i > Z_j$. Les performances $Z_i, Z_j$ sont les **variables latentes** : on n'observe que le résultat.

---

## De la Gumbel à la sigmoïde

Les performances $Z_i, Z_j$ sont inobservées. Pour obtenir $P(i \succ j)$ en fonction des seuls paramètres $s_i, s_j$, on **intègre sur toutes les valeurs possibles** des performances (c'est la marginalisation) :

$$P(i \succ j) = \int\!\!\int \mathbb{1}[z_i > z_j] \, p(z_i \mid s_i) \, p(z_j \mid s_j) \, dz_i \, dz_j$$

La condition $Z_i > Z_j$ se réécrit $\epsilon_j - \epsilon_i < s_i - s_j$. Or la différence de deux Gumbel indépendantes suit une distribution logistique, dont la fonction de répartition est la sigmoïde :

$$P(\epsilon_j - \epsilon_i < t) = \sigma(t) \qquad \Rightarrow \qquad P(i \succ j) = \sigma(s_i - s_j)$$

---

## Estimation et prédiction avec Bradley-Terry

**Estimation** (à partir de comparaisons observées) :

1. Construire la matrice $\mathbf{X}$ : une ligne par comparaison, $+1$ pour l'objet $i$, $-1$ pour l'objet $j$
2. Appliquer la régression logistique sans ordonnée à l'origine sur $(\mathbf{X}, \mathbf{y})$
3. Les coefficients estimés $\hat{\boldsymbol{\theta}}$ sont directement les scores $\hat{s}_1, \ldots, \hat{s}_K$

**Prédiction** (pour une nouvelle comparaison $i$ vs $j$) :

$$\hat{P}(i \succ j) = \sigma(\hat{s}_i - \hat{s}_j)$$

Le classement global s'obtient en triant les objets par score décroissant. En RLHF, ce même principe est appliqué avec un réseau de neurones $r_\phi(\text{réponse})$ à la place du vecteur de scores.

---

## Le parallèle avec les MMG

| Aspect | MMG | Bradley-Terry |
|--------|-----|---------------|
| Variable latente | $z \in \{1, \ldots, K\}$ (discrète) | Performances $Z_i, Z_j \in \mathbb{R}$ (continues) |
| Observation | $\mathbf{x} \in \mathbb{R}^D$ | $y \in \{0, 1\}$ (qui gagne) |
| Marginalisation | $\sum_k$ (somme) | $\iint$ (intégrale) |
| Résultat | Densité de mélange | Sigmoïde de la différence |

La même idée, marginaliser des variables latentes pour obtenir la vraisemblance, apparaît dans des contextes très différents.

---

<!-- footer: "" -->

## Résumé : Modèles vus dans ce cours

| Modèle | Hypothèse clé | Usage |
|--------|---------------|-------|
| Naïf bayésien | Indépendance conditionnelle | Classification (texte, pourriels) |
| ADL / ADQ | Gaussien par classe | Classification supervisée |
| K-moyennes | Centroïdes, assignation dure | Partitionnement non supervisé |
| MMG | Mélange de gaussiennes | Partitionnement souple |
| Bradley-Terry | Scores latents, sigmoïde | Comparaisons par paires |

L'algorithme EM estime les paramètres des MMG par alternance E/M.

---

<!-- _class: lead -->

# Questions?

**Exercices recommandés** :
- Exercice 1 (ch5) : Entropie et divergence KL
- Exercice 1 (ch6) : Naive Bayes sur données binaires
- Exercice 4 (ch6) : Responsabilités GMM
- Exercice 5 (ch6) : Étape M de l'algorithme EM
