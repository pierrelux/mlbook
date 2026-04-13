---
title: Révision pour l'examen final
---

# Révision pour l'examen final

Ce chapitre est un guide de révision structuré pour l'examen final. Il couvre l'ensemble de la matière du cours, avec un accent sur le contenu post-intra. Chaque section résume les concepts clés et propose des questions de pratique dans le style de l'examen.

## Partie I — Rappel de la matière pré-intra

### Le problème d'apprentissage (ch. 1)

Le risque $\Risk(f) = \E[\ell(f(\mathbf{x}), y)]$ mesure la performance attendue d'un prédicteur $f$ sur de nouvelles données. Comme la distribution conjointe $p(\mathbf{x}, y)$ est inconnue, on minimise le risque empirique $\hat{\Risk}(f) = \frac{1}{N}\sum_{i=1}^N \ell(f(\mathbf{x}_i), y_i)$ sur un ensemble d'entraînement.

Le prédicteur de Bayes optimal est celui qui minimise le risque parmi toutes les fonctions mesurables. Pour la perte quadratique, c'est la moyenne conditionnelle $f^*(\mathbf{x}) = \E[y \mid \mathbf{x}]$. Pour la perte 0-1, c'est le mode conditionnel $f^*(\mathbf{x}) = \arg\max_k p(y=k \mid \mathbf{x})$. Le risque de Bayes $\Risk^*$ est la borne inférieure du risque : aucun classifieur ne peut faire mieux.

Le maximum de vraisemblance (EMV) estime les paramètres en maximisant $\prod_{i=1}^N p(\mathbf{x}_i, y_i \mid \boldsymbol{\theta})$, ce qui revient à minimiser la log-vraisemblance négative. Sous bruit gaussien, l'EMV coïncide avec les moindres carrés; sous un modèle de Bernoulli, avec l'entropie croisée.

### Régression linéaire et Ridge (ch. 2)

La solution des moindres carrés ordinaires (MCO) est $\hat{\boldsymbol{\theta}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$. Lorsque $\mathbf{X}^\top\mathbf{X}$ est mal conditionnée ($d$ grand ou $d \approx N$), cette solution est instable.

Ridge ajoute un terme de régularisation : $\hat{\boldsymbol{\theta}}_\lambda = (\mathbf{X}^\top\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$. Via la SVD $\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$, Ridge applique un facteur de rétrécissement $\frac{d_j^2}{d_j^2 + \lambda}$ à chaque direction singulière. Les directions de faible variance ($d_j$ petit) sont davantage atténuées, ce qui stabilise la solution.

### Classification (ch. 3)

La régression logistique modélise $p(y=1 \mid \mathbf{x}) = \sigma(\boldsymbol{\theta}^\top \mathbf{x})$ où $\sigma(z) = 1/(1+e^{-z})$. La fonction de perte est l'entropie croisée binaire. En multiclasse, le softmax généralise la sigmoïde : $p(y=k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_j e^{z_j}}$.

Propriétés du softmax à retenir : invariance par translation ($\text{softmax}(\mathbf{z}+c) = \text{softmax}(\mathbf{z})$) et préservation de l'ordre (la classe avec le plus grand logit a la plus grande probabilité).

### Généralisation (ch. 4)

La décomposition biais-variance décompose l'erreur de généralisation en trois termes : le biais (erreur systématique due au modèle), la variance (sensibilité à l'échantillon d'entraînement) et le bruit irréductible.

Un modèle trop simple (forte régularisation, peu de paramètres) a un biais élevé : c'est le sous-apprentissage. Un modèle trop complexe (faible régularisation, beaucoup de paramètres) a une variance élevée : c'est le surapprentissage. La validation croisée permet de choisir les hyperparamètres pour équilibrer ce compromis.

L'expansion de caractéristiques $\phi(\mathbf{x})$ (polynomiale, cyclique, etc.) enrichit l'espace d'entrée et permet à un modèle linéaire de capturer des relations non linéaires.

### Le cadre probabiliste (ch. 5)

Trois perspectives sur l'apprentissage mènent au même algorithme :

1. **Risque empirique** : minimiser $\frac{1}{N}\sum_i \ell(f(\mathbf{x}_i), y_i)$
2. **Maximum de vraisemblance** : maximiser $\prod_i p(y_i \mid \mathbf{x}_i; \boldsymbol{\theta})$
3. **Théorie de l'information** : minimiser $D_{\text{KL}}(\hat{p} \| p_{\boldsymbol{\theta}})$

Le maximum a posteriori (MAP) incorpore un a priori : $\hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\max_{\boldsymbol{\theta}} \left[\log p(\Data \mid \boldsymbol{\theta}) + \log p(\boldsymbol{\theta})\right]$. Un a priori gaussien $p(\boldsymbol{\theta}) = \mathcal{N}(\mathbf{0}, \frac{1}{\lambda}\mathbf{I})$ donne la régression Ridge.

L'entropie $H(p) = -\sum_k p_k \log p_k$ mesure l'incertitude d'une distribution. La divergence KL $D_{\text{KL}}(P \| Q) = \sum_k P_k \log \frac{P_k}{Q_k}$ mesure la différence entre deux distributions; elle est asymétrique et pénalise fortement les cas où $P_k$ est grand et $Q_k$ est petit.

### Modèles probabilistes génératifs (ch. 6)

Les modèles **discriminatifs** (régression logistique) modélisent directement $p(y \mid \mathbf{x})$. Les modèles **génératifs** (naïf bayésien, LDA, QDA) modélisent $p(\mathbf{x} \mid y)p(y)$ et utilisent Bayes pour classifier.

Le classifieur naïf bayésien suppose l'indépendance conditionnelle des caractéristiques : $p(\mathbf{x} \mid y) = \prod_d p(x_d \mid y)$. LDA suppose des gaussiennes avec covariance commune $\boldsymbol{\Sigma}$ (frontière linéaire). QDA autorise des covariances distinctes $\boldsymbol{\Sigma}_k$ par classe (frontière quadratique).

Les mélanges de gaussiennes sont entraînés par l'algorithme EM, qui alterne entre le calcul des responsabilités (étape E) et la mise à jour des paramètres (étape M). Le mélange d'experts combine routage dépendant de l'entrée et modèles locaux.

---

## Partie II — Matière post-intra

### Réseaux de neurones (ch. 7)

Un perceptron multicouche (MLP) empile des transformations affines suivies de non-linéarités :

$$\mathbf{z}_\ell = \varphi(W_\ell \mathbf{z}_{\ell-1} + \mathbf{b}_\ell)$$

La couche de sortie est choisie selon la tâche : linéaire pour la régression, softmax pour la classification multiclasse, sigmoïde pour la classification binaire. Ce choix est dicté par le maximum de vraisemblance.

Les fonctions d'activation courantes sont ReLU ($\max(0, a)$), qui évite la saturation pour $a > 0$ mais souffre du problème des neurones morts, et GELU, variante lisse utilisée dans les transformeurs.

### Dérivation automatique (ch. 7) {#sec-autodiff}

La dérivation automatique (AD) calcule les dérivées exactes d'un programme en appliquant la règle de la chaîne sur son graphe de calcul.

#### Mode avant et mode arrière

Pour une composition $f = f_L \circ \cdots \circ f_1$, le jacobien est $J_f = J_{f_L} \cdots J_{f_1}$.

- **Mode avant (JVP)** : propage un vecteur tangent $\mathbf{v}$ de gauche à droite. Calcule $J_f \cdot \mathbf{v}$ en une passe. Utile pour peu d'entrées.
- **Mode arrière (VJP)** : propage un covecteur adjoint $\mathbf{u}^\top$ de droite à gauche. Calcule $\mathbf{u}^\top \cdot J_f$ en une passe. Pour une perte scalaire $\mathcal{L} : \R^n \to \R$, le gradient complet $\nabla_{\boldsymbol{\theta}} \mathcal{L}$ est obtenu en **une seule passe arrière**, quel que soit le nombre de paramètres $n$. C'est pourquoi la rétropropagation utilise le mode arrière.

#### Graphe de calcul et tri topologique

Un programme se décompose en un graphe acyclique dirigé (DAG) où chaque nœud est une opération élémentaire. Le mode avant parcourt ce graphe dans l'ordre topologique; le mode arrière le parcourt en sens inverse.

Lorsqu'une variable a plusieurs successeurs dans le graphe (branchement), le mode arrière **accumule** les contributions de chaque chemin par sommation des adjoints :

$$\bar{u} = \sum_{v \in \text{succ}(u)} \bar{v} \cdot D_u \varphi_v$$

#### Règles VJP

Chaque opération élémentaire a une règle VJP qui calcule $\mathbf{u}^\top J_f$ sans former le jacobien explicitement :

| Opération | VJP : $\mathbf{u}^\top \cdot J_f$ |
|---|---|
| Couche affine ($W\mathbf{z} + \mathbf{b}$), par rapport à $\mathbf{z}$ | $\mathbf{u}^\top W$ |
| Couche affine, par rapport à $W$ | $\mathbf{u}\mathbf{z}^\top$ (produit extérieur) |
| Couche affine, par rapport à $\mathbf{b}$ | $\mathbf{u}$ |
| Activation élément par élément $\varphi(\mathbf{a})$ | $\mathbf{u} \odot \varphi'(\mathbf{a})$ |
| Somme $s = \sum_i x_i$ | $u \cdot \mathbf{1}$ |

Le point crucial est que la règle VJP de l'activation coûte $O(m)$ (produit de Hadamard) plutôt que $O(m^2)$ (multiplication par un jacobien diagonal).

#### Le traceur

En pratique, les bibliothèques comme JAX et PyTorch utilisent un *traceur* : un objet qui encapsule une valeur numérique et enregistre les opérations sur une bande (tape de Wengert). Chaque entrée de la bande contient une *fermeture* (closure) qui capture les valeurs intermédiaires de la passe avant et contient la règle VJP. La passe arrière parcourt la bande en sens inverse et accumule les adjoints.

### Entraîner un réseau de neurones (ch. 8)

#### Optimiseurs

La descente de gradient stochastique par mini-lots (SGD) estime le gradient sur un sous-ensemble $\mathcal{B}$ de taille $B$ :

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot \frac{1}{B}\sum_{i \in \mathcal{B}} \nabla_{\boldsymbol{\theta}} \ell_i$$

**Momentum** accumule une moyenne mobile des gradients passés : $\mathbf{m}_{t+1} = \beta \mathbf{m}_t + \hat{\mathbf{g}}_t$, puis $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{m}_{t+1}$. Réduit les oscillations et accélère la convergence ($\beta \approx 0{,}9$).

**Adam** combine momentum (premier moment $\mathbf{m}$) et normalisation adaptative par dimension (second moment $\mathbf{s}$) :

$$\hat{\mathbf{m}} = \frac{\mathbf{m}_{t+1}}{1 - \beta_1^{t+1}}, \quad \hat{\mathbf{s}} = \frac{\mathbf{s}_{t+1}}{1 - \beta_2^{t+1}}, \quad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{s}}} + \epsilon}$$

Adam est le choix par défaut pour la plupart des architectures.

#### Dissolution et explosion du gradient

Dans un réseau profond, le gradient par rapport aux premières couches est un produit de jacobiens $\prod_\ell \text{diag}(\varphi'(\mathbf{a}_\ell)) W_\ell$. Si le rayon spectral de ces jacobiens est inférieur à 1, le gradient s'estompe exponentiellement (dissolution); s'il est supérieur à 1, il explose.

Les solutions :

- **ReLU** : dérivée unitaire pour $a > 0$, pas de saturation
- **Initialisation He** : $W_{ij} \sim \mathcal{N}(0, 2/n)$ pour compenser la mise à zéro de la moitié des activations par ReLU
- **Normalisation par lots** : normalise les pré-activations à moyenne nulle et variance unitaire sur le mini-lot, puis recalibre via des paramètres appris $\gamma$ et $\beta$
- **Connexions résiduelles** : $\mathbf{z}_{\ell+1} = \mathbf{z}_\ell + f(\mathbf{z}_\ell)$, ce qui donne $\frac{\partial \mathbf{z}_{\ell+1}}{\partial \mathbf{z}_\ell} = I + \frac{\partial f}{\partial \mathbf{z}_\ell}$. Le terme identité garantit un chemin direct pour le gradient.
- **Écrêtage du gradient** : borne la norme du gradient pour prévenir l'explosion

#### Régularisation

- **Arrêt précoce** : stopper l'entraînement lorsque l'erreur de validation remonte
- **Dropout** : désactive chaque neurone avec probabilité $p$ pendant l'entraînement (avec normalisation $1/(1-p)$ pour préserver l'espérance), forçant le réseau à ne pas dépendre d'un neurone particulier
- **Décroissance des poids** : avec Adam, préférer AdamW qui applique la contraction $\boldsymbol{\theta} \leftarrow (1-\eta\lambda)\boldsymbol{\theta}$ directement, sans passer par les moments

#### Transfert de représentations

Un réseau pré-entraîné se décompose en un **tronc** (extracteur de caractéristiques générales) et une **tête** (couche de sortie spécifique à la tâche). Deux stratégies :

1. **Extraction de caractéristiques** : geler le tronc, entraîner seulement la tête
2. **Réglage fin** : entraîner le réseau entier avec un taux d'apprentissage réduit

### Auto-encodeurs (ch. 9)

Un auto-encodeur apprend une représentation comprimée en entraînant un encodeur $f_\phi$ et un décodeur $g_\psi$ à reconstruire l'entrée via un goulot d'étranglement :

$$\mathcal{L} = \frac{1}{N}\sum_{n=1}^N \|\mathbf{x}_n - g_\psi(f_\phi(\mathbf{x}_n))\|^2$$

Un auto-encodeur **linéaire** est équivalent à l'ACP : la solution optimale projette sur le sous-espace engendré par les $L$ premiers vecteurs propres de la matrice de covariance. L'erreur de reconstruction est $\sum_{k=L+1}^D \lambda_k$.

L'auto-encodeur **débruiteur** reçoit une version corrompue $\tilde{\mathbf{x}}$ et apprend à reconstruire l'entrée propre $\mathbf{x}$, ce qui force le réseau à capturer la structure sous-jacente des données.

### Réseaux récurrents (ch. 9)

Un réseau récurrent maintient un état caché $\mathbf{h}_t$ mis à jour à chaque pas de temps :

$$\mathbf{h}_t = \varphi(W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t + \mathbf{b}_h)$$

Les paramètres $W_{hh}, W_{xh}, \mathbf{b}_h$ sont partagés entre tous les pas de temps, ce qui permet de traiter des séquences de longueur variable.

La rétropropagation à travers le temps (BPTT) déroule le réseau et applique la rétropropagation au graphe déroulé. Le gradient par rapport aux premières positions implique un produit de jacobiens $\prod_j \text{diag}(\varphi'(\mathbf{a}_j)) W_{hh}$ qui souffre de dissolution du gradient pour les dépendances à long terme.

Le LSTM résout ce problème avec un état de cellule $\mathbf{c}_t$ qui a un chemin quasi-linéaire :

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

où $\mathbf{f}_t$ (porte d'oubli) et $\mathbf{i}_t$ (porte d'entrée) sont des sigmoïdes apprises. Lorsque $\mathbf{f}_t \approx 1$, le gradient passe sans atténuation. Le GRU est une variante simplifiée avec deux portes au lieu de trois.

**Limite structurelle des RNN** : le traitement séquentiel ($\mathbf{h}_t$ dépend de $\mathbf{h}_{t-1}$) empêche la parallélisation et comprime tout le contexte dans un vecteur de taille fixe.

### Attention et transformeurs (ch. 10)

Le mécanisme d'attention contourne la limite des RNN en permettant à chaque position d'accéder directement à toutes les autres :

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Les matrices $Q = XW_Q$, $K = XW_K$, $V = XW_V$ sont des projections apprises de la séquence d'entrée. Le facteur $1/\sqrt{d_k}$ prévient la saturation du softmax lorsque la dimension est grande.

L'attention **multi-têtes** exécute $H$ mécanismes d'attention en parallèle avec des projections différentes, puis concatène les résultats : $\text{MultiHead} = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W_O$. Chaque tête peut capturer un type de relation différent.

Le lien avec la régression à noyau (Nadaraya-Watson) : l'attention est une généralisation apprise où le noyau de similarité $q^\top k / \sqrt{d_k}$ est paramétré par les projections.

Un **bloc transformeur** combine : auto-attention multi-têtes → connexion résiduelle → normalisation de couche → réseau dense (MLP) → connexion résiduelle → normalisation de couche.

L'**encodage positionnel** injecte l'information d'ordre dans les représentations, puisque l'attention elle-même est invariante à la permutation. L'encodage sinusoïdal utilise $p_{t,2j} = \sin(t/10000^{2j/d})$, $p_{t,2j+1} = \cos(t/10000^{2j/d})$.

Pour la génération autorégressive (décodeur), un **masque causal** empêche chaque position d'observer les positions futures en fixant les entrées correspondantes à $-\infty$ avant le softmax. Le **cache clés-valeurs** (KV cache) stocke $K$ et $V$ des positions précédentes pour éviter de les recalculer à chaque pas de génération, réduisant le coût de $O(T^3 d)$ à $O(T^2 d)$.

### Arbres et ensembles (ch. 11)

La **descente de gradient fonctionnelle** optimise directement les prédictions plutôt que les paramètres. À chaque itération $m$, on calcule les pseudo-résidus $r_{im} = -\frac{\partial \ell(y_i, f(\mathbf{x}_i))}{\partial f(\mathbf{x}_i)}\big|_{f=f_{m-1}}$ et on entraîne un modèle de base $F_m$ à les prédire. Pour la perte quadratique, les pseudo-résidus sont les résidus ordinaires $y_i - f_{m-1}(\mathbf{x}_i)$.

Un **arbre de décision** partitionne l'espace en régions rectangulaires par des coupures binaires successives. L'apprentissage est glouton : à chaque nœud, on choisit la coupure qui réduit le plus l'impureté. En classification, l'indice de Gini est $G = 1 - \sum_c \hat{\pi}_c^2$, l'entropie est $H = -\sum_c \hat{\pi}_c \log \hat{\pi}_c$.

Le **bagging** (forêts aléatoires) réduit la variance en moyennant $M$ arbres entraînés sur des échantillons bootstrap différents, avec une sélection aléatoire de caractéristiques à chaque nœud. Le **boosting** (gradient boosting) réduit le biais en ajoutant séquentiellement des modèles faibles qui corrigent les erreurs du modèle courant.

---

## Questions de pratique

### Question 1 — Quel modèle pour quel problème? (8 points)

Pour chacun des scénarios, choisissez le modèle ou l'architecture la plus appropriée et justifiez en deux phrases maximum. Chaque modèle peut être utilisé une fois, plusieurs fois, ou pas du tout.

Modèles disponibles : MLP, RNN, LSTM, Transformeur, Arbre de décision, Forêt aléatoire, Gradient boosting, Auto-encodeur, Régression logistique.

**(a)** *(2 pts)* Un hôpital dispose de 5000 dossiers patients avec 30 variables structurées (âge, tension, résultats sanguins, etc.) et veut prédire le risque de réadmission. L'interprétabilité des facteurs de risque est importante.

**(b)** *(2 pts)* Une équipe de traduction automatique veut entraîner un modèle sur des millions de paires de phrases. Le temps d'entraînement est un facteur critique et l'équipe dispose de plusieurs GPU.

**(c)** *(2 pts)* Un système de détection d'anomalies reçoit des lectures de capteurs sur une turbine. L'objectif est d'apprendre une représentation compacte du fonctionnement normal (sans étiquettes de pannes) pour signaler les déviations.

**(d)** *(2 pts)* Un assistant vocal doit transcrire des commandes en texte. Le signal audio arrive séquentiellement et chaque trame dépend fortement du contexte local récent, mais les séquences sont courtes (< 20 trames).

### Question 2 — Dérivation automatique (8 points)

Soit le programme suivant :

```python
def f(x, y):
    a = x * y       # multiplication
    b = a + x       # addition
    c = relu(b)     # ReLU
    return c
```

**(a)** *(2 pts)* Dessinez le graphe de calcul de ce programme en identifiant chaque nœud intermédiaire.

**(b)** *(3 pts)* En utilisant les règles VJP, calculez $\frac{\partial c}{\partial x}$ et $\frac{\partial c}{\partial y}$ par le mode arrière pour $x = 2, y = 3$. Montrez chaque étape de la propagation des adjoints en partant de $\bar{c} = 1$.

**(c)** *(1 pt)* Dans le graphe, la variable $x$ a deux successeurs ($a$ et $b$). Comment le mode arrière gère-t-il cette situation?

**(d)** *(2 pts)* On veut maintenant calculer le produit jacobien-vecteur $J_f \cdot \mathbf{v}$ pour $\mathbf{v} = (1, 0)^\top$. Quel mode (avant ou arrière) est le plus efficace et pourquoi? Que calcule-t-on concrètement avec ce vecteur $\mathbf{v}$?

### Question 3 — Compréhension de code (6 points)

Considérez la fonction suivante.

```python
def step(params, x_seq):
    W_hh, W_xh, b_h, W_hy, b_y = params
    h = np.zeros(W_hh.shape[0])
    outputs = []
    for x_t in x_seq:
        h = np.tanh(W_hh @ h + W_xh @ x_t + b_h)
        y_t = W_hy @ h + b_y
        outputs.append(y_t)
    return outputs
```

**(a)** *(2 pts)* Quel type de réseau cette fonction implémente-t-elle? Identifiez les éléments du code qui le confirment.

**(b)** *(2 pts)* Si la séquence `x_seq` contient $T = 100$ pas de temps, expliquez pourquoi le gradient de la perte par rapport à $W_{hh}$ peut devenir problématique. Nommez ce phénomène.

**(c)** *(2 pts)* Proposez deux modifications architecturales concrètes qui atténuent ce problème.

### Question 4 — Attention et transformeurs (8 points)

Un transformeur traite une séquence de $T = 4$ jetons avec des vecteurs de dimension $d_k = 2$. Les matrices de requêtes et de clés (après projection) sont :

$$Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{pmatrix}, \quad K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 1 & 1 \end{pmatrix}$$

**(a)** *(2 pts)* Calculez la matrice de scores $S = QK^\top / \sqrt{d_k}$.

**(b)** *(2 pts)* Si ce transformeur est utilisé comme modèle de langage (décodeur), quelle modification doit-on appliquer à $S$ avant le softmax? Écrivez la matrice modifiée.

**(c)** *(2 pts)* Quel est le coût en mémoire et en calcul du mécanisme d'attention en fonction de $T$ et $d_k$? Pourquoi cela pose-t-il problème pour les longues séquences?

**(d)** *(2 pts)* Expliquez en quoi le cache clés-valeurs (KV cache) réduit le coût total de la génération autorégressive de $T$ jetons, et donnez les complexités avec et sans cache.

### Question 5 — Optimisation et régularisation (6 points)

**(a)** *(2 pts)* Un étudiant entraîne un MLP à 20 couches avec des activations sigmoïdes et observe que les gradients des premières couches sont essentiellement nuls après quelques itérations. Expliquez la cause de ce problème et proposez deux modifications.

**(b)** *(2 pts)* Expliquez la différence entre la régularisation $L_2$ classique et la décroissance des poids (weight decay) lorsqu'on utilise Adam. Pourquoi cette distinction est-elle importante?

**(c)** *(2 pts)* On observe la courbe d'entraînement suivante : l'erreur d'entraînement diminue régulièrement, mais l'erreur de validation commence à remonter après l'époque 30. Nommez deux techniques de régularisation (autres que réduire la taille du réseau) qui pourraient aider, et expliquez brièvement leur mécanisme.

### Question 6 — Auto-encodeurs et représentations (4 points)

**(a)** *(2 pts)* Un auto-encodeur linéaire avec un goulot de dimension $L$ est entraîné sur des données de dimension $D = 10$. Si les valeurs propres de la matrice de covariance sont $\lambda_1 = 5, \lambda_2 = 3, \lambda_3 = 1, \lambda_4 = \ldots = \lambda_{10} = 0{,}1$, quelle est l'erreur de reconstruction minimale pour $L = 2$?

**(b)** *(2 pts)* En quoi un auto-encodeur non linéaire peut-il capter de la structure que l'ACP (ou l'auto-encodeur linéaire) ne peut pas représenter? Donnez un exemple géométrique concret.

### Question 7 — Dissolution du gradient dans les RNN (4 points)

**(a)** *(2 pts)* Écrivez l'expression du jacobien $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}$ pour un RNN avec activation tanh. Expliquez pourquoi le produit de $T$ tels jacobiens tend vers zéro.

**(b)** *(2 pts)* Montrez que la mise à jour de l'état de cellule du LSTM, $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$, permet au gradient de circuler sans atténuation lorsque $\mathbf{f}_t \approx 1$. Quel est le rôle analogue dans un réseau résiduel?

### Question 8 — Gradient boosting (4 points)

**(a)** *(2 pts)* Dans le gradient boosting avec perte quadratique, montrez que les pseudo-résidus $r_{im} = -\frac{\partial \ell}{\partial f(\mathbf{x}_i)}\big|_{f = f_{m-1}}$ se simplifient en résidus ordinaires $y_i - f_{m-1}(\mathbf{x}_i)$.

**(b)** *(2 pts)* Expliquez en quoi le boosting et le bagging (forêts aléatoires) s'attaquent à des composantes différentes de l'erreur de généralisation. Lequel réduit le biais? Lequel réduit la variance?

---

## Résumé des correspondances entre concepts

Le tableau suivant relie les thèmes transversaux du cours :

| Thème | Pré-intra | Post-intra |
|---|---|---|
| Dissolution du gradient | — | Sigmoïde → ReLU, connexions résiduelles, LSTM |
| Régularisation | Ridge ($\lambda \|\boldsymbol{\theta}\|^2$), validation croisée | Dropout, arrêt précoce, décroissance des poids |
| Maximum de vraisemblance | MCO = EMV gaussien, logistique = EMV Bernoulli | Couche de sortie des réseaux (softmax, linéaire) |
| Réduction de dimension | SVD, ACP via Ridge | Auto-encodeur (linéaire = ACP, non linéaire) |
| Modèles à variables latentes | Mélange de gaussiennes, EM | Auto-encodeur variationnel |
| Routage / attention | Mélange d'experts (routage par entrée) | Attention (routage appris par $QK^\top$), Nadaraya-Watson |
| Biais-variance | Sous/surapprentissage, expansion de caractéristiques | Boosting (biais) vs bagging (variance) |
