# Réseaux de neurones et apprentissage profond

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Relier les réseaux de neurones au cadre du maximum de vraisemblance vu aux chapitres précédents
- Expliquer comment les réseaux de neurones apprennent leurs propres caractéristiques, contrairement aux modèles à expansion fixe
- Définir l'architecture d'un perceptron multicouche (MLP) et le rôle des fonctions d'activation
- Dériver l'algorithme de rétropropagation à partir de la règle de la chaîne
- Distinguer les modes avant (JVP) et arrière (VJP) de la différentiation automatique
- Expliquer le paradigme de la programmation différentiable
```

Aux chapitres 2 et 3, nous avons construit des modèles linéaires pour la régression et la classification. Au chapitre 4, nous avons vu comment enrichir ces modèles en transformant les entrées par une fonction $\boldsymbol{\phi}$ fixée à l'avance. Ce chapitre franchit une étape supplémentaire: au lieu de choisir $\boldsymbol{\phi}$ manuellement, nous allons l'apprendre à partir des données. Cette idée simple conduit aux réseaux de neurones.

Dans ce chapitre, nous commençons par rappeler le cadre probabiliste qui unifie régression et classification, puis nous montrons comment la volonté de dépasser les modèles linéaires mène naturellement aux réseaux de neurones. Nous dérivons ensuite l'algorithme de rétropropagation, avant d'introduire la différentiation automatique et le paradigme de la programmation différentiable.

## Le cadre unifié: prédire les paramètres d'une distribution

### Régression et classification comme maximum de vraisemblance

Revenons au cadre probabiliste des chapitres 2 et 5. Dans tous les modèles que nous avons vus, le problème d'apprentissage supervisé prend la même forme: étant donné une entrée $\mathbf{x}$, nous voulons prédire les paramètres d'une distribution conditionnelle $p(y | \mathbf{x}; \boldsymbol{\theta})$, puis trouver $\boldsymbol{\theta}$ par maximum de vraisemblance.

En **régression**, nous avons supposé un bruit gaussien:

$$
p(y | \mathbf{x}; \boldsymbol{\theta}) = \mathcal{N}(y \,|\, \mu(\mathbf{x}), \sigma^2)
$$

Le modèle prédit la **moyenne** $\mu(\mathbf{x})$ de la distribution. La log-vraisemblance négative donne, à une constante près, la perte des moindres carrés:

$$
-\sum_{i=1}^N \log p(y_i | \mathbf{x}_i; \boldsymbol{\theta}) \propto \sum_{i=1}^N (y_i - \mu(\mathbf{x}_i))^2
$$

En **classification binaire**, nous avons supposé une distribution de Bernoulli:

$$
p(y | \mathbf{x}; \boldsymbol{\theta}) = \mu(\mathbf{x})^y (1 - \mu(\mathbf{x}))^{1-y}
$$

Le modèle prédit la **probabilité** $\mu(\mathbf{x}) = p(y = 1 | \mathbf{x})$. La log-vraisemblance négative donne l'entropie croisée binaire. Pour la **classification multiclasse**, la distribution catégorielle et la fonction softmax jouent le même rôle, avec l'entropie croisée catégorielle comme perte.

Le point commun est le suivant: dans chaque cas, il y a une **fonction** $\mu(\mathbf{x})$ qui prend une entrée et produit les paramètres de la distribution de sortie. Toute la question est: quelle forme donner à cette fonction?

### Des modèles linéaires aux caractéristiques apprises

Jusqu'ici, nos modèles ont été **linéaires dans les entrées**. Pour la régression:

$$
\mu(\mathbf{x}) = \boldsymbol{\theta}^\top \mathbf{x}
$$

Pour la classification binaire, la probabilité passe par une sigmoïde, mais la pré-activation reste linéaire:

$$
\mu(\mathbf{x}) = \sigma(\boldsymbol{\theta}^\top \mathbf{x})
$$

Au chapitre 4, nous avons étendu cette approche avec l'expansion de caractéristiques. Au lieu d'utiliser $\mathbf{x}$ directement, nous le transformons par une fonction $\boldsymbol{\phi}: \mathbb{R}^d \to \mathbb{R}^D$ choisie à l'avance --- polynômes, fonctions trigonométriques, bases radiales, etc.:

$$
\mu(\mathbf{x}) = \boldsymbol{\theta}^\top \boldsymbol{\phi}(\mathbf{x})
$$

Le modèle reste linéaire dans les paramètres $\boldsymbol{\theta}$, ce qui facilite l'optimisation, mais il capture des relations non linéaires en $\mathbf{x}$ grâce au choix de $\boldsymbol{\phi}$.

Cette approche a une limite importante: le choix de $\boldsymbol{\phi}$ repose entièrement sur l'expertise du praticien. Pour des données tabulaires simples, cela peut fonctionner. Mais pour des images, du texte ou de l'audio, concevoir manuellement les bonnes caractéristiques est extrêmement difficile --- et souvent le facteur limitant de la performance.

L'idée des réseaux de neurones est de **paramétrer $\boldsymbol{\phi}$ et de l'apprendre à partir des données**. Au lieu d'écrire $\boldsymbol{\theta}^\top \boldsymbol{\phi}(\mathbf{x})$ avec $\boldsymbol{\phi}$ fixé, nous écrivons:

$$
\mu(\mathbf{x}) = \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}; \boldsymbol{\theta}_\phi)
$$ (eq:nn-key-idea)

où $\boldsymbol{\phi}(\cdot; \boldsymbol{\theta}_\phi)$ est elle-même une fonction paramétrique que nous optimisons en même temps que $\mathbf{w}$. Le modèle apprend simultanément la **représentation** et le **prédicteur**.

Cela soulève deux questions: quelle forme donner à $\boldsymbol{\phi}(\cdot; \boldsymbol{\theta}_\phi)$, et comment optimiser l'ensemble des paramètres? Le reste de ce chapitre répond à ces deux questions.

## Anatomie d'un réseau de neurones

### Un neurone: transformation affine et non-linéarité

La brique de base d'un réseau de neurones est simple: une transformation affine suivie d'une fonction non linéaire. Pour une entrée $\mathbf{x} \in \mathbb{R}^d$:

$$
h = \varphi(\mathbf{w}^\top \mathbf{x} + b)
$$

où $\mathbf{w} \in \mathbb{R}^d$ est un vecteur de poids, $b \in \mathbb{R}$ est un biais, et $\varphi: \mathbb{R} \to \mathbb{R}$ est une **fonction d'activation** non linéaire. La quantité $a = \mathbf{w}^\top \mathbf{x} + b$ est la **pré-activation** et $h$ est l'**activation** du neurone.

Une couche de $m$ neurones applique cette opération en parallèle, ce qui s'écrit sous forme matricielle:

$$
\mathbf{h} = \varphi(W \mathbf{x} + \mathbf{b})
$$

où $W \in \mathbb{R}^{m \times d}$ est la matrice de poids, $\mathbf{b} \in \mathbb{R}^m$ le vecteur de biais, et $\varphi$ est appliquée élément par élément.

### Pourquoi la non-linéarité est indispensable

Sans la fonction d'activation $\varphi$, une couche se réduit à une transformation affine $\mathbf{h} = W\mathbf{x} + \mathbf{b}$. Empiler plusieurs couches linéaires ne fait qu'en produire une autre:

$$
W_L(W_{L-1}(\cdots W_1 \mathbf{x} \cdots)) = (W_L W_{L-1} \cdots W_1) \mathbf{x} = W' \mathbf{x}
$$

La composition de fonctions linéaires est encore linéaire. Les non-linéarités sont ce qui donne aux réseaux de neurones leur pouvoir expressif.

### Fonctions d'activation

La **sigmoïde** devrait vous être familière: c'est la même fonction que nous avons utilisée en régression logistique au chapitre 3:

$$
\sigma(a) = \frac{1}{1 + e^{-a}}
$$

Elle transforme un score réel en une valeur dans $(0, 1)$. Sa dérivée est $\sigma'(a) = \sigma(a)(1 - \sigma(a))$, ce qui sera utile pour la rétropropagation. Cependant, la sigmoïde **sature** pour les grandes valeurs de $|a|$: dans ces régions, la dérivée est proche de zéro.

La **tangente hyperbolique** est similaire mais centrée autour de zéro:

$$
\tanh(a) = \frac{e^a - e^{-a}}{e^a + e^{-a}} = 2\sigma(2a) - 1
$$

Ses sorties sont dans $(-1, 1)$. On peut montrer que $\tanh$ est une version recentrée de la sigmoïde. Elle souffre du même problème de saturation.

L'**unité linéaire rectifiée** (ReLU, de l'anglais *rectified linear unit*) est aujourd'hui la fonction d'activation la plus utilisée:

$$
\text{ReLU}(a) = \max(0, a)
$$

Ses avantages sont sa simplicité de calcul et l'absence de saturation pour les valeurs positives. Sa dérivée vaut 1 pour $a > 0$ et 0 pour $a < 0$. Un inconvénient est que les neurones dont la pré-activation est toujours négative ont un gradient nul et cessent d'apprendre --- c'est le problème des « neurones morts ».

Plusieurs variantes de ReLU existent pour atténuer ce problème. La **Leaky ReLU** utilise une petite pente $\alpha \approx 0{,}01$ pour les valeurs négatives: $\text{LeakyReLU}(a) = \max(\alpha a, a)$. La **GELU** (*Gaussian Error Linear Unit*), définie par $\text{GELU}(a) = a \cdot \Phi(a)$ où $\Phi$ est la fonction de répartition normale, est utilisée dans les architectures modernes comme les transformers.

### Le perceptron multicouche

Un **perceptron multicouche** (MLP, de l'anglais *multilayer perceptron*) compose plusieurs couches de la forme décrite ci-dessus. Pour un réseau à $L$ couches:

$$
\begin{aligned}
\mathbf{z}_0 &= \mathbf{x} \\
\mathbf{a}_\ell &= W_\ell \mathbf{z}_{\ell-1} + \mathbf{b}_\ell \quad \text{pour } \ell = 1, \ldots, L \\
\mathbf{z}_\ell &= \varphi(\mathbf{a}_\ell) \quad \text{pour } \ell = 1, \ldots, L-1
\end{aligned}
$$

L'entrée $\mathbf{x}$ traverse $L-1$ couches cachées, chacune produisant des activations $\mathbf{z}_\ell$. Ces activations sont les **caractéristiques apprises** --- la fonction $\boldsymbol{\phi}(\mathbf{x}; \boldsymbol{\theta}_\phi)$ de l'équation {eq}`eq:nn-key-idea`. La dernière couche produit la sortie du réseau.

### Couche de sortie: le lien avec le maximum de vraisemblance

Le traitement de la dernière couche dépend du problème et de notre choix de distribution conditionnelle:

**Régression** (vraisemblance gaussienne). La couche de sortie est simplement linéaire, sans activation:

$$
\mu(\mathbf{x}) = \mathbf{w}^\top \mathbf{z}_{L-1} + b_L
$$

La perte est la somme des carrés, $\sum_i (y_i - \mu(\mathbf{x}_i))^2$, cohérente avec l'hypothèse de bruit gaussien.

**Classification binaire** (vraisemblance de Bernoulli). La couche de sortie applique une sigmoïde:

$$
\mu(\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{z}_{L-1} + b_L)
$$

La perte est l'entropie croisée binaire, exactement comme en régression logistique.

**Classification multiclasse** (vraisemblance catégorielle). La couche de sortie applique un softmax:

$$
\boldsymbol{\mu}(\mathbf{x}) = \text{softmax}(W_L \mathbf{z}_{L-1} + \mathbf{b}_L)
$$

La perte est l'entropie croisée catégorielle.

Un réseau de neurones pour la classification n'est donc rien d'autre qu'une **régression logistique dont les entrées sont des caractéristiques apprises**. Les couches cachées construisent une représentation $\mathbf{z}_{L-1} = \boldsymbol{\phi}(\mathbf{x}; \boldsymbol{\theta}_\phi)$ dans laquelle le problème devient (idéalement) linéairement séparable, et la dernière couche effectue la classification linéaire.

### Expressivité

Un réseau avec une seule couche cachée suffisamment large peut approximer toute fonction continue sur un ensemble compact. Ce résultat, connu sous le nom de **théorème d'approximation universelle** {cite}`hornik1989multilayer`, garantit l'expressivité théorique des MLP. Cependant, la largeur requise peut croître exponentiellement avec la complexité de la fonction cible. Les réseaux profonds (avec plusieurs couches) peuvent représenter certaines fonctions de manière beaucoup plus compacte que les réseaux larges mais peu profonds.

## Rétropropagation

### Le problème du gradient

Nous savons depuis le chapitre 3 que la descente de gradient (et ses variantes stochastiques) est l'outil d'optimisation standard pour les modèles paramétriques. Pour un réseau de neurones avec paramètres $\boldsymbol{\theta} = \{(W_\ell, \mathbf{b}_\ell)\}_{\ell=1}^L$, nous devons calculer le gradient de la perte par rapport à chaque paramètre:

$$
\nabla_{W_\ell} \mathcal{L}, \quad \nabla_{\mathbf{b}_\ell} \mathcal{L} \quad \text{pour } \ell = 1, \ldots, L
$$

Le défi est que la perte dépend des paramètres de la couche $\ell$ à travers toutes les couches suivantes $\ell+1, \ldots, L$. La **rétropropagation** (*backpropagation*) résout ce problème en appliquant systématiquement la règle de la chaîne.

### La règle de la chaîne pour les compositions

Considérons un réseau comme une composition de fonctions $f = f_L \circ f_{L-1} \circ \cdots \circ f_1$. La jacobienne de cette composition est le produit des jacobiennes individuelles:

$$
\mathbf{J}_f(\mathbf{x}) = \mathbf{J}_{f_L}(\mathbf{z}_{L-1}) \cdot \mathbf{J}_{f_{L-1}}(\mathbf{z}_{L-2}) \cdots \mathbf{J}_{f_1}(\mathbf{x})
$$

où $\mathbf{z}_\ell = f_\ell(\mathbf{z}_{\ell-1})$ sont les valeurs intermédiaires calculées lors de la passe avant. Ce produit de matrices peut être évalué de deux façons, et le choix fait toute la différence.

### Deux façons de multiplier: JVP et VJP

Le produit $\mathbf{J}_f \cdot \mathbf{v}$ d'une jacobienne par un vecteur peut être calculé sans jamais former la jacobienne complète. Selon la direction de multiplication, on obtient deux opérations distinctes.

Le **JVP** (*Jacobian-Vector Product*) propage un vecteur tangent $\mathbf{v}$ de gauche à droite:

$$
\mathbf{J}_f(\mathbf{x}) \, \mathbf{v} = \mathbf{J}_{f_L} \cdot (\mathbf{J}_{f_{L-1}} \cdot (\cdots (\mathbf{J}_{f_1} \cdot \mathbf{v}) \cdots))
$$

Chaque étape multiplie une jacobienne locale par un vecteur, ce qui coûte $O(mn)$ au lieu de $O(m n^2)$ pour le produit par une matrice. Le calcul se fait dans le même sens que la passe avant: c'est le **mode avant** de la différentiation automatique.

Le **VJP** (*Vector-Jacobian Product*) propage un vecteur adjoint $\mathbf{u}^\top$ de droite à gauche:

$$
\mathbf{u}^\top \mathbf{J}_f(\mathbf{x}) = ((\mathbf{u}^\top \cdot \mathbf{J}_{f_L}) \cdot \mathbf{J}_{f_{L-1}}) \cdots \mathbf{J}_{f_1}
$$

Le calcul se fait dans le sens inverse de la passe avant: c'est le **mode arrière**.

Pour une perte scalaire $\mathcal{L}: \mathbb{R}^n \to \mathbb{R}$, le gradient $\nabla_\mathbf{x} \mathcal{L}$ est exactement un VJP avec $\mathbf{u} = 1$. Le mode arrière calcule donc le gradient par rapport à **tous** les paramètres en une seule passe arrière, quel que soit le nombre de paramètres. C'est pourquoi la rétropropagation utilise le mode arrière.

### Exemple: MLP avec une couche cachée

Prenons un réseau à une couche cachée avec la perte des moindres carrés:

$$
\mathcal{L} = \frac{1}{2}\|y - \mathbf{w}_2^\top \varphi(W_1 \mathbf{x} + \mathbf{b}_1) - b_2\|^2
$$

La **passe avant** calcule les valeurs intermédiaires:

$$
\begin{aligned}
\mathbf{a}_1 &= W_1 \mathbf{x} + \mathbf{b}_1 \\
\mathbf{z}_1 &= \varphi(\mathbf{a}_1) \\
\hat{y} &= \mathbf{w}_2^\top \mathbf{z}_1 + b_2 \\
\mathcal{L} &= \frac{1}{2}(y - \hat{y})^2
\end{aligned}
$$

La **passe arrière** propage le gradient en sens inverse, couche par couche:

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \hat{y}} &= \hat{y} - y \\[4pt]
\frac{\partial \mathcal{L}}{\partial \mathbf{w}_2} &= \frac{\partial \mathcal{L}}{\partial \hat{y}} \, \mathbf{z}_1, \qquad
\frac{\partial \mathcal{L}}{\partial b_2} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \\[4pt]
\frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} &= \frac{\partial \mathcal{L}}{\partial \hat{y}} \, \mathbf{w}_2 \\[4pt]
\frac{\partial \mathcal{L}}{\partial \mathbf{a}_1} &= \frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} \odot \varphi'(\mathbf{a}_1) \\[4pt]
\frac{\partial \mathcal{L}}{\partial W_1} &= \frac{\partial \mathcal{L}}{\partial \mathbf{a}_1} \, \mathbf{x}^\top, \qquad
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}_1}
\end{aligned}
$$

où $\odot$ désigne le produit élément par élément. Chaque ligne utilise uniquement des quantités déjà calculées --- soit lors de la passe avant ($\mathbf{z}_1$, $\mathbf{a}_1$, $\mathbf{x}$), soit lors des étapes précédentes de la passe arrière. La structure est toujours la même: le gradient par rapport aux pré-activations d'une couche est propagé vers l'arrière pour obtenir le gradient de la couche précédente.

### L'algorithme général

Pour un réseau à $K$ couches, la rétropropagation suit cet algorithme:

```{prf:algorithm} Rétropropagation
:label: backprop

**Entrée**: Entrée $\mathbf{x}$, cible $y$, paramètres $\{(\boldsymbol{\theta}_k)\}_{k=1}^K$

**Sortie**: Perte $\mathcal{L}$, gradients $\nabla_{\boldsymbol{\theta}_k} \mathcal{L}$ pour $k = 1, \ldots, K$

// Passe avant
1. $\mathbf{z}_0 := \mathbf{x}$
2. Pour $k = 1, \ldots, K$:
   - $\mathbf{z}_k = f_k(\mathbf{z}_{k-1}; \boldsymbol{\theta}_k)$
3. $\mathcal{L} = \ell(\mathbf{z}_K, y)$

// Passe arrière
4. $\mathbf{u}_{K} := \nabla_{\mathbf{z}_K} \ell$
5. Pour $k = K, \ldots, 1$:
   - $\mathbf{g}_k := \mathbf{u}_{k}^\top \frac{\partial f_k}{\partial \boldsymbol{\theta}_k}$ (gradient des paramètres)
   - $\mathbf{u}_{k-1}^\top := \mathbf{u}_{k}^\top \frac{\partial f_k}{\partial \mathbf{z}_{k-1}}$ (propagation de l'adjoint)

6. Retourner $\mathcal{L}$, $\{\nabla_{\boldsymbol{\theta}_k} \mathcal{L} = \mathbf{g}_k^\top\}_{k=1}^K$
```

Le vecteur $\mathbf{u}_k$ est l'**adjoint**: il accumule la sensibilité de la perte aux activations de la couche $k$. À chaque étape, deux produits sont calculés: un pour obtenir le gradient des paramètres de la couche courante, et un pour propager l'adjoint vers la couche précédente.

## Différentiation automatique

### Des chaînes aux graphes de calcul

L'algorithme de rétropropagation que nous venons de décrire suppose une structure en chaîne: chaque couche reçoit la sortie de la précédente. Les architectures modernes sont plus complexes. Les connexions résiduelles, par exemple, ajoutent la sortie d'une couche à celle d'une couche plus profonde. Les architectures avec plusieurs branches ou des opérations de concaténation ne sont pas des chaînes simples.

Ces architectures se représentent naturellement comme des **graphes de calcul**: des graphes orientés acycliques (DAG) où chaque noeud applique une opération élémentaire différentiable. L'entrée du réseau et les paramètres sont les noeuds sources; la perte est le noeud puits.

Dans un DAG, un noeud peut influencer la sortie par plusieurs chemins. La règle de la chaîne se généralise: si un noeud $j$ a des enfants $k_1, \ldots, k_m$, le gradient total est la somme des contributions de chaque chemin:

$$
\frac{\partial \mathcal{L}}{\partial x_j} = \sum_{k \in \text{enfants}(j)} \frac{\partial \mathcal{L}}{\partial x_k} \frac{\partial x_k}{\partial x_j}
$$

La rétropropagation parcourt le graphe dans l'ordre topologique inverse, accumulant ces contributions.

### Trois approches pour calculer des dérivées

Pour calculer la dérivée d'un programme, trois approches existent:

La **différentiation numérique** approxime la dérivée par différences finies:

$$
\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + \epsilon \mathbf{e}_i) - f(\mathbf{x} - \epsilon \mathbf{e}_i)}{2\epsilon}
$$

Cette méthode est simple à implémenter mais souffre de deux problèmes: elle requiert $O(n)$ évaluations de $f$ pour un gradient en dimension $n$, et elle est sujette aux erreurs d'arrondi (le choix de $\epsilon$ est délicat). Elle reste utile pour vérifier des implémentations de gradient.

La **différentiation symbolique** applique les règles de dérivation formellement, comme on le ferait à la main. Elle produit des expressions exactes, mais ces expressions peuvent croître exponentiellement en taille pour des programmes complexes. De plus, elle requiert que le programme soit représenté sous forme symbolique, ce qui exclut les structures de contrôle (boucles, conditions).

La **différentiation automatique** (AD) est une troisième voie. Elle évalue la dérivée exacte d'un programme numérique en le décomposant en opérations élémentaires et en appliquant la règle de la chaîne à chaque étape. Contrairement à la différentiation numérique, elle est exacte (aux erreurs de virgule flottante près). Contrairement à la différentiation symbolique, elle opère sur des valeurs numériques, pas sur des expressions, et gère naturellement les boucles et les conditions.

La rétropropagation n'est rien d'autre que la différentiation automatique en mode arrière, appliquée au programme qui calcule la perte d'un réseau de neurones.

### La programmation différentiable

Les bibliothèques modernes comme JAX, PyTorch et TensorFlow implémentent la différentiation automatique de manière générale: toute fonction composée d'opérations dont on connaît les dérivées locales peut être différentiée automatiquement. C'est le paradigme de la **programmation différentiable** (*differentiable programming*).

L'idée est la suivante: au lieu de dériver manuellement les gradients pour chaque architecture, nous écrivons la passe avant comme un programme ordinaire, et la bibliothèque se charge de calculer les gradients.

Voici un exemple avec JAX. Nous définissons la passe avant d'un MLP à une couche cachée, puis utilisons `jax.grad` pour obtenir automatiquement la fonction qui calcule les gradients:

```{code-cell} python
:tags: [hide-input]
import jax
import jax.numpy as jnp

def predict(params, x):
    """Passe avant d'un MLP à une couche cachée."""
    W1, b1, W2, b2 = params
    h = jnp.tanh(W1 @ x + b1)  # couche cachée
    return W2 @ h + b2           # couche de sortie

def loss_fn(params, x, y):
    """Perte des moindres carrés."""
    y_pred = predict(params, x)
    return 0.5 * jnp.sum((y_pred - y) ** 2)

# Initialisation
key = jax.random.PRNGKey(0)
keys = jax.random.split(key, 4)
params = [
    jax.random.normal(keys[0], (4, 2)) * 0.1,   # W1
    jnp.zeros(4),                                  # b1
    jax.random.normal(keys[1], (1, 4)) * 0.1,    # W2
    jnp.zeros(1),                                  # b2
]

x = jnp.array([1.0, 2.0])
y = jnp.array([0.5])

# jax.grad calcule automatiquement les gradients
grad_fn = jax.grad(loss_fn)
grads = grad_fn(params, x, y)

for name, g in zip(['dL/dW1', 'dL/db1', 'dL/dW2', 'dL/db2'], grads):
    print(f"{name}: forme {g.shape}")
```

La fonction `loss_fn` est un programme Python ordinaire. L'appel `jax.grad(loss_fn)` produit une nouvelle fonction qui calcule le gradient par rapport au premier argument (`params`). Aucune dérivation manuelle n'est nécessaire: JAX applique la règle de la chaîne automatiquement, en mode arrière, sur la trace d'exécution du programme.

Ce paradigme change la façon de penser les modèles. Au lieu de concevoir une architecture puis de dériver ses gradients, on conçoit un programme de calcul quelconque --- avec des boucles, des conditions, des appels de fonctions --- et on le différentie automatiquement. La seule contrainte est que les opérations soient différentiables (ou différentiables presque partout, comme ReLU).

## Considérations pratiques

### Le gradient qui disparaît et le gradient qui explose

Dans un réseau profond, le gradient de la perte par rapport aux premières couches est un produit de jacobiennes locales:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_L} \prod_{\ell=2}^{L} \frac{\partial \mathbf{z}_\ell}{\partial \mathbf{z}_{\ell-1}}
$$

Si les jacobiennes ont un rayon spectral inférieur à 1 (ce qui arrive avec la sigmoïde, dont la dérivée est au plus 0,25), le produit décroît exponentiellement avec la profondeur: le gradient **disparaît** (*vanishing gradient*). Les premières couches reçoivent des signaux de gradient négligeables et cessent d'apprendre.

Inversement, si le rayon spectral est supérieur à 1, le gradient **explose** (*exploding gradient*). Les mises à jour deviennent instables et l'entraînement diverge.

Ces deux problèmes ont longtemps limité la profondeur des réseaux. Plusieurs solutions ont été développées.

### Initialisation des poids

Une bonne initialisation vise à maintenir la variance des activations et des gradients stable à travers les couches. Deux stratégies courantes:

L'**initialisation de Glorot** {cite}`glorot2010understanding` (aussi appelée Xavier) tire les poids $W_\ell \in \mathbb{R}^{m \times n}$ d'une distribution telle que:

$$
W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n + m}\right)
$$

où $n$ et $m$ sont les dimensions d'entrée et de sortie de la couche. Cette stratégie est adaptée aux activations sigmoïde et tanh.

L'**initialisation de He** {cite}`he2015delving`, conçue pour ReLU, utilise une variance plus grande:

$$
W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n}\right)
$$

Le facteur 2 compense le fait que ReLU annule environ la moitié des activations.

### Normalisation par lots

La **normalisation par lots** (*batch normalization*) {cite}`ioffe2015batch` normalise les pré-activations à chaque couche pour qu'elles aient une moyenne nulle et une variance unitaire sur le mini-lot courant:

$$
\hat{a}_j = \frac{a_j - \bar{a}_j}{\sqrt{s_j^2 + \epsilon}}
$$

où $\bar{a}_j$ et $s_j^2$ sont la moyenne et la variance empiriques de la pré-activation $j$ sur le mini-lot, et $\epsilon$ est une petite constante de stabilité. Des paramètres appris $\gamma_j$ et $\beta_j$ permettent ensuite de recalibrer: $\tilde{a}_j = \gamma_j \hat{a}_j + \beta_j$.

Cette technique stabilise l'entraînement en réduisant la dépendance des gradients à l'échelle des activations. Elle permet d'utiliser des taux d'apprentissage plus élevés et agit comme un régularisateur implicite.

### Connexions résiduelles

Les **connexions résiduelles** (*residual connections* ou *skip connections*) {cite}`he2016deep` ajoutent l'entrée d'un bloc à sa sortie:

$$
\mathbf{z}_{\ell+1} = \mathbf{z}_\ell + f(\mathbf{z}_\ell; \boldsymbol{\theta}_\ell)
$$

Au lieu d'apprendre la transformation complète, le bloc $f$ n'apprend que le **résidu** --- la différence entre la sortie désirée et l'entrée. Le gradient se propage directement à travers la connexion identité, ce qui atténue le problème du gradient qui disparaît:

$$
\frac{\partial \mathbf{z}_{\ell+1}}{\partial \mathbf{z}_\ell} = I + \frac{\partial f}{\partial \mathbf{z}_\ell}
$$

La présence du terme identité $I$ garantit que le gradient ne peut pas s'annuler complètement, même si $\frac{\partial f}{\partial \mathbf{z}_\ell}$ est petit. Cette architecture a permis d'entraîner des réseaux de plus de 100 couches.

### Écrêtage du gradient

L'**écrêtage du gradient** (*gradient clipping*) est une technique pragmatique pour empêcher l'explosion du gradient. Avant chaque mise à jour, on limite la norme du gradient:

$$
\mathbf{g}' = \min\left(1, \frac{c}{\|\mathbf{g}\|}\right) \mathbf{g}
$$

Si $\|\mathbf{g}\| > c$, le gradient est réduit pour avoir une norme $c$. Cette opération préserve la direction du gradient tout en bornant son amplitude.

## Résumé

Ce chapitre a montré comment les réseaux de neurones s'inscrivent naturellement dans la progression des modèles vus dans les chapitres précédents. Le point de départ est le même cadre de maximum de vraisemblance: un modèle prédit les paramètres d'une distribution conditionnelle (moyenne pour la régression, probabilité pour la classification). Les modèles linéaires utilisent directement les entrées; l'expansion de caractéristiques transforme les entrées par une fonction fixée; les réseaux de neurones apprennent cette transformation.

La rétropropagation applique la règle de la chaîne systématiquement pour calculer les gradients dans un réseau profond. Le mode arrière (VJP) est efficace pour les fonctions à sortie scalaire, ce qui est le cas de toutes les fonctions de perte. La différentiation automatique généralise ce principe à tout programme composé d'opérations différentiables.

Les chapitres suivants présenteront des architectures spécialisées qui exploitent la structure des données: les réseaux convolutifs pour les images et les réseaux récurrents pour les séquences.

## Exercices

````{admonition} Exercice 1: Composition linéaire ★
:class: hint dropdown

Montrez que la composition de deux transformations affines $\mathbf{z}_1 = W_1 \mathbf{x} + \mathbf{b}_1$ et $\mathbf{z}_2 = W_2 \mathbf{z}_1 + \mathbf{b}_2$ est une transformation affine. Trouvez la matrice $W'$ et le vecteur $\mathbf{b}'$ tels que $\mathbf{z}_2 = W' \mathbf{x} + \mathbf{b}'$.

Que conclure sur l'utilité d'un réseau à plusieurs couches sans fonctions d'activation?
````

````{admonition} Solution Exercice 1
:class: dropdown

En substituant:

$$
\mathbf{z}_2 = W_2(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (W_2 W_1) \mathbf{x} + (W_2 \mathbf{b}_1 + \mathbf{b}_2)
$$

Donc $W' = W_2 W_1$ et $\mathbf{b}' = W_2 \mathbf{b}_1 + \mathbf{b}_2$. Un réseau à plusieurs couches linéaires sans activation n'est pas plus expressif qu'un modèle linéaire à une seule couche.
````

````{admonition} Exercice 2: Dérivée de la sigmoïde ★
:class: hint dropdown

Montrez que la dérivée de la fonction sigmoïde $\sigma(a) = \frac{1}{1 + e^{-a}}$ s'écrit $\sigma'(a) = \sigma(a)(1 - \sigma(a))$.

Calculez la valeur maximale de $\sigma'(a)$ et identifiez en quel point elle est atteinte.
````

````{admonition} Solution Exercice 2
:class: dropdown

En posant $\sigma(a) = (1 + e^{-a})^{-1}$:

$$
\sigma'(a) = \frac{e^{-a}}{(1 + e^{-a})^2} = \frac{1}{1 + e^{-a}} \cdot \frac{e^{-a}}{1 + e^{-a}} = \sigma(a)(1 - \sigma(a))
$$

Le maximum de $\sigma'(a) = \sigma(a)(1 - \sigma(a))$ est atteint quand $\sigma(a) = 0{,}5$, soit $a = 0$. La valeur maximale est $0{,}5 \times 0{,}5 = 0{,}25$.

Ce maximum de 0,25 explique le gradient qui disparaît: à chaque couche utilisant la sigmoïde, le gradient est multiplié par un facteur d'au plus 0,25.
````

````{admonition} Exercice 3: Rétropropagation manuelle ★★
:class: hint dropdown

Considérez un réseau à une couche cachée avec 2 neurones ReLU, une entrée scalaire $x = 1$, une cible $y = 1$, et la perte des moindres carrés.

Les paramètres sont: $\mathbf{w}_1 = [0{,}5, -0{,}3]^\top$, $\mathbf{b}_1 = [0, 0]^\top$, $\mathbf{w}_2 = [0{,}8, 0{,}4]^\top$, $b_2 = 0$.

1. Calculez la passe avant: pré-activations, activations, prédiction, perte.
2. Calculez la passe arrière: tous les gradients.
3. Effectuez une mise à jour des paramètres avec un taux d'apprentissage $\eta = 0{,}1$.
````

````{admonition} Solution Exercice 3
:class: dropdown

**Passe avant:**

$$
\begin{aligned}
\mathbf{a}_1 &= \mathbf{w}_1 x + \mathbf{b}_1 = [0{,}5, -0{,}3]^\top \\
\mathbf{z}_1 &= \text{ReLU}(\mathbf{a}_1) = [0{,}5, 0]^\top \\
\hat{y} &= \mathbf{w}_2^\top \mathbf{z}_1 + b_2 = 0{,}8 \times 0{,}5 + 0{,}4 \times 0 = 0{,}4 \\
\mathcal{L} &= \frac{1}{2}(1 - 0{,}4)^2 = 0{,}18
\end{aligned}
$$

**Passe arrière:**

$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \hat{y}} &= 0{,}4 - 1 = -0{,}6 \\
\frac{\partial \mathcal{L}}{\partial \mathbf{w}_2} &= -0{,}6 \times [0{,}5, 0]^\top = [-0{,}3, 0]^\top \\
\frac{\partial \mathcal{L}}{\partial b_2} &= -0{,}6 \\
\frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} &= -0{,}6 \times [0{,}8, 0{,}4]^\top = [-0{,}48, -0{,}24]^\top \\
\frac{\partial \mathcal{L}}{\partial \mathbf{a}_1} &= [-0{,}48, -0{,}24]^\top \odot [1, 0]^\top = [-0{,}48, 0]^\top \\
\frac{\partial \mathcal{L}}{\partial \mathbf{w}_1} &= [-0{,}48, 0]^\top \times 1 = [-0{,}48, 0]^\top \\
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} &= [-0{,}48, 0]^\top
\end{aligned}
$$

Le deuxième neurone a un gradient nul car ReLU($-0{,}3$) = 0 et sa dérivée est 0.

**Mise à jour ($\eta = 0{,}1$):**

$$
\begin{aligned}
\mathbf{w}_1 &\leftarrow [0{,}5, -0{,}3]^\top - 0{,}1 \times [-0{,}48, 0]^\top = [0{,}548, -0{,}3]^\top \\
\mathbf{w}_2 &\leftarrow [0{,}8, 0{,}4]^\top - 0{,}1 \times [-0{,}3, 0]^\top = [0{,}83, 0{,}4]^\top \\
b_2 &\leftarrow 0 - 0{,}1 \times (-0{,}6) = 0{,}06
\end{aligned}
$$
````

````{admonition} Exercice 4: Vérification numérique du gradient ★★
:class: hint dropdown

Implémentez un MLP à une couche cachée avec ReLU en NumPy. Calculez les gradients par rétropropagation, puis vérifiez-les par différences finies centrées ($\epsilon = 10^{-5}$). L'erreur relative entre les deux devrait être inférieure à $10^{-5}$.

```python
import numpy as np

def numerical_gradient(f, params, eps=1e-5):
    """Calcule le gradient par différences finies centrées."""
    grads = []
    for p in params:
        grad_p = np.zeros_like(p)
        it = np.nditer(p, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            old_val = p[idx]
            p[idx] = old_val + eps
            loss_plus = f()
            p[idx] = old_val - eps
            loss_minus = f()
            grad_p[idx] = (loss_plus - loss_minus) / (2 * eps)
            p[idx] = old_val
            it.iternext()
        grads.append(grad_p)
    return grads
```

Conseil: calculez l'erreur relative comme $\frac{\|\mathbf{g}_\text{backprop} - \mathbf{g}_\text{num}\|}{\|\mathbf{g}_\text{backprop}\| + \|\mathbf{g}_\text{num}\| + \epsilon}$.
````

````{admonition} Solution Exercice 4
:class: dropdown

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

# Paramètres
np.random.seed(42)
W1 = np.random.randn(4, 3) * 0.1
b1 = np.zeros(4)
W2 = np.random.randn(1, 4) * 0.1
b2 = np.zeros(1)
params = [W1, b1, W2, b2]

x = np.random.randn(3)
y = np.array([1.0])

# Passe avant et arrière
def forward_and_loss():
    a1 = W1 @ x + b1
    z1 = relu(a1)
    y_pred = W2 @ z1 + b2
    return 0.5 * np.sum((y_pred - y) ** 2)

def backprop():
    a1 = W1 @ x + b1
    z1 = relu(a1)
    y_pred = W2 @ z1 + b2
    
    dL_dy = y_pred - y
    dL_dW2 = np.outer(dL_dy, z1)
    dL_db2 = dL_dy
    dL_dz1 = W2.T @ dL_dy
    dL_da1 = dL_dz1 * (a1 > 0).astype(float)
    dL_dW1 = np.outer(dL_da1, x)
    dL_db1 = dL_da1
    
    return [dL_dW1, dL_db1, dL_dW2, dL_db2]

grads_bp = backprop()
grads_num = numerical_gradient(forward_and_loss, params)

for name, g_bp, g_num in zip(['W1', 'b1', 'W2', 'b2'], grads_bp, grads_num):
    err = np.linalg.norm(g_bp - g_num) / (np.linalg.norm(g_bp) + np.linalg.norm(g_num) + 1e-8)
    print(f"{name}: erreur relative = {err:.2e}")
```

Toutes les erreurs relatives devraient être inférieures à $10^{-5}$.
````

````{admonition} Exercice 5: Gradient qui disparaît ★★★
:class: hint dropdown

Cet exercice explore le problème du gradient qui disparaît. Considérez un réseau de $L$ couches, chacune avec une seule unité sigmoïde et un poids $w_\ell$:

$$
z_\ell = \sigma(w_\ell z_{\ell-1}), \quad z_0 = x
$$

1. Montrez que $\frac{\partial z_L}{\partial w_1} = \prod_{\ell=2}^{L} w_\ell \sigma'(a_\ell) \cdot \sigma'(a_1) x$.

2. Si tous les poids sont $w_\ell = 1$ et toutes les pré-activations sont au point optimal $a_\ell = 0$ (où $\sigma'$ est maximale), quelle est la borne supérieure de $|\frac{\partial z_L}{\partial w_1}|$ en fonction de $L$?

3. Pour $L = 20$, calculez cette borne. Que conclure?

4. Répétez l'analyse avec ReLU. Que change-t-il?
````

````{admonition} Solution Exercice 5
:class: dropdown

1. Par la règle de la chaîne:

$$
\frac{\partial z_L}{\partial w_1} = \frac{\partial z_L}{\partial z_{L-1}} \cdot \frac{\partial z_{L-1}}{\partial z_{L-2}} \cdots \frac{\partial z_2}{\partial z_1} \cdot \frac{\partial z_1}{\partial w_1}
$$

Avec $\frac{\partial z_\ell}{\partial z_{\ell-1}} = w_\ell \sigma'(a_\ell)$ et $\frac{\partial z_1}{\partial w_1} = \sigma'(a_1) x$, on obtient le résultat.

2. Le maximum de $\sigma'(a)$ est 0,25 (atteint en $a = 0$). Si $w_\ell = 1$ et $a_\ell = 0$:

$$
\left|\frac{\partial z_L}{\partial w_1}\right| \leq (0{,}25)^{L-1} \cdot |x|
$$

3. Pour $L = 20$: $(0{,}25)^{19} \approx 3{,}6 \times 10^{-12}$. Le gradient est essentiellement nul.

4. Avec ReLU, $\text{ReLU}'(a) = 1$ pour $a > 0$. Si toutes les pré-activations sont positives:

$$
\frac{\partial z_L}{\partial w_1} = \prod_{\ell=2}^{L} w_\ell \cdot x
$$

Le gradient ne disparaît pas (mais peut exploser si $|w_\ell| > 1$). C'est l'une des raisons du succès de ReLU.
````

