---
kernelspec:
  name: python3
  display_name: Python 3
---

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

Aux chapitres 2 et 3, nous avons construit des modèles linéaires pour la régression et la classification. Au chapitre 4, nous avons vu comment enrichir ces modèles en transformant les entrées par une fonction $\boldsymbol{\phi}$ fixée à l'avance. Ce chapitre franchit une étape supplémentaire: au lieu de choisir $\boldsymbol{\phi}$ manuellement, nous allons l'apprendre à partir des données. Cette idée conduit aux réseaux de neurones.

Dans ce chapitre, nous commençons par rappeler le cadre probabiliste qui unifie régression et classification, puis nous montrons comment la volonté de dépasser les modèles linéaires mène aux réseaux de neurones. Nous présentons ensuite l'anatomie d'un réseau (couches, activations, architecture). La section sur la rétropropagation est plus technique: elle dérive l'algorithme de calcul des gradients pas à pas. Vous pouvez survoler les détails en première lecture et retenir le mécanisme général. Nous terminons par la différentiation automatique et les considérations pratiques pour l'entraînement.

## Le cadre unifié: prédire les paramètres d'une distribution

### Régression et classification comme maximum de vraisemblance

Revenons au cadre probabiliste des chapitres 2 et 5. Dans tous les modèles que nous avons vus, le problème d'apprentissage supervisé prend la même forme: étant donné une entrée $\mathbf{x}$, nous voulons prédire les paramètres d'une distribution conditionnelle $p(y | \mathbf{x}; \boldsymbol{\theta})$, puis trouver $\boldsymbol{\theta}$ par maximum de vraisemblance.

En régression, nous avons supposé un bruit gaussien:

$$
p(y | \mathbf{x}; \boldsymbol{\theta}) = \mathcal{N}(y \,|\, \mu(\mathbf{x}), \sigma^2)
$$

Le modèle prédit la moyenne $\mu(\mathbf{x})$ de la distribution. La log-vraisemblance négative donne, à une constante près, la perte des moindres carrés:

$$
-\sum_{i=1}^N \log p(y_i | \mathbf{x}_i; \boldsymbol{\theta}) \propto \sum_{i=1}^N (y_i - \mu(\mathbf{x}_i))^2
$$

En classification binaire, nous avons supposé une distribution de Bernoulli:

$$
p(y | \mathbf{x}; \boldsymbol{\theta}) = \mu(\mathbf{x})^y (1 - \mu(\mathbf{x}))^{1-y}
$$

Le modèle prédit la probabilité $\mu(\mathbf{x}) = p(y = 1 | \mathbf{x})$. La log-vraisemblance négative donne l'entropie croisée binaire. Pour la classification multiclasse, la distribution catégorielle et la fonction softmax jouent le même rôle, avec l'entropie croisée catégorielle comme perte.

Dans chaque cas, une fonction $\mu(\mathbf{x})$ prend une entrée et produit les paramètres de la distribution de sortie. Toute la question est: quelle forme donner à cette fonction?

### Des modèles linéaires aux caractéristiques apprises

Jusqu'ici, nos modèles ont été linéaires dans les entrées. Pour la régression:

$$
\mu(\mathbf{x}) = \boldsymbol{\theta}^\top \mathbf{x}
$$

Pour la classification binaire, la probabilité passe par une sigmoïde, mais la pré-activation reste linéaire:

$$
\mu(\mathbf{x}) = \sigma(\boldsymbol{\theta}^\top \mathbf{x})
$$

Au chapitre 4, nous avons étendu cette approche avec l'expansion de caractéristiques. Au lieu d'utiliser $\mathbf{x}$ directement, nous le transformons par une fonction $\boldsymbol{\phi}: \mathbb{R}^d \to \mathbb{R}^D$ choisie à l'avance (polynômes, fonctions trigonométriques, bases radiales, etc.):

$$
\mu(\mathbf{x}) = \boldsymbol{\theta}^\top \boldsymbol{\phi}(\mathbf{x})
$$

Le modèle reste linéaire dans les paramètres $\boldsymbol{\theta}$, ce qui facilite l'optimisation, mais il capture des relations non linéaires en $\mathbf{x}$ grâce au choix de $\boldsymbol{\phi}$.

Cette approche a une limite importante: le choix de $\boldsymbol{\phi}$ repose entièrement sur l'expertise du praticien. Pour des données tabulaires simples, cela peut fonctionner. Mais pour des images, du texte ou de l'audio, concevoir manuellement les bonnes caractéristiques est très difficile, et souvent le facteur limitant de la performance.

Les réseaux de neurones paramètrent $\boldsymbol{\phi}$ et l'apprennent à partir des données. Au lieu d'écrire $\boldsymbol{\theta}^\top \boldsymbol{\phi}(\mathbf{x})$ avec $\boldsymbol{\phi}$ fixé, nous écrivons:

$$
\mu(\mathbf{x}) = \mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x}; \boldsymbol{\theta}_\phi)
$$ (eq:nn-key-idea)

où $\boldsymbol{\phi}(\cdot; \boldsymbol{\theta}_\phi)$ est elle-même une fonction paramétrique. Les paramètres $\boldsymbol{\theta}_\phi$ contrôlent la transformation des entrées (les "caractéristiques apprises"), tandis que $\mathbf{w}$ sont les poids de la couche de sortie. On optimise les deux simultanément: le modèle apprend la représentation et le prédicteur en même temps.

Cela soulève deux questions: quelle forme donner à $\boldsymbol{\phi}(\cdot; \boldsymbol{\theta}_\phi)$, et comment optimiser l'ensemble des paramètres? Le reste de ce chapitre répond à ces deux questions.

## Anatomie d'un réseau de neurones

### Un neurone: transformation affine et non-linéarité

Un réseau de neurones est construit à partir d'une opération élémentaire: une transformation affine suivie d'une fonction non linéaire. Pour une entrée $\mathbf{x} \in \mathbb{R}^d$:

$$
h = \varphi(\mathbf{w}^\top \mathbf{x} + b)
$$

où $\mathbf{w} \in \mathbb{R}^d$ est un vecteur de poids, $b \in \mathbb{R}$ est un biais, et $\varphi: \mathbb{R} \to \mathbb{R}$ est une **fonction d'activation** non linéaire. La quantité $a = \mathbf{w}^\top \mathbf{x} + b$ est la **pré-activation** et $h$ est l'**activation** du neurone.

Une couche de $m$ neurones applique cette opération en parallèle, ce qui s'écrit sous forme matricielle:

$$
\mathbf{h} = \varphi(W \mathbf{x} + \mathbf{b})
$$

où $W \in \mathbb{R}^{m \times d}$ est la matrice de poids, $\mathbf{b} \in \mathbb{R}^m$ le vecteur de biais, et $\varphi$ est appliquée élément par élément.

On peut représenter cette couche comme un graphe de calcul: les entrées et paramètres sont les nœuds sources, les opérations ($\times$, $+$, $\varphi$) sont des transformations, et l'activation $\mathbf{z}$ est le nœud de sortie. Cette perspective sera centrale dans la section sur la différentiation automatique.

```{mermaid}
graph LR
    x("x")
    W("W")
    b("b")
    a("a = Wx + b")
    z("z = φ(a)")

    x --> a
    W --> a
    b --> a
    a -- "φ" --> z

    style W fill:#fff2cc,stroke:#d6b656
    style b fill:#fff2cc,stroke:#d6b656
    style x fill:#dae8fc,stroke:#6c8ebf
    style a fill:#f5f5f5,stroke:#666666
    style z fill:#d5e8d4,stroke:#82b366
```

Les nœuds en jaune ($W$, $\mathbf{b}$) sont les **paramètres** (feuilles du graphe); le nœud bleu ($\mathbf{x}$) est l'entrée; le nœud vert ($\mathbf{z}$) est la sortie de la couche.

### Rôle de la non-linéarité

Sans la fonction d'activation $\varphi$, une couche se réduit à une transformation affine $\mathbf{h} = W\mathbf{x} + \mathbf{b}$. Empiler plusieurs couches linéaires ne fait qu'en produire une autre:

$$
W_L(W_{L-1}(\cdots W_1 \mathbf{x} \cdots)) = (W_L W_{L-1} \cdots W_1) \mathbf{x} = W' \mathbf{x}
$$

La composition de fonctions linéaires est encore linéaire. Les non-linéarités sont ce qui donne aux réseaux de neurones leur pouvoir expressif.

### Fonctions d'activation

Nous avons déjà rencontré la **sigmoïde** en régression logistique au chapitre 3:

$$
\sigma(a) = \frac{1}{1 + e^{-a}}
$$

Elle transforme un score réel en une valeur dans $(0, 1)$. Sa dérivée est $\sigma'(a) = \sigma(a)(1 - \sigma(a))$, ce qui sera utile pour la rétropropagation. Cependant, la sigmoïde sature pour les grandes valeurs de $|a|$: dans ces régions, la dérivée est proche de zéro.

La **tangente hyperbolique** est similaire mais centrée autour de zéro:

$$
\tanh(a) = \frac{e^a - e^{-a}}{e^a + e^{-a}} = 2\sigma(2a) - 1
$$

Ses sorties sont dans $(-1, 1)$. On peut montrer que $\tanh$ est une version recentrée de la sigmoïde. Elle souffre du même problème de saturation.

L'**unité linéaire rectifiée** (ReLU, de l'anglais *rectified linear unit*) est aujourd'hui la fonction d'activation la plus utilisée:

$$
\text{ReLU}(a) = \max(0, a)
$$

Ses avantages sont sa simplicité de calcul et l'absence de saturation pour les valeurs positives. Sa dérivée vaut 1 pour $a > 0$ et 0 pour $a < 0$. Un inconvénient est que les neurones dont la pré-activation est toujours négative ont un gradient nul et cessent d'apprendre: c'est le problème des « neurones morts ».

Plusieurs variantes de ReLU existent pour atténuer ce problème. La **Leaky ReLU** utilise une petite pente $\alpha \approx 0{,}01$ pour les valeurs négatives: $\text{LeakyReLU}(a) = \max(\alpha a, a)$. La **GELU** (*Gaussian Error Linear Unit*), définie par $\text{GELU}(a) = a \cdot \Phi(a)$ où $\Phi$ est la fonction de répartition normale, est utilisée dans les architectures modernes comme les transformers.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
%config InlineBackend.figure_format = 'retina'

a = np.linspace(-4, 4, 400)

sigmoid = lambda x: 1 / (1 + np.exp(-x))
gelu    = lambda x: x * 0.5 * (1 + erf(x / np.sqrt(2)))

d_sigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x))
d_tanh    = lambda x: 1 - np.tanh(x)**2
d_relu    = lambda x: (x > 0).astype(float)
eps = 1e-5
d_gelu    = lambda x: (gelu(x + eps) - gelu(x - eps)) / (2 * eps)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
ax.plot(a, sigmoid(a),       'C0', linewidth=2, label='Sigmoïde')
ax.plot(a, np.tanh(a),       'C1', linewidth=2, label='Tanh')
ax.plot(a, np.maximum(0, a), 'C2', linewidth=2, label='ReLU')
ax.plot(a, gelu(a),          'C3', linewidth=2, label='GELU', linestyle='--')
ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
ax.axvline(0, color='k', linewidth=0.5, linestyle=':')
ax.set_xlabel('$a$ (pré-activation)')
ax.set_ylabel('$\\varphi(a)$')
ax.set_title("Fonctions d'activation")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-4, 4)

ax = axes[1]
ax.plot(a, d_sigmoid(a), 'C0', linewidth=2, label="$\\sigma'(a)$")
ax.plot(a, d_tanh(a),    'C1', linewidth=2, label="$\\tanh'(a)$")
ax.plot(a, d_relu(a),    'C2', linewidth=2, label="ReLU$'(a)$")
ax.plot(a, d_gelu(a),    'C3', linewidth=2, label="GELU$'(a)$", linestyle='--')
ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
ax.axvline(0, color='k', linewidth=0.5, linestyle=':')
ax.annotate(
    "$\\sigma'(0) = 0{,}25$",
    xy=(0, 0.25), xytext=(1.3, 0.42),
    arrowprops=dict(arrowstyle='->', color='C0', lw=1.5),
    fontsize=9, color='C0'
)
ax.set_xlabel('$a$ (pré-activation)')
ax.set_ylabel("$\\varphi'(a)$")
ax.set_title('Dérivées des fonctions d\'activation')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-4, 4)
ax.set_ylim(-0.1, 1.1)

plt.tight_layout()
```

La dérivée de la sigmoïde est bornée par 0,25: à chaque couche, le gradient est multiplié par un facteur d'au plus 0,25. Ce phénomène de **saturation** est la cause principale du gradient qui disparaît dans les réseaux profonds (section "Instabilité du gradient en profondeur").

### Le perceptron multicouche

Un **perceptron multicouche** (MLP, de l'anglais *multilayer perceptron*) compose plusieurs couches de la forme décrite ci-dessus. Pour un réseau à $L$ couches:

$$
\begin{aligned}
\mathbf{z}_0 &= \mathbf{x} \\
\mathbf{a}_\ell &= W_\ell \mathbf{z}_{\ell-1} + \mathbf{b}_\ell \quad \text{pour } \ell = 1, \ldots, L \\
\mathbf{z}_\ell &= \varphi(\mathbf{a}_\ell) \quad \text{pour } \ell = 1, \ldots, L-1
\end{aligned}
$$

L'entrée $\mathbf{x}$ traverse $L-1$ couches cachées, chacune produisant des activations $\mathbf{z}_\ell$. Ces activations sont les caractéristiques apprises, c'est-à-dire la fonction $\boldsymbol{\phi}(\mathbf{x}; \boldsymbol{\theta}_\phi)$ de l'équation {eq}`eq:nn-key-idea`. La dernière couche produit la sortie du réseau.

Le graphe de calcul d'un MLP à deux couches cachées est une chaîne d'opérations: chaque couche correspond à une transformation affine suivie d'une non-linéarité.

```{mermaid}
graph LR
    x("x")
    p1("W1, b1")
    a1("a1")
    z1("z1")
    p2("W2, b2")
    a2("a2")
    z2("z2")
    pL("WL, bL")
    yhat("y-hat")

    x --> a1
    p1 --> a1
    a1 -- "φ" --> z1
    z1 --> a2
    p2 --> a2
    a2 -- "φ" --> z2
    z2 --> yhat
    pL --> yhat

    style p1 fill:#fff2cc,stroke:#d6b656
    style p2 fill:#fff2cc,stroke:#d6b656
    style pL fill:#fff2cc,stroke:#d6b656
    style x fill:#dae8fc,stroke:#6c8ebf
    style yhat fill:#d5e8d4,stroke:#82b366
    style a1 fill:#f5f5f5,stroke:#666666
    style a2 fill:#f5f5f5,stroke:#666666
```

Ce graphe est une structure de données: il encode toutes les dépendances entre variables. La rétropropagation consiste à le parcourir en sens inverse pour calculer les gradients par rapport à chaque paramètre.

### Couche de sortie: le lien avec le maximum de vraisemblance

Le traitement de la dernière couche dépend du problème et de notre choix de distribution conditionnelle:

Pour la régression (vraisemblance gaussienne), la couche de sortie est linéaire, sans activation:

$$
\mu(\mathbf{x}) = \mathbf{w}^\top \mathbf{z}_{L-1} + b_L
$$

La perte est la somme des carrés, $\sum_i (y_i - \mu(\mathbf{x}_i))^2$, cohérente avec l'hypothèse de bruit gaussien.

Pour la classification binaire (vraisemblance de Bernoulli), la couche de sortie applique une sigmoïde:

$$
\mu(\mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{z}_{L-1} + b_L)
$$

La perte est l'entropie croisée binaire, exactement comme en régression logistique.

Pour la classification multiclasse (vraisemblance catégorielle), la couche de sortie applique un softmax:

$$
\boldsymbol{\mu}(\mathbf{x}) = \text{softmax}(W_L \mathbf{z}_{L-1} + \mathbf{b}_L)
$$

La perte est l'entropie croisée catégorielle.

Un réseau de neurones pour la classification est donc une régression logistique dont les entrées sont des caractéristiques apprises. Les couches cachées construisent une représentation $\mathbf{z}_{L-1} = \boldsymbol{\phi}(\mathbf{x}; \boldsymbol{\theta}_\phi)$ dans laquelle le problème devient (idéalement) linéairement séparable, et la dernière couche effectue la classification linéaire.

### Expressivité

Un réseau avec une seule couche cachée suffisamment large peut approximer toute fonction continue sur un ensemble compact. Ce résultat, connu sous le nom de **théorème d'approximation universelle** {cite}`hornik1989multilayer`, garantit l'expressivité théorique des MLP. Cependant, la largeur requise peut croître exponentiellement avec la complexité de la fonction cible. Les réseaux profonds (avec plusieurs couches) peuvent représenter certaines fonctions de manière beaucoup plus compacte que les réseaux larges mais peu profonds.

La figure ci-dessous illustre cette propriété: un réseau peu profond mais large et un réseau profond mais étroit approximent tous deux la même fonction, mais avec des complexités très différentes.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

np.random.seed(0)

def relu(x):
    return np.maximum(0, x)

def forward_shallow(x, W1, b1, W2, b2):
    """Réseau large: 1 couche cachée, beaucoup de neurones."""
    h = relu(x[:, None] * W1[None, :] + b1[None, :])
    return h @ W2 + b2

def forward_deep(x, params):
    """Réseau profond: plusieurs couches, peu de neurones."""
    h = x[:, None]
    for W, b in params[:-1]:
        h = relu(h @ W + b)
    W, b = params[-1]
    return (h @ W + b).ravel()

# Cible: fonction non triviale
x_grid = np.linspace(0, 1, 200)
f_target = np.sin(2 * np.pi * x_grid) + 0.5 * np.sin(6 * np.pi * x_grid)

# Réseau peu profond, large (1 couche cachée, 40 neurones)
n_wide = 40
W1_s = np.random.randn(n_wide) * 3
b1_s = np.random.randn(n_wide)
# Ajuster W2 par pseudoinverse pour approximer la cible
H_s = relu(x_grid[:, None] * W1_s[None, :] + b1_s[None, :])
W2_s, _, _, _ = np.linalg.lstsq(
    np.column_stack([H_s, np.ones(len(x_grid))]),
    f_target, rcond=None
)
b2_s = W2_s[-1]
W2_s = W2_s[:-1]
pred_shallow = H_s @ W2_s + b2_s

# Réseau profond, étroit (4 couches cachées, 8 neurones)
n_deep = 8
params_deep = []
d_in = 1
for layer in range(4):
    W = np.random.randn(d_in, n_deep) * np.sqrt(2 / d_in)
    b = np.zeros(n_deep)
    params_deep.append((W, b))
    d_in = n_deep
W_out = np.random.randn(d_in, 1) * np.sqrt(2 / d_in)
b_out = np.zeros(1)
params_deep.append((W_out, b_out))

# Ajuster la dernière couche par pseudoinverse
h = x_grid[:, None]
for W, b in params_deep[:-1]:
    h = relu(h @ W + b)
W_out_fit, _, _, _ = np.linalg.lstsq(
    np.column_stack([h, np.ones(len(x_grid))]),
    f_target, rcond=None
)
b_out_fit = W_out_fit[-1]
W_out_fit = W_out_fit[:-1, None]
params_deep[-1] = (W_out_fit, np.array([b_out_fit]))

h = x_grid[:, None]
for W, b in params_deep[:-1]:
    h = relu(h @ W + b)
W_f, b_f = params_deep[-1]
pred_deep = (h @ W_f + b_f).ravel()

# Figure
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, pred, title, n_params in zip(
    axes,
    [pred_shallow, pred_deep],
    [f'Peu profond, large\n(1 couche cachée, {n_wide} neurones)',
     f'Profond, étroit\n(4 couches cachées, {n_deep} neurones)'],
    [n_wide * 2 + n_wide + 1, 1 * n_deep + n_deep + 3 * n_deep * n_deep + n_deep + n_deep + 1]
):
    ax.plot(x_grid, f_target, 'k--', linewidth=2, label='Cible $f(x)$', alpha=0.7)
    ax.plot(x_grid, pred, 'C0', linewidth=2, label='Approximation')
    ax.set_xlabel('$x$')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel('$f(x)$')
plt.suptitle("Théorème d'approximation universelle: deux architectures, une même fonction", fontsize=11)
plt.tight_layout()
```

Les deux architectures approximent raisonnablement bien la même fonction cible. La différence réside dans l'organisation des paramètres: le réseau profond compose des représentations intermédiaires hiérarchiques, ce qui lui permet d'être plus compact pour des fonctions structurées.

À ce stade, vous avez vu la structure d'un réseau de neurones: des couches qui alternent transformations linéaires et non-linéarités, avec une couche de sortie adaptée au problème (régression ou classification). La question suivante est: comment apprendre les paramètres?

## Rétropropagation

Cette section est plus technique que les précédentes. Nous y dérivons l'algorithme de rétropropagation pas à pas, en utilisant la règle de la chaîne et les produits jacobien-vecteur. Si les détails des jacobiennes semblent abstraits au premier abord, concentrez-vous sur l'exemple travaillé (section "Exemple: MLP avec une couche cachée") qui rend ces idées concrètes. Vous pouvez aussi sauter directement à la section sur la différentiation automatique et revenir ici plus tard: en pratique, les bibliothèques modernes calculent les gradients automatiquement.

### Le problème du gradient

Nous savons depuis le chapitre 3 que la descente de gradient (et ses variantes stochastiques) est l'outil d'optimisation standard pour les modèles paramétriques. Pour un réseau de neurones avec paramètres $\boldsymbol{\theta} = \{(W_\ell, \mathbf{b}_\ell)\}_{\ell=1}^L$, nous devons calculer le gradient de la perte par rapport à chaque paramètre:

$$
\nabla_{W_\ell} \mathcal{L}, \quad \nabla_{\mathbf{b}_\ell} \mathcal{L} \quad \text{pour } \ell = 1, \ldots, L
$$

Le défi est que la perte dépend des paramètres de la couche $\ell$ à travers toutes les couches suivantes $\ell+1, \ldots, L$. La rétropropagation (*backpropagation*) résout ce problème en appliquant systématiquement la règle de la chaîne.

### La règle de la chaîne pour les compositions

Considérons un réseau comme une composition de fonctions $f = f_L \circ f_{L-1} \circ \cdots \circ f_1$. La jacobienne de cette composition est le produit des jacobiennes individuelles:

$$
\mathbf{J}_f(\mathbf{x}) = \mathbf{J}_{f_L}(\mathbf{z}_{L-1}) \cdot \mathbf{J}_{f_{L-1}}(\mathbf{z}_{L-2}) \cdots \mathbf{J}_{f_1}(\mathbf{x})
$$

où $\mathbf{z}_\ell = f_\ell(\mathbf{z}_{\ell-1})$ sont les valeurs intermédiaires calculées lors de la passe avant. Ce produit de matrices peut être évalué de deux façons, et le choix fait toute la différence.

### Produits jacobien-vecteur

Le produit $\mathbf{J}_f \cdot \mathbf{v}$ d'une jacobienne par un vecteur peut être calculé sans jamais former la jacobienne complète. Selon la direction de multiplication, on obtient deux opérations distinctes.

Le JVP (*Jacobian-Vector Product*) propage un vecteur tangent $\mathbf{v}$ de gauche à droite:

$$
\mathbf{J}_f(\mathbf{x}) \, \mathbf{v} = \mathbf{J}_{f_L} \cdot (\mathbf{J}_{f_{L-1}} \cdot (\cdots (\mathbf{J}_{f_1} \cdot \mathbf{v}) \cdots))
$$

Chaque étape multiplie une jacobienne locale par un vecteur, ce qui coûte $O(mn)$ au lieu de $O(m n^2)$ pour le produit par une matrice. Le calcul se fait dans le même sens que la passe avant: c'est le mode avant de la différentiation automatique.

Le VJP (*Vector-Jacobian Product*) propage un vecteur adjoint $\mathbf{u}^\top$ de droite à gauche:

$$
\mathbf{u}^\top \mathbf{J}_f(\mathbf{x}) = ((\mathbf{u}^\top \cdot \mathbf{J}_{f_L}) \cdot \mathbf{J}_{f_{L-1}}) \cdots \mathbf{J}_{f_1}
$$

Le calcul se fait dans le sens inverse de la passe avant: c'est le mode arrière.

Pour une perte scalaire $\mathcal{L}: \mathbb{R}^n \to \mathbb{R}$, le gradient $\nabla_\mathbf{x} \mathcal{L}$ est exactement un VJP avec $\mathbf{u} = 1$. Le mode arrière calcule donc le gradient par rapport à tous les paramètres en une seule passe arrière, quel que soit le nombre de paramètres. C'est pourquoi la rétropropagation utilise le mode arrière.

La figure ci-dessous contraste les deux modes sur une chaîne de trois fonctions $f_1 \circ f_2 \circ f_3$. Le mode avant (JVP) propage un vecteur tangent de gauche à droite, ce qui coûte une passe par paramètre. Le mode arrière (VJP) propage l'adjoint de droite à gauche en une seule passe.

```{code-cell} python
:tags: [hide-input]

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
%config InlineBackend.figure_format = 'retina'

fig, axes = plt.subplots(2, 1, figsize=(10, 4.5))

node_labels = ['$f_1$', '$f_2$', '$f_3$', '$\\mathcal{L}$']
xs = [1.5, 3.5, 5.5, 7.5]

for ax, (title, color, direction, vec_labels, vec_pos) in zip(axes, [
    (
        'Mode avant (JVP) — tangent $\\tilde{v}$ se propage de gauche à droite',
        '#1f77b4',
        'forward',
        ['$\\tilde{v}_0$', '$\\tilde{v}_1 = J_{f_1}\\tilde{v}_0$',
         '$\\tilde{v}_2 = J_{f_2}\\tilde{v}_1$', '$\\tilde{v}_3 = J_{f_3}\\tilde{v}_2$'],
        [0.6, 2.5, 4.5, 6.5]
    ),
    (
        'Mode arrière (VJP) — adjoint $\\bar{u}$ se propage de droite à gauche',
        '#d62728',
        'backward',
        ['$\\bar{u}_0 = J_{f_1}^\\top\\bar{u}_1$', '$\\bar{u}_1 = J_{f_2}^\\top\\bar{u}_2$',
         '$\\bar{u}_2 = J_{f_3}^\\top\\bar{u}_3$', '$\\bar{u}_3 = 1$'],
        [0.6, 2.5, 4.5, 6.5]
    )
]):
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 2)
    ax.axis('off')
    ax.set_title(title, fontsize=10, pad=4)

    # Draw function nodes
    for x, label in zip(xs, node_labels):
        circ = plt.Circle((x, 1), 0.45, color='#f0f0f0', ec='#444444', linewidth=1.5, zorder=3)
        ax.add_patch(circ)
        ax.text(x, 1, label, ha='center', va='center', fontsize=11, zorder=4)

    # Draw arrows between nodes
    arrow_color = color
    arrow_xs = list(zip(xs[:-1], xs[1:]))
    if direction == 'forward':
        for x1, x2 in arrow_xs:
            ax.annotate('', xy=(x2 - 0.47, 1), xytext=(x1 + 0.47, 1),
                        arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
        # Vector labels below arrows
        for vl, vx in zip(vec_labels, vec_pos):
            ax.text(vx + 0.45, 0.35, vl, ha='center', va='center',
                    fontsize=8, color=arrow_color,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=arrow_color, alpha=0.85))
    else:
        for x1, x2 in arrow_xs:
            ax.annotate('', xy=(x1 + 0.47, 1), xytext=(x2 - 0.47, 1),
                        arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2))
        for vl, vx in zip(vec_labels, vec_pos):
            ax.text(vx + 0.45, 0.35, vl, ha='center', va='center',
                    fontsize=8, color=arrow_color,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=arrow_color, alpha=0.85))

plt.tight_layout()
```

Pour une perte scalaire avec $n$ paramètres, le mode avant nécessite $n$ passes (une par direction de base $\mathbf{e}_i$), tandis que le mode arrière calcule tout le gradient en une seule passe. C'est l'argument central qui justifie la rétropropagation dans les réseaux avec des millions de paramètres.

### Exemple: MLP avec une couche cachée

Prenons un réseau à une couche cachée avec la perte des moindres carrés:

$$
\mathcal{L} = \frac{1}{2}\|y - \mathbf{w}_2^\top \varphi(W_1 \mathbf{x} + \mathbf{b}_1) - b_2\|^2
$$

La passe avant calcule les valeurs intermédiaires:

$$
\begin{aligned}
\mathbf{a}_1 &= W_1 \mathbf{x} + \mathbf{b}_1 \\
\mathbf{z}_1 &= \varphi(\mathbf{a}_1) \\
\hat{y} &= \mathbf{w}_2^\top \mathbf{z}_1 + b_2 \\
\mathcal{L} &= \frac{1}{2}(y - \hat{y})^2
\end{aligned}
$$

La passe arrière propage le gradient en sens inverse, couche par couche:

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

où $\odot$ désigne le produit élément par élément. Chaque ligne utilise uniquement des quantités déjà calculées, soit lors de la passe avant ($\mathbf{z}_1$, $\mathbf{a}_1$, $\mathbf{x}$), soit lors des étapes précédentes de la passe arrière. La structure est toujours la même: le gradient par rapport aux pré-activations d'une couche est propagé vers l'arrière pour obtenir le gradient de la couche précédente.

**Point de contrôle:** Si vous pouvez suivre cet exemple du début à la fin, vous avez compris le mécanisme de la rétropropagation. L'algorithme général ci-dessous ne fait que formaliser cette procédure pour un nombre arbitraire de couches. Si certaines étapes restent floues, l'exercice 3 vous permettra de refaire ce calcul vous-même avec des valeurs numériques.

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

Le vecteur $\mathbf{u}_k$ est l'adjoint: il accumule la sensibilité de la perte aux activations de la couche $k$. À chaque étape, deux produits sont calculés: un pour obtenir le gradient des paramètres de la couche courante, et un pour propager l'adjoint vers la couche précédente.

## Différentiation automatique

*Cette section ne dépend pas des détails de la rétropropagation: vous pouvez la lire même si vous avez survolé la section précédente.*

La rétropropagation est un cas particulier d'un principe plus général. Cette section élargit la perspective: nous montrons comment le calcul de gradients se généralise à des programmes arbitraires, au-delà des réseaux en chaîne.

### Graphes de calcul

L'algorithme de rétropropagation que nous venons de décrire suppose une structure en chaîne: chaque couche reçoit la sortie de la précédente. Les architectures modernes sont plus complexes. Les connexions résiduelles, par exemple, ajoutent la sortie d'une couche à celle d'une couche plus profonde. Les architectures avec plusieurs branches ou des opérations de concaténation ne sont pas des chaînes simples.

Ces architectures se représentent naturellement comme des **graphes de calcul**: des graphes orientés acycliques (DAG) où chaque noeud applique une opération élémentaire différentiable. L'entrée du réseau et les paramètres sont les noeuds sources; la perte est le noeud puits.

Dans un DAG, un noeud peut influencer la sortie par plusieurs chemins. La règle de la chaîne se généralise: si un noeud $j$ a des enfants $k_1, \ldots, k_m$, le gradient total est la somme des contributions de chaque chemin:

$$
\frac{\partial \mathcal{L}}{\partial x_j} = \sum_{k \in \text{enfants}(j)} \frac{\partial \mathcal{L}}{\partial x_k} \frac{\partial x_k}{\partial x_j}
$$

La rétropropagation parcourt le graphe dans l'ordre topologique inverse, accumulant ces contributions.

### Différentiation numérique, symbolique et automatique

Pour calculer la dérivée d'un programme, trois approches existent:

La différentiation numérique approxime la dérivée par différences finies:

$$
\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + \epsilon \mathbf{e}_i) - f(\mathbf{x} - \epsilon \mathbf{e}_i)}{2\epsilon}
$$

Cette méthode est simple à implémenter mais souffre de deux problèmes: elle requiert $O(n)$ évaluations de $f$ pour un gradient en dimension $n$, et elle est sujette aux erreurs d'arrondi (le choix de $\epsilon$ est délicat). Elle reste utile pour vérifier des implémentations de gradient.

La différentiation symbolique applique les règles de dérivation formellement, comme on le ferait à la main. Elle produit des expressions exactes, mais ces expressions peuvent croître exponentiellement en taille pour des programmes complexes. De plus, elle requiert que le programme soit représenté sous forme symbolique, ce qui exclut les structures de contrôle (boucles, conditions).

La différentiation automatique (AD) est une troisième voie. Elle évalue la dérivée exacte d'un programme numérique en le décomposant en opérations élémentaires et en appliquant la règle de la chaîne à chaque étape. Contrairement à la différentiation numérique, elle est exacte (aux erreurs de virgule flottante près). Contrairement à la différentiation symbolique, elle opère sur des valeurs numériques, pas sur des expressions, et gère naturellement les boucles et les conditions.

La rétropropagation n'est rien d'autre que la différentiation automatique en mode arrière, appliquée au programme qui calcule la perte d'un réseau de neurones.

### La programmation différentiable

Les bibliothèques modernes comme JAX, PyTorch et TensorFlow implémentent la différentiation automatique de manière générale: toute fonction composée d'opérations dont on connaît les dérivées locales peut être différentiée automatiquement. C'est le paradigme de la **programmation différentiable** (*differentiable programming*).

Au lieu de dériver manuellement les gradients pour chaque architecture, nous écrivons la passe avant comme un programme ordinaire, et la bibliothèque se charge de calculer les gradients.

Voici un exemple avec JAX. Nous définissons la passe avant d'un MLP à une couche cachée, puis utilisons `jax.grad` pour obtenir automatiquement la fonction qui calcule les gradients:

```python
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

# jax.grad retourne une FONCTION qui calcule le gradient
grad_fn = jax.grad(loss_fn)

# Un seul appel donne les gradients par rapport à tous les paramètres
grads = grad_fn(params, x, y)
```

La fonction `loss_fn` est un programme Python ordinaire. L'appel `jax.grad(loss_fn)` produit une nouvelle fonction qui calcule le gradient par rapport au premier argument (`params`). Aucune dérivation manuelle n'est nécessaire: JAX applique la règle de la chaîne automatiquement, en mode arrière, sur la trace d'exécution du programme.

Ce paradigme change la façon de penser les modèles. Au lieu de concevoir une architecture puis de dériver ses gradients, on conçoit un programme de calcul quelconque, avec des boucles, des conditions, des appels de fonctions, et on le différentie automatiquement. La seule contrainte est que les opérations soient différentiables (ou différentiables presque partout, comme ReLU).

## Considérations pratiques

*Cette section suppose que vous avez compris le mécanisme général de la rétropropagation (le gradient se propage de la sortie vers l'entrée), mais pas nécessairement les détails des jacobiennes.*

### Instabilité du gradient en profondeur

Dans un réseau profond, le gradient de la perte par rapport aux premières couches est un produit de jacobiennes locales:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{z}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}_L} \prod_{\ell=2}^{L} \frac{\partial \mathbf{z}_\ell}{\partial \mathbf{z}_{\ell-1}}
$$

Si les jacobiennes ont un rayon spectral inférieur à 1 (ce qui arrive avec la sigmoïde, dont la dérivée est au plus 0,25), le produit décroît exponentiellement avec la profondeur: le gradient disparaît (*vanishing gradient*). Les premières couches reçoivent des signaux de gradient négligeables et cessent d'apprendre.

Inversement, si le rayon spectral est supérieur à 1, le gradient explose (*exploding gradient*). Les mises à jour deviennent instables et l'entraînement diverge.

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

Au lieu d'apprendre la transformation complète, le bloc $f$ n'apprend que le résidu, c'est-à-dire la différence entre la sortie désirée et l'entrée. Le gradient se propage directement à travers la connexion identité, ce qui atténue le problème du gradient qui disparaît:

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

Ce chapitre a montré comment les réseaux de neurones s'inscrivent dans la progression des modèles vus dans les chapitres précédents. Le point de départ est le même cadre de maximum de vraisemblance: un modèle prédit les paramètres d'une distribution conditionnelle (moyenne pour la régression, probabilité pour la classification). Les modèles linéaires utilisent directement les entrées; l'expansion de caractéristiques transforme les entrées par une fonction fixée; les réseaux de neurones apprennent cette transformation.

La rétropropagation applique la règle de la chaîne systématiquement pour calculer les gradients dans un réseau profond. Le mode arrière (VJP) est efficace pour les fonctions à sortie scalaire, ce qui est le cas de toutes les fonctions de perte. La différentiation automatique généralise ce principe à tout programme composé d'opérations différentiables.

Les chapitres suivants présenteront des architectures spécialisées qui exploitent la structure des données: les réseaux convolutifs pour les images et les réseaux récurrents pour les séquences.

```{admonition} Ce que vous devez retenir
:class: tip

1. **Les réseaux de neurones apprennent leurs caractéristiques.** Contrairement aux modèles à expansion fixe (chapitre 4), la fonction $\boldsymbol{\phi}$ est paramétrique et optimisée avec les données.

2. **La non-linéarité est ce qui donne le pouvoir expressif.** Sans fonctions d'activation, empiler des couches linéaires ne donne qu'une transformation linéaire.

3. **La rétropropagation calcule les gradients efficacement.** Elle propage le gradient de la perte vers l'arrière, couche par couche, en utilisant la règle de la chaîne.

4. **La différentiation automatique généralise ce principe.** Les bibliothèques modernes calculent automatiquement les gradients de tout programme différentiable.

5. **L'entraînement en profondeur pose des défis spécifiques.** Le gradient qui disparaît ou explose nécessite des techniques comme l'initialisation soignée, la normalisation par lots, ou les connexions résiduelles.
```

## Exercices

Les exercices ★ vérifient la compréhension de base. Les exercices ★★ demandent d'appliquer les concepts à des calculs concrets. L'exercice ★★★ approfondit le problème du gradient qui disparaît et est optionnel pour IFT3395.

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

