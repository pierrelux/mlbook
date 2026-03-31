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
- Expliquer pourquoi le perceptron simple ne peut pas résoudre certains problèmes (comme XOR) et pourquoi cela motive les architectures multicouches
- Expliquer comment les réseaux de neurones apprennent leurs propres caractéristiques, contrairement aux modèles à expansion fixe
- Définir l'architecture d'un perceptron multicouche (MLP) et le rôle des fonctions d'activation
- Formuler les algorithmes de dérivation automatique en mode avant (JVP) et en mode arrière (VJP) sur un graphe de calcul
- Relier la rétropropagation au mode arrière appliqué au graphe d'un réseau de neurones
- Expliquer le paradigme de la programmation différentiable
- Identifier l'architecture d'un MLP pour la régression et la classification, et la relier aux modèles linéaires des chapitres 2 et 3
```

Aux chapitres 2 et 3, nous avons construit des modèles linéaires pour la régression et la classification. Au chapitre 4, nous avons vu comment enrichir ces modèles en transformant les entrées par une fonction $\boldsymbol{\phi}$ fixée à l'avance. Ce chapitre franchit une étape supplémentaire: au lieu de choisir $\boldsymbol{\phi}$ manuellement, nous allons l'apprendre à partir des données. Cette idée conduit aux réseaux de neurones.

Dans ce chapitre, nous rappelons d'abord le cadre probabiliste qui unifie régression et classification, puis nous montrons comment le perceptron simple atteint une limite structurelle (illustrée par le problème XOR). Nous présentons ensuite l'anatomie d'un réseau multicouche (couches, activations, architecture). La section sur la dérivation automatique est plus technique: elle développe la règle de la chaîne, les produits jacobien-vecteur, et l'algorithme de rétropropagation, puis montre comment les bibliothèques modernes implémentent ces idées. Vous pouvez survoler les détails en première lecture et retenir le mécanisme général. La section d'implémentation propose un MLP complet en NumPy, et le chapitre se termine par une mise en perspective montrant comment les MLP sont utilisés en pratique pour la régression et la classification. Les algorithmes d'optimisation, la stabilisation de l'entraînement et la régularisation sont couverts au chapitre suivant.

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

## Le perceptron: aux origines des réseaux de neurones

La régression logistique et les réseaux de neurones partagent un ancêtre commun: le **perceptron**, proposé par Rosenblatt en 1958 {cite}`rosenblatt1958perceptron`. Comprendre ce modèle et sa limitation éclaire pourquoi les réseaux multicouches ont été inventés, et pourquoi leur développement a pris plusieurs décennies.

### Un modèle inspiré du neurone biologique

En 1943, McCulloch et Pitts {cite}`mccullochpitts1943` ont formalisé le comportement du neurone biologique: une unité qui reçoit des signaux pondérés et s'active si leur somme dépasse un seuil. Le modèle se résume à:

$$
\hat{y} = \mathbf{1}[\boldsymbol{\theta}^\top \mathbf{x} \geq 0]
$$

Ce neurone calcule une combinaison linéaire des entrées, puis prend une décision binaire: actif ($\hat{y} = 1$) ou inactif ($\hat{y} = 0$). Rosenblatt y a ajouté un algorithme pour ajuster les poids $\boldsymbol{\theta}$ à partir d'exemples étiquetés. L'enthousiasme de l'époque était considérable: des démonstrateurs matériels ont été construits, et la presse grand public annonçait une machine capable «d'apprendre à reconnaître».

Le point de départ est la neuroscience computationnelle, pas l'optimisation. McCulloch et Pitts voulaient formaliser le comportement des neurones corticaux; Rosenblatt s'inspirait de la vision artificielle et des réseaux nerveux. C'est cette origine qui distingue la trajectoire intellectuelle du perceptron de celle de la régression logistique, même si les deux modèles aboutissent à une structure mathématique très proche.

### Lien avec la régression logistique

Les deux modèles calculent la même pré-activation linéaire $z = \boldsymbol{\theta}^\top \mathbf{x}$, mais diffèrent dans la façon dont ils l'interprètent:

| | Régression logistique | Perceptron |
|---|---|---|
| Activation | $\sigma(z)$ (sigmoïde, continue) | $\mathbf{1}[z \geq 0]$ (échelon, discontinue) |
| Sortie | probabilité $\in (0,1)$ | décision $\in \{0, 1\}$ |
| Frontière de décision | $\boldsymbol{\theta}^\top \mathbf{x} = 0$ | $\boldsymbol{\theta}^\top \mathbf{x} = 0$ |

La frontière de décision est dans les deux cas le même hyperplan $\{\mathbf{x} : \boldsymbol{\theta}^\top \mathbf{x} = 0\}$. La sigmoïde peut être vue comme une version lisse et probabiliste de la fonction échelon: au lieu de trancher brusquement, elle exprime l'incertitude via une probabilité.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# --- Panneau gauche: activations ---
z = np.linspace(-5, 5, 500)
sigmoid = 1 / (1 + np.exp(-z))

ax = axes[0]
ax.plot(z, sigmoid, '#1f77b4', lw=2.5, label=r'Sigmoïde $\sigma(z)$')
ax.plot(z[z < 0],  np.zeros(np.sum(z < 0)),  '#d62728', lw=2.5)
ax.plot(z[z >= 0], np.ones(np.sum(z >= 0)),  '#d62728', lw=2.5,
        label=r'Échelon $\mathbf{1}[z \geq 0]$')
ax.scatter([0], [1], color='#d62728', s=55, zorder=5)
ax.scatter([0], [0], color='white', edgecolors='#d62728', linewidths=2, s=55, zorder=5)
ax.axvline(0, color='gray', lw=1, linestyle=':', alpha=0.6)
ax.set_xlabel(r'Pré-activation $z = \boldsymbol{\theta}^\top \mathbf{x}$')
ax.set_ylabel('Sortie')
ax.set_title('Activations comparées')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 5); ax.set_ylim(-0.1, 1.2)
ax.annotate(r'Frontière: $z=0$', xy=(0, 0.5), xytext=(1.8, 0.25),
            fontsize=9, color='#555555',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

# --- Panneau droit: frontière de décision 2D ---
rng = np.random.default_rng(0)
n = 40
X_pos = rng.multivariate_normal([ 1.5,  0.8], [[0.4, 0], [0, 0.4]], n)
X_neg = rng.multivariate_normal([-1.5, -0.8], [[0.4, 0], [0, 0.4]], n)

ax = axes[1]
xx, yy = np.meshgrid(np.linspace(-3.5, 3.5, 200), np.linspace(-2.5, 2.5, 200))
Z = xx + yy   # theta = [1, 1], frontière: x1 + x2 = 0
ax.contourf(xx, yy, Z, levels=[-100, 0, 100],
            colors=['#fdd8d8', '#d8e8fd'], alpha=0.45)
ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
ax.scatter(X_pos[:, 0], X_pos[:, 1], color='#1f77b4', marker='o', s=35,
           alpha=0.85, label='Classe 1', edgecolors='white', linewidths=0.5)
ax.scatter(X_neg[:, 0], X_neg[:, 1], color='#d62728', marker='s', s=35,
           alpha=0.85, label='Classe 0', edgecolors='white', linewidths=0.5)
ax.text( 1.8, -2.0, 'Rég. log.: prob. $\\to 1$\nPerceptron: classe 1',
         fontsize=7.5, color='#1f77b4', ha='center',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#1f77b4', alpha=0.8))
ax.text(-1.8,  1.8, 'Rég. log.: prob. $\\to 0$\nPerceptron: classe 0',
         fontsize=7.5, color='#d62728', ha='center',
         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#d62728', alpha=0.8))
ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
ax.set_title(r'Même frontière: $\boldsymbol{\theta}^\top \mathbf{x} = 0$')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

plt.suptitle('Régression logistique et perceptron: deux lectures du même hyperplan', fontsize=10)
plt.tight_layout()
```

### La règle d'apprentissage et la difficulté de l'optimisation

La régression logistique minimise l'entropie croisée, une fonction différentiable, ce qui autorise la descente de gradient. Le perceptron minimise la **perte perceptron** (avec des étiquettes $y_i \in \{-1, +1\}$):

$$
\mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^n \max\bigl(0,\; -y_i \cdot \boldsymbol{\theta}^\top \mathbf{x}_i\bigr)
$$

Cette perte est nulle pour les exemples bien classés et pénalise les erreurs proportionnellement à leur amplitude. Elle n'est toutefois pas différentiable au point exact où $\boldsymbol{\theta}^\top \mathbf{x}_i = 0$. On utilise alors le **sous-gradient**, qui généralise le gradient aux fonctions non différentiables:

$$
\nabla_{\boldsymbol{\theta}} \mathcal{L} \;\ni\; -\sum_{i:\; y_i \boldsymbol{\theta}^\top \mathbf{x}_i \leq 0} y_i \mathbf{x}_i
$$

Cela donne la règle de mise à jour classique: pour chaque exemple mal classé, corriger les poids dans la direction de cet exemple,

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \eta\, y_i \mathbf{x}_i \qquad \text{si } y_i \boldsymbol{\theta}^\top \mathbf{x}_i \leq 0
$$

La convergence est plus délicate qu'avec la descente de gradient sur une fonction convexe et différentiable. Si les données sont linéairement séparables, le théorème de convergence du perceptron {cite}`novikoff1962convergence` garantit que l'algorithme s'arrête en un nombre fini de mises à jour, borné par $(R/\gamma)^2$ où $R$ est le rayon des données et $\gamma$ la marge de séparation. Si les données ne sont pas linéairement séparables, l'algorithme peut cycler sans jamais converger.

### Une limite structurelle

Toutes ces variantes (perceptron, régression logistique, moindres carrés) partagent la même contrainte: leur frontière de décision est un hyperplan. Quelle que soit la façon dont on choisit ou entraîne les poids $\boldsymbol{\theta}$, on ne peut séparer que des classes linéairement séparables.

C'est précisément ce que Minsky et Papert ont formalisé en 1969 {cite}`minsky1969perceptrons`, en montrant que certaines fonctions booléennes sont impossibles à apprendre pour un perceptron simple. L'exemple canonique est la fonction XOR (ou exclusif):

| $x_1$ | $x_2$ | $y = x_1 \oplus x_2$ |
|:-----:|:-----:|:--------------------:|
| 0     | 0     | 0                    |
| 0     | 1     | 1                    |
| 1     | 0     | 1                    |
| 1     | 1     | 0                    |

Les points de classe 0 sont disposés en diagonale, $(0,0)$ et $(1,1)$, et ceux de classe 1 sur l'autre, $(0,1)$ et $(1,0)$. Aucune droite ne peut séparer ces deux groupes. Mais Minsky et Papert montraient aussi la solution: empiler deux couches de perceptrons suffit, car la première couche peut transformer l'espace de sorte que les classes deviennent linéairement séparables. Nous verrons dans la section d'implémentation qu'un petit MLP résout XOR sans difficulté.

Leur analyse a contribué à un ralentissement de la recherche sur les réseaux de neurones pendant plusieurs années, jusqu'à ce que les avancées en optimisation et en calcul redonnent vie au domaine. La section suivante formalise l'architecture multicouche qui résout cette limitation.

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

On peut représenter cette couche comme un graphe de calcul: les entrées et paramètres sont les nœuds sources, les opérations ($\times$, $+$, $\varphi$) sont des transformations, et l'activation $\mathbf{z}$ est le nœud de sortie. Cette perspective sera centrale dans la section sur la dérivation automatique.

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

Plusieurs variantes de ReLU existent pour atténuer ce problème. La **Leaky ReLU** utilise une petite pente $\alpha \approx 0{,}01$ pour les valeurs négatives: $\text{LeakyReLU}(a) = \max(\alpha a, a)$. La **GELU** (*Gaussian Error Linear Unit*), définie par $\text{GELU}(a) = a \cdot \Phi(a)$ où $\Phi$ est la fonction de répartition normale, est utilisée dans les architectures modernes comme les transformeurs.

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

La dérivée de la sigmoïde est bornée par 0,25: à chaque couche, le gradient est multiplié par un facteur d'au plus 0,25. Ce phénomène de **saturation** est la cause principale de la dissolution du gradient dans les réseaux profonds. Nous y reviendrons en détail au chapitre suivant.

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

## Dérivation automatique

Pour optimiser les paramètres d'un réseau, nous avons besoin du gradient de la perte par rapport à chaque paramètre. Dans un réseau à $L$ couches, la perte dépend des paramètres de la couche $\ell$ à travers toutes les couches suivantes $\ell+1, \ldots, L$: le calcul du gradient exige d'appliquer la règle de la chaîne à travers tout le graphe de calcul.

Cette section présente la **dérivation automatique** (DA), également appelée dérivation algorithmique ou, dans la littérature anglophone, *automatic differentiation* (AD). C'est le cadre général qui formalise ce calcul. Nous commençons par situer la DA parmi les approches de calcul de dérivées, puis nous développons la règle de la chaîne sous forme de produits jacobien-vecteur (JVP et VJP). Nous introduisons ensuite les graphes de calcul (DAG) et formulons les algorithmes du mode avant et du mode arrière sur un graphe arbitraire. La rétropropagation (*backpropagation*) apparaît alors comme un cas particulier du mode arrière, appliqué au graphe en chaîne d'un réseau de neurones. Enfin, nous montrons comment les bibliothèques modernes (JAX, PyTorch) implémentent ces principes via le traçage d'opérations.

````{admonition} Rappel: jacobiennes, hessiennes et conventions d'agencement
:class: note dropdown

**Jacobienne.** Pour une fonction $f: \mathbb{R}^n \to \mathbb{R}^m$, la **jacobienne** $\mathbf{J}_f(\mathbf{x}) \in \mathbb{R}^{m \times n}$ est la matrice de toutes les dérivées partielles du premier ordre:

$$
(\mathbf{J}_f(\mathbf{x}))_{ij} = \frac{\partial f_i}{\partial x_j}
$$

La ligne $i$ de $\mathbf{J}_f$ est le gradient de la $i$-ième composante de sortie; la colonne $j$ encode la sensibilité de toutes les sorties à l'entrée $x_j$.

*Cas particulier: le gradient.* Quand $f: \mathbb{R}^n \to \mathbb{R}$ est scalaire, la jacobienne se réduit à un vecteur ligne $\mathbf{J}_f \in \mathbb{R}^{1 \times n}$. Le **gradient** $\nabla_\mathbf{x} f \in \mathbb{R}^n$ est sa transposée (vecteur colonne): $\nabla_\mathbf{x} f = \mathbf{J}_f^\top$.

**Hessienne.** Pour $f: \mathbb{R}^n \to \mathbb{R}$, la **hessienne** $\mathbf{H}_f(\mathbf{x}) \in \mathbb{R}^{n \times n}$ contient les dérivées partielles du second ordre:

$$
(\mathbf{H}_f(\mathbf{x}))_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

Quand $f$ est deux fois continûment différentiable, $\mathbf{H}_f$ est symétrique (théorème de Schwarz). La hessienne intervient dans les méthodes du second ordre (Newton, BFGS), mais **pas** dans la rétropropagation standard qui n'utilise que des dérivées du premier ordre.

**Conventions d'agencement.** Deux conventions coexistent dans la littérature et sont source de confusion:

- *Convention du numérateur* (utilisée dans ce livre): $\mathbf{J}_f \in \mathbb{R}^{m \times n}$ avec les sorties en lignes et les entrées en colonnes.
- *Convention du dénominateur*: la transposée de la convention précédente, $\mathbf{J}_f \in \mathbb{R}^{n \times m}$.

Ces deux conventions sont cohérentes en interne, mais les formules de la règle de la chaîne et des VJP s'écrivent différemment selon celle qu'on adopte. La conséquence pratique: avec la convention du numérateur, pour $\mathcal{L}: \mathbb{R}^m \to \mathbb{R}$, le gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{z}} \in \mathbb{R}^m$ est un **vecteur colonne**, et le VJP $\mathbf{u}^\top \mathbf{J}_f$ est le produit d'un vecteur ligne par une matrice $m \times n$.

**Dimensions dans un MLP.** Les trois types de jacobiennes locales qui apparaissent en rétropropagation:

| Opération | Fonction | Jacobienne |
|-----------|----------|------------|
| Couche affine | $f(\mathbf{z}) = W\mathbf{z} + \mathbf{b}$, $W \in \mathbb{R}^{m \times n}$ | $\mathbf{J}_f = W \in \mathbb{R}^{m \times n}$ |
| Activation élémentaire | $g(\mathbf{a}) = \varphi(\mathbf{a})$, appliquée composante par composante | $\mathbf{J}_g = \operatorname{diag}(\varphi'(\mathbf{a})) \in \mathbb{R}^{m \times m}$ |
| Perte scalaire | $\mathcal{L}: \mathbb{R}^m \to \mathbb{R}$ | $\mathbf{J}_\mathcal{L} = \left(\frac{\partial \mathcal{L}}{\partial \mathbf{z}}\right)^\top \in \mathbb{R}^{1 \times m}$ |

Ces dimensions se composent cohéremment: le produit $\mathbf{J}_\mathcal{L} \cdot \mathbf{J}_g \cdot \mathbf{J}_f \in \mathbb{R}^{1 \times n}$ donne bien un vecteur ligne, dont la transposée est le gradient par rapport à l'entrée $\mathbf{z}$.
````

### Dérivation numérique, symbolique et automatique

Pour calculer la dérivée d'un programme, trois approches existent:

La **dérivation numérique** approxime la dérivée par différences finies:

$$
\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + \epsilon \mathbf{e}_i) - f(\mathbf{x} - \epsilon \mathbf{e}_i)}{2\epsilon}
$$

Cette méthode est simple à implémenter mais souffre de deux problèmes: elle requiert $O(n)$ évaluations de $f$ pour un gradient en dimension $n$, et elle est sujette aux erreurs d'arrondi (le choix de $\epsilon$ est délicat). Elle reste utile pour *vérifier* des implémentations de gradient.

La **dérivation symbolique** applique les règles de dérivation formellement, comme on le ferait à la main. Le résultat est une *expression mathématique*, pas un nombre. La bibliothèque SymPy permet de s'en convaincre:

```{code-cell} python
:tags: [remove-output]

import sympy as sp

x = sp.Symbol('x')
f = sp.sin(x**2) * sp.exp(-x)

df = sp.diff(f, x)
print(df)
```

L'appel `sp.diff(f, x)` retourne une nouvelle expression symbolique: $2x\cos(x^2)e^{-x} - \sin(x^2)e^{-x}$. Le système manipule des formules, pas des valeurs numériques. Pour obtenir un nombre, il faut ensuite *évaluer* cette expression en un point:

```{code-cell} python
:tags: [remove-output]

df.subs(x, 1.0).evalf()  # évalue la dérivée en x = 1
```

Cette distinction entre *construire une expression* et *évaluer un nombre* est au cœur de la différence entre dérivation symbolique et automatique.

L'approche symbolique produit des résultats exacts, mais les expressions intermédiaires peuvent croître de façon exponentielle. Considérons une composition itérée $h(x) = \sin(\sin(\cdots\sin(x)\cdots))$ sur $k$ niveaux. Chaque application de la règle de la chaîne multiplie l'expression par un facteur $\cos(\cdots)$, et l'expression de la dérivée accumule un produit de cosinus imbriqués dont la taille croît avec $k$. Pour des programmes réels avec des centaines d'opérations, cette croissance rend l'approche impraticable. De plus, la dérivation symbolique requiert que le programme soit représenté sous forme d'expression mathématique, ce qui exclut les structures de contrôle comme les boucles et les conditions.

La **dérivation automatique** (DA) est une troisième voie. Au lieu de construire l'expression symbolique de la dérivée puis de l'évaluer, elle *évalue directement la dérivée* en un point donné, en propageant des valeurs numériques à travers le programme. Chaque opération élémentaire (addition, multiplication, $\sin$, $\exp$, ...) est accompagnée de sa règle de dérivation locale, et la règle de la chaîne assemble ces dérivées locales au fur et à mesure de l'exécution. Le résultat est un nombre, la valeur exacte de la dérivée au point considéré, obtenu sans jamais former une expression intermédiaire. Contrairement à la dérivation numérique, ce résultat est exact (aux erreurs de virgule flottante près). Contrairement à la dérivation symbolique, il gère naturellement les boucles et les conditions, puisqu'il opère sur l'exécution concrète du programme.

La rétropropagation n'est rien d'autre que la dérivation automatique en mode arrière, appliquée au programme qui calcule la perte d'un réseau de neurones.


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

Chaque étape multiplie une jacobienne locale par un vecteur, ce qui coûte $O(mn)$ au lieu de $O(m n^2)$ pour le produit par une matrice. Le calcul se fait dans le même sens que la passe avant: c'est le mode avant de la dérivation automatique.

Le VJP (*Vector-Jacobian Product*) propage un vecteur adjoint $\mathbf{u}^\top$ de droite à gauche:

$$
\mathbf{u}^\top \mathbf{J}_f(\mathbf{x}) = ((\mathbf{u}^\top \cdot \mathbf{J}_{f_L}) \cdot \mathbf{J}_{f_{L-1}}) \cdots \mathbf{J}_{f_1}
$$

Le calcul se fait dans le sens inverse de la passe avant: c'est le mode arrière.

Pour une perte scalaire $\mathcal{L}: \mathbb{R}^n \to \mathbb{R}$, le gradient $\nabla_\mathbf{x} \mathcal{L}$ est exactement un VJP avec $\mathbf{u} = 1$. Le mode arrière calcule donc le gradient par rapport à tous les paramètres en une seule passe arrière, quel que soit le nombre de paramètres. C'est pourquoi la rétropropagation utilise le mode arrière.

La figure ci-dessous contraste les deux modes sur une chaîne de trois fonctions $f_1 \circ f_2 \circ f_3$. Le mode avant (JVP) propage un vecteur tangent de gauche à droite, ce qui coûte une passe par paramètre. Le mode arrière (VJP) propage l'adjoint de droite à gauche en une seule passe.

**Mode avant (JVP)**: le vecteur tangent $\tilde{v}$ se propage de gauche à droite.

```{mermaid}
graph LR
    v0["ṽ₀"] -->|"J_f₁"| f1((f₁))
    f1 -->|"ṽ₁ = J_f₁ ṽ₀"| f2((f₂))
    f2 -->|"ṽ₂ = J_f₂ ṽ₁"| f3((f₃))
    f3 -->|"ṽ₃ = J_f₃ ṽ₂"| L((ℒ))

    style v0 fill:none,stroke:none
    style f1 fill:#f0f0f0,stroke:#444
    style f2 fill:#f0f0f0,stroke:#444
    style f3 fill:#f0f0f0,stroke:#444
    style L fill:#f0f0f0,stroke:#444
    linkStyle 0,1,2,3 stroke:#1f77b4,stroke-width:2px
```

**Mode arrière (VJP)**: l'adjoint $\bar{u}$ se propage de droite à gauche.

```{mermaid}
graph RL
    L((ℒ)) -->|"ū₃ = 1"| f3((f₃))
    f3 -->|"ū₂ = ū₃ J_f₃"| f2((f₂))
    f2 -->|"ū₁ = ū₂ J_f₂"| f1((f₁))
    f1 -->|"ū₀ = ū₁ J_f₁"| x0["∇ℒ"]

    style x0 fill:none,stroke:none
    style f1 fill:#f0f0f0,stroke:#444
    style f2 fill:#f0f0f0,stroke:#444
    style f3 fill:#f0f0f0,stroke:#444
    style L fill:#f0f0f0,stroke:#444
    linkStyle 0,1,2,3 stroke:#d62728,stroke-width:2px
```

Pour une perte scalaire avec $n$ paramètres, le mode avant nécessite $n$ passes (une par direction de base $\mathbf{e}_i$), tandis que le mode arrière calcule tout le gradient en une seule passe. C'est l'argument central qui justifie la rétropropagation dans les réseaux avec des millions de paramètres.

Le carnet [](exercises/manual_vjps.ipynb) illustre ce mécanisme pas à pas sur un réseau à trois couches, en vérifiant à chaque étape que les VJP produisent le même résultat que les produits matriciels explicites.

### Graphes de calcul

Les deux modes ont été présentés pour une composition en chaîne. Mais les programmes réels ont des structures plus riches: une variable peut intervenir dans plusieurs opérations, créant des embranchements dans le graphe de calcul.

Toute expression arithmétique peut se décomposer en une séquence d'opérations élémentaires. Pour $f(x, y) = \sin(x) \cdot (x + y)$, cette décomposition introduit deux variables intermédiaires:

$$
v_1 = \sin(x), \quad v_2 = x + y, \quad v_3 = v_1 \cdot v_2 = f(x, y)
$$

On représente ces dépendances par un graphe orienté acyclique (DAG): chaque noeud est une valeur (entrée, intermédiaire ou sortie) et chaque arête indique qu'une valeur est utilisée pour calculer une autre.

```{mermaid}
graph LR
    x("x") --> sin_x("v₁ = sin(x)")
    x("x") --> add("v₂ = x + y")
    y("y") --> add("v₂ = x + y")
    sin_x --> mul("v₃ = v₁ · v₂")
    add --> mul
    mul --> L("f")

    style x    fill:#dae8fc,stroke:#6c8ebf
    style y    fill:#dae8fc,stroke:#6c8ebf
    style L    fill:#f8cecc,stroke:#b85450
    style sin_x fill:#f5f5f5,stroke:#666
    style add   fill:#f5f5f5,stroke:#666
    style mul   fill:#f5f5f5,stroke:#666
```

#### Règle de la chaîne sur un DAG

```{admonition} Convention: opérations locales et jacobiennes
:class: note

Chaque nœud $v$ du graphe calcule une opération locale $\phi_v$ à partir de ses prédécesseurs. On note $D_u \phi_v$ la jacobienne de $\phi_v$ par rapport à l'argument $u$, évaluée aux valeurs de la passe avant. Cette notation distingue la fonction $\phi_v$ de sa valeur $v$, et rend chaque dérivée non ambiguë.
```

Remarquez que $x$ a **deux arêtes sortantes**: il alimente $v_1 = \sin(x)$ et $v_2 = x + y$. Cela signifie que $x$ contribue à $f$ par deux chemins distincts dans le graphe. Notons $\phi_1(x) = \sin(x)$, $\phi_2(x,y) = x + y$ et $\phi_3(v_1,v_2) = v_1 \cdot v_2$ les opérations locales de chaque noeud. La fonction composée est $f(x,y) = \phi_3(\phi_1(x),\, \phi_2(x,y))$. En appliquant la règle de la chaîne multivariée, la dérivée totale de $f$ par rapport à $x$ est:

$$
\frac{df}{dx} = D_{v_1} \phi_3 \cdot D_x \phi_1 + D_{v_2} \phi_3 \cdot D_x \phi_2 = v_2 \cdot \cos(x) + v_1 \cdot 1
$$

La somme comporte deux termes, un par chemin de $x$ à $f$ dans le graphe:
- **Chemin** $x \to v_1 \to f$: le produit des jacobiennes locales le long des arêtes, $D_{v_1}\phi_3 \cdot D_x \phi_1$
- **Chemin** $x \to v_2 \to f$: le produit $D_{v_2}\phi_3 \cdot D_x \phi_2$

C'est la structure générale: la dérivée totale par rapport à une variable est la **somme sur tous les chemins** de cette variable à la sortie, où chaque chemin contribue le produit des jacobiennes locales le long de ses arêtes. Un noeud avec $k$ arêtes sortantes génère $k$ termes dans cette somme.

De manière générale, notons $\text{pred}(v)$ l'ensemble des **prédécesseurs** d'un noeud $v$ (les noeuds dont $v$ dépend directement) et $\text{succ}(u)$ l'ensemble de ses **successeurs** (les noeuds qui dépendent directement de $u$). La règle de la chaîne s'écrit dans les deux sens:

**Direction avant (tangentes).** Étant donné un vecteur tangent $\dot{x}_i$ pour chaque entrée, la tangente d'un noeud intermédiaire se propage vers l'avant. Le noeud $v$ reçoit les tangentes de tous ses prédécesseurs et les combine:

```{mermaid}
graph LR
    u1(("u₁")) -->|"D_u₁ φ_v · u̇₁"| v(("v"))
    u2(("u₂")) -->|"D_u₂ φ_v · u̇₂"| v
    u3(("u₃")) -->|"D_u₃ φ_v · u̇₃"| v

    style u1 fill:#f5f5f5,stroke:#666
    style u2 fill:#f5f5f5,stroke:#666
    style u3 fill:#f5f5f5,stroke:#666
    style v fill:#dae8fc,stroke:#6c8ebf
    linkStyle 0,1,2 stroke:#185fa5,stroke-width:2px
```

$$
\dot{v} = \sum_{u \in \text{pred}(v)} D_u \phi_v \; \dot{u}
$$

Chaque arête entrante contribue un JVP (jacobienne locale $\times$ tangente du prédécesseur). Le noeud $v$ somme ces contributions.

**Direction arrière (adjoints).** Étant donné un adjoint $\bar{f} = 1$ pour la sortie, l'adjoint de chaque noeud se propage vers l'arrière. Le noeud $u$ reçoit les adjoints de tous ses successeurs:

```{mermaid}
graph RL
    v1(("v₁")) -->|"v̄₁ · D_u φ_v₁"| u(("u"))
    v2(("v₂")) -->|"v̄₂ · D_u φ_v₂"| u
    v3(("v₃")) -->|"v̄₃ · D_u φ_v₃"| u

    style v1 fill:#f5f5f5,stroke:#666
    style v2 fill:#f5f5f5,stroke:#666
    style v3 fill:#f5f5f5,stroke:#666
    style u fill:#fcebeb,stroke:#a32d2d
    linkStyle 0,1,2 stroke:#a32d2d,stroke-width:2px
```

$$
\bar{u} = \sum_{v \in \text{succ}(u)} \bar{v} \, D_u \phi_v
$$

Chaque arête sortante (parcourue à rebours) contribue un VJP (adjoint du successeur $\times$ jacobienne locale). Le noeud $u$ somme ces contributions.

Les deux formules sont symétriques: la première applique $D_u \phi_v$ à droite d'un vecteur tangent (JVP), la seconde applique $D_u \phi_v$ à gauche d'un vecteur adjoint (VJP). C'est la règle de la chaîne multivariée.

### Tri topologique

Les deux formules ci-dessus posent un problème d'ordre. Pour calculer $\dot{v}$ (direction avant), il faut d'abord connaître $\dot{u}$ pour tous les prédécesseurs $u$ de $v$. Pour calculer $\bar{u}$ (direction arrière), il faut d'abord connaître $\bar{v}$ pour tous les successeurs $v$ de $u$.

Un **tri topologique** fournit un ordre de traitement qui respecte ces contraintes: chaque noeud apparaît après tous ses prédécesseurs. La passe avant suit cet ordre; la passe arrière le parcourt à rebours.

Le diagramme ci-dessous montre un tri topologique valide pour notre exemple. Les numéros indiquent l'ordre de traitement de la passe arrière (qui parcourt la liste à rebours):

```{mermaid}
graph LR
    x("x ⑥") --> sin_x("v₁ = sin(x) ⑤")
    x("x ⑥") --> add("v₂ = x + y ③")
    y("y ④") --> add("v₂ = x + y ③")
    sin_x --> mul("v₃ = v₁ · v₂ ②")
    add --> mul
    mul --> L("f ①")

    style x    fill:#dae8fc,stroke:#6c8ebf
    style y    fill:#dae8fc,stroke:#6c8ebf
    style L    fill:#f8cecc,stroke:#b85450
    style sin_x fill:#f5f5f5,stroke:#666
    style add   fill:#f5f5f5,stroke:#666
    style mul   fill:#f5f5f5,stroke:#666
```

Remarquez que $x$ est traité en dernier (⑥) par la passe arrière: comme $x$ contribue à *deux* branches ($v_1$ et $v_2$), il faut avoir accumulé les deux contributions avant de pouvoir calculer $\bar{x}$.

L'algorithme classique de tri topologique repose sur un parcours en profondeur (DFS):

```{prf:algorithm} Tri topologique par DFS
:label: ch7-topo-sort

**Entrée**: un DAG $G = (V, E)$ avec un noeud racine $r$

**Initialiser**: $\text{topo} \leftarrow []$, $\text{visités} \leftarrow \emptyset$

**Fonction** $\text{DFS}(v)$:
1. Si $v \in \text{visités}$: retourner
2. Ajouter $v$ à $\text{visités}$
3. Pour chaque prédécesseur $u$ de $v$ dans $G$: appeler $\text{DFS}(u)$
4. Ajouter $v$ à la fin de $\text{topo}$

**Appeler** $\text{DFS}(r)$

**Sortie**: $\text{topo}$ (ordre de la passe avant); $\text{reversed}(\text{topo})$ pour la passe arrière
```

### Mode avant (forward-mode AD)

Le mode avant propage les tangentes dans le même sens que l'exécution du programme, de l'entrée vers la sortie. On calcule simultanément la valeur de chaque noeud et sa tangente.

```{prf:algorithm} Dérivation automatique en mode avant
:label: forward-mode-ad

**Entrée**: DAG $G = (V, E)$, valeurs d'entrée $\mathbf{x}$, vecteur tangent $\dot{\mathbf{x}} \in \mathbb{R}^n$

**Sortie**: Valeur $f(\mathbf{x})$, produit jacobien-vecteur $\mathbf{J}_f(\mathbf{x}) \, \dot{\mathbf{x}}$

1. Initialiser les tangentes: $\dot{x}_i := (\dot{\mathbf{x}})_i$ pour chaque entrée $x_i$
2. Pour chaque noeud $v$ dans l'ordre topologique:
   - Calculer la valeur: $v := \text{op}_v(\{u : u \in \text{pred}(v)\})$
   - Calculer la tangente: $\displaystyle\dot{v} := \sum_{u \in \text{pred}(v)} D_u \phi_v \; \dot{u}$
3. Retourner $f(\mathbf{x})$ et $\dot{f}$
```

Une passe avant calcule un seul produit jacobien-vecteur $\mathbf{J}_f \dot{\mathbf{x}}$, c'est-à-dire la dérivée directionnelle de $f$ dans la direction $\dot{\mathbf{x}}$. Pour obtenir le gradient complet d'une fonction scalaire $f: \mathbb{R}^n \to \mathbb{R}$, il faudrait effectuer $n$ passes (une par vecteur de base $\mathbf{e}_i$). Le mode avant est donc efficace quand le nombre d'entrées est petit par rapport au nombre de sorties.

### Mode arrière (reverse-mode AD)

Le mode arrière procède en deux temps: d'abord une passe avant qui calcule et stocke les valeurs de chaque noeud, puis une passe arrière qui propage les adjoints de la sortie vers les entrées.

```{prf:algorithm} Dérivation automatique en mode arrière
:label: backprop

**Entrée**: DAG $G = (V, E)$, valeurs d'entrée $\mathbf{x}$

**Sortie**: Valeur $f(\mathbf{x})$, gradient $\nabla_{\mathbf{x}} f$

// Passe avant
1. Pour chaque noeud $v$ dans l'ordre topologique:
   - Calculer et stocker la valeur: $v := \text{op}_v(\{u : u \in \text{pred}(v)\})$

// Passe arrière
2. Initialiser les adjoints: $\bar{v} := 0$ pour tout $v$; $\bar{f} := 1$
3. Pour chaque noeud $v$ dans l'ordre topologique **inverse**:
   - Pour chaque prédécesseur $u \in \text{pred}(v)$:
     - $\bar{u} \mathrel{+}= \bar{v} \, D_u \phi_v$
4. Retourner $f(\mathbf{x})$ et $\{\bar{x}_i\}$ pour chaque entrée $x_i$
```

Une seule passe arrière calcule le gradient par rapport à *toutes* les entrées. Pour une perte scalaire avec $n$ paramètres, le mode arrière calcule le gradient complet en une passe, là où le mode avant en nécessiterait $n$. C'est l'argument central qui justifie l'usage du mode arrière pour l'entraînement des réseaux de neurones.

Le mode arrière a été décrit pour la première fois par Seppo Linnainmaa {cite}`linnainmaa1970` dans sa thèse de maîtrise à l'Université d'Helsinki. Dans le contexte de l'apprentissage profond, cet algorithme porte le nom de **rétropropagation** (*backpropagation*), popularisé par Rumelhart, Hinton et Williams en 1986 {cite}`rumelhart1986learning`. Un réseau à $K$ couches définit un DAG en chaîne $f_1 \circ f_2 \circ \cdots \circ f_K$, où chaque noeud n'a qu'un seul prédécesseur et un seul successeur. La passe arrière se simplifie alors en une boucle sur les couches $K, K-1, \ldots, 1$. Mais le mode arrière est plus général: il s'applique à tout programme différentiable, y compris ceux qui contiennent des embranchements, des variables réutilisées, des boucles ou des conditions. C'est ce qui permet à des bibliothèques comme JAX ou PyTorch de différentier n'importe quelle fonction Python.

L'animation interactive ci-dessous illustre les deux algorithmes sur le DAG de $f(x,y) = \sin(x) \cdot (x + y)$ avec les valeurs $(x, y) = (0.5, \; 1.2)$. En mode avant, les tangentes se propagent de gauche à droite; en mode arrière, les adjoints se propagent de droite à gauche. Observez en particulier comment $\bar{x}$ accumule les contributions de ses deux successeurs $\phi_1$ et $\phi_2$ lors de la passe arrière.

```{code-cell} python
:tags: [remove-input]

from IPython.display import HTML
from pathlib import Path
import html as _html

_content = Path("_static/ad_stepper.html").read_text()
_doc = f'<!DOCTYPE html><html><head><meta charset="utf-8"></head><body style="margin:0;padding:0">{_content}</body></html>'
HTML(f'<iframe srcdoc="{_html.escape(_doc, quote=True)}" width="100%" height="580" style="border:none;" scrolling="no"></iframe>')
```

### Règles VJP: une bibliothèque d'opérateurs adjoints

L'algorithme du mode arrière fait intervenir les jacobiennes locales $D_u \phi_v$ à chaque étape. Une question reste ouverte: comment calcule-t-on $\bar{v} \, D_u \phi_v$ efficacement, *sans* construire la jacobienne complète?

La réponse repose sur une distinction fondamentale. La jacobienne $\mathbf{J}_f(\mathbf{x}) \in \mathbb{R}^{m \times n}$ est une *représentation matricielle* d'un objet plus abstrait: la différentielle $df_\mathbf{x}$, qui est un opérateur linéaire $df_\mathbf{x}: \mathbb{R}^n \to \mathbb{R}^m$. Ce qui importe dans le mode arrière n'est pas $df_\mathbf{x}$ lui-même, mais son **opérateur adjoint** $df_\mathbf{x}^*: \mathbb{R}^m \to \mathbb{R}^n$, qui envoie les vecteurs du co-domaine vers le domaine (c'est-à-dire qui propage le signal en sens inverse). En coordonnées, le VJP est $\mathbf{u}^\top \mathbf{J}_f(\mathbf{x})$: le produit d'un vecteur adjoint (ligne) par la jacobienne.

Pour définir un opérateur linéaire, il n'est pas nécessaire d'en donner la matrice: on peut spécifier son *action* sur des vecteurs. Une bibliothèque de DA (JAX, PyTorch) maintient pour chaque opération primitive une **règle VJP**: une fonction qui calcule directement $\mathbf{u}^\top \mathbf{J}_f(\mathbf{x})$ à partir de $\mathbf{u}$, $\mathbf{x}$, et éventuellement $f(\mathbf{x})$, en n'utilisant que des opérations arithmétiques simples.

Lorsque deux opérations se composent, $h = g \circ f$, la règle VJP de $h$ est le produit des règles VJP de $g$ et $f$:

$$
\mathbf{u}^\top \mathbf{J}_h = \underbrace{(\mathbf{u}^\top \mathbf{J}_g)}_{\text{appel récursif}} \mathbf{J}_f
$$

La passe arrière n'est rien d'autre que l'exécution récursive de ces règles, de la sortie vers l'entrée. Le système est entièrement **sans matrice jacobienne explicite**: aucune matrice $\mathbf{J}_f$ n'est jamais construite ni stockée.

Le tableau ci-dessous liste les règles VJP pour les opérations clés d'un MLP. À chaque fois, la règle VJP évite de former la jacobienne correspondante:

| Opération | $f(\mathbf{x})$ | Jacobienne (non formée) | Règle VJP: $\mathbf{u}^\top \mathbf{J}_f$ |
|-----------|-----------------|--------------------------|-------------------------------------------|
| Couche affine (entrée $\mathbf{z}$) | $W\mathbf{z} + \mathbf{b}$ | $W \in \mathbb{R}^{m \times n}$ | $\mathbf{u}^\top W$ |
| Couche affine (poids $W$) | $W\mathbf{z} + \mathbf{b}$ | $\mathbf{z}^\top \otimes I_m$ | $\mathbf{u}\mathbf{z}^\top$ (produit externe) |
| Couche affine (biais $\mathbf{b}$) | $W\mathbf{z} + \mathbf{b}$ | $I_m \in \mathbb{R}^{m \times m}$ | $\mathbf{u}$ |
| Activation élémentaire | $\varphi(\mathbf{a})$ | $\operatorname{diag}(\varphi'(\mathbf{a})) \in \mathbb{R}^{m \times m}$ | $\mathbf{u} \odot \varphi'(\mathbf{a})$ |
| Somme $s = \sum_i x_i$ | scalaire | $\mathbf{1}^\top \in \mathbb{R}^{1 \times n}$ | $u \cdot \mathbf{1}$ (diffusion) |

L'exemple de l'activation élémentaire illustre parfaitement le bénéfice: la jacobienne serait une matrice $m \times m$ coûtant $O(m^2)$ en mémoire, alors que la règle VJP, $\mathbf{u} \odot \varphi'(\mathbf{a})$, est un produit élément par élément en $O(m)$.

En JAX, `jax.custom_vjp` permet d'enregistrer exactement ce type de règle pour une opération personnalisée. Écrire une règle VJP correcte pour un nouvel opérateur est une compétence essentielle en apprentissage profond avancé; les exercices 9 à 11 vous entraînent à cette dérivation.

### Exemple: MLP avec une couche cachée

Prenons un réseau à une couche cachée avec la perte des moindres carrés:

$$
\mathcal{L} = \frac{1}{2}\|y - \mathbf{w}_2^\top \varphi(W_1 \mathbf{x} + \mathbf{b}_1) - b_2\|^2
$$

Le graphe de calcul de ce réseau rend explicites toutes les dépendances. Les nœuds en jaune sont les paramètres (feuilles du graphe); la passe avant suit les flèches de gauche à droite, et la passe arrière les remonte de droite à gauche.

```{mermaid}
graph LR
    x("x")
    y("y")
    W1("W1, b1")
    a1("a1 = W1 x + b1")
    z1("z1 = phi(a1)")
    w2("w2, b2")
    yhat("y-hat = w2^T z1 + b2")
    L("L = (1/2)(y - y-hat)^2")

    x --> a1
    W1 --> a1
    a1 -- "phi" --> z1
    z1 --> yhat
    w2 --> yhat
    yhat --> L
    y --> L

    style W1 fill:#fff2cc,stroke:#d6b656
    style w2 fill:#fff2cc,stroke:#d6b656
    style x fill:#dae8fc,stroke:#6c8ebf
    style y fill:#dae8fc,stroke:#6c8ebf
    style L fill:#f8cecc,stroke:#b85450
    style a1 fill:#f5f5f5,stroke:#666666
    style z1 fill:#f5f5f5,stroke:#666666
```

La passe arrière calcule les adjoints en remontant ce graphe nœud par nœud, en appliquant les règles VJP de chaque opération.

La passe avant calcule les valeurs intermédiaires:

$$
\begin{aligned}
\mathbf{a}_1 &= W_1 \mathbf{x} + \mathbf{b}_1 \\
\mathbf{z}_1 &= \varphi(\mathbf{a}_1) \\
\hat{y} &= \mathbf{w}_2^\top \mathbf{z}_1 + b_2 \\
\mathcal{L} &= \frac{1}{2}(y - \hat{y})^2
\end{aligned}
$$

La passe arrière propage les adjoints en sens inverse, couche par couche. La notation $\bar{v}$ désigne l'adjoint du noeud $v$, c'est-à-dire la sensibilité de $\mathcal{L}$ à $v$:

$$
\begin{aligned}
\bar{\hat{y}} &= \hat{y} - y \\[4pt]
\bar{\mathbf{w}}_2 &= \bar{\hat{y}} \, \mathbf{z}_1, \qquad
\bar{b}_2 = \bar{\hat{y}} \\[4pt]
\bar{\mathbf{z}}_1 &= \bar{\hat{y}} \, \mathbf{w}_2 \\[4pt]
\bar{\mathbf{a}}_1 &= \bar{\mathbf{z}}_1 \odot \varphi'(\mathbf{a}_1) \\[4pt]
\bar{W}_1 &= \bar{\mathbf{a}}_1 \, \mathbf{x}^\top, \qquad
\bar{\mathbf{b}}_1 = \bar{\mathbf{a}}_1
\end{aligned}
$$

où $\odot$ désigne le produit élément par élément. Chaque ligne utilise uniquement des quantités déjà calculées, soit lors de la passe avant ($\mathbf{z}_1$, $\mathbf{a}_1$, $\mathbf{x}$), soit lors des étapes précédentes de la passe arrière. La structure est toujours la même: l'adjoint des pré-activations d'une couche est propagé vers l'arrière pour obtenir l'adjoint de la couche précédente.

**Point de contrôle:** Si vous pouvez suivre cet exemple du début à la fin, vous avez compris le mécanisme de la rétropropagation. C'est une instance de l'algorithme du mode arrière de la section précédente, spécialisée à un réseau en chaîne. Si certaines étapes restent floues, l'exercice 3 vous permettra de refaire ce calcul vous-même avec des valeurs numériques.

L'animation interactive ci-dessous déroule la passe avant et la rétropropagation sur un réseau à deux couches avec des valeurs numériques concrètes. Observez comment chaque couche produit les gradients de ses paramètres ($\bar{w}$, $\bar{b}$) tout en propageant le signal vers l'arrière, et comment la dérivée de l'activation ($\sigma'$) joue le rôle de «porte» qui laisse passer ou bloque le gradient.

```{code-cell} python
:tags: [remove-input]

from IPython.display import HTML
from pathlib import Path
import html as _html

_content = Path("_static/nn_backprop_stepper.html").read_text()
_doc = f'<!DOCTYPE html><html><head><meta charset="utf-8"></head><body style="margin:0;padding:0">{_content}</body></html>'
HTML(f'<iframe srcdoc="{_html.escape(_doc, quote=True)}" width="100%" height="580" style="border:none;" scrolling="no"></iframe>')
```

### La liste de Wengert

En 1964, Wengert {cite}`wengert1964simple` a proposé de représenter toute fonction calculable comme une séquence ordonnée d'opérations élémentaires, chacune ayant une dérivée connue. Cette séquence, appelée **liste de Wengert** (ou *tape*, bande), est le graphe de calcul sérialisé en ordre topologique. La contribution de Wengert était de rendre la dérivation *algorithmique*: en décomposant un programme en étapes atomiques et en les écrivant dans l'ordre, une machine peut appliquer la règle de la chaîne mécaniquement, sans intervention humaine.

En mode avant, cette liste guide la propagation des tangentes: chaque étape calcule simultanément sa valeur et sa dérivée, sans rien retenir. Mais en mode arrière, la bande devient indispensable comme structure de *stockage*: il faut rejouer les opérations à rebours, ce qui exige de conserver les valeurs intermédiaires de la passe avant. C'est ce rôle de stockage qui domine dans les bibliothèques modernes de DA (JAX, PyTorch), puisque l'entraînement des réseaux utilise le mode arrière.

Concrètement, pendant la passe avant, chaque opération enregistre sur la bande ses entrées, sa sortie, et une **fonction VJP locale**. Cette fonction est construite au moment de l'opération, et elle *capture* les valeurs intermédiaires dont elle aura besoin plus tard pour calculer le gradient. En programmation, on appelle cela une *fermeture* (*closure*). Prenons l'opération $v_3 = v_1 \cdot v_2$ comme exemple. Au moment du calcul, la passe avant connaît les valeurs de $v_1$ et $v_2$. La règle VJP de la multiplication est $(\bar{v}_1, \bar{v}_2) = (v_2 \cdot \bar{v}_3,\; v_1 \cdot \bar{v}_3)$: elle a besoin des valeurs $v_1$ et $v_2$ de la passe avant. La fermeture les capture:

```python
# Pendant la passe avant, au moment de v3 = v1 * v2 :
v1_val, v2_val = v1, v2           # valeurs connues maintenant

def mul_vjp(v3_bar):              # sera appelée pendant la passe arrière
    return (v2_val * v3_bar,      # gradient pour v1
            v1_val * v3_bar)      # gradient pour v2

tape.append(mul_vjp)              # on stocke la fermeture, pas les valeurs brutes
```

La fonction `mul_vjp` ne sera appelée que plus tard, pendant la passe arrière, avec l'adjoint $\bar{v}_3$ comme argument. Mais elle a déjà accès à `v1_val` et `v2_val`, capturés au moment de sa création. C'est ce mécanisme de capture qui rend la bande autonome: chaque entrée contient tout ce qu'il faut pour calculer sa contribution au gradient, sans consulter à nouveau le programme original.

La passe arrière rejoue la liste à rebours, en appelant chaque VJP locale dans l'ordre inverse. Contrairement au DAG, la bande est une **liste ordonnée** (un tableau linéaire), ce qui la rend simple à parcourir dans les deux sens. Les flèches pleines montrent l'enregistrement (passe avant); les flèches pointillées montrent le rejeu (passe arrière).

```{mermaid}
graph LR
    E1["<b>Entrée 1</b><br/>op: <tt>sin</tt><br/>entrée: x &rarr; v₁<br/><i>VJP: x̄ += cos(x)·v̄₁</i>"]
    E2["<b>Entrée 2</b><br/>op: <tt>add</tt><br/>entrées: x, y &rarr; v₂<br/><i>VJP: x̄ += v̄₂ &nbsp; ȳ += v̄₂</i>"]
    E3["<b>Entrée 3</b><br/>op: <tt>mul</tt><br/>entrées: v₁, v₂ &rarr; v₃<br/><i>VJP: v̄₁ += v₂·v̄₃ &nbsp; v̄₂ += v₁·v̄₃</i>"]

    E1 -->|"① enregistrer"| E2
    E2 -->|"② enregistrer"| E3
    E3 -.->|"③ rejouer"| E2
    E2 -.->|"④ rejouer"| E1

    style E1 fill:#eef3fb,stroke:#6c8ebf
    style E2 fill:#eef3fb,stroke:#6c8ebf
    style E3 fill:#f8cecc,stroke:#b85450
```

La bande démarre vide et grandit à chaque opération (flèches pleines, gauche à droite). Une fois $f$ calculée, la passe arrière initialise $\bar{v}_3 = 1$ puis remonte la liste à rebours (flèches pointillées, droite à gauche).

Le tableau ci-dessous détaille le contenu de chaque entrée et les formules de VJP associées. La barre $\bar{v}$ désigne l'adjoint $\partial f / \partial v$; la passe arrière parcourt les étapes 3 → 2 → 1.

| Étape | Opération | Entrées | Sortie | VJP locale |
|:-----:|:---------:|:-------:|:------:|:-----------|
| 1 | sin | $x$ | $v_1$ | $\bar{x}\ {+}{=}\ \cos(x) \cdot \bar{v}_1$ |
| 2 | add | $x, y$ | $v_2$ | $\bar{x}\ {+}{=}\ \bar{v}_2$, $\quad \bar{y}\ {+}{=}\ \bar{v}_2$ |
| 3 | mul | $v_1, v_2$ | $v_3$ | $\bar{v}_1\ {+}{=}\ v_2 \cdot \bar{v}_3$, $\quad \bar{v}_2\ {+}{=}\ v_1 \cdot \bar{v}_3$ |

Chaque ligne de la bande correspond à une opération élémentaire. La passe arrière part de l'adjoint $\bar{v}_3 = 1$ (gradient de $f$ par rapport à lui-même) et remonte: l'étape 3 envoie des gradients à $v_1$ et $v_2$, puis les étapes 2 et 1 envoient leurs gradients à $x$ et $y$.

### Le traceur

Comment une bibliothèque comme JAX construit-elle cette bande automatiquement, sans modifier le programme utilisateur? La réponse est le **traceur** (*tracer*).

Lorsque JAX différentie une fonction, il ne l'appelle pas avec des nombres ordinaires. Il l'appelle avec des objets spéciaux, des traceurs, qui se font passer pour des nombres mais enregistrent discrètement toutes les opérations qu'on leur applique.

Concrètement, un traceur est un objet qui:
1. **Stocke une valeur concrète** (le résultat numérique de l'opération),
2. **Enregistre l'opération sur la bande**, avec ses entrées et une fermeture (*closure*) qui sait calculer les gradients locaux,
3. **Retourne un nouveau traceur** comme résultat, de sorte que les opérations suivantes soient également interceptées.

Le diagramme ci-dessous montre comment les objets traceurs se construisent et se connectent lors de l'évaluation de $f(x,y) = \sin(x) \cdot (x + y)$. Chaque noeud porte sa valeur concrète et son adjoint (initialement nul); les arêtes représentent les dépendances enregistrées par chaque fermeture.

```{mermaid}
graph TD
    x["<b>x</b><br/>data = 0.5<br/>grad = 0.0"]
    y["<b>y</b><br/>data = 1.2<br/>grad = 0.0"]
    v1["<b>v₁ = sin(x)</b><br/>data = 0.479<br/>grad = 0.0<br/><i>passe arr.: x.grad += cos(x)·v̄₁</i>"]
    v2["<b>v₂ = x + y</b><br/>data = 1.7<br/>grad = 0.0<br/><i>passe arr.: x.grad += v̄₂, y.grad += v̄₂</i>"]
    v3["<b>v₃ = v₁ · v₂</b><br/>data = 0.814<br/><b>grad = 1.0</b><br/><i>passe arr.: v₁.grad += v₂·v̄₃, v₂.grad += v₁·v̄₃</i>"]

    x -->|"sin"| v1
    x -->|"add"| v2
    y -->|"add"| v2
    v1 -->|"mul"| v3
    v2 -->|"mul"| v3

    style x  fill:#dae8fc,stroke:#6c8ebf
    style y  fill:#dae8fc,stroke:#6c8ebf
    style v3 fill:#f8cecc,stroke:#b85450
    style v1 fill:#f5f5f5,stroke:#888
    style v2 fill:#f5f5f5,stroke:#888
```

La passe arrière initialise $\bar{v}_3 = 1$, puis parcourt les arêtes à rebours: chaque fermeture accumule les adjoints dans les noeuds parents. Quand les deux fermetures de $v_1$ et $v_2$ ont été appelées, `x.grad` contient la somme des deux contributions.

Le tableau ci-dessous montre la correspondance entre l'exécution Python et la bande construite automatiquement. Chaque ligne de code qui effectue une opération tracée ajoute une entrée sur la bande, avec la règle VJP correspondante.

| Ligne Python | Bande: opération enregistrée | VJP locale |
|:-------------|:-----------------------------|:-----------|
| `x = Var(0.5)` | *(entrée, pas d'opération)* | — |
| `y = Var(1.2)` | *(entrée, pas d'opération)* | — |
| `v1 = sin(x)` | `(sin, x → v₁)` | $\bar{x}\ {+}{=}\ \cos(x) \cdot \bar{v}_1$ |
| `v2 = x + y` | `(add, x, y → v₂)` | $\bar{x}\ {+}{=}\ \bar{v}_2$, $\quad \bar{y}\ {+}{=}\ \bar{v}_2$ |
| `v3 = v1 * v2` | `(mul, v₁, v₂ → v₃)` | $\bar{v}_1\ {+}{=}\ v_2 \cdot \bar{v}_3$, $\quad \bar{v}_2\ {+}{=}\ v_1 \cdot \bar{v}_3$ |

La passe arrière parcourt la bande de bas en haut (étapes 3 → 2 → 1), en appelant chaque règle VJP.

L'exécution Python se déroule normalement, ligne par ligne. Python ne sait pas qu'il trace un graphe: il appelle simplement les méthodes `__add__`, `__mul__`, `sin` sur les objets traceurs, et ces méthodes enregistrent discrètement les opérations. Quand l'exécution est terminée, la bande est complète, et la passe arrière peut s'exécuter.

Ce mécanisme explique pourquoi la dérivation automatique gère naturellement les boucles et les conditions: Python les exécute normalement, et les traceurs enregistrent les opérations qui sont effectivement effectuées lors de cette exécution particulière.

### L'astuce d'importation

Le mécanisme du traceur explique une convention qui surprend souvent les débutants. Dans tout code JAX, on écrit:

```python
import jax.numpy as jnp   # et non: import numpy as np
```

Pourquoi? Lorsque JAX différentie une fonction, il lui passe des traceurs à la place des tableaux NumPy ordinaires. Si vous appelez `np.sin(tracer)`, NumPy ne connaît pas les traceurs: il va tenter de convertir l'objet en tableau numérique, ce qui casse la trace et donne un résultat incorrect (ou lève une erreur).

En revanche, `jnp.sin(tracer)` est une opération que JAX connaît. JAX intercepte l'appel, enregistre l'opération sur la bande, calcule la valeur concrète, et retourne un nouveau traceur. La trace reste intacte.

```python
import jax
import jax.numpy as jnp
import numpy as np

def f_jnp(x):
    return jnp.sin(x) * x  # correct: jnp intercepte le traceur

def f_np(x):
    return np.sin(x) * x   # incorrect: np ne comprend pas les traceurs

grad_jnp = jax.grad(f_jnp)(1.0)   # fonctionne: retourne cos(1)*1 + sin(1) ≈ 1.382
# grad_np = jax.grad(f_np)(1.0)   # lèverait une erreur ou donnerait un résultat faux
```

`jax.numpy` est un espace de noms qui réimplémente toutes les fonctions NumPy de manière à intercepter les traceurs. Pour les tableaux NumPy ordinaires (sans traceur), `jnp` et `np` produisent les mêmes résultats numériques. La différence n'apparaît que pendant la trace.

### Implémentation minimale

*Cette sous-section est optionnelle pour IFT3395. Elle montre comment implémenter un moteur de dérivation automatique en mode arrière en une soixantaine de lignes de Python pur.*

La section précédente a identifié trois mécanismes: les règles VJP locales, la bande d'enregistrement, et la passe arrière. Nous allons maintenant les assembler en une implémentation fonctionnelle, structurée comme le seraient JAX ou autograd en version simplifiée {cite}`maclaurin2015autograd`. L'architecture se décompose en trois parties:

1. **Une bibliothèque de règles VJP.** Pour chaque opération primitive, une fonction qui prend les résidus (valeurs de la passe avant nécessaires au calcul du gradient) et le cotangent amont $\bar{v}$, et retourne les cotangents pour chaque entrée. Ces fonctions sont les mêmes que celles du tableau de la section précédente.
2. **Un traceur avec bande.** Un objet `Var` qui encapsule un flottant et enregistre chaque opération sur une bande globale. C'est l'analogue simplifié des traceurs de JAX.
3. **Une fonction `grad`.** Un opérateur d'ordre supérieur, analogue à `jax.grad`, qui trace la fonction puis parcourt la bande à rebours en appelant les règles VJP.

```{code-cell} ipython3
import math

# ---- 1. Bibliothèque de règles VJP ----
# Signature commune: vjp(résidus, cotangent_sortie) → cotangents_entrées

def add_vjp(res, g):
    return (g, g)                        # ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1

def mul_vjp(res, g):
    a, b = res
    return (b * g, a * g)                # ∂(a·b)/∂a = b, ∂(a·b)/∂b = a

def sin_vjp(res, g):
    (a,) = res
    return (math.cos(a) * g,)            # ∂sin(a)/∂a = cos(a)

def relu_vjp(res, g):
    (a,) = res
    return (float(a > 0) * g,)           # ∂relu(a)/∂a = 𝟙(a > 0)


# ---- 2. Traceur et bande ----

_tape = []      # bande globale: [(vjp_fn, résidus, ids_entrées, id_sortie)]
_n_vars = 0     # compteur d'identifiants

class Var:
    """Traceur scalaire: encapsule un flottant et un identifiant unique."""

    def __init__(self, data):
        global _n_vars
        self.data = float(data)
        self.id = _n_vars
        _n_vars += 1

    def _record(self, vjp_fn, res, inputs, out_data):
        """Enregistre une opération sur la bande et retourne un nouveau Var."""
        out = Var(out_data)
        _tape.append((vjp_fn, res, [v.id for v in inputs], out.id))
        return out

    def __add__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        return self._record(add_vjp, (self.data, other.data),
                            [self, other], self.data + other.data)

    def __radd__(self, other): return self.__add__(other)

    def __mul__(self, other):
        other = other if isinstance(other, Var) else Var(other)
        return self._record(mul_vjp, (self.data, other.data),
                            [self, other], self.data * other.data)

    def __rmul__(self, other): return self.__mul__(other)

    def sin(self):
        return self._record(sin_vjp, (self.data,),
                            [self], math.sin(self.data))

    def relu(self):
        return self._record(relu_vjp, (self.data,),
                            [self], max(0.0, self.data))


# ---- 3. Fonction grad (analogue à jax.grad) ----

def grad(f):
    """Retourne une fonction qui calcule le gradient de f."""
    def grad_fn(*args):
        global _tape, _n_vars
        _tape, _n_vars = [], 0               # réinitialiser la bande

        # Passe avant: tracer l'exécution
        traced = [Var(a) for a in args]
        result = f(*traced)

        # Passe arrière: propager les cotangents
        adjoints = [0.0] * _n_vars
        adjoints[result.id] = 1.0            # ∂f/∂f = 1

        for vjp_fn, res, in_ids, out_id in reversed(_tape):
            cotangents = vjp_fn(res, adjoints[out_id])
            for idx, ct in zip(in_ids, cotangents):
                adjoints[idx] += ct          # accumulation (embranchement)

        return tuple(adjoints[v.id] for v in traced)
    return grad_fn
```

La séparation en trois parties n'est pas un choix esthétique: c'est la structure réelle des bibliothèques de DA. Dans JAX, les règles VJP sont enregistrées via `jax.custom_vjp`, la bande est construite par le traceur interne, et `jax.grad` orchestre la passe arrière. Notre implémentation reproduit cette architecture en miniature.

Vérifions sur $f(x, y) = \sin(x) \cdot (x + y)$:

```{code-cell} ipython3
import math

# --- Valeurs de test ---
x0, y0 = 0.5, 1.2

# --- Avec notre moteur de DA ---
def f(x, y):
    return x.sin() * (x + y)

df_dx, df_dy = grad(f)(x0, y0)

print(f'f({x0}, {y0})          = {math.sin(x0) * (x0 + y0):.6f}')
print(f'∂f/∂x (AD)           = {df_dx:.6f}')
print(f'∂f/∂y (AD)           = {df_dy:.6f}')

# --- Vérification analytique ---
df_dx_exact = math.cos(x0) * (x0 + y0) + math.sin(x0)
df_dy_exact = math.sin(x0)
print(f'∂f/∂x (exact)        = {df_dx_exact:.6f}')
print(f'∂f/∂y (exact)        = {df_dy_exact:.6f}')

# --- Vérification par différences finies ---
eps = 1e-5
df_dx_num = (math.sin(x0+eps)*(x0+eps+y0) - math.sin(x0-eps)*(x0-eps+y0)) / (2*eps)
df_dy_num = (math.sin(x0)*(x0+y0+eps)     - math.sin(x0)*(x0+y0-eps))     / (2*eps)
print(f'∂f/∂x (diff. fin.)  = {df_dx_num:.6f}')
print(f'∂f/∂y (diff. fin.)  = {df_dy_num:.6f}')
```

Les trois méthodes sont en accord. Remarquez que `grad` est une fonction d'ordre supérieur qui retourne une nouvelle fonction, exactement comme `jax.grad`. L'accumulation des adjoints (ligne `adjoints[idx] += ct`) gère automatiquement le cas où une variable contribue à plusieurs branches du calcul: c'est la somme des deux chemins pour $x$.

### La programmation différentiable

Les bibliothèques modernes comme JAX, PyTorch et TensorFlow implémentent la dérivation automatique de manière générale: toute fonction composée d'opérations dont on connaît les dérivées locales peut être différentiée automatiquement. C'est le paradigme de la **programmation différentiable** (*differentiable programming*).

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

## Implémentation

Cette section réunit les concepts du chapitre dans une implémentation complète d'un MLP en NumPy. L'optimiseur Adam utilisé ici est décrit en détail au chapitre suivant. Le code est volontairement auto-contenu et commenté pas à pas: l'objectif est de rendre le lien entre les équations et le code aussi direct que possible.

### Classe MLP avec Adam

```{code-cell} python

import numpy as np

class MLP:
    """
    Perceptron multicouche à une couche cachée.
    Activation cachée: ReLU. Activation de sortie: sigmoïde (classification binaire).
    Optimiseur: Adam.
    """

    def __init__(self, n_input, n_hidden, n_output,
                 eta=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 lam=0.0, p_drop=0.0, seed=0):
        rng = np.random.default_rng(seed)
        # Initialisation He pour ReLU
        self.W1 = rng.standard_normal((n_input,  n_hidden)) * np.sqrt(2 / n_input)
        self.b1 = np.zeros(n_hidden)
        # Initialisation Glorot pour la couche de sortie (sigmoïde)
        self.W2 = rng.standard_normal((n_hidden, n_output)) * np.sqrt(2 / (n_hidden + n_output))
        self.b2 = np.zeros(n_output)

        self.eta   = eta
        self.beta1 = beta1;  self.beta2 = beta2;  self.eps = eps
        self.lam   = lam     # décroissance des poids
        self.p_drop = p_drop # taux de dropout

        # État interne Adam (moments)
        self._t = 0
        self._m = {k: np.zeros_like(v)
                   for k, v in [('W1',self.W1),('b1',self.b1),
                                 ('W2',self.W2),('b2',self.b2)]}
        self._s = {k: np.zeros_like(v) for k, v in self._m.items()}

    # ------------------------------------------------------------------
    def _relu(self, x):     return np.maximum(0, x)
    def _sigmoid(self, x):  return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

    # ------------------------------------------------------------------
    def forward(self, X, training=False):
        """Passe avant. Retourne les sorties et les caches."""
        a1 = X @ self.W1 + self.b1          # pré-activations couche cachée
        z1 = self._relu(a1)                  # activations ReLU

        # Dropout (entraînement uniquement)
        if training and self.p_drop > 0:
            mask = (np.random.rand(*z1.shape) > self.p_drop) / (1 - self.p_drop)
            z1 = z1 * mask
        else:
            mask = np.ones_like(z1)

        a2   = z1 @ self.W2 + self.b2       # pré-activations couche de sortie
        pred = self._sigmoid(a2)             # probabilités
        cache = {'X': X, 'a1': a1, 'z1': z1, 'mask': mask}
        return pred, cache

    # ------------------------------------------------------------------
    def backward(self, pred, y, cache):
        """
        Passe arrière. Retourne les gradients par rapport à tous les paramètres.
        y: vecteur colonne de cibles binaires.
        """
        B = len(y)
        X, a1, z1, mask = cache['X'], cache['a1'], cache['z1'], cache['mask']

        # Gradient de l'entropie croisée + sigmoïde: dp = pred - y
        dp  = (pred - y) / B

        # Couche de sortie
        dW2 = z1.T @ dp + self.lam * self.W2 / B
        db2 = dp.sum(axis=0)

        # Propagation vers la couche cachée
        dz1 = dp @ self.W2.T
        dz1 = dz1 * mask            # rétropropagation à travers le dropout
        da1 = dz1 * (a1 > 0)        # dérivée de ReLU

        # Couche cachée
        dW1 = X.T @ da1 + self.lam * self.W1 / B
        db1 = da1.sum(axis=0)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    # ------------------------------------------------------------------
    def _adam_update(self, grads):
        """Applique une étape Adam à tous les paramètres."""
        self._t += 1
        for name, param in [('W1',self.W1),('b1',self.b1),
                              ('W2',self.W2),('b2',self.b2)]:
            g = grads[name]
            self._m[name] = self.beta1 * self._m[name] + (1 - self.beta1) * g
            self._s[name] = self.beta2 * self._s[name] + (1 - self.beta2) * g**2
            mhat = self._m[name] / (1 - self.beta1**self._t)
            shat = self._s[name] / (1 - self.beta2**self._t)
            param -= self.eta * mhat / (np.sqrt(shat) + self.eps)

    # ------------------------------------------------------------------
    def train_step(self, X, y):
        """Une étape d'entraînement sur un mini-lot (X, y)."""
        pred, cache = self.forward(X, training=True)
        grads = self.backward(pred, y.reshape(-1,1).astype(float), cache)
        self._adam_update(grads)
        y_col = y.reshape(-1,1).astype(float)
        p = np.clip(pred, 1e-7, 1-1e-7)
        loss = -np.mean(y_col*np.log(p) + (1-y_col)*np.log(1-p))
        return loss

    def predict_proba(self, X):
        pred, _ = self.forward(X, training=False)
        return pred

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int).ravel()
```

## Le MLP en pratique

Les sections précédentes ont présenté le MLP comme un objet mathématique: une composition de transformations affines et de non-linéarités. Mais à quoi ressemble cette composition pour un problème concret de régression ou de classification? La réponse nous ramène directement aux modèles linéaires des chapitres 2 et 3.

### Régression avec un MLP

Au chapitre 2, la régression linéaire prédit la moyenne d'une gaussienne conditionnelle par une transformation affine:

$$
\hat{y} = \mathbf{w}^\top \mathbf{x} + b
$$

Passer à un MLP revient à composer des transformations affines et des non-linéarités avant cette sortie linéaire. Pour un réseau à deux couches cachées avec $\mathbf{x} \in \mathbb{R}^d$:

$$
\hat{y}(\mathbf{x}) = \mathbf{w}_3^\top\, \varphi(W_2\, \varphi(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2) + b_3
$$

où $W_1 \in \mathbb{R}^{h \times d}$, $W_2 \in \mathbb{R}^{h \times h}$, $\mathbf{w}_3 \in \mathbb{R}^h$, et $\varphi$ est une activation (typiquement ReLU). La dernière couche est linéaire (pas d'activation), et la perte reste la somme des carrés, exactement comme au chapitre 2.

Si l'on veut prédire un vecteur $\mathbf{y} \in \mathbb{R}^K$ (par exemple des coordonnées, ou plusieurs cibles simultanément), la dernière couche devient une transformation affine $W_3 \in \mathbb{R}^{K \times h}$:

$$
\hat{\mathbf{y}}(\mathbf{x}) = W_3\, \varphi(W_2\, \varphi(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2) + \mathbf{b}_3
$$

La perte est la somme des carrés sur les $K$ sorties: $\sum_{k=1}^K (y_k - \hat{y}_k)^2$.

### Classification avec un MLP

Au chapitre 3, la régression logistique modélise la probabilité de la classe positive par:

$$
p(y = 1 | \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b)
$$

Un MLP pour la classification binaire compose des couches cachées avant cette sigmoïde:

$$
p(y = 1 | \mathbf{x}) = \sigma\!\Big(\mathbf{w}_3^\top\, \varphi(W_2\, \varphi(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2) + b_3\Big)
$$

La perte est l'entropie croisée binaire, comme en régression logistique. Pour la classification multiclasse avec $K$ classes, la sigmoïde est remplacée par un softmax et la dernière couche produit $K$ sorties:

$$
p(y = k | \mathbf{x}) = \text{softmax}_k\!\Big(W_3\, \varphi(W_2\, \varphi(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2) + \mathbf{b}_3\Big)
$$

La perte est l'entropie croisée catégorielle.

### Extracteur de caractéristiques et tête linéaire

Dans tous les cas, on peut écrire le réseau comme la composition de deux parties. Posons $\boldsymbol{\phi}(\mathbf{x}; \boldsymbol{\theta}_\phi) = \varphi(W_2\, \varphi(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2) \in \mathbb{R}^h$, la représentation apprise par les couches cachées. Les prédictions s'écrivent alors:

$$
\begin{aligned}
\text{Régression:} \quad & \hat{y} = \mathbf{w}_3^\top \boldsymbol{\phi}(\mathbf{x}) + b_3 \\
\text{Classification binaire:} \quad & p(y = 1 | \mathbf{x}) = \sigma(\mathbf{w}_3^\top \boldsymbol{\phi}(\mathbf{x}) + b_3) \\
\text{Classification multiclasse:} \quad & p(y = k | \mathbf{x}) = \text{softmax}_k(W_3\, \boldsymbol{\phi}(\mathbf{x}) + \mathbf{b}_3)
\end{aligned}
$$

C'est exactement l'équation {eq}`eq:nn-key-idea` du début du chapitre. La dernière couche est un modèle linéaire (chapitre 2) ou une régression logistique (chapitre 3) appliqué aux caractéristiques apprises $\boldsymbol{\phi}(\mathbf{x})$. Les modèles des chapitres 2 et 3 sont le cas particulier $\boldsymbol{\phi}(\mathbf{x}) = \mathbf{x}$ (aucune couche cachée).

### Limites du MLP

Le MLP traite son entrée comme un vecteur plat $\mathbf{x} \in \mathbb{R}^d$: chaque multiplication $W_\ell \mathbf{z}_{\ell-1}$ opère sur toutes les composantes de $\mathbf{z}_{\ell-1}$ sans distinction. La matrice $W_\ell$ est pleine (dense), ce qui signifie que chaque composante de la sortie dépend de toutes les composantes de l'entrée. Il n'y a aucune notion de structure spatiale ou temporelle. Pour des données tabulaires (âge, revenu, nombre de pièces), c'est approprié: il n'y a pas d'ordre naturel entre les variables.

Mais pour une image de $28 \times 28$ pixels, le MLP la transforme en un vecteur de 784 entrées. La matrice $W_1 \in \mathbb{R}^{h \times 784}$ mélange toutes les positions spatiales: elle ne sait pas que le pixel $(0, 0)$ est voisin du pixel $(0, 1)$ mais éloigné du pixel $(27, 27)$. Pour une phrase de 10 mots, le MLP a besoin d'une entrée $\mathbf{x} \in \mathbb{R}^{10d}$ de taille fixe. Que fait-on avec une phrase de 20 mots?

Ces limitations motivent des architectures qui exploitent la structure des données. Les réseaux convolutifs remplacent la matrice dense par une opération de convolution qui respecte la structure spatiale des images. Les réseaux récurrents traitent les séquences élément par élément en maintenant un état interne. Le mécanisme d'attention et les transformeurs permettent à chaque position d'une séquence de consulter directement toutes les autres, sans contrainte de longueur fixe.

## Résumé

Ce chapitre a montré comment les réseaux de neurones s'inscrivent dans la progression des modèles vus dans les chapitres précédents. Le point de départ est toujours le cadre de maximum de vraisemblance: un modèle prédit les paramètres d'une distribution conditionnelle. La nouveauté est que la transformation des entrées (la fonction $\boldsymbol{\phi}$) est désormais apprise plutôt que fixée à l'avance. Le problème XOR a illustré pourquoi cette flexibilité est nécessaire: certaines fonctions simples sont inaccessibles aux modèles linéaires, et une couche cachée suffit à les résoudre en transformant l'espace des entrées.

La dérivation automatique calcule les gradients en décomposant un programme en opérations élémentaires et en appliquant la règle de la chaîne. Le mode arrière (VJP) produit le gradient par rapport à tous les paramètres en une seule passe, ce qui en fait la base de l'entraînement des réseaux. Les bibliothèques modernes (JAX, PyTorch) implémentent ce mécanisme automatiquement via le traçage d'opérations.

En pratique, un MLP se décompose en un extracteur de caractéristiques (les couches cachées) et une tête linéaire (la couche de sortie), ce qui généralise directement les modèles des chapitres 2 et 3. Cependant, le MLP traite ses entrées comme des vecteurs plats, sans exploiter la structure spatiale ou temporelle des données. Le chapitre suivant couvre les algorithmes d'optimisation, la stabilisation de l'entraînement et la régularisation. Les chapitres sur les réseaux récurrents, l'attention et les transformeurs présentent ensuite des architectures qui remédient aux limitations du MLP.

```{admonition} Ce que vous devez retenir
:class: tip

1. **Le problème XOR motive les architectures multicouches.** Un modèle linéaire ne peut pas séparer XOR; une couche cachée transforme l'espace et rend le problème linéairement séparable.

2. **Les réseaux de neurones apprennent leurs caractéristiques.** Contrairement aux modèles à expansion fixe (chapitre 4), la transformation $\boldsymbol{\phi}$ est paramétrique et optimisée avec les données.

3. **La non-linéarité est indispensable.** Sans fonctions d'activation, empiler des couches linéaires ne donne qu'une transformation linéaire équivalente à une seule couche.

4. **La dérivation automatique en mode arrière calcule les gradients efficacement.** Un seul VJP (avec $\mathbf{u} = 1$) propage le gradient de la perte vers l'arrière à travers tout le graphe de calcul, quel que soit le nombre de paramètres.

5. **Un MLP en pratique = extracteur de caractéristiques + tête linéaire.** Les couches cachées construisent une représentation $\boldsymbol{\phi}(\mathbf{x})$, et la couche de sortie effectue la régression ou la classification linéaire sur cette représentation. Les modèles des chapitres 2 et 3 sont le cas particulier $\boldsymbol{\phi}(\mathbf{x}) = \mathbf{x}$.
```

## Exercices

Les exercices ★ vérifient la compréhension de base. Les exercices ★★ demandent d'appliquer les concepts à des calculs concrets. Les exercices ★★★ approfondissent le sujet et sont optionnels pour IFT3395.

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

Ce maximum de 0,25 explique la dissolution du gradient: à chaque couche utilisant la sigmoïde, le gradient est multiplié par un facteur d'au plus 0,25.
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
\bar{\hat{y}} &= 0{,}4 - 1 = -0{,}6 \\
\bar{\mathbf{w}}_2 &= -0{,}6 \times [0{,}5, 0]^\top = [-0{,}3, 0]^\top \\
\bar{b}_2 &= -0{,}6 \\
\bar{\mathbf{z}}_1 &= -0{,}6 \times [0{,}8, 0{,}4]^\top = [-0{,}48, -0{,}24]^\top \\
\bar{\mathbf{a}}_1 &= [-0{,}48, -0{,}24]^\top \odot [1, 0]^\top = [-0{,}48, 0]^\top \\
\bar{\mathbf{w}}_1 &= [-0{,}48, 0]^\top \times 1 = [-0{,}48, 0]^\top \\
\bar{\mathbf{b}}_1 &= [-0{,}48, 0]^\top
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


````{admonition} Exercice 5: Gradient de l'entropie croisée avec softmax ★★★ (optionnel IFT3395)
:class: hint dropdown

Cet exercice dérive un résultat très utilisé en pratique: le gradient de l'entropie croisée par rapport aux pré-activations de la couche softmax est une soustraction simple.

Soit $\mathbf{a} \in \mathbb{R}^K$ le vecteur de pré-activations de la dernière couche, $\mathbf{p} = \text{softmax}(\mathbf{a})$ les probabilités prédites, et $c \in \{1,\ldots,K\}$ la vraie classe. La perte est $\mathcal{L} = -\log p_c$.

1. Montrez que $\frac{\partial p_k}{\partial a_j} = p_k(\delta_{kj} - p_j)$, où $\delta_{kj}$ est le delta de Kronecker.

2. En utilisant la règle de la chaîne, montrez que:

$$
\frac{\partial \mathcal{L}}{\partial a_j} = p_j - \mathbb{1}(j = c)
$$

3. Écrivez ce résultat sous forme vectorielle: $\nabla_{\mathbf{a}} \mathcal{L} = \mathbf{p} - \mathbf{e}_c$, où $\mathbf{e}_c$ est le vecteur unité avec 1 en position $c$.

4. Pourquoi ce résultat est-il remarquable du point de vue de l'implémentation?
````

````{admonition} Solution Exercice 5
:class: dropdown

**1. Jacobienne du softmax:**

Par définition, $p_k = \frac{e^{a_k}}{\sum_j e^{a_j}}$. Notons $Z = \sum_j e^{a_j}$.

Pour $k = j$:
$$
\frac{\partial p_k}{\partial a_k} = \frac{e^{a_k} Z - e^{a_k} e^{a_k}}{Z^2} = p_k - p_k^2 = p_k(1-p_k) = p_k(\delta_{kk} - p_k)
$$

Pour $k \neq j$:
$$
\frac{\partial p_k}{\partial a_j} = \frac{-e^{a_k} e^{a_j}}{Z^2} = -p_k p_j = p_k(\delta_{kj} - p_j)
$$

Les deux cas se résument à $\frac{\partial p_k}{\partial a_j} = p_k(\delta_{kj} - p_j)$.

**2. Gradient de la perte:**

$$
\frac{\partial \mathcal{L}}{\partial a_j} = \sum_k \frac{\partial \mathcal{L}}{\partial p_k}\frac{\partial p_k}{\partial a_j}
$$

La seule composante non nulle de $\frac{\partial \mathcal{L}}{\partial p_k} = \frac{\partial}{\partial p_k}(-\log p_c)$ est pour $k = c$, où elle vaut $-1/p_c$.

$$
\frac{\partial \mathcal{L}}{\partial a_j} = -\frac{1}{p_c} \cdot p_c(\delta_{cj} - p_j) = -({\delta_{cj} - p_j}) = p_j - \mathbb{1}(j=c)
$$

**3. Forme vectorielle:** $\nabla_{\mathbf{a}} \mathcal{L} = \mathbf{p} - \mathbf{e}_c$.

**4. Remarque d'implémentation:**

Le gradient par rapport aux pré-activations de la couche softmax se réduit à soustraire 1 à la probabilité prédite pour la vraie classe. On n'a pas besoin de calculer explicitement la jacobienne $K \times K$ du softmax (qui serait coûteuse pour de grandes sorties). En pratique:

```python
def softmax_cross_entropy_grad(logits, c):
    """Gradient de l'entropie croisée par rapport aux logits."""
    exp_a = np.exp(logits - logits.max())  # stabilité numérique
    p = exp_a / exp_a.sum()
    p[c] -= 1  # soustraction de e_c
    return p
```
````

````{admonition} Exercice 6: VJP de la couche affine ★★
:class: hint dropdown

Soit $f(\mathbf{z}) = W\mathbf{z} + \mathbf{b}$ avec $W \in \mathbb{R}^{m \times n}$, $\mathbf{z} \in \mathbb{R}^n$, et un vecteur adjoint $\mathbf{u} \in \mathbb{R}^m$.

1. Écrivez la jacobienne $\mathbf{J}_f$ par rapport à $\mathbf{z}$. Quelle est sa taille?

2. Calculez $\mathbf{u}^\top \mathbf{J}_f$ (la règle VJP par rapport à $\mathbf{z}$). Montrez que le résultat est $\mathbf{u}^\top W$. Interprétez: la multiplication à gauche par $W$ (JVP) se transpose en multiplication à droite par $W$ (VJP).

3. Traitez maintenant $W$ comme variable. Soit $g(W) = W\mathbf{z} + \mathbf{b}$ avec $\mathbf{z}$ fixé. Écrivez $g$ composante par composante: $(g(W))_i = \sum_j W_{ij} z_j + b_i$. En déduire $\frac{\partial g_i}{\partial W_{kl}}$, puis la règle VJP par rapport à $W$.

4. Montrez que la règle VJP par rapport à $W$ est $\mathbf{u}\mathbf{z}^\top$ (produit externe), et par rapport à $\mathbf{b}$ est $\mathbf{u}$.

5. Retrouvez ces formules dans les équations de rétropropagation de la section "Exemple: MLP avec une couche cachée". Comment s'appellent-elles dans ce contexte?
````

````{admonition} Solution Exercice 6
:class: dropdown

**1. Jacobienne par rapport à $\mathbf{z}$:**

$(g(\mathbf{z}))_i = \sum_j W_{ij} z_j + b_i$, donc $\frac{\partial g_i}{\partial z_j} = W_{ij}$.

La jacobienne est $\mathbf{J}_f = W \in \mathbb{R}^{m \times n}$.

**2. VJP par rapport à $\mathbf{z}$:**

$$
\mathbf{u}^\top \mathbf{J}_f = \mathbf{u}^\top W \in \mathbb{R}^{1 \times n}
$$

Le JVP multiplie $W$ à gauche ($W \mathbf{v}$); le VJP multiplie $W$ à droite ($\mathbf{u}^\top W$). Le même opérateur, appliqué dans l'autre sens. Aucune matrice supplémentaire n'est formée: $W$ est déjà disponible depuis la passe avant.

**3. Jacobienne par rapport à $W$:**

Vectorisons $W$ en $\operatorname{vec}(W) \in \mathbb{R}^{mn}$ (concaténation des colonnes). La jacobienne $\mathbf{J}_g \in \mathbb{R}^{m \times mn}$ a pour blocs $(\mathbf{J}_g)_{i, (k-1)n+l} = \frac{\partial g_i}{\partial W_{kl}} = z_l \cdot \mathbb{1}(i=k)$.

**4. VJP par rapport à $W$:**

Sans vectoriser: le résultat du VJP doit avoir la même forme que $W$, soit $\mathbb{R}^{m \times n}$. On a:

$$
\left(\mathbf{u}^\top \mathbf{J}_W\right)_{kl} = \sum_i u_i \frac{\partial g_i}{\partial W_{kl}} = u_k z_l
$$

Ce qui donne la matrice $\mathbf{u}\mathbf{z}^\top \in \mathbb{R}^{m \times n}$, un produit externe $O(mn)$, exactement le coût minimal pour produire une matrice de cette taille.

Par rapport à $\mathbf{b}$: $\frac{\partial g_i}{\partial b_k} = \mathbb{1}(i=k)$, donc $\mathbf{u}^\top \mathbf{J}_\mathbf{b} = \mathbf{u}^\top$.

**5. Correspondance avec la rétropropagation:**

Dans la section "Exemple: MLP avec une couche cachée", avec $\mathbf{u} = \bar{\mathbf{a}}_1$:

$$
\bar{W}_1 = \bar{\mathbf{a}}_1 \, \mathbf{x}^\top = \mathbf{u}\mathbf{z}^\top, \qquad
\bar{\mathbf{b}}_1 = \bar{\mathbf{a}}_1 = \mathbf{u}
$$

Ce sont exactement les règles VJP dérivées ci-dessus, appliquées avec $\mathbf{z} = \mathbf{x}$.
````

````{admonition} Exercice 7: VJP de l'activation élémentaire ★★
:class: hint dropdown

Soit $g: \mathbb{R}^m \to \mathbb{R}^m$ définie par $g(\mathbf{a})_i = \varphi(a_i)$, où $\varphi: \mathbb{R} \to \mathbb{R}$ est une activation scalaire différentiable.

1. Calculez la jacobienne $\mathbf{J}_g(\mathbf{a}) \in \mathbb{R}^{m \times m}$. Quelle est sa structure particulière? Quel coût en mémoire si $m = 10^4$?

2. Calculez la règle VJP: $\mathbf{u}^\top \mathbf{J}_g$. Montrez qu'elle se réduit à $\mathbf{u} \odot \varphi'(\mathbf{a})$. Quel est le coût en mémoire?

3. Spécialisez à $\varphi = \sigma$ (sigmoïde). En utilisant la formule $\sigma'(a) = \sigma(a)(1-\sigma(a))$ de l'exercice 2, écrivez la règle VJP uniquement en termes de $\mathbf{u}$ et $g(\mathbf{a}) = \sigma(\mathbf{a})$ (sans recalculer la passe avant).

4. Pour $\varphi = \text{ReLU}$, la dérivée $\text{ReLU}'(a) = \mathbb{1}(a > 0)$ n'est pas définie en $a = 0$. Comment les bibliothèques de DA gèrent-elles conventionnellement ce cas?
````

````{admonition} Solution Exercice 7
:class: dropdown

**1. Jacobienne de l'activation élémentaire:**

$\frac{\partial g_i}{\partial a_j} = \varphi'(a_i) \cdot \mathbb{1}(i = j)$, donc:

$$
\mathbf{J}_g(\mathbf{a}) = \operatorname{diag}(\varphi'(\mathbf{a})) \in \mathbb{R}^{m \times m}
$$

C'est une matrice *diagonale*. Malgré cela, si on la stockait naïvement, le coût serait $O(m^2)$. Pour $m = 10^4$: $10^8$ flottants $\approx$ 800 Mo, ce qui est prohibitif.

**2. Règle VJP:**

$$
\mathbf{u}^\top \mathbf{J}_g = \mathbf{u}^\top \operatorname{diag}(\varphi'(\mathbf{a})) = \mathbf{u} \odot \varphi'(\mathbf{a})
$$

Coût: $O(m)$ en temps et en mémoire. La matrice diagonale n'est jamais formée.

**3. VJP de la sigmoïde avec réutilisation:**

Puisque $\sigma'(a) = \sigma(a)(1 - \sigma(a))$ et $g(\mathbf{a}) = \sigma(\mathbf{a})$ est disponible depuis la passe avant:

$$
\mathbf{u}^\top \mathbf{J}_g = \mathbf{u} \odot g(\mathbf{a}) \odot (1 - g(\mathbf{a}))
$$

Pas besoin de recalculer $\sigma(\mathbf{a})$: la passe avant l'a déjà produit et stocké. C'est la raison pour laquelle les implémentations de rétropropagation *cachent* les activations intermédiaires.

**4. ReLU en $a = 0$:**

Par convention, la quasi-totalité des bibliothèques (JAX, PyTorch, TensorFlow) définissent $\text{ReLU}'(0) = 0$. Cette convention est cohérente avec le sous-différentiel de la fonction convexe $\max(0, a)$, et le point $a = 0$ forme un ensemble de mesure nulle qui n'affecte pas l'entraînement en pratique.
````

````{admonition} Exercice 8: JVP de la couche affine ★★
:class: hint dropdown

Soit $f(\mathbf{z}) = W\mathbf{z} + \mathbf{b}$ avec $W \in \mathbb{R}^{m \times n}$, et un vecteur tangent $\mathbf{v} \in \mathbb{R}^n$.

1. Calculez la règle JVP: $\mathbf{J}_f \cdot \mathbf{v}$. Quel est son coût?

2. Comparez: calculer le gradient complet $\nabla_\mathbf{z} \mathcal{L}$ via $n$ JVPs (un par composante $\mathbf{e}_i$) versus un seul VJP. Combien d'opérations arithmétiques chaque approche nécessite-t-elle?

3. Supposez $n = m = 1000$. Donnez les coûts numériques des deux approches pour calculer le gradient complet par rapport à tous les paramètres du réseau.

4. Dans quel cas le mode avant (JVP) est-il *préférable* au mode arrière (VJP)?
````

````{admonition} Solution Exercice 8
:class: dropdown

**1. Règle JVP:**

$$
\mathbf{J}_f \cdot \mathbf{v} = W\mathbf{v} \in \mathbb{R}^m
$$

Coût: $O(mn)$, un produit matrice-vecteur.

**2. Comparaison JVP vs VJP pour le gradient complet:**

- **$n$ JVPs**: calculer $\mathbf{J}_f \mathbf{e}_1, \ldots, \mathbf{J}_f \mathbf{e}_n$ reconstruit la jacobienne colonne par colonne. Coût total: $n \times O(mn) = O(mn^2)$.
- **1 VJP**: calculer $\mathbf{u}^\top W$ pour un seul $\mathbf{u}$. Coût: $O(mn)$.

Pour une perte scalaire, le VJP en mode arrière est donc $n$ fois plus efficace que le JVP en mode avant.

**3. Avec $n = m = 1000$:**

- Mode avant ($n$ JVPs): $1000 \times 10^6 = 10^9$ opérations
- Mode arrière (1 VJP): $10^6$ opérations

Le mode arrière est $10^3 = 1000$ fois moins coûteux, et c'est par couche. Sur un réseau de 100 couches, l'avantage s'accumule.

**4. Cas favorables au mode avant:**

Le JVP (mode avant) est préférable quand le nombre de sorties $m$ est grand mais le nombre d'entrées $n$ est petit. En pratique:
- Calcul de produits $\mathbf{J}_f \mathbf{v}$ pour des directions $\mathbf{v}$ spécifiques (e.g., directions de courbure en optimisation du second ordre)
- Dérivation par rapport à un petit nombre de paramètres scalaires (e.g., hyperparamètres)
- Sensibilités directionnelles en analyse d'incertitude
````

### Lire un graphe de calcul

Les exercices suivants portent sur les graphes de calcul (DAGs) et la règle de la chaîne. Pour chaque fonction, on décompose le calcul en opérations élémentaires et on représente les dépendances par un DAG.

````{admonition} Exercice 9: du graphe à l'expression (chaîne linéaire) ★
:class: hint dropdown

Considérez le graphe de calcul suivant:

```{mermaid}
graph LR
    x("x") --> exp_x("v₁ = exp(x)")
    exp_x --> add("v₂ = v₁ + 1")
    add --> log_v("v₃ = log(v₂)")
    log_v --> f("f")

    style x fill:#dae8fc,stroke:#6c8ebf
    style exp_x fill:#f5f5f5,stroke:#666
    style add fill:#f5f5f5,stroke:#666
    style log_v fill:#f5f5f5,stroke:#666
    style f fill:#f8cecc,stroke:#b85450
```

**(a)** Écrivez l'expression mathématique $f(x)$ que ce graphe calcule.

**(b)** Évaluez $f(0)$.

**(c)** Cette fonction porte un nom courant en apprentissage profond. Lequel?
````

````{admonition} Solution exercice 9
:class: dropdown

**(a)** En remplaçant les variables intermédiaires:

$$
v_1 = e^x, \quad v_2 = e^x + 1, \quad v_3 = \log(e^x + 1)
$$

Donc $f(x) = \log(e^x + 1)$.

**(b)** $f(0) = \log(e^0 + 1) = \log(2) \approx 0{,}693$.

**(c)** C'est la fonction **softplus**, une approximation lisse de ReLU.
````

````{admonition} Exercice 10: du graphe à l'expression (embranchement) ★
:class: hint dropdown

Considérez le graphe de calcul suivant:

```{mermaid}
graph LR
    x("x") --> cos_x("v₁ = cos(x)")
    x("x") --> sq("v₂ = x²")
    cos_x --> add("v₃ = v₁ + v₂")
    sq --> add
    add --> f("f")

    style x fill:#dae8fc,stroke:#6c8ebf
    style cos_x fill:#f5f5f5,stroke:#666
    style sq fill:#f5f5f5,stroke:#666
    style add fill:#f5f5f5,stroke:#666
    style f fill:#f8cecc,stroke:#b85450
```

**(a)** Écrivez l'expression mathématique $f(x)$.

**(b)** Combien d'arêtes sortantes a le noeud $x$? Qu'est-ce que cela signifie pour la règle de la chaîne?

**(c)** Évaluez $f(\pi)$.
````

````{admonition} Solution exercice 10
:class: dropdown

**(a)** $f(x) = \cos(x) + x^2$.

**(b)** Le noeud $x$ a **deux arêtes sortantes**: il alimente $v_1 = \cos(x)$ et $v_2 = x^2$. Cela signifie que $x$ contribue à $f$ par deux chemins distincts, et la règle de la chaîne devra sommer les deux contributions.

**(c)** $f(\pi) = \cos(\pi) + \pi^2 = -1 + \pi^2 \approx 8{,}870$.
````

````{admonition} Exercice 11: du graphe à l'expression (trois entrées) ★
:class: hint dropdown

Considérez le graphe de calcul suivant:

```{mermaid}
graph LR
    x("x") --> mul("v₁ = x · y")
    y("y") --> mul
    z("z") --> log_z("v₂ = log(z)")
    mul --> add("v₃ = v₁ + v₂")
    log_z --> add
    add --> f("f")

    style x fill:#dae8fc,stroke:#6c8ebf
    style y fill:#dae8fc,stroke:#6c8ebf
    style z fill:#dae8fc,stroke:#6c8ebf
    style mul fill:#f5f5f5,stroke:#666
    style log_z fill:#f5f5f5,stroke:#666
    style add fill:#f5f5f5,stroke:#666
    style f fill:#f8cecc,stroke:#b85450
```

**(a)** Écrivez l'expression mathématique $f(x, y, z)$.

**(b)** Évaluez $f(2, 3, 1)$.

**(c)** Quelles variables d'entrée ont un embranchement (fan-out) dans ce graphe?
````

````{admonition} Solution exercice 11
:class: dropdown

**(a)** $f(x, y, z) = x \cdot y + \log(z)$.

**(b)** $f(2, 3, 1) = 6 + \log(1) = 6 + 0 = 6$.

**(c)** Aucune variable d'entrée n'a d'embranchement: $x$ alimente uniquement $v_1$, $y$ alimente uniquement $v_1$, et $z$ alimente uniquement $v_2$. Le graphe a deux branches indépendantes qui se rejoignent à l'addition.
````

### Décomposer une fonction en graphe de calcul

````{admonition} Exercice 12: de l'expression au graphe (embranchement simple) ★
:class: hint dropdown

Soit $f(x) = x \cdot e^x$.

**(a)** Identifiez les variables intermédiaires en décomposant $f$ en opérations élémentaires.

**(b)** Dessinez le DAG correspondant. Combien d'arêtes sortantes a le noeud $x$?

**(c)** Évaluez $f(1)$.
````

````{admonition} Solution exercice 12
:class: dropdown

**(a)** Deux opérations élémentaires:

$$
v_1 = e^x, \quad v_2 = x \cdot v_1 = f(x)
$$

**(b)** Le DAG est:

```{mermaid}
graph LR
    x("x") --> exp_x("v₁ = exp(x)")
    x("x") --> mul("v₂ = x · v₁")
    exp_x --> mul
    mul --> f("f")

    style x fill:#dae8fc,stroke:#6c8ebf
    style exp_x fill:#f5f5f5,stroke:#666
    style mul fill:#f5f5f5,stroke:#666
    style f fill:#f8cecc,stroke:#b85450
```

Le noeud $x$ a **deux arêtes sortantes**: une vers $\exp$ et une vers la multiplication. C'est un embranchement.

**(c)** $f(1) = 1 \cdot e^1 = e \approx 2{,}718$.
````

````{admonition} Exercice 13: de l'expression au graphe (diamant) ★
:class: hint dropdown

Soit $f(x, y) = e^{x - y} + (x - y)^2$.

**(a)** Décomposez $f$ en opérations élémentaires et identifiez les variables intermédiaires.

**(b)** Dessinez le DAG. Quel noeud intermédiaire a un embranchement?

**(c)** Évaluez $f(1, 0)$.
````

````{admonition} Solution exercice 13
:class: dropdown

**(a)** Quatre opérations élémentaires:

$$
v_1 = x - y, \quad v_2 = e^{v_1}, \quad v_3 = v_1^2, \quad v_4 = v_2 + v_3 = f(x, y)
$$

**(b)** Le DAG est:

```{mermaid}
graph LR
    x("x") --> sub("v₁ = x − y")
    y("y") --> sub
    sub --> exp_v("v₂ = exp(v₁)")
    sub --> sq("v₃ = v₁²")
    exp_v --> add("v₄ = v₂ + v₃")
    sq --> add
    add --> f("f")

    style x fill:#dae8fc,stroke:#6c8ebf
    style y fill:#dae8fc,stroke:#6c8ebf
    style sub fill:#f5f5f5,stroke:#666
    style exp_v fill:#f5f5f5,stroke:#666
    style sq fill:#f5f5f5,stroke:#666
    style add fill:#f5f5f5,stroke:#666
    style f fill:#f8cecc,stroke:#b85450
```

Le noeud $v_1$ a un **embranchement**: il alimente à la fois $v_2 = e^{v_1}$ et $v_3 = v_1^2$. La passe arrière devra accumuler les contributions des deux branches pour obtenir $\bar{v}_1$.

**(c)** $f(1, 0) = e^1 + 1^2 = e + 1 \approx 3{,}718$.
````

````{admonition} Exercice 14: de l'expression au graphe (triple embranchement) ★★
:class: hint dropdown

Soit $f(x) = x \cdot \cos(x) + x^2$.

**(a)** Décomposez $f$ en opérations élémentaires.

**(b)** Dessinez le DAG. Combien d'arêtes sortantes a le noeud $x$?

**(c)** Évaluez $f(\pi/2)$.
````

````{admonition} Solution exercice 14
:class: dropdown

**(a)** Quatre opérations élémentaires:

$$
v_1 = \cos(x), \quad v_2 = x \cdot v_1, \quad v_3 = x^2, \quad v_4 = v_2 + v_3 = f(x)
$$

**(b)** Le DAG est:

```{mermaid}
graph LR
    x("x") --> cos_x("v₁ = cos(x)")
    x("x") --> mul("v₂ = x · v₁")
    cos_x --> mul
    x("x") --> sq("v₃ = x²")
    mul --> add("v₄ = v₂ + v₃")
    sq --> add
    add --> f("f")

    style x fill:#dae8fc,stroke:#6c8ebf
    style cos_x fill:#f5f5f5,stroke:#666
    style mul fill:#f5f5f5,stroke:#666
    style sq fill:#f5f5f5,stroke:#666
    style add fill:#f5f5f5,stroke:#666
    style f fill:#f8cecc,stroke:#b85450
```

Le noeud $x$ a **trois arêtes sortantes**: vers $\cos$, vers la multiplication, et vers le carré. La règle de la chaîne devra sommer trois contributions pour obtenir $\bar{x}$.

**(c)** $f(\pi/2) = (\pi/2) \cdot \cos(\pi/2) + (\pi/2)^2 = 0 + \pi^2/4 \approx 2{,}467$.
````

### Règle de la chaîne dans un DAG

Les exercices suivants partent de la règle de la chaîne sous forme de jacobiennes, puis montrent comment la réécrire comme une composition de fonctions VJP.

````{admonition} Exercice 15: des jacobiennes aux VJPs (chaîne linéaire) ★★
:class: hint dropdown

Considérez le graphe de calcul suivant, où chaque noeud est une fonction scalaire:

```{mermaid}
graph LR
    x("x") --> sin_x("v₁ = sin(x)")
    sin_x --> exp_v("v₂ = exp(v₁)")
    exp_v --> f("f")

    style x fill:#dae8fc,stroke:#6c8ebf
    style sin_x fill:#f5f5f5,stroke:#666
    style exp_v fill:#f5f5f5,stroke:#666
    style f fill:#f8cecc,stroke:#b85450
```

**(a)** Écrivez la jacobienne (ici, la dérivée) de chaque opération élémentaire: $J_{\sin}(x)$ et $J_{\exp}(v_1)$.

**(b)** En appliquant la règle de la chaîne, écrivez $\frac{df}{dx}$ comme un produit de jacobiennes.

**(c)** Pour calculer le gradient, on peut multiplier de droite à gauche. Définissons la fonction $\text{vjp}(g, a, \bar{u}) = \bar{u} \cdot J_g(a)$, c'est-à-dire le produit de l'adjoint entrant par la jacobienne locale de $g$ évaluée en $a$. En partant de $\bar{v}_2 = 1$, écrivez le calcul du gradient comme une composition d'appels à $\text{vjp}$:

$$
\bar{x} = \text{vjp}\!\big(\sin,\; x,\; \text{vjp}(\exp,\; \ldots,\; \ldots)\big)
$$

Complétez les arguments.

**(d)** Développez chaque appel $\text{vjp}$ en utilisant les jacobiennes de **(a)**. Vérifiez que vous retrouvez le résultat de **(b)**.

**(e)** Évaluez numériquement en $x = 0$.

**(f)** Vérifiez vos résultats en JAX. Complétez le code ci-dessous qui compare trois méthodes: le produit des jacobiennes, la composition manuelle des VJPs avec `jax.vjp`, et `jax.grad`.

```python
import jax
import jax.numpy as jnp

x = jnp.array(0.0)

# Méthode 1: produit des jacobiennes
v1 = jnp.sin(x)
J_sin = ???   # jacobienne de sin évaluée en x
J_exp = ???   # jacobienne de exp évaluée en v1
grad_jacobians = ???  # produit J_exp * J_sin

# Méthode 2: composition manuelle des VJPs
v1, vjp_sin = jax.vjp(jnp.sin, x)
v2, vjp_exp = jax.vjp(jnp.exp, v1)
(v1_bar,) = vjp_exp(jnp.array(1.0))  # ū = 1
(x_bar,)  = vjp_sin(v1_bar)

# Méthode 3: jax.grad
grad_auto = jax.grad(lambda x: jnp.exp(jnp.sin(x)))(x)

print(f'Jacobiennes:  {grad_jacobians}')
print(f'VJPs manuels: {x_bar}')
print(f'jax.grad:     {grad_auto}')
```
````

````{admonition} Solution exercice 15
:class: dropdown

La fonction est $f(x) = e^{\sin(x)}$.

**(a)** Jacobiennes locales (dérivées scalaires):

$$
J_{\sin}(x) = \cos(x), \qquad J_{\exp}(v_1) = e^{v_1}
$$

**(b)** Par la règle de la chaîne:

$$
\frac{df}{dx} = J_{\exp}(v_1) \cdot J_{\sin}(x) = e^{v_1} \cdot \cos(x)
$$

**(c)** La passe arrière compose les VJPs de droite à gauche, en partant de la sortie:

$$
\bar{x} = \text{vjp}\!\big(\sin,\; x, \;\text{vjp}(\exp,\; v_1,\; \underbrace{1}_{\bar{v}_2})\big)
$$

**(d)** Développons:

- $\text{vjp}(\exp, v_1, 1) = 1 \cdot J_{\exp}(v_1) = e^{v_1} = e^{\sin(x)}$
- $\text{vjp}(\sin, x, e^{\sin(x)}) = e^{\sin(x)} \cdot J_{\sin}(x) = e^{\sin(x)} \cdot \cos(x)$

On retrouve bien $\frac{df}{dx} = \cos(x) \, e^{\sin(x)}$.

**(e)** En $x = 0$:

Passe avant: $v_1 = \sin(0) = 0$, $v_2 = e^0 = 1$, $f = 1$.

Passe arrière: $\text{vjp}(\exp, 0, 1) = e^0 = 1$, puis $\text{vjp}(\sin, 0, 1) = 1 \cdot \cos(0) = 1$.

Donc $f'(0) = 1$.

**(f)** Code JAX complété:

```python
import jax
import jax.numpy as jnp

x = jnp.array(0.0)

# Méthode 1: produit des jacobiennes
v1 = jnp.sin(x)
J_sin = jnp.cos(x)       # cos(0) = 1
J_exp = jnp.exp(v1)      # exp(0) = 1
grad_jacobians = J_exp * J_sin  # 1.0

# Méthode 2: composition manuelle des VJPs
v1, vjp_sin = jax.vjp(jnp.sin, x)
v2, vjp_exp = jax.vjp(jnp.exp, v1)
(v1_bar,) = vjp_exp(jnp.array(1.0))
(x_bar,)  = vjp_sin(v1_bar)

# Méthode 3: jax.grad
grad_auto = jax.grad(lambda x: jnp.exp(jnp.sin(x)))(x)

print(f'Jacobiennes:  {grad_jacobians}')  # 1.0
print(f'VJPs manuels: {x_bar}')           # 1.0
print(f'jax.grad:     {grad_auto}')       # 1.0
```

Les trois méthodes donnent 1,0. La composition manuelle des VJPs avec `jax.vjp` fait exactement ce que `jax.grad` fait en interne: elle parcourt la bande à rebours en appelant chaque VJP locale.
````

````{admonition} Exercice 16: jacobiennes et accumulation (embranchement) ★★
:class: hint dropdown

Reprenons le graphe de $f(x, y) = e^{x - y} + (x - y)^2$ de l'exercice 7:

```{mermaid}
graph LR
    x("x") --> sub("v₁ = x − y")
    y("y") --> sub
    sub --> exp_v("v₂ = exp(v₁)")
    sub --> sq("v₃ = v₁²")
    exp_v --> add("v₄ = v₂ + v₃")
    sq --> add
    add --> f("f")

    style x fill:#dae8fc,stroke:#6c8ebf
    style y fill:#dae8fc,stroke:#6c8ebf
    style sub fill:#f5f5f5,stroke:#666
    style exp_v fill:#f5f5f5,stroke:#666
    style sq fill:#f5f5f5,stroke:#666
    style add fill:#f5f5f5,stroke:#666
    style f fill:#f8cecc,stroke:#b85450
```

**(a)** Écrivez la jacobienne de chaque opération élémentaire: $J_{\text{sub}}(x, y)$, $J_{\exp}(v_1)$, $J_{\text{sq}}(v_1)$, $J_{\text{add}}(v_2, v_3)$.

**(b)** Le noeud $v_1$ a deux successeurs ($v_2$ et $v_3$). En utilisant la règle de la chaîne multivariée, écrivez $\bar{v}_1$ comme une somme de deux termes (un par chemin).

**(c)** Écrivez la passe arrière comme une suite d'appels à $\text{vjp}$. Le noeud $v_1$ reçoit deux contributions: montrez comment elles s'accumulent.

$$
\bar{v}_2, \bar{v}_3 = \text{vjp}(\text{add}, v_2, v_3,\; \bar{v}_4) \quad \text{puis} \quad \bar{v}_1 = \text{vjp}(\exp, v_1,\; \bar{v}_2) + \text{vjp}(\text{sq}, v_1,\; \bar{v}_3)
$$

Continuez jusqu'à $\bar{x}$ et $\bar{y}$.

**(d)** Évaluez numériquement en $(x, y) = (1, 0)$. Vérifiez par dérivation directe.

**(e)** Vérifiez en JAX. Complétez le code ci-dessous. La partie importante est l'accumulation sur $v_1$: le noeud reçoit les contributions de deux VJPs distincts qu'il faut additionner.

```python
import jax
import jax.numpy as jnp

x, y = jnp.array(1.0), jnp.array(0.0)

# Passe avant: enregistrer les VJPs
v1, vjp_sub = jax.vjp(lambda x, y: x - y, x, y)
v2, vjp_exp = jax.vjp(jnp.exp, v1)
v3, vjp_sq  = jax.vjp(lambda v: v**2, v1)
v4, vjp_add = jax.vjp(lambda a, b: a + b, v2, v3)

# Passe arrière: composer les VJPs
v4_bar = jnp.array(1.0)
(v2_bar, v3_bar) = vjp_add(v4_bar)
(v1_bar_exp,) = vjp_exp(v2_bar)
(v1_bar_sq,)  = vjp_sq(v3_bar)
v1_bar = ???                          # accumulation
(x_bar, y_bar) = vjp_sub(v1_bar)

# Comparaison avec jax.grad
f = lambda x, y: jnp.exp(x - y) + (x - y)**2
print(f'VJPs manuels: dx={x_bar}, dy={y_bar}')
print(f'jax.grad:     dx={jax.grad(f, 0)(x, y)}, '
      f'dy={jax.grad(f, 1)(x, y)}')
```
````

````{admonition} Solution exercice 16
:class: dropdown

**(a)** Jacobiennes locales:

$$
J_{\text{sub}}(x, y) = \begin{bmatrix} 1 & -1 \end{bmatrix}, \quad J_{\exp}(v_1) = e^{v_1}, \quad J_{\text{sq}}(v_1) = 2v_1, \quad J_{\text{add}}(v_2, v_3) = \begin{bmatrix} 1 & 1 \end{bmatrix}
$$

**(b)** Le noeud $v_1$ contribue à $f$ par deux chemins (via $\exp$ et via le carré). La règle de la chaîne multivariée donne:

$$
\bar{v}_1 = \bar{v}_2 \, D_{v_1} \phi_2 + \bar{v}_3 \, D_{v_1} \phi_3 = \bar{v}_2 \cdot e^{v_1} + \bar{v}_3 \cdot 2v_1
$$

**(c)** Passe arrière complète:

- Initialiser $\bar{v}_4 = 1$.
- $\bar{v}_2, \bar{v}_3 = \text{vjp}(\text{add}, v_2, v_3,\; 1)$. Comme $J_{\text{add}} = [1, 1]$, on obtient $\bar{v}_2 = 1$, $\bar{v}_3 = 1$.
- Accumulation sur $v_1$:

$$
\bar{v}_1 = \text{vjp}(\exp, v_1,\; \bar{v}_2) + \text{vjp}(\text{sq}, v_1,\; \bar{v}_3) = \bar{v}_2 \cdot e^{v_1} + \bar{v}_3 \cdot 2v_1 = e^{v_1} + 2v_1
$$

- $\bar{x}, \bar{y} = \text{vjp}(\text{sub}, x, y,\; \bar{v}_1)$. Comme $J_{\text{sub}} = [1, -1]$:

$$
\bar{x} = \bar{v}_1 \cdot 1 = e^{v_1} + 2v_1, \qquad \bar{y} = \bar{v}_1 \cdot (-1) = -(e^{v_1} + 2v_1)
$$

**(d)** En $(x, y) = (1, 0)$:

Passe avant: $v_1 = 1$, $v_2 = e$, $v_3 = 1$, $v_4 = e + 1 \approx 3{,}718$.

Passe arrière: $\bar{v}_4 = 1$, $\bar{v}_2 = 1$, $\bar{v}_3 = 1$, $\bar{v}_1 = e + 2 \approx 4{,}718$.

$$
\bar{x} = e + 2 \approx 4{,}718, \quad \bar{y} = -(e + 2) \approx -4{,}718
$$

Vérification directe: $\frac{\partial f}{\partial x} = e^{x-y} + 2(x - y) = e + 2$. Correct. On note que $\partial f / \partial x = -\partial f / \partial y$, ce qui est attendu puisque $f$ dépend de $x$ et $y$ uniquement à travers $x - y$.

**(e)** Code JAX complété:

```python
import jax
import jax.numpy as jnp

x, y = jnp.array(1.0), jnp.array(0.0)

# Passe avant: enregistrer les VJPs
v1, vjp_sub = jax.vjp(lambda x, y: x - y, x, y)
v2, vjp_exp = jax.vjp(jnp.exp, v1)
v3, vjp_sq  = jax.vjp(lambda v: v**2, v1)
v4, vjp_add = jax.vjp(lambda a, b: a + b, v2, v3)

# Passe arrière: composer les VJPs
v4_bar = jnp.array(1.0)
(v2_bar, v3_bar) = vjp_add(v4_bar)
(v1_bar_exp,) = vjp_exp(v2_bar)
(v1_bar_sq,)  = vjp_sq(v3_bar)
v1_bar = v1_bar_exp + v1_bar_sq       # accumulation!
(x_bar, y_bar) = vjp_sub(v1_bar)

# Comparaison avec jax.grad
f = lambda x, y: jnp.exp(x - y) + (x - y)**2
print(f'VJPs manuels: dx={x_bar}, dy={y_bar}')
# dx=4.718..., dy=-4.718...
print(f'jax.grad:     dx={jax.grad(f, 0)(x, y)}, '
      f'dy={jax.grad(f, 1)(x, y)}')
# dx=4.718..., dy=-4.718...
```

La ligne `v1_bar = v1_bar_exp + v1_bar_sq` est l'accumulation: le noeud $v_1$ reçoit un adjoint de chaque branche. C'est exactement ce que fait le système de DA en interne quand une variable a un fan-out.
````

````{admonition} Exercice 17: ReLU et gradient nul ★★★
:class: hint dropdown

Considérez le graphe de calcul suivant, qui correspond à $f(x, y) = \text{relu}(x + y) \cdot (x - y)$:

```{mermaid}
graph LR
    x("x") --> add("v₁ = x + y")
    y("y") --> add
    add --> relu("v₂ = relu(v₁)")
    x("x") --> sub("v₃ = x − y")
    y("y") --> sub
    relu --> mul("v₄ = v₂ · v₃")
    sub --> mul
    mul --> f("f")

    style x fill:#dae8fc,stroke:#6c8ebf
    style y fill:#dae8fc,stroke:#6c8ebf
    style add fill:#f5f5f5,stroke:#666
    style relu fill:#f5f5f5,stroke:#666
    style sub fill:#f5f5f5,stroke:#666
    style mul fill:#f5f5f5,stroke:#666
    style f fill:#f8cecc,stroke:#b85450
```

On rappelle que $\text{relu}(t) = \max(0, t)$ et que $J_{\text{relu}}(t) = \mathbb{1}[t > 0]$.

**(a)** Écrivez la jacobienne de chaque opération élémentaire. Quelles variables d'entrée ont un embranchement?

**(b)** Écrivez la passe arrière comme une suite d'appels à $\text{vjp}$, en montrant comment $\bar{x}$ et $\bar{y}$ accumulent chacun deux contributions (une par branche).

**(c)** Évaluez en $(x, y) = (3, 1)$ (ReLU actif). Vérifiez en simplifiant $f$ quand $x + y > 0$.

**(d)** Évaluez en $(x, y) = (-3, 1)$ (ReLU inactif). Que se passe-t-il pour les gradients?

**(e)** Vérifiez en JAX pour les deux cas. Complétez le code ci-dessous. Notez que $x$ et $y$ alimentent chacun deux opérations: il faut accumuler les contributions des deux branches.

```python
import jax
import jax.numpy as jnp

def backward_manual(x, y):
    # Passe avant
    v1, vjp_add = jax.vjp(lambda x, y: x + y, x, y)
    v2, vjp_relu = jax.vjp(jax.nn.relu, v1)
    v3, vjp_sub = jax.vjp(lambda x, y: x - y, x, y)
    v4, vjp_mul = jax.vjp(lambda a, b: a * b, v2, v3)

    # Passe arrière
    (v2_bar, v3_bar) = vjp_mul(jnp.array(1.0))
    (v1_bar,) = vjp_relu(v2_bar)
    (x_bar_add, y_bar_add) = vjp_add(v1_bar)
    (x_bar_sub, y_bar_sub) = vjp_sub(v3_bar)
    x_bar = ???  # accumulation des deux branches
    y_bar = ???
    return x_bar, y_bar

# Cas 1: ReLU actif
print('ReLU actif (3, 1):', backward_manual(
    jnp.array(3.0), jnp.array(1.0)))
# Cas 2: ReLU inactif
print('ReLU inactif (-3, 1):', backward_manual(
    jnp.array(-3.0), jnp.array(1.0)))

# Comparaison avec jax.grad
f = lambda x, y: jax.nn.relu(x + y) * (x - y)
print('jax.grad (3, 1):',
      jax.grad(f, 0)(jnp.array(3.0), jnp.array(1.0)),
      jax.grad(f, 1)(jnp.array(3.0), jnp.array(1.0)))
print('jax.grad (-3, 1):',
      jax.grad(f, 0)(jnp.array(-3.0), jnp.array(1.0)),
      jax.grad(f, 1)(jnp.array(-3.0), jnp.array(1.0)))
```
````

````{admonition} Solution exercice 17
:class: dropdown

**(a)** Jacobiennes locales:

$$
J_{\text{add}}(x, y) = \begin{bmatrix} 1 & 1 \end{bmatrix}, \quad J_{\text{relu}}(v_1) = \mathbb{1}[v_1 > 0], \quad J_{\text{sub}}(x, y) = \begin{bmatrix} 1 & -1 \end{bmatrix}
$$

$$
J_{\text{mul}}(v_2, v_3) = \begin{bmatrix} v_3 & v_2 \end{bmatrix}
$$

Les deux variables $x$ et $y$ ont un embranchement: chacune alimente $v_1$ (addition) et $v_3$ (soustraction).

**(b)** Passe arrière:

- Initialiser $\bar{v}_4 = 1$.
- $\bar{v}_2, \bar{v}_3 = \text{vjp}(\text{mul}, v_2, v_3,\; 1) = (v_3, \; v_2)$
- $\bar{v}_1 = \text{vjp}(\text{relu}, v_1,\; \bar{v}_2) = \mathbb{1}[v_1 > 0] \cdot \bar{v}_2$

Accumulation sur $x$ et $y$ (deux contributions chacun):

$$
\bar{x} = \text{vjp}(\text{add}, x, y,\; \bar{v}_1)_x + \text{vjp}(\text{sub}, x, y,\; \bar{v}_3)_x = \bar{v}_1 + \bar{v}_3
$$

$$
\bar{y} = \text{vjp}(\text{add}, x, y,\; \bar{v}_1)_y + \text{vjp}(\text{sub}, x, y,\; \bar{v}_3)_y = \bar{v}_1 + (-\bar{v}_3)
$$

**(c)** En $(x, y) = (3, 1)$ (ReLU actif):

Passe avant: $v_1 = 4$, $v_2 = 4$, $v_3 = 2$, $v_4 = 8$.

Passe arrière: $\bar{v}_4 = 1$. $\text{vjp}(\text{mul}) \to \bar{v}_2 = 2$, $\bar{v}_3 = 4$. $\text{vjp}(\text{relu}) \to \bar{v}_1 = \mathbb{1}[4 > 0] \cdot 2 = 2$.

$$
\bar{x} = \bar{v}_1 + \bar{v}_3 = 2 + 4 = 6, \quad \bar{y} = \bar{v}_1 - \bar{v}_3 = 2 - 4 = -2
$$

Vérification: quand $x + y > 0$, $f(x, y) = (x+y)(x-y) = x^2 - y^2$, donc $\partial f/\partial x = 2x = 6$ et $\partial f/\partial y = -2y = -2$. Correct.

**(d)** En $(x, y) = (-3, 1)$ (ReLU inactif):

Passe avant: $v_1 = -2$, $v_2 = 0$, $v_3 = -4$, $v_4 = 0$.

Passe arrière: $\bar{v}_4 = 1$. $\text{vjp}(\text{mul}) \to \bar{v}_2 = -4$, $\bar{v}_3 = 0$. $\text{vjp}(\text{relu}) \to \bar{v}_1 = \mathbb{1}[-2 > 0] \cdot (-4) = 0$.

$$
\bar{x} = 0 + 0 = 0, \quad \bar{y} = 0 - 0 = 0
$$

Les deux gradients sont nuls. Le ReLU inactif bloque le gradient à travers la branche gauche ($\bar{v}_1 = 0$), et la multiplication par $v_2 = 0$ bloque le gradient à travers la branche droite ($\bar{v}_3 = v_2 \cdot \bar{v}_4 = 0$). Quand la sortie du ReLU est nulle, aucun gradient ne peut remonter, quelle que soit la branche.

**(e)** Code JAX complété:

```python
import jax
import jax.numpy as jnp

def backward_manual(x, y):
    v1, vjp_add = jax.vjp(lambda x, y: x + y, x, y)
    v2, vjp_relu = jax.vjp(jax.nn.relu, v1)
    v3, vjp_sub = jax.vjp(lambda x, y: x - y, x, y)
    v4, vjp_mul = jax.vjp(lambda a, b: a * b, v2, v3)

    (v2_bar, v3_bar) = vjp_mul(jnp.array(1.0))
    (v1_bar,) = vjp_relu(v2_bar)
    (x_bar_add, y_bar_add) = vjp_add(v1_bar)
    (x_bar_sub, y_bar_sub) = vjp_sub(v3_bar)
    x_bar = x_bar_add + x_bar_sub  # accumulation
    y_bar = y_bar_add + y_bar_sub
    return x_bar, y_bar

# ReLU actif:  (6.0, -2.0)
print(backward_manual(jnp.array(3.0), jnp.array(1.0)))
# ReLU inactif: (0.0, 0.0)
print(backward_manual(jnp.array(-3.0), jnp.array(1.0)))
```

Les résultats confirment les calculs manuels. Le cas $(−3, 1)$ illustre le phénomène du neurone mort: les deux gradients sont exactement zéro, ce que `jax.grad` confirme aussi.
````

