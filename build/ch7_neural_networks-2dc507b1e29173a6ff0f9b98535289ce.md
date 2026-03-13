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
- Expliquer pourquoi le problème XOR motive les architectures multicouches
- Expliquer comment les réseaux de neurones apprennent leurs propres caractéristiques, contrairement aux modèles à expansion fixe
- Définir l'architecture d'un perceptron multicouche (MLP) et le rôle des fonctions d'activation
- Dériver l'algorithme de rétropropagation à partir de la règle de la chaîne
- Distinguer les modes avant (JVP) et arrière (VJP) de la différentiation automatique
- Expliquer le paradigme de la programmation différentiable
- Décrire les algorithmes d'optimisation courants (SGD, momentum, Adam) et leurs différences
- Appliquer les techniques de régularisation (décroissance des poids, dropout) pour réduire le surapprentissage
```

Aux chapitres 2 et 3, nous avons construit des modèles linéaires pour la régression et la classification. Au chapitre 4, nous avons vu comment enrichir ces modèles en transformant les entrées par une fonction $\boldsymbol{\phi}$ fixée à l'avance. Ce chapitre franchit une étape supplémentaire: au lieu de choisir $\boldsymbol{\phi}$ manuellement, nous allons l'apprendre à partir des données. Cette idée conduit aux réseaux de neurones.

Dans ce chapitre, nous commençons par un exemple concret (le problème XOR) qui illustre en quatre points pourquoi les modèles linéaires atteignent une limite fondamentale. Nous rappelons ensuite le cadre probabiliste qui unifie régression et classification, puis nous présentons l'anatomie d'un réseau (couches, activations, architecture). La section sur la rétropropagation est plus technique: elle dérive l'algorithme de calcul des gradients pas à pas. Vous pouvez survoler les détails en première lecture et retenir le mécanisme général. Nous couvrons ensuite les algorithmes d'optimisation utilisés en pratique (SGD, momentum, Adam), la différentiation automatique, les techniques de stabilisation de l'entraînement, et la régularisation. La section d'implémentation clôt le chapitre avec un MLP complet en NumPy.

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

C'est précisément ce que Minsky et Papert ont formalisé en 1969 {cite}`minsky1969perceptrons`, en montrant que certaines fonctions booléennes (notamment XOR) sont impossibles à apprendre pour un perceptron simple. Leur analyse a eu un impact considérable sur le domaine, contribuant à un ralentissement de la recherche sur les réseaux de neurones pendant plusieurs années. Ce que leur livre contenait également, sans que cela soit toujours mis en avant, c'est la solution: empiler deux couches de perceptrons suffit à résoudre XOR. C'est ce que nous allons explorer maintenant.

## Le problème XOR: pourquoi les modèles linéaires ne suffisent pas

Avant de définir formellement les réseaux de neurones, partons d'un exemple concret qui illustre la limite des modèles linéaires et motive directement l'idée d'empiler plusieurs couches.

### Une fonction simple que la régression logistique ne peut pas apprendre

La fonction XOR (ou exclusif) prend deux entrées binaires et retourne 1 si exactement l'une d'elles vaut 1:

| $x_1$ | $x_2$ | $y = x_1 \oplus x_2$ |
|:-----:|:-----:|:--------------------:|
| 0     | 0     | 0                    |
| 0     | 1     | 1                    |
| 1     | 0     | 1                    |
| 1     | 1     | 0                    |

Un modèle linéaire pour la classification binaire prédit $\hat{y} = \sigma(\theta_1 x_1 + \theta_2 x_2 + b)$. La frontière de décision $\theta_1 x_1 + \theta_2 x_2 + b = 0$ est une droite dans le plan $(x_1, x_2)$.

Le problème est géométrique: les deux points de la classe 0, $(0,0)$ et $(1,1)$, sont disposés en diagonale, et les deux points de la classe 1, $(0,1)$ et $(1,0)$, sont sur l'autre diagonale. Aucune droite ne peut séparer ces deux groupes. La figure ci-dessous le rend évident.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

# Points XOR
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0, 1, 1, 0])

markers = ['s', 'o']
colors  = ['#4878CF', '#D65F5F']
labels  = ['Classe 0', 'Classe 1']

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# --- Panneau gauche: espace original ---
ax = axes[0]
for cls in [0, 1]:
    mask = y == cls
    ax.scatter(X[mask, 0], X[mask, 1],
               marker=markers[cls], color=colors[cls],
               s=180, zorder=5, label=labels[cls], edgecolors='k', linewidths=1.2)
# Tentative de droite séparatrice (impossible)
xline = np.linspace(-0.3, 1.3, 100)
ax.plot(xline, -xline + 1, 'k--', lw=1.5, alpha=0.5, label='Meilleure droite')
ax.plot(xline, -xline + 0.5, 'k:', lw=1.5, alpha=0.3)
ax.set_xlim(-0.4, 1.4)
ax.set_ylim(-0.4, 1.4)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_title('Espace original: XOR non séparable')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.text(0.5, 1.1, 'Aucune droite ne sépare\nles deux classes',
        ha='center', fontsize=8.5, color='#555555',
        bbox=dict(boxstyle='round,pad=0.3', fc='#fffbe6', ec='#ccbb00', alpha=0.9))

# --- Panneau droit: espace transformé ---
# h1 = ReLU(x1 + x2 - 0.5),  h2 = ReLU(x1 + x2 - 1.5)
W1 = np.array([[1., 1.], [1., 1.]])
b1 = np.array([-0.5, -1.5])
H = np.maximum(0, X @ W1.T + b1)   # shape (4, 2)

ax = axes[1]
for cls in [0, 1]:
    mask = y == cls
    ax.scatter(H[mask, 0], H[mask, 1],
               marker=markers[cls], color=colors[cls],
               s=180, zorder=5, label=labels[cls], edgecolors='k', linewidths=1.2)
# Frontière séparatrice dans l'espace transformé: w2=[1,-3], b=-0.4 → h1 - 3*h2 = 0.4
hline = np.linspace(-0.1, 1.8, 100)
ax.plot(hline, (hline - 0.4) / 3, 'g-', lw=2, label='Séparateur linéaire')
ax.set_xlim(-0.15, 1.8)
ax.set_ylim(-0.2, 0.8)
ax.set_xlabel(r'$h_1 = \mathrm{ReLU}(x_1+x_2-0{,}5)$')
ax.set_ylabel(r'$h_2 = \mathrm{ReLU}(x_1+x_2-1{,}5)$')
ax.set_title('Espace transformé: XOR devient séparable')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.annotate('$(0,0)$ et $(0{,}5, 0)$\ncollapsent: $(1,0)=(0,1)$',
            xy=(0.5, 0.0), xytext=(0.8, 0.35),
            fontsize=7.5, color='#555555',
            arrowprops=dict(arrowstyle='->', color='gray', lw=1),
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.8))

plt.tight_layout()
```

À gauche, aucune droite ne peut séparer les classes: le problème est géométriquement impossible pour un modèle linéaire. À droite, après une transformation par deux neurones ReLU, les quatre points se retrouvent dans un espace où une droite les sépare.

### La solution: transformer l'espace d'entrée

Nous avons vu à la section précédente qu'un perceptron (ou n'importe quel modèle linéaire) ne peut pas apprendre XOR. L'idée est alors de transformer les entrées dans un espace où les classes deviennent séparables linéairement, puis d'appliquer un classifieur linéaire dans cet espace transformé.

Pour XOR, voici une telle transformation construite à la main. Définissons deux neurones ReLU:

$$
h_1 = \text{ReLU}(x_1 + x_2 - 0{,}5), \qquad h_2 = \text{ReLU}(x_1 + x_2 - 1{,}5)
$$

Calculons les valeurs pour chaque point:

| $(x_1, x_2)$ | $x_1+x_2$ | $h_1$ | $h_2$ | $y$ |
|:------------:|:---------:|:-----:|:-----:|:---:|
| $(0,0)$      | $0$       | $0$   | $0$   | $0$ |
| $(0,1)$      | $1$       | $0{,}5$ | $0$ | $1$ |
| $(1,0)$      | $1$       | $0{,}5$ | $0$ | $1$ |
| $(1,1)$      | $2$       | $1{,}5$ | $0{,}5$ | $0$ |

Dans l'espace $(h_1, h_2)$, les points de classe 1 se trouvent en $(0{,}5, 0)$, et les points de classe 0 en $(0, 0)$ et $(1{,}5, 0{,}5)$. Une droite $h_1 - 3h_2 = 0{,}4$ les sépare.

Le classifieur complet est donc:

$$
\hat{y} = \sigma\!\left(\underbrace{h_1 - 3h_2 - 0{,}4}_{\text{couche de sortie}}\right), \quad \text{où} \quad
\begin{pmatrix} h_1 \\ h_2 \end{pmatrix} = \text{ReLU}\!\left(\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\mathbf{x} + \begin{pmatrix} -0{,}5 \\ -1{,}5 \end{pmatrix}\right)
$$

C'est exactement la structure d'un réseau à une couche cachée: une transformation non linéaire $\mathbf{x} \mapsto \mathbf{h}$ suivie d'un classifieur linéaire $\mathbf{h} \mapsto \hat{y}$.

### Ce que l'exemple révèle

Trois points méritent d'être notés:

**La couche cachée crée des caractéristiques.** Les activations $h_1$ et $h_2$ ne sont pas des entrées fournies à l'avance, elles sont calculées par le réseau. $h_1$ détecte si au moins une entrée est active; $h_2$ détecte si les deux sont actives. Ensemble, elles encodent l'information nécessaire pour XOR.

**La non-linéarité est indispensable.** Si les fonctions ReLU étaient remplacées par des transformations linéaires, la composition des deux couches se réduirait à une seule transformation linéaire (exercice 1), et le problème XOR resterait insoluble.

**Les paramètres ont été choisis à la main ici.** En pratique, pour des données complexes, nous ne savons pas quelle transformation construire. C'est là qu'intervient l'apprentissage: l'algorithme d'optimisation ajuste $W_1$, $\mathbf{b}_1$, $\mathbf{w}_2$, $b_2$ pour que le réseau découvre lui-même la bonne représentation.

La figure ci-dessous montre la frontière de décision apprise par un MLP entraîné sur XOR. Le réseau n'a pas besoin qu'on lui indique la transformation à faire: il la découvre à partir des quatre exemples.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

np.random.seed(7)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

# --- Entraînement d'un petit MLP sur XOR ---
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([[0],[1],[1],[0]], dtype=float)

H = 8   # neurones cachés
W1 = np.random.randn(2, H) * 0.5
b1 = np.zeros(H)
W2 = np.random.randn(H, 1) * 0.5
b2 = np.zeros(1)

lr = 0.5
for _ in range(8000):
    # Passe avant
    a1 = X_xor @ W1 + b1
    z1 = relu(a1)
    a2 = z1 @ W2 + b2
    p  = sigmoid(a2)
    # Perte entropie croisée
    # Passe arrière
    dp  = p - y_xor                      # (4,1)
    dW2 = z1.T @ dp / 4
    db2 = dp.mean(axis=0)
    dz1 = dp @ W2.T
    da1 = dz1 * (a1 > 0).astype(float)
    dW1 = X_xor.T @ da1 / 4
    db1 = da1.mean(axis=0)
    W1 -= lr * dW1;  b1 -= lr * db1
    W2 -= lr * dW2;  b2 -= lr * db2

# --- Grille de décision ---
xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 300),
                     np.linspace(-0.3, 1.3, 300))
Xg = np.column_stack([xx.ravel(), yy.ravel()])
zz = sigmoid(relu(Xg @ W1 + b1) @ W2 + b2).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(5, 4.5))
cf = ax.contourf(xx, yy, zz, levels=50, cmap='RdBu_r', alpha=0.7, vmin=0, vmax=1)
ax.contour(xx, yy, zz, levels=[0.5], colors='k', linewidths=2)
plt.colorbar(cf, ax=ax, label=r'$p(y=1 \mid \mathbf{x})$')

markers = ['s', 'o'];  colors_pt = ['#4878CF', '#D65F5F']
for cls in [0, 1]:
    mask = y_xor.ravel() == cls
    ax.scatter(X_xor[mask, 0], X_xor[mask, 1],
               marker=markers[cls], color=colors_pt[cls],
               s=200, zorder=5, edgecolors='k', linewidths=1.5,
               label=f'Classe {cls}')

ax.set_xlim(-0.3, 1.3);  ax.set_ylim(-0.3, 1.3)
ax.set_xlabel(r'$x_1$');  ax.set_ylabel(r'$x_2$')
ax.set_title('Frontière de décision apprise par un MLP sur XOR')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
```

La frontière de décision est clairement non linéaire: le réseau a appris à isoler les deux régions où $y=1$ (en haut à gauche et en bas à droite). Un modèle linéaire produirait une seule droite, incapable de réaliser cette séparation.

**Point de contrôle.** Si vous comprenez pourquoi aucune droite ne peut séparer les classes dans le panneau gauche de la première figure, et comment la couche cachée résout ce problème dans le panneau droit, vous avez saisi la motivation centrale des réseaux multicouches. La suite du chapitre formalisera et généralisera cette idée.

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

### Règles VJP: une bibliothèque d'opérateurs adjoints

La section précédente a montré que le mode arrière évite de former les jacobiennes complètes — mais une question reste ouverte: comment calcule-t-on $\mathbf{u}^\top \mathbf{J}_f(\mathbf{x})$ *sans* construire $\mathbf{J}_f$?

La réponse repose sur une distinction fondamentale. La jacobienne $\mathbf{J}_f(\mathbf{x}) \in \mathbb{R}^{m \times n}$ est une *représentation matricielle* d'un objet plus abstrait: la différentielle $df_\mathbf{x}$, qui est un opérateur linéaire $df_\mathbf{x}: \mathbb{R}^n \to \mathbb{R}^m$. Ce qui importe dans le mode arrière n'est pas $df_\mathbf{x}$ lui-même, mais son **opérateur adjoint** $df_\mathbf{x}^*: \mathbb{R}^m \to \mathbb{R}^n$, qui mappe les vecteurs du co-domaine vers le domaine (c'est-à-dire qui propage le signal en sens inverse). En coordonnées, $df_\mathbf{x}^*(\mathbf{u}) = \mathbf{J}_f(\mathbf{x})^\top \mathbf{u}$.

Le point essentiel: pour définir un opérateur linéaire, il n'est pas nécessaire d'en donner la matrice — il suffit de spécifier son *action* sur des vecteurs. Une bibliothèque d'AD (JAX, PyTorch) maintient pour chaque opération primitive une **règle VJP**: une fonction qui calcule directement $\mathbf{J}_f(\mathbf{x})^\top \mathbf{u}$ à partir de $\mathbf{u}$, $\mathbf{x}$, et éventuellement $f(\mathbf{x})$, en n'utilisant que des opérations arithmétiques simples.

Lorsque deux opérations se composent — $h = g \circ f$ — la règle VJP de $h$ est le produit des règles VJP de $g$ et $f$:

$$
\mathbf{J}_h^\top \mathbf{u} = \mathbf{J}_f^\top \underbrace{(\mathbf{J}_g^\top \mathbf{u})}_{\text{appel récursif}}
$$

La passe arrière n'est rien d'autre que l'exécution récursive de ces règles, de la sortie vers l'entrée. Le système est entièrement **sans matrice jacobienne explicite**: aucune matrice $\mathbf{J}_f$ n'est jamais construite ni stockée.

Le tableau ci-dessous liste les règles VJP pour les opérations clés d'un MLP. À chaque fois, la règle VJP évite de former la jacobienne correspondante:

| Opération | $f(\mathbf{x})$ | Jacobienne (non formée) | Règle VJP: $\mathbf{J}_f^\top \mathbf{u}$ |
|-----------|-----------------|--------------------------|-------------------------------------------|
| Couche affine (entrée $\mathbf{z}$) | $W\mathbf{z} + \mathbf{b}$ | $W \in \mathbb{R}^{m \times n}$ | $W^\top \mathbf{u}$ |
| Couche affine (poids $W$) | $W\mathbf{z} + \mathbf{b}$ | $\mathbf{z}^\top \otimes I_m$ | $\mathbf{u}\mathbf{z}^\top$ (produit externe) |
| Couche affine (biais $\mathbf{b}$) | $W\mathbf{z} + \mathbf{b}$ | $I_m \in \mathbb{R}^{m \times m}$ | $\mathbf{u}$ |
| Activation élémentaire | $\varphi(\mathbf{a})$ | $\operatorname{diag}(\varphi'(\mathbf{a})) \in \mathbb{R}^{m \times m}$ | $\mathbf{u} \odot \varphi'(\mathbf{a})$ |
| Somme $s = \sum_i x_i$ | scalaire | $\mathbf{1}^\top \in \mathbb{R}^{1 \times n}$ | $u \cdot \mathbf{1}$ (diffusion) |

L'exemple de l'activation élémentaire illustre parfaitement le bénéfice: la jacobienne serait une matrice $m \times m$ coûtant $O(m^2)$ en mémoire, alors que la règle VJP — $\mathbf{u} \odot \varphi'(\mathbf{a})$ — est un produit élément par élément en $O(m)$.

En JAX, `jax.custom_vjp` permet d'enregistrer exactement ce type de règle pour une opération personnalisée. Écrire une règle VJP correcte pour un nouvel opérateur est une compétence essentielle en apprentissage profond avancé — les exercices 9 à 11 vous entraînent à cette dérivation.

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

La passe arrière calcule les adjoints en remontant ce graphe nœud par nœud, en multipliant les gradients locaux de chaque opération.

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

## Optimisation

La rétropropagation produit les gradients $\nabla_{\boldsymbol{\theta}} \mathcal{L}$. Il reste à décider comment utiliser ces gradients pour mettre à jour les paramètres. Cette section présente les algorithmes d'optimisation les plus utilisés en pratique, de la descente de gradient stochastique jusqu'à Adam.

### Descente de gradient stochastique par mini-lots

Nous avons vu la descente de gradient au chapitre 3. Pour un réseau de neurones entraîné sur $N$ exemples, calculer le gradient exact sur tout le jeu de données à chaque itération est coûteux. La **descente de gradient stochastique par mini-lots** (*minibatch SGD*) estime le gradient sur un sous-ensemble aléatoire de $B$ exemples:

$$
\hat{\nabla}_{\boldsymbol{\theta}} \mathcal{L} = \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla_{\boldsymbol{\theta}} \ell(\mathbf{x}_i, y_i; \boldsymbol{\theta})
$$

où $\mathcal{B}$ est un mini-lot de taille $B$ tiré aléatoirement. Cette estimation est non biaisée: $\mathbb{E}[\hat{\nabla}_{\boldsymbol{\theta}} \mathcal{L}] = \nabla_{\boldsymbol{\theta}} \mathcal{L}$. La mise à jour des paramètres est:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \hat{\nabla}_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)
$$

où $\eta > 0$ est le **taux d'apprentissage**. Une **époque** correspond à un passage complet sur le jeu de données, soit $N/B$ mises à jour. Les données sont brassées aléatoirement à chaque époque pour éviter les biais d'ordre.

```{prf:algorithm} Descente de gradient stochastique par mini-lots
:label: minibatch-sgd

**Entrée**: Jeu de données $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$, taille de lot $B$, taux $\eta$, nombre d'époques $T$

**Sortie**: Paramètres $\boldsymbol{\theta}$

1. Initialiser $\boldsymbol{\theta}$ (Glorot ou He selon l'activation)
2. Pour $t = 1, \ldots, T$:
   - Brasser $\mathcal{D}$ aléatoirement
   - Pour chaque mini-lot $\mathcal{B} \subset \mathcal{D}$ de taille $B$:
     - Calculer $\hat{\mathbf{g}} = \frac{1}{B}\sum_{i \in \mathcal{B}} \nabla_{\boldsymbol{\theta}} \ell_i(\boldsymbol{\theta})$ (rétropropagation)
     - $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \hat{\mathbf{g}}$
3. Retourner $\boldsymbol{\theta}$
```

En pratique, $B$ est souvent entre 32 et 512. Un $B$ petit introduit plus de bruit dans l'estimation du gradient (haute variance), ce qui peut aider à échapper aux minima locaux mais ralentit la convergence. Un $B$ grand donne une estimation plus précise mais réduit l'effet régularisateur du bruit stochastique.

### Momentum

Un problème de SGD est l'**oscillation**: sur une surface de perte avec des directions de courbures très différentes (une vallée étroite et allongée, par exemple), le gradient pointe perpendiculairement aux parois et le pas fait zigzaguer d'une paroi à l'autre, progressant lentement dans la direction de la vallée.

L'idée du **momentum** {cite}`polyak1964some` est d'accumuler une vitesse dans les directions stables et d'amortir les oscillations. On maintient un vecteur de vitesse $\mathbf{m}_t$ qui est une moyenne pondérée exponentiellement des gradients passés:

$$
\mathbf{m}_{t+1} = \beta \mathbf{m}_t + \hat{\mathbf{g}}_t, \qquad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{m}_{t+1}
$$

Le paramètre $\beta \in [0, 1)$ (typiquement $0{,}9$) contrôle la "mémoire": avec $\beta = 0{,}9$, la mise à jour actuelle contribue à 10% de la vitesse, et les gradients des 10 derniers pas ont encore une influence notable. On peut vérifier que $\mathbf{m}_t = \sum_{k=0}^{t} \beta^k \hat{\mathbf{g}}_{t-k}$, ce qui montre que $\mathbf{m}_t$ est bien une moyenne pondérée des gradients passés, avec des poids décroissant géométriquement.

La variante **Nesterov** {cite}`nesterov1983method` calcule le gradient à la position anticipée $\boldsymbol{\theta}_t + \beta \mathbf{m}_t$ plutôt qu'à la position courante:

$$
\hat{\mathbf{g}}_t = \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t + \beta \mathbf{m}_t), \qquad \mathbf{m}_{t+1} = \beta \mathbf{m}_t + \hat{\mathbf{g}}_t, \qquad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{m}_{t+1}
$$

Cette "anticipation" améliore la convergence en théorie et souvent en pratique.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

def grad_f(t): return np.array([2*t[0], 20*t[1]])

theta0 = np.array([-0.9, 0.85])
eta    = 0.08
beta   = 0.5
n_steps = 40

# SGD
traj_sgd = [theta0.copy()]
t = theta0.copy()
for _ in range(n_steps):
    t = t - eta * grad_f(t)
    traj_sgd.append(t.copy())
traj_sgd = np.array(traj_sgd)

# SGD + Momentum
traj_mom = [theta0.copy()]
t = theta0.copy(); m = np.zeros(2)
for _ in range(n_steps):
    m = beta * m + grad_f(t)
    t = t - eta * m
    traj_mom.append(t.copy())
traj_mom = np.array(traj_mom)

t1 = np.linspace(-1.1, 1.1, 300)
t2 = np.linspace(-1.0, 1.0, 300)
T1, T2 = np.meshgrid(t1, t2)
Z = T1**2 + 10*T2**2

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
plt.suptitle(r'SGD vs momentum sur $f(\theta) = \theta_1^2 + 10\theta_2^2$', fontsize=11)

trajs  = [traj_sgd, traj_mom]
titles = ['SGD (zigzags)', r'SGD + Momentum ($\beta=0{,}5$)']
colors = ['#1f77b4', '#d62728']

for ax, traj, title, color in zip(axes, trajs, titles, colors):
    ax.contourf(T1, T2, Z, levels=20, cmap='Greys', alpha=0.5)
    ax.contour( T1, T2, Z, levels=20, colors='gray', linewidths=0.4, alpha=0.6)
    # Trajectoire complète avec dégradé d'opacité (début → fin)
    n = len(traj) - 1
    for i in range(n):
        alpha = 0.25 + 0.75 * (i / n)
        ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], '-', color=color, lw=2, alpha=alpha)
    ax.plot(*traj[0],  'ko', ms=8,  zorder=5, label='Départ')
    ax.plot(*traj[-1], 'o',  ms=7,  color=color, zorder=6, label='Arrivée')
    ax.plot(0, 0, 'r*', ms=12, zorder=5, label='Minimum')
    ax.set_xlabel(r'$\theta_1$')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.0, 1.0)
    ax.grid(True, alpha=0.2)

axes[0].set_ylabel(r'$\theta_2$')
plt.tight_layout()
```

Sur cette surface allongée, SGD zigzague entre les parois de la vallée et progresse lentement vers le minimum. Le momentum ($\beta=0{,}5$) accumule de la vitesse dans la direction $\theta_1$ (la direction de la vallée) et amortit les oscillations dans la direction $\theta_2$ (perpendiculaire aux parois).

### Taux d'apprentissage adaptatifs: RMSProp

SGD et momentum utilisent le même taux d'apprentissage $\eta$ pour tous les paramètres. Cela peut être sous-optimal quand les gradients ont des magnitudes très différentes selon les dimensions: un $\eta$ adapté aux grandes dimensions sera trop grand pour les petites, et vice versa.

**RMSProp** {cite}`tieleman2012rmsprop` maintient une estimation de la variance du gradient par dimension $j$, et divise le gradient par la racine de cette variance:

$$
s_{t+1,j} = \beta s_{t,j} + (1-\beta) g_{t,j}^2, \qquad \theta_{t+1,j} = \theta_{t,j} - \frac{\eta}{\sqrt{s_{t+1,j} + \epsilon}} g_{t,j}
$$

où $g_{t,j} = [\hat{\mathbf{g}}_t]_j$ est la $j$-ème composante du gradient, $\beta \approx 0{,}9$ et $\epsilon \approx 10^{-8}$ évite la division par zéro. La quantité $s_{t,j}$ est une moyenne pondérée exponentiellement des carrés des gradients passés: elle estime $\mathbb{E}[g_j^2]$. Diviser par $\sqrt{s_{t,j}}$ normalise effectivement le gradient par son écart-type empirique, ce qui donne un taux d'apprentissage effectif de magnitude similaire pour toutes les dimensions.

### Adam

**Adam** (*Adaptive Moment Estimation*) {cite}`kingma2014adam` combine le momentum (premier moment du gradient) et RMSProp (deuxième moment du gradient):

$$
\mathbf{m}_{t+1} = \beta_1 \mathbf{m}_t + (1-\beta_1) \hat{\mathbf{g}}_t \qquad \text{(premier moment)}
$$

$$
\mathbf{s}_{t+1} = \beta_2 \mathbf{s}_t + (1-\beta_2) \hat{\mathbf{g}}_t^2 \qquad \text{(deuxième moment, élément par élément)}
$$

Au début de l'entraînement, $\mathbf{m}_t$ et $\mathbf{s}_t$ sont initialisés à zéro. Pendant les premières itérations, ils sous-estiment les moments réels (biais vers zéro). Adam corrige ce biais:

$$
\hat{\mathbf{m}}_{t+1} = \frac{\mathbf{m}_{t+1}}{1 - \beta_1^{t+1}}, \qquad \hat{\mathbf{s}}_{t+1} = \frac{\mathbf{s}_{t+1}}{1 - \beta_2^{t+1}}
$$

La mise à jour finale est:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \frac{\hat{\mathbf{m}}_{t+1}}{\sqrt{\hat{\mathbf{s}}_{t+1}} + \epsilon}
$$

Les valeurs par défaut sont $\beta_1 = 0{,}9$, $\beta_2 = 0{,}999$, $\epsilon = 10^{-8}$, $\eta = 10^{-3}$.

```{prf:algorithm} Adam
:label: ch7-adam

**Entrée**: Taux $\eta$, paramètres $\beta_1, \beta_2, \epsilon$, nombre d'itérations $T$

**Initialiser**: $\boldsymbol{\theta}_0$, $\mathbf{m}_0 = \mathbf{0}$, $\mathbf{s}_0 = \mathbf{0}$

1. Pour $t = 0, 1, \ldots, T-1$:
   - Calculer $\hat{\mathbf{g}}_t = \nabla_{\boldsymbol{\theta}} \hat{\mathcal{L}}(\boldsymbol{\theta}_t)$ sur un mini-lot
   - $\mathbf{m}_{t+1} \leftarrow \beta_1 \mathbf{m}_t + (1-\beta_1)\hat{\mathbf{g}}_t$
   - $\mathbf{s}_{t+1} \leftarrow \beta_2 \mathbf{s}_t + (1-\beta_2)\hat{\mathbf{g}}_t^2$
   - $\hat{\mathbf{m}} \leftarrow \mathbf{m}_{t+1} / (1 - \beta_1^{t+1})$
   - $\hat{\mathbf{s}} \leftarrow \mathbf{s}_{t+1} / (1 - \beta_2^{t+1})$
   - $\boldsymbol{\theta}_{t+1} \leftarrow \boldsymbol{\theta}_t - \eta\, \hat{\mathbf{m}} / (\sqrt{\hat{\mathbf{s}}} + \epsilon)$
2. Retourner $\boldsymbol{\theta}_T$
```

Adam est actuellement l'optimiseur le plus utilisé pour les réseaux de neurones. Il converge généralement plus vite que SGD ou momentum grâce à la normalisation adaptative, et il est moins sensible au choix du taux d'apprentissage initial.

La figure ci-dessous compare les courbes de convergence des trois algorithmes sur un MLP entraîné à classer deux spirales enchevêtrées.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

np.random.seed(0)

# --- Génération de données: spirales ---
def make_spirals(n=200, noise=0.15):
    t = np.linspace(0, 4*np.pi, n)
    x1 = np.column_stack([t*np.cos(t), t*np.sin(t)]) / (4*np.pi) + noise*np.random.randn(n,2)
    x2 = np.column_stack([-t*np.cos(t), -t*np.sin(t)]) / (4*np.pi) + noise*np.random.randn(n,2)
    X = np.vstack([x1, x2])
    y = np.array([0]*n + [1]*n)
    return X, y

X_sp, y_sp = make_spirals(150, 0.12)
N = len(y_sp)

def relu(x):     return np.maximum(0, x)
def sigmoid(x):  return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

def ce_loss(p, y):
    p = np.clip(p, 1e-7, 1-1e-7)
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))

def mlp_train(optimizer='sgd', eta=0.01, n_epochs=200, B=32, beta1=0.9, beta2=0.999):
    H = 32
    W1 = np.random.randn(2, H) * np.sqrt(2/2)
    b1 = np.zeros(H)
    W2 = np.random.randn(H, 1) * np.sqrt(2/H)
    b2 = np.zeros(1)
    params = [W1, b1, W2, b2]

    # Optimizer state
    m = [np.zeros_like(p) for p in params]
    s = [np.zeros_like(p) for p in params]
    t_step = 0
    eps = 1e-8

    losses = []
    idx = np.arange(N)
    for epoch in range(n_epochs):
        np.random.shuffle(idx)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N, B):
            batch = idx[start:start+B]
            Xb = X_sp[batch];  yb = y_sp[batch].reshape(-1,1).astype(float)
            # Forward
            a1 = Xb @ W1 + b1
            z1 = relu(a1)
            a2 = z1 @ W2 + b2
            p  = sigmoid(a2)
            epoch_loss += ce_loss(p, yb)
            n_batches += 1
            # Backward
            dp  = (p - yb) / len(batch)
            dW2 = z1.T @ dp
            db2 = dp.sum(axis=0)
            dz1 = dp @ W2.T
            da1 = dz1 * (a1 > 0)
            dW1 = Xb.T @ da1
            db1 = da1.sum(axis=0)
            grads = [dW1, db1, dW2, db2]
            t_step += 1
            for i, (p_i, g) in enumerate(zip(params, grads)):
                if optimizer == 'sgd':
                    p_i -= eta * g
                elif optimizer == 'momentum':
                    m[i] = beta1 * m[i] + g
                    p_i -= eta * m[i]
                elif optimizer == 'adam':
                    m[i] = beta1*m[i] + (1-beta1)*g
                    s[i] = beta2*s[i] + (1-beta2)*g**2
                    mhat = m[i] / (1 - beta1**t_step)
                    shat = s[i] / (1 - beta2**t_step)
                    p_i -= eta * mhat / (np.sqrt(shat) + eps)
        losses.append(epoch_loss / n_batches)
    return losses

np.random.seed(0)
losses_sgd  = mlp_train('sgd',      eta=0.08,  n_epochs=200)
np.random.seed(0)
losses_mom  = mlp_train('momentum', eta=0.02,  n_epochs=200)
np.random.seed(0)
losses_adam = mlp_train('adam',     eta=0.005, n_epochs=200)

fig, ax = plt.subplots(figsize=(8, 4))
epochs = np.arange(1, 201)
ax.plot(epochs, losses_sgd,  'C0-',  lw=2, label='SGD')
ax.plot(epochs, losses_mom,  'C1--', lw=2, label='SGD + Momentum')
ax.plot(epochs, losses_adam, 'C3-',  lw=2.5, label='Adam')
ax.set_xlabel('Époque')
ax.set_ylabel('Perte (entropie croisée)')
ax.set_title('Convergence des optimiseurs sur le problème des spirales')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 200)
plt.tight_layout()
```

Adam converge plus rapidement et de façon plus régulière grâce à la normalisation adaptative par dimension. SGD seul oscille davantage et converge plus lentement sur ce problème. Le momentum offre un compromis intermédiaire.

### Quel optimiseur choisir?

En pratique, **Adam** est un bon point de départ pour la plupart des architectures. SGD avec momentum peut surpasser Adam sur certains problèmes de vision (comme l'entraînement de réseaux convolutifs sur CIFAR-10 ou ImageNet) si l'on prend le temps d'ajuster le taux d'apprentissage et un calendrier de décroissance. Pour la recherche, il est courant d'essayer les deux et de comparer.

Le **taux d'apprentissage** est l'hyperparamètre le plus important pour tous ces algorithmes. Un $\eta$ trop grand provoque des oscillations ou une divergence; un $\eta$ trop petit converge lentement. Les bibliothèques modernes offrent des **calendriers de taux d'apprentissage** (*learning rate schedules*): décroissance linéaire, décroissance cosinus, ou réchauffement (*warmup*) suivi d'une décroissance.

## Différentiation automatique

*Cette section ne dépend pas des détails de la rétropropagation: vous pouvez la lire même si vous avez survolé la section précédente.*

La rétropropagation est un cas particulier d'un principe plus général. Cette section élargit la perspective: nous montrons comment le calcul de gradients se généralise à des programmes arbitraires, au-delà des réseaux en chaîne. Nous verrons pourquoi on écrit `import jax.numpy as jnp` plutôt que `import numpy as np`, et comment une bibliothèque comme JAX peut différentier n'importe quelle fonction Python sans jamais voir son code source.

Pour illustrer les mécanismes, nous utilisons tout au long de cette section la fonction

$$
f(x, y) = \sin(x) \cdot (x + y)
$$

Elle est suffisamment simple pour être traitée à la main, mais suffisamment riche pour révéler les points délicats: $x$ intervient dans deux branches distinctes du calcul.

### Différentiation numérique, symbolique et automatique

Pour calculer la dérivée d'un programme, trois approches existent:

La **différentiation numérique** approxime la dérivée par différences finies:

$$
\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + \epsilon \mathbf{e}_i) - f(\mathbf{x} - \epsilon \mathbf{e}_i)}{2\epsilon}
$$

Cette méthode est simple à implémenter mais souffre de deux problèmes: elle requiert $O(n)$ évaluations de $f$ pour un gradient en dimension $n$, et elle est sujette aux erreurs d'arrondi (le choix de $\epsilon$ est délicat). Elle reste utile pour *vérifier* des implémentations de gradient.

La **différentiation symbolique** applique les règles de dérivation formellement, comme on le ferait à la main. Elle produit des expressions exactes, mais ces expressions peuvent croître exponentiellement en taille pour des programmes complexes. De plus, elle requiert que le programme soit représenté sous forme symbolique, ce qui exclut les structures de contrôle (boucles, conditions).

La **différentiation automatique** (AD) est une troisième voie. Elle évalue la dérivée exacte d'un programme numérique en le décomposant en opérations élémentaires et en appliquant la règle de la chaîne à chaque étape. Contrairement à la différentiation numérique, elle est exacte (aux erreurs de virgule flottante près). Contrairement à la différentiation symbolique, elle opère sur des valeurs numériques, pas sur des expressions, et gère naturellement les boucles et les conditions.

La rétropropagation n'est rien d'autre que la différentiation automatique en mode arrière, appliquée au programme qui calcule la perte d'un réseau de neurones.

### Graphes de calcul

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

Remarquez que $x$ a **deux arêtes sortantes**: il alimente $v_1 = \sin(x)$ et $v_2 = x + y$. Cela signifie que $x$ contribue à $f$ par deux chemins distincts. La règle de la chaîne doit sommer les deux contributions:

$$
\frac{\partial f}{\partial x} = \frac{\partial f}{\partial v_1}\frac{\partial v_1}{\partial x} + \frac{\partial f}{\partial v_2}\frac{\partial v_2}{\partial x} = \cos(x) \cdot v_2 + v_1 \cdot 1
$$

De manière générale, si un noeud $u$ a des successeurs $k_1, \ldots, k_m$ dans le graphe (c'est-à-dire des noeuds qui dépendent directement de $u$), alors:

$$
\frac{\partial f}{\partial u} = \sum_{k \in \text{successeurs}(u)} \frac{\partial f}{\partial k} \cdot \frac{\partial k}{\partial u}
$$

C'est la règle de la chaîne multivariée. La rétropropagation dans un DAG est simplement l'application systématique de cette règle, en partant de la sortie et en remontant vers les entrées.

### Tri topologique

La règle de la chaîne ci-dessus pose un problème d'ordre: pour calculer $\partial f / \partial u$, il faut d'abord connaître $\partial f / \partial k$ pour tous les successeurs $k$ de $u$. On ne peut pas calculer le gradient d'un noeud avant d'avoir accumulé les contributions de tous ses successeurs.

Un **tri topologique** fournit un ordre de traitement qui respecte ces contraintes: dans cet ordre, chaque noeud apparaît après tous ceux qui en dépendent. Pour la passe arrière, on parcourt cet ordre à rebours.

Il n'existe pas un seul tri topologique valide, mais plusieurs. La seule contrainte est que chaque noeud apparaisse avant ses prédécesseurs dans la liste inversée, c'est-à-dire que la passe arrière ne traite un noeud qu'après avoir traité tous ses successeurs. Les deux diagrammes ci-dessous montrent deux ordres distincts, tous les deux corrects.

**Ordre A** — le DFS visite $v_1$ avant $v_2$ depuis $v_3$:

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

**Ordre B** — le DFS visite $v_2$ avant $v_1$ depuis $v_3$:

```{mermaid}
graph LR
    x("x ⑥") --> sin_x("v₁ = sin(x) ③")
    x("x ⑥") --> add("v₂ = x + y ④")
    y("y ⑤") --> add("v₂ = x + y ④")
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

Dans les deux ordres, $x$ est traité en dernier (⑥): c'est la conséquence directe du fait que $x$ contribue à *deux* branches ($v_1$ et $v_2$). La passe arrière ne peut considérer $x$ comme "terminé" qu'après avoir reçu les contributions des deux. En revanche, $y$ — qui ne contribue qu'à $v_2$ — peut être traité dès que $v_2$ a envoyé son gradient, ce qui laisse de la liberté sur sa position (④ ou ⑤ selon l'ordre de visite).

Les seules contraintes strictes sont:
- $f$ (①) et $v_3$ (②) sont toujours les deux premiers,
- $v_1$ et $v_2$ doivent précéder $x$,
- $v_2$ doit précéder $y$.

L'ordre relatif de $v_1$ et $v_2$, et la position de $y$ entre eux, sont laissés libres.

L'algorithme classique de tri topologique repose sur un parcours en profondeur (DFS). Il est décrit dans l'encadré ci-dessous.

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

### La liste de Wengert

Une fois l'ordre établi, il faut stocker les informations nécessaires à la passe arrière. C'est le rôle de la **liste de Wengert** {cite}`wengert1964simple`, aussi appelée *tape* (bande). Proposée en 1964, cette structure est à la base de tous les systèmes d'AD modernes.

Le principe est simple: pendant la passe avant, on enregistre, pour chaque opération, les entrées, la sortie, et la fonction locale qui calcule la contribution au gradient (la VJP locale). La passe arrière rejoue la liste à rebours, en appelant chaque VJP locale dans l'ordre inverse.

Contrairement au DAG, la bande est une **liste ordonnée** (un tableau linéaire), ce qui la rend simple à parcourir dans les deux sens. Les flèches pleines montrent l'enregistrement (passe avant); les flèches pointillées montrent le rejeu (passe arrière).

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

Le tableau ci-dessous détaille le contenu de chaque entrée et les formules de VJP associées.

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

%config InlineBackend.figure_format = 'retina'

fig, ax = plt.subplots(figsize=(11, 3.2))
ax.set_xlim(0, 11)
ax.set_ylim(0, 3.5)
ax.axis('off')

columns = ['Étape', 'Opération', 'Entrées', 'Sortie', 'VJP locale']
col_x   = [0.3, 1.6, 3.5, 5.5, 6.8]
col_w   = [1.1, 1.7, 1.8, 1.1, 4.0]
row_h   = 0.60
rows = [
    ['1', 'sin',  'x',       'v₁', r'$\bar{x}$ += cos(x) · $\bar{v}_1$'],
    ['2', 'add',  'x, y',    'v₂', r'$\bar{x}$ += $\bar{v}_2$, $\bar{y}$ += $\bar{v}_2$'],
    ['3', 'mul',  'v₁, v₂',  'v₃', r'$\bar{v}_1$ += v₂ · $\bar{v}_3$,  $\bar{v}_2$ += v₁ · $\bar{v}_3$'],
]

header_color = '#4472c4'
row_colors   = ['#eef3fb', '#dde6f5']

# En-têtes
for j, (col, cx, cw) in enumerate(zip(columns, col_x, col_w)):
    rect = FancyBboxPatch((cx, 2.8), cw - 0.05, row_h - 0.05,
                          boxstyle='round,pad=0.04', linewidth=0,
                          facecolor=header_color)
    ax.add_patch(rect)
    ax.text(cx + (cw - 0.05)/2, 2.8 + (row_h - 0.05)/2, col,
            ha='center', va='center', fontsize=9.5, color='white', fontweight='bold')

# Lignes
for i, row in enumerate(rows):
    bg = row_colors[i % 2]
    y0 = 2.8 - (i + 1) * row_h
    for j, (cell, cx, cw) in enumerate(zip(row, col_x, col_w)):
        rect = FancyBboxPatch((cx, y0), cw - 0.05, row_h - 0.05,
                              boxstyle='round,pad=0.04', linewidth=0.5,
                              edgecolor='#aaaaaa', facecolor=bg)
        ax.add_patch(rect)
        ax.text(cx + (cw - 0.05)/2, y0 + (row_h - 0.05)/2, cell,
                ha='center', va='center', fontsize=9,
                usetex=False)

ax.text(5.5, 3.35, r'Tape pour $f(x,y) = \sin(x)\cdot(x+y)$',
        ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5.5, 0.18,
        r'La barre $\bar{v}$ désigne $\partial f/\partial v$ (adjoint). '
        r'La passe arrière parcourt les étapes 3 → 2 → 1.',
        ha='center', va='center', fontsize=8.5, color='#444444')

plt.tight_layout()
```

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

La passe arrière initialise `v3.grad = 1.0`, puis parcourt les arêtes à rebours: chaque fermeture accumule les adjoints dans les noeuds parents. Quand les deux fermetures de $v_1$ et $v_2$ ont été appelées, `x.grad` contient la somme des deux contributions.

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

%config InlineBackend.figure_format = 'retina'

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.patch.set_facecolor('#fafafa')

# ---- Panneau gauche: exécution du code ----
ax = axes[0]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
ax.set_facecolor('#fafafa')
ax.text(5, 9.5, 'Exécution Python', ha='center', fontsize=11, fontweight='bold')

code_lines = [
    (1, r'x = Tracer(0.5)',       '#dae8fc'),
    (2, r'y = Tracer(1.2)',       '#dae8fc'),
    (3, r'v1 = sin(x)',           '#fff2cc'),
    (4, r'v2 = x + y',            '#fff2cc'),
    (5, r'v3 = v1 * v2',          '#fff2cc'),
]
for i, (num, line, color) in enumerate(code_lines):
    y_pos = 8.2 - i * 1.3
    rect = plt.Rectangle((0.5, y_pos - 0.4), 9, 0.85,
                          facecolor=color, edgecolor='#888', linewidth=0.8)
    ax.add_patch(rect)
    ax.text(1.0, y_pos + 0.02, f'{num}.', fontsize=9, color='#555', va='center')
    ax.text(2.0, y_pos + 0.02, line, fontsize=9.5, va='center', family='monospace')

ax.annotate('', xy=(5, 1.5), xytext=(5, 2.5),
            arrowprops=dict(arrowstyle='->', color='#444', lw=1.5))
ax.text(5, 1.2, 'Construit la bande\nautomatiquement', ha='center', fontsize=9, color='#444')

# ---- Panneau droit: état de la bande ----
ax2 = axes[1]
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis('off')
ax2.set_facecolor('#fafafa')
ax2.text(5, 9.5, 'Bande (construite ligne par ligne)', ha='center', fontsize=11, fontweight='bold')

tape_entries = [
    (r'après ligne 3: (sin, x → v₁)',  r'fermeture: $\bar{x}$ += cos(x)·$\bar{v}_1$'),
    (r'après ligne 4: (add, x,y → v₂)',r'fermeture: $\bar{x}$ += $\bar{v}_2$, $\bar{y}$ += $\bar{v}_2$'),
    (r'après ligne 5: (mul, v₁,v₂ → v₃)',r'fermeture: $\bar{v}_1$ += v₂·$\bar{v}_3$, $\bar{v}_2$ += v₁·$\bar{v}_3$'),
]
colors2 = ['#fff2cc', '#ffe6cc', '#e2efda']
for i, (title, detail) in enumerate(tape_entries):
    y_pos = 7.8 - i * 2.3
    rect = plt.Rectangle((0.3, y_pos - 0.9), 9.4, 1.75,
                          facecolor=colors2[i], edgecolor='#888', linewidth=0.8)
    ax2.add_patch(rect)
    ax2.text(0.7, y_pos + 0.5, title,   fontsize=9,   va='center', fontweight='bold')
    ax2.text(0.7, y_pos - 0.2, detail,  fontsize=8.5, va='center', color='#333')

ax2.text(5, 0.8,
    'Passe arrière: parcourir la bande\nde bas en haut, appeler chaque fermeture',
    ha='center', fontsize=9, color='#444',
    bbox=dict(facecolor='#f8cecc', edgecolor='#b85450', boxstyle='round,pad=0.3'))

plt.tight_layout()
```

L'exécution Python se déroule normalement, ligne par ligne. Python ne sait pas qu'il trace un graphe: il appelle simplement les méthodes `__add__`, `__mul__`, `sin` sur les objets traceurs, et ces méthodes enregistrent discrètement les opérations. Quand l'exécution est terminée, la bande est complète, et la passe arrière peut s'exécuter.

Ce mécanisme explique pourquoi la différentiation automatique gère naturellement les boucles et les conditions: Python les exécute normalement, et les traceurs enregistrent les opérations qui sont effectivement effectuées lors de cette exécution particulière.

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

`jax.numpy` est essentiellement un espace de noms qui réimplémente toutes les fonctions NumPy de manière à intercepter les traceurs. Pour les tableaux NumPy ordinaires (sans traceur), `jnp` et `np` produisent les mêmes résultats numériques. La différence n'apparaît que pendant la trace.

### Implémentation minimale

*Cette sous-section est optionnelle pour IFT3395. Elle montre comment implémenter un moteur de différentiation automatique en mode arrière en une soixantaine de lignes de Python pur.*

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
                adjoints[idx] += ct          # accumulation (fan-out)

        return tuple(adjoints[v.id] for v in traced)
    return grad_fn
```

La séparation en trois parties n'est pas un choix esthétique: c'est la structure réelle des bibliothèques d'AD. Dans JAX, les règles VJP sont enregistrées via `jax.custom_vjp`, la bande est construite par le traceur interne, et `jax.grad` orchestre la passe arrière. Notre implémentation reproduit cette architecture en miniature.

Vérifions sur $f(x, y) = \sin(x) \cdot (x + y)$:

```{code-cell} ipython3
import math

# --- Valeurs de test ---
x0, y0 = 0.5, 1.2

# --- Avec notre moteur d'AD ---
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

Les trois méthodes sont en accord. Remarquez que `grad` est une fonction d'ordre supérieur qui retourne une nouvelle fonction, exactement comme `jax.grad`. L'accumulation des adjoints (ligne `adjoints[idx] += ct`) gère automatiquement le cas où une variable contribue à plusieurs branches du calcul — c'est la somme des deux chemins pour $x$.

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

La figure ci-dessous simule la norme du gradient en fonction de la profondeur pour des réseaux sigmoïdes et ReLU. La décroissance exponentielle avec la sigmoïde illustre pourquoi les réseaux profonds étaient difficiles à entraîner avant l'adoption de ReLU.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

np.random.seed(42)
n_layers = 20
n_trials = 50
d = 50  # dimension des couches

def simulate_gradient_norm(activation_deriv, n_layers, n_trials, d):
    """Simule la norme du gradient à la première couche."""
    norms = []
    for _ in range(n_trials):
        g = np.ones(d) / np.sqrt(d)  # gradient normalisé en sortie
        for _ in range(n_layers):
            W = np.random.randn(d, d) / np.sqrt(d)
            # Pré-activations aléatoires pour simuler un réseau typique
            a = np.random.randn(d)
            dphi = activation_deriv(a)
            # Jacobienne locale: diag(phi'(a)) @ W
            g = (dphi * g) @ W  # VJP simplifié
        norms.append(np.linalg.norm(g))
    return norms

d_sigmoid = lambda a: np.exp(-a) / (1 + np.exp(-a))**2
d_relu    = lambda a: (a > 0).astype(float)

layers = np.arange(1, n_layers + 1)
norms_sigmoid = np.array([
    np.mean(simulate_gradient_norm(d_sigmoid, l, n_trials, d))
    for l in layers
])
norms_relu = np.array([
    np.mean(simulate_gradient_norm(d_relu, l, n_trials, d))
    for l in layers
])

fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(layers, norms_sigmoid, 'C0-o', markersize=4, linewidth=2, label='Sigmoïde')
ax.semilogy(layers, norms_relu,    'C2-s', markersize=4, linewidth=2, label='ReLU')
ax.axhspan(0, 1e-8, alpha=0.1, color='C0', label='Zone de disparition')
ax.set_xlabel('Profondeur (nombre de couches)')
ax.set_ylabel('Norme du gradient $\\|\\nabla_{W_1}\\mathcal{L}\\|$')
ax.set_title('Gradient qui disparaît: sigmoïde vs ReLU')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(1, n_layers)
plt.tight_layout()
```

Avec la sigmoïde, la norme du gradient décroît de façon essentiellement exponentielle: à 20 couches de profondeur, le signal est réduit de plusieurs ordres de grandeur. ReLU maintient un gradient plus stable grâce à sa dérivée égale à 1 pour les activations positives.

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

La figure ci-dessous montre les distributions des activations à différentes profondeurs pour trois stratégies d'initialisation: variance trop petite (saturation), variance trop grande (explosion), et initialisation de Glorot (stable).

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

np.random.seed(1)
d = 100
n_samples = 500
n_layers = 5
check_layers = [1, 2, 3, 4, 5]

def propagate(x, n_layers, var, activation=np.tanh):
    activations = []
    h = x.copy()
    for _ in range(n_layers):
        W = np.random.randn(d, d) * np.sqrt(var)
        h = activation(h @ W)
        activations.append(h.ravel())
    return activations

x0 = np.random.randn(n_samples, d)

configs = [
    ('Trop petite\n($\\sigma^2 = 0{,}01$)', 0.01,  'C3'),
    ('Glorot\n($\\sigma^2 = 2/(n+m)$)', 2/d,    'C2'),
    ('Trop grande\n($\\sigma^2 = 4$)',    4.0,    'C0'),
]

fig, axes = plt.subplots(3, 5, figsize=(12, 6), sharey='row')

for row, (label, var, color) in enumerate(configs):
    acts = propagate(x0, n_layers, var)
    for col, (ax, h) in enumerate(zip(axes[row], acts)):
        ax.hist(h, bins=40, color=color, alpha=0.75, density=True, edgecolor='none')
        ax.set_xlim(-1.1, 1.1)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel(label, fontsize=8)
        if row == 0:
            ax.set_title(f'Couche {col+1}', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

plt.suptitle("Distribution des activations (tanh) selon l'initialisation", fontsize=11)
plt.tight_layout()
```

Avec une variance trop petite, les activations tanh se concentrent autour de zéro dès les premières couches (zone de gradient quasi nul). Avec une variance trop grande, elles saturent à $\pm 1$. L'initialisation de Glorot maintient des distributions stables, ce qui préserve le signal des gradients à travers la profondeur.

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

Comparé à une couche standard, le bloc résiduel ajoute simplement une connexion directe (*skip connection*) dans le graphe de calcul:

```{mermaid}
graph LR
    subgraph standard["Couche standard"]
        direction LR
        z_in("z") --> f("f(z ; θ)")
        f --> z_out("z'")
    end

    subgraph residuel["Bloc résiduel"]
        direction LR
        z_in2("z") --> f2("f(z ; θ)")
        f2 --> add("+")
        z_in2 -- "identité" --> add
        add --> z_out2("z + f(z)")
    end

    style f fill:#f5f5f5,stroke:#666666
    style f2 fill:#f5f5f5,stroke:#666666
    style add fill:#fff2cc,stroke:#d6b656
    style z_in fill:#dae8fc,stroke:#6c8ebf
    style z_out fill:#d5e8d4,stroke:#82b366
    style z_in2 fill:#dae8fc,stroke:#6c8ebf
    style z_out2 fill:#d5e8d4,stroke:#82b366
```

La connexion identité crée un **chemin autoroutier** pour le gradient: lors de la passe arrière, le gradient peut contourner le bloc $f$ et se propager directement vers les couches précédentes, sans multiplication par les jacobiennes potentiellement petites de $f$.

### Écrêtage du gradient

L'**écrêtage du gradient** (*gradient clipping*) est une technique pragmatique pour empêcher l'explosion du gradient. Avant chaque mise à jour, on limite la norme du gradient:

$$
\mathbf{g}' = \min\left(1, \frac{c}{\|\mathbf{g}\|}\right) \mathbf{g}
$$

Si $\|\mathbf{g}\| > c$, le gradient est réduit pour avoir une norme $c$. Cette opération préserve la direction du gradient tout en bornant son amplitude.

### Décroissance des poids

Comme nous l'avons vu au chapitre 3, la **régularisation L2** ajoute une pénalité sur la norme des paramètres à la perte:

$$
\mathcal{L}_{\text{rég}} = \mathcal{L} + \frac{\lambda}{2}\|\boldsymbol{\theta}\|^2
$$

Le gradient de la perte régularisée est $\nabla_{\boldsymbol{\theta}} \mathcal{L}_{\text{rég}} = \nabla_{\boldsymbol{\theta}} \mathcal{L} + \lambda \boldsymbol{\theta}$. La mise à jour de paramètres devient:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta(\nabla_{\boldsymbol{\theta}} \mathcal{L} + \lambda \boldsymbol{\theta}_t) = (1 - \eta\lambda)\boldsymbol{\theta}_t - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}
$$

Le facteur $(1-\eta\lambda)$ contracte légèrement les poids à chaque pas, d'où le nom de **décroissance des poids** (*weight decay*). Comme au chapitre 3, cela correspond à un prior gaussien $p(\boldsymbol{\theta}) \propto \exp(-\frac{\lambda}{2}\|\boldsymbol{\theta}\|^2)$ sur les paramètres, et l'estimation MAP avec ce prior est équivalente à la régularisation L2.

En pratique, la décroissance des poids ne s'applique généralement pas aux biais. La valeur $\lambda = 10^{-4}$ à $10^{-2}$ est courante.

### Dropout

La **décroissance des poids** pénalise les paramètres individuellement. Le **dropout** {cite}`srivastava2014dropout` agit différemment: il désactive aléatoirement des neurones à chaque passe avant pendant l'entraînement. Formellement, pour chaque couche cachée $\ell$, on applique un masque de Bernoulli:

$$
\boldsymbol{\epsilon} \sim \text{Ber}(1-p)^{\otimes m}, \qquad \tilde{\mathbf{z}}_\ell = \frac{1}{1-p}(\boldsymbol{\epsilon} \odot \mathbf{z}_\ell)
$$

où $p \in [0, 1)$ est le **taux de dropout** (probabilité qu'un neurone soit désactivé), $\boldsymbol{\epsilon} \in \{0,1\}^m$ est le masque aléatoire, et le facteur $\frac{1}{1-p}$ est un **rescaling inversé** (*inverted dropout*): il compense l'absence de neurones pendant l'entraînement, de sorte que l'espérance des activations reste inchangée:

$$
\mathbb{E}\left[\frac{1}{1-p}\epsilon_j z_j\right] = \frac{1}{1-p}(1-p) z_j = z_j
$$

À l'inférence, on désactive le dropout et on utilise le réseau complet sans rescaling.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
%config InlineBackend.figure_format = 'retina'

np.random.seed(3)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
titles = ['Réseau complet (inférence)', 'Dropout $p=0{,}5$ (entraînement)']

layer_sizes = [3, 5, 5, 2]
p_drop = 0.5
np.random.seed(3)
dropped = [np.random.rand(n) < p_drop for n in layer_sizes]
dropped[0]  = [False]*layer_sizes[0]  # inputs never dropped
dropped[-1] = [False]*layer_sizes[-1] # outputs never dropped

for ax_idx, ax in enumerate(axes):
    ax.set_xlim(-0.5, len(layer_sizes)-0.5)
    ax.set_ylim(-0.5, max(layer_sizes)-0.5)
    ax.axis('off')
    ax.set_title(titles[ax_idx], fontsize=10)

    positions = []
    for l, n in enumerate(layer_sizes):
        offset = (max(layer_sizes) - n) / 2
        pos = [(l, offset + i) for i in range(n)]
        positions.append(pos)

    # Draw edges
    for l in range(len(layer_sizes)-1):
        for i, (x1, y1) in enumerate(positions[l]):
            for j, (x2, y2) in enumerate(positions[l+1]):
                if ax_idx == 1 and (dropped[l][i] or dropped[l+1][j]):
                    lw, alpha, col = 0.5, 0.12, 'gray'
                else:
                    lw, alpha, col = 1.2, 0.4, '#555555'
                ax.plot([x1, x2], [y1, y2], '-', color=col, lw=lw, alpha=alpha, zorder=1)

    # Draw nodes
    for l, pos in enumerate(positions):
        for i, (x, y) in enumerate(pos):
            is_dropped = ax_idx == 1 and dropped[l][i]
            fc = '#dddddd' if is_dropped else ('#dae8fc' if l == 0 else ('#d5e8d4' if l == len(layer_sizes)-1 else '#fff2cc'))
            ec = '#aaaaaa' if is_dropped else '#333333'
            circ = plt.Circle((x, y), 0.28, fc=fc, ec=ec, lw=1.5, zorder=3)
            ax.add_patch(circ)
            if is_dropped:
                ax.text(x, y, r'$\times$', ha='center', va='center', fontsize=11, color='#cc0000', zorder=4)

    # Layer labels
    for l, label in enumerate(['Entrée', 'Cachée 1', 'Cachée 2', 'Sortie']):
        ax.text(l, -0.3, label, ha='center', va='top', fontsize=8, color='#555')

legend_elems = [
    mpatches.Patch(fc='#dae8fc', ec='#333', label='Neurone actif (entrée)'),
    mpatches.Patch(fc='#fff2cc', ec='#333', label='Neurone actif (caché)'),
    mpatches.Patch(fc='#dddddd', ec='#aaa', label='Neurone désactivé'),
]
axes[1].legend(handles=legend_elems, fontsize=7.5, loc='upper right')

plt.suptitle('Dropout: désactivation aléatoire de neurones pendant l\'entraînement', fontsize=11)
plt.tight_layout()
```

La justification intuitive du dropout est double. D'abord, il empêche la **co-adaptation** des neurones: chaque neurone ne peut pas compter sur les autres pour corriger ses erreurs, ce qui l'oblige à apprendre des caractéristiques utiles de façon indépendante. Ensuite, l'entraînement avec dropout revient à entraîner simultanément un ensemble de $2^m$ sous-réseaux différents qui partagent leurs paramètres, et l'inférence avec le réseau complet est une approximation de la moyenne de cet ensemble.

Des taux de dropout courants sont $p = 0{,}1$ à $0{,}2$ pour les couches convolutives et $p = 0{,}5$ pour les couches entièrement connectées.

La figure ci-dessous illustre l'effet de la régularisation sur un petit réseau sujet au surapprentissage.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

np.random.seed(1)

def make_moons(n=120, noise=0.2):
    t = np.linspace(0, np.pi, n//2)
    x1 = np.column_stack([np.cos(t), np.sin(t)]) + noise*np.random.randn(n//2, 2)
    x2 = np.column_stack([1-np.cos(t), -np.sin(t)+0.5]) + noise*np.random.randn(n//2, 2)
    X = np.vstack([x1, x2])
    y = np.array([0]*(n//2) + [1]*(n//2))
    return X, y

X_all, y_all = make_moons(120, 0.25)
# Split: 40 train, 80 val
idx = np.random.permutation(len(y_all))
X_tr, y_tr = X_all[idx[:40]], y_all[idx[:40]]
X_val, y_val = X_all[idx[40:]], y_all[idx[40:]]

def relu(x):    return np.maximum(0, x)
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,- 50,50)))
def ce(p, y):
    p = np.clip(p, 1e-7, 1-1e-7)
    return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))

def train_mlp(X_tr, y_tr, X_val, y_val, H=32, n_epochs=300,
              eta=0.01, lam=0.0, p_drop=0.0):
    np.random.seed(0)
    W1 = np.random.randn(2, H)*np.sqrt(2/2)
    b1 = np.zeros(H)
    W2 = np.random.randn(H,1)*np.sqrt(2/H)
    b2 = np.zeros(1)
    tr_losses, val_losses = [], []
    for _ in range(n_epochs):
        # Forward + dropout
        a1 = X_tr @ W1 + b1
        z1 = relu(a1)
        if p_drop > 0:
            mask = (np.random.rand(*z1.shape) > p_drop).astype(float)
            z1_drop = z1 * mask / (1-p_drop)
        else:
            z1_drop = z1
        a2 = z1_drop @ W2 + b2
        p  = sigmoid(a2)
        yb = y_tr.reshape(-1,1).astype(float)
        # Backward
        dp  = (p - yb) / len(y_tr)
        dW2 = z1_drop.T @ dp + lam*W2/len(y_tr)
        db2 = dp.sum(0)
        dz1 = dp @ W2.T
        if p_drop > 0:
            dz1 = dz1 * mask / (1-p_drop)
        da1 = dz1 * (a1 > 0)
        dW1 = X_tr.T @ da1 + lam*W1/len(y_tr)
        db1 = da1.sum(0)
        W1 -= eta*dW1; b1 -= eta*db1
        W2 -= eta*dW2; b2 -= eta*db2
        # Losses
        tr_losses.append(ce(sigmoid(relu(X_tr@W1+b1)@W2+b2), y_tr.reshape(-1,1)))
        val_losses.append(ce(sigmoid(relu(X_val@W1+b1)@W2+b2), y_val.reshape(-1,1)))
    return W1, b1, W2, b2, tr_losses, val_losses

W1a,b1a,W2a,b2a,tr_a,va_a = train_mlp(X_tr,y_tr,X_val,y_val, H=64, lam=0.0, p_drop=0.0)
W1b,b1b,W2b,b2b,tr_b,va_b = train_mlp(X_tr,y_tr,X_val,y_val, H=64, lam=1e-2, p_drop=0.4)

xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 200), np.linspace(-1, 1.8, 200))
Xg = np.column_stack([xx.ravel(), yy.ravel()])

def predict(Xg, W1,b1,W2,b2):
    return sigmoid(relu(Xg@W1+b1)@W2+b2).reshape(xx.shape)

Za = predict(Xg, W1a,b1a,W2a,b2a)
Zb = predict(Xg, W1b,b1b,W2b,b2b)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
epochs = np.arange(1, 301)
colors_cls = ['#4878CF', '#D65F5F']

for col, (tr_l, va_l, Za_Zb, W1_,b1_,W2_,b2_, title) in enumerate([
    (tr_a, va_a, Za, W1a,b1a,W2a,b2a, 'Sans régularisation'),
    (tr_b, va_b, Zb, W1b,b1b,W2b,b2b, 'Avec dropout + décroissance des poids'),
]):
    # Courbes de convergence
    ax = axes[0, col]
    ax.plot(epochs, tr_l, 'C0-',  lw=2, label='Entraînement')
    ax.plot(epochs, va_l, 'C1--', lw=2, label='Validation')
    ax.set_xlabel('Époque');  ax.set_ylabel('Perte')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)

    # Frontière de décision
    ax = axes[1, col]
    ax.contourf(xx, yy, Za_Zb, levels=50, cmap='RdBu_r', alpha=0.65, vmin=0, vmax=1)
    ax.contour(xx, yy, Za_Zb, levels=[0.5], colors='k', linewidths=1.5)
    for cls in [0,1]:
        m_tr  = y_tr  == cls;  m_val = y_val == cls
        ax.scatter(X_tr[m_tr,0],  X_tr[m_tr,1],  c=colors_cls[cls], marker='o', s=60, edgecolors='k', lw=0.8, zorder=5)
        ax.scatter(X_val[m_val,0],X_val[m_val,1], c=colors_cls[cls], marker='s', s=40, alpha=0.5, zorder=4)
    ax.set_xlabel(r'$x_1$');  ax.set_ylabel(r'$x_2$')
    ax.set_title('Frontière de décision', fontsize=10)
    ax.grid(True, alpha=0.2)

axes[0,0].text(0.97, 0.97, 'Surapprentissage', transform=axes[0,0].transAxes,
               ha='right', va='top', fontsize=9, color='C1',
               bbox=dict(boxstyle='round', fc='#fff3f3', ec='C1', alpha=0.9))
fig.legend(['Entraîn. (ronds)', 'Valid. (carrés)'],
           loc='lower center', ncol=2, fontsize=9, framealpha=0.9)
plt.suptitle('Effet de la régularisation sur le surapprentissage', fontsize=12, y=1.01)
plt.tight_layout()
```

Sans régularisation, le réseau mémorise les 40 exemples d'entraînement et produit une frontière très irrégulière qui généralise mal (écart entre les pertes d'entraînement et de validation). Avec dropout et décroissance des poids, la frontière est plus lisse et la perte de validation reste proche de la perte d'entraînement.

## Implémentation

Cette section réunit les concepts du chapitre dans une implémentation complète d'un MLP en NumPy. Le code est volontairement auto-contenu et commenté pas à pas: l'objectif est de rendre le lien entre les équations et le code aussi direct que possible.

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

### Entraînement sur XOR

Reprenons le problème XOR du début du chapitre et entraînons le MLP jusqu'à la convergence.

```{code-cell} python

import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])

mlp = MLP(n_input=2, n_hidden=8, n_output=1, eta=0.01, seed=42)

losses = []
for epoch in range(2000):
    # Sur un problème à 4 exemples, on fait un seul mini-lot = tout le dataset
    loss = mlp.train_step(X_xor, y_xor)
    losses.append(loss)

print("Prédictions après entraînement:")
probs = mlp.predict_proba(X_xor).ravel()
for (x1, x2), p, yv in zip(X_xor, probs, y_xor):
    print(f"  x=({int(x1)},{int(x2)}): p(y=1)={p:.3f}  →  ŷ={int(p>=0.5)}  (vrai: {yv})")

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(losses, 'C0', lw=1.5)
ax.set_xlabel('Époque')
ax.set_ylabel('Perte (entropie croisée)')
ax.set_title('Convergence du MLP sur XOR')
ax.grid(True, alpha=0.3)
plt.tight_layout()
```

Après quelques centaines d'époques, la perte converge vers zéro et le réseau classe correctement les quatre points. La courbe de convergence montre le profil typique de l'entraînement d'un petit réseau: une descente rapide au début, suivie d'un plateau puis d'une nouvelle descente à mesure que l'optimiseur trouve un bon bassin d'attraction.

## Résumé

Ce chapitre a montré comment les réseaux de neurones s'inscrivent dans la progression des modèles vus dans les chapitres précédents. Le point de départ est toujours le cadre de maximum de vraisemblance: un modèle prédit les paramètres d'une distribution conditionnelle. La nouveauté est que la transformation des entrées (la fonction $\boldsymbol{\phi}$) est désormais apprise plutôt que fixée à l'avance. Le problème XOR a illustré pourquoi cette flexibilité est nécessaire: certaines fonctions simples sont inaccessibles aux modèles linéaires, et une couche cachée suffit à les résoudre en transformant l'espace des entrées.

La rétropropagation applique la règle de la chaîne pour calculer les gradients dans un réseau profond. Le mode arrière (VJP) est efficace pour les fonctions à sortie scalaire. La différentiation automatique généralise ce principe à tout programme composé d'opérations différentiables.

Pour optimiser les paramètres, SGD par mini-lots reste la base. Le momentum amortit les oscillations en accumulant une vitesse dans les directions stables. Adam combine momentum et taux d'apprentissage adaptatifs par dimension, ce qui en fait l'optimiseur par défaut pour la plupart des applications. Pour éviter le surapprentissage, la décroissance des poids pénalise les paramètres de grande norme, et le dropout désactive aléatoirement des neurones pendant l'entraînement.

Les chapitres suivants présenteront des architectures spécialisées qui exploitent la structure des données: les réseaux convolutifs pour les images et les réseaux récurrents pour les séquences.

```{admonition} Ce que vous devez retenir
:class: tip

1. **Le problème XOR motive les architectures multicouches.** Un modèle linéaire ne peut pas séparer XOR; une couche cachée transforme l'espace et rend le problème linéairement séparable.

2. **Les réseaux de neurones apprennent leurs caractéristiques.** Contrairement aux modèles à expansion fixe (chapitre 4), la transformation $\boldsymbol{\phi}$ est paramétrique et optimisée avec les données.

3. **La non-linéarité est indispensable.** Sans fonctions d'activation, empiler des couches linéaires ne donne qu'une transformation linéaire équivalente à une seule couche.

4. **La rétropropagation calcule les gradients efficacement.** Elle propage le gradient de la perte vers l'arrière, couche par couche, via le mode arrière (VJP) de la règle de la chaîne.

5. **Adam est l'optimiseur par défaut.** Il combine momentum et taux adaptatifs par dimension, avec correction du biais pour les premières itérations.

6. **Dropout et décroissance des poids réduisent le surapprentissage.** Le dropout entraîne un ensemble implicite de sous-réseaux; la décroissance des poids correspond à un prior gaussien sur les paramètres.

7. **L'entraînement en profondeur pose des défis spécifiques.** Le gradient qui disparaît ou explose nécessite l'initialisation soignée (Glorot, He), la normalisation par lots, ou les connexions résiduelles.
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

````{admonition} Exercice 6: Adam à la main ★★
:class: hint dropdown

Considérez la fonction scalaire $f(\theta) = \theta^2$ et un gradient calculé à $\theta_0 = 1$ (soit $g_0 = 2$). Partez de $\theta_0 = 1$, $m_0 = 0$, $s_0 = 0$, avec $\eta = 0{,}1$, $\beta_1 = 0{,}9$, $\beta_2 = 0{,}999$, $\epsilon = 10^{-8}$.

1. Calculez les valeurs $m_1$, $s_1$, $\hat{m}_1$, $\hat{s}_1$ après la première itération.
2. Calculez la mise à jour $\theta_1$.
3. Comparez avec la mise à jour SGD pure $\theta_1^{\text{SGD}} = \theta_0 - \eta g_0$. Quel algorithme fait un pas plus grand? Pourquoi?
4. Que se passerait-il sans la correction du biais $\hat{m}_1 = m_1/(1-\beta_1)$? Calculez la mise à jour sans correction.
````

````{admonition} Solution Exercice 6
:class: dropdown

**1. Moments après la première itération** ($t=1$, $g_0 = 2\theta_0 = 2$):

$$
m_1 = 0{,}9 \times 0 + 0{,}1 \times 2 = 0{,}2
$$
$$
s_1 = 0{,}999 \times 0 + 0{,}001 \times 4 = 0{,}004
$$

**Correction du biais** ($t=1$):

$$
\hat{m}_1 = \frac{0{,}2}{1 - 0{,}9^1} = \frac{0{,}2}{0{,}1} = 2{,}0
$$
$$
\hat{s}_1 = \frac{0{,}004}{1 - 0{,}999^1} = \frac{0{,}004}{0{,}001} = 4{,}0
$$

**2. Mise à jour Adam:**

$$
\theta_1 = 1 - 0{,}1 \times \frac{2{,}0}{\sqrt{4{,}0} + 10^{-8}} = 1 - 0{,}1 \times \frac{2}{2} = 1 - 0{,}1 = 0{,}9
$$

**3. Comparaison avec SGD:**

$$
\theta_1^{\text{SGD}} = 1 - 0{,}1 \times 2 = 0{,}8
$$

SGD fait un pas plus grand (de $0{,}2$ vs $0{,}1$ pour Adam). Adam normalise le gradient par sa magnitude: $\hat{m}_1 / \sqrt{\hat{s}_1} = 2/2 = 1$. Pour un gradient de magnitude constante, Adam fait un pas de taille $\eta$ indépendamment de la magnitude du gradient. C'est précisément la normalisation adaptative qui rend Adam moins sensible au choix de $\eta$.

**4. Sans correction du biais:**

$$
\theta_1^{\text{sans correction}} = 1 - 0{,}1 \times \frac{m_1}{\sqrt{s_1} + \epsilon} = 1 - 0{,}1 \times \frac{0{,}2}{\sqrt{0{,}004}} = 1 - 0{,}1 \times \frac{0{,}2}{0{,}0632} \approx 1 - 0{,}316 = 0{,}684
$$

Sans correction, le premier pas serait très grand (les moments sont sous-estimés par rapport à leur valeur asymptotique, et $s_1$ est petit, donc $1/\sqrt{s_1}$ est grand). La correction du biais ramène les moments à leur vraie valeur dès la première itération.
````

````{admonition} Exercice 7: Dropout et espérance des activations ★★
:class: hint dropdown

Soit $z_j$ l'activation d'un neurone et $\epsilon_j \sim \text{Ber}(1-p)$ le masque de dropout. La sortie avec dropout inversé est $\tilde{z}_j = \frac{\epsilon_j}{1-p} z_j$.

1. Montrez que $\mathbb{E}[\tilde{z}_j] = z_j$. Pourquoi cette propriété est-elle importante pour l'inférence?

2. Calculez $\text{Var}[\tilde{z}_j]$ en fonction de $z_j$ et $p$.

3. Pour $p = 0{,}5$ et $z_j = 1$, calculez la variance. Que se passe-t-il quand $p \to 1$?

4. Implémentez une fonction `dropout(z, p, training)` en NumPy qui applique le dropout inversé pendant l'entraînement et retourne $z$ inchangé à l'inférence.
````

````{admonition} Solution Exercice 7
:class: dropdown

**1. Espérance:**

$\mathbb{E}[\epsilon_j] = 1-p$ car $\epsilon_j \sim \text{Ber}(1-p)$. Donc:

$$
\mathbb{E}[\tilde{z}_j] = \frac{1}{1-p}\mathbb{E}[\epsilon_j] z_j = \frac{1}{1-p}(1-p) z_j = z_j
$$

Cette propriété garantit que les activations ont la même espérance à l'entraînement et à l'inférence. Sans elle, il faudrait rescaler les poids à l'inférence, ce qui complique le déploiement.

**2. Variance:**

$\text{Var}[\epsilon_j] = p(1-p)$ (variance de Bernoulli). Puisque $z_j$ est déterministe:

$$
\text{Var}[\tilde{z}_j] = \frac{z_j^2}{(1-p)^2}\text{Var}[\epsilon_j] = \frac{z_j^2}{(1-p)^2} p(1-p) = \frac{p}{1-p} z_j^2
$$

**3. Pour $p = 0{,}5$, $z_j = 1$:**

$$
\text{Var}[\tilde{z}_j] = \frac{0{,}5}{0{,}5} \times 1 = 1
$$

Quand $p \to 1$, $\text{Var} \to \infty$: le bruit devient arbitrairement grand, ce qui rend l'entraînement instable. En pratique, on ne dépasse pas $p = 0{,}5$.

**4. Implémentation:**

```python
import numpy as np

def dropout(z, p, training=True):
    """Dropout inversé. p = taux de désactivation."""
    if not training or p == 0.0:
        return z
    mask = (np.random.rand(*z.shape) > p) / (1 - p)
    return z * mask
```
````

````{admonition} Exercice 8: Gradient de l'entropie croisée avec softmax ★★★ (optionnel IFT3395)
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

````{admonition} Solution Exercice 8
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

Ce résultat est élégant: pour calculer le gradient par rapport aux pré-activations de la couche softmax, il suffit de soustraire 1 à la probabilité prédite pour la vraie classe. Pas besoin de calculer explicitement la jacobienne $K \times K$ du softmax (qui serait coûteuse pour de grandes sorties). En pratique:

```python
def softmax_cross_entropy_grad(logits, c):
    """Gradient de l'entropie croisée par rapport aux logits."""
    exp_a = np.exp(logits - logits.max())  # stabilité numérique
    p = exp_a / exp_a.sum()
    p[c] -= 1  # soustraction de e_c
    return p
```
````

````{admonition} Exercice 9: VJP de la couche affine ★★
:class: hint dropdown

Soit $f(\mathbf{z}) = W\mathbf{z} + \mathbf{b}$ avec $W \in \mathbb{R}^{m \times n}$, $\mathbf{z} \in \mathbb{R}^n$, et un vecteur adjoint $\mathbf{u} \in \mathbb{R}^m$.

1. Écrivez la jacobienne $\mathbf{J}_f$ par rapport à $\mathbf{z}$. Quelle est sa taille?

2. Calculez $\mathbf{J}_f^\top \mathbf{u}$ (la règle VJP par rapport à $\mathbf{z}$). Montrez que le résultat est $W^\top \mathbf{u}$. Interprétez: $W^\top$ est l'opérateur adjoint de $W$.

3. Traitez maintenant $W$ comme variable. Soit $g(W) = W\mathbf{z} + \mathbf{b}$ avec $\mathbf{z}$ fixé. Écrivez $g$ composante par composante: $(g(W))_i = \sum_j W_{ij} z_j + b_i$. En déduire $\frac{\partial g_i}{\partial W_{kl}}$, puis la règle VJP par rapport à $W$.

4. Montrez que la règle VJP par rapport à $W$ est $\mathbf{u}\mathbf{z}^\top$ (produit externe), et par rapport à $\mathbf{b}$ est $\mathbf{u}$.

5. Retrouvez ces formules dans les équations de rétropropagation de la section "Exemple: MLP avec une couche cachée". Comment s'appellent-elles dans ce contexte?
````

````{admonition} Solution Exercice 9
:class: dropdown

**1. Jacobienne par rapport à $\mathbf{z}$:**

$(g(\mathbf{z}))_i = \sum_j W_{ij} z_j + b_i$, donc $\frac{\partial g_i}{\partial z_j} = W_{ij}$.

La jacobienne est $\mathbf{J}_f = W \in \mathbb{R}^{m \times n}$.

**2. VJP par rapport à $\mathbf{z}$:**

$$
\mathbf{J}_f^\top \mathbf{u} = W^\top \mathbf{u} \in \mathbb{R}^n
$$

$W^\top$ est bien l'adjoint de l'opérateur $W$: si $W$ projette $\mathbb{R}^n \to \mathbb{R}^m$, son adjoint $W^\top$ projette en sens inverse $\mathbb{R}^m \to \mathbb{R}^n$. Aucune matrice supplémentaire n'est formée: $W$ est déjà disponible depuis la passe avant.

**3. Jacobienne par rapport à $W$:**

Vectorisons $W$ en $\operatorname{vec}(W) \in \mathbb{R}^{mn}$ (concaténation des colonnes). La jacobienne $\mathbf{J}_g \in \mathbb{R}^{m \times mn}$ a pour blocs $(\mathbf{J}_g)_{i, (k-1)n+l} = \frac{\partial g_i}{\partial W_{kl}} = z_l \cdot \mathbb{1}(i=k)$.

**4. VJP par rapport à $W$:**

Sans vectoriser: le résultat du VJP doit avoir la même forme que $W$, soit $\mathbb{R}^{m \times n}$. On a:

$$
\left(\mathbf{J}_W^\top \mathbf{u}\right)_{kl} = \sum_i u_i \frac{\partial g_i}{\partial W_{kl}} = u_k z_l
$$

Ce qui donne la matrice $\mathbf{u}\mathbf{z}^\top \in \mathbb{R}^{m \times n}$ — un produit externe $O(mn)$, exactement le coût minimal pour produire une matrice de cette taille.

Par rapport à $\mathbf{b}$: $\frac{\partial g_i}{\partial b_k} = \mathbb{1}(i=k)$, donc $\mathbf{J}_\mathbf{b}^\top \mathbf{u} = \mathbf{u}$.

**5. Correspondance avec la rétropropagation:**

Dans la section "Exemple: MLP avec une couche cachée", avec $\mathbf{u} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}_1}$:

$$
\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}_1} \mathbf{x}^\top = \mathbf{u}\mathbf{z}^\top, \qquad
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}_1} = \mathbf{u}
$$

Ce sont exactement les règles VJP dérivées ci-dessus, appliquées avec $\mathbf{z} = \mathbf{x}$.
````

````{admonition} Exercice 10: VJP de l'activation élémentaire ★★
:class: hint dropdown

Soit $g: \mathbb{R}^m \to \mathbb{R}^m$ définie par $g(\mathbf{a})_i = \varphi(a_i)$, où $\varphi: \mathbb{R} \to \mathbb{R}$ est une activation scalaire différentiable.

1. Calculez la jacobienne $\mathbf{J}_g(\mathbf{a}) \in \mathbb{R}^{m \times m}$. Quelle est sa structure particulière? Quel coût en mémoire si $m = 10^4$?

2. Calculez la règle VJP: $\mathbf{J}_g^\top \mathbf{u}$. Montrez qu'elle se réduit à $\mathbf{u} \odot \varphi'(\mathbf{a})$. Quel est le coût en mémoire?

3. Spécialisez à $\varphi = \sigma$ (sigmoïde). En utilisant la formule $\sigma'(a) = \sigma(a)(1-\sigma(a))$ de l'exercice 2, écrivez la règle VJP uniquement en termes de $\mathbf{u}$ et $g(\mathbf{a}) = \sigma(\mathbf{a})$ (sans recalculer la passe avant).

4. Pour $\varphi = \text{ReLU}$, la dérivée $\text{ReLU}'(a) = \mathbb{1}(a > 0)$ n'est pas définie en $a = 0$. Comment les bibliothèques d'AD gèrent-elles conventionnellement ce cas?
````

````{admonition} Solution Exercice 10
:class: dropdown

**1. Jacobienne de l'activation élémentaire:**

$\frac{\partial g_i}{\partial a_j} = \varphi'(a_i) \cdot \mathbb{1}(i = j)$, donc:

$$
\mathbf{J}_g(\mathbf{a}) = \operatorname{diag}(\varphi'(\mathbf{a})) \in \mathbb{R}^{m \times m}
$$

C'est une matrice *diagonale*. Malgré cela, si on la stockait naïvement, le coût serait $O(m^2)$. Pour $m = 10^4$: $10^8$ flottants $\approx$ 800 Mo — prohibitif.

**2. Règle VJP:**

$$
\mathbf{J}_g^\top \mathbf{u} = \operatorname{diag}(\varphi'(\mathbf{a}))^\top \mathbf{u} = \operatorname{diag}(\varphi'(\mathbf{a})) \mathbf{u} = \mathbf{u} \odot \varphi'(\mathbf{a})
$$

Coût: $O(m)$ en temps et en mémoire. La matrice diagonale n'est jamais formée.

**3. VJP de la sigmoïde avec réutilisation:**

Puisque $\sigma'(a) = \sigma(a)(1 - \sigma(a))$ et $g(\mathbf{a}) = \sigma(\mathbf{a})$ est disponible depuis la passe avant:

$$
\mathbf{J}_g^\top \mathbf{u} = \mathbf{u} \odot g(\mathbf{a}) \odot (1 - g(\mathbf{a}))
$$

Pas besoin de recalculer $\sigma(\mathbf{a})$: la passe avant l'a déjà produit et stocké. C'est la raison pour laquelle les implémentations de rétropropagation *cachent* les activations intermédiaires.

**4. ReLU en $a = 0$:**

Par convention, la quasi-totalité des bibliothèques (JAX, PyTorch, TensorFlow) définissent $\text{ReLU}'(0) = 0$. Cette convention est cohérente avec le sous-différentiel de la fonction convexe $\max(0, a)$, et le point $a = 0$ forme un ensemble de mesure nulle qui n'affecte pas l'entraînement en pratique.
````

````{admonition} Exercice 11: JVP de la couche affine ★★
:class: hint dropdown

Soit $f(\mathbf{z}) = W\mathbf{z} + \mathbf{b}$ avec $W \in \mathbb{R}^{m \times n}$, et un vecteur tangent $\mathbf{v} \in \mathbb{R}^n$.

1. Calculez la règle JVP: $\mathbf{J}_f \cdot \mathbf{v}$. Quel est son coût?

2. Comparez: calculer le gradient complet $\nabla_\mathbf{z} \mathcal{L}$ via $n$ JVPs (un par composante $\mathbf{e}_i$) versus un seul VJP. Combien d'opérations arithmétiques chaque approche nécessite-t-elle?

3. Supposez $n = m = 1000$. Donnez les coûts numériques des deux approches pour calculer le gradient complet par rapport à tous les paramètres du réseau.

4. Dans quel cas le mode avant (JVP) est-il *préférable* au mode arrière (VJP)?
````

````{admonition} Solution Exercice 11
:class: dropdown

**1. Règle JVP:**

$$
\mathbf{J}_f \cdot \mathbf{v} = W\mathbf{v} \in \mathbb{R}^m
$$

Coût: $O(mn)$ — un produit matrice-vecteur.

**2. Comparaison JVP vs VJP pour le gradient complet:**

- **$n$ JVPs**: calculer $\mathbf{J}_f \mathbf{e}_1, \ldots, \mathbf{J}_f \mathbf{e}_n$ reconstruit la jacobienne colonne par colonne. Coût total: $n \times O(mn) = O(mn^2)$.
- **1 VJP**: calculer $W^\top \mathbf{u}$ pour un seul $\mathbf{u}$. Coût: $O(mn)$.

Pour une perte scalaire, le VJP en mode arrière est donc $n$ fois plus efficace que le JVP en mode avant.

**3. Avec $n = m = 1000$:**

- Mode avant ($n$ JVPs): $1000 \times 10^6 = 10^9$ opérations
- Mode arrière (1 VJP): $10^6$ opérations

Le mode arrière est $10^3 = 1000$ fois moins coûteux — et c'est par couche. Sur un réseau de 100 couches, l'avantage s'accumule.

**4. Cas favorables au mode avant:**

Le JVP (mode avant) est préférable quand le nombre de sorties $m$ est grand mais le nombre d'entrées $n$ est petit. En pratique:
- Calcul de produits $\mathbf{J}_f \mathbf{v}$ pour des directions $\mathbf{v}$ spécifiques (e.g., directions de courbure en optimisation du second ordre)
- Différentiation par rapport à un petit nombre de paramètres scalaires (e.g., hyperparamètres)
- Sensibilités directionnelles en analyse d'incertitude
````

