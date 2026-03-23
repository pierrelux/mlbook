---
kernelspec:
  name: python3
  display_name: Python 3
---

# Attention et transformeurs

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Expliquer le mécanisme d'attention comme un accès sélectif à l'information
- Définir l'attention par produit scalaire avec requêtes, clés et valeurs
- Distinguer l'attention croisée et l'auto-attention
- Décrire l'architecture d'un bloc transformeur (attention multi-têtes, réseau à propagation avant, connexions résiduelles, normalisation)
- Expliquer le rôle de l'encodage positionnel
- Distinguer les variantes encodeur, décodeur et encodeur-décodeur
- Relier le mécanisme d'attention à l'estimateur de Nadaraya-Watson
- Expliquer le compromis entre parallélisme et coût quadratique dans les transformeurs
```

Le chapitre précédent a montré comment les réseaux récurrents traitent les séquences en maintenant un état caché mis à jour à chaque pas de temps. Cette approche a deux limitations: le traitement est séquentiel (on ne peut pas paralléliser le calcul des différentes positions), et toute l'information doit transiter par un vecteur de taille fixe $\mathbf{h}_t$.

Le mécanisme d'attention résout ces deux problèmes en permettant à chaque position d'une séquence de consulter directement toutes les autres. Au lieu de compresser l'information dans un seul vecteur, l'attention calcule une moyenne pondérée sur toutes les positions, avec des poids appris qui dépendent du contenu. Le transformeur construit une architecture complète autour de ce mécanisme, en remplaçant entièrement la récurrence par de l'attention.

Dans ce chapitre, nous commençons par l'intuition derrière l'attention, puis nous formalisons le mécanisme avec les requêtes, clés et valeurs. Nous passons ensuite à l'auto-attention (où une séquence s'interroge elle-même), à l'attention multi-têtes, puis à l'architecture complète du transformeur. Nous terminons par les variantes (encodeur, décodeur, encodeur-décodeur) et les raisons de la domination actuelle des transformeurs.

## L'attention: accès sélectif à l'information

Considérons la traduction d'une phrase du français vers l'anglais avec un RNN encodeur-décodeur. L'encodeur lit la phrase source mot par mot et produit un état final $\mathbf{h}_T$ qui résume toute la phrase. Le décodeur génère ensuite la traduction mot par mot à partir de cet état.

Le problème est que $\mathbf{h}_T$ doit contenir toute l'information nécessaire pour traduire chaque mot de la phrase cible. Pour une phrase courte, c'est faisable. Pour une phrase longue, l'information est inévitablement compressée et dégradée.

Un traducteur humain ne procède pas ainsi. En traduisant le cinquième mot de la phrase cible, il retourne consulter les mots pertinents de la phrase source, pas un résumé compressé de toute la phrase. Le mécanisme d'attention imite ce comportement: à chaque pas de la génération, le décodeur peut "regarder en arrière" vers toutes les positions de l'encodeur et sélectionner l'information pertinente.

Cette idée, proposée par Bahdanau et al. {cite}`bahdanau2014neural`, a transformé la traduction automatique neuronale. Nous allons la formaliser, puis la généraliser au-delà du cadre encodeur-décodeur.

## Requêtes, clés et valeurs

Le mécanisme d'attention repose sur trois ingrédients. Nous disposons d'un ensemble de $T$ positions, chacune associée à un vecteur de représentation. L'attention extrait de l'information de ces positions en fonction d'une requête.

Soit une séquence de représentations $(\mathbf{x}_1, \ldots, \mathbf{x}_T)$, chaque $\mathbf{x}_t \in \mathbb{R}^d$. Nous définissons:

- la **requête** (*query*) $\mathbf{q} \in \mathbb{R}^{d_k}$: ce que nous cherchons;
- les **clés** (*keys*) $\mathbf{k}_t \in \mathbb{R}^{d_k}$: ce que chaque position "annonce" contenir;
- les **valeurs** (*values*) $\mathbf{v}_t \in \mathbb{R}^{d_v}$: l'information effectivement stockée à chaque position.

Cette terminologie s'éclaire par analogie avec un dictionnaire Python. Un dictionnaire fait une consultation dure: `d[clé]` retourne exactement une valeur, ou échoue si la clé est absente. L'attention fait une consultation souple: la requête est comparée à toutes les clés simultanément, et la sortie est un mélange des valeurs pondéré par le degré de correspondance entre la requête et chaque clé.

```{mermaid}
graph LR
    q["Requête q"]
    k1["Clé k₁"]
    k2["Clé k₂"]
    k3["Clé k₃"]
    s1["similarité"]
    s2["similarité"]
    s3["similarité"]
    sm["softmax"]
    v1["Valeur v₁"]
    v2["Valeur v₂"]
    v3["Valeur v₃"]
    out["Σ αₜ vₜ"]

    q --> s1
    q --> s2
    q --> s3
    k1 --> s1
    k2 --> s2
    k3 --> s3
    s1 --> sm
    s2 --> sm
    s3 --> sm
    sm --> out
    v1 --> out
    v2 --> out
    v3 --> out

    style q fill:#dae8fc,stroke:#6c8ebf
    style out fill:#d5e8d4,stroke:#82b366
    style sm fill:#fff2cc,stroke:#d6b656
```

Quand les poids sont concentrés sur une seule position, la consultation souple se rapproche d'une consultation dure; quand ils sont uniformes, la sortie est la moyenne de toutes les valeurs.

L'attention calcule un score de similarité entre la requête et chaque clé, normalise ces scores en probabilités, puis retourne la moyenne pondérée des valeurs:

$$
\alpha_t = \frac{\exp(\mathbf{q}^\top \mathbf{k}_t / \sqrt{d_k})}{\sum_{s=1}^T \exp(\mathbf{q}^\top \mathbf{k}_s / \sqrt{d_k})}
$$ (eq:attention-weight)

$$
\text{Attention}(\mathbf{q}, K, V) = \sum_{t=1}^T \alpha_t\, \mathbf{v}_t
$$ (eq:attention-output)

Le score $\mathbf{q}^\top \mathbf{k}_t$ mesure la compatibilité entre la requête et la clé à la position $t$. Le softmax {eq}`eq:attention-weight` transforme ces scores en poids positifs qui somment à 1. La sortie {eq}`eq:attention-output` est une combinaison convexe des valeurs, pondérée par la pertinence de chaque position.

Prenons un exemple concret avec 3 positions et une requête $\mathbf{q} = (1, 0)$. Si les clés sont $\mathbf{k}_1 = (1, 0)$, $\mathbf{k}_2 = (0, 1)$, $\mathbf{k}_3 = (1, 1)$, les scores sont $1$, $0$ et $1$: la requête « sélectionne » les positions dont les clés ont une forte composante en première dimension. Après le softmax, les positions 1 et 3 reçoivent chacune environ 42% du poids, tandis que la position 2 n'en reçoit que 16%. La sortie est dominée par les valeurs des positions les plus compatibles avec la requête.

La division par $\sqrt{d_k}$ est une normalisation. Sans elle, quand la dimension $d_k$ est grande, les produits scalaires tendent à avoir des magnitudes élevées, ce qui pousse le softmax vers des distributions très piquées (presque concentrées sur un seul élément). La normalisation par $\sqrt{d_k}$ maintient la variance des scores à une échelle raisonnable.

En notation matricielle, si nous traitons toutes les requêtes simultanément ($Q \in \mathbb{R}^{T_q \times d_k}$ est la matrice des requêtes, $K \in \mathbb{R}^{T \times d_k}$ celle des clés, $V \in \mathbb{R}^{T \times d_v}$ celle des valeurs):

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
$$ (eq:attention-matrix)

Le produit $Q K^\top \in \mathbb{R}^{T_q \times T}$ contient tous les scores de similarité. Le softmax est appliqué sur chaque ligne (chaque requête). La multiplication par $V$ produit la sortie $\in \mathbb{R}^{T_q \times d_v}$.

### Lien avec l'estimation non paramétrique

La formule d'attention n'est pas nouvelle: elle apparaît sous une forme plus ancienne en régression non paramétrique. L'estimateur de Nadaraya-Watson {cite}`nadaraya1964estimating,watson1964smooth`, que nous avons rencontré dans le chapitre sur les $k$ plus proches voisins, prédit la sortie pour un point $\mathbf{x}$ par une moyenne pondérée des observations:

$$
\hat{y}(\mathbf{x}) = \sum_{i=1}^n \frac{K_\lambda(\mathbf{x}, \mathbf{x}_i)}{\sum_{j=1}^n K_\lambda(\mathbf{x}, \mathbf{x}_j)}\, y_i
$$

où $K_\lambda$ est un noyau qui mesure la similarité entre $\mathbf{x}$ et chaque observation $\mathbf{x}_i$. La structure est identique à celle de l'attention: le noyau joue le rôle du score de similarité $\mathbf{q}^\top \mathbf{k}_t$, la normalisation par la somme des noyaux est l'analogue du softmax, et les observations $y_i$ sont les valeurs.

La différence tient à ce qui est fixe et ce qui est appris. Dans Nadaraya-Watson, la fonction de similarité est choisie a priori (noyau gaussien, par exemple), et les « valeurs » sont les données brutes $y_i$. Dans l'attention, la similarité est paramétrée par des projections linéaires apprises ($W_Q$ et $W_K$), et les valeurs sont elles aussi transformées par $W_V$. L'attention est une généralisation paramétrique de la moyenne pondérée non paramétrique: elle apprend simultanément quoi chercher, comment comparer, et quoi retourner.

## Auto-attention

Dans le mécanisme d'attention décrit ci-dessus, les requêtes proviennent d'une séquence et les clés/valeurs d'une autre (par exemple, le décodeur interroge l'encodeur). C'est l'attention croisée.

L'auto-attention (*self-attention*) est le cas où les requêtes, clés et valeurs proviennent toutes de la même séquence. Chaque position de la séquence interroge toutes les autres positions (y compris elle-même) pour construire une nouvelle représentation qui intègre le contexte global.

Concrètement, soit $X \in \mathbb{R}^{T \times d}$ la matrice des représentations d'entrée (une ligne par position). On définit trois projections linéaires apprises:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

où $W_Q, W_K \in \mathbb{R}^{d \times d_k}$ et $W_V \in \mathbb{R}^{d \times d_v}$ sont des matrices de paramètres. La sortie est:

$$
\text{SelfAttention}(X) = \text{softmax}\!\left(\frac{(X W_Q)(X W_K)^\top}{\sqrt{d_k}}\right) (X W_V)
$$

La sortie a la même forme que l'entrée: $T$ vecteurs de dimension $d_v$. Chaque vecteur de sortie est une combinaison pondérée de toutes les valeurs, les poids étant déterminés par la similarité entre la requête de cette position et les clés de toutes les autres positions.

Une propriété importante de l'auto-attention est qu'elle est équivariante par permutation: si l'on permute les lignes de $X$, les lignes de la sortie sont permutées de la même façon. L'auto-attention ne contient aucune notion d'ordre intrinsèque. Contrairement au RNN, où la position $t$ reçoit nécessairement l'information des positions $1, \ldots, t-1$ via $\mathbf{h}_{t-1}$, l'auto-attention traite toutes les positions de façon symétrique. Nous verrons plus loin comment l'encodage positionnel restaure la notion d'ordre.

La complexité de calcul de l'auto-attention est $O(T^2 d)$: le produit $Q K^\top$ a $T^2$ entrées. Pour de longues séquences, cette complexité quadratique peut devenir un goulot d'étranglement. C'est le principal inconvénient du mécanisme, et il existe des travaux sur des variantes plus efficaces (attention linéaire, attention éparse), que nous ne détaillerons pas ici.

### Chemins directs entre positions

Dans un RNN, l'information de la position 1 doit traverser la chaîne $\mathbf{h}_1 \to \mathbf{h}_2 \to \cdots \to \mathbf{h}_T$ pour atteindre la position $T$. À chaque transition, le signal est transformé et potentiellement atténué. Comme nous l'avons vu au chapitre 9, cette chaîne est précisément ce qui cause la dissolution du gradient: le gradient subit $T - 1$ multiplications matricielles successives, et sa norme tend à décroître exponentiellement avec la longueur de la séquence.

```{mermaid}
graph LR
    x1["x₁"] --> h1["h₁"]
    x2["x₂"] --> h2["h₂"]
    x3["x₃"] --> h3["h₃"]
    x4["x₄"] --> h4["h₄"]
    h1 --> h2 --> h3 --> h4

    style x1 fill:#dae8fc,stroke:#6c8ebf
    style x2 fill:#dae8fc,stroke:#6c8ebf
    style x3 fill:#dae8fc,stroke:#6c8ebf
    style x4 fill:#dae8fc,stroke:#6c8ebf
    style h1 fill:#fff2cc,stroke:#d6b656
    style h2 fill:#fff2cc,stroke:#d6b656
    style h3 fill:#fff2cc,stroke:#d6b656
    style h4 fill:#fff2cc,stroke:#d6b656
```

L'auto-attention élimine ce goulot d'étranglement. Chaque position de sortie est directement connectée à chaque position d'entrée en une seule couche, sans intermédiaire:

```{mermaid}
graph LR
    x1["x₁"] --> y1["y₁"]
    x1 --> y2["y₂"]
    x1 --> y3["y₃"]
    x1 --> y4["y₄"]
    x2 --> y1
    x2 --> y2
    x2 --> y3
    x2 --> y4
    x3 --> y1
    x3 --> y2
    x3 --> y3
    x3 --> y4
    x4 --> y1
    x4 --> y2
    x4 --> y3
    x4 --> y4

    style x1 fill:#dae8fc,stroke:#6c8ebf
    style x2 fill:#dae8fc,stroke:#6c8ebf
    style x3 fill:#dae8fc,stroke:#6c8ebf
    style x4 fill:#dae8fc,stroke:#6c8ebf
    style y1 fill:#d5e8d4,stroke:#82b366
    style y2 fill:#d5e8d4,stroke:#82b366
    style y3 fill:#d5e8d4,stroke:#82b366
    style y4 fill:#d5e8d4,stroke:#82b366
```

La conséquence pour l'entraînement est directe: un chemin court entre deux positions signifie un chemin court pour le gradient. Les connexions résiduelles du bloc transformeur (section suivante) amplifient cet effet en ajoutant un chemin d'identité à travers chaque couche.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'retina'

mots = ['Le', 'chat', 'mange', 'la', 'souris']
T = len(mots)

# Scores construits à la main pour un motif plausible :
# "chat" attend "mange" (sujet-verbe), "mange" attend "souris" (verbe-objet),
# chaque mot attend aussi ses voisins immédiats.
scores = np.array([
    [1.0, 2.0, 0.5, 0.1, 0.1],   # Le -> chat (déterminant-nom)
    [1.5, 1.0, 2.5, 0.2, 0.3],   # chat -> mange (sujet-verbe)
    [0.3, 2.0, 1.0, 0.5, 2.2],   # mange -> chat, souris (verbe-arguments)
    [0.1, 0.1, 0.5, 1.0, 2.0],   # la -> souris (déterminant-nom)
    [0.1, 0.3, 2.5, 1.5, 1.0],   # souris -> mange (objet-verbe)
])

# Softmax par ligne
exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
alpha = exp_scores / exp_scores.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(alpha, cmap='Blues', vmin=0, vmax=alpha.max())
ax.set_xticks(range(T))
ax.set_xticklabels(mots)
ax.set_yticks(range(T))
ax.set_yticklabels(mots)
ax.set_xlabel('Position consultée (clé)')
ax.set_ylabel('Position qui consulte (requête)')
ax.set_title("Poids d'attention")

for i in range(T):
    for j in range(T):
        color = 'white' if alpha[i, j] > 0.35 else 'black'
        ax.text(j, i, f'{alpha[i,j]:.2f}', ha='center', va='center',
                fontsize=9, color=color)

fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
```

Chaque ligne de cette carte de chaleur montre comment une position distribue son attention sur les autres. Les poids sur chaque ligne somment à 1. On observe que « chat » concentre son attention sur « mange » (relation sujet-verbe), et que « mange » distribue la sienne entre « chat » et « souris » (ses arguments).

## Attention multi-têtes

Un seul mécanisme d'attention ne peut capturer qu'un seul type de relation entre les positions. En pratique, différentes positions peuvent être reliées de multiples façons: en syntaxe (sujet-verbe), en sémantique (coréférence), en proximité, etc.

L'attention multi-têtes exécute $H$ mécanismes d'attention en parallèle, chacun avec ses propres projections. La tête $h$ calcule:

$$
\text{head}_h = \text{Attention}(X W_Q^{(h)},\; X W_K^{(h)},\; X W_V^{(h)})
$$

où $W_Q^{(h)}, W_K^{(h)} \in \mathbb{R}^{d \times d_k}$ et $W_V^{(h)} \in \mathbb{R}^{d \times d_v}$. On choisit typiquement $d_k = d_v = d / H$ pour que le coût total soit comparable à une seule tête de dimension $d$.

Les sorties des $H$ têtes sont concaténées puis projetées:

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)\, W_O
$$ (eq:multihead)

où $W_O \in \mathbb{R}^{Hd_v \times d}$ ramène la dimension à $d$.

```{mermaid}
graph LR
    X["X ∈ ℝ<sup>T×d</sup>"]
    h1["Tête 1<br/>(W<sub>Q</sub>¹, W<sub>K</sub>¹, W<sub>V</sub>¹)"]
    h2["Tête 2<br/>(W<sub>Q</sub>², W<sub>K</sub>², W<sub>V</sub>²)"]
    hH["Tête H<br/>(W<sub>Q</sub>ᴴ, W<sub>K</sub>ᴴ, W<sub>V</sub>ᴴ)"]
    cat["Concat"]
    wo["W<sub>O</sub>"]
    out["Sortie ∈ ℝ<sup>T×d</sup>"]

    X --> h1
    X --> h2
    X --> hH
    h1 --> cat
    h2 --> cat
    hH --> cat
    cat --> wo --> out

    style X fill:#dae8fc,stroke:#6c8ebf
    style out fill:#d5e8d4,stroke:#82b366
    style h1 fill:#fff2cc,stroke:#d6b656
    style h2 fill:#fff2cc,stroke:#d6b656
    style hH fill:#fff2cc,stroke:#d6b656
    style cat fill:#f5f5f5,stroke:#666666
    style wo fill:#f5f5f5,stroke:#666666
```

Chaque tête peut apprendre à capturer un type de relation différent: une tête peut se spécialiser dans les dépendances locales, une autre dans les dépendances à longue portée, une autre encore dans les relations syntaxiques.

## Le bloc transformeur

Le transformeur {cite}`vaswani2017attention` construit une architecture profonde en empilant des blocs identiques. Chaque bloc combine l'attention multi-têtes avec un réseau à propagation avant (un petit MLP), le tout stabilisé par des connexions résiduelles et la normalisation de couche.

Un bloc transformeur applique les opérations suivantes:

$$
\begin{aligned}
\mathbf{z} &= \text{LayerNorm}(X + \text{MultiHead}(X)) \\
\text{sortie} &= \text{LayerNorm}(\mathbf{z} + \text{FFN}(\mathbf{z}))
\end{aligned}
$$

où FFN est un réseau à propagation avant appliqué indépendamment à chaque position:

$$
\text{FFN}(\mathbf{z}_t) = W_2\, \varphi(W_1\, \mathbf{z}_t + \mathbf{b}_1) + \mathbf{b}_2
$$

avec $W_1 \in \mathbb{R}^{d_{ff} \times d}$, $W_2 \in \mathbb{R}^{d \times d_{ff}}$, et $d_{ff}$ typiquement 4 fois $d$.

```{mermaid}
graph TB
    input["Entrée X ∈ ℝ<sup>T×d</sup>"]
    mha["Attention<br/>multi-têtes"]
    add1["+ (résiduelle)"]
    ln1["LayerNorm"]
    ffn["FFN<br/>(MLP 2 couches)"]
    add2["+ (résiduelle)"]
    ln2["LayerNorm"]
    output["Sortie ∈ ℝ<sup>T×d</sup>"]

    input --> mha
    mha --> add1
    input --> add1
    add1 --> ln1
    ln1 --> ffn
    ffn --> add2
    ln1 --> add2
    add2 --> ln2
    ln2 --> output

    style input fill:#dae8fc,stroke:#6c8ebf
    style output fill:#d5e8d4,stroke:#82b366
    style mha fill:#fff2cc,stroke:#d6b656
    style ffn fill:#fff2cc,stroke:#d6b656
    style add1 fill:#f5f5f5,stroke:#666666
    style add2 fill:#f5f5f5,stroke:#666666
    style ln1 fill:#f5f5f5,stroke:#666666
    style ln2 fill:#f5f5f5,stroke:#666666
```

Les connexions résiduelles ($X + \text{MultiHead}(X)$ au lieu de $\text{MultiHead}(X)$ seul) sont le même mécanisme que dans les réseaux résiduels (ResNets): elles permettent au gradient de circuler sans atténuation à travers les couches, ce qui rend possible l'entraînement de réseaux profonds.

La normalisation de couche (*layer normalization*) normalise les activations à chaque position indépendamment, ce qui stabilise l'entraînement. Elle remplace la normalisation par lots (*batch normalization*) vue au chapitre 8, car cette dernière est mal adaptée aux séquences de longueurs variables.

Un transformeur complet empile $N$ de ces blocs (typiquement $N = 6$ à $N = 96$ selon la taille du modèle). L'entrée du premier bloc est la séquence de représentations initiales (par exemple, les vecteurs d'un plongement de mots). La sortie du dernier bloc est la séquence de représentations contextualisées.

## Encodage positionnel

L'auto-attention est équivariante par permutation: elle produit le même résultat quelle que soit l'ordre des positions. Considérons les mots [mange, le, chat, souris, la] et [le, chat, mange, la, souris]. Sans encodage positionnel, l'auto-attention produit les mêmes représentations de sortie (à permutation des lignes près), car elle opère sur un ensemble de vecteurs, pas sur une séquence ordonnée. L'encodage positionnel brise cette symétrie pour que le transformeur distingue l'ordre des mots.

L'encodage positionnel ajoute un vecteur dépendant de la position à chaque représentation d'entrée:

$$
\tilde{\mathbf{x}}_t = \mathbf{x}_t + \mathbf{p}_t
$$

où $\mathbf{p}_t \in \mathbb{R}^d$ encode la position $t$. Le transformeur original {cite}`vaswani2017attention` utilise des fonctions sinusoïdales:

$$
\begin{aligned}
p_{t, 2i} &= \sin\!\left(\frac{t}{10000^{2i/d}}\right) \\
p_{t, 2i+1} &= \cos\!\left(\frac{t}{10000^{2i/d}}\right)
\end{aligned}
$$

Chaque dimension de l'encodage oscille à une fréquence différente. Les basses fréquences (grands indices $i$) varient lentement avec la position, encodant l'information de position grossière. Les hautes fréquences (petits indices $i$) varient rapidement, encodant la position fine. Ce choix permet au modèle d'apprendre des relations de position relatives, puisque $\mathbf{p}_{t+k}$ peut s'exprimer comme une transformation linéaire de $\mathbf{p}_t$ pour tout décalage $k$.

Une alternative courante est d'utiliser des encodages positionnels appris: chaque position $t$ a un vecteur $\mathbf{p}_t$ qui est un paramètre du modèle, optimisé pendant l'entraînement. Les deux approches fonctionnent bien en pratique.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'retina'

d = 64
T = 50
PE = np.zeros((T, d))
for t in range(T):
    for i in range(d // 2):
        freq = 1.0 / (10000 ** (2 * i / d))
        PE[t, 2*i] = np.sin(t * freq)
        PE[t, 2*i+1] = np.cos(t * freq)

fig, ax = plt.subplots(figsize=(8, 4))
im = ax.imshow(PE.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xlabel('Position $t$')
ax.set_ylabel('Dimension')
ax.set_title('Encodage positionnel sinusoïdal ($d = 64$)')
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
```

Chaque colonne de cette carte de chaleur représente l'encodage d'une position. Les dimensions basses (en haut) oscillent rapidement, les dimensions hautes (en bas) oscillent lentement. Deux positions proches ont des encodages similaires; deux positions éloignées ont des encodages distincts.

```{admonition} Lien avec l'encodage cyclique du projet
:class: note

Le projet sur la prédiction de consommation énergétique utilise $\sin(2\pi h / 24)$ et $\cos(2\pi h / 24)$ pour encoder l'heure du jour. Le principe est le même: représenter une quantité périodique par une paire sin/cos pour que des valeurs proches sur le cycle aient des représentations proches. L'heure 23 et l'heure 0 sont voisines en représentation sin/cos, alors qu'elles sont éloignées en valeur scalaire. L'encodage positionnel du transformeur généralise cette idée avec un spectre de fréquences: chaque paire de dimensions utilise une fréquence différente, ce qui permet d'encoder des relations de position à plusieurs échelles.
```

## Variantes d'architecture

Le mécanisme de base (auto-attention + FFN + résiduel + normalisation) s'assemble de différentes façons selon la tâche.

### Encodeur seul

L'encodeur traite la séquence d'entrée avec de l'auto-attention bidirectionnelle: chaque position peut consulter toutes les autres, y compris celles qui la suivent. La sortie est une séquence de représentations contextualisées.

```{mermaid}
graph LR
    input["Entrée + pos."]
    enc["Bloc × N<br/>(attention bidirectionnelle)"]
    repr["Représentations<br/>contextualisées"]

    input --> enc --> repr

    style input fill:#dae8fc,stroke:#6c8ebf
    style enc fill:#fff2cc,stroke:#d6b656
    style repr fill:#d5e8d4,stroke:#82b366
```

BERT {cite}`devlin2019bert` est l'exemple le plus connu. On l'utilise pour la classification de texte (en ajoutant une tête de classification sur la représentation du premier jeton), la recherche d'information, ou l'extraction de caractéristiques pour d'autres modèles.

### Décodeur seul

Le décodeur génère une séquence un élément à la fois, de gauche à droite. Pour que le modèle ne puisse pas "tricher" en regardant les mots futurs, l'auto-attention est masquée: la position $t$ ne peut consulter que les positions $1, \ldots, t$. Cela se fait en mettant à $-\infty$ les entrées correspondantes dans $Q K^\top$ avant le softmax.


GPT {cite}`radford2018improving` et les grands modèles de langage (LLM) utilisent cette architecture. Le modèle est entraîné à prédire le prochain mot à chaque position, et la génération se fait de façon autorégressive: on échantillonne un mot, on l'ajoute à la séquence, et on prédit le suivant.

### Encodeur-décodeur

L'architecture originale du transformeur {cite}`vaswani2017attention` combine un encodeur et un décodeur. L'encodeur traite la séquence source avec de l'auto-attention bidirectionnelle. Le décodeur génère la séquence cible avec de l'auto-attention causale, mais à chaque bloc, il inclut aussi une couche d'attention croisée: les requêtes viennent du décodeur, et les clés/valeurs viennent de l'encodeur. Cela permet au décodeur de consulter la séquence source à chaque étape de la génération.


Cette architecture est naturelle pour les tâches de transduction (traduction, résumé, réponse à une question), où l'entrée et la sortie sont des séquences de nature différente.

## Pourquoi les transformeurs dominent

Les transformeurs ont remplacé les RNN comme architecture dominante pour le traitement des séquences, et ils se sont étendus bien au-delà (vision, audio, protéines, etc.). Cette domination repose sur un avantage structurel lié au matériel moderne, mais elle s'accompagne d'un coût que les RNN n'avaient pas.

### Parallélisme et matériel moderne

L'auto-attention calcule $QK^\top$ comme un produit matriciel: exactement l'opération pour laquelle les GPU sont conçus. Un GPU contient des milliers de cœurs qui exécutent la même opération arithmétique sur des données différentes, et la multiplication matricielle exploite pleinement ce parallélisme.

Dans un RNN, le calcul de $\mathbf{h}_t$ attend $\mathbf{h}_{t-1}$: les $T$ pas de temps sont séquentiels. Sur un GPU avec des milliers de cœurs, la plupart restent inactifs pendant que le RNN traite un pas à la fois. L'attention, elle, traite toutes les paires de positions en une seule opération matricielle. Sur une séquence de 512 positions, un RNN exécute 512 pas séquentiels, tandis que l'attention calcule toutes les interactions en un seul produit $QK^\top$.

### Le coût quadratique de l'attention

Cet avantage a une contrepartie. La matrice $QK^\top$ contient $T^2$ entrées: une pour chaque paire de positions. Pour une séquence de $T = 4096$ positions avec 16 têtes et $d_k = 64$, la matrice d'attention d'une seule tête contient $4096^2 \approx 16{,}8$ millions d'entrées, soit environ 268 millions pour l'ensemble des têtes d'une seule couche. La mémoire et le temps de calcul croissent quadratiquement avec la longueur de la séquence.

C'est pourquoi les premiers transformeurs étaient limités à environ 512 jetons. L'adoption du transformeur n'est pas une victoire pure sur les RNN: on échange une profondeur séquentielle $O(T)$ contre un coût mémoire $O(T^2)$. Pour de courtes séquences, un RNN peut être plus économe. Pour de longues séquences, des variantes d'attention sous-quadratique (attention linéaire, attention éparse) tentent de retrouver le meilleur des deux régimes.

### Mise à l'échelle

Au-delà du parallélisme, les transformeurs possèdent une propriété empirique remarquable: leur performance s'améliore de façon prévisible quand on augmente le nombre de paramètres, la taille des données, et la quantité de calcul. Des lois de puissance (*scaling laws*) relient ces trois quantités à la perte du modèle. Cette prévisibilité permet de planifier les ressources nécessaires pour atteindre un niveau de performance donné.

Les chemins de gradient courts (une seule couche d'attention suffit pour connecter deux positions quelconques) facilitent l'entraînement de modèles très profonds. Combinée au parallélisme sur GPU, cette propriété a conduit aux grands modèles de langage (LLM) qui comptent des centaines de milliards de paramètres. La domination des transformeurs est donc indissociable du matériel sur lequel ils s'exécutent: sur des processeurs séquentiels, leur avantage sur les RNN serait bien moindre.

## Résumé

Le mécanisme d'attention permet à chaque position d'une séquence de consulter sélectivement toutes les autres positions, en calculant une moyenne pondérée des valeurs dont les poids dépendent de la similarité entre requêtes et clés. L'auto-attention applique ce mécanisme au sein d'une même séquence, et l'attention multi-têtes exécute plusieurs mécanismes en parallèle pour capturer différents types de relations.

Le transformeur empile des blocs composés d'attention multi-têtes et de réseaux à propagation avant, stabilisés par des connexions résiduelles et la normalisation de couche. L'encodage positionnel injecte la notion d'ordre, absente de l'auto-attention elle-même.

Les trois variantes principales (encodeur seul, décodeur seul, encodeur-décodeur) correspondent à des familles de tâches différentes. Le parallélisme, les connexions directes à longue portée, et les propriétés de mise à l'échelle expliquent la domination actuelle des transformeurs, au prix d'un coût quadratique en la longueur de la séquence.

```{admonition} Ce que vous devez retenir
:class: tip

1. L'attention calcule une moyenne pondérée des valeurs, avec des poids déterminés par la similarité entre requêtes et clés: $\text{Attention}(Q, K, V) = \text{softmax}(QK^\top / \sqrt{d_k})\, V$.

2. L'auto-attention permet à chaque position de consulter toutes les autres au sein de la même séquence. Les projections $W_Q$, $W_K$, $W_V$ sont apprises.

3. L'attention multi-têtes exécute $H$ mécanismes d'attention en parallèle, chacun pouvant capturer un type de relation différent.

4. Un bloc transformeur = attention multi-têtes + FFN + connexions résiduelles + normalisation de couche. Un transformeur complet empile $N$ blocs.

5. L'auto-attention est équivariante par permutation. L'encodage positionnel (sinusoïdal ou appris) restaure la notion d'ordre.

6. Encodeur (bidirectionnel, BERT), décodeur (causal, GPT), encodeur-décodeur (traduction): trois variantes d'une même architecture.

7. L'attention est une généralisation paramétrique de l'estimateur de Nadaraya-Watson: même structure de moyenne pondérée normalisée, mais avec des projections apprises pour la similarité et les valeurs.

8. Les transformeurs échangent la profondeur séquentielle $O(T)$ des RNN contre un coût mémoire $O(T^2)$. Leur domination repose sur le parallélisme massif des GPU et les propriétés de mise à l'échelle.
```

## Exercices

Les exercices ★ vérifient la compréhension de base. Les exercices ★★ demandent d'appliquer les concepts à des calculs concrets. Les exercices ★★★ approfondissent le sujet et sont optionnels pour IFT3395.

````{admonition} Exercice 1: Attention à la main ★
:class: hint dropdown

Soit trois positions avec les clés et valeurs suivantes (en dimension $d_k = 2$, $d_v = 1$):

| Position $t$ | $\mathbf{k}_t$ | $v_t$ |
|:---:|:---:|:---:|
| 1 | $(1, 0)$ | $10$ |
| 2 | $(0, 1)$ | $20$ |
| 3 | $(1, 1)$ | $30$ |

La requête est $\mathbf{q} = (1, 0)$.

1. Calculez les scores $\mathbf{q}^\top \mathbf{k}_t$ pour chaque position (sans normalisation par $\sqrt{d_k}$).
2. Appliquez le softmax pour obtenir les poids d'attention $\alpha_t$.
3. Calculez la sortie $\sum_t \alpha_t v_t$.
4. Quel est l'effet de la normalisation par $\sqrt{d_k}$?
````

````{admonition} Solution Exercice 1
:class: dropdown

1. Scores: $\mathbf{q}^\top \mathbf{k}_1 = 1$, $\mathbf{q}^\top \mathbf{k}_2 = 0$, $\mathbf{q}^\top \mathbf{k}_3 = 1$.

2. $\text{softmax}(1, 0, 1) = \left(\frac{e^1}{2e^1 + 1}, \frac{1}{2e^1 + 1}, \frac{e^1}{2e^1 + 1}\right) \approx (0{,}422,\; 0{,}155,\; 0{,}422)$.

3. Sortie: $0{,}422 \times 10 + 0{,}155 \times 20 + 0{,}422 \times 30 \approx 4{,}22 + 3{,}10 + 12{,}66 = 19{,}98 \approx 20$.

4. Avec $\sqrt{d_k} = \sqrt{2} \approx 1{,}41$, les scores deviennent $(0{,}71,\; 0,\; 0{,}71)$. Le softmax donne une distribution plus uniforme: $\approx (0{,}39,\; 0{,}22,\; 0{,}39)$. La normalisation "adoucit" l'attention en réduisant l'écart entre les scores, ce qui est particulièrement utile quand $d_k$ est grand.
````

````{admonition} Exercice 2: Nombre de paramètres d'un bloc transformeur ★★
:class: hint dropdown

Considérez un bloc transformeur avec $d = 512$, $H = 8$ têtes, et $d_{ff} = 2048$.

1. Combien de paramètres dans une tête d'attention ($W_Q^{(h)}$, $W_K^{(h)}$, $W_V^{(h)}$)?
2. Combien au total pour l'attention multi-têtes (incluant $W_O$)?
3. Combien dans le FFN ($W_1$, $W_2$, biais)?
4. Quel est le total pour un bloc? Et pour un transformeur à $N = 6$ blocs?

(Ignorez les paramètres de la normalisation de couche.)
````

````{admonition} Solution Exercice 2
:class: dropdown

Avec $d_k = d_v = d / H = 64$:

1. Par tête: $W_Q^{(h)} \in \mathbb{R}^{512 \times 64}$: 32 768. Idem pour $W_K^{(h)}$ et $W_V^{(h)}$. Total par tête: $3 \times 32\,768 = 98\,304$.

2. Huit têtes: $8 \times 98\,304 = 786\,432$. $W_O \in \mathbb{R}^{512 \times 512}$: 262 144. Total attention multi-têtes: 1 048 576 $\approx$ 1M paramètres.

3. $W_1 \in \mathbb{R}^{2048 \times 512}$: 1 048 576. $\mathbf{b}_1 \in \mathbb{R}^{2048}$: 2 048. $W_2 \in \mathbb{R}^{512 \times 2048}$: 1 048 576. $\mathbf{b}_2 \in \mathbb{R}^{512}$: 512. Total FFN: 2 099 712 $\approx$ 2M.

4. Total par bloc: $\approx$ 3,1M. Pour 6 blocs: $\approx$ 18,9M paramètres (sans compter les plongements d'entrée et de sortie).
````

````{admonition} Exercice 3: Masque causal ★★
:class: hint dropdown

L'attention causale empêche la position $t$ de consulter les positions $t+1, t+2, \ldots, T$.

1. Écrivez la matrice de masque $M \in \mathbb{R}^{T \times T}$ pour $T = 4$, où $M_{ij} = 0$ si la position $i$ peut consulter la position $j$, et $M_{ij} = -\infty$ sinon.
2. Montrez que $\text{softmax}((QK^\top + M) / \sqrt{d_k})$ donne bien des poids nuls pour les positions futures.
3. Pourquoi est-ce nécessaire pour un modèle de langage?
````

````{admonition} Solution Exercice 3
:class: dropdown

1. La matrice de masque est triangulaire inférieure:

$$
M = \begin{pmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{pmatrix}
$$

2. Quand $M_{ij} = -\infty$, le score $s_{ij} + M_{ij} = -\infty$, et $\exp(-\infty) = 0$. Le softmax donne un poids nul pour cette position: $\alpha_{ij} = 0$. Les positions futures sont ignorées.

3. Un modèle de langage prédit le prochain mot à chaque position. Pendant l'entraînement, toutes les positions sont traitées en parallèle. Sans masque, la position $t$ pourrait "voir" le mot qu'elle est censée prédire (position $t+1$), et le modèle n'apprendrait rien. Le masque causal garantit que la prédiction à la position $t$ ne dépend que des mots $1, \ldots, t$.
````

````{admonition} Exercice 4: Auto-attention et permutation ★★
:class: hint dropdown

Soit $\Pi$ une matrice de permutation ($\Pi$ réarrange les lignes de $X$).

1. Montrez que $\text{SelfAttention}(\Pi X) = \Pi\, \text{SelfAttention}(X)$. (L'auto-attention est équivariante par permutation.)
2. Qu'est-ce que cela implique pour un transformeur sans encodage positionnel?
3. Pourquoi le RNN n'a-t-il pas cette propriété?
````

````{admonition} Solution Exercice 4
:class: dropdown

1. Posons $X' = \Pi X$. Alors $Q' = X' W_Q = \Pi X W_Q = \Pi Q$, et de même $K' = \Pi K$, $V' = \Pi V$. Le score: $Q'K'^\top = (\Pi Q)(\Pi K)^\top = \Pi Q K^\top \Pi^\top$. Le softmax est appliqué par ligne, et $\Pi$ permute les lignes: $\text{softmax}(\Pi A \Pi^\top) = \Pi\, \text{softmax}(A)\, \Pi^\top$. La sortie: $\Pi\, \text{softmax}(A)\, \Pi^\top \Pi V = \Pi\, \text{softmax}(A)\, V$ (car $\Pi^\top \Pi = I$). Donc $\text{SelfAttention}(X') = \Pi\, \text{SelfAttention}(X)$.

2. Sans encodage positionnel, le transformeur traite "le chat mange la souris" et "la souris mange le chat" de façon identique (à permutation près). Il ne peut pas distinguer l'ordre des mots. L'encodage positionnel brise cette symétrie.

3. Le RNN traite les positions séquentiellement: $\mathbf{h}_t$ dépend de $\mathbf{h}_{t-1}$, qui dépend de $\mathbf{h}_{t-2}$, etc. La récurrence impose un ordre de traitement, ce qui rend le réseau sensible à l'ordre des entrées sans encodage positionnel explicite.
````

````{admonition} Exercice 5: Nadaraya-Watson et attention ★★
:class: hint dropdown

L'estimateur de Nadaraya-Watson avec un noyau gaussien prédit:

$$
\hat{y}(\mathbf{x}) = \sum_{i=1}^n \frac{\exp\!\left(-\frac{\|\mathbf{x} - \mathbf{x}_i\|^2}{2\lambda^2}\right)}{\sum_{j=1}^n \exp\!\left(-\frac{\|\mathbf{x} - \mathbf{x}_j\|^2}{2\lambda^2}\right)}\, y_i
$$

1. Montrez que cette formule a la même structure que le mécanisme d'attention: identifiez ce qui joue le rôle de la requête, des clés, des valeurs et du score de similarité.
2. Développez $\|\mathbf{x} - \mathbf{x}_i\|^2$ et montrez que, sous certaines simplifications, le score se réduit à un produit scalaire entre $\mathbf{x}$ et $\mathbf{x}_i$.
3. Qu'est-ce qu'il faudrait changer dans Nadaraya-Watson pour obtenir exactement l'attention par produit scalaire utilisée dans les transformeurs?
````

````{admonition} Solution Exercice 5
:class: dropdown

1. La formule de Nadaraya-Watson est une moyenne pondérée normalisée, exactement comme l'attention. Les correspondances sont:
   - La requête est le point $\mathbf{x}$ pour lequel on prédit.
   - Les clés sont les points d'entraînement $\mathbf{x}_i$.
   - Les valeurs sont les observations $y_i$.
   - Le score de similarité est $-\|\mathbf{x} - \mathbf{x}_i\|^2 / (2\lambda^2)$, qui est grand (proche de 0) quand $\mathbf{x}$ et $\mathbf{x}_i$ sont proches.
   - La normalisation par la somme des exponentielles est le softmax.

2. En développant: $\|\mathbf{x} - \mathbf{x}_i\|^2 = \|\mathbf{x}\|^2 - 2\mathbf{x}^\top \mathbf{x}_i + \|\mathbf{x}_i\|^2$. Le terme $\|\mathbf{x}\|^2$ est constant pour toutes les positions $i$ et disparaît après le softmax (il s'annule dans le rapport). Si les données sont normalisées de sorte que $\|\mathbf{x}_i\|^2$ est approximativement constant pour tout $i$, ce terme disparaît aussi. Le score se réduit alors à $\mathbf{x}^\top \mathbf{x}_i / \lambda^2$, un produit scalaire mis à l'échelle.

3. Deux modifications sont nécessaires:
   - Remplacer la similarité fixe (noyau gaussien) par des projections linéaires apprises: $\mathbf{q} = W_Q \mathbf{x}$, $\mathbf{k}_i = W_K \mathbf{x}_i$, avec un score $\mathbf{q}^\top \mathbf{k}_i / \sqrt{d_k}$.
   - Remplacer les valeurs brutes $y_i$ par des projections apprises: $\mathbf{v}_i = W_V \mathbf{x}_i$.

   Nadaraya-Watson utilise des similarités et des valeurs fixées par les données; l'attention apprend les trois projections ($W_Q$, $W_K$, $W_V$), ce qui lui permet d'adapter simultanément ce qu'elle cherche, comment elle compare, et ce qu'elle retourne.
````
