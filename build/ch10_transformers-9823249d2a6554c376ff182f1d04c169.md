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
- Comparer la génération autorégressive par RNN et par transformeur, et expliquer le rôle du cache clés-valeurs
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

:::{figure} _static/attention_mechanism.svg
:name: fig-attention-mechanism
:align: center
:width: 80%

Le mécanisme d'attention: la requête est comparée à chaque clé, les scores sont normalisés par softmax, et la sortie est la somme pondérée des valeurs.
:::

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

L'animation interactive ci-dessous permet de manipuler la requête $\mathbf{q}$ et d'observer comment les poids d'attention et la sortie changent en temps réel. Les boutons de préréglage illustrent les scénarios clés: sélection d'une position, attention uniforme, ou mélange intermédiaire. Le curseur de température $\tau$ généralise la normalisation par $\sqrt{d_k}$: une valeur faible produit une consultation quasi dure, une valeur élevée adoucit la distribution.

```{code-cell} python
:tags: [remove-input]

from IPython.display import HTML
from pathlib import Path
import html as _html

_content = Path("_static/attention_explorer.html").read_text()
_doc = f'<!DOCTYPE html><html><head><meta charset="utf-8"></head><body style="margin:0;padding:0">{_content}</body></html>'
HTML(f'<iframe srcdoc="{_html.escape(_doc, quote=True)}" width="100%" height="900" style="border:none;" scrolling="no"></iframe>')
```

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

Prenons un exemple concret. Dans la phrase « Le chat qui dormait sur le canapé s'est réveillé », le mot « réveillé » doit savoir que son sujet est « chat » pour construire une représentation utile. L'auto-attention peut capturer cette dépendance: les matrices $W_Q$ et $W_K$ projettent les représentations de « réveillé » et « chat » dans un espace où leur produit scalaire est élevé, ce qui donne un poids d'attention fort entre ces deux positions. Le vecteur de sortie pour « réveillé » incorpore alors l'information de « chat ». Les matrices de projection $W_Q$, $W_K$ et $W_V$ déterminent quel type de relation l'attention capture. Un seul jeu de matrices ne peut encoder qu'une seule notion de similarité entre positions. Nous verrons à la section suivante comment l'attention multi-têtes dépasse cette limitation.

Une propriété importante de l'auto-attention est qu'elle est équivariante par permutation: si l'on permute les lignes de $X$, les lignes de la sortie sont permutées de la même façon. L'auto-attention ne contient aucune notion d'ordre intrinsèque. Contrairement au RNN, où la position $t$ reçoit nécessairement l'information des positions $1, \ldots, t-1$ via $\mathbf{h}_{t-1}$, l'auto-attention traite toutes les positions de façon symétrique. Nous verrons plus loin comment l'encodage positionnel restaure la notion d'ordre.

```{code-cell} python
import numpy as np
from scipy.special import softmax

def self_attention(X, W_Q, W_K, W_V):
    d_k = W_Q.shape[1]
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    scores = Q @ K.T / d_k**0.5
    weights = softmax(scores, axis=-1)
    return weights @ V

rng = np.random.default_rng(0)
T, d, d_k = 4, 8, 3
X = rng.standard_normal((T, d))
W_Q = rng.standard_normal((d, d_k))
W_K = rng.standard_normal((d, d_k))
W_V = rng.standard_normal((d, d_k))

# Sortie sur l'entrée originale
Y = self_attention(X, W_Q, W_K, W_V)

# Permuter les lignes de X, puis calculer l'auto-attention
perm = [2, 0, 3, 1]
Y_perm = self_attention(X[perm], W_Q, W_K, W_V)

# Comparer: permuter la sortie vs calculer sur l'entrée permutée
print("Y permutée (Y[perm]):")
print(np.round(Y[perm], 4))
print("\nAuto-attention sur X permutée:")
print(np.round(Y_perm, 4))
print("\nIdentiques ?", np.allclose(Y[perm], Y_perm))
```

### Chemins directs entre positions

Dans un RNN, l'information de la position 1 doit traverser la chaîne $\mathbf{h}_1 \to \mathbf{h}_2 \to \cdots \to \mathbf{h}_T$ pour atteindre la position $T$. À chaque transition, le signal est transformé et potentiellement atténué. Comme nous l'avons vu au chapitre 9, cette chaîne est précisément ce qui cause la dissolution du gradient: le gradient subit $T - 1$ multiplications matricielles successives, et sa norme tend à décroître exponentiellement avec la longueur de la séquence.

:::{figure} _static/rnn_vs_attention.svg
:name: fig-rnn-vs-attention
:align: center
:width: 90%

À gauche, la chaîne séquentielle d'un RNN: l'information de $x_1$ doit traverser chaque état caché pour atteindre $h_4$. À droite, l'auto-attention connecte directement chaque position d'entrée à chaque position de sortie.
:::

L'auto-attention élimine ce goulot d'étranglement: chaque position de sortie est directement connectée à chaque position d'entrée en une seule couche, sans intermédiaire. Là où le RNN impose un chemin de longueur $T - 1$, l'auto-attention n'a besoin que d'un seul pas.

La conséquence pour l'entraînement est directe: un chemin court entre deux positions signifie un chemin court pour le gradient. Les connexions résiduelles du bloc transformeur (section suivante) amplifient cet effet en ajoutant un chemin d'identité à travers chaque couche.

## Attention multi-têtes

Comme nous l'avons vu, un seul jeu de matrices de projection ne capture qu'un seul type de relation. Or, dans « Le chat qui dormait sur le canapé s'est réveillé », le mot « dormait » entretient à la fois une relation syntaxique avec « chat » (son sujet) et une relation de proximité avec « canapé » (son complément de lieu). Un seul mécanisme d'attention doit choisir: il ne peut pas donner un poids élevé à « chat » et à « canapé » pour des raisons différentes avec une seule paire $W_Q, W_K$.

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

:::{figure} _static/multihead_attention.svg
:name: fig-multihead-attention
:align: center
:width: 90%

Attention multi-têtes: chaque tête apprend des projections distinctes et capture un type de relation différent. Ici, trois têtes sur la phrase « Le chat mange la souris » illustrent des spécialisations possibles.
:::

Chaque tête peut apprendre à capturer un type de relation différent: une tête peut se spécialiser dans les dépendances locales, une autre dans les dépendances à longue portée, une autre encore dans les relations syntaxiques.

## Lien avec la régression à noyau

L'attention par produit scalaire est un cas particulier d'une famille plus ancienne: la régression non paramétrique à noyau. L'estimateur de Nadaraya-Watson {cite}`nadaraya1964estimating,watson1964smooth` prédit la sortie pour un point $\mathbf{x}$ par une moyenne pondérée des observations:

$$
\hat{y}(\mathbf{x}) = \sum_{i=1}^n \frac{K_\lambda(\mathbf{x}, \mathbf{x}_i)}{\sum_{j=1}^n K_\lambda(\mathbf{x}, \mathbf{x}_j)}\, y_i
$$

Avec un noyau gaussien $K_\lambda(\mathbf{x}, \mathbf{x}_i) = \exp(-\|\mathbf{x} - \mathbf{x}_i\|^2 / 2\lambda^2)$, le lien avec l'attention est exact, pas seulement structurel. En développant $\|\mathbf{x} - \mathbf{x}_i\|^2 = \|\mathbf{x}\|^2 - 2\mathbf{x}^\top \mathbf{x}_i + \|\mathbf{x}_i\|^2$, le terme $\|\mathbf{x}\|^2$ est constant pour toutes les positions et disparaît dans le rapport du softmax. Si les clés ont des normes comparables, le score se réduit à $\mathbf{x}^\top \mathbf{x}_i / \lambda^2$: un produit scalaire mis à l'échelle, exactement la forme de l'attention avec $\sqrt{d_k}$ jouant le rôle de $\lambda$ (l'exercice 5 développe cette dérivation en détail).

La correspondance devient plus révélatrice quand on examine ce que l'attention apprend. Dans Nadaraya-Watson, le noyau est fixé a priori. Dans l'attention, les projections $W_Q$ et $W_K$ induisent un noyau effectif $\kappa(\mathbf{x}_i, \mathbf{x}_j) = \exp(\mathbf{x}_i^\top W_Q W_K^\top \mathbf{x}_j / \sqrt{d_k})$. La matrice $M = W_Q W_K^\top$ apprend quelles directions de l'espace d'entrée sont pertinentes pour la comparaison, de façon analogue à l'apprentissage de la forme et de la bande passante du noyau {cite}`tsai2019transformer`. Contrairement aux noyaux classiques en statistique, qui mesurent une similarité symétrique ($K(\mathbf{x}, \mathbf{x}') = K(\mathbf{x}', \mathbf{x})$), cette matrice est en général asymétrique ($M \neq M^\top$), ce qui permet à l'attention de traiter différemment la relation « sujet → verbe » et la relation « verbe → sujet » {cite}`wright2021transformers`.

La divergence la plus profonde concerne $W_V$. Dans Nadaraya-Watson, les « valeurs » $y_i$ sont les observations brutes: l'estimateur ne peut retourner que des moyennes pondérées des données. Si l'on pose $W_V = I$, l'attention se réduit exactement à Nadaraya-Watson. Mais une projection $W_V$ apprise transforme ce que chaque position offre avant l'agrégation. L'attention apprend simultanément comment pondérer (via $W_Q$, $W_K$) et quoi retourner (via $W_V$). L'attention multi-têtes pousse cette idée plus loin: $H$ estimateurs à noyau indépendants, chacun avec ses propres projections $(W_Q^{(h)}, W_K^{(h)}, W_V^{(h)})$, combinés linéairement par $W_O$ — un ensemble de régresseurs à noyau spécialisés.

Le lien avec la régression à noyau dépasse le cas gaussien. En régression non paramétrique, le noyau gaussien attribue un poids non nul à tous les points, même très éloignés. D'autres noyaux classiques — Epanechnikov, biweight, triweight — attribuent un poids nul au-delà d'une certaine distance (on dit qu'ils sont à support compact). Ils diffèrent par la douceur de la transition vers zéro: l'Epanechnikov est linéaire, le biweight quadratique, le triweight cubique. Santos et al. {cite}`santos2026sparse` montrent que remplacer le softmax par sparsemax dans le mécanisme d'attention produit exactement le noyau d'Epanechnikov, et que $\alpha$-entmax avec $\alpha = 1 + 1/n$ engendre toute cette hiérarchie: Epanechnikov ($n = 1$), biweight ($n = 2$), triweight ($n = 3$), gaussien ($n \to \infty$). Le choix de la fonction de normalisation dans l'attention est donc équivalent au choix du noyau en régression non paramétrique, chacun avec son propre compromis biais-variance.

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

:::{figure} _static/transformer_block.svg
:name: fig-transformer-block
:align: center
:width: 60%

Un bloc transformeur. Le chemin résiduel (flèche verticale) transporte le signal sans transformation, tandis que l'attention multi-têtes et le réseau à propagation avant modifient ce signal par addition.
:::

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

Chaque paire de dimensions $(2i, 2i+1)$ oscille à une fréquence différente, déterminée par $\omega_i = 1 / 10000^{2i/d}$. La première paire ($i = 0$) a une longueur d'onde de $2\pi \approx 6$ positions: elle change rapidement et distingue des positions voisines. La dernière paire ($i = d/2 - 1$) a une longueur d'onde d'environ $2\pi \times 10000 \approx 63\,000$ positions: elle varie à peine sur une séquence de quelques centaines de mots. Cette combinaison de fréquences fonctionne comme un système de numération: les basses fréquences encodent la position grossière (début, milieu ou fin de la séquence) tandis que les hautes fréquences encodent la position fine (la distinction entre positions adjacentes). Le modèle peut extraire des relations de position relatives par combinaison linéaire, puisque $\mathbf{p}_{t+k}$ s'exprime comme une transformation linéaire de $\mathbf{p}_t$ pour tout décalage $k$ fixé (chaque paire sin/cos se transforme par une rotation d'angle $k\omega_i$).

Une alternative courante est d'utiliser des encodages positionnels appris: chaque position $t$ a un vecteur $\mathbf{p}_t$ qui est un paramètre du modèle, optimisé pendant l'entraînement. Les deux approches fonctionnent bien en pratique.

Comment des vecteurs initialisés aléatoirement apprennent-ils à encoder la position? Pour le voir, considérons une expérience simple. On entraîne un unique bloc d'auto-attention sur la tâche « prédire le vecteur à la position précédente »: étant donné une séquence $\mathbf{x}_0, \ldots, \mathbf{x}_{T-1}$, la sortie à la position $t$ doit être $\mathbf{x}_{t-1}$. Cette tâche exige de savoir quelle position est « la précédente », ce qui est impossible sans information de position. Les encodages positionnels $\mathbf{p}_t$, initialisés aléatoirement, sont la seule source de cette information. Le gradient de la perte les organise pour que l'attention puisse distinguer les positions.

```{code-cell} python
:tags: [hide-input]

import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

rng = np.random.default_rng(42)
T, d = 16, 16
B = 32  # taille de lot

# Paramètres (initialisés aléatoirement)
P = rng.standard_normal((T, d)) * 0.05      # encodages positionnels à apprendre
Wq = rng.standard_normal((d, d)) * (2/d)**0.5
Wk = rng.standard_normal((d, d)) * (2/d)**0.5
Wv = rng.standard_normal((d, d)) * (2/d)**0.5
W_out = rng.standard_normal((d, d)) * (2/d)**0.5

# Attention positionnelle avant entraînement
Q0i = P @ Wq; K0i = P @ Wk
A_init = softmax(Q0i @ K0i.T / d**0.5, axis=-1)

# Adam pour chaque paramètre
params = [P, Wq, Wk, Wv, W_out]
m_s = [np.zeros_like(p) for p in params]
v_s = [np.zeros_like(p) for p in params]
eta, b1, b2, eps = 0.001, 0.9, 0.999, 1e-8

losses = []
for step in range(3000):
    x = rng.standard_normal((B, T, d)) * 0.5
    X = x + P  # ajouter les encodages positionnels

    # Cible: le vecteur à la position précédente
    target = np.zeros_like(x)
    target[:, 1:, :] = x[:, :-1, :]

    # Passe avant
    Q = X @ Wq; K = X @ Wk; V = X @ Wv
    scores = Q @ K.transpose(0, 2, 1) / d**0.5
    A = softmax(scores, axis=-1)
    Z = A @ V
    y = Z @ W_out
    losses.append(0.5 * np.mean((y - target)**2))

    # Passe arrière (rétropropagation manuelle à travers l'attention)
    dy = (y - target) / (B * T * d)
    dW_out = np.einsum('btk,btj->kj', Z, dy)
    dZ = dy @ W_out.T
    dA = dZ @ V.transpose(0, 2, 1)
    dV = A.transpose(0, 2, 1) @ dZ
    dS = A * (dA - (A * dA).sum(axis=-1, keepdims=True)) / d**0.5
    dQ = dS @ K; dK = dS.transpose(0, 2, 1) @ Q
    dX = dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T
    dP = dX.sum(axis=0)  # somme sur le lot
    dWq = np.einsum('bti,btj->ij', X, dQ)
    dWk = np.einsum('bti,btj->ij', X, dK)
    dWv = np.einsum('bti,btj->ij', X, dV)

    grads = [dP, dWq, dWk, dWv, dW_out]
    t_ = step + 1
    for i in range(5):
        m_s[i] = b1 * m_s[i] + (1 - b1) * grads[i]
        v_s[i] = b2 * v_s[i] + (1 - b2) * grads[i]**2
        mhat = m_s[i] / (1 - b1**t_)
        vhat = v_s[i] / (1 - b2**t_)
        params[i] -= eta * mhat / (np.sqrt(vhat) + eps)

# Attention positionnelle après entraînement (x = 0, position seule)
Q0 = P @ Wq; K0 = P @ Wk
A_final = softmax(Q0 @ K0.T / d**0.5, axis=-1)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

im0 = axes[0].imshow(A_init, cmap='Blues', vmin=0, vmax=0.3)
axes[0].set_title('Attention avant entraînement', fontsize=10)
axes[0].set_xlabel('Position source (clé)')
axes[0].set_ylabel('Position cible (requête)')
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(A_final, cmap='Blues', vmin=0, vmax=1)
axes[1].set_title('Attention après entraînement', fontsize=10)
axes[1].set_xlabel('Position source (clé)')
axes[1].set_ylabel('Position cible (requête)')
plt.colorbar(im1, ax=axes[1], shrink=0.8)

axes[2].semilogy(losses, 'C0-', lw=1.5)
axes[2].set_xlabel('Itération')
axes[2].set_ylabel('Perte (MSE)')
axes[2].set_title('Convergence', fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.suptitle("Apprentissage d'encodages positionnels sur la tâche « prédire le mot précédent »",
             fontsize=11)
plt.tight_layout()
```

Avant l'entraînement, l'attention est quasi uniforme: chaque position regarde toutes les autres de façon indiscriminée. Après 3000 itérations, une diagonale décalée apparaît: la position $t$ concentre presque tout son poids d'attention sur la position $t-1$. Les encodages positionnels, partis de vecteurs aléatoires, se sont organisés pour que le produit $\mathbf{q}_t^\top \mathbf{k}_{t-1}$ soit maximal — le modèle a appris de lui-même la notion de « position précédente ».

L'animation interactive ci-dessous permet d'explorer l'encodage positionnel sinusoïdal. La carte de chaleur montre l'encodage de chaque position: les dimensions basses oscillent rapidement (position fine), les dimensions hautes oscillent lentement (position grossière). Le visualiseur de vagues isole des paires sin/cos individuelles, et le graphique de similarité cosinus montre que deux positions proches ont des encodages similaires — une propriété approximativement invariante par translation.

```{code-cell} python
:tags: [remove-input]

from IPython.display import HTML
from pathlib import Path
import html as _html

_content = Path("_static/positional_encoding_explorer.html").read_text()
_doc = f'<!DOCTYPE html><html><head><meta charset="utf-8"></head><body style="margin:0;padding:0">{_content}</body></html>'
HTML(f'<iframe srcdoc="{_html.escape(_doc, quote=True)}" width="100%" height="700" style="border:none;" scrolling="no"></iframe>')
```

```{admonition} Lien avec l'encodage cyclique du projet
:class: note

Le projet sur la prédiction de consommation énergétique utilise $\sin(2\pi h / 24)$ et $\cos(2\pi h / 24)$ pour encoder l'heure du jour. Le principe est le même: représenter une quantité périodique par une paire sin/cos pour que des valeurs proches sur le cycle aient des représentations proches. L'heure 23 et l'heure 0 sont voisines en représentation sin/cos, alors qu'elles sont éloignées en valeur scalaire. L'encodage positionnel du transformeur généralise cette idée avec un spectre de fréquences: chaque paire de dimensions utilise une fréquence différente, ce qui permet d'encoder des relations de position à plusieurs échelles.
```

## Variantes d'architecture

Le mécanisme de base (auto-attention + FFN + résiduel + normalisation) s'assemble de différentes façons selon la tâche. La {numref}`fig-architecture-variants` illustre la structure interne de chaque variante.

:::{figure} _static/architecture_variants.svg
:name: fig-architecture-variants
:align: center
:width: 100%

Les trois variantes d'architecture du transformeur. L'encodeur utilise l'auto-attention bidirectionnelle; le décodeur utilise l'auto-attention causale (masquée); l'encodeur-décodeur combine les deux avec une couche d'attention croisée.
:::

### Encodeur seul

L'encodeur traite la séquence d'entrée avec de l'auto-attention bidirectionnelle: chaque position peut consulter toutes les autres, y compris celles qui la suivent. La sortie est une séquence de représentations contextualisées.

BERT {cite}`devlin2019bert` est l'exemple le plus connu. On l'utilise pour la classification de texte (en ajoutant une tête de classification sur la représentation du premier jeton), la recherche d'information, ou l'extraction de caractéristiques pour d'autres modèles.

### Décodeur seul

Le décodeur génère une séquence un élément à la fois, de gauche à droite. Un modèle de langage est entraîné à prédire le prochain mot à chaque position: la cible à la position $t$ est le mot $t + 1$. Pendant l'entraînement, toutes les positions sont traitées en parallèle par l'auto-attention, ce qui signifie que la position $t$ a accès à toute la séquence, y compris les mots futurs qu'elle est censée prédire. Sans contrainte supplémentaire, le modèle pourrait simplement copier le mot suivant au lieu d'apprendre à le prédire.

Le masque causal résout ce problème en restreignant l'attention: la position $t$ ne peut consulter que les positions $1, \ldots, t$. Concrètement, on définit une matrice de masque $M \in \mathbb{R}^{T \times T}$ dont les entrées valent:

$$
M_{ij} = \begin{cases} 0 & \text{si } i \geq j \\ -\infty & \text{si } i < j \end{cases}
$$

Cette matrice est triangulaire inférieure (avec des zéros sur la diagonale et en dessous, $-\infty$ au-dessus). Pour $T = 4$:

$$
M = \begin{pmatrix} 0 & -\infty & -\infty & -\infty \\ 0 & 0 & -\infty & -\infty \\ 0 & 0 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{pmatrix}
$$

On ajoute ce masque aux scores d'attention avant le softmax:

$$
\text{CausalAttention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top + M}{\sqrt{d_k}}\right) V
$$

Le mécanisme repose sur le comportement du softmax face à $-\infty$: lorsque $M_{ij} = -\infty$, le score $s_{ij} + M_{ij} = -\infty$, et $\exp(-\infty) = 0$. Le poids d'attention $\alpha_{ij}$ est donc nul, et la position $i$ ignore complètement la position $j$. Le softmax renormalise les poids restants pour qu'ils somment à 1, de sorte que chaque position ne combine que les valeurs des positions passées et présente.

GPT {cite}`radford2018improving` et les grands modèles de langage (LLM) utilisent cette architecture. La génération se fait de façon autorégressive: on échantillonne un mot, on l'ajoute à la séquence, et on prédit le suivant.

### Encodeur-décodeur

L'architecture originale du transformeur {cite}`vaswani2017attention` combine un encodeur et un décodeur. L'encodeur traite la séquence source avec de l'auto-attention bidirectionnelle. Le décodeur génère la séquence cible avec de l'auto-attention causale, mais à chaque bloc, il inclut aussi une couche d'attention croisée: les requêtes viennent du décodeur, et les clés/valeurs viennent de l'encodeur. Cela permet au décodeur de consulter la séquence source à chaque étape de la génération.

Cette architecture est naturelle pour les tâches de transduction (traduction, résumé, réponse à une question), où l'entrée et la sortie sont des séquences de nature différente.

## Entraînement et génération autorégressive

La section précédente a décrit la structure du décodeur et le masque causal, mais elle n'a pas détaillé comment on entraîne un modèle de langage ni comment on génère du texte avec. Cette section rend ces deux procédures explicites et les compare à leur équivalent pour les réseaux récurrents. Nous commençons par préciser ce qu'est un jeton.

### Des mots aux jetons

Jusqu'ici, nous avons parlé de « mots » et de « positions » de manière informelle. En pratique, un modèle de langage ne travaille pas directement sur des mots: il opère sur des **jetons** (*tokens*), des unités de texte issues d'un découpage appelé **tokénisation** (*tokenization*).

Pourquoi ne pas utiliser les mots directement? Le vocabulaire d'une langue est immense et ouvert: noms propres, termes techniques, néologismes, fautes de frappe. Un vocabulaire de mots entiers contiendrait des centaines de milliers d'entrées, et tout mot absent du vocabulaire serait inutilisable. À l'autre extrême, découper en caractères individuels résout le problème du vocabulaire (l'alphabet est fini et petit), mais produit des séquences très longues — une phrase de 20 mots devient une séquence de 100 caractères — et chaque caractère isolé porte peu d'information sémantique.

La tokénisation par sous-mots (*subword tokenization*) offre un compromis. L'idée, popularisée par l'algorithme BPE (*Byte Pair Encoding*) {cite}`sennrich2016neural`, est de construire un vocabulaire de taille fixe (typiquement 30 000 à 100 000 entrées) en fusionnant progressivement les paires de caractères les plus fréquentes dans un corpus. Les mots courants sont représentés par un seul jeton (« le », « pour », « transformer »), tandis que les mots rares sont découpés en sous-mots (« anticonstitutionnellement » → « anti », « constitu », « tion », « nellement »). Ce mécanisme garantit que tout texte peut être découpé en jetons du vocabulaire, sans mot inconnu.

Chaque jeton du vocabulaire est associé à un indice entier. La première couche du modèle est une table de représentations vectorielles (*embedding table*) $E \in \mathbb{R}^{|\mathcal{V}| \times d}$, où $|\mathcal{V}|$ est la taille du vocabulaire et $d$ la dimension des représentations. Le jeton d'indice $i$ est converti en vecteur $\mathbf{e}_i = E[i, :] \in \mathbb{R}^d$. Ce sont ces vecteurs qui entrent dans le transformeur comme la séquence $(\mathbf{x}_1, \ldots, \mathbf{x}_T)$, et les paramètres de $E$ sont appris conjointement avec le reste du modèle.

La dernière couche d'un modèle de langage fait l'opération inverse: elle projette la représentation de sortie sur le vocabulaire pour obtenir un score par jeton. Souvent, cette projection réutilise la même matrice $E$ (transposée), ce qui réduit le nombre de paramètres.

### Génération mot par mot

Générer du texte avec un modèle de langage consiste à produire une séquence de jetons un par un. À chaque étape $t$, le modèle calcule une distribution sur le vocabulaire, conditionnée sur les jetons déjà produits $(x_1, \ldots, x_t)$, puis échantillonne le jeton suivant $x_{t+1}$ à partir de cette distribution. Ce jeton est ajouté à la séquence, et le processus se répète.

Avec un RNN, cette boucle repose sur l'état caché $\mathbf{h}_t$, qui résume tout le contexte en un vecteur de taille fixe.

```{prf:algorithm} Génération autorégressive avec un RNN
:label: ch10-rnn-generation

**Entrée**: Jeton initial $x_1$, modèle RNN

1. Initialiser $\mathbf{h}_0 = \mathbf{0}$
2. Pour $t = 1, 2, \ldots$:
   - $\mathbf{h}_t = \varphi(W_{hh}\, \mathbf{h}_{t-1} + W_{xh}\, \mathbf{e}(x_t) + \mathbf{b}_h)$
   - $p_t = \text{softmax}(W_{hy}\, \mathbf{h}_t + \mathbf{b}_y)$
   - Échantillonner $x_{t+1} \sim \text{Catégorielle}(p_t)$
   - Si $x_{t+1} = \langle\text{fin}\rangle$, arrêter
3. Retourner $(x_1, x_2, \ldots)$
```

Le coût de chaque étape est $O(m^2 + md)$, où $m$ est la dimension de l'état caché et $d$ celle des représentations vectorielles. Ce coût est constant quel que soit le nombre de jetons déjà générés, car tout le contexte est comprimé dans $\mathbf{h}_t$.

Avec un transformeur décodeur, il n'y a pas d'état caché récurrent. À chaque étape, le modèle reçoit la séquence complète $(x_1, \ldots, x_t)$ et calcule l'auto-attention sur toutes ces positions.

```{prf:algorithm} Génération autorégressive avec un transformeur (version naïve)
:label: ch10-transformer-generation

**Entrée**: Jeton initial $x_1$, transformeur décodeur à $N$ blocs

1. Initialiser $S \leftarrow (x_1)$
2. Pour $t = 1, 2, \ldots$:
   - Empiler les représentations: $X \leftarrow (\mathbf{e}(x_1), \ldots, \mathbf{e}(x_t)) \in \mathbb{R}^{t \times d}$
   - Pour chaque bloc $\ell = 1, \ldots, N$:
     - $Q, K, V \leftarrow XW_Q^{(\ell)},\; XW_K^{(\ell)},\; XW_V^{(\ell)}$ chacune dans $\mathbb{R}^{t \times d_k}$
     - $A \leftarrow \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$ où $QK^\top \in \mathbb{R}^{t \times t}$
     - $X \leftarrow \text{MLP}(X + A)$ avec normalisation et connexion résiduelle
   - $p_t \leftarrow \text{softmax}(W_{\text{sortie}}\, \mathbf{x}_t^{(N)} + \mathbf{b})$ où $\mathbf{x}_t^{(N)}$ est la dernière ligne de $X$
   - Échantillonner $x_{t+1} \sim \text{Catégorielle}(p_t)$
   - Si $x_{t+1} = \langle\text{fin}\rangle$, arrêter
   - $S \leftarrow S \,\|\, (x_{t+1})$
3. Retourner $S$
```

À l'étape $t$, l'attention calcule le produit $QK^\top \in \mathbb{R}^{t \times t}$: le coût est $O(t^2 d)$ par couche. En sommant sur les $T$ étapes de génération, le coût total est $O\!\left(\sum_{t=1}^T t^2 d\right) = O(T^3 d)$, cubique en la longueur de la séquence. La sous-section sur le cache clés-valeurs montrera comment ramener ce coût à $O(T^2 d)$.

À l'inférence, les deux architectures sont donc séquentielles: on ne peut pas produire le jeton $t+1$ avant d'avoir choisi le jeton $t$. La différence est le coût par étape. Pour le RNN, ce coût est constant car l'état $\mathbf{h}_t$ a une taille fixe. Pour le transformeur, il croît avec la longueur de la séquence car l'attention revisite tout le contexte à chaque pas.

### Entraînement parallèle

Pendant l'entraînement, la séquence cible $(w_1, w_2, \ldots, w_T)$ est connue entièrement. L'objectif est de maximiser la vraisemblance du prochain jeton à chaque position. La perte est la somme des entropies croisées:

$$
\mathcal{L} = \sum_{t=1}^{T-1} \bigl[-\log p_\theta(w_{t+1} \mid w_1, \ldots, w_t)\bigr]
$$ (eq:ntp-loss)

Pour un RNN, on utilise le *teacher forcing*: à chaque pas $t$, on fournit comme entrée le vrai jeton $w_t$ du corpus (et non la prédiction du modèle). La mise à jour de l'état caché devient $\mathbf{h}_t = \varphi(W_{hh}\, \mathbf{h}_{t-1} + W_{xh}\, \mathbf{e}(w_t) + \mathbf{b}_h)$, où $\mathbf{e}(w_t)$ est la représentation vectorielle du jeton de référence. Cela stabilise l'entraînement en évitant que les erreurs du modèle s'accumulent d'un pas à l'autre. Mais le calcul reste séquentiel: $\mathbf{h}_t$ dépend de $\mathbf{h}_{t-1}$, et les $T$ pas de temps sont traités dans l'ordre.

Pour un transformeur, le masque causal rend le teacher forcing inutile, car il accomplit le même effet de manière parallèle. On fournit la séquence complète $(w_1, \ldots, w_T)$ en une seule passe avant. Le masque garantit que la position $t$ ne consulte que $(w_1, \ldots, w_t)$, exactement comme si on avait appliqué le teacher forcing position par position. Les $T$ prédictions sont obtenues simultanément, et la perte {eq}`eq:ntp-loss` est calculée sur toutes les positions en un seul passage. Le calcul est massivement parallèle: c'est un produit matriciel $QK^\top$ sur toute la séquence, l'opération pour laquelle les GPU sont conçus.

L'asymétrie entre entraînement et inférence est donc la suivante. Pendant l'entraînement, tous les jetons cibles sont connus à l'avance, ce qui permet de traiter toutes les positions en parallèle (transformeur) ou au moins d'utiliser les vrais jetons comme entrée (RNN avec teacher forcing). Pendant la génération, chaque jeton dépend du précédent, et le calcul est nécessairement séquentiel pour les deux architectures. Cette asymétrie a une conséquence: pendant l'entraînement, le modèle ne voit jamais ses propres erreurs, car il reçoit toujours les vrais jetons. Pendant la génération, une erreur à l'étape $t$ affecte toutes les étapes suivantes. Cet écart entre les conditions d'entraînement et de génération est appelé biais d'exposition (*exposure bias*).

### Le cache clés-valeurs

L'algorithme de génération naïf ({prf:ref}`ch10-transformer-generation`) recalcule l'attention sur toute la séquence à chaque nouvelle étape. Or, à l'étape $t$, les clés et valeurs des positions $1, \ldots, t-1$ ont déjà été calculées à l'étape $t-1$: seule la position $t$ est nouvelle.

Le cache clés-valeurs (*KV cache*) exploite cette observation. À chaque couche et chaque tête, on conserve en mémoire les matrices de clés et de valeurs des positions passées. À l'étape $t$, on calcule uniquement la requête, la clé et la valeur pour la nouvelle position $t$:

$$
\mathbf{q}_t = \mathbf{x}_t W_Q, \quad \mathbf{k}_t = \mathbf{x}_t W_K, \quad \mathbf{v}_t = \mathbf{x}_t W_V
$$

On concatène $\mathbf{k}_t$ et $\mathbf{v}_t$ au cache:

$$
K_t = \begin{pmatrix} K_{t-1} \\ \mathbf{k}_t \end{pmatrix}, \quad
V_t = \begin{pmatrix} V_{t-1} \\ \mathbf{v}_t \end{pmatrix}
$$

L'attention pour la position $t$ est alors un produit vecteur-matrice, pas un produit matrice-matrice:

$$
\text{attention}_t = \text{softmax}\!\left(\frac{\mathbf{q}_t\, K_t^\top}{\sqrt{d_k}}\right) V_t
$$

Le coût de l'étape $t$ passe de $O(t^2 d)$ à $O(td)$: un produit de taille $1 \times t$ au lieu de $t \times t$. Sur $T$ étapes, le coût total est $O\!\left(\sum_{t=1}^T td\right) = O(T^2 d)$ au lieu de $O(T^3 d)$. La contrepartie est la mémoire: on stocke $K_t$ et $V_t$ pour chaque couche et chaque tête, soit une quantité de mémoire qui croît linéairement avec la longueur de la séquence.

Avec le cache, le transformeur à l'inférence ressemble au RNN en ce qu'il étend un état à chaque pas. Mais cet état (le cache) grandit avec $t$, alors que l'état caché $\mathbf{h}_t$ du RNN a une taille fixe $m$. C'est le même compromis que celui discuté au chapitre précédent: le RNN comprime toute l'information dans un vecteur de taille fixe, au risque du goulot d'information; le transformeur conserve tout le contexte, au prix d'une mémoire croissante.

### Démonstration: génération avec GPT-2

Mettons en pratique la boucle de génération autorégressive avec GPT-2, un transformeur décodeur pré-entraîné sur un large corpus de texte anglais. Le code ci-dessous implémente la boucle décrite dans l'algorithme {prf:ref}`ch10-transformer-generation`, sans recourir à une fonction de génération toute faite: on calcule les logits, on applique le softmax avec un paramètre de température, on échantillonne, et on ajoute le jeton au contexte.

```{code-cell} python
:tags: [hide-input]
import logging, warnings, os
logging.disable(logging.INFO)
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def generer(prompt, max_jetons=40, temperature=1.0, graine=42):
    """Génération autorégressive, jeton par jeton."""
    torch.manual_seed(graine)
    ids = tokenizer.encode(prompt, return_tensors="pt")
    generes = []

    for _ in range(max_jetons):
        with torch.no_grad():
            sorties = model(ids)
        # Logits du dernier jeton
        logits = sorties.logits[0, -1, :] / temperature
        probs = torch.softmax(logits, dim=0)
        # Échantillonner le prochain jeton
        jeton = torch.multinomial(probs, num_samples=1)
        generes.append(jeton.item())
        ids = torch.cat([ids, jeton.unsqueeze(0)], dim=1)
        # Arrêter au jeton de fin
        if jeton.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generes)

prompt = "The meaning of life is"
print(f"Invite : {prompt!r}\n")
print(f"Texte généré :\n{prompt}{generer(prompt)}")
```

À chaque étape, le modèle reçoit la séquence complète (invite + jetons déjà générés), calcule les logits sur le vocabulaire, et échantillonne le jeton suivant. La boucle est séquentielle: on ne peut pas choisir le jeton $t+1$ avant d'avoir produit le jeton $t$.

On peut extraire les poids d'attention de GPT-2 pour observer le masque causal. La matrice d'attention d'une tête a pour entrée $(i, j)$ le poids $\alpha_{ij}$ que la position $i$ accorde à la position $j$. Le masque causal force $\alpha_{ij} = 0$ pour tout $j > i$: chaque position ne consulte que les positions précédentes et elle-même. La matrice résultante est triangulaire inférieure.

```{code-cell} python
:tags: [hide-input]
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt

# Charger le modèle avec l'implémentation d'attention qui expose les poids
model_attn = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
model_attn.eval()

ids = tokenizer.encode(prompt, return_tensors="pt")
with torch.no_grad():
    sorties = model_attn(ids, output_attentions=True)

jetons = [tokenizer.decode(i) for i in ids[0]]
T = len(jetons)

# Couche 0, 4 premières têtes
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for h, ax in enumerate(axes):
    attn = sorties.attentions[0][0, h, :T, :T].numpy()
    im = ax.imshow(attn, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(T))
    ax.set_xticklabels(jetons, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(T))
    ax.set_yticklabels(jetons, fontsize=8)
    ax.set_title(f'Tête {h}', fontsize=10)
    ax.set_xlabel('Clé (position consultée)')
    if h == 0:
        ax.set_ylabel('Requête (position courante)')

fig.suptitle(f'Poids d\'attention de la première couche de GPT-2 — masque causal',
             fontsize=11, y=1.06)
plt.tight_layout()
```

La structure triangulaire est visible dans chaque tête: les entrées au-dessus de la diagonale sont nulles. Au-delà de cette contrainte commune, chaque tête apprend un motif d'attention différent. Certaines têtes concentrent leur attention sur le jeton immédiatement précédent; d'autres répartissent l'attention plus uniformément sur tout le contexte disponible.

Le paramètre de **température** contrôle la forme de la distribution. Une température basse ($\tau \ll 1$) concentre la masse de probabilité sur les jetons les plus probables, rendant la génération quasi déterministe. Une température élevée ($\tau > 1$) aplatit la distribution et augmente la diversité, au prix de la cohérence.

```{code-cell} python
:tags: [hide-input]
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt

# Montrer la distribution sur le vocabulaire pour le dernier jeton de l'invite
ids = tokenizer.encode(prompt, return_tensors="pt")
with torch.no_grad():
    logits = model(ids).logits[0, -1, :]

temperatures = [0.3, 1.0, 2.0]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

for ax, tau in zip(axes, temperatures):
    probs = torch.softmax(logits / tau, dim=0)
    top_probs, top_ids = probs.topk(10)
    top_mots = [tokenizer.decode(i).strip() for i in top_ids]

    bars = ax.barh(range(10), top_probs.numpy(), color='#2196F3')
    ax.set_yticks(range(10))
    ax.set_yticklabels(top_mots, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Probabilité')
    ax.set_title(f'$\\tau = {tau}$')
    ax.set_xlim(0, min(1.0, top_probs[0].item() * 1.3))

plt.suptitle(f'Distribution sur les 10 jetons les plus probables après "{prompt}"',
             fontsize=11, y=1.04)
plt.tight_layout()
```

À température $\tau = 0{,}3$, le jeton le plus probable concentre l'essentiel de la masse: la génération suit un chemin prévisible. À $\tau = 1$ (la valeur par défaut), la distribution est celle apprise par le modèle. À $\tau = 2$, les jetons moins probables reçoivent une part non négligeable, ce qui produit un texte plus varié mais moins cohérent. La température ne modifie pas les logits: elle change uniquement la forme du softmax, $p_i = \exp(z_i / \tau) / \sum_j \exp(z_j / \tau)$.

```{code-cell} python
:tags: [hide-input]
print(f"Invite : {prompt!r}\n")
for tau in [0.3, 1.0, 2.0]:
    texte = generer(prompt, temperature=tau)
    print(f"τ = {tau:.1f} : {prompt}{texte}")
```

## Pourquoi les transformeurs dominent

Les transformeurs ont remplacé les RNN comme architecture dominante pour le traitement des séquences, et ils se sont étendus bien au-delà (vision, audio, protéines, etc.). Cette domination repose sur un avantage structurel lié au matériel moderne, mais elle s'accompagne d'un coût que les RNN n'avaient pas.

### Parallélisme et matériel moderne

L'auto-attention calcule $QK^\top$ comme un produit matriciel: exactement l'opération pour laquelle les GPU sont conçus. Un GPU contient des milliers de cœurs qui exécutent la même opération arithmétique sur des données différentes, et la multiplication matricielle exploite pleinement ce parallélisme.

Dans un RNN, le calcul de $\mathbf{h}_t$ attend $\mathbf{h}_{t-1}$: les $T$ pas de temps sont séquentiels. Sur un GPU avec des milliers de cœurs, la plupart restent inactifs pendant que le RNN traite un pas à la fois. L'attention, elle, traite toutes les paires de positions en une seule opération matricielle. Sur une séquence de 512 positions, un RNN exécute 512 pas séquentiels, tandis que l'attention calcule toutes les interactions en un seul produit $QK^\top$.

### Le coût quadratique de l'attention

Cet avantage a une contrepartie. La complexité de calcul de l'auto-attention est $O(T^2 d)$: la matrice $QK^\top$ contient $T^2$ entrées, une pour chaque paire de positions. Pour une séquence de $T = 4096$ positions avec 16 têtes et $d_k = 64$, la matrice d'attention d'une seule tête contient $4096^2 \approx 16{,}8$ millions d'entrées, soit environ 268 millions pour l'ensemble des têtes d'une seule couche. La mémoire et le temps de calcul croissent quadratiquement avec la longueur de la séquence.

C'est pourquoi les premiers transformeurs étaient limités à environ 512 jetons. L'adoption du transformeur n'est pas une victoire pure sur les RNN: on échange une profondeur séquentielle $O(T)$ contre un coût mémoire $O(T^2)$. Pour de courtes séquences, un RNN peut être plus économe. Pour de longues séquences, des variantes d'attention sous-quadratique (attention linéaire, attention éparse) tentent de retrouver le meilleur des deux régimes.

### Mise à l'échelle

Au-delà du parallélisme, les transformeurs possèdent une propriété empirique remarquable: leur performance s'améliore de façon prévisible quand on augmente le nombre de paramètres, la taille des données, et la quantité de calcul. Des lois de puissance (*scaling laws*) relient ces trois quantités à la perte du modèle. Cette prévisibilité permet de planifier les ressources nécessaires pour atteindre un niveau de performance donné.

Les chemins de gradient courts (une seule couche d'attention suffit pour connecter deux positions quelconques) facilitent l'entraînement de modèles très profonds. Combinée au parallélisme sur GPU, cette propriété a conduit aux grands modèles de langage (LLM) qui comptent des centaines de milliards de paramètres. La domination des transformeurs est donc indissociable du matériel sur lequel ils s'exécutent: sur des processeurs séquentiels, leur avantage sur les RNN serait bien moindre.

## Résumé

Le mécanisme d'attention permet à chaque position d'une séquence de consulter sélectivement toutes les autres positions, en calculant une moyenne pondérée des valeurs dont les poids dépendent de la similarité entre requêtes et clés. L'auto-attention applique ce mécanisme au sein d'une même séquence, et l'attention multi-têtes exécute plusieurs mécanismes en parallèle pour capturer différents types de relations.

Le transformeur empile des blocs composés d'attention multi-têtes et de réseaux à propagation avant, stabilisés par des connexions résiduelles et la normalisation de couche. L'encodage positionnel injecte la notion d'ordre, absente de l'auto-attention elle-même.

Les trois variantes principales (encodeur seul, décodeur seul, encodeur-décodeur) correspondent à des familles de tâches différentes. Le parallélisme, les connexions directes à longue portée, et les propriétés de mise à l'échelle expliquent la domination actuelle des transformeurs, au prix d'un coût quadratique en la longueur de la séquence. Pendant l'entraînement, le masque causal permet de calculer toutes les prédictions en parallèle; pendant la génération, les jetons sont produits un par un, et le cache clés-valeurs évite de recalculer l'attention sur tout le contexte à chaque pas.

```{admonition} Ce que vous devez retenir
:class: tip

1. L'attention calcule une moyenne pondérée des valeurs, avec des poids déterminés par la similarité entre requêtes et clés: $\text{Attention}(Q, K, V) = \text{softmax}(QK^\top / \sqrt{d_k})\, V$.

2. L'auto-attention permet à chaque position de consulter toutes les autres au sein de la même séquence. Les projections $W_Q$, $W_K$, $W_V$ sont apprises.

3. L'attention multi-têtes exécute $H$ mécanismes d'attention en parallèle, chacun pouvant capturer un type de relation différent.

4. Un bloc transformeur = attention multi-têtes + FFN + connexions résiduelles + normalisation de couche. Un transformeur complet empile $N$ blocs.

5. L'auto-attention est équivariante par permutation. L'encodage positionnel (sinusoïdal ou appris) restaure la notion d'ordre.

6. Encodeur (bidirectionnel, BERT), décodeur (causal, GPT), encodeur-décodeur (traduction): trois variantes d'une même architecture.

7. L'attention par produit scalaire est une généralisation paramétrique de Nadaraya-Watson: les projections $W_Q$, $W_K$ apprennent la forme du noyau, $W_V$ apprend une transformation des valeurs absente de l'estimateur classique, et le choix de la fonction de normalisation (softmax, sparsemax) détermine la famille de noyaux.

8. Les transformeurs échangent la profondeur séquentielle $O(T)$ des RNN contre un coût mémoire $O(T^2)$. Leur domination repose sur le parallélisme massif des GPU et les propriétés de mise à l'échelle.

9. Pendant l'entraînement, le masque causal permet de calculer les $T$ prédictions en parallèle (toutes les positions dans une seule passe avant). Pendant la génération, les jetons sont produits un par un. Le cache clés-valeurs réduit le coût par étape de $O(t^2 d)$ à $O(td)$.
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

````{admonition} Exercice 6: Coût de la génération autorégressive ★★
:class: hint dropdown

On génère $T$ jetons avec un transformeur décodeur à un seul bloc, de dimension $d$, avec une seule tête d'attention.

1. Sans cache clés-valeurs, quel est le coût du produit $QK^\top$ à l'étape $t$ (quand la séquence contient $t$ jetons)? Quel est le coût total pour générer $T$ jetons?
2. Avec le cache clés-valeurs, quel est le coût de l'attention à l'étape $t$? Le coût total?
3. Un RNN avec un état caché de dimension $m$ génère $T$ jetons. Quel est le coût total? Comparez avec les résultats précédents.
````

````{admonition} Solution Exercice 6
:class: dropdown

1. À l'étape $t$, les matrices $Q$ et $K$ ont $t$ lignes et $d$ colonnes. Le produit $QK^\top \in \mathbb{R}^{t \times t}$ coûte $O(t^2 d)$. Le coût total est $\sum_{t=1}^T O(t^2 d) = O(T^3 d / 3) = O(T^3 d)$.

2. Avec le cache, à l'étape $t$, on calcule un seul vecteur requête $\mathbf{q}_t \in \mathbb{R}^d$ et on le multiplie par $K_t^\top \in \mathbb{R}^{d \times t}$. Le coût est $O(td)$. Le coût total est $\sum_{t=1}^T O(td) = O(T^2 d / 2) = O(T^2 d)$.

3. Pour le RNN, chaque étape coûte $O(m^2 + md)$ (mise à jour de $\mathbf{h}_t$ et calcul de la sortie). Le coût total est $O(T(m^2 + md))$, linéaire en $T$. Le RNN est donc plus économe en calcul par jeton, mais il comprime tout le contexte dans un vecteur de taille fixe $m$, ce qui limite sa capacité à exploiter les dépendances à long terme.
````
