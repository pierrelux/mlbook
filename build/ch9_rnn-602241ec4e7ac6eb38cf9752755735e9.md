---
kernelspec:
  name: python3
  display_name: Python 3
---

# Réseaux récurrents

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Décrire le problème des séquences et expliquer pourquoi le MLP est insuffisant
- Définir l'architecture d'un réseau récurrent (RNN) et le rôle de l'état caché
- Expliquer le déroulement dans le temps et le partage de paramètres
- Dériver la rétropropagation à travers le temps (BPTT)
- Expliquer la dissolution du gradient et les mécanismes de portes (LSTM, GRU)
```

Le chapitre 7 a montré comment le MLP apprend une représentation $\boldsymbol{\phi}(\mathbf{x})$ à partir de données tabulaires, puis effectue une prédiction linéaire sur cette représentation. Mais le MLP traite son entrée comme un vecteur plat de taille fixe: chaque dimension est indépendante de ses voisines, et la taille de l'entrée est déterminée à l'avance.

Or beaucoup de données ont une structure séquentielle. Un texte est une suite de mots, un signal audio est une suite d'échantillons, une série temporelle de température est une suite de mesures à intervalles réguliers. Ces données ont deux propriétés que le MLP ne sait pas exploiter: elles sont ordonnées (le mot "pas" a un sens différent au début et à la fin d'une phrase), et elles sont de longueur variable (une phrase peut avoir 5 ou 50 mots).

Dans ce chapitre, nous présentons les réseaux récurrents, une famille d'architectures conçues pour traiter les séquences. Nous commençons par montrer pourquoi l'approche naïve (tout mettre dans un vecteur plat) échoue, puis nous introduisons l'idée d'état caché récurrent, le mécanisme de rétropropagation à travers le temps, et les variantes à portes (LSTM, GRU) qui atténuent la dissolution du gradient.

## Le problème des séquences

Supposons que nous voulions prédire le mot suivant dans une phrase, ou classifier le sentiment d'un commentaire. L'entrée est une séquence $(\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T)$ où chaque $\mathbf{x}_t \in \mathbb{R}^d$ représente un élément (un mot, une mesure, un échantillon). La longueur $T$ varie d'un exemple à l'autre.

L'approche la plus directe serait de concaténer tous les éléments en un seul vecteur et d'utiliser un MLP:

$$
\mathbf{x}_{\text{concat}} = [\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T] \in \mathbb{R}^{Td}
$$

Cette approche a trois problèmes. Le premier est la taille fixe: le MLP attend un vecteur de dimension $Td$, mais $T$ varie d'un exemple à l'autre. Pour des phrases de longueurs différentes, il faudrait tronquer ou rembourrer, ce qui est inélégant et gaspille de l'information. Le deuxième est l'absence de partage: les paramètres qui traitent $\mathbf{x}_1$ sont complètement distincts de ceux qui traitent $\mathbf{x}_2$, même si ces deux positions jouent un rôle analogue. Un motif appris en début de séquence (par exemple, reconnaître une négation) ne se transfère pas aux autres positions. Le troisième est l'absence de notion d'ordre: si l'on permute les éléments de l'entrée, un MLP n'a aucune façon de savoir que l'ordre a changé, à moins de l'encoder explicitement dans l'architecture.

Nous avons besoin d'une architecture qui traite la séquence élément par élément, qui partage ses paramètres entre les positions, et qui maintient une forme de mémoire de ce qu'elle a vu jusqu'ici.

## L'état caché récurrent

L'idée des réseaux récurrents est de traiter la séquence un élément à la fois, en maintenant un vecteur d'état $\mathbf{h}_t \in \mathbb{R}^m$ qui résume l'historique de la séquence jusqu'au pas $t$. À chaque pas de temps, le réseau lit le nouvel élément $\mathbf{x}_t$, le combine avec l'état précédent $\mathbf{h}_{t-1}$, et produit un nouvel état $\mathbf{h}_t$.

La mise à jour prend la forme:

$$
\mathbf{h}_t = \varphi(W_{hh}\, \mathbf{h}_{t-1} + W_{xh}\, \mathbf{x}_t + \mathbf{b}_h)
$$ (eq:rnn-update)

où $W_{hh} \in \mathbb{R}^{m \times m}$ et $W_{xh} \in \mathbb{R}^{m \times d}$ sont des matrices de poids, $\mathbf{b}_h \in \mathbb{R}^m$ est un biais, et $\varphi$ est une fonction d'activation (typiquement $\tanh$). L'état initial est $\mathbf{h}_0 = \mathbf{0}$.

Cette équation est un MLP appliqué à la concaténation de $\mathbf{h}_{t-1}$ et $\mathbf{x}_t$. On peut réécrire {eq}`eq:rnn-update` sous la forme:

$$
\mathbf{h}_t = \varphi\!\left(
\begin{bmatrix} W_{hh} & W_{xh} \end{bmatrix}
\begin{bmatrix} \mathbf{h}_{t-1} \\ \mathbf{x}_t \end{bmatrix}
+ \mathbf{b}_h
\right)
$$

Pour produire une sortie à chaque pas de temps (par exemple, prédire le mot suivant), on ajoute une couche de sortie:

$$
\mathbf{y}_t = W_{hy}\, \mathbf{h}_t + \mathbf{b}_y
$$

où $W_{hy} \in \mathbb{R}^{K \times m}$ projette l'état caché vers l'espace de sortie.

Les trois paramètres $(W_{hh}, W_{xh}, W_{hy})$ et les biais sont les mêmes à chaque pas de temps. C'est le partage de paramètres qui distingue le RNN du MLP: un motif appris à une position fonctionne à toutes les positions.

## Déroulement dans le temps

Pour visualiser le calcul, on peut "dérouler" le RNN dans le temps. Le même réseau est copié $T$ fois, une copie par pas de temps, avec les mêmes poids partout:

```{mermaid}
graph LR
    h0["h₀ = 0"]
    x1["x₁"]
    h1["h₁"]
    x2["x₂"]
    h2["h₂"]
    dots["⋯"]
    xT["x_T"]
    hT["h_T"]

    h0 --> h1
    x1 --> h1
    h1 --> h2
    x2 --> h2
    h2 --> dots
    dots --> hT
    xT --> hT

    style h0 fill:#f5f5f5,stroke:#666666
    style h1 fill:#f5f5f5,stroke:#666666
    style h2 fill:#f5f5f5,stroke:#666666
    style hT fill:#f5f5f5,stroke:#666666
    style x1 fill:#dae8fc,stroke:#6c8ebf
    style x2 fill:#dae8fc,stroke:#6c8ebf
    style xT fill:#dae8fc,stroke:#6c8ebf
    style dots fill:#ffffff,stroke:#ffffff
```

Le réseau déroulé ressemble à un MLP très profond (avec $T$ "couches"), mais avec une différence: toutes les couches partagent les mêmes poids. À chaque couche, un nouvel élément d'entrée $\mathbf{x}_t$ est injecté.

Selon la tâche, on utilise l'état caché de différentes façons:

- Classification de séquence (plusieurs entrées, une sortie): on lit $\mathbf{h}_T$, l'état final, et on le passe à un classifieur. Exemple: classifier le sentiment d'un commentaire.
- Étiquetage de séquence (une sortie par entrée): on produit $\mathbf{y}_t$ à chaque pas de temps. Exemple: identifier la catégorie grammaticale de chaque mot.
- Séquence à séquence: un premier RNN (l'encodeur) lit la séquence d'entrée et produit un état $\mathbf{h}_T$; un second RNN (le décodeur) génère la séquence de sortie à partir de cet état. Exemple: traduction automatique.

## Rétropropagation à travers le temps

Pour entraîner un RNN, nous devons calculer les gradients de la perte par rapport aux paramètres. Le réseau déroulé est un graphe de calcul comme un autre: on peut appliquer la rétropropagation vue au chapitre 7.

Considérons une perte $\mathcal{L} = \sum_{t=1}^T \ell_t$ qui accumule un terme à chaque pas de temps. Le gradient par rapport à $W_{hh}$ fait intervenir la chaîne de dépendances $\mathbf{h}_1 \to \mathbf{h}_2 \to \cdots \to \mathbf{h}_T$. Par la règle de la chaîne:

$$
\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial \ell_t}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial W_{hh}}
$$

Le terme $\frac{\partial \mathbf{h}_t}{\partial W_{hh}}$ dépend de tous les états précédents. En développant:

$$
\frac{\partial \mathbf{h}_t}{\partial W_{hh}} = \sum_{k=1}^t \left(\prod_{j=k+1}^t \frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}}\right) \frac{\partial^+ \mathbf{h}_k}{\partial W_{hh}}
$$

où $\frac{\partial^+ \mathbf{h}_k}{\partial W_{hh}}$ désigne la dérivée directe (en traitant $\mathbf{h}_{k-1}$ comme une constante), et le produit de jacobiennes $\prod_{j=k+1}^t \frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}}$ propage le gradient à travers le temps.

Ce produit de jacobiennes est la source du problème principal des RNN.

## Dissolution du gradient

Chaque facteur du produit $\prod_{j=k+1}^t \frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}}$ est la jacobienne de la mise à jour récurrente:

$$
\frac{\partial \mathbf{h}_j}{\partial \mathbf{h}_{j-1}} = \text{diag}(\varphi'(\mathbf{a}_j))\, W_{hh}
$$

où $\mathbf{a}_j = W_{hh}\, \mathbf{h}_{j-1} + W_{xh}\, \mathbf{x}_j + \mathbf{b}_h$. Pour $\varphi = \tanh$, la dérivée $\varphi'$ est comprise entre 0 et 1. Si les valeurs propres de $W_{hh}$ sont inférieures à 1 en module, le produit de $t - k$ matrices décroît exponentiellement. Le gradient "disparaît" et le réseau ne peut plus apprendre les dépendances entre des éléments éloignés dans la séquence.

À l'inverse, si les valeurs propres sont supérieures à 1, le produit croît exponentiellement: le gradient "explose". Ce problème est plus facile à traiter (on peut tronquer la norme du gradient, une technique appelée écrêtage du gradient (*gradient clipping*)), mais la dissolution du gradient est plus insidieuse, car elle ne produit pas d'erreur visible: l'entraînement semble fonctionner, mais le réseau ignore silencieusement les dépendances à long terme.

C'est le même phénomène que dans les MLP profonds (chapitre 8), mais aggravé par le fait que $T$ peut être très grand (des centaines ou des milliers de pas de temps).

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'retina'

np.random.seed(42)
m = 50
W = np.random.randn(m, m) * 0.9 / np.sqrt(m)

norms = []
v = np.random.randn(m)
v = v / np.linalg.norm(v)
for t in range(100):
    v = np.tanh(W @ v)  # simplified: tanh(W h)
    scale = np.linalg.norm(np.diag(1 - v**2) @ W, ord=2)
    norms.append(scale)

grad_product = np.cumprod(norms)

fig, ax = plt.subplots(figsize=(7, 3))
ax.semilogy(grad_product, 'C0', lw=1.5)
ax.set_xlabel('Nombre de pas de temps ($t - k$)')
ax.set_ylabel('Norme du produit de jacobiennes')
ax.set_title('Décroissance exponentielle du gradient dans un RNN')
ax.grid(True, alpha=0.3)
plt.tight_layout()
```

La figure montre la norme du produit de jacobiennes en fonction du nombre de pas de temps. Après quelques dizaines de pas, le gradient est essentiellement nul: le réseau ne reçoit plus de signal d'apprentissage pour les dépendances à long terme.

## LSTM et GRU: des mécanismes de portes

Le LSTM (*Long Short-Term Memory*) {cite}`hochreiter1997long` résout la dissolution du gradient en introduisant un état de cellule $\mathbf{c}_t$ qui circule d'un pas à l'autre de façon (presque) linéaire. Des portes apprises contrôlent le flux d'information:

$$
\begin{aligned}
\mathbf{f}_t &= \sigma(W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) && \text{(porte d'oubli)} \\
\mathbf{i}_t &= \sigma(W_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) && \text{(porte d'entrée)} \\
\tilde{\mathbf{c}}_t &= \tanh(W_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) && \text{(candidat)} \\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t && \text{(mise à jour de la cellule)} \\
\mathbf{o}_t &= \sigma(W_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) && \text{(porte de sortie)} \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
\end{aligned}
$$

où $\odot$ est le produit élément par élément et $[\mathbf{h}_{t-1}, \mathbf{x}_t]$ désigne la concaténation.

L'état de cellule $\mathbf{c}_t$ agit comme un chemin direct pour le gradient. Quand la porte d'oubli $\mathbf{f}_t \approx 1$ et la porte d'entrée $\mathbf{i}_t \approx 0$, l'état de cellule est simplement copié: $\mathbf{c}_t \approx \mathbf{c}_{t-1}$. Le gradient circule sans atténuation à travers cette connexion linéaire, ce qui permet d'apprendre des dépendances sur des centaines de pas de temps.

Le GRU (*Gated Recurrent Unit*) {cite}`cho2014learning` simplifie le LSTM en fusionnant l'état de cellule et l'état caché, et en utilisant deux portes au lieu de trois:

$$
\begin{aligned}
\mathbf{z}_t &= \sigma(W_z [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z) && \text{(porte de mise à jour)} \\
\mathbf{r}_t &= \sigma(W_r [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r) && \text{(porte de réinitialisation)} \\
\tilde{\mathbf{h}}_t &= \tanh(W_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_h) \\
\mathbf{h}_t &= (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
\end{aligned}
$$

La porte de mise à jour $\mathbf{z}_t$ joue le rôle combiné des portes d'oubli et d'entrée du LSTM. En pratique, LSTM et GRU ont des performances comparables sur la plupart des tâches. Le GRU a moins de paramètres, ce qui peut être un avantage quand les données sont limitées.

Ce qui compte pour l'intuition, c'est le mécanisme commun: les portes permettent au gradient de circuler sans être multiplié par $W_{hh}$ à chaque pas, ce qui atténue la dissolution du gradient.

## Limites des réseaux récurrents

Malgré les mécanismes de portes, les réseaux récurrents ont des limitations qui expliquent pourquoi ils ont été largement remplacés par les transformeurs pour beaucoup de tâches.

Le traitement séquentiel est le premier problème. Le calcul de $\mathbf{h}_t$ dépend de $\mathbf{h}_{t-1}$, qui dépend de $\mathbf{h}_{t-2}$, et ainsi de suite. On ne peut pas paralléliser le traitement des différentes positions: il faut les traiter dans l'ordre. Sur du matériel moderne (GPU, TPU), conçu pour le parallélisme massif, cette contrainte est un goulot d'étranglement.

Le goulot d'information est le second problème. Toute l'information sur la séquence passe par le vecteur $\mathbf{h}_t$ de dimension fixe $m$. Pour une longue séquence, il est difficile de compresser tout le contexte pertinent dans ce vecteur. Même avec LSTM ou GRU, les dépendances à très long terme restent difficiles à capturer.

Ces deux limitations motivent le mécanisme d'attention, que nous verrons au chapitre suivant. L'attention permet à chaque position d'une séquence de consulter directement toutes les autres positions, sans passer par une chaîne d'états cachés, et de manière parallélisable.

## Résumé

Les réseaux récurrents étendent le MLP aux données séquentielles en introduisant un état caché $\mathbf{h}_t$ qui résume l'historique. Les mêmes paramètres sont partagés à chaque pas de temps, ce qui permet de traiter des séquences de longueur variable. Le réseau déroulé dans le temps est un graphe de calcul profond auquel on applique la rétropropagation standard (BPTT).

Le produit de jacobiennes qui apparaît dans BPTT décroît (ou croît) exponentiellement avec le nombre de pas de temps. Cette dissolution du gradient empêche l'apprentissage de dépendances à long terme. Le LSTM et le GRU introduisent des portes qui créent un chemin linéaire pour le gradient, atténuant ce problème.

Malgré ces progrès, les RNN souffrent d'un traitement séquentiel non parallélisable et d'un goulot d'information dû à la compression de toute la séquence dans un vecteur de taille fixe. Le chapitre suivant introduit le mécanisme d'attention, qui résout ces deux problèmes en permettant l'accès direct entre toutes les positions d'une séquence.

```{admonition} Ce que vous devez retenir
:class: tip

1. Un RNN maintient un état caché $\mathbf{h}_t$ mis à jour à chaque pas de temps par $\mathbf{h}_t = \varphi(W_{hh}\, \mathbf{h}_{t-1} + W_{xh}\, \mathbf{x}_t + \mathbf{b}_h)$.

2. Le partage de paramètres entre les positions permet de traiter des séquences de longueur variable et de transférer les motifs appris d'une position à l'autre.

3. La rétropropagation à travers le temps (BPTT) applique la règle de la chaîne au réseau déroulé. Le gradient implique un produit de jacobiennes qui décroît exponentiellement: c'est la dissolution du gradient.

4. Le LSTM et le GRU atténuent ce problème grâce à des portes qui créent un chemin linéaire pour le gradient via l'état de cellule.

5. Les RNN ont deux limitations fondamentales: le traitement séquentiel (non parallélisable) et le goulot d'information ($\mathbf{h}_t$ de taille fixe). Le mécanisme d'attention y remédie.
```

## Exercices

Les exercices ★ vérifient la compréhension de base. Les exercices ★★ demandent d'appliquer les concepts à des calculs concrets. Les exercices ★★★ approfondissent le sujet et sont optionnels pour IFT3395.

````{admonition} Exercice 1: Nombre de paramètres d'un RNN ★
:class: hint dropdown

Considérez un RNN avec une entrée $\mathbf{x}_t \in \mathbb{R}^{50}$, un état caché $\mathbf{h}_t \in \mathbb{R}^{128}$, et une sortie $\mathbf{y}_t \in \mathbb{R}^{10}$.

1. Combien de paramètres contiennent les matrices $W_{xh}$, $W_{hh}$ et $W_{hy}$ (sans les biais)?
2. Combien de paramètres au total (avec les biais)?
3. Ce nombre dépend-il de la longueur $T$ de la séquence? Pourquoi?
````

````{admonition} Solution Exercice 1
:class: dropdown

1. $W_{xh} \in \mathbb{R}^{128 \times 50}$: 6 400 paramètres. $W_{hh} \in \mathbb{R}^{128 \times 128}$: 16 384 paramètres. $W_{hy} \in \mathbb{R}^{10 \times 128}$: 1 280 paramètres. Total sans biais: 24 064.

2. Biais: $\mathbf{b}_h \in \mathbb{R}^{128}$ (128) et $\mathbf{b}_y \in \mathbb{R}^{10}$ (10). Total avec biais: 24 202.

3. Le nombre de paramètres ne dépend pas de $T$, car les mêmes poids sont réutilisés à chaque pas de temps. C'est une propriété du partage de paramètres.
````

````{admonition} Exercice 2: Déroulement d'un RNN à la main ★
:class: hint dropdown

Considérez un RNN à une dimension ($d = m = 1$) avec $w_{hh} = 0{,}5$, $w_{xh} = 1$, $b_h = 0$, $\varphi = \text{identité}$ (pas d'activation) et $h_0 = 0$.

Pour la séquence d'entrée $(x_1, x_2, x_3) = (1, 0, -1)$:

1. Calculez $h_1$, $h_2$ et $h_3$.
2. Quelle serait la valeur de $h_t$ si la séquence continuait avec des entrées nulles ($x_t = 0$ pour $t > 3$)?
````

````{admonition} Solution Exercice 2
:class: dropdown

1. $h_1 = 0{,}5 \cdot 0 + 1 \cdot 1 = 1$. $h_2 = 0{,}5 \cdot 1 + 1 \cdot 0 = 0{,}5$. $h_3 = 0{,}5 \cdot 0{,}5 + 1 \cdot (-1) = -0{,}75$.

2. Pour $t > 3$, $h_t = 0{,}5 \cdot h_{t-1}$. L'état décroît géométriquement vers 0: $h_4 = -0{,}375$, $h_5 = -0{,}1875$, etc. Puisque $|w_{hh}| = 0{,}5 < 1$, le réseau "oublie" progressivement. C'est une manifestation de la dissolution du gradient dans un cas linéaire simplifié.
````

````{admonition} Exercice 3: Dissolution du gradient ★★
:class: hint dropdown

Pour le RNN linéaire de l'exercice 2 (sans activation), montrez que:

$$
\frac{\partial h_t}{\partial h_k} = w_{hh}^{t-k}
$$

Que se passe-t-il quand $|w_{hh}| < 1$ et $t - k$ est grand? Et quand $|w_{hh}| > 1$?
````

````{admonition} Solution Exercice 3
:class: dropdown

Sans activation, $h_t = w_{hh}\, h_{t-1} + w_{xh}\, x_t$. La dérivée par rapport à $h_{t-1}$ est $w_{hh}$. Par la règle de la chaîne:

$$
\frac{\partial h_t}{\partial h_k} = \prod_{j=k+1}^t \frac{\partial h_j}{\partial h_{j-1}} = \prod_{j=k+1}^t w_{hh} = w_{hh}^{t-k}
$$

Si $|w_{hh}| < 1$, ce terme décroît exponentiellement vers 0: le gradient se dissout et le réseau ne peut pas apprendre les dépendances entre $h_k$ et la perte au temps $t$. Si $|w_{hh}| > 1$, le terme croît exponentiellement: le gradient explose. Le cas multidimensionnel est analogue, avec les valeurs propres de $W_{hh}$ jouant le rôle de $w_{hh}$.
````

````{admonition} Exercice 4: Porte d'oubli et gradient ★★
:class: hint dropdown

Dans un LSTM simplifié à une dimension, l'état de cellule se met à jour par $c_t = f_t \cdot c_{t-1} + i_t \cdot \tilde{c}_t$, où $f_t$ est la porte d'oubli.

1. Calculez $\frac{\partial c_t}{\partial c_{t-1}}$.
2. Calculez $\frac{\partial c_t}{\partial c_k}$ pour $k < t$, en supposant les portes constantes ($f_j = f$ pour tout $j$).
3. Comparez avec le RNN simple. Pour quelle valeur de $f$ le gradient ne disparaît-il pas?
````

````{admonition} Solution Exercice 4
:class: dropdown

1. $\frac{\partial c_t}{\partial c_{t-1}} = f_t$ (le terme $i_t \cdot \tilde{c}_t$ ne dépend pas directement de $c_{t-1}$ dans cette version simplifiée).

2. $\frac{\partial c_t}{\partial c_k} = \prod_{j=k+1}^t f_j = f^{t-k}$ (avec portes constantes).

3. Dans le RNN simple, le facteur est $w_{hh}^{t-k}$ multiplié par $\varphi'$, ce qui pousse le gradient vers 0. Dans le LSTM, le facteur est $f^{t-k}$, et le réseau peut apprendre $f \approx 1$, ce qui donne $\frac{\partial c_t}{\partial c_k} \approx 1$. Le gradient circule sans atténuation le long de l'état de cellule. C'est le mécanisme qui permet au LSTM de capturer les dépendances à long terme.
````
