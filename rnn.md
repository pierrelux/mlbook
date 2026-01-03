# Réseaux de neurones récurrents

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Expliquer le fonctionnement des réseaux récurrents (RNN)
- Formuler le modèle génératif sous-jacent aux RNN
- Comprendre la rétropropagation dans le temps (BPTT)
- Décrire le problème de la disparition et de l'explosion du gradient
- Expliquer les architectures GRU et LSTM
- Comprendre les mécanismes d'attention
```

## Motivation: traiter des séquences

Les réseaux de neurones que nous avons vus jusqu'à présent — MLP et CNN — traitent des entrées de taille fixe. Cependant, de nombreuses applications impliquent des **séquences de longueur variable**:
- Texte et langage naturel
- Séries temporelles financières ou météorologiques
- Signaux audio et parole
- Vidéos (séquences d'images)

Les **réseaux de neurones récurrents** (RNN, pour *Recurrent Neural Networks*) sont conçus spécifiquement pour traiter de telles données séquentielles. Leur caractéristique principale est de maintenir un **état caché** qui résume l'information des pas de temps précédents.

## Formulation probabiliste

### Le modèle génératif

Les RNN peuvent être vus comme un modèle génératif conditionnel. Étant donné une entrée optionnelle $\boldsymbol{x}$, nous voulons modéliser la distribution d'une séquence de sortie $\boldsymbol{y}_{1:T} = (\boldsymbol{y}_1, \boldsymbol{y}_2, \ldots, \boldsymbol{y}_T)$:

$$
p(\boldsymbol{y}_{1:T} \mid \boldsymbol{x}) = \sum_{\boldsymbol{h}_{1:T}} p(\boldsymbol{y}_{1:T}, \boldsymbol{h}_{1:T} \mid \boldsymbol{x}) = \sum_{\boldsymbol{h}_{1:T}} \prod_{t=1}^T p(\boldsymbol{y}_t \mid \boldsymbol{h}_t) \, p(\boldsymbol{h}_t \mid \boldsymbol{h}_{t-1}, \boldsymbol{y}_{t-1}, \boldsymbol{x})
$$

où $\boldsymbol{h}_t$ est l'**état caché** au temps $t$. Cette factorisation révèle la structure du RNN:
1. L'état caché $\boldsymbol{h}_t$ dépend de l'état précédent $\boldsymbol{h}_{t-1}$, de la sortie précédente $\boldsymbol{y}_{t-1}$ et de l'entrée $\boldsymbol{x}$
2. La sortie $\boldsymbol{y}_t$ dépend uniquement de l'état caché $\boldsymbol{h}_t$

### Distribution de sortie

La distribution de probabilité des sorties dépend du type de données:

**Pour la classification** (par exemple, prédire le prochain caractère):
$$
p(\boldsymbol{y}_t \mid \boldsymbol{h}_t) = \text{Cat}(\boldsymbol{y}_t \mid \text{softmax}(\mathbf{W}_{hy} \boldsymbol{h}_t + \boldsymbol{b}_y))
$$

où $\mathbf{W}_{hy}$ sont les poids de la couche cachée vers la sortie et $\boldsymbol{b}_y$ est le biais.

**Pour la régression** (par exemple, prédire une valeur continue):
$$
p(\boldsymbol{y}_t \mid \boldsymbol{h}_t) = \mathcal{N}(\boldsymbol{y}_t \mid \mathbf{W}_{hy} \boldsymbol{h}_t + \boldsymbol{b}_y, \sigma^2 \mathbf{I})
$$

### Fonction de transition

L'état caché est généralement calculé de manière **déterministe**:
$$
p(\boldsymbol{h}_t \mid \boldsymbol{h}_{t-1}, \boldsymbol{y}_{t-1}, \boldsymbol{x}) = \mathbb{I}(\boldsymbol{h}_t = f(\boldsymbol{h}_{t-1}, \boldsymbol{y}_{t-1}, \boldsymbol{x}))
$$

La fonction de mise à jour $f$ est typiquement:
$$
\boldsymbol{h}_t = \varphi\left(\mathbf{W}_{xh} [\boldsymbol{x}; \boldsymbol{y}_{t-1}] + \mathbf{W}_{hh} \boldsymbol{h}_{t-1} + \boldsymbol{b}_h\right)
$$

où:
- $\mathbf{W}_{hh}$ sont les poids **cachés-cachés** (récurrence)
- $\mathbf{W}_{xh}$ sont les poids **entrée-cachés**
- $\boldsymbol{b}_h$ est le biais
- $\varphi$ est une fonction d'activation (tanh, ReLU, etc.)

La notation $[\boldsymbol{x}; \boldsymbol{y}_{t-1}]$ désigne la concaténation des vecteurs.

## Types d'architectures RNN

### Vec2Seq: génération de séquences

Dans le cas **vec2seq** (*vector to sequence*), nous générons une séquence de sortie de longueur variable à partir d'un vecteur d'entrée fixe $\boldsymbol{x}$.

Les étapes de génération sont:
1. Initialiser l'état caché à partir de $\boldsymbol{x}$: $\boldsymbol{h}_1 = f(\boldsymbol{x})$
2. Échantillonner: $\tilde{\boldsymbol{y}}_t \sim p(\boldsymbol{y}_t \mid \boldsymbol{h}_t)$
3. Mettre à jour l'état: $\boldsymbol{h}_{t+1} = f(\boldsymbol{h}_t, \tilde{\boldsymbol{y}}_t, \boldsymbol{x})$
4. Répéter jusqu'à générer un token de fin ou atteindre une longueur maximale

**Exemple:** génération de texte caractère par caractère. Voici un exemple de sortie d'un RNN entraîné sur le livre *La Machine à explorer le temps* de H.G. Wells:

> "the githa some thong the time traveller held in his hand was a glittering metallic framework scarcely larger than a small clock..."

Ce texte, bien qu'imparfait, montre que le modèle a appris la structure du langage anglais sans supervision explicite.

### Seq2Vec: classification de séquences

Dans le cas **seq2vec** (*sequence to vector*), nous produisons un vecteur de sortie fixe à partir d'une séquence d'entrée de longueur variable.

Pour classifier une séquence, on peut:
1. Utiliser l'état caché final $\boldsymbol{h}_T$
2. Moyenner tous les états cachés
3. Utiliser un mécanisme d'attention (voir plus loin)

### Seq2Seq: transformation de séquences

L'architecture **seq2seq** (*sequence to sequence*) combine un encodeur et un décodeur:
1. **Encodeur**: lit la séquence d'entrée et produit un vecteur de contexte
2. **Décodeur**: génère la séquence de sortie conditionnée par ce contexte

Cette architecture est utilisée pour:
- La traduction automatique
- Le résumé de texte
- La génération de légendes d'images (avec un CNN comme encodeur)

### RNN bidirectionnels

Dans un RNN standard, l'état caché $\boldsymbol{h}_t$ ne dépend que du contexte **passé** $\boldsymbol{x}_{1:t}$. Pour certaines tâches (comme la reconnaissance d'entités nommées), le contexte **futur** est également informatif.

Un **RNN bidirectionnel** combine deux RNN:
- Un qui traite la séquence de gauche à droite (état avant $\overrightarrow{\boldsymbol{h}}_t$)
- Un qui traite de droite à gauche (état arrière $\overleftarrow{\boldsymbol{h}}_t$)

L'état combiné est $\boldsymbol{h}_t = [\overrightarrow{\boldsymbol{h}}_t; \overleftarrow{\boldsymbol{h}}_t]$.

## Rétropropagation dans le temps (BPTT)

### Calcul du gradient

Pour entraîner un RNN, nous cherchons les paramètres qui maximisent la vraisemblance:
$$
\boldsymbol{\theta}^* = \operatorname{argmax}_{\boldsymbol{\theta}} \, p(\boldsymbol{y}_{1:T} \mid \boldsymbol{x}_{1:T}, \boldsymbol{\theta})
$$

Considérons le modèle simplifié:
$$
\begin{aligned}
\boldsymbol{h}_t &= \mathbf{W}_{hx} \boldsymbol{x}_t + \mathbf{W}_{hh} \boldsymbol{h}_{t-1} \\
\boldsymbol{o}_t &= \mathbf{W}_{ho} \boldsymbol{h}_t
\end{aligned}
$$

La fonction de perte moyenne est:
$$
L = \frac{1}{T} \sum_{t=1}^T \ell(y_t, o_t)
$$

Pour calculer le gradient, nous **déroulons** le graphe de calcul dans le temps et appliquons la rétropropagation. Cette technique s'appelle **BPTT** (*Backpropagation Through Time*).

### Dérivation du gradient

De manière générale, avec $\boldsymbol{h}_t = f(\boldsymbol{x}_t, \boldsymbol{h}_{t-1}, \boldsymbol{w}_h)$ et $\boldsymbol{o}_t = g(\boldsymbol{h}_t, \boldsymbol{w}_o)$, la règle de la chaîne donne:

$$
\frac{\partial L}{\partial \boldsymbol{w}_h} = \frac{1}{T} \sum_{t=1}^T \frac{\partial \ell(y_t, \boldsymbol{o}_t)}{\partial \boldsymbol{o}_t} \frac{\partial g(\boldsymbol{h}_t, \boldsymbol{w}_o)}{\partial \boldsymbol{h}_t} \frac{\partial \boldsymbol{h}_t}{\partial \boldsymbol{w}_h}
$$

Le terme $\frac{\partial \boldsymbol{h}_t}{\partial \boldsymbol{w}_h}$ nécessite une récursion:
$$
\frac{\partial \boldsymbol{h}_t}{\partial \boldsymbol{w}_h} = \frac{\partial f(\boldsymbol{x}_t, \boldsymbol{h}_{t-1}, \boldsymbol{w}_h)}{\partial \boldsymbol{w}_h} + \frac{\partial f(\boldsymbol{x}_t, \boldsymbol{h}_{t-1}, \boldsymbol{w}_h)}{\partial \boldsymbol{h}_{t-1}} \frac{\partial \boldsymbol{h}_{t-1}}{\partial \boldsymbol{w}_h}
$$

En déroulant cette récursion:
$$
\frac{\partial \boldsymbol{h}_t}{\partial \boldsymbol{w}_h} = \frac{\partial f_t}{\partial \boldsymbol{w}_h} + \sum_{i=1}^{t-1} \left( \prod_{j=i+1}^t \frac{\partial f_j}{\partial \boldsymbol{h}_{j-1}} \right) \frac{\partial f_i}{\partial \boldsymbol{w}_h}
$$

### BPTT tronquée

Le produit imbriqué dans la somme rend le calcul $O(T^2)$, ce qui est prohibitif pour les longues séquences. Une solution pratique est la **BPTT tronquée**: on limite la rétropropagation aux $K$ pas de temps les plus récents.

Cette approximation:
- Réduit la complexité à $O(TK)$
- Introduit un biais dans l'estimation du gradient
- Est généralement acceptable si $K$ est suffisamment grand

## Le problème du gradient

### Disparition et explosion du gradient

Le produit de Jacobiennes $\prod_{j=i+1}^t \frac{\partial f_j}{\partial \boldsymbol{h}_{j-1}}$ pose un problème fondamental. À chaque pas de temps, nous multiplions approximativement par la matrice $\mathbf{W}_{hh}$.

Si les valeurs propres de $\mathbf{W}_{hh}$ ont:
- Un **module < 1**: le produit tend vers zéro → **disparition du gradient**
- Un **module > 1**: le produit explose → **explosion du gradient**

Ces problèmes sont analogues à ceux rencontrés dans les réseaux profonds, mais amplifiés car la même matrice est appliquée à chaque pas de temps.

### Conséquences pratiques

- **Disparition**: le réseau ne peut pas apprendre les dépendances à long terme
- **Explosion**: l'entraînement devient instable

### Solutions

1. **Rognage de gradient** (*gradient clipping*): si $\|\nabla\| > \theta$, on normalise: $\nabla \leftarrow \frac{\theta}{\|\nabla\|} \nabla$

2. **Contrôle du rayon spectral**: initialiser $\mathbf{W}_{hh}$ de sorte que son rayon spectral soit proche de 1

3. **Architectures avec portes**: GRU et LSTM (voir ci-dessous)

## Unités récurrentes avec portes (GRU)

Les **GRU** (*Gated Recurrent Units*) résolvent le problème du gradient en introduisant des **portes** qui contrôlent le flux d'information.

### Intuition

L'idée clé est d'apprendre **quand**:
1. **Mettre à jour** l'état caché avec de nouvelles informations
2. **Réinitialiser** l'état caché et oublier les informations obsolètes

### Architecture

Les GRU utilisent deux portes:

**Porte de réinitialisation** $\mathbf{R}_t$:
$$
\mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xr} + \mathbf{H}_{t-1} \mathbf{W}_{hr} + \boldsymbol{b}_r)
$$

**Porte de mise à jour** $\mathbf{Z}_t$:
$$
\mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xz} + \mathbf{H}_{t-1} \mathbf{W}_{hz} + \boldsymbol{b}_z)
$$

Ces portes prennent des valeurs entre 0 et 1 grâce à la fonction sigmoïde $\sigma$.

**État candidat** (utilise la porte de réinitialisation):
$$
\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xh} + (\mathbf{R}_t \odot \mathbf{H}_{t-1}) \mathbf{W}_{hh} + \boldsymbol{b}_h)
$$

La porte de réinitialisation $\mathbf{R}_t$ contrôle combien de l'état précédent $\mathbf{H}_{t-1}$ influence le candidat.

**Mise à jour de l'état** (utilise la porte de mise à jour):
$$
\mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1} + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t
$$

Quand $\mathbf{Z}_t \approx 1$, l'état précédent est conservé; quand $\mathbf{Z}_t \approx 0$, le nouvel état candidat est adopté. Cette interpolation permet de créer des **chemins de gradient** qui ne passent pas par les multiplications matricielles.

## Mémoire à court et long terme (LSTM)

Les **LSTM** (*Long Short-Term Memory*) sont une architecture plus élaborée qui augmente l'état caché avec une **cellule de mémoire** dédiée.

### Architecture à trois portes

Les LSTM utilisent trois portes:

**Porte de sortie** $\mathbf{O}_t$ — détermine ce qui est **lu** de la cellule:
$$
\mathbf{O}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xo} + \mathbf{H}_{t-1} \mathbf{W}_{ho} + \boldsymbol{b}_o)
$$

**Porte d'entrée** $\mathbf{I}_t$ — détermine ce qui est **écrit** dans la cellule:
$$
\mathbf{I}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xi} + \mathbf{H}_{t-1} \mathbf{W}_{hi} + \boldsymbol{b}_i)
$$

**Porte d'oubli** $\mathbf{F}_t$ — détermine ce qui est **effacé** de la cellule:
$$
\mathbf{F}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xf} + \mathbf{H}_{t-1} \mathbf{W}_{hf} + \boldsymbol{b}_f)
$$

### Mise à jour de la cellule

Le contenu candidat pour la cellule:
$$
\tilde{\mathbf{C}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xc} + \mathbf{H}_{t-1} \mathbf{W}_{hc} + \boldsymbol{b}_c)
$$

La cellule est mise à jour par:
$$
\mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t
$$

Cette équation montre clairement les deux flux:
- $\mathbf{F}_t \odot \mathbf{C}_{t-1}$: conservation sélective de la mémoire précédente
- $\mathbf{I}_t \odot \tilde{\mathbf{C}}_t$: ajout sélectif de nouvelles informations

### État caché

L'état caché est une version transformée de la cellule:
$$
\mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t)
$$

La distinction entre $\mathbf{C}_t$ et $\mathbf{H}_t$ est importante:
- $\mathbf{C}_t$ est la **mémoire à long terme** — elle peut conserver l'information sur de nombreux pas de temps
- $\mathbf{H}_t$ est la **mémoire à court terme** — elle sert de sortie pour le pas de temps actuel

### Pourquoi les LSTM fonctionnent

Le flux de gradient à travers la cellule est:
$$
\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} = \mathbf{F}_t
$$

Contrairement au RNN standard où le gradient doit passer par $\mathbf{W}_{hh}$ à chaque pas, ici le gradient peut circuler directement à travers la cellule, modulé uniquement par $\mathbf{F}_t$. Si le réseau apprend $\mathbf{F}_t \approx 1$, l'information peut être préservée indéfiniment.

## Mécanismes d'attention

### Motivation

Les architectures seq2seq classiques encodent toute la séquence d'entrée dans un seul vecteur de contexte. Cette approche a deux limitations:
1. **Goulot d'étranglement**: toute l'information doit passer par un vecteur de taille fixe
2. **Pas de focus**: le décodeur n'a pas accès direct aux parties pertinentes de l'entrée

Les **mécanismes d'attention** permettent au décodeur de "regarder" dynamiquement différentes parties de la séquence d'entrée.

### Formulation

L'attention peut être vue comme une **recherche souple dans un dictionnaire**:
- **Clés** $\mathbf{K} \in \mathbb{R}^{m \times k}$: représentations des $m$ éléments de la séquence d'entrée
- **Valeurs** $\mathbf{V} \in \mathbb{R}^{m \times v}$: informations associées à chaque élément
- **Requête** $\boldsymbol{q} \in \mathbb{R}^q$: ce que le décodeur cherche à un instant donné

La sortie de l'attention est une combinaison convexe des valeurs:
$$
\text{Attn}(\boldsymbol{q}, \mathbf{K}, \mathbf{V}) = \sum_{i=1}^m \alpha_i(\boldsymbol{q}, \boldsymbol{k}_{1:m}) \, \boldsymbol{v}_i
$$

Les poids d'attention $\alpha_i$ satisfont $\alpha_i \geq 0$ et $\sum_i \alpha_i = 1$.

### Calcul des poids d'attention

Les poids sont calculés à partir d'un **score d'attention** $a(\boldsymbol{q}, \boldsymbol{k}_i)$ qui mesure la similarité entre la requête et chaque clé:
$$
\alpha_i(\boldsymbol{q}, \boldsymbol{k}_{1:m}) = \text{softmax}_i\left([a(\boldsymbol{q}, \boldsymbol{k}_1), \ldots, a(\boldsymbol{q}, \boldsymbol{k}_m)]\right) = \frac{\exp(a(\boldsymbol{q}, \boldsymbol{k}_i))}{\sum_{j=1}^m \exp(a(\boldsymbol{q}, \boldsymbol{k}_j))}
$$

Exemples de fonctions de score:
- **Produit scalaire**: $a(\boldsymbol{q}, \boldsymbol{k}) = \boldsymbol{q}^\top \boldsymbol{k}$
- **Produit scalaire normalisé**: $a(\boldsymbol{q}, \boldsymbol{k}) = \frac{\boldsymbol{q}^\top \boldsymbol{k}}{\sqrt{d}}$
- **Additif**: $a(\boldsymbol{q}, \boldsymbol{k}) = \boldsymbol{w}^\top \tanh(\mathbf{W}_q \boldsymbol{q} + \mathbf{W}_k \boldsymbol{k})$

### Application à seq2seq

Dans un modèle seq2seq avec attention:
1. L'encodeur produit une séquence d'états cachés $(\boldsymbol{h}_1^{enc}, \ldots, \boldsymbol{h}_T^{enc})$
2. Ces états servent de clés et de valeurs
3. L'état caché du décodeur $\boldsymbol{h}_t^{dec}$ sert de requête
4. Le contexte attentionnel $\boldsymbol{c}_t = \text{Attn}(\boldsymbol{h}_t^{dec}, \mathbf{H}^{enc}, \mathbf{H}^{enc})$ est utilisé pour la prédiction

Cette architecture permet au modèle d'aligner dynamiquement les sorties avec les parties pertinentes de l'entrée, ce qui est particulièrement utile pour la traduction automatique.

## Applications

### Génération de légendes d'images

En combinant un CNN (encodeur d'image) avec un RNN (décodeur de texte), on peut générer automatiquement des descriptions d'images:
1. Le CNN extrait un vecteur de caractéristiques de l'image
2. Ce vecteur initialise l'état caché du RNN
3. Le RNN génère une séquence de mots décrivant l'image

### Traduction automatique

Les modèles seq2seq avec attention ont révolutionné la traduction:
1. L'encodeur lit la phrase source
2. Le décodeur génère la phrase cible, en utilisant l'attention pour aligner les mots

### Modèles de langage

Les RNN peuvent modéliser $p(w_1, w_2, \ldots, w_T)$ en factorisant:
$$
p(w_1, \ldots, w_T) = \prod_{t=1}^T p(w_t \mid w_1, \ldots, w_{t-1})
$$

Ces modèles sont utilisés pour:
- La complétion de texte
- La correction orthographique
- La génération de texte

## Résumé

Les réseaux de neurones récurrents permettent de traiter des séquences de longueur variable en maintenant un état caché qui résume l'historique. Points clés:

- **RNN de base**: simple mais souffre de la disparition/explosion du gradient
- **GRU**: deux portes (réinitialisation, mise à jour) pour contrôler le flux d'information
- **LSTM**: trois portes et une cellule de mémoire dédiée pour les dépendances à long terme
- **Attention**: permet au décodeur de se concentrer sur les parties pertinentes de l'entrée

Ces architectures ont été dominantes en NLP jusqu'à l'avènement des **Transformers**, qui généralisent l'attention à des architectures entièrement basées sur ce mécanisme.

## Exercices

```{admonition} Exercice 1: Déroulement d'un RNN
:class: tip

Considérez un RNN simple avec $\boldsymbol{h}_t = \tanh(\mathbf{W}_{xh} \boldsymbol{x}_t + \mathbf{W}_{hh} \boldsymbol{h}_{t-1})$ où:
- $\mathbf{W}_{xh} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$
- $\mathbf{W}_{hh} = \begin{pmatrix} 0.5 & 0 \\ 0 & 0.5 \end{pmatrix}$
- $\boldsymbol{h}_0 = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$

Calculez $\boldsymbol{h}_1$, $\boldsymbol{h}_2$, $\boldsymbol{h}_3$ pour la séquence d'entrée $x_1 = 1$, $x_2 = -1$, $x_3 = 0.5$.
```

```{admonition} Exercice 2: Explosion du gradient
:class: tip

Soit $\mathbf{W}_{hh} = \begin{pmatrix} 1.5 & 0 \\ 0 & 1.5 \end{pmatrix}$.
1. Calculez le rayon spectral de $\mathbf{W}_{hh}$.
2. Calculez $\mathbf{W}_{hh}^{10}$ et $\mathbf{W}_{hh}^{100}$.
3. Expliquez pourquoi cela pose problème pour la rétropropagation dans le temps.
```

```{admonition} Exercice 3: Porte d'oubli LSTM
:class: tip

Dans un LSTM, supposons que la porte d'oubli $\mathbf{F}_t = \mathbf{1}$ (tout à 1) et la porte d'entrée $\mathbf{I}_t = \mathbf{0}$ (tout à 0) pour tous les pas de temps.
1. Que devient l'équation de mise à jour de la cellule?
2. Comment évolue $\mathbf{C}_t$ au cours du temps?
3. Quel est l'effet sur le gradient lors de la rétropropagation?
```

```{admonition} Exercice 4: Attention
:class: tip

Considérez une séquence de 3 clés $\boldsymbol{k}_1, \boldsymbol{k}_2, \boldsymbol{k}_3$ avec les valeurs associées $\boldsymbol{v}_1 = [1, 0]$, $\boldsymbol{v}_2 = [0, 1]$, $\boldsymbol{v}_3 = [0.5, 0.5]$.

Si les scores d'attention (avant softmax) sont $a_1 = 2$, $a_2 = 1$, $a_3 = 1$:
1. Calculez les poids d'attention $\alpha_1, \alpha_2, \alpha_3$.
2. Calculez la sortie de l'attention.
3. Comment changerait la sortie si $a_1 = 10$?
```
