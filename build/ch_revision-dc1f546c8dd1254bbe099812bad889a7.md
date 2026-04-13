---
title: Révision pour l'examen final
kernelspec:
  name: python3
  display_name: Python 3
---

# Révision pour l'examen final

Le cours suit un fil conducteur : comment estimer une fonction à partir de données, et comment s'assurer qu'elle généralise à de nouvelles observations. Chaque méthode que nous avons étudiée répond à une limite de la précédente.

Les méthodes non paramétriques (k-ppv, fenêtres de Parzen) sont les plus simples : elles consultent directement les données sans paramètres. Mais elles souffrent du fléau de la dimensionnalité. Pour passer à l'échelle, il faut s'engager sur un modèle paramétrique : la régression linéaire, la régression logistique. Ces modèles linéaires sont limités à des relations linéaires dans l'espace d'entrée. L'expansion de caractéristiques permet de contourner cette limite, mais le choix des caractéristiques est fait à la main. Les réseaux de neurones résolvent ce problème en apprenant les caractéristiques eux-mêmes, par composition de transformations non linéaires. Pour les entraîner, il faut pouvoir différentier des compositions arbitraires : c'est le rôle de la dérivation automatique. En profondeur, le gradient tend à disparaître, ce qui motive ReLU, la normalisation par lots et les connexions résiduelles. Pour les séquences, les RNN déplient le réseau dans le temps, mais le gradient y disparaît aussi et le traitement est séquentiel. Les transformeurs éliminent ces deux limites grâce au mécanisme d'attention, qui donne à chaque position un accès direct à toutes les autres. Pour les données tabulaires, les méthodes d'ensemble à base d'arbres (gradient boosting, forêts aléatoires) restent compétitives.

## Partie I — Rappel de la matière pré-intra

### Méthodes non paramétriques (intro)

Les méthodes non paramétriques conservent les données d'entraînement et les consultent au moment de la prédiction. Les trois variantes à retenir forment une progression : voisinage dur → noyau lisse → moyenne pondérée par noyau.

```{admonition} k-ppv, KDE et Nadaraya-Watson
:class: tip

**k plus proches voisins (k-ppv)** : pour prédire en $\mathbf{x}$, trouver les $k$ points les plus proches et voter (classification) ou moyenner (régression).

$$\hat{y}(\mathbf{x}) = \frac{1}{k} \sum_{i \in \mathcal{N}_k(\mathbf{x})} y_i$$

- $k$ petit → variance élevée (frontière irrégulière), $k$ grand → biais élevé (frontière trop lisse)
- Complexité de prédiction : $O(Nd)$ par requête (force brute)
- Fléau de la dimensionnalité : en haute dimension, tous les points deviennent équidistants et le voisinage perd son sens

**Estimation de densité par noyaux (KDE / fenêtres de Parzen)** : estimer la densité $p(\mathbf{x})$ en plaçant un noyau $K_\lambda$ sur chaque observation.

$$\hat{p}(\mathbf{x}) = \frac{1}{N} \sum_{i=1}^{N} K_\lambda(\mathbf{x} - \mathbf{x}_i)$$

- Le noyau gaussien $K_\lambda(u) = \frac{1}{\lambda\sqrt{2\pi}}\exp\!\left(-\frac{u^2}{2\lambda^2}\right)$ est le plus courant
- La largeur de bande $\lambda$ joue le même rôle que $k$ : $\lambda$ petit → estimée variable, $\lambda$ grand → estimée lisse

**Nadaraya-Watson** : régression par moyenne pondérée par noyau.

$$\hat{y}(\mathbf{x}) = \frac{\sum_{i=1}^N K_\lambda(\mathbf{x} - \mathbf{x}_i)\, y_i}{\sum_{i=1}^N K_\lambda(\mathbf{x} - \mathbf{x}_i)}$$

Chaque point contribue proportionnellement à sa proximité. Les poids $w_i(\mathbf{x}) = K_\lambda(\mathbf{x} - \mathbf{x}_i) / \sum_j K_\lambda(\mathbf{x} - \mathbf{x}_j)$ sont normalisés et somment à 1. Le lien avec l'attention dans les transformeurs : l'attention est une version de Nadaraya-Watson où le noyau $q^\top k / \sqrt{d_k}$ est appris.
```

Les méthodes non paramétriques souffrent du fléau de la dimensionnalité : en haute dimension, la notion de voisinage perd son sens. Pour passer à l'échelle, il faut formaliser ce que signifie « bien prédire » et s'engager sur une famille de fonctions paramétriques.

### Risque et prédicteur de Bayes (ch. 1)

Le cours repose sur un cadre commun : on cherche un prédicteur $f$ qui minimise le risque $\Risk(f) = \E[\ell(f(\mathbf{x}), y)]$. Comme la distribution conjointe est inconnue, on minimise à la place le risque empirique sur un ensemble d'entraînement. Le prédicteur de Bayes optimal, qui minimise le risque parmi toutes les fonctions, fixe la borne inférieure (risque de Bayes $\Risk^*$).

```{admonition} Risque et prédicteur de Bayes
:class: tip

- **Risque** : $\Risk(f) = \E[\ell(f(\mathbf{x}), y)]$ — inaccessible en pratique
- **Risque empirique** : $\hat{\Risk}(f) = \frac{1}{N}\sum_{i=1}^N \ell(f(\mathbf{x}_i), y_i)$ — ce qu'on minimise
- **Prédicteur optimal** : $f^*(\mathbf{x}) = \E[y \mid \mathbf{x}]$ (perte quadratique) ou $\arg\max_k p(y{=}k \mid \mathbf{x})$ (perte 0-1)
- **Risque de Bayes** : $\Risk^* = \Risk(f^*)$ — aucun classifieur ne peut faire mieux
```

Le cadre du risque nous dit *quoi* minimiser. Les modèles linéaires sont le premier choix de famille paramétrique : ils ont une solution en forme fermée et une interprétation probabiliste directe.

### Régression linéaire et Ridge (ch. 2)

Les moindres carrés ordinaires donnent la solution en forme fermée $\hat{\boldsymbol{\theta}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$, mais cette solution est instable quand $d \approx N$ ou $d > N$. Ridge stabilise en ajoutant $\lambda \mathbf{I}$. Via la SVD, Ridge atténue sélectivement les directions de faible variance.

```{admonition} Ridge et SVD
:class: tip

- **MCO** : $\hat{\boldsymbol{\theta}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$
- **Ridge** : $\hat{\boldsymbol{\theta}}_\lambda = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^\top \mathbf{y}$
- **Facteur de rétrécissement** : $\frac{d_j^2}{d_j^2 + \lambda} \in [0, 1]$ — les directions faibles ($d_j$ petit) sont les plus atténuées
- **EMV** : sous bruit gaussien, MCO = maximum de vraisemblance
```

### Classification (ch. 3)

En régression, la vraisemblance gaussienne mène aux moindres carrés. En classification, la vraisemblance de Bernoulli mène à l'entropie croisée et à la régression logistique, qui modélise directement $p(y \mid \mathbf{x})$. Le softmax généralise la sigmoïde au cas multiclasse. La perte associée est l'entropie croisée, dérivée du maximum de vraisemblance.

```{admonition} Régression logistique et softmax
:class: tip

- **Binaire** : $p(y{=}1 \mid \mathbf{x}) = \sigma(\boldsymbol{\theta}^\top \mathbf{x})$ où $\sigma(z) = 1/(1+e^{-z})$
- **Multiclasse** : $p(y{=}k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_j e^{z_j}}$ (softmax)
- **Invariance** : $\text{softmax}(\mathbf{z}+c) = \text{softmax}(\mathbf{z})$
- **Perte** : entropie croisée $= -\sum_k y_k \log \hat{p}_k$
```

### Généralisation (ch. 4)

Un modèle linéaire peut être trop simple pour les données, ou trop complexe si l'espace de caractéristiques est enrichi. Le compromis biais-variance structure la première moitié du cours. Un modèle trop simple sous-apprend (biais élevé); un modèle trop complexe surapprend (variance élevée). La validation croisée sert à choisir les hyperparamètres.

```{admonition} Décomposition biais-variance
:class: tip

$$\E\bigl[(\hat{f}(\mathbf{x}) - y)^2\bigr] = \underbrace{\bigl(\E[\hat{f}(\mathbf{x})] - f^*(\mathbf{x})\bigr)^2}_{\text{biais}^2} + \underbrace{\Var[\hat{f}(\mathbf{x})]}_{\text{variance}} + \underbrace{\sigma^2_\epsilon}_{\text{bruit}}$$

- **Sous-apprentissage** : biais élevé (modèle trop rigide, $\lambda$ trop grand)
- **Surapprentissage** : variance élevée (modèle trop flexible, $\lambda$ trop petit)
- **Expansion de caractéristiques** $\phi(\mathbf{x})$ : enrichit l'espace pour qu'un modèle linéaire capture des relations non linéaires
```

### Cadre probabiliste (ch. 5)

La régularisation contrôle la complexité, mais d'où vient-elle? Le cadre probabiliste unifie trois perspectives sur l'apprentissage. Le MAP relie la régularisation à un a priori bayésien, et la théorie de l'information fournit une troisième lecture via la divergence KL.

```{admonition} EMV, MAP et divergence KL
:class: tip

| Perspective | On minimise | Lien |
|---|---|---|
| Risque empirique | $\frac{1}{N}\sum_i \ell(f(\mathbf{x}_i), y_i)$ | Définition de départ |
| Maximum de vraisemblance | $-\frac{1}{N}\sum_i \log p(y_i \mid \mathbf{x}_i; \boldsymbol{\theta})$ | Gaussien → MCO, Bernoulli → entropie croisée |
| Théorie de l'information | $D_{\text{KL}}(\hat{p} \| p_{\boldsymbol{\theta}})$ | Minimiser la divergence KL = maximiser la vraisemblance |

- **MAP** : $\hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\max [\log p(\Data \mid \boldsymbol{\theta}) + \log p(\boldsymbol{\theta})]$ — a priori gaussien $\Rightarrow$ Ridge
- **Entropie** : $H(p) = -\sum_k p_k \log p_k$ — incertitude d'une distribution
- **KL** : $D_{\text{KL}}(P \| Q) = \sum_k P_k \log \frac{P_k}{Q_k}$ — asymétrique, pénalise $Q_k$ petit où $P_k$ est grand
```

### Modèles probabilistes génératifs (ch. 6)

La régression logistique modélise $p(y \mid \mathbf{x})$ directement (approche discriminative). Une alternative est de modéliser $p(\mathbf{x} \mid y)$ et d'appliquer Bayes (approche générative). Les modèles génératifs modélisent $p(\mathbf{x} \mid y)p(y)$ puis appliquent Bayes pour classifier, contrairement aux modèles discriminatifs qui modélisent directement $p(y \mid \mathbf{x})$. L'algorithme EM est la méthode d'estimation pour les modèles à variables latentes.

```{admonition} Modèles génératifs et EM
:class: tip

- **Naïf bayésien** : $p(\mathbf{x} \mid y) = \prod_d p(x_d \mid y)$ — indépendance conditionnelle
- **LDA** : gaussien avec $\boldsymbol{\Sigma}$ commune → frontière linéaire
- **QDA** : gaussien avec $\boldsymbol{\Sigma}_k$ par classe → frontière quadratique
- **EM** : E = calculer les responsabilités $r_{nk}$, M = mettre à jour $\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, \pi_k$
- **Mélange d'experts** : routage $p(z{=}k \mid \mathbf{x})$ dépendant de l'entrée + modèles locaux
```

---

## Partie II — Matière post-intra

Tous les modèles vus jusqu'ici reposent sur des caractéristiques choisies à la main : polynomiales, MFCC, mesures cliniques. Les réseaux de neurones apprennent leurs propres caractéristiques par composition de transformations non linéaires.

### Réseaux de neurones (ch. 7)

Un perceptron multicouche (MLP) empile des couches de la forme $\mathbf{z}_\ell = \varphi(W_\ell \mathbf{z}_{\ell-1} + \mathbf{b}_\ell)$. La couche de sortie est dictée par le maximum de vraisemblance : linéaire pour la régression, softmax pour la classification multiclasse, sigmoïde pour la classification binaire.

```{admonition} Perceptron multicouche
:class: tip

$$\mathbf{z}_\ell = \varphi(W_\ell \mathbf{z}_{\ell-1} + \mathbf{b}_\ell)$$

- **ReLU** : $\max(0, a)$ — pas de saturation pour $a > 0$, mais neurones morts si $a < 0$ en permanence
- **Couche de sortie** : linéaire ($\hat{y} = W\mathbf{z} + \mathbf{b}$) pour la régression, softmax pour la classification
- **Expressivité** : un MLP avec une couche cachée suffisamment large peut approximer toute fonction continue (théorème d'approximation universelle)
```

### Dérivation automatique (ch. 7)

La dérivation automatique (AD) calcule les dérivées exactes d'un programme en appliquant la règle de la chaîne sur son graphe de calcul.

#### Mode avant vs mode arrière

Pour une composition $f = f_L \circ \cdots \circ f_1$, le jacobien est $J_f = J_{f_L} \cdots J_{f_1}$. Les deux modes d'AD correspondent aux deux sens d'évaluation de ce produit.

```{mermaid}
graph LR
    v0["ṽ₀ = v"] -->|"J_f₁ · ṽ₀"| f1(("f₁"))
    f1 -->|"ṽ₁"| f2(("f₂"))
    f2 -->|"ṽ₂"| f3(("f₃"))
    f3 -->|"ṽ₃ = J_f · v"| out["résultat"]

    style v0 fill:none,stroke:none
    style out fill:none,stroke:none
    style f1 fill:#dae8fc,stroke:#6c8ebf
    style f2 fill:#dae8fc,stroke:#6c8ebf
    style f3 fill:#dae8fc,stroke:#6c8ebf
    linkStyle 0,1,2,3 stroke:#1f77b4,stroke-width:2px
```

```{mermaid}
graph RL
    out["ℒ"] -->|"ū₃ = 1"| f3(("f₃"))
    f3 -->|"ū₂"| f2(("f₂"))
    f2 -->|"ū₁"| f1(("f₁"))
    f1 -->|"ū₀ = ū₁ · J_f₁"| L["∇ℒ"]

    style L fill:none,stroke:none
    style out fill:none,stroke:none
    style f1 fill:#f8cecc,stroke:#b85450
    style f2 fill:#f8cecc,stroke:#b85450
    style f3 fill:#f8cecc,stroke:#b85450
    linkStyle 0,1,2,3 stroke:#d62728,stroke-width:2px
```

```{admonition} Mode avant vs mode arrière
:class: tip

- **Mode avant (JVP)** : propage un tangent $\mathbf{v}$ de gauche à droite → calcule $J_f \cdot \mathbf{v}$ en 1 passe
- **Mode arrière (VJP)** : propage un adjoint $\bar{\mathbf{u}}$ de droite à gauche → calcule $\bar{\mathbf{u}}^\top J_f$ en 1 passe
- Pour $\mathcal{L} : \R^n \to \R$ (1 sortie, $n$ paramètres), le mode arrière donne $\nabla_{\boldsymbol{\theta}} \mathcal{L}$ en **une seule passe**, peu importe $n$. Le mode avant nécessiterait $n$ passes.
```

#### Graphe de calcul et branchement

Un programme se décompose en un DAG où chaque nœud est une opération élémentaire. Lorsqu'une variable a plusieurs successeurs (branchement), le mode arrière **accumule** les adjoints de chaque chemin.

$$\bar{u} = \sum_{v \in \text{succ}(u)} \bar{v} \cdot D_u \varphi_v$$

#### Règles VJP

Chaque opération a une règle VJP qui calcule $\bar{\mathbf{u}}^\top J_f$ **sans former le jacobien** :

```{admonition} Règles VJP
:class: tip

| Opération | VJP : $\bar{\mathbf{u}}^\top \cdot J_f$ | Coût |
|---|---|---|
| Affine $W\mathbf{z} + \mathbf{b}$, par rapport à $\mathbf{z}$ | $\bar{\mathbf{u}}^\top W$ | $O(mn)$ |
| Affine, par rapport à $W$ | $\bar{\mathbf{u}}\mathbf{z}^\top$ (produit extérieur) | $O(mn)$ |
| Affine, par rapport à $\mathbf{b}$ | $\bar{\mathbf{u}}$ | $O(m)$ |
| Activation $\varphi(\mathbf{a})$ | $\bar{\mathbf{u}} \odot \varphi'(\mathbf{a})$ | $O(m)$ ← pas $O(m^2)$ ! |
| Somme $s = \sum_i x_i$ | $\bar{u} \cdot \mathbf{1}$ | $O(n)$ |
| Multiplication $a \cdot b$ | $(\bar{u} \cdot b,\; \bar{u} \cdot a)$ | $O(1)$ |
| ReLU $\max(0, a)$ | $\bar{u} \cdot \indicator(a > 0)$ | $O(1)$ |
```

#### Le traceur

En pratique (JAX, PyTorch), un *traceur* encapsule chaque valeur numérique et enregistre les opérations sur une bande de Wengert. Chaque entrée de la bande contient une *fermeture* (closure) qui capture les valeurs de la passe avant et la règle VJP. La passe arrière parcourt la bande en sens inverse et accumule les adjoints.

```{admonition} Le traceur
:class: tip

1. **Passe avant** : exécuter le programme avec des traceurs au lieu de nombres → construire la bande
2. **Initialiser** : $\bar{\mathcal{L}} = 1$ (l'adjoint de la sortie)
3. **Passe arrière** : parcourir la bande en sens inverse, appliquer chaque règle VJP, accumuler les adjoints aux branchements
```

### Entraîner un réseau de neurones (ch. 8)

#### Optimiseurs

```{admonition} Optimiseurs
:class: tip

| | SGD | Momentum | Adam |
|---|---|---|---|
| **Mise à jour** | $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \hat{\mathbf{g}}$ | $\mathbf{m} \leftarrow \beta \mathbf{m} + \hat{\mathbf{g}}$ | $\mathbf{m}, \mathbf{s}$ + correction de biais |
| **Idée** | Gradient brut | Moyenne mobile des gradients | Momentum + normalisation par dimension |
| **Hyperparamètres** | $\eta$ | $\eta, \beta \approx 0{,}9$ | $\eta \approx 10^{-3}, \beta_1 = 0{,}9, \beta_2 = 0{,}999$ |
| **Usage** | Rarement seul | Vision (avec schedule) | **Choix par défaut** |

**Adam** : $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \dfrac{\hat{\mathbf{m}}_{t+1}}{\sqrt{\hat{\mathbf{s}}_{t+1}} + \epsilon}$, avec $\hat{\mathbf{m}} = \frac{\mathbf{m}}{1 - \beta_1^t}$, $\hat{\mathbf{s}} = \frac{\mathbf{s}}{1 - \beta_2^t}$
```

#### Dissolution et explosion du gradient

Dans un réseau à $L$ couches, le gradient par rapport aux premières couches est un produit de jacobiens. Si le rayon spectral de chaque jacobien est $< 1$, le gradient disparaît exponentiellement.

```{admonition} Dissolution du gradient et solutions
:class: tip

**Cause** : $\dfrac{\partial \mathcal{L}}{\partial \mathbf{z}_1} = \dfrac{\partial \mathcal{L}}{\partial \mathbf{z}_L} \prod_{\ell=2}^{L} \underbrace{\text{diag}(\varphi'(\mathbf{a}_\ell)) W_\ell}_{J_\ell}$ — produit de $L{-}1$ jacobiens

| Solution | Mécanisme |
|---|---|
| **ReLU** | $\varphi'(a) = 1$ pour $a > 0$ → pas de saturation |
| **Initialisation He** | $W_{ij} \sim \mathcal{N}(0, 2/n)$ → variance stable couche après couche |
| **Normalisation par lots** | $\tilde{a}_j = \gamma \dfrac{a_j - \bar{a}_j}{\sqrt{s_j^2 + \epsilon}} + \beta$ → pré-activations centrées et normalisées |
| **Connexions résiduelles** | $\mathbf{z}_{\ell+1} = \mathbf{z}_\ell + f(\mathbf{z}_\ell)$ → jacobien $I + \frac{\partial f}{\partial \mathbf{z}_\ell}$, chemin direct via $I$ |
```

:::{figure} _static/residual_block.svg
:name: fig-revision-residual
:align: center
:width: 50%

Bloc résiduel : le terme identité garantit un chemin direct pour le gradient.
:::

#### Régularisation

```{admonition} Régularisation
:class: tip

| Technique | Mécanisme | En pratique |
|---|---|---|
| **Arrêt précoce** | Stopper quand l'erreur de validation remonte | Patience de 5-20 époques |
| **Dropout** | Désactiver chaque neurone avec probabilité $p$, normaliser par $1/(1{-}p)$ | $p = 0{,}1$–$0{,}5$ selon la couche |
| **Décroissance des poids** | $\boldsymbol{\theta} \leftarrow (1-\eta\lambda)\boldsymbol{\theta} - \eta \hat{\mathbf{m}}/(\sqrt{\hat{\mathbf{s}}}+\epsilon)$ | AdamW sépare optimisation et régularisation |
```

#### Transfert de représentations

Un réseau se décompose en **tronc** (caractéristiques générales) et **tête** (spécifique à la tâche). Deux stratégies : geler le tronc et entraîner la tête (extraction de caractéristiques), ou régler finement le réseau entier avec un taux d'apprentissage réduit.

:::{figure} _static/trunk_head_architecture.svg
:name: fig-revision-trunk-head
:align: center
:width: 60%

Architecture tronc-tête : le tronc pré-entraîné extrait des caractéristiques générales.
:::

### Auto-encodeurs (ch. 9)

Un auto-encodeur comprime l'entrée via un goulot d'étranglement puis la reconstruit. L'auto-encodeur linéaire est exactement l'ACP.

```{admonition} Auto-encodeurs
:class: tip

$$\mathcal{L} = \frac{1}{N}\sum_{n=1}^N \|\mathbf{x}_n - g_\psi(f_\phi(\mathbf{x}_n))\|^2$$

- **Linéaire** = ACP : projette sur les $L$ premiers vecteurs propres. Erreur $= \sum_{k=L+1}^D \lambda_k$
- **Non linéaire** : capte des variétés courbes que l'ACP (sous-espace linéaire) ne peut pas représenter
- **Débruiteur** : reçoit $\tilde{\mathbf{x}}$ corrompu, reconstruit $\mathbf{x}$ propre → force à capter la structure
```

### Réseaux récurrents (ch. 9)

Les RNN traitent des séquences en maintenant un état caché $\mathbf{h}_t$. Le partage de paramètres à travers le temps permet de traiter des séquences de longueur variable.

```{mermaid}
graph LR
    x1["x₁"] --> h1["h₁"]
    x2["x₂"] --> h2["h₂"]
    x3["x₃"] --> h3["h₃"]
    xT["x_T"] --> hT["h_T"]
    h0["h₀ = 0"] --> h1
    h1 --> h2
    h2 --> h3
    h3 -.->|"···"| hT
    h1 --> y1["y₁"]
    h2 --> y2["y₂"]
    h3 --> y3["y₃"]
    hT --> yT["y_T"]

    style h0 fill:#f0f0f0,stroke:#999
    style h1 fill:#dae8fc,stroke:#6c8ebf
    style h2 fill:#dae8fc,stroke:#6c8ebf
    style h3 fill:#dae8fc,stroke:#6c8ebf
    style hT fill:#dae8fc,stroke:#6c8ebf
    style x1 fill:#fff2cc,stroke:#d6b656
    style x2 fill:#fff2cc,stroke:#d6b656
    style x3 fill:#fff2cc,stroke:#d6b656
    style xT fill:#fff2cc,stroke:#d6b656
    style y1 fill:#d5e8d4,stroke:#82b366
    style y2 fill:#d5e8d4,stroke:#82b366
    style y3 fill:#d5e8d4,stroke:#82b366
    style yT fill:#d5e8d4,stroke:#82b366
```

```{admonition} RNN, LSTM et dissolution du gradient
:class: tip

**Mise à jour** : $\mathbf{h}_t = \varphi(W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t + \mathbf{b}_h)$

**BPTT** : le gradient implique $\prod_j \text{diag}(\varphi'(\mathbf{a}_j))\, W_{hh}$ → dissolution si valeurs propres de $W_{hh}$ < 1

**LSTM** : chemin linéaire via l'état de cellule $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$
- Quand $\mathbf{f}_t \approx 1$ : le gradient passe sans atténuation (analogie : connexion résiduelle)

**GRU** : variante simplifiée avec 2 portes (mise à jour $\mathbf{z}_t$, réinitialisation $\mathbf{r}_t$) au lieu de 3

**Limite** : traitement séquentiel → pas de parallélisation; contexte comprimé dans un vecteur fixe $\mathbf{h}_t$
```

### Attention et transformeurs (ch. 10)

Dans le mécanisme d'attention, chaque position accède directement à toutes les autres via des requêtes, clés et valeurs apprises, sans passer par un état caché séquentiel.

:::{figure} _static/attention_mechanism.svg
:name: fig-revision-attention
:align: center
:width: 70%

Attention par produit scalaire : les requêtes interrogent les clés pour pondérer les valeurs.
:::

```{admonition} Attention par produit scalaire
:class: tip

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

- $Q = XW_Q$, $K = XW_K$, $V = XW_V$ — projections apprises
- $1/\sqrt{d_k}$ — empêche la saturation du softmax pour $d_k$ grand
- **Multi-têtes** : $H$ attentions en parallèle, concaténées → $\text{Concat}(\text{head}_1, \ldots, \text{head}_H) W_O$
- **Lien Nadaraya-Watson** : l'attention est une régression à noyau avec noyau appris $q^\top k / \sqrt{d_k}$
```

:::{figure} _static/transformer_block.svg
:name: fig-revision-transformer-block
:align: center
:width: 50%

Un bloc transformeur : auto-attention → résiduel → LayerNorm → MLP → résiduel → LayerNorm.
:::

```{admonition} Masque causal et KV cache
:class: tip

- **Encodage positionnel** : $p_{t,2j} = \sin(t/10000^{2j/d})$, $p_{t,2j+1} = \cos(t/10000^{2j/d})$ — injecte l'ordre
- **Masque causal** (décodeur) : $M_{ij} = -\infty$ pour $j > i$ → chaque position ne voit que le passé
- **KV cache** : stocker $K, V$ des positions passées → coût de génération de $O(T^3 d)$ à $O(T^2 d)$
- **Coût quadratique** : la matrice $QK^\top \in \R^{T \times T}$ coûte $O(T^2 d_k)$ en calcul et $O(T^2)$ en mémoire
```

### Arbres et ensembles (ch. 11)

La descente de gradient fonctionnelle optimise les prédictions elles-mêmes plutôt que les paramètres. Les arbres de décision servent de modèles de base.

```{admonition} Gradient boosting et ensembles
:class: tip

**Gradient boosting** : à l'itération $m$, entraîner $F_m$ sur les pseudo-résidus $r_{im} = -\frac{\partial \ell}{\partial f(\mathbf{x}_i)}\big|_{f_{m-1}}$, puis $f_m = f_{m-1} + \nu F_m$

Pour la perte quadratique : $r_{im} = y_i - f_{m-1}(\mathbf{x}_i)$ (résidus ordinaires)

| | **Boosting** | **Bagging** (forêts aléatoires) |
|---|---|---|
| Réduit | le **biais** | la **variance** |
| Comment | Modèles faibles séquentiels qui corrigent les erreurs | Moyennage de $M$ arbres sur des échantillons bootstrap |
| Risque | Surapprentissage si $M$ trop grand | Robuste, peu sensible à $M$ |
```

---

## Questions de pratique

### Question 1 — Quel modèle pour quel problème?

```{admonition} Énoncé
:class: note

Modèles disponibles : MLP, RNN, LSTM, Transformeur, Arbre de décision, Forêt aléatoire, Gradient boosting, Auto-encodeur, Régression logistique.

**(a)** Un hôpital dispose de 5000 dossiers patients avec 30 variables structurées (âge, tension, résultats sanguins, etc.) et veut prédire le risque de réadmission. L'interprétabilité des facteurs de risque est importante.

**(b)** Une équipe de traduction automatique veut entraîner un modèle sur des millions de paires de phrases. Le temps d'entraînement est un facteur critique et l'équipe dispose de plusieurs GPU.

**(c)** Un système de détection d'anomalies reçoit des lectures de capteurs sur une turbine. L'objectif est d'apprendre une représentation compacte du fonctionnement normal (sans étiquettes de pannes) pour signaler les déviations.

**(d)** Un assistant vocal doit transcrire des commandes en texte. Le signal audio arrive séquentiellement et chaque trame dépend fortement du contexte local récent, mais les séquences sont courtes (< 20 trames).
```

```{admonition} Solution
:class: hint dropdown

**(a)** **Gradient boosting** (ou forêt aléatoire). Données tabulaires structurées → les méthodes d'ensemble à base d'arbres sont l'état de l'art. L'importance des caractéristiques (par permutation ou diminution d'impureté) fournit l'interprétabilité demandée.

**(b)** **Transformeur** (encodeur-décodeur). Le mécanisme d'attention permet la parallélisation sur GPU (contrairement aux RNN séquentiels). L'attention croisée entre l'encodeur (langue source) et le décodeur (langue cible) est le mécanisme standard en traduction.

**(c)** **Auto-encodeur**. Apprentissage non supervisé → pas d'étiquettes. L'auto-encodeur apprend à reconstruire le fonctionnement normal; une erreur de reconstruction élevée signale une anomalie.

**(d)** **LSTM** (ou GRU). Séquences courtes → la limite du traitement séquentiel n'est pas un problème. Les dépendances temporelles locales sont bien captées par les portes du LSTM. Un transformeur serait surdimensionné pour 20 trames.
```

### Question 2 — Dérivation automatique

```{admonition} Énoncé
:class: note

Soit le programme :

~~~python
def f(x, y):
    a = x * y       # multiplication
    b = a + x       # addition
    c = relu(b)     # ReLU
    return c
~~~

**(a)** Dessinez le graphe de calcul en identifiant chaque nœud intermédiaire.

**(b)** Pour $x = 2, y = 3$, calculez $\frac{\partial c}{\partial x}$ et $\frac{\partial c}{\partial y}$ par le mode arrière. Montrez chaque étape en partant de $\bar{c} = 1$.

**(c)** La variable $x$ a deux successeurs ($a$ et $b$). Comment le mode arrière gère-t-il cette situation?

**(d)** Quel mode (avant ou arrière) est le plus efficace pour calculer $J_f \cdot (1, 0)^\top$? Que calcule-t-on concrètement?
```

```{admonition} Solution
:class: hint dropdown

**(a)** Graphe de calcul :

$$x, y \;\xrightarrow{\times}\; a = xy \;\xrightarrow{+x}\; b = a + x \;\xrightarrow{\text{relu}}\; c$$

La variable $x$ a deux arêtes sortantes : l'une vers le nœud $\times$ (pour calculer $a$) et l'autre vers le nœud $+$ (pour calculer $b$).

**(b)** Passe avant : $a = 2 \times 3 = 6$, $b = 6 + 2 = 8$, $c = \text{relu}(8) = 8$.

Passe arrière (adjoints) :

| Étape | Règle VJP | Résultat |
|---|---|---|
| $\bar{c} = 1$ | (initialisation) | $\bar{c} = 1$ |
| $\bar{b} = \bar{c} \cdot \indicator(b > 0) = 1 \cdot 1$ | VJP de relu | $\bar{b} = 1$ |
| $\bar{a} = \bar{b} \cdot 1 = 1$ | VJP de add (par rapport à $a$) | $\bar{a} = 1$ |
| $\bar{x}_{\text{via } b} = \bar{b} \cdot 1 = 1$ | VJP de add (par rapport à $x$) | contribution = 1 |
| $\bar{x}_{\text{via } a} = \bar{a} \cdot y = 1 \cdot 3 = 3$ | VJP de mul (par rapport à $x$) | contribution = 3 |
| $\bar{x} = 1 + 3 = 4$ | **accumulation** | $\frac{\partial c}{\partial x} = 4$ |
| $\bar{y} = \bar{a} \cdot x = 1 \cdot 2 = 2$ | VJP de mul (par rapport à $y$) | $\frac{\partial c}{\partial y} = 2$ |

**(c)** Lorsqu'une variable a plusieurs successeurs, le mode arrière **accumule** les adjoints par sommation : $\bar{x} = \bar{x}_{\text{via } a} + \bar{x}_{\text{via } b} = 3 + 1 = 4$.

**(d)** Le **mode avant** est plus efficace. Le vecteur $(1, 0)^\top$ est le vecteur tangent unitaire en $x$ : on calcule $\frac{\partial f}{\partial x}$ directement (une dérivée directionnelle). Avec 2 entrées et 1 sortie, le mode avant nécessite 2 passes et le mode arrière 1 passe → le mode arrière est en fait légèrement mieux ici. Mais si l'on ne cherche que $\partial f / \partial x$, le mode avant avec $\mathbf{v} = (1, 0)^\top$ donne la réponse en une seule passe, sans calculer $\partial f / \partial y$.
```

### Question 3 — Compréhension de code

```{admonition} Énoncé
:class: note

~~~python
def step(params, x_seq):
    W_hh, W_xh, b_h, W_hy, b_y = params
    h = np.zeros(W_hh.shape[0])
    outputs = []
    for x_t in x_seq:
        h = np.tanh(W_hh @ h + W_xh @ x_t + b_h)
        y_t = W_hy @ h + b_y
        outputs.append(y_t)
    return outputs
~~~

**(a)** Quel type de réseau cette fonction implémente-t-elle? Identifiez les éléments du code qui le confirment.

**(b)** Si `x_seq` contient $T = 100$ pas de temps, pourquoi le gradient par rapport à $W_{hh}$ peut-il devenir problématique?

**(c)** Proposez deux modifications architecturales concrètes qui atténuent ce problème.
```

```{admonition} Solution
:class: hint dropdown

**(a)** Un **réseau récurrent (RNN)**. Indices : (1) `h = np.tanh(W_hh @ h + ...)` — l'état caché `h` est mis à jour récursivement en fonction de lui-même ($W_{hh}$) et de l'entrée courante ($W_{xh} @ x_t$); (2) la boucle `for x_t in x_seq` parcourt la séquence pas à pas; (3) les paramètres $W_{hh}, W_{xh}$ sont partagés à tous les pas de temps.

**(b)** **Dissolution du gradient.** Pour $T = 100$, la rétropropagation à travers le temps (BPTT) produit un produit de 99 jacobiens $\prod_{j} \text{diag}(\tanh'(\mathbf{a}_j)) W_{hh}$. Comme $\tanh'(a) \leq 1$, ce produit décroît exponentiellement avec $T$ si les valeurs propres de $W_{hh}$ sont inférieures à 1. Les premières positions de la séquence ne reçoivent presque aucun signal de gradient.

**(c)** (1) Remplacer le RNN par un **LSTM** : l'état de cellule $\mathbf{c}_t$ offre un chemin quasi-linéaire pour le gradient. (2) Ajouter de l'**écrêtage de gradient** (gradient clipping) pour prévenir l'explosion du gradient, combiné avec l'utilisation de **connexions résiduelles** entre couches si le réseau a plusieurs couches empilées.
```

### Question 4 — Attention et transformeurs

```{admonition} Énoncé
:class: note

Un transformeur traite $T = 4$ jetons avec $d_k = 2$. Après projection :

$$Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{pmatrix}, \quad K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 1 & 1 \end{pmatrix}$$

**(a)** Calculez la matrice de scores $S = QK^\top / \sqrt{d_k}$.

**(b)** Si ce transformeur est un décodeur (modèle de langage), écrivez la matrice de scores après application du masque causal.

**(c)** Quel est le coût du mécanisme d'attention en fonction de $T$ et $d_k$?

**(d)** Expliquez le KV cache et donnez les complexités avec et sans cache pour générer $T$ jetons.
```

```{admonition} Solution
:class: hint dropdown

**(a)**

$$QK^\top = \begin{pmatrix} 1 & 0 & 0 & 1 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 2 \\ 0 & 0 & 0 & 0 \end{pmatrix}, \quad S = \frac{QK^\top}{\sqrt{2}} = \begin{pmatrix} 0{,}71 & 0 & 0 & 0{,}71 \\ 0 & 0{,}71 & 0 & 0{,}71 \\ 0{,}71 & 0{,}71 & 0 & 1{,}41 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

**(b)** Le masque causal fixe à $-\infty$ les positions $j > i$ (au-dessus de la diagonale) :

$$S_{\text{masqué}} = \begin{pmatrix} 0{,}71 & -\infty & -\infty & -\infty \\ 0 & 0{,}71 & -\infty & -\infty \\ 0{,}71 & 0{,}71 & 0 & -\infty \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

**(c)** Le calcul de $QK^\top$ coûte $O(T^2 d_k)$. La matrice de scores est de taille $T \times T$, donc $O(T^2)$ en mémoire. Pour les longues séquences ($T$ grand), le coût quadratique domine.

**(d)** **KV cache** : à chaque pas de génération $t$, on ne calcule que $\mathbf{q}_t$ (une ligne) et on la multiplie par les clés stockées $K_{1:t}$. On évite de recalculer $K$ et $V$ pour les positions passées.

- **Sans cache** : au pas $t$, on recalcule $Q, K, V$ pour les $t$ positions → coût $O(t^2 d)$ par pas → total $O(T^3 d)$
- **Avec cache** : au pas $t$, un seul vecteur requête contre $t$ clés → coût $O(t d)$ par pas → total $O(T^2 d)$
```

Matrice de scores et version masquée pour cet exemple :

```{code-cell} python
:tags: [hide-input]
import numpy as np
import matplotlib.pyplot as plt

Q = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
K = np.array([[1, 0], [0, 1], [0, 0], [1, 1]], dtype=float)
d_k = 2

S = Q @ K.T / np.sqrt(d_k)

mask = np.triu(np.full((4, 4), -np.inf), k=1)
S_masked = S + mask

fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

for ax, mat, title in [(axes[0], S, "Scores $S = QK^\\top / \\sqrt{d_k}$"),
                        (axes[1], S_masked, "Scores avec masque causal")]:
    display_mat = np.where(np.isneginf(mat), np.nan, mat)
    im = ax.imshow(display_mat, cmap="YlOrRd", vmin=-0.5, vmax=1.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Clé (position $j$)")
    ax.set_ylabel("Requête (position $i$)")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    for i in range(4):
        for j in range(4):
            val = mat[i, j]
            if np.isneginf(val):
                ax.text(j, i, "$-\\infty$", ha="center", va="center", fontsize=9, color="gray")
            else:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.show()
```

### Question 5 — Optimisation et régularisation

```{admonition} Énoncé
:class: note

**(a)** Un étudiant entraîne un MLP à 20 couches avec des activations sigmoïdes et observe que les gradients des premières couches sont quasi nuls. Expliquez la cause et proposez deux modifications.

**(b)** Expliquez la différence entre la régularisation $L_2$ et la décroissance des poids avec Adam. Pourquoi cette distinction est-elle importante?

**(c)** L'erreur d'entraînement diminue, mais l'erreur de validation remonte après l'époque 30. Nommez deux techniques de régularisation et expliquez leur mécanisme.
```

```{admonition} Solution
:class: hint dropdown

**(a)** **Dissolution du gradient** due à la saturation de la sigmoïde : $\sigma'(a) \leq 0{,}25$, donc après 19 couches le gradient est atténué par un facteur $(0{,}25)^{19} \approx 10^{-12}$. Modifications : (1) remplacer les sigmoïdes par des **ReLU** (dérivée 1 pour $a > 0$); (2) ajouter des **connexions résiduelles** pour créer des chemins directs de gradient.

**(b)** Avec **SGD**, les deux sont équivalentes. Avec **Adam**, elles divergent : la régularisation $L_2$ ajoute $\lambda \boldsymbol{\theta}$ au gradient *avant* la normalisation par les moments, ce qui atténue l'effet pour les paramètres à gradient élevé. La **décroissance des poids** (AdamW) applique la contraction $(1-\eta\lambda)\boldsymbol{\theta}$ *directement*, sans passer par les moments adaptatifs. AdamW est préféré car il régularise de manière uniforme, indépendamment de l'historique des gradients.

**(c)** Surapprentissage classique. (1) **Arrêt précoce** : stopper l'entraînement à l'époque où l'erreur de validation est minimale (ici, vers l'époque 30). Limite implicitement la complexité du modèle. (2) **Dropout** : désactive aléatoirement une fraction $p$ des neurones à chaque itération, empêchant la co-adaptation et forçant le réseau à apprendre des caractéristiques robustes.
```

### Question 6 — Auto-encodeurs et représentations

```{admonition} Énoncé
:class: note

**(a)** Un auto-encodeur linéaire avec un goulot de dimension $L = 2$ est entraîné sur des données de dimension $D = 10$. Les valeurs propres de la matrice de covariance sont $\lambda_1 = 5, \lambda_2 = 3, \lambda_3 = 1, \lambda_4 = \ldots = \lambda_{10} = 0{,}1$. Quelle est l'erreur de reconstruction minimale?

**(b)** En quoi un auto-encodeur non linéaire peut-il capter de la structure que l'ACP ne peut pas? Donnez un exemple géométrique.
```

```{admonition} Solution
:class: hint dropdown

**(a)** L'auto-encodeur linéaire optimal conserve les $L = 2$ directions de plus grande variance (valeurs propres $\lambda_1 = 5$ et $\lambda_2 = 3$). L'erreur de reconstruction minimale est la somme des valeurs propres ignorées :

$$\mathcal{L}^* = \lambda_3 + \lambda_4 + \cdots + \lambda_{10} = 1 + 7 \times 0{,}1 = 1{,}7$$

**(b)** L'ACP projette sur un **sous-espace linéaire** (plan, droite). Si les données vivent sur une variété courbe — par exemple un **demi-cercle en 2D** ou une **surface en spirale en 3D** (Swiss roll) — l'ACP ne peut pas capturer cette courbure et étale les données projetées de manière inappropriée. Un auto-encodeur non linéaire apprend une transformation courbe qui « déplie » la variété dans l'espace latent.
```

### Question 7 — Dissolution du gradient dans les RNN

```{admonition} Énoncé
:class: note

**(a)** Écrivez le jacobien $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}$ pour un RNN avec tanh. Pourquoi le produit de $T$ tels jacobiens tend-il vers zéro?

**(b)** Montrez que la mise à jour du LSTM $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$ permet au gradient de circuler quand $\mathbf{f}_t \approx 1$. Quel est le rôle analogue dans un réseau résiduel?
```

```{admonition} Solution
:class: hint dropdown

**(a)**

$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}} = \text{diag}(\tanh'(\mathbf{a}_t))\, W_{hh}$$

où $\mathbf{a}_t = W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t + \mathbf{b}_h$. Puisque $\tanh'(a) = 1 - \tanh^2(a) \in [0, 1]$, les éléments diagonaux sont au plus 1. Le produit de $T$ tels jacobiens donne $\prod_{j=1}^T \text{diag}(\tanh'(\mathbf{a}_j)) W_{hh}$. Si le rayon spectral de $\text{diag}(\tanh'(\mathbf{a}_j)) W_{hh}$ est strictement inférieur à 1 (ce qui est fréquent en pratique), le produit décroît exponentiellement vers zéro.

**(b)** Le jacobien de $\mathbf{c}_t$ par rapport à $\mathbf{c}_{t-1}$ est :

$$\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} = \text{diag}(\mathbf{f}_t) + \text{termes en } \frac{\partial \mathbf{f}_t}{\partial \mathbf{c}_{t-1}}$$

Quand $\mathbf{f}_t \approx 1$, le terme dominant est $\text{diag}(\mathbf{f}_t) \approx I$, donc $\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} \approx I$ et le gradient passe sans atténuation sur de longues séquences. On retrouve le même mécanisme dans une **connexion résiduelle** $\mathbf{z}_{\ell+1} = \mathbf{z}_\ell + f(\mathbf{z}_\ell)$, où le jacobien $I + \frac{\partial f}{\partial \mathbf{z}_\ell}$ contient un chemin identité direct.
```

### Question 8 — Gradient boosting

```{admonition} Énoncé
:class: note

**(a)** Avec la perte quadratique $\ell(y, f) = \frac{1}{2}(y - f)^2$, montrez que les pseudo-résidus se simplifient en résidus ordinaires.

**(b)** Le boosting et le bagging s'attaquent à des composantes différentes de l'erreur. Lesquelles?
```

```{admonition} Solution
:class: hint dropdown

**(a)**

$$r_{im} = -\frac{\partial \ell(y_i, f(\mathbf{x}_i))}{\partial f(\mathbf{x}_i)}\bigg|_{f = f_{m-1}} = -\frac{\partial}{\partial f}\left[\frac{1}{2}(y_i - f)^2\right]_{f = f_{m-1}} = -(-(y_i - f_{m-1}(\mathbf{x}_i))) = y_i - f_{m-1}(\mathbf{x}_i)$$

Ce sont les résidus ordinaires : la différence entre la cible et la prédiction actuelle.

**(b)** Le **boosting** réduit le **biais** : chaque modèle faible corrige séquentiellement les erreurs du modèle courant, augmentant la complexité effective du prédicteur. Le **bagging** (forêts aléatoires) réduit la **variance** : en moyennant $M$ modèles divers entraînés sur des échantillons bootstrap avec des sous-ensembles aléatoires de caractéristiques, on réduit la variabilité des prédictions. La formule $\Var[\bar{f}] = \rho \sigma^2 + \frac{1-\rho}{M}\sigma^2$ montre que le bagging est d'autant plus efficace que la corrélation $\rho$ entre les arbres est faible.
```

---

## Tableau de synthèse

Les thèmes transversaux du cours relient la matière pré-intra et post-intra :

| Thème | Pré-intra | Post-intra |
|---|---|---|
| Dissolution du gradient | — | Sigmoïde → ReLU, connexions résiduelles, LSTM |
| Régularisation | Ridge ($\lambda \|\boldsymbol{\theta}\|^2$), validation croisée | Dropout, arrêt précoce, décroissance des poids |
| Maximum de vraisemblance | MCO = EMV gaussien, logistique = EMV Bernoulli | Couche de sortie des réseaux (softmax, linéaire) |
| Réduction de dimension | SVD, ACP via Ridge | Auto-encodeur (linéaire = ACP, non linéaire) |
| Modèles à variables latentes | Mélange de gaussiennes, EM | Auto-encodeur variationnel |
| Moyenne pondérée | Nadaraya-Watson (noyau fixe), mélange d'experts (routage appris) | Attention (noyau appris $q^\top k / \sqrt{d_k}$) |
| Biais-variance | $k$ (k-ppv), $\lambda$ (KDE), $\lambda$ (Ridge), validation croisée | Boosting (biais) vs bagging (variance) |
