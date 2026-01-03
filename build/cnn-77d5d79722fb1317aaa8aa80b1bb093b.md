# Réseaux de neurones convolutifs

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Expliquer pourquoi les MLP ne sont pas adaptés au traitement d'images
- Définir l'opération de convolution et ses propriétés d'invariance
- Décrire l'architecture d'un réseau convolutif (CNN)
- Comprendre les couches de convolution, pooling et normalisation
- Implémenter la régularisation par dropout
- Discuter des architectures classiques (LeNet, VGG, ResNet)
```

## Motivation: les limites des MLP pour les images

Dans le chapitre précédent, nous avons vu comment les perceptrons multicouches (MLP) peuvent apprendre des fonctions complexes en composant plusieurs transformations non linéaires. Cependant, lorsqu'il s'agit de traiter des **images**, les MLP présentent une limitation fondamentale: ils ne sont pas **invariants à la translation**.

Considérons un exemple simple. Supposons que nous voulons détecter un motif particulier (par exemple, un visage) dans une image. Si nous utilisons un MLP classique avec une couche entièrement connectée, le vecteur de poids appris sera spécifique à une position particulière dans l'image. Si l'objet se trouve à gauche de l'image, le réseau produira une forte réponse; mais si ce même objet est décalé vers la droite, la réponse sera faible, car les poids ne correspondent plus à la nouvelle position.

Ce problème est particulièrement grave pour les images, car:
1. Le même objet peut apparaître n'importe où dans l'image
2. Le nombre de paramètres d'un MLP croît quadratiquement avec la taille de l'image
3. Chaque position doit être apprise indépendamment, ce qui nécessite énormément de données

Les **réseaux de neurones convolutifs** (CNN, pour *Convolutional Neural Networks*) résolvent ces problèmes en remplaçant la multiplication matricielle par une opération de **convolution**, qui est intrinsèquement invariante à la translation.

## L'opération de convolution

### Définition mathématique

La **convolution** entre deux fonctions $f, g: \mathbb{R}^D \rightarrow \mathbb{R}$ est définie par l'intégrale:

$$
[f \circledast g](\boldsymbol{z}) = \int_{\mathbb{R}^D} f(\boldsymbol{u}) g(\boldsymbol{z} - \boldsymbol{u}) \, d\boldsymbol{u}
$$

Intuitivement, la convolution "glisse" une fonction (le **noyau** ou **filtre**) sur l'autre et calcule la somme pondérée à chaque position. Le résultat est une mesure de la similarité locale entre les deux fonctions.

### Convolution discrète en 2D

Dans le cas des images numériques, nous travaillons avec des tableaux discrets. La convolution 2D discrète entre une image $\mathbf{X}$ et un filtre $\mathbf{W}$ de taille $H \times W$ est:

$$
[\mathbf{W} \circledast \mathbf{X}](i, j) = \sum_{u=0}^{H-1} \sum_{v=0}^{W-1} w_{u,v} \, x_{i+u, j+v}
$$

Prenons un exemple concret. Considérons la convolution d'une image $3 \times 3$ avec un noyau $2 \times 2$:

$$
\mathbf{Y} = \begin{pmatrix} w_1 & w_2 \\ w_3 & w_4 \end{pmatrix} \circledast \begin{pmatrix} x_1 & x_2 & x_3 \\ x_4 & x_5 & x_6 \\ x_7 & x_8 & x_9 \end{pmatrix}
$$

Le résultat est une image $2 \times 2$:

$$
\mathbf{Y} = \begin{pmatrix} 
w_1 x_1 + w_2 x_2 + w_3 x_4 + w_4 x_5 & w_1 x_2 + w_2 x_3 + w_3 x_5 + w_4 x_6 \\
w_1 x_4 + w_2 x_5 + w_3 x_7 + w_4 x_8 & w_1 x_5 + w_2 x_6 + w_3 x_8 + w_4 x_9
\end{pmatrix}
$$

Chaque élément de la sortie est calculé en appliquant le même filtre à une région locale de l'image. C'est cette propriété de **partage des poids** qui confère aux CNN leur invariance à la translation.

### Cartes de réponse

Lorsque nous appliquons un filtre à une image, nous obtenons une **carte de réponse** (ou *feature map*). Chaque point de cette carte indique la force de la réponse du filtre à cette position.

Par exemple, un filtre conçu pour détecter des lignes diagonales produira une carte de réponse avec des valeurs élevées aux endroits où l'image contient effectivement des lignes diagonales. Les CNN apprennent automatiquement ces filtres à partir des données, plutôt que de les concevoir manuellement.

### Convolution comme opérateur linéaire

La convolution est un **opérateur linéaire**, ce qui signifie qu'elle peut être représentée par une multiplication matricielle. En "aplatissant" l'image 2D $\mathbf{X}$ en un vecteur 1D $\boldsymbol{x}$, la convolution devient:

$$
\boldsymbol{y} = \mathbf{C} \boldsymbol{x}
$$

où $\mathbf{C}$ est une **matrice de Toeplitz** dérivée du noyau. Pour notre exemple précédent:

$$
\mathbf{C} = \begin{pmatrix}
w_1 & w_2 & 0 & w_3 & w_4 & 0 & 0 & 0 & 0 \\
0 & w_1 & w_2 & 0 & w_3 & w_4 & 0 & 0 & 0 \\
0 & 0 & 0 & w_1 & w_2 & 0 & w_3 & w_4 & 0 \\
0 & 0 & 0 & 0 & w_1 & w_2 & 0 & w_3 & w_4
\end{pmatrix}
$$

Cette représentation révèle que les CNN sont en fait des MLP avec une **structure creuse spéciale** dans les matrices de poids. Les éléments non nuls sont liés entre eux (ils correspondent aux mêmes poids du filtre), ce qui:
1. Impose l'invariance à la translation
2. Réduit massivement le nombre de paramètres

## Architecture d'un CNN

### Structure générale

Un CNN typique est composé de plusieurs types de couches empilées:

1. **Couches de convolution**: extraient des caractéristiques locales
2. **Couches d'activation**: introduisent la non-linéarité (ReLU, etc.)
3. **Couches de pooling**: réduisent la résolution spatiale
4. **Couches de normalisation**: stabilisent l'entraînement
5. **Couches entièrement connectées**: effectuent la classification finale

### Couches de convolution

Une couche de convolution applique plusieurs filtres à son entrée pour produire plusieurs cartes de réponse. Si l'entrée a $C_{in}$ canaux (par exemple, 3 pour une image RGB), chaque filtre a également $C_{in}$ canaux, et la sortie a $C_{out}$ canaux (un par filtre).

Paramètres importants:
- **Taille du noyau** (*kernel size*): dimensions spatiales du filtre (ex: $3 \times 3$)
- **Pas** (*stride*): décalage entre positions successives du filtre
- **Rembourrage** (*padding*): ajout de zéros autour de l'image pour contrôler la taille de sortie

### Couches de pooling

Lors de la classification d'images, nous voulons souvent savoir simplement si un objet est présent, sans nous soucier de sa position exacte. Les couches de **pooling** réduisent la résolution spatiale en agrégeant l'information locale.

Deux variantes principales existent:

1. **Pooling maximum** (*max pooling*): prend le maximum sur une fenêtre locale
   $$
   y_{i,j} = \max_{(u,v) \in \mathcal{R}_{i,j}} x_{u,v}
   $$

2. **Pooling moyen** (*average pooling*): calcule la moyenne sur une fenêtre locale
   $$
   y_{i,j} = \frac{1}{|\mathcal{R}_{i,j}|} \sum_{(u,v) \in \mathcal{R}_{i,j}} x_{u,v}
   $$

Le pooling maximum est généralement préféré car il préserve les caractéristiques les plus saillantes et introduit une certaine invariance aux petites translations.

### Couches de normalisation

Pour stabiliser l'entraînement des réseaux profonds, on utilise des couches de **normalisation**. La plus courante est la **normalisation par lots** (*batch normalization*).

Le problème fondamental est le **décalage covariant interne** (*internal covariate shift*): les distributions des activations des couches internes changent constamment pendant l'entraînement, ce qui rend l'optimisation difficile.

La normalisation par lots normalise les pré-activations $\boldsymbol{z}_n$ pour chaque mini-lot $\mathcal{B}$:

$$
\hat{\boldsymbol{z}}_n = \frac{\boldsymbol{z}_n - \boldsymbol{\mu}_{\mathcal{B}}}{\sqrt{\boldsymbol{\sigma}_{\mathcal{B}}^2 + \epsilon}}
$$

où:
- $\boldsymbol{\mu}_{\mathcal{B}} = \frac{1}{|\mathcal{B}|} \sum_{\boldsymbol{z} \in \mathcal{B}} \boldsymbol{z}$ est la moyenne sur le mini-lot
- $\boldsymbol{\sigma}_{\mathcal{B}}^2 = \frac{1}{|\mathcal{B}|} \sum_{\boldsymbol{z} \in \mathcal{B}} (\boldsymbol{z} - \boldsymbol{\mu}_{\mathcal{B}})^2$ est la variance
- $\epsilon$ est une petite constante pour la stabilité numérique

On ajoute ensuite des paramètres apprenables $\boldsymbol{\gamma}$ et $\boldsymbol{\beta}$ pour redonner de l'expressivité au réseau:

$$
\tilde{\boldsymbol{z}}_n = \boldsymbol{\gamma} \odot \hat{\boldsymbol{z}}_n + \boldsymbol{\beta}
$$

Pendant l'entraînement, les statistiques sont calculées sur chaque mini-lot. Pour l'inférence, on utilise des moyennes mobiles calculées sur l'ensemble d'entraînement.

## Régularisation: Dropout

### Le problème du surapprentissage

Les réseaux de neurones profonds ont une grande capacité et peuvent facilement surapprendre les données d'entraînement. Plusieurs techniques de régularisation sont utilisées pour améliorer la généralisation.

### Régularisation $\ell_2$

Comme en régression ridge, nous pouvons imposer un **a priori gaussien** sur les poids:
$$
p(\boldsymbol{w}) = \mathcal{N}(\boldsymbol{w} \mid \mathbf{0}, \alpha^2 \mathbf{I})
$$

Cela revient à ajouter un terme de pénalisation $\ell_2$ à la fonction de perte:
$$
\mathcal{L}_{reg} = \mathcal{L} + \lambda \|\boldsymbol{w}\|_2^2
$$

Cette technique est aussi appelée **dégradation des poids** (*weight decay*).

### Dropout

Une approche plus radicale est le **dropout**, qui désactive aléatoirement certaines unités pendant l'entraînement. À chaque passage, chaque neurone est désactivé avec une probabilité $p$ (typiquement $p = 0.5$).

Mathématiquement, si $\epsilon_{li} \sim \text{Ber}(1-p)$, alors les poids effectifs deviennent:
$$
\theta_{lij} = w_{lij} \, \epsilon_{li}
$$

Si $\epsilon_{li} = 0$, tous les poids sortants de l'unité $i$ dans la couche $l-1$ vers n'importe quelle unité $j$ dans la couche $l$ seront nuls.

Le dropout peut être interprété comme:
1. Une régularisation qui empêche la **co-adaptation complexe** des unités cachées
2. Un entraînement implicite d'un **ensemble** de sous-réseaux

### Considérations pratiques

Plusieurs points techniques sont importants:

1. **Mise à l'échelle à l'inférence**: Pour que les activations aient la même espérance pendant le test, on multiplie les poids par $(1-p)$ avant de faire des prédictions (ou de manière équivalente, on divise par $(1-p)$ pendant l'entraînement).

2. **Désactivation en test**: Normalement, le dropout est désactivé pendant l'inférence pour obtenir des prédictions déterministes.

3. **Monte Carlo Dropout**: En gardant le dropout actif pendant l'inférence et en moyennant plusieurs prédictions, on obtient une approximation de l'incertitude prédictive:
$$
p(\boldsymbol{y} \mid \boldsymbol{x}, \mathcal{D}) \approx \frac{1}{S} \sum_{s=1}^S p(\boldsymbol{y} \mid \boldsymbol{x}, \hat{\mathbf{W}} \epsilon^s + \hat{\boldsymbol{b}})
$$

Cette technique fournit une approximation de la distribution prédictive a posteriori bayésienne.

## Réseaux de neurones bayésiens

Les **réseaux de neurones bayésiens** (BNN) capturent l'incertitude de manière plus rigoureuse en marginalisant sur les paramètres:

$$
p(\boldsymbol{y} \mid \boldsymbol{x}, \mathcal{D}) = \int p(\boldsymbol{y} \mid \boldsymbol{x}, \boldsymbol{\theta}) \, p(\boldsymbol{\theta} \mid \mathcal{D}) \, d\boldsymbol{\theta}
$$

Cela correspond à un ensemble infini de réseaux, pondérés par leur probabilité a posteriori. Cette marginalisation:
- Évite le surapprentissage naturellement
- Fournit des estimations d'incertitude calibrées
- Est cependant difficile à réaliser en pratique pour les grands réseaux

Le Monte Carlo Dropout offre une approximation pratique de cette intégrale.

## Architectures classiques

### Le Neocognitron (1980)

L'idée d'alterner convolutions et pooling remonte au **neocognitron** de Kunihiko Fukushima (1980), inspiré par le modèle des cellules simples et complexes dans le cortex visuel découvert par Hubel et Wiesel.

### LeNet (1998)

Yann LeCun a popularisé cette architecture avec **LeNet-5**, conçu pour la reconnaissance de chiffres manuscrits. L'architecture typique alterne:
1. Couche de convolution
2. Couche de pooling
3. Répétition des étapes 1-2
4. Couches entièrement connectées pour la classification

### Architectures modernes

Depuis LeNet, de nombreuses architectures plus profondes et plus sophistiquées ont été développées:

- **AlexNet** (2012): premier CNN à gagner ImageNet avec une grande marge
- **VGGNet** (2014): architecture très profonde avec des convolutions $3 \times 3$ uniformes
- **GoogLeNet/Inception** (2014): modules "inception" avec convolutions parallèles de différentes tailles
- **ResNet** (2015): connexions résiduelles permettant l'entraînement de réseaux très profonds (100+ couches)
- **DenseNet** (2017): connexions denses entre toutes les couches

Ces architectures ont démontré que la profondeur est cruciale pour les performances, à condition d'utiliser les bonnes techniques de régularisation et de normalisation.

## Résumé

Les réseaux de neurones convolutifs constituent l'architecture de choix pour le traitement d'images. Leurs caractéristiques principales sont:

- **Invariance à la translation** grâce à l'opération de convolution
- **Partage des poids** qui réduit drastiquement le nombre de paramètres
- **Extraction hiérarchique de caractéristiques** des motifs simples aux concepts abstraits
- **Robustesse** grâce au pooling et aux techniques de régularisation

Les couches de normalisation (batch norm) et de régularisation (dropout) sont essentielles pour entraîner des réseaux profonds efficacement. Le dropout offre également une voie vers l'estimation de l'incertitude via le Monte Carlo Dropout.

Dans le prochain chapitre, nous étudierons les réseaux récurrents, qui étendent ces idées au traitement de données séquentielles.

## Exercices

```{admonition} Exercice 1: Dimensions de sortie
:class: tip

Soit une image d'entrée de taille $32 \times 32$ avec 3 canaux (RGB). On applique une couche de convolution avec 16 filtres de taille $5 \times 5$, un pas de 1, et aucun rembourrage.
1. Quelle est la taille de la sortie?
2. Combien de paramètres cette couche a-t-elle (incluant les biais)?
3. Si on ajoute un rembourrage de 2 pixels, quelle devient la taille de sortie?
```

```{admonition} Exercice 2: Matrice de Toeplitz
:class: tip

Considérez une image 1D $\boldsymbol{x} = [x_1, x_2, x_3, x_4]$ et un filtre $\boldsymbol{w} = [w_1, w_2]$.
1. Écrivez la matrice de Toeplitz $\mathbf{C}$ correspondant à cette convolution (sans rembourrage).
2. Calculez explicitement le résultat $\boldsymbol{y} = \mathbf{C}\boldsymbol{x}$.
3. Vérifiez que cela correspond bien à la définition de la convolution discrète.
```

```{admonition} Exercice 3: Batch normalization
:class: tip

Soit un mini-lot de 4 exemples avec des activations $z_1 = 2, z_2 = 4, z_3 = 6, z_4 = 8$ pour un neurone particulier.
1. Calculez $\mu_{\mathcal{B}}$ et $\sigma_{\mathcal{B}}^2$.
2. Calculez les activations normalisées $\hat{z}_1, \hat{z}_2, \hat{z}_3, \hat{z}_4$.
3. Si $\gamma = 2$ et $\beta = 1$, calculez les activations finales $\tilde{z}_i$.
```

```{admonition} Exercice 4: Dropout et variance
:class: tip

Considérez un neurone avec activation $h$ suivi d'un dropout avec probabilité $p = 0.5$.
1. Quelle est l'espérance de l'activation effective pendant l'entraînement?
2. Pourquoi faut-il multiplier par $(1-p)$ à l'inférence (ou diviser par $(1-p)$ à l'entraînement)?
3. Calculez la variance de l'activation effective pendant l'entraînement.
```
