# Réseaux de neurones

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Expliquer le problème XOR et la motivation des réseaux multicouches
- Définir l'architecture d'un perceptron multicouche (MLP)
- Distinguer les principales fonctions d'activation et leurs propriétés
- Dériver l'algorithme de rétropropagation
- Expliquer la différence entre JVP et VJP
- Identifier les causes du problème du gradient qui disparaît
- Implémenter un MLP simple avec rétropropagation
```

## Introduction

Les réseaux de neurones étendent les modèles linéaires en composant plusieurs transformations non linéaires. Cette approche permet d'apprendre automatiquement des représentations hiérarchiques des données, sans spécifier manuellement les caractéristiques.

Ce chapitre introduit les perceptrons multicouches (MLP) et l'algorithme de rétropropagation qui permet de les entraîner.

## Le problème du XOR

### Limitations des modèles linéaires

Le perceptron et la régression logistique ne peuvent représenter que des frontières de décision linéaires. Certaines fonctions simples sont donc impossibles à apprendre.

La fonction XOR (ou exclusif) illustre cette limitation:

| $x_1$ | $x_2$ | $y = x_1 \oplus x_2$ |
|-------|-------|----------------------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Aucun hyperplan ne peut séparer les deux classes. Les points $(0,0)$ et $(1,1)$ appartiennent à la classe 0, tandis que $(0,1)$ et $(1,0)$ appartiennent à la classe 1.

La publication du livre *Perceptrons* par Minsky et Papert en 1969, qui démontra formellement cette limitation, contribua à un ralentissement de la recherche en réseaux de neurones.

### Solution: composition de fonctions

Minsky et Papert montrèrent également qu'en empilant deux couches de perceptrons, la fonction XOR peut être représentée parfaitement. L'idée est de transformer les entrées dans un espace où les classes deviennent linéairement séparables.

## Perceptrons multicouches

### Architecture

Un **perceptron multicouche** (MLP, de l'anglais *multilayer perceptron*) est une composition de transformations:

$$
f(x; \theta) = f_L(f_{L-1}(\cdots f_1(x) \cdots))
$$

où chaque couche $\ell$ applique une transformation affine suivie d'une non-linéarité:

$$
z_\ell = \varphi_\ell(W_\ell z_{\ell-1} + b_\ell)
$$

Les composantes sont:
- $z_{\ell-1}$: les **unités cachées** (ou activations) de la couche précédente
- $W_\ell$: la matrice de poids
- $b_\ell$: le vecteur de biais
- $a_\ell = W_\ell z_{\ell-1} + b_\ell$: les **pré-activations**
- $\varphi_\ell$: la **fonction d'activation**
- $z_\ell$: les activations de la couche $\ell$

La première couche reçoit l'entrée $z_0 = x$. La dernière couche produit la sortie du réseau.

### Exemple: classification binaire

Un MLP pour la classification binaire en 2D pourrait avoir l'architecture suivante:

$$
\begin{aligned}
p(y | x; \theta) &= \text{Ber}(y | \sigma(a_3)) \\
a_3 &= w_3^\top z_2 + b_3 \\
z_2 &= \varphi(W_2 z_1 + b_2) \\
z_1 &= \varphi(W_1 x + b_1)
\end{aligned}
$$

Ce réseau a deux couches cachées ($z_1$ et $z_2$) et une couche de sortie.

### Expressivité

Un MLP avec une couche cachée suffisamment large peut approximer toute fonction continue sur un compact (théorème d'approximation universelle). Cependant, la taille de la couche peut croître exponentiellement avec la complexité de la fonction.

Les réseaux profonds (avec plusieurs couches) peuvent représenter certaines fonctions de manière plus compacte que les réseaux larges mais peu profonds.

## Fonctions d'activation

### Nécessité de la non-linéarité

Sans fonction d'activation non linéaire, un MLP se réduit à un modèle linéaire. En effet, la composition de transformations linéaires reste linéaire:

$$
W_L(W_{L-1}(\cdots W_1 x \cdots)) = (W_L W_{L-1} \cdots W_1) x = W' x
$$

Les fonctions d'activation non linéaires sont essentielles à l'expressivité du réseau.

### Sigmoïde

La **fonction sigmoïde** (ou logistique):

$$
\sigma(a) = \frac{1}{1 + e^{-a}}
$$

transforme les valeurs réelles dans l'intervalle $(0, 1)$. Sa dérivée est $\sigma'(a) = \sigma(a)(1 - \sigma(a))$.

**Problème**: la sigmoïde **sature** pour les grandes valeurs de $|a|$. Dans ces régions, la dérivée est proche de zéro, ce qui bloque la propagation du gradient.

### Tangente hyperbolique

La **tangente hyperbolique**:

$$
\tanh(a) = \frac{e^a - e^{-a}}{e^a + e^{-a}} = 2\sigma(2a) - 1
$$

est similaire à la sigmoïde mais centrée autour de zéro, avec des sorties dans $(-1, 1)$. Elle souffre du même problème de saturation.

### ReLU

L'**unité linéaire rectifiée** (ReLU, de l'anglais *rectified linear unit*):

$$
\text{ReLU}(a) = \max(0, a) = a \cdot \mathbb{1}(a > 0)
$$

est la fonction d'activation la plus utilisée en apprentissage profond moderne. Ses avantages:

- **Pas de saturation** pour les valeurs positives
- **Calcul efficace**
- **Sparsité**: de nombreuses unités sont à zéro

Sa dérivée est:

$$
\text{ReLU}'(a) = \begin{cases}
0 & \text{si } a < 0 \\
1 & \text{si } a > 0
\end{cases}
$$

**Problème**: les neurones peuvent "mourir" si leurs pré-activations sont toujours négatives. Le gradient est alors zéro et l'apprentissage s'arrête.

### Variantes de ReLU

**Leaky ReLU**:
$$
\text{LeakyReLU}(a) = \max(\alpha a, a)
$$
où $\alpha \approx 0.01$ permet un petit gradient pour les valeurs négatives.

**ELU** (Exponential Linear Unit):
$$
\text{ELU}(a) = \begin{cases}
a & \text{si } a > 0 \\
\alpha(e^a - 1) & \text{si } a \leq 0
\end{cases}
$$

**GELU** (Gaussian Error Linear Unit):
$$
\text{GELU}(a) = a \cdot \Phi(a)
$$
où $\Phi$ est la fonction de répartition de la loi normale standard. Utilisée dans les transformers modernes.

## Rétropropagation

### Motivation

L'entraînement d'un réseau de neurones requiert le gradient de la perte par rapport à tous les paramètres. La **rétropropagation** (backpropagation) est l'algorithme standard pour ce calcul.

La rétropropagation exploite la structure de composition du réseau pour calculer efficacement les gradients par la règle de la chaîne.

### Règle de la chaîne

Pour une composition $f = f_L \circ \cdots \circ f_1$, la jacobienne est:

$$
\mathbf{J}_f(x) = \mathbf{J}_{f_L}(x_L) \cdot \mathbf{J}_{f_{L-1}}(x_{L-1}) \cdot \ldots \cdot \mathbf{J}_{f_1}(x)
$$

où $x_\ell = f_\ell(x_{\ell-1})$ sont les valeurs intermédiaires.

### JVP et VJP

Le calcul de la jacobienne complète est coûteux. En pratique, nous calculons des produits matrice-vecteur.

**JVP** (Jacobian-Vector Product): multiplication à droite par un vecteur $v$:
$$
\mathbf{J}_f(x) v = \mathbf{J}_{f_L} \cdot \mathbf{J}_{f_{L-1}} \cdot \ldots \cdot \mathbf{J}_{f_1} \cdot v
$$

Le calcul se fait de droite à gauche, propageant $v$ à travers les couches.

**VJP** (Vector-Jacobian Product): multiplication à gauche par un vecteur ligne $u^\top$:
$$
u^\top \mathbf{J}_f(x) = u^\top \cdot \mathbf{J}_{f_L} \cdot \mathbf{J}_{f_{L-1}} \cdot \ldots \cdot \mathbf{J}_{f_1}
$$

Le calcul se fait de gauche à droite.

### Accumulation avant vs arrière

- **Mode avant** (JVP): efficace quand la sortie a plus de dimensions que l'entrée ($m > n$)
- **Mode arrière** (VJP): efficace quand l'entrée a plus de dimensions que la sortie ($n > m$)

Pour une fonction de perte scalaire ($m = 1$), le mode arrière est optimal. C'est le mode utilisé par la rétropropagation.

### Algorithme

```{prf:algorithm} Rétropropagation
:label: backprop

**Entrée**: Entrée $x$, cible $y$, paramètres $\theta$

**Sortie**: Perte $\mathcal{L}$, gradients $\nabla_{\theta_k} \mathcal{L}$ pour $k = 1, \ldots, K$

// Passe avant
1. $x_1 := x$
2. Pour $k = 1, \ldots, K$:
   - $x_{k+1} = f_k(x_k; \theta_k)$

// Passe arrière
3. $u_{K+1} := 1$
4. Pour $k = K, \ldots, 1$:
   - $g_k := u_{k+1}^\top \frac{\partial f_k(x_k; \theta_k)}{\partial \theta_k}$ (gradient des paramètres)
   - $u_k^\top := u_{k+1}^\top \frac{\partial f_k(x_k; \theta_k)}{\partial x_k}$ (propagation de l'adjoint)

5. Retourner $\mathcal{L} = x_{K+1}$, $\{\nabla_{\theta_k} \mathcal{L} = g_k^\top\}$
```

Le vecteur $u_k$ est appelé **adjoint** ou **delta**. Il accumule la sensibilité de la perte aux activations de la couche $k$.

### Exemple: MLP avec une couche cachée

Considérons la perte MSE pour un MLP:

$$
\mathcal{L} = \frac{1}{2}\|y - W_2 \varphi(W_1 x)\|^2
$$

**Passe avant**:
$$
\begin{aligned}
a_1 &= W_1 x \\
z_1 &= \varphi(a_1) \\
a_2 &= W_2 z_1 \\
\mathcal{L} &= \frac{1}{2}\|a_2 - y\|^2
\end{aligned}
$$

**Passe arrière**:
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial a_2} &= a_2 - y \\
\frac{\partial \mathcal{L}}{\partial W_2} &= \frac{\partial \mathcal{L}}{\partial a_2} z_1^\top \\
\frac{\partial \mathcal{L}}{\partial z_1} &= W_2^\top \frac{\partial \mathcal{L}}{\partial a_2} \\
\frac{\partial \mathcal{L}}{\partial a_1} &= \frac{\partial \mathcal{L}}{\partial z_1} \odot \varphi'(a_1) \\
\frac{\partial \mathcal{L}}{\partial W_1} &= \frac{\partial \mathcal{L}}{\partial a_1} x^\top
\end{aligned}
$$

où $\odot$ désigne le produit élément par élément.

## Problème du gradient qui disparaît

### Cause

Dans un réseau profond, le gradient doit traverser de nombreuses couches. À chaque couche, il est multiplié par la jacobienne locale:

$$
\frac{\partial \mathcal{L}}{\partial z_\ell} = \frac{\partial \mathcal{L}}{\partial z_{\ell+1}} \cdot \frac{\partial z_{\ell+1}}{\partial z_\ell}
$$

Si les jacobiennes ont un rayon spectral inférieur à 1, le gradient diminue exponentiellement avec la profondeur. C'est le **gradient qui disparaît** (vanishing gradient).

Inversement, si le rayon spectral est supérieur à 1, le gradient **explose**.

### Solutions

**Fonctions d'activation appropriées**: ReLU et ses variantes évitent la saturation.

**Normalisation par lots** (Batch Normalization): normalise les activations à chaque couche pour stabiliser l'entraînement.

**Connexions résiduelles**: permettent au gradient de "sauter" des couches.

**Initialisation appropriée**: les poids sont initialisés pour que les activations et gradients restent dans une plage raisonnable.

**Écrêtage du gradient** (Gradient Clipping): limite la norme du gradient:
$$
g' = \min\left(1, \frac{c}{\|g\|}\right) g
$$

## Implémentation

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def mse_loss(y_pred, y_true):
    return 0.5 * np.mean((y_pred - y_true)**2)

class MLP:
    """Perceptron multicouche simple avec une couche cachée."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Initialisation Xavier/Glorot
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((output_dim, 1))
        
    def forward(self, X):
        """Passe avant."""
        # X: (input_dim, batch_size)
        self.a1 = self.W1 @ X + self.b1
        self.z1 = relu(self.a1)
        self.a2 = self.W2 @ self.z1 + self.b2
        return self.a2
    
    def backward(self, X, y, y_pred):
        """Passe arrière (rétropropagation)."""
        batch_size = X.shape[1]
        
        # Gradient de la perte MSE
        dL_da2 = (y_pred - y) / batch_size
        
        # Gradients de W2 et b2
        dL_dW2 = dL_da2 @ self.z1.T
        dL_db2 = np.sum(dL_da2, axis=1, keepdims=True)
        
        # Propagation vers la couche cachée
        dL_dz1 = self.W2.T @ dL_da2
        dL_da1 = dL_dz1 * relu_grad(self.a1)
        
        # Gradients de W1 et b1
        dL_dW1 = dL_da1 @ X.T
        dL_db1 = np.sum(dL_da1, axis=1, keepdims=True)
        
        return {'W1': dL_dW1, 'b1': dL_db1, 'W2': dL_dW2, 'b2': dL_db2}
    
    def train_step(self, X, y, lr=0.01):
        """Une étape d'entraînement."""
        y_pred = self.forward(X)
        loss = mse_loss(y_pred, y)
        grads = self.backward(X, y, y_pred)
        
        # Mise à jour des paramètres
        self.W1 -= lr * grads['W1']
        self.b1 -= lr * grads['b1']
        self.W2 -= lr * grads['W2']
        self.b2 -= lr * grads['b2']
        
        return loss

# Exemple: apprendre XOR
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])  # 2x4
y = np.array([[0, 1, 1, 0]])  # 1x4

mlp = MLP(input_dim=2, hidden_dim=4, output_dim=1)

for epoch in range(1000):
    loss = mlp.train_step(X, y, lr=0.5)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

## Graphes de calcul

### Définition

Les MLP ont une structure de chaîne simple. Les réseaux modernes ont des architectures plus complexes représentées par des **graphes de calcul**: des graphes orientés acycliques (DAG) où chaque nœud est une opération différentiable.

### Dérivée totale

Dans un DAG, un nœud peut influencer la sortie par plusieurs chemins. La règle de la chaîne devient:

$$
\frac{\partial o}{\partial x_j} = \sum_{k \in \text{enfants}(j)} \frac{\partial o}{\partial x_k} \frac{\partial x_k}{\partial x_j}
$$

La rétropropagation parcourt le graphe dans l'ordre topologique inverse, accumulant les contributions de tous les chemins.

### Différentiation automatique

Les bibliothèques modernes (PyTorch, JAX, TensorFlow) implémentent la **différentiation automatique**: le graphe de calcul est construit lors de l'exécution, et les gradients sont calculés automatiquement.

## Résumé

Ce chapitre a introduit les réseaux de neurones:

- Le **problème XOR** motive les architectures multicouches
- Un **MLP** compose des transformations affines et des non-linéarités
- Les **fonctions d'activation** (ReLU, sigmoïde, etc.) apportent la non-linéarité essentielle
- La **rétropropagation** calcule efficacement les gradients par la règle de la chaîne
- Le mode **VJP** (arrière) est optimal pour les fonctions à sortie scalaire
- Le **gradient qui disparaît** est un défi pour les réseaux profonds
- Les **graphes de calcul** généralisent les architectures en chaîne

Les chapitres suivants présentent des architectures spécialisées: les réseaux convolutifs pour les images et les réseaux récurrents pour les séquences.
