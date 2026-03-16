# Exercices: graphes de calcul et règle de la chaîne

Ces exercices accompagnent la section sur les graphes de calcul du {ref}`chapitre 7 <ch7>`. Pour chaque fonction, on décompose le calcul en opérations élémentaires et on représente les dépendances par un graphe orienté acyclique (DAG).

---

## Partie 1: lire un graphe de calcul

Dans ces exercices, un DAG est donné. L'objectif est d'écrire l'expression mathématique qu'il représente.

````{admonition} Exercice 1: du graphe à l'expression (chaîne linéaire) ★
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

````{admonition} Solution exercice 1
:class: dropdown

**(a)** En remplaçant les variables intermédiaires:

$$
v_1 = e^x, \quad v_2 = e^x + 1, \quad v_3 = \log(e^x + 1)
$$

Donc $f(x) = \log(e^x + 1)$.

**(b)** $f(0) = \log(e^0 + 1) = \log(2) \approx 0{,}693$.

**(c)** C'est la fonction **softplus**, une approximation lisse de ReLU.
````

````{admonition} Exercice 2: du graphe à l'expression (embranchement) ★
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

````{admonition} Solution exercice 2
:class: dropdown

**(a)** $f(x) = \cos(x) + x^2$.

**(b)** Le noeud $x$ a **deux arêtes sortantes**: il alimente $v_1 = \cos(x)$ et $v_2 = x^2$. Cela signifie que $x$ contribue à $f$ par deux chemins distincts, et la règle de la chaîne devra sommer les deux contributions.

**(c)** $f(\pi) = \cos(\pi) + \pi^2 = -1 + \pi^2 \approx 8{,}870$.
````

````{admonition} Exercice 3: du graphe à l'expression (trois entrées) ★
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

````{admonition} Solution exercice 3
:class: dropdown

**(a)** $f(x, y, z) = x \cdot y + \log(z)$.

**(b)** $f(2, 3, 1) = 6 + \log(1) = 6 + 0 = 6$.

**(c)** Aucune variable d'entrée n'a de fan-out: $x$ alimente uniquement $v_1$, $y$ alimente uniquement $v_1$, et $z$ alimente uniquement $v_2$. Le graphe a deux branches indépendantes qui se rejoignent à l'addition.
````

---

## Partie 2: décomposer une fonction en graphe de calcul

Dans ces exercices, une expression mathématique est donnée. L'objectif est de la décomposer en opérations élémentaires (une seule opération par noeud) et de dessiner le DAG correspondant.

````{admonition} Exercice 4: de l'expression au graphe (fan-out simple) ★
:class: hint dropdown

Soit $f(x) = x \cdot e^x$.

**(a)** Identifiez les variables intermédiaires en décomposant $f$ en opérations élémentaires.

**(b)** Dessinez le DAG correspondant. Combien d'arêtes sortantes a le noeud $x$?

**(c)** Évaluez $f(1)$.
````

````{admonition} Solution exercice 4
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

Le noeud $x$ a **deux arêtes sortantes**: une vers $\exp$ et une vers la multiplication. C'est un fan-out.

**(c)** $f(1) = 1 \cdot e^1 = e \approx 2{,}718$.
````

````{admonition} Exercice 5: de l'expression au graphe (diamant) ★
:class: hint dropdown

Soit $f(x, y) = e^{x - y} + (x - y)^2$.

**(a)** Décomposez $f$ en opérations élémentaires et identifiez les variables intermédiaires.

**(b)** Dessinez le DAG. Quel noeud intermédiaire a un fan-out?

**(c)** Évaluez $f(1, 0)$.
````

````{admonition} Solution exercice 5
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

Le noeud $v_1$ a un **fan-out**: il alimente à la fois $v_2 = e^{v_1}$ et $v_3 = v_1^2$. La passe arrière devra accumuler les contributions des deux branches pour obtenir $\bar{v}_1$.

**(c)** $f(1, 0) = e^1 + 1^2 = e + 1 \approx 3{,}718$.
````

````{admonition} Exercice 6: de l'expression au graphe (triple fan-out) ★★
:class: hint dropdown

Soit $f(x) = x \cdot \cos(x) + x^2$.

**(a)** Décomposez $f$ en opérations élémentaires.

**(b)** Dessinez le DAG. Combien d'arêtes sortantes a le noeud $x$?

**(c)** Évaluez $f(\pi/2)$.
````

````{admonition} Solution exercice 6
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

---

## Partie 3: règle de la chaîne dans un DAG

Dans ces exercices, un DAG est donné. L'objectif est d'écrire les règles VJP (passe arrière) et de calculer les gradients, d'abord symboliquement, puis numériquement.

On utilise la notation $\bar{v} = \partial f / \partial v$ pour l'adjoint d'un noeud $v$. La passe arrière initialise $\bar{v}_{\text{sortie}} = 1$ et remonte le graphe en accumulant avec $\mathrel{{+}{=}}$.

````{admonition} Exercice 7: passe arrière sur une chaîne linéaire ★★
:class: hint dropdown

Considérez le graphe de calcul suivant:

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

**(a)** Écrivez la table VJP (une ligne par opération).

**(b)** Exécutez la passe arrière symboliquement en partant de $\bar{v}_2 = 1$. Quelle est l'expression de $\bar{x} = \partial f / \partial x$?

**(c)** Évaluez la passe avant et la passe arrière en $x = 0$.
````

````{admonition} Solution exercice 7
:class: dropdown

La fonction est $f(x) = e^{\sin(x)}$.

**(a)** Table VJP:

| Étape | Opération | Entrée | Sortie | VJP locale |
|:-----:|:---------:|:------:|:------:|:-----------|
| 1 | sin | $x$ | $v_1$ | $\bar{x}\ {+}{=}\ \cos(x) \cdot \bar{v}_1$ |
| 2 | exp | $v_1$ | $v_2$ | $\bar{v}_1\ {+}{=}\ e^{v_1} \cdot \bar{v}_2$ |

**(b)** Passe arrière (étapes 2 → 1):

- Initialiser: $\bar{v}_2 = 1$
- Étape 2: $\bar{v}_1 = e^{v_1} \cdot \bar{v}_2 = e^{\sin(x)} \cdot 1 = e^{\sin(x)}$
- Étape 1: $\bar{x} = \cos(x) \cdot \bar{v}_1 = \cos(x) \cdot e^{\sin(x)}$

Donc $\dfrac{\partial f}{\partial x} = \cos(x) \, e^{\sin(x)}$.

**(c)** En $x = 0$:

Passe avant: $v_1 = \sin(0) = 0$, $v_2 = e^0 = 1$, $f = 1$.

Passe arrière: $\bar{v}_2 = 1$, $\bar{v}_1 = e^0 \cdot 1 = 1$, $\bar{x} = \cos(0) \cdot 1 = 1$.

Donc $f'(0) = 1$.
````

````{admonition} Exercice 8: passe arrière avec accumulation ★★
:class: hint dropdown

Reprenons le graphe de $f(x, y) = e^{x - y} + (x - y)^2$ de l'exercice 5:

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

**(a)** Écrivez la table VJP complète.

**(b)** Exécutez la passe arrière symboliquement. Le noeud $v_1$ a deux successeurs: montrez comment $\bar{v}_1$ accumule les deux contributions.

**(c)** Évaluez numériquement en $(x, y) = (1, 0)$. Vérifiez votre réponse par dérivation directe.
````

````{admonition} Solution exercice 8
:class: dropdown

**(a)** Table VJP:

| Étape | Opération | Entrées | Sortie | VJP locale |
|:-----:|:---------:|:-------:|:------:|:-----------|
| 1 | sub | $x, y$ | $v_1$ | $\bar{x}\ {+}{=}\ \bar{v}_1$, $\quad \bar{y}\ {+}{=}\ {-}\bar{v}_1$ |
| 2 | exp | $v_1$ | $v_2$ | $\bar{v}_1\ {+}{=}\ e^{v_1} \cdot \bar{v}_2$ |
| 3 | square | $v_1$ | $v_3$ | $\bar{v}_1\ {+}{=}\ 2 v_1 \cdot \bar{v}_3$ |
| 4 | add | $v_2, v_3$ | $v_4$ | $\bar{v}_2\ {+}{=}\ \bar{v}_4$, $\quad \bar{v}_3\ {+}{=}\ \bar{v}_4$ |

**(b)** Passe arrière (étapes 4 → 3 → 2 → 1):

- Initialiser: $\bar{v}_4 = 1$
- Étape 4 (add): $\bar{v}_2 = \bar{v}_4 = 1$, $\quad \bar{v}_3 = \bar{v}_4 = 1$
- Étape 3 (square): $\bar{v}_1\ {+}{=}\ 2 v_1 \cdot \bar{v}_3 = 2 v_1$
- Étape 2 (exp): $\bar{v}_1\ {+}{=}\ e^{v_1} \cdot \bar{v}_2 = e^{v_1}$

Après accumulation: $\bar{v}_1 = e^{v_1} + 2 v_1$. Les deux contributions viennent des deux chemins partant de $v_1$ (via $\exp$ et via le carré).

- Étape 1 (sub): $\bar{x} = \bar{v}_1 = e^{v_1} + 2 v_1$, $\quad \bar{y} = -\bar{v}_1 = -(e^{v_1} + 2 v_1)$

**(c)** En $(x, y) = (1, 0)$:

Passe avant: $v_1 = 1$, $v_2 = e$, $v_3 = 1$, $v_4 = e + 1 \approx 3{,}718$.

Passe arrière: $\bar{v}_4 = 1$, $\bar{v}_2 = 1$, $\bar{v}_3 = 1$, $\bar{v}_1 = e + 2 \approx 4{,}718$.

$$
\bar{x} = e + 2 \approx 4{,}718, \quad \bar{y} = -(e + 2) \approx -4{,}718
$$

Vérification par dérivation directe: $\frac{\partial f}{\partial x} = e^{x-y} + 2(x - y) = e + 2$. Correct. On note aussi que $\partial f / \partial x = -\partial f / \partial y$, ce qui est attendu puisque $f$ dépend de $x$ et $y$ uniquement à travers $x - y$.
````

````{admonition} Exercice 9: ReLU et gradient nul ★★★
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

On rappelle que $\text{relu}(t) = \max(0, t)$ et que sa dérivée vaut $\mathbb{1}[t > 0]$ (indicatrice: 1 si $t > 0$, 0 sinon).

**(a)** Écrivez la table VJP complète. Quelles variables d'entrée ont un fan-out?

**(b)** Évaluez la passe avant et la passe arrière en $(x, y) = (3, 1)$. Le ReLU est actif dans ce cas.

**(c)** Évaluez la passe avant et la passe arrière en $(x, y) = (-3, 1)$. Le ReLU est inactif dans ce cas. Que se passe-t-il pour les gradients?

**(d)** Vérifiez vos réponses. Quand $x + y > 0$, simplifiez $f(x, y)$ et dérivez directement.
````

````{admonition} Solution exercice 9
:class: dropdown

**(a)** Table VJP:

| Étape | Opération | Entrées | Sortie | VJP locale |
|:-----:|:---------:|:-------:|:------:|:-----------|
| 1 | add | $x, y$ | $v_1$ | $\bar{x}\ {+}{=}\ \bar{v}_1$, $\quad \bar{y}\ {+}{=}\ \bar{v}_1$ |
| 2 | relu | $v_1$ | $v_2$ | $\bar{v}_1\ {+}{=}\ \mathbb{1}[v_1 > 0] \cdot \bar{v}_2$ |
| 3 | sub | $x, y$ | $v_3$ | $\bar{x}\ {+}{=}\ \bar{v}_3$, $\quad \bar{y}\ {+}{=}\ {-}\bar{v}_3$ |
| 4 | mul | $v_2, v_3$ | $v_4$ | $\bar{v}_2\ {+}{=}\ v_3 \cdot \bar{v}_4$, $\quad \bar{v}_3\ {+}{=}\ v_2 \cdot \bar{v}_4$ |

Les deux variables d'entrée $x$ et $y$ ont un fan-out: chacune alimente $v_1$ (addition) et $v_3$ (soustraction).

**(b)** En $(x, y) = (3, 1)$ — ReLU actif:

Passe avant: $v_1 = 4$, $v_2 = \text{relu}(4) = 4$, $v_3 = 2$, $v_4 = 4 \cdot 2 = 8$.

Passe arrière:
- $\bar{v}_4 = 1$
- Étape 4 (mul): $\bar{v}_2 = v_3 \cdot 1 = 2$, $\quad \bar{v}_3 = v_2 \cdot 1 = 4$
- Étape 3 (sub): $\bar{x}\ {+}{=}\ 4$, $\quad \bar{y}\ {+}{=}\ {-4}$
- Étape 2 (relu): $\bar{v}_1 = \mathbb{1}[4 > 0] \cdot 2 = 2$
- Étape 1 (add): $\bar{x}\ {+}{=}\ 2$, $\quad \bar{y}\ {+}{=}\ 2$

Total: $\bar{x} = 4 + 2 = 6$, $\quad \bar{y} = -4 + 2 = -2$.

**(c)** En $(x, y) = (-3, 1)$ — ReLU inactif:

Passe avant: $v_1 = -2$, $v_2 = \text{relu}(-2) = 0$, $v_3 = -4$, $v_4 = 0 \cdot (-4) = 0$.

Passe arrière:
- $\bar{v}_4 = 1$
- Étape 4 (mul): $\bar{v}_2 = v_3 \cdot 1 = -4$, $\quad \bar{v}_3 = v_2 \cdot 1 = 0$
- Étape 3 (sub): $\bar{x}\ {+}{=}\ 0$, $\quad \bar{y}\ {+}{=}\ 0$
- Étape 2 (relu): $\bar{v}_1 = \mathbb{1}[-2 > 0] \cdot (-4) = 0 \cdot (-4) = 0$
- Étape 1 (add): $\bar{x}\ {+}{=}\ 0$, $\quad \bar{y}\ {+}{=}\ 0$

Total: $\bar{x} = 0$, $\quad \bar{y} = 0$.

Les deux gradients sont nuls. Ce phénomène s'explique par deux effets combinés: le ReLU inactif ($v_1 \leq 0$) bloque le gradient à travers la branche gauche ($\bar{v}_1 = 0$), et la multiplication par $v_2 = 0$ bloque le gradient à travers la branche droite ($\bar{v}_3 = v_2 \cdot \bar{v}_4 = 0$). Quand la sortie du ReLU est nulle, aucun gradient ne peut remonter, quelle que soit la branche.

**(d)** Quand $x + y > 0$, $\text{relu}(x + y) = x + y$, donc $f(x, y) = (x + y)(x - y) = x^2 - y^2$.

Les dérivées directes sont $\partial f / \partial x = 2x$ et $\partial f / \partial y = -2y$.

En $(3, 1)$: $\bar{x} = 2 \cdot 3 = 6$ et $\bar{y} = -2 \cdot 1 = -2$. Correct.
````
