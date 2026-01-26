---
kernelspec:
  name: python3
  display_name: Python 3
---

# Le cadre probabiliste

```{admonition} Objectifs d'apprentissage
:class: note

À la fin de ce chapitre, vous serez en mesure de:
- Expliquer le cadre bayésien et le rôle de l'a priori, la vraisemblance et l'a posteriori
- Comprendre pourquoi l'inférence bayésienne complète est souvent intractable
- Appliquer le maximum a posteriori (MAP) et montrer son équivalence avec la régularisation
- Interpréter l'EMV comme la minimisation de la divergence KL
- Unifier les perspectives décisionnelle, probabiliste et informationnelle de l'apprentissage
```

Les chapitres précédents ont utilisé le maximum de vraisemblance pour justifier nos fonctions de perte: le [chapitre 1](ch1_learning_problem.md) a introduit le principe, le [chapitre 2](ch2_linear_regression.md) l'a appliqué à la régression (donnant les moindres carrés), et le [chapitre 3](ch3_classification.md) à la classification (donnant l'entropie croisée). Ce chapitre va plus loin en présentant le **cadre bayésien complet**, qui offre une perspective plus riche sur l'apprentissage.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

# Configuration pour des figures haute résolution
%config InlineBackend.figure_format = 'retina'
```

## Le cadre probabiliste

Le maximum de vraisemblance, introduit au chapitre 1, nous a donné un principe pour choisir les paramètres: maximiser la probabilité des données observées. Mais l'EMV n'est qu'un point de départ. Comment intégrer des connaissances préalables? Comment quantifier notre incertitude sur les paramètres?

Le **cadre bayésien** offre des réponses à ces questions en traitant les paramètres eux-mêmes comme des variables aléatoires. Cette section présente d'abord le cadre général de l'inférence bayésienne, puis montre comment le maximum a posteriori (MAP) relie l'approche bayésienne à la régularisation.

### Le cadre bayésien

La **statistique bayésienne** propose un cadre général pour l'estimation de paramètres. Au lieu d'estimer un point unique, elle caractérise notre **incertitude** sur les paramètres par une distribution de probabilité.

Le théorème de Bayes nous dit comment mettre à jour nos croyances sur les paramètres $\boldsymbol{\theta}$ après avoir observé des données $\mathcal{D}$:

$$
p(\boldsymbol{\theta} | \mathcal{D}) = \frac{p(\boldsymbol{\theta}) \, p(\mathcal{D} | \boldsymbol{\theta})}{p(\mathcal{D})}
$$

Chaque terme a un nom et un rôle précis:

- $p(\boldsymbol{\theta} | \mathcal{D})$ est la **distribution a posteriori**: notre croyance sur $\boldsymbol{\theta}$ après avoir vu les données
- $p(\boldsymbol{\theta})$ est la **distribution a priori**: notre croyance sur $\boldsymbol{\theta}$ avant d'observer les données
- $p(\mathcal{D} | \boldsymbol{\theta})$ est la **vraisemblance**: la probabilité des données pour un choix de paramètres
- $p(\mathcal{D}) = \int p(\boldsymbol{\theta}') p(\mathcal{D} | \boldsymbol{\theta}') d\boldsymbol{\theta}'$ est la **vraisemblance marginale**: une constante de normalisation

L'a priori encode notre connaissance préalable. Pour une pièce de monnaie, nous pourrions croire que $\theta$ est probablement proche de 0,5. L'a posteriori combine cette croyance avec l'évidence des données.

### Prédiction bayésienne et distribution prédictive a posteriori

En pratique, nous ne connaissons pas $p(y|\mathbf{x})$. Nous avons un modèle paramétrique $p(y|\mathbf{x}, \boldsymbol{\theta})$ et une distribution a posteriori $p(\boldsymbol{\theta}|\mathcal{D})$ sur les paramètres. L'approche bayésienne complète consiste à **moyenner les prédictions sur tous les paramètres possibles**, pondérés par leur probabilité a posteriori:

$$
p(y|\mathbf{x}, \mathcal{D}) = \int p(y|\mathbf{x}, \boldsymbol{\theta}) \, p(\boldsymbol{\theta}|\mathcal{D}) \, d\boldsymbol{\theta}
$$

Cette **distribution prédictive a posteriori** intègre l'incertitude sur les paramètres. Elle ne s'engage pas sur une valeur unique de $\boldsymbol{\theta}$, mais considère toutes les valeurs plausibles.

Le problème: cette intégrale est rarement calculable analytiquement. Elle nécessite d'intégrer sur un espace de paramètres de grande dimension, ce qui est coûteux ou impossible en pratique. C'est pourquoi nous recourons souvent à des **estimateurs ponctuels**: plutôt que d'intégrer sur tous les $\boldsymbol{\theta}$, nous en choisissons un seul, comme l'EMV ou le MAP.

### Utilité du modèle probabiliste

Si nous finissons souvent par utiliser un estimateur ponctuel, pourquoi adopter le cadre probabiliste? Plusieurs raisons:

1. **Justifier la fonction de perte**: La perte quadratique découle naturellement de l'hypothèse de bruit gaussien. La perte logarithmique vient du principe de maximum de vraisemblance. Le cadre probabiliste explique *pourquoi* ces choix sont raisonnables.

2. **Quantifier l'incertitude**: Au-delà de la prédiction ponctuelle $\hat{y} = f(\mathbf{x}; \hat{\boldsymbol{\theta}})$, nous pouvons donner un **intervalle de prédiction**. Sous un modèle gaussien, $y$ a environ 95% de chances de tomber dans $[f(\mathbf{x}) - 2\sigma, f(\mathbf{x}) + 2\sigma]$.

3. **Comparer des modèles**: La vraisemblance marginale $p(\mathcal{D})$ permet de comparer des modèles de complexités différentes, pénalisant automatiquement les modèles trop complexes.

4. **Ouvrir la porte à l'inférence complète**: Quand les ressources le permettent (méthodes de Monte Carlo, inférence variationnelle), nous pouvons approximer la distribution prédictive complète plutôt que de nous limiter à un point.

## Maximum de vraisemblance: rappel et approfondissement

Le [chapitre 1](ch1_learning_problem.md) a introduit le principe du maximum de vraisemblance: choisir les paramètres $\boldsymbol{\theta}$ qui maximisent la probabilité des données observées sous l'hypothèse i.i.d. Le [chapitre 2](ch2_linear_regression.md) a montré que ce principe, appliqué à un modèle gaussien, donne les moindres carrés. Le [chapitre 3](ch3_classification.md) a montré qu'appliqué à un modèle de Bernoulli, il donne l'entropie croisée.

Cette section approfondit ces idées en explorant des extensions importantes: la régression hétéroscédastique et le lien avec la minimisation du risque empirique.

### L'EMV comme minimisation du risque empirique

La log-vraisemblance négative (LVN) prend la forme:

$$
\text{LVN}(\boldsymbol{\theta}) = -\sum_{i=1}^N \log p(y_i | \mathbf{x}_i; \boldsymbol{\theta})
$$

Remarquez la structure: c'est une somme sur les exemples d'une quantité $-\log p(y_i | \mathbf{x}_i; \boldsymbol{\theta})$ qui dépend de chaque observation. Cette quantité joue le rôle d'une fonction de perte. Le maximum de vraisemblance est donc un cas particulier de la minimisation du risque empirique, où la perte est définie par le modèle probabiliste lui-même.

### Régression homoscédastique et hétéroscédastique

Le [chapitre 2](ch2_linear_regression.md) a montré que sous le modèle $y = f(\mathbf{x}; \boldsymbol{\theta}) + \varepsilon$ avec $\varepsilon \sim \mathcal{N}(0, \sigma^2)$, minimiser la LVN revient à minimiser la somme des erreurs quadratiques. La perte quadratique découle donc de l'hypothèse gaussienne.

Dans ce modèle, la variance $\sigma^2$ est constante pour toutes les entrées $\mathbf{x}$. C'est ce qu'on appelle la **régression homoscédastique** (du grec *homos*, même, et *skedasis*, dispersion). C'est l'hypothèse standard en régression linéaire.

En pratique, l'incertitude peut varier selon l'entrée. Par exemple, les mesures à haute vitesse peuvent être plus bruitées que celles à basse vitesse. La **régression hétéroscédastique** modélise cette variation en faisant dépendre la variance de $\mathbf{x}$:

$$
p(y|\mathbf{x}; \boldsymbol{\theta}) = \mathcal{N}(y | f_\mu(\mathbf{x}; \boldsymbol{\theta}), f_\sigma(\mathbf{x}; \boldsymbol{\theta})^2)
$$

où $f_\mu$ prédit la moyenne et $f_\sigma$ prédit l'écart-type. Ce modèle est plus flexible mais requiert d'apprendre des paramètres supplémentaires.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
from IPython.display import HTML

# Générer des données synthétiques
np.random.seed(42)
N = 100
x_data = np.random.uniform(0.5, 9.5, N)
f_mu = lambda x: 0.5 * x + 1

# Homoscédastique: variance constante
sigma_homo = 0.7
y_homo = f_mu(x_data) + np.random.normal(0, sigma_homo, N)

# Hétéroscédastique: variance croissante
f_sigma = lambda x: 0.3 + 0.12 * x
y_hetero = f_mu(x_data) + np.random.normal(0, f_sigma(x_data))

# Configuration de la figure
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
x_line = np.linspace(0, 10, 100)
y_pdf_range = np.linspace(-3, 9, 200)
scale = 2.5  # échelle pour afficher les PDFs

def init():
    for ax in axes:
        ax.clear()
    return []

def animate(frame):
    x_current = 0.5 + frame * 9 / 59  # balayer de 0.5 à 9.5
    
    for idx, (ax, y_data, title, color, get_sigma) in enumerate([
        (axes[0], y_homo, 'Régression homoscédastique', 'steelblue', lambda x: sigma_homo),
        (axes[1], y_hetero, 'Régression hétéroscédastique', 'coral', f_sigma)
    ]):
        ax.clear()
        
        # Données et ligne de régression
        ax.scatter(x_data, y_data, alpha=0.4, s=20, c='gray', zorder=1)
        ax.plot(x_line, f_mu(x_line), 'k-', linewidth=2, zorder=2)
        
        # Gaussienne à la position actuelle
        mu = f_mu(x_current)
        sigma = get_sigma(x_current)
        pdf = norm.pdf(y_pdf_range, mu, sigma)
        
        # Afficher la gaussienne "horizontalement"
        ax.fill_betweenx(y_pdf_range, x_current, x_current + scale * pdf, 
                         alpha=0.5, color=color, zorder=3)
        ax.plot(x_current + scale * pdf, y_pdf_range, color=color, linewidth=2, zorder=4)
        
        # Ligne verticale indiquant la position
        ax.axvline(x_current, color=color, linestyle='--', alpha=0.5, linewidth=1)
        
        # Point sur la courbe de régression
        ax.scatter([x_current], [mu], color='black', s=50, zorder=5)
        
        # Bande ±2σ
        ax.fill_between([x_current - 0.1, x_current + 0.1], 
                        [mu - 2*sigma, mu - 2*sigma], 
                        [mu + 2*sigma, mu + 2*sigma],
                        alpha=0.2, color=color, zorder=0)
        
        ax.set_xlim(-0.5, 12)
        ax.set_ylim(-2, 8)
        ax.set_xlabel(r'$x$', fontsize=11)
        ax.set_ylabel(r'$y$', fontsize=11)
        sigma_label = r'$\sigma^2$ constant' if idx == 0 else r'$\sigma^2(x)$ variable'
        ax.set_title(f'{title}\n{sigma_label}', fontsize=11)
    
    fig.tight_layout()
    return []

anim = FuncAnimation(fig, animate, init_func=init, frames=60, interval=80, blit=True)
anim.save('_static/regression_scedasticity.gif', writer='pillow', fps=12, dpi=100)
plt.close()

# Afficher le GIF
from IPython.display import Image
Image(filename='_static/regression_scedasticity.gif')
```

```{margin} Pourquoi la gaussienne est-elle verticale?
La gaussienne représente $p(y|x)$: la distribution de $y$ sachant $x$. En régression standard, on suppose que $x$ est mesuré sans erreur et que seul $y$ est bruité. L'objectif des moindres carrés ordinaires minimise donc les distances **verticales**:

$$\sum_{i=1}^N (y_i - f(x_i))^2$$

Si les deux variables avaient de l'incertitude, on utiliserait la **régression orthogonale** (*total least squares*). Dans ce cas, on minimise les distances **perpendiculaires** à la droite:

$$\sum_{i=1}^N \frac{(y_i - \theta_0 - \theta_1 x_i)^2}{1 + \theta_1^2}$$

Le dénominateur $1 + \theta_1^2$ convertit la distance verticale en distance perpendiculaire. Ce modèle est approprié quand $x$ et $y$ sont tous deux des mesures bruitées, par exemple deux instruments mesurant la même quantité physique.
```

L'animation illustre la différence fondamentale entre les deux modèles. À chaque position $x$, la distribution conditionnelle $p(y|x)$ est une gaussienne (la «cloche» colorée) centrée sur la courbe de régression $f_\mu(x)$. Dans le cas **homoscédastique** (gauche), la cloche garde la même largeur partout. Dans le cas **hétéroscédastique** (droite), la largeur varie avec $x$. Ici, l'incertitude augmente vers la droite, ce qui se traduit par une dispersion plus grande des points.

### Classification binaire

La perte 0-1 pour la classification est discontinue, ce qui empêche l'utilisation de méthodes de gradient. La fonction **sigmoïde** $\sigma(z) = 1/(1 + e^{-z})$ contourne ce problème: c'est une **approximation lisse de la fonction échelon** (*step function*). Elle transforme n'importe quel score réel en une valeur dans l'intervalle $(0, 1)$, que nous pouvons interpréter comme une probabilité.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

# Define x range
z = np.linspace(-4, 4, 200)

# Step function (Heaviside)
step = (z >= 0).astype(float)

# Sigmoid function with temperature parameter
def sigmoid(z, alpha=1):
    return 1 / (1 + np.exp(-alpha * z))

# Initialize plot
line_step, = ax.plot(z, step, 'k--', linewidth=2, label='Fonction échelon', alpha=0.7)
line_sigmoid, = ax.plot([], [], 'b-', linewidth=2, label='Sigmoïde $\\sigma(\\alpha z)$')
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax.set_xlim(-4, 4)
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel('$z$')
ax.set_ylabel('$\\sigma(\\alpha z)$')
ax.set_title('Approximation de la fonction échelon par la sigmoïde')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Animation function
def animate(frame):
    # Alpha increases from 0.5 to 10
    alpha = 0.5 + (frame / 100) * 9.5
    y = sigmoid(z, alpha)
    line_sigmoid.set_data(z, y)
    ax.set_title(f'Approximation de la fonction échelon par la sigmoïde ($\\alpha = {alpha:.2f}$)')
    return line_sigmoid,

# Create animation
anim = FuncAnimation(fig, animate, frames=100, interval=50, blit=True, repeat=True)
anim.save('_static/sigmoid_approximation.gif', writer='pillow', fps=20, dpi=100)
plt.close()

# Afficher le GIF
from IPython.display import Image
Image(filename='_static/sigmoid_approximation.gif')
```

L'animation montre comment la sigmoïde $\sigma(\alpha z)$ se rapproche de la fonction échelon lorsque le paramètre $\alpha$ augmente. Pour $\alpha = 1$, la sigmoïde est douce; pour $\alpha$ grand, elle devient presque aussi abrupte que la fonction échelon, tout en restant différentiable.

Cette interprétation probabiliste n'est pas qu'une astuce numérique. Elle correspond exactement à modéliser $Y | \mathbf{X}$ par une distribution de **Bernoulli** dont le paramètre dépend de l'entrée.

Pour la classification binaire avec $y \in \{0, 1\}$, nous modélisons la probabilité de la classe positive par:

$$
p(y = 1 | \mathbf{x}; \boldsymbol{\theta}) = \sigma(f(\mathbf{x}; \boldsymbol{\theta})) = \frac{1}{1 + e^{-f(\mathbf{x}; \boldsymbol{\theta})}}
$$

où $\sigma$ est la fonction sigmoïde et $f(\mathbf{x}; \boldsymbol{\theta})$ est le **logit** (ou log-odds), le score brut du modèle avant transformation. Le logit est le logarithme du rapport des probabilités: $\log \frac{p(y=1|\mathbf{x})}{p(y=0|\mathbf{x})} = \log \frac{p}{1-p}$. La distribution conditionnelle suit une loi de Bernoulli:

$$
p(y|\mathbf{x}; \boldsymbol{\theta}) = \sigma(f(\mathbf{x}; \boldsymbol{\theta}))^y (1 - \sigma(f(\mathbf{x}; \boldsymbol{\theta})))^{1-y}
$$

La log-vraisemblance négative est:

$$
\text{LVN}(\boldsymbol{\theta}) = -\sum_{i=1}^N \left[ y_i \log \sigma(f(\mathbf{x}_i; \boldsymbol{\theta})) + (1-y_i) \log(1 - \sigma(f(\mathbf{x}_i; \boldsymbol{\theta}))) \right]
$$

Cette quantité est l'**entropie croisée binaire**. Elle correspond à la perte logistique, à une reparamétrisation près.

### Classification multiclasse

Pour la classification avec $C$ classes ($C > 2$), nous généralisons le modèle binaire en utilisant la **distribution catégorielle** (ou multinomiale). Au lieu de modéliser une seule probabilité $p(y=1|\mathbf{x})$, nous modélisons un vecteur de probabilités $\boldsymbol{\pi}(\mathbf{x}) = [\pi_1(\mathbf{x}), \ldots, \pi_C(\mathbf{x})]$ où $\pi_c(\mathbf{x}) = p(y=c|\mathbf{x})$ et $\sum_{c=1}^C \pi_c(\mathbf{x}) = 1$.

Pour transformer les scores bruts du modèle en probabilités, nous utilisons la fonction **softmax**:

$$
\pi_c(\mathbf{x}; \boldsymbol{\theta}) = \frac{\exp(f_c(\mathbf{x}; \boldsymbol{\theta}))}{\sum_{j=1}^C \exp(f_j(\mathbf{x}; \boldsymbol{\theta}))}
$$

où $f_c(\mathbf{x}; \boldsymbol{\theta})$ est le score pour la classe $c$. La fonction softmax généralise la sigmoïde au cas multiclasse: elle transforme $C$ scores réels en un vecteur de probabilités qui somme à 1.

La distribution conditionnelle suit une loi catégorielle:

$$
p(y|\mathbf{x}; \boldsymbol{\theta}) = \prod_{c=1}^C \pi_c(\mathbf{x}; \boldsymbol{\theta})^{\mathbf{1}[y = c]}
$$

où $\mathbf{1}[y = c]$ vaut 1 si $y = c$ et 0 sinon. En utilisant l'encodage one-hot $\mathbf{y} = [\mathbf{1}[y=1], \ldots, \mathbf{1}[y=C]]^\top$, cette expression devient:

$$
p(y|\mathbf{x}; \boldsymbol{\theta}) = \prod_{c=1}^C \pi_c(\mathbf{x}; \boldsymbol{\theta})^{y_c}
$$

La log-vraisemblance négative est:

$$
\text{LVN}(\boldsymbol{\theta}) = -\sum_{i=1}^N \sum_{c=1}^C y_{ic} \log \pi_c(\mathbf{x}_i; \boldsymbol{\theta})
$$

où $y_{ic} = \mathbf{1}[y_i = c]$. Cette quantité est l'**entropie croisée multiclasse**. Elle généralise l'entropie croisée binaire au cas où il y a plus de deux classes.

Pour la classification binaire avec $C=2$, le softmax se réduit à la sigmoïde. En effet, si nous définissons $s = f_1(\mathbf{x}) - f_2(\mathbf{x})$, alors:

$$
\pi_1 = \frac{e^{f_1}}{e^{f_1} + e^{f_2}} = \frac{1}{1 + e^{-(f_1 - f_2)}} = \sigma(s)
$$

Le modèle binaire et le modèle multiclasse partagent donc la même structure probabiliste, avec la distribution catégorielle comme généralisation naturelle de la distribution de Bernoulli.

## Maximum a posteriori

Plutôt que de travailler avec la distribution a posteriori complète (ce qui peut être coûteux), nous pouvons chercher son mode: la valeur des paramètres la plus probable a posteriori. C'est l'**estimateur du maximum a posteriori** (MAP):

$$
\hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\max_{\boldsymbol{\theta}} p(\boldsymbol{\theta} | \mathcal{D}) = \arg\max_{\boldsymbol{\theta}} p(\boldsymbol{\theta}) \, p(\mathcal{D} | \boldsymbol{\theta})
$$

Le dénominateur $p(\mathcal{D})$ ne dépend pas de $\boldsymbol{\theta}$ et peut être ignoré pour l'optimisation. En passant au logarithme:

$$
\hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\max_{\boldsymbol{\theta}} \left[ \log p(\mathcal{D} | \boldsymbol{\theta}) + \log p(\boldsymbol{\theta}) \right]
$$

Cette expression révèle une structure familière. Développons la log-vraisemblance et posons $R(\boldsymbol{\theta}) = -\log p(\boldsymbol{\theta})$:

$$
\hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\min_{\boldsymbol{\theta}} \left[ -\log p(\mathcal{D} | \boldsymbol{\theta}) - \log p(\boldsymbol{\theta}) \right] = \arg\min_{\boldsymbol{\theta}} \left[ \underbrace{\sum_{i=1}^N -\log p(y_i | \mathbf{x}_i; \boldsymbol{\theta})}_{\text{LVN}(\boldsymbol{\theta})} + R(\boldsymbol{\theta}) \right]
$$

Comparons avec le risque empirique régularisé introduit au [chapitre 2](ch2_linear_regression.md):

$$
\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta}} \left[ \frac{1}{N} \sum_{i=1}^N \ell(y_i, f(\mathbf{x}_i; \boldsymbol{\theta})) + \lambda \, C(\boldsymbol{\theta}) \right]
$$

La structure est identique: une somme de pertes sur les exemples, plus un terme de régularisation. La différence est l'absence du facteur $1/N$ devant la somme dans l'objectif MAP. Cette différence n'affecte pas le minimiseur (multiplier par une constante positive ne change pas l'argmin), mais elle a une conséquence importante: le poids relatif de l'a priori $R(\boldsymbol{\theta})$ par rapport aux données diminue quand $N$ augmente. Avec plus de données, l'a priori a moins d'influence, ce qui est le comportement souhaité.

La régularisation correspond donc à l'ajout d'un a priori sur les paramètres. Le terme $R(\boldsymbol{\theta}) = -\log p(\boldsymbol{\theta})$ joue le rôle du régulariseur $\lambda \, C(\boldsymbol{\theta})$.

### Le maximum de vraisemblance comme cas particulier

Que se passe-t-il si nous n'avons aucune préférence a priori sur les paramètres? Cela correspond à un a priori **uniforme** (ou constant): $p(\boldsymbol{\theta}) = \text{constante}$.

Dans ce cas, $\log p(\boldsymbol{\theta})$ est une constante qui n'affecte pas l'optimisation, et le MAP se réduit à l'EMV:

$$
\hat{\boldsymbol{\theta}}_{\text{MAP}} = \hat{\boldsymbol{\theta}}_{\text{EMV}} \quad \text{quand } p(\boldsymbol{\theta}) = \text{constante}
$$

L'EMV est donc un cas particulier du MAP: celui où nous supposons implicitement que toutes les valeurs de paramètres sont également plausibles avant d'observer les données. Cette perspective unifie les deux approches dans un même cadre.

### Limites de l'a priori uniforme

L'a priori uniforme (et donc l'EMV) peut être problématique quand les données sont peu nombreuses. Illustrons ceci avec un exemple concret.

```{admonition} Exemple: EMV pour une pièce de monnaie
:class: tip

Supposons que nous lancions une pièce 3 fois et obtenions 3 faces. Quel est l'EMV du paramètre $\theta = P(\text{face})$?

Pour une distribution de Bernoulli, la probabilité d'observer $k$ faces sur $N$ lancers est:

$$
\mathcal{L}(\theta) = \theta^k (1 - \theta)^{N-k}
$$

Avec $k = 3$ et $N = 3$, nous avons $\mathcal{L}(\theta) = \theta^3$. Pour trouver le maximum, passons au logarithme et dérivons:

$$
\log \mathcal{L}(\theta) = 3 \log \theta \quad \Rightarrow \quad \frac{d}{d\theta} \log \mathcal{L}(\theta) = \frac{3}{\theta}
$$

Cette dérivée est toujours positive sur $(0, 1)$: la log-vraisemblance croît avec $\theta$. Le maximum est donc atteint à la borne $\theta = 1$.

Résultat: $\hat{\theta}_{\text{EMV}} = 1$. L'EMV prédit que la pièce tombe toujours sur face!

Dans le cas général avec $k$ faces sur $N$ lancers, en posant la dérivée égale à zéro:

$$
\frac{d}{d\theta} \log \mathcal{L}(\theta) = \frac{k}{\theta} - \frac{N - k}{1 - \theta} = 0 \quad \Rightarrow \quad \hat{\theta}_{\text{EMV}} = \frac{k}{N}
$$

L'EMV est simplement la fréquence empirique des faces.
```

Cette estimation de 100% est peu plausible pour une vraie pièce. Le problème est que l'EMV (avec son a priori uniforme implicite) n'a aucun mécanisme pour modérer les estimations extrêmes quand les données sont peu nombreuses. Un a priori informatif peut atténuer ce problème.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# MLE vs MAP for Bernoulli with few observations
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Different sample sizes
samples_list = [
    [1, 1, 1],           # 3 heads
    [1, 1, 1, 0],        # 3 heads, 1 tail
    [1, 1, 1, 0, 0, 1, 1, 0, 1, 1]  # 7 heads, 3 tails
]

theta_grid = np.linspace(0.001, 0.999, 200)

for ax, samples in zip(axes, samples_list):
    n1 = sum(samples)  # heads
    n0 = len(samples) - n1  # tails
    
    # MLE
    theta_mle = n1 / (n0 + n1)
    
    # Likelihood (unnormalized)
    likelihood = theta_grid**n1 * (1 - theta_grid)**n0
    likelihood = likelihood / likelihood.max()
    
    ax.plot(theta_grid, likelihood, 'b-', linewidth=2, label='Vraisemblance')
    ax.axvline(theta_mle, color='b', linestyle='--', alpha=0.7,
               label=f'EMV: {theta_mle:.2f}')
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label=r'$\theta = 0.5$')
    
    ax.set_xlabel(r'$\theta$')
    ax.set_ylabel('Vraisemblance (normalisée)')
    ax.set_title(f'{n1} faces, {n0} piles (N={len(samples)})')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

plt.tight_layout()
```

La figure montre la vraisemblance pour différents échantillons. Avec seulement 3 observations (toutes faces), la vraisemblance est maximale à $\theta = 1$. En augmentant la taille de l'échantillon, l'estimation devient plus raisonnable. Voyons comment un a priori non uniforme peut aider.

### Exemple: lissage de Laplace

Revenons à notre exemple de la pièce de monnaie. Utilisons un a priori **Beta** sur $\theta$:

$$
p(\theta) = \text{Beta}(\theta | a, b) \propto \theta^{a-1} (1-\theta)^{b-1}
$$

Les paramètres $a$ et $b$ contrôlent la forme de l'a priori. Pour $a = b = 2$, l'a priori favorise des valeurs de $\theta$ proches de 0,5.

Le logarithme de l'a posteriori (vraisemblance plus a priori) est:

$$
\log p(\theta | \mathcal{D}) \propto N_1 \log \theta + N_0 \log(1-\theta) + (a-1) \log \theta + (b-1) \log(1-\theta)
$$

En dérivant et en résolvant, l'estimateur MAP est:

$$
\hat{\theta}_{\text{MAP}} = \frac{N_1 + a - 1}{N_1 + N_0 + a + b - 2}
$$

Avec $a = b = 2$ et nos 3 observations de faces:

$$
\hat{\theta}_{\text{MAP}} = \frac{3 + 2 - 1}{3 + 0 + 2 + 2 - 2} = \frac{4}{5} = 0,8
$$

Cette estimation est plus raisonnable que l'EMV $\hat{\theta}_{\text{EMV}} = 1$. L'a priori «tire» l'estimation vers des valeurs moins extrêmes.

Le choix $a = b = 2$ correspond au **lissage de Laplace** (ou *add-one smoothing*): c'est comme si nous avions observé une face et une pile supplémentaires avant de commencer. Cette technique est particulièrement utile quand certains événements n'ont jamais été observés dans les données.

```{code-cell} python
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

theta_grid = np.linspace(0.001, 0.999, 200)

# Left: Prior, likelihood, posterior
ax = axes[0]
n1, n0 = 3, 0  # 3 heads, 0 tails
a, b = 2, 2    # Beta prior parameters

# Prior
prior = stats.beta.pdf(theta_grid, a, b)
prior = prior / prior.max()

# Likelihood
likelihood = theta_grid**n1 * (1 - theta_grid)**n0
likelihood = likelihood / likelihood.max()

# Posterior (Beta(a + n1, b + n0))
posterior = stats.beta.pdf(theta_grid, a + n1, b + n0)
posterior = posterior / posterior.max()

ax.plot(theta_grid, prior, 'g-', linewidth=2, label='A priori Beta(2,2)')
ax.plot(theta_grid, likelihood, 'b--', linewidth=2, label='Vraisemblance')
ax.plot(theta_grid, posterior, 'r-', linewidth=2, label='A posteriori')

theta_mle = n1 / (n1 + n0)
theta_map = (n1 + a - 1) / (n1 + n0 + a + b - 2)
ax.axvline(theta_mle, color='b', linestyle=':', alpha=0.7, label=f'EMV: {theta_mle:.2f}')
ax.axvline(theta_map, color='r', linestyle=':', alpha=0.7, label=f'MAP: {theta_map:.2f}')

ax.set_xlabel(r'$\theta$')
ax.set_ylabel('Densité (normalisée)')
ax.set_title('3 faces, 0 pile')
ax.legend(fontsize=8)
ax.set_xlim(0, 1)

# Right: Effect of different priors
ax = axes[1]
priors = [(1, 1, 'Uniforme'), (2, 2, 'Beta(2,2)'), (5, 5, 'Beta(5,5)')]

for a, b, label in priors:
    theta_map = (n1 + a - 1) / (n1 + n0 + a + b - 2)
    posterior = stats.beta.pdf(theta_grid, a + n1, b + n0)
    posterior = posterior / posterior.max()
    ax.plot(theta_grid, posterior, linewidth=2, label=f'{label}: MAP={theta_map:.2f}')

ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5, label='EMV: 1.00')
ax.set_xlabel(r'$\theta$')
ax.set_ylabel('A posteriori (normalisé)')
ax.set_title('Effet de différents a priori')
ax.legend(fontsize=8)
ax.set_xlim(0, 1)

plt.tight_layout()
```

La figure de gauche montre comment l'a posteriori combine l'a priori et la vraisemblance. L'a priori Beta(2,2) «tire» l'estimation vers 0,5, résultant en un MAP de 0,8 au lieu de l'EMV de 1,0. La figure de droite montre l'effet de différents a priori: plus l'a priori est fort (variance faible), plus l'estimation est proche de 0,5.

### Régression ridge = MAP avec a priori gaussien

Appliquons maintenant ce cadre bayésien à la régression linéaire. Si nous plaçons un a priori gaussien isotrope sur les paramètres:

$$
p(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta} | \mathbf{0}, \sigma_\theta^2 \mathbf{I})
$$

cet a priori exprime la croyance que les paramètres sont probablement proches de zéro, avec une incertitude contrôlée par $\sigma_\theta^2$.

Le logarithme négatif de cet a priori est:

$$
-\log p(\boldsymbol{\theta}) = \frac{1}{2\sigma_\theta^2} \|\boldsymbol{\theta}\|_2^2 + \text{constante}
$$

L'estimateur MAP devient:

$$
\hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\min_{\boldsymbol{\theta}} \left[ \text{LVN}(\boldsymbol{\theta}) + \frac{1}{2\sigma_\theta^2}\|\boldsymbol{\theta}\|_2^2 \right]
$$

C'est exactement Ridge, avec $\lambda = 1/(2\sigma_\theta^2)$. Cette correspondance nous donne une interprétation de l'hyperparamètre:

- **Grande valeur de $\lambda$** (petite variance $\sigma_\theta^2$): forte croyance que les paramètres sont proches de zéro
- **Petite valeur de $\lambda$** (grande variance $\sigma_\theta^2$): a priori peu informatif, on fait confiance aux données

L'a priori gaussien sur les paramètres est parfois appelé **dégradation des poids** (*weight decay*) dans le contexte des réseaux de neurones, car il «tire» les paramètres vers zéro pendant l'entraînement.

## Unification: deux langages pour un même problème

Les sections précédentes ont présenté deux approches pour l'apprentissage supervisé. La première, fondée sur la théorie de la décision, définit une fonction de perte et minimise le risque empirique. La seconde, probabiliste, modélise la distribution des données et estime les paramètres par maximum de vraisemblance ou maximum a posteriori.

Ces deux approches semblent différentes, mais elles aboutissent aux mêmes algorithmes. En choisissant la **perte logarithmique** $\ell(y, \hat{y}) = -\log p(y | \hat{y})$, le risque empirique devient exactement la log-vraisemblance négative (à un facteur $1/N$ près). Minimiser l'un revient à minimiser l'autre. Sous bruit gaussien, cette perte se réduit à la perte quadratique; sous modèle de Bernoulli, à l'entropie croisée.

De même, ajouter une régularisation $\ell_2$ au risque empirique revient à supposer un a priori gaussien sur les paramètres. Ridge n'est rien d'autre que l'estimation MAP avec cet a priori. Le coefficient $\lambda$ encode la force de notre croyance a priori: plus $\lambda$ est grand, plus nous «tirons» les paramètres vers zéro.

Pourquoi alors utiliser deux langages? Parce qu'ils éclairent des aspects différents du problème. Le langage décisionnel (risque, perte, minimisation) est opérationnel: il dit comment construire un algorithme. Le langage probabiliste (vraisemblance, a priori, a posteriori) est interprétatif: il dit ce que nous supposons sur les données et pourquoi nos choix sont raisonnables. Ensemble, ils permettent de *concevoir* des algorithmes et de *comprendre* leur comportement.

### Interprétation informationnelle

La théorie de l'information offre une troisième perspective sur l'apprentissage, et révèle une connexion profonde entre le maximum de vraisemblance et la notion de distance entre distributions.

#### Entropie et quantité d'information

L'**entropie** d'une distribution discrète $p$ mesure l'incertitude moyenne ou la quantité d'information nécessaire pour décrire un tirage:

$$
\mathbb{H}(p) = -\sum_y p(y) \log p(y)
$$

Une distribution concentrée (faible incertitude) a une entropie faible. Une distribution uniforme (incertitude maximale) a une entropie maximale. L'entropie est toujours positive ou nulle.

Pour une distribution continue, l'**entropie différentielle** est définie de manière analogue:

$$
\mathbb{H}(p) = -\int p(y) \log p(y) \, dy
$$

Par exemple, une gaussienne $\mathcal{N}(\mu, \sigma^2)$ a une entropie $\frac{1}{2}\log(2\pi e \sigma^2)$: plus la variance est grande, plus l'incertitude est grande.

#### La divergence de Kullback-Leibler

La **divergence de Kullback-Leibler** (ou divergence KL) mesure combien une distribution $q$ diffère d'une distribution de référence $p$:

$$
D_{\text{KL}}(p \| q) = \sum_y p(y) \log \frac{p(y)}{q(y)} = \mathbb{E}_{y \sim p}\left[\log \frac{p(y)}{q(y)}\right]
$$

Cette quantité a plusieurs propriétés importantes:
- **Non-négative**: $D_{\text{KL}}(p \| q) \geq 0$ toujours (inégalité de Gibbs)
- **Nulle si et seulement si** $p = q$ (les distributions sont identiques)
- **Non symétrique**: $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$ en général

On peut réécrire la divergence KL en faisant apparaître l'entropie:

$$
D_{\text{KL}}(p \| q) = -\mathbb{H}(p) + \mathbb{H}_{\text{ce}}(p, q)
$$

où $\mathbb{H}_{\text{ce}}(p, q) = -\sum_y p(y) \log q(y)$ est l'**entropie croisée** entre $p$ et $q$.

#### L'EMV minimise la divergence KL

La **distribution empirique** place une masse $1/N$ sur chaque observation:

$$
p_{\mathcal{D}}(y) = \frac{1}{N} \sum_{i=1}^N \delta(y - y_i)
$$

Considérons maintenant la divergence KL entre cette distribution empirique et notre modèle paramétrique $p(y | \boldsymbol{\theta})$:

$$
D_{\text{KL}}(p_{\mathcal{D}} \| p(\cdot | \boldsymbol{\theta})) = -\mathbb{H}(p_{\mathcal{D}}) + \mathbb{H}_{\text{ce}}(p_{\mathcal{D}}, p(\cdot | \boldsymbol{\theta}))
$$

Le terme $\mathbb{H}(p_{\mathcal{D}})$ ne dépend pas de $\boldsymbol{\theta}$. Donc:

$$
\arg\min_{\boldsymbol{\theta}} D_{\text{KL}}(p_{\mathcal{D}} \| p(\cdot|\boldsymbol{\theta})) = \arg\min_{\boldsymbol{\theta}} \mathbb{H}_{\text{ce}}(p_{\mathcal{D}}, p(\cdot|\boldsymbol{\theta}))
$$

L'entropie croisée avec la distribution empirique est exactement la log-vraisemblance négative:

$$
\mathbb{H}_{\text{ce}}(p_{\mathcal{D}}, p(\cdot|\boldsymbol{\theta})) = -\frac{1}{N} \sum_{i=1}^N \log p(y_i | \mathbf{x}_i; \boldsymbol{\theta}) = \text{LVN}(\boldsymbol{\theta})
$$

Ainsi, **le maximum de vraisemblance trouve les paramètres qui minimisent la divergence KL entre notre modèle et la distribution empirique**. C'est une interprétation géométrique de l'EMV: nous cherchons le modèle paramétrique le plus «proche» des données observées, au sens de la divergence KL.

#### Unification: régression et classification

Cette perspective informationnelle unifie nos deux modèles principaux:

| Problème | Modèle probabiliste | LVN à minimiser |
|----------|---------------------|-----------------|
| Régression | $p(y|\mathbf{x}; \boldsymbol{\theta}) = \mathcal{N}(y | f(\mathbf{x}; \boldsymbol{\theta}), \sigma^2)$ | $\frac{1}{2\sigma^2}\sum_i (y_i - f(\mathbf{x}_i; \boldsymbol{\theta}))^2 + \text{cst}$ |
| Classification binaire | $p(y|\mathbf{x}; \boldsymbol{\theta}) = \text{Ber}(y | \sigma(\boldsymbol{\theta}^\top\mathbf{x}))$ | $-\sum_i [y_i \log \mu_i + (1-y_i)\log(1-\mu_i)]$ |

Dans les deux cas:
1. Nous spécifions un modèle probabiliste pour les observations
2. La log-vraisemblance négative définit la fonction de perte
3. Minimiser cette perte revient à minimiser la divergence KL avec la distribution empirique

La perte quadratique n'est pas un choix arbitraire: elle découle de l'hypothèse que le bruit est gaussien. L'entropie croisée n'est pas arbitraire non plus: elle découle de l'hypothèse que les étiquettes suivent une distribution de Bernoulli (binaire) ou catégorielle (multiclasse).

Cette unification révèle que **le choix de la fonction de perte encode notre hypothèse sur la distribution du bruit**. Changer de modèle probabiliste change la perte optimale. Inversement, choisir une perte particulière revient implicitement à supposer un certain modèle de bruit.

## Résumé

Ce chapitre a développé le cadre probabiliste pour l'apprentissage supervisé:

- Le **cadre bayésien** utilise le théorème de Bayes pour mettre à jour nos croyances sur les paramètres après avoir observé des données. L'a posteriori combine l'a priori (nos croyances initiales) et la vraisemblance (ce que les données nous disent).

- Le **prédicteur de Bayes optimal** minimise le risque pour une perte donnée, en supposant la connaissance de la vraie distribution. Pour la perte quadratique, c'est la moyenne conditionnelle; pour la perte 0-1, c'est le mode conditionnel.

- Le **maximum de vraisemblance** (EMV) trouve les paramètres qui rendent les données observées les plus probables. Sous bruit gaussien, l'EMV coïncide avec les moindres carrés.

- Le **maximum a posteriori** (MAP) incorpore un a priori sur les paramètres. Avec un a priori gaussien, le MAP correspond exactement à Ridge. Le coefficient de régularisation encode la force de l'a priori.

- L'**interprétation informationnelle** révèle que l'EMV minimise la divergence KL entre notre modèle et la distribution empirique. Cette perspective unifie régression (bruit gaussien → perte quadratique) et classification (distribution de Bernoulli → entropie croisée).

- Les trois perspectives (décisionnelle, probabiliste, informationnelle) sont complémentaires: la première est opérationnelle (comment construire l'algorithme), la deuxième est interprétative (ce que nous supposons sur les données), la troisième est géométrique (quelle distance minimisons-nous).

Le chapitre suivant étend ces fondations aux **réseaux de neurones**, où la capacité d'apprendre des représentations non linéaires ouvre de nouvelles possibilités et de nouveaux défis.

## Exercices

````{admonition} Exercice 1: Maximum de vraisemblance
:class: hint dropdown

Soit $\{y_1, \ldots, y_N\}$ un échantillon i.i.d. d'une distribution exponentielle de paramètre $\lambda > 0$:

$$
p(y | \lambda) = \lambda e^{-\lambda y}, \quad y \geq 0
$$

1. Écrivez la vraisemblance $\mathcal{L}(\lambda)$ et la log-vraisemblance $\log \mathcal{L}(\lambda)$.

2. Dérivez l'estimateur du maximum de vraisemblance (EMV) $\hat{\lambda}_{\text{EMV}}$.

3. Si les observations sont $y = \{0.5, 1.2, 0.8, 2.1, 0.3\}$, calculez $\hat{\lambda}_{\text{EMV}}$.
````

```{admonition} Solution Exercice 1
:class: dropdown

1. **Vraisemblance et log-vraisemblance:**

   $$\mathcal{L}(\lambda) = \prod_{i=1}^N \lambda e^{-\lambda y_i} = \lambda^N \exp\left(-\lambda \sum_{i=1}^N y_i\right)$$
   
   $$\log \mathcal{L}(\lambda) = N \log \lambda - \lambda \sum_{i=1}^N y_i$$

2. **Dérivation de l'EMV:**

   On dérive par rapport à $\lambda$ et on égale à zéro:
   
   $$\frac{d}{d\lambda} \log \mathcal{L}(\lambda) = \frac{N}{\lambda} - \sum_{i=1}^N y_i = 0$$
   
   D'où:
   
   $$\hat{\lambda}_{\text{EMV}} = \frac{N}{\sum_{i=1}^N y_i} = \frac{1}{\bar{y}}$$
   
   L'EMV est l'inverse de la moyenne empirique.

3. **Application numérique:**

   $\bar{y} = \frac{0.5 + 1.2 + 0.8 + 2.1 + 0.3}{5} = \frac{4.9}{5} = 0.98$
   
   $\hat{\lambda}_{\text{EMV}} = \frac{1}{0.98} \approx 1.02$
```

````{admonition} Exercice 2: Maximum de vraisemblance pour classification binaire ★
:class: hint dropdown

Soit un problème de classification binaire avec $N$ observations $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$ où $y_i \in \{0, 1\}$. On modélise la probabilité de la classe positive par:

$$
p(y = 1 | \mathbf{x}; \boldsymbol{\theta}) = \sigma(\boldsymbol{\theta}^\top \mathbf{x}) = \frac{1}{1 + e^{-\boldsymbol{\theta}^\top \mathbf{x}}}
$$

1. Écrivez la vraisemblance $\mathcal{L}(\boldsymbol{\theta})$ pour $N$ observations i.i.d. suivant une distribution de Bernoulli.

2. Écrivez la log-vraisemblance $\log \mathcal{L}(\boldsymbol{\theta})$.

3. Montrez que maximiser la log-vraisemblance revient à minimiser l'entropie croisée binaire.

4. **Application**: Pour les observations $(y_1, y_2, y_3) = (1, 0, 1)$ avec les scores $\boldsymbol{\theta}^\top \mathbf{x}_i$ égaux à $(2, -1, 0.5)$, calculez la log-vraisemblance.
````

```{admonition} Solution Exercice 2
:class: dropdown

1. **Vraisemblance:**

   Chaque observation suit une loi de Bernoulli: $p(y_i | \mathbf{x}_i; \boldsymbol{\theta}) = \sigma_i^{y_i} (1 - \sigma_i)^{1-y_i}$ où $\sigma_i = \sigma(\boldsymbol{\theta}^\top \mathbf{x}_i)$.
   
   Pour $N$ observations i.i.d.:
   
   $$
   \mathcal{L}(\boldsymbol{\theta}) = \prod_{i=1}^N \sigma_i^{y_i} (1 - \sigma_i)^{1-y_i}
   $$

2. **Log-vraisemblance:**

   $$
   \log \mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^N \left[ y_i \log \sigma_i + (1-y_i) \log(1 - \sigma_i) \right]
   $$

3. **Lien avec l'entropie croisée:**

   L'entropie croisée binaire (ECB) est définie comme:
   
   $$
   \text{ECB}(\boldsymbol{\theta}) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log \sigma_i + (1-y_i) \log(1 - \sigma_i) \right]
   $$
   
   On voit que $\text{ECB}(\boldsymbol{\theta}) = -\frac{1}{N} \log \mathcal{L}(\boldsymbol{\theta})$.
   
   Donc: $\max_{\boldsymbol{\theta}} \log \mathcal{L}(\boldsymbol{\theta}) \Leftrightarrow \min_{\boldsymbol{\theta}} \text{ECB}(\boldsymbol{\theta})$

4. **Application numérique:**

   - $\sigma_1 = \sigma(2) = 1/(1 + e^{-2}) \approx 0.881$
   - $\sigma_2 = \sigma(-1) = 1/(1 + e^{1}) \approx 0.269$
   - $\sigma_3 = \sigma(0.5) = 1/(1 + e^{-0.5}) \approx 0.622$
   
   Log-vraisemblance:
   $$
   \log \mathcal{L} = 1 \cdot \log(0.881) + 1 \cdot \log(1 - 0.269) + 1 \cdot \log(0.622)
   $$
   $$
   = \log(0.881) + \log(0.731) + \log(0.622) \approx -0.127 - 0.313 - 0.475 = -0.915
   $$
```

````{admonition} Exercice 3: EMV comme cas particulier de MAP ★
:class: hint dropdown

Le théorème de Bayes nous donne la distribution a posteriori des paramètres:

$$
p(\boldsymbol{\theta} | \mathcal{D}) = \frac{p(\mathcal{D} | \boldsymbol{\theta}) \, p(\boldsymbol{\theta})}{p(\mathcal{D})}
$$

L'estimateur MAP maximise cette distribution a posteriori.

1. Écrivez la log-posterior $\log p(\boldsymbol{\theta} | \mathcal{D})$ en fonction de la log-vraisemblance et du log a priori.

2. Supposons un a priori uniforme (constant): $p(\boldsymbol{\theta}) = c$ pour tout $\boldsymbol{\theta}$. Montrez que l'estimateur MAP se réduit à l'estimateur du maximum de vraisemblance (EMV).

3. Pour quels autres types d'a priori l'EMV et le MAP coïncident-ils?

4. Expliquez pourquoi utiliser un a priori uniforme peut être problématique dans certains cas.
````

```{admonition} Solution Exercice 3
:class: dropdown

1. **Log-posterior:**

   $$
   \log p(\boldsymbol{\theta} | \mathcal{D}) = \log p(\mathcal{D} | \boldsymbol{\theta}) + \log p(\boldsymbol{\theta}) - \log p(\mathcal{D})
   $$
   
   Le terme $\log p(\mathcal{D})$ ne dépend pas de $\boldsymbol{\theta}$, donc pour l'optimisation:
   
   $$
   \hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\max_{\boldsymbol{\theta}} \left[ \log p(\mathcal{D} | \boldsymbol{\theta}) + \log p(\boldsymbol{\theta}) \right]
   $$

2. **A priori uniforme:**

   Si $p(\boldsymbol{\theta}) = c$ (constante), alors $\log p(\boldsymbol{\theta}) = \log c$ est aussi une constante.
   
   $$
   \hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\max_{\boldsymbol{\theta}} \left[ \log p(\mathcal{D} | \boldsymbol{\theta}) + \log c \right] = \arg\max_{\boldsymbol{\theta}} \log p(\mathcal{D} | \boldsymbol{\theta}) = \hat{\boldsymbol{\theta}}_{\text{EMV}}
   $$
   
   L'EMV est donc un cas particulier du MAP avec a priori uniforme.

3. **Autres a priori:**

   L'EMV et le MAP coïncident pour tout a priori qui est constant sur le domaine des paramètres, ou plus généralement, pour tout a priori dont le log est constant (à une constante additive près). Cela inclut les a priori impropres (non normalisables) qui sont uniformes sur $\mathbb{R}^d$.

4. **Problèmes de l'a priori uniforme:**

   - **Avec peu de données**: l'EMV peut être extrême. Exemple: 3 lancers de pièce donnant 3 faces → EMV = 100% de probabilité de face.
   - **Paramètres non bornés**: un a priori uniforme sur $\mathbb{R}$ n'est pas une vraie distribution de probabilité (a priori impropre).
   - **Invariance**: un a priori uniforme sur $\theta$ n'est pas uniforme sur $g(\theta)$ pour une transformation non-linéaire $g$.
   - **Pas d'information**: on ignore toute connaissance préalable sur les valeurs plausibles des paramètres.
```

````{admonition} Exercice 4: Softmax et classification multiclasse ★★
:class: hint dropdown

La fonction softmax transforme un vecteur de scores en un vecteur de probabilités:

$$
\text{softmax}(\mathbf{s})_c = \frac{e^{s_c}}{\sum_{j=1}^C e^{s_j}}
$$

1. Pour $C = 3$ classes et les scores $\mathbf{s} = [2, 1, 0]$, calculez manuellement $\text{softmax}(\mathbf{s})$. Vérifiez que les probabilités somment à 1.

2. Montrez que le softmax est invariant par translation: $\text{softmax}(\mathbf{s} + c\mathbf{1}) = \text{softmax}(\mathbf{s})$ pour tout scalaire $c$.

3. Pour $C = 2$ classes, montrez que le softmax se réduit à la sigmoïde. Posez $s = s_1 - s_2$ et montrez que $\text{softmax}(\mathbf{s})_1 = \sigma(s)$.

4. Pour un problème à 3 classes avec les vraies étiquettes one-hot $\mathbf{y} = [0, 1, 0]$ (classe 2) et les probabilités prédites $\hat{\mathbf{p}} = [0.1, 0.7, 0.2]$, calculez l'entropie croisée.
````

```{admonition} Solution Exercice 4
:class: dropdown

1. **Calcul du softmax:**

   $$
   \text{dénominateur} = e^2 + e^1 + e^0 = 7.389 + 2.718 + 1 = 11.107
   $$
   
   $$
   \text{softmax}([2, 1, 0]) = \left[ \frac{7.389}{11.107}, \frac{2.718}{11.107}, \frac{1}{11.107} \right] = [0.665, 0.245, 0.090]
   $$
   
   Vérification: $0.665 + 0.245 + 0.090 = 1.000$ ✓

2. **Invariance par translation:**

   $$
   \text{softmax}(\mathbf{s} + c\mathbf{1})_c = \frac{e^{s_c + c}}{\sum_j e^{s_j + c}} = \frac{e^{s_c} \cdot e^c}{\sum_j e^{s_j} \cdot e^c} = \frac{e^c \cdot e^{s_c}}{e^c \cdot \sum_j e^{s_j}} = \frac{e^{s_c}}{\sum_j e^{s_j}} = \text{softmax}(\mathbf{s})_c
   $$
   
   Cette propriété est utile numériquement: on peut soustraire $\max_j s_j$ pour éviter les débordements.

3. **Cas $C = 2$:**

   $$
   \text{softmax}([s_1, s_2])_1 = \frac{e^{s_1}}{e^{s_1} + e^{s_2}} = \frac{1}{1 + e^{s_2 - s_1}} = \frac{1}{1 + e^{-(s_1 - s_2)}} = \sigma(s_1 - s_2)
   $$
   
   En posant $s = s_1 - s_2$, on retrouve bien $\sigma(s) = 1/(1 + e^{-s})$.

4. **Entropie croisée:**

   $$
   \text{EC} = -\sum_{c=1}^C y_c \log \hat{p}_c = -(0 \cdot \log 0.1 + 1 \cdot \log 0.7 + 0 \cdot \log 0.2)
   $$
   
   $$
   = -\log 0.7 \approx -(-0.357) = 0.357
   $$
   
   Seule la composante correspondant à la vraie classe contribue à la perte.
```

````{admonition} Exercice 5: MAP avec a priori gaussien et Ridge ★★
:class: hint dropdown

Considérons un modèle de régression linéaire gaussien:

$$
p(y | \mathbf{x}, \boldsymbol{\theta}) = \mathcal{N}(y | \boldsymbol{\theta}^\top \mathbf{x}, \sigma^2)
$$

avec un a priori gaussien isotrope sur les paramètres:

$$
p(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta} | \mathbf{0}, \tau^2 \mathbf{I})
$$

1. Écrivez la log-vraisemblance pour $N$ observations i.i.d.

2. Écrivez le log a priori $\log p(\boldsymbol{\theta})$.

3. Montrez que l'estimateur MAP s'écrit:

   $$
   \hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\min_{\boldsymbol{\theta}} \left[ \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2 + \lambda \|\boldsymbol{\theta}\|^2 \right]
   $$
   
   et identifiez $\lambda$ en fonction de $\sigma^2$ et $\tau^2$.

4. Interprétez: que signifie un grand $\tau^2$? Un petit $\tau^2$?
````

```{admonition} Solution Exercice 5
:class: dropdown

1. **Log-vraisemblance:**

   $$
   \log p(\mathbf{y} | \mathbf{X}, \boldsymbol{\theta}) = \sum_{i=1}^N \log \mathcal{N}(y_i | \boldsymbol{\theta}^\top \mathbf{x}_i, \sigma^2)
   $$
   
   $$
   = -\frac{N}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - \boldsymbol{\theta}^\top \mathbf{x}_i)^2
   $$
   
   $$
   = -\frac{N}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
   $$

2. **Log a priori:**

   $$
   \log p(\boldsymbol{\theta}) = -\frac{d}{2} \log(2\pi\tau^2) - \frac{1}{2\tau^2} \|\boldsymbol{\theta}\|^2
   $$

3. **Estimateur MAP:**

   $$
   \hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\max_{\boldsymbol{\theta}} \left[ \log p(\mathbf{y} | \mathbf{X}, \boldsymbol{\theta}) + \log p(\boldsymbol{\theta}) \right]
   $$
   
   En ignorant les constantes:
   
   $$
   = \arg\max_{\boldsymbol{\theta}} \left[ -\frac{1}{2\sigma^2} \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2 - \frac{1}{2\tau^2} \|\boldsymbol{\theta}\|^2 \right]
   $$
   
   En multipliant par $-2\sigma^2$ (qui ne change pas l'argmax):
   
   $$
   = \arg\min_{\boldsymbol{\theta}} \left[ \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2 + \frac{\sigma^2}{\tau^2} \|\boldsymbol{\theta}\|^2 \right]
   $$
   
   Donc $\boxed{\lambda = \sigma^2 / \tau^2}$.

4. **Interprétation:**

   - **Grand $\tau^2$** (a priori large): on est peu sûr que les paramètres sont proches de zéro, donc $\lambda$ petit, peu de régularisation, MAP proche de EMV.
   - **Petit $\tau^2$** (a priori concentré): on croit fortement que les paramètres sont proches de zéro, donc $\lambda$ grand, forte régularisation, coefficients tirés vers zéro.
   
   Le rapport $\sigma^2/\tau^2$ compare l'incertitude dans les données ($\sigma^2$) à l'incertitude dans l'a priori ($\tau^2$). Plus les données sont bruitées, plus on fait confiance à l'a priori.
```

````{admonition} Exercice 6: Inférence bayésienne complète ★★★
:class: hint dropdown

L'inférence bayésienne complète calcule la distribution a posteriori des paramètres, pas seulement son mode (MAP).

Considérons un modèle de régression linéaire bayésien avec:
- Vraisemblance: $p(y | x, \theta) = \mathcal{N}(y | \theta x, \sigma^2)$
- Prior: $p(\theta) = \mathcal{N}(\theta | 0, \tau^2)$

Supposons $\sigma^2 = 1$ et $\tau^2 = 1$.

1. Pour une seule observation $(x_1, y_1) = (2, 3)$, calculez la distribution a posteriori $p(\theta | x_1, y_1)$. Utilisez le fait que le produit de deux gaussiennes est une gaussienne.

2. Quelle est la moyenne a posteriori $\mu_{\text{post}}$ et la variance a posteriori $\sigma^2_{\text{post}}$?

3. Calculez l'estimateur MAP $\hat{\theta}_{\text{MAP}}$ et comparez avec $\mu_{\text{post}}$.

4. Ajoutez une deuxième observation $(x_2, y_2) = (1, 0.5)$. Mettez à jour la distribution a posteriori.

5. Tracez les distributions a priori, a posteriori après 1 observation, et a posteriori après 2 observations. Que remarquez-vous sur l'évolution de l'incertitude?
````

````{admonition} Solution Exercice 6
:class: dropdown

1. **Calcul de la distribution a posteriori:**

   La distribution a posteriori est proportionnelle à:
   $$
   p(\theta | y_1) \propto p(y_1 | \theta) \cdot p(\theta)
   $$
   
   En prenant le log:
   $$
   \log p(\theta | y_1) \propto -\frac{(y_1 - \theta x_1)^2}{2\sigma^2} - \frac{\theta^2}{2\tau^2}
   $$
   
   C'est une forme quadratique en $\theta$, donc la distribution a posteriori est gaussienne.

2. **Paramètres de la distribution a posteriori:**

   Pour le modèle conjugué gaussien-gaussien:
   
   $$
   \sigma^2_{\text{post}} = \left( \frac{1}{\tau^2} + \frac{x_1^2}{\sigma^2} \right)^{-1} = \left( 1 + 4 \right)^{-1} = 0.2
   $$
   
   $$
   \mu_{\text{post}} = \sigma^2_{\text{post}} \cdot \frac{x_1 y_1}{\sigma^2} = 0.2 \times \frac{2 \times 3}{1} = 1.2
   $$
   
   Donc $p(\theta | x_1, y_1) = \mathcal{N}(\theta | 1.2, 0.2)$.

3. **Estimateur MAP:**

   Pour une distribution a posteriori gaussienne, le mode = la moyenne:
   $$
   \hat{\theta}_{\text{MAP}} = \mu_{\text{post}} = 1.2
   $$
   
   Pour un modèle gaussien conjugué, MAP = moyenne a posteriori.

4. **Mise à jour séquentielle:**

   On utilise la distribution a posteriori après la première observation comme nouvel a priori:
   
   $$
   \sigma^2_{\text{post,2}} = \left( \frac{1}{0.2} + \frac{1^2}{1} \right)^{-1} = \left( 5 + 1 \right)^{-1} = \frac{1}{6} \approx 0.167
   $$
   
   $$
   \mu_{\text{post,2}} = \sigma^2_{\text{post,2}} \cdot \left( \frac{1.2}{0.2} + \frac{1 \times 0.5}{1} \right) = \frac{1}{6} \times (6 + 0.5) = 1.083
   $$

5. **Visualisation:**

   ```python
   theta = np.linspace(-2, 3, 200)
   prior = scipy.stats.norm.pdf(theta, 0, 1)
   post1 = scipy.stats.norm.pdf(theta, 1.2, np.sqrt(0.2))
   post2 = scipy.stats.norm.pdf(theta, 1.083, np.sqrt(1/6))
   
   plt.plot(theta, prior, label='A priori')
   plt.plot(theta, post1, label='A posteriori (1 obs)')
   plt.plot(theta, post2, label='A posteriori (2 obs)')
   plt.legend()
   ```
   
   Observations:
   - L'a priori est large (grande incertitude)
   - Après 1 observation, la distribution a posteriori se concentre autour de 1.2
   - Après 2 observations, la distribution a posteriori se concentre davantage (variance diminue)
   - La moyenne a posteriori est une moyenne pondérée de l'a priori et des données
````

````{admonition} Exercice 7: Entropie croisée et divergence de Kullback-Leibler ★★★
:class: hint dropdown

La divergence de Kullback-Leibler (KL) mesure la différence entre deux distributions. L'entropie croisée est liée à la KL divergence.

Définitions pour des distributions discrètes $p$ (vraie) et $q$ (modèle):

$$
H(p) = -\sum_i p_i \log p_i \quad \text{(entropie)}
$$

$$
H(p, q) = -\sum_i p_i \log q_i \quad \text{(entropie croisée)}
$$

$$
D_{\text{KL}}(p \| q) = \sum_i p_i \log \frac{p_i}{q_i} \quad \text{(divergence KL)}
$$

1. Montrez que $H(p, q) = H(p) + D_{\text{KL}}(p \| q)$.

2. Pour la vraie distribution $p = [0.7, 0.3]$ et deux modèles $q_1 = [0.6, 0.4]$ et $q_2 = [0.9, 0.1]$, calculez $D_{\text{KL}}(p \| q_1)$ et $D_{\text{KL}}(p \| q_2)$. Quel modèle est «meilleur»?

3. Montrez que minimiser l'entropie croisée $H(p, q)$ par rapport à $q$ revient à minimiser $D_{\text{KL}}(p \| q)$.

4. Expliquez pourquoi la KL divergence n'est pas symétrique: $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$. Calculez les deux pour l'exemple de la question 2.

5. Reliez ceci au maximum de vraisemblance: si $p$ est la distribution empirique des données et $q_\theta$ est le modèle, montrez que minimiser la LVN revient à minimiser $D_{\text{KL}}(p \| q_\theta)$.
````

```{admonition} Solution Exercice 7
:class: dropdown

1. **Relation entropie croisée et KL:**

   $$
   H(p, q) = -\sum_i p_i \log q_i
   $$
   
   $$
   H(p) + D_{\text{KL}}(p \| q) = -\sum_i p_i \log p_i + \sum_i p_i \log \frac{p_i}{q_i}
   $$
   
   $$
   = -\sum_i p_i \log p_i + \sum_i p_i \log p_i - \sum_i p_i \log q_i
   $$
   
   $$
   = -\sum_i p_i \log q_i = H(p, q) \quad \checkmark
   $$

2. **Calcul des KL divergences:**

   $$
   D_{\text{KL}}(p \| q_1) = 0.7 \log\frac{0.7}{0.6} + 0.3 \log\frac{0.3}{0.4}
   $$
   
   $$
   = 0.7 \times 0.154 + 0.3 \times (-0.288) = 0.108 - 0.086 = 0.022
   $$
   
   $$
   D_{\text{KL}}(p \| q_2) = 0.7 \log\frac{0.7}{0.9} + 0.3 \log\frac{0.3}{0.1}
   $$
   
   $$
   = 0.7 \times (-0.251) + 0.3 \times 1.099 = -0.176 + 0.330 = 0.154
   $$
   
   $q_1$ est meilleur car $D_{\text{KL}}(p \| q_1) < D_{\text{KL}}(p \| q_2)$.

3. **Minimisation:**

   Puisque $H(p, q) = H(p) + D_{\text{KL}}(p \| q)$ et que $H(p)$ ne dépend pas de $q$:
   
   $$
   \arg\min_q H(p, q) = \arg\min_q D_{\text{KL}}(p \| q)
   $$

4. **Asymétrie de la KL:**

   $$
   D_{\text{KL}}(q_1 \| p) = 0.6 \log\frac{0.6}{0.7} + 0.4 \log\frac{0.4}{0.3}
   $$
   
   $$
   = 0.6 \times (-0.154) + 0.4 \times 0.288 = -0.092 + 0.115 = 0.023
   $$
   
   $D_{\text{KL}}(p \| q_1) = 0.022 \neq 0.023 = D_{\text{KL}}(q_1 \| p)$.
   
   L'asymétrie vient du fait que la KL pénalise différemment selon quelle distribution est au numérateur du log.

5. **Lien avec le maximum de vraisemblance:**

   La distribution empirique est $\hat{p}(y) = \frac{1}{N} \sum_{i=1}^N \delta(y - y_i)$.
   
   $$
   D_{\text{KL}}(\hat{p} \| q_\theta) = \sum_y \hat{p}(y) \log \frac{\hat{p}(y)}{q_\theta(y)} = H(\hat{p}) - \frac{1}{N} \sum_{i=1}^N \log q_\theta(y_i)
   $$
   
   Puisque $H(\hat{p})$ est constant:
   
   $$
   \arg\min_\theta D_{\text{KL}}(\hat{p} \| q_\theta) = \arg\max_\theta \sum_{i=1}^N \log q_\theta(y_i) = \arg\min_\theta \text{LVN}(\theta)
   $$
   
   Le maximum de vraisemblance cherche le modèle $q_\theta$ le plus proche (au sens KL) de la distribution empirique.
```
