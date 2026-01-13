# Annexe: Révision de probabilités

Cette annexe révise les concepts de probabilité utilisés tout au long du livre.

## Variables aléatoires

### Définitions

Une **variable aléatoire** est une fonction qui associe un résultat d'une expérience aléatoire à un nombre réel.

- **Variable discrète**: prend un nombre fini ou dénombrable de valeurs
- **Variable continue**: prend des valeurs dans un intervalle de $\mathbb{R}$

### Fonction de masse de probabilité (PMF)

Pour une variable discrète $X$, la PMF est:
$$
p(x) = P(X = x)
$$

Propriétés:
- $p(x) \geq 0$ pour tout $x$
- $\sum_x p(x) = 1$

### Fonction de densité de probabilité (PDF)

Pour une variable continue $X$, la PDF $f(x)$ satisfait:
$$
P(a \leq X \leq b) = \int_a^b f(x) \, dx
$$

Propriétés:
- $f(x) \geq 0$ pour tout $x$
- $\int_{-\infty}^{\infty} f(x) \, dx = 1$

**Note**: $f(x)$ peut être $> 1$; c'est une densité, pas une probabilité.

## Distributions courantes

### Distribution de Bernoulli

Variable binaire $X \in \{0, 1\}$:
$$
p(x \mid \theta) = \theta^x (1-\theta)^{1-x}
$$

- Paramètre: $\theta \in [0, 1]$ (probabilité de succès)
- Espérance: $\mathbb{E}[X] = \theta$
- Variance: $\text{Var}(X) = \theta(1-\theta)$

### Distribution catégorique (multinoulli)

Variable à $K$ catégories $X \in \{1, \ldots, K\}$:
$$
p(x = k \mid \boldsymbol{\theta}) = \theta_k
$$

où $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_K)$ avec $\sum_k \theta_k = 1$.

Avec le codage one-hot $\boldsymbol{x} \in \{0,1\}^K$:
$$
p(\boldsymbol{x} \mid \boldsymbol{\theta}) = \prod_{k=1}^K \theta_k^{x_k}
$$

### Distribution binomiale

Nombre de succès en $n$ essais de Bernoulli:
$$
p(k \mid n, \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k}
$$

- Espérance: $\mathbb{E}[X] = n\theta$
- Variance: $\text{Var}(X) = n\theta(1-\theta)$

### Distribution de Poisson

Nombre d'événements dans un intervalle:
$$
p(k \mid \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}
$$

- Espérance: $\mathbb{E}[X] = \lambda$
- Variance: $\text{Var}(X) = \lambda$

### Distribution gaussienne (normale)

**Univariée**:
$$
\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

- Espérance: $\mathbb{E}[X] = \mu$
- Variance: $\text{Var}(X) = \sigma^2$

**Multivariée**:
$$
\mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\right)
$$

où:
- $\boldsymbol{\mu} \in \mathbb{R}^D$ est le vecteur moyenne
- $\boldsymbol{\Sigma} \in \mathbb{R}^{D \times D}$ est la matrice de covariance (symétrique, définie positive)

### Distribution uniforme

**Continue** sur $[a, b]$:
$$
p(x) = \frac{1}{b-a} \mathbb{I}(a \leq x \leq b)
$$

### Distribution exponentielle

$$
p(x \mid \lambda) = \lambda e^{-\lambda x} \mathbb{I}(x \geq 0)
$$

### Distribution Beta

Pour $x \in [0, 1]$:
$$
\text{Beta}(x \mid \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}
$$

où $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$.

Utile comme a priori pour les paramètres de Bernoulli (conjugué).

### Distribution de Dirichlet

Généralisation de Beta à $K$ dimensions:
$$
\text{Dir}(\boldsymbol{\theta} \mid \boldsymbol{\alpha}) = \frac{1}{B(\boldsymbol{\alpha})} \prod_{k=1}^K \theta_k^{\alpha_k - 1}
$$

Utile comme a priori pour les paramètres catégoriques (conjugué).

## Espérance et variance

### Espérance

L'**espérance** (ou moyenne) d'une variable aléatoire:

**Discrète**: $\mathbb{E}[X] = \sum_x x \, p(x)$

**Continue**: $\mathbb{E}[X] = \int x \, f(x) \, dx$

Propriétés:
- Linéarité: $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$
- $\mathbb{E}[g(X)] = \sum_x g(x) p(x)$ (ou intégrale pour le continu)

### Variance

La **variance** mesure la dispersion autour de la moyenne:
$$
\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2
$$

L'**écart-type**: $\sigma = \sqrt{\text{Var}(X)}$

Propriétés:
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$
- Si $X \perp Y$: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

### Covariance

$$
\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]
$$

**Corrélation** (covariance normalisée):
$$
\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} \in [-1, 1]
$$

**Matrice de covariance** pour un vecteur aléatoire $\boldsymbol{X}$:
$$
\boldsymbol{\Sigma} = \mathbb{E}[(\boldsymbol{X} - \boldsymbol{\mu})(\boldsymbol{X} - \boldsymbol{\mu})^\top]
$$

## Probabilités jointes et marginales

### Règle de la somme (marginalisation)

$$
p(X) = \sum_Y p(X, Y) \quad \text{ou} \quad p(x) = \int p(x, y) \, dy
$$

### Règle du produit

$$
p(X, Y) = p(X \mid Y) p(Y) = p(Y \mid X) p(X)
$$

### Règle de la chaîne

$$
p(X_1, \ldots, X_n) = \prod_{i=1}^n p(X_i \mid X_1, \ldots, X_{i-1})
$$

## Théorème de Bayes

### Formulation

$$
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) p(\theta)}{p(\mathcal{D})}
$$

où:
- $p(\theta)$: **a priori** (croyance avant les données)
- $p(\mathcal{D} \mid \theta)$: **vraisemblance** (probabilité des données étant donné $\theta$)
- $p(\theta \mid \mathcal{D})$: **a posteriori** (croyance après avoir vu les données)
- $p(\mathcal{D})$: **évidence** ou constante de normalisation

### Constante de normalisation

$$
p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) p(\theta) \, d\theta
$$

Cette intégrale est souvent difficile à calculer (inférence approximative).

### Forme proportionnelle

Souvent, on écrit:
$$
p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) p(\theta)
$$

(a posteriori proportionnel à vraisemblance × a priori)

## Indépendance

### Indépendance marginale

$X \perp Y$ si:
$$
p(X, Y) = p(X) p(Y)
$$

Équivalent à: $p(X \mid Y) = p(X)$

### Indépendance conditionnelle

$X \perp Y \mid Z$ si:
$$
p(X, Y \mid Z) = p(X \mid Z) p(Y \mid Z)
$$

**Attention**: $X \perp Y$ n'implique pas $X \perp Y \mid Z$ (et vice versa).

## Divergence de Kullback-Leibler

### Définition

La **divergence KL** mesure la "distance" entre deux distributions:
$$
D_{\text{KL}}(p \| q) = \mathbb{E}_p\left[\log \frac{p(x)}{q(x)}\right] = \sum_x p(x) \log \frac{p(x)}{q(x)}
$$

### Propriétés

- $D_{\text{KL}}(p \| q) \geq 0$ (inégalité de Gibbs)
- $D_{\text{KL}}(p \| q) = 0$ ssi $p = q$
- **Non symétrique**: $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$ en général
- Ce n'est pas une distance (ne satisfait pas l'inégalité triangulaire)

### Pour les gaussiennes

Pour $p = \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1)$ et $q = \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)$:
$$
D_{\text{KL}}(p \| q) = \frac{1}{2}\left[\log\frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|} - D + \text{tr}(\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_1) + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^\top \boldsymbol{\Sigma}_2^{-1}(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)\right]
$$

## Entropie

### Entropie de Shannon

L'**entropie** mesure l'incertitude d'une distribution:
$$
\mathbb{H}(X) = -\mathbb{E}[\log p(X)] = -\sum_x p(x) \log p(x)
$$

- Maximale pour une distribution uniforme
- Minimale (= 0) pour une distribution déterministe

### Entropie croisée

$$
\mathbb{H}(p, q) = -\mathbb{E}_p[\log q(X)] = -\sum_x p(x) \log q(x)
$$

Relation avec KL:
$$
\mathbb{H}(p, q) = \mathbb{H}(p) + D_{\text{KL}}(p \| q)
$$

## Inégalités utiles

### Inégalité de Jensen

Pour une fonction convexe $\phi$:
$$
\phi(\mathbb{E}[X]) \leq \mathbb{E}[\phi(X)]
$$

Pour une fonction concave: inégalité inversée.

### Inégalité de Markov

Pour $X \geq 0$ et $a > 0$:
$$
P(X \geq a) \leq \frac{\mathbb{E}[X]}{a}
$$

### Inégalité de Chebyshev

$$
P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}
$$

### Inégalité de Hoeffding

Pour des variables $X_1, \ldots, X_n$ indépendantes bornées dans $[a_i, b_i]$:
$$
P\left(\left|\frac{1}{n}\sum_{i=1}^n X_i - \mathbb{E}\left[\frac{1}{n}\sum_{i=1}^n X_i\right]\right| \geq t\right) \leq 2\exp\left(-\frac{2n^2t^2}{\sum_{i=1}^n(b_i - a_i)^2}\right)
$$
