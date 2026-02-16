---
marp: true
theme: mila
paginate: true
math: mathjax
---

<!-- _class: lead -->

# K-moyennes, GMM, mélange d'experts et ELBO
## Du partitionnement dur à l'inférence variationnelle

*Pierre-Luc Bacon*
IFT6390 – Fondements de l'apprentissage machine

---

## Plan de la présentation

1. **K-moyennes** : partitionnement dur, distorsion, diagramme de Voronoï
2. **Modèles de mélange gaussien** : du partitionnement dur au souple
3. **Algorithme EM** : responsabilités et moyennes pondérées
4. **ELBO et inférence variationnelle** : pourquoi EM converge
5. **Mélange d'experts** : EM pour la régression par morceaux

---

<!-- _class: lead -->

# K-moyennes
## Partitionnement dur par centroïdes

---

<!-- footer: "📖 Chapitre 6 : Modèles probabilistes génératifs" -->

## Pourquoi le partitionnement?

Les étiquettes de classe ne sont pas toujours disponibles. Un biologiste mesure des fleurs sans connaître l'espèce, un commerce enregistre des transactions sans profil client, un généticien séquence des tumeurs sans sous-type défini.

Le partitionnement (*clustering*) regroupe les observations en groupes homogènes, sans supervision.

![w:850](_static/clustering_motivation.png)

---

## L'algorithme des k-moyennes

On se donne $K$ centroïdes $\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_K$ et on alterne deux étapes :

**Assignation.** Chaque point va au centroïde le plus proche :
$$r_{nk} = \begin{cases} 1 & \text{si } k = \arg\min_j \|\mathbf{x}_n - \boldsymbol{\mu}_j\|^2 \\ 0 & \text{sinon} \end{cases}$$

**Mise à jour.** Chaque centroïde devient la moyenne de ses points :
$$\boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{n : r_{nk}=1} \mathbf{x}_n$$

Ces deux étapes se répètent jusqu'à stabilisation des assignations.

---

## K-moyennes en action

![w:800](_static/kmeans_convergence.gif)

À partir d'une initialisation arbitraire (étoiles), l'algorithme alterne assignation et mise à jour. En quelques itérations, les centroïdes se stabilisent.

---

## La fonction de distorsion

K-moyennes minimise la **distorsion**, qui mesure la distance totale entre chaque point et son centroïde :

$$J(\mathbf{r}, \boldsymbol{\mu}) = \sum_{n=1}^N \sum_{k=1}^K r_{nk} \|\mathbf{x}_n - \boldsymbol{\mu}_k\|^2$$

| Étape | Variable optimisée | Autre fixée |
|-------|--------------------|-------------|
| Assignation | $r_{nk} \in \{0, 1\}$ | $\boldsymbol{\mu}_k$ |
| Mise à jour | $\boldsymbol{\mu}_k$ | $r_{nk}$ |

Chaque étape diminue (ou maintient) $J$, mais l'optimisation conjointe est un problème combinatoire NP-difficile.

---

## Pourquoi pas la descente de gradient?

La distorsion $J$ dépend des assignations $r_{nk} \in \{0,1\}$ : des variables discrètes. On ne peut pas calculer $\frac{\partial J}{\partial r_{nk}}$ au sens usuel.

On pourrait relâcher les $r_{nk}$ en valeurs continues dans $[0,1]$ via un softmax, ce qui donnerait un objectif différentiable. Mais on n'a pas besoin de la descente de gradient : quand on passe au GMM, l'optimisation se fait par **descente de coordonnées** où chaque étape admet une **solution en forme fermée**.

| Étape | Variable | Solution |
|-------|----------|----------|
| E : fixer $\boldsymbol{\theta}$, optimiser $q$ | Responsabilités $r_{nk}$ | Formule de Bayes (fermée) |
| M : fixer $q$, optimiser $\boldsymbol{\theta}$ | $\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k$ | Moyennes pondérées (fermée) |

C'est l'algorithme EM : pas du gradient, mais de l'alternance avec des solutions exactes à chaque pas.

---

## Régions d'assignation : le diagramme de Voronoï

Puisque chaque point est assigné au centroïde le plus proche au sens euclidien, la séparation entre deux groupes adjacents est la **médiatrice** du segment reliant leurs centroïdes : une droite perpendiculaire passant par le milieu.

L'ensemble de ces médiatrices forme un **diagramme de Voronoï** dont les cellules sont des polygones convexes.

| Conséquence | Implication |
|-------------|-------------|
| Séparations linéaires | Les groupes sont séparés par des droites |
| Cellules convexes | Les groupes sont nécessairement « sphériques » |
| Indépendant de la forme réelle | K-moyennes ne peut pas capturer des groupes allongés |

---

## K-moyennes et ses limites

![w:850](_static/kmeans_centroids.png)

K-moyennes fonctionne bien quand les groupes sont compacts et de taille comparable. Il échoue quand les groupes ont des formes elliptiques, des tailles différentes, ou se chevauchent.

---

## Résumé : k-moyennes

| Élément | Détail |
|---------|--------|
| Objectif | Minimiser la distorsion $J$ |
| Assignation | Dure : chaque point à un seul groupe |
| Frontières | Médiatrices (Voronoï) |
| Convergence | Garantie (décroissance de $J$), mais maximum local |
| Limitation | Groupes sphériques uniquement |

Comment dépasser ces limitations? En remplaçant les assignations dures par des assignations souples et les sphères par des ellipsoïdes.

---

<!-- _class: lead -->

# Modèles de mélange gaussien
## Du partitionnement dur au partitionnement souple

---

## Du discret au probabiliste

K-moyennes assigne chaque point à un seul groupe. Mais un point situé entre deux centroïdes pourrait raisonnablement appartenir à l'un ou l'autre. Pourquoi ne pas quantifier cette incertitude?

L'idée : modéliser les données comme un **mélange** de $K$ distributions gaussiennes, chacune représentant un groupe. La probabilité qu'un point appartienne à un groupe remplace l'assignation binaire.

$$p(\mathbf{x} \mid \boldsymbol{\theta}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

---

## Formulation du GMM

![w:900](_static/gmm_formulation.png)

Chaque composant a ses propres paramètres : poids $\pi_k$, moyenne $\boldsymbol{\mu}_k$, covariance $\boldsymbol{\Sigma}_k$.

---

## Variable latente et processus génératif

On peut interpréter le GMM avec une variable latente $z \in \{1, \ldots, K\}$ :

$$p(z = k) = \pi_k, \qquad p(\mathbf{x} \mid z = k) = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

![w:900](_static/gmm_generative_process.png)

Pour générer un point : d'abord tirer un composant $k$ selon les poids $\pi_k$, puis tirer $\mathbf{x}$ selon la gaussienne correspondante.

---

## Responsabilités : partitionnement souple

La **responsabilité** du composant $k$ pour le point $\mathbf{x}_n$ :

$$r_{nk} = p(z_n = k \mid \mathbf{x}_n) = \frac{\pi_k \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^K \pi_j \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

C'est le théorème de Bayes appliqué à la variable latente $z_n$.

Les $r_{nk}$ vivent dans $[0,1]$ et somment à 1 : chaque point « appartient » partiellement à plusieurs composants.

---

## Partitionnement dur vs souple

![w:900](_static/gmm_clustering.png)

À gauche : chaque point est assigné au composant le plus probable. À droite : la couleur reflète les responsabilités. Les points près des frontières sont partagés entre composants.

---

## Le softmax des distances

Avec des covariances sphériques identiques $\boldsymbol{\Sigma}_k = \sigma^2 \mathbf{I}$ et des poids uniformes, les responsabilités se simplifient en un **softmax** sur les distances :

$$r_{nk} = \frac{\exp\!\big(-\frac{1}{2\sigma^2}\|\mathbf{x}_n - \boldsymbol{\mu}_k\|^2\big)}{\sum_{j} \exp\!\big(-\frac{1}{2\sigma^2}\|\mathbf{x}_n - \boldsymbol{\mu}_j\|^2\big)}$$

C'est la même transformation « scores → probabilités » que nous avions vue en régression logistique. Le paramètre $\sigma^2$ contrôle la « douceur » des assignations.

| $\sigma^2$ | Comportement |
|-------------|-------------|
| Grand | Assignations souples (incertitude) |
| Petit | Assignations presque dures |
| $\to 0$ | K-moyennes exact |

---

## K-moyennes comme cas limite du GMM

Quand $\sigma^2 \to 0$ dans le softmax, les responsabilités deviennent binaires : toute la masse va au centroïde le plus proche. On retrouve exactement l'assignation de k-moyennes.

Le GMM généralise k-moyennes sur trois axes :

| K-moyennes | GMM |
|-----------|-----|
| Assignation dure ($r_{nk} \in \{0,1\}$) | Assignation souple ($r_{nk} \in [0,1]$) |
| Groupes sphériques ($\sigma^2 \mathbf{I}$) | Groupes elliptiques ($\boldsymbol{\Sigma}_k$ quelconque) |
| Poids uniformes ($1/K$) | Poids variables ($\pi_k$) |

---

<!-- _class: lead -->

# L'algorithme EM
## Estimation des paramètres du GMM

---

## Le problème d'estimation

La log-vraisemblance du GMM est :

$$\log p(\mathbf{X} \mid \boldsymbol{\theta}) = \sum_{n=1}^N \log \left( \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right)$$

La somme à l'intérieur du $\log$ empêche de séparer les contributions de chaque composant. Pas de solution analytique.

| Difficulté | Détail |
|-----------|--------|
| Somme dans le log | Pas de formule fermée |
| $\boldsymbol{\pi} \in \Delta_K$ | Poids sur le simplexe |
| $\boldsymbol{\Sigma}_k \succ 0$ | Covariance définie positive |
| Non convexe | Plusieurs optima locaux |

---

## L'intuition : k-moyennes avec des responsabilités

Si nous connaissions les $z_n$ (les assignations), le problème serait simple : estimer chaque composant séparément, comme en k-moyennes. Mais les $z_n$ sont inconnus.

L'algorithme EM reprend la même stratégie d'alternance que k-moyennes, mais avec des assignations **souples** :

| K-moyennes | EM |
|-----------|-----|
| Assigner chaque point à un groupe | Calculer les responsabilités $r_{nk}$ |
| Moyenne des points assignés | Moyenne pondérée par les $r_{nk}$ |

---

## Étape E (Espérance)

Fixer les paramètres $\boldsymbol{\theta}^{(t)}$ et calculer les responsabilités :

$$r_{nk}^{(t)} = \frac{\pi_k^{(t)} \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k^{(t)}, \boldsymbol{\Sigma}_k^{(t)})}{\sum_{j} \pi_j^{(t)} \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j^{(t)}, \boldsymbol{\Sigma}_j^{(t)})}$$

C'est le théorème de Bayes : on calcule $p(z_n = k \mid \mathbf{x}_n, \boldsymbol{\theta}^{(t)})$.

---

## Étape M (Maximisation)

Fixer les responsabilités et mettre à jour les paramètres. Soit $N_k = \sum_n r_{nk}$ le nombre effectif de points dans le composant $k$.

**Poids :**
$$\pi_k^{(t+1)} = \frac{N_k^{(t)}}{N}$$

**Moyennes :**
$$\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{N_k^{(t)}} \sum_{n=1}^N r_{nk}^{(t)} \mathbf{x}_n$$

**Covariances :**
$$\boldsymbol{\Sigma}_k^{(t+1)} = \frac{1}{N_k^{(t)}} \sum_{n=1}^N r_{nk}^{(t)} (\mathbf{x}_n - \boldsymbol{\mu}_k^{(t+1)})(\mathbf{x}_n - \boldsymbol{\mu}_k^{(t+1)})^\top$$

---

## Convergence de l'algorithme EM

![w:800](_static/em_convergence.gif)

À partir d'une initialisation arbitraire, EM ajuste progressivement les composants (ellipses = contours à 1 et 2 écarts-types). La log-vraisemblance augmente à chaque itération.

---

## Résumé de l'algorithme EM

**Entrée** : données $\mathbf{X}$, nombre de composants $K$

1. Initialiser $\boldsymbol{\theta}^{(0)} = (\boldsymbol{\pi}, \boldsymbol{\mu}, \boldsymbol{\Sigma})$
2. Répéter jusqu'à convergence :

   **E** : $\quad r_{nk} \leftarrow \dfrac{\pi_k \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \pi_j \, \mathcal{N}(\mathbf{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$

   **M** : $\quad \pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k \leftarrow$ formules fermées

La log-vraisemblance augmente (ou reste stable) à chaque itération.

---

## Considérations pratiques

| Aspect | Problème | Solution |
|--------|----------|----------|
| **Initialisation** | Maximum local | Plusieurs essais, k-means++ |
| **Choix de $K$** | Hyperparamètre | BIC, AIC, validation croisée |
| **Singularités** | Covariance dégénérée | Régularisation $\boldsymbol{\Sigma}_k + \epsilon \mathbf{I}$ |
| **Convergence** | Lente parfois | Critère d'arrêt sur $\Delta\ell$ |

**Initialisation recommandée** : exécuter k-moyennes d'abord, puis utiliser les centroïdes comme moyennes initiales.

---

<!-- _class: lead -->

# ELBO et inférence variationnelle
## Pourquoi EM converge

---

## La question

Nous avons affirmé que la log-vraisemblance augmente à chaque itération d'EM. Mais pourquoi? Y a-t-il un objectif que chaque étape améliore?

La réponse repose sur une **borne inférieure** de la log-vraisemblance, appelée ELBO (*Evidence Lower Bound*). Nous allons la construire, puis montrer que les étapes E et M la maximisent par alternance.

---

## Point de départ : la règle de Bayes

Pour tout $\mathbf{Z}$ (les assignations latentes), la règle de Bayes donne :

$$p(\mathbf{X} \mid \boldsymbol{\theta}) = \frac{p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})}{p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})}$$

En prenant le logarithme :

$$\log p(\mathbf{X} \mid \boldsymbol{\theta}) = \log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta}) - \log p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})$$

Le membre de gauche ne dépend pas de $\mathbf{Z}$ : c'est une constante par rapport aux latentes.

---

## Introduire une distribution auxiliaire $q$

Prenons une distribution $q(\mathbf{Z})$ quelconque sur les latentes. En moyennant l'égalité précédente sous $q$ (le membre de gauche ne change pas) :

$$\log p(\mathbf{X} \mid \boldsymbol{\theta}) = \mathbb{E}_{q}\!\big[ \log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta}) \big] - \mathbb{E}_{q}\!\big[ \log p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}) \big]$$

Ajoutons et retranchons $\mathbb{E}_q[\log q(\mathbf{Z})]$, puis regroupons :

$$\log p(\mathbf{X} \mid \boldsymbol{\theta}) = \underbrace{\mathbb{E}_{q}\!\big[\log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta}) - \log q(\mathbf{Z})\big]}_{\text{ELBO}(q, \boldsymbol{\theta})} + \underbrace{D_{\text{KL}}(q \| p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}))}_{\geq\, 0}$$

---

## La borne inférieure de l'évidence (ELBO)

Puisque $D_{\text{KL}} \geq 0$ toujours :

$$\boxed{\log p(\mathbf{X} \mid \boldsymbol{\theta}) \;\geq\; \text{ELBO}(q, \boldsymbol{\theta})}$$

| Propriété | Détail |
|-----------|--------|
| C'est une borne | L'ELBO ne dépasse jamais la log-vraisemblance |
| Borne serrée | Égalité quand $q(\mathbf{Z}) = p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})$ |
| Dépend de $q$ et $\boldsymbol{\theta}$ | Deux « boutons » à tourner |

L'« évidence » désigne $p(\mathbf{X} \mid \boldsymbol{\theta})$, la vraisemblance marginale des données.

---

## EM = maximisation alternée de l'ELBO

$$\log p(\mathbf{X} \mid \boldsymbol{\theta}) = \text{ELBO}(q, \boldsymbol{\theta}) + D_{\text{KL}}(q \| p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta}))$$

**Étape E** : fixer $\boldsymbol{\theta}$, maximiser l'ELBO en $q$.

Puisque la log-vraisemblance ne dépend pas de $q$, augmenter l'ELBO = diminuer la KL. Le maximum est atteint quand $D_{\text{KL}} = 0$, soit :

$$q^*(\mathbf{Z}) = p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})$$

Pour le GMM, c'est exactement le calcul des responsabilités $r_{nk}$.

---

## EM = maximisation alternée de l'ELBO (suite)

**Étape M** : fixer $q$ (= l'a posteriori), maximiser l'ELBO en $\boldsymbol{\theta}$.

$$\text{ELBO}(q, \boldsymbol{\theta}) = \mathbb{E}_q[\log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})] - \underbrace{\mathbb{E}_q[\log q(\mathbf{Z})]}_{\text{ne dépend pas de } \boldsymbol{\theta}}$$

Maximiser l'ELBO en $\boldsymbol{\theta}$ revient à maximiser $\mathbb{E}_q[\log p(\mathbf{X}, \mathbf{Z} \mid \boldsymbol{\theta})]$, la log-vraisemblance conjointe pondérée par les responsabilités. Ce sont les formules de l'étape M.

À chaque demi-pas, l'ELBO augmente → la log-vraisemblance aussi → convergence.

---

## Schéma de la convergence

```
Itération t                          Itération t+1
                                     
     log p(X|θ)  ─────────────────────── log p(X|θ')
         │                                   │
    KL > 0                              KL = 0
         │                                   │
    ELBO(q_old, θ)                     ELBO(q_new, θ)
                                             │
                    Étape E                  │
                  (q → post.)          ELBO(q_new, θ)
                                             │
                    Étape M                  │
                  (θ → θ')             ELBO(q_new, θ')
```

L'étape E serre la borne (KL → 0). L'étape M pousse la borne vers le haut.

---

## Inférence variationnelle : au-delà d'EM

Le raisonnement « maximiser l'ELBO par rapport à $q$ et $\boldsymbol{\theta}$ » s'appelle l'**inférence variationnelle**. EM en est le cas le plus favorable :

| | EM | Inférence variationnelle générale |
|---|---|---|
| A posteriori $p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})$ | Calculable (formule fermée) | Intractable |
| Famille pour $q$ | Toutes les distributions | Famille restreinte $\mathcal{Q}$ |
| Optimisation de $q$ | $q = p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})$ exact | $q \approx p(\mathbf{Z} \mid \mathbf{X}, \boldsymbol{\theta})$ |
| Borne | Serrée à chaque E-step | Gap résiduel |

Quand l'a posteriori n'est pas calculable (VAE, modèles de thèmes, etc.), on restreint $q$ à une famille paramétrique et on optimise par gradient.

---

<!-- _class: lead -->

# Mélange d'experts
## EM pour la régression par morceaux

---

## Du partitionnement à la prédiction

Un GMM partitionne l'espace des observations en groupes, mais il ne fait pas de prédiction.

Considérons des données où la relation entre $x$ et $y$ change selon la région : un système physique avec plusieurs régimes, ou un marché dont la dynamique varie selon la conjoncture.

Un seul modèle linéaire ne peut pas capturer ces régimes. Mais si l'on dispose de **plusieurs modèles linéaires** (un par régime) et d'un **mécanisme de routage** pour aiguiller chaque observation vers le bon modèle, on obtient une prédiction flexible à partir de composants simples.

---

## Le problème

![center w:700](_static/moe_convergence.gif)

Un modèle linéaire unique (trait pointillé) ne capture pas les changements de pente. Le mélange d'experts (lignes colorées) spécialise chaque expert dans une région.

---

## Le modèle de mélange d'experts

$$p(y \mid \mathbf{x}) = \sum_{k=1}^K g_k(\mathbf{x}) \; p_k(y \mid \mathbf{x})$$

| Composant | Rôle | Forme |
|-----------|------|-------|
| $g_k(\mathbf{x})$ | **Réseau de routage** (gating) | $\text{softmax}(\mathbf{V}\mathbf{x})_k$ |
| $p_k(y \mid \mathbf{x})$ | **Expert** $k$ | $\mathcal{N}(y \mid \mathbf{w}_k^\top \mathbf{x}, \sigma_k^2)$ |

Les poids du mélange $g_k(\mathbf{x})$ dépendent de l'entrée : chaque expert se spécialise dans une région de l'espace.

---

## Comparaison avec le GMM

| | GMM | Mélange d'experts |
|---|---|---|
| Poids $\pi_k$ | Fixes | Dépendent de $\mathbf{x}$ : $g_k(\mathbf{x})$ |
| Composants | $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ | $\mathcal{N}(y \mid \mathbf{w}_k^\top \mathbf{x}, \sigma_k^2)$ |
| Cadre | Non supervisé | Supervisé |
| Variable latente | $z$ = composant | $z$ = expert responsable |

La structure est la même : un mélange avec une variable latente discrète. L'algorithme EM s'applique de la même manière.

---

## EM pour le mélange d'experts

**Étape E.** Calculer les responsabilités (quel expert « explique » chaque point?) :

$$r_{nk} = \frac{g_k(\mathbf{x}_n) \; p_k(y_n \mid \mathbf{x}_n)}{\sum_j g_j(\mathbf{x}_n) \; p_j(y_n \mid \mathbf{x}_n)}$$

**Étape M.** Mettre à jour chaque expert par régression pondérée :

$$\mathbf{w}_k \leftarrow (\mathbf{X}^\top \mathbf{R}_k \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{R}_k \mathbf{y}$$

où $\mathbf{R}_k = \text{diag}(r_{1k}, \ldots, r_{Nk})$.

Le réseau de routage est mis à jour par montée de gradient sur la log-vraisemblance pondérée.

---

## Pourquoi le routage dépendant de l'entrée?

Avec des poids **fixes** (comme dans un GMM), les experts peinent à se spécialiser : chacun tente de couvrir l'ensemble des données.

Avec un routage **dépendant de l'entrée**, chaque expert se concentre sur la région où il excelle, et la prédiction suit la structure par morceaux des données.

| Poids fixes | Routage adaptatif |
|-----------|----------|
| Experts généralistes | Experts spécialisés |
| Prédiction = compromis | Prédiction = expert local |
| Comme un GMM sur $(x, y)$ | Comme un classifieur + régresseurs |

---

## Convergence du mélange d'experts

![w:800](_static/moe_convergence.gif)

L'animation montre l'alternance E/M : les experts (droites colorées) se spécialisent progressivement, tandis que le réseau de routage apprend à aiguiller les observations.

---

## MoE en apprentissage profond

Les notions de routage et de spécialisation réapparaissent dans les grands modèles de langage (LLM) et l'apprentissage profond.

| Concept du cours | Équivalent en DL |
|----------|----------|
| Expert (modèle linéaire) | Sous-réseau spécialisé |
| Réseau de routage | Couche de gating |
| $K$ experts actifs | Routage « sparse » (quelques experts actifs) |

Des modèles comme Mixtral utilisent des couches MoE pour augmenter la capacité du réseau sans augmenter proportionnellement le coût de calcul : seuls quelques experts sont activés pour chaque entrée.

Nous reviendrons sur ces architectures quand nous aborderons les réseaux de neurones.

---

<!-- _class: lead -->

# Résumé

---

<!-- footer: "" -->

## Les modèles vus aujourd'hui

| Modèle | Assignation | Forme des groupes | Cadre |
|--------|------------|-------------------|-------|
| K-moyennes | Dure | Sphérique | Non supervisé |
| GMM | Souple | Elliptique | Non supervisé |
| Mélange d'experts | Souple, dépend de $\mathbf{x}$ | N/A (régression) | Supervisé |

L'algorithme EM estime les paramètres dans les trois cas par alternance E/M.

La convergence est garantie par la monotonie de l'ELBO : chaque E-step serre la borne, chaque M-step la pousse vers le haut.

---

## Le fil conducteur

```
K-moyennes                          GMM
(assignation dure,           (assignation souple,
 groupes sphériques)          groupes elliptiques)
      │                              │
      │     σ² → 0 : on retrouve     │
      │◄─────── k-moyennes ──────────┘
      │                              │
      └──────────────┬───────────────┘
                     │
              Algorithme EM
           (alternance E/M,
            monotonie ELBO)
                     │
              Mélange d'experts
           (EM pour la régression,
            routage adaptatif)
```

---

<!-- _class: lead -->

# Questions?

**Exercices recommandés** :
- Exercice 4 (ch6) : Responsabilités GMM
- Exercice 5 (ch6) : Étape M de l'algorithme EM
- Exercice 6 (ch6) : K-moyennes comme cas limite
- Exercice 7 (ch6) : ELBO et inférence variationnelle
