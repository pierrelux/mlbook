"""Generate figures for ch7 neural networks slides."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.special import erf
import os

OUT = os.path.join(os.path.dirname(__file__), "_static")
os.makedirs(OUT, exist_ok=True)

MILA_PURPLE = "#662E7D"
MILA_BG = "#f4f5f1"
plt.rcParams.update({
    "figure.facecolor": MILA_BG,
    "axes.facecolor": MILA_BG,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── 1. Perceptron vs logistic regression ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=MILA_BG)

z = np.linspace(-5, 5, 500)
sigmoid = 1 / (1 + np.exp(-z))

ax = axes[0]
ax.set_facecolor(MILA_BG)
ax.plot(z, sigmoid, '#1f77b4', lw=2.5, label=r'Sigmoïde $\sigma(z)$')
ax.plot(z[z < 0],  np.zeros(np.sum(z < 0)),  '#d62728', lw=2.5)
ax.plot(z[z >= 0], np.ones(np.sum(z >= 0)),  '#d62728', lw=2.5,
        label=r'Échelon $\mathbf{1}[z \geq 0]$')
ax.scatter([0], [1], color='#d62728', s=55, zorder=5)
ax.scatter([0], [0], color='white', edgecolors='#d62728', linewidths=2, s=55, zorder=5)
ax.axvline(0, color='gray', lw=1, linestyle=':', alpha=0.6)
ax.set_xlabel(r'Pré-activation $z = \boldsymbol{\theta}^\top \mathbf{x}$')
ax.set_ylabel('Sortie')
ax.set_title('Activations comparées', color=MILA_PURPLE, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 5); ax.set_ylim(-0.1, 1.2)

rng = np.random.default_rng(0)
n = 40
X_pos = rng.multivariate_normal([ 1.5,  0.8], [[0.4, 0], [0, 0.4]], n)
X_neg = rng.multivariate_normal([-1.5, -0.8], [[0.4, 0], [0, 0.4]], n)

ax = axes[1]
ax.set_facecolor(MILA_BG)
xx, yy = np.meshgrid(np.linspace(-3.5, 3.5, 200), np.linspace(-2.5, 2.5, 200))
Z = xx + yy
ax.contourf(xx, yy, Z, levels=[-100, 0, 100], colors=['#fdd8d8', '#d8e8fd'], alpha=0.45)
ax.contour(xx, yy, Z, levels=[0], colors='k', linewidths=2)
ax.scatter(X_pos[:, 0], X_pos[:, 1], color='#1f77b4', marker='o', s=35,
           alpha=0.85, label='Classe 1', edgecolors='white', linewidths=0.5)
ax.scatter(X_neg[:, 0], X_neg[:, 1], color='#d62728', marker='s', s=35,
           alpha=0.85, label='Classe 0', edgecolors='white', linewidths=0.5)
ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
ax.set_title(r'Même frontière: $\boldsymbol{\theta}^\top \mathbf{x} = 0$', color=MILA_PURPLE, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)

plt.suptitle('Régression logistique et perceptron', fontsize=11, color=MILA_PURPLE, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/ch7_perceptron_vs_logistic.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ perceptron_vs_logistic")

# ── 2. XOR problem ────────────────────────────────────────────────────────────
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0, 1, 1, 0])
markers = ['s', 'o']
colors  = ['#4878CF', '#D65F5F']
labels  = ['Classe 0', 'Classe 1']

fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=MILA_BG)

ax = axes[0]
ax.set_facecolor(MILA_BG)
for cls in [0, 1]:
    mask = y == cls
    ax.scatter(X[mask, 0], X[mask, 1], marker=markers[cls], color=colors[cls],
               s=180, zorder=5, label=labels[cls], edgecolors='k', linewidths=1.2)
xline = np.linspace(-0.3, 1.3, 100)
ax.plot(xline, -xline + 1, 'k--', lw=1.5, alpha=0.5, label='Meilleure droite')
ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.4, 1.4)
ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
ax.set_title('Espace original: XOR non séparable', color=MILA_PURPLE, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
ax.text(0.5, 1.1, 'Aucune droite ne sépare\nles deux classes',
        ha='center', fontsize=8.5, color='#555555',
        bbox=dict(boxstyle='round,pad=0.3', fc='#fffbe6', ec='#ccbb00', alpha=0.9))

W1 = np.array([[1., 1.], [1., 1.]])
b1 = np.array([-0.5, -1.5])
H = np.maximum(0, X @ W1.T + b1)

ax = axes[1]
ax.set_facecolor(MILA_BG)
for cls in [0, 1]:
    mask = y == cls
    ax.scatter(H[mask, 0], H[mask, 1], marker=markers[cls], color=colors[cls],
               s=180, zorder=5, label=labels[cls], edgecolors='k', linewidths=1.2)
hline = np.linspace(-0.1, 1.8, 100)
ax.plot(hline, (hline - 0.4) / 3, 'g-', lw=2, label='Séparateur linéaire')
ax.set_xlim(-0.15, 1.8); ax.set_ylim(-0.2, 0.8)
ax.set_xlabel(r'$h_1 = \mathrm{ReLU}(x_1+x_2-0{,}5)$')
ax.set_ylabel(r'$h_2 = \mathrm{ReLU}(x_1+x_2-1{,}5)$')
ax.set_title('Espace transformé: XOR devient séparable', color=MILA_PURPLE, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

plt.suptitle('Le problème XOR: la couche cachée transforme l\'espace', fontsize=11, color=MILA_PURPLE, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/ch7_xor_problem.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ xor_problem")

# ── 3. MLP trained on XOR ─────────────────────────────────────────────────────
np.random.seed(7)

def relu(x): return np.maximum(0, x)
def sigmoid_fn(x): return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([[0],[1],[1],[0]], dtype=float)

H = 8
W1 = np.random.randn(2, H) * 0.5; b1 = np.zeros(H)
W2 = np.random.randn(H, 1) * 0.5; b2 = np.zeros(1)

lr = 0.5
for _ in range(8000):
    a1 = X_xor @ W1 + b1
    z1 = relu(a1)
    a2 = z1 @ W2 + b2
    p  = sigmoid_fn(a2)
    dp  = p - y_xor
    dW2 = z1.T @ dp / 4; db2 = dp.mean(axis=0)
    dz1 = dp @ W2.T
    da1 = dz1 * (a1 > 0).astype(float)
    dW1 = X_xor.T @ da1 / 4; db1 = da1.mean(axis=0)
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 300), np.linspace(-0.3, 1.3, 300))
Xg = np.column_stack([xx.ravel(), yy.ravel()])
zz = sigmoid_fn(relu(Xg @ W1 + b1) @ W2 + b2).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(5, 4.5), facecolor=MILA_BG)
ax.set_facecolor(MILA_BG)
cf = ax.contourf(xx, yy, zz, levels=50, cmap='RdBu_r', alpha=0.7, vmin=0, vmax=1)
ax.contour(xx, yy, zz, levels=[0.5], colors='k', linewidths=2)
plt.colorbar(cf, ax=ax, label=r'$p(y=1 \mid \mathbf{x})$')

markers2 = ['s', 'o']; colors_pt = ['#4878CF', '#D65F5F']
for cls in [0, 1]:
    mask = y_xor.ravel() == cls
    ax.scatter(X_xor[mask, 0], X_xor[mask, 1], marker=markers2[cls], color=colors_pt[cls],
               s=200, zorder=5, edgecolors='k', linewidths=1.5, label=f'Classe {cls}')

ax.set_xlim(-0.3, 1.3); ax.set_ylim(-0.3, 1.3)
ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')
ax.set_title('Frontière de décision d\'un MLP sur XOR', color=MILA_PURPLE, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUT}/ch7_xor_mlp.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ xor_mlp")

# ── 4. Activation functions ───────────────────────────────────────────────────
a = np.linspace(-4, 4, 400)
gelu = lambda x: x * 0.5 * (1 + erf(x / np.sqrt(2)))
d_sigmoid = lambda x: (1/(1+np.exp(-x))) * (1 - 1/(1+np.exp(-x)))
d_tanh    = lambda x: 1 - np.tanh(x)**2
d_relu    = lambda x: (x > 0).astype(float)
eps = 1e-5
d_gelu = lambda x: (gelu(x + eps) - gelu(x - eps)) / (2 * eps)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=MILA_BG)

ax = axes[0]
ax.set_facecolor(MILA_BG)
ax.plot(a, 1/(1+np.exp(-a)), 'C0', linewidth=2, label='Sigmoïde')
ax.plot(a, np.tanh(a),       'C1', linewidth=2, label='Tanh')
ax.plot(a, np.maximum(0, a), 'C2', linewidth=2, label='ReLU')
ax.plot(a, gelu(a),          'C3', linewidth=2, label='GELU', linestyle='--')
ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
ax.axvline(0, color='k', linewidth=0.5, linestyle=':')
ax.set_xlabel('$a$ (pré-activation)'); ax.set_ylabel('$\\varphi(a)$')
ax.set_title("Fonctions d'activation", color=MILA_PURPLE, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(-4, 4)

ax = axes[1]
ax.set_facecolor(MILA_BG)
ax.plot(a, d_sigmoid(a), 'C0', linewidth=2, label="$\\sigma'(a)$")
ax.plot(a, d_tanh(a),    'C1', linewidth=2, label="$\\tanh'(a)$")
ax.plot(a, d_relu(a),    'C2', linewidth=2, label="ReLU$'(a)$")
ax.plot(a, d_gelu(a),    'C3', linewidth=2, label="GELU$'(a)$", linestyle='--')
ax.axhline(0, color='k', linewidth=0.5, linestyle=':')
ax.axvline(0, color='k', linewidth=0.5, linestyle=':')
ax.annotate("$\\sigma'(0) = 0{,}25$", xy=(0, 0.25), xytext=(1.3, 0.42),
            arrowprops=dict(arrowstyle='->', color='C0', lw=1.5), fontsize=9, color='C0')
ax.set_xlabel('$a$ (pré-activation)'); ax.set_ylabel("$\\varphi'(a)$")
ax.set_title('Dérivées des fonctions d\'activation', color=MILA_PURPLE, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(-4, 4); ax.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig(f"{OUT}/ch7_activation_functions.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ activation_functions")

# ── 5. SGD vs Momentum ────────────────────────────────────────────────────────
def grad_f(t): return np.array([2*t[0], 20*t[1]])

theta0 = np.array([-0.9, 0.85])
eta = 0.08; beta = 0.5; n_steps = 40

traj_sgd = [theta0.copy()]
t = theta0.copy()
for _ in range(n_steps):
    t = t - eta * grad_f(t)
    traj_sgd.append(t.copy())
traj_sgd = np.array(traj_sgd)

traj_mom = [theta0.copy()]
t = theta0.copy(); m = np.zeros(2)
for _ in range(n_steps):
    m = beta * m + grad_f(t)
    t = t - eta * m
    traj_mom.append(t.copy())
traj_mom = np.array(traj_mom)

t1 = np.linspace(-1.1, 1.1, 300)
t2 = np.linspace(-1.0, 1.0, 300)
T1, T2 = np.meshgrid(t1, t2)
Z = T1**2 + 10*T2**2

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, facecolor=MILA_BG)
plt.suptitle(r'SGD vs momentum sur $f(\theta) = \theta_1^2 + 10\theta_2^2$', fontsize=11, color=MILA_PURPLE, fontweight='bold')

trajs  = [traj_sgd, traj_mom]
titles = ['SGD (zigzags)', r'SGD + Momentum ($\beta=0{,}5$)']
plot_colors = ['#1f77b4', '#d62728']

for ax, traj, title, color in zip(axes, trajs, titles, plot_colors):
    ax.set_facecolor(MILA_BG)
    ax.contourf(T1, T2, Z, levels=20, cmap='Greys', alpha=0.5)
    ax.contour( T1, T2, Z, levels=20, colors='gray', linewidths=0.4, alpha=0.6)
    n = len(traj) - 1
    for i in range(n):
        alpha_val = 0.25 + 0.75 * (i / n)
        ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], '-', color=color, lw=2, alpha=alpha_val)
    ax.plot(*traj[0], 'ko', ms=8, zorder=5, label='Départ')
    ax.plot(*traj[-1], 'o', ms=7, color=color, zorder=6, label='Arrivée')
    ax.plot(0, 0, 'r*', ms=12, zorder=5, label='Minimum')
    ax.set_xlabel(r'$\theta_1$'); ax.set_title(title, fontsize=10, color=MILA_PURPLE, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.0, 1.0); ax.grid(True, alpha=0.2)

axes[0].set_ylabel(r'$\theta_2$')
plt.tight_layout()
plt.savefig(f"{OUT}/ch7_sgd_vs_momentum.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ sgd_vs_momentum")

# ── 6. Vanishing gradient ──────────────────────────────────────────────────────
np.random.seed(42)
n_layers = 20; n_trials = 50; d = 50

def simulate_gradient_norm(activation_deriv, n_layers, n_trials, d):
    norms = []
    for _ in range(n_trials):
        g = np.ones(d) / np.sqrt(d)
        for _ in range(n_layers):
            W = np.random.randn(d, d) / np.sqrt(d)
            a_val = np.random.randn(d)
            dphi = activation_deriv(a_val)
            g = (dphi * g) @ W
        norms.append(np.linalg.norm(g))
    return norms

d_sigmoid2 = lambda a: np.exp(-a) / (1 + np.exp(-a))**2
d_relu2    = lambda a: (a > 0).astype(float)

layers = np.arange(1, n_layers + 1)
norms_sigmoid = np.array([np.mean(simulate_gradient_norm(d_sigmoid2, l, n_trials, d)) for l in layers])
norms_relu    = np.array([np.mean(simulate_gradient_norm(d_relu2,    l, n_trials, d)) for l in layers])

fig, ax = plt.subplots(figsize=(8, 4), facecolor=MILA_BG)
ax.set_facecolor(MILA_BG)
ax.semilogy(layers, norms_sigmoid, 'C0-o', markersize=4, linewidth=2, label='Sigmoïde')
ax.semilogy(layers, norms_relu,    'C2-s', markersize=4, linewidth=2, label='ReLU')
ax.axhspan(0, 1e-8, alpha=0.1, color='C0', label='Zone de disparition')
ax.set_xlabel('Profondeur (nombre de couches)')
ax.set_ylabel('Norme du gradient $\\|\\nabla_{W_1}\\mathcal{L}\\|$')
ax.set_title('Gradient qui disparaît: sigmoïde vs ReLU', color=MILA_PURPLE, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, which='both'); ax.set_xlim(1, n_layers)
plt.tight_layout()
plt.savefig(f"{OUT}/ch7_vanishing_gradient.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ vanishing_gradient")

# ── 7. Weight initialization ──────────────────────────────────────────────────
np.random.seed(1)
d = 100; n_samples = 500; n_layers_init = 5

def propagate(x, n_layers, var, activation=np.tanh):
    activations = []
    h = x.copy()
    for _ in range(n_layers):
        W = np.random.randn(d, d) * np.sqrt(var)
        h = activation(h @ W)
        activations.append(h.ravel())
    return activations

x0 = np.random.randn(n_samples, d)

configs = [
    ('Trop petite\n($\\sigma^2 = 0{,}01$)', 0.01,  'C3'),
    ('Glorot\n($\\sigma^2 = 2/(n+m)$)', 2/d,    'C2'),
    ('Trop grande\n($\\sigma^2 = 4$)',    4.0,    'C0'),
]

fig, axes = plt.subplots(3, 5, figsize=(12, 6), sharey='row', facecolor=MILA_BG)
for row, (label, var, color) in enumerate(configs):
    acts = propagate(x0, n_layers_init, var)
    for col, (ax, h) in enumerate(zip(axes[row], acts)):
        ax.set_facecolor(MILA_BG)
        ax.hist(h, bins=40, color=color, alpha=0.75, density=True, edgecolor='none')
        ax.set_xlim(-1.1, 1.1); ax.tick_params(labelsize=7)
        if col == 0:
            ax.set_ylabel(label, fontsize=8)
        if row == 0:
            ax.set_title(f'Couche {col+1}', fontsize=9, color=MILA_PURPLE)
        ax.grid(True, alpha=0.3, axis='y')

plt.suptitle("Distribution des activations (tanh) selon l'initialisation", fontsize=11, color=MILA_PURPLE, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/ch7_weight_init.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ weight_initialization")

# ── 8. Dropout illustration ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor=MILA_BG)
titles_d = ['Réseau complet (inférence)', 'Dropout $p=0{,}5$ (entraînement)']
layer_sizes = [3, 5, 5, 2]

np.random.seed(3)
for ax_idx, ax in enumerate(axes):
    ax.set_facecolor(MILA_BG)
    ax.axis('off')
    ax.set_title(titles_d[ax_idx], fontsize=10, color=MILA_PURPLE, fontweight='bold')

    positions = []
    for l, size in enumerate(layer_sizes):
        x = l / (len(layer_sizes) - 1)
        ys = np.linspace(0.1, 0.9, size)
        positions.append([(x, y) for y in ys])

    # Determine dropped neurons for dropout panel
    dropped = set()
    if ax_idx == 1:
        for l in [1, 2]:
            for i in range(layer_sizes[l]):
                if np.random.rand() < 0.5:
                    dropped.add((l, i))

    # Draw connections
    for l in range(len(positions) - 1):
        for i, (x1, y1) in enumerate(positions[l]):
            for j, (x2, y2) in enumerate(positions[l+1]):
                is_dead = (l, i) in dropped or (l+1, j) in dropped
                color_e = '#cccccc' if is_dead else '#aaaaaa'
                alpha_e = 0.2 if is_dead else 0.5
                ax.plot([x1, x2], [y1, y2], '-', color=color_e, lw=1, alpha=alpha_e, zorder=1)

    # Draw neurons
    for l, layer_pos in enumerate(positions):
        for i, (x, y) in enumerate(layer_pos):
            is_dead = (l, i) in dropped
            if is_dead:
                fc = '#dddddd'; ec = '#999999'; lw = 1
            else:
                fc = MILA_PURPLE if l == 0 or l == len(positions)-1 else '#ffffff'
                ec = MILA_PURPLE; lw = 2
            circle = plt.Circle((x, y), 0.04, fc=fc, ec=ec, lw=lw, zorder=3)
            ax.add_patch(circle)
            if is_dead:
                ax.text(x, y, '✕', ha='center', va='center', fontsize=8, color='#999999', zorder=4)

    ax.set_xlim(-0.1, 1.1); ax.set_ylim(0.0, 1.0)
    layer_labels = ['Entrée', 'Cachée 1', 'Cachée 2', 'Sortie']
    for l, (lbl, (x, _)) in enumerate(zip(layer_labels, [p[0] for p in positions])):
        ax.text(x, 0.02, lbl, ha='center', va='bottom', fontsize=8, color='#555555')

plt.suptitle('Dropout: désactivation aléatoire de neurones', fontsize=11, color=MILA_PURPLE, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT}/ch7_dropout.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ dropout")

# ── 9. JVP vs VJP mode ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), facecolor=MILA_BG)

node_labels = ['$f_1$', '$f_2$', '$f_3$', '$\\mathcal{L}$']
xs = [1.5, 3.5, 5.5, 7.5]

for ax, (title, color, direction, vec_labels, vec_pos) in zip(axes, [
    ('Mode avant (JVP) — tangent $\\tilde{v}$ se propage de gauche à droite',
     '#1f77b4', 'forward',
     ['$\\tilde{v}_0$', '$\\tilde{v}_1 = J_{f_1}\\tilde{v}_0$',
      '$\\tilde{v}_2 = J_{f_2}\\tilde{v}_1$', '$\\tilde{v}_3 = J_{f_3}\\tilde{v}_2$'],
     [0.6, 2.5, 4.5, 6.5]),
    ('Mode arrière (VJP) — adjoint $\\bar{u}$ se propage de droite à gauche',
     '#d62728', 'backward',
     ['$\\bar{u}_0 = J_{f_1}^\\top\\bar{u}_1$', '$\\bar{u}_1 = J_{f_2}^\\top\\bar{u}_2$',
      '$\\bar{u}_2 = J_{f_3}^\\top\\bar{u}_3$', '$\\bar{u}_3 = 1$'],
     [0.6, 2.5, 4.5, 6.5])
]):
    ax.set_facecolor(MILA_BG)
    ax.set_xlim(0, 9); ax.set_ylim(0, 2); ax.axis('off')
    ax.set_title(title, fontsize=10, pad=4, color=MILA_PURPLE)
    for x, label in zip(xs, node_labels):
        circ = plt.Circle((x, 1), 0.45, color='#f0f0f0', ec='#444444', linewidth=1.5, zorder=3)
        ax.add_patch(circ)
        ax.text(x, 1, label, ha='center', va='center', fontsize=11, zorder=4)
    arrow_xs = list(zip(xs[:-1], xs[1:]))
    if direction == 'forward':
        for x1, x2 in arrow_xs:
            ax.annotate('', xy=(x2 - 0.47, 1), xytext=(x1 + 0.47, 1),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
    else:
        for x1, x2 in arrow_xs:
            ax.annotate('', xy=(x1 + 0.47, 1), xytext=(x2 - 0.47, 1),
                        arrowprops=dict(arrowstyle='->', color=color, lw=2))
    for vl, vx in zip(vec_labels, vec_pos):
        ax.text(vx + 0.45, 0.35, vl, ha='center', va='center', fontsize=8, color=color,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, alpha=0.85))

plt.tight_layout()
plt.savefig(f"{OUT}/ch7_jvp_vjp.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ jvp_vjp")

print("\nAll figures saved to", OUT)
