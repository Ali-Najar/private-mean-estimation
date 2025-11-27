import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import scipy.linalg as la

def sens(C: np.ndarray, k: int, b: int):

    # pick out the k columns [0, b, 2b, ..., (k-1)b]
    M = C[:, :k*b:b]
    # form matrix G = M^T M, then sum all entries
    G = M.T @ M 
    return np.sqrt(float(np.sum(G)))

def generate_A(n):
    A = np.zeros((n,n))
    for i in range(n):
        A[i,:i+1] = 1.0/(i+1)
    return A

def generate_D(n):
    return np.diag(1.0/np.arange(1,n+1))

def generate_A1(n):
    return np.tri(n)

def generate_D_toep(n):
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            M[i,j] = 1.0/(i-j+1)
    return M


def banded(C: np.ndarray, p: int):

    C = np.asarray(C)
    n = C.shape[0]

    # build mask: True when i-j < p, False otherwise
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    mask = (i - j) < p

    # apply mask
    Cb = C.copy()
    Cb[~mask] = 0
    return Cb

def obj(B, C, k ,b):
    n = B.shape[0]
    return np.linalg.norm(B,'fro') * sens(C, k, b) / np.sqrt(n)

def bsr_obj(A, C, k, b, p):
    
    C_p = banded(C, p)
    C_p_inv = la.inv(C_p)
    return obj(A @ C_p_inv, C_p, k, b)

def bisr_obj(A, C, k, b, p):

    C_inv = la.inv(C)
    C_p_inv = banded(C_inv, p)
    new_C = la.inv(C_p_inv)
    return obj(A @ C_p_inv, new_C, k, b)


n = 2048
k = 128
b = n // k

p_vals = [2**i for i in range(1, np.log2(n).astype(int) + 1)]

number_of_plots_banded = 3
banded_cache_path = 'cache/multi_banded_over_p_cache_logb.pkl'


# -- Caching setup ---------------------------------------------------------
if os.path.exists(banded_cache_path):
    with open(banded_cache_path, 'rb') as f:
        res_banded = pickle.load(f)
    print("Loaded cached banded results.")
else:
    res_banded = {p: {"bisr": { k_val: {i: {} for i in range(1, number_of_plots_banded + 1)} for k_val in p_vals}, "bsr": { k_val: {i: {} for i in range(1, number_of_plots_banded + 1)} for k_val in p_vals} } for p in p_vals}
    
    print("Initialized empty banded cache.")

for p in p_vals:
    if p not in res_banded:
        res_banded[p] = {"bisr": { k_val: {i: {} for i in range(1, number_of_plots_banded + 1)} for k_val in p_vals}, "bsr": { k_val: {i: {} for i in range(1, number_of_plots_banded + 1)} for k_val in p_vals} }

for p in p_vals:

    flag = True
    
    for i in range(1, number_of_plots_banded + 1):
        if n not in res_banded[p]['bisr'][k][i] or n not in res_banded[p]['bsr'][k][i]:
            flag = False
            break
    
    if flag:
        print(f"Completed p={p}, k={k}, n={n}")
        continue


    A    = generate_A(n)
    A1   = generate_A1(n)
    Dtp  = generate_D_toep(n)


    A1_s = la.sqrtm(A1)
    Dtp_s  = la.sqrtm(Dtp)

    
    if n not in res_banded[p]['bisr'][k][1]:
        res_banded[p]['bisr'][k][1][n] = bisr_obj(A, Dtp_s, k, b, p)
    if n not in res_banded[p]['bisr'][k][2]:
        res_banded[p]['bisr'][k][2][n] = bisr_obj(A, Dtp, k, b, p)
    if n not in res_banded[p]['bisr'][k][3]:
        res_banded[p]['bisr'][k][3][n] = bisr_obj(A, A1_s, k, b, p)
    
    if n not in res_banded[p]['bsr'][k][1]:
        res_banded[p]['bsr'][k][1][n] = bsr_obj(A, Dtp_s, k, b, p)
    if n not in res_banded[p]['bsr'][k][2]:
        res_banded[p]['bsr'][k][2][n] = bsr_obj(A, Dtp, k, b, p)
    if n not in res_banded[p]['bsr'][k][3]:
        res_banded[p]['bsr'][k][3][n] = bsr_obj(A, A1_s, k, b, p)
        

    print(f"Completed p={p}, k={k}, n={n}")
    with open(banded_cache_path, 'wb') as f:
        pickle.dump(res_banded, f)

    
plt.figure(figsize=(10,6))


plt.rcParams.update({
    # use LaTeX to render all text
    "text.usetex":    True,
    # default font family for text
    "font.family":    "serif",
    # ask LaTeX for Computer Modern Roman
    "font.serif":     ["Computer Modern Roman"],
    # match your desired size/weight
    "font.size":      14,
    "axes.labelsize": 22,   # x/y label font size
    "xtick.labelsize": 20,  # x tick font size
    "ytick.labelsize": 20,  # y tick font size
    "font.weight":    "normal",
    # ensure math uses the same font
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}"
})

labels = [
    "(i) $\mathbf{D} \mathbf{A}_1^{1/2}, \mathbf{A}_1^{1/2}$",
    "$\mathbf{I}$",
    "(vi) $\mathbf{A} \mathbf{D}_{\mathrm{Toep}}^{-1/2}, \mathbf{D}_{\mathrm{Toep}}^{1/2}$",
    "$\mathbf{D}_{\mathrm{Toep}}$",
]

labels_bsr = [
    "$\mathbf{D}_{\mathrm{Toep}}^{1/2}$ (Banded)",
    "$\mathbf{D}_{\mathrm{Toep}}$ (Banded)",
    " $\mathbf{A} \mathbf{E}_{1}^{-1/2}, \mathbf{E}_{1}^{1/2}$ (BSR)",
]
labels_bisr = [
    "$\mathbf{D}_{\mathrm{Toep}}^{1/2}$ (Banded Inverse)",
    "$\mathbf{D}_{\mathrm{Toep}}$ (Banded Inverse)",
    " $\mathbf{A} \mathbf{E}_{1}^{-1/2}, \mathbf{E}_{1}^{1/2}$ (BISR)",
]

markers = [None,'o', 's', '>', 'D']

markers_bsr = [None, 'd', 'x', '+']
markers_bisr = [None, '^', 'v', ',']

colors = [
    None,
    'tab:cyan',   # 1
    'tab:orange', # 2
    'tab:green',    # 6
    'tab:red',   # 8
]

colors_bsr = [
    None,
    'tab:olive',
    'tab:brown',
    'teal',
]
colors_bisr = [
    None,
    'tab:purple',
    'tab:blue',
    'tab:pink',
]

plot_all = True
plot_log_scale = True
plot_ratio = False

ax = plt.gca()
lines = []

for i in range(1, number_of_plots_banded + 1):
    if i == 1:
        continue 
    ns = sorted(p_vals)
    if plot_ratio:
        ys = [res_banded[p]['bsr'][k][i][n] / res_banded[p]['bsr'][k][2][n] for p in ns]
    else:
        ys = [res_banded[p]['bsr'][k][i][n] for p in ns]
    (line_bsr,) = plt.plot(ns, ys, marker=markers_bsr[i], color=colors_bsr[i] , label=labels_bsr[i-1],  markersize=10)

    lines.append(line_bsr)

    ns = sorted(p_vals)
    if plot_ratio:
        ys = [res_banded[p]['bisr'][k][i][n] / res_banded[p]['bsr'][k][2][n] for p in ns]
    else: 
        ys = [res_banded[p]['bisr'][k][i][n] for p in ns]
    (line_bisr,) = plt.plot(ns, ys, marker=markers_bisr[i], color=colors_bisr[i] , label=labels_bisr[i-1],  markersize=10)
    
    lines.append(line_bisr)

pb = np.log2(n)
for line in lines:
    x = np.asarray(line.get_xdata())
    y = np.asarray(line.get_ydata(), dtype=float)
    mask = np.isfinite(y)
    if not np.any(mask):
        continue
    x_valid, y_valid = x[mask], y[mask]
    # find the data point whose p is closest to b (works even if b not in p_vals)
    idx = int(np.argmin(np.abs(x_valid - pb)))
    xb, yb = float(x_valid[idx]), float(y_valid[idx])
    c = line.get_color()
    # bold hollow circle + small filled dot
    ax.scatter([xb], [yb], s=220, facecolors='none', edgecolors=c, linewidths=2.8, zorder=10)
    ax.scatter([xb], [yb], s=36, color=c, zorder=11)

# optional guide line
ax.axvline(pb, linestyle='--', linewidth=1, color='k', alpha=0.25)

plt.xlabel('$p$')
if plot_log_scale:
    plt.xscale('log')
plt.ylabel('$\mathcal{E}(\mathbf{B},\mathbf{C})$')
plt.legend()
plt.tight_layout(pad=0)
savefile_name = f"_multi_error_vs_p_val_k{k}_n{n}_ratio{plot_ratio}.pdf"
savefile_name = os.path.join("plots", savefile_name)
plt.savefig(savefile_name, format="pdf")
# plt.show()
