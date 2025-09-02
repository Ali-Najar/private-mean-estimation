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


EXPS = 14
k_values = [4,16,64]
exponents = np.arange(0, EXPS + 1)   # 2^0 ... 2^12 = 4096
n_range   = 2**exponents
# n_range_aof = 2**np.arange(0, EXPS_AOF + 1)   # 2^0 ... 2^9 = 512
number_of_plots = 4
number_of_plots_banded = 3
cache_path = 'cache/multi_res_cache.pkl'
banded_cache_path = 'cache/multi_banded_cache.pkl'


# -- Caching setup ---------------------------------------------------------
if os.path.exists(banded_cache_path):
    with open(banded_cache_path, 'rb') as f:
        res_banded = pickle.load(f)
    print("Loaded cached banded results.")
else:
    res_banded = {k: {"bisr": {i: {} for i in range(1, number_of_plots_banded + 1)}, "bsr": {i: {} for i in range(1, number_of_plots_banded + 1)}} for k in k_values}
    
    print("Initialized empty banded cache.")
    
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        res = pickle.load(f)
    print("Loaded cached results.")
else:
    res = {k: {i: {} for i in range(1, number_of_plots + 1)} for k in k_values}
    print("Initialized empty cache.")

for n in n_range:

    flag = True

    for k in k_values:
        if k > n:
            continue    

        b = n // k
        for i in range(1, number_of_plots + 1):
            if n not in res[k][i]:
                flag = False
                break
        for i in range(1, number_of_plots_banded + 1):
            if n not in res_banded[k]['bisr'][i] or n not in res_banded[k]['bsr'][i]:
                flag = False
                break
    
    if flag:
        print(f"Completed n={n}")
        continue

    A    = generate_A(n)
    D    = generate_D(n)
    A1   = generate_A1(n)
    Dtp  = generate_D_toep(n)
    I    = np.eye(n)


    A1_s = la.sqrtm(A1)
    Dtp_s  = la.sqrtm(Dtp)
    Dtp_is = la.inv(Dtp_s)

    

    for k in k_values:
        if k > n:
            continue    

        b = n // k

        # 1. D @ A1^½, A1^½
        if n not in res[k][1]:
            res[k][1][n] = obj(D @ A1_s, A1_s, k, b)

        if n not in res[k][2]:
            res[k][2][n] = obj(A, I, k, b)

        # 6. A @ D_toep^{-½}, D_toep^½
        if n not in res[k][3]:
            res[k][3][n] = obj(A @ Dtp_is, Dtp_s, k, b)

        # 8. A @ D_toep^{-1}, D_toep
        if n not in res[k][4]:
            res[k][4][n] = obj(A @ la.inv(Dtp), Dtp, k, b)

        p = b
        if n not in res_banded[k]['bisr'][1]:
            res_banded[k]['bisr'][1][n] = bisr_obj(A, Dtp_s, k, b, p)
        if n not in res_banded[k]['bisr'][2]:
            res_banded[k]['bisr'][2][n] = bisr_obj(A, Dtp, k, b, p)
        if n not in res_banded[k]['bisr'][3]:
            res_banded[k]['bisr'][3][n] = bisr_obj(A, A1_s, k, b, p)
        
        if n not in res_banded[k]['bsr'][1]:
            res_banded[k]['bsr'][1][n] = bsr_obj(A, Dtp_s, k, b, p)
        if n not in res_banded[k]['bsr'][2]:
            res_banded[k]['bsr'][2][n] = bsr_obj(A, Dtp, k, b, p)
        if n not in res_banded[k]['bsr'][3]:
            res_banded[k]['bsr'][3][n] = bsr_obj(A, A1_s, k, b, p)
        

    print(f"Completed n={n}")
    with open(cache_path, 'wb') as f:
        pickle.dump(res, f)
    with open(banded_cache_path, 'wb') as f:
        pickle.dump(res_banded, f)

for k in k_values:

    
    plt.figure(figsize=(10,6))


    plt.rcParams.update({
        # use LaTeX to render all text
        "text.usetex":    True,
        # default font family for text
        "font.family":    "serif",
        # ask LaTeX for Computer Modern Roman
        "font.serif":     ["Computer Modern Roman"],
        # match your desired size/weight
        "font.size":      24,
        "axes.labelsize": 40,   # x/y label font size
        "xtick.labelsize": 40,  # x tick font size
        "ytick.labelsize": 40,  # y tick font size
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
        " $\mathbf{A} \mathbf{A}_{1}^{-1/2}, \mathbf{A}_{1}^{1/2}$ (BSR)",
    ]
    labels_bisr = [
        "$\mathbf{D}_{\mathrm{Toep}}^{1/2}$ (Banded Inverse)",
        "$\mathbf{D}_{\mathrm{Toep}}$ (Banded Inverse)",
        " $\mathbf{A} \mathbf{A}_{1}^{-1/2}, \mathbf{A}_{1}^{1/2}$ (BISR)",
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
    plot_ratio = True

    
    for i in range(1,number_of_plots+1):
        if i!=4:
            continue
        ns = sorted(res[k][i].keys())
        if plot_ratio:
            ys = [res[k][i][n] / res_banded[k]['bsr'][2][n] for n in ns]
        else:
            ys = [res[k][i][n] for n in ns]
        plt.plot(ns, ys, marker=markers[i], color=colors[i] , label=labels[i-1],  markersize=10)

    for i in range(1, number_of_plots_banded + 1):
        if i == 3:
            continue
        ns = sorted(res_banded[k]['bisr'][i].keys())
        if plot_ratio:
            ys = [res_banded[k]['bisr'][i][n] / res_banded[k]['bsr'][2][n] for n in ns]
        else: 
            ys = [res_banded[k]['bisr'][i][n] for n in ns]
        plt.plot(ns, ys, marker=markers_bisr[i], color=colors_bisr[i] , label=labels_bisr[i-1],  markersize=10)
        
    for i in range(1, number_of_plots_banded + 1):
        if i == 3:
            continue
        ns = sorted(res_banded[k]['bsr'][i].keys())
        if plot_ratio:
            ys = [res_banded[k]['bsr'][i][n] / res_banded[k]['bsr'][2][n] for n in ns]
        else:
            ys = [res_banded[k]['bsr'][i][n] for n in ns]
        plt.plot(ns, ys, marker=markers_bsr[i], color=colors_bsr[i] , label=labels_bsr[i-1],  markersize=10)

    plt.xlabel('Matrix Size')
    if plot_log_scale:
        plt.xscale('log')
    plt.ylabel('$\mathcal{E}(\mathbf{B},\mathbf{C})$')
    plt.legend()
    plt.tight_layout()
    savefile_name = f"_multi_error_vs_mat_size_k{k}_ratio{plot_ratio}.pdf"
    savefile_name = os.path.join("plots", savefile_name)
    plt.savefig(savefile_name, format="pdf")
# plt.show()
