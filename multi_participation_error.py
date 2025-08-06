import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.linalg as la
def sens(C: np.ndarray, k: int, b: int) -> float:

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


def obj(B, C, k ,b):
    n = B.shape[0]
    return np.linalg.norm(B,'fro') * sens(C, k, b) / np.sqrt(n)



EXPS = 9
k_values = [4,16,64]
exponents = np.arange(0, EXPS + 1)   # 2^0 ... 2^12 = 4096
n_range   = 2**exponents
# n_range_aof = 2**np.arange(0, EXPS_AOF + 1)   # 2^0 ... 2^9 = 512
number_of_plots = 4

res = {k:{i:[] for i in range(1,number_of_plots+1)} for k in k_values}


for n in n_range:
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
        res[k][1].append(obj(D @ A1_s, A1_s, k, b))

        res[k][2].append(obj(A, I, k, b))

        # 6. A @ D_toep^{-½}, D_toep^½
        res[k][3].append(obj(A @ Dtp_is, Dtp_s, k, b))

        # 8. A @ D_toep^{-1}, D_toep
        res[k][4].append(obj(A @ la.inv(Dtp), Dtp, k, b))

    print(f"n={n}")

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
        "(ii) $\mathbf{A}, \mathbf{I}$",
        "(vi) $\mathbf{A} \mathbf{D}_{\mathrm{Toep}}^{-1/2}, \mathbf{D}_{\mathrm{Toep}}^{1/2}$",
        "(viii) $\mathbf{A} \mathbf{D}_{\mathrm{Toep}}^{-1}, \mathbf{D}_{\mathrm{Toep}}$",
    ]

    markers = [None,'o', 's', '>', 'D']
    colors = [
        None,
        'tab:blue',   # 1
        'tab:orange', # 2
        'tab:green',    # 6
        'tab:red',   # 8
    ]
    plot_all = True
    plot_log_scale = True

    for i in range(1,number_of_plots+1):
        plt.plot(n_range[EXPS - len(res[k][i]) + 1:], res[k][i], marker=markers[i], color=colors[i] , label=labels[i-1],  markersize=10)
 
    plt.xlabel('Matrix Size')
    if plot_log_scale:
        plt.xscale('log')
    plt.ylabel('$\mathcal{E}(\mathbf{B},\mathbf{C})$')
    plt.legend()
    plt.tight_layout(pad=0)
    savefile_name = f"_multi_error_vs_mat_size_k{k}.pdf"
    savefile_name = os.path.join("plots", savefile_name)
    plt.savefig(savefile_name, format="pdf")
# plt.show()
