import numpy as np
from scipy import linalg as la
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import cvxpy as cp
import os


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

def obj(B,C):
    n = C.shape[1]
    return np.linalg.norm(B,'fro') * np.max(np.linalg.norm(C,axis=0)) / np.sqrt(n)


def aof_factorization(A):
    
    n = A.shape[0]
    X = cp.Variable((n,n), PSD=True)

    objective = cp.Minimize(cp.matrix_frac(A.T, X))
    constraints = [cp.diag(X) <= 1]

    prob = cp.Problem(objective, constraints)
    raw_obj = prob.solve(solver='SCS')

    Xval = (X.value + X.value.T) / 2
    C = cholesky(Xval, lower=False)

    B = A @ np.linalg.inv(C)

    return B, C, np.sqrt(raw_obj/n)


def solve_sdp(n_dim, A):
    """
    Implements and solves the semidefinite program described.

    Args:
        n_dim (int): The dimension 'n' for the matrices.
                     (Here, we assume m = n as per the problem description).
    """

    # Define the variable for eta (scalar)
    eta = cp.Variable()

    # Define the matrix variables
    # X1 is n_dim x n_dim symmetric
    X1 = cp.Variable((n_dim, n_dim), symmetric=True)
    # X2 is n_dim x n_dim
    X2 = cp.Variable((n_dim, n_dim))
    # X3 is n_dim x n_dim symmetric
    X3 = cp.Variable((n_dim, n_dim), symmetric=True)

    # Construct the block matrix X
    # X = [[X1, X2],
    #      [X2.H, X3]]  (X2.H is conjugate transpose, but for real matrices, it's just transpose)
    # Since X is supposed to be symmetric (Hermitian for complex), X2.H is indeed X2.T
    X = cp.bmat([[X1, X2],
                 [X2.T, X3]])

    # Define A as np.tri(n_dim)

    # Constraints
    constraints = []

    # 1. X is positive semidefinite
    constraints.append(X >> 0)  # X >= 0 (positive semidefinite)

    # 2. X2 = A
    constraints.append(X2 == A)

    # 3. Tr(X1) = eta
    constraints.append(cp.trace(X1) == eta)

    # 4. X3[i,i] <= eta for all i
    constraints.append(cp.diag(X3) <= eta)

    # Objective: Minimize eta
    objective = cp.Minimize(eta)

    # Define the problem
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve(solver=cp.SCS)

    return eta.value / np.sqrt(n_dim)


def load_or_compute_aof(A, path="aof_factors.npz"):
    if os.path.exists(path):
        npz = np.load(path)
        B, C, obj_value = npz["B"], npz["C"], npz["obj_value"]
        print(f"Loaded B, C, obj_value from {path}")
        return B, C, obj_value
    else:
        B, C, obj_value = aof_factorization(A)
        np.savez(path, B=B, C=C, obj_value=obj_value)
        return B, C, obj_value
    


def load_or_compute_sdp(n, A, path="sdp_obj.npz"):
    cache = f"{os.path.splitext(path)[0]}_{n}.npz"
    if os.path.exists(cache):
        data = np.load(cache)
        obj = data["obj"]
        print(f"Loaded SDP objective for n={n} from {cache}")
        return obj
    else:
        obj = solve_sdp(n,A)
        np.savez(cache, obj=obj)
        print(f"Computed SDP for n={n}, saved obj={obj:.6f} to {cache}")
        return obj
    

EXPS = 11
EXPS_AOF = 8
exponents = np.arange(0, EXPS + 1)   # 2^0 ... 2^12 = 4096
n_range   = 2**exponents
n_range_aof = 2**np.arange(0, EXPS_AOF + 1)   # 2^0 ... 2^9 = 512
number_of_plots = 10

res = {i:[] for i in range(1,number_of_plots+1)}


for n in n_range:
    A    = generate_A(n)
    D    = generate_D(n)
    A1   = generate_A1(n)
    Dtp  = generate_D_toep(n)
    I    = np.eye(n)

    # 1. D @ A1^½, A1^½
    A1_s = la.sqrtm(A1)
    res[1].append(obj(D @ A1_s, A1_s))

    res[2].append(obj(A, I))

    res[3].append(obj(I, A))

    # 4. A^½, A^½
    A_s = la.sqrtm(A)
    res[4].append(obj(A_s, A_s))

    # 5. A1^½, A1^{-½} @ A
    A1_is = la.inv(A1_s)
    res[5].append(obj(A1_s, A1_is @ A))

    # 6. A @ D_toep^{-½}, D_toep^½
    Dtp_s  = la.sqrtm(Dtp)
    Dtp_is = la.inv(Dtp_s)
    res[6].append(obj(A @ Dtp_is, Dtp_s))

    # 7. D_toep^½, D_toep^{-½}@A
    res[7].append(obj(Dtp_s, Dtp_is @ A))

    # 8. A @ D_toep^{-1}, D_toep
    res[8].append(obj(A @ la.inv(Dtp), Dtp))

for n in n_range_aof:
    # 9. AOF
    B, C, obj_value = load_or_compute_aof(A, path=f"cache/my_aof_{n}.npz")
    res[9].append(obj_value)
    
    # sdp_obj = load_or_compute_sdp(n, A, path=f"sdp_obj_{n}.npz")
    # res[9].append(sdp_obj)

    print(f"n={n}")


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
    "(iii) $\mathbf{I}, \mathbf{A}$",
    "(iv) $\mathbf{A}^{1/2}, \mathbf{A}^{1/2}$",
    "(v) $\mathbf{A}_1^{1/2}, \mathbf{A}_1^{-1/2} \mathbf{A}$",
    "(vi) $\mathbf{A} \mathbf{D}_{\mathrm{Toep}}^{-1/2}, \mathbf{D}_{\mathrm{Toep}}^{1/2}$",
    "(vii) $\mathbf{D}_{\mathrm{Toep}}^{1/2}, \mathbf{D}_{\mathrm{Toep}}^{-1/2} \mathbf{A}$",
    "(viii) $\mathbf{A} \mathbf{D}_{\mathrm{Toep}}^{-1}, \mathbf{D}_{\mathrm{Toep}}$",
    "AOF" 
]

markers = [None,'o', 's', 'p', '^', 'v', '>', '<', 'D', 'x']
colors = [
    None,
    'tab:blue',   # 1
    'tab:orange', # 2
    'tab:cyan',  # 3
    'tab:brown',  # 4
    'tab:purple', # 5
    'tab:green',    # 6
    'tab:pink',   # 7
    'tab:red',   # 8
    'tab:olive',  # 9
]
plot_all = True
plot_ratios = False
plot_log_scale = True

ratio_ii_aof = [res[2][i] / res[9][i] for i in range(len(res[9]))]
ratio_vi_aof = [res[6][i] / res[9][i] for i in range(len(res[9]))]
ratio_viii_aof = [res[8][i] / res[9][i] for i in range(len(res[9]))]
# ratio_sdp = [res[6][i] / res[9][i] for i in range(len(res[6]))]

for i in range(1,number_of_plots):
    if plot_all is False and i not in [2, 6, 8, 9]:
        continue
    if plot_ratios and i in [2, 6, 8]:
        plt.plot(n_range, ratio_ii_aof, label='(ii)', linestyle='--')
        plt.plot(n_range, ratio_vi_aof, label='(vi)', linestyle='--')
        plt.plot(n_range, ratio_viii_aof, label='(viii)', linestyle='--')   
    else:
        if i == 9:
            plt.plot(n_range_aof, res[i], marker=markers[i], color=colors[i], label=labels[i-1],  markersize=10)
        else:
            plt.plot(n_range, res[i], marker=markers[i], color=colors[i] , label=labels[i-1],  markersize=10)

plt.xlabel('Matrix Size')
if plot_log_scale:
    plt.xscale('log')
plt.ylabel('$\mathcal{E}(\mathbf{B},\mathbf{C})$')
plt.legend()
plt.tight_layout(pad=0)
savefile_name = "error_vs_log_mat_size.pdf" if plot_log_scale else "error_vs_mat_size.pdf"
savefile_name = "ratio_" + savefile_name if plot_ratios else savefile_name
savefile_name = os.path.join("plots", savefile_name)
plt.savefig(savefile_name, format="pdf")
plt.show()
