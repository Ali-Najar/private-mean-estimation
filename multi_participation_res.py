import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import scipy.linalg as la

EXPS = 13
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
    
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        res = pickle.load(f)
    print("Loaded cached results.")

n = n_range[-1]

labels = [
    "(i) $\mathbf{D} \mathbf{A}_1^{1/2}, \mathbf{A}_1^{1/2}$",
    "(ii) $\mathbf{A}, \mathbf{I}$",
    "(vi) $\mathbf{A} \mathbf{D}_{\mathrm{Toep}}^{-1/2}, \mathbf{D}_{\mathrm{Toep}}^{1/2}$",
    "(viii) $\mathbf{A} \mathbf{D}_{\mathrm{Toep}}^{-1}, \mathbf{D}_{\mathrm{Toep}}$",
]

labels_bsr = [
    "(vi) $\mathbf{A} \mathbf{D}_{\mathrm{Toep}}^{-1/2}, \mathbf{D}_{\mathrm{Toep}}^{1/2}$ (BSR)",
    "(viii) $\mathbf{A} \mathbf{D}_{\mathrm{Toep}}^{-1}, \mathbf{D}_{\mathrm{Toep}}$ (BSR)",
    " $\mathbf{A} \mathbf{A}_{1}^{-1/2}, \mathbf{A}_{1}^{1/2}$ (BSR)",
]
labels_bisr = [
    "(vi) $\mathbf{A} \mathbf{D}_{\mathrm{Toep}}^{-1/2}, \mathbf{D}_{\mathrm{Toep}}^{1/2}$ (BISR)",
    "(viii) $\mathbf{A} \mathbf{D}_{\mathrm{Toep}}^{-1}, \mathbf{D}_{\mathrm{Toep}}$ (BISR)",
    " $\mathbf{A} \mathbf{A}_{1}^{-1/2}, \mathbf{A}_{1}^{1/2}$ (BISR)",
]

for k in k_values:
    for i in range(1, number_of_plots + 1):
        print(f'{res[k][i][n]}, N = {n}, k = {k}, {labels[i-1]}')
    for i in range(1, number_of_plots_banded + 1):
        print(f" {res_banded[k]['bsr'][i][n]}, N = {n}, k = {k}, {labels_bsr[i-1]}")
    for i in range(1, number_of_plots_banded + 1):
        print(f"{res_banded[k]['bisr'][i][n]}, N = {n}, k = {k}, {labels_bisr[i-1]}")