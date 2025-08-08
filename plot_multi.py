import numpy as np
import pickle
import os
import matplotlib.pyplot as plt



EXPS = 15
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

    markers = [None,'o', 's', '>', 'D']

    markers_bsr = [None, '^', 'v', '<']
    markers_bisr = [None, '+', 'x', ',']
    
    colors = [
        None,
        'tab:blue',   # 1
        'tab:orange', # 2
        'tab:green',    # 6
        'tab:red',   # 8
    ]

    colors_bsr = [
        None,
        'tab:brown',
        'tab:olive',
        'teal',
    ]
    colors_bisr = [
        None,
        'tab:purple',
        'tab:cyan',
        'tab:pink',
    ]

    plot_all = True
    plot_log_scale = True
    plot_ratio = True

    
    for i in range(1,number_of_plots+1):

        ns = sorted(res[k][i].keys())
        if plot_ratio:
            ys = [res[k][i][n] / res_banded[k]['bsr'][2][n] for n in ns]
        else:
            ys = [res[k][i][n] for n in ns]
        plt.plot(ns, ys, marker=markers[i], color=colors[i] , label=labels[i-1],  markersize=10)

    for i in range(1, number_of_plots_banded + 1):
        ns = sorted(res_banded[k]['bsr'][i].keys())
        if plot_ratio:
            ys = [res_banded[k]['bsr'][i][n] / res_banded[k]['bsr'][2][n] for n in ns]
        else:
            ys = [res_banded[k]['bsr'][i][n] for n in ns]
        plt.plot(ns, ys, marker=markers_bsr[i], color=colors_bsr[i] , label=labels_bsr[i-1],  markersize=10)

        ns = sorted(res_banded[k]['bisr'][i].keys())
        if plot_ratio:
            ys = [res_banded[k]['bisr'][i][n] / res_banded[k]['bsr'][2][n] for n in ns]
        else: 
            ys = [res_banded[k]['bisr'][i][n] for n in ns]
        plt.plot(ns, ys, marker=markers_bisr[i], color=colors_bisr[i] , label=labels_bisr[i-1],  markersize=10)

    plt.xlabel('Matrix Size')
    if plot_log_scale:
        plt.xscale('log')
    plt.ylabel('$\mathcal{E}(\mathbf{B},\mathbf{C})$')
    plt.legend()
    plt.tight_layout(pad=0)
    savefile_name = f"_multi_error_vs_mat_size_k{k}_ratio{plot_ratio}.pdf"
    savefile_name = os.path.join("plots", savefile_name)
    plt.savefig(savefile_name, format="pdf")
# plt.show()
