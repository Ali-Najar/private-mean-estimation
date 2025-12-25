
import os
import numpy as np
import matplotlib.pyplot as plt

CACHE_FILE = 'cache/efficient_error_data.npz'
BANDED_CACHE_FILE = 'cache/efficient_banded_cache.npz'

prev_len = 0
prev_len_banded = 0

errors_for_A_BandMF = []
# D A1_sqrt / A1_sqrt
errors_for_D_A1_sqrt = []
# A D_toep^{-1/2} / D_toep^{1/2}
errors_for_A_D_sqrt_inv = []
# A D_toep^{-1} / D_toep
errors_for_A_D_inv = []
# A I / I
errors_for_A_I = []

if os.path.exists(CACHE_FILE):
    # load whatever was computed before
    data = np.load(CACHE_FILE)
    errors_for_D_A1_sqrt    = list(data['errors_for_D_A1_sqrt'])
    errors_for_A_I          = list(data['errors_for_A_I'])
    errors_for_A_D_sqrt_inv = list(data['errors_for_A_D_sqrt_inv'])
    errors_for_A_D_inv      = list(data['errors_for_A_D_inv'])
    prev_len = len(errors_for_A_I)

if os.path.exists(BANDED_CACHE_FILE):
    # load whatever was computed before
    data = np.load(BANDED_CACHE_FILE)
    errors_for_A_BandMF     = list(data.get('errors_for_A_BandMF', []))
    prev_len_banded = len(errors_for_A_BandMF)

exponents = np.arange(0, prev_len + 1)   # 2^0 ... 2^12 = 4096
n_range   = 2**exponents

plot_ratio = True

plt.figure(figsize=(10/1.3,6/1.3))

plt.rcParams.update({
    # use LaTeX to render all text
    "text.usetex":    True,
    # default font family for text
    "font.family":    "serif",
    # ask LaTeX for Computer Modern Roman
    "font.serif":     ["Computer Modern Roman"],
    # match your desired size/weight
    "font.size":      16,
    "font.weight":    "normal",
    # ensure math uses the same font
    "text.latex.preamble": r"\usepackage{amsmath}"
})

K = 15

if plot_ratio == False:
    plt.plot(n_range[:K], errors_for_D_A1_sqrt[:K], marker='^', label='(i) $\mathbf{D} \mathbf{E}_1^{1/2}, \mathbf{E}_1^{1/2}$')
    plt.plot(n_range[:K], errors_for_A_I[:K], marker='^', label='(ii) $\mathbf{A}, \mathbf{I}$')
    plt.plot(n_range[:K], errors_for_A_D_sqrt_inv[:K], marker='s', label='(vi) $\mathbf{A} \mathbf{D}_{\mathrm{Toep}}^{-1/2}, \mathbf{D}_{\mathrm{Toep}}^{1/2}$')
    plt.plot(n_range[:K], errors_for_A_D_inv[:K], marker='D', label='RMSE of (viii) $\mathbf{A} \mathbf{D}_{\mathrm{Toep}}^{-1}, \mathbf{D}_{\mathrm{Toep}}$')
    plt.plot(n_range[:K], errors_for_A_BandMF[:K], label='Optimal')
    plt.xlabel('Matrix Size')
    plt.xscale('log')
    plt.ylabel('$\mathrm{RMSE}(\mathbf{B},\mathbf{C})$')
    plt.legend()
    plt.tight_layout(pad=0)
    plt.savefig("plots/efficient_error_vs_log_mat_size.pdf", format="pdf")
    plt.show()
else:
    ratios_D_A1_sqrt = [errors_for_D_A1_sqrt[i] / errors_for_A_BandMF[i] for i in range(K)]
    ratios_AI = [errors_for_A_I[i] / errors_for_A_BandMF[i] for i in range(K)]
    ratios_D_toep = [errors_for_A_D_inv[i] / errors_for_A_BandMF[i] for i in range(K)]
    ratios_D_toep_sqrt = [errors_for_A_D_sqrt_inv[i] / errors_for_A_BandMF[i] for i in range(K)]
    plt.plot(n_range[:K],ratios_D_A1_sqrt[:K], color='tab:blue', marker='o', label='(i) $\mathbf{D} \mathbf{E}_1^{1/2}, \mathbf{E}_1^{1/2}$')
    plt.plot(n_range[:K],ratios_AI[:K], color='tab:orange', marker='s', label='(ii) $\mathbf{A}, \mathbf{I}$')
    plt.plot(n_range[:K],ratios_D_toep_sqrt[:K], color='tab:green', marker='>' , label='(vi) $\mathbf{A} \mathbf{D}_{\mathrm{Toep}}^{-1/2}, \mathbf{D}_{\mathrm{Toep}}^{1/2}$')
    plt.plot(n_range[:K],ratios_D_toep[:K], color='tab:red', marker='D', label='(viii) $\mathbf{A} \mathbf{D}_{\mathrm{Toep}}^{-1}, \mathbf{D}_{\mathrm{Toep}}$')
    plt.axhline(
    y=1.0,
    linestyle='--',
    label='BandMF'
    )
    plt.xlabel('Matrix Size')
    plt.xscale('log')
    plt.ylabel('Ratio to the Optimal Error')
    plt.legend()
    plt.tight_layout(pad=0)
    plt.savefig("plots/efficient_error_ratio_vs_log_mat_size.pdf", format="pdf")
    plt.show()