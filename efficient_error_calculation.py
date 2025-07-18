import numpy as np
import matplotlib.pyplot as plt

import jax_privacy
from jax_privacy.matrix_factorization import toeplitz
import jax.numpy as jnp
import numpy as np
import functools
import matplotlib.pyplot as plt
import scipy

def expected_mean_error_BandMF(coef, n: int) -> float:
  coef = jnp.pad(coef, (0, n - coef.size))
  inv_coef = toeplitz.inverse_coef(coef)
  inv_coef_cum_sum_squared = jnp.cumsum(inv_coef) ** 2
  weights = jnp.cumsum((1 / jnp.arange(1, n + 1) ** 2)[::-1])[::-1]

  B_norm_squared = (inv_coef_cum_sum_squared * weights).sum()
  
  sensitivity_squared = (coef ** 2).sum()

  return sensitivity_squared * B_norm_squared

def init_BandMF(n, p):
  return 1/ (jnp.arange(p) + 1)
  
def Band_matrix_factorization(n, p, steps=10):
    C_init = init_BandMF(n, p)
    C_opt = toeplitz.optimize_banded_toeplitz(
      n=n,
      bands=p,
      strategy_coef=C_init,
      loss_fn=functools.partial(expected_mean_error_BandMF),
      max_optimizer_steps=steps,
    )

    return expected_mean_error_BandMF(C_opt, n=n)**0.5 / np.sqrt(n)


def compute_mat_sqrt(N, Toep=True):
    """
    Compute b[0..N] satisfying
        sum_{i=0}^n b[i]*b[n-i] = 1/(n+1) or sum_{i=0}^n b[i]*b[n-i] = 1
    using NumPy dot for the convolution sum.
    """
    b = np.zeros(N, dtype=np.float64)
    b[0] = 1.0
    for n in range(1, N):
        # use a single C-speed dot product instead of the inner Python loop
        #    sum_{i=1..n-1} b[i]*b[n-i]

        if n > 1:
            s = np.dot(b[1:n], b[n-1:0:-1])
        else:
            s = 0.0
        if Toep:
            b[n] = (1.0/(n+1) - s) * 0.5
        else:
            b[n] = (1.0 - s) * 0.5
    return b


def invert_toeplitz_first_column(b):
    """
    Given b[0..N] with b[0]!=0, returns c[0..N] such that
       sum_{i=0}^n b[i]*c[n-i] = (n==0 ? 1 : 0).
    """
    N = len(b) - 1
    c = np.zeros_like(b)
    c[0] = 1.0 / b[0]
    for n in range(1, N+1):
        # sum_{i=1..n} b[i]*c[n-i]
        s = np.dot(b[1:n+1], c[n-1::-1])
        c[n] = -s / b[0]
    return c


def reverse_cumsum_inv_squares(n):
    # 1) build [1/1^2, 1/2^2, …, 1/n^2]
    arr = 1.0 / np.arange(1, n+1, dtype=np.float64)**2
    # 2) reverse it: [1/n^2, …, 1/1^2]
    rev = arr[::-1]
    # 3) do a forward cumsum on the reversed array
    rev_cum = np.cumsum(rev)
    # 4) reverse back so index k-1 holds sum_{i=k}^n 1/i^2
    return rev_cum[::-1]


CACHE_FILE = 'cache/efficient_error_data.npz'
BANDED_CACHE_FILE = 'cache/efficient_banded_cache.npz'
import os

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

EXPS = 19
exponents = np.arange(0, EXPS + 1)   # 2^0 ... 2^12 = 4096
n_range   = 2**exponents

if prev_len - 1 < EXPS:


    clipped_range = n_range[prev_len:]

    for N in clipped_range:

        print(f"Processing N={N}...")

        # D A1_sqrt , A1_sqrt
        A1_sqrt = compute_mat_sqrt(N, Toep=False)
        F_norm_D_A1_sqrt = np.dot(A1_sqrt**2, 1.0 / np.arange(1, N+1, dtype=np.float64))
        errors_for_D_A1_sqrt.append(1/np.sqrt(N) * np.sqrt(F_norm_D_A1_sqrt) * np.linalg.norm(A1_sqrt))

        # A I / I
        ordered_array = np.arange(1, N + 1, dtype=np.float64)
        H = np.cumsum(1.0 / ordered_array)
        error_AI = 1/np.sqrt(N) * np.sqrt(H[-1])
        errors_for_A_I.append(error_AI)

        # A D_toep^{-1/2} / D_toep^{1/2}
        D_toep_sqrt = compute_mat_sqrt(N, Toep=True)
        D_toep_sqrt_inv = invert_toeplitz_first_column(D_toep_sqrt)
        cumsum_D_toep_sqrt_inv_sqr = np.cumsum(D_toep_sqrt_inv)**2
        inv_squares_cumsum = reverse_cumsum_inv_squares(N)
        F_norm_A_D_sqrt_inv = np.dot(cumsum_D_toep_sqrt_inv_sqr, inv_squares_cumsum)
        errors_for_A_D_sqrt_inv.append(1/np.sqrt(N) * np.sqrt(F_norm_A_D_sqrt_inv) * np.linalg.norm(D_toep_sqrt))

        # A D_toep^{-1} / D_toep
        D_toep = 1.0 / np.arange(1, N+1, dtype=np.float64)
        column_norm = inv_squares_cumsum[0]
        D_toep_inv = invert_toeplitz_first_column(D_toep)
        cumsum_D_toep_inv_sqr = np.cumsum(D_toep_inv)**2
        D_toep_inv_sqr = D_toep_inv**2
        F_norm_A_D_inv = np.dot(cumsum_D_toep_inv_sqr, inv_squares_cumsum)
        errors_for_A_D_inv.append(1/np.sqrt(N) * np.sqrt(F_norm_A_D_inv) * np.sqrt(column_norm))
    
    # Save the results
    np.savez_compressed(
            CACHE_FILE,
            errors_for_D_A1_sqrt      = np.array(errors_for_D_A1_sqrt),
            errors_for_A_I            = np.array(errors_for_A_I),
            errors_for_A_D_sqrt_inv   = np.array(errors_for_A_D_sqrt_inv),
            errors_for_A_D_inv        = np.array(errors_for_A_D_inv),
        )

if prev_len_banded - 1 < EXPS:

    clipped_range_banded = n_range[prev_len_banded:]

    for N in clipped_range_banded:
        print(f"Processing N={N} for banded matrix factorization...")
        error_BandMF = Band_matrix_factorization(N, min(N, 128), steps=10)
        errors_for_A_BandMF.append(error_BandMF)

    # Save the results
    np.savez_compressed(
            BANDED_CACHE_FILE,
            errors_for_A_BandMF = np.array(errors_for_A_BandMF),
        )


# plot_ratio = True

# plt.figure()

# plt.rcParams.update({
#     # use LaTeX to render all text
#     "text.usetex":    True,
#     # default font family for text
#     "font.family":    "serif",
#     # ask LaTeX for Computer Modern Roman
#     "font.serif":     ["Computer Modern Roman"],
#     # match your desired size/weight
#     "font.size":      16,
#     "font.weight":    "normal",
#     # ensure math uses the same font
#     "text.latex.preamble": r"\usepackage{amsmath}"
# })

# if plot_ratio == False:
#     plt.plot(n_range, errors_for_A_I, marker='^', label='(i) $\mathbf{D} \mathbf{A}_1^{1/2}, \mathbf{A}_1^{1/2}$')
#     plt.plot(n_range, errors_for_A_I, marker='^', label='(ii) $\mathbf{A}, \mathbf{I}$')
#     plt.plot(n_range, errors_for_A_D_sqrt_inv, marker='s', label='(vi) $\mathbf{A} \mathbf{D}_{\mathrm{Toep}}^{-1/2}, \mathbf{D}_{\mathrm{Toep}}^{1/2}$')
#     plt.plot(n_range, errors_for_A_D_inv, marker='D', label='RMSE of (viii) $\mathbf{A} \mathbf{D}_{\mathrm{Toep}}^{-1}, \mathbf{D}_{\mathrm{Toep}}$')
#     plt.plot(n_range, errors_for_A_BandMF, label='Optimal')
#     plt.xlabel('Matrix Size')
#     plt.xscale('log')
#     plt.ylabel('$\mathrm{RMSE}(\mathbf{B},\mathbf{C})$')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("plots/efficient_error_vs_log_mat_size.pdf", format="pdf")
#     plt.show()
# else:
#     ratios_D_A1_sqrt = [errors_for_D_A1_sqrt[i] / errors_for_A_BandMF[i] for i in range(len(errors_for_D_A1_sqrt))]
#     ratios_AI = [errors_for_A_I[i] / errors_for_A_BandMF[i] for i in range(len(errors_for_A_I))]
#     ratios_D_toep = [errors_for_A_D_inv[i] / errors_for_A_BandMF[i] for i in range(len(errors_for_A_D_inv))]
#     ratios_D_toep_sqrt = [errors_for_A_D_sqrt_inv[i] / errors_for_A_BandMF[i] for i in range(len(errors_for_A_D_sqrt_inv))]
#     plt.plot(n_range,ratios_AI, marker='^', label='(i) $\mathbf{D} \mathbf{A}_1^{1/2}, \mathbf{A}_1^{1/2}$')
#     plt.plot(n_range,ratios_AI, marker='^', label='(ii) $\mathbf{A}, \mathbf{I}$')
#     plt.plot(n_range,ratios_D_toep_sqrt, marker='s' , label='(vi) $\mathbf{A} \mathbf{D}_{\mathrm{Toep}}^{-1/2}, \mathbf{D}_{\mathrm{Toep}}^{1/2}$')
#     plt.plot(n_range,ratios_D_toep, marker='D', label='(viii) $\mathbf{A} \mathbf{D}_{\mathrm{Toep}}^{-1}, \mathbf{D}_{\mathrm{Toep}}$')
#     plt.axhline(
#     y=1.0,
#     linestyle='--',
#     label='Optimal'
#     )
#     plt.xlabel('Matrix Size')
#     plt.xscale('log')
#     plt.ylabel('RMSE Ratio to The Optimal Value')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("plots/efficient_error_ratio_vs_log_mat_size.pdf", format="pdf")
#     plt.show()
