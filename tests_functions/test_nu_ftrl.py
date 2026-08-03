import numpy as np
from scipy.linalg import sqrtm, toeplitz
from multiprocessing import Pool

def compute_sensitivity(c, b, k):
    c_sum = np.zeros_like(c)
    for i in range(k):
        c_sum[b * i:] += c[:(len(c) - b * i)]
    sens = np.sqrt((c_sum ** 2).sum())
    return sens

def compute_square_root(x, n):
    y = np.zeros(n)
    y[0] = np.sqrt(x[0])
    for k in range(1, n):
        y[k] = (x[k] -np.dot(y[1:k], y[1:k][::-1])) / (2 * y[0])
    return y

def Toeplitz_inverse(r):
  n = len(r)
  y = np.zeros(n)
  y[0] = 1 / r[0]
  for i in range(1, n):
    y[i] = -(y[:i] * r[i:0:-1]).sum() / r[0]
  return y

def Toeplitz_product(s1, s2):
  return np.convolve(s1, s2)[:len(s1)]

def _compute_error_fixed_nu_FTRL(args):
  nu, b, k, n, p = args
  c = compute_square_root(np.ones(n), n) * nu ** np.arange(n)
  c_inv = Toeplitz_inverse(c)
  c_inv  = np.array(list(c_inv[:p]) + [0] * (n - p))
  c = Toeplitz_inverse(c_inv)
  sens = compute_sensitivity(c, b, k)
  b_arr = Toeplitz_product(np.ones(n), c_inv)
  w = np.cumsum((1 / np.arange(1, n + 1) ** 2)[::-1])[::-1]
  return c_inv, np.sqrt((b_arr ** 2 * w).sum() / n) * sens


def compute_nu_FTRL_inv_coef(n, k, b, p=None):
  if p is None:
    p = n
  nus = np.linspace(0, 1, 25)
  res = []
  with Pool(processes=8) as pool:
      res = pool.map(_compute_error_fixed_nu_FTRL, [(nu, b, k, n, p) for nu in nus])
  nu_opt = nus[np.argmin([r[1] for r in res])]
  c_inv, err = _compute_error_fixed_nu_FTRL((nu_opt, b, k, n, p))
  return c_inv



def inv_series(c, n):
    c = np.asarray(c, dtype=float)
    g = np.zeros(n, dtype=float)
    g[0] = 1.0 / c[0]
    L = len(c)
    for m in range(1, n):
        tmax = min(m, L - 1)
        s = 0.0
        for t in range(1, tmax + 1):
            s += c[t] * g[m - t]
        g[m] = -s / c[0]
    return g

def seq_Dtoep(n):
    i = np.arange(n, dtype=float)
    return 1.0 / (i + 1.0)

def seq_A1_sqrt_rec(n):
    a = np.empty(n, dtype=float)
    if n == 0: return a
    a[0] = 1.0
    for m in range(0, n-1):
        a[m+1] = a[m] * ((2*m + 1) * (2*m + 2) / (4.0 * (m + 1) * (m + 1)))
    return a

def sqrt_series(a, n):
    s = np.zeros(n, dtype=float); s[0] = np.sqrt(a[0])
    for m in range(1, n):
        conv = 0.0 if m < 2 else float(np.dot(s[1:m], s[m-1:0:-1]))
        a_m = a[m] if m < len(a) else 0.0
        s[m] = (a_m - conv) / (2.0 * s[0])
    return s


def sens(c, n, k, b):
    """
    Compute sens(C, k, b) for C = T(c) lower-triangular Toeplitz (first column c).
    Exactly equals: M = C[:, :k*b:b]; G = M.T @ M; sqrt(sum(G)) but avoids building M/G.

    Parameters
    ----------
    c : array-like
        First column of the Toeplitz matrix (length may be >= n or < n).
    n : int
        Matrix size (n x n). Only first n rows/cols considered.
    k : int
        Number of sampled columns (columns 0, b, 2b, ..., (k-1)b).
    b : int
        Column spacing.
    """
    c = np.asarray(c, dtype=float)
    P = min(len(c), n)  # available diagonal entries

    # Precompute prefix sums of products for lags L = 0, b, 2b, ..., (k-1)*b
    pref = {}
    for d in range(k):                  # d = 0..k-1
        L = d * b
        if L >= P:
            pref[L] = np.array([0.0])   # no overlap for this lag
            continue
        prod = c[: P - L] * c[L : P]   # elementwise products length = P-L
        p = np.empty(len(prod) + 1, dtype=float)
        p[0] = 0.0
        p[1:] = np.cumsum(prod)
        pref[L] = p                     # pref[L][t] = sum_{u=0..t-1} c[u]*c[u+L]

    total = 0.0
    # Sum over sampled column indices i,j (columns at offsets i*b and j*b)
    for i in range(k):
        si = i * b
        for j in range(k):
            sj = j * b
            L = abs(i - j) * b
            # valid rows for overlap start at max(si,sj) and go to n-1 -> count = n - max(si,sj)
            count = n - max(si, sj)
            if count <= 0:
                continue
            M = max(0, P - L)            # available products for this lag
            take = min(count, M)
            if take <= 0:
                continue
            total += pref[L][take]
    return float(np.sqrt(total))

def compute_nu_FTRL_g(n, k, b, p, nus=None):
    """
    Returns g (length p): first column of the *banded* C^{-1} used by nu-FTRL.

    Important: This is designed to be usable exactly like the g returned for Dtoep/A1_sqrt/I.
    """
    if nus is None:
        # avoid nu=1 exactly if you worry about numerical conditioning; 0.999 is typically safer
        nus = np.linspace(0.0, 0.999, 25)

    # This is sqrt(1/(1-z)) coefficients (same family as your A1_sqrt), O(n)
    a = seq_A1_sqrt_rec(n).astype(float)

    # weights from your objective
    w = np.cumsum((1.0 / (np.arange(1, n + 1, dtype=float) ** 2))[::-1])[::-1]
    idx = np.arange(n, dtype=float)

    best_err = np.inf
    best_g = None

    for nu in nus:
        # c = a_t * nu^t
        if nu == 0.0:
            pow_nu = np.zeros(n, dtype=float)
            pow_nu[0] = 1.0
        else:
            pow_nu = nu ** idx
        c = a * pow_nu

        # g_full would be length n; you only need first p
        g = inv_series(c, p)          # first p coeffs of C^{-1}
        c_band = inv_series(g, n)     # implied C when truncating C^{-1} beyond p as 0

        sensitivity = sens(c_band, n, k, b)

        # Toeplitz_product(ones, c_inv) with c_inv padded by zeros is just a cumulative sum
        g_pad = np.zeros(n, dtype=float)
        g_pad[:p] = g
        b_arr = np.cumsum(g_pad)

        err = np.sqrt(np.sum((b_arr ** 2) * w) / n) * sensitivity

        if err < best_err:
            best_err = err
            best_g = g

    return best_g

import time

# if __name__ == '__main__':

#     start = time.time()
#     aa = compute_nu_FTRL_g(2**16, 8, 2**16//8, 16)
#     end = time.time()
#     print(end - start)
#     start = time.time()
#     bb = compute_nu_FTRL_inv_coef(2**16, 8, 2**16//8, 16)
#     end = time.time()
#     print(end - start)
    # for i in range(100):
        # if abs(aa[i] - bb[i]) < 1e-4:
        #     print(i, "Pass", aa[i], bb[i])
        # else:
        #     print(i, "Fail", aa[i], bb[i])
    