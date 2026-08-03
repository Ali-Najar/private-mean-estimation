import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from pathlib import Path
import scipy.linalg as la
from tqdm import tqdm
from scipy.optimize import minimize

# ------------------------ cache path helpers ------------------------

CACHE_ROOT = Path("cache")

def _fmt_value(v):
    if isinstance(v, float):
        s = f"{v:.12g}"
    else:
        s = str(v)
    return (
        s.replace(" ", "")
         .replace("/", "_per_")
         .replace("\\", "_")
         .replace(":", "_")
    )

def setting_dir(n, kind, b, p):
    """
    Returns: cache/<n>/<kind>/<b>/<p> (creates dirs if needed)
    kind ∈ {"nonbanded", "banded", "banded inverse"}
    """
    d = CACHE_ROOT / str(n) / kind / _fmt_value(b) / _fmt_value(p)
    d.mkdir(parents=True, exist_ok=True)
    return d

def _val_path(n, kind, b, p, name):
    return setting_dir(n, kind, b, p) / f"{name}.pkl"

def save_value(value, *, n, kind, b, p, name):
    path = _val_path(n, kind, b, p, name)
    with open(path, "wb") as f:
        pickle.dump(float(value), f, protocol=pickle.HIGHEST_PROTOCOL)
    return path

def load_value(*, n, kind, b, p, name):
    path = _val_path(n, kind, b, p, name)
    with open(path, "rb") as f:
        return float(pickle.load(f))

def has_value(*, n, kind, b, p, name):
    return _val_path(n, kind, b, p, name).exists()

# ------------------------ Main Code ------------------------


import numpy as np
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
  nu, b, k, n = args
  c = compute_square_root(np.ones(n), n) * nu ** np.arange(n)
  sens = compute_sensitivity(c, b, k)
  c_inv = Toeplitz_inverse(c)
  b_arr = Toeplitz_product(np.ones(n), c_inv)
  w = np.cumsum((1 / np.arange(1, n + 1) ** 2)[::-1])[::-1]
  return np.sqrt((b_arr ** 2 * w).sum() / n) * sens
 
 
def compute_error_nu_FTRL(n, k, b):
  nus = np.linspace(0, 1, 25)
  res = []
  with Pool(processes=8) as p:
      res = p.map(_compute_error_fixed_nu_FTRL, [(nu, b, k, n) for nu in nus])
  return min(res)


def seq_A1(n):           # first column of A1
    return np.ones(n, dtype=float)

def seq_Dtoep(n):        # first column of D_toep: 1, 1/2, 1/3, ...
    i = np.arange(n, dtype=float)
    return 1.0 / (i + 1.0)

def seq_A1_sqrt_rec(n: int) -> np.ndarray:
    a = np.empty(n, dtype=float)
    if n == 0:
        return a
    a[0] = 1.0
    for m in range(0, n-1):
        num = (2*m + 1) * (2*m + 2)
        den = 4.0 * (m + 1) * (m + 1)
        a[m+1] = a[m] * (num / den)
    return a

def band_seq(c, p):
    return c[:min(p, len(c))].copy()

def inv_series(c, n):
    """Recurrence for series inverse: g[0]=1/c0; for m>=1 g[m] = -1/c0 * sum_{t=1..min(m,L-1)} c[t] * g[m-t]."""
    g = np.zeros(n, dtype=float)
    c0 = c[0]
    g[0] = 1.0 / c0
    L = len(c)
    for m in range(1, n):
        tmax = min(m, L - 1)
        s = 0.0
        for t in range(1, tmax + 1):
            s += c[t] * g[m - t]
        g[m] = - s / c0
    return g

def sqrt_series(a, n):
    """First column s with T(s)^2 = T(a) up to length n."""
    s = np.zeros(n, dtype=float)
    s[0] = np.sqrt(a[0])
    for m in range(1, n):
        if m >= 2:
            conv = float(np.dot(s[1:m], s[m-1:0:-1]))  # sum_{t=1..m-1} s_t s_{m-t}
        else:
            conv = 0.0
        a_m = a[m] if m < len(a) else 0.0
        s[m] = (a_m - conv) / (2.0 * s[0])
    return s

# ---------- Frobenius for B = D · T(h) ----------
def frob_D_T(A, p):
    """
    A: lower-triangular Toeplitz (first column)
    D: diagonal matrix as 1D array
    p: bandwidth (number of subdiagonals)
    """
    n = len(A)
    D = 1/np.arange(1, n+1)
    h = A  # first column of LTT
    total = 0.0
    for j in range(n):  # column index
        # sum over the allowed band entries
        # row index i >= j, but i-j < p => i in [j, min(j+p-1,n-1)]
        i_max = min(j + p, n)
        col_sum = np.sum((D[j:i_max] * h[:i_max-j])**2)
        total += col_sum
    return np.sqrt(total)

# Special case A = D·A1 => h = ones  ->  ||A||_F^2 = H_n (harmonic number)
def frob_A(n):
    return np.sqrt(np.sum(1.0 / (np.arange(1, n+1, dtype=float))))


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

# --------- A · T(g) has Toeplitz part T(cumsum(g)) ----------
def frob_A_times_T(g, n):
    h = np.zeros(n)
    h[:len(g)] = g
    return frob_D_T(np.cumsum(h)[:n], n)

# ---------- High-level “obj” builders matching your four/banded cases ----------
def obj_nonbanded_case1(n, k, b):
    # B = D · A1^{1/2}  (Toeplitz h = seq_A1_sqrt),  C = A1^{1/2}
    c = seq_A1_sqrt_rec(n)
    frobB = frob_D_T(c, n)
    sensC = sens(c, n, k, b)
    return (frobB * sensC) / np.sqrt(n)

def obj_nonbanded_case2(n, k, b):
    # B = A = D·A1,  C = I
    frobB = frob_A(n)            # exact, cheap
    sensC = np.sqrt(k)           # sum of k basis columns
    return (frobB * sensC) / np.sqrt(n)

def obj_nonbanded_case3(n, k, b):
    # B = A · (D_toep^{1/2})^{-1},  C = D_toep^{1/2}
    a  = seq_Dtoep(n)
    s  = sqrt_series(a, n)       # first col of sqrt(D_toep)
    is_ = inv_series(s, n)       # first col of inv(sqrt(D_toep))
    frobB = frob_A_times_T(is_, n)
    sensC = sens(s, n, k, b)
    return (frobB * sensC) / np.sqrt(n)

def obj_nonbanded_case4(n, k, b):
    # B = A · (D_toep)^{-1},  C = D_toep
    c  = seq_Dtoep(n)
    invc = inv_series(c, n)
    frobB = frob_A_times_T(invc, n)
    sensC = sens(c, n, k, b)
    return (frobB * sensC) / np.sqrt(n)



def obj_bsr(n, k, b, p, C_kind):
    # C_p = banded(C, p); B = A · C_p^{-1}; C = C_p
    if C_kind == "Dtoep":
        c = seq_Dtoep(n)
    elif C_kind == "Dtoep_sqrt":
        c = sqrt_series(seq_Dtoep(n), n)
    elif C_kind == "A1_sqrt":
        c = seq_A1_sqrt_rec(n)
    else:
        raise ValueError("C_kind must be one of {'Dtoep','Dtoep_sqrt','A1_sqrt'}")
    cb   = band_seq(c, p)
    invb = inv_series(cb, n)
    frobB = frob_A_times_T(invb, n)
    sensC = sens(cb, n, k, b)
    return (frobB * sensC) / np.sqrt(n)

def obj_bisr(n, k, b, p, C_kind):
    # C_inv = inv(C); C_p_inv = banded(C_inv, p); new_C = inv(C_p_inv)
    if C_kind == "Dtoep":
        c = seq_Dtoep(n)
    elif C_kind == "Dtoep_sqrt":
        c = sqrt_series(seq_Dtoep(n), n)
    elif C_kind == "A1_sqrt":
        c = seq_A1_sqrt_rec(n)
    else:
        raise ValueError("C_kind must be one of {'Dtoep','Dtoep_sqrt','A1_sqrt'}")
    invc = inv_series(c, n)
    invc_p = band_seq(invc, p)
    newC = inv_series(invc_p, n)          # first column of (C_p_inv)^{-1}
    frobB = frob_A_times_T(invc_p, n)  # B = A·C_p_inv
    sensC = sens(newC, n, k, b)
    return (frobB * sensC) / np.sqrt(n)


def p_for_Ckind(n: int, C_kind: str, b) -> int:
    """
    Bandwidth policy p(n) depending on which C we use.
    """
    if n <= 1:
        return 1

    if C_kind == "A1_sqrt":
        # p = int(np.ceil(np.log2(n)))                      # ceil(log2 n)
        p = int(np.ceil(np.log2(b)))                      # ceil(log2 n)
        # p = max(1, int(np.log2(b)))
    elif C_kind == "Dtoep":
        # p = int(np.ceil(np.exp(np.sqrt(np.log(n)))))      # ceil(exp(sqrt(ln n)))
        p = b      # ceil(exp(sqrt(ln n)))
    elif C_kind == "Dtoep_sqrt":
        # Assumption: use the same p(n) as Dtoep (change if you want different behavior)
        p = int(np.ceil(np.exp(np.sqrt(np.log(n)))))
    else:
        raise ValueError(f"Unknown C_kind: {C_kind}")

    return max(1, min(p, n))  # keep it in [1, n]

def expected_mean_error_BandMF(coef, n: int) -> float:
    coef = np.asarray(coef, dtype=float)
    if coef.size < n:
        coef = np.pad(coef, (0, n - coef.size))
    else:
        coef = coef[:n]

    inv_coef = inv_series(coef, n)
    inv_coef_cum_sum_squared = np.cumsum(inv_coef) ** 2
    weights = np.cumsum((1.0 / np.arange(1, n + 1, dtype=float) ** 2)[::-1])[::-1]

    B_norm_squared = float(np.dot(inv_coef_cum_sum_squared, weights))
    sensitivity_squared = float(np.dot(coef, coef))

    return sensitivity_squared * B_norm_squared


def init_BandMF(n, p):
    return 1.0 / (np.arange(p, dtype=float) + 1.0)


def _optimize_banded_toeplitz_BandMF(n, p, steps=10):
    x0 = init_BandMF(n, p)

    # keep the first entry positive so the inverse series is well-defined
    bounds = [(1e-12, None)] + [(None, None)] * (p - 1)

    res = minimize(
        lambda x: expected_mean_error_BandMF(x, n=n),
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": steps},
    )
    return res.x if res.success else x0


def compute_error_BandMF(n, k, p, steps=10):
    C_opt = _optimize_banded_toeplitz_BandMF(n=n, p=p, steps=steps)
    single_part = np.sqrt(expected_mean_error_BandMF(C_opt, n=n)) / np.sqrt(n)
    return np.sqrt(k) * single_part

BANDMF_CACHE_FILE = Path("cache/efficient_banded_cache.npz")
def load_bandmf_from_legacy_cache(n, k):
    if not BANDMF_CACHE_FILE.exists():
        return None
    data = np.load(BANDMF_CACHE_FILE)
    arr = data["errors_for_A_BandMF"]

    idx = int(np.log2(n))
    if idx < 0 or idx >= len(arr):
        return None

    return np.sqrt(k) * float(arr[idx])

if __name__ == '__main__':
    # ------------------------ experiment grid ------------------------

    EXPS = 15
    k_values = [4,16,64]
    exponents = np.arange(0, EXPS + 1)   # 2^0 ... 2^12 = 4096
    n_range   = 2**exponents
    # n_range_aof = 2**np.arange(0, EXPS_AOF + 1)   # 2^0 ... 2^9 = 512
    number_of_plots = 4
    number_of_plots_banded = 3
    number_of_plots_nuftrl = 1
    number_of_plots_bandmf = 1
    set_p_eq_b = True
    set_p_eq_logb = True
    p = 1

    Path("plots").mkdir(parents=True, exist_ok=True)

    # # -- Caching setup ---------------------------------------------------------
    # if os.path.exists(banded_cache_path):
    #     with open(banded_cache_path, 'rb') as f:
    #         res_banded = pickle.load(f)
    #     print("Loaded cached banded results.")
    # else:
    #     res_banded = {k: {"bisr": {i: {} for i in range(1, number_of_plots_banded + 1)}, "bsr": {i: {} for i in range(1, number_of_plots_banded + 1)}} for k in k_values}
        
    #     print("Initialized empty banded cache.")
        
    # if os.path.exists(cache_path):
    #     with open(cache_path, 'rb') as f:
    #         res = pickle.load(f)
    #     print("Loaded cached results.")
    # else:
    #     res = {k: {i: {} for i in range(1, number_of_plots + 1)} for k in k_values}
    #     print("Initialized empty cache.")

    def n_is_complete(n: int) -> bool:
        for k in k_values:
            if k > n:
                continue
            b = n // k

            # nonbanded: store with a fixed p (since p is irrelevant mathematically here)
            p_nb = n
            for i in range(1, number_of_plots + 1):
                if not has_value(n=n, kind="nonbanded", b=b, p=p_nb, name=f"i{i}"):
                    return False

            # banded inverse (BISR)
            bisr_map = {1: "Dtoep_sqrt", 2: "Dtoep", 3: "A1_sqrt"}
            for i in range(1, number_of_plots_banded + 1):
                C_kind = bisr_map[i]
                p = p_for_Ckind(n, C_kind, b)
                if not has_value(n=n, kind="banded inverse", b=b, p=p, name=f"bisr_i{i}"):
                    return False

            # banded (BSR)
            bsr_map = {1: "Dtoep_sqrt", 2: "Dtoep", 3: "A1_sqrt"}
            for i in range(1, number_of_plots_banded + 1):
                C_kind = bsr_map[i]
                p = p_for_Ckind(n, C_kind, b)
                if not has_value(n=n, kind="banded", b=b, p=p, name=f"bsr_i{i}"):
                    return False

            # nuftrl uses p=n
            for i in range(1, number_of_plots_nuftrl + 1):
                if not has_value(n=n, kind="nuftrl", b=b, p=n, name=f"nuftrl_i{i}"):
                    return False
                
            # BandMF
            for i in range(1, number_of_plots_bandmf + 1):
                if not has_value(n=n, kind="bandmf", b=b, p=n, name=f"bandmf_i{i}"):
                    return False

        return True


    tasks_per_pair = 11

    for n in n_range:

        if n_is_complete(n):
            tqdm.write(f"Completed n={n} (from cache)")
            continue

        eligible_k = [kk for kk in k_values if kk <= n]
        for k in eligible_k:

            b = n // k
            
            def check(kind, name, p_task):
                return has_value(n=n, kind=kind, b=b, p=p_task, name=name)

            task_ids = [
                ("nonbanded", "i1"),
                ("nonbanded", "i2"),
                ("nonbanded", "i3"),
                ("nonbanded", "i4"),
                ("banded inverse", "bisr_i1"),
                ("banded inverse", "bisr_i2"),
                ("banded inverse", "bisr_i3"),
                ("banded", "bsr_i1"),
                ("banded", "bsr_i2"),
                ("banded", "bsr_i3"),
            ]

            # --- nonbanded (i=1..4) ---
            # # (i) D @ A1^{1/2}, A1^{1/2}
            p_nb = n
            # if not check("nonbanded", "i1", p_nb):
            #     val = obj_nonbanded_case1(n, k, b)
            #     save_value(val, n=n, kind="nonbanded", b=b, p=p_nb, name="i1")

            # # (ii) A, I
            # if not check("nonbanded", "i2", p_nb):
            #     val = obj_nonbanded_case2(n, k, b)
            #     save_value(val, n=n, kind="nonbanded", b=b, p=p_nb, name="i2")

            # # (vi) A @ D_toep^{-1/2}, D_toep^{1/2}
            # if not check("nonbanded", "i3", p_nb):
            #     val = obj_nonbanded_case3(n, k, b)
            #     save_value(val, n=n, kind="nonbanded", b=b, p=p_nb, name="i3")

            # (viii) A @ D_toep^{-1}, D_toep
            print("p_nb", p_nb)
            if not check("nonbanded", "i4", p_nb):
                val = obj_nonbanded_case4(n, k, b)
                save_value(val, n=n, kind="nonbanded", b=b, p=p_nb, name="i4")

            # --- banded inverse (BISR, i=1..3) ---
            # p_bisr1 = p_for_Ckind(n, "Dtoep_sqrt")
            # if not check("banded inverse", "bisr_i1", p_bisr1):
            #     val = obj_bisr(n, k, b, p_bisr1, "Dtoep_sqrt")
            #     save_value(val, n=n, kind="banded inverse", b=b, p=p_bisr1, name="bisr_i1")

            p_bisr2 = p_for_Ckind(n, "Dtoep", b)
            print("p_bisr2", p_bisr2)
            if not check("banded inverse", "bisr_i2", p_bisr2):
                val = obj_bisr(n, k, b, p_bisr2, "Dtoep")
                save_value(val, n=n, kind="banded inverse", b=b, p=p_bisr2, name="bisr_i2")


            p_bisr3 = p_for_Ckind(n, "A1_sqrt", b)
            print("p_bisr3", p_bisr3)
            if not check("banded inverse", "bisr_i3", p_bisr3):
                val = obj_bisr(n, k, b, p_bisr3, "A1_sqrt")
                save_value(val, n=n, kind="banded inverse", b=b, p=p_bisr3, name="bisr_i3")

            # --- banded (BSR, i=1..3) ---
            # p_bsr1 = p_for_Ckind(n, "Dtoep_sqrt")
            # if not check("banded", "bsr_i1", p_bsr1):
            #     val = obj_bsr(n, k, b, p_bsr1, "Dtoep_sqrt")
            #     save_value(val, n=n, kind="banded", b=b, p=p_bsr1, name="bsr_i1")

            p_bsr2 = p_for_Ckind(n, "Dtoep", b)
            print("p_bsr2", p_bsr2)
            if not check("banded", "bsr_i2", p_bsr2):
                val = obj_bsr(n, k, b, p_bsr2, "Dtoep")
                save_value(val, n=n, kind="banded", b=b, p=p_bsr2, name="bsr_i2")

            # p_bsr3 = p_for_Ckind(n, "A1_sqrt")
            # if not check("banded", "bsr_i3", p_bsr3):
            #     val = obj_bsr(n, k, b, p_bsr3, "A1_sqrt")
            #     save_value(val, n=n, kind="banded", b=b, p=p_bsr3, name="bsr_i3")
            
            if not check("nuftrl", "nuftrl_i1", n):
                val = compute_error_nu_FTRL(n, k, b)
                save_value(val, n=n, kind="nuftrl", b=b, p=n, name="nuftrl_i1")

            p_bandmf = min(b, 64)
            print("p_bandmf", p_bandmf)
            if not check("bandmf", "bandmf_i1", p_bandmf):
                # legacy_val = load_bandmf_from_legacy_cache(n, k)
                # if legacy_val is not None:
                #     val = legacy_val
                # else:
                steps_bandmf = 50 if int(np.log2(n)) < 10 else (20 if int(np.log2(n)) < 14 else 10)
                val = compute_error_BandMF(n, k, p_bandmf, steps=steps_bandmf)

                save_value(val, n=n, kind="bandmf", b=b, p=p_bandmf, name="bandmf_i1")



    def _series_name(kind: str, i: int) -> str:
        if kind == "nonbanded":
            return f"i{i}"
        if kind == "banded":
            return f"bsr_i{i}"
        if kind == "banded inverse":
            return f"bisr_i{i}"
        if kind == "nuftrl":
            return f"nuftrl_i{i}"
        if kind == "bandmf":
            return f"bandmf_i{i}"
        raise ValueError(f"unknown kind: {kind}")

    def _p_from_b(b: int) -> int:
        if set_p_eq_b:
            return b
        if set_p_eq_logb:
            return max(1, int(np.log2(b)))
        return p

    def load_curve(kind: str, i: int, k: int, n_values) -> dict[int, float]:
        name = _series_name(kind, i)
        out = {}
        for n in n_values:
            if k > n:
                continue
            b = n // k
            if kind == "nonbanded":
                p = n
            elif kind == "nuftrl":
                p = n
            elif kind == "bandmf":
                p = min(b, 64)
            elif kind in ("banded", "banded inverse"):
                C_kind = {1: "Dtoep_sqrt", 2: "Dtoep", 3: "A1_sqrt"}[i]
                p = p_for_Ckind(n, C_kind, b)
            else:
                raise ValueError(f"unknown kind: {kind}")
            if has_value(n=n, kind=kind, b=b, p=p, name=name):
                out[n] = load_value(n=n, kind=kind, b=b, p=p, name=name)
            else:
                print(n, kind, b, p, name)
        return out


    for k in k_values:
        # Styling (keep LaTeX if you have it installed; otherwise set "text.usetex": False)
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({
            "text.usetex":    True,
            "font.family":    "serif",
            "font.serif":     ["Computer Modern Roman"],
            "font.size":      16,
            "axes.labelsize": 40,
            "xtick.labelsize": 40,
            "ytick.labelsize": 40,
            "font.weight":    "normal",
            "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}"
        })

        labels = [
            "$\\mathbf{E}_{1}^{1/2}$",
            "$\\mathbf{I}$",
            "(vi) $\\mathbf{A} \\mathbf{D}_{\\mathrm{Toep}}^{-1/2}, \\mathbf{D}_{\\mathrm{Toep}}^{1/2}$",
            "$\\mathbf{D}_{\\mathrm{Toep}}$",
        ]
        labels_bsr = [
            "$\\mathbf{D}_{\\mathrm{Toep}}^{1/2}$ (Banded)",
            "$\\mathbf{D}_{\\mathrm{Toep}}$ (Banded)",
            "$\\mathbf{E}_{1}^{1/2}$ (Banded)",
        ]
        labels_bisr = [
            "$\\mathbf{D}_{\\mathrm{Toep}}^{1/2}$ (Banded Inverse)",
            "$\\mathbf{D}_{\\mathrm{Toep}}$ (Banded Inverse)",
            "$\\mathbf{E}_{1}^{1/2}$ (Banded Inverse)",
        ]
        labels_nuftrl = [
            "$\\mathbf{E}_{\\nu}^{1/2}$",
        ]
        labels_bandmf = [
            "BandMF",
        ]

        markers      = [None, 'o', 's', '>', 'D']         # 1..4
        markers_bsr  = [None, 'd', 'x', '>']              # 1..3
        markers_bisr = [None, '^', 'o', '^']              # 1..3
        markers_nuftrl = [None, '<']                       # 1
        markers_bandmf = [None, '']                      # 1

        colors      = [None, 'tab:cyan', 'tab:orange', 'tab:green', 'tab:red']
        colors_bsr  = [None, 'tab:olive', 'tab:brown', 'purple']
        colors_bisr = [None, 'tab:purple', 'tab:blue', 'tab:orange']
        colors_nuftrl = [None, 'tab:purple']
        colors_bandmf = [None, 'tab:pink']

        plot_log_scale = True
        plot_ratio = True

        # Denominator for ratios: BSR i=2 (Dtoep)
        denom = load_curve("bandmf", 1, k, n_range)
        # denom = load_curve("banded", 2, k, n_range)
        denom_ns = set(denom.keys())
        # --- nonbanded (only i=4, as in your original loop) ---
        i = 4
        nonband_4 = load_curve("nonbanded", i, k, n_range)
        ns = sorted(set(nonband_4.keys()) & denom_ns) if plot_ratio else sorted(nonband_4.keys())
        if ns:
            ys = [(nonband_4[n] / denom[n]) for n in ns] if plot_ratio else [nonband_4[n] for n in ns]
            plt.plot(ns, ys, marker=markers[i], color=colors[i], label=labels[i-1], markersize=10)

        # --- banded inverse (BISR) i=1,2 (skip 3 to match your original) ---
        for i in (2, 3):
            bisr_i = load_curve("banded inverse", i, k, n_range)
            ns = sorted(set(bisr_i.keys()) & denom_ns) if plot_ratio else sorted(bisr_i.keys())
            if not ns:
                continue
            ys = [(bisr_i[n] / denom[n]) for n in ns] if plot_ratio else [bisr_i[n] for n in ns]
            plt.plot(ns, ys, marker=markers_bisr[i], color=colors_bisr[i], label=labels_bisr[i-1], markersize=10)

        # --- banded (BSR) i=1,2 (skip 3) ---
        for i in (2, 3):
            if i == 3:
                continue
            bsr_i = load_curve("banded", i, k, n_range)
            ns = sorted(set(bsr_i.keys()) & denom_ns) if plot_ratio else sorted(bsr_i.keys())
            if not ns:
                continue
            ys = [(bsr_i[n] / denom[n]) for n in ns] if plot_ratio else [bsr_i[n] for n in ns]
            plt.plot(ns, ys, marker=markers_bsr[i], color=colors_bsr[i], label=labels_bsr[i-1], markersize=10)
        
        # --- NuFTRL ---
        i = 1
        nuftrl_i = load_curve("nuftrl", i, k, n_range)
        ns = sorted(set(nuftrl_i.keys()) & denom_ns) if plot_ratio else sorted(nuftrl_i.keys())
        if ns:
            ys = [(nuftrl_i[n] / denom[n]) for n in ns] if plot_ratio else [nuftrl_i[n] for n in ns]
            plt.plot(ns, ys, marker=markers_nuftrl[i], color=colors_nuftrl[i], label=labels_nuftrl[i-1], markersize=10)

        # --- BandMF ---
        i = 1
        bandmf_i = load_curve("bandmf", i, k, n_range)
        ns = sorted(set(bandmf_i.keys()) & denom_ns) if plot_ratio else sorted(bandmf_i.keys())
        if ns:
            ys = [(bandmf_i[n] / denom[n]) for n in ns] if plot_ratio else [bandmf_i[n] for n in ns]
            # plt.plot(ns, ys, marker=markers_bandmf[i], color=colors_bandmf[i], label=labels_bandmf[i-1], markersize=10)
            plt.plot(
                ns,
                ys,
                color='tab:blue',
                linestyle='--',
                linewidth=2.5,
                label=r'$\mathrm{BandMF}$',
            )

        plt.xlabel('Matrix Size')
        if plot_log_scale:
            plt.xscale('log')
        plt.ylabel('$\\mathcal{E}(\\mathbf{B},\\mathbf{C})$')
        plt.legend(loc = "upper right")
        plt.tight_layout()

        if set_p_eq_b:
            p_name = "p=b"
        elif set_p_eq_logb:
            p_name = "p=logb"
        else:
            p_name = f"p={p}"
        savefile_name = f"_multi_error_vs_mat_size{2**EXPS}_k{k}_ratio{plot_ratio}.pdf"
        savefile_name = os.path.join("plots", savefile_name)
        plt.savefig(savefile_name, format="pdf")
        plt.close()