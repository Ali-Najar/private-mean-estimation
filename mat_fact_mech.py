import numpy as np
import matplotlib.pyplot as plt

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

def build_g_from_Ckind(n, p, C_kind):
    """Return g = first p coeffs of C^{-1} for the chosen C."""
    if C_kind == "Dtoep":
        c = seq_Dtoep(n)
    elif C_kind == "A1_sqrt":
        c = seq_A1_sqrt_rec(n)
    else:
        raise ValueError("C_kind must be {'Dtoep','A1_sqrt'}")
    g = inv_series(c, p)      # only need p coeffs
    return g

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


# ---- Algorithm 1 with A = running mean and banded C^{-1}
def continual_mean_banded_Cinv(
    X, g, sigma, xi=None, Z=None, seed=None, dtype=None
):
    """
    X: (n,d) stream; clipping to xi
    g: (p,) first column of C^{-1} (banded)
    sigma: noise std for z_t ~ N(0, sigma^2 I_d)
    Z: optional pre-drawn (n,d) noise; otherwise sampled with seed
    Returns: mu_hat with shape (n,d)
    """
    X = np.asarray(X, dtype=float if dtype is None else dtype)
    n, d = X.shape
    p = int(len(g))
    g = np.asarray(g, dtype=X.dtype)

    Zbuf = np.zeros((p, d), dtype=X.dtype)   # ring buffer for last p noise vecs
    run_sum = np.zeros(d, dtype=X.dtype)

    if Z is None:
        rng = np.random.default_rng(seed)
        draw = lambda: rng.normal(0.0, sigma, size=d).astype(X.dtype, copy=False)
    else:
        Z = np.asarray(Z, dtype=X.dtype)
        draw = None

    mu_hat = np.empty((n, d), dtype=X.dtype)

    for t in range(n):
        x = X[t]
        # if xi is not None:
        #     norm = np.linalg.norm(x)
        #     if norm > xi:
        #         x = (xi / norm) * x

        z_t = draw() if draw is not None else Z[t]
        idx = t % p
        Zbuf[idx] = z_t

        L = min(p, t + 1)
        rows = (idx - np.arange(L)) % p       # indices of z_t, z_{t-1}, ...
        noise_combo = g[:L] @ Zbuf[rows]      # (L,) @ (L,d) -> (d,)

        u_t = x + noise_combo
        run_sum += u_t
        mu_hat[t] = run_sum / (t + 1)

    return mu_hat

import os

EXP = 19
n = 2 ** EXP
k = 64
b = n // k
p = 16
eps = 1
delta = 1e-3 
xi = 1
mu = 0.5

# Two C_kinds
C_kinds = ["Dtoep", "A1_sqrt"]  
import time
import csv

runtimes = {kind: [] for kind in C_kinds}  # per-method list of per-seed runtimes (seconds)

def make_X_bernoulli(n, p=0.5, seed=None):
    rng = np.random.default_rng(seed)
    return rng.binomial(1, p, size=(n, 1)).astype(float)

def sigma_eps_delta(eps, delta):
    return np.sqrt(2.0 * np.log(1.25 / delta)) / eps

os.makedirs("cache/mat_fact_algo", exist_ok=True)

for C_kind in C_kinds:
    # Create folders: cache/mat_fact_alg/C_kind/mu{mu}
    save_dir = os.path.join("cache", "mat_fact_algo", C_kind, f"mu{mu}")
    os.makedirs(save_dir, exist_ok=True)

    # Loop over multiple seeds
    for SEED in range(50):
        print(f"\n=== Running {C_kind} with SEED={SEED} ===")

        ss = np.random.SeedSequence(SEED)
        s_X, s_noise = ss.spawn(2)

        # create a unique filename
        cache_name = (
            f"mu_hat_EXP{EXP}_k{k}_b{b}_p{p}"
            f"_eps{eps}_delta{delta}_xi{xi}_seed{SEED}.npy"
        )
        cache_path = os.path.join(save_dir, cache_name)

        if os.path.exists(cache_path):
            print(f"Loading cached result from {cache_path}")
            mu_hat = np.load(cache_path)
            # Do not record runtime for cached results
        else:
            print("Running computation...")
            t0 = time.perf_counter()

            g = build_g_from_Ckind(n=n, p=p, C_kind=C_kind)
            newC = inv_series(g, n)

            sensitivity = sens(newC, n, k, b)
            sigma = sigma_eps_delta(eps, delta) * xi * sensitivity

            X = make_X_bernoulli(n, mu, seed=s_X)
            mu_hat = continual_mean_banded_Cinv(X, g, sigma, seed=s_noise)

            dt = time.perf_counter() - t0
            runtimes[C_kind].append(dt)  # seconds for this seed/run

            np.save(cache_path, mu_hat)
            print(f"Saved result to {cache_path}  (runtime: {dt:.3f}s)")

# --- Write runtime summary CSV ---
os.makedirs("plots", exist_ok=True)

def fmt(v):
    return f"{v:.6g}" if v is not None else ""

summary_rows = []
for kind, times in runtimes.items():
    if len(times) == 0:
        mean_t = None
        std_t  = None
        n_used = 0
    else:
        arr = np.asarray(times, dtype=float)
        mean_t = float(arr.mean())
        std_t  = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        n_used = int(arr.size)

    summary_rows.append({
        "method": kind,
        "n_runs_used": n_used,
        "mean_runtime_sec": fmt(mean_t) if mean_t is not None else "",
        "std_runtime_sec": fmt(std_t) if std_t is not None else "",
        "EXP": EXP,
        "m": k,
        "k": k,
        "p": p,
        "eps": eps,
        "delta": delta,
        "xi": xi,
        "mu": mu,
    })

csv_name = (
    f"runtime_summary_EXP{EXP}_m{k}_k{k}_p{p}_"
    f"eps{eps}_delta{delta}_xi{xi}_mu{mu}.csv"
)
csv_path = os.path.join("cache", csv_name)

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "method", "n_runs_used", "mean_runtime_sec", "std_runtime_sec",
            "EXP", "m", "k", "p", "eps", "delta", "xi", "mu"
        ],
    )
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"\nSaved runtime summary to {csv_path}")
print("Note: cached runs are excluded from timing averages.")

# --- Plotting ---
abs_err = np.abs(mu_hat - mu).ravel()

plt.figure(figsize=(8, 4))
plt.plot(abs_err)
plt.xlabel("t")
plt.ylabel("|mu_hat - mu|")
plt.yscale('log')
plt.title("Absolute error of private running mean (d=1, Bernoulli)")
plt.tight_layout()
plt.show()