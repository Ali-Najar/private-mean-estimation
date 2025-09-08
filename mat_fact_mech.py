import numpy as np

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
        if xi is not None:
            norm = np.linalg.norm(x)
            if norm > xi:
                x = (xi / norm) * x

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
