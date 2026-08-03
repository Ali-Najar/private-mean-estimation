import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import pandas as pd
import math 
from multiprocessing import Pool
from scipy.optimize import minimize


def prepare_events(df, lower_clip=1.0, upper_clip=1000.0):
    df2 = df.copy()
    # clip
    df2["amount"] = df2["amount"].clip(lower_clip, upper_clip)
    # sort
    df2 = df2.sort_values("event_time").reset_index(drop=True)
    # reindex user ids to small ints (0..U-1)
    unique_users = df2["user_id"].unique()
    id_map = {u: i for i, u in enumerate(unique_users)}
    df2["user_id_int"] = df2["user_id"].map(id_map)
    return df2[["user_id_int", "event_time", "amount"]].rename(columns={"user_id_int":"user_id"})


def enforce_b_min_separation(events: pd.DataFrame, b: int):
    assert {"user_id", "event_time", "amount"}.issubset(events.columns)
    events = events.sort_values("event_time").reset_index(drop=True)

    # per-user FIFO queues of buffered events (store rows)
    buffers = defaultdict(deque)

    last_idx = defaultdict(lambda: -10**18)

    released_rows = []
    curr_idx = 0  # index in the released stream

    def flush_eligible():
        nonlocal curr_idx
        while True:
            best_user = None
            best_time = None
            # find the earliest-time buffered head that is eligible now
            for u, q in buffers.items():
                if not q:
                    continue
                if curr_idx - last_idx[u] >= b:
                    head_time = q[0]["event_time"]
                    if best_time is None or head_time < best_time:
                        best_time = head_time
                        best_user = u
            if best_user is None:
                break
            row = buffers[best_user].popleft()
            released_rows.append(row)
            last_idx[best_user] = curr_idx
            curr_idx += 1

    for i, row in events.iterrows():
        u = int(row["user_id"])
        if curr_idx - last_idx[u] >= b:
            released_rows.append(row.to_dict())
            last_idx[u] = curr_idx
            curr_idx += 1

            flush_eligible()
        else:
            buffers[u].append(row.to_dict())

    flush_eligible()


    released = pd.DataFrame(released_rows)
    if not released.empty:
        released = released[["user_id", "event_time", "amount"]]
        released["event_time"] = pd.to_datetime(released["event_time"])
        released.sort_index(inplace=True)
    return released

def to_stream_matrix(released: pd.DataFrame, d=1):
    X = released["amount"].to_numpy(dtype=float).reshape(-1, d)
    return X




def inv_series(c, n):
    c = np.asarray(c, dtype=float)
    g = np.zeros(n, dtype=float)
    g[0] = 1.0 / c[0]
    L = len(c)
    for m in range(1, n):
        if m % 1000 == 0:
            print(m)
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

def build_g_from_Ckind(n, p, C_kind, k, b, bandmf_cache_file=None):
    """Return g = first p coeffs of C^{-1} for the chosen C."""

    cache_dir = "cache/g/"
    os.makedirs(cache_dir, exist_ok=True)
    if C_kind == "nu-FTRL":
        if k is None or b is None:
            raise ValueError("For C_kind='nu-FTRL' you must pass k=... and b=...")
        fname = os.path.join(cache_dir, f"g_{C_kind}_n{n}_k{k}_b{b}_p{p}.npy")
    else:
        fname = os.path.join(cache_dir, f"g_{C_kind}_n{n}_p{p}.npy")


    if os.path.isfile(fname):
        try:
            g = np.load(fname)
            return g
        except Exception:
            pass

    if C_kind == "Dtoep":
        c = seq_Dtoep(n)
        g = inv_series(c, p)      # only need p coeffs
    elif C_kind == "A1_sqrt":
        c = seq_A1_sqrt_rec(n)
        g = inv_series(c, p)      # only need p coeffs
    elif C_kind == "I":
        g = np.zeros(p, dtype=float)
        if p > 0:
            g[0] = 1.0
    elif C_kind == "nu-FTRL":
        g = compute_nu_FTRL_g(n, k, b, p)
    elif C_kind == "BandMF":
        if bandmf_cache_file is None:
            raise ValueError("For C_kind='BandMF' you must pass bandmf_cache_file=...")
        g = load_bandmf_cached_g(bandmf_cache_file, p)
        # g = compute_BandMF_g(n, p)
    else:
        raise ValueError("C_kind must be {'Dtoep','A1_sqrt','I','nu-FTRL','BandMF'}")
    
    tmp = fname + ".tmp.npy"
    np.save(tmp, g)
    os.replace(tmp, fname)   # now this works, because tmp really exists

    return g

def sens(c, n, k, b):
    
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

        if t % 1000 == 0:
            print("TIMESTEP: ", t)

    return mu_hat


###################### nu-FTRL #####################################


def compute_nu_FTRL_g(n, k, b, p, nus=None):
    
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

###################### nu-FTRL #####################################

import os

def make_X_bernoulli(n, p=0.5, seed=None):
        rng = np.random.default_rng(seed)
        return rng.binomial(1, p, size=(n, 1)).astype(float)

from scipy.stats import norm

def sigma_eps_delta(eps, delta):
    gaussian_delta_fn = lambda sigma, eps: norm.cdf(1/(2*sigma) - eps * sigma) - np.exp(eps) * norm.cdf(-1/(2 * sigma) - eps * sigma)
    low, high = 1e-3, 1000.0   # search interval for sigma
    tol = 1e-12
    while high - low > tol:
        mid = (low + high) / 2
        if gaussian_delta_fn(mid, eps) > delta:
            # need more noise
            low = mid
        else:
            # enough noise
            high = mid
    return high


def expected_mean_error_BandMF(coef, n: int) -> float:
    coef = np.asarray(coef, dtype=float)
    if coef.size < n:
        coef = np.pad(coef, (0, n - coef.size))
    else:
        coef = coef[:n]

    if coef[0] <= 0:
        return np.inf

    inv_coef = inv_series(coef, n)
    inv_coef_cum_sum_squared = np.cumsum(inv_coef) ** 2
    weights = np.cumsum((1.0 / np.arange(1, n + 1, dtype=float) ** 2)[::-1])[::-1]

    B_norm_squared = float(np.dot(inv_coef_cum_sum_squared, weights))
    sensitivity_squared = float(np.dot(coef, coef))

    val = sensitivity_squared * B_norm_squared
    return val if np.isfinite(val) else np.inf


def init_BandMF(n, p):
    return 1.0 / (np.arange(p, dtype=float) + 1.0)


def compute_BandMF_g(n, p, steps=25):
    
    x0 = init_BandMF(n, p)

    # keep c[0] positive so inverse series exists
    bounds = [(1e-8, None)] + [(None, None)] * (p - 1)

    res = minimize(
        lambda x: expected_mean_error_BandMF(x, n=n),
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": steps},
    )

    c_opt = res.x if res.success and np.isfinite(res.fun) else x0
    g = inv_series(c_opt, p)
    return g


def mat_fact():

    EXP = 19
    n = 2 ** EXP
    k = 128
    b = n // k
    p = 16
    eps = 1
    delta = 1e-6
    xi = 1
    mu = 0.5

    C_kinds = ["Dtoep", "A1_sqrt", "I", "nu-FTRL", "BandMF"]
    import time
    import csv

    runtimes = {kind: [] for kind in C_kinds}  # per-method list of per-seed runtimes (seconds)

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

            p_run = effective_p(C_kind, b, p)

            # create a unique filename
            cache_name = (
                f"mu_hat_EXP{EXP}_k{k}_b{b}_p{p_run}"
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


                g = build_g_from_Ckind(n=n, p=p_run, C_kind=C_kind, k=k, b=b, bandmf_cache_file=BANDMF_G_CACHE_FILE_SYNTH if C_kind == "BandMF" else None)
                newC = inv_series(g, n)

                sensitivity = sens(newC, n, k, b)
                sigma = sigma_eps_delta(eps, delta) * xi * sensitivity

                X = make_X_bernoulli(n, mu, seed=s_X)
                mu_hat = continual_mean_banded_Cinv(X, g, sigma, seed=s_noise)

                dt = time.perf_counter() - t0
                runtimes[C_kind].append(dt)  # seconds for this seed/run

                np.save(cache_path, mu_hat)
                print(f"Saved result to {cache_path}  (runtime: {dt:.3f}s)")

                # --- NEW: save cumulative sum of squared errors S_t = sum_{j<=t} (mu_j - muhat_j)^2
                true_rm = np.cumsum(X.reshape(-1)) / np.arange(1, n + 1)   # μ_t (noiseless running mean)
                priv_rm = mu_hat.reshape(-1)
                T = min(len(true_rm), len(priv_rm))
                err = true_rm[:T] - priv_rm[:T]
                sum_sqerr = np.cumsum(err**2)

                sumsq_name = (
                    f"sum_sqerr_EXP{EXP}_k{k}_b{b}_p{p}"
                    f"_eps{eps}_delta{delta}_xi{xi}_seed{SEED}.npy"
                )
                sumsq_path = os.path.join(save_dir, sumsq_name)
                np.save(sumsq_path, sum_sqerr)
                print(f"Saved cumulative squared-error series to {sumsq_path}")


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

        p_row = effective_p(kind, b, p)
        summary_rows.append({
            "method": kind,
            "n_runs_used": n_used,
            "mean_runtime_sec": fmt(mean_t) if mean_t is not None else "",
            "std_runtime_sec": fmt(std_t) if std_t is not None else "",
            "EXP": EXP,
            "m": k,
            "k": k,
            "p": p_row,
            "eps": eps,
            "delta": delta,
            "xi": xi,
            "mu": mu,
        })

    csv_name = (
        f"runtime_summary_EXP{EXP}_m{k}_k{k}_p{p_run}_"
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

BANDMF_G_CACHE_FILE = "cache/bandmf_coef_p_64_n_194116_inv_matrix_first_512_elements.npy"   # <-- change to your actual file
BANDMF_G_CACHE_FILE_SYNTH = "cache/bandmf_coef_p_64_n_2_19_inv_matrix_first_512_elements.npy"
BANDMF_G_LEN = 512


def effective_p(C_kind, b, p_default):
    if C_kind == "BandMF":
        return min(BANDMF_G_LEN, b)
    return p_default

def load_bandmf_cached_g(cache_file, p=None):
    g = np.load(cache_file).astype(float).ravel()
    if p is None:
        return g.copy()
    return g[:min(p, len(g))].copy()


import math
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Real-data-only preparation helpers.
# They preserve both the unclipped amount (for the real running mean) and the
# clipped amount (for the private mechanism), while enforcing b-separation once.
# ---------------------------------------------------------------------------

def prepare_credit_card_events(
    df_raw: pd.DataFrame,
    clip_lower: float = 0.0,
    clip_upper: float = 1000.0,
) -> pd.DataFrame:
    required = {"user_id", "event_time", "amount"}
    missing = required.difference(df_raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df_raw.loc[:, ["user_id", "event_time", "amount"]].copy()
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["user_id", "event_time", "amount"]).copy()

    # Stable tie-breaking makes the released stream reproducible even when
    # several transactions have the same timestamp.
    df["source_row"] = np.arange(len(df), dtype=np.int64)
    df = df.sort_values(
        ["event_time", "source_row"], kind="mergesort"
    ).reset_index(drop=True)

    unique_users = df["user_id"].drop_duplicates().tolist()
    id_map = {user_id: i for i, user_id in enumerate(unique_users)}
    df["user_id"] = df["user_id"].map(id_map).astype(np.int64)

    df["amount_true"] = df["amount"].astype(float)
    df["amount_private"] = df["amount_true"].clip(clip_lower, clip_upper)

    return df[
        [
            "user_id",
            "event_time",
            "source_row",
            "amount_true",
            "amount_private",
        ]
    ]


def enforce_b_min_separation_real(
    events: pd.DataFrame,
    b: int,
    max_releases: int,
) -> pd.DataFrame:
    """Release at most ``max_releases`` real events with b-separation.

    This function emits only genuine dataset events. If the scheduler reaches a
    point where no buffered user is eligible, it stops; the caller may extend
    the *published output stream* by carrying forward its last released value.
    Carry-forward rows are post-processing and are not new user contributions.
    """
    if b <= 0:
        raise ValueError("b must be a positive integer")
    if max_releases <= 0:
        raise ValueError("max_releases must be a positive integer")

    required = {
        "user_id",
        "event_time",
        "source_row",
        "amount_true",
        "amount_private",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    events = events.sort_values(
        ["event_time", "source_row"], kind="mergesort"
    ).reset_index(drop=True)

    buffers: dict[int, deque] = defaultdict(deque)
    last_idx: dict[int, int] = defaultdict(lambda: -10**18)
    released_rows: list[dict] = []
    curr_idx = 0

    def append_real(row: dict) -> None:
        nonlocal curr_idx
        released_rows.append(dict(row))
        last_idx[int(row["user_id"])] = curr_idx
        curr_idx += 1

    def flush_eligible() -> None:
        nonlocal curr_idx
        while curr_idx < max_releases:
            best_user = None
            best_key = None

            for user_id, queue in buffers.items():
                if not queue:
                    continue
                if curr_idx - last_idx[user_id] < b:
                    continue

                head = queue[0]
                key = (head["event_time"], head["source_row"])
                if best_key is None or key < best_key:
                    best_key = key
                    best_user = user_id

            if best_user is None:
                break

            append_real(buffers[best_user].popleft())

    for row in events.itertuples(index=False):
        if curr_idx >= max_releases:
            break

        row_dict = row._asdict()
        user_id = int(row_dict["user_id"])

        if curr_idx - last_idx[user_id] >= b:
            append_real(row_dict)
            flush_eligible()
        else:
            buffers[user_id].append(row_dict)

    if curr_idx < max_releases:
        flush_eligible()

    released = pd.DataFrame(released_rows)
    if released.empty:
        raise ValueError(f"No events were released for b={b}")

    return released.reset_index(drop=True)


def prepare_stream_for_b(
    df_raw: pd.DataFrame,
    b: int,
    n: int,
    clip_lower: float,
    clip_upper: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Prepare genuine released events, capped at the fixed horizon ``n``."""
    if n <= 0:
        raise ValueError("n must be a positive integer")

    events = prepare_credit_card_events(
        df_raw,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )
    released = enforce_b_min_separation_real(
        events,
        b=b,
        max_releases=n,
    )

    x_private = released["amount_private"].to_numpy(dtype=float).reshape(-1, 1)
    true_values = released["amount_true"].to_numpy(dtype=float)
    true_running_mean = np.cumsum(true_values) / np.arange(
        1, len(true_values) + 1, dtype=float
    )

    return released, x_private, true_running_mean


# ---------------------------------------------------------------------------
# Grid runner: multiple b values x multiple seeds x selected factorizations.
# ---------------------------------------------------------------------------

def _bandmf_cache_file_for_kind(c_kind: str):
    if c_kind != "BandMF":
        return None
    try:
        return BANDMF_G_CACHE_FILE
    except NameError as exc:
        raise NameError(
            "C_kind='BandMF' requires BANDMF_G_CACHE_FILE to be defined "
            "in the original code."
        ) from exc


def run_real_dataset_grid(
    df_clean: pd.DataFrame,
    b_values: Sequence[int],
    n: int,
    num_seeds: int,
    *,
    seed_start: int = 1,
    c_kinds: Sequence[str] = ("Dtoep", "A1_sqrt"),
    p: int = 16,
    eps: float = 10.0,
    delta: float = 5e-6,
    xi: float = 1000.0,
    clip_lower: float = 0.0,
    clip_upper: float = 1000.0,
    out_dir: str | os.PathLike = "cache/credit_card_b_sweep",
    prefix: str = "running_means",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Generate fixed-length result CSVs for every experiment combination.

    ``n`` is the common published horizon for every b value. The mechanism is
    calibrated using this same n. If fewer than n genuine events can be
    released under b-separation, the private and non-private running-mean
    outputs are extended to n by repeating their final released values. This
    extension is post-processing; it does not duplicate a user's transaction.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive")

    b_values = [int(b) for b in b_values]
    if not b_values or any(b <= 0 for b in b_values):
        raise ValueError("b_values must contain positive integers")

    seeds = list(range(int(seed_start), int(seed_start) + int(num_seeds)))
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []

    for b in b_values:
        print(f"\n===== Preparing released credit-card stream for b={b}, n={n} =====")
        released, x_private, true_running_mean_real = prepare_stream_for_b(
            df_clean,
            b=b,
            n=n,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
        )

        n_real = len(released)
        n_carry_forward = n - n_real
        k = math.ceil(n / b)
        print(
            f"b={b}: fixed_n={n}, real_releases={n_real}, "
            f"carry_forward={n_carry_forward}, k=ceil(n/b)={k}"
        )

        for c_kind in c_kinds:
            p_run = effective_p(c_kind, b, p)
            print(
                f"\n--- Precomputing mechanism for b={b}, "
                f"C_kind={c_kind}, p={p_run}, n={n} ---"
            )

            # Calibration uses the same fixed horizon n for every b.
            g = build_g_from_Ckind(
                n=n,
                p=p_run,
                C_kind=c_kind,
                k=k,
                b=b,
                bandmf_cache_file=_bandmf_cache_file_for_kind(c_kind),
            )
            c_first_col = inv_series(g, n)
            sensitivity = sens(c_first_col, n, k, b)
            sigma = sigma_eps_delta(eps, delta) * xi * sensitivity

            for seed in seeds:
                filename = (
                    f"{prefix}_b{b}_seed{seed}_C{c_kind}_n{n}_k{k}_p{p_run}_"
                    f"eps{eps:g}_delta{delta:g}_clipL{clip_lower:g}_"
                    f"clipU{clip_upper:g}_xi{xi:g}.csv"
                )
                csv_path = out_path / filename

                status = "created"
                runtime_sec = np.nan

                if csv_path.exists() and not overwrite:
                    print(f"Skipping existing file: {csv_path}")
                    status = "existing"
                else:
                    print(
                        f"Running b={b}, C_kind={c_kind}, seed={seed} "
                        f"(fixed n={n}, real releases={n_real})"
                    )
                    start = time.perf_counter()
                    mu_hat_real = continual_mean_banded_Cinv(
                        x_private,
                        g,
                        sigma,
                        xi=xi,
                        seed=seed,
                    ).reshape(-1)
                    runtime_sec = time.perf_counter() - start

                    if len(mu_hat_real) != n_real:
                        raise RuntimeError(
                            f"Expected {n_real} private means, got {len(mu_hat_real)}"
                        )

                    # Carry forward the last published running means. This is
                    # post-processing and therefore uses no extra user event and
                    # no additional noise draw.
                    if n_carry_forward > 0:
                        true_running_mean = np.pad(
                            true_running_mean_real,
                            (0, n_carry_forward),
                            mode="edge",
                        )
                        mu_hat = np.pad(
                            mu_hat_real,
                            (0, n_carry_forward),
                            mode="edge",
                        )
                    else:
                        true_running_mean = true_running_mean_real
                        mu_hat = mu_hat_real

                    if len(mu_hat) != n or len(true_running_mean) != n:
                        raise RuntimeError("Fixed-length output construction failed")

                    error = true_running_mean - mu_hat
                    squared_error = error**2
                    cumulative_squared_error = np.cumsum(squared_error)
                    rmse_through_t = np.sqrt(
                        cumulative_squared_error
                        / np.arange(1, n + 1, dtype=float)
                    )
                    is_carry_forward = np.arange(n, dtype=np.int64) >= n_real

                    result = pd.DataFrame(
                        {
                            "t": np.arange(1, n + 1, dtype=np.int64),
                            "is_carry_forward": is_carry_forward,
                            "true_running_mean": true_running_mean,
                            "private_running_mean": mu_hat,
                            "squared_error": squared_error,
                            "cumulative_squared_error": cumulative_squared_error,
                            "rmse_through_t": rmse_through_t,
                        }
                    )
                    result.to_csv(csv_path, index=False)
                    print(
                        f"Saved {csv_path} "
                        f"(runtime {runtime_sec:.3f} seconds)"
                    )

                manifest_rows.append(
                    {
                        "csv_path": str(csv_path.resolve()),
                        "status": status,
                        "b": b,
                        "seed": seed,
                        "C_kind": c_kind,
                        "n": n,
                        "n_released": n,
                        "n_real_released": n_real,
                        "n_carry_forward": n_carry_forward,
                        "k": k,
                        "p": p_run,
                        "eps": eps,
                        "delta": delta,
                        "xi": xi,
                        "clip_lower": clip_lower,
                        "clip_upper": clip_upper,
                        "sensitivity": sensitivity,
                        "sigma": sigma,
                        "runtime_sec": runtime_sec,
                    }
                )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_path / "run_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nSaved run manifest to {manifest_path}")
    return manifest


# ---------------------------------------------------------------------------
# Credit-card dataset loading and experiment configuration.
# ---------------------------------------------------------------------------

def load_credit_card_dataset(
    csv_path: str | os.PathLike = "credit_card_transactions.csv",
    max_rows: int | None = 200_000,
) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        usecols=["cc_num", "trans_date_trans_time", "amt"],
        dtype={"cc_num": "string"},
        parse_dates=["trans_date_trans_time"],
        nrows=max_rows,
    )

    df = df.rename(
        columns={
            "cc_num": "user_id",
            "trans_date_trans_time": "event_time",
            "amt": "amount",
        }
    )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df.dropna(subset=["user_id", "event_time", "amount"]).copy()


if __name__ == "__main__":
    # Edit only these experiment settings.
    CREDIT_CARD_CSV = "credit_card_transactions.csv"
    MAX_ROWS = 200_000
    N = 200_000  # common published horizon for every b value

    # B_VALUES = [25, 50, 100, 200, 300, 400, 500]
    B_VALUES = [750, 850, 900, 950]  # for the final figure in the paper
    NUM_SEEDS = 10                 # uses seeds 1, 2, ..., 10
    SEED_START = 1

    # Figure 4 compares these two mechanisms.
    C_KINDS = ["Dtoep", "A1_sqrt"]

    P = 16
    EPS = 10.0
    DELTA = 5e-6
    XI = 200.0
    CLIP_LOWER = 0.0
    CLIP_UPPER = 200.0
    OUTPUT_DIR = "cache/credit_card_b_sweep"

    df_credit_card = load_credit_card_dataset(
        CREDIT_CARD_CSV,
        max_rows=MAX_ROWS,
    )

    run_real_dataset_grid(
        df_credit_card,
        b_values=B_VALUES,
        n=N,
        num_seeds=NUM_SEEDS,
        seed_start=SEED_START,
        c_kinds=C_KINDS,
        p=P,
        eps=EPS,
        delta=DELTA,
        xi=XI,
        clip_lower=CLIP_LOWER,
        clip_upper=CLIP_UPPER,
        out_dir=OUTPUT_DIR,
        overwrite=False,
    )
