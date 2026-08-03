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
# UCI real-data experiment with online per-user buffering.
#
# Stream semantics (one published output per raw contribution):
#   1. If the arriving user's last Matrix-Mechanism participation was within
#      the previous b-1 *mechanism updates*, append the contribution to that
#      user's buffer.
#   2. Then, if some buffered user is eligible, release exactly one user: take
#      the average of every contribution currently in that user's buffer,
#      clear the buffer, and advance the Matrix Mechanism by one step.
#   3. If no buffered user is eligible, publish the previous private mean and
#      do not advance the Matrix Mechanism.
#   4. If the arriving user is already eligible, combine the new contribution
#      with that user's existing buffer, release their average immediately,
#      clear the buffer, and advance the Matrix Mechanism by one step.
#
# Thus every released buffer-average is one Matrix-Mechanism datapoint, and a
# user can appear only once in any b consecutive Matrix-Mechanism updates.
# ---------------------------------------------------------------------------

import heapq
from dataclasses import dataclass


@dataclass
class _UserBuffer:
    count: int = 0
    sum_true: float = 0.0
    sum_private: float = 0.0
    first_arrival: int = -1
    version: int = 0


def load_uci_dataset(
    csv_path: str | os.PathLike = "uci_data.csv",
    max_rows: int | None = 200_000,
) -> pd.DataFrame:
    """Load the supplied UCI stream in its existing row order.

    The supplied file has columns ``user_id`` and ``total_price`` and no time
    column, so the CSV row order is treated as the contribution arrival order.
    """
    df = pd.read_csv(
        csv_path,
        usecols=["user_id", "total_price"],
        nrows=max_rows,
    )
    df["total_price"] = pd.to_numeric(df["total_price"], errors="coerce")
    df = df.dropna(subset=["user_id", "total_price"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("The UCI dataset contains no valid rows.")
    return df


def prepare_uci_events(
    df_raw: pd.DataFrame,
    clip_lower: float = -200.0,
    clip_upper: float = 200.0,
) -> pd.DataFrame:
    """Create the ordered real/private event stream used by the experiment."""
    required = {"user_id", "total_price"}
    missing = required.difference(df_raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if clip_lower > clip_upper:
        raise ValueError("clip_lower must be <= clip_upper")

    df = df_raw.loc[:, ["user_id", "total_price"]].copy()
    df["total_price"] = pd.to_numeric(df["total_price"], errors="coerce")
    df = df.dropna(subset=["user_id", "total_price"]).reset_index(drop=True)
    df["source_row"] = np.arange(len(df), dtype=np.int64)

    # Reindex users in first-appearance order. Row order itself is unchanged.
    users = df["user_id"].drop_duplicates().tolist()
    id_map = {user_id: i for i, user_id in enumerate(users)}
    df["user_id_original"] = df["user_id"]
    df["user_id"] = df["user_id"].map(id_map).astype(np.int64)

    df["value_true"] = df["total_price"].astype(float)
    df["value_private"] = df["value_true"].clip(clip_lower, clip_upper)
    return df[
        [
            "source_row",
            "user_id",
            "user_id_original",
            "value_true",
            "value_private",
        ]
    ]


def build_withhold_release_schedule(
    events: pd.DataFrame,
    b: int,
    n: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Apply the online buffering rule using memory-efficient typed arrays.

    Eligibility is measured in Matrix-Mechanism update indices, not raw arrival
    indices. When several buffered users are eligible, the user whose oldest
    buffered contribution arrived first is released; ties use the integer user
    id. The returned columns are the same as in the original implementation,
    but the schedule is not accumulated as hundreds of thousands of Python
    dictionaries.
    """
    if b <= 0:
        raise ValueError("b must be a positive integer")
    if n <= 0:
        raise ValueError("n must be a positive integer")

    required = {
        "source_row",
        "user_id",
        "user_id_original",
        "value_true",
        "value_private",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    n = min(int(n), len(events))
    stream = events.iloc[:n]
    if stream.empty:
        raise ValueError("No UCI events are available for the requested horizon.")

    # Compact, fixed-size schedule storage. This replaces schedule_rows, which
    # used one large Python dictionary per arrival and caused MemoryError for
    # horizons such as n=400_000.
    source_rows = stream["source_row"].to_numpy(dtype=np.int64, copy=True)
    arriving_users = stream["user_id"].to_numpy(dtype=np.int64, copy=True)
    arriving_true = stream["value_true"].to_numpy(dtype=np.float64, copy=True)
    arriving_private = stream["value_private"].to_numpy(dtype=np.float64, copy=True)
    arriving_original = pd.Categorical(stream["user_id_original"].to_numpy(copy=False))

    raw_running_mean = np.empty(n, dtype=np.float64)
    update_occurred = np.zeros(n, dtype=np.bool_)
    mechanism_steps = np.zeros(n, dtype=np.int64)
    released_user_ids = np.full(n, np.nan, dtype=np.float64)
    released_counts = np.zeros(n, dtype=np.int32)
    released_true = np.full(n, np.nan, dtype=np.float64)
    released_private = np.full(n, np.nan, dtype=np.float64)
    true_running_mean = np.full(n, np.nan, dtype=np.float64)

    # 0 = carry forward, 1 = arriving user, 2 = another eligible buffer.
    release_reason_codes = np.zeros(n, dtype=np.int8)
    release_reason_categories = [
        "carry_forward_no_eligible_user",
        "arriving_user_eligible",
        "eligible_buffer_after_withhold",
    ]

    last_update: dict[int, int] = defaultdict(lambda: -10**18)
    buffers: dict[int, _UserBuffer] = defaultdict(_UserBuffer)
    future_heap: list[tuple[int, int, int, int]] = []
    ready_heap: list[tuple[int, int, int]] = []

    mechanism_update_count = 0
    cumulative_true_updates = 0.0
    private_updates = np.empty(n, dtype=np.float64)
    true_updates = np.empty(n, dtype=np.float64)

    def eligible_at(user_id: int) -> int:
        return int(last_update[user_id] + b)

    def buffer_add(
        user_id: int,
        arrival_idx: int,
        true_value: float,
        private_value: float,
    ) -> None:
        buf = buffers[user_id]
        if buf.count == 0:
            buf.version += 1
            buf.first_arrival = arrival_idx
            heapq.heappush(
                future_heap,
                (eligible_at(user_id), buf.first_arrival, user_id, buf.version),
            )
        buf.count += 1
        buf.sum_true += true_value
        buf.sum_private += private_value

    def promote_ready() -> None:
        while future_heap and future_heap[0][0] <= mechanism_update_count:
            _, first_arrival, user_id, version = heapq.heappop(future_heap)
            buf = buffers[user_id]
            if buf.count == 0 or buf.version != version:
                continue
            heapq.heappush(ready_heap, (first_arrival, user_id, version))

    def pop_ready_user() -> int | None:
        promote_ready()
        while ready_heap:
            _, user_id, version = heapq.heappop(ready_heap)
            buf = buffers[user_id]
            if buf.count == 0 or buf.version != version:
                continue
            if mechanism_update_count < eligible_at(user_id):
                heapq.heappush(
                    future_heap,
                    (eligible_at(user_id), buf.first_arrival, user_id, version),
                )
                continue
            return user_id
        return None

    def release_user(
        user_id: int,
        arrival_idx: int,
        include_current: tuple[float, float] | None,
        reason_code: int,
    ) -> tuple[int, int, int, float, float, int, float]:
        nonlocal mechanism_update_count, cumulative_true_updates

        buf = buffers[user_id]
        count = buf.count
        sum_true = buf.sum_true
        sum_private = buf.sum_private

        if include_current is not None:
            current_true, current_private = include_current
            count += 1
            sum_true += current_true
            sum_private += current_private

        if count <= 0:
            raise RuntimeError("Attempted to release an empty user buffer.")

        update_idx = mechanism_update_count
        if update_idx - last_update[user_id] < b:
            raise RuntimeError(
                f"Internal error: user {user_id} violates b={b} separation."
            )

        avg_true = sum_true / count
        avg_private = sum_private / count

        # Invalidate stale heap entries and clear the buffer.
        buf.version += 1
        buf.count = 0
        buf.sum_true = 0.0
        buf.sum_private = 0.0
        buf.first_arrival = -1

        last_update[user_id] = update_idx
        mechanism_update_count += 1
        cumulative_true_updates += avg_true
        true_updates[update_idx] = avg_true
        private_updates[update_idx] = avg_private

        return (
            mechanism_update_count,  # one-based step
            user_id,
            count,
            avg_true,
            avg_private,
            reason_code,
            cumulative_true_updates / mechanism_update_count,
        )

    raw_cumulative = 0.0
    previous_target = np.nan

    for arrival_idx in range(n):
        user_id = int(arriving_users[arrival_idx])
        true_value = float(arriving_true[arrival_idx])
        private_value = float(arriving_private[arrival_idx])
        raw_cumulative += true_value
        raw_running_mean[arrival_idx] = raw_cumulative / (arrival_idx + 1)

        update_info = None
        if mechanism_update_count - last_update[user_id] >= b:
            update_info = release_user(
                user_id=user_id,
                arrival_idx=arrival_idx,
                include_current=(true_value, private_value),
                reason_code=1,
            )
        else:
            buffer_add(
                user_id=user_id,
                arrival_idx=arrival_idx,
                true_value=true_value,
                private_value=private_value,
            )
            ready_user = pop_ready_user()
            if ready_user is not None:
                update_info = release_user(
                    user_id=ready_user,
                    arrival_idx=arrival_idx,
                    include_current=None,
                    reason_code=2,
                )

        if update_info is None:
            mechanism_steps[arrival_idx] = mechanism_update_count
            true_running_mean[arrival_idx] = previous_target
        else:
            (
                step,
                released_user,
                released_count,
                released_avg_true,
                released_avg_private,
                reason_code,
                target,
            ) = update_info
            update_occurred[arrival_idx] = True
            mechanism_steps[arrival_idx] = step
            released_user_ids[arrival_idx] = released_user
            released_counts[arrival_idx] = released_count
            released_true[arrival_idx] = released_avg_true
            released_private[arrival_idx] = released_avg_private
            release_reason_codes[arrival_idx] = reason_code
            true_running_mean[arrival_idx] = target
            previous_target = target

        if arrival_idx and arrival_idx % 100_000 == 0:
            print(
                f"Schedule progress: {arrival_idx:,}/{n:,} arrivals, "
                f"{mechanism_update_count:,} mechanism updates"
            )

    if mechanism_update_count == 0:
        raise RuntimeError("The schedule produced no Matrix-Mechanism updates.")

    schedule = pd.DataFrame(
        {
            "t": np.arange(1, n + 1, dtype=np.int64),
            "source_row": source_rows,
            "arriving_user_id": arriving_users,
            "arriving_user_id_original": arriving_original,
            "arriving_value_true": arriving_true,
            "arriving_value_private": arriving_private,
            "raw_running_mean": raw_running_mean,
            "update_occurred": update_occurred,
            "is_carry_forward": ~update_occurred,
            "mechanism_step": mechanism_steps,
            "released_user_id": released_user_ids,
            "released_count": released_counts,
            "released_value_true": released_true,
            "released_value_private": released_private,
            "release_reason": pd.Categorical.from_codes(
                release_reason_codes,
                categories=release_reason_categories,
            ),
            "true_running_mean": true_running_mean,
        },
        copy=False,
    )

    pending_rows = []
    for user_id, buf in buffers.items():
        if buf.count > 0:
            pending_rows.append(
                {
                    "user_id": user_id,
                    "count": buf.count,
                    "average_true": buf.sum_true / buf.count,
                    "average_private": buf.sum_private / buf.count,
                    "first_arrival": buf.first_arrival,
                    "eligible_at_mechanism_step_zero_based": eligible_at(user_id),
                }
            )
    pending = pd.DataFrame(pending_rows)

    return (
        schedule,
        private_updates[:mechanism_update_count].reshape(-1, 1).copy(),
        true_updates[:mechanism_update_count].reshape(-1, 1).copy(),
        pending,
    )

def attach_private_mechanism_outputs(
    schedule: pd.DataFrame,
    mu_hat_updates: np.ndarray,
) -> pd.DataFrame:
    """Map update-indexed Matrix-Mechanism outputs to every raw arrival."""
    # Share the schedule columns instead of duplicating the full 400k-row frame.
    result = schedule.copy(deep=False)
    mu = np.asarray(mu_hat_updates, dtype=float).reshape(-1)

    update_mask = result["update_occurred"].to_numpy(dtype=bool)
    update_steps = result.loc[update_mask, "mechanism_step"].to_numpy(dtype=np.int64)
    if len(update_steps) != len(mu):
        raise ValueError(
            f"Schedule has {len(update_steps)} updates, but mechanism returned {len(mu)} outputs."
        )
    if len(update_steps) and not np.array_equal(update_steps, np.arange(1, len(mu) + 1)):
        raise RuntimeError("Mechanism steps in the schedule are not consecutive.")

    public = np.empty(len(result), dtype=float)
    last_value = np.nan
    next_update = 0
    for i, did_update in enumerate(update_mask):
        if did_update:
            last_value = mu[next_update]
            next_update += 1
        public[i] = last_value

    result["private_running_mean"] = public
    error = result["true_running_mean"].to_numpy(dtype=float) - public
    result["squared_error"] = error**2
    result["cumulative_squared_error"] = np.cumsum(result["squared_error"].to_numpy(dtype=float))
    result["rmse_through_t"] = np.sqrt(
        result["cumulative_squared_error"].to_numpy(dtype=float)
        / np.arange(1, len(result) + 1, dtype=float)
    )
    return result


def _bandmf_cache_file_for_kind(c_kind: str):
    if c_kind != "BandMF":
        return None
    try:
        return BANDMF_G_CACHE_FILE
    except NameError as exc:
        raise NameError(
            "C_kind='BandMF' requires BANDMF_G_CACHE_FILE to be defined."
        ) from exc


def run_uci_dataset_grid(
    df_uci: pd.DataFrame,
    b_values: Sequence[int],
    n: int,
    num_seeds: int,
    *,
    seed_start: int = 1,
    c_kinds: Sequence[str] = ("Dtoep", "A1_sqrt"),
    p: int = 16,
    eps: float = 10.0,
    delta: float = 5e-6,
    xi: float = 200.0,
    clip_lower: float = -200.0,
    clip_upper: float = 200.0,
    out_dir: str | os.PathLike = "cache/uci_b_sweep",
    prefix: str = "uci_running_means",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Run the UCI experiment for all b values, mechanisms, and seeds.

    ``n`` is the common raw-arrival/publication horizon and is also retained as
    the conservative Matrix-Mechanism calibration horizon, matching the
    original real-data code. The actual number of mechanism updates can be
    smaller because no-eligible-user arrivals only carry forward the previous
    published estimate.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if num_seeds <= 0:
        raise ValueError("num_seeds must be positive")
    if abs(clip_lower) > xi or abs(clip_upper) > xi:
        raise ValueError(
            "For scalar data, clipping bounds must lie inside [-xi, xi] so the "
            "claimed L2 clipping norm is valid."
        )

    events = prepare_uci_events(
        df_uci,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )
    n_arrivals = min(int(n), len(events))
    if n_arrivals < n:
        print(
            f"Requested n={n}, but only {n_arrivals} valid rows are available; "
            f"using n={n_arrivals}."
        )
    n = n_arrivals

    b_values = [int(b) for b in b_values]
    if not b_values or any(b <= 0 for b in b_values):
        raise ValueError("b_values must contain positive integers")

    seeds = list(range(int(seed_start), int(seed_start) + int(num_seeds)))
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []

    for b in b_values:
        print(f"\n===== Building UCI withhold-release schedule for b={b}, n={n} =====")
        schedule, x_private_updates, x_true_updates, pending = build_withhold_release_schedule(
            events,
            b=b,
            n=n,
        )
        n_updates = len(x_private_updates)
        n_carry_forward = int((~schedule["update_occurred"]).sum())
        n_pending_values = int(pending["count"].sum()) if not pending.empty else 0
        max_user_participations = int(
            schedule.loc[schedule["update_occurred"], "released_user_id"]
            .value_counts()
            .max()
        )

        # Keep the original fixed-horizon calibration convention.
        k = math.ceil(n / b)
        schedule_path = out_path / f"{prefix}_schedule_b{b}_n{n}.csv"
        pending_path = out_path / f"{prefix}_pending_buffers_b{b}_n{n}.csv"
        if overwrite or not schedule_path.exists():
            schedule.to_csv(schedule_path, index=False)
        if overwrite or not pending_path.exists():
            pending.to_csv(pending_path, index=False)

        print(
            f"b={b}: arrivals={n}, mechanism_updates={n_updates}, "
            f"carry_forward_outputs={n_carry_forward}, "
            f"pending_raw_values={n_pending_values}, calibration_k={k}, "
            f"observed_max_user_updates={max_user_participations}"
        )

        for c_kind in c_kinds:
            p_run = effective_p(c_kind, b, p)
            print(
                f"\n--- Precomputing mechanism for b={b}, "
                f"C_kind={c_kind}, p={p_run}, calibration_n={n} ---"
            )
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
                    f"{prefix}_b{b}_seed{seed}_C{c_kind}_n{n}_updates{n_updates}_"
                    f"k{k}_p{p_run}_eps{eps:g}_delta{delta:g}_"
                    f"clipL{clip_lower:g}_clipU{clip_upper:g}_xi{xi:g}.csv"
                )
                csv_path = out_path / filename
                runtime_sec = np.nan
                status = "created"

                if csv_path.exists() and not overwrite:
                    print(f"Skipping existing file: {csv_path}")
                    status = "existing"
                else:
                    print(
                        f"Running b={b}, C_kind={c_kind}, seed={seed}: "
                        f"{n_updates} Matrix-Mechanism updates over {n} arrivals"
                    )
                    start = time.perf_counter()
                    mu_hat_updates = continual_mean_banded_Cinv(
                        x_private_updates,
                        g,
                        sigma,
                        xi=xi,
                        seed=seed,
                    ).reshape(-1)
                    runtime_sec = time.perf_counter() - start

                    result = attach_private_mechanism_outputs(
                        schedule=schedule,
                        mu_hat_updates=mu_hat_updates,
                    )
                    result.to_csv(csv_path, index=False)
                    print(f"Saved {csv_path} (runtime {runtime_sec:.3f} seconds)")
                    del result, mu_hat_updates

                manifest_rows.append(
                    {
                        "csv_path": str(csv_path.resolve()),
                        "status": status,
                        "b": b,
                        "seed": seed,
                        "C_kind": c_kind,
                        "n_arrivals": n,
                        "n_mechanism_updates": n_updates,
                        "n_carry_forward": n_carry_forward,
                        "n_pending_values": n_pending_values,
                        "observed_max_user_updates": max_user_participations,
                        "calibration_n": n,
                        "calibration_k": k,
                        "p": p_run,
                        "eps": eps,
                        "delta": delta,
                        "xi": xi,
                        "clip_lower": clip_lower,
                        "clip_upper": clip_upper,
                        "sensitivity": sensitivity,
                        "sigma": sigma,
                        "runtime_sec": runtime_sec,
                        "schedule_path": str(schedule_path.resolve()),
                        "pending_buffers_path": str(pending_path.resolve()),
                    }
                )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_path / "run_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nSaved run manifest to {manifest_path}")
    return manifest


if __name__ == "__main__":
    # Edit only these experiment settings.
    UCI_CSV = "uci_data.csv"
    MAX_ROWS = 400_000
    N = 400_000  # raw-arrival/publication horizon and calibration horizon

    # Kept the same final-figure b sweep as the supplied real-data script.
    B_VALUES = [500, 1000, 1500, 2000, 2500, 3000, 3294]
    NUM_SEEDS = 10
    SEED_START = 1

    # Same two mechanisms used for the real-data figure.
    C_KINDS = ["Dtoep", "A1_sqrt"]

    P = 16
    EPS = 8.0
    DELTA = 5e-6
    XI = 50.0

    # UCI total_price contains returns/negative values, so use symmetric clipping.
    # Set CLIP_LOWER = 0.0 to reproduce the nonnegative clipping convention from
    # the credit-card experiment exactly.
    CLIP_LOWER = 0.0
    CLIP_UPPER = 50.0
    OUTPUT_DIR = "cache/uci_b_sweep"

    df_uci = load_uci_dataset(
        UCI_CSV,
        max_rows=MAX_ROWS,
    )

    run_uci_dataset_grid(
        df_uci,
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
