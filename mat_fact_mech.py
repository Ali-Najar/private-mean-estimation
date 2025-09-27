import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import pandas as pd
import math 


def prepare_events(df, lower_clip=1.0, upper_clip=1000.0):
    """
    df: expects columns ['user_id', 'event_time', 'amount'] with event_time as datetime
    returns events time-sorted + clipped amount, plus integer user ids
    """
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
    """
    events: DataFrame with columns ['user_id', 'event_time', 'amount'], time-sorted
    b: minimum separation (in released index steps) between two events of the same user

    Strategy:
      - Stream events in chronological order.
      - If current event's user is eligible (curr_idx - last_idx[u] >= b), release it.
      - Otherwise, push to that user's FIFO buffer.
      - After every release, greedily flush any now-eligible buffered heads.
        Among all eligible buffered heads, release the one with the earliest original event_time
        (this preserves time order as much as the constraint allows).
      - If nothing is eligible, we keep buffering (no release). Some items may never be released
        if not enough other users arrive between repeats — by design of b-min-separation.

    Returns:
      released: DataFrame with the same columns, in the release order
    """
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

    cache_dir = "cache/g/"
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, f"g_{C_kind}_n{n}_p{p}.npy")


    if os.path.isfile(fname):
        try:
            g = np.load(fname)
            return g
        except Exception:
            pass

    if C_kind == "Dtoep":
        c = seq_Dtoep(n)
    elif C_kind == "A1_sqrt":
        c = seq_A1_sqrt_rec(n)
    else:
        raise ValueError("C_kind must be {'Dtoep','A1_sqrt'}")
    g = inv_series(c, p)      # only need p coeffs

    tmp = fname + ".tmp.npy"
    np.save(tmp, g)
    os.replace(tmp, fname)   # now this works, because tmp really exists

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

        if t % 1000 == 0:
            print("TIMESTEP: ", t)

    return mu_hat

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

def mat_fact():

    EXP = 19
    n = 2 ** EXP
    k = 8
    b = n // k
    p = 16
    eps = 1
    delta = 1e-6
    xi = 1
    mu = 0.5

    # Two C_kinds
    C_kinds = ["Dtoep", "A1_sqrt"]  
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


def run_private_running_mean(
    df_raw,
    b=500,
    C_kind="Dtoep",
    p=16,
    eps=10.0,
    delta=5e-6,
    xi=1000,
    clip_lower=0.0,
    clip_upper=1000.0,
    seed=1234,
    save_csv=True,
    out_dir="cache/mat_fact_algo_csv",
    prefix="running_means",
):
    # 1) prep + clip
    events = prepare_events(df_raw, clip_lower, clip_upper)

    events_unclipped = prepare_events(df_raw, - 1e-20, 1e20)
    released_unclipped = enforce_b_min_separation(events, b=b)

    # 2) enforce b-min-separation
    released = enforce_b_min_separation(events, b=b)

    n = len(released)

    print("N: ", n)

    # 3) stream matrix for d=1
    X = to_stream_matrix(released, d=1)
    
    # 4) build C^{-1} band (g) and sensitivity for this n and b
    g = build_g_from_Ckind(n=n, p=p, C_kind=C_kind)
    newC = inv_series(g, n)  # first column of C (length n)

    # k is the max per-user participations bound over length n: ceil(n / b)
    k = math.ceil(n / b)

    sensitivity = sens(newC, n, k, b)
    sigma = sigma_eps_delta(eps, delta) * xi * sensitivity

    mu_hat = continual_mean_banded_Cinv(X, g, sigma, xi=xi, seed=seed)

    csv_path = None
    if save_csv:
        # true (noiseless) running mean over the released, clipped stream
        true_rm = np.cumsum(released_unclipped["amount"].to_numpy(float)) / np.arange(1, n + 1)
        priv_rm = mu_hat.reshape(-1)

        if len(true_rm) != len(priv_rm):
            raise ValueError("Lengths mismatch between true and private running means.")

        df_out = pd.DataFrame({
            "true_running_mean": true_rm,
            "private_running_mean": priv_rm
        })

        os.makedirs(out_dir, exist_ok=True)
        parts = [prefix, f"b{b}", f"k{k}", f"p{p}", f"eps{eps}", f"delta{delta}", f"xi{xi}", f"C{C_kind}"]
        fname = "_".join(parts) + ".csv"
        csv_path = os.path.join(out_dir, fname)
        df_out.to_csv(csv_path, index=False)

    diagnostics = {
        "n_input": int(len(df_raw)),
        "n_released": int(n),
        "num_users": int(events["user_id"].nunique()),
        "b": int(b),
        "k": int(k),
        "C_kind": C_kind,
        "p": int(p),
        "eps": float(eps),
        "delta": float(delta),
        "xi": float(xi),
        "sigma": float(sigma),
        "sensitivity": float(sensitivity),
    }
    return released, mu_hat, diagnostics

import kagglehub
import pandas as pd
import glob, os

use_real_data = True

if use_real_data:
    path = kagglehub.dataset_download("priyamchoksi/credit-card-transactions-dataset")
    print("Path to dataset files:", path)

    files = glob.glob(os.path.join(path, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {path}")

    dfs = []
    for f in files:
        df_part = pd.read_csv(
            f,
            usecols=["cc_num", "trans_date_trans_time", "amt"],
            dtype={"cc_num": "string"},
            parse_dates=["trans_date_trans_time"],
            infer_datetime_format=True
        )
        dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True)

    df = df[:200000]

    df_clean = df.rename(columns={
        "cc_num": "user_id",
        "trans_date_trans_time": "event_time",
        "amt": "amount",
    })

    df_clean = df_clean.dropna(subset=["user_id", "event_time", "amount"]).copy()
    df_clean["amount"] = pd.to_numeric(df_clean["amount"], errors="coerce")
    df_clean = df_clean.dropna(subset=["amount"])

    for idx, ckind in enumerate(["Dtoep", "A1_sqrt"]):
        print(f"Processing {ckind}")
        run_private_running_mean(df_clean, C_kind=ckind,seed = idx)


else:
    mat_fact()