import argparse
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.ticker as ticker

# ---------- Locations ----------
BASE_DIR = Path("cache")
METHODS = {
    "bin_mech_algo": BASE_DIR / "bin_mech_algo",
    "bin_mech_algo_approx_dp": BASE_DIR / "bin_mech_algo_approx_dp",
    "A1_sqrt": BASE_DIR / "mat_fact_algo" / "A1_sqrt",
    "Dtoep": BASE_DIR / "mat_fact_algo" / "Dtoep",
    "nu-FTRL": BASE_DIR / "mat_fact_algo" / "nu-FTRL",
    "I": BASE_DIR / "mat_fact_algo" / "I",
    "BandMF": BASE_DIR / "mat_fact_algo" / "BandMF",
}
PLOT_DIR = Path("plots") 

# ---------- Filename parsing ----------
# PATTERNS = [
#     # mat_fact_algo
#     re.compile(
#         r"^mu_hat_"
#         r"EXP(?P<EXP>\d+)"
#         r"_k(?P<k>\d+)"
#         r"_b(?P<b>\d+)"
#         r"_p(?P<p>\d+)"
#         r"_eps(?P<eps>[\d.]+)"
#         r"_delta(?P<delta>[\d.eE+-]+)"
#         r"_xi(?P<xi>[\d.]+)"
#         r"_seed(?P<seed>\d+)\.npy$"
#     ),
#     # bin_mech_algo
#     re.compile(
#         r"^mu_hat_"
#         r"EXP(?P<EXP>\d+)"
#         r"_m(?P<m>\d+)"
#         r"_eps(?P<eps>[\d.]+)"
#         r"_delta(?P<delta>[\d.eE+-]+)"
#         r"_seed(?P<seed>\d+)\.npy$"
#     ),
# ]

# SUM_PATTERNS = [
#     re.compile(
#         r"^sum_sqerr_"
#         r"EXP(?P<EXP>\d+)"
#         r"_k(?P<k>\d+)"
#         r"_b(?P<b>\d+)"
#         r"_p(?P<p>\d+)"
#         r"_eps(?P<eps>[\d.]+)"
#         r"_delta(?P<delta>[\d.eE+-]+)"
#         r"_xi(?P<xi>[\d.]+)"
#         r"_seed(?P<seed>\d+)\.npy$"
#     ),
#     re.compile(
#         r"^sum_sqerr_"
#         r"EXP(?P<EXP>\d+)"
#         r"_m(?P<m>\d+)"
#         r"_eps(?P<eps>[\d.]+)"
#         r"_delta(?P<delta>[\d.eE+-]+)"
#         r"_seed(?P<seed>\d+)\.npy$"
#     ),
# ]

PATTERNS = [
    # Approximate-DP binary mechanism. Parse the selected branch, but do not
    # filter it by default: the method's expected error uses the full mixture.
    re.compile(
        r"^mu_hat_"
        r"EXP(?P<EXP>\d+)"
        r"_m(?P<m>\d+)"
        r"_eps(?P<eps>[\d.eE+-]+)"
        r"_algdelta(?P<algdelta>[\d.eE+-]+)"
        r"_approxdelta(?P<approxdelta>[\d.eE+-]+)"
        r"_branch(?P<branch>private|truthful)"
        r"_seed(?P<seed>\d+)\.npy$"
    ),
    re.compile(
        r"^mu_hat_"
        r"EXP(?P<EXP>\d+)"
        r"_k(?P<k>\d+)"
        r"_b(?P<b>\d+)"
        r"_p(?P<p>\d+)"
        r"_eps(?P<eps>[\d.eE+-]+)"
        r"_delta(?P<delta>[\d.eE+-]+)"
        r"_xi(?P<xi>[\d.eE+-]+)"
        r"_seed(?P<seed>\d+)\.npy$"
    ),
    re.compile(
        r"^mu_hat_"
        r"EXP(?P<EXP>\d+)"
        r"_m(?P<m>\d+)"
        r"_eps(?P<eps>[\d.eE+-]+)"
        r"_delta(?P<delta>[\d.eE+-]+)"
        r"_seed(?P<seed>\d+)\.npy$"
    ),
]

SUM_PATTERNS = [
    re.compile(
        r"^sum_sqerr_"
        r"EXP(?P<EXP>\d+)"
        r"_m(?P<m>\d+)"
        r"_eps(?P<eps>[\d.eE+-]+)"
        r"_algdelta(?P<algdelta>[\d.eE+-]+)"
        r"_approxdelta(?P<approxdelta>[\d.eE+-]+)"
        r"_branch(?P<branch>private|truthful)"
        r"_seed(?P<seed>\d+)\.npy$"
    ),
    re.compile(
        r"^sum_sqerr_"
        r"EXP(?P<EXP>\d+)"
        r"_k(?P<k>\d+)"
        r"_b(?P<b>\d+)"
        r"_p(?P<p>\d+)"
        r"_eps(?P<eps>[\d.eE+-]+)"
        r"_delta(?P<delta>[\d.eE+-]+)"
        r"_xi(?P<xi>[\d.eE+-]+)"
        r"_seed(?P<seed>\d+)\.npy$"
    ),
    re.compile(
        r"^sum_sqerr_"
        r"EXP(?P<EXP>\d+)"
        r"_m(?P<m>\d+)"
        r"_eps(?P<eps>[\d.eE+-]+)"
        r"_delta(?P<delta>[\d.eE+-]+)"
        r"_seed(?P<seed>\d+)\.npy$"
    ),
]
def _values_match(meta_val, filter_val):
    if filter_val is None:
        return True
    if meta_val is None or meta_val == "":
        return False

    try:
        return np.isclose(float(meta_val), float(filter_val), rtol=0.0, atol=1e-12)
    except Exception:
        return str(meta_val) == str(filter_val)


def _apply_filters(meta: dict, filters: dict, *, require_present=False, debug=False, filename=None) -> bool:
    for k, v in filters.items():
        if v is None:
            continue

        if k not in meta:
            if require_present:
                if debug:
                    print(f"[DEBUG] {filename}: missing key {k!r} in meta={meta}")
                return False
            continue

        if not _values_match(meta[k], v):
            if debug:
                print(
                    f"[DEBUG] {filename}: mismatch on {k!r}: "
                    f"meta={meta[k]!r}, filter={v!r}, meta_all={meta}"
                )
            return False

    return True

def parse_sum_params_from_name(name: str):
    for pat in SUM_PATTERNS:
        mm = pat.match(name)
        if mm:
            return mm.groupdict()
    return {}

FILENAME_PARAM_ORDER = ["EXP", "m", "k", "p", "eps", "delta", "algdelta", "approxdelta", "xi", "mu"]

def parse_params_from_name(name: str):
    for pat in PATTERNS:
        m = pat.match(name)
        if m:
            return m.groupdict()
    return {}

# def load_arrays_from_dir(dir_path: Path, mu_val: float, filters: dict = None):
#     arrays, params = [], []
#     if not dir_path.exists():
#         return arrays, params
#     for f in sorted(dir_path.glob("mu_hat_EXP*.npy")):
#         try:
#             p = parse_params_from_name(f.name)

#             # --- apply filtering here ---
#             if filters:
#                 skip = False
#                 for k, v in filters.items():
#                     if v is not None and k in p:  # only check keys that exist in this filename
#                         if str(p[k]) != str(v):
#                             skip = True
#                             break
#                 if skip:
#                     continue
#             # ----------------------------

#             arr = np.load(f)
#             arr = np.asarray(arr, dtype=float).ravel()
#             arr = arr - mu_val
#             if not np.isfinite(arr).all():
#                 arr = arr[np.isfinite(arr)]
#             if arr.size == 0:
#                 continue
#             arrays.append(arr)
#             params.append(p)
#         except Exception as e:
#             print(f"[WARN] Skipping {f}: {e}")
#     return arrays, params

def load_arrays_from_dir(dir_path: Path, mu_val: float, filters: dict = None, method_name=None):
    arrays, params = [], []
    if not dir_path.exists():
        print(f"[WARN] Missing directory: {dir_path}")
        return arrays, params

    for f in sorted(dir_path.glob("mu_hat_EXP*.npy")):
        try:
            meta = parse_params_from_name(f.name)
            if not meta:
                print(f"[WARN] Could not parse filename: {f.name}")
                continue

            matched = True
            if filters:
                matched = _apply_filters(
                    meta,
                    filters,
                    require_present=(method_name == "BandMF"),
                    debug=(method_name == "BandMF"),
                    filename=f.name,
                )

            if not matched:
                continue

            arr = np.load(f)
            arr = np.asarray(arr, dtype=float).ravel()
            arr = arr - mu_val
            if not np.isfinite(arr).all():
                arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue

            arrays.append(arr)
            params.append(meta)

        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}")

    print(f"[DEBUG] {method_name}: loaded {len(arrays)} arrays from {dir_path}")
    return arrays, params

def choose_t_indices(T, mode="fixed", *, step=4, num_points=250, growth=1.15):
    """
    Returns indices in [0, T) to sample for plotting.

    mode:
      - "fixed":     constant gap, e.g. 0, 4, 8, ...
      - "progress":  gaps grow geometrically (dense early, sparse later)
      - "log":       ~log-spaced indices (good when xscale is linear)

    Parameters:
      step:       gap for "fixed" (t = 1,5,9,... corresponds to step=4)
      num_points: target count for "log" (and rough cap for "progress")
      growth:     multiplicative growth of the gap in "progress"
    """
    if T <= 1:
        return np.arange(T, dtype=int)

    if mode == "fixed":
        return np.arange(0, T, step, dtype=int)

    if mode == "progress":
        idx = [0]
        gap = 1.0
        i = 0
        # grow the gap gradually so we keep detail early on
        while i + int(gap) < T:
            i += int(gap)
            idx.append(i)
            gap *= growth
            if len(idx) >= num_points:  # safety cap
                break
        if idx[-1] != T - 1:
            idx.append(T - 1)
        return np.array(sorted(set(idx)), dtype=int)

    if mode == "log":
        # unique, increasing, log-distributed indices
        idx = np.unique(np.logspace(0, np.log10(T - 1), num=num_points, base=10).astype(int))
        idx[0] = 0
        if idx[-1] != T - 1:
            idx = np.append(idx, T - 1)
        return idx

    # fallback: plot all
    return np.arange(T, dtype=int)


def trim_to_common_length(arrays):
    if not arrays:
        return arrays
    min_len = min(len(a) for a in arrays)
    return [a[:min_len] for a in arrays]

def mean_and_95ci(arrays):
    A = np.stack(arrays, axis=0)
    n = A.shape[0]
    mean = A.mean(axis=0)
    std = A.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    sem = std / np.sqrt(n) if n > 0 else np.zeros_like(mean)
    z = 1.96
    low, high = mean - z * sem, mean + z * sem
    return mean.ravel(), low.ravel(), high.ravel(), n

def consolidate_params(all_params_per_method):
    bucket = defaultdict(list)
    for plist in all_params_per_method.values():
        for p in plist:
            for k, v in p.items():
                if k in ("seed", "b", "branch"):   # skip run-specific fields
                    continue
                if v is not None and v != "":
                    bucket[k].append(v)

    chosen = {}
    for k, values in bucket.items():
        if not values:
            continue
        if all(v == values[0] for v in values):
            chosen[k] = values[0]
        else:
            chosen[k] = values[0]
    return chosen

def build_output_filename(chosen_params, mu_str, out_dir: Path):
    chosen = dict(chosen_params)
    chosen["mu"] = str(mu_str)
    parts = [f"{key}{chosen[key]}" for key in FILENAME_PARAM_ORDER if key in chosen]
    return out_dir / ("plot_" + "_".join(parts) + ".pdf")

def main(
    mu,
    EXP=None,
    m=None,
    k=None,
    p=None,
    p_bandmf=None,
    eps_bandmf=None,
    delta_bin=None,
    algdelta_bin_approx=None,
    approx_delta_bin=None,
    approx_branch=None,
    delta_mat=None,
    delta_bandmf=None,
    xi_bandmf=None,
):
    
    mu_val = float(mu)
    mu_folder = f"mu{mu}"
    method_dirs = {mth: pth / mu_folder for mth, pth in METHODS.items()}

    arrays_by_method, params_by_method = {}, {}
    for method, dir_path in method_dirs.items():
        if method == "BandMF":
            filters = {
                "EXP": EXP,
                "k": k,
                "p": p_bandmf,
                "eps": eps_bandmf,
                "delta": delta_bandmf if delta_bandmf is not None else delta_mat,
                "xi": xi_bandmf,
            }
        else:
            filters = {"EXP": EXP, "m": m, "k": k, "p": p}

            if method == "bin_mech_algo" and delta_bin is not None:
                filters["delta"] = delta_bin
            if method == "bin_mech_algo_approx_dp":
                if algdelta_bin_approx is not None:
                    filters["algdelta"] = algdelta_bin_approx
                if approx_delta_bin is not None:
                    filters["approxdelta"] = approx_delta_bin
                if approx_branch is not None:
                    filters["branch"] = approx_branch
            if method in ("A1_sqrt", "Dtoep", "I", "nu-FTRL") and delta_mat is not None:
                filters["delta"] = delta_mat

        # arrays, plist = load_arrays_from_dir(dir_path, mu_val, filters)
        arrays, plist = load_arrays_from_dir(dir_path, mu_val, filters, method_name=method)
        if not arrays:
            print(f"[WARN] No arrays in {dir_path} matching {filters}")
            continue
        arrays = trim_to_common_length(arrays)
        arrays_by_method[method] = arrays
        params_by_method[method] = plist

    if not arrays_by_method:
        raise SystemExit("No data found for any method.")

    # Compute stats
    stats = {}
    for mth, arrs in arrays_by_method.items():
        arrs_abs = [np.abs(a) for a in arrs]
        stats[mth] = dict(zip(["mean","low","high","n"], mean_and_95ci(arrs_abs)))

    chosen_params = consolidate_params(params_by_method)
    overrides = {"EXP": EXP, "m": m, "k": k, "p": p}
    if delta_bin is not None:
        chosen_params["delta"] = delta_bin
    if algdelta_bin_approx is not None:
        chosen_params["algdelta"] = algdelta_bin_approx
    if approx_delta_bin is not None:
        chosen_params["approxdelta"] = approx_delta_bin
    if delta_mat is not None:
        chosen_params["delta"] = delta_mat
    for k_, v_ in overrides.items():
        if v_ is not None:
            chosen_params[k_] = str(v_)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_pdf = build_output_filename(chosen_params, mu, PLOT_DIR)

    T = next(iter(stats.values()))["mean"].shape[0]
    x = np.arange(1, T + 1)
    # idx = choose_t_indices(T, mode="fixed", step=4)

    # 2) Dense early, sparse later
    idx = choose_t_indices(T, mode="progress", growth=1.07, num_points=300)
    idx = idx[idx <= 500000]
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        
        "legend.fontsize": 20,
        "font.size": 26,
        "axes.labelsize": 26,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}"
    })
    plt.rcParams['pdf.fonttype'] = 42

    labels = {
        "bin_mech_algo": "CME (George et al., 2024), $(\\varepsilon,0)$-DP",
        "bin_mech_algo_approx_dp": "CME (George et al., 2024), $(\\varepsilon,\\delta)$-DP",
        "A1_sqrt": "$\\mathbf{E}_1^{1/2}$",
        "Dtoep": "$\\mathbf{D}_{\\mathrm{Toep}}$",
        "nu-FTRL": "$\\mathbf{E}_{\\nu}^{1/2}$",
        "I": "$\\mathbf{I}$",
        "BandMF": "$\\mathrm{BandMF}$",
    }

    # for method, s in stats.items():
    #     mean, low, high = s["mean"][idx], s["low"][idx], s["high"][idx]
    #     plt.plot(x[idx], mean, label=f"{labels[method]}", linewidth=0.9)
    #     plt.fill_between(x[idx], low, high, alpha=0.25)

    for method, s in stats.items():
        mean, low, high = s["mean"][idx], s["low"][idx], s["high"][idx]

        if method == "BandMF":
            plt.plot(
                x[idx],
                mean,
                label=labels[method],
                linewidth=1.5,
                color="tab:blue",
                linestyle="--",
            )
            plt.fill_between(x[idx], low, high, alpha=0.15, color="tab:blue")
        elif method == "bin_mech_algo_approx_dp":
            plt.plot(
                x[idx], mean, label=labels[method], linewidth=1.5, linestyle="--"
            )
            plt.fill_between(x[idx], low, high, alpha=0.20)
        else:
            plt.plot(x[idx], mean, label=labels[method], linewidth=0.9)
            plt.fill_between(x[idx], low, high, alpha=0.25)

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Timestep")
    plt.ylabel(r"$\left|\widehat{\mu}_t - \mu\right|$")
    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20))
    plt.legend()
    plt.tight_layout(pad=0)
    plt.savefig(output_pdf, format="pdf")
    plt.close()

    print(f"Saved plot to: {output_pdf}")


    # --- Compute sqrt( (1/t) * sum_{j=1}^t E[(mu_j - mu_hat_j)^2] ) with 95% CI ---
    sum_stats_by_method = {}  # method -> dict(mean, low, high, n)

    p_dim = p  # preserve the function argument

    for method, dir_path in method_dirs.items():
        if method == "BandMF":
            filters = {
                "EXP": EXP,
                "k": k,
                "p": p_bandmf,
                "eps": eps_bandmf,
                "delta": delta_bandmf if delta_bandmf is not None else delta_mat,
                "xi": xi_bandmf,
            }
        else:
            filters = {"EXP": EXP, "m": m, "k": k, "p": p_dim}

            if method == "bin_mech_algo" and delta_bin is not None:
                filters["delta"] = delta_bin
            if method == "bin_mech_algo_approx_dp":
                if algdelta_bin_approx is not None:
                    filters["algdelta"] = algdelta_bin_approx
                if approx_delta_bin is not None:
                    filters["approxdelta"] = approx_delta_bin
                if approx_branch is not None:
                    filters["branch"] = approx_branch
            if method in ("A1_sqrt", "Dtoep", "I", "nu-FTRL") and delta_mat is not None:
                filters["delta"] = delta_mat

        per_seed_S = []
        for f in sorted(dir_path.glob("sum_sqerr_EXP*.npy")):
            meta = parse_sum_params_from_name(f.name)
            if not meta:
                print(f"[WARN] Could not parse sumsq filename: {f.name}")
                continue

            if not _apply_filters(
                meta,
                filters,
                require_present=(method == "BandMF"),
                debug=(method == "BandMF"),
                filename=f.name,
            ):
                continue

            S = np.load(f).astype(float)
            per_seed_S.append(S)

        if method == "BandMF":
            print("SFD")
        if not per_seed_S:
            continue

        if method == "BandMF":
            print("SFD")


        # trim all seeds to the same length
        Tmin = min(len(S) for S in per_seed_S)
        per_seed_S = [S[:Tmin] for S in per_seed_S]
        S_stack = np.stack(per_seed_S, axis=0)               # (n_seeds, T)

        t_arr = np.arange(1, Tmin + 1, dtype=float)          # 1..T
        tiny = np.finfo(float).tiny

        # per-seed transformed curve: sqrt(S_t / t)
        Y = np.sqrt(np.maximum(S_stack / t_arr, tiny))        # (n_seeds, T)

        n = Y.shape[0]
        mean = Y.mean(axis=0)
        std  = Y.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
        sem  = std / np.sqrt(n) if n > 0 else np.zeros_like(mean)
        z = 1.96  # normal approx
        low = mean - z * sem
        high = mean + z * sem

        sum_stats_by_method[method] = dict(mean=mean, low=low, high=high, n=n, T=Tmin)

    # ---- Plot
    if sum_stats_by_method:
        T2 = min(v["T"] for v in sum_stats_by_method.values())
        x2 = np.arange(1, T2 + 1)
        idx2 = choose_t_indices(T2, mode="progress", growth=1.07, num_points=300)

        plt.figure(figsize=(10, 6))
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "legend.fontsize": 16,
            "font.size": 26,
            "axes.labelsize": 26,
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
            "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}"
        })
        plt.rcParams['pdf.fonttype'] = 42

        labels = {
            "bin_mech_algo": "CME (George et al., 2024), $(\\varepsilon,0)$-DP",
            "bin_mech_algo_approx_dp": "CME (George et al., 2024), $(\\varepsilon,\\delta)$-DP",
            "A1_sqrt": "$\\mathbf{E}_1^{1/2}$",
            "Dtoep": "$\\mathbf{D}_{\\mathrm{Toep}}$",
            "nu-FTRL": "$\\mathbf{E}_{\\nu}^{1/2}$",
            "I": "$\\mathbf{I}$",
            "BandMF": "$\\mathrm{BandMF}$",
        }

        plot_ratio = False

        # for method, s in sum_stats_by_method.items():
        #     print(method)

        #     if plot_ratio:
        #         best = sum_stats_by_method["Dtoep"]
        #         best_m, best_lo, best_hi = best["mean"][idx2], best["low"][idx2], best["high"][idx2]
        #         m, lo, hi = s["mean"][idx2], s["low"][idx2], s["high"][idx2]
        #         plt.plot(x2[idx2], m / best_m, label=labels[method], linewidth=2)
        #     else:
        #         m, lo, hi = s["mean"][idx2], s["low"][idx2], s["high"][idx2]
        #         if "bin" in method:
        #             print(m[22])
        #         plt.plot(x2[idx2], m, label=labels[method], linewidth=2)
        #         plt.fill_between(x2[idx2], lo, hi, alpha=0.25)

        for method, s in sum_stats_by_method.items():

            if plot_ratio:
                best = sum_stats_by_method["Dtoep"]
                best_m = best["mean"][idx2]
                m, lo, hi = s["mean"][idx2], s["low"][idx2], s["high"][idx2]

                if method == "BandMF":
                    plt.plot(
                        x2[idx2],
                        m / best_m,
                        label=labels[method],
                        linewidth=2.5,
                        color="tab:blue",
                        linestyle="--",
                    )
                elif method == "bin_mech_algo_approx_dp":
                    plt.plot(
                        x2[idx2],
                        m / best_m,
                        label=labels[method],
                        linewidth=2,
                        linestyle="--",
                    )
                else:
                    plt.plot(x2[idx2], m / best_m, label=labels[method], linewidth=2)
            else:
                m, lo, hi = s["mean"][idx2], s["low"][idx2], s["high"][idx2]

                if method == "BandMF":
                    plt.plot(
                        x2[idx2],
                        m,
                        label=labels[method],
                        linewidth=2.5,
                        color="tab:blue",
                        linestyle="--",
                    )
                    plt.fill_between(x2[idx2], lo, hi, alpha=0.15, color="tab:blue")
                elif method == "bin_mech_algo_approx_dp":
                    plt.plot(
                        x2[idx2],
                        m,
                        label=labels[method],
                        linewidth=2,
                        linestyle="--",
                    )
                    plt.fill_between(x2[idx2], lo, hi, alpha=0.20)
                else:
                    plt.plot(x2[idx2], m, label=labels[method], linewidth=2)
                    plt.fill_between(x2[idx2], lo, hi, alpha=0.25)

        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("Timestep")
        plt.ylabel(r"RMSE")
        plt.legend(loc="upper right")

        # plt.grid()
        plt.tight_layout(pad=0)
        if plot_ratio:
            save_path = output_pdf.parent / f"avg_sum_sqerrs_{output_pdf.stem}_ratio.pdf"
        else:
            save_path = output_pdf.parent / f"avg_sum_sqerrs_{output_pdf.stem}.pdf"
        plt.savefig(save_path, format="pdf")
        plt.close()



# Example run
mu = 0.5
# main(mu, EXP=19, m=8, k=128, p=16, p_bandmf=64, delta_bin=1e-3, delta_mat=1e-6)
main(
    mu,
    EXP=19,
    m=8,
    k=128,
    p=16,
    p_bandmf=512,
    eps_bandmf=1,
    delta_bin=1e-3,
    algdelta_bin_approx=1e-3,
    approx_delta_bin=1e-6,
    # Keep None for the correct mixture average. Set to "private" or
    # "truthful" only for branch-specific diagnostics.
    approx_branch=None,
    delta_mat=1e-6,
    delta_bandmf=1e-6,
    xi_bandmf=1,
)