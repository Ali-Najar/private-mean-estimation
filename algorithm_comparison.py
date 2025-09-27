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
    "A1_sqrt": BASE_DIR / "mat_fact_algo" / "A1_sqrt",
    "Dtoep": BASE_DIR / "mat_fact_algo" / "Dtoep",
}
PLOT_DIR = Path("plots") 

# ---------- Filename parsing ----------
PATTERNS = [
    # mat_fact_algo
    re.compile(
        r"^mu_hat_"
        r"EXP(?P<EXP>\d+)"
        r"_k(?P<k>\d+)"
        r"_b(?P<b>\d+)"
        r"_p(?P<p>\d+)"
        r"_eps(?P<eps>[\d.]+)"
        r"_delta(?P<delta>[\d.eE+-]+)"
        r"_xi(?P<xi>[\d.]+)"
        r"_seed(?P<seed>\d+)\.npy$"
    ),
    # bin_mech_algo
    re.compile(
        r"^mu_hat_"
        r"EXP(?P<EXP>\d+)"
        r"_m(?P<m>\d+)"
        r"_eps(?P<eps>[\d.]+)"
        r"_delta(?P<delta>[\d.eE+-]+)"
        r"_seed(?P<seed>\d+)\.npy$"
    ),
]
FILENAME_PARAM_ORDER = ["EXP", "m", "k", "p", "eps", "delta", "xi", "mu"]

def parse_params_from_name(name: str):
    for pat in PATTERNS:
        m = pat.match(name)
        if m:
            return m.groupdict()
    return {}

def load_arrays_from_dir(dir_path: Path, mu_val: float, filters: dict = None):
    arrays, params = [], []
    if not dir_path.exists():
        return arrays, params
    for f in sorted(dir_path.glob("*.npy")):
        try:
            p = parse_params_from_name(f.name)

            # --- apply filtering here ---
            if filters:
                skip = False
                for k, v in filters.items():
                    if v is not None and k in p:  # only check keys that exist in this filename
                        if str(p[k]) != str(v):
                            skip = True
                            break
                if skip:
                    continue
            # ----------------------------

            arr = np.load(f)
            arr = np.asarray(arr, dtype=float).ravel()
            arr = arr - mu_val
            if not np.isfinite(arr).all():
                arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            arrays.append(arr)
            params.append(p)
        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}")
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
                if k in ("seed", "b"):   # skip these
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

def main(mu, EXP=None, m=None, k=None, p=None, delta_bin=None, delta_mat=None):
    mu_val = float(mu)
    mu_folder = f"mu{mu}"
    method_dirs = {mth: pth / mu_folder for mth, pth in METHODS.items()}

    arrays_by_method, params_by_method = {}, {}
    for method, dir_path in method_dirs.items():
        # choose correct delta depending on method
        filters = {"EXP": EXP, "m": m, "k": k, "p": p}
        if method == "bin_mech_algo" and delta_bin is not None:
            filters["delta"] = delta_bin
        if method in ("A1_sqrt", "Dtoep") and delta_mat is not None:
            filters["delta"] = delta_mat

        arrays, plist = load_arrays_from_dir(dir_path, mu_val, filters)
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
    if delta_mat is not None:
        chosen_params["delta"] = delta_mat
    for k_, v_ in overrides.items():
        if v_ is not None:
            chosen_params[k_] = str(v_)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_pdf = build_output_filename(chosen_params, mu, PLOT_DIR)

    T = next(iter(stats.values()))["mean"].shape[0]
    x = np.arange(T)
    # idx = choose_t_indices(T, mode="fixed", step=4)

    # 2) Dense early, sparse later
    idx = choose_t_indices(T, mode="progress", growth=1.07, num_points=300)
    idx = idx[idx <= 500000]
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        
        "legend.fontsize": 24,
        "font.size": 26,
        "axes.labelsize": 26,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}"
    })
    plt.rcParams['pdf.fonttype'] = 42

    labels = {
        "bin_mech_algo": "CME (George et al., 2024)",
        "A1_sqrt": "$\\mathbf{A}_1^{1/2}$",
        "Dtoep": "$\\mathbf{D}_{\\mathrm{Toep}}$",
    }

    for method, s in stats.items():
        mean, low, high = s["mean"][idx], s["low"][idx], s["high"][idx]
        plt.plot(x[idx], mean, label=f"{labels[method]}", linewidth=0.9)
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

# Example run
mu = 0.5
main(mu, EXP=19, m=8, k=8, p=16, delta_bin=1e-3, delta_mat=1e-6)
