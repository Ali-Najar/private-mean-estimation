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
    "I": BASE_DIR / "mat_fact_algo" / "I",
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
    if T <= 1:
        return np.arange(T, dtype=int)

    if mode == "fixed":
        return np.arange(0, T, step, dtype=int)

    if mode == "progress":
        idx = [0]
        gap = 1.0
        i = 0
        while i + int(gap) < T:
            i += int(gap)
            idx.append(i)
            gap *= growth
            if len(idx) >= num_points:
                break
        if idx[-1] != T - 1:
            idx.append(T - 1)
        return np.array(sorted(set(idx)), dtype=int)

    if mode == "log":
        idx = np.unique(np.logspace(0, np.log10(T - 1), num=num_points, base=10).astype(int))
        idx[0] = 0
        if idx[-1] != T - 1:
            idx = np.append(idx, T - 1)
        return idx

    return np.arange(T, dtype=int)

def trim_to_common_length(arrays):
    if not arrays:
        return arrays
    min_len = min(len(a) for a in arrays)
    return [a[:min_len] for a in arrays]

# ---------- New: compute √(∑_{j≤t} E[(μ̂_j - μ)^2]) ----------
def rms_cumsum_curve(arrs, *, bootstrap=0, random_state=None):
    """
    arrs: list of 1D arrays (one per seed) of errors (μ̂_t - μ), all same length.
    Returns dict with keys: 'y' (curve), optional 'low','high' (95% CI), and 'n'.
    """
    A = np.stack(arrs, axis=0)              # (n_seeds, T), entries are (μ̂_t - μ)
    A2 = A**2                               # squared error per seed per t
    mse_t = A2.mean(axis=0)                 # E[(μ̂_t - μ)^2] estimated over seeds
    y = np.sqrt(np.cumsum(mse_t))           # sqrt of cumulative sum over time

    out = {"y": y, "n": A.shape[0]}

    # Optional bootstrap CIs (resample seeds)
    if bootstrap and A.shape[0] > 1:
        rng = np.random.default_rng(random_state)
        B = int(bootstrap)
        Ys = []
        for _ in range(B):
            idx = rng.integers(0, A.shape[0], size=A.shape[0])
            mse_b = (A2[idx].mean(axis=0))
            Ys.append(np.sqrt(np.cumsum(mse_b)))
        Ys = np.stack(Ys, axis=0)
        out["low"] = np.quantile(Ys, 0.025, axis=0)
        out["high"] = np.quantile(Ys, 0.975, axis=0)
    return out

def consolidate_params(all_params_per_method):
    bucket = defaultdict(list)
    for plist in all_params_per_method.values():
        for p in plist:
            for k, v in p.items():
                if k in ("seed", "b"):
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
    # distinguish from the |μ̂_t-μ| plot
    return out_dir / ("plot_rmscum_" + "_".join(parts) + ".pdf")

def main(mu, EXP=None, m=None, k=None, p=None, delta_bin=None, delta_mat=None, bootstrap_CI=0, seed=None):
    """
    bootstrap_CI: set to e.g. 200 for 95% CIs via bootstrap over seeds (0 = off).
    seed: RNG seed for bootstrap.
    """
    mu_val = float(mu)
    mu_folder = f"mu{mu}"
    method_dirs = {mth: pth / mu_folder for mth, pth in METHODS.items()}

    arrays_by_method, params_by_method = {}, {}
    for method, dir_path in method_dirs.items():
        # choose correct delta depending on method
        filters = {"EXP": EXP, "m": m, "k": k, "p": p}
        if method == "bin_mech_algo" and delta_bin is not None:
            filters["delta"] = delta_bin
        if method in ("A1_sqrt", "Dtoep", "I") and delta_mat is not None:
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

    # Compute stats: √(∑_{j≤t} E[(μ̂_j-μ)^2])
    stats = {}
    for mth, arrs in arrays_by_method.items():
        stats[mth] = rms_cumsum_curve(arrs, bootstrap=bootstrap_CI, random_state=seed)

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

    T = next(iter(stats.values()))["y"].shape[0]
    x = np.arange(T)

    # Dense early, sparse later
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
        "A1_sqrt": "$\\mathbf{E}_1^{1/2}$",
        "Dtoep": "$\\mathbf{D}_{\\mathrm{Toep}}$",
        "I": "$\\mathbf{I}$",
    }

    # Decide y-scale: log only if all positive
    ymins = []
    for method, s in stats.items():
        ymins.append(np.min(s["y"][idx]))
    all_pos = np.min(ymins) > 0

    for method, s in stats.items():
        y = s["y"][idx]
        plt.plot(x[idx], y, label=f"{labels[method]}", linewidth=0.9)
        if "low" in s and "high" in s:
            plt.fill_between(x[idx], s["low"][idx], s["high"][idx], alpha=0.25)

    if all_pos:
        plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Timestep")
    plt.ylabel(r"$\sqrt{\sum_{j=1}^{t}\mathbb{E}\!\left[(\widehat{\mu}_j-\mu)^2\right]}$")
    plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20)) if all_pos else None
    plt.legend()
    plt.tight_layout(pad=0)
    plt.savefig(output_pdf, format="pdf")
    plt.close()

    print(f"Saved plot to: {output_pdf}")

# Example run
if __name__ == "__main__":
    mu = 0.5
    # Set bootstrap_CI>0 (e.g., 200) to draw 95% CIs; leave 0 for speed.
    main(mu, EXP=19, m=8, k=8, p=16, delta_bin=1e-3, delta_mat=1e-6, bootstrap_CI=0)
