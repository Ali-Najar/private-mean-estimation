#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import numpy as np
from collections import defaultdict

# ---------- Locations ----------
BASE_DIR = Path("cache")
METHODS = {
    "bin_mech_algo": BASE_DIR / "bin_mech_algo",
    "A1_sqrt": BASE_DIR / "mat_fact_algo" / "A1_sqrt",
    "Dtoep": BASE_DIR / "mat_fact_algo" / "Dtoep",
}
OUT_DIR = Path("plots")  # where CSVs will be saved

# ---------- Filename parsing ----------
PATTERNS = [
    # mat_fact_algo (A1_sqrt / Dtoep)
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

def load_arrays_from_dir(dir_path: Path, mu_val: float, filters: dict | None = None):
    """
    Load all runs from dir_path, subtract mu, and return list of 1D error arrays (signed),
    plus the parsed parameter dicts.
    Filters: dict of {param_name -> value}; only keys present in filename are checked.
    """
    arrays, params = [], []
    if not dir_path.exists():
        return arrays, params

    for f in sorted(dir_path.glob("*.npy")):
        try:
            meta = parse_params_from_name(f.name)

            # Apply filters only on keys that exist in this filename
            if filters:
                for k, v in filters.items():
                    if v is not None and k in meta and str(meta[k]) != str(v):
                        break
                else:
                    pass  # all ok
                # if loop didn't 'break', continue as normal; if it did, skip file
                if any(v is not None and k in meta and str(meta[k]) != str(v) for k, v in filters.items()):
                    continue

            arr = np.load(f)
            arr = np.asarray(arr, dtype=float).ravel()
            err = arr - mu_val  # signed error
            if not np.isfinite(err).all():
                err = err[np.isfinite(err)]
            if err.size == 0:
                continue

            arrays.append(err)
            params.append(meta)
        except Exception as e:
            print(f"[WARN] Skipping {f} due to error: {e}")

    return arrays, params

def trim_to_common_length(arrays):
    if not arrays:
        return arrays
    min_len = min(len(a) for a in arrays)
    return [a[:min_len] for a in arrays]

def consolidate_params(all_params_per_method):
    """
    Keep a representative value for each key so filenames include m/k/p/etc even if
    only one method has them. If conflicting values appear, take the first seen.
    """
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
        if values:
            chosen[k] = values[0]
    return chosen

def build_output_stem(chosen_params, mu_str):
    chosen = dict(chosen_params)
    chosen["mu"] = str(mu_str)
    parts = [f"{key}{chosen[key]}" for key in FILENAME_PARAM_ORDER if key in chosen]
    return "_".join(parts)

def main(mu, EXP=None, m=None, k=None, p=None):
    mu_val = float(mu)
    mu_folder = f"mu{mu}"
    method_dirs = {mth: pth / mu_folder for mth, pth in METHODS.items()}

    # Only check keys that exist per filename type
    filters = {"EXP": EXP, "m": m, "k": k, "p": p}

    arrays_by_method, params_by_method = {}, {}
    for method, dir_path in method_dirs.items():
        arrays, plist = load_arrays_from_dir(dir_path, mu_val, filters)
        if not arrays:
            print(f"[WARN] No arrays in {dir_path} matching {filters}")
            continue
        arrays = trim_to_common_length(arrays)
        arrays_by_method[method] = arrays
        params_by_method[method] = plist

    if not arrays_by_method:
        raise SystemExit("No data found for any method.")

    # Choose params for naming, then apply user overrides so they appear in filenames.
    chosen_params = consolidate_params(params_by_method)
    overrides = {"EXP": EXP, "m": m, "k": k, "p": p}
    for kk, vv in overrides.items():
        if vv is not None:
            chosen_params[kk] = str(vv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name_stem = build_output_stem(chosen_params, mu)

    # For each method: compute MSE_t (mean over runs of (err^2)), and scalar MSE (mean over t of MSE_t)
    summary_rows = []
    for method, arrays in arrays_by_method.items():
        # squared errors per run
        sq = [a.astype(np.float64, copy=False)**2 for a in arrays]  # list of (T,)
        S = np.stack(sq, axis=0)  # (n_runs, T)

        mse_t = S.mean(axis=0)         # (T,)  E[(err_t)^2] across seeds
        mse_scalar = float(mse_t.mean())  # single number: average over time

        # Save per-method time-indexed MSE to CSV
        per_method_csv = OUT_DIR / f"mse_{name_stem}_{method}.csv"
        with open(per_method_csv, "w", newline="") as f:
            f.write("t,mse_t\n")
            # 1-based timestep in CSV; change to range(len(mse_t)) if you prefer 0-based
            for t_idx, val in enumerate(mse_t, start=1):
                f.write(f"{t_idx},{val:.10g}\n")
        print(f"[INFO] Wrote per-timestep MSE for {method} -> {per_method_csv}")

        summary_rows.append((method, mse_scalar, S.shape[0], len(mse_t)))

    # Save combined summary CSV
    summary_csv = OUT_DIR / f"mse_summary_{name_stem}.csv"
    with open(summary_csv, "w", newline="") as f:
        f.write("method,mse_scalar,n_runs,T\n")
        for method, mse_scalar, n_runs, Tlen in summary_rows:
            f.write(f"{method},{mse_scalar:.10g},{n_runs},{Tlen}\n")
    print(f"[INFO] Wrote summary MSE -> {summary_csv}")

if __name__ == "__main__":
    mu = 0.5
    EXP = 19
    m = 64
    k = 64
    p = 16
    main(mu=mu, EXP=EXP, m=m, k=k, p=p)
