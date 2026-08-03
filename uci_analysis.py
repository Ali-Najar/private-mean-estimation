"""
UCI result analysis without run_manifest.csv.

The script performs only two analyses:

1. Final-timestep RMSE tables arranged as epsilon-by-b matrices.
   Every selected CSV must already contain the same fixed horizon n.
   The UCI experiment runner writes one row per raw arrival and already
   carries the previous estimate forward whenever no user is eligible, so this
   analysis never pads or invents additional rows.
   For every seed,

       RMSE_seed(n) = sqrt(sum_{t=1}^n (mu_t - mu_hat_t)^2 / n),

   and the table reports the arithmetic mean over the available seeds.
   One table is printed for each fixed (xi, delta, clipping interval, method),
   with epsilon on the rows and b on the columns.

2. Average absolute-error plots.
   For each fixed (b, epsilon, delta, clipping interval, xi), one plot compares
   the selected methods using

       average_over_seeds(|mu_t - mu_hat_t|)

   versus t. The y-axis uses a base-2 logarithmic scale. Every plot uses
   the same explicit fixed horizon n.

No RMSE-versus-time plots are generated.

The script discovers uci_running_means_*.csv files directly. It filters by the
requested fixed n and caches file lengths, final RMSE values, and aggregated
absolute-error plot data
so unchanged runs are much faster on later executions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import warnings
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


TRUE_COL = "true_running_mean"
PRIVATE_COL = "private_running_mean"
CUM_SQERR_COL = "cumulative_squared_error"
RMSE_CURVE_COL = "rmse_through_t"

RESULTS_DIR = "cache/uci_b_sweep"
PLOT_DIR = "plots/uci/avg_abs_error/vary_method"
CACHE_DIR_NAME = ".analysis_cache"

# Common fixed horizon written by the experiment runner and required here.
N = 400_000

# Edit these arrays once and run the script once.
EPS_VALUES = [1, 2, 4, 8, 10.0]
XI_VALUES = [50, 100.0]
B_VALUES = [500, 1000, 1500, 2000, 2500, 3000, 3294]
DELTA_VALUES = None
METHODS = ["A1_sqrt", "Dtoep"]
TABLE_METHODS = ["A1_sqrt", "Dtoep"]

RUN_KEY_COLS = [
    "b",
    "seed",
    "C_kind",
    "eps",
    "delta",
    "clip_lower",
    "clip_upper",
    "xi",
    "n_arrivals",
]
GROUP_COLS = [
    "eps",
    "delta",
    "clip_lower",
    "clip_upper",
    "xi",
    "n_arrivals",
    "b",
    "C_kind",
]
FIXED_ABS_PLOT_COLS = [
    "b",
    "eps",
    "delta",
    "clip_lower",
    "clip_upper",
    "xi",
    "n_arrivals",
]

METHOD_LABELS = {
    "A1_sqrt": r"$\mathbf{E}_1^{1/2}$",
    "Dtoep": r"$\mathbf{D}_{\mathrm{Toep}}$",
    "nu-FTRL": r"$\mathbf{E}_{\nu}^{1/2}$",
    "I": r"$\mathbf{I}$",
    "BandMF": r"$\mathrm{BandMF}$",
}

RUN_FILENAME_PATTERNS = [
    # Current UCI withhold-release runner. ``n`` is the raw-arrival/publication
    # horizon, while ``updates`` is the number of actual Matrix-Mechanism steps.
    re.compile(
        r"^uci_running_means_"
        r"b(?P<b>\d+)"
        r"_seed(?P<seed>\d+)"
        r"_C(?P<C_kind>.+?)"
        r"_n(?P<n_arrivals>\d+)"
        r"_updates(?P<n_mechanism_updates>\d+)"
        r"_k(?P<k>\d+)"
        r"_p(?P<p>\d+)"
        r"_eps(?P<eps>[\d.eE+-]+)"
        r"_delta(?P<delta>[\d.eE+-]+)"
        r"_clipL(?P<clip_lower>[\d.eE+-]+)"
        r"_clipU(?P<clip_upper>[\d.eE+-]+)"
        r"_xi(?P<xi>[\d.eE+-]+)"
        r"\.csv$"
    ),
    # Backward-compatible UCI filename without explicit clipping tokens.
    re.compile(
        r"^uci_running_means_"
        r"b(?P<b>\d+)"
        r"_seed(?P<seed>\d+)"
        r"_C(?P<C_kind>.+?)"
        r"_n(?P<n_arrivals>\d+)"
        r"_updates(?P<n_mechanism_updates>\d+)"
        r"_k(?P<k>\d+)"
        r"_p(?P<p>\d+)"
        r"_eps(?P<eps>[\d.eE+-]+)"
        r"_delta(?P<delta>[\d.eE+-]+)"
        r"_xi(?P<xi>[\d.eE+-]+)"
        r"\.csv$"
    ),
]


def _token(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    return str(value)


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", r"\_"))


def _parse_run_filename(path: Path) -> dict[str, object] | None:
    for pattern in RUN_FILENAME_PATTERNS:
        match = pattern.match(path.name)
        if match is None:
            continue

        meta = match.groupdict()
        if meta.get("clip_lower") is None:
            meta["clip_lower"] = 0.0
        if meta.get("clip_upper") is None:
            # A legacy UCI runner used xi as the clipping upper bound
            # when clipL/clipU are not encoded in the filename.
            meta["clip_upper"] = meta["xi"]

        resolved = path.resolve()
        stat = resolved.stat()
        meta["csv_path"] = str(path)
        meta["csv_path_resolved"] = resolved
        meta["file_size"] = int(stat.st_size)
        meta["mtime_ns"] = int(stat.st_mtime_ns)
        return meta
    return None


def discover_runs(results_dir: str | Path) -> pd.DataFrame:
    root = Path(results_dir)
    if not root.exists():
        raise FileNotFoundError(f"Results directory not found: {root}")

    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("uci_running_means_b*_seed*.csv")):
        meta = _parse_run_filename(path)
        if meta is not None:
            rows.append(meta)

    if not rows:
        raise FileNotFoundError(
            f"No parseable uci_running_means_*.csv files were found under {root}."
        )

    runs = pd.DataFrame(rows)
    numeric_cols = [
        "b",
        "seed",
        "n_arrivals",
        "n_mechanism_updates",
        "k",
        "p",
        "eps",
        "delta",
        "clip_lower",
        "clip_upper",
        "xi",
        "file_size",
        "mtime_ns",
    ]
    for col in numeric_cols:
        runs[col] = pd.to_numeric(runs[col], errors="coerce")

    required = ["b", "seed", "n_arrivals", "n_mechanism_updates", "eps", "delta", "xi"]
    if runs[required].isna().any().any():
        bad = runs.loc[runs[required].isna().any(axis=1), "csv_path"]
        raise ValueError(
            "Could not parse required parameters from these filenames:\n"
            + "\n".join(map(str, bad))
        )

    runs["b"] = runs["b"].astype(int)
    runs["seed"] = runs["seed"].astype(int)
    runs["file_size"] = runs["file_size"].astype(np.int64)
    runs["mtime_ns"] = runs["mtime_ns"].astype(np.int64)
    runs["C_kind"] = runs["C_kind"].astype(str)

    # Keep one file for an exact repeated run. Sorting by modification time
    # chooses the newest matching file.
    runs = runs.sort_values(["mtime_ns", "csv_path"])
    runs = runs.drop_duplicates(subset=RUN_KEY_COLS, keep="last")
    return runs.reset_index(drop=True)


def _matches_any(value: object, allowed: Sequence | None) -> bool:
    if allowed is None:
        return True
    for candidate in allowed:
        try:
            if np.isclose(float(value), float(candidate), rtol=0.0, atol=1e-12):
                return True
        except (TypeError, ValueError):
            if str(value) == str(candidate):
                return True
    return False


def filter_runs(
    runs: pd.DataFrame,
    *,
    n: int,
    b_values: Sequence[int] | None,
    eps_values: Sequence[float] | None,
    xi_values: Sequence[float] | None,
    delta_values: Sequence[float] | None,
    methods: Sequence[str] | None,
) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("n must be a positive integer")

    keep = _numeric_mask(runs["n_arrivals"], n)
    filters = {
        "b": b_values,
        "eps": eps_values,
        "xi": xi_values,
        "delta": delta_values,
        "C_kind": methods,
    }
    for col, allowed in filters.items():
        if allowed is not None:
            keep &= runs[col].map(lambda x: _matches_any(x, allowed)).to_numpy()

    selected = runs.loc[keep].copy()
    if selected.empty:
        raise ValueError(
            "No discovered files match the requested n/b/epsilon/xi/delta/method values."
        )
    return selected


def _load_csv_cache(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        warnings.warn(f"Ignoring unreadable cache file {path}: {exc}")
        return pd.DataFrame(columns=columns)
    for col in columns:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame[columns]


def _save_csv_cache(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def _signature_key(row: object) -> tuple[str, int, int]:
    return (
        str(Path(row.csv_path_resolved)),
        int(row.file_size),
        int(row.mtime_ns),
    )


def attach_actual_file_lengths(
    runs: pd.DataFrame,
    *,
    cache_dir: Path,
    rebuild_cache: bool,
) -> pd.DataFrame:
    """Attach actual row counts, reusing cached counts for unchanged files."""
    cache_path = cache_dir / "file_lengths.csv"
    columns = ["csv_path_resolved", "file_size", "mtime_ns", "file_length"]
    cache = pd.DataFrame(columns=columns) if rebuild_cache else _load_csv_cache(cache_path, columns)

    cached: dict[tuple[str, int, int], int] = {}
    for row in cache.itertuples(index=False):
        try:
            cached[(str(row.csv_path_resolved), int(row.file_size), int(row.mtime_ns))] = int(row.file_length)
        except (TypeError, ValueError):
            continue

    lengths: list[int] = []
    updated_rows: list[dict[str, object]] = []
    hits = 0
    misses = 0

    for row in runs.itertuples(index=False):
        key = _signature_key(row)
        if key in cached:
            length = cached[key]
            hits += 1
        else:
            path = Path(row.csv_path_resolved)
            header = pd.read_csv(path, nrows=0)
            if "t" in header.columns:
                length = len(pd.read_csv(path, usecols=["t"]))
            else:
                with path.open("rb") as handle:
                    length = max(sum(1 for _ in handle) - 1, 0)
            misses += 1

        lengths.append(int(length))
        updated_rows.append(
            {
                "csv_path_resolved": key[0],
                "file_size": key[1],
                "mtime_ns": key[2],
                "file_length": int(length),
            }
        )

    result = runs.copy()
    result["file_length"] = lengths
    _save_csv_cache(pd.DataFrame(updated_rows), cache_path)
    print(f"File-length cache: {hits} hit(s), {misses} miss(es).")
    return result


def _final_rmse_from_csv(path: Path, n: int) -> float:
    """Return RMSE at fixed horizon ``n``; the CSV must contain at least n rows."""
    if n <= 0:
        raise ValueError("n must be a positive integer")

    columns = set(pd.read_csv(path, nrows=0).columns)

    if {TRUE_COL, PRIVATE_COL}.issubset(columns):
        frame = pd.read_csv(path, usecols=[TRUE_COL, PRIVATE_COL], nrows=n)
        if len(frame) != n:
            raise ValueError(f"{path} has {len(frame)} rows; expected fixed n={n}")
        error = (
            frame[TRUE_COL].to_numpy(dtype=float)
            - frame[PRIVATE_COL].to_numpy(dtype=float)
        )
        cumulative = float(np.dot(error, error))

    elif CUM_SQERR_COL in columns:
        frame = pd.read_csv(path, usecols=[CUM_SQERR_COL], nrows=n)
        if len(frame) != n:
            raise ValueError(f"{path} has {len(frame)} rows; expected fixed n={n}")
        cumulative = max(float(frame[CUM_SQERR_COL].iloc[-1]), 0.0)

    elif RMSE_CURVE_COL in columns:
        frame = pd.read_csv(path, usecols=[RMSE_CURVE_COL], nrows=n)
        if len(frame) != n:
            raise ValueError(f"{path} has {len(frame)} rows; expected fixed n={n}")
        final_rmse = float(frame[RMSE_CURVE_COL].iloc[-1])
        cumulative = max(float(n) * final_rmse * final_rmse, 0.0)

    else:
        raise ValueError(
            f"{path} must contain '{RMSE_CURVE_COL}', '{CUM_SQERR_COL}', "
            f"or both '{TRUE_COL}' and '{PRIVATE_COL}'."
        )

    return float(math.sqrt(max(cumulative, 0.0) / float(n)))


def _build_final_rmse_lookup(
    *,
    cache_dir: Path,
    rebuild_cache: bool,
) -> tuple[dict[tuple[str, int, int, int], float], Path]:
    # Cache final RMSE values for unchanged fixed-horizon CSV files.
    cache_path = cache_dir / "final_rmse_values_fixed_n.csv"
    columns = [
        "csv_path_resolved",
        "file_size",
        "mtime_ns",
        "t_max",
        "rmse_at_t_max",
    ]
    cache = pd.DataFrame(columns=columns) if rebuild_cache else _load_csv_cache(cache_path, columns)
    lookup: dict[tuple[str, int, int, int], float] = {}
    for row in cache.itertuples(index=False):
        try:
            key = (
                str(row.csv_path_resolved),
                int(row.file_size),
                int(row.mtime_ns),
                int(row.t_max),
            )
            lookup[key] = float(row.rmse_at_t_max)
        except (TypeError, ValueError):
            continue
    return lookup, cache_path


def build_final_rmse_tables(
    runs: pd.DataFrame,
    *,
    output_dir: str | Path,
    cache_dir: Path,
    n: int,
    expected_num_seeds: int | None,
    strict_seed_count: bool,
    rebuild_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute final-timestep RMSE values at the common fixed horizon n."""
    rmse_lookup, rmse_cache_path = _build_final_rmse_lookup(
        cache_dir=cache_dir,
        rebuild_cache=rebuild_cache,
    )

    per_seed_rows: list[dict[str, object]] = []
    cache_rows: list[dict[str, object]] = []
    cache_hits = 0
    cache_misses = 0

    for group_values, group in runs.groupby(GROUP_COLS, dropna=False, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        meta = dict(zip(GROUP_COLS, group_values))

        for row in group.sort_values("seed").itertuples(index=False):
            path_str, size, mtime = _signature_key(row)
            key = (path_str, size, mtime, int(n))
            if key in rmse_lookup:
                rmse = rmse_lookup[key]
                cache_hits += 1
            else:
                rmse = _final_rmse_from_csv(Path(path_str), int(n))
                cache_misses += 1

            per_seed_rows.append(
                {
                    **meta,
                    "seed": int(row.seed),
                    "n": int(n),
                    "observed_rows": int(row.file_length),
                    "rmse_at_n": float(rmse),
                    "csv_path": path_str,
                }
            )
            cache_rows.append(
                {
                    "csv_path_resolved": path_str,
                    "file_size": size,
                    "mtime_ns": mtime,
                    "t_max": int(n),
                    "rmse_at_t_max": float(rmse),
                }
            )

    _save_csv_cache(pd.DataFrame(cache_rows).drop_duplicates(), rmse_cache_path)
    print(f"Final-RMSE cache: {cache_hits} hit(s), {cache_misses} miss(es).")

    per_seed = pd.DataFrame(per_seed_rows).sort_values(GROUP_COLS + ["seed"])
    summary = (
        per_seed.groupby(GROUP_COLS + ["n"], dropna=False, as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_rmse=("rmse_at_n", "mean"),
            std_rmse_across_seeds=("rmse_at_n", "std"),
        )
        .sort_values(["eps", "xi", "b", "C_kind"])
    )

    if expected_num_seeds is not None:
        bad = summary.loc[summary["n_seeds"] != int(expected_num_seeds)]
        if not bad.empty:
            message = (
                "Some exact parameter groups do not have the expected number "
                "of seeds. Available seeds will still be used:\n"
                + bad[
                    ["eps", "xi", "n", "b", "C_kind", "n_seeds"]
                ].to_string(index=False)
            )
            if strict_seed_count:
                raise ValueError(message)
            warnings.warn(message)

    table_index = [
        "n",
        "xi",
        "delta",
        "clip_lower",
        "clip_upper",
        "C_kind",
        "eps",
    ]
    final_table = summary.pivot_table(
        index=table_index,
        columns="b",
        values="mean_rmse",
        aggfunc="first",
        dropna=False,
    ).sort_index()
    final_table.columns.name = "b"

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out / "final_rmse_per_seed.csv", index=False)
    summary.to_csv(out / "final_rmse_summary.csv", index=False)
    final_table.to_csv(out / "final_rmse_eps_by_b.csv", index=True)
    return per_seed, summary, final_table

def _numeric_mask(series: pd.Series, value: float) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return np.isclose(values, float(value), rtol=0.0, atol=1e-12)


def _requested_or_available(
    requested: Sequence[float] | None,
    available: pd.Series,
) -> list[float]:
    if requested is not None:
        return [float(value) for value in requested]
    return sorted(
        float(value)
        for value in pd.to_numeric(available, errors="coerce").dropna().unique()
    )


def print_final_rmse_tables(
    summary: pd.DataFrame,
    *,
    requested_eps_values: Sequence[float] | None,
    requested_xi_values: Sequence[float] | None,
    requested_b_values: Sequence[int] | None,
) -> None:
    """Print epsilon-by-b final-RMSE tables.

    One table is printed for each fixed xi, delta, clipping interval, and
    method. Rows are epsilon values and columns are b values.
    """
    eps_values = _requested_or_available(requested_eps_values, summary["eps"])
    xi_values = _requested_or_available(requested_xi_values, summary["xi"])
    if requested_b_values is None:
        b_values = sorted(int(value) for value in summary["b"].dropna().unique())
    else:
        b_values = [int(value) for value in requested_b_values]

    print("\n" + "=" * 88)
    print("FINAL-TIMESTEP RMSE TABLES: EPSILON BY b")
    print("Rows are epsilon values; columns are b values.")
    print("Each entry averages RMSE_seed(n) over the available seeds.")
    if summary.empty:
        print("No RMSE rows are available.")
    else:
        evaluation_n = int(summary["n"].iloc[0])
        print(f"fixed n={evaluation_n}; every selected CSV has this length.")
    print("=" * 88)

    for xi in xi_values:
        xi_block = summary.loc[_numeric_mask(summary["xi"], xi)]
        if xi_block.empty:
            print("\n" + "-" * 88)
            print(f"xi={_token(xi)}: [NO MATCHING EXPERIMENTS FOUND]")
            continue

        for fixed_values, config_block in xi_block.groupby(
            ["delta", "clip_lower", "clip_upper"],
            dropna=False,
            sort=True,
        ):
            delta, clip_lower, clip_upper = fixed_values

            for method in TABLE_METHODS:
                block = config_block.loc[config_block["C_kind"] == method]
                print("\n" + "-" * 88)
                print(
                    f"method={method}, xi={_token(xi)}, delta={_token(delta)}, "
                    f"clip=[{_token(clip_lower)}, {_token(clip_upper)}]"
                )
                if block.empty:
                    print("[NO MATCHING EXPERIMENTS FOUND]")
                    continue

                table = block.pivot_table(
                    index="eps",
                    columns="b",
                    values="mean_rmse",
                    aggfunc="first",
                )
                table = table.reindex(index=eps_values, columns=b_values)
                table.index.name = "epsilon"
                table.columns.name = "b"
                print(table.to_string(float_format=lambda x: f"{x:.8g}"))

def _read_abs_error(path: Path, t_max: int) -> np.ndarray:
    columns = set(pd.read_csv(path, nrows=0).columns)
    if not {TRUE_COL, PRIVATE_COL}.issubset(columns):
        raise ValueError(
            f"{path} must contain '{TRUE_COL}' and '{PRIVATE_COL}' "
            "for the average absolute-error plot."
        )
    frame = pd.read_csv(path, usecols=[TRUE_COL, PRIVATE_COL], nrows=t_max)
    if len(frame) < t_max:
        raise RuntimeError(f"{path} has fewer than {t_max} rows")
    return np.abs(
        frame[TRUE_COL].to_numpy(dtype=float)
        - frame[PRIVATE_COL].to_numpy(dtype=float)
    )


def _mean_and_ci(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.stack(curves, axis=0)
    mean = matrix.mean(axis=0)
    if matrix.shape[0] > 1:
        sem = matrix.std(axis=0, ddof=1) / math.sqrt(matrix.shape[0])
    else:
        sem = np.zeros_like(mean)
    low = np.maximum(mean - 1.96 * sem, 0.0)
    high = mean + 1.96 * sem
    return mean, low, high


def _plot_indices(length: int, max_points: int, growth: float = 1.07) -> np.ndarray:
    if length <= max_points:
        return np.arange(length, dtype=int)
    indices = [0]
    gap = 1.0
    current = 0
    while current + max(1, int(gap)) < length and len(indices) < max_points:
        current += max(1, int(gap))
        indices.append(current)
        gap *= growth
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return np.asarray(sorted(set(indices)), dtype=int)


def _apply_plot_style(legend_fontsize: int = 20) -> None:
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "legend.fontsize": legend_fontsize,
            "font.size": 26,
            "axes.labelsize": 26,
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
            "text.latex.preamble": (
                r"\usepackage{amsmath} "
                r"\usepackage{amssymb} "
                r"\usepackage{amsfonts}"
            ),
        }
    )
    plt.rcParams["pdf.fonttype"] = 42


def _plot_cache_fingerprint(block: pd.DataFrame, t_max: int) -> str:
    payload = {
        "version": 1,
        "t_max": int(t_max),
        "rows": [
            {
                "path": str(Path(row.csv_path_resolved)),
                "size": int(row.file_size),
                "mtime_ns": int(row.mtime_ns),
                "seed": int(row.seed),
                "method": str(row.C_kind),
            }
            for row in block.sort_values(["C_kind", "seed"]).itertuples(index=False)
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_or_build_abs_plot_stats(
    block: pd.DataFrame,
    *,
    n: int,
    cache_dir: Path,
    rebuild_cache: bool,
) -> tuple[int, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]], bool]:
    """Return per-method mean/CI curves and whether the aggregate cache hit."""
    # Every selected CSV is required to have the same fixed horizon.
    t_max = int(n)
    fingerprint = _plot_cache_fingerprint(block, t_max)
    cache_path = cache_dir / "abs_plot_aggregates" / f"{fingerprint}.npz"

    if cache_path.exists() and not rebuild_cache:
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                methods = [str(value) for value in data["methods"].tolist()]
                stats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
                for index, method in enumerate(methods):
                    stats[method] = (
                        data[f"mean_{index}"],
                        data[f"low_{index}"],
                        data[f"high_{index}"],
                        int(data[f"n_{index}"][0]),
                    )
                return int(data["t_max"][0]), stats, True
        except Exception as exc:
            warnings.warn(f"Ignoring unreadable plot cache {cache_path}: {exc}")

    stats = {}
    for method, method_rows in block.groupby("C_kind", sort=True):
        curves = [
            _read_abs_error(Path(row.csv_path_resolved), t_max)
            for row in method_rows.sort_values("seed").itertuples(index=False)
        ]
        mean, low, high = _mean_and_ci(curves)
        stats[str(method)] = (mean, low, high, len(curves))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    methods = sorted(stats)
    payload: dict[str, np.ndarray] = {
        "methods": np.asarray(methods, dtype="U64"),
        "t_max": np.asarray([t_max], dtype=np.int64),
    }
    for index, method in enumerate(methods):
        mean, low, high, n_seeds = stats[method]
        payload[f"mean_{index}"] = mean
        payload[f"low_{index}"] = low
        payload[f"high_{index}"] = high
        payload[f"n_{index}"] = np.asarray([n_seeds], dtype=np.int64)

    temp = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **payload)
    temp.replace(cache_path)
    return t_max, stats, False


def make_avg_abs_error_plots(
    runs: pd.DataFrame,
    *,
    n: int,
    plot_dir: str | Path,
    cache_dir: Path,
    show_ci: bool,
    max_plot_points: int,
    rebuild_cache: bool,
) -> None:
    """Compare methods for fixed b, epsilon, xi, delta, and clipping interval."""
    out = Path(plot_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_count = 0
    cache_hits = 0
    cache_misses = 0

    for fixed_values, block in runs.groupby(
        FIXED_ABS_PLOT_COLS,
        dropna=False,
        sort=True,
    ):
        fixed = dict(zip(FIXED_ABS_PLOT_COLS, fixed_values))
        t_max, method_stats, cache_hit = _load_or_build_abs_plot_stats(
            block,
            n=n,
            cache_dir=cache_dir,
            rebuild_cache=rebuild_cache,
        )
        cache_hits += int(cache_hit)
        cache_misses += int(not cache_hit)

        _apply_plot_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        idx = _plot_indices(t_max, max_plot_points)
        x = np.arange(1, t_max + 1, dtype=float)[idx]

        for method in sorted(method_stats):
            mean, low, high, _n_seeds = method_stats[method]
            if method == "BandMF":
                ax.plot(
                    x,
                    mean[idx],
                    label=_method_label(method),
                    linewidth=2.5,
                    color="tab:blue",
                    linestyle="--",
                )
                if show_ci:
                    ax.fill_between(
                        x, low[idx], high[idx], alpha=0.15, color="tab:blue"
                    )
            else:
                ax.plot(
                    x,
                    mean[idx],
                    label=_method_label(method),
                    linewidth=2.0,
                )
                if show_ci:
                    ax.fill_between(x, low[idx], high[idx], alpha=0.25)

        ax.set_xscale("log")
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_locator(ticker.LogLocator(base=2.0))
        ax.set_xlabel("Timestep")
        ax.set_ylabel(
            r"$|\mu_t-\widehat{\mu}_t|$"
        )
        ax.legend(loc="upper right")
        fig.tight_layout(pad=0)

        suffix = "_".join(
            [
                f"n{int(fixed['n_arrivals'])}",
                f"b{int(fixed['b'])}",
                f"eps{_token(fixed['eps'])}",
                f"delta{_token(fixed['delta'])}",
                f"clipL{_token(fixed['clip_lower'])}",
                f"clipU{_token(fixed['clip_upper'])}",
                f"xi{_token(fixed['xi'])}",
            ]
        )
        base = out / f"avg_abs_error_methods_{suffix}"
        fig.savefig(base.with_suffix(".pdf"), format="pdf")
        fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
        plt.close(fig)
        plot_count += 1
        print(
            f"Saved average-absolute-error plot: {base.with_suffix('.pdf')} "
            f"(T={t_max})"
        )

    print(f"\nGenerated {plot_count} average-absolute-error method-comparison plots.")
    print(f"Absolute-error aggregate cache: {cache_hits} hit(s), {cache_misses} miss(es).")


def analyze_uci_results(
    results_dir: str | Path = RESULTS_DIR,
    output_dir: str | Path | None = None,
    plot_dir: str | Path = PLOT_DIR,
    cache_dir: str | Path | None = None,
    n: int = N,
    expected_num_seeds: int | None = 10,
    *,
    strict_seed_count: bool = False,
    rebuild_cache: bool = False,
    b_values: Sequence[int] | None = B_VALUES,
    eps_values: Sequence[float] | None = EPS_VALUES,
    xi_values: Sequence[float] | None = XI_VALUES,
    delta_values: Sequence[float] | None = DELTA_VALUES,
    methods: Sequence[str] | None = METHODS,
    show_ci: bool = True,
    max_plot_points: int = 300,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = discover_runs(results_dir)
    runs = filter_runs(
        runs,
        n=n,
        b_values=b_values,
        eps_values=eps_values,
        xi_values=xi_values,
        delta_values=delta_values,
        methods=methods,
    )

    print(
        "Selected run counts by n, epsilon, xi, b, and method:\n"
        + runs.groupby(["n_arrivals", "eps", "xi", "b", "C_kind"]).size().to_string()
    )

    cache_root = (
        Path(cache_dir)
        if cache_dir is not None
        else Path(results_dir) / CACHE_DIR_NAME
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    runs = attach_actual_file_lengths(
        runs,
        cache_dir=cache_root,
        rebuild_cache=rebuild_cache,
    )
    if (runs["file_length"] <= 0).any():
        bad = runs.loc[runs["file_length"] <= 0, "csv_path_resolved"]
        raise ValueError("Empty experiment CSVs found:\n" + "\n".join(map(str, bad)))

    wrong_length = runs.loc[runs["file_length"].astype(int) != int(n)]
    if not wrong_length.empty:
        raise ValueError(
            f"All selected files must contain fixed n={int(n)} rows. Bad files:\n"
            + wrong_length[
                ["eps", "xi", "b", "C_kind", "seed", "n_arrivals", "file_length", "csv_path"]
            ].to_string(index=False)
        )

    out = Path(output_dir) if output_dir is not None else Path(results_dir)
    per_seed, summary, final_table = build_final_rmse_tables(
        runs,
        output_dir=out,
        cache_dir=cache_root,
        n=n,
        expected_num_seeds=expected_num_seeds,
        strict_seed_count=strict_seed_count,
        rebuild_cache=rebuild_cache,
    )
    print_final_rmse_tables(
        summary,
        requested_eps_values=eps_values,
        requested_xi_values=xi_values,
        requested_b_values=b_values,
    )
    make_avg_abs_error_plots(
        runs,
        n=n,
        plot_dir=plot_dir,
        cache_dir=cache_root,
        show_ci=show_ci,
        max_plot_points=max_plot_points,
        rebuild_cache=rebuild_cache,
    )

    print(f"\nSaved final RMSE tables under: {out}")
    print(f"Saved average absolute-error plots under: {plot_dir}")
    print(f"Analysis cache directory: {cache_root}")
    return per_seed, summary, final_table


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print final-timestep UCI RMSE tables and generate "
            "average absolute-error method-comparison plots."
        )
    )
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--plot-dir", default=PLOT_DIR)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Analysis cache directory. Defaults to <results-dir>/.analysis_cache.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help="Fixed raw-arrival/publication horizon encoded in every selected UCI CSV.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=10,
        help=(
            "Expected seeds per exact group. Incomplete groups produce a warning "
            "and are still analyzed. Use -1 to disable the check."
        ),
    )
    parser.add_argument(
        "--strict-seed-count",
        action="store_true",
        help="Raise an error instead of warning when a group has fewer seeds.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore cached analysis values and rebuild them from the CSVs.",
    )
    parser.add_argument("--b-values", nargs="*", type=int, default=B_VALUES)
    parser.add_argument("--eps-values", nargs="+", type=float, default=EPS_VALUES)
    parser.add_argument("--xi-values", nargs="+", type=float, default=XI_VALUES)
    parser.add_argument("--delta-values", nargs="*", type=float, default=DELTA_VALUES)
    parser.add_argument("--methods", nargs="*", default=METHODS)
    parser.add_argument("--no-ci", action="store_true")
    parser.add_argument("--max-plot-points", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    expected = None if args.expected_seeds < 0 else args.expected_seeds
    analyze_uci_results(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        plot_dir=args.plot_dir,
        cache_dir=args.cache_dir,
        n=args.n,
        expected_num_seeds=expected,
        strict_seed_count=args.strict_seed_count,
        rebuild_cache=args.rebuild_cache,
        b_values=args.b_values,
        eps_values=args.eps_values,
        xi_values=args.xi_values,
        delta_values=args.delta_values,
        methods=args.methods,
        show_ci=not args.no_ci,
        max_plot_points=args.max_plot_points,
    )
