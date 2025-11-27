import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator

def set_xlabels_as_times_1e4(ax, n, step=5):
    kmax = int(np.ceil(n / 1e4))
    if kmax <= step:
        ax.set_xlabel("Timesteps")
        return

    labels = list(range(step, kmax + 1, step))
    ticks  = (1e4 * np.array(labels)).astype(int)

    ax.xaxis.set_major_locator(FixedLocator(ticks))

    def _fmt(x, pos):
        k = x / 1e4
        if np.isclose(k, round(k)) and (round(k) in labels):
            return fr"${int(round(k))}\times 10^{{4}}$"
        return ""  

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt))

    ax.set_xlabel("Timesteps")


def log_thin_indices(n: int, dense_first: int = 2000, per_decade: int = 200):
    if n <= 0:
        return np.array([], dtype=int)
    idxs = set(range(min(dense_first, n)))
    if n > dense_first + 1:
        start = max(dense_first, 1)
        a, b = start, n - 1
        if b > a:
            decades = max(1, int(math.ceil(math.log10((b) / max(1, a)))))
            left = a
            for d in range(decades):
                right = min(b, int(a * (10 ** (d + 1))))
                if right <= left:
                    continue
                k = max(2, per_decade)
                pts = np.unique(np.logspace(np.log10(left), np.log10(right), num=k).astype(int))
                idxs.update(pts.tolist())
                left = right
    idxs.add(0); idxs.add(n - 1)
    return np.array(sorted(idxs), dtype=int)

def plot_three_running_means(
    csv_a1_sqrt: str,
    csv_dtoep: str,
    output_pdf: str = "plots/running_means_combined.pdf",
    dense_first: int = 2000,
    per_decade: int = 200,
    use_latex: bool = True,
    log_x: bool = True,
    y_label: str = r"Running mean"
):
    """
    Make a single plot with:
      - True running mean  (dashed)
      - A1_sqrt            (solid)
      - Dtoep              (solid)
    The two CSVs should each contain columns: 'true_running_mean' and 'private_running_mean'.
    The first CSV is used as the canonical 'true' series.
    """
    import matplotlib.ticker as ticker

    df_a1 = pd.read_csv(csv_a1_sqrt)
    df_dt = pd.read_csv(csv_dtoep)

    required = {"true_running_mean", "private_running_mean"}
    if not required.issubset(df_a1.columns):
        raise ValueError(f"A1_sqrt CSV must contain {required}. Found: {list(df_a1.columns)}")
    if not required.issubset(df_dt.columns):
        raise ValueError(f"Dtoep CSV must contain {required}. Found: {list(df_dt.columns)}")

    true_rm = df_a1["true_running_mean"].to_numpy()
    a1_rm   = df_a1["private_running_mean"].to_numpy()
    dt_rm   = df_dt["private_running_mean"].to_numpy()

    # Ensure same length (trim to the shortest if needed)
    n = min(len(true_rm), len(a1_rm), len(dt_rm))
    true_rm = true_rm[:n]
    a1_rm   = a1_rm[:n]
    dt_rm   = dt_rm[:n]

    # (Optional) sanity check: true series in both files roughly matches
    # If you want to enforce this strictly, uncomment:
    # if np.max(np.abs(true_rm - df_dt["true_running_mean"].to_numpy()[:n])) > 1e-8:
    #     print("Warning: true_running_mean differs between CSVs (using the first CSV as canonical).")

    x_full = np.arange(1, n + 1, dtype=int)
    keep = log_thin_indices(n, dense_first=dense_first, per_decade=per_decade)
    x = x_full[keep]
    y_true = true_rm[keep]
    y_a1   = a1_rm[keep]
    y_dt   = dt_rm[keep]

    plt.rcParams.update({
        "text.usetex": bool(use_latex),
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "legend.fontsize": 24,
        "font.size": 26,
        "axes.labelsize": 26,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}",
        "pdf.fonttype": 42,
    })

    plt.figure(figsize=(10, 6))
    # if log_x:
    #     plt.xscale("log", base=10)

    plt.plot(x, y_true, linestyle="--", linewidth=1.2, label="Running mean")
    plt.plot(x, y_a1,   linewidth=1.2, label="$\\mathbf{E}_1^{1/2}$")
    plt.plot(x, y_dt,   linewidth=1.2, label="$\\mathbf{D}_{\\mathrm{Toep}}$")

    ax = plt.gca()
    set_xlabels_as_times_1e4(ax, n, step=5)
    ax.minorticks_on()
    # plt.xlabel("Timestep")
    plt.ylabel(y_label)
    plt.ylim(top=100, bottom=40)
    plt.legend()
    plt.tight_layout(pad=0)

    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
    plt.savefig(output_pdf, format="pdf")
    plt.close()
    print(f"Saved plot to: {output_pdf}")

# --------------------------- Example call with your paths

plot_three_running_means(
    csv_a1_sqrt="cache/mat_fact_algo_csv/running_means_b500_k389_p16_eps10.0_delta5e-06_xi1000_CA1_sqrt.csv",
    csv_dtoep="cache/mat_fact_algo_csv/running_means_b500_k389_p16_eps10.0_delta5e-06_xi1000_CDtoep.csv",
    output_pdf="plots/running_means_b500_k389_p16_eps10.0_delta5e-06_xi1000_combined.pdf",
    dense_first=2000,
    per_decade=200,
    use_latex=True,
    log_x=True,
    y_label=r"Amount (\$)"
)
