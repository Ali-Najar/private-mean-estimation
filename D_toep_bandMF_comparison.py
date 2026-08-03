import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ------------------------------------------------------------------
# Uses your existing functions:
#   inv_series
#   seq_Dtoep
#   compute_BandMF_g
# ------------------------------------------------------------------

CACHE_DIR_G = "cache/g"
CACHE_DIR_C = "cache/c"
os.makedirs(CACHE_DIR_G, exist_ok=True)
os.makedirs(CACHE_DIR_C, exist_ok=True)


def seq_Dtoep(n):
    i = np.arange(n, dtype=float)
    return 1.0 / (i + 1.0)


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


# ------------------------------------------------------------------
# Cache helpers
# ------------------------------------------------------------------

def make_power_of_two_n_values(min_exp=1, max_exp=10):
    return [2**e for e in range(min_exp, max_exp + 1)]


def get_bandmf_g_cache_path(n, p, steps):
    return os.path.join(CACHE_DIR_G, f"g_BandMF_n{n}_p{p}_steps{steps}.npy")


def get_bandmf_c_cache_path(n, p, steps):
    return os.path.join(CACHE_DIR_C, f"c_BandMF_n{n}_p{p}_steps{steps}.npy")


def get_bandmf_g_for_n(n, p, use_cache=True, steps=25):
    """
    Return BandMF inverse coefficients g for this (n, p).
    Cached per (n, p, steps).
    """
    fname = get_bandmf_g_cache_path(n, p, steps)

    if use_cache and os.path.isfile(fname):
        try:
            return np.load(fname)
        except Exception:
            pass

    g = compute_BandMF_g(n, p, steps=steps)

    tmp = fname + ".tmp.npy"
    np.save(tmp, g)
    os.replace(tmp, fname)
    return g


def get_bandmf_c_for_n(n, p, use_cache=True, steps=25):
    """
    Return reconstructed BandMF coefficients c = inv_series(g, n),
    cached per (n, p, steps).
    """
    fname = get_bandmf_c_cache_path(n, p, steps)

    if use_cache and os.path.isfile(fname):
        try:
            return np.load(fname)
        except Exception:
            pass

    g = get_bandmf_g_for_n(n, p, use_cache=use_cache, steps=steps)
    c = inv_series(g, n)

    tmp = fname + ".tmp.npy"
    np.save(tmp, c)
    os.replace(tmp, fname)
    return c


def get_dtoep_g_for_n(n, p):
    c_dtoep = seq_Dtoep(n)
    g_dtoep = inv_series(c_dtoep, p)
    return g_dtoep


# ------------------------------------------------------------------
# Comparison plots
# ------------------------------------------------------------------

def compare_bandmf_vs_dtoep_for_n_list(
    n_list,
    p_rule=None,
    steps=25,
    use_cache=True,
    save_prefix="plots/bandmf_vs_dtoep_across_n",
):
    os.makedirs("plots", exist_ok=True)

    if p_rule is None:
        p_rule = lambda n: min(64, n)

    l2_diffs = []
    linf_diffs = []
    used_ps = []

    fig, axs = plt.subplots(len(n_list), 2, figsize=(12, 3.5 * len(n_list)))
    if len(n_list) == 1:
        axs = np.array([axs])

    for row, n in enumerate(n_list):
        print(f"Processing inverse coefficients for n={n}")
        p = int(p_rule(n))
        used_ps.append(p)

        g_bandmf = get_bandmf_g_for_n(n, p, use_cache=use_cache, steps=steps)
        g_dtoep = get_dtoep_g_for_n(n, p)

        diff = g_bandmf - g_dtoep
        idx = np.arange(p)

        l2 = np.linalg.norm(diff)
        linf = np.max(np.abs(diff))

        l2_diffs.append(l2)
        linf_diffs.append(linf)

        axs[row, 0].plot(idx, g_dtoep, label=r"$g_{\mathrm{Dtoep}}$", linewidth=1.8)
        axs[row, 0].plot(idx, g_bandmf, label=r"$g_{\mathrm{BandMF}}$", linewidth=1.8, linestyle="--")
        axs[row, 0].set_title(f"Inverse coefficients, n={n}, p={p}")
        axs[row, 0].set_xlabel("Coefficient index")
        axs[row, 0].legend()

        axs[row, 1].plot(idx, diff, linewidth=1.8)
        axs[row, 1].axhline(0.0, linestyle="--", linewidth=1.0)
        axs[row, 1].set_title(
            f"Difference, n={n}, p={p}\n"
            f"$\\|\\Delta\\|_2$={l2:.3e}, "
            f"$\\|\\Delta\\|_\\infty$={linf:.3e}"
        )
        axs[row, 1].set_xlabel("Coefficient index")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_coeffs.pdf", format="pdf")
    plt.show()

    n_arr = np.array(n_list, dtype=int)

    plt.figure(figsize=(10, 6))
    plt.plot(n_arr, l2_diffs, marker="o", linewidth=2, label=r"$\|g_{\mathrm{BandMF}}-g_{\mathrm{Dtoep}}\|_2$")
    plt.plot(n_arr, linf_diffs, marker="s", linewidth=2, label=r"$\|g_{\mathrm{BandMF}}-g_{\mathrm{Dtoep}}\|_\infty$")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("Difference size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_norms_vs_n.pdf", format="pdf")
    plt.show()

    print("n\tp\tL2 diff\t\tLinf diff")
    for n, p, l2, linf in zip(n_list, used_ps, l2_diffs, linf_diffs):
        print(f"{n}\t{p}\t{l2:.6e}\t{linf:.6e}")


def compare_reconstructed_C_for_n_list(
    n_list,
    p_rule=None,
    steps=25,
    first_m=100,
    use_cache=True,
    save_prefix="plots/bandmf_vs_dtoep_reconstructedC",
):
    os.makedirs("plots", exist_ok=True)

    if p_rule is None:
        p_rule = lambda n: min(64, n)

    fig, axs = plt.subplots(len(n_list), 2, figsize=(12, 3.5 * len(n_list)))
    if len(n_list) == 1:
        axs = np.array([axs])

    l2_diffs = []
    linf_diffs = []
    used_ps = []

    for row, n in enumerate(n_list):
        print(f"Processing reconstructed C for n={n}")
        p = int(p_rule(n))
        used_ps.append(p)

        c_bandmf = get_bandmf_c_for_n(n, p, use_cache=use_cache, steps=steps)
        c_dtoep = seq_Dtoep(n)

        m = min(first_m, n)
        idx = np.arange(m)
        diff = c_bandmf[:m] - c_dtoep[:m]

        l2 = np.linalg.norm(diff)
        linf = np.max(np.abs(diff))

        l2_diffs.append(l2)
        linf_diffs.append(linf)

        axs[row, 0].plot(idx, c_dtoep[:m], label=r"$c_{\mathrm{Dtoep}}$", linewidth=1.8)
        axs[row, 0].plot(idx, c_bandmf[:m], label=r"$c_{\mathrm{BandMF}}$", linewidth=1.8, linestyle="--")
        axs[row, 0].set_title(f"Reconstructed C coefficients, n={n}, p={p}")
        axs[row, 0].set_xlabel("Coefficient index")
        axs[row, 0].legend()

        axs[row, 1].plot(idx, diff, linewidth=1.8)
        axs[row, 1].axhline(0.0, linestyle="--", linewidth=1.0)
        axs[row, 1].set_title(
            f"Difference, first {m} coeffs, n={n}, p={p}\n"
            f"$\\|\\Delta\\|_2$={l2:.3e}, "
            f"$\\|\\Delta\\|_\\infty$={linf:.3e}"
        )
        axs[row, 1].set_xlabel("Coefficient index")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_coeffs.pdf", format="pdf")
    plt.show()

    n_arr = np.array(n_list, dtype=int)

    plt.figure(figsize=(10, 6))
    plt.plot(n_arr, l2_diffs, marker="o", linewidth=2, label=r"$\|c_{\mathrm{BandMF}}-c_{\mathrm{Dtoep}}\|_2$")
    plt.plot(n_arr, linf_diffs, marker="s", linewidth=2, label=r"$\|c_{\mathrm{BandMF}}-c_{\mathrm{Dtoep}}\|_\infty$")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("Difference size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_norms_vs_n.pdf", format="pdf")
    plt.show()

    print("n\tp\tL2 diff\t\tLinf diff")
    for n, p, l2, linf in zip(n_list, used_ps, l2_diffs, linf_diffs):
        print(f"{n}\t{p}\t{l2:.6e}\t{linf:.6e}")


# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------

n_values = make_power_of_two_n_values(min_exp=1, max_exp=12)

compare_bandmf_vs_dtoep_for_n_list(
    n_list=n_values,
    p_rule=lambda n: min(64, n),
    steps=25,
    use_cache=True,
    save_prefix="plots/bandmf_vs_dtoep_inverse_across_n",
)

# compare_reconstructed_C_for_n_list(
#     n_list=n_values,
#     p_rule=lambda n: min(64, n),
#     steps=25,
#     first_m=100,
#     use_cache=True,
#     save_prefix="plots/bandmf_vs_dtoep_C_across_n",
# )