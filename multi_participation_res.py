#!/usr/bin/env python3
# Read-only cache printer. Change EXP below and run.
# It expects the same cache layout you already use:
#   cache/<n>/<kind>/<b>/<p>/<name>.pkl
# p is chosen as p = max(1, int(log2(b))) to mirror your plotting code.

from pathlib import Path
import pickle
import math

# === EDIT THIS ONLY ===
EXP = 13                 # n = 2**EXP
K_VALUES = [4, 16, 64]   # which k’s to show
INCLUDE_RATIOS = True    # print value / (BSR i=2, Dtoep)
PRECISION = 6            # printed digits
SHOW_PATHS = False       # also print file paths
# ======================

CACHE_ROOT = Path("cache")

TASKS = (
    ("nonbanded", "i1"),
    ("nonbanded", "i2"),
    ("nonbanded", "i3"),
    ("nonbanded", "i4"),
    ("banded inverse", "bisr_i1"),
    ("banded inverse", "bisr_i2"),
    ("banded inverse", "bisr_i3"),
    ("banded", "bsr_i1"),
    ("banded", "bsr_i2"),
    ("banded", "bsr_i3"),
)

def _fmt_value(v):
    s = f"{v:.12g}" if isinstance(v, float) else str(v)
    return (s.replace(" ", "")
             .replace("/", "_per_")
             .replace("\\", "_")
             .replace(":", "_"))

def _val_path(n, kind, b, p, name):
    # READ-ONLY: do NOT create directories
    return CACHE_ROOT / str(n) / kind / _fmt_value(b) / _fmt_value(p) / f"{name}.pkl"

def has_value(*, n, kind, b, p, name):
    return _val_path(n, kind, b, p, name).exists()

def load_value(*, n, kind, b, p, name):
    with open(_val_path(n, kind, b, p, name), "rb") as f:
        return float(pickle.load(f))

def pick_p(b):
    # default behavior from your plotter: p = max(1, int(log2(b)))
    return max(1, int(math.log2(b))) if b > 0 else 1
    # return b

def header(include_ratios):
    h = f"{'kind':<15} {'name':<10} {'value':>16}"
    if include_ratios:
        h += f" {'/ bsr_i2':>14}"
    return h

def main():
    n = 2 ** int(EXP)
    ks = sorted(set(int(k) for k in K_VALUES))

    print(f"\n=== Cached results for EXP={EXP} (n={n}) ===")
    missing_any = False

    for k in ks:
        if k > n:
            continue
        b = n // k
        p = pick_p(b)

        # denominator for ratios: BSR i=2 (Dtoep)
        denom = None
        if INCLUDE_RATIOS and has_value(n=n, kind="banded", b=b, p=p, name="bsr_i2"):
            denom = load_value(n=n, kind="banded", b=b, p=p, name="bsr_i2")

        title = f"-- k={k}, b={b}, p={p} --"
        print("\n" + title)
        print("=" * len(title))
        head = header(INCLUDE_RATIOS)
        print(head)
        print("-" * len(head))

        for (kind, name) in TASKS:
            if not has_value(n=n, kind=kind, b=b, p=p, name=name):
                missing_any = True
                v_str = "—"
                r_str = "—" if INCLUDE_RATIOS else ""
            else:
                v = load_value(n=n, kind=kind, b=b, p=p, name=name)
                v_str = f"{v:.{PRECISION}g}"
                if INCLUDE_RATIOS and denom not in (None, 0.0):
                    r_str = f"{(v/denom):.{PRECISION}g}"
                elif INCLUDE_RATIOS:
                    r_str = "—"
                else:
                    r_str = ""

            line = f"{kind:<15} {name:<10} {v_str:>16}"
            if INCLUDE_RATIOS:
                line += f" {r_str:>14}"
            print(line)

            if SHOW_PATHS:
                print(f"    path: {_val_path(n, kind, b, p, name)}")

    if missing_any:
        print("\nSome entries were missing in cache (shown as '—').")
    else:
        print("\nAll requested entries were found in cache.")

if __name__ == "__main__":
    main()
