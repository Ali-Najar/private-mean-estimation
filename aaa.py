from pathlib import Path
import re

bandmf_dir = Path("cache/mat_fact_algo/BandMF/mu0.5")

pattern = re.compile(
    r"^(sum_sqerr_EXP19_k128_b4096_)p16(_eps1(?:\.0)?_delta1e-06_xi1(?:\.0)?_seed\d+\.npy)$"
)

count = 0
for path in bandmf_dir.glob("sum_sqerr_EXP19_k128_b4096_p16_eps1*_delta1e-06_xi1*_seed*.npy"):
    m = pattern.match(path.name)
    if not m:
        print(f"Skipping: {path.name}")
        continue

    new_name = f"{m.group(1)}p512{m.group(2)}"
    new_path = path.with_name(new_name)

    if new_path.exists():
        print(f"Target exists, skipping: {new_path.name}")
        continue

    path.rename(new_path)
    print(f"Renamed: {path.name} -> {new_path.name}")
    count += 1

print(f"Done. Renamed {count} files.")