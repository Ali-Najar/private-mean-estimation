from __future__ import annotations
import math
import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional


def laplace(b: float) -> float:
    if b <= 0:
        return 0.0
    u = random.random()  # in [0,1)
    if u < 0.5:
        return b * math.log(max(2 * u, 1e-300))  
    else:
        return -b * math.log(max(2 * (1 - u), 1e-300))  


class BinaryMechanism:
    def __init__(self, noise_b: float):
        self.noise_b = float(noise_b)
        self.stream: List[float] = []  
        self.noisy_partial: Dict[int, float] = {}

    @staticmethod
    def _lowbit(x: int) -> int:
        return x & -x

    def add(self, v: float) -> None:
        k = len(self.stream) + 1 
        self.stream.append(v)
        block = self._lowbit(k)
        s = 0.0
        for i in range(k - block, k):
            s += self.stream[i]
        s_noisy = s + laplace(self.noise_b)
        self.noisy_partial[k] = s_noisy

    def prefix_sum(self) -> float:
        k = len(self.stream)
        idx = k
        total = 0.0
        while idx > 0:
            total += self.noisy_partial.get(idx, 0.0)
            idx -= self._lowbit(idx)
        return total


class PrivateMedian:

    def __init__(self, eps_prime: float, ell: int, beta: float):
        self.eps_prime = float(eps_prime)
        self.ell = int(ell)
        self.beta = float(beta)

    @staticmethod
    def _k_required(eps_prime: float, ell: int, beta: float) -> int:
        num = (2 ** ell) / 2.0
        k = math.ceil((16.0 / max(eps_prime, 1e-12)) * math.log(max(num / max(beta, 1e-300), 1.0)))
        return max(1, k)

    @staticmethod
    def _bin_midpoints(ell: int) -> List[float]:
        width = 2.0 * (2.0 ** (-ell / 2.0))
        if width <= 0:
            return [0.5]
        m = max(1, int(math.ceil(1.0 / width)))
        mids = []
        for i in range(m):
            a = i * width
            b = min(1.0, (i + 1) * width)
            mids.append((a + b) / 2.0)
        if mids[0] > 0:
            mids.insert(0, 0.5 * width)
        if mids[-1] < 1:
            mids.append(min(1.0, mids[-1] + width))
        return mids

    @staticmethod
    def _nearest(x: float, grid: List[float]) -> float:
        best = grid[0]
        bestd = abs(x - best)
        for g in grid[1:]:
            d = abs(x - g)
            if d < bestd:
                bestd = d
                best = g
        return best

    def run(self, user_samples: Dict[int, List[float]]) -> Optional[float]:
        ell = self.ell
        size = 2 ** max(ell - 1, 0)  
        if size <= 0:
            return 0.5
        k = self._k_required(self.eps_prime, ell, self.beta)

        arrays: List[List[float]] = []
        cur: List[float] = []
        for u in sorted(user_samples.keys()):
            s = user_samples[u]
            r = min(len(s), size)
            take = s[:r]
            for v in take:
                cur.append(v)
                if len(cur) == size:
                    arrays.append(cur)
                    cur = []
                    if len(arrays) == k:
                        break
            if len(arrays) == k:
                break
        if len(arrays) < k:
            return None

        Ys = [sum(arr) / float(size) for arr in arrays]
        grid = self._bin_midpoints(ell)
        Ysnapped = [self._nearest(y, grid) for y in Ys]

        costs = []
        for y in grid:
            left = sum(1 for yy in Ysnapped if yy < y)
            right = sum(1 for yy in Ysnapped if yy > y)
            c = max(left, right)
            costs.append(c)

        scale = self.eps_prime / 4.0
        weights = [math.exp(-scale * c) for c in costs]
        Z = sum(weights)
        if Z <= 0:
            return 0.5
        r = random.random() * Z
        acc = 0.0
        for w, y in zip(weights, grid):
            acc += w
            if r <= acc:
                return y
        return grid[-1]


class ContinualMeanEstimator:

    def __init__(self, n_max: int, m_max: int, epsilon: float, delta: float):
        assert n_max >= 1 and m_max >= 1
        self.n_max = int(n_max)
        self.m_max = int(m_max)
        self.epsilon = float(epsilon)
        self.delta = float(delta)

        self.L = int(math.ceil(math.log(max(self.m_max, 1), 2))) if self.m_max > 1 else 0

        self._delta_per = self.delta  
        self._eps_per_mech = self.epsilon / (2.0 * (self.L + 1))  # for binary mechs
        self._eps_per_median = self.epsilon / (2.0 * max(self.L, 1))  # split for medians across ℓ>=1

        self.Delta = {}
        for ell in range(0, self.L + 1):
            self.Delta[ell] = self._compute_delta(ell)

        self.mechanisms: Dict[int, BinaryMechanism] = {}
        for ell in range(0, self.L + 1):
            noise_b = self._noise_scale_for_scale(ell)
            self.mechanisms[ell] = BinaryMechanism(noise_b=noise_b)

        self.inactive: set[int] = set(range(2, self.L + 1))  
        self.buffers: Dict[int, List[float]] = {ell: [] for ell in range(2, self.L + 1)}

        self.mu_tilde: Dict[int, Optional[float]] = {ell: None for ell in range(0, self.L + 1)}

        # User state
        self.user_counts: Dict[int, int] = defaultdict(int)      # M(u)
        self.user_samples: Dict[int, List[float]] = defaultdict(list)  # full per-user stream (0/1)

        # Running totals
        self.total_included: int = 0  # denominator (# samples represented in current DP sum)

        # maintain M_t (max samples from any user)
        self.M_t: int = 0

    def _compute_delta(self, ell: int) -> float:
        n = self.n_max
        m = self.m_max
        L = max(self.L, 1)
        # Split delta as in paper
        delta1 = max(self.delta / 3.0, 1e-12)
        delta2 = max(self.delta / (3.0 * L), 1e-12)
        eps_med = max(self._eps_per_median, 1e-12)

        # k(eps', ell, beta)
        beta = delta2
        num = (2 ** ell) / 2.0 if ell >= 1 else 0.5
        k = max(1, math.ceil((16.0 / eps_med) * math.log(max(num / beta, 1.0))))

        # Two components (cf. Eq. (9) structure):
        term1 = math.sqrt(max(2 ** max(ell - 1, 0) / 2.0, 0.0) * math.log(max(2.0 * n * max(math.log(max(m, 2)), 1.0) / delta1, 1.0)))
        term2 = math.sqrt(max(2 ** ell, 1.0) * math.log(max(2.0 * k / delta2, 1.0)))
        return term1 + term2

    def _noise_scale_for_scale(self, ell: int) -> float:
        n = self.n_max
        L = self.L
        Delta_ell = self.Delta.get(ell, 1.0)
        print(Delta_ell)
        return (4.0 * Delta_ell * (1.0 + math.log(max(n, 2))) * (L + 1)) / max(self.epsilon, 1e-12)

    def _activation_threshold(self, ell: int) -> int:
        L = max(self.L, 1)
        left_unit = 2 ** max(ell - 1, 0)
        factor = (16.0 / max(self.epsilon, 1e-12)) * (2.0 * L * math.log(max(3.0 * L * (2.0 ** (ell / 2.0)) / max(self.delta, 1e-12), 1.0)))
        rhs = left_unit * factor
        return int(math.ceil(rhs))

    def _try_activate_scales(self) -> None:
        if not self.inactive:
            return

        t_sum = None
        for ell in sorted(list(self.inactive)):
            left_unit = 2 ** max(ell - 1, 0)
            lhs = 0
            for u, cnt in self.user_counts.items():
                lhs += min(cnt, left_unit)
            rhs = self._activation_threshold(ell)
            # print(lhs, rhs, ell)
            if lhs >= rhs:
                med = PrivateMedian(eps_prime=self._eps_per_median, ell=ell, beta=max(self.delta / (3.0 * max(self.L, 1)), 1e-12))
                mu_tilde = med.run(self.user_samples)
                if mu_tilde is None:
                    continue  # not enough data filled into arrays yet
                self.mu_tilde[ell] = mu_tilde

                center = (2 ** max(ell - 1, 0)) * mu_tilde
                Delta_ell = self.Delta[ell]
                lo = center - Delta_ell
                hi = center + Delta_ell

                buf = self.buffers.get(ell, [])
                if buf:
                    mech = self.mechanisms[ell]
                    block_size = 1 if ell <= 1 else (2 ** (ell - 1))
                    for sigma in buf:
                        sigma_proj = min(max(sigma, lo), hi)
                        mech.add(sigma_proj)
                        self.total_included += block_size
                    self.buffers[ell] = []

                # Mark active
                self.inactive.discard(ell)

    def update(self, x_t: float, u_t: int) -> float:

        x_t = float(x_t)
        x_t = 1.0 if x_t >= 0.5 else 0.0  # ensure Bernoulli support
        self.user_counts[u_t] += 1
        self.user_samples[u_t].append(x_t)
        self.M_t = max(self.M_t, self.user_counts[u_t])

        M_u = self.user_counts[u_t]
        is_power_of_two = (M_u & (M_u - 1) == 0)
        if is_power_of_two:
            ell = round(math.log(M_u, 2))
            if ell == 0:
                sigma = x_t  # first sample goes directly
                mech = self.mechanisms[0]
                mech.add(sigma)
                self.total_included += 1
            else:
                start = (2 ** (ell - 1))  # 1-indexed -> 0-indexed start-1
                end = (2 ** ell)         # inclusive 1-indexed -> slice end
                arr = self.user_samples[u_t]
                block_vals = arr[start:end]
                sigma = float(sum(block_vals))
                block_size = 1 if ell == 1 else (2 ** (ell - 1))
                if ell >= 2 and ell in self.inactive:
                    self.buffers[ell].append(sigma)
                else:
                    if ell >= 2:
                        mu_tilde = self.mu_tilde.get(ell, 0.5)
                        center = (2 ** (ell - 1)) * mu_tilde
                        Delta_ell = self.Delta[ell]
                        lo = center - Delta_ell
                        hi = center + Delta_ell
                        sigma = min(max(sigma, lo), hi)
                    self.mechanisms[ell].add(sigma)
                    self.total_included += block_size

        self._try_activate_scales()

        S = 0.0
        for ell in range(0, self.L + 1):
            S += self.mechanisms[ell].prefix_sum()

        denom = max(self.total_included, 1)
        mu_hat = S / float(denom)
        return mu_hat

    def diversity_holds(self) -> bool:
        if self.M_t <= 0:
            return False
        Mt = self.M_t
        L = max(self.L, 1)
        lhs = sum(min(c, Mt // 2) for c in self.user_counts.values())
        rhs = int(math.ceil((Mt // 2) * (16.0 / max(self.epsilon, 1e-12)) * (2.0 * L * math.log(max(3.0 * L * math.sqrt(max(Mt, 1)) / max(self.delta, 1e-12), 1.0)))))
        return lhs >= rhs

    def current_state(self) -> Dict[str, float]:
        return {
            "L": self.L,
            "M_t": self.M_t,
            "total_included": self.total_included,
            "num_users": len(self.user_counts),
            "inactive_scales": len(self.inactive),
        }



class ApproxDPContinualMeanEstimator:
    """Mixture wrapper that turns a pure-epsilon continual mechanism into
    an (epsilon, approx_delta)-DP mechanism.

    One Bernoulli draw is made for the *entire continual transcript*:
      - with probability approx_delta, release the exact running mean;
      - otherwise, run the original ContinualMeanEstimator unchanged.

    The original estimator is not modified.
    """

    def __init__(
        self,
        n_max: int,
        m_max: int,
        epsilon: float,
        algorithm_delta: float,
        approx_delta: float,
        branch_seed: Optional[int] = None,
        force_truthful_branch: Optional[bool] = None,
    ):
        if not 0.0 <= approx_delta <= 1.0:
            raise ValueError("approx_delta must lie in [0, 1].")

        self.approx_delta = float(approx_delta)
        self.algorithm_delta = float(algorithm_delta)

        # Use an independent RNG so that the mixture coin does not change the
        # generated data stream or the random numbers used by the base method.
        branch_rng = random.Random(branch_seed)
        if force_truthful_branch is None:
            self.return_true = branch_rng.random() < self.approx_delta
        else:
            self.return_true = bool(force_truthful_branch)

        # Instantiate the original algorithm only on the private branch.
        self.base: Optional[ContinualMeanEstimator]
        if self.return_true:
            self.base = None
        else:
            self.base = ContinualMeanEstimator(
                n_max=n_max,
                m_max=m_max,
                epsilon=epsilon,
                delta=algorithm_delta,
            )

        self._true_sum = 0.0
        self._num_updates = 0

    @property
    def branch_name(self) -> str:
        return "truthful" if self.return_true else "private"

    def update(self, x_t: float, u_t: int) -> float:
        # Match the Bernoulli preprocessing used by the original update().
        x_bin = 1.0 if float(x_t) >= 0.5 else 0.0
        self._true_sum += x_bin
        self._num_updates += 1

        if self.return_true:
            return self._true_sum / float(self._num_updates)

        assert self.base is not None
        return self.base.update(x_t=x_bin, u_t=u_t)

    def diversity_holds(self) -> bool:
        if self.base is None:
            return True
        return self.base.diversity_holds()

    def current_state(self) -> Dict[str, object]:
        if self.base is None:
            return {
                "branch": self.branch_name,
                "approx_delta": self.approx_delta,
                "num_updates": self._num_updates,
            }
        state: Dict[str, object] = dict(self.base.current_state())
        state.update({
            "branch": self.branch_name,
            "approx_delta": self.approx_delta,
            "num_updates": self._num_updates,
        })
        return state


from pathlib import Path
import numpy as np
import random, time, csv

SEED = 50
EXP = 19
p = 0.5
eps = 1.0

# The original code uses delta inside its thresholds/concentration parameters.
# It is kept separate from the new approximate-DP mixture probability.
algorithm_delta = 1e-3

# New approximate-DP parameter: with this probability, return the exact
# continual running-mean transcript. Set equal to the desired privacy delta.
approx_delta = 1e-6

T = 2 ** EXP
m_values = [8, 32, 64]

rows = []

for m in m_values:
    n_users = T // m

    outdir = Path(f"cache/bin_mech_algo_approx_dp/mu{p:g}")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Running m={m} ===")
    print(f"T = {T:,} steps, users = {n_users:,}, seeds = {SEED}")
    print(
        f"epsilon={eps:g}, algorithm_delta={algorithm_delta:g}, "
        f"approx_delta={approx_delta:g}"
    )

    times = []
    private_times = []
    truthful_times = []
    branch_counts = {"private": 0, "truthful": 0}

    for s in range(SEED):
        # This preserves the original per-seed data and base-mechanism randomness.
        random.seed(s)

        # Independent branch seed: drawing the approximate-DP branch does not
        # consume from the global RNG used by the data and original mechanism.
        branch_seed = 10_000_000 + 100_000 * m + s

        est = ApproxDPContinualMeanEstimator(
            n_max=n_users,
            m_max=m,
            epsilon=eps,
            algorithm_delta=algorithm_delta,
            approx_delta=approx_delta,
            branch_seed=branch_seed,
        )

        stream = []
        for u in range(n_users):
            for _ in range(m):
                x = 1.0 if random.random() < p else 0.0
                stream.append((u, x))
        random.shuffle(stream)

        t0 = time.perf_counter()

        estimates = []
        for (u, x) in stream:
            mu_hat = est.update(x_t=x, u_t=u)
            estimates.append(mu_hat)

        dt = time.perf_counter() - t0
        times.append(dt)
        branch = est.branch_name
        branch_counts[branch] += 1
        if branch == "private":
            private_times.append(dt)
        else:
            truthful_times.append(dt)

        arr = np.asarray(estimates, dtype=np.float32)
        common_name = (
            f"EXP{EXP}_m{m}_eps{eps}_algdelta{algorithm_delta}_"
            f"approxdelta{approx_delta}_branch{branch}_seed{s}"
        )

        save_path = outdir / f"mu_hat_{common_name}.npy"
        np.save(save_path, arr)

        # Save S_t = sum_{j<=t} (mu_j - muhat_j)^2.
        csum = 0.0
        true_rm = np.empty(len(stream), dtype=np.float32)
        for t_idx, (_, x) in enumerate(stream, start=1):
            csum += x
            true_rm[t_idx - 1] = csum / t_idx

        Tlen = min(len(true_rm), len(arr))
        err2 = (true_rm[:Tlen] - arr[:Tlen]) ** 2
        sum_sqerr = np.cumsum(err2, dtype=np.float64).astype(np.float32)

        sum_path = outdir / f"sum_sqerr_{common_name}.npy"
        np.save(sum_path, sum_sqerr)

        print(
            f"Seed {s:02d}: branch={branch:8s}; wrote {save_path.name} & "
            f"{sum_path.name} (len={arr.size:,}), runtime={dt:.3f}s"
        )

    def mean_or_nan(values: List[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    def std_or_nan(values: List[float]) -> float:
        return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")

    rows.append([
        "bin_mech_algo_approx_dp",
        m,
        EXP,
        eps,
        algorithm_delta,
        approx_delta,
        p,
        mean_or_nan(times),
        std_or_nan(times),
        len(times),
        branch_counts["private"],
        branch_counts["truthful"],
        mean_or_nan(private_times),
        mean_or_nan(truthful_times),
    ])

Path("plots").mkdir(parents=True, exist_ok=True)
csv_path = Path("cache") / (
    f"bin_mech_approx_runtime_EXP{EXP}_eps{eps}_algdelta{algorithm_delta}_"
    f"approxdelta{approx_delta}.csv"
)

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "method",
        "m",
        "EXP",
        "eps",
        "algorithm_delta",
        "approx_delta",
        "p",
        "mean_runtime_sec",
        "std_runtime_sec",
        "n_runs",
        "n_private_branches",
        "n_truthful_branches",
        "mean_private_runtime_sec",
        "mean_truthful_runtime_sec",
    ])
    writer.writerows(rows)

print(f"\nSaved runtime summary to {csv_path}")
