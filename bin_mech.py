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


random.seed(2)
n = 40480
m = 10
p = 0.35
eps = 1.0
delta = 0.5

est = ContinualMeanEstimator(n_max=n, m_max=m, epsilon=eps, delta=delta)

# Build a randomized arrival order (arbitrary user order allowed)
stream: List[Tuple[int, float]] = []  # (user, sample)
for u in range(n):
    k = m  # each user contributes m samples
    for _ in range(k):
        x = 1.0 if random.random() < p else 0.0
        stream.append((u, x))
random.shuffle(stream)

# Run the continual estimator
estimates = []
true_sum = 0.0
for t, (u, x) in enumerate(stream, start=1):
    true_sum += x
    mu_hat = est.update(x_t=x, u_t=u)
    estimates.append(mu_hat)
    if t % 100 == 0:
        print(f"t={t:4d}  mu_hat={mu_hat:.4f}  true_mean_so_far={true_sum/t:.4f}  state={est.current_state()}")

print("Done.")

import matplotlib.pyplot as plt

def plot_estimation_error(mu_hats, xs, p=None, title=None, use_log_y=False):
    """
    Plot |mu_hat_t - true_mean_t| over time.

    Args:
        mu_hats: list of DP estimates [mu_hat_1, ..., mu_hat_T]
        xs:      list of observed samples [x_1, ..., x_T] (same length/order)
        p:       float or None. If provided, uses this as the true mean.
                 If None, uses empirical mean so far (sum_{i<=t} x_i / t).
        title:   optional plot title (str)
        use_log_y: bool, if True sets y-axis to log-scale
    """
    assert len(mu_hats) == len(xs), "mu_hats and xs must have same length"
    errors = []
    csum = 0.0
    for t, (mh, x) in enumerate(zip(mu_hats, xs), start=1):
        csum += x
        true_mean_t = p if p is not None else (csum / t)
        errors.append(abs(mh - true_mean_t))

    plt.figure()
    plt.plot(range(1, len(errors) + 1), errors)
    if use_log_y:
        plt.yscale("log")
    plt.xlabel("t")
    # plt.xscale("log")
    plt.ylabel("|mu_hat - true_mean|")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

plot_estimation_error(estimates, range(n*m), p=p, title="Estimation error over time", use_log_y=True)