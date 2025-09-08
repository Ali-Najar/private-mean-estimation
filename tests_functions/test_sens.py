import numpy as np

def sens_new(c, n, k, b):
    """
    Compute sens(C, k, b) for C = T(c) lower-triangular Toeplitz (first column c).
    Exactly equals: M = C[:, :k*b:b]; G = M.T @ M; sqrt(sum(G)) but avoids building M/G.

    Parameters
    ----------
    c : array-like
        First column of the Toeplitz matrix (length may be >= n or < n).
    n : int
        Matrix size (n x n). Only first n rows/cols considered.
    k : int
        Number of sampled columns (columns 0, b, 2b, ..., (k-1)b).
    b : int
        Column spacing.
    """
    c = np.asarray(c, dtype=float)
    P = min(len(c), n)  # available diagonal entries

    # Precompute prefix sums of products for lags L = 0, b, 2b, ..., (k-1)*b
    pref = {}
    for d in range(k):                  # d = 0..k-1
        L = d * b
        if L >= P:
            pref[L] = np.array([0.0])   # no overlap for this lag
            continue
        prod = c[: P - L] * c[L : P]   # elementwise products length = P-L
        p = np.empty(len(prod) + 1, dtype=float)
        p[0] = 0.0
        p[1:] = np.cumsum(prod)
        pref[L] = p                     # pref[L][t] = sum_{u=0..t-1} c[u]*c[u+L]

    total = 0.0
    # Sum over sampled column indices i,j (columns at offsets i*b and j*b)
    for i in range(k):
        si = i * b
        for j in range(k):
            sj = j * b
            L = abs(i - j) * b
            # valid rows for overlap start at max(si,sj) and go to n-1 -> count = n - max(si,sj)
            count = n - max(si, sj)
            if count <= 0:
                continue
            M = max(0, P - L)            # available products for this lag
            take = min(count, M)
            if take <= 0:
                continue
            total += pref[L][take]
    return float(np.sqrt(total))


def sens(C: np.ndarray, k: int, b: int):

    # pick out the k columns [0, b, 2b, ..., (k-1)b]
    M = C[:, :k*b:b]
    # form matrix G = M^T M, then sum all entries
    G = M.T @ M 
    return np.sqrt(float(np.sum(G)))


def generate_A(n):
    A = np.zeros((n,n))
    for i in range(n):
        A[i,:i+1] = 1.0/(i+1)
    return A

def random_lower_triangular_toeplitz(n, low=0, high=10):
    # Generate random values for each diagonal
    diagonals = np.random.randint(low, high, size=n)
    
    # Create an n x n zero matrix
    LTT = np.zeros((n, n), dtype=int)
    
    # Fill in the diagonals
    for i in range(n):
        for j in range(i+1):
            LTT[i, j] = diagonals[i-j]
    
    return LTT

n = 1000
A = random_lower_triangular_toeplitz(n, 1, 110)

import time
start = time.time()
print(sens(A, 50, 20))
tt = time.time()
print(tt - start)
print(sens_new(A[:,0], n, 50, 20))
ss = time.time()
print(ss - tt)