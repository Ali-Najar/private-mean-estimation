import numpy as np

######## NEW CODE ########


def seq_A1(n):           # first column of A1
    return np.ones(n, dtype=float)

def seq_Dtoep(n):        # first column of D_toep: 1, 1/2, 1/3, ...
    i = np.arange(n, dtype=float)
    return 1.0 / (i + 1.0)

def seq_A1_sqrt_rec(n: int) -> np.ndarray:
    a = np.empty(n, dtype=float)
    if n == 0:
        return a
    a[0] = 1.0
    for m in range(0, n-1):
        num = (2*m + 1) * (2*m + 2)
        den = 4.0 * (m + 1) * (m + 1)
        a[m+1] = a[m] * (num / den)
    return a

def band_seq(c, p):
    return c[:min(p, len(c))].copy()

def inv_series(c, n):
    """Recurrence for series inverse: g[0]=1/c0; for m>=1 g[m] = -1/c0 * sum_{t=1..min(m,L-1)} c[t] * g[m-t]."""
    g = np.zeros(n, dtype=float)
    c0 = c[0]
    g[0] = 1.0 / c0
    L = len(c)
    for m in range(1, n):
        tmax = min(m, L - 1)
        s = 0.0
        for t in range(1, tmax + 1):
            s += c[t] * g[m - t]
        g[m] = - s / c0
    return g

def sqrt_series(a, n):
    """First column s with T(s)^2 = T(a) up to length n."""
    s = np.zeros(n, dtype=float)
    s[0] = np.sqrt(a[0])
    for m in range(1, n):
        if m >= 2:
            conv = float(np.dot(s[1:m], s[m-1:0:-1]))  # sum_{t=1..m-1} s_t s_{m-t}
        else:
            conv = 0.0
        a_m = a[m] if m < len(a) else 0.0
        s[m] = (a_m - conv) / (2.0 * s[0])
    return s

# ---------- Frobenius for B = D · T(h) ----------
def frob_D_T(A, p):
    """
    A: lower-triangular Toeplitz (first column)
    D: diagonal matrix as 1D array
    p: bandwidth (number of subdiagonals)
    """
    n = len(A)
    D = 1/np.arange(1, n+1)
    h = A  # first column of LTT
    total = 0.0
    for j in range(n):  # column index
        # sum over the allowed band entries
        # row index i >= j, but i-j < p => i in [j, min(j+p-1,n-1)]
        i_max = min(j + p, n)
        col_sum = np.sum((D[j:i_max] * h[:i_max-j])**2)
        total += col_sum
    return np.sqrt(total)

# Special case A = D·A1 => h = ones  ->  ||A||_F^2 = H_n (harmonic number)
def frob_A(n):
    return np.sqrt(np.sum(1.0 / (np.arange(1, n+1, dtype=float))))


def sens(c, n, k, b):
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

# --------- A · T(g) has Toeplitz part T(cumsum(g)) ----------
def frob_A_times_T(g, n):
    h = np.zeros(n)
    h[:len(g)] = g
    return frob_D_T(np.cumsum(h)[:n], n)

# ---------- High-level “obj” builders matching your four/banded cases ----------
def obj_nonbanded_case1(n, k, b):
    # B = D · A1^{1/2}  (Toeplitz h = seq_A1_sqrt),  C = A1^{1/2}
    c = seq_A1_sqrt_rec(n)
    frobB = frob_D_T(c, n)
    sensC = sens(c, n, k, b)
    return (frobB * sensC) / np.sqrt(n)

def obj_nonbanded_case2(n, k, b):
    # B = A = D·A1,  C = I
    frobB = frob_A(n)            # exact, cheap
    sensC = np.sqrt(k)           # sum of k basis columns
    return (frobB * sensC) / np.sqrt(n)

def obj_nonbanded_case3(n, k, b):
    # B = A · (D_toep^{1/2})^{-1},  C = D_toep^{1/2}
    a  = seq_Dtoep(n)
    s  = sqrt_series(a, n)       # first col of sqrt(D_toep)
    is_ = inv_series(s, n)       # first col of inv(sqrt(D_toep))
    frobB = frob_A_times_T(is_, n)
    sensC = sens(s, n, k, b)
    return (frobB * sensC) / np.sqrt(n)

def obj_nonbanded_case4(n, k, b):
    # B = A · (D_toep)^{-1},  C = D_toep
    c  = seq_Dtoep(n)
    invc = inv_series(c, n)
    frobB = frob_A_times_T(invc, n)
    sensC = sens(c, n, k, b)
    return (frobB * sensC) / np.sqrt(n)


##########################


######## OLD CODE ########


def generate_A(n):
    A = np.zeros((n,n))
    for i in range(n):
        A[i,:i+1] = 1.0/(i+1)
    return A

def generate_D(n):
    return np.diag(1.0/np.arange(1,n+1))

def generate_A1(n):
    return np.tri(n)

def generate_D_toep(n):
    M = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            M[i,j] = 1.0/(i-j+1)
    return M

def sens_old(C: np.ndarray, k: int, b: int):

    # pick out the k columns [0, b, 2b, ..., (k-1)b]
    M = C[:, :k*b:b]
    # form matrix G = M^T M, then sum all entries
    G = M.T @ M 
    return np.sqrt(float(np.sum(G)))

def obj(B, C, k ,b):
    n = B.shape[0]
    return np.linalg.norm(B,'fro') * sens_old(C, k, b) / np.sqrt(n)

##########################

import scipy.linalg as la

n = 2000
k = 20
b = 100
A    = generate_A(n)
D    = generate_D(n)
A1   = generate_A1(n)
Dtp  = generate_D_toep(n)
I    = np.eye(n)


A1_s = la.sqrtm(A1)
Dtp_s  = la.sqrtm(Dtp)
Dtp_is = la.inv(Dtp_s)

print("Case 1: B = D·A1^{1/2}, C = A1^{1/2}")
val = obj(D @ A1_s, A1_s, k, b)
val2 = obj_nonbanded_case1(n, k, b)
print(f"obj = {val}, obj_nonbanded_case1 = {val2}")

print("Case 2: B = A = D·A1, C = I")
val = obj(A, I, k, b)
val2 = obj_nonbanded_case2(n, k, b)
print(f"obj = {val}, obj_nonbanded_case2 = {val2}")

print("Case 3: B = A·(D_toep^{1/2})^{-1}, C = D_toep^{1/2}")
val = obj(A @ Dtp_is, Dtp_s, k, b)
val2 = obj_nonbanded_case3(n, k, b)
print(f"obj = {val}, obj_nonbanded_case3 = {val2}")

print("Case 4: B = A·(D_toep)^{-1}, C = D_toep")
val = obj(A @ la.inv(Dtp), Dtp, k, b)
val2 = obj_nonbanded_case4(n, k, b)
print(f"obj = {val}, obj_nonbanded_case4 = {val2}")