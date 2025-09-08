import numpy as np

def frob_D_T(A, p):
    """
    A: lower-triangular Toeplitz (first column)
    D: diagonal matrix as 1D array
    p: bandwidth (number of subdiagonals)
    """
    n = len(A)
    D = 1.0 / np.arange(1, n + 1)  # diagonal entries
    h = A  # first column of LTT
    total = 0.0
    for j in range(n):  # column index
        # sum over the allowed band entries
        # row index i >= j, but i-j < p => i in [j, min(j+p-1,n-1)]
        i_max = min(j + p, n)
        col_sum = np.sum((D[j:i_max] * h[:i_max-j])**2)
        total += col_sum
    return np.sqrt(total)

def frob_A_times_T(g, n):
    h = np.zeros(n)
    h[:len(g)] = g
    return frob_D_T(np.cumsum(h)[:n], n)

def generate_D(n):
    return np.diag(1.0/np.arange(1,n+1))

def generate_A1(n):
    return np.tri(n)


def generate_A(n):
    A = np.zeros((n,n))
    for i in range(n):
        A[i,:i+1] = 1.0/(i+1)
    return A

def banded(C: np.ndarray, p: int):

    C = np.asarray(C)
    n = C.shape[0]

    # build mask: True when i-j < p, False otherwise
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    mask = (i - j) < p

    # apply mask
    Cb = C.copy()
    Cb[~mask] = 0
    return Cb

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

n = 3000
A = random_lower_triangular_toeplitz(n, 1, 110)
D = generate_D(n)
A1 = generate_A1(n)
AA = generate_A(n)

import time
start = time.time()
print(np.linalg.norm(D @ A,'fro'))
tt = time.time()
print(frob_D_T(A[:,0], n))
ttt = time.time()
print(f"Direct frobenius norm took {tt - start} seconds")
print(f"frob_D_T took {ttt - tt} seconds")

p = 20
start = time.time()
print(np.linalg.norm(D @ banded(A , p=p),'fro'))
tt = time.time()
print(frob_D_T(A[:,0], p))
ttt = time.time()
print(f"Direct frobenius norm took {tt - start} seconds")
print(f"frob_D_T took {ttt - tt} seconds")

print("frob_A_times_T")
print(np.linalg.norm(AA @ A,'fro'))
print(frob_A_times_T(A[:,0], n))

print("frob_A_times_T_Banded")
print(np.linalg.norm(AA @ banded(A, p=p),'fro'))
print(frob_A_times_T(banded(A, p=p)[:,0], n))