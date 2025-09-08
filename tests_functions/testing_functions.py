# Fix inv_series and re-run tests

import numpy as np
from math import comb, isclose
import scipy.linalg as la

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
    """Robust recurrence for series inverse: g[0]=1/c0; for m>=1 g[m] = -1/c0 * sum_{t=1..min(m,L-1)} c[t] * g[m-t]."""
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

# ---- tests (same as before) ----

def test_seq_A1():
    n = 8
    out = seq_A1(n)
    assert out.shape == (n,)
    assert np.allclose(out, np.ones(n))
    print("test_seq_A1 passed.")

def test_seq_Dtoep():
    n = 8
    out = seq_Dtoep(n)
    expected = 1.0 / (np.arange(1, n+1))
    assert out.shape == (n,)
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-15)
    print("test_seq_Dtoep passed.")

def test_seq_A1_sqrt_rec_small():
    n = 20
    out = seq_A1_sqrt_rec(n)
    expected = np.array([comb(2*m, m) / (4.0**m) for m in range(n)], dtype=float)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=0)
    print("test_seq_A1_sqrt_rec_small passed.")

def test_seq_A1_sqrt_rec_tail_behavior():
    n = 1000
    out = seq_A1_sqrt_rec(n)
    assert out[0] == 1.0
    assert np.all(out[1:] <= out[:-1] + 1e-14)
    m = np.arange(100, 1000)
    approx = 1.0 / np.sqrt(np.pi * m)
    rel_err = np.abs(out[100:] / approx - 1.0)
    assert np.median(rel_err) < 0.02
    print("test_seq_A1_sqrt_rec_tail_behavior passed. median rel err ~", np.median(rel_err))

def test_band_seq():
    c = np.arange(10, dtype=float)
    p = 4
    b = band_seq(c, p)
    assert isinstance(b, np.ndarray)
    assert b.shape[0] == p
    np.testing.assert_allclose(b, c[:p])
    b[0] = -123.0
    assert c[0] == 0.0
    print("test_band_seq passed.")

def toeplitz_from_firstcol(c, n):
    T = np.zeros((n,n), dtype=float)
    for i in range(n):
        for j in range(i+1):
            T[i,j] = c[i-j]
    return T

def test_inv_series_full():
    n = 10
    c = seq_Dtoep(n)
    g = inv_series(c, n)
    T = toeplitz_from_firstcol(c, n)
    Tinv = la.inv(T)
    np.testing.assert_allclose(g, Tinv[:,0], rtol=1e-12, atol=1e-12)
    e0 = np.zeros(n); e0[0] = 1.0
    np.testing.assert_allclose(T @ g, e0, rtol=1e-12, atol=1e-12)
    print("test_inv_series_full passed.")

def test_inv_series_banded():
    n = 12
    L = 3
    c = np.zeros(L)
    c[0] = 2.0
    c[1] = -0.5
    c[2] = 0.25
    g = inv_series(c, n)
    T = toeplitz_from_firstcol(np.concatenate([c, np.zeros(n-L)]), n)
    Tinv = la.inv(T)
    np.testing.assert_allclose(g, Tinv[:,0], rtol=1e-12, atol=1e-12)
    print("test_inv_series_banded passed.")

# Run tests
test_seq_A1()
test_seq_Dtoep()
test_seq_A1_sqrt_rec_small()
test_seq_A1_sqrt_rec_tail_behavior()
test_band_seq()
test_inv_series_full()
test_inv_series_banded()

print("All tests passed.")

