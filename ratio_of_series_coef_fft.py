import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# ---- Settings ----
N = 100000                         # number of coefficients

import numpy as np

def next_pow2(n):
    """Smallest power of two ≥ n."""
    return 1 << ((n - 1).bit_length())

def fft_convolve(a, b, n):
    """
    Convolve two real sequences a, b and return only the first n terms of the product.
    Uses zero‐padding + FFT in O(n log n).
    """
    size = next_pow2(max(len(a) + len(b) - 1, n))
    fa = np.fft.rfft(a, size)
    fb = np.fft.rfft(b, size)
    c = np.fft.irfft(fa * fb, size)
    return c[:n]

def poly_inv(B, n):
    """
    Compute the series inverse D = 1/B mod x^n,
    i.e. B(x)*D(x) ≡ 1 (mod x^n), via Newton iteration in O(n log n).
    """
    assert B[0] != 0.0
    # start with D0 = 1/B[0]
    D = np.array([1.0 / B[0]])
    m = 1
    while m < n:
        m2 = min(2*m, n)
        # E(x) = 2 - B(x)*D(x)  mod x^m2
        BD = fft_convolve(B[:m2], D, m2)
        E = np.zeros(m2)
        E[0] = 2.0 - BD[0]
        E[1:] = -BD[1:]
        # D_new = D * E  mod x^m2
        D = fft_convolve(D, E, m2)
        m = m2
    return D

def poly_sqrt(F, n):
    """
    Compute the series square root B = sqrt(F) mod x^n,
    assuming F[0] = 1, via Newton iteration in O(n log n).
    """
    assert abs(F[0] - 1.0) < 1e-14, "Need F[0]=1"
    B = np.array([1.0])
    m = 1
    while m < n:
        m2 = min(2*m, n)
        # Compute 1/B up to x^m2
        invB = poly_inv(B, m2)
        # T(x) = F(x)/B(x)  ≡ F * invB  mod x^m2
        T = fft_convolve(F[:m2], invB, m2)
        # Newton step: B_new = (B + T)/2
        B = (np.pad(B, (0, m2 - len(B))) + T) * 0.5
        m = m2
    return B

# build F-coeffs: F[n] = 1/(n+1)
F = np.fromiter((1.0/(i+1) for i in range(N)), dtype=np.float64)
# compute b[0..N-1]
coeffs = poly_sqrt(F, N)



# ---- Compute Maclaurin coefficients b_n ----
# x = sp.symbols('x')
# f = sp.sqrt(-sp.log(1 - x) / x)
# series = sp.series(f, x, 0, N + 1).removeO()
# coeffs = [float(series.coeff(x, n)) for n in range(N + 1)]  # convert to floats


# ---- Prepare the comparison sequence 1/(n+1) ----
n_vals = np.arange(N)
inv_x = 1.0 / (n_vals + 1)             # 1/(n+1) to avoid division by zero at n = 0
# sqrt_x = 1/5 * np.sqrt(n_vals + 1)            
ratios = [inv_x[n] / coeffs[n] for n in range(N)]

# S = []
# SS = []
# n_vals = np.arange(N + 1)
# for n in n_vals:
#     conv = sum((k+1) / (k+2) * coeffs[k] * coeffs[n - k] for k in range(1,n))
#     S.append(conv + 1/(2*n+4))
#     conv = sum(2* coeffs[k] * coeffs[k] for k in range(1,round(n/2)))
#     SS.append(conv)

# ---- Plot ----
plt.figure(figsize=(6, 4))
# plt.plot(n_vals[:N], ratio, label=r'$b_k1$')
plt.plot(n_vals, ratios, label=r'$ratio of b_k and 1/k$')
plt.xscale('log')
# plt.plot(n_vals, sqrt_x, label=r'$sqrt k$')
# plt.plot(n_vals[80:], SS[80:], label=r'$prod$')
# plt.plot(n_vals[120:], coeffs[120:], label=r'$b_n$ (coefficients)')
# plt.plot(n_vals[80:], inv_x[80:], linestyle='--', label=r'$1/(n+1)$')
plt.xlabel(r'Index $n$')
plt.ylabel('Value')
# plt.title(r'Coefficients $b_n$ vs. $1/(n+1)$ (first {} terms)'.format(N + 1))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
