import numpy as np
import matplotlib.pyplot as plt

def compute_b(N):
    """
    Compute b[0..N] satisfying
        sum_{i=0}^n b[i]*b[n-i] = 1/(n+1)
    using NumPy dot for the convolution sum.
    """
    b = np.zeros(N+1, dtype=np.float64)
    b[0] = 1.0
    for n in range(1, N+1):
        # use a single C-speed dot product instead of the inner Python loop
        #    sum_{i=1..n-1} b[i]*b[n-i]
        if n > 1:
            s = np.dot(b[1:n], b[n-1:0:-1])
        else:
            s = 0.0
        b[n] = (1.0/(n+1) - s) * 0.5
    return b

N = 100000
b = compute_b(N)  
n_vals = np.arange(N)
inv_x = 1.0 / ((n_vals + 1) * np.sqrt(np.log(n_vals + 1)+1))        # 1/(n+1) to avoid division by zero at n = 0
# sqrt_x = 1/5 * np.sqrt(n_vals + 1)            
ratios = [(inv_x[n] / b[n]) for n in range(N)]

plt.figure(figsize=(6, 4))
# plt.plot(n_vals[:N], ratio, label=r'$b_k1$')
plt.plot(n_vals, ratios, label=r'$ratio of b_k and 1/k$')
# plt.xscale('log')
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

print(b)