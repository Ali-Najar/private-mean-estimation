import math
import matplotlib.pyplot as plt

def compute_S(max_n):
    # Precompute b_i = binomial(2i, i) / 4^i via recurrence: b_0 = 1, b_n = b_{n-1} * (2n-1)/(2n)
    b = [1.0] * (max_n + 1)
    for n in range(1, max_n + 1):
        b[n] = b[n-1] * (2*n - 1) / (2*n)
    
    # Compute prefix sums of b_i^2
    cumsum_b2 = [0.0] * (max_n + 1)
    cumsum_b2[0] = b[0]**2
    for i in range(1, max_n + 1):
        cumsum_b2[i] = cumsum_b2[i-1] + b[i]**2
    
    # Compute S(n) = sum_{k=1}^n [1/k**2 * sum_{i=0}^{k-1} b_i^2]
    S = [0.0] * (max_n + 1)
    for k in range(1, max_n + 1):
        S[k] = S[k-1] + cumsum_b2[k-1] / k**2
    
    return S

# Example usage and plotting
max_n = 1000000
S_values = compute_S(max_n)
n_values = list(range(1, max_n + 1))

plt.figure()
plt.plot(n_values, S_values[1:])
plt.xlabel('n')
plt.ylabel('S(n)')
plt.title('Growth of S(n) = sum_{k=1}^n 1/k**2 * sum_{i=0}^{k-1} b_i^2')
plt.show()

# Print the value for n = max_n
print(f"S({max_n}) = {S_values[max_n]:.6f}")
