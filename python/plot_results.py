import numpy as np
import matplotlib.pyplot as plt


# N = 100
# Timing: 0.531127871 ms

# N = 200
# Timing: 1.987639743 ms

# N = 300
# Timing: 4.415483715 ms

# N = 400
# Timing: 7.879197977 ms

# N = 500
# Timing: 12.259367472 ms

# N = 600
# Timing: 17.571775384 ms

# N = 700
# Timing: 23.961638105 ms

# N = 800
# Timing: 31.390946119 ms

# N = 900
# Timing: 39.161435509 ms

# N = 1000
# Timing: 48.401889008 ms

cpp_timing = np.array([0.53, 1.99, 4.42, 7.88, 12.26, 17.57, 23.96, 31.39, 39.16, 48.40])
python_timing = np.array([207.68, 844.19, 1924.18, 3481.79, 5442.59, 7473.03, 10418.46, 13646.51, 17150.21, 20890.03])
N_sizes = np.array(range(100, 1100, 100))

performance_flops = np.array([
    (n**2 * 2 * 5**2) / time
    for n, time in zip(N_sizes, python_timing / 1000)
])

# hw 2

n_ops = 4.2e10 + 4

timings = np.array([8.35, 4.84, 3.34, 2.71, 3.34])
perfs = n_ops / timings / 1e9
n_threads = np.array([1, 2, 4, 8, 16]) 

plt.plot(n_threads, perfs)
plt.ylabel('Performance (Gflops)')
plt.xlabel('Amount of threads')
plt.show()
