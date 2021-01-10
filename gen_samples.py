import numpy as np
from numpy.random import multivariate_normal
import sys

n_dim_str = sys.argv[1]
n_dim = int(n_dim_str)
n_samples = int(sys.argv[2])

cov = np.zeros((n_dim, n_dim))

for i in range(0, n_dim):
    cov[i, i] = 10.

mean_val = 5.0

means = mean_val * np.ones(n_dim)
samples = multivariate_normal(means, cov, n_samples)

filename = 'data_' + n_dim_str + '.txt'
np.savetxt(filename, samples)
