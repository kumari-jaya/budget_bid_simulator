import numpy as np

A = range(1,101)
bin_count = 10
hist = np.histogram(A, bins=bin_count)
x = np.linspace(1,100,10)
res = np.fmin(np.digitize(A, x), bin_count)