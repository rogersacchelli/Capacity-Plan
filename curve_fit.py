import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from scipy.stats import norm

x = []

with open('bras_int_subs','r') as f:
    for line in f:
       x.append(int(line.strip('\n')))

X = np.array([x])

X_plot = np.linspace(np.amin(X), np.amax(X), math.sqrt(np.unique(X.size)))[:, np.newaxis]

fig, ax = plt.subplots()

ax.hist(x, bins=int(math.sqrt(np.unique(X.size))),normed=True)

for kernel in ['gaussian','epanechnikov']:
    kde = KernelDensity(kernel=kernel, bandwidth=150).fit(X.transpose())
    log_dens = kde.score_samples(X_plot)
    ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
            label="kernel = '{0}'".format(kernel))


ax.legend(loc='upper right')
plt.show()
