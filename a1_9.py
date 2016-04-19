#!/usr/local/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt

N = int(sys.argv[1])
MU1, SIGMA1 = 2, 1.5
MU2, SIGMA2 = 3, 2.2

X1 = np.random.normal(MU1, SIGMA1, N)
X2 = np.random.normal(MU2, SIGMA2, N)

plt.hist(X1/X2, 50, normed=True)

plt.show()
