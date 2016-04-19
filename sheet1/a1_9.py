#!/usr/local/bin/python3

import sys
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    try:
        N = int(sys.argv[1])
    except IndexError:
        N = 100

    # sheet values
    MU1, SIGMA1 = 2, 1.5
    MU2, SIGMA2 = 3, 2.2

    # data
    X1 = np.random.normal(MU1, SIGMA1, N)
    X2 = np.random.normal(MU2, SIGMA2, N)

    # calculate propagated error from means, sigma
    PROP_X1 = 1 / np.mean(X2)**2 * SIGMA1**2
    PROP_X2 = np.mean(X1)**2 / np.mean(X2)**4 * SIGMA2**2
    PROP = PROP_X1 + PROP_X2

    # output
    print("mean X1: %s" % (np.mean(X1)))
    print("mean X2: %s" % (np.mean(X2)))
    print("mean X1/X2: %s" % (np.mean(X1/X2)))
    print("stddev X1: %s" % (np.std(X1)))
    print("stddev X2: %s" % (np.std(X2)))
    print("stddev X1/X2: %s" % (np.std(X1/X2)))
    print("propagated uncertainty (sq): %s" % (PROP))
    print("propagated uncertainty (rt): %s" % (np.sqrt(PROP)))

    # plot
    plt.hist(X1/X2, 50, normed=True)
    plt.show()
