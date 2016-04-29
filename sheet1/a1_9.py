#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # input
    TITLE = "Moderne Methoden der Datenanalyse - Blatt 01 Aufgabe 1.9"
    PARSER = argparse.ArgumentParser(description=TITLE)
    PARSER.add_argument('--n', type=int, default=100,
                        help='number of events')
    PARSER.add_argument('--mu1', type=float, default=2,
                        help='mean of normal distribution #1')
    PARSER.add_argument('--mu2', type=float, default=3,
                        help='mean of normal distribution #2')
    PARSER.add_argument('--sig1', type=float, default=1.5,
                        help='standard deviation of distribution #1')
    PARSER.add_argument('--sig2', type=float, default=2.2,
                        help='standard deviation of distribution #2')
    ARGS = PARSER.parse_args()

    # assign values
    N = ARGS.n
    MU1, SIGMA1 = ARGS.mu1, ARGS.sig1
    MU2, SIGMA2 = ARGS.mu2, ARGS.sig2

    # data
    X1 = np.random.normal(MU1, SIGMA1, N)
    X2 = np.random.normal(MU2, SIGMA2, N)
    DATA = X1/X2

    # calculate propagated error from means, sigma
    D_X1 = 1 / np.mean(X2)**2 * SIGMA1**2
    D_X2 = np.mean(X1)**2 / np.mean(X2)**4 * SIGMA2**2
    ERR_PROP = D_X1 + D_X2

    # output
    print("mean X1: %s" % (np.mean(X1)))
    print("mean X2: %s" % (np.mean(X2)))
    print("mean X1/X2: %s" % (np.mean(X1/X2)))
    print("stddev X1: %s" % (np.std(X1)))
    print("stddev X2: %s" % (np.std(X2)))
    print("stddev X1/X2: %s" % (np.std(X1/X2)))
    print("propagated uncertainty (sq): %s" % (ERR_PROP))
    print("propagated uncertainty (rt): %s" % (np.sqrt(ERR_PROP)))

    # plot
    AXIS = np.linspace(-10, 10, 100)
    plt.hist(DATA.ravel(), AXIS)
    plt.show()
