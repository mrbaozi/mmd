#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


# def simple_error(xval, covmat):
#     return np.sqrt(delta_a**2 + (xval * delta_b)**2)


def get_cor_mat(covmat):
    ''' calculate correlation matrix from covariance matrix '''
    dim = covmat.shape
    cormat = np.empty(dim)
    for i in range(dim[0]):
        for j in range(dim[1]):
            cormat[i, j] = covmat[i, j] / np.sqrt(covmat[i, i] * covmat[j, j])
    return cormat

if __name__ == "__main__":
    NPOINTS = 10
    XOF = 10

    # generate random data and errors
    X = np.array([x for x in range(XOF, XOF + NPOINTS, 1)])
    Y = np.array([y for y in range(XOF, XOF + NPOINTS, 1)])
    EX = np.array([np.random.normal(0, 0.5) for _ in range(NPOINTS)])
    EY = np.array([np.random.normal(0, 0.5) for _ in range(NPOINTS)])
    XDATA = np.add(X, EX)
    YDATA = np.add(Y, EY)

    # data cov & cor

    COVDATA = np.cov(XDATA, YDATA)
    CORDATA = get_cor_mat(COVDATA)

    # linear fit
    PARAMS, COVFIT = np.polyfit(XDATA, YDATA, 1, cov=True)
    CORFIT = get_cor_mat(COVFIT)

    # matrix output
    print("data covariance matrix:\n%s\n" % (COVDATA))
    print("data correlation matrix:\n%s\n" % (CORDATA))
    print("fit parameters (highest order first):\n%s\n" % (PARAMS))
    print("fit covariance matrix:\n%s\n" % (COVFIT))
    print("fit correlation matrix:\n%s\n" % (CORFIT))

    # error band

    # plot
    T1 = np.arange(XOF, XOF + NPOINTS, 1)
    plt.plot(T1, PARAMS[1] + PARAMS[0]*T1)
    plt.plot(XDATA, YDATA, 'ro')
    plt.show()
