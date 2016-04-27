#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt


def get_cor_mat(covmat):
    ''' calculates correlation matrix from covariance matrix '''
    dim = covmat.shape
    cormat = np.empty(dim)
    for i in range(dim[0]):
        for j in range(dim[1]):
            cormat[i, j] = covmat[i, j] / np.sqrt(covmat[i, i] * covmat[j, j])
    return cormat

if __name__ == "__main__":
    NPOINTS = 10
    XOF = 10
    PD = 1  # polynomial degree

    try:
        IS_CORR = int(sys.argv[1])
    except IndexError:
        IS_CORR = 0
    try:
        NSIG = int(sys.argv[2])
    except IndexError:
        NSIG = 1

    # generate random data and errors
    X = np.array([x for x in range(XOF, XOF + NPOINTS, 1)])
    Y = np.array([y for y in range(XOF, XOF + NPOINTS, 1)])
    EX = np.array([np.random.normal(0, 0.5) for _ in range(NPOINTS)])
    EY = np.array([np.random.normal(0, 0.5) for _ in range(NPOINTS)])
    XDATA = np.add(X, EX)
    YDATA = np.add(Y, EY)

    # data covariance & correlation
    COVDATA = np.cov(XDATA, YDATA)
    CORDATA = get_cor_mat(COVDATA)

    # polynomial fit
    PARAMS, COVFIT = np.polyfit(XDATA, YDATA, PD, cov=True)
    CORFIT = get_cor_mat(COVFIT)

    # interpolation for plotting
    PLOTRANGE = np.linspace(XOF, XOF + NPOINTS, 100)                # x-axis
    TT = np.vstack([PLOTRANGE**(PD - i) for i in range(PD + 1)]).T  # polynom.
    YI = np.dot(TT, PARAMS)                                         # y-axis

    # y-axis errors from covariance matrix
    COVMAT = COVFIT if IS_CORR else np.diag(np.diag(COVFIT))
    COV_Y = np.dot(TT, np.dot(COVMAT, TT.T))

    # standard deviation (sigma)
    SIG_Y = np.sqrt(np.diag(COV_Y))

    # matrix output
    print("data covariance matrix:\n%s\n" % (COVDATA))
    print("data correlation matrix:\n%s\n" % (CORDATA))
    print("fit parameters (highest order first):\n%s\n" % (PARAMS))
    print("fit covariance matrix:\n%s\n" % (COVFIT))
    print("fit correlation matrix:\n%s\n" % (CORFIT))

    # plotting
    FG, AX = plt.subplots(1, 1)
    plt.fill_between(PLOTRANGE, YI + NSIG*SIG_Y, YI - NSIG*SIG_Y, alpha=.25)
    AX.plot(PLOTRANGE, YI, '-')
    AX.plot(XDATA, YDATA, 'ro')
    AX.axis('tight')

    FG.canvas.draw()
    plt.show()
