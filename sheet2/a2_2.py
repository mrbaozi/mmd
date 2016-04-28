#!/usr/bin/env python3

import argparse
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

    # input
    TITLE = "Moderne Methoden der Datenanalyse - Blatt 02 Aufgabe 2.2"
    PARSER = argparse.ArgumentParser(description=TITLE)
    PARSER.add_argument('--corr', type=int, default=0,
                        help='specify if fit arguments are correlated')
    PARSER.add_argument('--sig', type=int, default=1,
                        help='specify width of error band (n * sigma)')
    PARSER.add_argument('--xof', type=int, default=10,
                        help='start value for x')
    PARSER.add_argument('--npoints', type=int, default=10,
                        help='number of random points for fit')
    PARSER.add_argument('--pd', type=int, default=1,
                        help='polynomial degree for fit')
    PARSER.add_argument('--xerr', type=float, default=.5,
                        help='standard error for x-data')
    PARSER.add_argument('--yerr', type=float, default=.5,
                        help='standard error for y-data')
    ARGS = PARSER.parse_args()

    # assign variables
    IS_CORR = ARGS.corr
    NSIG = ARGS.sig
    XOF = ARGS.xof
    NPOINTS = ARGS.npoints
    PD = ARGS.pd
    XERR = ARGS.xerr
    YERR = ARGS.yerr

    # generate random data and errors
    X = np.array([x for x in range(XOF, XOF + NPOINTS, 1)])
    Y = np.array([y for y in range(XOF, XOF + NPOINTS, 1)])
    EX = np.array([np.random.normal(0, XERR) for _ in range(NPOINTS)])
    EY = np.array([np.random.normal(0, YERR) for _ in range(NPOINTS)])
    XDATA = np.add(X, EX)
    YDATA = np.add(Y, EY)

    # data covariance & correlation
    COVDATA = np.cov(XDATA, YDATA)
    CORDATA = get_cor_mat(COVDATA)

    # polynomial fit
    PARAMS, COVFIT = np.polyfit(XDATA, YDATA, PD, cov=True)
    CORFIT = get_cor_mat(COVFIT)

    # interpolation for plotting
    PLOTRANGE = np.linspace(min(XDATA) - 1, max(XDATA) + 1, 200)    # x-axis
    TT = np.vstack([PLOTRANGE**(PD - i) for i in range(PD + 1)]).T  # polynom.
    YI = np.dot(TT, PARAMS)                                         # y-axis

    # y-axis errors from covariance matrix
    COVMAT = COVFIT if IS_CORR else np.diag(np.diag(COVFIT))
    COV_Y = np.dot(TT, np.dot(COVMAT, TT.T))

    # standard deviation (sigma)
    VAR_Y = np.diag(COV_Y)
    SIG_Y = np.sqrt(VAR_Y)

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
    AX.errorbar(XDATA, YDATA, xerr=XERR, yerr=YERR, fmt='ro')
    AX.axis('tight')

    FG.canvas.draw()
    plt.show()
