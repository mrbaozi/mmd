#!/usr/bin/env python3
#! -*- coding: utf-8 -*-

import argparse
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def get_input():
    title = "Moderne Methoden der Datenanalyse - Blatt 06 Aufgabe 7.1"
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--pd', type=int, default=2,
                        help='polynomial degree')
    return parser.parse_args()

def P(x, *p):
    y = np.zeros_like(x)
    for i in range(len(p)):
        y = y + p[i] * x**i
    return y

def L(x, *p):
    y = np.zeros_like(x)
    for i in range(len(p)):
        y = y + p[i] * legendre(x, i)
    return y

def legendre(x, k):
    if k == 0:
        return 1
    if k == 1:
        return x
    return ((2*k - 1) * x * legendre(x, k-1) - (k-1) * legendre(x, k-2)) / float(k)

def get_cor_mat(covmat):
    D = np.linalg.inv(np.diag(np.sqrt(np.diag(covmat))))
    return D.dot(covmat).dot(D)

if __name__ == "__main__":
    args = get_input()
    pd = args.pd
    data = np.loadtxt('./a7_1_data.txt')
    data_x, data_y = data.T
    sig_y = np.full_like(data_y, 0.5)

    # fits
    p0 = np.ones(pd + 1)
    popt_p, pcov_p = curve_fit(P,
                               data_x,
                               data_y,
                               p0=p0,
                               sigma=sig_y,
                               method='trf')
    popt_l, pcov_l = curve_fit(L,
                               data_x,
                               data_y,
                               p0=p0,
                               sigma=sig_y,
                               method='trf')
    pcor_p = get_cor_mat(pcov_p)
    pcor_l = get_cor_mat(pcov_l)

    # output
    np.set_printoptions(precision=5)
    print("\nbest fit (polynomial / legendre):")
    print(popt_p)
    print(popt_l)
    print("\ncorrelations (polynomial / legendre):")
    print(pcor_p)
    print(pcor_l)
    print("\naverage correlation (polynomial / legendre):")
    iu = np.triu_indices(pd + 1, 1)
    il = np.tril_indices(pd + 1, -1)
    print(np.absolute(np.concatenate((pcor_p[iu], pcor_p[il]))).mean())
    print(np.absolute(np.concatenate((pcor_l[iu], pcor_l[il]))).mean())

    # plot
    rng = np.linspace(min(data_x), max(data_x), 1000)
    fit_p = P(rng, *popt_p)
    fit_l = L(rng, *popt_l)

    fg, ax = plt.subplots(1, 1)
    ax.errorbar(data_x, data_y, yerr=sig_y, fmt='kx')
    ax.plot(rng, fit_p, 'r', rng, fit_l, 'b')
    ax.legend(['Polynomial', 'Legendre', 'Data'], loc='best')
    ax.axis('tight')
    plt.show()
