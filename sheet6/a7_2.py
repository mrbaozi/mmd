#!/usr/bin/env python3
#! -*- coding: utf-8 -*-

import argparse
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import gridspec


def get_input():
    title = "Moderne Methoden der Datenanalyse - Blatt 06 Aufgabe 7.2"
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--pd', type=int, default=2,
                        help='polynomial degree')
    parser.add_argument('--xl', type=float, default=0.75,
                        help='lower bound')
    parser.add_argument('--xh', type=float, default=1.25,
                        help='upper bound')
    return parser.parse_args()

def P(x, *p):
    y = np.zeros_like(x)
    for i in range(len(p)):
        y = y + p[i] * x**i
    return y

def L(x, *p):
    y = np.zeros_like(x)
    for i in range(0, len(p), 3):
        y = y + (p[i+1] * p[i+2] / 2.) / (np.pi * ((x - p[i])**2 + (p[i+2] / 2.)**2))
    return y

def PL(x, *p):
    skip = int(p[0]) + 2
    return L(x, *p[skip:]) + P(x, *p[1:skip])

def get_parameter_array(pd, candidates):
    return np.concatenate(((pd,), np.ones(pd + 1), candidates))

def method_a(data_x, data_y, pd, peak):
    sig_y = np.sqrt(data_y)
    p0 = get_parameter_array(pd, peak)
    popt, pcov = curve_fit(PL,
                           data_x,
                           data_y,
                           p0=p0,
                           sigma=sig_y,
                           method='trf')
    return popt, pcov

def method_b(data_x, data_y, pd, peak, xl, xh):
    sig_y = np.sqrt(data_y) # errors

    # fit polynomial
    idb = np.where((data_x < xl) | (data_x > xh)) # index of background values
    p0_p = np.ones(pd + 1)
    popt_p, pcov_p = curve_fit(P,
                               data_x[idb],
                               data_y[idb],
                               p0=p0_p,
                               sigma=sig_y[idb],
                               method='trf')

    # subtract polynomial from data
    fit_y = data_y - P(data_x, *popt_p)

    # fit lorentz
    ids = np.where((data_x >= xl) & (data_x <= xh)) # index of signal values
    p0_l = np.array(peak)
    popt_l, pcov_l = curve_fit(L,
                               data_x[ids],
                               fit_y[ids],
                               p0=p0_l,
                               sigma=sig_y[ids],
                               method='trf')

    return popt_p, pcov_p, popt_l, pcov_l

if __name__ == "__main__":
    args = get_input()
    data = np.loadtxt('./a7_2_data.txt')
    x, y = data.T
    pd = args.pd
    xl, xh = args.xl, args.xh
    peak = [1, 50, 0.25]
    popt_pl1, pcov_pl1 = method_a(x, y, pd, peak)
    popt_p2, pcov_p2, popt_l2, pcov_l2 = method_b(x, y, pd, peak, xl, xh)

    # assignments
    rng = np.linspace(min(x), max(x), 1000)
    skip = pd + 2
    popt_p1 = popt_pl1[1:skip]
    popt_l1 = popt_pl1[skip:]
    pcov_p1 = pcov_pl1[1:skip]
    pcov_l1 = pcov_pl1[skip:]
    fit_p1 = P(rng, *popt_p1)
    fit_l1 = L(rng, *popt_l1)
    fit_p2 = P(rng, *popt_p2)
    fit_l2 = L(rng, *popt_l2)
    fit_pl1 = PL(rng, *popt_pl1)
    fit_pl2 = fit_p2 + fit_l2

    # output
    np.set_printoptions(precision=5)
    print("\npolynomial parameters (linear combination / background fit)")
    print(popt_p1)
    print(popt_p2)
    print("\npolynomial variance (linear combination / background fit)")
    print(np.diag(pcov_p1))
    print(np.diag(pcov_p2))
    print("\nlorentz parameters (linear combination / background fit)")
    print(popt_l1)
    print(popt_l2)
    print("\nlorentz variance (linear combination / background fit)")
    print(np.diag(pcov_l1))
    print(np.diag(pcov_l2))

    # plot
    fg = plt.figure(figsize=(15, 7))
    gs = gridspec.GridSpec(2, 2)
    ax1 = fg.add_subplot(gs[:, 0])
    ax1.errorbar(x, y, yerr=np.sqrt(y), fmt='kx')
    ax1.plot(rng, fit_pl1, 'r', rng, fit_pl2, 'b')
    ax1.legend(['Method 1', 'Method 2', 'Data'], loc='best')
    ax1.axis('tight')
    ax2 = fg.add_subplot(gs[0, 1])
    ax2.plot(rng, fit_p1, 'r', rng, fit_p2, 'b')
    ax2.legend(['Method 1', 'Method 2'], loc='best')
    ax2.axis('tight')
    ax3 = fg.add_subplot(gs[1, 1])
    ax3.plot(rng, fit_l1, 'r', rng, fit_l2, 'b')
    ax3.legend(['Method 1', 'Method 2'], loc='best')
    ax3.axis('tight')

    gs.update(wspace=0.1, hspace=0.2)
    plt.show()
