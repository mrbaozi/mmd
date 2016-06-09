#!/usr/bin/env python3
#! -*- coding: utf-8 -*-

import argparse
import numpy as np
from numpy.polynomial import Legendre, Polynomial
import matplotlib.pyplot as plt


def get_input():
    title = "Moderne Methoden der Datenanalyse - Blatt 06 Aufgabe 7.1"
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--pd', type=int, default=2,
                        help='polynomial degree')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_input()
    pd = args.pd
    data = np.loadtxt('./a7_1_data.txt').astype(np.double)
    data_x, data_y = data.T
    sig_y = np.full_like(data_y, 0.5)

    # fits
    pl = Legendre.fit(data_x, data_y, pd)
    pn = Polynomial.fit(data_x, data_y, pd)

    # output
    np.set_printoptions(precision=5)
    print("\nbest fit (polynomial / legendre):")
    print(pn.coef)
    print(pl.coef)

    # plot
    fg, ax = plt.subplots(1, 1)
    ax.errorbar(data_x, data_y, yerr=sig_y, fmt='kx')
    ax.plot(*pn.linspace(), 'r', *pl.linspace(), 'b')
    ax.legend(['Polynomial', 'Legendre', 'Data'], loc='best')
    ax.axis('tight')
    plt.show()
