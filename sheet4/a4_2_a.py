#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def eval_chi2(x, y, va):
    delta = y - x
    chi2 = (delta.T).dot(np.linalg.inv(va)).dot(delta)
    return chi2

def vphi(phi):
    return np.array([np.cos(phi), np.sin(phi)])

if __name__ == "__main__":
    y = np.array([8., 8.5])
    eps = 0.1
    sig = 0.02 * y

    # construct covariance matrix
    va = np.diag(sig**2) + eps**2 * np.outer(y, y)

    # minimize
    res = minimize(eval_chi2, 1, args=(y, va))

    # output
    print("V_a:")
    print(va)
    print("\nminimization result:")
    print(res)

    # ellipse
    n = 1000
    s = []
    e1, e2 =[], []
    for iphi in range(n):
        phi = vphi(2. * iphi * np.pi / float(n))
        r = 1. / np.sqrt((phi.T).dot(np.linalg.inv(va)).dot(phi))
        e1.append(y + r * phi)
        e2.append(y + 2 * r * phi)
    e1, e2 = np.array(e1), np.array(e2)

    # plot
    fg, ax = plt.subplots(1, 1)
    ax.plot(*zip(*e1), label=r'$c = 1$')
    ax.plot(*zip(*e2), label=r'$c = 2$')
    ax.plot((6, 11), (6, 11), label=r'$y_1 = y_2$')
    ax.plot(*y, '^', label=r'center')
    ax.plot(res.x, res.x, '^', label=r'minimize')
    ax.set_title("exercise 4.2 a")
    ax.set_xlabel(r"$y_1$")
    ax.set_ylabel(r"$y_2$")
    ax.grid()
    ax.axis('tight')
    plt.legend(loc='best')
    plt.show()
