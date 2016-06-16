#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def chi2_a(x, y, sig, eps):
    return np.sum(((x[1] * y - x[0]) / sig)**2) + ((x[1] - 1) / eps)**2

def chi2_b(x, y, sig, eps):
    return np.sum(((y - x[1] * x[0]) / sig)**2) + ((x[1] - 1) / eps)**2

def vphi(phi):
    return np.array([np.cos(phi), np.sin(phi)])

if __name__ == "__main__":
    y = np.array([8., 8.5])
    f = 1.
    eps = 0.1
    sig = 0.02 * y

    # minimize
    res1 = minimize(chi2_a, (1, 1), args=(y, sig, eps))
    res2 = minimize(chi2_b, (1, 1), args=(y, sig, eps))

    # construct covariance matrix
    v1 = np.diag(sig**2) + eps**2 * np.outer(y, y)
    va = np.diag(sig**2) + eps**2 * np.full_like(np.diag(sig), res1.x[0]**2)
    vb = np.diag(sig**2) + eps**2 * np.full_like(np.diag(sig), res2.x[0]**2)

    # output
    print("V_a:")
    print(va)
    print("\nV_b:")
    print(vb)
    print("\n\nminimization result:")
    print("\nmethod a:")
    print(res1)
    print("\nmethod b:")
    print(res2)

    # ellipse
    n = 1000
    s = []
    e, e1, e2 =[], [], []
    for iphi in range(n):
        phi = vphi(2. * iphi * np.pi / float(n))
        r = 1. / np.sqrt((phi.T).dot(np.linalg.inv(v1)).dot(phi))
        r1 = 1. / np.sqrt((phi.T).dot(np.linalg.inv(va)).dot(phi))
        r2 = 1. / np.sqrt((phi.T).dot(np.linalg.inv(vb)).dot(phi))
        e.append(y + 2 * r * phi)
        e1.append(y + 2 * r1 * phi)
        e2.append(y + 2 * r2 * phi)
    e, e1, e2 = np.array(e), np.array(e1), np.array(e2)

    # plot
    fg, ax = plt.subplots(1, 1)
    ax.plot(*zip(*e), 'r-', label=r'exercise 4.2 a')
    ax.plot(*zip(*e1), 'g-', label=r'method a')
    ax.plot(*zip(*e2), 'b-', label=r'method b')
    ax.plot((6, 11), (6, 11), 'k-', label=r'$y_1 = y_2$')
    ax.plot(*y, 'ko', label=r'center')
    ax.plot(res1.x[0], res1.x[0], 'go', label=r'method a')
    ax.plot(res2.x[0], res2.x[0], 'bo', label=r'method b')
    ax.set_title("exercise 4.2 b")
    ax.set_xlabel(r"$y_1$")
    ax.set_ylabel(r"$y_2$")
    ax.grid()
    ax.axis('tight')
    plt.legend(loc='best')
    plt.show()
