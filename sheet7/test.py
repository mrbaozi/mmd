#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def regularize(R, obs, thresh=0):
    w, v = la.eig(R)
    R_diag = (la.inv(v)).dot(R_a).dot(v)
    g_diag = la.inv(v).dot(obs) * 1. / w
    for i in range(len(g_diag)):
        if w[i] < thresh:
            g_diag[i] = 0
    return w, v, g_diag

def min_reg(l_reg, dist, R, g):
    _, v, g_diag = regularize(R, g, l_reg)
    return np.average(np.abs(dist - v.dot(g_diag)))

if __name__ == "__main__":
    # data
    tdist = np.array([35, 218, 814, 1069, 651, 195, 18], dtype=np.double)
    g_obs = np.array([99, 386, 695, 877, 618, 254, 71], dtype=np.double)

    # construct migration matrix
    diag_m = np.diag(np.full_like(tdist, 0.4, dtype=np.double))
    diag_u = np.diag(np.full(len(tdist) - 1, 0.3, dtype=np.double), 1)
    diag_l = np.diag(np.full(len(tdist) - 1, 0.3, dtype=np.double), -1)
    R_a = diag_m + diag_u + diag_l
    R_a[0, 0], R_a[-1, -1] = 0.7, 0.7

    # print values for different l_reg
    for lreg in np.arange(-1, 1.1, 0.1):
        print("lreg: {:4.1f}   avg: {:6.3f}".format(lreg, min_reg(lreg, tdist, R_a, g_obs)))

    # regularize
    reg = 0.1
    w, v, g_diag = regularize(R_a, g_obs, reg)

    # plot
    N = len(tdist)
    width = 0.25
    fg, ax = plt.subplots(1, 1)
    ax.bar(np.arange(N), tdist, width, color='r', label='true')
    ax.bar(np.arange(N) + width, g_obs, width, color='g', label='observed')
    ax.bar(np.arange(N) + 2 * width, v.dot(g_diag), width, color='b', label='unfolded')
    title = r'$\lambda_{reg} = %s$ ' % (reg)
    ax.set_title(title)
    plt.legend(loc='best')
    plt.show()
