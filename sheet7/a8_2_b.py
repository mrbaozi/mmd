#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


if __name__ == "__main__":
    tdist = np.array([35, 218, 814, 1069, 651, 195, 18])
    g_obs = np.array([99, 386, 695, 877, 618, 254, 71])

    diag_m = np.diag(np.full_like(tdist, 0.4, dtype=np.double))
    diag_u = np.diag(np.full(len(tdist) - 1, 0.3, dtype=np.double), 1)
    diag_l = np.diag(np.full(len(tdist) - 1, 0.3, dtype=np.double), -1)
    R_a = diag_m + diag_u + diag_l
    R_a[0, 0], R_a[-1, -1] = 0.7, 0.7

    g = R_a.dot(tdist)

    # results
    print(tdist)
    print(g)
    print(g.dot(la.inv(R_a)))

    # plot
    N = len(tdist)
    width = 0.25
    fg, ax = plt.subplots(1, 1)
    ax.bar(np.arange(N), tdist, width, color='r', label='true')
    ax.bar(np.arange(N) + width, g_obs, width, color='g', label='observed')
    ax.bar(np.arange(N) + 2 * width, g_obs.dot(la.inv(R_a)), width, color='b', label='unfolded')
    plt.legend(loc='best')
    plt.show()
