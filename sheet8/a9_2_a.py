#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from frequentist import *


def no_background():
    v_tB = 0
    n_0 = 3
    lim = limit(n_0, v_tB, 0.1, func=pvalue_function_classical)
    print("Method a):")
    print("CL_SB(n_0={}, v_tB={}) = {}".format(n_0, v_tB, lim))

def background(func):
    # init
    n = 6
    CL_SB = 0.1 # this should be either 10% or 90%, probably 10%
    upper_limit = 10
    n_points = 100
    v_tS = np.zeros((n, n_points))
    rng = np.linspace(0, upper_limit, n_points)

    # get v_tS points for all n
    for n_0 in range(n):
        for v_tB in range(n_points):
            v_tS[n_0, v_tB] = limit(n_0, rng[v_tB], 1 - CL_SB, func)

    # plots
    fg, ax = plt.subplots(1, 1)
    for i in range(len(v_tS)):
        ax.plot(rng, v_tS[i], label=r'n_0 = {}'.format(i))
    ax.set_title('Frequentist approach, func = {}'.format(func.__name__))
    ax.set_xlabel('v_tB')
    ax.set_ylabel('v_tS')
    ax.axis('tight')
    ax.legend(loc='best')

if __name__ == "__main__":
    no_background()
    background(func=pvalue_function_classical)
    background(func=pvalue_function_cls)
    plt.show()
