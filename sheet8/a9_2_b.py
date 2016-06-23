#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize, brentq
import matplotlib.pyplot as plt

from frequentist import *


def log_likelihood(v_t, n_0, b=0):
    return np.log(probability(n_0, v_t, b))

def delta_ln_L(v_t, ln_L_min, n_0, b=0):
    return ln_L_min - log_likelihood(v_t, n_0, b)

def likelihood_approach():
    # init
    v_tB = 0
    n_0 = 3
    upper_limit = 10
    n_points = 100
    rng = np.linspace(0.1, upper_limit, n_points)
    ln_L = np.zeros(n_points)

    # get v_tS points for all n
    for v_t in range(n_points):
        ln_L[v_t] = -2 * log_likelihood(rng[v_t], n_0)

    # find minimum
    res = minimize(lambda x: -2 * log_likelihood(x, n_0), x0=(1.))
    ln_L_min_x = res.x[0]
    ln_L_min_y = log_likelihood(ln_L_min_x, n_0)

    # calculate CLs
    cut68 = np.sqrt(2) * TMath.ErfInverse(2 * 0.68 - 1) # 68% CL
    cut90 = np.sqrt(2) * TMath.ErfInverse(2 * 0.9 - 1)  # 90% CL

    # find y values of cuts
    cut68_y = -2 * ln_L_min_y + cut68
    cut90_y = -2 * ln_L_min_y + cut90

    # find roots (GetX)
    def errfunc(x, cut):
        return -2 * log_likelihood(x, n_0) - cut
    lo68 = brentq(errfunc, 0.1, ln_L_min_x, args=(cut68_y))
    hi68 = brentq(errfunc, ln_L_min_x, 10, args=(cut68_y))
    lo90 = brentq(errfunc, 0.1, ln_L_min_x, args=(cut90_y))
    hi90 = brentq(errfunc, ln_L_min_x, 10, args=(cut90_y))

    # print upper limit for comparison
    print("Method b):")
    print("CL_SB(n_0={}, v_tB= {}) = {}".format(n_0, v_tB, hi90))

    # plots
    fg, ax = plt.subplots(1, 1)
    ax.plot(rng, ln_L, 'k-', label='-2 ln(L)')          # -2 ln(L)
    ax.plot(ln_L_min_x, -2 * ln_L_min_y, 'ro',          # minimum
            label='L_min = ({:.1f}, {:.1f})'.format(ln_L_min_x, -2 * ln_L_min_y))
    ax.axvspan(lo68, hi68,  # 68% CL
               alpha=0.2, color='blue', linewidth=2, hatch='/', label='68% CL')
    ax.axvspan(lo90, hi90,  # 90% CL
               alpha=0.3, color='blue', label='90% CL')
    ax.axvline(hi90, color='red',         # 90% upper lim
               label='upper limit = {:.3f}'.format(hi90))
    ax.set_title('Likelihood approach, n_0 = {}, v_tB = {}'.format(n_0, v_tB))
    ax.set_xlabel('v_t')
    ax.set_ylabel('-2 ln(L)')
    ax.axis('tight')
    ax.set_ylim((min(ln_L) - 0.5, None))
    plt.legend(loc='best')

if __name__ == "__main__":
    likelihood_approach()
    plt.show()
