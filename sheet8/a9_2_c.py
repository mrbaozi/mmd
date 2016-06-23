#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from frequentist import *


def bayes(v_t, prior, n_0=3, b=0):
    L = np.zeros_like(v_t)
    L[v_t > 0] = np.array([probability(n_0, v, b) for v in v_t[v_t > 0]])
    return (L * prior) / np.sum(L * prior)

def bayesian_approach():
    v_tB = 0
    n_0 = 3
    v_t = np.linspace(0, 15, 100)

    # prior (i)
    prior = np.full_like(v_t, 1)
    bayes_i = bayes(v_t, prior)

    # prior (ii)
    prior = np.zeros_like(v_t)
    prior[v_t > 0] = np.array([1. / v for v in v_t[v_t > 0]])
    bayes_ii = bayes(v_t, prior)

    # find 90% CL cut
    cdf_i = np.array([np.sum(bayes_i[:i+1]) for i in range(len(bayes_i))])
    cdf_ii = np.array([np.sum(bayes_ii[:i+1]) for i in range(len(bayes_ii))])

    # print result for comparison with other methods
    print("Method c), constant prior:")
    print("CL_SB(n_0={}, v_tB={}) = {}".format(n_0, v_tB, v_t[cdf_i > 0.9][0]))
    print("Method c), 1 / v_t prior:")
    print("CL_SB(n_0={}, v_tB={}) = {}".format(n_0, v_tB, v_t[cdf_ii > 0.9][0]))

    # plots
    fg, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(v_t, bayes_i, 'r-', label='prior = const.')
    ax[0].plot(v_t, bayes_ii, 'b-', label='prior = 1 / v_t')
    ax[0].set_title('Bayesian approach, n_0 = {}, v_tB = {}'.format(n_0, v_tB))
    ax[0].set_ylabel('PDF(v_t | n_0)')
    ax[0].axis('tight')
    ax[0].legend(loc='best')

    ax[1].plot(v_t, cdf_i, 'r-')
    ax[1].plot(v_t, cdf_ii, 'b-')
    ax[1].axvline(v_t[cdf_i > 0.9][0], color='r', linestyle='dashed',
                  label='90% CL upper limit')
    ax[1].axvline(v_t[cdf_ii > 0.9][0], color='b', linestyle='dashed',
                  label='90% CL upper limit')
    ax[1].set_xlabel('v_t')
    ax[1].set_ylabel('CDF(v_t | n_0)')
    ax[1].axis('tight')
    ax[1].legend(loc='best')

if __name__ == "__main__":
    bayesian_approach()
    plt.show()
