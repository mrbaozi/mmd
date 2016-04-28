#!/usr/bin/env python3

import sys
import numpy as np


def get_probability(pr_prior, pwr_prior, pwnr_prior):
    ''' returns probability P(R|W) from P(W|R) (pwr_prior) '''
    num = pr_prior * pwr_prior
    denum = num + (1 - pr_prior) * pwnr_prior
    return num / denum

if __name__ == "__main__":
    try:
        N = int(sys.argv[1])
    except IndexError:
        N = 100

    # priors from sheet
    PR_PRIOR = np.array([0.05, 0.5, 0.95])
    PWR_PRIOR = np.array([0.8, 0.9])

    # rain probability from priors
    P_RW = get_probability(PR_PRIOR, PWR_PRIOR[0], 1 - PWR_PRIOR[1])
    P_RNW = get_probability(PR_PRIOR, 1 - PWR_PRIOR[0], PWR_PRIOR[1])

    # output (4 probabilites for each city)
    print("P(R|W) :\t", np.round(100 * P_RW, 2))
    print("P(NR|W):\t", np.round(100 * (1 - P_RW), 2))
    print("P(R|NW):\t", np.round(100 * P_RNW, 2))
    print("P(R|NW):\t", np.round(100 * (1 - P_RNW), 2))

    # generate N uniformly distributed random numbers in [0, 1) & events
    RAIN = [np.random.uniform() for _ in range(N)]
    FORECAST = [np.random.uniform() for _ in range(N)]
