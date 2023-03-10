#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
# from numpy.random import poisson


def regularize_goal(R, obs, f0, goal=0.1):
    '''
    regularization, takes an accuracy goal as input (for exercise a)
    '''
    g = np.empty_like(obs) # initialize result array
    f = np.copy(f0)        # don't modify f0
    i = 0                  # counter
    while not np.allclose(g, obs, rtol=goal):
        g = R.dot(f)
        c = obs / g
        f *= R.dot(c)
        i += 1
    return g, i

def regularize_iter(R, obs, f0, iterations=10):
    '''
    regularization, takes the number of iterations as input (for exercise b)
    '''
    f = np.copy(f0) # don't modify f0
    for _ in range(iterations):
        g = R.dot(f)
        c = obs / g
        f *= R.dot(c)
    return g

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


    # regularize
    f0 = np.full(len(tdist), sum(tdist) / float(len(tdist))) # 3000 / 7
    g_a, i = regularize_goal(R_a, tdist, f0, 0.1)            # goal < 0.1 fails

    # ouput
    print("iterations: {}".format(i))
    print("true     observed   delta")
    for x, y in zip(tdist, g_a):
        print("{:6.1f}   {:8.3f}   {:8.3f}".format(x, y, np.abs(x - y)))
