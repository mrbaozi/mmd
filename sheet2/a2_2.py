#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

if __name__ == "__main__":
    NPOINTS = 10
    XOF = 10

    X = [x for x in range(XOF, XOF + NPOINTS, 1)]
    Y = [y for y in range(XOF, XOF + NPOINTS, 1)]
    EX = [np.random.normal(0, 0.5) for _ in range(NPOINTS)]
    EY = [np.random.normal(0, 0.5) for _ in range(NPOINTS)]

    b, a, r_value, p_value, std_err = stats.linregress(X + EX,Y + EY)
    print(b)
    print(a)
    print(r_value)
    print(p_value)
    print(std_err)
