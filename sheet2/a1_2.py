#!/usr/bin/env python3

import sys
import numpy as np

if __name__ == "__main__":
    try:
        N = int(sys.argv[1])
    except IndexError:
        N = 100

    FORECAST = np.random.uniform(size=N)
