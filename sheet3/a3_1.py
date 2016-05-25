#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt


def get_args():
    '''
    get the required arguments for the program to run
    '''
    title = "Moderne Methoden der Datenanalyse - Blatt 03 Aufgabe 3.1"
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--samples', type=int, default=1000,
                        help='number of samples')
    parser.add_argument('--n', type=int, default=10,
                        help='number of values (per sample)')
    parser.add_argument('--tau', type=int, default=1,
                        help='lifetime tau')
    return parser.parse_args()


def main(samples, n, tau):
    # get exponential distribution from uniform
    x = [[np.random.uniform(0, 1, n)] for _ in range(samples)]
    t = -tau * np.log(tau * x)

    tmean = [np.mean(ti) for ti in t]
    lmean = [np.mean(1/ti) for ti in t]

    # plotting
    fg, ax = plt.subplots(1, 1)
    ax.hist(lmean, 50, normed=1)
    fg.canvas.draw()
    plt.axis([0, 8, 0, 0.3])
    plt.show()

if __name__ == "__main__":
    args = get_args()
    main(args.samples, args.n, args.tau)
