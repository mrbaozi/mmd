#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import chi2


def main():
    track = [1, 2, 3, 4, 5, 6, 7, 8]
    winners = [29, 19, 18, 25, 17, 10, 15, 11]
    nraces = sum(winners)
    losers = [nraces - x for x in winners]
    nwin = nraces
    nloose = sum(losers)

    test = [(win - 18.)**2 / 18. for win in winners]
    test_stat = sum(test)
    print("Chi2 (Gewinner): %s" % (round(test_stat, 2)))

    prob_test = round((1 - chi2.cdf(test_stat, 7)) * 100, 2)
    print("%s%%" % (prob_test))

if __name__ == "__main__":
    main()
