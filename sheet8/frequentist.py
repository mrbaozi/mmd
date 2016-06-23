#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ROOT import TMath, TF1


def probability(n, s, b):
    '''
    Returns the probability to observe n events if you expect s signal events
    and b background events
    Equivalent to: exp(-s-b)*(s+b)**n/TMath::Factorial(n)
    '''
    return TMath.PoissonI(n, s + b)

def pvalue_function_classical(x, par):
    '''
    If you expect s signal events and b background events, return the
    probability to observe more than n events: used for classical upper-limits
    on the total number of events (signal + background): CL_SB
    '''
    s = x[0]
    n = par[0]
    b = par[1]

    pvalue_sum = 0.
    for i in range(int(round(n)+1)):
      pvalue_sum += probability(i, s, b)

    if (s + b < 0):
        pvalue_sum = 1.  # s+b should be positive

    return pvalue_sum

def pvalue_function_cls(x, par):
    '''
    If you expect s signal events and b background events, return the
    probability to observe more than n events: used for normalized upper-limits
    on the total number of signal events: CL_S = CL_SB / CL_B
    '''

    s = x[0]
    n = par[0]
    b = par[1]

    pvalue_sum = 0.
    for i in range(int(round(n)+1)):
        pvalue_sum += probability(i, s, b)

    if (s + b < 0):
        pvalue_sum = 1. # s+b should be positive

    pvalue_sum2 = 0.    # probability to observe n or less events if no signal is expected
    for i in range(int(round(n)+1)):
        pvalue_sum2 += probability(i, 0, b)

    result = pvalue_sum / pvalue_sum2

    return result

def limit(n, b, pvalue, normalizeCLs=0):
    '''
    Compute the value of s for which you reach the given p-value, if you
    observe n events and expect b background events
    set normalizeCLs to != 0 to use CL_S instead of CL_SB
    '''

    min = -1        # almost -infinity, since negetive values are not meaningful
    max = +10*(n+1) # almost +infinity on relevant scale

    # function of one parameter varying between min and max
    pvf1 = TF1("pvf1", pvalue_function_classical, min, max, 2)
    pvf2 = TF1("pvf2", pvalue_function_cls, min, max, 2)

    pvf1.SetParameter(0, n)
    pvf1.SetParameter(1, b)
    pvf2.SetParameter(0, n)
    pvf2.SetParameter(1,b)

    limit = 0
    if normalizeCLs:
        limit = pvf2.GetX(pvalue, min, max)
    else:
        limit = pvf1.GetX(pvalue, min, max)

    return limit
