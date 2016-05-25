#!/usr/bin/env python3

import argparse
import numpy as np


def get_probability(pr_prior, pwr_prior, pwnr_prior):
    ''' returns probability P(R|W) from P(W|R) (pwr_prior) '''
    num = pr_prior * pwr_prior
    denum = num + (1 - pr_prior) * pwnr_prior
    return num / denum

if __name__ == "__main__":
    # input
    title = "Moderne Methoden der Datenanalyse - Blatt 02 Aufgabe 1.2"
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--n', type=int, default=100,
                        help='number of events to simulate')
    args = parser.parse_args()

    N = args.n

    # priors from sheet
    PR_PRIOR = np.array([0.05, 0.5, 0.95])
    PWR_PRIOR = np.array([0.8, 0.9])

    # rain probability from priors
    P_RW = get_probability(PR_PRIOR, PWR_PRIOR[0], 1 - PWR_PRIOR[1])
    P_RNW = get_probability(PR_PRIOR, 1 - PWR_PRIOR[0], PWR_PRIOR[1])

    # output (4 probabilites for each city)
    print("P(R|W)  :\t", np.round(100 * P_RW, 2))
    print("P(NR|W) :\t", np.round(100 * (1 - P_RW), 2))
    print("P(R|NW) :\t", np.round(100 * P_RNW, 2))
    print("P(NR|NW):\t", np.round(100 * (1 - P_RNW), 2))

    # generate N uniformly distributed random numbers in [0, 1) & events
    RAIN = [np.random.uniform() for _ in range(N)]
    FORECAST = [np.random.uniform() for _ in range(N)]

    rain_forecast = [0, 0, 0]
    sun_forecast = [0, 0, 0]
    rain_event = [0, 0, 0]
    sun_event = [0, 0, 0]

    for event, pred in zip(RAIN, FORECAST):
        for i in range(len(PR_PRIOR)):
            if event <= PR_PRIOR[i]:
                rain_event[i] += 1
                if pred <= PWR_PRIOR[0]:
                    rain_forecast[i] += 1
            if event > PR_PRIOR[i]:
                sun_event[i] += 1
                if pred <= PWR_PRIOR[1]:
                    sun_forecast[i] += 1

    correct_forecast_rain = np.divide(rain_forecast, rain_event)
    correct_forecast_sun = np.divide(sun_forecast, sun_event)
    print(np.round(100*correct_forecast_rain, 2))
    print(np.round(100*correct_forecast_sun, 2))
