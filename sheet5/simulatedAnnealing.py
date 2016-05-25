#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt


def get_input():
    title = "Moderne Methoden der Datenanalyse - Blatt 05 Aufgabe 6.1"
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument('--debug', type=int, default=0,
                        help='give debug output for annealing steps')
    parser.add_argument('--startx', type=float, default=0,
                        help='initial value for x')
    parser.add_argument('--starty', type=float, default=0,
                        help='initial value for y')
    parser.add_argument('--t1', type=float, default=100,
                        help='initial temperature')
    parser.add_argument('--t2', type=float, default=1,
                        help='final temperature')
    parser.add_argument('--step', type=float, default=1,
                        help='step size (variance of normal distribution)')
    parser.add_argument('--speed', type=float, default=1,
                        help='cooling speed')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of simulations')
    parser.add_argument('--path', type=int, default=0,
                        help='show path of particles')
    parser.add_argument('--rnd', type=int, default=0,
                        help='use random seed')
    parser.add_argument('--plot', type=int, default=0,
                        help='generate plots')
    parser.add_argument('--silent', type=int, default=1,
                        help='suppress output for best (x, y)')
    parser.add_argument('--qbox', type=int, default=1,
                        help='print pretty quadrant box')
    return parser.parse_args()


def modified_rosenbrock_function(params, abc):
    '''
    The modified Rosenbrock function as described in exercise 6
    '''
    x, y = params
    a, b, c = abc
    return (x**2 + y - a)**2 + (x + y**2 - b)**2 + c * (x + y)**2


def simulated_annealing(initialXvalue=0, initialYvalue=0, params=[11, 7, 0.1],
                        initialTemperature=100, finalTemperature=1,
                        coolingSpeed=1, stepSize=1, rnd=0, debug=0):
    '''
    Code fragment for exercise 6.2 of the Computerpraktikum Datenanlyse 2014
    Authors: Ralf Ulrich, Frank Schroeder (Karlsruhe Institute of Technology)
    This code fragment probably is not the best and fastest implementation
    for "simulated annealing", but it is a simple implementation which does its job.
    '''
    nParameter = 2  # 2 parameters: x and y

    # Current parameters and cost
    initialXvalue = np.random.uniform(-4, 4) if rnd else initialXvalue
    initialYvalue = np.random.uniform(-4, 4) if rnd else initialYvalue
    currentParameters = [initialXvalue, initialYvalue]
    currentFunctionValue = modified_rosenbrock_function(currentParameters, params)  # you have to implement the function first!

    # keep reference of best parameters
    bestParameters = currentParameters
    bestFunctionValue = currentFunctionValue

    # Heat the system
    temperature = initialTemperature

    iteration = 0
    plog = []  # parameter log for debug output
    listOfPoints = [[initialXvalue, initialYvalue]]

    # Start to slowly cool the system
    while temperature > finalTemperature:

        # Change parameters
        newParameters = [0]*nParameter

        for ipar in range(nParameter):
            newParameters[ipar] = np.random.normal(currentParameters[ipar], stepSize)

        # Get the new value of the function
        newFunctionValue = modified_rosenbrock_function(newParameters, params)

        # Compute Boltzman probability
        deltaFunctionValue = newFunctionValue - currentFunctionValue
        saProbability = np.exp(-deltaFunctionValue / temperature)

        # Acceptance rules :
        # if newFunctionValue < currentFunctionValue then saProbability > 1
        # else accept the new state with a probability = saProbability
        # if (saProbability > gRandom.Uniform()):
        if saProbability > np.random.uniform():
            currentParameters = newParameters
            currentFunctionValue = newFunctionValue
            listOfPoints.append(currentParameters)  # log keeping: keep track of path

        if currentFunctionValue < bestFunctionValue:
            bestFunctionValue = currentFunctionValue
            bestParameters = currentParameters

        if debug:
            plog.append([temperature, currentParameters, currentFunctionValue, deltaFunctionValue])

        # Cool the system
        temperature *= 1 - coolingSpeed/100.
        iteration += 1

    if debug:
        print("{:9}  {:8}  {:8}  {:9}  {:10}".format("temp", "x", "y", "value", "delta"))
        for row in plog:
            txt = "{:9.5f}  {:8.5f}  {:8.5f}  {:9.5f}  {:10.5f}"
            print(txt.format(row[0], row[1][0], row[1][1], row[2], row[3]))

    return bestParameters, listOfPoints


def eval_annealing(args):
    data = []
    paths = []
    for _ in range(args.runs):
        start = time.time()
        result, path = simulated_annealing(initialXvalue=args.startx,
                                                     initialYvalue=args.starty,
                                                     initialTemperature=args.t1,
                                                     finalTemperature=args.t2,
                                                     coolingSpeed=args.speed,
                                                     stepSize=args.step,
                                                     rnd=args.rnd,
                                                     debug=args.debug)
        end = time.time()
        data.append([result[0], result[1], end - start])
        if args.path:
            paths.append(path)
        if not args.silent:
            print("best (x, y): {:11.8f}  {:11.8f}  |  {:6.4f}s".format(result[0], result[1], end - start))

    q1, q2, q3, q4 = [], [], [], []
    for x, y, _ in data:
        if x > 0 and y > 0:
            q1.append([x, y])
        if x > 0 and y < 0:
            q2.append([x, y])
        if x < 0 and y > 0:
            q4.append([x, y])
        if x < 0 and y < 0:
            q3.append([x, y])

    if args.plot:
        xall, yall, tall = zip(*data)
        fg, ax = plt.subplots(1, 1)
        if args.path:
            for p in paths:
                ax.plot(*zip(*p))
        ax.plot(xall, yall, 'bo')
        plt.show()

    _, _, t = zip(*data)
    return [len(q1), len(q2), len(q3), len(q4)], np.average(t)


if __name__ == "__main__":
    args = get_input()
    [q1, q2, q3, q4], t = eval_annealing(args)
    if args.qbox:
        print("\n{:4}|{:4}".format(q4, q1))
        print("____|____")
        print("    |    ")
        print("{:4}|{:4}\n".format(q3, q2))
    print("Correct: {:4.2f}%".format(q4 / sum([q1, q2, q3, q4]) * 100))
    print("Average time per run: {:4.2f}s".format(t))
