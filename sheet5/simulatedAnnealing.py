#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt


class SimAnneal:
    '''
    Code fragment for exercise 6.2 of the Computerpraktikum Datenanlyse 2014
    Authors: Ralf Ulrich, Frank Schroeder (Karlsruhe Institute of Technology)
    This code fragment probably is not the best and fastest implementation
    for "simulated annealing", but it is a simple implementation which does its job.
    '''

    def __init__(self, initialXvalue=0, initialYvalue=0, rndArgs=0,
                 initialTemperature=100, finalTemperature=1,
                 coolingSpeed=1, stepSize=1,
                 rosenParams=None):
        if rndArgs:
            initialXvalue = np.random.uniform(-4, 4)
            initialYvalue = np.random.uniform(-4, 4)
        self.initialXvalue = initialXvalue
        self.initialYvalue = initialYvalue

        if rosenParams is None:
            rosenParams = [11., 7., 0.1]
        self.rosenParams = rosenParams

        self.rndArgs = rndArgs
        self.initialTemperature = initialTemperature
        self.finalTemperature = finalTemperature
        self.coolingSpeed = coolingSpeed
        self.stepSize = stepSize

        self.runData = []


    def run(self, runs=1, silent=1, clearLast=1, dbg=0):
        if clearLast:
            del self.runData[:]
        for _ in range(runs):
            start = time.time()
            result, path = self.anneal(dbg)
            end = time.time()
            self.runData.append([result[0], result[1], end - start, path])
            if not silent:
                print("best (x, y): {:11.8f}  {:11.8f}  |  {:6.4f}s".format(result[0], result[1], end - start))


    def evaluate(self):
        q1, q2, q3, q4 = [], [], [], []
        x, y, t, _ = zip(*self.runData)
        for xi, yi in zip(x, y):
            if xi > 0 and yi > 0:
                q1.append([xi, yi])
            if xi > 0 and yi < 0:
                q2.append([xi, yi])
            if xi < 0 and yi > 0:
                q4.append([xi, yi])
            if xi < 0 and yi < 0:
                q3.append([xi, yi])

        q1_l = len(q1)
        q2_l = len(q2)
        q3_l = len(q3)
        q4_l = len(q4)

        print("\n{:4}|{:4}".format(q4_l, q1_l))
        print("____|____")
        print("    |    ")
        print("{:4}|{:4}\n".format(q3_l, q2_l))
        print("Correct: {:4.2f}%".format(q4_l / sum([q1_l, q2_l, q3_l, q4_l]) * 100))
        print("Average time per run: {:6.4f}s".format(np.average(t)))

        if q4_l > 0:
            q4_x, q4_y = zip(*q4)
            print("Average Global Minimum: ({:6.5f}, {:6.5f})".format(np.average(q4_x), np.average(q4_y)))
            print("Standard Deviation: ({:6.5f}, {:6.5f})".format(np.std(q4_x), np.std(q4_y)))


    def plot(self, path=0):
        x, y, _, paths = zip(*self.runData)
        fg, ax = plt.subplots(1, 1)
        if path:
            for p in paths:
                ax.plot(*zip(*p))
        ax.scatter(x, y)
        plt.show()


    def rosenbrock(self, xy, abc):
        '''
        The modified Rosenbrock function as described in exercise 6
        '''
        x, y = xy
        a, b, c = abc
        return (x**2 + y - a)**2 + (x + y**2 - b)**2 + c * (x + y)**2

    def anneal(self, dbg):
        nParameter = 2  # 2 parameters: x and y

        # Current parameters and cost
        currentParameters = [self.initialXvalue, self.initialYvalue]
        currentFunctionValue = self.rosenbrock(currentParameters, self.rosenParams)

        # keep reference of best parameters
        bestParameters = currentParameters
        bestFunctionValue = currentFunctionValue

        # Heat the system
        temperature = self.initialTemperature

        iteration = 0
        plog = []  # parameter log for debug output
        listOfPoints = [[self.initialXvalue, self.initialYvalue]]

        # Start to slowly cool the system
        while temperature > self.finalTemperature:

            # Change parameters
            newParameters = [0]*nParameter

            for ipar in range(nParameter):
                newParameters[ipar] = np.random.normal(currentParameters[ipar], self.stepSize)

            # Get the new value of the function
            newFunctionValue = self.rosenbrock(newParameters, self.rosenParams)

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

            if dbg:
                plog.append([temperature, currentParameters, currentFunctionValue, deltaFunctionValue])

            # Cool the system
            temperature *= 1 - self.coolingSpeed/100.
            iteration += 1

        if dbg:
            print("{:9}  {:8}  {:8}  {:9}  {:10}".format("temp", "x", "y", "value", "delta"))
            for row in plog:
                txt = "{:9.5f}  {:8.5f}  {:8.5f}  {:9.5f}  {:10.5f}"
                print(txt.format(row[0], row[1][0], row[1][1], row[2], row[3]))

        return bestParameters, listOfPoints


if __name__ == "__main__":
    sim = SimAnneal()
    sim.run(dbg=1)
