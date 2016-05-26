#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from simulatedAnnealing import SimAnneal

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
    return parser.parse_args()

if __name__ == "__main__":
    args = get_input()

    sim = SimAnneal(args.startx, args.starty, args.rnd,
                    args.t1, args.t2, args.speed, args.step)
    sim.run(args.runs, args.silent, dbg=args.debug)
    sim.evaluate()
    if args.plot:
        sim.plot(args.path)
