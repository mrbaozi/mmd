#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from frequentist import limit


def no_background():
    v_tB = 0
    n_0 = 3
    print(limit(n_0, v_tB, 0.1))

def background():
    v_tB = 0
    n_0 = 3
    print(limit(n_0, v_tB, 0.1))

if __name__ == "__main__":
    no_background()
