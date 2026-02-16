# Key generation functions
import __future__ 
from math import prod
from random import randint, shuffle, randbytes

import numpy as np

from fhetch.data import MRP, Vector


def gen_noise(moduli: list[int], degree: int, sigma=3.2):
    coeffs = []
    rng = np.random.default_rng()
    while len(coeffs) < degree:
        coeffInt = round(rng.normal(loc=0, scale=1, size=1)[0] * sigma)
        if abs(coeffInt) <= 6 * sigma:
            coeffs.append(coeffInt)
    return MRP.from_coeffs(moduli, coeffs)


def gen_sk(hw: int, degree: int)->Vector:
    if hw > degree:
        hw = degree // 2
    num_pos = randint(0, hw)
    num_neg = hw - num_pos
    coeffs = ([1] * num_pos) + ([-1] * num_neg) + ([0] * (degree - hw))
    shuffle(coeffs)
    return Vector(np.array(coeffs))





