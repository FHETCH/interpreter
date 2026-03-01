from functools import cache
from math import e, pi
import math
import numpy as np

def bit_rev(vec: list):
    if len(vec) <= 2:
        return vec
    return bit_rev(vec[::2]) + bit_rev(vec[1::2])

@cache
def get_roots(N: int):
    rotGroup = [0] * N
    fivePows = 1

    for i in range(N // 2):
        rotGroup[i] = fivePows
        fivePows *= 5
        fivePows &= 2 * N - 1

    roots = [np.power(e, pi * i * 1j / N) for i in range(N * 2 + 1)]
    roots[N * 2] = roots[0]
    roots[N] = complex(-1, 0)
    roots[N//2] = complex(0, 1)
    roots[3*N//2] = complex(0, -1)
    return rotGroup, roots

def special_fft(coeffs, scale):
    n = len(coeffs) // 2

    values = np.array(bit_rev(coeffs), dtype=np.complex128) / scale
    values = values[::2] + values[1::2] * 1j

    rotGroup, roots = get_roots(n * 2)
    log_n = int(math.log(n, 2))
    log_m = int(math.log(n * 4, 2))

    for log_len in range(1, log_n + 1, 1):
        length = 1 << log_len
        len_h = length >> 1
        len_q = length << 2
        logGap = log_m - 2 - log_len
        mask = len_q - 1
        for i in range(0, n, length):
            j = 0
            k = i
            while j < len_h:
                values[k + len_h] *= roots[(rotGroup[j] & mask) << logGap]
                values[k], values[k + len_h] = (
                    values[k] + values[k + len_h],
                    values[k] - values[k + len_h],
                )
                j += 1
                k += 1

    return [complex(m) for m in values]


def special_ifft(message, scale):
    values = np.array(message, dtype=np.complex128)
    n = len(message)
    rotGroup, roots = get_roots(n * 2)
    log_n = int(math.log(n, 2))
    log_m = int(math.log(n * 4, 2))
    for log_len in range(log_n, 0, -1):
        length = 1 << log_len
        len_h = length >> 1
        len_q = length << 2
        logGap = log_m - 2 - log_len
        mask = len_q - 1
        for i in range(0, n, length):
            j = 0
            k = i
            while j < len_h:
                values[k], values[k + len_h] = (
                    values[k] + values[k + len_h],
                    (values[k] - values[k + len_h])
                    * roots[(len_q - (rotGroup[j] & mask)) << logGap],
                )
                j += 1
                k += 1
    values = bit_rev((values * scale / n).tolist())

    coeffs = [0] * (n * 2)
    for i in range(n):
        coeffs[i] = round(values[i].real)
        coeffs[i + n] = round(values[i].imag)

    return coeffs

