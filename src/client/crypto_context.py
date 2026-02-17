from dataclasses import dataclass
from math import e, pi, prod
import numpy as np
from client.keygen import gen_noise, gen_sk
from fhetch.data import MRP
from client.encode import embedding, unpacking


@dataclass
class Plaintext:
    scale: int
    poly: MRP

@dataclass
class Ciphertext:
    scale: int
    polynomials: list[MRP]


@dataclass
class Parameters:
    q: list[int]
    p: list[int]
    log_n: int
    log_slots: int
    h: int = 32

    @property
    def moduli(self):
        return self.q + self.p

    @property
    def degree(self):
        return 1 << self.log_n

    @property
    def slots(self):
        return 1 << self.log_slots

    def Q(self, level):
        return prod(self.q[:level])

    def P(self, k):
        assert k > 0
        result = prod(self.p[-k:])
        if k > len(self.p):
            extras = k - len(self.p)
            result *= prod(self.q[-extras:])
        return result


class CryptoContext:
    def __init__(self, params: Parameters) -> None:
        self._params = params
        self._sk = gen_sk(params.h, params.degree)

    def encrypt_msg(self, msg: np.array, scale) -> Ciphertext:
        pt = encode(msg,scale,self._params.moduli)
        return self.encrypt(pt)

    def encrypt(self, pt: Plaintext) -> Ciphertext:
        degree = len(next(iter(pt.poly.values.values())).value)
        pt_base = pt.poly.values.keys()
        # Create MRP sk from a vector based on the pt base
        sk = MRP.from_coeffs(base = pt_base, coeffs=self._sk.value)
        a = random_poly(pt_base, degree)
        b = pt.poly - a * sk + gen_noise(pt_base, degree)
        return Ciphertext(pt.scale, [b, a])

    def decrypt_msg(self, cipher: Ciphertext) -> np.array:
        b, a = cipher.polynomials
        sk = MRP.from_coeffs(a.values.keys(), self._sk.value)
        return decode(Plaintext(cipher.scale, b + a * sk))


def random_poly(base: list[int], degree: int) -> MRP:
    coeffs = []
    rng = np.random.default_rng()
    for _ in range(degree):
        coeffs.append(rng.integers(-(2**31), 2**31, dtype=np.int32))

    return MRP.from_coeffs(base, coeffs)


def encode(msg:list[int], scale,base:list[int]) -> Plaintext:
    coeffs = embedding(msg,scale)
    poly = MRP.from_coeffs(base,coeffs)
    return Plaintext(scale,poly)


def decode(pt: Plaintext) -> np.array:
    p = pt.poly.reconstruct(exact=True).value
    return unpacking(list(p),pt.scale)
