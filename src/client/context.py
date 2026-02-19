from dataclasses import dataclass
from math import e, pi, prod
import numpy as np
from client.crypto import Ciphertext, Parameters, Plaintext
from client.keygen import gen_noise, gen_sk
from client.utils import random_poly
from fhetch.data import MRP, Vector
from client.encode import embedding, unpacking


class CryptoContext:
    def __init__(self, params: Parameters,sk:Vector) -> None:
        self._params = params
        self._sk:Vector = sk

    def encrypt_msg(self, msg: np.array, scale) -> Ciphertext:
        pt = encode(msg,scale,self._params.moduli)
        return self.encrypt(pt)

    def encrypt(self, pt: Plaintext) -> Ciphertext:
        degree = pt.poly.degree()
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



def encode(msg:list[int], scale,base:list[int]) -> Plaintext:
    coeffs = embedding(msg,scale)
    poly = MRP.from_coeffs(base,coeffs)
    return Plaintext(scale,poly)


def decode(pt: Plaintext) -> np.array:
    p = pt.poly.reconstruct(exact=True).value
    return unpacking(list(p),pt.scale)


