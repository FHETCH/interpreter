import numpy as np
from client.crypto import Parameters
from client.keygen import gen_noise
from client.utils import random_poly
from fhetch.data import MRP, Vector
from client.encode import special_ifft, special_fft


class CryptoContext:
    def __init__(self, params: Parameters, sk: Vector) -> None:
        self._params = params
        self._sk: Vector = sk

    def encrypt_msg(self, msg: np.array) -> tuple[MRP, MRP]:
        pt = encode(msg, self._params.scaling_factor(), self._params.q)
        return self.encrypt(pt)

    def encrypt(self, pt: MRP) -> tuple[MRP, MRP]:
        degree = pt.degree()
        if degree != len(self._sk.value):
            raise ValueError(f"Degree mismatch: plaintext degree {degree} != secret key length {len(self._sk.value)}")
        pt_base = pt.values.keys()
        # Create MRP sk from a vector based on the pt base
        sk = MRP.from_coeffs(base=pt_base, coeffs=self._sk.value)
        a = random_poly(pt_base, degree)
        b = pt - a * sk + gen_noise(pt_base, degree)
        return (b, a)

    def decrypt_msg(self, cipher: tuple[MRP, MRP]) -> np.array:
        b, a = cipher
        sk = MRP.from_coeffs(a.values.keys(), self._sk.value)
        return decode(b + a * sk, self._params.scaling_factor())


def encode(msg: list[int], scale, base: list[int]) -> MRP:
    coeffs = special_ifft(msg, scale)
    poly = MRP.from_coeffs(base, coeffs)
    return poly


def decode(pt: MRP, scale: int) -> np.array:
    p = pt.reconstruct(exact=True).value
    return special_fft(list(p), scale)
