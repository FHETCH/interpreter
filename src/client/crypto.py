from dataclasses import dataclass
from math import prod


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
    def scaling_factor(self):
        return 2.0 ** self.q[0].bit_length()