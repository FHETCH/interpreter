from dataclasses import dataclass


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
