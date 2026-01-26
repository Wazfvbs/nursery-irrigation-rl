from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass
class UCBConfig:
    enabled: bool = True
    bins: int = 16
    c: float = 1.0  # UCB coefficient

class ActionBinUCB:
    def __init__(self, cfg: UCBConfig, a_max: float):
        self.cfg = cfg
        self.a_max = max(a_max, 1e-8)
        self.counts = [0] * max(cfg.bins, 1)

    def reset(self):
        self.counts = [0] * len(self.counts)

    def bin_id(self, I: float) -> int:
        b = int((max(I, 0.0) / self.a_max) * len(self.counts))
        return max(0, min(b, len(self.counts) - 1))

    def update(self, b: int):
        self.counts[b] += 1

    def bonus(self, t: int, b: int) -> float:
        if not self.cfg.enabled:
            return 0.0
        n = self.counts[b]
        return float(self.cfg.c * math.sqrt(math.log(t + 1.0) / (n + 1.0)))
