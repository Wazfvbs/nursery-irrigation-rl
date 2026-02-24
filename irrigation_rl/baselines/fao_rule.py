from __future__ import annotations

class FAORulePolicy:
    """
    FAO-inspired rule:
    keep Dr <= RAW by irrigating when depletion exceeds RAW.
    """
    def __init__(self, RAW: float, irrigation_mm: float):
        self.RAW = RAW
        self.irrigation_mm = irrigation_mm

    def act(self, Dr: float) -> float:
        return self.irrigation_mm if Dr > self.RAW else 0.0
