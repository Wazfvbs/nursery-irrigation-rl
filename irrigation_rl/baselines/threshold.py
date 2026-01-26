from __future__ import annotations

class ThresholdPolicy:
    """
    Simple fixed-threshold baseline in Dr-space.
    If Dr > threshold -> irrigate fixed amount, else 0.
    """
    def __init__(self, Dr_threshold: float, irrigation_mm: float):
        self.Dr_threshold = Dr_threshold
        self.irrigation_mm = irrigation_mm

    def act(self, Dr: float) -> float:
        return self.irrigation_mm if Dr > self.Dr_threshold else 0.0
