# irrigation_rl/baselines/calendar.py
from __future__ import annotations

class CalendarPolicy:
    """
    Fixed-interval (calendar) baseline:
      - irrigate a fixed amount every k days, independent of Dr.
    """
    def __init__(self, interval_days: int, irrigation_mm: float, offset_days: int = 0, a_max_mm: float | None = None):
        assert interval_days >= 1
        self.k = int(interval_days)
        self.irrigation_mm = float(irrigation_mm)
        self.offset = int(offset_days)
        self.a_max = None if a_max_mm is None else float(a_max_mm)

    def act(self, day: int, Dr: float | None = None) -> float:
        # day 从 0 开始计
        do_water = ((day + self.offset) % self.k) == 0
        if not do_water:
            return 0.0
        I = self.irrigation_mm
        if self.a_max is not None:
            I = min(max(I, 0.0), self.a_max)
        return float(I)
