from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass
class WaterBalanceInputs:
    P: float = 0.0   # precipitation (mm/day)
    RO: float = 0.0  # runoff (mm/day)
    CR: float = 0.0  # capillary rise (mm/day)
    DP: float = 0.0  # deep percolation (mm/day)

def update_Dr(Dr_prev: float, ETc: float, I: float, wb: WaterBalanceInputs, TAW: float) -> float:
    """
    FAO-56 root-zone depletion update (mm):
      Dr_{t+1} = Dr_t - (P-RO) - I - CR + ETc + DP
    Clamp to [0, TAW].
    """
    Dr_next = Dr_prev - (wb.P - wb.RO) - I - wb.CR + ETc + wb.DP
    Dr_next = max(min(Dr_next, TAW), 0.0)
    return Dr_next
