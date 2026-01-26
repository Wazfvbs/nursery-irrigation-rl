from __future__ import annotations
from typing import List, Callable

def choose_worst_delta(candidates: List[float], score_fn: Callable[[float], float]) -> float:
    """
    Minimal adversarial selection:
    pick delta in candidates that maximizes score_fn(delta).
    """
    best = candidates[0]
    best_score = score_fn(best)
    for d in candidates[1:]:
        s = score_fn(d)
        if s > best_score:
            best_score = s
            best = d
    return float(best)
