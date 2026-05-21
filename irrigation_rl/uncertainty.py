from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn


@dataclass
class UncertaintyConfig:
    enabled: bool = True
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    update_epochs: int = 1


class UncertaintyEstimator:
    """Online next-Dr predictor used for UC-PPO reward modification."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: UncertaintyConfig | None = None,
        seed: int = 0,
        device: str = "cpu",
    ):
        self.cfg = cfg or UncertaintyConfig()
        self.device = torch.device(device)
        torch.manual_seed(int(seed))

        input_dim = int(state_dim) + int(action_dim)
        hidden = int(self.cfg.hidden_dim)
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.cfg.learning_rate))
        self.loss_fn = nn.MSELoss()

    def _input_tensor(self, state, action) -> torch.Tensor:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        x = np.concatenate([s, a], axis=0)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

    def predict(self, state, action) -> float:
        if not self.cfg.enabled:
            return 0.0
        self.model.eval()
        with torch.no_grad():
            y = self.model(self._input_tensor(state, action))
        return float(y.reshape(-1)[0].detach().cpu().item())

    def uncertainty(self, state, action, next_dr: float, taw: float) -> Tuple[float, float]:
        pred = self.predict(state, action)
        u = abs(float(next_dr) - pred) / max(float(taw), 1e-8)
        return float(u), float(pred)

    def update(self, state, action, next_dr: float) -> float:
        if not self.cfg.enabled:
            return 0.0

        self.model.train()
        target = torch.as_tensor([[float(next_dr)]], dtype=torch.float32, device=self.device)
        loss_value = 0.0
        for _ in range(max(int(self.cfg.update_epochs), 1)):
            pred = self.model(self._input_tensor(state, action))
            loss = self.loss_fn(pred, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value = float(loss.detach().cpu().item())
        return loss_value

