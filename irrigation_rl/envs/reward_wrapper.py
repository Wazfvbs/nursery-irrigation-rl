from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from irrigation_rl.exploration.ucb_bonus import ActionBinUCB, UCBConfig
from irrigation_rl.rewards.reward import RewardConfig, RewardFunction, distance_to_interval
from irrigation_rl.rewards.target import DynamicTarget, TargetConfig
from irrigation_rl.uncertainty import UncertaintyConfig, UncertaintyEstimator


@dataclass
class WrapperFlags:
    """Ablation switches wired from configs/train.yaml."""

    use_dynamic_target: bool = True
    use_reward_shaping: bool = True
    use_ucb_bonus: bool = True
    use_uncertainty_constraint: bool = True

    fixed_lo_frac_TAW: float = 0.15
    fixed_hi_frac_TAW: float = 0.35


class RewardWrapper(gym.Wrapper):
    """
    Adds UC-PPO reward terms to the pure FAO-56 environment.

    Base env reward remains zero; this wrapper computes the training signal:
    multi-objective reward + UCB bonus - uncertainty constraint.
    """

    def __init__(
        self,
        env: gym.Env,
        reward_cfg: Optional[RewardConfig] = None,
        target_cfg: Optional[TargetConfig] = None,
        ucb_cfg: Optional[UCBConfig] = None,
        uncertainty_cfg: Optional[UncertaintyConfig] = None,
        flags: Optional[WrapperFlags] = None,
        seed: int = 0,
    ):
        super().__init__(env)

        self.flags = flags or WrapperFlags()
        self.reward_cfg = reward_cfg or RewardConfig()
        self.target_cfg = target_cfg or TargetConfig()
        self.ucb_cfg = ucb_cfg or UCBConfig(enabled=self.flags.use_ucb_bonus)
        self.uncertainty_cfg = uncertainty_cfg or UncertaintyConfig(enabled=True)

        if not self.flags.use_ucb_bonus:
            self.reward_cfg.w_ucb = 0.0
            self.ucb_cfg.enabled = False
        if not self.flags.use_uncertainty_constraint:
            self.reward_cfg.w_uncertainty = 0.0

        self.reward_fn = RewardFunction(self.reward_cfg)
        self.target = DynamicTarget(self.target_cfg)

        a_max = float(getattr(getattr(self.env, "cfg", None), "a_max_mm", 1.0))
        self.ucb = ActionBinUCB(self.ucb_cfg, a_max=a_max)

        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        self.uncertainty = UncertaintyEstimator(
            state_dim=obs_dim,
            action_dim=action_dim,
            cfg=self.uncertainty_cfg,
            seed=seed,
        )

        self.t = 0
        self._last_obs: Optional[np.ndarray] = None
        self._last_info: Dict[str, Any] = {}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reward_fn.reset()
        self.t = 0
        self._last_obs = np.asarray(obs, dtype=np.float32).copy()
        self._last_info = dict(info)

        Dr_lo_train, Dr_hi_train = self._get_train_interval(info, obs)
        Dr_lo_ref, Dr_hi_ref = self._get_ref_interval(info, obs)
        self._write_interval_info(info, Dr_lo_train, Dr_hi_train, Dr_lo_ref, Dr_hi_ref)
        return obs, info

    def step(self, action):
        state_before = None if self._last_obs is None else self._last_obs.copy()
        info_before = dict(self._last_info)

        obs_next, _, terminated, truncated, info = self.env.step(action)
        obs_next_arr = np.asarray(obs_next, dtype=np.float32).copy()

        Dr = float(info.get("Dr_mm", 0.0))
        Dr_prev = float(info.get("Dr_prev_mm", info_before.get("Dr_mm", Dr)))
        theta = float(info.get("theta", 0.0))
        TAW = float(info.get("TAW_mm", getattr(self.env, "TAW", 1.0)))
        RAW = float(info.get("RAW_mm", getattr(self.env, "RAW", 1.0)))
        DP = float(info.get("DP", 0.0))
        P = float(info.get("P", 0.0))

        if "I_mm" in info:
            I = float(info["I_mm"])
        else:
            I = float(np.asarray(action).reshape(-1)[0])
        I_prev = float(state_before[3]) if state_before is not None and state_before.size >= 4 else 0.0
        Imax = float(getattr(getattr(self.env, "cfg", None), "a_max_mm", max(abs(I), 1.0)))

        # UC-PPO uses the target generated from s_t.
        target_info = info_before if info_before else info
        target_obs = state_before if state_before is not None else obs_next_arr
        Dr_lo_train, Dr_hi_train = self._get_train_interval(target_info, target_obs)
        Dr_lo_ref, Dr_hi_ref = self._get_ref_interval(target_info, target_obs)
        Dr_mid_ref = 0.5 * (Dr_lo_ref + Dr_hi_ref)

        b = self.ucb.bin_id(I)
        bonus = self.ucb.bonus(self.t, b)
        self.ucb.update(b)

        uncertainty = 0.0
        pred_next_dr = 0.0
        predictor_loss = 0.0
        if state_before is not None and self.uncertainty_cfg.enabled:
            action_arr = np.array([I], dtype=np.float32)
            uncertainty, pred_next_dr = self.uncertainty.uncertainty(state_before, action_arr, Dr, TAW)
            predictor_loss = self.uncertainty.update(state_before, action_arr, Dr)
        c_uncertain = float(uncertainty) * (abs(I) / max(Imax, 1e-8))

        theta_wp = float(getattr(getattr(self.env, "cfg", None), "theta_wp", 0.0))
        unsafe = theta < theta_wp

        if not self.flags.use_reward_shaping:
            err = distance_to_interval(Dr, Dr_lo_train, Dr_hi_train)
            violation = 1.0 if err > 0.0 else 0.0
            r_track = -self.reward_cfg.w_track * violation
            r_water = -self.reward_cfg.w_water * (I / max(Imax, 1e-8))
            r_safe = -self.reward_cfg.w_safe if unsafe else 0.0
            r_ucb = self.reward_cfg.w_ucb * bonus
            r_uncertainty = -self.reward_cfg.w_uncertainty * c_uncertain
            reward = r_track + r_water + r_safe + r_ucb + r_uncertainty
            terms = {
                "mode": "sparse",
                "e_target": float(err / max(TAW, 1e-8)),
                "e_target_mm": float(err),
                "water_use": float(I / max(Imax, 1e-8)),
                "stress": float(max(0.0, Dr - RAW) / max(TAW, 1e-8)),
                "over_irrigation": float(DP / max(TAW, 1e-8)),
                "smoothness": float(abs(I - I_prev) / max(Imax, 1e-8)),
                "safety_violation": float(violation),
                "ucb_bonus": float(bonus),
                "uncertainty": float(uncertainty),
                "c_uncertain": float(c_uncertain),
                "r_track": float(r_track),
                "r_water": float(r_water),
                "r_safe": float(r_safe),
                "r_ucb": float(r_ucb),
                "r_uncertainty": float(r_uncertainty),
                "reward": float(reward),
            }
        else:
            reward, terms = self.reward_fn.compute(
                Dr=Dr,
                Dr_prev=Dr_prev,
                Dr_lo=Dr_lo_train,
                Dr_hi=Dr_hi_train,
                I=I,
                I_prev=I_prev,
                Imax=Imax,
                TAW=TAW,
                RAW=RAW,
                DP=DP,
                P=P,
                ucb_bonus=bonus,
                uncertainty=uncertainty,
                c_uncertain=c_uncertain,
                unsafe=unsafe,
            )

        terms["pred_next_Dr"] = float(pred_next_dr)
        terms["uncertainty_loss"] = float(predictor_loss)

        self._write_interval_info(info, Dr_lo_train, Dr_hi_train, Dr_lo_ref, Dr_hi_ref)
        info["Dr_mid_ref"] = float(Dr_mid_ref)
        info["ucb_bonus"] = float(bonus)
        info["ucb_bin"] = int(b)
        info["ucb_count"] = int(self.ucb.counts[b])
        info["uncertainty"] = float(uncertainty)
        info["c_uncertain"] = float(c_uncertain)
        info["pred_next_Dr"] = float(pred_next_dr)
        info["reward_terms"] = terms

        self.t += 1
        self._last_obs = obs_next_arr
        self._last_info = dict(info)
        return obs_next, float(reward), terminated, truncated, info

    def _write_interval_info(
        self,
        info: Dict[str, Any],
        Dr_lo_train: float,
        Dr_hi_train: float,
        Dr_lo_ref: float,
        Dr_hi_ref: float,
    ) -> None:
        info["Dr_lo"] = float(Dr_lo_train)
        info["Dr_hi"] = float(Dr_hi_train)
        info["Dr_lo_train"] = float(Dr_lo_train)
        info["Dr_hi_train"] = float(Dr_hi_train)
        info["Dr_lo_ref"] = float(Dr_lo_ref)
        info["Dr_hi_ref"] = float(Dr_hi_ref)
        info["Dr_mid_ref"] = float(0.5 * (Dr_lo_ref + Dr_hi_ref))

    def _get_train_interval(self, info: Dict[str, Any], obs=None) -> Tuple[float, float]:
        TAW = float(info.get("TAW_mm", getattr(self.env, "TAW", 1.0)))
        if self.flags.use_dynamic_target:
            return self.target.get_interval(
                stage_norm=self._stage_norm(info, obs),
                et0=self._et0(info, obs),
                taw=TAW,
            )

        low = self.flags.fixed_lo_frac_TAW * TAW
        high = self.flags.fixed_hi_frac_TAW * TAW
        if high < low:
            high = low
        return float(low), float(high)

    def _get_ref_interval(self, info: Dict[str, Any], obs=None) -> Tuple[float, float]:
        TAW = float(info.get("TAW_mm", getattr(self.env, "TAW", 1.0)))
        return self.target.get_interval(
            stage_norm=self._stage_norm(info, obs),
            et0=self._et0(info, obs),
            taw=TAW,
        )

    @staticmethod
    def _stage_norm(info: Dict[str, Any], obs=None) -> float:
        if obs is not None:
            arr = np.asarray(obs).reshape(-1)
            if arr.size >= 3:
                return float(arr[2])
        return float(info.get("stage_norm", 0.5))

    @staticmethod
    def _et0(info: Dict[str, Any], obs=None) -> float:
        if obs is not None:
            arr = np.asarray(obs).reshape(-1)
            if arr.size >= 2:
                return float(arr[1])
        return float(info.get("ET0", info.get("ET0_obs", 0.0)))

