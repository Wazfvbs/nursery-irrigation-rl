from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from irrigation_rl.rewards.reward import RewardFunction, RewardConfig
from irrigation_rl.rewards.target import DynamicTarget, TargetConfig
from irrigation_rl.exploration.ucb_bonus import ActionBinUCB, UCBConfig


@dataclass
class WrapperFlags:
    """用于消融实验的开关（对应 configs/train.yaml 的 ablation 字段）"""
    use_dynamic_target: bool = True
    use_reward_shaping: bool = True
    use_ucb_bonus: bool = True

    # ✅ 固定目标区间（用于 w/o_Target），用 TAW 的比例定义，确保与 dynamic 不等价
    fixed_lo_frac_TAW: float = 0.15
    fixed_hi_frac_TAW: float = 0.35



class RewardWrapper(gym.Wrapper):
    """
    RewardWrapper：把 reward / target / UCB 完整封装进 wrapper，
    让 env 保持“纯物理模型”，训练阶段 reward 不为 0。

    - env.step() 内部仍返回 reward=0（物理层）
    - wrapper.step() 计算 reward，并把 reward_terms 写入 info
    """

    def __init__(
            self,
            env: gym.Env,
            reward_cfg: Optional[RewardConfig] = None,
            target_cfg: Optional[TargetConfig] = None,
            ucb_cfg: Optional[UCBConfig] = None,
            flags: Optional[WrapperFlags] = None,
    ):
        super().__init__(env)

        self.flags = flags or WrapperFlags()

        # reward config（可在这里统一调参）
        self.reward_cfg = reward_cfg or RewardConfig()
        self.target_cfg = target_cfg or TargetConfig()
        self.ucb_cfg = ucb_cfg or UCBConfig(enabled=self.flags.use_ucb_bonus)

        # 如果不启用 reward shaping：我们保留核心目标（跟踪+节水），去掉 shaping 项
        if not self.flags.use_reward_shaping:
            self.reward_cfg.w_improve = 0.0
            self.reward_cfg.w_smooth = 0.0
            self.reward_cfg.w_safe = 0.0

        # 如果不启用 UCB：w_ucb 归零即可
        if not self.flags.use_ucb_bonus:
            self.reward_cfg.w_ucb = 0.0
            self.ucb_cfg.enabled = False

        self.reward_fn = RewardFunction(self.reward_cfg)
        self.target = DynamicTarget(self.target_cfg)

        # ucb 根据动作离散 bins 来计数
        a_max = float(getattr(self.env, "cfg", None).a_max_mm) if hasattr(self.env, "cfg") else 1.0
        self.ucb = ActionBinUCB(self.ucb_cfg, a_max=a_max)

        # internal time step for ucb bonus
        self.t = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reward_fn.reset()
        self.ucb.reset()
        self.t = 0

        # reset 时也把 target interval 写入 info（方便 debug/画图）
        Dr_lo, Dr_hi = self._get_target_interval(info)
        info["Dr_lo"] = float(Dr_lo)
        info["Dr_hi"] = float(Dr_hi)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # 取 env 的状态变量
        Dr = float(info.get("Dr_mm", 0.0))
        theta = float(info.get("theta", 0.0))

        # 灌溉动作（用 env 里已经 clip 后的 I_mm 最稳）
        if "I_mm" in info:
            I = float(info["I_mm"])
        else:
            I = float(np.asarray(action).reshape(-1)[0])

        Dr_lo, Dr_hi = self._get_target_interval(info)

        # UCB bonus
        b = self.ucb.bin_id(I)
        bonus = self.ucb.bonus(self.t, b)
        self.ucb.update(b)

        # 安全判定（论文里可写 as safety constraint / termination condition）
        theta_wp = float(getattr(getattr(self.env, "cfg", None), "theta_wp", 0.0))
        unsafe = theta < theta_wp

        # ==============================
        # ✅ Sparse reward for w/o_Shaping
        # ==============================
        if not self.flags.use_reward_shaping:
            # 越界判定（sparse）
            if Dr < Dr_lo:
                err = Dr_lo - Dr
            elif Dr > Dr_hi:
                err = Dr - Dr_hi
            else:
                err = 0.0

            violation = 1.0 if err > 0 else 0.0

            # 水量惩罚（防止疯狂浇水作弊）
            # 这里保持你的 w_water 仍有效
            reward = - self.reward_cfg.w_track * violation - self.reward_cfg.w_water * float(I)

            # 安全惩罚（保持 safety 仍能约束极端缺水）
            if unsafe:
                reward -= self.reward_cfg.w_safe

            terms = {
                "mode": "sparse",
                "err": float(err),
                "violation": float(violation),
                "I": float(I),
                "unsafe": float(unsafe),
                "ucb_bonus": float(bonus),
            }

        else:
            # ✅ Full：dense reward（连续误差）
            reward, terms = self.reward_fn.compute(
                Dr=Dr,
                Dr_lo=Dr_lo,
                Dr_hi=Dr_hi,
                I=I,
                theta=theta,
                theta_wp=theta_wp,
                ucb_bonus=bonus,
                unsafe=unsafe,
            )

        # 把关键内容写进 info，方便 trajectory.csv / fig / table 直接生成
        info["Dr_lo"] = float(Dr_lo)
        info["Dr_hi"] = float(Dr_hi)
        info["ucb_bonus"] = float(bonus)
        info["reward_terms"] = terms

        self.t += 1
        return obs, float(reward), terminated, truncated, info

    def _get_target_interval(self, *args, **kwargs) -> Tuple[float, float]:
        TAW = float(getattr(self.env, "TAW", 1.0))
        RAW = float(getattr(self.env, "RAW", 1.0))

        if self.flags.use_dynamic_target:
            return self.target.get_interval(TAW=TAW, RAW=RAW)

        low = self.flags.fixed_lo_frac_TAW * TAW
        high = self.flags.fixed_hi_frac_TAW * TAW
        if high < low:
            high = low
        return float(low), float(high)


