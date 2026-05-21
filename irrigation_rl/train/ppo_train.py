from __future__ import annotations

import os
from typing import Any, Dict

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from irrigation_rl.envs.nursery_env import EnvConfig, NurseryIrrigationEnv
from irrigation_rl.envs.fao56 import calc_ET0_PM, calc_ET0_fallback
from irrigation_rl.envs.reward_wrapper import RewardWrapper, WrapperFlags
from irrigation_rl.envs.weather import (
    AssumptionWeatherConfig,
    AssumptionWeatherProvider,
    ExternalCSVWeatherProvider,
)
from irrigation_rl.rewards.reward import RewardConfig
from irrigation_rl.rewards.target import TargetConfig
from irrigation_rl.exploration.ucb_bonus import UCBConfig
from irrigation_rl.robust.domain_randomization_wrapper import DomainRandomizationWrapper
from irrigation_rl.robust.obs_noise_wrapper import ObsNoiseConfig, ObsNoiseWrapper
from irrigation_rl.robust.randomization import RandomizationConfig
from irrigation_rl.uncertainty import UncertaintyConfig


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_randomization_cfg(train_cfg: Dict[str, Any]) -> RandomizationConfig:
    path = str(train_cfg.get("robust_train_cfg_path", "configs/noise_train.yaml"))
    raw: Dict[str, Any] = {}
    if path and os.path.exists(path):
        try:
            raw = load_yaml(path) or {}
        except Exception:
            raw = {}

    weather = raw.get("weather_bias", {}) if isinstance(raw, dict) else {}
    sensor = raw.get("sensor_noise", {}) if isinstance(raw, dict) else {}
    params = raw.get("param_noise", {}) if isinstance(raw, dict) else {}

    return RandomizationConfig(
        enabled=bool(raw.get("enabled", True)),
        ET0_mult_min=float(weather.get("ET0_mult_min", 0.95)),
        ET0_mult_max=float(weather.get("ET0_mult_max", 1.05)),
        Dr_sigma_mm=float(sensor.get("Dr_sigma_mm", raw.get("Dr_sigma_mm", 0.0))),
        theta_sigma=float(sensor.get("theta_sigma", 0.005)),
        ET0_sigma=float(sensor.get("ET0_sigma", raw.get("ET0_sigma", 0.0))),
        Kc_mult_min=float(params.get("Kc_mult_min", 0.95)),
        Kc_mult_max=float(params.get("Kc_mult_max", 1.05)),
        Zr_mult_min=float(params.get("Zr_mult_min", 0.95)),
        Zr_mult_max=float(params.get("Zr_mult_max", 1.05)),
    )


def _estimate_et0_mean(weather, horizon_days: int) -> float:
    vals = []
    for day in range(max(int(horizon_days), 1)):
        try:
            w = weather.get_day(day)
            try:
                et0 = float(calc_ET0_PM(w))
            except Exception:
                et0 = float(calc_ET0_fallback(w))
            if et0 >= 0.0:
                vals.append(et0)
        except Exception:
            continue
    return float(sum(vals) / len(vals)) if vals else 0.0


def build_env(env_cfg: dict, train_cfg: dict, seed: int = 0):
    cfg = EnvConfig(
        horizon_days=env_cfg["scenario"]["horizon_days"],
        a_max_mm=env_cfg["scenario"]["a_max_mm"],
        dt_days=env_cfg["scenario"]["dt_days"],
        theta_fc=env_cfg["soil"]["theta_fc"],
        theta_wp=env_cfg["soil"]["theta_wp"],
        Zr_m=env_cfg["soil"]["Zr_m"],
        p=env_cfg["soil"]["p"],
        Kc_ini=env_cfg["crop"]["Kc_ini"],
        Kc_mid=env_cfg["crop"]["Kc_mid"],
        Kc_end=env_cfg["crop"]["Kc_end"],
        stage_ini_days=env_cfg["crop"]["stage_ini_days"],
        stage_mid_days=env_cfg["crop"]["stage_mid_days"],
        stage_end_days=env_cfg["crop"]["stage_end_days"],
        terminate_on_theta_below_wp=env_cfg["termination"]["terminate_on_theta_below_wp"],
        terminate_on_Dr_above_TAW=env_cfg["termination"]["terminate_on_Dr_above_TAW"],
    )

    mode = env_cfg["weather"]["mode"]
    if mode == "external" and env_cfg["weather"].get("csv_path"):
        weather = ExternalCSVWeatherProvider(env_cfg["weather"]["csv_path"])
    else:
        weather = AssumptionWeatherProvider(
            AssumptionWeatherConfig(
                T_mean_C=env_cfg["weather"].get("T_mean_C", 20.0),
                RH_pct=env_cfg["weather"].get("RH_pct", 60.0),
                u2_mps=env_cfg["weather"].get("u2_mps", 1.0),
                Rs_MJ_m2_day=env_cfg["weather"].get("Rs_MJ_m2_day", 15.0),
                noise_sigma=0.0,
            )
        )

    env = NurseryIrrigationEnv(cfg=cfg, weather=weather, seed=seed)
    ab = train_cfg.get("ablation", {}) if isinstance(train_cfg, dict) else {}

    flags = WrapperFlags(
        use_dynamic_target=bool(ab.get("use_dynamic_target", True)),
        use_reward_shaping=bool(ab.get("use_reward_shaping", True)),
        use_ucb_bonus=bool(ab.get("use_ucb_bonus", True)),
        use_uncertainty_constraint=bool(ab.get("use_uncertainty_constraint", True)),
        fixed_lo_frac_TAW=float(ab.get("fixed_lo_frac_TAW", 0.15)),
        fixed_hi_frac_TAW=float(ab.get("fixed_hi_frac_TAW", 0.35)),
    )

    reward_cfg = RewardConfig()
    reward_block = train_cfg.get("reward", {})
    if isinstance(reward_block, dict):
        reward_cfg.w_track = float(reward_block.get("w_track", reward_cfg.w_track))
        reward_cfg.w_water = float(reward_block.get("w_water", reward_cfg.w_water))
        reward_cfg.w_stress = float(reward_block.get("w_stress", reward_cfg.w_stress))
        reward_cfg.w_over = float(reward_block.get("w_over", reward_cfg.w_over))
        reward_cfg.w_smooth = float(reward_block.get("w_smooth", reward_cfg.w_smooth))
        reward_cfg.w_safe = float(reward_block.get("w_safe", reward_cfg.w_safe))
        reward_cfg.w_ucb = float(reward_block.get("w_ucb", reward_cfg.w_ucb))
        reward_cfg.w_uncertainty = float(
            reward_block.get("w_uncertainty", reward_block.get("eta", reward_cfg.w_uncertainty))
        )
        reward_cfg.dp_max = float(reward_block.get("dp_max", reward_cfg.dp_max))

    target_cfg = TargetConfig()
    target_block = train_cfg.get("target", {})
    if isinstance(target_block, dict):
        for key in (
            "fixed_low_frac_TAW",
            "fixed_high_frac_TAW",
            "early_low_frac_TAW",
            "early_high_frac_TAW",
            "mid_low_frac_TAW",
            "mid_high_frac_TAW",
            "late_low_frac_TAW",
            "late_high_frac_TAW",
            "lambda_et",
            "min_width",
        ):
            if key in target_block:
                setattr(target_cfg, key, float(target_block[key]))
        if "et0_mean" in target_block and target_block["et0_mean"] is not None:
            target_cfg.et0_mean = float(target_block["et0_mean"])
        else:
            target_cfg.et0_mean = _estimate_et0_mean(weather, cfg.horizon_days)
    else:
        target_cfg.et0_mean = _estimate_et0_mean(weather, cfg.horizon_days)

    ucb_cfg = UCBConfig(enabled=flags.use_ucb_bonus)
    ucb_block = train_cfg.get("ucb", {})
    if isinstance(ucb_block, dict):
        ucb_cfg.enabled = bool(ucb_block.get("enabled", ucb_cfg.enabled)) and flags.use_ucb_bonus
        ucb_cfg.bins = int(ucb_block.get("bins", ucb_cfg.bins))
        ucb_cfg.c = float(ucb_block.get("c", ucb_cfg.c))

    uncertainty_cfg = UncertaintyConfig(enabled=True)
    uncertainty_block = train_cfg.get("uncertainty", {})
    if isinstance(uncertainty_block, dict):
        uncertainty_cfg.enabled = bool(uncertainty_block.get("enabled", uncertainty_cfg.enabled))
        uncertainty_cfg.learning_rate = float(
            uncertainty_block.get("learning_rate", uncertainty_cfg.learning_rate)
        )
        uncertainty_cfg.hidden_dim = int(uncertainty_block.get("hidden_dim", uncertainty_cfg.hidden_dim))
        uncertainty_cfg.update_epochs = int(
            uncertainty_block.get("update_epochs", uncertainty_cfg.update_epochs)
        )

    env = RewardWrapper(
        env,
        reward_cfg=reward_cfg,
        target_cfg=target_cfg,
        ucb_cfg=ucb_cfg,
        uncertainty_cfg=uncertainty_cfg,
        flags=flags,
        seed=seed,
    )

    # Train-time domain randomization: enabled for Full, disabled for Vanilla.
    if bool(ab.get("use_robust_training", False)):
        dr_cfg = _load_randomization_cfg(train_cfg)
        if dr_cfg.enabled:
            env = DomainRandomizationWrapper(env, cfg=dr_cfg, seed=seed)
            if float(dr_cfg.theta_sigma) > 0.0:
                obs_cfg = ObsNoiseConfig(
                    enabled=True,
                    Dr_sigma_mm=float(dr_cfg.Dr_sigma_mm),
                    theta_sigma=float(dr_cfg.theta_sigma),
                    ET0_sigma=float(dr_cfg.ET0_sigma),
                )
                env = ObsNoiseWrapper(env, cfg=obs_cfg, seed=seed)

    return env


def train_ppo(train_cfg_path: str) -> str:
    train_cfg = load_yaml(train_cfg_path)
    env_cfg = load_yaml(train_cfg["paths"]["env_config"])

    seed = int(train_cfg["seed"])
    out_dir = train_cfg["paths"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    def _make():
        return build_env(env_cfg, train_cfg, seed=seed)

    vec_env = DummyVecEnv([_make])

    ppo_cfg = train_cfg["ppo"]
    model = PPO(
        train_cfg["policy"],
        vec_env,
        learning_rate=float(ppo_cfg["learning_rate"]),
        n_steps=int(ppo_cfg["n_steps"]),
        batch_size=int(ppo_cfg["batch_size"]),
        n_epochs=int(ppo_cfg["n_epochs"]),
        gamma=float(ppo_cfg["gamma"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        clip_range=float(ppo_cfg["clip_range"]),
        ent_coef=float(ppo_cfg["ent_coef"]),
        verbose=1,
        seed=seed,
    )

    model.learn(total_timesteps=int(train_cfg["total_timesteps"]))

    save_path = os.path.join(out_dir, f"ppo_seed{seed}.zip")
    model.save(save_path)
    return save_path
