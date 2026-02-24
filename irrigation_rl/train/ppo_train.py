from __future__ import annotations

import os
from typing import Any, Dict

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from irrigation_rl.envs.nursery_env import EnvConfig, NurseryIrrigationEnv
from irrigation_rl.envs.reward_wrapper import RewardWrapper, WrapperFlags
from irrigation_rl.envs.weather import (
    AssumptionWeatherConfig,
    AssumptionWeatherProvider,
    ExternalCSVWeatherProvider,
)
from irrigation_rl.rewards.reward import RewardConfig
from irrigation_rl.robust.domain_randomization_wrapper import DomainRandomizationWrapper
from irrigation_rl.robust.obs_noise_wrapper import ObsNoiseConfig, ObsNoiseWrapper
from irrigation_rl.robust.randomization import RandomizationConfig


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
        theta_sigma=float(sensor.get("theta_sigma", 0.005)),
        Kc_mult_min=float(params.get("Kc_mult_min", 0.95)),
        Kc_mult_max=float(params.get("Kc_mult_max", 1.05)),
        Zr_mult_min=float(params.get("Zr_mult_min", 0.95)),
        Zr_mult_max=float(params.get("Zr_mult_max", 1.05)),
    )


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
        fixed_lo_frac_TAW=float(ab.get("fixed_lo_frac_TAW", 0.15)),
        fixed_hi_frac_TAW=float(ab.get("fixed_hi_frac_TAW", 0.35)),
    )

    reward_cfg = RewardConfig()
    reward_block = train_cfg.get("reward", {})
    if isinstance(reward_block, dict):
        reward_cfg.w_track = float(reward_block.get("w_track", reward_cfg.w_track))
        reward_cfg.w_water = float(reward_block.get("w_water", reward_cfg.w_water))
        reward_cfg.w_improve = float(reward_block.get("w_improve", reward_cfg.w_improve))
        reward_cfg.w_smooth = float(reward_block.get("w_smooth", reward_cfg.w_smooth))
        reward_cfg.w_safe = float(reward_block.get("w_safe", reward_cfg.w_safe))
        reward_cfg.w_ucb = float(reward_block.get("w_ucb", reward_cfg.w_ucb))

    env = RewardWrapper(env, reward_cfg=reward_cfg, flags=flags)

    # Train-time domain randomization: enabled for Full, disabled for Vanilla.
    if bool(ab.get("use_robust_training", False)):
        dr_cfg = _load_randomization_cfg(train_cfg)
        if dr_cfg.enabled:
            env = DomainRandomizationWrapper(env, cfg=dr_cfg, seed=seed)
            if float(dr_cfg.theta_sigma) > 0.0:
                obs_cfg = ObsNoiseConfig(
                    enabled=True,
                    Dr_sigma_mm=0.0,
                    theta_sigma=float(dr_cfg.theta_sigma),
                    ET0_sigma=0.0,
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
