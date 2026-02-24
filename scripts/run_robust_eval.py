from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import sys
from typing import Any, Dict, List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO

from irrigation_rl.robust.et0_mult_wrapper import ET0MultConfig, ET0MultWrapper
from irrigation_rl.robust.obs_noise_wrapper import ObsNoiseConfig, ObsNoiseWrapper
from irrigation_rl.train.evaluate import evaluate_policy
from irrigation_rl.train.metrics import compute_metrics_from_csv
from irrigation_rl.train.ppo_train import build_env, load_yaml


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_env_compat(env_cfg: dict, train_cfg: dict, seed: int):
    sig = inspect.signature(build_env)
    if len(sig.parameters) >= 3:
        return build_env(env_cfg, train_cfg, seed=seed)
    return build_env(env_cfg, seed=seed)


def normalize_noise_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    raw = raw or {}
    sensor_noise = raw.get("sensor_noise", {}) if isinstance(raw, dict) else {}
    weather_bias = raw.get("weather_bias", {}) if isinstance(raw, dict) else {}

    dr_sigma = float(sensor_noise.get("Dr_sigma_mm", raw.get("Dr_sigma_mm", raw.get("Dr_sigma", 0.0))))
    theta_sigma = float(sensor_noise.get("theta_sigma", raw.get("theta_sigma", 0.0)))
    et0_sigma = float(sensor_noise.get("ET0_sigma", raw.get("ET0_sigma", 0.0)))

    et0_low = float(weather_bias.get("ET0_mult_min", raw.get("ET0_mult_min", 0.9)))
    et0_high = float(weather_bias.get("ET0_mult_max", raw.get("ET0_mult_max", 1.1)))
    et0_per_day = bool(raw.get("et0_per_day", True))

    return {
        "Dr_sigma_mm": dr_sigma,
        "theta_sigma": theta_sigma,
        "ET0_sigma": et0_sigma,
        "et0_mult": {
            "per_day": et0_per_day,
            "low": et0_low,
            "high": et0_high,
        },
    }


def canonical_setting(setting: str) -> str:
    s = str(setting).strip().lower()
    alias = {
        "nominal": "nominal",
        "none": "nominal",
        "noise": "noise_only",
        "noise-only": "noise_only",
        "noiseonly": "noise_only",
        "obs_noise": "noise_only",
        "et0": "et0_only",
        "et0-only": "et0_only",
        "et0only": "et0_only",
        "noise+et0": "noise_et0",
        "noise-et0": "noise_et0",
    }
    return alias.get(s, s)


def apply_robust_wrappers(env, setting: str, noise_cfg: Dict[str, Any], seed: int):
    setting = canonical_setting(setting)

    if setting in ("noise_only", "noise_et0"):
        obs_cfg = ObsNoiseConfig(
            enabled=True,
            Dr_sigma_mm=float(noise_cfg.get("Dr_sigma_mm", 0.0)),
            theta_sigma=float(noise_cfg.get("theta_sigma", 0.0)),
            ET0_sigma=float(noise_cfg.get("ET0_sigma", 0.0)),
        )
        env = ObsNoiseWrapper(env, cfg=obs_cfg, seed=seed)

    if setting in ("et0_only", "noise_et0"):
        et0_block = noise_cfg.get("et0_mult", {})
        et0_cfg = ET0MultConfig(
            enabled=True,
            per_day=bool(et0_block.get("per_day", True)),
            low=float(et0_block.get("low", 0.9)),
            high=float(et0_block.get("high", 1.1)),
        )
        env = ET0MultWrapper(env, cfg=et0_cfg, seed=seed)

    return env


def find_model_zip(ckpt_root: str, seed: int, method_name: str) -> str:
    candidates = [
        os.path.join(ckpt_root, f"seed{seed}", f"ppo_seed{seed}.zip"),
        os.path.join(ckpt_root, f"{method_name.lower()}_seed{seed}.zip"),
        os.path.join(ckpt_root, f"ppo_seed{seed}.zip"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    seed_dir = os.path.join(ckpt_root, f"seed{seed}")
    if os.path.isdir(seed_dir):
        zips: List[str] = []
        for root, _, files in os.walk(seed_dir):
            for fn in files:
                if fn.endswith(".zip"):
                    zips.append(os.path.join(root, fn))
        if zips:
            zips.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return zips[0]

    raise FileNotFoundError(f"cannot find checkpoint zip for seed={seed} under: {ckpt_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    ap.add_argument("--ckpt_root", type=str, required=True, help="checkpoint root, e.g. outputs/ppo_runs")
    ap.add_argument("--method", type=str, default="", help="override method name in output metadata")
    ap.add_argument("--seed_start", type=int, default=42)
    ap.add_argument("--num_seeds", type=int, default=10)
    ap.add_argument(
        "--settings",
        type=str,
        nargs="+",
        default=["nominal", "noise_only", "et0_only", "noise_et0"],
    )
    ap.add_argument("--noise_config", type=str, default="configs/noise_test.yaml")
    ap.add_argument("--out", type=str, default="outputs/robust_test_ref")
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--stochastic", action="store_true", help="use stochastic policy actions")
    args = ap.parse_args()

    base_cfg = load_yaml(args.config)
    env_cfg = load_yaml(base_cfg["paths"]["env_config"])
    method_name = str(args.method).strip() or str(base_cfg.get("method_name", "PPO"))

    noise_cfg_raw = load_yaml(args.noise_config) if os.path.exists(args.noise_config) else {}
    noise_cfg = normalize_noise_config(noise_cfg_raw)

    settings = [canonical_setting(s) for s in args.settings]
    valid = {"nominal", "noise_only", "et0_only", "noise_et0"}
    for s in settings:
        if s not in valid:
            raise ValueError(f"unsupported setting: {s}")

    ensure_dir(args.out)
    all_runs: List[Dict[str, Any]] = []
    max_steps = None if int(args.max_steps) <= 0 else int(args.max_steps)

    for setting in settings:
        for i in range(int(args.num_seeds)):
            seed = int(args.seed_start) + i

            eval_cfg = copy.deepcopy(base_cfg)
            eval_cfg["seed"] = seed
            eval_cfg.setdefault("ablation", {})
            eval_cfg["ablation"]["use_ucb_bonus"] = False
            eval_cfg["ablation"]["use_robust_training"] = False
            eval_cfg.setdefault("reward", {})
            eval_cfg["reward"]["w_ucb"] = 0.0

            env = build_env_compat(env_cfg, eval_cfg, seed=seed)
            env = apply_robust_wrappers(env, setting=setting, noise_cfg=noise_cfg, seed=seed)

            model_zip = find_model_zip(args.ckpt_root, seed=seed, method_name=method_name)
            model = PPO.load(model_zip)

            run_out = os.path.join(args.out, method_name, setting, f"seed{seed}")
            ensure_dir(run_out)

            res = evaluate_policy(
                env=env,
                model=model,
                out_dir=run_out,
                deterministic=not bool(args.stochastic),
                max_steps=max_steps,
                use_ucb=False,
            )
            traj_csv = res.get("trajectory_csv", os.path.join(run_out, "trajectory.csv"))
            metrics = compute_metrics_from_csv(traj_csv)

            run_info = {
                "method": method_name,
                "seed": seed,
                "scenario": "robustness",
                "setting": setting,
                "model_path": model_zip,
                "trajectory_csv": traj_csv,
                "out_dir": run_out,
                "noise": noise_cfg,
                **metrics,
            }
            save_json(os.path.join(run_out, "metrics.json"), run_info)
            all_runs.append(run_info)

            print(f"[OK] method={method_name} setting={setting} seed={seed}")

    summary_path = os.path.join(args.out, method_name, "_robust_metrics_all.json")
    save_json(summary_path, {"method": method_name, "settings": settings, "runs": all_runs})
    print(f"[OK] Robust evaluation finished. Saved: {summary_path}")


if __name__ == "__main__":
    main()
