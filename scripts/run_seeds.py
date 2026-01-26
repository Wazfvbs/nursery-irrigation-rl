from __future__ import annotations

import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import copy
import inspect
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import yaml

from stable_baselines3 import PPO

from irrigation_rl.train.ppo_train import load_yaml, train_ppo, build_env
from irrigation_rl.train.evaluate import evaluate_policy


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def compute_metrics_from_csv(csv_path: str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)

    required = ["Dr", "Dr_lo", "Dr_hi", "I", "reward"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"error": "missing_cols", "missing": missing, "trajectory_csv": csv_path}

    Dr = df["Dr"].to_numpy(dtype=float)
    Dr_lo = df["Dr_lo"].to_numpy(dtype=float)
    Dr_hi = df["Dr_hi"].to_numpy(dtype=float)
    I = df["I"].to_numpy(dtype=float)
    R = df["reward"].to_numpy(dtype=float)
    # ✅ 控制平滑性指标（体现 shaping 的作用）
    action_std = float(np.std(I))
    action_tv = float(np.sum(np.abs(I[1:] - I[:-1]))) if len(I) > 1 else 0.0

    err = np.zeros_like(Dr)
    err[Dr < Dr_lo] = Dr_lo[Dr < Dr_lo] - Dr[Dr < Dr_lo]
    err[Dr > Dr_hi] = Dr[Dr > Dr_hi] - Dr_hi[Dr > Dr_hi]

    metrics = {
        "N_steps": int(len(df)),
        "MAE_mm": float(np.mean(np.abs(err))),
        "RMSE_mm": float(np.sqrt(np.mean(err ** 2))),
        "WithinTargetRatio": float(np.mean((Dr >= Dr_lo) & (Dr <= Dr_hi))),
        "StressDays": int(np.sum(Dr > Dr_hi)),
        "UnderTargetDays": int(np.sum(Dr < Dr_lo)),
        "TotalIrrigation_mm": float(np.sum(I)),
        "AvgIrrigation_mm_per_day": float(np.mean(I)),
        "RewardMean": float(np.mean(R)),
        "RewardSum": float(np.sum(R)),
        "ActionStd": action_std,
        "ActionTV": action_tv,

    }
    return metrics


def build_env_compat(env_cfg: dict, train_cfg: dict, seed: int):
    sig = inspect.signature(build_env)
    params = list(sig.parameters.keys())
    if len(params) >= 3:
        return build_env(env_cfg, train_cfg, seed=seed)
    return build_env(env_cfg, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", type=str, default="outputs/ppo_runs", help="root output for ppo multi-seed")
    ap.add_argument("--eval_steps", type=int, default=0, help="not used (kept for compatibility)")
    args = ap.parse_args()

    base_train_cfg = load_yaml(args.config)
    env_cfg = load_yaml(base_train_cfg["paths"]["env_config"])

    root_out = args.out
    ensure_dir(root_out)

    all_runs: List[Dict[str, Any]] = []
    base_seed = int(base_train_cfg.get("seed", 42))

    for i in range(args.seeds):
        seed = base_seed + i

        # 1) 复制一份 train_cfg 写入临时 yaml（保证每个 seed 可复现）
        train_cfg = copy.deepcopy(base_train_cfg)
        train_cfg["seed"] = seed

        seed_dir = os.path.join(root_out, f"seed{seed}")
        ensure_dir(seed_dir)

        tmp_cfg_path = os.path.join(seed_dir, "train_seed.yaml")
        with open(tmp_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(train_cfg, f, sort_keys=False, allow_unicode=True)

        # 2) 训练
        model_path = train_ppo(tmp_cfg_path)

        # 3) 评估（导出 trajectory.csv）
        # 3) 评估：构建一个 “eval专用配置”，关掉 UCB（避免 reward 虚高）
        eval_cfg = copy.deepcopy(train_cfg)
        eval_cfg.setdefault("ablation", {})
        eval_cfg["ablation"]["use_ucb_bonus"] = False

        eval_cfg.setdefault("reward", {})
        eval_cfg["reward"]["w_ucb"] = 0.0

        env = build_env_compat(env_cfg, eval_cfg, seed=seed)

        model = PPO.load(model_path)
        eval_out = os.path.join(seed_dir, "eval")
        res = evaluate_policy(env, model, eval_out)   # ✅ 不传 use_ucb


        traj_csv = res.get("trajectory_csv", os.path.join(eval_out, "trajectory.csv"))
        metrics = compute_metrics_from_csv(traj_csv)

        run_info = {
            "method": "PPO",
            "seed": seed,
            "model_path": model_path,
            "eval_out": eval_out,
            "trajectory_csv": traj_csv,
            **metrics,
        }

        save_json(os.path.join(seed_dir, "metrics.json"), run_info)
        all_runs.append(run_info)

        print(f"[SEED {seed}] model={model_path} metrics_saved={os.path.join(seed_dir, 'metrics.json')}")

    save_json(os.path.join(root_out, "_ppo_metrics_all.json"), {"runs": all_runs})
    print(f"[OK] PPO seeds done. Saved: {os.path.join(root_out, '_ppo_metrics_all.json')}")


if __name__ == "__main__":
    main()
