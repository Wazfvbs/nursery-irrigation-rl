from __future__ import annotations

import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import copy
import json
import inspect
from typing import Dict, Any, List

import yaml
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from irrigation_rl.train.ppo_train import load_yaml, train_ppo, build_env
from irrigation_rl.train.evaluate import evaluate_policy


# ========== utils ==========
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

    return {
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


def build_env_compat(env_cfg: dict, train_cfg: dict, seed: int):
    """
    兼容 build_env 两种签名：
      build_env(env_cfg, seed)
      build_env(env_cfg, train_cfg, seed)
    """
    sig = inspect.signature(build_env)
    if len(sig.parameters) >= 3:
        return build_env(env_cfg, train_cfg, seed=seed)
    return build_env(env_cfg, seed=seed)


# ========== ablation settings ==========
def ablation_cases() -> Dict[str, Dict[str, bool]]:
    """
    4 组最小消融组合（论文 Table10 常规写法）
    """
    return {
        "Full": {
            "use_dynamic_target": True,
            "use_reward_shaping": True,
            "use_ucb_bonus": True,
        },
        "wo_UCB": {
            "use_dynamic_target": True,
            "use_reward_shaping": True,
            "use_ucb_bonus": False,
        },
        "wo_Shaping": {
            "use_dynamic_target": True,
            "use_reward_shaping": False,
            "use_ucb_bonus": True,
        },
        "wo_Target": {
            "use_dynamic_target": False,
            "use_reward_shaping": True,
            "use_ucb_bonus": True,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml", help="base train.yaml")
    ap.add_argument("--seeds", type=int, default=10, help="how many seeds")
    ap.add_argument("--out", type=str, default="outputs/ablation", help="output root")
    args = ap.parse_args()

    base_cfg = load_yaml(args.config)
    env_cfg = load_yaml(base_cfg["paths"]["env_config"])
    base_seed = int(base_cfg.get("seed", 42))

    ensure_dir(args.out)

    all_runs: List[Dict[str, Any]] = []

    cases = ablation_cases()

    for case_name, ab_flags in cases.items():
        case_dir = os.path.join(args.out, case_name)
        ensure_dir(case_dir)

        print(f"\n========== [CASE] {case_name} ==========")

        for i in range(args.seeds):
            seed = base_seed + i

            seed_dir = os.path.join(case_dir, f"seed{seed}")
            ensure_dir(seed_dir)

            # 1) 生成本次 case 的 train_cfg
            train_cfg = copy.deepcopy(base_cfg)
            train_cfg["seed"] = seed
            train_cfg.setdefault("ablation", {})
            train_cfg["ablation"].update(ab_flags)

            # 训练阶段保持 UCB（由 ablation 控制），评估阶段建议关 UCB（公平且稳定）
            eval_cfg = copy.deepcopy(train_cfg)
            eval_cfg.setdefault("ablation", {})
            eval_cfg["ablation"]["use_ucb_bonus"] = False
            eval_cfg.setdefault("reward", {})
            eval_cfg["reward"]["w_ucb"] = 0.0

            # 2) 写入临时 yaml（保证可复现）
            cfg_path = os.path.join(seed_dir, "train_case.yaml")
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(train_cfg, f, sort_keys=False, allow_unicode=True)

            # 3) 训练
            model_path = train_ppo(cfg_path)

            # 4) 评估并导出 trajectory.csv
            env = build_env_compat(env_cfg, eval_cfg, seed=seed)
            model = PPO.load(model_path)

            eval_out = os.path.join(seed_dir, "eval")
            res = evaluate_policy(env, model, eval_out)

            traj_csv = res.get("trajectory_csv", os.path.join(eval_out, "trajectory.csv"))
            metrics = compute_metrics_from_csv(traj_csv)

            run_info = {
                "method": f"Ablation::{case_name}",
                "case": case_name,
                "seed": seed,
                "model_path": model_path,
                "eval_out": eval_out,
                "trajectory_csv": traj_csv,
                **metrics,
            }

            save_json(os.path.join(seed_dir, "metrics.json"), run_info)
            all_runs.append(run_info)

            print(f"[{case_name} seed={seed}] saved metrics.json")

    # 保存全集
    save_json(os.path.join(args.out, "_ablation_metrics_all.json"), {"runs": all_runs})
    print(f"\n[OK] Ablation done. Saved: {os.path.join(args.out, '_ablation_metrics_all.json')}")


if __name__ == "__main__":
    main()
