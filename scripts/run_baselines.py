from __future__ import annotations

import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import csv
import copy
import inspect
from typing import Dict, Any, List, Tuple

import numpy as np
import yaml

from irrigation_rl.train.ppo_train import load_yaml, build_env
from irrigation_rl.baselines.threshold import ThresholdPolicy
from irrigation_rl.baselines.fao_rule import FAORulePolicy


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def rollout_policy(env, policy_name: str, policy, out_dir: str) -> Dict[str, Any]:
    """
    用 baseline policy 在 env 上跑一条 episode
    自动保存：
      - trajectory.csv
      - metrics.json
    """
    ensure_dir(out_dir)

    obs, info = env.reset()
    done = False

    traj: Dict[str, List[Any]] = {
        "day": [],
        "Dr": [],
        "theta": [],
        "Dr_lo": [],
        "Dr_hi": [],
        "I": [],
        "reward": [],
        "err": [],
    }

    t = 0
    while not done:
        # Dr 从 info 里拿（RewardWrapper 会写入）
        Dr = float(info.get("Dr_mm", info.get("Dr", 0.0)))

        # baseline 给出动作
        I = float(policy.act(Dr))

        obs_next, reward, terminated, truncated, info_next = env.step(np.array([I], dtype=np.float32))
        done = bool(terminated or truncated)

        # 记录轨迹
        traj["day"].append(float(info_next.get("day", t)))
        traj["Dr"].append(float(info_next.get("Dr_mm", 0.0)))
        traj["theta"].append(float(info_next.get("theta", 0.0)))
        traj["Dr_lo"].append(float(info_next.get("Dr_lo", 0.0)))
        traj["Dr_hi"].append(float(info_next.get("Dr_hi", 0.0)))
        traj["I"].append(float(info_next.get("I_mm", I)))
        traj["reward"].append(float(reward))

        # err（如果 RewardWrapper 写了 reward_terms 就直接拿）
        terms = info_next.get("reward_terms", {})
        if isinstance(terms, dict) and "err" in terms:
            traj["err"].append(float(terms["err"]))
        else:
            # fallback：用区间距离算
            Dr_val = float(info_next.get("Dr_mm", 0.0))
            lo = float(info_next.get("Dr_lo", 0.0))
            hi = float(info_next.get("Dr_hi", 0.0))
            if Dr_val < lo:
                e = lo - Dr_val
            elif Dr_val > hi:
                e = Dr_val - hi
            else:
                e = 0.0
            traj["err"].append(float(e))

        obs = obs_next
        info = info_next
        t += 1

    # 保存 trajectory.csv
    traj_csv = os.path.join(out_dir, "trajectory.csv")
    with open(traj_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(traj.keys())
        for i in range(len(traj["day"])):
            w.writerow([traj[k][i] for k in traj.keys()])

    # 计算 metrics
    Dr = np.asarray(traj["Dr"], dtype=float)
    Dr_lo = np.asarray(traj["Dr_lo"], dtype=float)
    Dr_hi = np.asarray(traj["Dr_hi"], dtype=float)
    I = np.asarray(traj["I"], dtype=float)
    reward_arr = np.asarray(traj["reward"], dtype=float)

    err = np.zeros_like(Dr)
    err[Dr < Dr_lo] = Dr_lo[Dr < Dr_lo] - Dr[Dr < Dr_lo]
    err[Dr > Dr_hi] = Dr[Dr > Dr_hi] - Dr_hi[Dr > Dr_hi]

    metrics = {
        "method": policy_name,
        "N_steps": int(len(Dr)),
        "MAE_mm": float(np.mean(np.abs(err))),
        "RMSE_mm": float(np.sqrt(np.mean(err ** 2))),
        "WithinTargetRatio": float(np.mean((Dr >= Dr_lo) & (Dr <= Dr_hi))),
        "StressDays": int(np.sum(Dr > Dr_hi)),
        "UnderTargetDays": int(np.sum(Dr < Dr_lo)),
        "TotalIrrigation_mm": float(np.sum(I)),
        "AvgIrrigation_mm_per_day": float(np.mean(I)),
        "RewardMean": float(np.mean(reward_arr)),
        "RewardSum": float(np.sum(reward_arr)),
        "trajectory_csv": traj_csv,
        "out_dir": out_dir,
    }

    save_json(os.path.join(out_dir, "metrics.json"), metrics)
    return metrics


def build_env_compat(env_cfg: dict, train_cfg: dict, seed: int):
    """
    兼容你当前 build_env 的两种签名：
      build_env(env_cfg, seed)
      build_env(env_cfg, train_cfg, seed)
    """
    sig = inspect.signature(build_env)
    params = list(sig.parameters.keys())
    if len(params) >= 3:
        return build_env(env_cfg, train_cfg, seed=seed)
    return build_env(env_cfg, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml", help="train.yaml")
    ap.add_argument("--seeds", type=int, default=10, help="number of seeds")
    ap.add_argument("--out", type=str, default="outputs/baselines", help="baseline output dir")
    args = ap.parse_args()

    train_cfg = load_yaml(args.config)
    env_cfg = load_yaml(train_cfg["paths"]["env_config"])

    # 为了公平：baseline 的 reward 不需要 UCB bonus（避免 reward 虚高）
    eval_train_cfg = copy.deepcopy(train_cfg)
    if "ablation" in eval_train_cfg:
        eval_train_cfg["ablation"]["use_ucb_bonus"] = False
    if "reward" in eval_train_cfg:
        eval_train_cfg["reward"]["w_ucb"] = 0.0

    base_out = args.out
    ensure_dir(base_out)

    all_metrics: List[Dict[str, Any]] = []
    base_seed = int(train_cfg.get("seed", 42))

    for i in range(args.seeds):
        seed = base_seed + i

        # 构建环境（含 RewardWrapper）
        env = build_env_compat(env_cfg, eval_train_cfg, seed=seed)

        # ===== baseline 1：Threshold =====
        # 用 target 上界作为阈值：Dr > Dr_hi 就浇
        # irrigation_mm 取 a_max（更“保守”，能快速救旱）
        TAW = float(getattr(env, "TAW", 1.0))
        RAW = float(getattr(env, "RAW", 1.0))
        a_max = float(getattr(getattr(env, "cfg", None), "a_max_mm", 15.0))

        # 通过 TargetConfig 的比例估算 Dr_hi（与 wrapper 一致）
        low_frac_TAW = 0.20
        high_frac_RAW = 0.90
        Dr_hi = high_frac_RAW * RAW
        thr = ThresholdPolicy(Dr_threshold=Dr_hi, irrigation_mm=a_max)

        out_dir_thr = os.path.join(base_out, "Threshold", f"seed{seed}")
        m1 = rollout_policy(env, "Threshold", thr, out_dir_thr)
        m1["seed"] = seed
        all_metrics.append(m1)

        # ===== baseline 2：FAO Rule =====
        # Dr > RAW 就浇
        fao = FAORulePolicy(RAW=RAW, irrigation_mm=a_max)
        out_dir_fao = os.path.join(base_out, "FAORule", f"seed{seed}")
        m2 = rollout_policy(env, "FAORule", fao, out_dir_fao)
        m2["seed"] = seed
        all_metrics.append(m2)

    # 保存 baseline 总表（逐次实验明细）
    baseline_all_path = os.path.join(base_out, "_baseline_metrics_all.json")
    save_json(baseline_all_path, {"runs": all_metrics})
    print(f"[OK] Baselines done. Saved: {baseline_all_path}")


if __name__ == "__main__":
    main()
