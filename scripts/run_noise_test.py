from __future__ import annotations

import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import copy
import inspect
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO

from irrigation_rl.train.ppo_train import load_yaml, build_env
from irrigation_rl.train.evaluate import evaluate_policy
from irrigation_rl.metrics.metrics import compute_metrics_from_csv
from irrigation_rl.baselines.threshold import ThresholdPolicy
from irrigation_rl.baselines.fao_rule import FAORulePolicy
from irrigation_rl.robust.obs_noise_wrapper import ObsNoiseWrapper, ObsNoiseConfig


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


 


def build_env_compat(env_cfg: dict, train_cfg: dict, seed: int):
    sig = inspect.signature(build_env)
    if len(sig.parameters) >= 3:
        return build_env(env_cfg, train_cfg, seed=seed)
    return build_env(env_cfg, seed=seed)


def find_model_zip(seed_dir: str) -> str:
    zips = []
    for root, _, files in os.walk(seed_dir):
        for fn in files:
            if fn.endswith(".zip"):
                path = os.path.join(root, fn)
                zips.append(path)

    if not zips:
        raise FileNotFoundError(f"No .zip model found under: {seed_dir}")

    # ✅ 选最新修改的那个（mtime 最大）
    zips.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return zips[0]


def rollout_baseline_noise(env, policy_name: str, policy, out_dir: str) -> Dict[str, Any]:
    """
    baseline 在“噪声观测”环境上跑一条 episode
    baseline 用 obs[0]（噪声后的 Dr_obs）做决策，才能公平
    """
    ensure_dir(out_dir)
    obs, info = env.reset()
    done = False

    traj = {
        "day": [],
        "stage_norm": [],
        "Dr": [],
        "theta": [],
        # train interval (debug)
        "Dr_lo": [],
        "Dr_hi": [],
        "Dr_lo_train": [],
        "Dr_hi_train": [],
        # reference interval (paper metrics)
        "Dr_lo_ref": [],
        "Dr_hi_ref": [],
        "Dr_mid_ref": [],
        "in_ref": [],
        "e_mid_ref": [],
        # action/reward
        "I": [],
        "reward": [],
    }

    t = 0
    while not done:
        Dr_obs = float(obs[0])  # ← 使用噪声后的观测
        I = float(policy.act(Dr_obs))

        obs_next, reward, terminated, truncated, info_next = env.step(np.array([I], dtype=np.float32))
        done = bool(terminated or truncated)

        Dr_val = float(info_next.get("Dr_mm", 0.0))
        lo_train = float(info_next.get("Dr_lo_train", info_next.get("Dr_lo", 0.0)))
        hi_train = float(info_next.get("Dr_hi_train", info_next.get("Dr_hi", 0.0)))
        lo_ref = float(info_next.get("Dr_lo_ref", lo_train))
        hi_ref = float(info_next.get("Dr_hi_ref", hi_train))
        mid_ref = float(info_next.get("Dr_mid_ref", 0.5 * (lo_ref + hi_ref)))

        traj["day"].append(float(info_next.get("day", t)))
        traj["stage_norm"].append(float(info_next.get("stage_norm", 0.0)))
        traj["Dr"].append(Dr_val)
        traj["theta"].append(float(info_next.get("theta", 0.0)))

        traj["Dr_lo"].append(float(info_next.get("Dr_lo", lo_train)))
        traj["Dr_hi"].append(float(info_next.get("Dr_hi", hi_train)))
        traj["Dr_lo_train"].append(lo_train)
        traj["Dr_hi_train"].append(hi_train)

        traj["Dr_lo_ref"].append(lo_ref)
        traj["Dr_hi_ref"].append(hi_ref)
        traj["Dr_mid_ref"].append(mid_ref)
        traj["in_ref"].append(int((Dr_val >= lo_ref) and (Dr_val <= hi_ref)))
        traj["e_mid_ref"].append(float(abs(Dr_val - mid_ref)))

        traj["I"].append(float(info_next.get("I_mm", I)))
        traj["reward"].append(float(reward))

        obs = obs_next
        info = info_next
        t += 1

    # 保存 trajectory.csv
    df = pd.DataFrame(traj)
    traj_csv = os.path.join(out_dir, "trajectory.csv")
    df.to_csv(traj_csv, index=False, encoding="utf-8-sig")

    metrics = compute_metrics_from_csv(traj_csv)
    metrics.update({
        "method": policy_name,
        "scenario": "noise_test",
        "trajectory_csv": traj_csv,
        "out_dir": out_dir,
    })

    save_json(os.path.join(out_dir, "metrics.json"), metrics)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    ap.add_argument("--ppo_root", type=str, default="outputs/ppo_runs", help="你的 PPO 多seed目录")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", type=str, default="outputs/noise_test", help="噪声测试输出根目录")

    # 噪声强度：建议先从小到大做敏感性分析
    ap.add_argument("--Dr_sigma", type=float, default=0.5, help="Dr obs noise sigma (mm)")
    ap.add_argument("--theta_sigma", type=float, default=0.01, help="theta obs noise sigma")
    ap.add_argument("--ET0_sigma", type=float, default=0.2, help="ET0 obs noise sigma (mm/day)")
    args = ap.parse_args()

    base_cfg = load_yaml(args.config)
    env_cfg = load_yaml(base_cfg["paths"]["env_config"])
    base_seed = int(base_cfg.get("seed", 42))

    ensure_dir(args.out)

    noise_cfg = ObsNoiseConfig(
        enabled=True,
        Dr_sigma_mm=float(args.Dr_sigma),
        theta_sigma=float(args.theta_sigma),
        ET0_sigma=float(args.ET0_sigma),
    )

    all_metrics: List[Dict[str, Any]] = []

    # ====== 评估 PPO（噪声观测）======
    for i in range(args.seeds):
        seed = base_seed + i
        seed_dir = os.path.join(args.ppo_root, f"seed{seed}")
        if not os.path.isdir(seed_dir):
            print(f"[WARN] missing PPO seed dir: {seed_dir}")
            continue

        model_zip = None
        metrics_json = os.path.join(seed_dir, "metrics.json")
        if os.path.exists(metrics_json):
            with open(metrics_json, "r", encoding="utf-8") as f:
                j = json.load(f)
            mp = j.get("model_path")
            if isinstance(mp, str) and os.path.exists(mp):
                model_zip = mp

        # ===== 如果 metrics.json 没写 / 路径失效，就按命名规则找 =====
        if model_zip is None:
            candidate = os.path.join(args.ppo_root, f"ppo_seed{seed}.zip")
            if os.path.exists(candidate):
                model_zip = candidate

        # ===== 再不行就兜底递归搜索 =====
        if model_zip is None:
            model_zip = find_model_zip(seed_dir)

        # 评估时建议关 UCB，避免 reward 被 bonus 抬高（更公平）
        eval_cfg = copy.deepcopy(base_cfg)
        eval_cfg.setdefault("ablation", {})
        eval_cfg["ablation"]["use_ucb_bonus"] = False
        eval_cfg.setdefault("reward", {})
        eval_cfg["reward"]["w_ucb"] = 0.0

        env = build_env_compat(env_cfg, eval_cfg, seed=seed)
        env = ObsNoiseWrapper(env, noise_cfg, seed=seed)

        model = PPO.load(model_zip)

        out_dir = os.path.join(args.out, "PPO", f"seed{seed}")
        ensure_dir(out_dir)

        res = evaluate_policy(env, model, out_dir)
        traj_csv = res.get("trajectory_csv", os.path.join(out_dir, "trajectory.csv"))
        metrics = compute_metrics_from_csv(traj_csv)
        metrics.update({
            "method": "PPO",
            "seed": seed,
            "scenario": "noise_test",
            "model_path": model_zip,
            "trajectory_csv": traj_csv,
            "out_dir": out_dir,
            "noise": {
                "Dr_sigma_mm": noise_cfg.Dr_sigma_mm,
                "theta_sigma": noise_cfg.theta_sigma,
                "ET0_sigma": noise_cfg.ET0_sigma,
            }
        })

        save_json(os.path.join(out_dir, "metrics.json"), metrics)
        all_metrics.append(metrics)
        print(f"[PPO noise seed={seed}] OK -> {os.path.join(out_dir, 'metrics.json')}")

    # ====== 评估 Baselines（噪声观测）======
    # baseline 用同一个 noise env（公平）
    for i in range(args.seeds):
        seed = base_seed + i

        eval_cfg = copy.deepcopy(base_cfg)
        eval_cfg.setdefault("ablation", {})
        eval_cfg["ablation"]["use_ucb_bonus"] = False
        eval_cfg.setdefault("reward", {})
        eval_cfg["reward"]["w_ucb"] = 0.0

        env = build_env_compat(env_cfg, eval_cfg, seed=seed)
        env = ObsNoiseWrapper(env, noise_cfg, seed=seed)

        RAW = float(getattr(env, "RAW", 1.0))
        a_max = float(getattr(getattr(env, "cfg", None), "a_max_mm", 15.0))

        # Threshold：Dr > (0.9*RAW) 就灌溉
        thr = ThresholdPolicy(Dr_threshold=0.9 * RAW, irrigation_mm=a_max)
        out_thr = os.path.join(args.out, "Threshold", f"seed{seed}")
        m1 = rollout_baseline_noise(env, "Threshold", thr, out_thr)
        m1["seed"] = seed
        all_metrics.append(m1)
        print(f"[Threshold noise seed={seed}] OK")

        # FAO Rule：Dr > RAW 就灌溉
        env2 = build_env_compat(env_cfg, eval_cfg, seed=seed)  # 重新建一个 env，避免 episode 状态污染
        env2 = ObsNoiseWrapper(env2, noise_cfg, seed=seed)

        fao = FAORulePolicy(RAW=RAW, irrigation_mm=a_max)
        out_fao = os.path.join(args.out, "FAORule", f"seed{seed}")
        m2 = rollout_baseline_noise(env2, "FAORule", fao, out_fao)
        m2["seed"] = seed
        all_metrics.append(m2)
        print(f"[FAORule noise seed={seed}] OK")

    # 保存总表
    save_json(os.path.join(args.out, "_noise_metrics_all.json"), {"runs": all_metrics})
    print(f"\n[OK] Noise test done. Saved: {os.path.join(args.out, '_noise_metrics_all.json')}")


if __name__ == "__main__":
    main()
