from __future__ import annotations

import argparse
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import copy
import inspect
from typing import Dict, Any, List

import yaml

from stable_baselines3 import PPO

from irrigation_rl.train.ppo_train import load_yaml, train_ppo, build_env
from irrigation_rl.train.evaluate import evaluate_policy
from irrigation_rl.train.metrics import compute_metrics_from_csv


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


 


def build_env_compat(env_cfg: dict, train_cfg: dict, seed: int):
    sig = inspect.signature(build_env)
    params = list(sig.parameters.keys())
    if len(params) >= 3:
        return build_env(env_cfg, train_cfg, seed=seed)
    return build_env(env_cfg, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    ap.add_argument("--seeds", type=int, default=10, help="legacy alias for --num_seeds")
    ap.add_argument("--seed_start", type=int, default=None, help="start seed (default: config.seed)")
    ap.add_argument("--num_seeds", type=int, default=None, help="number of seeds (default: --seeds)")
    ap.add_argument("--out", type=str, default="", help="root output for multi-seed runs")
    ap.add_argument("--eval_steps", type=int, default=0, help="not used (kept for compatibility)")
    args = ap.parse_args()

    base_train_cfg = load_yaml(args.config)
    env_cfg = load_yaml(base_train_cfg["paths"]["env_config"])

    root_out = str(args.out).strip() or str(base_train_cfg["paths"]["out_dir"])
    ensure_dir(root_out)

    all_runs: List[Dict[str, Any]] = []
    base_seed = int(base_train_cfg.get("seed", 42))
    seed_start = int(args.seed_start) if args.seed_start is not None else base_seed
    num_seeds = int(args.num_seeds) if args.num_seeds is not None else int(args.seeds)
    method_name = str(base_train_cfg.get("method_name", "PPO"))

    for i in range(num_seeds):
        seed = seed_start + i

        # 1) 复制一份 train_cfg 写入临时 yaml（保证每个 seed 可复现）
        train_cfg = copy.deepcopy(base_train_cfg)
        train_cfg["seed"] = seed

        seed_dir = os.path.join(root_out, f"seed{seed}")
        ensure_dir(seed_dir)
        train_cfg.setdefault("paths", {})
        train_cfg["paths"]["out_dir"] = seed_dir

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
        eval_cfg["ablation"]["use_robust_training"] = False

        eval_cfg.setdefault("reward", {})
        eval_cfg["reward"]["w_ucb"] = 0.0

        env = build_env_compat(env_cfg, eval_cfg, seed=seed)

        model = PPO.load(model_path)
        eval_out = os.path.join(seed_dir, "eval")
        res = evaluate_policy(env, model, eval_out, use_ucb=False)


        traj_csv = res.get("trajectory_csv", os.path.join(eval_out, "trajectory.csv"))
        metrics = compute_metrics_from_csv(traj_csv)

        run_info = {
            "method": method_name,
            "seed": seed,
            "scenario": "nominal",
            "setting": "nominal",
            "model_path": model_path,
            "eval_out": eval_out,
            "trajectory_csv": traj_csv,
            **metrics,
        }

        save_json(os.path.join(seed_dir, "metrics.json"), run_info)
        all_runs.append(run_info)

        print(f"[SEED {seed}] model={model_path} metrics_saved={os.path.join(seed_dir, 'metrics.json')}")

    summary_name = f"_{method_name}_metrics_all.json"
    save_json(os.path.join(root_out, summary_name), {"runs": all_runs, "method": method_name})
    print(f"[OK] {method_name} seeds done. Saved: {os.path.join(root_out, summary_name)}")


if __name__ == "__main__":
    main()
