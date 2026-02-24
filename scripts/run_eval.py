from __future__ import annotations

import argparse
import copy
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO

from irrigation_rl.train.evaluate import evaluate_policy
from irrigation_rl.train.ppo_train import build_env, load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--out", type=str, default="outputs/eval_run")
    ap.add_argument("--stochastic", action="store_true", help="use stochastic policy actions")
    ap.add_argument("--max_steps", type=int, default=0)
    args = ap.parse_args()

    train_cfg = load_yaml(args.config)
    env_cfg = load_yaml(train_cfg["paths"]["env_config"])
    seed = int(train_cfg.get("seed", 0))

    eval_cfg = copy.deepcopy(train_cfg)
    eval_cfg.setdefault("ablation", {})
    eval_cfg["ablation"]["use_ucb_bonus"] = False
    eval_cfg["ablation"]["use_robust_training"] = False
    eval_cfg.setdefault("reward", {})
    eval_cfg["reward"]["w_ucb"] = 0.0
    env = build_env(env_cfg, eval_cfg, seed=seed)

    model_path = args.model.strip() or os.path.join(train_cfg["paths"]["out_dir"], f"ppo_seed{seed}.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")

    model = PPO.load(model_path)
    max_steps = None if args.max_steps <= 0 else int(args.max_steps)

    res = evaluate_policy(
        env=env,
        model=model,
        out_dir=args.out,
        deterministic=not bool(args.stochastic),
        max_steps=max_steps,
        use_ucb=False,
    )
    print("[OK] eval finished")
    print(f"  model: {model_path}")
    print(f"  out_dir: {res['out_dir']}")
    print(f"  trajectory: {res['trajectory_csv']}")
    print(f"  steps: {res['steps']}")


if __name__ == "__main__":
    main()
